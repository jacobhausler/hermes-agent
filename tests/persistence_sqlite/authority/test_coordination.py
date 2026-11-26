"""The delegation ledger's writes re-race a busy in-process writer, a bounded number of times.

A queue of writers longer than the ledgers' lock wait starves a ledger verb's first acquire of the
store's writer mutex (``WriterMutexTimeout`` is raised before ``BEGIN IMMEDIATE``: the callback never
ran, so re-racing is exactly-once safe). Like ``_ledger_setup``, the verb re-races through the store's
own jitter up to ``_LEDGER_ATTEMPTS`` times, each attempt with a fresh patience. Under the 2026-11-26
dogfood load the async-completions sweep got one attempt and one patience, the transcript-commit
queue drained slower, and ``StoreContention`` tracebacks landed on every sweep tick.
"""

from __future__ import annotations

import sqlite3
import threading
import time

import pytest

from persistence import StoreContention
from persistence.authority.coordination import (
    DELEGATION_CLAIM_STALE_SECONDS, DELEGATION_ORPHAN_STALE_SECONDS, DELEGATION_REPLAY_MAX_AGE_SECONDS,
    OwnerStamp, RecordedOwner,
)
from persistence.sqlite.authority import coordination as coordination_module
from tests.contracts.authority.test_coordination import dead_pid
from tests.persistence_sqlite.authority.conftest import NOW
from tests.persistence_sqlite.conftest import store_for

pytestmark = pytest.mark.usefixtures("clock")

PATIENCE = 0.6  # the ledgers' wait, shortened: the tests exercise the bound, not the length
WEDGE_S = PATIENCE + 0.3  # longer than one patience, far shorter than what the retry budget allows
DEAD_PID = dead_pid()
DEAD = RecordedOwner(pid=DEAD_PID, started_at=11)
# Past the orphan staleness and the replay age cap: the sweep's one DB effect is the row going 'dropped'.
SWEPT_NOW = NOW + max(DELEGATION_ORPHAN_STALE_SECONDS, DELEGATION_CLAIM_STALE_SECONDS,
                      DELEGATION_REPLAY_MAX_AGE_SECONDS) + 60


def _seed_orphan(store) -> None:
    store.ready()
    store.coordination.record_delegation_dispatch(
        delegation_id="wedged", origin_session="agent:main:cli", origin_ui_session_id="ui-1",
        parent_session_id=None, dispatched_at=NOW, owner=OwnerStamp(pid=DEAD_PID, started_at=11),
        task={"goal": "g", "is_batch": False})
    store.coordination.record_delegation_completion(
        delegation_id="wedged", state="completed", completed_at=NOW,
        event={"type": "async_delegation", "delegation_id": "wedged", "status": "completed",
               "summary": "done", "session_key": "agent:main:cli"},
        result={"status": "completed", "summary": "done"})


def _delivery_state(scope_a) -> str:
    with sqlite3.connect(store_for(scope_a)) as conn:
        return conn.execute("SELECT delivery_state FROM async_delegations WHERE delegation_id='wedged'").fetchone()[0]


def _wedged_writer(execute, release: threading.Event) -> None:
    """A thread holding the store's writer mutex through the real write path until ``release`` is set."""
    entered = threading.Event()

    def wedge():
        def hold(conn):
            entered.set()
            release.wait(30)
        execute(hold)

    threading.Thread(target=wedge, daemon=True).start()
    assert entered.wait(5)


def test_a_wedge_past_one_patience_re_races_and_the_write_lands(backend, scope_a, monkeypatch):
    """The sweep that the incident refused: the wedge outlives one patience, not the retry."""
    monkeypatch.setattr(coordination_module, "LEDGER_PATIENCE_S", PATIENCE)
    store = backend.open(scope_a)
    _seed_orphan(store)
    release = threading.Event()
    _wedged_writer(store.holder()._execute_write, release)
    threading.Timer(WEDGE_S, release.set).start()
    assert store.coordination.sweep_orphaned_completions(now=SWEPT_NOW, dead_owners=frozenset({DEAD})) == ()
    assert _delivery_state(scope_a) == "dropped"  # the sweep's effect landed, not just its error suppressed


def test_a_still_wedged_writer_raises_after_three_fresh_patiences(backend, scope_a, monkeypatch):
    """The retry is bounded: a writer that never yields still ends in contention, never a hang."""
    monkeypatch.setattr(coordination_module, "LEDGER_PATIENCE_S", PATIENCE)
    store = backend.open(scope_a)
    _seed_orphan(store)
    holder = store.holder()
    attempts: list = []
    real_execute_write = holder._execute_write

    def counting(fn, patience_s=None):
        attempts.append(patience_s)
        return real_execute_write(fn, patience_s=patience_s)

    monkeypatch.setattr(holder, "_execute_write", counting)
    release = threading.Event()
    _wedged_writer(real_execute_write, release)  # the wedge's own write bypasses the attempt counter
    started = time.monotonic()
    with pytest.raises(StoreContention):
        store.coordination.sweep_orphaned_completions(now=SWEPT_NOW, dead_owners=frozenset({DEAD}))
    took = time.monotonic() - started
    release.set()
    assert attempts == [PATIENCE] * coordination_module._LEDGER_ATTEMPTS
    # Every attempt got its own patience (the raise comes after all three waits) and the total stays bounded.
    assert 3 * PATIENCE * 0.9 <= took < 3 * (PATIENCE + 0.2)
    assert _delivery_state(scope_a) == "pending"  # the refused sweep touched nothing

"""est-2ek.1.885: the turn-lease wait is configurable and never parks behind an
unprovable holder.

A desktop prompt parked 1800 s in ``acquire_session_turn_lease`` behind a CLI holder
in a foreign PID namespace — a wait that can only end at the row's TTL, because the
reclaim probe defers to expiry for anything this process cannot prove dead. Three
invariants:

* the wait comes from ``agent.turn_lease.wait_seconds`` (the bare constant is gone);
* a foreign/unstamped holder clamps the wait to the TTL floor and the rejection
  names the holder's pid/pidns/platform and since-time;
* a provable holder keeps the full configured wait — its reclaim may never come.
"""

from __future__ import annotations

import time

import pytest

import hermes_state_pidns
from agent.turn_facade_lease import (
    LEASE_TTL_SECONDS,
    LEASE_WAIT_SECONDS,
    _FOREIGN_NS_WAIT_FLOOR_S,
    admit_durable_turn_lease,
    resolve_lease_wait_seconds,
)
from hermes_state_pidns import LocalPidNamespace
from run_agent import AIAgent

_OURS, _SIBLING = "111", "222"


@pytest.fixture(autouse=True)
def _pinned_namespace(monkeypatch):
    # Pin so "foreign" means foreign on every host, including platforms with no
    # PID namespaces (where everything is checkable).
    monkeypatch.setattr(hermes_state_pidns, "_LOCAL_PID_NS", LocalPidNamespace(_OURS, True))


def _foreign_holder() -> str:
    return f"pid=999:pidns={_SIBLING}:turn=cli-turn:platform=cli"


def _ours_holder() -> str:
    return f"pid=999:pidns={_OURS}:turn=cli-turn:platform=cli"


class _WaitDB:
    """Records the admission kwargs; hands back whatever row/verdict the test staged."""

    def __init__(self, row=None, acquire_result: bool = False):
        self.row = row
        self.acquire_result = acquire_result
        self.kwargs: dict | None = None

    def get_session(self, session_id):
        return {"id": session_id}

    def get_session_turn_lease(self, session_id):
        return self.row

    def acquire_session_turn_lease(self, session_id, holder, **kwargs):
        self.kwargs = kwargs
        return self.acquire_result

    def resolve_resume_session_id(self, session_id):
        return session_id

    def get_messages_as_conversation(self, session_id, **kwargs):
        return []

    def refresh_session_turn_lease(self, session_id, holder, **kwargs):
        return True

    def release_session_turn_lease(self, session_id, holder):
        pass


def _agent(db, *, interrupted: bool = False):
    agent = AIAgent.__new__(AIAgent)
    agent.session_id = "sess-885"
    agent.platform = "desktop"
    agent.model = "test-model"
    agent._session_db = db
    agent._session_db_created = True
    agent._persist_disabled = False
    agent._parent_session_id = None
    agent._relay_pending_turn_id = None
    agent._reset_activity_labels_after_turn = lambda: None
    agent._conversation_root_id = lambda: "sess-885"
    agent.log_prefix = ""
    agent._vprint = lambda *a, **k: None
    agent.status_callback = None
    agent._interrupt_requested = interrupted
    agent._interrupt_message = None
    agent._pending_redirect = None
    agent._execution_thread_id = None
    agent._interrupt_thread_signal_pending = False
    agent._emit_status = lambda *a, **k: None
    agent._emit_warning = lambda *a, **k: None
    agent._active_session_turn_lease_holder = None
    agent._active_session_turn_lease_ttl_seconds = None
    agent._liveness_activity_lock = lambda: __import__("threading").Lock()
    return agent


def _admit(agent, db):
    return admit_durable_turn_lease(
        agent, session_id="sess-885", relay_turn_id="turn-1",
        task_context={"platform": "desktop", "session_id": "sess-885"},
        conversation_history=[{"role": "user", "content": "hi"}],
    )


# ---------------------------------------------------------------- config knob


def test_wait_defaults_and_valid_override():
    assert resolve_lease_wait_seconds(None) == LEASE_WAIT_SECONDS
    assert resolve_lease_wait_seconds({}) == LEASE_WAIT_SECONDS
    assert resolve_lease_wait_seconds({"agent": {"turn_lease": {"wait_seconds": 45}}}) == 45.0
    assert resolve_lease_wait_seconds({"agent": {"turn_lease": {"wait_seconds": "120"}}}) == 120.0


@pytest.mark.parametrize("bad", ["abc", float("nan"), float("inf"), None, {"nested": 1}])
def test_invalid_wait_falls_back_and_never_raises(bad):
    # None lands on the section default (also the bare constant), not a crash.
    expected = LEASE_WAIT_SECONDS
    assert resolve_lease_wait_seconds({"agent": {"turn_lease": {"wait_seconds": bad}}}) == expected


def test_admission_uses_configured_wait(monkeypatch):
    import hermes_cli.config as hconfig

    monkeypatch.setattr(
        hconfig, "load_config_readonly",
        lambda: {"agent": {"turn_lease": {"wait_seconds": 45}}},
    )
    db = _WaitDB(row=None, acquire_result=True)
    admission = _admit(_agent(db), db)
    assert admission.lease is not None and admission.early_result is None
    assert db.kwargs["wait_seconds"] == 45.0


# ------------------------------------------------------------ fail-fast class


def test_foreign_pidns_holder_clamps_wait_and_names_the_holder():
    row = (_foreign_holder(), time.time() - 60.0, time.time() + 240.0)
    db = _WaitDB(row=row, acquire_result=False)
    admission = _admit(_agent(db), db)

    assert admission.lease is None
    assert db.kwargs["wait_seconds"] == _FOREIGN_NS_WAIT_FLOOR_S
    assert _FOREIGN_NS_WAIT_FLOOR_S > LEASE_TTL_SECONDS  # never before the sweeper can run
    text = admission.early_result["final_response"]
    for fact in ("pid=999", "pidns=222", "platform=cli", "held since", "failing fast"):
        assert fact in text, fact
    assert admission.early_result["failure_reason"] == "session_busy"
    assert admission.early_result["failure_retryable"] is True


def test_unstamped_holder_fails_fast_as_unprovable():
    legacy = "pid=999:turn=old:platform=cli"  # pre-upgrade row: no pidns stamp
    db = _WaitDB(row=(legacy, time.time() - 60.0, time.time() + 240.0), acquire_result=False)
    admission = _admit(_agent(db), db)

    assert db.kwargs["wait_seconds"] == _FOREIGN_NS_WAIT_FLOOR_S
    assert "pidns=unstamped(pre-upgrade)" in admission.early_result["final_response"]


def test_same_namespace_holder_keeps_full_wait():
    row = (_ours_holder(), time.time() - 60.0, time.time() + 240.0)
    db = _WaitDB(row=row, acquire_result=False)
    admission = _admit(_agent(db), db)

    # The holder is probe-able: it may genuinely still be running; only the
    # configured wait (default 1800) may end the wait. Facts still surface.
    assert db.kwargs["wait_seconds"] == LEASE_WAIT_SECONDS
    text = admission.early_result["final_response"]
    assert "pid=999" in text and "held since" in text
    assert "failing fast" not in text


def test_interrupted_wait_carries_holder_facts():
    row = (_foreign_holder(), time.time() - 60.0, time.time() + 240.0)
    db = _WaitDB(row=row, acquire_result=False)
    admission = _admit(_agent(db, interrupted=True), db)

    assert admission.early_result["interrupted"] is True
    text = admission.early_result["final_response"]
    assert "pid=999" in text and "platform=cli" in text and "held since" in text


def test_no_diagnostics_read_never_blocks_admission():
    """A diagnostic row read that raises is swallowed: admission behaves as before the read."""

    class _RaisingDB(_WaitDB):
        def get_session_turn_lease(self, session_id):
            raise RuntimeError("state.db is having a day")

    db = _RaisingDB(acquire_result=True)
    admission = _admit(_agent(db), db)
    assert admission.lease is not None
    assert db.kwargs["wait_seconds"] == LEASE_WAIT_SECONDS


def test_db_without_diagnostic_method_still_admits():
    """Shim DBs (MagicMock-style, old fakes) may lack get_session_turn_lease entirely."""

    class _LegacyDB:
        def __init__(self):
            self.kwargs = None

        def acquire_session_turn_lease(self, session_id, holder, **kwargs):
            self.kwargs = kwargs
            return True

        def resolve_resume_session_id(self, session_id):
            return session_id

        def get_session(self, session_id):
            return {"id": session_id}

        def get_messages_as_conversation(self, session_id, **kwargs):
            return []

        def refresh_session_turn_lease(self, session_id, holder, **kwargs):
            return True

        def release_session_turn_lease(self, session_id, holder):
            pass

    db = _LegacyDB()
    admission = _admit(_agent(db), db)
    assert admission.lease is not None
    assert db.kwargs["wait_seconds"] == LEASE_WAIT_SECONDS

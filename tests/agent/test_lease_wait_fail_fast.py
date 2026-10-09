"""The turn-lease admission wait is configurable, and a lease-wait timeout names its holder.

The admission wait was a bare ``LEASE_WAIT_SECONDS`` (1800 s) constant: an operator
under a wedged sibling process had no way to shorten it. And when the wait finally
failed, the one error log named nobody — no pid, no pidns, no platform, no since-time.

Four invariants:

* the wait comes from ``agent.turn_lease.wait_seconds`` (validated like agent.turn_liveness:
  typo/NaN/Inf/bool/non-positive warn and fall back to the constant; never raises);
* the configured value reaches ``acquire_session_turn_lease`` through the real loader;
* the user-visible timeout and interrupt texts, ``failure_reason``, ``error`` and
  ``failure_retryable`` stay byte-identical to the pre-config behaviour;
* on timeout, the error log names the holder (pid/pidns/platform/since) from a
  diagnostic read taken AFTER the wait — pure diagnostics, never a decision input.
"""

from __future__ import annotations

import logging
import time

import pytest

import hermes_state_pidns
from agent.turn_facade_lease import (
    LEASE_WAIT_SECONDS,
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
    agent.session_id = "sess-lease"
    agent.platform = "desktop"
    agent.model = "test-model"
    agent._session_db = db
    agent._session_db_created = True
    agent._persist_disabled = False
    agent._parent_session_id = None
    agent._relay_pending_turn_id = None
    agent._reset_activity_labels_after_turn = lambda: None
    agent._conversation_root_id = lambda: "sess-lease"
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
        agent, session_id="sess-lease", relay_turn_id="turn-1",
        task_context={"platform": "desktop", "session_id": "sess-lease"},
        conversation_history=[{"role": "user", "content": "hi"}],
    )


# ---------------------------------------------------------------- config knob


def test_wait_defaults_and_valid_override():
    assert resolve_lease_wait_seconds(None) == LEASE_WAIT_SECONDS
    assert resolve_lease_wait_seconds({}) == LEASE_WAIT_SECONDS
    assert resolve_lease_wait_seconds({"agent": {"turn_lease": {"wait_seconds": 45}}}) == 45.0
    assert resolve_lease_wait_seconds({"agent": {"turn_lease": {"wait_seconds": "120"}}}) == 120.0


@pytest.mark.parametrize("bad", ["abc", float("nan"), float("inf"), True, 0, -5])
def test_invalid_wait_falls_back_and_never_raises(bad, caplog):
    with caplog.at_level(logging.WARNING):
        assert resolve_lease_wait_seconds({"agent": {"turn_lease": {"wait_seconds": bad}}}) == LEASE_WAIT_SECONDS
    assert "falling back to default" in caplog.text


def test_explicit_none_is_the_section_default_silently(caplog):
    # `wait_seconds:` with no YAML value parses to None: the registered default, not a typo.
    with caplog.at_level(logging.WARNING):
        assert resolve_lease_wait_seconds({"agent": {"turn_lease": {"wait_seconds": None}}}) == LEASE_WAIT_SECONDS
    assert not caplog.records


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


def test_admission_uses_default_when_config_broken(monkeypatch):
    import hermes_cli.config as hconfig

    monkeypatch.setattr(hconfig, "load_config_readonly", lambda: (_ for _ in ()).throw(RuntimeError("no config")))
    db = _WaitDB(acquire_result=True)
    _admit(_agent(db), db)
    assert db.kwargs["wait_seconds"] == LEASE_WAIT_SECONDS


def test_wait_seconds_reaches_admission_through_the_real_loader(tmp_path, monkeypatch):
    """`agent.turn_lease.wait_seconds` in config.yaml is what admission passes to the DB —
    through ``load_config_readonly``, the loader the admission path reads (DEFAULT_CONFIG
    registers the key, so a user value deep-merges instead of being dropped). The load cache
    is keyed on the file signature (path, mtime_ns, size), so a fresh temp HERMES_HOME cannot
    alias another home."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    (home / "config.yaml").write_text("agent:\n  turn_lease:\n    wait_seconds: 7\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_cli.config as hconfig

    assert resolve_lease_wait_seconds(hconfig.load_config_readonly()) == 7.0
    db = _WaitDB(acquire_result=True)
    _admit(_agent(db), db)
    assert db.kwargs["wait_seconds"] == 7.0
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert resolve_lease_wait_seconds(DEFAULT_CONFIG) == LEASE_WAIT_SECONDS


# ------------------------------------------------------- unchanged user contracts


def test_timeout_rejection_is_byte_identical_and_codes_stable():
    db = _WaitDB(row=(_foreign_holder(), time.time() - 60.0, time.time() + 240.0), acquire_result=False)
    admission = _admit(_agent(db), db)

    assert admission.lease is None
    assert db.kwargs["wait_seconds"] == LEASE_WAIT_SECONDS  # no clamp: the wait is the knob's job
    result = admission.early_result
    assert result["final_response"] == (
        "⏳ Another Hermes process kept this session busy too long. Your message was not "
        "processed - wait for the other process to finish, then send it again."
    )
    assert result["error"] == "session_turn_lease_timeout:sess-lease"
    assert result["failure_reason"] == "session_busy"
    assert result["failure_retryable"] is True


def test_interrupt_rejection_text_unchanged():
    db = _WaitDB(row=(_foreign_holder(), time.time() - 60.0, time.time() + 240.0), acquire_result=False)
    admission = _admit(_agent(db, interrupted=True), db)

    result = admission.early_result
    assert result["interrupted"] is True
    assert result["final_response"] == (
        "Stopped waiting for another Hermes process on this session. Your message was not processed."
    )


# ------------------------------------------------------------ holder diagnostics


def test_timeout_log_names_the_foreign_holder(caplog):
    db = _WaitDB(row=(_foreign_holder(), time.time() - 60.0, time.time() + 240.0), acquire_result=False)
    with caplog.at_level(logging.ERROR, logger="run_agent"):
        _admit(_agent(db), db)

    text = caplog.text
    for fact in ("pid=999", "pidns=222", "platform=cli", "since", "unprovable"):
        assert fact in text, fact


def test_timeout_log_marks_same_namespace_holder_probeable(caplog):
    db = _WaitDB(row=(_ours_holder(), time.time() - 60.0, time.time() + 240.0), acquire_result=False)
    with caplog.at_level(logging.ERROR, logger="run_agent"):
        _admit(_agent(db), db)

    assert "pid=999" in caplog.text and "probe-able" in caplog.text
    assert "unprovable" not in caplog.text


def test_timeout_log_handles_unstamped_holder(caplog):
    legacy = "pid=999:turn=old:platform=cli"  # pre-upgrade row: no pidns stamp
    db = _WaitDB(row=(legacy, time.time() - 60.0, time.time() + 240.0), acquire_result=False)
    with caplog.at_level(logging.ERROR, logger="run_agent"):
        _admit(_agent(db), db)

    assert "pid=999" in caplog.text and "unstamped(pre-upgrade)" in caplog.text


def test_malformed_diag_row_never_breaks_admission():
    """A shim returning garbage (dict, wrong arity, garbage timestamp) must not raise out of
    admission and must still produce the standard timeout result."""
    for junk in ({"holder": "pid=999"}, ("pid=999", "not-a-time"), ("pid=999",), ("pid=999", 1e30, 1e40)):
        db = _WaitDB(row=junk, acquire_result=False)
        admission = _admit(_agent(db), db)  # must not raise
        assert admission.early_result["failure_reason"] == "session_busy"
        assert db.kwargs["wait_seconds"] == LEASE_WAIT_SECONDS


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


def test_interrupt_branch_keeps_its_own_log_and_no_false_timeout_line(caplog):
    db = _WaitDB(row=(_foreign_holder(), time.time() - 60.0, time.time() + 240.0), acquire_result=False)
    with caplog.at_level(logging.INFO, logger="run_agent"):
        _admit(_agent(db, interrupted=True), db)
    assert "aborted by interrupt" in caplog.text
    assert "wait timed out" not in caplog.text  # never claims a timeout that did not happen


def test_timeout_log_prefix_survives_a_readerless_db(caplog):
    """Main's exact error line must still be emitted when the row is unreadable (shim DB):
    the holder facts are a suffix, never a replacement for the pre-existing log."""

    class _NoReaderDB(_WaitDB):
        get_session_turn_lease = None  # type: ignore[assignment]  # hides the method: callable(getattr(...)) is False

    db = _NoReaderDB(acquire_result=False)
    with caplog.at_level(logging.ERROR, logger="run_agent"):
        _admit(_agent(db), db)
    assert "session turn lease wait timed out for sess-lease" in caplog.text


# ------------------------------------------- the diagnostic read itself (real DB)


def test_get_session_turn_lease_against_real_sqlite(tmp_path):
    from hermes_state import SessionDB

    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.create_session("lease-read", source="test")

    assert db.get_session_turn_lease("lease-read") is None  # absent row
    assert db.get_session_turn_lease("") is None            # empty id, never queried

    holder = f"pid={__import__('os').getpid()}:turn=reader"
    assert db.try_acquire_session_turn_lease("lease-read", holder, ttl_seconds=30)
    row = db.get_session_turn_lease("lease-read")
    assert row is not None and row[0] == holder and row[1] <= time.time() <= row[2]

    db.release_session_turn_lease("lease-read", holder)
    assert db.get_session_turn_lease("lease-read") is None  # gone after release

    assert db.try_acquire_session_turn_lease("lease-read", holder, ttl_seconds=0.05)
    time.sleep(0.15)
    assert db.get_session_turn_lease("lease-read") is None  # gone after expiry


def test_live_holder_release_ends_the_wait_and_dead_row_expires_on_its_own(tmp_path):
    """The two real-behaviour anchors (measured, not asserted): a live foreign-namespace
    holder that releases ends the waiter's wait at release, and a dead one (no refresher)
    is reclaimed at its expiry — both without ever consulting the pidns probe. These are
    why no per-namespace clamp is warranted: only the configured wait should cut a wait."""
    import os

    from hermes_state import SessionDB

    path = tmp_path / "state.db"
    holder = f"pid={os.getpid() + 7}:pidns={_SIBLING}:turn=sibling:platform=cli"  # foreign stamp
    waiter = f"pid={os.getpid()}:pidns={_OURS}:turn=waiter:platform=desktop"
    seeded = SessionDB(path)
    waiter_db = SessionDB(path)

    # Dead holder: row expires in ~0.5 s -> reclaimed at expiry, far below any wait.
    assert seeded.try_acquire_session_turn_lease("sess-lease", holder, ttl_seconds=0.5)
    t0 = time.monotonic()
    assert waiter_db.acquire_session_turn_lease(
        "sess-lease", waiter, ttl_seconds=300.0, wait_seconds=30.0, poll_interval_seconds=0.1)
    assert time.monotonic() - t0 < 10.0

    # Live holder releasing mid-wait: the wait ends at the release, not at expiry.
    waiter_db.release_session_turn_lease("sess-lease", waiter)
    assert seeded.try_acquire_session_turn_lease("sess-lease", holder, ttl_seconds=300.0)
    import threading

    threading.Timer(1.0, lambda: seeded.release_session_turn_lease("sess-lease", holder)).start()
    t0 = time.monotonic()
    assert waiter_db.acquire_session_turn_lease(
        "sess-lease", f"{waiter}-2", ttl_seconds=300.0, wait_seconds=30.0, poll_interval_seconds=0.1)
    assert time.monotonic() - t0 < 10.0

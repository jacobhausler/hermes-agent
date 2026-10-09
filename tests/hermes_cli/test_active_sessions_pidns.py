"""Active-session liveness is PID-namespace qualified.

Two Hermes processes sharing one HERMES_HOME across PID namespaces (e.g.
``hermes serve`` and an agent container on one shared volume) each saw the
other's live processes as absent — pids are only meaningful inside their own
PID namespace — and pruned each other's ``runtime/active_sessions.json``
entries as dead ("Pruned 3 stale active session lease(s)"), so a second
writer could be admitted into an actively-owned session. A foreign-stamped
entry is resolved by its kernel lease witness, not its pid: held -> live
(fences this namespace out), unheld -> dead (reclaimed, so a crashed
sibling's rows cannot immortal-block the session). Unstamped entries keep the
exact pre-stamp probe (legacy rollout), like the state.db flock records in
``hermes_state_pidns``.
"""
import fcntl
import os
import sys

import pytest

from hermes_cli import active_sessions as asc
import hermes_state_pidns as pidns_mod


OWN = "4026533480"   # writer container
OTHER = "4026533757"  # sibling container

posix_flock = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX flock witness semantics only")


@pytest.fixture
def linux_ns(monkeypatch):
    """Deterministic Linux-style namespaces: local == OWN.

    Only ``_LOCAL_PID_NS`` is patched: every decision now flows through
    hermes_state_pidns' shared matrix, and patching two sources of truth could
    hide a divergence between them.
    """
    monkeypatch.setattr(pidns_mod, "_LOCAL_PID_NS", pidns_mod.LocalPidNamespace(OWN, True))
    return OWN


@pytest.fixture
def fake_start_times(monkeypatch):
    """Pin start-time probes so liveness asserts do not depend on psutil."""
    monkeypatch.setattr(asc, "_process_start_time", lambda pid: 1234.5)
    monkeypatch.setattr(asc, "_own_start_time", lambda: 1234.5)


def _entry(**over):
    e = {"lease_id": "l1", "session_id": "s1", "surface": "desktop",
         "pid": 41000, "process_start_time": 1234.5, "started_at": 1.0,
         "updated_at": 1.0, "track_liveness": True}
    e.update(over)
    return e


# ---------------------------------------------------------------- liveness

def test_foreign_stamped_pid_probe_is_unknowable(linux_ns):
    # A sibling container's live process looks absent to our /proc: the pid
    # alone must yield "unknowable", and `lenient` must NOT degrade it to dead
    # — that degradation is exactly the wrong-the-container prune this stops.
    assert asc._pid_liveness(41000, 1234.5, pidns=OTHER, lenient=False) is None
    assert asc._pid_liveness(41000, 1234.5, pidns=OTHER, lenient=True) is None


def test_same_ns_probe_unchanged(linux_ns, fake_start_times):
    # same pid + mismatched start time reads dead inside the writer's namespace
    assert asc._pid_liveness(os.getpid(), 1.0, pidns=OWN, lenient=True) is False
    assert asc._pid_liveness(os.getpid(), pidns=OWN, lenient=True) is True


def test_unstamped_entry_keeps_probing(linux_ns):
    # Legacy rollout: registry entries never expire; refusing unstamped probes
    # would permanently disable dead-owner cleanup (flock-record policy).
    assert asc._pid_liveness(999_999_999, 1234.5, pidns=None, lenient=False) is False


@pytest.mark.parametrize("stamp", ["", "   ", 42, True])
def test_garbage_stamp_treated_as_unstamped(linux_ns, stamp):
    assert asc._pid_liveness(999_999_999, 1234.5, pidns=stamp, lenient=False) is False


# ---------------------------------------------------------------- witness

@posix_flock
def test_witness_held_is_live_unheld_is_dead(tmp_path):
    state = tmp_path / "active_sessions.json"
    assert asc._hold_witness(state, "l1") is True
    try:
        assert asc._witness_liveness(
            {"lease_id": "l1", "lease_witness": asc._LEASE_WITNESS_VERSION}, state) is True
    finally:
        asc._drop_witness(state, "l1")
    # holder gone (the kernel auto-releases flocks on process death) -> dead
    assert asc._witness_liveness(
        {"lease_id": "l1", "lease_witness": asc._LEASE_WITNESS_VERSION}, state) is False


def test_pre_witness_or_broken_entry_is_unknowable(tmp_path):
    # A stamped row from a build without the witness (or a foreign witness
    # version) can never be proven dead by probe — kept live, never wrongly
    # pruned (half-upgraded registries fail closed for exclusivity).
    assert asc._witness_liveness({"lease_id": "l1"}, tmp_path) is None
    assert asc._witness_liveness({"lease_id": "l1", "lease_witness": "leases-v2"}, tmp_path) is None


@posix_flock
def test_prune_keeps_witness_held_and_reclaims_dead(linux_ns, tmp_path):
    state = tmp_path / "active_sessions.json"
    held = _entry(lease_id="held", session_id="s-held", pidns=OTHER,
                  lease_witness=asc._LEASE_WITNESS_VERSION)
    gone = _entry(lease_id="gone", session_id="s-gone", pidns=OTHER,
                  lease_witness=asc._LEASE_WITNESS_VERSION)
    dead_local = _entry(lease_id="dead", session_id="s-dead", pid=999_999_999, pidns=OWN)
    assert asc._hold_witness(state, "held") is True
    try:
        kept = asc._prune_dead([held, gone, dead_local], strict=True, state_path=state)
    finally:
        asc._drop_witness(state, "held")
    assert [e["lease_id"] for e in kept] == ["held"]
    # the reclaimed row's stale witness file is unlinked
    assert not asc._witness_path_for(state, "gone").exists()


def test_prune_without_state_path_keeps_foreign_live(linux_ns):
    # No registry path => witness unlocatable => conservative keep-live.
    kept = asc._prune_dead(
        [_entry(pidns=OTHER, lease_witness=asc._LEASE_WITNESS_VERSION)], strict=True)
    assert len(kept) == 1


def test_prune_pre_witness_foreign_stamped_kept(linux_ns, tmp_path):
    # Half-upgraded registry: stamped but witness-less => unknowable => kept,
    # and the targeted claim does NOT surface a registry error.
    kept = asc._prune_dead([_entry(pidns=OTHER)], strict=True,
                           target_session_id="s1",
                           state_path=tmp_path / "active_sessions.json")
    assert len(kept) == 1


# ---------------------------------------------------------------- pruning

@posix_flock
def test_prune_targeted_keeps_foreign_and_exclusivity_refuses(linux_ns, tmp_path, monkeypatch):
    # A live foreign entry (witness held) for the session you want to claim
    # stays LIVE — the claim fails through the normal owner-held path, never
    # overwrites the row.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    state_path = asc._state_path(tmp_path)
    asc._write_entries(state_path, [_entry(lease_id="l1", session_id="s1", pidns=OTHER,
                                           lease_witness=asc._LEASE_WITNESS_VERSION)])
    assert asc._hold_witness(state_path, "l1") is True
    try:
        lease, refusal = asc.try_acquire_active_session(
            session_id="s1", surface="cli", config={}, track_liveness=True)
    finally:
        asc._drop_witness(state_path, "l1")
    assert lease is None
    assert refusal is not None and refusal.reason == asc.SESSION_NOT_OWNED
    assert "PID namespace" in str(refusal)  # operator sees WHY, not just "busy"


@posix_flock
def test_dead_foreign_row_is_reclaimed_and_session_claimable(linux_ns, tmp_path, monkeypatch):
    # The crash-restart case both council seats demanded a test for: the
    # sibling died in an OLD namespace (new pids, old stamp, witness no longer
    # held). Its row must NOT immortal-fence the session — the next writer
    # claims it and the reclaimed witness file is cleaned up.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    state_path = asc._state_path(tmp_path)
    asc._write_entries(state_path, [_entry(lease_id="ghost", session_id="s1", pidns=OTHER,
                                           lease_witness=asc._LEASE_WITNESS_VERSION)])
    ghost_witness = asc._witness_path_for(state_path, "ghost")
    ghost_witness.parent.mkdir(parents=True, exist_ok=True)
    ghost_witness.touch()  # the crashed holder left its witness file behind
    lease, refusal = asc.try_acquire_active_session(
        session_id="s1", surface="cli", config={}, track_liveness=True)
    assert refusal is None and lease is not None
    assert not ghost_witness.exists()
    assert asc._witness_liveness({"lease_id": lease.lease_id,
                                  "lease_witness": asc._LEASE_WITNESS_VERSION},
                                 state_path) is True
    asc.release_active_session(lease)


@posix_flock
def test_capacity_counts_live_foreign_rows(linux_ns, tmp_path, monkeypatch):
    # Foreign live rows count toward max_concurrent_sessions (they fence too).
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    state_path = asc._state_path(tmp_path)
    asc._write_entries(state_path, [
        _entry(lease_id="a", session_id="sa", pidns=OTHER,
               lease_witness=asc._LEASE_WITNESS_VERSION),
        _entry(lease_id="b", session_id="sb", pidns=OTHER, surface="tui",
               lease_witness=asc._LEASE_WITNESS_VERSION)])
    assert asc._hold_witness(state_path, "a") is True
    assert asc._hold_witness(state_path, "b") is True
    try:
        lease, refusal = asc.try_acquire_active_session(
            session_id="sc", surface="cli", config={"max_concurrent_sessions": 2},
            track_liveness=True)
    finally:
        asc._drop_witness(state_path, "a")
        asc._drop_witness(state_path, "b")
    assert lease is None
    assert refusal is not None and refusal.reason == asc.MAX_CONCURRENT_SESSIONS


def test_prune_untracked_target_keeps_sibling_foreign(linux_ns, tmp_path):
    # Unrelated foreign sibling under an untracked probe: stays, does not block.
    entries = [_entry(pidns=OTHER)]
    kept = asc._prune_dead(entries, strict=False, target_session_id="other-session",
                           state_path=tmp_path / "active_sessions.json")
    assert kept == entries


# ---------------------------------------------------------------- self-orphan sweep

def test_self_orphan_sweep_spares_foreign_pid_collision(linux_ns):
    collide = _entry(lease_id="mine", session_id="s1", pid=os.getpid(), pidns=OTHER,
                     started_at=0.0)  # ancient, not in live ids
    ours = _entry(lease_id="ghost", session_id="s2", pid=os.getpid(), pidns=OWN,
                  started_at=0.0)
    kept = asc._drop_self_orphans([collide, ours], set())
    assert [e["lease_id"] for e in kept] == ["mine"]  # foreign survives; ours is reaped


def test_self_orphan_sweep_legacy_unstamped_still_reaped(linux_ns):
    legacy = _entry(lease_id="ghost", session_id="s1", pid=os.getpid(), started_at=0.0)
    assert asc._drop_self_orphans([legacy], set()) == []


# ---------------------------------------------------------------- re-entrancy

def test_same_writer_vetoed_across_namespaces(linux_ns):
    meta = {"live_session_id": "abc"}
    entry = _entry(pid=os.getpid(), pidns=OTHER, metadata=meta)
    assert asc._is_same_writer(entry, meta) is False          # foreign ns: two writers
    entry["pidns"] = OWN
    assert asc._is_same_writer(entry, meta) is True            # same ns: re-entrancy
    del entry["pidns"]
    assert asc._is_same_writer(entry, meta) is True            # unstamped legacy


# ---------------------------------------------------------------- strict schema

@pytest.mark.parametrize("bad", ["x", 4026533480, {}, True, "  "])
def test_strict_read_rejects_malformed_pidns(tmp_path, bad):
    state = tmp_path / "active_sessions.json"
    asc._write_entries(state, [_entry(pidns=bad)])
    with pytest.raises(asc.ActiveSessionRegistryError):
        asc._read_entries(state, strict=True)


def test_strict_read_rejects_unknown_witness(tmp_path):
    state = tmp_path / "active_sessions.json"
    asc._write_entries(state, [_entry(lease_witness="leases-v99")])
    with pytest.raises(asc.ActiveSessionRegistryError):
        asc._read_entries(state, strict=True)


# ---------------------------------------------------------------- acquisition stamp

@posix_flock
def test_acquired_entry_carries_stamp_and_witness(linux_ns, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    lease, refusal = asc.try_acquire_active_session(
        session_id="s1", surface="cli", config={}, track_liveness=True)
    assert refusal is None and lease is not None
    state_path = asc._state_path(tmp_path)
    entries = asc._read_entries(state_path)
    assert entries[0]["pidns"] == OWN
    assert entries[0]["lease_witness"] == asc._LEASE_WITNESS_VERSION
    asc.release_active_session(lease)
    # releasing drops the row AND unlinks the witness file
    assert asc._read_entries(state_path) == []
    assert not asc._witness_path_for(state_path, lease.lease_id).exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX stamping is a Linux concept")
def test_unwritable_witness_keeps_entry_unstamped(linux_ns, tmp_path, monkeypatch):
    # If the witness cannot be held there is no trustworthy death signal: the
    # row must NOT be born stamped (immortal-fence risk) — it stays on the
    # legacy bare-pid probe instead.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(asc, "_hold_witness", lambda state_path, lease_id: False)
    lease, refusal = asc.try_acquire_active_session(
        session_id="s1", surface="cli", config={}, track_liveness=True)
    assert refusal is None and lease is not None
    entries = asc._read_entries(asc._state_path(tmp_path))
    assert "pidns" not in entries[0] and "lease_witness" not in entries[0]


# ---------------------------------------------------------------- turn marker

@posix_flock
def test_writer_identity_stamps_turn_marker_writer(linux_ns):
    from tui_gateway.turn_marker import _writer_identity, marker_writer_state
    ident = _writer_identity()
    assert ident["writer_pid"] == os.getpid()
    assert ident.get("writer_pidns") == OWN
    # The reader treats a foreign writer as unknown, never dead.
    assert marker_writer_state({"writer_pid": 41000, "writer_start_time": 1234.5,
                                "writer_pidns": OTHER}) == "unknown"


@posix_flock
def test_read_turn_marker_round_trips_writer_pidns(linux_ns, tmp_path):
    from tui_gateway.turn_marker import read_turn_marker, record_turn_start
    record_turn_start(tmp_path, "s1", "keep going", attempts=1)
    marker = read_turn_marker(tmp_path, "s1")
    assert marker is not None
    assert marker["writer_pid"] == os.getpid()
    assert marker.get("writer_pidns") == OWN

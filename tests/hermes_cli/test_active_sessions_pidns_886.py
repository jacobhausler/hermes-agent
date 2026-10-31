"""est-2ek.1.886 — active-session liveness is PID-namespace qualified.

serve and agent containers share one HERMES_HOME; each saw the other's live
processes as absent (pids are only meaningful inside their own PID namespace)
and pruned each other's ``runtime/active_sessions.json`` entries as dead
("Pruned 3 stale active session lease(s)"), so a CLI could spawn into a
desktop-owned session. Stamped entries are treated as UNPROVABLE — like the
state.db turn leases — never as dead.
"""
import os

import pytest

from hermes_cli import active_sessions as asc
import hermes_state_pidns as pidns_mod


OWN = "4026533480"   # serve container
OTHER = "4026533757"  # agent container


@pytest.fixture
def linux_ns(monkeypatch):
    """Deterministic Linux-style namespaces: local == OWN."""
    monkeypatch.setattr(asc, "pid_namespace_id", lambda: OWN)
    monkeypatch.setattr(pidns_mod, "_LOCAL_PID_NS", pidns_mod.LocalPidNamespace(OWN, True))
    return OWN


def _entry(**over):
    e = {"lease_id": "l1", "session_id": "s1", "surface": "desktop",
         "pid": 41000, "process_start_time": 1234.5, "started_at": 1.0,
         "updated_at": 1.0, "track_liveness": True}
    e.update(over)
    return e


# ---------------------------------------------------------------- liveness

def test_foreign_stamped_entry_is_never_dead(linux_ns):
    # The intake's incident: agent container prunes the serve writer.
    assert asc._pid_liveness(41000, 1234.5, pidns=OTHER, lenient=False) is None
    # lenient must NOT degrade it to dead — that is the wrong-the-container prune.
    assert asc._pid_liveness(41000, 1234.5, pidns=OTHER, lenient=True) is None


def test_same_ns_probe_unchanged(linux_ns):
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


# ---------------------------------------------------------------- pruning

def test_prune_keeps_foreign_stamped_live_entry(linux_ns):
    live_foreign = _entry(pid=41000, pidns=OTHER)          # not dead here
    mine_dead = _entry(lease_id="l2", session_id="s2", pid=999_999_999,
                       pidns=OWN)                           # provably dead
    kept = asc._prune_dead([live_foreign, mine_dead], strict=True)
    assert [e["lease_id"] for e in kept] == ["l1"]


def test_prune_targeted_keeps_foreign_and_exclusivity_refuses(linux_ns, tmp_path, monkeypatch):
    # A foreign entry for the session you want to claim stays LIVE — the claim
    # then fails through the normal owner-held path, never overwrites the row.
    entries = [_entry(session_id="s1", pidns=OTHER)]
    kept = asc._prune_dead(entries, strict=True, target_session_id="s1")
    assert kept == entries  # not pruned, not a registry error

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    state_path = asc._state_path(tmp_path)
    asc._write_entries(state_path, [_entry(session_id="s1", pidns=OTHER)])
    lease, refusal = asc.try_acquire_active_session(
        session_id="s1", surface="cli", config={}, track_liveness=True)
    assert lease is None
    assert refusal is not None and refusal.reason == asc.SESSION_NOT_OWNED


def test_prune_untracked_target_keeps_sibling_foreign(linux_ns):
    # Unrelated foreign sibling under an untracked probe: stays, does not block.
    entries = [_entry(pidns=OTHER)]
    kept = asc._prune_dead(entries, strict=False, target_session_id="other-session")
    assert kept == entries


# ---------------------------------------------------------------- self-orphan sweep

def test_self_orphan_sweep_spares_foreign_pid_collision(linux_ns):
    collide = _entry(lease_id="mine", session_id="s1", pid=os.getpid(), pidns=OTHER,
                     started_at=0.0)  # ancient, not in live ids
    ours = _entry(lease_id="ghost", session_id="s2", pid=os.getpid(), pidns=OWN,
                  started_at=0.0)
    kept = asc._drop_self_orphans([collide, ours], set())
    assert [e["lease_id"] for e in kept] == ["mine"]  # foreign collision survives; ours is reaped


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


# ---------------------------------------------------------------- acquisition stamp

def test_acquired_entry_carries_pidns_stamp(linux_ns, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    lease, refusal = asc.try_acquire_active_session(
        session_id="s1", surface="cli", config={}, track_liveness=True)
    assert refusal is None and lease is not None
    entries = asc._read_entries(asc._state_path(tmp_path))
    assert entries[0]["pidns"] == OWN


def test_writer_identity_stamps_turn_marker_writer(linux_ns):
    from tui_gateway.turn_marker import _writer_identity
    ident = _writer_identity()
    assert ident["writer_pid"] == os.getpid()
    assert ident.get("writer_pidns") == OWN
    # The reader treats a foreign writer as unknown, never dead.
    from tui_gateway.turn_marker import marker_writer_state
    assert marker_writer_state({"writer_pid": 41000, "writer_start_time": 1234.5,
                                "writer_pidns": OTHER}) == "unknown"

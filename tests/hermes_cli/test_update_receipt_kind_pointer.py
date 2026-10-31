"""Tests: the latest receipt pointer is scoped per receipt kind (est-4kc1i).

The shared ``latest.json`` names the newest receipt of ANY kind — the plugin-check cadence
and pm syncs replace it independently. It must not be the only way an update receipt stays
reachable: when it held a plugin-check, an update receipt that still owed a fleet restart
could no longer be read or settled from, and the restart-debt warning could never discharge
(review 2026-10-31, seat_a #4 / seat_b F2). ``latest-update.json`` is the scoped pointer
only update writes touch. These tests pin the invariants:

- both kinds coexist: a plugin-check replacing the shared pointer leaves the scoped
  update pointer (and every obligation reader) intact;
- the discharge path survives: ``settle_latest_receipt_fleet`` still finds and settles
  the update receipt after a plugin-check write, without clobbering the newer
  plugin-check off the shared pointer;
- a legacy home with only a shared pointer still answers, but only for update-kind
  payloads — a plugin-check alone there reads as "nothing owed", the safe direction.
"""

from __future__ import annotations

import json

import pytest

import hermes_cli.update_receipt as ur


@pytest.fixture
def homed(tmp_path, monkeypatch):
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    import pm.receipt as pm_receipt_mod

    monkeypatch.setattr(pm_receipt_mod, "_receipt_dir", lambda: tmp_path / "logs" / "update_receipts")
    monkeypatch.setattr(ur, "_receipt_dir", lambda: tmp_path / "logs" / "update_receipts")
    return tmp_path / "logs" / "update_receipts"


@pytest.fixture(autouse=True)
def _isolated_receipt_state():
    import pm.receipt as pm_receipt_mod

    pm_receipt_mod._current.set(None)
    pm_receipt_mod._completed_by_update.set(None)
    ur._current.set(None)
    with ur.update_receipt_scope():
        pm_receipt_mod._current.set(None)
        pm_receipt_mod._completed_by_update.set(None)
        ur._current.set(None)
        yield


def _plugin_check_write():
    """A plugin-check receipt through the real pm writer (the cadence's exact path)."""
    import pm.receipt as pm_receipt_mod

    token = pm_receipt_mod.begin("plugin-check")
    pm_receipt_mod.record_plugin_checks(
        [{"name": "hermes-workflows", "class": "drift", "update_available": None}]
    )
    path = pm_receipt_mod.finalize("ok", token=token)
    assert path is not None
    return json.loads(path.read_text(encoding="utf-8"))


def _owe_a_restart():
    """Finalize an update receipt that reports a stale fleet and owes the restart."""
    ur.begin_update_receipt()
    ur.record_gateway_restart(incomplete=True, phase_error="gateway default still on pre-pull modules")
    update_id = ur.current_correlation_id()
    path = ur.finalize_update_receipt(
        "partial",
        fleet=[{"profile": "default", "pid": 1, "code_sha": "0" * 40, "state": "stale"}],
    )
    assert path is not None
    return update_id


class TestKindsCoexist:
    def test_finalized_update_receipt_is_kind_update(self, homed):
        update_id = _owe_a_restart()
        stored = ur.read_latest_receipt()
        assert stored is not None
        assert stored["kind"] == "update"
        assert stored["update_id"] == update_id

    def test_plugin_check_replaces_shared_pointer_not_scoped(self, homed):
        update_id = _owe_a_restart()
        check = _plugin_check_write()

        # Both kinds now coexist in the store: shared pointer answers pm/desktop,
        # the scoped pointer still answers the update obligation reader.
        shared = json.loads((homed / "latest.json").read_text(encoding="utf-8"))
        assert shared["kind"] == "plugin-check"
        assert shared is not None and check["update_id"] != update_id
        scoped = json.loads((homed / "latest-update.json").read_text(encoding="utf-8"))
        assert scoped["kind"] == "update"
        assert scoped["update_id"] == update_id
        assert ur.read_latest_receipt()["update_id"] == update_id

    def test_running_receipt_survives_a_plugin_check_midrun(self, homed):
        """The cadence tick during an update must not hide the open run's record."""
        ur.begin_update_receipt()
        update_id = ur.current_correlation_id()
        _plugin_check_write()
        assert ur.read_latest_receipt()["update_id"] == update_id


class TestDischargePathSurvives:
    def test_settle_reaches_update_receipt_after_plugin_check(self, homed):
        update_id = _owe_a_restart()
        _plugin_check_write()

        settled = ur.settle_latest_receipt_fleet(
            [{"profile": "default", "pid": 2, "code_sha": "a" * 40, "state": "current"}],
            discharges=lambda receipt: True,
        )
        assert settled is True

        scoped = json.loads((homed / "latest-update.json").read_text(encoding="utf-8"))
        assert scoped["update_id"] == update_id
        assert scoped["gateway_restart"]["incomplete"] is False
        assert scoped["gateway_restart"]["settled_from_live_fleet_at"]
        assert scoped["fleet"][0]["state"] == "current"
        # The newer plugin-check keeps the shared pointer: a settle is not the newest event.
        shared = json.loads((homed / "latest.json").read_text(encoding="utf-8"))
        assert shared["kind"] == "plugin-check"

    def test_settle_refused_discharge_leaves_pointers_untouched(self, homed):
        update_id = _owe_a_restart()
        before = (homed / "latest-update.json").read_bytes()
        _plugin_check_write()

        assert ur.settle_latest_receipt_fleet([], discharges=lambda receipt: False) is False
        assert (homed / "latest-update.json").read_bytes() == before
        assert ur.read_latest_receipt()["update_id"] == update_id

    def test_settle_still_mirrors_shared_pointer_when_run_is_newest(self, homed):
        update_id = _owe_a_restart()
        assert ur.settle_latest_receipt_fleet(
            [{"profile": "default", "pid": 2, "code_sha": "a" * 40, "state": "current"}],
            discharges=lambda receipt: True,
        ) is True
        shared = json.loads((homed / "latest.json").read_text(encoding="utf-8"))
        assert shared["kind"] == "update"
        assert shared["update_id"] == update_id
        assert shared["gateway_restart"]["incomplete"] is False


class TestLegacyHomeFallback:
    def test_kindless_latest_json_answers_as_update_receipt(self, homed):
        """A home whose last update predates kind stamping keeps working byte-for-byte."""
        legacy = {"schema": 1, "update_id": "e" * 32, "outcome": "partial",
                  "fleet": [], "gateway_restart": {}}
        homed.mkdir(parents=True, exist_ok=True)
        (homed / "latest.json").write_text(json.dumps(legacy), encoding="utf-8")
        assert ur.read_latest_receipt()["update_id"] == "e" * 32

    def test_plugin_check_only_latest_json_reads_as_nothing_owed(self, homed):
        """No scoped pointer + a plugin-check on the shared one must not return a
        foreign receipt to the obligation readers; None is the safe answer."""
        _plugin_check_write()
        assert not (homed / "latest-update.json").exists()
        assert ur.read_latest_receipt() is None


def _pm_write(kind: str, outcome: str = "failed", exit_code: int = 1):
    """A pm receipt of ``kind`` through the real pm writer (its exact cadence/CLI path)."""
    import pm.receipt as pm_receipt_mod

    token = pm_receipt_mod.begin(kind)
    path = pm_receipt_mod.finalize(outcome, exit_code, token=token)
    assert path is not None
    return json.loads(path.read_text(encoding="utf-8"))


class TestObligationSurvival:
    """Folded contracts from the unit's second seat of the same defect (est-4kc1i): the
    writer seam is pure for EVERY pm kind, the clobber does not leak into the amend path
    or the manual-serve carry-forward chain, and pm's own ``update`` command kind never
    masquerades as an update obligation on the shared pointer."""

    @pytest.mark.parametrize("kind", ["plugin-check", "sync", "update"])
    def test_no_pm_kind_reaches_the_scoped_pointer(self, homed, kind):
        update_id = _owe_a_restart()
        pm_receipt = _pm_write(kind)

        assert ur.read_latest_receipt()["update_id"] == update_id
        shared = json.loads((homed / "latest.json").read_text(encoding="utf-8"))
        assert shared["kind"] == pm_receipt["kind"] == kind

    def test_terminal_finalize_leaves_both_pointers_byte_identical(self, homed):
        """No clobber between begin and finalize: the scoped pointer and the shared mirror
        are the same terminal payload (a reader of either sees the same run)."""
        _owe_a_restart()
        assert (homed / "latest-update.json").read_bytes() == (homed / "latest.json").read_bytes()

    def test_amended_followup_lands_on_the_obligation_after_a_clobber(self, homed):
        """A post-finalize follow-up amends the UPDATE receipt even once the cadence owns
        the shared pointer (the amend guard consults the scoped resolution)."""
        update_id = _owe_a_restart()
        _plugin_check_write()

        ur.amend_terminal_followup(update_id, "build", "venv rebuild died")

        read = ur.read_latest_receipt()
        assert read["update_id"] == update_id
        assert [f["step"] for f in read["followups"]] == ["build"]

    def test_manual_serve_carry_forward_survives_a_clobber_between_runs(self, homed, monkeypatch):
        """Begin's carry-forward snapshot reads the UPDATE receipt after a plugin-check moved
        the shared pointer: the production chain (plan row -> carried at begin -> filed owed
        at finalize) drops nothing."""
        import hermes_cli.update_serve_obligations as serve_obligations

        monkeypatch.setattr(serve_obligations, "defer_manual_serve", lambda runtime, **kw: False)
        row = {"kind": "serve", "profile": "work", "supervisor": "manual-serve",
               "restart_via": "respawn-argv", "pid": 900, "detail": {"create_time": 1000.0}}
        ur.begin_update_receipt()
        ur.record_fact("plan", {"runtimes": [row]})
        ur.finalize_update_receipt("partial", fleet=[])
        _plugin_check_write()

        ur.begin_update_receipt()
        assert ur.read_latest_receipt()["carried_manual_serves"] == [row]
        ur.finalize_update_receipt("success", fleet=[])
        assert ur.read_latest_receipt()["pending_manual_serves"] == [row]

    def test_pm_update_command_kind_never_masquerades_on_the_shared_pointer(self, homed):
        """A legacy home (no scoped pointer) whose shared pointer holds pm's own
        ``update``-command receipt must read as NOTHING OWED: the kind spells ``update``
        but only an updater receipt records ``argv``."""
        _pm_write("update")
        assert not (homed / "latest-update.json").exists()
        assert ur.read_latest_receipt() is None

    def test_settle_never_stamps_the_fleet_matrix_onto_a_pm_archive(self, homed):
        """The discharge lands on the update receipt and its pointer only: no pm_*.json
        archive — the plugin-check's or any other — absorbs a fleet matrix it never ran."""
        _owe_a_restart()
        _plugin_check_write()

        assert ur.settle_latest_receipt_fleet(
            [{"profile": "default", "pid": 2, "code_sha": "a" * 40, "state": "current"}],
            discharges=lambda receipt: True,
        ) is True
        archives = list(homed.glob("pm_*.json"))
        assert archives
        for path in archives:
            assert "fleet" not in json.loads(path.read_text(encoding="utf-8"))

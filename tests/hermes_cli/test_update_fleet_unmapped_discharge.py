"""Unmapped shadow rows settle from the live fleet; stale evidence still warns (est-4fmxp, UPD-4 seat B).

Both discharge legs of the restart-debt banner, on a single-profile host:

- the marker leg (``_marker_only_restart_obsolete``): an inventory row stopped without its
  profile (``stopped_unmapped``) is settled when the live fleet is current on the target code
  — the former requirement of a live row NO mapped obligation explains can never be met here;
- the marker-less reader leg: a receipt whose owed identities cannot be verified discharges
  when the live gateway SHA equals the SHA the obligation was written against;
- the regression bar the epic sets: a synthetic STALE gateway keeps every leg warning.
"""

import json

import pytest

from hermes_cli import update_cmd_fleet as fleet, update_receipt
from hermes_cli import update_host_obligation as host

_SHADOW = {"kind": "gateway", "profile": None, "pid": 101, "stopped_unmapped": True}
_CURRENT = {"profile": "default", "state": "current", "code_sha": "head", "pid": 8}
_STALE = {"profile": "default", "state": "stale", "code_sha": "old", "pid": 8}


@pytest.fixture()
def marker_home(monkeypatch, tmp_path):
    """Isolated host obligation + a checkout SHA of ``head`` for the marker leg."""
    monkeypatch.setattr(host, "host_obligation_path", lambda: tmp_path / "host-update-restart.json")
    monkeypatch.setattr(host, "host_obligation_present", lambda: (tmp_path / "host-update-restart.json").is_file())
    monkeypatch.setattr(fleet, "_current_checkout_sha", lambda: "head")
    return tmp_path


def _arm(marker_home, runtimes):
    armed = fleet._write_fleet_restart_pending_marker(expected_sha="head", runtimes=runtimes)
    assert armed and fleet._fleet_restart_obligation_armed()


def _rows(monkeypatch, rows):
    monkeypatch.setattr(update_receipt, "collect_fleet_versions", lambda **kw: list(rows))


class TestMarkerLeg:
    def test_single_profile_host_discharges_on_its_current_gateway(self, marker_home, monkeypatch):
        _arm(marker_home, [{"kind": "gateway", "profile": "default", "pid": 7}, _SHADOW])
        _rows(monkeypatch, [_CURRENT])
        assert fleet._update_owes_fleet_restart(receipt={}, pending_manual=[]) is False
        assert not fleet._fleet_restart_obligation_armed()

    def test_synthetic_stale_gateway_still_warns(self, marker_home, monkeypatch):
        """The epic's bar: the discharge must not become a blanket silence — a gateway still on
        pre-update code keeps the marker and the warning."""
        _arm(marker_home, [{"kind": "gateway", "profile": "default", "pid": 7}, _SHADOW])
        _rows(monkeypatch, [_STALE])
        assert fleet._update_owes_fleet_restart(receipt={}, pending_manual=[]) is True
        assert fleet._fleet_restart_obligation_armed()
        assert fleet._pending_fleet_restart_needed(receipt={}, pending_manual=[]) is True

    def test_empty_host_never_settles_the_shadow(self, marker_home, monkeypatch):
        _arm(marker_home, [{"kind": "gateway", "profile": "default", "pid": 7}, _SHADOW])
        _rows(monkeypatch, [])
        assert fleet._update_owes_fleet_restart(receipt={}, pending_manual=[]) is True
        assert fleet._fleet_restart_obligation_armed()


_RECEIPT_WITH_SHADOW = {
    "argv": ["hermes", "update"], "outcome": "success", "exit_code": 0,
    "plan": {"expected_sha": "pulled", "runtimes": [dict(_SHADOW)]},
    "fleet": [{"profile": "default", "state": "stale", "code_sha": "old"}],
}


class TestReaderLegWithoutMarker:
    """No marker (discharged or never armed): the receipt path must still hear the live fleet."""

    @pytest.fixture(autouse=True)
    def _no_marker(self, monkeypatch, tmp_path):
        monkeypatch.setattr(host, "host_obligation_present", lambda: False)
        monkeypatch.setattr(fleet, "_current_checkout_sha", lambda: "pulled")
        monkeypatch.setattr("hermes_cli.update_cmd._current_checkout_sha", lambda: "pulled")

    def test_live_fleet_at_pulled_sha_discharges_despite_unverifiable_owed(self, monkeypatch):
        _rows(monkeypatch, [{"profile": "default", "state": "current", "code_sha": "pulled", "pid": 5}])
        assert fleet._update_owes_fleet_restart(receipt=dict(_RECEIPT_WITH_SHADOW), pending_manual=[]) is False
        assert fleet._pending_fleet_restart_needed(receipt=dict(_RECEIPT_WITH_SHADOW), pending_manual=[]) is False

    def test_synthetic_stale_gateway_still_warns(self, monkeypatch):
        _rows(monkeypatch, [{"profile": "default", "state": "stale", "code_sha": "old", "pid": 5}])
        assert fleet._update_owes_fleet_restart(receipt=dict(_RECEIPT_WITH_SHADOW), pending_manual=[]) is True
        assert fleet._pending_fleet_restart_needed(receipt=dict(_RECEIPT_WITH_SHADOW), pending_manual=[]) is True

    def test_wrong_sha_current_still_warns(self, monkeypatch):
        """``current`` is checkout-relative bookkeeping; the pulled SHA is the contract."""
        _rows(monkeypatch, [{"profile": "default", "state": "current", "code_sha": "other", "pid": 5}])
        assert fleet._update_owes_fleet_restart(receipt=dict(_RECEIPT_WITH_SHADOW), pending_manual=[]) is True

    def test_empty_host_still_warns(self, monkeypatch):
        _rows(monkeypatch, [])
        assert fleet._update_owes_fleet_restart(receipt=dict(_RECEIPT_WITH_SHADOW), pending_manual=[]) is True

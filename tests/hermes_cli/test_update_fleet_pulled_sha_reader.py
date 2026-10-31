"""The marker-less reader discharges a VERIFIED-owed receipt at the PULLED SHA (est-4fmxp follow-up).

#23 (UPD-4 seat B) opened the marker-less pulled-code discharge only for receipts whose owed
identities cannot be verified. A receipt with a verified owed set — the common single-profile
shape after an incomplete restart phase — stayed pinned to checkout HEAD: on a bot-pushed fork
HEAD moves minutes after the pull, the operator's `hermes gateway restart` leaves the fleet
`stale` (the label is checkout-relative) at the SHA the update actually pulled, and the startup
banner refired forever on a fleet provably past the update. These fixtures pin the extension
and its fail-closed directions: coverage must be proven for a verified owed set, a pre-pull
gateway still warns, and a receipt naming no pulled SHA never discharges through this leg.
"""

import pytest

from hermes_cli import update_cmd, update_cmd_fleet as fleet, update_receipt
from hermes_cli import update_host_obligation as host

_INCOMPLETE_KNOWN = {
    "outcome": "failed",
    "gateway_restart": {"incomplete": True, "phase_error": "one gateway stopped unmapped"},
    "plan": {"expected_sha": "PULL",
             "runtimes": [{"kind": "gateway", "profile": "default", "pid": 7, "code_sha": "OLD"}]},
    "post_update": {"sha": "PULL"},
    "fleet": [{"profile": "default", "state": "stale", "code_sha": "OLD"}],
}


@pytest.fixture(autouse=True)
def _no_marker_moved_head(monkeypatch, tmp_path):
    """No armed obligation, HEAD two pulls past the receipt's SHA."""
    monkeypatch.setattr(host, "host_obligation_path", lambda: tmp_path / "host-update-restart.json")
    monkeypatch.setattr(host, "host_obligation_present", lambda: False)
    monkeypatch.setattr(fleet, "_current_checkout_sha", lambda: "HEAD2")
    monkeypatch.setattr("hermes_cli.update_cmd._current_checkout_sha", lambda: "HEAD2")


def _rows(monkeypatch, rows):
    monkeypatch.setattr(update_receipt, "collect_fleet_versions", lambda **kw: list(rows))


def test_verified_owed_receipt_discharges_at_the_pulled_sha(monkeypatch):
    """The estate shape: restart bookkeeping incomplete, operator restarted, HEAD moved."""
    _rows(monkeypatch, [{"profile": "default", "state": "stale", "code_sha": "PULL", "pid": 5}])
    assert fleet._update_owes_fleet_restart(receipt=dict(_INCOMPLETE_KNOWN), pending_manual=[]) is False
    assert fleet._pending_fleet_restart_needed(receipt=dict(_INCOMPLETE_KNOWN), pending_manual=[]) is False


def test_synthetic_stale_gateway_still_warns(monkeypatch):
    """The epic's bar, reader side: a gateway still on PRE-PULL code keeps every leg loud."""
    _rows(monkeypatch, [{"profile": "default", "state": "stale", "code_sha": "OLD", "pid": 5}])
    assert fleet._update_owes_fleet_restart(receipt=dict(_INCOMPLETE_KNOWN), pending_manual=[]) is True
    assert fleet._pending_fleet_restart_needed(receipt=dict(_INCOMPLETE_KNOWN), pending_manual=[]) is True


def test_a_restarted_gateway_does_not_silence_a_still_down_profile(monkeypatch):
    """Verified owed sets need COVERAGE at the pulled SHA, not just a fleet running it:
    `work` named by the plan but absent from the live fleet keeps the debt standing."""
    receipt = dict(_INCOMPLETE_KNOWN)
    receipt["plan"] = {"expected_sha": "PULL", "runtimes": [
        {"kind": "gateway", "profile": "default", "pid": 7, "code_sha": "OLD"},
        {"kind": "gateway", "profile": "work", "pid": 9, "code_sha": "OLD"},
    ]}
    _rows(monkeypatch, [{"profile": "default", "state": "stale", "code_sha": "PULL", "pid": 5}])
    assert fleet._update_owes_fleet_restart(receipt=receipt, pending_manual=[]) is True


def test_current_at_a_foreign_sha_still_warns(monkeypatch):
    """``current`` is checkout-relative bookkeeping; the pulled SHA is the contract."""
    _rows(monkeypatch, [{"profile": "default", "state": "current", "code_sha": "OTHER", "pid": 5}])
    assert fleet._update_owes_fleet_restart(receipt=dict(_INCOMPLETE_KNOWN), pending_manual=[]) is True


def test_receipt_without_a_pulled_sha_never_takes_this_leg(monkeypatch):
    """No expected SHA, no post-update identity: the leg is skipped and HEAD currency still rules."""
    receipt = {k: v for k, v in _INCOMPLETE_KNOWN.items() if k != "post_update"}
    receipt["plan"] = {"runtimes": [{"kind": "gateway", "profile": "default", "pid": 7, "code_sha": "OLD"}]}
    _rows(monkeypatch, [{"profile": "default", "state": "current", "code_sha": "HEAD2", "pid": 5}])
    assert fleet._update_owes_fleet_restart(receipt=receipt, pending_manual=[]) is False  # final HEAD leg rules
    _rows(monkeypatch, [{"profile": "default", "state": "stale", "code_sha": "OLD", "pid": 5}])
    assert fleet._update_owes_fleet_restart(receipt=receipt, pending_manual=[]) is True

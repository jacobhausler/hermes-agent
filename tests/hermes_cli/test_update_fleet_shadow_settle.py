"""The stopped-unmapped shadow rows settle from the live fleet (UPD-4 seat B, est-4fmxp).

The single-profile trap: an unmapped shadow row demanded a local gateway that no mapped
owed row explains — impossible where one gateway serves the whole host — so the marker
stayed armed forever and the startup banner fired on every CLI start even after the
operator's remedy (``hermes gateway restart`` onto the pulled code). Both discharge legs
are pinned here with their fail-closed selves:

- fleet wholly current at the target SHA + shadow pid provably gone -> marker settles, banner silent;
- fleet wholly current, shadow pid alive -> still owed (a live pre-pull process IS the debt);
- a SYNTHETIC STALE gateway -> still owed AND the startup banner still fires (the hard gate);
- reader leg with no marker armed: a fleet serving the receipt's PULLED sha discharges even
  when the restart-phase bookkeeping stayed incomplete; a fleet serving pre-pull code warns.
"""

import pytest

from hermes_cli import process_identity, update_cmd, update_cmd_fleet as fleet
from hermes_cli import update_host_obligation as host, update_receipt

_MAPPED = {"kind": "gateway", "profile": "default", "pid": 7}
_SHADOW = {"kind": "gateway", "profile": None, "pid": 101, "stopped_unmapped": True}
_SHADOW_NO_PID = {"kind": "gateway", "profile": None, "stopped_unmapped": True}
_CURRENT = [{"profile": "default", "state": "current", "code_sha": "head", "pid": 8}]


@pytest.fixture
def armed_single_profile(monkeypatch, tmp_path):
    """Owes `default` plus its unmapped shadow; expected SHA and checkout both `head`."""
    monkeypatch.setattr(host, "host_obligation_path", lambda: tmp_path / "host-update-restart.json")
    monkeypatch.setattr(fleet, "_current_checkout_sha", lambda: "head")
    monkeypatch.setattr(update_cmd, "_current_checkout_sha", lambda: "head")
    fleet._write_fleet_restart_pending_marker(expected_sha="head", runtimes=[dict(_MAPPED), dict(_SHADOW)])
    assert fleet._fleet_restart_obligation_armed()


def _live_fleet(monkeypatch, rows):
    monkeypatch.setattr(update_receipt, "collect_fleet_versions", lambda **kw: [dict(r) for r in rows])


def _debt_receipt(live_sha: str, pulled: str = "PULL"):
    """A failed-completion update receipt: restart phase incomplete, fleet matrix skewed."""
    return {"outcome": "failed", "gateway_restart": {"incomplete": True, "phase_error": "unmapped stop"},
            "plan": {"runtimes": [dict(_MAPPED)]}, "post_update": {"sha": pulled},
            "fleet": [{"profile": "default", "state": "stale", "code_sha": live_sha}]}


def test_single_profile_shadows_settle_from_the_live_fleet(armed_single_profile, monkeypatch, capsys):
    _live_fleet(monkeypatch, _CURRENT)
    monkeypatch.setattr(process_identity, "_pid_alive_matches", lambda *a: False)
    assert fleet._marker_only_restart_obsolete() is True
    assert not fleet._fleet_restart_obligation_armed()
    fleet._warn_pending_fleet_restart_on_startup()
    assert "did not restart running gateways" not in capsys.readouterr().err


def test_live_shadow_pid_is_never_discharged_by_the_mapped_successor(armed_single_profile, monkeypatch):
    """The mapped gateway coming back current is `default`'s successor, not the shadow's,
    while the shadow process is alive: the debt stands fail-closed."""
    _live_fleet(monkeypatch, _CURRENT)
    monkeypatch.setattr(process_identity, "_pid_alive_matches", lambda *a: True)
    assert fleet._marker_only_restart_obsolete() is False
    assert fleet._fleet_restart_obligation_armed()
    monkeypatch.setattr(process_identity, "_pid_alive_matches", lambda *a: None)  # undecidable
    assert fleet._marker_only_restart_obsolete() is False
    assert fleet._fleet_restart_obligation_armed()


def test_shadow_row_without_a_pid_keeps_the_marker(armed_single_profile, monkeypatch, tmp_path):
    """Nothing to prove gone -> nothing settled (fail-closed), even with the fleet current."""
    fleet._write_fleet_restart_pending_marker(expected_sha="head",
                                              runtimes=[dict(_MAPPED), dict(_SHADOW_NO_PID)])
    _live_fleet(monkeypatch, _CURRENT)
    monkeypatch.setattr(process_identity, "_pid_alive_matches", lambda *a: False)
    assert fleet._marker_only_restart_obsolete() is False
    assert fleet._fleet_restart_obligation_armed()


def test_synthetic_stale_gateway_still_warns(armed_single_profile, monkeypatch, capsys):
    """The mandated regression gate: with the fleet still serving pre-pull code the settle
    must refuse AND the startup banner must still fire, live shadow pid notwithstanding."""
    _live_fleet(monkeypatch, [{"profile": "default", "state": "stale", "code_sha": "old", "pid": 8}])
    monkeypatch.setattr(process_identity, "_pid_alive_matches", lambda *a: False)
    assert fleet._marker_only_restart_obsolete() is False
    assert fleet._fleet_restart_obligation_armed()
    fleet._warn_pending_fleet_restart_on_startup()
    assert "did not restart running gateways" in capsys.readouterr().err
    assert fleet._update_owes_fleet_restart(receipt={}, pending_manual=[]) is True


def test_reader_discharges_at_the_pulled_sha_with_no_marker(monkeypatch):
    """No armed marker + incomplete restart bookkeeping is not evidence of live pre-pull
    modules once the fleet answers the SHA the update pulled (HEAD has since moved)."""
    monkeypatch.setattr(fleet, "_current_checkout_sha", lambda: "HEAD2")
    monkeypatch.setattr(update_cmd, "_current_checkout_sha", lambda: "HEAD2")
    _live_fleet(monkeypatch, [{"profile": "default", "state": "stale", "code_sha": "PULL", "pid": 9}])
    assert fleet._update_owes_fleet_restart(receipt=_debt_receipt("PULL"), pending_manual=[]) is False


def test_reader_still_warns_for_a_pre_pull_gateway(monkeypatch):
    """Same reader, same moved HEAD — but the gateway answers the PRE-PULL code: loud."""
    monkeypatch.setattr(fleet, "_current_checkout_sha", lambda: "HEAD2")
    monkeypatch.setattr(update_cmd, "_current_checkout_sha", lambda: "HEAD2")
    _live_fleet(monkeypatch, [{"profile": "default", "state": "stale", "code_sha": "OLD", "pid": 9}])
    assert fleet._update_owes_fleet_restart(receipt=_debt_receipt("OLD"), pending_manual=[]) is True

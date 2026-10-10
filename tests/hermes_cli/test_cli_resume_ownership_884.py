"""est-2ek.1.884 — the CLI refuses to resume a session the desktop/TUI created.

Intake: `hermes chat -c <title> --create-if-missing` resolved to a
desktop-created session (created_source=desktop) and ran a headless CLI turn
in it, holding the session's state.db turn lease for 50 minutes while the
user's desktop window waited for its own prompt. The registry fences a second
writer only while the owner process is live and pid-ns-provable; immutable
provenance is the cheaper, namespace-proof gate.
"""
from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import pytest

from hermes_constants import get_hermes_home
from hermes_state import SessionDB


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # conftest re-pins the import-time snapshot per test; without this the
    # guard's default-path SessionDB(read_only=True) reads the sandbox home,
    # not this test's (same pattern as tests/tui_gateway/test_history_inline_images).
    import hermes_state
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    monkeypatch.delenv("HERMES_CLI_RESUME_ANY_SURFACE", raising=False)
    return tmp_path


@pytest.fixture
def main_mod():
    import hermes_cli.main as mod
    return mod


def _mk(home: Path, sid: str, source: str):
    with SessionDB(home / "state.db") as db:
        db.create_session(sid, source=source)


def _guard(mod, resume, use_tui=False):
    args = Namespace(resume=resume, continue_last=None)
    mod._guard_cli_resume_ownership(args, use_tui=use_tui)
    return args


def test_cli_refuses_desktop_created(home, main_mod, capsys):
    _mk(home, "20261004_202600_1b6946", "desktop")
    with pytest.raises(SystemExit) as exc:
        _guard(main_mod, "20261004_202600_1b6946")
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "desktop" in err and "HERMES_CLI_RESUME_ANY_SURFACE" in err


def test_cli_refuses_tui_created(home, main_mod):
    _mk(home, "s-tui", "tui")
    with pytest.raises(SystemExit):
        _guard(main_mod, "s-tui")


def test_created_source_wins_over_live_routing_source(home, main_mod):
    # Desktop-created, later routed through telegram: provenance is the gate,
    # not the live routing column.
    _mk(home, "s-mix", "desktop")
    with SessionDB(home / "state.db") as db:
        db.record_gateway_session_peer(
            "s-mix", source="telegram", session_key="agent:main:telegram:dm:1", chat_id="1"
        )
    with SessionDB(home / "state.db", read_only=True) as db:
        row = db.get_session("s-mix")
    assert row["source"] == "telegram" and row["created_source"] == "desktop"
    with pytest.raises(SystemExit):
        _guard(main_mod, "s-mix")


def test_cli_may_resume_cli_created(home, main_mod):
    _mk(home, "s-cli", "cli")
    args = _guard(main_mod, "s-cli")  # must not raise
    assert args.resume == "s-cli"


def test_unknown_session_passes_through(home, main_mod):
    # The guard never invents "not found"; _init_agent owns that error.
    args = _guard(main_mod, "does-not-exist")
    assert args.resume == "does-not-exist"


def test_tui_launch_exempt(home, main_mod):
    # desktop and TUI are the same interactive class and legitimately reattach.
    _mk(home, "s-desk", "desktop")
    args = _guard(main_mod, "s-desk", use_tui=True)
    assert args.resume == "s-desk"


def test_override_env_proceeds_loudly(home, main_mod, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_CLI_RESUME_ANY_SURFACE", "1")
    _mk(home, "s-desk", "desktop")
    args = _guard(main_mod, "s-desk")
    assert args.resume == "s-desk"
    assert "anyway" in capsys.readouterr().err


def test_guard_never_breaks_on_db_failure(home, main_mod, monkeypatch):
    # A guard that can raise would brick every chat launch.
    _mk(home, "s-cli", "cli")

    def boom():
        raise RuntimeError("db down")
    monkeypatch.setattr(main_mod, "_session_db", boom)
    args = _guard(main_mod, "s-cli")
    assert args.resume == "s-cli"


def test_no_resume_flag_is_noop(home, main_mod):
    args = _guard(main_mod, None)
    assert args.resume is None

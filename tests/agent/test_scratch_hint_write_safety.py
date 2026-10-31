"""est-2ek.1.703 — the scratch hint must not invite writes the gate will deny.

Regression: the runtime environment block told every agent "Scratch directory:
<profiles>/<p>/cache/scratch (write temporary files and probes there ...)"
while HERMES_WRITE_SAFE_ROOT excluded that exact path, so every wake that
obeyed the instruction took a write_file denial — and a cron wake whose duty
starts with staging a probe died on the contradiction.

The producer must ask the gate itself (``agent.file_safety`` denies the write
on the same resolution path) instead of maintaining its own opinion about
writability: two opinions is exactly the drift this test exists to keep
closed. A hint the gate contradicts is worse than no hint — the prompt is
authoritative for the model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent import file_safety
from agent.prompt_builder import _local_host_hints


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_WRITE_SAFE_ROOT", raising=False)
    return home


def _scratch_hint():
    return "\n".join(part for part in _local_host_hints() if "Scratch directory" in part)


def test_gate_denies_scratch_hint_is_absent(hermes_home, monkeypatch):
    # Jail the deployment exactly like the estate fleet: work/ allowed,
    # profile scratch NOT. This is the est-2ek.1.703 incident configuration.
    (hermes_home / "work").mkdir()
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(hermes_home / "work"))

    from hermes_constants import get_scratch_dir

    scratch = str(get_scratch_dir())
    assert file_safety.is_write_denied(scratch), "fixture must reproduce the denial"

    hint = _scratch_hint()
    assert hint == "", f"prompt must not invite writes the gate denies, got: {hint!r}"
    assert "Scratch directory" not in "\n".join(_local_host_hints())


def test_gate_allows_scratch_hint_is_present(hermes_home, monkeypatch):
    # Default deployment (no safe-root jail): scratch is writable, the habit
    # hint must survive — the fix must not delete the feature it secures.
    from hermes_constants import get_scratch_dir

    assert not file_safety.is_write_denied(str(get_scratch_dir()))
    assert "Scratch directory" in _scratch_hint()


def test_gate_allows_scratch_via_configured_root(hermes_home, monkeypatch):
    # Safe root configured to INCLUDE scratch: the hint returns.
    from hermes_constants import get_scratch_dir

    scratch = str(get_scratch_dir())  # created on first access
    Path(scratch).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", scratch)
    assert not file_safety.is_write_denied(scratch)
    assert "Scratch directory" in _scratch_hint()

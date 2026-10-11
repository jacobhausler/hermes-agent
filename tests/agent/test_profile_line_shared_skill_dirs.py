"""est-2ek.1.941: the profile line must name configured shared skill dirs.

``_active_profile_line`` tells every profile agent that the default root's
skills/ "belongs to a different session … Do NOT modify" — even when the
operator configured ``skills.create_dir`` / ``skills.external_dirs`` to mount
exactly that directory (or any other dir outside the profile home) as the
session's own shared skill surface. The two messages contradicted each other,
so agents refused to use or write the very dirs ``skill_manage`` creates into.

Drive the real chain: a real config.yaml under a real HERMES_HOME, no mocks
of skill_utils (the function must consult the SAME config resolution the
skills list / skill_view use).
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent.system_prompt import build_system_prompt_parts


def _make_agent(**overrides):
    base = dict(
        load_soul_identity=False,
        skip_context_files=False,
        valid_tool_names=[],
        _task_completion_guidance=False,
        _tool_use_enforcement=False,
        _environment_probe=False,
        _kanban_worker_guidance="",
        _memory_store=None,
        _memory_manager=None,
        model="",
        provider="",
        platform="",
        pass_session_id=False,
        session_id="",
        _emit_status=lambda *_args, **_kwargs: None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _prompt_parts(agent):
    with (
        patch("agent.prompt_builder.load_soul_md", return_value=""),
        patch("agent.prompt_builder.build_environment_hints", return_value=""),
        patch("agent.prompt_builder.build_context_files_prompt", return_value=""),
    ):
        return build_system_prompt_parts(agent)


def _profile_line(agent) -> str:
    parts = _prompt_parts(agent)
    for chunk in parts["volatile"].split("\n\n"):
        if chunk.startswith("Active Hermes profile:"):
            return chunk
    raise AssertionError("profile line missing from volatile tier")


def _profile_env(tmp_path, monkeypatch, cfg_for_root):
    """A bound profile session (HERMES_HOME under profiles/); *cfg_for_root(root)*
    renders the config.yaml text once the root exists."""
    root = tmp_path / ".hermes"
    profile_home = root / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text(cfg_for_root(root), encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    from agent.skill_utils import _external_dirs_cache_clear
    _external_dirs_cache_clear()
    return root, profile_home


def test_create_dir_at_root_skills_is_named_shared_and_writable(tmp_path, monkeypatch):
    """The exact estate shape: profile session, skills.create_dir -> root skills."""
    root, _profile_home = _profile_env(
        tmp_path, monkeypatch,
        lambda root: f"skills:\n  create_dir: {root / 'skills'}\n",
    )
    (root / "skills").mkdir()  # a mount the resolution actually sees
    line = _profile_line(_make_agent(valid_tool_names=["read_file"]))
    shared = str((root / "skills").resolve())
    assert shared in line, line
    assert "SHARED" in line and "writable" in line, line
    # The isolation warning must survive for the NON-shared subtrees.
    assert "Do NOT modify another profile's" in line, line


def test_external_dirs_are_named_shared(tmp_path, monkeypatch):
    root, _profile_home = _profile_env(
        tmp_path, monkeypatch,
        lambda root: f"skills:\n  external_dirs:\n    - {root / 'team-skills'}\n",
    )
    (root / "team-skills").mkdir()
    line = _profile_line(_make_agent(valid_tool_names=["read_file"]))
    assert str((root / "team-skills").resolve()) in line, line
    assert "SHARED" in line, line


def test_missing_external_dir_is_not_promised(tmp_path, monkeypatch):
    """A configured-but-absent external dir stays out of the writable promise —
    the note mirrors get_external_skills_dirs (existing dirs only), not raw config."""
    _profile_env(
        tmp_path, monkeypatch,
        lambda root: f"skills:\n  external_dirs:\n    - {root / 'ghost'}\n",
    )
    line = _profile_line(_make_agent(valid_tool_names=["read_file"]))
    assert "SHARED" not in line, line


def test_no_shared_note_when_nothing_configured(tmp_path, monkeypatch):
    _profile_env(tmp_path, monkeypatch, lambda root: "model: foo\n")
    line = _profile_line(_make_agent(valid_tool_names=["read_file"]))
    assert "SHARED" not in line, line
    assert "Do NOT modify another profile's" in line, line


def test_local_skills_dir_never_double_listed_as_shared(tmp_path, monkeypatch):
    """create_dir equal to the profile's own skills/ resolves to unset
    (skill_utils contract) — no shared note for the session's own dir."""
    own_skills = tmp_path / ".hermes" / "profiles" / "coder" / "skills"
    _profile_env(
        tmp_path, monkeypatch,
        lambda root: f"skills:\n  create_dir: {own_skills}\n",
    )
    own_skills.mkdir(parents=True)
    line = _profile_line(_make_agent(valid_tool_names=["read_file"]))
    assert "SHARED" not in line, line  # not listed as a shared/writable mount


def test_shared_note_stays_out_of_the_stable_tier(tmp_path, monkeypatch):
    """Config-derived text rides the volatile tier with the rest of the line,
    so the cacheable stable prefix stays byte-identical across homes."""
    _profile_env(
        tmp_path, monkeypatch,
        lambda root: f"skills:\n  create_dir: {root / 'skills'}\n",
    )
    (tmp_path / ".hermes" / "skills").mkdir()
    parts = _prompt_parts(_make_agent(valid_tool_names=["read_file"]))
    assert "SHARED" in parts["volatile"]  # the note is actually present
    assert "SHARED" not in parts["stable"]
    assert "SHARED" not in parts.get("context", "")

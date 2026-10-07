"""est-2ek.1.381: skill discovery prunes backup/probe dirs STRUCTURALLY.

The walkers (agent.skill_utils.iter_skill_index_files and
agent.prompt_builder._build_skills_manifest) must never materialize a
directory that follows a prune convention (probes/, *.backup-*, .archive/,
.grave/, cut backups) as a skill — at any depth — regardless of what a
name-based denylist covers. EXCLUDED_SKILL_DIRS stays as the second line;
the structural rule is the first. Symlinked skill dirs keep working, but
following directory symlinks during the walk is now an explicit opt-in
(``skills.followlinks: true``), off by default.

Red reproduction for the estate incident: the owner cut 4 skills and parked
the backups under ``skills/probes/referrer-cut-backup/``; the name denylist
missed 'probes' and they nearly re-entered the system-prompt catalog.
"""

import pytest

from agent import skill_utils
from agent.skill_utils import is_excluded_skill_path, iter_skill_index_files


def _skill(dir_, name):
    dir_.mkdir(parents=True, exist_ok=True)
    path = dir_ / "SKILL.md"
    path.write_text(
        f"---\nname: {name}\ndescription: Test skill {name}.\n---\nBody.\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def home_skills(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    skills = home / "skills"
    skills.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    skill_utils._external_dirs_cache_clear()
    yield skills
    skill_utils._external_dirs_cache_clear()


def test_cut_backup_under_probes_is_not_discovered(home_skills):
    """The exact incident shape: skills/probes/referrer-cut-backup/<skill>/SKILL.md."""
    zombie = _skill(home_skills / "probes" / "referrer-cut-backup" / "pdf", "pdf")
    real = _skill(home_skills / "research" / "arxiv", "arxiv")

    found = list(iter_skill_index_files(home_skills, "SKILL.md"))

    assert zombie not in found
    assert found == [real]
    assert is_excluded_skill_path(zombie) is True


def test_prune_marker_dir_is_excluded_at_any_depth(home_skills):
    """A backup/cut marker on ANY component prunes it, wherever it sits."""
    helper = _skill(home_skills / "research" / "my-helper", "my-helper")
    backup = _skill(
        home_skills / "research" / "my-helper.backup-2026-09-28", "my-helper"
    )
    cut = _skill(home_skills / "creative" / "design" / "referrer-cut-backup" / "docx", "docx")

    found = list(iter_skill_index_files(home_skills, "SKILL.md"))

    assert found == [helper]
    assert backup not in found and cut not in found


def test_ordinary_deep_and_support_named_skills_still_discovered(home_skills):
    """Deep categories and support-named categories keep working (walk preserved)."""
    deep = _skill(home_skills / "research" / "nested" / "deep-skill", "deep-skill")
    scripts_cat = _skill(home_skills / "scripts" / "bash-helper", "bash-helper")
    # A category literally named 'backups' (plural, no marker boundary match on a
    # skill-name-shaped word) — only the prune CONVENTION words bite.
    found = list(iter_skill_index_files(home_skills, "SKILL.md"))
    assert deep in found
    assert scripts_cat in found


def test_followlinks_default_off_and_opt_in(home_skills, tmp_path):
    """A symlinked skill dir is skipped by default and followed only with
    ``skills.followlinks: true`` in config.yaml."""
    import os

    outside = tmp_path / "skill-vault" / "chained-skill"
    _skill(outside, "chained-skill")
    (home_skills / "research").mkdir()
    link = home_skills / "research" / "chained-skill"
    link.symlink_to(outside, target_is_directory=True)
    real = _skill(home_skills / "research" / "plain-skill", "plain-skill")

    assert skill_utils.skill_discovery_followlinks() is False
    found_default = list(iter_skill_index_files(home_skills, "SKILL.md"))
    assert found_default == [real]

    (tmp_path / ".hermes" / "config.yaml").write_text(
        "skills:\n  followlinks: true\n", encoding="utf-8"
    )
    skill_utils._external_dirs_cache_clear()
    assert skill_utils.skill_discovery_followlinks() is True
    found_opt_in = list(iter_skill_index_files(home_skills, "SKILL.md"))
    assert found_opt_in == sorted([real, outside / "SKILL.md"])


def test_manifest_and_prompt_exclude_pruned_skills(home_skills):
    """The disk-snapshot manifest and the rendered index must agree with the
    walker: pruned paths never enter the catalog under any path."""
    from agent import prompt_builder as pb

    zombie = _skill(home_skills / "probes" / "referrer-cut-backup" / "pdf", "zombie-skill")
    _skill(home_skills / "probes" / "claude-design.pre-fold-1790944314", "zombie-two")
    real = _skill(home_skills / "research" / "arxiv", "arxiv")

    pb.clear_skills_system_prompt_cache(clear_snapshot=True)
    try:
        manifest = pb._build_skills_manifest(home_skills)
        assert any(str(real).endswith(k) or k.endswith("research/arxiv/SKILL.md") for k in manifest)
        assert not any("probes" in k for k in manifest), manifest.keys()

        rendered = pb._build_skills_system_prompt_inner(home_skills, [], None, None, None)
        assert "arxiv" in rendered
        assert "zombie-skill" not in rendered and "zombie-two" not in rendered
    finally:
        pb.clear_skills_system_prompt_cache(clear_snapshot=True)


def test_snapshot_version_bumped_for_structural_manifest():
    """The manifest key-set changed shape; v3 snapshots must not be trusted."""
    from agent import prompt_builder as pb

    assert pb._SKILLS_SNAPSHOT_VERSION >= 4

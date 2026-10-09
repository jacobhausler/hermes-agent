"""est-2ek.1.309: skill discovery must not index git working checkouts.

``skills/haus-docs`` is a symlink to a git checkout.  Following it put the
checkout's UNCOMMITTED SKILL.md edits into every agent's skills index with no
review gate.  The walkers (agent.skill_utils.iter_skill_index_files and
agent.prompt_builder._build_skills_manifest) now stop at a walked directory
that contains ``.git`` (the skills root itself stays indexable) and visit each
real directory once, so several symlinks to one skill are not read repeatedly.
"""

import pytest

from agent import skill_utils
from agent.skill_utils import iter_skill_index_files


def _skill(dir_, name, body="Body."):
    dir_.mkdir(parents=True, exist_ok=True)
    path = dir_ / "SKILL.md"
    path.write_text(
        f"---\nname: {name}\ndescription: Test skill {name}.\n---\n{body}\n",
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


def _checkout(tmp_path, name="haus-docs", *, gitfile=False):
    repo = tmp_path / "checkouts" / name
    repo.mkdir(parents=True)
    if gitfile:
        (repo / ".git").write_text("gitdir: ../elsewhere/.git/worktrees/x\n", encoding="utf-8")
    else:
        (repo / ".git").mkdir()
    return repo


def test_symlinked_git_checkout_is_not_indexed(home_skills, tmp_path):
    """Acceptance shape: an uncommitted edit in the linked checkout never reaches the index."""
    repo = _checkout(tmp_path)
    _skill(repo, "haus-docs", body="UNCOMMITTED EDIT")
    (home_skills / "haus-docs").symlink_to(repo, target_is_directory=True)
    real = _skill(home_skills / "research" / "arxiv", "arxiv")

    assert list(iter_skill_index_files(home_skills, "SKILL.md")) == [real]


def test_git_worktree_gitfile_checkout_is_not_indexed(home_skills, tmp_path):
    repo = _checkout(tmp_path, gitfile=True)
    _skill(repo, "haus-docs")
    (home_skills / "haus-docs").symlink_to(repo, target_is_directory=True)

    assert list(iter_skill_index_files(home_skills, "SKILL.md")) == []


def test_real_subdir_with_git_is_not_indexed_and_nested_skills_are_skipped(home_skills):
    cloned = home_skills / "cloned"
    (cloned / ".git").mkdir(parents=True)
    _skill(cloned, "cloned")
    _skill(cloned / "sub" / "inner", "inner")

    assert list(iter_skill_index_files(home_skills, "SKILL.md")) == []


def test_reviewed_path_still_loads_while_checkout_is_skipped(home_skills, tmp_path):
    """The same skill stays available through a reviewed (non-checkout) path."""
    repo = _checkout(tmp_path)
    _skill(repo, "haus-docs", body="UNCOMMITTED EDIT")
    (home_skills / "haus-docs").symlink_to(repo, target_is_directory=True)
    reviewed = _skill(home_skills / "reviewed" / "haus-docs", "haus-docs", body="COMMITTED")

    assert list(iter_skill_index_files(home_skills, "SKILL.md")) == [reviewed]


def test_skills_root_with_git_stays_indexable(home_skills):
    """The skills house is itself a git repo: its root must not be pruned."""
    (home_skills / ".git").mkdir()
    real = _skill(home_skills / "research" / "arxiv", "arxiv")
    top = _skill(home_skills, "root-skill")

    assert list(iter_skill_index_files(home_skills, "SKILL.md")) == [top, real]


def test_symlink_into_git_repo_subdir_without_own_git_is_indexed(home_skills, tmp_path):
    """Profile skills dirs link to subdirs of the skills-house repo; those must keep working."""
    repo = _checkout(tmp_path, "house")
    target = _skill(repo / "software-development" / "spike", "spike")
    (home_skills / "spike").symlink_to(target.parent, target_is_directory=True)

    assert list(iter_skill_index_files(home_skills, "SKILL.md")) == [home_skills / "spike" / "SKILL.md"]


def test_duplicate_symlinks_to_one_directory_are_read_once(home_skills, tmp_path):
    approved = tmp_path / "approved"
    _skill(approved, "approved")
    (home_skills / "alias-b").symlink_to(approved, target_is_directory=True)
    (home_skills / "alias-a").symlink_to(approved, target_is_directory=True)

    assert list(iter_skill_index_files(home_skills, "SKILL.md")) == [home_skills / "alias-a" / "SKILL.md"]


def test_same_leaf_name_in_distinct_directories_is_kept(home_skills):
    first = _skill(home_skills / "first" / "same", "same")
    second = _skill(home_skills / "second" / "same", "same")

    assert list(iter_skill_index_files(home_skills, "SKILL.md")) == [first, second]


def test_manifest_and_prompt_skip_git_checkout(home_skills, tmp_path):
    """Snapshot manifest and rendered index agree with the walker."""
    from agent import prompt_builder as pb

    repo = _checkout(tmp_path)
    _skill(repo, "haus-docs", body="UNCOMMITTED EDIT")
    (home_skills / "haus-docs").symlink_to(repo, target_is_directory=True)
    _skill(home_skills / "research" / "arxiv", "arxiv")

    pb.clear_skills_system_prompt_cache(clear_snapshot=True)
    try:
        manifest = pb._build_skills_manifest(home_skills)
        assert "research/arxiv/SKILL.md" in manifest
        assert not any(k.startswith("haus-docs") for k in manifest), manifest.keys()

        rendered = pb._build_skills_system_prompt_inner(home_skills, [], None, None, None)
        assert "arxiv" in rendered
        assert "haus-docs" not in rendered
    finally:
        pb.clear_skills_system_prompt_cache(clear_snapshot=True)

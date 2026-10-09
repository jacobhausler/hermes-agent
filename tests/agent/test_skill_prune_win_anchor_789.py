"""Windows path-flavor root-anchor judgment for the prune convention.

The prune-convention rule judges only components BELOW the declared root, for
absolute paths of ANY flavor — POSIX, drive-anchored, or UNC — while relative
callers keep whole-path judgment. These Linux executions drive the actual
public filter with its path-parser constructor substituted by stdlib
PureWindowsPath (the established qualification for Windows-flavor probes);
they are not a native Windows OS run. est-2ek.1.789.
"""
from pathlib import PurePosixPath, PureWindowsPath

import pytest

from agent import skill_utils as su


@pytest.fixture
def windows_parser(monkeypatch):
    """Substitute the module's pure-path constructor with PureWindowsPath."""
    monkeypatch.setattr(su, "PurePath", PureWindowsPath)


def test_windows_absolute_legit_under_prune_root_not_excluded(windows_parser):
    # A legitimate skill of a root that itself sits under a prune-token
    # ancestor keeps its tail-only judgment (drive anchor is absoluteness).
    assert su.is_prune_convention_path(
        r"C:\backups\external-corpus\research\legitimate\SKILL.md",
        root=PureWindowsPath(r"C:\backups\external-corpus"),
    ) is False


def test_windows_relative_caller_judged_whole(windows_parser):
    assert su.is_prune_convention_path(
        r"research\legitimate\SKILL.md",
        root=PureWindowsPath(r"C:\backups\external-corpus"),
    ) is False


def test_windows_prune_below_root_still_excluded(windows_parser):
    assert su.is_prune_convention_path(
        r"C:\backups\external-corpus\probes\evil\SKILL.md",
        root=PureWindowsPath(r"C:\backups\external-corpus"),
    ) is True


def test_windows_drive_root_prune_below_excluded(windows_parser):
    assert su.is_prune_convention_path(
        r"C:\skills\x\backup-2026\SKILL.md",
        root=PureWindowsPath("C:\\"),
    ) is True


def test_windows_drive_root_clean_not_excluded(windows_parser):
    assert su.is_prune_convention_path(
        r"C:\skills\x\research\SKILL.md",
        root=PureWindowsPath("C:\\"),
    ) is False


def test_windows_unc_legit_under_prune_share_not_excluded(windows_parser):
    assert su.is_prune_convention_path(
        r"\\server\share\backups\corpus\research\legitimate\SKILL.md",
        root=PureWindowsPath(r"\\server\share\backups\corpus"),
    ) is False


def test_windows_unc_prune_below_root_excluded(windows_parser):
    assert su.is_prune_convention_path(
        r"\\server\share\backups\corpus\quarantine\evil\SKILL.md",
        root=PureWindowsPath(r"\\server\share\backups\corpus"),
    ) is True


def test_windows_different_drive_judged_whole(windows_parser):
    # Not under the declared root (other drive): judged whole, so the prune
    # token in the tail excludes.
    assert su.is_prune_convention_path(
        r"D:\work\probes\evil\SKILL.md",
        root=PureWindowsPath(r"C:\backups\external-corpus"),
    ) is True


def test_posix_absolute_legit_under_prune_root_not_excluded():
    assert su.is_prune_convention_path(
        "/home/u/backups/external-corpus/research/legitimate/SKILL.md",
        root=PurePosixPath("/home/u/backups/external-corpus"),
    ) is False


def test_posix_absolute_prune_below_root_excluded():
    assert su.is_prune_convention_path(
        "/root/skills/x/probes/evil/SKILL.md",
        root=PurePosixPath("/root/skills"),
    ) is True


def test_no_root_skills_anchor_judges_tail_only():
    assert su.is_prune_convention_path(
        "/home/u/quarantine-state/x/skills/good/SKILL.md") is False
    assert su.is_prune_convention_path(
        "/home/u/x/skills/quarantine-state/evil/SKILL.md") is True

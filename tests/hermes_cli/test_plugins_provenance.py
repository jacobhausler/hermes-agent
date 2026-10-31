"""Tests: plugin provenance reconciliation — the 2x2.

Row-presence says "hermes installed this"; `.git`-presence cross-checks
it. The reconciliation IS the disambiguation (settled 2026-09-03,
plugin-auto-update plan Task 1):
  row+git   = git install         (updatable via the recorded source)
  neither   = manual drop         (not auto-updatable, listed as such)
  git only  = self-cloned         (adoptable: origin URL ready)
  row only  = provenance drift    (flagged with a reinstall remedy)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli.plugins_provenance import (
    Provenance,
    ProvenanceClass,
    plugins_provenance,
)


def _write_sidecar(home: Path, rows: dict) -> None:
    plugins = home / "plugins"
    plugins.mkdir(parents=True, exist_ok=True)
    (plugins / ".install-metadata.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )


def _fake_git_dir(plugin_dir: Path, origin_url: str = "https://example/o/r") -> None:
    plugin_dir.mkdir(parents=True, exist_ok=True)
    git = plugin_dir / ".git"
    git.mkdir()
    # a real-enough git dir: remote config the adopt path reads
    (git / "config").write_text(
        f'[remote "origin"]\n\turl = {origin_url}\n', encoding="utf-8"
    )


def _manual_dir(plugin_dir: Path) -> None:
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.yaml").write_text("name: x\n", encoding="utf-8")


def test_git_install_row_and_git(tmp_path):
    _write_sidecar(tmp_path, {"plug": {"pinned": False, "revision": "a" * 40, "source": "https://example/o/r"}})
    _fake_git_dir(tmp_path / "plugins" / "plug")
    prov = plugins_provenance(tmp_path / "plugins")
    assert len(prov) == 1
    assert prov[0].klass is ProvenanceClass.GIT
    assert prov[0].row is not None and prov[0].row["revision"] == "a" * 40


def test_manual_drop_neither(tmp_path):
    _write_sidecar(tmp_path, {})
    _manual_dir(tmp_path / "plugins" / "handmade")
    prov = plugins_provenance(tmp_path / "plugins")
    assert prov[0].klass is ProvenanceClass.MANUAL


def test_self_cloned_git_without_row(tmp_path):
    _write_sidecar(tmp_path, {})
    _fake_git_dir(tmp_path / "plugins" / "selfcloned", "https://example/m/r")
    prov = plugins_provenance(tmp_path / "plugins")
    assert prov[0].klass is ProvenanceClass.SELF_CLONED
    assert prov[0].origin_url == "https://example/m/r"


def test_drift_row_without_git(tmp_path):
    _write_sidecar(tmp_path, {"ghost": {"pinned": False, "revision": "b" * 40, "source": "https://example/g/r"}})
    _manual_dir(tmp_path / "plugins" / "ghost")
    prov = plugins_provenance(tmp_path / "plugins")
    assert prov[0].klass is ProvenanceClass.DRIFT
    assert prov[0].row["source"] == "https://example/g/r"


def test_missing_sidecar_is_all_manual_or_selfcloned(tmp_path):
    # no sidecar at all: everything classifies from the dir alone
    _fake_git_dir(tmp_path / "plugins" / "cloned")
    _manual_dir(tmp_path / "plugins" / "dropped")
    prov = plugins_provenance(tmp_path / "plugins")
    classes = {p.name: p.klass for p in prov}
    assert classes["cloned"] is ProvenanceClass.SELF_CLONED
    assert classes["dropped"] is ProvenanceClass.MANUAL


def test_update_url_tag_round_trips(tmp_path):
    # a saved update_url tag rides the row; the manifest is claims
    _write_sidecar(
        tmp_path,
        {"plug": {"pinned": False, "revision": "c" * 40, "source": "https://example/o/r",
                  "update_url": "https://example/feed.yml"}},
    )
    _fake_git_dir(tmp_path / "plugins" / "plug")
    prov = plugins_provenance(tmp_path / "plugins")
    assert prov[0].saved_update_url == "https://example/feed.yml"

def test_origin_url_uses_the_resolved_git_and_survives_no_git_at_all(tmp_path, monkeypatch):
    """Provenance runs from the gateway service, where PATH can be minimal: the probe must use the
    same resolved git as install/update, and with no git anywhere the .git/config parse still
    yields the origin (never a bare ``git`` spawn that raises or picks up a stray binary)."""
    import subprocess
    from hermes_cli import plugins_cmd, plugins_provenance

    plugin = tmp_path / "plugins" / "p"
    plugin.mkdir(parents=True)
    _fake_git_dir(plugin, "https://example/from-config")
    spawned = []
    real_run = subprocess.run

    def record(argv, *a, **k):
        spawned.append(argv[0])
        return real_run(argv, *a, **k)

    monkeypatch.setattr(subprocess, "run", record)
    monkeypatch.setattr(plugins_cmd, "_resolve_git_executable", lambda: None)
    assert plugins_provenance._git_origin_url(plugin) == "https://example/from-config"
    assert spawned == []

    monkeypatch.setattr(plugins_cmd, "_resolve_git_executable", lambda: "/nonexistent/resolved-git")
    assert plugins_provenance._git_origin_url(plugin) == "https://example/from-config"
    assert spawned == ["/nonexistent/resolved-git"]


# ── artifact-pinned cell (est-raqgj): row + install.json, no .git ────
#
# The pack/atomic-installer lane installs signed archives: the dir has
# no .git and install.json is the provenance record. Both sidecar
# shapes the lane ships are fixtures here: the richer archive shape
# (the live 1.4.3 install) and the 3-key contract
# {version, source_commit, packaged_at} (the frozen 1.4.4 archive).

_ARCHIVE_SHAPE = {
    "package": "hermes-workflows",
    "version": "1.4.3",
    "source_repo": "https://github.com/jacobhausler/hermes-workflows.git",
    "source_commit": "911ab3f",
    "merged_sha": "32cef07530a345db05a36e43f370bb5ffda1e3f6",
    "artifact": "hermes-workflows-1.4.3.zip",
    "artifact_sha256": "1c226a83d3224522d59df1d041c2f37b673f381a95d29aea2215596106987203",
    "method": "INSTALL.md pack.py flow (ship lane wf-qa-remediation)",
    "installed_at": "2026-10-28T03:20:32Z",
}
_CONTRACT_SHAPE = {
    "version": "1.4.4",
    "source_commit": "f6cf76afc87051848feddd3d868ee1812964486a",
    "packaged_at": "2026-10-30T12:18:33-05:00",
}
_ARTIFACT_ROW = {
    "pinned": True,
    "revision": "9a6f8ac9963f38964f91c772a5f85e32259738ea",
    "source": "https://github.com/jacobhausler/hermes-workflows.git",
}


def _artifact_dir(plugin_dir: Path, install_json: dict) -> None:
    """An archive-installed plugin dir: manifest + install.json, no .git."""
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text("name: x\n", encoding="utf-8")
    (plugin_dir / "install.json").write_text(
        json.dumps(install_json), encoding="utf-8"
    )


@pytest.mark.parametrize(
    "install_json", [_ARCHIVE_SHAPE, _CONTRACT_SHAPE],
    ids=["archive-shape", "3key-contract"],
)
def test_artifact_pinned_row_and_install_json_proven(tmp_path, install_json):
    _write_sidecar(tmp_path, {"hermes-workflows": dict(_ARTIFACT_ROW)})
    _artifact_dir(tmp_path / "plugins" / "hermes-workflows", install_json)
    prov = plugins_provenance(tmp_path / "plugins")
    assert len(prov) == 1
    assert prov[0].klass is ProvenanceClass.ARTIFACT
    # the receipt-facing verdict is "proven", not "drift"
    assert prov[0].klass.value == "proven"
    assert prov[0].install_json == install_json
    assert prov[0].row is not None and prov[0].row["source"] == _ARTIFACT_ROW["source"]


def test_install_json_without_row_is_manual(tmp_path):
    # the cell needs the sidecar row; an unrecorded dir stays a manual drop
    _write_sidecar(tmp_path, {})
    _artifact_dir(tmp_path / "plugins" / "dropped", _CONTRACT_SHAPE)
    prov = plugins_provenance(tmp_path / "plugins")
    assert prov[0].klass is ProvenanceClass.MANUAL
    assert prov[0].install_json is None


def test_git_wins_over_install_json(tmp_path):
    # row + .git is a git install even if an install.json rode the archive
    _write_sidecar(tmp_path, {"hermes-workflows": dict(_ARTIFACT_ROW)})
    _fake_git_dir(tmp_path / "plugins" / "hermes-workflows")
    (tmp_path / "plugins" / "hermes-workflows" / "install.json").write_text(
        json.dumps(_CONTRACT_SHAPE), encoding="utf-8"
    )
    prov = plugins_provenance(tmp_path / "plugins")
    assert prov[0].klass is ProvenanceClass.GIT


@pytest.mark.parametrize(
    "broken",
    [
        pytest.param("not json at all {", id="unparseable"),
        pytest.param('{"version": "1.0.0"}', id="missing-source_commit"),
        pytest.param('{"source_commit": "' + "a" * 40 + '"}', id="missing-version"),
        pytest.param('{"version": "", "source_commit": "' + "a" * 40 + '"}', id="empty-version"),
    ],
)
def test_broken_install_json_keeps_the_row_drifted(tmp_path, broken):
    # an artifact record must actually record something
    _write_sidecar(tmp_path, {"ghost": dict(_ARTIFACT_ROW)})
    plugin = tmp_path / "plugins" / "ghost"
    plugin.mkdir(parents=True)
    (plugin / "install.json").write_text(broken, encoding="utf-8")
    prov = plugins_provenance(tmp_path / "plugins")
    assert prov[0].klass is ProvenanceClass.DRIFT

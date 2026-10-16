"""Per-child write grant survives a served profile's own HERMES_WRITE_SAFE_ROOT (est-2mhl).

``HERMES_WRITE_SAFE_ROOT`` is a per-profile ceiling: under multiplex (or any routed
profile) the bound scope's value REPLACES the process environment's. A launcher that
appends a child's workdir to that variable therefore loses the grant the moment the
served profile carries its own root. ``HERMES_WRITE_SAFE_ROOT_EXTRA`` is the launcher's
channel: process environment only, absolute directories only, unioned while a jail is
in force, never able to relax a credential denial or to create a jail by itself.
"""

import os
from pathlib import Path

import pytest

from agent import secret_scope as ss
from agent.file_safety import get_safe_write_roots, is_write_denied

GRANT = "HERMES_WRITE_SAFE_ROOT_EXTRA"
ROOT = "HERMES_WRITE_SAFE_ROOT"


def _real(p: Path) -> str:
    return os.path.realpath(str(p))


@pytest.fixture
def layout(tmp_path: Path, monkeypatch):
    base = tmp_path / "profile-base"
    child = tmp_path / "run" / "work" / "child"
    sibling = tmp_path / "run" / "work" / "sibling"
    home = tmp_path / "profile-home"
    for d in (base, child, sibling, home):
        d.mkdir(parents=True)
    monkeypatch.delenv(ROOT, raising=False)
    monkeypatch.delenv(GRANT, raising=False)
    return base, child, sibling, home


def _multiplex(scope: dict, home: Path):
    t_mux = ss.set_multiplex_context(True)
    t_scope = ss.set_secret_scope(scope, profile_home=str(home))
    return t_mux, t_scope


def _release(tokens):
    t_mux, t_scope = tokens
    ss.reset_secret_scope(t_scope)
    ss.reset_multiplex_context(t_mux)


def test_standalone_launcher_grant_is_honoured(layout, monkeypatch):
    base, child, sibling, _home = layout
    monkeypatch.setenv(ROOT, str(base))
    monkeypatch.setenv(GRANT, str(child))
    roots = get_safe_write_roots()
    assert _real(base) in roots and _real(child) in roots
    assert _real(sibling) not in roots
    assert is_write_denied(str(child / "x.txt")) is False
    assert is_write_denied(str(sibling / "x.txt")) is True


def test_multiplex_scope_root_replaces_process_root_but_not_the_grant(layout, monkeypatch):
    """The est-2mhl defect: bound scope carries its own root, the child workdir is not in it."""
    base, child, sibling, home = layout
    monkeypatch.setenv(ROOT, str(child))  # the old append-to-root grant: dropped by the scope
    monkeypatch.setenv(GRANT, str(child))  # the launcher grant: must survive
    tokens = _multiplex({ROOT: str(base)}, home)
    try:
        roots = get_safe_write_roots()
        assert _real(base) in roots, "the served profile's own ceiling is kept"
        assert _real(child) in roots, "the per-child grant survives the replacing scope"
        assert _real(sibling) not in roots, "a sibling workdir is NOT granted"
        assert is_write_denied(str(child / "x.txt")) is False
        assert is_write_denied(str(sibling / "x.txt")) is True
    finally:
        _release(tokens)


def test_old_root_append_alone_is_still_dropped_under_multiplex(layout, monkeypatch):
    """Profile isolation unchanged: without the grant var the replace semantics are intact."""
    base, child, _sibling, home = layout
    monkeypatch.setenv(ROOT, str(child))
    tokens = _multiplex({ROOT: str(base)}, home)
    try:
        roots = get_safe_write_roots()
        assert roots == {_real(base)}
    finally:
        _release(tokens)


def test_grant_never_creates_a_jail(layout, monkeypatch):
    """No root anywhere = guard inert; a grant must not turn that into a jail on the grant."""
    _base, child, sibling, _home = layout
    monkeypatch.setenv(GRANT, str(child))
    assert get_safe_write_roots() == set()
    assert is_write_denied(str(sibling / "x.txt")) is False


def test_grant_never_creates_a_jail_under_multiplex(layout, monkeypatch):
    _base, child, sibling, home = layout
    monkeypatch.setenv(GRANT, str(child))
    tokens = _multiplex({}, home)
    try:
        assert get_safe_write_roots() == set()
        assert is_write_denied(str(sibling / "x.txt")) is False
    finally:
        _release(tokens)


def test_grant_is_process_env_only_never_read_from_a_profile_scope(layout, monkeypatch):
    """A profile .env / scope cannot grant itself extra roots through the companion."""
    base, child, _sibling, home = layout
    tokens = _multiplex({ROOT: str(base), GRANT: str(child)}, home)
    try:
        assert get_safe_write_roots() == {_real(base)}
    finally:
        _release(tokens)


def test_relative_and_filesystem_root_entries_are_ignored(layout, monkeypatch):
    base, child, _sibling, _home = layout
    monkeypatch.setenv(ROOT, str(base))
    monkeypatch.setenv(GRANT, os.pathsep.join(["relative/dir", "~", os.sep, str(child)]))
    roots = get_safe_write_roots()
    assert roots == {_real(base), _real(child)}


def test_grant_cannot_relax_a_credential_denial(layout, tmp_path, monkeypatch):
    base, _child, _sibling, _home = layout
    fake_home = tmp_path / "fake-home"
    (fake_home / ".ssh").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv(ROOT, str(base))
    monkeypatch.setenv(GRANT, str(fake_home / ".ssh"))
    assert is_write_denied(str(fake_home / ".ssh" / "id_rsa")) is True

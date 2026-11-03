"""Tests for the symlink-safe, offenders-only Docker stage2 ownership repair.

The pass must repair exactly the mis-owned entries: one read-only find
traversal whose offenders are chowned via `-exec chown ... +`, never a
`chown -R` over the whole tree. On multi-million-entry profiles/ trees
(HDD/sshfs-backed data volumes) the old early-exit pre-scan + `chown -R`
rewrote every inode for a handful of `docker exec`-root files — a ~20
minute restart tax on every container boot. Symlinks inside the tree are
never passed to chown: the link and its referent (possibly outside the
target, possibly dangling) keep their owners — the bind-mount contract of
issue #19788.

The helpers are extracted verbatim from docker/stage2-hook.sh and executed
under /bin/sh with a tmp PATH-shimmed bin dir (fake `find`, fake `chown`)
so no test ever chowns or stats real system paths, and every traversal is
recorded verbatim. One real-semantics test runs when the suite executes
as root.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE2_HOOK = REPO_ROOT / "docker" / "stage2-hook.sh"

@pytest.fixture(scope="module")
def stage2_text() -> str:
    if not STAGE2_HOOK.exists():
        pytest.skip("docker/stage2-hook.sh not present in this checkout")
    return STAGE2_HOOK.read_text()

def _ownership_region(text: str) -> str:
    """path_has_symlink_component / refuse_symlinked_path / chown_hermes_tree."""
    start = text.index("path_has_symlink_component() {")
    end = text.index("\n\nneeds_chown=false", start)
    return text[start:end]

def _strip_comments(source: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in source.splitlines())

CHOWN_SHIM = """#!/bin/sh
if [ -n "$FAKE_CHOWN_LOG" ]; then
    printf chown >> "$FAKE_CHOWN_LOG"
    for a in "$@"; do printf ' %s' "$a" >> "$FAKE_CHOWN_LOG"; done
    printf ' @%s\\n' "$PWD" >> "$FAKE_CHOWN_LOG"
fi
exit ${FAKE_CHOWN_RC:-0}
"""

# Records every invocation (blank-line-separated argv), then either hands
# the call to the real find (mode=passthrough) or, in report mode, replays
# the -execdir batch over a canned offender list the way GNU find does:
# from inside each offender's parent directory, naming it `./basename`.
FIND_SHIM = """#!/bin/sh
printf '\\n' >> "$FAKE_FIND_LOG"
for a in "$@"; do printf '%s\\n' "$a" >> "$FAKE_FIND_LOG"; done
if [ "$FAKE_FIND_MODE" = passthrough ]; then
    exec "$FAKE_FIND_REAL" "$@"
fi
state=pre; cmd=""; preargs=""
for a in "$@"; do
    if [ "$state" = pre ] && [ "$a" = "-execdir" ]; then state=cmd; continue; fi
    if [ "$state" = cmd ]; then
        if [ "$a" = "{}" ]; then state=done; continue; fi
        if [ -z "$cmd" ]; then cmd=$a; else preargs="$preargs $a"; fi
    fi
done
rc=${FAKE_FIND_RC:-0}
if [ "$state" = done ] && [ -n "$FAKE_FIND_NAMES" ] && [ -f "$FAKE_FIND_NAMES" ]; then
    while IFS= read -r n; do
        [ -n "$n" ] || continue
        # word-split intentional: canned names carry no spaces.
        (cd "$(dirname "$n")" && $cmd $preargs "./$(basename "$n")") || rc=${FAKE_FAIL_RC:-1}
    done < "$FAKE_FIND_NAMES"
fi
exit "$rc"
"""

class Pass:
    """Runs the extracted ownership helpers against a tmp tree."""

    def __init__(
        self,
        text: str,
        target: Path,
        tmp: Path,
        *,
        offenders: list[Path] | None = None,
        passthrough: bool = False,
        chown_rc: str = "0",
        find_rc: str = "0",
        fail_rc: str = "1",
        hermes_home: Path | None = None,
        owner: tuple[str, str] | None = None,
        real_chown: bool = False,
    ) -> None:
        shell = shutil.which("sh")
        if shell is None:
            pytest.skip("sh not available")
        real_find = _system_find()
        if passthrough and real_find is None:
            pytest.skip("GNU find required")
        bindir = tmp / "bin"
        bindir.mkdir(exist_ok=True)
        shims = [("find", FIND_SHIM)] + ([] if real_chown else [("chown", CHOWN_SHIM)])
        for name, body in shims:
            shim = bindir / name
            shim.write_text(body)
            shim.chmod(0o755)
        self.find_log = tmp / "find.log"
        self.chown_log = tmp / "chown.log"
        names = tmp / "offenders.txt"
        if offenders is not None:
            names.write_text("".join(f"{p}\n" for p in offenders))
        region = _ownership_region(text)
        if owner is not None:
            # Real-find runs are judged against the invoking user so they
            # hold on any CI runner, not only where the user is `hermes`.
            user, group = owner
            for literal in ("hermes:hermes", "-user hermes", "-group hermes"):
                assert literal in region, f"harness: {literal!r} not found in region"
            region = (
                region.replace("hermes:hermes", f"{user}:{group}")
                .replace("-user hermes", f"-user {user}")
                .replace("-group hermes", f"-group {group}")
            )
        self.script = tmp / "region.sh"
        self.script.write_text(
            "set -eu\n"
            f'HERMES_HOME="{hermes_home or target}"\n'
            f"{region}\n"
            f'chown_hermes_tree "{target}"\n'
        )
        self.shell = shell
        self.env = {
            "PATH": f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}",
            "FAKE_FIND_LOG": str(self.find_log),
            "FAKE_CHOWN_LOG": str(self.chown_log),
            "FAKE_FIND_MODE": "passthrough" if passthrough else "report",
            "FAKE_FIND_REAL": real_find or "",
            "FAKE_FIND_NAMES": str(names),
            "FAKE_CHOWN_RC": chown_rc,
            "FAKE_FIND_RC": find_rc,
            "FAKE_FAIL_RC": fail_rc,
        }
        self.find_log.write_text("")

    def run(self) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env.update(self.env)
        return subprocess.run(
            [self.shell, str(self.script)], env=env,
            capture_output=True, text=True, check=False, timeout=60,
        )

    def find_invocations(self) -> list[list[str]]:
        if not self.find_log.exists():
            return []
        return [
            [line for line in block.splitlines() if line]
            for block in self.find_log.read_text().split("\n\n")
            if block.strip()
        ]

    def chown_calls(self) -> list[str]:
        if not self.chown_log.exists():
            return []
        return [line for line in self.chown_log.read_text().splitlines() if line]

def _system_find() -> str | None:
    """Absolute GNU find (fixed system locations; never walks PATH)."""
    for candidate in ("/usr/bin/find", "/bin/find", "/usr/local/bin/find"):
        if not os.path.isfile(candidate):
            continue
        try:
            out = subprocess.run(
                [candidate, "--version"],
                capture_output=True, text=True, timeout=10,
            ).stdout
        except OSError:
            continue
        if "GNU find" in out:
            return candidate
    return None

def _seed_tree(root: Path) -> None:
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "healthy.txt").write_text("x")
    (root / "healthy.txt").write_text("x")

def _invoking_owner() -> tuple[str, str]:
    import grp
    import pwd

    return pwd.getpwuid(os.getuid()).pw_name, grp.getgrgid(os.getgid()).gr_name

def _contains(hay: list[str], needle: list[str]) -> bool:
    return any(
        hay[i : i + len(needle)] == needle
        for i in range(len(hay) - len(needle) + 1)
    )


# --- contract: one traversal, offenders-only --------------------------------

def test_dirty_tree_chowns_only_the_reported_offenders(
    stage2_text: str, tmp_path: Path
) -> None:
    target = tmp_path / "profiles"
    _seed_tree(target)
    (target / "rootdir").mkdir()
    offenders = [target / "sub" / "rootfile", target / "sub", target / "rootdir"]
    p = Pass(stage2_text, target, tmp_path, offenders=offenders)

    proc = p.run()

    assert proc.returncode == 0, proc.stderr
    invs = p.find_invocations()
    assert len(invs) == 1, f"exactly one traversal (no separate pre-scan), got {invs}"
    argv = invs[0]
    assert argv[0] == str(target)
    assert _contains(argv, ["(", "!", "-user", "hermes", "-o", "!", "-group", "hermes", ")"])
    # Race-safe form: chown runs from the directory find opened (-execdir)
    # and never dereferences the final component (-h).
    assert _contains(argv, ["-execdir", "chown", "-h", "hermes:hermes", "{}", "+"])
    assert "-exec" not in argv, "plain -exec re-resolves full paths (symlink race)"
    assert "-R" not in argv
    assert p.chown_calls() == [
        f"chown -h hermes:hermes ./rootfile @{target / 'sub'}",
        f"chown -h hermes:hermes ./sub @{target}",
        f"chown -h hermes:hermes ./rootdir @{target}",
    ], p.chown_calls()
    # The healthy tree root is never chowned just because a file inside it is.
    assert not any(c.startswith(f"chown -h hermes:hermes ./profiles @") for c in p.chown_calls())

def test_healthy_tree_is_a_single_read_only_traversal(
    stage2_text: str, tmp_path: Path
) -> None:
    # Real find over a fully hermes-owned tmp tree: the pass must walk once,
    # call chown zero times, and stay silent.
    target = tmp_path / "home"
    _seed_tree(target)
    p = Pass(stage2_text, target, tmp_path, passthrough=True, owner=_invoking_owner())

    proc = p.run()

    assert proc.returncode == 0, proc.stderr
    assert len(p.find_invocations()) == 1
    assert p.chown_calls() == []
    assert "Warning" not in proc.stdout

def test_batched_chown_failure_warns_and_boot_survives(
    stage2_text: str, tmp_path: Path
) -> None:
    target = tmp_path / "cron"
    _seed_tree(target)
    p = Pass(
        stage2_text, target, tmp_path,
        offenders=[target / "x.txt"], chown_rc="1", fail_rc="1",
    )

    proc = p.run()

    assert proc.returncode == 0, "chown failure MUST NOT fail the boot"
    assert f"Warning: chown {target} failed" in proc.stdout

def test_find_failure_warns_and_boot_survives(
    stage2_text: str, tmp_path: Path
) -> None:
    target = tmp_path / "logs"
    _seed_tree(target)
    p = Pass(stage2_text, target, tmp_path, offenders=[], find_rc="1")

    proc = p.run()

    assert proc.returncode == 0
    assert f"Warning: chown {target} failed" in proc.stdout

def test_relative_path_entry_fails_closed_without_chowning(
    stage2_text: str, tmp_path: Path
) -> None:
    # Real GNU find refuses -execdir when PATH carries a relative entry (a
    # chown planted in the cwd would otherwise run as root). The pass must
    # fail closed: warning, exit 0, chown never invoked.
    target = tmp_path / "profiles"
    _seed_tree(target)
    p = Pass(stage2_text, target, tmp_path, passthrough=True, owner=_invoking_owner())
    p.env["PATH"] = p.env["PATH"].replace(os.pathsep, f"{os.pathsep}.{os.pathsep}", 1)

    proc = p.run()

    assert proc.returncode == 0, proc.stderr
    assert f"Warning: chown {target} failed" in proc.stdout
    assert p.chown_calls() == []


# --- #19788: nothing is ever chowned through a symlink ----------------------

def test_chown_helper_refuses_symlinked_directories(
    stage2_text: str, tmp_path: Path
) -> None:
    real_home = tmp_path / "real-home"
    real_home.mkdir()
    symlinked_home = tmp_path / "hermes-home"
    try:
        symlinked_home.symlink_to(real_home, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("directory symlinks are not available on this platform")
    p = Pass(
        stage2_text, symlinked_home, tmp_path,
        passthrough=True, owner=_invoking_owner(),
    )

    proc = p.run()

    assert proc.returncode == 0, proc.stderr
    assert p.chown_calls() == []
    assert len(p.find_invocations()) == 0, "a refused target must not even scan"
    assert "refusing recursive chown through symlinked path" in proc.stdout

def test_chown_helper_refuses_target_under_symlinked_home(
    stage2_text: str, tmp_path: Path,
) -> None:
    real_home = tmp_path / "real-home"
    (real_home / "cron").mkdir(parents=True)
    linked_home = tmp_path / "linked-home"
    try:
        linked_home.symlink_to(real_home, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("directory symlinks are not available on this platform")
    p = Pass(
        stage2_text, linked_home / "cron", tmp_path,
        passthrough=True, hermes_home=linked_home, owner=_invoking_owner(),
    )

    proc = p.run()

    assert proc.returncode == 0, proc.stderr
    assert p.chown_calls() == [], "must not chown through a symlinked HERMES_HOME"
    assert len(p.find_invocations()) == 0
    assert "refusing recursive chown through symlinked path" in proc.stdout

@pytest.mark.skipif(os.geteuid() != 0, reason="root creates the mis-owned tree")
def test_real_root_owned_tree_repairs_offenders_only(
    stage2_text: str, tmp_path: Path
) -> None:
    """End-to-end with real find + real chown: repairs mis-owned entries and
    leaves healthy entries, symlinks, and out-of-tree referents untouched."""
    import pwd

    try:
        hermes = pwd.getpwnam("hermes")
    except KeyError:
        pytest.skip("no hermes user on this host")
    target = tmp_path / "profiles"
    _seed_tree(target)
    outside = tmp_path / "outside-referent"
    outside.write_text("host")
    os.chown(outside, 0, 0)
    dirty_file = target / "sub" / "rootfile"
    dirty_file.write_text("x")
    os.chown(dirty_file, 0, 0)
    os.chown(target / "sub", 0, 0)
    (target / "linkout").symlink_to(outside)
    (target / "linkdangling").symlink_to(tmp_path / "no-such-goal")
    os.chown(target, hermes.pw_uid, hermes.pw_gid)
    os.chown(target / "healthy.txt", hermes.pw_uid, hermes.pw_gid)
    healthy_ctime = (target / "healthy.txt").stat().st_ctime_ns

    p = Pass(stage2_text, target, tmp_path, passthrough=True, real_chown=True)
    proc = p.run()

    assert proc.returncode == 0, proc.stderr
    assert "Warning" not in proc.stdout, proc.stdout
    for path in (dirty_file, target / "sub"):
        st = path.stat()
        assert (st.st_uid, st.st_gid) == (hermes.pw_uid, hermes.pw_gid), path
    # The healthy entry was never passed to chown: ctime unchanged.
    assert (target / "healthy.txt").stat().st_ctime_ns == healthy_ctime
    # Nothing was chowned through the symlink: the root-owned referent is
    # still root's while the (root-created) link itself was re-owned via
    # `chown -h` — physical `chown -R` parity. The dangling link neither
    # failed the pass (no Warning above) nor was dereferenced.
    assert (outside.stat().st_uid, outside.stat().st_gid) == (0, 0)
    assert (target / "linkout").lstat().st_uid == hermes.pw_uid
    assert os.path.islink(target / "linkdangling")
    assert (target / "linkdangling").lstat().st_uid == hermes.pw_uid


# --- static structure: gates gone, anchors stable ---------------------------

def test_pre_scan_helper_is_gone_and_call_sites_ungated(stage2_text: str) -> None:
    assert "tree_has_non_hermes_owner" not in stage2_text
    for sub in ("profiles", "cron", "platforms/pairing", "pairing"):
        assert f'if [ -d "$HERMES_HOME/{sub}" ]; then' in stage2_text, sub
    assert "&& tree_has" not in stage2_text

def test_region_has_no_recursive_chown_invocation(stage2_text: str) -> None:
    code = _strip_comments(_ownership_region(stage2_text))
    assert "chown -R" not in code
    assert "-execdir chown -h hermes:hermes {} +" in code
    assert "-exec chown" not in code

def test_sibling_test_anchor_still_present(stage2_text: str) -> None:
    # tests/tools/test_stage2_hook_{api_server_keygen,nous_routing_env,
    # seed_one_symlinks}.py slice their regions on this anchor.
    assert "\n\nchown_hermes_tree() {" in stage2_text

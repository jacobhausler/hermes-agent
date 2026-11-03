"""est-pm8sg: the connect budget must never pay for a fresh catalog disk walk.

Incident (2026-11-03, estate gateway): every Discord reconnect attempt ran
``_register_slash_commands`` — a disk walk of command-registry config gates, plugin
command entries and the skill catalog — BEFORE ``client.start()``, inside the 30s
connect budget. Under disk starvation the walk took minutes: attempts died at exactly
the budget with login never started (17:33:27 / 17:34:59 / 17:37:30), and the cancelled
``asyncio.to_thread`` left each walk running as an orphan thread (three finished
together at 17:38:01, two at 18:16:52). The timeout itself was logged only at DEBUG.

The fix (this patch): the walk is split from the tree apply, cached per profile home
and reused across adapter instances; a cold walk runs off the critical path under a
single-flight lock and lands after READY; a timed-out attempt logs WARNING with
elapsed time and stage.

Timing scale: WALK_SECONDS/CONNECT_BUDGET = 0.4/0.3 reproduces the incident's 40s-walk
/ 30s-budget shape (walk outlives the budget) without slowing the suite.
"""

import asyncio
import logging
import re
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.run_adapters import UNRESOLVED_PROFILE_HOME, GatewayAdapterLifecycleMixin

WALK_SECONDS = 0.4
CONNECT_BUDGET = 0.3


# The gateway test conftest installs a fake ``discord`` package at collection time.
# ``_register_skill_group`` decorates with app_commands.autocomplete; extend whichever
# module is loaded (real discord.py already provides it).
_discord_mod = sys.modules.get("discord")
if _discord_mod is not None and not hasattr(_discord_mod.app_commands, "autocomplete"):
    _discord_mod.app_commands.autocomplete = lambda **_kwargs: (lambda fn: fn)

import plugins.platforms.discord.adapter as discord_platform
from plugins.platforms.discord.adapter import DiscordAdapter


@pytest.fixture(autouse=True)
def _isolate_spec_cache():
    """The command-spec cache is process-global by design (that is the reuse); tests
    must not inherit each other's warm cache. Tolerates the attribute not existing yet
    so this file reports behavioral RED (not collection errors) on the base revision."""
    lock = getattr(discord_platform, "_SLASH_SPEC_LOCK", None)
    if lock is None:
        yield
        return

    def _reset():
        with lock:
            discord_platform._SLASH_SPECS.clear()
            discord_platform._SLASH_SPEC_WALKING.clear()
            discord_platform._SLASH_SPEC_WAITERS.clear()
    _reset()
    yield
    _reset()


@pytest.fixture(autouse=True)
def _no_metric_rows(monkeypatch):
    monkeypatch.setattr("gateway.run_adapters.record_platform_connect", lambda *a, **k: None)
    monkeypatch.setattr("gateway.run_adapters.record_platform_disconnect", lambda *a, **k: None)


class FakeTree:
    def __init__(self):
        self._commands = []

    def command(self, *_args, **_kwargs):
        return lambda fn: fn

    def get_commands(self, *_args, **_kwargs):
        return list(self._commands)

    def add_command(self, cmd):
        self._commands.append(cmd)


class _FakeBotBase:
    def __init__(self, *, intents, proxy=None, allowed_mentions=None, **_):
        self.intents = intents
        self.allowed_mentions = allowed_mentions
        self.application_id = 999
        self.user = SimpleNamespace(id=999, name="Hermes")
        self._events = {}
        self.tree = FakeTree()
        self.http = SimpleNamespace(
            upsert_global_command=AsyncMock(),
            edit_global_command=AsyncMock(),
            delete_global_command=AsyncMock(),
        )
        # Command names present on the tree the moment client.start() was called —
        # the proof of whether the attempt paid the catalog before login or not.
        self.commands_at_start = None
        self.starts = 0

    def event(self, fn):
        self._events[fn.__name__] = fn
        return fn

    def _snapshot(self):
        self.starts += 1
        self.commands_at_start = [cmd.name for cmd in self.tree.get_commands()]

    async def close(self):
        return None

    def is_closed(self):
        return False


class _ReadyAtStartBot(_FakeBotBase):
    """A Discord that answers instantly: READY fires from within start()."""

    async def start(self, _token):
        self._snapshot()
        if "on_ready" in self._events:
            await self._events["on_ready"]()


class _NeverReadyBot(_FakeBotBase):
    """A Discord whose login never completes: the attempt dies at the budget with the
    stage stamped 'waiting for Discord READY'."""

    async def start(self, _token):
        self._snapshot()
        await asyncio.Event().wait()


def _canned_spec():
    return discord_platform.SlashCommandSpec(
        registry_entries=(
            ("bg", "run a background task", "command"),
            ("background", "alias of bg", "command"),
        ),
        plugin_entries=(("lcm", "long context management", "subcommand"),),
        skill_entries=(("pdf", "process a pdf", "pdf-skill"),),
        skill_hidden=0,
        built_at=time.monotonic(),
    )


def _install_slow_walk(monkeypatch, *, seconds, calls, hold=None):
    """Stub the disk half so it takes ``seconds`` (the starved-disk stand-in).

    ``_register_slash_commands`` is patched to route through the same slow builder so
    this exact test is RED on the base revision, where connect() awaited that method on
    the critical path; on the patched revision connect() never awaits it."""

    def _build(_self):
        calls.append(threading.get_ident())
        if hold is not None:
            hold.wait(10.0)
        time.sleep(seconds)
        return _canned_spec()

    def _register(self):
        spec = _build(self)
        with discord_platform._SLASH_SPEC_LOCK:
            discord_platform._SLASH_SPECS[discord_platform._slash_spec_cache_key()] = spec
        apply = getattr(self, "_apply_slash_command_spec", None)
        if apply is not None:
            apply(self._client.tree, spec)

    monkeypatch.setattr(DiscordAdapter, "_build_slash_command_spec", _build, raising=False)
    monkeypatch.setattr(DiscordAdapter, "_register_slash_commands", _register)


def _make_adapter(monkeypatch, bot_cls):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test"))
    created = []

    def factory(*, command_prefix, intents, proxy=None, allowed_mentions=None, **_):
        bot = bot_cls(intents=intents, allowed_mentions=allowed_mentions)
        created.append(bot)
        return bot

    monkeypatch.setattr(discord_platform.commands, "Bot", factory)
    adapter._resolve_allowed_usernames = AsyncMock()
    adapter._wire_plugin_handlers = lambda _client: None
    adapter._run_post_connect_initialization = AsyncMock()
    adapter._created_bots = created
    return adapter


class _Runner(GatewayAdapterLifecycleMixin):
    """Bare lifecycle host: the production 30s default is replaced with the scaled budget."""

    def __init__(self, budget):
        self._budget = float(budget)

    def _platform_connect_timeout_secs(self, platform=None, *, initial=False):
        return self._budget


class _LadderRunner(_Runner):
    """Enough runner surface for _run_secondary_profile_reconnect's failure path."""

    def __init__(self, budget):
        super().__init__(budget)
        self._running = True
        self._profile_adapters = {}
        self._profile_failed_platforms = {}

    async def _safe_adapter_disconnect(self, adapter, platform):
        try:
            await adapter.disconnect()
        except Exception:
            pass

    def _flag_reconnect_needs_attention(self, *args, **kwargs):
        return None

    def _routed_profile_home(self, profile_name):
        return UNRESOLVED_PROFILE_HOME

    def _sync_voice_mode_state_to_adapter(self, adapter):
        return None

    async def _redeliver_failed_obligations_for_platform(self, platform, profile=None):
        return None

    def _schedule_planned_restart_replay(self):
        return None


def _walk_threads():
    return [t for t in threading.enumerate() if t.name.startswith("discord-slash-catalog-walk")]


@pytest.mark.asyncio
async def test_slow_catalog_walk_no_longer_delays_ready(monkeypatch):
    """Acceptance 1 (regression test from the row, time-scaled): a slash-command walk
    slower than the connect budget still reaches READY on the FIRST reconnect attempt,
    and the commands land on the tree right after the walk finishes."""
    calls = []
    _install_slow_walk(monkeypatch, seconds=WALK_SECONDS, calls=calls)
    adapter = _make_adapter(monkeypatch, _ReadyAtStartBot)
    runner = _Runner(CONNECT_BUDGET)

    started = time.monotonic()
    ok = await runner._connect_adapter_with_timeout(adapter, Platform.DISCORD, is_reconnect=True)
    elapsed = time.monotonic() - started

    assert ok is True, "first reconnect attempt must reach READY without paying the walk"
    assert elapsed < WALK_SECONDS, f"ready came after the walk finished ({elapsed:.2f}s)"
    # The walk ran — exactly once, on another thread (never the event loop).
    assert len(calls) == 1
    assert calls[0] != threading.get_ident()
    # Commands were NOT on the tree when the client started: login outran the disk.
    assert adapter._created_bots[0].commands_at_start == []
    # Deferred apply lands the commands the moment the walk stores the spec, then runs
    # the post-connect command sync so Discord actually sees them.
    assert adapter._slash_apply_task is not None
    await asyncio.wait_for(adapter._slash_apply_task, WALK_SECONDS + 5.0)
    names = {cmd.name for cmd in adapter._created_bots[0].tree.get_commands()}
    assert {"bg", "background", "lcm", "skill"} <= names
    await asyncio.sleep(0)  # let on_ready's own post-connect task record too
    assert adapter._run_post_connect_initialization.await_count >= 2


@pytest.mark.asyncio
async def test_second_adapter_instance_reuses_cached_spec_without_walk(monkeypatch):
    """Acceptance 1 (reuse half): after one good walk, a NEW adapter instance rebuilds
    the tree from the cached spec — synchronous, no walk thread, no disk — so every
    subsequent budget inside the TTL is walk-free."""
    calls = []
    _install_slow_walk(monkeypatch, seconds=0.0, calls=calls)
    first = _make_adapter(monkeypatch, _ReadyAtStartBot)
    assert await _Runner(30.0)._connect_adapter_with_timeout(
        first, Platform.DISCORD, is_reconnect=True
    ) is True
    await asyncio.wait_for(first._slash_apply_task, 5.0)
    assert len(calls) == 1

    def _boom(_self):
        raise AssertionError("connect must reuse the cached spec, not re-walk the disk")

    monkeypatch.setattr(DiscordAdapter, "_build_slash_command_spec", _boom)
    second = _make_adapter(monkeypatch, _ReadyAtStartBot)
    started = time.monotonic()
    assert await _Runner(30.0)._connect_adapter_with_timeout(
        second, Platform.DISCORD, is_reconnect=True
    ) is True
    assert time.monotonic() - started < 1.0
    assert second._slash_apply_task is None, "cache hit must not defer anything"
    # The tree was complete BEFORE client.start() — the fast path, not the deferred one.
    assert set(second._created_bots[0].commands_at_start) >= {"bg", "background", "lcm", "skill"}
    assert len(calls) == 1, "cache hit ran no walk"


@pytest.mark.asyncio
async def test_timed_out_attempts_warn_with_elapsed_and_stage_and_never_double_walk(
    monkeypatch, caplog
):
    """Acceptance 2 + 3: attempts that time out log WARNING naming elapsed and stage,
    and a cold walk is still in flight when the NEXT ladder attempt starts — exactly one
    walk thread exists (no orphan pile-up), one build call, and both attempts report."""
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_reconnect_backoff", lambda attempts: 0)
    calls = []
    hold = threading.Event()
    _install_slow_walk(monkeypatch, seconds=0.0, calls=calls, hold=hold)

    runner = _LadderRunner(CONNECT_BUDGET)

    attempt_calls = []

    async def attempt(profile_name, platform, predecessor=None):
        attempt_calls.append(1)
        if len(attempt_calls) >= 3:
            # Cap the ladder after the two budget deaths the test is about: the next
            # loop pass sees _running cleared and the task returns.
            runner._running = False
            return _make_adapter(monkeypatch, _NeverReadyBot), False
        adapter = _make_adapter(monkeypatch, _NeverReadyBot)
        try:
            ok = await runner._connect_adapter_with_timeout(adapter, platform, is_reconnect=True)
        except BaseException:
            await runner._safe_adapter_disconnect(adapter, platform)
            raise
        return adapter, ok

    runner._secondary_reconnect_attempt = attempt

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        ladder = asyncio.create_task(
            runner._run_secondary_profile_reconnect("clankville", Platform.DISCORD, None)
        )
        # Two budget deaths (each ~CONNECT_BUDGET plus scheduling slack).
        for _ in range(2 * 100):
            warnings = [
                r for r in caplog.records
                if "reconnect attempt failed after" in r.getMessage()
            ]
            if len(warnings) >= 2:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("ladder did not report two timed-out attempts as WARNING")
        # Single-flight held across both attempts: the second attempt ATTACHED to the
        # in-flight walk instead of starting (and stranding) another one.
        assert len(calls) == 1, f"two attempts must share one walk: {calls}"
        assert len(_walk_threads()) == 1
        runner._running = False
        hold.set()
        await asyncio.wait_for(ladder, 5.0)

    warnings = [
        r for r in caplog.records
        if "reconnect attempt failed after" in r.getMessage()
    ]
    assert len(warnings) == 2
    for rec in warnings:
        assert rec.levelno == logging.WARNING
        assert rec.name == "gateway.run"
        message = rec.getMessage()
        assert re.search(r"failed after \d+\.\d+s", message), message
        assert "clankville" in message
        # Where the attempt died: post-fix the budget cannot be spent in the walk, so
        # the honest stage is the login/READY wait.
        assert "waiting for Discord READY" in message, message

    # One builder call total across the whole ladder; the released walk finishes, caches,
    # and leaves no orphan thread behind.
    deadline = time.monotonic() + 5.0
    while len(calls) < 1 or _walk_threads() and time.monotonic() < deadline:
        await asyncio.sleep(0.02)
    assert len(calls) == 1
    assert not _walk_threads(), "no orphan catalog-walk thread may outlive the attempts"
    key = discord_platform._slash_spec_cache_key()
    assert key in discord_platform._SLASH_SPECS
    assert key not in discord_platform._SLASH_SPEC_WALKING


@pytest.mark.asyncio
async def test_stale_spec_serves_ram_and_refreshes_in_background(monkeypatch):
    """TTL half of the reuse law: an expired spec still builds the tree from RAM (the
    budget never waits for disk) while a single background walk refreshes it."""
    calls = []
    _install_slow_walk(monkeypatch, seconds=0.0, calls=calls)
    first = _make_adapter(monkeypatch, _ReadyAtStartBot)
    assert await _Runner(30.0)._connect_adapter_with_timeout(
        first, Platform.DISCORD, is_reconnect=True
    ) is True
    await asyncio.wait_for(first._slash_apply_task, 5.0)

    key = discord_platform._slash_spec_cache_key()
    import dataclasses
    with discord_platform._SLASH_SPEC_LOCK:
        warm = discord_platform._SLASH_SPECS[key]
        discord_platform._SLASH_SPECS[key] = dataclasses.replace(
            warm, built_at=time.monotonic() - discord_platform._SLASH_SPEC_TTL_SECS - 1,
        )
    second = _make_adapter(monkeypatch, _ReadyAtStartBot)
    assert await _Runner(30.0)._connect_adapter_with_timeout(
        second, Platform.DISCORD, is_reconnect=True
    ) is True
    assert second._slash_apply_task is None, "stale spec is served synchronously, not deferred"
    assert set(second._created_bots[0].commands_at_start) >= {"bg", "skill"}
    deadline = time.monotonic() + 5.0
    while (len(calls) < 2 or _walk_threads()) and time.monotonic() < deadline:
        await asyncio.sleep(0.02)
    assert len(calls) == 2, "exactly one refresh walk runs"
    assert not _walk_threads()
    assert key in discord_platform._SLASH_SPECS


@pytest.mark.asyncio
async def test_sync_register_slash_commands_compat_walks_and_caches(monkeypatch):
    """Compatibility seam: the direct ``_register_slash_commands()`` path (tests, manual
    re-registration) still walks synchronously, applies, and repopulates the cache."""
    calls = []
    _install_slow_walk(monkeypatch, seconds=0.0, calls=calls)
    adapter = _make_adapter(monkeypatch, _ReadyAtStartBot)
    adapter._client = _ReadyAtStartBot(intents=None)
    adapter._register_slash_commands()
    names = {cmd.name for cmd in adapter._client.tree.get_commands()}
    assert {"bg", "background", "lcm", "skill"} <= names
    assert discord_platform._slash_spec_cache_key() in discord_platform._SLASH_SPECS

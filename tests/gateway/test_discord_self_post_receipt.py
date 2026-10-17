"""``DiscordAdapter.has_self_post_since`` — the delivery-receipt probe behind silence-notice
suppression, pinned against a fake client: no network.

The capability answers True only on a positively-found message authored by THIS bot, newer than
every supplied floor (trigger snowflake id and/or epoch timestamp). Every miss — unknown channel,
dead client, REST error, flooded thread, only pre-trigger posts, other authors — answers False so
the caller keeps the unexpected-silence notice visible.
"""

from datetime import datetime, timedelta, timezone, UTC
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from plugins.platforms.discord.adapter import DiscordAdapter

BOT_ID = 4242
USER_ID = 99
TRIGGER_ID = 900_000_000_000_000_001
TRIGGER_TIME = datetime(2026, 10, 16, 12, 0, 0, tzinfo=UTC)


def _msg(*, author_id, created_at):
    return SimpleNamespace(
        id=author_id, author=SimpleNamespace(id=author_id), created_at=created_at)


class _Channel:
    """Fake text channel: yields the given newest-first history, records the call kwargs."""

    def __init__(self, messages, *, raises=None):
        self._messages = messages
        self._raises = raises
        self.kwargs = None

    def history(self, **kwargs):
        self.kwargs = kwargs
        messages, raises = self._messages, self._raises
        limit = kwargs.get("limit")

        async def _gen():
            if raises is not None:
                raise raises
            for index, m in enumerate(messages):
                if limit is not None and index >= limit:
                    break  # a real bounded page never yields past ``limit``
                yield m
        return _gen()


def _adapter(client, channel):
    adapter = object.__new__(DiscordAdapter)
    adapter.platform = SimpleNamespace(value="discord")
    adapter._client = client
    adapter._resolve_channel = AsyncMock(return_value=channel)
    return adapter


def _client():
    return SimpleNamespace(user=SimpleNamespace(id=BOT_ID, bot=True))


def _trigger_snowflake_at(dt_value):
    """Snowflake whose decoded creation time is ``dt_value`` (ms since Discord epoch << 22)."""
    epoch = datetime(2015, 1, 1, tzinfo=dt_value.tzinfo)
    ms = int((dt_value - epoch).total_seconds() * 1000)
    return (ms << 22) + 1


@pytest.mark.asyncio
async def test_bot_post_after_trigger_is_a_receipt():
    channel = _Channel([
        _msg(author_id=BOT_ID, created_at=TRIGGER_TIME + timedelta(seconds=30)),
        _msg(author_id=USER_ID, created_at=TRIGGER_TIME),
    ])
    adapter = _adapter(_client(), channel)
    assert await adapter.has_self_post_since(
        chat_id="c-1", after_id=str(TRIGGER_ID), after_ts=TRIGGER_TIME.timestamp()) is True
    # Bounded scan requested after the trigger id; order-agnostic (the fake discord module stands
    # in for discord.py's Object, so the floor is pinned behaviorally in the floor tests below).
    assert channel.kwargs["oldest_first"] is False
    assert channel.kwargs["after"] is not None


@pytest.mark.asyncio
async def test_only_human_posts_after_trigger_is_not_a_receipt():
    channel = _Channel([
        _msg(author_id=USER_ID, created_at=TRIGGER_TIME + timedelta(seconds=30)),
        _msg(author_id=USER_ID, created_at=TRIGGER_TIME),
    ])
    adapter = _adapter(_client(), channel)
    assert await adapter.has_self_post_since(
        chat_id="c-1", after_id=str(TRIGGER_ID), after_ts=TRIGGER_TIME.timestamp()) is False


@pytest.mark.asyncio
async def test_bot_post_older_than_the_trigger_does_not_count():
    """A stale answer from a previous turn must not silence this turn's notice: the floor filters
    every scanned message and the older self-post never qualifies as newer-than-every-floor."""
    channel = _Channel([
        _msg(author_id=USER_ID, created_at=TRIGGER_TIME + timedelta(seconds=5)),
        _msg(author_id=BOT_ID, created_at=TRIGGER_TIME - timedelta(minutes=5)),
    ])
    adapter = _adapter(_client(), channel)
    assert await adapter.has_self_post_since(
        chat_id="c-1", after_id=str(TRIGGER_ID), after_ts=TRIGGER_TIME.timestamp()) is False


@pytest.mark.asyncio
async def test_timestamp_floor_alone_works_without_message_id():
    """Crash recovery has no trigger id — the epoch floor must still gate the scan, strictly."""
    adapter = _adapter(_client(), _Channel(
        [_msg(author_id=BOT_ID, created_at=TRIGGER_TIME + timedelta(seconds=10))]))
    assert await adapter.has_self_post_since(chat_id="c-1", after_ts=TRIGGER_TIME.timestamp()) is True
    # A post exactly AT the marker time is not "newer than" it — rejected.
    adapter2 = _adapter(_client(), _Channel([_msg(author_id=BOT_ID, created_at=TRIGGER_TIME)]))
    assert await adapter2.has_self_post_since(
        chat_id="c-1", after_ts=TRIGGER_TIME.timestamp()) is False


@pytest.mark.asyncio
async def test_id_and_ts_floors_combine_to_the_stricter():
    """With both floors, a message that clears only the id floor must not count."""
    ts_later = TRIGGER_TIME + timedelta(minutes=5)
    later_id = _trigger_snowflake_at(ts_later)
    adapter = _adapter(_client(), _Channel(
        [_msg(author_id=BOT_ID, created_at=TRIGGER_TIME + timedelta(seconds=30))]))
    assert await adapter.has_self_post_since(
        chat_id="c-1", after_id=str(later_id), after_ts=TRIGGER_TIME.timestamp()) is False


@pytest.mark.asyncio
async def test_failed_or_missing_channel_fails_closed():
    adapter = _adapter(_client(), _Channel([], raises=RuntimeError("REST 403")))
    assert await adapter.has_self_post_since(chat_id="c-1", after_ts=TRIGGER_TIME.timestamp()) is False
    adapter2 = object.__new__(DiscordAdapter)
    adapter2.platform = SimpleNamespace(value="discord")
    adapter2._client = _client()
    adapter2._resolve_channel = AsyncMock(side_effect=RuntimeError("unknown channel"))
    assert await adapter2.has_self_post_since(
        chat_id="c-1", after_ts=TRIGGER_TIME.timestamp()) is False


@pytest.mark.asyncio
async def test_disconnected_client_or_missing_floors_is_false():
    adapter = _adapter(None, _Channel([]))
    assert await adapter.has_self_post_since(chat_id="c-1", after_ts=1.0) is False
    adapter._client = _client()
    assert await adapter.has_self_post_since(chat_id="c-1") is False  # no floor, no verdict
    assert await adapter.has_self_post_since(chat_id="c-1", after_id="not-a-snowflake") is False


@pytest.mark.asyncio
async def test_thread_target_scans_the_thread_not_the_parent():
    channel = _Channel([_msg(author_id=BOT_ID, created_at=TRIGGER_TIME + timedelta(seconds=30))])
    adapter = _adapter(_client(), channel)
    assert await adapter.has_self_post_since(
        chat_id="parent-1", thread_id="thread-7", after_ts=TRIGGER_TIME.timestamp()) is True
    assert adapter._resolve_channel.await_args.args[0] == "thread-7"


@pytest.mark.asyncio
async def test_flooded_thread_beyond_the_scan_limit_is_false():
    """A busy thread whose self-post sits past the bounded page is a miss: the scan is capped at
    the request's ``limit``, so an answer it never looks at can never suppress the notice."""
    page = [_msg(author_id=USER_ID, created_at=TRIGGER_TIME + timedelta(seconds=60 - i))
            for i in range(120)]
    channel = _Channel(page)
    adapter = _adapter(_client(), channel)
    assert await adapter.has_self_post_since(
        chat_id="c-1", after_id=str(TRIGGER_ID), after_ts=TRIGGER_TIME.timestamp()) is False
    assert channel.kwargs is not None and channel.kwargs["limit"] == 100


@pytest.mark.asyncio
async def test_scan_is_order_agnostic_when_floors_interleave():
    """Regression: the floor must filter, never break — ``history(after=...)`` has no
    iteration-order guarantee (the repo's own recovery backfill collects then re-sorts). With the
    id floor looser than the ts floor, a pre-ts-floor post yielded FIRST must not abort the scan:
    the receipt later in the page still counts."""
    id_floor = _trigger_snowflake_at(TRIGGER_TIME)
    stricter_ts = (TRIGGER_TIME + timedelta(minutes=5)).timestamp()
    channel = _Channel([  # ascending order: pre-floor interloper first, receipt after it
        _msg(author_id=BOT_ID, created_at=TRIGGER_TIME + timedelta(seconds=30)),
        _msg(author_id=BOT_ID, created_at=TRIGGER_TIME + timedelta(minutes=6)),
    ])
    adapter = _adapter(_client(), channel)
    assert await adapter.has_self_post_since(
        chat_id="c-1", after_id=str(id_floor), after_ts=stricter_ts) is True

"""est-2ek.1.887 — byte-loud-path-silent caps class.

Every cap message must carry {limit, actual, delta, overflow-path} in ONE message,
so knowledge-banking late in a session never degenerates into a byte-measuring loop.

Sites: skill_manage's 100K append-abort (tools/skill_manager_tool.py) and the
AGENTS.md / context-file 32K truncation (agent/prompt_builder.py::_truncate_content).
"""

import json
import re

import pytest

from tools.skill_manager_tool import (
    MAX_SKILL_CONTENT_CHARS,
    _validate_content_size,
    skill_manage,
)


def _numbers(msg: str):
    """Every comma-grouped or bare integer in the message."""
    return [int(m.replace(",", "")) for m in re.findall(r"\d[\d,]*", msg)]


# --- skill_manage 100K append-abort ------------------------------------------------

class TestSkillManageByteLoud:
    def test_rejection_names_limit_actual_delta_overflow_path(self):
        actual = MAX_SKILL_CONTENT_CHARS + 2500
        err = _validate_content_size("x" * actual)
        assert err is not None
        nums = _numbers(err)
        assert MAX_SKILL_CONTENT_CHARS in nums, "limit missing"
        assert actual in nums, "actual missing"
        assert actual - MAX_SKILL_CONTENT_CHARS in nums, "delta missing"
        assert "references/" in err, "overflow-path missing"

    def test_rejection_via_create_tool_call(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        monkeypatch.setattr("tools.skill_manager_tool.SKILLS_DIR", skills_dir)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        frontmatter = "---\nname: fat-skill\ndescription: Too fat\n---\n# Fat\n\n"
        content = frontmatter + "x" * (MAX_SKILL_CONTENT_CHARS - len(frontmatter) + 5000)
        result = json.loads(skill_manage(action="create", name="fat-skill", content=content))
        assert result["success"] is not True
        msg = json.dumps(result)
        nums = _numbers(msg)
        assert MAX_SKILL_CONTENT_CHARS in nums and len(content) in nums
        assert (len(content) - MAX_SKILL_CONTENT_CHARS) in nums
        assert "references/" in msg


# --- prompt_builder context-file truncation ----------------------------------------

class TestTruncateContentByteLoud:
    @pytest.fixture
    def cap32k(self, monkeypatch):
        def fake_load_config():
            return {"context_file_max_chars": 32000}
        monkeypatch.setattr("hermes_cli.config.load_config", fake_load_config)
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", fake_load_config)
        return 32000

    def _check(self, msg, limit, actual, path):
        nums = _numbers(msg)
        assert limit in nums, f"limit missing from: {msg}"
        assert actual in nums, f"actual missing from: {msg}"
        assert actual - limit in nums, f"delta missing from: {msg}"
        assert path in msg, f"overflow-path missing from: {msg}"

    def test_queued_warning_is_byte_loud(self, cap32k, caplog):
        from agent.prompt_builder import _truncate_content, drain_truncation_warnings
        actual = cap32k + 2106  # mirrors the real 34106 vs 32000 AGENTS.md case
        _truncate_content("x" * actual, "AGENTS.md", read_path="/work/AGENTS.md")
        (warn,) = drain_truncation_warnings()
        self._check(warn, cap32k, actual, "/work/AGENTS.md")
        assert any("TRUNCATED" in r.message for r in caplog.records if r.levelname == "WARNING")

    def test_quiet_preview_path_is_byte_loud_in_log(self, cap32k, caplog):
        """queue_warning=False (subdirectory-hint previews) used to be log-only AND
        delta-less — the 'path-silent' half of the class. It must log the full tuple."""
        from agent.prompt_builder import _truncate_content, drain_truncation_warnings
        actual = cap32k + 77
        _truncate_content("y" * actual, "AGENTS.md", read_path="/work/sub/AGENTS.md",
                          queue_warning=False)
        assert drain_truncation_warnings() == []  # still never queues a status line
        logged = [r.message for r in caplog.records if r.levelname == "WARNING"]
        (msg,) = [m for m in logged if "TRUNCATED" in m]
        self._check(msg, cap32k, actual, "/work/sub/AGENTS.md")

    def test_truncated_body_keeps_recovery_path(self, cap32k):
        from agent.prompt_builder import _truncate_content
        actual = cap32k + 1000
        out = _truncate_content("z" * actual, "AGENTS.md", read_path="/work/AGENTS.md")
        assert "/work/AGENTS.md" in out
        assert len(out) < actual

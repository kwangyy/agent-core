# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Tests for the per-session cua_agent.log files."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from openjiuwen.core.common.logging import LogManager, logger
from openjiuwen.core.common.logging.cua_context import (
    cua_agent_logger_name,
    get_cua_agent_log_session,
    is_cua_agent_log_context,
)
from openjiuwen.core.foundation.llm import AssistantMessage, ToolCall
from openjiuwen.core.single_agent.rail.base import ModelCallInputs
from openjiuwen.harness.tools.cua import cua_logging
from openjiuwen.harness.tools.cua.desktop_tools import CuaDesktop, build_cua_desktop_tools
from openjiuwen.harness.tools.cua.rails import CuaRuntimeRail


class _Backend:
    def screen_size(self) -> tuple[int, int]:
        return 1920, 1080

    def cursor_position(self) -> tuple[int, int]:
        return 7, 9


class _BrokenBackend(_Backend):
    def cursor_position(self) -> tuple[int, int]:
        raise OSError("display gone")


def _ctx(session_id: str = "sub-1", query: str = "open notepad") -> SimpleNamespace:
    session = MagicMock()
    session.get_session_id.return_value = session_id
    return SimpleNamespace(extra={}, session=session, inputs=SimpleNamespace(query=query))


def _read(log_dir: Path, session_id: str) -> str:
    path = log_dir / session_id / "cua_agent.log"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _action_record(text: str) -> dict:
    line = next(line for line in text.splitlines() if "[CUA_ACTION]" in line)
    return json.loads(line.split("[CUA_ACTION] ", 1)[1])


def _tool(backend: _Backend, name: str):
    (tool,) = build_cua_desktop_tools(CuaDesktop(lambda: backend), names=(name,))
    return tool


def test_log_path_is_per_session_and_can_be_disabled(cua_log_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert cua_logging.get_cua_agent_log_path("sub-1") == cua_log_dir.resolve() / "sub-1" / "cua_agent.log"
    # A session id never escapes the log directory.
    assert cua_logging.get_cua_agent_log_path("../../etc").parent.parent == cua_log_dir.resolve()

    monkeypatch.delenv("OPENJIUWEN_CUA_AGENT_LOG_DIR")
    assert cua_logging.get_cua_agent_log_dir() == (Path.cwd() / "logs" / "cua_agent").resolve()

    monkeypatch.setenv("OPENJIUWEN_CUA_AGENT_LOG_DIR", "off")
    assert cua_logging.get_cua_agent_log_path("sub-1") is None


@pytest.mark.asyncio
async def test_run_writes_actions_and_framework_lines_to_its_session_folder(cua_log_dir: Path) -> None:
    LogManager.reset()
    rail = CuaRuntimeRail(CuaDesktop(_Backend))
    ctx = _ctx("sub-1")

    await rail.before_invoke(ctx)
    assert get_cua_agent_log_session() == "sub-1"
    await _tool(_Backend(), "get_cursor_position").invoke({"note": "x"})
    logger.info("framework line inside the cua run")
    await rail.after_invoke(ctx)

    assert not is_cua_agent_log_context()
    assert not logging.getLogger(cua_agent_logger_name("sub-1")).handlers, "a finished run keeps no file open"
    logger.info("framework line after the cua run")
    text = _read(cua_log_dir, "sub-1")
    assert "[CUA_INVOKE_START] session=sub-1 query=open notepad" in text
    assert "framework line inside the cua run" in text
    assert "[CUA_INVOKE_END] session=sub-1" in text
    assert "framework line after the cua run" not in text
    record = _action_record(text)
    assert record["tool"] == "get_cursor_position"
    assert record["args"] == {"note": "x"}
    assert record["success"] is True
    assert "(7, 9)" in record["result"]
    assert record["images"] == 0


@pytest.mark.asyncio
async def test_model_reply_is_written_to_the_session_file_only_inside_a_run(cua_log_dir: Path) -> None:
    rail = CuaRuntimeRail(CuaDesktop(_Backend))
    response = AssistantMessage(
        content="Notepad is open, typing now.",
        finish_reason="tool_calls",
        tool_calls=[ToolCall(id="c1", type="function", name="type_text", arguments='{"text": "hello"}')],
    )
    reply_ctx = SimpleNamespace(extra={}, inputs=ModelCallInputs(response=response, react_iteration=3))

    await rail.after_model_call(reply_ctx)  # outside a run: nothing is written anywhere
    ctx = _ctx("sub-1")
    await rail.before_invoke(ctx)
    await rail.after_model_call(reply_ctx)
    await rail.after_invoke(ctx)

    text = _read(cua_log_dir, "sub-1")
    assert text.count("[CUA_MODEL]") == 1
    line = next(line for line in text.splitlines() if "[CUA_MODEL]" in line)
    record = json.loads(line.split("[CUA_MODEL] ", 1)[1])
    assert record == {
        "iteration": 3,
        "finish_reason": "tool_calls",
        "content": "Notepad is open, typing now.",
        "reasoning": None,
        "tool_calls": [{"name": "type_text", "arguments": '{"text": "hello"}'}],
    }


@pytest.mark.asyncio
async def test_parallel_runs_write_to_separate_folders(cua_log_dir: Path) -> None:
    async def run(session_id: str, backend: _Backend) -> None:
        rail = CuaRuntimeRail(CuaDesktop(lambda: backend))
        ctx = _ctx(session_id)
        await rail.before_invoke(ctx)
        await asyncio.sleep(0)
        await _tool(backend, "get_cursor_position").invoke({"run": session_id})
        await rail.after_invoke(ctx)

    await asyncio.gather(run("sub-a", _Backend()), run("sub-b", _BrokenBackend()))

    a, b = _read(cua_log_dir, "sub-a"), _read(cua_log_dir, "sub-b")
    assert _action_record(a)["args"] == {"run": "sub-a"}
    assert _action_record(b)["args"] == {"run": "sub-b"}
    assert "sub-b" not in a and "sub-a" not in b
    assert "| WARNING |" in next(line for line in b.splitlines() if "[CUA_ACTION]" in line)
    assert "display gone" in b


@pytest.mark.asyncio
async def test_resumed_session_appends_to_its_file(cua_log_dir: Path) -> None:
    rail = CuaRuntimeRail(CuaDesktop(_Backend))
    for _ in range(2):
        ctx = _ctx("sub-1")
        await rail.before_invoke(ctx)
        await rail.after_invoke(ctx)

    assert _read(cua_log_dir, "sub-1").count("[CUA_INVOKE_START]") == 2


@pytest.mark.asyncio
async def test_runtime_rail_reentry_holds_a_single_context_token() -> None:
    rail = CuaRuntimeRail(CuaDesktop(_Backend))
    outer, reentry = _ctx(), _ctx()

    await rail.before_invoke(outer)
    await rail.before_invoke(reentry)
    assert "__cua_agent_log_context_token__" not in reentry.extra
    await rail.after_invoke(reentry)
    assert is_cua_agent_log_context()

    await rail.after_invoke(outer)
    assert not is_cua_agent_log_context()


@pytest.mark.asyncio
async def test_action_outside_a_cua_run_goes_to_the_common_log(cua_log_dir: Path) -> None:
    await _tool(_Backend(), "get_cursor_position").invoke({})

    assert _read(cua_log_dir, "unknown_session") == ""
    assert not any(cua_log_dir.parent.glob("cua_logs/*"))


@pytest.mark.asyncio
async def test_runtime_rail_leaves_context_off_when_files_are_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENJIUWEN_CUA_AGENT_LOG_DIR", "none")
    rail = CuaRuntimeRail(CuaDesktop(_Backend))
    ctx = _ctx()

    await rail.before_invoke(ctx)

    assert not is_cua_agent_log_context()
    assert ctx.extra == {}

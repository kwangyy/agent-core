# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Tests for the cua runtime rail."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from openjiuwen.core.common.exception.errors import BaseError
from openjiuwen.core.foundation.tool import McpServerConfig
from openjiuwen.harness.tools.cua.rails import CuaRuntimeRail


def _mcp_cfg() -> McpServerConfig:
    return McpServerConfig(
        server_id="cua_driver_stdio",
        server_name="cua-driver",
        server_path="stdio://cua-driver",
        client_type="stdio",
        params={"command": "cua-driver", "args": ["mcp"]},
    )


def _ctx(session_id: str = "session-1") -> SimpleNamespace:
    session = MagicMock()
    session.get_session_id.return_value = session_id
    return SimpleNamespace(agent=MagicMock(), session=session)


def _probe_process(returncode: int = 0, stderr: bytes = b"") -> MagicMock:
    process = MagicMock()
    process.returncode = returncode
    process.communicate = AsyncMock(return_value=(b"{}", stderr))
    process.kill = MagicMock()
    return process


@pytest.mark.asyncio
async def test_before_invoke_fails_loud_when_daemon_probe_fails(monkeypatch) -> None:
    # A dead daemon must abort the run with remediation, not degrade silently.
    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        AsyncMock(return_value=_probe_process(returncode=1, stderr=b"connect error")),
    )
    rail = CuaRuntimeRail(_mcp_cfg())

    with pytest.raises(BaseError, match="autostart kick"):
        await rail.before_invoke(_ctx())


@pytest.mark.asyncio
async def test_before_invoke_fails_loud_when_binary_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        AsyncMock(side_effect=FileNotFoundError()),
    )
    rail = CuaRuntimeRail(_mcp_cfg())

    with pytest.raises(BaseError, match="binary not found"):
        await rail.before_invoke(_ctx())


@pytest.mark.asyncio
async def test_before_invoke_applies_allowlist_and_starts_session(monkeypatch) -> None:
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=_probe_process()))
    start_tool = MagicMock()
    start_tool.invoke = AsyncMock()
    resource_mgr = MagicMock()
    resource_mgr.get_tool.return_value = start_tool
    from openjiuwen.core.runner.runner import Runner

    monkeypatch.setattr(Runner, "resource_mgr", resource_mgr)

    cfg = _mcp_cfg()
    rail = CuaRuntimeRail(cfg, ("list_apps", "click"))
    ctx = _ctx(session_id="abc123")

    await rail.before_invoke(ctx)

    ability_manager = ctx.agent.ability_manager
    ability_manager.add.assert_called_once_with(cfg)
    ability_manager.set_mcp_tool_allowlist.assert_called_once_with(cfg, ("list_apps", "click"))
    resource_mgr.get_tool.assert_called_once_with("cua_driver_stdio.cua-driver.start_session")
    start_tool.invoke.assert_awaited_once_with({"session": "cua-abc123"})


@pytest.mark.asyncio
async def test_session_failures_never_block_the_run(monkeypatch) -> None:
    # Session identity is cosmetic; the daemon idle TTL reclaims leaks.
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=_probe_process()))
    failing_tool = MagicMock()
    failing_tool.invoke = AsyncMock(side_effect=RuntimeError("denied"))
    resource_mgr = MagicMock()
    resource_mgr.get_tool.return_value = failing_tool
    from openjiuwen.core.runner.runner import Runner

    monkeypatch.setattr(Runner, "resource_mgr", resource_mgr)

    rail = CuaRuntimeRail(_mcp_cfg())
    ctx = _ctx()

    await rail.before_invoke(ctx)  # start_session raises internally -> warning only
    await rail.after_invoke(ctx)  # end_session raises internally -> warning only


@pytest.mark.asyncio
async def test_after_invoke_ends_the_declared_session(monkeypatch) -> None:
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=_probe_process()))
    tool = MagicMock()
    tool.invoke = AsyncMock()
    resource_mgr = MagicMock()
    resource_mgr.get_tool.return_value = tool
    from openjiuwen.core.runner.runner import Runner

    monkeypatch.setattr(Runner, "resource_mgr", resource_mgr)

    rail = CuaRuntimeRail(_mcp_cfg())
    ctx = _ctx(session_id="xyz789")

    await rail.before_invoke(ctx)
    await rail.after_invoke(ctx)

    end_call = tool.invoke.await_args_list[-1]
    assert end_call.args[0] == {"session": "cua-xyz789"}
    # A second after_invoke is a no-op (session already cleared).
    tool.invoke.reset_mock()
    await rail.after_invoke(ctx)
    tool.invoke.assert_not_awaited()

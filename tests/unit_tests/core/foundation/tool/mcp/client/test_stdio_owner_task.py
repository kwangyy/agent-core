# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""StdioClient must keep the transport's anyio scopes off the caller's task.

mcp's stdio transport opens an anyio task group when entered. If StdioClient
entered it on the caller's task, that cancel scope would stay on the caller's
stack after ``connect()`` returned, and any scope enclosing the caller -- the
``anyio.fail_after`` around a ``task_tool`` delegation whose subagent connects
its MCP server on first use -- would fail to exit with "Attempted to exit a
cancel scope that isn't the current task's current cancel scope". Reproduced
live with the cua_agent subagent; these tests pin the fix.
"""

from __future__ import annotations

import asyncio
import contextlib

import anyio
import pytest

from openjiuwen.core.foundation.tool import McpServerConfig
from openjiuwen.core.foundation.tool.mcp.client.stdio_client import StdioClient


class _FakeTransport:
    """Stand-in for mcp.client.stdio.stdio_client with the same scope shape."""

    def __init__(self) -> None:
        self.enter_task: asyncio.Task | None = None
        self.exit_task: asyncio.Task | None = None
        self.exited = False

    @contextlib.asynccontextmanager
    async def stdio_client(self, params):
        self.enter_task = asyncio.current_task()
        async with anyio.create_task_group():
            yield ("read", "write")
        self.exit_task = asyncio.current_task()
        self.exited = True


class _FakeSession:
    def __init__(self, read, write, sampling_callback=None) -> None:
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.closed = True

    async def initialize(self) -> None:
        return None


@pytest.fixture
def transport(monkeypatch):
    import mcp
    import mcp.client.stdio as stdio_module

    fake = _FakeTransport()
    monkeypatch.setattr(stdio_module, "stdio_client", fake.stdio_client)
    monkeypatch.setattr(mcp, "ClientSession", _FakeSession)
    return fake


def _config() -> McpServerConfig:
    return McpServerConfig(
        server_id="fake_stdio",
        server_name="fake",
        server_path="stdio://fake",
        client_type="stdio",
        params={"command": "fake", "args": []},
    )


@pytest.mark.asyncio
async def test_connect_inside_an_enclosing_scope_keeps_that_scope_exitable(transport):
    client = StdioClient(_config())

    # The scope task_tool wraps around a delegation. With the transport entered
    # on this task, leaving it raised RuntimeError; it must exit cleanly.
    with anyio.fail_after(5):
        assert await client.connect() is True

    assert transport.enter_task is not asyncio.current_task()
    assert await client.disconnect() is True
    assert transport.exited is True
    assert transport.exit_task is transport.enter_task


@pytest.mark.asyncio
async def test_connect_failure_is_reported_and_leaves_no_owner_task(transport, monkeypatch):
    async def _boom(self) -> None:
        raise RuntimeError("init failed")

    monkeypatch.setattr(_FakeSession, "initialize", _boom)
    client = StdioClient(_config())

    assert await client.connect() is False

    assert isinstance(client._last_connect_error, RuntimeError)
    assert client._owner_task is None
    assert client._session is None
    assert transport.exited is True


@pytest.mark.asyncio
async def test_client_can_reconnect_after_disconnect(transport):
    client = StdioClient(_config())

    assert await client.connect() is True
    assert await client.disconnect() is True
    assert await client.connect() is True
    assert client._session is not None
    assert await client.disconnect() is True

# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import asyncio
from contextlib import AsyncExitStack
from typing import Any, List, Optional

from openjiuwen.core.common.logging import logger
from openjiuwen.core.foundation.tool import McpServerConfig, McpToolCard
from openjiuwen.core.foundation.tool.mcp.base import NO_TIMEOUT, extract_mcp_tool_result_content
from openjiuwen.core.foundation.tool.mcp.client.mcp_client import McpClient

# How long disconnect() lets the owner task close the transport on its own
# before cancelling it, and then how long the cancelled task gets to unwind.
_DISCONNECT_GRACE_S = 30.0
_CANCEL_GRACE_S = 5.0


class StdioClient(McpClient):
    """Stdio transport based MCP client.

    The transport's async context managers (``stdio_client`` + ``ClientSession``)
    are entered and exited by a dedicated owner task, never by the caller of
    ``connect``. mcp's stdio transport opens an anyio task group on the task
    that enters it; entered on the caller's task, that cancel scope would stay
    on the caller's scope stack after ``connect`` returned, and any scope
    enclosing the caller -- e.g. the ``anyio.fail_after`` around a ``task_tool``
    delegation whose subagent connects its MCP server on first use -- would then
    fail to exit with "Attempted to exit a cancel scope that isn't the current
    task's current cancel scope". Same actor pattern as BrowserMoveStdioClient.
    """
    __client_name__ = "stdio"

    def __init__(self, config: McpServerConfig):
        super().__init__(config)
        self._name = config.server_name
        self._client = None
        self._session = None
        self._read = None
        self._write = None
        self._params = config.params if config.params else {}
        self._exit_stack = AsyncExitStack()
        self._is_disconnected: bool = False
        self._owner_task: Optional[asyncio.Task] = None
        self._owner_ready: Optional[asyncio.Event] = None
        self._owner_close: Optional[asyncio.Event] = None
        self._connect_error: Optional[BaseException] = None

    async def connect(self, *, timeout: float = NO_TIMEOUT) -> bool:
        """Establish Stdio connection to the tool server"""
        if self._owner_task is not None and not self._owner_task.done():
            logger.warning("Stdio client already connecting or connected")
            return self._session is not None

        valid_handlers = {"strict", "ignore", "replace"}
        handler = self._params.get('encoding_error_handler', 'strict')
        if handler not in valid_handlers:
            handler = 'strict'

        self._connect_error = None
        self._owner_ready = asyncio.Event()
        self._owner_close = asyncio.Event()
        self._owner_task = asyncio.create_task(self._run_owner(handler))

        try:
            if timeout is not None and timeout > 0:
                await asyncio.wait_for(self._owner_ready.wait(), timeout=timeout)
            else:
                await self._owner_ready.wait()
        except asyncio.TimeoutError as e:
            logger.error(f"Stdio connection timed out after {timeout}s")
            self._last_connect_error = e
            await self.disconnect()
            return False

        if self._connect_error is not None:
            logger.error(f"Stdio connection failed: {self._connect_error}")
            self._last_connect_error = self._connect_error
            await self.disconnect()
            return False

        logger.info("Stdio client connected successfully")
        return True

    async def _run_owner(self, handler: str) -> None:
        """Own the transport context managers for the connection's lifetime.

        Runs as its own task so the transport's anyio scopes are entered and
        exited on the same task, and never on a caller's scope stack (see the
        class docstring).
        """
        try:
            try:
                from mcp import ClientSession, StdioServerParameters
                from mcp.client.stdio import stdio_client

                params = StdioServerParameters(command=self._params.get('command'),
                                               args=self._params.get('args'),
                                               env=self._params.get('env'),
                                               cwd=self._params.get('cwd'),
                                               encoding_error_handler=handler
                                               )
                self._exit_stack = AsyncExitStack()
                self._client = stdio_client(params)
                self._read, self._write = await self._exit_stack.enter_async_context(self._client)
                self._session = await self._exit_stack.enter_async_context(
                    ClientSession(self._read, self._write, sampling_callback=None))
                await self._session.initialize()
                self._is_disconnected = False
            except Exception as e:  # noqa: BLE001 - surfaced to connect() through _connect_error
                self._connect_error = e
            finally:
                self._owner_ready.set()

            if self._connect_error is None:
                await self._owner_close.wait()
        except asyncio.CancelledError:
            logger.info("Stdio client owner task cancelled")
        finally:
            try:
                await self._exit_stack.aclose()
            except Exception as e:  # noqa: BLE001 - teardown must not mask the caller's outcome
                logger.error(f"Stdio disconnection failed: {e}")
            self._session = None
            self._client = None
            self._read = None
            self._write = None
            self._exit_stack = AsyncExitStack()
            self._is_disconnected = True

    async def disconnect(self, *, timeout: float = NO_TIMEOUT) -> bool:
        """Close Stdio connection"""
        task = self._owner_task
        if task is None:
            self._is_disconnected = True
            logger.info("Stdio client disconnected successfully")
            return True

        if self._owner_close is not None:
            self._owner_close.set()
        grace = timeout if (timeout is not None and timeout > 0) else _DISCONNECT_GRACE_S
        try:
            done, _ = await asyncio.wait({task}, timeout=grace)
            if not done:
                logger.warning(f"Stdio client owner task did not finish within {grace}s; cancelling")
                task.cancel()
                done, _ = await asyncio.wait({task}, timeout=_CANCEL_GRACE_S)
                if not done:
                    # The transport may be stuck closing the subprocess; keep
                    # the warning loud so a pipe leak is visible to operators.
                    logger.warning("Stdio client owner task did not terminate after cancel; subprocess pipe may leak")
        finally:
            if self._owner_task is task:
                self._owner_task = None
            self._session = None
            self._client = None
            self._read = None
            self._write = None
            self._is_disconnected = True

        logger.info("Stdio client disconnected successfully")
        return True

    async def list_tools(self, *, timeout: float = NO_TIMEOUT) -> List[Any]:
        """List available tools via Stdio"""
        if not self._session:
            raise RuntimeError("Not connected to Stdio server")

        try:
            tools_response = await self._session.list_tools()
            tools_list = [
                McpToolCard(
                    name=tool.name,
                    server_name=self._name,
                    description=getattr(tool, "description", ""),
                    input_params=getattr(tool, "inputSchema", {}),
                )
                for tool in tools_response.tools
            ]
            logger.info(f"Retrieved {len(tools_list)} tools from Stdio server")
            return tools_list
        except Exception as e:
            logger.error(f"Failed to list tools via Stdio: {e}")
            raise

    async def call_tool(self, tool_name: str, arguments: dict, *, timeout: float = NO_TIMEOUT) -> Any:
        """Call tool via Stdio"""
        if not self._session:
            raise RuntimeError("Not connected to Stdio server")

        try:
            logger.info(f"Calling tool '{tool_name}' via Stdio with arguments: {arguments}")
            tool_result = await self._session.call_tool(tool_name, arguments=arguments)
            result_content = extract_mcp_tool_result_content(
                tool_result,
                include_structured_content=self._include_structured_content,
                include_image_content=self._include_image_content,
                tool_name=tool_name,
            )
            logger.info(f"Tool '{tool_name}' call completed via Stdio")
            return result_content
        except Exception as e:
            logger.error(f"Tool call failed via Stdio: {e}")
            raise

    async def get_tool_info(self, tool_name: str, *, timeout: float = NO_TIMEOUT) -> Optional[Any]:
        """Get specific tool info via Stdio"""
        tools = await self.list_tools(timeout=timeout)
        for tool in tools:
            if tool.name == tool_name:
                logger.debug(f"Found tool info for '{tool_name}' via Stdio")
                return tool
        logger.warning(f"Tool '{tool_name}' not found via Stdio")
        return None

    async def list_resources(self, *, timeout: float = NO_TIMEOUT) -> List[Any]:
        """List available resources via Stdio"""
        if not self._session:
            raise RuntimeError("Not connected to Stdio server")
        try:
            response = await self._session.list_resources()
            return response.resources
        except Exception as e:
            logger.error(f"Failed to list resources via Stdio: {e}")
            raise

    async def read_resource(self, uri: str, *, timeout: float = NO_TIMEOUT) -> Any:
        """Read a resource by URI via Stdio"""
        if not self._session:
            raise RuntimeError("Not connected to Stdio server")
        try:
            response = await self._session.read_resource(uri)
            return response.contents
        except Exception as e:
            logger.error(f"Failed to read resource '{uri}' via Stdio: {e}")
            raise

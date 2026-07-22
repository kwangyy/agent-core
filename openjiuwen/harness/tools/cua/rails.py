# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Rails for the cua-driver desktop runtime."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Optional, Sequence

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import raise_error
from openjiuwen.core.common.logging import logger
from openjiuwen.core.foundation.tool import McpServerConfig
from openjiuwen.core.single_agent.rail.base import AgentCallbackContext, AgentRail

_DAEMON_REMEDIATION = (
    "cua-driver daemon unreachable. Start it with `cua-driver autostart kick` "
    "(or `cua-driver serve` in an interactive desktop session) and retry."
)

# get_screen_size is cheap, read-only, and risk-classified on all platforms;
# health_report is denied under the driver's default permission mode
# ("no reviewed risk classification"), so it cannot serve as the probe.
_PROBE_TOOL = "get_screen_size"


class CuaRuntimeRail(AgentRail):
    """Lifecycle rail for the cua-driver MCP runtime.

    Per invoke: verify the daemon is reachable (fail loud with remediation),
    enforce the resolved capability allowlist on the driver MCP server, and
    declare/end a driver session so concurrent agent runs get distinct cursor
    identities and session-scoped driver state.
    """

    def __init__(
        self,
        mcp_cfg: McpServerConfig,
        allowed_tool_names: Optional[Sequence[str]] = None,
        *,
        probe_timeout_s: float = 15.0,
    ) -> None:
        super().__init__()
        self._mcp_cfg = mcp_cfg
        self._allowed_tool_names = tuple(allowed_tool_names) if allowed_tool_names is not None else None
        self._probe_timeout_s = probe_timeout_s
        self._session_key: Optional[str] = None

    async def before_invoke(self, ctx: AgentCallbackContext) -> None:
        await self._ensure_daemon_reachable()
        self._ensure_cua_mcp_ability(ctx)
        await self._start_driver_session(ctx)

    async def after_invoke(self, ctx: AgentCallbackContext) -> None:
        # Driver sessions are also reclaimed by the daemon's idle TTL, so a
        # missed cleanup (crash before after_invoke) self-heals server-side.
        await self._end_driver_session()

    async def _ensure_daemon_reachable(self) -> None:
        command = str(self._mcp_cfg.params.get("command") or "cua-driver")
        reason: Optional[str] = None
        try:
            process = await asyncio.create_subprocess_exec(
                command,
                "call",
                _PROBE_TOOL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, stderr = await asyncio.wait_for(process.communicate(), timeout=self._probe_timeout_s)
            except asyncio.TimeoutError:
                process.kill()
                reason = f"probe `{command} call {_PROBE_TOOL}` timed out after {self._probe_timeout_s}s"
            else:
                if process.returncode != 0:
                    detail = (stderr or b"").decode(errors="replace").strip()[:200]
                    reason = f"probe `{command} call {_PROBE_TOOL}` failed: {detail}"
        except FileNotFoundError:
            reason = f"cua-driver binary not found at {command!r}"

        if reason is not None:
            raise_error(
                StatusCode.RESOURCE_MCP_SERVER_CONNECTION_ERROR,
                server_config=self._mcp_cfg.server_id,
                reason=f"{reason}. {_DAEMON_REMEDIATION}",
            )

    def _ensure_cua_mcp_ability(self, ctx: AgentCallbackContext) -> None:
        agent = getattr(ctx, "agent", None)
        ability_manager = getattr(agent, "ability_manager", None)
        if ability_manager is None:
            return
        ability_manager.add(self._mcp_cfg)
        if self._allowed_tool_names is not None:
            ability_manager.set_mcp_tool_allowlist(self._mcp_cfg, self._allowed_tool_names)

    def _driver_tool(self, tool_name: str) -> Optional[Any]:
        from openjiuwen.core.runner.runner import Runner

        tool_id = f"{self._mcp_cfg.server_id}.{self._mcp_cfg.server_name}.{tool_name}"
        return Runner.resource_mgr.get_tool(tool_id)

    async def _start_driver_session(self, ctx: AgentCallbackContext) -> None:
        """Declare a driver session. Cosmetic identity: failure never blocks the run."""
        session = getattr(ctx, "session", None)
        session_id = None
        if session is not None:
            try:
                session_id = session.get_session_id()
            except Exception:  # noqa: BLE001 - identity fallback only
                session_id = None
        self._session_key = f"cua-{str(session_id or uuid.uuid4().hex)[:16]}"

        tool = self._driver_tool("start_session")
        if tool is None:
            logger.warning("[CuaRuntimeRail] start_session tool not registered; running session-less")
            return
        try:
            await tool.invoke({"session": self._session_key})
        except Exception as exc:  # noqa: BLE001 - session identity is non-critical
            logger.warning("[CuaRuntimeRail] start_session failed: %s", exc)

    async def _end_driver_session(self) -> None:
        session_key = self._session_key
        self._session_key = None
        if not session_key:
            return
        tool = self._driver_tool("end_session")
        if tool is None:
            return
        try:
            await tool.invoke({"session": session_key})
        except Exception as exc:  # noqa: BLE001 - daemon idle TTL reclaims leaked sessions
            logger.warning("[CuaRuntimeRail] end_session failed: %s", exc)


__all__ = ["CuaRuntimeRail"]

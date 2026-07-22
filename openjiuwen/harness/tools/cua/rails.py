# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Rails for the cua-driver desktop runtime."""

from __future__ import annotations

from typing import Optional, Sequence

from openjiuwen.core.foundation.tool import McpServerConfig
from openjiuwen.core.single_agent.rail.base import AgentCallbackContext, AgentRail


class CuaRuntimeRail(AgentRail):
    """Enforce the resolved cua capability allowlist on the driver MCP server.

    The allowlist is applied in ``before_invoke`` (after DeepAgent has
    registered the configured MCP servers) so the model never sees or executes
    driver tools outside the task-scoped capability selection.
    """

    def __init__(
        self,
        mcp_cfg: McpServerConfig,
        allowed_tool_names: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self._mcp_cfg = mcp_cfg
        self._allowed_tool_names = tuple(allowed_tool_names) if allowed_tool_names is not None else None

    async def before_invoke(self, ctx: AgentCallbackContext) -> None:
        self._ensure_cua_mcp_ability(ctx)

    def _ensure_cua_mcp_ability(self, ctx: AgentCallbackContext) -> None:
        agent = getattr(ctx, "agent", None)
        ability_manager = getattr(agent, "ability_manager", None)
        if ability_manager is None:
            return
        ability_manager.add(self._mcp_cfg)
        if self._allowed_tool_names is not None:
            ability_manager.set_mcp_tool_allowlist(self._mcp_cfg, self._allowed_tool_names)


__all__ = ["CuaRuntimeRail"]

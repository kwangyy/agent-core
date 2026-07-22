# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Factory helpers for the cua (computer-use) desktop subagent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from openjiuwen.core.common.logging import logger
from openjiuwen.core.foundation.llm.model import Model
from openjiuwen.core.foundation.tool import McpServerConfig, Tool, ToolCard
from openjiuwen.core.single_agent.rail.base import AgentRail
from openjiuwen.core.single_agent.schema.agent_card import AgentCard
from openjiuwen.core.sys_operation import SysOperation
from openjiuwen.harness.deep_agent import DeepAgent
from openjiuwen.harness.factory import create_deep_agent
from openjiuwen.harness.schema.config import SubAgentConfig
from openjiuwen.harness.tools.cua.config import build_cua_driver_mcp_config
from openjiuwen.harness.tools.cua.cua_capabilities import (
    DEFAULT_CUA_AGENT_CAPABILITY_NAMES,
    DEFAULT_CUA_CAPABILITIES,
    resolve_cua_capabilities,
)
from openjiuwen.harness.tools.cua.rails import CuaRuntimeRail

try:
    from openjiuwen.harness.prompts import resolve_language
except ImportError:

    def resolve_language(language: Optional[str] = None) -> str:  # type: ignore[misc]
        return language if language in {"cn", "en"} else "cn"


if TYPE_CHECKING:
    from openjiuwen.harness.workspace.workspace import Workspace


CUA_AGENT_FACTORY_NAME = "cua_agent"

DEFAULT_CUA_AGENT_SYSTEM_PROMPT_EN = (
    "You are a desktop automation agent that operates the host computer through cua-driver tools. "
    "Plan and decide at this agent level, then observe and act on real application windows. "
    "Perception: start with list_windows (or list_apps) to find the target pid and window_id, then "
    "call get_window_state(pid, window_id) to get the element tree plus a screenshot. Re-snapshot "
    "with get_window_state before every element-addressed action: element_index values are only "
    "valid against the latest snapshot of that window. Bound large trees with max_elements or "
    "max_depth, and pass include_screenshot=false when you only need to re-index elements. "
    "Acting: prefer element_index (with pid and window_id) over raw x,y pixels — element actions "
    "work on backgrounded windows, do not move the user's cursor, and tell you what you are acting "
    "on. Use x,y only for surfaces that do not appear in the element tree, reading coordinates "
    "straight off the latest screenshot. Keep the default delivery_mode 'background' so the user's "
    "focus is never stolen; escalate a single action to 'foreground' only after a background "
    "attempt verifiably failed. "
    "Verification: input actions are not self-verifying. After clicks, keys, or typed text, "
    "confirm the effect with a fresh get_window_state screenshot before claiming progress. "
    "Prefer launch_app to start applications (it does not steal focus); use kill_app only after "
    "the cooperative close path failed, since unsaved state is lost. "
    "Browser tasks are not yours: web page automation belongs to the browser agent — report back "
    "instead of driving a browser through desktop input. "
    "Avoid redundant actions, and only claim completion when the requested desktop outcome is "
    "actually evidenced on screen."
)

DEFAULT_CUA_AGENT_SYSTEM_PROMPT_CN = (
    "你是桌面自动化代理，通过 cua-driver 工具操作本机计算机。"
    "请在当前代理层面规划和决策，然后基于真实应用窗口进行观察和操作。"
    "感知：先用 list_windows（或 list_apps）找到目标 pid 和 window_id，"
    "再调用 get_window_state(pid, window_id) 获取元素树和截图。"
    "每次基于元素的操作前都要重新调用 get_window_state：element_index 只对该窗口最新一次快照有效。"
    "元素树过大时用 max_elements 或 max_depth 限制；只需重建索引时传 include_screenshot=false。"
    "操作：优先使用 element_index（配合 pid 和 window_id），而不是原始 x,y 像素坐标——"
    "元素级操作可作用于后台窗口、不会移动用户光标，并能明确操作对象。"
    "只有目标不在元素树中时才使用 x,y，坐标直接从最新截图上读取。"
    "保持默认 delivery_mode 'background'，绝不抢占用户焦点；"
    "只有后台尝试确认失败后，才对单个操作升级为 'foreground'。"
    "验证：输入类操作不会自我验证。点击、按键或输入文本后，"
    "必须用新的 get_window_state 截图确认效果，然后才能声明进展。"
    "启动应用优先使用 launch_app（不抢焦点）；kill_app 只在协作式关闭失败后使用，因为未保存状态会丢失。"
    "浏览器任务不属于你：网页自动化由浏览器代理负责——遇到此类任务应如实汇报，"
    "而不是通过桌面输入去驱动浏览器。"
    "避免重复动作；只有屏幕上有具体证据证明任务完成时，才声明完成。"
)

DEFAULT_CUA_AGENT_SYSTEM_PROMPT: Dict[str, str] = {
    "cn": DEFAULT_CUA_AGENT_SYSTEM_PROMPT_CN,
    "en": DEFAULT_CUA_AGENT_SYSTEM_PROMPT_EN,
}

DEFAULT_CUA_AGENT_DESCRIPTION_EN = (
    "Dedicated desktop subagent that controls host applications through cua-driver MCP tools."
)
DEFAULT_CUA_AGENT_DESCRIPTION_CN = "专用桌面子代理，通过 cua-driver MCP 工具操作本机应用程序。"
DEFAULT_CUA_AGENT_DESCRIPTION: Dict[str, str] = {
    "cn": DEFAULT_CUA_AGENT_DESCRIPTION_CN,
    "en": DEFAULT_CUA_AGENT_DESCRIPTION_EN,
}


def _resolve_capability_selection(cua_capabilities: Optional[List[str]]):
    """Validate the capability selection against the trusted catalog."""
    if cua_capabilities is not None and (
        not isinstance(cua_capabilities, list)
        or not all(isinstance(capability, str) for capability in cua_capabilities)
    ):
        raise ValueError("cua_capabilities must be a list of strings")

    requested = cua_capabilities if cua_capabilities is not None else list(DEFAULT_CUA_AGENT_CAPABILITY_NAMES)
    resolved = resolve_cua_capabilities(requested)
    if resolved.rejected_names:
        rejected = ", ".join(resolved.rejected_names)
        available = ", ".join(capability.name for capability in DEFAULT_CUA_CAPABILITIES)
        raise ValueError(f"Unsupported cua capabilities: {rejected}. Available capabilities: {available}")
    return resolved


def build_cua_agent_config(
    model: Model,
    *,
    card: Optional[AgentCard] = None,
    system_prompt: Optional[str] = None,
    tools: Optional[List[Tool | ToolCard]] = None,
    mcps: Optional[List[McpServerConfig]] = None,
    rails: Optional[List[AgentRail]] = None,
    enable_task_loop: bool = False,
    max_iterations: int = 25,
    workspace: Optional[str | "Workspace"] = None,
    skills: Optional[List[str]] = None,
    backend: Optional[Any] = None,
    sys_operation: Optional[SysOperation] = None,
    language: Optional[str] = None,
    prompt_mode: Optional[str] = None,
    cua_capabilities: Optional[List[str]] = None,
    cua_instance_key: str = "",
) -> SubAgentConfig:
    """Build a SubAgentConfig that materializes as create_cua_agent()."""
    resolved_language = resolve_language(language)
    _resolve_capability_selection(cua_capabilities)
    return SubAgentConfig(
        agent_card=card
        or AgentCard(
            name="cua_agent",
            description=DEFAULT_CUA_AGENT_DESCRIPTION.get(
                resolved_language,
                DEFAULT_CUA_AGENT_DESCRIPTION["cn"],
            ),
        ),
        system_prompt=system_prompt
        or DEFAULT_CUA_AGENT_SYSTEM_PROMPT.get(
            resolved_language,
            DEFAULT_CUA_AGENT_SYSTEM_PROMPT["cn"],
        ),
        tools=list(tools or []),
        mcps=list(mcps or []),
        model=model,
        rails=rails,
        skills=skills,
        backend=backend,
        workspace=workspace,
        sys_operation=sys_operation,
        language=resolved_language,
        prompt_mode=prompt_mode,
        enable_task_loop=enable_task_loop,
        max_iterations=max_iterations,
        factory_name=CUA_AGENT_FACTORY_NAME,
        factory_kwargs={
            "cua_capabilities": list(cua_capabilities) if cua_capabilities is not None else None,
            "cua_instance_key": cua_instance_key,
        },
    )


def create_cua_agent(
    model: Model,
    *,
    card: Optional[AgentCard] = None,
    system_prompt: Optional[str] = None,
    tools: Optional[List[Tool | ToolCard]] = None,
    mcps: Optional[List[McpServerConfig]] = None,
    subagents: Optional[List[SubAgentConfig | DeepAgent]] = None,
    rails: Optional[List[AgentRail]] = None,
    enable_task_loop: bool = False,
    max_iterations: int = 25,
    workspace: Optional[str | "Workspace"] = None,
    skills: Optional[List[str]] = None,
    backend: Optional[Any] = None,
    sys_operation: Optional[SysOperation] = None,
    language: Optional[str] = None,
    prompt_mode: Optional[str] = None,
    cua_capabilities: Optional[List[str]] = None,
    cua_instance_key: str = "",
    **config_kwargs: Any,
) -> DeepAgent:
    """Create the cua desktop subagent with a task-scoped tool allowlist.

    ``cua_capabilities`` is resolved against the trusted capability catalog;
    when omitted, the default grant (input + app_lifecycle on top of the
    always-included read-only core) applies. The resolved allowlist is
    enforced on the driver MCP server by :class:`CuaRuntimeRail`.
    """
    resolved_capabilities = _resolve_capability_selection(cua_capabilities)

    logger.info(
        "Resolved cua capabilities: requested=%s, selected=%s, allowed_tools=%s",
        resolved_capabilities.requested_names,
        resolved_capabilities.selected_names,
        resolved_capabilities.allowed_tool_names,
    )

    resolved_language = resolve_language(language)
    mcp_cfg = build_cua_driver_mcp_config(cua_instance_key)

    final_card = card or AgentCard(
        name="cua_agent",
        description=DEFAULT_CUA_AGENT_DESCRIPTION.get(
            resolved_language,
            DEFAULT_CUA_AGENT_DESCRIPTION["cn"],
        ),
    )
    final_prompt = system_prompt or DEFAULT_CUA_AGENT_SYSTEM_PROMPT.get(
        resolved_language,
        DEFAULT_CUA_AGENT_SYSTEM_PROMPT["cn"],
    )

    injected_rails: List[AgentRail] = [CuaRuntimeRail(mcp_cfg, resolved_capabilities.allowed_tool_names)]
    final_mcps = list(mcps or []) + [mcp_cfg]
    final_rails = list(rails or []) + injected_rails

    return create_deep_agent(
        model=model,
        card=final_card,
        system_prompt=final_prompt,
        tools=list(tools or []),
        mcps=final_mcps,
        subagents=subagents,
        rails=final_rails,
        enable_task_loop=enable_task_loop,
        max_iterations=max_iterations,
        workspace=workspace,
        skills=skills,
        backend=backend,
        sys_operation=sys_operation,
        language=resolved_language,
        prompt_mode=prompt_mode,
        **config_kwargs,
    )


__all__ = [
    "CUA_AGENT_FACTORY_NAME",
    "build_cua_agent_config",
    "create_cua_agent",
]

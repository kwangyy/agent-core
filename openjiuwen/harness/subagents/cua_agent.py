# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Factory helpers for the cua (computer-use) desktop subagent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from openjiuwen.core.common.logging import logger
from openjiuwen.core.context_engine import ToolResultWindowProcessorConfig
from openjiuwen.core.foundation.llm.model import Model
from openjiuwen.core.foundation.tool import McpServerConfig, Tool, ToolCard
from openjiuwen.core.single_agent.rail.base import AgentRail
from openjiuwen.core.single_agent.schema.agent_card import AgentCard
from openjiuwen.core.sys_operation import SysOperation
from openjiuwen.harness.deep_agent import DeepAgent
from openjiuwen.harness.factory import create_deep_agent
from openjiuwen.harness.rails.context_engineer import ContextProcessorRail
from openjiuwen.harness.rails.multimodal_context_summarizer_rail import MultimodalContextSummarizerRail
from openjiuwen.harness.schema.config import SubAgentConfig
from openjiuwen.harness.tools.cua.config import build_cua_driver_mcp_config
from openjiuwen.harness.tools.cua.cua_capabilities import (
    DEFAULT_CUA_AGENT_CAPABILITY_NAMES,
    DEFAULT_CUA_CAPABILITIES,
    resolve_cua_capabilities,
)
from openjiuwen.harness.tools.cua.rails import (
    CuaDeliveryModeRail,
    CuaElementAddressingRail,
    CuaRepeatFailureRail,
    CuaRuntimeRail,
    CuaScreenshotDownscaleRail,
    CuaSnapshotDedupRail,
    CuaSnapshotFreshnessRail,
)

try:
    from openjiuwen.harness.prompts import resolve_language
except ImportError:

    def resolve_language(language: Optional[str] = None) -> str:  # type: ignore[misc]
        return language if language in {"cn", "en"} else "cn"


if TYPE_CHECKING:
    from openjiuwen.harness.workspace.workspace import Workspace


CUA_AGENT_FACTORY_NAME = "cua_agent"

# Characters of a superseded snapshot left inline after it is offloaded.
# The placeholder also carries the offload handle, so the full result stays
# reachable; this only bounds the inline preview. The processor default of
# 3000 keeps most of an element tree whose element_index handles are already
# stale by construction -- ~19k tokens of dead weight over a 25-step run, and
# measured at only 3.6% net saving from the window processor because of it.
# 200 keeps the window header for orientation and drops the stale body.
_CUA_OFFLOAD_PREVIEW_CHARS = 200

DEFAULT_CUA_AGENT_SYSTEM_PROMPT_EN = (
    "You are a desktop automation agent that operates the host computer through cua-driver tools. "
    "Plan and decide at this agent level, then observe and act on real application windows. "
    "Perception: start with list_windows (or list_apps) to find the target pid and window_id, then "
    "call get_window_state(pid, window_id) to get the element tree plus a screenshot. Re-snapshot "
    "with get_window_state before every element-addressed action: element_index values are only "
    "valid against the latest snapshot of that window. Bound large trees with max_elements or "
    "max_depth, and pass include_screenshot=false when you only need to re-index elements. "
    "Electron and Chromium-based apps build their accessibility tree lazily: the first "
    "get_window_state on such a window can return a single bare Document. That does not mean the "
    "window is empty — snapshot the same window once more and the real tree appears. "
    "An element whose frame is null is not laid out on screen (typically virtualized out of a "
    "scrolling list); scroll it into view and re-snapshot rather than acting on it. "
    "A window whose x,y is near -32000 is minimized, not positioned off-screen; bring_to_front "
    "before trusting its geometry. "
    "Acting: prefer element_index (with pid and window_id) over raw x,y pixels — element actions "
    "work on backgrounded windows, do not move the user's cursor, and tell you what you are acting "
    "on. Use x,y only for surfaces that do not appear in the element tree, reading coordinates "
    "straight off the latest screenshot. Mind the two coordinate spaces: element frames from "
    "get_window_state are SCREEN-absolute pixels, while click/drag x,y are WINDOW-local pixels in "
    "the space of that window's screenshot. Never pass a frame's x,y straight into a click — "
    "subtract the window origin, or use scope='desktop' with screen coordinates. Addressing by "
    "element_index avoids the conversion entirely. "
    "When you act by x,y coordinates read off a screenshot, treat the target region's edges as "
    "unreliable: your visual estimate of a boundary can be off by tens of pixels, and input that "
    "starts outside the real region usually fails silently. Start clicks and drags well inside "
    "the region (30+ px from every estimated edge), and read a silent no-effect result as a "
    "likely aim miss. "
    "Keep the default delivery_mode 'background' so the user's "
    "focus is never stolen; escalate a single action to 'foreground' only after a background "
    "attempt verifiably failed. Do not pass 'foreground' preemptively because a target looks like "
    "a canvas or a Chromium surface — the driver reports when background delivery is impossible, "
    "and a guessed foreground action needlessly steals the user's focus. "
    "But once a background action reports success without verification or the follow-up "
    "snapshot shows nothing changed, do not retry it in background: one verified no-op is "
    "the signal. Escalate that action to 'foreground' on the second attempt. Modern "
    "packaged apps (UWP/WinUI -- Calculator, Settings) and some Chromium, Java, and Qt "
    "surfaces silently discard posted background input, so repeating it cannot succeed. "
    "Verification: input actions are not self-verifying. After clicks, keys, or typed text, "
    "confirm the effect with a fresh get_window_state screenshot before claiming progress. "
    "Prefer launch_app to start applications (it does not steal focus); use kill_app only after "
    "the cooperative close path failed, since unsaved state is lost. "
    "Browser tasks are not yours: web page automation belongs to the browser agent — report back "
    "instead of driving a browser through desktop input. This means an actual web browser such as "
    "Chrome, Edge, or Firefox. Electron desktop applications — Discord, Slack, VS Code, Spotify "
    "and the like — are ordinary native application windows even though Chromium renders them, "
    "and driving them IS your job; do not hand them to the browser agent, which cannot reach "
    "them. "
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
    "Electron 和基于 Chromium 的应用采用惰性构建无障碍树：对这类窗口首次调用 get_window_state "
    "可能只返回一个空的 Document。这不代表窗口是空的——对同一窗口再快照一次，真正的元素树就会出现。"
    "frame 为 null 的元素并未在屏幕上布局（通常是滚动列表虚拟化的结果）；"
    "应先滚动使其可见并重新快照，而不是直接对其操作。"
    "x,y 接近 -32000 的窗口是被最小化了，而不是位于屏幕外的真实位置；"
    "在信任其几何信息前先调用 bring_to_front。"
    "操作：优先使用 element_index（配合 pid 和 window_id），而不是原始 x,y 像素坐标——"
    "元素级操作可作用于后台窗口、不会移动用户光标，并能明确操作对象。"
    "只有目标不在元素树中时才使用 x,y，坐标直接从最新截图上读取。"
    "注意两套坐标系：get_window_state 返回的元素 frame 是屏幕绝对像素，"
    "而 click/drag 的 x,y 是该窗口截图空间内的窗口相对像素。"
    "绝不要把 frame 的 x,y 直接传给 click——应减去窗口原点，或使用 scope='desktop' 配合屏幕坐标。"
    "使用 element_index 寻址则完全无需换算。"
    "通过截图读取 x,y 坐标操作时，把目标区域的边缘当作不可靠信息：你对边界的视觉估计"
    "可能偏差数十像素，而起点落在真实区域之外的输入通常会静默失效。"
    "点击和拖拽的起点应落在区域内部足够深处（离每条估计边缘至少 30 像素）；"
    "操作后毫无效果时，优先怀疑是瞄准偏差。"
    "保持默认 delivery_mode 'background'，绝不抢占用户焦点；"
    "只有后台尝试确认失败后，才对单个操作升级为 'foreground'。"
    "不要因为目标看起来像 canvas 或 Chromium 界面就预先传 'foreground'——"
    "驱动会在后台投递不可行时报错，而擅自使用前台操作会无谓地抢走用户焦点。"
    "但一旦后台操作返回了未经验证的成功、或随后的快照显示界面毫无变化，就不要再用后台重试："
    "一次确认的无效果就是信号，第二次尝试就应将该操作升级为 'foreground'。"
    "UWP/WinUI 等现代打包应用（如计算器、设置）以及部分 Chromium、Java、Qt 界面"
    "会静默丢弃后台投递的输入，重复后台尝试不可能成功。"
    "验证：输入类操作不会自我验证。点击、按键或输入文本后，"
    "必须用新的 get_window_state 截图确认效果，然后才能声明进展。"
    "启动应用优先使用 launch_app（不抢焦点）；kill_app 只在协作式关闭失败后使用，因为未保存状态会丢失。"
    "浏览器任务不属于你：网页自动化由浏览器代理负责——遇到此类任务应如实汇报，"
    "而不是通过桌面输入去驱动浏览器。这里指的是真正的浏览器，例如 Chrome、Edge、Firefox。"
    "Discord、Slack、VS Code、Spotify 等 Electron 桌面应用虽然由 Chromium 渲染，"
    "但它们就是普通的原生应用窗口，操作它们正是你的职责；"
    "不要把它们交给浏览器代理，浏览器代理无法访问这些窗口。"
    "避免重复动作；只有屏幕上有具体证据证明任务完成时，才声明完成。"
)

DEFAULT_CUA_AGENT_SYSTEM_PROMPT: Dict[str, str] = {
    "cn": DEFAULT_CUA_AGENT_SYSTEM_PROMPT_CN,
    "en": DEFAULT_CUA_AGENT_SYSTEM_PROMPT_EN,
}

# Appended to the system prompt when a delivery mode is pinned. The base prompt
# teaches the model to manage delivery_mode itself (keep background, escalate
# after a verified failure); with CuaDeliveryModeRail enforcing a pin, that
# advice guarantees a rejected call per run, so the pin must supersede it.
CUA_DELIVERY_MODE_PROMPT_SUFFIX_EN: Dict[str, str] = {
    "background": (
        " Delivery override: the harness pins delivery_mode to 'background' for every input action. "
        "Do not pass delivery_mode yourself, and never attempt 'foreground' — such calls are rejected "
        "before they reach the driver. If the driver reports that background delivery is impossible "
        "for a surface, report that limitation as the task outcome instead of escalating. This "
        "override supersedes any earlier guidance about choosing or escalating delivery_mode."
    ),
    "foreground": (
        " Delivery override: the harness pins delivery_mode to 'foreground' for every input action. "
        "Do not pass delivery_mode yourself; actions are delivered via a foreground swap, and moving "
        "the user's focus is expected, not a failure. This override supersedes any earlier guidance "
        "about keeping 'background' or escalating only after a failed attempt."
    ),
}
CUA_DELIVERY_MODE_PROMPT_SUFFIX_CN: Dict[str, str] = {
    "background": (
        "投递模式覆盖：框架已将所有输入操作的 delivery_mode 固定为 'background'。"
        "不要自行传入 delivery_mode，也绝不要尝试 'foreground'——此类调用会在到达驱动前被拒绝。"
        "如果驱动报告某个界面无法后台投递，应将该限制如实作为任务结果汇报，而不是升级为前台。"
        "本覆盖优先于前文关于选择或升级 delivery_mode 的任何指引。"
    ),
    "foreground": (
        "投递模式覆盖：框架已将所有输入操作的 delivery_mode 固定为 'foreground'。"
        "不要自行传入 delivery_mode；操作通过前台切换投递，用户焦点移动是预期行为，不是失败。"
        "本覆盖优先于前文关于保持 'background' 或失败后才升级的任何指引。"
    ),
}
CUA_DELIVERY_MODE_PROMPT_SUFFIX: Dict[str, Dict[str, str]] = {
    "cn": CUA_DELIVERY_MODE_PROMPT_SUFFIX_CN,
    "en": CUA_DELIVERY_MODE_PROMPT_SUFFIX_EN,
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


def _validate_delivery_mode(cua_delivery_mode: Optional[str]) -> None:
    """Reject an unusable delivery mode at config time, not at first tool call."""
    if cua_delivery_mode is not None and cua_delivery_mode not in ("background", "foreground"):
        raise ValueError(f"cua_delivery_mode must be 'background', 'foreground', or None, got {cua_delivery_mode!r}")


def _validate_snapshot_keep_last_k(cua_snapshot_keep_last_k: int) -> None:
    """Reject an unusable retention count at config time, not at first tool call."""
    if (
        not isinstance(cua_snapshot_keep_last_k, int)
        or isinstance(cua_snapshot_keep_last_k, bool)
        or cua_snapshot_keep_last_k < 1
    ):
        raise ValueError(f"cua_snapshot_keep_last_k must be an int >= 1, got {cua_snapshot_keep_last_k!r}")


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
    cua_delivery_mode: Optional[Literal["background", "foreground"]] = None,
    cua_screenshot_multimodal: bool = False,
    cua_snapshot_keep_last_k: int = 3,
    permissions: Optional[dict] = None,
    permission_host: Optional[Any] = None,
) -> SubAgentConfig:
    """Build a SubAgentConfig that materializes as create_cua_agent().

    ``permissions`` / ``permission_host`` (optional) enable the tool-permission
    engine on the materialized subagent. Inside a task-tool delegation the
    built-in ASK interrupt cannot reach the outer conversation, so pair any
    policy that can produce ASK with a
    ``ToolPermissionHost.request_permission_confirmation`` hook that answers
    in-process; DENY rules need no host.
    """
    resolved_language = resolve_language(language)
    _resolve_capability_selection(cua_capabilities)
    _validate_delivery_mode(cua_delivery_mode)
    _validate_snapshot_keep_last_k(cua_snapshot_keep_last_k)
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
            "cua_delivery_mode": cua_delivery_mode,
            "cua_screenshot_multimodal": cua_screenshot_multimodal,
            "cua_snapshot_keep_last_k": cua_snapshot_keep_last_k,
            # Only present when provided so older configs keep their exact
            # factory_kwargs shape; DeepAgentConfig owns the defaults.
            **({"permissions": permissions} if permissions is not None else {}),
            **({"permission_host": permission_host} if permission_host is not None else {}),
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
    cua_delivery_mode: Optional[Literal["background", "foreground"]] = None,
    cua_screenshot_multimodal: bool = False,
    cua_snapshot_keep_last_k: int = 3,
    **config_kwargs: Any,
) -> DeepAgent:
    """Create the cua desktop subagent with a task-scoped tool allowlist.

    ``cua_capabilities`` is resolved against the trusted capability catalog;
    when omitted, the default grant (input + app_lifecycle on top of the
    always-included read-only core) applies. The resolved allowlist is
    enforced on the driver MCP server by :class:`CuaRuntimeRail`.

    ``cua_delivery_mode`` pins input delivery to ``background`` (never raise a
    window or steal focus) or ``foreground`` (always deliver via a foreground
    swap) through :class:`CuaDeliveryModeRail`. When omitted the driver's own
    per-call default applies and the model chooses. Pinning a mode also
    appends a matching delivery-override suffix to the system prompt so the
    base prompt's "manage delivery_mode yourself" guidance cannot steer the
    model into calls the rail is guaranteed to reject.

    ``cua_screenshot_multimodal`` attaches cua-driver screenshots to context as
    image input (requires a vision-capable ``model``). Off by default: the
    agent then perceives through element trees and structured content only,
    with screenshots reduced to text placeholders.

    ``cua_snapshot_keep_last_k`` is how many recent desktop snapshots stay in
    context: it drives both the tool-result window on the snapshot tools and,
    in multimodal mode, the number of retained screenshot images. Tasks that
    juggle two windows need both windows' latest frames in reach at once, so
    the default is 3; pass 1 for single-window tasks to spend the least
    context on perception history.
    """
    resolved_capabilities = _resolve_capability_selection(cua_capabilities)
    _validate_delivery_mode(cua_delivery_mode)
    _validate_snapshot_keep_last_k(cua_snapshot_keep_last_k)

    logger.info(
        "Resolved cua capabilities: requested=%s, selected=%s, allowed_tools=%s",
        resolved_capabilities.requested_names,
        resolved_capabilities.selected_names,
        resolved_capabilities.allowed_tool_names,
    )

    resolved_language = resolve_language(language)
    mcp_cfg = build_cua_driver_mcp_config(cua_instance_key, include_image_content=cua_screenshot_multimodal)

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
    # Appended even to a caller-supplied prompt: the rail enforces the pin
    # regardless of prompt, so the prompt must not teach the opposite. This is
    # also the only place the build_cua_agent_config path (which carries the
    # base prompt in the config) can pick the suffix up.
    if cua_delivery_mode is not None:
        final_prompt = (
            final_prompt
            + CUA_DELIVERY_MODE_PROMPT_SUFFIX.get(
                resolved_language,
                CUA_DELIVERY_MODE_PROMPT_SUFFIX["cn"],
            )[cua_delivery_mode]
        )

    injected_rails: List[AgentRail] = [
        CuaRuntimeRail(mcp_cfg, resolved_capabilities.allowed_tool_names),
        # Both are unconditional: they repair failure modes of the driver
        # contract itself, so there is no configuration under which the
        # agent is better off re-sending a call the driver already refused.
        CuaElementAddressingRail(mcp_cfg),
        CuaRepeatFailureRail(mcp_cfg),
        CuaSnapshotFreshnessRail(mcp_cfg),
        # Rides the same retention knob as the window processor below: the
        # rail must never collapse more consecutive snapshots than the
        # processor keeps in full, or the tree itself leaves context.
        CuaSnapshotDedupRail(mcp_cfg, keep_last_k=cua_snapshot_keep_last_k),
    ]

    # Only installed when the caller pinned a mode; otherwise delivery stays a
    # per-call model decision against the driver's own default.
    if cua_delivery_mode is not None:
        injected_rails.append(CuaDeliveryModeRail(mcp_cfg, cua_delivery_mode))

    if cua_screenshot_multimodal:
        injected_rails.append(CuaScreenshotDownscaleRail(mcp_cfg))
        # Screenshot UserMessages are not ToolMessages, so the window processor
        # below cannot evict them; retention rides the same knob so that a
        # two-window task keeps both windows' screens in reach, matching the
        # tool-result window below.
        injected_rails.append(MultimodalContextSummarizerRail(cua_snapshot_keep_last_k))

    # Window the large desktop perception results unless the caller already
    # manages context processors via their own ContextProcessorRail. Snapshot
    # tools emit huge payloads (element tree + base64 screenshot), and their
    # element_index handles go stale after every action, so old results lose
    # their actionable handles; they are persisted to the workspace offload
    # directory and replaced in context by a preview placeholder. Retention
    # defaults to 3, not 1: a task that works across two windows needs both
    # windows' latest snapshots in context at once (live-diagnosed — with
    # keep_last_k=1 the first window's frames evict the moment the second
    # window is snapshotted, and the agent loops on read-only re-queries).
    cua_windowed_tool_names = [
        "get_window_state",
        "get_desktop_state",
        "get_accessibility_tree",
    ]
    if not any(isinstance(rail, ContextProcessorRail) for rail in (rails or [])):
        injected_rails.append(
            ContextProcessorRail(
                processors=[
                    (
                        "ToolResultWindowProcessor",
                        ToolResultWindowProcessorConfig(
                            tool_names=cua_windowed_tool_names,
                            keep_last_k=cua_snapshot_keep_last_k,
                            trim_size=_CUA_OFFLOAD_PREVIEW_CHARS,
                        ),
                    )
                ],
                preset=False,
            )
        )
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

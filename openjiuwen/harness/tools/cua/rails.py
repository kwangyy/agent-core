# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Rails for the cua-driver desktop runtime."""

from __future__ import annotations

import asyncio
import base64
import collections
import hashlib
import io
import json
import re
import time
import uuid
from typing import Any, Awaitable, Callable, Literal, Optional, Sequence, Tuple

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import raise_error
from openjiuwen.core.common.logging import logger
from openjiuwen.core.foundation.llm import ToolMessage
from openjiuwen.core.foundation.tool import McpServerConfig
from openjiuwen.core.foundation.tool.mcp.base import mcp_model_tool_name, mcp_model_tool_prefix
from openjiuwen.core.single_agent.rail.base import AgentCallbackContext, AgentRail, ToolCallInputs
from openjiuwen.harness.tools.cua.user_activity import UserActivityProbe, build_default_user_activity_probe

_DAEMON_REMEDIATION = (
    "cua-driver daemon unreachable. Start it with `cua-driver autostart kick` "
    "(or `cua-driver serve` in an interactive desktop session) and retry."
)

# get_screen_size is cheap, read-only, and risk-classified on all platforms;
# health_report is denied under the driver's default permission mode
# ("no reviewed risk classification"), so it cannot serve as the probe.
_PROBE_TOOL = "get_screen_size"

# cua-driver tools that accept a delivery_mode argument. Verified against
# `cua-driver describe <tool>` (0.10.0): set_value, launch_app, kill_app and
# bring_to_front take no delivery_mode, so they are deliberately absent even
# though set_value ships in the same "input" capability bundle.
_DELIVERY_MODE_TOOL_NAMES: tuple[str, ...] = (
    "click",
    "double_click",
    "drag",
    "hotkey",
    "press_key",
    "right_click",
    "scroll",
    "type_text",
)

# cua-driver tools that inject real input events. Under foreground delivery
# these register with the OS exactly like a human's, so the takeover rail must
# be able to tell its own recent input apart from the user's.
_INPUT_TOOL_NAMES: tuple[str, ...] = _DELIVERY_MODE_TOOL_NAMES + ("set_value", "move_cursor")

_USER_TAKEOVER_BLOCKED = (
    "Paused for {waited:.0f}s because the user is actively using the desktop (mouse/keyboard "
    "input detected) and has not stopped. Do not send further desktop actions. Report that the "
    "user has taken over the desktop, summarize what was completed so far, and state that the "
    "task can be resumed once they hand control back."
)
_USER_TAKEOVER_RESUMED_NOTE = (
    "Note: this action was held for {waited:.0f}s because the user was using the desktop; it ran "
    "once the desktop had been idle for {idle:.0f}s. The screen may have changed meanwhile -- "
    "re-snapshot with get_window_state before element-addressed actions."
)

_BACKGROUND_ONLY_REJECTION = (
    "This desktop agent is configured background-only; delivery_mode 'foreground' is not "
    "available. Re-issue the action without delivery_mode (background is the default). If "
    "the driver reports that background delivery is impossible for this target, report that "
    "back instead of escalating — foreground input is disabled for this run."
)


# Driver-side success marker. cua-driver prefixes a confirmed action with it;
# failures and unverified deliveries ("Sent ... (not verified)") do not carry
# it. Verified against 165 recorded runs: every identical call repeated >=5
# times legitimately (press_key tab x27, hotkey ctrl+a x16) is marked, and the
# only unmarked high-repeat call is a genuine stuck loop.
_DRIVER_SUCCESS_MARKER = "✅"

# Volatile argument keys excluded from the repeat-identity of a call: they
# change between runs without changing what the call does.
_REPEAT_IDENTITY_IGNORED_ARGS = frozenset({"session"})

_REPEAT_ADVISORY = (
    "This exact call has now returned this same non-success response {count} times. "
    "Repeating it again will not change the result. Re-snapshot the window with "
    "get_window_state and address the target differently, or report what is blocking you."
)

_SNAPSHOT_TOOLS = frozenset({"get_window_state", "get_accessibility_tree"})

# cua tools that observe without mutating window state or the element cache.
_FRESHNESS_NEUTRAL_TOOLS = frozenset(
    {
        "get_desktop_state",
        "list_windows",
        "list_apps",
        "get_cursor_position",
        "get_screen_size",
        "get_session_state",
        "zoom",
        "health_report",
        "check_permissions",
        "start_session",
        "end_session",
    }
)

_STALE_SNAPSHOT_ADVISORY = (
    "Note: this element_index came from a snapshot taken BEFORE you last acted on this window, "
    "so the element tree may have changed underneath it. Call get_window_state(pid, window_id) "
    "again and retry with a fresh element_index before trying anything else."
)

# Fields the driver regenerates on every snapshot even when nothing changed:
# screenshot bytes (cursor blink / encoder noise), the monotonic snapshot_id
# counter, and -- MCP structuredContent only, the CLI payload has no token
# fields -- per-element element_token values ("s0016:0"), which embed the
# snapshot_id and therefore ALL rotate on every capture. Verified live
# 2026-08-25: consecutive idle-window payloads differed in exactly these
# fields. Matched tolerantly (JSON or key=value text) so the comparison works
# whatever text rendering the MCP layer picks.
_SNAPSHOT_VOLATILE_FIELD_RE = re.compile(
    r'("?(?:screenshot_png_b64|snapshot_id|element_token)"?\s*[:=]\s*)"?[A-Za-z0-9+/=_.:\-]*"?'
)

# The MCP bridge replaces the screenshot block with a text placeholder that
# embeds the base64 length -- '[image content: image/png, 51439 base64
# chars]' -- and that length changes with every capture (encoder noise), so
# it must be normalized away too. Verified via StdioClient: with this and
# the field normalization, consecutive idle-window MCP texts are identical.
_SNAPSHOT_IMAGE_PLACEHOLDER_RE = re.compile(r"\[image content: [^\]]{0,80}\]")

_SNAPSHOT_UNCHANGED_NOTE = (
    _DRIVER_SUCCESS_MARKER + " Unchanged: this {tool} snapshot of pid={pid}, window_id={window_id} "
    "matches your previous one ({chars} chars elided; only volatile fields differ: screenshot "
    "bytes, snapshot_id, element_token values). The element tree and every element_index from "
    "that snapshot are still valid; address elements by element_index, not element_token."
)

_SELF_CHANGED_WINDOW_NOTE = (
    "Note: this window's tree changed since your previous snapshot even though you performed no "
    "action in between -- the app updates its own UI (loading, animation, async content). "
    "Re-snapshot immediately before element actions on this window; element_index values here "
    "can go stale on their own."
)

_REPEAT_BLOCKED = (
    "Blocked: this exact call has already returned the same non-success response {count} times "
    "in this run and is not being sent again. The approach is not working — re-snapshot with "
    "get_window_state and change the arguments or the strategy, or report that the step is "
    "blocked. Repeating this call verbatim will stay blocked."
)

# CuaProgressRail thresholds -- unlike every other threshold in this file,
# these are NOT corpus-validated yet: no recorded cua run traces exist for
# the state-revisit signal. Treat as a reasonable starting point, not a
# measured constant, until real runs can confirm or retune it.
_STATE_REVISIT_ADVISE_AFTER = 3
_STATE_REVISIT_HISTORY_SIZE = 8
_CUA_BLOCKER_NOTE_MAX = 200
_CUA_MAX_TRACKED_BLOCKERS = 5
_CUA_PROGRESS_STATE_KEY = "cua_agent_progress_state"

_STATE_REVISIT_ADVISORY = (
    "Note: this window's content matches a state you already visited earlier in this run (not "
    "just the last snapshot -- {revisits} times now). Repeating the same approach is cycling, "
    "not progressing. Re-snapshot, try a materially different strategy, or report what is "
    "blocking you instead of continuing the same sequence."
)


class CuaRuntimeRail(AgentRail):
    """Lifecycle rail for the cua-driver MCP runtime.

    Run-scoped setup, cached after first success: verify the daemon is
    reachable (fail loud with remediation), enforce the resolved capability
    allowlist on the driver MCP server, and declare a driver session so
    concurrent agent runs get distinct cursor identities and session-scoped
    driver state.

    ``before_invoke`` fires on every entry into the agent loop, and the
    security gate's approval path re-enters it before each gated call.
    Uncached, that repeated the probe subprocess, the ability registration
    and a start_session/end_session pair per approval: across 73 recorded
    runs, 38.9% of all driver calls were session lifecycle (~7 pairs per
    run, 57-59% provable no-ops). Now only the first successful pass does
    work, and the session is left to the daemon's idle TTL -- the reclaim
    path the driver already treats as normal -- instead of an explicit
    end_session.
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
        self._daemon_verified = False
        self._ability_registered = False

    async def before_invoke(self, ctx: AgentCallbackContext) -> None:
        # Each step caches its own first success; only a failed step is
        # retried on the next entry into the loop.
        if not self._daemon_verified:
            await self._ensure_daemon_reachable()
            self._daemon_verified = True
        self._ensure_cua_mcp_ability(ctx)
        await self._start_driver_session(ctx)

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
        if self._ability_registered:
            # Re-adding is what produced the "Duplicate MCP server ability"
            # warning once per gated call on the approval path.
            return
        agent = getattr(ctx, "agent", None)
        ability_manager = getattr(agent, "ability_manager", None)
        if ability_manager is None:
            return
        ability_manager.add(self._mcp_cfg)
        if self._allowed_tool_names is not None:
            ability_manager.set_mcp_tool_allowlist(self._mcp_cfg, self._allowed_tool_names)
        self._ability_registered = True

    def _driver_tool(self, tool_name: str) -> Optional[Any]:
        from openjiuwen.core.runner.runner import Runner

        tool_id = f"{self._mcp_cfg.server_id}.{self._mcp_cfg.server_name}.{tool_name}"
        return Runner.resource_mgr.get_tool(tool_id)

    async def _start_driver_session(self, ctx: AgentCallbackContext) -> None:
        """Declare a driver session once. Cosmetic identity: failure never blocks the run."""
        if self._session_key is not None:
            return
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


def _normalize_tool_args(raw_args: Any) -> Tuple[Optional[dict], bool]:
    """Return tool args as a dict plus whether they arrived JSON-encoded.

    ``tool_args`` reaches ``before_tool_call`` as the raw ``ToolCall.arguments``
    value, which is a JSON string until the ability manager parses it. The
    caller must write the mutated args back in the same shape it received.
    """
    if isinstance(raw_args, dict):
        return dict(raw_args), False
    if isinstance(raw_args, str) and raw_args.strip():
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError:
            return None, True
        return (parsed, True) if isinstance(parsed, dict) else (None, True)
    return None, False


def _response_text(inputs: ToolCallInputs) -> Optional[str]:
    """Read the driver's response text, preferring the built ToolMessage."""
    content = getattr(inputs.tool_msg, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    data = getattr(inputs.tool_result, "data", None)
    if isinstance(data, dict) and isinstance(data.get("content"), str):
        return data["content"].strip() or None
    if isinstance(inputs.tool_result, dict) and isinstance(inputs.tool_result.get("content"), str):
        return inputs.tool_result["content"].strip() or None
    return None


def _append_advisory(inputs: ToolCallInputs, advisory: str) -> None:
    """Annotate a tool result in place; the ToolMessage already exists in after_tool_call."""
    suffix = "\n\n" + advisory
    data = getattr(inputs.tool_result, "data", None)
    if isinstance(data, dict) and isinstance(data.get("content"), str):
        data["content"] += suffix
    tool_msg = inputs.tool_msg
    if tool_msg is not None and isinstance(getattr(tool_msg, "content", None), str):
        tool_msg.content += suffix


def _set_content(inputs: ToolCallInputs, text: str) -> None:
    """Replace a tool result's text in place (both stores, str content only)."""
    data = getattr(inputs.tool_result, "data", None)
    if isinstance(data, dict) and isinstance(data.get("content"), str):
        data["content"] = text
    tool_msg = inputs.tool_msg
    if tool_msg is not None and isinstance(getattr(tool_msg, "content", None), str):
        tool_msg.content = text


class CuaDeliveryModeRail(AgentRail):
    """Pin cua-driver input delivery to a single mode.

    ``delivery_mode`` is a per-call cua-driver argument that decides whether
    input is posted to a window without raising it (``background``) or
    delivered via a brief foreground swap (``foreground``). The driver defaults
    to ``background``, but nothing stops a model from passing ``foreground``
    preemptively and stealing the user's focus.

    This rail makes the choice an operator setting rather than a model one:

    - The mode is injected into every delivery-capable call that did not
      already request it, so a configured mode also holds for calls that simply
      omit the argument (required for ``foreground``, whose absence would
      otherwise fall back to the driver's ``background`` default).
    - Under ``background``, an explicit ``foreground`` request is rejected
      rather than rewritten. The driver's own ``background_unavailable`` error
      tells the model to retry with ``foreground``; silently rewriting that
      retry back to ``background`` would loop the model against the driver's
      advice with no signal that its escalation was discarded.

    Not installed unless a mode is configured, so the default agent keeps the
    driver's own behavior.
    """

    def __init__(
        self,
        mcp_cfg: McpServerConfig,
        delivery_mode: Literal["background", "foreground"],
    ) -> None:
        super().__init__()
        if delivery_mode not in ("background", "foreground"):
            raise ValueError(f"cua delivery_mode must be 'background' or 'foreground', got {delivery_mode!r}")
        self._delivery_mode = delivery_mode
        # Build the model-facing names from the server name so instance-isolated
        # registrations (cua_instance_key) are matched too.
        self._tool_names = frozenset(
            mcp_model_tool_name(mcp_cfg.server_name, tool_name) for tool_name in _DELIVERY_MODE_TOOL_NAMES
        )

    async def before_tool_call(self, ctx: AgentCallbackContext) -> None:
        inputs = ctx.inputs
        if not isinstance(inputs, ToolCallInputs):
            return
        tool_name = str(inputs.tool_name or "")
        if tool_name not in self._tool_names:
            return

        args, was_json_encoded = _normalize_tool_args(inputs.tool_args)
        if args is None:
            return

        requested = args.get("delivery_mode")
        if self._delivery_mode == "background" and requested == "foreground":
            logger.info("[CuaDeliveryModeRail] Rejected foreground escalation for %s", tool_name)
            self._reject_tool(ctx, inputs, _BACKGROUND_ONLY_REJECTION)
            return

        if requested == self._delivery_mode:
            return

        if requested is not None:
            logger.info(
                "[CuaDeliveryModeRail] Overrode delivery_mode %r with %r for %s",
                requested,
                self._delivery_mode,
                tool_name,
            )
        args["delivery_mode"] = self._delivery_mode
        inputs.tool_args = json.dumps(args) if was_json_encoded else args

    @staticmethod
    def _reject_tool(ctx: AgentCallbackContext, inputs: ToolCallInputs, error_msg: str) -> None:
        """Hard-block a tool call using the shared rail contract."""
        tool_call = inputs.tool_call
        tool_call_id = tool_call.id if tool_call else ""
        ctx.extra["_skip_tool"] = True
        inputs.tool_result = {"error": error_msg}
        inputs.tool_msg = ToolMessage(content=error_msg, tool_call_id=tool_call_id)


class CuaUserTakeoverRail(AgentRail):
    """Pause desktop actions while the user is using the machine.

    "Once the user takes over the screen, the agent should pause." The driver
    has no such signal, so the rail asks the OS (``UserActivityProbe``) how
    long ago the last mouse/keyboard event happened, and holds every
    cua-driver call while that is recent:

    - the user counts as active when the desktop was touched within
      ``active_within_s``;
    - a held call proceeds once the desktop has been idle for
      ``resume_after_idle_s`` (pause-and-wait, not stop), and the tool result
      is annotated so the model re-snapshots before trusting old indices;
    - after ``max_wait_s`` of continuous user activity the call is rejected
      with an instruction to report the takeover, so a delegation never sits
      silent forever.

    Under foreground delivery the agent's own clicks and keystrokes are real
    OS input too. An input event whose age matches the agent's last own input
    action (within ``self_input_grace_s``) is attributed to the agent, not the
    user. Without a probe (non-Windows hosts) the rail is inert.
    """

    def __init__(
        self,
        mcp_cfg: McpServerConfig,
        *,
        probe: Optional[UserActivityProbe] = None,
        active_within_s: float = 3.0,
        resume_after_idle_s: float = 5.0,
        max_wait_s: float = 60.0,
        poll_interval_s: float = 0.5,
        self_input_grace_s: float = 1.5,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        if active_within_s <= 0 or resume_after_idle_s <= 0 or max_wait_s <= 0 or poll_interval_s <= 0:
            raise ValueError("cua takeover thresholds must be positive seconds")
        self._tool_prefix = mcp_model_tool_prefix(mcp_cfg.server_name)
        self._input_tool_names = frozenset(
            mcp_model_tool_name(mcp_cfg.server_name, tool_name) for tool_name in _INPUT_TOOL_NAMES
        )
        self._probe = probe if probe is not None else build_default_user_activity_probe()
        self._active_within_s = float(active_within_s)
        self._resume_after_idle_s = float(resume_after_idle_s)
        self._max_wait_s = float(max_wait_s)
        self._poll_interval_s = float(poll_interval_s)
        self._self_input_grace_s = float(self_input_grace_s)
        self._sleep = sleep
        self._monotonic = monotonic
        self._last_own_input_at: Optional[float] = None
        self._pending_note: Optional[str] = None

    @property
    def active(self) -> bool:
        """Whether a probe is available, i.e. the rail can actually observe the user."""
        return self._probe is not None

    async def before_tool_call(self, ctx: AgentCallbackContext) -> None:
        inputs = ctx.inputs
        if not isinstance(inputs, ToolCallInputs) or self._probe is None:
            return
        tool_name = str(inputs.tool_name or "")
        if not tool_name.startswith(self._tool_prefix) or ctx.extra.get("_skip_tool"):
            return

        waited = 0.0
        idle = self._user_idle_seconds()
        if idle is None or idle >= self._active_within_s:
            return
        logger.warning(
            "[CuaUserTakeoverRail] user is using the desktop (idle %.1fs); holding %s", idle, tool_name
        )
        while True:
            if waited >= self._max_wait_s:
                logger.warning(
                    "[CuaUserTakeoverRail] user still active after %.0fs; rejecting %s", waited, tool_name
                )
                self._reject_tool(ctx, inputs, _USER_TAKEOVER_BLOCKED.format(waited=waited))
                return
            await self._sleep(self._poll_interval_s)
            waited += self._poll_interval_s
            idle = self._user_idle_seconds()
            if idle is None or idle >= self._resume_after_idle_s:
                break
        logger.info("[CuaUserTakeoverRail] desktop idle again after %.1fs; releasing %s", waited, tool_name)
        self._pending_note = _USER_TAKEOVER_RESUMED_NOTE.format(
            waited=waited, idle=self._resume_after_idle_s
        )

    async def after_tool_call(self, ctx: AgentCallbackContext) -> None:
        inputs = ctx.inputs
        if not isinstance(inputs, ToolCallInputs):
            return
        tool_name = str(inputs.tool_name or "")
        if not tool_name.startswith(self._tool_prefix):
            return
        if tool_name in self._input_tool_names:
            self._last_own_input_at = self._monotonic()
        if self._pending_note is not None:
            _append_advisory(inputs, self._pending_note)
            self._pending_note = None

    def _user_idle_seconds(self) -> Optional[float]:
        idle = self._probe.seconds_since_last_input()
        if idle is None:
            return None
        if self._last_own_input_at is not None:
            since_own = self._monotonic() - self._last_own_input_at
            # The last OS input event coincides with our own injected action:
            # that is the agent, not the user. Anything more recent is the user.
            if abs(idle - since_own) <= self._self_input_grace_s:
                return float("inf")
        return idle

    @staticmethod
    def _reject_tool(ctx: AgentCallbackContext, inputs: ToolCallInputs, error_msg: str) -> None:
        tool_call = inputs.tool_call
        tool_call_id = tool_call.id if tool_call else ""
        ctx.extra["_skip_tool"] = True
        inputs.tool_result = {"error": error_msg}
        inputs.tool_msg = ToolMessage(content=error_msg, tool_call_id=tool_call_id)


class CuaScreenshotDownscaleRail(AgentRail):
    """Downscale oversized cua-driver screenshots before they enter context.

    A full-desktop PNG can reach millions of base64 chars; re-encode any
    multimodal image over the token budget as a bounded JPEG. Mutates the
    ``multimodal`` items of ``ctx.inputs.tool_result`` in place — the same
    object the ability manager returns to the agent, so reassigning
    ``tool_result`` here would silently not propagate.
    """

    # base64 chars × 0.125 ≈ tokens, and the same budget as ReadFileTool's
    # image path (filesystem.py); duplicated rather than shared to keep this
    # rail free of the filesystem tool's sys_operation coupling.
    _MAX_IMAGE_TOKENS = 25_000

    def __init__(
        self,
        mcp_cfg: McpServerConfig,
        *,
        max_edge: int = 1536,
        jpeg_quality: int = 70,
    ) -> None:
        super().__init__()
        self._max_edge = max_edge
        self._jpeg_quality = jpeg_quality
        # Prefix-match the server's model-facing tool names so instance-isolated
        # registrations (cua_instance_key) are matched too.
        self._tool_prefix = mcp_model_tool_prefix(mcp_cfg.server_name)

    async def after_tool_call(self, ctx: AgentCallbackContext) -> None:
        inputs = ctx.inputs
        if not isinstance(inputs, ToolCallInputs):
            return
        if not str(inputs.tool_name or "").startswith(self._tool_prefix):
            return
        data = getattr(inputs.tool_result, "data", None)
        if not isinstance(data, dict):
            return
        items = data.get("multimodal")
        if not isinstance(items, list):
            return
        scale_notes = []
        for item in items:
            if not isinstance(item, dict) or item.get("type") != "image":
                continue
            data_url = item.get("data_url")
            if not isinstance(data_url, str) or not data_url.startswith("data:image/"):
                continue
            if len(data_url) * 0.125 <= self._MAX_IMAGE_TOKENS:
                continue
            downscaled = self._downscale_data_url(data_url)
            if downscaled is not None:
                new_url, (orig_w, orig_h), (new_w, new_h) = downscaled
                item["data_url"] = new_url
                item["mime_type"] = "image/jpeg"
                if new_w and new_w < orig_w:
                    scale_notes.append(
                        f"[screenshot scaled] The attached screenshot is {new_w}x{new_h}, downscaled "
                        f"from the window's true {orig_w}x{orig_h}. Any x,y you read off the screenshot "
                        f"must be multiplied by {orig_w / new_w:.2f} to get window coordinates before "
                        f"click/drag. element_index addressing needs no conversion."
                    )
        if scale_notes:
            self._append_scale_notes(inputs, data, scale_notes)

    @staticmethod
    def _append_scale_notes(inputs: ToolCallInputs, data: dict, scale_notes: list) -> None:
        """Surface the scale factor to the model or its screenshot reads misclick.

        The ToolMessage is built before after_tool_call rails run, so the note
        must be appended to the existing message object in place; updating only
        ``data["content"]`` would never reach the model.
        """
        suffix = "\n\n" + "\n".join(scale_notes)
        if isinstance(data.get("content"), str):
            data["content"] += suffix
        tool_msg = inputs.tool_msg
        if tool_msg is not None and isinstance(getattr(tool_msg, "content", None), str):
            tool_msg.content += suffix

    def _downscale_data_url(self, data_url: str) -> Optional[tuple]:
        """Return ``(jpeg_data_url, (orig_w, orig_h), (new_w, new_h))``, or None."""
        try:
            from PIL import Image
        except ImportError as exc:
            logger.debug("[CuaScreenshotDownscaleRail] Pillow unavailable: %s", exc)
            return None

        try:
            _, encoded = data_url.split(",", 1)
            raw = base64.b64decode(encoded)
        except ValueError as exc:  # binascii.Error is a ValueError subclass
            logger.debug("[CuaScreenshotDownscaleRail] malformed data URL: %s", exc)
            return None

        # Ladder mirrors ReadFileTool._read_image: bounded resize first, then
        # an aggressive fallback if the budget is still exceeded.
        for max_edge, quality in ((self._max_edge, self._jpeg_quality), (800, 40)):
            try:
                with Image.open(io.BytesIO(raw)) as img:
                    orig_size = img.size
                    img = img.convert("RGB")
                    img.thumbnail((max_edge, max_edge))
                    new_size = img.size
                    out = io.BytesIO()
                    img.save(out, format="JPEG", quality=quality)
            except (OSError, ValueError) as exc:
                logger.debug("[CuaScreenshotDownscaleRail] re-encode failed: %s", exc)
                return None
            candidate = base64.b64encode(out.getvalue()).decode("ascii")
            if len(candidate) * 0.125 <= self._MAX_IMAGE_TOKENS:
                return f"data:image/jpeg;base64,{candidate}", orig_size, new_size
        # Over budget even at the aggressive rung: ship the smallest attempt.
        return f"data:image/jpeg;base64,{candidate}", orig_size, new_size


class CuaSnapshotFreshnessRail(AgentRail):
    """Attach a re-snapshot hint when a stale-window element action fails.

    The driver scopes the element_index cache per (pid, window_id) and replaces
    it on every snapshot, and the agent prompt says to re-snapshot before every
    element action -- but the recorded corpus shows the rule is routinely and
    successfully broken: 357 element actions issued after the window was acted
    on succeeded (Windows trees are stable across consecutive clicks), while 70
    failed. A hard browser-style generation gate would therefore block far more
    legitimate work than it saved, and the driver already rejects truly stale
    element_tokens with its own actionable message.

    So this rail never blocks. It tracks, per (pid, window_id), whether the
    window has been acted on since its last snapshot, and when an
    element-indexed action on such a dirty window comes back without the driver
    success marker, it appends the one hint that resolves the likeliest cause:
    re-snapshot and re-index. The 70 corpus failures would each have carried
    the hint on their first failure instead of waiting for the repeat-failure
    advisory at three.
    """

    def __init__(self, mcp_cfg: McpServerConfig) -> None:
        super().__init__()
        self._tool_prefix = mcp_model_tool_prefix(mcp_cfg.server_name)
        # (pid, window_id) -> True while the latest snapshot is still fresh.
        self._fresh: dict = {}

    async def before_invoke(self, ctx: AgentCallbackContext) -> None:
        self._fresh.clear()

    async def after_tool_call(self, ctx: AgentCallbackContext) -> None:
        inputs = ctx.inputs
        if not isinstance(inputs, ToolCallInputs):
            return
        tool_name = str(inputs.tool_name or "")
        if not tool_name.startswith(self._tool_prefix):
            return
        if ctx.extra.get("_skip_tool"):
            # A rail-side rejection never reached the driver: window state and
            # the element cache are exactly what they were.
            return
        short_name = tool_name[len(self._tool_prefix) :]
        args, _ = _normalize_tool_args(inputs.tool_args)
        if args is None:
            return
        key = (args.get("pid"), args.get("window_id"))

        if short_name in _SNAPSHOT_TOOLS:
            self._fresh[key] = True
            return
        if short_name in _FRESHNESS_NEUTRAL_TOOLS:
            return

        was_fresh = self._fresh.get(key, False)
        self._fresh[key] = False
        if args.get("element_index") is None or was_fresh:
            return
        response = _response_text(inputs)
        if response is None or response.startswith(_DRIVER_SUCCESS_MARKER):
            return
        if "get_window_state" in response:
            # The driver (or another rail) already told the model to re-snapshot.
            return
        logger.info(
            "[CuaSnapshotFreshnessRail] %s failed on a window acted on since its last snapshot",
            tool_name,
        )
        _append_advisory(inputs, _STALE_SNAPSHOT_ADVISORY)


class CuaSnapshotDedupRail(AgentRail):
    """Collapse byte-identical repeat snapshots; flag windows that change by themselves.

    A perceive-act-verify loop re-reads the same window constantly, and Windows
    element trees are stable, so consecutive ``get_window_state`` calls often
    return the exact same payload -- each a full element tree that the
    ToolResultWindowProcessor keeps verbatim while it is inside the retention
    window. When a snapshot's text matches the previous snapshot of the same
    (tool, pid, window_id) -- compared after stripping the fields the driver
    regenerates every call (screenshot bytes, snapshot_id), which otherwise
    make every payload unique -- this rail replaces it with a one-line
    "unchanged" note: the driver has still refreshed its element cache, so the
    indices from the retained earlier copy stay valid.

    At most ``keep_last_k - 1`` consecutive snapshots are collapsed per key:
    the window processor keeps only the newest ``keep_last_k`` snapshot results
    in full, so a longer run of notes could push every full copy of the tree
    out of context. With ``keep_last_k=1`` the rail never collapses anything.

    The same digest comparison detects the opposite case for free: a snapshot
    that DIFFERS from the previous one although no cua action ran in between
    means the app mutates its own UI (loading screens, async content). That is
    exactly the dynamic case the action-based CuaSnapshotFreshnessRail cannot
    see, so the first such change per window gets a warning that element
    indices there go stale on their own.
    """

    def __init__(self, mcp_cfg: McpServerConfig, *, keep_last_k: int = 3) -> None:
        super().__init__()
        if not isinstance(keep_last_k, int) or isinstance(keep_last_k, bool) or keep_last_k < 1:
            raise ValueError(f"keep_last_k must be an int >= 1, got {keep_last_k!r}")
        self._tool_prefix = mcp_model_tool_prefix(mcp_cfg.server_name)
        self._max_consecutive = keep_last_k - 1
        # (short_name, pid, window_id) -> (digest of last driver payload, action count when taken)
        self._last: dict = {}
        # (short_name, pid, window_id) -> consecutive snapshots collapsed so far
        self._collapsed: dict = {}
        self._self_change_noted: set = set()
        self._action_count = 0

    async def before_invoke(self, ctx: AgentCallbackContext) -> None:
        self._last.clear()
        self._collapsed.clear()
        self._self_change_noted.clear()
        self._action_count = 0

    async def after_tool_call(self, ctx: AgentCallbackContext) -> None:
        inputs = ctx.inputs
        if not isinstance(inputs, ToolCallInputs):
            return
        tool_name = str(inputs.tool_name or "")
        if not tool_name.startswith(self._tool_prefix):
            return
        if ctx.extra.get("_skip_tool"):
            # A rail-side rejection never reached the driver.
            return
        short_name = tool_name[len(self._tool_prefix) :]
        if short_name in _FRESHNESS_NEUTRAL_TOOLS:
            return
        args, _ = _normalize_tool_args(inputs.tool_args)
        if args is None:
            return
        if short_name not in _SNAPSHOT_TOOLS:
            self._action_count += 1
            return

        key = (short_name, args.get("pid"), args.get("window_id"))
        response = _response_text(inputs)
        # Snapshot payloads do NOT carry the driver's action-success marker --
        # a real get_window_state text starts with a plain header line
        # ("window_id=... pid=... elements=32"). The usable-baseline gate is
        # therefore the presence of tree content, which driver error texts
        # ("window not found", stale-token rejections) never contain. This
        # also keeps identical *errors* from being collapsed into a note that
        # would wrongly claim the tree is intact.
        if (
            response is None
            or ("element_index" not in response and "element_count" not in response)
            or not isinstance(getattr(inputs.tool_msg, "content", None), str)
        ):
            # A failed (or non-textual) snapshot leaves no baseline to compare
            # against; the next success must go through in full.
            self._last.pop(key, None)
            self._collapsed.pop(key, None)
            return

        comparable = _SNAPSHOT_VOLATILE_FIELD_RE.sub(r"\1<volatile>", response)
        comparable = _SNAPSHOT_IMAGE_PLACEHOLDER_RE.sub("[image content: <volatile>]", comparable)
        digest = hashlib.sha256(comparable.encode("utf-8")).hexdigest()
        previous = self._last.get(key)
        self._last[key] = (digest, self._action_count)

        if previous is not None and previous[0] == digest:
            if self._collapsed.get(key, 0) >= self._max_consecutive:
                # Let a full copy through so the tree stays reachable inside
                # the window processor's keep_last_k retention.
                self._collapsed[key] = 0
                return
            self._collapsed[key] = self._collapsed.get(key, 0) + 1
            logger.info(
                "[CuaSnapshotDedupRail] Collapsed identical %s (%d chars) for pid=%r window_id=%r",
                short_name,
                len(response),
                args.get("pid"),
                args.get("window_id"),
            )
            _set_content(
                inputs,
                _SNAPSHOT_UNCHANGED_NOTE.format(
                    tool=short_name,
                    pid=args.get("pid"),
                    window_id=args.get("window_id"),
                    chars=len(response),
                ),
            )
            return

        self._collapsed[key] = 0
        if previous is not None and previous[1] == self._action_count and key not in self._self_change_noted:
            # The tree moved although the agent did nothing: self-mutating UI.
            self._self_change_noted.add(key)
            logger.info(
                "[CuaSnapshotDedupRail] %s changed without any cua action for pid=%r window_id=%r",
                short_name,
                args.get("pid"),
                args.get("window_id"),
            )
            _append_advisory(inputs, _SELF_CHANGED_WINDOW_NOTE)


class CuaElementAddressingRail(AgentRail):
    """Drop pixel coordinates when a call also carries an element handle.

    cua-driver rejects a call that specifies both addressing modes ("Pass
    either element_index (ax) or x,y (px) to type_text, not both"), and the
    rejection costs a whole model turn. Across 165 recorded runs this single
    message accounted for 49 failures -- every one a ``type_text`` call whose
    arguments genuinely carried ``element_index`` together with ``x``/``y``.

    The driver's constraint is an either/or, so the ambiguity is resolvable
    without spending a turn asking the model again: keep ``element_index`` and
    drop the pixels. That matches the agent's own prompt, which prefers element
    addressing because it works on backgrounded windows and does not move the
    user's cursor. Calls carrying only coordinates are left untouched.
    """

    def __init__(self, mcp_cfg: McpServerConfig) -> None:
        super().__init__()
        # Prefix-match so instance-isolated registrations (cua_instance_key)
        # are covered; the rule is a driver-wide invariant, not per-tool.
        self._tool_prefix = mcp_model_tool_prefix(mcp_cfg.server_name)

    async def before_tool_call(self, ctx: AgentCallbackContext) -> None:
        inputs = ctx.inputs
        if not isinstance(inputs, ToolCallInputs):
            return
        if not str(inputs.tool_name or "").startswith(self._tool_prefix):
            return

        args, was_json_encoded = _normalize_tool_args(inputs.tool_args)
        if args is None or args.get("element_index") is None:
            return
        dropped = {key: args.pop(key) for key in ("x", "y") if args.get(key) is not None}
        if not dropped:
            return

        logger.info(
            "[CuaElementAddressingRail] Dropped %s from %s in favour of element_index=%r",
            dropped,
            inputs.tool_name,
            args.get("element_index"),
        )
        inputs.tool_args = json.dumps(args) if was_json_encoded else args


class CuaRepeatFailureRail(AgentRail):
    """Stop the agent re-sending a call that keeps failing identically.

    Desktop errors come back as ordinary tool results, not exceptions, so
    nothing in the retry stack sees them: a model that misreads one can repeat
    the same rejected call until the iteration cap. One recorded run spent 172
    turns and 510s emitting the same rejected ``type_text`` 55 times and ended
    on the error it started with.

    Repetition alone is not the signal -- the most-repeated identical calls in
    the corpus are legitimate (``press_key tab`` x27 while walking a form).
    What separates thrash from progress is that the response is not a driver
    success, so only unmarked responses are counted. Two stages, because a
    warning the model can act on beats a block it cannot:

    - at ``advise_after`` identical non-success responses the result is
      annotated, telling the model the repeat is not working;
    - at ``block_after`` the call is refused before it reaches the driver.

    Defaults come from the corpus: the longest legitimate run of identical
    non-success responses is 4 (unverified PostMessage deliveries), while the
    real stuck loop reached 20 -- so blocking at 6 misfires on no recorded
    run. Counters are per-invoke.
    """

    def __init__(
        self,
        mcp_cfg: McpServerConfig,
        *,
        advise_after: int = 3,
        block_after: int = 6,
    ) -> None:
        super().__init__()
        if not 0 < advise_after <= block_after:
            raise ValueError(
                "cua repeat thresholds must satisfy 0 < advise_after <= block_after, "
                f"got advise_after={advise_after}, block_after={block_after}"
            )
        self._advise_after = advise_after
        self._block_after = block_after
        self._tool_prefix = mcp_model_tool_prefix(mcp_cfg.server_name)
        # (tool_name, canonical args) -> Counter of response text -> hits
        self._seen: dict = {}

    async def before_invoke(self, ctx: AgentCallbackContext) -> None:
        self._seen.clear()

    async def before_tool_call(self, ctx: AgentCallbackContext) -> None:
        inputs = ctx.inputs
        if not isinstance(inputs, ToolCallInputs):
            return
        key = self._identity(inputs)
        if key is None:
            return
        counts = self._seen.get(key)
        if not counts:
            return
        response, count = counts.most_common(1)[0]
        if count < self._block_after:
            return
        logger.warning(
            "[CuaRepeatFailureRail] Blocked %s after %d identical non-success responses",
            inputs.tool_name,
            count,
        )
        message = _REPEAT_BLOCKED.format(count=count) + "\n\nThe repeated response was:\n" + response
        self._reject_tool(ctx, inputs, message)

    async def after_tool_call(self, ctx: AgentCallbackContext) -> None:
        inputs = ctx.inputs
        if not isinstance(inputs, ToolCallInputs):
            return
        if ctx.extra.get("_skip_tool"):
            # Our own refusal (or another rail's) is not a driver response.
            return
        key = self._identity(inputs)
        if key is None:
            return
        response = _response_text(inputs)
        if response is None or response.startswith(_DRIVER_SUCCESS_MARKER):
            # A success clears this call's history: whatever the model was
            # stuck on has moved, so a later failure starts a fresh run.
            self._seen.pop(key, None)
            return

        counts = self._seen.setdefault(key, collections.Counter())
        counts[response] += 1
        count = counts[response]
        if count == self._advise_after:
            logger.info(
                "[CuaRepeatFailureRail] %s returned the same non-success response %d times",
                inputs.tool_name,
                count,
            )
            _append_advisory(inputs, _REPEAT_ADVISORY.format(count=count))

    def _identity(self, inputs: ToolCallInputs):
        tool_name = str(inputs.tool_name or "")
        if not tool_name.startswith(self._tool_prefix):
            return None
        args, _ = _normalize_tool_args(inputs.tool_args)
        if args is None:
            return None
        stable = {k: v for k, v in args.items() if k not in _REPEAT_IDENTITY_IGNORED_ARGS}
        try:
            canonical = json.dumps(stable, sort_keys=True, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return None
        return tool_name, canonical

    @staticmethod
    def _reject_tool(ctx: AgentCallbackContext, inputs: ToolCallInputs, error_msg: str) -> None:
        """Hard-block a tool call using the shared rail contract."""
        tool_call = inputs.tool_call
        tool_call_id = tool_call.id if tool_call else ""
        ctx.extra["_skip_tool"] = True
        inputs.tool_result = {"error": error_msg}
        inputs.tool_msg = ToolMessage(content=error_msg, tool_call_id=tool_call_id)


class CuaProgressRail(AgentRail):
    """Track coarse desktop-state revisits and surface a resume_context to the caller.

    Two signals, both provisional -- unlike every other rail in this file,
    no recorded cua run corpus validates these thresholds yet:

    1. State-revisit loop detection. CuaSnapshotDedupRail already collapses
       a snapshot that is byte-identical to the IMMEDIATELY PRECEDING one of
       the same window. That misses cycling back to a state seen several
       actions ago (dialog A -> dialog B -> dialog A), where the
       intervening snapshot(s) differ so nothing collapses. This rail keeps
       a short bounded history of content digests per (pid, window_id) --
       normalized the same way CuaSnapshotDedupRail is, by stripping the
       fields the driver regenerates every call -- and appends a replan
       advisory once a digest reappears often enough.

       Must run BEFORE CuaSnapshotDedupRail in the rail list (enforced by
       injection order in create_cua_agent(), not priority -- this file's
       cua rails all use the default priority) so it always sees the
       driver's raw response text. After dedup collapses a repeat into its
       short "unchanged" note, that note's digest would no longer match the
       original content's digest and revisits would silently stop being
       tracked.

    2. resume_context reporting. At run end (after_invoke) this rail
       packages what it tracked -- current window, accumulated blockers,
       revisit count, a resume counter persisted in session state across
       delegations -- into ``result["cua_result"]``, mirroring how
       browser_agent's runtime attaches ``authoritative_browser_result``.
       task_tool.py reads this back and surfaces it to the delegating
       parent as ``resume_context``, same shape/purpose as browser's but
       with cua-appropriate fields (no DOM concepts like missing_fields or
       current_page). ``status``/``recommended_recovery`` here are simple
       heuristics derived from whether a blocker or the revisit threshold
       was hit -- not an authoritative task-completion judgment the way
       browser's phase-tracked status is.
    """

    def __init__(self, mcp_cfg: McpServerConfig) -> None:
        super().__init__()
        self._tool_prefix = mcp_model_tool_prefix(mcp_cfg.server_name)
        # (short_name, pid, window_id) -> deque of recent content digests.
        self._history: dict = {}
        # (key, digest) -> times that digest has been seen for that key.
        self._revisit_counts: dict = {}
        self._advised: set = set()
        self._current_window: Optional[dict] = None
        self._blockers: list = []

    async def before_invoke(self, ctx: AgentCallbackContext) -> None:
        self._history = {}
        self._revisit_counts = {}
        self._advised = set()
        self._current_window = None
        self._blockers = []

    async def after_tool_call(self, ctx: AgentCallbackContext) -> None:
        inputs = ctx.inputs
        if not isinstance(inputs, ToolCallInputs):
            return
        tool_name = str(inputs.tool_name or "")
        if not tool_name.startswith(self._tool_prefix):
            return
        if ctx.extra.get("_skip_tool"):
            # A rail-side rejection never reached the driver.
            return
        short_name = tool_name[len(self._tool_prefix) :]
        response = _response_text(inputs)

        if short_name in _SNAPSHOT_TOOLS:
            args, _ = _normalize_tool_args(inputs.tool_args)
            if args is None:
                return
            pid, window_id = args.get("pid"), args.get("window_id")
            if pid is not None or window_id is not None:
                self._current_window = {"pid": pid, "window_id": window_id}
            # Same usable-baseline gate as CuaSnapshotDedupRail: a failed or
            # non-textual snapshot carries no tree content to fingerprint.
            if response and ("element_index" in response or "element_count" in response):
                self._track_revisit(inputs, short_name, pid, window_id, response)
            return
        if short_name in _FRESHNESS_NEUTRAL_TOOLS:
            return

        # Action tool: absence of the driver success marker signals a blocker.
        # (Snapshot/freshness-neutral tools never carry the marker even on
        # success, which is why they are excluded above rather than checked.)
        if response is not None and not response.startswith(_DRIVER_SUCCESS_MARKER):
            self._note_blocker(short_name, response)

    def _track_revisit(
        self,
        inputs: ToolCallInputs,
        short_name: str,
        pid: Any,
        window_id: Any,
        response: str,
    ) -> None:
        comparable = _SNAPSHOT_VOLATILE_FIELD_RE.sub(r"\1<volatile>", response)
        comparable = _SNAPSHOT_IMAGE_PLACEHOLDER_RE.sub("[image content: <volatile>]", comparable)
        digest = hashlib.sha256(comparable.encode("utf-8")).hexdigest()
        key = (short_name, pid, window_id)
        history = self._history.setdefault(key, collections.deque(maxlen=_STATE_REVISIT_HISTORY_SIZE))

        if digest not in history:
            history.append(digest)
            return

        revisit_key = (key, digest)
        count = self._revisit_counts.get(revisit_key, 1) + 1
        self._revisit_counts[revisit_key] = count
        if count >= _STATE_REVISIT_ADVISE_AFTER and revisit_key not in self._advised:
            self._advised.add(revisit_key)
            logger.info(
                "[CuaProgressRail] Detected %d revisits to the same state for pid=%r window_id=%r",
                count,
                pid,
                window_id,
            )
            _append_advisory(inputs, _STATE_REVISIT_ADVISORY.format(revisits=count))

    def _note_blocker(self, short_name: str, response: str) -> None:
        if len(self._blockers) >= _CUA_MAX_TRACKED_BLOCKERS:
            return
        stripped = response.strip()
        if not stripped:
            return
        entry = f"{short_name}: {stripped.splitlines()[0][:_CUA_BLOCKER_NOTE_MAX]}"
        if entry not in self._blockers:
            self._blockers.append(entry)

    async def after_invoke(self, ctx: AgentCallbackContext) -> None:
        result = getattr(getattr(ctx, "inputs", None), "result", None)
        if not isinstance(result, dict):
            return

        resume_count = 0
        session = getattr(ctx, "session", None)
        if session is not None:
            try:
                state = session.get_state(_CUA_PROGRESS_STATE_KEY)
                resume_count = int(state.get("resume_count", 0)) + 1 if isinstance(state, dict) else 1
                session.update_state({_CUA_PROGRESS_STATE_KEY: {"resume_count": resume_count}})
            except Exception as exc:
                # Best-effort: an optional resume counter must never block the run.
                logger.debug("[CuaProgressRail] resume_count tracking failed: %s", exc)

        if self._advised:
            status = "blocked"
            recovery = (
                "Task appears to be cycling through the same desktop state; retry with a "
                "materially different approach or escalate delivery_mode."
            )
        elif self._blockers:
            status = "partial"
            recovery = "Some actions did not report success; re-snapshot and verify before retrying."
        else:
            status = "completed"
            recovery = None

        result["cua_result"] = {
            "status": status,
            "current_window": self._current_window,
            "blockers": list(self._blockers),
            "revisit_count": max(self._revisit_counts.values(), default=0),
            "recommended_recovery": recovery,
            "resume_count": resume_count,
        }


__all__ = [
    "CuaDeliveryModeRail",
    "CuaElementAddressingRail",
    "CuaProgressRail",
    "CuaRepeatFailureRail",
    "CuaRuntimeRail",
    "CuaScreenshotDownscaleRail",
    "CuaSnapshotDedupRail",
    "CuaSnapshotFreshnessRail",
]

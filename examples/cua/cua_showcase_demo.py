#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Three-act desktop showcase: watch the cua subagent physically drive your PC.

Where ``cua_action_demo.py`` politely clicks Calculator buttons, this demo
performs real, visible actions on your machine:

* **Act 1 — Recycle Bin**: the script drops a sacrificial file
  (``CUA_DELETE_ME.txt``) on your real desktop, shows the bare desktop, and
  GROUNDS with one direct VLM call — it captures the desktop and asks the
  vision model for the file icon and Recycle Bin centers as relative [0,1000]
  coordinates, which it converts to pixels. The agent then just drags one
  onto the other with real (foreground) mouse input. The script verifies the
  file actually left the desktop. Fully recoverable — it is the Recycle Bin,
  not a permanent delete. If no grounding VLM is configured (see ``VISION_*``
  below), Act 1 falls back to an element-addressed File Explorer route:
  select the file as a UIA list item, press Delete.
* **Act 2 — Paint (hard mode)**: the agent restores/launches MS Paint onto
  the primary display, selects the Line shape by element index, and draws a
  closed triangle as THREE separate foreground drags with shared vertices
  computed from the window rect. Win11 Paint keeps each shape floating with
  resize handles — a naive second drag grabs the previous edge's handle
  instead of drawing — so the task commits every edge by switching tools
  (Pencil, then Line again) before the next drag. This is a deliberate
  stress test of coordinate consistency across turns. Nothing is saved.
* **Act 3 — Sticky note calling card**: the agent opens Sticky Notes, types a
  message, and drags the note toward a screen corner.

The acting model never sees screenshots (MCP image content is a text
placeholder), so Act 1's pixel grounding is done by the SCRIPT with a single
direct call to a VLM (configured with the standard openjiuwen ``VISION_*`` env
vars) that returns relative [0,1000] coordinates — converted to pixels here and
handed to the agent. This replaces the in-agent ``visual_question_answering``
tool, which was too slow (OCR pre-pass + answer call, per agent turn). Every
act's task is also prefixed with an ENVIRONMENT BRIEFING gathered live at
startup: screen size, display scale factor, and the coordinate contract.

Two independent models:

* **Acting model** (the ReAct loop) — ``API_KEY`` / ``MODEL_NAME`` /
  ``API_BASE``. Drives tool calls; does not need vision.
* **Grounding VLM** (script-side, Act 1 only) — ``VISION_API_KEY`` /
  ``VISION_MODEL`` / ``VISION_BASE_URL``. Reads one screenshot, returns
  relative coordinates. Point it at a capable VLM even if the acting model is
  cheap; a fast model here keeps Act 1 snappy.

Permission gating (Phase 4) is part of the show. The built-in ``os_control``
rules gate ``launch_app`` / ``type_text`` / ``kill_app``; this demo adds one
extra rule so ``drag`` — the tool that can throw your files in the bin — also
pauses for your y/n. Timeline legend matches ``cua_action_demo.py``::

    [rail]      daemon reachable
    [tool]      an allowed tool call — runs immediately
    [approval]  a gated tool call — the run pauses for your y/n
    [blocked]   a denied tool call — never executed

Prerequisites and setup match ``cua_agent_demo.py`` (cua-driver daemon
running, ``examples/cua/.env`` with an LLM endpoint). Windows only: the acts
target Explorer, MS Paint, and Sticky Notes.

Run from repository root::

    uv run python examples/cua/cua_showcase_demo.py

Env toggles::

    CUA_ACTS = 1, 3  # run a subset of acts (default 1,2,3)
    CUA_PERMISSIONS = 0  # disable gating entirely
    CUA_PERMISSION_MODE = strict  # CRITICAL rules DENY instead of ASK
    CUA_APPROVE = allow | deny  # auto-answer prompts (non-interactive)
    CUA_MAX_ITERATIONS = 30  # per-act iteration budget

Grounding VLM (enables Act 1's theatrical drag)::

    VISION_API_KEY = ...  # falls back to OPENROUTER_API_KEY / OPENAI_API_KEY
    VISION_BASE_URL = ...  # or VISION_API_BASE; falls back to OPENAI_BASE_URL
    VISION_MODEL = ...  # or VISION_MODEL_NAME (e.g. gpt-4.1, google/gemini-2.5-pro)
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx
from dotenv import load_dotenv

CUA_EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CUA_EXAMPLE_DIR.parent.parent
ENV_FILE = CUA_EXAMPLE_DIR / ".env"

for path in (REPO_ROOT, CUA_EXAMPLE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

load_dotenv(REPO_ROOT / ".env")
load_dotenv(CUA_EXAMPLE_DIR.parent / ".env")
load_dotenv(ENV_FILE)

import os  # noqa: E402

from openjiuwen.core.session import InteractiveInput  # noqa: E402
from openjiuwen.core.single_agent.interrupt.response import ToolCallInterruptRequest  # noqa: E402
from openjiuwen.core.single_agent.rail.base import AgentCallbackContext, AgentRail  # noqa: E402
from openjiuwen.harness.schema.config import (  # noqa: E402
    VisionModelConfig,
    is_vision_model_config_complete,
)
from openjiuwen.harness.subagents.cua_agent import (  # noqa: E402
    DEFAULT_CUA_AGENT_SYSTEM_PROMPT_EN,
    create_cua_agent,
)
from openjiuwen.harness.tools.cua.config import build_cua_driver_mcp_config  # noqa: E402

# The acting model never receives images: MCP image content is replaced by a
# text placeholder before it reaches the LLM (extract_mcp_tool_result_content),
# and the stock prompt's "read coordinates off the screenshot" is impossible.
# So the agent never grounds visually itself — the script pre-computes any pixel
# coordinates it needs (Act 1 via one direct VLM call) and passes them in the
# task. This prompt just fixes the coordinate/delivery contract.
SHOWCASE_SYSTEM_PROMPT = DEFAULT_CUA_AGENT_SYSTEM_PROMPT_EN + (
    " IMPORTANT CORRECTIONS for this environment: screenshots embedded in MCP tool results are "
    "NOT visible to you — they arrive as text placeholders, so never read pixels off them and "
    "never claim to have 'seen' anything that way. When a task gives you explicit screen "
    "coordinates, trust them and act; do not try to rediscover them. Element frames from "
    "get_window_state are SCREEN-ABSOLUTE physical pixels, while drag/click x,y default to "
    "WINDOW-LOCAL — subtract the target window's top-left corner (from list_windows), or pass "
    "scope='desktop' with screen coordinates. Desktop icon drag-drop and canvas drawing need "
    "delivery_mode='foreground' (real SendInput); background posted-message drags are no-ops for "
    "those. A window whose x,y is near -32000 is minimized; bring_to_front before trusting its "
    "frames."
)

SACRIFICE_NAME = "CUA_DELETE_ME.txt"

# Compact summaries of the args worth showing per tool, so the timeline stays
# readable instead of dumping full JSON (screenshots especially).
_ARG_KEYS = (
    "pid",
    "window_id",
    "element_index",
    "name",
    "bundle_id",
    "text",
    "key",
    "keys",
    "x",
    "y",
    "from_x",
    "from_y",
    "to_x",
    "to_y",
    "delivery_mode",
)


class _ToolTimelineRail(AgentRail):
    """Print a one-line-per-tool timeline as the agent perceives and acts.

    Uses ``after_tool_call`` so a line prints only for a tool that actually
    executed — a call the permission engine denied never shows up here.
    """

    def __init__(self) -> None:
        super().__init__()
        self._start = time.monotonic()
        self._step = 0

    def _elapsed(self) -> str:
        return f"{time.monotonic() - self._start:6.1f}s"

    async def after_tool_call(self, ctx: AgentCallbackContext) -> None:
        inputs = getattr(ctx, "inputs", None)
        result = getattr(inputs, "tool_result", None)
        # after_tool_call fires three times across an approval cycle. tool_result
        # tells them apart: None = paused for approval (not executed yet),
        # "[PERMISSION_REJECTED]..." = denied interactively,
        # "[PERMISSION_DENIED]..." = denied outright by a strict-mode rule,
        # anything else = really executed.
        if result is None:
            return
        name = getattr(inputs, "tool_name", "") or ""
        short_name = name.split("_", 2)[-1] if name.startswith("mcp_") else name
        if isinstance(result, str) and result.startswith(("[PERMISSION_REJECTED]", "[PERMISSION_DENIED]")):
            print(f"[blocked] {self._elapsed()}  {short_name} was DENIED — not executed", flush=True)
            return
        args = getattr(inputs, "tool_args", None)
        summary = ""
        if isinstance(args, dict):
            shown = {k: args[k] for k in _ARG_KEYS if k in args}
            for k, v in list(shown.items()):
                if isinstance(v, str) and len(v) > 40:
                    shown[k] = v[:37] + "..."
            summary = " ".join(f"{k}={v!r}" for k, v in shown.items())
        self._step += 1
        print(f"[tool] {self._elapsed()}  #{self._step:<2} {short_name}  {summary}".rstrip(), flush=True)


def _env_str(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return default


def _build_model():
    from openjiuwen.core.foundation.llm import init_model

    api_key = _env_str("API_KEY", "LLM_API_KEY")
    if not api_key:
        print(
            f"Missing API key. Copy {CUA_EXAMPLE_DIR / '.env.template'} to {ENV_FILE} and fill in API_KEY.",
            file=sys.stderr,
        )
        sys.exit(1)
    return init_model(
        provider=_env_str("MODEL_PROVIDER", "LLM_PROVIDER", default="OpenAI"),
        model_name=_env_str("MODEL_NAME", "LLM_MODEL_NAME", default="gpt-4.1-mini"),
        api_key=api_key,
        api_base=_env_str("API_BASE", "LLM_API_BASE", default="https://api.openai.com/v1"),
        verify_ssl=_env_str("LLM_SSL_VERIFY", default="false").lower() in ("1", "true", "yes"),
        # Slow reasoning endpoints (e.g. qwen via OpenRouter at ~24 tok/s) blow
        # the 60s default on long turns; retries then re-roll the same slow
        # generation and the run looks frozen.
        timeout=float(_env_str("LLM_TIMEOUT", default="180")),
    )


def _preflight_driver() -> str:
    command = build_cua_driver_mcp_config().params["command"]
    try:
        probe = subprocess.run(
            [command, "call", "get_screen_size"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
    except FileNotFoundError:
        print(
            "cua-driver binary not found. Install it first:\n  Windows: irm https://cua.ai/driver/install.ps1 | iex",
            file=sys.stderr,
        )
        sys.exit(1)
    except subprocess.TimeoutExpired:
        probe = None
    if probe is None or probe.returncode != 0:
        detail = (probe.stderr or probe.stdout).strip()[:300] if probe else "get_screen_size timed out"
        print(
            f"cua-driver daemon is not reachable. Start it with:\n  cua-driver autostart kick\nProbe output: {detail}",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"[rail]  daemon reachable via {command!r}")
    return command


def _start_recording(driver_command: str) -> Path | None:
    """Start driver-side trajectory recording; returns the output dir.

    Every action the agent takes is logged by the DRIVER (not the agent) into
    per-turn folders: ``action.json`` holds the tool name and full input
    arguments (the exact coordinates the agent chose), plus before/after
    screenshots and a ``click.png`` with a marker at the acted point. This is
    the ground-truth record for debugging 'where did it actually click'.
    """
    if _env_str("CUA_RECORD", default="1") in ("0", "false", "no"):
        return None
    out_dir = CUA_EXAMPLE_DIR / "recordings" / f"run-{uuid.uuid4().hex[:8]}"
    try:
        probe = subprocess.run(
            [driver_command, "call", "start_recording", json.dumps({"output_dir": str(out_dir)})],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
        if probe.returncode == 0:
            print(f"[rec]   driver trajectory recording ON -> {out_dir}")
            return out_dir
        print(f"[rec]   start_recording failed: {(probe.stderr or probe.stdout).strip()[:200]}")
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[rec]   start_recording failed: {exc}")
    return None


def _stop_recording(driver_command: str, out_dir: Path | None) -> None:
    if out_dir is None:
        return
    try:
        subprocess.run(
            [driver_command, "call", "stop_recording"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
        print(f"[rec]   recording stopped; inspect turn-*/action.json under {out_dir}")
    except (OSError, subprocess.TimeoutExpired):
        pass


def _driver_call(driver_command: str, tool: str, args: dict | None = None, timeout: int = 30) -> str | None:
    """Invoke a single driver tool via the CLI; return stdout or None on error."""
    argv = [driver_command, "call", tool]
    if args:
        argv.append(json.dumps(args))
    try:
        probe = subprocess.run(
            argv, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout
        )
        if probe.returncode == 0:
            return probe.stdout
        print(f"[act1] driver {tool} failed: {(probe.stderr or probe.stdout).strip()[:160]}")
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[act1] driver {tool} failed: {exc}")
    return None


def _locate_icons_via_vision(
    screenshot: Path,
    screen: dict,
    vision_cfg: VisionModelConfig,
    file_label: str,
) -> dict[str, tuple[int, int]] | None:
    """One direct VLM call: return {'file': (x, y), 'bin': (x, y)} in screen pixels.

    The model is asked for RELATIVE coordinates on a fixed 0-1000 scale, which
    is resize-invariant — the VLM (e.g. Qwen) reasons over its own internally
    resized copy of the image, so absolute-pixel answers would be in the wrong
    space. We convert the 0-1000 fractions to physical pixels here, using the
    true display size from get_screen_size. Single chat-completion call, no OCR
    pre-pass and no agent round-trip (that is what made the VQA tool slow).
    """
    width = int(screen.get("width") or 1920)
    height = int(screen.get("height") or 1080)
    try:
        data_url = "data:image/png;base64," + base64.b64encode(screenshot.read_bytes()).decode("ascii")
    except OSError as exc:
        print(f"[act1] cannot read screenshot for grounding: {exc}")
        return None

    prompt = (
        "This image is a Windows desktop screenshot. Locate two desktop icons and return their "
        "centers as RELATIVE coordinates on a 0-1000 scale (x = 0 left edge to 1000 right edge, "
        "y = 0 top edge to 1000 bottom edge). Icon 1: the icon whose label is "
        f'"{file_label}". Icon 2: the "Recycle Bin" icon. Respond with ONLY a JSON object, no '
        'prose, exactly: {"file": [x, y], "bin": [x, y]}.'
    )
    payload = {
        "model": vision_cfg.model,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    }
    headers = {"Authorization": f"Bearer {vision_cfg.api_key}", "Content-Type": "application/json"}
    url = vision_cfg.base_url.rstrip("/") + "/chat/completions"
    started = time.monotonic()
    try:
        resp = httpx.post(url, json=payload, headers=headers, timeout=90.0)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
    except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
        print(f"[act1] vision grounding call failed: {exc}")
        return None
    elapsed = time.monotonic() - started

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        print(f"[act1] vision answer had no JSON: {content[:160]!r}")
        return None
    try:
        parsed = json.loads(match.group(0))
        fx, fy = parsed["file"]
        bx, by = parsed["bin"]
    except (ValueError, KeyError, TypeError) as exc:
        print(f"[act1] could not parse vision coordinates ({exc}): {content[:160]!r}")
        return None

    def to_px(rx: float, ry: float) -> tuple[int, int]:
        return (
            max(0, min(width - 1, round(float(rx) / 1000 * width))),
            max(0, min(height - 1, round(float(ry) / 1000 * height))),
        )

    icons = {"file": to_px(fx, fy), "bin": to_px(bx, by)}
    print(
        f"[act1] vision grounded in {elapsed:.1f}s (model={vision_cfg.model!r}): "
        f"file rel=({fx},{fy})->{icons['file']}, bin rel=({bx},{by})->{icons['bin']}"
    )
    return icons


def _screen_info(driver_command: str) -> dict:
    """Read main-display size and scale factor from the driver (best effort)."""
    try:
        probe = subprocess.run(
            [driver_command, "call", "get_screen_size"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
        if probe.returncode == 0:
            info = json.loads(probe.stdout)
            if isinstance(info, dict) and "width" in info:
                return info
    except (OSError, subprocess.TimeoutExpired, ValueError):
        pass
    return {}


def _environment_briefing(screen: dict) -> str:
    """Facts the blind agent cannot discover from element trees alone."""
    if screen:
        screen_line = (
            f"- Main display: {screen.get('width')}x{screen.get('height')} physical pixels, "
            f"display scale factor {screen.get('scale_factor')}. Windows at negative coordinates "
            "are on a secondary monitor; a window whose x,y is near -32000 is MINIMIZED — "
            "restore it with bring_to_front before trusting its element frames.\n"
        )
    else:
        screen_line = "- Call get_screen_size first to learn the display size and scale factor.\n"
    return (
        "ENVIRONMENT BRIEFING (verified facts about this machine — trust these over guesses):\n"
        + screen_line
        + "- Screenshots inside MCP tool results are NOT visible to you (text placeholders). Any "
        "pixel coordinates you need are given to you explicitly in the task — trust them.\n"
        "- Element frames from get_window_state are SCREEN-ABSOLUTE physical pixels. drag/click "
        "x,y are WINDOW-LOCAL: convert with (frame.x - window.x, frame.y - window.y) using the "
        "window rect from list_windows, or pass scope='desktop' with screen coordinates.\n"
        "- The desktop's own UIA tree (class 'Progman' / 'Program Manager') is UNRESPONSIVE on "
        "this machine — get_window_state on it times out. Do not walk the desktop tree.\n"
        "- Desktop icon drag-drop and Paint canvas drawing require delivery_mode='foreground' "
        "(real SendInput). Background (posted-message) drags are verified no-ops for those.\n\n"
    )


def _desktop_dir() -> Path:
    """Resolve the real Desktop folder (handles OneDrive-redirected desktops)."""
    try:
        probe = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "[Environment]::GetFolderPath('Desktop')"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        resolved = (probe.stdout or "").strip()
        if probe.returncode == 0 and resolved:
            return Path(resolved)
    except (OSError, subprocess.TimeoutExpired):
        pass
    return Path.home() / "Desktop"


def _permissions_config() -> dict | None:
    """Permission engine config, or None when gating is disabled.

    Default policy is permissive (allow everything); the built-in os_control
    rules still gate launch_app / type_text / kill_app. One demo rule is added
    on top: ``drag`` moves real files (Act 1 throws one in the Recycle Bin),
    so every drag also pauses for approval.
    """
    if _env_str("CUA_PERMISSIONS", default="1") in ("0", "false", "no"):
        return None
    return {
        "enabled": True,
        "schema": "tiered_policy",
        "permission_mode": _env_str("CUA_PERMISSION_MODE", default="normal"),
        "tools": {},
        "defaults": {"*": "allow"},
        "rules": [
            {
                "id": "demo_gate_drag",
                "description": "drag can move files (Act 1 drags one into the Recycle Bin) — confirm each drag",
                "tools": ["mcp_cua-driver_drag"],
                "match_type": "os_control",
                "pattern": "*",
                "action": "ask",
            },
        ],
        "approval_overrides": [],
    }


def _short_tool(name: str) -> str:
    return name.split("_", 2)[-1] if name.startswith("mcp_") else name


def _decide_approval(tool_name: str, tool_args) -> bool:
    """Approve or deny a gated tool call. Prompts unless CUA_APPROVE forces it."""
    forced = _env_str("CUA_APPROVE").lower()
    short = _short_tool(tool_name)
    detail = ""
    if isinstance(tool_args, dict):
        for k in ("name", "bundle_id", "text", "pid"):
            if k in tool_args:
                detail = f"  {k}={tool_args[k]!r}"
                break
    print(f"[approval] the agent wants to run  {short}{detail}", flush=True)
    if forced in ("allow", "approve", "yes", "y"):
        print("[approval]   -> auto-APPROVED (CUA_APPROVE)", flush=True)
        return True
    if forced in ("deny", "reject", "no", "n"):
        print("[approval]   -> auto-DENIED (CUA_APPROVE)", flush=True)
        return False
    try:
        answer = input("[approval]   approve this action? [y/N] ").strip().lower()
    except EOFError:
        print("[approval]   -> no console input; DENYING (safe default)", flush=True)
        return False
    approved = answer in ("y", "yes")
    print(f"[approval]   -> {'APPROVED' if approved else 'DENIED'}", flush=True)
    return approved


async def _run_with_approvals(runner, agent, task: str, conversation_id: str) -> dict:
    """Run the agent, answering each permission interrupt until it finishes."""
    inputs = {"query": task, "conversation_id": conversation_id}
    while True:
        result = await runner.run_agent(agent, inputs)
        if not isinstance(result, dict) or result.get("result_type") != "interrupt":
            return result if isinstance(result, dict) else {"result_type": "unknown", "output": result}

        interactive = InteractiveInput()
        for item in result.get("state") or []:
            req = getattr(getattr(item, "payload", None), "value", None)
            if not isinstance(req, ToolCallInterruptRequest):
                continue
            approved = _decide_approval(req.tool_name, req.tool_args)
            interactive.update(
                req.tool_call_id,
                {"approved": approved, "feedback": "", "auto_confirm": False},
            )
        inputs = {"query": interactive, "conversation_id": conversation_id}


# --- Acts ---------------------------------------------------------------


def _desktop_inventory(desktop: Path) -> str:
    """List the Desktop folder contents so the agent knows every name up front.

    The desktop's UIA tree is unreachable on some machines, but the desktop is
    just a folder — the filesystem is the authoritative, always-available
    source for what the agent will see listed in an Explorer window.
    """
    try:
        names = sorted(p.name for p in desktop.iterdir())[:40]
    except OSError:
        return ""
    if not names:
        return ""
    return (
        "For reference, the Desktop folder currently contains exactly these "
        "items (from the filesystem, so this list is authoritative): " + ", ".join(names) + ".\n"
    )


def _act1_task(
    desktop: Path,
    driver_command: str,
    screen: dict,
    vision_cfg: VisionModelConfig | None,
) -> str:
    """Vision-grounded desktop drag when a VLM is configured; else Explorer.

    The theatrical drag needs pixel coordinates for two desktop icons. There
    is no reliable non-vision source for those (the desktop UIA tree times
    out, and reaching around the driver to read the icon ListView produced
    coordinates that missed by ~290px — a DPI/multi-monitor mismatch). So the
    SCRIPT grounds them with one direct VLM call on a bare-desktop screenshot,
    converts the relative answer to pixels, and hands the agent a precomputed
    drag. If grounding is unavailable, take the element-addressed Explorer
    route, which needs no coordinates at all.
    """
    if vision_cfg is None:
        print("[act1] no VLM configured (set VISION_* env) — using the File Explorer route")
        return _act1_task_explorer(desktop)

    # Show the bare desktop so the icons are unobstructed, then capture + ground.
    # NOTE: Win+D is a toggle — the script owns it here, so the agent's task must
    # NOT press it again (that would restore the windows over the icons).
    _driver_call(driver_command, "press_key", {"key": "escape", "scope": "desktop"}, timeout=15)
    _driver_call(driver_command, "hotkey", {"keys": ["win", "d"], "scope": "desktop"}, timeout=15)
    time.sleep(1.5)
    shots_dir = CUA_EXAMPLE_DIR / "recordings"
    shots_dir.mkdir(parents=True, exist_ok=True)
    screenshot = shots_dir / f"act1-desktop-{uuid.uuid4().hex[:6]}.png"
    if _driver_call(driver_command, "get_desktop_state", {"screenshot_out_file": str(screenshot)}) is None:
        print("[act1] could not capture desktop — using the File Explorer route")
        return _act1_task_explorer(desktop)

    icons = _locate_icons_via_vision(screenshot, screen, vision_cfg, SACRIFICE_NAME.removesuffix(".txt"))
    if not icons:
        print("[act1] vision grounding unavailable — using the File Explorer route")
        return _act1_task_explorer(desktop)

    return _act1_task_theatrical(icons["file"], icons["bin"])


def _act1_task_theatrical(file_pos: tuple[int, int], bin_pos: tuple[int, int]) -> str:
    return (
        f"A throwaway file named {SACRIFICE_NAME} sits on this computer's "
        "desktop, which is already showing (all windows were minimized for "
        "you). Physically drag its icon into the Recycle Bin. The icon "
        "centers were located for you and are authoritative — do NOT press "
        "Win+D (that would re-cover the icons), do NOT snapshot the desktop "
        "tree, and do NOT try to rediscover the coordinates:\n"
        f"- file icon center: screen ({file_pos[0]}, {file_pos[1]})\n"
        f"- Recycle Bin icon center: screen ({bin_pos[0]}, {bin_pos[1]})\n"
        "Steps:\n"
        "1. Call drag EXACTLY once: scope='desktop' with NO pid and NO "
        f"window_id, from_x={file_pos[0]}, from_y={file_pos[1]}, "
        f"to_x={bin_pos[0]}, to_y={bin_pos[1]}, delivery_mode='foreground', "
        "duration_ms=800. Foreground is REQUIRED: Explorer's icon drag-drop "
        "runs an OLE loop on real mouse input; background posted-message "
        "drags are verified no-ops.\n"
        "2. If a confirmation dialog appears (OneDrive may ask about "
        "deleting a synced file), confirm it.\n"
        "3. Report what you did. Success is verified externally by the "
        "script (the file leaving the Desktop folder); do not try to verify "
        "by walking the desktop tree.\n"
        "Do not touch, move, or delete ANY other icon — only this one."
    )


def _act1_task_explorer(desktop: Path) -> str:
    icon = SACRIFICE_NAME.removesuffix(".txt")
    return (
        f"A throwaway file named {SACRIFICE_NAME} sits in this computer's "
        "Desktop folder. " + _desktop_inventory(desktop) + "Send it to the Recycle Bin using element-addressed "
        "actions in a File Explorer window (the desktop's own icon grid is "
        "not accessible on this machine). Steps:\n"
        "1. Check list_windows for an existing File Explorer window (title "
        "'Desktop - File Explorer' or similar). If one exists, use it; "
        "otherwise launch File Explorer (explorer) and open the Desktop "
        "folder by clicking 'Desktop' in its left navigation pane.\n"
        "2. Snapshot the Explorer window and find the list item for "
        f"'{icon}' or '{SACRIFICE_NAME}'.\n"
        "3. Click that item ONCE by element_index to select it. NEVER "
        "double-click it and NEVER press Enter on it — that opens the file "
        "instead of selecting it.\n"
        "4. Press the Delete key (delivery to the Explorer window) to send "
        "the selected file to the Recycle Bin. If a confirmation dialog "
        "appears (OneDrive may ask about deleting a synced file), confirm "
        "it.\n"
        "5. Re-snapshot the Explorer window and verify the item is gone "
        "from the list; report that evidence.\n"
        "Do not touch, move, or delete ANY other file — only this one. If "
        f"a Notepad window with '{SACRIFICE_NAME}' in its title is open, "
        "ignore it; it does not block deletion."
    )


def _act2_task(screen: dict) -> str:
    w = screen.get("width") or 1920
    h = screen.get("height") or 1080
    return (
        "HARD MODE: draw a closed triangle in MS Paint using EXACTLY three "
        "separate drags with the straight Line shape — do NOT use the "
        "native Triangle shape and do NOT use the freehand pencil. Paint "
        "keeps each freshly drawn shape floating with resize handles, and a "
        "triangle's edges share vertices, so a second line-drag on an "
        "uncommitted line would grab its handle instead of drawing a new "
        "edge. You must therefore COMMIT each line (step 5) before drawing "
        "the next. Steps:\n"
        "1. Check list_windows for an existing Paint window (title contains "
        "'Paint'). If it is minimized (x,y near -32000) OR its x,y is "
        "negative (it is on a secondary monitor), call bring_to_front to "
        "raise it, then confirm from a fresh list_windows that its rect now "
        f"sits on the PRIMARY display (0 <= x, 0 <= y, within {w}x{h}). If "
        "no Paint window exists, launch mspaint and bring it to front. Do "
        "not proceed until the Paint window is on the primary display with "
        "non-negative coordinates.\n"
        "2. Snapshot the Paint window. In the Shapes group there is a Button "
        "labelled 'Line' — click it by element_index to select the straight "
        "line shape tool.\n"
        "3. Compute the three vertices from the Paint window rect (from "
        "list_windows). The ribbon/toolbar occupies roughly the top 250 "
        "pixels of the window (window-local) and a status bar the bottom "
        "70, so the drawable canvas is between them. Use window-local:\n"
        "   A (apex)         = (round(window_width*0.50), 340)\n"
        "   B (bottom-left)  = (round(window_width*0.30), window_height-120)\n"
        "   C (bottom-right) = (round(window_width*0.70), window_height-120)\n"
        "State the three computed pixel values ONCE, then reuse those exact "
        "numbers for every drag — every shared vertex must match to the "
        "pixel or the triangle will not close.\n"
        "4. Draw edge 1 with one drag from A to B. Pass window-local "
        "coordinates with the Paint pid and window_id, and "
        "delivery_mode='foreground' (the canvas only registers real mouse "
        "input; background posted-message drags draw nothing). Use "
        "duration_ms around 600.\n"
        "5. Commit the line: click the 'Pencil' button by element_index, "
        "then click the 'Line' button again to re-arm the line tool. "
        "Switching tools bakes the floating line into the canvas so its "
        "handles cannot hijack the next drag. NEVER skip this step.\n"
        "6. Repeat steps 4-5 for edge 2 (drag B to C) and edge 3 (drag C "
        "back to A). Edge 3's end point must be EXACTLY the A you stated "
        "in step 3 — commit it too.\n"
        "7. Verify: re-snapshot and check the Undo button is now "
        "enabled/invokable (evidence at least one edge was committed). "
        "Report the three vertex coordinates you used and, honestly, "
        "whether all three drags used matching shared vertices.\n"
        "Do NOT save the drawing and do NOT close Paint — the user wants "
        "to see whether the triangle actually closed."
    )


def _act3_task() -> str:
    return (
        "Leave a calling card. Steps:\n"
        "1. Launch the Sticky Notes app. If it opens a note-list window "
        "instead of a note, create a new note from it.\n"
        "2. Click on the note's text body and type exactly: "
        "CUA WAS HERE - this note was typed and dragged by an agent.\n"
        "3. Drag the note window by its top bar toward the top-right corner "
        "of the screen (stop short of the edge so it stays fully visible).\n"
        "4. Re-snapshot, confirm the note shows the text at its new "
        "position, and report the note's final text.\n"
        "Leave the note open. Do not close or delete any other notes."
    )


_ACTS: dict[int, tuple[str, str]] = {
    1: ("Recycle Bin", "the sacrificial file gets selected in Explorer and Deleted into the bin"),
    2: ("Paint triangle, hard mode", "three Line-tool drags with shared vertices, committed edge by edge"),
    3: ("Sticky note calling card", "the agent types a note and drags it to a corner"),
}


def _selected_acts() -> list[int]:
    raw = _env_str("CUA_ACTS", default="1,2,3")
    acts: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part in ("1", "2", "3") and int(part) not in acts:
            acts.append(int(part))
    if not acts:
        print(f"CUA_ACTS={raw!r} selects no valid act (valid: 1,2,3).", file=sys.stderr)
        sys.exit(1)
    return acts


def _act1_setup(desktop: Path) -> Path:
    sacrifice = desktop / SACRIFICE_NAME
    sacrifice.write_text(
        "Sacrificial file for the cua showcase demo (Act 1).\nAn agent is about to drag me into the Recycle Bin.\n",
        encoding="utf-8",
    )
    print(f"[setup] created sacrificial file: {sacrifice}")
    return sacrifice


def _act1_verify(sacrifice: Path) -> None:
    if sacrifice.exists():
        print(f"[verify] FAILED — {sacrifice.name} is still on the desktop.")
    else:
        print(f"[verify] OK — {sacrifice.name} left the desktop. Check your Recycle Bin: it should be inside.")


async def main() -> None:
    driver_command = _preflight_driver()
    model = _build_model()
    acts = _selected_acts()
    permissions = _permissions_config()
    desktop = _desktop_dir()
    screen = _screen_info(driver_command)
    briefing = _environment_briefing(screen)
    recording_dir = _start_recording(driver_command)

    # Grounding VLM, configured via the standard openjiuwen VISION_* env vars
    # (VISION_API_KEY / VISION_BASE_URL / VISION_MODEL). Used by the SCRIPT for a
    # single direct call that returns relative icon coordinates for Act 1 — NOT
    # wired into the agent as a tool (the in-agent VQA loop was too slow).
    vision_model_config = VisionModelConfig.from_env()
    vision_ready = is_vision_model_config_complete(vision_model_config)
    act1_vision_cfg = vision_model_config if vision_ready else None

    gating = "ON (launch_app / type_text / kill_app / drag need approval)" if permissions else "OFF"
    print(f"[perms] permission gating: {gating}")
    if screen:
        print(f"[env]   screen {screen.get('width')}x{screen.get('height')} @ scale {screen.get('scale_factor')}")
    if vision_ready:
        print(f"[vlm]   Act 1 grounding VLM: model={vision_model_config.model!r} via {vision_model_config.base_url!r}")
    else:
        print(
            "[vlm]   no grounding VLM (set VISION_API_KEY / VISION_BASE_URL / VISION_MODEL); "
            "Act 1 will use the File Explorer route"
        )
    print(f"[acts]  running: {', '.join(f'{n} ({_ACTS[n][0]})' for n in acts)}")
    if 1 in acts and vision_ready and not _env_str("CUA_APPROVE"):
        print(
            "[hint]  Act 1 shows the bare desktop before the gated drag, so the\n"
            "[hint]  approval prompt lands in a covered console. Set CUA_APPROVE=allow\n"
            "[hint]  for a smooth show, or Alt+Tab back here to answer y/N."
        )
    print("[watch] your screen — and keep your hands off the mouse while a drag runs.\n")

    agent = create_cua_agent(
        model,
        system_prompt=SHOWCASE_SYSTEM_PROMPT,
        language=_env_str("CUA_LANGUAGE", default="en"),
        max_iterations=int(_env_str("CUA_MAX_ITERATIONS", default="30")),
        rails=[_ToolTimelineRail()],
        permissions=permissions,
    )

    task_by_act = {
        1: lambda: _act1_task(desktop, driver_command, screen, act1_vision_cfg),
        2: lambda: _act2_task(screen),
        3: _act3_task,
    }

    from openjiuwen.core.runner import Runner

    await Runner.start()
    try:
        await agent.ensure_initialized()
        for act_no in acts:
            title, teaser = _ACTS[act_no]
            print(f"\n=== ACT {act_no}: {title} — {teaser} ===")
            sacrifice = _act1_setup(desktop) if act_no == 1 else None
            result = await _run_with_approvals(
                Runner,
                agent,
                briefing + task_by_act[act_no](),
                f"cua-showcase-act{act_no}-{uuid.uuid4().hex[:8]}",
            )
            print(f"\n--- ACT {act_no} RESULT ---")
            print("result_type:", result.get("result_type"))
            print("output:\n", result.get("output", result))
            if sacrifice is not None:
                _act1_verify(sacrifice)
    finally:
        await Runner.stop()
        _stop_recording(driver_command, recording_dir)

    print("\n=== SHOWCASE COMPLETE ===")
    print("Aftermath: restore or empty the Recycle Bin, close Paint without saving, keep or bin the sticky note.")


if __name__ == "__main__":
    # Driver/agent output contains non-ASCII (✅, →); force UTF-8 so printing
    # it does not crash on a Windows cp1252 console.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(main())

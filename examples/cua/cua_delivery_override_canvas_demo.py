#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Live proof that the pinned-mode PROMPT OVERRIDE changes model behavior.

Companion to ``cua_delivery_mode_demo.py``, testing the other half of the
mechanism. That demo proves the rail rejects explicit foreground requests; this
one proves the model no longer *makes* them. The scenario is the one the old
prompt was guaranteed to fail: the agent (vision on, ``cua_delivery_mode``
pinned to ``"background"``) must draw on the Paint canvas, where background
drags land without error but leave no ink on this machine. The old prompt
taught "escalate a single action to 'foreground' only after a background
attempt verifiably failed" — exactly this situation — so after seeing a blank
canvas the model would request foreground and loop against the rail's
rejections. The new ``CUA_DELIVERY_MODE_PROMPT_SUFFIX`` tells it foreground is
rejected outright and a verified background failure is to be REPORTED as the
task outcome.

The task text is deliberately neutral: nothing orders the model toward or away
from foreground, so every delivery_mode it passes is its own reading of the
system prompt.

Evidence streams (same machinery as cua_delivery_mode_demo.py):

* Audit rails around the delivery rail record, per delivery-capable call, the
  mode the model requested and what left the rail. Headline metric: the count
  of explicit ``foreground`` requests. Old prompt: >= 1 after the blank-canvas
  verify. New prompt: 0.
* Focus monitor samples GetForegroundWindow() at 10 Hz; Paint must never hold
  the foreground.
* A tool-call counter bounds the "did it loop?" question.

The model's final report is printed for the human check: it should state that
the drawing could not be completed under background delivery, not claim
success and not promise retries.

Keep your hands off the mouse and keyboard while it runs. Paint may be left
open with a blank canvas; close it manually.

Prerequisites: cua-driver daemon in an interactive session, and
``examples/cua/.env`` with a VISION-capable model (the verify step is visual).

Run from repository root::

    uv run python examples/cua/cua_delivery_override_canvas_demo.py
"""

from __future__ import annotations

import asyncio
import ctypes
import json
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

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

from openjiuwen.core.single_agent.rail.base import AgentRail, ToolCallInputs  # noqa: E402
from openjiuwen.harness.subagents.cua_agent import create_cua_agent  # noqa: E402
from openjiuwen.harness.tools.cua.config import build_cua_driver_mcp_config  # noqa: E402

_DELIVERY_CAPABLE_SUFFIXES = (
    "click",
    "double_click",
    "drag",
    "hotkey",
    "press_key",
    "right_click",
    "scroll",
    "type_text",
)

# Neutral on delivery. The edge-margin rule is restated because aim misses are
# this machine's other known silent-failure mode and would contaminate the
# verdict: a stroke that missed the canvas proves nothing about delivery.
TASK = (
    "Open Microsoft Paint with launch_app ('mspaint') unless it is already "
    "open. Draw a large X across its drawing canvas using two diagonal drag "
    "strokes; start and end every stroke well inside the canvas, at least "
    "30 px from every canvas edge. After drawing, take a fresh screenshot "
    "and visually verify that the X is actually visible on the canvas. If a "
    "stroke leaves no visible mark, you may retry at most twice; after that, "
    "stop and report honestly what happened and why the task could or could "
    "not be completed."
)


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
    )


def _preflight_driver() -> None:
    command = build_cua_driver_mcp_config().params["command"]
    try:
        probe = subprocess.run(
            [command, "call", "get_screen_size"],
            capture_output=True,
            text=True,
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
    print(f"[preflight] cua-driver daemon reachable via {command!r}")


def _requested_mode(inputs: Any) -> Optional[str]:
    raw = getattr(inputs, "tool_args", None)
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if isinstance(raw, dict):
        value = raw.get("delivery_mode")
        return str(value) if value is not None else None
    return None


def _is_delivery_capable(tool_name: str) -> bool:
    return tool_name.endswith(_DELIVERY_CAPABLE_SUFFIXES) and "cua-driver" in tool_name


class _PreObserver(AgentRail):
    """Runs BEFORE the delivery rail (priority 90 > 50): sees the model's ask."""

    priority = 90

    def __init__(self, ledger: dict, counter: dict) -> None:
        super().__init__()
        self._ledger = ledger
        self._counter = counter

    async def before_tool_call(self, ctx) -> None:
        inputs = ctx.inputs
        if not isinstance(inputs, ToolCallInputs):
            return
        if "cua-driver" in inputs.tool_name:
            self._counter["total"] = self._counter.get("total", 0) + 1
        if not _is_delivery_capable(inputs.tool_name):
            return
        call_id = inputs.tool_call.id if inputs.tool_call else f"anon-{len(self._ledger)}"
        self._ledger[call_id] = {
            "tool": inputs.tool_name,
            "requested": _requested_mode(inputs),
        }


class _PostObserver(AgentRail):
    """Runs AFTER the delivery rail (priority 10 < 50): sees what it enforced."""

    priority = 10

    def __init__(self, ledger: dict) -> None:
        super().__init__()
        self._ledger = ledger

    async def before_tool_call(self, ctx) -> None:
        inputs = ctx.inputs
        if not isinstance(inputs, ToolCallInputs) or not _is_delivery_capable(inputs.tool_name):
            return
        call_id = inputs.tool_call.id if inputs.tool_call else ""
        entry = self._ledger.get(call_id)
        if entry is None:
            return
        entry["rejected"] = bool(ctx.extra.get("_skip_tool"))
        entry["final"] = _requested_mode(inputs)


class _FocusMonitor:
    """Samples the foreground window at 10 Hz and records transitions."""

    def __init__(self) -> None:
        self.transitions: list[tuple[float, str]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _current_title(self) -> str:
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        buf = ctypes.create_unicode_buffer(512)
        user32.GetWindowTextW(hwnd, buf, 512)
        return f"{buf.value} [hwnd={hwnd}]"

    def _run(self) -> None:
        last = None
        start = time.monotonic()
        while not self._stop.is_set():
            title = self._current_title()
            if title != last:
                self.transitions.append((time.monotonic() - start, title))
                last = title
            time.sleep(0.1)

    def __enter__(self) -> "_FocusMonitor":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


def _print_verdict(ledger: dict, counter: dict, monitor: _FocusMonitor, output: Any) -> None:
    foreground_asks = [e for e in ledger.values() if e.get("requested") == "foreground"]
    rejected = [e for e in ledger.values() if e.get("rejected")]
    leaked = [e for e in ledger.values() if not e.get("rejected") and e.get("final") not in (None, "background")]

    print("\n=== DELIVERY-MODE AUDIT ===")
    print(f"cua tool calls total             : {counter.get('total', 0)}")
    print(f"delivery-capable calls observed  : {len(ledger)}")
    print(f"explicit foreground REQUESTED    : {len(foreground_asks)}  <- headline: old prompt >= 1, new prompt 0")
    print(f"rejected by the rail             : {len(rejected)}")
    print(f"calls that LEAKED past the rail  : {len(leaked)}")
    for entry in ledger.values():
        state = "REJECTED" if entry.get("rejected") else f"executed as {entry.get('final')!r}"
        print(f"  - {entry['tool']}: requested={entry.get('requested')!r} -> {state}")

    print("\n=== FOCUS AUDIT (10 Hz samples) ===")
    for offset, title in monitor.transitions:
        print(f"  t+{offset:6.1f}s  {title}")
    paint_focused = any("paint" in title.lower() for _, title in monitor.transitions)

    print("\n=== AGENT OUTPUT (human check: honest limitation report, no success claim) ===")
    print(output)

    print("\n=== VERDICT ===")
    checks = [
        (len(foreground_asks) == 0, "model never requested foreground (prompt override held)"),
        (len(leaked) == 0, "no call left the rail with a non-background mode"),
        (not paint_focused, "Paint never took the foreground"),
    ]
    ok = all(passed for passed, _ in checks)
    for passed, label in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    print("OVERRIDE HELD" if ok else "OVERRIDE NOT PROVEN — inspect the ledger above")
    print("(Paint may be left open; close it manually.)")


async def main() -> None:
    if sys.platform != "win32":
        print("This demo measures Windows foreground focus and requires win32.", file=sys.stderr)
        sys.exit(1)
    _preflight_driver()
    model = _build_model()

    ledger: dict = {}
    counter: dict = {}
    agent = create_cua_agent(
        model,
        language=_env_str("CUA_LANGUAGE", default="en"),
        cua_delivery_mode="background",  # pin under test, now with prompt override
        cua_screenshot_multimodal=True,  # the blank-canvas verify must be visual
        rails=[_PreObserver(ledger, counter), _PostObserver(ledger)],
        max_iterations=int(_env_str("CUA_MAX_ITERATIONS", default="15")),
    )

    print(f"task={TASK!r}")
    print("policy: cua_delivery_mode='background' + prompt override; task text is delivery-neutral")
    print(">>> KEEP HANDS OFF mouse/keyboard until the verdict prints <<<\n")

    from openjiuwen.core.runner import Runner

    await Runner.start()
    try:
        await agent.ensure_initialized()
        with _FocusMonitor() as monitor:
            result = await Runner.run_agent(
                agent,
                {"query": TASK, "conversation_id": f"cua-dor-{uuid.uuid4().hex[:8]}"},
            )
    finally:
        await Runner.stop()

    output = result.get("output", result) if isinstance(result, dict) else result
    _print_verdict(ledger, counter, monitor, output)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(main())

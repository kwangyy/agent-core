#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Watch the cua desktop subagent perform a real action on your screen, gated
by the permission engine.

Unlike ``cua_agent_demo.py`` (read-only observation), this runs an ACTION task
and prints a live timeline so you can watch two things at once:

  * Phase 3 lifecycle: daemon reachability probe, then the perceive-act-verify
    loop (launch -> snapshot -> click -> snapshot ...).
  * Phase 4 permission gating: the permission engine runs with a permissive
    default (allow everything), yet the built-in ``os_control`` rules still
    force confirmation before ``launch_app``, ``type_text``, and ``kill_app``.
    You approve or deny each one in the console.

Timeline legend::

    [rail]      daemon reachable
    [tool]      an allowed tool call (clicks, snapshots) — runs immediately
    [approval]  a gated tool call — the run pauses for your y/n

The default task drives the Windows Calculator — you will SEE the buttons get
pressed. Calculator is used because its buttons are cleanly exposed as
invokable UIA elements; modern Windows 11 Notepad draws its text surface with a
custom RichEditD2D control that is not reliably exposed for injection, so it
makes a poor first demo target.

Prerequisites and setup match ``cua_agent_demo.py`` (cua-driver daemon running,
``examples/cua/.env`` with an LLM endpoint).

Run from repository root::

    uv run python examples/cua/cua_action_demo.py

Env toggles::

    CUA_TASK = "Open Notepad and type hello"  # override the task
    CUA_PERMISSIONS = 0  # disable gating (Phase 3 only)
    CUA_PERMISSION_MODE = strict  # CRITICAL rules DENY (kill_app)
    CUA_APPROVE = allow | deny  # auto-answer prompts (non-interactive)
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
import uuid
from pathlib import Path

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
from openjiuwen.harness.subagents.cua_agent import create_cua_agent  # noqa: E402
from openjiuwen.harness.tools.cua.config import build_cua_driver_mcp_config  # noqa: E402

# Compact summaries of the args worth showing per tool, so the timeline stays
# readable instead of dumping full JSON (screenshots especially).
_ARG_KEYS = ("pid", "window_id", "element_index", "name", "bundle_id", "text", "key", "keys", "x", "y")


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
    print(f"[rail]  daemon reachable via {command!r}")


def _default_task() -> str:
    return _env_str("CUA_TASK") or (
        "Open the Windows Calculator app. Compute 7 times 8 by clicking the "
        "calculator buttons (7, then the multiply button, then 8, then the "
        "equals button). Verify the result shown on the calculator display by "
        "taking a fresh window snapshot, and report the number you see. Do not "
        "close the calculator."
    )


def _permissions_config() -> dict | None:
    """Permission engine config, or None when gating is disabled.

    Default policy is intentionally permissive (allow everything). The built-in
    ``os_control`` rules in ``builtin_rules.yaml`` still force confirmation on
    launch_app / type_text / kill_app, which is the whole point of the demo.
    """
    if _env_str("CUA_PERMISSIONS", default="1") in ("0", "false", "no"):
        return None
    return {
        "enabled": True,
        "schema": "tiered_policy",
        "permission_mode": _env_str("CUA_PERMISSION_MODE", default="normal"),
        "tools": {},
        "defaults": {"*": "allow"},
        "rules": [],
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


async def main() -> None:
    _preflight_driver()
    model = _build_model()
    task = _default_task()
    permissions = _permissions_config()

    print(f"[task]  {task}")
    gating = "ON (launch_app / type_text / kill_app need approval)" if permissions else "OFF"
    print(f"[perms] permission gating: {gating}")
    print("[watch] your screen — the agent runs in the background without stealing focus.\n")

    agent = create_cua_agent(
        model,
        language=_env_str("CUA_LANGUAGE", default="en"),
        max_iterations=int(_env_str("CUA_MAX_ITERATIONS", default="20")),
        rails=[_ToolTimelineRail()],
        permissions=permissions,
    )

    from openjiuwen.core.runner import Runner

    await Runner.start()
    try:
        await agent.ensure_initialized()
        result = await _run_with_approvals(Runner, agent, task, f"cua-action-{uuid.uuid4().hex[:8]}")
    finally:
        await Runner.stop()

    print("\n=== RESULT ===")
    print("result_type:", result.get("result_type"))
    print("output:\n", result.get("output", result))


if __name__ == "__main__":
    # Driver/agent output contains non-ASCII (✅, →); force UTF-8 so printing
    # it does not crash on a Windows cp1252 console.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(main())

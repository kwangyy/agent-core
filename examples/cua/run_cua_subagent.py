#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Coordinator DeepAgent that delegates a desktop task to the cua subagent.

A coordinator agent is configured with three specialists declared as
``SubAgentConfig`` specs (not live instances):

* ``cua_agent``     — host desktop, via cua-driver MCP tools
* ``browser_agent`` — web pages, via the Playwright MCP server
* ``code_agent``    — code and files

The coordinator has no desktop or browser tools of its own: it must call
the task tool, which materializes the matching spec through its
``factory_name`` (``DeepAgent.create_subagent`` dispatch) on first use.

The default task deliberately crosses the desktop/browser boundary to show
how mixed work gets split:

1. Desktop observation (``cua_agent``): the user's own Chrome window is a
   NATIVE DESKTOP WINDOW — reading its title is desktop work, but its web
   content must never be driven through desktop input.
2. Web (``browser_agent``): pages load and run in the specialist's own
   Playwright-managed browser, never in the user's Chrome.
3. Desktop action (``cua_agent``): the number extracted from the web feeds
   a Calculator computation, so the coordinator must sequence the
   delegations instead of firing them in parallel.

``code_agent`` stays dormant (specs are lazy, so its runtime is never
materialized).

Prerequisites: everything from ``cua_agent_demo.py`` (cua-driver daemon
running, ``examples/cua/.env`` with an LLM endpoint), plus Node.js —
``browser_agent`` starts the Playwright MCP server via
``npx -y @playwright/mcp@latest``.

Run from repository root::

    uv run python examples/cua/run_cua_subagent.py

Override the delegated task without editing the file::

    CUA_TASK="..." uv run python examples/cua/run_cua_subagent.py

Desktop tool gating (on by default)::

    CUA_PERMISSIONS = 0  # disable the permission engine entirely
    CUA_APPROVE = allow  # auto-answer approval prompts (non-interactive)

``launch_app`` of shells/script interpreters is hard-denied (a live run showed
the subagent improvising an elevated PowerShell as a Calculator fallback),
known demo apps (calc, calculator, notepad) are pre-approved, and any other
gated call pauses for a console y/N answered in-process -- the built-in ASK
interrupt cannot cross the task tool.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

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

from openjiuwen.core.single_agent.schema.agent_card import AgentCard  # noqa: E402
from openjiuwen.harness.factory import create_deep_agent  # noqa: E402
from openjiuwen.harness.security.host import (  # noqa: E402
    PermissionConfirmationRequest,
    ToolPermissionHost,
)
from openjiuwen.harness.security.models import PermissionConfirmResponse  # noqa: E402
from openjiuwen.harness.subagents.browser_agent import build_browser_agent_config  # noqa: E402
from openjiuwen.harness.subagents.code_agent import build_code_agent_config  # noqa: E402
from openjiuwen.harness.subagents.cua_agent import build_cua_agent_config  # noqa: E402
from openjiuwen.harness.tools.cua.config import build_cua_driver_mcp_config  # noqa: E402

_BASE64_PATTERN = re.compile(r"(data:image/[^;]+;base64,|\"data\": ?\")([A-Za-z0-9+/=]{16})[A-Za-z0-9+/=]{64,}")

COORDINATOR_SYSTEM_PROMPT = (
    "You are a coordinator agent. You do not control the desktop, the browser, "
    "or the filesystem yourself. You have three specialists reachable through "
    "the task tool: 'cua_agent' operates the host desktop and its native "
    "application windows; 'browser_agent' automates web pages; 'code_agent' "
    "works with code and files. "
    "Where the desktop and browser domains intersect, split along this line: "
    "the user's own browser windows (Chrome, Edge, ...) are native desktop "
    "windows — observing them (titles, layout, which app is open) is desktop "
    "work for cua_agent — but their web content must never be driven through "
    "desktop clicks or keystrokes. Anything that loads, reads, or interacts "
    "with a web page happens in browser_agent's own Playwright-managed "
    "browser, which is separate from the user's browser. "
    "When one delegation's output feeds another, run them in sequence and "
    "pass the concrete value along; only independent delegations may run in "
    "parallel. When the specialists report back, synthesize the findings into "
    "a concise final answer yourself; do not delegate the summarizing."
)


class _TruncatingStream:
    """Truncate huge base64 screenshot blobs in console output."""

    def __init__(self, original: Any) -> None:
        self._original = original

    def write(self, text: str) -> None:
        if isinstance(text, str) and len(text) > 4096:
            text = _BASE64_PATTERN.sub(r"\1\2...(truncated)", text)
        self._original.write(text)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


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
    """Fail fast with a clear hint when the cua-driver daemon is unreachable."""
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


def _default_task() -> str:
    return _env_str("CUA_TASK") or (
        "Three-part job; part 3 depends on part 2.\n"
        "1. On this desktop, report the title of whatever page is currently "
        "open in the user's Chrome window. Do not click, type, or interact "
        "with Chrome in any way — its window title is enough.\n"
        "2. Navigate to https://example.com and report the exact text of the "
        "page's main heading.\n"
        "3. Count the characters in that heading text (including spaces) "
        "yourself, then compute <count> times 3 in the Windows Calculator by "
        "clicking its buttons (launch it if it is not already open), and "
        "report the number shown on the calculator display.\n"
        "Finish with a short report of which specialist handled which part "
        "and why the user's own Chrome was never driven."
    )


def _permissions_config() -> dict | None:
    """Tool-permission policy for the materialized cua subagent.

    ``strict`` mode turns the CRITICAL shell rule into a hard DENY: a live
    coordinator run showed the desktop agent improvising an elevated
    PowerShell when Calculator ignored its background clicks, and a shell is
    never an acceptable fallback for a GUI task. Known-safe demo apps are
    pre-approved so the happy path never prompts; every other launch_app /
    type_text asks through the hosted hook below.
    """
    if _env_str("CUA_PERMISSIONS", default="1").lower() in ("0", "false", "no"):
        return None
    return {
        "enabled": True,
        "schema": "tiered_policy",
        "permission_mode": "strict",
        "tools": {},
        "defaults": {"*": "allow"},
        "rules": [
            {
                "id": "cua_deny_shell_fallback",
                "description": "launch_app must never spawn shells or script interpreters",
                "tools": ["mcp_cua-driver_launch_app"],
                "match_type": "os_control",
                "pattern": (
                    "re:(?i)^(powershell|pwsh|cmd|wt|conhost|wsl|bash|sh|"
                    "regedit|rundll32|mshta|cscript|wscript)(\\.exe)?$"
                ),
                "severity": "CRITICAL",
            }
        ],
        # DENY rules above are checked BEFORE these overrides, so widening the
        # allowlist can never re-admit a shell.
        "approval_overrides": [
            {
                "id": "cua_allow_demo_apps",
                "action": "allow",
                "tools": ["mcp_cua-driver_launch_app"],
                "match_type": "os_control",
                "pattern": "re:(?i)^(calc|calculator|notepad)(\\.exe)?$",
            }
        ],
    }


def _permission_host() -> ToolPermissionHost:
    """Answer ASK decisions in-process: console y/N, or CUA_APPROVE to force."""

    async def _confirm(req: PermissionConfirmationRequest) -> PermissionConfirmResponse:
        tool_call = req.tool_call
        name = str(getattr(tool_call, "name", "") or "")
        short = name.split("_", 2)[-1] if name.startswith("mcp_") else name
        args = getattr(tool_call, "arguments", None)
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = None
        detail = ""
        if isinstance(args, dict):
            for key in ("name", "bundle_id", "text", "pid"):
                if key in args:
                    detail = f"  {key}={args[key]!r}"
                    break
        print(f"[approval] cua subagent wants to run  {short}{detail}", flush=True)
        forced = _env_str("CUA_APPROVE").lower()
        if forced in ("allow", "approve", "yes", "y"):
            print("[approval]   -> auto-APPROVED (CUA_APPROVE)", flush=True)
            return PermissionConfirmResponse(approved=True)
        if forced in ("deny", "reject", "no", "n"):
            print("[approval]   -> auto-DENIED (CUA_APPROVE)", flush=True)
            return PermissionConfirmResponse(approved=False)
        try:
            answer = input("[approval]   approve this action? [y/N] ").strip().lower()
        except EOFError:
            print("[approval]   -> no console input; DENYING (safe default)", flush=True)
            return PermissionConfirmResponse(approved=False)
        approved = answer in ("y", "yes")
        print(f"[approval]   -> {'APPROVED' if approved else 'DENIED'}", flush=True)
        return PermissionConfirmResponse(approved=approved)

    return ToolPermissionHost(request_permission_confirmation=_confirm)


def _build_coordinator(model, permissions, permission_host):
    language = _env_str("CUA_LANGUAGE", default="en")
    specs = [
        build_cua_agent_config(
            model,
            language=language,
            # Factory default (25): a perceive-act-verify loop re-snapshots
            # before every element action, so budgets tighter than ~20 make the
            # subagent run dry mid-task and force the coordinator to re-delegate.
            max_iterations=int(_env_str("CUA_MAX_ITERATIONS", default="25")),
            permissions=permissions,
            permission_host=permission_host,
        ),
        build_browser_agent_config(model, language=language),
        build_code_agent_config(model, language=language),
    ]
    return create_deep_agent(
        model=model,
        card=AgentCard(
            name="coordinator",
            description="Coordinator that routes work to desktop, browser, and code specialists.",
        ),
        system_prompt=COORDINATOR_SYSTEM_PROMPT,
        subagents=specs,
        language=language,
        max_iterations=int(_env_str("COORDINATOR_MAX_ITERATIONS", default="10")),
    )


async def main() -> None:
    _preflight_driver()
    model = _build_model()
    task = _default_task()

    permissions = _permissions_config()
    permission_host = _permission_host() if permissions else None

    print(f"task={task!r}")
    print("specialists: cua_agent (desktop) / browser_agent (web) / code_agent (code)")
    gating = (
        "ON (shell launch_app denied; calc/calculator/notepad pre-approved; other gated calls prompt)"
        if permissions
        else "OFF (CUA_PERMISSIONS=0)"
    )
    print(f"[perms] cua subagent gating: {gating}")

    agent = _build_coordinator(model, permissions, permission_host)

    from openjiuwen.core.runner import Runner

    await Runner.start()
    try:
        await agent.ensure_initialized()
        result = await Runner.run_agent(
            agent,
            {"query": task, "conversation_id": f"cua-coord-{uuid.uuid4().hex[:8]}"},
        )
    finally:
        await Runner.stop()

    print("\n=== RESULT ===")
    if isinstance(result, dict):
        print("result_type:", result.get("result_type"))
        print("output:\n", result.get("output", result))
    else:
        print(result)


if __name__ == "__main__":
    sys.stdout = _TruncatingStream(sys.stdout)
    sys.stderr = _TruncatingStream(sys.stderr)
    asyncio.run(main())

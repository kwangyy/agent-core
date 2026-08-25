#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Run the cua desktop subagent against the local cua-driver daemon.

Prerequisites:
1. cua-driver installed and its daemon running in an interactive desktop
   session (install: https://cua.ai/docs — then ``cua-driver autostart kick``).
2. An LLM endpoint configured in ``examples/cua/.env`` (copy
   ``.env.template`` and fill in the values).

Run from repository root::

    uv run python examples/cua/cua_agent_demo.py

Override the task without editing the file::

    CUA_TASK="Open Notepad and type hello" uv run python examples/cua/cua_agent_demo.py
"""

from __future__ import annotations

import asyncio
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

from openjiuwen.harness.subagents.cua_agent import create_cua_agent  # noqa: E402
from openjiuwen.harness.tools.cua.config import build_cua_driver_mcp_config  # noqa: E402

_BASE64_PATTERN = re.compile(r"(data:image/[^;]+;base64,|\"data\": ?\")([A-Za-z0-9+/=]{16})[A-Za-z0-9+/=]{64,}")


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
        # get_screen_size is cheap, read-only, and risk-classified on all
        # platforms (health_report is denied under the driver's default
        # permission mode: "no reviewed risk classification").
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
        detail = (probe.stderr or probe.stdout).strip()[:300] if probe else "health_report timed out"
        print(
            f"cua-driver daemon is not reachable. Start it with:\n  cua-driver autostart kick\nProbe output: {detail}",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"[preflight] cua-driver daemon reachable via {command!r}")


def _default_task() -> str:
    return _env_str("CUA_TASK") or (
        "List the application windows currently open on this desktop and summarize what the user "
        "appears to be working on. Do not click, type, or change anything."
    )


async def main() -> None:
    _preflight_driver()
    model = _build_model()
    task = _default_task()
    capabilities_raw = _env_str("CUA_CAPABILITIES")
    cua_capabilities = [item.strip() for item in capabilities_raw.split(",") if item.strip()] or None

    print(f"task={task!r}")
    print(f"cua_capabilities={cua_capabilities or 'default (core + input + app_lifecycle)'}")

    agent = create_cua_agent(
        model,
        language=_env_str("CUA_LANGUAGE", default="en"),
        max_iterations=int(_env_str("CUA_MAX_ITERATIONS", default="15")),
        cua_capabilities=cua_capabilities,
    )

    from openjiuwen.core.runner import Runner

    await Runner.start()
    try:
        await agent.ensure_initialized()
        result = await Runner.run_agent(
            agent,
            {"query": task, "conversation_id": f"cua-demo-{uuid.uuid4().hex[:8]}"},
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

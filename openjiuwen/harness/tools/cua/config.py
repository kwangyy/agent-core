# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""MCP configuration for the cua-driver desktop runtime."""

from __future__ import annotations

import os
import re
import shlex
import shutil
from pathlib import Path
from typing import Any, Dict

from openjiuwen.core.foundation.tool import McpServerConfig


DEFAULT_CUA_DRIVER_MCP_COMMAND = "cua-driver"
DEFAULT_CUA_DRIVER_MCP_ARGS = "mcp"
DEFAULT_CUA_DRIVER_MCP_TIMEOUT_S = 120

CUA_DRIVER_MCP_SERVER_ID = "cua_driver_stdio"
CUA_DRIVER_MCP_SERVER_NAME = "cua-driver"

# Environment variables the driver honors that are worth forwarding to the
# spawned MCP proxy. CUA_DRIVER_POLICY_FILE enables the driver's own YAML/Rego
# permission layer as defense in depth below the agent-side allowlist.
_FORWARDED_ENV_KEYS = (
    "CUA_DRIVER_POLICY_FILE",
    "CUA_DRIVER_MANAGED_POLICY_FILE",
)

# Default Windows install location used by the official installer; the binary
# may not be on PATH in processes started before the installer's PATH update.
_WINDOWS_DEFAULT_INSTALL_RELATIVE = Path("Programs") / "Cua" / "cua-driver" / "bin" / "cua-driver.exe"


def _sanitize_instance_key(instance_key: str) -> str:
    """Reduce an instance key to id-safe characters (``[A-Za-z0-9_-]``)."""
    return re.sub(r"[^A-Za-z0-9_-]+", "-", (instance_key or "").strip()).strip("-")


def _resolve_command() -> str:
    configured = (os.getenv("CUA_DRIVER_MCP_COMMAND") or "").strip()
    if configured:
        return configured
    if shutil.which(DEFAULT_CUA_DRIVER_MCP_COMMAND):
        return DEFAULT_CUA_DRIVER_MCP_COMMAND
    local_app_data = (os.getenv("LOCALAPPDATA") or "").strip()
    if local_app_data:
        candidate = Path(local_app_data) / _WINDOWS_DEFAULT_INSTALL_RELATIVE
        if candidate.is_file():
            return str(candidate)
    return DEFAULT_CUA_DRIVER_MCP_COMMAND


def _resolve_timeout_s() -> int:
    raw = (os.getenv("CUA_DRIVER_MCP_TIMEOUT_S") or "").strip()
    if not raw:
        return DEFAULT_CUA_DRIVER_MCP_TIMEOUT_S
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_CUA_DRIVER_MCP_TIMEOUT_S
    return value if value >= 1 else DEFAULT_CUA_DRIVER_MCP_TIMEOUT_S


def build_cua_driver_mcp_config(instance_key: str = "") -> McpServerConfig:
    """Build the stdio MCP config that spawns ``cua-driver mcp``.

    The spawned process is a thin proxy: all tool execution happens in the
    machine-owned ``cua-driver serve`` daemon, which must already be running
    in an interactive desktop session.

    Agents sharing the same ``instance_key`` (or none) intentionally share one
    MCP registration; a distinct key isolates the ``server_id``.
    """
    command = _resolve_command()
    args = shlex.split(os.getenv("CUA_DRIVER_MCP_ARGS", DEFAULT_CUA_DRIVER_MCP_ARGS))

    env_map: Dict[str, str] = {}
    for key in _FORWARDED_ENV_KEYS:
        value = os.getenv(key)
        if value:
            env_map[key] = value

    params: Dict[str, Any] = {
        "command": command,
        "args": args,
        "timeout_s": _resolve_timeout_s(),
    }
    if env_map:
        params["env"] = env_map

    server_id = CUA_DRIVER_MCP_SERVER_ID
    server_name = CUA_DRIVER_MCP_SERVER_NAME
    sanitized_key = _sanitize_instance_key(instance_key)
    if sanitized_key:
        server_id = f"{server_id}__{sanitized_key}"
        server_name = f"{server_name}-{sanitized_key}"

    return McpServerConfig(
        server_id=server_id,
        server_name=server_name,
        server_path="stdio://cua-driver",
        client_type="stdio",
        params=params,
        # cua-driver puts window bounds / screen size only in structuredContent;
        # without this the model never sees any coordinates.
        include_structured_content=True,
    )


__all__ = [
    "CUA_DRIVER_MCP_SERVER_ID",
    "CUA_DRIVER_MCP_SERVER_NAME",
    "DEFAULT_CUA_DRIVER_MCP_ARGS",
    "DEFAULT_CUA_DRIVER_MCP_COMMAND",
    "DEFAULT_CUA_DRIVER_MCP_TIMEOUT_S",
    "build_cua_driver_mcp_config",
]

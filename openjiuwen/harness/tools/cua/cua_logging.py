# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Per-session ``cua_agent.log`` files, modeled on the browser agent's log.

Each cua run writes to ``<log dir>/<session id>/cua_agent.log`` so parallel runs
stay apart. While a run is active, ``CuaRuntimeRail`` holds its file open and
sets the cua log context to its session id; every framework log line emitted in
that context (LLM calls, tool execution, rails) then goes to that file instead
of the combined application logs, and the desktop tools add one line per action.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from threading import Lock
from typing import Any

from openjiuwen.core.common.logging import logger as common_logger
from openjiuwen.core.common.logging.cua_context import cua_agent_logger_name, get_cua_agent_log_session

_CUA_HANDLER_MARKER = "_openjiuwen_cua_agent_file_handler"
_LOG_FILE_NAME = "cua_agent.log"
_FALSE_VALUES = {"0", "false", "no", "off", ""}
_DISABLE_VALUES = {"0", "false", "no", "off", "none", "null", "-"}
_UNSAFE_PATH_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_LOCK = Lock()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in _FALSE_VALUES


def _get_level() -> int:
    level_name = os.getenv("OPENJIUWEN_CUA_AGENT_LOG_LEVEL", "INFO")
    return getattr(logging, level_name.strip().upper(), logging.INFO)


def _safe_dir_name(session_id: str) -> str:
    """A session id reduced to a single safe path component."""
    return _UNSAFE_PATH_CHARS.sub("_", session_id).strip("._") or "unknown_session"


def get_cua_agent_log_dir() -> Path | None:
    """Return the root under which each session gets its own folder, or None when disabled.

    Defaults to ./logs/cua_agent. Override with OPENJIUWEN_CUA_AGENT_LOG_DIR, or
    set it to one of 0/false/no/off/none/null/- to disable the dedicated files.
    """
    configured = os.getenv("OPENJIUWEN_CUA_AGENT_LOG_DIR")
    if configured is None:
        return (Path.cwd() / "logs" / "cua_agent").resolve()

    configured = configured.strip()
    if configured.lower() in _DISABLE_VALUES:
        return None

    return Path(configured).expanduser().resolve()


def get_cua_agent_log_path(session_id: str) -> Path | None:
    """Return ``<log dir>/<session id>/cua_agent.log``, or None when disabled."""
    log_dir = get_cua_agent_log_dir()
    if log_dir is None:
        return None
    return log_dir / _safe_dir_name(session_id) / _LOG_FILE_NAME


def _file_handlers(cua_logger: logging.Logger) -> list[logging.Handler]:
    return [h for h in cua_logger.handlers if getattr(h, _CUA_HANDLER_MARKER, False)]


def open_cua_agent_log(session_id: str) -> logging.Logger | None:
    """Attach the session's file handler (append mode) and return its logger; None when disabled.

    Logs are not propagated to the combined application log unless
    OPENJIUWEN_CUA_AGENT_LOG_MIRROR_COMMON=1.
    """
    log_path = get_cua_agent_log_path(session_id)
    if log_path is None:
        return None

    cua_logger = logging.getLogger(cua_agent_logger_name(session_id))
    cua_logger.setLevel(_get_level())
    cua_logger.propagate = _env_bool("OPENJIUWEN_CUA_AGENT_LOG_MIRROR_COMMON", default=False)

    with _LOCK:
        if any(getattr(h, "baseFilename", None) == str(log_path) for h in _file_handlers(cua_logger)):
            return cua_logger
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        setattr(file_handler, _CUA_HANDLER_MARKER, True)
        file_handler.setLevel(_get_level())
        file_handler.setFormatter(logging.Formatter("%(asctime)s | cua_agent | %(levelname)s | %(message)s"))
        cua_logger.addHandler(file_handler)
    return cua_logger


def close_cua_agent_log(session_id: str) -> None:
    """Detach and close the session's file handler so a finished run holds no file open."""
    cua_logger = logging.getLogger(cua_agent_logger_name(session_id))
    with _LOCK:
        for handler in _file_handlers(cua_logger):
            cua_logger.removeHandler(handler)
            handler.close()


def _write(level: int, message: str, args: tuple[Any, ...], session_id: str | None) -> None:
    session_id = session_id or get_cua_agent_log_session()
    if session_id is None or get_cua_agent_log_dir() is None:
        if level >= logging.WARNING:
            common_logger.warning(message, *args)
        else:
            common_logger.info(message, *args)
        return
    cua_logger = logging.getLogger(cua_agent_logger_name(session_id))
    # A line for a session whose run is not active (e.g. TaskTool's line before the
    # subagent starts) opens the file just for this write.
    transient = not _file_handlers(cua_logger)
    if transient:
        open_cua_agent_log(session_id)
    try:
        cua_logger.log(level, message, *args)
    finally:
        if transient:
            close_cua_agent_log(session_id)


def cua_agent_log_info(message: str, *args: Any, session_id: str | None = None) -> None:
    """Write to the cua log of ``session_id`` (default: the active cua run), else the common log."""
    _write(logging.INFO, message, args, session_id)


def cua_agent_log_warning(message: str, *args: Any, session_id: str | None = None) -> None:
    """Write to the cua log of ``session_id`` (default: the active cua run), else the common log."""
    _write(logging.WARNING, message, args, session_id)


__all__ = [
    "close_cua_agent_log",
    "cua_agent_log_info",
    "cua_agent_log_warning",
    "get_cua_agent_log_dir",
    "get_cua_agent_log_path",
    "open_cua_agent_log",
]

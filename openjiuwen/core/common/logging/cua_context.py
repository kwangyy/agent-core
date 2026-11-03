# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

from __future__ import annotations

from contextvars import ContextVar, Token

CUA_AGENT_LOGGER_NAME = "openjiuwen.cua_agent"

# The session id of the cua run whose logs this context belongs to, or None outside a cua run.
_CUA_AGENT_LOG_SESSION: ContextVar[str | None] = ContextVar(
    "openjiuwen_cua_agent_log_session",
    default=None,
)


def get_cua_agent_log_session() -> str | None:
    """Session id of the active cua run, or None outside one."""
    return _CUA_AGENT_LOG_SESSION.get()


def is_cua_agent_log_context() -> bool:
    """Whether log lines are currently routed to a cua session file."""
    return _CUA_AGENT_LOG_SESSION.get() is not None


def cua_agent_logger_name(session_id: str) -> str:
    """Name of the stdlib logger that writes one cua session's log file."""
    return f"{CUA_AGENT_LOGGER_NAME}.{session_id}"


def set_cua_agent_log_context(session_id: str) -> Token[str | None]:
    """Route log lines in this context to the session's cua_agent.log."""
    return _CUA_AGENT_LOG_SESSION.set(session_id)


def reset_cua_agent_log_context(token: Token[str | None]) -> None:
    """Undo set_cua_agent_log_context with the token it returned."""
    _CUA_AGENT_LOG_SESSION.reset(token)

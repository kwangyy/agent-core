# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def cua_log_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep every cua test's per-session log files out of the repository's logs/ folder."""
    log_dir = tmp_path / "cua_logs"
    monkeypatch.setenv("OPENJIUWEN_CUA_AGENT_LOG_DIR", str(log_dir))
    monkeypatch.delenv("OPENJIUWEN_CUA_AGENT_LOG_MIRROR_COMMON", raising=False)
    return log_dir

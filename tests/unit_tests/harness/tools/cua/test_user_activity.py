# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""The user-activity probe answers "how long since a human touched the desktop"."""

import sys

import pytest

from openjiuwen.harness.tools.cua.user_activity import (
    WindowsLastInputProbe,
    build_default_user_activity_probe,
)


@pytest.mark.skipif(sys.platform != "win32", reason="GetLastInputInfo is Windows-only")
def test_windows_probe_reports_a_non_negative_age() -> None:
    age = WindowsLastInputProbe().seconds_since_last_input()
    assert age is not None
    assert 0.0 <= age < 60 * 60 * 24 * 60


def test_default_probe_matches_the_platform() -> None:
    probe = build_default_user_activity_probe()
    if sys.platform == "win32":
        assert isinstance(probe, WindowsLastInputProbe)
    else:
        assert probe is None

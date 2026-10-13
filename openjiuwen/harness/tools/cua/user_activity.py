# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Probes that report how long ago the user last touched the desktop.

The cua-driver daemon exposes no "user is active" signal, so the takeover rail
asks the OS directly. On Windows ``GetLastInputInfo`` returns the tick of the
last mouse or keyboard event in the interactive session, which is exactly the
question "is a human using this machine right now". Other platforms have no
probe yet; the rail is inert without one.
"""

from __future__ import annotations

import sys
from typing import Optional, Protocol


class UserActivityProbe(Protocol):
    """Reports how long ago the user last touched the desktop."""

    def seconds_since_last_input(self) -> Optional[float]:
        """Seconds since the last user input event, or ``None`` when unknown."""


class WindowsLastInputProbe:
    """``GetLastInputInfo``-backed probe (Windows only).

    Both ``dwTime`` and ``GetTickCount`` are 32-bit millisecond counters that
    wrap together every ~49.7 days, so the difference is taken modulo 2**32.
    """

    @staticmethod
    def seconds_since_last_input() -> Optional[float]:
        """Seconds since the last mouse or keyboard event, or ``None`` if the call fails."""
        try:
            import ctypes
            from ctypes import wintypes

            class LastInputInfo(ctypes.Structure):
                """Win32 LASTINPUTINFO structure."""

                _fields_ = [("cb_size", wintypes.UINT), ("dw_time", wintypes.DWORD)]

            info = LastInputInfo()
            info.cb_size = ctypes.sizeof(LastInputInfo)
            if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
                return None
            now = ctypes.windll.kernel32.GetTickCount()
            return ((int(now) - int(info.dw_time)) & 0xFFFFFFFF) / 1000.0
        except Exception:  # noqa: BLE001 - a broken probe must never block desktop work
            return None


def build_default_user_activity_probe() -> Optional[UserActivityProbe]:
    """The platform probe, or ``None`` where user activity cannot be observed."""
    if sys.platform == "win32":
        return WindowsLastInputProbe()
    return None


__all__ = [
    "UserActivityProbe",
    "WindowsLastInputProbe",
    "build_default_user_activity_probe",
]

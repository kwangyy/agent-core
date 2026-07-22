# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Tests for the os_control (cua-driver desktop) tier of the permission policy."""

from openjiuwen.harness.security.models import PermissionLevel
from openjiuwen.harness.security.permission_engine.toolguard.builtin_rules import (
    inline_package_command_rules,
)
from openjiuwen.harness.security.tiered_policy import (
    _os_control_pattern_matches,
    _tool_category,
    evaluate_tiered_policy,
    rule_tools_category_consistent,
)


def test_desktop_control_tools_are_os_control_category() -> None:
    assert _tool_category("mcp_cua-driver_click") == "os_control"
    assert _tool_category("mcp_cua-driver_type_text") == "os_control"
    assert _tool_category("mcp_cua-driver_kill_app") == "os_control"
    # Read-only tools are intentionally uncategorized (baseline/default govern them).
    assert _tool_category("mcp_cua-driver_list_windows") is None
    assert _tool_category("mcp_cua-driver_get_window_state") is None


def test_os_control_tools_do_not_mix_with_other_categories() -> None:
    assert rule_tools_category_consistent(["mcp_cua-driver_kill_app", "mcp_cua-driver_launch_app"])
    # Mixing an os_control tool with a shell tool must be rejected.
    assert not rule_tools_category_consistent(["mcp_cua-driver_kill_app", "bash"])


def test_wildcard_pattern_matches_any_invocation() -> None:
    # kill_app carries only a pid (no string args); "*" must still match.
    assert _os_control_pattern_matches("*", {"pid": 844})
    assert _os_control_pattern_matches("*", {})


def test_arg_pattern_matches_relevant_string_args() -> None:
    assert _os_control_pattern_matches("re:(?i)password", {"text": "my PASSWORD is x"})
    assert not _os_control_pattern_matches("re:(?i)password", {"text": "hello world"})
    # keys arrays (hotkey) are joined before matching.
    assert _os_control_pattern_matches("re:cmd", {"keys": ["cmd", "c"]})


def _builtin_only_config() -> dict:
    # No user rules / tools / defaults: isolate the built-in os_control rules.
    # Builtin rules only take effect once inlined (``layer: builtin``);
    # evaluate_tiered_policy itself never loads the package YAML.
    return inline_package_command_rules({})


def test_kill_app_is_denied_by_default() -> None:
    # CRITICAL severity -> DENY, same convention as every other builtin rule
    # (e.g. shell_system_shutdown_or_reboot); unsaved data loss is not an
    # ask-first decision.
    level, matched = evaluate_tiered_policy(_builtin_only_config(), "mcp_cua-driver_kill_app", {"pid": 844})
    assert level == PermissionLevel.DENY
    assert "cua_desktop_force_terminate_app" in matched


def test_launch_app_and_type_text_ask_by_default() -> None:
    launch, launch_rule = evaluate_tiered_policy(
        _builtin_only_config(), "mcp_cua-driver_launch_app", {"name": "Notepad"}
    )
    assert launch == PermissionLevel.ASK
    assert "cua_desktop_launch_app" in launch_rule

    typed, typed_rule = evaluate_tiered_policy(
        _builtin_only_config(), "mcp_cua-driver_type_text", {"text": "hello", "pid": 5}
    )
    assert typed == PermissionLevel.ASK
    assert "cua_desktop_type_text" in typed_rule


def test_read_only_and_click_tools_are_not_forced_to_ask() -> None:
    # Uncategorized cua tools fall through to the no-config fallback, not a
    # built-in ASK rule — deployers decide via defaults/tools config.
    _, matched = evaluate_tiered_policy(_builtin_only_config(), "mcp_cua-driver_list_windows", {})
    assert "cua_desktop" not in matched
    _, click_matched = evaluate_tiered_policy(
        _builtin_only_config(), "mcp_cua-driver_click", {"pid": 5, "element_index": 2}
    )
    assert "cua_desktop" not in click_matched

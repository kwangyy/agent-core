# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Tests for cua-driver capability resolution."""

from openjiuwen.harness.tools.cua.cua_capabilities import (
    APP_LIFECYCLE_CUA_TOOL_NAMES,
    BROWSER_CUA_TOOL_NAMES,
    CORE_CUA_TOOL_NAMES,
    DEFAULT_CUA_AGENT_CAPABILITY_NAMES,
    DEFAULT_CUA_CAPABILITIES,
    INPUT_CUA_TOOL_NAMES,
    resolve_cua_capabilities,
)


def test_core_only_selection_exposes_exactly_core_tools() -> None:
    resolved = resolve_cua_capabilities([])

    assert resolved.requested_names == ()
    assert resolved.selected_names == ("core",)
    assert resolved.rejected_names == ()
    assert resolved.allowed_tool_names == CORE_CUA_TOOL_NAMES


def test_input_selection_adds_input_tools_to_core() -> None:
    resolved = resolve_cua_capabilities(["input"])

    assert resolved.selected_names == ("core", "input")
    assert resolved.allowed_tool_names == CORE_CUA_TOOL_NAMES + INPUT_CUA_TOOL_NAMES
    assert not set(APP_LIFECYCLE_CUA_TOOL_NAMES).intersection(resolved.allowed_tool_names)


def test_multiple_categories_preserve_requested_order() -> None:
    resolved = resolve_cua_capabilities(["app_lifecycle", "input"])

    assert resolved.requested_names == ("app_lifecycle", "input")
    assert resolved.selected_names == ("core", "app_lifecycle", "input")
    assert resolved.allowed_tool_names == (CORE_CUA_TOOL_NAMES + APP_LIFECYCLE_CUA_TOOL_NAMES + INPUT_CUA_TOOL_NAMES)


def test_duplicate_categories_are_deduplicated_stably() -> None:
    resolved = resolve_cua_capabilities(["input", "input", "recording", "input"])

    assert resolved.requested_names == ("input", "recording")
    assert resolved.selected_names == ("core", "input", "recording")
    assert len(resolved.allowed_tool_names) == len(set(resolved.allowed_tool_names))


def test_unknown_category_is_reported_as_rejected() -> None:
    resolved = resolve_cua_capabilities(["unknown", "input"])

    assert resolved.selected_names == ("core", "input")
    assert resolved.rejected_names == ("unknown",)


def test_browser_bundle_is_not_offered_by_the_catalog() -> None:
    # Browser automation is the browser agent's job: cua-driver's CDP tools
    # must stay unreachable even if a caller names the bundle explicitly.
    resolved = resolve_cua_capabilities(["browser"])

    assert resolved.rejected_names == ("browser",)
    assert not set(BROWSER_CUA_TOOL_NAMES).intersection(resolved.allowed_tool_names)


def test_default_agent_capabilities_exclude_browser_tools() -> None:
    resolved = resolve_cua_capabilities(DEFAULT_CUA_AGENT_CAPABILITY_NAMES)

    assert resolved.rejected_names == ()
    assert resolved.selected_names == ("core", "input", "app_lifecycle")
    assert not set(BROWSER_CUA_TOOL_NAMES).intersection(resolved.allowed_tool_names)


def test_catalog_bundles_do_not_overlap() -> None:
    # A tool in two bundles would make capability revocation unreliable: the
    # allowlist would still contain it via the other bundle.
    all_names = [name for capability in DEFAULT_CUA_CAPABILITIES for name in capability.tool_names]
    assert len(all_names) == len(set(all_names))

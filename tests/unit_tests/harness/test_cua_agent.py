# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Tests for the cua desktop subagent factory and dispatch."""

from pathlib import Path
from unittest.mock import patch

import pytest

from openjiuwen.core.context_engine import ToolResultWindowProcessorConfig
from openjiuwen.core.foundation.llm import (
    Model,
    ModelClientConfig,
    ModelRequestConfig,
)
from openjiuwen.core.single_agent.schema.agent_card import AgentCard
from openjiuwen.harness.factory import create_deep_agent
from openjiuwen.harness.rails.context_engineer import ContextProcessorRail
from openjiuwen.harness.rails.multimodal_context_summarizer_rail import MultimodalContextSummarizerRail
from openjiuwen.harness.schema.config import SubAgentConfig
from openjiuwen.harness.subagents.cua_agent import (
    CUA_AGENT_FACTORY_NAME,
    CUA_DELIVERY_MODE_PROMPT_SUFFIX,
    DEFAULT_CUA_AGENT_SYSTEM_PROMPT,
    build_cua_agent_config,
    create_cua_agent,
)
from openjiuwen.harness.tools.cua.config import (
    CUA_DRIVER_MCP_SERVER_ID,
    build_cua_driver_mcp_config,
)
from openjiuwen.harness.tools.cua.cua_capabilities import BROWSER_CUA_TOOL_NAMES
from openjiuwen.harness.tools.cua.rails import (
    CuaDeliveryModeRail,
    CuaElementAddressingRail,
    CuaRepeatFailureRail,
    CuaRuntimeRail,
    CuaScreenshotDownscaleRail,
    CuaSnapshotDedupRail,
    CuaSnapshotFreshnessRail,
)
from openjiuwen.harness.workspace.workspace import Workspace


def _create_dummy_model() -> Model:
    model_client_config = ModelClientConfig(
        client_provider="OpenAI",
        api_key="test-key",
        api_base="http://test-base",
        verify_ssl=False,
    )
    model_config = ModelRequestConfig(model="test-model")
    return Model(model_client_config=model_client_config, model_config=model_config)


def test_build_cua_driver_mcp_config_is_stdio_and_isolatable() -> None:
    cfg = build_cua_driver_mcp_config()

    assert cfg.client_type == "stdio"
    assert cfg.server_id == CUA_DRIVER_MCP_SERVER_ID
    assert cfg.params["args"] == ["mcp"]

    isolated = build_cua_driver_mcp_config("agent one!")
    assert isolated.server_id == f"{CUA_DRIVER_MCP_SERVER_ID}__agent-one"
    assert isolated.server_id != cfg.server_id


def test_build_cua_driver_mcp_config_honors_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("CUA_DRIVER_MCP_COMMAND", "/custom/cua-driver")
    monkeypatch.setenv("CUA_DRIVER_MCP_ARGS", "mcp --verbose")
    monkeypatch.setenv("CUA_DRIVER_MCP_TIMEOUT_S", "45")
    monkeypatch.setenv("CUA_DRIVER_POLICY_FILE", "/policies/cua.yaml")

    cfg = build_cua_driver_mcp_config()

    assert cfg.params["command"] == "/custom/cua-driver"
    assert cfg.params["args"] == ["mcp", "--verbose"]
    assert cfg.params["timeout_s"] == 45
    assert cfg.params["env"]["CUA_DRIVER_POLICY_FILE"] == "/policies/cua.yaml"


def test_build_cua_agent_config_uses_cua_factory() -> None:
    spec = build_cua_agent_config(_create_dummy_model(), language="en")

    assert isinstance(spec, SubAgentConfig)
    assert spec.agent_card.name == "cua_agent"
    assert spec.system_prompt == DEFAULT_CUA_AGENT_SYSTEM_PROMPT["en"]
    assert spec.factory_name == CUA_AGENT_FACTORY_NAME


def test_create_cua_agent_registers_mcp_and_allowlist_rail() -> None:
    agent = create_cua_agent(_create_dummy_model(), language="en")

    assert agent.card.name == "cua_agent"
    deep_config = agent.deep_config
    assert deep_config is not None
    mcp_ids = [mcp.server_id for mcp in deep_config.mcps or []]
    assert CUA_DRIVER_MCP_SERVER_ID in mcp_ids

    cua_rails = [rail for rail in agent._pending_rails if isinstance(rail, CuaRuntimeRail)]
    assert len(cua_rails) == 1
    allowed = cua_rails[0]._allowed_tool_names
    assert allowed is not None
    # Default grant: read-only core + input + app_lifecycle, never browser.
    assert "get_window_state" in allowed
    assert "click" in allowed
    assert "launch_app" in allowed
    assert not set(BROWSER_CUA_TOOL_NAMES).intersection(allowed)


def test_default_wiring_windows_cua_snapshot_results() -> None:
    """Snapshot results carry stale element_index handles and huge payloads,
    so the sliding window must be on by default or the agent burns its
    context window on unusable history. Retention must default to more than
    one: a two-window task needs both windows' latest snapshots in context at
    once, and with keep_last_k=1 the first window's frames evict the moment
    the second window is snapshotted (live-diagnosed loop of read-only
    re-queries)."""
    agent = create_cua_agent(_create_dummy_model(), language="en")

    context_rails = [rail for rail in agent._pending_rails if isinstance(rail, ContextProcessorRail)]
    assert len(context_rails) == 1
    assert context_rails[0]._preset is False
    processors = context_rails[0]._user_processors
    assert len(processors) == 1
    key, cfg = processors[0]
    assert key == "ToolResultWindowProcessor"
    assert isinstance(cfg, ToolResultWindowProcessorConfig)
    # Pin the intended contract literally (not against a source constant) so a
    # regression like a renamed tool silently dropping out is actually caught.
    assert cfg.tool_names == [
        "get_window_state",
        "get_desktop_state",
        "get_accessibility_tree",
    ]
    assert cfg.keep_last_k == 3


def test_snapshot_keep_last_k_drives_window_and_screenshot_retention() -> None:
    """One knob rules both evictions: the tool-result window and the retained
    screenshot images fail the same way on multi-window tasks, so they must
    move in lockstep rather than drift apart."""
    agent = create_cua_agent(
        _create_dummy_model(),
        language="en",
        cua_screenshot_multimodal=True,
        cua_snapshot_keep_last_k=5,
    )

    context_rails = [rail for rail in agent._pending_rails if isinstance(rail, ContextProcessorRail)]
    _, cfg = context_rails[0]._user_processors[0]
    assert cfg.keep_last_k == 5
    summarizer_rails = [r for r in agent._pending_rails if isinstance(r, MultimodalContextSummarizerRail)]
    assert summarizer_rails[0]._screenshots_to_keep == 5


def test_snapshot_keep_last_k_is_validated_at_config_time() -> None:
    # factory_kwargs are not resolved until the subagent is materialized, so an
    # unusable retention count must fail when the config is built, not mid-run.
    with pytest.raises(ValueError, match="cua_snapshot_keep_last_k"):
        build_cua_agent_config(_create_dummy_model(), cua_snapshot_keep_last_k=0)
    with pytest.raises(ValueError, match="cua_snapshot_keep_last_k"):
        create_cua_agent(_create_dummy_model(), cua_snapshot_keep_last_k=True)


def test_build_cua_agent_config_forwards_snapshot_keep_last_k() -> None:
    spec = build_cua_agent_config(_create_dummy_model(), cua_snapshot_keep_last_k=2)

    assert spec.factory_kwargs["cua_snapshot_keep_last_k"] == 2


def test_caller_context_processor_rail_suppresses_cua_injection() -> None:
    caller_rail = ContextProcessorRail(preset=False)
    agent = create_cua_agent(_create_dummy_model(), language="en", rails=[caller_rail])

    context_rails = [rail for rail in agent._pending_rails if isinstance(rail, ContextProcessorRail)]
    assert context_rails == [caller_rail]


def test_create_cua_agent_rejects_unknown_capabilities() -> None:
    with pytest.raises(ValueError, match="Unsupported cua capabilities: browser"):
        create_cua_agent(_create_dummy_model(), cua_capabilities=["browser"])


def test_delivery_mode_rail_is_absent_unless_a_mode_is_pinned() -> None:
    """Delivery stays the driver's own per-call decision by default, so
    existing callers keep their current behavior."""
    agent = create_cua_agent(_create_dummy_model(), language="en")

    assert [rail for rail in agent._pending_rails if isinstance(rail, CuaDeliveryModeRail)] == []


@pytest.mark.parametrize("mode", ["background", "foreground"])
def test_pinned_delivery_mode_installs_the_rail(mode: str) -> None:
    agent = create_cua_agent(_create_dummy_model(), language="en", cua_delivery_mode=mode)

    delivery_rails = [rail for rail in agent._pending_rails if isinstance(rail, CuaDeliveryModeRail)]
    assert len(delivery_rails) == 1
    assert delivery_rails[0]._delivery_mode == mode


@pytest.mark.parametrize("language", ["en", "cn"])
@pytest.mark.parametrize("mode", ["background", "foreground"])
def test_pinned_delivery_mode_appends_prompt_override(language: str, mode: str) -> None:
    """The base prompt teaches the model to manage delivery_mode itself
    (keep background, escalate after a verified failure). With a pinned mode
    the rail rejects or overrides exactly what that advice produces, costing
    a wasted rejected turn per run — so pinning must also rewrite what the
    model is told."""
    agent = create_cua_agent(_create_dummy_model(), language=language, cua_delivery_mode=mode)

    expected_suffix = CUA_DELIVERY_MODE_PROMPT_SUFFIX[language][mode]
    assert agent.deep_config.system_prompt == DEFAULT_CUA_AGENT_SYSTEM_PROMPT[language] + expected_suffix


def test_unpinned_delivery_mode_leaves_the_prompt_untouched() -> None:
    agent = create_cua_agent(_create_dummy_model(), language="en")

    assert agent.deep_config.system_prompt == DEFAULT_CUA_AGENT_SYSTEM_PROMPT["en"]


def test_delivery_override_is_appended_to_a_caller_supplied_prompt() -> None:
    """The rail enforces the pin regardless of whose prompt is in play, so a
    custom prompt must carry the override too — including the prompt baked
    into a SubAgentConfig by build_cua_agent_config, which reaches this
    factory as an explicit system_prompt."""
    agent = create_cua_agent(
        _create_dummy_model(),
        language="en",
        system_prompt="custom desktop prompt",
        cua_delivery_mode="background",
    )

    expected_suffix = CUA_DELIVERY_MODE_PROMPT_SUFFIX["en"]["background"]
    assert agent.deep_config.system_prompt == "custom desktop prompt" + expected_suffix


@pytest.mark.parametrize("language", ["en", "cn"])
def test_background_override_teaches_report_instead_of_escalate(language: str) -> None:
    """Under a background pin the driver's own 'background_unavailable'
    escalation advice is a trap: following it guarantees a rejection. The
    override must give the model a terminal alternative — report the
    limitation — or it will loop on foreground attempts."""
    suffix = CUA_DELIVERY_MODE_PROMPT_SUFFIX[language]["background"]
    marker = "report that limitation" if language == "en" else "如实作为任务结果汇报"
    assert marker in suffix


def test_delivery_mode_is_validated_at_config_time() -> None:
    # factory_kwargs are not resolved until the subagent is materialized, so an
    # unusable mode must fail when the config is built, not mid-run.
    with pytest.raises(ValueError, match="cua_delivery_mode"):
        build_cua_agent_config(_create_dummy_model(), cua_delivery_mode="sideways")


def test_build_cua_agent_config_forwards_delivery_mode_to_the_factory() -> None:
    spec = build_cua_agent_config(_create_dummy_model(), cua_delivery_mode="background")

    assert spec.factory_kwargs["cua_delivery_mode"] == "background"


def test_build_cua_driver_mcp_config_image_content_is_opt_in() -> None:
    assert build_cua_driver_mcp_config().include_image_content is False
    assert build_cua_driver_mcp_config(include_image_content=True).include_image_content is True


def test_screenshot_multimodal_is_off_by_default() -> None:
    """Blind mode is the status quo: no image bridge, no vision rails, so
    existing non-vision deployments keep byte-identical wiring."""
    agent = create_cua_agent(_create_dummy_model(), language="en")

    deep_config = agent.deep_config
    cua_mcps = [mcp for mcp in deep_config.mcps or [] if mcp.server_id == CUA_DRIVER_MCP_SERVER_ID]
    assert len(cua_mcps) == 1
    assert cua_mcps[0].include_image_content is False
    assert [r for r in agent._pending_rails if isinstance(r, CuaScreenshotDownscaleRail)] == []
    assert [r for r in agent._pending_rails if isinstance(r, MultimodalContextSummarizerRail)] == []


def test_screenshot_multimodal_wires_image_bridge_and_vision_rails() -> None:
    agent = create_cua_agent(_create_dummy_model(), language="en", cua_screenshot_multimodal=True)

    deep_config = agent.deep_config
    cua_mcps = [mcp for mcp in deep_config.mcps or [] if mcp.server_id == CUA_DRIVER_MCP_SERVER_ID]
    assert len(cua_mcps) == 1
    assert cua_mcps[0].include_image_content is True

    downscale_rails = [r for r in agent._pending_rails if isinstance(r, CuaScreenshotDownscaleRail)]
    assert len(downscale_rails) == 1

    summarizer_rails = [r for r in agent._pending_rails if isinstance(r, MultimodalContextSummarizerRail)]
    assert len(summarizer_rails) == 1
    # Screenshot retention follows cua_snapshot_keep_last_k (default 3) so a
    # two-window task keeps both windows' screens, mirroring the tool window.
    assert summarizer_rails[0]._screenshots_to_keep == 3


def test_build_cua_agent_config_forwards_screenshot_multimodal_to_the_factory() -> None:
    spec = build_cua_agent_config(_create_dummy_model(), cua_screenshot_multimodal=True)

    assert spec.factory_kwargs["cua_screenshot_multimodal"] is True


def test_create_subagent_uses_cua_agent_factory(tmp_path) -> None:
    workspace_root = tmp_path / "parent_workspace"
    parent = create_deep_agent(
        model=_create_dummy_model(),
        card=AgentCard(name="parent", description="parent"),
        system_prompt="parent prompt",
        workspace=Workspace(root_path=str(workspace_root)),
        subagents=[build_cua_agent_config(_create_dummy_model(), language="en")],
    )
    factory_result = object()

    with patch(
        "openjiuwen.harness.subagents.cua_agent.create_cua_agent",
        return_value=factory_result,
    ) as mock_create_cua_agent:
        sub = parent.create_subagent("cua_agent", "sub_session_id")

    assert sub is factory_result
    mock_create_cua_agent.assert_called_once()
    call_kwargs = mock_create_cua_agent.call_args.kwargs
    assert call_kwargs["card"].name == "cua_agent"
    assert Path(call_kwargs["workspace"].root_path).name == "sub_session_id"


@pytest.mark.parametrize("language", ["en", "cn"])
def test_prompt_claims_electron_apps_for_the_desktop_agent(language: str) -> None:
    """Electron apps must not be deferred to the browser agent.

    The prompt tells the agent that browser work belongs to browser_agent.
    Read without a carve-out, that also disowns Discord/Slack/VS Code, which
    are Chromium-rendered but are native windows the browser agent cannot
    reach by URL — so the task would be handed to an agent that must fail it.
    """
    prompt = DEFAULT_CUA_AGENT_SYSTEM_PROMPT[language]
    assert "Electron" in prompt
    for app in ("Discord", "Slack", "VS Code"):
        assert app in prompt, f"{language} prompt no longer names {app} as desktop work"


@pytest.mark.parametrize("language", ["en", "cn"])
def test_prompt_warns_that_the_first_chromium_snapshot_can_be_empty(language: str) -> None:
    """A bare first snapshot must read as 'retry', not 'this window is empty'.

    Chromium builds its accessibility tree only once something queries it, so
    the first get_window_state on an Electron window returns a lone Document.
    Without this, the agent concludes the window has no content and gives up
    one call before the tree it needs appears.
    """
    prompt = DEFAULT_CUA_AGENT_SYSTEM_PROMPT[language]
    assert "Chromium" in prompt
    assert "Document" in prompt


@pytest.mark.parametrize("language", ["en", "cn"])
def test_prompt_separates_the_two_coordinate_spaces(language: str) -> None:
    """Element frames and click coordinates live in different spaces.

    get_window_state reports frames in screen-absolute pixels while click/drag
    x,y are window-local. Feeding a frame straight into a click silently hits
    the wrong place — it does not error — so the distinction has to be stated.
    """
    prompt = DEFAULT_CUA_AGENT_SYSTEM_PROMPT[language]
    assert "scope='desktop'" in prompt
    assert "-32000" in prompt, "minimized-window sentinel dropped from the prompt"


@pytest.mark.parametrize(
    ("language", "marker", "margin_marker"),
    [("en", "fails silently", "30+ px"), ("cn", "静默失效", "30 像素")],
)
def test_prompt_warns_that_region_edges_are_unreliable(language: str, marker: str, margin_marker: str) -> None:
    """Coordinate actions starting outside a visually-estimated region no-op.

    VLM edge localization is off by tens of pixels, and a drag whose
    mouse-down lands just outside a surface (canvas, slider, drop zone)
    produces no ink, no error, and no signal — live-diagnosed: five runs of
    misdiagnosis because strokes started 20px above a drawing area. The
    prompt must both demand an inside-the-region margin and teach that a
    silent no-effect result means aim, not delivery.
    """
    prompt = DEFAULT_CUA_AGENT_SYSTEM_PROMPT[language]
    assert marker in prompt
    assert margin_marker in prompt


@pytest.mark.parametrize(("language", "marker"), [("en", "preemptively"), ("cn", "预先")])
def test_prompt_forbids_preemptive_foreground_delivery(language: str, marker: str) -> None:
    """Guessing 'foreground' steals the user's focus for no reason.

    cua-driver resolves background delivery itself and reports when it is
    impossible, so escalation must be driven by that error rather than by the
    agent's guess about what the target surface looks like. The marker is
    language-specific on purpose: 'foreground'/'background' already appeared
    in the prompt before this guidance existed, so asserting on those words
    would pass even if the guidance were deleted.
    """
    assert marker in DEFAULT_CUA_AGENT_SYSTEM_PROMPT[language]


def test_driver_contract_rails_are_installed_unconditionally() -> None:
    """Both repair failure modes of the cua-driver contract itself, so there is
    no configuration under which the agent is better off without them: one
    resolves an either/or the driver would reject outright, the other stops the
    agent re-sending a call the driver has already refused six times."""
    agent = create_cua_agent(_create_dummy_model(), language="en")

    assert any(isinstance(r, CuaElementAddressingRail) for r in agent._pending_rails)
    assert any(isinstance(r, CuaRepeatFailureRail) for r in agent._pending_rails)
    assert any(isinstance(r, CuaSnapshotFreshnessRail) for r in agent._pending_rails)
    assert any(isinstance(r, CuaSnapshotDedupRail) for r in agent._pending_rails)


def test_superseded_snapshots_keep_only_a_short_inline_preview() -> None:
    """A superseded element tree has stale element_index handles by
    construction, so the inline preview is dead weight: the processor default
    of 3000 chars leaves roughly 19k tokens of unusable tree over a 25-step run
    and is why the window processor measured only a 3.6% net saving. The
    placeholder still carries the offload handle, so the full result stays
    reachable when the agent actually wants it."""
    agent = create_cua_agent(_create_dummy_model(), language="en")

    context_rails = [rail for rail in agent._pending_rails if isinstance(rail, ContextProcessorRail)]
    _, cfg = context_rails[0]._user_processors[0]
    assert cfg.trim_size == 200


def test_snapshot_dedup_rail_rides_the_retention_knob() -> None:
    agent = create_cua_agent(_create_dummy_model(), language="en", cua_snapshot_keep_last_k=2)

    dedup_rails = [r for r in agent._pending_rails if isinstance(r, CuaSnapshotDedupRail)]
    assert len(dedup_rails) == 1
    # One below the window processor's retention, so a full tree always stays
    # reachable in context.
    assert dedup_rails[0]._max_consecutive == 1


def test_build_cua_agent_config_forwards_permissions_and_host_to_the_factory() -> None:
    host = object()
    spec = build_cua_agent_config(
        _create_dummy_model(),
        permissions={"enabled": True, "schema": "tiered_policy"},
        permission_host=host,
    )

    assert spec.factory_kwargs["permissions"] == {"enabled": True, "schema": "tiered_policy"}
    assert spec.factory_kwargs["permission_host"] is host


def test_factory_kwargs_omit_permission_keys_unless_provided() -> None:
    # Older configs must keep their exact factory_kwargs shape.
    spec = build_cua_agent_config(_create_dummy_model())

    assert "permissions" not in spec.factory_kwargs
    assert "permission_host" not in spec.factory_kwargs

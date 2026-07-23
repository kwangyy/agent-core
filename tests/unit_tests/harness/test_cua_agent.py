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
from openjiuwen.harness.schema.config import SubAgentConfig
from openjiuwen.harness.subagents.cua_agent import (
    CUA_AGENT_FACTORY_NAME,
    DEFAULT_CUA_AGENT_SYSTEM_PROMPT,
    build_cua_agent_config,
    create_cua_agent,
)
from openjiuwen.harness.tools.cua.config import (
    CUA_DRIVER_MCP_SERVER_ID,
    build_cua_driver_mcp_config,
)
from openjiuwen.harness.tools.cua.cua_capabilities import BROWSER_CUA_TOOL_NAMES
from openjiuwen.harness.tools.cua.rails import CuaRuntimeRail
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
    context window on unusable history."""
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
    assert cfg.keep_last_k == 1


def test_caller_context_processor_rail_suppresses_cua_injection() -> None:
    caller_rail = ContextProcessorRail(preset=False)
    agent = create_cua_agent(_create_dummy_model(), language="en", rails=[caller_rail])

    context_rails = [rail for rail in agent._pending_rails if isinstance(rail, ContextProcessorRail)]
    assert context_rails == [caller_rail]


def test_create_cua_agent_rejects_unknown_capabilities() -> None:
    with pytest.raises(ValueError, match="Unsupported cua capabilities: browser"):
        create_cua_agent(_create_dummy_model(), cua_capabilities=["browser"])


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

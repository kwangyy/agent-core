# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Tests for the cua runtime and delivery-mode rails."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from openjiuwen.core.common.exception.errors import BaseError
from openjiuwen.core.foundation.llm import ToolMessage
from openjiuwen.core.foundation.tool import McpServerConfig
from openjiuwen.core.single_agent.rail.base import ToolCallInputs
from openjiuwen.harness.tools.cua.rails import (
    CuaDeliveryModeRail,
    CuaElementAddressingRail,
    CuaRepeatFailureRail,
    CuaRuntimeRail,
    CuaScreenshotDownscaleRail,
    CuaSnapshotDedupRail,
    CuaSnapshotFreshnessRail,
)


def _mcp_cfg() -> McpServerConfig:
    return McpServerConfig(
        server_id="cua_driver_stdio",
        server_name="cua-driver",
        server_path="stdio://cua-driver",
        client_type="stdio",
        params={"command": "cua-driver", "args": ["mcp"]},
    )


def _ctx(session_id: str = "session-1") -> SimpleNamespace:
    session = MagicMock()
    session.get_session_id.return_value = session_id
    return SimpleNamespace(agent=MagicMock(), session=session)


def _probe_process(returncode: int = 0, stderr: bytes = b"") -> MagicMock:
    process = MagicMock()
    process.returncode = returncode
    process.communicate = AsyncMock(return_value=(b"{}", stderr))
    process.kill = MagicMock()
    return process


@pytest.mark.asyncio
async def test_before_invoke_fails_loud_when_daemon_probe_fails(monkeypatch) -> None:
    # A dead daemon must abort the run with remediation, not degrade silently.
    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        AsyncMock(return_value=_probe_process(returncode=1, stderr=b"connect error")),
    )
    rail = CuaRuntimeRail(_mcp_cfg())

    with pytest.raises(BaseError, match="autostart kick"):
        await rail.before_invoke(_ctx())


@pytest.mark.asyncio
async def test_before_invoke_fails_loud_when_binary_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        AsyncMock(side_effect=FileNotFoundError()),
    )
    rail = CuaRuntimeRail(_mcp_cfg())

    with pytest.raises(BaseError, match="binary not found"):
        await rail.before_invoke(_ctx())


@pytest.mark.asyncio
async def test_before_invoke_applies_allowlist_and_starts_session(monkeypatch) -> None:
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=_probe_process()))
    start_tool = MagicMock()
    start_tool.invoke = AsyncMock()
    resource_mgr = MagicMock()
    resource_mgr.get_tool.return_value = start_tool
    from openjiuwen.core.runner.runner import Runner

    monkeypatch.setattr(Runner, "resource_mgr", resource_mgr)

    cfg = _mcp_cfg()
    rail = CuaRuntimeRail(cfg, ("list_apps", "click"))
    ctx = _ctx(session_id="abc123")

    await rail.before_invoke(ctx)

    ability_manager = ctx.agent.ability_manager
    ability_manager.add.assert_called_once_with(cfg)
    ability_manager.set_mcp_tool_allowlist.assert_called_once_with(cfg, ("list_apps", "click"))
    resource_mgr.get_tool.assert_called_once_with("cua_driver_stdio.cua-driver.start_session")
    start_tool.invoke.assert_awaited_once_with({"session": "cua-abc123"})


@pytest.mark.asyncio
async def test_session_failures_never_block_the_run(monkeypatch) -> None:
    # Session identity is cosmetic; the daemon idle TTL reclaims leaks.
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=_probe_process()))
    failing_tool = MagicMock()
    failing_tool.invoke = AsyncMock(side_effect=RuntimeError("denied"))
    resource_mgr = MagicMock()
    resource_mgr.get_tool.return_value = failing_tool
    from openjiuwen.core.runner.runner import Runner

    monkeypatch.setattr(Runner, "resource_mgr", resource_mgr)

    rail = CuaRuntimeRail(_mcp_cfg())
    ctx = _ctx()

    await rail.before_invoke(ctx)  # start_session raises internally -> warning only


@pytest.mark.asyncio
async def test_lifecycle_setup_is_cached_after_the_first_invoke(monkeypatch) -> None:
    # The security gate's approval path re-enters before_invoke before every
    # gated call; in the recorded corpus that made session lifecycle 38.9% of
    # all driver calls (~7 start/end pairs per run, 57-59% provable no-ops).
    # Setup must therefore run once per run, not once per loop entry.
    probe = AsyncMock(return_value=_probe_process())
    monkeypatch.setattr(asyncio, "create_subprocess_exec", probe)
    tool = MagicMock()
    tool.invoke = AsyncMock()
    resource_mgr = MagicMock()
    resource_mgr.get_tool.return_value = tool
    from openjiuwen.core.runner.runner import Runner

    monkeypatch.setattr(Runner, "resource_mgr", resource_mgr)

    rail = CuaRuntimeRail(_mcp_cfg(), ("click",))
    ctx = _ctx(session_id="abc123")

    await rail.before_invoke(ctx)
    await rail.after_invoke(ctx)  # an approval round-trip crosses the invoke boundary
    await rail.before_invoke(ctx)

    assert probe.await_count == 1
    ctx.agent.ability_manager.add.assert_called_once()
    ctx.agent.ability_manager.set_mcp_tool_allowlist.assert_called_once()
    tool.invoke.assert_awaited_once_with({"session": "cua-abc123"})


@pytest.mark.asyncio
async def test_the_session_is_left_to_the_daemon_idle_ttl(monkeypatch) -> None:
    # Explicit end_session per invoke was half the lifecycle churn. The daemon
    # reclaims idle sessions itself and the rail already documented that as
    # the self-heal path, so the rail no longer ends sessions at all.
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=_probe_process()))
    tool = MagicMock()
    tool.invoke = AsyncMock()
    resource_mgr = MagicMock()
    resource_mgr.get_tool.return_value = tool
    from openjiuwen.core.runner.runner import Runner

    monkeypatch.setattr(Runner, "resource_mgr", resource_mgr)

    rail = CuaRuntimeRail(_mcp_cfg())
    ctx = _ctx(session_id="xyz789")

    await rail.before_invoke(ctx)
    await rail.after_invoke(ctx)

    tool.invoke.assert_awaited_once_with({"session": "cua-xyz789"})  # start only, never end
    requested = [str(c) for c in resource_mgr.get_tool.call_args_list]
    assert not any("end_session" in r for r in requested)


@pytest.mark.asyncio
async def test_a_failed_probe_is_retried_on_the_next_invoke(monkeypatch) -> None:
    # Only success is cached: a daemon that was down must be re-checked, or
    # the run would proceed against a runtime never seen alive.
    probe = AsyncMock(return_value=_probe_process(returncode=1, stderr=b"down"))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", probe)
    resource_mgr = MagicMock()
    resource_mgr.get_tool.return_value = None
    from openjiuwen.core.runner.runner import Runner

    monkeypatch.setattr(Runner, "resource_mgr", resource_mgr)
    rail = CuaRuntimeRail(_mcp_cfg())

    with pytest.raises(BaseError):
        await rail.before_invoke(_ctx())

    probe.return_value = _probe_process()
    await rail.before_invoke(_ctx())

    assert probe.await_count == 2


# ---------------------------------------------------------------------------
# CuaDeliveryModeRail
# ---------------------------------------------------------------------------


def _tool_ctx(tool_name: str, tool_args) -> SimpleNamespace:
    # A real ToolCallInputs: the rail narrows on it before touching tool_args,
    # so a duck-typed stand-in would pass the test while skipping the rail.
    inputs = ToolCallInputs(
        tool_call=SimpleNamespace(id="call-1"),
        tool_name=tool_name,
        tool_args=tool_args,
    )
    return SimpleNamespace(inputs=inputs, extra={})


@pytest.mark.asyncio
async def test_non_tool_call_inputs_are_ignored() -> None:
    rail = CuaDeliveryModeRail(_mcp_cfg(), "background")
    ctx = SimpleNamespace(inputs={"tool_name": "mcp_cua-driver_click"}, extra={})

    await rail.before_tool_call(ctx)  # must not raise on a non-ToolCallInputs payload

    assert ctx.extra == {}


@pytest.mark.asyncio
async def test_background_mode_injects_when_the_model_omits_delivery_mode() -> None:
    rail = CuaDeliveryModeRail(_mcp_cfg(), "background")
    ctx = _tool_ctx("mcp_cua-driver_click", {"pid": 1, "window_id": 2})

    await rail.before_tool_call(ctx)

    assert ctx.inputs.tool_args["delivery_mode"] == "background"
    assert not ctx.extra.get("_skip_tool")


@pytest.mark.asyncio
async def test_foreground_mode_must_inject_because_the_driver_default_is_background() -> None:
    # The whole reason injection exists: omitting delivery_mode falls back to
    # the driver's "background" default, which silently violates a foreground
    # -only policy. Rejection alone could never enforce this direction.
    rail = CuaDeliveryModeRail(_mcp_cfg(), "foreground")
    ctx = _tool_ctx("mcp_cua-driver_type_text", {"pid": 1, "text": "hi"})

    await rail.before_tool_call(ctx)

    assert ctx.inputs.tool_args["delivery_mode"] == "foreground"


@pytest.mark.asyncio
async def test_background_mode_rejects_explicit_foreground_rather_than_rewriting() -> None:
    # cua-driver's background_unavailable error tells the model to retry with
    # foreground. Rewriting that retry back to background would loop the model
    # against the driver's own advice with no signal, so it must be rejected.
    rail = CuaDeliveryModeRail(_mcp_cfg(), "background")
    ctx = _tool_ctx("mcp_cua-driver_click", {"pid": 1, "delivery_mode": "foreground"})

    await rail.before_tool_call(ctx)

    assert ctx.extra["_skip_tool"] is True
    assert "background-only" in ctx.inputs.tool_result["error"]
    assert ctx.inputs.tool_msg.tool_call_id == "call-1"
    # The rejected args are never handed to the driver.
    assert ctx.inputs.tool_args["delivery_mode"] == "foreground"


@pytest.mark.asyncio
async def test_foreground_mode_overrides_explicit_background() -> None:
    # Safe to override here: foreground delivery cannot fail the way background
    # can, so there is no retry loop to warn the model about.
    rail = CuaDeliveryModeRail(_mcp_cfg(), "foreground")
    ctx = _tool_ctx("mcp_cua-driver_scroll", {"pid": 1, "delivery_mode": "background"})

    await rail.before_tool_call(ctx)

    assert ctx.inputs.tool_args["delivery_mode"] == "foreground"
    assert not ctx.extra.get("_skip_tool")


@pytest.mark.asyncio
async def test_json_encoded_args_are_written_back_as_json() -> None:
    # tool_args arrives as the raw ToolCall.arguments string before the ability
    # manager parses it; writing a dict back would change the type downstream.
    rail = CuaDeliveryModeRail(_mcp_cfg(), "background")
    ctx = _tool_ctx("mcp_cua-driver_click", json.dumps({"pid": 1}))

    await rail.before_tool_call(ctx)

    assert isinstance(ctx.inputs.tool_args, str)
    assert json.loads(ctx.inputs.tool_args)["delivery_mode"] == "background"


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["set_value", "launch_app", "bring_to_front", "get_window_state"])
async def test_tools_without_a_delivery_mode_argument_are_left_alone(tool_name: str) -> None:
    # set_value ships in the same "input" capability bundle as click but takes
    # no delivery_mode; injecting one would send an unknown argument.
    rail = CuaDeliveryModeRail(_mcp_cfg(), "background")
    ctx = _tool_ctx(f"mcp_cua-driver_{tool_name}", {"pid": 1})

    await rail.before_tool_call(ctx)

    assert "delivery_mode" not in ctx.inputs.tool_args


@pytest.mark.asyncio
async def test_instance_isolated_server_names_are_matched() -> None:
    # cua_instance_key renames the server, which is what makes the built-in
    # os_control permission rules miss. Building names from the config avoids it.
    cfg = _mcp_cfg()
    cfg.server_name = "cua-driver-run7"
    rail = CuaDeliveryModeRail(cfg, "background")
    ctx = _tool_ctx("mcp_cua-driver-run7_click", {"pid": 1})

    await rail.before_tool_call(ctx)

    assert ctx.inputs.tool_args["delivery_mode"] == "background"


def test_invalid_delivery_mode_fails_at_construction() -> None:
    with pytest.raises(ValueError, match="background"):
        CuaDeliveryModeRail(_mcp_cfg(), "sideways")


# ---------------------------------------------------------------------------
# CuaDeliveryModeRail end-to-end through the real tool-dispatch pipeline
#
# The tests above hand-build the callback context, which proves the rail
# mutates it but silently trusts the framework to honor those mutations. These
# drive a real ReActAgent + AbilityManager loop with a scripted LLM, so a
# refactor that stops honoring _skip_tool or tool_args rewrites (e.g. copying
# args before rails run) fails here even though every unit test still passes.
# ---------------------------------------------------------------------------

# Shared with the registered tool's closure: Runner.resource_mgr is
# process-global, so the first registered instance may serve later tests.
_CLICK_INVOCATIONS: list[dict] = []


def _make_cua_agent():
    """ReActAgent with a recording fake mcp_cua-driver_click tool."""
    import os

    from openjiuwen.core.foundation.llm import ModelClientConfig, ModelRequestConfig
    from openjiuwen.core.foundation.tool import LocalFunction, ToolCard
    from openjiuwen.core.runner import Runner
    from openjiuwen.core.single_agent import AgentCard, ReActAgent, ReActAgentConfig

    os.environ.setdefault("LLM_SSL_VERIFY", "false")
    _CLICK_INVOCATIONS.clear()
    tool = LocalFunction(
        card=ToolCard(
            id="mcp_cua-driver_click",
            name="mcp_cua-driver_click",
            description="fake cua click",
            input_params={
                "type": "object",
                "properties": {
                    "pid": {"type": "integer"},
                    "delivery_mode": {"type": "string"},
                },
                "required": ["pid"],
            },
        ),
        func=lambda **kwargs: (_CLICK_INVOCATIONS.append(kwargs), "clicked")[1],
    )
    config = ReActAgentConfig(
        model_config_obj=ModelRequestConfig(model="test-model"),
        model_client_config=ModelClientConfig(
            client_provider="OpenAI", api_key="mock_key", api_base="mock_url", verify_ssl=False
        ),
        prompt_template=[dict(role="system", content="desktop test agent")],
    )
    agent = ReActAgent(card=AgentCard(description="cua rail test")).configure(config)
    agent.ability_manager.add(tool.card)
    if Runner.resource_mgr.get_tool(tool.card.id) is None:
        Runner.resource_mgr.add_tool(tool)
    return agent


async def _invoke_with_click(agent, arguments: str) -> object:
    from unittest.mock import patch as mock_patch

    from tests.unit_tests.fixtures.mock_llm import (
        MockLLMModel,
        create_text_response,
        create_tool_call_response,
    )

    mock_llm = MockLLMModel()
    mock_llm.set_responses(
        [
            create_tool_call_response("mcp_cua-driver_click", arguments),
            create_text_response("done"),
        ]
    )
    with mock_patch.object(agent, "_get_llm", return_value=mock_llm):
        await agent.invoke({"query": "click it"})
    return mock_llm


@pytest.mark.asyncio
async def test_pipeline_injects_delivery_mode_into_the_executed_tool() -> None:
    agent = _make_cua_agent()
    await agent.register_rail(CuaDeliveryModeRail(_mcp_cfg(), "background"))

    await _invoke_with_click(agent, '{"pid": 42}')

    # The tool itself received the injected mode — not just the rail context.
    assert _CLICK_INVOCATIONS == [{"pid": 42, "delivery_mode": "background"}]


@pytest.mark.asyncio
async def test_pipeline_blocks_foreground_and_returns_the_rejection_to_the_model() -> None:
    agent = _make_cua_agent()
    await agent.register_rail(CuaDeliveryModeRail(_mcp_cfg(), "background"))

    mock_llm = await _invoke_with_click(agent, '{"pid": 42, "delivery_mode": "foreground"}')

    # Enforcement: the driver tool never executed.
    assert _CLICK_INVOCATIONS == []
    # Teaching: the model's next turn sees the rejection, not a silent stall.
    followup_tool_msgs = [msg.content for msg in mock_llm.call_history[1] if getattr(msg, "role", None) == "tool"]
    assert any("background-only" in str(content) for content in followup_tool_msgs)


@pytest.mark.asyncio
async def test_pipeline_foreground_pin_overrides_explicit_background() -> None:
    agent = _make_cua_agent()
    await agent.register_rail(CuaDeliveryModeRail(_mcp_cfg(), "foreground"))

    await _invoke_with_click(agent, '{"pid": 42, "delivery_mode": "background"}')

    assert _CLICK_INVOCATIONS == [{"pid": 42, "delivery_mode": "foreground"}]


# ---------------------------------------------------------------------------
# CuaScreenshotDownscaleRail
# ---------------------------------------------------------------------------


def _noise_png_data_url(width: int = 600, height: int = 400) -> str:
    """Random-noise PNG: incompressible, so it reliably exceeds the budget."""
    import base64
    import io
    import os

    from PIL import Image

    img = Image.frombytes("RGB", (width, height), os.urandom(width * height * 3))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _screenshot_ctx(tool_name: str, tool_result) -> SimpleNamespace:
    from openjiuwen.core.foundation.llm import ToolMessage

    inputs = ToolCallInputs(
        tool_call=SimpleNamespace(id="call-1"),
        tool_name=tool_name,
        tool_args={},
    )
    inputs.tool_result = tool_result
    inputs.tool_msg = ToolMessage(content="tree summary", tool_call_id="call-1")
    return SimpleNamespace(inputs=inputs, extra={})


@pytest.mark.asyncio
async def test_oversized_screenshot_is_downscaled_in_place() -> None:
    pytest.importorskip("PIL")
    original = _noise_png_data_url()
    item = {"type": "image", "source": "mcp", "mime_type": "image/png", "data_url": original}
    tool_result = SimpleNamespace(data={"content": "tree", "multimodal": [item]})
    ctx = _screenshot_ctx("mcp_cua-driver_get_window_state", tool_result)

    rail = CuaScreenshotDownscaleRail(_mcp_cfg())
    await rail.after_tool_call(ctx)

    # In-place mutation: react_agent consumes the same result object the
    # ability manager returned, so a replaced object would silently be lost.
    assert ctx.inputs.tool_result is tool_result
    assert tool_result.data["multimodal"][0] is item
    assert item["data_url"].startswith("data:image/jpeg;base64,")
    assert len(item["data_url"]) < len(original)
    assert item["mime_type"] == "image/jpeg"
    # 600x400 fits within max_edge: re-encode without resize keeps the
    # coordinate space, so no scale note may appear (it would be false).
    assert "[screenshot scaled]" not in ctx.inputs.tool_msg.content


@pytest.mark.asyncio
async def test_resized_screenshot_appends_scale_note_to_tool_message() -> None:
    # A model reading pixels off a resized screenshot misclicks by the scale
    # factor unless told; the note must reach the ALREADY-BUILT ToolMessage,
    # not just data["content"] (live-confirmed failure mode: coordinates were
    # read off an 800px image of a ~1330px window).
    pytest.importorskip("PIL")
    original = _noise_png_data_url(2000, 1200)
    item = {"type": "image", "source": "mcp", "mime_type": "image/png", "data_url": original}
    tool_result = SimpleNamespace(data={"content": "tree summary", "multimodal": [item]})
    ctx = _screenshot_ctx("mcp_cua-driver_get_window_state", tool_result)

    rail = CuaScreenshotDownscaleRail(_mcp_cfg())
    await rail.after_tool_call(ctx)

    assert item["data_url"].startswith("data:image/jpeg;base64,")
    note = ctx.inputs.tool_msg.content
    assert "[screenshot scaled]" in note
    assert "2000x1200" in note
    assert "multiplied by 2.50" in note  # 2000 -> 800px rung for incompressible noise
    assert "element_index" in note
    assert "[screenshot scaled]" in tool_result.data["content"]


@pytest.mark.asyncio
async def test_small_screenshot_is_left_untouched() -> None:
    small = "data:image/png;base64,QQ=="
    item = {"type": "image", "mime_type": "image/png", "data_url": small}
    tool_result = SimpleNamespace(data={"content": "tree", "multimodal": [item]})
    ctx = _screenshot_ctx("mcp_cua-driver_get_window_state", tool_result)

    await CuaScreenshotDownscaleRail(_mcp_cfg()).after_tool_call(ctx)

    assert item["data_url"] == small
    assert item["mime_type"] == "image/png"


@pytest.mark.asyncio
async def test_non_cua_tools_are_ignored() -> None:
    pytest.importorskip("PIL")
    original = _noise_png_data_url()
    item = {"type": "image", "mime_type": "image/png", "data_url": original}
    tool_result = SimpleNamespace(data={"content": "x", "multimodal": [item]})
    ctx = _screenshot_ctx("read_file", tool_result)

    await CuaScreenshotDownscaleRail(_mcp_cfg()).after_tool_call(ctx)

    assert item["data_url"] == original


@pytest.mark.asyncio
async def test_instance_isolated_server_names_are_downscale_matched() -> None:
    pytest.importorskip("PIL")
    cfg = _mcp_cfg()
    cfg.server_name = "cua-driver-run7"
    original = _noise_png_data_url()
    item = {"type": "image", "mime_type": "image/png", "data_url": original}
    tool_result = SimpleNamespace(data={"content": "x", "multimodal": [item]})
    ctx = _screenshot_ctx("mcp_cua-driver-run7_get_window_state", tool_result)

    await CuaScreenshotDownscaleRail(cfg).after_tool_call(ctx)

    assert item["data_url"].startswith("data:image/jpeg;base64,")


@pytest.mark.asyncio
async def test_legacy_result_shape_is_a_noop() -> None:
    # Blind-mode results are {"result": str} dicts with no .data attribute.
    ctx = _screenshot_ctx("mcp_cua-driver_get_window_state", {"result": "snapshot text"})

    await CuaScreenshotDownscaleRail(_mcp_cfg()).after_tool_call(ctx)

    assert ctx.inputs.tool_result == {"result": "snapshot text"}


@pytest.mark.asyncio
async def test_malformed_data_url_is_left_untouched() -> None:
    # Oversized so the decode path actually runs; not valid base64 image data.
    malformed = "data:image/png;base64," + "?" * 300_000
    item = {"type": "image", "mime_type": "image/png", "data_url": malformed}
    tool_result = SimpleNamespace(data={"content": "x", "multimodal": [item]})
    ctx = _screenshot_ctx("mcp_cua-driver_get_window_state", tool_result)

    await CuaScreenshotDownscaleRail(_mcp_cfg()).after_tool_call(ctx)

    assert item["data_url"] == malformed


# ---------------------------------------------------------------------------
# CuaElementAddressingRail
# ---------------------------------------------------------------------------


def _result_ctx(tool_name: str, tool_args, content: str) -> SimpleNamespace:
    """A ToolCallInputs carrying a driver response, as after_tool_call sees it."""
    inputs = ToolCallInputs(
        tool_call=SimpleNamespace(id="call-1"),
        tool_name=tool_name,
        tool_args=tool_args,
    )
    inputs.tool_result = SimpleNamespace(data={"content": content})
    inputs.tool_msg = ToolMessage(content=content, tool_call_id="call-1")
    return SimpleNamespace(inputs=inputs, extra={})


@pytest.mark.asyncio
async def test_pixels_are_dropped_when_an_element_index_is_also_present() -> None:
    # The driver refuses a call carrying both addressing modes, and that
    # refusal costs a full model turn -- 49 of the failures in the recorded
    # corpus were exactly this, all on type_text. The either/or is resolvable
    # without asking the model again, so the rail resolves it.
    rail = CuaElementAddressingRail(_mcp_cfg())
    ctx = _tool_ctx(
        "mcp_cua-driver_type_text",
        {"pid": 1, "window_id": 2, "element_index": 7, "x": 100, "y": 200, "text": "hi"},
    )

    await rail.before_tool_call(ctx)

    assert "x" not in ctx.inputs.tool_args
    assert "y" not in ctx.inputs.tool_args
    # Element addressing survives: it is what the agent prompt prefers, and it
    # works on backgrounded windows without moving the user's cursor.
    assert ctx.inputs.tool_args["element_index"] == 7
    assert ctx.inputs.tool_args["text"] == "hi"
    assert not ctx.extra.get("_skip_tool")


@pytest.mark.asyncio
async def test_coordinate_only_calls_are_untouched() -> None:
    # Surfaces outside the element tree (canvases, ribbons) can only be
    # addressed by pixel, so stripping coordinates there would break the one
    # fallback the agent has.
    rail = CuaElementAddressingRail(_mcp_cfg())
    ctx = _tool_ctx("mcp_cua-driver_click", {"pid": 1, "x": 100, "y": 200})

    await rail.before_tool_call(ctx)

    assert ctx.inputs.tool_args == {"pid": 1, "x": 100, "y": 200}


@pytest.mark.asyncio
async def test_json_encoded_args_are_written_back_json_encoded() -> None:
    # tool_args arrives as a JSON string until the ability manager parses it;
    # writing a dict back in that case would hand the driver the wrong shape.
    rail = CuaElementAddressingRail(_mcp_cfg())
    ctx = _tool_ctx(
        "mcp_cua-driver_type_text",
        json.dumps({"element_index": 3, "x": 5, "y": 6, "text": "hi"}),
    )

    await rail.before_tool_call(ctx)

    assert isinstance(ctx.inputs.tool_args, str)
    assert json.loads(ctx.inputs.tool_args) == {"element_index": 3, "text": "hi"}


@pytest.mark.asyncio
async def test_non_cua_tools_are_left_alone() -> None:
    rail = CuaElementAddressingRail(_mcp_cfg())
    ctx = _tool_ctx("read_file", {"element_index": 1, "x": 2, "y": 3})

    await rail.before_tool_call(ctx)

    assert ctx.inputs.tool_args == {"element_index": 1, "x": 2, "y": 3}


# ---------------------------------------------------------------------------
# CuaRepeatFailureRail
# ---------------------------------------------------------------------------


_FAILURE = "Pass either element_index (ax) or x,y (px) to type_text, not both."
_SUCCESS = "\u2705 Sent tab via SendInput on pid 44452 (delivery_mode:foreground)."


async def _replay(rail, tool_name, args, content, times):
    """Drive the same call through the rail `times` times, returning contexts."""
    seen = []
    for _ in range(times):
        ctx = _result_ctx(tool_name, dict(args) if isinstance(args, dict) else args, content)
        await rail.before_tool_call(ctx)
        if not ctx.extra.get("_skip_tool"):
            await rail.after_tool_call(ctx)
        seen.append(ctx)
    return seen


@pytest.mark.asyncio
async def test_identical_failures_are_blocked_before_reaching_the_driver() -> None:
    # One recorded run burned 172 turns and 510s re-sending the same rejected
    # type_text 55 times and ended on the error it started with. Desktop errors
    # arrive as ordinary tool results, so no retry rail ever sees them.
    rail = CuaRepeatFailureRail(_mcp_cfg())
    args = {"pid": 1, "element_index": 7, "x": 1, "y": 2, "text": "hi"}

    ctxs = await _replay(rail, "mcp_cua-driver_type_text", args, _FAILURE, 7)

    # Blocking starts on the 7th attempt: 6 identical failures were recorded
    # before it, which is the block_after threshold.
    assert not any(c.extra.get("_skip_tool") for c in ctxs[:6])
    assert ctxs[6].extra["_skip_tool"] is True
    assert "Blocked" in ctxs[6].inputs.tool_result["error"]
    # The model is told what to do instead, not merely refused.
    assert "get_window_state" in ctxs[6].inputs.tool_result["error"]


@pytest.mark.asyncio
async def test_the_model_is_warned_before_it_is_blocked() -> None:
    # A warning the model can act on is worth more than a block it cannot, so
    # the advisory lands well before the hard stop.
    rail = CuaRepeatFailureRail(_mcp_cfg())
    args = {"pid": 1, "text": "hi"}

    ctxs = await _replay(rail, "mcp_cua-driver_type_text", args, _FAILURE, 3)

    assert not any(c.extra.get("_skip_tool") for c in ctxs)
    # Annotated in place on both the result data and the already-built message.
    assert "will not change the result" in ctxs[2].inputs.tool_msg.content
    assert "will not change the result" in ctxs[2].inputs.tool_result.data["content"]
    # ...and only the third one is annotated.
    assert "will not change the result" not in ctxs[0].inputs.tool_msg.content


@pytest.mark.asyncio
async def test_repeated_successes_are_never_blocked() -> None:
    # The most-repeated identical calls in the corpus are legitimate: one run
    # sent press_key tab 27 times walking a form. Repetition alone must never
    # trip the breaker, or real workflows break.
    rail = CuaRepeatFailureRail(_mcp_cfg())
    args = {"pid": 44452, "key": "tab"}

    ctxs = await _replay(rail, "mcp_cua-driver_press_key", args, _SUCCESS, 27)

    assert not any(c.extra.get("_skip_tool") for c in ctxs)
    assert all("will not change the result" not in c.inputs.tool_msg.content for c in ctxs)


@pytest.mark.asyncio
async def test_a_success_clears_the_history_for_that_call() -> None:
    # If the same call starts working, whatever the model was stuck on has
    # moved; a later unrelated failure must start from a clean count rather
    # than inheriting a block from earlier in the run.
    rail = CuaRepeatFailureRail(_mcp_cfg())
    args = {"pid": 1, "text": "hi"}

    await _replay(rail, "mcp_cua-driver_type_text", args, _FAILURE, 5)
    await _replay(rail, "mcp_cua-driver_type_text", args, _SUCCESS, 1)
    ctxs = await _replay(rail, "mcp_cua-driver_type_text", args, _FAILURE, 3)

    assert not any(c.extra.get("_skip_tool") for c in ctxs)


@pytest.mark.asyncio
async def test_differing_arguments_are_tracked_separately() -> None:
    # Changing the arguments is exactly the recovery the block asks for, so a
    # different call must not inherit the failed one's count.
    rail = CuaRepeatFailureRail(_mcp_cfg())

    await _replay(rail, "mcp_cua-driver_type_text", {"pid": 1, "text": "a"}, _FAILURE, 6)
    ctxs = await _replay(rail, "mcp_cua-driver_type_text", {"pid": 1, "text": "b"}, _FAILURE, 1)

    assert not ctxs[0].extra.get("_skip_tool")


@pytest.mark.asyncio
async def test_session_identity_is_not_part_of_the_repeat_key() -> None:
    # Concurrent agents run under distinct driver session keys; if session
    # were part of the key the breaker could silently never fire when the
    # identity shifts mid-run.
    rail = CuaRepeatFailureRail(_mcp_cfg())
    for i in range(7):
        ctx = _result_ctx(
            "mcp_cua-driver_type_text",
            {"pid": 1, "text": "hi", "session": f"cua-{i}"},
            _FAILURE,
        )
        await rail.before_tool_call(ctx)
        if not ctx.extra.get("_skip_tool"):
            await rail.after_tool_call(ctx)

    assert ctx.extra["_skip_tool"] is True


@pytest.mark.asyncio
async def test_counters_reset_between_invokes() -> None:
    # Counters are per-invoke: a new task must not start life already blocked
    # by the previous one's failures.
    rail = CuaRepeatFailureRail(_mcp_cfg())
    args = {"pid": 1, "text": "hi"}

    await _replay(rail, "mcp_cua-driver_type_text", args, _FAILURE, 6)
    await rail.before_invoke(_ctx())
    ctxs = await _replay(rail, "mcp_cua-driver_type_text", args, _FAILURE, 1)

    assert not ctxs[0].extra.get("_skip_tool")


@pytest.mark.asyncio
async def test_a_skipped_call_is_not_counted_as_a_driver_response() -> None:
    # Another rail's rejection (e.g. the background-only delivery gate) never
    # reached the driver, so counting it would blame the driver for a local
    # policy decision.
    rail = CuaRepeatFailureRail(_mcp_cfg())
    for _ in range(9):
        ctx = _result_ctx("mcp_cua-driver_click", {"pid": 1}, "rejected by policy")
        ctx.extra["_skip_tool"] = True
        await rail.after_tool_call(ctx)

    fresh = _result_ctx("mcp_cua-driver_click", {"pid": 1}, "rejected by policy")
    await rail.before_tool_call(fresh)

    assert not fresh.extra.get("_skip_tool")


def test_invalid_thresholds_fail_at_construction() -> None:
    with pytest.raises(ValueError):
        CuaRepeatFailureRail(_mcp_cfg(), advise_after=5, block_after=2)
    with pytest.raises(ValueError):
        CuaRepeatFailureRail(_mcp_cfg(), advise_after=0, block_after=3)


# ---------------------------------------------------------------------------
# CuaSnapshotFreshnessRail
# ---------------------------------------------------------------------------


_ELEMENT_FAILURE = "Element 7 did not respond to Invoke."


async def _observe(rail, tool_short, args, content):
    """Feed one completed cua driver call through the freshness rail."""
    ctx = _result_ctx("mcp_cua-driver_" + tool_short, dict(args), content)
    await rail.after_tool_call(ctx)
    return ctx


@pytest.mark.asyncio
async def test_failed_element_action_on_a_stale_window_gets_a_resnapshot_hint() -> None:
    # The corpus shows 70 element actions failing after the window had been
    # acted on since its snapshot; each waited for the repeat-failure advisory
    # (3 identical failures) before being told to re-snapshot. The hint on the
    # FIRST such failure names the likeliest cause immediately.
    rail = CuaSnapshotFreshnessRail(_mcp_cfg())
    win = {"pid": 1, "window_id": 2}

    await _observe(rail, "get_window_state", win, "\u2705 tree")
    first = await _observe(rail, "click", {**win, "element_index": 7}, _ELEMENT_FAILURE)
    second = await _observe(rail, "click", {**win, "element_index": 7}, _ELEMENT_FAILURE)

    # Right after a snapshot the index is legitimately fresh: a failure there
    # is NOT a staleness problem, so no hint.
    assert "snapshot taken BEFORE" not in first.inputs.tool_msg.content
    # The first click dirtied the window; the next failure carries the hint.
    assert "snapshot taken BEFORE" in second.inputs.tool_msg.content
    assert "get_window_state" in second.inputs.tool_msg.content


@pytest.mark.asyncio
async def test_successful_element_actions_are_never_annotated() -> None:
    # 357 corpus element actions on acted-on windows SUCCEEDED (Windows trees
    # are stable across consecutive clicks) -- which is exactly why this rail
    # must never block and must never nag on success.
    rail = CuaSnapshotFreshnessRail(_mcp_cfg())
    win = {"pid": 1, "window_id": 2}

    await _observe(rail, "get_window_state", win, "\u2705 tree")
    ctxs = [await _observe(rail, "click", {**win, "element_index": i}, "\u2705 clicked") for i in (7, 8, 9, 10)]

    assert all("snapshot taken BEFORE" not in c.inputs.tool_msg.content for c in ctxs)


@pytest.mark.asyncio
async def test_a_fresh_snapshot_clears_the_stale_state() -> None:
    rail = CuaSnapshotFreshnessRail(_mcp_cfg())
    win = {"pid": 1, "window_id": 2}

    await _observe(rail, "get_window_state", win, "\u2705 tree")
    await _observe(rail, "click", {**win, "element_index": 7}, _ELEMENT_FAILURE)
    await _observe(rail, "get_window_state", win, "\u2705 tree")
    after = await _observe(rail, "click", {**win, "element_index": 7}, _ELEMENT_FAILURE)

    assert "snapshot taken BEFORE" not in after.inputs.tool_msg.content


@pytest.mark.asyncio
async def test_an_unsnapshotted_window_counts_as_stale() -> None:
    # An element_index with no snapshot at all is the same mistake in a worse
    # form; the driver rejects it and the hint tells the model the way out.
    rail = CuaSnapshotFreshnessRail(_mcp_cfg())

    ctx = await _observe(rail, "click", {"pid": 1, "window_id": 9, "element_index": 3}, _ELEMENT_FAILURE)

    assert "snapshot taken BEFORE" in ctx.inputs.tool_msg.content


@pytest.mark.asyncio
async def test_read_only_observers_do_not_mark_the_window_stale() -> None:
    # zoom / list_windows / get_desktop_state neither mutate the window nor
    # replace the element cache, so they must not cost the model its fresh
    # snapshot.
    rail = CuaSnapshotFreshnessRail(_mcp_cfg())
    win = {"pid": 1, "window_id": 2}

    await _observe(rail, "get_window_state", win, "\u2705 tree")
    await _observe(rail, "zoom", win, "\u2705 zoomed")
    await _observe(rail, "list_windows", {}, "\u2705 windows")
    ctx = await _observe(rail, "click", {**win, "element_index": 7}, _ELEMENT_FAILURE)

    assert "snapshot taken BEFORE" not in ctx.inputs.tool_msg.content


@pytest.mark.asyncio
async def test_the_drivers_own_resnapshot_message_is_not_doubled() -> None:
    # The driver already rejects truly stale element_tokens with "call
    # get_window_state again to refresh"; appending the same advice twice
    # teaches the model to skim advisories.
    rail = CuaSnapshotFreshnessRail(_mcp_cfg())
    win = {"pid": 1, "window_id": 2}

    await _observe(rail, "click", {**win, "element_index": 1}, "boom")
    ctx = await _observe(
        rail,
        "click",
        {**win, "element_index": 7},
        "element_token is stale; call get_window_state again to refresh",
    )

    assert "snapshot taken BEFORE" not in ctx.inputs.tool_msg.content


@pytest.mark.asyncio
async def test_windows_are_tracked_independently() -> None:
    rail = CuaSnapshotFreshnessRail(_mcp_cfg())

    await _observe(rail, "get_window_state", {"pid": 1, "window_id": 2}, "\u2705 tree")
    await _observe(rail, "click", {"pid": 3, "window_id": 4, "element_index": 1}, "\u2705 ok")
    ctx = await _observe(rail, "click", {"pid": 1, "window_id": 2, "element_index": 7}, _ELEMENT_FAILURE)

    # Acting on window (3,4) does not invalidate (1,2)'s snapshot.
    assert "snapshot taken BEFORE" not in ctx.inputs.tool_msg.content


@pytest.mark.asyncio
async def test_freshness_resets_between_invokes() -> None:
    rail = CuaSnapshotFreshnessRail(_mcp_cfg())
    win = {"pid": 1, "window_id": 2}

    await _observe(rail, "get_window_state", win, "\u2705 tree")
    await rail.before_invoke(_ctx())
    ctx = await _observe(rail, "click", {**win, "element_index": 7}, _ELEMENT_FAILURE)

    # The previous invoke's snapshot is gone with its element cache.
    assert "snapshot taken BEFORE" in ctx.inputs.tool_msg.content


# ---------------------------------------------------------------------------
# CuaSnapshotDedupRail
# ---------------------------------------------------------------------------


_TREE_A = "\u2705 Window state\n[element_index 1] Button '7'\n[element_index 2] Button '8'"
_TREE_B = "\u2705 Window state\n[element_index 1] Button '7'\n[element_index 2] Button '9'"


async def _snap(rail, tool_short, args, content):
    """Feed one completed cua driver call through the dedup rail."""
    ctx = _result_ctx("mcp_cua-driver_" + tool_short, dict(args), content)
    await rail.after_tool_call(ctx)
    return ctx


@pytest.mark.asyncio
async def test_an_identical_repeat_snapshot_is_collapsed_to_an_unchanged_note() -> None:
    # A verify loop re-reads the same stable window; the second identical tree
    # buys no information, so only a success-marked one-liner should reach
    # context (the driver still refreshed its element cache, so the retained
    # first copy keeps its valid indices).
    rail = CuaSnapshotDedupRail(_mcp_cfg())
    win = {"pid": 1, "window_id": 2}

    first = await _snap(rail, "get_window_state", win, _TREE_A)
    second = await _snap(rail, "get_window_state", win, _TREE_A)

    assert first.inputs.tool_msg.content == _TREE_A
    note = second.inputs.tool_msg.content
    assert note.startswith("\u2705") and "Unchanged" in note
    assert "[element_index" not in note
    assert second.inputs.tool_result.data["content"] == note


@pytest.mark.asyncio
async def test_collapse_is_capped_below_the_context_retention_window() -> None:
    # The window processor keeps the newest keep_last_k snapshot results in
    # full; a run of notes longer than keep_last_k - 1 could evict every full
    # copy of the tree from context, so the cap lets one through periodically.
    rail = CuaSnapshotDedupRail(_mcp_cfg(), keep_last_k=3)
    win = {"pid": 1, "window_id": 2}

    ctxs = [await _snap(rail, "get_window_state", win, _TREE_A) for _ in range(5)]

    collapsed = ["Unchanged" in c.inputs.tool_msg.content for c in ctxs]
    assert collapsed == [False, True, True, False, True]


@pytest.mark.asyncio
async def test_keep_last_k_of_one_disables_collapsing() -> None:
    # With a single-slot window a note would BE the only retained snapshot,
    # leaving the model with no tree at all.
    rail = CuaSnapshotDedupRail(_mcp_cfg(), keep_last_k=1)
    win = {"pid": 1, "window_id": 2}

    ctxs = [await _snap(rail, "get_window_state", win, _TREE_A) for _ in range(3)]

    assert all(c.inputs.tool_msg.content == _TREE_A for c in ctxs)


@pytest.mark.asyncio
async def test_a_changed_tree_is_never_collapsed() -> None:
    rail = CuaSnapshotDedupRail(_mcp_cfg())
    win = {"pid": 1, "window_id": 2}

    await _snap(rail, "get_window_state", win, _TREE_A)
    await _snap(rail, "click", {**win, "element_index": 2}, "\u2705 clicked")
    changed = await _snap(rail, "get_window_state", win, _TREE_B)

    assert changed.inputs.tool_msg.content == _TREE_B


@pytest.mark.asyncio
async def test_a_failed_snapshot_clears_the_baseline() -> None:
    # The failure delivered no tree, so eliding the next success against the
    # pre-failure baseline would leave the model with nothing to act on.
    rail = CuaSnapshotDedupRail(_mcp_cfg())
    win = {"pid": 1, "window_id": 2}

    await _snap(rail, "get_window_state", win, _TREE_A)
    await _snap(rail, "get_window_state", win, "window not found")
    recovered = await _snap(rail, "get_window_state", win, _TREE_A)

    assert recovered.inputs.tool_msg.content == _TREE_A


@pytest.mark.asyncio
async def test_windows_are_deduped_independently() -> None:
    rail = CuaSnapshotDedupRail(_mcp_cfg())

    await _snap(rail, "get_window_state", {"pid": 1, "window_id": 2}, _TREE_A)
    other = await _snap(rail, "get_window_state", {"pid": 3, "window_id": 4}, _TREE_A)

    # Same bytes, different window: not a repeat of anything.
    assert other.inputs.tool_msg.content == _TREE_A


@pytest.mark.asyncio
async def test_collapse_survives_interleaved_actions_when_the_tree_is_stable() -> None:
    # Windows trees are stable across consecutive clicks (357 recorded
    # successes); if the re-read after an action comes back byte-identical,
    # it is still redundant.
    rail = CuaSnapshotDedupRail(_mcp_cfg())
    win = {"pid": 1, "window_id": 2}

    await _snap(rail, "get_window_state", win, _TREE_A)
    await _snap(rail, "click", {**win, "element_index": 1}, "\u2705 clicked")
    reread = await _snap(rail, "get_window_state", win, _TREE_A)

    assert "Unchanged" in reread.inputs.tool_msg.content


@pytest.mark.asyncio
async def test_a_window_that_changes_without_any_action_is_flagged_dynamic() -> None:
    # The action-based freshness rail cannot see an app that mutates its own
    # UI; the digest comparison can, and warns exactly once per window.
    rail = CuaSnapshotDedupRail(_mcp_cfg())
    win = {"pid": 1, "window_id": 2}

    await _snap(rail, "get_window_state", win, _TREE_A)
    changed = await _snap(rail, "get_window_state", win, _TREE_B)
    changed_again = await _snap(rail, "get_window_state", win, _TREE_A)

    note = changed.inputs.tool_msg.content
    assert note.startswith(_TREE_B)  # the tree itself is retained
    assert "changed since your previous snapshot" in note
    assert "changed since your previous snapshot" not in changed_again.inputs.tool_msg.content


@pytest.mark.asyncio
async def test_no_dynamic_flag_when_the_agent_acted_in_between() -> None:
    rail = CuaSnapshotDedupRail(_mcp_cfg())
    win = {"pid": 1, "window_id": 2}

    await _snap(rail, "get_window_state", win, _TREE_A)
    await _snap(rail, "click", {**win, "element_index": 1}, "\u2705 clicked")
    changed = await _snap(rail, "get_window_state", win, _TREE_B)

    # The agent's own click explains the change; no dynamic-UI warning.
    assert "changed since your previous snapshot" not in changed.inputs.tool_msg.content


@pytest.mark.asyncio
async def test_read_only_observers_do_not_mask_the_dynamic_flag() -> None:
    rail = CuaSnapshotDedupRail(_mcp_cfg())
    win = {"pid": 1, "window_id": 2}

    await _snap(rail, "get_window_state", win, _TREE_A)
    await _snap(rail, "zoom", win, "\u2705 zoomed")
    await _snap(rail, "list_windows", {}, "\u2705 windows")
    changed = await _snap(rail, "get_window_state", win, _TREE_B)

    # zoom / list_windows cannot have changed the window.
    assert "changed since your previous snapshot" in changed.inputs.tool_msg.content


@pytest.mark.asyncio
async def test_dedup_state_resets_between_invokes() -> None:
    rail = CuaSnapshotDedupRail(_mcp_cfg())
    win = {"pid": 1, "window_id": 2}

    await _snap(rail, "get_window_state", win, _TREE_A)
    await rail.before_invoke(_ctx())
    fresh = await _snap(rail, "get_window_state", win, _TREE_A)

    # A new conversation starts from an empty context: the previous invoke's
    # retained copy is gone, so nothing may be elided against it.
    assert fresh.inputs.tool_msg.content == _TREE_A


@pytest.mark.asyncio
async def test_volatile_snapshot_fields_do_not_defeat_the_dedup() -> None:
    # The driver stamps every snapshot with fresh screenshot bytes and a
    # monotonic snapshot_id even when the tree is untouched (verified by
    # diffing two consecutive live payloads) -- raw byte comparison would
    # therefore never fire at all.
    rail = CuaSnapshotDedupRail(_mcp_cfg())
    win = {"pid": 1, "window_id": 2}
    body = (
        '"element_token": "{sid}:0", "element_index": 0,\n'
        + '"snapshot_id": "{sid}",\n"screenshot_png_b64": "{png}",\n'
        + _TREE_A[2:]
    )

    await _snap(rail, "get_window_state", win, "\u2705 " + body.format(sid="s0012", png="AAAB"))
    second = await _snap(rail, "get_window_state", win, "\u2705 " + body.format(sid="s0013", png="AACD"))

    assert "Unchanged" in second.inputs.tool_msg.content


@pytest.mark.asyncio
async def test_a_real_tree_change_still_defeats_the_normalized_dedup() -> None:
    rail = CuaSnapshotDedupRail(_mcp_cfg())
    win = {"pid": 1, "window_id": 2}
    body = '"snapshot_id": "{sid}",\n' + "{tree}"

    await _snap(rail, "get_window_state", win, "\u2705 " + body.format(sid="s0012", tree=_TREE_A))
    changed = await _snap(rail, "get_window_state", win, "\u2705 " + body.format(sid="s0013", tree=_TREE_B))

    assert "Unchanged" not in changed.inputs.tool_msg.content


@pytest.mark.asyncio
async def test_the_screenshot_placeholder_length_does_not_defeat_the_dedup() -> None:
    # The MCP bridge renders the screenshot block as '[image content:
    # image/png, NNNNN base64 chars]' and the length changes every capture
    # (encoder noise), so it must not count as a tree change.
    rail = CuaSnapshotDedupRail(_mcp_cfg())
    win = {"pid": 1, "window_id": 2}
    body = _TREE_A + " [image content: image/png, {n} base64 chars]"

    await _snap(rail, "get_window_state", win, body.format(n=51439))
    second = await _snap(rail, "get_window_state", win, body.format(n=51163))

    assert "Unchanged" in second.inputs.tool_msg.content


@pytest.mark.asyncio
async def test_real_snapshot_payloads_carry_no_success_marker_and_still_collapse() -> None:
    # A real get_window_state text begins with a plain header line, NOT the
    # driver's action-success marker (that convention is for input actions).
    # Gating the baseline on the marker silently disabled the dedup in live
    # runs; the gate is tree content instead.
    rail = CuaSnapshotDedupRail(_mcp_cfg())
    win = {"pid": 8276, "window_id": 16192158}
    body = (
        "window_id=16192158 pid=8276 elements=32 [element_index 0] Document"
        + " "
        + "[image content: image/png, {n} base64 chars]"
    )

    await _snap(rail, "get_window_state", win, body.format(n=51439))
    second = await _snap(rail, "get_window_state", win, body.format(n=51163))

    assert "Unchanged" in second.inputs.tool_msg.content

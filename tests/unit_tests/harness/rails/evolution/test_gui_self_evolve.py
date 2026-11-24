# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Tests for GUI agent self-evolution wiring (D-20261123-0925-gui-self-evolve).

What these protect:

* A code verifier's verdict, not TTSE's LLM judge, decides whether a GUI run
  succeeded. TTSE detects on a detached snapshot with no ctx, so the score
  must reach the detector through the per-session queue, paired with its own run.
* A verifier failure never fails or alters the agent's run.
* Learned state stays with the agent instance: no default bank path, and
  never one inside the openjiuwen package.
* Attaching through rails= leaves the factories' own rails and permissions alone.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import openjiuwen
from openjiuwen.agent_evolving.trajectory.model import Trajectory
from openjiuwen.agent_evolving.trajectory.processor import TrajectorySpanProcessor
from openjiuwen.agent_evolving.trajectory.schema import SESSION_ID, TRAJECTORY_ID
from openjiuwen.agent_evolving.trajectory.spans import attributes_from_map
from openjiuwen.agent_evolving.ttse.induce_context import build_induce_evidence
from openjiuwen.core.common.exception.errors import BaseError
from openjiuwen.core.foundation.llm import Model, ModelClientConfig, ModelRequestConfig
from openjiuwen.core.single_agent.rail.base import AgentCallbackEvent
from openjiuwen.harness.rails.evolution import (
    GuiOutcomeRail,
    GuiSelfEvolveConfig,
    SignalBasedSuccessDetector,
    SuccessDetector,
    SuccessOutcome,
    TTSEConfig,
    TTSERail,
    VerifiedSuccessDetector,
    build_gui_self_evolve_rails,
)
from openjiuwen.harness.rails.evolution.evolution_rail import EvolutionRail

_PROCESSOR = TrajectorySpanProcessor()


def _model() -> Model:
    return Model(
        model_client_config=ModelClientConfig(
            client_provider="OpenAI", api_key="test-key", api_base="http://test-base", verify_ssl=False
        ),
        model_config=ModelRequestConfig(model="test-model"),
    )


def _trajectory(session_id: str, marker: str = "1") -> Trajectory:
    return Trajectory.from_otlp(
        {
            "resourceSpans": [
                {
                    "resource": {"attributes": attributes_from_map({TRAJECTORY_ID: f"exec-{marker}", SESSION_ID: session_id})},
                    "scopeSpans": [{"scope": {"name": "test"}, "spans": []}],
                }
            ]
        }
    )


def _invoke_ctx(session_id: str, result: object) -> SimpleNamespace:
    session = SimpleNamespace(get_session_id=lambda: session_id)
    return SimpleNamespace(session=session, inputs=SimpleNamespace(result=result, conversation_id=""))


class _RecordingDetector(SuccessDetector):
    """Fallback that records the snapshot it was asked to judge."""

    def __init__(self) -> None:
        self.snapshots: list[dict | None] = []

    async def detect(self, trajectory, messages, *, ctx=None, snapshot=None) -> SuccessOutcome:
        self.snapshots.append(snapshot)
        return SuccessOutcome(outcome="skip", score=0.0)


def _verifier(value):
    async def verify(_result):
        if isinstance(value, Exception):
            raise value
        return value

    return verify


# ---- verifier score reaches TTSE (V-02, V-03) ----


@pytest.mark.asyncio
async def test_verifier_score_reaches_detector_for_that_session_and_result_is_unchanged() -> None:
    """The bench's code verdict must be what TTSE judges, even though TTSE detects without ctx."""
    fallback = _RecordingDetector()
    detector = VerifiedSuccessDetector(fallback)
    rail = GuiOutcomeRail(_verifier(1.0), detector)
    result = {"output": "done"}
    ctx = _invoke_ctx("s-1", result)

    await rail.after_invoke(ctx)
    await detector.detect(_trajectory("s-1"), [], ctx=None, snapshot={"messages": []})

    assert fallback.snapshots[-1]["ttse_score"] == 1.0
    assert ctx.inputs.result is result and result == {"output": "done"}


@pytest.mark.asyncio
async def test_failing_verifier_leaves_run_alone_and_falls_back_to_ttse_detection() -> None:
    """A broken verifier must never fail the agent run, nor fabricate a score."""
    fallback = _RecordingDetector()
    detector = VerifiedSuccessDetector(fallback)
    rail = GuiOutcomeRail(_verifier(RuntimeError("verifier crashed")), detector)

    await rail.after_invoke(_invoke_ctx("s-1", {"output": "x"}))
    await detector.detect(_trajectory("s-1"), [], snapshot={"messages": []})

    assert "ttse_score" not in fallback.snapshots[-1]


@pytest.mark.asyncio
async def test_scores_pair_with_their_own_runs_in_order_and_per_session() -> None:
    """Background evolution judges runs later; a later run's score must not be credited to an earlier run."""
    fallback = _RecordingDetector()
    detector = VerifiedSuccessDetector(fallback)
    await GuiOutcomeRail(_verifier(1.0), detector).after_invoke(_invoke_ctx("a", {}))
    await GuiOutcomeRail(_verifier(None), detector).after_invoke(_invoke_ctx("a", {}))
    await GuiOutcomeRail(_verifier(0.0), detector).after_invoke(_invoke_ctx("a", {}))
    await GuiOutcomeRail(_verifier(0.5), detector).after_invoke(_invoke_ctx("b", {}))

    for _ in range(3):
        await detector.detect(_trajectory("a"), [], snapshot={})
    await detector.detect(_trajectory("b"), [], snapshot={})

    seen = [snap.get("ttse_score") for snap in fallback.snapshots]
    assert seen == [1.0, None, 0.0, 0.5]


@pytest.mark.asyncio
async def test_verifier_score_uses_ttse_own_success_threshold_without_calling_the_judge() -> None:
    """Outcome mapping stays TTSE's: a passing verdict is success and the LLM judge is never asked."""
    llm = MagicMock()
    detector = VerifiedSuccessDetector(SignalBasedSuccessDetector(llm=llm, model="m", config=TTSEConfig()))
    await GuiOutcomeRail(_verifier(1.0), detector).after_invoke(_invoke_ctx("s", {}))
    await GuiOutcomeRail(_verifier(0.0), detector).after_invoke(_invoke_ctx("s", {}))

    passed = await detector.detect(_trajectory("s", "1"), [], snapshot={"messages": []})
    failed = await detector.detect(_trajectory("s", "2"), [], snapshot={"messages": []})

    assert passed.outcome == "success"
    assert failed.outcome == "fail"
    llm.invoke.assert_not_called()


def test_outcome_rail_runs_before_the_evolution_rail_after_invoke() -> None:
    """Higher priority runs first: the score must be queued before TTSE prepares or starts evolution."""
    rail = GuiOutcomeRail(_verifier(1.0), VerifiedSuccessDetector(_RecordingDetector()))

    assert AgentCallbackEvent.AFTER_INVOKE in rail.get_callbacks()
    assert rail.callback_priority(AgentCallbackEvent.AFTER_INVOKE) > EvolutionRail.priority


# ---- builder and bank path (V-08, V-09) ----


def test_builder_requires_an_explicit_store_path() -> None:
    """No fallback to TTSE's cwd-relative default, which would land in the repo checkout."""
    with pytest.raises(BaseError, match="store_path is required"):
        build_gui_self_evolve_rails(GuiSelfEvolveConfig(llm=_model(), model="m", trajectory_span_processor=_PROCESSOR, store_path=" "))


def test_builder_refuses_a_bank_inside_the_openjiuwen_package() -> None:
    """A running agent must never write learned rules into agent-core's own package."""
    inside = Path(openjiuwen.__file__).resolve().parent / ".ttse" / "bank.json"

    with pytest.raises(BaseError, match="inside the openjiuwen package"):
        build_gui_self_evolve_rails(GuiSelfEvolveConfig(llm=_model(), model="m", trajectory_span_processor=_PROCESSOR, store_path=str(inside)))
    assert not inside.parent.exists()


def test_builder_wires_learn_and_inject_flags_to_the_instance_bank(tmp_path) -> None:
    """Production learns and injects, both against this instance's own bank only."""
    caller_cfg = TTSEConfig()
    bank = tmp_path / "instance" / "bank.json"
    rails = build_gui_self_evolve_rails(
        GuiSelfEvolveConfig(
            llm=_model(),
            model="m",
            trajectory_span_processor=_PROCESSOR,
            store_path=str(bank),
            inject_enabled=True,
            evolve_enabled=True,
            verifier=_verifier(1.0),
            ttse_config=caller_cfg,
        )
    )

    outcome, ttse = rails
    assert isinstance(outcome, GuiOutcomeRail) and isinstance(ttse, TTSERail)
    cfg = ttse._ttse_config
    assert (cfg.store_path, cfg.inject_enabled, cfg.evolve_enabled) == (str(bank.resolve()), True, True)
    assert isinstance(ttse._success_detector, VerifiedSuccessDetector)
    assert ttse._success_detector is outcome._detector
    # The caller's config object is not mutated.
    assert (caller_cfg.store_path, caller_cfg.inject_enabled) == (TTSEConfig().store_path, TTSEConfig().inject_enabled)


def test_builder_defaults_to_learn_only_and_skips_the_bridge_without_a_verifier(tmp_path) -> None:
    """Learn-only is the default rollout stage; with no verifier TTSE's own detector decides."""
    rails = build_gui_self_evolve_rails(GuiSelfEvolveConfig(llm=_model(), model="m", trajectory_span_processor=_PROCESSOR, store_path=str(tmp_path / "b.json")))

    [ttse] = rails
    assert isinstance(ttse, TTSERail)
    assert ttse._ttse_config.inject_enabled is False
    assert isinstance(ttse._success_detector, SignalBasedSuccessDetector)


# ---- attaching to the GUI factories (V-01, V-05) ----


def test_cua_agent_gains_only_the_self_evolve_rails_and_keeps_its_permissions(tmp_path) -> None:
    """Attaching via rails= adds exactly the two rails; the cua rails and permission rules are untouched."""
    from openjiuwen.harness.subagents.cua_agent import create_cua_agent

    plain = create_cua_agent(_model(), language="en")
    extra = build_gui_self_evolve_rails(
        GuiSelfEvolveConfig(llm=_model(), model="m", trajectory_span_processor=_PROCESSOR, store_path=str(tmp_path / "b.json"), verifier=_verifier(1.0))
    )
    evolved = create_cua_agent(_model(), language="en", rails=extra)

    plain_types = [type(r) for r in plain._pending_rails]
    evolved_types = [type(r) for r in evolved._pending_rails]
    assert evolved_types == [GuiOutcomeRail, TTSERail] + plain_types
    assert evolved.deep_config.permissions == plain.deep_config.permissions
    assert [c.name for c in evolved.deep_config.tools or []] == [c.name for c in plain.deep_config.tools or []]


def test_browser_agent_receives_the_self_evolve_rails_through_rails(tmp_path) -> None:
    """The same builder serves the browser agent; its factory forwards caller rails unchanged."""
    from openjiuwen.harness.subagents import browser_agent

    captured: dict = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    extra = build_gui_self_evolve_rails(GuiSelfEvolveConfig(llm=_model(), model="m", trajectory_span_processor=_PROCESSOR, store_path=str(tmp_path / "b.json")))
    from openjiuwen.core.foundation.tool import McpServerConfig
    from openjiuwen.harness.tools.browser_move.playwright_runtime.config import BrowserRunGuardrails, RuntimeSettings

    settings = RuntimeSettings(
        provider="openai",
        api_key="test-key",
        api_base="https://example.invalid/v1",
        model_name="test-model",
        mcp_cfg=McpServerConfig(
            server_id="test", server_name="test", server_path="stdio://playwright", client_type="stdio", params={}
        ),
        guardrails=BrowserRunGuardrails(max_steps=3, max_failures=1, timeout_s=30, retry_once=False),
    )
    with ExitStack() as stack:
        stack.enter_context(patch.object(browser_agent, "BrowserAgentRuntime"))
        stack.enter_context(patch.object(browser_agent, "build_browser_runtime_tools", return_value=[]))
        stack.enter_context(patch.object(browser_agent, "create_deep_agent", side_effect=fake_create))
        browser_agent.create_browser_agent(MagicMock(spec=Model), settings=settings, rails=extra)

    assert captured["rails"][: len(extra)] == extra


# ---- screenshots stay out of induction (V-06) ----


def test_screenshots_reach_the_induction_prompt_only_as_a_bounded_fragment() -> None:
    """GUI trajectories carry screenshots; a whole image in the induce prompt would overflow it.

    TTSE truncates each tool result, so only a bounded fragment of the data URL
    gets through (a known gap: it is not stripped). This pins the bound so a
    TTSE change that starts forwarding whole tool payloads fails here.
    """
    import re

    image = "data:image/png;base64," + "A" * 50_000
    messages = [
        {"role": "user", "content": "Open Notepad and type hello"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "screenshot", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": f"captured {image}"},
        {"role": "tool", "tool_call_id": "c1", "content": [{"type": "image_url", "image_url": {"url": image}}]},
        {"role": "assistant", "content": "Done."},
    ]

    text = build_induce_evidence(messages, task_query="Open Notepad").evidence_text

    assert "captured" in text, "tool output must still reach induction, or this test proves nothing"
    longest_base64_run = max((len(run) for run in re.findall(r"A+", text)), default=0)
    assert longest_base64_run < 1_000
    assert len(text) < 5_000

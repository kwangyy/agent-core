# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Self-evolution wiring for the GUI subagents (cua, browser, mobile).

``build_gui_self_evolve_rails`` returns rails a caller passes through a GUI
factory's existing ``rails=`` parameter. The factories themselves are not
changed, so an agent built without these rails behaves exactly as before.

TTSE learns FACT/TIP rules from each run and injects them into later prompts.
A GUI run rarely has a ground truth the model can see, so this module lets
the caller supply a code ``verifier`` whose score decides success instead of
TTSE's LLM judge.

TTSE runs its success detection on a detached snapshot, after the invoke and
possibly in the background, with no callback context. A score therefore cannot
travel on ``ctx``: ``GuiOutcomeRail`` hands it to ``VerifiedSuccessDetector``,
queued per session, and the detector pops it when TTSE judges that session's
trajectory. Within one session invokes run in order, and so do their
evolutions, so a FIFO queue pairs each score with its own run.
"""

from __future__ import annotations

from collections import deque
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

import openjiuwen
from openjiuwen.agent_evolving.trajectory.processor import TrajectorySpanProcessor
from openjiuwen.agent_evolving.ttse import SignalBasedSuccessDetector, SuccessDetector, SuccessOutcome, TTSEConfig
from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import raise_error
from openjiuwen.core.common.logging import logger
from openjiuwen.core.foundation.llm.model import Model
from openjiuwen.core.single_agent.rail.base import AgentCallbackContext, AgentCallbackEvent, AgentRail
from openjiuwen.harness.rails.evolution.evolution_rail import EvolutionRail
from openjiuwen.harness.rails.evolution.ttse_rail import TTSERail

GuiVerifier = Callable[[Any], Awaitable[float | None]]
"""Scores one finished run from its invoke result: 1.0 success, 0.0 fail, None unknown."""


class VerifiedSuccessDetector(SuccessDetector):
    """Prefer a queued verifier score for the trajectory's session, else delegate.

    The score is passed to the fallback as ``ttse_score``, which every TTSE
    detector already treats as an explicit grader score, so the success
    threshold and outcome mapping stay TTSE's own.
    """

    def __init__(self, fallback: SuccessDetector) -> None:
        self._fallback = fallback
        self._pending: dict[str, deque[float | None]] = {}

    def push(self, session_id: str, score: float | None) -> None:
        """Queue the verifier result of the session's latest invoke (None = no verdict)."""
        self._pending.setdefault(session_id, deque()).append(score)

    def _pop(self, session_id: str | None) -> float | None:
        queue = self._pending.get(session_id or "")
        if not queue:
            return None
        score = queue.popleft()
        if not queue:
            self._pending.pop(session_id or "", None)
        return score

    def update_llm(self, llm: Model, model: str) -> None:
        update = getattr(self._fallback, "update_llm", None)
        if callable(update):
            update(llm, model)

    async def detect(
        self,
        trajectory: Any,
        messages: Any,
        *,
        ctx: Any = None,
        snapshot: dict | None = None,
    ) -> SuccessOutcome:
        score = self._pop(getattr(trajectory, "session_id", None))
        if score is not None:
            snapshot = {**(snapshot or {}), "ttse_score": score}
        return await self._fallback.detect(trajectory, messages, ctx=ctx, snapshot=snapshot)


def _session_key(ctx: AgentCallbackContext) -> str:
    # Same resolution as EvolutionRail uses to stamp the trajectory's session_id.
    session = getattr(ctx, "session", None)
    if session is not None and hasattr(session, "get_session_id"):
        return str(session.get_session_id())
    return str(getattr(getattr(ctx, "inputs", None), "conversation_id", "") or "")


class GuiOutcomeRail(AgentRail):
    """Run the caller's verifier after each invoke and queue its score for TTSE.

    It runs ahead of the evolution rail's ``after_invoke`` so the score is
    queued before TTSE prepares, or starts, that run's evolution. It never
    changes the result returned to the caller.
    """

    def __init__(self, verifier: GuiVerifier, detector: VerifiedSuccessDetector) -> None:
        super().__init__()
        self._verifier = verifier
        self._detector = detector

    def callback_priority(self, event: AgentCallbackEvent) -> int:
        if event == AgentCallbackEvent.AFTER_INVOKE:
            return EvolutionRail.priority + 1
        return self.priority

    async def after_invoke(self, ctx: AgentCallbackContext) -> None:
        result = getattr(getattr(ctx, "inputs", None), "result", None)
        score: float | None = None
        try:
            raw = await self._verifier(result)
            score = None if raw is None else float(raw)
        except Exception as exc:  # noqa: BLE001 - a verifier failure must not fail the run
            logger.warning("[GuiOutcomeRail] verifier failed, falling back to TTSE detection: %s", exc)
        # Push even when there is no verdict, so later scores stay paired with their own runs.
        self._detector.push(_session_key(ctx), score)


@dataclass(frozen=True)
class GuiSelfEvolveConfig:
    """Settings for one GUI agent instance's self-evolution.

    ``store_path`` is required: learned rules belong to this agent instance
    alone and must never fall back to a shared default location.
    ``trajectory_span_processor`` is the process-wide processor the agent's
    spans are captured through (``get_trajectory_span_processor()``). Spans
    only exist when observability is acquired, the agent carries an
    ``AgentObservabilityRail`` and each run sits inside a run root span
    (``open_agent_run_span``); without them nothing is learned.
    """

    llm: Model
    model: str
    store_path: str
    trajectory_span_processor: TrajectorySpanProcessor
    inject_enabled: bool = False
    evolve_enabled: bool = True
    verifier: GuiVerifier | None = None
    ttse_config: TTSEConfig | None = None


def _checked_store_path(store_path: str) -> str:
    if not str(store_path or "").strip():
        raise_error(
            StatusCode.DEEPAGENT_CONFIG_PARAM_ERROR,
            error_msg="GuiSelfEvolveConfig.store_path is required: give this agent instance its own bank file",
        )
    resolved = Path(store_path).expanduser().resolve()
    package_dir = Path(openjiuwen.__file__).resolve().parent
    if resolved == package_dir or package_dir in resolved.parents:
        raise_error(
            StatusCode.DEEPAGENT_CONFIG_PARAM_ERROR,
            error_msg=(
                f"GuiSelfEvolveConfig.store_path {resolved} is inside the openjiuwen package ({package_dir}); "
                "store learned rules with the agent instance instead"
            ),
        )
    return str(resolved)


def build_gui_self_evolve_rails(config: GuiSelfEvolveConfig) -> list[AgentRail]:
    """Build the rails that make a GUI agent learn from, and optionally inject, its runs.

    Pass the result through the factory's ``rails=``, e.g.
    ``create_cua_agent(model, rails=build_gui_self_evolve_rails(cfg))``.
    """
    # Copied so a caller's TTSEConfig is never mutated, matching TTSERail itself.
    ttse_config = TTSEConfig() if config.ttse_config is None else copy(config.ttse_config)
    ttse_config.store_path = _checked_store_path(config.store_path)
    ttse_config.inject_enabled = config.inject_enabled
    ttse_config.evolve_enabled = config.evolve_enabled

    def ttse_rail(detector: SuccessDetector) -> TTSERail:
        return TTSERail(
            llm=config.llm,
            model=config.model,
            ttse_config=ttse_config,
            success_detector=detector,
            trajectory_span_processor=config.trajectory_span_processor,
        )

    fallback = SignalBasedSuccessDetector(llm=config.llm, model=config.model, config=ttse_config)
    if config.verifier is None:
        return [ttse_rail(fallback)]
    detector = VerifiedSuccessDetector(fallback)
    return [GuiOutcomeRail(config.verifier, detector), ttse_rail(detector)]

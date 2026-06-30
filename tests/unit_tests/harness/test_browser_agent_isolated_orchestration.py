# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Unit tests for the browser subagent's plan/execute task separation.

Two concerns:

* ``build_browser_agent_config`` must thread ``enable_isolated_orchestration``
  into ``factory_kwargs`` so the factory materializes a ``BrowserAgent``.
* ``run_isolated_subtasks`` must run each planned task in its OWN session
  (bounded context) and thread prior results forward as text — the core of the
  separation, distinct from a shared-context task loop.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List
from unittest import IsolatedAsyncioTestCase

from openjiuwen.core.foundation.llm import (
    Model,
    ModelClientConfig,
    ModelRequestConfig,
)
from openjiuwen.harness.schema.task import TaskPlan, TodoItem
from openjiuwen.harness.subagents.browser_agent import build_browser_agent_config
from openjiuwen.harness.tools.browser_move.browser_orchestration import (
    run_isolated_subtasks,
)


def _make_model() -> Model:
    return Model(
        model_client_config=ModelClientConfig(
            client_provider="OpenAI",
            api_key="test-key",
            api_base="https://example.invalid/v1",
            verify_ssl=False,
        ),
        model_config=ModelRequestConfig(model_name="mock-model"),
    )


class _FakeExecutor:
    """Records each isolated subtask invocation (query + session it ran in)."""

    def __init__(self, results: List[str]) -> None:
        self._results = list(results)
        self.calls: List[Dict[str, Any]] = []
        self.card = SimpleNamespace(id="browser_agent")

    async def invoke(self, query: Any, session: Any) -> Dict[str, Any]:
        idx = len(self.calls)
        self.calls.append({"query": query, "session_id": session.get_session_id()})
        return {"result_type": "answer", "output": self._results[idx]}


class _InterruptExecutor:
    def __init__(self) -> None:
        self.calls: List[Any] = []
        self.card = SimpleNamespace(id="browser_agent")

    async def invoke(self, query: Any, session: Any) -> Dict[str, Any]:
        self.calls.append(query)
        return {"result_type": "interrupt", "output": "awaiting user"}


def _two_task_plan() -> TaskPlan:
    return TaskPlan(
        goal="inspect two sites",
        tasks=[
            TodoItem(id="t0", content="open site A and read the title"),
            TodoItem(id="t1", content="open site B and read the title"),
        ],
    )


class TestBuildBrowserAgentConfigFlag(IsolatedAsyncioTestCase):
    async def test_flag_threaded_into_factory_kwargs_when_enabled(self):
        cfg = build_browser_agent_config(_make_model(), enable_isolated_orchestration=True)
        self.assertIs(cfg.factory_kwargs["enable_isolated_orchestration"], True)

    async def test_flag_defaults_off(self):
        cfg = build_browser_agent_config(_make_model())
        self.assertIs(cfg.factory_kwargs["enable_isolated_orchestration"], False)


class TestRunIsolatedSubtasksSeparation(IsolatedAsyncioTestCase):
    def _collect(self, events: List[Dict[str, Any]], etype: str) -> List[Dict[str, Any]]:
        return [e["data"] for e in events if e["type"] == etype]

    async def test_each_task_runs_in_its_own_session(self):
        executor = _FakeExecutor(["title A", "title B"])
        created: List[str] = []

        def session_factory(sid: str):
            created.append(sid)
            return SimpleNamespace(get_session_id=lambda: sid)

        events = [
            e
            async for e in run_isolated_subtasks(
                executor,
                _two_task_plan(),
                session_id="SESS",
                session_factory=session_factory,
            )
        ]

        # Two tasks → two invokes, each in a DISTINCT isolated session. This is
        # the separation: per-task context never shares a session.
        self.assertEqual(len(executor.calls), 2)
        self.assertEqual(created, ["SESS__subtask__0", "SESS__subtask__1"])
        self.assertEqual(
            [c["session_id"] for c in executor.calls],
            ["SESS__subtask__0", "SESS__subtask__1"],
        )
        self.assertEqual(len({c["session_id"] for c in executor.calls}), 2)

    async def test_event_sequence_and_final_reply(self):
        executor = _FakeExecutor(["title A", "title B"])

        def session_factory(sid: str):
            return SimpleNamespace(get_session_id=lambda: sid)

        events = [
            e
            async for e in run_isolated_subtasks(
                executor, _two_task_plan(), session_id="S", session_factory=session_factory
            )
        ]

        types = [e["type"] for e in events]
        # First (non-last) task emits task_complete; the last emits assistant_final.
        self.assertEqual(types.count("task_started"), 2)
        self.assertEqual(types.count("task_complete"), 1)
        self.assertEqual(types.count("assistant_final"), 1)
        self.assertEqual(self._collect(events, "task_complete")[0]["reply"], "title A")
        self.assertEqual(self._collect(events, "assistant_final")[0]["reply"], "title B")

    async def test_prior_result_threaded_into_next_task_query(self):
        # WHY: isolated sessions don't share history, so the earlier task's
        # output must be threaded into the next task's opening query as text —
        # otherwise the executor loses what it already learned.
        executor = _FakeExecutor(["SITE-A-TITLE", "title B"])

        def session_factory(sid: str):
            return SimpleNamespace(get_session_id=lambda: sid)

        _ = [
            e
            async for e in run_isolated_subtasks(
                executor, _two_task_plan(), session_id="S", session_factory=session_factory
            )
        ]

        second_query = executor.calls[1]["query"]
        self.assertIn("SITE-A-TITLE", str(second_query))

    async def test_interrupt_stops_after_current_task(self):
        executor = _InterruptExecutor()

        def session_factory(sid: str):
            return SimpleNamespace(get_session_id=lambda: sid)

        events = [
            e
            async for e in run_isolated_subtasks(
                executor, _two_task_plan(), session_id="S", session_factory=session_factory
            )
        ]

        # First task interrupts → second task never runs.
        self.assertEqual(len(executor.calls), 1)
        types = [e["type"] for e in events]
        self.assertIn("interrupt", types)
        self.assertNotIn("assistant_final", types)

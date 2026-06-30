# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Unit tests for TaskTool routing of browser-style subagents.

Covers the plan/execute split wiring: when a delegated subagent exposes
``run_isolated_tasks`` (a ``BrowserAgent``), ``TaskTool`` drives that
orchestration stream and aggregates its events into the tool output; otherwise
it falls back to a single plain ``invoke``. The split must NOT be reached when
the planner yields no tasks, so partial/empty plans degrade gracefully.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock

from openjiuwen.core.foundation.tool import ToolCard
from openjiuwen.core.session.agent import Session
from openjiuwen.harness.tools.subagent.task_tool import TaskTool


# ── event builders mirroring browser_orchestration's yielded contract ────────
def _plan(num_tasks: int) -> Dict[str, Any]:
    return {
        "type": "plan_created",
        "data": {"plan": {"tasks": [{"id": f"t{i}"} for i in range(num_tasks)]}},
    }


def _task_complete(reply: str) -> Dict[str, Any]:
    return {"type": "task_complete", "data": {"reply": reply}}


def _assistant_final(reply: str) -> Dict[str, Any]:
    return {"type": "assistant_final", "data": {"reply": reply}}


def _interrupt(output: Any = None) -> Dict[str, Any]:
    result = {} if output is None else {"output": output}
    return {"type": "interrupt", "data": {"result": result}}


class _FakeBrowserSubagent:
    """Stands in for a BrowserAgent: exposes run_isolated_tasks + invoke."""

    def __init__(self, events: List[Dict[str, Any]], invoke_output: str = "invoke-fallback") -> None:
        self._events = events
        self.last_query: str | None = None
        self.card = SimpleNamespace(id="browser_agent")
        self.invoke = AsyncMock(return_value={"output": invoke_output})

    async def run_isolated_tasks(self, query: str):
        self.last_query = query
        for event in self._events:
            yield event


class _FakePlainSubagent:
    """A non-browser subagent: no run_isolated_tasks, only invoke."""

    def __init__(self, output: str = "plain-result") -> None:
        self.card = SimpleNamespace(id="general_agent")
        self.invoke = AsyncMock(return_value={"output": output})


def _make_task_tool(parent_agent: Any) -> TaskTool:
    card = ToolCard(
        name="task_tool",
        description="delegate to a subagent",
        input_params={"type": "object", "properties": {}},
    )
    return TaskTool(card=card, parent_agent=parent_agent, language="en")


class TestRunIsolatedTasksAggregation(IsolatedAsyncioTestCase):
    """The static aggregator that consumes the orchestration event stream."""

    async def test_returns_assistant_final_reply(self):
        sub = _FakeBrowserSubagent(
            [_plan(2), _task_complete("first-done"), _assistant_final("FINAL")]
        )
        out = await TaskTool._run_isolated_tasks(sub, "do the thing")
        self.assertEqual(out, "FINAL")
        # The planner query is the task forwarded to run_isolated_tasks.
        self.assertEqual(sub.last_query, "do the thing")

    async def test_empty_plan_returns_none_for_fallback(self):
        # WHY: no tasks means the planner produced nothing actionable; the
        # caller must fall back to a plain invoke rather than return "".
        sub = _FakeBrowserSubagent([_plan(0)])
        out = await TaskTool._run_isolated_tasks(sub, "q")
        self.assertIsNone(out)

    async def test_no_assistant_final_joins_task_replies(self):
        # Stream ends without assistant_final (e.g. truncated): aggregate the
        # intermediate task replies instead of losing them.
        sub = _FakeBrowserSubagent([_plan(2), _task_complete("A"), _task_complete("B")])
        out = await TaskTool._run_isolated_tasks(sub, "q")
        self.assertEqual(out, "A\nB")

    async def test_interrupt_surfaces_result_output(self):
        sub = _FakeBrowserSubagent([_plan(2), _task_complete("A"), _interrupt(output="needs-input")])
        out = await TaskTool._run_isolated_tasks(sub, "q")
        self.assertEqual(out, "needs-input")

    async def test_interrupt_without_output_falls_back_to_task_replies(self):
        sub = _FakeBrowserSubagent([_plan(2), _task_complete("A"), _interrupt()])
        out = await TaskTool._run_isolated_tasks(sub, "q")
        self.assertEqual(out, "A")


class TestTaskToolRouting(IsolatedAsyncioTestCase):
    """End-to-end TaskTool.invoke routing between the two execution paths."""

    def _invoke_kwargs(self) -> Dict[str, Any]:
        return {"session": Session(session_id="parent_sess")}

    async def test_browser_subagent_routed_through_isolated_orchestration(self):
        sub = _FakeBrowserSubagent([_plan(1), _assistant_final("ORCH-RESULT")])
        parent = Mock()
        parent.create_subagent = Mock(return_value=sub)
        tool = _make_task_tool(parent)

        out = await tool.invoke(
            {"subagent_type": "browser_agent", "task_description": "open site"},
            **self._invoke_kwargs(),
        )

        self.assertTrue(out.success)
        self.assertEqual(out.data["output"], "ORCH-RESULT")
        self.assertEqual(out.data["agent_id"], "browser_agent")
        # The orchestration path must NOT also do a plain single invoke.
        sub.invoke.assert_not_awaited()

    async def test_plain_subagent_uses_single_invoke(self):
        sub = _FakePlainSubagent(output="PLAIN")
        parent = Mock()
        parent.create_subagent = Mock(return_value=sub)
        tool = _make_task_tool(parent)

        out = await tool.invoke(
            {"subagent_type": "general_agent", "task_description": "summarize"},
            **self._invoke_kwargs(),
        )

        self.assertTrue(out.success)
        self.assertEqual(out.data["output"], "PLAIN")
        sub.invoke.assert_awaited_once()

    async def test_browser_subagent_empty_plan_falls_back_to_invoke(self):
        # Orchestration yields no tasks → _run_isolated_tasks returns None →
        # TaskTool falls back to the plain invoke surface.
        sub = _FakeBrowserSubagent([_plan(0)], invoke_output="FALLBACK")
        parent = Mock()
        parent.create_subagent = Mock(return_value=sub)
        tool = _make_task_tool(parent)

        out = await tool.invoke(
            {"subagent_type": "browser_agent", "task_description": "open site"},
            **self._invoke_kwargs(),
        )

        self.assertTrue(out.success)
        self.assertEqual(out.data["output"], "FALLBACK")
        sub.invoke.assert_awaited_once()

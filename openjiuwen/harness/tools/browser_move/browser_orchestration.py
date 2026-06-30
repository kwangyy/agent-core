# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Isolated-subtask orchestration for the browser subagent.

The built-in task loop (``DeepAgent._run_task_loop``) binds one session and
reuses its context for every round (SHARED context). For long browser tasks
this lets per-task context grow without bound and couples unrelated steps.

This module adds an ISOLATED alternative that lives with the browser agent:
each planned task runs in its own session/workspace, and prior-task results
are threaded into the next task's query as bounded text summaries rather than
shared message history. It reuses agent-core's native ``TaskPlan``/``TodoItem``
for the plan (no external ``todo.json``) and drives each task through the
public ``agent.invoke(query, session)`` surface.

Mirrors the behavior of ``browser_pilot``'s ``_run_orchestration_tasks`` family
but stays inside ``openjiuwen.harness`` and uses core primitives.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import FrameworkError, build_error
from openjiuwen.core.common.logging import agent_logger
from openjiuwen.core.session.agent import Session
from openjiuwen.core.single_agent.rail.base import ToolCallInputs
from openjiuwen.harness.rails.base import DeepAgentRail
from openjiuwen.harness.rails.task_planning_rail import TaskPlanningRail
from openjiuwen.harness.schema.task import TaskPlan

# Truncation limits for the per-task memory handoff. Kept small so each task's
# opening context stays bounded regardless of how much earlier tasks produced.
_TASK_DESC_LIMIT = 120
_RESULT_LIMIT = 600

LANGUAGE_INSTRUCTION: Dict[str, str] = {
    "en": "Respond in English.",
    "cn": "请用中文回复。",
}

# Factory that maps a session id to a runnable Session. Injectable so tests can
# pass a lightweight fake instead of a real core Session.
SessionFactory = Callable[[str], Session]


def build_planning_query(user_query: str) -> str:
    """Wrap the user query so the agent plans (calls todo_create) immediately."""
    return (
        "Break the following request into a sequence of discrete, actionable steps "
        "by calling todo_create. Call it IMMEDIATELY.\n\n"
        f"Request: {user_query}"
    )


def _truncate(text: str, limit: int) -> str:
    """Trim ``text`` to ``limit`` chars, appending an ellipsis when cut."""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


# Optional hint fed to subtasks after the first when ``reuse_browser_tab`` is on:
# the executor shares one browser across isolated subtasks, so later subtasks can
# reuse the tab the previous task left on the site instead of re-navigating.
_BROWSER_TAB_REUSE_HINT = (
    "A browser tab is already open and was left on the target site by the previous "
    "task. Reuse that open tab — do NOT navigate to the site's homepage again. "
    "Inspect the current page with a probe/snapshot first, then go directly to your "
    "search or action from there."
)


_TRACE_STEP_LIMIT = 16  # max tool calls kept in an injected trace


def _extract_tool_trace(agent: Any, session: Session) -> str:
    """Compact ``name(args)`` trace of a completed subtask — the METHOD, not the data.

    Only tool names + arguments are kept (args are small: URLs, search terms); the
    large tool RESULTS are intentionally dropped so injecting this stays cheap.
    Best-effort: returns "" if the session context is unavailable.
    """
    try:
        messages = agent.get_current_context(session.get_session_id())
    except Exception:  # noqa: BLE001 - best-effort; never break the loop over a trace
        return ""
    steps: List[str] = []
    for message in messages or []:
        for tc in getattr(message, "tool_calls", None) or []:
            name = getattr(tc, "name", "") or ""
            args = getattr(tc, "arguments", "")
            if not isinstance(args, str):
                try:
                    args = json.dumps(args, ensure_ascii=False)
                except (TypeError, ValueError):
                    args = str(args)
            steps.append(f"{name}({_truncate(args, 100)})")
            if len(steps) >= _TRACE_STEP_LIMIT:
                return " → ".join(steps)
    return " → ".join(steps)


def build_subtask_query(
    task_content: str,
    completed: List[Dict[str, str]],
    task_index: int,
    total_tasks: int,
    *,
    overall_goal: Optional[str] = None,
    tab_preamble: str = "",
    inject_tool_trace: bool = False,
    system_prompt: Optional[str] = None,
    language: Optional[str] = None,
) -> str:
    """Build the opening query for one isolated task.

    The memory-threading contract: the ``overall_goal`` anchors every isolated
    task (the per-task context is fresh, so without it the executor loses the
    original request — e.g. which URL to visit); completed-task summaries are
    included and truncated; ``tab_preamble`` is emitted for the first task only;
    optional system/language instructions are injected; and the task is scoped
    with a "Your task (i of N)" / "Execute ONLY this task" frame.
    """
    parts: List[str] = []

    if system_prompt:
        parts.append(f"[System instruction: {system_prompt.strip()}]")
    if language and language in LANGUAGE_INSTRUCTION:
        parts.append(f"[{LANGUAGE_INSTRUCTION[language]}]")

    if overall_goal:
        parts.append(f"Overall goal: {overall_goal.strip()}")
        parts.append("")

    if tab_preamble:
        parts.append(tab_preamble)

    if completed:
        parts.append("Context — completed tasks:")
        for i, c in enumerate(completed):
            desc = _truncate(c.get("task", ""), _TASK_DESC_LIMIT)
            result = _truncate(c.get("output", ""), _RESULT_LIMIT)
            parts.append(f"  Task {i + 1}: {desc}")
            parts.append(f"  Result: {result}")
            if inject_tool_trace and c.get("trace"):
                parts.append(f"  Tools used (reuse this method): {c['trace']}")
        parts.append("")

    parts.append(f"Your task ({task_index + 1} of {total_tasks}): {task_content}")
    parts.append("")
    parts.append("Execute ONLY this task. Stop as soon as it is done.")
    # Isolated tasks lose the prior task's raw observations (only the summaries
    # above survive), so anchor the answer in real evidence rather than letting
    # the model fabricate from a thin context.
    parts.append(
        "Ground your answer in what you actually observe through your tools or the "
        "context above. If a needed fact is missing, gather it with your tools — "
        "do not fabricate results or invent tool outputs."
    )

    return "\n".join(parts)


async def run_isolated_subtasks(
    agent: Any,
    plan: TaskPlan,
    *,
    session_id: str,
    session_factory: SessionFactory,
    completed: Optional[List[Dict[str, str]]] = None,
    start_index: int = 0,
    tab_preamble: str = "",
    reuse_browser_tab: bool = False,
    inject_tool_trace: bool = False,
    system_prompt: Optional[str] = None,
    language: Optional[str] = None,
    resume_session: Optional[Session] = None,
    resume_input: Optional[Any] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Run ``plan.tasks[start_index:]`` with isolated per-task sessions.

    Each task gets a fresh session ``{session_id}__subtask__{i}`` (and therefore
    its own context/workspace). Prior-task outputs are summarised as text in the
    next task's query so context stays bounded per task. The plan's task statuses
    are advanced by this loop (the executor sessions do not own ``todo_*``).

    On an interrupt, the loop parks ``{current_index, completed, plan}`` in an
    ``interrupt`` event and stops; the caller resumes by calling again with
    ``start_index`` = the parked index plus ``resume_session`` / ``resume_input``.

    Yields orchestration events: ``task_started``, ``plan_update``,
    ``task_complete`` (intermediate tasks), ``assistant_final`` (last task),
    and ``interrupt``.
    """
    completed = list(completed or [])
    tasks = plan.tasks
    total = len(tasks)

    for i in range(start_index, total):
        task = tasks[i]
        task_content = task.content or task.activeForm or ""

        if i == start_index and resume_session is not None:
            sub_session: Session = resume_session
            invoke_input: Any = resume_input
        else:
            sub_session = session_factory(f"{session_id}__subtask__{i}")
            if i == 0:
                preamble = tab_preamble
            elif reuse_browser_tab:
                preamble = _BROWSER_TAB_REUSE_HINT
            else:
                preamble = ""
            invoke_input = build_subtask_query(
                task_content,
                completed,
                task_index=i,
                total_tasks=total,
                overall_goal=plan.goal or None,
                tab_preamble=preamble,
                inject_tool_trace=inject_tool_trace,
                system_prompt=system_prompt,
                language=language,
            )

        yield {
            "type": "task_started",
            "data": {"task_index": i, "task_total": total, "task": task_content},
        }

        result = await agent.invoke(invoke_input, sub_session)
        result_type = result.get("result_type")
        output = str(result.get("output", ""))

        if result_type == "interrupt":
            yield {
                "type": "interrupt",
                "data": {
                    "task_index": i,
                    "completed": list(completed),
                    "plan": plan.to_dict(),
                    "result": result,
                },
            }
            return

        entry = {"task": task_content, "output": output}
        if inject_tool_trace:
            entry["trace"] = _extract_tool_trace(agent, sub_session)
        completed.append(entry)

        if i < total - 1:
            yield {
                "type": "task_complete",
                "data": {
                    "task_index": i,
                    "task_total": total,
                    "task": task_content[:100],
                    "reply": output,
                },
            }
        else:
            yield {"type": "assistant_final", "data": {"reply": output}}

        # Loop-owned status advancement: this task completed, next in progress.
        plan.mark_completed(task.id, summary=_truncate(output, _RESULT_LIMIT))
        if i + 1 < total:
            plan.mark_in_progress(tasks[i + 1].id)
        yield {"type": "plan_update", "data": {"plan": plan.to_dict()}}


async def _load_plan_from_todos(agent: Any, session_id: str, *, goal: str = "") -> TaskPlan:
    """Build a ``TaskPlan`` from the todos the agent's ``TaskPlanningRail`` wrote.

    ``todo_create`` persists todos to a session-scoped ``todo.json`` (it does not
    touch ``state.task_plan``, which is a task-loop-only object). We therefore
    read them back through the rail's ``TodoTool``.

    Raises if the agent has no ``TaskPlanningRail`` — i.e. it was not created with
    ``enable_task_planning=True`` — since no plan could ever be produced.
    """
    rails = agent.find_rails_by_type((TaskPlanningRail,))
    if not rails:
        raise build_error(
            StatusCode.DEEPAGENT_RUNTIME_ERROR,
            error_msg=(
                "Isolated orchestration requires a TaskPlanningRail; create the "
                "browser agent with enable_task_planning=True."
            ),
        )
    tool = rails[0].find_todo_tool()
    if tool is None:
        raise build_error(
            StatusCode.DEEPAGENT_RUNTIME_ERROR,
            error_msg="TaskPlanningRail has no todo tool registered yet.",
        )
    try:
        todos = await tool.load_todos(session_id)
    except FrameworkError as exc:
        # load_todos raises TOOL_TODOS_LOAD_FAILED (a FrameworkError) when the
        # todo file is absent, which for the planning round means it never
        # called todo_create -- i.e. no plan was produced. An empty plan is the
        # correct signal for that, not a failure to propagate. Catch only the
        # framework-level load failure (not bare BaseError) so control-flow
        # exceptions such as Termination/ExecutionError still surface.
        agent_logger.debug("plan_tasks: no todos for session %s (%s); empty plan", session_id, exc)
        return TaskPlan(goal=goal)
    return TaskPlan(goal=goal, tasks=list(todos))


class _PlanOnlyRail(DeepAgentRail):
    """Force-finish the planning round the instant ``todo_create`` runs.

    Without this, the planner keeps going after creating the plan — calling
    ``todo_modify`` and starting to *execute* it — burning model calls. The
    ``todo.json`` is already written by the time ``after_tool_call`` fires, so
    finishing here yields a clean, un-started plan that the isolated loop owns.

    Scoped to the planning session id: the same agent later runs the isolated
    subtasks, and ``strip_rails_by_type`` only marks a registered rail stale
    (it can still fire), so without this guard the rail would force-finish any
    subtask that happens to call ``todo_create``.
    """

    priority = 5

    def __init__(self, planning_session_id: str) -> None:
        super().__init__()
        self._planning_session_id = planning_session_id

    async def after_tool_call(self, ctx: Any) -> None:
        if ctx.session is None or ctx.session.get_session_id() != self._planning_session_id:
            return
        inputs = ctx.inputs
        if isinstance(inputs, ToolCallInputs) and inputs.tool_name == "todo_create":
            ctx.request_force_finish({"output": "plan created", "result_type": "answer"})


async def plan_tasks(agent: Any, user_query: str, session: Session) -> TaskPlan:
    """Run a single, one-shot planning round and return the ``TaskPlan``.

    A temporary ``_PlanOnlyRail`` (scoped to this session) stops the round right
    after ``todo_create`` so the planner only plans (it does not start
    executing). The created todos are read back from the ``TaskPlanningRail``'s
    ``todo.json`` into a ``TaskPlan``. Returns a plan with no tasks if the agent
    produced none. The agent must carry a ``TaskPlanningRail``.
    """
    rail = _PlanOnlyRail(session.get_session_id())
    agent.add_rail(rail)
    try:
        await agent.invoke(build_planning_query(user_query), session)
    finally:
        agent.strip_rails_by_type((_PlanOnlyRail,))
    return await _load_plan_from_todos(agent, session.get_session_id(), goal=user_query)


async def run_browser_isolated_tasks(
    agent: Any,
    user_query: str,
    *,
    session: Session,
    executor: Optional[Any] = None,
    session_factory: Optional[SessionFactory] = None,
    tab_preamble: str = "",
    reuse_browser_tab: bool = False,
    inject_tool_trace: bool = False,
    system_prompt: Optional[str] = None,
    language: Optional[str] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Plan ``user_query`` on ``agent`` then execute its tasks in isolated sessions.

    Two-agent split: ``agent`` is the planner (must carry a ``TaskPlanningRail``)
    and ``executor`` runs the isolated subtasks. Passing a dedicated ``executor``
    built *without* ``enable_task_planning`` (e.g. ``create_browser_agent(...)``)
    keeps todo tools and the todo prompt out of execution, so each subtask opens
    with a much smaller, domain-only context. When ``executor`` is omitted it
    defaults to ``agent`` (one agent does both — backwards compatible).

    When ``session_factory`` is omitted, a default factory builds plain core
    ``Session`` objects carrying the executor's card.
    """
    executor = executor or agent

    if session_factory is None:

        def session_factory(sid: str) -> Session:  # type: ignore[misc]
            return Session(session_id=sid, card=executor.card)

    plan = await plan_tasks(agent, user_query, session)
    yield {"type": "plan_created", "data": {"plan": plan.to_dict()}}

    if not plan.tasks:
        return

    async for event in run_isolated_subtasks(
        executor,
        plan,
        session_id=session.get_session_id(),
        session_factory=session_factory,
        tab_preamble=tab_preamble,
        reuse_browser_tab=reuse_browser_tab,
        inject_tool_trace=inject_tool_trace,
        system_prompt=system_prompt,
        language=language,
    ):
        yield event


__all__ = [
    "LANGUAGE_INSTRUCTION",
    "build_planning_query",
    "build_subtask_query",
    "run_isolated_subtasks",
    "plan_tasks",
    "run_browser_isolated_tasks",
]

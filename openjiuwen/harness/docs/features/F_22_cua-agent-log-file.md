# cua_agent per-session log files

## Metadata

| Item | Value |
| --- | --- |
| Date | 2026-11-03 |
| Scope | `tools/cua/cua_logging.py` (new); `CuaRuntimeRail` log context; `CuaDesktopTool` action lines; `core/common/logging/cua_context.py` (new) and `DefaultLogger` routing |
| Specs | S_18 |
| Test baseline | 267 passed, 1 skipped across `tests/unit_tests/harness/tools/cua/`, `tools/subagent/`, `test_cua_agent.py`, `test_subagent_rail.py`, `test_task_tool*.py`, browser status logging (2026-11-03) |
| Refs | None |

## Background

When a main agent delegated to `cua_agent`, its activity was spread over
`llm.log`, `runner.log`, `sys_operation.log` and the combined log, with
`trace_id=default_trace_id` everywhere and no line linking it to the parent.
The desktop tools themselves logged nothing, so whether an action ran, how long
it took and what it returned were only visible if the next LLM request echoed
the tool message. The browser agent already solves the first part with
`logs/browser_agent.log`. One shared file is not enough for cua, though: a
main agent can run several cua subagents in parallel, and their lines would
interleave with nothing on them to tell the runs apart.

## Decision and state

- **One folder per session.** Each run writes to
  `<log dir>/<session id>/cua_agent.log` (default `./logs/cua_agent/`). The
  session id is the subagent's own (`ctx.session.get_session_id()`); under
  TaskTool that is the `sub_session_id` printed in the combined log's
  `Invoking subagent` line, so the parent's log names the folder to open.
- **Context carries the session id.** A `ContextVar`
  (`core/common/logging/cua_context.py`) holds the active run's session id.
  `CuaRuntimeRail.before_invoke` opens the session's file and sets it;
  `after_invoke` resets it and closes the file. While it is set,
  `DefaultLogger._emit` writes every framework log line to the stdlib logger
  `openjiuwen.cua_agent.<session id>` instead of the combined log, so LLM
  events, `Executing tool` lines and rail messages land in the run's folder.
  Rail callbacks are awaited inline by the callback framework, so the value
  reaches the agent loop; tool calls run in child tasks that inherit it, which
  is what keeps parallel runs apart.
- **Action lines.** `CuaDesktopTool.invoke` writes one `[CUA_ACTION]` JSON line
  per call: tool, args, success, elapsed_ms, result text or error, and the
  number of attached images. Failures are WARNING. Outside a cua run the line
  goes to the common logger.
- **Model replies.** `CuaRuntimeRail.after_model_call` writes one `[CUA_MODEL]`
  JSON line per model call: iteration, finish_reason, reply content, reasoning
  and each tool call's name and raw arguments. The framework logs only the
  length of a streamed reply, so without this the file showed what the agent
  did but not what it said. Nothing is written outside a cua run. With the
  task in `[CUA_INVOKE_START]` and tool results in `[CUA_ACTION]`, the file
  holds the whole conversation except the system prompt and screenshots.
- **Run markers.** `[CUA_INVOKE_START] session=... query=...` /
  `[CUA_INVOKE_END] session=...` bracket each run; a preflight failure logs
  `[CUA_PREFLIGHT]` before raising. Before switching, the rail writes
  `[CUA_AGENT_LOG] session <id> logging to <path>` to the combined log.
- **Append, and close when done.** Files open in append mode, so a resumed
  session (same `sub_session_id`) adds to its history. The handler is closed
  at the end of each run, so a long-running server does not hold one open file
  per past session.
- **Re-entry.** `before_invoke` fires again on approval re-entry. The rail sets
  the context only when it is off, so one invoke holds exactly one token.
- **Path safety.** Characters outside `[A-Za-z0-9._-]` in the session id become
  `_`, so a session id cannot escape the log directory.
- **Configuration.** `OPENJIUWEN_CUA_AGENT_LOG_DIR` (root folder, or
  0/false/no/off/none/null/- to disable), `OPENJIUWEN_CUA_AGENT_LOG_LEVEL`,
  `OPENJIUWEN_CUA_AGENT_LOG_MIRROR_COMMON=1`.

## Rejected alternatives

- **One `cua_agent.log` like the browser agent's, with the session id on each
  line.** Parallel runs would still interleave in one file; reading one run
  means filtering. Folders also leave room for per-run artifacts later.
- **Writing TaskTool's `Invoking subagent` line into the run's file.** It is
  the parent's pointer to the run, so it stays in the combined log.
- **Generalising `browser_context` into a named agent-log context.** Cleaner,
  but it changes a module the browser runtime and `DefaultLogger` already
  depend on; kept as a parallel module to stay additive.
- **Copying the browser raw-audit artifacts.** cua observations are
  screenshots; dumping them is a storage and privacy cost nobody asked for.
  Action lines count images instead.
- **Redacting args.** The request was to log everything the agent does; the
  browser agent's argument redaction is not applied. See known gaps.

## Verification

- `tests/unit_tests/harness/tools/cua/test_cua_logging.py`: per-session path,
  path-escape guard and disable switch; a run writes its markers, action line
  and a framework `logger.info` to its own folder and closes the file after;
  two concurrent runs write to separate folders; a resumed session appends;
  approval re-entry holds one token; actions outside a run stay out of the
  folder; disabled files leave the context off.
- `tests/unit_tests/harness/tools/cua/conftest.py` points every cua test at a
  temporary log directory, so tests do not write into the repository's `logs/`.
- Regression: the suites in the test baseline above.
- Live on Windows (2026-11-03), main agent delegating through TaskTool: one
  run, two parallel runs, an iteration-limit stop, a timeout cancel and a
  `resume_task_id` resume. Each run's lines stayed in its own folder, the
  combined log had none of them, START/END paired, the file was closed after,
  and the resume appended to the original folder.

## Known gaps

- Not redacted: `type_text` content, the full task description and the
  model's replies and reasoning are written in plain text. Do not ship these files from machines where the agent types
  credentials.
- Only the `default` logging backend routes by context (as for the browser
  agent); with `backend: loguru` the framework lines stay in the combined log,
  and only the action and marker lines reach the session file.
- Files are not rotated and folders are never cleaned up.
- If a run ends without `after_invoke` (abort, crash), its handler stays open
  until the process exits; the context itself dies with the run's task.
- The file is opened directly rather than through
  `normalize_and_validate_log_path`, as the browser log does.

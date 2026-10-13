# cua_agent desktop subagent

## Metadata

| Item | Value |
| --- | --- |
| Date | 2026-10-13 |
| Scope | `cua_agent` preset, `tools/cua/` (driver config, capability catalog, rails), TaskTool cua delegation and resume, `os_control` permission category, shared `MultimodalContextSummarizerRail`, MCP `structuredContent` opt-in, `StdioClient` owner task |
| Specs | S_04, S_05, S_08, S_18 |
| Test baseline | 218 passed, 0 failed across the targeted suites listed under Verification (2026-10-13) |
| Refs | Local request for a desktop counterpart to browser_agent; no issue assigned |

## Background

The harness could drive web pages (browser_agent) and Android devices
(mobile_gui), but not native desktop applications. cua-driver is a native
daemon (`cua-driver serve`) that exposes desktop inspection and input over
MCP: window lists, accessibility element trees, screenshots, and mouse,
keyboard and app-lifecycle actions. `cua-driver mcp` is a thin stdio proxy
to that daemon.

`cua_agent` wraps the driver as a preset subagent. It reuses the browser
agent's pattern: a factory pair, a trusted capability catalog, and an MCP
tool allowlist. Most of the work is in rails that repair failure modes seen
in recorded runs. Driver errors come back as ordinary tool results rather
than exceptions, so nothing in the retry stack sees them.

## Decision and state

### Runtime and tool surface

- The driver is consumed as a stdio MCP server, not through its Python SDK,
  the same way browser_agent consumes Playwright MCP. All OS authority stays
  in the daemon. Daemon preflight and command resolution are in F_21.
- `resolve_cua_capabilities` always includes `core` (read-only observation
  and session tools). The factories grant `input` and `app_lifecycle` by
  default; `recording` and `app_termination` (`kill_app`) must be
  requested. The built-in rules deny `kill_app`, so the default grant
  leaves it out, and the prompt teaches it only when it is granted. Tool names are listed explicitly,
  so a tool added in a new driver release is not exposed until it is
  reviewed. The driver's browser tools are in no capability: web tasks go to
  browser_agent, and the prompt tells the model so. Electron desktop apps
  such as Discord or VS Code are still desktop work.
- `CuaRuntimeRail` caches each setup step after its first success: preflight,
  ability registration with the allowlist, and `start_session`. Before
  caching, the security gate's approval path re-entered `before_invoke`
  before every gated call. Across 73 recorded runs, 38.9% of all driver
  calls were session lifecycle, and most of those were no-ops. Live runs
  after the change measured 8.5%. The rail no longer calls `end_session`;
  the daemon's idle TTL reclaims the session.
- The driver puts window bounds and screen size only in MCP
  `structuredContent`, which the core MCP client dropped. Without them the
  model never sees coordinates. `McpServerConfig.include_structured_content`
  is a new opt-in (default `False`). When on, the structured JSON is appended
  to the result text, skipped when it repeats the text (FastMCP's
  `{"result": ...}` wrapper), and capped at 4000 characters. cua-driver is
  the only production config that turns it on.

### Perception modes

- Text mode is the default (`cua_screenshot_multimodal=False`). Screenshots
  reach context only as text placeholders, and the model works from the
  element tree. Multimodal mode attaches screenshots as images and needs a
  vision model.
- The system prompt is assembled per mode. In a bench run, a prompt that
  called screenshots "optional" doubled `include_screenshot=true` requests,
  from 7 to 14, for images a text-mode model cannot see. The text-mode
  prompt now says screenshots are never delivered. With the shipped prompt,
  0 of 49 snapshots in a 12-run bench asked for a screenshot.
- In multimodal mode, `CuaScreenshotDownscaleRail` re-encodes any screenshot
  over 25,000 image tokens as a bounded JPEG. It appends the scale factor to
  the tool message. Live runs showed the model reading pixel positions off a
  downscaled image and clicking them as window coordinates; with the note it
  converts correctly.
- The multimodal prompt tells the model to start clicks and drags at least
  30 px inside a surface. The Paint drag failures came from estimates 10–30 px
  outside the canvas edge, not from the driver: foreground drags inked 7 of 7
  when issued directly.

### Context retention

- `ToolResultWindowProcessor` keeps the last `cua_snapshot_keep_last_k`
  results of the three snapshot tools. Older results are offloaded and leave
  a 200-character preview. With the processor default of 3000 characters,
  superseded element trees, whose `element_index` handles are stale by
  construction, cost about 19k tokens over a 25-step run, and the processor
  saved only 3.6% net.
- `cua_snapshot_keep_last_k` defaults to 3 and drives both this window and
  the number of screenshots `MultimodalContextSummarizerRail` keeps. With 1,
  a two-window task evicted the first window's snapshot as soon as the
  second was taken, and the agent looped on read-only re-queries.
- `MultimodalContextSummarizerRail` moved from `tools/mobile_gui/rails/` into
  `harness/rails/` so cua and mobile share one implementation. It gained
  `protected_user_message_names`; the mobile module stays as a subclass that
  pre-fills mobile's protected names, so its import path still works.

### Action repair rails

- `CuaElementAddressingRail` drops `x`/`y` when a call also carries
  `element_index`. The driver rejects calls with both. Across 165 recorded
  runs that rejection caused 49 failures, all on `type_text`. The prompt
  already prefers element addressing, so the rail keeps the handle.
- `CuaRepeatFailureRail` counts identical calls that get the same
  non-success response. At 3 it appends advice; at 6 it refuses the call
  before it reaches the driver. In the corpus the longest legitimate run of
  identical non-success responses was 4, and the stuck loop reached 20. One
  recorded run spent 172 turns and 510 s sending the same rejected
  `type_text` 55 times. Repetition alone is not the signal: `press_key tab`
  ×27 while walking a form is legitimate and driver-confirmed.
- `CuaSnapshotFreshnessRail` removes `query` from the first snapshot of a
  window after an action on it, and says so in the result. A filtered
  verification snapshot hid the evidence being checked: a filter on the
  clicked item's name hid the "1 item selected" status bar and started a
  20-step recovery. A prompt rule against filtering verification snapshots
  did not change model behaviour (query-after-action went from 15 to 18), so
  the rail enforces it. In a 12-run bench the model requested `query` 11
  times; the rail dropped the 7 that followed actions, and no verification
  snapshot was filtered.
- `CuaDeliveryModeRail` (installed only when `cua_delivery_mode` is set)
  makes delivery an operator setting. It injects the mode when the argument
  is absent. Under `background` it rejects an explicit `foreground` rather
  than rewriting it, because the driver's own `background_unavailable` error
  tells the model to retry in foreground; a silent rewrite would loop the
  model against that advice. Pinning a mode also appends a prompt suffix,
  so the base prompt's "escalate to foreground" guidance does not cause a
  guaranteed rejection on every run.
- `CuaUserTakeoverRail` (opt-in, `cua_pause_on_user_input`) holds driver
  calls while the OS reports recent user input. It releases after 5 s idle
  and rejects with a report after 60 s. Under foreground delivery the
  agent's own input is real OS input, so an input event that falls inside
  the agent's last input call, padded by 1.5 s, is attributed to the agent.
  It is off by default because bench and demo runs on a shared desktop were
  paused by the operator's own mouse.

### Progress and resume

- The driver prefixes a confirmed action with `✅` and a failed one with
  `❌`; a reply with neither ran but could not be verified.
  `CuaRepeatFailureRail` counts every unmarked reply, since a loop of
  unverified repeats is still a loop. `CuaProgressRail` lists only `❌`
  replies as blockers. One OSWorld-bridge run had 86 unverified coordinate
  clicks and 1 real failure; treating every unmarked reply as a blocker
  listed 87.
- `CuaProgressRail` keeps up to 8 normalized content digests per window and
  appends a replan advisory when one state recurs 3 times (dialog A → B →
  A). Normalization strips the fields the driver regenerates on every
  capture: screenshot bytes, `snapshot_id`, `element_token` values, and the
  screenshot placeholder's length.
- At run end the rail writes `result["cua_result"]`. A repeat-failure hard
  block or a revisit advisory makes the status `blocked`.
- cua_agent is non-sticky. With a sticky session, two unrelated desktop
  errands in one parent conversation shared a sub-session and leaked
  context, the same bug browser_agent had. Resume is an explicit
  `resume_task_id`, shared with browser_agent through
  `_EXPLICIT_RESUME_SUBAGENT_TYPES`.
- TaskTool cancels a cua delegation 20 s before the tool ceiling and returns
  a successful, resumable result instead of a bare timeout that loses every
  step. The ceiling is now configurable through
  `OPENJIUWEN_SUBAGENT_TASK_TIMEOUT_S`. The driver's `get_session_state` has
  no last-action record, so the timeout result sets
  `last_action_confirmed=False`. The resumed run re-snapshots and checks
  whether the interrupted action took effect.
- A resume with a stored record rewrites the task description to carry the
  last status, window, blockers and recovery hint, followed by the parent's
  remaining task. browser_agent narrows a resume to unresolved evidence
  slots; a desktop run has no slot model, so the stored context is that
  record.

### Permissions

- `_tool_category` has a new `os_control` category for the 12 mutating
  desktop tools, keyed by model-facing MCP name (`mcp_cua-driver_<tool>`),
  because the permission engine sees that name. Before this, desktop tools
  had no category and parameter rules could not match them. Read-only tools
  stay uncategorized.
- Built-in rules: `kill_app` is denied (CRITICAL); `launch_app` and
  `type_text` ask (HIGH). These apply even when a config sets
  `defaults: {"*": "allow"}`, and to instance-isolated servers:
  `mcp_cua-driver-<key>_<tool>` is mapped back to the default name first.
- `build_cua_agent_config` forwards `permissions` / `permission_host` to the
  materialized subagent. TaskTool does not pass interrupt results back to
  the parent, so a built-in ASK cannot reach the outer conversation from
  inside a delegation. A policy that can produce ASK needs a
  `ToolPermissionHost.request_permission_confirmation` hook that answers
  in-process.

### MCP stdio transport

- `StdioClient` now enters and exits the transport's context managers on a
  dedicated owner task. mcp's stdio transport opens an anyio task group on
  the task that enters it. Entered on the caller's task, that cancel scope
  stayed on the caller's stack, and the `asyncio.timeout` around a TaskTool
  delegation, whose subagent connects its MCP server on first use, failed to
  exit with "Attempted to exit a cancel scope that isn't the current task's
  current cancel scope". `BrowserMoveStdioClient` already used this actor
  pattern.

## Rejected alternatives

- **cua's own agent loop (`cua-agent`) or `cua-sandbox`.** The agent loop
  duplicates DeepAgent; the sandbox targets disposable VM desktops, not the
  user's machine.
- **Prefix-matched tool allowlist.** It would expose new driver tools before
  their capability and permission policy were reviewed.
- **Rewriting a foreground request to background.** See
  `CuaDeliveryModeRail` above: it hides the rejection and loops the model
  against the driver's advice.
- **A hard stale-element gate.** Blocking element actions on a window acted
  on since its last snapshot was measured against the corpus: 357 such
  actions succeeded and 70 failed, so a gate would have blocked mostly
  working calls. The driver already rejects truly stale element tokens with
  its own message.
- **Prompt-only rule against filtered verification snapshots.** Measured as
  ignored (above); the freshness rail enforces it instead.
- **Porting browser_agent's working-context subsystem** (phase and field
  coverage tracking). It is built around DOM pages and evidence slots.
  Desktop tasks are shorter; its one transferable idea, forcing a replan
  after no progress, became the revisit advisory.
- **Tracking the in-flight action in a rail** to confirm it after a
  timeout. The security gate re-enters `before_invoke` mid-run, which would
  produce false positives. The timeout record lives in TaskTool.
- **Sticky cua sessions.** They leaked context between unrelated
  delegations; see Progress and resume.

## Verification

- Unit tests: `tests/unit_tests/harness/test_cua_agent.py`,
  `tests/unit_tests/harness/tools/cua/` (rails, capabilities, user
  activity), `tests/unit_tests/harness/tools/test_task_tool.py`,
  `tests/unit_tests/harness/security/test_tiered_policy_os_control.py`,
  `tests/unit_tests/harness/rails/test_multimodal_context_summarizer_rail.py`,
  `tests/unit_tests/core/foundation/tool/mcp/client/test_stdio_owner_task.py`,
  `tests/unit_tests/core/foundation/tool/test_streamable_http_client.py`.
  Also `tests/unit_tests/harness/tools/mobile_gui/test_multimodal_context_summarizer_rail.py`
  for the mobile shim. Result: 218 passed, 0 failed.
- Permission engine, checked through `PermissionEngine` rather than the raw
  rule dict: with `defaults: {"*": "allow"}`, `kill_app` returns DENY,
  `launch_app` and `type_text` return ASK, and `list_windows` and `click`
  return ALLOW.
- Bench (tasks notepad_type, calc_multiply, explorer_select,
  settings_about; each verified in code against the OS):
  12/12 baseline runs at 3 repeats; 12/12 with the freshness rail;
  4/4 on driver 0.28.1 (F_21).
- Live demos on Windows:
  - A task that orders foreground input under a background pin. The model kept to background, focus never moved,
    and it reported that it could not comply.
  - A Paint canvas under a background pin. No foreground requests, focus untouched, and an honest
    could-not-complete report.
  - The approve and deny paths of the `launch_app` ASK rule.
  - A coordinator delegating to cua_agent. A DENY rule blocked `launch_app powershell`, and the agent reported the block.

## Limits

- Verified on Windows only. The user-activity probe exists only on Windows;
  elsewhere the takeover rail is inert. The daemon needs an interactive
  desktop session, so SSH and headless Session 0 cannot see the GUI.
- The revisit threshold has measured precision but no measured recall: it
  fired 3 times in 96 bench runs, all in runs that were cycling, but 7 of
  9 runs that hit the iteration cap got no advisory. The bench now
  records `revisit_count`; retune from that data.
- `cua_result.status` is a heuristic, not a task-completion judgment.
- Blockers depend on the driver emitting `❌`. A driver that does not mark
  failures produces no blockers.
- Without Pillow, an oversized screenshot is sent at full size, the same as
  `ReadFileTool`.
- On UWP apps `hotkey` does not reach the window; `press_key` with
  modifiers does. The prompt does not say this yet.

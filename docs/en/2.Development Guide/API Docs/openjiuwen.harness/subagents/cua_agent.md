# openjiuwen.harness.subagents.cua_agent

Desktop sub-agent that operates native applications on the host computer through [cua-driver](https://github.com/trycua/cua/tree/main/libs/cua-driver).

---

## Overview

`cua_agent` drives real application windows: it lists windows, reads a window's accessibility element tree, and sends mouse, keyboard and app-lifecycle actions. cua-driver runs as a stdio MCP server (`cua-driver mcp`), a thin proxy to a `cua-driver serve` daemon that executes every tool in the interactive desktop session.

Web pages are not this agent's job. Browser automation belongs to [`browser_agent`](./browser_agent.md), and cua-driver's own browser tools are not offered. Electron desktop apps such as Discord, Slack or VS Code are ordinary desktop windows and are handled here.

Use `create_cua_agent` to build a standalone agent, or `build_cua_agent_config` to register it as a sub-agent that a parent reaches through `task_tool` with `subagent_type="cua_agent"`. `DeepAgent` also accepts the factory name `"desktop_agent"` as an alias.

**Platform**: verified on Windows. The daemon needs an interactive desktop session, so SSH or headless sessions cannot see the GUI.

### Setup

`uv sync --extra cua` (or `pip install 'openjiuwen[cua]'`) installs the `cua-driver` package (version 0.28.1 or later), which bundles the driver binary. From 0.28.1, the MCP proxy launches its own daemon when none is running and stops it when the proxy exits, so no separate install or daemon is needed.

| Environment variable | Purpose |
|---|---|
| `CUA_DRIVER_MCP_COMMAND` | Driver command. Default: the binary bundled next to the Python interpreter, then `cua-driver` on `PATH`, then the Windows installer location. |
| `CUA_DRIVER_MCP_ARGS` | Arguments for the MCP proxy. Default: `mcp`. |
| `CUA_DRIVER_MCP_TIMEOUT_S` | MCP request timeout in seconds. Default: `120`. |
| `CUA_DRIVER_POLICY_FILE`, `CUA_DRIVER_MANAGED_POLICY_FILE` | Forwarded to the driver to enable its own permission policy below the agent's. |
| `OPENJIUWEN_SUBAGENT_TASK_TIMEOUT_S` | Ceiling for one `task_tool` delegation, for all sub-agent types. Default: `720`. |

Before the first model call, `CuaRuntimeRail` checks that the driver can reach a working daemon and fails with a remediation message if it cannot.

---

## function openjiuwen.harness.subagents.create_cua_agent

```python
create_cua_agent(
    model: Model,
    *,
    card: AgentCard | None = None,
    system_prompt: str | None = None,
    tools: list[Tool | ToolCard] | None = None,
    mcps: list[McpServerConfig] | None = None,
    subagents: list[SubAgentConfig | DeepAgent] | None = None,
    rails: list[AgentRail] | None = None,
    enable_task_loop: bool = False,
    max_iterations: int = 25,
    workspace: str | Workspace | None = None,
    skills: list[str] | None = None,
    backend: Any | None = None,
    sys_operation: SysOperation | None = None,
    language: str | None = None,
    prompt_mode: str | None = None,
    cua_capabilities: list[str] | None = None,
    cua_instance_key: str = "",
    cua_delivery_mode: Literal["background", "foreground"] | None = None,
    cua_screenshot_multimodal: bool = False,
    cua_snapshot_keep_last_k: int = 3,
    cua_pause_on_user_input: bool = False,
    **config_kwargs,
) -> DeepAgent
```

Create the desktop sub-agent. The cua-driver MCP server and the [built-in rails](#built-in-rails) are added after any `mcps` and `rails` you pass.

**Parameters**:

- **model** (Model): Model instance. Multimodal mode needs a vision-capable model.
- **card** (AgentCard, optional): Agent identity card. Default: `None`, which creates a card named `cua_agent`.
- **system_prompt** (str, optional): Replaces the default prompt. Default: `None`, which selects the prompt for the current [perception mode](#perception-modes). A [delivery-mode](#delivery-mode) suffix is still appended when `cua_delivery_mode` is set.
- **tools**, **mcps**, **subagents**, **rails**, **enable_task_loop**, **workspace**, **skills**, **backend**, **sys_operation**, **prompt_mode**: Passed to [`create_deep_agent`](../factory.md#function-openjiuwenharnesscreate_deep_agent).
- **max_iterations** (int, optional): Maximum iterations. Default: `25`. Every element action is preceded by a fresh snapshot, so budgets under about 20 tend to run out mid-task.
- **language** (str, optional): `"cn"` or `"en"`. Default: `None` (resolved by the harness).
- **cua_capabilities** (list[str], optional): Capabilities to grant; see [Capabilities](#capabilities). Default: `None`, which grants `input` and `app_lifecycle`. `core` is always included.
- **cua_instance_key** (str, optional): Isolates the MCP registration. Agents with the same key, or no key, share one driver server registration. Default: `""`.
- **cua_delivery_mode** (`"background"` | `"foreground"`, optional): Pins input delivery for every action. Default: `None`, which lets the model choose per call.
- **cua_screenshot_multimodal** (bool, optional): Attach screenshots to context as images. Default: `False`.
- **cua_snapshot_keep_last_k** (int, optional): Number of recent snapshots, and in multimodal mode screenshots, kept in context. Must be at least 1. Default: `3`. Use `1` for single-window tasks to save context.
- **cua_pause_on_user_input** (bool, optional): Hold desktop actions while the user is using the mouse or keyboard. Default: `False`. Only works on Windows.
- ****config_kwargs**: Passed to `create_deep_agent`, for example `permissions` and `permission_host`.

**Returns**:

**[DeepAgent](../deep_agent.md#class-openjiuwenharnessdeepagent)**: The configured desktop sub-agent.

**Raises**:

- **ValueError**: An unknown capability name, a `cua_delivery_mode` other than `"background"`, `"foreground"` or `None`, or `cua_snapshot_keep_last_k` below 1.

**Example**:

```python
from openjiuwen.core.runner import Runner
from openjiuwen.harness.subagents import create_cua_agent

agent = create_cua_agent(model, language="en", cua_capabilities=[])  # read-only: core only

await Runner.start()
try:
    result = await Runner.run_agent(
        agent,
        {"query": "List the open application windows.", "conversation_id": "cua-demo"},
    )
finally:
    await Runner.stop()
```

---

## function openjiuwen.harness.subagents.build_cua_agent_config

```python
build_cua_agent_config(
    model: Model,
    *,
    card: AgentCard | None = None,
    system_prompt: str | None = None,
    tools: list[Tool | ToolCard] | None = None,
    mcps: list[McpServerConfig] | None = None,
    rails: list[AgentRail] | None = None,
    enable_task_loop: bool = False,
    max_iterations: int = 25,
    workspace: str | Workspace | None = None,
    skills: list[str] | None = None,
    backend: Any | None = None,
    sys_operation: SysOperation | None = None,
    language: str | None = None,
    prompt_mode: str | None = None,
    cua_capabilities: list[str] | None = None,
    cua_instance_key: str = "",
    cua_delivery_mode: Literal["background", "foreground"] | None = None,
    cua_screenshot_multimodal: bool = False,
    cua_snapshot_keep_last_k: int = 3,
    cua_pause_on_user_input: bool = False,
    permissions: dict | None = None,
    permission_host: Any | None = None,
) -> SubAgentConfig
```

Build a [`SubAgentConfig`](../schema/config.md#class-openjiuwenharnessschemasubagentconfig) that the parent materializes with `create_cua_agent` when it first delegates. The arguments are validated now, with the same `ValueError` cases as `create_cua_agent`.

**Parameters**: Same as `create_cua_agent`, without `subagents` and `**config_kwargs`, plus:

- **permissions** (dict, optional): Permission policy for the sub-agent. Default: `None`.
- **permission_host** (ToolPermissionHost, optional): Host that answers permission confirmations. Default: `None`. See [Permissions](#permissions).

**Returns**:

**[SubAgentConfig](../schema/config.md#class-openjiuwenharnessschemasubagentconfig)**: The sub-agent configuration, with `factory_name="cua_agent"`.

**Example**:

```python
from openjiuwen.harness.factory import create_deep_agent
from openjiuwen.harness.subagents import build_browser_agent_config, build_cua_agent_config

coordinator = create_deep_agent(
    model=model,
    subagents=[
        build_cua_agent_config(model, language="en"),
        build_browser_agent_config(model, language="en"),
    ],
)
```

---

## Capabilities

`cua_capabilities` selects driver tools from a fixed catalog (`openjiuwen.harness.tools.cua`). The selection becomes the only tool allowlist on the driver MCP server. New driver tools are not exposed until they are added to the catalog.

| Capability | Tools | Granted by default |
|---|---|---|
| `core` | `list_apps`, `list_windows`, `get_window_state`, `get_accessibility_tree`, `get_desktop_state`, `get_screen_size`, `get_cursor_position`, `get_session_state`, `start_session`, `end_session`, `check_permissions`, `health_report`, `zoom` | Always |
| `input` | `click`, `double_click`, `right_click`, `drag`, `type_text`, `press_key`, `hotkey`, `scroll`, `set_value` | Yes |
| `app_lifecycle` | `launch_app`, `bring_to_front` | Yes |
| `app_termination` | `kill_app` | No |
| `recording` | `start_recording`, `stop_recording`, `get_recording_state`, `replay_trajectory` | No |

Pass `[]` for a read-only agent (core only).

### function openjiuwen.harness.tools.cua.resolve_cua_capabilities

```python
resolve_cua_capabilities(
    requested_names: Iterable[str] | None,
    available_capabilities: Iterable[CuaCapability] = DEFAULT_CUA_CAPABILITIES,
) -> ResolvedCuaCapabilities
```

Expand capability names into a tool allowlist. `core` is always added, duplicates are removed, and unknown names are reported in `rejected_names` instead of raising. The factories raise `ValueError` when `rejected_names` is not empty.

**Returns**:

**ResolvedCuaCapabilities**: A frozen dataclass with `requested_names`, `selected_names`, `rejected_names` and `allowed_tool_names`.

---

## Perception modes

The agent observes a window with `get_window_state(pid, window_id)`, which returns a snapshot: the element tree, plus a screenshot when requested. Actions prefer `element_index` addressing: it works on background windows and does not move the user's cursor. Every element action needs a fresh snapshot, because `element_index` values are valid only against the latest snapshot of that window.

| | Text mode (default) | Multimodal mode |
|---|---|---|
| Flag | `cua_screenshot_multimodal=False` | `cua_screenshot_multimodal=True` |
| Model | Any | Vision-capable |
| Screenshots | Reach context as text placeholders; the prompt tells the model it never receives them | Attached as images on request; oversized images are downscaled to JPEG with a note giving the scale factor |
| Pixel coordinates | Derived from element frames only | Read off the latest screenshot |
| If a window has no usable tree | The agent reports that the window needs a vision-capable agent | The agent can work from the screenshot |

Window bounds and screen size come from the driver's MCP `structuredContent`, which this agent always includes in tool results.

---

## Delivery mode

cua-driver input actions take a `delivery_mode` argument. `background` posts input without raising the window or taking focus. `foreground` briefly brings the window forward. By default the model keeps `background` and switches a single action to `foreground` only after a background attempt verifiably failed, which some apps (UWP/WinUI, some Chromium, Java and Qt surfaces) require.

Set `cua_delivery_mode` to make it an operator decision:

- `"background"`: every input action is sent in background. An explicit `foreground` request is rejected before it reaches the driver, and the agent reports the limitation instead.
- `"foreground"`: every input action is sent in foreground; focus changes are expected.

In both cases a matching instruction is appended to the system prompt.

---

## Built-in rails

`create_cua_agent` adds these rails after the caller's rails:

| Rail | When | What it does |
|---|---|---|
| `CuaRuntimeRail` | Always | Checks the driver and daemon, registers the MCP server with the capability allowlist, and starts one driver session per run. |
| `CuaElementAddressingRail` | Always | Drops `x`/`y` from a call that also has `element_index`, since the driver rejects calls that use both. |
| `CuaRepeatFailureRail` | Always | When an identical call keeps getting the same non-success response, adds advice on the 3rd time and refuses the call on the 6th. |
| `CuaSnapshotFreshnessRail` | Always | Removes `query` from the first snapshot after an action on that window, so the check is not filtered. |
| `CuaProgressRail` | Always | Advises a new approach when a window returns to the same state 3 times, and reports the run's progress (see below). |
| `CuaDeliveryModeRail` | `cua_delivery_mode` set | Enforces the pinned delivery mode. |
| `CuaUserTakeoverRail` | `cua_pause_on_user_input=True` | Holds actions while the user is active, resumes after 5 s idle, and stops with a report after 60 s. |
| `CuaScreenshotDownscaleRail` | Multimodal mode | Downscales oversized screenshots. |
| [`MultimodalContextSummarizerRail`](../rails/rails.md) | Multimodal mode | Keeps only the last `cua_snapshot_keep_last_k` screenshots. |
| `ContextProcessorRail` | No `ContextProcessorRail` passed in `rails` | Keeps the last `cua_snapshot_keep_last_k` snapshot results; older ones are offloaded to the workspace and replaced by a short preview. |

---

## Delegation results and resume

When a parent delegates through `task_tool`, the result `data` for `cua_agent` contains:

| Field | Meaning |
|---|---|
| `output` | The sub-agent's answer. |
| `resume_task_id` | Pass it back to continue the same desktop session. |
| `cua_result` | Progress report from `CuaProgressRail`. |
| `retryable` | `True` unless `cua_result.status` is `"completed"`. |
| `resume_context` | `status`, `blockers`, `current_window`, `recommended_recovery`, `resume_count`. |

`cua_result.status` is `"completed"`, `"partial"` (some actions failed) or `"blocked"` (a repeated failure was refused, or the window kept returning to the same state). It is a heuristic, not a judgment that the task succeeded. `blockers` lists up to 5 failed actions as `tool: first line of the error`.

**Timeouts**: a `cua_agent` delegation stops itself 20 s before the `task_tool` ceiling and returns a successful result with `status="timeout"`, `retryable=True` and `resume_context.last_action_confirmed=False`. The desktop session and the steps taken so far are kept, but the action running at the cut-off may have taken effect without being recorded.

**Resuming**: call `task_tool` again with `subagent_type="cua_agent"`, the same `resume_task_id`, and a `task_description` that covers only what is still missing. The sub-agent receives its last status, window, blockers and recovery hint, and is told to re-snapshot before acting. Without `resume_task_id`, each delegation starts a new session.

---

## Permissions

When the permission engine is enabled, mutating desktop tools are in the `os_control` category and these built-in rules apply:

| Rule | Tool | Decision |
|---|---|---|
| `cua_desktop_force_terminate_app` | `kill_app` | Deny |
| `cua_desktop_launch_app` | `launch_app` | Ask |
| `cua_desktop_type_text` | `type_text` | Ask |

Rules refer to tools by their model-facing names, such as `mcp_cua-driver_launch_app`. An `os_control` rule with `pattern: "*"` matches every call to the tool. Any other pattern (exact, glob or `re:`) is matched against the call's `text`, `name`, `bundle_id`, `value`, `key` or `keys` argument. The rules also cover agents created with a non-empty `cua_instance_key`, whose tool names are `mcp_cua-driver-<key>_<tool>`.

`kill_app` is denied by the built-in rules, so it is not granted by default. Request `app_termination` to grant it; the system prompt then adds guidance to close applications cooperatively first.

Inside a `task_tool` delegation, an Ask decision cannot reach the user through the parent conversation. Pass a `permission_host` whose `request_permission_confirmation` answers it in-process, or use only Allow and Deny rules.

```python
permissions = {
    "enabled": True,
    "schema": "tiered_policy",
    "defaults": {"*": "allow"},
    "rules": [
        {
            "id": "deny_shell_launch",
            "tools": ["mcp_cua-driver_launch_app"],
            "match_type": "os_control",
            "pattern": "re:(?i)^(powershell|pwsh|cmd)(\\.exe)?$",
            "action": "deny",
            "severity": "CRITICAL",
        }
    ],
}
```

Give every rule an explicit `action`. `severity` alone never makes a rule deny.

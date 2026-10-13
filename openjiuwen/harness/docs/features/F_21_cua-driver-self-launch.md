# cua-driver self-launching daemon

## Metadata

| Item | Value |
| --- | --- |
| Date | 2026-10-13 |
| Scope | `CuaRuntimeRail` daemon preflight, cua-driver command resolution, `cua-driver` dependency floor |
| Specs | S_18 (cua desktop subagent section); the subagent as a whole is archived in F_22 |
| Refs | Local request: "`uv sync` and go" setup for cua_agent; no issue assigned |

## Background

Running cua_agent took three manual steps: the vendor installer
(`irm https://cua.ai/driver/install.ps1 | iex`), `cua-driver autostart kick`
to start the `cua-driver serve` daemon, and an LLM `.env`. `CuaRuntimeRail`
aborted the run unless `cua-driver call get_screen_size` reached a running
daemon.

The `cua` extra installs the `cua-driver` PyPI package, and it bundles the
full native driver (`cua_driver/bin/cua-driver.exe`, the UIA helper), not
just a client. On a developer machine this gave two copies: the installer's
0.10.0 daemon, started from a logon Scheduled Task, and the venv's 0.28.1
CLI. Under `uv run` the venv client reached the old daemon, and every driver
call failed with "incompatible: Unknown method: metadata".

## Decision and state

Verified live on Windows with no daemon serving `\\.\pipe\cua-driver`:

| Command | 0.10.0 | 0.28.1 |
| --- | --- | --- |
| `cua-driver mcp`, the proxy the agent spawns | refuses to start | launches a child `serve` daemon and reaps it when the proxy exits |
| `cua-driver call <tool>` | fails fast | fails fast |

The MCP path therefore needs no daemon from 0.28.1 on. Only the rail's own
CLI probe still assumed one.

- `CuaRuntimeRail._ensure_daemon_reachable` still probes with
  `call get_screen_size`. If the probe fails, it runs `status`:
  - Daemon running: the run fails loud with the probe output. A stale or
    incompatible daemon would capture the proxy instead of a fresh one.
  - No daemon: it reads `--version` from stdout and passes when the version
    is at least `_SELF_LAUNCH_MIN_VERSION` (0.28.1). It reads stdout only
    because stderr can carry an "update available" banner that names a newer
    version.
  - A missing binary or an older driver fails loud, and the remediation
    names `uv sync --extra cua`.
- The rail still caches only success, so a failed check is retried on the
  next invoke.
- `_resolve_command` order: `CUA_DRIVER_MCP_COMMAND`, then the binary in
  `sysconfig.get_path("scripts")`, then PATH, then the installer's
  `%LOCALAPPDATA%` path. The bundled binary wins even when the venv is not
  activated or an older install is on PATH. Client and self-launched daemon
  then always share a version.
- `pyproject.toml` declares `cua-driver>=0.28.1; sys_platform != 'linux'` in
  the `cua` optional extra, like `mobile-gui`. 0.28.1 ships only
  `manylinux_2_31` Linux wheels, which the glibc 2.28 CI image cannot
  install, and the driver is verified on Windows only. The marker keeps
  `uv sync --all-extras` working there; a Linux desktop host installs
  `cua-driver` itself or sets `CUA_DRIVER_MCP_COMMAND`.

## Rejected alternatives

- **Rail auto-starts the daemon.** Running `autostart kick` or a detached
  `serve` on probe failure was the first plan. Live testing showed the 0.28.1
  proxy already does this, with the right lifetime: the daemon exits with the
  proxy. A rail-owned daemon would outlive the agent, needs process
  management, and duplicates driver behavior.
- **`autostart enable` as a setup step.** It registers a logon Scheduled
  Task, which is machine state that a package install cannot create. It ties
  the task to the venv path, and on the test machine deleting the old task
  needed admin rights. It remains an optional way to keep a daemon always on.
- **Probe through `cua-driver mcp`.** That would exercise the real path, but
  it spawns and reaps a full daemon only to check it, and the agent's own MCP
  connection then pays that cost again.
- **Drop the preflight.** MCP connects lazily, so a missing binary or
  incompatible daemon would surface later as tool errors instead of one
  actionable message before the first model call.

## Verification

- Unit tests in `tests/unit_tests/harness/tools/cua/test_cua_rails.py`:
  - no daemon with an old driver fails loud;
  - no daemon with 0.28.1 passes (subcommands `call`, `status`, `--version`);
  - a running daemon that fails the probe fails loud;
  - the version is read from stdout, not the stderr banner;
  - a failed probe is retried and then cached.
- Unit test in `tests/unit_tests/harness/test_cua_agent.py`: the bundled
  binary is preferred over PATH.
- Mutations: removing the version floor, or parsing the version from stderr,
  each fails 3 tests.
- Live run with no daemon and no `CUA_DRIVER_MCP_COMMAND`:
  - The agent resolved `.venv\Scripts\cua-driver.EXE` 0.28.1.
  - The rail logged that the proxy would launch the daemon.
  - 57 driver tools registered, and the task completed.
  - No cua-driver process remained afterwards.
  - Rail setup took about 1.6 s.
- Bench on 0.28.1 (four Windows tasks, one run per task, qwen3.7-plus,
  text mode): 4/4 verified successes.

## Limits

- `notepad_type` on 0.28.1 verified its result but ended `blocked`: it hit
  max iterations with 5 revisits in 179 s. On 0.10.0 it completed 6/6 in
  85–108 s. One run cannot separate a driver behavior change from noise;
  re-run with `--repeat 3`.
- Each agent process now gets its own proxy-launched daemon unless one is
  already serving the pipe. With concurrent agent processes, the first
  process's daemon serves the others and goes away when that process exits.
  This has not been tested.
- A leftover installer daemon (0.10.0) on the pipe makes the rail fail loud
  rather than self-heal. Removing its logon task may need admin rights.
- Behavior is verified on Windows only.

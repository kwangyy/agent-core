#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Discord Wordle: the cua subagent runs /wordle, then types the words you pick.

Two acts, run as ONE continuous agent conversation, against the REAL Discord
desktop client (an Electron/Chromium app, so a native desktop window —
cua_agent's domain, not browser_agent's):

* **Act 1 — reach the board**: locate the Discord window, switch to the target
  server (Ctrl+K quick switcher if it is not already there), run the ``/wordle``
  slash command in the message composer, then click the activity's OWN "Play"
  button on the splash screen behind it.
* **Act 2 — play**: you type a word at the prompt, the agent enters it into
  the game and reports what happened. Repeat until you stop or the guess
  budget runs out.

You are sitting in front of the board and can read it perfectly well, so this
demo does not try to read it for you. There is no solver and no screenshot
interpretation — the agent is a pair of hands, and the only LLM involved is
the one driving the tool calls. If you later want it to choose words too,
that is a separate layer on top, not a change to this one.

Both acts share a single ``conversation_id``, so the guess loop inherits what
Act 1 learned about the board instead of re-orienting from cold on every
guess. ``CUA_ACTS=1`` still runs the opening half alone, which is the useful
probe when something changes in Discord's UI.

Measured against this exact client (cua-driver 0.10.0, Windows, Discord.exe):

* Discord's element tree IS readable — but **the first ``get_window_state``
  returns a bare Document (element_count 1)**. Chromium only spins up its
  accessibility engine once something queries it, so the second snapshot is
  the real one. ``_wake_accessibility`` below spends that throwaway probe up
  front, so the agent never sees the empty tree.
* Launching the activity and starting the game are two separate gates: the
  activity opens on a splash screen carrying its own "Play" button. The
  activity iframe does expose its interior to UIA — the splash screen showed
  up as 7 elements under a ``Document 'Wordle'`` node, so the agent can
  address it rather than having to guess at pixels.
* The composer is ``role=Edit`` with a label like ``Message #jiuwenclaw-test``,
  and the window title has the form ``#channel | server - Discord``, which is
  how Act 1 checks it is on the right server.

SAFETY — this demo acts on a live server:

* Running ``/wordle`` and joining the activity are both visible to everyone in
  the server, and neither can be undone by this script.
* ``type_text`` is gated: ``/wordle`` and every guess pause for your y/n.
* The riskiest failure is the slash-command picker not opening — then Enter
  would post the literal text ``/wordle`` as a public message. Act 1 is told to
  verify the picker shows a Wordle entry BEFORE pressing Enter, and to stop if
  it does not.
* Automating a user account is a grey area under Discord's terms of service.
  You are running this against your own account at your own risk.

Prerequisites match ``cua_agent_demo.py`` (cua-driver daemon running,
``examples/cua/.env`` with an LLM endpoint), plus Discord already running and
logged in.

Run from repository root::

    uv run python examples/cua/discord_wordle_demo.py

Env toggles::

    CUA_ACTS=1                      # 1 = reach the board, 2 = play (default 1,2)
    DISCORD_SERVER=kwangyy's server  # server name, matched against the window title
    DISCORD_GUILD_ID=...            # server id, reported and used in the task text
    WORDLE_MAX_GUESSES=6            # guess budget for Act 2
    CUA_APPROVE=allow|deny          # auto-answer approval prompts
    CUA_MAX_ITERATIONS=30           # per-act iteration budget
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

CUA_EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CUA_EXAMPLE_DIR.parent.parent
ENV_FILE = CUA_EXAMPLE_DIR / ".env"

for path in (REPO_ROOT, CUA_EXAMPLE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

load_dotenv(REPO_ROOT / ".env")
load_dotenv(CUA_EXAMPLE_DIR.parent / ".env")
load_dotenv(ENV_FILE)

import os  # noqa: E402

from openjiuwen.core.session import InteractiveInput  # noqa: E402
from openjiuwen.core.single_agent.interrupt.response import ToolCallInterruptRequest  # noqa: E402
from openjiuwen.harness.subagents.cua_agent import (  # noqa: E402
    DEFAULT_CUA_AGENT_SYSTEM_PROMPT_EN,
    create_cua_agent,
)
from openjiuwen.harness.tools.cua.config import build_cua_driver_mcp_config  # noqa: E402

DISCORD_PROCESS_NAME = "Discord.exe"
WORDLE_WORD_LENGTH = 5

DISCORD_SYSTEM_PROMPT = DEFAULT_CUA_AGENT_SYSTEM_PROMPT_EN + (
    " IMPORTANT CORRECTIONS for this environment: Discord is an Electron app, so it IS a native "
    "desktop window and IS yours to drive — the 'browser tasks are not yours' rule does not apply "
    "to it. Screenshots embedded in MCP tool results are NOT visible to you: they arrive as text "
    "placeholders, so never read pixels off them and never claim to have 'seen' anything that way. "
    "Address every target by element_index from the latest get_window_state snapshot; this window "
    "may sit on a monitor with negative screen coordinates, where raw window-local x,y arithmetic "
    "is a reliable way to click the wrong thing. Elements whose frame is null are scrolled out of "
    "view and cannot be clicked — scroll them into view first or pick a different element."
)


def _env_str(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return default


def _build_model():
    from openjiuwen.core.foundation.llm import init_model

    api_key = _env_str("API_KEY", "LLM_API_KEY")
    if not api_key:
        print(
            f"Missing API key. Copy {CUA_EXAMPLE_DIR / '.env.template'} to {ENV_FILE} and fill in API_KEY.",
            file=sys.stderr,
        )
        sys.exit(1)
    return init_model(
        provider=_env_str("MODEL_PROVIDER", "LLM_PROVIDER", default="OpenAI"),
        model_name=_env_str("MODEL_NAME", "LLM_MODEL_NAME", default="gpt-4.1-mini"),
        api_key=api_key,
        api_base=_env_str("API_BASE", "LLM_API_BASE", default="https://api.openai.com/v1"),
        verify_ssl=_env_str("LLM_SSL_VERIFY", default="false").lower() in ("1", "true", "yes"),
    )


def _preflight_driver() -> str:
    """Fail fast with a clear hint when the cua-driver daemon is unreachable."""
    command = build_cua_driver_mcp_config().params["command"]
    try:
        probe = subprocess.run(
            [command, "call", "get_screen_size"],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
    except FileNotFoundError:
        print(
            "cua-driver binary not found. Install it first:\n  Windows: irm https://cua.ai/driver/install.ps1 | iex",
            file=sys.stderr,
        )
        sys.exit(1)
    except subprocess.TimeoutExpired:
        probe = None
    if probe is None or probe.returncode != 0:
        detail = (probe.stderr or probe.stdout).strip()[:300] if probe else "get_screen_size timed out"
        print(
            f"cua-driver daemon is not reachable. Start it with:\n  cua-driver autostart kick\nProbe output: {detail}",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"[preflight] cua-driver daemon reachable via {command!r}")
    return command


def _driver_call(driver_command: str, tool: str, args: dict | None = None, timeout: int = 60) -> dict | None:
    """Invoke one driver tool via the CLI and return its parsed JSON payload."""
    argv = [driver_command, "call", tool]
    if args:
        argv.append(json.dumps(args))
    try:
        # Decode as UTF-8 explicitly. text=True would use the Windows locale
        # codec (cp1252), which raises on the emoji in channel and window
        # titles — and the failure happens in subprocess's reader thread, so
        # stdout silently arrives as None instead of propagating the error.
        probe = subprocess.run(argv, capture_output=True, encoding="utf-8", errors="replace", timeout=timeout)
        if probe.returncode != 0:
            print(f"[driver] {tool} failed: {(probe.stderr or probe.stdout).strip()[:200]}")
            return None
        return json.loads(probe.stdout)
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        print(f"[driver] {tool} failed: {exc}")
        return None


def _find_discord_window(driver_command: str) -> dict:
    """Locate the Discord window. Deterministic lookup — no reason to spend an agent turn on it."""
    payload = _driver_call(driver_command, "list_windows") or {}
    candidates = [
        window
        for window in (payload.get("windows") or [])
        if str(window.get("app_name") or "") == DISCORD_PROCESS_NAME and not window.get("minimized")
    ]
    if not candidates:
        print(
            f"No open {DISCORD_PROCESS_NAME} window found. Start Discord, log in, and leave it "
            "unminimized, then re-run.",
            file=sys.stderr,
        )
        sys.exit(1)
    window = candidates[0]
    print(f"[discord] pid={window['pid']} window_id={window['window_id']} title={window.get('title')!r}")
    return window


def _wake_accessibility(driver_command: str, window: dict) -> int:
    """Trigger Chromium's lazy accessibility engine and report the woken tree size.

    Measured on Discord.exe: the first ``get_window_state`` returns a bare
    Document (element_count 1) because Chromium only builds its accessibility
    tree once a client asks for it. Spending this throwaway probe here means
    the agent's own first snapshot is already the populated one.
    """
    args = {
        "pid": window["pid"],
        "window_id": window["window_id"],
        "include_screenshot": False,
        "max_elements": 1,
    }
    # The first call is a pure trigger — bounded to one element, its own count
    # is meaningless. Only the second call reports the real tree size.
    _driver_call(driver_command, "get_window_state", args)
    args["max_elements"] = 4000
    second = _driver_call(driver_command, "get_window_state", args) or {}
    woken = int(second.get("element_count") or 0)
    print(f"[a11y] tree awake: {woken} elements")
    if woken <= 1:
        print(
            "[a11y] WARNING: Discord's element tree is still empty. Act 1 cannot address the "
            "button. Try restarting Discord, or launch it with --force-renderer-accessibility.",
            file=sys.stderr,
        )
    return woken


# --- Your guess ----------------------------------------------------------


def _prompt_for_guess(attempt: int, budget: int, tried: list[str]) -> str | None:
    """Ask for the word to play. You read the board; the agent just types.

    No solver and no board reading: you are sitting in front of the game and
    can see it perfectly well. The only thing worth automating is the typing.
    """
    if tried:
        print(f"[guess] played so far: {', '.join(word.upper() for word in tried)}")
    while True:
        try:
            answer = input(f"[guess] word {attempt}/{budget} (Enter to stop): ").strip().lower()
        except EOFError:
            return None
        if not answer:
            return None
        if len(answer) != WORDLE_WORD_LENGTH or not answer.isalpha():
            print(f"[guess]   {answer!r} is not a {WORDLE_WORD_LENGTH}-letter word — try again.")
            continue
        if answer in tried:
            print(f"[guess]   {answer.upper()} was already played — try again.")
            continue
        return answer


# --- Approvals ------------------------------------------------------------


def _permissions_config() -> dict:
    """Gate every typed guess: text lands in a live, public server."""
    return {
        "enabled": True,
        "schema": "tiered_policy",
        "permission_mode": _env_str("CUA_PERMISSION_MODE", default="normal"),
        "tools": {},
        "defaults": {"*": "allow"},
        "rules": [
            {
                "id": "gate_discord_typing",
                "description": "typed text reaches a live Discord server — confirm every guess",
                "tools": ["mcp_cua-driver_type_text"],
                "match_type": "os_control",
                "pattern": "*",
                "action": "ask",
            },
        ],
        "approval_overrides": [],
    }


def _decide_approval(tool_name: str, tool_args) -> bool:
    forced = _env_str("CUA_APPROVE").lower()
    short = tool_name.split("_", 2)[-1] if tool_name.startswith("mcp_") else tool_name
    detail = ""
    if isinstance(tool_args, dict) and "text" in tool_args:
        detail = f"  text={tool_args['text']!r}"
    print(f"[approval] the agent wants to run  {short}{detail}", flush=True)
    if forced in ("allow", "approve", "yes", "y"):
        print("[approval]   -> auto-APPROVED (CUA_APPROVE)", flush=True)
        return True
    if forced in ("deny", "reject", "no", "n"):
        print("[approval]   -> auto-DENIED (CUA_APPROVE)", flush=True)
        return False
    try:
        answer = input("[approval]   approve this action? [y/N] ").strip().lower()
    except EOFError:
        print("[approval]   -> no console input; DENYING (safe default)", flush=True)
        return False
    approved = answer in ("y", "yes")
    print(f"[approval]   -> {'APPROVED' if approved else 'DENIED'}", flush=True)
    return approved


async def _run_with_approvals(runner, agent, task: str, conversation_id: str) -> dict:
    """Run the agent, answering each permission interrupt until it finishes."""
    inputs = {"query": task, "conversation_id": conversation_id}
    while True:
        result = await runner.run_agent(agent, inputs)
        if not isinstance(result, dict) or result.get("result_type") != "interrupt":
            return result if isinstance(result, dict) else {"result_type": "unknown", "output": result}
        interactive = InteractiveInput()
        for item in result.get("state") or []:
            req = getattr(getattr(item, "payload", None), "value", None)
            if not isinstance(req, ToolCallInterruptRequest):
                continue
            approved = _decide_approval(req.tool_name, req.tool_args)
            interactive.update(req.tool_call_id, {"approved": approved, "feedback": "", "auto_confirm": False})
        inputs = {"query": interactive, "conversation_id": conversation_id}


# --- Acts -----------------------------------------------------------------


def _briefing(window: dict, server: str, guild_id: str) -> str:
    """Facts measured against this machine, so the agent does not rediscover them."""
    return (
        "ENVIRONMENT BRIEFING (verified facts — trust these over guesses):\n"
        f"- Target window: pid {window['pid']}, window_id {window['window_id']}, currently titled "
        f"{window.get('title')!r}. Do not go hunting for a different window.\n"
        f"- Target: the Discord server named {server!r} (id {guild_id}). The window title has the "
        "form '#channel | server - Discord', so the server name is in it.\n"
        "- The message composer is the element with role Edit whose label starts with 'Message #'. "
        "Anything typed there becomes a PUBLIC message the moment Enter is pressed without a "
        "command picker open.\n"
        "- Discord's accessibility tree has already been woken up for you, so your first "
        "get_window_state on this window returns the full tree (~800 elements). Bound it with "
        "max_elements=4000 and use query= to filter the markdown.\n"
        "- Screenshots in tool results are NOT visible to you — they are text placeholders. Work "
        "from the element tree only, and never claim to have seen the screen.\n"
        "- This window may be on a monitor with NEGATIVE screen coordinates. Always act by "
        "element_index, never by computing raw x,y.\n"
        "- Elements with a null frame are scrolled out of view and cannot be clicked.\n\n"
    )


def _act1_task(server: str) -> str:
    return (
        f"Start a Wordle game in the {server!r} Discord server by running the /wordle slash "
        "command. Steps:\n"
        f"1. Snapshot the target window. If the title does not already contain {server!r}, open "
        f"the quick switcher with the ctrl+k hotkey, type {server!r}, press Enter, then "
        "re-snapshot and confirm the title changed.\n"
        "2. Click the message composer (role Edit, label starting with 'Message #') to put "
        "keyboard focus in it.\n"
        "3. Type the text '/wordle' into the composer. Do NOT press Enter yet.\n"
        "4. Re-snapshot and look for the slash-command picker that Discord pops up while you type "
        "a command — it appears as a list/menu containing a Wordle command entry. THIS CHECK IS "
        "MANDATORY: if you cannot find a Wordle entry in the picker, the app is not available in "
        "this server. In that case STOP immediately, report that, and DO NOT press Enter — "
        "pressing Enter with no picker open would post the literal text '/wordle' as a public "
        "message in the server.\n"
        "5. Only once you have confirmed the Wordle entry exists: press Enter to select the "
        "highlighted command, re-snapshot to confirm the composer now holds a command (not plain "
        "text), then press Enter again to run it.\n"
        "6. Re-snapshot. The activity opens as a Document labelled 'Wordle' (its value is a "
        "discordsays.com URL) near the END of the tree. It lands on a splash screen, NOT on the "
        "board — so find the Button labelled 'Play' inside that Document and click it too. "
        "Launching the activity and starting the game are two separate gates.\n"
        "7. Re-snapshot once more and describe the game surface: list any elements that look like "
        "letter tiles, a grid, an on-screen keyboard, or an 'Enter'/'Backspace' control, with "
        "their element_index values. State clearly whether keyboard focus is now inside the game "
        "or still in the message composer."
    )


def _act2_task(guess: str, attempt: int, budget: int) -> str:
    return (
        f"Submit Wordle guess {attempt} of {budget}. The word to enter is: {guess.upper()}\n"
        "The board is already open from earlier in this conversation — reuse what you already "
        "know about it rather than re-exploring the window.\n"
        "Steps:\n"
        "1. Take ONE fresh snapshot (element indices from earlier snapshots are stale) and "
        "confirm keyboard focus is NOT in the channel's message box (role Edit, label starting "
        "with 'Message #'). If it is, click the game surface first to move focus into it. This "
        "matters: typing into the message box would post the guess publicly to the server.\n"
        f"2. Type the text '{guess.lower()}'. Leave delivery_mode at its default 'background' — "
        "only escalate that one action to 'foreground' if the driver actually reports background "
        "delivery is unavailable.\n"
        "3. Press the Enter key to submit.\n"
        "4. Re-snapshot and report whether the guess was accepted (for example the row filled in, "
        "or an error such as 'not in word list' appeared).\n"
        "Type the word exactly once. If the guess is rejected as an invalid word, say so and stop "
        "instead of retrying with a different word."
    )


_ACTS: dict[int, str] = {
    1: "reach the board — run /wordle, then click through the splash",
    2: "play — you name each word, the agent types it",
}


def _selected_acts() -> list[int]:
    raw = _env_str("CUA_ACTS", default="1,2")
    acts = [int(part) for part in (p.strip() for p in raw.split(",")) if part in ("1", "2")]
    if not acts:
        print(f"CUA_ACTS={raw!r} selects no valid act (valid: 1,2).", file=sys.stderr)
        sys.exit(1)
    return sorted(set(acts))


async def _play(runner, agent, conversation_id: str, prologue: str = "") -> None:
    """Act 2: you name a word, the agent types it into the game. Repeat.

    Every guess continues ``conversation_id`` rather than starting a fresh one,
    so the agent keeps what Act 1 established — which element is the game
    surface, where focus sits — instead of re-orienting from cold six times.
    ``prologue`` carries the environment briefing, and is only needed for the
    first message: when Act 1 already ran, the briefing is in the history.
    Context does not grow without bound, because the cua agent's
    ToolResultWindowProcessor keeps only the newest snapshot and offloads the
    rest to the workspace.
    """
    budget = int(_env_str("WORDLE_MAX_GUESSES", default="6"))
    tried: list[str] = []

    for attempt in range(1, budget + 1):
        guess = _prompt_for_guess(attempt, budget, tried)
        if guess is None:
            print(f"[act2] stopping. Words played: {', '.join(tried).upper() or '(none)'}")
            return
        tried.append(guess)

        print(f"\n=== GUESS {attempt}/{budget}: {guess.upper()} ===")
        result = await _run_with_approvals(
            runner, agent, prologue + _act2_task(guess, attempt, budget), conversation_id
        )
        prologue = ""
        print("output:\n", result.get("output", result))

    print(f"[act2] guess budget ({budget}) exhausted. Words played: {', '.join(tried).upper()}")


async def main() -> None:
    driver_command = _preflight_driver()
    acts = _selected_acts()
    server = _env_str("DISCORD_SERVER", default="kwangyy's server")
    guild_id = _env_str("DISCORD_GUILD_ID", default="1268967006888263863")

    window = _find_discord_window(driver_command)
    _wake_accessibility(driver_command, window)
    briefing = _briefing(window, server, guild_id)

    print(f"[target] server {server!r} (id {guild_id})")
    print("[perms] typing is gated — /wordle and every guess pause for your y/n")
    print(f"[acts]  running: {', '.join(f'{n} ({_ACTS[n]})' for n in acts)}")
    print("[note]  running /wordle and joining the activity are both visible to the server.\n")

    model = _build_model()
    agent = create_cua_agent(
        model,
        system_prompt=DISCORD_SYSTEM_PROMPT,
        language=_env_str("CUA_LANGUAGE", default="en"),
        max_iterations=int(_env_str("CUA_MAX_ITERATIONS", default="30")),
        permissions=_permissions_config(),
    )

    from openjiuwen.core.runner import Runner

    await Runner.start()
    try:
        await agent.ensure_initialized()
        # One conversation spans both acts: the guess loop inherits everything
        # Act 1 established about the board instead of re-discovering it.
        conversation_id = f"wordle-{uuid.uuid4().hex[:8]}"
        opened = False
        if 1 in acts:
            print(f"=== ACT 1: {_ACTS[1]} ===")
            result = await _run_with_approvals(Runner, agent, briefing + _act1_task(server), conversation_id)
            print("\n--- ACT 1 RESULT ---")
            print("result_type:", result.get("result_type"))
            print("output:\n", result.get("output", result))
            opened = True
        if 2 in acts:
            print(f"\n=== ACT 2: {_ACTS[2]} ===")
            await _play(Runner, agent, conversation_id, prologue="" if opened else briefing)
    finally:
        await Runner.stop()

    print("\n=== DEMO COMPLETE ===")


if __name__ == "__main__":
    # Channel names and agent output contain emoji; force UTF-8 so printing them
    # does not crash on a Windows cp1252 console.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(main())

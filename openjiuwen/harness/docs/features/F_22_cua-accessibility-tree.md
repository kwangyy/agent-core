# cua accessibility-tree track

## Metadata

| Item | Value |
| --- | --- |
| Date | 2026-10-28 |
| Scope | `tools/cua/accessibility.py` (UIA / AT-SPI backends, aliases, tree pruning); four new cua tools in `desktop_tools.py`; the `accessibility` capability; `os_control` category and the `set_value` ask rule; `CuaProgressRail` current window; cua prompt suffix; capture-result windowing; `uiautomation` in the `cua` extra |
| Specs | S_05, S_08, S_18 |
| Test baseline | 502 passed, 1 skipped across the targeted suites listed under Verification (2026-10-28) |
| Refs | #1043 |

## Background

F_21 made perception screenshot-only and listed accessibility trees as a
rejected alternative. Screenshot pointing is the main source of wasted steps
(F_21 Limits: 60 of 100 probe answers within 40 px). Most office and admin
apps (Win32, WinForms, WPF, UWP, Office, GTK, Qt, LibreOffice) expose a usable
tree, and acting through the tree needs no pointing at all. This feature adds
the tree as a second, opt-in track next to the screenshot tools, without
changing them.

## Decision and state

### Two tracks, no shared identifiers

- New tools: `list_accessible_windows`, `get_accessibility_tree`, `invoke`,
  `set_value`. They are the counterparts of `list_windows` and
  `get_window_state` plus actions, and use their own identifiers: window
  aliases `aw1, aw2, …` and element ids `e1, e2, …`.
- Nothing maps an alias to a `list_windows` window_id. On Linux an AT-SPI
  frame carries no X11 window id (at-spi2-core issue #21), so any mapping is a
  pid/title/bounds guess that fails on same-titled windows of one process.
  Keeping the tracks separate removes the guess; the prompt tells the model
  never to mix the identifiers.
- The window key behind an alias is the HWND on Windows (UIA
  `NativeWindowHandle`) and the AT-SPI frame object on Linux (its D-Bus bus
  name and object path). The pid in a row comes from the provider (UIA
  `ProcessId`; AT-SPI `get_process_id`, which the D-Bus daemon takes from the
  socket's peer credentials), not from matching.

### Aliases and element ids

- `AccessibilityIndex` (one per `CuaDesktop`, so one per agent) hands out
  aliases and ids. Numbers are never reused within a run, so a stale alias or
  id fails instead of naming something else.
- Each tree read of a window retires that window's previous element ids. A
  retired id gets "from an older tree of awN; read again", and its native
  handle is dropped. Ids are not carried across reads by fingerprint: every
  action is expected to follow a fresh read, as OSWorld does with its marks.
- Before acting, the tool re-reads the element and refuses when its role or
  name changed since the read (a recreated or reordered control), or when it
  no longer exists.

### Tree output

- One indented line per element: `[e7] push-button "Save" = "value" [enabled,
  focused] actions: click, set_value`. No bounds: the track never clicks
  pixels, so bounds would only invite mixing the tracks.
- Pruning: elements reported hidden (UIA `IsOffscreen`, AT-SPI not
  `SHOWING`) are dropped with their subtree; unnamed layout containers
  (`pane`, `group`, `panel`, `filler`, …) with no value and no action are
  skipped and their children lifted. Password fields never show a value.
- Bounds per reply: depth 25 by default (60 max), 1500 nodes read and 400
  shown; the reply counts what it left out and points to `expand=<id>`, which
  reads one element's subtree.

### Actions

- `invoke(element, action?)` runs a provider action: UIA `Invoke` / `Toggle`
  / `SelectionItem.Select` / `ExpandCollapse`, or an AT-SPI `doAction` by
  name. The tree lists each element's actions; model verbs (`click`,
  `toggle`, `select`, `expand`, `collapse`) map onto provider names
  (`press`, `activate`, `expand or contract`, …). Without `action` the first
  listed action runs.
- `set_value(element, text)` writes through UIA `ValuePattern` or AT-SPI
  `EditableText` (numbers through `Value`), then reads the value back and
  returns ❌ when the app did not keep it, telling the model to type into
  the field with the screenshot tools. Password fields are written but not
  read back.
- Neither action moves the mouse, sends keys, or needs the window in front;
  both work on covered windows. Some legacy Win32 controls behind the MSAA
  proxy still implement `Invoke` with window messages, so focus can move.

### Backends

- `create_accessibility_backend()` builds `_UiaImpl` (`uiautomation`) on
  Windows and `_AtspiImpl` (`pyatspi`) on Linux, imported on first use. A
  missing library becomes a ❌ with install hints on each call, and the build
  is retried on the next call rather than cached as a failure.
- Every backend call runs on one dedicated thread (`_ThreadedBackend`): UIA
  COM objects belong to the thread that created them (COM is initialised in
  that thread), and the AT-SPI GLib bindings are not driven from many
  threads. A call that does not return within 15 s (frozen app, modal dialog
  opened by an invoke) becomes a ❌; the stuck call still holds the thread.

### Wiring

- Capability `accessibility` holds the four tools. It is not in the default
  grant: on Linux it needs system packages pip cannot install, and the
  default prompt does not teach it. When granted, a prompt suffix (cn / en)
  explains the aliases, ids, actions and when to fall back to screenshots.
- `invoke` and `set_value` join the `os_control` category (14 tools). A new
  cua rule asks for `set_value`, like `type_text`, since it can write
  passwords; `invoke` follows the caller's defaults, like `click`.
- `CuaProgressRail` treats `get_accessibility_tree` as a snapshot and records
  `{"accessible_window": alias}` as the current window;
  `list_accessible_windows` is read-only for it. `ToolResultWindowProcessor`
  windows `get_accessibility_tree` results with the captures.
- `cua` extra: `uiautomation` on Windows (pulls `comtypes`). Linux needs
  `python3-pyatspi` (or PyGObject with `gir1.2-atspi-2.0`), `at-spi2-core`
  and a session with the accessibility bus.

## Rejected alternatives

- **One joined window list (match AT-SPI frames to X11 windows).** Needed
  only when tree elements are clicked by pixels. The match is heuristic on
  Linux and fails on same-titled windows of one process; keeping the tracks
  apart avoids it.
- **Clicking tree elements by their bounds.** Brings back window activation,
  coordinate origins and the matching above. The model switches to the
  screenshot tools instead.
- **Stable ids across reads by element fingerprint.** More code and an
  ambiguous-match failure mode, for runs that read the tree after every
  action anyway.
- **`set_value` escalating to paste or real typing.** That is real input,
  which needs the foreground window and therefore the window match. The
  failure reply points to `type_text` instead.
- **Default-on.** Linux needs system packages, and the change in default
  behaviour needs evaluation runs first.

## Verification

- Unit tests: `tests/unit_tests/harness/tools/cua/test_accessibility.py`
  (fake accessibility backend: aliases never reused, pruning, password
  masking, node cap, empty tree, `expand`, stale and changed ids,
  `invoke` / `set_value` including read-back failure, backend build failure
  and retry, thread and timeout, verb mapping, capability and category),
  plus `test_desktop_tools.py`, `test_cua_rails.py`, `test_cua_agent.py`
  (prompt suffix, `set_value` ask), `tests/unit_tests/harness/security/` and
  `tests/unit_tests/harness/tools/test_task_tool.py`, `test_subagent_rail.py` and `tests/unit_tests/harness/prompts/`. Result: 502 passed,
  1 skipped.
- Not yet verified against live UIA or AT-SPI: the platform backends ran only
  through fakes.

## Limits

- Live runs on Windows and Linux are pending; the UIA and AT-SPI code paths
  follow the libraries' documented APIs but have not been exercised.
- Chromium / Electron apps build their tree lazily and may show an empty tree
  until an assistive technology or `--force-renderer-accessibility` turns it
  on; canvas apps and remote-desktop windows have no useful tree.
- `set_value` bypasses keystroke handlers: masked fields and fields that
  commit on key events reject it (reported as ❌).
- A timed-out call keeps the backend thread busy until the app answers.

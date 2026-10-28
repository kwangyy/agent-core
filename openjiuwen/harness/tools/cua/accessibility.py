# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Accessibility-tree perception and semantic actions for the cua tools.

A second track next to the screenshot tools, deliberately not linked to them:
windows here are the accessibility API's own top-level windows (UI Automation
on Windows, AT-SPI frames on Linux), named by run-scoped aliases (``aw1``),
and elements by run-scoped ids (``e7``). Nothing maps an alias to a
``list_windows`` window_id -- on Linux there is no reliable link between an
AT-SPI frame and an X11 window -- so the two tracks share no identifiers.

Actions go through the app's accessibility provider (UIA patterns, AT-SPI
Action / EditableText), not through the mouse or keyboard: they need no
foreground window and work on covered windows.

The platform libraries (``uiautomation``, ``pyatspi``) are imported when a
backend is built, and every backend call runs on one dedicated thread: UIA COM
objects belong to the thread that created them, and AT-SPI's GLib bindings are
not safe to drive from many threads.
"""

from __future__ import annotations

import concurrent.futures
import itertools
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, TypeVar

from openjiuwen.harness.tools.cua.backend import _process_name

_T = TypeVar("_T")

# Bounds for one get_accessibility_tree reply. A large app (Office, VS Code)
# exposes thousands of nodes; past the cap the reply says how many were left out.
DEFAULT_TREE_DEPTH = 25
MAX_TREE_DEPTH = 60
MAX_TREE_NODES = 400
# Nodes read from the provider per call. Higher than the shown cap because
# pruning drops hidden nodes and layout scaffolding before counting.
MAX_READ_NODES = 1500
# Per-node text shown in the tree; longer values are cut with an ellipsis.
_MAX_TEXT_CHARS = 120

# A hung provider (frozen app, modal dialog opened by an invoke) blocks the
# call forever; the tool gives up after this and says so.
_CALL_TIMEOUT_S = 15.0

# States worth showing the model, in display order.
_SHOWN_STATES = ("enabled", "disabled", "focused", "checked", "selected", "expanded", "collapsed", "editable")

# Unnamed containers with nothing to act on are skipped and their children
# lifted, so the model reads controls rather than layout scaffolding.
_STRUCTURAL_ROLES = frozenset(
    {"filler", "group", "pane", "panel", "section", "custom", "unknown", "redundant-object", "grouping"}
)

# Model-facing action verbs and the provider action names each one accepts.
_ACTION_ALIASES: dict[str, tuple[str, ...]] = {
    "click": ("click", "press", "activate", "invoke", "jump"),
    "toggle": ("toggle",),
    "select": ("select",),
    "expand": ("expand", "expand or contract"),
    "collapse": ("collapse", "expand or contract"),
}


class AccessibilityError(Exception):
    """An accessibility call that cannot be carried out; its message becomes the ❌ reply."""


@dataclass(frozen=True)
class AxWindow:
    """One top-level accessible window.

    ``key`` identifies the window for the backend's lifetime (the HWND on
    Windows, the frame object on Linux); ``ref`` is what the backend reads.
    """

    key: Any
    ref: Any
    pid: int
    app_name: str
    title: str
    active: bool
    minimized: bool


@dataclass
class AxNode:
    """One element as read from the provider, before pruning."""

    ref: Any
    role: str
    name: str
    value: str | None = None
    states: frozenset[str] = frozenset()
    actions: tuple[str, ...] = ()
    settable: bool = False
    password: bool = False
    children: list[AxNode] = field(default_factory=list)


class AccessibilityBackend(Protocol):
    """Platform accessibility API. Every method runs on the backend's own thread."""

    def list_windows(self) -> list[AxWindow]:
        """Top-level accessible windows, frontmost order not guaranteed."""

    def read_tree(self, ref: Any, max_depth: int, max_nodes: int) -> AxNode:
        """The element ``ref`` and its descendants, read to ``max_depth`` and about ``max_nodes`` nodes."""

    def describe(self, ref: Any) -> AxNode:
        """The element's current role, name, value and states, without children.

        Raises AccessibilityError when the element no longer exists.
        """

    def perform(self, ref: Any, action: str | None) -> str:
        """Run ``action`` (a model verb) or the element's default action; return the provider path used."""

    def set_value(self, ref: Any, text: str) -> str:
        """Write ``text`` as the element's value; return the provider path used."""


def _short(text: str | None) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= _MAX_TEXT_CHARS else text[: _MAX_TEXT_CHARS - 1] + "…"


def _quoted(text: str) -> str:
    return '"' + text.replace('"', '\\"') + '"'


def resolve_action(verb: str | None, available: tuple[str, ...]) -> str | None:
    """The provider action name for a model verb, or None when the element has none that fits."""
    if verb is None:
        return available[0] if available else None
    wanted = str(verb).strip().lower()
    accepted = _ACTION_ALIASES.get(wanted, (wanted,))
    for name in available:
        if name.lower() in accepted:
            return name
    return None


# ---------------------------------------------------------------------------
# Run-scoped aliases and tree rendering
# ---------------------------------------------------------------------------


@dataclass
class _ElementEntry:
    ref: Any
    window_alias: str
    tree_serial: int
    role: str
    name: str


@dataclass
class _WindowEntry:
    window: AxWindow
    tree_serial: int = 0


class AccessibilityIndex:
    """Run-scoped aliases for windows (``aw1``) and elements (``e1``).

    Numbers are never reused, so a stale alias fails instead of silently
    naming something else. Each tree read retires the element ids of that
    window's previous tree: elements must be addressed from the latest read.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._window_numbers = itertools.count(1)
        self._element_numbers = itertools.count(1)
        self._windows: dict[str, _WindowEntry] = {}
        self._window_alias_by_key: dict[Any, str] = {}
        self._elements: dict[str, _ElementEntry] = {}

    def register_windows(self, windows: list[AxWindow]) -> list[tuple[str, AxWindow]]:
        """Alias each window, keeping the alias a window already had."""
        rows: list[tuple[str, AxWindow]] = []
        with self._lock:
            for win in windows:
                alias = self._window_alias_by_key.get(win.key)
                if alias is None:
                    alias = f"aw{next(self._window_numbers)}"
                    self._window_alias_by_key[win.key] = alias
                    self._windows[alias] = _WindowEntry(win)
                else:
                    self._windows[alias].window = win
                rows.append((alias, win))
        return rows

    def window(self, alias: str) -> AxWindow:
        """The window behind ``alias``, as of the latest listing."""
        with self._lock:
            entry = self._windows.get(str(alias).strip())
        if entry is None:
            raise AccessibilityError(f"unknown window {alias!r}; call list_accessible_windows first.")
        return entry.window

    def element(self, element_id: str) -> _ElementEntry:
        """The element behind ``element_id``, refusing an id from an older tree of its window."""
        key = str(element_id).strip()
        with self._lock:
            entry = self._elements.get(key)
            if entry is None:
                raise AccessibilityError(
                    f"unknown element {element_id!r}; read the window with get_accessibility_tree first."
                )
            current = self._windows[entry.window_alias].tree_serial
        if entry.tree_serial != current:
            raise AccessibilityError(
                f"{key} is from an older tree of {entry.window_alias}; call "
                f"get_accessibility_tree(window={entry.window_alias!r}) again and use the new ids."
            )
        return entry

    def render_tree(self, window_alias: str, root: AxNode, max_nodes: int = MAX_TREE_NODES) -> tuple[str, int]:
        """Prune, number and render ``root``; return the text and how many nodes were left out."""
        with self._lock:
            entry = self._windows[window_alias]
            entry.tree_serial += 1
            serial = entry.tree_serial
            # Ids of this window's older trees stay known (so a stale id gets a
            # clear refusal) but drop their native handles, never reassigned.
            for old in self._elements.values():
                if old.window_alias == window_alias:
                    old.ref = None
            lines: list[str] = []
            omitted = self._render(root, 0, lines, window_alias, serial, max_nodes)
        return "\n".join(lines), omitted

    def _render(self, node: AxNode, depth: int, lines: list[str], alias: str, serial: int, max_nodes: int) -> int:
        if "hidden" in node.states:
            return 0
        if _is_scaffolding(node) and depth > 0:
            return sum(self._render(child, depth, lines, alias, serial, max_nodes) for child in node.children)
        if len(lines) >= max_nodes:
            return 1 + _count(node.children)
        element_id = f"e{next(self._element_numbers)}"
        self._elements[element_id] = _ElementEntry(node.ref, alias, serial, node.role, node.name)
        lines.append("  " * depth + _render_line(element_id, node))
        return sum(self._render(child, depth + 1, lines, alias, serial, max_nodes) for child in node.children)


def _is_scaffolding(node: AxNode) -> bool:
    return node.role in _STRUCTURAL_ROLES and not node.name and not node.value and not node.actions


def _count(nodes: list[AxNode]) -> int:
    return sum(1 + _count(n.children) for n in nodes)


def _render_line(element_id: str, node: AxNode) -> str:
    parts = [f"[{element_id}]", node.role or "unknown"]
    if node.name:
        parts.append(_quoted(_short(node.name)))
    if node.password:
        parts.append("= (password, hidden)")
    elif node.value:
        parts.append("= " + _quoted(_short(node.value)))
    states = [s for s in _SHOWN_STATES if s in node.states]
    if states:
        parts.append("[" + ", ".join(states) + "]")
    verbs = list(node.actions) + (["set_value"] if node.settable else [])
    if verbs:
        parts.append("actions: " + ", ".join(verbs))
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class _ThreadedBackend(AccessibilityBackend):
    """Runs every call of the wrapped backend on one dedicated thread, with a timeout."""

    def __init__(self, build: Callable[[], Any], init_thread: Callable[[], None] | None = None) -> None:
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="cua-a11y", initializer=init_thread
        )
        try:
            self._impl = self._run(build)
        except BaseException:
            self._executor.shutdown(wait=False)
            raise

    def _run(self, fn: Callable[..., _T], *args: Any) -> _T:
        future = self._executor.submit(fn, *args)
        try:
            return future.result(timeout=_CALL_TIMEOUT_S)
        except concurrent.futures.TimeoutError as exc:
            raise AccessibilityError(
                f"the accessibility call did not return within {_CALL_TIMEOUT_S:.0f}s; the app may be busy "
                "or showing a modal dialog. Read the window again before retrying."
            ) from exc

    def list_windows(self) -> list[AxWindow]:
        return self._run(self._impl.list_windows)

    def read_tree(self, ref: Any, max_depth: int, max_nodes: int) -> AxNode:
        return self._run(self._impl.read_tree, ref, max_depth, max_nodes)

    def describe(self, ref: Any) -> AxNode:
        return self._run(self._impl.describe, ref)

    def perform(self, ref: Any, action: str | None) -> str:
        return self._run(self._impl.perform, ref, action)

    def set_value(self, ref: Any, text: str) -> str:
        return self._run(self._impl.set_value, ref, text)


class _UiaImpl(AccessibilityBackend):
    """Windows UI Automation through the ``uiautomation`` package."""

    # Shell surfaces that are top-level UIA children but not app windows.
    _SKIPPED_CLASSES = frozenset({"Shell_TrayWnd", "Shell_SecondaryTrayWnd", "Progman", "WorkerW"})

    def __init__(self) -> None:
        # pylint: disable-next=import-error
        import uiautomation  # noqa: PLC0415 - lazy: Windows-only, needs COM

        self._auto = uiautomation

    def list_windows(self) -> list[AxWindow]:
        auto = self._auto
        foreground = auto.GetForegroundWindow()
        windows: list[AxWindow] = []
        for ctrl in auto.GetRootControl().GetChildren():
            try:
                hwnd, title, pid = int(ctrl.NativeWindowHandle or 0), ctrl.Name, int(ctrl.ProcessId)
                if not hwnd or not title or ctrl.ClassName in self._SKIPPED_CLASSES:
                    continue
                minimized = self._minimized(ctrl)
            except Exception:  # noqa: BLE001 - a window that closed mid-listing is skipped
                continue
            windows.append(
                AxWindow(
                    key=hwnd,
                    ref=ctrl,
                    pid=pid,
                    app_name=_process_name(pid),
                    title=title,
                    active=hwnd == foreground,
                    minimized=minimized,
                )
            )
        return windows

    def _minimized(self, ctrl: Any) -> bool:
        pattern = ctrl.GetPattern(self._auto.PatternId.WindowPattern)
        return pattern is not None and pattern.WindowVisualState == self._auto.WindowVisualState.Minimized

    def read_tree(self, ref: Any, max_depth: int, max_nodes: int) -> AxNode:
        budget = [max_nodes]
        return self._read(ref, max_depth, budget)

    def _read(self, ctrl: Any, depth_left: int, budget: list[int]) -> AxNode:
        node = self.describe(ctrl)
        budget[0] -= 1
        if depth_left > 0 and budget[0] > 0 and "hidden" not in node.states:
            for child in ctrl.GetChildren():
                if budget[0] <= 0:
                    break
                node.children.append(self._read(child, depth_left - 1, budget))
        return node

    def describe(self, ref: Any) -> AxNode:
        try:
            role = ref.ControlTypeName
        except Exception as exc:  # noqa: BLE001 - a dead element raises a COMError on any property
            raise AccessibilityError("the element no longer exists; read the window again.") from exc
        role = role[: -len("Control")] if role.endswith("Control") else role
        states: set[str] = {"enabled" if ref.IsEnabled else "disabled"}
        if ref.IsOffscreen:
            states.add("hidden")
        if ref.HasKeyboardFocus:
            states.add("focused")
        actions = self._pattern_actions(ref, states)
        value_pattern = ref.GetPattern(self._auto.PatternId.ValuePattern)
        settable = value_pattern is not None and not value_pattern.IsReadOnly
        if settable:
            states.add("editable")
        password = bool(ref.IsPassword)
        value = None if password or value_pattern is None else value_pattern.Value
        return AxNode(
            ref=ref,
            role=role.lower(),
            name=ref.Name or "",
            value=value or None,
            states=frozenset(states),
            actions=actions,
            settable=settable,
            password=password,
        )

    def _pattern_actions(self, ref: Any, states: set[str]) -> tuple[str, ...]:
        """The verbs the element's control patterns support; adds the states they report to ``states``."""
        auto, pid = self._auto, self._auto.PatternId
        actions: list[str] = []
        if ref.GetPattern(pid.InvokePattern) is not None:
            actions.append("click")
        toggle = ref.GetPattern(pid.TogglePattern)
        if toggle is not None:
            actions.append("toggle")
            if toggle.ToggleState == auto.ToggleState.On:
                states.add("checked")
        selection = ref.GetPattern(pid.SelectionItemPattern)
        if selection is not None:
            actions.append("select")
            if selection.IsSelected:
                states.add("selected")
        expand = ref.GetPattern(pid.ExpandCollapsePattern)
        if expand is not None and expand.ExpandCollapseState != auto.ExpandCollapseState.LeafNode:
            expanded = expand.ExpandCollapseState == auto.ExpandCollapseState.Expanded
            states.add("expanded" if expanded else "collapsed")
            actions.append("collapse" if expanded else "expand")
        return tuple(actions)

    def perform(self, ref: Any, action: str | None) -> str:
        node = self.describe(ref)
        name = resolve_action(action, node.actions)
        if name is None:
            raise AccessibilityError(_no_action_message(action, node.actions))
        pid = self._auto.PatternId
        if name == "click":
            ref.GetPattern(pid.InvokePattern).Invoke()
            return "InvokePattern"
        if name == "toggle":
            ref.GetPattern(pid.TogglePattern).Toggle()
            return "TogglePattern"
        if name == "select":
            ref.GetPattern(pid.SelectionItemPattern).Select()
            return "SelectionItemPattern"
        pattern = ref.GetPattern(pid.ExpandCollapsePattern)
        if name == "expand":
            pattern.Expand()
        else:
            pattern.Collapse()
        return "ExpandCollapsePattern"

    def set_value(self, ref: Any, text: str) -> str:
        pattern = ref.GetPattern(self._auto.PatternId.ValuePattern)
        if pattern is None or pattern.IsReadOnly:
            raise AccessibilityError("the element has no writable value (ValuePattern).")
        pattern.SetValue(text)
        return "ValuePattern"


def _init_com() -> None:
    import ctypes  # noqa: PLC0415

    ctypes.windll.ole32.CoInitialize(None)  # type: ignore[attr-defined]


class _AtspiImpl(AccessibilityBackend):
    """Linux AT-SPI2 through ``pyatspi``."""

    _WINDOW_ROLES = ("frame", "dialog", "window", "alert", "file chooser")

    def __init__(self) -> None:
        # pylint: disable-next=import-error
        import pyatspi  # noqa: PLC0415 - lazy: system package, needs the accessibility bus

        self._atspi = pyatspi

    def list_windows(self) -> list[AxWindow]:
        windows: list[AxWindow] = []
        for app in self._atspi.Registry.getDesktop(0):
            if app is None:
                continue
            try:
                pid = int(app.get_process_id())
            except Exception:  # noqa: BLE001 - an app that exited mid-listing is skipped
                continue
            for frame in app:
                if frame is None or frame.getRoleName() not in self._WINDOW_ROLES:
                    continue
                state = frame.getState()
                windows.append(
                    AxWindow(
                        # The wrapper is stable while the frame lives; underneath
                        # it is the frame's (bus name, object path).
                        key=frame,
                        ref=frame,
                        pid=pid,
                        app_name=app.name or _process_name(pid),
                        title=frame.name or "",
                        active=state.contains(self._atspi.STATE_ACTIVE),
                        minimized=state.contains(self._atspi.STATE_ICONIFIED),
                    )
                )
        return windows

    def read_tree(self, ref: Any, max_depth: int, max_nodes: int) -> AxNode:
        budget = [max_nodes]
        return self._read(ref, max_depth, budget)

    def _read(self, acc: Any, depth_left: int, budget: list[int]) -> AxNode:
        node = self.describe(acc)
        budget[0] -= 1
        if depth_left > 0 and budget[0] > 0 and "hidden" not in node.states:
            for child in acc:
                if budget[0] <= 0:
                    break
                if child is not None:
                    node.children.append(self._read(child, depth_left - 1, budget))
        return node

    def describe(self, ref: Any) -> AxNode:
        atspi = self._atspi
        try:
            state = ref.getState()
            role = ref.getRoleName()
        except Exception as exc:  # noqa: BLE001 - a dead object raises a D-Bus error on any call
            raise AccessibilityError("the element no longer exists; read the window again.") from exc
        if state.contains(atspi.STATE_DEFUNCT):
            raise AccessibilityError("the element no longer exists; read the window again.")
        states: set[str] = {"enabled" if state.contains(atspi.STATE_ENABLED) else "disabled"}
        if not state.contains(atspi.STATE_SHOWING):
            states.add("hidden")
        for flag, label in (
            (atspi.STATE_FOCUSED, "focused"),
            (atspi.STATE_CHECKED, "checked"),
            (atspi.STATE_SELECTED, "selected"),
            (atspi.STATE_EDITABLE, "editable"),
        ):
            if state.contains(flag):
                states.add(label)
        if state.contains(atspi.STATE_EXPANDABLE):
            states.add("expanded" if state.contains(atspi.STATE_EXPANDED) else "collapsed")
        password = role == "password text"
        return AxNode(
            ref=ref,
            role=role.replace(" ", "-"),
            name=ref.name or "",
            value=None if password else self._value(ref),
            states=frozenset(states),
            actions=self._actions(ref),
            settable=self._settable(ref, state),
            password=password,
        )

    @staticmethod
    def _actions(ref: Any) -> tuple[str, ...]:
        try:
            action = ref.queryAction()
        except NotImplementedError:
            return ()
        return tuple(action.getName(i) for i in range(action.nActions))

    def _settable(self, ref: Any, state: Any) -> bool:
        if state.contains(self._atspi.STATE_EDITABLE):
            try:
                ref.queryEditableText()
                return True
            except NotImplementedError:
                pass
        try:
            ref.queryValue()
            return True
        except NotImplementedError:
            return False

    @staticmethod
    def _value(ref: Any) -> str | None:
        try:
            value = ref.queryValue()
            return str(value.currentValue)
        except NotImplementedError:
            pass
        try:
            text = ref.queryText()
            return text.getText(0, min(text.characterCount, _MAX_TEXT_CHARS * 2)) or None
        except NotImplementedError:
            return None

    def perform(self, ref: Any, action: str | None) -> str:
        names = self._actions(ref)
        name = resolve_action(action, names)
        if name is None:
            raise AccessibilityError(_no_action_message(action, names))
        ref.queryAction().doAction(names.index(name))
        return f"AT-SPI action {name!r}"

    def set_value(self, ref: Any, text: str) -> str:
        try:
            ref.queryEditableText().setTextContents(text)
            return "AT-SPI EditableText"
        except NotImplementedError:
            pass
        try:
            number = float(text)
        except ValueError as exc:
            raise AccessibilityError("the element is neither editable text nor a numeric value.") from exc
        try:
            ref.queryValue().currentValue = number
        except NotImplementedError as exc:
            raise AccessibilityError("the element is neither editable text nor a numeric value.") from exc
        return "AT-SPI Value"


def _no_action_message(action: str | None, available: tuple[str, ...]) -> str:
    if not available:
        return "the element has no accessibility action; use the screenshot tools for it."
    if action is None:
        return f"the element has no default action; available: {', '.join(available)}."
    return f"the element does not support {action!r}; available: {', '.join(available)}."


_INSTALL_HINTS = {
    "win32": "pip install 'openjiuwen[cua]' (installs uiautomation)",
    "linux": (
        "install the system packages python3-pyatspi (or python3-gi with gir1.2-atspi-2.0) and at-spi2-core, "
        "make them importable from this Python, and run inside a session with the accessibility bus"
    ),
}


def create_accessibility_backend() -> AccessibilityBackend:
    """Build the accessibility backend for this host, failing with an install hint where none can run."""
    if sys.platform == "win32":
        impl: Callable[[], Any] = _UiaImpl
        init: Callable[[], None] | None = _init_com
        hint = _INSTALL_HINTS["win32"]
    elif sys.platform.startswith("linux"):
        impl, init, hint = _AtspiImpl, None, _INSTALL_HINTS["linux"]
    else:
        raise AccessibilityError(f"no accessibility backend for platform {sys.platform!r}.")
    try:
        return _ThreadedBackend(impl, init)
    except ImportError as exc:
        raise AccessibilityError(f"accessibility support is not installed ({exc}); {hint}.") from exc


__all__ = [
    "DEFAULT_TREE_DEPTH",
    "MAX_TREE_DEPTH",
    "MAX_READ_NODES",
    "MAX_TREE_NODES",
    "AccessibilityBackend",
    "AccessibilityError",
    "AccessibilityIndex",
    "AxNode",
    "AxWindow",
    "create_accessibility_backend",
    "resolve_action",
]

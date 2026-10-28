# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Tests for the accessibility-tree tools against a fake accessibility backend."""

from __future__ import annotations

import sys
import threading

import pytest

from openjiuwen.harness.security.tiered_policy import _tool_category
from openjiuwen.harness.tools.cua import accessibility as accessibility_module
from openjiuwen.harness.tools.cua.accessibility import (
    AccessibilityError,
    AccessibilityIndex,
    AxNode,
    AxWindow,
    create_accessibility_backend,
    resolve_action,
)
from openjiuwen.harness.tools.cua.cua_capabilities import (
    ACCESSIBILITY_CUA_TOOL_NAMES,
    DEFAULT_CUA_AGENT_CAPABILITY_NAMES,
    resolve_cua_capabilities,
)
from openjiuwen.harness.tools.cua.desktop_tools import CuaDesktop, build_cua_desktop_tools


def _window(key: str, pid: int = 10, title: str | None = None, **flags) -> AxWindow:
    return AxWindow(
        key=key,
        ref=f"ref-{key}",
        pid=pid,
        app_name="gedit",
        title=title or f"{key}.txt - gedit",
        active=flags.get("active", False),
        minimized=flags.get("minimized", False),
    )


class _Element:
    """A provider element: what describe() reads, and what actions change."""

    def __init__(self, role: str, name: str = "", *, value=None, actions=(), settable=False, **kw) -> None:
        self.role, self.name, self.value = role, name, value
        self.actions, self.settable = tuple(actions), settable
        self.states = set(kw.get("states", {"enabled"}))
        self.password = kw.get("password", False)
        self.children: list[_Element] = list(kw.get("children", []))
        self.keeps_value = kw.get("keeps_value", True)
        self.alive = True


class _FakeAx:
    def __init__(self, windows: list[AxWindow], trees: dict[str, _Element]) -> None:
        self.windows = list(windows)
        self.trees = trees
        self.calls: list[tuple] = []

    def list_windows(self) -> list[AxWindow]:
        return list(self.windows)

    def _node(self, el: _Element) -> AxNode:
        return AxNode(
            ref=el,
            role=el.role,
            name=el.name,
            value=el.value,
            states=frozenset(el.states),
            actions=el.actions,
            settable=el.settable,
            password=el.password,
        )

    def read_tree(self, ref, max_depth, max_nodes) -> AxNode:
        root = self.trees[ref] if isinstance(ref, str) else ref

        def read(el: _Element, depth: int) -> AxNode:
            node = self._node(el)
            if depth < max_depth:
                node.children = [read(child, depth + 1) for child in el.children]
            return node

        return read(root, 0)

    def describe(self, ref) -> AxNode:
        if not ref.alive:
            raise AccessibilityError("the element no longer exists; read the window again.")
        return self._node(ref)

    def perform(self, ref, action) -> str:
        name = resolve_action(action, ref.actions)
        if name is None:
            raise AccessibilityError(f"the element does not support {action!r}; available: {', '.join(ref.actions)}.")
        self.calls.append(("perform", ref.name, name))
        return f"fake {name}"

    def set_value(self, ref, text) -> str:
        self.calls.append(("set_value", ref.name, text))
        if ref.keeps_value:
            ref.value = text
        return "fake value"


def _gedit() -> _FakeAx:
    save = _Element("push-button", "Save", actions=("click",))
    body = _Element("text", "", value="hello", states={"enabled", "editable", "focused"}, settable=True)
    # Unnamed layout panels around the controls, and a hidden menu.
    layout = _Element("panel", children=[_Element("filler", children=[save]), body])
    hidden = _Element("menu", "Recent", states={"enabled", "hidden"}, children=[_Element("menu-item", "a.txt")])
    frame = _Element("frame", "todo.txt - gedit", children=[layout, hidden])
    other = _Element("frame", "notes.txt - gedit", children=[_Element("push-button", "Open", actions=("click",))])
    return _FakeAx([_window("notes", active=True), _window("todo")], {"ref-todo": frame, "ref-notes": other})


def _tools(ax: _FakeAx) -> dict:
    desktop = CuaDesktop(backend_factory=lambda: None, accessibility_factory=lambda: ax)
    return {tool.card.name: tool for tool in build_cua_desktop_tools(desktop)}


async def _ok(tools: dict, name: str, **args) -> str:
    result = await tools[name].invoke(args)
    assert result.success, result.error
    return result.data["content"]


async def _fail(tools: dict, name: str, **args) -> str:
    result = await tools[name].invoke(args)
    assert not result.success
    assert result.error.startswith("❌")
    return result.error


def _line(tree: str, needle: str) -> str:
    return next(line for line in tree.splitlines() if needle in line)


def _id(line: str) -> str:
    return line.strip().split("]")[0].lstrip("[")


# ── list_accessible_windows ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_windows_get_stable_aliases_that_are_never_reused() -> None:
    ax = _gedit()
    tools = _tools(ax)

    first = await _ok(tools, "list_accessible_windows")
    assert '[aw1] pid 10 gedit "notes.txt - gedit"  active' in first
    assert '[aw2] pid 10 gedit "todo.txt - gedit"' in first
    assert "not for list_windows" in first

    # A window that closes keeps its number retired; a new one gets a fresh alias.
    ax.windows = [_window("todo"), _window("draft")]
    second = await _ok(tools, "list_accessible_windows")
    assert "[aw2] pid 10" in second and '[aw3] pid 10 gedit "draft.txt - gedit"' in second
    assert "aw1" not in second


@pytest.mark.asyncio
async def test_pid_filter_and_minimized_flag() -> None:
    ax = _FakeAx([_window("a", pid=1), _window("b", pid=2, minimized=True)], {})
    listing = await _ok(_tools(ax), "list_accessible_windows", pid=2)

    assert "pid 1 " not in listing
    assert listing.rstrip().endswith("minimized")


# ── get_accessibility_tree ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tree_lifts_layout_scaffolding_and_drops_hidden_elements() -> None:
    tools = _tools(_gedit())
    await _ok(tools, "list_accessible_windows")

    tree = await _ok(tools, "get_accessibility_tree", window="aw2")

    assert 'Accessibility tree of aw2 (pid 10, "todo.txt - gedit")' in tree
    assert "panel" not in tree and "filler" not in tree
    assert "Recent" not in tree and "a.txt" not in tree
    save = _line(tree, '"Save"')
    assert save.startswith("  [e") and save.endswith('push-button "Save" [enabled] actions: click')
    body = _line(tree, "hello")
    assert body.endswith('text = "hello" [enabled, focused, editable] actions: set_value')


@pytest.mark.asyncio
async def test_password_values_are_never_shown() -> None:
    secret = _Element("password-text", "Password", value="hunter2", settable=True, password=True)
    ax = _FakeAx([_window("login")], {"ref-login": _Element("dialog", "Login", children=[secret])})
    tools = _tools(ax)
    await _ok(tools, "list_accessible_windows")

    tree = await _ok(tools, "get_accessibility_tree", window="aw1")

    assert "hunter2" not in tree
    assert '"Password" = (password, hidden)' in tree


@pytest.mark.asyncio
async def test_tree_cap_reports_left_out_nodes(monkeypatch) -> None:
    buttons = [_Element("push-button", f"b{i}", actions=("click",)) for i in range(10)]
    ax = _FakeAx([_window("big")], {"ref-big": _Element("frame", "Big", children=buttons)})
    tools = _tools(ax)
    await _ok(tools, "list_accessible_windows")
    original = AccessibilityIndex.render_tree
    monkeypatch.setattr(AccessibilityIndex, "render_tree", lambda self, a, r: original(self, a, r, max_nodes=4))

    tree = await _ok(tools, "get_accessibility_tree", window="aw1")

    assert '"b2"' in tree and '"b3"' not in tree
    assert "7 more node(s) not shown; pass expand=<element id>" in tree


@pytest.mark.asyncio
async def test_empty_tree_points_to_the_screenshot_tools() -> None:
    ax = _FakeAx([_window("canvas")], {"ref-canvas": _Element("frame", "", states={"enabled", "hidden"})})
    tools = _tools(ax)
    await _ok(tools, "list_accessible_windows")

    tree = await _ok(tools, "get_accessibility_tree", window="aw1")

    assert "no visible elements; use the screenshot tools" in tree


@pytest.mark.asyncio
async def test_expand_reads_one_subtree_of_the_same_window_only() -> None:
    inner = _Element("list-item", "deep", actions=("select",))
    tree_root = _Element("frame", "F", children=[_Element("list", "Files", children=[inner])])
    ax = _FakeAx([_window("f"), _window("g")], {"ref-f": tree_root, "ref-g": _Element("frame", "G")})
    tools = _tools(ax)
    await _ok(tools, "list_accessible_windows")
    files = _id(_line(await _ok(tools, "get_accessibility_tree", window="aw1", depth=1), '"Files"'))

    subtree = await _ok(tools, "get_accessibility_tree", window="aw1", expand=files)

    assert f"from {files}" in subtree and '"deep"' in subtree and '"F"' not in subtree
    files_again = _id(_line(subtree, '"Files"'))
    assert "belongs to aw1, not aw2" in await _fail(tools, "get_accessibility_tree", window="aw2", expand=files_again)


@pytest.mark.asyncio
async def test_tree_argument_errors() -> None:
    tools = _tools(_gedit())
    await _ok(tools, "list_accessible_windows")

    assert "unknown window 'aw9'" in await _fail(tools, "get_accessibility_tree", window="aw9")
    assert "Missing required string field window" in await _fail(tools, "get_accessibility_tree")
    assert "depth must be between 1 and" in await _fail(tools, "get_accessibility_tree", window="aw1", depth=0)


# ── invoke / set_value ───────────────────────────────────────────────────


async def _todo_tree(tools: dict) -> str:
    await _ok(tools, "list_accessible_windows")
    return await _ok(tools, "get_accessibility_tree", window="aw2")


@pytest.mark.asyncio
async def test_invoke_runs_the_default_or_named_action() -> None:
    ax = _gedit()
    tools = _tools(ax)
    save = _id(_line(await _todo_tree(tools), '"Save"'))

    reply = await _ok(tools, "invoke", element=save)

    assert reply.startswith(f"✅ Invoked push-button 'Save' [{save}] via fake click.")
    assert "get_accessibility_tree" in reply
    assert ax.calls == [("perform", "Save", "click")]
    assert "available: click" in await _fail(tools, "invoke", element=save, action="toggle")


@pytest.mark.asyncio
async def test_ids_from_an_older_tree_are_refused() -> None:
    tools = _tools(_gedit())
    old_save = _id(_line(await _todo_tree(tools), '"Save"'))

    new_save = _id(_line(await _ok(tools, "get_accessibility_tree", window="aw2"), '"Save"'))

    assert new_save != old_save
    error = await _fail(tools, "invoke", element=old_save)
    assert f"{old_save} is from an older tree of aw2" in error
    assert "unknown element 'e999'" in await _fail(tools, "invoke", element="e999")


@pytest.mark.asyncio
async def test_an_element_that_changed_identity_or_vanished_is_not_acted_on() -> None:
    ax = _gedit()
    tools = _tools(ax)
    tree = await _todo_tree(tools)
    save_el = ax.trees["ref-todo"].children[0].children[0].children[0]
    save = _id(_line(tree, '"Save"'))

    save_el.name = "Delete"
    assert "now reads as push-button 'Delete'" in await _fail(tools, "invoke", element=save)
    save_el.alive = False
    assert "no longer exists" in await _fail(tools, "invoke", element=save)
    assert ax.calls == []


@pytest.mark.asyncio
async def test_set_value_reads_the_value_back() -> None:
    ax = _gedit()
    tools = _tools(ax)
    body = _id(_line(await _todo_tree(tools), "hello"))

    reply = await _ok(tools, "set_value", element=body, text="new text")

    assert "read back the same value" in reply
    assert ax.calls == [("set_value", "", "new text")]


@pytest.mark.asyncio
async def test_set_value_fails_when_the_app_does_not_keep_the_value() -> None:
    ax = _gedit()
    tools = _tools(ax)
    tree = await _todo_tree(tools)
    ax.trees["ref-todo"].children[0].children[1].keeps_value = False
    body = _id(_line(tree, "hello"))

    error = await _fail(tools, "set_value", element=body, text="new text")

    assert "now reads 'hello': the app did not keep the value" in error
    assert "type_text" in error


@pytest.mark.asyncio
async def test_set_value_on_a_password_field_is_not_read_back() -> None:
    secret = _Element("password-text", "Password", settable=True, password=True, keeps_value=False)
    ax = _FakeAx([_window("login")], {"ref-login": _Element("dialog", "Login", children=[secret])})
    tools = _tools(ax)
    await _ok(tools, "list_accessible_windows")
    field = _id(_line(await _ok(tools, "get_accessibility_tree", window="aw1"), '"Password"'))

    reply = await _ok(tools, "set_value", element=field, text="s3cret")

    assert "cannot be read back" in reply and "s3cret" not in reply


@pytest.mark.asyncio
async def test_set_value_accepts_an_equal_number_and_rejects_a_missing_text() -> None:
    spin = _Element("spin-button", "Size", value="12", settable=True)
    ax = _FakeAx([_window("prefs")], {"ref-prefs": _Element("dialog", "Prefs", children=[spin])})
    tools = _tools(ax)
    await _ok(tools, "list_accessible_windows")
    size = _id(_line(await _ok(tools, "get_accessibility_tree", window="aw1"), '"Size"'))
    original = ax.set_value

    def set_number(ref, text):
        original(ref, text)
        ref.value = str(float(text))  # providers report numbers as floats
        return "fake value"

    ax.set_value = set_number

    assert "read back the same value" in await _ok(tools, "set_value", element=size, text="14")
    assert "Missing required string field text" in await _fail(tools, "set_value", element=size)


# ── backend construction ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_missing_backend_fails_each_call_with_its_hint_and_is_retried() -> None:
    attempts: list[int] = []

    def factory():
        attempts.append(1)
        raise AccessibilityError("accessibility support is not installed (No module named 'pyatspi'); install …")

    tools = {
        t.card.name: t
        for t in build_cua_desktop_tools(CuaDesktop(backend_factory=lambda: None, accessibility_factory=factory))
    }

    assert "not installed" in await _fail(tools, "list_accessible_windows")
    assert "not installed" in await _fail(tools, "list_accessible_windows")
    assert len(attempts) == 2


def test_create_backend_turns_a_missing_library_into_an_install_hint(monkeypatch) -> None:
    def missing() -> None:
        raise ImportError("No module named 'pyatspi'")

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(accessibility_module, "_AtspiImpl", missing)

    with pytest.raises(AccessibilityError, match="No module named 'pyatspi'.*python3-pyatspi"):
        create_accessibility_backend()

    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(AccessibilityError, match="no accessibility backend for platform 'darwin'"):
        create_accessibility_backend()


def test_backend_calls_run_on_one_thread_and_time_out(monkeypatch) -> None:
    release = threading.Event()

    class _Impl:
        def list_windows(self):
            return [threading.current_thread().name]

        def describe(self, ref):
            release.wait(5)
            return ref

    backend = accessibility_module._ThreadedBackend(_Impl)
    first, second = backend.list_windows(), backend.list_windows()
    assert first == second and first[0].startswith("cua-a11y")

    monkeypatch.setattr(accessibility_module, "_CALL_TIMEOUT_S", 0.05)
    with pytest.raises(AccessibilityError, match="did not return within"):
        backend.describe("x")
    release.set()


@pytest.mark.parametrize(
    ("verb", "available", "expected"),
    [
        (None, ("press", "activate"), "press"),
        ("click", ("Press",), "Press"),
        ("expand", ("activate", "expand or contract"), "expand or contract"),
        ("toggle", ("click",), None),
        (None, (), None),
    ],
)
def test_resolve_action_maps_model_verbs_onto_provider_names(verb, available, expected) -> None:
    assert resolve_action(verb, available) == expected


# ── capability and permission wiring ─────────────────────────────────────


def test_accessibility_is_an_opt_in_capability() -> None:
    assert "accessibility" not in DEFAULT_CUA_AGENT_CAPABILITY_NAMES
    assert not set(ACCESSIBILITY_CUA_TOOL_NAMES) & set(resolve_cua_capabilities(None).allowed_tool_names)

    granted = resolve_cua_capabilities(["accessibility"]).allowed_tool_names

    assert set(ACCESSIBILITY_CUA_TOOL_NAMES) <= set(granted)


def test_accessibility_actions_are_os_control_and_reads_are_not() -> None:
    assert _tool_category("invoke") == "os_control"
    assert _tool_category("set_value") == "os_control"
    assert _tool_category("get_accessibility_tree") is None
    assert _tool_category("list_accessible_windows") is None

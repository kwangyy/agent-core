import sys

import pytest

from openjiuwen.core.context_engine import ContextEngine
from openjiuwen.core.runner.resources_manager.tool_manager import ToolMgr

_BROWSER_TOOLS_MODULE = "openjiuwen.harness.tools.browser_move.playwright_runtime.browser_tools"


@pytest.fixture(autouse=True)
def _restore_mcp_client_factory():
    """Keep browser_move's MCP client patch isolated to the test that installs it.

    ``ensure_browser_runtime_client_patch`` rebinds ``ToolMgr._create_client`` for the
    whole process and latches itself, so any test that starts a browser runtime makes
    every later ``add_mcp_server`` build browser_move client subclasses instead of the
    core ones.
    """
    create_client = ToolMgr.__dict__["_create_client"]
    try:
        yield
    finally:
        if ToolMgr.__dict__["_create_client"] is not create_client:
            ToolMgr._create_client = create_client
            browser_tools = sys.modules.get(_BROWSER_TOOLS_MODULE)
            if browser_tools is not None:
                browser_tools._OPENJIUWEN_CLIENTS_PATCHED = False


@pytest.fixture(autouse=True)
def _restore_context_processor_registry():
    """Keep context processor overrides isolated to the test that activates them."""
    processor_map = dict(ContextEngine._PROCESSOR_MAP)
    try:
        yield
    finally:
        ContextEngine._PROCESSOR_MAP.clear()
        ContextEngine._PROCESSOR_MAP.update(processor_map)

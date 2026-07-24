# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import json
import uuid
from typing import Any, AsyncIterator, Dict, Optional

from pydantic import Field, BaseModel

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import build_error
from openjiuwen.core.common.utils.schema_utils import SchemaUtils
from openjiuwen.core.foundation.tool.base import Tool, ToolCard, Input, Output
from openjiuwen.core.foundation.tool.schema import McpToolInfo
from openjiuwen.core.runner.callback import trigger
from openjiuwen.core.runner.callback.events import ToolCallEvents

NO_TIMEOUT = -1

# Cap for the serialized structuredContent appended to a tool result. Bounds
# the context cost for tools whose structured payload is large (e.g. a full
# UIA element tree), while keeping small payloads (window bounds, screen
# size) fully visible to the model.
STRUCTURED_CONTENT_MAX_CHARS = 4000

_STRUCTURED_CONTENT_TAG = "[structured_content]"


def mcp_model_tool_prefix(server_name: str) -> str:
    """Prefix AbilityManager puts on a server's model-facing tool names."""
    return f"mcp_{server_name}_"


def mcp_model_tool_name(server_name: str, tool_name: str) -> str:
    """Model-facing name AbilityManager registers for an MCP tool."""
    return f"{mcp_model_tool_prefix(server_name)}{tool_name}"


def _serialize_structured_content(tool_result: Any) -> Optional[str]:
    """Serialize CallToolResult.structuredContent to compact JSON, or None."""
    structured = getattr(tool_result, "structuredContent", None)
    if structured is None:
        structured = getattr(tool_result, "structured_content", None)
    if not structured:
        return None
    if hasattr(structured, "model_dump"):
        structured = structured.model_dump(exclude_none=True)
    try:
        return json.dumps(structured, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(structured)


def _structured_duplicates_text(serialized: str, text: str) -> bool:
    """True when the structured payload carries no information beyond the text.

    Servers that auto-derive structuredContent from the text result (e.g.
    FastMCP's ``{"result": <text>}`` wrapper) would otherwise double every
    tool result in context.
    """
    if serialized == text or serialized in text:
        return True
    try:
        wrapper = json.loads(serialized)
    except (TypeError, ValueError):
        return False
    return isinstance(wrapper, dict) and set(wrapper) == {"result"} and str(wrapper.get("result")) == text


def _truncate_structured(serialized: str) -> str:
    if len(serialized) <= STRUCTURED_CONTENT_MAX_CHARS:
        return serialized
    return serialized[:STRUCTURED_CONTENT_MAX_CHARS] + "...(structured content truncated)"


def extract_mcp_tool_result_content(
    tool_result: Any,
    *,
    include_structured_content: bool = False,
    include_image_content: bool = False,
    tool_name: str = "",
) -> Any:
    """Return a compact value from an MCP CallToolResult.

    All content blocks are preserved: text blocks (and image placeholders)
    are joined in order, so multi-block results (e.g. cua-driver's element
    tree text plus screenshot) no longer lose everything but the last block.

    ``structuredContent`` is the canonical machine-readable half of an MCP
    result. It is dropped by default (historical behavior, and most servers
    mirror the text there anyway). Servers whose structured payload carries
    data the text summary lacks (e.g. cua-driver window bounds) opt in via
    ``McpServerConfig.include_structured_content``; their structured JSON is
    then appended, deduplicated against the text and size-capped.

    Image blocks become text placeholders by default. Servers whose images
    the model must see opt in via ``McpServerConfig.include_image_content``;
    the result is then an ``McpToolResult`` whose ``data`` holds the text
    plus data-URL image items, which the multimodal tool-result pipeline
    delivers to the model.
    """
    serialized_structured = _serialize_structured_content(tool_result) if include_structured_content else None
    content = getattr(tool_result, "content", None)
    if not content:
        if serialized_structured is not None:
            return f"{_STRUCTURED_CONTENT_TAG} {_truncate_structured(serialized_structured)}"
        return None

    text_parts = []
    images = []
    for item in content:
        text = getattr(item, "text", None)
        if text is not None:
            text_parts.append(text)
            continue

        mime_type = getattr(item, "mimeType", None) or getattr(item, "mime_type", None)
        data = getattr(item, "data", None)
        if data is not None:
            if mime_type and str(mime_type).startswith("image/"):
                if include_image_content:
                    images.append((str(mime_type), data))
                else:
                    text_parts.append(f"[image content: {mime_type}, {len(str(data))} base64 chars]")
                continue
            if len(content) == 1:
                return data
            text_parts.append(str(data))
            continue

        if hasattr(item, "model_dump"):
            dumped = item.model_dump(exclude_none=True)
            dumped.pop("data", None)
            if len(content) == 1:
                return dumped
            text_parts.append(str(dumped))
            continue
        if len(content) == 1:
            return str(item)
        text_parts.append(str(item))

    text = "\n\n".join(text_parts)
    if serialized_structured is not None and not _structured_duplicates_text(serialized_structured, text):
        structured_suffix = f"{_STRUCTURED_CONTENT_TAG} {_truncate_structured(serialized_structured)}"
        text = f"{text}\n\n{structured_suffix}" if text else structured_suffix

    if images:
        note = f"{len(images)} image(s) attached as multimodal input."
        return McpToolResult(
            data={
                "content": f"{text}\n\n{note}" if text else note,
                "multimodal": [
                    {
                        "type": "image",
                        "source": "mcp",
                        "source_path": tool_name or "mcp_tool",
                        "mime_type": mime_type,
                        "data_url": f"data:{mime_type};base64,{data}",
                    }
                    for mime_type, data in images
                ],
            }
        )
    return text


class McpServerConfig(BaseModel):
    server_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    server_name: str
    server_path: str
    client_type: str = 'sse'
    params: Dict[str, Any] = Field(default_factory=dict)
    auth_headers: dict = Field(default_factory=dict)
    auth_query_params: Dict[str, str] = Field(default_factory=dict)
    include_structured_content: bool = Field(
        default=False,
        description=(
            "Append each tool result's structuredContent (deduplicated, size-capped) to the extracted text. "
            "Opt-in for servers whose structured payload carries data the text summary lacks."
        ),
    )
    include_image_content: bool = Field(
        default=False,
        description=(
            "Bridge image content blocks into multimodal tool-result data instead of a text placeholder. "
            "Opt-in for servers whose images the model must see (e.g. cua-driver screenshots)."
        ),
    )


class McpToolResult(BaseModel):
    """Tool result carrying multimodal data from an MCP server.

    Duck-type-compatible with the harness ``ToolOutput`` shape
    (``success`` / ``data`` / ``error``) so the react-agent multimodal
    pipeline and tool-message building consume it without core importing
    harness.
    """

    success: bool = True
    data: Dict[str, Any]
    error: Optional[str] = None


class McpToolCard(ToolCard):
    server_name: str
    server_id: str = ''

    def tool_info(self):
        return McpToolInfo(name=self.name, description=self.description, parameters=self.input_params,
                           server_name=self.server_name)


class MCPTool(Tool):
    """MCP Tool class that wraps MCP server tools for LLM modules"""

    def __init__(self, mcp_client: Any, tool_info: McpToolCard):  # McpClient or its subclasses
        """
        Initialize MCP Tool

        Args:
            mcp_client: Instance of McpToolClient or its subclasses
            tool_name: Name of the MCP tool
            server_name: Name of the MCP server (for logging and identification)
        """
        super().__init__(tool_info)
        if mcp_client is None:
            raise build_error(StatusCode.TOOL_MCP_CLIENT_NOT_SUPPORTED, card=self._card)
        self._mcp_client = mcp_client

    async def stream(self, inputs: Input, **kwargs) -> AsyncIterator[Output]:
        raise build_error(StatusCode.TOOL_STREAM_NOT_SUPPORTED, card=self._card)

    async def invoke(self, inputs: Input, **kwargs) -> Output:
        try:
            # Prepare arguments for MCP tool call
            arguments = inputs if isinstance(inputs, dict) else {}
            if self._card.input_params is not None:
                await trigger(
                    ToolCallEvents.TOOL_PARSE_STARTED,
                    tool_name=self.card.name, tool_id=self.card.id,
                    raw_inputs=inputs, schema=self._card.input_params)
                skip_none_value = kwargs.get("skip_none_value", True)
                arguments = SchemaUtils.format_with_schema(inputs, self._card.input_params,
                                                           skip_none_value=False,
                                                           skip_validate=kwargs.get("skip_inputs_validate", False))
                if skip_none_value:
                    arguments = SchemaUtils.remove_none_values(arguments) or {}
                await trigger(
                    ToolCallEvents.TOOL_PARSE_FINISHED,
                    tool_name=self.card.name, tool_id=self.card.id,
                    formatted_inputs=arguments)

            result = await self._mcp_client.call_tool(tool_name=self._card.name, arguments=arguments)
            if isinstance(result, McpToolResult):
                return result
            return {"result": result}

        except Exception as e:
            raise build_error(StatusCode.TOOL_MCP_EXECUTION_ERROR, cause=e, reason=str(e), method="invoke",
                              card=self._card)

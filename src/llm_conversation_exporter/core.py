"""Export LLM conversations to markdown, HTML, plain text, and JSON.

Example::

    from llm_conversation_exporter import ConversationExporter, ExportFormat

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
    ]

    exporter = ConversationExporter(messages)
    print(exporter.to_markdown())
    print(exporter.to_text())
    print(exporter.to_html())
    print(exporter.to_json())

Standalone::

    from llm_conversation_exporter.core import export_markdown, export_text
    md = export_markdown(messages)
    txt = export_text(messages)
"""

from __future__ import annotations

import html
import json
from enum import Enum
from typing import Any

Message = dict[str, Any]

_ROLE_LABELS: dict[str, str] = {
    "system": "System",
    "user": "User",
    "assistant": "Assistant",
    "tool": "Tool",
}


class ExportFormat(str, Enum):
    """Supported export formats."""

    MARKDOWN = "markdown"
    HTML = "html"
    TEXT = "text"
    JSON = "json"


def _content_as_text(content: Any) -> str:
    """Extract plain text from content (string or list of blocks)."""
    if content is None:
        # OpenAI assistant messages carrying only tool_calls have
        # ``content: null``; render that as empty rather than "None".
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "tool_use":
                name = block.get("name", "unknown")
                args = block.get("input", {})
                parts.append(f"[tool_use: {name}({json.dumps(args)})]")
            elif btype == "tool_result":
                result_content = block.get("content", "")
                inner = _content_as_text(result_content)
                is_error = block.get("is_error", False)
                prefix = "[tool_result error]" if is_error else "[tool_result]"
                parts.append(f"{prefix} {inner}")
            elif btype == "image":
                parts.append("[image]")
            elif btype == "document":
                parts.append("[document]")
            else:
                # Unknown block type — try 'text' field
                fallback = block.get("text", "")
                if fallback:
                    parts.append(fallback)
        return "\n".join(parts)
    return str(content)


def _role_label(role: str) -> str:
    return _ROLE_LABELS.get(role, role.capitalize())


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def export_markdown(
    messages: list[Message],
    *,
    title: str = "",
    include_system: bool = True,
) -> str:
    """Export messages to a Markdown string.

    Args:
        messages:       Conversation messages.
        title:          Optional document title (rendered as h1).
        include_system: If ``False``, system messages are omitted.

    Returns:
        Markdown-formatted string.
    """
    lines: list[str] = []
    if title:
        lines.append(f"# {title}\n")
    for msg in messages:
        role = msg.get("role", "unknown")
        if role == "system" and not include_system:
            continue
        label = _role_label(role)
        content = _content_as_text(msg.get("content", ""))
        lines.append(f"**{label}**\n\n{content}\n")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------


def export_text(
    messages: list[Message],
    *,
    include_system: bool = True,
    separator: str = "-" * 40,
) -> str:
    """Export messages to plain text.

    Args:
        messages:       Conversation messages.
        include_system: If ``False``, system messages are omitted.
        separator:      Separator line between messages.

    Returns:
        Plain-text string.
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        if role == "system" and not include_system:
            continue
        label = _role_label(role)
        content = _content_as_text(msg.get("content", ""))
        parts.append(f"{label}:\n{content}")
    return ("\n" + separator + "\n").join(parts).rstrip() + "\n"


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
body {{ font-family: sans-serif; max-width: 800px; margin: 2rem auto;
  padding: 0 1rem; }}
.message {{ margin-bottom: 1.5rem; padding: 1rem; border-radius: 6px; }}
.system {{ background: #f0f0f0; }}
.user {{ background: #e8f4fd; }}
.assistant {{ background: #f0faf0; }}
.tool {{ background: #fff8e1; }}
.role-label {{ font-weight: bold; font-size: 0.9rem; margin-bottom: 0.5rem; }}
.content {{ white-space: pre-wrap; }}
</style>
</head>
<body>
{heading}{messages}
</body>
</html>
"""


def export_html(
    messages: list[Message],
    *,
    title: str = "Conversation",
    include_system: bool = True,
) -> str:
    """Export messages to an HTML string.

    Args:
        messages:       Conversation messages.
        title:          Page title and h1 heading.
        include_system: If ``False``, system messages are omitted.

    Returns:
        Full HTML document as a string.
    """
    msg_parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        if role == "system" and not include_system:
            continue
        label = _role_label(role)
        content = _content_as_text(msg.get("content", ""))
        safe_content = html.escape(content)
        css_class = role if role in _ROLE_LABELS else "assistant"
        msg_parts.append(
            f'<div class="message {css_class}">'
            f'<div class="role-label">{html.escape(label)}</div>'
            f'<div class="content">{safe_content}</div>'
            f"</div>"
        )
    heading = f"<h1>{html.escape(title)}</h1>\n" if title else ""
    return _HTML_TEMPLATE.format(
        title=html.escape(title),
        heading=heading,
        messages="\n".join(msg_parts),
    )


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def export_json(
    messages: list[Message],
    *,
    metadata: dict[str, Any] | None = None,
    indent: int | None = 2,
) -> str:
    """Export messages to a JSON string.

    Args:
        messages:  Conversation messages.
        metadata:  Optional top-level metadata dict.
        indent:    JSON indent level (``None`` for compact).

    Returns:
        JSON string.
    """
    payload: dict[str, Any] = {"messages": list(messages)}
    if metadata:
        payload["metadata"] = dict(metadata)
    return json.dumps(payload, indent=indent, ensure_ascii=False)


# ---------------------------------------------------------------------------
# ConversationExporter class
# ---------------------------------------------------------------------------


class ConversationExporter:
    """Export a conversation to multiple formats.

    Example::

        exporter = ConversationExporter(messages)
        print(exporter.to_markdown())
        print(exporter.to_html(title="My Chat"))
        print(exporter.to_text(include_system=False))
        print(exporter.to_json())
    """

    def __init__(
        self,
        messages: list[Message],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._messages = list(messages)
        self._metadata: dict[str, Any] = dict(metadata) if metadata else {}

    @property
    def messages(self) -> list[Message]:
        """The conversation messages."""
        return list(self._messages)

    @property
    def metadata(self) -> dict[str, Any]:
        """Optional metadata."""
        return dict(self._metadata)

    def message_count(self) -> int:
        """Number of messages."""
        return len(self._messages)

    def roles(self) -> list[str]:
        """Ordered list of roles in the conversation."""
        return [m.get("role", "") for m in self._messages]

    def to_markdown(
        self,
        *,
        title: str = "",
        include_system: bool = True,
    ) -> str:
        """Export to Markdown."""
        return export_markdown(
            self._messages,
            title=title,
            include_system=include_system,
        )

    def to_text(
        self,
        *,
        include_system: bool = True,
        separator: str = "-" * 40,
    ) -> str:
        """Export to plain text."""
        return export_text(
            self._messages,
            include_system=include_system,
            separator=separator,
        )

    def to_html(
        self,
        *,
        title: str = "Conversation",
        include_system: bool = True,
    ) -> str:
        """Export to HTML."""
        return export_html(
            self._messages,
            title=title,
            include_system=include_system,
        )

    def to_json(
        self,
        *,
        include_metadata: bool = True,
        indent: int | None = 2,
    ) -> str:
        """Export to JSON."""
        meta = self._metadata if include_metadata and self._metadata else None
        return export_json(self._messages, metadata=meta, indent=indent)

    def export(self, fmt: ExportFormat, **kwargs: Any) -> str:
        """Export using a :class:`ExportFormat` enum value.

        Args:
            fmt:    Target format.
            **kwargs: Passed to the underlying export function.

        Returns:
            Formatted string.

        Raises:
            ValueError: If the format is not recognised.
        """
        if fmt == ExportFormat.MARKDOWN:
            return self.to_markdown(**kwargs)
        if fmt == ExportFormat.TEXT:
            return self.to_text(**kwargs)
        if fmt == ExportFormat.HTML:
            return self.to_html(**kwargs)
        if fmt == ExportFormat.JSON:
            return self.to_json(**kwargs)
        raise ValueError(f"Unknown export format: {fmt!r}")

    def __repr__(self) -> str:
        return f"ConversationExporter(messages={self.message_count()})"

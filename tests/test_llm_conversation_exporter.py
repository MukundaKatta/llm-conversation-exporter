"""Tests for llm_conversation_exporter."""

from __future__ import annotations

import json

import pytest

from llm_conversation_exporter import ConversationExporter, ExportFormat
from llm_conversation_exporter.core import (
    export_html,
    export_json,
    export_markdown,
    export_text,
)

SIMPLE = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
]


# ---------------------------------------------------------------------------
# _content_as_text (via public API)
# ---------------------------------------------------------------------------


def test_markdown_plain_string_content():
    msgs = [{"role": "user", "content": "plain text"}]
    md = export_markdown(msgs)
    assert "plain text" in md


def test_markdown_list_content_text_block():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "block text"}]}]
    md = export_markdown(msgs)
    assert "block text" in md


def test_markdown_list_content_tool_use_block():
    msgs = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "name": "search", "input": {"q": "hello"}}
            ],
        }
    ]
    md = export_markdown(msgs)
    assert "tool_use: search" in md
    assert "hello" in md


def test_markdown_list_content_tool_result_block():
    msgs = [
        {
            "role": "user",
            "content": [{"type": "tool_result", "content": "42", "is_error": False}],
        }
    ]
    md = export_markdown(msgs)
    assert "[tool_result]" in md
    assert "42" in md


def test_markdown_tool_result_error():
    msgs = [
        {
            "role": "user",
            "content": [{"type": "tool_result", "content": "oops", "is_error": True}],
        }
    ]
    md = export_markdown(msgs)
    assert "[tool_result error]" in md


def test_markdown_image_block():
    msgs = [{"role": "user", "content": [{"type": "image", "source": {"url": "..."}}]}]
    md = export_markdown(msgs)
    assert "[image]" in md


def test_markdown_unknown_block_with_text():
    msgs = [{"role": "user", "content": [{"type": "custom", "text": "fallback"}]}]
    md = export_markdown(msgs)
    assert "fallback" in md


def test_markdown_unknown_block_no_text():
    msgs = [{"role": "user", "content": [{"type": "custom", "other": "x"}]}]
    md = export_markdown(msgs)
    # Should not crash; content section may be empty
    assert "User" in md


def test_markdown_tool_result_nested_list_content():
    # Anthropic tool_result blocks may carry a list of content blocks.
    msgs = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "content": [{"type": "text", "text": "nested ok"}],
                }
            ],
        }
    ]
    md = export_markdown(msgs)
    assert "[tool_result]" in md
    assert "nested ok" in md


def test_markdown_document_block():
    msgs = [{"role": "user", "content": [{"type": "document", "source": {}}]}]
    md = export_markdown(msgs)
    assert "[document]" in md


def test_markdown_multiple_blocks_joined():
    msgs = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"},
            ],
        }
    ]
    md = export_markdown(msgs)
    assert "first" in md
    assert "second" in md


def test_markdown_non_dict_block_skipped():
    msgs = [{"role": "user", "content": ["raw string", {"type": "text", "text": "ok"}]}]
    md = export_markdown(msgs)
    assert "ok" in md


def test_markdown_none_content_renders_empty():
    # OpenAI assistant messages with only tool_calls carry content=None.
    msgs = [{"role": "assistant", "content": None}]
    md = export_markdown(msgs)
    assert "**Assistant**" in md
    assert "None" not in md


def test_text_none_content_renders_empty():
    msgs = [{"role": "assistant", "content": None}]
    txt = export_text(msgs)
    assert "Assistant:" in txt
    assert "None" not in txt


def test_html_none_content_renders_empty():
    msgs = [{"role": "assistant", "content": None}]
    h = export_html(msgs)
    assert 'class="message assistant"' in h
    assert ">None<" not in h


# ---------------------------------------------------------------------------
# export_markdown
# ---------------------------------------------------------------------------


def test_markdown_contains_role_labels():
    md = export_markdown(SIMPLE)
    assert "**System**" in md
    assert "**User**" in md
    assert "**Assistant**" in md


def test_markdown_contains_content():
    md = export_markdown(SIMPLE)
    assert "You are helpful." in md
    assert "Hello" in md
    assert "Hi there!" in md


def test_markdown_with_title():
    md = export_markdown(SIMPLE, title="My Chat")
    assert md.startswith("# My Chat")


def test_markdown_no_title():
    md = export_markdown(SIMPLE, title="")
    assert not md.startswith("#")


def test_markdown_exclude_system():
    md = export_markdown(SIMPLE, include_system=False)
    assert "System" not in md
    assert "You are helpful." not in md
    assert "User" in md


def test_markdown_include_system_default():
    md = export_markdown(SIMPLE)
    assert "System" in md


def test_markdown_empty_messages():
    md = export_markdown([])
    assert md.strip() == ""


def test_markdown_unknown_role():
    msgs = [{"role": "tool", "content": "result"}]
    md = export_markdown(msgs)
    assert "Tool" in md


def test_markdown_ends_with_newline():
    md = export_markdown(SIMPLE)
    assert md.endswith("\n")


# ---------------------------------------------------------------------------
# export_text
# ---------------------------------------------------------------------------


def test_text_contains_role_labels():
    txt = export_text(SIMPLE)
    assert "System:" in txt
    assert "User:" in txt
    assert "Assistant:" in txt


def test_text_contains_content():
    txt = export_text(SIMPLE)
    assert "You are helpful." in txt
    assert "Hello" in txt
    assert "Hi there!" in txt


def test_text_exclude_system():
    txt = export_text(SIMPLE, include_system=False)
    assert "System" not in txt
    assert "You are helpful." not in txt


def test_text_custom_separator():
    txt = export_text(SIMPLE, separator="===")
    assert "===" in txt


def test_text_default_separator():
    txt = export_text(SIMPLE)
    assert "---" in txt


def test_text_empty_messages():
    txt = export_text([])
    assert txt.strip() == ""


def test_text_ends_with_newline():
    txt = export_text(SIMPLE)
    assert txt.endswith("\n")


def test_text_single_message():
    msgs = [{"role": "user", "content": "only one"}]
    txt = export_text(msgs)
    assert "only one" in txt
    assert "User:" in txt


# ---------------------------------------------------------------------------
# export_html
# ---------------------------------------------------------------------------


def test_html_is_valid_structure():
    h = export_html(SIMPLE)
    assert "<!DOCTYPE html>" in h
    assert "<html" in h
    assert "</html>" in h
    assert "<body>" in h
    assert "</body>" in h


def test_html_contains_role_content():
    h = export_html(SIMPLE)
    assert "System" in h
    assert "User" in h
    assert "Assistant" in h
    assert "You are helpful." in h
    assert "Hello" in h
    assert "Hi there!" in h


def test_html_title_in_head_and_h1():
    h = export_html(SIMPLE, title="Test Chat")
    assert "<title>Test Chat</title>" in h
    assert "<h1>Test Chat</h1>" in h


def test_html_default_title():
    h = export_html(SIMPLE)
    assert "<title>Conversation</title>" in h


def test_html_excludes_system():
    h = export_html(SIMPLE, include_system=False)
    assert 'class="message system"' not in h


def test_html_css_classes_present():
    h = export_html(SIMPLE)
    assert 'class="message system"' in h
    assert 'class="message user"' in h
    assert 'class="message assistant"' in h


def test_html_escapes_special_chars():
    msgs = [{"role": "user", "content": "<script>alert('xss')</script>"}]
    h = export_html(msgs)
    assert "<script>" not in h
    assert "&lt;script&gt;" in h


def test_html_empty_messages():
    h = export_html([])
    assert "<!DOCTYPE html>" in h


def test_html_no_title_heading():
    h = export_html(SIMPLE, title="")
    assert "<h1>" not in h


# ---------------------------------------------------------------------------
# export_json
# ---------------------------------------------------------------------------


def test_json_contains_messages():
    j = export_json(SIMPLE)
    data = json.loads(j)
    assert "messages" in data
    assert len(data["messages"]) == 3


def test_json_message_content():
    j = export_json(SIMPLE)
    data = json.loads(j)
    assert data["messages"][1]["role"] == "user"
    assert data["messages"][1]["content"] == "Hello"


def test_json_with_metadata():
    j = export_json(SIMPLE, metadata={"model": "claude-3", "turns": 2})
    data = json.loads(j)
    assert data["metadata"]["model"] == "claude-3"
    assert data["metadata"]["turns"] == 2


def test_json_no_metadata_key_when_none():
    j = export_json(SIMPLE)
    data = json.loads(j)
    assert "metadata" not in data


def test_json_compact():
    j = export_json(SIMPLE, indent=None)
    assert "\n" not in j


def test_json_indented():
    j = export_json(SIMPLE, indent=2)
    assert "\n" in j


def test_json_empty_messages():
    j = export_json([])
    data = json.loads(j)
    assert data["messages"] == []


# ---------------------------------------------------------------------------
# ConversationExporter class
# ---------------------------------------------------------------------------


def test_exporter_message_count():
    e = ConversationExporter(SIMPLE)
    assert e.message_count() == 3


def test_exporter_messages_property_is_copy():
    e = ConversationExporter(SIMPLE)
    msgs = e.messages
    msgs.append({"role": "user", "content": "extra"})
    assert e.message_count() == 3


def test_exporter_metadata_property():
    e = ConversationExporter(SIMPLE, metadata={"key": "val"})
    assert e.metadata == {"key": "val"}


def test_exporter_metadata_property_is_copy():
    e = ConversationExporter(SIMPLE, metadata={"key": "val"})
    m = e.metadata
    m["key"] = "changed"
    assert e.metadata["key"] == "val"


def test_exporter_roles():
    e = ConversationExporter(SIMPLE)
    assert e.roles() == ["system", "user", "assistant"]


def test_exporter_to_markdown():
    e = ConversationExporter(SIMPLE)
    md = e.to_markdown()
    assert "**User**" in md
    assert "Hello" in md


def test_exporter_to_markdown_with_title():
    e = ConversationExporter(SIMPLE)
    md = e.to_markdown(title="Chat")
    assert "# Chat" in md


def test_exporter_to_text():
    e = ConversationExporter(SIMPLE)
    txt = e.to_text()
    assert "User:" in txt


def test_exporter_to_html():
    e = ConversationExporter(SIMPLE)
    h = e.to_html()
    assert "<!DOCTYPE html>" in h


def test_exporter_to_json_no_metadata():
    e = ConversationExporter(SIMPLE)
    data = json.loads(e.to_json())
    assert "metadata" not in data


def test_exporter_to_json_with_metadata():
    e = ConversationExporter(SIMPLE, metadata={"k": "v"})
    data = json.loads(e.to_json())
    assert data["metadata"] == {"k": "v"}


def test_exporter_to_json_exclude_metadata():
    e = ConversationExporter(SIMPLE, metadata={"k": "v"})
    data = json.loads(e.to_json(include_metadata=False))
    assert "metadata" not in data


def test_exporter_export_markdown():
    e = ConversationExporter(SIMPLE)
    result = e.export(ExportFormat.MARKDOWN)
    assert "**User**" in result


def test_exporter_export_text():
    e = ConversationExporter(SIMPLE)
    result = e.export(ExportFormat.TEXT)
    assert "User:" in result


def test_exporter_export_html():
    e = ConversationExporter(SIMPLE)
    result = e.export(ExportFormat.HTML)
    assert "<!DOCTYPE html>" in result


def test_exporter_export_json():
    e = ConversationExporter(SIMPLE)
    result = e.export(ExportFormat.JSON)
    data = json.loads(result)
    assert "messages" in data


def test_exporter_export_invalid_format():
    e = ConversationExporter(SIMPLE)
    with pytest.raises((ValueError, AttributeError)):
        e.export("not_a_format")  # type: ignore[arg-type]


def test_exporter_repr():
    e = ConversationExporter(SIMPLE)
    assert repr(e) == "ConversationExporter(messages=3)"


def test_exporter_empty():
    e = ConversationExporter([])
    assert e.message_count() == 0
    assert e.roles() == []


# ---------------------------------------------------------------------------
# ExportFormat enum
# ---------------------------------------------------------------------------


def test_export_format_values():
    assert ExportFormat.MARKDOWN.value == "markdown"
    assert ExportFormat.HTML.value == "html"
    assert ExportFormat.TEXT.value == "text"
    assert ExportFormat.JSON.value == "json"

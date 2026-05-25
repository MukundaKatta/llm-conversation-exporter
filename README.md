# llm-conversation-exporter

Export LLM conversations to markdown, HTML, plain text, and JSON. Zero dependencies.

Works with Anthropic and OpenAI message formats, including multi-modal content blocks.

## Install

```bash
pip install llm-conversation-exporter
```

## Usage

```python
from llm_conversation_exporter import ConversationExporter

messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."},
]

exporter = ConversationExporter(messages)

print(exporter.to_markdown())
print(exporter.to_text())
print(exporter.to_html(title="My Chat"))
print(exporter.to_json())
```

## Formats

### Markdown

```python
md = exporter.to_markdown(title="Session 1", include_system=False)
```

### Plain text

```python
txt = exporter.to_text(include_system=False, separator="===")
```

### HTML

```python
html = exporter.to_html(title="Chat Log", include_system=True)
# Returns a full <!DOCTYPE html> document with inline CSS
```

### JSON

```python
j = exporter.to_json(include_metadata=True, indent=2)
```

### Via ExportFormat enum

```python
from llm_conversation_exporter import ExportFormat

result = exporter.export(ExportFormat.MARKDOWN, title="Chat")
```

## Standalone functions

```python
from llm_conversation_exporter.core import (
    export_markdown,
    export_text,
    export_html,
    export_json,
)

md = export_markdown(messages, title="Log", include_system=False)
txt = export_text(messages, separator="---")
html = export_html(messages, title="Conversation")
j = export_json(messages, metadata={"model": "claude-sonnet-4-6"})
```

## Multi-modal content

Content blocks are handled automatically:

| Block type | Output |
|------------|--------|
| `text` | Extracted text |
| `tool_use` | `[tool_use: name({args})]` |
| `tool_result` | `[tool_result] content` |
| `image` | `[image]` |
| `document` | `[document]` |

## ConversationExporter API

| Method | Description |
|--------|-------------|
| `message_count()` | Number of messages |
| `roles()` | Ordered list of roles |
| `to_markdown(**kwargs)` | Export to Markdown |
| `to_text(**kwargs)` | Export to plain text |
| `to_html(**kwargs)` | Export to HTML |
| `to_json(**kwargs)` | Export to JSON |
| `export(fmt, **kwargs)` | Export via ExportFormat enum |

## License

MIT

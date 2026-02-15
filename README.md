# barebone

LLM primitives for Python. You own the loop.

```python
from barebone import complete, user

response = complete("claude-sonnet-4", [user("What is 2+2?")])
print(response.content)
```

## Install

```bash
pip install barebone
```

## Philosophy

barebone provides LLM primitives — you write the orchestration in Python.

- **`complete()`** — single LLM call
- **`execute()`** — run a tool
- **`Tool`** — define tools
- **`Message`**, **`Response`**, **`ToolCall`** — types

No hidden loops. No magic. Just primitives.

## Basic Usage

```python
from barebone import complete, user

messages = [user("Explain quantum computing in one sentence")]
response = complete("claude-sonnet-4", messages)
print(response.content)
```

## Tools

Define tools as classes:

```python
from barebone import Tool, Param

class GetWeather(Tool):
    """Get weather for a city."""
    city: str = Param(description="City name")

    def execute(self) -> str:
        return f"72°F and sunny in {self.city}"
```

## Agent Loop

You own the loop:

```python
from barebone import complete, execute, user, tool_result

tools = [GetWeather]
messages = [user("What's the weather in Paris?")]

while True:
    response = complete("claude-sonnet-4", messages, tools=tools)

    if not response.tool_calls:
        print(response.content)
        break

    for tool_call in response.tool_calls:
        result = execute(tool_call, tools)
        messages.append(tool_result(tool_call, result))
```

## Async

```python
import asyncio
from barebone import acomplete, aexecute, user, tool_result

async def main():
    tools = [GetWeather]
    messages = [user("What's the weather in Tokyo?")]

    while True:
        response = await acomplete("claude-sonnet-4", messages, tools=tools)

        if not response.tool_calls:
            print(response.content)
            break

        for tool_call in response.tool_calls:
            result = await aexecute(tool_call, tools)
            messages.append(tool_result(tool_call, result))

asyncio.run(main())
```

## Built-in Tools

```python
from barebone import Read, Write, Edit, Bash, Glob, Grep
from barebone import WebFetch, WebSearch, HttpRequest
from barebone import Python
```

| Tool | Description |
|------|-------------|
| `Read` | Read files |
| `Write` | Write files |
| `Edit` | Find and replace |
| `Bash` | Run commands |
| `Glob` | Find files by pattern |
| `Grep` | Search file contents |
| `WebFetch` | Fetch web pages |
| `WebSearch` | Search the web |
| `HttpRequest` | HTTP requests |
| `Python` | Execute Python |

## Hooks

Add lifecycle hooks to tool execution:

```python
from barebone import Hooks, Deny, tool_result

hooks = Hooks()

@hooks.before
def validate(tool_call):
    if tool_call.name == "Bash":
        raise Deny("Bash not allowed")

@hooks.after
def log(tool_call, result):
    print(f"{tool_call.name}: {result[:50]}")

# Use hooks.run() instead of execute()
for tool_call in response.tool_calls:
    result = hooks.run(tool_call, tools)
    messages.append(tool_result(tool_call, result))
```

## System Prompt

```python
response = complete(
    "claude-sonnet-4",
    messages,
    system="You are a helpful assistant.",
)
```

## Memory

Log conversations:

```python
from barebone import Memory

memory = Memory("./chat.db")  # SQLite
memory.log("user", "Hello")
memory.log("assistant", "Hi there!")

# Retrieve
messages = memory.get_messages()
```

## Authentication

Set environment variables:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# or
export OPENROUTER_API_KEY=sk-or-...
```

Or pass explicitly:

```python
response = complete("claude-sonnet-4", messages, api_key="sk-ant-...")
```

## API Reference

### `complete(model, messages, **kwargs) -> Response`

Make a single LLM call.

| Param | Type | Description |
|-------|------|-------------|
| `model` | `str` | Model name |
| `messages` | `list[Message]` | Conversation |
| `system` | `str` | System prompt |
| `tools` | `list[Tool]` | Available tools |
| `api_key` | `str` | API key |
| `max_tokens` | `int` | Max response tokens |
| `temperature` | `float` | Sampling temperature |

### `execute(tool_call, tools) -> str`

Execute a tool call.

### `user(content) -> Message`

Create a user message.

### `assistant(content) -> Message`

Create an assistant message.

### `tool_result(tool_call, result) -> Message`

Create a tool result message.

### `Hooks`

Composable hooks for tool execution lifecycle.

| Method | Description |
|--------|-------------|
| `@hooks.before` | Register a before hook. Raise `Deny` to reject. |
| `@hooks.after` | Register an after hook. Return value replaces result. |
| `hooks.run(tool_call, tools)` | Execute with hooks: before → execute → after |
| `hooks.arun(tool_call, tools)` | Async version of run |

## License

MIT

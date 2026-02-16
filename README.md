# barebone

LLM primitives for Python. Build agents your way.

```python
import os
from barebone import Agent, tool

@tool
def get_weather(city: str) -> str:
    return f"72°F in {city}"

api_key = os.environ["ANTHROPIC_API_KEY"]
agent = Agent("claude-sonnet-4", api_key=api_key, tools=[get_weather])
print(agent.run_sync("Weather in Tokyo?").content)
```

## Install

```bash
pip install barebone
```

## Quick Start

```python
import os
from barebone import Agent, tool

@tool
def calculate(expression: str) -> str:
    return str(eval(expression))

api_key = os.environ["ANTHROPIC_API_KEY"]
agent = Agent("claude-sonnet-4", api_key=api_key, tools=[calculate])
response = agent.run_sync("What is 123 * 456?")
print(response.content)
```

## Agent

The `Agent` class handles the tool loop automatically:

```python
import os
from barebone import Agent

api_key = os.environ["ANTHROPIC_API_KEY"]
agent = Agent(
    "claude-sonnet-4",
    api_key=api_key,
    tools=[calculate, "Glob", "Read"],  # Mix custom and built-in tools
    system="You are a helpful assistant.",
    max_turns=10,  # Safety limit
)

# Sync
response = agent.run_sync("What files are here?")

# Async
response = await agent.run("What files are here?")

# Streaming
async for event in agent.stream("Write a poem"):
    if hasattr(event, "text"):
        print(event.text, end="")
```

### Multi-turn Conversations

```python
agent = Agent("claude-sonnet-4", api_key=api_key)

response = agent.run_sync("My name is Alice.")
response = agent.run_sync("What's my name?")  # Remembers context

agent.clear_messages()  # Reset conversation
```

## Tools

### @tool Decorator

```python
from barebone import tool

@tool
def search(query: str, limit: int = 10) -> str:
    return f"Found {limit} results for {query}"

@tool("custom_name")
def my_func(x: int) -> int:
    return x * 2
```

### Tool Class

For more control, use the `Tool` class:

```python
from barebone import Tool, Param

class GetWeather(Tool):
    """Get weather for a city."""
    city: str = Param(description="City name")
    units: str = Param(default="fahrenheit")

    def execute(self) -> str:
        return f"72° in {self.city}"
```

### Built-in Tools

```python
from barebone import Read, Write, Edit, Bash, Glob, Grep
from barebone import WebFetch, WebSearch, HttpRequest
from barebone import Python

# Use by name with Agent
agent = Agent("claude-sonnet-4", api_key=api_key, tools=["Read", "Bash", "Glob"])
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

Control tool execution:

```python
from barebone import Agent, Hooks

hooks = Hooks()

@hooks.before
def log_call(tool_call):
    print(f"Calling: {tool_call.name}")

@hooks.before
def block_dangerous(tool_call):
    if tool_call.name == "Bash":
        if "rm " in tool_call.arguments.get("command", ""):
            raise Hooks.Deny("Dangerous command blocked")

@hooks.after
def log_result(tool_call, result):
    print(f"Result: {result[:100]}")

agent = Agent("claude-sonnet-4", api_key=api_key, tools=["Bash"], hooks=hooks)
```

## Primitives

For full control, use the primitives directly:

```python
import os
from barebone import complete, execute, user, tool_result

api_key = os.environ["ANTHROPIC_API_KEY"]
tools = [GetWeather]
messages = [user("What's the weather in Paris?")]

while True:
    response = complete("claude-sonnet-4", messages, api_key=api_key, tools=tools)

    if not response.tool_calls:
        print(response.content)
        break

    for tc in response.tool_calls:
        result = execute(tc, tools)
        messages.append(tool_result(tc, result))
```

### Streaming

```python
from barebone import astream, user
from barebone.common.dataclasses import TextDelta, Done

async for event in astream("claude-sonnet-4", [user("Write a poem")], api_key=api_key):
    if isinstance(event, TextDelta):
        print(event.text, end="", flush=True)
    elif isinstance(event, Done):
        print(f"\n\nTokens: {event.response.usage.total_tokens}")
```

### Structured Output

```python
from pydantic import BaseModel
from barebone import complete, user

class Answer(BaseModel):
    answer: str
    confidence: float

response = complete(
    "claude-sonnet-4",
    [user("What is the capital of France?")],
    api_key=api_key,
    response_model=Answer,
)
print(response.parsed.answer)       # "Paris"
print(response.parsed.confidence)   # 0.99
```

### Async Primitives

```python
from barebone import acomplete, aexecute, astream

response = await acomplete("claude-sonnet-4", messages, api_key=api_key, tools=tools)
result = await aexecute(tool_call, tools)

async for event in astream("claude-sonnet-4", messages, api_key=api_key):
    ...
```

## Memory

Persist conversations:

```python
from barebone import Memory

memory = Memory("./chat.db")  # SQLite
memory.log("user", "Hello")
memory.log("assistant", "Hi there!")

messages = memory.get_messages()
```

## Authentication

Pass the API key explicitly:

```python
import os

api_key = os.environ["ANTHROPIC_API_KEY"]  # or OPENROUTER_API_KEY
agent = Agent("claude-sonnet-4", api_key=api_key)
```

## API Reference

### Agent

```python
Agent(
    model: str,
    *,
    api_key: str,             # Required
    tools: list = None,       # Tool classes, @tool functions, or "Read"/"Bash"
    system: str = None,
    memory: Memory = None,
    hooks: Hooks = None,
    max_turns: int = 10,
)
```

| Method | Description |
|--------|-------------|
| `run(prompt)` | Async tool loop, returns Response |
| `run_sync(prompt)` | Sync wrapper |
| `stream(prompt)` | Async generator yielding events |
| `clear_messages()` | Reset conversation |
| `add_tool(tool)` | Add tool dynamically |

| Property | Description |
|----------|-------------|
| `messages` | Conversation history |
| `tools` | Resolved ToolDefs |

### Primitives

| Function | Description |
|----------|-------------|
| `complete(model, messages, **kwargs)` | Single LLM call |
| `acomplete(model, messages, **kwargs)` | Async LLM call |
| `stream(model, messages, **kwargs)` | Stream response (returns async iterator) |
| `astream(model, messages, **kwargs)` | Async stream |
| `execute(tool_call, tools)` | Execute tool |
| `aexecute(tool_call, tools)` | Async execute |
| `user(content)` | Create user message |
| `assistant(content)` | Create assistant message |
| `tool_result(tool_call, result)` | Create tool result |

**complete/acomplete kwargs:**
- `api_key` — Required. Anthropic or OpenRouter API key
- `system` — System prompt
- `tools` — List of tools
- `response_model` — Pydantic model for structured output
- `max_tokens` — Max response tokens (default: 8192)
- `temperature` — Sampling temperature
- `timeout` — Timeout in seconds (raises `asyncio.TimeoutError`)

### Hooks

| Method | Description |
|--------|-------------|
| `@hooks.before` | Before hook. Raise `Deny` to reject. |
| `@hooks.after` | After hook. Return value replaces result. |
| `hooks.run(tool_call, tools)` | Execute with hooks |
| `hooks.arun(tool_call, tools)` | Async execute with hooks |

## License

MIT

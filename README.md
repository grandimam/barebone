# barebone

LLM primitives for Python. Build agents your way.

```python
from barebone import Agent, tool

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"72°F in {city}"

agent = Agent("claude-sonnet-4", tools=[get_weather])
print(agent.run_sync("Weather in Tokyo?").content)
```

## Install

```bash
pip install barebone
```

## Quick Start

```python
from barebone import Agent, tool

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

agent = Agent("claude-sonnet-4", tools=[calculate])
response = agent.run_sync("What is 123 * 456?")
print(response.content)
```

## Agent

The `Agent` class handles the tool loop automatically:

```python
from barebone import Agent

agent = Agent(
    model="claude-sonnet-4",
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
agent = Agent("claude-sonnet-4")

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
    """Search for documents."""
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
agent = Agent("claude-sonnet-4", tools=["Read", "Bash", "Glob"])
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

agent = Agent("claude-sonnet-4", tools=["Bash"], hooks=hooks)
```

## Primitives

For full control, use the primitives directly:

```python
from barebone import complete, execute, user, tool_result

tools = [GetWeather]
messages = [user("What's the weather in Paris?")]

while True:
    response = complete("claude-sonnet-4", messages, tools=tools)

    if not response.tool_calls:
        print(response.content)
        break

    for tc in response.tool_calls:
        result = execute(tc, tools)
        messages.append(tool_result(tc, result))
```

### Async Primitives

```python
from barebone import acomplete, aexecute

response = await acomplete("claude-sonnet-4", messages, tools=tools)
result = await aexecute(tool_call, tools)
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

Set environment variables:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# or
export OPENROUTER_API_KEY=sk-or-...
```

Or pass explicitly:

```python
agent = Agent("claude-sonnet-4", api_key="sk-ant-...")
```

## API Reference

### Agent

```python
Agent(
    model: str,
    tools: list = None,       # Tool classes, @tool functions, or "Read"/"Bash"
    system: str = None,
    api_key: str = None,
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
| `execute(tool_call, tools)` | Execute tool |
| `aexecute(tool_call, tools)` | Async execute |
| `user(content)` | Create user message |
| `assistant(content)` | Create assistant message |
| `tool_result(tool_call, result)` | Create tool result |

### Hooks

| Method | Description |
|--------|-------------|
| `@hooks.before` | Before hook. Raise `Deny` to reject. |
| `@hooks.after` | After hook. Return value replaces result. |
| `hooks.run(tool_call, tools)` | Execute with hooks |
| `hooks.arun(tool_call, tools)` | Async execute with hooks |

## License

MIT

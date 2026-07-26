# osso

Osso SDK provides lightweight primitives for building agents and agentic workflows in pure Python.

```bash
pip install osso
```

## Philosophy

osso provides simple primitives, not a framework. The orchestration layer is just Python.

- **No DSLs or abstractions** — Use `if`, `for`, `async/await`, and functions you already know
- **No hidden magic** — Tools are functions, agents are coroutines, everything is inspectable
- **Primitives over frameworks** — Small, composable pieces that fit your architecture
- **Python is the orchestrator** — Chain agents with function calls, not configuration files

Complex workflows don't need complex frameworks. A `for` loop that calls agents sequentially is a pipeline. An `asyncio.gather()` is parallel execution. A function that picks which agent to call is routing. You already know how to build these.

## Table of Contents

- [Quick Start](#quick-start)
- [Streaming](#streaming)
- [Conversation](#conversation)
- [Tools](#tools)
  - [@tool Decorator](#tool-decorator)
  - [Executing Tool Calls](#executing-tool-calls)
  - [Built-in Tools](#built-in-tools)
- [Agents](#agents)
  - [Defining an Agent](#defining-an-agent)
  - [Running Agents](#running-agents)
  - [Checkpointing](#checkpointing)
  - [Suspend & Resume](#suspend--resume)
  - [Persistent Storage](#persistent-storage)
- [Providers](#providers)
- [Types](#types)
- [Examples](#examples)
- [License](#license)

## Quick Start

`LLMClient` is a thin async wrapper around any OpenAI-compatible chat completions API (OpenRouter by default):

```python
import asyncio
import uuid
from osso import LLMClient, Request, Message, Done

async def main():
    client = LLMClient(api_key="sk-or-...", model="anthropic/claude-sonnet-4")

    request = Request(
        id=str(uuid.uuid4()),
        messages=[Message(role="user", content="What is 2 + 2?")],
    )

    async for event in client.stream(request):
        if isinstance(event, Done):
            print(event.response.content)

    await client.close()

asyncio.run(main())
```

## Streaming

`client.stream()` yields typed events as the response arrives:

```python
from osso import TextDelta, ToolCallStart, ToolCallEnd, Done, Error

async for event in client.stream(request):
    if isinstance(event, TextDelta):
        print(event.text, end="", flush=True)
    elif isinstance(event, ToolCallStart):
        print(f"\n[calling {event.name}...]")
    elif isinstance(event, ToolCallEnd):
        print(f"[{event.name} args: {event.arguments}]")
    elif isinstance(event, Error):
        print(f"error: {event.error}")
    elif isinstance(event, Done):
        response = event.response
```

## Conversation

The `Messages` builder makes multi-turn loops and tool calls easy:

```python
from osso import Messages, execute_tools

messages = Messages()
messages.user("What's the weather in Tokyo?")

while True:
    request = Request(id=str(uuid.uuid4()), messages=messages.list, tools=[get_weather])

    async for event in client.stream(request):
        if isinstance(event, Done):
            response = event.response

    if response.done:
        print(response.content)
        break

    # Model wants to call tools — execute and continue the loop
    messages.assistant(response)
    results = await execute_tools(response.tool_calls, [get_weather])
    messages.tool_results(results)
```

## Tools

### @tool Decorator

Plain functions become tools. The schema is inferred from the signature and docstring:

```python
from osso import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"72F in {city}"

@tool
async def fetch_data(url: str) -> str:
    """Fetch the contents of a URL."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.text
```

### Executing Tool Calls

When the model responds with tool calls, `execute_tools` runs them (sync and async tools supported):

```python
from osso import execute_tools

results = await execute_tools(response.tool_calls, [get_weather, fetch_data])
# -> list[ToolResult], ready to append via messages.tool_results(results)
```

### Built-in Tools

```python
from osso import read, write, edit, bash, glob, grep
from osso import web_fetch, web_search, http_request, ask_user_question
```

| Tool                | Description              |
| ------------------- | ------------------------ |
| `read`              | Read file contents       |
| `write`             | Write to file            |
| `edit`              | Find and replace in file |
| `bash`              | Execute shell commands   |
| `glob`              | Find files by pattern    |
| `grep`              | Search file contents     |
| `web_fetch`         | Fetch web pages          |
| `web_search`        | Search the web           |
| `http_request`      | HTTP requests            |
| `ask_user_question` | Prompt the user          |

## Agents

An agent is just an async function that receives a `Context`. The runner gives you lifecycle, state, checkpointing, and suspend/resume.

### Defining an Agent

```python
from osso import agent, Context

@agent
async def researcher(ctx: Context):
    topic = ctx.state["topic"]
    # ... call LLMClient, use tools, whatever you want ...
    ctx.state["report"] = f"Report on {topic}"
    return ctx.state["report"]
```

### Running Agents

```python
from osso import AgentRunner

runner = AgentRunner()

handle = await runner.start(researcher, state={"topic": "quantum computing"})

result = await handle.wait()
print(handle.status)  # AgentStatus.COMPLETED
```

Because handles are just asyncio tasks under the hood, orchestration is plain Python:

```python
# Parallel execution
handles = [await runner.start(researcher, state={"topic": t}) for t in topics]
reports = await asyncio.gather(*(h.wait() for h in handles))
```

### Checkpointing

Agents can persist their state at any point:

```python
@agent
async def worker(ctx: Context):
    for i in range(10):
        ctx.state["step"] = i
        await ctx.checkpoint()  # saves state to storage
```

### Suspend & Resume

Agents can pause and wait for external input (human-in-the-loop, approvals, webhooks):

```python
@agent
async def approval_flow(ctx: Context):
    ctx.state["draft"] = "..."
    await ctx.checkpoint()

    # Suspends until someone sends an event (or 1-hour timeout)
    decision = await ctx.suspend(reason="Waiting for approval", timeout=3600)

    if decision and decision.get("approved"):
        return "published"
    return "rejected"

# Elsewhere, from another task/process:
handle.send(AgentEvent(type="approval", data={"approved": True}))
```

### Persistent Storage

Use `FileStorage` to survive process restarts, then resume agents from their checkpoint:

```python
from osso import AgentRunner, FileStorage

runner = AgentRunner(storage=FileStorage("./checkpoints"))

# Resume a crashed or suspended agent by id
handle = await runner.resume(approval_flow, agent_id="flow-123")
```

## Providers

`LLMClient` works with any OpenAI-compatible API. Point `base_url` wherever you like:

```python
from osso import LLMClient

# OpenRouter (default) — access to many models
client = LLMClient(api_key="sk-or-...", model="anthropic/claude-sonnet-4")

# OpenAI
client = LLMClient(api_key="sk-...", model="gpt-4o", base_url="https://api.openai.com/v1")

# Local (Ollama, vLLM, LM Studio, ...)
client = LLMClient(api_key="ollama", model="llama3.1", base_url="http://localhost:11434/v1")
```

## Types

```python
from osso import Message, Request, Response, Tool, ToolCall, ToolResult
from osso import TextContent, ImageContent

# Message with text
Message(role="user", content="Hello")

# Message with images (vision)
Message(role="user", content=[
    TextContent(type="text", text="What's this?"),
    ImageContent(type="image", source="https://example.com/img.png"),
])

# Request
Request(
    id="...",
    messages=[...],
    system="You are a helpful assistant.",
    tools=[get_weather],
    max_tokens=8192,
    temperature=0.7,
)

# Response
response.content       # str | None
response.tool_calls    # list[ToolCall]
response.done          # True when no tool calls requested

# ToolCall
tc.id         # str
tc.name       # str
tc.arguments  # dict
```

## Examples

See `examples/` for runnable patterns:

- `simple_agent.py` — agent with checkpointing and human approval via suspend/resume
- `long_running_agent.py` — long-running agent with progress checkpoints
- `resume_agent.py` — resuming an agent from persistent storage

## License

MIT

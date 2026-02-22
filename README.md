# barebone

Primitives for building AI agents in Python.

```bash
pip install barebone
```

## Quick Start

```python
from barebone import Agent

agent = Agent(api_key="sk-ant-...", model="claude-sonnet-4-20250514")
response = agent.run_sync("What is 2 + 2?")
print(response.content)
```

## Agent

```python
from barebone import Agent, tool

@tool
def get_weather(city: str) -> str:
    return f"72F in {city}"

agent = Agent(
    api_key="sk-ant-...",
    model="claude-sonnet-4-20250514",
    tools=[get_weather],
    system="You are a helpful assistant.",
    max_tokens=8192,
    temperature=0.7,
    timeout=30.0,  # Optional timeout in seconds
)

# Sync
response = agent.run_sync("What's the weather in Tokyo?")

# Async
response = await agent.run("What's the weather in Tokyo?")
```

### Streaming

```python
async for event in agent.stream("Write a poem"):
    if event["type"] == "text_delta":
        print(event["text"], end="", flush=True)
    elif event["type"] == "done":
        print()
```

### Conversation

```python
response = agent.run_sync("My name is Alice.")
response = agent.run_sync("What's my name?")  # Remembers context

print(agent.messages)  # View history
agent.clear_messages()  # Reset conversation
```

### Resource Cleanup

```python
# Context manager (recommended)
async with Agent(api_key="...", model="...") as agent:
    response = await agent.run("Hello")

# Manual cleanup
agent = Agent(api_key="...", model="...")
try:
    response = await agent.run("Hello")
finally:
    await agent.close()
```

### Vision

```python
# Image URL
response = await agent.run(
    "What's in this image?",
    images=["https://example.com/photo.jpg"]
)

# Base64 data URI
response = await agent.run(
    "Describe this image",
    images=["data:image/png;base64,iVBORw0KGgo..."]
)

# Multiple images
response = await agent.run(
    "Compare these images",
    images=["https://example.com/a.jpg", "https://example.com/b.jpg"]
)
```

### Timeout

```python
# Per-agent timeout
agent = Agent(api_key="...", model="...", timeout=30.0)

# Per-request timeout
response = await agent.run("Hello", timeout=10.0)
```

## Tools

### @tool Decorator

```python
from barebone import tool

@tool
def calculate(expression: str) -> str:
    return str(eval(expression))

@tool("custom_name")
def my_func(x: int) -> int:
    return x * 2

@tool
async def fetch_data(url: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.text
```

### Dynamic Tools

```python
agent = Agent(api_key="...", model="...")

@tool
def new_tool() -> str:
    return "result"

agent.add_tool(new_tool)
```

### Built-in Tools

```python
from barebone import read, write, edit, bash, glob, grep
from barebone import web_fetch, web_search, http_request

agent = Agent(
    api_key="...",
    model="...",
    tools=[read, write, bash, glob],
)
```

| Tool | Description |
|------|-------------|
| `read` | Read file contents |
| `write` | Write to file |
| `edit` | Find and replace in file |
| `bash` | Execute shell commands |
| `glob` | Find files by pattern |
| `grep` | Search file contents |
| `web_fetch` | Fetch web pages |
| `web_search` | Search the web |
| `http_request` | HTTP requests |

## Providers

Auto-detected from API key prefix:

| Prefix | Provider |
|--------|----------|
| `sk-ant-` | Anthropic |
| `sk-` | OpenAI |

Or use providers directly:

```python
from barebone import Agent, AnthropicProvider, OpenAIProvider

provider = AnthropicProvider(api_key="sk-ant-...", model="claude-sonnet-4-20250514")
agent = Agent(provider=provider)

provider = OpenAIProvider(api_key="sk-...", model="gpt-4o")
agent = Agent(provider=provider)
```

## Types

```python
from barebone import Message, Response, Tool, ToolCall, ToolResult
from barebone import TextContent, ImageContent

# Message with text
Message(role="user", content="Hello")

# Message with images
Message(role="user", content=[
    TextContent(type="text", text="What's this?"),
    ImageContent(type="image", source="https://example.com/img.png"),
])

# Response
response.content      # str | None
response.tool_calls   # list[ToolCall]
response.stop_reason  # str

# ToolCall
tc.id         # str
tc.name       # str
tc.arguments  # dict
```

## Examples

See `examples/` for patterns:

**Basic**
- `01_basic.py` - Simple prompt/response
- `02_tools.py` - Agent with tools
- `03_streaming.py` - Real-time streaming
- `04_conversation.py` - Multi-turn conversation

**Patterns**
- `05_chaining.py` - Sequential prompts
- `06_routing.py` - Query routing
- `07_parallel.py` - Concurrent execution
- `08_reflection.py` - Self-review
- `09_planning.py` - Planning with tools
- `10_orchestrator.py` - Coordinator pattern
- `11_human_in_loop.py` - User confirmation

**Advanced**
- `12_vision.py` - Image/vision support
- `13_timeout.py` - Timeout handling

**Multi-Agent**
- `14_pipeline.py` - Sequential agent pipeline
- `15_parallel.py` - Parallel analysis with synthesis
- `16_handoff.py` - Agent-to-agent transfers
- `17_debate.py` - Adversarial debate pattern

## License

MIT

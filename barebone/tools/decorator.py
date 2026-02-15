from __future__ import annotations

import asyncio
import inspect
from typing import Any
from typing import Callable
from typing import get_type_hints

from pydantic import Field
from pydantic import create_model

from .types import ToolDef


def tool(
    fn_or_name: Callable | str | None = None,
    description: str | None = None,
) -> Any:
    def decorator(fn: Callable) -> _ToolWrapper:
        name = fn_or_name if isinstance(fn_or_name, str) else fn.__name__
        desc = description or _extract_description(fn)
        return _ToolWrapper(fn, name, desc)

    if callable(fn_or_name):
        return decorator(fn_or_name)
    return decorator


def _extract_description(fn: Callable) -> str:
    if fn.__doc__:
        first_para = fn.__doc__.strip().split("\n\n")[0]
        return " ".join(line.strip() for line in first_para.split("\n"))
    return f"Execute {fn.__name__}"


def _build_pydantic_model(fn: Callable) -> type:
    sig = inspect.signature(fn)
    hints = get_type_hints(fn) if hasattr(fn, "__annotations__") else {}

    fields = {}
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue

        annotation = hints.get(name, Any)
        if param.default is inspect.Parameter.empty:
            fields[name] = (annotation, ...)
        else:
            fields[name] = (annotation, Field(default=param.default))

    return create_model(fn.__name__, **fields)


class _ToolWrapper:

    def __init__(self, fn: Callable, name: str, description: str):
        self._fn = fn
        self._name = name
        self._description = description
        self._model = _build_pydantic_model(fn)

    def to_tool_def(self) -> ToolDef:
        fn = self._fn
        is_async = asyncio.iscoroutinefunction(fn)

        if is_async:
            async def handler(**kwargs: Any) -> Any:
                return await fn(**kwargs)
        else:
            def handler(**kwargs: Any) -> Any:
                return fn(**kwargs)

        schema = self._model.model_json_schema()
        properties = {
            k: {key: val for key, val in v.items() if key != "title"}
            for k, v in schema.get("properties", {}).items()
        }

        return ToolDef(
            name=self._name,
            description=self._description,
            parameters={
                "type": "object",
                "properties": properties,
                "required": schema.get("required", []),
            },
            handler=handler,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._fn(*args, **kwargs)

    @property
    def name(self) -> str:
        return self._name


__all__ = ["tool"]

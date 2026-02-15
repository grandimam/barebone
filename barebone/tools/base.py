from __future__ import annotations

import asyncio

from abc import abstractmethod
from typing import Any
from typing import ClassVar

from pydantic import BaseModel
from pydantic import Field as PydanticField

from .types import ToolDef


Param = PydanticField


class Tool(BaseModel):
    _tool_name: ClassVar[str | None] = None
    _tool_description: ClassVar[str | None] = None

    class Config:
        extra = "forbid"

    @abstractmethod
    def execute(self) -> Any:
        pass

    @classmethod
    def get_name(cls) -> str:
        if hasattr(cls, "_tool_name") and cls._tool_name:
            return cls._tool_name
        return cls.__name__

    @classmethod
    def get_description(cls) -> str:
        if hasattr(cls, "_tool_description") and cls._tool_description:
            return cls._tool_description.strip()
        if cls.__doc__:
            first_para = cls.__doc__.strip().split("\n\n")[0]
            return " ".join(line.strip() for line in first_para.split("\n"))
        return f"Execute {cls.get_name()}"

    @classmethod
    def get_parameters(cls) -> dict[str, Any]:
        schema = cls.model_json_schema()
        properties = {}
        for name, prop in schema.get("properties", {}).items():
            if name.startswith("_"):
                continue
            cleaned = {k: v for k, v in prop.items() if k != "title"}
            properties[name] = cleaned

        required = [
            r for r in schema.get("required", [])
            if not r.startswith("_")
        ]

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    @classmethod
    def to_tool_def(cls) -> ToolDef:
        if asyncio.iscoroutinefunction(cls.execute):
            async def handler(**kwargs: Any) -> Any:
                instance = cls(**kwargs)
                return await instance.execute()
        else:
            def handler(**kwargs: Any) -> Any:
                instance = cls(**kwargs)
                return instance.execute()

        return ToolDef(
            name=cls.get_name(),
            description=cls.get_description(),
            parameters=cls.get_parameters(),
            handler=handler,
        )


def is_tool_class(obj: Any) -> bool:
    return isinstance(obj, type) and issubclass(obj, Tool) and obj is not Tool


def resolve_tool(tool_input: ToolDef | type[Tool]) -> ToolDef:
    if isinstance(tool_input, ToolDef):
        return tool_input

    if is_tool_class(tool_input):
        return tool_input.to_tool_def()

    raise TypeError(
        f"Cannot convert {type(tool_input).__name__} to ToolDef. "
        "Use a Tool subclass or ToolDef instance."
    )


def tools_to_schema(tools: list[ToolDef]) -> list[dict[str, Any]]:
    return [
        {
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
        }
        for t in tools
    ]


__all__ = [
    "Tool",
    "Param",
]

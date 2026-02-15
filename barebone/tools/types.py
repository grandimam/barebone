from dataclasses import dataclass
from typing import Any
from typing import Callable


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable
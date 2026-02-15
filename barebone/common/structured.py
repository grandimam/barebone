from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel
from pydantic import ValidationError

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(Exception):

    def __init__(self, message: str, response_content: str | None = None):
        super().__init__(message)
        self.response_content = response_content


def get_schema_prompt(model: type[BaseModel]) -> str:
    schema = model.model_json_schema()
    schema_str = json.dumps(schema, indent=2)

    return f"""
        Respond with valid JSON matching this schema:
        
        ```json
        {schema_str}
        ```
            
        Return ONLY the JSON object, no other text.
    """


def parse_json_response(content: str) -> dict:
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    patterns = [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
        r'\{[\s\S]*\}',
    ]

    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            try:
                json_str = match.group(1) if '```' in pattern else match.group(0)
                return json.loads(json_str)
            except (json.JSONDecodeError, IndexError):
                continue

    raise StructuredOutputError(
        "Could not parse JSON from response",
        response_content=content,
    )


def parse_response(content: str, model: type[T]) -> T:
    try:
        data = parse_json_response(content)
        return model.model_validate(data)
    except ValidationError as e:
        raise StructuredOutputError(
            f"Validation failed: {e}",
            response_content=content,
        )


__all__ = [
    "StructuredOutputError",
    "get_schema_prompt",
    "parse_json_response",
    "parse_response",
]

"""Structured output examples using JSON responses."""

import asyncio
import json
import os

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic import Field

from barebone import Agent
from barebone import AnthropicProvider

load_dotenv()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MODEL = "claude-sonnet-4-20250514"


class Sentiment(BaseModel):
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(description="Confidence score 0-1")
    reasoning: str = Field(description="Brief explanation")


class Entity(BaseModel):
    name: str
    type: str = Field(description="person, place, organization, etc.")


class Extraction(BaseModel):
    entities: list[Entity]
    summary: str


class Decision(BaseModel):
    choice: str
    pros: list[str]
    cons: list[str]
    confidence: float


async def sentiment_analysis():
    """Structured sentiment analysis."""
    print("=" * 60)
    print("Sentiment Analysis")
    print("=" * 60)

    texts = [
        "I absolutely love this product! Best purchase ever.",
        "The service was okay, nothing special.",
        "Terrible experience. Would not recommend.",
    ]

    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)

    for text in texts:
        agent = Agent(
            provider=provider,
            system=f"""Analyze the sentiment and return a JSON object with:
- sentiment: "positive", "negative", or "neutral"
- confidence: number from 0 to 1
- reasoning: brief explanation

Return only valid JSON, no other text.""",
        )
        response = await agent.run(f"Analyze: {text}")

        try:
            result = Sentiment(**json.loads(response.content))
            print(f"\nText: {text}")
            print(f"  Sentiment: {result.sentiment} ({result.confidence:.0%})")
            print(f"  Reasoning: {result.reasoning}")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"\nText: {text}")
            print(f"  Raw response: {response.content}")


async def entity_extraction():
    """Extract named entities from text."""
    print("\n" + "=" * 60)
    print("Entity Extraction")
    print("=" * 60)

    text = """
    Apple CEO Tim Cook announced a partnership with Microsoft at their
    Cupertino headquarters. The deal was also celebrated in New York.
    """

    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)
    agent = Agent(
        provider=provider,
        system="""Extract entities and return a JSON object with:
- entities: array of {name, type} objects
- summary: brief summary of the text

Return only valid JSON, no other text.""",
    )
    response = await agent.run(f"Extract entities from:\n\n{text}")

    try:
        result = Extraction(**json.loads(response.content))
        print(f"Summary: {result.summary}")
        print("\nEntities:")
        for entity in result.entities:
            print(f"  - {entity.name} ({entity.type})")
    except (json.JSONDecodeError, ValueError):
        print(f"Raw response: {response.content}")


async def structured_decision():
    """Get a structured decision with pros/cons."""
    print("\n" + "=" * 60)
    print("Structured Decision")
    print("=" * 60)

    question = "Should a startup use Python or Rust for their backend?"

    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)
    agent = Agent(
        provider=provider,
        system="""Provide a decision and return a JSON object with:
- choice: your recommendation
- pros: array of advantages
- cons: array of disadvantages
- confidence: number from 0 to 1

Return only valid JSON, no other text.""",
    )
    response = await agent.run(question)

    try:
        result = Decision(**json.loads(response.content))
        print(f"Question: {question}")
        print(f"\nChoice: {result.choice} (confidence: {result.confidence:.0%})")
        print("\nPros:")
        for pro in result.pros:
            print(f"  + {pro}")
        print("\nCons:")
        for con in result.cons:
            print(f"  - {con}")
    except (json.JSONDecodeError, ValueError):
        print(f"Raw response: {response.content}")


async def main():
    await sentiment_analysis()
    await entity_extraction()
    await structured_decision()


if __name__ == "__main__":
    asyncio.run(main())

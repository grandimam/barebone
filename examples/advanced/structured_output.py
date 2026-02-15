"""
Structured Output pattern.

Get typed, validated responses using Pydantic models.
"""

from pydantic import BaseModel, Field

from barebone import complete, user


class Sentiment(BaseModel):
    """Sentiment analysis result."""
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(description="Confidence score 0-1")
    reasoning: str = Field(description="Brief explanation")


class Entity(BaseModel):
    """An extracted entity."""
    name: str
    type: str = Field(description="person, place, organization, etc.")


class Extraction(BaseModel):
    """Extracted entities from text."""
    entities: list[Entity]
    summary: str


class Decision(BaseModel):
    """A structured decision."""
    choice: str
    pros: list[str]
    cons: list[str]
    confidence: float


def sentiment_analysis():
    """Extract structured sentiment."""
    print("=" * 60)
    print("Sentiment Analysis")
    print("=" * 60)

    texts = [
        "I absolutely love this product! Best purchase ever.",
        "The service was okay, nothing special.",
        "Terrible experience. Would not recommend.",
    ]

    for text in texts:
        response = complete(
            "claude-sonnet-4-20250514",
            [user(f"Analyze the sentiment: {text}")],
            response_model=Sentiment,
        )
        result = response.parsed
        print(f"\nText: {text}")
        print(f"  Sentiment: {result.sentiment} ({result.confidence:.0%})")
        print(f"  Reasoning: {result.reasoning}")


def entity_extraction():
    """Extract entities with types."""
    print("\n" + "=" * 60)
    print("Entity Extraction")
    print("=" * 60)

    text = """
    Apple CEO Tim Cook announced a partnership with Microsoft at their
    Cupertino headquarters. The deal was also celebrated in New York.
    """

    response = complete(
        "claude-sonnet-4-20250514",
        [user(f"Extract all entities from this text:\n\n{text}")],
        response_model=Extraction,
    )
    result = response.parsed

    print(f"Summary: {result.summary}")
    print("\nEntities:")
    for entity in result.entities:
        print(f"  - {entity.name} ({entity.type})")


def structured_decision():
    """Get a structured decision."""
    print("\n" + "=" * 60)
    print("Structured Decision")
    print("=" * 60)

    question = "Should a startup use Python or Rust for their backend?"

    response = complete(
        "claude-sonnet-4-20250514",
        [user(question)],
        response_model=Decision,
    )
    result = response.parsed

    print(f"Question: {question}")
    print(f"\nChoice: {result.choice} (confidence: {result.confidence:.0%})")
    print("\nPros:")
    for pro in result.pros:
        print(f"  + {pro}")
    print("\nCons:")
    for con in result.cons:
        print(f"  - {con}")


def chained_structured():
    """Chain structured outputs."""
    print("\n" + "=" * 60)
    print("Chained Structured Output")
    print("=" * 60)

    class Step(BaseModel):
        """A single step."""
        step_number: int
        action: str
        expected_outcome: str

    class Plan(BaseModel):
        """A multi-step plan."""
        goal: str
        steps: list[Step]

    class Evaluation(BaseModel):
        """Plan evaluation."""
        feasibility: float = Field(description="0-1 score")
        risks: list[str]
        recommendation: str

    # Generate plan
    plan_response = complete(
        "claude-sonnet-4-20250514",
        [user("Create a plan to learn machine learning in 3 months")],
        response_model=Plan,
    )
    plan = plan_response.parsed

    print(f"Goal: {plan.goal}")
    print("\nSteps:")
    for step in plan.steps:
        print(f"  {step.step_number}. {step.action}")
        print(f"     Expected: {step.expected_outcome}")

    # Evaluate plan
    eval_response = complete(
        "claude-sonnet-4-20250514",
        [user(f"Evaluate this plan:\n\n{plan.model_dump_json()}")],
        response_model=Evaluation,
    )
    evaluation = eval_response.parsed

    print(f"\nFeasibility: {evaluation.feasibility:.0%}")
    print(f"Recommendation: {evaluation.recommendation}")
    print("Risks:")
    for risk in evaluation.risks:
        print(f"  - {risk}")


if __name__ == "__main__":
    sentiment_analysis()
    entity_extraction()
    structured_decision()
    chained_structured()

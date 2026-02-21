"""Structured output with Pydantic models."""

from pydantic import BaseModel

from barebone import Agent


class Sentiment(BaseModel):
    """Sentiment analysis result."""

    sentiment: str  # positive, negative, neutral
    confidence: float  # 0 to 1
    summary: str


class CodeReview(BaseModel):
    """Code review result."""

    score: int  # 1-10
    issues: list[str]
    approved: bool


def main():
    agent = Agent("claude-sonnet-4-20250514")

    # Sentiment analysis
    result = agent.run_sync("Analyze: I love this product! Great quality.", output=Sentiment)
    print(f"Sentiment: {result.sentiment} ({result.confidence:.0%})")

    # Code review
    code = "def avg(nums): return sum(nums)/len(nums)"
    review = agent.run_sync(f"Review this code: {code}", output=CodeReview)
    print(f"Score: {review.score}/10, Approved: {review.approved}")


if __name__ == "__main__":
    main()

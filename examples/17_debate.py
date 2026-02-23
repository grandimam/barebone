import asyncio
import os

from dotenv import load_dotenv

from barebone import Agent

load_dotenv()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MODEL = "claude-sonnet-4-20250514"


async def main():
    pro = Agent(
        api_key=API_KEY,
        model=MODEL,
        system="You argue FOR the proposition. Be persuasive but factual. Keep responses concise.",
    )

    con = Agent(
        api_key=API_KEY,
        model=MODEL,
        system="You argue AGAINST the proposition. Be persuasive but factual. Keep responses concise.",
    )

    judge = Agent(
        api_key=API_KEY,
        model=MODEL,
        system="You are an impartial judge. Evaluate both arguments fairly and provide a balanced conclusion.",
    )

    proposition = "Companies should adopt a 4-day work week"
    rounds = 2

    print(f"Proposition: {proposition}\n")
    print("=" * 60)

    debate_history = []

    for round_num in range(1, rounds + 1):
        print(f"\n--- Round {round_num} ---\n")

        if round_num == 1:
            pro_prompt = f"Argue FOR: {proposition}"
            con_prompt = f"Argue AGAINST: {proposition}"
        else:
            pro_prompt = f"Respond to the opposing argument:\n\n{debate_history[-1]}"
            con_prompt = f"Respond to the opposing argument:\n\n{debate_history[-2]}"

        pro_response, con_response = await asyncio.gather(
            pro.run(pro_prompt),
            con.run(con_prompt),
        )

        print(f"PRO: {pro_response.content}\n")
        print(f"CON: {con_response.content}\n")

        debate_history.append(pro_response.content)
        debate_history.append(con_response.content)

    print("=" * 60)
    print("\n--- Judge's Decision ---\n")

    debate_summary = "\n\n".join(
        [
            f"Round {i // 2 + 1} - PRO: {debate_history[i]}\n\nRound {i // 2 + 1} - CON: {debate_history[i + 1]}"
            for i in range(0, len(debate_history), 2)
        ]
    )

    verdict = await judge.run(f"Evaluate this debate and provide your verdict:\n\n{debate_summary}")
    print(verdict.content)


if __name__ == "__main__":
    asyncio.run(main())

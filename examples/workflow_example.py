"""
DAG-based workflow example.

Three-node workflow: research → (critique + expand in parallel) → final_report

Requires: ANTHROPIC_API_KEY or OPENAI_API_KEY
Run: python examples/workflow_example.py
"""

import asyncio

from deepcrew import Agent, WorkflowBuilder
from deepcrew.types import EventType


async def main():
    researcher = Agent(
        name="researcher",
        model="openai/gpt-4o-mini",
        system_prompt=(
            "You are a research specialist. Given a topic, produce a concise but "
            "comprehensive 3-paragraph summary covering key facts, recent developments, "
            "and open questions."
        ),
    )

    critic = Agent(
        name="critic",
        model="openai/gpt-4o-mini",
        system_prompt=(
            "You are a critical thinker. Given research output, identify gaps, "
            "potential biases, and areas that need more evidence. Be constructive."
        ),
    )

    expander = Agent(
        name="expander",
        model="openai/gpt-4o-mini",
        system_prompt=(
            "You are a domain expert. Given research output, add technical depth, "
            "specific examples, and practical implications."
        ),
    )

    writer = Agent(
        name="writer",
        model="openai/gpt-4o",
        system_prompt=(
            "You are a senior technical writer. Given research, critical notes, and "
            "expanded details, synthesize a polished, comprehensive report. "
            "Use clear headings and structure."
        ),
    )

    workflow = (
        WorkflowBuilder()
        .add_agent("research", researcher, task="{input}")
        .add_agent("critique", critic, task="Critically review this research:\n\n{research}")
        .add_agent(
            "expand", expander, task="Expand on this research with technical depth:\n\n{research}"
        )
        .add_agent(
            "report",
            writer,
            task=(
                "Write a final report on the topic based on:\n\n"
                "**Research:**\n{research}\n\n"
                "**Critical Review:**\n{critique}\n\n"
                "**Technical Details:**\n{expand}"
            ),
        )
        .then("research", "critique")
        .then("research", "expand")
        .then("critique", "report")
        .then("expand", "report")
    )

    topic = "The impact of large language models on software engineering"
    print(f"Running workflow for: {topic!r}\n")

    # --- Streaming mode ---
    print("=== Streaming events ===")
    async for event in workflow.stream(topic):
        if event.event == EventType.STEP_START:
            print(f"[START] {event.data['node']}")
        elif event.event == EventType.STEP_DONE:
            print(f"[DONE]  {event.data['node']}")
        elif event.event == EventType.DONE:
            print("\n=== Final Report ===")
            print(event.data.get("final_text", ""))

    print("\n--- Non-streaming mode ---")
    result = await workflow.run(topic)
    print(f"Total tokens: {result.total_tokens}")
    print(f"Nodes completed: {list(result.outputs.keys())}")


if __name__ == "__main__":
    asyncio.run(main())

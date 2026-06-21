"""
Simple single-agent example.

Requires: OPENAI_API_KEY environment variable.
Run: python examples/simple_agent.py
"""

import asyncio

from deepcrew import Agent, run_agent, tool


@tool
def get_weather(city: str, units: str = "celsius") -> dict:
    """Get current weather for a city.

    Args:
        city (str): The city name.
        units (str): Temperature units, either 'celsius' or 'fahrenheit'.
    """
    # Mock implementation — replace with a real API call
    return {"city": city, "temperature": 22, "units": units, "condition": "sunny"}


async def main():
    agent = Agent(
        name="weather_assistant",
        model="openai/gpt-4o-mini",
        system_prompt="You are a helpful weather assistant. Use the get_weather tool to answer questions.",
        tools=[get_weather],
        max_turns=3,
    )

    messages = [{"role": "user", "content": "What's the weather like in Tokyo and Paris?"}]
    result = await run_agent(agent, messages)

    print(f"Agent: {result.agent_id}")
    print(f"Model: {result.model}")
    print(f"Tokens: {result.input_tokens} in / {result.output_tokens} out")
    print(f"\nResponse:\n{result.text}")
    if result.tool_calls:
        print(f"\nTools called: {[tc['tool'] for tc in result.tool_calls]}")


if __name__ == "__main__":
    asyncio.run(main())

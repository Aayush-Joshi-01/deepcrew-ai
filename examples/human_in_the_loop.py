"""
Human-in-the-loop tool approval with AgentHooks.

approve_tool returning False denies the call outright: the callable never
runs, the tool result becomes "Tool call denied by user.", and a TOOL_DENIED
stream event is emitted. This is different from the observe-only event
stream -- hooks can actually change what happens.

Requires: OPENAI_API_KEY environment variable.
Run: python examples/human_in_the_loop.py
"""

import asyncio

from deepcrew import Agent, AgentHooks, run_agent, tool


@tool
def read_file(path: str) -> str:
    """Read a file's contents (mock implementation)."""
    return f"(contents of {path})"


@tool
def delete_file(path: str) -> str:
    """Delete a file (mock implementation)."""
    return f"deleted {path}"


async def approve_tool(tool_name: str, args: dict) -> bool:
    if tool_name == "delete_file":
        print(f"[approval] denying delete_file({args})")
        return False
    return True


async def on_tool_start(tool_name: str, args: dict) -> None:
    print(f"[hook] starting tool={tool_name} args={args}")


async def main():
    agent = Agent(
        name="assistant",
        model="openai/gpt-4o",
        system_prompt="You help manage files. Use tools as needed.",
        tools=[read_file, delete_file],
        hooks=AgentHooks(approve_tool=approve_tool, on_tool_start=on_tool_start),
    )

    messages = [
        {"role": "user", "content": "Read notes.txt, then delete it."},
    ]
    result = await run_agent(agent, messages)
    print(f"\nFinal answer:\n{result.text}")


if __name__ == "__main__":
    asyncio.run(main())

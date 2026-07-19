"""
Multimodal query: attach an image and a PDF to a single message.

image()/pdf() accept a URL, a data: URI, a local file path, or raw bytes.
Local/byte sources are size-checked and base64-encoded automatically.

Requires: ANTHROPIC_API_KEY environment variable (or swap the model below).
Run: python examples/multimodal_agent.py path/to/chart.png path/to/report.pdf
"""

import asyncio
import sys

from deepcrew import Agent, image, pdf, run_agent, user_message


async def main():
    if len(sys.argv) != 3:
        print("Usage: python examples/multimodal_agent.py <image_path> <pdf_path>")
        return

    image_path, pdf_path = sys.argv[1], sys.argv[2]

    agent = Agent(name="analyst", model="anthropic/claude-opus-4-8")

    msg = user_message(
        "Summarize what's in this image, then check whether it matches the attached report.",
        image(image_path),
        pdf(pdf_path),
    )

    result = await run_agent(agent, [msg])
    print(result.text)


if __name__ == "__main__":
    asyncio.run(main())

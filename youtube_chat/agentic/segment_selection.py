"""Agent for selecting a segment to study."""

from agentic.context import Context
from agents import Agent, RunContextWrapper


def instructions(ctx: RunContextWrapper[Context], agent: Agent[Context]) -> str:
    return f"""
You help the user select a segment to study.

Display a bulleted list of the segments to the user. Ask the user to select one.

Below is the list of segments:
{ctx.context.segmented_transcript.get_summaries()}
"""


agent = Agent[Context](
    name="Segment Selection Agent",
    instructions=instructions,
    model="gpt-4.1-mini",
)

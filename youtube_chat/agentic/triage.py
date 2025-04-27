"""This agent is responsible for managing the conversation.

Concepts demonstrated:
- Agents hand-off and tool use
- ContextWrapper as a way to share data between agent calls
"""

import logging

from agents import Agent, RunContextWrapper, Runner, function_tool

from youtube_chat.agentic.context import Context
from youtube_chat.agentic.segment_selection import agent as segment_selection_agent
from youtube_chat.agentic.teacher import agent as teacher_agent

logger = logging.getLogger(__name__)

instructions = """
You help the user with learning Korean through Youtube videos.
Note that you're only the orchestrator of the conversation, and you don't assist with the learning process yourself.

If the user has not selected a segment to study, or if the user has completed the current segment,
use the segment selection agent to help them choose a segment.

Once they have selected a segment, call the `get_segment_transcript` tool to get the transcript of the selected segment.
This function only needs to be called when the user has selected or changed to a new segment.

Always use the teaching agent to teach the content.
"""


@function_tool
def get_segment_transcript(ctx: RunContextWrapper[Context], index: int) -> str:
    """Get the transcript of the selected segment.

    Parameters:
        index: The index of the segment to get the transcript for (starts from 0).

    Returns:
        The transcript of the selected segment.
    """
    logger.info(f"User selected segment index: {index}")
    # Saving data in context is a way to share data across agents (can be non-sequential)
    ctx.context.current_selected_segment = ctx.context.segmented_transcript.segments[
        index
    ].text
    return ctx.context.current_selected_segment


agent = Agent[Context](
    name="Triage agent",
    instructions=instructions,
    handoffs=[segment_selection_agent, teacher_agent],
    tools=[get_segment_transcript],
    model="gpt-4.1",
)

if __name__ == "__main__":
    result = Runner.run_sync(agent, "Hello")
    print(result.final_output)

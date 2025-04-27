# TODO:
# - Try MCP server
# - Use AgentOps.ai to do logging and dashboard monitoring

import logging

import gradio as gr
from agentic.context import Context
from agentic.transcript_processing import TranscriptProcessor
from agents import Runner
from dotenv import load_dotenv
from pydantic_models import SegmentedTranscript

from youtube_chat.agentic.starter import agent as starter_agent
from youtube_chat.agentic.triage import agent as triage_agent

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

context = Context()


async def process_transcript(video_id: str) -> SegmentedTranscript:
    processor = TranscriptProcessor()
    return await processor.run(video_id)


def make_input(message, history) -> list[dict]:
    """Make the input which includes the history and the new message."""
    input = [{"role": m["role"], "content": m["content"]} for m in history]
    input.append({"role": "user", "content": message})

    return input


async def inference(message, history):
    input = make_input(message, history)
    logger.info(f"Input: {input}")

    # Check whether the conversation start part is completed. It's more efficient to do it
    # here instead of inside the triage agent as the state-based checking is more efficient
    # than LLM-based.
    if context.user_info is None or context.user_info.youtube_video_id == "":
        result = await Runner.run(starter_agent, input, context=context)
        logger.info(f"Starter agent result: {result}")
        context.user_info = result.final_output
        if not context.user_info.youtube_video_id:
            return context.user_info.response_message
        logger.info(f"Processing transcript for {context.user_info.youtube_video_id}")
        segmented_transcript = await process_transcript(
            context.user_info.youtube_video_id
        )
        context.segmented_transcript = segmented_transcript
        # This is needed to pass the steps from the starter agent to the triage agent
        input = result.to_input_list()

    result = await Runner.run(triage_agent, input, context=context)
    logger.info(f"Response: {result}")
    # TODO: result.to_input_list should be managed somehow so that it's passed to the inference method
    return result.final_output


chat = gr.ChatInterface(
    fn=inference,
    type="messages",
)

chat.launch()

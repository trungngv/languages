import logging
import re

import gradio as gr
from dotenv import load_dotenv

from youtube_chat.llms import OpenAIClient
from youtube_chat.transcript_processor import TranscriptProcessor
from youtube_chat.youtube_processor import YouTubeProcessor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

youtube_processor = YouTubeProcessor()
transcript_processor = TranscriptProcessor(OpenAIClient())

state = {
    "url": None,
    "segmented_transcript": None,
    "current_selected_segment": None,
}
openai_client = OpenAIClient()


def extract_user_intent(message) -> str:
    user_intent_pattern = r"user_intent:\s*(.+)\n"
    match = re.search(user_intent_pattern, message)
    return match.group(1) if match else None


def extract_youtube_url(message) -> str:
    """
    Extracts the YouTube URL from a given message string.

    Args:
        message (str): The input message containing the YouTube URL.

    Returns:
        str: The extracted YouTube URL, or None if no URL is found.
    """
    youtube_url_pattern = (
        r"(https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+|https?://youtu\.be/[\w-]+)"
    )
    match = re.search(youtube_url_pattern, message)
    return match.group(0) if match else None


systemp_prompt = """
You are a helpful assistant for learning Korean through YouTube videos.
Depending on the current state of the conversation, you will need to perform different actions.

1. If the user has not provided a Youtube Video URL, ask them to do so.
Respond with message as a string with following this exact format:

user_intent: providing youtube url
youtube_url: [the value of the youtube url]

2. Whenever the user selects a segment to study, explain to them each sentence in the segment individually.

For each sentence, providing the following explanation:

- An English translation
- List key vocabulary with their meanings (include Hanja if applicable)
- Explain common expressions or idioms (if any)
- Note 1 -2 important grammar points 

Note that the vocabulary, expressions/idioms, and grammar points should be selected based on the difficulty level of the text. I.e. if the text is advanced, the user is at the advanced level, so note more complex learning concepts.

You MUST only explain a sentence at a time, asking the user if they want to continue to the next sentence each time.
Otherwise you will overwhelm the user with too much information.
"""


def call_agent(message, history):
    messages = [
        {
            "role": "system",
            "content": systemp_prompt,
        }
    ]
    for m in history:
        messages.append(m)

    # Adding the state here as a message (not ideal for long transcripts)
    if state["segmented_transcript"]:
        messages.append(
            {
                "role": "assistant",
                "content": f"Here are the segments:\n {state['segmented_transcript']}\n",
            }
        )

    messages.append(
        {
            "role": "user",
            "content": message,
        }
    )

    return openai_client.chat(messages)


def inference(message, history):
    response = call_agent(
        message=message,
        history=history,
    )
    logger.info(f"Response: {response[0:100]} ...")

    user_intent = extract_user_intent(response)
    logger.info(f"User Intent: {user_intent}")

    if user_intent == "providing youtube url":
        youtube_url = extract_youtube_url(response)
        if youtube_url:
            state["url"] = youtube_url
            transcript = youtube_processor.get_transcript(youtube_url)
            segmented_transcript = transcript_processor.process_transcript(transcript)
            # This is stored in the state but not available to the agent withour retrieval
            state["segmented_transcript"] = segmented_transcript

            return f"""
Here are the summary segments in this video:\n {segmented_transcript.get_summaries()}

Which segment do you want to study?
            """
        else:
            return "Please provide a valid YouTube URL."

    return response


chat = gr.ChatInterface(
    fn=inference,
    type="messages",
)

chat.launch()

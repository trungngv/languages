import logging

import gradio as gr
from agents.single_agent import extract_user_intent
from dotenv import load_dotenv

from youtube_chat.agents import LanguageTeachingAgent, VideoProcessor
from youtube_chat.llms import OpenAIClient
from youtube_chat.youtube import extract_youtube_url

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

video_processor = VideoProcessor(OpenAIClient())
main_agent = LanguageTeachingAgent(OpenAIClient(model="gpt-4o"))

state = {
    "url": None,
    "segmented_transcript": None,
    "current_selected_segment": None,
}


def inference(message, history):
    response = main_agent.call(message=message, history=history, state=state)
    logger.info(f"Response: {response[0:100]} ...")

    user_intent = extract_user_intent(response)
    logger.info(f"User Intent: {user_intent}")

    if user_intent == "providing youtube url":
        youtube_url = extract_youtube_url(response)
        if youtube_url:
            state["url"] = youtube_url
            segmented_transcript = video_processor.process(youtube_url)
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

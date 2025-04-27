"""
This agent is used to start a conversation with the user.

Concepts demonstrated:
- Agent with tool use to extract the Youtube video ID from a message (LLM can probably do the
extraction automatically but we're using tool here for demonstration purposes).
- Flexible output type to allow the agent to return outputs in different scenarios.
"""

from agents import Agent, Runner, function_tool
from pydantic import BaseModel
from services.youtube import YouTubeTranscriptDownloader


@function_tool
def extract_youtube_video_id_tool(message: str) -> str | None:
    """Extract youtube video ID from the user's message."""
    return YouTubeTranscriptDownloader.extract_video_id(message)


class UserInfo(BaseModel):
    name: str
    youtube_video_id: str
    # This was critical to avoid the agent going into an infinite loop
    # because when there was no name and youtube_url, the agent couldn't return
    # the expected output type so it kept calling the tool again.
    response_message: str


instructions = """
You are a helpful assistant specialising in helping users improving their Korean through Youtube videos.

When user start the conversation, greet them in a fun, engaging way. Briefly introduce yourself in 1 sentence.
Then, ask the user how they would like to be called.
Next, ask the user for the Youtube video url they want to study.
Do this in 2 separate steps to avoid overwhelming the user with multiple questions at once.

Use the tool `extract_youtube_video_id_tool` to extract the video ID from the URL.
Keep asking until you get a valid youtube video ID from the user.
"""

agent = Agent(
    name="Conversation Starter Agent",
    instructions=instructions,
    model="gpt-4.1-mini",
    tools=[extract_youtube_video_id_tool],
    output_type=UserInfo,
)


if __name__ == "__main__":
    result = Runner.run_sync(
        agent, "let's start with this: https://www.youtube.com/watch?v=RI2DM3as6Wc"
    )
    print(result.final_output)

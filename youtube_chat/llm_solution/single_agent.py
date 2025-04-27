import re

from llm_solution.llms import OpenAIClient

system_prompt = """
You are a helpful assistant for learning Korean through YouTube videos.
Depending on the current state of the conversation, you will need to perform different actions.

1. If the user has not provided a Youtube Video URL, ask them to do so.
Respond with message as a string with following this exact format:

user_intent: providing youtube url
youtube_url: [the value of the youtube url]

2. Whenever the user selects a new segment to study, do the following.

- Identify the segment index based on the user request and retrieve its full content given in the conversation history
- Explain the first sentence of the segment, provide the following explanation:
    - An English translation
    - List key vocabulary with their meanings (include Hanja if applicable)
    - Explain common expressions or idioms (if any)
    - Note 1 -2 important grammar points 

    IMPORTANT: Note that the vocabulary, expressions/idioms, and grammar points should be selected based on the difficulty level of the text. I.e. if the text is advanced, the user is at the advanced level, so note more complex learning concepts.

- After providing the explanation, ask the user if they want to continue to the next sentence. If they agree, follow the same process again.
If they don't, say one encouraging sentence and ask if they would like to study a different segment.
"""


def extract_user_intent(message: str) -> str:
    user_intent_pattern = r"user_intent:\s*(.+)\n"
    match = re.search(user_intent_pattern, message)
    return match.group(1) if match else None


class LanguageTeachingAgent:
    def __init__(self, openai_client: OpenAIClient):
        self.openai_client = openai_client

    def call(
        self, message: str, history: list[dict[str, str]], state: dict[str, any]
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": system_prompt,
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

        return self.openai_client.chat(messages)

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()


class OpenAIClient:
    """
    A thin wrapper around the OpenAI API client that handles retries.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
    ):
        self.model = model
        self.max_retries = max_retries
        self._client = OpenAI()

    def call(
        self, system_prompt: str, user_prompt: str, output_model: BaseModel
    ) -> BaseModel:
        if user_prompt == "" and system_prompt == "":
            raise ValueError(
                "At least one of user_prompt or system_prompt must be provided."
            )

        for _ in range(self.max_retries):
            completion = self._client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=output_model,
            )
            result = completion.choices[0].message
            if result.parsed:
                return result.parsed
            else:
                print("Failure: ", completion)

        raise Exception("Max retries exceeded")

    def chat(self, messages: list) -> str:
        """
        Send a chat message to the OpenAI API and return the response.
        """
        print("chat messages: ", messages)
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
        )
        return response.choices[0].message.content

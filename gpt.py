from openai import AsyncOpenAI
import os

class OpenAIAPI:
    def __init__(self, model_name="gpt-3.5-turbo", max_tokens=2000):
        self.client = AsyncOpenAI(
            api_key="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"
        )

        self.model_name = model_name
        self.max_tokens = max_tokens

    async def proofread(self, text: str, history: list):
        response = await self.client.chat.completions.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": "Please proofread this speech transcription. It can be any language. It can be just a part of a sentence so do not capitalize the sentence. Please return only the proofreading results without any comments",
                },
                {"role": "user", "content": text},
            ],
        )

        choice = response.choices[0]

        if choice:
            return choice.message.content
        else:
            return ''
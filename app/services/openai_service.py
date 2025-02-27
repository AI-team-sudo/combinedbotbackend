from openai import OpenAI
from app.config import get_settings
from app.utils.prompts import SYSTEM_PROMPT, get_chat_prompt

settings = get_settings()

class OpenAIService:
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)

    async def generate_response(self, context: str, user_query: str):
        try:
            chat_prompt = get_chat_prompt(context, user_query)
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": chat_prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API call failed: {str(e)}")

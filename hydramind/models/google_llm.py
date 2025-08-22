import os
from dotenv import load_dotenv
import google.generativeai as genai
from config.keys import GOOGLE_API_KEY
from google.generativeai import GenerativeModel
import google.generativeai as genai

class GoogleLLM:
    def __init__(self, model_name="gemini-2.0-flash", token_limit = 1024,**kwargs):
        self.model_name = model_name

        genai.configure(api_key=GOOGLE_API_KEY)

        self.token_limit = token_limit

        # Anything else in kwargs goes to GenerativeModel
        self.client = genai.GenerativeModel(model_name=model_name, **kwargs)

    def chat(self, messages, max_tokens=None):
        """
        Converts OpenAI-style messages into a prompt that preserves distinct agent roles.
        Example input:
            {"role": "strategist", "content": "..."},
            {"role": "critic", "content": "..."},
            {"role": "user", "content": "..."}
        """
        prompt_lines = []

        for msg in messages:
            role = msg["role"].capitalize()
            content = msg["content"].strip()
            prompt_lines.append(f"{role}: {content}")

        full_prompt = "\n".join(prompt_lines)

        response = self.client.generate_content(
            full_prompt,
            generation_config={
                "max_output_tokens": max_tokens or self.token_limit
            }
        )

        return response.text.strip()
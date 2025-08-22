from .base_agent import Agent
from models.base_llm import LLMInterface

class InformantAgent(Agent):
    def __init__(self, llm: LLMInterface):
        role = (
            "You are an Informant. Your job is to gather and summarize all useful and relevant information "
            "that will help build a detailed and accurate plan to achieve the user's goal."
        )
        super().__init__("Informant", role, llm.model_name, llm.token_limit)
        self.llm = llm

    def respond(self, stage: str, user_input: str, shared_context=None) -> dict:
        chat_history = shared_context.get("chat_stream", [])
        long_term_memory = shared_context.get("long_term_memory", {})
        
        user_goal = long_term_memory.get("user_goal", "")

        system_prompt = (
            f"You are {self.name}. Extract and summarize all relevant, helpful, and factual information from the conversation "
            f"to help other agents build a good plan. Focus on context and knowledge helpful for planning."
        )

        prompt = f"""
User Goal:
{user_goal}

Chat History:
{'\n'.join([f"{m['role'].capitalize()}: {m['content']}" for m in chat_history])}

Provide a concise but complete summary of relevant knowledge, ideas, or constraints mentioned in the conversation.
"""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]

            response = self.llm.chat(messages, max_tokens=500)

            return {
                "role": "informant",
                "output": response,
                "update": {
                    "informant_info": response
                },
                "short_term_memory": {}
            }

        except Exception as e:
            print(f"[ERROR - {self.name}] {e}")
            return {
                "role": "informant",
                "output": "[ERROR] LLM failure.",
                "update": {},
                "short_term_memory": {}
            }

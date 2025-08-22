from .base_agent import Agent
from models.base_llm import LLMInterface

class StrategistAgent(Agent):
    def __init__(self, llm: LLMInterface):
        role = (
            "Use the information provided to create or revise a multi-step plan that achieves the user's goal. "
            "Incorporate any feedback or criticism to improve the plan. "
            "Be clear and sequential."
        )
        super().__init__("Strategist", role, llm.model_name, llm.token_limit)
        self.llm = llm

    def respond(self, stage: str, user_input: str, shared_context=None) -> dict:
        chat_history = shared_context.get("chat_stream", [])
        long_term_memory = shared_context.get("long_term_memory", {})
        short_term_memory = shared_context.get("short_term_memory", {})

        info = long_term_memory.get("informant_info", "")
        critique = short_term_memory.get("critique", "")

        system_prompt = (
            f"You are {self.name}, responsible for devising an actionable, clear plan using the latest information and feedback."
        )

        prompt = f"""
User Goal:
{user_input}

Relevant Information from Informant:
{info}

Critique from Critic Agent:
{critique}

Chat History:
{'\n'.join([f"{m['role'].capitalize()}: {m['content']}" for m in chat_history])}

Based on the information and critique above, provide a refined step-by-step plan.
"""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            response = self.llm.chat(messages, max_tokens=600)

            return {
                "role": "strategist",
                "output": response,
                "update": {
                    "final_plan": response
                },
                "short_term_memory": {
                    "plan": response
                }
            }

        except Exception as e:
            print(f"[ERROR - {self.name}] {e}")
            return {
                "role": "strategist",
                "output": "[ERROR] LLM failure.",
                "update": {},
                "short_term_memory": {"plan": "[ERROR] LLM failure."}
            }

from .base_agent import Agent
from models.base_llm import LLMInterface

class CriticAgent(Agent):
    def __init__(self, llm: LLMInterface):
        role = "Critically analyze the proposed plan and identify possible flaws, ambiguities, or missing considerations."
        super().__init__("Critic", role, llm.model_name, llm.token_limit)
        self.llm = llm

    def respond(self, stage: str, user_input: str, shared_context=None) -> dict:
        plan = shared_context.get("long_term_memory", {}).get("final_plan", "No plan provided.")
        prior_critique = shared_context.get("short_term_memory", {}).get("critic", "")
        chat_history = shared_context.get("chat_stream", [])

        system_prompt = (
            f"You are {self.name}. Your role is to critically analyze the user's proposed plan "
            f"and point out potential issues, missing details, or areas of ambiguity. "
            f"Be thorough, constructive, and analytical."
        )

        history_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in chat_history])

        prompt = f"""
User Goal:
{user_input}

Plan to Critique:
{plan}

Prior Critique (if any):
{prior_critique}

Chat History:
{history_str}

Please critique the plan thoroughly and provide suggestions for improvement.
"""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            response = self.llm.chat(messages, max_tokens=800)

            return {
                "role": "critic",
                "output": response,
                "update": {
                    "critic_comments": response
                },
                "short_term_memory": response
            }

        except Exception as e:
            print(f"[ERROR - {self.name}] {e}")
            return {
                "role": "critic",
                "output": "[ERROR] LLM failure.",
                "update": {},
                "short_term_memory": "[ERROR] LLM failure."
            }

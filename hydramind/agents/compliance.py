from .base_agent import Agent
from models.base_llm import LLMInterface

class ComplianceAgent(Agent):
    def __init__(self, llm: LLMInterface):
        role = (
            "Verify whether the proposed plan satisfies the user's original goal and adheres to all given constraints. "
            "Return a clear yes/no judgment followed by a justification."
        )
        super().__init__("Compliance", role, llm.model_name, llm.token_limit)
        self.llm = llm

    def respond(self, stage: str, user_input: str, shared_context=None) -> dict:
        chat_history = shared_context.get("chat_stream", [])
        revised_plan = shared_context.get("long_term_memory", {}).get("final_plan", "")

        system_prompt = (
            f"You are {self.name}, responsible for validating whether the current plan is compliant with the user's stated goal and constraints."
        )

        prompt = f"""
User Goal:
{user_input}

Final Plan:
{revised_plan}

Chat History:
{'\n'.join([f"{m['role'].capitalize()}: {m['content']}" for m in chat_history])}

Respond with a clear 'Yes' or 'No' on whether the plan satisfies the goal and constraints, followed by a short justification.
"""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            response = self.llm.chat(messages, max_tokens=400)

            return {
                "role": "compliance",
                "output": response,
                "update": {
                    "compliance_check": response
                },
                "short_term_memory": response
            }

        except Exception as e:
            print(f"[ERROR - {self.name}] {e}")
            return {
                "role": "compliance",
                "output": "[ERROR] LLM failure.",
                "update": {},
                "short_term_memory": "[ERROR] LLM failure."
            }

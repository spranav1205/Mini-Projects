from .base_agent import Agent 
from typing import Optional

class SummarizerAgent(Agent):
    def __init__(self, llm, name="Summarizer", model_name="gpt-4", token_limit=4096):
        super().__init__(
            name=name,
            role_description="Convert structured multi-phase plans into concise summaries and actionable task lists for the user.",
            model_name=model_name,
            token_limit=token_limit
        )
        self.llm = llm  # Expects an object with a .chat(messages) method

    def respond(self, stage: str, user_input: str, shared_context: Optional[dict] = None) -> dict:
        """
        Converts a detailed plan into a summary + actionable list.
        shared_context: Optional dict that may include `goal_description`
        """
        plan_text = user_input
        goal_description = shared_context.get("goal_description") if shared_context else None

        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a helpful assistant named {self.name}, and your job is to read a multi-phase plan and summarize it."
                    " Output a human-readable summary, followed by a bulleted list of tasks. Keep it short and friendly."
                )
            },
            {
                "role": "user",
                "content": f"Goal: {goal_description or 'N/A'}\n\nPlan:\n{plan_text}"
            }
        ]

        response = self.llm.chat(messages, max_tokens=800)

        self.add_to_short_term_memory(plan_text)
        self.add_to_short_term_memory(response)

        return {
            "role": "summarizer",
            "output": response
        }


    # TODO: This splitting does not work
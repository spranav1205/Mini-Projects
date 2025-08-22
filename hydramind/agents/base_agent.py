from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(self, name, role_description, model_name, token_limit=4096):
        self.name = name
        self.role_description = role_description
        self.model_name = model_name
        self.token_limit = token_limit

    def get_context(self, shared_context: dict) -> str:
        """
        Retrieve relevant short-term context from shared memory passed by the orchestrator.
        """
        return "\n".join(shared_context.get("short_term_memory", []))

    def format_prompt(self, stage: str, user_input: str, shared_context: dict) -> str:
        """
        Construct the prompt using centralized memory passed in.
        """
        return f"""You are {self.name}, an AI assistant with the role: {self.role_description}.

Conversation Stage: {stage.upper()}

User Input:
{user_input}

Recent Context:
{self.get_context(shared_context)}

Respond with your output for this stage.
"""

    @abstractmethod
    def respond(self, stage: str, user_input: str, shared_context: dict) -> str:
        """
        Subclasses must override this method to define how the agent responds.
        """
        raise NotImplementedError("Each agent must implement its own respond() method.")

    def __str__(self):
        return f"<Agent {self.name} | Model: {self.model_name}>"

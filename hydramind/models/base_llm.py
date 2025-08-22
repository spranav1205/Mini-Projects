from abc import ABC, abstractmethod

class LLMInterface(ABC):
    def __init__(self, model_name: str, token_limit: int = 4096):
        self.model_name = model_name
        self.token_limit = token_limit

    @abstractmethod
    def chat(self, messages: list[dict], max_tokens: int = 1024) -> str:
        """Return model's response to a list of messages."""
        pass

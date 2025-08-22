from agents.strategist import StrategistAgent
from agents.critic import CriticAgent
from agents.compliance import ComplianceAgent
from agents.informant import InformantAgent
from models.base_llm import LLMInterface

class PlannerAgent:
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        self.strategist = StrategistAgent(llm)
        self.critic = CriticAgent(llm)
        self.compliance = ComplianceAgent(llm)
        self.informant = InformantAgent(llm)

        # Persistent state across planning cycles
        self.chat_stream = []
        self.long_term_memory = {
            "user_goal": "",
            "informant_info": {},
        }

    def gather_context(self, user_input: str):
        info = self.informant.respond(
            user_input=user_input,
            stage="inform",
            shared_context={
                "chat_stream": self.chat_stream,
                "long_term_memory": self.long_term_memory,
                "short_term_memory": {}
            },
        )
        self.long_term_memory["user_goal"] = user_input
        self.long_term_memory["informant_info"] = info

    def run_planning_cycle(self, user_input: str, iterations: int = 5):
        # Step 1: Add user message to chat stream
        self.chat_stream.append({"role": "user", "content": user_input})

        # Step 2: Gather informant context (only once per run)
        self.gather_context(user_input)

        short_term_memory = {"criticisms": []}

        for step in range(iterations):
            print(f"\n--- Iteration {step + 1} ---")

            shared_context = {
                "chat_stream": self.chat_stream,
                "long_term_memory": self.long_term_memory,
                "short_term_memory": short_term_memory,
            }

            # 1. Strategist
            strategy = self.strategist.respond(stage="plan", user_input=user_input, shared_context=shared_context)
            self.chat_stream.append({"role": "strategist", "content": strategy["output"]})
            print("\n[Strategy]:", strategy["output"])

            # 2. Critic
            critique = self.critic.respond(stage="critique", user_input=strategy["output"], shared_context=shared_context)
            self.chat_stream.append({"role": "critic", "content": critique["output"]})
            short_term_memory["criticisms"].append(critique["output"])
            print("\n[Critique]:", critique["output"])

            # 3. Compliance
            compliance = self.compliance.respond(stage="compliance", user_input=strategy["output"], shared_context=shared_context)
            self.chat_stream.append({"role": "compliance", "content": compliance["output"]})
            print("\n[Compliance]:", compliance["output"])

        print("\n✅ Final plan:")
        print(strategy["output"])
        return strategy["output"]

import sys
if sys.prefix != sys.base_prefix:
    print("Virtual environment is active.")
else:
    print("Not in a virtual environment.")


from config.keys import GOOGLE_API_KEY

print(GOOGLE_API_KEY)  # Should work if .env is present

from models import GoogleLLM

from agents.strategist import StrategistAgent
from agents.critic import CriticAgent
from agents.compliance import ComplianceAgent
from agents.summarizer import SummarizerAgent
from agents.informant import InformantAgent

llm = GoogleLLM(model_name="gemini-1.5-flash")


from orchestrator.basic_planner import PlannerAgent

def test_planner():
    # Initialize with a mock or real LLM
    planner = PlannerAgent(llm)

    # Simulated user request
    user_input = "I want to plan a trip to 7 day trip to Japan next spring."

    # Run planner
    final_plan = planner.run_planning_cycle(user_input, iterations=3)

    print("\n🧠 Finalized Strategy Output:")
    print(final_plan)

if __name__ == "__main__":
    test_planner()

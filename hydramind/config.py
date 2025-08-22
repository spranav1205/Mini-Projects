"""
HydraMind Config: Agent-role mappings, model settings.
"""

AGENT_CONFIG = [
    {
        "name": "Informant",
        "role": "Provides factual background for tasks.",
        "model": "gemma:7b",
        "token_limit": 2048
    },
    {
        "name": "Strategist",
        "role": "Recommends actionable plans.",
        "model": "gpt-4o",
        "token_limit": 8192
    },
    {
        "name": "Critic",
        "role": "Critiques the proposed strategy.",
        "model": "mixtral:8x7b",
        "token_limit": 8192
    },
    {
        "name": "Auditor",
        "role": "Checks for user requirement adherence.",
        "model": "qwen:7b",
        "token_limit": 4096
    },
    {
        "name": "Summarizer",
        "role": "Summarizes agent outputs and conversation notes.",
        "model": "zephyr:7b",
        "token_limit": 2048
    }
]

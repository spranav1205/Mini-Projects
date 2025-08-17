from eval_engine import EvalEngine
import pandas as pd
import importlib

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_csv("list.csv")

sys.path.append("./agents/")

def round_robin(rounds: int):
    scores = {}
    for player in df["names"]:
        scores[player] = 0
    for i in range(rounds):
        evaluator = EvalEngine()
        keys = list(df["names"])
        for j, player1 in enumerate(keys):
            for player2 in keys[j + 1:]:  # Interesting fix
                try:
                    p1 = importlib.import_module(player1)
                    p2 = importlib.import_module(player2)
                    h, points = evaluator.evaluate(p1.Agent, p2.Agent)
                    scores.update({player1: (scores[player1] + points[0])})
                    scores.update({player2: (scores[player2] + points[1])})

                except ModuleNotFoundError as e:

                    print(f"Error: {e}")

    return scores


total = round_robin(15)
print(total["TitForTat"], total["Random"])
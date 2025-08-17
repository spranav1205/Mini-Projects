import numpy as np
import random

class EvalEngine:
    def __init__(self):
        self.rounds = 200
        self.streak = 0
        self.error = False
        self.error_percent = 2
        self.score = np.array([0, 0])
        self.history = {"round":0}

    def matrix(self, x, y):
        score_matrix = {
            1:
                {
                    1: np.array([30 + 5 * (self.streak // 5), 30 + 5 * (self.streak // 5)]),
                    -1: np.array([-5 * (self.streak // 5), 40])
                },
            -1:
                {
                    1: np.array([40, -5 * (self.streak // 5)]),
                    -1: np.array([-5 * (self.streak // 5), -5 * (self.streak // 5)]),
                }
        }
        return score_matrix[x][y]

    def evaluate(self, player_a, player_b):

        instanceA = player_a(1)
        instanceB = player_b(2)

        for i in range(self.rounds):

            self.history.update({"round":i})
            x = instanceA.next_move(self.history)
            y = instanceB.next_move(self.history)

            if random.randint(0,100) < self.error_percent : x = -x
            if random.randint(0, 100) < self.error_percent: y = -y

            self.history[i] = {
                1: x,
                2: y
            }

            if x == 1 & y == 1:
                self.score += self.matrix(x, y)
                self.streak += 1
            elif x == 1 & y == -1:
                self.score += self.matrix(x, y)
                self.streak = self.streak // 2 if self.error is True else 0
            elif x == -1 & y == -1:
                self.score += self.matrix(x, y)
                self.streak = self.streak // 2 if self.error is True else 0
            elif x == -1 & y == -1:
                self.score += self.matrix(x, y)
                self.streak = 0

        print(self.score)
        return self.history, self.score

from Base_Agent import base_agent


class Agent(base_agent):
    def __init__(self,id):
        super().__init__(id)
        self.op_id = 2 if self.id == 1 else 1

    def next_move(self, dict):
        if dict["round"] == 0:
            return 1
        else:
            return dict[dict["round"]-1][self.op_id]

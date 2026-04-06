import numpy as np

class RLAgent:
    def __init__(self):
        # Simple Q-table (10 states x 3 actions)
        self.q_table = np.zeros((10, 3))

    def get_state(self, prob):
        return min(int(prob * 10), 9)

    def get_action(self, state):
        return np.argmax(self.q_table[state])

    def decide(self, prob):
        state = self.get_state(prob)
        action = self.get_action(state)

        actions = {
            0: "Continue Operation",
            1: "Schedule Maintenance",
            2: "Shutdown Immediately"
        }

        return actions[action]
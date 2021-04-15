from util import *


class Evaluator:
    def __init__(self, agent):
        self.agent = agent

    def eval(self, n_trials=100):
        total_rewards = []
        for i in range(n_trials):
            _, rewards, _ = self.agent.simulate()
            tr = np.mean(rewards)
            total_rewards.append(tr)
        return total_rewards, np.mean(total_rewards), np.std(total_rewards)
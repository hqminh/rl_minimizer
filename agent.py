from util import *
from policy import *


class DQN_Agent:
    def __init__(self, env, policy, memory):
        self.env = env
        self.policy = policy
        self.memory = memory

    # states: (n by) w by (k * vocab_size)
    # rewards: (n by) 1
    # actions
    def simulate(self, policy, id=None):
        states, rewards, actions, next_states, terminates = [], [], [], [], []
        s = self.env.reset(id)
        terminate = False
        while not terminate:
            a = policy(s.unsqueeze(0)).squeeze()
            next_state, r, terminate = self.env.step(a)
            states.append(s)
            rewards.append(r)
            actions.append(onehots(a, self.env.w))
            next_states.append(next_state)
            terminates.append(terminate)
            s = next_state
        return states, rewards, actions, next_states, terminates

    def eval(self):
        total_rewards = []
        _, rewards, _, _, _ = self.simulate(self.policy, id=None)
        tr = 1.0 - np.mean(rewards)
        total_rewards.append(tr)
        return total_rewards, np.mean(total_rewards)

    def train_qnet(self, n_epoch=1000, batch_size=32, sync_interval=10, eval_interval=10):
        # burn in
        random_policy = RandomPolicy(self.env.w, self.env.name)
        print('Random Burn-in')
        while len(self.memory.buffer) < self.memory.burn_in:
            states, rewards, actions, next_states, terminates = self.simulate(random_policy)
            for i in range(len(states)):
                self.memory.append((states[i], actions[i], rewards[i], next_states[i], terminates[i]))

        print('Training DQN')
        total_rewards, mean_rewards = self.eval()
        print(0, mean_rewards)
        for epoch in range(n_epoch):
            #print(epoch)
            self.policy.fit(self.memory, batch_size=batch_size, n_epoch=10)
            states, rewards, actions, next_states, terminates = self.simulate(self.policy)
            for i in range(len(states)):
                self.memory.append((states[i], actions[i], rewards[i], next_states[i], terminates[i]))
            if (epoch + 1) % sync_interval == 0:
                self.policy.clone_to_target()
            if (epoch + 1) % eval_interval == 0:
                total_rewards, mean_rewards = self.eval()
                print(epoch, mean_rewards)
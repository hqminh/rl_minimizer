from util import *
from policy import *
from matplotlib import pyplot as plt

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
            a = policy(s.unsqueeze(0)).view(-1)
            next_state, r, terminate = self.env.step(a)
            states.append(s)
            rewards.append(r)
            actions.append(onehots(a, self.env.w))
            next_states.append(next_state)
            terminates.append(terminate)
            s = next_state
        return states, rewards, actions, next_states, terminates

    def eval(self):
        density = []
        _, _, _, _, _ = self.simulate(self.policy, id=None)
        for i in range(self.env.loc.shape[0]):
            density.append(torch.mean(self.env.loc[i]))

        return density, np.mean(density)

    def train_qnet(self, n_epoch=1000, batch_size=32, sync_interval=10, eval_interval=10):
        # burn in
        random_policy = RandomPolicy(self.env.w)
        print('Random Burn-in')
        while len(self.memory.buffer) < self.memory.burn_in:
            states, rewards, actions, next_states, terminates = self.simulate(random_policy)
            for i in range(len(states)):
                self.memory.append((states[i], actions[i], rewards[i], next_states[i], terminates[i]))

        print('Training DQN')
        total_rewards, mean_rewards = self.eval()
        # x = [0]
        # ys = [[density] for density in total_rewards]
        # y = [mean_rewards]
        x = []
        ys = [[] for _ in total_rewards]
        y = []
        best_y = 1000
        convergence_counter = 0
        for epoch in range(n_epoch):
            self.policy.fit(self.memory, batch_size=batch_size, n_epoch=10)
            states, rewards, actions, next_states, terminates = self.simulate(self.policy)
            for i in range(len(states)):
                self.memory.append((states[i], actions[i], rewards[i], next_states[i], terminates[i]))
            if (epoch + 1) % sync_interval == 0:
                self.policy.clone_to_target()
            if (epoch + 1) % eval_interval == 0:
                total_rewards, mean_rewards = self.eval()
                print(epoch, mean_rewards)
                x.append(epoch)
                y.append(mean_rewards)
                plt.figure()
                plt.xlabel('No. epochs')
                plt.ylabel('Density')
                plt.plot(x, y, label='Average')
                for j, density in enumerate(total_rewards):
                    ys[j].append(density)
                    plt.plot(x, ys[j], label=f'Seq {j+1}')
                plt.legend()
                plt.savefig('./rl_performance_no_sigmoid_w14k6_large_h85.png')
                if mean_rewards < best_y:
                    convergence_counter = 0
                    best_y = mean_rewards
                else:
                    convergence_counter += 1
            if convergence_counter == 200:
                break
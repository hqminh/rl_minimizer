from util import *
import random


class Memory:
    def __init__(self, memory_size=50000, burn_in=10000):
        self.burn_in = burn_in
        self.max_size = memory_size
        self.buffer = deque(maxlen=memory_size)

    def sample(self, batch_size=32):
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        batch = random.sample(self.buffer, batch_size)

        for item in batch:
            state, action, reward, next_state, done = item
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        return torch.stack(states), torch.stack(actions), torch.tensor(rewards), torch.stack(next_states), dones

    def append(self, transition):
        self.buffer.append(transition)
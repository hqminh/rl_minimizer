from util import *


class RandomPolicy(nn.Module):
    def __init__(self, w):
        super(RandomPolicy, self).__init__()
        self.w = w

    # s : batch size (by n) by w by (k * vocab)
    # a: return
    def forward(self, s):
        return torch.randint(0, self.w, (1, s.shape[1]))



class RankNet(nn.Module):
    def __init__(self, k, w, vocab_size, hidden_dim):
        super(RankNet, self).__init__()
        self.k = k
        self.w = w
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(self.k * self.vocab_size, 2 * self.hidden_dim),
            nn.Tanh(),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            #nn.Sigmoid()
        )

    # context: batch_size (by n) by w by (k * vocab_size) tensor
    # return q values
    def forward(self, context):
        return self.net(context).squeeze(-1)


class WindowNet(nn.Module):
    def __init__(self, k, w, vocab_size, hidden_dim):
        super(WindowNet, self).__init__()
        self.k = k
        self.w = w
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(self.w * self.k * self.vocab_size, 2 * self.hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.w),
            nn.Softmax(dim=-1)
        )

    # context: batch_size (by n) by w by (k * vocab_size) tensor
    # return q values: batch_size (by n) by w
    def forward(self, context):
        return self.net(torch.flatten(context, start_dim=-2))


class QNet(nn.Module):
    def __init__(self, w, k, gamma=0.99, vocab_size=4, hidden_dim=100):
        super(QNet, self).__init__()
        self.w = w
        self.k = k
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        # self.q = WindowNet(self.k, self.w, self.vocab_size, self.hidden_dim)
        # self.q_target = WindowNet(self.k, self.w, self.vocab_size, self.hidden_dim)
        self.q = RankNet(self.k, self.w, self.vocab_size, self.hidden_dim)
        self.q_target = RankNet(self.k, self.w, self.vocab_size, self.hidden_dim)
        self.clone_to_target()
        self.opt = Adam(self.q.parameters(), lr=2e-5)

    # context: batch_size (by n) by w by (k * vocab_size) tensor
    # return action: batch_size (by n)
    def forward(self, context):
        q_value = self.q(context)
        return torch.argmax(q_value, dim=-1)

    # states: (tensor) batch_size (by n) by w by (k * vocab_size)
    # next_states: (tensor) batch_size (by n) by w by (k * vocab_size)
    # actions: (tensor) batch_size (by n) by w
    # rewards: (tensor) batch_size by 1
    def dqn_loss(self, states, actions, rewards, next_states):
        # q_target: (tensor) batch_size (by n)
        # q_value: (tensor) batch_size (by n)
        q_target = torch.max(self.q_target(next_states), dim=-1).values
        q_value = torch.sum(self.q(states) * actions, dim=-1)
        mean_q_target = torch.mean(q_target, dim=1)
        mean_q_value = torch.mean(q_value, dim=1)
        loss = (rewards + self.gamma * mean_q_target - mean_q_value) ** 2
        return torch.mean(loss, dim=0)

    def clone_to_target(self):
        self.q_target.load_state_dict(deepcopy(self.q.state_dict()))

    # Memory bank
    def fit(self, memory, batch_size=10, n_epoch=100):
        for i in range(n_epoch):
            self.opt.zero_grad()
            states, actions, rewards, next_states, _ = memory.sample(batch_size=batch_size)
            loss = self.dqn_loss(states, actions, rewards, next_states)
            loss.backward()
            self.opt.step()
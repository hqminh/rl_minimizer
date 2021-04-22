from util import *


class SeqEnvironment:
    def __init__(self, env_arg):
        self.name = env_arg['env_name']
        self.w = env_arg['window_size']
        self.k = env_arg['kmer_size']
        self.l = env_arg['sequence_length']
        self.n = env_arg['num_sequence']
        self.w_size = self.w + self.k - 1
        self.n_window = self.l - self.w_size
        self.vocab = env_arg['sequence_vocab']
        self.vocab_size = len(self.vocab)
        self.vocab_prob = env_arg['sequence_vocab_probability']
        self.seq_list = np.random.choice(len(self.vocab), (self.n, self.l), p=self.vocab_prob)

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError


class SingleSeqEnvironment(SeqEnvironment):
    def __init__(self, env_arg):
        super(SingleSeqEnvironment, self).__init__(env_arg)
        self.curr_seq = self.seq_list[0]
        self.loc = []
        self.curr_loc = 0
        self.state = self.get_window(self.curr_loc)

    def get_window(self, loc):
        return self.curr_seq[loc: loc + self.w + self.k - 1]

    def reset(self, id=None):
        id = np.random.randint(self.n) if id is None else id
        self.curr_seq = self.seq_list[np.random.randint(id)]
        self.loc = []
        self.curr_loc = 0
        self.state = self.get_window(self.curr_loc)
        return window_to_multihot(self.state, self.w, self.k, len(self.vocab))

    def step(self, action):
        new_loc = self.curr_loc + action
        if new_loc not in self.loc:
            self.loc.append(self.curr_loc + action)
            reward = 0
        else:
            reward = 1
        self.curr_loc += 1
        self.state = self.get_window(self.curr_loc)
        self.state = window_to_multihot(self.state, self.w, self.k, len(self.vocab))
        return self.state, reward, int(self.curr_loc + self.w + self.k > self.l)


class MultiSeqEnvironment(SeqEnvironment):
    def __init__(self, env_arg):
        super(MultiSeqEnvironment, self).__init__(env_arg)
        self.loc = np.zeros((self.n, self.l))
        self.dataset = self.generate_tensor_data(self.seq_list)
        self.curr_loc = 0
        self.state = self.get_window(self.curr_loc)

    def get_window(self, loc):
        return self.dataset[:, loc]

    def generate_tensor_data(self, seq_list):
        dataset = torch.zeros(self.n, self.n_window, self.w, self.k * self.vocab_size)
        for i in range(self.n):
            for loc in range(self.n_window):
                window = seq_list[i, loc: loc + self.w_size]
                dataset[i, loc] = window_to_multihot(window, self.w, self.k, self.vocab_size)
        return dataset

    # state is [n_seq by w by (k * vocab_size)]
    def reset(self, id=None):
        self.loc = np.zeros((self.n, self.l))
        self.curr_loc = 0
        self.state = self.get_window(self.curr_loc)
        return self.state

    # action is [n_seq]
    def step(self, action):
        new_loc = self.curr_loc + action
        reward = 0.0
        for i in range(new_loc.shape[0]):
            reward += self.loc[i, new_loc[i]]
            self.loc[i, new_loc[i]] = 1
        self.curr_loc += 1
        self.state = self.get_window(self.curr_loc)
        return self.state, reward / self.n, int(self.curr_loc >= self.n_window - 1)
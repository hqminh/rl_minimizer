from util import *


class SeqEnvironment:
    def __init__(self, env_arg):
        self.w = env_arg['window_size']
        self.k = env_arg['kmer_size']
        self.l = env_arg['sequence_length']
        self.n = env_arg['num_sequence']
        self.w_size = self.w + self.k - 1
        self.n_window = self.l - self.w_size
        self.vocab = env_arg['sequence_vocab']
        self.vocab_size = len(self.vocab)
        self.vocab_prob = env_arg['sequence_vocab_probability']
        self.seq_list = [
            torch.tensor(np.random.choice(len(self.vocab), (1, self.l), p=self.vocab_prob[i].numpy()))
            for i in range(self.vocab_prob.shape[0])
        ]
        self.seq_list = torch.cat(self.seq_list)

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError


class MultiSeqEnvironment(SeqEnvironment):
    def __init__(self, env_arg):
        super(MultiSeqEnvironment, self).__init__(env_arg)
        self.loc = torch.zeros((self.n, self.l))
        self.dataset = self.generate_tensor_data(self.seq_list)
        self.kmer_dataset = self.generate_kmer_data(self.seq_list)
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

    def generate_kmer_data(self, seq_list):
        kmer_dataset = []
        for i in range(self.n):
            for j in range(self.l - self.k + 1):
                kmer_dataset.append(kmer_to_multihot([seq_list[i, j + t] for t in range(self.k)]))
        return kmer_dataset

    # state is [n_seq by w by (k * vocab_size)]
    def reset(self, id=None):
        self.loc = torch.zeros((self.n, self.l))
        self.curr_loc = 0
        self.state = self.get_window(self.curr_loc)
        return self.state

    # action is [n_seq]
    def step(self, action):
        new_loc = self.curr_loc + action
        reward = torch.zeros(new_loc.shape[0])
        for i in range(new_loc.shape[0]):
            reward[i] += self.loc[i, new_loc[i]]
            self.loc[i, new_loc[i]] = 1
        self.curr_loc += 1
        self.state = self.get_window(self.curr_loc)
        return self.state, torch.mean(reward), int(self.curr_loc >= self.n_window - 1)

    def quick_step(self, policy):
        kmer_score = dict()
        for i in range(self.n):
            curr_window = deque(maxlen=self.w)
            for kmer in self.kmer_dataset:
                if kmer not in kmer_score:
                    kmer_score[kmer] = policy(kmer)
                curr_window.append(kmer_score[kmer])



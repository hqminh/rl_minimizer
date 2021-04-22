import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from policy import *
from rembo import REMBO
from matplotlib import pyplot as plt

class Simulator(nn.Module):
    def __init__(self, env_arg):
        super(Simulator, self).__init__()
        self.w = env_arg['window_size']
        self.k = env_arg['kmer_size']
        self.l = env_arg['sequence_length']
        self.n = env_arg['num_sequence']
        self.n_window = self.l - self.w - self.k + 1
        self.vocab = env_arg['sequence_vocab']
        self.vocab_size = len(self.vocab)
        self.vocab_prob = env_arg['sequence_vocab_probability']
        self.rank_net = RankNet(self.k, self.w, len(self.vocab), env_arg['hidden_dim'])
        self.rank_net_weight = self.rank_net.state_dict()
        self.rank_net_weight_tensor = ensure_not_1D(state_dict_to_tensor(self.rank_net_weight))
        self.seq_list = np.random.choice(self.vocab_size , (self.n, self.l), p=self.vocab_prob)
        self.seq_data = self.generate_tensor_data(self.seq_list)
        self.offset = torch.arange(self.n_window).repeat(self.n, 1)

    # Dataset = tensor (n_seq, l-w-k+1, w, k * vocab_size)
    def generate_tensor_data(self, seq_list):
        dataset = torch.zeros(self.n, self.l - self.w - self.k + 1, self.w, self.k * self.vocab_size)
        for i in range(self.n):
            for loc in range(self.n_window):
                window = seq_list[i, loc: loc + self.w + self.k - 1]
                dataset[i, loc] = window_to_multihot(window, self.w, self.k, self.vocab_size)
        return dataset

    def forward(self, net_weight):
        self.rank_net_weight_tensor = net_weight
        tensor_to_state_dict(self.rank_net_weight_tensor, self.rank_net_weight)
        self.rank_net.load_state_dict(self.rank_net_weight)
        rank = self.rank_net(self.seq_data)
        rank = torch.argmin(rank, dim=-1)
        rank += self.offset
        density = 0.0
        for i in range(self.n):
            density += torch.numel(torch.unique(rank[i])) / self.l
        return density / self.n

def main():
    env_arg = {
        'env_name': 'MultiSeq',
        'window_size': 7,
        'kmer_size': 3,
        'sequence_length': 1000,
        'num_sequence': 10,
        'sequence_vocab': ['A', 'T', 'G', 'C'],
        'sequence_vocab_probability': [0.25, 0.25, 0.25, 0.25],
        'hidden_dim': 50,
    }
    print('Generating Dataset')
    sim = Simulator(env_arg)

    n_dims = torch.numel(sim.rank_net_weight_tensor)
    d_embedding = 10
    n_trials = 1250
    batch_size = 8
    original_boundaries = np.array([[-1, 1]] * n_dims)
    print("original_boundaries.shape: {}".format(original_boundaries.shape))
    opt = REMBO(original_boundaries, d_embedding)

    print('Running BO')
    x = []
    y = []
    for i in range(n_trials):
        print(f'Trial {i}')
        X_queries, X_queries_embedded = opt.select_query_point(batch_size=batch_size)

        # Ensure not 1D (i.e. size (D,))
        X_queries = ensure_not_1D(X_queries)

        # Evaluate the batch of query points 1-by-1
        for row_idx in range(len(X_queries)):
            X_query = X_queries[row_idx]
            X_query_embedded = X_queries_embedded[row_idx]

            # Ensure no 1D tensors (i.e. expand tensors of size (D,))
            X_query = ensure_not_1D(X_query)
            X_query_embedded = ensure_not_1D(X_query_embedded)
            y_query = -sim(X_query)
            opt.update(X_query, y_query, X_query_embedded)

        x.append((i + 1) * batch_size)
        y.append(-opt.best_value())
        print(f'best density value: {y[-1]}')
        print("---------------------")
        plt.figure()
        plt.xlabel('No. queries')
        plt.ylabel('Best density')
        plt.plot(x, y)
        plt.savefig('./bo_performance_10seq.png')


if __name__ == '__main__':
    main()

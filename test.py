from environment import *
from agent import *
from evaluator import *
from policy import *
from memory import *

def main():
    torch.manual_seed(2603)
    np.random.seed(2603)
    n_seq = 1
    vocab = ['A', 'T', 'G', 'C']
    env_arg = {
        'window_size': 14,
        'kmer_size': 6,
        'sequence_length': 10000,
        'num_sequence': 1,
        'sequence_vocab': vocab,
        #'sequence_vocab_probability': torch.softmax(torch.randn((n_seq, len(vocab))), dim=1)
        'sequence_vocab_probability': (1.0 / len(vocab)) * torch.ones((n_seq, len(vocab)))
    }
    env = MultiSeqEnvironment(env_arg)
    policy = QNet(env.w, env.k, gamma=1, vocab_size=len(env.vocab), hidden_dim=85)
    memory = Memory(memory_size=10000, burn_in=1000)
    agent = DQN_Agent(env, policy, memory)
    agent.train_qnet(n_epoch=10000, batch_size=5, sync_interval=50)


def install(package):
    subprocess.check_call([sys.executable, "-q", "-m", "pip", "install", package])


if __name__ == '__main__':
    q = deque(maxlen=10)
    for i in range(11):
        print(i)
        q.append(i)
    print(q)

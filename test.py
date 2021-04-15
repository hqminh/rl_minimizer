from environment import *
from agent import *
from evaluator import *
from policy import *
from memory import *
from pprint import pprint

def main():
    env_arg = {
        'env_name': 'MultiSeq',
        'window_size': 7,
        'kmer_size': 3,
        'sequence_length': 1000,
        'num_sequence': 10,
        'sequence_vocab': ['A', 'T', 'G', 'C'],
        'sequence_vocab_probability': [0.55, 0.15, 0.15, 0.15]
    }
    pprint(env_arg)
    env = MultiSeqEnvironment(env_arg)
    policy = QNet(env.w, env.k, gamma=1, vocab_size=len(env.vocab), hidden_dim=50, env_name= env.name)
    memory = Memory(memory_size=10000, burn_in=100)
    agent = DQN_Agent(env, policy, memory)
    agent.train_qnet(n_epoch=10000, batch_size=32, sync_interval=50)


if __name__ == '__main__':
    main()
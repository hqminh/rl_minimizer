import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from torch.optim import Adam
from collections import deque


# Return 1 by vocab_size
def onehot(t, vocab_size=4):
    v = torch.zeros(vocab_size)
    v[t] = 1
    return v

# Return n by vocab_size
def onehots(T, vocab_size=4):
    v = torch.zeros(T.shape[0], vocab_size)
    for i in range(T.shape[0]):
        v[i] = onehot(T[i], vocab_size)
    return v

# Return 1 by (k * vocab_size)
def kmer_to_multihot(kmer, vocab_size=4):
    v = []
    for c in kmer:
        v.append(onehot(c, vocab_size))
    return torch.cat(v)


# Return n_kmer by (k * vocab_size)
def kmers_to_multihot(kmers, vocab_size=4):
    v = torch.zeros(kmers.shape[0], kmers.shape[1], vocab_size)
    for i in range(kmers.shape[0]):
        for j in range(kmers.shape[1]):
            v[i, j] = onehot(kmers[i, j], vocab_size)
    return v.view(kmers.shape[0], -1)

# Return w by (k * vocab_size)
def window_to_multihot(window, w, k, vocab_size=4):
    v = []
    for i in range(w):
        kmer = window[i: i + k]
        v.append(kmer_to_multihot(kmer, vocab_size))
    return torch.stack(v)


# Return n_window by w by (k * vocab_size)
def windows_to_multihot(windows, w, k, vocab_size=4):
    v = []
    for i in range(w):
        kmer = windows[:, i: i + k]
        v.append(kmers_to_multihot(kmer, vocab_size))

    return torch.stack(v).transpose(0, 1)
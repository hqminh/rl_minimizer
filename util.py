import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from torch.optim import Adam
from collections import deque
from pprint import pprint
import subprocess
import sys, os
import itertools

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


def state_dict_to_tensor(state_dict):
    return torch.cat([t.view(-1) for t in state_dict.values()])


def tensor_to_state_dict(w, state_dict):
    w = w.squeeze()
    curr = 0
    for k in state_dict:
        tsize = torch.numel(state_dict[k])
        state_dict[k] = w[curr: curr + tsize].reshape(state_dict[k].shape)
        curr += tsize

def ensure_not_1D(x):
    if x.ndim == 1:
        if isinstance(x, np.ndarray):
            x = torch.tensor(np.expand_dims(x, axis=0))
        elif isinstance(x, torch.Tensor):
            x = x.unsqueeze(0)
    return x
#
# OtterTune - prioritized_replay_memory.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
# from: https://github.com/KqSMea8/CDBTune
# Zhang, Ji, et al. "An end-to-end automatic cloud database tuning system using
# deep reinforcement learning." Proceedings of the 2019 International Conference
# on Management of Data. ACM, 2019

import random
import pickle
import numpy as np


class SumTree(object):
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.num_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.num_entries < self.capacity:
            self.num_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return [idx, self.tree[idx], self.data[data_idx]]


class PrioritizedReplayMemory(object):

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.e = 0.01  # pylint: disable=invalid-name
        self.a = 0.6  # pylint: disable=invalid-name
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001

    def _get_priority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        # (s, a, r, s, t)
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def __len__(self):
        return self.tree.num_entries

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
        return batch, idxs

        # sampling_probabilities = priorities / self.tree.total()
        # is_weight = np.power(self.tree.num_entries * sampling_probabilities, -self.beta)
        # is_weight /= is_weight.max()

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def save(self, path):
        f = open(path, 'wb')
        pickle.dump({"tree": self.tree}, f)
        f.close()

    def load_memory(self, path):
        with open(path, 'rb') as f:
            _memory = pickle.load(f)
        self.tree = _memory['tree']

    def get(self):
        return pickle.dumps({"tree": self.tree})

    def set(self, binary):
        self.tree = pickle.loads(binary)['tree']

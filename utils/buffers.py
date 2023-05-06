# part of the code are from https://github.com/hill-a/stable-baselines/
# https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/common/buffers.py
# https://github.com/BY571/Soft-Actor-Critic-and-Extensions
# https://github.com/rlcode/per
import random
import numpy as np
from typing import List, Union
from collections import namedtuple

from utils.segment_tree import SumSegmentTree, MinSegmentTree

# Transition = namedtuple(
#     'Transition', ('obs_sp', 'score_binary', 'v_act_idx', 'selected_act_idx', 'rwd',
#                    'next_obs_sp', 'next_score_binary', 'done')
# )
# 'v_act_idx', 'tmpl_dist', 'obj1_dist', 'obj2_dist', 'selected_act_idx',
Transition = namedtuple(
    'Transition', ('obs_sp', 'score', 'v_act_idx', 'selected_act_idx', 'next_if_valid', 'rwd',
                   'next_obs_sp', 'next_score', 'next_v_act_idx', 'done')
)


class ReplayBuffer(object):
    def __init__(self, size: int):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self) -> int:
        return len(self._storage)

    def add(self, *args):
        data = Transition(*args)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes: Union[List[int], np.ndarray]):
        return [self._storage[i] for i in idxes]

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args):
        idx = self._next_idx
        super().add(*args)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        idxes = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            idxes.append(idx)
        return idxes

    def sample(self, batch_size, beta):
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights, previous_priorities = [], []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            previous_priorities.append(self._it_sum[idx])
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple([list(encoded_sample), previous_priorities, weights, idxes])

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


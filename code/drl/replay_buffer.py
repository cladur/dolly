import numpy as np
import random
import torch
from collections import deque


class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque()
        self.num_exp = 0

    def append(self, state, action, reward, step, new_state):
        experience = (state, action, reward, step, new_state)
        if self.num_exp < self.buffer_size:
            self.buffer.append(experience)
            self.num_exp += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.buffer_size

    def count(self):
        return self.num_exp

    def sample(self, batch_size, device):
        if self.num_exp < batch_size:
            batch = random.sample(self.buffer, self.num_exp)
        else:
            batch = random.sample(self.buffer, batch_size)

        item_count = 5
        res = []
        for i in range(5):
            k = torch.stack(tuple(item[i] for item in batch), dim=0)
            res.append(k.to(device))
        return res[0], res[1], res[2], res[3], res[4]

        state, action, reward, step, new_state = map(np.stack, zip(*batch))
        return state, action, reward, step, new_state

    def clear(self):
        self.buffer = deque()
        self.num_exp = 0

import numpy as np
import random
from collections import deque

class ReplayBuffer(object):
    def __init__(self, buffer_size, name_buffer=''):
        self.buffer_size=buffer_size  #choose buffer size
        self.num_exp=0
        self.buffer=deque()

    def add(self, state, action, reward, step, new_state):
        experience=(state, action, reward, step, new_state)
        if self.num_exp < self.buffer_size:
            self.buffer.append(experience)
            self.num_exp +=1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.buffer_size

    def count(self):
        return self.num_exp

    def sample(self, batch_size):
        if self.num_exp < batch_size:
            batch=random.sample(self.buffer, self.num_exp)
        else:
            batch=random.sample(self.buffer, batch_size)

        state, action, reward, step, new_state = map(np.stack, zip(*batch))

        return state, action, reward, step, new_state

    def clear(self):
        self.buffer = deque()
        self.num_exp=0

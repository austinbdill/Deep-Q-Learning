from collections import deque
from random import sample

class ReplayBuffer(object):

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, s, a, s_prime, r, done):
        self.memory.append((s, a, s_prime, r, done))

    def sample(self, batch_size):
        return sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

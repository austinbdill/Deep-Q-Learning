import numpy as np

class ReplayBuffer(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = [None] * (self.capacity)
        self.i = 0
        self.total_seen = 0

    def push(self, s, a, s_prime, r, done):
        self.memory[self.i] = (s, a, s_prime, r, done)
        self.i = (self.i + 1) % self.capacity
        self.total_seen += 1

    def get_sample(self, idx):
        return self.memory[idx]

    def sample(self, batch_size):
        sample_idx = np.random.randint(3, min(self.capacity, self.total_seen), size=batch_size)
        sample = [self.get_sample(i) for i in sample_idx]
        return sample

    def __len__(self):
        return min(self.capacity, self.total_seen)

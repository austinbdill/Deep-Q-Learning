import torch
import numpy as np

class ReplayBuffer(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = [None] * (self.capacity+3)
        self.i = 0
        self.total_seen = 0

    def push(self, f, a, f_prime, r, done):
        f = np.mean(f, -1).astype(np.uint8)
        f = f[::2, ::2]
        f_prime = np.mean(f_prime, -1).astype(np.uint8)
        f_prime = f_prime[::2, ::2]
        self.memory[self.i] = (f, a, f_prime, r, done)
        self.i = (self.i + 1) % self.capacity
        self.total_seen += 1
        
    def get_state_and_successor(self, idx):
        s = np.zeros((105, 80, 4))
        s_prime = np.zeros((105, 80, 4))
        for i, j in zip(range(idx-3, idx+1), range(4)):
            f, _, f_prime, _, _ = self.memory[i]
            s[:, :, j] = f
            s_prime[:, :, j] = f_prime
        s = s.transpose(2, 0, 1)
        s = torch.from_numpy(s).unsqueeze(0).float() / 255.0
        s_prime = s_prime.transpose(2, 0, 1)
        s_prime = torch.from_numpy(s_prime).unsqueeze(0).float() / 255.0
        return s, s_prime

    def get_sample(self, idx):
        _, a, _, r, done = self.memory[idx]
        s, s_prime = self.get_state_and_successor(idx)
        return s, a, s_prime, r, done

    def sample(self, batch_size):
        sample_idx = np.random.randint(3, min(self.capacity, self.total_seen), size=batch_size)
        sample = [self.get_sample(i) for i in sample_idx]
        return sample

    def __len__(self):
        return min(self.capacity, self.total_seen)

import numpy as np
import torch

class NoiseGenerator:
    def __init__(self, seed=123):
        self.seed = seed

    def get(self, index, dim):
        rng = np.random.RandomState(self.seed + index)
        return torch.tensor(rng.randn(dim), dtype=torch.float32)

    def sample_index(self):
        return np.random.randint(0, 10**8)  
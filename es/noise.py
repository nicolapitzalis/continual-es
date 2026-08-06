import numpy as np
import torch


class NoiseGenerator:
    def __init__(self, seed):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def get(self, index, dim):
        rng = np.random.RandomState(self.seed + index)
        return torch.tensor(rng.randn(dim), dtype=torch.float32)

    def sample_indices_and_episode_seeds(self, count):
        indices = self.rng.integers(0, 10**8, size=count).tolist()
        episode_seeds = self.rng.integers(0, 2**32 - 1, size=count).tolist()
        return indices, episode_seeds


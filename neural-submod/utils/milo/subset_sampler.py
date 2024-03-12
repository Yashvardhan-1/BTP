import random
import torch 

class RandomSubsetSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, subset_size):
        self.dataset = dataset
        self.subset_size = subset_size

    def __iter__(self):
        indices = random.sample(range(len(self.dataset)), self.subset_size)
        return iter(indices)

    def __len__(self):
        return self.subset_size

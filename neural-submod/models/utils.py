import random
import torch
from torch.utils.data.dataset import Dataset


class RandomSubsetSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, subset_size):
        self.dataset = dataset
        self.subset_size = subset_size

    def __iter__(self):
        indices = random.sample(range(len(self.dataset)), self.subset_size)
        return iter(indices)

    def __len__(self):
        return self.subset_size
    
class SubDataset(Dataset):
    def __init__(self, indices, dataset):
        self.indices = indices
        self.dataset = dataset

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        data_point = self.dataset[index]
        return data_point
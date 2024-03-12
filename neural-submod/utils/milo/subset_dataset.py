from torch.utils.data import Dataset

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
import torch
from torch.utils.data import Dataset

class PairDataset(Dataset):
    def __init__(self, adataX, labels):
        """
        adataX: numpy array or torch tensor of shape [N, D]
        labels: numpy array or torch tensor of shape [N]
        """
        self.adataX = torch.as_tensor(adataX, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)
        assert self.adataX.shape[0] == self.labels.shape[0], "x and c must have the same number of rows"

    def __len__(self):
        return self.adataX.shape[0]

    def __getitem__(self, idx):
        return self.adataX[idx], self.labels[idx]
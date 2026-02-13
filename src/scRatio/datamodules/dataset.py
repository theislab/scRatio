from typing import Tuple

import torch
from torch.utils.data import Dataset


class ArrayDataset(Dataset):
    def __init__(self, x, cond, dtype=torch.float32):
        self.x = torch.as_tensor(x, dtype=dtype)
        self.cond = torch.as_tensor(cond, dtype=dtype)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.cond[idx]

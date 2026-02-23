from typing import Tuple

import torch
from torch.utils.data import Dataset


class ArrayDataset(Dataset):
    """
    Dataset wrapping arrays of samples and conditions.

    Args:
        x (array-like): Data samples.
        cond (array-like): Conditioning data.
        dtype (torch.dtype, optional): Tensor dtype.
    """

    def __init__(self, x, cond, dtype=torch.float32):
        self.x = torch.as_tensor(x, dtype=dtype)
        self.cond = torch.as_tensor(cond, dtype=dtype)

    def __len__(self) -> int:
        """
        Return dataset size.

        Returns:
            int: Number of samples.
        """
        return self.x.shape[0]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve sample and condition.

        Args:
            idx (int): Index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (x, cond).
        """
        return self.x[idx], self.cond[idx]

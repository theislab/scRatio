from typing import List, Optional, Tuple

import anndata as ad
import lightning as L
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader

from .dataset import ArrayDataset


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        return OneHotEncoder(sparse=False, handle_unknown="ignore")


class AnnDataDataModule(L.LightningDataModule):
    def __init__(
        self,
        adata_path: str,
        conditions: List[str],
        num_features: Optional[int],
        train_batch_size: int,
        val_batch_size: int,
        test_batch_size: int,
        val_split: float = 0.0,
        test_split: float = 0.0,
        train_sample_size: Optional[int] = None,
        num_workers: int = 0,
        seed: int = 0,
        representation_key: str = "X_pca",
        pin_memory: bool = True,
        drop_last: bool = True,
    ) -> None:
        super().__init__()
        self.adata_path = adata_path
        self.conditions = list(conditions)
        self.num_features = num_features
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.train_sample_size = train_sample_size
        self.num_workers = num_workers
        self.seed = seed
        self.representation_key = representation_key
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        self.condition_dims: List[int] = []
        self.condition_categories: List[List[str]] = []

        self.train_dataset: Optional[ArrayDataset] = None
        self.val_dataset: Optional[ArrayDataset] = None
        self.test_dataset: Optional[ArrayDataset] = None
        self._prepared = False

    def prepare_data(self) -> None:
        ad.read_h5ad(self.adata_path)

    def setup(self, stage: Optional[str] = None) -> None:
        if self._prepared:
            return

        adata = ad.read_h5ad(self.adata_path)
        x = self._get_representation(adata)
        cond = self._encode_conditions(adata)

        train_idx, val_idx, test_idx = self._split_indices(x.shape[0])
        if self.train_sample_size is not None and self.train_sample_size < len(train_idx):
            rng = np.random.default_rng(self.seed)
            train_idx = rng.choice(train_idx, size=self.train_sample_size, replace=False)

        self.train_dataset = ArrayDataset(x[train_idx], cond[train_idx])
        self.val_dataset = None
        self.test_dataset = None

        if len(val_idx) > 0:
            self.val_dataset = ArrayDataset(x[val_idx], cond[val_idx])
        if len(test_idx) > 0:
            self.test_dataset = ArrayDataset(x[test_idx], cond[test_idx])

        self._prepared = True

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )

    def _get_representation(self, adata: ad.AnnData) -> np.ndarray:
        if self.representation_key not in adata.obsm:
            raise ValueError(
                f"AnnData missing '{self.representation_key}' in .obsm. "
                "Store the representation in adata.obsm before training."
            )
        x = adata.obsm[self.representation_key]
        if self.num_features is None:
            return np.asarray(x)
        if self.num_features > x.shape[1]:
            raise ValueError(
                f"Requested num_features={self.num_features}, but '{self.representation_key}' has {x.shape[1]} columns."
            )
        return np.asarray(x[:, :self.num_features])

    def _encode_conditions(self, adata: ad.AnnData) -> np.ndarray:
        if not self.conditions:
            raise ValueError("conditions must be a non-empty list.")

        encoded_parts = []
        condition_dims: List[int] = []
        condition_categories: List[List[str]] = []

        for key in self.conditions:
            if key not in adata.obs:
                raise KeyError(f"Condition '{key}' not found in adata.obs")
            encoder = _make_one_hot_encoder()
            encoded = encoder.fit_transform(adata.obs[[key]])
            encoded_parts.append(encoded)
            condition_dims.append(encoded.shape[1])
            condition_categories.append([str(c) for c in encoder.categories_[0]])

        self.condition_dims = condition_dims
        self.condition_categories = condition_categories
        return np.concatenate(encoded_parts, axis=1)

    def _split_indices(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.val_split + self.test_split >= 1.0:
            raise ValueError("val_split + test_split must be < 1.0")

        rng = np.random.default_rng(self.seed)
        indices = np.arange(n_samples)
        rng.shuffle(indices)

        n_test = int(round(n_samples * self.test_split))
        n_val = int(round(n_samples * self.val_split))
        n_train = n_samples - n_val - n_test

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val] if n_val > 0 else np.array([], dtype=int)
        test_idx = indices[n_train + n_val :] if n_test > 0 else np.array([], dtype=int)
        return train_idx, val_idx, test_idx

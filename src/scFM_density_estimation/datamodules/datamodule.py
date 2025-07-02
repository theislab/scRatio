import torch
import scanpy as sc
import lightning as L
import numpy as np

from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader, RandomSampler, Subset
from .dataset import MixedConditionDataset

class scFMDataModule(L.LightningDataModule):
    def __init__(self, adata_path, conditions, num_conditions, condition_dims, batch_size,
                 probability, index, num_pcs, num_workers, val_split, test_split, train_sample_size):
        super().__init__()
        self.adata_path = adata_path
        self.conditions = conditions
        self.num_conditions = num_conditions
        self.condition_dims = condition_dims
        self.batch_size = batch_size
        self.num_pcs = num_pcs
        self.probability = probability
        self.index = index
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.train_sample_size = train_sample_size

    def setup(self, stage=None):
        adata = sc.read_h5ad(self.adata_path)
        X = adata.obsm["X_pca"][:, :self.num_pcs]
        
        C = []
        for condition in self.conditions:
            C.append(adata.obs[condition].cat.codes.values.copy())
        C = np.stack(C, axis=1)
            
        dataset = MixedConditionDataset(adataX=X, conditions=C, num_conditions=self.num_conditions,
                                        condition_dims=self.condition_dims, probability=self.probability,
                                        index=self.index)

        n = len(dataset)
        n_test = int(self.test_split * n)
        n_val = int(self.val_split * n)
        n_train = n - n_val - n_test
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            dataset, [n_train, n_val, n_test],
        )

    def train_dataloader(self):
        sampler = RandomSampler(self.train_dataset, replacement=True, num_samples=self.train_sample_size)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

import scanpy as sc
from torch.utils.data import DataLoader
import lightning as L
from .dataset import PairDataset

class scFMDataModule(L.LightningDataModule):
    def __init__(self, adata_path, label_key='leiden', batch_size=32, num_workers=1, val_split=0.2, test_split=0.1, shuffle=True):
        super().__init__()
        self.adata_path = adata_path
        self.label_key = label_key
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.shuffle = shuffle

    def prepare_data(self):
        # Nothing to do here, as scanpy will read the file in setup
        pass

    def setup(self, stage=None):
        # initialize dataset
        adata = sc.read_h5ad(self.adata_path)
        X = adata.X.toarray()
        labels = adata.obs[self.label_key].values
        dataset = PairDataset(X, labels)

        # train, val, test split
        n = len(dataset)
        n_test = int(self.test_split * n)
        n_val = int(self.val_split * n)
        n_train = n - n_val - n_test
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            dataset, [n_train, n_val, n_test], generator=torch.Generator()
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

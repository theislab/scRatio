import torch
import numpy as np
from torch.utils.data import Dataset
    
class MixedConditionDataset(Dataset):
    def __init__(self, adataX, conditions, num_conditions, condition_dims, probability, index):
        """
        adataX: numpy array or torch tensor of shape [N, D]
        labels: numpy array or torch tensor of shape [N]
        """
        self.adataX = torch.as_tensor(adataX).float()
        self.conditions = conditions
        self.condition_dims = condition_dims
        self.num_conditions = num_conditions
        self.probability = probability
        self.index = index
        self.prepare_conditions()

        assert self.adataX.shape[0] == self.conditions.shape[0], "x and c must have the same number of rows"
        assert self.conditions.shape[1] == np.sum(self.condition_dims), "condition one hot encoding is incorrect"

    def __len__(self):
        return self.adataX.shape[0]

    def __getitem__(self, idx):
        condition = self.conditions[idx]
        
        if np.random.uniform() < self.probability:
            col_mask = [True if sum(self.condition_dims[:self.index]) <= i < sum(self.condition_dims[:self.index+1])
                        else False for i in range(sum(self.condition_dims))]
            condition[col_mask] = 0
            
        return self.adataX[idx], condition
    
    def prepare_conditions(self):
        cond = []
        for i in range(self.num_conditions):
            cond.append(torch.nn.functional.one_hot(torch.from_numpy(self.conditions[:, i]).long(),
                                                    num_classes=self.condition_dims[i]).float())      
        self.conditions = torch.cat(cond, dim=1)
import torch
import numpy as np
from torch.utils.data import Dataset, IterableDataset, DataLoader, RandomSampler
    
class MixedConditionDataset(Dataset):
    def __init__(self, adataX, conditions, num_conditions, condition_dims, probability, index):
        """
        This class is used to create a MixedConditionDataset for the scFM model.
        It takes in the following parameters:
        - adataX: The input data (X) in the form of numpy object.
        - conditions: The conditions for the data.
        - num_conditions: The number of conditions.
        - condition_dims: The dimensions of the conditions.
        - probability: The probability of a condition being dropped.
        - index: The index of the condition to be dropped.
        """
        self.adataX = torch.as_tensor(adataX).float()
        self.conditions = conditions
        self.condition_dims = condition_dims
        self.num_conditions = num_conditions
        # self.probability = probability
        # self.index = index
        self.prepare_conditions()

        assert self.adataX.shape[0] == self.conditions.shape[0], "x and c must have the same number of rows"
        assert self.conditions.shape[1] == np.sum(self.condition_dims), "condition one hot encoding is incorrect"

    def __len__(self):
        return self.adataX.shape[0]

    def __getitem__(self, idx):
        # condition = self.conditions[idx]
        
        # if np.random.uniform() < self.probability:
        #     col_mask = [True if sum(self.condition_dims[:self.index]) <= i < sum(self.condition_dims[:self.index+1])
        #                 else False for i in range(sum(self.condition_dims))]
        #     condition[col_mask] = 0
            
        return self.adataX[idx], self.conditions[idx]
    
    def prepare_conditions(self):
        cond = []
        for i in range(self.num_conditions):
            cond.append(torch.nn.functional.one_hot(torch.from_numpy(self.conditions[:, i]).long(),
                                                    num_classes=self.condition_dims[i]).float())      
        self.conditions = torch.cat(cond, dim=1)
        
class ProbabilisticMixedConditionDataset(IterableDataset):
    def __init__(self, X, conditions):
        pass
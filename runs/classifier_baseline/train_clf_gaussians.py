import os
import sys
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchdyn.core import NeuralODE
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from scRatio.models.flow_matching import ConditionalFlowMatchingWithScore
from utils import NODEWrapper_with_ratio_generic_models
from model import ProbabilisticClassifier
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

def prepare_dataset(n, N, cond_dim, locs, device):
    C = np.random.randint(low=0, high=cond_dim, size=(N))
    X = np.concatenate([np.random.normal(loc=locs[c], scale=1, size=(1, n)) for c in C])

    X_train, X_test, C_train, C_test = train_test_split(X, C, test_size=10_000)
    X_train, X_val, C_train, C_val = train_test_split(X_train, C_train, test_size=10_000)

    X_train = torch.tensor(X_train).to(device).float()
    C_train = F.one_hot(torch.tensor(C_train).long(), num_classes=cond_dim).to(device).float()

    X_test = torch.tensor(X_test).to("cuda").float()
    C_test = F.one_hot(torch.tensor(C_test).long(), num_classes=cond_dim).to(device).float()
    
    X_val = torch.tensor(X_val).to(device).float()
    C_val = F.one_hot(torch.tensor(C_val).long(), num_classes=cond_dim).to(device).float()
    return X_train, X_val, X_test, C_train, C_val, C_test

@hydra.main(config_path="./config", config_name="train", version_base=None)
def main(config: DictConfig):    

    # Initialize device 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Prepare dataset
    np.random.seed(42)  # initialize random seed for reproducibility
    
    dimension = config.paths.dimensions
    n_samples = 100000
    
    # Initialize dataset
    X_train, X_val, X_test, C_train, C_val, C_test = prepare_dataset(n=dimension, 
                                                                     N=n_samples, 
                                                                     cond_dim=10, 
                                                                     locs=np.arange(10), 
                                                                     device=device)
    
    
    input_dim = X_train.shape[1]
    output_dim = 2  # Binary classification
    
    # Resiult directory
    res_dir = Path(config.paths.res_dir)  
    res_dir = res_dir / f"{input_dim}_{config.model.hidden_dim}_{config.model.num_hidden_layers}"
    res_dir.mkdir(parents=True, exist_ok=True)  
    
    # Initialize the model 
    model = ProbabilisticClassifier(input_dim, 
                                    config.model.hidden_dim, 
                                    output_dim,
                                    config.model.num_hidden_layers).to(device)
    
    # Train the model
    metrics = {"accuracy_val": None, "accuracy_test": None}
    model = train_classifier(X_train, y_train, model)
    model.eval()
    with torch.no_grad():
        accuracy_val = (model(X_val).argmax(dim=1) == C_val).float().mean().item()
        accuracy_test = (model(X_test).argmax(dim=1) == C_test).float().mean().item()
        
    metrics["accuracy_val"] = accuracy_val
    metrics["accuracy_test"] = accuracy_test
    
    # Compute ratios 
    log_ratios_val = model.compute_log_ratio(X_val, y_idx=1, y_prime_idx=0).cpu().numpy()
    log_ratios_test = model.compute_log_ratio(X_test, y_idx=1, y_prime_idx=0).cpu().numpy()
    np.save(res_dir / "log_ratios_val.npy", log_ratios_val)
    np.save(res_dir / "log_ratios_test.npy", log_ratios_test)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(res_dir / "metrics.csv", index=False)

if __name__ == "__main__":
    # running the experiment
    try: 
        main()
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)

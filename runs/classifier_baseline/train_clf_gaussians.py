import sys
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import hydra
from omegaconf import DictConfig
from model import ProbabilisticClassifier
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from train_clf_mi import train_classifier

def prepare_dataset(n, N, cond_dim, locs, device="cuda"):
    # Set random seed 
    np.random.seed(42)
    
    # Generate synthetic dataset of N samples, where each sample is drawn from a Gaussian distribution whose mean is determined by its class label.
    C = np.random.randint(low=0, high=cond_dim, size=(N))  # Label from 0 to cond_dim-1
    X = np.concatenate([np.random.normal(loc=locs[c], scale=1, size=(1, n)) for c in C])  # N x n 

    X_train, X_val_test, C_train, C_val_test = train_test_split(X, C, test_size=20_000)
    X_val, X_test, C_val, C_test = train_test_split(X_val_test, C_val_test, test_size=10_000)

    X_train = torch.tensor(X_train).to(device).float()
    X_val = torch.tensor(X_val).to(device).float()
    X_test = torch.tensor(X_test).to(device).float()
    
    C_train = torch.tensor(C_train).to(device)
    C_val = torch.tensor(C_val).to(device)
    C_test = torch.tensor(C_test).to(device)
    return X_train, X_val, X_test, C_train, C_val, C_test

@hydra.main(config_path="./config", config_name="train", version_base=None)
def main(config: DictConfig): 
    # Initialize device 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Prepare dataset
    dimension = config.paths.dimensions 
    output_dim = 2 # binary classification task (two Gaussians)
    n_samples = 100000
    loc = config.paths.loc
    locs = [0., loc]

    X_train, X_val, X_test, C_train, C_val, C_test = prepare_dataset(dimension,
                                                                     n_samples, 
                                                                     cond_dim=2,
                                                                     locs=locs, 
                                                                     device=device)[:6]
    
    # Real likelihoods 
    log_condition_val = -0.5 * ((X_val.cpu().numpy() - np.array(locs[1])) ** 2).sum(axis=1) - 0.5 * X_val.shape[1] * np.log(2 * np.pi)
    log_control_val = -0.5 * ((X_val.cpu().numpy() - np.array(locs[0])) ** 2).sum(axis=1) - 0.5 * X_val.shape[1] * np.log(2 * np.pi)
    true_log_ratios_val = log_condition_val - log_control_val   
    
    log_condition_test = -0.5 * ((X_test.cpu().numpy() - np.array(locs[1])) ** 2).sum(axis=1) - 0.5 * X_test.shape[1] * np.log(2 * np.pi)
    log_control_test = -0.5 * ((X_test.cpu().numpy() - np.array(locs[0])) ** 2).sum(axis=1) - 0.5 * X_test.shape[1] * np.log(2 * np.pi)
    true_log_ratios_test = log_condition_test - log_control_test
    
    # Resiult directory
    for iteration in range(3):
        # Initialize target folders for results
        res_dir = Path(config.paths.res_dir)  
        res_dir = res_dir / f"loc_{loc}_dimensions_{dimension}_hidden_dim_{config.model.hidden_dim}_hidden_layers_no_{config.model.num_hidden_layers}_iteration_{iteration}"
        res_dir.mkdir(parents=True, exist_ok=True)  
    
        # Initialize the model 
        model = ProbabilisticClassifier(dimension, 
                                        config.model.hidden_dim, 
                                        output_dim,
                                        config.model.num_hidden_layers).to(device)
    
        # Train the model
        metrics = {"accuracy_val": None, 
                   "accuracy_test": None,  
                   "mse_log_ratios_val": None,
                   "mse_log_ratios_test": None}
        
        model = train_classifier(X_train, C_train, model)
        model.eval()
        with torch.no_grad():
            accuracy_val = (model(X_val).argmax(dim=1) == C_val).float().mean().item()
            accuracy_test = (model(X_test).argmax(dim=1) == C_test).float().mean().item()
        
        metrics["accuracy_val"] = accuracy_val
        metrics["accuracy_test"] = accuracy_test
    
        # Compute ratios 
        with torch.no_grad():
            log_ratios_val = model.compute_log_ratio(X_val, y_idx=1, y_prime_idx=0).cpu().numpy()
            log_ratios_test = model.compute_log_ratio(X_test, y_idx=1, y_prime_idx=0).cpu().numpy()
            
        mse_log_ratios_val = np.mean((log_ratios_val - true_log_ratios_val) ** 2)
        mse_log_ratios_test = np.mean((log_ratios_test - true_log_ratios_test) ** 2)
        metrics["mse_log_ratios_val"] = mse_log_ratios_val
        metrics["mse_log_ratios_test"] = mse_log_ratios_test
        
        np.save(res_dir / "log_ratios_val.npy", log_ratios_val)
        np.save(res_dir / "log_ratios_test.npy", log_ratios_test)
        np.save(res_dir / "true_log_ratios_val.npy", true_log_ratios_val)
        np.save(res_dir / "X_val.npy", X_val.cpu().numpy())
        np.save(res_dir / "X_test.npy", X_test.cpu().numpy())
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(res_dir / "metrics.csv", index=False)

if __name__ == "__main__":
    # running the experiment
    try: 
        main()
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)

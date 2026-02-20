from pathlib import Path
from tqdm import tqdm
import traceback
import sys
import numpy as np
import scanpy as sc
import torch
from lightning import Trainer
import hydra
from omegaconf import DictConfig
from sklearn.preprocessing import OneHotEncoder
from scRatio.datamodules.datamodule import AnnDataDataModule
from scRatio.models.flow_matching import ConditionalFlowMatchingWithScore

import sys 
sys.path.insert(0, "..")
from utils_scratio import train

torch.set_float32_matmul_precision("medium")

def train_scratio(adata: sc.AnnData, scheduler_type: str, res_dir: Path, run_name: str):
    # Initialize device 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Fix number of PCs
    N_pcs = 20 
    
    # Observatoions and conditioning variables
    X = torch.from_numpy(adata.obsm["X_pca"][:, :N_pcs]).float().to(device)
    C = OneHotEncoder().fit_transform(adata.obs[["treatment"]]).toarray()
    C = torch.from_numpy(C).float().to(device)
    
    # Different schedules 
    if scheduler_type == "deterministic":
        sigma = 0
        sigma_min = 0
    elif scheduler_type == "deterministic_sigma_min":
        sigma = 0
        sigma_min = 0.1
    elif scheduler_type == "stochastic":
        sigma = 0.01
        sigma_min = 0
    else:
        raise NotImplementedError
    
    # Initialize schedules 
    lambda_t = lambda t: torch.sqrt((1 - (1 - sigma_min) * t) ** 2 + sigma * t * (1 - t))
    lambda_sp_t = lambda t: (sigma * (1 - 2 * t) - 2 * (1 - sigma_min) * (1 - (1 - sigma_min) * t)) / 2
    
    # Training setup 
    n_steps = 150_000
    batch_size = 512

    # Model initialization 
    model = ConditionalFlowMatchingWithScore(
        input_dim=N_pcs,
        cond_dims=[2],
        hidden_dims=[1024, 1024, 1024],
        encoder_hidden_dims=[256],
        encoder_out_dim=256,
        encoder_out_dim_cond=50, 
        time_feature_dim=50, 
        lambda_t=lambda_t,
        lambda_sp_t=lambda_sp_t,
        betas=[0],
        lr = 1e-4,
        dropout = 0).to(device)
    
    # Train model 
    optimizer = model.configure_optimizers()
    model = train(batch_size, n_steps, model, optimizer, X, C)
    
    # Annotate AnnData 
    from torch.utils.data import DataLoader, TensorDataset
    # Control and condition labels
    control = np.array([1., 0.])
    condition = np.array([0., 1.])
    # Create dataloader to estimate the ratio 
    dataloader_ratio = DataLoader(TensorDataset(X, C), 
                                  batch_size=1000, 
                                  drop_last=False)
    
    lik_ratios_data = []
    with torch.no_grad():
        for batch in tqdm(dataloader_ratio):
            X_batch = batch[0]
            C_batch = batch[1]
            ratios = model.estimate_log_density_ratio(data_samples=X_batch, 
                                                        control=control.unsqueeze(0).repeat(X_batch.shape[0], 1), 
                                                        condition=condition.unsqueeze(0).repeat(X_batch.shape[0], 1), 
                                                        point=C_batch, 
                                                        n_steps=2)
            lik_ratios_data.append(ratios)
    
    lik_ratios_data = np.concatenate(lik_ratios_data)
    adata.obs["log_ratios"] = lik_ratios_data 
    
    # Save results 
    run_dir = res_dir / scheduler_type
    run_dir.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(run_dir / f"{run_name}.h5ad") 

@hydra.main(config_path="./config", config_name="train", version_base=None)
def main(config: DictConfig):
    # Initialize directiories and adata path 
    res_dir = Path(config.paths.res_dir)  
    res_dir.mkdir(parents=True, exist_ok=True)  # make directory if doesn't exist
    
    # Read AnnData
    adata_path = Path(config.paths.adata_path)
    
    # Scheduler parameters 
    scheduler_type = config.scheduler.scheduler_type  # Deterministic or stochastic 

    # Read AnnData 
    base_adata = sc.read_h5ad(adata_path)
    adata_fname = adata_path.name
    tag = adata_fname.replace(".h5ad", "").split("_")[1]

    for i in range(3):
        run_name = f"oversamp_{tag}_{i}"
        adata = base_adata.copy()
        train_scratio(adata, scheduler_type, res_dir, run_name)
    
if __name__ == "__main__":
    # running the experiment
    try: 
        main()
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)

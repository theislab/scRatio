import sys
import traceback
from pathlib import Path
import numpy as np
import scanpy as sc
import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from torchdyn.core import NeuralODE
from sklearn.preprocessing import OneHotEncoder
import hydra
from omegaconf import DictConfig
from tqdm.auto import tqdm
from utils import *

torch.set_float32_matmul_precision("medium")

def train(batch_size, n_steps, model, optimizer, X, C):
    for k in tqdm(range(n_steps)):
        optimizer.zero_grad()
    
        indices = np.random.choice(range(X.shape[0]), size=batch_size, replace=False)
        loss = model.shared_step(X[indices], C[indices], k)
        
        loss.backward()
        optimizer.step()
    return model

def compute_ratio(model, data_samples, cond, cond_dim, condition, control, batch_size):
    # Initialize the device 
    device = data_samples.device   

    # Initialize torch dataloader to iterate through the samples 
    dataloader = DataLoader(TensorDataset(data_samples, cond), batch_size=batch_size, drop_last=False)
    log_ratios = []
    for batch in tqdm(dataloader):
        X_batch = batch[0]
        C_batch = batch[1]
        
        # correction term - its own field
        node = NeuralODE(
            NODEWrapper_with_ratio_tvf_rl(model, control=torch.tensor(control).float().expand(X_batch.shape[0], cond_dim).to(device),
                condition=torch.tensor(condition).float().expand(X_batch.shape[0], cond_dim).to(device), point=C_batch),
            solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
        )
        
        with torch.no_grad():
            traj = node.trajectory(
                torch.cat([X_batch, torch.zeros(X_batch.shape[0], 1).to(device)], dim=-1),
                t_span=torch.linspace(1, 0, 2).to(device)
            )
        _, ratio = traj[-1][:, :-1], traj[-1][:, -1]
        log_ratio_hat = -ratio.cpu().numpy()
        log_ratios.append(log_ratio_hat)

    return np.concatenate(log_ratios)

def train_scratio(adata: sc.AnnData, scheduler_type: str, res_dir: Path, run_name: str):
    # Fix number of PCs
    N_pcs = 20 
    
    # Observatoions and conditioning variables
    X = torch.from_numpy(adata.obsm["X_pca"][:, :N_pcs]).float().cuda()
    C = OneHotEncoder().fit_transform(adata.obs[["treatment"]]).toarray()
    C = torch.from_numpy(C).float().cuda()
    
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
        use_sinusoidal_embeddings=True,
        sinusoidal_feature_dim=50,
        lambda_t=lambda_t,
        lambda_sp_t=lambda_sp_t,
        betas=[0], 
        lr=1e-4
    ).to("cuda")
    optimizer = model.configure_optimizers()
    model = train(batch_size, n_steps, model, optimizer, X, C)
    
    # Compute ratio
    control = np.array([1, 0])
    condition = np.array([0, 1])
    log_ratio = compute_ratio(model, X, C, 2, condition, control, batch_size=1000)
    adata.obs["log_ratios"] = log_ratio 
    
    # Save results 
    run_dir = res_dir / scheduler_type
    run_dir.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(run_dir / f"{run_name}.h5ad") 

@hydra.main(config_path="/home/icb/alessandro.palma/environment/scFM_density_estimation/experiments/differential_abundance/scRatio/config", config_name="train", version_base=None)
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

        # IMPORTANT: copy AnnData for each run
        adata = base_adata.copy()

        train_scratio(adata, scheduler_type, res_dir, run_name)
    
if __name__ == "__main__":
    # running the experiment
    try: 
        main()
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)

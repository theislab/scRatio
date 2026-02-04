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

class NODEWrapper_uncond(torch.nn.Module):
    def __init__(self, model, control, condition):
        super().__init__()
        self.model = model
        self.cond_v = control
        self.cond_u = condition
        self.div_fn, self.eps_fn = div_fn_hutch_trace, torch.randn_like

    def forward(self, t, x, *args, **kwargs):
        x = x[..., :-1]
        
        def vecfield(y):
            ut, _ = self.model(y.unsqueeze(0), t, self.cond_u[:1])
            vt, _ = self.model(y.unsqueeze(0), t, self.cond_v[:1])
            return vt.squeeze() - ut.squeeze()
            
        div = torch.vmap(self.div_fn(vecfield))(x, self.eps_fn(x))
        
        ut, score_u = self.model(x, t, self.cond_u)
        vt, score_v = self.model(x, t, self.cond_v)
        ft, _ = self.model(x, t, None, use_conds=[False])
        
        correction_term_u = torch.linalg.vecdot(ft - ut, score_u)
        correction_term_v = torch.linalg.vecdot(vt - ft, score_v)
        dr = div + correction_term_u + correction_term_v
        
        return torch.cat([ft, dr[:, None]], dim=-1)

def compute_ratio(model, data_samples, cond, cond_dim, condition, control, batch_size):
    # Initialize the device 
    device = data_samples.device   

    # Initialize torch dataloader to iterate through the samples 
    dataloader = DataLoader(TensorDataset(data_samples, cond), 
                            batch_size=batch_size, 
                            drop_last=False, 
                            shuffle=False)
    
    # Initialize log-ratio dict
    log_ratios = []
    
    control_tensor = torch.tensor(control).float().expand(batch_size, cond_dim).to(device)
    condition_tensor = torch.tensor(condition).float().expand(batch_size, cond_dim).to(device)

    for batch in tqdm(dataloader):
        X_batch = batch[0]
        C_batch = batch[1]
        
        # correction term - its own field
        node_dopri = NeuralODE(
            NODEWrapper_uncond(model, 
                               control=control_tensor,
                               condition=condition_tensor, 
                               point=C_batch),
            solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
        )
        
        with torch.no_grad():
            # Dopri simulation
            traj = node_dopri.trajectory(
                torch.cat([X_batch, torch.zeros(X_batch.shape[0], 1).to(device)], dim=-1),
                t_span=torch.linspace(1, 0, 2).to(device)
            )            
            _, ratio = traj[-1][:, :-1], traj[-1][:, -1]
            log_ratio_hat = -ratio.cpu().numpy()
            log_ratios.append(log_ratio_hat)

    log_ratios = np.concatenate(log_ratios)
    return log_ratios

def train_scratio(adata: sc.AnnData,
                  dimensions: int,  
                  scheduler_type: str,
                  batch_size: int, 
                  sigma: str, 
                  res_dir: Path, 
                  run_name: str):
    
    # Fix number of PCs
    N_pcs = dimensions
    
    # Observatoions and conditioning variables
    X = torch.from_numpy(adata.obsm["X_pca"][:, :N_pcs]).float().cuda()
    C = OneHotEncoder().fit_transform(adata.obs[["treatment"]]).toarray()
    C = torch.from_numpy(C).float().cuda()
    
    # Train a very deterministic flow 
    if scheduler_type == "deterministic":
        sigma = 0
        sigma_min = 0
    elif scheduler_type == "sigmamin":
        sigma = 0
        sigma_min = sigma
    elif scheduler_type == "stochastic":
        sigma = sigma
        sigma_min = 0
    else:
        raise NotImplementedError
    
    # Initialize schedules 
    lambda_t = lambda t: torch.sqrt((1 - (1 - sigma_min) * t) ** 2 + sigma * t * (1 - t))
    lambda_sp_t = lambda t: (sigma * (1 - 2 * t) - 2 * (1 - sigma_min) * (1 - (1 - sigma_min) * t)) / 2
    
    # Training setup 
    n_steps = 200_000
    batch_size = batch_size

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
        betas=[0.5],  # Now train both conditional and unconditional 
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
    adata.obs.to_csv(res_dir / f"{run_name}.csv") 

@hydra.main(config_path="/home/icb/alessandro.palma/environment/scFM_density_estimation/experiments/differential_abundance/scRatio_sweeps/config", config_name="train", version_base=None)
def main(config: DictConfig):
    # Read AnnData
    adata_path = Path(config.hparams.adata_path)  # Gonna be an AnnData path with certain oversampling rate 
    base_adata = sc.read_h5ad(adata_path)  # Read AnnData
    
    # Hyperparameters
    dimensions = config.hparams.dimensions
    batch_size = config.hparams.batch_size
    scheduler_type = config.hparams.scheduler_type  # Deterministic or stochastic 
    if "sigmamin" in scheduler_type or "stochastic" in scheduler_type:
        scheduler_type, sigma = scheduler_type.split("_")
        sigma = float(sigma)
    else:
        sigma = 0
        
    # Create config result folder 
    hparam_config = f"dimensions_{dimensions}_batch_size_{batch_size}_scheduler_type_{scheduler_type}_sigma_{sigma}"
    res_dir = Path(config.hparams.res_dir)  
    res_dir = res_dir / hparam_config
    res_dir.mkdir(parents=True, exist_ok=True) 

    # Read AnnData 
    adata_fname = adata_path.name
    tag = adata_fname.replace(".h5ad", "").split("_")[1]
    run_name = f"oversamp_{tag}"
    
    train_scratio(base_adata, 
                  dimensions=dimensions, 
                  scheduler_type=scheduler_type, 
                  batch_size=batch_size,
                  sigma=sigma, 
                  res_dir=res_dir,
                  run_name=run_name)
    
if __name__ == "__main__":
    try: 
        main()
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)

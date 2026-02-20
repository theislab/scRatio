import sys
import traceback
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchdyn.core import NeuralODE
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from scRatio.models.flow_matching import ConditionalFlowMatchingWithScore
from utils import NODEWrapper_with_ratio_generic_models

sys.path.insert(0, "../../../../experiments/differential_abundance/")
from utils_scratio import train

# Steps specific for each dimensionality
N_STEPS = {20: 100_000, 
           40: 100_000,
           80: 100_000,
           160: 100_000,
           320: 100_000}

BATCH_SIZES = {20: 512,
                40: 512,
                80: 512,
                160: 512,
                320: 512}

def compute_ratio(data_samples, model_num, model_den, model_vf, batch_size):
    # Initialize the device 
    device = data_samples.device   

    # Initialize torch dataloader to iterate through the samples 
    dataloader = DataLoader(TensorDataset(data_samples), batch_size=batch_size, drop_last=False)
    log_ratios = []
    for batch in tqdm(dataloader):
        X_batch = batch[0]
        
        # correction term - its own field
        node = NeuralODE(
            NODEWrapper_with_ratio_generic_models(model_num,
                                                  model_den,
                                                  model_vf),
            solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
        )
        
        with torch.no_grad():
            traj = node.trajectory(
                torch.cat([X_batch, torch.zeros(X_batch.shape[0], 1).to(device)], dim=-1),
                t_span=torch.linspace(1, 0, 100).to(device)
            )
        z0, ratio = traj[-1][:, :-1], traj[-1][:, -1]
        log_ratio_hat = -ratio.cpu().numpy()
        log_ratios.append(log_ratio_hat)

    return np.concatenate(log_ratios)

def train_scratio(X_block_train, X_prior_train, X_block_test, X_prior_test, scheduler_type, sigma):
     
    # Initialize scheduler 
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
    models = {"model_prior": None, 
              "model_block": None, 
              "model_uncond": None}
    
    X_trains = {"model_prior": X_prior_train, 
                "model_block": X_block_train,
                "model_uncond": torch.cat([X_block_train,
                                           X_prior_train], dim=0)}
    
    X_test = {"model_prior": X_prior_test, 
                "model_block": X_block_test,
                "model_uncond": torch.cat([X_block_test,
                                           X_prior_test], dim=0)}
    
    n_steps = N_STEPS[X_prior_train.shape[1]]
    batch_size = BATCH_SIZES[X_prior_train.shape[1]]

    for model in models:
        # Model initialization 
        models[model] = ConditionalFlowMatchingWithScore(input_dim=X_trains[model].shape[1],
                                                            cond_dims=[0],
                                                            hidden_dims=[1024, 1024, 1024],
                                                            encoder_hidden_dims=[256],
                                                            encoder_out_dim=256,
                                                            encoder_out_dim_cond=50,
                                                            time_feature_dim=50,
                                                            lambda_t=lambda_t,
                                                            lambda_sp_t=lambda_sp_t,
                                                            betas=[1], 
                                                            lr=1e-4, 
                                                            dropout = 0
                                                        ).to("cuda")
    
        optimizer = models[model].configure_optimizers()
        model = train(batch_size, n_steps, models[model], optimizer, X_trains[model], None)
    
    ratio_estimations = compute_ratio(X_test["model_block"], 
                                      models["model_block"], 
                                      models["model_prior"], 
                                      models["model_uncond"], 
                                      1000)
    return ratio_estimations

@hydra.main(config_path="./config", config_name="train", version_base=None)
def main(config: DictConfig):
    # Initialize directiories and adata path 
    res_dir = Path(config.paths.res_dir) 
    
    # Read data
    data_path = Path(config.paths.data_path)
    dimensions = config.paths.dimensions
    X_block = np.load(data_path / f"block_sigma_{dimensions}.npy")
    X_prior = np.load(data_path / f"identity_sigma_{dimensions}.npy")

    # Get training and test sets 
    X_block_train, X_prior_train =  X_block[:-10000], X_prior[:-10000] 
    X_block_test, X_prior_test = X_block[90000:], X_prior[90000:] 

    X_block_train, X_prior_train =  torch.from_numpy(X_block_train).to("cuda"), torch.from_numpy(X_prior_train).to("cuda")
    X_block_test, X_prior_test = torch.from_numpy(X_block_test).to("cuda"), torch.from_numpy(X_prior_test).to("cuda")
        
    # Scheduler parameters 
    scheduler_type = config.misc.scheduler_type  # Deterministic or stochastic 
    
    res_dir = res_dir / scheduler_type
    res_dir.mkdir(parents=True, exist_ok=True)  # make directory if doesn't exist
    
    if "sigmamin" in scheduler_type or "stochastic" in scheduler_type:
        scheduler_type, sigma = scheduler_type.split("_")
        sigma = float(sigma)
    else:
        sigma = 0.0

    estimated_ratios = []
    for _ in range(3):
        estimated_ratios_i = train_scratio(X_block_train, X_prior_train, X_block_test, X_prior_test, scheduler_type, sigma)
        estimated_ratios.append(estimated_ratios_i)
    
    estimated_ratios = np.concatenate(estimated_ratios)
    np.save(res_dir / f"ratios_{dimensions}.npy", estimated_ratios)
    
if __name__ == "__main__":
    # running the experiment
    try: 
        main()
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)

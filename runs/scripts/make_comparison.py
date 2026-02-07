import torch
import torch.nn.functional as F
import numpy as np
import pickle
import time
import hydra

from typing import Optional, Callable
from omegaconf import DictConfig
from tqdm import tqdm
from scFM_density_estimation.models import *
from scFM_density_estimation.node_wrappers import *
from sklearn.model_selection import train_test_split

def prepare_dataset(n, N, cond_dim, locs):
    C = np.random.randint(low=0, high=cond_dim, size=(N))
    X = np.concatenate([np.random.normal(loc=locs[c], scale=1, size=(1, n)) for c in C])

    X_train, X_test, C_train, C_test = train_test_split(X, C, test_size=10_000)

    X_train = torch.tensor(X_train).to("cuda").float()
    C_train = F.one_hot(torch.tensor(C_train).long(), num_classes=cond_dim).to("cuda").float()

    X_test = torch.tensor(X_test).to("cuda").float()
    C_test = F.one_hot(torch.tensor(C_test).long(), num_classes=cond_dim).to("cuda").float()
    return X_train, X_test, C_train, C_test

def train(batch_size, n_steps, model, optimizer, X, C):
    for k in range(n_steps):
        optimizer.zero_grad()
    
        indices = np.random.choice(range(X.shape[0]), size=batch_size, replace=False)
        loss = model.shared_step(X[indices], C[indices])
        
        loss.backward()
        optimizer.step()
    return model

def evaluate_model(model, data_samples, cond, cond_dim, condition, control, locs):
    device = data_samples.device
    
    start_true = time.time()

    log_condition_true = -0.5 * ((data_samples.cpu().numpy() - np.array(locs[np.argmax(condition)])) ** 2).sum(axis=1) - 0.5 * data_samples.shape[1] * np.log(2 * np.pi)
    log_control_true = -0.5 * ((data_samples.cpu().numpy() - np.array(locs[np.argmax(control)])) ** 2).sum(axis=1) - 0.5 * data_samples.shape[1] * np.log(2 * np.pi)
    log_ratio_true = log_condition_true - log_control_true
    
    time_true = time.time() - start_true
    
    start_hat = time.time()

    # naive evaluation
    node = NeuralODE(
        NODEWrapper_with_trace_div(model, torch.tensor(condition).float().expand(data_samples.shape[0], cond_dim).to(device)),
        solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
    )
    
    with torch.no_grad():
        traj = node.trajectory(
            torch.cat([data_samples, torch.zeros(data_samples.shape[0], 1).to(device)], dim=-1),
            t_span=torch.linspace(1, 0, 100).to(device)
        )
    z0, div = traj[-1][:, :-1], traj[-1][:, -1]
    log_condition_hat = (-0.5 * (z0 ** 2).sum(dim=1) - 0.5 * z0.shape[1] * np.log(2 * np.pi) + div).cpu().numpy()
    
    node = NeuralODE(
        NODEWrapper_with_trace_div(model, torch.tensor(control).float().expand(data_samples.shape[0], cond_dim).to(device)),
        solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
    )
    
    with torch.no_grad():
        traj = node.trajectory(
            torch.cat([data_samples, torch.zeros(data_samples.shape[0], 1).to(device)], dim=-1),
            t_span=torch.linspace(1, 0, 100).to(device)
        )
    z0, div = traj[-1][:, :-1], traj[-1][:, -1]
    log_control_hat = (-0.5 * (z0 ** 2).sum(dim=1) - 0.5 * z0.shape[1] * np.log(2 * np.pi) + div).cpu().numpy()
    
    log_ratio_hat = log_condition_hat - log_control_hat
    
    time_hat = time.time() - start_hat

    start_hat_v2 = time.time()

    # correction term - its own field
    node = NeuralODE(
        NODEWrapper_with_ratio_tvf(model, control=torch.tensor(control).float().expand(data_samples.shape[0], cond_dim).to(device),
            condition=torch.tensor(condition).float().expand(data_samples.shape[0], cond_dim).to(device), point=cond),
        solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
    )
    
    with torch.no_grad():
        traj = node.trajectory(
            torch.cat([data_samples, torch.zeros(data_samples.shape[0], 1).to(device)], dim=-1),
            t_span=torch.linspace(1, 0, 100).to(device)
        )
    z0, ratio = traj[-1][:, :-1], traj[-1][:, -1]
    log_ratio_hat_v2 = -ratio.cpu().numpy()
    
    time_hat_v2 = time.time() - start_hat_v2
    
    return [log_ratio_true, log_ratio_hat, log_ratio_hat_v2], [time_true, time_hat, time_hat_v2]

@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    n = cfg.num_dims
    N = 30_000
    cond_dim = 2
    batch_size = 512
    n_steps = 50_000
    n_tries = 10
    control = np.array([1, 0])
    condition = np.array([0, 1])
    locs = [[0 for _ in range(n)]] + [[cfg.loc for _ in range(n)]]
    sigmas = [1, 0.75, 0.5, 0.25, 0]
    sigma_mins = [1e-1, 1e-2, 1e-3, 1e-4, 0]

    names = ["true", "naive", "ct own rl"]

    X_train, X_test, C_train, C_test = prepare_dataset(n, N, cond_dim, locs)

    mask_condition = np.all(C_test.cpu().numpy() == condition, axis=1)
    mask_control = np.all(C_test.cpu().numpy() == control, axis=1)
    mask_both = mask_condition | mask_control 

    results = {}
    for sigma in sigmas:
        for sigma_min in sigma_mins:
            lambda_t = lambda t: torch.sqrt((1 - (1 - sigma_min) * t) ** 2 + sigma * t * (1 - t))
            lambda_sp_t = lambda t: (sigma * (1 - 2 * t) - 2 * (1 - sigma_min) * (1 - (1 - sigma_min) * t)) / 2
            
            tmp_results = {name: np.array([]) for name in names}
            time_results = {name: [] for name in names}
            for _ in tqdm(range(n_tries)):
                model = ConditionalFlowMatchingWithScore(
                    input_dim=n,
                    cond_dims=[cond_dim],
                    hidden_dims=[1024, 1024, 1024],
                    encoder_hidden_dims=[256],
                    encoder_out_dim=50,
                    lambda_t=lambda_t,
                    lambda_sp_t=lambda_sp_t,
                    betas=[0.1]
                ).to("cuda")
                optimizer = model.configure_optimizers()
                
                model = train(batch_size, n_steps, model, optimizer, X_train, C_train)
                log_ratios, times = evaluate_model(model, X_test, C_test, cond_dim, condition, control, locs)
        
                for i, name in enumerate(names):
                    if tmp_results[name].shape[0] == 0:
                        tmp_results[name] = log_ratios[i].reshape(-1, 1)
                    else:
                        tmp_results[name] = np.concatenate([tmp_results[name], log_ratios[i].reshape(-1, 1)], axis=1)
                        
                    time_results[name].append(times[i])
                        
            for name in names:
                if name == "true":
                    key =  "- - true"
                elif name == "naive":
                    key = str(sigma_min) + " " + str(sigma) + " naive"
                elif name == "ct own rl":
                    key = str(sigma_min) + " " + str(sigma) + " rl"
                else:
                    raise ValueError("Unknown value for name")
                
                if key in results:
                    results[key]["value"] = np.concatenate([results[key]["value"], tmp_results[name]], axis=1)
                    results[key]["time"] += time_results[name]
                else:
                    results[key] = {}
                    results[key]["value"] = tmp_results[name]
                    results[key]["time"] = time_results[name]

    results["mask_condition"] = mask_condition
    results["mask_control"] = mask_control
    results["mask_both"] = mask_both

    with open(f"./notebooks/icml_tests/table_results/results_icml_{cfg.loc}_{cfg.num_dims}.pkl", "wb") as f:
        pickle.dump(results, f)

    print("FINISHED")
    
if __name__ == "__main__":
    main()
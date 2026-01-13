import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import anndata as ad
import scipy as sp
import pandas as pd
import pickle
import time
import hydra

from typing import Optional, Callable
from omegaconf import DictConfig
from tqdm import tqdm
from scFM_density_estimation.models import *
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

class Encoder(nn.Module):
    def __init__(self, cond_dim: int = 1, cond_hidden_dims: list = [],
                 cond_out_dim: int = 2, dropout: float = 0):
        super().__init__()
        layers = []
        prev_dim = cond_dim
        for dim in cond_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.SELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, cond_out_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, cond):
        return self.encoder(cond)

class ConditionalFlowMatchingWithScore(L.LightningModule):
    def __init__(
        self,
        input_dim: int,
        cond_dims: list,
        hidden_dims: list,
        encoder_hidden_dims: list,
        encoder_out_dim: int,
        lambda_t: Callable,
        lambda_sp_t: Callable,
        betas: list,
        lr: float = 1e-3,
        use_ot_sampler: bool = False,
        ot_method: str = "exact",
        dropout: float = 0
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_encoder = Encoder(input_dim, encoder_hidden_dims, encoder_out_dim, dropout)
        self.cond_encoders = nn.ModuleList([
            Encoder(cond_dim, encoder_hidden_dims, encoder_out_dim, dropout)
            for cond_dim in cond_dims
        ])
        self.vf_mlp = FlowMatchingMLP(encoder_out_dim + 1, hidden_dims, input_dim, dropout)
        self.score_mlp = FlowMatchingMLP(encoder_out_dim + 1, hidden_dims, input_dim, dropout)
        
        self.lambda_t = lambda_t
        self.lambda_sp_t = lambda_sp_t

        self.betas = betas
        self.encoder_out_dim = encoder_out_dim
        self.use_ot_sampler = use_ot_sampler
        self.cond_dims = cond_dims
        self.lr = lr

    def forward(self, x, t, cond, use_conds=[True]):
        if t.dim() == 0 or t.size()[0] == 1:
            t = t.expand(x.shape[0]).unsqueeze(1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)
        
        start = 0
        xc = self.data_encoder(x)
        for i, cond_dim in enumerate(self.cond_dims):
            if use_conds[i]:
                xc += self.cond_encoders[i](cond[:, start:(start + cond_dim)])
            start += cond_dim
        
        xtc = torch.cat([xc, t], dim=1)
        
        vf = self.vf_mlp.mlp(xtc)
        score = self.score_mlp.mlp(xtc)
        
        return vf, score

    def shared_step(self, x1, cond):
        device = x1.device

        x0 = torch.randn_like(x1).to(device)
        t = torch.rand(x1.shape[0]).unsqueeze(1).to(device)

        xt = t * x1 + self.lambda_t(t) * x0
        ut = x1 + self.lambda_sp_t(t) / self.lambda_t(t) * x0
        c_t = self.lambda_t(t) ** 2 - self.lambda_sp_t(t) * t

        use_conds = (np.random.uniform(size=len(self.betas)) >= np.array(self.betas))
        pred_ut, pred_score = self(xt, t, cond, use_conds)
        
        vf_loss = F.mse_loss(pred_ut, ut)
        score_loss = F.mse_loss(c_t * pred_score, t * ut - xt)

        return vf_loss + score_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def train(batch_size, n_steps, model, optimizer, X, C):
    for k in range(n_steps):
        optimizer.zero_grad()
    
        indices = np.random.choice(range(X.shape[0]), size=batch_size, replace=False)
        loss = model.shared_step(X[indices], C[indices])
        
        loss.backward()
        optimizer.step()
    return model

def div_fn_hutch_trace(u):
    def div_fn(x, eps):
        _, vjpfunc = torch.func.vjp(u, x)
        return (vjpfunc(eps)[0] * eps).sum()

    return div_fn

class NODEWrapper_with_trace_div(torch.nn.Module):
    def __init__(self, model, cond):
        super().__init__()
        self.model = model
        self.cond = cond
        self.div_fn, self.eps_fn = div_fn_hutch_trace, torch.randn_like

    def forward(self, t, x, *args, **kwargs):
        x = x[..., :-1]
        
        def vecfield(y):
            vf, _ = self.model(y.unsqueeze(0), t, self.cond[:1])
            return vf.squeeze()

        div = torch.vmap(self.div_fn(vecfield))(x, self.eps_fn(x))
        dx, _ = self.model(x, t, self.cond)
            
        return torch.cat([dx, div[:, None]], dim=-1)

class NODEWrapper_with_ratio_tvf_rl(torch.nn.Module):
    def __init__(self, model, control, condition, point):
        super().__init__()
        self.model = model
        self.cond_v = control
        self.cond_u = condition
        self.cond_f = point
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
        ft, _ = self.model(x, t, self.cond_f)
        
        correction_term_u = torch.linalg.vecdot(ft - ut, score_u)
        correction_term_v = torch.linalg.vecdot(vt - ft, score_v)
        dr = div + correction_term_u + correction_term_v
        
        return torch.cat([ft, dr[:, None]], dim=-1)

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
        NODEWrapper_with_ratio_tvf_rl(model, control=torch.tensor(control).float().expand(data_samples.shape[0], cond_dim).to(device),
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

    if cfg.start_from_scratch:
        X_train, X_test, C_train, C_test = prepare_dataset(n, N, cond_dim, locs)

        results = {}

        sigma_last = 2
        sigma_min_last = 2
    else:
        with open(f"./runs/scripts/checkpoints/checkpoint_{cfg.loc}_{cfg.num_dims}.pkl", "rb") as f:
            saved_vars = pickle.load(f)

        results = saved_vars["results"]
        sigma_last = saved_vars["sigma_last"]
        sigma_min_last = saved_vars["sigma_min_last"]
        X_train = saved_vars["X_train"]
        X_test = saved_vars["X_test"]
        C_train = saved_vars["C_train"]
        C_test = saved_vars["C_test"]

    for sigma in sigmas:
        for sigma_min in sigma_mins:
            if (sigma >= sigma_last) and (sigma_min >= sigma_min_last):
                print(f"Skipping sigma={sigma}, sigma_min={sigma_min} since already done")
                continue
            else:
                print(f"Starting sigma={sigma}, sigma_min={sigma_min}")

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

            sigma_last = sigma
            sigma_min_last = sigma_min

            with open(f"./runs/scripts/checkpoints/checkpoint_{cfg.loc}_{cfg.num_dims}.pkl", "wb") as f:
                pickle.dump({
                    "results": results,
                    "sigma_last": sigma_last,
                    "sigma_min_last": sigma_min_last,
                    "X_train": X_train,
                    "X_test": X_test,
                    "C_train": C_train,
                    "C_test": C_test
                }, f)

    mask_condition = np.all(C_test.cpu().numpy() == condition, axis=1)
    mask_control = np.all(C_test.cpu().numpy() == control, axis=1)
    mask_both = mask_condition | mask_control 

    results["mask_condition"] = mask_condition
    results["mask_control"] = mask_control
    results["mask_both"] = mask_both

    with open(f"/home/icb/egor.antipov/scFM_density_estimation/notebooks/icml_tests/table_results/results_icml_{cfg.loc}_{cfg.num_dims}.pkl", "wb") as f:
        pickle.dump(results, f)

    print("FINISHED")
    
if __name__ == "__main__":
    main()
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

def prepare_dataset_indep(n, N, cond_dim, locs1, locs2, a, b):
    C1 = np.random.randint(low=0, high=cond_dim, size=(N))
    C2 = np.random.randint(low=0, high=cond_dim, size=(N))
    C = np.stack([C1, C2], axis=1)
    
    X1 = np.concatenate([np.random.normal(loc=locs1[c], scale=1, size=(1, n)) for c in C1])
    X2 = np.concatenate([np.random.normal(loc=locs2[c], scale=1, size=(1, n)) for c in C2])
    X = a * X1 + b * X2
    
    X_train, X_test, C_train, C_test = train_test_split(X, C, test_size=10_000)

    X_train = torch.tensor(X_train).to("cuda").float()
    C_train = F.one_hot(torch.tensor(C_train).long(), num_classes=cond_dim).reshape(-1, 2*cond_dim).to("cuda").float()

    X_test = torch.tensor(X_test).to("cuda").float()
    C_test = F.one_hot(torch.tensor(C_test).long(), num_classes=cond_dim).reshape(-1, 2*cond_dim).to("cuda").float()
    return X_train, X_test, C_train, C_test

def prepare_dataset_dep(n, N, cond_dim, locs1, locs2, a, b):
    C1 = np.random.randint(low=0, high=cond_dim, size=(N))
    C2 = (C1 + np.random.randint(low=0, high=2, size=(N))) % cond_dim
    C = np.stack([C1, C2], axis=1)
    
    X1 = np.concatenate([np.random.normal(loc=locs1[c], scale=1, size=(1, n)) for c in C1])
    X2 = np.concatenate([np.random.normal(loc=locs1[c], scale=1, size=(1, n)) for c in C2])
    X = a * X1 + b * X2
    
    X_train, X_test, C_train, C_test = train_test_split(X, C, test_size=10_000)

    X_train = torch.tensor(X_train).to("cuda").float()
    C_train = F.one_hot(torch.tensor(C_train).long(), num_classes=cond_dim).reshape(-1, 2*cond_dim).to("cuda").float()

    X_test = torch.tensor(X_test).to("cuda").float()
    C_test = F.one_hot(torch.tensor(C_test).long(), num_classes=cond_dim).reshape(-1, 2*cond_dim).to("cuda").float()
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

    def forward(self, x, t, cond, use_conds):
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

def div_fn_hutch_trace_with_cond(u):
    def div_fn(x, cond, eps):
        _, vjpfunc = torch.func.vjp(u(cond), x)
        return (vjpfunc(eps)[0] * eps).sum()

    return div_fn

class NODEWrapper_with_trace_div(torch.nn.Module):
    def __init__(self, model, cond, use_conds):
        super().__init__()
        self.model = model
        self.cond = cond
        self.div_fn, self.eps_fn = div_fn_hutch_trace_with_cond, torch.randn_like
        self.use_conds = use_conds

    def forward(self, t, x, *args, **kwargs):
        x = x[..., :-1]
        
        def prep_vecfield(cond):
            def vecfield(y):
                vf, _ = self.model(y.unsqueeze(0), t, cond.unsqueeze(0), self.use_conds)
                return vf.squeeze()
            return vecfield

        div = torch.vmap(self.div_fn(prep_vecfield))(x, self.cond, self.eps_fn(x))
        dx, _ = self.model(x, t, self.cond, self.use_conds)
            
        return torch.cat([dx, div[:, None]], dim=-1)

class NODEWrapper_indep_test(torch.nn.Module):
    def __init__(self, model, cond):
        super().__init__()
        self.model = model
        self.cond = cond
        self.div_fn, self.eps_fn = div_fn_hutch_trace_with_cond, torch.randn_like

    def forward(self, t, x, *args, **kwargs):
        x = x[..., :-1]

        def prep_vecfield(cond):
            def vecfield(y):
                u_c12, _ = self.model(y.unsqueeze(0), t, cond.unsqueeze(0), [True, True])
                u_c1, _ = self.model(y.unsqueeze(0), t, cond.unsqueeze(0), [True, False])
                u_c2, _ = self.model(y.unsqueeze(0), t, cond.unsqueeze(0), [False, True])
                u, _ = self.model(y.unsqueeze(0), t, cond.unsqueeze(0), [False, False])
                return u_c1.squeeze() + u_c2.squeeze() - u_c12.squeeze() - u.squeeze()
            return vecfield
            
        div = torch.vmap(self.div_fn(prep_vecfield))(x, self.cond, self.eps_fn(x))
        
        u_c12, score_c12 = self.model(x, t, self.cond, [True, True])
        u_c1, score_c1 = self.model(x, t, self.cond, [True, False])
        u_c2, score_c2 = self.model(x, t, self.cond, [False, True])
        u, score = self.model(x, t, self.cond, [False, False])
        
        correction_term_c12 = torch.linalg.vecdot(u - u_c12, score_c12)
        correction_term_c1 = torch.linalg.vecdot(u - u_c1, score_c1)
        correction_term_c2 = torch.linalg.vecdot(u - u_c2, score_c2)
        dr = div + correction_term_c12 - correction_term_c1 - correction_term_c2
        
        return torch.cat([u, dr[:, None]], dim=-1)
    
def train_model(model, optimizer, X, C, batch_size, n_steps):
    for _ in range(n_steps):
        optimizer.zero_grad()

        indices = np.random.choice(range(X.shape[0]), size=batch_size, replace=False)
        loss = model.shared_step(X[indices], C[indices])
        
        loss.backward()
        optimizer.step()

    return model

def evaluate_model(model, X_test, C_test):
    device = X_test.device

    # single trajectory
    start_hat = time.time()
    
    node = NeuralODE(
        NODEWrapper_indep_test(model=model, cond=C_test),
        solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
    )

    with torch.no_grad():
        traj = node.trajectory(
            torch.cat([X_test, torch.zeros(X_test.shape[0], 1).to(device)], dim=-1),
            t_span=torch.linspace(1, 0, 100).to(device)
        )
    z0, ratio = traj[-1][:, :-1], traj[-1][:, -1]
    log_ratio_hat = -ratio.cpu().numpy()
    
    time_hat = time.time() - start_hat
    
    # direct method
    start_hat_direct = time.time()
    
    node = NeuralODE(
        NODEWrapper_with_trace_div(model=model, cond=C_test, use_conds=[True, True]),
        solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
    )

    with torch.no_grad():
        traj = node.trajectory(
            torch.cat([X_test, torch.zeros(X_test.shape[0], 1).to(device)], dim=-1),
            t_span=torch.linspace(1, 0, 100).to(device)
        )
    z0, div = traj[-1][:, :-1], traj[-1][:, -1]
    log_both_hat = (-0.5 * (z0 ** 2).sum(dim=1) - 0.5 * z0.shape[1] * np.log(2 * np.pi) + div).cpu().numpy()

    node = NeuralODE(
        NODEWrapper_with_trace_div(model=model, cond=C_test, use_conds=[True, False]),
        solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
    )

    with torch.no_grad():
        traj = node.trajectory(
            torch.cat([X_test, torch.zeros(X_test.shape[0], 1).to(device)], dim=-1),
            t_span=torch.linspace(1, 0, 100).to(device)
        )
    z0, div = traj[-1][:, :-1], traj[-1][:, -1]
    log_first_hat = (-0.5 * (z0 ** 2).sum(dim=1) - 0.5 * z0.shape[1] * np.log(2 * np.pi) + div).cpu().numpy()

    node = NeuralODE(
        NODEWrapper_with_trace_div(model=model, cond=C_test, use_conds=[False, True]),
        solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
    )

    with torch.no_grad():
        traj = node.trajectory(
            torch.cat([X_test, torch.zeros(X_test.shape[0], 1).to(device)], dim=-1),
            t_span=torch.linspace(1, 0, 100).to(device)
        )
    z0, div = traj[-1][:, :-1], traj[-1][:, -1]
    log_second_hat = (-0.5 * (z0 ** 2).sum(dim=1) - 0.5 * z0.shape[1] * np.log(2 * np.pi) + div).cpu().numpy()

    node = NeuralODE(
        NODEWrapper_with_trace_div(model=model, cond=C_test, use_conds=[False, False]),
        solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
    )

    with torch.no_grad():
        traj = node.trajectory(
            torch.cat([X_test, torch.zeros(X_test.shape[0], 1).to(device)], dim=-1),
            t_span=torch.linspace(1, 0, 100).to(device)
        )
    z0, div = traj[-1][:, :-1], traj[-1][:, -1]
    log_none_hat = (-0.5 * (z0 ** 2).sum(dim=1) - 0.5 * z0.shape[1] * np.log(2 * np.pi) + div).cpu().numpy()
    
    log_ratio_hat_direct = log_both_hat + log_none_hat - log_first_hat - log_second_hat
    
    time_hat_direct = time.time() - start_hat_direct
    
    return [log_ratio_hat_direct, log_ratio_hat], [time_hat_direct, time_hat]
    
@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    n = 10
    N = 30_000
    cond_dim = 4
    cond_dims = [cond_dim, cond_dim]
    batch_size = 512
    n_steps = 50_000
    n_tries = 10
    sigmas = [1, 0.75, 0.5, 0.25, 0]
    sigma_mins = [1e-1, 1e-2, 1e-3, 1e-4, 0]
    
    names = ["direct", "ct own rl"]
    
    results = {}
    
    locs1 = [np.random.choice([-1, 1], size=n) for _ in range(cond_dim)]
    locs2 = [np.random.choice([-2, 2], size=n) for _ in range(cond_dim)]

    if cfg.run_type == "dep":
        X_train, X_test, C_train, C_test = prepare_dataset_dep(n, N, cond_dim, locs1, locs2, 1, 1)
    elif cfg.run_type == "indep":
        X_train, X_test, C_train, C_test = prepare_dataset_indep(n, N, cond_dim, locs1, locs2, 1, 1)
    else:
        raise ValueError("Unknown value for run_type")
    
    for sigma in sigmas:
        for sigma_min in sigma_mins:
            lambda_t = lambda t: torch.sqrt((1 - (1 - sigma_min) * t) ** 2 + sigma * t * (1 - t))
            lambda_sp_t = lambda t: (sigma * (1 - 2 * t) - 2 * (1 - sigma_min) * (1 - (1 - sigma_min) * t)) / 2
            
            tmp_results = {name: np.array([]) for name in names}
            time_results = {name: [] for name in names}
            for _ in tqdm(range(n_tries)):
                model = ConditionalFlowMatchingWithScore(
                    input_dim=n,
                    cond_dims=cond_dims,
                    hidden_dims=[1024, 1024, 1024],
                    encoder_hidden_dims=[256],
                    encoder_out_dim=50,
                    lambda_t=lambda_t,
                    lambda_sp_t=lambda_sp_t,
                    betas=[0.2, 0.2]
                ).to("cuda")
                optimizer = model.configure_optimizers()

                model = train_model(model, optimizer, X_train, C_train, batch_size, n_steps)
                log_ratios, times = evaluate_model(model, X_test, C_test)
        
                for i, name in enumerate(names):
                    if tmp_results[name].shape[0] == 0:
                        tmp_results[name] = log_ratios[i].reshape(-1, 1)
                    else:
                        tmp_results[name] = np.concatenate([tmp_results[name], log_ratios[i].reshape(-1, 1)], axis=1)
                        
                    time_results[name].append(times[i])
                        
            for name in names:
                if name == "direct":
                    key = str(sigma_min) + " " + str(sigma) + " direct"
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
    
    results["X"] = X_test
    results["C"] = C_test

    with open(f"/home/icb/egor.antipov/scFM_density_estimation/notebooks/tests/table_results/{cfg.run_type}_results_indep_test.pkl", "wb") as f:
        pickle.dump(results, f)

    print("FINISHED")  
    
if __name__ == "__main__":
    main()
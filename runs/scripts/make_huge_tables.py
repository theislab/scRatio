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

from typing import Optional
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

class ConditionalFlowMatchingWithScore(L.LightningModule):
    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        hidden_dims: list = [],
        cond_hidden_dims: list = [16],
        cond_out_dim: int = 8,
        lr: float = 1e-3,
        use_encoder: bool = False,
        use_ot_sampler: bool = False,
        ot_method: str = "exact",
        dropout: float = 0,
        sigma_min: float = 0,
        requires_noise: bool = False,
        use_score_head: bool = False,
        score_loss_type: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        if not use_encoder:
            cond_out_dim = cond_dim

        self.requires_noise = requires_noise
        self.use_score_head = use_score_head
        self.score_loss_type = score_loss_type

        self.cond_encoder = ConditionEncoder(cond_dim, cond_hidden_dims, cond_out_dim, dropout)
        self.vf_mlp = FlowMatchingMLP(input_dim + 1 + cond_out_dim, hidden_dims, input_dim, dropout)
        self.score_mlp = (
            FlowMatchingMLP(input_dim + 1 + cond_out_dim, hidden_dims, input_dim, dropout)
            if self.use_score_head
            else None
        )
        self.sigma_min = sigma_min

        self.use_encoder = use_encoder
        self.use_ot_sampler = use_ot_sampler
        self.cond_dim = cond_dim
        self.lr = lr

    def forward(self, x, t, cond):
        if t.dim() == 0 or t.size()[0] == 1:
            t = t.expand(x.shape[0]).unsqueeze(1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)

        cond_enc = self.cond_encoder(cond) if self.use_encoder else cond
        xtc = torch.cat([x, t, cond_enc], dim=1)
        vf = self.vf_mlp.mlp(xtc)

        if self.use_score_head:
            score = self.score_mlp.mlp(xtc)
            return vf, score
        return vf

    def shared_step(self, x1, cond):
        device = x1.device

        x0 = torch.randn_like(x1).to(device)
        t = torch.rand(x1.shape[0]).unsqueeze(1).to(device)

        eps = torch.randn_like(x1).to(device)
        sigma_t = torch.sqrt(t * (1 - t))

        mu_t = x0 + t * (x1 - (1 - self.sigma_min) * x0)
        drift = x1 - (1 - self.sigma_min) * x0

        if self.requires_noise:
            noise_term = (1 - 2 * t) / (2 * t * (1 - t) + 1e-8) * sigma_t * eps
            ut = noise_term + drift
        else:
            ut = drift
            
        xt = mu_t + eps * sigma_t if self.requires_noise else mu_t

        outputs = self(xt, t, cond)
        if self.use_score_head:
            pred_ut, pred_score = outputs
        else:
            pred_ut, pred_score = outputs, None

        vf_loss = F.mse_loss(pred_ut, ut)

        if self.use_score_head:
            if self.score_loss_type == "tong":
                score_loss = F.mse_loss(2 * sigma_t * pred_score, -eps)
            elif self.score_loss_type == "rl":
                score_loss = F.mse_loss((1 - t) * pred_score, t * ut - xt)
            else:
                score_loss = torch.tensor(0.0, device=xt.device)
            total_loss = vf_loss + score_loss
        else:
            total_loss = vf_loss

        return total_loss

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
        
        if self.model.use_score_head:
            def vecfield(y):
                vf, _ = self.model(y.unsqueeze(0), t, self.cond[:1])
                return vf.squeeze()
        else:
            def vecfield(y):
                return self.model(y.unsqueeze(0), t, self.cond[:1]).squeeze()

        div = torch.vmap(self.div_fn(vecfield))(x, self.eps_fn(x))
        
        if self.model.use_score_head:
            dx, _ = self.model(x, t, self.cond)
        else:
            dx = self.model(x, t, self.cond)
            
        return torch.cat([dx, div[:, None]], dim=-1)

class NODEWrapper_with_ratio_tvf(torch.nn.Module):
    def __init__(self, model, control, condition, point):
        super().__init__()
        self.model = model
        self.cond_v = control
        self.cond_u = condition
        self.cond_f = point
        self.div_fn, self.eps_fn = div_fn_hutch_trace, torch.randn_like

    def forward(self, t, x, *args, **kwargs):
        x = x[..., :-1]
        
        if self.model.use_score_head:
            def vecfield(y):
                ut, _ = self.model(y.unsqueeze(0), t, self.cond_u[:1])
                vt, _ = self.model(y.unsqueeze(0), t, self.cond_v[:1])
                return vt.squeeze() - ut.squeeze()
        else:
            def vecfield(y):
                return self.model(y.unsqueeze(0), t, self.cond_v[:1]).squeeze() \
                - self.model(y.unsqueeze(0), t, self.cond_u[:1]).squeeze()
            
        div = torch.vmap(self.div_fn(vecfield))(x, self.eps_fn(x))
        
        if self.model.use_score_head:
            ut, score_u = self.model(x, t, self.cond_u)
            vt, score_v = self.model(x, t, self.cond_v)
            ft, _ = self.model(x, t, self.cond_f)
        else:
            ut = self.model(x, t, self.cond_u)
            vt = self.model(x, t, self.cond_v)
            ft = self.model(x, t, self.cond_f)
            
            if self.model.sigma_min == 0:
                score_u = (t * ut - x) / (1 - t + 1e-2)
            else:
                score_u = (t * ut - x) / (1 - (1 - self.model.sigma_min) * t)
            
            if self.model.sigma_min == 0:
                score_v = (t * vt - x) / (1 - t + 1e-2)
            else:
                score_v = (t * vt - x) / (1 - (1 - self.model.sigma_min) * t)
        
        correction_term_u = torch.linalg.vecdot(ft - ut, score_u)
        correction_term_v = torch.linalg.vecdot(vt - ft, score_v)
        dr = div + correction_term_u + correction_term_v
        
        return torch.cat([ft, dr[:, None]], dim=-1)

def evaluate_model(model, nwtd, nwr, data_samples, cond, cond_dim, condition, control, locs):
    device = data_samples.device
    
    start_true = time.time()

    log_condition_true = -0.5 * ((data_samples.cpu().numpy() - np.array(locs[np.argmax(condition)])) ** 2).sum(axis=1) - 0.5 * data_samples.shape[1] * np.log(2 * np.pi)
    log_control_true = -0.5 * ((data_samples.cpu().numpy() - np.array(locs[np.argmax(control)])) ** 2).sum(axis=1) - 0.5 * data_samples.shape[1] * np.log(2 * np.pi)
    log_ratio_true = log_condition_true - log_control_true
    
    time_true = time.time() - start_true
    
    start_hat = time.time()

    # direct evaluation
    node = NeuralODE(
        nwtd(model, torch.tensor(condition).float().expand(data_samples.shape[0], cond_dim).to(device)),
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
        nwtd(model, torch.tensor(control).float().expand(data_samples.shape[0], cond_dim).to(device)),
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

    start_hat_v3 = time.time()

    # correction term - its own field
    node = NeuralODE(
        nwr(model, control=torch.tensor(control).float().expand(data_samples.shape[0], cond_dim).to(device),
            condition=torch.tensor(condition).float().expand(data_samples.shape[0], cond_dim).to(device), point=cond),
        solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
    )
    
    with torch.no_grad():
        traj = node.trajectory(
            torch.cat([data_samples, torch.zeros(data_samples.shape[0], 1).to(device)], dim=-1),
            t_span=torch.linspace(1, 0, 100).to(device)
        )
    z0, ratio = traj[-1][:, :-1], traj[-1][:, -1]
    log_ratio_hat_v3 = -ratio.cpu().numpy()
    
    time_hat_v3 = time.time() - start_hat_v3
    
    return [log_ratio_true, log_ratio_hat, log_ratio_hat_v3], [time_true, time_hat, time_hat_v3]

def get_locs(run_type):
    if run_type == "diff":
        locs = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 0, 0, 0, 0, 2, 2, 2, 2], [2, 0, 0, 2, 0, 0, 0, 2, 0, 0], [3, 3, 3, 3, -1, 3, 3, 3, 3, -1]]
        condition = np.array([0, 0, 0, 1])
    elif run_type == "easy":
        locs = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 2, 0, 2, 2, 2, 0, 0, 2], [2, 2, 0, 0, 2, 2, 2, 0, 0, 2], [-1, -1, 3, -1, 3, 3, 3, -1, -1, 3]]
        condition = np.array([0, 0, 0, 1])
    elif run_type == "really_easy":
        locs = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 0, 0, 0, 0, 2, 2, 2, 2], [2, 0, 0, 2, 0, 0, 0, 2, 0, 0], [3, 3, 3, 3, -1, 3, 3, 3, 3, -1]]
        condition = np.array([0, 1, 0, 0])
    else:
        raise ValueError("Unknown value for run_type")
    return locs, condition

@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    n = 10
    N = 30_000
    cond_dim = 4
    batch_size = 512
    n_steps = 50_000
    n_tries = 10
    control = np.array([1, 0, 0, 0])
    nwtd, nwr = NODEWrapper_with_trace_div, NODEWrapper_with_ratio_tvf

    noise = [True, False]
    score_loss_types = ["rnl", "tong", "rl"]
    sigmas = [1e-1, 1e-2, 1e-3, 1e-4, 0]
    names = ["true", "direct", "ct own"]

    results = {}

    locs, condition = get_locs(cfg.run_type)

    X_train, X_test, C_train, C_test = prepare_dataset(n, N, cond_dim, locs)

    mask_condition = np.all(C_test.cpu().numpy() == condition, axis=1)
    mask_control = np.all(C_test.cpu().numpy() == control, axis=1)
    mask_both = mask_condition | mask_control 

    for requires_noise in noise:
        for score_loss_type in score_loss_types:
            for sigma_min in sigmas:
                tmp_results = {name: np.array([]) for name in names}
                time_results = {name: [] for name in names}
                for _ in tqdm(range(n_tries)):
                    model = ConditionalFlowMatchingWithScore(
                        input_dim=n,
                        hidden_dims=[1024, 1024, 1024],
                        cond_dim=cond_dim,
                        use_encoder=False,
                        use_ot_sampler=False,
                        sigma_min=sigma_min,
                        requires_noise=requires_noise,
                        use_score_head=(score_loss_type != "rnl"),
                        score_loss_type=score_loss_type,
                    ).to("cuda")
                    optimizer = model.configure_optimizers()
                    
                    model = train(batch_size, n_steps, model, optimizer, X_train, C_train)
                    log_ratios, times = evaluate_model(model, nwtd, nwr, X_test, C_test, cond_dim, condition, control, locs)
            
                    for i, name in enumerate(names):
                        if tmp_results[name].shape[0] == 0:
                            tmp_results[name] = log_ratios[i].reshape(-1, 1)
                        else:
                            tmp_results[name] = np.concatenate([tmp_results[name], log_ratios[i].reshape(-1, 1)], axis=1)
                            
                        time_results[name].append(times[i])
                            
                for name in names:
                    if name == "true":
                        key =  "- - -"
                    elif name == "direct":
                        key = str(sigma_min) + " - " + ("gaussian" if requires_noise else "dirac")
                    elif name == "ct own":
                        key = str(sigma_min) + " " + score_loss_type + " " + ("gaussian" if requires_noise else "dirac")
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

    with open(f"/home/icb/egor.antipov/scFM_density_estimation/notebooks/tests/table_results/{cfg.run_type}_results_time.pkl", "wb") as f:
        pickle.dump(results, f)

    print("FINISHED")
    
if __name__ == "__main__":
    main()
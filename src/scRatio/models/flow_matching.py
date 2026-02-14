import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, Optional
from torchdyn.core import NeuralODE
from .node_wrappers import NODEWrapper, NODEWrapper_with_ratio_tvf, NODEWrapper_with_trace_div

def sinusoidal_time_features(t: torch.Tensor, num_freqs: int = 128, max_period: int = 10000):
    if len(t.shape)==1:
        t = t.unsqueeze(1)
        
    half = num_freqs // 2
    freqs = torch.exp(
        -np.log(max_period)
        * torch.arange(start=0, 
                       end=half, 
                       dtype=torch.float32, 
                       device=t.device)
        / half
    )
    args = t.float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if num_freqs % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class FlowMatchingMLP(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dims: list = [],
                 output_dim: int = 128, dropout: float = 0):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.SELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        if t.dim() == 0 or t.size()[0] == 1:
            t = t.expand(x.shape[0]).unsqueeze(1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)
        
        xt = torch.cat([x, t], dim=1)
        return self.mlp(xt)

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
        encoder_out_dim_cond: int, 
        time_feature_dim: int, 
        lambda_t: Callable,
        lambda_sp_t: Callable,
        betas: list,
        lr: float = 1e-4,
        dropout: float = 0
    ):
        super().__init__()

        self.data_encoder = Encoder(input_dim, encoder_hidden_dims, encoder_out_dim, dropout)
        self.cond_encoders = nn.ModuleList([
            Encoder(cond_dim, encoder_hidden_dims, encoder_out_dim_cond, dropout)
            for cond_dim in cond_dims
        ])
        self.vf_mlp = FlowMatchingMLP(encoder_out_dim + encoder_out_dim_cond * len(cond_dims) + time_feature_dim, hidden_dims, input_dim, dropout)
        self.score_mlp = FlowMatchingMLP(encoder_out_dim + encoder_out_dim_cond * len(cond_dims) + time_feature_dim, hidden_dims, input_dim, dropout)
        
        self.lambda_t = lambda_t
        self.lambda_sp_t = lambda_sp_t
        self.time_feature_dim = time_feature_dim
        self.betas = betas
        self.encoder_out_dim = encoder_out_dim
        self.encoder_out_dim_cond = encoder_out_dim_cond
        self.cond_dims = cond_dims
        self.lr = lr

    def forward(self, x, t, cond, use_conds=None):
        if use_conds is None:
            use_conds = [True for _ in range(len(self.cond_dims))]

        if t.dim() == 0 or t.size()[0] == 1:
            t = t.expand(x.shape[0]).unsqueeze(1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)

        start = 0
        xc = self.data_encoder(x)
        for i, cond_dim in enumerate(self.cond_dims):
            if use_conds[i]:
                cond_enc = self.cond_encoders[i](cond[:, start:(start + cond_dim)])
            else:
                cond_enc = torch.zeros(xc.shape[0], self.encoder_out_dim_cond).to(xc.device).float()
                
            xc = torch.cat([xc, cond_enc], dim=1)
            start += cond_dim

        if self.time_feature_dim > 1:
            xtc = torch.cat([xc, sinusoidal_time_features(t, self.time_feature_dim)], dim=1)
        else:
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

        use_conds = (np.random.uniform(size=len(self.betas)) >= np.array(self.betas)).tolist()
        pred_ut, pred_score = self(xt, t, cond, use_conds)
        
        vf_loss = F.mse_loss(pred_ut, ut)
        score_loss = F.mse_loss(c_t * pred_score, t * ut - xt)

        return vf_loss + score_loss

    def _unpack_batch(self, batch):
        if isinstance(batch, (tuple, list)):
            if len(batch) != 2:
                raise ValueError("Expected batch to be (x, cond).")
            x, cond = batch
        else:
            raise ValueError("Expected batch to be (x, cond).")
        return x, cond

    def training_step(self, batch, batch_idx):
        x, cond = self._unpack_batch(batch)
        loss = self.shared_step(x, cond)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, cond = self._unpack_batch(batch)
        loss = self.shared_step(x, cond)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, cond = self._unpack_batch(batch)
        loss = self.shared_step(x, cond)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def get_node(self, condition, control=None, point=None, node_type="simulation", estimator_type="exact"):
        if node_type == "simulation":
            return NeuralODE(NODEWrapper(self, condition), solver="dopri5",
                              sensitivity="adjoint", atol=1e-4, rtol=1e-4)
        elif node_type == "density":
            return NeuralODE(NODEWrapper_with_trace_div(self, condition, estimator_type), solver="dopri5",
                              sensitivity="adjoint", atol=1e-4, rtol=1e-4)
        elif node_type == "ratio":
            return NeuralODE(NODEWrapper_with_ratio_tvf(self, condition, control, point, estimator_type), solver="dopri5",
                              sensitivity="adjoint", atol=1e-4, rtol=1e-4)
        
    def run_simulation(self, data_samples, condition, n_steps=100):
        device = data_samples.device
        node = self.get_node(condition, node_type="simulation")
        
        with torch.no_grad():
            traj = node.trajectory(
                torch.randn_like(data_samples).to(device),
                t_span=torch.linspace(0, 1, n_steps).to(device)
            )
            
        return traj[-1]
        
    def estimate_log_density(self, data_samples, condition, n_steps=100):
        device = data_samples.device
        node = self.get_node(condition, node_type="density", estimator_type="hutch_gaussian")
        
        with torch.no_grad():
            traj = node.trajectory(
                torch.cat([data_samples, torch.zeros(data_samples.shape[0], 1).to(device)], dim=-1),
                t_span=torch.linspace(1, 0, n_steps).to(device)
            )
        z0, div = traj[-1, :, :-1], traj[-1, :, -1]
        log_p1 = -0.5 * (z0 ** 2).sum(dim=1) - 0.5 * z0.shape[1] * np.log(2 * np.pi) + div
        
        return log_p1.cpu().numpy()
    
    def estimate_log_density_ratio(self, data_samples, condition, control, point, n_steps=100):
        device = data_samples.device
        node = self.get_node(condition, control, point, node_type="ratio", estimator_type="hutch_gaussian")
        
        with torch.no_grad():
            traj = node.trajectory(
                torch.cat([data_samples, torch.zeros(data_samples.shape[0], 1).to(device)], dim=-1),
                t_span=torch.linspace(1, 0, n_steps).to(device)
            )
        log_ratio = traj[-1, :, -1]
        
        return -log_ratio.cpu().numpy()
    
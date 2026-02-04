from typing import Callable

import lightning as L
from sklearn.model_selection import train_test_split
import torch
from torch import nn

from scFM_density_estimation.models import *

def sinusoidal_time_features(t: torch.Tensor, 
                             num_freqs: int = 128, 
                             max_period: int = 10000):
    if len(t.shape)==1:
        t = t.unsqueeze(1)
        
    half = num_freqs // 2
    freqs = torch.exp(
        -math.log(max_period)
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
        use_sinusoidal_embeddings: bool, 
        sinusoidal_feature_dim: int, 
        lambda_t: Callable,
        lambda_sp_t: Callable,
        betas: list,
        lr: float = 1e-4,
        use_ot_sampler: bool = False,
        ot_method: str = "exact",
        dropout: float = 0,
        unconditional: bool = False
    ):
        super().__init__()
        self.save_hyperparameters()
        
        if use_sinusoidal_embeddings:
            self.sinusoidal_feature_dim = sinusoidal_feature_dim
        else:
            self.sinusoidal_feature_dim = 1

        self.data_encoder = Encoder(input_dim, encoder_hidden_dims, encoder_out_dim, dropout)
        if not unconditional:
            self.cond_encoders = nn.ModuleList([
                Encoder(cond_dim, encoder_hidden_dims, encoder_out_dim_cond, dropout)
                for cond_dim in cond_dims
            ])
        else: 
            encoder_out_dim_cond = 0
            
        self.vf_mlp = FlowMatchingMLP(encoder_out_dim + encoder_out_dim_cond * len(cond_dims) + self.sinusoidal_feature_dim, hidden_dims, input_dim, dropout)
        self.score_mlp = FlowMatchingMLP(encoder_out_dim + encoder_out_dim_cond * len(cond_dims) + self.sinusoidal_feature_dim, hidden_dims, input_dim, dropout)
        
        self.lambda_t = lambda_t
        self.lambda_sp_t = lambda_sp_t
        self.use_sinusoidal_embeddings = use_sinusoidal_embeddings
        self.betas = betas
        self.encoder_out_dim = encoder_out_dim
        self.encoder_out_dim_cond = encoder_out_dim_cond
        self.use_ot_sampler = use_ot_sampler
        self.cond_dims = cond_dims
        self.lr = lr
        self.unconditional = unconditional

        self.vf_losses = []
        self.score_losses = []

    def forward(self, x, t, cond, use_conds=[True]):
        if t.dim() == 0 or t.size()[0] == 1:
            t = t.expand(x.shape[0]).unsqueeze(1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)

        start = 0
        xc = self.data_encoder(x)
        
        if not self.unconditional:
            for i, cond_dim in enumerate(self.cond_dims):
                if use_conds[i]:
                    cond_enc = self.cond_encoders[i](cond[:, start:(start + cond_dim)])
                else:
                    cond_enc = torch.zeros(xc.shape[0], self.encoder_out_dim_cond)
                    
                xc = torch.cat([xc, cond_enc], dim=1)
                start += cond_dim

        if self.use_sinusoidal_embeddings:
            xtc = torch.cat([xc, sinusoidal_time_features(t, self.sinusoidal_feature_dim)], dim=1)
        else:
            xtc = torch.cat([xc, t], dim=1)
        
        vf = self.vf_mlp.mlp(xtc)
        score = self.score_mlp.mlp(xtc)
        return vf, score

    def shared_step(self, x1, cond, step):
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

        if step % 100 == 0:
            self.vf_losses.append(vf_loss.item())
            self.score_losses.append(score_loss.item())

        return vf_loss + score_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def div_fn_hutch_trace(u):
    def div_fn(x, eps):
        _, vjpfunc = torch.func.vjp(u, x)
        return (vjpfunc(eps)[0] * eps).sum()

    return div_fn

def div_fn_hutch_trace_with_cond(u):
    def div_fn(x, cond, eps):
        _, vjpfunc = torch.func.vjp(u(cond), x)
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

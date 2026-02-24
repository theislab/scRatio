from typing import Callable

import lightning as L
from sklearn.model_selection import train_test_split
import torch
from torch import nn

from scFM_density_estimation.models import *


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

    def forward(self, x, t, cond, use_conds, return_score=True):
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
        if return_score:
            return vf, score
        return vf

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
    def __init__(self, model, cond, use_conds):
        super().__init__()
        self.model = model
        self.cond = cond
        self.div_fn, self.eps_fn = div_fn_hutch_trace, torch.randn_like
        self.use_conds = use_conds

    def forward(self, t, x, *args, **kwargs):
        x = x[..., :-1]
        
        def vecfield(y):
            vf, _ = self.model(y.unsqueeze(0), t, self.cond[:1], self.use_conds)
            return vf.squeeze()

        div = torch.vmap(self.div_fn(vecfield))(x, self.eps_fn(x))
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
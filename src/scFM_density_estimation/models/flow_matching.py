import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
import matplotlib.pyplot as plt

from .optimal_transport import OTPlanSampler, wasserstein
from torchdyn.core import NeuralODE
from umap import UMAP

#---Unconditional model---
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
    
class FlowMatching(L.LightningModule):
    def __init__(self, input_dim: int = 128, hidden_dims: list = [],
                 lr: float = 1e-3, dropout: float = 0):
        super().__init__()
        self.save_hyperparameters()
        self.model = FlowMatchingMLP(input_dim + 1, hidden_dims, input_dim, dropout)
        self.lr = lr

    def forward(self, x, t):
        return self.model(x, t)

    def shared_step(self, x1):
        # sample x0 and t
        x0 = torch.randn_like(x1).to(x1.device)
        t = torch.rand(x1.shape[0]).to(x1.device)

        # compute xt and ut
        xt = x0 + t * (x1 - x0)
        ut = x1 - x0
        pred_ut = self(xt, t)

        loss = F.mse_loss(pred_ut, ut)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('val/loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('test/loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
#---Conditional model---
class ConditionEncoder(nn.Module):
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
    
class ConditionalFlowMatching(L.LightningModule):
    def __init__(self,
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
                 ):
        super().__init__()
        self.save_hyperparameters()
        if not use_encoder:
            cond_out_dim = cond_dim
        
        self.cond_encoder = ConditionEncoder(cond_dim, cond_hidden_dims, cond_out_dim, dropout)
        self.mlp = FlowMatchingMLP(input_dim + 1 + cond_out_dim, hidden_dims, input_dim, dropout)
        self.ot_sampler = OTPlanSampler(method=ot_method)
        
        self.use_encoder = use_encoder
        self.use_ot_sampler = use_ot_sampler
        self.cond_dim = cond_dim
        self.lr = lr

    def forward(self, x, t, cond):
        if t.dim() == 0 or t.size()[0] == 1:
            t = t.expand(x.shape[0]).unsqueeze(1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)
            
        if self.use_encoder:
            cond_enc = self.cond_encoder(cond)
        else:
            cond_enc = cond
            
        xtc = torch.cat([x, t, cond_enc], dim=1)
        return self.mlp.mlp(xtc)

    def shared_step(self, x1, cond):
        device = x1.device
        
        # sample x0 and t
        x0 = torch.randn_like(x1).to(device)
        t = torch.rand(x1.shape[0]).unsqueeze(1).to(device)
        
        # use ot sampler
        if self.use_ot_sampler:
            x0, x1, cond = self.sample_ot(x0, x1, cond)
        
        # compute xt and ut
        xt = x0 + t * (x1 - x0)
        ut = x1 - x0
        pred_ut = self(xt, t, cond)
        
        loss = F.mse_loss(pred_ut, ut)
        return loss

    def training_step(self, batch, batch_idx=0):
        x1, cond = batch
        loss = self.shared_step(x1, cond)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx=0):
        x1, cond = batch
        loss = self.shared_step(x1, cond)
        ws_dist = self.weighted_wasserstein(x1, cond)
        
        self.log('val/loss', loss)
        self.log('val/wasserstein_distance', ws_dist)
        
        return loss
    
    def test_step(self, batch, batch_idx=0):
        x1, cond = batch
        loss = self.shared_step(x1, cond)
        ws_dist = self.weighted_wasserstein(x1, cond)
        classif_test = self.classification_test(x1, cond, x1.device)
        
        self.log('test/loss', loss)
        self.log('test/wasserstein_distance', ws_dist)
        self.log('test/classification_test', classif_test)
        
        return loss
    
    def weighted_wasserstein(self, X, C):
        generated_samples = self.run_simulation(X, C, n_steps=100)
        
        ws_dist = 0
        combined_conditions = C.cpu().numpy()
        unique_conditions = np.unique(combined_conditions, axis=0)
        
        for condition in unique_conditions:
            mask = np.all(combined_conditions == condition, axis=1)
            ws_dist += wasserstein(X[mask], generated_samples[mask]) * np.sum(mask)

        return ws_dist / C.shape[0]
    
    def classification_test(self, x1, cond_orig, device):
        log_density = []
        unique_conditions = np.unique(cond_orig.cpu().numpy(), axis=0)
        for condition in unique_conditions:
            cond = torch.from_numpy(condition).expand(cond_orig.shape).float().to(device)
            log_density.append(self.estimate_log_density(x1, cond, n_steps=100).reshape(-1, 1))
        log_density = np.concatenate(log_density, axis=1)
        color = np.argmax(log_density, axis=1)

        color_orig = np.zeros_like(color)
        for i in range(unique_conditions.shape[0]):
            mask = np.all(cond_orig.cpu().numpy() == unique_conditions[i], axis=1)
            color_orig[mask] = i
            
        return np.sum(color == color_orig) / color.shape[0]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def sample_ot(self, x0, x1, cond):
        device = x1.device
        x0_final = []
        x1_final = []
        cond_final = []
        conditions = np.unique(cond.cpu().numpy(), axis=0)
        
        for condition in conditions:
            mask = np.all(cond.cpu().numpy() == condition, axis=1)
            x0_tmp, x1_tmp = x0[mask], x1[mask]
            x0_tmp, x1_tmp = self.ot_sampler.sample_plan(x0_tmp, x1_tmp)
            
            x0_final.append(x0_tmp)
            x1_final.append(x1_tmp)
            cond_final.append(cond[mask])
        
        x0_final = torch.cat(x0_final, dim=0).to(device)
        x1_final = torch.cat(x1_final, dim=0).to(device)
        cond_final = torch.cat(cond_final, dim=0).to(device)
        return x0_final, x1_final, cond_final
    
    def get_node(self, cond, node_type="simulation", estimator_type="exact"):
        if node_type == "simulation":
            return NeuralODE(NODEWrapper(self, cond), solver="dopri5",
                              sensitivity="adjoint", atol=1e-4, rtol=1e-4)
        elif node_type == "density":
            return NeuralODE(NODEWrapper_with_trace_div(self, cond, estimator_type), solver="dopri5",
                              sensitivity="adjoint", atol=1e-4, rtol=1e-4)
        
    def get_umap_reducer(self, random_state=42):
        self.reducer = UMAP(n_neighbors=15, min_dist=0.5, n_components=2, random_state=random_state)
        
    def run_simulation(self, data_samples, cond, n_steps=100):
        device = data_samples.device
        node = self.get_node(cond)
        
        with torch.no_grad():
            traj = node.trajectory(
                torch.randn_like(data_samples).to(device),
                t_span=torch.linspace(0, 1, n_steps).to(device)
            )
            
        return traj[-1]
        
    def estimate_log_density(self, data_samples, cond, n_steps=100):
        device = data_samples.device
        node = self.get_node(cond, node_type="density", estimator_type="hutch_gaussian")
        
        with torch.no_grad():
            traj = node.trajectory(
                torch.cat([data_samples, torch.zeros(data_samples.shape[0], 1).to(device)], dim=-1),
                t_span=torch.linspace(1, 0, n_steps).to(device)
            )
        z0, div = traj[-1][:, :-1], traj[-1][:, -1]
        log_p1 = -0.5 * (z0 ** 2).sum(dim=1) - 0.5 * z0.shape[1] * np.log(2 * np.pi) + div
        
        return log_p1.cpu().numpy()
    
class NODEWrapper(torch.nn.Module):
    """
    Wraps model to torchdyn compatible format.
    """
    def __init__(self, model, cond):
        super().__init__()
        self.model = model
        self.cond = cond

    def forward(self, t, x, *args, **kwargs):
        return self.model(x, t, self.cond)
    
def exact_div_fn(u):
    """Accepts a function u:R^D -> R^D."""
    J = torch.func.jacrev(u)
    return lambda x, *args: torch.trace(J(x))


def div_fn_hutch_trace(u):
    def div_fn(x, eps):
        _, vjpfunc = torch.func.vjp(u, x)
        return (vjpfunc(eps)[0] * eps).sum()

    return div_fn

class NODEWrapper_with_trace_div(torch.nn.Module):
    """
    Wraps model to torchdyn compatible format with trace of the Jacobian.
    """
    def __init__(self, model, cond, likelihood_estimator="exact"):
        super().__init__()
        self.model = model
        self.cond = cond
        self.div_fn, self.eps_fn = self.get_div_and_eps(likelihood_estimator)

    def get_div_and_eps(self, likelihood_estimator):
        if likelihood_estimator == "exact":
            return exact_div_fn, None
        if likelihood_estimator == "hutch_gaussian":
            return div_fn_hutch_trace, torch.randn_like
        if likelihood_estimator == "hutch_rademacher":

            def eps_fn(x):
                return torch.randint_like(x, low=0, high=2).float() * 2 - 1.0

            return div_fn_hutch_trace, eps_fn
        raise NotImplementedError(
            f"likelihood estimator {likelihood_estimator} is not implemented"
        )

    def forward(self, t, x, *args, **kwargs):
        x = x[..., :-1]
        
        def vecfield(y):
            return self.model(y.unsqueeze(0), t, self.cond[:1]).squeeze()

        if self.eps_fn is None:
            div = torch.vmap(self.div_fn(vecfield))(x)
        else:
            div = torch.vmap(self.div_fn(vecfield))(x, self.eps_fn(x))
        dx = self.model(x, t, self.cond)
        return torch.cat([dx, div[:, None]], dim=-1)
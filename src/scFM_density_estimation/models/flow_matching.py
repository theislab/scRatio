import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

# --- Unconditional Model ---
class FlowMatchingMLP(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dims: list = [], output_dim: int = 128):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.SELU(),
                nn.Dropout(0)
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


class FlowMatchingLightningModule(L.LightningModule):
    def __init__(self, input_dim: int = 128, hidden_dims: list = [], lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = FlowMatchingMLP(input_dim + 1, hidden_dims, input_dim)
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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# --- Conditional Model ---
class ConditionEncoder(nn.Module):
    def __init__(self, cond_dim: int = 1, cond_hidden_dims: list = [], cond_out_dim: int = 2):
        super().__init__()
        layers = []
        prev_dim = cond_dim
        for dim in cond_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.SELU(),
                nn.Dropout(0)
            ])
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, cond_out_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, cond):
        return self.encoder(cond)

class ConditionalFlowMatchingMLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 cond_dim: int,
                 hidden_dims: list = [64],
                 cond_hidden_dims: list = [16],
                 cond_out_dim: int = 8,
                 use_encoder: bool = False,
                 ):
        super().__init__()
        self.cond_encoder = ConditionEncoder(cond_dim, cond_hidden_dims, cond_out_dim)
        
        if not use_encoder:
            cond_out_dim = cond_dim
            
        self.mlp = FlowMatchingMLP(input_dim + 1 + cond_out_dim, hidden_dims, input_dim)
        self.use_encoder = use_encoder

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

class ConditionalFlowMatchingLightningModule(L.LightningModule):
    def __init__(self,
                 input_dim: int,
                 cond_dim: int,
                 hidden_dims: list = [],
                 cond_hidden_dims: list = [16],
                 cond_out_dim: int = 8,
                 lr: float = 1e-3,
                 use_encoder: bool = False,
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.model = ConditionalFlowMatchingMLP(input_dim, cond_dim, hidden_dims, cond_hidden_dims, cond_out_dim)
        self.lr = lr

    def forward(self, x, t, cond):
        return self.model(x, t, cond)

    def shared_step(self, x1, cond):
        # sample x0 and t
        x0 = torch.randn_like(x1).to(x1.device)
        t = torch.rand(x1.shape[0]).to(x1.device)
        
        # cumpute xt and ut
        xt = x0 + t * (x1 - x0)
        ut = x1 - x0
        pred_ut = self(xt, t, cond)
        
        loss = F.mse_loss(pred_ut, ut)
        return loss

    def training_step(self, batch, batch_idx):
        x1, cond = batch
        loss = self.shared_step(x1, cond)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x1, cond = batch
        loss = self.shared_step(x1, cond)
        self.log('val/loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

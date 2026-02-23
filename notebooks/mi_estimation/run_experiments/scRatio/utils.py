import torch 
from scRatio.models.node_wrappers import div_fn_hutch_trace

class NODEWrapper_with_ratio_generic_models(torch.nn.Module):
    def __init__(self, model_num, model_den, model_vf):
        super().__init__()
        self.model_den = model_den
        self.model_num = model_num
        self.model_vf = model_vf
        self.div_fn, self.eps_fn = div_fn_hutch_trace, torch.randn_like

    def forward(self, t, x, *args, **kwargs):
        x = x[..., :-1]
        
        def vecfield(y):
            ut, _ = self.model_num(y.unsqueeze(0), t, None)
            vt, _ = self.model_den(y.unsqueeze(0), t, None)
            return vt.squeeze() - ut.squeeze()
            
        div = torch.vmap(self.div_fn(vecfield))(x, self.eps_fn(x))
        
        ut, score_u = self.model_num(x, t, None)
        vt, score_v = self.model_den(x, t, None)
        ft, _ = self.model_vf(x, t, None)
        
        correction_term_u = torch.linalg.vecdot(ft - ut, score_u)
        correction_term_v = torch.linalg.vecdot(vt - ft, score_v)
        dr = div + correction_term_u + correction_term_v
        
        return torch.cat([ft, dr[:, None]], dim=-1)
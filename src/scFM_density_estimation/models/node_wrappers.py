import torch

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

def get_div_and_eps(likelihood_estimator):
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

class NODEWrapper_with_trace_div(torch.nn.Module):
    """
    Wraps model to torchdyn compatible format with trace of the Jacobian.
    """
    def __init__(self, model, condition, likelihood_estimator="exact"):
        super().__init__()
        self.model = model
        self.cond = condition
        self.div_fn, self.eps_fn = get_div_and_eps(likelihood_estimator)

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
    
class NODEWrapper_with_ratio(torch.nn.Module):
    def __init__(self, model, condition, control, likelihood_estimator="exact"):
        super().__init__()
        self.model = model
        self.cond_v = control
        self.cond_u = condition
        self.div_fn, self.eps_fn = get_div_and_eps(likelihood_estimator)

    def forward(self, t, x, *args, **kwargs):
        x = x[..., :-1]
        
        def vecfield(y):
            return self.model(y.unsqueeze(0), t, self.cond_v[:1]).squeeze() \
            - self.model(y.unsqueeze(0), t, self.cond_u[:1]).squeeze()
        
        if self.eps_fn is None:
            div = torch.vmap(self.div_fn(vecfield))(x)
        else:
            div = torch.vmap(self.div_fn(vecfield))(x, self.eps_fn(x))
            
        ut = self.model(x, t, self.cond_u)
        vt = self.model(x, t, self.cond_v)
        
        score = (t * vt - x) / (1 - t + 1e-8)
        correction_term = torch.linalg.vecdot(vt - ut, score)
        dr = div + correction_term
        
        return torch.cat([ut, dr[:, None]], dim=-1)
    
class NODEWrapper_with_ratio_tvf(torch.nn.Module):
    def __init__(self, model, condition, control, point, likelihood_estimator="exact"):
        super().__init__()
        self.model = model
        self.cond_v = control
        self.cond_u = condition
        self.cond_f = point
        self.div_fn, self.eps_fn = get_div_and_eps(likelihood_estimator)

    def forward(self, t, x, *args, **kwargs):
        x = x[..., :-1]
        
        def vecfield(y):
            return self.model(y.unsqueeze(0), t, self.cond_v[:1]).squeeze() \
            - self.model(y.unsqueeze(0), t, self.cond_u[:1]).squeeze()
        
        if self.eps_fn is None:
            div = torch.vmap(self.div_fn(vecfield))(x)
        else:
            div = torch.vmap(self.div_fn(vecfield))(x, self.eps_fn(x))
            
        ut = self.model(x, t, self.cond_u)
        vt = self.model(x, t, self.cond_v)
        ft = self.model(x, t, self.cond_f)
        
        score_u = (t * ut - x) / (1 - t + 1e-8)
        correction_term_u = torch.linalg.vecdot(ft - ut, score_u)
        
        score_v = (t * vt - x) / (1 - t + 1e-8)
        correction_term_v = torch.linalg.vecdot(vt - ft, score_v)
        
        dr = div + correction_term_u + correction_term_v
        
        return torch.cat([ft, dr[:, None]], dim=-1)
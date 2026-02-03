import torch
import numpy as np
from torch.distributions import MultivariateNormal, Normal, Independent
from torch.distributions.transforms import ReshapeTransform
from torch.distributions import TransformedDistribution

class CorrelatedGaussianMI:
    """
    Block-diagonal correlated Gaussian MI benchmark (TRE-style).

    p(x): correlated Gaussian in R^dim with 2x2 blocks
    q(x): standard normal in R^dim

    Mutual information:
        I = E_p[ log p(x) - log q(x) ]

    MI is determined internally by dimensionality, matching
    Telescoping Density-Ratio Estimation (Section 4.2).
    """

    # TRE experiment schedule
    DIM_TO_MI = {
        20: 5,
        40: 10,
        80: 20,
        160: 40,
        320: 80,
    }

    def __init__(self, dim, device="cpu"):
        assert dim in self.DIM_TO_MI, (
            f"dim must be one of {list(self.DIM_TO_MI.keys())}"
        )
        assert dim % 2 == 0, "Dimension must be even."

        self.dim = dim
        self.device = device
        self.target_mi = self.DIM_TO_MI[dim]

        # Correlation coefficient rho from MI
        self.rho = self._rho_from_mi(self.target_mi, dim)

        # --- p(x): correlated Gaussian ---
        block_cov = torch.tensor(
            [[[1.0, self.rho], [self.rho, 1.0]]] * (dim // 2),
            dtype=torch.float32,
            device=device,
        )

        base_dist = Independent(
            MultivariateNormal(
                loc=torch.zeros(
                    dim // 2, 2,
                    dtype=torch.float32,
                    device=device
                ),
                covariance_matrix=block_cov,
            ),
            1,
        )

        self.p_dist = TransformedDistribution(
            base_dist,
            [ReshapeTransform((dim // 2, 2), (dim,))]
        )

        # --- q(x): standard normal ---
        self.q_dist = Independent(
            Normal(
                loc=torch.zeros(dim, device=device),
                scale=torch.ones(dim, device=device),
            ),
            1,
        )

    # ---------- MI / rho relations ----------

    @staticmethod
    def _rho_from_mi(mi, dim):
        """
        Invert:
            I = (dim / 2) * (-1/2) * log(1 - rho^2)
        """
        return np.sqrt(1.0 - np.exp(-4.0 * mi / dim))

    @staticmethod
    def mi_from_rho(rho, dim):
        return -0.25 * dim * np.log(1.0 - rho ** 2)

    def true_mi(self):
        """Analytic mutual information."""
        return self.mi_from_rho(self.rho, self.dim)

    # ---------- Sampling ----------

    def sample_p(self, n):
        """Sample from p(x) (correlated Gaussian)."""
        return self.p_dist.sample((n,))

    def sample_q(self, n):
        """Sample from q(x) (standard normal)."""
        return self.q_dist.sample((n,))

    # ---------- Log probabilities ----------

    def log_p(self, x):
        return self.p_dist.log_prob(x)

    def log_q(self, x):
        return self.q_dist.log_prob(x)

    # ---------- Monte Carlo MI ----------

    def empirical_mi(self, n_samples=100_000):
        """
        Monte Carlo estimate of:
            I = E_p[log p(x) - log q(x)]
        """
        with torch.no_grad():
            x = self.sample_p(n_samples)
            mi = torch.mean(self.log_p(x) - self.log_q(x))
        return mi.item()

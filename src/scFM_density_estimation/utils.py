import math

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import anndata as ad
import pandas as pd
import scanpy as sc
import torch
import torchdyn
from torchdyn.datasets import generate_moons

def eight_normal_sample(n, dim, scale=1, var=1):
    m = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(dim), math.sqrt(var) * torch.eye(dim)
    )
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
    ]
    centers = torch.tensor(centers) * scale
    noise = m.sample((n,))
    multi = torch.multinomial(torch.ones(8), n, replacement=True)
    data = []
    for i in range(n):
        data.append(centers[multi[i]] + noise[i])
    data = torch.stack(data)
    return data


def sample_moons(n):
    x0, _ = generate_moons(n, noise=0.2)
    return x0 * 3 - 1


def sample_8gaussians(n):
    return eight_normal_sample(n, 2, scale=5, var=0.1).float()


class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model):
        super().__init__()
        self.model = model.to("cuda")

    def forward(self, t, x, *args, **kwargs):
        return self.model(x, t)
    
class cond_torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model, condition, n_conds):
        super().__init__()
        self.model = model.to("cuda")
        self.condition = condition
        self.n_conds = n_conds

    def forward(self, t, x, *args, **kwargs):
        cond = torch.from_numpy(np.array([1 if i == self.condition else 0 for i in range(self.n_conds)]
                                         ).reshape(1, -1)
                                ).float().expand(x.shape[0], self.n_conds).to(x.device)
        return self.model(x, t, cond)

def plot_trajectories(traj):
    """Plot trajectories of some selected samples."""
    n = 2000
    plt.figure(figsize=(6, 6))
    plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.8, c="black")
    plt.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.2, alpha=0.2, c="olive")
    plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=4, alpha=1, c="blue")
    plt.legend(["Prior sample z(S)", "Flow", "z(0)"])
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
def plot_simulation(generated_samples, data_samples, condition, size=1, plot_size=(6, 6)):
    color = np.zeros(condition.shape[0])
    unique_conditions = np.unique(condition, axis=0)
    for i in range(unique_conditions.shape[0]):
        mask = np.all(condition == unique_conditions[i], axis=1)
        color[mask] = i
    
    # Show generated samples and data samples
    plt.figure(figsize=plot_size)
    plt.scatter(generated_samples[:, 0], generated_samples[:, 1], s=size*0.8, alpha=0.8, c="black")
    plt.scatter(data_samples[:, 0], data_samples[:, 1], s=size, alpha=1, c="blue")
    plt.legend(["generated", "true"])
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
    # Show generated samples colored according to condition
    plt.figure(figsize=plot_size)
    plt.title("Generated samples")
    plt.scatter(generated_samples[:, 0], generated_samples[:, 1], s=size, c=color)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
    # Show data samples colored according to condition
    plt.figure(figsize=plot_size)
    plt.title("Data samples")
    plt.scatter(data_samples[:, 0], data_samples[:, 1], s=size, c=color)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
def plot_densities(data_samples, densities, size=1, plot_size=(6, 6)):
    # Show data samples colored according to density
    plt.figure(figsize=plot_size)
    plt.title("Density estimation")
    plt.scatter(data_samples[:, 0], data_samples[:, 1], s=size, c=densities, cmap="viridis")
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
def plot_simulation_scanpy(generated_samples, data_samples, condition, cond_dims,
                           num_cond, subsample_size=None, size=1, figsize=(6, 6)):
    labels = []
    start = 0
    for i in range(num_cond):
        labels.append(np.argmax(condition[:, start:start+cond_dims[i]], axis=1))
        start += cond_dims[i]
    labels = np.stack(labels, axis=1)
    labels = ["_".join(map(str, list(label))) for label in labels]
    
    obs = {"data_type": ["Real" for _ in range(data_samples.shape[0])] + ["Generated" for _ in range(generated_samples.shape[0])],
           "condition": labels + labels}
    adata = ad.AnnData(X=np.concatenate([data_samples, generated_samples], axis=0), obs=obs)
    adata.obsm["X_pca"] = adata.X.copy()

    if subsample_size is not None:
        sc.pp.subsample(adata, subsample_size)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    
    plt.rcParams['figure.figsize'] = figsize
    
    sc.pl.umap(adata, color="data_type", s=size, title="Comparison")
    sc.pl.umap(adata[adata.obs.data_type == "Generated"], color="condition", s=size, title="Generated")
    sc.pl.umap(adata[adata.obs.data_type == "Real"], color="condition", s=size, title="Real")
    
def plot_classification_scanpy(data_samples, color, color_orig, entropy,
                               subsample_size=None, size=1, figsize=(6, 6)):
    obs = {"color": list(map(str, color)), "color_orig": list(map(str, color_orig)), "entropy": entropy}
    adata = ad.AnnData(X=data_samples, obs=obs)
    adata.obsm["X_pca"] = adata.X.copy()
    
    plt.rcParams['figure.figsize'] = figsize
    
    if subsample_size is not None:
        sc.pp.subsample(adata, subsample_size)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    
    sc.pl.umap(adata, color="color", s=size, title="Generated")
    sc.pl.umap(adata, color="color_orig", s=size, title="True")
    sc.pl.umap(adata, color="entropy", s=size, cmap="viridis", title="Entropy")
    
def plot_densities_scanpy(data_samples, densities, subsample_size=None,
                          size=1, figsize=(6, 6), title="Density", **kwargs):
    obs = {"density": densities}
    adata = ad.AnnData(X=data_samples, obs=obs)
    adata.obsm["X_pca"] = adata.X.copy()
    
    plt.rcParams['figure.figsize'] = figsize
    
    if subsample_size is not None:
        sc.pp.subsample(adata, subsample_size)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.pl.umap(adata, color="density", s=size, cmap="viridis", title=title, **kwargs)
    
def plot_density_ratios_scanpy(data_samples, densities, mask, subsample_size=None, cmap="viridis",
                               size=1, figsize=(6, 6), title="Density ratio", **kwargs):
    densities[~mask] = None
    obs = {"density": densities}
    adata = ad.AnnData(X=data_samples, obs=obs)
    adata.obsm["X_pca"] = adata.X.copy()
    
    plt.rcParams['figure.figsize'] = figsize
    
    if subsample_size is not None:
        sc.pp.subsample(adata, subsample_size)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.pl.umap(adata, color="density", s=size, cmap=cmap, title=title, **kwargs)
    
def plot_boxplot_comparison(data1, data2, label1, label2):
    df = pd.DataFrame({
        'value': data1 + data2,
        'group': [label1]*len(data1) + [label2]*len(data2)
    })

    sns.boxplot(x='group', y='value', data=df)
    plt.title('Boxplot comparison')
    plt.show()
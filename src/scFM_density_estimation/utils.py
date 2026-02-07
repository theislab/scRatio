import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import anndata as ad
import pandas as pd
import scanpy as sc
    
def plot_simulation(generated_samples, data_samples, condition, size=1, plot_size=(6, 6)):
    color = np.zeros(condition.shape[0])
    unique_conditions = np.unique(condition, axis=0)
    for i in range(unique_conditions.shape[0]):
        mask = np.all(condition == unique_conditions[i], axis=1)
        color[mask] = i
    
    plt.figure(figsize=plot_size)
    plt.scatter(generated_samples[:, 0], generated_samples[:, 1], s=size*0.8, alpha=0.8, c="black")
    plt.scatter(data_samples[:, 0], data_samples[:, 1], s=size, alpha=1, c="blue")
    plt.legend(["generated", "true"])
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
    plt.figure(figsize=plot_size)
    plt.title("Generated samples")
    plt.scatter(generated_samples[:, 0], generated_samples[:, 1], s=size, c=color)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
    plt.figure(figsize=plot_size)
    plt.title("Data samples")
    plt.scatter(data_samples[:, 0], data_samples[:, 1], s=size, c=color)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
def plot_densities(data_samples, densities, size=1, plot_size=(6, 6)):
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
    
def plot_density_ratios_scanpy_new(adata, densities, mask, cmap="viridis", size=1, figsize=(6, 6),
                                   title="Density ratio", **kwargs):
    densities[~mask] = None
    adata.obs["density"] = densities
    
    plt.rcParams['figure.figsize'] = figsize
    
    sc.pl.umap(adata, color="density", s=size, cmap=cmap, title=title, **kwargs)
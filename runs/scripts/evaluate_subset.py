import hydra
import os
import torch

import anndata as ad
import scanpy as sc
import numpy as np

from tqdm import tqdm
from omegaconf import DictConfig

from scFM_density_estimation.models import ConditionalFlowMatching

@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    adata = ad.read_h5ad("/lustre/groups/ml01/projects/inverse_perturbation_models/datasets/pbmc/adata_for_cellflow_datasets_with_embeddings_and_200PCs.h5ad")
    X = adata.obsm["X_pca"][:, :cfg.datamodule.num_pcs]
    C = []
    for label in cfg.datamodule.conditions:
        C.append(adata.obs[label].cat.codes.values.copy())
    C = np.stack(C, axis=1)
    
    ckpt_path = os.path.join(cfg.training.output_dir, cfg.logger.name+"_best-checkpoint.ckpt")
    model = ConditionalFlowMatching.load_from_checkpoint(ckpt_path)
    device = model.device
    
    mask = adata.obs["cytokine"].apply(lambda x: "IFN" in x or "PBS" in x)
    x1 = torch.from_numpy(X[mask]).float().to(device)
    
    cond = []
    for i in range(cfg.datamodule.num_conditions):
        cond.append(torch.nn.functional.one_hot(torch.from_numpy(C[:, i][mask]).long(),
                                                num_classes=cfg.datamodule.condition_dims[i]).float()
                   )
    cond_orig = torch.cat(cond, dim=1).to(device)
    
    unique_conditions = np.unique(cond_orig.cpu().numpy(), axis=0)
    log_density = []
    for condition in tqdm(unique_conditions):
        cond = torch.from_numpy(condition).expand(cond_orig.shape).float().to(device)
        log_density.append(model.estimate_log_density(x1, cond, n_steps=100).reshape(-1, 1))
    log_density = np.concatenate(log_density, axis=1)
    
    code_maps = dict()
    for label in cfg.datamodule.conditions:
        drugs = list(adata.obs[label].unique())
        codes = list(adata.obs[label].cat.codes.unique())
        code_maps.update({label: dict(zip(drugs, codes))})

    adata = ad.AnnData(X=x1.cpu().numpy())
    adata.obsm["X_pca"] = adata.X.copy()
    
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    
    adata.uns["log_density"] = log_density
    adata.uns["code_maps"] = code_maps
    adata.uns["conditions"] = cond_orig.cpu().numpy()
    
    adata.write_h5ad("./outputs/evaluation_results/"+cfg.logger.name+"_subset.h5ad")

if __name__ == "__main__":
    main()

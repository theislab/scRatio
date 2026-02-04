import argparse
from pathlib import Path
from omegaconf import DictConfig
import sys
import traceback

import hydra
import scanpy as sc
from scvi.external import MRVI


def train_mrvi(adata, res_dir: Path, run_name: str):
    # Initialize sample key 
    sample_key = "treatment"

    # Each run gets its own output directory
    run_dir = res_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Setup AnnData
    MRVI.setup_anndata(
        adata,
        sample_key=sample_key,
        layer="X_counts",
        backend="torch",
    )

    # Train model
    model = MRVI(adata, backend="torch")
    model.train(max_epochs=200)

    # Latent representation
    u = model.get_latent_representation()
    adata.obsm[f"u_{run_name}"] = u

    sc.pp.neighbors(adata, use_rep=f"u_{run_name}")
    sc.tl.umap(adata)
    adata.obsm[f"X_umap_{run_name}"] = adata.obsm["X_umap"]

    # Differential abundance
    da_res = model.differential_abundance()
    trt_log_probs = da_res.log_probs.loc[{"sample": 1}]
    ctr_log_probs = da_res.log_probs.loc[{"sample": 0}]
    log_prob_ratio = trt_log_probs - ctr_log_probs

    adata.obs[f"log_ratio_{run_name}"] = log_prob_ratio

    # Save outputs
    adata.write_h5ad(run_dir / f"{run_name}.h5ad")
    model.save(run_dir, overwrite=True)

    return model

@hydra.main(config_path="/home/icb/alessandro.palma/environment/scFM_density_estimation/experiments/differential_abundance/mrvi/config", config_name="train", version_base=None)
def main(config: DictConfig):
    res_dir = Path(config.paths.res_dir)
    res_dir.mkdir(parents=True, exist_ok=True)  # make directory if doesn' exist
    
    # Read adata 
    adata_path = Path(config.paths.adata_path)
    base_adata = sc.read_h5ad(adata_path)

    adata_fname = adata_path.name
    tag = adata_fname.replace(".h5ad", "").split("_")[1]  # abundance

    for i in range(3):
        run_name = f"oversamp_{tag}_{i}"
        adata = base_adata.copy()
        train_mrvi(adata, res_dir, run_name)
    
if __name__ == "__main__":
    # running the experiment
    try: 
        main()
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)

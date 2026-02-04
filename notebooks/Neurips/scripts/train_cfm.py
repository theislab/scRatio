from datetime import datetime
import logging
import os
import sys
import random
import traceback
import uuid

from omegaconf import OmegaConf
import hydra
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import scvi
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
import torch
from torchdyn.core import NeuralODE
from tqdm.auto import tqdm 

torch.set_float32_matmul_precision('medium')


logger = logging.getLogger(__name__)


def get_cfm_model(n_dims, cond_dims, config):
    sys.path.insert(0, "/home/icb/lorenzo.consoli/repos/scFM_density_estimation/notebooks/Neurips/")
    from utils import ConditionalFlowMatchingWithScore

    # 4. Prepare interpolation functions
    lambda_t = lambda t: torch.sqrt(
        (1 - (1 - config.cfm.sigma_min) * t) ** 2 + config.cfm.sigma * t * (1 - t))
    lambda_sp_t = lambda t: (
        config.cfm.sigma * (1 - 2 * t) - 2 * (1 - config.cfm.sigma_min) * (1 - (1 - config.cfm.sigma_min) * t)) / 2

    # 5. Initialize model
    return ConditionalFlowMatchingWithScore(
        input_dim=n_dims,
        cond_dims=cond_dims,
        hidden_dims=config.cfm.hidden_dims,
        encoder_hidden_dims=config.cfm.encoder_hidden_dims,
        encoder_out_dim=config.cfm.encoder_out_dim,
        lambda_t=lambda_t,
        lambda_sp_t=lambda_sp_t,
        betas=config.cfm.betas,
        lr=config.cfm.lr,
        use_ot_sampler=config.cfm.use_ot_sampler,
    ).to("cuda")


def train_cfm_model(
    X_train,
    C_train,
    model,
    n_steps,
    batch_size
):

    # prepare for training
    optimizer = model.configure_optimizers()
    pbar = tqdm(range(n_steps))

    # iterate over the gradient steps
    for k in pbar:
        optimizer.zero_grad()

        indices = np.random.choice(range(X_train.shape[0]), size=batch_size, replace=False)
        loss = model.shared_step(X_train[indices], C_train[indices])
        
        loss.backward()
        optimizer.step()

        if (k + 1) % 100 == 0:
            pbar.set_description(f"Step {k+1}, Loss: {loss.item():.3f}")
            pbar.update()


def save_cfm_model(model, save_dir, model_name, logger,):
    model_path = os.path.join(save_dir, f"{model_name}_model.pt")
    logger.info(f"Saving model at {model_path}...")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model dumped!")


@hydra.main(
    config_path="/home/icb/lorenzo.consoli/repos/scFM_density_estimation/notebooks/Neurips/configs/cfm",
    config_name="train_cfm"
)
def main(config):
    # 1. Create run id and output dir
    run_id = uuid.uuid4().hex[:8]
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{ts}_{run_id}"
    save_dir = os.path.join(config.paths.output_dir, "cfm_runs", run_id)
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, "config.yaml")
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(config))
    logger.info(f"Starting scvi run {run_id}...")

    # 2. Set reproducibility
    logger.info(f"Reproducibility set with random seed {config.reproducibility.random_seed}")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(config.reproducibility.random_seed)
    random.seed(config.reproducibility.random_seed)
    np.random.seed(config.reproducibility.random_seed)

    # 3. Read data
    logger.info(f"Reading data from {config.paths.adata_path}...")
    adata_full = sc.read_h5ad(config.paths.adata_path)
    logger.info(f"{adata_full=}")

    # 4. Optionally select HVGs
    if config.adata_setup.hvgs_only:
        logger.info(f"Subseting gene panel to HVGs only...")
        n_genes_before = adata_full.X.shape[1]
        adata_full = adata_full[:, adata_full.var["highly_variable"]]
        n_genes_after = adata_full.X.shape[1]
        logger.info(f"{n_genes_after} HVGs selected out of the {n_genes_before} original genes.")

    # 5. Read scvi model, compute PCs on uncorrected data using latent dimensionality
    # and extract latent representation
    logger.info(f"Loading SCVI model from {config.paths.scvi_model_dir} and extracting latent representations...")
    scvi_model = scvi.model.SCVI.load(config.paths.scvi_model_dir, adata=adata_full.copy())
    adata_full.obsm[config.adata_setup.scvi_latent_key] = scvi_model.get_latent_representation()
    logger.info(f"Latent representation of shape {adata_full.obsm[config.adata_setup.scvi_latent_key].shape}")
    n_dims = adata_full.obsm[config.adata_setup.scvi_latent_key].shape[1]
    logger.info("Computing PCs...")
    sc.pp.pca(adata_full, n_comps=n_dims)
    logger.info("PCs compted!")

    # 7. Retrieve condition dimensions
    cond_dims = [
        adata_full.obs[config.adata_setup.labels_key].nunique(),
        adata_full.obs[config.adata_setup.batch_key].nunique()
    ]
    logger.info(f"Condition dimensions set to {cond_dims}")

    # 8. Retrieve condition data
    C_train = np.concatenate(
        (
            OneHotEncoder().fit_transform(adata_full.obs[[config.adata_setup.labels_key]]).toarray(),
            OneHotEncoder().fit_transform(adata_full.obs[[config.adata_setup.batch_key]]).toarray(),
        ), axis=1
    )
    C_train = torch.from_numpy(C_train).float().cuda()
    logger.info(f"Condition data of shape {C_train.shape}")

    # 9. Train model on uncorrected data
    logger.info("Fitting CFM model on uncorrected representations...")
    X_train = torch.from_numpy(adata_full.obsm["X_pca"]).float().cuda()
    logger.info(f"Train data of shape {X_train.shape}")
    logger.info("Initializing model...")
    umodel = get_cfm_model(n_dims, cond_dims, config)
    logger.info(f"Model initialized {umodel}, starting training...")
    train_cfm_model(
        X_train,
        C_train,
        umodel,
        config.train.n_steps,
        config.train.batch_size
    )
    model_name = "uncorrected"
    logger.info(f"Model trained, saving it with name \"{model_name}\" at {save_dir}...")
    save_cfm_model(umodel, save_dir, model_name=model_name, logger=logger)
    logger.info("Model saved")

    # 10. Train model on corrected data
    logger.info("Fitting CFM model on corrected representations...")
    X_train = torch.from_numpy(adata_full.obsm[config.adata_setup.scvi_latent_key]).float().cuda()
    logger.info(f"Train data of shape {X_train.shape}")
    logger.info("Initializing model...")
    cmodel = get_cfm_model(n_dims, cond_dims, config)
    logger.info(f"Model initialized {cmodel}, starting training...")
    train_cfm_model(
        X_train,
        C_train,
        cmodel,
        config.train.n_steps,
        config.train.batch_size
    )
    model_name = "corrected"
    logger.info(f"Model trained, saving it with name \"{model_name}\" at {save_dir}...")
    save_cfm_model(cmodel, save_dir, model_name=model_name, logger=logger)
    logger.info("Model saved")
    return 0


if __name__ == "__main__":

    try:
        main()
    except Exception as e:
        logger.info(f"An error occurred: {e}")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
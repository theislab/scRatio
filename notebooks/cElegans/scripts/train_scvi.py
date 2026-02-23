from datetime import datetime
import logging
import os
import sys
import random
import traceback
import uuid


import hydra
import numpy as np
import scanpy as sc
import scipy.sparse as sp



logger = logging.getLogger(__name__)


@hydra.main(
    config_path="../configs/scvi",
    config_name="train_scvi"
)
def main(config):
    # Lazy import of modules
    import torch
    import scvi
    
    # 0. Create run id 
    run_id = uuid.uuid4().hex[:8]
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{ts}_{run_id}"
    logger.info(f"Starting scvi run {run_id}...")

    # 1. Set reproducibility
    logger.info(f"Reproducibility set with random seed {config.reproducibility.random_seed}")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(config.reproducibility.random_seed)
    random.seed(config.reproducibility.random_seed)
    np.random.seed(config.reproducibility.random_seed)

    # 2. Read data
    logger.info(f"Reading data from {config.paths.adata_path}...")
    adata_full = sc.read_h5ad(config.paths.adata_path)
    logger.info(f"{adata_full=}")

    # 3. Optionally select HVGs
    if config.adata_setup.hvgs_only:
        logger.info(f"Subseting gene panel to HVGs only...")
        n_genes_before = adata_full.X.shape[1]
        adata_full = adata_full[:, adata_full.var["highly_variable"]]
        n_genes_after = adata_full.X.shape[1]
        logger.info(f"{n_genes_after} HVGs selected out of the {n_genes_before} original genes.")

    # 4. Convert data to sparse to train scvi
    if config.adata_setup.layers_key is not None:
        if sp.issparse(adata_full.layers[config.adata_setup.layers_key]):
            logger.info(f"Converting layer key {config.adata_setup.layers_key} to CSR format...")
            adata_full.layers[config.adata_setup.layers_key] = adata_full.layers[config.adata_setup.layers_key].tocsr()

    # 5. Prepare adata
    logger.info(
        "Setting up anndata for SCVI with configurations:\n"
        f"\t layer={config.adata_setup.layers_key}"
        f"\t batch_key={config.adata_setup.batch_key}"
        f"\t labels_key={config.adata_setup.labels_key}"
        f"\t size_factor_key={config.adata_setup.size_factor_key}"
        f"\t categorical_covariate_keys={config.adata_setup.categorical_covariate_keys}"
        f"\t continuous_covariate_keys={config.adata_setup.continuous_covariate_keys}"
    )
    scvi.model.SCVI.setup_anndata(
        adata_full,
        layer=config.adata_setup.layers_key,
        batch_key=config.adata_setup.batch_key,
        labels_key=config.adata_setup.labels_key,
        size_factor_key=config.adata_setup.size_factor_key,
        categorical_covariate_keys=config.adata_setup.categorical_covariate_keys,
        continuous_covariate_keys=config.adata_setup.continuous_covariate_keys
    )

    # 6. Initialize model
    logger.info(
        "Setting up SCVI model with configurations:\n"
        f"\t n_hidden={config.model.n_hidden}\n"
        f"\t n_latent={config.model.n_latent}\n"
        f"\t n_layers={config.model.n_layers}\n"
        f"\t dropout_rate={config.model.dropout_rate}\n"
        f"\t dispersion={config.model.dispersion}\n"
        f"\t gene_likelihood={config.model.gene_likelihood}\n"
        f"\t use_observed_lib_size={config.model.use_observed_lib_size}\n"
        f"\t latent_distribution={config.model.latent_distribution}\n"
        f"\t log_variational={config.model.log_variational}\n"
        f"\t encode_covariates={config.model.encode_covariates}\n"
        f"\t deeply_inject_covariates={config.model.deeply_inject_covariates}\n"
        f"\t batch_representation={config.model.batch_representation}\n"
        f"\t use_batch_norm={config.model.use_batch_norm}\n"
        f"\t use_layer_norm={config.model.use_layer_norm}\n"
        f"\t extra_payload_autotune={config.model.extra_payload_autotune}\n"
        f"\t extra_encoder_kwargs={config.model.extra_encoder_kwargs}\n"
        f"\t extra_decoder_kwargs={config.model.extra_decoder_kwargs}\n"
    )
    model = scvi.model.SCVI(
        adata_full,
        n_hidden=config.model.n_hidden,
        n_latent=config.model.n_latent,
        n_layers=config.model.n_layers,
        dropout_rate=config.model.dropout_rate,
        dispersion=config.model.dispersion,
        gene_likelihood=config.model.gene_likelihood,
        use_observed_lib_size=config.model.use_observed_lib_size,
        latent_distribution=config.model.latent_distribution,
        log_variational=config.model.log_variational,
        encode_covariates=config.model.encode_covariates,
        deeply_inject_covariates=config.model.deeply_inject_covariates,
        batch_representation=config.model.batch_representation,
        use_batch_norm=config.model.use_batch_norm,
        use_layer_norm=config.model.use_layer_norm,
        extra_payload_autotune=config.model.extra_payload_autotune,
        extra_encoder_kwargs=config.model.extra_encoder_kwargs,
        extra_decoder_kwargs=config.model.extra_decoder_kwargs,
    )

    # 7. Train model
    logger.info(
        "Training the model with configurations:\n"
        f"\t max_epochs={config.train.max_epochs}"
        f"\t train_size={config.train.train_size}"
        f"\t batch_size={config.train.batch_size}"
        f"\t early_stopping={config.train.early_stopping}"
        f"\t plan_kwargs={config.train.plan_kwargs}"
        f"\t check_val_every_n_epoch={config.train.check_val_every_n_epoch}"
    )
    model.train(
        max_epochs=config.train.max_epochs,
        train_size=config.train.train_size,
        validation_size=1-config.train.train_size,
        batch_size=config.train.batch_size,
        early_stopping=config.train.early_stopping,
        plan_kwargs=config.train.plan_kwargs,
        check_val_every_n_epoch=config.train.check_val_every_n_epoch        
    )

    # 8. Save model
    save_dir = os.path.join(config.paths.output_dir, "scvi_runs", f"dim-{config.model.n_latent}-{run_id}")
    os.makedirs(
        save_dir,
        exist_ok=True
    )
    logger.info(f"Saving model to {save_dir}...")
    model.save(
        save_dir,
        overwrite=True
    )
    return 0


if __name__ == "__main__":

    try:
        main()
    except Exception as e:
        logger.info(f"An error occurred: {e}")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
import logging
import os
import sys
import traceback

import hydra
from omegaconf import OmegaConf
import numpy as np
import scanpy as sc
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import scvi
import torch
from tqdm import tqdm
from torchdyn.core import NeuralODE
from scFM_density_estimation.models import NODEWrapper

logger = logging.getLogger(__name__)


def div_fn_hutch_trace(u):
    def div_fn(x, eps):
        _, vjpfunc = torch.func.vjp(u, x)
        return (vjpfunc(eps)[0] * eps).sum()

    return div_fn


class NODEWrapper_with_ratio_tvf_rl(torch.nn.Module):
    def __init__(self, model, C):
        super().__init__()
        self.model = model
        self.C = C
        self.div_fn, self.eps_fn = div_fn_hutch_trace, torch.randn_like

    def forward(self, t, x, **kwargs):
        x = x[..., :-1]
        
        def vecfield(y):
            ut, _ = self.model(y.unsqueeze(0), t, self.C, [True, True])
            vt, _ = self.model(y.unsqueeze(0), t, self.C, [True, False])
            return vt.squeeze() - ut.squeeze()
            
        div = torch.vmap(self.div_fn(vecfield))(x, self.eps_fn(x))
        
        ut = self.model(x, t, self.C, [True, True],  return_score=False)
        vt, score_v = self.model(x, t, self.C, [True, False], return_score=True)

        correction_term_u = torch.linalg.vecdot(ut - vt, score_v)
        dr = div + correction_term_u
        
        return torch.cat([ut, dr[:, None]], dim=-1)


def push_forward_noise(
    X_train,
    idx,
    unique_conds,
    n_samples,
    model,
    n_steps
):
    # push forward noise
    C = unique_conds[idx].unsqueeze(0).repeat(n_samples, 1)
    node = NeuralODE(
        NODEWrapper(model=model, cond=C, use_conds=[True, True], return_score=False),
        solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
    )
    with torch.no_grad():
        traj = node.trajectory(
            torch.randn((n_samples, X_train.shape[1])).float().cuda(),
            t_span=torch.linspace(0, 1, n_steps).to("cuda")
        )
    return traj


def pull_back_data_and_compute_llr(
    C_train,
    X_train,
    idx,
    unique_conds,
    model,
):
    C_mask_train = torch.all(C_train == unique_conds[idx], axis=1)
    x =  X_train[C_mask_train]
    c = unique_conds[idx].unsqueeze(0)
    node_llr = NeuralODE(
        NODEWrapper_with_ratio_tvf_rl(model, c),
        solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
    )
    with torch.no_grad():
        traj = node_llr.trajectory(
            torch.cat([x, torch.zeros(x.shape[0], 1).to("cuda")], dim=-1),
            t_span=torch.linspace(1, 0, 100).to("cuda")
        )
    res = traj[-1].detach().cpu().numpy()
    return res[:,:-1], -res[:,-1]


def get_ct_and_batch_names(
    adata_full,
    cond_id,
    unique_conds,
    ct_key="cell_type",
    batch_key="batch"
):
    ohe_ct = OneHotEncoder().fit(adata_full.obs[[ct_key]])
    ohe_batch = OneHotEncoder().fit(adata_full.obs[[batch_key]])
    cond_dims = [adata_full.obs[ct_key].nunique(), adata_full.obs[batch_key].nunique()]
    ct_vec = ohe_ct.inverse_transform(unique_conds[cond_id][:cond_dims[0]][None].detach().cpu().numpy())
    batch_vec = ohe_batch.inverse_transform(unique_conds[cond_id][cond_dims[0]:][None].detach().cpu().numpy())
    return ct_vec[0][0], batch_vec[0][0]


def plot_llr_densities(
    cllr,
    ullr,
    ct_name,
    batch_name,
):
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(f"Cell type: {ct_name}/ Batch: {batch_name}", size=12)
    sns.histplot(cllr, alpha=0.4, color="blue", label="corrected", stat="density", ax=ax)
    sns.kdeplot(cllr, color="blue", ax=ax)
    sns.histplot(ullr, alpha=0.4, color="red", label="uncorrected", stat="density", ax=ax)
    sns.kdeplot(ullr, color="red", ax=ax)
    ax.grid(True)
    fig.legend(
        loc="lower left",
        bbox_to_anchor=(0.95, 0.05),
        fontsize=7,
        markerscale=0.8 
    )
    fig.tight_layout()
    return fig


@hydra.main(
    config_path="/home/icb/lorenzo.consoli/repos/scFM_density_estimation/notebooks/cElegans/configs/run_plots",
    config_name="run_plots"
)
def main(config):
    import sys
    sys.path.insert(0, "/home/icb/lorenzo.consoli/repos/scFM_density_estimation/notebooks/cElegans/scripts")
    from train_cfm import get_cfm_model

    # Read data
    logger.info(f"Reading data from {config.paths.adata_path}...")
    adata_full = sc.read_h5ad(config.paths.adata_path)
    logger.info(f"{adata_full=}")

    # Optionally select HVGs
    if config.adata_setup.hvgs_only:
        logger.info(f"Subseting gene panel to HVGs only...")
        n_genes_before = adata_full.X.shape[1]
        adata_full = adata_full[:, adata_full.var["highly_variable"]]
        n_genes_after = adata_full.X.shape[1]
        logger.info(f"{n_genes_after} HVGs selected out of the {n_genes_before} original genes.")

    # Read config
    base_conf_path = os.path.join(config.paths.cfm_model_dir, "config.yaml")
    base_config_yaml = OmegaConf.load(base_conf_path)
    scvi_model_dir = base_config_yaml.paths.scvi_model_dir

    # Read scvi model, compute PCs on uncorrected data using latent dimensionality
    # and extract latent representation
    logger.info(f"Loading SCVI model from {scvi_model_dir} and extracting latent representations...")
    scvi_model = scvi.model.SCVI.load(scvi_model_dir, adata=adata_full.copy())
    adata_full.obsm[config.adata_setup.scvi_latent_key] = scvi_model.get_latent_representation()
    logger.info(f"Latent representation of shape {adata_full.obsm[config.adata_setup.scvi_latent_key].shape}")
    n_dims = adata_full.obsm[config.adata_setup.scvi_latent_key].shape[1]
    logger.info("Computing PCs...")
    sc.pp.pca(adata_full, n_comps=n_dims)
    logger.info("PCs compted!")

    # retrieve condition dimensions
    cond_dims = [
        adata_full.obs[config.adata_setup.labels_key].nunique(),
        adata_full.obs[config.adata_setup.batch_key].nunique()
    ]
    logger.info(f"Condition dimensions set to {cond_dims}")

    # Load CFM Models
    logger.info(f"Loading CFM Models from {config.paths.cfm_model_dir}")
    corr_path = os.path.join(config.paths.cfm_model_dir, "corrected_model.pt")
    uncorr_path = os.path.join(config.paths.cfm_model_dir, "uncorrected_model.pt")
    corr_state_dict = torch.load(corr_path)
    uncorr_state_dict = torch.load(uncorr_path)
    model_uncorr = get_cfm_model(
        n_dims,
        cond_dims,
        config,
    )
    model_uncorr.load_state_dict(uncorr_state_dict)
    model_corr = get_cfm_model(
        n_dims,
        cond_dims,
        config,
    )
    model_corr.load_state_dict(corr_state_dict)

    # 8. Retrieve condition data
    C_train = np.concatenate(
        (
            OneHotEncoder().fit_transform(adata_full.obs[[config.adata_setup.labels_key]]).toarray(),
            OneHotEncoder().fit_transform(adata_full.obs[[config.adata_setup.batch_key]]).toarray(),
        ), axis=1
    )
    C_train = torch.from_numpy(C_train).float().cuda()
    unique_conds = torch.unique(C_train, dim=0)
    logger.info(f"Condition data of shape {C_train.shape}, {unique_conds.shape[0]} unique conditions found.")

    # 9. Train model on uncorrected data
    logger.info("Fitting CFM model on uncorrected representations...")
    X_train_uncorr = torch.from_numpy(adata_full.obsm["X_pca"]).float().cuda()
    X_train_corr = torch.from_numpy(adata_full.obsm[config.adata_setup.scvi_latent_key]).float().cuda()

    # define plot directory
    plot_dir = os.path.join(config.paths.cfm_model_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    results_dir = os.path.join(config.paths.cfm_model_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # iterating over unique conditions
    for idx in tqdm(range(unique_conds.shape[0])):
        # retrieve cell type and batch names
        cell_type, batch = get_ct_and_batch_names(
            adata_full,
            idx,
            unique_conds,
            ct_key="cell_type",
            batch_key="batch"
        )
        cell_type = cell_type.replace("/", "-")
        batch = batch.replace("/", "-")
        id_name = f"CT:{cell_type}-B:{batch}"

        # corrected model
        noise_corr, llr_corr = pull_back_data_and_compute_llr(
            C_train,
            X_train_corr,
            idx,
            unique_conds,
            model_corr,
        )

        # uncorrected model
        noise_uncorr, llr_uncorr = pull_back_data_and_compute_llr(
            C_train,
            X_train_uncorr,
            idx,
            unique_conds,
            model_uncorr,
        )

        # plot densities
        fig = plot_llr_densities(
            llr_corr,
            llr_uncorr,
            cell_type,
            batch,
        )
        fig_path = os.path.join(plot_dir, f"{id_name}.png")
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")

        # save results
        array_dict = {
            "noise_corrected": noise_corr,
            "llr_corrected": llr_corr,
            "llr_uncorrected": llr_uncorr,
            "noise_uncorrected": noise_uncorr,
        }
        array_path = os.path.join(results_dir, f"{id_name}.npz")
        np.savez(array_path, **array_dict)
    return 0


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.info(f"An error occurred: {e}")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

import lightning as L
import numpy as np
import cloudpickle
import torch
import torch.nn.functional as F
import time
import os

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchdyn.core import NeuralODE
from tqdm import tqdm

from scRatio.datamodules import ArrayDataset
from scRatio.models.flow_matching import ConditionalFlowMatchingWithScore


loc = 2.0
n = 30

N = 100_000
cond_dim = 2
batch_size = 512

control = np.array([1, 0])
condition = np.array([0, 1])
locs = [[0 for _ in range(n)]] + [[loc for _ in range(n)]]

sigma = 0
sigma_min = 0.0

latent_dim = 16
time_feature_dim = 63
cond_latent_dim = 8

lambda_t = lambda t: torch.sqrt((1 - (1 - sigma_min) * t) ** 2 + sigma * t * (1 - t))
lambda_sp_t = lambda t: (sigma * (1 - 2 * t) - 2 * (1 - sigma_min) * (1 - (1 - sigma_min) * t)) / 2


train_models = True

num_steps_list = [
    1_000,
    5_000,
    10_000,
    50_000,
    100_000,
]


def prepare_dataset(n, N, cond_dim, locs):
    C = np.random.randint(low=0, high=cond_dim, size=(N))
    X = np.concatenate([np.random.normal(loc=locs[c], scale=1, size=(1, n)) for c in C])

    X_train, X_test, C_train, C_test = train_test_split(X, C, test_size=10_000)

    X_train = torch.tensor(X_train).to("cuda").float()
    C_train = F.one_hot(torch.tensor(C_train).long(), num_classes=cond_dim).to("cuda").float()

    X_test = torch.tensor(X_test).to("cuda").float()
    C_test = F.one_hot(torch.tensor(C_test).long(), num_classes=cond_dim).to("cuda").float()
    return X_train, X_test, C_train, C_test


def build_train_loader(x_train, c_train, batch_size):
    dataset = ArrayDataset(x_train, c_train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def div_fn_hutch_trace(u):
    def div_fn(x, eps):
        _, vjpfunc = torch.func.vjp(u, x)
        return (vjpfunc(eps)[0] * eps).sum()

    return div_fn

def evaluate_model(model, data_samples, cond, cond_dim, condition, control, locs):
    device = data_samples.device
    model = model.to(device)

    # ground truth
    start_true = time.time()

    log_condition_true = -0.5 * ((data_samples.cpu().numpy() - np.array(locs[np.argmax(condition)])) ** 2).sum(axis=1) - 0.5 * data_samples.shape[1] * np.log(2 * np.pi)
    log_control_true = -0.5 * ((data_samples.cpu().numpy() - np.array(locs[np.argmax(control)])) ** 2).sum(axis=1) - 0.5 * data_samples.shape[1] * np.log(2 * np.pi)
    log_ratio_true = log_condition_true - log_control_true
    
    time_true = time.time() - start_true
    
    condition = torch.from_numpy(condition).float().to(device).expand(data_samples.shape[0], cond_dim)
    control = torch.from_numpy(control).float().to(device).expand(data_samples.shape[0], cond_dim)

    # naive evaluation
    start_hat = time.time()

    log_condition_hat = model.estimate_log_density(data_samples, condition, n_steps=100, solver="euler")
    log_control_hat = model.estimate_log_density(data_samples, control, n_steps=100, solver="euler")
    log_ratio_hat_naive = log_condition_hat - log_control_hat
    
    time_hat = time.time() - start_hat

    # correction term - its own field
    start_hat_v2 = time.time()

    log_ratio_hat_v2_scratio = model.estimate_log_density_ratio(data_samples, condition, control, cond, n_steps=100, solver="euler")
    
    time_hat_v2 = time.time() - start_hat_v2
    
    return [log_ratio_true, log_condition_true, log_control_true, log_ratio_hat_naive, log_ratio_hat_v2_scratio, log_condition_hat, log_control_hat], [time_true, time_hat, time_hat_v2]


def fit_model_per_step(n_steps, train_loader):

    model = ConditionalFlowMatchingWithScore(
        input_dim=n,
        cond_dims=[cond_dim],
        hidden_dims=[1024, 1024, 1024],
        encoder_hidden_dims=[256],
        encoder_out_dim=latent_dim,
        lambda_t=lambda_t,
        lambda_sp_t=lambda_sp_t,
        betas=[0],
        lr=1e-4,
        time_feature_dim=time_feature_dim,
        encoder_out_dim_cond=cond_latent_dim,
        init_shared_encoder=False,
        init_cond_encoder=False,
    )
    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_steps=n_steps,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_dataloaders=train_loader)
    return model


if __name__ == "__main__":

    X_train, X_test, C_train, C_test = prepare_dataset(n, N, cond_dim, locs)
    train_loader = build_train_loader(X_train, C_train, batch_size)

    dims_dir = f"out/{n}-dims-{int(loc)}-loc"
    if not os.path.isdir(dims_dir):
        os.mkdir(dims_dir)

    if train_models:
        steps2model = {}

        for n_steps in num_steps_list:
            print(f"Training model for {n_steps} steps")
            model = fit_model_per_step(n_steps, train_loader)
            steps2model[n_steps] = model
            with open(os.path.join(dims_dir, f"{n_steps}-steps_model.pkl"), "wb") as fb:
                cloudpickle.dump(model, fb)
    else:
        steps2model = {}
        for n_steps in num_steps_list:
            try:
                with open(os.path.join(dims_dir, f"{n_steps}-steps_model.pkl"), "rb") as fb:
                    model = cloudpickle.load(fb)
                print(f"loaded {n_steps=}")
                steps2model[n_steps] = model
            except:
                print(f"skipping {n_steps=}")
                continue

    score_altered_preds = {}
    for dim, model in steps2model.items():
        model = model.to("cuda")
        dim_preds = {}
        for dim_other, model_other in steps2model.items():
            model_other = model_other.to("cuda")
            print(f"{dim=}-{dim_other=}")
            modified_model = ConditionalFlowMatchingWithScore(
                input_dim=n,
                cond_dims=[cond_dim],
                hidden_dims=[1024, 1024, 1024],
                encoder_hidden_dims=[256],
                encoder_out_dim=latent_dim,
                lambda_t=lambda_t,
                lambda_sp_t=lambda_sp_t,
                betas=[0],
                lr=1e-4,
                time_feature_dim=time_feature_dim,
                encoder_out_dim_cond=cond_latent_dim,
            )
            modified_model.to("cuda")
            modified_model.vf_mlp = model.vf_mlp
            modified_model.score_mlp = model_other.score_mlp

            # corrected model
            [log_ratio_true, log_condition_true, log_control_true, log_ratio_hat_naive, log_ratio_hat_v2_scratio, log_condition_hat, log_control_hat], [time_true, time_hat_naive, time_hat_scratio] = evaluate_model(model, X_test, C_test, cond_dim, condition, control, locs)
            dim_preds[dim_other] = {
                "log_ratio_true": log_ratio_true,
                "log_condition_true": log_condition_true,
                "log_control_true": log_control_true,
                "log_ratio_hat_naive": log_ratio_hat_naive,
                "log_ratio_hat_v2_scratio": log_ratio_hat_v2_scratio,
                "log_condition_hat": log_condition_hat,
                "log_control_hat": log_control_hat,
                "time_true": time_true,
                "time_hat_naive": time_hat_naive,
                "time_hat_scratio": time_hat_scratio
            }
        score_altered_preds[dim] = dim_preds

    with open(os.path.join(dims_dir, "results.pkl"), "wb") as fb:
        cloudpickle.dump(score_altered_preds, fb)
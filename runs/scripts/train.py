import hydra
from omegaconf import DictConfig
import lightning as L
import torch
from torch import nn

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from scFM_density_estimation.datamodules import scFMDataModule
from scFM_density_estimation.models import ConditionalFlowMatching

@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    datamodule = scFMDataModule(
        adata_path=cfg.datamodule.adata_path,
        conditions=cfg.datamodule.conditions,
        num_conditions=cfg.datamodule.num_conditions,
        condition_dims=cfg.datamodule.condition_dims,
        batch_size=cfg.datamodule.batch_size,
        probability=cfg.datamodule.probability,
        index=cfg.datamodule.index,
        num_pcs=cfg.datamodule.num_pcs,
        num_workers=cfg.datamodule.num_workers,
        val_split=cfg.datamodule.val_split,
        test_split=cfg.datamodule.test_split,
        train_sample_size=int(cfg.datamodule.train_sample_size)
    )
    
    model = ConditionalFlowMatching(
        input_dim=cfg.datamodule.num_pcs,
        cond_dim=sum(cfg.datamodule.condition_dims),
        hidden_dims=cfg.model.hidden_dims,
        cond_hidden_dims=cfg.model.cond_hidden_dims,
        cond_out_dim=cfg.model.cond_out_dim,
        lr=cfg.training.lr,
        use_encoder=cfg.model.use_encoder,
        use_ot_sampler=cfg.model.use_ot_sampler,
        ot_method=cfg.model.ot_method,
        dropout=cfg.model.dropout
    )
    
    wandb_logger = WandbLogger(
        project=cfg.logger.project,
        name=cfg.logger.name
    )
    
    callbacks = [
        ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            dirpath=cfg.training.output_dir,
            filename=cfg.logger.name+"_best-checkpoint"
        ),
    ]
    
    trainer = L.Trainer(
        max_steps=int(cfg.training.max_steps),
        val_check_interval=int(cfg.training.val_check_interval),
        log_every_n_steps=int(cfg.training.log_every_n_steps),
        default_root_dir=cfg.training.output_dir,
        callbacks=callbacks,
        accelerator="gpu",
        logger=wandb_logger
    )
    
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

if __name__ == "__main__":
    main()

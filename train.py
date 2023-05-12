import argparse

import pandas as pd
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import KFold

from dataset import LitDataModule
from model import LitModule


def train(fold, cfg, dataframe):
    pl.seed_everything(cfg.seed)
    datamodule = LitDataModule(fold, cfg.data, dataframe)

    datamodule.setup()

    module = LitModule(cfg.model, cfg.train)

    model_checkpoint = ModelCheckpoint(
        dirpath=cfg.checkpoint,
        monitor="val_loss",
        mode="min",
        verbose="True",
    )

    trainer = pl.Trainer(
        callbacks=[model_checkpoint],
        benchmark=True,
        deterministic=True,
        accelerator=cfg.train.accelerator,
        strategy="ddp" if cfg.train.ddp else "auto",
        devices=cfg.train.devices,
        max_epochs=cfg.train.epoch,
        precision=cfg.train.precision,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        log_every_n_steps=cfg.train.log_every_n_steps,
        logger=WandbLogger("PointNet-Reg", "lightning_logs", project="PointNet-Reg"),
    )

    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Scripts.")
    parser.add_argument("-cfg", "--config", default="config/default.yaml", type=str)
    parser.add_argument("-csv", "--csv-file", default="data/final_data.csv", type=str)
    parser.add_argument(
        "-ckpt_dir", "--checkpoint-dir", default="checkpoints", type=str
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.data.dataset.dataset_csv = args.csv_file
    cfg.checkpoint = args.checkpoint_dir

    dataframe = pd.read_csv(cfg.data.dataset.dataset_csv)

    for fold, (train_idx, valid_idx) in enumerate(
        KFold(n_splits=cfg.kfold, random_state=cfg.seed, shuffle=True).split(dataframe)
    ):
        dataframe.loc[valid_idx, "fold"] = fold

    import gc

    import wandb

    for i in range(cfg.kfold):
        trainer = train(i, cfg, dataframe)
        wandb.finish()
        del trainer
        gc.collect()

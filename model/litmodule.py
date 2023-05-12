from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import *
from torch.optim.lr_scheduler import *

from .pointnet_reg import PointNetReg


class LitModule(pl.LightningModule):
    def __init__(
        self,
        cfg_model,
        cfg_train,
    ) -> None:
        super(LitModule, self).__init__()
        self.cfg_model = cfg_model
        self.cfg_train = cfg_train

        self.model = PointNetReg(
            cfg_model.num_classes,
            cfg_model.num_channels,
            cfg_model.with_dropout,
            cfg_model.dropout,
        )

        self.loss_fn = nn.L1Loss()

        self.save_hyperparameters()

    def forward(self, X: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(X["input"])

    def configure_optimizers(self) -> dict:
        # Setup the optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=self.cfg_train.optimizer.learning_rate,
            weight_decay=self.cfg_train.optimizer.weight_decay,
        )

        scheduler = StepLR(
            optimizer,
            step_size=self.cfg_train.scheduler.step_size,
            gamma=self.cfg_train.scheduler.gamma,
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")

    def _step(self, batch: Dict[str, torch.Tensor], step: str) -> torch.Tensor:
        heatmap = self(batch)
        loss = self.loss_fn(heatmap, batch["gt_heatmap"])
        self.log(f"{step}_loss", loss, sync_dist=True, prog_bar=True)

        return loss

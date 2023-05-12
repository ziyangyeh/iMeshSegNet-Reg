from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import vedo
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader

from .toothlandmark_dataset import ToothLandmarkDataset


def rotate_transform(mesh: vedo.Mesh, pts: np.ndarray) -> tuple:
    rot = R.random()
    mesh.points(rot.apply(mesh.points()))
    pts = rot.apply(pts)
    return mesh, pts


class LitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        fold,
        cfg,
        dataframe: pd.DataFrame,
    ) -> None:
        super(LitDataModule, self).__init__()
        self.fold = fold
        self.batch_size = cfg.dataloader.batch_size
        self.num_workers = cfg.dataloader.num_workers
        self.sample_num = cfg.dataset.sample_num
        self.max_num_classes = cfg.dataset.max_num_classes
        self.dataframe = dataframe
        self.transform = rotate_transform if cfg.dataset.transform else None

        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            train_df = self.dataframe[self.dataframe["fold"] != self.fold].reset_index()
            val_df = self.dataframe[self.dataframe["fold"] == self.fold].reset_index()
            self.train_dataset = ToothLandmarkDataset(
                train_df, self.sample_num, self.max_num_classes, self.transform
            )
            self.val_dataset = ToothLandmarkDataset(
                val_df,
                self.sample_num,
                self.max_num_classes,
            )
        if stage == "test" or stage is None:
            self.test_dataset = ToothLandmarkDataset(
                self.dataframe,
                self.sample_num,
                self.max_num_classes,
            )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, train=True, val=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset, train=False, val=True)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_dataset)

    def _dataloader(
        self, dataset: ToothLandmarkDataset, train: bool = False, val: bool = False
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True if train and val else False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True if train and val else False,
        )

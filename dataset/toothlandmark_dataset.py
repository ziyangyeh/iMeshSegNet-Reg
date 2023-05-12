from typing import Callable, Optional

import numpy as np
import pandas as pd
import vedo
from torch.utils.data import Dataset

from utils import gaussian


class ToothLandmarkDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        decimate_to: int,
        max_num_classes: int,
        transform: Optional[Callable] = None,
    ) -> None:
        super(ToothLandmarkDataset, self).__init__()
        self.dataframe = dataframe
        self.max_num_classes = max_num_classes
        self.decimate_to = decimate_to
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx) -> dict:
        key_pts = pd.read_csv(self.dataframe["key"][idx]).values
        mesh = vedo.Mesh(self.dataframe["tooth"][idx])
        ratio = self.decimate_to / mesh.ncells
        mesh = mesh.decimate(ratio)

        if self.transform:
            mesh, key_pts = self.transform(mesh, key_pts)

        # move mesh to origin
        points = mesh.points()
        mean_cell_centers = mesh.center_of_mass()
        points[:, 0:3] -= mean_cell_centers[0:3]
        key_pts[:, 0:3] -= mean_cell_centers[0:3]

        ids = np.array(mesh.faces())
        cells = points[ids].reshape(mesh.ncells, 9).astype(dtype="float32")

        normals = mesh.normals(cells=True)

        # move mesh to origin
        barycenters = mesh.cell_centers()  # don't need to copy
        barycenters -= mean_cell_centers[0:3]

        # normalized data
        maxs = points.max(axis=0)
        mins = points.min(axis=0)
        means = points.mean(axis=0)
        stds = points.std(axis=0)
        nmeans = normals.mean(axis=0)
        nstds = normals.std(axis=0)

        for i in range(3):
            cells[:, i] = (cells[:, i] - means[i]) / stds[i]  # point 1
            cells[:, i + 3] = (cells[:, i + 3] - means[i]) / stds[i]  # point 2
            cells[:, i + 6] = (cells[:, i + 6] - means[i]) / stds[i]  # point 3
            barycenters[:, i] = (barycenters[:, i] - mins[i]) / (maxs[i] - mins[i])
            normals[:, i] = (normals[:, i] - nmeans[i]) / nstds[i]
            key_pts[:, i] = (key_pts[:, i] - mins[i]) / (maxs[i] - mins[i])

        key_nums = key_pts.shape[0]
        if key_nums < self.max_num_classes:
            key_pts = np.vstack(
                [
                    key_pts,
                    np.asarray(
                        [np.inf, np.inf, np.inf] * (self.max_num_classes - key_nums)
                    ).reshape(-1, 3),
                ]
            )

        _X = np.column_stack((cells, barycenters, normals))
        _gt_heatmap = np.asarray(
            [gaussian(barycenters, i, axis=1) for i in key_pts[:]]
        ).transpose()

        assert _X.shape[0] == _gt_heatmap.shape[0]

        selected_idx = (
            np.random.choice(
                np.arange(_X.shape[0]), size=self.decimate_to, replace=False
            )
            if self.decimate_to == mesh.ncells
            else np.random.choice(
                np.arange(_X.shape[0]), size=self.decimate_to, replace=True
            )
        )
        selected_idx = np.sort(selected_idx, axis=None)

        X = np.zeros([self.decimate_to, _X.shape[1]], dtype="float32")
        gt_heatmap = np.zeros([self.decimate_to, _gt_heatmap.shape[1]], dtype="float32")

        X[:] = _X[selected_idx, :]
        gt_heatmap[:] = _gt_heatmap[selected_idx, :]

        return {
            "input": X.transpose(1, 0),
            "gt_heatmap": gt_heatmap,
        }

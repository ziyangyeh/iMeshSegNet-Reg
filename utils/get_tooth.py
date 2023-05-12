import numpy as np
import vedo


def get_tooth(mesh, label, index):
    clone = mesh.clone()
    cell_idx = np.where(label != index)[0]
    clone.delete_cells(cell_idx)
    return clone


if "__main__" == __name__:
    path = "data/raw_label/lower1.vtp"

    mesh = vedo.Mesh(path)
    mesh.compute_normals()
    label = mesh.celldata["Label"]
    tooth = get_tooth(mesh, label, 1)
    print(tooth.normals())

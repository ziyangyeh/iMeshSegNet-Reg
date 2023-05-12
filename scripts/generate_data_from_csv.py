import argparse
import os
import re
import shutil
import sys

sys.path.append(os.getcwd())


import numpy as np
import pandas as pd
import vedo
from tqdm import tqdm

from utils import get_tooth


def generate_data(dataframe: pd.DataFrame, output_path: str):
    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        mesh = vedo.Mesh(row["raw_label"])
        mesh.compute_normals()
        label = mesh.celldata["Label"]
        for i in range(1, label.max() + 1):
            tooth = get_tooth(mesh, label, i)
            sub_dir = row["raw_label"].split("/")[-1].split(".")[0]
            os.makedirs(os.path.join(output_path, sub_dir), exist_ok=True)
            tooth.write(os.path.join(output_path, sub_dir, f"tooth_{i}.vtp"))


if "__main__" == __name__:
    parser = argparse.ArgumentParser(description="Generates date.")
    parser.add_argument("-csv", type=str, help="CSV file to the input", required=True)
    parser.add_argument("-keys", type=str, help="Path to keypoints", required=True)
    parser.add_argument("-output", type=str, help="Path to output", required=True)
    args = parser.parse_args()

    if os.path.exists(args.output):
        shutil.rmtree(args.output)
        os.makedirs(args.output)
    else:
        os.makedirs(args.output)

    df = pd.read_csv(args.csv)
    generate_data(df, args.output)
    file_list = [
        str(os.path.join(root, file))
        for root, _, files in os.walk(args.output)
        for file in files
    ]
    file_list.sort(
        key=lambda x: (
            int(re.findall(r"\d+", x.split("/")[-2])[0]),
            int(re.findall(r"\d+", x.split("/")[-1].split(".")[0])[0]),
        )
    )
    file_list = np.asarray(file_list)

    key_list = [
        str(os.path.join(root, file))
        for root, _, files in os.walk(args.keys)
        for file in files
        if file.endswith(".csv")
    ]
    key_list.sort(
        key=lambda x: (
            int(re.findall(r"\d+", x.split("/")[-1].split(".")[0])[0]),
            int(re.findall(r"\d+", x.split("/")[-2])[0]),
        )
    )
    key_list = np.asarray(key_list)

    new_df = pd.DataFrame()
    new_df["tooth"] = file_list
    new_df["key"] = key_list

    new_df.to_csv(os.path.join("data", "final_data.csv"), index=False)

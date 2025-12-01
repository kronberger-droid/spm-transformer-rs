import h5py
import numpy as np
from pathlib import Path
import argparse


def inspect_data(path):
    path = Path(path)

    if path.suffix.lower() == ".mat":
        inspect_mat(path)
    elif path.suffix.lower() == ".npz":
        inspect_npz(path)


def inspect_mat(path):
    with h5py.File(path, "r") as f:
        for key in f.keys():
            data = f[key]
            print(f"\n{key}:")
            print(f"  Type: {type(data)}")
            print(f"  Shape: {data.shape}")
            print(f"  Dtype: {data.dtype}")

            # If it's small enough, show some values
            if data.size < 100:
                print(f"  Values: {np.array(data).flatten()}")
            else:
                print(f"  Sample values: {np.array(data).flatten()[:10]}")


def inspect_npz(path):
    f = np.load(path)

    print(f.files)

    for name in f:
        arr = f[name]
        print(f"{name}: shape {arr.shape}, dtype {arr.dtype}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get basic information about MAT and NPZ files"
    )
    parser.add_argument("path", help="Path to input .mat or npz file")

    args = parser.parse_args()

    inspect_data(args.path)

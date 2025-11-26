import h5py
import numpy as np

print("Processed Data")

with np.load("data/processed_data.npz") as data:
    print(data.files)
    for key in data:
        print(f"{key}: {data[key].shape}, {data[key].dtype}")

print("Processed Data")
with np.load("data/line_data.npz") as data:
    print(data.files)
    for key in data:
        print(f"{key}: {data[key].shape}, {data[key].dtype}")

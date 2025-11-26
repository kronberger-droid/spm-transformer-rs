import h5py
import numpy as np

# Load the .mat file
with h5py.File("/home/kronberger/Downloads/LineDataNewNew128.mat", "r") as f:
    # List variables
    print("Keys:", list(f.keys()))

    # Load data (adjust variable name as needed, e.g., 'LineData')
    mat_data = {key: np.array(f[key]) for key in f.keys()}

# Save as .npz
np.savez("LineDataNewNew128.npz", **mat_data)

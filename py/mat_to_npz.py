from enum import unique
import h5py
import numpy as np
import argparse


def convert_mat_to_npz(mat_path, output_path):
    """
    Convert MAT file with rectangular STM images to NPZ format.
    Expected MAT files structure:
    - x_raw: [pixels_per_line, num_columns] - Image data where pixels_per_line = lines_per_image
    - y: [num_classes, num_columns] - One-hot encoded labels

    """
    with h5py.File(mat_path, "r") as f:
        x_raw = np.array(f["x_raw"])  # [128, num_columns]
        y_raw = np.array(f["y"])  # [num_classes, num_columns]

        pixels_per_line = x_raw.shape[0]

        # Calculate the number of images
        num_images = x_raw.shape[1] // pixels_per_line
        print(f"Number of images: {num_images}")

        # Reshape images: [128, num_images * 128] -> [num_images, 128, 128]
        images = x_raw.reshape(pixels_per_line, num_images, pixels_per_line).transpose(
            1, 0, 2
        )
        print(f"Images shape: {images.shape}")

        # Extract labels: take first column of each images
        labels_onehot = y_raw[:, ::128]  # [num_classes, num_images]
        labels = np.argmax(labels_onehot, axis=0)  # [num_images,]
        print(f"Labels shape: {labels.shape}")

        # Verify labels are consistent within each image block
        inconsistent_count = 0
        for i in range(num_images):
            start_col = i * 128
            end_col = start_col + 128
            block_labels = y_raw[:, start_col:end_col]
            # check if all columns in this block have the same label
            if not np.all(block_labels == block_labels[:, [0]], axis=1).all():
                inconsistent_count += 1

        if inconsistent_count > 0:
            print(
                f"WARNING: {inconsistent_count} images have inconsistent labels across columns!"
            )
        else:
            print("All labels are conisistent within image block")

        # Show label distribution
        print("\nLabel distribution:")
        unique, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  Lablel {label}: {count} images ({count / num_images * 100:.1f}%)")

        images = images.astype(np.float32)
        labels = labels.astype(np.int32)
        np.savez(output_path, images=images, labels=labels)
        print(f"\nSaved to {output_path}")
        print(f"  - images: {images.shape} (dtype: {images.dtype})")
        print(f"  - labels: {labels.shape} (dtype: {labels.dtype})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MAT file to NPZ")

    parser.add_argument("mat_file", help="Path to input .mat file")
    parser.add_argument("output_file", help="Path to output .npz file")

    args = parser.parse_args()

    convert_mat_to_npz(args.mat_file, args.output_file)

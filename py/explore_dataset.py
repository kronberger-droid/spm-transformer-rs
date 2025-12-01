import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="Explore NPZ dataset with image viewer")
parser.add_argument("npz_file", help="Path to .npz file")
args = parser.parse_args()

# Load data
data = np.load(args.npz_file)
print("Arrays in file:")
for name in data.files:
    print(f"  {name}: shape={data[name].shape}, dtype={data[name].dtype}")

# Get the actual arrays (not just the names)
images = data[data.files[0]]
labels = data[data.files[1]]

print(f"\nLoaded {len(images)} images")
print(f"First label: {labels[0]}")

# %%
# Label mapping - update these with your actual label names
LABEL_NAMES = {
    0: "0: individual atoms",
    1: "1: horizontal_line",
    2: "2: vertical_line",
    3: "3: diagonal_line",
    4: "4: cross",
    5: "5: defect",
}


# %%
class ImageViewer:
    def __init__(self, images, labels, label_names=None):
        self.images = images
        self.labels = labels
        self.label_names = label_names or {}
        self.idx = 0
        self.random_mode = False

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.update()

    def on_key(self, event):
        if event.key in ["right", "l"]:
            if self.random_mode:
                self.idx = np.random.randint(0, len(self.images))
            else:
                self.idx = (self.idx + 1) % len(self.images)
            self.update()
        elif event.key in ["left", "h"]:
            if self.random_mode:
                self.idx = np.random.randint(0, len(self.images))
            else:
                self.idx = (self.idx - 1) % len(self.images)
            self.update()
        elif event.key == "r":
            self.random_mode = not self.random_mode
            mode = "RANDOM" if self.random_mode else "SEQUENTIAL"
            print(f"Mode: {mode}")
            self.update()

    def update(self):
        self.ax.clear()
        self.ax.imshow(self.images[self.idx], cmap="gray")

        # Get label name or use numeric label
        label = self.labels[self.idx]
        label_text = self.label_names.get(label, f"Label {label}")

        mode_indicator = " [RANDOM]" if self.random_mode else ""
        self.ax.set_title(
            f"Image {self.idx}/{len(self.images) - 1} - {label_text}{mode_indicator}"
        )
        self.ax.axis("off")
        plt.draw()


viewer = ImageViewer(images, labels, LABEL_NAMES)
plt.show()

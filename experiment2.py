"""
Experiment 2:
  - Same as experiment 1, but first dimension is not intensity, but anomaly volume
"""
import os
from typing import Tuple

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.draw import disk
from sklearn.metrics import average_precision_score
from tqdm import tqdm
from typing import Tuple, List

from utils import MOODROOT, load_nii


def plot_landscape(X, Y, Z, ax_labels: Tuple[str, str, str]=None,
                   path: str=None):
    _, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X_, Y_ = np.meshgrid(X, Y)

    # Plot the surface.
    surf = ax.plot_surface(X_, Y_, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=True)

    if ax_labels is not None:
        ax.set_xlabel(ax_labels[0])
        ax.set_ylabel(ax_labels[1])
        ax.set_zlabel(ax_labels[2])

    # Limit z-axis
    ax.set_zlim(0., 1.)

    # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)


def show(imgs: List[np.ndarray], seg: List[np.ndarray]=None,
         path: str=None) -> None:

    if not isinstance(imgs, list):
        imgs = [imgs]
    n = len(imgs)
    fig = plt.figure(figsize=(4 * n, 4))

    for i in range(n):
        fig.add_subplot(1, n, i + 1)
        plt.imshow(imgs[i], cmap="gray", vmin=0., vmax=1.)

        if seg is not None:
            plt.imshow(seg[i], cmap="jet", alpha=0.3)

    if path is None:
        plt.show()
    else:
        plt.axis("off")
        plt.savefig(path, bbox_inches='tight', pad_inches=0)


def disk_anomaly(img: np.ndarray, position: Tuple[int, int], radius: int,
                   intensity: float) -> np.ndarray:
    """Draw a disk on a grayscale image.

    Args:
        img (np.ndarray): Grayscale image
        position (Tuple[int, int]): Position of disk
        radius (int): Radius of disk
        intensity (float): Intensity of pixels inside the disk
    Returns:
        disk_img (np.ndarray): img with ball drawn on it
    """
    assert img.ndim == 2, f"Invalid shape {img.shape}. Use a grayscale image"
    # Create disk
    rr, cc = disk(position, radius)
    # Draw disk on image
    disk_img = img.copy()
    disk_img[rr, cc] = intensity
    # Create label
    label = np.zeros(img.shape, dtype=np.uint8)
    label[rr, cc] = 1

    return disk_img, label


def blur_img(img: np.ndarray, sigma: float) -> np.ndarray:
    return gaussian_filter(img, sigma=sigma)


def average_precision(target: np.ndarray, pred: np.ndarray) -> float:
    return average_precision_score(target.reshape(-1), pred.reshape(-1))


def run_experiment_sample(img: np.ndarray, blur: float, intensity: float,
            position: Tuple[int, int], radius: int):
    """Run the experiment for a specific configuration"""
    img_blur = blur_img(img, blur)
    img_anomal, label = disk_anomaly(img, position, radius, intensity)
    pred = np.abs(img_blur - img_anomal)
    ap = average_precision(label, pred)
    return (img_anomal, img_blur, pred, label, ap)


def summary(img: np.ndarray, blur: float, intensity: float,
            position: Tuple[int, int], radius: int) -> None:
    img_anomal, img_blur, pred, label, ap = run_experiment_sample(
        img, blur, intensity, position, radius
    )
    print(f"Average precision: {ap:.4f}")
    show([img, img_anomal, img_blur, pred, label])



if __name__ == "__main__":
    # Load image
    img_path = "/home/felix/datasets/MOOD/brain/test_raw/00529.nii.gz"
    # img_path = os.path.join(MOODROOT, "brain/test_raw/00480.nii.gz")
    volume, _ = load_nii(img_path, primary_axis=2)
    img = volume[volume.shape[0] // 2]

    # Select ball position and radius
    position = (128, 180)
    intensity = 0.6

    results = []  # Gather ap results here
    radii = np.linspace(1, 51, num=100).astype(np.int)  # Second dimension
    blurrings = np.linspace(0., 5., num=100)  # Second dimension

    # Perform experiment
    for radius in tqdm(radii):
        result_row = []
        for blur in blurrings:
            img_blur = blur_img(img, blur)
            img_anomal, label = disk_anomaly(img, position, radius, intensity)
            pred = np.abs(img_blur - img_anomal)
            ap = average_precision(label, pred)
            result_row.append(ap)
        results.append(result_row)

    results = np.array(results)
    plot_landscape(blurrings, radii, results, ("blur", "radius", "ap"))
    import IPython ; IPython.embed() ; exit(1)

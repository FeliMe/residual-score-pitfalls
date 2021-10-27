"""
Experiment 1:
  - Get an image (mid-slice of a brain)
  - Create an anomaly by putting a circle of specific intensity inside the image
  - Subtract the original image from the anomaly (assuming an Autoencoder that has learned the distribution perfectly)
  - Do this for all intensities from 0 to 1 and report the average precision
  - In a second dimension, add gaussian blur to the image (simulates imperfect reconstruction of the Autoencoder)
"""
from typing import Tuple
from typing import Tuple

import numpy as np
from tqdm import tqdm

from utils import (
    average_precision,
    blur_img,
    disk_anomaly,
    load_nii,
    plot_landscape,
    show,
)


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
    position = (128, 200)
    radius = 20

    results = []  # Gather ap results here
    intensities = np.linspace(0., 1., num=100)  # First dimension
    blurrings = np.linspace(0., 5., num=100)  # Second dimension

    # Perform experiment
    for intensity in tqdm(intensities):
        result_row = []
        for blur in blurrings:
            img_blur = blur_img(img, blur)
            img_anomal, label = disk_anomaly(img, position, radius, intensity)
            pred = np.abs(img_blur - img_anomal)
            ap = average_precision(label, pred)
            result_row.append(ap)
        results.append(result_row)

    results = np.array(results)
    plot_landscape(blurrings, intensities, results, ("blur", "intensity", "ap"))
    import IPython ; IPython.embed() ; exit(1)

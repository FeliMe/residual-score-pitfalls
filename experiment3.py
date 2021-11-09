"""
Experiment 3:
  - Get an image (mid-slice of a brain)
  - Create anomaly source deformation (or just mix the pixels in a patch)
  - Add increasing blur to the 'reconstructed' image
  - Subtract the reconstructed image from the anomaly (simulates imperfect reconstruction of the Autoencoder)
"""
from typing import Tuple

import numpy as np

from utils import (
    average_precision,
    blur_img,
    load_nii,
    plot_curve,
    show,
    sink_deformation_anomaly,
    source_deformation_anomaly,
    pixel_shuffle_anomaly,
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
    volume, _ = load_nii(img_path, primary_axis=2)
    img = volume[volume.shape[0] // 2]

    # Select ball position and radius
    position = (128, 200)
    radius = 20

    results = []  # Gather ap results here
    blurrings = np.linspace(0., 5., num=100)  # First dimension

    # Perform experiment
    for blur in blurrings:
        img_blur = blur_img(img, blur)
        img_anomal, label = pixel_shuffle_anomaly(img, position, radius)
        pred = np.abs(img_blur - img_anomal)
        ap = average_precision(label, pred)
        results.append(ap)

    results = np.array(results)
    # np.save("./results/experiment3_numbers.npy", results)
    plot_curve(blurrings, results, ("blur", "ap"))
    import IPython ; IPython.embed() ; exit(1)

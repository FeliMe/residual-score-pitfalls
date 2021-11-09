"""
Experiment 2:
  - Same as experiment 1, but first dimension is not intensity, but anomaly volume
"""
from typing import Tuple

import numpy as np
from tqdm import tqdm
from typing import Tuple

from utils import (
    average_precision,
    blur_img,
    disk_anomaly,
    load_nii,
    plot_landscape,
    plot_heatmap,
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
    np.save("./results/experiment2_numbers.npy", results)
    plot_landscape(blurrings, radii, results, ("blur", "radius", "ap"))
    plot_heatmap(blurrings, radii, results, ("blur", "radius"))
    import IPython ; IPython.embed() ; exit(1)

"""
Experiment 1:
  - Get an image (mid-slice of a brain)
  - Create an anomaly by putting a circle of specific intensity inside the image
  - Subtract the original image from the anomaly (assuming an Autoencoder that has learned the distribution perfectly)
  - Do this for all intensities from 0 to 1 and report the average precision
  - In a second dimension, add gaussian blur to the image (simulates imperfect reconstruction of the Autoencoder)
"""
from typing import Tuple
import random

import numpy as np
from tqdm import tqdm

from artificial_anomalies import disk_anomaly, sample_position
from utils import (
    average_precision,
    blur_img,
    load_nii,
    load_mood_test_data,
    plot_landscape,
    plot_heatmap,
    show,
)


if __name__ == "__main__":
    # Place random seeds
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    # Load data
    # img_path = "/home/felix/datasets/MOOD/brain/test_raw/00529.nii.gz"
    # volume, _ = load_nii(img_path, primary_axis=2)
    # img = volume[volume.shape[0] // 2]
    imgs = load_mood_test_data()

    # Select ball radius
    # position = (128, 200)
    radius = 20

    results = []  # Gather ap results here
    intensities = np.linspace(0., 1., num=100)  # First dimension
    blurrings = np.linspace(0., 5., num=100)  # Second dimension

    # Perform experiment
    for intensity in tqdm(intensities):
        result_row = []
        for blur in blurrings:
            aps = []
            for img in imgs:
                # Blur the normal image (simulates imperfect reconstruction)
                img_blur = blur_img(img, blur)
                # Create an anomaly at a random position
                random.seed(seed)
                np.random.seed(seed)
                position = sample_position(img)
                img_anomal, label = disk_anomaly(img, position, radius, intensity)
                # Compute the reconstruction error
                pred = np.abs(img_blur - img_anomal)
                # Compute the average precision
                ap = average_precision(label, pred)
                aps.append(ap)
            result_row.append(np.mean(aps))
        results.append(result_row)

    results = np.array(results)
    np.save("./results/experiment1_full_numbers.npy", results)
    # plot_landscape(blurrings, intensities, results, ("blur", "intensity", "ap"),
    #                path="./results/experiment1_full_landscape.png")
    # plot_heatmap(blurrings, intensities, results, ("blur", "intensity"),
    #              path="./results/experiment1_full_heatmap.png")
    # plot_landscape(blurrings, intensities, results, ("blur", "intensity", "ap"))
    # plot_heatmap(blurrings, intensities, results, ("blur", "intensity"))
    import IPython ; IPython.embed() ; exit(1)

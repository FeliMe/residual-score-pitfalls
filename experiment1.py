"""
Experiment 1:
  - Get an image (mid-slice of a brain)
  - Create an anomaly by putting a circle of specific intensity inside the image
  - Subtract the original image from the anomaly (assuming an Autoencoder that has learned the distribution perfectly)
  - Do this for all intensities from 0 to 1 and report the average precision
  - In a second dimension, add gaussian blur to the image (simulates imperfect reconstruction of the Autoencoder)
"""
import argparse
import random

import numpy as np
from tqdm import tqdm

from artificial_anomalies import disk_anomaly, sample_position
from utils import (
    average_precision,
    blur_img,
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', type=str, default=None)
    args = parser.parse_args()

    intensities = np.linspace(0., 1., num=100)  # First dimension
    blurrings = np.linspace(0., 5., num=100)  # Second dimension

    if args.results_path is None:
        # Load data
        print("Loading data...")
        imgs = load_mood_test_data()

        # Select ball size
        radius = 20

        ap_results = []  # Gather ap results here
        rec_results = []  # Gather reconstruction error results here

        # Perform experiment
        for intensity in tqdm(intensities):
            ap_result_row = []
            rec_result_row = []
            for blur in blurrings:
                aps = []
                rec_errs = []
                # Reset the random seed so for every intensity and blurring we get the same positions
                random.seed(seed)
                np.random.seed(seed)
                for img in imgs:
                    # Blur the normal image (simulates imperfect reconstruction)
                    img_blur = blur_img(img, blur)
                    # Create an anomaly at a random position
                    position = sample_position(img)
                    img_anomal, label = disk_anomaly(img, position, radius, intensity)
                    # Compute the reconstruction error
                    pred = np.abs(img_blur - img_anomal)
                    # Compute the average precision
                    ap = average_precision(label, pred)
                    aps.append(ap)
                    rec_errs.append(pred.mean())
                ap_result_row.append(np.mean(aps))
                rec_result_row.append(np.mean(rec_errs))

            ap_results.append(ap_result_row)
            rec_results.append(rec_result_row)
        ap_results = np.array(ap_results)
        rec_results = np.array(rec_results)
        np.save("./results/experiment1/experiment1_full_aps.npy", ap_results)
        np.save("./results/experiment1/experiment1_full_rec_errs.npy", rec_results)
    else:
        ap_results = np.load(args.results_path)

    plot_landscape(blurrings, intensities, ap_results, ("σ", "intensity", "ap"),
                   path="./results/experiment1/experiment1_full_landscape.png")
    plot_heatmap(blurrings, intensities, ap_results, ("σ", "intensity"),
                 path="./results/experiment1/experiment1_full_heatmap.png")
    plot_landscape(blurrings, intensities, ap_results, ("σ", "intensity", "ap"))
    plot_heatmap(blurrings, intensities, ap_results, ("σ", "intensity"))

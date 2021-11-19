"""
Experiment 3:
  - Get an image (mid-slice of a brain)
  - Create anomaly source deformation (or just mix the pixels in a patch)
  - Add increasing blur to the 'reconstructed' image
  - Subtract the reconstructed image from the anomaly (simulates imperfect reconstruction of the Autoencoder)
"""
import argparse
import random
from tqdm import tqdm

import numpy as np

from artificial_anomalies import (
    sample_position,
    pixel_shuffle_anomaly,
    sink_deformation_anomaly,
    source_deformation_anomaly
)
from utils import (
    average_precision,
    blur_img,
    load_nii,
    load_mood_test_data,
    plot_curve,
    show,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--anomaly', type=str,
                        choices=['pixel_shuffle', 'sink_deformation', 'source_deformation'],
                        default='source_deformation')
    args = parser.parse_args()
    anomaly = args.anomaly

    # Place random seeds
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    # Load image
    # img_path = "/home/felix/datasets/MOOD/brain/test_raw/00529.nii.gz"
    # volume, _ = load_nii(img_path, primary_axis=2)
    # img = volume[volume.shape[0] // 2]
    imgs = load_mood_test_data()

    # Select ball position and radius
    # position = (128, 200)
    radius = 20

    ap_results = []  # Gather ap results here
    rec_results = []  # Gather reconstruction error results here
    blurrings = np.linspace(0., 5., num=100)  # First dimension

    # Perform experiment
    for blur in tqdm(blurrings):
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
            if anomaly == 'source_deformation':
                img_anomal, label = source_deformation_anomaly(img, position, radius)
            elif anomaly == 'sink_deformation':
                img_anomal, label = sink_deformation_anomaly(img, position, radius)
            elif anomaly == 'pixel_shuffle':
                img_anomal, label = pixel_shuffle_anomaly(img, position, radius)
            else:
                raise ValueError(f'Unknown anomaly type {anomaly}')
            # Compute the reconstruction error
            pred = np.abs(img_blur - img_anomal)
            # Compute the average precision
            ap = average_precision(label, pred)
            aps.append(ap)
            rec_errs.append(pred.mean())
        ap_results.append(np.mean(aps))
        rec_results.append(np.mean(rec_errs))

    ap_results = np.array(ap_results)
    rec_results = np.array(rec_results)
    np.save(f"./results/experiment3_full_{anomaly}_aps.npy", ap_results)
    np.save(f"./results/experiment3_full_{anomaly}_rec_errs.npy", rec_results)
    # plot_curve(blurrings, results, ("blur", "ap"),
    #            f"./results/experiment3_full_{anomaly}.png")
    # plot_curve(blurrings, results, ("blur", "ap"))
    import IPython; IPython.embed(); exit(1)

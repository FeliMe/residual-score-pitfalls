"""
Experiment 2:
  - Same as experiment 1, but first dimension is not intensity, but anomaly volume
"""
import numpy as np
import random
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

    # Load image
    # img_path = "/home/felix/datasets/MOOD/brain/test_raw/00529.nii.gz"
    # volume, _ = load_nii(img_path, primary_axis=2)
    # img = volume[volume.shape[0] // 2]
    imgs = load_mood_test_data()

    # Select ball position and radius
    # position = (128, 180)
    intensity = 0.6

    ap_results = []  # Gather ap results here
    rec_results = []  # Gather reconstruction error results here
    radii = np.linspace(1, 51, num=100).astype(np.int)  # Second dimension
    blurrings = np.linspace(0., 5., num=100)  # Second dimension

    # Perform experiment
    for radius in tqdm(radii):
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
    np.save("./results/experiment2/experiment2_full_aps_intensity06.npy", ap_results)
    np.save("./results/experiment2/experiment2_full_rec_errs_intensity06.npy", rec_results)
    # plot_landscape(blurrings, radii, ap_results, ("blur", "radius", "ap"),
    #                path="./results/experiment2/experiment2_full_landscape_intensity06.png")
    # plot_heatmap(blurrings, radii, ap_results, ("blur", "radius"),
    #              path="./results/experiment2/experiment2_full_heatmap_intensity06.png")
    # plot_landscape(blurrings, radii, ap_results, ("blur", "radius", "ap"))
    # plot_heatmap(blurrings, radii, ap_results, ("blur", "radius"))
    import IPython; IPython.embed(); exit(1)

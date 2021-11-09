"""
Experiment 4:
- Repeat experiment 1, but with an Autoencoder trained on the MOOD training set
- Subexperiment 1: Use reconstruction of normal image, vary intensity
  (Maybe use multiple checkpoints with different validation losses)
- Subexperiment 2: Use reconstruction of anomal image, again vary intensity
"""
import argparse

import numpy as np
import torch

from models import load_autoencoder
from utils import (
    average_precision,
    disk_anomaly,
    load_nii,
    plot_curve,
    show,
)


def get_model_reconstruction(model, inp):
    with torch.no_grad():
        x = torch.tensor(inp, dtype=torch.float32)[None, None]
        rec = model(x)[0, 0].numpy()
    return rec


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_ckpt', type=str, default='1ex8fxcl/best.pt')
    parser.add_argument('--experiment', type=int, default=2, choices=[1, 2])
    args = parser.parse_args()

    # Load image
    img_path = "/home/felix/datasets/MOOD/brain/test_raw/00529.nii.gz"
    volume, _ = load_nii(img_path, primary_axis=2)
    img = volume[volume.shape[0] // 2]

    # Select ball position and radius
    position = (128, 200)
    radius = 20

    intensities = np.linspace(0., 1., num=100)  # First dimension

    # Load model
    model = load_autoencoder(args.model_ckpt)

    # Perform experiment
    results = []  # Gather ap results here
    for intensity in intensities:
        img_anomal, label = disk_anomaly(img, position, radius, intensity)
        if args.experiment == 1:
            # Experiment 4.1, use reconstruction of normal image
            rec = get_model_reconstruction(model, img)
        elif args.experiment == 2:
            # Experiment 4.2, use reconstruction of anomal image
            rec = get_model_reconstruction(model, img_anomal)
        else:
            raise ValueError()
        pred = np.abs(rec - img_anomal)
        ap = average_precision(label, pred)
        results.append(ap)

    results = np.array(results)
    plot_curve(intensities, results, ("intensity", "ap"))
    import IPython ; IPython.embed() ; exit(1)


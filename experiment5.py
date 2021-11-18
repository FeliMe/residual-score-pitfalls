"""
Experiment 5:
- Repeat experiment 2 (vary anomaly size), but with an Autoencoder trained on the MOOD training set
- Subexperiment 1: Use reconstruction of normal image, vary intensity
  (Maybe use multiple checkpoints with different validation losses)
- Subexperiment 2: Use reconstruction of anomal image, again vary intensity
"""
import argparse
import random
from tqdm import tqdm

import numpy as np
import torch

from artificial_anomalies import disk_anomaly, sample_position
from models import load_autoencoder
from utils import (
    average_precision,
    load_mood_test_data,
    load_nii,
    plot_curve,
    show,
)


def get_model_reconstruction(model, inp, device):
    with torch.no_grad():
        x = torch.tensor(inp, dtype=torch.float32)[None, None].to(device)
        rec = model(x)[0, 0].cpu().numpy()
    return rec


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_ckpt', type=str, default='1ex8fxcl/best.pt')
    parser.add_argument('--experiment', type=int, default=1, choices=[1, 2])
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    # Place random seeds
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    # Select device
    device = args.device if torch.cuda.is_available() else 'cpu'

    # Load image
    # img_path = "/home/felix/datasets/MOOD/brain/test_raw/00529.nii.gz"
    # volume, _ = load_nii(img_path, primary_axis=2)
    # img = volume[volume.shape[0] // 2]
    imgs = load_mood_test_data()

    # Select ball position and intensity
    # position = (128, 200)
    intensity = 0.6

    radii = np.linspace(1, 51, num=100).astype(np.int)

    # Load model
    model = load_autoencoder(args.model_ckpt).to(device)

    # Perform experiment
    results = []  # Gather ap results here
    for radius in tqdm(radii):
        aps = []
        for img in imgs:
            random.seed(seed)
            np.random.seed(seed)
            position = sample_position(img)
            img_anomal, label = disk_anomaly(img, position, radius, intensity)
            if args.experiment == 1:
                # Experiment 4.1, use reconstruction of normal image
                rec = get_model_reconstruction(model, img, device)
            elif args.experiment == 2:
                # Experiment 4.2, use reconstruction of anomal image
                rec = get_model_reconstruction(model, img_anomal, device)
            else:
                raise ValueError()
            pred = np.abs(rec - img_anomal)
            ap = average_precision(label, pred)
            aps.append(ap)
        results.append(np.mean(aps))

    results = np.array(results)

    # Save results
    ex = 'normal' if args.experiment == 1 else 'anomal'
    np.save(f"./results/experiment5_full_{ex}-rec_AE_best_intensity06_numbers.npy", results)
    # plot_curve(radii, results, ("radius", "ap"),
    #            path=f"./results/experiment5_full_{ex}-rec_AE_best_intensity06.png")
    # plot_curve(radii, results, ("radius", "ap"))
    import IPython ; IPython.embed() ; exit(1)

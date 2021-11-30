"""
Experiment 4:
- Repeat experiment 1, but with an Autoencoder trained on the MOOD training set
- Subexperiment 1: Use reconstruction of normal image, vary intensity
  (Maybe use multiple checkpoints with different validation losses)
- Subexperiment 2: Use reconstruction of anomal image, again vary intensity
"""
import argparse
import random
from tqdm import tqdm

import numpy as np
import torch

from artificial_anomalies import sample_position, disk_anomaly
from fae import load_fae
from models import load_autoencoder
from vqvae import load_vqvae
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
        rec = model(x)
    # If model has multiple outputs, reconstruction is the first one
    if isinstance(rec, tuple):
        rec = rec[0]
    rec = rec[0, 0].cpu().numpy()
    return rec


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['ae', 'vq-vae'])
    parser.add_argument('--model_ckpt', type=str, default='1ex8fxcl/best.pt')
    parser.add_argument('--experiment', type=int, default=1, choices=[1, 2])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--results_path', type=str, default=None)
    args = parser.parse_args()

    ex = 'normal' if args.experiment == 1 else 'anomal'

    intensities = np.linspace(0., 1., num=100)  # First dimension

    if args.results_path is None:
        # Place random seeds
        seed = 0
        random.seed(seed)
        np.random.seed(seed)

        # Select device
        device = args.device if torch.cuda.is_available() else 'cpu'

        # Load image
        img_path = "/home/felix/datasets/MOOD/brain/test_raw/00529.nii.gz"
        volume, _ = load_nii(img_path, primary_axis=2)
        img = volume[volume.shape[0] // 2]
        # imgs = load_mood_test_data()

        # Select ball position and radius
        # position = (128, 200)
        radius = 20

        # Load model
        if args.model_type == 'ae':
            print('Loading AE')
            model, config = load_autoencoder(args.model_ckpt)
            model = model.to(device)
        elif args.model_type == 'vq-vae':
            print('Loading VQ-VAE')
            model, config = load_vqvae(args.model_ckpt)
            if 'latent_dim' not in config:
                config.latent_dim = ""
            model = model.to(device)
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")
        print(f'latent dim {config.latent_dim}')

        # Perform experiment
        ap_results = []  # Gather ap results here
        rec_results = []  # Gather reconstruction error results here
        for intensity in tqdm(intensities):
            aps = []
            rec_errs = []
            # Reset the random seed so for every intensity and blurring we get the same positions
            random.seed(seed)
            np.random.seed(seed)

            # TODO: Remove
            position = sample_position(img)
            img_anomal, label = disk_anomaly(img, position, radius, intensity)
            rec = get_model_reconstruction(model, img_anomal, device)
            rec_err = np.abs(rec - img_anomal)
            import IPython ; IPython.embed() ; exit(1)
            # TODO: Remove end

            for img in imgs:
                # Create anomaly
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
                # Compute reconstruction error
                pred = np.abs(rec - img_anomal)
                # Compute average precision
                ap = average_precision(label, pred)
                aps.append(ap)
                rec_errs.append(pred.mean())
            ap_results.append(np.mean(aps))
            print(f'Intensity: {intensity:.4f} - AP: {ap_results[-1]:.4f}')
            rec_results.append(np.mean(rec_errs))

        ap_results = np.array(ap_results)
        rec_results = np.array(rec_results)

        # Save results
        np.save(f"./results/experiment4/experiment4_full_{ex}-rec_{args.model_type}_lat{config.latent_dim}_best_aps.npy", ap_results)
        np.save(f"./results/experiment4/experiment4_full_{ex}-rec_{args.model_type}_lat{config.latent_dim}_best_rec_errs.npy", rec_results)
    else:
        ap_results = np.load(args.results_path)

    # Plot results
    # plot_curve(intensities, ap_results, ("intensity", "ap"),
    #            path=f"./results/experiment4/experiment4_full_{ex}-rec_{args.model_type}_lat{config.latent_dim}_best.png")
    # plot_curve(intensities, ap_results, ("intensity", "ap"))
    import IPython; IPython.embed(); exit(1)

import argparse
from collections import defaultdict
from glob import glob
import os
import random
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

from dataset import TrainDataset, TestDataset
from models import AutoEncoder, Discriminator, weights_init_gan
from utils import get_training_timings, average_precision


class Trainer:
    def __init__(self, config):

        # Handle initializing from checkpoint
        ckpt = None
        self.config = config
        if config.model_ckpt is not None:
            # Load checkpoint
            ckpt = self.restore_ckpt(config.model_ckpt)
            # Overwrite certain loaded config
            ckpt["config"].update({"model_ckpt": config.model_ckpt}, allow_val_change=True)
            ckpt["config"].update({"debug": config.debug}, allow_val_change=True)
            ckpt["config"].update({"train": config.train}, allow_val_change=True)
            # Use restored config
            self.config.update(ckpt["config"], allow_val_change=True)

        # Just shortening often useed variables
        self.device = self.config.device

        # Init models and optimizer
        self.generator = AutoEncoder(
            inp_size=self.config.inp_size,
            intermediate_resolution=self.config.intermediate_resolution,
            latent_dim=self.config.latent_dim,
            width=self.config.model_width,
            final_activation='sigmoid'
        ).to(self.device)
        self.discriminator = Discriminator(inp_size=self.config.inp_size).to(self.device)
        self.optimizer_g = torch.optim.AdamW(self.generator.parameters(),
                                             lr=self.config.lr,
                                             betas=(0.5, 0.999))
        self.optimizer_d = torch.optim.AdamW(self.discriminator.parameters(),
                                             lr=self.config.lr_d,
                                             betas=(0.5, 0.999))

        wandb.watch(self.generator)

        # If loaded from checkpoint, apply state dict
        if ckpt is not None:
            self.generator.load_state_dict(ckpt["generator_state_dict"])
            self.discriminator.load_state_dict(ckpt["discriminator_state_dict"])
            self.optimizer_g.load_state_dict(ckpt["optimizer_g_state_dict"])
            self.optimizer_d.load_state_dict(ckpt["optimizer_d_state_dict"])
            self.global_step = ckpt["global_step"]
        else:
            self.global_step = 0
            self.generator.apply(weights_init_gan)
            self.discriminator.apply(weights_init_gan)

    def init_train_ds(self, train_files):
        ds = TrainDataset(files=train_files,
                          img_size=self.config.inp_size[0],
                          slice_range=self.config.slice_range)
        print(f"Training on {len(ds)} slices")
        loader = DataLoader(ds, batch_size=self.config.batch_size,
                            shuffle=True,
                            num_workers=self.config.num_workers)
        return loader

    def init_val_ds(self, val_files):
        ds = TestDataset(files=val_files,
                         img_size=self.config.inp_size[0],
                         slice_range=self.config.slice_range)
        print(f"Validating on {len(ds)} slices")
        loader = DataLoader(ds, batch_size=self.config.batch_size,
                            shuffle=True,
                            num_workers=self.config.num_workers)
        return loader

    def save_ckpt(self, name: str):
        torch.save({
            "config": dict(self.config),
            "global_step": self.global_step,
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "optimizer_g_state_dict": self.optimizer_g.state_dict(),
            "optimizer_d_state_dict": self.optimizer_d.state_dict(),
        }, os.path.join(wandb.run.dir, name))

    @staticmethod
    def restore_ckpt(model_ckpt: str):
        run_name, model_name = os.path.split(model_ckpt)
        run_path = f"felix-meissen/reconstruction-score-bias/{run_name}"
        loaded = wandb.restore(model_name, run_path=run_path)
        return torch.load(loaded.name)

    def step(self, x):
        label_real = torch.ones((x.shape[0],), device=self.device)
        label_fake = torch.zeros((x.shape[0],), device=self.device)

        """ 1. Train Generator, maximize log(D(G(z))) """
        self.optimizer_g.zero_grad()

        # Generate fake
        rec = self.generator(x)

        # Discriminate fake
        pred_fake = self.discriminator(rec)

        # Generator GAN loss, try to make it fool the discriminator
        gan_loss_g = F.binary_cross_entropy_with_logits(pred_fake, label_real)
        rec_error = F.l1_loss(rec, x, reduction='none')
        rec_loss = rec_error.mean()

        # Combine losses and backward
        generator_loss = gan_loss_g + self.config.rec_loss_weight * rec_loss
        generator_loss.backward()
        self.optimizer_g.step()

        """ 2. Train critic, maximize log(D(x)) + log(1 - D(G(z))) """
        self.optimizer_d.zero_grad()

        # All real batch
        pred_real = self.discriminator(x)
        loss_real = F.binary_cross_entropy_with_logits(pred_real, label_real)

        # All fake batch
        pred_fake = self.discriminator(rec.detach())
        loss_fake = F.binary_cross_entropy_with_logits(pred_fake, label_fake)

        # Combine losses and backward
        gan_loss_d = (loss_real + loss_fake) / 2.
        gan_loss_d.backward()
        self.optimizer_d.step()

        return {
            'gan_loss_d': gan_loss_d.item(),
            'gan_loss_g': gan_loss_g.item(),
            'rec_loss': rec_loss.item(),
            'gan_loss': gan_loss_d.item() + gan_loss_g.item(),
        }, rec, rec_error

    def val_step(self, x):
        x = x.to(self.device)
        label_real = torch.ones((x.shape[0],), device=self.device)
        label_fake = torch.zeros((x.shape[0],), device=self.device)

        with torch.no_grad():
            # 1.1 Autoencoder GAN loss
            rec = self.generator(x)
            pred_fake = self.discriminator(rec)
            gan_loss_g = F.binary_cross_entropy_with_logits(pred_fake, label_real)

            # 1.2 Autoencoder reconstruction loss
            rec_error = F.l1_loss(rec, x, reduction="none")
            rec_loss = rec_error.mean()

            # 2.1 Critic on real batch
            pred_real = self.discriminator(x)
            d_loss_real = F.binary_cross_entropy_with_logits(pred_real, label_real)

            # 2.2 Critic on fake batch
            pred_fake = self.discriminator(rec.detach())
            d_loss_fake = F.binary_cross_entropy_with_logits(pred_fake, label_fake)

            # 3. Combine losses
            gan_loss_d = d_loss_real + d_loss_fake
            gan_loss = gan_loss_d + gan_loss_g

        return {
            'gan_loss_d': gan_loss_d.item(),
            'gan_loss_g': gan_loss_g.item(),
            'rec_loss': rec_loss.item(),
            'gan_loss': gan_loss.item(),
        }, rec.cpu(), rec_error.cpu()

    def log_progress(self, metrics: dict, images: dict):
        # Print metrics
        log_str = ""
        for name, val in metrics.items():
            log_str += f"{name}: {val:.4f} - "
        print(log_str[:-3])

        # Log metrics to w&b
        wandb.log(metrics, step=self.global_step)
        wandb.log(images, step=self.global_step)

    def train(self, num_steps, train_files, val_files=None):
        # Initialize train dataloader
        trainloader = self.init_train_ds(train_files)
        valloader = None if val_files is None else self.init_val_ds(val_files)

        print(f"START TRAINING FOR {num_steps} STEPS")
        # Initialize helper variables
        start_time = time()
        i_step = 0
        train_losses = defaultdict(list)
        best_val_ap = 0.
        best_val_loss = float('inf')

        self.generator.train()
        self.discriminator.train()
        while True:  # Stopping is handled by num_steps
            for x in trainloader:
                # x = x * 2. - 1.  # Normalize to -1, 1, TODO: remove this
                x = x.to(self.device)

                # Update step
                losses, rec, rec_error = self.step(x)
                for k, v in losses.items():
                    train_losses[k].append(v)

                # Increment counters
                self.global_step += 1
                i_step += 1

                # Log if necessary
                if i_step % self.config.log_interval == 0:
                    # Log timings
                    time_elapsed, _, time_left = get_training_timings(
                        start_time, i_step, num_steps
                    )
                    print(f"Step [{i_step}|{num_steps}] - "
                          f"Time elapsed: {time_elapsed}, "
                          f"Time left: {time_left} - ", end='')

                    # Log metrics
                    x_log = x[:self.config.num_imgs_log].detach().cpu()
                    rec_log = rec[:self.config.num_imgs_log].detach().cpu()
                    rec_error_log = rec_error[:self.config.num_imgs_log].detach().cpu()
                    self.log_progress(
                        {f'train/{k}': np.mean(v) for k, v in train_losses.items()},
                        {"train/inputs": wandb.Image(x_log),
                         "train/reconstructions": wandb.Image(rec_log),
                         "train/rec error": wandb.Image(rec_error_log)}
                    )

                    # Empty accumulated losses
                    train_losses = defaultdict(list)

                # Validate if necessary and possible
                if i_step % self.config.val_interval == 0 and valloader is not None:
                    val_loss, val_ap = self.validate(valloader)
                    if val_ap > best_val_ap:
                        print(f"New best validation AP: {val_ap:.4f}")
                        best_val_ap = val_ap
                        self.save_ckpt("best_ap.pt")
                    if val_loss < best_val_loss:
                        print(f"New best validation loss: {val_loss:.4f}")
                        best_val_loss = val_loss
                        self.save_ckpt("best_loss.pt")

                # Stop training if maximum number of steps is reached
                if i_step >= num_steps:
                    self.save_ckpt("last.pt")
                    return

    def validate(self, valloader):
        # Set models to evaluation
        self.generator.eval()
        self.discriminator.eval()

        # Initialize helper variables
        val_losses = defaultdict(list)
        aps = []

        # Validation loop
        for _, x, label in valloader:
            # x = x * 2. - 1.  # Normalize to -1, 1, TODO: remove this
            losses, rec, rec_error = self.val_step(x)
            for k, v in losses.items():
                val_losses[k].append(v)
            aps.append(average_precision(label.cpu(), rec_error.cpu()))

        # Log results of validation
        x_log = x[:self.config.num_imgs_log].detach().cpu()
        rec_log = rec[:self.config.num_imgs_log].detach().cpu()
        rec_error_log = rec_error[:self.config.num_imgs_log].detach().cpu()
        self.log_progress(
            {**{f'val/{k}': np.mean(v) for k, v in val_losses.items()},
             "val/average precision": np.mean(aps)},
            {"val/inputs": wandb.Image(x_log),
             "val/reconstructions": wandb.Image(rec_log),
             "val/rec error": wandb.Image(rec_error_log)}
        )

        # Set model back to train
        self.generator.train()
        self.discriminator.train()

        return np.mean(val_losses['rec_loss']), np.mean(aps)

    def test(self):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # General scripting params
    parser.add_argument("--no_train", action="store_false", dest="train")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model_ckpt", type=str, default=None,
                        help="in the form <run_name>/<last or best>.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    # Data params
    parser.add_argument("--inp_size", nargs='+', default=[256, 256])
    parser.add_argument("--slice_range", nargs='+', default=(120, 140))
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--load_to_ram", type=bool, default=True)
    # Model params
    parser.add_argument("--model_width", type=int, default=32)
    parser.add_argument("--intermediate_resolution", nargs='+', default=[8, 8])
    parser.add_argument("--latent_dim", type=int, default=128)
    # Training params
    parser.add_argument("--num_steps", type=int, default=int(1e4))
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--val_interval", type=int, default=200)
    parser.add_argument("--num_imgs_log", type=int, default=12)
    # Real hparams
    parser.add_argument("--lr", type=int, default=1e-4)
    parser.add_argument("--lr_d", type=int, default=1e-5)  # 5e-5
    parser.add_argument("--rec_loss_weight", type=int, default=100)
    parser.add_argument("--rec_loss_weight_min", type=int, default=50)
    config = parser.parse_args()

    config.device = config.device if torch.cuda.is_available() else "cpu"
    print(f"Training on {config.device}")

    # Set random seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True

    # Init w&b
    wandb.init(project="reconstruction-score-bias", entity="felix-meissen",
               mode="disabled" if config.debug else "online")
    wandb.config.update(config)

    # Get train files
    files = glob("/home/felix/datasets/MOOD/brain/train/*.nii.gz")
    split_idx = int(len(files) * config.val_fraction)
    train_files = files[split_idx:]
    val_files = files[:split_idx]
    print(f"Found {len(files)} files, using {len(train_files)} "
          f"for training and {len(val_files)} for validation")

    # Init trainer
    trainer = Trainer(wandb.config)

    # Train
    trainer.train(wandb.config.num_steps, train_files, val_files)

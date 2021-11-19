from argparse import Namespace
from math import log
import os
from typing import List

import torch
from torch import Tensor
import torch.nn as nn

import wandb


""""""""""""""""""""""""""""""""" Utilities """""""""""""""""""""""""""""""""


def weights_init_relu(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def weights_init_leaky_relu(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def weights_init_relu_normal(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Reshape(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x: Tensor) -> Tensor:
        return x.view(self.size)


""""""""""""""""""""""""""""""""" Unified AE """""""""""""""""""""""""""""""""


def build_encoder(in_channels: int, hidden_dims: List[int]) -> nn.Module:
    # Build encoder
    encoder = []
    for h_dim in hidden_dims:
        encoder.append(
            nn.Sequential(
                nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2,
                          padding=1, bias=False),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU(),
            )
        )
        in_channels = h_dim
    return nn.Sequential(*encoder)


def build_decoder(out_channels: int, hidden_dims: List[int]) -> nn.Module:
    # Build decoder
    decoder = []
    for i in range(len(hidden_dims) - 1, 0, -1):
        decoder.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i - 1],
                                   kernel_size=3, stride=2, padding=1,
                                   output_padding=1, bias=False),
                nn.BatchNorm2d(hidden_dims[i - 1]),
                nn.LeakyReLU(),
            )
        )
    decoder.append(
        nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[0], hidden_dims[0],
                               kernel_size=3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.LeakyReLU(),
        )
    )
    # Final layer
    decoder.append(
        nn.Conv2d(hidden_dims[0], out_channels, kernel_size=1, bias=False)
    )
    return nn.Sequential(*decoder)


""""""""""""""""""""""""""""""""" AutoEncoder """""""""""""""""""""""""""""""""


class AutoEncoder(nn.Module):
    def __init__(self,
                 inp_size,
                 intermediate_resolution=[8, 8],
                 in_channels: int = 1,
                 out_channels: int = 1,
                 latent_dim: int = 256,
                 width: int = 32,
                 hidden_dims: List = None) -> None:
        super().__init__()

        self.latent_dim = latent_dim

        assert len(inp_size) == 2
        if isinstance(inp_size, list) or isinstance(inp_size, tuple):
            inp_size = torch.tensor(inp_size)
        assert len(intermediate_resolution) == 2
        if isinstance(intermediate_resolution, list) or isinstance(intermediate_resolution, tuple):
            intermediate_resolution = torch.tensor(intermediate_resolution)

        if hidden_dims is None:
            size = inp_size[-1]
            res = intermediate_resolution[-1]
            num_layers = int(log(size, 2) - log(res, 2))
            hidden_dims = [min(128, width * (2**i)) for i in range(num_layers)]
        self.hidden_dims = hidden_dims

        intermediate_feats = torch.prod(intermediate_resolution) * hidden_dims[-1]

        # Encoder
        self.encoder = build_encoder(in_channels, hidden_dims)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(intermediate_feats, latent_dim, bias=False),
        )
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, intermediate_feats, bias=False),
            Reshape((-1, hidden_dims[-1], *intermediate_resolution)),
        )

        # Decoder
        self.decoder = build_decoder(1, hidden_dims)

    def forward(self, inp: Tensor) -> Tensor:
        # Encoder
        z = self.encoder(inp)

        # Bottleneck
        z = self.bottleneck(z)
        z = self.decoder_input(z)

        # Decoder
        y = self.decoder(z)
        return y


def load_autoencoder(model_ckpt: str) -> AutoEncoder:
    """Load a model of the AutoEncoder class from a path of the format
    <run_name>/<last or best>.pt"""

    # Restore checkpoint
    run_name, model_name = os.path.split(model_ckpt)
    run_path = f"felix-meissen/reconstruction-score-bias/{run_name}"
    loaded = wandb.restore(model_name, run_path=run_path)
    ckpt = torch.load(loaded.name)
    os.remove(loaded.name)

    # Extract config
    config = Namespace(**ckpt['config'])

    # Init model
    model = AutoEncoder(
        inp_size=config.inp_size,
        intermediate_resolution=config.intermediate_resolution,
        latent_dim=config.latent_dim,
        width=config.model_width
    )

    # Load weights
    model.load_state_dict(ckpt["model_state_dict"])

    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.rand((2, 1, 256, 256)).to(device)
    model = AutoEncoder(inp_size=(256, 256)).to(device)
    print(model)
    y = model(x)
    print(y.shape)

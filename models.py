from argparse import Namespace
from math import log
import os
from typing import List

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding

import wandb

# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Partly adapted from from https://github.com/rosinality/vq-vae-2-pytorch


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


def weights_init_gan(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Reshape(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x: Tensor) -> Tensor:
        return x.view(self.size)


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


""""""""""""""""""""""""""""""""" Unified AE """""""""""""""""""""""""""""""""


def vanilla_encoder(in_channels: int, hidden_dims: List[int]) -> nn.Module:
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


def vanilla_decoder(out_channels: int, hidden_dims: List[int]) -> nn.Module:
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
                 hidden_dims: List = None,
                 final_activation='identity') -> None:
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
            max_width = width * 4
            num_layers = int(log(size, 2) - log(res, 2))
            hidden_dims = [min(max_width, width * (2**i)) for i in range(num_layers)]
        self.hidden_dims = hidden_dims

        intermediate_feats = torch.prod(intermediate_resolution) * hidden_dims[-1]

        # Encoder
        self.encoder = vanilla_encoder(in_channels, hidden_dims)

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
        self.decoder = vanilla_decoder(out_channels, hidden_dims)

        if final_activation == 'identity':
            self.final_activation = lambda x: x
        elif final_activation == 'sigmoid':
            self.final_activation = torch.sigmoid
        elif final_activation == 'tanh':
            self.final_activation = torch.tanh
        elif final_activation == 'relu':
            self.final_activation = torch.relu
        else:
            raise ValueError(f"Unknown activation {final_activation}")

    def forward(self, inp: Tensor) -> Tensor:
        # Encoder
        z = self.encoder(inp)

        # Bottleneck
        z = self.bottleneck(z)
        z = self.decoder_input(z)

        # Decoder
        y = self.decoder(z)

        return self.final_activation(y)


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

    return model, config


""""""""""""""""""""""""""""""""" Spatial AE """""""""""""""""""""""""""""""""


class SpatialAutoEncoder(nn.Module):
    def __init__(self,
                 inp_size,
                 intermediate_resolution=[8, 8],
                 latent_channels: int = 1,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 width: int = 32,
                 hidden_dims: List = None,
                 final_activation='identity') -> None:
        super().__init__()

        assert len(inp_size) == 2
        if isinstance(inp_size, list) or isinstance(inp_size, tuple):
            inp_size = torch.tensor(inp_size)
        assert len(intermediate_resolution) == 2
        if isinstance(intermediate_resolution, list) or isinstance(intermediate_resolution, tuple):
            intermediate_resolution = torch.tensor(intermediate_resolution)

        if hidden_dims is None:
            size = inp_size[-1]
            res = intermediate_resolution[-1]
            max_width = width * 4
            num_layers = int(log(size, 2) - log(res, 2))
            hidden_dims = [min(max_width, width * (2**i)) for i in range(num_layers)]
        self.hidden_dims = hidden_dims

        # Encoder
        self.encoder = vanilla_encoder(in_channels, hidden_dims)

        # Bottleneck
        self.bottleneck = nn.Conv2d(hidden_dims[-1], latent_channels,
                                    kernel_size=1, bias=False)
        self.decoder_input = nn.ConvTranspose2d(latent_channels, hidden_dims[-1],
                                                kernel_size=1, bias=False)

        # Decoder
        self.decoder = vanilla_decoder(out_channels, hidden_dims)

        if final_activation == 'identity':
            self.final_activation = lambda x: x
        elif final_activation == 'sigmoid':
            self.final_activation = torch.sigmoid
        elif final_activation == 'tanh':
            self.final_activation = torch.tanh
        elif final_activation == 'relu':
            self.final_activation = torch.relu
        else:
            raise ValueError(f"Unknown activation {final_activation}")

    def forward(self, inp: Tensor) -> Tensor:
        # Encoder
        z = self.encoder(inp)

        # Bottleneck
        z = self.bottleneck(z)
        z = self.decoder_input(z)

        # Decoder
        y = self.decoder(z)

        return self.final_activation(y)


def load_spatial_autoencoder(model_ckpt: str) -> SpatialAutoEncoder:
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
    if 'latent_channels' not in config:
        config.latent_channels = 1

    # Init model
    model = SpatialAutoEncoder(
        inp_size=config.inp_size,
        intermediate_resolution=config.intermediate_resolution,
        latent_channels=config.latent_channels,
        width=config.model_width
    )

    # Load weights
    model.load_state_dict(ckpt["model_state_dict"])

    return model, config


""""""""""""""""""""""""""""""""" Skip-AE """""""""""""""""""""""""""""""""


class SkipAutoEncoder(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()

        def down_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2,
                          padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )

        def up_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3,
                                   stride=2, padding=1, output_padding=1,
                                   bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )

        max_width = width * 4
        hidden_dims = [min(max_width, width * (2**i)) for i in range(5)]

        # Encoder
        self.down1 = down_block(1, hidden_dims[0])
        self.down2 = down_block(hidden_dims[0], hidden_dims[1])
        self.down3 = down_block(hidden_dims[1], hidden_dims[2])
        self.down4 = down_block(hidden_dims[2], hidden_dims[3])
        self.down5 = down_block(hidden_dims[3], hidden_dims[4])

        # Decoder
        self.up1 = up_block(hidden_dims[4], hidden_dims[3])
        self.up2 = up_block(hidden_dims[3], hidden_dims[2])
        self.up3 = up_block(hidden_dims[2], hidden_dims[1])
        self.up4 = up_block(hidden_dims[1], hidden_dims[0])
        self.up5 = up_block(hidden_dims[0], 1)

        self.final_conv = nn.Conv2d(1, 1, kernel_size=1, stride=1)

    def forward(self, inp: Tensor) -> Tensor:
        # Encoder
        x1 = self.down1(inp)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        # Decoder
        y1 = self.up1(x5)
        y2 = self.up2(y1 + x4)
        y3 = self.up3(y2)
        y4 = self.up4(y3)
        y5 = self.up5(y4)

        y = self.final_conv(y5)
        y = torch.tanh(y)

        return y


def load_skip_autoencoder(model_ckpt: str) -> SkipAutoEncoder:
    """Load a model of the SkipAutoEncoder class from a path of the format
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
    model = SkipAutoEncoder(
        width=config.model_width
    )

    # Load weights
    model.load_state_dict(ckpt["model_state_dict"])

    return model, config


""""""""""""""""""""""""""""""""" Pix2Pix """""""""""""""""""""""""""""""""


class Discriminator(nn.Module):
    def __init__(self, inp_size, depth=5, in_channels=1):
        super().__init__()

        def discriminator_block(in_filters, out_filters):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        assert len(inp_size) == 2
        if isinstance(inp_size, list) or isinstance(inp_size, tuple):
            inp_size = torch.tensor(inp_size)
        out_res = torch.div(inp_size, 2**depth, rounding_mode='floor')

        model = []
        for i in range(depth):
            out_channels = min(128, in_channels * 2 ** (i + 1))
            model += discriminator_block(in_channels, out_channels)
            in_channels = out_channels
        model.append(nn.Conv2d(out_channels, 1, out_res, bias=False))
        self.model = nn.Sequential(*model)

    def forward(self, img):
        return self.model(img).view(-1)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    x = torch.rand((2, 1, 256, 256)).to(device)
    model = SpatialAutoEncoder(inp_size=(256, 256)).to(device)
    # model = SkipAutoEncoder().to(device)
    print(model)
    y = model(x)
    print(y.shape)
    import IPython ; IPython.embed() ; exit(1)

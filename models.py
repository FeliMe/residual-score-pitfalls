from typing import List
from math import log

import torch
from torch import Tensor
import torch.nn as nn


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


""""""""""""""""""""""""""""""""" AutoEncoder """""""""""""""""""""""""""""""""


class AutoEncoder(nn.Module):
    def __init__(self,
                 inp_size,
                 intermediate_resolution = [8, 8],
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

        """ Build encoder """
        encoder = []
        self.feat_size = inp_size
        for h_dim in self.hidden_dims:
            encoder.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=4,
                              stride=2,
                              padding=1,
                              bias=False),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
            # Floor divide
            self.feat_size = torch.div(self.feat_size, 2, rounding_mode="trunc")

        self.encoder = nn.Sequential(*encoder)

        n_feats = int(torch.prod(self.feat_size)) * hidden_dims[-1]
        self.bottleneck = nn.Linear(n_feats, latent_dim)

        """ Build decoder """
        decoder = []

        self.decoder_input = nn.Linear(latent_dim, n_feats)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i+1],
                                       kernel_size=4,
                                       stride=2,
                                       padding=1,
                                       bias=False),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.LeakyReLU()
                )
            )

        decoder.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1],
                                   kernel_size=4,
                                   stride=2,
                                   padding=1,
                                   bias=False),
                nn.BatchNorm2d(hidden_dims[-1]),
                nn.LeakyReLU(),
                nn.Conv2d(hidden_dims[-1], out_channels, kernel_size=3,
                          padding=1),
                nn.Tanh()
            )
        )

        self.decoder = nn.Sequential(*decoder)


    def forward(self, inp: Tensor) -> Tensor:
        # Encoder
        z = self.encoder(inp)

        # Bottleneck
        z = torch.flatten(z, start_dim=1)
        z = self.bottleneck(z)
        z = self.decoder_input(z)
        y = z.view(-1, self.hidden_dims[0], *self.feat_size.tolist())

        # Decoder
        y = self.decoder(y)
        return y


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.rand((2, 1, 256, 256)).to(device)
    model = AutoEncoder(inp_size=(256, 256)).to(device)
    print(model)
    y = model(x)
    print(y.shape)

    import IPython ; IPython.embed() ; exit(1)

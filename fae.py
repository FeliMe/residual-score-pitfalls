from argparse import Namespace
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models
import wandb


RESNETLAYERS = ['layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']


def _set_requires_grad_false(layer):
    for param in layer.parameters():
        param.requires_grad = False


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, resnet, layer_names=RESNETLAYERS):
        """
        Returns features on multiple levels from a ResNet18.
        Available layers: 'layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool'
        Args:
            resnet (nn.Module): Type of resnet used
            layer_names (list): List of string of layer names where to return
                                the features. Must be ordered
        Returns:
            out (dict): Dictionary containing the extracted features as
                        torch.tensors
        """
        super().__init__()

        _set_requires_grad_false(resnet)

        # [b, 3, 256, 256]
        self.layer0 = nn.Sequential(
            *list(resnet.children())[:4])  # [b, 64, 64, 64]
        self.layer1 = resnet.layer1  # [b, 64, 64, 64]
        self.layer2 = resnet.layer2  # [b, 128, 32, 32]
        self.layer3 = resnet.layer3  # [b, 256, 16, 16]
        self.layer4 = resnet.layer4  # [b, 512, 8, 8]
        self.avgpool = resnet.avgpool  # [b, 512, 1, 1]

        self.layer_names = layer_names

    def forward(self, inp):
        if inp.shape[1] == 1:
            inp = inp.repeat(1, 3, 1, 1)
        out = {}
        for name, module in self._modules.items():
            inp = module(inp)
            if name in self.layer_names:
                out[name] = inp
            if name == self.layer_names[-1]:
                break
        return out


class ResNet18FeatureExtractor(ResNetFeatureExtractor):
    def __init__(self, layer_names=RESNETLAYERS):
        super().__init__(tv_models.resnet18(pretrained=True), layer_names)


class Extractor(nn.Module):
    """
    Muti-scale regional feature based on VGG-feature maps.
    """
    def __init__(
        self,
        cnn_layers=['layer1', 'layer2', 'layer3'],
        upsample='bilinear',
        inp_size=256,
        keep_feature_prop=1.0,
    ):
        super().__init__()

        self.backbone = ResNet18FeatureExtractor(layer_names=cnn_layers)
        self.inp_size = inp_size
        self.featmap_size = inp_size // 4
        self.upsample = upsample
        self.align_corners = True if upsample == "bilinear" else None

        # Find out how many channels we got from the backbone
        c_out = self.get_out_channels()

        # Create mask to drop random features_channels
        self.register_buffer('feature_mask', torch.Tensor(c_out).uniform_() < keep_feature_prop)
        self.c_out = self.feature_mask.sum().item()

    def get_out_channels(self):
        device = next(self.backbone.parameters()).device
        inp = torch.randn((2, 1, self.inp_size, self.inp_size), device=device)
        return sum([feat_map.shape[1] for feat_map in self.backbone(inp).values()])

    def forward(self, inp):
        if type(inp) is dict:
            feat_maps = inp
        else:
            feat_maps = self.backbone(inp)

        features = []
        for feat_map in feat_maps.values():
            # Resizing
            feat_map = F.interpolate(feat_map, size=self.featmap_size,
                                     mode=self.upsample,
                                     align_corners=self.align_corners)
            features.append(feat_map)

        # Concatenate to tensor
        features = torch.cat(features, dim=1)

        # Drop out feature maps
        features = features[:, self.feature_mask]

        return features


class FeatureAE(nn.Module):
    def __init__(self, c_in, c_z, ks=3, use_batchnorm=True):
        super().__init__()

        pad = ks // 2

        # Encoder
        enc = []
        # Layer 1
        enc += [nn.Conv2d(c_in, (c_in + 2 * c_z) // 2, kernel_size=ks,
                          padding=pad, bias=False)]
        if use_batchnorm:
            enc += [nn.BatchNorm2d((c_in + 2 * c_z) // 2)]
        enc += [nn.LeakyReLU()]
        # Layer 2
        enc += [nn.Conv2d((c_in + 2 * c_z) // 2, 4 * c_z, kernel_size=ks,
                          padding=pad, bias=False)]
        if use_batchnorm:
            enc += [nn.BatchNorm2d(4 * c_z)]
        enc += [nn.LeakyReLU()]
        # Layer 2.1
        # ---------------------------------------------------------------
        enc += [nn.Conv2d(4 * c_z, 2 * c_z, kernel_size=ks,
                          padding=pad, bias=False)]
        if use_batchnorm:
            enc += [nn.BatchNorm2d(2 * c_z)]
        enc += [nn.LeakyReLU()]
        # ---------------------------------------------------------------
        # Layer 3
        enc += [nn.Conv2d(2 * c_z, c_z, kernel_size=ks, padding=pad,
                          bias=False)]
        self.encoder = nn.Sequential(*enc)

        # Decoder
        dec = []
        # Layer 1
        dec += [nn.Conv2d(c_z, 2 * c_z, kernel_size=ks, padding=pad,
                          bias=False)]
        if use_batchnorm:
            dec += [nn.BatchNorm2d(2 * c_z)]
        dec += [nn.LeakyReLU()]
        # Layer 2.1
        # ---------------------------------------------------------------
        dec += [nn.Conv2d(2 * c_z, 4 * c_z, kernel_size=ks, padding=pad,
                          bias=False)]
        if use_batchnorm:
            dec += [nn.BatchNorm2d(4 * c_z)]
        dec += [nn.LeakyReLU()]
        # ---------------------------------------------------------------
        # Layer 2
        dec += [nn.Conv2d(4 * c_z, (c_in + 2 * c_z) // 2, kernel_size=ks,
                          padding=pad, bias=False)]
        if use_batchnorm:
            dec += [nn.BatchNorm2d((c_in + 2 * c_z) // 2)]
        dec += [nn.LeakyReLU()]
        # Layer 3
        dec += [nn.Conv2d((c_in + 2 * c_z) // 2, c_in, kernel_size=ks,
                          padding=pad, bias=False)]
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y


def load_fae(model_ckpt: str) -> FeatureAE:
    """Load a model of the FeatureAE class from a path of the format
    <run_name>/<last or best>.pt"""

    # Restore checkpoint
    run_name, model_name = os.path.split(model_ckpt)
    run_path = f"felix-meissen/reconstruction-score-bias/{run_name}"
    loaded = wandb.restore(model_name, run_path=run_path)
    ckpt = torch.load(loaded.name)
    os.remove(loaded.name)

    # Extract config
    config = Namespace(**ckpt['config'])

    # Init feature extractor
    extractor = Extractor(cnn_layers=['layer1', 'layer2', 'layer3'],
                          inp_size=config.inp_size[0])
    extractor.load_state_dict(ckpt["extractor_state_dict"])

    # Init model
    model = FeatureAE(c_in=extractor.feature_mask.sum().item(), c_z=config.latent_dim)
    model.load_state_dict(ckpt["model_state_dict"])

    return model, extractor, config


if __name__ == '__main__':
    device = "cpu"
    inp_size = 256
    keep_feature_prop = 0.8
    extractor = Extractor(inp_size=inp_size,
                          keep_feature_prop=keep_feature_prop,
                          cnn_layers=['layer1', 'layer2', 'layer3'],
                          ).to(device)
    ae = FeatureAE(c_in=extractor.c_out, c_z=256).to(device)
    x = torch.randn((8, 1, inp_size, inp_size), device=device)
    from time import perf_counter
    t_start = perf_counter()
    feats = extractor(x)
    print(feats.shape)
    print(f"Extraction took {perf_counter() - t_start:.2f} seconds")
    y = ae(feats)
    print(y.shape)
    print(f"AE took {perf_counter() - t_start:.2f} seconds")

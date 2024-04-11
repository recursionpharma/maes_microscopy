# © Recursion Pharmaceuticals 2024
from typing import Dict

import timm.models.vision_transformer as vit
import torch


def build_imagenet_baselines() -> Dict[str, torch.jit.ScriptModule]:
    """This returns the prepped imagenet encoders from timm, not bad for microscopy data."""
    vit_backbones = [
        _make_vit(vit.vit_small_patch16_384),
        _make_vit(vit.vit_base_patch16_384),
        _make_vit(vit.vit_base_patch8_224),
        _make_vit(vit.vit_large_patch16_384),
    ]
    model_names = [
        "vit_small_patch16_384",
        "vit_base_patch16_384",
        "vit_base_patch8_224",
        "vit_large_patch16_384",
    ]
    imagenet_encoders = list(map(_make_torchscripted_encoder, vit_backbones))
    return {name: model for name, model in zip(model_names, imagenet_encoders)}


def _make_torchscripted_encoder(vit_backbone) -> torch.jit.ScriptModule:
    dummy_input = torch.testing.make_tensor(
        (2, 6, 256, 256),
        low=0,
        high=255,
        dtype=torch.uint8,
        device=torch.device("cpu"),
    )
    encoder = torch.nn.Sequential(
        Normalizer(),
        torch.nn.LazyInstanceNorm2d(
            affine=False, track_running_stats=False
        ),  # this module performs self-standardization, very important
        vit_backbone,
    ).to(device="cpu")
    _ = encoder(dummy_input)  # get those lazy modules built
    return torch.jit.freeze(torch.jit.script(encoder.eval()))


def _make_vit(constructor):
    return constructor(
        pretrained=True,  # download imagenet weights
        img_size=256,  # 256x256 crops
        in_chans=6,  # we expect 6-channel microscopy images
        num_classes=0,
        fc_norm=None,
        class_token=True,
        global_pool="avg",  # minimal perf diff btwn "cls" and "avg"
    )


class Normalizer(torch.nn.Module):
    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        pixels = pixels.float()
        pixels /= 255.0
        return pixels

import json

import torch
from hydra.utils import instantiate

model_config = {
    "_target_": "mae.MAE",
    "mask_ratio": 0.0,
    "input_norm": {
        "_target_": "torch.nn.Sequential",
        "_args_": [
            {"_target_": "normalizer.Normalizer"},
            {
                "_target_": "torch.nn.LazyInstanceNorm2d",
                "affine": False,
                "track_running_stats": False,
            },
        ],
    },
    "encoder": {
        "_target_": "mae_modules.MAEEncoder",
        "channel_agnostic": True,
        "max_in_chans": 11,
        "vit_backbone": {
            "_target_": "vit.sincos_positional_encoding_vit",
            "vit_backbone": {
                "_target_": "vit.vit_small_patch16_256",
                "global_pool": "avg",
            },
        },
    },
    "decoder": {
        "_target_": "mae_modules.CAMAEDecoder",
        "num_modalities": 6,
        "tokens_per_modality": 256,
        "embed_dim": 256,
        "depth": 4,
        "num_heads": 16,
        "mlp_ratio": 4,
        "qkv_bias": True,
        "norm_layer": {
            "_target_": "torch.nn.LayerNorm",
            "_partial_": True,
            "eps": 1e-6,
        },
    },
    "norm_pix_loss": False,
    "fourier_loss_weight": 0.01,
    "fourier_loss": {
        "_target_": "loss.FourierLoss",
        "num_multimodal_modalities": 6,  # Refers to the *num_modalities
    },
    "loss": {"_target_": "torch.nn.MSELoss", "reduction": "none"},
    "optimizer": {
        "_target_": "timm.optim.lion.Lion",
        "_partial_": True,
        "lr": 1e-4,
        "weight_decay": 0.05,
        "betas": (0.9, 0.95),
    },
    "lr_scheduler": {
        "_target_": "torch.optim.lr_scheduler.OneCycleLR",
        "_partial_": True,
        "max_lr": 1e-4,
        "pct_start": 0.1,
        "anneal_strategy": "cos",
    },
}

path_to_model = "models/phenom_beta/last.pickle"

instantiated_mae = instantiate(model_config)
state_dict = torch.load(path_to_model, map_location="cpu")
instantiated_mae.load_state_dict(state_dict["state_dict"])

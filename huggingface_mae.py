import torch

from timm.optim.lion import Lion
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn import MSELoss, LayerNorm, LazyInstanceNorm2d, Sequential
from hydra.utils import instantiate

from mae_modules import MAEEncoder, CAMAEDecoder
from loss import FourierLoss
from vit import sincos_positional_encoding_vit, vit_small_patch16_256
from mae import MAE, Normalizer

model_config = {
    "_target_": MAE,
    "mask_ratio": 0.75,
    "input_norm": {
        "_target_": Sequential,
        "_args_": [
            {"_target_": Normalizer},
            {
                "_target_": LazyInstanceNorm2d,
                "affine": False,
                "track_running_stats": False
            }
        ]
    },
    "encoder": {
        "_target_": MAEEncoder,
        "channel_agnostic": True,
        "max_in_chans": 11,
        "vit_backbone": {
            "_target_": sincos_positional_encoding_vit,
            "vit_backbone": {
                "_target_": vit_small_patch16_256,
                "global_pool": "avg"
            }
        }
    },
    "decoder": {
        "_target_": CAMAEDecoder,
        "num_modalities": 6,
        "tokens_per_modality": 256,
        "embed_dim": 256,
        "depth": 4,
        "num_heads": 16,
        "mlp_ratio": 4,
        "qkv_bias": True,
        "norm_layer": {
            "_target_": LayerNorm,
            "_partial_": True,
            "eps": 1e-6
        }
    },
    "norm_pix_loss": False,
    "fourier_loss_weight": 0.01,
    "fourier_loss": {
        "_target_": FourierLoss,
        "num_multimodal_modalities": 6  # Refers to the *num_modalities
    },
    "loss": {
        "_target_": MSELoss,
        "reduction": "none"
    },
    "optimizer": {
        "_target_": Lion,
        "_partial_": True,
        "lr": 1e-4,
        "weight_decay": 0.05,
        "betas": (0.9, 0.95)
    },
    "lr_scheduler": {
        "_target_": OneCycleLR,
        "_partial_": True,
        "max_lr": 1e-4,
        "pct_start": 0.1,
        "anneal_strategy": "cos"
    }
}

path_to_model = "models/phenom_beta/last.pickle"

instantiated_mae = instantiate(model_config)
state_dict = torch.load(path_to_model, map_location="cpu")
instantiated_mae.load_state_dict(state_dict["state_dict"])
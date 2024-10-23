from huggingface_mae import MAEConfig, MAEModel

config = MAEConfig(
    input_norm={
        "_target_": "torch.nn.Sequential",
        "_args_": [
            {"_target_": "normalizer.Normalizer"},
            {"_target_": "torch.nn.LazyInstanceNorm2d", "affine": False, "track_running_stats": False}
        ]
    },
    encoder={
        "_target_": "mae_modules.MAEEncoder",
        "channel_agnostic": True,
        "max_in_chans": 11,
        "vit_backbone": {
            "_target_": "vit.sincos_positional_encoding_vit",
            "vit_backbone": {
                "_target_": "vit.vit_small_patch16_256",
                "global_pool": "avg"
            }
        }
    },
    decoder={
        "_target_": "mae_modules.CAMAEDecoder",
        "num_modalities": 6,
        "tokens_per_modality": 256,
        "embed_dim": 256,
        "depth": 4,
        "num_heads": 16,
        "mlp_ratio": 4,
        "qkv_bias": True,
        "norm_layer": {"_target_": "torch.nn.LayerNorm", "_partial_": True, "eps": 1e-6}
    },
    loss={"_target_": "torch.nn.MSELoss", "reduction": "none"},
    fourier_loss={
        "_target_": "loss.FourierLoss",
        "num_multimodal_modalities": 6
    }
)

# model = MAEModel.from_pretrained("models/phenom_beta", filename="last.pickle", from_state_dict=True)
model = MAEModel.from_pretrained("models/phenom_beta", filename="last.pickle")

huggingface_modelpath="recursionpharma/test-pb-model"
model.push_to_hub(huggingface_modelpath)
# model.save_pretrained(huggingface_modelpath, push_to_hub=True, repo_id=huggingface_modelpath)

localpath = "models/phenom_beta_huggingface"
model.save_pretrained(localpath)
model = MAEModel.from_pretrained(localpath)
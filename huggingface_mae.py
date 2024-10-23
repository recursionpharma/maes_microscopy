import warnings
import logging

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

# from functools import partial
from typing import Any, Dict, Optional, Tuple, Union
from hydra.utils import instantiate

# from torchmetrics import MeanMetric, Metric
# from torch.types import Number

from vit import generate_2d_sincos_pos_embeddings
from mae_modules import MAEEncoder, CAMAEDecoder, MAEDecoder
from mae_utils import unflatten_tokens, flatten_images, apply_norm_pix

TensorDict = Dict[str, torch.Tensor]

class MAEConfig(PretrainedConfig):
    model_type = "MAE"

    def __init__(self,
                 mask_ratio=0.75,
                 encoder=None,
                 decoder=None,
                 loss=None,
                 optimizer=None,
                 input_norm=None,
                 norm_pix_loss=False,
                 apply_loss_unmasked=False,
                 fourier_loss=None,
                 fourier_loss_weight=0.0,
                 lr_scheduler=None,
                 use_MAE_weight_init=False,
                 crop_size=-1,
                 num_blocks_to_freeze=0,
                 trim_encoder_blocks=None,
                 layernorm_unfreeze=True,
                 mask_fourier_loss=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.mask_ratio = mask_ratio
        self.encoder = encoder
        self.decoder = decoder
        self.loss = loss
        self.optimizer = optimizer
        self.input_norm = input_norm
        self.norm_pix_loss = norm_pix_loss
        self.apply_loss_unmasked = apply_loss_unmasked
        self.fourier_loss = fourier_loss
        self.fourier_loss_weight = fourier_loss_weight
        self.lr_scheduler = lr_scheduler
        self.use_MAE_weight_init = use_MAE_weight_init
        self.crop_size = crop_size
        self.num_blocks_to_freeze = num_blocks_to_freeze
        self.trim_encoder_blocks = trim_encoder_blocks
        self.layernorm_unfreeze = layernorm_unfreeze
        self.mask_fourier_loss = mask_fourier_loss

class MAEModel(PreTrainedModel):
    config_class = MAEConfig

    # Loss metrics
    TOTAL_LOSS = "loss"
    RECON_LOSS = "reconstruction_loss"
    FOURIER_LOSS = "fourier_loss"

    def __init__(self, config: MAEConfig):
        super().__init__(config)

        self.mask_ratio = config.mask_ratio
        self.encoder = instantiate(config.encoder)
        self.decoder = instantiate(config.decoder)
        self.input_norm = instantiate(config.input_norm)
        self.norm_pix_loss = config.norm_pix_loss
        self.fourier_loss_weight = config.fourier_loss_weight
        self.apply_loss_unmasked = config.apply_loss_unmasked
        self.blocks_to_freeze = config.num_blocks_to_freeze
        self.trim_encoder_blocks = config.trim_encoder_blocks
        self.layernorm_unfreeze = config.layernorm_unfreeze
        self.mask_fourier_loss = config.mask_fourier_loss

        # loss stuff
        self.apply_loss_unmasked = config.apply_loss_unmasked
        self.loss = instantiate(config.loss)
        if hasattr(self.loss, "reduction") and self.loss.reduction != "none" and not self.apply_loss_unmasked:
            warnings.warn("loss reduction not set to 'none', setting to 'none' in MAE constructor")
            self.loss.reduction = "none"

        self.fourier_loss = instantiate(config.fourier_loss)
        if self.fourier_loss_weight > 0 and self.fourier_loss is None:
            raise ValueError("FourierLoss weight is activated but no fourier_loss was defined in constructor")
        elif self.fourier_loss_weight >= 1:
            raise ValueError("FourierLoss weight is too large to do mixing factor, weight should be < 1")

        if self.mask_fourier_loss and self.apply_loss_unmasked:
            raise ValueError("mask_fourier_loss and apply_loss_unmasked cannot both be True")

        self.patch_size = int(self.encoder.vit_backbone.patch_embed.patch_size[0])

        # projection layer between the encoder and decoder
        self.encoder_decoder_proj = nn.Linear(self.encoder.embed_dim, self.decoder.embed_dim, bias=True)

        self.decoder_pred = nn.Linear(
            self.decoder.embed_dim,
            self.patch_size**2 * (1 if self.encoder.channel_agnostic else self.in_chans),
            bias=True,
        )  # linear layer from decoder embedding to input dims

        # overwrite decoder pos embeddings based on encoder params
        self.decoder.pos_embeddings = generate_2d_sincos_pos_embeddings(  # type: ignore[assignment]
            self.decoder.embed_dim,
            length=self.encoder.vit_backbone.patch_embed.grid_size[0],
            use_class_token=self.encoder.vit_backbone.cls_token is not None,
            num_modality=self.decoder.num_modalities if self.encoder.channel_agnostic else 1,
        )

        if config.use_MAE_weight_init:
            w = self.encoder.vit_backbone.patch_embed.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

            torch.nn.init.normal_(self.encoder.vit_backbone.cls_token, std=0.02)
            torch.nn.init.normal_(self.decoder.mask_token, std=0.02)

            self.apply(self._MAE_init_weights)

    def setup(self, stage: str) -> None:
        super().setup(stage)
        if self.trim_encoder_blocks is not None:
            logging.info(f"Trimming encoder to {self.trim_encoder_blocks} blocks!")
            self.encoder.vit_backbone.blocks = self.encoder.vit_backbone.blocks[: self.trim_encoder_blocks]

    def _MAE_init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def decode_to_reconstruction(
        encoder_latent: torch.Tensor,
        ind_restore: torch.Tensor,
        proj: torch.nn.Module,
        decoder: MAEDecoder | CAMAEDecoder,
        pred: torch.nn.Module,
    ) -> torch.Tensor:
        """Feed forward the encoder latent through the decoders necessary projections and transformations."""
        decoder_latent_projection = proj(encoder_latent)  # projection from encoder.embed_dim to decoder.embed_dim
        decoder_tokens = decoder.forward_masked(decoder_latent_projection, ind_restore)  # decoder.embed_dim output
        predicted_reconstruction = pred(decoder_tokens)  # linear projection to input dim
        return predicted_reconstruction[:, 1:, :]  # drop class token

    def forward(
        self, imgs: torch.Tensor, constant_noise: Union[torch.Tensor, None] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        imgs = self.input_norm(imgs)
        latent, mask, ind_restore = self.encoder.forward_masked(imgs, self.mask_ratio, constant_noise)  # encoder blocks
        reconstruction = self.decode_to_reconstruction(
            latent, ind_restore, self.encoder_decoder_proj, self.decoder, self.decoder_pred
        )
        return latent, reconstruction, mask

    def compute_MAE_loss(
        self,
        reconstruction: torch.Tensor,
        img: torch.Tensor,
        mask: torch.Tensor,
        apply_loss_to_unmasked_tokens: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes final loss and returns specific values of component losses for metric reporting."""
        loss_dict = {}
        img = self.input_norm(img)
        target_flattened = flatten_images(
            img, patch_size=self.patch_size, channel_agnostic=self.encoder.channel_agnostic
        )
        if self.norm_pix_loss:
            target_flattened = apply_norm_pix(target_flattened)

        loss: torch.Tensor = self.loss(
            reconstruction, target_flattened
        )  # should be with MSE or MAE (L1) with reduction='none'
        loss = loss.mean(dim=-1)  # average over embedding dim -> mean loss per patch (N,L)
        if apply_loss_to_unmasked_tokens:
            loss = loss.mean()
        else:
            loss = (loss * mask).sum() / mask.sum()  # mean loss on masked patches only
        loss_dict[self.RECON_LOSS] = loss.item()

        # compute fourier loss
        if self.fourier_loss_weight > 0:
            floss: torch.Tensor = self.fourier_loss(reconstruction, target_flattened)
            if not self.mask_fourier_loss:
                floss = floss.mean()
            else:
                floss = floss.mean(dim=-1)
                floss = (floss * mask).sum() / mask.sum()

            loss_dict[self.FOURIER_LOSS] = floss.item()

        # here we use a mixing factor to keep the loss magnitude appropriate with fourier
        if self.fourier_loss_weight > 0:
            loss = (1 - self.fourier_loss_weight) * loss + (self.fourier_loss_weight * floss)
        return loss, loss_dict

    def compute_unmasked_loss(
        self, reconstruction: torch.Tensor, img: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes final loss and returns specific values of component losses for metric reporting."""
        loss_dict = {}
        target = self.input_norm(img)
        if self.norm_pix_loss:
            target = apply_norm_pix(img)

        reconstruction_ = unflatten_tokens(
            reconstruction,
            patch_size=self.patch_size,
            num_modalities=self.in_chans if self.encoder.channel_agnostic else 1,
        )

        loss: torch.Tensor = self.loss(reconstruction_, target)

        if isinstance(loss, list):
            raise ValueError(
                "Unmasked loss needs to return a tensor."
                "You may need to set reduction to 'mean' or 'sum' in your loss function."
            )

        # Take the mean of the loss if it is not already reduced to make it a scalar
        if loss.ndim > 1:
            loss = loss.mean()
        loss_dict[self.RECON_LOSS] = loss.item()

        # compute fourier loss
        if self.fourier_loss_weight > 0:
            floss: torch.Tensor = self.fourier_loss(reconstruction_, target)
            if floss.ndim > 1:
                floss = floss.mean()

            floss = floss * self.fourier_loss_weight
            loss_dict[self.FOURIER_LOSS] = floss.item()

        # here we use a mixing factor to keep the loss magnitude appropriate with fourier
        if self.fourier_loss_weight > 0:
            loss = loss + floss
        return loss, loss_dict

    def training_step(self, batch: TensorDict, batch_idx: int) -> TensorDict:
        img = batch["pixels"]
        latent, reconstruction, mask = self(img.clone())
        # TODO: support configuration such that some losses are applied masked and some are not.
        if self.apply_loss_unmasked:
            full_loss, loss_dict = self.compute_unmasked_loss(reconstruction, img.float())
        else:
            full_loss, loss_dict = self.compute_MAE_loss(reconstruction, img.float(), mask)
        return {
            "loss": full_loss,
            **loss_dict,  # type: ignore[dict-item]
        }

    def validation_step(self, batch: TensorDict, batch_idx: int) -> TensorDict:
        return self.training_step(batch, batch_idx)

    def update_metrics(self, outputs: TensorDict, batch: TensorDict) -> None:
        self.metrics["lr"].update(value=self.lr_scheduler.get_last_lr())
        for key, value in outputs.items():
            if key.endswith("loss"):
                self.metrics[key].update(value)

    def on_validation_batch_end(  # type: ignore[override]
        self,
        outputs: TensorDict,
        batch: TensorDict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        super().on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)
    
    def save_pretrained(self, save_directory: str, **kwargs):
        filename = kwargs.pop("filename", "model.safetensors")
        modelpath = f"{save_directory}/{filename}"
        self.config.save_pretrained(save_directory)
        torch.save({"state_dict": self.state_dict()}, modelpath)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        filename = kwargs.pop("filename", "model.safetensors")
        from_state_dict = kwargs.pop("from_state_dict", False)

        modelpath = f"{pretrained_model_name_or_path}/{filename}"
        config = MAEConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        state_dict = torch.load(modelpath, map_location="cpu")
        model = cls(config, *model_args, **kwargs)
        model.load_state_dict(state_dict["state_dict"])
        # if from_state_dict:
        #     model.load_state_dict(state_dict["state_dict"])
        # else:
        #     model.load_state_dict(state_dict)
        return model

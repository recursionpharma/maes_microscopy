# © Recursion Pharmaceuticals 2024
import timm.models.vision_transformer as vit
import torch


def generate_2d_sincos_pos_embeddings(
    embedding_dim: int,
    length: int,
    scale: float = 10000.0,
    use_class_token: bool = True,
    num_modality: int = 1,
) -> torch.nn.Parameter:
    """
    Generate 2Dimensional sin/cosine positional embeddings

    Parameters
    ----------
    embedding_dim : int
        embedding dimension used in vit
    length : int
        number of tokens along height or width of image after patching (assuming square)
    scale : float
        scale for sin/cos functions
    use_class_token : bool
        True - add zero vector to be added to class_token, False - no vector added
    num_modality: number of modalities. If 0, a single modality is assumed.
        Otherwise one-hot modality encoding is added and sincos encoding size is appropriately reduced.

    Returns
    -------
    positional_encoding : torch.Tensor
        positional encoding to add to vit patch encodings
        [num_modality*length*length, embedding_dim] or [1+num_modality*length*length, embedding_dim]
        (w/ or w/o cls_token)
    """

    linear_positions = torch.arange(length, dtype=torch.float32)
    height_mesh, width_mesh = torch.meshgrid(
        linear_positions, linear_positions, indexing="ij"
    )
    positional_dim = embedding_dim // 4  # accomodate h and w x cos and sin embeddings
    positional_weights = (
        torch.arange(positional_dim, dtype=torch.float32) / positional_dim
    )
    positional_weights = 1.0 / (scale**positional_weights)

    height_weights = torch.outer(height_mesh.flatten(), positional_weights)
    width_weights = torch.outer(width_mesh.flatten(), positional_weights)

    positional_encoding = torch.cat(
        [
            torch.sin(height_weights),
            torch.cos(height_weights),
            torch.sin(width_weights),
            torch.cos(width_weights),
        ],
        dim=1,
    )[None, :, :]

    # repeat positional encoding for multiple channel modalities
    positional_encoding = positional_encoding.repeat(1, num_modality, 1)

    if use_class_token:
        class_token = torch.zeros([1, 1, embedding_dim], dtype=torch.float32)
        positional_encoding = torch.cat([class_token, positional_encoding], dim=1)

    positional_encoding = torch.nn.Parameter(positional_encoding, requires_grad=False)

    return positional_encoding


class ChannelAgnosticPatchEmbed(vit.PatchEmbed):  # type: ignore[misc]
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        embed_dim: int,
        bias: bool = True,
    ) -> None:
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=1,  # in_chans is used by self.proj, which we override anyway
            embed_dim=embed_dim,
            norm_layer=None,
            flatten=False,
            bias=bias,
        )
        # channel-agnostic MAE has a single projection for all chans
        self.proj = torch.nn.Conv2d(
            1, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_chans = x.shape[1]
        x = torch.stack(
            [self.proj(x[:, i : i + 1]) for i in range(in_chans)], dim=2
        )  # single project for all chans
        x = x.flatten(2).transpose(1, 2)  # BCMHW -> BNC
        return x


class ChannelAgnosticViT(vit.VisionTransformer):  # type: ignore[misc]
    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        # rewrite https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L586
        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))

        # TODO: upgrade timm to get access to register tokens
        # if self.vit_backbone.reg_token is not None:
        #     to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        # MAIN DIFFERENCE with Timm - we DYNAMICALLY ADDING POS EMBEDDINGS based on shape of inputs
        # this supports having CA-MAEs actually be channel-agnostic at inference time
        if self.no_embed_class:
            x = x + self.pos_embed[:, : x.shape[1]]
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + self.pos_embed[:, : x.shape[1]]
        return self.pos_drop(x)  # type: ignore[no-any-return]


def channel_agnostic_vit(
    vit_backbone: vit.VisionTransformer, max_in_chans: int
) -> vit.VisionTransformer:
    # replace patch embedding with channel-agnostic version
    vit_backbone.patch_embed = ChannelAgnosticPatchEmbed(
        img_size=vit_backbone.patch_embed.img_size[0],
        patch_size=vit_backbone.patch_embed.patch_size[0],
        embed_dim=vit_backbone.embed_dim,
    )

    # replace positional embedding with channel-agnostic version
    vit_backbone.pos_embed = generate_2d_sincos_pos_embeddings(
        embedding_dim=vit_backbone.embed_dim,
        length=vit_backbone.patch_embed.grid_size[0],
        use_class_token=vit_backbone.cls_token is not None,
        num_modality=max_in_chans,
    )

    # change the class to be ChannelAgnostic so that it actually uses the new _pos_embed
    vit_backbone.__class__ = ChannelAgnosticViT
    return vit_backbone


def sincos_positional_encoding_vit(
    vit_backbone: vit.VisionTransformer, scale: float = 10000.0
) -> vit.VisionTransformer:
    """Attaches no-grad sin-cos positional embeddings to a pre-constructed ViT backbone model.

    Parameters
    ----------
    vit_backbone : timm.models.vision_transformer.VisionTransformer
        the constructed vision transformer from timm
    scale : float (default 10000.0)
        hyperparameter for sincos positional embeddings, recommend keeping at 10,000

    Returns
    -------
    timm.models.vision_transformer.VisionTransformer
        the same ViT but with fixed no-grad positional encodings to add to vit patch encodings
    """
    # length: number of tokens along height or width of image after patching (assuming square)
    length = (
        vit_backbone.patch_embed.img_size[0] // vit_backbone.patch_embed.patch_size[0]
    )
    pos_embeddings = generate_2d_sincos_pos_embeddings(
        vit_backbone.embed_dim,
        length=length,
        scale=scale,
        use_class_token=vit_backbone.cls_token is not None,
    )
    # note, if the model had weight_init == 'skip', this might get overwritten
    vit_backbone.pos_embed = pos_embeddings
    return vit_backbone


def vit_small_patch16_256(**kwargs):
    default_kwargs = dict(
        img_size=256,
        in_chans=6,
        num_classes=0,
        fc_norm=None,
        class_token=True,
        drop_path_rate=0.1,
        init_values=0.0001,
        block_fn=vit.ParallelScalingBlock,
        qkv_bias=False,
        qk_norm=True,
    )
    for k, v in kwargs.items():
        default_kwargs[k] = v
    return vit.vit_small_patch16_224(**default_kwargs)


def vit_small_patch32_512(**kwargs):
    default_kwargs = dict(
        img_size=512,
        in_chans=6,
        num_classes=0,
        fc_norm=None,
        class_token=True,
        drop_path_rate=0.1,
        init_values=0.0001,
        block_fn=vit.ParallelScalingBlock,
        qkv_bias=False,
        qk_norm=True,
    )
    for k, v in kwargs.items():
        default_kwargs[k] = v
    return vit.vit_small_patch32_384(**default_kwargs)


def vit_base_patch8_256(**kwargs):
    default_kwargs = dict(
        img_size=256,
        in_chans=6,
        num_classes=0,
        fc_norm=None,
        class_token=True,
        drop_path_rate=0.1,
        init_values=0.0001,
        block_fn=vit.ParallelScalingBlock,
        qkv_bias=False,
        qk_norm=True,
    )
    for k, v in kwargs.items():
        default_kwargs[k] = v
    return vit.vit_base_patch8_224(**default_kwargs)


def vit_base_patch16_256(**kwargs):
    default_kwargs = dict(
        img_size=256,
        in_chans=6,
        num_classes=0,
        fc_norm=None,
        class_token=True,
        drop_path_rate=0.1,
        init_values=0.0001,
        block_fn=vit.ParallelScalingBlock,
        qkv_bias=False,
        qk_norm=True,
    )
    for k, v in kwargs.items():
        default_kwargs[k] = v
    return vit.vit_base_patch16_224(**default_kwargs)


def vit_base_patch32_512(**kwargs):
    default_kwargs = dict(
        img_size=512,
        in_chans=6,
        num_classes=0,
        fc_norm=None,
        class_token=True,
        drop_path_rate=0.1,
        init_values=0.0001,
        block_fn=vit.ParallelScalingBlock,
        qkv_bias=False,
        qk_norm=True,
    )
    for k, v in kwargs.items():
        default_kwargs[k] = v
    return vit.vit_base_patch32_384(**default_kwargs)


def vit_large_patch8_256(**kwargs):
    default_kwargs = dict(
        img_size=256,
        in_chans=6,
        num_classes=0,
        fc_norm=None,
        class_token=True,
        patch_size=8,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        drop_path_rate=0.3,
        init_values=0.0001,
        block_fn=vit.ParallelScalingBlock,
        qkv_bias=False,
        qk_norm=True,
    )
    for k, v in kwargs.items():
        default_kwargs[k] = v
    return vit.VisionTransformer(**default_kwargs)


def vit_large_patch16_256(**kwargs):
    default_kwargs = dict(
        img_size=256,
        in_chans=6,
        num_classes=0,
        fc_norm=None,
        class_token=True,
        drop_path_rate=0.3,
        init_values=0.0001,
        block_fn=vit.ParallelScalingBlock,
        qkv_bias=False,
        qk_norm=True,
    )
    for k, v in kwargs.items():
        default_kwargs[k] = v
    return vit.vit_large_patch16_384(**default_kwargs)

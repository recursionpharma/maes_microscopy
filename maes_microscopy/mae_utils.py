# © Recursion Pharmaceuticals 2024
import math

import torch


def flatten_images(
    img: torch.Tensor, patch_size: int, channel_agnostic: bool = False
) -> torch.Tensor:
    """
    Flattens 2D images into tokens with the same pixel values

    Parameters
    ----------
    img : input image tensor (N, C, H, W)

    Returns
    -------
    flattened_img: flattened image tensor (N, L, patch_size**2 * C)
    """

    if (img.shape[2] != img.shape[3]) or (img.shape[2] % patch_size != 0):
        raise ValueError("image H must equal image W and be divisible by patch_size")
    in_chans = img.shape[1]

    h = w = int(img.shape[2] // patch_size)
    x = img.reshape(shape=(img.shape[0], in_chans, h, patch_size, w, patch_size))

    if channel_agnostic:
        x = torch.permute(x, (0, 1, 2, 4, 3, 5))  # NCHPWQ -> NCHWPQ
        x = x.reshape(shape=(img.shape[0], in_chans * h * w, int(patch_size**2)))
    else:
        x = torch.permute(x, (0, 2, 4, 3, 5, 1))  # NCHPWQ -> NHWPQC
        x = x.reshape(shape=(img.shape[0], h * w, int(patch_size**2 * in_chans)))
    return x


def unflatten_tokens(
    tokens: torch.Tensor,
    patch_size: int,
    num_modalities: int = 1,
    channel_agnostic: bool = False,
) -> torch.Tensor:
    """
    Unflattens tokens (N,L,patch_size**2 * C) into image tensor (N,C,H,W) with the pixel values

    Parameters
    ----------
    tokens : input token tensor (N,L,patch_size**2 * C)

    Returns
    -------
    img: image tensor (N,C,H,W)
    """
    if num_modalities > 1 and not channel_agnostic:
        raise ValueError("Multiple modalities requires channel agnostic unflattening.")

    h = w = int(math.sqrt(tokens.shape[1] // num_modalities))
    if h * w != (tokens.shape[1] // num_modalities):
        raise ValueError("sqrt of number of tokens not integer")

    if channel_agnostic:
        x = tokens.reshape(shape=(tokens.shape[0], -1, h, w, patch_size, patch_size))
        x = torch.permute(x, (0, 1, 2, 4, 3, 5))  # NCHWPQ -> NCHPWQ
    else:
        x = tokens.reshape(shape=(tokens.shape[0], h, w, patch_size, patch_size, -1))
        x = torch.permute(x, (0, 5, 1, 3, 2, 4))  # NHWPQC -> NCHPWQ
    img = x.reshape(shape=(x.shape[0], -1, h * patch_size, h * patch_size))

    return img

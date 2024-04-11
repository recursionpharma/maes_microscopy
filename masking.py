# © Recursion Pharmaceuticals 2024
from typing import Tuple, Union

import torch


def transformer_random_masking(
    x: torch.Tensor, mask_ratio: float, constant_noise: Union[torch.Tensor, None] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Random mask patches per sample

    Parameters
    ----------
    x : token tensor (N, L, D)
    mask_ratio: float - ratio of image to mask
    constant_noise: None, if provided should be a tensor of shape (N, L) to produce consistent masks

    Returns
    -------
    x_masked : sub-sampled version of x ( int(mask_ratio * N), L, D)
    mask : binary mask indicated masked tokens (1 where masked) (N, L)
    ind_restore : locations of masked tokens, needed for decoder
    """

    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    # use random noise to generate batch based random masks
    if constant_noise is not None:
        noise = constant_noise
    else:
        noise = torch.rand(N, L, device=x.device)

    shuffled_tokens = torch.argsort(noise, dim=1)  # shuffled index
    ind_restore = torch.argsort(shuffled_tokens, dim=1)  # unshuffled index

    # get masked input
    tokens_to_keep = shuffled_tokens[:, :len_keep]  # keep the first len_keep indices
    x_masked = torch.gather(
        x, dim=1, index=tokens_to_keep.unsqueeze(-1).repeat(1, 1, D)
    )

    # get binary mask used for loss masking: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(
        mask, dim=1, index=ind_restore
    )  # unshuffle to get the binary mask

    return x_masked, mask, ind_restore

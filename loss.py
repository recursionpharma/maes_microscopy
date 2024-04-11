# © Recursion Pharmaceuticals 2024
import torch
import torch.nn as nn


class FourierLoss(nn.Module):
    def __init__(
        self,
        use_l1_loss: bool = True,
        num_multimodal_modalities: int = 1,  # set to 1 for vanilla MAE, 6 for channel-agnostic MAE
    ) -> None:
        """
        Fourier transform loss is only sound when using L1 or L2 loss to compare the frequency domains
        between the images / their radial histograms.

        We will always set `reduction="none"` and enforce that the computation of any reductions from the
        output of this loss be managed by the model under question.
        """
        super().__init__()
        self.loss = (
            nn.L1Loss(reduction="none") if use_l1_loss else nn.MSELoss(reduction="none")
        )
        self.num_modalities = num_multimodal_modalities

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # input = reconstructed image, target = original image
        # flattened images from MAE are (B, H*W, C), so, here we convert to B x C x H x W (note we assume H == W)
        flattened_images = len(input.shape) == len(target.shape) == 3
        if flattened_images:
            B, H_W, C = input.shape
            H_W = H_W // self.num_modalities
            four_d_shape = (B, C * self.num_modalities, int(H_W**0.5), int(H_W**0.5))
            input = input.view(*four_d_shape)
            target = target.view(*four_d_shape)
        else:
            B, C, h, w = input.shape
            H_W = h * w

        if len(input.shape) != len(target.shape) != 4:
            raise ValueError(
                f"Invalid input shape: got {input.shape} and {target.shape}."
            )

        fft_reconstructed = torch.fft.fft2(input)
        fft_original = torch.fft.fft2(target)

        magnitude_reconstructed = torch.abs(fft_reconstructed)
        magnitude_original = torch.abs(fft_original)

        loss_tensor: torch.Tensor = self.loss(
            magnitude_reconstructed, magnitude_original
        )

        if (
            flattened_images and not self.num_bins
        ):  # then output loss should be reshaped
            loss_tensor = loss_tensor.reshape(B, H_W * self.num_modalities, C)

        return loss_tensor

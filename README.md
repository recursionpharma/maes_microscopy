# Masked Autoencoders are Scalable Learners of Cellular Morphology
Official repo for Recursion's accepted spotlight paper at [NeurIPS 2023 Generative AI &amp; Biology workshop](https://openreview.net/group?id=NeurIPS.cc/2023/Workshop/GenBio).

Paper: https://arxiv.org/abs/2309.16064

![vit_diff_mask_ratios](https://github.com/recursionpharma/maes_microscopy/assets/109550980/409ac47b-a7d4-4158-b030-a88234e2b21f)

## Provided code
The baseline Vision Transformer architecture backbone used in this work can be built with the following code snippet from Timm:
```
import timm.models.vision_transformer as vit

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
```

Additional code will be released as the date of the workshop gets closer.

## Provided models
Stay tuned...

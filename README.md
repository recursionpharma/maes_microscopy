[![scorecard-score](https://github.com/recursionpharma/octo-guard-badges/blob/trunk/badges/repo/maes_microscopy/maturity_score.svg?raw=true)](https://infosec-docs.prod.rxrx.io/octoguard/scorecards/maes_microscopy)
[![scorecard-status](https://github.com/recursionpharma/octo-guard-badges/blob/trunk/badges/repo/maes_microscopy/scorecard_status.svg?raw=true)](https://infosec-docs.prod.rxrx.io/octoguard/scorecards/maes_microscopy)
# Masked Autoencoders are Scalable Learners of Cellular Morphology
Official repo for Recursion's two recently accepted papers:
- Spotlight full-length paper at [CVPR 2024](https://cvpr.thecvf.com/Conferences/2024/AcceptedPapers) -- Masked Autoencoders for Microscopy are Scalable Learners of Cellular Biology
  - Paper: https://arxiv.org/abs/2404.10242
  - CVPR poster page with video: https://cvpr.thecvf.com/virtual/2024/poster/31565
- Spotlight workshop paper at [NeurIPS 2023 Generative AI &amp; Biology workshop](https://openreview.net/group?id=NeurIPS.cc/2023/Workshop/GenBio)
  - Paper: https://arxiv.org/abs/2309.16064

![vit_diff_mask_ratios](https://github.com/recursionpharma/maes_microscopy/assets/109550980/c15f46b1-cdb9-41a7-a4af-bdc9684a971d)


## Provided code
See the repo for ingredients required for defining our MAEs. Users seeking to re-implement training will need to stitch together the Encoder and Decoder modules according to their usecase.

Furthermore the baseline Vision Transformer architecture backbone used in this work can be built with the following code snippet from Timm:
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

## Provided models
A publicly available model for research that handles inference and auto-scaling can be found at: https://www.rxrx.ai/phenom

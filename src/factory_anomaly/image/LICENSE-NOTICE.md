# Image module — dataset licensing

This module is designed against two datasets:

## MVTec AD — CC BY-NC-SA 4.0 (non-commercial research only)

- Source: https://www.mvtec.com/company/research/datasets/mvtec-ad
- License: https://www.mvtec.com/company/research/datasets/mvtec-ad/license-mvtec-ad
- This repo **does not** redistribute MVTec AD images.
- This repo **does not** commit trained PatchCore memory banks derived from MVTec AD — they are a derivative of the data and inherit the non-commercial restriction.
- Download is on-demand via `scripts/download_mvtec.py` (operator-initiated); the script writes into `data/mvtec/` which is git-ignored.

If you want to publish results from this module in any commercial context, you must use a different dataset.

## VisA — CC-BY-4.0 (permissive)

- Source: https://github.com/amazon-science/spot-diff
- License: https://creativecommons.org/licenses/by/4.0/
- Used in this module's evaluation reports for cross-dataset generalisation testing (see `docs/evaluation/image-baseline.md` when published).

The permissive license on VisA means trained memory banks from VisA *may* be committed, but the project still treats all trained image-modality artifacts as non-committed for consistency.

## Backbone weights

ResNet50 with `IMAGENET1K_V2` weights from torchvision. License: BSD-3-Clause (torchvision); weights themselves are derived from the ImageNet dataset which has its own usage terms (research use is uncontroversial; commercial deployment of ImageNet-trained models is an open legal question that has nothing to do with this repo).

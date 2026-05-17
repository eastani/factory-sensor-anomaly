"""Locally-aware patch features from a frozen ImageNet ResNet50.

PatchCore relies on mid-level CNN activations as a generic "is this image
patch unusual?" detector. The backbone is **never fine-tuned** — the
hypothesis (validated empirically in the paper) is that ImageNet-trained
features are already discriminative enough that a non-parametric NN search
over normal patches finds defects.

Two ResNet50 layers are used: ``layer2`` and ``layer3``. ``layer1`` is too
low-level (edges, textures); ``layer4`` is too high-level (class-specific
concepts). ``layer2`` and ``layer3`` give a good balance of locality and
semantic content.

The "locally aware" part: each pixel of a feature map is replaced by an
average of its 3x3 neighbourhood. This injects spatial context cheaply
and improves robustness vs. single-pixel descriptors.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812 — PyTorch-standard alias
from torchvision.models import ResNet50_Weights, resnet50

# Layer names recognised by torchvision's ResNet implementation.
_VALID_LAYERS = frozenset({"layer1", "layer2", "layer3", "layer4"})


@dataclass(frozen=True)
class FeatureExtractorConfig:
    """Knobs that affect the output shape and should travel with the model."""

    layers: tuple[str, ...] = ("layer2", "layer3")
    neighborhood: int = 3  # 3x3 avg pool for local-context aggregation
    target_spatial: int = 28  # all layer maps resized to this H=W before concat

    def __post_init__(self) -> None:
        unknown = set(self.layers) - _VALID_LAYERS
        if unknown:
            raise ValueError(
                f"unknown ResNet50 layer name(s): {sorted(unknown)}; "
                f"valid options are {sorted(_VALID_LAYERS)}"
            )
        if self.neighborhood < 1 or self.neighborhood % 2 == 0:
            raise ValueError(
                f"neighborhood must be a positive odd integer, got {self.neighborhood}"
            )
        if self.target_spatial < 1:
            raise ValueError(f"target_spatial must be >= 1, got {self.target_spatial}")


class PatchFeatureExtractor:
    """Frozen ResNet50 with forward hooks on the configured layers.

    Use ``extract(images_bchw)`` to get patch features. Output shape is
    ``(B, H * W, D)`` where ``H = W = config.target_spatial`` and ``D`` is
    the sum of channel counts across the configured layers (e.g. for
    ``layer2`` + ``layer3``, ``D = 512 + 1024 = 1536``).
    """

    def __init__(
        self,
        config: FeatureExtractorConfig | None = None,
        *,
        device: str = "cpu",
    ) -> None:
        self.config = config or FeatureExtractorConfig()
        self.device = torch.device(device)

        # IMAGENET1K_V2 weights — slightly higher accuracy than V1 and the
        # current torchvision default. Pinning the enum makes the artifact
        # provenance unambiguous.
        self._backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self._backbone.eval()
        self._backbone.to(self.device)
        for p in self._backbone.parameters():
            p.requires_grad_(False)

        self._captured: dict[str, torch.Tensor] = {}
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        for name in self.config.layers:
            module = getattr(self._backbone, name)
            handle = module.register_forward_hook(self._make_hook(name))
            self._hook_handles.append(handle)

    def _make_hook(self, name: str) -> object:
        def hook(_module: object, _inputs: object, output: torch.Tensor) -> None:
            self._captured[name] = output

        return hook

    @torch.inference_mode()
    def extract(self, images_bchw: torch.Tensor) -> torch.Tensor:
        """Extract patch features.

        Parameters
        ----------
        images_bchw
            ``(B, 3, H_in, W_in)`` float tensor, ImageNet-normalised.

        Returns
        -------
        torch.Tensor
            ``(B, H * W, D)`` where ``H = W = target_spatial`` and ``D``
            is the concatenated channel count of all configured layers.
        """
        if images_bchw.ndim != 4 or images_bchw.shape[1] != 3:
            raise ValueError(
                f"images_bchw must be (B, 3, H, W); got shape {tuple(images_bchw.shape)}"
            )

        self._captured.clear()
        self._backbone(images_bchw.to(self.device))

        # Apply local averaging + resize each captured map to common spatial size.
        resized: list[torch.Tensor] = []
        target = self.config.target_spatial
        pad = self.config.neighborhood // 2
        for name in self.config.layers:
            feats = self._captured[name]  # (B, C, h, w)
            feats = F.avg_pool2d(feats, kernel_size=self.config.neighborhood, stride=1, padding=pad)
            feats = F.interpolate(
                feats, size=(target, target), mode="bilinear", align_corners=False
            )
            resized.append(feats)

        concat = torch.cat(resized, dim=1)  # (B, D, H, W)
        b, d, h, w = concat.shape
        # (B, D, H, W) -> (B, H*W, D); contiguous for downstream numpy conversion.
        return concat.permute(0, 2, 3, 1).reshape(b, h * w, d).contiguous()

    @property
    def feature_dim(self) -> int:
        """Total channel count of the concatenated output (``D`` in the docstring)."""
        # ResNet50 channel counts per layer.
        channels = {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}
        return sum(channels[name] for name in self.config.layers)

    def close(self) -> None:
        """Remove forward hooks. Idempotent."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def __del__(self) -> None:
        # Best-effort cleanup if the caller forgets ``close()``.
        with contextlib.suppress(Exception):  # pragma: no cover  (defensive in finaliser)
            self.close()

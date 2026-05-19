"""Pure functions backing the Streamlit dashboard's image-anomaly panel.

Kept separate from ``dashboard/app.py`` so they can be unit-tested without
spinning up Streamlit, and so the type-checked production code lives under
``src/``. This module is **deliberately torch-free** — the dashboard
container runs from the sensor image (no PyTorch) and hits the image API
over HTTP. All work here is pure matplotlib + numpy + PIL on small arrays.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import matplotlib

# Use the non-interactive backend explicitly. Streamlit imports matplotlib
# lazily; pinning Agg here avoids the macOS GUI backend being selected when
# the test suite (or CI) imports this module without a display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from PIL import Image

DEMO_IMAGES_ROOT = Path("docs/assets/demo-images/test")


@dataclass(frozen=True)
class DemoImage:
    label: str
    path: Path

    def read_bytes(self) -> bytes:
        return self.path.read_bytes()


def list_demo_images(root: Path = DEMO_IMAGES_ROOT) -> list[DemoImage]:
    """Enumerate the bundled synthetic demo images, label-sorted.

    Returns an empty list if ``root`` is missing — the dashboard then
    falls back to upload-only mode without erroring.
    """
    if not root.exists():
        return []
    labels: dict[str, str] = {
        "clean.png": "Clean plate (no defect)",
        "defect_scratch.png": "Defect — diagonal scratch",
        "defect_missing.png": "Defect — missing feature",
    }
    out: list[DemoImage] = []
    for p in sorted(root.glob("*.png")):
        out.append(DemoImage(label=labels.get(p.name, p.name), path=p))
    return out


def downsample_for_upload(raw: bytes, *, max_size: int = 512) -> bytes:
    """Resize an uploaded image to keep upload payload small.

    The server resamples to 224x224 anyway; sending a 4-megapixel PNG just
    wastes bandwidth and operator-laptop CPU on the encode/decode trip.
    Preserves aspect ratio and re-encodes as PNG (so JPEG-uploaded images
    don't suffer a second lossy pass).

    Returns the original bytes unchanged when the image already fits.
    """
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    if max(img.size) <= max_size:
        # Re-encode anyway to normalise on PNG and to confirm decodability.
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    scale = max_size / max(img.size)
    new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
    resized = img.resize(new_size, Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    resized.save(buf, format="PNG")
    return buf.getvalue()


def compose_heatmap_figure(
    image_bytes: bytes,
    anomaly_map: np.ndarray,
    score: float,
    *,
    model_version: str = "",
) -> Figure:
    """Three-panel matplotlib figure: input | heatmap | overlay.

    Parameters
    ----------
    image_bytes
        The original uploaded image (any size). Resized to a square for
        display so the overlay aligns with the per-patch heatmap.
    anomaly_map
        ``(H, W)`` float array of per-patch anomaly scores.
    score
        Scalar image-level score (max of ``anomaly_map``). Shown in the title.
    model_version
        Optional model_version string to surface in the title — recruiters
        like seeing it.
    """
    if anomaly_map.ndim != 2:
        raise ValueError(f"anomaly_map must be 2-D, got shape {anomaly_map.shape}")

    display_size = 224
    pil = (
        Image.open(io.BytesIO(image_bytes))
        .convert("RGB")
        .resize((display_size, display_size), Image.Resampling.BILINEAR)
    )
    rgb = np.asarray(pil)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))
    axes[0].imshow(rgb)
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(anomaly_map, cmap="viridis")
    axes[1].set_title("Anomaly map (raw patches)")
    axes[1].axis("off")

    axes[2].imshow(rgb)
    axes[2].imshow(
        anomaly_map,
        cmap="hot",
        alpha=0.55,
        extent=(0, display_size, display_size, 0),
        interpolation="bilinear",
    )
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    title = f"score = {score:.3f}"
    if model_version:
        title = f"{title}    (model: {model_version})"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def is_stub_model(model_version: str | None) -> bool:
    """Heuristic: is the loaded bank a stub (built into the image)?

    Phase 3.4's Dockerfile.image tags stub banks ``stub-demo-<timestamp>``
    and earlier ``stub-baked-<timestamp>`` / ``stub-noise``. A real
    operator-trained bank is named ``mvtec-<category>-v1`` (or similar).
    The dashboard uses this to display an informative banner.
    """
    if not model_version:
        return False
    return model_version.startswith("stub-")

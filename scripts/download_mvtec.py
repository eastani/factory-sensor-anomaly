"""Download MVTec AD categories with explicit license acknowledgement.

The MVTec AD dataset is published under **CC BY-NC-SA 4.0** — research /
non-commercial use only. See ADR-0006 and `src/factory_anomaly/image/LICENSE-NOTICE.md`.

This script never automatically accepts the license. Running it without
``--accept-license`` prints the license URL and exits with code 2; the
operator has to read the terms and re-run with the flag.

Honest note on mirror availability (verified 2026-05-17)
--------------------------------------------------------

The default ``--url-template`` follows the per-category pattern referenced
in older academic code and **currently returns HTTP 404** against the live
``mvtec.com`` site. MVTec's own download page requires a form submission
and yields a single ~5 GB archive with all categories — not per-category
URLs. We deliberately keep the broken URL as the default because:

1. We will not bake unofficial mirrors into the script — those redistribute
   MVTec data without explicit permission and their lifetime is not ours
   to guarantee.
2. The script still serves its purpose: license gate, safe extraction,
   ``--url-template`` override for operators with a working source.

If you need data, the realistic paths are:

- **Official:** submit the MVTec form at
  https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads ,
  then drop the resulting per-category ``.tar.xz`` files into
  ``data/mvtec/_archives/`` and re-run with ``--accept-license`` (the
  script's download step will skip files that already exist).
- **Operator-trusted mirror:** point ``--url-template`` at a URL pattern
  you trust, with ``{category}`` as the placeholder.

See ``docs/evaluation/image-baseline.md`` for the documented findings of
the 2026-05-17 evaluation run.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path
from typing import Final

import httpx

# All 15 MVTec AD categories. Listed for ``--list`` / validation; we do not
# attempt to validate against a remote catalogue.
ALL_CATEGORIES: Final[tuple[str, ...]] = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)

LICENSE_URL: Final[str] = (
    "https://www.mvtec.com/company/research/datasets/mvtec-ad/license-mvtec-ad"
)
MANUAL_DOWNLOAD_URL: Final[str] = (
    "https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads"
)

# Primary mirror. Per-category tar.xz layout used by many academic projects.
DEFAULT_URL_TEMPLATE: Final[str] = (
    "https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/"
    "research-datasets/mvtec_ad/{category}.tar.xz"
)


class DownloadError(RuntimeError):
    """Raised when the per-category download cannot be completed."""


def _print_license_notice() -> None:
    print(
        "\n"
        "MVTec AD is licensed under CC BY-NC-SA 4.0 (non-commercial research only).\n"
        "Read the full license terms before downloading:\n\n"
        f"  {LICENSE_URL}\n\n"
        "Re-run with --accept-license to acknowledge and proceed.\n",
        file=sys.stderr,
    )


def _validate_categories(requested: list[str]) -> list[str]:
    invalid = [c for c in requested if c not in ALL_CATEGORIES]
    if invalid:
        raise ValueError(
            f"unknown MVTec category/categories: {invalid}; "
            f"valid options are {list(ALL_CATEGORIES)}"
        )
    return requested


def download_category(
    category: str,
    out_dir: Path,
    *,
    url_template: str = DEFAULT_URL_TEMPLATE,
    chunk_size: int = 1024 * 256,
    timeout: float = 300.0,
) -> Path:
    """Fetch one category's tar.xz archive into ``out_dir``.

    Returns the path to the downloaded archive (still compressed). Does
    not extract — see :func:`extract_archive`.

    Raises
    ------
    DownloadError
        If the HTTP request fails or returns non-2xx.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    url = url_template.format(category=category)
    target = out_dir / f"{category}.tar.xz"

    if target.exists() and target.stat().st_size > 0:
        print(f"[skip] {target} already exists ({target.stat().st_size:,} bytes)")
        return target

    print(f"[download] {url} -> {target}")
    try:
        with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as response:
            if response.status_code != 200:
                raise DownloadError(
                    f"HTTP {response.status_code} fetching {url}; "
                    f"if this URL is stale, download manually from {MANUAL_DOWNLOAD_URL}"
                )
            with target.open("wb") as fp:
                for chunk in response.iter_bytes(chunk_size=chunk_size):
                    fp.write(chunk)
    except httpx.HTTPError as exc:
        raise DownloadError(
            f"network error fetching {url}: {exc}; "
            f"download manually from {MANUAL_DOWNLOAD_URL} and place {category}.tar.xz in {out_dir}"
        ) from exc

    return target


def extract_archive(archive_path: Path, dest_root: Path) -> Path:
    """Extract a MVTec category archive into ``dest_root/<category>/``.

    Returns the per-category directory. Idempotent: if the directory exists
    and contains the expected ``train/good`` subfolder, extraction is skipped.
    """
    category = archive_path.name.removesuffix(".tar.xz")
    dest_root.mkdir(parents=True, exist_ok=True)
    category_dir = dest_root / category

    if (category_dir / "train" / "good").exists():
        print(f"[skip] {category_dir} already extracted")
        return category_dir

    print(f"[extract] {archive_path} -> {dest_root}")
    with tarfile.open(archive_path, mode="r:xz") as tar:
        # Safe extraction: reject absolute paths and parent-relative traversal.
        # Also pass filter="data" so tarfile applies its own safety pass and
        # silences the Python 3.14 deprecation warning.
        for member in tar.getmembers():
            target = (dest_root / member.name).resolve()
            if not str(target).startswith(str(dest_root.resolve())):
                raise RuntimeError(f"unsafe path in archive: {member.name}")
        tar.extractall(dest_root, filter="data")

    if not (category_dir / "train" / "good").exists():
        raise RuntimeError(
            f"archive {archive_path} extracted without expected train/good directory"
        )
    return category_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Download MVTec AD categories.")
    parser.add_argument(
        "--category",
        action="append",
        default=[],
        help="category name (repeat for multiple). Default: bottle.",
    )
    parser.add_argument("--out", type=Path, default=Path("data/mvtec"), help="extraction root")
    parser.add_argument(
        "--archives", type=Path, default=Path("data/mvtec/_archives"), help="archive cache"
    )
    parser.add_argument(
        "--accept-license",
        action="store_true",
        help="acknowledge the MVTec AD non-commercial license terms",
    )
    parser.add_argument("--list", action="store_true", help="list valid category names and exit")
    parser.add_argument(
        "--url-template",
        default=DEFAULT_URL_TEMPLATE,
        help="override the per-category URL template (advanced)",
    )
    args = parser.parse_args()

    if args.list:
        for cat in ALL_CATEGORIES:
            print(cat)
        return 0

    if not args.accept_license:
        _print_license_notice()
        return 2

    categories = args.category or ["bottle"]
    try:
        categories = _validate_categories(categories)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    failures: list[tuple[str, str]] = []
    for cat in categories:
        try:
            archive = download_category(cat, args.archives, url_template=args.url_template)
            extract_archive(archive, args.out)
        except DownloadError as exc:
            failures.append((cat, str(exc)))
            print(f"[fail] {cat}: {exc}", file=sys.stderr)
        except RuntimeError as exc:
            failures.append((cat, str(exc)))
            print(f"[fail] {cat}: {exc}", file=sys.stderr)

    if failures:
        print(
            f"\n{len(failures)}/{len(categories)} categories failed. "
            f"Manual download from: {MANUAL_DOWNLOAD_URL}",
            file=sys.stderr,
        )
        return 3

    print(f"\nDone. Extracted into {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

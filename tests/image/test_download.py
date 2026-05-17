"""Tests for the MVTec download script.

Real downloads are never exercised in tests — the EULA precludes it and
the archives are large. We verify the license-gate, argument parsing, and
the extraction helper against a synthetic tar.xz.
"""

from __future__ import annotations

import sys
import tarfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import download_mvtec  # noqa: E402


def test_license_gate_exits_with_code_2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["download_mvtec.py", "--category", "bottle"])
    rc = download_mvtec.main()
    assert rc == 2


def test_list_categories_exits_zero(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["download_mvtec.py", "--list"])
    rc = download_mvtec.main()
    captured = capsys.readouterr()
    assert rc == 0
    listed = captured.out.strip().splitlines()
    assert "bottle" in listed
    assert len(listed) == len(download_mvtec.ALL_CATEGORIES)


def test_unknown_category_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["download_mvtec.py", "--accept-license", "--category", "not-a-real-category"],
    )
    rc = download_mvtec.main()
    assert rc == 1


def test_extract_archive_round_trip(tmp_path: Path) -> None:
    """Build a tiny tar.xz mimicking MVTec's category layout, then extract."""
    cat = "widget"
    staging = tmp_path / "staging" / cat / "train" / "good"
    staging.mkdir(parents=True)
    (staging / "001.png").write_bytes(b"fake-png-bytes")

    archive_path = tmp_path / f"{cat}.tar.xz"
    with tarfile.open(archive_path, "w:xz") as tar:
        tar.add(tmp_path / "staging" / cat, arcname=cat)

    dest_root = tmp_path / "out"
    result = download_mvtec.extract_archive(archive_path, dest_root)
    assert result == dest_root / cat
    assert (result / "train" / "good" / "001.png").read_bytes() == b"fake-png-bytes"


def test_extract_archive_is_idempotent(tmp_path: Path) -> None:
    cat = "widget"
    staging = tmp_path / "staging" / cat / "train" / "good"
    staging.mkdir(parents=True)
    (staging / "001.png").write_bytes(b"x")
    archive_path = tmp_path / f"{cat}.tar.xz"
    with tarfile.open(archive_path, "w:xz") as tar:
        tar.add(tmp_path / "staging" / cat, arcname=cat)

    dest_root = tmp_path / "out"
    download_mvtec.extract_archive(archive_path, dest_root)
    # Second call should be a no-op (skip print, no error).
    download_mvtec.extract_archive(archive_path, dest_root)
    assert (dest_root / cat / "train" / "good" / "001.png").exists()


def test_extract_archive_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "bad.tar.xz"
    target = tmp_path / "evil.txt"
    target.write_text("payload")
    with tarfile.open(archive_path, "w:xz") as tar:
        # Member name escapes the destination.
        tar.add(target, arcname="../escape.txt")

    dest_root = tmp_path / "out"
    dest_root.mkdir()
    with pytest.raises(RuntimeError, match="unsafe path"):
        download_mvtec.extract_archive(archive_path, dest_root)

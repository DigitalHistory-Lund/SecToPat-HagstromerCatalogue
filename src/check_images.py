#!/usr/bin/env python3
"""Scan extracted_images/ and extracted_cards/ for corrupt images.

Checks (fast to slow):
  1. Zero-byte files
  2. Truncated PNGs (missing IEND trailer)
  3. Unreadable by OpenCV

Usage:
    python -m src.check_images
"""

import sys
from pathlib import Path

import cv2

from .config import load_config

PNG_IEND = b"\x49\x45\x4e\x44\xae\x42\x60\x82"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def check_file(path: Path) -> str | None:
    """Return a reason string if the image is bad, or None if OK."""
    size = path.stat().st_size
    if size == 0:
        return "zero-byte file"

    if path.suffix.lower() == ".png" and size >= 8:
        with open(path, "rb") as f:
            f.seek(-8, 2)
            if f.read(8) != PNG_IEND:
                return "truncated PNG (missing IEND)"

    if cv2.imread(str(path)) is None:
        return "unreadable by OpenCV"

    return None


def scan_directory(directory: Path) -> list[tuple[Path, str]]:
    """Return list of (path, reason) for bad images in *directory*."""
    if not directory.is_dir():
        return []
    problems: list[tuple[Path, str]] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        reason = check_file(path)
        if reason:
            problems.append((path, reason))
    return problems


def main() -> int:
    cfg = load_config()
    dirs = [cfg.extracted_images_dir, cfg.extracted_cards_dir]

    all_problems: list[tuple[Path, str]] = []
    for d in dirs:
        print(f"Scanning {d} …")
        problems = scan_directory(d)
        all_problems.extend(problems)

    if all_problems:
        print(f"\n{len(all_problems)} problem(s) found:\n")
        for path, reason in all_problems:
            print(f"  {path}  — {reason}")
        return 1

    print("\nAll clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

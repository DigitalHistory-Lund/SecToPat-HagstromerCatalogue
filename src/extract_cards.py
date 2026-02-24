"""Detect and extract individual catalogue cards from page images using OpenCV."""

import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .config import Config, load_config


def _classify_background(gray: np.ndarray) -> str:
    """Classify page background as 'light' or 'dark' by sampling border pixels."""
    h, w = gray.shape
    border = 20
    samples = np.concatenate([
        gray[:border, :].ravel(),       # top
        gray[-border:, :].ravel(),      # bottom
        gray[:, :border].ravel(),       # left
        gray[:, -border:].ravel(),      # right
    ])
    return "dark" if np.median(samples) < 128 else "light"


def _find_card_contours_light(gray: np.ndarray, debug_dir: Path | None = None) -> list:
    """Find card contours on a light background using Canny + dilation."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)

    if debug_dir:
        cv2.imwrite(str(debug_dir / "01_edges.png"), edges)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    dilated = cv2.dilate(edges, kernel, iterations=3)

    if debug_dir:
        cv2.imwrite(str(debug_dir / "02_dilated.png"), dilated)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def _find_card_contours_dark(gray: np.ndarray, debug_dir: Path | None = None) -> list:
    """Find card contours on a dark background using binary threshold."""
    _, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)

    if debug_dir:
        cv2.imwrite(str(debug_dir / "01_thresh.png"), thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    if debug_dir:
        cv2.imwrite(str(debug_dir / "02_closed.png"), closed)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def _filter_and_sort_contours(
    contours: list,
    img_h: int,
    img_w: int,
    debug_dir: Path | None = None,
    debug_img: np.ndarray | None = None,
) -> list[tuple[int, int, int, int, int, int]]:
    """Filter contours by size/shape and assign (col, row) grid coordinates.

    Returns list of (x, y, w, h, col, row) sorted by row then col.
    """
    img_area = img_h * img_w
    min_area_ratio = 0.02
    max_area_ratio = 0.40
    min_dim = 200
    min_aspect = 1.0
    max_aspect = 3.0
    min_rectangularity = 0.60

    candidates: list[tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        area_ratio = area / img_area

        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue
        if w < min_dim or h < min_dim:
            continue

        aspect = w / h if h > 0 else 0
        if aspect < min_aspect or aspect > max_aspect:
            continue

        cnt_area = cv2.contourArea(cnt)
        rectangularity = cnt_area / area if area > 0 else 0
        if rectangularity < min_rectangularity:
            continue

        candidates.append((x, y, w, h))

    if not candidates:
        return []

    if debug_dir and debug_img is not None:
        vis = debug_img.copy()
        for x, y, w, h in candidates:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 4)
        cv2.imwrite(str(debug_dir / "03_candidates.png"), vis)

    # Cluster into grid rows by y-center proximity
    y_centers = [y + h // 2 for _, y, _, h in candidates]
    row_threshold = img_h * 0.05  # 5% of image height

    # Sort by y-center to cluster rows
    indexed = sorted(enumerate(candidates), key=lambda t: y_centers[t[0]])
    rows: list[list[int]] = []
    current_row: list[int] = [indexed[0][0]]
    current_y = y_centers[indexed[0][0]]

    for idx, _ in indexed[1:]:
        if abs(y_centers[idx] - current_y) < row_threshold:
            current_row.append(idx)
        else:
            rows.append(current_row)
            current_row = [idx]
            current_y = y_centers[idx]
    rows.append(current_row)

    # Sort each row by x, assign (col, row) coordinates
    result: list[tuple[int, int, int, int, int, int]] = []
    for row_idx, row_members in enumerate(rows):
        row_members.sort(key=lambda i: candidates[i][0])
        for col_idx, member_idx in enumerate(row_members):
            x, y, w, h = candidates[member_idx]
            result.append((x, y, w, h, col_idx, row_idx))

    if debug_dir and debug_img is not None:
        vis = debug_img.copy()
        for x, y, w, h, col, row in result:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 4)
            label = f"({col},{row})"
            cv2.putText(vis, label, (x + 10, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.imwrite(str(debug_dir / "04_grid.png"), vis)

    return result


def extract_cards_from_page(
    page_path: Path,
    config: Config,
    force: bool = False,
    max_cards: int | None = None,
    debug_dir: Path | None = None,
) -> list[Path]:
    """Detect and extract cards from a single page image.

    Returns list of output card paths.
    """
    config.extracted_cards_dir.mkdir(parents=True, exist_ok=True)

    # Parse volume and page number from filename like "01_0005.png"
    stem = page_path.stem
    volume, page_str = stem.split("_", 1)
    page_nr = int(page_str)

    img = cv2.imread(str(page_path))
    if img is None:
        print(f"  WARNING: Could not read {page_path}")
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    bg = _classify_background(gray)

    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / "00_gray.png"), gray)

    if bg == "dark":
        contours = _find_card_contours_dark(gray, debug_dir)
    else:
        contours = _find_card_contours_light(gray, debug_dir)

    cards = _filter_and_sort_contours(contours, h, w, debug_dir, img)

    if max_cards is not None:
        cards = cards[:max_cards]

    output_paths: list[Path] = []
    for x, y, cw, ch, col, row in cards:
        out_name = f"{volume}_{page_nr:04d}_{col}_{row}.png"
        out_path = config.extracted_cards_dir / out_name

        if out_path.exists() and not force:
            output_paths.append(out_path)
            continue

        card_img = img[y : y + ch, x : x + cw]
        cv2.imwrite(str(out_path), card_img)
        print(f"  {out_name}")
        output_paths.append(out_path)

    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract cards from page images")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--debug", action="store_true", help="Save intermediate images for debugging")
    args = parser.parse_args()

    config = load_config()
    page_images = sorted(config.extracted_images_dir.glob("*.png"))

    if not page_images:
        print("No page images found. Run extract_pages first.")
        return

    debug_base = None
    if args.debug:
        debug_base = Path(tempfile.mkdtemp(prefix="card_debug_"))
        print(f"Debug images will be saved to: {debug_base}")

    for page_path in page_images:
        print(f"Processing {page_path.name}...")
        debug_dir = debug_base / page_path.stem if debug_base else None
        extract_cards_from_page(page_path, config, force=args.force, debug_dir=debug_dir)


if __name__ == "__main__":
    main()

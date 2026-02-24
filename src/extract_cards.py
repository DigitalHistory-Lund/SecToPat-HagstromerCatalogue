"""Detect and extract individual catalogue cards from page images using OpenCV."""

import argparse
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from tqdm import tqdm

from .config import Config, load_config


def _classify_background(gray: np.ndarray, config: Config) -> str:
    """Classify page background as 'light' or 'dark' by sampling border pixels."""
    h, w = gray.shape
    border = config.cv_bg_border
    samples = np.concatenate([
        gray[:border, :].ravel(),       # top
        gray[-border:, :].ravel(),      # bottom
        gray[:, :border].ravel(),       # left
        gray[:, -border:].ravel(),      # right
    ])
    return "dark" if np.median(samples) < config.cv_bg_threshold else "light"


def _find_card_contours_light(gray: np.ndarray, config: Config, debug_dir: Path | None = None) -> list:
    """Find card contours on a light background using Canny + dilation."""
    k = config.cv_light_blur_size
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    edges = cv2.Canny(blurred, config.cv_light_canny_low, config.cv_light_canny_high)

    if debug_dir:
        cv2.imwrite(str(debug_dir / "01_edges.png"), edges)

    dk = config.cv_light_dilate_size
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dk, dk))
    dilated = cv2.dilate(edges, kernel, iterations=config.cv_light_dilate_iter)

    if debug_dir:
        cv2.imwrite(str(debug_dir / "02_dilated.png"), dilated)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def _find_card_contours_dark(gray: np.ndarray, config: Config, debug_dir: Path | None = None) -> list:
    """Find card contours on a dark background using binary threshold."""
    _, thresh = cv2.threshold(gray, config.cv_dark_thresh_value, 255, cv2.THRESH_BINARY)

    if debug_dir:
        cv2.imwrite(str(debug_dir / "01_thresh.png"), thresh)

    ck = config.cv_dark_close_size
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ck, ck))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=config.cv_dark_close_iter)

    if debug_dir:
        cv2.imwrite(str(debug_dir / "02_closed.png"), closed)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def _filter_and_sort_contours(
    contours: list,
    img_h: int,
    img_w: int,
    config: Config,
    debug_dir: Path | None = None,
    debug_img: np.ndarray | None = None,
) -> list[tuple[int, int, int, int, int, int]]:
    """Filter contours by size/shape and assign (col, row) grid coordinates.

    Returns list of (x, y, w, h, col, row) sorted by row then col.
    """
    img_area = img_h * img_w

    candidates: list[tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        area_ratio = area / img_area

        if area_ratio < config.cv_min_area_ratio or area_ratio > config.cv_max_area_ratio:
            continue
        if w < config.cv_min_dim or h < config.cv_min_dim:
            continue

        aspect = w / h if h > 0 else 0
        if aspect < config.cv_min_aspect or aspect > config.cv_max_aspect:
            continue

        cnt_area = cv2.contourArea(cnt)
        rectangularity = cnt_area / area if area > 0 else 0
        if rectangularity < config.cv_min_rectangularity:
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
    row_threshold = img_h * config.cv_row_threshold

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

    # Fast path: if boxes JSON exists and all card PNGs exist, skip OpenCV
    if not force:
        boxes_path = config.extracted_cards_dir / f"{volume}_{page_nr:04d}_boxes.json"
        if boxes_path.exists():
            boxes = json.loads(boxes_path.read_text(encoding="utf-8"))
            card_paths = [config.extracted_cards_dir / f"{s}.png" for s in boxes]
            if all(p.exists() for p in card_paths):
                return card_paths[:max_cards] if max_cards is not None else card_paths

    img = cv2.imread(str(page_path))
    if img is None:
        print(f"  WARNING: Could not read {page_path}")
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    bg = _classify_background(gray, config)

    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / "00_gray.png"), gray)

    if bg == "dark":
        contours = _find_card_contours_dark(gray, config, debug_dir)
    else:
        contours = _find_card_contours_light(gray, config, debug_dir)

    cards = _filter_and_sort_contours(contours, h, w, config, debug_dir, img)

    if max_cards is not None:
        cards = cards[:max_cards]

    output_paths: list[Path] = []
    boxes: dict[str, list[int]] = {}
    all_exist = True
    for x, y, cw, ch, col, row in cards:
        card_stem = f"{volume}_{page_nr:04d}_{col}_{row}"
        out_path = config.extracted_cards_dir / f"{card_stem}.png"
        boxes[card_stem] = [x, y, cw, ch]

        if out_path.exists() and not force:
            output_paths.append(out_path)
            continue

        all_exist = False
        card_img = img[y : y + ch, x : x + cw]
        cv2.imwrite(str(out_path), card_img)
        output_paths.append(out_path)

    # Write boxes sidecar JSON
    if boxes:
        boxes_path = config.extracted_cards_dir / f"{volume}_{page_nr:04d}_boxes.json"
        if force or not all_exist or not boxes_path.exists():
            boxes_path.write_text(json.dumps(boxes, indent=2), encoding="utf-8")

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

    with tqdm(page_images, unit="page") as bar:
        for page_path in bar:
            bar.desc = f"Cards {page_path.stem}"
            debug_dir = debug_base / page_path.stem if debug_base else None
            extract_cards_from_page(page_path, config, force=args.force, debug_dir=debug_dir)


if __name__ == "__main__":
    main()

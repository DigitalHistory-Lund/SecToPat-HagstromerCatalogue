"""Detect and extract catalogue cards from page images using OpenCV."""

import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .config import Config, load_config


def _classify_background(gray: np.ndarray, config: Config) -> str:
    """Classify background as 'light' or 'dark' via border pixels."""
    h, w = gray.shape
    border = config.cv_bg_border
    samples = np.concatenate(
        [
            gray[:border, :].ravel(),  # top
            gray[-border:, :].ravel(),  # bottom
            gray[:, :border].ravel(),  # left
            gray[:, -border:].ravel(),  # right
        ]
    )
    return "dark" if np.median(samples) < config.cv_bg_threshold else "light"


def _find_card_contours_light(
    gray: np.ndarray, config: Config, debug_dir: Path | None = None
) -> list:
    """Find card contours on a light background using Canny + dilation."""
    k = config.cv_light_blur_size
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    edges = cv2.Canny(
        blurred, config.cv_light_canny_low, config.cv_light_canny_high
    )

    if debug_dir:
        cv2.imwrite(str(debug_dir / "01_edges.png"), edges)

    dk = config.cv_light_dilate_size
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dk, dk))
    dilated = cv2.dilate(edges, kernel, iterations=config.cv_light_dilate_iter)

    if debug_dir:
        cv2.imwrite(str(debug_dir / "02_dilated.png"), dilated)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def _find_card_contours_dark(
    gray: np.ndarray, config: Config, debug_dir: Path | None = None
) -> list:
    """Find card contours on a dark background using binary threshold."""
    _, thresh = cv2.threshold(
        gray, config.cv_dark_thresh_value, 255, cv2.THRESH_BINARY
    )

    if debug_dir:
        cv2.imwrite(str(debug_dir / "01_thresh.png"), thresh)

    ck = config.cv_dark_close_size
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ck, ck))
    closed = cv2.morphologyEx(
        thresh, cv2.MORPH_CLOSE, kernel, iterations=config.cv_dark_close_iter
    )

    if debug_dir:
        cv2.imwrite(str(debug_dir / "02_closed.png"), closed)

    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
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

        if (
            area_ratio < config.cv_min_area_ratio
            or area_ratio > config.cv_max_area_ratio
        ):
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

    # Assign absolute (col, row) coordinates based on position in image
    result: list[tuple[int, int, int, int, int, int]] = []
    for row_members in rows:
        mean_y = int(np.mean([y_centers[i] for i in row_members]))
        abs_row = min(int(mean_y / (img_h / 4)), 3)
        row_members.sort(key=lambda i: candidates[i][0])
        for member_idx in row_members:
            x, y, w, h = candidates[member_idx]
            x_center = x + w // 2
            col = 0 if x_center < img_w / 2 else 1
            result.append((x, y, w, h, col, abs_row))

    if debug_dir and debug_img is not None:
        vis = debug_img.copy()
        for x, y, w, h, col, row in result:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 4)
            label = f"({col},{row})"
            cv2.putText(
                vis,
                label,
                (x + 10, y + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                3,
            )
        cv2.imwrite(str(debug_dir / "04_grid.png"), vis)

    return result


def _boxes_overlap(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> bool:
    """Return True if two (x, y, w, h) boxes have any pixel overlap."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by


def _recover_missing_cards(
    cards: list[tuple[int, int, int, int, int, int]],
    gray: np.ndarray,
    img_h: int,
    img_w: int,
    config: Config,
    debug_dir: Path | None = None,
    debug_img: np.ndarray | None = None,
) -> list[tuple[int, int, int, int, int, int]]:
    """Recover cards missed by contour detection using known 2×4 grid geometry.

    Builds a standard set of 8 expected boxes from detected card positions.
    For each standard box that has ZERO overlap with any detected card,
    checks pixel variance to confirm a card is present, then adds it.
    """
    if not cards:
        return cards

    # Median card dimensions from detections
    med_w = int(np.median([cw for _, _, cw, _, _, _ in cards]))
    med_h = int(np.median([ch for _, _, _, ch, _, _ in cards]))

    # Per-column x-positions (median x of detected cards in each column)
    col_xs: dict[int, list[int]] = {0: [], 1: []}
    for x, _, _, _, col, _ in cards:
        col_xs[col].append(x)

    col_x_pos: dict[int, int] = {}
    for col in (0, 1):
        if col_xs[col]:
            col_x_pos[col] = int(np.median(col_xs[col]))

    # Per-row y-positions (median y of detected cards in each row)
    row_ys: dict[int, list[int]] = {r: [] for r in range(4)}
    for _, y, _, _, _, row in cards:
        row_ys[row].append(y)

    row_y_pos: dict[int, int] = {}
    for row in range(4):
        if row_ys[row]:
            row_y_pos[row] = int(np.median(row_ys[row]))

    # Extrapolate missing row positions using median step between known rows
    known_rows = sorted(row_y_pos.keys())
    if len(known_rows) >= 2:
        steps = []
        for i in range(len(known_rows) - 1):
            dy = row_y_pos[known_rows[i + 1]] - row_y_pos[known_rows[i]]
            row_gap = known_rows[i + 1] - known_rows[i]
            steps.append(dy / row_gap)
        med_step = int(np.median(steps))

        for row in range(4):
            if row not in row_y_pos:
                nearest = min(known_rows, key=lambda r: abs(r - row))
                row_y_pos[row] = row_y_pos[nearest] + med_step * (
                    row - nearest
                )

    # Extrapolate missing column positions
    if len(col_x_pos) == 1:
        known_col = next(iter(col_x_pos))
        missing_col = 1 - known_col
        if known_col == 0:
            col_x_pos[missing_col] = (
                col_x_pos[known_col] + med_w + (img_w - 2 * med_w) // 3
            )
        else:
            col_x_pos[missing_col] = (
                col_x_pos[known_col] - med_w - (img_w - 2 * med_w) // 3
            )

    # Build the 8 standard boxes
    standard_boxes: list[tuple[int, int, int, int, int, int]] = []
    for col in (0, 1):
        if col not in col_x_pos:
            continue
        for row in range(4):
            if row not in row_y_pos:
                continue
            sx = max(0, col_x_pos[col])
            sy = max(0, row_y_pos[row])
            sw = min(med_w, img_w - sx)
            sh = min(med_h, img_h - sy)
            standard_boxes.append((sx, sy, sw, sh, col, row))

    # For each standard box, check overlap against every detected card
    detected_rects = [(x, y, cw, ch) for x, y, cw, ch, _, _ in cards]

    recovered: list[tuple[int, int, int, int, int, int]] = []
    for sx, sy, sw, sh, col, row in standard_boxes:
        std_rect = (sx, sy, sw, sh)
        has_overlap = any(
            _boxes_overlap(std_rect, det) for det in detected_rects
        )
        if has_overlap:
            continue

        # No detected card overlaps this slot — check variance
        region = gray[sy : sy + sh, sx : sx + sw]
        if region.size == 0:
            continue

        variance = float(np.var(region))
        if variance >= config.cv_recovery_min_variance:
            recovered.append((sx, sy, sw, sh, col, row))

    if debug_dir and debug_img is not None:
        vis = debug_img.copy()
        for x, y, cw, ch, col, row in cards:
            cv2.rectangle(vis, (x, y), (x + cw, y + ch), (0, 255, 0), 4)
            cv2.putText(
                vis,
                f"({col},{row})",
                (x + 10, y + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3,
            )
        for x, y, cw, ch, col, row in recovered:
            cv2.rectangle(vis, (x, y), (x + cw, y + ch), (0, 165, 255), 4)
            cv2.putText(
                vis,
                f"({col},{row}) R",
                (x + 10, y + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 165, 255),
                3,
            )
        cv2.imwrite(str(debug_dir / "05_recovery.png"), vis)

    all_cards = cards + recovered
    all_cards.sort(key=lambda c: (c[5], c[4]))  # sort by row, then col
    return all_cards


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

    bg = _classify_background(gray, config)

    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / "00_gray.png"), gray)

    if bg == "dark":
        contours = _find_card_contours_dark(gray, config, debug_dir)
    else:
        contours = _find_card_contours_light(gray, config, debug_dir)

    cards = _filter_and_sort_contours(contours, h, w, config, debug_dir, img)

    if len(cards) < 8:
        cards = _recover_missing_cards(
            cards, gray, h, w, config, debug_dir, img
        )

    if max_cards is not None:
        cards = cards[:max_cards]

    output_paths: list[Path] = []
    for x, y, cw, ch, col, row in cards:
        card_stem = f"{volume}_{page_nr:04d}_{col}_{row}"
        out_path = config.extracted_cards_dir / f"{card_stem}.png"

        if out_path.exists() and not force:
            output_paths.append(out_path)
            continue

        card_img = img[y : y + ch, x : x + cw]
        cv2.imwrite(str(out_path), card_img)
        output_paths.append(out_path)

    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract cards from page images"
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing files"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save intermediate images for debugging",
    )
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
            extract_cards_from_page(
                page_path, config, force=args.force, debug_dir=debug_dir
            )


if __name__ == "__main__":
    main()

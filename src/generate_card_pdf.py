#!/usr/bin/env python3
"""Generate a PDF showing extracted cards with their OCR text side by side.

Layout: Each row has the card image on the left and the extracted text on
the right in a monospaced font. Row heights adapt to text length, with
cards packed greedily onto pages. Cards are ordered by filename so that
the left column of the original page comes first, followed by the right.

Usage:
    python -m src.generate_card_pdf                 # all volumes
    python -m src.generate_card_pdf 01              # single volume, all pages
    python -m src.generate_card_pdf 01 0009         # single volume, single page
"""

import glob
import io
import os
import sys
from collections import OrderedDict

import cv2
import fitz  # PyMuPDF

from .config import load_config

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MARGIN = 28
GAP = 14  # horizontal gap between image and text
ROW_GAP = 12  # vertical gap between card rows

# Page size (A4)
PAGE_W, PAGE_H = fitz.paper_size("A4")

# Font – prefer a nice monospaced font available on macOS
FONT_PATH = "/System/Library/Fonts/Supplemental/PTMono.ttc"
FONT_SIZE = 9

# Image target width in pixels (for downscaling before embedding)
IMG_TARGET_W = 800
IMG_JPEG_QUALITY = 85


def discover_volumes(cards_dir: str) -> list[str]:
    """Discover available volume numbers from extracted cards."""
    pattern = os.path.join(cards_dir, "*.png")
    volumes = set()
    for path in glob.glob(pattern):
        stem = os.path.splitext(os.path.basename(path))[0]
        parts = stem.split("_")
        if len(parts) >= 2:
            volumes.add(parts[0])
    return sorted(volumes)


def discover_pages(volume: str, cards_dir: str) -> list[str]:
    """Discover available page numbers for a volume from extracted cards."""
    pattern = os.path.join(cards_dir, f"{volume}_*_*.png")
    pages = set()
    for path in glob.glob(pattern):
        # filename: {volume}_{page}_{col}_{row}.png
        stem = os.path.splitext(os.path.basename(path))[0]
        parts = stem.split("_")
        if len(parts) >= 2:
            pages.add(parts[1])
    return sorted(pages)


def find_cards(volume: str, page: str, cards_dir: str) -> list[str]:
    """Return sorted list of card image paths for a given volume/page."""
    pattern = os.path.join(cards_dir, f"{volume}_{page}_*.png")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"Warning: no cards found matching {pattern}")
    return paths


def read_ocr(stem: str, ocr_dir: str) -> str:
    """Read the OCR text file for a card, return empty string if missing."""
    txt_path = os.path.join(ocr_dir, f"{stem}.txt")
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return "(no OCR text)"


def downscale_to_jpeg(path: str) -> tuple[bytes, float]:
    """Read an image, downscale it, return JPEG bytes and aspect ratio."""
    img = cv2.imread(path)
    h, w = img.shape[:2]
    aspect = w / h

    if w > IMG_TARGET_W:
        new_w = IMG_TARGET_W
        new_h = int(new_w / aspect)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, IMG_JPEG_QUALITY])
    if not ok:
        raise RuntimeError(f"Failed to encode {path}")
    return bytes(buf), aspect


def build_pdf(card_paths: list[str], ocr_dir: str, output: str) -> None:
    if not card_paths:
        sys.exit("No cards found")

    doc = fitz.open()

    # Load the monospaced font for TextWriter use
    if os.path.exists(FONT_PATH):
        font = fitz.Font(fontfile=FONT_PATH)
    else:
        font = fitz.Font("cour")
        print(f"Warning: {FONT_PATH} not found, falling back to Courier")

    usable_w = PAGE_W - 2 * MARGIN
    usable_h = PAGE_H - 2 * MARGIN

    img_col_w = (usable_w - GAP) * 0.48
    txt_col_w = (usable_w - GAP) * 0.52

    # Row overhead: 18pt top (label) + 12pt bottom (link area)
    ROW_OVERHEAD = 30
    MIN_ROW_H = 80  # minimum height for any card row

    # --- Pre-compute text requirements, grouped by source column ---
    # Group key: "{vol}_{page}_{col}" — cards from the same original column
    column_groups: OrderedDict[str, list[tuple]] = OrderedDict()
    for card_path in card_paths:
        stem = os.path.splitext(os.path.basename(card_path))[0]
        parts = stem.split("_")
        col_key = f"{parts[0]}_{parts[1]}_{parts[2]}"
        ocr_text = read_ocr(stem, ocr_dir)
        num_lines = max(ocr_text.count("\n") + 1, 1)
        text_h = num_lines * FONT_SIZE * 1.2 + ROW_OVERHEAD
        min_h = max(text_h, MIN_ROW_H)
        column_groups.setdefault(col_key, []).append(
            (card_path, stem, ocr_text, min_h)
        )

    # --- Pack each column group into its own page(s) ---
    page_batches: list[list[tuple]] = []
    for col_key, cards in column_groups.items():
        current: list[tuple] = []
        current_h = 0.0
        for info in cards:
            needed = info[3]
            gap = ROW_GAP if current else 0
            if current and current_h + gap + needed > usable_h:
                page_batches.append(current)
                current = []
                current_h = 0.0
                gap = 0
            current.append(info)
            current_h += gap + needed
        if current:
            page_batches.append(current)

    # --- Render each page ---
    for page_num_idx, batch in enumerate(page_batches):
        pdf_page = doc.new_page(width=PAGE_W, height=PAGE_H)

        # Distribute extra vertical space proportionally among rows
        total_min_h = sum(info[3] for info in batch)
        total_gaps = max(len(batch) - 1, 0) * ROW_GAP
        scale = (usable_h - total_gaps) / total_min_h if total_min_h > 0 else 1.0
        scale = max(scale, 1.0)  # never shrink below requested minimum

        # --- Page header ---
        first_stem = batch[0][1]
        parts = first_stem.split("_")
        vol_num, page_num, col_num = parts[0], parts[1], parts[2]
        col_label = "left" if col_num == "0" else "right"
        header_text = f"{vol_num}.pdf ; page {page_num} ; {col_label} column"
        header_width = font.text_length(header_text, fontsize=7)
        tw_header = fitz.TextWriter(pdf_page.rect)
        tw_header.append(
            fitz.Point((PAGE_W - header_width) / 2, MARGIN - 4),
            header_text, font=font, fontsize=7,
        )
        tw_header.write_text(pdf_page, color=(0.5, 0.5, 0.5))

        y_cursor = float(MARGIN)
        for row_idx, (card_path, stem, ocr_text, min_h) in enumerate(batch):
            row_h = min_h * scale

            # --- Separator line between rows ---
            if row_idx > 0:
                sep_y = y_cursor - ROW_GAP / 2
                shape = pdf_page.new_shape()
                shape.draw_line(
                    fitz.Point(MARGIN, sep_y),
                    fitz.Point(PAGE_W - MARGIN, sep_y),
                )
                shape.finish(color=(0.75, 0.75, 0.75), width=0.5)
                shape.commit()

            y_top = y_cursor
            y_bottom = y_top + row_h

            # --- Insert card image (downscaled) ---
            jpeg_bytes, aspect = downscale_to_jpeg(card_path)

            fit_w = img_col_w
            fit_h = fit_w / aspect
            if fit_h > row_h:
                fit_h = row_h
                fit_w = fit_h * aspect

            y_offset = (row_h - fit_h) / 2
            img_place = fitz.Rect(
                MARGIN,
                y_top + y_offset,
                MARGIN + fit_w,
                y_top + y_offset + fit_h,
            )
            pdf_page.insert_image(img_place, stream=jpeg_bytes)

            # --- Insert OCR text ---
            txt_x = MARGIN + img_col_w + GAP

            # Card label (grey, smaller)
            tw_label = fitz.TextWriter(pdf_page.rect)
            tw_label.append(fitz.Point(txt_x, y_top + 10), stem, font=font, fontsize=7)
            tw_label.write_text(pdf_page, color=(0.5, 0.5, 0.5))

            # OCR text
            ocr_rect = fitz.Rect(txt_x, y_top + 18, txt_x + txt_col_w, y_bottom - 12)
            pdf_page.insert_textbox(
                ocr_rect,
                ocr_text,
                fontname="F0",
                fontfile=FONT_PATH if os.path.exists(FONT_PATH) else None,
                fontsize=FONT_SIZE,
                align=fitz.TEXT_ALIGN_LEFT,
            )

            # GitHub edit link (small, grey, clickable) at bottom of row
            edit_url = f"https://github.com/DigitalHistory-Lund/SecToPat-CatCards/edit/main/transcriptions/{stem}.txt"
            link_label = "Suggest improved transcription: "
            full_text = link_label + edit_url
            text_width = font.text_length(full_text, fontsize=5)
            link_x = (PAGE_W - text_width) / 2
            tw_link = fitz.TextWriter(pdf_page.rect)
            link_y = y_bottom - 4
            label_end = tw_link.append(fitz.Point(link_x, link_y), link_label, font=font, fontsize=5)
            url_start = label_end[1]
            url_end = tw_link.append(url_start, edit_url, font=font, fontsize=5)
            tw_link.write_text(pdf_page, color=(0.5, 0.5, 0.5))
            url_rect = url_end[0]
            link_rect = fitz.Rect(url_start.x, url_rect.y0, url_rect.x1, url_rect.y1)
            pdf_page.insert_link({"kind": fitz.LINK_URI, "from": link_rect, "uri": edit_url})

            y_cursor = y_bottom + ROW_GAP

        # --- Page number (bottom-right) ---
        page_num_text = f"Page {page_num_idx + 1}"
        pnum_width = font.text_length(page_num_text, fontsize=7)
        tw_pnum = fitz.TextWriter(pdf_page.rect)
        tw_pnum.append(
            fitz.Point(PAGE_W - MARGIN - pnum_width, PAGE_H - 10),
            page_num_text, font=font, fontsize=7,
        )
        tw_pnum.write_text(pdf_page, color=(0.5, 0.5, 0.5))

    doc.save(output, deflate=True, garbage=4)
    doc.close()
    print(f"Saved {output} ({len(card_paths)} cards, {len(page_batches)} pages)")


def main() -> None:
    """Entry point for CLI and __main__.py dispatch."""
    cfg = load_config()
    cards_dir = str(cfg.extracted_cards_dir)
    ocr_dir = str(cfg.transcriptions_dir)

    if len(sys.argv) > 2:
        # Single volume + single page
        volume, page = sys.argv[1], sys.argv[2]
        card_paths = find_cards(volume, page, cards_dir)
        output = f"{volume}_{page}_cards.pdf"
    elif len(sys.argv) > 1:
        # Single volume, all pages
        volume = sys.argv[1]
        pages = discover_pages(volume, cards_dir)
        if not pages:
            sys.exit(f"No pages found for volume {volume}")
        print(f"Volume {volume}: {len(pages)} pages ({pages[0]}–{pages[-1]})")
        card_paths = []
        for p in pages:
            card_paths.extend(find_cards(volume, p, cards_dir))
        output = f"{volume}_cards.pdf"
    else:
        # All volumes
        volumes = discover_volumes(cards_dir)
        if not volumes:
            sys.exit("No volumes found")
        card_paths = []
        for volume in volumes:
            pages = discover_pages(volume, cards_dir)
            if pages:
                print(f"Volume {volume}: {len(pages)} pages ({pages[0]}–{pages[-1]})")
                for p in pages:
                    card_paths.extend(find_cards(volume, p, cards_dir))
        output = "all_cards.pdf"

    build_pdf(card_paths, ocr_dir, output)


if __name__ == "__main__":
    main()

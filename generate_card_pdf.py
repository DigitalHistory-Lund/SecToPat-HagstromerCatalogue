#!/usr/bin/env python3
"""Generate a PDF showing extracted cards with their OCR text side by side.

Layout: 4 cards per page. Each row has the card image on the left and
the extracted text on the right in a monospaced font. Cards are ordered
by filename so that the left column of the original page comes first
(page 1), followed by the right column (page 2).

Usage:
    python generate_card_pdf.py [volume] [page]
    python generate_card_pdf.py 01 0009
"""

import glob
import io
import os
import sys

import cv2
import fitz  # PyMuPDF

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CARDS_DIR = "extracted_cards"
OCR_DIR = "ocr_output"
CARDS_PER_PAGE = 4
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


def find_cards(volume: str, page: str, base: str) -> list[str]:
    """Return sorted list of card image paths for a given volume/page."""
    pattern = os.path.join(base, CARDS_DIR, f"{volume}_{page}_*.png")
    paths = sorted(glob.glob(pattern))
    if not paths:
        sys.exit(f"No cards found matching {pattern}")
    return paths


def read_ocr(stem: str, base: str) -> str:
    """Read the OCR text file for a card, return empty string if missing."""
    txt_path = os.path.join(base, OCR_DIR, f"{stem}.txt")
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


def build_pdf(volume: str, page: str, base: str, output: str) -> None:
    card_paths = find_cards(volume, page, base)

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

    row_h = (usable_h - (CARDS_PER_PAGE - 1) * ROW_GAP) / CARDS_PER_PAGE

    for page_idx in range(0, len(card_paths), CARDS_PER_PAGE):
        batch = card_paths[page_idx : page_idx + CARDS_PER_PAGE]
        pdf_page = doc.new_page(width=PAGE_W, height=PAGE_H)

        for row_idx, card_path in enumerate(batch):
            stem = os.path.splitext(os.path.basename(card_path))[0]
            ocr_text = read_ocr(stem, base)

            y_top = MARGIN + row_idx * (row_h + ROW_GAP)
            y_bottom = y_top + row_h

            # --- Separator line between rows ---
            if row_idx > 0:
                sep_y = y_top - ROW_GAP / 2
                shape = pdf_page.new_shape()
                shape.draw_line(
                    fitz.Point(MARGIN, sep_y),
                    fitz.Point(PAGE_W - MARGIN, sep_y),
                )
                shape.finish(color=(0.75, 0.75, 0.75), width=0.5)
                shape.commit()

            # --- Insert card image (downscaled) ---
            jpeg_bytes, aspect = downscale_to_jpeg(card_path)

            # Fit within the image column preserving aspect ratio
            fit_w = img_col_w
            fit_h = fit_w / aspect
            if fit_h > row_h:
                fit_h = row_h
                fit_w = fit_h * aspect

            # Center vertically
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
            ocr_rect = fitz.Rect(txt_x, y_top + 18, txt_x + txt_col_w, y_bottom - 4)
            rc = pdf_page.insert_textbox(
                ocr_rect,
                ocr_text,
                fontname="F0",
                fontfile=FONT_PATH if os.path.exists(FONT_PATH) else None,
                fontsize=FONT_SIZE,
                align=fitz.TEXT_ALIGN_LEFT,
            )
            if rc < 0:
                pdf_page.insert_textbox(
                    ocr_rect,
                    ocr_text,
                    fontname="F0",
                    fontfile=FONT_PATH if os.path.exists(FONT_PATH) else None,
                    fontsize=FONT_SIZE - 1,
                    align=fitz.TEXT_ALIGN_LEFT,
                )

    output_path = os.path.join(base, output)
    doc.save(output_path, deflate=True, garbage=4)
    doc.close()
    print(f"Saved {output_path} ({len(card_paths)} cards, "
          f"{(len(card_paths) + CARDS_PER_PAGE - 1) // CARDS_PER_PAGE} pages)")


if __name__ == "__main__":
    volume = sys.argv[1] if len(sys.argv) > 1 else "01"
    page = sys.argv[2] if len(sys.argv) > 2 else "0009"

    base = os.path.dirname(os.path.abspath(__file__))
    output = f"{volume}_{page}_cards.pdf"

    build_pdf(volume, page, base, output)

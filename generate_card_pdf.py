#!/usr/bin/env python3
"""Generate a PDF showing extracted cards with their OCR text side by side.

Layout: 4 cards per page. Each row has the card image on the left and
the extracted text on the right in a monospaced font. Cards are ordered
by filename so that the left column of the original page comes first
(page 1), followed by the right column (page 2).

Usage:
    python generate_card_pdf.py <volume> [page]
    python generate_card_pdf.py 01          # all discovered pages
    python generate_card_pdf.py 01 0009     # single page
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
OCR_DIR = "transcriptions"
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


def discover_pages(volume: str, base: str) -> list[str]:
    """Discover available page numbers for a volume from extracted cards."""
    pattern = os.path.join(base, CARDS_DIR, f"{volume}_*_*.png")
    pages = set()
    for path in glob.glob(pattern):
        # filename: {volume}_{page}_{col}_{row}.png
        stem = os.path.splitext(os.path.basename(path))[0]
        parts = stem.split("_")
        if len(parts) >= 2:
            pages.add(parts[1])
    return sorted(pages)


def find_cards(volume: str, page: str, base: str) -> list[str]:
    """Return sorted list of card image paths for a given volume/page."""
    pattern = os.path.join(base, CARDS_DIR, f"{volume}_{page}_*.png")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"Warning: no cards found matching {pattern}")
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


def build_pdf(volume: str, pages: list[str], base: str, output: str) -> None:
    card_paths = []
    for page in pages:
        card_paths.extend(find_cards(volume, page, base))

    if not card_paths:
        sys.exit(f"No cards found for volume {volume}, pages {pages}")

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

    header_h = 14  # space reserved for page header

    for page_idx in range(0, len(card_paths), CARDS_PER_PAGE):
        batch = card_paths[page_idx : page_idx + CARDS_PER_PAGE]
        pdf_page = doc.new_page(width=PAGE_W, height=PAGE_H)

        # --- Page header: "XX.pdf ; page Y ; left|right column" ---
        first_stem = os.path.splitext(os.path.basename(batch[0]))[0]
        parts = first_stem.split("_")
        page_num = parts[1]
        col_num = parts[2]
        col_label = "left" if col_num == "0" else "right"
        header_text = f"{volume}.pdf ; page {page_num} ; {col_label} column"
        header_width = font.text_length(header_text, fontsize=7)
        tw_header = fitz.TextWriter(pdf_page.rect)
        tw_header.append(
            fitz.Point((PAGE_W - header_width) / 2, MARGIN - 4),
            header_text, font=font, fontsize=7,
        )
        tw_header.write_text(pdf_page, color=(0.5, 0.5, 0.5))

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
            ocr_rect = fitz.Rect(txt_x, y_top + 18, txt_x + txt_col_w, y_bottom - 12)
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

    output_path = os.path.join(base, output)
    doc.save(output_path, deflate=True, garbage=4)
    doc.close()
    print(f"Saved {output_path} ({len(card_paths)} cards, "
          f"{(len(card_paths) + CARDS_PER_PAGE - 1) // CARDS_PER_PAGE} pages)")


if __name__ == "__main__":
    volume = sys.argv[1] if len(sys.argv) > 1 else "01"
    base = os.path.dirname(os.path.abspath(__file__))

    if len(sys.argv) > 2:
        # Single page mode
        page = sys.argv[2]
        output = f"{volume}_{page}_cards.pdf"
        pages = [page]
    else:
        # Multi-page mode: discover available pages
        pages = discover_pages(volume, base)
        if not pages:
            sys.exit(f"No pages found for volume {volume}")
        print(f"Discovered {len(pages)} pages for volume {volume}: {pages[0]}–{pages[-1]}")
        output = f"{volume}_cards.pdf"

    build_pdf(volume, pages, base, output)

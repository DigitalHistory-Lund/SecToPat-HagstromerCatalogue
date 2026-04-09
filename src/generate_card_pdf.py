#!/usr/bin/env python3
"""Generate a PDF showing extracted cards with their OCR text side by side.

Layout: Each row has the card image on the left and the extracted text on
the right in a monospaced font. Row heights adapt to text length, with
cards packed greedily onto pages. Cards are ordered by filename so that
the left column of the original page comes first, followed by the right.

Usage:
    python -m src.generate_card_pdf                 # all volumes
    python -m src.generate_card_pdf 01              # single volume, all pages
    python -m src.generate_card_pdf 01 0009  # single page
"""

import glob
import json
import os
import sys
import tomllib
from collections import OrderedDict
from pathlib import Path

import cv2
import fitz  # PyMuPDF
from tqdm import tqdm

from .config import PROJECT_ROOT

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
    pattern = os.path.join(cards_dir, "*.jpg")
    volumes = set()
    for path in glob.glob(pattern):
        stem = os.path.splitext(os.path.basename(path))[0]
        parts = stem.split("_")
        if len(parts) >= 2:
            volumes.add(parts[0])
    return sorted(volumes)


def discover_pages(volume: str, cards_dir: str) -> list[str]:
    """Discover available page numbers for a volume."""
    pattern = os.path.join(cards_dir, f"{volume}_*_*.jpg")
    pages = set()
    for path in glob.glob(pattern):
        stem = os.path.splitext(os.path.basename(path))[0]
        parts = stem.split("_")
        if len(parts) >= 2:
            pages.add(parts[1])
    return sorted(pages)


def find_cards(volume: str, page: str, cards_dir: str) -> list[str]:
    """Return sorted card image paths for a given volume/page."""
    pattern = os.path.join(cards_dir, f"{volume}_{page}_*.jpg")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"Warning: no cards found matching {pattern}")
    return paths


def read_ocr(stem: str, ocr_dir: str) -> str:
    """Read the OCR text file for a card."""
    txt_path = os.path.join(ocr_dir, f"{stem}.txt")
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return "(no OCR text)"


def downscale_to_jpeg(path: str) -> tuple[bytes, float]:
    """Read an image, downscale it, return JPEG bytes and aspect."""
    img = cv2.imread(path)
    h, w = img.shape[:2]
    aspect = w / h

    if w > IMG_TARGET_W:
        new_w = IMG_TARGET_W
        new_h = int(new_w / aspect)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(
        ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, IMG_JPEG_QUALITY]
    )
    if not ok:
        raise RuntimeError(f"Failed to encode {path}")
    return bytes(buf), aspect


# -------------------------------------------------------------------
# Metadata & special pages
# -------------------------------------------------------------------


def load_metadata() -> dict:
    """Read metadata.json from the project root."""
    meta_path = Path(__file__).resolve().parent.parent / "metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _add_footer(page, font, version_text, page_number):
    """Render version (bottom-left) and page number (bottom-right)."""
    tw_ver = fitz.TextWriter(page.rect)
    tw_ver.append(
        fitz.Point(MARGIN, PAGE_H - 10),
        version_text,
        font=font,
        fontsize=7,
    )
    tw_ver.write_text(page, color=(0.5, 0.5, 0.5))

    page_num_text = f"Page {page_number}"
    pnum_w = font.text_length(page_num_text, fontsize=7)
    tw_pnum = fitz.TextWriter(page.rect)
    tw_pnum.append(
        fitz.Point(PAGE_W - MARGIN - pnum_w, PAGE_H - 10),
        page_num_text,
        font=font,
        fontsize=7,
    )
    tw_pnum.write_text(page, color=(0.5, 0.5, 0.5))


def _centred_link(page, font, label, url, y, fontsize=9):
    """Render a centred clickable link and return the y after it."""
    tw = fitz.TextWriter(page.rect)
    lw = font.text_length(label, fontsize=fontsize)
    lx = (PAGE_W - lw) / 2
    end = tw.append(fitz.Point(lx, y), label, font=font, fontsize=fontsize)
    tw.write_text(page, color=(0.2, 0.2, 0.8))
    page.insert_link({"kind": fitz.LINK_URI, "from": end[0], "uri": url})
    return y + fontsize + 6


def insert_cover_page(doc, font, metadata, version_text):
    """Insert the overall cover page (page 1)."""
    page = doc.new_page(width=PAGE_W, height=PAGE_H)

    # Title — large, centred
    title = metadata["title"]
    tw = fitz.TextWriter(page.rect)
    title_w = font.text_length(title, fontsize=24)
    tw.append(
        fitz.Point((PAGE_W - title_w) / 2, PAGE_H * 0.22),
        title,
        font=font,
        fontsize=24,
    )
    tw.write_text(page)

    # Summary — centred block (tall enough for 2 long paragraphs)
    raw_summary = metadata.get("summary", [])
    if isinstance(raw_summary, list):
        summary = "\n\n".join(raw_summary)
    else:
        summary = raw_summary
    summary_top = PAGE_H * 0.27
    summary_bottom = PAGE_H * 0.58
    if summary:
        summary_rect = fitz.Rect(
            MARGIN + 40,
            summary_top,
            PAGE_W - MARGIN - 40,
            summary_bottom,
        )
        page.insert_textbox(
            summary_rect,
            summary,
            fontname="F0",
            fontfile=(FONT_PATH if os.path.exists(FONT_PATH) else None),
            fontsize=10,
            align=fitz.TEXT_ALIGN_CENTER,
        )

    # Authors — positioned below summary block
    y = summary_bottom + 20
    for author in metadata.get("authors", []):
        tw_a = fitz.TextWriter(page.rect)
        aw = font.text_length(author, fontsize=11)
        tw_a.append(
            fitz.Point((PAGE_W - aw) / 2, y),
            author,
            font=font,
            fontsize=11,
        )
        tw_a.write_text(page)
        y += 16

    # Links — flow below authors
    y += 12
    library_url = metadata.get("library_url", "")
    if library_url:
        y = _centred_link(page, font, library_url, library_url, y)

    doi = metadata.get("doi", "")
    if doi:
        doi_url = f"https://doi.org/{doi}"
        y = _centred_link(page, font, f"DOI: {doi}", doi_url, y)

    repo_url = metadata.get("repo_url", "")
    if repo_url:
        y = _centred_link(page, font, repo_url, repo_url, y)
        releases_url = repo_url + "/releases"
        y = _centred_link(
            page, font, "Download latest PDF version", releases_url, y
        )

    # Version — below links
    y += 10
    tw_v = fitz.TextWriter(page.rect)
    vw = font.text_length(version_text, fontsize=9)
    tw_v.append(
        fitz.Point((PAGE_W - vw) / 2, y),
        version_text,
        font=font,
        fontsize=9,
    )
    tw_v.write_text(page, color=(0.5, 0.5, 0.5))

    # License
    y += 20
    _centred_link(
        page,
        font,
        "Licensed under CC BY-NC 4.0",
        "https://creativecommons.org/licenses/by-nc/4.0/",
        y,
        fontsize=9,
    )


def insert_toc_page(doc, font, metadata, volume_page_map, version_text):
    """Insert the Table of Contents (page 2)."""
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    volumes_meta = metadata.get("volumes", {})

    # Heading
    heading = "Table of Contents"
    tw_h = fitz.TextWriter(page.rect)
    hw = font.text_length(heading, fontsize=18)
    tw_h.append(
        fitz.Point((PAGE_W - hw) / 2, MARGIN + 30),
        heading,
        font=font,
        fontsize=18,
    )
    tw_h.write_text(page)

    # One line per volume
    y = MARGIN + 60
    for vol_id, pg in volume_page_map.items():
        vol_info = volumes_meta.get(vol_id, {})
        vol_title = vol_info.get("title", f"Volume {vol_id}")
        page_label = f"Page {pg}"

        tw_e = fitz.TextWriter(page.rect)
        tw_e.append(
            fitz.Point(MARGIN + 20, y),
            vol_title,
            font=font,
            fontsize=11,
        )
        pn_w = font.text_length(page_label, fontsize=11)
        tw_e.append(
            fitz.Point(PAGE_W - MARGIN - 20 - pn_w, y),
            page_label,
            font=font,
            fontsize=11,
        )
        tw_e.write_text(page, color=(0.1, 0.1, 0.1))
        y += 20

    _add_footer(page, font, version_text, 2)


def insert_volume_cover(
    doc, font, vol_id, vol_meta, version_text, page_number
):
    """Insert a cover page for a single volume."""
    page = doc.new_page(width=PAGE_W, height=PAGE_H)

    title = vol_meta.get("title", f"Volume {vol_id}")
    tw = fitz.TextWriter(page.rect)
    tw_w = font.text_length(title, fontsize=22)
    tw.append(
        fitz.Point((PAGE_W - tw_w) / 2, PAGE_H * 0.4),
        title,
        font=font,
        fontsize=22,
    )
    tw.write_text(page)

    source_url = vol_meta.get("source_url", "")
    if source_url:
        _centred_link(
            page,
            font,
            "Original PDF on ALVIN",
            source_url,
            PAGE_H * 0.47,
        )

    _add_footer(page, font, version_text, page_number)


# -------------------------------------------------------------------
# Card page rendering
# -------------------------------------------------------------------


def _render_card_page(
    doc,
    font,
    batch,
    version_text,
    page_number,
    usable_h,
    img_col_w,
    txt_col_w,
    repo_url="",
):
    """Render a single page of card rows."""
    pdf_page = doc.new_page(width=PAGE_W, height=PAGE_H)

    # Distribute extra vertical space proportionally
    total_min_h = sum(info[3] for info in batch)
    total_gaps = max(len(batch) - 1, 0) * ROW_GAP
    scale = (usable_h - total_gaps) / total_min_h if total_min_h > 0 else 1.0
    scale = max(scale, 1.0)

    # Page header
    first_stem = batch[0][1]
    parts = first_stem.split("_")
    vol_num, page_num, col_num = (
        parts[0],
        parts[1],
        parts[2],
    )
    col_label = "left" if col_num == "0" else "right"
    header_text = f"{vol_num}.pdf ; page {page_num} ;" f" {col_label} column"
    header_width = font.text_length(header_text, fontsize=7)
    tw_header = fitz.TextWriter(pdf_page.rect)
    tw_header.append(
        fitz.Point((PAGE_W - header_width) / 2, MARGIN - 4),
        header_text,
        font=font,
        fontsize=7,
    )
    tw_header.write_text(pdf_page, color=(0.5, 0.5, 0.5))

    y_cursor = float(MARGIN)
    for row_idx, (card_path, stem, ocr_text, min_h) in enumerate(batch):
        row_h = min_h * scale

        # Separator line between rows
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

        # Card image (downscaled)
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

        # OCR text column
        txt_x = MARGIN + img_col_w + GAP

        # Card label (grey, smaller)
        tw_label = fitz.TextWriter(pdf_page.rect)
        tw_label.append(
            fitz.Point(txt_x, y_top + 10),
            stem,
            font=font,
            fontsize=7,
        )
        tw_label.write_text(pdf_page, color=(0.5, 0.5, 0.5))

        # OCR text
        ocr_rect = fitz.Rect(
            txt_x,
            y_top + 18,
            txt_x + txt_col_w,
            y_bottom - 12,
        )
        pdf_page.insert_textbox(
            ocr_rect,
            ocr_text,
            fontname="F0",
            fontfile=(FONT_PATH if os.path.exists(FONT_PATH) else None),
            fontsize=FONT_SIZE,
            align=fitz.TEXT_ALIGN_LEFT,
        )

        # GitHub edit link (small, grey, clickable)
        edit_url = f"{repo_url}/edit/main/transcriptions/{stem}.txt"
        link_label = "Suggest improved transcription: "
        full_text = link_label + edit_url
        text_width = font.text_length(full_text, fontsize=5)
        link_x = (PAGE_W - text_width) / 2
        tw_link = fitz.TextWriter(pdf_page.rect)
        link_y = y_bottom - 4
        label_end = tw_link.append(
            fitz.Point(link_x, link_y),
            link_label,
            font=font,
            fontsize=5,
        )
        url_start = label_end[1]
        url_end = tw_link.append(
            url_start,
            edit_url,
            font=font,
            fontsize=5,
        )
        tw_link.write_text(pdf_page, color=(0.5, 0.5, 0.5))
        url_rect = url_end[0]
        link_rect = fitz.Rect(
            url_start.x,
            url_rect.y0,
            url_rect.x1,
            url_rect.y1,
        )
        pdf_page.insert_link(
            {
                "kind": fitz.LINK_URI,
                "from": link_rect,
                "uri": edit_url,
            }
        )

        y_cursor = y_bottom + ROW_GAP

    _add_footer(pdf_page, font, version_text, page_number)


# -------------------------------------------------------------------
# PDF assembly
# -------------------------------------------------------------------


def build_pdf(
    card_paths: list[str],
    ocr_dir: str,
    output: str,
    metadata: dict | None = None,
) -> None:
    """Build the full PDF from card images and OCR text."""
    if not card_paths:
        sys.exit("No cards found")

    doc = fitz.open()

    # Load the monospaced font for TextWriter use
    if os.path.exists(FONT_PATH):
        font = fitz.Font(fontfile=FONT_PATH)
    else:
        font = fitz.Font("cour")
        print(f"Warning: {FONT_PATH} not found, " "falling back to Courier")

    usable_w = PAGE_W - 2 * MARGIN
    usable_h = PAGE_H - 2 * MARGIN

    img_col_w = (usable_w - GAP) * 0.48
    txt_col_w = (usable_w - GAP) * 0.52

    # Row overhead: 18pt top (label) + 12pt bottom (link)
    ROW_OVERHEAD = 30
    MIN_ROW_H = 80  # minimum height for any card row

    # --- Pre-compute text requirements by source column ---
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

    # --- Pack each column group into page(s) ---
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

    # --- Read version once ---
    _pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with open(_pyproject, "rb") as f:
        version_text = "v" + tomllib.load(f)["project"]["version"]

    # --- Render ---
    repo_url = metadata.get("repo_url", "") if metadata else ""
    if metadata:
        # Group page batches by volume
        volume_batches: OrderedDict[str, list] = OrderedDict()
        for batch in page_batches:
            vol_id = batch[0][1].split("_")[0]
            volume_batches.setdefault(vol_id, []).append(batch)

        # First pass: compute page numbers
        # Page 1 = cover, Page 2 = ToC
        current_page = 3
        volume_page_map: OrderedDict[str, int] = OrderedDict()
        for vol_id, batches in volume_batches.items():
            volume_page_map[vol_id] = current_page
            current_page += 1 + len(batches)

        # Second pass: render pages
        insert_cover_page(doc, font, metadata, version_text)
        insert_toc_page(
            doc,
            font,
            metadata,
            volume_page_map,
            version_text,
        )

        overall_page = 3
        for vol_id, batches in volume_batches.items():
            vol_meta = metadata.get("volumes", {}).get(vol_id, {})
            insert_volume_cover(
                doc,
                font,
                vol_id,
                vol_meta,
                version_text,
                overall_page,
            )
            overall_page += 1

            for batch in tqdm(
                batches,
                unit="page",
                desc=f"Vol {vol_id}",
            ):
                _render_card_page(
                    doc,
                    font,
                    batch,
                    version_text,
                    overall_page,
                    usable_h,
                    img_col_w,
                    txt_col_w,
                    repo_url,
                )
                overall_page += 1

        total_pages = overall_page - 1
    else:
        # Original flow without cover/ToC
        for page_num_idx, batch in tqdm(
            enumerate(page_batches),
            total=len(page_batches),
            unit="page",
            desc="PDF",
        ):
            _render_card_page(
                doc,
                font,
                batch,
                version_text,
                page_num_idx + 1,
                usable_h,
                img_col_w,
                txt_col_w,
                repo_url,
            )
        total_pages = len(page_batches)

    doc.save(output, deflate=True, garbage=4)
    doc.close()
    print(f"Saved {output}" f" ({len(card_paths)} cards, {total_pages} pages)")


def main() -> None:
    """Entry point for CLI and __main__.py dispatch."""
    cards_dir = str(PROJECT_ROOT / "cards_web")
    ocr_dir = str(PROJECT_ROOT / "transcriptions")

    if len(sys.argv) > 2:
        # Single volume + single page
        volume, page = sys.argv[1], sys.argv[2]
        card_paths = find_cards(volume, page, cards_dir)
        output = f"{volume}_{page}_cards.pdf"
        metadata = None
    elif len(sys.argv) > 1:
        # Single volume, all pages
        volume = sys.argv[1]
        pages = discover_pages(volume, cards_dir)
        if not pages:
            sys.exit(f"No pages found for volume {volume}")
        print(
            f"Volume {volume}: {len(pages)} pages"
            f" ({pages[0]}\u2013{pages[-1]})"
        )
        card_paths = []
        for p in pages:
            card_paths.extend(find_cards(volume, p, cards_dir))
        output = f"{volume}_cards.pdf"
        metadata = None
    else:
        # All volumes
        volumes = discover_volumes(cards_dir)
        if not volumes:
            sys.exit("No volumes found")
        card_paths = []
        for volume in volumes:
            pages = discover_pages(volume, cards_dir)
            if pages:
                print(
                    f"Volume {volume}: {len(pages)} pages"
                    f" ({pages[0]}\u2013{pages[-1]})"
                )
                for p in pages:
                    card_paths.extend(find_cards(volume, p, cards_dir))
        metadata = load_metadata()
        _pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        with open(_pyproject, "rb") as f:
            version = tomllib.load(f)["project"]["version"]
        output = f"HagstromerCatalogue_{version}.pdf"

    build_pdf(card_paths, ocr_dir, output, metadata)


if __name__ == "__main__":
    main()

"""Embed OCR text into PDF copies as an invisible searchable layer."""

import argparse
import json
from pathlib import Path

import fitz
from tqdm import tqdm

from .config import Config, load_config


def embed_text_in_pdf(pdf_path: Path, config: Config, force: bool = False) -> Path | None:
    """Create a searchable copy of a PDF by embedding OCR text at card locations.

    Returns the output path, or None if skipped.
    """
    config.searchable_pdfs_dir.mkdir(parents=True, exist_ok=True)

    volume = pdf_path.stem
    out_path = config.searchable_pdfs_dir / f"{volume}.pdf"

    if out_path.exists() and not force:
        return None

    # Collect all boxes files for this volume
    boxes_files = sorted(config.extracted_cards_dir.glob(f"{volume}_*_boxes.json"))
    if not boxes_files:
        return None

    doc = fitz.open(pdf_path)

    for boxes_file in boxes_files:
        # Parse page number from filename like "03_0005_boxes.json"
        parts = boxes_file.stem.replace("_boxes", "").split("_", 1)
        page_nr = int(parts[1])

        if page_nr >= len(doc):
            continue

        page = doc[page_nr]

        boxes: dict[str, list[int]] = json.loads(boxes_file.read_text(encoding="utf-8"))

        for card_stem, (x, y, w, h) in boxes.items():
            ocr_path = config.ocr_output_dir / f"{card_stem}.txt"
            if not ocr_path.exists():
                continue

            text = ocr_path.read_text(encoding="utf-8").strip()
            if not text:
                continue

            rect = fitz.Rect(x, y, x + w, y + h)
            page.insert_textbox(rect, text, fontsize=10, render_mode=3)

    doc.save(str(out_path))
    doc.close()
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed OCR text into searchable PDF copies")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    config = load_config()
    pdf_files = sorted(config.raw_cat_path.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in RAW_CAT_PATH.")
        return

    if not args.force:
        pdf_files = [p for p in pdf_files if not (config.searchable_pdfs_dir / f"{p.stem}.pdf").exists()]

    with tqdm(pdf_files, unit="vol") as bar:
        for pdf_path in bar:
            bar.desc = f"Embed {pdf_path.stem}"
            embed_text_in_pdf(pdf_path, config, force=args.force)


if __name__ == "__main__":
    main()

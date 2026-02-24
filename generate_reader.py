"""Generate a reader/ directory with .md files for browsing extracted cards and OCR text."""

import os
import shutil
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from src.config import load_config

def parse_stem(stem: str) -> tuple[str, str, str, str]:
    """Parse a card stem like '05_0001_0_2' into (vol, page, col, row)."""
    vol, page, col, row = stem.split("_")
    return vol, page, col, row


def generate_reader() -> None:
    cfg = load_config()
    reader_dir = cfg.reader_dir

    # Collect qualifying card stems (those with an OCR .txt file)
    txt_files = sorted(cfg.ocr_output_dir.glob("*.txt"))
    if not txt_files:
        print("No OCR text files found — nothing to generate.")
        return

    # Group by volume -> page -> list of (col, row, stem)
    volumes: dict[str, dict[str, list[tuple[str, str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for txt_path in txt_files:
        stem = txt_path.stem
        vol, page, col, row = parse_stem(stem)
        volumes[vol][page].append((col, row, stem))

    # Clean and recreate reader/
    if reader_dir.exists():
        shutil.rmtree(reader_dir)
    reader_dir.mkdir()

    # Generate overview
    overview_lines = ["# Reader Overview", ""]
    for vol in sorted(volumes):
        overview_lines.append(f"- [Volume {vol}]({vol}.md)")
    overview_lines.append("")
    (reader_dir / "overview.md").write_text("\n".join(overview_lines))

    # Generate per-volume and per-page files
    total_pages = sum(len(pages) for pages in volumes.values())
    with tqdm(total=total_pages, unit="page") as bar:
        for vol in sorted(volumes):
            pages = volumes[vol]
            vol_lines = [f"# Volume {vol}", ""]
            for page in sorted(pages):
                vol_lines.append(f"- [Page {page}]({vol}/{page}.md)")
            vol_lines.append("")
            (reader_dir / f"{vol}.md").write_text("\n".join(vol_lines))

            vol_dir = reader_dir / vol
            vol_dir.mkdir()

            # Relative path from page file (vol_dir/) to extracted_cards_dir
            cards_rel = os.path.relpath(cfg.extracted_cards_dir, vol_dir)

            for page in sorted(pages):
                bar.desc = f"Vol {vol} p.{page}"
                cards = sorted(pages[page])
                page_lines: list[str] = []
                for col, row, stem in cards:
                    ocr_text = (cfg.ocr_output_dir / f"{stem}.txt").read_text().rstrip()
                    img_src = f"{cards_rel}/{stem}.png"
                    ocr_html = ocr_text.replace("&", "&amp;").replace("<", "&lt;")
                    page_lines.append("<table><tr>")
                    page_lines.append(f'<td><img src="{img_src}" width="350"></td>')
                    page_lines.append(f"<td><pre>{ocr_html}</pre></td>")
                    page_lines.append("</tr></table>")
                    page_lines.append("")

                (vol_dir / f"{page}.md").write_text("\n".join(page_lines))
                bar.update(1)


if __name__ == "__main__":
    generate_reader()

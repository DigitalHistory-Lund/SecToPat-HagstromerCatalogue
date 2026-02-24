"""Extract page images from catalogue PDFs using PyMuPDF."""

import argparse
from pathlib import Path

import fitz

from tqdm import tqdm

from .config import Config, load_config


def extract_pages_from_pdf(
    pdf_path: Path,
    config: Config,
    force: bool = False,
    page_indices: list[int] | None = None,
) -> list[Path]:
    """Render PDF pages to PNG files.

    Returns list of output paths (newly created or already existing).
    """
    volume = pdf_path.stem  # e.g. "01"
    config.extracted_images_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    indices = page_indices if page_indices is not None else range(len(doc))

    output_paths: list[Path] = []
    for page_idx in indices:
        out_name = f"{volume}_{page_idx:04d}.png"
        out_path = config.extracted_images_dir / out_name

        if out_path.exists() and not force:
            output_paths.append(out_path)
            continue

        page = doc.load_page(page_idx)
        pix = page.get_pixmap()
        pix.save(str(out_path))
        output_paths.append(out_path)

    doc.close()
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract page images from PDFs")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    config = load_config()
    pdf_files = sorted(config.raw_cat_path.glob("*.pdf"))

    with tqdm(pdf_files, unit="vol") as bar:
        for pdf_path in bar:
            bar.desc = f"Pages {pdf_path.stem}"
            extract_pages_from_pdf(pdf_path, config, force=args.force)


if __name__ == "__main__":
    main()

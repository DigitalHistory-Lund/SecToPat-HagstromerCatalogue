"""CLI orchestrator for the catalogue card extraction pipeline."""

import argparse
import tempfile
from pathlib import Path

from .config import load_config
from .extract_cards import extract_cards_from_page
from .extract_pages import extract_pages_from_pdf
from .subset import select_subset


def main() -> None:
    parser = argparse.ArgumentParser(description="Hagstrom catalogue extraction pipeline")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--subset", action="store_true", help="Process only the seeded subset")
    parser.add_argument("--step", choices=["pages", "cards"], help="Run a single step")
    parser.add_argument("--debug", action="store_true", help="Save intermediate card-detection images")
    args = parser.parse_args()

    config = load_config()
    run_pages = args.step in (None, "pages")
    run_cards = args.step in (None, "cards")

    debug_base = None
    if args.debug and run_cards:
        debug_base = Path(tempfile.mkdtemp(prefix="card_debug_"))
        print(f"Debug images will be saved to: {debug_base}")

    if args.subset:
        selection = select_subset(config)
        print(f"Subset: {selection.volumes} (max {selection.num_cards} cards/page)")

        for volume, page_indices in selection.volumes.items():
            pdf_path = config.raw_cat_path / f"{volume}.pdf"

            if run_pages:
                print(f"Extracting pages from {pdf_path.name}...")
                page_paths = extract_pages_from_pdf(pdf_path, config, force=args.force, page_indices=page_indices)
            else:
                # Build expected page paths for card extraction
                page_paths = [
                    config.extracted_images_dir / f"{volume}_{idx:04d}.png"
                    for idx in page_indices
                ]

            if run_cards:
                for page_path in page_paths:
                    if not page_path.exists():
                        print(f"  Skipping {page_path.name} (not found)")
                        continue
                    print(f"Extracting cards from {page_path.name}...")
                    debug_dir = debug_base / page_path.stem if debug_base else None
                    extract_cards_from_page(
                        page_path, config, force=args.force,
                        max_cards=selection.num_cards, debug_dir=debug_dir,
                    )
    else:
        pdf_files = sorted(config.raw_cat_path.glob("*.pdf"))

        for pdf_path in pdf_files:
            if run_pages:
                print(f"Extracting pages from {pdf_path.name}...")
                page_paths = extract_pages_from_pdf(pdf_path, config, force=args.force)

            if run_cards:
                # Process all page images for this volume
                volume = pdf_path.stem
                page_images = sorted(config.extracted_images_dir.glob(f"{volume}_*.png"))
                for page_path in page_images:
                    print(f"Extracting cards from {page_path.name}...")
                    debug_dir = debug_base / page_path.stem if debug_base else None
                    extract_cards_from_page(page_path, config, force=args.force, debug_dir=debug_dir)

    print("Done.")


if __name__ == "__main__":
    main()

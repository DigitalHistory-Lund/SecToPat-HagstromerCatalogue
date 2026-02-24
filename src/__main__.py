"""CLI orchestrator for the catalogue card extraction pipeline."""

import argparse
import tempfile
from pathlib import Path

from tqdm import tqdm

from .config import load_config
from .embed_text import embed_text_in_pdf
from .extract_cards import extract_cards_from_page, page_cards_complete
from .extract_pages import extract_pages_from_pdf
from .ocr_cards import ocr_card
from .subset import select_subset


def main() -> None:
    parser = argparse.ArgumentParser(description="Hagstrom catalogue extraction pipeline")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--all", action="store_true", help="Process all volumes instead of the seeded subset")
    parser.add_argument("--step", choices=["pages", "cards", "ocr", "embed"], help="Run a single step")
    parser.add_argument("--debug", action="store_true", help="Save intermediate card-detection images")
    args = parser.parse_args()

    config = load_config()
    run_pages = args.step in (None, "pages")
    run_cards = args.step in (None, "cards")
    run_ocr = args.step in (None, "ocr")
    run_embed = args.step in (None, "embed")

    debug_base = None
    if args.debug and run_cards:
        debug_base = Path(tempfile.mkdtemp(prefix="card_debug_"))
        print(f"Debug images will be saved to: {debug_base}")

    if not args.all:
        selection = select_subset(config)

        # --- Pages ---
        # Always build expected paths (needed by cards step)
        page_paths_by_vol: dict[str, list[Path]] = {}
        for volume, page_indices in selection.volumes.items():
            page_paths_by_vol[volume] = [
                config.extracted_images_dir / f"{volume}_{idx:04d}.png"
                for idx in page_indices
            ]

        if run_pages:
            # Filter to pages that actually need rendering
            pending: list[tuple[str, list[int]]] = []
            for volume in sorted(selection.volumes):
                indices = [
                    idx for idx in selection.volumes[volume]
                    if args.force or not (config.extracted_images_dir / f"{volume}_{idx:04d}.png").exists()
                ]
                if indices:
                    pending.append((volume, indices))

            total = sum(len(idxs) for _, idxs in pending)
            with tqdm(total=total, unit="page") as bar:
                for volume, indices in pending:
                    pdf_path = config.raw_cat_path / f"{volume}.pdf"
                    bar.desc = f"Pages {volume}"
                    extract_pages_from_pdf(pdf_path, config, force=args.force, page_indices=indices)
                    bar.update(len(indices))

        # --- Cards ---
        card_paths: list[Path] = []
        if run_cards:
            all_page_paths = [
                p for vol in sorted(page_paths_by_vol) for p in page_paths_by_vol[vol]
                if p.exists() and (args.force or not page_cards_complete(p, config))
            ]
            with tqdm(all_page_paths, unit="page") as bar:
                for page_path in bar:
                    bar.desc = f"Cards {page_path.stem}"
                    debug_dir = debug_base / page_path.stem if debug_base else None
                    card_paths.extend(extract_cards_from_page(
                        page_path, config, force=args.force,
                        max_cards=selection.num_cards, debug_dir=debug_dir,
                    ))

        # --- OCR ---
        if run_ocr:
            if not card_paths:
                for volume, page_indices in selection.volumes.items():
                    for idx in page_indices:
                        card_paths.extend(sorted(
                            config.extracted_cards_dir.glob(f"{volume}_{idx:04d}_*.png")
                        ))
                card_paths = card_paths[: selection.num_cards]

            if not args.force:
                card_paths = [p for p in card_paths if not (config.ocr_output_dir / f"{p.stem}.txt").exists()]

            with tqdm(card_paths, unit="card") as bar:
                for card_path in bar:
                    bar.desc = f"OCR {card_path.stem}"
                    ocr_card(card_path, config, force=args.force)

        # --- Embed ---
        if run_embed:
            volumes = sorted(selection.volumes)
            pdf_files = [config.raw_cat_path / f"{v}.pdf" for v in volumes]
            pdf_files = [p for p in pdf_files if p.exists()]
            if not args.force:
                pdf_files = [p for p in pdf_files if not (config.searchable_pdfs_dir / f"{p.stem}.pdf").exists()]
            with tqdm(pdf_files, unit="vol") as bar:
                for pdf_path in bar:
                    bar.desc = f"Embed {pdf_path.stem}"
                    embed_text_in_pdf(pdf_path, config, force=args.force)
    else:
        pdf_files = sorted(config.raw_cat_path.glob("*.pdf"))

        if run_pages:
            with tqdm(pdf_files, unit="vol") as bar:
                for pdf_path in bar:
                    bar.desc = f"Pages {pdf_path.stem}"
                    extract_pages_from_pdf(pdf_path, config, force=args.force)

        if run_cards:
            page_images = sorted(config.extracted_images_dir.glob("*.png"))
            if not args.force:
                page_images = [p for p in page_images if not page_cards_complete(p, config)]
            with tqdm(page_images, unit="page") as bar:
                for page_path in bar:
                    bar.desc = f"Cards {page_path.stem}"
                    debug_dir = debug_base / page_path.stem if debug_base else None
                    extract_cards_from_page(page_path, config, force=args.force, debug_dir=debug_dir)

        if run_ocr:
            card_images = sorted(config.extracted_cards_dir.glob("*.png"))
            if not args.force:
                card_images = [p for p in card_images if not (config.ocr_output_dir / f"{p.stem}.txt").exists()]
            with tqdm(card_images, unit="card") as bar:
                for card_path in bar:
                    bar.desc = f"OCR {card_path.stem}"
                    ocr_card(card_path, config, force=args.force)

        if run_embed:
            embed_pdfs = sorted(config.raw_cat_path.glob("*.pdf"))
            if not args.force:
                embed_pdfs = [p for p in embed_pdfs if not (config.searchable_pdfs_dir / f"{p.stem}.pdf").exists()]
            with tqdm(embed_pdfs, unit="vol") as bar:
                for pdf_path in bar:
                    bar.desc = f"Embed {pdf_path.stem}"
                    embed_text_in_pdf(pdf_path, config, force=args.force)


if __name__ == "__main__":
    main()

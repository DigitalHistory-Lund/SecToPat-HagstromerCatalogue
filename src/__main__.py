"""CLI orchestrator for the catalogue card extraction pipeline."""

import argparse
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from .check_images import main as check_images_main
from .config import load_config
from .extract_cards import extract_cards_from_page
from .extract_pages import extract_pages_from_pdf
from .generate_card_pdf import main as generate_card_pdf_main
from .generate_reader import generate_reader
from .generate_site import generate_site as generate_site_main
from .ocr_cards import ocr_card
from .subset import select_subset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hagstrom catalogue extraction pipeline"
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing outputs"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all volumes instead of the seeded subset",
    )
    parser.add_argument(
        "--step",
        choices=[
            "pages",
            "cards",
            "ocr",
            "check-images",
            "card-pdf",
            "reader",
            "site",
        ],
        help="Run a single step",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save intermediate card-detection images",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="CPU workers for pages/cards (default: 1)",
    )
    args = parser.parse_args()

    # Dispatch standalone steps that are not part of the default pipeline
    if args.step == "check-images":
        sys.exit(check_images_main())
    if args.step == "card-pdf":
        generate_card_pdf_main()
        return
    if args.step == "reader":
        generate_reader()
        return
    if args.step == "site":
        generate_site_main()
        return

    config = load_config()
    run_pages = args.step in (None, "pages")
    run_cards = args.step in (None, "cards")
    run_ocr = args.step in (None, "ocr")

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
            t0 = time.perf_counter()
            # Filter to pages that actually need rendering
            pending: list[tuple[str, list[int]]] = []
            for volume in sorted(selection.volumes):
                indices = [
                    idx
                    for idx in selection.volumes[volume]
                    if args.force
                    or not (
                        config.extracted_images_dir / f"{volume}_{idx:04d}.png"
                    ).exists()
                ]
                if indices:
                    pending.append((volume, indices))

            total = sum(len(idxs) for _, idxs in pending)

            if args.workers > 1 and len(pending) > 1:
                with tqdm(total=total, unit="page") as bar:
                    with ProcessPoolExecutor(max_workers=args.workers) as pool:
                        futures = {
                            pool.submit(
                                extract_pages_from_pdf,
                                config.raw_cat_path / f"{volume}.pdf",
                                config,
                                args.force,
                                indices,
                            ): (volume, len(indices))
                            for volume, indices in pending
                        }
                        for future in as_completed(futures):
                            future.result()
                            vol, n = futures[future]
                            bar.desc = f"Pages {vol}"
                            bar.update(n)
            else:
                with tqdm(total=total, unit="page") as bar:
                    for volume, indices in pending:
                        pdf_path = config.raw_cat_path / f"{volume}.pdf"
                        bar.desc = f"Pages {volume}"
                        extract_pages_from_pdf(
                            pdf_path,
                            config,
                            force=args.force,
                            page_indices=indices,
                        )
                        bar.update(len(indices))

            elapsed = time.perf_counter() - t0
            print(
                f"[timing] Pages: {elapsed:.1f}s"
                f" ({total} pages, {args.workers} workers)"
            )

        # --- Cards ---
        card_paths: list[Path] = []
        if run_cards:
            t0 = time.perf_counter()
            all_page_paths = [
                p
                for vol in sorted(page_paths_by_vol)
                for p in page_paths_by_vol[vol]
                if p.exists()
            ]

            if args.workers > 1 and len(all_page_paths) > 1:
                with tqdm(total=len(all_page_paths), unit="page") as bar:
                    with ProcessPoolExecutor(max_workers=args.workers) as pool:
                        futures = {
                            pool.submit(
                                extract_cards_from_page,
                                page_path,
                                config,
                                args.force,
                                selection.num_cards,
                                (
                                    debug_base / page_path.stem
                                    if debug_base
                                    else None
                                ),
                            ): page_path
                            for page_path in all_page_paths
                        }
                        for future in as_completed(futures):
                            bar.desc = f"Cards {futures[future].stem}"
                            card_paths.extend(future.result())
                            bar.update(1)
            else:
                with tqdm(all_page_paths, unit="page") as bar:
                    for page_path in bar:
                        bar.desc = f"Cards {page_path.stem}"
                        debug_dir = (
                            debug_base / page_path.stem if debug_base else None
                        )
                        card_paths.extend(
                            extract_cards_from_page(
                                page_path,
                                config,
                                force=args.force,
                                max_cards=selection.num_cards,
                                debug_dir=debug_dir,
                            )
                        )

            elapsed = time.perf_counter() - t0
            print(
                f"[timing] Cards: {elapsed:.1f}s"
                f" ({len(all_page_paths)} pages,"
                f" {len(card_paths)} cards,"
                f" {args.workers} workers)"
            )

        # --- OCR ---
        if run_ocr:
            t0 = time.perf_counter()
            if not card_paths:
                for volume, page_indices in selection.volumes.items():
                    for idx in page_indices:
                        card_paths.extend(
                            sorted(
                                config.extracted_cards_dir.glob(
                                    f"{volume}_{idx:04d}_*.png"
                                )
                            )
                        )
                card_paths = card_paths[: selection.num_cards]

            if not args.force:
                card_paths = [
                    p
                    for p in card_paths
                    if not (config.ocr_output_dir / f"{p.stem}.txt").exists()
                ]

            with tqdm(card_paths, unit="card") as bar:
                for card_path in bar:
                    bar.desc = f"OCR {card_path.stem}"
                    ocr_card(card_path, config, force=args.force)

            elapsed = time.perf_counter() - t0
            print(f"[timing] OCR: {elapsed:.1f}s ({len(card_paths)} cards)")

    else:
        pdf_files = sorted(config.raw_cat_path.glob("*.pdf"))

        if run_pages:
            t0 = time.perf_counter()
            if args.workers > 1 and len(pdf_files) > 1:
                with tqdm(total=len(pdf_files), unit="vol") as bar:
                    with ProcessPoolExecutor(max_workers=args.workers) as pool:
                        futures = {
                            pool.submit(
                                extract_pages_from_pdf,
                                pdf_path,
                                config,
                                args.force,
                            ): pdf_path
                            for pdf_path in pdf_files
                        }
                        for future in as_completed(futures):
                            bar.desc = f"Pages {futures[future].stem}"
                            future.result()
                            bar.update(1)
            else:
                with tqdm(pdf_files, unit="vol") as bar:
                    for pdf_path in bar:
                        bar.desc = f"Pages {pdf_path.stem}"
                        extract_pages_from_pdf(
                            pdf_path, config, force=args.force
                        )
            elapsed = time.perf_counter() - t0
            print(
                f"[timing] Pages: {elapsed:.1f}s"
                f" ({len(pdf_files)} volumes,"
                f" {args.workers} workers)"
            )

        if run_cards:
            t0 = time.perf_counter()
            page_images = sorted(config.extracted_images_dir.glob("*.png"))
            if args.workers > 1 and len(page_images) > 1:
                with tqdm(total=len(page_images), unit="page") as bar:
                    with ProcessPoolExecutor(max_workers=args.workers) as pool:
                        futures = {
                            pool.submit(
                                extract_cards_from_page,
                                page_path,
                                config,
                                args.force,
                                None,
                                (
                                    debug_base / page_path.stem
                                    if debug_base
                                    else None
                                ),
                            ): page_path
                            for page_path in page_images
                        }
                        for future in as_completed(futures):
                            bar.desc = f"Cards {futures[future].stem}"
                            future.result()
                            bar.update(1)
            else:
                with tqdm(page_images, unit="page") as bar:
                    for page_path in bar:
                        bar.desc = f"Cards {page_path.stem}"
                        debug_dir = (
                            debug_base / page_path.stem if debug_base else None
                        )
                        extract_cards_from_page(
                            page_path,
                            config,
                            force=args.force,
                            debug_dir=debug_dir,
                        )
            elapsed = time.perf_counter() - t0
            print(
                f"[timing] Cards: {elapsed:.1f}s"
                f" ({len(page_images)} pages,"
                f" {args.workers} workers)"
            )

        if run_ocr:
            t0 = time.perf_counter()
            card_images = sorted(config.extracted_cards_dir.glob("*.png"))
            if not args.force:
                card_images = [
                    p
                    for p in card_images
                    if not (config.ocr_output_dir / f"{p.stem}.txt").exists()
                ]

            with tqdm(card_images, unit="card") as bar:
                for card_path in bar:
                    bar.desc = f"OCR {card_path.stem}"
                    ocr_card(card_path, config, force=args.force)

            elapsed = time.perf_counter() - t0
            print(f"[timing] OCR: {elapsed:.1f}s ({len(card_images)} cards)")


if __name__ == "__main__":
    main()

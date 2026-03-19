"""Seeded deterministic subset selection for incremental development."""

import random
from dataclasses import dataclass

import fitz

from .config import Config


@dataclass(frozen=True)
class SubsetSelection:
    """Maps volume stems to lists of selected page indices (0-based)."""

    volumes: dict[str, list[int]]  # e.g. {"05": [3, 17]}
    num_cards: int  # per-page cap applied after detection


def select_subset(config: Config) -> SubsetSelection:
    rng = random.Random(config.seed)

    pdf_files = sorted(config.raw_cat_path.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {config.raw_cat_path}")

    selected_pdfs = rng.sample(pdf_files, min(config.num_pdf, len(pdf_files)))
    selected_pdfs.sort()

    volumes: dict[str, list[int]] = {}
    for pdf_path in selected_pdfs:
        volume = pdf_path.stem  # e.g. "05"
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        doc.close()

        pages = rng.sample(
            range(page_count), min(config.num_pages, page_count)
        )
        pages.sort()
        volumes[volume] = pages

    return SubsetSelection(volumes=volumes, num_cards=config.num_cards)

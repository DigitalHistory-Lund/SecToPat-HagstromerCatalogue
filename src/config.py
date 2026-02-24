"""Load .env and expose a frozen Config dataclass."""

from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    raw_cat_path: Path
    extracted_images_dir: Path
    extracted_cards_dir: Path
    seed: int
    num_pdf: int
    num_pages: int
    num_cards: int


def load_config() -> Config:
    load_dotenv(PROJECT_ROOT / ".env")

    raw = os.environ.get("RAW_CAT_PATH")
    if not raw:
        raise RuntimeError("RAW_CAT_PATH must be set in .env")

    return Config(
        raw_cat_path=Path(raw),
        extracted_images_dir=PROJECT_ROOT / os.environ.get("EXTRACTED_IMAGES_DIR", "extracted_images"),
        extracted_cards_dir=PROJECT_ROOT / os.environ.get("EXTRACTED_CARDS_DIR", "extracted_cards"),
        seed=int(os.environ.get("SEED", "1")),
        num_pdf=int(os.environ.get("NUM_PDF", "1")),
        num_pages=int(os.environ.get("NUM_PAGES", "1")),
        num_cards=int(os.environ.get("NUM_CARDS", "1")),
    )

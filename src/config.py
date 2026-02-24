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
    ocr_output_dir: Path
    reader_dir: Path
    ocr_model: str
    seed: int
    num_pdf: int
    num_pages: int
    num_cards: int
    # Background classification
    cv_bg_border: int
    cv_bg_threshold: int
    # Light background: Canny + dilation
    cv_light_blur_size: int
    cv_light_canny_low: int
    cv_light_canny_high: int
    cv_light_dilate_size: int
    cv_light_dilate_iter: int
    # Dark background: binary threshold + morphology close
    cv_dark_thresh_value: int
    cv_dark_close_size: int
    cv_dark_close_iter: int
    # Contour filtering
    cv_min_area_ratio: float
    cv_max_area_ratio: float
    cv_min_dim: int
    cv_min_aspect: float
    cv_max_aspect: float
    cv_min_rectangularity: float
    cv_row_threshold: float


def load_config() -> Config:
    load_dotenv(PROJECT_ROOT / ".env")

    raw = os.environ.get("RAW_CAT_PATH")
    if not raw:
        raise RuntimeError("RAW_CAT_PATH must be set in .env")

    return Config(
        raw_cat_path=Path(raw),
        extracted_images_dir=PROJECT_ROOT / os.environ.get("EXTRACTED_IMAGES_DIR", "extracted_images"),
        extracted_cards_dir=PROJECT_ROOT / os.environ.get("EXTRACTED_CARDS_DIR", "extracted_cards"),
        ocr_output_dir=PROJECT_ROOT / os.environ.get("OCR_OUTPUT_DIR", "ocr_output"),
        reader_dir=PROJECT_ROOT / os.environ.get("READER_DIR", "reader"),
        ocr_model=os.environ.get("OCR_MODEL", "qwen3-vl:2b"),
        seed=int(os.environ.get("SEED", "1")),
        num_pdf=int(os.environ.get("NUM_PDF", "1")),
        num_pages=int(os.environ.get("NUM_PAGES", "1")),
        num_cards=int(os.environ.get("NUM_CARDS", "1")),
        # Background classification
        cv_bg_border=int(os.environ.get("CV_BG_BORDER", "20")),
        cv_bg_threshold=int(os.environ.get("CV_BG_THRESHOLD", "128")),
        # Light background
        cv_light_blur_size=int(os.environ.get("CV_LIGHT_BLUR_SIZE", "5")),
        cv_light_canny_low=int(os.environ.get("CV_LIGHT_CANNY_LOW", "30")),
        cv_light_canny_high=int(os.environ.get("CV_LIGHT_CANNY_HIGH", "100")),
        cv_light_dilate_size=int(os.environ.get("CV_LIGHT_DILATE_SIZE", "25")),
        cv_light_dilate_iter=int(os.environ.get("CV_LIGHT_DILATE_ITER", "3")),
        # Dark background
        cv_dark_thresh_value=int(os.environ.get("CV_DARK_THRESH_VALUE", "160")),
        cv_dark_close_size=int(os.environ.get("CV_DARK_CLOSE_SIZE", "15")),
        cv_dark_close_iter=int(os.environ.get("CV_DARK_CLOSE_ITER", "2")),
        # Contour filtering
        cv_min_area_ratio=float(os.environ.get("CV_MIN_AREA_RATIO", "0.02")),
        cv_max_area_ratio=float(os.environ.get("CV_MAX_AREA_RATIO", "0.40")),
        cv_min_dim=int(os.environ.get("CV_MIN_DIM", "200")),
        cv_min_aspect=float(os.environ.get("CV_MIN_ASPECT", "1.0")),
        cv_max_aspect=float(os.environ.get("CV_MAX_ASPECT", "3.0")),
        cv_min_rectangularity=float(os.environ.get("CV_MIN_RECTANGULARITY", "0.60")),
        cv_row_threshold=float(os.environ.get("CV_ROW_THRESHOLD", "0.05")),
    )

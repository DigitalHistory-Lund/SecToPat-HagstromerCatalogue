"""Patch missing Ms codes into transcriptions.

Two-step process:
1. ocr_corners: Crop top-right corners and OCR them, saving raw
   text to ocr_output_corners/.
2. patch_ms_codes: Read the corner OCR output and prepend validated
   Ms codes to transcriptions/ in place.
"""

import re
from pathlib import Path

import cv2
import httpx
import ollama
from tqdm import tqdm

from .config import PROJECT_ROOT, Config

MS_CODE_LINE_RE = re.compile(r"^(\d+\.\s+)?(I:\s*)?Ms\s?\d+", re.IGNORECASE)
MS_CODE_EXTRACT_RE = re.compile(
    r"(?:I:\s*)?Ms\s?\d+(?:[:\s]\S*)*", re.IGNORECASE
)

CORNERS_DIR = PROJECT_ROOT / "extracted_corners"
OCR_CORNERS_DIR = PROJECT_ROOT / "ocr_output_corners"


def has_ms_code(text: str) -> bool:
    """Return True if the first line already contains an Ms code."""
    first_line = text.split("\n", 1)[0]
    return bool(MS_CODE_LINE_RE.match(first_line))


def crop_top_right(
    card_path: Path, out_dir: Path, force: bool = False
) -> Path | None:
    """Crop the top-right corner of a card image and save it.

    Returns the path to the saved crop, or None on failure.
    """
    out_path = out_dir / f"{card_path.stem}.png"
    if out_path.exists() and not force:
        return out_path

    img = cv2.imread(str(card_path))
    if img is None:
        return None

    h, w = img.shape[:2]
    top = 0
    bottom = int(h * 0.20)
    left = int(w * 0.45)
    right = w

    crop = img[top:bottom, left:right]
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), crop)
    return out_path


def ocr_crop(crop_path: Path, model: str, timeout: int = 120) -> str:
    """OCR a cropped corner image, returning raw text."""
    client = ollama.Client(timeout=timeout)
    response = client.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Extract all text from this image.",
                "images": [str(crop_path)],
            },
        ],
    )
    return response.message.content.strip()


def extract_ms_code(ocr_text: str) -> str | None:
    """Extract and validate an Ms code from OCR output.

    Returns the cleaned code string, or None if no valid code found.
    """
    if "NONE" in ocr_text.upper():
        return None
    m = MS_CODE_EXTRACT_RE.search(ocr_text)
    return m.group(0) if m else None


def ocr_corners(config: Config, force: bool = False) -> None:
    """Crop top-right corners and OCR them, saving raw output
    to ocr_output_corners/."""
    transcription_files = sorted(config.transcriptions_dir.glob("*.txt"))
    if not transcription_files:
        print("No transcription files found.")
        return

    CORNERS_DIR.mkdir(parents=True, exist_ok=True)
    OCR_CORNERS_DIR.mkdir(parents=True, exist_ok=True)

    skipped_has_code = 0
    skipped_no_image = 0
    skipped_timeout = 0
    ocr_done = 0
    failures: list[str] = []

    with tqdm(transcription_files, unit="card") as bar:
        for txt_path in bar:
            bar.desc = f"Corner {txt_path.stem}"
            text = txt_path.read_text(encoding="utf-8")

            if has_ms_code(text):
                skipped_has_code += 1
                continue

            ocr_out = OCR_CORNERS_DIR / f"{txt_path.stem}.txt"
            if ocr_out.exists() and not force:
                ocr_done += 1
                continue

            card_path = config.extracted_cards_dir / f"{txt_path.stem}.png"
            if not card_path.exists():
                skipped_no_image += 1
                continue

            crop_path = crop_top_right(card_path, CORNERS_DIR, force=force)
            if crop_path is None:
                skipped_no_image += 1
                continue

            try:
                raw = ocr_crop(crop_path, config.ocr_model)
            except httpx.TimeoutException:
                tqdm.write(f"Timeout: {txt_path.stem}")
                failures.append(txt_path.stem)
                skipped_timeout += 1
                continue
            ocr_out.write_text(raw, encoding="utf-8")
            ocr_done += 1

    if failures:
        log_path = OCR_CORNERS_DIR / "failures.log"
        log_path.write_text("\n".join(failures) + "\n", encoding="utf-8")
        print(f"Failures logged to {log_path}")

    print(
        f"Done. {ocr_done} corners OCR'd, "
        f"{skipped_has_code} already had code, "
        f"{skipped_no_image} no image, "
        f"{skipped_timeout} timed out."
    )


def patch_ms_codes(config: Config, force: bool = False) -> None:
    """Read corner OCR output from ocr_output_corners/ and prepend
    validated Ms codes to transcriptions/ in place."""
    ocr_files = sorted(OCR_CORNERS_DIR.glob("*.txt"))
    if not ocr_files:
        print("No corner OCR output found. " "Run --step ocr-corners first.")
        return

    skipped_has_code = 0
    skipped_no_ms = 0
    patched = 0

    for ocr_file in tqdm(ocr_files, unit="card"):
        txt_path = config.transcriptions_dir / ocr_file.name
        if not txt_path.exists():
            continue

        text = txt_path.read_text(encoding="utf-8")
        if has_ms_code(text) and not force:
            skipped_has_code += 1
            continue

        raw = ocr_file.read_text(encoding="utf-8").strip()
        ms_code = extract_ms_code(raw)

        if ms_code is None:
            skipped_no_ms += 1
            continue

        patched_text = ms_code + "\n" + text
        txt_path.write_text(patched_text, encoding="utf-8")
        patched += 1

    print(
        f"Done. {patched} patched, "
        f"{skipped_has_code} already had code, "
        f"{skipped_no_ms} no Ms code found."
    )

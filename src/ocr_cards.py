"""OCR catalogue card images using a local ollama VLM."""

import argparse
from pathlib import Path

import ollama

from tqdm import tqdm

from .config import Config, load_config


def ocr_card(card_path: Path, config: Config, force: bool = False) -> Path:
    """OCR a single card image and write the text to a .txt file.

    Returns the path to the output .txt file.
    """
    config.ocr_output_dir.mkdir(parents=True, exist_ok=True)

    out_path = config.ocr_output_dir / f"{card_path.stem}.txt"

    if out_path.exists() and not force:
        return out_path

    response = ollama.chat(
        model=config.ocr_model,
        messages=[
            {
                "role": "user",
                "content": "Extract all text from this image.",
                "images": [str(card_path)],
            },
        ],
    )

    text = response.message.content
    out_path.write_text(text, encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR card images using ollama VLM")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    config = load_config()
    card_images = sorted(config.extracted_cards_dir.glob("*.png"))

    if not card_images:
        print("No card images found. Run extract_cards first.")
        return

    if not args.force:
        card_images = [p for p in card_images if not (config.ocr_output_dir / f"{p.stem}.txt").exists()]

    with tqdm(card_images, unit="card") as bar:
        for card_path in bar:
            bar.desc = f"OCR {card_path.stem}"
            ocr_card(card_path, config, force=args.force)


if __name__ == "__main__":
    main()

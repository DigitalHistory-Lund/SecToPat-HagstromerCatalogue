import re

from .config import PROJECT_ROOT, load_config

cfg = load_config()

OCR_CORNERS_DIR = PROJECT_ROOT / "ocr_output_corners"


def has_ms_line(lines):
    """Check if line 1 or 2 starts with 'Ms'."""
    for line in lines[:2]:
        if line.startswith("Ms"):
            return True
    return False


def insert_ms_code(stem, lines):
    """If a corner OCR file exists and starts with 'Ms',
    prepend it as line 1."""
    ocr_file = OCR_CORNERS_DIR / f"{stem}.txt"
    if not ocr_file.exists():
        return lines
    code = ocr_file.read_text(encoding="utf-8").strip()
    if code.startswith("Ms"):
        return [code] + lines
    return lines


def clean_lines(lines):
    for idx, line in enumerate(lines):
        if line.startswith("Ms"):
            continue

        starts_with_plus = line.strip().startswith("+")

        if starts_with_plus:
            line = line.strip()[1:].strip()

        if "," in line and idx < 2:
            first, *rest = line.split(",")

            if not first.count(" "):
                continue

            first = re.sub(
                r"(?<=[A-Za-zÀ-ÖØ-öø-ÿ])\s(?<!=[a-zà-öø-ÿ])", "", first
            )
            first = re.sub(
                r"(?<=[A-Za-zÀ-ÖØ-öø-ÿ]),(?<!=[a-zà-öø-ÿ])", ", ", first
            )
            first = re.sub(r"ae", "æ", first)
            first = re.sub(r"AE", "Æ", first)
            # assemble

            line = ",".join(([first] + rest))

        if starts_with_plus:
            line = "+ " + line
        if "- " in line:
            print(line)
            line = re.sub(r"-\s", "", line)
            print(line)
        yield line


for card in cfg.transcriptions_dir.glob("*.txt"):
    raw_content = card.read_text(encoding="utf-8")
    lines = raw_content.splitlines()

    if not has_ms_line(lines):
        lines = insert_ms_code(card.stem, lines)

    merged = "\n".join(clean_lines(lines))
    if merged != raw_content:
        print(f"Updating {card} …")
        card.write_text(merged, encoding="utf-8")

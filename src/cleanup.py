import re

from .config import load_config

cfg = load_config()


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
    clean_lines(lines)

    merged = "\n".join(clean_lines(lines))
    if merged != raw_content:
        print(f"Updating {card} …")
        card.write_text(merged, encoding="utf-8")

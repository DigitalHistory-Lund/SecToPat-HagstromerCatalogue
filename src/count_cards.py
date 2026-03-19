import re
from collections import Counter

from tqdm import tqdm

from .config import load_config

config = load_config()

extracted_cards_dir = config.extracted_cards_dir

cntr = Counter()

for card_path in tqdm(extracted_cards_dir.glob("*.png")):
    stem = card_path.stem
    parts = stem.split("_")
    if len(parts) < 4:
        print(f"Skipping {card_path} with unexpected name format")
        continue
    volume, page_str, col_str, row_str = parts[:4]
    cntr[(volume, page_str)] += 1


for page, count in cntr.most_common():
    if count != 8:
        volume, page_str = re.match(
            r"(\d{2})_(\d{4})", "_".join(page)
        ).groups()
        print(f"{volume}.pdf p.{int(page_str)}\t{count} cards")

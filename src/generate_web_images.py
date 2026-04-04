"""Generate web-optimized JPEG card images for the Quarto site."""

import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
from tqdm import tqdm

from .config import Config, PROJECT_ROOT

WEB_IMG_WIDTH = 800
WEB_IMG_QUALITY = 80


def downscale_card(src: Path, dst: Path) -> Path:
    """Read a PNG card, downscale to JPEG, write to *dst*."""
    img = cv2.imread(str(src))
    if img is None:
        warnings.warn(f"Could not read {src}, skipping")
        return dst

    h, w = img.shape[:2]
    if w > WEB_IMG_WIDTH:
        aspect = w / h
        new_w = WEB_IMG_WIDTH
        new_h = int(new_w / aspect)
        img = cv2.resize(
            img, (new_w, new_h), interpolation=cv2.INTER_AREA
        )

    ok, buf = cv2.imencode(
        ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, WEB_IMG_QUALITY]
    )
    if not ok:
        warnings.warn(f"Failed to encode {src}, skipping")
        return dst

    dst.write_bytes(bytes(buf))
    return dst


def generate_web_images(
    config: Config,
    force: bool = False,
    workers: int = 1,
) -> None:
    """Downscale all extracted cards to web-friendly JPEGs."""
    src_dir = config.extracted_cards_dir
    dst_dir = config.cards_web_dir
    dst_dir.mkdir(exist_ok=True)

    sources = sorted(src_dir.glob("*.png"))
    if not sources:
        print("No card PNGs found — nothing to convert.")
        return

    # Filter to cards that need (re)generation
    pending: list[tuple[Path, Path]] = []
    for src in sources:
        dst = dst_dir / f"{src.stem}.jpg"
        if not force and dst.exists():
            if dst.stat().st_mtime >= src.stat().st_mtime:
                continue
        pending.append((src, dst))

    if not pending:
        print("All web images up to date.")
        return

    t0 = time.perf_counter()

    if workers > 1 and len(pending) > 1:
        with tqdm(total=len(pending), unit="card") as bar:
            with ProcessPoolExecutor(
                max_workers=workers
            ) as pool:
                futures = {
                    pool.submit(downscale_card, s, d): s
                    for s, d in pending
                }
                for future in as_completed(futures):
                    future.result()
                    bar.desc = f"Web {futures[future].stem}"
                    bar.update(1)
    else:
        with tqdm(pending, unit="card") as bar:
            for src, dst in bar:
                bar.desc = f"Web {src.stem}"
                downscale_card(src, dst)

    elapsed = time.perf_counter() - t0
    print(
        f"[timing] Web images: {elapsed:.1f}s"
        f" ({len(pending)} cards, {workers} workers)"
    )

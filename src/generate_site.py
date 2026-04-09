"""Generate Quarto website .qmd files from transcriptions."""

import json
import shutil
import tomllib
from collections import defaultdict
from datetime import date
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTIONS_DIR = PROJECT_ROOT / "transcriptions"
METADATA_PATH = PROJECT_ROOT / "metadata.json"
SITE_DIR = PROJECT_ROOT / "site"
INDEX_PATH = PROJECT_ROOT / "index.qmd"
BUILD_INFO_PATH = PROJECT_ROOT / "_build_info.yml"
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"

CHUNK_SIZE = 10


def load_metadata() -> dict:
    """Load project metadata from metadata.json."""
    with open(METADATA_PATH) as f:
        return json.load(f)


def parse_stem(stem: str) -> tuple[str, str, str, str]:
    """Parse '05_0001_0_2' into (vol, page, col, row)."""
    vol, page, col, row = stem.split("_")
    return vol, page, col, row


def discover_structure(
    transcriptions_dir: Path,
) -> dict[str, dict[str, list[tuple[str, str, str]]]]:
    """Scan transcriptions/ to build {vol: {page: [(col, row, stem)]}}."""
    txt_files = sorted(transcriptions_dir.glob("*.txt"))
    volumes: dict[str, dict[str, list[tuple[str, str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for txt_path in txt_files:
        stem = txt_path.stem
        vol, page, col, row = parse_stem(stem)
        volumes[vol][page].append((col, row, stem))
    return volumes


def chunk_pages(
    sorted_page_keys: list[str],
) -> list[list[str]]:
    """Split sorted page keys into chunks of CHUNK_SIZE."""
    return [
        sorted_page_keys[i : i + CHUNK_SIZE]
        for i in range(0, len(sorted_page_keys), CHUNK_SIZE)
    ]


def chunk_filename(page_keys: list[str]) -> str:
    """Return filename stem for a chunk, e.g. '0000-0009'."""
    if len(page_keys) == 1:
        return page_keys[0]
    return f"{page_keys[0]}_{page_keys[-1]}"


def chunk_label(page_keys: list[str]) -> str:
    """Return display label for a chunk, e.g. 'Pages 0000–0009'."""
    if len(page_keys) == 1:
        return f"Page {page_keys[0]}"
    return f"Pages {page_keys[0]}–{page_keys[-1]}"


def render_index_qmd(metadata: dict, volumes: dict) -> str:
    """Generate root index.qmd content."""
    lines = [
        "---",
        f'title: "{metadata["title"]}"',
        "---",
        "",
    ]
    short = metadata.get("summary_short")
    if short:
        lines.append(short)
        lines.append("")
    else:
        for paragraph in metadata.get("summary", []):
            lines.append(paragraph)
            lines.append("")

    doi = metadata.get("doi", "")
    if doi:
        doi_url = f"https://doi.org/{doi}"
        lines.append(
            f"[![DOI](https://zenodo.org/badge/DOI/{doi}.svg)]" f"({doi_url})"
        )
        lines.append("")

    lib_url = metadata.get("library_url", "")
    repo_url = metadata.get("repo_url", "")
    if lib_url:
        lines.append(f"[About the Hagströmer Library]({lib_url})")
        lines.append("")
    if repo_url:
        lines.append(f"[GitHub repository]({repo_url})")
        lines.append("")
        lines.append(f"[Download latest PDF version]({repo_url}/releases)")
        lines.append("")

    lines.append(
        "[![CC BY-NC 4.0]"
        "(https://licensebuttons.net/l/by-nc/4.0/88x31.png)]"
        "(https://creativecommons.org/licenses/by-nc/4.0/)"
    )
    lines.append("")
    lines.append(
        "This work is licensed under a"
        " [Creative Commons Attribution-NonCommercial"
        " 4.0 International License]"
        "(https://creativecommons.org/licenses/by-nc/4.0/)"
        " (CC BY-NC 4.0)."
    )
    lines.append("")
    lines.append("## Volumes")
    lines.append("")
    lines.append("| Volume | Pages | Cards |")
    lines.append("|--------|------:|------:|")
    for vol in sorted(volumes):
        pages = volumes[vol]
        n_pages = len(pages)
        n_cards = sum(len(cards) for cards in pages.values())
        lines.append(
            f"| [Volume {vol}](site/{vol}/index.qmd)"
            f" | {n_pages} | {n_cards} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_volume_index(
    vol: str,
    pages: dict[str, list[tuple[str, str, str]]],
    source_url: str = "",
) -> str:
    """Generate site/{vol}/index.qmd with chunk listing."""
    n_cards = sum(len(cards) for cards in pages.values())
    chunks = chunk_pages(sorted(pages))
    lines = [
        "---",
        f'title: "Volume {vol}"',
        "---",
        "",
        f"{len(pages)} pages, {n_cards} cards.",
        "",
    ]
    if source_url:
        lines.append(f"[Original PDF on ALVIN]({source_url})")
        lines.append("")
    lines += [
        "| Pages | Cards |",
        "|-------|------:|",
    ]
    for ch in chunks:
        n = sum(len(pages[p]) for p in ch)
        fname = chunk_filename(ch)
        label = chunk_label(ch)
        lines.append(f"| [{label}]({fname}.qmd) | {n} |")
    lines.append("")
    return "\n".join(lines)


def render_chunk_qmd(
    vol: str,
    page_keys: list[str],
    pages: dict[str, list[tuple[str, str, str]]],
    prev_chunk: list[str] | None,
    next_chunk: list[str] | None,
    transcriptions_dir: Path,
    repo_url: str,
) -> str:
    """Generate .qmd for a chunk of pages."""
    label = chunk_label(page_keys)
    lines = [
        "---",
        f'title: "Volume {vol}, {label}"',
        "---",
        "",
    ]

    # Prev/next navigation
    nav_parts = []
    if prev_chunk:
        prev_label = chunk_label(prev_chunk)
        prev_fname = chunk_filename(prev_chunk)
        nav_parts.append(f"[← {prev_label}]({prev_fname}.qmd)")
    else:
        nav_parts.append("")
    if next_chunk:
        next_label = chunk_label(next_chunk)
        next_fname = chunk_filename(next_chunk)
        nav_parts.append(f"[{next_label} →]({next_fname}.qmd)")
    else:
        nav_parts.append("")
    lines.append(
        "::: {.page-nav}\n"
        + nav_parts[0]
        + " &nbsp; "
        + nav_parts[1]
        + "\n:::"
    )
    lines.append("")

    # Render each page in the chunk
    for page in page_keys:
        lines.append(f"## Page {page}")
        lines.append("")
        sorted_cards = sorted(pages[page])
        for col, row, stem in sorted_cards:
            txt_path = transcriptions_dir / f"{stem}.txt"
            text = txt_path.read_text().rstrip() if txt_path.exists() else ""
            img_path = f"../../cards_web/{stem}.jpg"
            edit_url = f"{repo_url}/edit/main/transcriptions/" f"{stem}.txt"

            lines.append("::: {.card-entry}")
            lines.append(":::: {.grid}")
            lines.append("::::: {.g-col-12 .g-col-md-6}")
            lines.append(f"![]({img_path}){{width=100%}}\n\n##### {stem}")
            lines.append(":::::")
            lines.append("::::: {.g-col-12 .g-col-md-6}")
            lines.append("")
            lines.append("```")
            lines.append(text)
            lines.append("```")
            lines.append("")
            lines.append(
                "[Suggest improved transcription]"
                f"({edit_url})"
                "{.btn .btn-sm .btn-outline-secondary}"
            )
            lines.append(":::::")
            lines.append("::::")
            lines.append(":::")
            lines.append("")

    # Bottom navigation
    lines.append(
        "::: {.page-nav}\n"
        + nav_parts[0]
        + " &nbsp; "
        + nav_parts[1]
        + "\n:::"
    )
    lines.append("")
    return "\n".join(lines)


def write_build_info() -> None:
    """Write _build_info.yml with version and today's date."""
    with open(PYPROJECT_PATH, "rb") as f:
        version = tomllib.load(f)["project"]["version"]
    today = date.today().isoformat()
    BUILD_INFO_PATH.write_text(
        "website:\n" "  page-footer:\n" f'    center: "v{version} | {today}"\n'
    )
    print(f"Wrote {BUILD_INFO_PATH}")


def generate_site() -> None:
    """Generate all .qmd files for the Quarto catalogue site."""
    write_build_info()
    metadata = load_metadata()
    repo_url = metadata.get("repo_url", "")
    volumes = discover_structure(TRANSCRIPTIONS_DIR)

    if not volumes:
        print("No transcription files found — nothing to generate.")
        return

    # Clean and recreate site/
    if SITE_DIR.exists():
        shutil.rmtree(SITE_DIR)
    SITE_DIR.mkdir()

    # Write root index.qmd
    INDEX_PATH.write_text(render_index_qmd(metadata, volumes))
    print(f"Wrote {INDEX_PATH}")

    # Generate per-volume files
    total_chunks = 0
    for vol in volumes:
        total_chunks += len(chunk_pages(sorted(volumes[vol])))

    with tqdm(total=total_chunks, unit="chunk") as bar:
        for vol in sorted(volumes):
            pages = volumes[vol]
            vol_dir = SITE_DIR / vol
            vol_dir.mkdir()

            # Volume index
            vol_meta = metadata.get("volumes", {}).get(vol, {})
            source_url = vol_meta.get("source_url", "")
            vol_index = render_volume_index(vol, pages, source_url)
            (vol_dir / "index.qmd").write_text(vol_index)

            # Chunk files
            chunks = chunk_pages(sorted(pages))
            for i, ch in enumerate(chunks):
                label = chunk_label(ch)
                bar.desc = f"Vol {vol} {label}"
                prev_ch = chunks[i - 1] if i > 0 else None
                next_ch = chunks[i + 1] if i < len(chunks) - 1 else None
                content = render_chunk_qmd(
                    vol,
                    ch,
                    pages,
                    prev_ch,
                    next_ch,
                    TRANSCRIPTIONS_DIR,
                    repo_url,
                )
                fname = chunk_filename(ch)
                (vol_dir / f"{fname}.qmd").write_text(content)
                bar.update(1)

    print(
        f"Generated {total_chunks} chunk files"
        f" across {len(volumes)} volumes."
    )


if __name__ == "__main__":
    generate_site()

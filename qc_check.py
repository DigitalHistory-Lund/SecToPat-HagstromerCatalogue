"""Check that each marker in qc.txt has exactly one match
among the first lines of ./transcriptions/*.txt.

Three-pass matching (exact → no-spaces → normalized) with
fuzzy character equivalences for OCR errors.
"""

from pathlib import Path


def load_markers(path):
    lines = Path(path).read_text().splitlines()
    return [line.strip() for line in lines if line.strip()]


def load_first_lines(directory):
    result = {}
    for p in sorted(Path(directory).glob("*.txt")):
        first = p.read_text().split("\n", 1)[0].strip()
        result[p.name] = first
    return result


def strip_spaces(s):
    return s.replace(" ", "")


def normalize(s):
    s = strip_spaces(s).lower()
    for ch in "oO0":
        s = s.replace(ch, "_round_")
    for ch in "l1I":
        s = s.replace(ch, "_stick_")
    return s


def match_markers(markers, first_lines):
    for marker in markers:
        print(marker)

        # Pass 1 — exact
        hits = [fn for fn, fl in first_lines.items() if fl == marker]
        if len(hits) == 1:
            print(f"  PASS 1 (exact): {hits[0]}")
            print()
            continue
        if len(hits) > 1:
            print("  PASS 1 (exact), multiple matches:")
            for fn in hits:
                print(f'    {fn}  →  "{first_lines[fn]}"')
            print()
            continue

        # Pass 2 — no spaces
        m_ns = strip_spaces(marker)
        hits = [
            fn for fn, fl in first_lines.items() if strip_spaces(fl) == m_ns
        ]
        if len(hits) == 1:
            print(
                f"  PASS 2 (no spaces): {hits[0]}"
                f'  →  "{first_lines[hits[0]]}"'
            )
            print()
            continue
        if len(hits) > 1:
            print("  PASS 2 (no spaces), multiple matches:")
            for fn in hits:
                print(f'    {fn}  →  "{first_lines[fn]}"')
            print()
            continue

        # Pass 3 — normalized
        m_norm = normalize(marker)
        hits = [
            fn for fn, fl in first_lines.items() if normalize(fl) == m_norm
        ]
        if len(hits) == 1:
            print(
                f"  PASS 3 (normalized): {hits[0]}"
                f'  →  "{first_lines[hits[0]]}"'
            )
            print()
            continue
        if len(hits) > 1:
            print("  PASS 3 (normalized), multiple matches:")
            for fn in hits:
                print(f'    {fn}  →  "{first_lines[fn]}"')
            print()
            continue

        # No match
        print("  NO MATCH")
        print()


if __name__ == "__main__":
    markers = load_markers("qc.txt")
    first_lines = load_first_lines("transcriptions")
    match_markers(markers, first_lines)

# Experiment: Lower render resolution (0.5x) vs OCR quality

**Date:** 2025-02-24
**Result:** 0.5x degrades OCR quality significantly. Keep full resolution.

## Motivation

Page rendering is the main bottleneck (`--all` takes minutes). The PDF mediaboxes match scanner pixel dimensions (3626x4691 = 17MP), and `get_pixmap()` with no matrix renders at 1:1. Rendering at 0.5x would quarter the pixel count and roughly 4x the speed, but may degrade OCR quality.

## Method

- **Parameters:** seed=1, 5 PDFs, 23 pages, 31 cards OCR'd before stopping
- **Baseline (scale=1.0):** default pipeline → `extracted_images/`, `extracted_cards/`, `ocr_output/`
- **Experiment (scale=0.5):** lower-res rendering → `extracted_images_05x/`, `extracted_cards_05x/`, `ocr_output_05x/`
- **Comparison:** `diff` on the 31 overlapping OCR text files
- **OCR model:** qwen3-vl:2b via Ollama

## Rendering speed

At 0.5x, pages rendered at ~1.94 pages/s. Card detection ran at ~9.5 pages/s. Both steps are fast enough that they're not the bottleneck — OCR is (~7-10s per card).

## Results

### No quality impact (cosmetic only)

Several files differed only in trailing whitespace or were identical. These are not meaningful differences.

### Systematic reordering

In 7 of 31 cards, the 0.5x model moved the manuscript reference number (e.g. "Ms 26:66") from the first line to the last line. This is a consistent behavioral change — the model reads the card in a different order at lower resolution.

Affected files: `03_0006_0_1`, `03_0006_1_1`, `03_0010_0_1`, `03_0010_0_3`, `03_0010_1_2`, `03_0014_0_0`, `03_0014_1_2`.

### OCR quality degradation

| Baseline (1.0x) | 0.5x | Issue |
|---|---|---|
| `Denke` | `Den eke` | Name split with spaces (multiple cards) |
| `Drake, Hans Leon` | `D r a k e, Hans Leon` | Spaced out |
| `Drufva, Gerhard` | `D r u f v a, Gerhard` | Spaced out |
| `De vaux, Jean` | `D ev a u x, Jean` | Spaced out |
| `Dichman, Joh` | `D ichman, Joh` | Spaced out |
| `Bäck` | `Böck` | Wrong diacritic |
| `Förtekning` | `Förtekening` | Misspelling |
| `Sthlm` | `Stahl` | Abbreviation misread |
| `Hår ågs av Carl Lørich` | `Jour 1904 av Carl Torvich` | Completely garbled |
| `Ms har afgis av Earl doich` | `Ms har ågbs so Garl dori eh` | Even worse |
| `Fol.` | `Pol.` | First letter wrong |
| `4:o` | `410` | Format lost |
| `Lic. avh.` | `Lic.ahv.` | Abbreviation garbled |
| `Brev` | `Breiv` | Insertion error |
| `derer Praeparation` | `deiner Praeparation` | Word substitution |
| `Brev till Axel Key` | `Breiv till Axel Key` | Insertion error |
| `opérations` | `operations` | Lost diacritic |

The most common failure mode is **name splitting** — the model inserts spaces into surnames when reading lower-resolution text. Handwritten or faded text is also misread far more often, with some readings becoming completely unintelligible.

### Minor 0.5x improvements (rare)

Two cases where 0.5x produced arguably better output:

- `Dumrath, O\nH` → `Dumrath, O H` (correctly merged a split line)
- `Oförgripeligit` → `Oförgripeligt` (possibly more correct spelling)

These are too few and minor to offset the degradation.

## Conclusion

0.5x rendering is not viable for this corpus with the current OCR model. The speed gain in rendering (~4x) is irrelevant since OCR dominates runtime anyway (~7-10s per card vs <0.5s per page render). Full resolution (1.0x) should be kept as the default.

The `render_scale` config parameter remains available for future experiments with larger or more capable OCR models that might tolerate lower resolution.

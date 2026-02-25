# catalogue-hagstrom

Extraction pipeline for the Hagstrom manuscript catalogue. Converts scanned PDF
volumes into individual card images, runs OCR on each card, and produces
browsable output.

## Pipeline

```
PDF volumes ─► Page images ─► Card images ─► OCR text
                                                │
                                    ┌───────────┴───────────┐
                                    ▼                       ▼
                              Markdown reader         Card PDF
```

Each PDF page contains up to 8 catalogue cards arranged in a 2×4 grid.
The pipeline detects and extracts individual cards using OpenCV, then runs
OCR via a local Ollama vision-language model.

## Setup

```bash
cp .env.example .env
# Edit .env — set RAW_CAT_PATH to the directory containing the PDF volumes
pip install -r requirements.txt  # or use the venv
```

Requires [Ollama](https://ollama.com) running locally with the configured model
(default: `qwen3-vl:2b`).

## Usage

### Main pipeline

```bash
# Run full pipeline (subset controlled by .env)
python -m src

# Run all volumes
python -m src --all

# Run a single step
python -m src --step pages
python -m src --step cards
python -m src --step ocr

# Parallel processing
python -m src --workers 4

# Overwrite existing outputs
python -m src --force

# Save intermediate card-detection debug images
python -m src --debug
```

### Generate markdown reader

Produces a `reader/` directory with per-volume/per-page markdown files showing
each card image alongside its OCR text.

```bash
python generate_reader.py
```

### Generate card PDF

Creates a PDF with 4 cards per page — card image on the left, OCR text on
the right. Cards are ordered by filename (left column first, right column
second) to preserve the original page layout.

```bash
python generate_card_pdf.py [volume] [page]
python generate_card_pdf.py 01 0009
```

## Project structure

```
src/
  __main__.py        CLI orchestrator
  config.py          Configuration from .env
  extract_pages.py   PDF → page images (PyMuPDF)
  extract_cards.py   Page images → card images (OpenCV)
  ocr_cards.py       Card images → text (Ollama VLM)
  subset.py          Subset selection for development

generate_reader.py   Markdown reader generator
generate_card_pdf.py PDF card+text report generator

extracted_images/    Page images (PNG)
extracted_cards/     Individual card images (PNG)
ocr_output/          OCR text files (TXT, one per card)
reader/              Generated markdown reader

.env.example         Configuration template
```

## Naming convention

Files follow the pattern `{volume}_{page}_{column}_{row}`:

- `01_0009_0_2.png` — Volume 01, page 9, column 0, row 2

Columns 0–3 are the left half of the original page, columns 1–3 the right.
Within each column, rows run top to bottom.

## Configuration

All parameters are set via `.env`. Card detection thresholds (the `CV_*`
variables) control the OpenCV edge detection, morphology, and contour
filtering used to locate cards on each page. See `.env.example` for defaults.

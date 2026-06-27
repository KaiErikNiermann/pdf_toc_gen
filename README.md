# pdftoc

CLI tool to add table of contents bookmarks to PDFs.

## Installation

```bash
poetry install
```

## Usage

```bash
pdftoc --from source.pdf --to output.pdf
```

### Options

- `--from`, `-f`: Source PDF file (required)
- `--to`, `-t`: Output PDF file (required)
- `--skip-ocr`: Skip OCR even if PDF appears to need it
- `--force-ocr`: Force OCR even if PDF already has text
- `--lang`, `-l`: OCR language (default: `eng`)
- `--ocr-backend`: `auto` | `ocrmypdf` | `paddle` | `marker` (see below)
- `--verbose`, `-v`: Verbose output

## OCR backends

Select with `--ocr-backend`:

- `ocrmypdf` (tesseract) — lightweight, CPU, zero VRAM. The default floor.
- `paddle` (PaddleOCR PP-OCRv5) — classic pipeline, more accurate than tesseract,
  still CPU-viable (~23 pages/min). Best balance for large books on modest
  hardware. **Pin paddlepaddle 3.1.x** for the fast CPU path — see
  [docs/ocr-landscape-and-benchmarks.md](docs/ocr-landscape-and-benchmarks.md).
- `marker` (Surya) — highest quality, **GPU only**. `poetry install -E marker`.
- `auto` — marker if available, else ocrmypdf.

See [docs/ocr-landscape-and-benchmarks.md](docs/ocr-landscape-and-benchmarks.md)
for the full landscape survey + local benchmarks (incl. the paddle version
footgun and why VLM engines like Baidu Unlimited-OCR aren't a fit here).

## How it works

1. Checks if the PDF needs OCR (no extractable text)
2. Runs OCR using `ocrmypdf` if needed
3. Extracts table of contents entries from the PDF text
4. Adds bookmarks to the PDF based on detected TOC entries

## Requirements

- Tesseract OCR (for the `ocrmypdf` backend)

Install Tesseract:

```bash
# Debian/Ubuntu
apt install tesseract-ocr

# macOS
brew install tesseract

# Arch
pacman -S tesseract tesseract-data-eng
```

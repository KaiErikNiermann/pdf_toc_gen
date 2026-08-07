# pdftoc

CLI tool to add table of contents bookmarks to PDFs.

## Installation

For development:

```bash
poetry install
```

To install `pdftoc` as a global command:

```bash
./install.sh
```

The installer prefers pipx, and falls back to a self-managed venv in
`~/.local/share/pdftoc` symlinked into `~/.local/bin` — which works on
PEP 668 "externally managed" distributions (Arch, Debian 12+, Fedora 38+)
where `pip install --user` is refused. Set `PDFTOC_PYTHON` to pick the
interpreter, or `PDFTOC_PREFIX` to relocate the fallback venv.

## Usage

```bash
pdftoc --from source.pdf --to output.pdf
```

To bookmark a PDF where it sits, use `--in-place` instead of `--to`:

```bash
pdftoc --in-place --from book.pdf
pdftoc -if book.pdf              # same thing, short form
```

The source is only overwritten once processing succeeds; a failure part-way
through leaves the original file untouched.

> Note the flag order in the short form. `-f` takes a value, so `-fi book.pdf`
> reads as `--from=i` and fails — it has to be `-if book.pdf`.

### Options

- `--from`, `-f`: Source PDF file (required)
- `--to`, `-t`: Output PDF file (required unless `--in-place`)
- `--in-place`, `-i`: Rewrite the source PDF instead of writing to `--to`
- `--skip-ocr`: Skip OCR even if PDF appears to need it
- `--force-ocr`: Force OCR even if PDF already has text
- `--lang`, `-l`: OCR language (default: `eng`)
- `--ocr-backend`: `auto` | `ocrmypdf` | `paddle` | `marker` (see below)
- `--verbose`, `-v`: Verbose output
- `--version`, `-V`: Show version and build provenance, then exit

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

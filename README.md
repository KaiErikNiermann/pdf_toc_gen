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
- `--verbose`, `-v`: Verbose output

## How it works

1. Checks if the PDF needs OCR (no extractable text)
2. Runs OCR using `ocrmypdf` if needed
3. Extracts table of contents entries from the PDF text
4. Adds bookmarks to the PDF based on detected TOC entries

## Requirements

- Tesseract OCR (for OCR functionality)

Install Tesseract:

```bash
# Debian/Ubuntu
apt install tesseract-ocr

# macOS
brew install tesseract
```

"""OCR processing for PDFs."""

import subprocess
from pathlib import Path

import fitz  # type: ignore


def pdf_has_text(pdf_path: Path) -> bool:
    """Check if a PDF already has extractable text."""
    doc: fitz.Document = fitz.open(pdf_path)
    try:
        # Check first few pages for text
        pages_to_check = min(5, len(doc))
        total_text = 0
        for i in range(pages_to_check):
            page: fitz.Page = doc[i]
            text = page.get_text()
            total_text += len(text.strip())
        # If we have a reasonable amount of text, assume it's OCR'd
        return total_text > 100
    finally:
        doc.close()


def run_ocr(
    source: Path, output: Path, language: str, verbose: bool, optimize: int = 1
) -> None:
    """Run OCR on a PDF using ocrmypdf.

    Args:
        source: Input PDF path
        output: Output PDF path
        language: OCR language code
        verbose: Whether to show verbose output
        optimize: Optimization level 0-3 (2+ requires jbig2enc)
    """
    cmd = [
        "ocrmypdf",
        "--force-ocr",  # Force OCR on all pages, avoids Ghostscript issues
        "--output-type",
        "pdf",  # Avoid Ghostscript issues with certain versions
        "--optimize",
        str(optimize),
        "-l",
        language,
        str(source),
        str(output),
    ]

    if verbose:
        print(f"Running OCR: {' '.join(cmd)}")

    # Run with live output so user sees progress bar
    result = subprocess.run(cmd)
    if result.returncode != 0 and result.returncode != 6:
        # Return code 6 means "file already has text" which is fine
        # Don't add extra message - ocrmypdf already printed the error
        raise RuntimeError("OCR failed (see error above)")

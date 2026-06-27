"""OCR processing for PDFs."""

import subprocess
import tempfile
from pathlib import Path

import fitz  # type: ignore

from pdftoc.models import OcrBackend


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


def extract_text(
    pdf_path: Path,
    backend: OcrBackend = OcrBackend.AUTO,
    language: str = "eng",
    verbose: bool = False,
    optimize: int = 1,
) -> dict[int, str]:
    """Extract text from a PDF, returning {1-indexed page_num: text}.

    For marker backend: uses surya OCR directly via GPU, returns extracted text.
    For paddle backend: PaddleOCR PP-OCRv5 classic pipeline (CPU-viable).
    For ocrmypdf backend: creates a temp searchable PDF, then extracts text with pymupdf.
    """
    from pdftoc.marker_ocr import is_marker_available

    if backend == OcrBackend.PADDLE:
        from pdftoc.paddle_ocr import extract_text_with_paddle, is_paddle_available

        if not is_paddle_available():
            raise RuntimeError(
                "PaddleOCR is not installed. "
                "Install with: poetry install -E paddle"
            )
        return extract_text_with_paddle(
            pdf_path, verbose=verbose, language=language
        )

    use_marker = backend == OcrBackend.MARKER or (
        backend == OcrBackend.AUTO and is_marker_available()
    )

    if use_marker:
        if not is_marker_available():
            raise RuntimeError(
                "marker-pdf is not installed. "
                "Install with: poetry install -E marker"
            )
        from pdftoc.marker_ocr import extract_text_with_marker

        return extract_text_with_marker(pdf_path, verbose=verbose)

    # ocrmypdf path: create temp searchable PDF, extract text from it
    if verbose:
        print("Using ocrmypdf backend...")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        run_ocr(pdf_path, tmp_path, language, verbose, optimize)
        doc: fitz.Document = fitz.open(tmp_path)
        try:
            page_texts: dict[int, str] = {}
            for i in range(len(doc)):
                page: fitz.Page = doc[i]
                page_texts[i + 1] = page.get_text()
            return page_texts
        finally:
            doc.close()
    finally:
        tmp_path.unlink(missing_ok=True)

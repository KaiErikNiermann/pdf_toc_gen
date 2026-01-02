"""Core PDF processing logic."""

import shutil
import tempfile
from pathlib import Path

import fitz  # type: ignore

from pdftoc.bookmarks import add_bookmarks, get_existing_bookmarks, verify_bookmarks
from pdftoc.models import ExtractionMode, TocEntry
from pdftoc.ocr import pdf_has_text, run_ocr
from pdftoc.section_headers import extract_section_headers
from pdftoc.toc_extraction import extract_toc_from_text

# Re-export for backwards compatibility
__all__ = [
    "ExtractionMode",
    "TocEntry",
    "pdf_has_text",
    "run_ocr",
    "extract_toc_from_text",
    "extract_section_headers",
    "add_bookmarks",
    "get_existing_bookmarks",
    "verify_bookmarks",
    "process_pdf",
]


def process_pdf(
    source: Path,
    output: Path,
    skip_ocr: bool = False,
    force_ocr: bool = False,
    language: str = "eng",
    verbose: bool = False,
    optimize: int = 1,
    mode: ExtractionMode = ExtractionMode.AUTO,
    fix_bookmarks: bool = True,
) -> None:
    """Main processing function.

    Args:
        source: Input PDF path
        output: Output PDF path
        skip_ocr: Skip OCR even if PDF appears to need it
        force_ocr: Force OCR even if PDF already has text
        language: OCR language code
        verbose: Show verbose output
        optimize: OCR optimization level (0-3)
        mode: TOC extraction mode (auto, toc-page, section-headers)
        fix_bookmarks: If True, verify and fix incorrect existing bookmarks
    """
    print(f"Processing: {source}")

    # Step 0: Check existing bookmarks
    doc: fitz.Document = fitz.open(source)
    existing_bookmarks = get_existing_bookmarks(doc)
    doc.close()

    if existing_bookmarks and not force_ocr:
        print(f"PDF has {len(existing_bookmarks)} existing bookmark(s)")
        doc = fitz.open(source)
        is_valid, issues = verify_bookmarks(doc, existing_bookmarks, verbose)
        doc.close()

        if is_valid:
            print("Existing bookmarks appear correct, copying to output")
            # Just copy the file
            shutil.copy(source, output)
            print(f"Done! Output saved to: {output}")
            return
        elif fix_bookmarks:
            print("Existing bookmarks appear incorrect, will regenerate")
        else:
            print(
                "Warning: Existing bookmarks may be incorrect (use --fix to regenerate)"
            )

    # Step 1: Check if OCR is needed
    needs_ocr = not pdf_has_text(source)
    if force_ocr:
        needs_ocr = True
    if skip_ocr:
        needs_ocr = False

    if verbose:
        print(f"PDF has text: {pdf_has_text(source)}")
        print(f"Will run OCR: {needs_ocr}")

    # Step 2: Run OCR if needed
    if needs_ocr:
        print("Running OCR (this may take a while)...")
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            run_ocr(source, tmp_path, language, verbose, optimize)
            working_pdf = tmp_path
        except RuntimeError as e:
            print(f"Warning: OCR failed ({e}), continuing with original PDF")
            working_pdf = source
            tmp_path.unlink(missing_ok=True)
    else:
        print("PDF already has text, skipping OCR")
        working_pdf = source
        tmp_path = None  # type: ignore

    # Step 3: Extract TOC from the PDF
    print("Extracting table of contents...")
    doc = fitz.open(working_pdf)
    try:
        toc_entries: list[TocEntry] = []

        if mode == ExtractionMode.AUTO:
            # Try TOC page extraction first
            toc_entries = extract_toc_from_text(doc, verbose)
            # Fall back to section headers if no TOC found
            if not toc_entries:
                if verbose:
                    print("No TOC page found, trying section header extraction...")
                toc_entries = extract_section_headers(doc, verbose)
        elif mode == ExtractionMode.TOC_PAGE:
            toc_entries = extract_toc_from_text(doc, verbose)
        elif mode == ExtractionMode.SECTION_HEADERS:
            toc_entries = extract_section_headers(doc, verbose)
    finally:
        doc.close()

    if not toc_entries:
        print("Warning: No table of contents entries found in the PDF")
        print(
            "The PDF might not have a traditional TOC, or the format is not recognized"
        )

    # Step 4: Add bookmarks
    print("Adding bookmarks...")
    add_bookmarks(working_pdf, toc_entries, output, verbose)

    # Cleanup temp file
    if needs_ocr and tmp_path and tmp_path.exists():
        tmp_path.unlink()

    print(f"Done! Output saved to: {output}")
    if toc_entries:
        print(f"Added {len(toc_entries)} bookmark(s)")

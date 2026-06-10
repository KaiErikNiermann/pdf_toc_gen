"""Core PDF processing logic."""

import shutil
from pathlib import Path

import fitz  # type: ignore

from pdftoc.bookmarks import add_bookmarks, get_existing_bookmarks, verify_bookmarks
from pdftoc.deepdoc_headers import (
    extract_section_headers_with_deepdoc,
    is_deepdoctection_available,
)
from pdftoc.models import ExtractionMode, OcrBackend, TocEntry
from pdftoc.ocr import extract_text, pdf_has_text, run_ocr
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
    "is_deepdoctection_available",
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
    ocr_backend: OcrBackend = OcrBackend.AUTO,
    searchable: bool = False,
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
        ocr_backend: OCR backend to use (auto, marker, ocrmypdf)
        searchable: If True, embed OCR text layer in output via ocrmypdf
    """
    print(f"Processing: {source}")

    # Step 0: Check existing bookmarks
    doc: fitz.Document = fitz.open(source)
    existing_bookmarks = get_existing_bookmarks(doc)
    doc.close()

    if existing_bookmarks and not force_ocr:
        print(f"PDF has {len(existing_bookmarks)} existing bookmark(s)")
        doc = fitz.open(source)
        is_valid, _issues = verify_bookmarks(doc, existing_bookmarks, verbose)
        doc.close()

        if is_valid:
            print("Existing bookmarks appear correct, copying to output")
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

    # Step 2: Extract text (OCR if needed)
    page_texts: dict[int, str] | None = None
    working_pdf = source
    tmp_path: Path | None = None

    if needs_ocr:
        print("Running OCR (this may take a while)...")
        try:
            page_texts = extract_text(
                source,
                backend=ocr_backend,
                language=language,
                verbose=verbose,
                optimize=optimize,
            )
            # For marker backend, we work with the original PDF
            # (no searchable PDF intermediate needed)
            working_pdf = source
        except RuntimeError as e:
            # If user explicitly requested a backend, don't silently fall back
            if ocr_backend != OcrBackend.AUTO:
                raise
            print(f"Warning: OCR failed ({e}), continuing with original PDF")
            working_pdf = source
    else:
        print("PDF already has text, skipping OCR")

    # Free OCR models from GPU before TOC extraction (LLM may need GPU)
    if needs_ocr and ocr_backend != OcrBackend.OCRMYPDF:
        from pdftoc.marker_ocr import unload_models

        unload_models()

    # Step 3: Extract TOC from the PDF
    print("Extracting table of contents...")
    doc = fitz.open(working_pdf)
    try:
        toc_entries: list[TocEntry] = []

        if mode == ExtractionMode.AUTO:
            toc_entries = extract_toc_from_text(doc, verbose, page_texts=page_texts)
            if not toc_entries:
                if verbose:
                    print("No TOC page found, trying section header extraction...")
                toc_entries = extract_section_headers_with_deepdoc(
                    doc, working_pdf, verbose, page_texts=page_texts
                )
        elif mode == ExtractionMode.TOC_PAGE:
            toc_entries = extract_toc_from_text(doc, verbose, page_texts=page_texts)
        elif mode == ExtractionMode.SECTION_HEADERS:
            toc_entries = extract_section_headers_with_deepdoc(
                doc, working_pdf, verbose, page_texts=page_texts
            )
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

    # Step 5: Optionally embed OCR text layer via ocrmypdf
    if searchable and not pdf_has_text(output):
        print("Embedding searchable text layer (ocrmypdf)...")
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            searchable_tmp = Path(tmp.name)
        try:
            run_ocr(output, searchable_tmp, language, verbose, optimize)
            # Replace output with the searchable version
            shutil.move(str(searchable_tmp), str(output))
            print("Searchable text layer embedded")
        except RuntimeError as e:
            print(f"Warning: failed to embed text layer ({e})")
            searchable_tmp.unlink(missing_ok=True)

    # Cleanup temp file
    if tmp_path and tmp_path.exists():
        tmp_path.unlink()

    print(f"Done! Output saved to: {output}")
    if toc_entries:
        print(f"Added {len(toc_entries)} bookmark(s)")

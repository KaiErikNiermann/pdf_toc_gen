#!/usr/bin/env python3
"""CLI tool to add TOC bookmarks to PDFs."""

from pathlib import Path
from typing import Annotated

import fitz  # type: ignore
import typer

from pdftoc.arxiv import get_arxiv_source
from pdftoc.core import ExtractionMode, process_pdf
from pdftoc.models import OcrBackend
from pdftoc.deepdoc_headers import extract_section_headers_with_deepdoc
from pdftoc.models import TocEntry, format_toc_plaintext
from pdftoc.toc_extraction import extract_toc_from_text

app = typer.Typer(
    name="pdftoc",
    help="Add table of contents bookmarks to PDFs. OCRs if needed.",
)


@app.command()
def main(
    source: Annotated[
        Path,
        typer.Option(
            "--from",
            "-f",
            help="Source PDF file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--to",
            "-t",
            help="Output PDF file (required unless using --get-arxiv-source)",
            file_okay=True,
            dir_okay=False,
            writable=True,
            resolve_path=True,
        ),
    ] = None,
    skip_ocr: Annotated[
        bool,
        typer.Option(
            "--skip-ocr",
            help="Skip OCR even if PDF appears to need it",
        ),
    ] = False,
    force_ocr: Annotated[
        bool,
        typer.Option(
            "--force-ocr",
            help="Force OCR even if PDF already has text",
        ),
    ] = False,
    language: Annotated[
        str,
        typer.Option(
            "--lang",
            "-l",
            help="OCR language (e.g., 'eng', 'deu', 'eng+deu')",
        ),
    ] = "eng",
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Verbose output",
        ),
    ] = False,
    optimize: Annotated[
        int,
        typer.Option(
            "--optimize",
            "-O",
            help="OCR optimization level (0-3). Higher = smaller file, slower. 2+ needs jbig2enc.",
            min=0,
            max=3,
        ),
    ] = 1,
    mode: Annotated[
        str,
        typer.Option(
            "--mode",
            "-m",
            help="TOC extraction mode: 'auto' (try TOC page then headers), 'toc-page', 'section-headers'",
        ),
    ] = "auto",
    no_fix: Annotated[
        bool,
        typer.Option(
            "--no-fix",
            help="Don't fix incorrect existing bookmarks (keep them as-is)",
        ),
    ] = False,
    get_arxiv_source_flag: Annotated[
        bool,
        typer.Option(
            "--get-arxiv-source",
            "-gs",
            help="Download arXiv LaTeX source files instead of processing TOC",
        ),
    ] = False,
    arxiv_output_dir: Annotated[
        Path | None,
        typer.Option(
            "--arxiv-output",
            "-ao",
            help="Directory for arXiv source download (defaults to PDF's directory)",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
    dump_toc: Annotated[
        bool,
        typer.Option(
            "--dump-toc",
            "-d",
            help="Extract and print the table of contents as formatted plaintext (no PDF output)",
        ),
    ] = False,
    toc_width: Annotated[
        int,
        typer.Option(
            "--toc-width",
            help="Width for TOC plaintext output (default: 80)",
        ),
    ] = 80,
    start_page: Annotated[
        int | None,
        typer.Option(
            "--start-page",
            "-sp",
            help="First page to scan for section headers (1-indexed). Useful for testing on large PDFs.",
        ),
    ] = None,
    end_page: Annotated[
        int | None,
        typer.Option(
            "--end-page",
            "-ep",
            help="Last page to scan for section headers (1-indexed, inclusive).",
        ),
    ] = None,
    ocr_backend: Annotated[
        str,
        typer.Option(
            "--ocr-backend",
            help="OCR backend: 'auto' (marker if available, else ocrmypdf), 'marker', 'ocrmypdf'",
        ),
    ] = "auto",
    searchable: Annotated[
        bool,
        typer.Option(
            "--searchable",
            help="Embed OCR text layer in output PDF (requires ocrmypdf + tesseract).",
        ),
    ] = False,
) -> None:
    """Process a PDF to add TOC bookmarks based on detected table of contents."""
    # Handle arXiv source download mode
    if get_arxiv_source_flag:
        get_arxiv_source(source, arxiv_output_dir, verbose)
        return

    # Convert mode string to enum
    mode_map = {
        "auto": ExtractionMode.AUTO,
        "toc-page": ExtractionMode.TOC_PAGE,
        "section-headers": ExtractionMode.SECTION_HEADERS,
    }
    extraction_mode = mode_map.get(mode, ExtractionMode.AUTO)

    # Handle dump-toc mode
    if dump_toc:
        _dump_toc_plaintext(source, extraction_mode, verbose, toc_width, start_page, end_page)
        return

    # Regular TOC processing requires output path
    if output is None:
        print("Error: --to/-t output path is required for TOC processing")
        raise typer.Exit(1)

    # Convert backend string to enum
    backend_map = {
        "auto": OcrBackend.AUTO,
        "marker": OcrBackend.MARKER,
        "ocrmypdf": OcrBackend.OCRMYPDF,
    }
    backend = backend_map.get(ocr_backend, OcrBackend.AUTO)

    process_pdf(
        source=source,
        output=output,
        skip_ocr=skip_ocr,
        force_ocr=force_ocr,
        language=language,
        verbose=verbose,
        optimize=optimize,
        mode=extraction_mode,
        fix_bookmarks=not no_fix,
        ocr_backend=backend,
        searchable=searchable,
    )


def _dump_toc_plaintext(
    source: Path,
    mode: ExtractionMode,
    verbose: bool,
    width: int,
    start_page: int | None = None,
    end_page: int | None = None,
) -> None:
    """Extract and print TOC as formatted plaintext."""
    from pdftoc.deepdoc_headers import extract_headers_deepdoctection

    doc: fitz.Document = fitz.open(source)
    try:
        toc_entries: list[TocEntry] = []

        if mode == ExtractionMode.AUTO:
            # Try TOC page extraction first
            toc_entries = extract_toc_from_text(doc, verbose)
            # Fall back to section headers if no TOC found
            if not toc_entries:
                if verbose:
                    print("No TOC page found, trying section header extraction...")
                # Use deepdoctection with page range if specified
                toc_entries = extract_headers_deepdoctection(
                    source, verbose, start_page, end_page
                )
                if not toc_entries:
                    # Fall back to legacy
                    toc_entries = extract_section_headers_with_deepdoc(doc, source, verbose)
        elif mode == ExtractionMode.TOC_PAGE:
            toc_entries = extract_toc_from_text(doc, verbose)
        elif mode == ExtractionMode.SECTION_HEADERS:
            # Use deepdoctection with page range if specified
            toc_entries = extract_headers_deepdoctection(
                source, verbose, start_page, end_page
            )
            if not toc_entries:
                # Fall back to legacy
                toc_entries = extract_section_headers_with_deepdoc(doc, source, verbose)
    finally:
        doc.close()

    print(format_toc_plaintext(toc_entries, page_width=width))


def run() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    run()

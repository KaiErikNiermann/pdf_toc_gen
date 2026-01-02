#!/usr/bin/env python3
"""CLI tool to add TOC bookmarks to PDFs."""

from pathlib import Path
from typing import Annotated

import typer

from pdftoc.arxiv import get_arxiv_source
from pdftoc.core import ExtractionMode, process_pdf

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
) -> None:
    """Process a PDF to add TOC bookmarks based on detected table of contents."""
    # Handle arXiv source download mode
    if get_arxiv_source_flag:
        get_arxiv_source(source, arxiv_output_dir, verbose)
        return

    # Regular TOC processing requires output path
    if output is None:
        print("Error: --to/-t output path is required for TOC processing")
        raise typer.Exit(1)

    # Convert mode string to enum
    mode_map = {
        "auto": ExtractionMode.AUTO,
        "toc-page": ExtractionMode.TOC_PAGE,
        "section-headers": ExtractionMode.SECTION_HEADERS,
    }
    extraction_mode = mode_map.get(mode, ExtractionMode.AUTO)

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
    )


def run() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    run()

#!/usr/bin/env python3
"""CLI tool to add TOC bookmarks to PDFs."""

from pathlib import Path
from typing import Annotated

import typer

from pdftoc.core import process_pdf

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
        Path,
        typer.Option(
            "--to",
            "-t",
            help="Output PDF file",
            file_okay=True,
            dir_okay=False,
            writable=True,
            resolve_path=True,
        ),
    ],
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
) -> None:
    """Process a PDF to add TOC bookmarks based on detected table of contents."""
    process_pdf(
        source=source,
        output=output,
        skip_ocr=skip_ocr,
        force_ocr=force_ocr,
        language=language,
        verbose=verbose,
    )


def run() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    run()

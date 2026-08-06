"""Section header extraction using deepdoctection for improved accuracy."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import fitz  # type: ignore

from pdftoc.models import TocEntry
from pdftoc.page_labels import PageRef

if TYPE_CHECKING:
    pass

# Global analyzer instance (lazy-loaded)
_analyzer: object | None = None
_dd_available: bool | None = None


def is_deepdoctection_available() -> bool:
    """Check if deepdoctection is available."""
    global _dd_available
    if _dd_available is None:
        try:
            import deepdoctection  # type: ignore # noqa: F401

            _dd_available = True
        except ImportError:
            _dd_available = False
    return _dd_available


def _create_truncated_pdf(
    pdf_path: Path,
    start_page: int = 0,
    end_page: int | None = None,
) -> Path:
    """
    Create a temporary truncated PDF with only the specified page range.

    Args:
        pdf_path: Path to the original PDF
        start_page: First page to include (0-indexed)
        end_page: Last page to include (exclusive, 0-indexed). None = all pages.

    Returns:
        Path to temporary truncated PDF
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    if end_page is None or end_page > total_pages:
        end_page = total_pages

    # Create a new document with selected pages
    new_doc = fitz.open()
    new_doc.insert_pdf(doc, from_page=start_page, to_page=end_page - 1)

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = Path(tmp.name)  # kept on disk deliberately; caller reads it
    new_doc.save(tmp_path)
    new_doc.close()
    doc.close()

    return tmp_path


def _get_analyzer() -> object:
    """Get or create the deepdoctection analyzer (lazy-loaded singleton)."""
    global _analyzer
    if _analyzer is None:
        import deepdoctection as dd  # type: ignore
        import torch  # pyright: ignore[reportMissingImports]

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Use PDF miner for text extraction (no OCR needed for digital PDFs)
        # and specify device explicitly
        _analyzer = dd.get_dd_analyzer(
            config_overwrite=[
                f"DEVICE={device}",
                "USE_OCR=False",
                "USE_PDF_MINER=True",
            ]
        )
    return _analyzer


def _infer_section_level(title: str, page_num: int, font_size: float | None) -> int:
    """
    Infer the hierarchical level of a section title.

    Uses numbering patterns and heuristics to determine level:
    - Level 1: Chapter N, unnumbered major sections
    - Level 2: N. Title (main sections)
    - Level 3: N.N Title (subsections)
    - Level 4: N.N.N Title (sub-subsections)
    """
    title_stripped = title.strip()

    # Chapter patterns -> level 1
    if re.match(r"^(Chapter|CHAPTER)\s+\d+", title_stripped, re.IGNORECASE):
        return 1

    # N.N.N pattern -> level 4
    if re.match(r"^\d+\.\d+\.\d+\s+", title_stripped):
        return 4

    # N.N pattern -> level 3
    if re.match(r"^\d+\.\d+\s+", title_stripped):
        return 3

    # N. pattern -> level 2
    if re.match(r"^\d+\.?\s+", title_stripped):
        return 2

    # Unnumbered but detected as title -> likely level 1 or 2
    # Use font size hint if available, otherwise default to 1
    return 1


def _clean_title(text: str) -> str:
    """Clean up extracted title text."""
    # Remove excessive whitespace
    text = " ".join(text.split())
    # Remove trailing page numbers that might have been included
    text = re.sub(r"\s+\d+\s*$", "", text)
    return text.strip()


def _is_valid_section_title(text: str) -> bool:
    """Check if text looks like a valid section title."""
    text = text.strip()

    # Too short or too long
    if len(text) < 3 or len(text) > 150:
        return False

    # Looks like a page number
    if re.match(r"^\d+$", text):
        return False

    # Looks like a date
    if re.match(r"^\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}$", text):
        return False

    # Ends with sentence punctuation (but allow colons for subtitles)
    if text.endswith((".", ",", ";", "?")):
        return False

    # Looks like a reference citation
    return not re.match(r"^\[\d+\]", text)


def extract_headers_deepdoctection(
    pdf_path: Path,
    verbose: bool = False,
    start_page: int | None = None,
    end_page: int | None = None,
) -> list[TocEntry]:
    """
    Extract section headers using deepdoctection's layout analysis.

    This uses deep learning models to identify title/section_header elements
    in the document, providing more accurate detection than heuristics alone.

    Args:
        pdf_path: Path to the PDF file
        verbose: Whether to print debug information
        start_page: First page to scan (1-indexed). None = start from page 1.
        end_page: Last page to scan (1-indexed, inclusive). None = scan all pages.

    Returns:
        List of TocEntry objects representing detected section headers
    """
    if not is_deepdoctection_available():
        if verbose:
            print("deepdoctection not available, cannot use DL-based extraction")
        return []

    if verbose:
        print("Using deepdoctection for section header extraction...")

    # Handle page range - create truncated PDF if needed
    working_pdf = pdf_path
    page_offset = 0
    temp_pdf: Path | None = None

    if start_page is not None or end_page is not None:
        # Convert from 1-indexed to 0-indexed for internal use
        start_idx = (start_page - 1) if start_page else 0
        end_idx = end_page if end_page else None  # end_page is inclusive, so no -1

        if verbose:
            doc = fitz.open(pdf_path)
            total = len(doc)
            doc.close()
            print(f"  Scanning pages {start_idx + 1}-{end_idx or total} of {total}")

        temp_pdf = _create_truncated_pdf(pdf_path, start_idx, end_idx)
        working_pdf = temp_pdf
        page_offset = start_idx  # To correct page numbers in results

    analyzer = _get_analyzer()

    toc_entries: list[TocEntry] = []
    seen: set[tuple[str, int]] = set()

    try:
        # Analyze the PDF
        df = analyzer.analyze(path=str(working_pdf))  # type: ignore[union-attr]
        df.reset_state()

        for page in df:
            # page_number is 0-indexed in deepdoctection, convert to 1-indexed
            # and add offset if we're using a truncated PDF
            _page_num = page.page_number if hasattr(page, "page_number") else 0
            page_num: int = (
                int(_page_num) + 1 + page_offset
            )  # Convert to 1-indexed + offset

            # Get all layout segments
            layouts = getattr(page, "layouts", [])

            if verbose:
                # Debug: show all categories found on this page
                categories_found: set[str] = set()
                for layout in layouts:
                    cat = getattr(layout, "category_name", "unknown")
                    categories_found.add(str(cat))
                if categories_found:
                    print(f"  Page {page_num} categories: {sorted(categories_found)}")

            for layout in layouts:
                category_raw = getattr(layout, "category_name", "")
                category = str(category_raw).lower()

                # Look for title and section_header categories
                # Different models may use different category names
                # DocLayNet uses: title, section_header, text, list, table, figure, etc.
                if not any(
                    kw in category
                    for kw in ("title", "section", "header", "headline", "heading")
                ):
                    continue

                text = getattr(layout, "text", "")
                if not text:
                    continue

                text = _clean_title(text)

                if verbose:
                    print(
                        f"    Candidate [{category}]: '{text[:60]}...' "
                        if len(text) > 60
                        else f"    Candidate [{category}]: '{text}'"
                    )

                if not _is_valid_section_title(text):
                    if verbose:
                        print("      -> Rejected by validation")
                    continue

                # Infer section level
                level = _infer_section_level(text, page_num, None)

                # Deduplicate
                key = (text.lower(), page_num)
                if key in seen:
                    continue
                seen.add(key)

                entry = TocEntry(
                    level=level, title=text, page=page_num, page_ref=PageRef.PDF
                )
                toc_entries.append(entry)

                if verbose:
                    print(f"      -> Accepted: L{level} -> p.{page_num}")

    except Exception as e:
        if verbose:
            import traceback

            traceback.print_exc()
            print(f"deepdoctection extraction failed: {e}")
        return []
    finally:
        # Clean up temp file if we created one
        if temp_pdf and temp_pdf.exists():
            temp_pdf.unlink()

    # Sort by page, then by reading order/level
    toc_entries.sort(key=lambda e: e.sort_key)

    if verbose:
        print(f"deepdoctection found {len(toc_entries)} section headers")

    return toc_entries


def extract_section_headers_with_deepdoc(
    doc: fitz.Document,
    pdf_path: Path | None,
    verbose: bool = False,
    page_texts: dict[int, str] | None = None,
) -> list[TocEntry]:
    """
    Extract section headers, preferring deepdoctection if available.

    Falls back to legacy heuristic extraction if deepdoctection is not
    installed or fails.

    Args:
        doc: PyMuPDF document object (used for legacy fallback)
        pdf_path: Path to the PDF file (needed for deepdoctection)
        verbose: Whether to print debug information
        page_texts: Optional pre-extracted text per page {1-indexed: text}.

    Returns:
        List of TocEntry objects representing detected section headers
    """
    # Try deepdoctection first if available and we have a path
    if pdf_path and is_deepdoctection_available():
        entries = extract_headers_deepdoctection(pdf_path, verbose)
        if entries:
            return entries
        elif verbose:
            print("deepdoctection found no headers, falling back to legacy extraction")

    # Fall back to legacy heuristic extraction
    from pdftoc.section_headers import extract_section_headers

    if verbose:
        print("Using legacy heuristic section header extraction...")

    return extract_section_headers(doc, verbose, page_texts=page_texts)

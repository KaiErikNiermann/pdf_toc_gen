"""Data models for pdftoc."""

from dataclasses import dataclass
from enum import Enum, StrEnum


class OcrBackend(StrEnum):
    """OCR backend selection."""

    AUTO = "auto"  # marker if available, else ocrmypdf
    MARKER = "marker"  # GPU-accelerated via surya, best quality
    PADDLE = "paddle"  # PaddleOCR PP-OCRv5 classic pipeline, CPU-viable
    OCRMYPDF = "ocrmypdf"  # legacy tesseract-based


class ExtractionMode(Enum):
    """Mode for TOC extraction."""

    AUTO = "auto"  # Try TOC pages first, then section headers
    TOC_PAGE = "toc-page"  # Only look for TOC pages
    SECTION_HEADERS = "section-headers"  # Extract from section headers in content


@dataclass
class TocEntry:
    """A table of contents entry."""

    level: int
    title: str
    page: int


def format_toc_plaintext(
    entries: list["TocEntry"],
    indent_str: str = "  ",
    page_width: int = 80,
    show_page_numbers: bool = True,
) -> str:
    """Format TOC entries as nicely formatted plaintext.

    Args:
        entries: List of TocEntry objects
        indent_str: String to use for each level of indentation
        page_width: Width for dot-leader alignment
        show_page_numbers: Whether to include page numbers

    Returns:
        Formatted plaintext string
    """
    if not entries:
        return "No table of contents entries found."

    lines: list[str] = []
    lines.append("=" * page_width)
    lines.append("TABLE OF CONTENTS".center(page_width))
    lines.append("=" * page_width)
    lines.append("")

    for entry in entries:
        indent = indent_str * (entry.level - 1)
        title = f"{indent}{entry.title}"

        if show_page_numbers:
            # Calculate space for dot leaders
            page_str = str(entry.page)
            # Leave room for at least 3 dots and the page number
            available_width = page_width - len(title) - len(page_str) - 2

            if available_width > 3:
                dots = "." * available_width
                line = f"{title} {dots} {page_str}"
            else:
                # Title too long, just append page number
                line = f"{title} ... {page_str}"
        else:
            line = title

        lines.append(line)

    lines.append("")
    lines.append("=" * page_width)
    lines.append(f"Total entries: {len(entries)}")

    return "\n".join(lines)

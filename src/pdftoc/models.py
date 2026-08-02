"""Data models for pdftoc."""

from dataclasses import dataclass
from enum import Enum, StrEnum

from pdftoc.page_labels import PageRef, format_page_label


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
    """A table of contents entry.

    `page` is meaningless without `page_ref`: printed page 9 of the body,
    printed page "ix" of the front matter and PDF page 9 are three different
    pages. Entries scraped from a TOC carry printed numbers; entries found by
    scanning the document itself carry `PageRef.PDF`.
    """

    level: int
    title: str
    page: int
    page_ref: PageRef = PageRef.PRINTED_ARABIC

    @property
    def page_label(self) -> str:
        """The page number as printed, e.g. `"ix"` for roman front matter."""
        return format_page_label(self.page, self.page_ref)

    @property
    def sort_key(self) -> tuple[int, int, int]:
        """Document order: front matter precedes the body, whatever the digits."""
        return (
            0 if self.page_ref == PageRef.PRINTED_ROMAN else 1,
            self.page,
            self.level,
        )


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
            page_str = entry.page_label
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

"""
Browser-compatible TOC extraction for PDFs.

This module contains the pure Python logic for TOC extraction,
without any fitz/PyMuPDF dependencies. It takes page texts as input
instead of a PDF document object.

This allows it to be used with Pyodide in the browser, where JavaScript
handles the PDF reading/writing via pdf.js and pdf-lib.
"""

import re
from dataclasses import dataclass

from pdftoc.page_labels import (
    PageMap,
    PageMapSource,
    PageRef,
    folios_from_text,
    parse_page_label,
    resolve_page_map,
)


@dataclass
class TocEntry:
    """A table of contents entry.

    `page` is meaningless without `page_ref`: printed page 9 of the body,
    printed page "ix" of the front matter and PDF page 9 are three different
    pages.
    """

    level: int
    title: str
    page: int
    page_ref: PageRef = PageRef.PRINTED_ARABIC

    @property
    def sort_key(self) -> tuple[int, int, int]:
        """Document order: front matter precedes the body, whatever the digits."""
        return (
            0 if self.page_ref == PageRef.PRINTED_ROMAN else 1,
            self.page,
            self.level,
        )


def extract_toc_from_pages(
    page_texts: list[str], total_pages: int, verbose: bool = False
) -> list[TocEntry]:
    """
    Extract TOC entries from page texts.

    Args:
        page_texts: List of text content from each page (first ~15 pages)
        total_pages: Total number of pages in the document
        verbose: Whether to print debug info

    Returns:
        List of TocEntry objects

    Raises:
        ValueError: If no TOC is detected
    """
    toc_entries: list[TocEntry] = []

    # Look for TOC pages in the provided pages
    toc_pages: list[tuple[int, str]] = []

    for i, text in enumerate(page_texts):
        # Check if this looks like a TOC page
        toc_indicators = [
            "contents",
            "table of contents",
            "inhaltsverzeichnis",
            "índice",
            "sommaire",
        ]
        text_lower = text.lower()
        is_toc_page = any(indicator in text_lower for indicator in toc_indicators)

        # Also check if the page has many numbers (page refs) at line ends
        lines = text.strip().split("\n")
        number_lines = sum(1 for line in lines if re.match(r"^\d+$", line.strip()))
        if number_lines >= 5:
            is_toc_page = True

        if is_toc_page:
            toc_pages.append((i, text))

    if not toc_pages:
        if verbose:
            print("No TOC pages detected")
        raise ValueError(
            "No Table of Contents detected in this PDF. "
            "This tool only works with PDFs that have an existing TOC page."
        )

    # Combine all TOC page text
    toc_text = "\n".join(text for _, text in toc_pages)

    if verbose:
        print(
            f"TOC text extracted ({len(toc_text)} chars) from {len(toc_pages)} page(s)"
        )

    # Strategy 1: Try dotted leader patterns first
    toc_entries = _extract_dotted_leader_format(toc_text, total_pages, verbose)

    # Strategy 2: If no entries found, try line-by-line format
    if not toc_entries:
        if verbose:
            print("No dotted leader format found, trying line-by-line format...")
        toc_entries = _extract_line_by_line_format(toc_text, total_pages, verbose)

    if not toc_entries:
        raise ValueError(
            "TOC page was found but couldn't extract any entries. "
            "The TOC format may not be supported."
        )

    # Sort by page number, then by level
    toc_entries.sort(key=lambda e: e.sort_key)

    if verbose:
        print(f"Found {len(toc_entries)} TOC entries")
        for entry in toc_entries[:15]:
            print(f"  L{entry.level}: {entry.title} -> p.{entry.page}")
        if len(toc_entries) > 15:
            print(f"  ... and {len(toc_entries) - 15} more")

    return toc_entries


def _extract_dotted_leader_format(
    toc_text: str, total_pages: int, verbose: bool
) -> list[TocEntry]:
    """Extract TOC using dotted leader patterns (Title ... page)."""
    toc_entries: list[TocEntry] = []
    seen: set[tuple[str, int]] = set()

    patterns = [
        # "Chapter 1: Title ... 15"
        (
            re.compile(
                r"^(Chapter|CHAPTER)\s+(\d+)[:\s]+(.+?)\s*[\.…·\-_\s]{3,}\s*(\d+)\s*$",
                re.MULTILINE,
            ),
            "chapter",
        ),
        # "Part I: Title ... 5"
        (
            re.compile(
                r"^(Part|PART)\s+([IVX\d]+)[:\s]+(.+?)\s*[\.…·\-_\s]{3,}\s*(\d+)\s*$",
                re.MULTILINE,
            ),
            "part",
        ),
        # "1.1.1 Title ... 15"
        (
            re.compile(
                r"^(\d+\.\d+\.\d+)\s+(.+?)\s*[\.…·\-_\s]{3,}\s*(\d+)\s*$",
                re.MULTILINE,
            ),
            "subsub",
        ),
        # "1.1 Title ... 15"
        (
            re.compile(
                r"^(\d+\.\d+)\s+(.+?)\s*[\.…·\-_\s]{3,}\s*(\d+)\s*$",
                re.MULTILINE,
            ),
            "sub",
        ),
        # "1. Title ... 15"
        (
            re.compile(
                r"^(\d+)[\.\)]\s+(.+?)\s*[\.…·\-_\s]{3,}\s*(\d+)\s*$",
                re.MULTILINE,
            ),
            "numbered",
        ),
        # Generic "Title ... 15" pattern
        (
            re.compile(
                r"^([A-Z][A-Za-z\s,\-:]+?)\s*[\.…·\-_\s]{3,}\s*(\d+)\s*$",
                re.MULTILINE,
            ),
            "generic",
        ),
    ]

    for pattern, ptype in patterns:
        for match in pattern.findall(toc_text):
            if ptype == "chapter":
                prefix, num, title, page_str = match
                title = f"{prefix} {num}: {title.strip()}"
                level = 2
            elif ptype == "part":
                prefix, num, title, page_str = match
                title = f"{prefix} {num}: {title.strip()}"
                level = 1
            elif ptype == "subsub":
                num, title, page_str = match
                title = f"{num} {title.strip()}"
                level = 4
            elif ptype == "sub":
                num, title, page_str = match
                title = f"{num} {title.strip()}"
                level = 3
            elif ptype == "numbered":
                num, title, page_str = match
                title = f"{num}. {title.strip()}"
                level = 2
            else:  # generic
                title, page_str = match
                title = title.strip()
                level = 2

            try:
                page_num = int(page_str)
            except ValueError:
                continue

            key = (title.lower(), page_num)
            if key in seen or page_num < 1 or page_num > total_pages + 50:
                continue
            seen.add(key)
            toc_entries.append(TocEntry(level=level, title=title, page=page_num))

    return toc_entries


def _extract_line_by_line_format(
    toc_text: str, total_pages: int, verbose: bool
) -> list[TocEntry]:
    """
    Extract TOC from line-by-line format where structure is:

    Chapter/Part number or title
    Page number

    e.g.:
    1
    A whirlwind history
    1
    I
    Core Concepts
    7
    """
    toc_entries: list[TocEntry] = []
    lines = [line.strip() for line in toc_text.split("\n") if line.strip()]

    # Filter out header/footer noise
    skip_patterns = [
        r"^contents?$",
        r"^table of contents$",
        r"^\w+\s+\d+,\s+\d{4}$",  # Date like "February 2, 2010"
        r"^[ivxlc]+$",  # Roman numerals alone (but keep as potential part numbers)
    ]

    i = 0
    while i < len(lines):
        line = lines[i]

        # Skip noise
        if any(re.match(p, line, re.IGNORECASE) for p in skip_patterns[:3]):
            i += 1
            continue

        entry = _try_parse_toc_entry(lines, i, total_pages)
        if entry:
            toc_entries.append(entry[0])
            i = entry[1]  # Move to position after this entry
        else:
            i += 1

    # Deduplicate
    seen: set[tuple[str, int]] = set()
    unique_entries: list[TocEntry] = []
    for entry in toc_entries:
        key = (entry.title.lower(), entry.page)
        if key not in seen:
            seen.add(key)
            unique_entries.append(entry)

    return unique_entries


def _try_parse_toc_entry(
    lines: list[str], idx: int, total_pages: int
) -> tuple[TocEntry, int] | None:
    """Try to parse a TOC entry starting at the given index."""
    if idx >= len(lines):
        return None

    line = lines[idx]

    # Check for chapter/section number pattern: just a number like "1", "2", "10"
    chapter_num_match = re.match(r"^(\d+)$", line)
    if chapter_num_match and idx + 2 < len(lines):
        num = chapter_num_match.group(1)
        title_line = lines[idx + 1]
        page_line = lines[idx + 2]

        if not re.match(r"^\d+$", title_line) and not re.match(
            r"^[IVXivx]+$", title_line
        ):
            parsed = _parse_page_number(page_line, total_pages)
            if parsed is not None:
                page, page_ref = parsed
                title = f"{num}. {title_line}"
                return (
                    TocEntry(level=2, title=title, page=page, page_ref=page_ref),
                    idx + 3,
                )

    # Check for Part pattern: Roman numeral like "I", "II", "III"
    part_match = re.match(r"^([IVX]+)$", line, re.IGNORECASE)
    if part_match and idx + 2 < len(lines):
        roman = part_match.group(1).upper()
        title_line = lines[idx + 1]
        page_line = lines[idx + 2]

        if not re.match(r"^\d+$", title_line) and not re.match(
            r"^[IVXivx]+$", title_line
        ):
            parsed = _parse_page_number(page_line, total_pages)
            if parsed is not None:
                page, page_ref = parsed
                title = f"Part {roman}: {title_line}"
                return (
                    TocEntry(level=1, title=title, page=page, page_ref=page_ref),
                    idx + 3,
                )

    # Check for simple "Title" followed by "page" pattern
    if re.match(r"^[A-Z][A-Za-z\s,\-:]+$", line) and idx + 1 < len(lines):
        page_line = lines[idx + 1]
        parsed = _parse_page_number(page_line, total_pages)
        if parsed is not None:
            page, page_ref = parsed
            return (
                TocEntry(level=2, title=line, page=page, page_ref=page_ref),
                idx + 2,
            )

    # Check for subsection pattern: "1.1" or "1.2.3"
    subsec_match = re.match(r"^(\d+(?:\.\d+)+)$", line)
    if subsec_match and idx + 2 < len(lines):
        num = subsec_match.group(1)
        title_line = lines[idx + 1]
        page_line = lines[idx + 2]

        if not re.match(r"^\d+$", title_line):
            parsed = _parse_page_number(page_line, total_pages)
            if parsed is not None:
                page, page_ref = parsed
                level = num.count(".") + 2
                title = f"{num} {title_line}"
                return (
                    TocEntry(level=level, title=title, page=page, page_ref=page_ref),
                    idx + 3,
                )

    return None


def _parse_page_number(s: str, total_pages: int) -> tuple[int, PageRef] | None:
    """Parse a printed page number, keeping its numbering scheme.

    Roman numerals must stay tagged as roman: they belong to the front
    matter, which is numbered independently of the body.
    """
    parsed = parse_page_label(s)
    if parsed is None:
        return None
    if parsed[1] == PageRef.PRINTED_ARABIC and parsed[0] > total_pages + 50:
        return None
    return parsed


def normalize_levels(toc_entries: list[TocEntry]) -> list[TocEntry]:
    """
    Normalize TOC levels so first entry is level 1 and no levels are skipped.
    """
    if not toc_entries:
        return []

    min_level = min(e.level for e in toc_entries)
    shifted = [
        TocEntry(
            level=e.level - min_level + 1,
            title=e.title,
            page=e.page,
            page_ref=e.page_ref,
        )
        for e in toc_entries
    ]

    result: list[TocEntry] = []
    prev_level = 0
    for entry in shifted:
        new_level = entry.level
        if new_level > prev_level + 1:
            new_level = prev_level + 1
        result.append(
            TocEntry(
                level=new_level,
                title=entry.title,
                page=entry.page,
                page_ref=entry.page_ref,
            )
        )
        prev_level = new_level

    return result


def find_page_offset(
    page_texts: list[str], toc_entries: list[TocEntry], verbose: bool = False
) -> int:
    """
    Find the offset between printed page numbers and PDF page indices.

    Args:
        page_texts: Text content of all pages
        toc_entries: Extracted TOC entries
        verbose: Whether to print debug info

    Returns:
        Offset such that: pdf_page = printed_page + offset
    """
    return find_page_map(page_texts, toc_entries, verbose).offset


def _toc_page_indices(page_texts: list[str]) -> frozenset[int]:
    """1-indexed pages that look like the table of contents itself.

    The keyword fallback must skip these: a TOC page mentions every section
    title in the book and would match any offset.
    """
    indices: set[int] = set()
    for i, text in enumerate(page_texts[:15]):
        if "contents" in text.lower():
            # The TOC usually spans a few pages; exclude its neighbours too.
            indices.update(
                j + 1 for j in range(max(0, i - 1), i + 3) if j < len(page_texts)
            )
    return frozenset(indices)


def find_page_map(
    page_texts: list[str], toc_entries: list[TocEntry], verbose: bool = False
) -> PageMap:
    """
    Resolve the printed -> PDF page mapping for a document.

    Args:
        page_texts: Text content of all pages
        toc_entries: Extracted TOC entries
        verbose: Whether to print debug info

    Returns:
        A PageMap covering the arabic body and the roman front matter.
    """
    if not toc_entries:
        return PageMap(0, 0.0, PageMapSource.DEFAULT, "no TOC entries")

    total_pages = len(page_texts)

    # Without page geometry (pdf.js gives us plain text only), folios can only
    # be read off the first/last line of each page.
    page_map = resolve_page_map(
        folios_by_page={
            i + 1: folios_from_text(text) for i, text in enumerate(page_texts)
        },
        titled_pages=[
            (e.title, e.page)
            for e in toc_entries
            if e.page_ref == PageRef.PRINTED_ARABIC and e.page >= 1
        ],
        page_text=lambda pdf_page: page_texts[pdf_page - 1],
        total_pages=total_pages,
        skip_pages=_toc_page_indices(page_texts),
    )

    if verbose:
        print(
            f"  Page offset {page_map.offset:+d} "
            f"via {page_map.source} ({page_map.detail})"
        )

    return page_map


def resolve_pdf_pages(
    page_texts: list[str], toc_entries: list[TocEntry], verbose: bool = False
) -> list[int]:
    """Resolve each TOC entry to its 1-indexed PDF page.

    Prefer this over `find_page_offset`: a single offset cannot express a book
    whose front matter is numbered in roman numerals, since those pages sit on
    a different scale from the arabic body.
    """
    page_map = find_page_map(page_texts, toc_entries, verbose)
    return [
        page_map.to_pdf_page(entry.page, len(page_texts), entry.page_ref)
        for entry in toc_entries
    ]

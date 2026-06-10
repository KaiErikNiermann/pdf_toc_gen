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


@dataclass
class TocEntry:
    """A table of contents entry."""

    level: int
    title: str
    page: int


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
    toc_entries.sort(key=lambda e: (e.page, e.level))

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
            page_num = _parse_page_number(page_line, total_pages)
            if page_num is not None:
                title = f"{num}. {title_line}"
                return (TocEntry(level=2, title=title, page=page_num), idx + 3)

    # Check for Part pattern: Roman numeral like "I", "II", "III"
    part_match = re.match(r"^([IVX]+)$", line, re.IGNORECASE)
    if part_match and idx + 2 < len(lines):
        roman = part_match.group(1).upper()
        title_line = lines[idx + 1]
        page_line = lines[idx + 2]

        if not re.match(r"^\d+$", title_line) and not re.match(
            r"^[IVXivx]+$", title_line
        ):
            page_num = _parse_page_number(page_line, total_pages)
            if page_num is not None:
                title = f"Part {roman}: {title_line}"
                return (TocEntry(level=1, title=title, page=page_num), idx + 3)

    # Check for simple "Title" followed by "page" pattern
    if re.match(r"^[A-Z][A-Za-z\s,\-:]+$", line) and idx + 1 < len(lines):
        page_line = lines[idx + 1]
        page_num = _parse_page_number(page_line, total_pages)
        if page_num is not None:
            if idx + 2 < len(lines) and re.match(r"^\d+$", lines[idx + 2]):
                return (TocEntry(level=2, title=line, page=page_num), idx + 2)
            return (TocEntry(level=2, title=line, page=page_num), idx + 2)

    # Check for subsection pattern: "1.1" or "1.2.3"
    subsec_match = re.match(r"^(\d+(?:\.\d+)+)$", line)
    if subsec_match and idx + 2 < len(lines):
        num = subsec_match.group(1)
        title_line = lines[idx + 1]
        page_line = lines[idx + 2]

        if not re.match(r"^\d+$", title_line):
            page_num = _parse_page_number(page_line, total_pages)
            if page_num is not None:
                level = num.count(".") + 2
                title = f"{num} {title_line}"
                return (TocEntry(level=level, title=title, page=page_num), idx + 3)

    return None


def _parse_page_number(s: str, total_pages: int) -> int | None:
    """Parse a page number string, handling both arabic and roman numerals."""
    s = s.strip().lower()

    # Try arabic numeral
    if re.match(r"^\d+$", s):
        num = int(s)
        if 1 <= num <= total_pages + 50:
            return num
        return None

    # Try roman numeral (for preface pages etc.)
    roman_map = {
        "i": 1,
        "ii": 2,
        "iii": 3,
        "iv": 4,
        "v": 5,
        "vi": 6,
        "vii": 7,
        "viii": 8,
        "ix": 9,
        "x": 10,
        "xi": 11,
        "xii": 12,
        "xiii": 13,
        "xiv": 14,
        "xv": 15,
        "xvi": 16,
        "xvii": 17,
        "xviii": 18,
        "xix": 19,
        "xx": 20,
    }
    if s in roman_map:
        return roman_map[s]

    return None


def normalize_levels(toc_entries: list[TocEntry]) -> list[TocEntry]:
    """
    Normalize TOC levels so first entry is level 1 and no levels are skipped.
    """
    if not toc_entries:
        return []

    min_level = min(e.level for e in toc_entries)
    shifted = [
        TocEntry(level=e.level - min_level + 1, title=e.title, page=e.page)
        for e in toc_entries
    ]

    result: list[TocEntry] = []
    prev_level = 0
    for entry in shifted:
        new_level = entry.level
        if new_level > prev_level + 1:
            new_level = prev_level + 1
        result.append(TocEntry(level=new_level, title=entry.title, page=entry.page))
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
    from collections import Counter

    if not toc_entries:
        return 0

    total_pages = len(page_texts)

    # First, identify TOC pages to skip them during search
    toc_page_indices: set[int] = set()
    for i, text in enumerate(page_texts[:15]):
        text_lower = text.lower()
        if "contents" in text_lower or "table of contents" in text_lower:
            toc_page_indices.add(i)
            if i > 0:
                toc_page_indices.add(i - 1)
            if i + 1 < total_pages:
                toc_page_indices.add(i + 1)
            if i + 2 < total_pages:
                toc_page_indices.add(i + 2)

    # Try to find entries with higher page numbers (more distinctive)
    test_entries = sorted(
        [e for e in toc_entries if e.page > 20],
        key=lambda x: x.page,
    )[:5]

    if not test_entries:
        test_entries = [e for e in toc_entries if e.page > 5][:5]

    offsets: list[int] = []
    for entry in test_entries:
        words = re.findall(r"[A-Za-z]{5,}", entry.title)
        if len(words) < 2:
            continue

        for test_offset in range(-20, 30):
            pdf_page_idx = entry.page + test_offset - 1

            if pdf_page_idx < 0 or pdf_page_idx >= total_pages:
                continue

            if pdf_page_idx in toc_page_indices:
                continue

            text = page_texts[pdf_page_idx].lower()

            matches = sum(1 for w in words if w.lower() in text)
            if matches >= min(2, len(words)):
                offsets.append(test_offset)
                if verbose:
                    print(
                        f"  Found '{entry.title[:30]}...' at PDF page {pdf_page_idx + 1}"
                    )
                break

    if not offsets:
        if verbose:
            print("  Could not determine page offset, using 0")
        return 0

    offset_counts = Counter(offsets)
    best_offset = offset_counts.most_common(1)[0][0]

    if verbose:
        print(f"  Detected page offset: {best_offset}")

    return best_offset

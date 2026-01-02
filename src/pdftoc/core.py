"""Core PDF processing logic."""

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import fitz  # type: ignore


@dataclass
class TocEntry:
    """A table of contents entry."""

    level: int
    title: str
    page: int


def pdf_has_text(pdf_path: Path) -> bool:
    """Check if a PDF already has extractable text."""
    doc: fitz.Document = fitz.open(pdf_path)
    try:
        # Check first few pages for text
        pages_to_check = min(5, len(doc))
        total_text = 0
        for i in range(pages_to_check):
            page: fitz.Page = doc[i]
            text = page.get_text()
            total_text += len(text.strip())
        # If we have a reasonable amount of text, assume it's OCR'd
        return total_text > 100
    finally:
        doc.close()


def run_ocr(source: Path, output: Path, language: str, verbose: bool) -> None:
    """Run OCR on a PDF using ocrmypdf."""
    cmd = [
        "ocrmypdf",
        "--skip-text",  # Skip pages that already have text
        "--optimize",
        "1",
        "-l",
        language,
        str(source),
        str(output),
    ]
    if not verbose:
        cmd.insert(1, "--quiet")

    if verbose:
        print(f"Running OCR: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 and result.returncode != 6:
        # Return code 6 means "file already has text" which is fine
        raise RuntimeError(f"OCR failed: {result.stderr}")


def extract_toc_from_text(doc: fitz.Document, verbose: bool) -> list[TocEntry]:
    """
    Extract TOC entries by finding the table of contents pages and parsing them.

    Handles multiple formats:
    - "Chapter 1 .......... 5" (dotted leader)
    - "1. Introduction ... 10" (dotted leader)
    - Line-by-line format where number and title are on separate lines
    """
    toc_entries: list[TocEntry] = []

    # Look for TOC pages in the first ~15 pages
    toc_pages: list[tuple[int, str]] = []
    pages_to_search = min(15, len(doc))

    for i in range(pages_to_search):
        page: fitz.Page = doc[i]
        text = page.get_text()

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
        return []

    # Combine all TOC page text
    toc_text = "\n".join(text for _, text in toc_pages)

    if verbose:
        print(
            f"TOC text extracted ({len(toc_text)} chars) from {len(toc_pages)} page(s)"
        )

    # Strategy 1: Try dotted leader patterns first
    toc_entries = _extract_dotted_leader_format(toc_text, len(doc), verbose)

    # Strategy 2: If no entries found, try line-by-line format
    if not toc_entries:
        if verbose:
            print("No dotted leader format found, trying line-by-line format...")
        toc_entries = _extract_line_by_line_format(toc_text, len(doc), verbose)

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
    ]

    for pattern, ptype in patterns:
        for match in pattern.findall(toc_text):
            if ptype in ("chapter", "part"):
                prefix, num, title, page_str = match
                title = f"{prefix} {num}: {title.strip()}"
                level = 1 if ptype == "part" else 2
            elif ptype == "subsub":
                num, title, page_str = match
                title = f"{num} {title.strip()}"
                level = 4
            elif ptype == "sub":
                num, title, page_str = match
                title = f"{num} {title.strip()}"
                level = 3
            else:
                num, title, page_str = match
                title = f"{num}. {title.strip()}"
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

        # Try to identify TOC entry patterns

        # Pattern A: "NUMBER" followed by "TITLE" followed by "PAGE"
        # e.g., "1" -> "A whirlwind history" -> "1"
        # Pattern B: "ROMAN" followed by "TITLE" followed by "PAGE" (Parts)
        # e.g., "I" -> "Core Concepts" -> "7"
        # Pattern C: "TITLE" followed by "PAGE" (for entries like "Preface")
        # e.g., "Preface" -> "ix"

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
        # Next line should be title, line after should be page number
        title_line = lines[idx + 1]
        page_line = lines[idx + 2]

        # Title should be text (not just a number)
        if not re.match(r"^\d+$", title_line) and not re.match(
            r"^[IVXivx]+$", title_line
        ):
            # Page could be number or roman numeral
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

    # Check for simple "Title" followed by "page" pattern (e.g., "Preface" -> "ix")
    # Title should start with capital letter and be reasonable text
    if re.match(r"^[A-Z][A-Za-z\s,\-:]+$", line) and idx + 1 < len(lines):
        page_line = lines[idx + 1]
        page_num = _parse_page_number(page_line, total_pages)
        if page_num is not None:
            # Make sure this isn't a chapter number we're about to see
            if idx + 2 < len(lines) and re.match(r"^\d+$", lines[idx + 2]):
                # This might be "Title\nPage\nChapterNum" - skip
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
                level = num.count(".") + 2  # 1.1 -> level 3, 1.1.1 -> level 4
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
        # Roman numerals typically map to early pages, return as-is
        # (these are usually before page 1 in the PDF)
        return roman_map[s]

    return None

    if verbose:
        print(f"Found {len(toc_entries)} TOC entries")
        for entry in toc_entries[:10]:
            print(f"  Level {entry.level}: {entry.title} -> p.{entry.page}")
        if len(toc_entries) > 10:
            print(f"  ... and {len(toc_entries) - 10} more")

    return toc_entries


def add_bookmarks(
    pdf_path: Path, toc_entries: list[TocEntry], output_path: Path, verbose: bool
) -> None:
    """Add bookmarks to a PDF based on TOC entries."""
    doc: fitz.Document = fitz.open(pdf_path)

    try:
        # Find page offset by searching for chapter/section text on expected pages
        page_offset = _find_page_offset(doc, toc_entries, verbose)

        # Normalize levels - PyMuPDF requires first entry to be level 1
        # and levels must not skip (e.g., can't go from 1 to 3)
        normalized_entries = _normalize_levels(toc_entries)

        # Build TOC in PyMuPDF format: [level, title, page]
        toc: list[list[int | str]] = []
        for entry in normalized_entries:
            # Apply page offset to convert printed page to PDF page
            pdf_page = entry.page + page_offset
            if pdf_page < 1:
                pdf_page = 1
            if pdf_page > len(doc):
                pdf_page = len(doc)

            toc.append([entry.level, entry.title, pdf_page])

        if toc:
            doc.set_toc(toc)  # type: ignore
            if verbose:
                print(f"Added {len(toc)} bookmarks to PDF")
        else:
            if verbose:
                print("No TOC entries to add")

        doc.save(output_path)
    finally:
        doc.close()


def _normalize_levels(toc_entries: list[TocEntry]) -> list[TocEntry]:
    """
    Normalize TOC levels so first entry is level 1 and no levels are skipped.
    """
    if not toc_entries:
        return []

    # Find minimum level
    min_level = min(e.level for e in toc_entries)

    # Shift all levels so minimum is 1
    shifted = [
        TocEntry(level=e.level - min_level + 1, title=e.title, page=e.page)
        for e in toc_entries
    ]

    # Ensure no level skips (level can only increase by 1 at a time)
    result: list[TocEntry] = []
    prev_level = 0
    for entry in shifted:
        new_level = entry.level
        if new_level > prev_level + 1:
            new_level = prev_level + 1
        result.append(TocEntry(level=new_level, title=entry.title, page=entry.page))
        prev_level = new_level

    return result


def _find_page_offset(
    doc: fitz.Document, toc_entries: list[TocEntry], verbose: bool
) -> int:
    """
    Find the offset between printed page numbers and PDF page indices.

    Returns offset such that: pdf_page = printed_page + offset
    """
    if not toc_entries:
        return 0

    # First, identify TOC pages to skip them during search
    toc_page_indices: set[int] = set()
    for i in range(min(15, len(doc))):
        page: fitz.Page = doc[i]
        text = page.get_text().lower()
        if "contents" in text or "table of contents" in text:
            toc_page_indices.add(i)
            # Also mark adjacent pages as TOC pages
            if i > 0:
                toc_page_indices.add(i - 1)
            if i + 1 < len(doc):
                toc_page_indices.add(i + 1)
            if i + 2 < len(doc):
                toc_page_indices.add(i + 2)

    # Try to find entries with higher page numbers (more distinctive)
    test_entries = sorted(
        [e for e in toc_entries if e.page > 20],
        key=lambda x: x.page,
    )[:5]

    # If no high-page entries, fall back to any entries
    if not test_entries:
        test_entries = [e for e in toc_entries if e.page > 5][:5]

    offsets: list[int] = []
    for entry in test_entries:
        # Extract keywords from title (skip numbers and common words)
        words = re.findall(r"[A-Za-z]{5,}", entry.title)
        if len(words) < 2:
            continue

        # Search for these words in pages around the expected location
        # Start with offset 0 assumption
        for test_offset in range(-20, 30):
            pdf_page_idx = entry.page + test_offset - 1  # -1 for 0-indexing

            if pdf_page_idx < 0 or pdf_page_idx >= len(doc):
                continue

            # Skip TOC pages
            if pdf_page_idx in toc_page_indices:
                continue

            page = doc[pdf_page_idx]
            text = page.get_text().lower()

            # Check if multiple keywords appear on this page
            matches = sum(1 for w in words if w.lower() in text)
            if matches >= min(2, len(words)):
                offsets.append(test_offset)
                if verbose:
                    print(
                        f"  Found '{entry.title[:30]}...' at PDF page {pdf_page_idx + 1} "
                        f"(printed: {entry.page}, offset: {test_offset})"
                    )
                break

    if not offsets:
        if verbose:
            print("  Could not determine page offset, using 0")
        return 0

    # Use the most common offset
    from collections import Counter

    offset_counts = Counter(offsets)
    best_offset = offset_counts.most_common(1)[0][0]

    if verbose:
        print(f"  Detected page offset: {best_offset}")

    return best_offset


def process_pdf(
    source: Path,
    output: Path,
    skip_ocr: bool = False,
    force_ocr: bool = False,
    language: str = "eng",
    verbose: bool = False,
) -> None:
    """Main processing function."""
    print(f"Processing: {source}")

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
            run_ocr(source, tmp_path, language, verbose)
            working_pdf = tmp_path
        except RuntimeError as e:
            print(f"Warning: OCR failed ({e}), continuing with original PDF")
            working_pdf = source
            tmp_path.unlink(missing_ok=True)
    else:
        print("PDF already has text, skipping OCR")
        working_pdf = source
        tmp_path = None  # type: ignore

    # Step 3: Extract TOC from the PDF text
    print("Extracting table of contents...")
    doc: fitz.Document = fitz.open(working_pdf)
    try:
        toc_entries = extract_toc_from_text(doc, verbose)
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

"""Bookmark management for PDFs."""

import re
from collections import Counter
from pathlib import Path

import fitz  # type: ignore

from pdftoc.models import TocEntry


def get_existing_bookmarks(doc: fitz.Document) -> list[TocEntry]:
    """Get existing bookmarks from a PDF as TocEntry list."""
    toc = doc.get_toc()
    entries = []
    for item in toc:
        level, title, page = item[:3]
        entries.append(TocEntry(level=int(level), title=str(title), page=int(page)))
    return entries


def verify_bookmarks(
    doc: fitz.Document, bookmarks: list[TocEntry], verbose: bool
) -> tuple[bool, list[str]]:
    """
    Verify if existing bookmarks are correct by checking structure and content.

    Returns (is_valid, issues) where issues is a list of problems found.
    """
    issues: list[str] = []

    if not bookmarks:
        return True, []

    # Check 1: Bookmarks should have some structure (not all level 1 pointing to page 1)
    all_same_page = len(set(b.page for b in bookmarks)) == 1
    all_level_1 = all(b.level == 1 for b in bookmarks)
    if all_same_page and all_level_1 and len(bookmarks) <= 3:
        issues.append(
            f"Bookmarks lack structure: {len(bookmarks)} entries all pointing to page {bookmarks[0].page}"
        )

    # Check 2: Should have reasonable number of entries for document size
    if len(bookmarks) < 3 and len(doc) > 10:
        issues.append(
            f"Too few bookmarks ({len(bookmarks)}) for document size ({len(doc)} pages)"
        )

    # Check 3: Verify content on pages for a sample of bookmarks
    sample_size = min(5, len(bookmarks))
    # Prefer bookmarks not on page 1 for content verification
    sample = sorted(bookmarks, key=lambda b: (b.page == 1, b.page))[:sample_size]

    content_issues = 0
    for entry in sample:
        if entry.page < 1 or entry.page > len(doc):
            issues.append(
                f"Bookmark '{entry.title}' points to invalid page {entry.page}"
            )
            continue

        page: fitz.Page = doc[entry.page - 1]
        text = page.get_text().lower()

        # Extract keywords from title (words with 4+ chars)
        keywords = [w.lower() for w in re.findall(r"[A-Za-z]{4,}", entry.title)]

        if not keywords:
            continue

        # Check if at least some keywords appear on the page
        matches = sum(1 for kw in keywords if kw in text)
        if matches < len(keywords) // 2 and matches < 1:
            content_issues += 1

    if content_issues > sample_size // 2:
        issues.append(
            f"{content_issues} of {sample_size} sampled bookmarks have content mismatch"
        )

    is_valid = len(issues) == 0
    if verbose:
        if is_valid:
            print("Existing bookmarks appear valid")
        else:
            print(f"Found {len(issues)} issue(s) with existing bookmarks:")
            for issue in issues:
                print(f"  - {issue}")

    return is_valid, issues


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
    offset_counts = Counter(offsets)
    best_offset = offset_counts.most_common(1)[0][0]

    if verbose:
        print(f"  Detected page offset: {best_offset}")

    return best_offset

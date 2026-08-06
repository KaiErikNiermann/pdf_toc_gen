"""Bookmark management for PDFs."""

import re
from pathlib import Path

import fitz  # type: ignore

from pdftoc.models import TocEntry
from pdftoc.page_labels import (
    Folio,
    PageMap,
    PageMapSource,
    PageRef,
    folios_from_text,
    resolve_page_map,
)
from pdftoc.pdf_text import page_text

# Fraction of page height at the top and bottom treated as margin, where a
# printed folio lives.
_MARGIN_FRACTION = 0.12


def get_existing_bookmarks(doc: fitz.Document) -> list[TocEntry]:
    """Get existing bookmarks from a PDF as TocEntry list."""
    # Bookmarks in a PDF already point at PDF pages, never printed ones.
    return [
        TocEntry(
            level=int(item[0]),
            title=str(item[1]),
            page=int(item[2]),
            page_ref=PageRef.PDF,
        )
        for item in doc.get_toc()
    ]


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
    all_same_page = len({b.page for b in bookmarks}) == 1
    all_level_1 = all(b.level == 1 for b in bookmarks)
    if all_same_page and all_level_1 and len(bookmarks) <= 3:
        issues.append(
            f"Bookmarks lack structure: {len(bookmarks)} entries all pointing to page {bookmarks[0].page}"
        )

    # Check 2: Bookmarks with numeric-only titles are useless page labels
    numeric_titles = sum(1 for b in bookmarks if b.title.strip().isdigit())
    if numeric_titles > len(bookmarks) * 0.8:
        issues.append(
            f"Most bookmarks ({numeric_titles}/{len(bookmarks)}) have numeric-only titles"
        )

    # Check 3: Should have reasonable number of entries for document size
    if len(bookmarks) < 3 and len(doc) > 10:
        issues.append(
            f"Too few bookmarks ({len(bookmarks)}) for document size ({len(doc)} pages)"
        )

    # Check 4: Verify content on pages for a sample of bookmarks
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
        text = page_text(page).lower()

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
        page_map = find_page_map(doc, toc_entries, verbose)

        # Normalize levels - PyMuPDF requires first entry to be level 1
        # and levels must not skip (e.g., can't go from 1 to 3)
        normalized_entries = _normalize_levels(toc_entries)

        # Build TOC in PyMuPDF format: [level, title, page]. Each entry is
        # mapped through its own frame: roman front matter, arabic body, and
        # entries already expressed as PDF pages all resolve differently.
        toc: list[list[int | str]] = [
            [
                entry.level,
                entry.title,
                page_map.to_pdf_page(entry.page, len(doc), entry.page_ref),
            ]
            for entry in normalized_entries
        ]

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
        TocEntry(
            level=e.level - min_level + 1,
            title=e.title,
            page=e.page,
            page_ref=e.page_ref,
        )
        for e in toc_entries
    ]

    # Ensure no level skips (level can only increase by 1 at a time)
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


def _margin_folios(page: fitz.Page) -> tuple[Folio, ...]:
    """Folio candidates from text blocks physically in a page's margins.

    More reliable than reading the first/last line of the extracted text: it
    ignores body text and running headers that happen to be bare numbers.
    """
    height = page.rect.height
    if height <= 0:
        return ()

    top_band = height * _MARGIN_FRACTION
    bottom_band = height * (1 - _MARGIN_FRACTION)

    folios: set[Folio] = set()
    for block in page.get_text("blocks"):
        _x0, y0, _x1, y1, text = block[:5]
        if y1 <= top_band or y0 >= bottom_band:
            folios.update(folios_from_text(str(text)))
    return tuple(sorted(folios, key=lambda f: (f.ref, f.number)))


def _decimal_label_ranges(doc: fitz.Document) -> tuple[tuple[int, int], ...]:
    """`/PageLabels` ranges numbered in plain decimals, as `(start, first_num)`.

    `start` is a 0-indexed PDF page. Prefixed and roman ranges are ignored:
    a TOC's arabic page numbers can only refer to a plain decimal range.
    """
    try:
        labels = doc.get_page_labels()
    except Exception:  # pragma: no cover - older PyMuPDF without the API
        return ()

    return tuple(
        (int(rule["startpage"]), int(rule.get("firstpagenum", 1) or 1))
        for rule in labels or ()
        if str(rule.get("style", "")).upper() == "D" and not rule.get("prefix")
    )


def _page_label_offset(doc: fitz.Document) -> tuple[int, str] | None:
    """Offset implied by the PDF's own `/PageLabels`, if it declares one.

    Only decimal-numbered ranges matter: a TOC cites arabic page numbers, so
    the widest decimal range is the body of the book.
    """
    decimal_ranges = _decimal_label_ranges(doc)
    if not decimal_ranges:
        return None

    # Widest decimal range wins: that is the body of the book.
    starts = sorted(start for start, _ in decimal_ranges)
    width_from = {
        start: (starts[i + 1] if i + 1 < len(starts) else len(doc)) - start
        for i, start in enumerate(starts)
    }
    start_page, first_num = max(decimal_ranges, key=lambda r: width_from[r[0]])

    offset = (start_page + 1) - first_num
    return offset, f"/PageLabels: PDF p.{start_page + 1} is printed p.{first_num}"


def _toc_page_indices(doc: fitz.Document) -> frozenset[int]:
    """1-indexed PDF pages that look like the table of contents itself."""
    indices: set[int] = set()
    for i in range(min(15, len(doc))):
        if "contents" in str(doc[i].get_text()).lower():
            # The TOC usually spans a few pages; exclude its neighbours too.
            indices.update(j for j in range(i, i + 3) if 0 <= j < len(doc))
            if i > 0:
                indices.add(i - 1)
    return frozenset(i + 1 for i in indices)


def find_page_map(
    doc: fitz.Document, toc_entries: list[TocEntry], verbose: bool
) -> PageMap:
    """
    Find the mapping between printed page numbers and PDF page indices.

    Returns a PageMap such that: pdf_page = printed_page + offset
    """
    if not toc_entries:
        return PageMap(0, 0.0, PageMapSource.DEFAULT, "no TOC entries")

    # Entries already in PDF coordinates say nothing about the offset, and
    # roman front matter is numbered on a different scale. Only printed arabic
    # pages can locate the body.
    probes = [
        (e.title, e.page)
        for e in toc_entries
        if e.page_ref == PageRef.PRINTED_ARABIC and e.page >= 1
    ]

    page_map = resolve_page_map(
        folios_by_page={i + 1: _margin_folios(doc[i]) for i in range(len(doc))},
        titled_pages=probes,
        page_text=lambda pdf_page: str(doc[pdf_page - 1].get_text()),
        total_pages=len(doc),
        label_offset=_page_label_offset(doc),
        skip_pages=_toc_page_indices(doc),
    )

    if verbose:
        print(
            f"  Page offset {page_map.offset:+d} "
            f"via {page_map.source} ({page_map.detail})"
        )
        if page_map.source == PageMapSource.DEFAULT:
            print("  Warning: bookmarks may be misaligned")

    return page_map

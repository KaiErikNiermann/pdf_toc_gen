"""End-to-end test that roman-numbered front matter lands in the front matter.

Builds a small synthetic book rather than downloading one, so this runs
offline and pins the exact behaviour that used to be wrong: an entry citing
printed page "ix" was collapsed to the integer 9 and then shifted by the body
offset, landing deep in chapter 1.
"""

from __future__ import annotations

from pathlib import Path

import fitz  # type: ignore
import pytest

from pdftoc.bookmarks import add_bookmarks, find_page_map
from pdftoc.models import TocEntry
from pdftoc.page_labels import PageRef, format_page_label

FRONT_MATTER_PAGES = 8
BODY_PAGES = 40


@pytest.fixture
def synthetic_book(tmp_path: Path) -> Path:
    """A book with roman front matter (i..viii) then an arabic body (1..40)."""
    doc: fitz.Document = fitz.open()

    for i in range(FRONT_MATTER_PAGES):
        page = doc.new_page()
        folio = format_page_label(i + 1, PageRef.PRINTED_ROMAN)
        page.insert_text((72, 60), "Preface" if i >= 2 else "Front Matter")
        page.insert_text((72, 100), "front matter prose about the book")
        page.insert_text((300, 780), folio)  # folio in the bottom margin

    for printed in range(1, BODY_PAGES + 1):
        page = doc.new_page()
        page.insert_text((300, 40), str(printed))  # folio in the top margin
        page.insert_text((72, 100), "CHAPTER 1. MEASURE THEORY")
        if printed == 12:
            page.insert_text((72, 140), "1.5 Properties of the Integral")
        page.insert_text((72, 180), "body prose discussing measure and integration")

    path = tmp_path / "synthetic.pdf"
    doc.save(path)
    doc.close()
    return path


def test_detects_both_numbering_schemes(synthetic_book: Path) -> None:
    doc: fitz.Document = fitz.open(synthetic_book)
    try:
        page_map = find_page_map(
            doc,
            [TocEntry(level=2, title="1.5 Properties of the Integral", page=12)],
            verbose=False,
        )
    finally:
        doc.close()

    assert page_map.offset == FRONT_MATTER_PAGES
    assert page_map.roman_offset == 0


def test_roman_entries_land_in_the_front_matter(
    synthetic_book: Path, tmp_path: Path
) -> None:
    """Regression: 'Preface -> ix' used to be treated as printed page 9."""
    entries = [
        TocEntry(level=1, title="Preface", page=3, page_ref=PageRef.PRINTED_ROMAN),
        TocEntry(level=1, title="1. Measure Theory", page=1),
        TocEntry(level=2, title="1.5 Properties of the Integral", page=12),
    ]
    output = tmp_path / "out.pdf"
    add_bookmarks(synthetic_book, entries, output, verbose=False)

    doc: fitz.Document = fitz.open(output)
    try:
        pages = {title: page for _level, title, page in doc.get_toc()}
    finally:
        doc.close()

    # Roman page iii is the third PDF page, not page 3 + 8 of the body.
    assert pages["Preface"] == 3
    # The arabic body still maps through the body offset.
    assert pages["1. Measure Theory"] == 1 + FRONT_MATTER_PAGES
    assert pages["1.5 Properties of the Integral"] == 12 + FRONT_MATTER_PAGES


def test_pdf_frame_entries_are_not_shifted(
    synthetic_book: Path, tmp_path: Path
) -> None:
    """Section-header scanning reports PDF pages; they must pass through."""
    entries = [
        TocEntry(level=1, title="Scanned Heading", page=20, page_ref=PageRef.PDF),
        TocEntry(level=1, title="Printed Heading", page=20),
    ]
    output = tmp_path / "out.pdf"
    add_bookmarks(synthetic_book, entries, output, verbose=False)

    doc: fitz.Document = fitz.open(output)
    try:
        pages = {title: page for _level, title, page in doc.get_toc()}
    finally:
        doc.close()

    assert pages["Scanned Heading"] == 20
    assert pages["Printed Heading"] == 20 + FRONT_MATTER_PAGES

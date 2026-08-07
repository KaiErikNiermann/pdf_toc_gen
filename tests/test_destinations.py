"""Tests for malformed bookmark destinations.

Regression cover for a real book (Lyons, *Probability on Trees and Networks*)
whose 191-entry contents named every page correctly yet could not be navigated:
each destination was written `/Dest [<page> /XYZ 0]`, where the spec requires
`/XYZ left top zoom`. The pages were right, so every content-based check passed
and the file was copied through with its broken destinations intact.
"""

import math
from pathlib import Path

import pymupdf
import pytest

from pdftoc.bookmarks import (
    BookmarkDefect,
    find_malformed_destinations,
    get_existing_bookmarks,
    repair_destinations,
    verify_bookmarks,
)
from pdftoc.core import process_pdf

# Destination arrays as they appear after the page reference. Only the first is
# malformed; the rest are the well-formed forms this check must not fire on.
TRUNCATED_XYZ = "/XYZ 0"
WELL_FORMED = {
    "xyz": "/XYZ 72 806 0",
    "xyz_nulls": "/XYZ null null null",
    "fit": "/Fit",
    "fit_h": "/FitH 806",
    "fit_r": "/FitR 0 0 200 200",
}

CHAPTERS = [
    (1, "Chapter 1: Branching Number", 2),
    (2, "Section 1.1: Graph Terminology", 3),
    (1, "Chapter 2: Electric Networks", 6),
    (2, "Section 2.1: Circuit Basics", 8),
]


def _build_pdf(path: Path, dest_form: str, pages: int = 12) -> Path:
    """A PDF whose outline entries all use `dest_form` as their destination.

    Each target page carries its own title text so the content checks pass,
    isolating the destination as the only defect.
    """
    doc = pymupdf.open()
    for i in range(pages):
        page = doc.new_page()
        title = next((t for _, t, p in CHAPTERS if p == i + 1), f"Body page {i + 1}")
        page.insert_text((72, 100), title, fontsize=14)

    outline = doc.get_new_xref()
    items = [doc.get_new_xref() for _ in CHAPTERS]

    for idx, (_level, title, page_no) in enumerate(CHAPTERS):
        page_xref = doc.page_xref(page_no - 1)
        parts = [
            f"/Dest[{page_xref} 0 R{dest_form}]",
            f"/Parent {outline} 0 R",
            f"/Title({title})",
        ]
        if idx + 1 < len(items):
            parts.append(f"/Next {items[idx + 1]} 0 R")
        if idx:
            parts.append(f"/Prev {items[idx - 1]} 0 R")
        # Levels are flattened deliberately: nesting is irrelevant here, and a
        # flat list keeps the fixture's object graph easy to reason about.
        doc.update_object(items[idx], "<<" + "".join(parts) + ">>")

    doc.update_object(
        outline,
        f"<</Type/Outlines/Count {len(items)}/First {items[0]} 0 R"
        f"/Last {items[-1]} 0 R>>",
    )
    doc.xref_set_key(doc.pdf_catalog(), "Outlines", f"{outline} 0 R")
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def broken_pdf(tmp_path: Path) -> Path:
    return _build_pdf(tmp_path / "broken.pdf", TRUNCATED_XYZ)


# ============================================================================
# Detection
# ============================================================================


def test_truncated_xyz_is_detected(broken_pdf: Path) -> None:
    doc = pymupdf.open(broken_pdf)
    try:
        assert len(find_malformed_destinations(doc)) == len(CHAPTERS)
    finally:
        doc.close()


@pytest.mark.parametrize("form", sorted(WELL_FORMED), ids=sorted(WELL_FORMED))
def test_well_formed_destinations_are_not_flagged(tmp_path: Path, form: str) -> None:
    """Guards the check against firing on legitimate destination types."""
    path = _build_pdf(tmp_path / f"{form}.pdf", WELL_FORMED[form])

    doc = pymupdf.open(path)
    try:
        assert find_malformed_destinations(doc) == []
    finally:
        doc.close()


def test_broken_destinations_survive_the_page_checks(broken_pdf: Path) -> None:
    """The exact hole the Lyons book fell through.

    Its pages and titles were correct, so every content-based check passed;
    without a destination check the file was declared clean.
    """
    doc = pymupdf.open(broken_pdf)
    try:
        is_valid, issues = verify_bookmarks(doc, get_existing_bookmarks(doc), False)

        assert not is_valid, "malformed destinations must not read as a clean TOC"
        assert [i.defect for i in issues] == [BookmarkDefect.MALFORMED_DESTINATIONS], (
            "destinations should be the *only* complaint: pages and titles are fine"
        )
    finally:
        doc.close()


# ============================================================================
# Repair
# ============================================================================


def test_repair_fixes_destinations_and_keeps_the_table(
    broken_pdf: Path, tmp_path: Path
) -> None:
    doc = pymupdf.open(broken_pdf)
    try:
        before = doc.get_toc()
        repaired = repair_destinations(doc)
        out = tmp_path / "repaired.pdf"
        doc.save(out)
    finally:
        doc.close()

    assert repaired == len(CHAPTERS)

    fixed = pymupdf.open(out)
    try:
        assert find_malformed_destinations(fixed) == []
        assert fixed.get_toc() == before, "titles, levels and pages must be untouched"
        for entry in fixed.get_toc(simple=False):
            target = entry[3]["to"]
            assert math.isfinite(target.x) and math.isfinite(target.y)
    finally:
        fixed.close()


def test_process_pdf_repairs_instead_of_re_extracting(
    broken_pdf: Path, tmp_path: Path
) -> None:
    """End-to-end: a repairable document keeps its own contents.

    Re-extraction would be strictly worse here -- the document's hand-built
    table is better than anything inferred -- and it would cost OCR and an LLM.
    """
    out = tmp_path / "out.pdf"
    doc = pymupdf.open(broken_pdf)
    expected = doc.get_toc()
    doc.close()

    process_pdf(source=broken_pdf, output=out, skip_ocr=True, verbose=False)

    result = pymupdf.open(out)
    try:
        assert result.get_toc() == expected, "the original table should be preserved"
        assert find_malformed_destinations(result) == []
    finally:
        result.close()


def test_already_good_destinations_are_left_alone(tmp_path: Path) -> None:
    """A genuinely clean document must still take the fast copy path."""
    path = _build_pdf(tmp_path / "good.pdf", WELL_FORMED["xyz"])
    out = tmp_path / "good_out.pdf"

    process_pdf(source=path, output=out, skip_ocr=True, verbose=False)

    result = pymupdf.open(out)
    try:
        assert find_malformed_destinations(result) == []
        assert len(result.get_toc()) == len(CHAPTERS)
    finally:
        result.close()

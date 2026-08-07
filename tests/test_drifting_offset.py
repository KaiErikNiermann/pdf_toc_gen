"""Tests for drifting page offsets and keyword-less chapter lines.

Both regressions come from one real book (Aliprantis & Border, *Infinite
Dimensional Analysis*):

* Its printed-to-PDF offset is not constant. The PDF omits pages the printed
  book had, so the offset walks from +20 at chapter 1 to +13 by the index, and
  no single number is right for more than a stretch at a time.
* Its contents lists chapters as a bare number and title with no `Chapter`
  keyword, no dot after the number and no leader dots, so every chapter was
  dropped while the sections (which do have leaders) came through.
"""

import pytest

from pdftoc.page_labels import Folio, PageMap, PageMapSource, PageRef, resolve_page_map
from pdftoc.toc_extraction import _extract_dotted_leader_format

# ============================================================================
# Drifting offset
# ============================================================================


def _arabic(number: int) -> tuple[Folio, ...]:
    return (Folio(number=number, ref=PageRef.PRINTED_ARABIC),)


def _drifting_book() -> dict[int, tuple[Folio, ...]]:
    """Printed 1..60, where the PDF drops a page every 20 printed pages.

    Offset starts at +20 and decays, exactly like the book this models.
    """
    folios: dict[int, tuple[Folio, ...]] = {}
    pdf_page = 21
    for printed in range(1, 61):
        folios[pdf_page] = _arabic(printed)
        pdf_page += 1
        if printed % 20 == 0:  # a printed page with no PDF page of its own
            pdf_page -= 1
    return folios


def _resolve(folios: dict[int, tuple[Folio, ...]], **kwargs: object) -> PageMap:
    return resolve_page_map(
        folios_by_page=folios,
        titled_pages=[],
        page_text=lambda _page: "",
        total_pages=800,
        **kwargs,  # type: ignore[arg-type]
    )


def test_drifting_offset_is_tracked_per_page() -> None:
    folios = _drifting_book()
    page_map = _resolve(folios)

    for pdf_page, seen in folios.items():
        printed = seen[0].number
        assert page_map.to_pdf_page(printed, 800) == pdf_page, (
            f"printed {printed} should map to PDF {pdf_page}"
        )


def test_a_single_offset_cannot_express_the_drift() -> None:
    """Guards the premise: this is why anchors exist rather than a better constant."""
    folios = _drifting_book()
    anchors = _resolve(folios).folio_anchors
    offsets = {pdf - printed for printed, pdf in anchors}

    assert len(offsets) > 1, "fixture must actually drift"


def test_unobserved_pages_fall_back_to_the_nearest_anchor() -> None:
    """Pages with no folio of their own still land near the right place."""
    folios = _drifting_book()
    del folios[30]  # a page whose folio was never read
    page_map = _resolve(folios)

    # Printed 10 sat on PDF 30; the surrounding anchors still bracket it.
    assert page_map.to_pdf_page(10, 800) == 30


def test_stray_folio_in_the_back_matter_is_discarded() -> None:
    """An index page printing "1" must not anchor page 1 to the end of the book."""
    folios = _drifting_book()
    folios[780] = _arabic(1)  # "1" printed far past where printed 1 lives

    page_map = _resolve(folios)

    assert page_map.to_pdf_page(1, 800) == 21, "the outlier must not win"


def test_decorative_page_labels_do_not_override_observed_folios() -> None:
    """/PageLabels is metadata; some producers just count physical sheets.

    The real book carries a label tree claiming PDF page 2 is printed page 1,
    contradicting every folio on every page.
    """
    page_map = _resolve(_drifting_book(), label_offset=(1, "decorative"))

    assert page_map.source is not PageMapSource.PAGE_LABELS
    assert page_map.to_pdf_page(1, 800) == 21


def test_page_labels_still_used_when_no_folios_exist() -> None:
    """Without folio evidence the label tree remains the best source."""
    page_map = _resolve({}, label_offset=(12, "from /PageLabels"))

    assert page_map.source is PageMapSource.PAGE_LABELS
    assert page_map.to_pdf_page(1, 800) == 13


# ============================================================================
# Keyword-less chapter lines
# ============================================================================

# The two layouts a bare chapter line takes, depending on whether extraction
# groups the number into the title's span.
THREE_LINE = "1\nOdds and ends\n1\n"
ONE_LINE = "10 Charges and measures\n371\n"
SECTIONS = (
    "1.1\nNumbers . . . . . . . . . . . . . . . .\n1\n"
    "1.2\nSets . . . . . . . . . . . . . . . . . .\n2\n"
)


@pytest.mark.parametrize(
    ("text", "title", "page"),
    [
        (THREE_LINE, "1 Odds and ends", 1),
        (ONE_LINE, "10 Charges and measures", 371),
    ],
    ids=["three-line", "one-line"],
)
def test_chapter_without_keyword_or_leader_dots(
    text: str, title: str, page: int
) -> None:
    entries = _extract_dotted_leader_format(text, 800, False)

    assert [(e.title, e.page) for e in entries] == [(title, page)]


def test_chapters_and_sections_are_both_recovered() -> None:
    """The bug: sections parsed, chapters silently vanished."""
    entries = _extract_dotted_leader_format(THREE_LINE + SECTIONS, 800, False)
    titles = [e.title for e in entries]

    assert "1 Odds and ends" in titles
    assert any(t.startswith("1.1") for t in titles)
    assert any(t.startswith("1.2") for t in titles)


def test_chapter_pattern_does_not_eat_section_numbers() -> None:
    """`1.1` must not be read as chapter 1 titled "1"."""
    entries = _extract_dotted_leader_format(SECTIONS, 800, False)

    assert all(e.title.startswith(("1.1", "1.2")) for e in entries), (
        f"unexpected entries: {[e.title for e in entries]}"
    )


def test_page_number_followed_by_a_section_is_not_a_chapter() -> None:
    """A trailing page number must not pair with the next entry's number."""
    entries = _extract_dotted_leader_format(
        "18\n2.1\nFilters . . . . .\n33\n", 800, False
    )

    assert all(not e.title.startswith("18 ") for e in entries), (
        f"page number misread as a chapter: {[e.title for e in entries]}"
    )

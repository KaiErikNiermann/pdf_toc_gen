"""Entry shapes the deterministic parser must handle, drawn from a real corpus.

Text extraction splits a contents entry's number, title and page across one,
two, three or four lines depending on how the PDF groups its spans, and many
books print no leader dots at all. The patterns previously assumed a single
line ending in a dot run, so each shape below is a book that came out sparse or
empty.
"""

import pytest

from pdftoc.toc_extraction import _extract_dotted_leader_format, _is_plausible_title

TOTAL_PAGES = 800


def _parse(text: str) -> list[tuple[str, int]]:
    return [
        (e.title, e.page)
        for e in _extract_dotted_leader_format(text, TOTAL_PAGES, False)
    ]


# ============================================================================
# Separator shapes
# ============================================================================

SHAPES = {
    # Williams: keyword chapter, no leader dots, page on the next line. Only a
    # trailing space and a newline separate title from page -- one character
    # short of the run the dotted form demanded.
    "keyword-no-leaders": (
        "Chapter 0: A Branching-Process Example \n1\n",
        "Chapter 0: A Branching-Process Example",
        1,
    ),
    # Aliprantis: number and title share a line, page on the next.
    "number-shares-line": (
        "10 Charges and measures\n371\n",
        "10 Charges and measures",
        371,
    ),
    # Aliprantis: number alone, then title, then page.
    "number-own-line": ("1\nOdds and ends\n1\n", "1 Odds and ends", 1),
    # Fremlin: three-digit section numbering sharing the title's line.
    "three-digit-number": (
        "311 Boolean algebras\n13\n",
        "311 Boolean algebras",
        13,
    ),
    # Tao: section mark, trailing dot, title and page on following lines.
    "section-mark": ("§1.2.\nLebesgue measure\n17\n", "1.2 Lebesgue measure", 17),
    # Classic single line with leaders.
    "one-line-leaders": ("1.1 Numbers . . . . . . . 1\n", "1.1 Numbers", 1),
    # Leader dots given a line of their own -- a four-line entry.
    "leaders-own-line": (
        "1.6\nOrders and such\n. . . . . . . . . .\n7\n",
        "1.6 Orders and such",
        7,
    ),
}


@pytest.mark.parametrize(("text", "title", "page"), SHAPES.values(), ids=list(SHAPES))
def test_entry_shape_is_parsed(text: str, title: str, page: int) -> None:
    assert (title, page) in _parse(text)


# ============================================================================
# Guards against over-matching
# ============================================================================


def test_lone_three_digit_line_is_not_a_chapter_number() -> None:
    """A bare number on its own line is ambiguous with the previous page number.

    "311 / Introduction to Volume 3 / 11" must not become chapter 311: in a
    contents, a lone three-digit line is far more often a page.
    """
    entries = _parse("311\nIntroduction to Volume 3\n11\n")

    assert not any(t.startswith("311 ") for t, _ in entries), entries


def test_page_number_does_not_pair_with_the_next_entry() -> None:
    entries = _parse("18\n2.1\nFilters . . . . .\n33\n")

    assert not any(t.startswith("18 ") for t, _ in entries), entries


def test_section_numbers_are_not_read_as_chapters() -> None:
    entries = _parse("1.1\nNumbers . . . . .\n1\n")

    assert all(t.startswith("1.1") for t, _ in entries), entries


# ============================================================================
# Title plausibility
# ============================================================================


@pytest.mark.parametrize(
    "title",
    [
        "Part I: FOUNDATIONS",
        "Part I",  # a division marker is a complete title on its own
        "Appendix B",
        "L^p spaces (revisited)",  # math punctuation is not code
        "Sets & Functions",
        "σ-algebras",
        "0. Prerequisites",
        "Index",
    ],
)
def test_real_titles_are_kept(title: str) -> None:
    assert _is_plausible_title(title)


@pytest.mark.parametrize(
    "title",
    [
        "Contents",  # the page's own running head
        "Part XII: CONTENTS",  # that running head read as a part number
        "II",  # numbering with nothing left after it
        "x = f(y); return {a[0]}",  # a code listing, not a contents
    ],
)
def test_debris_is_rejected(title: str) -> None:
    assert not _is_plausible_title(title)


def test_running_head_does_not_become_a_part_entry() -> None:
    """Roman-numeral matching on a contents page invented "Part VI: Contents"."""
    entries = _parse("Part VI: Contents . . . . . . 40\n")

    assert entries == []

"""Tests for printed -> PDF page mapping.

Hermetic: `pdftoc.page_labels` is stdlib-only, so these need neither a PDF nor
the network.
"""

from __future__ import annotations

from pdftoc.page_labels import (
    PageMap,
    PageMapSource,
    folios_from_text,
    offset_from_folios,
    offset_from_keywords,
    resolve_page_map,
)

# A book with 8 pages of front matter: printed page N is PDF page N + 8.
FRONT_MATTER = 8


class TestFoliosFromText:
    """Reading the printed page number off a page's text."""

    def test_reads_folio_from_first_line(self) -> None:
        assert folios_from_text("42\nCHAPTER 3. LIMITS\nbody text here") == (42,)

    def test_reads_folio_from_last_line(self) -> None:
        assert folios_from_text("3.1. LIMITS\nbody text here\n43") == (43,)

    def test_ignores_numbers_in_body_text(self) -> None:
        assert folios_from_text("Chapter 3\n17 is prime\nmore text") == ()

    def test_rejects_implausibly_long_numbers(self) -> None:
        """Five digits is an identifier or an equation label, not a folio."""
        assert folios_from_text("Title\n123456") == ()

    def test_handles_blank_page(self) -> None:
        assert folios_from_text("   \n\n  ") == ()


class TestOffsetFromFolios:
    """Voting on the offset implied by observed folios."""

    def test_unanimous_folios_give_exact_offset(self) -> None:
        folios = {pdf: (pdf - FRONT_MATTER,) for pdf in range(9, 60)}
        result = offset_from_folios(folios)
        assert result is not None
        offset, consensus, observations = result
        assert offset == FRONT_MATTER
        assert consensus == 1.0
        assert observations == 51

    def test_tolerates_a_few_stray_readings(self) -> None:
        folios: dict[int, tuple[int, ...]] = {
            pdf: (pdf - FRONT_MATTER,) for pdf in range(9, 60)
        }
        folios[20] = (999,)  # a misread number
        result = offset_from_folios(folios)
        assert result is not None
        assert result[0] == FRONT_MATTER
        assert result[1] > 0.9

    def test_returns_none_without_observations(self) -> None:
        assert offset_from_folios({1: (), 2: ()}) is None


def _synthetic_book() -> tuple[dict[int, str], list[tuple[str, int]]]:
    """A book whose running headers repeat each chapter title on every page.

    This is the shape that defeats naive keyword search: searching for
    "Properties of the Integral" matches many pages, and the section's own
    words also appear in the surrounding discussion well before the section
    actually starts.
    """
    pages: dict[int, str] = {}
    for pdf_page in range(1, FRONT_MATTER + 1):
        pages[pdf_page] = "front matter\n"

    sections = {
        24: ("1.5", "Properties of the Integral"),
        28: ("1.6", "Expected Value"),
    }
    for printed in range(1, 41):
        header = "CHAPTER 1. MEASURE THEORY"
        body = "integration of measurable functions with properties of the integral"
        if printed in sections:
            num, title = sections[printed]
            page = f"{printed}\n{header}\n{num} {title}\n{body}"
        else:
            page = f"{printed}\n{header}\n{body}"
        pages[printed + FRONT_MATTER] = page

    entries = [
        (f"{num} {title}", printed) for printed, (num, title) in sections.items()
    ]
    return pages, entries


class TestOffsetFromKeywords:
    """The fallback used when no folios can be read (e.g. noisy scans)."""

    def test_ignores_running_header_false_positives(self) -> None:
        """Regression: the old code took the first match scanning from -20,
        so repeated running headers made it return far too small an offset."""
        pages, entries = _synthetic_book()
        # Strip folios so only the keyword path can succeed.
        text = {p: t.split("\n", 1)[-1] for p, t in pages.items()}

        result = offset_from_keywords(entries, lambda p: text[p], total_pages=len(text))
        assert result is not None
        assert result[0] == FRONT_MATTER

    def test_returns_none_when_nothing_matches(self) -> None:
        result = offset_from_keywords(
            [("Nonexistent Section", 5)], lambda _p: "unrelated", total_pages=10
        )
        assert result is None


class TestResolvePageMap:
    """Precedence between the sources of evidence."""

    def test_folios_win_when_they_agree(self) -> None:
        pages, entries = _synthetic_book()
        page_map = resolve_page_map(
            folios_by_page={p: folios_from_text(t) for p, t in pages.items()},
            titled_pages=entries,
            page_text=lambda p: pages[p],
            total_pages=len(pages),
            # Deliberately wrong labels: strong folio evidence must override.
            label_offset=(99, "bogus"),
        )
        assert page_map.source == PageMapSource.FOLIO
        assert page_map.offset == FRONT_MATTER

    def test_falls_back_to_page_labels(self) -> None:
        page_map = resolve_page_map(
            folios_by_page={},
            titled_pages=[("Chapter 1", 1)],
            page_text=lambda _p: "",
            total_pages=50,
            label_offset=(12, "/PageLabels"),
        )
        assert page_map.source == PageMapSource.PAGE_LABELS
        assert page_map.offset == 12

    def test_falls_back_to_keywords(self) -> None:
        pages, entries = _synthetic_book()
        text = {p: t.split("\n", 1)[-1] for p, t in pages.items()}
        page_map = resolve_page_map(
            folios_by_page={},
            titled_pages=entries,
            page_text=lambda p: text[p],
            total_pages=len(text),
        )
        assert page_map.source == PageMapSource.KEYWORD
        assert page_map.offset == FRONT_MATTER

    def test_defaults_to_zero_without_evidence(self) -> None:
        page_map = resolve_page_map(
            folios_by_page={},
            titled_pages=[("Whatever", 3)],
            page_text=lambda _p: "nothing relevant",
            total_pages=5,
        )
        assert page_map.source == PageMapSource.DEFAULT
        assert page_map.offset == 0
        assert page_map.confidence == 0.0


class TestPageMapClamping:
    """Bookmarks must always point at a page that exists."""

    def test_clamps_below_first_page(self) -> None:
        assert PageMap(-10, 1.0, PageMapSource.FOLIO).to_pdf_page(3, 100) == 1

    def test_clamps_past_last_page(self) -> None:
        assert PageMap(50, 1.0, PageMapSource.FOLIO).to_pdf_page(99, 100) == 100

    def test_maps_within_range(self) -> None:
        assert PageMap(8, 1.0, PageMapSource.FOLIO).to_pdf_page(24, 100) == 32

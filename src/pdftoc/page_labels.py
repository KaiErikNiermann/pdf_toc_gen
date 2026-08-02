"""Printed-page-number (folio) detection and printed -> PDF page mapping.

A book's table of contents cites *printed* page numbers, but bookmarks must
point at *PDF* page indices. The two differ by an offset introduced by front
matter (title, copyright, preface, the TOC itself).

This module is stdlib-only so both the PyMuPDF pipeline (`bookmarks`) and the
Pyodide/browser pipeline (`toc_extraction_browser`) can share it.

Three sources of evidence, strongest first:

1. **Folios** — the page number printed in the top/bottom margin of each page.
   Every page casts a vote for `offset = pdf_page - printed_folio`; a large
   unanimous majority is effectively ground truth.
2. **Embedded page labels** — the PDF `/PageLabels` tree, when the producer
   bothered to write one.
3. **Keyword search** — locate a section title in the body text. Only a
   fallback: running headers repeat a chapter title on every page of that
   chapter, so a single probe is ambiguous and only the aggregate vote across
   many probes is meaningful.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "PageMap",
    "PageMapSource",
    "folios_from_text",
    "offset_from_folios",
    "offset_from_keywords",
    "resolve_page_map",
]


class PageMapSource(StrEnum):
    """Where a printed -> PDF page mapping came from."""

    FOLIO = "folio"  # page numbers printed in the page margins
    PAGE_LABELS = "page-labels"  # the PDF's own /PageLabels tree
    KEYWORD = "keyword"  # section titles located in body text
    DEFAULT = "default"  # nothing worked; assume no offset


@dataclass(frozen=True, slots=True)
class PageMap:
    """Resolved mapping from printed page numbers to PDF page indices.

    `pdf_page = printed_page + offset`, both 1-indexed.
    """

    offset: int
    confidence: float
    source: PageMapSource
    detail: str = ""

    def to_pdf_page(self, printed_page: int, total_pages: int) -> int:
        """Map a printed page number onto a 1-indexed PDF page, clamped in range."""
        return max(1, min(total_pages, printed_page + self.offset))


# A folio is a short bare integer on its own line. Cap the width so that stray
# years ("2019") and equation numbers are less likely to be mistaken for one.
_FOLIO_RE = re.compile(r"^(\d{1,4})$")

# Leading section number of a TOC title: "1.6.2 Integration to the Limit".
_SECTION_NUM_RE = re.compile(r"^(\d+(?:\.\d+)*)[.\)]?\s+")

# Words too common to identify a section by.
_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "under", "from", "their", "this", "that",
        "some", "other", "into", "over", "when", "which", "these", "those",
        "than", "then", "more", "most", "such", "also", "part", "chapter",
        "section", "appendix", "introduction",
    }
)  # fmt: skip

_MIN_FOLIO_OBSERVATIONS = 5
_STRONG_FOLIO_OBSERVATIONS = 10
_STRONG_FOLIO_CONSENSUS = 0.8
_MIN_FOLIO_CONSENSUS = 0.6


def folios_from_text(text: str) -> tuple[int, ...]:
    """Folio candidates from a page's plain text: the first and last lines.

    Running headers push the folio to the last line on some pages and the first
    line on others, so both ends are checked.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ()
    candidates = {lines[0], lines[-1]}
    folios = {int(m.group(1)) for line in candidates if (m := _FOLIO_RE.match(line))}
    return tuple(sorted(f for f in folios if f > 0))


def offset_from_folios(
    folios_by_page: Mapping[int, Iterable[int]],
) -> tuple[int, float, int] | None:
    """Vote on the offset implied by observed folios.

    Args:
        folios_by_page: 1-indexed PDF page -> folio candidates seen on it.

    Returns:
        `(offset, consensus, observations)`, or None if nothing was observed.
        `consensus` is the winning offset's share of all votes.
    """
    votes: Counter[int] = Counter()
    for pdf_page, folios in folios_by_page.items():
        for folio in folios:
            votes[pdf_page - folio] += 1

    total = sum(votes.values())
    if not total:
        return None

    offset, count = votes.most_common(1)[0]
    return offset, count / total, total


def _probe_terms(title: str) -> tuple[str | None, tuple[str, ...]]:
    """Split a TOC title into its section number and its distinctive words."""
    match = _SECTION_NUM_RE.match(title.strip())
    section_num = match.group(1) if match else None
    rest = title[match.end() :] if match else title
    return section_num, tuple(
        word.lower()
        for word in re.findall(r"[A-Za-z]{4,}", rest)
        if word.lower() not in _STOPWORDS
    )


def _score_page(text: str, section_num: str | None, words: Sequence[str]) -> float:
    """Score how much a page looks like the *start* of a given section.

    Substring matching alone is far too weak — a chapter's running header
    repeats its title on every page of the chapter. Position matters: a real
    section start has the title near the top, ideally next to its number.
    """
    if not words:
        return 0.0

    lowered = text.lower()
    # Heading zone: the top of the page, where a section title would appear.
    heading = lowered[: max(400, len(lowered) // 4)]

    hits = sum(1 for w in words if re.search(rf"\b{re.escape(w)}\b", lowered))
    coverage = hits / len(words)
    if coverage < 0.75:  # require nearly all distinctive words to be present
        return 0.0

    score = coverage
    if all(re.search(rf"\b{re.escape(w)}\b", heading) for w in words):
        score += 1.0  # the words appear together near the top
    if section_num and re.search(rf"(?<!\d){re.escape(section_num)}(?!\d)", heading):
        score += 2.0  # numbered heading in the heading zone: strongest signal
    return score


def offset_from_keywords(
    titled_pages: Sequence[tuple[str, int]],
    page_text: Callable[[int], str],
    total_pages: int,
    skip_pages: frozenset[int] = frozenset(),
    search_range: tuple[int, int] = (-30, 60),
    max_probes: int = 12,
) -> tuple[int, float, int] | None:
    """Locate section titles in the body text and vote on the implied offset.

    Args:
        titled_pages: `(title, printed_page)` pairs from the extracted TOC.
        page_text: 1-indexed PDF page -> its text.
        total_pages: Page count of the document.
        skip_pages: 1-indexed PDF pages to ignore (e.g. the TOC itself).
        search_range: Inclusive-exclusive offsets to try.
        max_probes: Cap on probes, sampled evenly across the document.

    Returns:
        `(offset, agreement, probes_used)`, or None if no probe matched.
        `agreement` is the fraction of probes endorsing the winning offset.
    """
    probes = [
        (title, printed, *_probe_terms(title))
        for title, printed in sorted(titled_pages, key=lambda tp: tp[1])
        if printed >= 1
    ]
    probes = [p for p in probes if p[3]]  # need at least one distinctive word
    if not probes:
        return None

    # Spread probes across the whole book rather than clustering at the front:
    # a wrong offset that happens to fit chapter 1 rarely fits chapter 9 too.
    if len(probes) > max_probes:
        step = len(probes) / max_probes
        probes = [probes[int(i * step)] for i in range(max_probes)]

    votes: Counter[int] = Counter()
    weights: defaultdict[int, float] = defaultdict(float)
    lo, hi = search_range

    for _title, printed, section_num, words in probes:
        # Score every candidate offset, then let this probe endorse only its
        # best-scoring ones. Taking the first match instead would systematically
        # return the lowest offset in range.
        scored: list[tuple[float, int]] = []
        for offset in range(lo, hi):
            pdf_page = printed + offset
            if not (1 <= pdf_page <= total_pages) or pdf_page in skip_pages:
                continue
            if (score := _score_page(page_text(pdf_page), section_num, words)) > 0:
                scored.append((score, offset))

        if not scored:
            continue
        best = max(score for score, _ in scored)
        for score, offset in scored:
            if score >= best - 1e-9:
                votes[offset] += 1
                weights[offset] += score

    if not votes:
        return None

    # Most endorsing probes wins; break ties on total score, then on the
    # smaller shift (front matter is usually short).
    offset = min(votes, key=lambda o: (-votes[o], -weights[o], abs(o)))
    return offset, votes[offset] / len(probes), len(probes)


def resolve_page_map(
    folios_by_page: Mapping[int, Iterable[int]],
    titled_pages: Sequence[tuple[str, int]],
    page_text: Callable[[int], str],
    total_pages: int,
    label_offset: tuple[int, str] | None = None,
    skip_pages: frozenset[int] = frozenset(),
) -> PageMap:
    """Resolve the printed -> PDF page mapping from all available evidence.

    Args:
        folios_by_page: 1-indexed PDF page -> folio candidates seen on it.
        titled_pages: `(title, printed_page)` pairs from the extracted TOC.
        page_text: 1-indexed PDF page -> its text.
        total_pages: Page count of the document.
        label_offset: `(offset, detail)` derived from the PDF's `/PageLabels`.
        skip_pages: 1-indexed PDF pages to ignore during keyword search.
    """
    folio = offset_from_folios(folios_by_page)

    # Observed folios beat everything when they agree overwhelmingly: they are
    # what the reader actually sees on the page.
    if (
        folio is not None
        and folio[2] >= _STRONG_FOLIO_OBSERVATIONS
        and folio[1] >= _STRONG_FOLIO_CONSENSUS
    ):
        offset, consensus, observations = folio
        return PageMap(
            offset=offset,
            confidence=consensus,
            source=PageMapSource.FOLIO,
            detail=f"{observations} folios, {consensus:.0%} agreement",
        )

    if label_offset is not None:
        offset, detail = label_offset
        return PageMap(
            offset=offset,
            confidence=0.9,
            source=PageMapSource.PAGE_LABELS,
            detail=detail,
        )

    if (
        folio is not None
        and folio[2] >= _MIN_FOLIO_OBSERVATIONS
        and folio[1] >= _MIN_FOLIO_CONSENSUS
    ):
        offset, consensus, observations = folio
        return PageMap(
            offset=offset,
            confidence=consensus,
            source=PageMapSource.FOLIO,
            detail=f"{observations} folios, {consensus:.0%} agreement",
        )

    keyword = offset_from_keywords(
        titled_pages, page_text, total_pages, skip_pages=skip_pages
    )
    if keyword is not None:
        offset, agreement, probes = keyword
        return PageMap(
            offset=offset,
            confidence=agreement,
            source=PageMapSource.KEYWORD,
            detail=f"{round(agreement * probes)}/{probes} probes agree",
        )

    return PageMap(
        offset=0,
        confidence=0.0,
        source=PageMapSource.DEFAULT,
        detail="no page-number evidence found",
    )

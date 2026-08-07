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
from bisect import bisect_left
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "Folio",
    "FolioVote",
    "PageMap",
    "PageMapSource",
    "PageRef",
    "folios_from_text",
    "format_page_label",
    "offset_from_folios",
    "offset_from_keywords",
    "parse_page_label",
    "resolve_page_map",
    "roman_to_int",
]


class PageRef(StrEnum):
    """The frame of reference a page number is expressed in.

    A bare integer is ambiguous: printed page 9 of the body, printed page "ix"
    of the front matter, and PDF page 9 are three different pages. Every page
    number must therefore carry the frame it belongs to, or it cannot be mapped
    onto a PDF page correctly.
    """

    PRINTED_ARABIC = "printed-arabic"  # "42" in the body's own numbering
    PRINTED_ROMAN = "printed-roman"  # "ix" in the front matter's numbering
    PDF = "pdf"  # already a 1-indexed PDF page; needs no mapping


#: Frames that denote a *printed* page number and so require an offset.
PRINTED_REFS = frozenset({PageRef.PRINTED_ARABIC, PageRef.PRINTED_ROMAN})


class PageMapSource(StrEnum):
    """Where a printed -> PDF page mapping came from."""

    FOLIO = "folio"  # page numbers printed in the page margins
    PAGE_LABELS = "page-labels"  # the PDF's own /PageLabels tree
    KEYWORD = "keyword"  # section titles located in body text
    DEFAULT = "default"  # nothing worked; assume no offset


@dataclass(frozen=True, slots=True)
class Folio:
    """A page number observed printed on a page."""

    number: int
    ref: PageRef  # PRINTED_ARABIC or PRINTED_ROMAN


@dataclass(frozen=True, slots=True)
class FolioVote:
    """The winning offset for one numbering scheme, and how well supported."""

    offset: int
    consensus: float
    observations: int

    def is_strong(self, min_observations: int, min_consensus: float) -> bool:
        """Whether enough observations agree for this offset to be trusted."""
        return self.observations >= min_observations and self.consensus >= min_consensus


@dataclass(frozen=True, slots=True)
class PageMap:
    """Resolved mapping from printed page numbers to PDF page indices.

    Books number their front matter and their body separately, so one offset
    is not enough: `offset` maps the arabic body, `roman_offset` maps the roman
    front matter. Both are 1-indexed, `pdf_page = printed_page + offset`.

    A single offset also assumes the two numberings stay in lockstep for the
    whole body, which they do not when a PDF omits pages the printed book had
    (dropped blanks, missing plates). The offset then drifts -- one real book
    walks from +20 at chapter 1 to +13 by the index -- and any constant is
    wrong almost everywhere. `folio_anchors` records printed pages actually
    observed on specific PDF pages, which pins the mapping locally and lets the
    offset serve only where nothing was observed.
    """

    offset: int
    confidence: float
    source: PageMapSource
    detail: str = ""
    roman_offset: int | None = None
    # Sorted, strictly increasing in both coordinates: (printed arabic, PDF page).
    folio_anchors: tuple[tuple[int, int], ...] = ()

    def _from_anchors(self, page: int) -> int | None:
        """Locate `page` against the observed folios, or None if unanchored.

        An exact observation wins outright. Otherwise the surrounding anchors
        bracket the answer: pages between two anchors are shifted by the nearer
        one's offset, which tracks drift instead of averaging it away.
        """
        if not self.folio_anchors:
            return None

        index = bisect_left(self.folio_anchors, (page, 0))
        if index < len(self.folio_anchors) and self.folio_anchors[index][0] == page:
            return self.folio_anchors[index][1]

        before = self.folio_anchors[index - 1] if index else None
        after = self.folio_anchors[index] if index < len(self.folio_anchors) else None
        nearest = min(
            (a for a in (before, after) if a is not None),
            key=lambda a: abs(a[0] - page),
            default=None,
        )
        if nearest is None:
            return None
        return page + (nearest[1] - nearest[0])

    def to_pdf_page(
        self,
        page: int,
        total_pages: int,
        ref: PageRef = PageRef.PRINTED_ARABIC,
    ) -> int:
        """Map a page number in frame `ref` onto a 1-indexed PDF page."""
        if ref == PageRef.PDF:
            return max(1, min(total_pages, page))

        if ref == PageRef.PRINTED_ROMAN:
            # Front matter is usually numbered from the very first PDF page, so
            # an offset of 0 is the right guess when nothing was observed.
            pdf_page = page + (
                self.roman_offset if self.roman_offset is not None else 0
            )
            # Roman numbering only ever covers the front matter, which is
            # exactly the pages preceding printed arabic page 1.
            if self.offset > 0:
                pdf_page = min(pdf_page, self.offset)
            return max(1, min(total_pages, pdf_page))

        anchored = self._from_anchors(page)
        pdf_page = anchored if anchored is not None else page + self.offset
        return max(1, min(total_pages, pdf_page))


# A folio is a short bare integer on its own line. Cap the width so that stray
# years ("2019") and equation numbers are less likely to be mistaken for one.
_FOLIO_RE = re.compile(r"^(\d{1,4})$")

_ROMAN_RE = re.compile(r"^[IVXLCDM]+$", re.IGNORECASE)
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}

# Front matter never runs to hundreds of pages, so a large roman value is a
# misparse — most often a stray capital ("C", "D", "M") on its own line.
_MAX_ROMAN_PAGE = 100

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
# Front matter is short, so roman folios are few even when detection works.
_MIN_ROMAN_OBSERVATIONS = 3


def roman_to_int(text: str) -> int | None:
    """Parse a roman numeral, or None if `text` is not one."""
    text = text.strip().upper()
    if not text or not _ROMAN_RE.match(text):
        return None

    total = 0
    previous = 0
    for char in reversed(text):
        value = _ROMAN_VALUES[char]
        total += -value if value < previous else value
        previous = max(previous, value)
    return total if total > 0 else None


def int_to_roman(number: int) -> str:
    """Render a positive integer as a lowercase roman numeral."""
    numerals = (
        (1000, "m"), (900, "cm"), (500, "d"), (400, "cd"),
        (100, "c"), (90, "xc"), (50, "l"), (40, "xl"),
        (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i"),
    )  # fmt: skip
    out: list[str] = []
    for value, numeral in numerals:
        count, number = divmod(number, value)
        out.append(numeral * count)
    return "".join(out)


def parse_page_label(value: object) -> tuple[int, PageRef] | None:
    """Parse a page number as printed, keeping its numbering scheme.

    `"ix"` and `9` are different pages; collapsing the roman one to an integer
    loses the only signal that it belongs to the front matter.
    """
    if isinstance(value, bool):  # bool is an int subclass; never a page number
        return None
    if isinstance(value, int):
        return (value, PageRef.PRINTED_ARABIC) if value > 0 else None
    if not isinstance(value, str):
        return None

    text = value.strip()
    if _FOLIO_RE.match(text):
        return (int(text), PageRef.PRINTED_ARABIC) if int(text) > 0 else None
    if (roman := roman_to_int(text)) is not None and roman <= _MAX_ROMAN_PAGE:
        return roman, PageRef.PRINTED_ROMAN
    return None


def format_page_label(page: int, ref: PageRef) -> str:
    """Render a page number back in its own numbering scheme."""
    return int_to_roman(page) if ref == PageRef.PRINTED_ROMAN else str(page)


def folios_from_text(text: str) -> tuple[Folio, ...]:
    """Folio candidates from a page's plain text: the first and last lines.

    Running headers push the folio to the last line on some pages and the first
    line on others, so both ends are checked.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ()

    folios: set[Folio] = set()
    for line in {lines[0], lines[-1]}:
        if (parsed := parse_page_label(line)) is not None:
            number, ref = parsed
            folios.add(Folio(number, ref))
    return tuple(sorted(folios, key=lambda f: (f.ref, f.number)))


def offset_from_folios(
    folios_by_page: Mapping[int, Iterable[Folio]],
) -> dict[PageRef, FolioVote]:
    """Vote on the offset implied by observed folios, per numbering scheme.

    Args:
        folios_by_page: 1-indexed PDF page -> folios seen on it.

    Returns:
        A `FolioVote` per scheme that had any observations. `consensus` is the
        winning offset's share of the votes cast within that scheme.
    """
    votes: defaultdict[PageRef, Counter[int]] = defaultdict(Counter)
    for pdf_page, folios in folios_by_page.items():
        for folio in folios:
            votes[folio.ref][pdf_page - folio.number] += 1

    result: dict[PageRef, FolioVote] = {}
    for ref, counter in votes.items():
        total = sum(counter.values())
        offset, count = counter.most_common(1)[0]
        result[ref] = FolioVote(offset, count / total, total)
    return result


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


@dataclass(frozen=True, slots=True)
class _Probe:
    """One TOC entry used to locate the body text it names."""

    printed_page: int
    section_num: str | None
    words: tuple[str, ...]


def _select_probes(
    titled_pages: Sequence[tuple[str, int]], max_probes: int
) -> tuple[_Probe, ...]:
    """Pick usable probes, spread evenly across the document.

    Spreading matters: a wrong offset that happens to fit chapter 1 rarely
    fits chapter 9 too.
    """
    probes: list[_Probe] = []
    for title, printed in sorted(titled_pages, key=lambda tp: tp[1]):
        if printed < 1:
            continue
        section_num, words = _probe_terms(title)
        if words:  # need at least one distinctive word
            probes.append(_Probe(printed, section_num, words))

    if len(probes) <= max_probes:
        return tuple(probes)
    step = len(probes) / max_probes
    return tuple(probes[int(i * step)] for i in range(max_probes))


def _endorsed_offsets(
    probe: _Probe,
    page_text: Callable[[int], str],
    total_pages: int,
    skip_pages: frozenset[int],
    search_range: tuple[int, int],
) -> tuple[tuple[int, float], ...]:
    """Offsets this probe scores best at, as `(offset, score)` pairs.

    Every candidate offset is scored and only the best-scoring ones are
    endorsed. Returning the first match instead would systematically yield the
    lowest offset in range, since running headers match on many pages.
    """
    scored: list[tuple[int, float]] = []
    for offset in range(*search_range):
        pdf_page = probe.printed_page + offset
        if not (1 <= pdf_page <= total_pages) or pdf_page in skip_pages:
            continue
        score = _score_page(page_text(pdf_page), probe.section_num, probe.words)
        if score > 0:
            scored.append((offset, score))

    if not scored:
        return ()
    best = max(score for _, score in scored)
    return tuple((o, s) for o, s in scored if s >= best - 1e-9)


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
    probes = _select_probes(titled_pages, max_probes)
    if not probes:
        return None

    votes: Counter[int] = Counter()
    weights: defaultdict[int, float] = defaultdict(float)
    for probe in probes:
        for offset, score in _endorsed_offsets(
            probe, page_text, total_pages, skip_pages, search_range
        ):
            votes[offset] += 1
            weights[offset] += score

    if not votes:
        return None

    # Most endorsing probes wins; break ties on total score, then on the
    # smaller shift (front matter is usually short).
    offset = min(votes, key=lambda o: (-votes[o], -weights[o], abs(o)))
    return offset, votes[offset] / len(probes), len(probes)


def _unambiguous_folios(
    folios_by_page: Mapping[int, Iterable[Folio]],
) -> list[tuple[int, int]]:
    """(printed arabic, PDF page) pairs where exactly one page claims the folio.

    A printed number claimed by several pages is running-header or index noise,
    not a folio, so it anchors nothing.
    """
    pages_by_folio: dict[int, set[int]] = {}
    for pdf_page, folios in folios_by_page.items():
        for folio in folios:
            if folio.ref == PageRef.PRINTED_ARABIC:
                pages_by_folio.setdefault(folio.number, set()).add(pdf_page)
    return sorted(
        (folio, next(iter(pages)))
        for folio, pages in pages_by_folio.items()
        if len(pages) == 1
    )


def _longest_consistent_run(
    pairs: list[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    """Keep the largest subset where PDF pages rise with printed pages.

    Both numberings only ever advance, so any pair breaking that order is a
    misread. Discarding the *fewest* pairs that restores monotonicity keeps the
    genuine folios and drops outliers -- a stray "1" printed in the back matter
    would otherwise anchor page 1 to the far end of the book.

    Longest strictly increasing subsequence over the PDF coordinate, with the
    predecessor chain kept so the winning run can be rebuilt.
    """
    if not pairs:
        return ()

    tail_index: list[int] = []  # position in `pairs` ending each run length
    previous: list[int] = [-1] * len(pairs)
    tails: list[int] = []  # smallest achievable end value per run length

    for i, (_printed, pdf_page) in enumerate(pairs):
        slot = bisect_left(tails, pdf_page)
        if slot == len(tails):
            tails.append(pdf_page)
            tail_index.append(i)
        else:
            tails[slot] = pdf_page
            tail_index[slot] = i
        previous[i] = tail_index[slot - 1] if slot else -1

    run: list[tuple[int, int]] = []
    node = tail_index[-1]
    while node != -1:
        run.append(pairs[node])
        node = previous[node]
    return tuple(reversed(run))


def _folio_anchors(
    folios_by_page: Mapping[int, Iterable[Folio]],
) -> tuple[tuple[int, int], ...]:
    """Printed-to-PDF anchors that survive both consistency filters."""
    anchors = _longest_consistent_run(_unambiguous_folios(folios_by_page))
    return anchors if len(anchors) >= _MIN_FOLIO_OBSERVATIONS else ()


def resolve_page_map(
    folios_by_page: Mapping[int, Iterable[Folio]],
    titled_pages: Sequence[tuple[str, int]],
    page_text: Callable[[int], str],
    total_pages: int,
    label_offset: tuple[int, str] | None = None,
    skip_pages: frozenset[int] = frozenset(),
) -> PageMap:
    """Resolve the printed -> PDF page mapping from all available evidence.

    Args:
        folios_by_page: 1-indexed PDF page -> folios seen on it.
        titled_pages: `(title, printed_arabic_page)` pairs from the TOC.
        page_text: 1-indexed PDF page -> its text.
        total_pages: Page count of the document.
        label_offset: `(offset, detail)` derived from the PDF's `/PageLabels`.
        skip_pages: 1-indexed PDF pages to ignore during keyword search.
    """
    votes = offset_from_folios(folios_by_page)
    arabic = votes.get(PageRef.PRINTED_ARABIC)
    roman_offset = _roman_offset(votes.get(PageRef.PRINTED_ROMAN))

    # Anchors are per-page evidence, so they stay useful even when the folios
    # disagree about any single offset -- which is exactly what a drifting book
    # looks like. Every branch below carries them.
    anchors = _folio_anchors(folios_by_page)

    # Observed folios beat everything when they agree overwhelmingly: they are
    # what the reader actually sees on the page.
    if arabic is not None and arabic.is_strong(
        _STRONG_FOLIO_OBSERVATIONS, _STRONG_FOLIO_CONSENSUS
    ):
        return _folio_map(arabic, roman_offset, anchors)

    # /PageLabels is only metadata, and some producers write a decorative
    # sequence that just counts the physical sheets. Anchors are what is
    # actually printed on the page, so labels may not override them.
    if label_offset is not None and not anchors:
        offset, detail = label_offset
        return PageMap(offset, 0.9, PageMapSource.PAGE_LABELS, detail, roman_offset)

    if arabic is not None and arabic.is_strong(
        _MIN_FOLIO_OBSERVATIONS, _MIN_FOLIO_CONSENSUS
    ):
        return _folio_map(arabic, roman_offset, anchors)

    if anchors:
        return _anchor_map(anchors, roman_offset)

    keyword = offset_from_keywords(
        titled_pages, page_text, total_pages, skip_pages=skip_pages
    )
    if keyword is not None:
        offset, agreement, probes = keyword
        return PageMap(
            offset,
            agreement,
            PageMapSource.KEYWORD,
            f"{round(agreement * probes)}/{probes} probes agree",
            roman_offset,
        )

    return PageMap(
        0, 0.0, PageMapSource.DEFAULT, "no page-number evidence found", roman_offset
    )


def _roman_offset(vote: FolioVote | None) -> int | None:
    """The front matter's offset, if its folios agree among themselves.

    Front matter is short, so this vote is always a small one; it is judged on
    its own rather than against the far larger arabic vote.
    """
    if vote is not None and vote.is_strong(
        _MIN_ROMAN_OBSERVATIONS, _MIN_FOLIO_CONSENSUS
    ):
        return vote.offset
    return None


def _folio_map(
    vote: FolioVote,
    roman_offset: int | None,
    anchors: tuple[tuple[int, int], ...] = (),
) -> PageMap:
    roman_detail = "" if roman_offset is None else f", roman {roman_offset:+d}"
    anchor_detail = f", {len(anchors)} anchors" if anchors else ""
    return PageMap(
        offset=vote.offset,
        confidence=vote.consensus,
        source=PageMapSource.FOLIO,
        detail=f"{vote.observations} folios, {vote.consensus:.0%} "
        f"agreement{roman_detail}{anchor_detail}",
        roman_offset=roman_offset,
        folio_anchors=anchors,
    )


def _anchor_map(
    anchors: tuple[tuple[int, int], ...], roman_offset: int | None
) -> PageMap:
    """A map carried entirely by per-page anchors.

    Reached when no single offset commands a majority -- the signature of a
    book whose offset drifts. The stored offset is only the fallback for
    printed pages no anchor covers, so it is taken from the last anchor, where
    the drift has gone furthest.
    """
    fallback = anchors[-1][1] - anchors[-1][0]
    spread = anchors[0][1] - anchors[0][0]
    return PageMap(
        offset=fallback,
        confidence=0.75,
        source=PageMapSource.FOLIO,
        detail=f"{len(anchors)} folio anchors, offset drifts "
        f"{spread:+d} to {fallback:+d}",
        roman_offset=roman_offset,
        folio_anchors=anchors,
    )

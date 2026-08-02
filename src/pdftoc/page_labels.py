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


@dataclass(frozen=True, slots=True)
class PageMap:
    """Resolved mapping from printed page numbers to PDF page indices.

    Books number their front matter and their body separately, so one offset
    is not enough: `offset` maps the arabic body, `roman_offset` maps the roman
    front matter. Both are 1-indexed, `pdf_page = printed_page + offset`.
    """

    offset: int
    confidence: float
    source: PageMapSource
    detail: str = ""
    roman_offset: int | None = None

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

        return max(1, min(total_pages, page + self.offset))


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

    # The roman front matter is short, so its vote is always a small one; take
    # it whenever it is self-consistent, independently of the arabic evidence.
    roman_vote = votes.get(PageRef.PRINTED_ROMAN)
    roman_offset = (
        roman_vote.offset
        if roman_vote is not None
        and roman_vote.observations >= _MIN_ROMAN_OBSERVATIONS
        and roman_vote.consensus >= _MIN_FOLIO_CONSENSUS
        else None
    )

    def folio_map(vote: FolioVote) -> PageMap:
        roman_detail = "" if roman_offset is None else f", roman {roman_offset:+d}"
        return PageMap(
            offset=vote.offset,
            confidence=vote.consensus,
            source=PageMapSource.FOLIO,
            detail=(
                f"{vote.observations} folios, "
                f"{vote.consensus:.0%} agreement{roman_detail}"
            ),
            roman_offset=roman_offset,
        )

    # Observed folios beat everything when they agree overwhelmingly: they are
    # what the reader actually sees on the page.
    if (
        arabic is not None
        and arabic.observations >= _STRONG_FOLIO_OBSERVATIONS
        and arabic.consensus >= _STRONG_FOLIO_CONSENSUS
    ):
        return folio_map(arabic)

    if label_offset is not None:
        offset, detail = label_offset
        return PageMap(
            offset=offset,
            confidence=0.9,
            source=PageMapSource.PAGE_LABELS,
            detail=detail,
            roman_offset=roman_offset,
        )

    if (
        arabic is not None
        and arabic.observations >= _MIN_FOLIO_OBSERVATIONS
        and arabic.consensus >= _MIN_FOLIO_CONSENSUS
    ):
        return folio_map(arabic)

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
            roman_offset=roman_offset,
        )

    return PageMap(
        offset=0,
        confidence=0.0,
        source=PageMapSource.DEFAULT,
        detail="no page-number evidence found",
        roman_offset=roman_offset,
    )

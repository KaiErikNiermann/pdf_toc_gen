"""Per-line features describing a contents page.

The regex pass decides whether a line starts an entry by matching shapes. That
works when a book prints one of the shapes it knows and fails silently
otherwise, which is what leaves half a contents unread on unusual layouts. The
same decision can be posed as: given this line and its neighbours, does it start
an entry, and at what depth?

These features are the input to that decision. Deliberately stdlib-only and
text-only:

* stdlib-only so the core package keeps no ML dependency and the features stay
  trivially testable;
* text-only so the model sees exactly the text the regex pass sees, which makes
  it a drop-in third strategy rather than a parallel pipeline. Layout features
  (indent, font size, weight) are the obvious next lever and would need span
  data that the current text pipeline discards.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields

# A leading entry number: "3", "3.1", "3.1.4", optionally marked and punctuated.
_LEADING_NUMBER = re.compile(r"^[§¶]?\s*(\d+(?:\.\d+)*)\s*[.):]?\s")
# The same, but the number is the entire line -- extraction often gives it its
# own span, which is why so many entries span several lines.
_ONLY_NUMBER = re.compile(r"^[§¶]?\s*(\d+(?:\.\d+)*)\s*[.):]?\s*$")
_LEADING_KEYWORD = re.compile(r"^\s*(chapter|part|section|appendix)\b", re.IGNORECASE)
_TRAILING_PAGE = re.compile(r"(\d{1,4})\s*$")
_LEADER_RUN = re.compile(r"(?:[.…·\-_][ \t]*){3,}")
_ROMAN_ONLY = re.compile(r"^\s*[ivxlcIVXLC]+\s*$")


@dataclass(frozen=True, slots=True)
class LineFeatures:
    """Numeric description of one contents line and its immediate context.

    Field order is the feature-vector order; `as_vector` and `FEATURE_NAMES`
    both derive from it so the two can never drift apart.
    """

    # -- shape of this line -------------------------------------------------
    length: float
    word_count: float
    alpha_fraction: float
    digit_fraction: float
    upper_fraction: float
    leading_spaces: float

    # -- what it starts with ------------------------------------------------
    has_leading_number: float
    number_depth: float  # 1 for "3", 2 for "3.1", 3 for "3.1.4"
    leading_number_value: float
    is_only_a_number: float
    has_leading_keyword: float
    starts_upper: float
    is_roman_only: float

    # -- what it ends with --------------------------------------------------
    has_trailing_page: float
    trailing_page_value: float
    has_leader_dots: float
    ends_with_colon: float

    # -- context ------------------------------------------------------------
    prev_is_only_number: float
    next_is_only_number: float
    next_has_leading_number: float
    prev_blank: float
    relative_position: float  # 0 at the top of the page, 1 at the bottom

    def as_vector(self) -> list[float]:
        return [getattr(self, f.name) for f in fields(self)]


FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in fields(LineFeatures))


def _fraction(text: str, predicate: object) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if predicate(c)) / len(text)  # type: ignore[operator]


def _only_number(line: str) -> bool:
    return bool(_ONLY_NUMBER.match(line))


def line_features(lines: list[str], index: int) -> LineFeatures:
    """Describe `lines[index]` together with the neighbours that disambiguate it.

    Context matters because a bare number is the single most ambiguous line on a
    contents page: it is an entry's number when a title follows, and the
    *previous* entry's page number when one does not.
    """
    line = lines[index]
    stripped = line.strip()
    previous = lines[index - 1] if index > 0 else ""
    following = lines[index + 1] if index + 1 < len(lines) else ""

    leading = _LEADING_NUMBER.match(stripped) or _ONLY_NUMBER.match(stripped)
    number = leading.group(1) if leading else ""
    trailing = _TRAILING_PAGE.search(stripped)

    return LineFeatures(
        length=float(len(stripped)),
        word_count=float(len(stripped.split())),
        alpha_fraction=_fraction(stripped, str.isalpha),
        digit_fraction=_fraction(stripped, str.isdigit),
        upper_fraction=_fraction(stripped, str.isupper),
        leading_spaces=float(len(line) - len(line.lstrip())),
        has_leading_number=float(bool(leading)),
        number_depth=float(number.count(".") + 1 if number else 0),
        leading_number_value=float(number.split(".")[0]) if number else 0.0,
        is_only_a_number=float(_only_number(stripped)),
        has_leading_keyword=float(bool(_LEADING_KEYWORD.match(stripped))),
        starts_upper=float(bool(stripped[:1].isupper())),
        is_roman_only=float(bool(_ROMAN_ONLY.match(stripped))),
        has_trailing_page=float(bool(trailing)),
        trailing_page_value=float(trailing.group(1)) if trailing else 0.0,
        has_leader_dots=float(bool(_LEADER_RUN.search(stripped))),
        ends_with_colon=float(stripped.endswith(":")),
        prev_is_only_number=float(_only_number(previous.strip())),
        next_is_only_number=float(_only_number(following.strip())),
        # Either form counts: the next entry announces itself with a number
        # whether or not its title shares the line. Testing only the
        # number-with-title form missed precisely the multi-line layout this
        # feature exists to detect.
        next_has_leading_number=float(
            bool(
                _LEADING_NUMBER.match(following.strip())
                or _ONLY_NUMBER.match(following.strip())
            )
        ),
        prev_blank=float(not previous.strip()),
        relative_position=index / max(1, len(lines) - 1),
    )


def features_for_page(lines: list[str]) -> list[LineFeatures]:
    """Describe every line of one contents page."""
    return [line_features(lines, i) for i in range(len(lines))]

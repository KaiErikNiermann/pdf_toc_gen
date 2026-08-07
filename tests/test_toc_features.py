"""Tests for the contents-line features behind the learned pass.

These are the model's entire view of a line, so a silently wrong feature would
be untraceable from the model's metrics. They are pure functions of text, which
makes them cheap to pin exactly.
"""

from pdftoc.toc_features import (
    FEATURE_NAMES,
    LineFeatures,
    features_for_page,
    line_features,
)


def _one(line: str, before: str = "", after: str = "") -> LineFeatures:
    return line_features([before, line, after], 1)


def test_vector_matches_declared_feature_names() -> None:
    """The two are derived from the same fields and must not drift apart."""
    vector = _one("1.1 Numbers . . . 3").as_vector()

    assert len(vector) == len(FEATURE_NAMES)
    assert all(isinstance(v, float) for v in vector)


def test_numbering_depth_is_captured() -> None:
    assert _one("3 Topology").number_depth == 1
    assert _one("3.1 Nets").number_depth == 2
    assert _one("3.1.4 Filters").number_depth == 3
    assert _one("Preface").number_depth == 0


def test_section_mark_does_not_hide_the_numbering() -> None:
    """Mathematics writes "§1.2"; the mark is noise, the number is signal."""
    assert _one("§1.2. Lebesgue measure").has_leading_number == 1.0
    assert _one("§1.2. Lebesgue measure").number_depth == 2


def test_bare_number_is_flagged() -> None:
    """The most ambiguous line on a contents page gets its own feature."""
    assert _one("17").is_only_a_number == 1.0
    assert _one("1.1").is_only_a_number == 1.0
    assert _one("17 Correspondences").is_only_a_number == 0.0


def test_context_distinguishes_a_number_from_its_neighbours() -> None:
    """A number line means different things depending on what follows it.

    "1.1 / Numbers / 3" -- the number starts an entry.
    "3 / 1.2 / Sets"    -- the number ended the previous one.
    """
    starts_entry = _one("1.1", before="Contents", after="Numbers")
    ends_entry = _one("3", before="Numbers", after="1.2")

    assert starts_entry.next_is_only_number == 0.0
    assert ends_entry.next_is_only_number == 1.0
    assert ends_entry.next_has_leading_number == 1.0


def test_leader_dots_and_trailing_page_are_detected() -> None:
    entry = _one("1.1 Numbers . . . . . . . 3")

    assert entry.has_leader_dots == 1.0
    assert entry.has_trailing_page == 1.0
    assert entry.trailing_page_value == 3.0


def test_keyword_lines_are_flagged() -> None:
    assert _one("Chapter 4: Measurability").has_leading_keyword == 1.0
    assert _one("Part I").has_leading_keyword == 1.0
    assert _one("Measurability").has_leading_keyword == 0.0


def test_roman_only_line_is_flagged() -> None:
    """Front-matter page numbers are roman and must not read as titles."""
    assert _one("xvii").is_roman_only == 1.0
    assert _one("xvii Preface").is_roman_only == 0.0


def test_relative_position_spans_the_page() -> None:
    page = features_for_page(["Contents", "1 A", "2 B", "3 C", "Index"])

    assert page[0].relative_position == 0.0
    assert page[-1].relative_position == 1.0


def test_features_are_defined_for_first_and_last_line() -> None:
    """Context lookups must not run off either end of the page."""
    page = features_for_page(["only line"])

    assert len(page) == 1
    assert page[0].prev_blank == 1.0

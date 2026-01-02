"""Section header extraction using NLP heuristics."""

import re
from functools import lru_cache
from pathlib import Path

import fitz  # type: ignore
import yaml

from pdftoc.models import TocEntry


def _load_word_list(filename: str) -> set[str]:
    """Load a word list from a YAML file in the data directory."""
    data_dir = Path(__file__).parent / "data"
    filepath = data_dir / filename
    with filepath.open() as f:
        data = yaml.safe_load(f)
    return set(data.get("words", []))


@lru_cache(maxsize=1)
def _get_academic_vocabulary() -> set[str]:
    """Get academic section vocabulary (cached)."""
    return _load_word_list("academic_vocabulary.yaml")


@lru_cache(maxsize=1)
def _get_body_text_starters() -> set[str]:
    """Get body text starter words (cached)."""
    return _load_word_list("body_text_starters.yaml")


def extract_section_headers(doc: fitz.Document, verbose: bool) -> list[TocEntry]:
    """
    Extract TOC entries by scanning document for section headers.

    Uses NLP heuristics including:
    - Academic vocabulary boosting (Introduction, Methods, Results, etc.)
    - Title structure analysis (title case, no trailing punctuation)
    - Negative pattern filtering (body text starters, references)

    This is useful for academic papers and documents without a traditional TOC page.
    """
    toc_entries: list[TocEntry] = []
    seen: set[tuple[str, int]] = set()

    if verbose:
        print("Scanning document for section headers...")

    for page_idx in range(len(doc)):
        page: fitz.Page = doc[page_idx]
        text = page.get_text()
        lines = [line.strip() for line in text.split("\n")]

        i = 0
        while i < len(lines):
            line = lines[i]
            if not line or len(line) > 100:
                i += 1
                continue

            # Score and match section header
            score, entry = _score_section_header(line, page_idx + 1)

            # If no match and this looks like a section number alone,
            # try combining with next line (but be careful of page numbers!)
            if not entry and i + 1 < len(lines):
                next_line = lines[i + 1]
                if next_line and len(next_line) < 80:
                    # Check if current line is just a number (potential section number)
                    if re.match(r"^\d+(\.\d+)*$", line):
                        # Skip if this looks like a page number header/footer
                        # (single number at very top or bottom of page)
                        is_likely_page_number = False
                        try:
                            num = int(line.split(".")[0])
                            # Page numbers are typically in first 3 or last 3 lines
                            is_header_footer_position = i < 3 or i >= len(lines) - 3
                            # And often match the page number
                            matches_page_num = num == page_idx + 1
                            is_likely_page_number = (
                                is_header_footer_position and matches_page_num
                            )
                        except ValueError:
                            pass

                        if is_likely_page_number:
                            i += 1
                            continue

                        # Only combine if next line looks like a title (starts with capital)
                        if re.match(r"^[A-Z][A-Za-z]", next_line):
                            combined = f"{line} {next_line}"
                            score, entry = _score_section_header(combined, page_idx + 1)
                            if entry and score >= 0.4:
                                i += 1  # Skip the next line

            if entry and score >= 0.4:  # Threshold for acceptance
                key = (entry.title.lower(), entry.page)
                if key not in seen:
                    seen.add(key)
                    toc_entries.append(entry)

            i += 1

    # Sort by page, then by level
    toc_entries.sort(key=lambda e: (e.page, e.level))

    if verbose:
        print(f"Found {len(toc_entries)} section headers")
        for entry in toc_entries[:10]:
            print(f"  L{entry.level}: {entry.title} -> p.{entry.page}")
        if len(toc_entries) > 10:
            print(f"  ... and {len(toc_entries) - 10} more")

    return toc_entries


def _score_section_header(
    line: str,
    page_num: int,
) -> tuple[float, TocEntry | None]:
    """
    Score a line as a potential section header using NLP heuristics.

    Returns (score, entry) where score is 0.0-1.0 confidence.
    """
    # Quick rejection for obvious non-headers
    if len(line) < 3 or len(line) > 80:
        return 0.0, None

    # Try to match section numbering pattern
    entry = _try_match_section_pattern(line, page_num)
    if not entry:
        return 0.0, None

    # Start with base score - pattern match gives us a reasonable starting point
    score = 0.35

    # Load word lists
    academic_vocabulary = _get_academic_vocabulary()
    body_starters = _get_body_text_starters()

    # === Academic vocabulary heuristic (0.0 - 0.35 points) ===
    # This is our strongest signal for academic papers
    title_words = set(w.lower() for w in re.findall(r"[a-zA-Z]+", entry.title))
    academic_matches = title_words & academic_vocabulary
    if academic_matches:
        # More matches = higher confidence
        vocab_boost = min(0.35, len(academic_matches) * 0.15)
        score += vocab_boost

    # === Title structure heuristics ===
    title_part = re.sub(r"^\d+\.?\s*", "", entry.title).strip()

    # Boost for ALL CAPS titles (common in older papers)
    if title_part.isupper() and len(title_part) > 3:
        score += 0.2

    # Boost for title case (first letter of significant words capitalized)
    elif title_part and title_part[0].isupper():
        words = title_part.split()
        if len(words) >= 1 and all(
            w[0].isupper()
            or w.lower()
            in {"a", "an", "the", "of", "and", "in", "on", "for", "to", "with"}
            for w in words
            if w
        ):
            score += 0.1

    # Penalize titles ending with sentence punctuation (except colon for subtitles)
    if title_part.endswith((".", ",", ";")):
        score -= 0.2

    # Penalize very short single-word titles (unless academic vocab)
    word_count = len(title_part.split())
    if word_count == 1 and not academic_matches:
        score -= 0.15

    # Penalize very long titles (likely body text)
    if word_count > 10:
        score -= 0.2

    # === Negative patterns (strong rejection) ===
    # References pattern: "N. AUTHOR, ..." or "N. Name,"
    if re.match(r"^\d+\.\s+[A-Z][a-z]*\s*,", line):
        score -= 0.5

    # Body text starters - these strongly indicate body text, not headers
    first_word = title_part.split()[0].lower() if title_part else ""
    if first_word in body_starters:
        score -= 0.3

    # === Additional negative patterns ===
    # Author patterns: "N. A. Name and B. Name" or "N. Firstname Lastname"
    if re.match(
        r"^\d+\.\s+[A-Z]\.\s+[A-Z]", line
    ):  # "1. A. B. Name" or "1. K. Matsuda"
        score -= 0.5

    # Copyright/publication patterns
    if re.search(r"ACM|IEEE|Springer|©|\d{4}[-/]\d{2,4}", line):
        score -= 0.5

    # Reference numbers typically > 15
    sec_num_match = re.match(r"^(\d+)\.", entry.title)
    if sec_num_match:
        sec_num = int(sec_num_match.group(1))
        if sec_num > 15:  # Very unlikely to have 15+ main sections
            score -= 0.3
        if sec_num == 0:  # Section 0 is unusual
            score -= 0.3

    return max(0.0, min(1.0, score)), entry


def _try_match_section_pattern(line: str, page_num: int) -> TocEntry | None:
    """Try to match a line against section numbering patterns."""
    # Pattern: "Chapter N: Title" or "CHAPTER N Title"
    chapter_match = re.match(
        r"^(Chapter|CHAPTER)\s+(\d+)[:\s]+(.+)$", line, re.IGNORECASE
    )
    if chapter_match:
        num = chapter_match.group(2)
        title = chapter_match.group(3).strip()
        if title and len(title) > 2:
            return TocEntry(level=1, title=f"Chapter {num}: {title}", page=page_num)

    # Pattern: "N.N.N Title" (sub-subsection)
    subsubsec_match = re.match(r"^(\d+\.\d+\.\d+)\s+(.+)$", line)
    if subsubsec_match:
        num = subsubsec_match.group(1)
        title = subsubsec_match.group(2).strip()
        if len(title) >= 3:
            return TocEntry(level=4, title=f"{num} {title}", page=page_num)

    # Pattern: "N.N Title" (subsection)
    subsec_match = re.match(r"^(\d+\.\d+)\s+(.+)$", line)
    if subsec_match:
        num = subsec_match.group(1)
        title = subsec_match.group(2).strip()
        if len(title) >= 3:
            return TocEntry(level=3, title=f"{num} {title}", page=page_num)

    # Pattern: "N. Title" or "N Title" (main section)
    sec_match = re.match(r"^(\d{1,2})\.?\s+(.+)$", line)
    if sec_match:
        num = sec_match.group(1)
        title = sec_match.group(2).strip()
        if len(title) >= 3 and int(num) <= 20:
            return TocEntry(level=2, title=f"{num}. {title}", page=page_num)

    return None

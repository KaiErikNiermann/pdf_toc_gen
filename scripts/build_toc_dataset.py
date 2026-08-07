#!/usr/bin/env python3
"""Build a line-labelled contents-page dataset from PDFs that ship an outline.

The supervision is free and already correct: a publisher-authored outline says
exactly which headings exist and how they nest. Aligning those entries back onto
the lines of the book's own printed contents page yields, per line, whether it
starts an entry and at what depth -- which is precisely what the regex pass has
to guess.

Crucially this labels entries the regex *misses*, so a model trained on it can
close the recall gap rather than merely reproduce the patterns.

Usage:
    poetry run python scripts/build_toc_dataset.py --library PATH --out data/toc_lines.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdftoc.toc_extraction import find_toc_pages  # noqa: E402

warnings.filterwarnings("ignore")

# An outline this short is usually "Cover / Chapter 1 / Index" boilerplate
# rather than a real table, and gives too little signal to align against.
_MIN_OUTLINE = 8
_MIN_PAGES = 40

# Fraction of outline titles that must be found on the contents pages before the
# alignment is trusted. A book whose printed contents disagrees with its outline
# (abridged contents, appendices only in one) would otherwise contribute noise.
_MIN_ALIGNMENT = 0.45


@dataclass
class LabelledLine:
    """One line of a contents page, with what the publisher's outline says."""

    doc: str
    page: int
    index: int
    text: str
    # 0 when the line starts no entry, else the outline depth (1 = top level).
    level: int
    # Normalised outline title this line's entry span matched, "" when none.
    # Recorded because an entry's line text is not comparable to a parser's
    # output on its own: extraction splits "1.1 Numbers" across two lines, and
    # the number line alone normalises to nothing.
    key: str = ""


def normalise(title: str) -> str:
    """Collapse a title to a form comparable across outline and printed page."""
    title = re.sub(r"^\s*(chapter|part|section|appendix)\s+", " ", title, flags=re.I)
    title = re.sub(r"^\s*[\dIVXivx]+(?:\.\d+)*\s*[.):]?\s*", " ", title)
    return re.sub(r"[^a-z0-9]+", "", title.lower())


# An entry is spread over at most this many consecutive lines: its number, its
# title (occasionally wrapped), and its page.
_MAX_SPAN = 3


def _align_entries(
    lines: list[str], unique: dict[str, int], matched: set[str]
) -> tuple[list[int], list[str]]:
    """Label the line each outline entry *starts* on, consuming its span.

    Greedy left to right over windows of one to three lines. Labelling every
    line whose text merely resembles a title double-counts an entry that
    extraction split -- the number line and the title line both match once the
    numbering is normalised away -- and lets a page-number line be labelled
    because joining it with the following title happens to match. Consuming the
    matched window prevents both.
    """
    levels = [0] * len(lines)
    keys = [""] * len(lines)
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue

        hit = 0
        for span in range(1, _MAX_SPAN + 1):
            if i + span > len(lines):
                break
            key = normalise(" ".join(lines[i : i + span]))
            if len(key) >= 4 and key in unique:
                levels[i] = unique[key]
                keys[i] = key
                matched.add(key)
                hit = span
                break

        i += hit if hit else 1
    return levels, keys


def _label_document(pdf: Path) -> list[LabelledLine]:
    doc = pymupdf.open(pdf)
    try:
        outline = doc.get_toc()
        if len(outline) < _MIN_OUTLINE or len(doc) < _MIN_PAGES:
            return []

        toc_pages = find_toc_pages(doc)
        if not toc_pages:
            return []

        # Outline titles keyed by their normalised form. Ambiguous titles
        # ("Introduction" under several chapters) are dropped rather than
        # guessed at: a wrong level is worse than a missing one.
        by_title: dict[str, list[int]] = {}
        for level, title, _page in outline:
            key = normalise(title)
            if len(key) >= 4:
                by_title.setdefault(key, []).append(int(level))
        unique = {k: v[0] for k, v in by_title.items() if len(set(v)) == 1}

        rows: list[LabelledLine] = []
        matched: set[str] = set()
        for page_no, text in toc_pages:
            lines = [raw.rstrip() for raw in text.split("\n")]
            levels, keys = _align_entries(lines, unique, matched)
            rows.extend(
                LabelledLine(
                    doc=pdf.name,
                    page=page_no,
                    index=i,
                    text=line,
                    level=levels[i],
                    key=keys[i],
                )
                for i, line in enumerate(lines)
                if line.strip()
            )

        # Reject the document if too little of its outline was located: the
        # labels would be mostly false negatives.
        if not unique or len(matched) / len(unique) < _MIN_ALIGNMENT:
            return []
        return rows
    except Exception:  # noqa: BLE001 - one bad PDF must not stop the build
        return []
    finally:
        doc.close()


def _pdfs(library: Path) -> list[Path]:
    return sorted({p.resolve() for p in library.glob("*/*.pdf") if p.exists()})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    pdfs = _pdfs(args.library)
    print(f"scanning {len(pdfs)} PDFs from {args.library}")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        batches = list(pool.map(_label_document, pdfs))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    docs = entries = lines = 0
    with args.out.open("w") as handle:
        for batch in batches:
            if not batch:
                continue
            docs += 1
            for row in batch:
                handle.write(json.dumps(asdict(row)) + "\n")
                lines += 1
                entries += row.level > 0

    print(f"usable documents : {docs}")
    print(f"contents lines   : {lines}")
    print(f"  labelled entry : {entries} ({entries / lines:.1%})" if lines else "")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

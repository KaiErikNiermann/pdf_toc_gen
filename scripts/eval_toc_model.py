#!/usr/bin/env python3
"""Measure what the learned pass would add on top of the regex pass.

Raw F1 is the wrong question. The model is not a replacement -- the regex pass
is far more precise about what it does find -- so the number that matters is:
of the entries the regex misses, how many does the model recover, and at what
cost in false positives?

Answered on held-out documents across a range of confidence thresholds, so the
operating point can be chosen rather than assumed.

Usage:
    poetry run python scripts/eval_toc_model.py --data data/toc_lines.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_toc_dataset import normalise  # noqa: E402
from train_toc_model import split_by_document  # noqa: E402

from pdftoc.toc_extraction import (  # noqa: E402
    _extract_dotted_leader_format,
    _extract_line_by_line_format,
)
from pdftoc.toc_features import line_features  # noqa: E402


@dataclass(frozen=True, slots=True)
class Sample:
    doc: str
    page_key: tuple[str, int]
    text: str
    is_entry: bool
    # Normalised outline title of this line's entry, recorded at build time.
    # Comparing the parser's output against the raw line text would understate
    # its coverage, because a multi-line entry's first line carries only the
    # number and normalises to nothing.
    key: str


def _regex_titles(lines: list[str]) -> set[str]:
    """Normalised titles the deterministic pass recovers from these lines."""
    text = "\n".join(lines)
    entries = max(
        _extract_dotted_leader_format(text, 5000, False),
        _extract_line_by_line_format(text, 5000, False),
        key=len,
    )
    return {normalise(e.title) for e in entries if normalise(e.title)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()

    pages: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for raw in args.data.read_text().splitlines():
        row = json.loads(raw)
        pages[(row["doc"], row["page"])].append(row)

    vectors: list[list[float]] = []
    samples: list[Sample] = []
    regex_titles: dict[tuple[str, int], set[str]] = {}

    for key, rows in pages.items():
        rows.sort(key=lambda r: int(r["index"]))  # type: ignore[arg-type]
        lines = [str(r["text"]) for r in rows]
        regex_titles[key] = _regex_titles(lines)
        for i, row in enumerate(rows):
            vectors.append(line_features(lines, i).as_vector())
            samples.append(
                Sample(
                    doc=key[0],
                    page_key=key,
                    text=str(row["text"]),
                    is_entry=int(row["level"]) > 0,  # type: ignore[arg-type]
                    key=str(row.get("key", "")),
                )
            )

    x = np.array(vectors, dtype=float)
    y = np.array([s.is_entry for s in samples])
    doc_ids = np.array([s.doc for s in samples])

    train, test = split_by_document(doc_ids)
    model = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.08, random_state=0
    )
    model.fit(x[train], y[train])
    prob = model.predict_proba(x[test])[:, 1]

    held = [s for s, keep in zip(samples, test, strict=True) if keep]
    truth = y[test]
    # A true entry counts as already covered when the deterministic pass
    # recovered a title matching this line.
    covered = np.array(
        [bool(s.key) and s.key in regex_titles[s.page_key] for s in held]
    )
    missed = truth & ~covered

    print(f"held-out documents  {len({s.doc for s in held})}")
    print(f"held-out lines      {len(held)}")
    print(f"true entries        {int(truth.sum())}")
    print(f"  found by regex    {int((truth & covered).sum())}")
    print(f"  missed by regex   {int(missed.sum())}\n")

    print(f"{'thresh':>7}{'precision':>11}{'recall':>9}{'of-missed':>11}{'new-FP':>9}")
    for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        pred = prob >= t
        tp = int((pred & truth).sum())
        fp = int((pred & ~truth).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / max(1, int(truth.sum()))
        recovered = int((pred & missed).sum())
        share = recovered / max(1, int(missed.sum()))
        print(f"{t:>7.2f}{prec:>11.2f}{rec:>9.2f}{share:>10.0%}{fp:>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

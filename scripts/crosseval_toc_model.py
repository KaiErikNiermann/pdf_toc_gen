#!/usr/bin/env python3
"""Train on one corpus, evaluate on another that shares no documents.

The within-corpus split already holds out documents, but every book in it came
from the same shelf. Training on the papis library and testing on the
measure-theory corpus is the honest generalisation question: does this transfer
to books the model's training distribution never saw?

Usage:
    poetry run python scripts/crosseval_toc_model.py \
        --train data/toc_lines.jsonl --test data/eval_lines.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_toc_dataset import normalise  # noqa: E402

from pdftoc.toc_extraction import (  # noqa: E402
    _extract_dotted_leader_format,
    _extract_line_by_line_format,
)
from pdftoc.toc_features import line_features  # noqa: E402


def load(path: Path):
    pages: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for raw in path.read_text().splitlines():
        row = json.loads(raw)
        pages[(row["doc"], row["page"])].append(row)

    vectors, labels, keys, page_keys, docs = [], [], [], [], []
    regex_titles: dict[tuple[str, int], set[str]] = {}
    for key, rows in pages.items():
        rows.sort(key=lambda r: int(r["index"]))
        lines = [str(r["text"]) for r in rows]
        text = "\n".join(lines)
        found = max(
            _extract_dotted_leader_format(text, 5000, False),
            _extract_line_by_line_format(text, 5000, False),
            key=len,
        )
        regex_titles[key] = {normalise(e.title) for e in found if normalise(e.title)}
        for i, row in enumerate(rows):
            vectors.append(line_features(lines, i).as_vector())
            labels.append(int(row["level"]) > 0)
            keys.append(str(row.get("key", "")))
            page_keys.append(key)
            docs.append(key[0])
    return (
        np.array(vectors, float),
        np.array(labels),
        keys,
        page_keys,
        regex_titles,
        docs,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    args = parser.parse_args()

    x_tr, y_tr, *_ = load(args.train)
    x_te, y_te, keys, page_keys, regex_titles, docs = load(args.test)

    model = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.08, random_state=0
    )
    model.fit(x_tr, y_tr)
    prob = model.predict_proba(x_te)[:, 1]

    covered = np.array(
        [
            bool(k) and k in regex_titles[pk]
            for k, pk in zip(keys, page_keys, strict=True)
        ]
    )
    missed = y_te & ~covered

    print(f"train lines {len(y_tr)} | test lines {len(y_te)}")
    print(f"test documents {len(set(docs))}")
    print(f"true entries    {int(y_te.sum())}")
    print(
        f"  found by regex {int((y_te & covered).sum())} "
        f"({(y_te & covered).sum() / max(1, y_te.sum()):.0%})"
    )
    print(f"  missed         {int(missed.sum())}\n")

    print(f"{'thresh':>7}{'precision':>11}{'recall':>9}{'of-missed':>11}{'new-FP':>9}")
    for t in (0.5, 0.7, 0.8, 0.9, 0.95):
        pred = prob >= t
        tp, fp = int((pred & y_te).sum()), int((pred & ~y_te).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        print(
            f"{t:>7.2f}{prec:>11.2f}{tp / max(1, int(y_te.sum())):>9.2f}"
            f"{int((pred & missed).sum()) / max(1, int(missed.sum())):>10.0%}{fp:>9}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

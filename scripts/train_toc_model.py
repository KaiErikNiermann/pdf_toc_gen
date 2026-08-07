#!/usr/bin/env python3
"""Train and evaluate the contents-line classifier.

Splits by *document*, never by line: lines from one book are highly correlated,
so a random line split would leak the answer and report a score the model cannot
reproduce on an unseen book.

Usage:
    poetry run python scripts/train_toc_model.py --data data/toc_lines.jsonl \
        --out data/toc_model.joblib
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdftoc.toc_features import FEATURE_NAMES, line_features  # noqa: E402

# Depths past this are rare enough that the model cannot learn them and their
# presence only blurs the classes that matter.
MAX_LEVEL = 3


def load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, doc ids), rebuilding each page's line context."""
    pages: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for raw in path.read_text().splitlines():
        row = json.loads(raw)
        pages[(row["doc"], row["page"])].append(row)

    xs: list[list[float]] = []
    ys: list[int] = []
    docs: list[str] = []
    for (doc, _page), rows in pages.items():
        rows.sort(key=lambda r: int(r["index"]))  # type: ignore[arg-type]
        lines = [str(r["text"]) for r in rows]
        for i, row in enumerate(rows):
            xs.append(line_features(lines, i).as_vector())
            ys.append(min(int(row["level"]), MAX_LEVEL))  # type: ignore[arg-type]
            docs.append(doc)
    return np.array(xs, dtype=float), np.array(ys), np.array(docs)


def split_by_document(
    docs: np.ndarray, holdout: float = 0.3, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    names = np.array(sorted(set(docs.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(names)
    cut = max(1, int(len(names) * holdout))
    test_docs = set(names[:cut].tolist())
    is_test = np.array([d in test_docs for d in docs])
    return ~is_test, is_test


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    x, y, docs = load(args.data)
    print(f"lines {len(y)} | documents {len(set(docs.tolist()))}")
    print(f"label distribution: {dict(sorted(Counter(y.tolist()).items()))}")
    print("  (0 = not an entry, 1..3 = outline depth)\n")

    train, test = split_by_document(docs)
    print(f"train lines {train.sum()} | test lines {test.sum()} (split by document)\n")

    candidates = {
        "majority-baseline": DummyClassifier(strategy="most_frequent"),
        "logistic-regression": LogisticRegression(max_iter=2000),
        "random-forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=-1
        ),
        "gradient-boosting": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.08, random_state=0
        ),
    }

    best_name, best_model, best_score = "", None, -1.0
    for name, model in candidates.items():
        model.fit(x[train], y[train])
        pred = model.predict(x[test])
        macro = f1_score(y[test], pred, average="macro", zero_division=0)
        binary = f1_score(y[test] > 0, pred > 0, zero_division=0)
        print(f"{name:<22} macro-F1 {macro:.3f}   is-entry F1 {binary:.3f}")
        if macro > best_score:
            best_name, best_model, best_score = name, model, macro

    print(f"\nbest: {best_name}\n")
    assert best_model is not None
    pred = best_model.predict(x[test])
    print(
        classification_report(
            y[test],
            pred,
            target_names=["not-entry", "level-1", "level-2", "level-3"],
            zero_division=0,
        )
    )

    if hasattr(best_model, "feature_importances_"):
        order = np.argsort(best_model.feature_importances_)[::-1][:10]
        print("most informative features:")
        for i in order:
            print(f"  {FEATURE_NAMES[i]:<26}{best_model.feature_importances_[i]:.3f}")

    if args.out:
        import joblib

        args.out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": best_model, "features": FEATURE_NAMES}, args.out)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

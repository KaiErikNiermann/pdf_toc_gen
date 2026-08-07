"""Benchmark OCR backends on the same PDF: speed + accuracy proxy.

Design: a born-digital PDF already contains a perfect text layer (via PyMuPDF),
so we use that as ground truth. We rasterize each page to an image, OCR it with
each backend, and report wall-clock throughput plus a character-similarity proxy
(difflib ratio) against the ground-truth text. This gives a fair, fully-local
comparison without needing a hand-labelled scanned corpus.

Usage:
    poetry run python scripts/ocr_benchmark.py <pdf> [--pages N] [--start P]
        [--backends paddle,ocrmypdf,marker] [--device cpu|gpu]

Note: the similarity proxy measures agreement with the PDF's own text layer, not
absolute OCR truth. It is meaningful for *relative* comparison between engines on
clean born-digital pages; on genuinely degraded scans, judge by reading samples.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import fitz  # type: ignore

# Import the tool's real backends so we benchmark the exact code paths shipped.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pdftoc.models import OcrBackend  # noqa: E402
from pdftoc.ocr import extract_text  # noqa: E402

_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    """Collapse whitespace and lowercase for a layout-insensitive comparison."""
    return _WS.sub(" ", text).strip().lower()


def _similarity(truth: str, got: str) -> float:
    """Character-level similarity in [0, 1] (1.0 == identical after normalizing)."""
    t, g = _norm(truth), _norm(got)
    if not t and not g:
        return 1.0
    if not t or not g:
        return 0.0
    return SequenceMatcher(None, t, g).ratio()


@dataclass
class BackendResult:
    """Per-backend benchmark outcome."""

    backend: str
    ok: bool
    seconds: float
    pages: int
    total_chars: int
    mean_similarity: float
    error: str = ""

    @property
    def pages_per_min(self) -> float:
        return (self.pages / self.seconds * 60.0) if self.seconds > 0 else 0.0


def _make_subpdf(src: Path, start: int, count: int, dst: Path) -> int:
    """Write the first `count` pages starting at 0-indexed `start` to `dst`."""
    doc = fitz.open(src)
    try:
        end = min(start + count, len(doc)) - 1
        out = fitz.open()
        out.insert_pdf(doc, from_page=start, to_page=end)
        out.save(str(dst))
        n = len(out)
        out.close()
        return n
    finally:
        doc.close()


def _ground_truth(pdf: Path) -> dict[int, str]:
    """Embedded text layer per 1-indexed page (the comparison baseline)."""
    doc = fitz.open(pdf)
    try:
        return {i + 1: doc[i].get_text() for i in range(len(doc))}
    finally:
        doc.close()


def _run_backend(
    backend: OcrBackend, pdf: Path, truth: dict[int, str]
) -> BackendResult:
    name = backend.value
    try:
        t0 = time.perf_counter()
        page_texts = extract_text(pdf, backend=backend, language="eng", verbose=False)
        elapsed = time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001 - benchmark harness: report, don't crash
        return BackendResult(
            name, False, 0.0, 0, 0, 0.0, error=f"{type(e).__name__}: {e}"
        )

    sims = [_similarity(truth.get(p, ""), txt) for p, txt in page_texts.items()]
    mean_sim = sum(sims) / len(sims) if sims else 0.0
    total_chars = sum(len(t) for t in page_texts.values())
    return BackendResult(name, True, elapsed, len(page_texts), total_chars, mean_sim)


def _print_report(results: list[BackendResult], pages: int, sample: str) -> None:
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title=f"OCR backend benchmark ({pages} pages)")
        table.add_column("backend")
        table.add_column("ok")
        table.add_column("sec", justify="right")
        table.add_column("pages/min", justify="right")
        table.add_column("chars", justify="right")
        table.add_column("similarity", justify="right")
        table.add_column("note")
        for r in results:
            table.add_row(
                r.backend,
                "✓" if r.ok else "✗",
                f"{r.seconds:.1f}" if r.ok else "-",
                f"{r.pages_per_min:.1f}" if r.ok else "-",
                f"{r.total_chars:,}" if r.ok else "-",
                f"{r.mean_similarity:.3f}" if r.ok else "-",
                r.error[:60],
            )
        console.print(table)
    except ImportError:
        for r in results:
            print(r)
    if sample:
        print(sample)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--pages", type=int, default=5, help="number of pages to OCR")
    ap.add_argument("--start", type=int, default=0, help="0-indexed start page")
    ap.add_argument(
        "--backends",
        default="paddle,ocrmypdf",
        help="comma-separated: paddle,ocrmypdf,marker",
    )
    args = ap.parse_args()

    if not args.pdf.exists():
        print(f"No such file: {args.pdf}")
        return 1

    scratch = Path(__file__).resolve().parent.parent / ".bench_tmp.pdf"
    n = _make_subpdf(args.pdf, args.start, args.pages, scratch)
    truth = _ground_truth(scratch)
    print(f"Benchmarking {n} pages from {args.pdf.name} (start={args.start})\n")

    name_map = {
        "paddle": OcrBackend.PADDLE,
        "ocrmypdf": OcrBackend.OCRMYPDF,
        "marker": OcrBackend.MARKER,
    }
    results: list[BackendResult] = []
    sample = ""
    try:
        for token in args.backends.split(","):
            token = token.strip()
            if token not in name_map:
                continue
            print(f"Running {token} ...", flush=True)
            r = _run_backend(name_map[token], scratch, truth)
            results.append(r)
            print(f"  -> {'ok' if r.ok else r.error}\n", flush=True)
    finally:
        scratch.unlink(missing_ok=True)

    _print_report(results, n, sample)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

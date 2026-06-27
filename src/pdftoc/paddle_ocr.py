"""OCR processing using PaddleOCR PP-OCRv5 (classic detect+recognize pipeline).

This is the CPU-viable "middle tier" between the legacy tesseract/ocrmypdf
backend and the GPU-only marker/surya backend: markedly more accurate than
tesseract on degraded scans while still running at a usable pace on CPU.

PaddleOCR is a detect-then-recognize pipeline (not a vision-language model),
so it returns text line boxes rather than structured markdown. We render each
PDF page to an image, run the pipeline, sort the recognized lines into reading
order, and join them into per-page plaintext for TOC matching.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import fitz  # type: ignore

# Tesseract-style language codes (used elsewhere in the tool) -> PaddleOCR codes.
# PaddleOCR uses ISO-639-1-ish short codes; tesseract uses 3-letter codes.
_LANG_MAP: dict[str, str] = {
    "eng": "en",
    "deu": "german",
    "fra": "french",
    "spa": "es",
    "ita": "it",
    "por": "pt",
    "nld": "nl",
    "rus": "ru",
    "chi_sim": "ch",
    "chi_tra": "chinese_cht",
    "jpn": "japan",
    "kor": "korean",
}

# Render DPI for rasterizing PDF pages before OCR. 200 is a good accuracy/speed
# tradeoff for printed text; higher helps tiny fonts at a throughput cost.
_DEFAULT_DPI = 200

# Force the MOBILE detector for the perf-constrained CPU target. PaddleOCR's
# default picks the heavy PP-OCRv5_server_det, which is slow and RAM-hungry on
# CPU; the mobile detector is far lighter at a small accuracy cost. The
# recognition model is auto-selected per language (already a mobile variant).
_DEFAULT_DET_MODEL = "PP-OCRv5_mobile_det"

_paddle_available: bool | None = None
_ocr_singleton: Any | None = None
_singleton_key: tuple[str, str, bool] | None = None


def is_paddle_available() -> bool:
    """Check if PaddleOCR (and a paddle backend) is installed."""
    global _paddle_available
    if _paddle_available is None:
        try:
            import paddleocr  # type: ignore # noqa: F401

            _paddle_available = True
        except ImportError:
            _paddle_available = False
    return _paddle_available


def _to_paddle_lang(language: str) -> str:
    """Map a tesseract-style language code to a PaddleOCR language code.

    Falls back to the primary language if a combined code like 'eng+deu' is
    given, then to 'en' if unknown.
    """
    primary = language.split("+", 1)[0].strip().lower()
    return _LANG_MAP.get(primary, "en")


def unload_models() -> None:
    """Free the PaddleOCR singleton (mirrors marker_ocr.unload_models)."""
    global _ocr_singleton, _singleton_key
    _ocr_singleton = None
    _singleton_key = None
    gc.collect()


def _get_ocr(lang: str, device: str, enable_mkldnn: bool) -> Any:
    """Get or build the PaddleOCR pipeline (lazy singleton, keyed by config)."""
    global _ocr_singleton, _singleton_key
    key = (lang, device, enable_mkldnn)
    if _ocr_singleton is None or _singleton_key != key:
        from paddleocr import PaddleOCR  # type: ignore

        # Pin to PP-OCRv5 (well-validated classic pipeline) with the mobile
        # detector and mkldnn/oneDNN CPU acceleration on — mkldnn is ~6x faster
        # on CPU (18 vs 3 pages/min here). NOTE: this requires paddlepaddle 3.1.x;
        # paddle 3.3.x regressed the new-IR oneDNN path (ConvertPirAttribute crash),
        # so the env must pin paddle<3.2 for the fast CPU path. Skip the
        # doc-orientation / unwarping sub-models: printed book pages are already
        # upright, so they only add load time. Line-level angle classification stays.
        _ocr_singleton = PaddleOCR(
            lang=lang,
            ocr_version="PP-OCRv5",
            text_detection_model_name=_DEFAULT_DET_MODEL,
            device=device,
            enable_mkldnn=enable_mkldnn,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
        _singleton_key = key
    return _ocr_singleton


def _page_to_bgr(page: fitz.Page, dpi: int) -> Any:
    """Rasterize a PDF page to a BGR numpy array (PaddleOCR/OpenCV convention)."""
    import numpy as np

    pix = page.get_pixmap(dpi=dpi)
    # pixmap samples are RGB(A); reshape to (h, w, n) then drop alpha + flip to BGR.
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    rgb = arr[:, :, :3]
    return rgb[:, :, ::-1].copy()  # RGB -> BGR


def _lines_from_result(result: Any) -> list[tuple[float, float, str]]:
    """Normalize a PaddleOCR page result into (y, x, text) line tuples.

    Handles both the 3.x ``predict`` API (result objects exposing ``rec_texts``
    and ``rec_polys``/``dt_polys``) and the legacy 2.x ``ocr`` API
    (``[[box, (text, score)], ...]``).
    """
    lines: list[tuple[float, float, str]] = []

    # 3.x: result is a dict-like with parallel arrays.
    texts = _get_field(result, "rec_texts")
    polys = _get_field(result, "rec_polys")
    if polys is None:
        polys = _get_field(result, "dt_polys")
    if texts is not None and polys is not None:
        for text, poly in zip(texts, polys):
            if not text:
                continue
            ys = [float(p[1]) for p in poly]
            xs = [float(p[0]) for p in poly]
            lines.append((min(ys), min(xs), str(text)))
        return lines

    # 2.x: result is a list of [box, (text, score)] entries.
    if isinstance(result, list):
        for entry in result:
            try:
                box, (text, _score) = entry
            except (ValueError, TypeError):
                continue
            if not text:
                continue
            ys = [float(p[1]) for p in box]
            xs = [float(p[0]) for p in box]
            lines.append((min(ys), min(xs), str(text)))
    return lines


def _get_field(result: Any, name: str) -> Any:
    """Read a field from a PaddleOCR result that may be a dict, mapping, or object."""
    if isinstance(result, dict):
        return result.get(name)
    if hasattr(result, "__getitem__"):
        try:
            return result[name]
        except (KeyError, TypeError, IndexError):
            pass
    return getattr(result, name, None)


def _join_reading_order(lines: list[tuple[float, float, str]], line_tol: float = 10.0) -> str:
    """Sort recognized lines top-to-bottom, left-to-right and join into text.

    Lines whose top edges are within ``line_tol`` pixels are treated as the same
    visual row and ordered left-to-right within it.
    """
    if not lines:
        return ""
    lines = sorted(lines, key=lambda t: (t[0], t[1]))
    rows: list[list[tuple[float, float, str]]] = []
    for y, x, text in lines:
        if rows and abs(y - rows[-1][0][0]) <= line_tol:
            rows[-1].append((y, x, text))
        else:
            rows.append([(y, x, text)])
    out: list[str] = []
    for row in rows:
        row.sort(key=lambda t: t[1])
        out.append(" ".join(t[2] for t in row))
    return "\n".join(out)


def _run_page(ocr: Any, image: Any) -> Any:
    """Run a single page through PaddleOCR, tolerating 3.x vs 2.x APIs."""
    if hasattr(ocr, "predict"):
        results = ocr.predict(image)
        # predict returns a list (one entry per input image).
        return results[0] if results else None
    # Legacy 2.x: ocr.ocr(img, cls=True) -> [page_result]
    results = ocr.ocr(image)
    if results and isinstance(results, list):
        return results[0]
    return results


def extract_text_with_paddle(
    pdf_path: Path,
    verbose: bool = False,
    language: str = "eng",
    device: str = "cpu",
    dpi: int = _DEFAULT_DPI,
    enable_mkldnn: bool = True,
) -> dict[int, str]:
    """Extract text from a PDF using the PaddleOCR PP-OCRv5 pipeline.

    Args:
        pdf_path: Path to the PDF file.
        verbose: Whether to print progress information.
        language: Tesseract-style language code (e.g. 'eng'); mapped to Paddle's.
        device: 'cpu' or 'gpu' (requires the matching paddlepaddle build).
        dpi: Rasterization DPI for each page before OCR.
        enable_mkldnn: oneDNN CPU acceleration (~6x faster). Requires paddle 3.1.x;
            set False on paddle 3.2+ where the oneDNN new-IR path is broken.

    Returns:
        Dict mapping 1-indexed page numbers to extracted plaintext.
    """
    lang = _to_paddle_lang(language)
    if verbose:
        print(f"Using PaddleOCR (PP-OCRv5, lang={lang}, device={device}, dpi={dpi})...")

    ocr = _get_ocr(lang, device, enable_mkldnn)

    doc: fitz.Document = fitz.open(pdf_path)
    page_texts: dict[int, str] = {}
    try:
        total = len(doc)
        for i in range(total):
            if verbose and (i == 0 or (i + 1) % 25 == 0 or i + 1 == total):
                print(f"  PaddleOCR page {i + 1}/{total}")
            image = _page_to_bgr(doc[i], dpi)
            result = _run_page(ocr, image)
            page_texts[i + 1] = _join_reading_order(_lines_from_result(result))
    finally:
        doc.close()

    if verbose:
        total_chars = sum(len(t) for t in page_texts.values())
        print(
            f"Extracted text from {len(page_texts)} pages "
            f"({total_chars:,} total characters)"
        )

    return page_texts

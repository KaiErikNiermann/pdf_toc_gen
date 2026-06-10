"""OCR processing using marker-pdf (GPU-accelerated, high quality)."""

from __future__ import annotations

import gc
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import fitz  # type: ignore

# Modest batch-size bumps for 16GB+ GPUs. Only bump recognition/detection —
# larger defaults for layout/table_rec trigger more aggressive cell
# decomposition which adds OCR work without proportional throughput gain.
# torch.compile intentionally NOT enabled: on short-lived workloads the
# compile cost dwarfs the gain (verified empirically).
_SURYA_ENV_DEFAULTS: dict[str, str] = {
    "RECOGNITION_BATCH_SIZE": "384",
    "DETECTOR_BATCH_SIZE": "48",
}
for _k, _v in _SURYA_ENV_DEFAULTS.items():
    os.environ.setdefault(_k, _v)

# Processors we can safely skip for text extraction (no math/LLM output).
# TableProcessor MUST stay — TOC pages are laid out as tables and removing
# it drops nearly all content.
_SKIP_PROCESSORS = {
    "EquationProcessor",  # runs texify on every equation region — expensive
    "LLMTableProcessor",
    "LLMTableMergeProcessor",
    "LLMFormProcessor",
    "LLMComplexRegionProcessor",
    "LLMImageDescriptionProcessor",
    "LLMEquationProcessor",
    "LLMHandwritingProcessor",
    "LLMMathBlockProcessor",
    "LLMSectionHeaderProcessor",
    "LLMPageCorrectionProcessor",
    "DebugProcessor",
}

_marker_available: bool | None = None


def is_marker_available() -> bool:
    """Check if marker-pdf is installed."""
    global _marker_available
    if _marker_available is None:
        try:
            import marker.converters.pdf  # noqa: F401

            _marker_available = True
        except ImportError:
            _marker_available = False
    return _marker_available


# Lazy-loaded singletons
_artifact_dict: dict[str, Any] | None = None


def unload_models() -> None:
    """Free marker models from GPU memory."""
    global _artifact_dict
    _artifact_dict = None
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    gc.collect()


def _get_artifact_dict() -> dict[str, Any]:
    """Get or create the marker model artifacts (lazy-loaded singleton)."""
    global _artifact_dict
    if _artifact_dict is None:
        from marker.models import create_model_dict

        _artifact_dict = create_model_dict()
    return _artifact_dict


def _strip_markdown(text: str) -> str:
    """Strip markdown formatting to get clean plaintext for TOC matching."""
    # Remove headers markers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove bold/italic
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}(.+?)_{1,3}", r"\1", text)
    # Remove inline code
    text = re.sub(r"`(.+?)`", r"\1", text)
    # Remove links, keep text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove image references
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    # Remove horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    return text


def extract_text_with_marker(
    pdf_path: Path,
    verbose: bool = False,
    chunk_size: int = 100,
) -> dict[int, str]:
    """Extract text from a PDF using marker-pdf.

    Uses GPU acceleration automatically if available via PyTorch.
    Processes the PDF in chunks to avoid OOM on large documents.

    Args:
        pdf_path: Path to the PDF file.
        verbose: Whether to print progress information.
        chunk_size: Number of pages to process at once. Lower = less memory.

    Returns:
        Dict mapping 1-indexed page numbers to extracted text.
    """
    import torch

    doc: fitz.Document = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()

    # Small PDFs: process in one shot
    if total_pages <= chunk_size:
        return _process_pdf_chunk(pdf_path, page_offset=0, verbose=verbose)

    # Large PDFs: split into chunks, process each, merge results
    if verbose:
        print(
            f"Large PDF ({total_pages} pages), processing in chunks of {chunk_size}..."
        )

    page_texts: dict[int, str] = {}

    for start in range(0, total_pages, chunk_size):
        end = min(start + chunk_size, total_pages)
        chunk_num = start // chunk_size + 1
        total_chunks = (total_pages + chunk_size - 1) // chunk_size

        if verbose:
            print(f"Chunk {chunk_num}/{total_chunks}: pages {start + 1}-{end}")

        # Extract chunk to a temp PDF
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            doc = fitz.open(pdf_path)
            chunk_doc = fitz.open()
            chunk_doc.insert_pdf(doc, from_page=start, to_page=end - 1)
            chunk_doc.save(str(tmp_path))
            chunk_doc.close()
            doc.close()

            chunk_texts = _process_pdf_chunk(tmp_path, page_offset=start, verbose=False)
            page_texts.update(chunk_texts)
        finally:
            tmp_path.unlink(missing_ok=True)

        # Free GPU memory between chunks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    if verbose:
        total_chars = sum(len(t) for t in page_texts.values())
        print(
            f"Extracted text from {len(page_texts)} pages "
            f"({total_chars:,} total characters)"
        )

    return page_texts


def _process_pdf_chunk(
    pdf_path: Path,
    page_offset: int = 0,
    verbose: bool = False,
) -> dict[int, str]:
    """Process a single PDF (or chunk) through marker and return page texts."""
    from marker.config.parser import ConfigParser
    from marker.converters.pdf import PdfConverter

    if verbose:
        print("Loading marker-pdf models (this may take a moment on first run)...")

    artifact_dict = _get_artifact_dict()

    config: dict[str, Any] = {"output_format": "json"}
    config_parser = ConfigParser(config)

    trimmed_processors = [
        f"{p.__module__}.{p.__name__}"
        for p in PdfConverter.default_processors
        if getattr(p, "__name__", "") not in _SKIP_PROCESSORS
    ]

    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=artifact_dict,
        processor_list=trimmed_processors,
        renderer=config_parser.get_renderer(),
    )

    if verbose:
        print(f"Running marker-pdf OCR on {pdf_path.name}...")

    rendered = converter(str(pdf_path))

    page_texts: dict[int, str] = {}
    children = getattr(rendered, "children", [])
    for page_idx, page_block in enumerate(children):
        page_text = _extract_block_text(page_block)
        page_text = _strip_markdown(page_text)
        page_texts[page_offset + page_idx + 1] = page_text  # 1-indexed

    # Drop the rendered object to free memory
    del rendered, children
    gc.collect()

    if verbose:
        total_chars = sum(len(t) for t in page_texts.values())
        print(
            f"Extracted text from {len(page_texts)} pages "
            f"({total_chars:,} total characters)"
        )

    return page_texts


def _extract_block_text(block: object) -> str:
    """Recursively extract text from a marker block and its children."""
    parts: list[str] = []

    # Try to get direct text/html content
    html = getattr(block, "html", None)
    if html and isinstance(html, str):
        # Strip HTML tags to get plain text
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            parts.append(text)

    # Recurse into children
    children = getattr(block, "children", None) or []
    for child in children:
        child_text = _extract_block_text(child)
        if child_text:
            parts.append(child_text)

    return "\n".join(parts)

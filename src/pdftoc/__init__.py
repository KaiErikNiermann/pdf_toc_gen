# PDF TOC Generator

from pdftoc.bookmarks import add_bookmarks, get_existing_bookmarks, verify_bookmarks
from pdftoc.core import process_pdf
from pdftoc.models import ExtractionMode, TocEntry
from pdftoc.ocr import pdf_has_text, run_ocr
from pdftoc.section_headers import extract_section_headers
from pdftoc.toc_extraction import extract_toc_from_text

__all__ = [
    "ExtractionMode",
    "TocEntry",
    "pdf_has_text",
    "run_ocr",
    "extract_toc_from_text",
    "extract_section_headers",
    "add_bookmarks",
    "get_existing_bookmarks",
    "verify_bookmarks",
    "process_pdf",
]

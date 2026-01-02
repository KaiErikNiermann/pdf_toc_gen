"""Data models for pdftoc."""

from dataclasses import dataclass
from enum import Enum


class ExtractionMode(Enum):
    """Mode for TOC extraction."""

    AUTO = "auto"  # Try TOC pages first, then section headers
    TOC_PAGE = "toc-page"  # Only look for TOC pages
    SECTION_HEADERS = "section-headers"  # Extract from section headers in content


@dataclass
class TocEntry:
    """A table of contents entry."""

    level: int
    title: str
    page: int

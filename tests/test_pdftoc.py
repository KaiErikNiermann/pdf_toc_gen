"""Tests for PDF TOC extraction and bookmark generation."""

from dataclasses import dataclass
from pathlib import Path

import fitz  # type: ignore
import pytest

from pdftoc.core import (
    TocEntry,
    add_bookmarks,
    extract_toc_from_text,
    pdf_has_text,
    process_pdf,
)


@dataclass
class ExpectedTocEntry:
    """Expected TOC entry for test validation."""

    title_contains: str
    page: int
    level: int


@dataclass
class PdfTestCase:
    """Test case definition for a PDF file."""

    name: str
    pdf_path: Path
    expected_entries: list[ExpectedTocEntry]
    min_total_entries: int
    has_text: bool = True


class TestPdfHasText:
    """Tests for pdf_has_text function."""

    def test_modal_logic_has_text(self, modal_logic_pdf: PdfTestCase) -> None:
        """Modal logic PDF should be detected as having text."""
        assert pdf_has_text(modal_logic_pdf.pdf_path) == modal_logic_pdf.has_text


class TestTocExtraction:
    """Tests for TOC extraction functionality."""

    def test_extracts_minimum_entries(self, modal_logic_pdf: PdfTestCase) -> None:
        """Should extract at least the minimum expected number of TOC entries."""
        doc: fitz.Document = fitz.open(modal_logic_pdf.pdf_path)
        try:
            entries = extract_toc_from_text(doc, verbose=False)
            assert len(entries) >= modal_logic_pdf.min_total_entries, (
                f"Expected at least {modal_logic_pdf.min_total_entries} entries, "
                f"got {len(entries)}"
            )
        finally:
            doc.close()

    def test_extracts_parts(self, modal_logic_pdf: PdfTestCase) -> None:
        """Should extract Part entries at level 1."""
        doc: fitz.Document = fitz.open(modal_logic_pdf.pdf_path)
        try:
            entries = extract_toc_from_text(doc, verbose=False)
            part_entries = [e for e in entries if "Part" in e.title]
            assert (
                len(part_entries) >= 4
            ), f"Expected 4 parts, found {len(part_entries)}"
        finally:
            doc.close()

    def test_extracts_chapters(self, modal_logic_pdf: PdfTestCase) -> None:
        """Should extract numbered chapter entries."""
        doc: fitz.Document = fitz.open(modal_logic_pdf.pdf_path)
        try:
            entries = extract_toc_from_text(doc, verbose=False)
            # Look for numbered chapters (e.g., "2. Basic language...")
            chapter_entries = [e for e in entries if e.title[0].isdigit()]
            assert (
                len(chapter_entries) >= 20
            ), f"Expected at least 20 chapters, found {len(chapter_entries)}"
        finally:
            doc.close()


class TestAddBookmarks:
    """Tests for bookmark addition functionality."""

    def test_adds_bookmarks_to_pdf(
        self, modal_logic_pdf: PdfTestCase, temp_output_pdf: Path
    ) -> None:
        """Should add bookmarks to the output PDF."""
        # Create some test entries
        test_entries = [
            TocEntry(level=1, title="Test Part", page=10),
            TocEntry(level=2, title="Test Chapter", page=20),
        ]

        add_bookmarks(
            modal_logic_pdf.pdf_path, test_entries, temp_output_pdf, verbose=False
        )

        # Verify bookmarks were added
        doc: fitz.Document = fitz.open(temp_output_pdf)
        try:
            toc = doc.get_toc()
            assert len(toc) == 2
            assert toc[0][1] == "Test Part"
            assert toc[1][1] == "Test Chapter"
        finally:
            doc.close()


class TestProcessPdf:
    """Integration tests for the full PDF processing pipeline."""

    def test_full_processing(
        self, modal_logic_pdf: PdfTestCase, temp_output_pdf: Path
    ) -> None:
        """Full processing should produce a PDF with correct bookmarks."""
        process_pdf(
            source=modal_logic_pdf.pdf_path,
            output=temp_output_pdf,
            skip_ocr=True,  # Skip OCR since PDF already has text
            verbose=False,
        )

        # Verify output exists and has bookmarks
        assert temp_output_pdf.exists()

        doc: fitz.Document = fitz.open(temp_output_pdf)
        try:
            toc = doc.get_toc()
            assert len(toc) >= modal_logic_pdf.min_total_entries
        finally:
            doc.close()

    def test_bookmark_hierarchy(
        self, modal_logic_pdf: PdfTestCase, temp_output_pdf: Path
    ) -> None:
        """Bookmarks should have proper hierarchy (Parts > Chapters)."""
        process_pdf(
            source=modal_logic_pdf.pdf_path,
            output=temp_output_pdf,
            skip_ocr=True,
            verbose=False,
        )

        doc: fitz.Document = fitz.open(temp_output_pdf)
        try:
            toc = doc.get_toc()

            # First entry should be level 1
            assert toc[0][0] == 1, "First TOC entry must be level 1"

            # Check that levels don't skip
            prev_level = 0
            for entry in toc:
                level = entry[0]
                assert (
                    level <= prev_level + 1
                ), f"Level jumped from {prev_level} to {level} at '{entry[1]}'"
                prev_level = level
        finally:
            doc.close()

    def test_expected_entries_present(
        self, modal_logic_pdf: PdfTestCase, temp_output_pdf: Path
    ) -> None:
        """Specific expected entries should be present with correct pages."""
        process_pdf(
            source=modal_logic_pdf.pdf_path,
            output=temp_output_pdf,
            skip_ocr=True,
            verbose=False,
        )

        doc: fitz.Document = fitz.open(temp_output_pdf)
        try:
            toc = doc.get_toc()

            for expected in modal_logic_pdf.expected_entries:
                _verify_entry_exists(toc, expected)
        finally:
            doc.close()

    def test_bookmarks_link_to_correct_content(
        self, modal_logic_pdf: PdfTestCase, temp_output_pdf: Path
    ) -> None:
        """Bookmark pages should contain the expected chapter content."""
        process_pdf(
            source=modal_logic_pdf.pdf_path,
            output=temp_output_pdf,
            skip_ocr=True,
            verbose=False,
        )

        doc: fitz.Document = fitz.open(temp_output_pdf)
        try:
            toc = doc.get_toc()

            # Spot check entries with multi-word titles
            checked = 0
            for entry in toc:
                level, title, page = entry[:3]

                # Extract key words from title (skip numbers, short words)
                keywords = [
                    w.lower() for w in title.split() if len(w) > 4 and w.isalpha()
                ]

                # Only check entries with at least 2 keywords
                if len(keywords) >= 2:
                    page_text = doc[page - 1].get_text().lower()
                    # At least one keyword should appear on the page
                    found = any(kw in page_text for kw in keywords[:3])
                    assert found, f"No keywords from '{title}' found on page {page}"
                    checked += 1
                    if checked >= 5:
                        break
        finally:
            doc.close()


def _verify_entry_exists(
    toc: list[list[int | str]], expected: ExpectedTocEntry
) -> None:
    """Verify an expected entry exists in the TOC."""
    for entry in toc:
        level, title, page = entry[:3]
        if expected.title_contains in str(title):
            # Allow some tolerance for page numbers (±5 pages due to offset detection variance)
            assert abs(int(page) - expected.page) <= 5, (
                f"Entry '{expected.title_contains}' at wrong page: "
                f"expected ~{expected.page}, got {page}"
            )
            assert level == expected.level, (
                f"Entry '{expected.title_contains}' at wrong level: "
                f"expected {expected.level}, got {level}"
            )
            return

    pytest.fail(
        f"Expected entry containing '{expected.title_contains}' not found in TOC"
    )

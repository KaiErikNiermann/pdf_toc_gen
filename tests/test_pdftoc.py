"""Tests for PDF TOC extraction and bookmark generation."""

from __future__ import annotations

from pathlib import Path

import fitz  # type: ignore
import pytest

from pdf_test_cases import ExpectedTocEntry, PdfTestCase, get_all_test_pdfs
from pdftoc.core import (
    ExtractionMode,
    TocEntry,
    add_bookmarks,
    extract_section_headers,
    extract_toc_from_text,
    get_existing_bookmarks,
    pdf_has_text,
    process_pdf,
    verify_bookmarks,
)


# ============================================================================
# Unit Tests - Basic functionality
# ============================================================================


class TestPdfHasText:
    """Tests for pdf_has_text function."""

    def test_detects_text_correctly(self, pdf_with_text: PdfTestCase) -> None:
        """PDFs with text should be detected as having text."""
        assert pdf_has_text(pdf_with_text.pdf_path) == pdf_with_text.has_text


class TestTocExtraction:
    """Tests for TOC extraction functionality."""

    def test_extracts_minimum_entries_toc_page(
        self, pdf_with_text: PdfTestCase
    ) -> None:
        """PDFs with TOC pages should extract entries from them."""
        if pdf_with_text.skip_content_check:
            pytest.skip(f"{pdf_with_text.name} uses section headers, not TOC page")

        doc: fitz.Document = fitz.open(pdf_with_text.pdf_path)
        try:
            entries = extract_toc_from_text(doc, verbose=False)
            assert len(entries) >= pdf_with_text.min_total_entries, (
                f"[{pdf_with_text.name}] Expected at least {pdf_with_text.min_total_entries} "
                f"entries, got {len(entries)}"
            )
        finally:
            doc.close()


class TestAddBookmarks:
    """Tests for bookmark addition functionality."""

    def test_adds_bookmarks_to_pdf(
        self, pdf_with_text: PdfTestCase, temp_output_pdf: Path
    ) -> None:
        """Should add bookmarks to the output PDF."""
        test_entries = [
            TocEntry(level=1, title="Test Part", page=10),
            TocEntry(level=2, title="Test Chapter", page=20),
        ]

        add_bookmarks(
            pdf_with_text.pdf_path, test_entries, temp_output_pdf, verbose=False
        )

        doc: fitz.Document = fitz.open(temp_output_pdf)
        try:
            toc = doc.get_toc()
            assert len(toc) == 2
            assert toc[0][1] == "Test Part"
            assert toc[1][1] == "Test Chapter"
        finally:
            doc.close()


# ============================================================================
# Integration Tests - Full pipeline
# ============================================================================


class TestProcessPdf:
    """Integration tests for the full PDF processing pipeline."""

    def test_full_processing(
        self, pdf_with_text: PdfTestCase, temp_output_pdf: Path
    ) -> None:
        """Full processing should produce a PDF with correct bookmarks."""
        process_pdf(
            source=pdf_with_text.pdf_path,
            output=temp_output_pdf,
            skip_ocr=True,
            verbose=False,
        )

        assert temp_output_pdf.exists()

        doc: fitz.Document = fitz.open(temp_output_pdf)
        try:
            toc = doc.get_toc()
            assert len(toc) >= pdf_with_text.min_total_entries, (
                f"[{pdf_with_text.name}] Expected at least {pdf_with_text.min_total_entries} "
                f"bookmarks, got {len(toc)}"
            )
        finally:
            doc.close()

    def test_bookmark_hierarchy(
        self, pdf_with_text: PdfTestCase, temp_output_pdf: Path
    ) -> None:
        """Bookmarks should have proper hierarchy (no level skips)."""
        process_pdf(
            source=pdf_with_text.pdf_path,
            output=temp_output_pdf,
            skip_ocr=True,
            verbose=False,
        )

        doc: fitz.Document = fitz.open(temp_output_pdf)
        try:
            toc = doc.get_toc()

            assert toc[0][0] == 1, "First TOC entry must be level 1"

            prev_level = 0
            for entry in toc:
                level = entry[0]
                assert level <= prev_level + 1, (
                    f"[{pdf_with_text.name}] Level jumped from {prev_level} to {level} "
                    f"at '{entry[1]}'"
                )
                prev_level = level
        finally:
            doc.close()

    def test_expected_entries_present(
        self, pdf_with_text: PdfTestCase, temp_output_pdf: Path
    ) -> None:
        """Specific expected entries should be present with correct pages."""
        if not pdf_with_text.expected_entries:
            pytest.skip(f"No expected entries defined for {pdf_with_text.name}")

        process_pdf(
            source=pdf_with_text.pdf_path,
            output=temp_output_pdf,
            skip_ocr=True,
            verbose=False,
        )

        doc: fitz.Document = fitz.open(temp_output_pdf)
        try:
            toc = doc.get_toc()
            for expected in pdf_with_text.expected_entries:
                _verify_entry_exists(toc, expected, pdf_with_text.name)
        finally:
            doc.close()

    def test_bookmarks_link_to_correct_content(
        self, pdf_with_text: PdfTestCase, temp_output_pdf: Path
    ) -> None:
        """Bookmark pages should contain the expected chapter content."""
        if pdf_with_text.skip_content_check:
            pytest.skip(f"Content check skipped for {pdf_with_text.name}")

        process_pdf(
            source=pdf_with_text.pdf_path,
            output=temp_output_pdf,
            skip_ocr=True,
            verbose=False,
        )

        doc: fitz.Document = fitz.open(temp_output_pdf)
        try:
            toc = doc.get_toc()
            checked = 0
            for entry in toc:
                level, title, page = entry[:3]
                keywords = [
                    w.lower() for w in title.split() if len(w) > 4 and w.isalpha()
                ]
                if len(keywords) >= 2:
                    page_text = doc[page - 1].get_text().lower()
                    found = any(kw in page_text for kw in keywords[:3])
                    assert found, (
                        f"[{pdf_with_text.name}] No keywords from '{title}' "
                        f"found on page {page}"
                    )
                    checked += 1
                    if checked >= 5:
                        break
        finally:
            doc.close()


# ============================================================================
# Modal Logic Specific Tests (backward compatibility)
# ============================================================================


class TestModalLogicSpecific:
    """Tests specific to the modal logic textbook structure."""

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
            chapter_entries = [e for e in entries if e.title[0].isdigit()]
            assert (
                len(chapter_entries) >= 20
            ), f"Expected at least 20 chapters, found {len(chapter_entries)}"
        finally:
            doc.close()


# ============================================================================
# Helper Functions
# ============================================================================


def _verify_entry_exists(
    toc: list[list[int | str]], expected: ExpectedTocEntry, pdf_name: str
) -> None:
    """Verify an expected entry exists in the TOC."""
    for entry in toc:
        level, title, page = entry[:3]
        if expected.title_contains in str(title):
            assert abs(int(page) - expected.page) <= 5, (
                f"[{pdf_name}] Entry '{expected.title_contains}' at wrong page: "
                f"expected ~{expected.page}, got {page}"
            )
            assert level == expected.level, (
                f"[{pdf_name}] Entry '{expected.title_contains}' at wrong level: "
                f"expected {expected.level}, got {level}"
            )
            return

    pytest.fail(
        f"[{pdf_name}] Expected entry containing '{expected.title_contains}' not found"
    )


# ============================================================================
# Section Header Extraction Tests
# ============================================================================


class TestSectionHeaderExtraction:
    """Tests for section header extraction mode."""

    def test_extracts_section_headers(self, pdf_with_text: PdfTestCase) -> None:
        """Section header extraction should find headers in the document."""
        doc: fitz.Document = fitz.open(pdf_with_text.pdf_path)
        try:
            entries = extract_section_headers(doc, verbose=False)
            # All PDFs should have some section headers
            assert len(entries) >= 1, (
                f"[{pdf_with_text.name}] Expected at least 1 section header, "
                f"got {len(entries)}"
            )
        finally:
            doc.close()

    def test_section_headers_have_valid_pages(self, pdf_with_text: PdfTestCase) -> None:
        """Section headers should point to valid page numbers."""
        doc: fitz.Document = fitz.open(pdf_with_text.pdf_path)
        try:
            entries = extract_section_headers(doc, verbose=False)
            for entry in entries:
                assert 1 <= entry.page <= len(doc), (
                    f"[{pdf_with_text.name}] Section '{entry.title}' has invalid "
                    f"page {entry.page}"
                )
        finally:
            doc.close()


# ============================================================================
# Bookmark Verification Tests
# ============================================================================


class TestBookmarkVerification:
    """Tests for bookmark verification functionality."""

    def test_detects_bad_bookmarks(self) -> None:
        """Should detect structurally bad bookmarks."""
        # Get a PDF with known bad bookmarks
        for test_case in get_all_test_pdfs():
            if (
                test_case.has_existing_bookmarks
                and test_case.expected_existing_bookmark_issues
            ):
                doc: fitz.Document = fitz.open(test_case.pdf_path)
                try:
                    existing = get_existing_bookmarks(doc)
                    is_valid, issues = verify_bookmarks(doc, existing, verbose=False)
                    assert (
                        not is_valid
                    ), f"[{test_case.name}] Expected bad bookmarks to be detected"
                    assert len(issues) > 0
                finally:
                    doc.close()
                return
        pytest.skip("No PDF with expected bookmark issues available")

    def test_accepts_good_bookmarks(
        self, pdf_with_text: PdfTestCase, temp_output_pdf: Path
    ) -> None:
        """Should accept well-formed bookmarks."""
        # Generate good bookmarks first
        process_pdf(
            source=pdf_with_text.pdf_path,
            output=temp_output_pdf,
            skip_ocr=True,
            verbose=False,
            mode=ExtractionMode.AUTO,
        )

        doc: fitz.Document = fitz.open(temp_output_pdf)
        try:
            existing = get_existing_bookmarks(doc)
            if len(existing) > 3:  # Only test if we have enough bookmarks
                is_valid, issues = verify_bookmarks(doc, existing, verbose=False)
                assert is_valid, (
                    f"[{pdf_with_text.name}] Generated bookmarks should be valid. "
                    f"Issues: {issues}"
                )
        finally:
            doc.close()


# ============================================================================
# Auto Mode Tests
# ============================================================================


class TestAutoMode:
    """Tests for auto extraction mode."""

    def test_auto_mode_finds_entries(
        self, pdf_with_text: PdfTestCase, temp_output_pdf: Path
    ) -> None:
        """Auto mode should find entries regardless of PDF type."""
        process_pdf(
            source=pdf_with_text.pdf_path,
            output=temp_output_pdf,
            skip_ocr=True,
            verbose=False,
            mode=ExtractionMode.AUTO,
        )

        doc: fitz.Document = fitz.open(temp_output_pdf)
        try:
            toc = doc.get_toc()
            assert len(toc) >= pdf_with_text.min_total_entries, (
                f"[{pdf_with_text.name}] Auto mode expected at least "
                f"{pdf_with_text.min_total_entries} entries, got {len(toc)}"
            )
        finally:
            doc.close()

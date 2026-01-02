"""Test fixtures and configuration for pdftoc tests."""

from dataclasses import dataclass
from pathlib import Path

import pytest

# Root directory for test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"
INPUT_DIR = FIXTURES_DIR / "input"
OUTPUT_DIR = FIXTURES_DIR / "output"

# Ensure directories exist
FIXTURES_DIR.mkdir(exist_ok=True)
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class ExpectedTocEntry:
    """Expected TOC entry for test validation."""

    title_contains: str  # Substring that should appear in the title
    page: int  # Expected PDF page number
    level: int  # Expected hierarchy level


@dataclass
class PdfTestCase:
    """Test case definition for a PDF file."""

    name: str
    pdf_path: Path
    expected_entries: list[ExpectedTocEntry]
    min_total_entries: int  # Minimum number of TOC entries expected
    has_text: bool = True  # Whether PDF already has OCR text


# Registry of test PDFs
TEST_PDFS: dict[str, PdfTestCase] = {}


def register_test_pdf(test_case: PdfTestCase) -> None:
    """Register a test PDF case."""
    TEST_PDFS[test_case.name] = test_case


def get_test_pdf(name: str) -> PdfTestCase | None:
    """Get a test PDF case by name."""
    return TEST_PDFS.get(name)


# Register the modal logic textbook
register_test_pdf(
    PdfTestCase(
        name="modal_logic_textbook",
        pdf_path=INPUT_DIR / "advanced_logic_course_book.pdf",
        min_total_entries=30,
        has_text=True,
        expected_entries=[
            # Parts (level 1)
            ExpectedTocEntry(title_contains="Part I", page=16, level=1),
            ExpectedTocEntry(title_contains="Part II", page=80, level=1),
            ExpectedTocEntry(title_contains="Part III", page=136, level=1),
            ExpectedTocEntry(title_contains="Part IV", page=264, level=1),
            # Sample chapters (level 2)
            ExpectedTocEntry(
                title_contains="Basic language and semantics", page=20, level=2
            ),
            ExpectedTocEntry(
                title_contains="Validity and decidability", page=46, level=2
            ),
            ExpectedTocEntry(
                title_contains="Computation and complexity", page=70, level=2
            ),
            # Back matter
            ExpectedTocEntry(title_contains="Index", page=388, level=2),
        ],
    )
)


@pytest.fixture
def modal_logic_pdf() -> PdfTestCase:
    """Fixture for the modal logic textbook PDF."""
    test_case = get_test_pdf("modal_logic_textbook")
    assert test_case is not None
    assert test_case.pdf_path.exists(), f"Test PDF not found: {test_case.pdf_path}"
    return test_case


@pytest.fixture
def temp_output_pdf(tmp_path: Path) -> Path:
    """Fixture providing a temporary output PDF path (cleaned up after test)."""
    return tmp_path / "output.pdf"


@pytest.fixture
def persistent_output_pdf(modal_logic_pdf: PdfTestCase) -> Path:
    """Fixture providing a persistent output PDF path in fixtures/output/."""
    output_path = OUTPUT_DIR / f"{modal_logic_pdf.name}_output.pdf"
    return output_path


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip tests if required PDF files are missing."""
    for item in items:
        # Check if test uses a PDF fixture
        for marker in item.iter_markers(name="requires_pdf"):
            pdf_name = marker.args[0] if marker.args else None
            if pdf_name:
                test_case = get_test_pdf(pdf_name)
                if test_case is None or not test_case.pdf_path.exists():
                    item.add_marker(
                        pytest.mark.skip(reason=f"Test PDF '{pdf_name}' not available")
                    )

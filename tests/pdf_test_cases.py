"""Test case definitions and registry for PDF testing."""

from dataclasses import dataclass, field
from pathlib import Path

# Root directory for test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"
INPUT_DIR = FIXTURES_DIR / "input"
OUTPUT_DIR = FIXTURES_DIR / "output"


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
    min_total_entries: int  # Minimum number of TOC entries expected
    has_text: bool = True  # Whether PDF already has OCR text
    expected_entries: list[ExpectedTocEntry] = field(default_factory=list)
    skip_content_check: bool = False  # Skip keyword-on-page verification
    has_existing_bookmarks: bool = False  # PDF has existing bookmarks
    expected_existing_bookmark_issues: bool = False  # Existing bookmarks are incorrect


# ============================================================================
# Test PDF Registry
# ============================================================================

TEST_PDFS: dict[str, PdfTestCase] = {}


def register_test_pdf(test_case: PdfTestCase) -> None:
    """Register a test PDF case."""
    TEST_PDFS[test_case.name] = test_case


def get_test_pdf(name: str) -> PdfTestCase | None:
    """Get a test PDF case by name."""
    return TEST_PDFS.get(name)


def get_all_test_pdfs() -> list[PdfTestCase]:
    """Get all registered test PDFs that exist on disk."""
    return [tc for tc in TEST_PDFS.values() if tc.pdf_path.exists()]


def get_test_pdfs_with_text() -> list[PdfTestCase]:
    """Get test PDFs that already have OCR text (no Tesseract needed)."""
    return [tc for tc in get_all_test_pdfs() if tc.has_text]


def get_test_pdfs_needing_ocr() -> list[PdfTestCase]:
    """Get test PDFs that need OCR processing."""
    return [tc for tc in get_all_test_pdfs() if not tc.has_text]


# ============================================================================
# PDF Test Case Definitions - Add new PDFs here
# ============================================================================

register_test_pdf(
    PdfTestCase(
        name="modal_logic_textbook",
        pdf_path=INPUT_DIR / "advanced_logic_course_book.pdf",
        min_total_entries=30,
        has_text=True,
        expected_entries=[
            ExpectedTocEntry(title_contains="Part I", page=16, level=1),
            ExpectedTocEntry(title_contains="Part II", page=80, level=1),
            ExpectedTocEntry(title_contains="Part III", page=136, level=1),
            ExpectedTocEntry(title_contains="Part IV", page=264, level=1),
            ExpectedTocEntry(
                title_contains="Basic language and semantics", page=20, level=2
            ),
            ExpectedTocEntry(
                title_contains="Validity and decidability", page=46, level=2
            ),
            ExpectedTocEntry(
                title_contains="Computation and complexity", page=70, level=2
            ),
            ExpectedTocEntry(title_contains="Index", page=388, level=2),
        ],
    )
)

register_test_pdf(
    PdfTestCase(
        name="applicative_bidirectional",
        pdf_path=INPUT_DIR / "Applicative_Bidirectional_Programming.pdf",
        min_total_entries=5,  # NLP heuristics find main sections only
        has_text=True,
        skip_content_check=True,  # Section headers mode doesn't need TOC page check
        expected_entries=[
            # Note: levels are normalized, first entry becomes level 1
            ExpectedTocEntry(title_contains="Introduction", page=1, level=1),
            ExpectedTocEntry(title_contains="Extensions", page=27, level=1),
        ],
    )
)

register_test_pdf(
    PdfTestCase(
        name="integrating_nominal_structural",
        pdf_path=INPUT_DIR / "integrating_nominal_structural.pdf",
        min_total_entries=5,
        has_text=True,
        has_existing_bookmarks=True,  # Has incorrect bookmarks that should be fixed
        expected_existing_bookmark_issues=True,
        skip_content_check=True,
        expected_entries=[
            # Note: levels are normalized, first entry becomes level 1
            ExpectedTocEntry(title_contains="Introduction", page=1, level=1),
            ExpectedTocEntry(title_contains="Motivating Examples", page=3, level=1),
        ],
    )
)

register_test_pdf(
    PdfTestCase(
        name="partial_computation_futamura",
        pdf_path=INPUT_DIR / "Partial_Computation_of_Programs_Futamura.pdf",
        min_total_entries=5,
        has_text=True,
        skip_content_check=True,
        expected_entries=[
            # Note: levels are normalized, first entry becomes level 1
            ExpectedTocEntry(title_contains="Partial Computation", page=2, level=1),
            ExpectedTocEntry(title_contains="Applications", page=6, level=1),
        ],
    )
)

register_test_pdf(
    PdfTestCase(
        name="concept_of_supercompiler",
        pdf_path=INPUT_DIR / "The_Concept_of_a_Supercompiler.pdf",
        min_total_entries=4,  # NLP heuristics find 4 main sections
        has_text=True,
        skip_content_check=True,
        expected_entries=[
            # Note: levels are normalized, first entry becomes level 1
            ExpectedTocEntry(title_contains="HISTORICAL", page=5, level=1),
            ExpectedTocEntry(title_contains="BASIC DEFINITIONS", page=7, level=1),
        ],
    )
)

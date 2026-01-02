"""Test fixtures and configuration for pdftoc tests."""

import shutil
import sys
from pathlib import Path

# Add tests directory to path so pdf_test_cases can be imported
sys.path.insert(0, str(Path(__file__).parent))

import pytest

from pdf_test_cases import (
    OUTPUT_DIR,
    PdfTestCase,
    get_test_pdf,
    get_test_pdfs_with_text,
)


# ============================================================================
# Utility Functions
# ============================================================================


def is_tesseract_available() -> bool:
    """Check if Tesseract OCR is available on the system."""
    return shutil.which("tesseract") is not None


def _pdf_id(test_case: PdfTestCase) -> str:
    """Generate a test ID from a PdfTestCase."""
    return test_case.name


# ============================================================================
# Pytest Fixtures
# ============================================================================


@pytest.fixture(params=get_test_pdfs_with_text(), ids=_pdf_id)
def pdf_with_text(request: pytest.FixtureRequest) -> PdfTestCase:
    """Parametrized fixture for all PDFs that already have OCR text."""
    return request.param  # type: ignore


@pytest.fixture
def modal_logic_pdf() -> PdfTestCase:
    """Fixture for the modal logic textbook PDF (backward compatibility)."""
    test_case = get_test_pdf("modal_logic_textbook")
    assert test_case is not None
    if not test_case.pdf_path.exists():
        pytest.skip(f"Test PDF not found: {test_case.pdf_path}")
    return test_case


@pytest.fixture
def temp_output_pdf(tmp_path: Path) -> Path:
    """Fixture providing a temporary output PDF path (cleaned up after test)."""
    return tmp_path / "output.pdf"


@pytest.fixture
def persistent_output_pdf(request: pytest.FixtureRequest) -> Path:
    """Fixture providing a persistent output PDF path in fixtures/output/."""
    # Get the test case from the test's parameters if available
    if hasattr(request, "param") and isinstance(request.param, PdfTestCase):
        test_case = request.param
    else:
        # Fallback to modal_logic for backward compat
        test_case = get_test_pdf("modal_logic_textbook")
        assert test_case is not None
    return OUTPUT_DIR / f"{test_case.name}_output.pdf"


# ============================================================================
# Pytest Hooks
# ============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_ocr: mark test as requiring Tesseract OCR"
    )
    config.addinivalue_line(
        "markers", "requires_pdf(name): mark test as requiring a specific PDF"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip tests based on available resources."""
    skip_ocr = pytest.mark.skip(reason="Tesseract OCR not available")

    for item in items:
        # Skip OCR tests if Tesseract isn't available
        if "requires_ocr" in [m.name for m in item.iter_markers()]:
            if not is_tesseract_available():
                item.add_marker(skip_ocr)

        # Skip tests for missing PDFs
        for marker in item.iter_markers(name="requires_pdf"):
            pdf_name = marker.args[0] if marker.args else None
            if pdf_name:
                test_case = get_test_pdf(pdf_name)
                if test_case is None or not test_case.pdf_path.exists():
                    item.add_marker(
                        pytest.mark.skip(reason=f"Test PDF '{pdf_name}' not available")
                    )

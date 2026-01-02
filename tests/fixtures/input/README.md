# Test PDF Input Files

Place test PDF files here. Each PDF should have a corresponding test case registered in `conftest.py`.

## Adding a new test PDF

1. Place the PDF file in this directory
2. Add a `PdfTestCase` in `conftest.py`:

```python
register_test_pdf(
    PdfTestCase(
        name="my_test_pdf",
        pdf_path=INPUT_DIR / "my_test_pdf.pdf",
        min_total_entries=10,
        has_text=True,  # Set to False if PDF needs OCR
        expected_entries=[
            ExpectedTocEntry(title_contains="Chapter 1", page=5, level=1),
            # Add more expected entries...
        ],
    )
)
```

3. Create a fixture in `conftest.py`:

```python
@pytest.fixture
def my_test_pdf() -> PdfTestCase:
    test_case = get_test_pdf("my_test_pdf")
    assert test_case is not None
    if not test_case.pdf_path.exists():
        pytest.skip(f"Test PDF not found: {test_case.pdf_path}")
    return test_case
```

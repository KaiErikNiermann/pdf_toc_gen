"""Typed access to PyMuPDF page text.

`Page.get_text()` is overloaded: its stubs return `str | list | dict` because
the return type depends on the `option` argument. Called bare it is always a
string, but every call site otherwise has to be narrowed by hand. This wrapper
does it in one place.
"""

import fitz  # type: ignore

__all__ = ["page_text"]


def page_text(page: fitz.Page) -> str:
    """The plain text of a page."""
    return str(page.get_text())

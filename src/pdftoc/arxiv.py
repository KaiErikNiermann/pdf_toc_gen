"""arXiv source download functionality."""

import re
import tarfile
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import fitz  # type: ignore


def extract_arxiv_id(pdf_path: Path) -> str | None:
    """
    Extract arXiv ID from a PDF.

    Looks for patterns like:
    - arXiv:2307.01234
    - arXiv:2307.01234v1
    - arxiv.org/abs/2307.01234
    - arxiv.org/pdf/2307.01234
    """
    doc: fitz.Document = fitz.open(pdf_path)
    try:
        # Check first few pages (arXiv ID usually on first page)
        pages_to_check = min(3, len(doc))
        full_text = ""
        for i in range(pages_to_check):
            page: fitz.Page = doc[i]
            full_text += page.get_text() + "\n"

        # Patterns to match arXiv IDs
        patterns = [
            # New format: YYMM.NNNNN (with optional version)
            r"arXiv[:\s]+(\d{4}\.\d{4,5}(?:v\d+)?)",
            r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)",
            # Old format: category/YYMMNNN
            r"arXiv[:\s]+([a-z\-]+/\d{7}(?:v\d+)?)",
            r"arxiv\.org/(?:abs|pdf)/([a-z\-]+/\d{7}(?:v\d+)?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                arxiv_id = match.group(1)
                # Strip version suffix for source URL
                arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
                return arxiv_id

        return None
    finally:
        doc.close()


def check_source_available(arxiv_id: str) -> bool:
    """Check if arXiv source is available for the given ID."""
    url = f"https://arxiv.org/src/{arxiv_id}"
    try:
        req = Request(url, method="HEAD")
        req.add_header("User-Agent", "pdftoc/1.0")
        with urlopen(req, timeout=10) as response:
            return response.status == 200
    except (HTTPError, Exception):
        return False


def download_arxiv_source(
    arxiv_id: str, output_dir: Path, verbose: bool
) -> Path | None:
    """
    Download and extract arXiv source files.

    Returns the path to the extracted directory, or None on failure.
    """
    url = f"https://arxiv.org/src/{arxiv_id}"

    if verbose:
        print(f"Downloading from: {url}")

    try:
        req = Request(url)
        req.add_header("User-Agent", "pdftoc/1.0")
        with urlopen(req, timeout=60) as response:
            content = response.read()
            content_type = response.headers.get("Content-Type", "")

        # Create output directory
        extract_dir = output_dir / f"arxiv-{arxiv_id.replace('/', '-')}"
        extract_dir.mkdir(parents=True, exist_ok=True)

        # Determine archive type and extract
        if "gzip" in content_type or "tar" in content_type:
            # Most common: .tar.gz
            with tarfile.open(fileobj=BytesIO(content), mode="r:*") as tar:
                tar.extractall(extract_dir)
            if verbose:
                print(f"Extracted tar archive to: {extract_dir}")
        elif "zip" in content_type:
            with zipfile.ZipFile(BytesIO(content)) as zf:
                zf.extractall(extract_dir)
            if verbose:
                print(f"Extracted zip archive to: {extract_dir}")
        else:
            # Try tar.gz first (most common), then other formats
            try:
                with tarfile.open(fileobj=BytesIO(content), mode="r:*") as tar:
                    tar.extractall(extract_dir)
                if verbose:
                    print(f"Extracted archive to: {extract_dir}")
            except tarfile.TarError:
                # Maybe it's a single file (e.g., .tex)
                # Save as-is
                single_file = extract_dir / "source.tex"
                single_file.write_bytes(content)
                if verbose:
                    print(f"Saved single file to: {single_file}")

        return extract_dir

    except HTTPError as e:
        if e.code == 404:
            return None
        raise
    except Exception as e:
        if verbose:
            print(f"Download error: {e}")
        return None


def get_arxiv_source(pdf_path: Path, output_dir: Path | None, verbose: bool) -> None:
    """
    Main function to get arXiv source for a PDF.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save source (defaults to PDF's directory)
        verbose: Show verbose output
    """
    print(f"Searching for arXiv ID in: {pdf_path.name}")

    # Extract arXiv ID
    arxiv_id = extract_arxiv_id(pdf_path)

    if not arxiv_id:
        print("✗ arXiv ID not found - paper may not be from arXiv")
        raise SystemExit(1)

    print(f"Found arXiv ID: {arxiv_id}")

    # Check if source is available
    print("Checking source availability...")
    if not check_source_available(arxiv_id):
        print("✗ Source not available - author didn't upload source files")
        raise SystemExit(1)

    # Download source
    if output_dir is None:
        output_dir = pdf_path.parent

    print("Downloading source...")
    result = download_arxiv_source(arxiv_id, output_dir, verbose)

    if result:
        print(f"✓ Source downloaded to: {result}")

        # List main files
        tex_files = list(result.glob("**/*.tex"))
        if tex_files:
            print(f"  Found {len(tex_files)} .tex file(s):")
            for tf in tex_files[:5]:
                print(f"    - {tf.relative_to(result)}")
            if len(tex_files) > 5:
                print(f"    ... and {len(tex_files) - 5} more")
    else:
        print("✗ Failed to download source")
        raise SystemExit(1)

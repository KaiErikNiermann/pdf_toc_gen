"""Tests for `--in-place` and the atomic staging it is built on."""

import os
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from pdftoc.cli import app
from pdftoc.core import atomic_output

runner = CliRunner()

ORIGINAL = b"%PDF-1.4 original\n"
REWRITTEN = b"%PDF-1.4 rewritten\n"


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    """A stand-in PDF. These tests never parse it, only track its bytes."""
    path = tmp_path / "book.pdf"
    path.write_bytes(ORIGINAL)
    return path


def _stray_files(pdf: Path) -> list[Path]:
    """Everything in the PDF's directory that is not the PDF itself."""
    return [p for p in pdf.parent.iterdir() if p != pdf]


# ============================================================================
# atomic_output
# ============================================================================


def test_atomic_output_replaces_target_on_clean_exit(pdf: Path) -> None:
    with atomic_output(pdf) as staged:
        assert staged != pdf, "must stage elsewhere, not hand back the target"
        staged.write_bytes(REWRITTEN)
        assert pdf.read_bytes() == ORIGINAL, "target must be untouched until exit"

    assert pdf.read_bytes() == REWRITTEN
    assert _stray_files(pdf) == []


def test_atomic_output_leaves_target_intact_on_failure(pdf: Path) -> None:
    """A crash mid-write must not truncate or clobber the original."""
    with pytest.raises(RuntimeError, match="boom"), atomic_output(pdf) as staged:
        staged.write_bytes(b"half-written gar")
        raise RuntimeError("boom")

    assert pdf.read_bytes() == ORIGINAL
    assert _stray_files(pdf) == [], "staging file must be cleaned up"


def test_atomic_output_stages_on_the_same_filesystem(pdf: Path) -> None:
    """os.replace is only atomic within one filesystem, so staging must be a sibling."""
    with atomic_output(pdf) as staged:
        assert staged.parent == pdf.parent


# ============================================================================
# CLI wiring
# ============================================================================


@pytest.fixture
def fake_process_pdf(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace process_pdf with a stub that records its kwargs and writes output."""
    calls: list[dict[str, Any]] = []

    def _fake(**kwargs: Any) -> None:
        calls.append(kwargs)
        output: Path = kwargs["output"]
        output.write_bytes(REWRITTEN)

    monkeypatch.setattr("pdftoc.cli.process_pdf", _fake)
    return calls


def test_in_place_rewrites_the_source(
    pdf: Path, fake_process_pdf: list[dict[str, Any]]
) -> None:
    result = runner.invoke(app, ["-i", "-f", str(pdf)])

    assert result.exit_code == 0, result.output
    assert pdf.read_bytes() == REWRITTEN
    assert _stray_files(pdf) == []


def test_in_place_accepts_the_combined_short_form(
    pdf: Path, fake_process_pdf: list[dict[str, Any]]
) -> None:
    """`-if FILE` is the documented shorthand; `-fi FILE` cannot work (see below)."""
    result = runner.invoke(app, ["-if", str(pdf)])

    assert result.exit_code == 0, result.output
    assert pdf.read_bytes() == REWRITTEN


def test_combined_short_form_is_order_sensitive(pdf: Path) -> None:
    """`-fi` makes "i" the value of -f, so it must fail rather than silently misparse."""
    result = runner.invoke(app, ["-fi", str(pdf)])

    assert result.exit_code != 0
    assert pdf.read_bytes() == ORIGINAL


def test_in_place_hands_process_pdf_a_staging_path(
    pdf: Path, fake_process_pdf: list[dict[str, Any]]
) -> None:
    """process_pdf reads `source` throughout, so it must never be told to write to it."""
    runner.invoke(app, ["-i", "-f", str(pdf)])

    (call,) = fake_process_pdf
    assert call["source"] == pdf
    assert call["output"] != pdf
    assert call["output_label"] == pdf, "progress messages should name the real file"


def test_output_mode_writes_to_the_given_path(
    pdf: Path, tmp_path: Path, fake_process_pdf: list[dict[str, Any]]
) -> None:
    dest = tmp_path / "out.pdf"
    result = runner.invoke(app, ["-f", str(pdf), "-t", str(dest)])

    assert result.exit_code == 0, result.output
    assert dest.read_bytes() == REWRITTEN
    assert pdf.read_bytes() == ORIGINAL, "source must be left alone without --in-place"

    (call,) = fake_process_pdf
    assert call["output"] == dest
    assert call["output_label"] is None


def test_in_place_and_to_are_mutually_exclusive(
    pdf: Path, tmp_path: Path, fake_process_pdf: list[dict[str, Any]]
) -> None:
    result = runner.invoke(app, ["-i", "-f", str(pdf), "-t", str(tmp_path / "out.pdf")])

    assert result.exit_code == 1
    assert "mutually exclusive" in result.output
    assert fake_process_pdf == []
    assert pdf.read_bytes() == ORIGINAL


def test_a_destination_is_still_required(
    pdf: Path, fake_process_pdf: list[dict[str, Any]]
) -> None:
    result = runner.invoke(app, ["-f", str(pdf)])

    assert result.exit_code == 1
    assert "--in-place" in result.output, "the error should point at the new option"
    assert fake_process_pdf == []


@pytest.mark.skipif(
    os.geteuid() == 0, reason="root bypasses the write bit, so the guard cannot trip"
)
def test_in_place_refuses_a_read_only_source(
    pdf: Path, fake_process_pdf: list[dict[str, Any]]
) -> None:
    """Fail before any work rather than after a long OCR run."""
    pdf.chmod(0o444)
    try:
        result = runner.invoke(app, ["-i", "-f", str(pdf)])
    finally:
        pdf.chmod(0o644)

    assert result.exit_code == 1
    assert "write permission" in result.output
    assert fake_process_pdf == []
    assert pdf.read_bytes() == ORIGINAL

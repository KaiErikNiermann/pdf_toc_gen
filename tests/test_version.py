"""Tests for `--version` and the provenance it reports."""

from datetime import UTC, datetime

import pytest
from typer.testing import CliRunner

from pdftoc.cli import app
from pdftoc.version import BuildInfo, collect_build_info, format_build_info

runner = CliRunner()

STAMP = datetime(2026, 8, 7, 18, 5, 12, tzinfo=UTC)


def test_git_provenance_reports_the_revision() -> None:
    out = format_build_info(
        BuildInfo(version="0.1.0", source="git", revision="a4d88bf", timestamp=STAMP)
    )

    assert "pdftoc 0.1.0" in out
    assert "a4d88bf" in out
    # A commit date is not a build date; the label has to say which it is.
    assert "committed" in out
    assert "installed" not in out


def test_dirty_checkout_is_flagged() -> None:
    """A version string pasted into a bug report must not imply a clean tree."""
    out = format_build_info(
        BuildInfo(
            version="0.1.0",
            source="git",
            revision="a4d88bf",
            dirty=True,
            timestamp=STAMP,
        )
    )

    assert "a4d88bf (dirty)" in out


def test_installed_provenance_claims_no_revision() -> None:
    """An installed copy knows when it was written, not what it was built from."""
    out = format_build_info(
        BuildInfo(version="0.1.0", source="install", timestamp=STAMP)
    )

    assert "installed" in out
    assert "revision" not in out
    assert "committed" not in out


def test_unknown_provenance_still_renders() -> None:
    out = format_build_info(BuildInfo(version="unknown", source="unknown"))

    assert "pdftoc unknown" in out
    assert "unknown" in out


def test_collect_build_info_works_in_this_tree() -> None:
    info = collect_build_info()

    assert info.version
    assert info.source in {"git", "install", "unknown"}


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_version_flag_prints_and_exits(flag: str) -> None:
    """Eager, so it must succeed despite --from being a required option."""
    result = runner.invoke(app, [flag])

    assert result.exit_code == 0, result.output
    assert "pdftoc" in result.output


def test_short_verbose_flag_is_not_version() -> None:
    """-v stays --verbose; only -V is version."""
    result = runner.invoke(app, ["-v"])

    assert result.exit_code != 0, "should still demand --from"
    assert "--from" in result.output

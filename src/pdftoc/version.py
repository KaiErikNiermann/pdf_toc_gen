"""Version and build provenance behind `pdftoc --version`.

Nothing stamps a build timestamp into the package -- Poetry has no build hook
here, and `pipx install .` copies sources verbatim -- so provenance is
recovered at runtime instead. A working checkout reports the commit it is on; an
installed copy reports when its files were written. The two are labelled
differently because they mean different things: a commit date is not a build
date, and conflating them would be a lie the user cannot detect.
"""

import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Literal

DISTRIBUTION = "pdf-toc-gen"

_PACKAGE_ROOT = Path(__file__).resolve().parent
_GIT_TIMEOUT_S = 2.0

# Where the reported timestamp came from, which decides how it is labelled.
BuildSource = Literal["git", "install", "unknown"]


@dataclass(frozen=True)
class BuildInfo:
    """Everything `--version` reports."""

    version: str
    source: BuildSource
    revision: str | None = None
    dirty: bool = False
    timestamp: datetime | None = None


def _distribution_version() -> str:
    try:
        return metadata.version(DISTRIBUTION)
    except metadata.PackageNotFoundError:
        # Running from a source tree that was never installed.
        return "unknown"


def _git(*args: str) -> str | None:
    """Run git against the package directory; None if it fails or is absent."""
    try:
        result = subprocess.run(
            ["git", "-C", str(_PACKAGE_ROOT), *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _git_build_info(version: str) -> BuildInfo | None:
    """Provenance from the checkout, or None if this is not one.

    Bails out inside site-packages: an installed copy can sit under an unrelated
    repository, and reporting that repository's HEAD would attribute a revision
    the code was never built from.
    """
    if "site-packages" in _PACKAGE_ROOT.parts:
        return None
    if _git("rev-parse", "--is-inside-work-tree") != "true":
        return None

    revision = _git("rev-parse", "--short", "HEAD")
    if revision is None:
        return None

    committed = _git("log", "-1", "--format=%cI")
    return BuildInfo(
        version=version,
        source="git",
        revision=revision,
        # Any tracked modification means the running code no longer matches the
        # revision named above, which matters when a version string is pasted
        # into a bug report.
        dirty=bool(_git("status", "--porcelain")),
        timestamp=datetime.fromisoformat(committed) if committed else None,
    )


def _installed_build_info(version: str) -> BuildInfo:
    """Fall back to when the package files were written."""
    try:
        written = (_PACKAGE_ROOT / "__init__.py").stat().st_mtime
    except OSError:
        return BuildInfo(version=version, source="unknown")
    return BuildInfo(
        version=version,
        source="install",
        timestamp=datetime.fromtimestamp(written, UTC).astimezone(),
    )


def collect_build_info() -> BuildInfo:
    """Best available provenance for the running code."""
    version = _distribution_version()
    return _git_build_info(version) or _installed_build_info(version)


def format_build_info(info: BuildInfo) -> str:
    """Render `--version` output."""
    revision = info.revision or "unknown"
    if info.dirty:
        revision += " (dirty)"

    # The label has to track the source: a commit date is not a build date.
    stamp_label = {"git": "committed", "install": "installed", "unknown": "built"}[
        info.source
    ]
    stamp = (
        info.timestamp.strftime("%Y-%m-%d %H:%M:%S %z")
        if info.timestamp is not None
        else "unknown"
    )

    lines = [f"pdftoc {info.version}"]
    if info.source == "git":
        lines.append(f"  revision   {revision}")
    lines.append(f"  {stamp_label:<10} {stamp}")
    lines.append(f"  python     {sys.version.split()[0]}")
    return "\n".join(lines)

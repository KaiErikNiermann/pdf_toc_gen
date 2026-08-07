#!/usr/bin/env bash
#
# Install pdftoc as a global CLI tool.
#
# Prefers pipx, which is the right tool for a CLI app: isolated venv, entry
# point on PATH, upgradable in place. Falls back to a self-managed venv plus a
# symlink, which still works on PEP 668 "externally managed" distributions
# (Arch, Debian 12+, Fedora 38+) where `pip install --user` is refused outright.
#
# Overrides:
#   PDFTOC_PYTHON   interpreter to install under (default: newest suitable)
#   PDFTOC_PREFIX   venv location for the fallback install
set -euo pipefail

# Resolve the repo from the script's own location. Installing `.` would
# otherwise mean the caller's working directory, so `bash /path/to/install.sh`
# from anywhere but the repo root tried to install whatever happened to be
# sitting in $PWD.
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO_ROOT
readonly PYPROJECT="$REPO_ROOT/pyproject.toml"
readonly BIN_DIR="$HOME/.local/bin"

# Set by whichever install path runs; verified at the end. Initialised so `set
# -u` reports a missing assignment rather than an unbound-variable trace.
INSTALLED_BIN=""
PYTHON=""

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

# True when $1 >= $2, comparing dotted versions numerically. `sort -V` keeps
# this correct across a major bump, which a naive per-component test does not:
# 4.0 must count as newer than 3.13, not older.
version_ge() {
    [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" = "$2" ]
}

# Every python3.N on PATH, plus the unversioned names, deduplicated.
discover_pythons() {
    local -a dirs
    IFS=: read -ra dirs <<<"$PATH"
    {
        local dir entry
        for dir in "${dirs[@]}"; do
            [ -n "$dir" ] || continue
            for entry in "$dir"/python3.[0-9] "$dir"/python3.[0-9][0-9]; do
                [ -x "$entry" ] && basename -- "$entry"
            done
        done
        printf 'python3\npython\n'
    } 2>/dev/null | sort -u
}

interpreter_version() {
    "$1" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null
}

# Newest interpreter satisfying requires-python, so the floor never drifts from
# pyproject.toml the way a hardcoded one did.
select_python() {
    local min="$1" candidate version best="" best_version=""

    if [ -n "${PDFTOC_PYTHON:-}" ]; then
        command -v "$PDFTOC_PYTHON" >/dev/null 2>&1 ||
            die "PDFTOC_PYTHON=$PDFTOC_PYTHON is not executable"
        version="$(interpreter_version "$PDFTOC_PYTHON")" ||
            die "PDFTOC_PYTHON=$PDFTOC_PYTHON does not look like a Python interpreter"
        version_ge "$version" "$min" ||
            die "PDFTOC_PYTHON=$PDFTOC_PYTHON is $version; $min or newer is required"
        printf '%s\n' "$PDFTOC_PYTHON"
        return 0
    fi

    while read -r candidate; do
        command -v "$candidate" >/dev/null 2>&1 || continue
        version="$(interpreter_version "$candidate")" || continue
        [ -n "$version" ] || continue
        version_ge "$version" "$min" || continue
        if [ -z "$best" ] || version_ge "$version" "$best_version"; then
            best="$candidate"
            best_version="$version"
        fi
    done < <(discover_pythons)

    [ -n "$best" ] || return 1
    printf '%s\n' "$best"
}

# PEP 668 marks an interpreter as owned by the system package manager. pip
# refuses to write into it, so detect that rather than letting the install fail
# several confusing lines later.
pip_is_usable() {
    local stdlib
    "$PYTHON" -m pip --version >/dev/null 2>&1 || return 1
    stdlib="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["stdlib"])')" || return 1
    [ ! -e "$stdlib/EXTERNALLY-MANAGED" ]
}

install_with_pipx() {
    echo "Installing with pipx..."
    pipx install "$REPO_ROOT" --force --python "$PYTHON"
    INSTALLED_BIN="${PIPX_BIN_DIR:-$HOME/.local/bin}/pdftoc"
}

install_with_pip() {
    echo "Installing with pip (--user)..."
    "$PYTHON" -m pip install --user "$REPO_ROOT"
    INSTALLED_BIN="$("$PYTHON" -m site --user-base)/bin/pdftoc"
}

# Works anywhere python3 does: venv bundles pip via ensurepip even when the
# system has no pip at all, and writing to our own prefix sidesteps PEP 668.
install_with_venv() {
    local prefix="${PDFTOC_PREFIX:-${XDG_DATA_HOME:-$HOME/.local/share}/pdftoc}"
    echo "pipx not found and system pip is unusable; installing into $prefix..."
    "$PYTHON" -m venv --clear "$prefix"
    "$prefix/bin/python" -m pip install --quiet --upgrade pip
    "$prefix/bin/python" -m pip install "$REPO_ROOT"
    mkdir -p "$BIN_DIR"
    ln -sf "$prefix/bin/pdftoc" "$BIN_DIR/pdftoc"
    echo "Linked $BIN_DIR/pdftoc -> $prefix/bin/pdftoc"
    INSTALLED_BIN="$BIN_DIR/pdftoc"
}

warn_if_not_on_path() {
    case ":$PATH:" in
        *":$BIN_DIR:"*) ;;
        *)
            printf '\nNote: %s is not on your PATH. Add it with:\n' "$BIN_DIR"
            printf '  export PATH="%s:$PATH"\n' "$BIN_DIR"
            ;;
    esac
}

main() {
    [ -f "$PYPROJECT" ] || die "no pyproject.toml at $PYPROJECT (is the repo intact?)"

    local min_python
    min_python="$(sed -n 's/^requires-python *= *">=\([0-9]\+\.[0-9]\+\).*/\1/p' "$PYPROJECT")"
    [ -n "$min_python" ] || die "could not read requires-python from $PYPROJECT"

    echo "Installing pdftoc from $REPO_ROOT (needs Python >= $min_python)"

    PYTHON="$(select_python "$min_python")" ||
        die "no Python >= $min_python found. Install one, or set PDFTOC_PYTHON."

    echo "Using $PYTHON ($("$PYTHON" --version 2>&1))"

    if command -v pipx >/dev/null 2>&1; then
        install_with_pipx
    elif pip_is_usable; then
        install_with_pip
        warn_if_not_on_path
    else
        install_with_venv
        warn_if_not_on_path
    fi

    # Verify the binary just installed, not whatever `pdftoc` resolves to: a
    # stale copy earlier on PATH would otherwise be reported as the new install.
    echo
    [ -x "$INSTALLED_BIN" ] || die "install reported success but $INSTALLED_BIN is missing"
    echo "Installed:"
    "$INSTALLED_BIN" --version | sed 's/^/  /'

    local resolved
    resolved="$(command -v pdftoc 2>/dev/null || true)"
    if [ -z "$resolved" ]; then
        echo
        echo "Note: 'pdftoc' is not on your PATH yet. Open a new shell, or use"
        echo "  $INSTALLED_BIN"
    elif [ "$resolved" != "$INSTALLED_BIN" ]; then
        # A leftover from an earlier install would keep shadowing the new one.
        echo
        echo "Warning: 'pdftoc' on your PATH resolves to a different install:"
        echo "  $resolved"
        echo "Remove it, or it will keep shadowing $INSTALLED_BIN"
    fi

    echo
    echo "Run 'pdftoc --help' to get started."
}

main "$@"

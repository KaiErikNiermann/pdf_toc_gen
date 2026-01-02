#!/usr/bin/env bash
# Install pdftoc as a global CLI tool
set -e

echo "Installing pdftoc..."

# Find Python 3.13+
PYTHON=""
for py in python3.13 python3 python; do
    if command -v "$py" &> /dev/null; then
        version=$("$py" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 13 ]; then
            PYTHON="$py"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3.13+ is required but not found."
    echo "Install Python 3.13 or later and try again."
    exit 1
fi

echo "Using $PYTHON (version $($PYTHON --version))"

# Check if pipx is available (preferred for CLI tools)
if command -v pipx &> /dev/null; then
    echo "Using pipx for installation..."
    pipx install . --force --python "$PYTHON"
    echo ""
    echo "✓ Installed with pipx. Run 'pdftoc --help' to get started."
else
    echo "pipx not found, using pip..."
    
    # Try pip with --user flag for non-root installs
    if [ "$EUID" -ne 0 ]; then
        "$PYTHON" -m pip install --user .
        echo ""
        echo "✓ Installed with pip (user). Make sure ~/.local/bin is in your PATH."
    else
        "$PYTHON" -m pip install .
        echo ""
        echo "✓ Installed with pip (system-wide)."
    fi
    echo "Run 'pdftoc --help' to get started."
fi

#!/usr/bin/env bash
# Detached installer for the marker/Surya GPU OCR backend.
# Installs a Blackwell-capable torch (cu128) from PyTorch's CDN, then marker-pdf
# (which pulls surya-ocr, transformers, etc.). Logs progress; safe to re-run
# (pip skips already-satisfied packages and resumes partial downloads).
set -uo pipefail

VENV="$(poetry env info --path)"
PIP="$VENV/bin/python -m pip"
LOG="${1:-$REPO_ROOT/scripts/.marker_install.log}"

echo "=== [$(date '+%H:%M:%S')] step 1/2: torch+torchvision cu128 (Blackwell) ===" | tee -a "$LOG"
$PIP install --resume-retries 50 torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128 >>"$LOG" 2>&1
rc1=$?
echo "=== [$(date '+%H:%M:%S')] torch step exit=$rc1 ===" | tee -a "$LOG"

if [ $rc1 -ne 0 ]; then
    echo "torch install failed (rc=$rc1); stopping before marker-pdf." | tee -a "$LOG"
    exit $rc1
fi

echo "=== [$(date '+%H:%M:%S')] step 2/2: marker-pdf (+surya, transformers) ===" | tee -a "$LOG"
$PIP install --resume-retries 50 "marker-pdf>=1.10.0" >>"$LOG" 2>&1
rc2=$?
echo "=== [$(date '+%H:%M:%S')] marker-pdf step exit=$rc2 ===" | tee -a "$LOG"

echo "=== [$(date '+%H:%M:%S')] DONE (torch=$rc1 marker=$rc2) ===" | tee -a "$LOG"
exit $rc2

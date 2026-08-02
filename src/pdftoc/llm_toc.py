"""LLM-based TOC extraction using OpenAI API (default) or local Ollama."""

from __future__ import annotations

import json
import os
import re
from enum import StrEnum
from pathlib import Path
from typing import Any

import requests

from pdftoc.models import TocEntry
from pdftoc.page_labels import PageRef, parse_page_label

_SYSTEM_PROMPT = (
    "You extract table of contents entries from OCR text. "
    "Return ONLY a JSON array. Each entry: "
    '{"level":N,"title":"...","page":N_or_null}. '
    "level: 1=part, 2=chapter, 3=section (N.N), 4=subsection (N.N.N). "
    "Include number prefixes in titles (e.g. '1. Introduction', '3.2 Linear FM Signals'). "
    "Copy the page number EXACTLY as printed: roman numerals stay roman "
    'strings ("ix", "xxiii"), arabic stay integers (19). Front matter is '
    "numbered separately from the body, so converting roman to arabic would "
    "point the entry at the wrong page. "
    "Set page to null if no page number is visible for that entry. "
    "JSON array only, no markdown fences, no explanation."
)


class LlmBackend(StrEnum):
    AUTO = "auto"  # OpenAI if key available, else ollama
    OPENAI = "openai"
    OLLAMA = "ollama"


def _load_api_key() -> str | None:
    """Load OpenAI API key from environment or .env file."""
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    # Try .env in current dir or project root
    for env_path in [Path(".env"), Path(__file__).parent.parent.parent / ".env"]:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    return line.split("=", 1)[1].strip()
    return None


def is_ollama_available(model: str = "qwen3.5:9b") -> bool:
    """Check if Ollama is running and the model is available."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        if resp.status_code != 200:
            return False
        models = [m["name"] for m in resp.json().get("models", [])]
        return any(model in m for m in models)
    except (requests.ConnectionError, requests.Timeout):
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_toc_with_llm(
    toc_text: str,
    verbose: bool = False,
    backend: LlmBackend = LlmBackend.AUTO,
    ollama_model: str = "qwen3.5:9b",
    openai_model: str = "gpt-4.1-mini",
) -> list[TocEntry]:
    """Extract TOC entries from text using an LLM.

    For a single PDF the text is sent as parallel API calls (one per chunk).
    """
    chunks = _make_chunks(toc_text)
    if not chunks:
        return []

    # Resolve backend
    use_openai = False
    if backend == LlmBackend.OPENAI or backend == LlmBackend.AUTO:
        api_key = _load_api_key()
        if api_key:
            use_openai = True
        elif backend == LlmBackend.OPENAI:
            raise RuntimeError("OPENAI_API_KEY not found in environment or .env")

    if use_openai:
        if verbose:
            print(f"  Using OpenAI ({openai_model}) for TOC extraction...")
        raw = _extract_openai(chunks, openai_model, verbose)
    else:
        if verbose:
            print(f"  Using Ollama ({ollama_model}) for TOC extraction...")
        raw = _extract_ollama(chunks, ollama_model, verbose)

    return _deduplicate(raw, verbose)


def extract_toc_batch(
    toc_texts: dict[str, str],
    verbose: bool = False,
    openai_model: str = "gpt-4.1-mini",
) -> dict[str, list[TocEntry]]:
    """Extract TOC entries for multiple PDFs via OpenAI Batch API.

    Args:
        toc_texts: Mapping of pdf_id -> toc_text.
        verbose: Print progress.
        openai_model: Model to use.

    Returns:
        Mapping of pdf_id -> list of TocEntry.
    """
    from openai import OpenAI

    api_key = _load_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY required for batch processing")

    client = OpenAI(api_key=api_key)

    # Build JSONL for batch
    import tempfile
    import time

    jsonl_lines: list[str] = []
    # Track which request_id maps to which pdf_id + chunk index
    request_map: dict[str, tuple[str, int]] = {}

    for pdf_id, toc_text in toc_texts.items():
        chunks = _make_chunks(toc_text)
        for chunk_idx, chunk in enumerate(chunks):
            req_id = f"{pdf_id}__chunk{chunk_idx}"
            request_map[req_id] = (pdf_id, chunk_idx)
            jsonl_lines.append(
                json.dumps(
                    {
                        "custom_id": req_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": openai_model,
                            "messages": [
                                {"role": "system", "content": _SYSTEM_PROMPT},
                                {"role": "user", "content": chunk},
                            ],
                            "temperature": 0.0,
                            "max_tokens": 4000,
                        },
                    }
                )
            )

    if verbose:
        print(f"  Batch: {len(jsonl_lines)} requests for {len(toc_texts)} PDFs")

    # Upload JSONL
    jsonl_content = "\n".join(jsonl_lines)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        tmp.write(jsonl_content)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            batch_file = client.files.create(file=f, purpose="batch")

        # Create batch
        batch = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        if verbose:
            print(f"  Batch created: {batch.id}")

        # Poll for completion
        while batch.status in ("validating", "in_progress", "finalizing"):
            time.sleep(5)
            batch = client.batches.retrieve(batch.id)
            if verbose:
                completed = (
                    batch.request_counts.completed if batch.request_counts else 0
                )
                total = batch.request_counts.total if batch.request_counts else 0
                print(f"  Batch status: {batch.status} ({completed}/{total})")

        if batch.status != "completed" or not batch.output_file_id:
            if verbose:
                print(f"  Batch failed: {batch.status}")
            return {pdf_id: [] for pdf_id in toc_texts}

        # Download results
        output_content = client.files.content(batch.output_file_id).text
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # Parse results
    pdf_raw: dict[str, list[dict[str, Any]]] = {pid: [] for pid in toc_texts}
    for line in output_content.strip().split("\n"):
        if not line.strip():
            continue
        result = json.loads(line)
        req_id = result["custom_id"]
        pdf_id, _ = request_map[req_id]
        body = result.get("response", {}).get("body", {})
        content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        entries = _parse_json_array(content)
        pdf_raw[pdf_id].extend(entries)

    return {pid: _deduplicate(raw, verbose=False) for pid, raw in pdf_raw.items()}


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _make_chunks(toc_text: str) -> list[str]:
    """Split TOC text into chunks suitable for LLM calls."""
    if "---PAGE BREAK---" in toc_text:
        pages = [p.strip() for p in toc_text.split("---PAGE BREAK---") if p.strip()]
    else:
        pages = _split_text(toc_text, max_chars=3000)

    # Group into pairs to reduce number of API calls
    chunks: list[str] = []
    for i in range(0, len(pages), 2):
        chunk = "\n".join(pages[i : i + 2])
        if len(chunk.strip()) >= 30:
            chunks.append(chunk)
    return chunks


def _extract_openai(
    chunks: list[str],
    model: str,
    verbose: bool,
) -> list[dict[str, Any]]:
    """Send chunks to OpenAI API in parallel via threads."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from openai import OpenAI

    api_key = _load_api_key()
    client = OpenAI(api_key=api_key)
    all_entries: list[dict[str, Any]] = []

    def process_chunk(idx: int, chunk: str) -> list[dict[str, Any]]:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": chunk},
            ],
            temperature=0.0,
            max_tokens=4000,
        )
        content = resp.choices[0].message.content or ""
        return _parse_json_array(content)

    with ThreadPoolExecutor(max_workers=min(len(chunks), 8)) as pool:
        futures = {pool.submit(process_chunk, i, c): i for i, c in enumerate(chunks)}
        for future in as_completed(futures):
            chunk_idx = futures[future]
            try:
                entries = future.result()
                all_entries.extend(entries)
                if verbose:
                    print(
                        f"    chunk {chunk_idx + 1}/{len(chunks)}: {len(entries)} entries"
                    )
            except Exception as e:
                if verbose:
                    print(f"    chunk {chunk_idx + 1}/{len(chunks)}: error - {e}")

    return all_entries


def _extract_ollama(
    chunks: list[str],
    model: str,
    verbose: bool,
) -> list[dict[str, Any]]:
    """Send chunks to local Ollama sequentially."""
    all_entries: list[dict[str, Any]] = []

    for i, chunk in enumerate(chunks):
        if verbose:
            print(f"    chunk {i + 1}/{len(chunks)}...")
        try:
            resp = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": model,
                    "think": False,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": chunk},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 4000},
                },
                timeout=600,
            )
            resp.raise_for_status()
            content = resp.json().get("message", {}).get("content", "").strip()
            entries = _parse_json_array(content)
            all_entries.extend(entries)
            if verbose:
                print(f"      -> {len(entries)} entries")
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            if verbose:
                print(f"      -> error: {e}")

    return all_entries


def _deduplicate(raw_entries: list[dict[str, Any]], verbose: bool) -> list[TocEntry]:
    """Deduplicate and convert raw dicts to TocEntry list."""
    seen: set[tuple[str, int, PageRef]] = set()
    result: list[TocEntry] = []

    for entry in raw_entries:
        title = str(entry.get("title", "")).strip()
        if not title:
            continue

        # Keeps roman numerals distinct from arabic: "ix" is a front matter
        # page, not page 9 of the body.
        parsed = parse_page_label(entry.get("page"))
        if parsed is None:
            continue
        page, page_ref = parsed

        key = (title.lower(), page, page_ref)
        if key in seen:
            continue
        seen.add(key)
        result.append(
            TocEntry(
                level=max(1, min(4, int(entry.get("level", 2)))),
                title=title,
                page=page,
                page_ref=page_ref,
            )
        )

    if verbose:
        roman = sum(1 for e in result if e.page_ref == PageRef.PRINTED_ROMAN)
        suffix = f" ({roman} in roman-numbered front matter)" if roman else ""
        print(f"  LLM extracted {len(result)} entries with page numbers{suffix}")

    return result


def _parse_json_array(text: str) -> list[dict[str, Any]]:
    """Extract a JSON array from text, handling markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
    if text.endswith("```"):
        text = "\n".join(text.split("\n")[:-1])
    text = text.strip()

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    return []


def _split_text(text: str, max_chars: int = 3000) -> list[str]:
    """Split text into chunks at line boundaries."""
    lines = text.split("\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in lines:
        if current_len + len(line) > max_chars and current:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        current.append(line)
        current_len += len(line) + 1

    if current:
        chunks.append("\n".join(current))

    return chunks

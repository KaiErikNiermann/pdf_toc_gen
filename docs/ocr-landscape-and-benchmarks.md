# OCR landscape & local benchmarks (mid-2026)

Evaluation of modern OCR engines for `pdftoc`, focused on the real target:
OCR'ing **large scanned books on performance-constrained machines** (laptops,
CPU-only or modest GPU, fully local/offline). Pairs a survey of the current
landscape with **empirical benchmarks run on this repo's actual code paths**.

## TL;DR

- The classic **tesseract** pipeline (via `ocrmypdf`) is the fast, zero-setup
  floor: ~75 pages/min on CPU, but the lowest accuracy.
- **PaddleOCR PP-OCRv5** (classic detect+recognize) is the pragmatic upgrade —
  meaningfully more accurate, and **CPU-viable at ~23 pages/min once the paddle
  version footgun is fixed** (see below). This is the new `paddle` backend.
- The **VLM-OCR wave** (DeepSeek-OCR, Baidu Unlimited-OCR, dots.ocr, olmOCR,
  Surya/marker, …) is a genuine *quality* leap but is **GPU-bound**; on CPU it
  runs at single-digit pages/min. Promising only with ≥8–12 GB VRAM.
- **Baidu Unlimited-OCR specifically: not worth integrating** for this tool — a
  days-old, GPU-only 3B VLM whose headline feature targets GPU long-context
  servers, not chunked local OCR.

## Backends in `pdftoc`

| backend (`--ocr-backend`) | engine | type | hardware | role |
|---|---|---|---|---|
| `ocrmypdf` | tesseract 5 | classic pipeline | CPU | lightweight floor / fallback |
| `paddle` | PaddleOCR PP-OCRv5 | classic pipeline | CPU (GPU optional) | accurate middle tier |
| `marker` | Surya | transformer/VLM | GPU | highest quality, GPU-only |
| `auto` | marker if available, else ocrmypdf | | | default |

## Benchmark method

`scripts/ocr_benchmark.py` uses a **born-digital PDF's embedded text layer as
ground truth**: it rasterizes each page, OCRs it through the tool's real backend
code, and reports throughput plus a character-similarity proxy (difflib ratio,
whitespace-normalized) against that ground truth. The similarity is meaningful
for *relative* comparison between engines, not as an absolute accuracy figure
(the test doc is a dense math paper whose ligatures/symbols deflate all scores).

Test doc: `2511.08162v1.pdf` (dense academic paper, 200 DPI). Machine: 24-core
CPU, 30 GB RAM (CPU figures), RTX 5080 16 GB available.

## Results — CPU

| backend | pages/min | similarity | 400-page book |
|---|---|---|---|
| tesseract (`ocrmypdf`) | ~75 | 0.65 | ~5 min |
| **PaddleOCR PP-OCRv5 + mkldnn** | **~23** (steady ~30) | **0.75** | ~17 min |
| PaddleOCR PP-OCRv5, **mkldnn broken** | ~3 | 0.75 | ~2.5 h |

PaddleOCR is **~3× slower but clearly more accurate** than tesseract — the
expected "middle tier" tradeoff. On cleaner pages the accuracy gap is larger
(~0.85 vs 0.65). DPI 150 vs 200 barely changes throughput (bottleneck is model
inference, not raster size).

### ⚠️ The paddle version footgun (important)

PaddleOCR's CPU speed depends entirely on the **oneDNN/mkldnn** acceleration
path, and that path is **broken in paddlepaddle 3.2+** (the new-IR executor hits
`NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support
[pir::ArrayAttribute<pir::DoubleAttribute>]` at inference). The breakage is
version-specific, not model-specific (it hits both PP-OCRv5 and v6), and PaddleX
forces the PIR executor so the usual `FLAGS_enable_pir_api=0` workaround does not
help.

**Fix:** pin **paddlepaddle 3.1.x** and keep `enable_mkldnn=True` (the backend
default). This is a **6.3× swing** (2.9 → 18.4 pages/min post-warmup). paddleocr
3.7 also defaults to **PP-OCRv6** models which trip an even earlier paddle-3.0
codegen bug, so the backend pins **`ocr_version="PP-OCRv5"`** and the lighter
**`PP-OCRv5_mobile_det`** detector (the default `server_det` is slow/RAM-hungry on
CPU).

### Install (CPU)

```bash
pip install "paddleocr>=3.7,<4.0"
pip install "paddlepaddle==3.1.1" -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
# then: pdftoc --from book.pdf --to out.pdf --ocr-backend paddle
```

## GPU notes (RTX 5080 / Blackwell sm_120)

- **PaddleOCR GPU: blocked here by bandwidth, not capability.** paddle's GPU
  wheels are hosted only on Baidu's China CDN (`bcebos.com`), which delivered at
  ~32 KB/s — a ~2.5–3 GB wheel ⇒ ~20 h ETA. No fast PyPI mirror exists. If you
  have the bandwidth: `pip install paddlepaddle-gpu -i
  https://www.paddlepaddle.org.cn/packages/stable/cu129/` and run with
  `device="gpu"`; expect a large speedup that makes the accuracy win "free".
- **marker/Surya GPU** is the viable GPU path (torch comes from PyTorch's fast
  CDN, needs a cu128 Blackwell build). [Benchmark pending.]

## The modern landscape (survey)

The shift is from classic **detect→recognize** pipelines (tesseract, PaddleOCR)
to **vision-language models** that read a page image and emit structured
markdown/HTML (layout, reading order, tables, math) in one pass. Quality and
structure improve markedly; the cost is **GPU dependence and multi-GB weights**.

| model | size | VRAM | CPU-viable? | license | notes |
|---|---|---|---|---|---|
| tesseract | ~tiny | none | ✅ ~75 pg/min | Apache-2.0 | floor; weak on degraded/handwriting |
| PaddleOCR PP-OCRv5 | small | optional | ✅ ~23 pg/min | Apache-2.0 | best CPU accuracy/speed balance |
| PaddleOCR-VL | 0.9B / ~2 GB | ~2 GB | ⚠️ only via llama.cpp | Apache-2.0 | smallest credible doc-VLM |
| GOT-OCR 2.0 | 580M / ~1 GB | <3 GB | ✗ | Apache-2.0 | papers/slides |
| dots.ocr | 1.7B / ~3.5 GB | ≥12 GB | ✗ | MIT | strong layout; hard vllm dep |
| Surya / marker | multi-model | ~5–7 GB | ~6 pg/min (slow) | OpenRAIL | this tool's `marker` backend |
| DeepSeek-OCR | 3B MoE / ~6 GB | ~8 GB | ✗ (20–100 s/pg) | MIT | "optical compression" long-context |
| **Baidu Unlimited-OCR** | 3B MoE / ~6 GB | ~8–12 GB | ✗ | MIT | DeepSeek-OCR successor; flat KV cache |
| olmOCR 2 | ~7B | ≥8 GB | ✗ | Apache-2.0 | Qwen-based |

(Sizes/VRAM are approximate; OmniDocBench scores for the newest models are
vendor-reported and lack independent verification.)

### Baidu Unlimited-OCR — verdict

A brand-new (~22 Jun 2026) **GPU-only document VLM**, architecturally a
DeepSeek-OCR descendant: 3B-param MoE (~500M active), ~6 GB weights, MIT, CUDA
required, served via SGLang. Its headline trick — *Reference Sliding Window
Attention* keeping the KV cache flat as output grows — powers an "unlimited
length / one-shot whole-document" claim. **But that is a GPU-server long-context
optimization; it does nothing for a laptop that chunks pages.** Days-old research
code, no independent benchmarks. **Not suitable** for the perf-constrained local
target. (A 16 GB GPU like the 5080 could run it as a curiosity.)

## Recommendation

1. **Ship the `paddle` backend** (done) as the accurate, CPU-viable middle tier,
   pinned to paddlepaddle 3.1.x for the working mkldnn path.
2. Keep **tesseract/`ocrmypdf`** as the lightweight floor and fallback.
3. Treat all VLM engines (including the existing `marker` backend) as **GPU-only
   opt-in** — never the default on constrained machines.
4. **Skip Baidu Unlimited-OCR / DeepSeek-OCR / dots.ocr / olmOCR** for this tool;
   revisit only if a quantized CPU path + independent benchmarks appear.

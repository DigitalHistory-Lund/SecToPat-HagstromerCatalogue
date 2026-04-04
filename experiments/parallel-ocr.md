# Experiment: Parallel OCR extraction

**Date:** 2026-02-24
**Machine:** Apple Silicon Mac (Metal GPU)
**Model:** qwen3-vl:2b via ollama
**Parameters:** SEED=165, NUM_PDF=2 (vols 01+02), NUM_PAGES=5, NUM_CARDS=10 (uncapped — 80 cards detected), --workers 2

## Results

| Run | Hosts | Pages (s) | Cards (s) | OCR (s) | OCR/card (s) | Total (s) |
|-----|-------|-----------|-----------|---------|--------------|-----------|
| baseline (1 host) | 1 | 7.7 | 1.9 | 395.7 | 4.9 | 405.3 |
| 2 native hosts | 2 | 7.8 | 2.0 | 374.0 | 4.7 | 383.8 |
| OLLAMA_NUM_PARALLEL=2 | 1 (×2 threads) | 7.7 | 1.9 | ~395* | ~4.9 | ~405* |

\* Aborted at 80% (64/80 cards in 312s) — pacing identical to baseline.

## Conclusion

**OCR cannot be meaningfully parallelized on a single Apple Silicon Mac.**
The Metal GPU serializes inference regardless of whether parallelism is attempted
via multiple processes, multiple hosts, or OLLAMA_NUM_PARALLEL.
Baseline throughput: **~5s/card** (~7.6h for the full corpus of ~5,500 cards).

Pages/cards extraction parallelizes well with `--workers` (CPU-bound, ~2× speedup
with 2 workers) but is <3% of total runtime — not worth optimizing further.

**Decision:** Removed OCR parallelization. Kept `--workers` for pages/cards.

## Approaches tested

1. **2 native ollama instances** (:11434 + :11435 via `ollama-cluster.sh`):
   Both share the same Metal GPU — only ~5% speedup. GPU serializes compute
   from competing processes.

2. **OLLAMA_NUM_PARALLEL=2** on single instance:
   No improvement. Same GPU bottleneck, just queued internally.

3. **Docker** (aborted): ollama in Docker on macOS runs CPU-only (~60s/card
   vs ~5s native). Docker Desktop's Linux VM has no Metal access.

# Boxwright

**Goal:** Train a small language model from scratch, end to end, that generates novel, solvable Sokoban levels at a requested difficulty. It should beat both the published 2023 baseline (Todd et al., fine-tuned GPT-2) and today's frontier models, and run as a playable game in the browser.

**Approach:** One Rust solver is used three ways: to label the data (Python bindings), as the RL reward (Python bindings), and in the web demo (WASM). The model is a ~25–50M decoder trained from scratch on Boxoban, with difficulty-conditioning prefix tokens. It is then post-trained with GRPO, using solver-verified rewards for playability, hitting the target difficulty, novelty and diversity. The browser demo uses a custom ONNX export with onnxruntime-web (WebGPU), so the architecture isn't constrained by what the export tooling supports.

**Thesis to prove with numbers:** "A 50M model beats GPT-2 (2023) and frontier models at generating solvable levels at a target difficulty, at $0 per level, in about 50ms, on your device."

**Files touched:**
- `solver/` — Rust crate: level parsing, move-optimal A* with deadlock pruning, difficulty features; `pyo3` bindings + `wasm-bindgen` target
- `boxwright/data/` — Boxoban download, solver labeling, symmetry augmentation, train/val/test splits, novelty index
- `boxwright/tokenizer.py` — tile-level vocab + control tokens
- `boxwright/model.py` — decoder (2D RoPE, QK-norm, SwiGLU)
- `boxwright/train_pretrain.py` — pretraining loop (Muon + AdamW, bf16, torch.compile)
- `boxwright/train_grpo.py` — custom GRPO loop with parallel solver rewards
- `boxwright/rewards.py` — reward components and weights
- `boxwright/eval/` — eval harness, Todd et al. reproduction, frontier-model comparison, plots
- `boxwright/export.py` — ONNX export + quantization
- `web/` — Vite + TS app: canvas game, difficulty slider, in-browser generation, solver replay
- `tests/` — solver correctness, tokenizer round-trip, reward unit tests

---

## Key decisions

See [`../design-decisions.md`](../design-decisions.md) (D1–D12, accepted 2026-10-04). Short version: Boxoban data, optimal move length as the difficulty target, autoregressive decoder first (masked diffusion as a stretch comparison), 2D RoPE, Muon + AdamW, RFT baseline → GRPO with a KL anchor and diversity reward, own Rust solver, ONNX + onnxruntime-web, Modal free tier for compute.

**Contracts**

```
solver.solve(level: str, max_nodes: int) -> Solution | None
  Solution: pushes: int, moves: int, path: str, nodes_expanded: int
solver.solve_batch(levels: list[str], max_nodes: int, threads: int) -> list[Solution | None]

reward(level_text: str, target_bucket: int) -> RewardBreakdown
  RewardBreakdown: valid, playable, difficulty_match, novel, total (all floats, logged separately)
```

---

## Todo

- [x] **Task 1: Project scaffold**
  - `uv` Python project, Rust workspace, `web/` stub, git repo, README stating the thesis (W&B project deferred until the first training run, since it needs your login)
  - Acceptance: `uv run pytest` and `cargo test` both pass on empty suites

- [ ] **Task 2: Rust solver + bindings**
  - Move-optimal A* with simple deadlock detection (dead squares, frozen boxes), node budget, solution path output; rayon batch API; pyo3 module
  - Acceptance: solves 99%+ of the Boxoban medium validation set within budget; optimal push counts match a reference solver on a sample; throughput of at least 1k levels/s across cores (needed for RL)
  - Files: `solver/`, `tests/test_solver.py`

- [ ] **Task 3: Dataset pipeline**
  - Download Boxoban (unfiltered + medium + hard), label every level with the solver, compute difficulty buckets, 8× dihedral augmentation, deduplicated train/val/test splits, novelty hash index
  - Acceptance: Parquet shards plus a dataset card with stats (difficulty histogram, % solvable, dedup counts)
  - Files: `boxwright/data/`

- [ ] **Task 4: Tokenizer**
  - Tile vocab, control tokens, encode/decode, structural validity check (10×10, exactly 1 player, boxes == goals)
  - Acceptance: lossless round-trip on the full dataset; validity check agrees with the solver's parser
  - Files: `boxwright/tokenizer.py`, `tests/test_tokenizer.py`

- [ ] **Task 5: Model + pretraining**
  - Decoder at ~3 sizes (6M / 20M / 50M) (5M / 20M / 50M); Muon for hidden matrices, AdamW for embeddings and head; bf16; torch.compile; cosine/WSD schedule; checkpoints; sample grids logged to W&B during training
  - Acceptance: the 50M model reaches stable val loss; unconditional samples are ≥95% structurally valid; the conditioned model shows a monotonic relation between requested and actual difficulty
  - Files: `boxwright/model.py`, `boxwright/train_pretrain.py`

- [ ] **Task 6: Eval harness**
  - Metrics: valid, playable, novel, diverse, Todd's combined "score", difficulty accuracy (within ±tolerance) and a calibration curve, nearest-neighbor edit distance, samples/sec
  - Acceptance: one command evaluates any checkpoint (or text file of levels) and emits JSON + plots; deterministic given a seed
  - Files: `boxwright/eval/`

- [ ] **Task 7: Reproduce Todd et al. baseline**
  - Run `gdrtodd/lm-pcg` GPT-2 fine-tune on Boxoban (or use released checkpoints), score its samples with *our* harness; cross-check against their reported numbers
  - Acceptance: baseline numbers in our harness, with any discrepancy vs. the paper explained
  - Files: `boxwright/eval/todd_baseline.py`

- [ ] **Task 8: GRPO post-training**
  - Prompt = difficulty bucket; G=16 samples per prompt; rewards from Task 2's batch solver in a process pool; reward = validity gate + playable + difficulty match (smooth in |actual − target|) + novelty; within-group diversity bonus if collapse appears; log each reward component
  - Acceptance: playable-novel rate and difficulty accuracy improve clearly over the pretrained checkpoint on the held-out eval without diversity collapse (track unique-level rate); a reward-hacking review of 200 samples by eye
  - Files: `boxwright/train_grpo.py`, `boxwright/rewards.py`, `tests/test_rewards.py`

- [ ] **Task 9: Frontier-model comparison**
  - Few-shot prompt current frontier models (Claude, GPT, Gemini) for levels at each difficulty bucket, N≈100 per bucket; same harness; record cost and latency per level
  - Acceptance: a single comparison table + calibration plot: Boxwright pre-RL / post-RL vs. Todd GPT-2 vs. frontier models
  - Files: `boxwright/eval/frontier.py`

- [ ] **Task 10: Ablations**
  - Model size (6M/20M/50M), RL vs. no RL, 1D vs. 2D positional signal, data-fraction scaling (mirrors Todd's scaling result)
  - Acceptance: plots + a short findings section in the writeup
  - Files: `boxwright/eval/`

- [ ] **Task 11: Export + quantization**
  - Export to ONNX (custom sampling loop); int8 and int4 (q4f16) variants; parity check vs. PyTorch logits; publish to Hugging Face Hub with a model card
  - Acceptance: quantized model's eval metrics within ~1 pt of fp32; model runs in onnxruntime-web
  - Files: `boxwright/export.py`

- [ ] **Task 12: Browser demo**
  - Vite + TS: canvas Sokoban with undo/reset, difficulty slider, "Generate" button running the model on WebGPU (WASM fallback), solver-in-WASM verifies each level before showing it and can replay the optimal solution; latency readout; "compare" page showing sample frontier-model failures
  - Acceptance: works offline after first load on desktop Chrome/Safari; level generated + verified in under 200ms on a laptop; deployed to a static host
  - Files: `web/`

- [ ] **Task 13: Writeup**
  - Blog post: thesis, pipeline diagram, results table, calibration plot, before/after RL samples, ablations, lessons; public W&B report; README with reproduce commands
  - Acceptance: someone can reproduce the main table from the README
  - Files: `README.md`, `docs/`

---

## Risks

- **Solver too slow for RL** → strict node budget, cache by canonical level hash, reward unsolved-in-budget as 0 (same as eval)
- **Reward hacking / mode collapse** (e.g., one easy template repeated) → novelty against train *and* within-batch, diversity metric watched every eval step
- **Hardest difficulty buckets are rare in the data** → oversample during pretraining; RL with dynamic sampling targets them
- **Frontier models do better than expected** → still a result; the cost, latency and on-device story remain, and the RL'd model's calibration is the differentiator

# Boxwright — Design Decisions

**Status:** Accepted (2026-10-04)

Each decision lists the options considered, the chosen option, and what would change it.

Summary of recommendations:

| # | Decision | Recommendation |
|---|---|---|
| D1 | Training data | Boxoban (unfiltered + medium), with existing 2024 A* labels as a cross-check |
| D2 | Difficulty target | Optimal solution length (moves), bucketed; search effort as a secondary metric |
| D3 | Generation paradigm | Autoregressive decoder first; masked diffusion as a planned head-to-head stretch |
| D4 | Sequence format | 100 tile tokens, raster order, difficulty prefix, no row separators |
| D5 | Positional encoding | 2D axial RoPE; 1D RoPE as ablation |
| D6 | Architecture & size | Small dense pre-norm decoder (RMSNorm, QK-norm, SwiGLU); 6M / 20M / 50M sweep |
| D7 | Optimizer & schedule | Muon (hidden matrices) + AdamW (rest), warmup-stable-decay; AdamW-only baseline run |
| D8 | Post-training | Rejection-sampling fine-tuning as baseline → GRPO-family RL with a KL anchor and diversity reward |
| D9 | Solver | Own Rust A* solver → Python bindings + WASM |
| D10 | Training code | Plain PyTorch, own loops, single GPU; no HF Trainer/TRL |
| D11 | Browser deployment | Custom ONNX export + onnxruntime-web (WebGPU), own sampling loop |
| D12 | Compute | Modal free tier + Lambda grant application; Vast/RunPod as paid fallback |

---

## D1 — Training data

**Finding: Boxoban is still the standard; nothing newer or larger has replaced it.** I found no new large-scale Sokoban *level* dataset released in 2024–2026. Recent work (SokoBench 2026, FAR AI's Sokoban RNN interpretability work 2024–25, various RL papers) all builds on Boxoban or on procedural generators.

| Source | Size | Notes |
|---|---|---|
| **Boxoban** (DeepMind, 2018) | unfiltered 900k train / 100k val / 1k test; medium 450k train / 50k val; hard 3,332 | 10×10, 4 boxes, procedurally generated (reverse-play), then filtered by agent difficulty. Same data as Todd et al. |
| **AlignmentResearch/boxoban-astar-solutions** (2024, Apache-2.0) | A* solutions for unfiltered train/val/test + medium val | **Move**-optimal under a Manhattan heuristic, 1M/5M node budgets; ~500–600 unsolved per split. Useful free labels and a solver cross-check. |
| Human-designed collections (Microban ~155–500, XSokoban 90, Sasquatch, …) | Hundreds | Variable sizes, curated by experts. Far too small to train on. |
| Procedural generators (gym-sokoban reverse play, Parberry's generator) | Unlimited | Can make extra levels for rare difficulty buckets if needed. |

**Recommendation:** Train on Boxoban unfiltered-train + medium-train (~1.35M unique levels, ~10.8M with 8× symmetry augmentation). Validate on unfiltered-val. Hold out unfiltered-test + medium-val for final eval and novelty checks. Use hard (3.3k) only to examine the top end of the difficulty range. Use human collections only for a qualitative "does it look designed?" comparison in the writeup, not for training.

**Open question to resolve in Task 7:** which Boxoban split(s) Todd et al. fine-tuned on, so the baseline comparison is exact.

**Would change if:** difficulty buckets above the 90th percentile are too sparse → supplement with a procedural generator (reverse-play from goal states) for hard levels.

---

## D2 — What "difficulty" means

| Option | Pros | Cons |
|---|---|---|
| **Optimal solution length (moves)** | Objective; exact with A*; what Todd et al. controlled → direct comparison; free labels exist | Long ≠ hard (a long corridor walk is tedious, not hard) |
| Optimal pushes | Closer to "puzzle-ness" than moves | Still length-based; push-optimal search is a bit slower |
| Search effort (nodes expanded by a fixed solver) | Correlates better with real difficulty | Depends on the solver; noisier |
| Multi-attribute (length + box-line changes + dead-square count…) | Richest control | More complex; harder to explain |

**Recommendation:** The control target is **optimal move length, split into 10 quantile buckets** (`<d0>`…`<d9>`), for direct comparability with the baseline. Log pushes and search effort for every level, and report how they correlate with length. If time allows, add a second control knob (search effort) after the main result is in.

---

## D3 — Generation paradigm: autoregressive vs. diffusion

This is the most consequential choice.

| Option | Pros | Cons |
|---|---|---|
| **Autoregressive (AR) decoder** | Mature, well-understood RL (exact log-probabilities → clean GRPO); direct comparison with Todd's GPT-2; simplest to deploy | Raster order is unnatural for a 2D grid with global constraints (box count = goal count, reachability) |
| Masked discrete diffusion (MDLM/LLaDA style) | Bidirectional, any-order, parallel decoding. Published results show large gains over AR on globally constrained grid tasks (e.g. Sudoku ~100% vs ~21% in one study). Generation is visually striking (the level "crystallizes"). | RL for diffusion is younger (diffu-GRPO / wd1 use approximate log-probabilities); more moving parts; custom sampler needed in the browser |
| Hybrid (block diffusion) | Middle ground | The most complexity for this project |

**Recommendation:** Build **AR first**, end to end, with RL and deployment. Then, as a stretch phase, train a **masked-diffusion model on the same data, eval and reward**, and report the head-to-head. The training loop, solver, reward and eval harness are all reused. "AR vs. diffusion for constrained level generation" would itself be new as a result, and the diffusion demo animation is a strong visual.

**Would change if:** you want diffusion as the main bet. That's defensible (more novel, possibly better results), at the cost of schedule risk in the RL stage.

---

## D4 — Sequence format

| Choice | Decision | Reason |
|---|---|---|
| Vocabulary | `# ␣ @ $ . * +` (wall, floor, player, box, goal, box-on-goal, player-on-goal) + `BOS`, `EOS`, `<d0..d9>`, `<uncond>` (~20 tokens) | One token per tile; BPE would only blur structure |
| Layout | 100 tokens, row-major, no row separators | Fixed 10×10 → separators carry no information; 2D position comes from D5 |
| Conditioning | `BOS <dK> grid EOS`; the difficulty token is replaced by `<uncond>` 10% of the time | Allows unconditional sampling and classifier-free guidance at inference |
| Loss | Grid tokens only | The prefix is given, not predicted |

Note to check: whether the Boxoban outer ring is always wall. If so, we could predict only the 8×8 interior (64 tokens). I'd still keep the full 100 for simplicity and generality.

---

## D5 — Positional encoding

| Option | Pros | Cons |
|---|---|---|
| Learned absolute (100 positions) | Trivial; fine at a fixed length | No built-in notion of rows and columns |
| 1D RoPE | Standard | "Directly above" is 10 tokens away, which the model must learn |
| **2D axial RoPE** (half the head dims rotate by row, half by column) | Grid-native; used in modern vision transformers and VLM M-RoPE | Not supported by stock transformers.js model classes (acceptable given D11) |

**Recommendation:** Use 2D axial RoPE; run 1D RoPE as an ablation. The ablation is cheap, and the result is a clean chart for the writeup.

---

## D6 — Architecture & size

The workload is tiny: about 102 tokens per sample, a vocabulary of about 20, and ~1.1B tokens per epoch with augmentation. Big-model features (MoE, MLA, sliding windows, GQA, long-context tricks) give no benefit here, and including them would look like cargo-culting.

| Component | Choice | Alternative considered |
|---|---|---|
| Block | Pre-norm decoder, RMSNorm, no biases | — |
| Attention | Standard multi-head + **QK-norm** (stability with Muon) | GQA: no benefit at ~100 tokens |
| MLP | **SwiGLU** | ReLU² (speedrun favorite): equivalent here; SwiGLU is the mainstream choice |
| Embeddings | Untied (the vocabulary is ~20, so it costs nothing) | — |
| Extras | None at first; logit soft-capping only if instability appears | Value embeddings / U-net skips from modded-nanogpt: small gains, more to explain |
| Sizes | **6M / 20M / 50M**, chosen from a scaling sweep | — |

A rough Chinchilla estimate (~20 tokens per parameter) suggests ~50M parameters for ~1B unique tokens. Level data is very low-entropy, though, so 20M may match 50M. The sweep decides. Browser download size is ~25 MB at int4 for 50M parameters, which is fine.

---

## D7 — Optimizer & schedule

- **Muon** for 2D hidden weight matrices (built into PyTorch as `torch.optim.Muon` since 2.9; single-GPU only, which is fine here). **AdamW** for embeddings, the output head and norms. This is now standard practice (nanochat, modded-nanogpt, several open frontier models).
- **Warmup-stable-decay (WSD)** schedule: a branch can decay from any stable-phase checkpoint, so the data-scaling ablation costs nothing extra.
- bf16 autocast, `torch.compile`.
- **One AdamW-only control run** at the 20M size, to show the Muon gain with our own numbers rather than claiming it.

---

## D8 — Post-training: RL algorithm

**Our setting differs from reasoning RL.** Outputs are short and fixed-length (100 tokens), and the reward is cheap and exact. Most importantly, **diversity is part of the product**: a generator that outputs one perfect level forever is useless. That changes which GRPO-family tweaks matter:

| Variant | What it fixes | Relevant here? |
|---|---|---|
| GRPO | Critic-free group baseline | ✅ core |
| Dr. GRPO | Length bias; std-normalization bias | Length: no (fixed length). Std-normalization: yes, worth an ablation |
| DAPO dynamic sampling | Wasted groups where every sample gets the same reward | ✅ (e.g. easy buckets where all 16 samples are solvable) |
| DAPO clip-higher | Entropy collapse when reusing samples for several update steps | ✅ if we do >1 update per batch |
| DAPO token-level loss / overlong shaping | Long variable-length chain-of-thought | ❌ not applicable |
| GSPO (sequence-level ratio) | MoE / long-sequence instability | ❌ not applicable |
| CISPO | Off-policy sample reuse | Maybe later, if generating samples becomes the bottleneck |
| Dropping the KL penalty (reasoning-RL fashion) | Lets reasoning models drift freely | ❌ **We keep a KL anchor to the pretrained model**, because collapsing diversity is our main failure mode |

**Recommendation:**
1. **Baseline: rejection-sampling fine-tuning (RFT).** Sample, keep the levels the solver verifies as playable, novel and on target, then supervised fine-tune on them. It's simple and strong, and RL has to beat it to justify itself.
2. **Main: GRPO** with group size 16, mean-centered advantages (std normalization as an ablation), DAPO dynamic sampling, clip-higher, and a **KL term to the pretrained model**. Try a mass-covering divergence (forward-KL / JS, per DPH-RL 2025) as an ablation.
3. **Reward** (each component logged separately):
   - validity gate (structurally invalid → −1)
   - playable (solved within budget)
   - difficulty match (smooth, e.g. `exp(−|len − target| / σ)`)
   - novelty (not in the training set under 8 symmetries)
   - within-group duplicate penalty
4. **Prompts:** difficulty buckets sampled uniformly, with extra weight on the hardest buckets as a curriculum.

Writeup charts: RFT vs. GRPO; with vs. without KL anchor (diversity curves); reward components over training.

---

## D9 — Solver

| Option | Pros | Cons |
|---|---|---|
| **Own Rust A*** (move-optimal, deadlock pruning, node budget) | Python bindings for data and RL; compiles to WASM for the browser; demonstrates systems skill | Must be written and tested |
| Existing strong solvers (Festival, YASS) | Very strong | Not embeddable in Python or WASM; built for hard human levels, which is overkill here |
| Python solver | Easy | Too slow for RL (needs ≥1k levels/s) |

**Recommendation:** Own Rust solver, validated against the AlignmentResearch A* labels (solution lengths must match where both solve).

---

## D10 — Training code

Plain PyTorch with our own pretraining and RL loops, on a single GPU (the model is too small to need multi-GPU), with W&B logging. TRL/verl/OpenRLHF assume Hugging Face model classes, long text and multi-GPU, so they would add weight without helping. Writing GRPO ourselves is also part of what we want to show.

---

## D11 — Browser deployment

| Option | Pros | Cons |
|---|---|---|
| transformers.js | Easy if the architecture is supported | Our 2D RoPE isn't; would force the architecture to fit the tool |
| WebLLM / MLC | Fast | Heavy toolchain for a 50M model |
| **ONNX export + onnxruntime-web (WebGPU, WASM fallback)** | Any architecture; tiny vocabulary makes our own sampling loop ~50 lines; diffusion sampler fits the same setup | Write our own KV cache or skip it (100 tokens → even recomputing without a cache is cheap) |
| Hand-written WGSL kernels | Impressive | Not worth the time |

**Recommendation:** ONNX + onnxruntime-web with our own sampler, plus the solver compiled to WASM for in-browser verification and solution replay. The demo shows how many samples were rejected before a verified level appeared, honestly, rather than hiding failures.

---

## D12 — Compute (to be finalized when we get there)

**Estimated need is small.** Pretraining 50M parameters on ~5B tokens ≈ 6·N·D ≈ 1.5e18 FLOPs, about 1–2 H100-hours. With the size sweep, ablations and RL, the total is probably **~20–40 GPU-hours**, plus a lot of **CPU** for the solver during RL.

| Option | What you get | Fit for an individual side project |
|---|---|---|
| **Modal Starter (free)** | $30/month in credits (≈7.5 H100-hours/month), per-second billing, cheap parallel CPU containers | ✅ Best fit: serverless Python, and the solver can fan out over CPU containers during RL |
| **Lambda Research Grant** | Up to $5k in cloud credits, rolling applications | ✅ Worth applying; a public open-source project with a writeup is a reasonable pitch |
| Google TPU Research Cloud | Free TPUs for accepted researchers; open to non-academics; expects published results | ⚠️ Free, but TPUs mean JAX or PyTorch/XLA, a detour from our stack |
| Lightning AI free tier | ~80 spot GPU-hours/month (smaller GPUs) | ✅ Good for development runs and small sweeps |
| Kaggle | 30 GPU-hours/week on T4/P100 | ✅ Free; fine for 6M/20M development runs |
| Colab free | T4, not guaranteed | Development only |
| Azure new account | $200 for 30 days | ⚠️ One-off; GPU quota on new accounts is uncertain |
| Google Cloud $300 trial | — | ❌ GPUs are not allowed on the free trial |
| Startup programs (AWS Activate, Google for Startups, MS Founders Hub, NVIDIA Inception, DO Hatch) | $1k–$350k | ❌ Require a company and usually VC/accelerator backing (MS Founders Hub's lowest tier is the easiest to get) |
| Nebius research credits | Up to 8 GPUs for a year | ❌ Academic affiliation required |
| **Paid fallback:** Vast.ai / RunPod / Thunder Compute / Lambda | H100 ≈ $1.9–2.5/hr on demand | ✅ Even 40 hours ≈ $100 |

**Recommendation:** Do development on the Mac (MPS) and Kaggle/Lightning free tiers. Use Modal for real runs (it covers ~7.5 H100-hours/month for free). Submit the Lambda grant application early, since it's rolling. Expected out-of-pocket cost: ≈$0–100. Prices and offers come from 2026 aggregator pages; verify them at sign-up.

---

## Sources

- Boxoban levels — https://github.com/google-deepmind/boxoban-levels
- AlignmentResearch/boxoban-astar-solutions — https://huggingface.co/datasets/AlignmentResearch/boxoban-astar-solutions
- Todd et al., Level Generation Through LLMs (2023) — https://arxiv.org/abs/2302.05817
- SokoBench (2026) — https://arxiv.org/pdf/2601.20856
- Beyond Autoregression: Discrete Diffusion for Complex Reasoning and Planning — https://arxiv.org/abs/2410.14157
- d1 / diffu-GRPO — https://arxiv.org/abs/2504.12216
- From GRPO to DAPO and GSPO — https://huggingface.co/blog/NormalUhr/grpo-to-dapo-and-gspo
- DPH-RL, divergence choice vs. diversity collapse — https://arxiv.org/abs/2509.07430
- Understanding Diversity Collapse in RLVR (2026) — https://arxiv.org/abs/2606.15455
- torch.optim.Muon — https://docs.pytorch.org/docs/2.13/generated/torch.optim.Muon.html
- modded-nanogpt speedrun techniques — https://www.lesswrong.com/posts/j3gp8tebQiFJqzBgg/how-the-nanogpt-speedrun-wr-dropped-by-20-in-3-months
- Transformers.js releases — https://github.com/huggingface/transformers.js/releases
- Modal pricing — https://modal.com/pricing
- Lambda Research Grant — https://www.businesswire.com/news/home/20241209358457/en
- Google TPU Research Cloud — https://sites.research.google/trc/about/
- Free GPU credit overview (2026) — https://www.thundercompute.com/blog/free-cloud-gpu-credits
- Free cloud GPUs, what works and what closed — https://gpuperhour.com/blog/free-cloud-gpus-and-credits
- H100 rental prices (2026) — https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison

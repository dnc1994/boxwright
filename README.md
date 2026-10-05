# Boxwright

A small language model, trained from scratch end to end, that designs novel, solvable Sokoban levels at a requested difficulty, and runs in your browser.

**Thesis:** a ~50M-parameter model beats both a fine-tuned GPT-2 (Todd et al., 2023) and today's frontier models at generating solvable levels on target difficulty, at $0 per level, in milliseconds, on device.

## Layout

| Path | What |
|---|---|
| `solver/` | Rust Sokoban solver (Python bindings for data + RL rewards, WASM for the browser) |
| `boxwright/` | Python package: data pipeline, tokenizer, model, pretraining, GRPO, evals, export |
| `web/` | Browser demo |
| `docs/` | [Design decisions](docs/design-decisions.md) and [plan](docs/plans/2026-10-04-boxwright.md) |

## Development

```sh
uv run pytest      # Python tests
cargo test         # solver tests
```

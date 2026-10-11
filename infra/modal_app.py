"""Run Boxwright training on Modal.

One-time setup (from the repo root, after `python -m boxwright.data.build`):
    uv run modal volume create boxwright-data && uv run modal volume create boxwright-runs
    for f in train.parquet val.parquet test.parquet hard.parquet buckets.json; do
        uv run modal volume put boxwright-data data/processed/$f processed/$f; done

Pretraining (arguments are passed through to boxwright.train_pretrain):
    uv run modal run infra/modal_app.py --args "--preset 6m --steps 3000 --wandb --out /runs/smoke-6m"
    uv run modal run infra/modal_app.py --gpu L40S --args "..."

Evaluation (arguments are passed through to boxwright.eval.harness; paths are on the runs volume):
    uv run modal run infra/modal_app.py::evaluate --args "--checkpoint /runs/pt-20m-2d-muon/final.pt --out /runs/pt-20m-2d-muon/eval"

Checkpoints and evals land in the `boxwright-runs` volume; fetch them with
    uv run modal volume get boxwright-runs smoke-6m/final.pt checkpoints/
"""

import shlex
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "build-essential")
    .run_commands("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal")
    .uv_pip_install("torch==2.14.1", "numpy", "pyarrow", "wandb", "rapidfuzz", "matplotlib")
    # Build the Rust solver extension inside the image.
    .add_local_file(REPO / "Cargo.toml", "/build/Cargo.toml", copy=True)
    .add_local_file(REPO / "Cargo.lock", "/build/Cargo.lock", copy=True)
    .add_local_dir(REPO / "solver", "/build/solver", copy=True, ignore=["target", "*.so"])
    .run_commands(". $HOME/.cargo/env && pip install /build/solver")
    .add_local_python_source("boxwright")
)

app = modal.App("boxwright", image=image)
data = modal.Volume.from_name("boxwright-data", create_if_missing=True)
runs = modal.Volume.from_name("boxwright-runs", create_if_missing=True)


@app.function(
    gpu="H100",
    cpu=8,  # the solver scores eval samples on CPU cores
    memory=32 * 1024,
    timeout=6 * 3600,
    volumes={"/data": data, "/runs": runs},
    secrets=[modal.Secret.from_name("wandb-secret")],
    env={"BOXWRIGHT_DATA": "/data/processed", "WANDB_ENTITY": "l1nghao-google-deepmind"},
)
def pretrain(argv: list[str]) -> None:
    from boxwright.train_pretrain import main

    main(argv)
    runs.commit()


@app.function(
    gpu="L4",  # sampling a 20–50M model is light; scoring is CPU-bound
    cpu=16,
    memory=32 * 1024,
    timeout=2 * 3600,
    volumes={"/data": data, "/runs": runs},
    env={"BOXWRIGHT_DATA": "/data/processed"},
)
def run_eval(argv: list[str]) -> None:
    from boxwright.eval.harness import main

    main(argv)
    runs.commit()


@app.local_entrypoint()
def run(args: str = "", gpu: str = "H100") -> None:
    pretrain.with_options(gpu=gpu).remote(shlex.split(args))


@app.local_entrypoint()
def evaluate(args: str = "") -> None:
    run_eval.remote(shlex.split(args))

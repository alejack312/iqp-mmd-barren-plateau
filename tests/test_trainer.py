from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np

from iqp_bp.iqp.model import IQPModel
from iqp_bp.training import Trainer


def _uniform_dataset(n: int, repeats: int = 32) -> np.ndarray:
    rows = [
        [int(bit) for bit in format(index, f"0{n}b")]
        for index in range(2**n)
    ]
    return np.repeat(np.asarray(rows, dtype=np.uint8), repeats, axis=0)


def _read_rows(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _workspace_tmp_dir() -> Path:
    path = Path("tests") / "_tmp_training" / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_trainer_decreases_exact_small_n_loss_monotonically():
    data = _uniform_dataset(2)
    model = IQPModel(G=np.eye(2, dtype=np.uint8), theta=np.array([0.8, -0.6], dtype=float))
    tmp_path = _workspace_tmp_dir()
    trainer = Trainer(
        model,
        data,
        output_dir=tmp_path / "run",
        kernel="gaussian",
        kernel_params={"sigma": 4.0},
        optimizer="sgd",
        lr=0.1,
        num_steps=4,
        checkpoint_every=1,
        num_a_samples=32,
        num_z_samples=64,
        loss_mode="exact_small_n",
        exact_loss_max_n=4,
        stream_seeds={"kernel": 1, "estimation": 2, "callback": 3},
    )
    trainer.run()
    losses = [row["loss"] for row in _read_rows(trainer.trajectory_path)]

    assert losses == sorted(losses, reverse=True)


def test_trainer_is_bit_reproducible_for_same_seed_bundle():
    data = _uniform_dataset(2)
    tmp_path = _workspace_tmp_dir()

    def run_once(name: str) -> list[dict]:
        model = IQPModel(G=np.eye(2, dtype=np.uint8), theta=np.array([0.5, -0.4], dtype=float))
        trainer = Trainer(
            model,
            data,
            output_dir=tmp_path / name,
            kernel="gaussian",
            kernel_params={"sigma": 3.0},
            optimizer="sgd",
            lr=0.1,
            num_steps=3,
            checkpoint_every=1,
            num_a_samples=16,
            num_z_samples=32,
            loss_mode="exact_small_n",
            exact_loss_max_n=4,
            stream_seeds={"kernel": 11, "estimation": 12, "callback": 13},
        )
        trainer.run()
        rows = _read_rows(trainer.trajectory_path)
        for row in rows:
            row.pop("wall_clock_sec", None)
            row.pop("checkpoint_path", None)
        return rows

    assert run_once("a") == run_once("b")


def test_trainer_writes_jsonl_and_npz_checkpoints():
    data = _uniform_dataset(2)
    model = IQPModel(G=np.eye(2, dtype=np.uint8), theta=np.array([0.5, 0.1], dtype=float))
    tmp_path = _workspace_tmp_dir()
    trainer = Trainer(
        model,
        data,
        output_dir=tmp_path / "run",
        kernel="gaussian",
        kernel_params={"sigma": 2.0},
        optimizer="sgd",
        lr=0.05,
        num_steps=2,
        checkpoint_every=1,
        num_a_samples=16,
        num_z_samples=32,
        loss_mode="exact_small_n",
        exact_loss_max_n=4,
        stream_seeds={"kernel": 21, "estimation": 22, "callback": 23},
    )
    result = trainer.run()
    rows = _read_rows(result["trajectory_path"])

    assert [row["step"] for row in rows] == [0, 1, 2]
    assert all(Path(row["checkpoint_path"]).exists() for row in rows)

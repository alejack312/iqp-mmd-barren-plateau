"""Minimal MMD trainer with per-step trajectory persistence."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from iqp_bp.experiments.run_validation import save_iqp_checkpoint
from iqp_bp.iqp.expectation import iqp_phase
from iqp_bp.iqp.model import IQPModel
from iqp_bp.mmd.gradients import grad_mmd2_analytic
from iqp_bp.mmd.loss import mmd2, mmd2_exact_small_n
from iqp_bp.rng import STREAM_ESTIMATION, STREAM_KERNEL, derive_seed

CheckpointCallback = Callable[
    [int, IQPModel, np.ndarray, dict[str, Any], np.random.Generator],
    dict[str, Any],
]


class Trainer:
    """Train one IQP model against one dataset and persist a checkpoint trajectory."""

    def __init__(
        self,
        model: IQPModel,
        data: np.ndarray,
        *,
        output_dir: str | Path,
        kernel: str = "gaussian",
        kernel_params: dict[str, Any] | None = None,
        optimizer: str = "adam",
        lr: float = 0.05,
        num_steps: int = 100,
        checkpoint_every: int = 10,
        num_a_samples: int = 512,
        num_z_samples: int = 1024,
        batch_size: int | None = None,
        stream_seeds: dict[str, int] | None = None,
        loss_mode: str = "auto",
        exact_loss_max_n: int = 12,
        checkpoint_callback: CheckpointCallback | None = None,
    ) -> None:
        self.model = model
        self.data = np.asarray(data, dtype=np.uint8)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.kernel = kernel
        self.kernel_params = dict(kernel_params or {})
        self.optimizer = optimizer.lower()
        self.lr = float(lr)
        self.num_steps = int(num_steps)
        self.checkpoint_every = int(checkpoint_every)
        self.num_a_samples = int(num_a_samples)
        self.num_z_samples = int(num_z_samples)
        self.batch_size = batch_size
        self.stream_seeds = dict(stream_seeds or {})
        self.loss_mode = loss_mode
        self.exact_loss_max_n = int(exact_loss_max_n)
        self.checkpoint_callback = checkpoint_callback

        if self.optimizer not in {"sgd", "adam"}:
            raise ValueError(f"unsupported optimizer {optimizer!r}; use 'sgd' or 'adam'")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.num_steps < 0:
            raise ValueError("num_steps must be >= 0")
        if self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive")
        if self.num_a_samples <= 0 or self.num_z_samples <= 0:
            raise ValueError("num_a_samples and num_z_samples must be positive")

        self.trajectory_path = self.output_dir / "trajectory.jsonl"
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._adam_m = np.zeros_like(self.model.theta, dtype=np.float64)
        self._adam_v = np.zeros_like(self.model.theta, dtype=np.float64)
        self._adam_t = 0

    def run(self) -> dict[str, Any]:
        """Run the configured optimization loop and return artifact metadata."""
        start_time = time.perf_counter()
        written_steps: list[int] = []

        with open(self.trajectory_path, "w", encoding="utf-8") as handle:
            initial_details = self._compute_loss_details(step=0, phase="snapshot")
            self._write_snapshot(
                handle=handle,
                step=0,
                loss_details=initial_details,
                wall_clock=time.perf_counter() - start_time,
            )
            written_steps.append(0)

            for step in range(1, self.num_steps + 1):
                gradient = self._compute_gradient(step=step)
                self._apply_update(gradient)

                if step % self.checkpoint_every == 0 or step == self.num_steps:
                    snapshot_details = self._compute_loss_details(step=step, phase="snapshot")
                    self._write_snapshot(
                        handle=handle,
                        step=step,
                        loss_details=snapshot_details,
                        wall_clock=time.perf_counter() - start_time,
                    )
                    written_steps.append(step)

        final_loss = self._compute_loss_details(step=self.num_steps, phase="summary")["mmd2"]
        return {
            "trajectory_path": str(self.trajectory_path),
            "checkpoint_dir": str(self.checkpoint_dir),
            "written_steps": written_steps,
            "final_theta": self.model.theta.tolist(),
            "final_loss": float(final_loss),
            "num_steps": self.num_steps,
        }

    def _compute_loss_details(self, *, step: int, phase: str) -> dict[str, Any]:
        use_exact = self.loss_mode == "exact_small_n" or (
            self.loss_mode == "auto" and self.model.n <= self.exact_loss_max_n
        )
        if use_exact:
            return mmd2_exact_small_n(
                theta=self.model.theta,
                G=self.model.G,
                data=self.data,
                kernel=self.kernel,
                return_details=True,
                **self.kernel_params,
            )

        rng = np.random.default_rng(self._seed_for(STREAM_KERNEL, phase, step))
        return mmd2(
            theta=self.model.theta,
            G=self.model.G,
            data=self.data,
            kernel=self.kernel,
            num_a_samples=self.num_a_samples,
            num_z_samples=self.num_z_samples,
            rng=rng,
            batch_size=self.batch_size,
            return_details=True,
            **self.kernel_params,
        )

    def _compute_gradient(self, *, step: int) -> np.ndarray:
        details = self._compute_loss_details(step=step, phase="gradient")
        use_exact = self.loss_mode == "exact_small_n" or (
            self.loss_mode == "auto" and self.model.n <= self.exact_loss_max_n
        )
        if use_exact:
            return self._compute_exact_small_n_gradient(details)

        gradient = np.zeros_like(self.model.theta, dtype=np.float64)
        observable_weights = details.get("weights")
        for param_idx in range(self.model.m):
            gradient[param_idx] = grad_mmd2_analytic(
                theta=self.model.theta,
                G=self.model.G,
                data=self.data,
                param_idx=param_idx,
                kernel=self.kernel,
                num_a_samples=self.num_a_samples,
                num_z_samples=self.num_z_samples,
                rng=np.random.default_rng(self._seed_for(STREAM_ESTIMATION, "gradient", step, param_idx)),
                batch_size=self.batch_size,
                a_samples=details["a_samples"],
                exp_p=details["exp_p"],
                observable_weights=observable_weights,
                **self.kernel_params,
            )
        return gradient

    def _compute_exact_small_n_gradient(self, details: dict[str, Any]) -> np.ndarray:
        weights = np.asarray(details["weights"], dtype=np.float64)
        gradient = np.zeros_like(self.model.theta, dtype=np.float64)
        for param_idx in range(self.model.m):
            weighted_sum = 0.0
            for a, exp_p, exp_q, weight in zip(
                details["a_samples"],
                details["exp_p"],
                details["exp_q"],
                weights,
                strict=False,
            ):
                dq = _grad_expectation_exact(self.model.theta, self.model.G, a, param_idx)
                weighted_sum += float(weight) * float(exp_p - exp_q) * dq
            gradient[param_idx] = -2.0 * weighted_sum
        return gradient

    def _apply_update(self, gradient: np.ndarray) -> None:
        if self.optimizer == "sgd":
            self.model.theta = self.model.theta - self.lr * gradient
            return

        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        self._adam_t += 1
        self._adam_m = beta1 * self._adam_m + (1.0 - beta1) * gradient
        self._adam_v = beta2 * self._adam_v + (1.0 - beta2) * (gradient**2)
        m_hat = self._adam_m / (1.0 - beta1**self._adam_t)
        v_hat = self._adam_v / (1.0 - beta2**self._adam_t)
        self.model.theta = self.model.theta - self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def _write_snapshot(
        self,
        *,
        handle,
        step: int,
        loss_details: dict[str, Any],
        wall_clock: float,
    ) -> None:
        checkpoint_path = save_iqp_checkpoint(
            self.model,
            self.checkpoint_dir / f"step_{step:04d}.npz",
            metadata={
                "step": int(step),
                "loss": float(loss_details["mmd2"]),
                "wall_clock_sec": float(wall_clock),
            },
        )
        row: dict[str, Any] = {
            "step": int(step),
            "loss": float(loss_details["mmd2"]),
            "theta": self.model.theta.tolist(),
            "wall_clock_sec": float(wall_clock),
            "checkpoint_path": str(checkpoint_path),
        }
        if "mc_diagnostics" in loss_details:
            row["loss_stderr"] = float(loss_details["mc_diagnostics"]["stderr"])
            row["loss_num_observables"] = int(loss_details["mc_diagnostics"]["num_samples"])
        elif "weights" in loss_details:
            row["loss_num_observables"] = int(len(loss_details["weights"]))

        if self.checkpoint_callback is not None:
            callback_rng = np.random.default_rng(self._seed_for("callback", "snapshot", step))
            row.update(
                self.checkpoint_callback(step, self.model, self.data, loss_details, callback_rng)
            )

        handle.write(json.dumps(row) + "\n")

    def _seed_for(self, stream: str, *parts: object) -> int:
        base_seed = self.stream_seeds.get(stream)
        if base_seed is None:
            base_seed = self.stream_seeds.get(STREAM_KERNEL, 0)
        return derive_seed(int(base_seed), "trainer", *parts)


def _grad_expectation_exact(
    theta: np.ndarray,
    G: np.ndarray,
    a: np.ndarray,
    param_idx: int,
) -> float:
    """Exact small-n derivative of ``<Z_a>`` by enumerating all ``z`` bitstrings."""
    n = G.shape[1]
    if n > 20:
        raise ValueError(f"exact gradient infeasible for n={n} > 20")

    g_i = G[param_idx]
    a_dot_gi = int((a @ g_i) % 2)
    if a_dot_gi == 0:
        return 0.0

    all_z = np.array(
        [[int(bit) for bit in format(index, f"0{n}b")] for index in range(2**n)],
        dtype=np.uint8,
    )
    phases = iqp_phase(theta, G, all_z, a)
    z_dot_gi = (all_z @ g_i) % 2
    sign_i = 1.0 - 2.0 * z_dot_gi.astype(np.float64)
    return float(-2.0 * a_dot_gi * np.mean(np.sin(phases) * sign_i))

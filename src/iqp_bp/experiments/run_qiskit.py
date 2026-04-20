"""Qiskit validation experiment runner.

Compares classical IQP expectations against:
  - Qiskit statevector (exact, noise-free)
  - Qiskit shot-based simulation
  - Qiskit Aer noise models

For every (family, n, n_shots, noise_params) cell in the sweep:
    1. Build the IQP circuit from a freshly-sampled hypergraph.
    2. Draw a small random batch of ``Z_a`` observables plus the all-ones
       Z observable.
    3. Compute ``<Z_a>`` four ways — classical closed-form, Qiskit
       statevector, Qiskit shots (noise-free), Qiskit shots (noisy).
    4. Persist absolute-error comparisons + the MMD^2 computed under each
       back-end so a later diff catches any divergence.

Output:
    results.jsonl — one row per (family, n, n_shots, noise_params).
    raw/*.json    — per-row payload with exp_classical / exp_sv / exp_shots
                    / exp_noisy plus shot-count histograms.
    qasm/*.qasm   — transpiled circuit exports (when ``save_qasm=True``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from iqp_bp.experiments.data_factory import make_dataset
from iqp_bp.iqp.expectation import iqp_expectation_exact
from iqp_bp.iqp.model import IQPModel
from iqp_bp.mmd.loss import mmd2_exact_small_n
from iqp_bp.rng import (
    STREAM_CIRCUIT,
    STREAM_DATA,
    STREAM_KERNEL,
    STREAM_QISKIT,
    STREAM_THETA,
    experiment_stream_bundle,
)

log = logging.getLogger(__name__)


def _as_list(value):
    """Coerce scalars to single-element lists so sweep iteration is uniform."""
    return value if isinstance(value, list) else [value]


def _broadcast_param_grid(params: dict[str, Any]) -> list[dict[str, Any]]:
    """Zip a dict of (possibly) list-valued params into a list of settings.

    The rule: every non-None parameter must be either a scalar (single
    value, used across the whole sweep) or a list of length ``max_len``.
    Any length-1 list is treated like a scalar. This lets a config say::

        error_rate: [0.001, 0.01, 0.1]
        readout_rate: 0.0

    and get three sweep cells paired correctly. A length mismatch
    between two multi-valued lists is a hard error.
    """
    # Filter out None values and normalize the rest to lists.
    active = {
        key: _as_list(value)
        for key, value in params.items()
        if value is not None
    }
    # No active parameters → a single empty settings dict (no-op sweep cell).
    if not active:
        return [{}]

    max_len = max(len(values) for values in active.values())
    # Shape sanity-check: either scalar (len 1) or matches the longest list.
    for key, values in active.items():
        if len(values) not in {1, max_len}:
            raise ValueError(
                f"Noise parameter {key!r} has length {len(values)}; expected 1 or {max_len}"
            )

    # Produce one dict per sweep index, broadcasting length-1 entries.
    grid: list[dict[str, Any]] = []
    for idx in range(max_len):
        entry = {}
        for key, values in active.items():
            entry[key] = values[0] if len(values) == 1 else values[idx]
        grid.append(entry)
    return grid


def _noise_param_sweep(noise_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a noise config block into the list of per-cell parameter dicts.

    Each noise model has its own native parameter set (gamma vs lam vs T1
    vs preset). The routing below picks the right set per model name.
    """
    model_name = str(noise_cfg.get("model", "combined"))

    # Simple depolarizing: one error rate per cell.
    if model_name == "depolarizing":
        return _broadcast_param_grid({"error_rate": noise_cfg.get("error_rate", [0.0])})
    # Readout: either asymmetric bit-flip probabilities or a symmetric
    # ``error_rate`` shorthand.
    if model_name == "readout":
        if "p0_given_1" in noise_cfg or "p1_given_0" in noise_cfg:
            return _broadcast_param_grid(
                {
                    "p0_given_1": noise_cfg.get("p0_given_1"),
                    "p1_given_0": noise_cfg.get("p1_given_0"),
                }
            )
        return _broadcast_param_grid({"error_rate": noise_cfg.get("error_rate", [0.0])})
    # Combined depolarizing + readout; readout_rate falls back to
    # error_rate/10 if omitted (see noise.py).
    if model_name == "combined":
        return _broadcast_param_grid(
            {
                "error_rate": noise_cfg.get("error_rate", [0.0]),
                "readout_rate": noise_cfg.get("readout_rate"),
            }
        )
    # Amplitude and phase damping take different keyword names.
    if model_name == "amplitude_damping":
        return _broadcast_param_grid(
            {
                "gamma_1q": noise_cfg.get("gamma_1q"),
                "gamma_2q": noise_cfg.get("gamma_2q"),
            }
        )
    if model_name == "phase_damping":
        return _broadcast_param_grid(
            {
                "lam_1q": noise_cfg.get("lam_1q"),
                "lam_2q": noise_cfg.get("lam_2q"),
            }
        )
    # Thermal relaxation: four times (T1, T2, 1q gate, 2q gate).
    if model_name == "thermal_relaxation":
        return _broadcast_param_grid(
            {
                "t1_ns": noise_cfg.get("t1_ns"),
                "t2_ns": noise_cfg.get("t2_ns"),
                "gate_time_1q_ns": noise_cfg.get("gate_time_1q_ns"),
                "gate_time_2q_ns": noise_cfg.get("gate_time_2q_ns"),
            }
        )
    # Backend preset: one parameter, the preset string.
    if model_name == "backend_preset":
        return _broadcast_param_grid({"preset": noise_cfg.get("preset")})

    raise ValueError(f"Unsupported qiskit.noise.model: {model_name!r}")


def _format_noise_params(params: dict[str, Any]) -> str:
    """Render a noise-params dict into a compact filename-safe suffix.

    Used in ``setting_id`` strings so each sweep cell's raw JSON /
    ``results.jsonl`` row is uniquely identifiable without regex.
    """
    if not params:
        return "none"

    # Sorted keys for deterministic output across dict orderings.
    parts: list[str] = []
    for key in sorted(params):
        value = params[key]
        # Floats: use general format, then rewrite "-" / "." so the string
        # is filename-safe (Unix doesn't like ".", Windows doesn't like some
        # shell chars; "m" / "p" are the project convention).
        if isinstance(value, float):
            formatted = f"{value:.6g}".replace("-", "m").replace(".", "p")
        else:
            formatted = str(value).replace("-", "_").replace(".", "p")
        parts.append(f"{key}{formatted}")
    return "_".join(parts)


def _legacy_error_rate(model_name: str, params: dict[str, Any]) -> float | None:
    """Return a legacy single-number ``error_rate`` column for old dashboards.

    Old plotting code expected a single scalar ``error_rate`` field on each
    row. For models where a single number no longer captures the picture
    (amplitude damping, thermal) this returns None; for readout we average
    the two bit-flip probs.
    """
    if "error_rate" in params:
        return float(params["error_rate"])
    if model_name == "readout":
        if "p0_given_1" in params and "p1_given_0" in params:
            return 0.5 * (float(params["p0_given_1"]) + float(params["p1_given_0"]))
        if "p0_given_1" in params:
            return float(params["p0_given_1"])
        if "p1_given_0" in params:
            return float(params["p1_given_0"])
    return None


def _extract_family_kwargs(circuit_cfg: dict[str, Any], family: str) -> dict[str, Any]:
    """Pull family-specific kwargs (dimension, p_edge, …) out of the config."""
    # Each circuit family expects a slightly different set of kwargs. We
    # return only the keys the family actually takes so ``make_hypergraph``
    # receives a tight kwargs dict.
    if family == "lattice":
        return {
            "dimension": circuit_cfg.get("lattice", {}).get("dimension", 1),
            "range_": circuit_cfg.get("lattice", {}).get("range", 1),
        }
    if family == "erdos_renyi":
        # Accept a bare float or a single-element list (common config shape
        # pattern across the project).
        default_p = circuit_cfg.get("erdos_renyi", {}).get("p_edge", 0.1)
        if isinstance(default_p, list):
            default_p = default_p[0]
        return {"p_edge": default_p}
    if family == "bounded_degree":
        return circuit_cfg.get("bounded_degree", {})
    if family == "dense":
        return {"expected_weight": circuit_cfg.get("dense", {}).get("expected_weight", 0.5)}
    if family == "community":
        return circuit_cfg.get("community", {})
    if family == "symmetric":
        return {"parity": circuit_cfg.get("symmetric", {}).get("parity", "even")}
    # Families with no extra kwargs (product_state, complete_graph) fall here.
    return {}


def _compute_m(n: int, formula: Any) -> int:
    """Resolve the config's ``n_generators`` into a concrete integer.

    Supports a bare int (fixed m) or the shorthand ``"n"`` (one generator
    per qubit). Anything else falls back to int(n) for safety.
    """
    if isinstance(formula, int):
        return formula
    if formula == "n":
        return n
    return int(n)


def _get_kernel_params(kernel_cfg: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Resolve the kernel config into (kernel_name, kwargs for sample_a/eval)."""
    kernel = str(kernel_cfg.get("type", "gaussian"))
    # Support `type: [gaussian]` as well as the canonical `type: gaussian`.
    if isinstance(kernel, list):
        kernel = str(kernel[0])

    # Gaussian/Laplacian: single bandwidth parameter.
    if kernel in {"gaussian", "laplacian"}:
        bandwidth = kernel_cfg.get("bandwidth", [1.0])
        sigma = float(bandwidth[0] if isinstance(bandwidth, list) else bandwidth)
        return kernel, {"sigma": sigma}
    # Multi-scale Gaussian: list of sigmas and optional per-scale weights.
    if kernel == "multi_scale_gaussian":
        msg = kernel_cfg.get("multi_scale_gaussian", {})
        return kernel, {
            "sigmas": list(msg.get("sigmas", [1.0])),
            "weights": msg.get("weights"),
        }
    # Polynomial kernel: degree + bias constant.
    if kernel == "polynomial":
        poly = kernel_cfg.get("polynomial", {})
        return kernel, {
            "degree": int(poly.get("degree", 2)),
            "constant": float(poly.get("constant", 1.0)),
        }
    # Linear kernel: no kwargs needed.
    if kernel == "linear":
        return kernel, {}
    return kernel, {}


def run(cfg: dict[str, Any]) -> None:
    """Entry point called by CLI.

    Orchestrates the whole Qiskit validation sweep from a single config
    dict. Signature matches the other experiment runners so the CLI can
    dispatch uniformly.
    """
    # --- Output wiring --------------------------------------------------
    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "results.jsonl"
    raw_dir = output_dir / "raw"
    qasm_dir = output_dir / "qasm"

    # --- Qiskit-side imports (delayed so CI without qiskit can import us)
    try:
        from iqp_bp.qiskit.circuit_builder import (
            build_iqp_circuit_measured,
            build_iqp_circuit_unmeasured,
            transpile_circuit,
        )
        from iqp_bp.qiskit.estimators import (
            batch_shot_expectations,
            batch_statevector_expectations,
        )
        from iqp_bp.qiskit.mmd import qiskit_mmd2
        from iqp_bp.qiskit.noise import get_noise_model
    except ImportError as e:
        raise ImportError("Qiskit not installed. Run: pip install qiskit qiskit-aer") from e

    # --- Pull sweep axes from config ------------------------------------
    base_seed = cfg["experiment"].get("seed", 0)
    circuit_cfg = cfg["circuit"]
    families = _as_list(circuit_cfg["family"])
    n_qubits_list = _as_list(circuit_cfg["n_qubits"])
    qiskit_cfg = cfg.get("qiskit", {})
    n_shots_list = _as_list(qiskit_cfg.get("n_shots", [10000]))
    # Hard cap on n so we don't accidentally launch a 2^24-dim statevector.
    max_n = int(qiskit_cfg.get("max_n", 20))
    # Separate cap for the exact MMD^2 reference (which costs 2^n).
    mmd_exact_max_n = int(qiskit_cfg.get("mmd_exact_max_n", 12))

    noise_cfg = qiskit_cfg.get("noise", {})
    noise_enabled = bool(noise_cfg.get("enabled", False))
    noise_model_name = str(noise_cfg.get("model", "combined"))
    # If noise is off, run exactly one "no-noise" cell to keep the sweep loop
    # structure uniform.
    noise_param_sweep = _noise_param_sweep(noise_cfg) if noise_enabled else [{}]

    transpile_cfg = qiskit_cfg.get("transpile", {})
    save_qasm = bool(transpile_cfg.get("save_qasm", False))
    opt_level = int(transpile_cfg.get("optimization_level", 1))
    # Fixed cap: we sample up to 16 random observables plus the all-Z one.
    n_observables = 16
    num_a_samples = int(cfg.get("estimation", {}).get("num_a_samples", 32))
    kernel_name, kernel_params = _get_kernel_params(cfg.get("kernel", {}))
    dataset_cfg = cfg.get("dataset", {})

    raw_dir.mkdir(parents=True, exist_ok=True)
    qasm_dir.mkdir(parents=True, exist_ok=True)

    # --- Main sweep loop ------------------------------------------------
    with open(out_path, "w", encoding="utf-8") as fout:
        for family in families:
            # Kwargs specific to this circuit family.
            family_kwargs = _extract_family_kwargs(circuit_cfg, family)
            for n in n_qubits_list:
                # Guardrail: respect the max_n limit set in config.
                if n > max_n:
                    log.warning("Skipping n=%s > max_n=%s", n, max_n)
                    continue

                # Derive named seeds for every stochastic component so we
                # can reproduce exactly. One bundle per (family, n) cell.
                streams = experiment_stream_bundle(base_seed, "run_qiskit", family, n)
                qiskit_seed = streams[STREAM_QISKIT]
                circuit_seed = streams[STREAM_CIRCUIT]
                theta_seed = streams[STREAM_THETA]
                kernel_seed = streams[STREAM_KERNEL]

                # Instantiate an IQP model for this cell. ``IQPModel.from_family``
                # handles the hypergraph sampling + provenance stamping.
                model = IQPModel.from_family(
                    family=family,
                    n=int(n),
                    m=_compute_m(int(n), circuit_cfg.get("n_generators", n)),
                    rng=np.random.default_rng(circuit_seed),
                    rng_seed=circuit_seed,
                    **family_kwargs,
                )
                # Fresh theta for this cell — uniform over [-pi, pi].
                model.theta = np.random.default_rng(theta_seed).uniform(
                    -np.pi, np.pi, size=model.m
                )
                # Dataset matching this n so the MMD^2 reference has
                # something to evaluate ``<Z_a>_p`` against.
                data, dataset_metadata = make_dataset(
                    dataset_cfg,
                    n=int(n),
                    seed=streams[STREAM_DATA],
                )

                # Build both circuit variants once per cell: unmeasured for
                # the statevector path, measured for the shot path.
                qc_unmeasured, spec = build_iqp_circuit_unmeasured(
                    model.G,
                    theta=model.theta,
                    parameterized=True,
                )
                qc_measured, _ = build_iqp_circuit_measured(
                    model.G,
                    theta=model.theta,
                    parameterized=True,
                )
                # Compile + capture compact metadata. We don't keep the
                # transpiled circuit object — just depth/size/basis.
                transpile_meta = transpile_circuit(
                    qc_unmeasured,
                    optimization_level=opt_level,
                )

                # Optional: dump QASM for archival (enables external audits).
                qasm_path = qasm_dir / f"{family}_n{n}.qasm"
                if save_qasm:
                    qasm_path.write_text(spec.qasm_str, encoding="utf-8")

                # Sample a batch of random observables. First one is the
                # all-Z observable — a canonical test case. The rest are
                # independent Bernoulli(1/2) masks.
                rng_kernel = np.random.default_rng(kernel_seed)
                k = min(n_observables, 2**int(n))
                observables = [
                    rng_kernel.integers(0, 2, size=int(n), dtype=np.uint8)
                    for _ in range(k)
                ]
                observables[0] = np.ones(int(n), dtype=np.uint8)

                # --- Baseline: classical closed-form ---------------------
                # Gold-standard <Z_a>_q computed from the IQP Fourier
                # expansion. Every other column is compared against this.
                exp_classical = np.array(
                    [
                        iqp_expectation_exact(model.theta, model.G, observable)
                        for observable in observables
                    ],
                    dtype=np.float64,
                )
                # Qiskit statevector (should agree with classical to
                # machine precision).
                exp_sv = batch_statevector_expectations(
                    qc_unmeasured,
                    observables,
                    model.theta,
                )
                abs_err_sv = float(np.mean(np.abs(exp_sv - exp_classical)))
                # MMD^2 closed-form (only tractable for small n — else None).
                mmd2_classical = (
                    float(
                        mmd2_exact_small_n(
                            model.theta,
                            model.G,
                            data,
                            kernel=kernel_name,
                            **kernel_params,
                        )
                    )
                    if int(n) <= mmd_exact_max_n
                    else None
                )

                # --- Shot / noise sweeps -------------------------------
                for n_shots in n_shots_list:
                    for noise_params in noise_param_sweep:
                        noise_suffix = _format_noise_params(noise_params)
                        # Unique identifier for this sweep cell —
                        # propagates into filenames and log lines.
                        setting_id = (
                            f"{family}_n{int(n)}_shots{int(n_shots)}_"
                            f"{noise_model_name}_{noise_suffix}"
                        )
                        # Instantiate the Aer noise model for this cell
                        # (or None if noise is globally off).
                        nm = (
                            get_noise_model(noise_model_name, **noise_params)
                            if noise_enabled
                            else None
                        )
                        legacy_error_rate = _legacy_error_rate(noise_model_name, noise_params)

                        # Shots, noise-free (cross-check against the
                        # classical baseline — expect agreement up to
                        # O(1/sqrt(n_shots))).
                        exp_shots, raw_counts = batch_shot_expectations(
                            qc_measured=qc_measured,
                            observables=observables,
                            theta=model.theta,
                            n_shots=int(n_shots),
                            seed=qiskit_seed,
                            noise_model=None,
                            return_counts=True,
                        )
                        abs_err_shots = float(np.mean(np.abs(exp_shots - exp_classical)))

                        # Shots, with the Aer noise model attached.
                        exp_noisy = batch_shot_expectations(
                            qc_measured=qc_measured,
                            observables=observables,
                            theta=model.theta,
                            n_shots=int(n_shots),
                            seed=qiskit_seed,
                            noise_model=nm,
                            return_counts=False,
                        )
                        abs_err_noisy = float(np.mean(np.abs(exp_noisy - exp_classical)))

                        # Three Qiskit-backed MMD^2 estimates, one per
                        # back-end variant. All share the same kernel
                        # seed so the observable draws are aligned.
                        mmd2_sv = qiskit_mmd2(
                            qc_unm=qc_unmeasured,
                            qc_meas=qc_measured,
                            theta=model.theta,
                            G=model.G,
                            data=data,
                            kernel=kernel_name,
                            num_a_samples=num_a_samples,
                            rng=np.random.default_rng(kernel_seed),
                            mode="statevector",
                            **kernel_params,
                        )
                        mmd2_shots = qiskit_mmd2(
                            qc_unm=qc_unmeasured,
                            qc_meas=qc_measured,
                            theta=model.theta,
                            G=model.G,
                            data=data,
                            kernel=kernel_name,
                            num_a_samples=num_a_samples,
                            rng=np.random.default_rng(kernel_seed),
                            mode="shots",
                            n_shots=int(n_shots),
                            seed=qiskit_seed,
                            noise_model=None,
                            **kernel_params,
                        )
                        mmd2_noisy = qiskit_mmd2(
                            qc_unm=qc_unmeasured,
                            qc_meas=qc_measured,
                            theta=model.theta,
                            G=model.G,
                            data=data,
                            kernel=kernel_name,
                            num_a_samples=num_a_samples,
                            rng=np.random.default_rng(kernel_seed),
                            mode="shots",
                            n_shots=int(n_shots),
                            seed=qiskit_seed,
                            noise_model=nm,
                            **kernel_params,
                        )

                        # Full per-cell payload. Lives in its own JSON
                        # file under raw/ and is also referenced by path
                        # from the main results.jsonl record below.
                        raw_path = raw_dir / f"{setting_id}_raw.json"
                        raw_artifact = {
                            "noise_model": noise_model_name,
                            "noise_params": noise_params,
                            "observables": [observable.tolist() for observable in observables],
                            "exp_classical": exp_classical.tolist(),
                            "exp_sv": exp_sv.tolist(),
                            "exp_shots": exp_shots.tolist(),
                            "exp_noisy": exp_noisy.tolist(),
                            "abs_errors": {
                                "sv": np.abs(exp_sv - exp_classical).tolist(),
                                "shots": np.abs(exp_shots - exp_classical).tolist(),
                                "noisy": np.abs(exp_noisy - exp_classical).tolist(),
                            },
                            "dataset_metadata": dataset_metadata,
                            "mmd2": {
                                "classical": mmd2_classical,
                                "sv": float(mmd2_sv),
                                "shots": float(mmd2_shots),
                                "noisy": float(mmd2_noisy),
                            },
                            "shot_counts": {key: int(value) for key, value in raw_counts.items()},
                        }
                        raw_path.write_text(
                            json.dumps(raw_artifact, indent=2),
                            encoding="utf-8",
                        )

                        # Compact summary record for results.jsonl. Keeps
                        # the absolute-error columns everyone dashboards on
                        # plus pointers to the raw file for drill-down.
                        record = {
                            "family": family,
                            "n": int(n),
                            "n_shots": int(n_shots),
                            "noise_model": noise_model_name,
                            "noise_params": noise_params,
                            "error_rate": legacy_error_rate,
                            "setting_id": setting_id,
                            "abs_err_sv": abs_err_sv,
                            "abs_err_shots": abs_err_shots,
                            "abs_err_noisy": abs_err_noisy,
                            "qasm_path": str(qasm_path) if save_qasm else None,
                            "raw_path": str(raw_path),
                            "dataset_metadata": dataset_metadata,
                            "transpile_depth": transpile_meta.depth,
                            "transpile_size": transpile_meta.size,
                            "transpile_basis_gates": transpile_meta.basis_gates,
                            "mmd2_classical": mmd2_classical,
                            "mmd2_sv": float(mmd2_sv),
                            "mmd2_shots": float(mmd2_shots),
                            "mmd2_noisy": float(mmd2_noisy),
                            # Absolute MMD^2 deltas only when the classical
                            # reference is tractable for this n.
                            "mmd2_abs_err_sv": (
                                None if mmd2_classical is None else abs(float(mmd2_sv) - mmd2_classical)
                            ),
                            "mmd2_abs_err_shots": (
                                None
                                if mmd2_classical is None
                                else abs(float(mmd2_shots) - mmd2_classical)
                            ),
                            "mmd2_abs_err_noisy": (
                                None
                                if mmd2_classical is None
                                else abs(float(mmd2_noisy) - mmd2_classical)
                            ),
                        }
                        fout.write(json.dumps(record) + "\n")
                        # Flush eagerly so a crashed sweep still leaves
                        # usable partial results on disk.
                        fout.flush()
                        log.info(
                            "Qiskit validation: %s - sv_err=%.4f, shots_err=%.4f, noisy_err=%.4f",
                            setting_id,
                            abs_err_sv,
                            abs_err_shots,
                            abs_err_noisy,
                        )

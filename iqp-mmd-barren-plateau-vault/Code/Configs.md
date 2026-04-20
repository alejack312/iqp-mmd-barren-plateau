---
title: Configs
tags:
  - config
  - experiments
---

# Experiment Configs

YAML config files under [`configs/`](../configs/) that drive the runners. All experiments inherit from `base.yaml` via [[Config|deep merge]].

## Files

```
configs/
├── base.yaml                           # project-wide defaults
├── hyperparameters.yaml                # iqp_mmd training hyperparameters (paper values)
├── schema.yaml                         # config schema (unused pending D1.2)
└── experiments/
    ├── scaling_v1.yaml                 # main scaling sweep
    ├── scaling_ac_smoke.yaml           # anti-concentration smoke test
    ├── validation.yaml                 # anti-concentration validation
    ├── validation_from_checkpoint.yaml # validation from saved checkpoint
    ├── qiskit_validation.yaml          # Qiskit cross-check
    └── forge_sprint.yaml               # Forge structural modeling
```

## base.yaml

The global default. Key sections:

```yaml
experiment:
  seed: 42
  output_dir: results/

circuit:
  family: product_state
  n_qubits: [4, 9, 16, 25, 36, 49, 64, ..., 1024]  # perfect squares for 2D lattice
  n_generators: "n"
  lattice: { dimension: 2, range: 1 }
  erdos_renyi: { p_edge: [2.0] }  # target avg degree c

kernel:
  type: gaussian
  bandwidth: [0.1, 0.5, 1.0, 2.0, 5.0]
  multi_scale_gaussian:
    sigmas: [0.5, 1.0, 2.0]
    weights: null  # uniform

init:
  scheme: uniform
  uniform: { low: -3.14159, high: 3.14159 }
  small_angle: { std: [0.01, 0.1, 0.3] }
  data_dependent: { dataset: ising }

dataset:
  type: product_bernoulli
  n_samples: 10000
  ising: { beta: 1.0, coupling_std: 1.0, topology: grid_2d }
  binary_mixture: { n_modes: 4, noise: 0.1 }

estimation:
  num_a_samples: 512
  num_z_samples: 1024
  num_seeds: 100
  use_jax: true

qiskit:
  backend: statevector
  n_shots: [1000, 10000, 100000]
  max_n: 20
  noise:
    enabled: false
    model: depolarizing
    error_rate: [0.001, 0.005, 0.01]

forge:
  max_n: 12
```

## Config Patterns

### Lists vs scalars

Any axis can be a list to create a sweep. The scaling runner detects lists via `_as_list` and expands them into Cartesian products in `resolve_scaling_settings`. See [[How a Scaling Run Works#The Explicit Grid]].

### Conditional sub-sweeps

Some sub-axes only apply to specific values:

- `bandwidth` only for `gaussian`/`laplacian`
- `er_p_edge` only for `erdos_renyi`
- `small_angle_std` only for `small_angle` init

### n_qubits constraint

For the 2D lattice family, `n` must be a perfect square. The default list in `base.yaml` is exactly `[L² for L in 2..32]`.

## Related

- [[Config]]
- [[CLI]]
- [[Scaling Runner]]
- [[How a Scaling Run Works]]

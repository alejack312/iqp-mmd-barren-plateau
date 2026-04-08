---
title: Initialization Schemes
tags:
  - theory
  - init
  - trainability
---

# Initialization Schemes

The three $\theta$ initialization schemes in the project. **All three are primary experimental axes**, not optional — the question of whether init can save or sink trainability is one of the central research questions.

## I1: Uniform

$$
\theta_i \sim U[-\pi, \pi]
$$

- **Worst-case / stress test** initialization
- The classic "random IQP" baseline used in BP theory
- Expected to **maximize** barren plateau severity
- Config: `init.scheme: uniform` with `init.uniform.low: -pi`, `init.uniform.high: pi`

## I2: Small-Angle

$$
\theta_i \sim \mathcal{N}(0, \sigma_\theta^2)
$$

- **Main trainability hypothesis** — small angles suppress BP via linearization
- Three sub-settings: $\sigma_\theta \in \{0.01, 0.1, 0.3\}$
- For $\sigma_\theta \to 0$: the cosine in [[IQP Expectation|$\langle Z_a\rangle$]] becomes approximately $1 - \Phi^2/2$, and gradients are dominated by first-order Taylor structure
- Config: `init.scheme: small_angle` with `init.small_angle.std: [0.01, 0.1, 0.3]`

> [!tip] Why small-angle might save trainability
> At small $\theta$, the model is effectively in the **linearized regime** around the maximally mixed initial state. Gradients are no longer dominated by high-order coherent cancellations, so the loss landscape has a well-behaved approximation neighborhood. If this hypothesis holds, small-angle init is a necessary condition for scalable IQP+MMD training.

See Mhiri et al. 2025 on warm-start guarantees ([[References#Mhiri 2025]]).

## I3: Data-Dependent

The current implementation is a lightweight warm start based on empirical parity expectations:

```python
init.scheme: data_dependent
```

```python
# In _make_theta:
scale = cfg.init.data_dependent.scale  # default 0.1
return scale * np.asarray(dataset_expectations_batch(data, G), dtype=np.float64)
```

Each $\theta_j$ is set proportional to the empirical expectation $\langle Z_{g_j}\rangle_p$ on the training data — so the init reflects what the data actually looks like through the lens of each generator's support.

> [!warning] Lightweight for now
> The current implementation is a **real and data-aware warm start**, but it is not the fully covariance-driven design the SMART spec envisions. A stronger data-dependent init is open work (see [[Implementation Choices#Why the init layer changed too]]).

## Effect on the Sweep

From [[How a Scaling Run Works]]:

- `uniform` — single value, no sub-sweep
- `small_angle` — sub-sweep over `std` values
- `data_dependent` — single value, requires `data` to be passed through

`_make_theta` dispatches on `init_scheme` and builds one theta per seed index.

## Related

- [[Gradient Variance]]
- [[Barren Plateaus]]
- [[Scaling Runner]]
- [[Research Questions]] — Q1, Q2, Outcome E
- [[References#Mhiri 2025]]

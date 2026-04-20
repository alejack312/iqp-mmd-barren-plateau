---
title: MMD Loss Module
tags:
  - code
  - iqp_bp
  - mmd
---

# `iqp_bp.mmd.loss`

The Monte Carlo MMD² estimator. One function, two nested loops.

**File:** [`src/iqp_bp/mmd/loss.py`](../src/iqp_bp/mmd/loss.py)

## Function

```python
def mmd2(
    theta, G, data,
    kernel="gaussian",
    num_a_samples=512,
    num_z_samples=1024,
    rng=None,
    **kernel_params,
) -> float:
```

## What It Does

1. **Sample Z-words** — `sample_a(kernel, n, num_a_samples)` draws $a_1, \ldots, a_B$ from $P_k$ via [[Kernel Module]].
2. **Data-side expectations** — `dataset_expectations_batch(data, a_samples)` computes $\langle Z_{a_i}\rangle_p$ for all $i$ in one batched matmul (see [[Mixture Module]]).
3. **Model-side expectations** — loops over $a_i$ and calls [[IQP Expectation|`iqp_expectation(theta, G, a_i, num_z_samples)`]].
4. **Average squared residuals** — $\frac{1}{B}\sum_i (\text{exp}_p[i] - \text{exp}_q[i])^2$.

Returns a scalar MMD² estimate.

## Cost

- Outer loop: $B = $ `num_a_samples` iterations
- Inner loop per iteration: IQP expectation at $B_z = $ `num_z_samples`, costing $O(B_z \cdot m \cdot n)$
- Total: $O(B \cdot B_z \cdot m \cdot n)$
- Defaults: $B = 512$, $B_z = 1024$

## Open Diagnostics

From the TODO comments in this file:

- **P5** — expose per-observable contributions so the pipeline can inspect one MMD² estimate end-to-end
- **D2.1** — add an exact small-$n$ MMD² path to cross-check each kernel against brute force for $n \le 10$

## Caller Graph

- Called by [[Gradients Module|`grad_mmd2_finite_diff`]]
- Called by experiment scripts / notebooks needing a direct loss evaluation
- Not directly called by the scaling runner (which uses `estimate_gradient_variance` instead)

## Related

- [[MMD Loss]] — the theory
- [[Kernel Module]] — the sampler dispatch
- [[Mixture Module]] — the data-side batched expectation
- [[IQP Expectation]] — the model-side inner loop
- [[Gradients Module]]

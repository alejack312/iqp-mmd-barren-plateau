---
title: Product Bernoulli Dataset
tags:
  - dataset
  - baseline
---

# D1 — Product Bernoulli

The simplest possible synthetic dataset. Each bit is an independent fair coin:

$$
p(x) = \prod_{i=1}^{n} \frac{1}{2}
$$

## Data-Side Expectations

For any Z-word $a$ with $|a| \ge 1$:

$$
\langle Z_a\rangle_p = \mathbb{E}_x[(-1)^{a \cdot x}] = \prod_{i: a_i = 1} \mathbb{E}[(-1)^{x_i}] = 0
$$

because each single-bit expectation is $\mathbb{E}[(-1)^{\text{Bern}(0.5)}] = 0$.

> [!info] What this means for the MMD²
> Under D1, $\langle Z_a\rangle_p = 0$ for every $|a| \ge 1$, so:
> $$\mathrm{MMD}^2(p, q_\theta) \to \sum_a w_k(a) \langle Z_a\rangle_{q_\theta}^2$$
> The loss becomes a pure function of the model's own Z-word expectations. **Any gradient structure you see on D1 is coming from the circuit + kernel, not the data.**

This is why D1 is the primary target for the scaling study: it cleanly isolates the (family, kernel, init) axes from data structure.

## In the Code

From [[Data Factory]]:

```python
if dataset_type == "product_bernoulli":
    data = rng.integers(0, 2, size=(n_samples, n), dtype=np.uint8)
    return data, {
        "type": dataset_type,
        "n_samples": n_samples,
        "seed": int(seed),
    }
```

## Config

```yaml
dataset:
  type: product_bernoulli
  n_samples: 10000
```

No additional parameters.

## Related

- [[Datasets]]
- [[Data Factory]]
- [[Ising Dataset]]
- [[Binary Mixture Dataset]]

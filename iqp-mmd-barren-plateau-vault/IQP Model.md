---
title: IQP Model
tags:
  - code
  - iqp_bp
  - iqp
---

# `iqp_bp.iqp.model.IQPModel`

Wrapper around a generator matrix and a parameter vector. The canonical way to represent an IQP circuit in memory.

**File:** [`src/iqp_bp/iqp/model.py`](../src/iqp_bp/iqp/model.py)

## Class

```python
class IQPModel:
    G: np.ndarray      # shape (m, n), uint8
    theta: np.ndarray  # shape (m,), float64
    n: int             # number of qubits
    m: int             # number of generators
```

## Construction

### From `G` directly

```python
model = IQPModel(G, theta=None)
```

If `theta` is None, defaults to zeros.

### From a family name

```python
model = IQPModel.from_family("lattice", n=16, m=24, rng=rng, dimension=2, range_=1)
```

Calls [[Hypergraph Families|`make_hypergraph`]] and wraps the result.

## Methods

### `expectation(a, num_z_samples=1024, rng=None)`

Monte Carlo estimate of $\langle Z_a\rangle_{q_\theta}$. Returns `(mean, stderr)` tuple. Wraps [[IQP Expectation|`iqp_expectation`]].

### `expectation_exact(a)`

Exact brute-force computation for small $n$ ($n \le 20$). Wraps `iqp_expectation_exact`.

### `probability_vector_exact(max_qubits=20)`

Returns the **full exact probability vector** $p(x)$ of length $2^n$. This is the entry point for the [[Anti-Concentration]] checker.

Procedure:

1. Enumerate all $z \in \{0,1\}^n$ via `_basis_bits_exact(n)`
2. Compute diagonal phase $\phi(z) = \sum_j \theta_j (-1)^{z \cdot g_j}$
3. Form $d(z) = e^{-i\phi(z)}$
4. Apply [[Walsh-Hadamard Transform|FWHT]] via `_fwht_inplace`
5. Divide by $2^n$ and square to get Born probabilities
6. Zero out tiny floating-point roundoff
7. Renormalize

### `output_probabilities_exact(max_qubits=20)` — alias

Identical to `probability_vector_exact`. Both names are kept to match planning docs.

## Helper Statics

### `_basis_bits_exact(n)`

Enumerate $\{0,1\}^n$ in integer index order (MSB-first). Returns shape `(2^n, n)` uint8.

### `_fwht_inplace(values)`

In-place unnormalized Walsh-Hadamard butterfly. See [[Walsh-Hadamard Transform]].

## Open TODOs

- **P4** — preserve family and generation metadata on the model so experiment records and Qiskit exports can recover the exact circuit provenance. Currently `from_family` takes the family name and kwargs but doesn't save them on `self`.

## Callers

- [[Validation Runner]] — `evaluate_anti_concentration_from_model`
- [[Scaling Runner]] — wraps `G` and the first theta seed for the anti-concentration block
- [[Tests]] — `test_iqp_probability_vector_small_n.py` and others

## Related

- [[IQP Circuits]]
- [[IQP Expectation]]
- [[Generator Matrix]]
- [[Walsh-Hadamard Transform]]
- [[Anti-Concentration]]

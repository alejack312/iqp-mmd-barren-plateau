---
title: Parity Algebra
tags:
  - theory
  - math
  - glossary
---

# Parity Algebra

The mod-2 binary arithmetic that drives every formula in the project. "Parity algebra" is shorthand for the fact that essentially every expensive operation reduces to a mod-2 matmul on binary vectors.

## The Definition

**Parity** of a bitstring $x$ against a mask $a$:

$$
(a \cdot x \bmod 2) \in \{0, 1\}
$$

which is $0$ if the number of selected 1-bits is even and $1$ if odd.

The associated **parity sign**:

$$
(-1)^{a \cdot x} \in \{+1, -1\}
$$

which is $+1$ for even parity and $-1$ for odd parity.

## Why Every IQP Formula Uses It

Three places in the pipeline reduce to the same operation:

### 1. Pauli Z observables

$$
\langle Z_a \rangle = \mathbb{E}_x\!\left[(-1)^{a \cdot x}\right]
$$

This is **exactly** a parity average. Z-words on computational-basis samples *are* parity checks.

### 2. IQP phase computation

$$
\Phi(\theta, z, a) = 2 \sum_j \theta_j (a \cdot g_j \bmod 2)(-1)^{z \cdot g_j}
$$

Both $(a \cdot g_j \bmod 2)$ and $(-1)^{z \cdot g_j}$ are parities — the whole phase formula is a parity-weighted linear combination.

### 3. Data-side expectations

```python
parities = (data @ a_batch.T) % 2       # (N, B)
signs = 1.0 - 2.0 * parities.astype(np.float64)
return signs.mean(axis=0)               # (B,)
```

Batched parity sign computation over the dataset.

## Why This Is Efficient

Because parity questions reduce to:

```python
(G @ a) % 2  # shape (m,)
(G @ z) % 2  # shape (m,)
```

both of which are just cheap matrix-vector products modulo 2. No $2^n$ enumeration, no state vectors, no shot sampling. The entire classical IQP pipeline is basically NumPy linear algebra on `uint8` arrays.

## Sign Computation Gotcha

> [!warning] Cast to float before signing
> On `uint8` arrays, `1 - 2 * parities` **underflows**: `uint8(0) - uint8(2)` wraps around to 254 instead of yielding -1. Always cast to `float64` first:
> ```python
> signs = 1.0 - 2.0 * parities.astype(np.float64)
> ```
>
> This is the exact bug that surfaced during the Gaussian lock work. See [[Mixture Module#The Parity Sign Bug]].

## Why "Parity" Is the Common Language

From [[Glossary#Parity]]:

> "Parity is the common language connecting the dataset, the Fourier decomposition, the kernel, and the IQP circuit observables."

- **Dataset** — $\langle Z_a\rangle_p$ is a parity average
- **Fourier decomposition** — Walsh characters are parity signs
- **Kernel** — spectral weights live over the parity-indexed basis
- **IQP circuit** — the phase formula is built from parity overlaps

## Related

- [[Generator Matrix]]
- [[Z-Word]]
- [[IQP Expectation]]
- [[Mixture Module]]
- [[Kernel Spectral Decomposition]]

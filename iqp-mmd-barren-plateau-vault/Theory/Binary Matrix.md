---
title: Binary Matrix
tags:
  - glossary
  - math
---

# Binary Matrix

A **binary matrix** is a matrix whose entries are only `0` or `1`. In this project, the [[Generator Matrix|generator matrix G]] is binary because each entry answers a yes/no question:

- `1` — qubit $i$ is included in generator $j$
- `0` — qubit $i$ is not included in generator $j$

## Why Binary

The point of using a binary matrix is that **overlap questions become cheap matrix arithmetic modulo 2**:

```python
(G @ a) % 2  # which generators have odd overlap with Z-word a
(G @ z) % 2  # which generators have odd overlap with random bitstring z
```

These two products drive the entire IQP classical pipeline — see [[Parity Algebra]].

## In `uint8`

The code stores `G` as `np.ndarray` with `dtype=uint8`:

- Tight memory — one byte per entry
- Cheap to multiply
- No floating-point noise

> [!warning] Cast to float before signing
> `1 - 2 * x` on `uint8` **underflows** to 254 instead of giving -1. Always cast to `float64` first when computing signs: `1.0 - 2.0 * x.astype(np.float64)`. See [[Mixture Module#The Parity Sign Bug]].

## Related

- [[Generator Matrix]]
- [[Parity Algebra]]
- [[Mixture Module]]
- [[Glossary]]

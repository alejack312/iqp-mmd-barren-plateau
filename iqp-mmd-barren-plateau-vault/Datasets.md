---
title: Datasets
tags:
  - datasets
  - theory
---

# Datasets

Three target distributions $p(x)$ are planned for the project's scaling experiments. All are over $\{0,1\}^n$ and feed directly into the data-side of [[MMD Loss]] via $\langle Z_a\rangle_p$.

## D1: Product Bernoulli (Primary)

See [[Product Bernoulli Dataset]].

$$
p(x) = \prod_i \text{Bernoulli}(0.5)
$$

- $\langle Z_a\rangle_p = 0$ for all $|a| \ge 1$ (all higher-order correlations vanish)
- The baseline — any data-side signal is structure-free
- Purpose: **isolate circuit/kernel effects from data structure**
- Status: **primary** scaling target

## D2: Ising-Like Synthetic

See [[Ising Dataset]].

$$
p(x) \propto \exp\!\left(-\beta \sum_{\langle i,j\rangle} J_{ij} x_i x_j\right)
$$

- Couplings $J_{ij} \sim \mathcal{N}(0, 1/n)$
- Coupling graph: 2D grid (or sparse ER)
- Purpose: **structured pairwise correlations** — how do data correlations interact with gradient landscape?
- Status: **secondary** for scaling comparison in later weeks

## D3: Structured Binary Mixture

See [[Binary Mixture Dataset]].

$$
p(x) = \frac{1}{K}\sum_{k=1}^K \mathcal{N}(\mu_k, \epsilon^2 I)\text{ projected to }\{0,1\}^n
$$

- $K = 4$ modes
- Random binary centers $\mu_k$
- Low noise $\epsilon = 0.1$
- Purpose: **multi-modal target**, closer to real data
- Status: hardest for MMD² to capture with shallow IQP; **comparison** target

## Real Datasets (Out of Scope for `iqp_bp`)

The sibling [[iqp_mmd Package]] supports real datasets for training:

- **MNIST (binarized)** — 784 qubits
- **D-Wave samples** — 484 qubits, quantum annealer outputs
- **Genomic SNP** — up to 10K qubits, INRIA 1000 Genomes

These are out of the `iqp_bp` scope — the barren plateau study only needs synthetic targets because the gradient-variance metric doesn't depend on the specific data distribution, only its parity statistics.

## Code Path

All three dataset types are built on the fly by [[Data Factory|`iqp_bp.experiments.data_factory.make_dataset`]].

## Related

- [[Data Factory]]
- [[MMD Loss]]
- [[Mixture Module]]
- [[Scope Lock]] — Section 5

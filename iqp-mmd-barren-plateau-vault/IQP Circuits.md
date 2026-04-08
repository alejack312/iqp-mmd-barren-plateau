---
title: IQP Circuits
tags:
  - theory
  - iqp
  - circuit
---

# IQP Circuits

**IQP** = **I**nstantaneous **Q**uantum **P**olynomial-time. IQP circuits are the central object of this study.

## Definition

A parameterized IQP circuit on $n$ qubits with $m$ generators is:

$$
|\psi(\theta)\rangle = H^{\otimes n} \cdot \exp\!\left(i \sum_{j=1}^m \theta_j X^{g_j}\right) \cdot H^{\otimes n} |0\rangle^{\otimes n}
$$

where:

- $g_j \in \{0,1\}^n$ is the generator bitmask (row $j$ of the [[Generator Matrix|generator matrix G]])
- $X^{g_j} \equiv \bigotimes_{i : g_{j,i}=1} X_i$
- $\theta \in \mathbb{R}^m$ are the trainable parameters
- Measurement is in the computational basis, giving $x \in \{0,1\}^n$

The output distribution is $q_\theta(x) = |\langle x | \psi(\theta)\rangle|^2$.

## Why the Hadamard Sandwich

The final Hadamard layer maps the X-basis diagonal block $\exp(i \sum_j \theta_j X^{g_j})$ into a Z-basis measurement. Equivalently, by basis change:

$$
H^{\otimes n} \exp\!\left(i \sum_j \theta_j X^{g_j}\right) H^{\otimes n} = \exp\!\left(i \sum_j \theta_j Z^{g_j}\right)
$$

So the diagonal block in the Z-basis is a phase $d(z) = e^{-i \phi(z)}$ with

$$
\phi(z) = \sum_j \theta_j (-1)^{z \cdot g_j}
$$

and the circuit is exactly $H^{\otimes n} \text{diag}(d(z)) H^{\otimes n}$. The [[Walsh-Hadamard Transform]] then converts this into amplitudes.

## Why IQP Are Interesting

> [!info] Complexity-theoretic motivation
> Sampling from IQP circuit output distributions is **classically hard** under plausible complexity assumptions (Bremner–Jozsa–Shepherd). Efficient classical sampling from arbitrary IQP circuits would collapse the polynomial hierarchy.

Yet, paradoxically, certain **expectation values** required for MMD training can be computed *classically* and efficiently. This is the "train on classical, deploy on quantum" split that motivates this study. See [[IQP Classical Sampling]].

## In the Codebase

- **Model wrapper**: [[IQP Model|`iqp_bp.iqp.model.IQPModel`]]
- **Monte Carlo estimator**: [[IQP Expectation|`iqp_bp.iqp.expectation.iqp_expectation`]]
- **Exact path**: `iqp_expectation_exact` and `probability_vector_exact` (for small $n$ only)
- **Qiskit builder**: [[Qiskit Circuit Builder|`iqp_bp.qiskit.circuit_builder.build_iqp_circuit`]]

## Related

- [[Generator Matrix]]
- [[Parity Algebra]]
- [[IQP Expectation]]
- [[Families MOC]]
- [[Walsh-Hadamard Transform]]

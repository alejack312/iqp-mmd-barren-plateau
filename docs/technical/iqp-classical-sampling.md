# How the paper samples: the classical description of IQP output distributions

The paper *"Train on classical, deploy on quantum"* does something clever. It trains a quantum circuit model entirely on a regular computer — no quantum hardware needed — and only at the end runs the circuit to draw samples. This page explains exactly how that works.

---

## The big picture first

Imagine you have a vending machine. The vending machine can dispense candy bars according to some pattern you've programmed in. You want the pattern to match a target distribution — say, the pattern of mutations in a genome, or spin configurations in a physics model.

The normal way to train this machine would be: generate some candy bars, compare them to what you wanted, adjust. The problem is that generating candy bars from a quantum vending machine is expensive — or at least, classically hard to simulate.

The paper's trick: for this particular type of quantum circuit (IQP circuits), you don't need to actually generate samples during training. There's a shortcut formula that tells you everything you need to compute the loss — using only cheap classical arithmetic.

---

## What's an IQP circuit?

IQP stands for **Instantaneous Quantum Polynomial-time**. It's a specific type of quantum circuit with two features:

1. All qubits start in the `|+⟩` state (a balanced superposition of 0 and 1).
2. The circuit applies a sequence of gates, each of the form:

$$\exp(i \, \theta_j \, X^{g_j})$$

where $\theta_j$ is a learnable parameter and $X^{g_j}$ means "apply a Pauli X gate to each qubit in the subset $g_j$."

After all the gates, you measure each qubit and get a bitstring — a sequence of 0s and 1s. The circuit defines a probability distribution over all such bitstrings. Training means adjusting the $\theta_j$ values so that distribution matches your data.

---

## The problem: you can't afford to simulate the output

For $n$ qubits, the output distribution has $2^n$ possible values. With even 50 qubits, that's over a quadrillion entries. You can't store that table, let alone compute it.

But you don't actually need the full distribution during training. You only need a number called the **MMD loss** — a measure of how different two distributions are.

---

## The MMD loss and Z-word operators

The MMD loss is built from quantities called **expectation values of Z-word operators**.

A Z-word is just a product of Pauli Z gates on a chosen subset of qubits. For example, if $a = [1, 0, 1]$ for a 3-qubit system, the Z-word $Z_a$ means "measure $Z$ on qubit 0 and qubit 2, ignore qubit 1." The eigenvalue of this operator on a bitstring $x$ is simply $(-1)^{x_0 + x_2}$ — it's +1 or –1 depending on the parity of the selected bits.

The expectation value $\langle Z_a \rangle$ under a distribution $q$ is:

$$\langle Z_a \rangle_q = \sum_x q(x) \cdot (-1)^{a \cdot x}$$

This is the mean parity of the bits selected by $a$ across all samples.

The MMD loss between the data distribution $p$ and the model distribution $q_\theta$ turns out to be a weighted sum of squared differences of these expectation values:

$$\text{MMD}^2(p, q_\theta) = \mathbb{E}_{a \sim P_\sigma} \left[ \left( \langle Z_a \rangle_p - \langle Z_a \rangle_{q_\theta} \right)^2 \right]$$

where $P_\sigma$ is a probability distribution over Z-word operators derived from the Gaussian kernel with bandwidth $\sigma$.

This is still a potentially expensive sum, but it can be estimated by Monte Carlo: sample a bunch of random $a$'s from $P_\sigma$, compute the squared difference for each, and average.

The question is: how do you compute $\langle Z_a \rangle_{q_\theta}$ without simulating the quantum circuit?

---

## The key formula: classical expectation values for IQP circuits

This is the heart of the paper. For IQP circuits, there's a closed-form identity (from Van den Nest's theorem) that says:

$$\langle Z_a \rangle_{q_\theta} = \mathbb{E}_{z \sim \text{Uniform}(\{0,1\}^n)} \left[ \cos\!\left(\Phi(\theta, z, a)\right) \right]$$

In words: **the expectation value of any Z-word operator under an IQP circuit is the average cosine of a certain phase function**, where the average is over uniformly random classical bitstrings $z$.

The phase function is:

$$\Phi(\theta, z, a) = \sum_j \theta_j \cdot (-1)^{g_j \cdot z} \cdot \left(1 - (-1)^{g_j \cdot a}\right)$$

Every term here is classical arithmetic:
- $g_j \cdot z$ is a dot product mod 2 (count how many bits in gate $j$'s support overlap with $z$, and check parity).
- $(-1)^{g_j \cdot z}$ is either $+1$ or $-1$.
- Same for $(-1)^{g_j \cdot a}$.
- $\theta_j$ is just a number.

No quantum simulation. No exponentially large tables.

---

## How this is estimated in practice

To compute $\langle Z_a \rangle_{q_\theta}$ for a given $a$ and $\theta$:

1. Sample $s$ random bitstrings $z_1, z_2, \ldots, z_s$ uniformly from $\{0,1\}^n$.
2. For each $z_i$, compute the phase $\Phi(\theta, z_i, a)$ using the formula above.
3. Return:

$$\widehat{\langle Z_a \rangle} = \frac{1}{s} \sum_{i=1}^{s} \cos\!\left(\Phi(\theta, z_i, a)\right)$$

This estimate improves as you increase $s$ (the number of samples). In the code, $s$ is called `n_samples` and the number of $a$'s sampled for the MMD estimate is `n_ops`.

So the full training loop uses two levels of Monte Carlo:
- Sample $a$'s from $P_\sigma$ to estimate the outer MMD expectation.
- Sample $z$'s uniformly to estimate each $\langle Z_a \rangle_{q_\theta}$.

Neither step requires touching a quantum computer.

---

## What about the data side?

The data side is simpler. To estimate $\langle Z_a \rangle_p$ from a dataset of $m$ binary samples $x_1, \ldots, x_m$:

$$\widehat{\langle Z_a \rangle}_p = \frac{1}{m} \sum_{i=1}^{m} (-1)^{a \cdot x_i}$$

Just compute the average parity of the selected bits across your training samples.

---

## Putting it together: training step by step

Each training iteration does this:

1. Sample a batch of operators $a_1, \ldots, a_K$ from $P_\sigma$.
2. For each $a_k$:
   - Estimate $\langle Z_{a_k} \rangle_p$ from the training data.
   - Estimate $\langle Z_{a_k} \rangle_{q_\theta}$ using the classical cosine formula.
3. Compute the MMD estimate: $\frac{1}{K} \sum_k \left(\langle Z_{a_k} \rangle_p - \langle Z_{a_k} \rangle_{q_\theta}\right)^2$.
4. Differentiate with respect to $\theta$ (using JAX's automatic differentiation through the cosine formula).
5. Update $\theta$ with the gradient.

This runs entirely on CPU/GPU. No qubits involved.

---

## And then: deployment uses actual quantum sampling

After training, the learned parameters $\theta$ are loaded into an actual IQP circuit (or a classical circuit simulator). The circuit is run, and each run produces one bitstring sample from $q_\theta$.

In the code, this is:

```python
iqp_circuit = iqp.IqpSimulator(gates=gates, **model_config)
samples = iqp_circuit.sample(params, num_samples=5000)
```

This step can run on quantum hardware — and is where the potential quantum advantage lies. The distribution $q_\theta$ is believed to be classically hard to sample from (under standard complexity assumptions). But computing the expectation values needed for training is not hard, which is why training works classically.

---

## Why does this matter?

This split — **train classically, sample quantumly** — is only possible because IQP circuits have this special structure. The cosine formula is not a general quantum result; it works specifically because:

- IQP gates all commute with each other.
- The output distribution has a Fourier structure that makes Z-word expectation values analytically tractable.
- The MMD loss happens to decompose into exactly these Z-word expectation values.

Other quantum circuit families (like random deep circuits) don't have a similar shortcut. There, computing even a single expectation value requires full quantum simulation — which is why barren plateaus and sampling costs are so much harder to avoid.

---

## Summary

| Step | What happens | Classical or quantum? |
|------|-------------|----------------------|
| Compute $\langle Z_a \rangle_{q_\theta}$ | Average $\cos(\Phi(\theta, z, a))$ over random $z$ | Classical |
| Compute $\langle Z_a \rangle_p$ | Average parity over training data | Classical |
| Compute MMD loss | Average squared differences over random $a$'s | Classical |
| Update parameters | Gradient step via autodiff | Classical |
| Generate samples | Run the IQP circuit | Quantum (or simulator) |

The key formula the paper relies on is:

$$\boxed{\langle Z_a \rangle_{q_\theta} = \mathbb{E}_{z \sim \text{Uniform}} \left[ \cos\!\left( \sum_j \theta_j \cdot (-1)^{g_j \cdot z} \cdot \left(1 - (-1)^{g_j \cdot a}\right) \right) \right]}$$

This is what makes large-scale classical training of IQP generative models possible.

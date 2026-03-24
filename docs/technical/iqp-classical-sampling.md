# How the paper samples: the classical description of IQP output distributions

The paper *"Train on classical, deploy on quantum"* trains a quantum circuit model entirely on a regular computer and only runs the circuit at the end to draw samples. This page explains exactly how that works, grounded in the upstream source at [`XanaduAI/scaling-gqml@b907bb1`](https://github.com/XanaduAI/scaling-gqml/tree/b907bb119c45ee85c87f1eb91867f4a7d281f5be) and the `iqpopt` library it depends on.

---

## The big picture

The model is a quantum circuit that outputs binary strings — sequences of 0s and 1s. Training means adjusting the circuit's parameters so its output distribution matches a dataset. The problem: simulating quantum circuits is expensive. For 50 qubits, the output distribution has over a quadrillion possible values.

The trick the paper exploits: for IQP circuits specifically, you never need the full output distribution during training. There's a closed-form formula that lets you compute the training loss using nothing but regular arithmetic on a normal computer.

---

## What's an IQP circuit?

IQP stands for Instantaneous Quantum Polynomial-time. Two features define it:

1. All qubits start in the `|+⟩` state (an equal superposition of 0 and 1).
2. The circuit applies $m$ gates — Hadamards first, then a sequence of `MultiRZ(2θ_j, wires=gen_j)` gates, then Hadamards again.

Each `MultiRZ(2θ_j, wires)` applies $\exp(-i\theta_j Z^{g_j})$ where $g_j$ is the set of qubits the gate acts on. After all gates, you measure each qubit and get a bitstring. Training adjusts the $\theta_j$ values so that distribution matches your data.

In the code (`iqp_optimizer.py:iqp_circuit`):

```python
for i in range(self.n_qubits):
    qml.Hadamard(i)
for par, gate in zip(params, self.gates):
    for gen in gate:
        qml.MultiRZ(2*par, wires=gen)
for i in range(self.n_qubits):
    qml.Hadamard(i)
```

The circuit structure is encoded in `self.generators` — a matrix of shape `(num_generators, n_qubits)` where each row is a binary vector specifying which qubits one gate touches.

---

## Two model variants: `bitflip=False` and `bitflip=True`

The paper evaluates two variants, controlled by the `bitflip` flag in `IqpSimulator`:

- **`bitflip=False`** — the actual quantum IQP circuit. Sampling requires running the circuit on a quantum state vector simulator or real hardware. Training uses a Monte Carlo cosine formula (described below).
- **`bitflip=True`** — a simpler classical stochastic model that mimics IQP structure. Sampling is just a random walk: each gate flips its target bits with probability $\sin^2(\theta_j)$. Training uses an exact product-of-cosines formula (no Monte Carlo needed).

The upstream training script (`paper/training/train_iqp.py`) has `bitflip = True` as the default for benchmarking, but the paper's main results use both. The discussion of classical training efficiency applies to both variants — neither requires quantum simulation during training.

---

## The training loss: MMD

The training loss is the Maximum Mean Discrepancy (MMD) between the data distribution $p$ and the model distribution $q_\theta$:

$$\text{MMD}^2(p, q_\theta) = \mathbb{E}_{a \sim P_\sigma} \left[ \left( \langle Z_a \rangle_p - \langle Z_a \rangle_{q_\theta} \right)^2 \right]$$

Here $Z_a$ is a "Z-word" operator — it measures the parity of the bits selected by the binary mask $a$:

$$\langle Z_a \rangle = \sum_x q(x) \cdot (-1)^{a \cdot x}$$

$P_\sigma$ is a distribution over Z-word masks derived from the Gaussian kernel with bandwidth $\sigma$. In `mmd_loss_iqp` (`iqpopt/gen_qml/iqp_methods.py`):

```python
p_MMD = (1 - jnp.exp(-1/2/sigma**2)) / 2
visible_ops = jnp.array(jax.random.binomial(subkey, 1, p_MMD, shape=(n_ops, len(wires))), dtype='float64')
```

Each row of `visible_ops` is one random mask $a$, where each bit is 1 with probability $p_\text{MMD}$. This is sampling from $P_\sigma$.

The loss is estimated by averaging the squared differences over these $n_\text{ops}$ masks. The data side (`tr_train`) is just average parities over training samples:

```python
tr_train = jnp.mean(1 - 2 * ((ground_truth @ visible_ops.T) % 2), axis=0)
```

The model side (`tr_iqp`) is the interesting part.

---

## The key formula: classical expectation values for IQP circuits

For the standard IQP circuit (`bitflip=False`), there's a closed-form result (from Van den Nest's theorem) that says:

$$\langle Z_a \rangle_{q_\theta} = \mathbb{E}_{z \sim \text{Uniform}(\{0,1\}^n)} \left[ \cos\!\left(\Phi(\theta, z, a)\right) \right]$$

The expectation value of any Z-word under an IQP circuit is the average cosine of a phase, where the average is over uniformly random classical bitstrings $z$.

The phase is (from `iqp_optimizer.py:op_expval_batch`, non-bitflip path):

$$\Phi(\theta, z, a) = \sum_j \underbrace{(a \cdot g_j \bmod 2)}_{\text{gate selector}} \cdot \underbrace{2\theta_j}_{\text{weight}} \cdot \underbrace{(-1)^{z \cdot g_j}}_{\text{parity sign}}$$

In code:

```python
ops_gen        = (ops @ self.generators.T) % 2           # (a · g_j) mod 2: 0 or 1
samples_gates  = 1 - 2 * ((samples @ self.generators.T) % 2)  # (-1)^(z · g_j): ±1
par_ops_gates  = 2 * effective_params * ops_gen           # 2θ_j · (a·g_j mod 2)
expvals        = jnp.cos(par_ops_gates @ samples_gates.T) # cos(Φ), shape (n_ops, n_samples)
```

`jnp.mean(expvals, axis=-1)` gives the Monte Carlo estimate of $\langle Z_a \rangle_{q_\theta}$ for each mask $a$.

Every term is plain integer arithmetic — no quantum state, no exponentially large tables.

### Why the code's form is simpler to compute

The phase can also be written as $\sum_j \theta_j \cdot (-1)^{g_j \cdot z} \cdot (1 - (-1)^{g_j \cdot a})$, which is equivalent since $(1 - (-1)^k) = 2 \cdot (k \bmod 2)$ for integer $k$.

The code uses the `mod 2` form because `(G @ a) % 2` computes the entire gate-selector vector in one matrix multiply, producing a plain 0/1 array directly. Multiplying by 0 or 1 is cheaper and differentiable without branching — compared to raising −1 to an integer power, which branches on parity.

### The `bitflip=True` variant is different

For `bitflip=True`, the expectation value is computed exactly — no Monte Carlo over $z$:

```python
par_ops_gates = 2 * effective_params * ops_gen   # 2θ_j · (a·g_j mod 2), shape (n_ops, num_gen)
expvals       = jnp.prod(jnp.cos(par_ops_gates), axis=-1)  # product over generators
```

This is $\langle Z_a \rangle = \prod_j \cos(2\theta_j \cdot (a \cdot g_j \bmod 2))$ — a product of cosines, not a cosine of a sum. It's exact and cheaper to compute, but it describes a different (classical stochastic) model.

---

## Two levels of Monte Carlo

For `bitflip=False`, the training loop uses two nested sampling loops:

**Outer loop** — sample $K$ Z-word masks $a_1, \ldots, a_K$ from $P_\sigma$ to estimate the MMD expectation. This is `n_ops` in `mmd_loss_iqp`.

**Inner loop** — for each $a_k$, sample $B$ bitstrings $z$ uniformly to estimate $\langle Z_{a_k} \rangle_{q_\theta}$. This is `n_samples` in `mmd_loss_iqp`.

```python
# from iqpopt/gen_qml/iqp_methods.py
def mmd_loss_iqp(params, iqp_circuit, ground_truth, sigma, n_ops, n_samples, key, ...)
```

```python
# from iqpopt/iqp_optimizer.py
samples = jax.random.randint(key, (n_samples, self.n_qubits), 0, 2)  # the z bitstrings
```

Neither loop touches a quantum computer. Both are just random integer generation and floating-point arithmetic.

### Where the simplifications accumulate

| What a general approach needs | What IQP + MMD needs |
|-------------------------------|----------------------|
| $2^n$-dimensional state vector | Vectors of size $n$ (one per qubit, one per gate) |
| Matrix exponentiation of large unitaries | One dot product mod 2 per gate |
| Parameter-shift rule or finite differences for gradients | Autodiff straight through `cos` |
| Quantum hardware or state-vector simulator during training | NumPy/JAX on a laptop |

Each reduction is exact, not an approximation. The cosine formula gives the true $\langle Z_a \rangle_{q_\theta}$; the Monte Carlo over $z$ is the only approximation, converging as $1/\sqrt{B}$ with sample count $B$.

---

## After training: sampling from the circuit

Once parameters are learned, generating new data requires running the actual IQP circuit. For `bitflip=False`, `IqpSimulator.sample()` uses PennyLane state vector simulation:

```python
# from iqpopt/iqp_optimizer.py
dev = qml.device(self.device, wires=self.n_qubits)  # default: "lightning.qubit"

@qml.set_shots(shots)
@qml.qnode(dev)
def sample_circuit(params):
    self.iqp_circuit(params, init_coefs)
    return qml.sample(wires=range(self.n_qubits))

return sample_circuit(params)
```

In the paper's code (`paper/README.md` at b907bb1):

```python
model = iqp.IqpSimulator(gates=gates, **model_config)
samples = model.sample(params_iqp, 100)   # API at b907bb1: positional shots
# local HEAD uses keyword: model.sample(params, shots=5000)
```

Each `.sample()` call runs the full IQP circuit and returns one bitstring. By default this runs on `lightning.qubit`, a classical state vector simulator — but the same circuit can run on real quantum hardware. The distribution $q_\theta$ is believed to be classically hard to sample from, which is where the potential quantum advantage lives.

Training never needed to sample from $q_\theta$ directly. It only needed expectation values, and those have the cosine formula.

---

## Why this works for IQP but not general circuits

The cosine formula isn't a general quantum result. It works for IQP circuits because:

- All IQP gates commute with each other (they're diagonal in the X-basis), so the circuit has no gate ordering to track.
- The output distribution has a Fourier structure that makes Z-word expectation values analytically tractable.
- The MMD loss decomposes into exactly these Z-word expectation values.

For random deep circuits, none of these hold. Computing even a single expectation value requires simulating the full quantum state, which grows exponentially.

---

## Summary

| Step | What computes it | Classical or quantum? |
|------|-----------------|----------------------|
| $\langle Z_a \rangle_{q_\theta}$ (`bitflip=False`) | Average $\cos(\Phi(\theta, z, a))$ over random $z$ | Classical |
| $\langle Z_a \rangle_{q_\theta}$ (`bitflip=True`) | Product $\prod_j \cos(2\theta_j \cdot (a \cdot g_j \bmod 2))$ | Classical (exact) |
| $\langle Z_a \rangle_p$ | Average parity over training data | Classical |
| MMD loss | Average squared differences over random $a$'s | Classical |
| Parameter update | Gradient step via JAX autodiff through the cosine | Classical |
| Generate samples (`bitflip=False`) | PennyLane state vector sim or real hardware | Quantum (or simulator) |
| Generate samples (`bitflip=True`) | Classical random walk with $\sin^2(\theta)$ flip probs | Classical |

The formula that makes large-scale classical training of standard IQP circuits possible is:

$$\boxed{\langle Z_a \rangle_{q_\theta} = \mathbb{E}_{z \sim \text{Uniform}} \left[ \cos\!\left( \sum_j 2\theta_j \cdot (a \cdot g_j \bmod 2) \cdot (-1)^{z \cdot g_j} \right) \right]}$$

implemented in `iqpopt/iqp_optimizer.py:op_expval_batch` (non-bitflip path) and mirrored in our `src/iqp_bp/iqp/expectation.py`.

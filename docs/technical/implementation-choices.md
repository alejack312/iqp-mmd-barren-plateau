# Implementation choices

This note explains the decisions behind the recent hypergraph, kernel, and validation changes. It is meant to answer the questions a new reader will have after opening the code:

- Why does the Gaussian kernel look like that?
- Why is the lattice restricted to perfect squares?
- Why is the sparse Erdos-Renyi family pairwise and intrinsic-`m`?
- What exactly are the Hypothesis tests checking?
- Why do some runners ignore the requested `m`?

If you only want the current formulas, start with [mmd-gaussian-fourier.md](./mmd-gaussian-fourier.md), [configuration.md](./configuration.md), and [architecture.md](./architecture.md). This document is the "why."

## What this round of changes was trying to do

There were really two threads here.

The first was to make the SMART family layer real instead of approximate. That meant replacing the old 2D lattice patch sampler with the exact nearest-neighbor grid family, and replacing the old generic sparse row sampler with a proper sparse Erdos-Renyi graph family.

The second was to stop hand-waving around the Gaussian kernel normalization. The repo had drifted into a state where different files were quietly using different meanings of `sigma`. That is the kind of mismatch that stays invisible until validation starts to matter. So the kernel path was tightened until the direct kernel formula, the Walsh decay, and the Z-word sampler all meant the same thing.

## The Gaussian convention

The Gaussian convention now follows the paper in [`docs/papers/2503.02934v2 (3).pdf`](/C:/Users/cuqui/iqp-mmd-barren-plateau/docs/papers/2503.02934v2%20(3).pdf).

The locked kernel is

```text
k(x, y) = exp(-H(x, y) / (2 sigma^2))
```

where `H(x, y)` is Hamming distance.

That choice now applies in all of these places:

- [`gaussian_kernel()`](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_bp/mmd/kernel.py#L32)
- [`gaussian_spectral_weights()`](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_bp/mmd/kernel.py#L38)
- [`gaussian_sample_a()`](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_bp/mmd/kernel.py#L44)
- [`multi_scale_gaussian_kernel()`](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_bp/mmd/kernel.py#L167)

The important follow-through is the Walsh decay parameter. Once the kernel is written as `exp(-H / (2 sigma^2))`, the per-bit mismatch factor is `q = exp(-1 / (2 sigma^2))`, which gives

```text
tau = (1 - q) / (1 + q) = tanh(1 / (4 sigma^2))
```

That is why the sampler now uses `tau = tanh(1 / (4 sigma^2))` instead of the older `tanh(1 / sigma^2)` or `tanh(1 / (2 sigma^2))` forms that had been floating around in the repo.

The practical reason for locking this down is simple: if `gaussian_kernel(...)` and `gaussian_sample_a(...)` disagree about what `sigma` means, the MMD estimator is no longer sampling from the kernel it claims to use.

## Why the lattice is square-only

The SMART lattice family is supposed to be the exact 2D nearest-neighbor ZZ baseline. That sounds minor, but it rules out a lot of earlier shortcuts.

The implemented family in [`families.py`](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_bp/hypergraph/families.py) means:

- qubits are arranged on an `L x L` square grid
- only open-boundary horizontal and vertical nearest-neighbor edges are included
- every generator row has Hamming weight 2
- `range_` must be `1`
- the family is deterministic for fixed `n`

Once that is the contract, `n` has to be a perfect square. There is no honest way to represent `n = 24` or `n = 48` as an exact square grid without either changing the geometry or sneaking in a different family. Failing fast on non-squares is cleaner than pretending.

This is also why the lattice ignores caller-provided `m`. The number of generators is part of the family definition:

```text
n = L^2
m = 2L(L - 1)
```

If the runner asks for some other `m`, that is a mismatch between the experiment config and the family, not something the family should silently "fix."

## Why Erdos-Renyi was changed this way

The old sparse family was not really an Erdos-Renyi graph family. It was a generic sparse row sampler: for each generator row, include each qubit independently with some probability. That produces random hyperedges, but it does not give the graph-theoretic baseline the SMART plan was asking for.

The implemented SMART Erdos-Renyi family now means:

- sample an undirected Erdos-Renyi graph on the `n` qubits
- use one generator row per sampled graph edge
- each row has Hamming weight 2
- `p_edge` is interpreted as the target average degree constant `c`
- the actual graph-edge probability is `p = min(1, c / n)`
- `m` is intrinsic and equals the number of sampled edges

Why this shape?

First, it keeps the family pairwise. That matters because the other primary baselines are pairwise too:

- `lattice` is pairwise and local
- `complete_graph` is pairwise and dense
- `erdos_renyi` is now pairwise and random sparse

That makes the family comparison about connectivity, not about interaction order.

Second, the `c / n` calibration gives bounded expected degree as the system grows. That is what "sparse regime" means in this setting. If the edge probability stayed fixed as `n` grew, the graph would get denser and denser and would stop being a meaningful sparse baseline.

Third, we chose intrinsic `m` for Erdos-Renyi, not fixed requested `m`. That choice lines up with the exact-family treatment already used for the lattice and complete graph. The family now means "sample this random graph model," not "manufacture exactly `m` rows that vaguely resemble it."

## Why the runners now trust `G.shape[0]`

Once some families have intrinsic row count, the runners cannot keep pretending the requested `m` is authoritative.

That is why the scaling and Forge paths were changed to build `G` first and then use:

```text
actual_m = G.shape[0]
```

This affects:

- [`run_scaling.py`](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_bp/experiments/run_scaling.py)
- [`run_forge.py`](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_bp/experiments/run_forge.py)

Without that change, exact families would either get the wrong `theta` shape or the wrong metadata in exported records.

## Why the Hypothesis layer was narrowed to SMART families

The earlier Hypothesis layer was still useful, but it was too generic for the current study. It mostly exercised arbitrary binary matrices and a couple of generic family placeholders. That is good for catching some shape bugs, but not good enough when the question is "are the exact SMART families really the ones the experiments are using?"

So the property-based layer was narrowed on purpose:

- [`smart_family_config()`](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_bp/hypergraph/hypothesis_strategies.py)
- [`smart_family_instance()`](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_bp/hypergraph/hypothesis_strategies.py)
- [`mmd_instance()`](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_bp/hypergraph/hypothesis_strategies.py)

The goal is not to prove every possible hypergraph routine correct. The goal is to exercise the four families the experiments are supposed to use.

## The invariants we test, and why

The family-specific Hypothesis tests are not arbitrary. Each one is meant to catch a particular class of regression.

### Common invariants

These are checked for every SMART family because they are basic sanity requirements:

- `G` is binary
- `G` has the expected shape
- no row is all zero

If any of those fail, the object is not a valid generator matrix for this codebase.

### Product state

For `product_state`, the cheap structural invariant is that the family is exactly the identity-like single-qubit baseline. It is the one family where `m = n` and every row should have weight 1.

Why test this? Because `product_state` is the simplest baseline in the whole study. If that family drifts, everything built on top of it becomes harder to interpret.

### Lattice

For `lattice`, the tests now check the exact edge set on the square grid:

- every row has weight 2
- the row count is exactly `2L(L - 1)`
- every row is a horizontal or vertical nearest-neighbor edge
- there are no duplicates
- the output is deterministic across RNG seeds

Why these invariants? Because they distinguish the real 2D nearest-neighbor family from the older patch-based approximation. If you only test "binary, right shape, row weight 2," a patch sampler can still sneak through.

### Erdos-Renyi

For `erdos_renyi`, the tests check:

- every row has weight 2
- there are no duplicate edges
- output is reproducible for a fixed RNG seed
- expected degree over repeated draws stays close to the target constant `c`

Why these invariants? Because they pin down the two things that matter for the SMART contract:

- it is actually a graph family, not a generic sparse hypergraph sampler
- it is actually sparse in the bounded-degree sense, not just "kind of not dense"

### Complete graph

For `complete_graph`, the structural invariant is that every pair of qubits appears exactly once.

Why test this? Because it is the dense extreme. If it does not contain every pair exactly once, it is not the family its name claims.

## Why the init layer changed too

The family refactor and the init refactor ended up touching the same surface, so they belong in the same note.

The init strategy now does three things:

- `uniform`
- `small_angle` with the exact sweep `{0.01, 0.1, 0.3}`
- `data_dependent`

The small-angle change is there because the SMART plan called for an exact sweep, not "some float in a range that feels small."

The `data_dependent` path now exists in both the Hypothesis layer and the scaling runner. Right now it is a simple data-aware warm start based on empirical parity expectations over generator supports. It is not pretending to be more sophisticated than that.

The reason to split "choose an init scheme" from "materialize `theta`" is that once an init depends on `G` and the data, it no longer makes sense to treat it as a pure parameter-vector sampler.

## The parity bug that surfaced while doing this

One small but important fix happened because of the Gaussian cleanup, not because it was on the original checklist.

In [`mmd/mixture.py`](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_bp/mmd/mixture.py), the code computing data-side parity expectations was doing arithmetic directly on unsigned integer arrays:

```text
1 - 2 * parities
```

With `uint8`, that can underflow instead of producing `-1`. Once the Gaussian tests started probing the kernel path more directly, that bug became obvious because the MMD values exploded.

The fix was simple:

- cast parities to `float64`
- build the sign array as `1.0 - 2.0 * parity`

That change is easy to miss in a diff, but it matters. Without it, the data-side expectation path is wrong even when the family and kernel logic are right.

## What this commit closed

This round of work closes these items in practical terms:

- `T1`: Gaussian normalization is now locked
- `T3`: lattice is the exact square-grid nearest-neighbor family
- `S1`: Erdos-Renyi is the bounded-degree pairwise graph family
- `V1`: Hypothesis now exercises SMART families rather than placeholders

That is why [`TODOS.md`](/C:/Users/cuqui/iqp-mmd-barren-plateau/TODOS.md) and [`src/iqp_bp/hypergraph/PLAN.md`](/C:/Users/cuqui/iqp-mmd-barren-plateau/src/iqp_bp/hypergraph/PLAN.md) were updated.

## What this commit did not settle

Two things are still separate from the choices documented here.

First, the stricter `V5` question is not fully settled if you want a stronger notion of `data_dependent` init. The current implementation is real and data-aware, but it is still a lightweight warm start rather than a more ambitious covariance-driven design.

Second, the small-`n` gradient validation path still deserves its own cleanup. The Gaussian normalization work exposed a data-side parity bug, which is now fixed, but the analytic-versus-finite-difference gradient test remains a separate issue. That is not a reason to back out the normalization changes. It is a reason to keep the gradient validation work scoped and honest.

## If you are changing this code later

There are a few rules worth keeping in mind.

- If you change the Gaussian formula, change the Walsh decay and the sampler in the same commit.
- If you change a family with intrinsic structure, update the runner logic that derives `m`.
- If you relax a family invariant, say why in the docs and adjust the Hypothesis tests on purpose. Do not just weaken a failing test and move on.
- If a config example stops matching the real family contract, fix the example. Those examples are part of the interface now.

That last one is not glamorous, but it is usually where confusion starts.

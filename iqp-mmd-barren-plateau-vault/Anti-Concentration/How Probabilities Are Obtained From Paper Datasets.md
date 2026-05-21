The short answer: the paper datasets do not provide exact probabilities p(x) for each bitstring. They provide samples from an unknown/implicit target distribution. For the anti-concentration plot, target probabilities are obtained by empirical histogramming those samples when n is small enough. The learned IQP probabilities q_theta(x) are obtained from the trained circuit, either by exact enumeration for small n or by IQPOpt's probability routine in the paper-code path.

  Exact Pipeline

  1. Paper dataset gives binary samples X = {x_i}.
      - Loaded as CSV rows in src/iqp_mmd/datasets/loaders.py:9.
      - {-1,+1} data is normalized to {0,1} in src/iqp_mmd/datasets/loaders.py:23.
      - Dataset paths are defined in src/iqp_mmd/config/paths.py:52.
  2. For a target anti-concentration curve, samples become empirical probabilities:
      - samples_to_probability_vector(X) in src/iqp_bp/experiments/run_validation.py:178
      - It maps each bitstring row to a big-endian integer index, counts occurrences with np.bincount, then divides by sample count: src/iqp_bp/
        experiments/run_validation.py:194.
      - Formula:

        ```text
        p_emp(x) = count_X(x) / |X|
        ```

  3. For learned IQP anti-concentration:
      - Small n local path: exact q_theta from src/iqp_bp/iqp/model.py:162.
      - Paper-code/IQPOpt path: sim.probs(theta) in scripts/grid5000_run_iqp_mmd_ac.py:434.
      - Then check_anti_concentration(q_theta) is run on that probability vector.
      - For native large-n checks, we do not enumerate q_theta(x). We estimate the scaled second moment from Pauli-Z expectations using Parseval:

        ```text
        2^n sum_x q_theta(x)^2 = sum_a <Z_a>_q_theta^2.
        ```

  Dataset Origins In The Paper
  From the local PDF docs/papers/2503.02934v2 (3).pdf, pages 17-19 and 23-30:

  - 2D Ising: MCMC samples from a 16-spin Ising model.
  - Binary blobs: synthetic samples near eight fixed 16-bit patterns.
  - D-Wave: samples from a 484-qubit D-Wave Advantage annealer dataset.
  - MNIST: binarized, flattened 784-pixel MNIST images.
  - Scale free: MCMC samples from a 1000-spin Ising model on a scale-free graph.
  - Genomic: 805-SNP haplotype bitstrings, train/test split with ratio 1/3.

  The corresponding local code is in:

  - src/iqp_mmd/datasets/ising.py:29
  - src/iqp_mmd/datasets/blobs.py:21
  - src/iqp_mmd/datasets/dwave.py:17
  - src/iqp_mmd/datasets/mnist.py:19
  - src/iqp_mmd/datasets/genomic.py:27

  Important caveat: for native large datasets like genomic-805, D-Wave-484, MNIST-784, or scale-free-1000, a full vector over 2^n outcomes is
  infeasible. The paper trains/evaluates mainly through MMD/KGEL/covariance expectation estimates, not full probability tables. Our exact anti-
  concentration plot is therefore only exact for small n or projected small-n controls.

  Sources: arXiv paper page https://arxiv.org/abs/2503.02934v2, upstream code repo https://github.com/XanaduAI/scaling-gqml.

## Long Version: How The Probabilities Are Obtained

The paper's datasets should be understood as sample datasets, not as probability tables. In Section 3.1 of `docs/papers/2503.02934v2 (3).pdf`, the paper defines the generative learning setup as having access to a dataset

```text
X = {x_i}
```

where each `x_i` is a bitstring in `{0,1}^n` sampled from a ground-truth distribution `p(x)`. That means the paper assumes a true distribution exists conceptually, but the files we actually have are finite lists of observed bitstrings. The dataset files do not contain one row per possible bitstring with an attached probability. They contain one row per observed sample.

This distinction matters for the anti-concentration plot. Anti-concentration is a property of a full probability distribution over all `2^n` bitstrings. For a distribution `p`, the anti-concentration diagnostic asks how many outcomes have probability at least some constant multiple of the uniform probability `2^-n`. In our code, this is evaluated by `check_anti_concentration` in `src/iqp_bp/experiments/run_validation.py`. The function expects a full probability vector whose length is exactly `2^n`.

For the target distribution from the paper data, we do not know the true `p(x)`. We only have samples from it. Therefore, the target probability vector used by the anti-concentration code is the empirical distribution induced by the sample table:

```text
p_emp(x) = number of dataset rows equal to x / total number of dataset rows.
```

Concretely, if the dataset contains `N` rows and the bitstring `x = 0101...` appears `c_x` times, then the empirical probability assigned to that bitstring is

```text
p_emp(x) = c_x / N.
```

If a bitstring never appears in the finite dataset, the empirical probability assigned to it is zero. This is not a claim that the true distribution gives that bitstring zero probability. It only means that the finite dataset did not observe it.

The implementation of this empirical conversion is `samples_to_probability_vector(samples)` in `src/iqp_bp/experiments/run_validation.py`. The function first checks that the input is a two-dimensional binary array. If the data has shape `(N, n)`, then `N` is the number of observed samples and `n` is the number of bits per sample. It then maps every binary row to an integer index in the range `0` to `2^n - 1`. The code uses big-endian bit weights:

```text
bit_weights = [2^(n-1), 2^(n-2), ..., 2^0]
```

so the row `[b_0, b_1, ..., b_(n-1)]` maps to

```text
index = b_0 2^(n-1) + b_1 2^(n-2) + ... + b_(n-1) 2^0.
```

This indexing convention is important because it keeps the empirical target probabilities aligned with the exact IQP probability vector returned by our local IQP model code. After computing the integer index for every row, the code calls `np.bincount` to count how many times each index occurred. It then divides the count vector by the total count:

```text
counts[index(x)] = number of rows equal to x
probabilities = counts / sum(counts)
```

The result is a vector of length `2^n`. Entry `j` is the empirical probability of the bitstring whose big-endian integer index is `j`.

The copied paper-code path in `src/iqp_mmd` loads the paper datasets in the same sample-based way. `src/iqp_mmd/datasets/loaders.py` reads CSV files into arrays with one sample per row. If a dataset is encoded as `{-1,+1}`, `normalize_binary` converts it into `{0,1}` by applying `(1 + X) // 2`. The train/test dataset locations are listed in `src/iqp_mmd/config/paths.py`; for example, the path map includes `2D_ising`, `8_blobs`, `dwave`, `MNIST`, `scale_free`, and `genomic-805`.

During IQP training, the paper-code path does not first build a full target probability table. In `src/iqp_mmd/training/iqp_trainer.py`, the training samples are passed into IQPOpt as `ground_truth`. The MMD loss uses the sample set directly to estimate expectation values of observables under the empirical training distribution. This matches the paper's training description in Sections 4.3 and 8.1: the target expectation values are estimated from sampled batches of data, and the loss is written as an MMD between `P_Xtrain`, the empirical training distribution, and the learned IQP distribution `q_theta`.

So there are two different probability objects involved:

1. The target probabilities for the paper dataset are empirical probabilities from sample counts, `p_emp(x)`.
2. The learned probabilities are circuit probabilities from the trained IQP model, `q_theta(x)`.

For the learned IQP model, our small-`n` local code can compute the exact probability of every bitstring by enumerating the full state vector. That path is `IQPModel.probability_vector_exact` in `src/iqp_bp/iqp/model.py`. It returns a normalized probability vector over all `2^n` computational-basis outcomes. This is feasible only when `n` is small enough, because the vector size doubles with every additional qubit.

The Grid5000/native paper-code path uses IQPOpt's simulator probability method instead. In `scripts/grid5000_run_iqp_mmd_ac.py`, the learned probability vector is obtained with:

```text
q_theta = sim.probs(theta)
```

Exactly, `sim` is an `iqpopt.iqp_optimizer.IqpSimulator` object. Its `probs` method builds a PennyLane state-vector device, applies the trained IQP circuit to the all-zero initial state, and asks PennyLane for the probabilities of all computational-basis measurement outcomes. In the installed IQPOpt source, the method does the following:

```text
dev = qml.device(self.device, wires=self.n_qubits)

@qml.qnode(dev)
def probs_circuit(params):
    self.iqp_circuit(params, init_coefs)
    return qml.probs(wires=range(self.n_qubits))

return probs_circuit(params)
```

So `sim.probs(theta)` is not sampling. It is exact state-vector simulation followed by exact probability extraction. It returns a dense vector:

```text
q_theta[j] = probability of measuring computational-basis bitstring j.
```

The vector length is `2^n`, where `n = sim.n_qubits`. The entries are ordered in the same big-endian/MSB-first computational-basis convention used by our histogram code:

```text
q_theta[0]       = q_theta(00...000)
q_theta[1]       = q_theta(00...001)
q_theta[2]       = q_theta(00...010)
...
q_theta[2^n - 1] = q_theta(11...111)
```

Internally, the circuit applied by `self.iqp_circuit(theta, init_coefs)` has these steps:

1. If `spin_sym=True`, IQPOpt first applies a symmetry-preparation rotation. The IQPOpt docstring describes this as making the circuit equivalent to starting from the spin-flip-symmetric state

```text
(|00...0> + |11...1>) / sqrt(2)
```

instead of starting only from `|00...0>`.

2. It applies a Hadamard gate to every qubit.

3. If fixed initialization gates are present, it applies their `MultiRZ(2 * coefficient)` rotations on the configured generator wires.

4. For each trained parameter `theta_j`, it applies `MultiRZ(2 * theta_j)` to every generator associated with that parameter. A generator is a subset of qubits. Applying `MultiRZ` on that subset implements the commuting phase rotation for that IQP interaction. Because the circuit is wrapped by Hadamards, these `Z`-basis phase rotations correspond to the paper's IQP gates generated by products of Pauli-`X` operators.

5. It applies a final Hadamard gate to every qubit.

6. PennyLane computes the final state vector and returns `qml.probs(wires=range(n))`, i.e. the squared magnitudes of the final amplitudes in computational-basis order:

```text
q_theta(x) = |<x| U(theta) |initial>|^2.
```

This is why the result can be passed directly into `check_anti_concentration(q_theta)`: it is already a normalized full probability distribution over all bitstrings. The cost is the reason this path is restricted to small `n`: state-vector simulation and the returned probability vector both scale as `2^n`.

That path is important for paper-faithful checkpoints, especially when the original IQPOpt model uses options such as spin-flip symmetry. After `q_theta` is obtained, the script evaluates:

```text
ac_learned = check_anti_concentration(q_theta)
p_emp = samples_to_probability_vector(X)
ac_target = check_anti_concentration(p_emp)
```

Thus the learned curve and target curve are made comparable by converting both into probability vectors over the same bitstring index set. The learned vector comes from the circuit. The target vector comes from histogramming the observed dataset samples.

The anti-concentration statistic itself is then computed from those probability vectors. For each threshold multiplier `alpha`, the code compares every probability to `alpha * 2^-n`. The reported beta value is the fraction of outcomes whose probability is above that threshold:

```text
beta_hat(alpha) = mean_x 1[p(x) >= alpha * 2^-n].
```

The code also reports the collision probability and scaled second moment:

```text
collision_probability = sum_x p(x)^2
scaled_second_moment = 2^n * sum_x p(x)^2.
```

The scaled second moment equals `1` for the uniform distribution. Larger values indicate that probability mass is more concentrated on fewer bitstrings.

## Theoretical Analysis From The Paper

The theoretical reason the code can avoid explicit probability tables during training is in the paper's Section 4. The paper rewrites the MMD loss in terms of Pauli-`Z` expectation values rather than probabilities `p(x)` and `q_theta(x)` themselves.

For a bit mask `a in {0,1}^n`, the paper defines the Pauli word

```text
Z_a = product_i Z_i^{a_i}
```

and the model expectation

```text
<Z_a>_q = <0| U(theta)^dagger Z_a U(theta) |0>.
```

This expectation is the Fourier/Walsh coefficient of the output probability distribution:

```text
<Z_a>_q = sum_x q_theta(x) (-1)^{a dot x}.
```

The paper's Proposition 1 gives the IQP-specific classical formula for this quantity. If the IQP generators are `g_j`, then

```text
<Z_a>_q = E_{z uniform} cos( sum_j theta_j (-1)^{g_j dot z} (1 - (-1)^{g_j dot a}) ).
```

The paper then estimates this by sampling uniform bitstrings `z` and averaging the cosine. This is exactly the theoretical object used by IQPOpt's `op_expval` routine: it computes Pauli-`Z` expectation values of the trained circuit without constructing all `2^n` probabilities.

The data side is analogous but simpler. The paper writes

```text
<Z_a>_p = E_{x ~ p} [(-1)^{x dot a}],
```

and estimates it from a dataset batch by

```text
<Z_a>_p_hat = (1 / |X|) sum_{x_i in X} (-1)^{x_i dot a}.
```

So the training data probabilities are never needed as an explicit table for MMD training. The samples are enough because the loss only asks for parity averages under the empirical training distribution.

The paper's Proposition 2 then expresses the Gaussian-kernel MMD as a mixture of these expectation mismatches:

```text
MMD^2(p, q_theta) = E_{a ~ P_sigma} [ (<Z_a>_p - <Z_a>_q)^2 ].
```

Here `P_sigma` is a product Bernoulli distribution over masks `a`; each bit is included with probability

```text
p_sigma = (1 - exp(-1 / (2 sigma^2))) / 2.
```

This is why the bandwidth controls which correlations are learned: small `sigma` samples larger Pauli masks on average, so the loss probes higher-order correlations. Section 8.1 of the paper uses the empirical training distribution `P_Xtrain` and averages MMD values over several bandwidths:

```text
L = (1 / L_count) sum_i MMD_i^2(P_Xtrain, q_theta).
```

This is the main theoretical link to the code: `ground_truth=X_train` supplies empirical estimates of `<Z_a>_p`; the IQP expectation routine supplies estimates of `<Z_a>_q`; the optimizer minimizes their squared difference over masks sampled according to the paper's `P_sigma`.

For the anti-concentration scaled second moment, our exact small-`n` route computes `q_theta(x)` first and then evaluates

```text
S(q_theta) = 2^n sum_x q_theta(x)^2.
```

For native large-`n` datasets, the better theoretical route is to use the same Pauli-expectation view. By Parseval's identity for the Walsh basis,

```text
2^n sum_x q_theta(x)^2 = sum_{a in {0,1}^n} <Z_a>_q^2.
```

The identity mask `a = 0` contributes `1`, so the large-`n` estimator samples nonzero Pauli masks uniformly and estimates

```text
S(q_theta) = 1 + (2^n - 1) E_{a != 0} [ <Z_a>_q^2 ].
```

The same identity applies to the empirical training distribution:

```text
S(p_emp) = 2^n sum_x p_emp(x)^2 = sum_a <Z_a>_p_emp^2.
```

When `n` is small, the code can compute this from the histogram counts:

```text
S(p_emp) = 2^n sum_x (count_X(x) / |X|)^2.
```

When `n` is large, the histogram over all `2^n` strings is impossible, but each `<Z_a>_p_emp` is still just an average parity over observed rows. Therefore the target scaled second moment can also be estimated by sampling Pauli masks `a`, computing empirical parity averages on the dataset, squaring them, and applying the same Parseval Monte Carlo scaling.

That is the conceptual bridge your supervisor was pointing to: the probabilities are explicit only in the small-`n` code path. At paper scale, the relevant distributional quantities are obtained from the theoretical Pauli expectation formulas. The anti-concentration scaled second moment can then be computed or estimated from those squared expectations, without materializing the `2^n` probability vector.

Dataset by dataset, the source of the samples is as follows:

- `2D_ising`: The paper describes 16-bit samples from a thermal Ising distribution on a `4 x 4` square lattice. The paper generated 800000 configurations using Metropolis-Hastings sampling on 8 Markov chains, then selected 5000 training samples and 50000 test samples. In the copied code, the generation path is represented by `src/iqp_mmd/datasets/ising.py`.
- `8_blobs`: The paper describes a synthetic 16-bit dataset with eight modes. A base pattern is selected and each bit is independently flipped with probability `0.05`. The copied code path is `src/iqp_mmd/datasets/blobs.py`.
- `dwave`: The paper describes samples from a 484-qubit D-Wave Advantage processor with Pegasus topology, using 10000 training samples and 60000 test samples. In our copied code, the dataset is loaded as CSV rows through the path map in `src/iqp_mmd/config/paths.py`.
- `MNIST`: The paper uses binarized MNIST images. Pixel values are thresholded and flattened, producing bitstrings of length `784`. The standard split is 50000 training samples and 10000 test samples.
- `scale_free`: The paper describes a 1000-spin Ising model on a scale-free graph. It generated one million configurations using Metropolis-Hastings sampling on 8 chains, then selected 20000 training samples and 20000 test samples.
- `genomic-805`: The paper describes 805-SNP haplotype bitstrings from Yelmen et al. The data comes from 2504 individuals, giving 5008 haplotypes because each individual contributes two haplotypes. The paper splits this data into train and test sets with test ratio `1/3`. In the copied code, `src/iqp_mmd/datasets/genomic.py` downloads the pinned 805-SNP haplotype file, drops the first two metadata columns, and uses the remaining columns as binary SNP features.

## Genomic Dataset Paragraph

The genomic dataset is a real-world binary haplotype dataset built from 805 highly differentiated biallelic single-nucleotide polymorphisms (SNPs). Each SNP has two possible allele values, so each haplotype becomes an 805-bit string where a `1` marks the presence of the variant allele at that SNP. The source data contains 2504 individuals from the 1000 Genomes project, giving 5008 haplotypes because each individual contributes two haplotypes. The task is generative modelling: train the IQP circuit so that its output bitstrings reproduce the empirical distribution of genomic haplotypes, especially the allele frequencies and correlations between SNPs. Because the native space has `2^805` possible haplotypes, neither the paper nor our native run can enumerate all probabilities; evaluation must use sample-based MMD/KGEL/covariance diagnostics or Pauli-expectation estimators for quantities such as the scaled second moment.

The main limitation is dimensionality. For the two 16-bit datasets, a full empirical probability vector has length:

```text
2^16 = 65,536.
```

That is large but manageable, so exact empirical target histograms and exact learned IQP probability vectors are realistic.

For the native large datasets, the full probability vector is not realistic:

```text
D-Wave:     2^484 outcomes
MNIST:      2^784 outcomes
Genomic:    2^805 outcomes
Scale free: 2^1000 outcomes
```

These spaces are astronomically large. Even storing one floating-point probability for every possible bitstring is impossible. For those native high-dimensional datasets, the paper does not train or evaluate by constructing full probability tables over all outcomes. It trains and evaluates using MMD, KGEL, and covariance-style expectation estimates that can be computed from samples and from IQP expectation routines. When we need a native large-`n` scaled second moment, we should describe it as a Pauli/Parseval estimator, not exact probability enumeration.

Therefore, when we make an exact anti-concentration plot, the probabilities are exact only in small enough settings where the full `2^n` vector can actually be built. In those cases, the target probabilities are empirical histogram probabilities from the dataset samples, and the learned probabilities are exact circuit probabilities from the trained IQP model. For native high-dimensional paper datasets, an exact full-distribution anti-concentration curve is not feasible without projecting to a smaller subset of bits, using a small-`n` control problem, or replacing exact enumeration with an estimator.

The safest wording for a supervisor is:

```text
The paper datasets provide samples, not exact probability masses. For anti-concentration at small n, we obtain the target distribution by empirical histogramming: each bitstring probability is its observed frequency in the finite dataset. The learned IQP distribution is obtained from the trained circuit's exact probability routine when n is small enough. The theoretical paper-scale route is to compute Pauli-Z expectation values <Z_a> for the data and for the IQP circuit, and MMD is an average of squared expectation mismatches. For scaled second moment, Parseval gives 2^n sum_x q(x)^2 = sum_a <Z_a>_q^2, so large-n runs estimate the anti-concentration scalar from Pauli expectations rather than from a full 2^n probability vector. For the native 484-, 784-, 805-, and 1000-bit datasets, exact probability enumeration is infeasible.
```

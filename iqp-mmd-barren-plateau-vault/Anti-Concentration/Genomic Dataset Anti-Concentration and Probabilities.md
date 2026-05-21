How the paper computes, or could compute, probabilities for the scaled second moment

$$
S = 2^n \sum_x p(x)^2.
$$

# How probabilities are (and aren't) handled in the paper

## For the training data

The training data is just samples. The empirical distribution

$$
P_{X_{\mathrm{train}}}(x)
= \frac{1}{|X_{\mathrm{train}}|}
\sum_i \mathbf{1}(x_i = x)
$$

is implicit in the dataset itself. No probability computation is needed because the paper never works with the empirical probability vector directly. Instead, every quantity it computes about the data uses expectation values:

$$
\langle Z_a \rangle_p
= \mathbb{E}_{x \sim p}\left[(-1)^{x \cdot a}\right]
\approx
\frac{1}{|X|}
\sum_i (-1)^{x_i \cdot a}.
$$

This is equation 19 in the paper. It is just averaging $\pm 1$ values over the training samples: cheap, no enumeration, and scalable to any $n$. The paper only ever needs $\langle Z_a \rangle_p$, never $p(x)$, so it never computes the empirical probability vector.

**Implication for the scaled second moment:** If we wanted to compute

$$
S_p = 2^n \sum_x p(x)^2
$$

from data, the paper does not show this directly, but Parseval's identity gives a clean route using only the primitives the paper does compute:

$$
S_p
= 2^n \sum_x p(x)^2
= \sum_a \langle Z_a \rangle_p^2.
$$

This is exact. Each $\langle Z_a \rangle_p^2$ is estimated by squaring the sample mean from equation 19. Summing over all $2^n$ values of $a$ is exponentially expensive, but at small $n$ it is trivial. For example, $n = 9$ has $512$ terms, and the paper's $n = 16$ datasets have $65{,}536$ terms. At the paper's $n \ge 484$ scales, we would estimate the sum by sampling $a$ uniformly and averaging, which is exactly the primitive the paper builds.

So the answer to "how does the paper compute probabilities of the training data?" is: **it doesn't**. It computes only Pauli expectations from samples. We can extract the scaled second moment via Parseval if we want it.

## For the output of the quantum circuit

For the learned $q_\theta$, the situation is the inverse. The paper cannot compute $q_\theta(x)$ directly for general bitstrings $x$ without exponential classical effort or quantum sampling, but it can compute any $\langle Z_a \rangle_{q_\theta}$ classically via Proposition 1. Equation 13 is the key formula:

$$
\langle Z_a \rangle_{q_\theta}
=
\mathbb{E}_{z \sim U}
\left[
\cos\left(
\sum_j \theta_j (-1)^{g_j \cdot z}
\left(1 - (-1)^{g_j \cdot a}\right)
\right)
\right].
$$

This is an expectation over uniformly random bitstrings $z$ of a cosine of an inner product. The empirical estimator in equation 14 is the corresponding sample mean. The cost is polynomial in $n$ for any single $\langle Z_a \rangle$. This is the algorithmic foundation of the paper: it lets the authors train at $n = 1000$ without ever touching the $2^{1000}$-dimensional probability vector.

**Implication for the scaled second moment of $q_\theta$:** by Parseval again,

$$
S_{q_\theta}
= 2^n \sum_x q_\theta(x)^2
= \sum_a \langle Z_a \rangle_{q_\theta}^2.
$$

Each $\langle Z_a \rangle_{q_\theta}$ is estimated by equation 14. The sum has $2^n$ terms. At the paper's large $n$, this is intractable to enumerate, but we can sample $a$ uniformly and form an unbiased estimator of $S_{q_\theta}$ using only Proposition 1. This is the same style of trick the paper uses for the KGEL observable in Appendix D, equations 111-115: convert a sum over all $2^n$ Pauli strings into an expectation over sampled $a$, then estimate by sampling.

**Critical point for the writeup:** the paper itself does not compute $S_{q_\theta}$ anywhere. The machinery is there in equations 13 and 14, and in the KGEL trick in Appendix D, but the paper never puts it together for the second moment. This is the gap the experiment fills.

## The bitflip surrogate

For the stochastic bitflip model, equation 32 gives a closed-form expression:

$$
\langle Z_a \rangle
=
\prod_{j:\{X_{g_j}, Z_a\}=0}
\cos(2\theta_j).
$$

No estimator and no sampling are needed; it is exact in $O(m)$ per expectation value, where $m$ is the number of gates. This is why the paper can train both models at scale with the same code.

## What the paper does not contain

It does not contain:

- Any computation of the empirical probability vector of the training data, only Pauli expectations from samples.
- Any computation of $q_\theta(x)$ for individual bitstrings, only $\langle Z_a \rangle_{q_\theta}$ via Proposition 1.
- Any anti-concentration diagnostic on either $p$ or $q_\theta$: no collision probability, no $\hat{\beta}(\alpha)$, and no scaled second moment.
- Any direct enumeration over the $2^n$ bitstring space. Every quantity is reformulated to scale polynomially.

The relevant equations to cite to our supervisor when explaining the probability handling are:

- **Equation 13** and the derivation in Section 4.1: how $\langle Z_a \rangle_{q_\theta}$ is reformulated as an expectation over uniform $z$, enabling polynomial-time estimation.
- **Equation 14**: the empirical estimator for $\langle Z_a \rangle_{q_\theta}$.
- **Equation 19**: the empirical estimator for $\langle Z_a \rangle_p$ from training samples.
- **Equation 32**: the closed-form expression for the bitflip surrogate.
- **Appendix D, equations 111-115**: the trick for converting a sum over all $2^n$ Pauli strings into an expectation over sampled $a$. This is the template for extending the same machinery to compute $S_{q_\theta}$.

# The genomic dataset paragraph

The paper's description, Experiment 6 on page 29 of the PDF, is brief, so this synthesizes the paper plus the Yelmen et al. source.

**Background:** Human genomes differ from one another at specific positions called single nucleotide polymorphisms (SNPs). At each SNP location, an individual carries one of two possible nucleotide variants, called alleles. We can encode them as $0$ for the reference allele and $1$ for the variant allele. The pattern of alleles a person carries across many SNPs is their haplotype. Population genetics studies which haplotypes occur, at what frequencies, and how they correlate across positions due to shared ancestry, selection, and recombination history.

**The specific dataset:** Yelmen et al. (2021) curated 805 highly differentiated biallelic SNPs from a variable region of the human genome. They took 2,504 individuals from the 1000 Genomes Project. Since each person has two copies of each chromosome, this produces 5,008 haplotype bitstrings, each of length 805. A $1$ at position $i$ means the individual carries the variant allele at SNP $i$; a $0$ means the reference allele. The Recio-Armengol paper splits this 2:1 into training, with 3,338 haplotypes, and test, with 1,670 haplotypes.

**The task:** Generative modeling of human haplotype distributions. The goal is to learn a distribution

$$
q_\theta \quad \text{over} \quad \{0,1\}^{805}
$$

such that samples from $q_\theta$ resemble real human haplotypes. In practice, this means preserving empirical allele frequencies at each SNP, linkage disequilibrium, and longer-range population-structure signals. Yelmen et al. discuss downstream uses such as privacy-preserving release of genetic data, data augmentation for genome-wide association studies, and benchmarking generative models for high-dimensional structured binary data.

**Why the paper chose this dataset:** Three reasons, mostly implicit in the paper:

1. It is a real-world high-dimensional binary dataset, exactly the regime IQP circuits naturally fit.
2. Yelmen et al. already published trained RBM and GAN baselines on this data with downloadable samples, so the paper can compare the IQP model against existing classical baselines without retraining them.
3. The data has rich correlation structure, including population-genetic linkage disequilibrium, that goes beyond simple Ising-style 2-body interactions.

**What the paper reports:** The IQP model achieves test MMD$^2$ values "generally lying in between the two classical models," meaning Yelmen's RBM and GAN. The covariance plots in Fig. 11 show that the IQP model captures the overall structure of inter-SNP correlations but with somewhat weaker magnitudes than the ground truth. This is consistent with the general "smoother than target" pattern the experiments observe on toy data. The paper flags two caveats: Yelmen's classical models appear to have been trained on the full data, train plus test, giving them an unfair edge; and the RBM samples turned out to be MCMC-correlated rather than i.i.d., making their reported MMD values somewhat untrustworthy.

**One caveat for the writeup:** The paper's claim is that the IQP model is competitive with published genomic baselines. That is true on MMD$^2$. But our supervisor's questions, whether the learned distribution agrees with the target on high-order marginals and whether it is anti-concentrated in the right way, are not answered in the genomic experiment any more than in the others. The covariance plots show 2-body agreement only. Whether the IQP model captures the higher-order correlation structure that matters in population genetics, such as haplotype block structure and rare-variant co-occurrence, is not tested. This is the same gap as elsewhere in the paper.

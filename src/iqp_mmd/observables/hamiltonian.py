"""Quantum expectation value calculations for Hamiltonian moments.

Provides functions to compute moments of Hamiltonians using IQP circuits
or classical samples, including magnetization and energy observables.
"""

import numpy as np
import jax.numpy as jnp
from jax._src.typing import Array


def z_ham_pow(ops: np.ndarray, coeffs: np.ndarray, p: int) -> tuple:
    """Compute H**p for a Hamiltonian H represented by Pauli-Z operators and coefficients.

    Args:
        ops: Matrix with 0/1 entries. Each row is an operator; 0=identity, 1=Z on that qubit.
        coeffs: Coefficient for each operator row.
        p: Power to raise H to.

    Returns:
        Tuple of (new_ops, new_coeffs) after simplification.
    """

    def recurrent_pow(ops, coeffs, p, tot_ops, tot_coef, res_ops=None, res_coef=None):
        if res_ops is None:
            res_ops = []
        if res_coef is None:
            res_coef = []
        ops = np.array(ops, dtype=int)
        for i, o in enumerate(ops):
            res_ops.append(o)
            res_coef.append(coeffs[i])
            if p != 1:
                recurrent_pow(ops, coeffs, p - 1, tot_ops, tot_coef, res_ops, res_coef)
            else:
                to = np.zeros_like(ops[0])
                for ro in res_ops:
                    to ^= ro
                tot_ops.append(to)
                tc = 1
                for rc in res_coef:
                    tc *= rc
                tot_coef.append(tc)
            res_ops.pop()
            res_coef.pop()

    tot_ops = []
    tot_coef = []
    recurrent_pow(ops, coeffs, p, tot_ops, tot_coef)

    seq = tot_ops.copy()
    arg_sort = np.array(sorted(range(len(seq)), key=lambda x: [i for i in seq[x]])).argsort()

    def gen(arr):
        for a in arr:
            yield a

    gen_sort = gen(arg_sort)
    tot_ops.sort(key=lambda x: next(gen_sort))
    gen_sort = gen(arg_sort)
    tot_coef.sort(key=lambda x: next(gen_sort))

    new_tot_ops = []
    new_tot_coef = []
    new_coef = 0
    for i, to in enumerate(tot_ops):
        new = True
        for nto in new_tot_ops:
            if (to == nto).all():
                new = False
        if not new:
            new_coef += tot_coef[i - 1]
        else:
            if new_tot_ops != []:
                new_tot_coef.append(new_coef + tot_coef[i - 1])
            new_tot_ops.append(to)
            new_coef = 0
    new_tot_coef.append(new_coef + tot_coef[i])

    return np.array(new_tot_ops), np.array(new_tot_coef)


def moment_ham_exp_val_iqp(
    iqp_circuit,
    params: np.ndarray,
    ops: np.ndarray,
    coeffs: np.ndarray,
    moment: int,
    n_samples: int,
    key: Array,
    indep_estimates: bool = False,
) -> tuple:
    """Compute the expected value of H^moment using an IQP circuit.

    Args:
        iqp_circuit: IqpSimulator instance.
        params: IQP gate parameters.
        ops: Operator matrix (rows are Z-string operators).
        coeffs: Coefficients for each operator.
        moment: Power of H.
        n_samples: Number of samples for expectation estimation.
        key: JAX random key.
        indep_estimates: Use independent estimates per operator.

    Returns:
        Tuple of (expectation_value, standard_deviation).
    """
    new_ops, new_coeffs = z_ham_pow(ops, coeffs, moment)
    mean, std = iqp_circuit.op_expval(params, new_ops, n_samples, key, indep_estimates)
    result = (new_coeffs * mean).sum()
    res_std = jnp.sqrt((new_coeffs**2 * std**2).sum())
    return result, res_std


def magnet_moment_iqp(
    iqp_circuit,
    params: np.ndarray,
    moment: int,
    n_samples: int,
    key: Array,
    wires: list = None,
    indep_estimates: bool = False,
) -> tuple:
    """Compute the magnetization moment of the IQP circuit distribution.

    Args:
        iqp_circuit: IqpSimulator instance.
        params: IQP gate parameters.
        moment: Moment order.
        n_samples: Number of samples.
        key: JAX random key.
        wires: Qubit subset to measure. None = all qubits.
        indep_estimates: Use independent estimates.

    Returns:
        Tuple of (expectation_value, standard_deviation).
    """
    if wires is None:
        wires = jnp.array(range(iqp_circuit.n_qubits))

    ops = []
    coefs = []
    for i in range(iqp_circuit.n_qubits):
        if i in wires:
            op = np.zeros(iqp_circuit.n_qubits)
            op[i] = 1
            ops.append(op)
            coefs.append(1 / iqp_circuit.n_qubits)

    return moment_ham_exp_val_iqp(
        iqp_circuit, params, jnp.array(ops), jnp.array(coefs), moment, n_samples, key, indep_estimates
    )


def energy_moment_iqp(
    iqp_circuit,
    params: np.ndarray,
    moment: int,
    j_matrix: np.ndarray,
    n_samples: int,
    key: Array,
    ext_field: bool = False,
    wires: list = None,
    indep_estimates: bool = False,
) -> tuple:
    """Compute the energy moment of the IQP circuit distribution.

    Args:
        iqp_circuit: IqpSimulator instance.
        params: IQP gate parameters.
        moment: Moment order.
        j_matrix: Spin coupling matrix (symmetric, zero diagonal).
        n_samples: Number of samples.
        key: JAX random key.
        ext_field: Include external field terms.
        wires: Qubit subset. None = all.
        indep_estimates: Use independent estimates.

    Returns:
        Tuple of (expectation_value, standard_deviation).
    """
    if wires is None:
        wires = jnp.array(range(iqp_circuit.n_qubits))

    if len(j_matrix) != len(j_matrix[0]):
        raise ValueError("j_matrix must be a square matrix.")
    if len(wires) != len(j_matrix):
        raise ValueError(f"j_matrix has {len(j_matrix)} qubits, but wires has {len(wires)} qubits.")

    ops = []
    coefs = []

    if ext_field:
        for i in range(len(wires)):
            if i in wires:
                op = np.zeros(iqp_circuit.n_qubits)
                op[wires[i]] = 1
                ops.append(op)
                coefs.append(-1)

    for i in range(0, len(wires) - 1):
        for k in range(i + 1, len(wires)):
            op = np.zeros(iqp_circuit.n_qubits)
            op[wires[i]], op[wires[k]] = 1, 1
            ops.append(op)
            coefs.append(-j_matrix[i][k])

    return moment_ham_exp_val_iqp(
        iqp_circuit, params, jnp.array(ops), jnp.array(coefs), moment, n_samples, key, indep_estimates
    )


def magnet_moment_samples(samples: np.ndarray, moment: int) -> tuple:
    """Compute magnetization moment from samples.

    Args:
        samples: Binary sample array (0/1).
        moment: Moment order.

    Returns:
        Tuple of (per_sample_moments, mean, standard_error).
    """
    samples = 1 - 2 * samples
    magnets = samples.sum(axis=1) / len(samples[0])
    mag_mom = magnets**moment
    return mag_mom, jnp.mean(mag_mom), jnp.std(mag_mom) / jnp.sqrt(len(mag_mom))


def energy_moment_samples(samples: np.ndarray, moment: int, j_matrix: np.ndarray) -> tuple:
    """Compute energy moment from samples.

    Args:
        samples: Binary sample array (0/1).
        moment: Moment order.
        j_matrix: Spin coupling matrix.

    Returns:
        Tuple of (per_sample_moments, mean, standard_error).
    """
    samples = 1 - 2 * samples
    energies = -((samples @ j_matrix) * samples).sum(axis=1) / 2
    en_mom = energies**moment
    return en_mom, jnp.mean(en_mom), jnp.std(en_mom) / jnp.sqrt(len(en_mom))

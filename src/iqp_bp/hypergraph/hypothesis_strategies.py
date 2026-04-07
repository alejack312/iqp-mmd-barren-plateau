"""Hypothesis strategies for property-based testing of hypergraph families.

Provides Hypothesis `st.composite` strategies that generate valid hypergraphs
for structured fuzz testing and property-based experiment sweeps.
"""

from __future__ import annotations

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from iqp_bp.mmd.mixture import dataset_expectations_batch


SMART_FAMILIES = ("product_state", "lattice", "erdos_renyi", "complete_graph")
SMALL_ANGLE_STDS = (0.01, 0.1, 0.3)


def _square_n_values(n_min: int, n_max: int) -> list[int]:
    side_min = int(np.ceil(np.sqrt(n_min)))
    side_max = int(np.floor(np.sqrt(n_max)))
    return [side * side for side in range(side_min, side_max + 1)]


def materialize_theta(
    G: np.ndarray,
    data: np.ndarray,
    scheme: str,
    rng: np.random.Generator | None = None,
    low: float = -np.pi,
    high: float = np.pi,
    std: float = 0.1,
    scale: float = 0.1,
) -> np.ndarray:
    """Construct theta from G, data, and an init configuration."""
    m = G.shape[0]
    if scheme == "uniform":
        if rng is None:
            raise ValueError("uniform init requires an RNG")
        return rng.uniform(low, high, size=m)
    if scheme == "small_angle":
        if rng is None:
            raise ValueError("small_angle init requires an RNG")
        return rng.normal(0.0, std, size=m)
    if scheme == "data_dependent":
        return scale * np.asarray(dataset_expectations_batch(data, G), dtype=np.float64)
    raise ValueError(f"Unknown scheme {scheme!r}")


@st.composite
def hypergraph_matrix(
    draw,
    n_min: int = 4,
    n_max: int = 16,
    m_min: int = 4,
    m_max: int = 32,
) -> tuple[np.ndarray, int, int]:
    """Draw a random binary hypergraph matrix G of shape (m, n).

    Returns:
        (G, n, m) where G is uint8 ndarray with no all-zero rows.
    """
    # TODO: Week 2 (D2.1) add strategies that sample the four SMART families
    # directly so validation covers the real experiment regimes, not just arbitrary G.
    # Read first: Hypothesis API https://hypothesis.readthedocs.io/en/latest/reference/api.html ;
    # strategies https://hypothesis.readthedocs.io/en/latest/reference/strategies.html ;
    # NumPy integration https://hypothesis.readthedocs.io/en/latest/numpy.html
    n = draw(st.integers(min_value=n_min, max_value=n_max))
    m = draw(st.integers(min_value=m_min, max_value=m_max))
    G = draw(
        arrays(
            dtype=np.uint8,
            shape=(m, n),
            elements=st.integers(min_value=0, max_value=1),
        )
    )
    # Ensure no all-zero generators
    empty_rows = G.sum(axis=1) == 0
    G[empty_rows, 0] = 1
    return G, n, m


@st.composite
def bounded_degree_hypergraph(
    draw,
    n_min: int = 4,
    n_max: int = 16,
    max_weight: int = 3,
) -> tuple[np.ndarray, int]:
    """Draw a k-local bounded-degree hypergraph."""
    from iqp_bp.hypergraph.families import bounded_degree
    n = draw(st.integers(min_value=n_min, max_value=n_max))
    m = draw(st.integers(min_value=n, max_value=2 * n))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)
    G = bounded_degree(n=n, m=m, max_weight=max_weight, rng=rng)
    return G, n


@st.composite
def iqp_parameters(
    draw,
    n_params: int,
    scheme: str = "uniform",
    std: float = 0.1,
    G: np.ndarray | None = None,
    data: np.ndarray | None = None,
    scale: float = 0.1,
) -> np.ndarray:
    """Draw IQP parameter vector θ under given initialization scheme."""
    # TODO: Week 2 (D2.1) add coverage for the data-dependent initializer and the
    # full small-angle sweep {0.01, 0.1, 0.3} required by the SMART spec.
    # Read first: Hypothesis API https://hypothesis.readthedocs.io/en/latest/reference/api.html ;
    # strategies https://hypothesis.readthedocs.io/en/latest/reference/strategies.html ;
    # NumPy integration https://hypothesis.readthedocs.io/en/latest/numpy.html
    if scheme == "uniform":
        return draw(
            arrays(
                dtype=np.float64,
                shape=(n_params,),
                elements=st.floats(min_value=-np.pi, max_value=np.pi),
            )
        )
    elif scheme == "small_angle":
        raw = draw(
            arrays(
                dtype=np.float64,
                shape=(n_params,),
                elements=st.floats(min_value=-4.0, max_value=4.0),
            )
        )
        return raw * std
    elif scheme == "data_dependent":
        if G is None or data is None:
            raise ValueError("data_dependent init requires both G and data")
        return materialize_theta(G=G, data=data, scheme=scheme, scale=scale)
    else:
        raise ValueError(f"Unknown scheme {scheme!r}")


@st.composite
def kernel_config(draw) -> tuple[str, dict]:
    """Draw a kernel configuration."""
    kernel = draw(st.sampled_from(["gaussian", "linear", "polynomial", "laplacian", "multi_scale_gaussian"]))
    if kernel == "gaussian":
        return ("gaussian", {"sigma": draw(st.sampled_from([0.5, 1.0, 2.0]))})
    if kernel == "linear":
        return ("linear", {})
    if kernel == "polynomial":
        return ("polynomial", {"degree": draw(st.integers(min_value=1, max_value=10)), "constant": draw(st.floats(min_value=0.01, max_value=10.0))})
    if kernel == "laplacian":
        return ("laplacian", {"sigma": draw(st.floats(min_value=0.01, max_value=10.0))})
    if kernel == "multi_scale_gaussian":
        sigmas = draw(
            st.lists(
                st.sampled_from([0.5, 1.0, 2.0]),
                min_size=2,
                max_size=3,
                unique=True,
            )
        )
        use_explicit_weights = draw(st.booleans())
        if use_explicit_weights:
            raw_weights = draw(
                st.lists(
                    st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
                    min_size=len(sigmas),
                    max_size=len(sigmas),
                )
            )
            total = sum(raw_weights)
            weights = [w / total for w in raw_weights]
        else:
            weights = None
        return ("multi_scale_gaussian", {"sigmas": sigmas, "weights": weights})
    raise ValueError(f"Unknown kernel {kernel!r}")

'''
 choices into separate strategies:                                                                                                                     
                                                                                                                                                                                  
  1. family_config()                                                                                                                                                              
     Returns (family_name, family_kwargs).                                                                                                                                        
     Start with just two families:                                                                                                                                                
      - bounded_degree                                                                                                                                                            
      - random_matrix via hypergraph_matrix()                                                                                                                                     
  2. init_config()                                                                                                                                                                
     Returns (scheme, scheme_kwargs).                                                                                                                                             
     Start with:                                                                                                                                                                  
      - uniform                                                                                                                                                                   
      - small_angle with std from {0.01, 0.1, 0.3}                                                                                                                                
  3. data_config()                                                                                                                                                                
     Returns (data_name, data_kwargs).                                                                                                                                            
     Start with:
      - product_bernoulli                                                                                                                                                         
      - maybe later a correlated target    
''' 

@st.composite
def family_config(draw):
    """Draw a family configuration."""
    family = draw(st.sampled_from(["bounded_degree", "hypergraph_matrix"]))
    if family == "bounded_degree":
        return ("bounded_degree", {})
    if family == "hypergraph_matrix":
        return ("hypergraph_matrix", {})
    raise ValueError(f"Unknown family {family!r}")

@st.composite
def smart_family_config(draw, family_name: str | None = None):
    """Draw a SMART family configuration."""
    family = family_name or draw(st.sampled_from(SMART_FAMILIES))
    if family == "product_state":
        return ("product_state", {})
    if family == "lattice":
        return ("lattice", {"dimension": 2, "range_": 1})
    if family == "erdos_renyi":
        return ("erdos_renyi", {"p_edge": draw(st.sampled_from([2.0, 4.0]))})
    if family == "complete_graph":
        return ("complete_graph", {})
    raise ValueError(f"Unknown family {family!r}")

@st.composite
def init_config(draw):
    """Draw an initialization configuration."""
    scheme = draw(st.sampled_from(["uniform", "small_angle", "data_dependent"]))
    if scheme == "uniform":
        return ("uniform", {})
    if scheme == "small_angle":
        return ("small_angle", {"std": draw(st.sampled_from(SMALL_ANGLE_STDS))})
    if scheme == "data_dependent":
        return ("data_dependent", {})
    raise ValueError(f"Unknown scheme {scheme!r}")


@st.composite
def smart_family_instance(
    draw,
    family_name: str | None = None,
    n_min: int = 4,
    n_max: int = 9,
):
    """Materialize a SMART family generator matrix."""
    from iqp_bp.hypergraph.families import make_hypergraph

    sampled_family, family_kwargs = draw(smart_family_config(family_name=family_name))
    if sampled_family == "lattice":
        square_values = _square_n_values(n_min=n_min, n_max=n_max)
        if not square_values:
            raise ValueError("No square n values available for SMART lattice sampling")
        n = draw(st.sampled_from(square_values))
        requested_m = n
    elif sampled_family == "erdos_renyi":
        n = draw(st.integers(min_value=max(n_min, 9), max_value=max(n_max, 9)))
        requested_m = draw(st.integers(min_value=n, max_value=2 * n))
    else:
        n = draw(st.integers(min_value=n_min, max_value=n_max))
        requested_m = n

    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)
    G = make_hypergraph(
        family=sampled_family,
        n=n,
        m=requested_m,
        rng=rng,
        **family_kwargs,
    )
    return {
        "family": sampled_family,
        "family_kwargs": family_kwargs,
        "n": n,
        "m": G.shape[0],
        "G": G,
    }


@st.composite
def data_config(draw) -> tuple[str, dict]:
    """Draw a data configuration."""
    data_family = draw(st.sampled_from(["product_bernoulli", "biased_product_bernoulli"]))
    if data_family == "product_bernoulli":
        return ("product_bernoulli", {})
    if data_family == "biased_product_bernoulli":
        return (
            "biased_product_bernoulli",
            {
                "p": draw(
                    st.floats(
                        min_value=0.1,
                        max_value=0.9,
                        allow_nan=False,
                        allow_infinity=False,
                    )
                )
            },
        )
    raise ValueError(f"Unknown data {data_family!r}")

@st.composite
def mmd_instance(draw):
    """Draw an MMD instance."""
    family_instance = draw(smart_family_instance())
    family_name = family_instance["family"]
    family_kwargs = family_instance["family_kwargs"]
    G = family_instance["G"]
    n = family_instance["n"]
    data_name, data_kwargs = draw(data_config())
    if data_name == "product_bernoulli":
        data = draw(arrays(dtype=np.uint8, shape=(128, n), elements=st.integers(min_value=0, max_value=1)))
    elif data_name == "biased_product_bernoulli":
        uniforms = draw(
            arrays(
                dtype=np.float64,
                shape=(128, n),
                elements=st.floats(
                    min_value=0.0,
                    max_value=1.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
            )
        )
        data = (uniforms < data_kwargs["p"]).astype(np.uint8)
    else:
        raise ValueError(f"Unknown data {data_name!r}")
    init_name, init_kwargs = draw(init_config())
    theta_seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    theta_rng = np.random.default_rng(theta_seed)
    theta = materialize_theta(G=G, data=data, scheme=init_name, rng=theta_rng, **init_kwargs)
    kernel, kernel_kwargs = draw(kernel_config())
    return {
        "family": family_name,
        "family_kwargs": family_kwargs,
        "init_scheme": init_name,
        "init_kwargs": init_kwargs,
        "data_family": data_name,
        "data_kwargs": data_kwargs,
        "kernel": kernel,
        "kernel_kwargs": kernel_kwargs,
        "G": G,
        "n": n,
        "m": G.shape[0],
        "theta": theta,
        "data": data
    }
    

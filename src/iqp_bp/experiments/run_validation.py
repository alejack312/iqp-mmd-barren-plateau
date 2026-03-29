"""Validation experiment runner.

Compares classical IQP estimator against:
  - Qiskit statevector (exact, noise-free)
  - Qiskit shot-based simulation
  - Qiskit Aer noise models
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from iqp_bp.hypergraph.families import make_hypergraph
from iqp_bp.mmd.gradients import estimate_gradient_variance
from iqp_bp.rng import split_seeds

log = logging.getLogger(__name__)


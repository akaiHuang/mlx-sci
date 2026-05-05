"""
mlx_sci.quantum — GPU-accelerated quantum information theory

No scipy equivalent — unique to this ecosystem.

Submodules
----------
Quantum Relative Entropy (mlx-qre):
    quantum_relative_entropy(rho, sigma)        — D(rho||sigma) = Tr[rho (ln rho - ln sigma)]
    quantum_relative_entropy_lanczos(rho, sigma, k=25, m=20)
                                                — O(k * N^2) Stochastic-Lanczos estimator,
                                                  beats eigh path by 1-2 orders at N >= 1000.
    von_neumann_entropy(rho)                    — S(rho) = -Tr[rho ln rho], eigh path.
    von_neumann_entropy_lanczos(rho, k, m)      — Hutchinson + Lanczos quadrature.
    stochastic_lanczos_logtr(A, k, m)           — Tr[ln A] estimator (the workhorse).
    matrix_log(A)
    kl_divergence(p, q)
    jensen_shannon_divergence(p, q)
    apply_channel(kraus, rho)
    thermal_attenuator(eta)
    depolarizing_channel(p)
    dephasing_channel(gamma)
    petz_recovery_map(kraus, sigma)
    petz_recovery_fidelity(kraus, rho, sigma)
    verify_petz_bound(kraus, rho, sigma)

Fisher Information (mlx-fisher):
    FisherMatrix
    fisher_matrix_cl(Cl, dCl_dtheta)
    NaturalGradientOptimizer

Circuit Simulation (mlx-quantum-sim):
    MLXQuantumSimulator   — statevector simulator (ideal + noisy)
    MLXBatchSimulator     — batched parallel circuits
    noise_profiles        — WILLOW_NOISE, HERON_NOISE, T9_NOISE
"""

from mlx_qre import (
    quantum_relative_entropy,
    matrix_log,
    is_density_matrix,
    random_density_matrix,
    von_neumann_entropy,
    kl_divergence,
    jensen_shannon_divergence,
    apply_channel,
    channel_entropy_production,
    thermal_attenuator,
    depolarizing_channel,
    dephasing_channel,
    petz_recovery_map,
    petz_recovery_fidelity,
    verify_petz_bound,
    # Stochastic Lanczos (new in mlx-qre 0.2.0)
    quantum_relative_entropy_lanczos,
    von_neumann_entropy_lanczos,
    stochastic_lanczos_logtr,
)

from mlx_fisher import (
    FisherMatrix,
    fisher_matrix_cl,
    NaturalGradientOptimizer,
)

try:
    from mlx_quantum_sim import MLXQuantumSimulator, MLXBatchSimulator
    from mlx_quantum_sim import noise_profiles
    _HAS_QUANTUM_SIM = True
except ImportError:
    # mlx-quantum-sim is an optional extra (`pip install mlx-sci[sim]`).
    # Circuit-simulation symbols are unavailable when it isn't installed.
    _HAS_QUANTUM_SIM = False
    MLXQuantumSimulator = None
    MLXBatchSimulator = None
    noise_profiles = None

__all__ = [
    # QRE core
    "quantum_relative_entropy",
    "matrix_log",
    "is_density_matrix",
    "random_density_matrix",
    "von_neumann_entropy",
    # Stochastic Lanczos estimators (mlx-qre >= 0.2.0)
    "quantum_relative_entropy_lanczos",
    "von_neumann_entropy_lanczos",
    "stochastic_lanczos_logtr",
    # Classical divergences
    "kl_divergence",
    "jensen_shannon_divergence",
    # Quantum channels
    "apply_channel",
    "channel_entropy_production",
    "thermal_attenuator",
    "depolarizing_channel",
    "dephasing_channel",
    # Petz recovery
    "petz_recovery_map",
    "petz_recovery_fidelity",
    "verify_petz_bound",
    # Fisher information
    "FisherMatrix",
    "fisher_matrix_cl",
    "NaturalGradientOptimizer",
    # Circuit simulation
    "MLXQuantumSimulator",
    "MLXBatchSimulator",
    "noise_profiles",
]

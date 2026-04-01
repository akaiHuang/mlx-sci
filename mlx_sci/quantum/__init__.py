"""
mlx_special.quantum — GPU-accelerated quantum information theory

No scipy equivalent — unique to this ecosystem.

Submodules
----------
Quantum Relative Entropy (mlx-qre):
    quantum_relative_entropy(rho, sigma)  — D(ρ||σ) = Tr[ρ(ln ρ - ln σ)]
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
)

from mlx_fisher import (
    FisherMatrix,
    fisher_matrix_cl,
    NaturalGradientOptimizer,
)

from mlx_quantum_sim import MLXQuantumSimulator, MLXBatchSimulator
from mlx_quantum_sim import noise_profiles

__all__ = [
    # QRE core
    "quantum_relative_entropy",
    "matrix_log",
    "is_density_matrix",
    "random_density_matrix",
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

"""
mlx-sci: The missing scipy toolkit for Apple Silicon
=====================================================

GPU-accelerated scientific computing via Apple MLX.

Submodules
----------
    mlx_sci.special   — Special functions  (≈ scipy.special)
    mlx_sci.linalg    — Matrix functions   (≈ scipy.linalg)
    mlx_sci.signal    — Signal processing  (≈ scipy.signal)
    mlx_sci.quantum   — Quantum information (unique)

Quick start
-----------
    from mlx_sci import special, linalg, signal, quantum

    # Airy functions
    Ai, Ai_prime, Bi, Bi_prime = special.airy(x)

    # Gamma functions
    y = special.gamma(x)

    # Hypergeometric
    y = special.hyp2f1(a, b, c, z)

    # Matrix exponential
    U = linalg.expm(-1j * H * t)

    # STFT
    spectrogram = signal.stft(audio)

    # Quantum relative entropy
    sigma = quantum.quantum_relative_entropy(rho, rho_ref)

    # Petz recovery bound
    ok = quantum.verify_petz_bound(kraus, rho, sigma)

    # Circuit simulation
    sim = quantum.MLXQuantumSimulator(3)
    sim.h(0); sim.cx(0, 1); sim.cx(0, 2)
    probs = sim.measure_probs()
"""

from mlx_sci import special, linalg, signal, quantum

__version__ = "0.1.0"
__author__ = "Sheng-Kai Huang"

__all__ = ["special", "linalg", "signal", "quantum"]

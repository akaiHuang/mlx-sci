"""
mlx-sci: The missing scipy toolkit for Apple Silicon
=====================================================

GPU-accelerated scientific computing via Apple MLX.

Submodules
----------
    mlx_sci.special   -- Special functions   (~ scipy.special)
    mlx_sci.linalg    -- Matrix functions    (~ scipy.linalg)
    mlx_sci.signal    -- Signal processing   (~ scipy.signal)
    mlx_sci.quantum   -- Quantum information (unique to this ecosystem)

Quick start
-----------
    import mlx.core as mx
    from mlx_sci import special, linalg, signal, quantum

    # Airy functions
    Ai, Ai_prime, Bi, Bi_prime = special.airy(x)

    # Gamma functions
    y = special.gamma(x)

    # Hypergeometric (auto-routes to a fused Metal kernel on Apple GPU)
    y = special.hyp2f1(a, b, c, z)

    # Matrix exponential
    U = linalg.expm(-1j * H * t)

    # STFT (class-based, from mlx-stft)
    stft = signal.STFT(n_fft=1024, hop_length=256, window=signal.hann_window(1024))
    spectrogram = stft(audio)

    # Quantum relative entropy (eigh path, exact)
    sigma = quantum.quantum_relative_entropy(rho, rho_ref)

    # Quantum relative entropy via Stochastic Lanczos quadrature
    # (O(k * N^2), wins by 1-2 orders at N >= 1000)
    sigma_lanczos = quantum.quantum_relative_entropy_lanczos(
        rho, rho_ref, k=25, m=20
    )

    # Petz recovery bound
    ok = quantum.verify_petz_bound(kraus, rho, sigma)

    # Circuit simulation
    sim = quantum.MLXQuantumSimulator(3)
    sim.h(0); sim.cx(0, 1); sim.cx(0, 2)
    probs = sim.measure_probs()
"""

from mlx_sci import special, linalg, signal, quantum

__version__ = "0.2.3"
__author__ = "Sheng-Kai Huang"

__all__ = ["special", "linalg", "signal", "quantum"]

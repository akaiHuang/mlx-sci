"""
mlx_sci.signal — GPU-accelerated signal processing (scipy.signal equivalent)

Class-based STFT / ISTFT layers from mlx-stft 0.1.2+.

Transforms
----------
    STFT(...)                 — Short-Time Fourier Transform layer
    ISTFT(...)                — Inverse STFT layer
    CompiledSTFT(...)         — mx.compile-fused STFT (faster, fixed shape)
    CompiledISTFT(...)        — mx.compile-fused inverse STFT
    AmpToDB(...)              — amplitude-to-dB conversion

Windows
-------
    hann_window(N)
    blackman_window(N)
    rect_window(N)
    get_window(name, N)
"""

from mlx_stft import STFT, ISTFT, CompiledSTFT, CompiledISTFT, AmpToDB
from mlx_stft.windows import (
    hann_window,
    blackman_window,
    rect_window,
    get_window,
)

__all__ = [
    "STFT",
    "ISTFT",
    "CompiledSTFT",
    "CompiledISTFT",
    "AmpToDB",
    "hann_window",
    "blackman_window",
    "rect_window",
    "get_window",
]

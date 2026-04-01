"""
mlx_special.signal — GPU-accelerated signal processing (scipy.signal equivalent)

Functions
---------
STFT / Spectrogram:
    stft(x, ...)              — Short-Time Fourier Transform
    istft(X, ...)             — Inverse STFT
    mel_spectrogram(x, ...)   — Mel spectrogram
    mel_filterbank(...)       — Mel filter bank matrix
    hz_to_mel(hz)
    mel_to_hz(mel)

Windows:
    hann_window(N)
    hamming_window(N)
    blackman_window(N)
    kaiser_window(N, beta)
    get_window(name, N)
"""

from mlx_stft import (
    stft,
    istft,
    mel_spectrogram,
    mel_filterbank,
    hz_to_mel,
    mel_to_hz,
    hann_window,
    hamming_window,
    blackman_window,
    kaiser_window,
    get_window,
)

__all__ = [
    "stft",
    "istft",
    "mel_spectrogram",
    "mel_filterbank",
    "hz_to_mel",
    "mel_to_hz",
    "hann_window",
    "hamming_window",
    "blackman_window",
    "kaiser_window",
    "get_window",
]

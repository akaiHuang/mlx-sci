"""
mlx_special.linalg — GPU-accelerated matrix functions (scipy.linalg equivalent)

Functions
---------
    expm(A)               — Matrix exponential via [13/13] Padé approximant
    expm_frechet(A, E)    — Fréchet derivative of the matrix exponential
    logm(A)               — Matrix logarithm (inverse scaling + squaring)
    sqrtm(A)              — Matrix square root (Denman-Beavers iteration)
"""

from mlx_expm import expm, expm_frechet, logm, sqrtm

__all__ = ["expm", "expm_frechet", "logm", "sqrtm"]

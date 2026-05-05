"""
mlx_sci.special — GPU-accelerated special functions (scipy.special equivalent)

Functions
---------
Airy:
    airy(x)                     -> (Ai, Ai', Bi, Bi')

Gamma:
    gamma(x), lgamma(x)
    digamma(x), beta(a, b)

Hypergeometric:
    hyp2f1(a, b, c, z)
    hyp1f1(a, b, z)
    hyp0f1(b, z)

    Note: as of mlx-hyp2f1 0.2.0 the user-facing `hyp2f1` / `hyp1f1` /
    `hyp0f1` automatically route to a fused Metal kernel when
    `mx.metal.is_available()`, falling back to the pure-MLX op path
    otherwise. There is no separate `hyp2f1_metal` symbol to call -- the
    routing is transparent. The fused path collapses ~200 per-element
    MLX ops into a single dispatch and gives the headline 7.4x speedup
    over SciPy at N=1M.

Bessel:
    BesselTable(ell_values, ...)  -- spherical j_l(x) and j_l'(x)

Angular momentum:
    wigner_3j(j1, j2, j3, m1, m2, m3)
    wigner_6j(j1, j2, j3, j4, j5, j6)
    wigner_9j(...)
    clebsch_gordan(j1, m1, j2, m2, J, M)
"""

from mlx_airy import airy

from mlx_gamma import gamma, lgamma, digamma, beta

from mlx_hyp2f1 import hyp2f1, hyp1f1, hyp0f1

from mlx_bessel import BesselTable

from mlx_wigner import wigner_3j, wigner_6j, wigner_9j, clebsch_gordan

__all__ = [
    # Airy
    "airy",
    # Gamma
    "gamma",
    "lgamma",
    "digamma",
    "beta",
    # Hypergeometric (Metal kernel auto-routing as of mlx-hyp2f1 0.2.0)
    "hyp2f1",
    "hyp1f1",
    "hyp0f1",
    # Bessel
    "BesselTable",
    # Angular momentum / Wigner
    "wigner_3j",
    "wigner_6j",
    "wigner_9j",
    "clebsch_gordan",
]

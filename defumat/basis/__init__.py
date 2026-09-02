"""The plane-wave basis: G-vectors, FFT grids, and per-k plane-wave sets.

Built once during setup from integer data (Miller indices, FFT dimensions,
index maps) that stays fixed for a run. Everything with a physical dimension is
derived from the cell on demand, so the cell remains differentiable.
"""

from defumat.basis.fft import g_to_r, gather_from_box, r_to_g, scatter_to_box
from defumat.basis.fftgrid import (
    allowed_fft_size,
    fft_grid_dimensions,
    gcut_from_ecut,
    good_fft_order,
)
from defumat.basis.gvectors import GVectors, generate_gvectors
from defumat.basis.planewaves import PlaneWaveBasis, build_plane_wave_basis

__all__ = [
    "GVectors",
    "PlaneWaveBasis",
    "allowed_fft_size",
    "build_plane_wave_basis",
    "fft_grid_dimensions",
    "g_to_r",
    "gather_from_box",
    "gcut_from_ecut",
    "generate_gvectors",
    "good_fft_order",
    "r_to_g",
    "scatter_to_box",
]

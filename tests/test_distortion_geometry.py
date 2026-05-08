"""Tests for the distortion-tensor irreducible decomposition utilities.

Covers the round-trip identity (parts recombine to give D), the
identity-matrix sanity check, the spin-2 rotation transformation
law, principal-axis range / magnitude consistency, and the complex
round trip.

References
----------
See the :mod:`mtpy.core.transfer_function.z_analysis.decomposition.
distortion_geometry` module docstring for the physical interpretation
and citations.
"""

from __future__ import annotations

import numpy as np

from mtpy.core.transfer_function.z_analysis.decomposition import (
    complex_to_gamma,
    gamma_field,
    gamma_magnitude,
    gamma_to_complex,
    irreducible_decomposition,
    principal_axis,
)


def test_irreducible_invariance():
    """``trace_a + gamma_1 + gamma_2 + beta`` recombine to give ``D``.

    Verifies the four-component decomposition is bijective: the
    diagonal entries are ``trace_a +/- gamma_1`` and the
    off-diagonals are ``gamma_2 +/- beta``.
    """
    rng = np.random.default_rng(0)
    D = rng.standard_normal((2, 2))
    parts = irreducible_decomposition(D)

    a = parts["trace_a"]
    g1 = parts["gamma_1"]
    g2 = parts["gamma_2"]
    b = parts["beta"]
    D_recombined = np.array(
        [
            [a + g1, g2 + b],
            [g2 - b, a - g1],
        ]
    )
    assert np.allclose(D, D_recombined), (
        f"recombination failed:\nD =\n{D}\nrecombined =\n{D_recombined}"
    )


def test_gamma_field_for_identity():
    """``C = I`` gives ``gamma_field`` of ``(0, 0)``."""
    C = np.eye(2)
    g1, g2 = gamma_field(C)
    assert g1 == 0.0
    assert g2 == 0.0


def test_gamma_rotation_transformation():
    """Under rotation by ``theta``, ``gamma`` transforms as
    ``exp(2 i theta) * gamma``.

    This is the defining property of the spin-2 representation.
    Build a ``C`` with non-trivial ``gamma``, rotate to get
    ``C' = R(theta) C R(theta).T``, recompute ``gamma'``, and
    verify ``gamma' == exp(2 i theta) * gamma`` to numerical
    precision.
    """
    C = np.array([[1.2, 0.1], [0.05, 0.8]])
    g1, g2 = gamma_field(C)
    g_complex = gamma_to_complex(g1, g2)

    theta = np.radians(30.0)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    C_rot = R @ C @ R.T
    g1_rot, g2_rot = gamma_field(C_rot)
    g_complex_rot = gamma_to_complex(g1_rot, g2_rot)

    expected = np.exp(2j * theta) * g_complex
    assert np.isclose(g_complex_rot, expected), (
        f"rotation law failed: got {g_complex_rot}, expected {expected}"
    )


def test_principal_axis_consistency():
    """``principal_axis`` returns an angle in ``[0, pi)`` and a
    magnitude that matches :func:`gamma_magnitude`.
    """
    rng = np.random.default_rng(1)
    C = np.eye(2) + 0.1 * rng.standard_normal((2, 2))
    psi, mag = principal_axis(C)
    assert 0.0 <= psi < np.pi, (
        f"principal-axis angle {psi} not in [0, pi)"
    )
    assert np.isclose(mag, gamma_magnitude(C)), (
        f"principal-axis magnitude {mag} != gamma_magnitude(C) "
        f"{gamma_magnitude(C)}"
    )


def test_complex_round_trip():
    """``gamma_to_complex(complex_to_gamma(g))`` returns ``g``."""
    g = 0.3 + 0.4j
    g1, g2 = complex_to_gamma(g)
    g_back = gamma_to_complex(g1, g2)
    assert g_back == g, f"round trip failed: {g_back} != {g}"

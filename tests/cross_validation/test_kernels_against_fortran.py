"""Cross-validation of pure-Python kernels against the strike_py
Fortran reference.

Tests in this module compare the pure-Python implementations in
:mod:`mtpy.core.transfer_function.z_analysis.decomposition` against
the f2py-wrapped Fortran kernels in the strike_py development repo
(separate clone at ``~/MT_Decomp`` by default; override via
``MT_DECOMP_PATH`` environment variable).

These tests are local-developer-only. They skip cleanly when the
strike_py reference is not present, so CI can run without it.

The skip pattern: at module import time, attempt to add
``MT_DECOMP_PATH`` to ``sys.path`` and import ``strike_py.kernels``.
If either step fails, the entire module is marked skip.

Why no ``_objfun`` cross-validation. The Fortran adapter
``objfun_wrapped`` and the Python ``_objfun`` parameterise the
GB optimisation problem differently:

- Fortran x is ``[re(a_k), im(a_k), re(b_k), im(b_k) per freq,
  tan(twist), tan(shear), theta]`` (size ``4*n_freqs + 3``); no
  gain or anisotropy parameter.
- Python x is ``[theta, twist, shear, log10_gain, anisotropy,
  log10_rho_a, phase_a, log10_rho_b, phase_b]`` (size
  ``5 + 4*n_freqs``).

Furthermore, the Fortran computes residuals in alpha-space
(Pauli-spin combinations) with a deliberately non-standard sigma
weighting (raw tensor-component sigma used for the combined
alphas; see the strike_py forensic notes). Element-wise comparison
of residual vectors or Jacobians is therefore not meaningful. The
forward-model continuity link is provided by the ``_estim_imp``
cross-validation below.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from mtpy.core.transfer_function.z_analysis.decomposition import (
    _calc_error,
    _convz2p,
    _convz2r,
    _estim_imp,
    _extreme,
    _jkvar,
    _mat_multiply,
)


MT_DECOMP_PATH = os.environ.get(
    "MT_DECOMP_PATH",
    os.path.expanduser("~/MT_Decomp"),
)

_HAS_REFERENCE = False
_f_mat_multiply = None
_f_extreme = None
_f_convz2r = None
_f_convz2p = None
_f_calc_error = None
_f_jkvar = None
_f_estim_imp = None

if os.path.isdir(MT_DECOMP_PATH):
    if MT_DECOMP_PATH not in sys.path:
        sys.path.insert(0, MT_DECOMP_PATH)
    try:
        from strike_py.kernels import calc_error as _f_calc_error
        from strike_py.kernels import convz2p as _f_convz2p
        from strike_py.kernels import convz2r as _f_convz2r
        from strike_py.kernels import estim_imp as _f_estim_imp
        from strike_py.kernels import extreme as _f_extreme
        from strike_py.kernels import jkvar as _f_jkvar
        from strike_py.kernels import mat_multiply as _f_mat_multiply

        _HAS_REFERENCE = True
    except ImportError:
        pass


pytestmark = pytest.mark.skipif(
    not _HAS_REFERENCE,
    reason=(
        f"strike_py Fortran reference not available at "
        f"{MT_DECOMP_PATH}. Set MT_DECOMP_PATH or skip "
        f"cross-validation."
    ),
)


# Tolerances for cross-validation. These are kernel-level checks
# where agreement is bounded below by the Fortran reference's
# numeric precision, not by the Python implementation's. The Fortran
# sources declare types as follows:
#
#   extreme.f, convz2r.f, convz2p.f, jkvar.f : ``real`` (float32)
#   estim_imp.f, mat_multiply.f               : ``real*8`` /
#                                                ``complex*16`` (float64)
#
# So _RTOL_F32 is used for the single-precision kernels (about 7
# decimal digits of agreement is the best the Fortran can give us)
# and _RTOL_F64 for the double-precision ones.
_RTOL_F32 = 1e-6
_RTOL_F64 = 1e-12
_ATOL = 1e-15


class TestMatMultiplyAgainstFortran:
    def test_known_complex_product(self):
        rng = np.random.default_rng(42)
        a = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        b = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        py_result = _mat_multiply(a, b)
        # strike_py wrapper signature: mat_multiply(mat1, mat2, [n])
        f_result = _f_mat_multiply(a, b)
        np.testing.assert_allclose(py_result, f_result, rtol=_RTOL_F64, atol=_ATOL)


class TestExtremeAgainstFortran:
    def test_random_array(self):
        rng = np.random.default_rng(123)
        x = rng.normal(size=50)
        py_min, py_max = _extreme(x)
        # strike_py wrapper signature: extreme(vect, [len_bn])
        # returns (vmax, vmin) -- max first. The Fortran helper is
        # single-precision; values get truncated at the f2py
        # boundary, so equality is at float32 ULP, not bitwise.
        f_max, f_min = _f_extreme(x)
        np.testing.assert_allclose(py_min, f_min, rtol=_RTOL_F32)
        np.testing.assert_allclose(py_max, f_max, rtol=_RTOL_F32)


class TestConvz2rAgainstFortran:
    @pytest.mark.parametrize(
        "z,period",
        [
            (complex(1e-3, 5e-4), 100.0),
            (complex(2e-3, -1e-3), 1.0),
            (complex(1e-4, 0), 1000.0),
        ],
    )
    def test_known_z(self, z, period):
        py_result = _convz2r(z, period)
        f_result = _f_convz2r(z, period)
        np.testing.assert_allclose(py_result, f_result, rtol=_RTOL_F32)


class TestConvz2pAgainstFortran:
    @pytest.mark.parametrize("deg", [0.0, 30.0, 45.0, 60.0, 89.0, -45.0, 135.0])
    def test_known_angles(self, deg):
        rad = np.radians(deg)
        z = complex(np.cos(rad), np.sin(rad))
        py_result = _convz2p(z, 1.0)
        f_result = _f_convz2p(z, 1.0)
        # convz2p is single-precision Fortran; absolute tolerance
        # of ~1e-4 degrees corresponds to float32 ULP near unit
        # circle inputs.
        np.testing.assert_allclose(py_result, f_result, atol=1e-4)


class TestCalcErrorAgainstFortran:
    def test_random_tensors(self):
        rng = np.random.default_rng(7)
        z_data = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        z_pred = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        sigma = np.abs(rng.normal(size=(2, 2))) + 0.1
        py_result = _calc_error(z_data, z_pred, sigma)
        f_result = _f_calc_error(z_data, z_pred, sigma)
        np.testing.assert_allclose(py_result, f_result, rtol=_RTOL_F64)


class TestJkvarAgainstFortran:
    def test_random_sample(self):
        rng = np.random.default_rng(11)
        x = rng.normal(size=30)
        py_result = _jkvar(x)
        # strike_py wrapper signature: jkvar(x, [n]) -- n optional.
        # jkvar.f is single-precision; expect float32 agreement.
        f_result = _f_jkvar(x)
        np.testing.assert_allclose(py_result, f_result, rtol=_RTOL_F32)

    def test_n_too_small_sentinel(self):
        x = np.array([5.0])
        assert _jkvar(x) == -1.0
        f_result = _f_jkvar(x)
        assert f_result == -1.0


class TestEstimImpAgainstFortran:
    """The Fortran's ``estim_imp(gamma1, gamma2, a, b, theta)``
    (with ``gamma1=shear_tan``, ``gamma2=twist_tan``) is the
    cornerstone of cross-validation. The cleaner Python form is
    checked against it for several parameter combinations."""

    @pytest.mark.parametrize(
        "a,b,theta_deg,twist_deg,shear_deg",
        [
            # No distortion, no rotation
            (complex(1e-3, 5e-4), complex(8e-4, 3e-4), 0, 0, 0),
            # Pure rotation
            (complex(1e-3, 5e-4), complex(8e-4, 3e-4), 30, 0, 0),
            # Pure twist
            (complex(1e-3, 5e-4), complex(8e-4, 3e-4), 0, 15, 0),
            # Pure shear
            (complex(1e-3, 5e-4), complex(8e-4, 3e-4), 0, 0, 10),
            # All together (the realistic case)
            (complex(1e-3, 5e-4), complex(8e-4, 3e-4), 30, 15, 10),
            # Larger angles
            (complex(2e-3, -1e-3), complex(1.5e-3, 5e-4), 75, 30, 25),
        ],
    )
    def test_against_fortran(self, a, b, theta_deg, twist_deg, shear_deg):
        theta = np.radians(theta_deg)
        t = np.tan(np.radians(twist_deg))
        e = np.tan(np.radians(shear_deg))

        py_z = _estim_imp(a, b, t, e, theta)
        # Fortran signature: estim_imp(gamma1=e, gamma2=t, a, b, theta).
        f_z = _f_estim_imp(e, t, a, b, theta)

        np.testing.assert_allclose(py_z, f_z, rtol=_RTOL_F64, atol=_ATOL)

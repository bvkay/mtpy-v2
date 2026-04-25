"""End-to-end continuity validation against the Fortran reference.

Runs the pure-Python ``decompose()`` on the strike_py canonical example
data and validates against the Fortran's converged parameters.

The reference data lives in the strike_py development repo at
``~/MT_Decomp`` (override via the ``MT_DECOMP_PATH`` environment
variable). Tests skip cleanly when that repo is absent, so CI can run
without it.

Comparison strategy. The Python ``decompose()`` reports ``strike`` in
``[0, 90)`` after folding through the GB ``(strike + 90, -shear)``
symmetry; the Fortran ``.dcmp`` reference reports ``strike`` in
``[0, 180)``. We compare per period using the symmetry-aware helper
from ``strike_py.symmetry`` (which tries both branches and takes the
smaller error) so that opposite-branch but physically equivalent
solutions still match.

Tolerances follow Task 7 / CLAUDE.md invariant 4 of the strike_py
project::

    azimuth     ±1.0 deg
    twist, shear ±0.5 deg
    log10 rho   ±0.01
    phases      ±0.5 deg
    RMS misfit  ±5% relative

This module is local-developer-only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from mtpy.core.transfer_function.z import Z
from mtpy.core.transfer_function.z_analysis.decomposition import decompose


MT_DECOMP_PATH = os.environ.get(
    "MT_DECOMP_PATH",
    os.path.expanduser("~/MT_Decomp"),
)

_STRIKE_EXAMPLE_DAT = os.path.join(
    MT_DECOMP_PATH, "strike61_xnag", "example", "strike_example.dat"
)
_STRIKE_EXAMPLE_DCMP = os.path.join(
    MT_DECOMP_PATH, "strike61_xnag", "example", "strike_example.dcmp"
)

_HAS_DATA = os.path.isfile(_STRIKE_EXAMPLE_DAT)
_HAS_REFERENCE_DCMP = os.path.isfile(_STRIKE_EXAMPLE_DCMP)

_HAS_STRIKE_PY = False
if os.path.isdir(MT_DECOMP_PATH):
    if MT_DECOMP_PATH not in sys.path:
        sys.path.insert(0, MT_DECOMP_PATH)
    try:
        from strike_py.io.dcmp import read_dcmp  # noqa: F401
        from strike_py.io.j_format import read_j  # noqa: F401

        _HAS_STRIKE_PY = True
    except ImportError:
        pass


pytestmark = pytest.mark.skipif(
    not (_HAS_DATA and _HAS_STRIKE_PY),
    reason=(
        "strike_py / strike_example.dat not available at "
        f"{MT_DECOMP_PATH}; continuity test is "
        "local-developer-only"
    ),
)


def _load_strike_example_z() -> Z:
    """Read strike_example.dat through strike_py's J-format reader and
    return as an mtpy-v2 Z object."""
    from strike_py.io.j_format import read_j

    md = read_j(_STRIKE_EXAMPLE_DAT)
    frequencies = 1.0 / md.periods
    return Z(z=md.z, z_error=md.sigma, frequency=frequencies)


def _circular_az_err_deg(rec_deg: float, ref_deg: float) -> float:
    """Smallest mod-180 distance between two azimuths in degrees."""
    diff = (rec_deg - ref_deg) % 180.0
    return float(min(diff, 180.0 - diff))


def _symmetry_aware_match(
    rec_strike_deg: float,
    rec_twist_deg: float,
    rec_shear_deg: float,
    ref_strike_deg: float,
    ref_twist_deg: float,
    ref_shear_deg: float,
) -> tuple[float, float, float, bool]:
    """Try both branches of the GB symmetry and pick the smaller-error
    one. Returns ``(az_err, twist_err, shear_err, flipped)``.

    Branch A: identity. Branch B: ``(strike+90 mod 180, -shear)``.
    Twist is invariant under the symmetry.
    """
    # Branch A
    az_a = _circular_az_err_deg(rec_strike_deg, ref_strike_deg)
    tw_a = abs(rec_twist_deg - ref_twist_deg)
    sh_a = abs(rec_shear_deg - ref_shear_deg)
    score_a = az_a + tw_a + sh_a
    # Branch B
    az_b = _circular_az_err_deg(rec_strike_deg + 90.0, ref_strike_deg)
    tw_b = abs(rec_twist_deg - ref_twist_deg)
    sh_b = abs(rec_shear_deg - (-ref_shear_deg))
    score_b = az_b + tw_b + sh_b
    if score_b < score_a:
        return az_b, tw_b, sh_b, True
    return az_a, tw_a, sh_a, False


class TestStrikeExampleContinuity:
    """Sanity checks against the strike_example reference."""

    def test_data_loads(self):
        z = _load_strike_example_z()
        assert isinstance(z, Z)
        assert z.z.shape[0] > 0
        assert z.z.shape[1:] == (2, 2)
        assert z.z_error is not None
        assert np.all(z.z_error > 0)

    def test_decompose_runs(self):
        z = _load_strike_example_z()
        result = decompose(z)
        assert result.method == "groom_bailey"
        assert result.frame == "measurement"
        assert result.metadata["n_bands"] >= 1
        # Strike values should all be finite for converged bands
        n_finite = np.sum(np.isfinite(result.parameters["strike"].values))
        assert n_finite > 0
        # Final RMS should be finite (not inf or NaN)
        assert np.isfinite(result.rms_misfit)

    def test_strikes_in_canonical_range(self):
        z = _load_strike_example_z()
        result = decompose(z)
        strikes = result.parameters["strike"].values
        finite = np.isfinite(strikes)
        # Canonicalised range is [0, 90).
        assert np.all(strikes[finite] >= 0.0)
        assert np.all(strikes[finite] < 90.0 + 1e-6)

    def test_regional_z_is_finite(self):
        z = _load_strike_example_z()
        result = decompose(z)
        rz = result.regional_z.z
        # Regional Z should be finite where the optimisation
        # converged. Allow nans only at edge periods.
        finite_count = np.sum(np.isfinite(rz))
        assert finite_count >= 0.5 * rz.size

    @pytest.mark.skipif(
        not _HAS_REFERENCE_DCMP,
        reason="strike_example.dcmp not present; skipping comparison",
    )
    def test_compares_to_dcmp_reference(self):
        """Symmetry-aware comparison against the Fortran .dcmp.

        The Fortran reference uses bands defined by its own
        partitioning algorithm. The Python uses ``_extract_bands`` with
        a default ``bandwidth=1.0`` decade. Band boundaries differ, so
        we compare per-period after nearest-neighbour matching.

        At time of writing this is a *qualitative* check rather than a
        Task 7 tolerance assertion, because:

          - Python ``decompose()`` does single-start optimisation; some
            bands can converge to local minima that disagree with the
            Fortran reference even though both fit the data well.
          - Band partitioning differs.
          - The Fortran's ``norm_type`` and the Python's residual
            weighting are not byte-identical.

        We assert: median azimuth error < 10 deg, twist median error
        < 5 deg, shear median error < 5 deg. Tighter Task 7 assertions
        await a multi-start optimiser and matched band partitioning,
        which are out of scope for this session.
        """
        from strike_py.io.dcmp import read_dcmp

        z = _load_strike_example_z()
        result = decompose(z)
        ref = read_dcmp(_STRIKE_EXAMPLE_DCMP)

        py_periods = result.parameters["period"].values
        py_strike = result.parameters["strike"].values
        py_twist = result.parameters["twist"].values
        py_shear = result.parameters["shear"].values

        az_errs = []
        tw_errs = []
        sh_errs = []
        for k, period in enumerate(py_periods):
            if not np.isfinite(py_strike[k]):
                continue
            ref_idx = int(np.argmin(np.abs(ref.periods - period)))
            az_err, tw_err, sh_err, _ = _symmetry_aware_match(
                py_strike[k],
                py_twist[k],
                py_shear[k],
                float(ref.regional_azimuth_deg[ref_idx]),
                float(ref.twist_deg[ref_idx]),
                float(ref.shear_deg[ref_idx]),
            )
            az_errs.append(az_err)
            tw_errs.append(tw_err)
            sh_errs.append(sh_err)

        assert len(az_errs) > 0, "no finite strikes to compare against reference"
        median_az = float(np.median(az_errs))
        median_tw = float(np.median(tw_errs))
        median_sh = float(np.median(sh_errs))
        # Regression-detection bounds. Session 4 (single-start TRF)
        # produced median azimuth error ~9.9 deg here; Session 5
        # multi-start (default n_starts=5) tightens the median to
        # ~6.3 deg. The maximum error is still ~33 deg at the long-
        # period bands, where Python and Fortran converge to
        # different valid local minima — that residual gap will
        # close only with matched band partitioning (Session 7) and
        # tighter optimiser tolerances (out of scope here). Tighter
        # Task 7 assertions (azimuth +-1 deg) likewise await those.
        assert median_az < 10.0, (
            f"median azimuth error {median_az:.2f} deg too large "
            f"(regression bound 10 deg; Session 5 baseline ~6.3 deg)"
        )
        assert median_tw < 5.0, f"median twist error {median_tw:.2f} deg too large"
        assert median_sh < 5.0, f"median shear error {median_sh:.2f} deg too large"

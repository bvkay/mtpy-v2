"""Synthetic test harness for the decomposition package.

Generates ground-truth synthetic ``Z(omega)`` data with a known
distortion tensor, runs every available decomposition method on it,
and computes a per-method recovery accuracy. Used for cross-method
validation: any method that returns a galvanic distortion ``C``
should recover the ground-truth ``C`` within tolerance on a clean
synthetic, and the harness makes that comparison uniform.

This module is *not* a pytest test file — its filename
intentionally does not match ``test_*`` so pytest does not collect
it. The caller is :mod:`tests.test_synthetics_harness`, which uses
these utilities and is itself ``@pytest.mark.slow``.

Forward model
-------------
The harness uses the GB89 (Pauli) form for the distortion matrix
matching the optimiser in :mod:`...decomposition.groom_bailey`:

    C(strike, twist, shear, gain) =
        gain * R(strike) * T(twist) * S(shear) * R(strike).T

with ``T(twist) = [[1, -tan(twist)], [tan(twist), 1]]`` and
``S(shear) = [[1, tan(shear)], [tan(shear), 1]]``. Both ``T`` and
``S`` are *unnormalised* — this is the same convention used by the
GB single-site optimiser (see :func:`._estim_imp`), so that the
harness's synthetic agrees with the optimiser's forward model and
the recovered ``(strike, twist, shear)`` reconstruct the same
``C`` we put in.

Regional model approximations
-----------------------------
The harness offers three regional types:

* ``"1D"``: ``Z`` is anti-diagonal in any frame with ``Z_TE =
  Z_TM`` (single complex impedance per period). Strictly
  rotationally invariant — the strike of a 1-D Earth is undefined,
  and decomposition methods are expected to fail / be unstable in
  this regime. Used here to verify that methods *don't* spuriously
  produce strikes / distortion angles when the regional is 1-D.
* ``"2D"``: ``Z`` is anti-diagonal in a strike frame with ``Z_TE
  != Z_TM``, rotated to the measurement frame by a known strike.
  This is the canonical decomposition target.
* ``"3D"``: a hand-crafted approximation of a 3-D regional that
  does **not** rely on any external 3-D forward solver (ModEM,
  MARE2DEM, etc.). Concretely we take the 2-D ``Z`` and add small
  Born-approximation-style diagonal entries proportional to the
  off-diagonals (~25 % of the off-diagonal magnitudes), simulating
  a localised 3-D conductivity perturbation on a 2-D host. Real
  3-D synthetics from a finite-difference solver would be more
  faithful but require external dependencies; this analytical
  approximation is sufficient to exercise the decomposition
  methods' graceful-degradation behaviour on a non-2-D regional.

The 3-D approximation is documented in the per-period dict's
``regional_type_notes`` field for traceability.

References
----------
Cagniard, L. (1953). Basic theory of the magnetotelluric method of
geophysical prospecting. Geophysics, 18, 605-635.

Groom, R. W., & Bailey, R. C. (1989). Decomposition of
magnetotelluric impedance tensors in the presence of local three-
dimensional galvanic distortion. Journal of Geophysical Research:
Solid Earth, 94(B2), 1913-1925.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mtpy.core.transfer_function.z import Z
from mtpy.core.transfer_function.z_analysis.decomposition import (
    decompose,
    decompose_bibby,
    decompose_lilley,
    decompose_marti,
)


# ---------------------------------------------------------------------------
# Level mappings
# ---------------------------------------------------------------------------


_DISTORTION_TWIST_DEG = {"weak": 5.0, "moderate": 15.0, "strong": 30.0}
_DISTORTION_SHEAR_DEG = {"low": 5.0, "moderate": 15.0, "high": 30.0}
_NOISE_FRACTION = {"clean": 0.0, "low": 0.01, "high": 0.05}
_REGIONAL_TYPES = {"1D", "2D", "3D"}


# ---------------------------------------------------------------------------
# Forward-model helpers
# ---------------------------------------------------------------------------


def _construct_C_gb89(
    strike_deg: float,
    twist_deg: float,
    shear_deg: float,
    gain: float = 1.0,
) -> np.ndarray:
    """Construct the ground-truth distortion ``C`` in GB89 form.

    Matches the optimiser's forward model exactly (see module
    docstring) so synthetic generation and decomposition recovery
    are consistent.
    """
    theta = np.radians(strike_deg)
    h = np.tan(np.radians(twist_deg))
    e = np.tan(np.radians(shear_deg))
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    T = np.array([[1.0, -h], [h, 1.0]])
    S = np.array([[1.0, e], [e, 1.0]])
    return gain * R @ T @ S @ R.T


def _build_regional_z(
    periods: np.ndarray,
    regional_type: str,
    strike_deg: float,
) -> np.ndarray:
    """Build the regional ``Z`` (no distortion, in measurement frame).

    1-D: anti-diagonal with ``Z_TE = Z_TM``.
    2-D: anti-diagonal with ``Z_TE != Z_TM`` rotated by ``strike_deg``.
    3-D: 2-D base plus 25 % diagonals (Born-approximation analogue).
    """
    omega = 2.0 * np.pi / np.asarray(periods)
    mu0 = 4.0 * np.pi * 1.0e-7

    rho_te, rho_tm = 100.0, 400.0
    phi_te_deg, phi_tm_deg = 60.0, 30.0

    if regional_type == "1D":
        rho_te = rho_tm = 0.5 * (rho_te + rho_tm)
        phi_te_deg = phi_tm_deg = 0.5 * (phi_te_deg + phi_tm_deg)

    z_te = np.sqrt(omega * mu0 * rho_te) * np.exp(1j * np.radians(phi_te_deg))
    z_tm = np.sqrt(omega * mu0 * rho_tm) * np.exp(1j * np.radians(phi_tm_deg))

    n = len(periods)
    z_strike = np.zeros((n, 2, 2), dtype=np.complex128)
    z_strike[:, 0, 1] = z_te
    z_strike[:, 1, 0] = -z_tm

    if regional_type == "3D":
        # Born-approximation analogue: small diagonals proportional to
        # the off-diagonals, with a phase shift that breaks the
        # 2-D-anti-diagonal symmetry.
        diag_fraction = 0.25
        phase_shift = np.radians(45.0)
        z_strike[:, 0, 0] = diag_fraction * z_te * np.exp(1j * phase_shift)
        z_strike[:, 1, 1] = -diag_fraction * z_tm * np.exp(-1j * phase_shift)

    theta = np.radians(strike_deg)
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    return np.einsum("ij,kjl,lm->kim", R, z_strike, R.T)


def _add_complex_noise(
    z: np.ndarray, fraction: float, rng: np.random.Generator
) -> np.ndarray:
    """Add complex Gaussian noise scaled by per-component magnitude."""
    if fraction <= 0:
        return z
    amp = fraction * np.abs(z)
    noise = (
        rng.standard_normal(z.shape) + 1j * rng.standard_normal(z.shape)
    ) / np.sqrt(2.0)
    return z + amp * noise


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def generate_synthetic_z(
    regional_type: str,
    distortion_strength: str,
    distortion_shear: str,
    noise_level: str,
    periods: np.ndarray,
    site_id: str,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate a synthetic ``Z(omega)`` with known distortion.

    Parameters
    ----------
    regional_type : {"1D", "2D", "3D"}
        Regional Earth model class. See module docstring for the
        3-D analytical approximation.
    distortion_strength : {"weak", "moderate", "strong"}
        Maps to GB twist angle: 5, 15, 30 degrees.
    distortion_shear : {"low", "moderate", "high"}
        Maps to GB shear angle: 5, 15, 30 degrees.
    noise_level : {"clean", "low", "high"}
        Maps to fractional Gaussian noise: 0, 1, 5 % of ``|Z|``.
    periods : ndarray
        Period grid in seconds.
    site_id : str
        Identifier for the synthetic site.
    seed : int, default 42
        RNG seed for the noise realisation.

    Returns
    -------
    dict
        Keys:

        * ``z_obj``: mtpy ``Z`` with the synthetic data.
        * ``true_C``: the ground-truth ``2x2`` distortion matrix in
          the measurement frame.
        * ``true_strike``: ground-truth regional strike in degrees.
        * ``true_twist``, ``true_shear``: ground-truth GB
          parameters (degrees).
        * ``true_gain``: ground-truth GB gain (1.0 in this Phase 1
          harness).
        * ``regional_type``, ``distortion_strength``,
          ``distortion_shear``, ``noise_level``, ``site_id``,
          ``seed``: input parameters echoed back.
        * ``periods``: the period grid.
        * ``regional_type_notes``: prose description of the
          regional Earth model used (3-D analytical approximation
          documented here).

    Raises
    ------
    ValueError
        On unrecognised level names.
    """
    if regional_type not in _REGIONAL_TYPES:
        raise ValueError(
            f"generate_synthetic_z: regional_type must be one of "
            f"{sorted(_REGIONAL_TYPES)}, got {regional_type!r}"
        )
    if distortion_strength not in _DISTORTION_TWIST_DEG:
        raise ValueError(
            f"generate_synthetic_z: distortion_strength must be one of "
            f"{sorted(_DISTORTION_TWIST_DEG)}, got {distortion_strength!r}"
        )
    if distortion_shear not in _DISTORTION_SHEAR_DEG:
        raise ValueError(
            f"generate_synthetic_z: distortion_shear must be one of "
            f"{sorted(_DISTORTION_SHEAR_DEG)}, got {distortion_shear!r}"
        )
    if noise_level not in _NOISE_FRACTION:
        raise ValueError(
            f"generate_synthetic_z: noise_level must be one of "
            f"{sorted(_NOISE_FRACTION)}, got {noise_level!r}"
        )

    periods = np.asarray(periods, dtype=np.float64)
    twist_deg = _DISTORTION_TWIST_DEG[distortion_strength]
    shear_deg = _DISTORTION_SHEAR_DEG[distortion_shear]
    noise_frac = _NOISE_FRACTION[noise_level]

    true_strike_deg = 30.0 if regional_type != "1D" else 0.0
    true_gain = 1.0

    z_regional = _build_regional_z(periods, regional_type, true_strike_deg)
    true_C = _construct_C_gb89(
        true_strike_deg, twist_deg, shear_deg, true_gain
    )

    z_obs = np.einsum("ij,kjl->kil", true_C, z_regional)
    rng = np.random.default_rng(seed)
    z_obs = _add_complex_noise(z_obs, noise_frac, rng)

    # Always populate z_error with a small positive floor so the
    # downstream optimisers don't reject the input.
    sigma = np.maximum(max(noise_frac, 0.005) * np.abs(z_obs), 1e-9)
    z_obj = Z(z=z_obs, z_error=sigma, frequency=1.0 / periods)

    notes = {
        "1D": "Layered Earth, anti-diagonal Z with Z_TE = Z_TM.",
        "2D": (
            f"2-D Earth, anti-diagonal in strike frame "
            f"(rho_TE=100, rho_TM=400 ohm-m; phi_TE=60, phi_TM=30 deg), "
            f"rotated by {true_strike_deg} deg."
        ),
        "3D": (
            "2-D base with 25%% diagonals at +/- 45 deg phase shift "
            "(Born-approximation analogue for a localised 3-D body; "
            "no external 3-D solver used)."
        ),
    }[regional_type]

    return {
        "z_obj": z_obj,
        "true_C": true_C,
        "true_strike": true_strike_deg,
        "true_twist": twist_deg,
        "true_shear": shear_deg,
        "true_gain": true_gain,
        "regional_type": regional_type,
        "distortion_strength": distortion_strength,
        "distortion_shear": distortion_shear,
        "noise_level": noise_level,
        "site_id": site_id,
        "seed": seed,
        "periods": periods,
        "regional_type_notes": notes,
    }


_DISAMBIGUATION_STRATEGIES = ("geometric", "identity", "pt_aligned", "min_shear")


def run_all_methods_on_synthetic(synthetic_dict: dict) -> dict[str, Any]:
    """Run every available decomposition method on the synthetic.

    Returns
    -------
    dict[str, result]
        Keys:

        * ``"gb_geometric"``, ``"gb_identity"``, ``"gb_pt_aligned"``,
          ``"gb_min_shear"``: single-site GB with each disambiguation
          strategy.
        * ``"bibby"``: Bibby-Caldwell-Brown.
        * ``"lilley"``: Lilley Mohr-circle.
        * ``"marti"``: Marti WALDIM.

        Methods that fail (raise) on the input report the exception
        string instead of a result; downstream callers can handle
        those by skipping them.
    """
    z = synthetic_dict["z_obj"]
    out: dict[str, Any] = {}

    for strat in _DISAMBIGUATION_STRATEGIES:
        try:
            out[f"gb_{strat}"] = decompose(z, disambiguation=strat, n_starts=3)
        except Exception as exc:  # pragma: no cover -- diagnostic only
            out[f"gb_{strat}"] = f"ERROR: {exc!r}"

    try:
        out["bibby"] = decompose_bibby(z)
    except Exception as exc:  # pragma: no cover
        out["bibby"] = f"ERROR: {exc!r}"

    try:
        out["lilley"] = decompose_lilley(z, n_realisations=50, seed=42)
    except Exception as exc:  # pragma: no cover
        out["lilley"] = f"ERROR: {exc!r}"

    try:
        out["marti"] = decompose_marti(z)
    except Exception as exc:  # pragma: no cover
        out["marti"] = f"ERROR: {exc!r}"

    return out


def compute_method_accuracy(
    method_name: str, result: Any, synthetic_dict: dict
) -> float:
    """Frobenius distance between the recovered and ground-truth ``C``.

    Returns
    -------
    float
        Frobenius norm of ``C_recovered - C_true``. ``nan`` for
        methods that do not return a recoverable ``C`` (Lilley,
        Marti — both characterise the tensor without producing a
        distortion matrix), or for failed runs.

    Notes
    -----
    GB recovery: takes the median of per-period ``(strike, twist,
    shear, gain)`` from the result's ``parameters`` dataset and
    reconstructs ``C`` in the same GB89 form used to generate the
    synthetic.

    Bibby recovery: ``result.C`` is used directly (already a 2x2
    real matrix in the measurement frame).
    """
    true_C = synthetic_dict["true_C"]

    if isinstance(result, str) and result.startswith("ERROR:"):
        return float("nan")

    if method_name.startswith("gb_"):
        params = result.parameters
        strike_deg = float(np.median(params["strike"].values))
        twist_deg = float(np.median(params["twist"].values))
        shear_deg = float(np.median(params["shear"].values))
        gain = float(np.median(params["gain"].values))
        recovered_C = _construct_C_gb89(strike_deg, twist_deg, shear_deg, gain)
        return float(np.linalg.norm(recovered_C - true_C))

    if method_name == "bibby":
        return float(np.linalg.norm(result.C - true_C))

    # Lilley and Marti characterise the tensor without producing C.
    return float("nan")

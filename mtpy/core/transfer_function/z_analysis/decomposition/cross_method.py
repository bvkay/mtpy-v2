"""Cross-method consolidation: run all decomposition methods on the
same site and project their outputs onto a common comparison space.

This is the canonical "run everything on this site" entry point. It
produces a :class:`CrossMethodResult` whose per-method dicts allow
direct apples-to-apples comparison (does method A's recovered
strike agree with method B's? does method A see the same regional
``(Z_TE, Z_TM)`` as method B?), and a companion
:func:`agreement_summary` quantifies the cross-method RMS
differences against a chosen reference.

The motivation is the empirical "three traditions converge" claim
of Paper 1: when GB / MJ (parametric), BCB (gauge-fixed), Lilley
(Mohr-circle), and Garcia-Jones (3-D regional) all agree on a
site's distortion, we have strong cross-tradition evidence the
recovery is correct; when they disagree, the disagreement is
informative about either 3-D regional structure or magnetic
distortion (cf. the
:mod:`...magnetic_distortion_diagnostic` module).

Adapter design
==============

Each method has a private ``_adapter_<name>`` function. Adapters:

* take ``(z_object, periods=None, **method_kwargs)``
* return a dict with keys ``strike``, ``twist_shear``,
  ``regional_z``, ``dimensionality``, ``status``, ``message``;
  each output field is per-period or omitted (``None``) if the
  method does not produce it
* never raise — failures are caught and reported via
  ``status="error"`` / ``status="no_solution"`` so that the rest
  of the cross-method run completes

Adding a new method is a localised change: add an adapter, register
it in :data:`ALL_METHODS`, write tests.

References
----------
See the per-method module docstrings for the foundational
citations (GB89, MJ01, BCB05, Lilley98, Marti09, Garcia02,
Gomez-Treviño 2018).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .bibby import decompose_bibby
from .gomez_trevino import decompose_gomez_trevino
from .groom_bailey import decompose, decompose_joint
from .lilley import decompose_lilley
from .marti import decompose_marti
from .results import CrossMethodResult

if TYPE_CHECKING:
    from mtpy.core.transfer_function.z import Z


ALL_METHODS: list[str] = [
    "groom_bailey",
    "mcneice_jones",
    "bibby",
    "lilley",
    "marti",
    "garcia_jones",
    "gomez_trevino",
]
"""The full list of registered method names. To add a new method,
write its adapter and append the name here."""


_MU_0 = 4.0 * np.pi * 1.0e-7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _period_window_mask(
    result_periods: np.ndarray,
    window: tuple[float, float] | None,
) -> np.ndarray:
    """Boolean mask selecting elements of ``result_periods`` that
    fall in ``window``.

    ``window`` is a ``(pmin, pmax)`` tuple or ``None`` (no
    filtering). Used by adapters whose underlying decomposition
    function does not accept a ``periods`` kwarg
    (:func:`decompose_lilley`, :func:`decompose_marti`) to filter
    results post-call so that every adapter returns arrays of
    matching length on the same period grid.
    """
    if window is None:
        return np.ones(result_periods.size, dtype=bool)
    pmin, pmax = window
    return (result_periods >= pmin) & (result_periods <= pmax)


def _z_strike_frame(
    z_meas: np.ndarray, strike_deg_per_period: np.ndarray
) -> np.ndarray:
    """Rotate per-period measurement-frame ``Z`` into per-period
    strike frame using the per-period strike.
    """
    n_periods = z_meas.shape[0]
    out = np.empty_like(z_meas)
    for k in range(n_periods):
        theta = float(np.radians(strike_deg_per_period[k]))
        cs, sn = np.cos(theta), np.sin(theta)
        R = np.array([[cs, -sn], [sn, cs]])
        out[k] = R.T @ z_meas[k] @ R
    return out


def _ok(strike=None, twist_shear=None, regional_z=None, dimensionality=None):
    return {
        "strike": strike,
        "twist_shear": twist_shear,
        "regional_z": regional_z,
        "dimensionality": dimensionality,
        "status": "success",
        "message": "",
    }


def _fail(status: str, message: str):
    return {
        "strike": None,
        "twist_shear": None,
        "regional_z": None,
        "dimensionality": None,
        "status": status,
        "message": message,
    }


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


def _adapter_groom_bailey(z, periods=None, **kwargs):
    """GB single-site: per-period (strike, twist, shear), regional Z.

    No native dimensionality classification.
    """
    try:
        result = decompose(z, periods=periods, **kwargs)
    except Exception as exc:
        return _fail("error", f"decompose raised: {exc!r}")

    p = result.parameters
    strike = np.mod(np.asarray(p["strike"].values, dtype=np.float64), 90.0)
    twist = np.asarray(p["twist"].values, dtype=np.float64)
    shear = np.asarray(p["shear"].values, dtype=np.float64)

    z_meas = np.asarray(result.regional_z.z, dtype=np.complex128)
    z_strike = _z_strike_frame(
        z_meas, np.asarray(p["strike"].values, dtype=np.float64)
    )
    z_te = z_strike[:, 0, 1]
    z_tm = -z_strike[:, 1, 0]

    return _ok(
        strike=strike,
        twist_shear=(twist, shear),
        regional_z=(z_te, z_tm),
    )


def _adapter_mcneice_jones(z, periods=None, **kwargs):
    """MJ in single-site mode (one-element list). Same comparison
    space as GB.

    Note: ``decompose_joint`` requires an ``MT``-shaped input
    (with ``.station`` and ``.Z``); we wrap ``z`` in a
    ``SimpleNamespace`` for the single-site call.
    """
    from types import SimpleNamespace

    try:
        mt_mock = SimpleNamespace(station="single_site", Z=z)
        result = decompose_joint([mt_mock], periods=periods, **kwargs)
    except Exception as exc:
        return _fail("error", f"decompose_joint raised: {exc!r}")

    p = result.parameters
    strike_per_period = np.asarray(p["strike"].values, dtype=np.float64)
    strike = np.mod(strike_per_period, 90.0)
    # Joint API: twist/shear/gain dimensions are ("period", "station");
    # in single-site mode we squeeze the station axis.
    twist = np.asarray(p["twist"].values, dtype=np.float64).squeeze(axis=-1)
    shear = np.asarray(p["shear"].values, dtype=np.float64).squeeze(axis=-1)

    regional_dict = result.regional_z
    if isinstance(regional_dict, dict) and regional_dict:
        # Single-site: pull the only entry's Z.
        z_meas_obj = next(iter(regional_dict.values()))
        z_meas = np.asarray(z_meas_obj.z, dtype=np.complex128)
        z_strike = _z_strike_frame(z_meas, strike_per_period)
        z_te = z_strike[:, 0, 1]
        z_tm = -z_strike[:, 1, 0]
        regional_z = (z_te, z_tm)
    else:
        regional_z = None

    return _ok(
        strike=strike,
        twist_shear=(twist, shear),
        regional_z=regional_z,
    )


def _adapter_bibby(z, periods=None, **kwargs):
    """BCB: distortion ``C`` per period (or per-band, depending on
    config) plus regional Z. Twist / shear extracted by closed-form
    solving of ``C = T(twist) S(shear)`` after rotating to the
    per-period strike frame from PT alpha.

    BCB itself does not produce a strike directly; we use the
    site's phase-tensor alpha (across-strike azimuth) plus 90°
    (the GB-PT convention offset, see F3) as the strike reference.

    Note (intentional design): :func:`decompose_bibby` returns a
    *single* band-averaged ``C`` tensor (Bibby et al. 2005); the
    per-period ``twist`` and ``shear`` arrays are therefore derived
    from a single ``C`` rotated to each period's PT-alpha frame.
    They have the correct shape (one entry per period in the
    window) but the values are *not* independent samples — close-by
    periods will report the same ``C``, only their PT-alpha-driven
    rotation differs. This matches the spec contract that adapter
    arrays must equal ``len(selected_periods)`` and is documented in
    the cross-method caveats. ``periods=(pmin, pmax)`` is honoured
    by passing the window through to :func:`decompose_bibby`.
    """
    period_window: tuple[float, float] | None = None
    if periods is not None and len(periods) >= 2:
        period_window = (float(periods[0]), float(periods[-1]))
    try:
        result = decompose_bibby(z, periods=period_window, **kwargs)
    except Exception as exc:
        return _fail("error", f"decompose_bibby raised: {exc!r}")

    n_periods = result.periods.size
    if n_periods == 0:
        return _fail("no_solution", "BCB returned zero valid periods")

    pt_alpha_deg = np.asarray(z.phase_tensor.alpha, dtype=np.float64)
    # Map periods to the PT alpha indexing (BCB may have filtered
    # periods); use the BCB-result periods.
    z_freq = np.asarray(z.frequency, dtype=np.float64)
    z_periods = 1.0 / z_freq
    sort_idx = np.argsort(z_periods)
    z_periods_sorted = z_periods[sort_idx]
    pt_sorted = pt_alpha_deg[sort_idx]

    strike_per_period_deg = np.full(n_periods, np.nan)
    for k, p in enumerate(result.periods):
        idx = int(np.argmin(np.abs(z_periods_sorted - p)))
        # PT alpha is across-strike; +90° gives along-strike.
        strike_per_period_deg[k] = (pt_sorted[idx] + 90.0) % 180.0

    # BCB.C is band-averaged (one (2,2) matrix). Use it as the
    # per-period C for now.
    c_meas = np.asarray(result.C, dtype=np.float64)
    twist_per_period = np.empty(n_periods, dtype=np.float64)
    shear_per_period = np.empty(n_periods, dtype=np.float64)
    for k in range(n_periods):
        theta = float(np.radians(strike_per_period_deg[k]))
        cs, sn = np.cos(theta), np.sin(theta)
        R = np.array([[cs, -sn], [sn, cs]])
        c_strike = R.T @ c_meas @ R
        # Solve C = T(twist) S(shear) closed-form (no gain extracted
        # — the gauge-unity convention pins it implicitly).
        gain = (c_strike[0, 0] + c_strike[1, 1]) / 2.0
        if abs(gain) < 1e-12:
            twist_per_period[k] = np.nan
            shear_per_period[k] = np.nan
            continue
        s = (c_strike[0, 1] + c_strike[1, 0]) / (2.0 * gain)
        t = (c_strike[1, 0] - c_strike[0, 1]) / (2.0 * gain)
        twist_per_period[k] = float(np.degrees(np.arctan(t)))
        shear_per_period[k] = float(np.degrees(np.arctan(s)))

    z_meas = np.asarray(result.Z_regional, dtype=np.complex128)
    z_strike = _z_strike_frame(z_meas, strike_per_period_deg)
    z_te = z_strike[:, 0, 1]
    z_tm = -z_strike[:, 1, 0]

    return _ok(
        strike=np.mod(strike_per_period_deg, 90.0),
        twist_shear=(twist_per_period, shear_per_period),
        regional_z=(z_te, z_tm),
    )


def _adapter_lilley(z, periods=None, **kwargs):
    """Lilley Mohr-circle: across-strike rotation + dimensionality.

    Strike is the across-strike rotation (per F3 docstring) plus
    90° to bring it to along-strike, mod 90° for the comparison
    space.

    ``decompose_lilley`` does not accept a ``periods`` kwarg, so
    we filter its full-grid output post-call to the requested
    window. This keeps every adapter's per-period output aligned
    on the same period set as :func:`compute_cross_method`'s
    ``selected_periods``, which is the contract :func:`agreement_summary`
    relies on.
    """
    try:
        result = decompose_lilley(z, **kwargs)
    except Exception as exc:
        return _fail("error", f"decompose_lilley raised: {exc!r}")

    mask = _period_window_mask(np.asarray(result.periods), periods)

    rotation_real_rad = np.asarray(
        result.rotation_real_rad, dtype=np.float64
    )[mask]
    along_strike_deg = (np.degrees(rotation_real_rad) + 90.0) % 180.0
    strike = np.mod(along_strike_deg, 90.0)

    full_dim = list(result.dimensionality)
    dimensionality = [
        full_dim[i] for i in range(len(full_dim)) if bool(mask[i])
    ]
    return _ok(strike=strike, dimensionality=dimensionality)


def _adapter_marti(z, periods=None, **kwargs):
    """Marti WALDIM: integer dimensionality codes per period.

    ``decompose_marti`` does not accept a ``periods`` kwarg, so
    we filter its full-grid output post-call to the requested
    window — same pattern as :func:`_adapter_lilley`.
    """
    try:
        result = decompose_marti(z, **kwargs)
    except Exception as exc:
        return _fail("error", f"decompose_marti raised: {exc!r}")
    mask = _period_window_mask(np.asarray(result.periods), periods)
    dim = np.asarray(result.dimensionality, dtype=np.int64)[mask]
    return _ok(dimensionality=dim)


def _adapter_garcia_jones(z, periods=None, **kwargs):
    """Garcia-Jones requires >= 2 sites; fail gracefully for the
    single-site cross-method entry point.
    """
    return _fail(
        "no_solution",
        "decompose_garcia_jones requires at least 2 sites; "
        "single-site cross-method comparison cannot run GJ.",
    )


def _adapter_gomez_trevino(z, periods=None, **kwargs):
    """Gomez-Treviño: rotational-invariant ``rho_+``, ``rho_-``.

    Maps to the comparison space via the user-spec'd formula
    ``Z_pm = rho_pm * sqrt(omega mu_0) * exp(i * phi_pm)``. No
    strike or twist/shear native; the (rho_+, rho_-) pair is
    used to produce a coarse dimensionality label by comparing
    the two magnitudes.
    """
    try:
        period_window: tuple[float, float] | None = None
        if periods is not None and len(periods) >= 2:
            period_window = (float(periods[0]), float(periods[-1]))
        result = decompose_gomez_trevino(z, periods=period_window, **kwargs)
    except Exception as exc:
        return _fail("error", f"decompose_gomez_trevino raised: {exc!r}")

    omega = 2.0 * np.pi / result.periods
    z_plus = (
        result.rho_plus
        * np.sqrt(omega * _MU_0)
        * np.exp(1j * result.phi_plus)
    )
    z_minus = (
        result.rho_minus
        * np.sqrt(omega * _MU_0)
        * np.exp(1j * result.phi_minus)
    )
    # Coarse dimensionality from the rho_+ / rho_- ratio.
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.abs(result.rho_plus) / np.maximum(
            np.abs(result.rho_minus), 1e-30
        )
    dim_labels: list[str] = []
    for r in ratio:
        if not np.isfinite(r):
            dim_labels.append("undetermined")
        elif abs(r - 1.0) < 0.05:
            dim_labels.append("1D")
        elif abs(r - 1.0) < 0.5:
            dim_labels.append("2D")
        else:
            dim_labels.append("3D-likely")

    return _ok(
        regional_z=(z_plus, z_minus),
        dimensionality=dim_labels,
    )


_ADAPTERS = {
    "groom_bailey": _adapter_groom_bailey,
    "mcneice_jones": _adapter_mcneice_jones,
    "bibby": _adapter_bibby,
    "lilley": _adapter_lilley,
    "marti": _adapter_marti,
    "garcia_jones": _adapter_garcia_jones,
    "gomez_trevino": _adapter_gomez_trevino,
}


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def compute_cross_method(
    z_object: "Z",
    *,
    site: str | None = None,
    methods: list[str] | None = None,
    periods: tuple[float, float] | None = None,
    method_kwargs: dict[str, dict[str, Any]] | None = None,
) -> CrossMethodResult:
    """Run one or more decomposition methods on a single ``Z`` and
    return a consolidated :class:`CrossMethodResult`.

    Parameters
    ----------
    z_object : Z
        mtpy ``Z`` instance for the site.
    site : str, optional
        Identifier; defaults to ``""``.
    methods : list of str, optional
        Subset of :data:`ALL_METHODS` to run. ``None`` (default)
        runs all.
    periods : (pmin, pmax), optional
        Restrict each method to this period window where
        applicable.
    method_kwargs : dict[str, dict], optional
        Per-method extra keyword arguments. Keys are method names
        (must be in :data:`ALL_METHODS`); values are dicts forwarded
        to the underlying decomposition function.
    """
    methods = methods if methods is not None else ALL_METHODS
    method_kwargs = method_kwargs or {}

    site_id = site or ""
    freq = np.asarray(z_object.frequency, dtype=np.float64)
    full_periods = 1.0 / freq

    if periods is not None:
        pmin, pmax = periods
        mask = (full_periods >= pmin) & (full_periods <= pmax)
    else:
        mask = np.ones(full_periods.size, dtype=bool)
    sort_idx = np.argsort(full_periods[mask])
    selected_periods = full_periods[mask][sort_idx]
    n_periods = selected_periods.size
    if n_periods == 0:
        raise ValueError(
            f"compute_cross_method: no periods in window {periods!r}"
        )

    period_arg = (
        (float(selected_periods[0]), float(selected_periods[-1]))
        if periods is not None
        else None
    )

    strike_estimates: dict[str, np.ndarray] = {}
    twist_shear_estimates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    regional_z_estimates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    dimensionality_estimates: dict[str, Any] = {}
    method_status: dict[str, str] = {}
    method_messages: dict[str, str] = {}

    for name in methods:
        if name not in _ADAPTERS:
            method_status[name] = "error"
            method_messages[name] = f"unknown method {name!r}"
            continue
        kw = method_kwargs.get(name, {})
        out = _ADAPTERS[name](z_object, periods=period_arg, **kw)
        method_status[name] = out["status"]
        method_messages[name] = out["message"]
        if out["status"] != "success":
            continue
        if out["strike"] is not None:
            strike_estimates[name] = np.asarray(
                out["strike"], dtype=np.float64
            )
        if out["twist_shear"] is not None:
            t, s = out["twist_shear"]
            twist_shear_estimates[name] = (
                np.asarray(t, dtype=np.float64),
                np.asarray(s, dtype=np.float64),
            )
        if out["regional_z"] is not None:
            te, tm = out["regional_z"]
            regional_z_estimates[name] = (
                np.asarray(te, dtype=np.complex128),
                np.asarray(tm, dtype=np.complex128),
            )
        if out["dimensionality"] is not None:
            dimensionality_estimates[name] = out["dimensionality"]

    return CrossMethodResult(
        site=site_id,
        periods=selected_periods,
        strike_estimates=strike_estimates,
        twist_shear_estimates=twist_shear_estimates,
        regional_z_estimates=regional_z_estimates,
        dimensionality_estimates=dimensionality_estimates,
        method_status=method_status,
        method_messages=method_messages,
    )


def compute_cross_method_collection(
    mt_collection_or_list,
    *,
    methods: list[str] | None = None,
    periods: tuple[float, float] | None = None,
    method_kwargs: dict[str, dict[str, Any]] | None = None,
    skip_failed: bool = True,
) -> dict[str, CrossMethodResult]:
    """Run :func:`compute_cross_method` on every site in a
    collection.

    Parameters
    ----------
    mt_collection_or_list : MTCollection or list[MT]
    methods, periods, method_kwargs
        See :func:`compute_cross_method`.
    skip_failed : bool, default True
        If True, sites where the cross-method call itself raises
        (rare — adapters catch their own failures) are logged and
        skipped. If False, the first such failure raises.
    """
    if hasattr(mt_collection_or_list, "dataframe") and hasattr(
        mt_collection_or_list, "get_tf"
    ):
        df = mt_collection_or_list.dataframe
        items = []
        if df is not None:
            for row in df.itertuples():
                tf_id = getattr(row, "tf_id", getattr(row, "station", None))
                if tf_id is None:
                    continue
                mt = mt_collection_or_list.get_tf(tf_id)
                items.append(
                    (getattr(mt, "station", tf_id), mt.Z)
                )
    else:
        items = [
            (getattr(mt, "station", str(i)), mt.Z)
            for i, mt in enumerate(mt_collection_or_list)
        ]

    out: dict[str, CrossMethodResult] = {}
    for sid, z in items:
        try:
            out[sid] = compute_cross_method(
                z,
                site=sid,
                methods=methods,
                periods=periods,
                method_kwargs=method_kwargs,
            )
        except Exception:
            if skip_failed:
                continue
            raise
    return out


# ---------------------------------------------------------------------------
# Agreement summary
# ---------------------------------------------------------------------------


def _circular_rms_deg(a: np.ndarray, b: np.ndarray, mod: float = 90.0) -> float:
    """RMS of angular differences modulo ``mod``."""
    finite = np.isfinite(a) & np.isfinite(b)
    if not finite.any():
        return float("nan")
    d = np.abs(a[finite] - b[finite]) % mod
    d = np.minimum(d, mod - d)
    return float(np.sqrt(np.mean(d**2)))


def _linear_rms(a: np.ndarray, b: np.ndarray) -> float:
    finite = np.isfinite(a) & np.isfinite(b)
    if not finite.any():
        return float("nan")
    return float(np.sqrt(np.mean((a[finite] - b[finite]) ** 2)))


def _log_amp_rms(a: np.ndarray, b: np.ndarray) -> float:
    """RMS in log10 amplitude (fractional in the underlying
    quantity).
    """
    finite = (
        np.isfinite(a)
        & np.isfinite(b)
        & (np.abs(a) > 1e-30)
        & (np.abs(b) > 1e-30)
    )
    if not finite.any():
        return float("nan")
    return float(
        np.sqrt(np.mean((np.log10(np.abs(a[finite] / b[finite]))) ** 2))
    )


def _phase_rms(a: np.ndarray, b: np.ndarray) -> float:
    finite = np.isfinite(a) & np.isfinite(b)
    if not finite.any():
        return float("nan")
    da = np.angle(a[finite]) - np.angle(b[finite])
    da = (da + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.sqrt(np.mean(da**2)))


def agreement_summary(
    result: CrossMethodResult,
    reference_method: str = "groom_bailey",
) -> dict[str, dict[str, float]]:
    """Per-method RMS difference vs a chosen reference method.

    Statistic semantics
    -------------------
    Each RMS is a **per-period-pair** statistic: at every period
    ``k`` in :attr:`CrossMethodResult.periods` we compute
    ``Δ_k = method[k] - ref[k]`` (with circular folding for
    strikes and log10-magnitude / radians-of-phase for regional
    ``Z``), then square-mean across periods. This is *not* a
    band-median-of-medians — every period contributes one
    residual.

    Per-period-pair comparison requires that every method's
    output array has the same length and is indexed on the same
    period grid. This is enforced by :func:`compute_cross_method`
    and the adapters (``_adapter_lilley`` /
    ``_adapter_marti`` post-filter the underlying methods'
    full-grid output to the requested window, so they match GB /
    BCB shapes); see :func:`_period_window_mask`.

    Returns
    -------
    dict[str, dict[str, float]]
        Outer key: method name (excludes the reference). Inner
        keys (where applicable):

        * ``strike_rms_deg`` (circular, mod 90°)
        * ``twist_rms_deg`` (linear)
        * ``shear_rms_deg`` (linear)
        * ``regional_z_te_log_amp_rms`` (log10 fractional)
        * ``regional_z_te_phase_rms_rad`` (radians)
        * ``regional_z_tm_log_amp_rms``
        * ``regional_z_tm_phase_rms_rad``
    """
    if reference_method not in ALL_METHODS:
        raise ValueError(
            f"reference_method {reference_method!r} not in ALL_METHODS"
        )

    out: dict[str, dict[str, float]] = {}

    ref_strike = result.strike_estimates.get(reference_method)
    ref_ts = result.twist_shear_estimates.get(reference_method)
    ref_z = result.regional_z_estimates.get(reference_method)

    for name in result.strike_estimates:
        if name == reference_method:
            continue
        out.setdefault(name, {})

    for name in result.twist_shear_estimates:
        if name == reference_method:
            continue
        out.setdefault(name, {})

    for name in result.regional_z_estimates:
        if name == reference_method:
            continue
        out.setdefault(name, {})

    for name, m_strike in result.strike_estimates.items():
        if name == reference_method or ref_strike is None:
            continue
        out[name]["strike_rms_deg"] = _circular_rms_deg(
            m_strike, ref_strike, mod=90.0
        )

    for name, (twist, shear) in result.twist_shear_estimates.items():
        if name == reference_method or ref_ts is None:
            continue
        out[name]["twist_rms_deg"] = _linear_rms(twist, ref_ts[0])
        out[name]["shear_rms_deg"] = _linear_rms(shear, ref_ts[1])

    for name, (z_te, z_tm) in result.regional_z_estimates.items():
        if name == reference_method or ref_z is None:
            continue
        out[name]["regional_z_te_log_amp_rms"] = _log_amp_rms(
            z_te, ref_z[0]
        )
        out[name]["regional_z_te_phase_rms_rad"] = _phase_rms(
            z_te, ref_z[0]
        )
        out[name]["regional_z_tm_log_amp_rms"] = _log_amp_rms(
            z_tm, ref_z[1]
        )
        out[name]["regional_z_tm_phase_rms_rad"] = _phase_rms(
            z_tm, ref_z[1]
        )

    return out

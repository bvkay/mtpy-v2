"""Result objects and serialisation for the decomposition package.

Result hierarchy
----------------
The package exposes a single user-facing result type,
:class:`DecompositionResult`, used by every method in the package
(both currently-implemented and planned). It is a frozen-shape
dataclass with the same field set regardless of which tradition
produced it; downstream tooling can therefore consume any
decomposition result with the same code path.

The shape adapts to the method via two conventions:

- **Single-site** (e.g. :func:`decompose`, future Bahr / Lilley /
  WAL): ``regional_z`` is a single :class:`Z`. ``parameters`` has a
  ``period`` coordinate only.
- **Joint multi-site** (e.g. :func:`decompose_joint`, future joint
  Bahr / Marti): ``regional_z`` is ``dict[str, Z]`` keyed by
  ``station_id``. ``parameters`` adds a ``station`` coordinate to
  the per-site fields (``twist``, ``shear``, ``gain``, etc.) while
  shared fields (``strike``) keep the ``period``-only coordinate.

The :meth:`DecompositionResult.regional_z_as_z` helper hides the
single/joint distinction so call sites that only need one
station's regional Z can be agnostic.

The class also exposes :meth:`alternate_branch` for the GB symmetry
(see :mod:`.symmetries`) so callers can examine the equivalent
solution on the other branch without re-running the optimiser.

Serialisation
-------------
Two serialisation paths are provided:

- :meth:`save` / :meth:`load` write a pickle file. Fastest path for
  in-session round trips; not robust across mtpy-v2 versions.
- :meth:`to_netcdf` / :meth:`from_netcdf` write the parameters
  Dataset and embedded regional Z to a NetCDF file with a sidecar
  JSON for ``metadata``. The archival format; safe across versions
  because both formats are stable.

The module-private helpers (``_z_to_serialisable``,
``_save_to_netcdf``, ``_json_serialise_numpy``, etc.) implement the
two paths and are not part of the public API.

References
----------
Groom, R. W., & Bailey, R. C. (1989). Decomposition of magnetotelluric
impedance tensors in the presence of local three-dimensional galvanic
distortion. Journal of Geophysical Research: Solid Earth, 94(B2),
1913-1925.

McNeice, G. W., & Jones, A. G. (2001). Multisite, multifrequency tensor
decomposition of magnetotelluric data. Geophysics, 66(1), 158-173.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import xarray as xr
from loguru import logger

if TYPE_CHECKING:
    from mtpy.core.transfer_function.z import Z


@dataclass
class DecompositionResult:
    """Result of a Groom-Bailey or McNeice-Jones decomposition.

    Attributes
    ----------
    parameters : xarray.Dataset
        Per-period decomposition parameters. Coordinates: ``period``
        (in seconds). For multi-site joint decompositions an
        additional ``station`` coordinate.

        Data variables (all real-valued):

        - ``strike`` : regional azimuth in degrees, in
          ``[0, 180)``.
        - ``twist``, ``shear`` : Groom-Bailey distortion angles in
          degrees.
        - ``gain`` : Groom-Bailey site gain (dimensionless).
        - ``anisotropy`` : Groom-Bailey anisotropy parameter
          (dimensionless; structurally non-identifiable from MT
          alone, reported as fitted but flagged in metadata).
        - ``strike_error``, ``twist_error``, ``shear_error``,
          ``gain_error``, ``anisotropy_error`` : 1-sigma
          uncertainties from the analytic Jacobian, in matching
          units.

    regional_z : Z
        The decomposed regional impedance, as a fresh
        :class:`mtpy.core.transfer_function.z.Z` object with
        propagated errors.

    chi_squared : xarray.DataArray
        Per-period (or per-band) chi-squared values; coordinate
        ``period``.

    rms_misfit : float
        Overall RMS misfit, weighted by the input ``z_error`` (or
        ``z_model_error`` if used).

    method : str
        Identifier for the algorithm used:

        - ``"groom_bailey"`` for single-site GB.
        - ``"mcneice_jones_joint"`` for multi-site joint.

    options : dict
        Options passed to :func:`decompose` or
        :func:`decompose_joint` for this result.

    metadata : dict
        Provenance: software versions, input identifier, RNG seed,
        coordinate frame, strike convention, timestamp.

    frame : str, default ``"measurement"``
        Coordinate frame of ``regional_z``. ``"measurement"`` means
        ``regional_z`` is in the same frame as the input ``z``;
        rotating it by ``-strike`` per period recovers the
        anti-diagonal strike-frame regional tensor. ``"strike"``
        means ``regional_z`` is already in the strike frame
        (anti-diagonal). Public callers receive ``"measurement"``
        by default so plotting and downstream tools behave
        consistently with the input ``z``.

    Notes
    -----
    The 90-degree strike branch is folded into a canonical form:
    each band's ``(strike, twist, shear)`` is mapped through the
    ``(strike + 90 mod 180, -shear, twist)`` symmetry so that
    ``strike`` lands in ``[0, 90)`` after the fold. The shear sign
    is flipped together with the strike shift; twist is unchanged.
    See :func:`_geometric_fold` (the default ``disambiguation``
    strategy); other strategies are documented on
    :func:`decompose`.

    The static-shift convention of ``gain`` matches that of
    :meth:`mtpy.core.mt.MT.remove_static_shift`. Applying
    :meth:`~mtpy.core.mt.MT.remove_static_shift` with the fitted
    ``gain`` recovers the un-shifted regional impedance.

    Examples
    --------
    Single-site decomposition of a station's impedance::

        from mtpy.core.transfer_function.z_analysis.decomposition import decompose
        result = decompose(my_mt.Z)
        result.regional_z.plot_resistivity_phase()
        print(result.parameters['strike'].values)
    """

    parameters: xr.Dataset
    regional_z: "Z"
    chi_squared: xr.DataArray
    rms_misfit: float
    method: str
    options: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    frame: str = "measurement"

    def __getstate__(self):
        """Pickle-friendly state.

        ``Z`` instances hold loguru references that cannot pickle
        (loguru sinks include open file handles). We convert each
        Z to a plain ``(z, z_error, frequency)`` tuple at pickle
        time and reconstitute on unpickle.
        """
        state = self.__dict__.copy()
        state["regional_z"] = _z_to_serialisable(self.regional_z)
        return state

    def __setstate__(self, state):
        state["regional_z"] = _z_from_serialisable(state["regional_z"])
        self.__dict__.update(state)

    def save(self, path: "str | Path") -> None:
        """Save this result to a pickle file.

        Pickle is the simplest and fastest serialisation, suitable for
        "save my session, restart Python, plot the same result"
        workflows. It is not robust across mtpy-v2 version changes —
        if class internals change, old pickles may fail to load. For
        archival or sharing, use :meth:`to_netcdf` instead.

        Parameters
        ----------
        path : str or Path
            Output path. Convention: ``.pkl`` extension.

        See Also
        --------
        load : Inverse operation.
        to_netcdf : Archival serialisation (more robust).
        """
        import pickle as _pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            _pickle.dump(self, f, protocol=_pickle.HIGHEST_PROTOCOL)
        logger.info(f"DecompositionResult saved to {path}")

    @classmethod
    def load(cls, path: "str | Path") -> "DecompositionResult":
        """Load a result previously written by :meth:`save`.

        Parameters
        ----------
        path : str or Path

        Returns
        -------
        DecompositionResult

        Raises
        ------
        FileNotFoundError
            If the file is missing.
        TypeError
            If the unpickled object is not a DecompositionResult.
        """
        import pickle as _pickle

        path = Path(path)
        with path.open("rb") as f:
            obj = _pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(
                f"DecompositionResult.load: file at {path} unpickled "
                f"to {type(obj).__name__}, not DecompositionResult"
            )
        logger.info(f"DecompositionResult loaded from {path}")
        return obj

    def to_netcdf(self, path: "str | Path") -> None:
        """Save this result to NetCDF + sidecar JSON for archival.

        The ``parameters`` Dataset is written as a NetCDF file. The
        ``regional_z`` (Z object for single-site, ``dict[str, Z]``
        for joint) is serialised into additional groups within the
        same NetCDF file. The ``metadata`` dict is written to a
        sidecar JSON file (``<path>.metadata.json``) with NumPy
        arrays converted to a tagged-list form that round-trips back
        to ndarrays on load.

        More robust than :meth:`save` (pickle) across mtpy-v2 version
        changes because NetCDF and JSON are stable formats. More
        expensive (~100x slower for large results).

        Parameters
        ----------
        path : str or Path
            NetCDF output path. Convention: ``.nc`` extension. The
            sidecar JSON is written to ``<path>.metadata.json``.
        """
        return _save_to_netcdf(self, path)

    @classmethod
    def from_netcdf(cls, path: "str | Path") -> "DecompositionResult":
        """Inverse of :meth:`to_netcdf`."""
        return _load_from_netcdf(cls, path)

    def regional_z_as_z(self, station_id: "str | None" = None) -> "Z":
        """Return the regional impedance for one station as a :class:`Z`.

        Single-site results store ``regional_z`` as a :class:`Z`
        directly; joint results store ``dict[station_id, Z]``. This
        helper is a unified dispatcher so callers can write the same
        code regardless of which kind of result they hold:

        - For single-site results, returns ``self.regional_z``
          unchanged. ``station_id`` is ignored.
        - For joint results, returns ``self.regional_z[station_id]``.

        Parameters
        ----------
        station_id : str, optional
            Required for joint results.

        Returns
        -------
        Z
            Regional impedance for the requested station, in
            measurement frame.

        Raises
        ------
        ValueError
            If the result is joint and ``station_id`` is None, or
            if ``station_id`` is not in the result's stations.

        Examples
        --------
        Single-site result, feed straight into mtpy-v2's phase-tensor
        tooling:

        >>> result = decompose(z)
        >>> z_regional = result.regional_z_as_z()
        >>> phase_tensor = z_regional.phase_tensor

        Joint result, pull a specific station:

        >>> joint = decompose_joint(stations)
        >>> z_sta1 = joint.regional_z_as_z(station_id="STA001")
        """
        if isinstance(self.regional_z, dict):
            if station_id is None:
                raise ValueError(
                    "regional_z_as_z: this is a joint result; " "station_id is required"
                )
            if station_id not in self.regional_z:
                raise ValueError(
                    f"regional_z_as_z: station_id={station_id!r} not "
                    f"in result; available: {sorted(self.regional_z)}"
                )
            return self.regional_z[station_id]
        return self.regional_z

    def alternate_branch(self) -> "DecompositionResult":
        """Return the GB-symmetry-equivalent solution on the other branch.

        The Groom-Bailey decomposition has an exact discrete symmetry:
        the parameter triples ``(strike, twist, shear)`` and
        ``(strike + 90 mod 180, twist, -shear)`` produce identical
        predicted impedances, identical galvanic-distortion C tensors,
        and identical fit quality (Groom & Bailey, 1989). At a single
        site the two branches are observationally indistinguishable;
        which one a single optimiser run reports depends on starting
        point and the chosen disambiguation strategy. This method
        returns a fresh :class:`DecompositionResult` representing the
        alternate solution so callers can examine both branches side
        by side.

        The transformation applied to ``parameters``:

        - ``strike`` -> ``(strike + 90) mod 180`` (degrees)
        - ``shear``  -> ``-shear`` (degrees)
        - ``twist``, ``gain``, ``anisotropy``: unchanged
        - all error and uncertainty fields
          (``strike_error`` etc., bootstrap CI bounds): unchanged.
          The 1-sigma uncertainties are invariant under the
          symmetry; CI bounds describe sensitivity in a frame that
          shifts with the strike but covers the same physical
          neighbourhood.

        For ``regional_z`` (single-site :class:`Z` or joint
        ``dict[str, Z]``), the off-diagonal components ``Z_xy`` and
        ``Z_yx`` are swapped — the TE/TM exchange that accompanies
        the 90-degree rotation of the strike axis. Diagonal terms
        and per-component errors are swapped with their off-diagonal
        partners.

        ``chi_squared``, ``rms_misfit``, ``method``, ``options``,
        ``metadata``, and ``frame`` are preserved unchanged; the two
        branches fit the data identically.

        Idempotent: ``r.alternate_branch().alternate_branch()`` is
        equal to ``r`` within floating-point tolerance.

        Returns
        -------
        DecompositionResult
            A new result with the alternate-branch parameters and a
            Z (or dict of Zs) with swapped off-diagonal components.
            ``self`` is not mutated.

        References
        ----------
        Groom, R. W., & Bailey, R. C. (1989). Decomposition of
        magnetotelluric impedance tensors in the presence of local
        three-dimensional galvanic distortion. Journal of
        Geophysical Research: Solid Earth, 94(B2), 1913-1925.
        """
        new_params = self.parameters.copy(deep=True)
        new_params["strike"] = (new_params["strike"] + 90.0) % 180.0
        new_params["shear"] = -new_params["shear"]
        # After the symmetry shift strike covers the full [0, 180)
        # range regardless of the original disambiguation choice.
        new_params["strike"].attrs.update(self.parameters["strike"].attrs)
        new_params["strike"].attrs["range"] = "[0, 180)"
        new_params["shear"].attrs.update(self.parameters["shear"].attrs)

        if isinstance(self.regional_z, dict):
            new_regional_z: Any = {
                sid: _swap_off_diagonals(z) for sid, z in self.regional_z.items()
            }
        else:
            new_regional_z = _swap_off_diagonals(self.regional_z)

        return replace(self, parameters=new_params, regional_z=new_regional_z)


@dataclass
class JointDecompositionResult:
    """Result of a McNeice-Jones (2001) multi-site joint decomposition.

    A purpose-built result type for the new
    :func:`decompose_mcneice_jones` API. Distinct from
    :class:`DecompositionResult` because the joint analysis has
    natively per-band and per-site structure that is most easily
    expressed as plain dicts rather than as an :class:`xr.Dataset`
    with mixed dimensionality. (The older
    :func:`decompose_joint` returns the dataset-shaped result; the
    two are siblings, not duplicates.)

    Attributes
    ----------
    per_site_distortion : dict[str, dict[str, float | np.ndarray]]
        ``{site_id: {parameter_name: value}}``. Per-site distortion
        parameters in the GB factorisation, with keys
        ``twist_deg``, ``shear_deg``, ``gain``, ``c_tensor`` (the
        ``2x2`` reconstructed real distortion matrix in the
        measurement frame), and per-band copies as
        ``twist_deg_per_band``, ``shear_deg_per_band``,
        ``gain_per_band`` (each shape ``(n_bands,)``). Twist /
        shear are in degrees, gain is dimensionless, ``c_tensor``
        is a band-averaged (median) ``2x2`` ndarray.
    per_band_strike : dict[int, float]
        ``{band_id: shared_strike_deg}``. The shared regional
        strike per band, in degrees, in the range determined by
        the chosen disambiguation strategy.
    per_band_per_site_z_regional : dict[tuple[int, str], np.ndarray]
        ``{(band_id, site_id): z_regional}`` where ``z_regional``
        is the regional 2-D impedance for that site at the periods
        of the band, shape ``(n_band_periods, 2, 2)``, in the
        measurement frame.
    chi_squared : float
        Total chi-squared across all bands and sites (sum of
        squared residuals of the joint fit).
    rms_misfit_per_site : dict[str, float]
        Per-site RMS misfit (over all bands and components),
        weighted by the input ``z_error``.
    metadata : dict
        Provenance: ``method='mcneice_jones_joint'``, ``n_starts``,
        ``seed``, ``disambiguation``, ``share_strike_within_band``,
        ``per_site_distortion``, ``max_iter`` request, ``period_bands``
        actually fitted, and ``per_band`` (one entry per band with
        mode info, n_iter, converged, RMS, etc.).

    References
    ----------
    McNeice, G. W., & Jones, A. G. (2001). Multisite, multifrequency
    tensor decomposition of magnetotelluric data. Geophysics,
    66(1), 158-173.
    """

    per_site_distortion: dict[str, dict[str, Any]] = field(default_factory=dict)
    per_band_strike: dict[int, float] = field(default_factory=dict)
    per_band_per_site_z_regional: dict[tuple[int, str], np.ndarray] = field(
        default_factory=dict
    )
    chi_squared: float = 0.0
    rms_misfit_per_site: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GarciaJonesResult:
    """Result of a Garcia-Jones (2002) extended decomposition.

    Garcia-Jones is the 3-D extension of the Groom-Bailey / McNeice-
    Jones tradition: the regional impedance is fitted as a free 3-D
    tensor (four complex components per period) instead of an anti-
    diagonal 2-D tensor, while each station retains its own real
    galvanic distortion. The two-(or-more-)site assumption is that
    neighbouring stations share the *same* regional response but
    have *different* distortion (Garcia & Jones 2002 Section 2),
    which makes the system over-determined for ``n_sites >= 2``.

    Phase 1 scope (this implementation):

    - Per-site distortion: real ``twist`` and ``shear`` only;
      ``gain`` is fixed to 1 and ``anisotropy`` is fixed to 0. The
      2002 paper demonstrates that the gain (and anisotropy) cannot
      be reliably recovered from MT alone (Section 4.2) and the
      added unknowns make the inverse problem unstable.
    - Per-band fitting via TRF nonlinear least-squares with multi-
      start.
    - Single regional 3-D Z shared across sites within each band;
      no smoothing across bands.

    Attributes
    ----------
    per_site_distortion : dict[str, dict[str, float | np.ndarray]]
        ``{site_id: {parameter_name: value}}`` with band-averaged
        ``twist_deg``, ``shear_deg``, ``gain`` (always 1.0 in Phase
        1), ``c_tensor`` (the real 2x2 distortion matrix in the
        measurement frame), and per-band copies as
        ``twist_deg_per_band``, ``shear_deg_per_band``.
    per_band_3d_z_regional : dict[int, np.ndarray]
        ``{band_id: z_regional}`` where ``z_regional`` is the
        recovered 3-D regional impedance with shape
        ``(n_band_periods, 2, 2)``, complex, in the measurement
        frame.
    chi_squared : float
        Total chi-squared across all bands and sites.
    rms_misfit_per_site : dict[str, float]
        Per-site RMS misfit (over all bands and components),
        weighted by the input ``z_error``.
    metadata : dict
        Provenance: ``method='garcia_jones'``, ``n_starts``,
        ``seed``, ``share_distortion_within_period``,
        ``per_band_3d``, ``period_bands`` actually fitted, and
        ``per_band`` (mode info, n_iter, RMS, etc.).

    References
    ----------
    Garcia, X., & Jones, A. G. (2002). Decomposition of three-
    dimensional magnetotelluric data. In *Three-Dimensional
    Electromagnetics* (M. S. Zhdanov & P. E. Wannamaker, eds.),
    Methods in Geochemistry and Geophysics, 35, 235-250.

    Garcia, X., Boerner, D., & Pedersen, L. B. (2003). Electric and
    magnetic galvanic distortion decomposition of tensor CSAMT
    data. Application to data from the Buchans Mine (Newfoundland,
    Canada). Geophysical Journal International, 154, 957-969.
    """

    per_site_distortion: dict[str, dict[str, Any]] = field(default_factory=dict)
    per_band_3d_z_regional: dict[int, np.ndarray] = field(default_factory=dict)
    chi_squared: float = 0.0
    rms_misfit_per_site: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MartiResult:
    """Result of a WAL-invariants / WALDIM dimensionality analysis.

    The Marti tradition (Marti et al. 2004, 2005, 2009, 2010, 2013;
    Weaver-Agarwal-Lilley 2000) classifies each period of an MT
    impedance tensor into a dimensionality regime based on the
    seven WAL rotational invariants ``I_1`` ... ``I_7`` and the
    auxiliary ``Q``. The classification per Marti et al. (2009)
    Table 1 distinguishes 1-D, 2-D, 3-D / 2-D-distorted (with
    several sub-cases), and 3-D regimes via direct sign tests on
    the invariants against a single user-tunable threshold.

    Attributes
    ----------
    periods : ndarray, shape ``(n_periods,)``
        Periods (seconds), sorted ascending.
    I1, I2, I3, I4, I5, I6, I7 : ndarray of float
        The seven WAL invariants per period. ``I_1`` and ``I_2``
        carry units of ``Z`` (impedance); the rest are
        dimensionless. See the :mod:`.marti` module docstring for
        the algebraic definitions and physical interpretation.
    Q : ndarray of float
        WAL auxiliary invariant — the denominator in the formula
        for ``I_7``. Small ``Q`` indicates an ill-defined ``I_7``
        and is itself a 2-D-versus-3-D-distorted discriminator.
    dimensionality : ndarray of int
        Per-period classification:

        * ``0`` — undetermined
        * ``1`` — 1-D
        * ``2`` — 2-D
        * ``3`` — 3-D / 2-D twist-only (Marti 2009 case 3a)
        * ``4`` — 3-D / 2-D general (Marti 2009 case 4)
        * ``5`` — 3-D
        * ``6`` — 3-D / 2-D with diagonal regional tensor
          (Marti 2009 case 3c)
        * ``7`` — 3-D / 2-D or 3-D / 1-D-2-D
          indistinguishable (Marti 2009 case 3b)

    strike_2d_real_rad, strike_2d_imag_rad : ndarray of float
        Per-period 2-D strike candidates from the in-phase and
        quadrature Mohr folds (Marti's ``St_3`` and ``St_4``,
        radians). For pure 2-D periods these agree (within the
        quadrant ambiguity); divergence is itself a 3-D indicator.
    strike_3d_2d_rad : ndarray of float
        Per-period 3-D / 2-D Bahr strike (Marti's ``St_5``,
        radians) — the rotation that simultaneously satisfies
        Bahr's equal-phase condition for the in-phase and
        quadrature parts. Computed by numerical root-finding;
        ``nan`` where no real solution exists.
    f1_rad, f2_rad : ndarray of float
        Smith (1995) distortion-angle linear combinations
        ``f1 = (twist + shear) / 2`` and
        ``f2 = (twist - shear) / 2`` (Marti's ``St_6``, ``St_7``,
        radians). Computed for 2-D and 3-D / 2-D classifications
        only; ``nan`` otherwise.
    twist_rad, shear_rad : ndarray of float
        Smith (1995) distortion angles (Marti's ``St_8``,
        ``St_9``, radians). Computed for 3-D / 2-D classifications;
        ``nan`` otherwise.
    metadata : dict
        Provenance: ``method``, the threshold used, etc.

    References
    ----------
    Marti, A., Queralt, P., Jones, A. G., & Ledo, J. (2005).
    Improving Bahr's invariant parameters using the WAL approach.
    Geophysical Journal International, 163, 38-41.

    Marti, A., Queralt, P., & Ledo, J. (2009). WALDIM: A code for
    the dimensionality analysis of magnetotelluric data using the
    rotational invariants of the magnetotelluric tensor. Computers
    & Geosciences, 35, 2295-2303.

    Booker, J. R. (2014). The magnetotelluric phase tensor: a
    critical review. Surveys in Geophysics, 35, 7-40.

    Weaver, J. T., Agarwal, A. K., & Lilley, F. E. M. (2000).
    Characterization of the magnetotelluric tensor in terms of
    its invariants. Geophysical Journal International, 141,
    321-336.
    """

    periods: np.ndarray
    I1: np.ndarray
    I2: np.ndarray
    I3: np.ndarray
    I4: np.ndarray
    I5: np.ndarray
    I6: np.ndarray
    I7: np.ndarray
    Q: np.ndarray
    dimensionality: np.ndarray
    strike_2d_real_rad: np.ndarray
    strike_2d_imag_rad: np.ndarray
    strike_3d_2d_rad: np.ndarray
    f1_rad: np.ndarray
    f2_rad: np.ndarray
    twist_rad: np.ndarray
    shear_rad: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LilleyResult:
    """Result of a Lilley Mohr-circle distortion analysis.

    The Lilley tradition (Lilley 1976, 1993, 2012, 2016, 2018, 2020)
    represents an MT impedance tensor parametrically-free as a pair
    of Mohr circles, one for the in-phase part of ``Z`` and one for
    the quadrature part. Each circle's centre, radius, and
    rotational invariants are tensor-axes-rotation-invariant
    summaries of the underlying physics.

    Convention
    ----------
    Throughout this result type the suffix ``_real`` denotes
    quantities derived from the in-phase part of ``Z`` (Lilley's
    subscript ``p``), and ``_imag`` denotes the quadrature part
    (Lilley's subscript ``q``). The 2-D Mohr circle centre is
    packed into a complex scalar
    ``c_x + 1j * c_y`` where ``c_x`` is the horizontal axis of the
    Mohr diagram (``Z'_xy``) and ``c_y`` is the vertical axis
    (``Z'_xx``). Radii are real positive scalars. See the
    :mod:`.lilley` module docstring for the algebraic definitions.

    Attributes
    ----------
    periods : ndarray, shape ``(n_periods,)``
        Periods (seconds), sorted ascending.
    center_real, center_imag : ndarray of complex, shape ``(n_periods,)``
        Mohr-circle centre per period for the in-phase / quadrature
        circle, packed as ``c_x + 1j * c_y``.
    radius_real, radius_imag : ndarray of float, shape ``(n_periods,)``
        Mohr-circle radius (Lilley's ``C_p`` and ``C_q``) per period.
    rotation_real_rad, rotation_imag_rad : ndarray of float
        Per-period rotation angles (radians) that bring the
        in-phase / quadrature radial arm onto the horizontal axis,
        which Lilley reads as a 2-D-strike candidate from each
        circle. Equal to ``-beta / 2`` in Lilley 2018 notation.
        **Convention: across-strike azimuth.** This is the
        direction *perpendicular* to the principal (along-strike)
        direction, so it differs from the GB-tradition ``strike``
        field returned by :func:`...groom_bailey.decompose` by
        90° (modulo 180°). The Mohr-circle algebra cannot
        distinguish along-strike from across-strike on its own
        (the two are related by the GB 90-degree symmetry); this
        implementation consistently returns the across-strike
        branch. To compare against a GB ``strike`` value, add
        90° (mod 180°). Worked example: a clean 2-D synthetic
        with TE / TM strike at 30° (GB ``strike = 30°``) yields
        ``rotation_real_rad = -60°`` (≡ 120° mod 180°); adding
        90° gives 30°. Renaming this field to
        ``across_strike_azimuth_rad`` is a candidate follow-up
        to make the convention explicit at the call site as well
        as in this docstring; deferred for now to avoid a
        breaking attribute change.
    central_impedance_real, central_impedance_imag : ndarray of float
        Lilley's ``Z^L_p`` and ``Z^L_q``: the distance from the
        origin to the circle centre. 1-D scale of the tensor.
    anisotropy_real_rad, anisotropy_imag_rad : ndarray of float
        Lilley's ``lambda_p`` and ``lambda_q`` angles (radians):
        ``arcsin(C / Z^L)``. 2-D-anisotropy measures.
    threed_real_rad, threed_imag_rad : ndarray of float
        Lilley's ``mu_p`` and ``mu_q`` angles (radians): the angle
        at the origin between the circle-centre line and the
        horizontal axis. Per-part 3-D measures.
    delta_beta_rad : ndarray of float
        Lilley's ``delta beta = beta_q - beta_p`` (radians), the
        angle between the in-phase and quadrature radial arms. The
        seventh "linking" invariant; near-zero values indicate the
        Bahr distortion-of-2-D regime.
    strike_mean_rad, strike_std_rad : ndarray of float
        Mean and standard deviation of the per-period strike across
        the noise-stability ensemble (radians). Sites with high
        ``strike_std_rad`` are unstable in Lilley's
        noise-stability sense.
    strike_histograms : list of ndarray
        One ``(n_realisations,)`` array per period holding the
        full strike distribution for downstream plotting.
    dimensionality : list of str
        One classification per period: ``"1D"``, ``"2D"``,
        ``"3D-distorted"`` (Bahr 2-D + 3-D galvanic distortion
        regime), or ``"3D"``.
    metadata : dict
        Provenance: ``method``, the dimensionality / noise-stability
        thresholds used, RNG seed, etc.

    References
    ----------
    Lilley, F. E. M. (1976). Diagrams for magnetotelluric data.
    Geophysics, 41(4), 766-770.

    Lilley, F. E. M. (1993). Magnetotelluric analysis using Mohr
    circles. Geophysics, 58(10), 1498-1506.

    Lilley, F. E. M. (2012). Magnetotelluric tensor decomposition:
    insights from linear algebra and Mohr diagrams. In *New
    Achievements in Geoscience*.

    Lilley, F. E. M. (2016). The distortion tensor of
    magnetotellurics: a tutorial on some properties. Exploration
    Geophysics, 47(2), 85-99.

    Lilley, F. E. M. (2018). The magnetotelluric tensor: improved
    invariants for its decomposition, especially the 7th.
    Exploration Geophysics, 49(5), 622-636.
    """

    periods: np.ndarray
    center_real: np.ndarray
    center_imag: np.ndarray
    radius_real: np.ndarray
    radius_imag: np.ndarray
    rotation_real_rad: np.ndarray
    rotation_imag_rad: np.ndarray
    central_impedance_real: np.ndarray
    central_impedance_imag: np.ndarray
    anisotropy_real_rad: np.ndarray
    anisotropy_imag_rad: np.ndarray
    threed_real_rad: np.ndarray
    threed_imag_rad: np.ndarray
    delta_beta_rad: np.ndarray
    strike_mean_rad: np.ndarray
    strike_std_rad: np.ndarray
    strike_histograms: list[np.ndarray] = field(default_factory=list)
    dimensionality: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BibbyResult:
    """Result of a Bibby-Caldwell-Brown decomposition.

    BCB fits a single real 2x2 distortion tensor ``C`` per period
    and band-averages it (Bibby et al., 2005). The output is
    deliberately minimal — no per-period parameter set, no errors
    or CIs — because the BCB decomposition is the *simplest*
    distortion-analysis method in the package and is included as a
    methodological-baseline comparison point for the parameterised
    Groom-Bailey result.

    Attributes
    ----------
    C : ndarray, shape (2, 2)
        The band-averaged real distortion matrix in the measurement
        frame. The averaging method (median or mean) is recorded in
        :attr:`band_method`.
    Z_regional : ndarray, shape (n_periods, 2, 2), complex
        Per-period regional 2-D impedance tensor in the measurement
        frame, anti-diagonal in the strike frame chosen from the
        phase tensor's principal axis. Provided per period (not
        band-averaged) so callers can plot or further-process the
        regional response.
    rms_misfit : float
        Root-mean-square of the per-period residuals
        ``Z_obs - C_period @ Z_R_period`` over all entries and all
        periods in the band, computed from the un-averaged C
        tensors.
    n_periods : int
        Number of valid periods in the band (periods with a finite
        phase-tensor strike).
    band_method : str
        ``"median"`` or ``"mean"``; the method used to average the
        per-period C tensors.
    periods : ndarray, shape (n_periods,)
        Periods (seconds) of the band, sorted ascending.
    gauge : str
        Identifier for the gauge convention used to fix the BCB
        non-determinable scale (see the :mod:`.bibby` module
        docstring). Defaults to ``"diagonal_unity"``: the regional
        Z's strike-frame off-diagonals match the observed Z's
        strike-frame off-diagonals, forcing C's strike-frame
        diagonal entries to ~1.

    References
    ----------
    Bibby, H. M., Caldwell, T. G., & Brown, C. (2005). Determinable
    and non-determinable parameters of galvanic distortion in
    magnetotellurics. Geophysical Journal International, 163(3),
    915-930.
    """

    C: np.ndarray
    Z_regional: np.ndarray
    rms_misfit: float
    n_periods: int
    band_method: str = "median"
    periods: np.ndarray = field(default_factory=lambda: np.empty(0))
    gauge: str = "diagonal_unity"


@dataclass
class GomezTrevinoResult:
    """Rotational-invariant TE / TM resistivities (Gomez-Treviño 2018).

    .. note::

        **Exploratory module.** The Gomez-Treviño 2018 framework
        constructs two rotational-invariant complex resistivities
        ``rho_s`` (series) and ``rho_p`` (parallel) from the
        impedance tensor and pulls invariant TE / TM analogues
        ``rho_+`` and ``rho_-`` out of a quadratic. In 2-D
        (anti-diagonal strike-frame ``Z``) the framework
        demonstrably reduces to the standard TE / TM apparent
        resistivities (verified by tests). The framework's
        usefulness for general 3-D data has not been independently
        validated in the literature; treat the per-period
        ``rho_+`` / ``rho_-`` outputs as exploratory / diagnostic
        rather than production-quality apparent-resistivity
        estimates.

    Fields
    ------
    site : str
        Identifier for the source ``Z``. Optional; defaults to
        ``""`` when the entry point isn't passed a station id.
    periods : ndarray, shape ``(n_periods,)``
        Periods (seconds), sorted ascending.
    rho_s, rho_p : ndarray of complex
        Series and parallel resistivities (Gomez-Treviño eq.
        ``rho_s = trace(Z^T Z)/(2 omega mu_0)``,
        ``rho_p = 2/(omega mu_0 trace(Y^T Y))`` with ``Y = Z^{-1}``).
        Per period.
    rho_plus, rho_minus : ndarray of float
        Magnitudes ``|rho_pm|`` of the quadratic-equation
        solutions ``rho_s ± sqrt(rho_s² − rho_s · rho_p)``. The
        ``+`` / ``−`` labels are purely mathematical; the mode
        assignment to TE vs TM is *ambiguous* in 3-D and even in
        2-D depends on the strike convention.
    phi_plus, phi_minus : ndarray of float
        Phases ``arg(rho_pm)`` (radians) of the quadratic solutions.
    rho_d : ndarray of float
        Magnitude of the determinant resistivity
        ``sqrt(rho_s · rho_p) = sqrt(rho_+ · rho_-)`` — the
        Berdichevsky-Dmitriev (1976) invariant up to a 90° phase
        convention. Included for direct comparison.
    phi_d : ndarray of float
        Phase ``arg(sqrt(rho_s · rho_p))`` (radians).
    convergence_iterations : ndarray of int, optional
        For the iterative ``arithmetic / harmonic mean`` chain
        described in :func:`...gomez_trevino.iterative_chain`,
        the iteration count to convergence per period. ``None``
        when ``decompose`` is called without the chain option.

    References
    ----------
    Gómez-Treviño, E., Esparza, F. J., & Romo, J. M. (2018). On
    the use of two new invariants of the magnetotelluric impedance
    tensor as natural rotational invariant TE and TM modes. Earth,
    Planets and Space, 70:35. doi:10.1186/s40623-018-0900-y
    """

    site: str
    periods: np.ndarray
    rho_s: np.ndarray
    rho_p: np.ndarray
    rho_plus: np.ndarray
    rho_minus: np.ndarray
    phi_plus: np.ndarray
    phi_minus: np.ndarray
    rho_d: np.ndarray
    phi_d: np.ndarray
    convergence_iterations: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MagneticDistortionFlag:
    """Heuristic per-site flag for *suspected* magnetic galvanic
    distortion.

    .. warning::

        **This is a heuristic flag, not a quantitative
        correction.** A flag does *not* prove magnetic distortion
        is present at the site; it indicates the standard MT
        decomposition assumption (magnetic galvanic distortion is
        negligible) may be violated. Two intended uses:

        1. Exclude flagged sites from continental aggregation, OR
        2. Annotate flagged sites in summary plots.

        Do **not** use the flag to "correct" individual-site
        analyses. Magnetic distortion handling is Paper 6 territory
        (Bayesian inversion with explicit ``Q_h``, ``Q_z`` priors
        per Garcia, Boerner & Pedersen 2003 and Chave & Smith 1994);
        a follow-up PR will integrate that.

    Three diagnostics combine into a single overall flag:

    * **Tipper-based** (:func:`...magnetic_distortion_diagnostic.tipper_diagnostic`)
      — anomalously large or strongly frequency-dependent vertical
      magnetic transfer function magnitude. The Garcia 2003 paper
      shows this manifests at intermediate periods.
    * **Frequency dependence of C** (:func:`...frequency_dependence_diagnostic`)
      — recovered ``C`` tensor varies across bands more than a
      static-galvanic model can explain.
    * **Cross-method inconsistency**
      (:func:`...method_inconsistency_diagnostic`)
      — GB / MJ (E-field-only distortion) disagrees with
      Garcia-Jones (3-D regional) on regional ``Z`` recovery.

    Flag combination rule:

    * ``"high_risk"``: 2 or more diagnostics flag, *or* peak
      tipper magnitude exceeds the strong-tipper override
      threshold (default 0.5).
    * ``"moderate_risk"``: exactly one diagnostic flags.
    * ``"low_risk"``: zero diagnostics flag.
    * ``"indeterminate"``: insufficient data (e.g. no Tipper
      available, fewer than 3 bands for the frequency-dependence
      diagnostic, no cross-method input).

    Fields
    ------
    site : str
        Identifier for the source site.
    overall_flag : str
        One of ``"low_risk"``, ``"moderate_risk"``, ``"high_risk"``,
        ``"indeterminate"``.
    tipper_diagnostic : dict
        Per-diagnostic detail (see :func:`...tipper_diagnostic`).
        ``None`` when Tipper data is unavailable.
    frequency_dependence_diagnostic : dict
        Per-diagnostic detail. ``None`` when fewer than 3 bands
        of recovered ``C`` are supplied.
    method_inconsistency_diagnostic : dict
        Per-diagnostic detail. ``None`` when no cross-method
        comparison is supplied.
    contributing_factors : list of str
        Human-readable list of which diagnostics contributed to a
        non-low-risk flag (or the reasons for ``indeterminate``).

    References
    ----------
    Chave, A. D., & Smith, J. T. (1994). On electric and magnetic
    galvanic distortion tensor decompositions. *Journal of
    Geophysical Research* 99(B3), 4669-4682.

    Garcia, X., Boerner, D., & Pedersen, L. B. (2003). Electric and
    magnetic galvanic distortion decomposition of tensor CSAMT
    data. *Geophysical Journal International* 154, 957-969.
    """

    site: str
    overall_flag: str
    tipper_diagnostic: dict[str, Any] | None = None
    frequency_dependence_diagnostic: dict[str, Any] | None = None
    method_inconsistency_diagnostic: dict[str, Any] | None = None
    contributing_factors: list[str] = field(default_factory=list)


def _swap_off_diagonals(z_obj: "Z") -> "Z":
    """Return a fresh :class:`Z` with ``Z_xy`` and ``Z_yx`` swapped.

    Used by :meth:`DecompositionResult.alternate_branch` to flip the
    regional impedance into its TE/TM-swapped representation that
    accompanies the GB 90-degree strike rotation.
    """
    from mtpy.core.transfer_function.z import Z as _Z

    z_arr = np.asarray(z_obj.z).copy()
    z_arr[:, [0, 1], [1, 0]] = z_arr[:, [1, 0], [0, 1]]

    z_err = z_obj.z_error
    if z_err is not None:
        z_err = np.asarray(z_err).copy()
        z_err[:, [0, 1], [1, 0]] = z_err[:, [1, 0], [0, 1]]

    return _Z(z=z_arr, z_error=z_err, frequency=z_obj.frequency)


def _z_to_serialisable(regional_z):
    """Convert a Z or dict[str, Z] into pickle/JSON-friendly arrays.

    Z instances hold loguru references that cannot pickle. The
    plain (z, z_error, frequency) form round-trips via
    :func:`_z_from_serialisable`.
    """
    from mtpy.core.transfer_function.z import Z

    if isinstance(regional_z, dict):
        return {
            "_kind": "dict",
            "items": {sid: _z_to_serialisable(z) for sid, z in regional_z.items()},
        }
    if isinstance(regional_z, Z):
        return {
            "_kind": "Z",
            "z": np.asarray(regional_z.z),
            "z_error": (
                None if regional_z.z_error is None else np.asarray(regional_z.z_error)
            ),
            "frequency": np.asarray(regional_z.frequency),
        }
    return regional_z

def _z_from_serialisable(payload):
    """Inverse of :func:`_z_to_serialisable`."""
    from mtpy.core.transfer_function.z import Z

    if not isinstance(payload, dict):
        return payload
    kind = payload.get("_kind")
    if kind == "Z":
        return Z(
            z=payload["z"],
            z_error=payload["z_error"],
            frequency=payload["frequency"],
        )
    if kind == "dict":
        return {sid: _z_from_serialisable(v) for sid, v in payload["items"].items()}
    return payload

def _sanitize_station_id(station_id: str) -> str:
    """Encode a station_id for use as a NetCDF group name.

    NetCDF group names follow CDL identifier rules: alphanumeric and
    underscore only, must start with a letter or underscore. Real
    station IDs (e.g. ``"KD-P5 R=KD-RR"``) routinely violate this.
    We percent-encode disallowed characters so the encoding is
    reversible.
    """
    import re
    import urllib.parse

    encoded = urllib.parse.quote(station_id, safe="").replace("%", "_p_")
    # Hyphens are not in CDL identifiers either; replace.
    encoded = encoded.replace("-", "_d_").replace(".", "_dt_")
    if not re.match(r"^[A-Za-z_]", encoded):
        encoded = "s_" + encoded
    return encoded

def _desanitize_station_id(group_name: str) -> str:
    """Inverse of :func:`_sanitize_station_id`."""
    import urllib.parse

    s = group_name
    if s.startswith("s_"):
        s = s[2:]
    s = s.replace("_d_", "-").replace("_dt_", ".")
    s = s.replace("_p_", "%")
    return urllib.parse.unquote(s)

def _json_serialise_numpy(obj):
    """Default function for json.dump when encountering NumPy types
    or Python complex numbers (which appear in bootstrap regional_z
    replicates)."""
    if isinstance(obj, np.ndarray):
        if np.issubdtype(obj.dtype, np.complexfloating):
            return {
                "__complex_array__": True,
                "real": obj.real.tolist(),
                "imag": obj.imag.tolist(),
                "dtype": str(obj.dtype),
                "shape": list(obj.shape),
            }
        return {
            "__numpy_array__": True,
            "data": obj.tolist(),
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
        }
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, complex):
        return {"__complex__": True, "real": obj.real, "imag": obj.imag}
    raise TypeError(f"_json_serialise_numpy: unhandled type {type(obj)}")

def _json_deserialise_numpy(d):
    """object_hook for json.load that reconstructs NumPy arrays and
    complex scalars.

    Bootstrap regional_z replicates contain both np.complex128
    arrays (the typed paths) and Python complex scalars (the latter
    arise when NumPy indexing returns a 0-d slice and is converted
    via .item()). The ``__complex_array__`` and ``__complex__`` tags
    handle both forms so round-trip via JSON is exact.
    """
    if isinstance(d, dict):
        if d.get("__numpy_array__"):
            arr = np.array(d["data"], dtype=d["dtype"])
            return arr.reshape(d["shape"])
        if d.get("__complex_array__"):
            real = np.array(d["real"])
            imag = np.array(d["imag"])
            arr = (real + 1j * imag).astype(d["dtype"])
            return arr.reshape(d["shape"])
        if d.get("__complex__"):
            return complex(d["real"], d["imag"])
    return d

def _z_to_dataset(z) -> xr.Dataset:
    """Convert a Z to an xarray Dataset for NetCDF storage."""
    z_arr = np.asarray(z.z)
    return xr.Dataset(
        {
            "z_real": (("period", "i", "j"), z_arr.real),
            "z_imag": (("period", "i", "j"), z_arr.imag),
            "z_error": (
                ("period", "i", "j"),
                np.asarray(z.z_error)
                if z.z_error is not None
                else np.full(z_arr.shape, np.nan),
            ),
            "frequency": (("period",), np.asarray(z.frequency)),
        },
        attrs={"z_error_present": bool(z.z_error is not None)},
    )

def _dataset_to_z(ds: xr.Dataset):
    """Inverse of :func:`_z_to_dataset`."""
    from mtpy.core.transfer_function.z import Z

    z_complex = ds["z_real"].values + 1j * ds["z_imag"].values
    z_error = ds["z_error"].values
    if not bool(ds.attrs.get("z_error_present", True)):
        z_error = None
    elif np.all(np.isnan(z_error)):
        z_error = None
    return Z(
        z=z_complex,
        z_error=z_error,
        frequency=ds["frequency"].values,
    )

def _save_to_netcdf(result, path):
    """Implementation of :meth:`DecompositionResult.to_netcdf`.

    Writes a single flat NetCDF file (scipy backend, no groups) with
    parameters + chi_squared + regional_z merged into one Dataset.
    Metadata and scalar fields go to a sidecar JSON.
    """
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    combined = result.parameters.copy()
    combined["chi_squared_per_period"] = result.chi_squared

    if isinstance(result.regional_z, dict):
        station_ids = list(result.parameters.coords["station"].values)
        n_periods = len(result.parameters.coords["period"])
        n_sites = len(station_ids)
        rz_real = np.full((n_sites, n_periods, 2, 2), np.nan)
        rz_imag = np.full((n_sites, n_periods, 2, 2), np.nan)
        rz_err = np.full((n_sites, n_periods, 2, 2), np.nan)
        err_present_all = True
        for i, sid in enumerate(station_ids):
            z = result.regional_z[sid]
            z_arr = np.asarray(z.z)
            rz_real[i] = z_arr.real
            rz_imag[i] = z_arr.imag
            if z.z_error is None:
                err_present_all = False
            else:
                rz_err[i] = np.asarray(z.z_error)
        combined["regional_z_real"] = (("station", "period", "i", "j"), rz_real)
        combined["regional_z_imag"] = (("station", "period", "i", "j"), rz_imag)
        combined["regional_z_error"] = (("station", "period", "i", "j"), rz_err)
        combined.attrs["regional_z_error_present"] = "1" if err_present_all else "0"
        combined.attrs["regional_z_kind"] = "dict"
        # All stations share frequency grid (validated at decompose_joint)
        first_z = next(iter(result.regional_z.values()))
        combined["regional_z_frequency"] = (
            ("period",),
            np.asarray(first_z.frequency),
        )
    else:
        z = result.regional_z
        z_arr = np.asarray(z.z)
        combined["regional_z_real"] = (("period", "i", "j"), z_arr.real)
        combined["regional_z_imag"] = (("period", "i", "j"), z_arr.imag)
        if z.z_error is None:
            combined["regional_z_error"] = (
                ("period", "i", "j"),
                np.full(z_arr.shape, np.nan),
            )
            combined.attrs["regional_z_error_present"] = "0"
        else:
            combined["regional_z_error"] = (
                ("period", "i", "j"),
                np.asarray(z.z_error),
            )
            combined.attrs["regional_z_error_present"] = "1"
        combined.attrs["regional_z_kind"] = "single"
        combined["regional_z_frequency"] = (
            ("period",),
            np.asarray(z.frequency),
        )

    combined.to_netcdf(path, mode="w")

    metadata_path = Path(str(path) + ".metadata.json")
    payload = {
        "_method": result.method,
        "_frame": result.frame,
        "_rms_misfit": float(result.rms_misfit),
        "_options": result.options,
        "metadata": result.metadata,
    }
    with metadata_path.open("w") as f:
        json.dump(payload, f, indent=2, default=_json_serialise_numpy)

    logger.info(f"DecompositionResult written to {path} (+ {metadata_path.name})")

def _load_from_netcdf(cls, path):
    """Implementation of :meth:`DecompositionResult.from_netcdf`."""
    import json

    from mtpy.core.transfer_function.z import Z

    path = Path(path)
    metadata_path = Path(str(path) + ".metadata.json")
    if not path.exists():
        raise FileNotFoundError(f"NetCDF file not found: {path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata sidecar not found: {metadata_path}")

    combined = xr.open_dataset(path).load()
    kind = str(combined.attrs.get("regional_z_kind", "single"))
    err_present = combined.attrs.get("regional_z_error_present", "1") == "1"

    rz_real = combined["regional_z_real"].values
    rz_imag = combined["regional_z_imag"].values
    rz_err = combined["regional_z_error"].values
    rz_freq = combined["regional_z_frequency"].values
    chi_squared = combined["chi_squared_per_period"]

    if kind == "dict":
        station_ids = [str(s) for s in combined.coords["station"].values]
        regional_z = {}
        for i, sid in enumerate(station_ids):
            z_complex = rz_real[i] + 1j * rz_imag[i]
            z_error = rz_err[i] if err_present else None
            regional_z[sid] = Z(z=z_complex, z_error=z_error, frequency=rz_freq)
    else:
        z_complex = rz_real + 1j * rz_imag
        z_error = rz_err if err_present else None
        regional_z = Z(z=z_complex, z_error=z_error, frequency=rz_freq)

    drop_vars = [
        "regional_z_real",
        "regional_z_imag",
        "regional_z_error",
        "regional_z_frequency",
        "chi_squared_per_period",
    ]
    parameters = combined.drop_vars([v for v in drop_vars if v in combined])
    for attr_key in ("regional_z_kind", "regional_z_error_present"):
        if attr_key in parameters.attrs:
            del parameters.attrs[attr_key]

    with metadata_path.open() as f:
        payload = json.load(f, object_hook=_json_deserialise_numpy)

    return cls(
        parameters=parameters,
        regional_z=regional_z,
        chi_squared=chi_squared,
        rms_misfit=float(payload.get("_rms_misfit", float("nan"))),
        method=payload.get("_method", "groom_bailey"),
        options=payload.get("_options", {}) or {},
        metadata=payload.get("metadata", {}) or {},
        frame=payload.get("_frame", "measurement"),
    )

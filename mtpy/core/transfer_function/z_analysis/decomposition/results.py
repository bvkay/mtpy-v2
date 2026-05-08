"""Result objects and serialisation for the GB decomposition package.

Holds :class:`DecompositionResult` plus the pickle and NetCDF
serialisation helpers used by its ``save`` / ``load`` / ``to_netcdf`` /
``from_netcdf`` methods.

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

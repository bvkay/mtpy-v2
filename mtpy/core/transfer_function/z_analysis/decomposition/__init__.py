"""Magnetotelluric distortion analysis: a multi-tradition package.

The MT community has produced several independent intellectual
traditions for separating the regional 2-D inductive response from
local galvanic distortion. They differ in parameterisation, in the
rotational invariants they emphasise, and in how they classify
dimensionality. This package collects implementations of these
methods in a single, unified API so that a user can run any of them
on the same impedance tensor and compare results without leaving
mtpy-v2.

Traditions (foundational citations below):

- **Groom-Bailey / McNeice-Jones (GB / MJ)** — factorise the 2x2
  distortion matrix into four named geometric parameters
  (``gain``, ``twist``, ``shear``, ``anisotropy``) plus a regional
  strike, fit jointly across multiple periods at one site (GB) or
  across multiple sites with a shared strike (MJ).
- **Bahr / WAL / Marti** — rotational-invariant decomposition and
  dimensionality classifiers (Bahr's "skew" and class scheme;
  Weaver-Agarwal-Lilley invariants ``I1`` ... ``I7``; Marti's
  WALDIM dimensionality codes).
- **Lilley** — Mohr-circle representation of the impedance tensor
  with a graphical decomposition into amplitude, phase, and strike.
- **García & Jones** — explicitly 3-D extension of GB-style
  decomposition.

Currently implemented
---------------------
- **GB single-site** (:func:`decompose`): per-band TRF least-squares
  with multi-start optimisation, parametric bootstrap, and a
  user-selectable strategy for resolving the GB 90-degree symmetry.
- **MJ joint** (:func:`decompose_joint`): the multi-site extension
  with a shared regional strike across stations.
- **BCB single-site** (:func:`decompose_bibby`): Bibby-Caldwell-
  Brown decomposition. Fits a single real 2x2 distortion matrix
  ``C`` per period and band-averages it; the methodologically
  simplest comparison point for the parameterised GB result.
- **Disambiguation framework** (:mod:`.symmetries`): four named
  fold strategies (``geometric``, ``identity``, ``pt_aligned``,
  ``min_shear``) and a callable hook for user-defined strategies,
  exposed on :func:`decompose`.

Planned
-------
- Bahr 1991 decomposition and class scheme.
- Weaver-Agarwal-Lilley (2000) invariants.
- Marti et al. (2009) WALDIM dimensionality classifier.
- Lilley (1998) Mohr-circle decomposition.
- García & Jones (2002) 3-D distortion decomposition.
- A common :class:`DecompositionResult` flavour shared across all
  methods so cross-tradition comparison is a one-liner.

Package layout
--------------
- :mod:`.common` : math primitives, residual normalisations, band
  partitioning, and joint-input validation that any decomposition
  method in this package can reuse.
- :mod:`.results` : :class:`DecompositionResult` and the pickle /
  NetCDF serialisation helpers.
- :mod:`.symmetries` : GB-symmetry fold strategies and mode
  clustering / disambiguation helpers used by the multi-start
  optimiser.
- :mod:`.groom_bailey` : the GB single-site and MJ joint
  algorithms, public ``decompose*`` entry points, and the
  parametric-bootstrap machinery.
- :mod:`.bibby` : the Bibby-Caldwell-Brown single-site
  decomposition (:func:`decompose_bibby`).

See also
--------
:mod:`mtpy.core.transfer_function.z_analysis.distortion` :
    Bibby (2005) frequency-averaged single-distortion-matrix
    decomposition. A simpler, complementary method.
:mod:`mtpy.core.transfer_function.pt` :
    Phase tensor (Caldwell-Bibby-Brown 2004); distortion-invariant
    representation that is the natural reference for
    cross-tradition comparison.

References
----------
Bahr, K. (1991). Geological noise in magnetotelluric data: a
classification of distortion types. Physics of the Earth and
Planetary Interiors, 66, 24-38.

Bibby, H. M., Caldwell, T. G., & Brown, C. (2005). Determinable and
non-determinable parameters of galvanic distortion in magnetotellurics.
Geophysical Journal International, 163(3), 915-930.

Caldwell, T. G., Bibby, H. M., & Brown, C. (2004). The magnetotelluric
phase tensor. Geophysical Journal International, 158, 457-469.

García, X., & Jones, A. G. (2002). Decomposition of three-dimensional
magnetotelluric data. In Three-Dimensional Electromagnetics
(M. S. Zhdanov & P. E. Wannamaker, eds.), Methods in Geochemistry and
Geophysics, 35, 235-250.

Groom, R. W., & Bailey, R. C. (1989). Decomposition of magnetotelluric
impedance tensors in the presence of local three-dimensional galvanic
distortion. Journal of Geophysical Research: Solid Earth, 94(B2),
1913-1925.

Lilley, F. E. M. (1998). Magnetotelluric tensor decomposition: Parts I
and II. Geophysics, 63(6), 1885-1907.

Marti, A., Queralt, P., & Ledo, J. (2009). WALDIM: A code for the
dimensionality analysis of magnetotelluric data using the rotational
invariants of the magnetotelluric tensor. Computers & Geosciences,
35, 2295-2303.

McNeice, G. W., & Jones, A. G. (2001). Multisite, multifrequency
tensor decomposition of magnetotelluric data. Geophysics, 66(1),
158-173.

Weaver, J. T., Agarwal, A. K., & Lilley, F. E. M. (2000).
Characterization of the magnetotelluric tensor in terms of its
invariants. Geophysical Journal International, 141, 321-336.

Notes
-----
This package ships pure-Python implementations of the GB and MJ
algorithms. Validation has been performed against historical Fortran
implementations (Strike, McNeice-Jones); see the project's
documentation for tolerance specifications and the optimiser-gap
analysis.
"""

from .common import (  # noqa: F401  -- private helpers re-exported for backward compatibility
    _BandResult,
    _band_arrays_to_z,
    _calc_error,
    _convz2p,
    _convz2r,
    _estim_imp,
    _extract_bands,
    _extreme,
    _jkvar,
    _mat_multiply,
    _normalise_collection_input,
    _unpack_x,
    _unpack_x_joint,
    _validate_joint_input,
    _z_to_band_arrays,
)
from .groom_bailey import (  # noqa: F401  -- private helpers re-exported for backward compatibility
    _bootstrap_decompose,
    _build_bounds,
    _build_bounds_joint,
    _canonical_initial_guess,
    _compute_ci_percentile,
    _generate_starting_points,
    _objfun,
    _objfun_joint,
    _perturbed_initial_guess,
    _predict_z_from_primary_modes,
    _resample_residuals,
    _rotated_initial_guess,
    _solve_band,
    _solve_band_multistart,
    decompose,
    decompose_each_station,
    decompose_joint,
)
from .bibby import decompose_bibby
from .results import (  # noqa: F401  -- private helpers re-exported for backward compatibility
    BibbyResult,
    DecompositionResult,
    _desanitize_station_id,
    _sanitize_station_id,
)
from .symmetries import (  # noqa: F401  -- private helpers re-exported for backward compatibility
    _DEFAULT_MODE_TOLERANCE,
    _cluster_modes,
    _compute_mode_probabilities,
    _detect_band_disagreement,
    _detect_primary_mode_warning,
    _geometric_fold,
    _identity_fold,
    _min_shear_fold,
    _pt_aligned_fold,
    _resolve_disambiguation,
)

__all__ = [
    "BibbyResult",
    "DecompositionResult",
    "decompose",
    "decompose_bibby",
    "decompose_each_station",
    "decompose_joint",
]

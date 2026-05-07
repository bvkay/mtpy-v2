"""Groom-Bailey / McNeice-Jones magnetotelluric tensor decomposition.

This package provides single-site (Groom & Bailey, 1989) and multi-site
joint (McNeice & Jones, 2001) decomposition of the magnetotelluric
impedance tensor, recovering the regional 2-D impedance and the
galvanic distortion parameters at each site.

It is a sibling to the existing
:mod:`mtpy.core.transfer_function.z_analysis.distortion` module which
provides Bibby et al. (2005) decomposition. The two methods are
distinct: Bibby fits a single 2x2 distortion matrix per site by
frequency-averaging; Groom-Bailey factorises the distortion into named
geometric parameters (gain, twist, shear, anisotropy) and fits these
jointly with the regional strike and impedances across multiple
frequencies. For most use cases, Groom-Bailey gives a richer and more
interpretable result; Bibby is faster and more robust at sites with
poor multi-frequency coverage.

The package is organised into:

- :mod:`.common` : shared math, normalisation, and validation helpers.
- :mod:`.results` : :class:`DecompositionResult` and serialisation.
- :mod:`.symmetries` : GB symmetry / canonicalisation and mode
  disambiguation.
- :mod:`.groom_bailey` : single-site GB and joint MJ decomposition
  algorithms.

References
----------
Groom, R. W., & Bailey, R. C. (1989). Decomposition of magnetotelluric
impedance tensors in the presence of local three-dimensional galvanic
distortion. Journal of Geophysical Research: Solid Earth, 94(B2),
1913-1925.

McNeice, G. W., & Jones, A. G. (2001). Multisite, multifrequency
tensor decomposition of magnetotelluric data. Geophysics, 66(1),
158-173.

See also
--------
:mod:`mtpy.core.transfer_function.z_analysis.distortion` :
    Bibby (2005) decomposition.
:mod:`mtpy.core.transfer_function.pt` :
    Phase tensor (Caldwell et al. 2004); distortion-invariant
    representation.

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
from .results import (  # noqa: F401  -- private helpers re-exported for backward compatibility
    DecompositionResult,
    _desanitize_station_id,
    _sanitize_station_id,
)
from .symmetries import (  # noqa: F401  -- private helpers re-exported for backward compatibility
    _DEFAULT_MODE_TOLERANCE,
    _canonicalise_solution,
    _cluster_modes,
    _compute_mode_probabilities,
    _detect_band_disagreement,
    _detect_primary_mode_warning,
)

__all__ = [
    "DecompositionResult",
    "decompose",
    "decompose_each_station",
    "decompose_joint",
]

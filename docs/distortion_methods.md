# Distortion analysis in mtpy-v2

The MT community has produced several independent intellectual
traditions for separating the regional inductive response from
local galvanic distortion. They differ in parameterisation, in the
rotational invariants they emphasise, and in how they classify
dimensionality. This package collects implementations of these
methods in a single, unified API so that a user can run any of them
on the same impedance tensor and compare results without leaving
mtpy-v2.

## Intellectual traditions

### Groom-Bailey / McNeice-Jones (parameterised)

Factorise the 2x2 distortion tensor `C` into four named geometric
parameters (`gain`, `twist`, `shear`, `anisotropy`) plus a regional
strike, and fit them jointly across periods at one site (Groom &
Bailey 1989) or across stations with a shared strike (McNeice &
Jones 2001). The 90-degree GB symmetry is resolved by a configurable
disambiguation strategy. This is the most widely-used tradition in
practical MT and is the natural choice for any 2-D regional analysis.

### Bahr / Weaver-Agarwal-Lilley / Marti (invariant)

Build rotational invariants of `Z` and use them as direct
dimensionality classifiers without ever fitting a distortion model.
Bahr (1988, 1991) introduced the original "skew" and class scheme;
Weaver-Agarwal-Lilley (2000) gave the seven invariants `I1` ... `I7`
and the auxiliary `Q`; Marti et al. (2009) WALDIM uses them to
classify each period into 1-D / 2-D / 3-D-distorted-2-D / 3-D
regimes. Best used as a *diagnostic* layer ahead of the parametric
methods.

### Lilley (Mohr-circle)

Represent the in-phase and quadrature parts of `Z` as a pair of
Mohr circles (Lilley 1976, 1993, 2018). The circles' centres,
radii, and rotation angles are gauge-clean tensor invariants;
Lilley's improved seventh invariant `delta_beta` and the noise-
stability strike (the strike's standard deviation under bootstrap
resampling) are powerful diagnostics for sites that the parametric
GB / MJ methods would otherwise quietly mis-fit.

### Bibby-Caldwell-Brown (gauge-fixed)

Fit a single real 2x2 distortion `C` per period without any GB
parameterisation, then resolve the inherent scale ambiguity with an
explicit gauge choice (we use diagonal-unity in the strike frame).
Bibby et al. (2005) is the methodologically simplest distortion
method in the package — minimal modelling assumptions, minimal
parameters — and is the natural baseline against which the
parameterised GB result should agree.

### Garcia-Jones (3-D regional extension)

When the regional structure is genuinely 3-D, the GB / MJ assumption
of an anti-diagonal regional `Z` in some strike frame is violated
and the recovered C is biased. Garcia & Jones (2002) extend MJ by
leaving the regional `Z` fully 3-D (four free complex components
per period) while sharing it across two-or-more sites with site-
specific real distortion. This is the method to reach for when the
phase tensor or the WAL `I_7` invariant indicates a 3-D regional.

## Implemented methods

| Method | Function | Result class | Tradition | Status |
|---|---|---|---|---|
| GB single-site | `decompose` | `DecompositionResult` | GB / MJ | Production |
| MJ joint (legacy) | `decompose_joint` | `DecompositionResult` | GB / MJ | Production |
| MJ joint (v2 API) | `decompose_mcneice_jones` | `JointDecompositionResult` | GB / MJ | Phase 1 |
| Each-station GB | `decompose_each_station` | dict[str, `DecompositionResult`] | GB / MJ | Production |
| Bibby-Caldwell-Brown | `decompose_bibby` | `BibbyResult` | BCB | Production |
| Lilley Mohr-circle | `decompose_lilley` | `LilleyResult` | Lilley | Phase 1 |
| Marti WALDIM | `decompose_marti` | `MartiResult` | Bahr / WAL / Marti | Phase 1 |
| Garcia-Jones | `decompose_garcia_jones` | `GarciaJonesResult` | Garcia-Jones | Phase 1 |
| Irreducible decomposition | `irreducible_decomposition`, `gamma_field`, `principal_axis` | dict / tuple | (utility) | Production |

All methods are exposed at the package top level:

```python
from mtpy.core.transfer_function.z_analysis.decomposition import (
    decompose,
    decompose_mcneice_jones,
    decompose_bibby,
    decompose_lilley,
    decompose_marti,
    decompose_garcia_jones,
    irreducible_decomposition,
    gamma_field,
)
```

## Recommendations: when to use which

### "I have one site and want a regional 2-D `Z` and a `C` matrix."

Use `decompose`. Pick a disambiguation strategy that matches your
prior knowledge:

* `"geometric"` (default): historical GB convention, deterministic.
* `"identity"`: don't fold; report both branches if the data is
  ambiguous.
* `"pt_aligned"`: fold to the branch whose strike agrees with the
  phase tensor's principal axis.
* `"min_shear"`: prefer the branch with smaller `|shear|`.

### "I have multiple sites with a shared regional 2-D structure."

Use `decompose_mcneice_jones`. The shared regional strike is fitted
jointly per band, with per-site distortion. The Phase 1 API
(`JointDecompositionResult`) is the cleaner public surface;
`decompose_joint` is retained for backward compatibility.

### "I want a sanity check on the GB result without GB parametrisation."

Use `decompose_bibby`. It fits a single real `C` per period, then
band-averages it. Compare its `C` (in measurement frame) to the
GB-reconstructed C; large disagreement is a diagnostic of either
GB-symmetry mis-folding or genuinely non-GB distortion.

### "I want to know if the data is even 2-D before fitting GB."

Use `decompose_marti`. The WALDIM dimensionality classifier
returns an integer per period (1 = 1-D, 2 = 2-D, 3-7 = various
3-D / 2-D-distorted-3-D regimes). If most periods classify as 5
(pure 3-D), the GB / MJ assumption is broken and you should reach
for Garcia-Jones instead.

### "I want a noise-stability strike."

Use `decompose_lilley` and read `result.strike_std_rad`. Sites
with large strike standard deviation under bootstrap are unstable
in Lilley's noise-stability sense; the parametric strike from GB
on those sites should be treated as suspect.

### "The regional structure is genuinely 3-D."

Use `decompose_garcia_jones` with at least two sites. The 3-D
regional Z is fitted directly (not reduced to a 2-D anti-diagonal
form). On a 2-D regional GJ also works, and the recovered Z's
diagonal entries collapse close to zero; on a 3-D regional GJ
out-performs MJ noticeably.

### "I want to study the distortion field across a survey."

Use `irreducible_decomposition` (or the convenience helpers
`gamma_field`, `principal_axis`) on each site's recovered `C`.
The spin-2 shear `gamma = gamma_1 + i gamma_2` is the
station-by-station input to array-level analyses, including
E / B-mode decompositions analogous to those used in weak-lensing
cosmology.

## Known limitations

* **Gain non-identifiability.** The GB factorisation
  `C = gain * R T S R.T` has a structural `gain`-vs-regional-`Z`-
  amplitude trade-off: the optimiser can attribute amplitude to
  either, and the data alone cannot distinguish. At weak distortion
  (~5°) the optimiser's gain stays near 1 and the recovered `C`
  matches the truth tightly; at moderate or strong distortion
  (~15°+) the recovered gain drifts from 1.0 and the recovered `C`
  is biased even though `(strike, twist, shear)` are individually
  recovered exactly. Methods that don't parametrise gain (BCB,
  Garcia-Jones) sidestep this. See Garcia & Jones (2002) Section
  4.2 and Bibby et al. (2005) for the formal discussion.

* **Garcia-Jones gain / anisotropy recovery.** Garcia & Jones
  (2002) demonstrate that recovering the gain and anisotropy
  simultaneously with twist / shear is unstable for
  controlled-source MT (their Figure 7c, 9). Phase 1 fixes
  ``gain = 1`` and ``anisotropy = 0`` for that reason; Phase 2
  experiments may revisit this.

* **Marti case 3c (diagonal regional).** The Marti 2009 case 3c
  discriminator requires the Bahr `xi_4` / `eta_4` test that
  presupposes a Bahr-style decomposition; Phase 1 conflates 3c with
  case 2 (pure 2-D). Listed as a Phase-2 follow-up.

* **Lilley `delta_beta` only, not the seventh WAL invariant.** The
  Lilley module returns `delta_beta = beta_q - beta_p` (radians)
  rather than the seventh WAL invariant `I_7` directly, because
  `I_7` is undefined when the auxiliary `Q` is small whereas
  `delta_beta` is always defined. For the WAL form, use
  `decompose_marti` instead.

* **3-D synthetics.** The synthetic test harness
  (`tests/synthetics.py`) does not call an external 3-D forward
  solver (ModEM, MARE2DEM); the 3-D regional case is generated by
  adding 25 % diagonals at ±45° phase shift to the 2-D base, a
  Born-approximation analogue suitable for the harness's purpose
  but not a substitute for genuine 3-D modelling when validating
  against real data.

* **NetCDF round-trip on `DecompositionResult`.** Six pre-existing
  tests in `tests/core/transfer_function/z_analysis/test_decomposition.py`
  fail because the netCDF4 library cannot store boolean metadata
  attributes (it expects integer types). The failure is unrelated
  to the multi-tradition refactor; remediation (cast booleans to
  `int8` at the NetCDF boundary) is on the planned list.

## References

Bahr, K. (1988). Interpretation of the magnetotelluric impedance
tensor: regional induction and local telluric distortion.
*Journal of Geophysics* 62, 119-127.

Bahr, K. (1991). Geological noise in magnetotelluric data: a
classification of distortion types. *Physics of the Earth and
Planetary Interiors* 66, 24-38.

Bibby, H. M., Caldwell, T. G., & Brown, C. (2005). Determinable and
non-determinable parameters of galvanic distortion in
magnetotellurics. *Geophysical Journal International* 163(3),
915-930.

Caldwell, T. G., Bibby, H. M., & Brown, C. (2004). The
magnetotelluric phase tensor. *Geophysical Journal International*
158, 457-469.

Garcia, X., & Jones, A. G. (2002). Decomposition of three-
dimensional magnetotelluric data. In *Three-Dimensional
Electromagnetics* (M. S. Zhdanov & P. E. Wannamaker, eds.),
Methods in Geochemistry and Geophysics, 35, 235-250.

Garcia, X., Boerner, D., & Pedersen, L. B. (2003). Electric and
magnetic galvanic distortion decomposition of tensor CSAMT data.
*Geophysical Journal International* 154, 957-969.

Groom, R. W., & Bailey, R. C. (1989). Decomposition of
magnetotelluric impedance tensors in the presence of local three-
dimensional galvanic distortion. *Journal of Geophysical Research:
Solid Earth* 94(B2), 1913-1925.

Lilley, F. E. M. (1976). Diagrams for magnetotelluric data.
*Geophysics* 41(4), 766-770.

Lilley, F. E. M. (1993). Magnetotelluric analysis using Mohr
circles. *Geophysics* 58(10), 1498-1506.

Lilley, F. E. M. (1998). Magnetotelluric tensor decomposition:
Parts I and II. *Geophysics* 63(6), 1885-1907.

Lilley, F. E. M. (2018). The magnetotelluric tensor: improved
invariants for its decomposition, especially the 7th. *Exploration
Geophysics* 49(5), 622-636.

Marti, A., Queralt, P., & Ledo, J. (2009). WALDIM: A code for the
dimensionality analysis of magnetotelluric data using the
rotational invariants of the magnetotelluric tensor. *Computers &
Geosciences* 35, 2295-2303.

McNeice, G. W., & Jones, A. G. (2001). Multisite, multifrequency
tensor decomposition of magnetotelluric data. *Geophysics* 66(1),
158-173.

Smith, J. T. (1995). Understanding telluric distortion matrices.
*Geophysical Journal International* 122, 219-226.

Weaver, J. T., Agarwal, A. K., & Lilley, F. E. M. (2000).
Characterization of the magnetotelluric tensor in terms of its
invariants. *Geophysical Journal International* 141, 321-336.

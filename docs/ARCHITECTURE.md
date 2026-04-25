# Architecture — GB decomposition contribution to mtpy-v2

The shape of the work, the boundaries, the data flow.

---

## Where the new code lives

```
mtpy-v2/
├── mtpy/
│   ├── core/
│   │   ├── transfer_function/
│   │   │   ├── z.py                          # existing: Z type
│   │   │   ├── pt.py                         # existing: PhaseTensor
│   │   │   └── z_analysis/
│   │   │       ├── distortion.py             # existing: Bibby 2005
│   │   │       ├── zinvariants.py            # existing: WAL
│   │   │       ├── niblettbostick.py         # existing
│   │   │       └── decomposition.py          # NEW: GB / MJ
│   │   ├── mt.py                             # existing: MT
│   │   └── mt_collection.py                  # existing: MTCollection
│   └── imaging/
│       └── plot_decomposition.py             # NEW: Session 7
├── tests/
│   ├── conftest.py                           # existing: fixtures
│   ├── core/
│   │   └── transfer_function/
│   │       └── z_analysis/
│   │           ├── test_distortion.py        # existing
│   │           ├── test_zinvariants.py       # existing
│   │           └── test_decomposition.py     # NEW
│   └── cross_validation/
│       └── test_against_fortran.py           # NEW: Session 4+
├── CLAUDE.md                                 # this fork's constitution
└── docs/
    ├── ARCHITECTURE.md                       # this file
    ├── OPERATIONAL.md
    └── WORKFLOW.md
```

The `decomposition.py` file may eventually grow into a subpackage
(`decomposition/__init__.py` plus `groom_bailey.py`, `mcneice_jones.py`,
`bayesian.py`) if it gets large enough. Single file is the starting
point.

The `~/MT_Decomp` repo is *not* part of this tree. It's a separate
clone elsewhere on the developer's machine. See `CLAUDE.md` for the
cross-repo workflow.

---

## Public API surface

Three entry points, mirroring the existing `find_distortion` /
`remove_distortion` pattern:

**Function-level (canonical)**:

```python
from mtpy.core.transfer_function.z_analysis.decomposition import decompose

result = decompose(
    z: Z,
    periods: tuple[float, float] | None = None,
    bandwidth: float = 1.0,
    overlap: float = 0.0,
    norm_type: str = "GAVSD2",
    bounds_override: dict | None = None,
    initial_guess: dict | None = None,
    realisations: int = 0,
    seed: int | None = None,
) -> DecompositionResult
```

```python
from mtpy.core.transfer_function.z_analysis.decomposition import decompose_joint

result = decompose_joint(
    collection: MTCollection,  # or list[MT]
    ...,
) -> DecompositionResult
```

**Method-level (convenience)**:

```python
result = z.decompose(...)        # thin wrapper on Z
result = mt.decompose(...)       # thin wrapper on MT (delegates to MT.Z)
result = collection.decompose(...)  # thin wrapper on MTCollection
```

The method-level wrappers are added in a later session, *not* in
Session 1. Session 1 establishes only the function-level entry points
as stubs.

---

## The DecompositionResult dataclass

The shape, declared early to lock the contract:

```python
@dataclass
class DecompositionResult:
    """Result of a Groom-Bailey / McNeice-Jones decomposition.

    Attributes
    ----------
    parameters : xarray.Dataset
        Per-period decomposition parameters. Coordinates: period.
        Data variables: strike, twist, shear, gain, anisotropy,
        plus error counterparts (strike_error, ...). For multi-site
        joint fits, an additional 'station' dimension.
    regional_z : Z
        The decomposed regional impedance, as a fresh Z object with
        propagated errors.
    chi_squared : xarray.DataArray
        Per-period (or per-band) χ² values. Coordinate: period.
    rms_misfit : float
        Overall RMS misfit.
    method : str
        "groom_bailey", "mcneice_jones_joint", etc.
    options : dict
        The options dict the result was computed with (for
        provenance).
    metadata : dict
        Additional provenance: software version, git hash, scipy
        versions, RNG seed, source MT/Z identifier.
    """
```

xarray-backed because:

- Per-period parameters need coordinates. Plain dicts of NumPy
  arrays lose track of which value goes with which period.
- The Bayesian extension (Track 2) will add a `sample` dimension to
  these arrays. xarray handles that natively without API change.
- mtpy-v2's `TFBase` already uses xarray throughout. Our result
  matches the project's pattern.
- Free `to_netcdf()` / `to_dataframe()` for users who want them.

`regional_z` is a real `Z` instance (not a numpy array, not a dict)
so users can chain it through the rest of mtpy: `result.regional_z.
plot_resistivity_phase()`, `result.regional_z.write(...)`, etc.

---

## Public/private boundary

Public (visible to users via `from
mtpy.core.transfer_function.z_analysis.decomposition import ...`):

- `decompose(z: Z, ...)`
- `decompose_joint(collection: MTCollection, ...)`
- `DecompositionResult`

Private (underscore-prefixed, not in `__all__`):

- `_solve_band(...)` — single-band optimiser invocation
- `_pauli_alpha_from_z(...)` — Pauli-spin combinations from Z
- `_sigma_to_al_dev(...)` — error propagation Z→alpha
- `_build_initial_x(...)`, `_build_bounds(...)` — optimiser setup
- `_unpack_x(...)` — state vector unpacking
- All of the GB89 algebraic helpers

The boundary is enforced by docstrings and `__all__`; Python doesn't
enforce it, but reviewers will.

---

## Validation strategy

Three test layers, in order of how much they test:

**Type-contract tests** (Session 1).
The Z type-translation pattern (read via `.z`, write by
constructing fresh Z) is exercised. `DecompositionResult` round-
trips through serialisation. No numerics; this just locks the
contracts.

**Synthetic known-answer tests** (Sessions 2-6).
Generate Z from known parameters via the GB89 forward model;
decompose; check parameters recovered within tolerance. Independent
of the Fortran reference. Establishes that the implementation
solves the math.

**Continuity tests** (Sessions 4+, cross-repo).
Compare against the Fortran reference at `~/MT_Decomp` for the
canonical input. Validates that we produce equivalent results to
historical Strike runs. Uses Task 7 tolerances. Skips when
`~/MT_Decomp` isn't present.

The Bayesian work in Track 2 adds a fourth layer: posterior
predictive checks. Not in scope for Track 1.

---

## Coordinate and unit conventions

`Z` has internal subtleties our code must respect. The contract:

**Reads**: always go through `z_object.z` (user-facing, post-scale).
Never `z_object._dataset.transfer_function` (internal,
pre-scale).

**Writes**: always construct fresh `Z(z=..., z_error=...,
frequency=...)`. Never modify `_dataset.transfer_function`
in-place.

**Coordinate frame**: documented per-result. `MT.coordinate_reference_
frame` selects between NED (x=N, y=E, z=down) and ENU. Our strike
convention is "clockwise from x-axis" where x is defined by the
station's coordinate reference frame. This matches `find_distortion`'s
existing `clockwise=True` flag.

**Strike numerical convention**: regional azimuth in degrees,
canonical range `[0°, 180°)`. The 90° symmetry is folded by the
canonicalisation; the equivalence-aware comparison utility
recognises both branches.

**Static shift convention**: GB's `g` (gain) is dimensionally a
static shift. `MT.remove_static_shift(ss_x, ss_y)` uses factors
that multiply apparent resistivity. Our `g` and their `ss` agree:
applying `MT.remove_static_shift(g_x, g_y)` after our decomposition
recovers the un-shifted regional impedance. Tested.

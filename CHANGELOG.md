# Changelog

All notable changes to the mtpy-v2 fork are recorded in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning will follow [Semantic Versioning](https://semver.org/) once a
release is cut from this branch.

## [Unreleased] — distortion-as-signal refactor

This branch refactors the `decomposition` module into a package that
hosts distortion-analysis methods from multiple intellectual
traditions in MT, and lays the groundwork for adding
non-Groom-Bailey methods (Bahr, WAL, Lilley, García & Jones) under
the same API.

### Changed
- **Restructured `decomposition.py` into a package** at
  `mtpy/core/transfer_function/z_analysis/decomposition/`. The
  monolithic 5400-line module is split into:
  - `__init__.py` — public API and re-exports for backward
    compatibility.
  - `common.py` — tensor algebra primitives, conversions, residual
    helpers, optimiser plumbing, period-band partitioning, and
    joint-input validation. Method-agnostic; reusable by future
    decomposition methods.
  - `results.py` — `DecompositionResult` and pickle / NetCDF
    serialisation.
  - `symmetries.py` — GB 90-degree symmetry handling and multi-start
    mode clustering / disambiguation.
  - `groom_bailey.py` — single-site GB and joint MJ algorithms
    plus parametric bootstrap.
- **Renamed `_canonicalise_solution` → `_geometric_fold`** to make
  room for sibling fold strategies. The `canonicalise=True/False`
  flag on `decompose()` is preserved as a backward-compatibility
  shim that maps to the new `disambiguation` argument.
- **Module docstrings** now describe each module's role within the
  multi-tradition package and cite the foundational papers.

### Added
- **Disambiguation strategy framework** (`symmetries.py`,
  `groom_bailey.py`). New `disambiguation` argument on
  `decompose()` accepts `"geometric"` (default, historical
  behaviour), `"identity"` (no fold; both branches visible),
  `"pt_aligned"` (choose the branch closest to the phase tensor's
  principal axis at the band's median period), `"min_shear"`
  (low-shear-preferring tiebreaker), or any
  `fold(strike, twist, shear) -> (strike, twist, shear)` callable.
  Mode clustering and the bootstrap path always use the geometric
  fold internally.
- **`DecompositionResult.alternate_branch()`** returns a fresh
  result on the GB-symmetry-equivalent branch
  (`strike + 90 mod 180`, `-shear`, with `Z_xy`/`Z_yx` swapped in
  `regional_z`). Idempotent; preserves chi-squared, RMS, errors,
  metadata, and the reconstructed C tensor.
- **CHANGELOG.md** (this file).

### Tests
- 262 pass across the GB / MJ / disambiguation / alternate-branch
  suites. The 6 `TestDecompositionResultNetcdf` failures pre-date
  this branch (a netCDF4 library limitation around boolean
  attributes) and are unrelated to the refactor.
- New test files: `tests/test_disambiguation.py`,
  `tests/test_alternate_branch.py`.

### Planned
- Implement Bahr (1991) decomposition and dimensionality classifier.
- Implement Weaver-Agarwal-Lilley (2000) rotational invariants and
  Marti et al. (2009) WALDIM dimensionality codes.
- Implement Lilley (1998) Mohr-circle decomposition.
- Implement García & Jones (2002) 3-D distortion decomposition.
- Adopt a shared `DecompositionResult` flavour across every method
  so cross-tradition comparison is a one-liner.
- Resolve the pre-existing `TestDecompositionResultNetcdf` failures
  by switching boolean metadata attributes to `int8` at the NetCDF
  boundary.

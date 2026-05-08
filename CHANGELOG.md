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
- **Bibby-Caldwell-Brown decomposition**
  (`decompose_bibby`, `BibbyResult`). Fits a single real 2x2
  distortion matrix per period and band-averages it. Uses the
  diagonal-unity gauge to fix the BCB non-determinable scale.
  Methodologically the simplest distortion-analysis method in
  the package and a comparison baseline for the parameterised
  Groom-Bailey result.
- **McNeice-Jones (Phase 1)**
  (`decompose_mcneice_jones`, `JointDecompositionResult`). New
  API for multi-site joint GB decomposition. Takes parallel
  `z_objs` / `site_ids` lists, supports the disambiguation
  framework, and returns a purpose-built result with per-site
  distortion dicts and per-band shared strikes. Thin façade
  over the existing joint optimisation machinery in
  `groom_bailey.py` (no algorithm reimplementation). Phase 2
  will add validation against published results (BC87) and
  edge-case coverage.
- **CHANGELOG.md** (this file).

### Tests
- 242 pass by default across the GB / MJ / BCB / disambiguation /
  alternate-branch suites; 2 opt-in slow validations skip
  cleanly. The 6 `TestDecompositionResultNetcdf` failures pre-date
  this branch (a netCDF4 library limitation around boolean
  attributes) and are unrelated to the refactor.
- New test files: `tests/test_disambiguation.py`,
  `tests/test_alternate_branch.py`, `tests/test_bibby.py`,
  `tests/test_mcneice_jones.py`,
  `tests/test_mj_bc87_validation.py`.

### Documentation
- `docs/decomposition_validation.md` records the validation
  strategy for each method and what is and is not validated
  against published results. The BC87 strike-magnitude
  comparison against McNeice & Jones (2001) Figure 12 is
  explicitly Phase-2 work.

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
- Investigate the BC87 LIT-line strike discrepancy
  (`test_mj_bc87_strike_recovery` finds ~5° in the geometric-fold
  convention vs Fig 12's ~25–40°). Likely follow-ups: per-period
  narrow-band fits, higher `n_starts`, side-by-side run against
  the original McNeice-Jones Fortran on the same EDIs.

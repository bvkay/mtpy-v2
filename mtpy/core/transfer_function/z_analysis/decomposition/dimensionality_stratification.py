"""Trust-tier stratification of the continental observables table.

Galvanic-distortion decomposition (GB / MJ) assumes a 2-D regional
Earth with local 3-D distortion. At sites where this assumption is
*violated* — true 3-D regional, very high phase-tensor skew, strong
magnetic distortion, optimisation that flagged competing modes —
the recovered C tensor carries unknown contamination from model
misfit. Continental aggregation that pools all sites equally then
mixes signal with model-violation noise.

This module makes the model-validity stratification systematic and
reproducible. Every row of the
:class:`...results.ObservableTable` is assigned to one of four
trust strata:

    high_trust       Tier A. 2-D + galvanic, model ideal.
    moderate_trust   Tier B. Borderline 2-D, mild model strain.
    low_trust        Tier C. Recoveries to be interpreted with care.
    excluded         Tier D. Model invalid; results not recoverable.

A continuous companion :func:`trust_score` (range ``[0, 1]``) is
the smooth-edged version of the same rules, so continental maps
can be *tinted* by trust rather than hard-filtered. A sigmoid is
used at every numeric threshold so the score has no
discontinuities and is suitable for a colour scale.

Important notes (please read)
=============================
1. **Stratification is a research choice, not a measurement.**
   The trust thresholds in :data:`DEFAULT_TRUST_RULES` are the
   canonical Paper-1 set; they encode editorial judgements about
   when the GB / MJ model can be trusted. Different choices of
   thresholds will change which sites land in each tier.

2. **The default thresholds are documented in the source.** They
   will be reported verbatim in Paper 1's methods section. Anyone
   adopting this module to support a paper of their own should
   either accept the defaults *and* report them, or define their
   own and report those.

3. **Sensitivity analysis is mandatory before drawing conclusions
   from a single stratification.** Use
   :func:`threshold_sensitivity` to sweep one threshold across a
   plausible range and confirm that the headline result (the
   continental map, the regional mean, the cross-correlation
   coefficient) is stable. A finding that vanishes when the
   ``PT_abs_beta_deg`` threshold goes from 3° to 4° is not robust.

4. **Excluded sites are NOT discarded from the underlying data.**
   They are *flagged*. The original
   :class:`...results.ObservableTable` is not modified by any
   function in this module; the stratification is layered on top
   so that downstream analyses can choose to include the
   excluded rows (with the appropriate caveats) or exclude them.

Default trust-rule structure
============================
:data:`DEFAULT_TRUST_RULES` is a dict keyed by stratum name. Each
stratum is itself a dict with two keys:

    ``logic``    : ``"all"`` (every criterion must match) or
                   ``"any"`` (at least one criterion must match).
    ``criteria`` : either a dict ``{column: (op, value)}`` (used
                   with ``"all"`` logic) or a list of triples
                   ``[(column, op, value), ...]`` (used with
                   ``"any"`` logic — a list is required because
                   the same column may appear multiple times,
                   e.g. ``"WALDIM_case == 1"`` and
                   ``"WALDIM_case in [6, 7]"``).

Operators: ``"<"``, ``"<="``, ``">"``, ``">="``, ``"=="``,
``"!="``, ``"in"``. The right-hand value type is whatever the
operator expects; for ``"in"`` it is a sequence.

Stratum priority: a row is assigned to ``"excluded"`` first if its
``"any"`` rules trigger; otherwise to ``"high_trust"`` if all of
those criteria pass; otherwise to ``"moderate_trust"`` if all of
those pass; otherwise to ``"low_trust"``. This priority makes
``"excluded"`` an absolute override (e.g. WALDIM = 1 always lands
in ``"excluded"``, regardless of any other column).

References
----------
The Paper-1 framing — distortion as a signal — and the per-method
caveats are documented in the parent ``mt_decomp`` repository's
``docs/notes/Paper_01.md``. The dimensionality classifiers used by
the rules:

* ``WALDIM_case`` : Marti et al. (2009) seven-case classifier
  (:mod:`...marti`).
* ``Lilley_category`` : Lilley (2020) phase-tensor / Bahr-
  eigenvector / Mohr-circle unified classifier
  (:mod:`...lilley_dimensionality`).
* ``magnetic_distortion_flag`` : Garcia 2003 / Chave 1994
  heuristic flag (:mod:`...magnetic_distortion_diagnostic`).

See Also
--------
:mod:`...continental_observables` : the long-format input table.
:mod:`...spatial_coherence` : the analysis layer that consumes
    the trust-filtered table to test for spatial structure.
"""

from __future__ import annotations

import copy as _copy
import datetime as _datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .results import ObservableTable, StratificationSummary

if TYPE_CHECKING:  # pragma: no cover -- type-only imports
    pass


__all__ = [
    "DEFAULT_SUMMARY_OBSERVABLES",
    "DEFAULT_TRUST_RULES",
    "STRATA_ORDER",
    "apply_trust_filter",
    "stratification_summary",
    "stratify_table",
    "threshold_sensitivity",
    "trust_score",
]


# ---------------------------------------------------------------------------
# Canonical trust rules
# ---------------------------------------------------------------------------


DEFAULT_TRUST_RULES: dict[str, dict[str, Any]] = {
    "high_trust": {
        "logic": "all",
        "criteria": {
            "WALDIM_case": ("in", [2, 3]),
            "Lilley_category": ("in", ["2D", "3D-2D"]),
            "PT_abs_beta_deg": ("<", 3.0),
            "magnetic_distortion_flag": ("==", "low_risk"),
            "GB_mode_warning": ("==", False),
            "cross_method_strike_disagreement_deg": ("<", 5.0),
            "GB_rms_misfit": ("<", 2.0),
        },
    },
    "moderate_trust": {
        "logic": "all",
        "criteria": {
            "WALDIM_case": ("in", [2, 3, 4]),
            "PT_abs_beta_deg": ("<", 6.0),
            "magnetic_distortion_flag": (
                "in", ["low_risk", "moderate_risk"]
            ),
            "cross_method_strike_disagreement_deg": ("<", 15.0),
        },
    },
    "excluded": {
        "logic": "any",
        "criteria": [
            ("WALDIM_case", "==", 1),
            ("WALDIM_case", "in", [6, 7]),
            ("magnetic_distortion_flag", "==", "high_risk"),
            ("PT_abs_beta_deg", ">=", 6.0),
            ("Lilley_category", "==", "indeterminate"),
        ],
    },
}
"""The Paper-1 canonical trust rules. See module docstring for the
structure and the rationale. ``"low_trust"`` is the default
catch-all and does not appear in this dict; rows that match
neither ``"excluded"``, ``"high_trust"``, nor ``"moderate_trust"``
land there.
"""


STRATA_ORDER: list[str] = [
    "high_trust", "moderate_trust", "low_trust", "excluded",
]
"""The canonical order of strata for reporting. Matches the
quality progression A → B → C → D.
"""


DEFAULT_SUMMARY_OBSERVABLES: list[str] = [
    "discordance_deg",
    "gamma_magnitude",
    "C_minus_I_F",
    "PT_abs_beta_deg",
    "GB_rms_misfit",
    "cross_method_strike_disagreement_deg",
]
"""Observables included in :class:`StratificationSummary` by
default. Pass ``summary_observables=`` to override.
"""


# ---------------------------------------------------------------------------
# Operator engines
# ---------------------------------------------------------------------------


def _is_missing(value: Any) -> bool:
    """True for any of: ``None``, ``pd.NA``, ``NaN``, ``pd.NaT``."""
    if value is None:
        return True
    if value is pd.NA:
        return True
    try:
        if isinstance(value, float) and np.isnan(value):
            return True
    except TypeError:  # pragma: no cover -- defensive
        pass
    if isinstance(value, np.floating) and np.isnan(value):
        return True
    return False


def _matches(value: Any, op: str, target: Any) -> bool:
    """Hard-edged criterion match. Missing → ``False``."""
    if _is_missing(value):
        return False
    if op == "in":
        return value in target
    if op == "==":
        return bool(value == target)
    if op == "!=":
        return bool(value != target)
    if op == "<":
        return bool(value < target)
    if op == "<=":
        return bool(value <= target)
    if op == ">":
        return bool(value > target)
    if op == ">=":
        return bool(value >= target)
    raise ValueError(f"_matches: unknown op {op!r}")


def _sigmoid(x: float) -> float:
    """Numerically stable logistic."""
    if x >= 0:
        return float(1.0 / (1.0 + np.exp(-x)))
    e = np.exp(x)
    return float(e / (1.0 + e))


def _criterion_score(
    value: Any, op: str, target: Any, *, scale: float | None = None,
) -> float:
    """Smoothed criterion score in ``[0, 1]``.

    For numeric inequality operators the score is a sigmoid
    centred at the threshold with width ``scale``; the default
    ``scale = max(0.05 * |target|, 0.3)`` gives a transition over
    ~``2 * scale`` units and matches the categorical hard-edged
    rule asymptotically. For categorical operators the score is
    binary.

    A missing value returns ``0.0`` (the cell did not contribute
    evidence).
    """
    if _is_missing(value):
        return 0.0
    if op == "in":
        return 1.0 if value in target else 0.0
    if op == "==":
        return 1.0 if value == target else 0.0
    if op == "!=":
        return 1.0 if value != target else 0.0
    if op in ("<", "<="):
        s = scale if scale is not None else max(0.05 * abs(target), 0.3)
        return _sigmoid((target - value) / s)
    if op in (">", ">="):
        s = scale if scale is not None else max(0.05 * abs(target), 0.3)
        return _sigmoid((value - target) / s)
    raise ValueError(f"_criterion_score: unknown op {op!r}")


def _stratum_matches(row: dict[str, Any], stratum_rules: dict) -> bool:
    """Test whether ``row`` matches a single tier's rules."""
    logic = stratum_rules.get("logic", "all")
    criteria = stratum_rules.get("criteria", [])
    if logic == "all":
        # ``criteria`` is dict[col -> (op, target)]; AND across all.
        for col, (op, target) in criteria.items():
            if not _matches(row.get(col), op, target):
                return False
        return True
    if logic == "any":
        # ``criteria`` is list[(col, op, target)]; OR across all.
        for col, op, target in criteria:
            if _matches(row.get(col), op, target):
                return True
        return False
    raise ValueError(f"_stratum_matches: unknown logic {logic!r}")


# ---------------------------------------------------------------------------
# Per-row stratum / trust score
# ---------------------------------------------------------------------------


def _row_to_dict(row) -> dict[str, Any]:
    """Coerce a pandas Series / dict / namedtuple-like row to a dict.

    We accept several row representations because callers iterate
    differently: ``df.itertuples`` yields namedtuples,
    ``df.iterrows`` yields ``(idx, Series)``, and tests pass
    plain dicts.
    """
    if isinstance(row, dict):
        return row
    if isinstance(row, pd.Series):
        return row.to_dict()
    if hasattr(row, "_asdict"):
        return dict(row._asdict())
    raise TypeError(
        f"_row_to_dict: unsupported row type {type(row).__name__}"
    )


def _assign_stratum(
    row: dict[str, Any], rules: dict[str, Any]
) -> str:
    """Priority-ordered tier assignment.

    ``"excluded"`` first (absolute override), then ``"high_trust"``,
    then ``"moderate_trust"``, defaulting to ``"low_trust"``.
    """
    if "excluded" in rules and _stratum_matches(row, rules["excluded"]):
        return "excluded"
    if (
        "high_trust" in rules
        and _stratum_matches(row, rules["high_trust"])
    ):
        return "high_trust"
    if (
        "moderate_trust" in rules
        and _stratum_matches(row, rules["moderate_trust"])
    ):
        return "moderate_trust"
    return "low_trust"


def trust_score(
    observable_table_row,
    rules: dict[str, Any] | None = None,
    *,
    scales: dict[str, float] | None = None,
) -> float:
    """Continuous trust score in ``[0, 1]`` for a single row.

    Smooth-edged version of the ``"high_trust"`` rules: every
    criterion contributes a score in ``[0, 1]`` (sigmoid for
    numeric thresholds, binary for categorical), and the overall
    score is the geometric mean. A categorical mismatch (e.g.
    ``magnetic_distortion_flag == "high_risk"``) zeroes-out the
    score; numeric near-misses smoothly reduce it.

    Parameters
    ----------
    observable_table_row : pd.Series, dict, or namedtuple-like
        A single row of the observable table.
    rules : dict, optional
        Override :data:`DEFAULT_TRUST_RULES`. Only the
        ``"high_trust"`` entry is consulted.
    scales : dict[str, float], optional
        Per-column sigmoid widths. Default
        ``max(0.1 * |threshold|, 0.5)`` per criterion; tighter
        values give a sharper edge.

    Returns
    -------
    float
        Geometric mean of per-criterion scores. Range ``[0, 1]``.
        A score of ``1.0`` means every criterion is well above
        its threshold; ``0.0`` means at least one categorical
        criterion is violated *and* there is no missing data
        elsewhere.

    Notes
    -----
    To avoid ``log(0)`` in the geometric mean, individual
    criterion scores are clipped to ``1e-12``. The smallest
    achievable trust score for a 7-criterion rule set is
    therefore ``1e-12 ** (1/7) ≈ 0.02``, well below any
    practical "trust" threshold.
    """
    rules = rules if rules is not None else DEFAULT_TRUST_RULES
    high = rules.get("high_trust", {})
    criteria = high.get("criteria", {})
    if not criteria:
        return 1.0
    row = _row_to_dict(observable_table_row)
    scales = scales or {}
    scores: list[float] = []
    for col, (op, target) in criteria.items():
        s = _criterion_score(
            row.get(col), op, target, scale=scales.get(col),
        )
        scores.append(max(s, 1e-12))
    log_scores = np.log(np.asarray(scores))
    return float(np.exp(np.mean(log_scores)))


# ---------------------------------------------------------------------------
# Table-level stratification
# ---------------------------------------------------------------------------


def _stratum_per_row(
    df: pd.DataFrame, rules: dict[str, Any]
) -> np.ndarray:
    """Vectorise-ish stratum assignment: returns an array of stratum
    names of length ``len(df)``.
    """
    out = np.empty(len(df), dtype=object)
    for i, row in enumerate(df.itertuples(index=False)):
        out[i] = _assign_stratum(row._asdict(), rules)
    return out


def stratify_table(
    observable_table: ObservableTable,
    rules: dict[str, Any] | None = None,
) -> dict[str, ObservableTable]:
    """Partition the rows of ``observable_table`` into trust strata.

    Returns a dict ``{stratum_name: ObservableTable}``. Every row
    of the input lands in exactly one stratum. The returned
    sub-tables share the *same* DataFrame columns and dtypes as
    the input; only rows are filtered. Each sub-table's metadata
    records the stratum name and the rules used.

    The input ``observable_table`` is **not modified** (the
    DataFrame is sliced, not mutated). Strata that have zero
    rows still appear in the returned dict with an empty
    DataFrame so callers can iterate the canonical four strata
    without conditional checks.

    Parameters
    ----------
    observable_table : ObservableTable
    rules : dict, optional
        Override :data:`DEFAULT_TRUST_RULES`.

    Returns
    -------
    dict[str, ObservableTable]
        Keys are :data:`STRATA_ORDER`.
    """
    rules = rules if rules is not None else DEFAULT_TRUST_RULES
    df = observable_table.dataframe
    if len(df) == 0:
        return {
            s: ObservableTable(dataframe=df.iloc[0:0].copy(), metadata={})
            for s in STRATA_ORDER
        }

    strata_arr = _stratum_per_row(df, rules)
    out: dict[str, ObservableTable] = {}
    base_meta = dict(observable_table.metadata or {})
    for s in STRATA_ORDER:
        sub = df[strata_arr == s].copy()
        meta = dict(base_meta)
        meta["stratum"] = s
        meta["rules_used"] = _copy.deepcopy(rules)
        out[s] = ObservableTable(dataframe=sub, metadata=meta)
    return out


def apply_trust_filter(
    observable_table: ObservableTable,
    *,
    min_trust: float = 0.5,
    rules: dict[str, Any] | None = None,
    scales: dict[str, float] | None = None,
) -> ObservableTable:
    """Subset the table to rows with ``trust_score >= min_trust``.

    The original table is **not modified**. A *new*
    :class:`ObservableTable` is returned; its DataFrame is a
    fresh copy with the surviving rows.

    Parameters
    ----------
    observable_table : ObservableTable
    min_trust : float, default 0.5
        Threshold on :func:`trust_score`. Use ``0.5`` for a
        balanced choice; ``0.7`` for a strict subset that
        includes only "well above all thresholds" sites.
    rules : dict, optional
    scales : dict[str, float], optional
        Per-column sigmoid widths for :func:`trust_score`.

    Returns
    -------
    ObservableTable
    """
    rules = rules if rules is not None else DEFAULT_TRUST_RULES
    df = observable_table.dataframe
    if len(df) == 0:
        return ObservableTable(
            dataframe=df.copy(),
            metadata={
                **(observable_table.metadata or {}),
                "trust_filter_applied": True,
                "trust_filter_min": float(min_trust),
            },
        )

    scores = np.empty(len(df))
    for i, row in enumerate(df.itertuples(index=False)):
        scores[i] = trust_score(
            row._asdict(), rules=rules, scales=scales,
        )
    keep = scores >= min_trust
    sub_df = df[keep].copy()
    meta = dict(observable_table.metadata or {})
    meta.update(
        {
            "trust_filter_applied": True,
            "trust_filter_min": float(min_trust),
            "trust_filter_kept_rows": int(np.sum(keep)),
            "trust_filter_dropped_rows": int(np.sum(~keep)),
        }
    )
    return ObservableTable(dataframe=sub_df, metadata=meta)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _safe_mean(s: pd.Series) -> float:
    v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(v)
    if not finite.any():
        return float("nan")
    return float(np.mean(v[finite]))


def _safe_std(s: pd.Series) -> float:
    v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(v)
    if finite.sum() < 2:
        return float("nan")
    return float(np.std(v[finite], ddof=1))


def stratification_summary(
    observable_table: ObservableTable,
    *,
    rules: dict[str, Any] | None = None,
    summary_observables: list[str] | None = None,
) -> StratificationSummary:
    """Per-stratum row counts, fractions, and observable
    means / stds.

    Parameters
    ----------
    observable_table : ObservableTable
    rules : dict, optional
    summary_observables : list[str], optional
        Columns to compute per-stratum statistics for. Defaults
        to :data:`DEFAULT_SUMMARY_OBSERVABLES`. Columns absent
        from the table are skipped silently.

    Returns
    -------
    StratificationSummary
    """
    rules = rules if rules is not None else DEFAULT_TRUST_RULES
    summary_observables = (
        summary_observables
        if summary_observables is not None
        else DEFAULT_SUMMARY_OBSERVABLES
    )
    sub_tables = stratify_table(observable_table, rules=rules)
    n_total = int(len(observable_table.dataframe))
    n_per: dict[str, int] = {}
    frac_per: dict[str, float] = {}
    means_per: dict[str, dict[str, float]] = {}
    stds_per: dict[str, dict[str, float]] = {}
    for s in STRATA_ORDER:
        df_s = sub_tables[s].dataframe
        n_per[s] = int(len(df_s))
        frac_per[s] = (n_per[s] / n_total) if n_total > 0 else 0.0
        m: dict[str, float] = {}
        sd: dict[str, float] = {}
        for col in summary_observables:
            if col not in df_s.columns:
                continue
            m[col] = _safe_mean(df_s[col])
            sd[col] = _safe_std(df_s[col])
        means_per[s] = m
        stds_per[s] = sd

    metadata = {
        "timestamp_utc": _datetime.datetime.now(
            _datetime.timezone.utc
        ).isoformat(),
        "summary_observables": list(summary_observables),
        "input_table_metadata": dict(observable_table.metadata or {}),
    }
    return StratificationSummary(
        n_total_rows=n_total,
        n_per_stratum=n_per,
        fraction_per_stratum=frac_per,
        observable_means_per_stratum=means_per,
        observable_stds_per_stratum=stds_per,
        rules_used=_copy.deepcopy(rules),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------


def threshold_sensitivity(
    observable_table: ObservableTable,
    observable_name: str,
    rule_column: str,
    threshold_range: np.ndarray | list[float],
    *,
    rules: dict[str, Any] | None = None,
    stratum: str = "high_trust",
) -> np.ndarray:
    """Sweep one trust threshold; report fraction-passing and mean
    observable value within the surviving stratum.

    For each threshold ``t`` in ``threshold_range``, override the
    value associated with ``rule_column`` inside
    ``rules[stratum]["criteria"]`` (which is a ``dict``-style
    "all" criterion set by default), re-run
    :func:`stratify_table`, and record:

    1. the fraction of rows that fall in ``stratum``,
    2. the mean of ``observable_name`` over the surviving rows.

    Parameters
    ----------
    observable_table : ObservableTable
    observable_name : str
        Column whose mean is reported per threshold value.
    rule_column : str
        Column of the criterion to perturb. Must be a key of
        ``rules[stratum]["criteria"]`` whose op is one of
        ``"<"``, ``"<="``, ``">"``, ``">="``.
    threshold_range : array_like
        Sequence of threshold values to try.
    rules : dict, optional
    stratum : str, default ``"high_trust"``
        Which stratum's threshold to sweep. Must use ``"all"``
        logic.

    Returns
    -------
    ndarray, shape ``(n_thresholds, 3)``
        Columns: ``threshold``, ``fraction_in_stratum``,
        ``mean_observable_in_stratum``.

    Raises
    ------
    ValueError
        If ``rule_column`` is not a numeric-criterion key in the
        named stratum's rules.
    """
    rules = rules if rules is not None else DEFAULT_TRUST_RULES
    if stratum not in rules:
        raise ValueError(
            f"threshold_sensitivity: stratum {stratum!r} not in rules"
        )
    base_stratum = rules[stratum]
    if base_stratum.get("logic") != "all":
        raise ValueError(
            f"threshold_sensitivity: stratum {stratum!r} does not use "
            f"'all' logic; only sweeping AND-style criteria is supported."
        )
    criteria = base_stratum.get("criteria", {})
    if rule_column not in criteria:
        raise ValueError(
            f"threshold_sensitivity: rule_column {rule_column!r} not in "
            f"{stratum!r} criteria; available: {list(criteria.keys())}"
        )
    op, _ = criteria[rule_column]
    if op not in ("<", "<=", ">", ">="):
        raise ValueError(
            f"threshold_sensitivity: criterion {rule_column!r} has op "
            f"{op!r}; only numeric inequality operators can be swept."
        )

    threshold_range = np.asarray(threshold_range, dtype=np.float64)
    n_total = int(len(observable_table.dataframe))
    out = np.zeros((threshold_range.size, 3))
    for i, t in enumerate(threshold_range):
        rules_i = _copy.deepcopy(rules)
        rules_i[stratum]["criteria"][rule_column] = (op, float(t))
        strata = stratify_table(observable_table, rules=rules_i)
        sub = strata[stratum].dataframe
        n_in = int(len(sub))
        fraction = (n_in / n_total) if n_total > 0 else 0.0
        if observable_name in sub.columns and n_in > 0:
            mean_obs = _safe_mean(sub[observable_name])
        else:
            mean_obs = float("nan")
        out[i] = [float(t), fraction, mean_obs]
    return out

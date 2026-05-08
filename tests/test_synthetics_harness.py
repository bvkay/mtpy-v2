"""Cross-method validation on the synthetic harness.

Marked ``slow`` because it runs every decomposition method end-to-
end (single-site GB with four disambiguation strategies, BCB,
Lilley with 50 noise realisations, Marti). Deselect with
``pytest -m "not slow"``.

See :mod:`tests.synthetics` for the ground-truth synthetic
generation and the per-method accuracy metric.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.synthetics import (
    compute_method_accuracy,
    generate_synthetic_z,
    run_all_methods_on_synthetic,
)


@pytest.mark.slow
def test_synthetic_harness_clean_2D():
    """Clean 2-D regional with weak distortion: every method that
    returns a ``C`` recovers the ground truth within Frobenius
    distance ``0.05``.

    Methods that don't return a ``C`` (Lilley Mohr-circle, Marti
    WALDIM) are exercised but skipped in the accuracy check —
    their ``compute_method_accuracy`` is ``nan`` by design.
    """
    periods = np.logspace(-1.0, 2.0, 12)
    syn = generate_synthetic_z(
        regional_type="2D",
        distortion_strength="weak",
        distortion_shear="low",
        noise_level="clean",
        periods=periods,
        site_id="S01",
        seed=42,
    )
    results = run_all_methods_on_synthetic(syn)

    # Every method should at least run without raising.
    for method_name, result in results.items():
        assert not (
            isinstance(result, str) and result.startswith("ERROR:")
        ), f"{method_name} failed: {result}"

    # Methods that produce a C must recover the ground truth.
    accuracy = {}
    for method_name, result in results.items():
        acc = compute_method_accuracy(method_name, result, syn)
        if np.isfinite(acc):
            accuracy[method_name] = acc

    assert accuracy, (
        "no method produced a recoverable C; expected GB and Bibby at minimum"
    )

    for method_name, acc in accuracy.items():
        assert acc < 0.05, (
            f"{method_name}: Frobenius |C_rec - C_true| = {acc:.4f} "
            f">= 0.05 tolerance"
        )

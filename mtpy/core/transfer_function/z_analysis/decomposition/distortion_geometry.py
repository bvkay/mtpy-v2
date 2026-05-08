"""Irreducible decomposition of the galvanic distortion tensor.

A real ``2x2`` distortion tensor ``C`` (e.g. the GB / MJ / BCB /
Garcia-Jones output) has four real degrees of freedom. Under the
``SO(2)`` action ``C -> R(theta) C R(theta).T`` these four
components decompose into three irreducible representations:

* **Spin-0 trace** (``trace_a``): scalar, invariant under rotation.
  Represents an isotropic dilation of the distortion tensor — when
  ``C`` differs from the identity by a uniform rescaling of both
  electric-field axes, only ``trace_a`` is non-zero.
* **Spin-2 deviatoric shear** (``gamma_1``, ``gamma_2``): a rank-2
  symmetric traceless tensor with two real components, transforming
  as a complex spin-2 quantity ``gamma = gamma_1 + i gamma_2``,
  i.e. ``gamma' = exp(2 i theta) gamma`` under rotation by
  ``theta``. Represents the *anisotropy* of the distortion: its
  principal axis (``arg(gamma) / 2``) is the strike of the
  near-surface heterogeneity that is causing the distortion, and
  its magnitude (``|gamma|``) is the strength of the anisotropy.
* **Spin-0 antisymmetric pseudo-scalar** (``beta``): scalar,
  invariant under rotation but reverses sign under reflection.
  Captures the rotation-like component of the distortion. The
  *physical* galvanic distortion driven by symmetric static charge
  accumulations is a real *symmetric* tensor with ``beta = 0``; a
  non-zero ``beta`` arises from the *parameterisation* (the
  Groom-Bailey ``T(twist)`` factor is a rotation matrix, which has
  both symmetric and antisymmetric components), or from departures
  from the symmetric-galvanic model (instrument misalignment,
  residual induction). Pure-shear distortion (``twist = 0`` in GB
  language) gives ``beta = 0``.

The convention used in this module is

    D = C - I    (deviation from identity)
    trace_a = (D[0, 0] + D[1, 1]) / 2
    gamma_1 = (D[0, 0] - D[1, 1]) / 2
    gamma_2 = (D[0, 1] + D[1, 0]) / 2
    beta    = (D[0, 1] - D[1, 0]) / 2

so that the four parts recombine as

    D[0, 0] = trace_a + gamma_1
    D[1, 1] = trace_a - gamma_1
    D[0, 1] = gamma_2 + beta
    D[1, 0] = gamma_2 - beta

Connection to E / B-mode analysis
---------------------------------
The spin-2 ``gamma`` field is the same kind of object used in
weak-lensing cosmology to describe galaxy-shape distortions, and it
admits the same Helmholtz-style separation into a curl-free
**E-mode** and a divergence-free **B-mode** when measured across a
spatial array of stations. Galvanic physics predicts E-mode
dominance for the distortion of a near-surface heterogeneity field;
B-mode power across an array is therefore a diagnostic of inductive
contamination, sensor anisotropy, or systematic decomposition
error. The ``gamma`` field returned by this module is the
station-by-station input to that array-level analysis.

References
----------
Groom, R. W., & Bailey, R. C. (1989). Decomposition of magnetotelluric
impedance tensors in the presence of local three-dimensional
galvanic distortion. Journal of Geophysical Research: Solid Earth,
94(B2), 1913-1925.

Smith, J. T. (1995). Understanding telluric distortion matrices.
Geophysical Journal International, 122, 219-226.

Schneider, P., van Waerbeke, L., Mellier, Y., Jain, B., Seitz, S.,
& Fort, B. (2002). Detection of shear due to weak lensing by
large-scale structure. Astronomy & Astrophysics, 396, 1-19.
(For the spin-2 / E-mode-B-mode formalism on the sphere; the
arithmetic for a 2-D tensor on a flat array is simpler but the
spin-2 transformation law is identical.)
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "complex_to_gamma",
    "gamma_field",
    "gamma_magnitude",
    "gamma_to_complex",
    "irreducible_decomposition",
    "principal_axis",
]


def _check_2x2(arr: np.ndarray, name: str) -> None:
    if arr.ndim < 2 or arr.shape[-2:] != (2, 2):
        raise ValueError(
            f"{name}: expected an array with trailing shape (2, 2); "
            f"got shape {arr.shape}"
        )


def irreducible_decomposition(D: np.ndarray) -> dict:
    """Irreducible decomposition of ``D`` under ``SO(2)``.

    Splits the four real components of a 2x2 real matrix into the
    three irreducible representations of the rotation group: the
    spin-0 trace (``trace_a``), the spin-2 deviatoric shear
    (``gamma_1``, ``gamma_2``), and the spin-0 antisymmetric
    pseudo-scalar (``beta``). See the module docstring for the
    physical interpretation.

    Parameters
    ----------
    D : ndarray
        Real array with trailing shape ``(2, 2)``: a single matrix
        of shape ``(2, 2)`` or a stack of shape ``(..., 2, 2)``.

    Returns
    -------
    dict
        Keys ``trace_a``, ``gamma_1``, ``gamma_2``, ``beta``. Values
        are Python ``float`` scalars when ``D`` has shape
        ``(2, 2)``, or ``ndarray`` of shape ``(...)`` (the leading
        dimensions of ``D``) for stacked inputs.

    Raises
    ------
    ValueError
        If ``D`` does not have trailing shape ``(2, 2)``.
    """
    arr = np.asarray(D, dtype=np.float64)
    _check_2x2(arr, "irreducible_decomposition")

    trace_a = 0.5 * (arr[..., 0, 0] + arr[..., 1, 1])
    gamma_1 = 0.5 * (arr[..., 0, 0] - arr[..., 1, 1])
    gamma_2 = 0.5 * (arr[..., 0, 1] + arr[..., 1, 0])
    beta = 0.5 * (arr[..., 0, 1] - arr[..., 1, 0])

    if arr.ndim == 2:
        return {
            "trace_a": float(trace_a),
            "gamma_1": float(gamma_1),
            "gamma_2": float(gamma_2),
            "beta": float(beta),
        }
    return {
        "trace_a": trace_a,
        "gamma_1": gamma_1,
        "gamma_2": gamma_2,
        "beta": beta,
    }


def gamma_field(C: np.ndarray) -> tuple:
    """Spin-2 deviatoric shear of the distortion ``D = C - I``.

    Parameters
    ----------
    C : ndarray
        Real distortion tensor with trailing shape ``(2, 2)``.

    Returns
    -------
    (gamma_1, gamma_2)
        Tuple of the two real components of the spin-2 shear. For a
        single ``(2, 2)`` input both are Python ``float``; for a
        stacked input ``(..., 2, 2)`` both are ``ndarray`` of shape
        ``(...)``.

        For a galvanic distortion of order 10-20 % the shear
        magnitude ``sqrt(gamma_1**2 + gamma_2**2)`` is in the range
        ``[0, 0.5]``; values approaching or exceeding ``0.5`` are a
        diagnostic of strong anisotropy (or of decomposition error).
    """
    arr = np.asarray(C, dtype=np.float64)
    _check_2x2(arr, "gamma_field")
    eye = np.eye(2)
    parts = irreducible_decomposition(arr - eye)
    return parts["gamma_1"], parts["gamma_2"]


def gamma_magnitude(C: np.ndarray) -> np.ndarray | float:
    """Magnitude of the spin-2 shear: ``sqrt(gamma_1**2 + gamma_2**2)``.

    Parameters
    ----------
    C : ndarray
        Real distortion tensor with trailing shape ``(2, 2)``.

    Returns
    -------
    float or ndarray
        ``sqrt(gamma_1**2 + gamma_2**2)``. Scalar for a single
        ``(2, 2)`` input; ``ndarray`` of shape ``(...)`` for a
        stacked input.
    """
    g1, g2 = gamma_field(C)
    g1_arr = np.asarray(g1)
    g2_arr = np.asarray(g2)
    mag = np.sqrt(g1_arr**2 + g2_arr**2)
    if mag.ndim == 0:
        return float(mag)
    return mag


def principal_axis(C: np.ndarray) -> tuple:
    """Principal axis and magnitude of the spin-2 shear.

    The principal axis ``psi`` is the angle of the long axis of the
    distortion ellipse — the strike along which the distortion
    pulls the regional electric field. Because the spin-2 shear
    transforms as ``exp(2 i theta)``, ``psi`` is determined modulo
    ``pi`` (the axis points both ways); we return it in the
    canonical range ``[0, pi)``.

    Parameters
    ----------
    C : ndarray
        Real distortion tensor with trailing shape ``(2, 2)``.

    Returns
    -------
    (azimuth_rad, magnitude)
        ``azimuth_rad`` is the principal-axis angle in radians, in
        ``[0, pi)``. ``magnitude`` matches :func:`gamma_magnitude`.
        Both are scalars for ``(2, 2)`` input; both are ``ndarray``
        for stacked input.
    """
    g1, g2 = gamma_field(C)
    g1_arr = np.asarray(g1)
    g2_arr = np.asarray(g2)
    psi = (0.5 * np.arctan2(g2_arr, g1_arr)) % np.pi
    mag = np.sqrt(g1_arr**2 + g2_arr**2)
    if psi.ndim == 0:
        return float(psi), float(mag)
    return psi, mag


def gamma_to_complex(gamma_1, gamma_2):
    """Pack ``(gamma_1, gamma_2)`` into a complex spin-2 number.

    Parameters
    ----------
    gamma_1, gamma_2 : float or ndarray
        The two real components of the spin-2 shear.

    Returns
    -------
    complex or ndarray
        ``gamma_1 + 1j * gamma_2``. Under rotation by ``theta``,
        this transforms as ``exp(2 i theta) * gamma``.
    """
    return np.asarray(gamma_1) + 1j * np.asarray(gamma_2)


def complex_to_gamma(g) -> tuple:
    """Unpack a complex spin-2 number into ``(gamma_1, gamma_2)``.

    Parameters
    ----------
    g : complex or ndarray
        Complex spin-2 shear ``gamma_1 + 1j * gamma_2``.

    Returns
    -------
    (gamma_1, gamma_2)
        The two real components.
    """
    g_arr = np.asarray(g)
    return g_arr.real, g_arr.imag

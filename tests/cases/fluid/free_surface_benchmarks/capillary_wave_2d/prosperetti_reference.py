#!/usr/bin/env python3
"""Reference solution of the capillary_wave_2d benchmark (tracker M3).

Prosperetti's initial-value solution for a small-amplitude standing
capillary wave on the free surface of a viscous liquid of infinite depth,
released from rest with the surface elevation a0*cos(kx):

    A. Prosperetti, "Viscous effects on small-amplitude surface waves",
    Phys. Fluids 19, 195 (1976); "Motion of two superposed viscous fluids",
    Phys. Fluids 24, 1217 (1981), one-fluid limit (upper density zero).

With sigma = nu*k^2 and omega0^2 = gamma*k^3/rho,

    a(t)/a0 = 4 sigma^2 / (8 sigma^2 + omega0^2) * erfc(sqrt(sigma t))
            + sum_i  z_i / Z_i * omega0^2 / (z_i^2 - sigma)
                     * exp((z_i^2 - sigma) t) * erfc(z_i sqrt(t)),

where z_i (i = 1..4) are the roots of

    z^4 + 2 sigma z^2 + 4 sigma^(3/2) z + sigma^2 + omega0^2 = 0

and Z_i = prod_{j != i} (z_j - z_i).  This is equation (12)-(13) of
Denner et al., Phys. Rev. E 94, 023110 (2016) with their density parameter
beta = rho_1 rho_2 / (rho_1 + rho_2)^2 set to zero.  Each product
exp(z^2 t) erfc(z sqrt(t)) is evaluated as the scaled complementary error
function erfcx(z sqrt(t)), so no term overflows.

Independent relations used to check the solution (see the benchmark tests):

- its Laplace transform, derived directly from the linearized
  Navier-Stokes equations with a stress-free surface and u(0) = 0,

      a_hat(s) = a0 D(s) / (s (D(s) + omega0^2)),
      D(s) = (s + 2 sigma)^2 - 4 sigma^2 sqrt(1 + s / sigma);

- Lamb's normal-mode dispersion relation D(s) + omega0^2 = 0, whose root
  s = -beta + i omega tends to omega -> omega0 and beta -> 2 nu k^2 as the
  viscosity parameter epsilon = sigma / omega0 tends to zero;
- the inviscid limit a(t) = a0 cos(omega0 t).

Only numpy is required.
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Scaled complementary error function for complex arguments
# ---------------------------------------------------------------------------
# Weideman, "Computation of the complex error function", SIAM J. Numer.
# Anal. 31, 1497 (1994): a rational expansion of the Faddeeva function
# w(z) = exp(-z^2) erfc(-iz) in the upper half plane.  N = 40 terms give an
# accuracy close to double-precision round-off (checked against the real
# math.erfc and the Taylor series in the tests).
_WEIDEMAN_TERMS = 40


def _weideman_coefficients(n: int) -> tuple[float, np.ndarray]:
    m = 2 * n
    k = np.arange(-m + 1, m)
    length = math.sqrt(n / math.sqrt(2.0))
    t = length * np.tan(k * math.pi / (2.0 * m))
    f = np.concatenate(([0.0], np.exp(-t ** 2) * (length ** 2 + t ** 2)))
    a = np.real(np.fft.fft(np.fft.fftshift(f))) / (2 * m)
    return length, a[1:n + 1][::-1]


_WEIDEMAN_L, _WEIDEMAN_A = _weideman_coefficients(_WEIDEMAN_TERMS)


def _faddeeva_upper(z: np.ndarray) -> np.ndarray:
    """w(z) for Im z >= 0."""
    lz = _WEIDEMAN_L - 1j * z
    p = np.polyval(_WEIDEMAN_A, (_WEIDEMAN_L + 1j * z) / lz)
    return 2.0 * p / lz ** 2 + (1.0 / math.sqrt(math.pi)) / lz


def faddeeva(z) -> np.ndarray:
    """Faddeeva function w(z) = exp(-z^2) erfc(-iz) for any complex z."""
    z = np.asarray(z, dtype=complex)
    upper = z.imag >= 0.0
    # w(z) = 2 exp(-z^2) - w(-z) maps the lower half plane to the upper one.
    zu = np.where(upper, z, -z)
    w = _faddeeva_upper(zu)
    return np.where(upper, w, 2.0 * np.exp(-z * z) - w)


def erfcx(z) -> np.ndarray:
    """Scaled complementary error function exp(z^2) erfc(z), complex z."""
    return faddeeva(1j * np.asarray(z, dtype=complex))


# ---------------------------------------------------------------------------
# Linear theory
# ---------------------------------------------------------------------------
def inviscid_frequency(wavenumber: float, surface_tension: float, density: float,
                       depth: float = math.inf) -> float:
    """omega0 with omega0^2 = gamma k^3 tanh(k H) / rho (tanh -> 1 for deep liquid)."""
    factor = 1.0 if math.isinf(depth) else math.tanh(wavenumber * depth)
    return math.sqrt(surface_tension * wavenumber ** 3 * factor / density)


def weak_damping_rate(wavenumber: float, kinematic_viscosity: float) -> float:
    """Lamb's small-viscosity amplitude decay rate 2 nu k^2."""
    return 2.0 * kinematic_viscosity * wavenumber ** 2


def _sigma_omega0(wavenumber, kinematic_viscosity, surface_tension, density):
    if not (wavenumber > 0.0 and surface_tension > 0.0 and density > 0.0
            and kinematic_viscosity >= 0.0):
        raise ValueError("need k, gamma, rho > 0 and nu >= 0")
    sigma = kinematic_viscosity * wavenumber ** 2
    return sigma, inviscid_frequency(wavenumber, surface_tension, density)


def prosperetti_roots(sigma: float, omega0: float) -> np.ndarray:
    """The four roots z_i of the free-surface quartic."""
    return np.roots([1.0, 0.0, 2.0 * sigma, 4.0 * sigma ** 1.5, sigma ** 2 + omega0 ** 2])


def prosperetti_amplitude(times, *, wavenumber: float, kinematic_viscosity: float,
                          surface_tension: float, density: float,
                          initial_amplitude: float) -> np.ndarray:
    """a(t) of the free-surface initial-value problem at the given times (t >= 0)."""
    t = np.atleast_1d(np.asarray(times, dtype=float))
    if np.any(~np.isfinite(t)) or np.any(t < 0.0):
        raise ValueError("times must be finite and non-negative")
    sigma, omega0 = _sigma_omega0(wavenumber, kinematic_viscosity, surface_tension, density)
    z = prosperetti_roots(sigma, omega0)
    gaps = np.abs(z[:, None] - z[None, :])[~np.eye(4, dtype=bool)]
    if np.min(gaps) < 1e-12 * math.sqrt(omega0):
        raise ValueError("repeated roots: the partial-fraction form does not apply")
    big_z = np.array([np.prod([z[j] - z[i] for j in range(4) if j != i]) for i in range(4)])
    coefficient = z / big_z * omega0 ** 2 / (z ** 2 - sigma)
    root_t = np.sqrt(t)
    value = np.zeros(t.shape, dtype=complex)
    for c, zi in zip(coefficient, z):
        value += c * erfcx(zi * root_t)
    value *= np.exp(-sigma * t)
    first = 4.0 * sigma ** 2 / (8.0 * sigma ** 2 + omega0 ** 2)
    value += first * np.array([math.erfc(math.sqrt(sigma * ti)) for ti in t])
    scale = max(1.0, float(np.max(np.abs(value.real))))
    if np.max(np.abs(value.imag)) > 1e-9 * scale:
        raise ArithmeticError("Prosperetti sum has a non-negligible imaginary part")
    return initial_amplitude * value.real


def laplace_transform(s, *, wavenumber: float, kinematic_viscosity: float,
                      surface_tension: float, density: float,
                      initial_amplitude: float) -> np.ndarray:
    """a_hat(s) from the linearized Navier-Stokes equations, for real s > 0.

    Derivation (liquid y < 0, surface a(t) cos(kx), u(0) = 0): potential
    part A exp(ky) cos(kx), stream function B exp(my) sin(kx) with
    m^2 = k^2 + s/nu.  Zero tangential stress gives B = 2 k^2 A / (m^2 + k^2);
    the kinematic condition s a_hat - a0 = k (A - B) and the normal stress
    rho s A + 2 mu (k^2 A - k m B) = -gamma k^2 a_hat then give
    a_hat = a0 D / (s (D + omega0^2)) with D = (s + 2 nu k^2)^2 - 4 nu^2 k^3 m.
    """
    s = np.asarray(s, dtype=float)
    sigma, omega0 = _sigma_omega0(wavenumber, kinematic_viscosity, surface_tension, density)
    m_term = 4.0 * sigma ** 2 * np.sqrt(1.0 + s / sigma) if sigma > 0.0 else 0.0
    d = (s + 2.0 * sigma) ** 2 - m_term
    return initial_amplitude * d / (s * (d + omega0 ** 2))


def normal_mode(*, wavenumber: float, kinematic_viscosity: float,
                surface_tension: float, density: float) -> dict:
    """Least-damped root s = -beta + i omega of Lamb's dispersion relation.

    Solves (s + 2 sigma)^2 + omega0^2 = 4 sigma^2 sqrt(1 + s/sigma) on the
    principal branch by Newton's method from s = -2 sigma + i omega0.
    """
    sigma, omega0 = _sigma_omega0(wavenumber, kinematic_viscosity, surface_tension, density)
    s = complex(-2.0 * sigma, omega0)
    if sigma > 0.0:
        for _ in range(100):
            root = np.sqrt(1.0 + s / sigma)
            f = (s + 2.0 * sigma) ** 2 + omega0 ** 2 - 4.0 * sigma ** 2 * root
            df = 2.0 * (s + 2.0 * sigma) - 2.0 * sigma / root
            step = f / df
            s -= step
            if abs(step) <= 1e-15 * omega0:
                break
        else:
            raise ArithmeticError("normal-mode Newton iteration did not converge")
        if not (np.sqrt(1.0 + s / sigma).real > 0.0 and s.imag > 0.0):
            raise ArithmeticError("normal-mode root left the principal branch")
    return {"omega": float(s.imag), "beta": float(-s.real), "omega0": omega0,
            "weak_damping_rate": 2.0 * sigma, "epsilon": sigma / omega0}

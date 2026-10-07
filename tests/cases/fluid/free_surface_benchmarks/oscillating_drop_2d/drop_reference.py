#!/usr/bin/env python3
"""Reference solution of the oscillating_drop_2d benchmark (tracker M3).

Small-amplitude shape oscillation of a two-dimensional (circular-cylinder)
viscous liquid drop of radius R, density rho, viscosity mu = rho nu and
surface tension gamma, with zero gravity and exterior pressure zero.  The
drop is released from rest with the surface r = R + a0 cos(n theta).  The
linearized Navier-Stokes problem is solved exactly (README.md, "Reference
solution", gives the derivation):

  velocity u = grad(phi) + curl(psi e_z) with phi = A r^n cos(n theta),
  psi = B I_n(q r) sin(n theta), q^2 = s / nu (Laplace variable s), and the
  pressure perturbation -rho s phi; zero tangential stress, the normal-stress
  balance -p + 2 mu du_r/dr = -gamma (n^2 - 1) a / R^2 and the kinematic
  condition s a_hat - a0 = u_r(R) at r = R give

      a_hat(s) = a0 (s + c Q) / (s (s + c Q) + omega0^2 P),
      omega0^2 = n (n^2 - 1) gamma / (rho R^3),   c = 2 n (n - 1) nu / R^2,
      P = 1 + 2 n (n - 1) / W,   Q = 1 + 2 n (r - 1) / W,
      W = 2 r - x^2 - 2 n^2,     r = x I_n'(x) / I_n(x),   x = R sqrt(s / nu).

P and Q depend on x^2 only, so a_hat(s) is meromorphic: a bounded drop has a
discrete spectrum, the normal-mode dispersion relation is

      s^2 + c Q s + omega0^2 P = 0,

and the initial-value solution is the residue sum over its roots: the
least-damped complex pair s = -beta +/- i omega and a sequence of real,
non-oscillatory viscous modes.  This follows the method of Prosperetti for
the spherical drop (J. Fluid Mech. 100, 333 (1980)); the dispersion relation
is the two-dimensional counterpart of Reid (Q. Appl. Math. 18, 86 (1960)) and
Chandrasekhar (Proc. London Math. Soc. 9, 141 (1959)).

Limits and independent checks (all in the benchmark tests):

- nu = 0: a(t) = a0 cos(omega0 t), omega0^2 = n (n^2 - 1) gamma / (rho R^3)
  (Rayleigh, Proc. R. Soc. Lond. 29, 71 (1879), the jet of zero wavenumber);
- weak viscosity, eps = nu / (omega0 R^2) -> 0: beta -> 2 n (n - 1) nu / R^2
  (Lamb's dissipation method; Aalilija, Gandin and Hachem, Comput. Fluids 197,
  104362 (2020)) with the boundary-layer corrections of
  weak_viscosity_mode();
- Stokes limit nu -> infinity: the slowest real mode tends to
  s = -n gamma / (2 mu R) (biharmonic stream function, README.md);
- large n at fixed n / R = k: the planar relation of Lamb,
  (s + 2 nu k^2)^2 + omega0^2 = 4 nu^2 k^3 sqrt(k^2 + s / nu);
- a_hat(s) from an independent Chebyshev collocation of the linearized
  equations in stream-function form (laplace_transform_collocation, no
  Bessel functions), and the Laplace transform of the residue sum.

Only numpy is required.
"""

from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# Closed-form limits
# ---------------------------------------------------------------------------
def inviscid_frequency(mode: int, surface_tension: float, density: float,
                       radius: float) -> float:
    """omega0 with omega0^2 = n (n^2 - 1) gamma / (rho R^3) (Rayleigh 1879)."""
    _check_mode(mode)
    return math.sqrt(mode * (mode ** 2 - 1) * surface_tension / (density * radius ** 3))


def weak_damping_rate(mode: int, kinematic_viscosity: float, radius: float) -> float:
    """Amplitude decay rate 2 n (n - 1) nu / R^2 of the weak-viscosity limit."""
    _check_mode(mode)
    return 2.0 * mode * (mode - 1) * kinematic_viscosity / radius ** 2


def weak_viscosity_mode(*, mode: int, kinematic_viscosity: float, surface_tension: float,
                        density: float, radius: float) -> dict:
    """Two-term expansion of the normal mode for eps = nu / (omega0 R^2) -> 0.

    beta  = 2 n (n - 1) nu / R^2 * (1 - (n - 1) sqrt(eps / 2) + O(eps)),
    omega = omega0 * (1 - sqrt(2) n (n - 1)^2 eps^(3/2) + O(eps^2)),

    from the large-x expansion x I_n'(x) / I_n(x) = x - 1/2 + O(1/x) of the
    dispersion relation (README.md).  For n -> infinity at fixed n / R = k
    both corrections become Lamb's planar ones, -sqrt(eps_k / 2) and
    -sqrt(2) eps_k^(3/2) with eps_k = nu k^2 / omega0 = n^2 eps.
    """
    n = mode
    omega0 = inviscid_frequency(n, surface_tension, density, radius)
    eps = kinematic_viscosity / (omega0 * radius ** 2)
    beta0 = weak_damping_rate(n, kinematic_viscosity, radius)
    return {"omega": omega0 * (1.0 - math.sqrt(2.0) * n * (n - 1) ** 2 * eps ** 1.5),
            "beta": beta0 * (1.0 - (n - 1) * math.sqrt(eps / 2.0)),
            "omega0": omega0, "weak_damping_rate": beta0, "epsilon": eps}


def stokes_rate(mode: int, viscosity: float, surface_tension: float, radius: float) -> float:
    """Relaxation rate n gamma / (2 mu R) of mode n in Stokes flow (inertia-free limit)."""
    _check_mode(mode)
    return mode * surface_tension / (2.0 * viscosity * radius)


def _check_mode(mode: int) -> None:
    if int(mode) != mode or mode < 2:
        raise ValueError("the shape mode n must be an integer >= 2")


# ---------------------------------------------------------------------------
# Modified Bessel functions
# ---------------------------------------------------------------------------
def bessel_ratio(mode: int, x) -> np.ndarray:
    """r(x) = x I_n'(x) / I_n(x) for complex x, by the continued fraction of I_(n+1) / I_n.

    I_(k)/I_(k-1) = 1 / (2k/x + I_(k+1)/I_k) is the ratio of the minimal
    solution of the recurrence, so the continued fraction converges for every
    x != 0 (after about |x| terms; modified Lentz algorithm).  r = n + x
    I_(n+1)(x) / I_n(x), and r(0) = n.  r depends on x^2 only.
    """
    x = np.atleast_1d(np.asarray(x, dtype=complex))
    out = np.full(x.shape, complex(mode))
    tiny = 1e-300
    for idx in np.ndindex(x.shape):
        z = x[idx]
        if z == 0:
            continue
        f = tiny
        c, d = f, 0.0
        k = 1
        while True:
            b = 2.0 * (mode + k) / z
            d = b + d
            d = 1.0 / (d if d != 0 else tiny)
            c = b + 1.0 / c
            if c == 0:
                c = tiny
            delta = c * d
            f *= delta
            if abs(delta - 1.0) < 1e-15:
                break
            k += 1
            if k > 100000 + 4 * int(abs(z)):
                raise ArithmeticError("Bessel continued fraction did not converge")
        out[idx] = mode + z * f
    return out


def _bessel_j_nodes(xi_max: float, mode: int) -> int:
    return int(math.ceil(xi_max + mode + 10.0 * max(xi_max, 1.0) ** (1.0 / 3.0) + 40.0))


def bessel_j(order: int, xi) -> np.ndarray:
    """J_order(xi) for real xi from the periodic trapezoidal rule on Bessel's integral.

    J_m(xi) = (1/pi) int_0^pi cos(m tau - xi sin tau) dtau; the integrand is
    smooth and periodic, so the rule converges geometrically once the node
    count exceeds |xi| + |m| (the aliased terms are J_(2M -+ m)(xi)).
    """
    xi = np.atleast_1d(np.asarray(xi, dtype=float))
    m = _bessel_j_nodes(float(np.max(np.abs(xi))) if xi.size else 1.0, abs(order))
    tau = np.linspace(0.0, math.pi, m + 1)
    w = np.full(m + 1, math.pi / m)
    w[[0, -1]] *= 0.5
    out = np.empty(xi.shape)
    flat, res = xi.ravel(), out.ravel()
    for start in range(0, flat.size, 512):
        chunk = flat[start:start + 512]
        res[start:start + 512] = np.cos(order * tau[None, :]
                                        - chunk[:, None] * np.sin(tau)[None, :]) @ w / math.pi
    # Near xi = 0, J_m ~ xi^m is far below the rule's absolute round-off:
    # use the power series there (full relative accuracy for |xi| <= 2).
    small = np.abs(flat) <= 2.0
    if np.any(small):
        m_abs = abs(order)
        z = 0.5 * flat[small]
        term = z ** m_abs / math.factorial(m_abs)
        total = term.copy()
        for k in range(1, 30):
            term = -term * z * z / (k * (k + m_abs))
            total += term
        res[small] = total * (-1.0) ** m_abs if order < 0 else total
    return out


# ---------------------------------------------------------------------------
# Linear viscous theory
# ---------------------------------------------------------------------------
class _Drop:
    """Parameters and the affine-in-r pieces of the transform."""

    def __init__(self, mode, kinematic_viscosity, surface_tension, density, radius):
        _check_mode(mode)
        if not (surface_tension > 0.0 and density > 0.0 and radius > 0.0
                and kinematic_viscosity > 0.0):
            raise ValueError("need gamma, rho, R > 0 and nu > 0 (nu = 0: use the inviscid limit)")
        self.n = int(mode)
        self.nu = float(kinematic_viscosity)
        self.radius = float(radius)
        self.omega0 = inviscid_frequency(mode, surface_tension, density, radius)
        self.c = 2.0 * mode * (mode - 1) * kinematic_viscosity / radius ** 2
        self.kappa = radius ** 2 / kinematic_viscosity           # d(x^2)/ds

    def pieces(self, s, x2, r):
        """E = W * (s (s + cQ) + omega0^2 P) and N = W * (s + cQ), affine in r."""
        n, c, w2 = self.n, self.c, self.omega0 ** 2
        n1 = s * s + c * (n + 1) * s + w2
        n0 = s * s * (x2 + 2 * n * n) + c * s * (x2 + 2 * n * n + 2 * n) + w2 * (x2 + 2 * n)
        e = 2.0 * r * n1 - n0
        num = 2.0 * r * (s + c * (n + 1)) - (s * (x2 + 2 * n * n) + c * (x2 + 2 * n * n + 2 * n))
        return e, num, n1

    def e_and_derivative(self, s: complex):
        x2 = self.kappa * s
        r = complex(bessel_ratio(self.n, np.sqrt(x2))[0])
        e, num, n1 = self.pieces(s, x2, r)
        n, c, w2 = self.n, self.c, self.omega0 ** 2
        dr = (x2 + n * n - r * r) / (2.0 * s)                   # dr/ds
        dn0 = (2.0 * s * (x2 + 2 * n * n) + s * x2 + c * (x2 + 2 * n * n + 2 * n)
               + c * x2 + w2 * x2 / s)
        de = 2.0 * dr * n1 + 2.0 * r * (2.0 * s + c * (n + 1)) - dn0
        return e, de, num


def _drop(mode, kinematic_viscosity, surface_tension, density, radius) -> _Drop:
    return _Drop(mode, kinematic_viscosity, surface_tension, density, radius)


def laplace_transform(s, *, mode: int, kinematic_viscosity: float, surface_tension: float,
                      density: float, radius: float, initial_amplitude: float) -> np.ndarray:
    """a_hat(s) of the released drop (closed form above), for complex s off the spectrum."""
    d = _drop(mode, kinematic_viscosity, surface_tension, density, radius)
    s = np.atleast_1d(np.asarray(s, dtype=complex))
    x2 = d.kappa * s
    r = bessel_ratio(d.n, np.sqrt(x2))
    e, num, _ = d.pieces(s, x2, r)
    return initial_amplitude * num / e


def dispersion_function(s, *, mode: int, kinematic_viscosity: float, surface_tension: float,
                        density: float, radius: float) -> np.ndarray:
    """s^2 + c Q s + omega0^2 P (zero at a normal mode)."""
    d = _drop(mode, kinematic_viscosity, surface_tension, density, radius)
    s = np.atleast_1d(np.asarray(s, dtype=complex))
    x2 = d.kappa * s
    r = bessel_ratio(d.n, np.sqrt(x2))
    w = 2.0 * r - x2 - 2 * d.n ** 2
    p = 1.0 + 2.0 * d.n * (d.n - 1) / w
    q = 1.0 + 2.0 * d.n * (r - 1.0) / w
    return s * s + d.c * q * s + d.omega0 ** 2 * p


def _newton(d: _Drop, s: complex, tol: float = 1e-14, max_iter: int = 60) -> complex:
    for _ in range(max_iter):
        e, de, _ = d.e_and_derivative(s)
        step = e / de
        s -= step
        if abs(step) <= tol * d.omega0:
            return s
    raise ArithmeticError("normal-mode Newton iteration did not converge")


def normal_mode(*, mode: int, kinematic_viscosity: float, surface_tension: float,
                density: float, radius: float) -> dict:
    """Least-damped oscillatory root s = -beta + i omega of the dispersion relation.

    Newton's method on W * (dispersion function), continued in nu from
    eps = nu / (omega0 R^2) = 1e-6, where the weak-viscosity expansion is an
    accurate start, to the requested viscosity in steps of a factor of at
    most 2.
    """
    omega0 = inviscid_frequency(mode, surface_tension, density, radius)
    nu_target = float(kinematic_viscosity)
    if nu_target <= 0.0:
        raise ValueError("nu must be positive (nu = 0 is the inviscid limit)")
    nu = min(nu_target, 1e-6 * omega0 * radius ** 2)
    weak = weak_viscosity_mode(mode=mode, kinematic_viscosity=nu, surface_tension=surface_tension,
                               density=density, radius=radius)
    s = complex(-weak["beta"], weak["omega"])
    while True:
        s = _newton(_drop(mode, nu, surface_tension, density, radius), s)
        if not s.imag > 0.0:
            raise ArithmeticError("the oscillatory mode became overdamped (no complex root)")
        if nu >= nu_target:
            break
        nu_next = min(nu_target, 2.0 * nu)
        s = complex(s.real * nu_next / nu, s.imag)             # damping scales with nu
        nu = nu_next
    weak = weak_viscosity_mode(mode=mode, kinematic_viscosity=nu_target,
                               surface_tension=surface_tension, density=density, radius=radius)
    return {"omega": float(s.imag), "beta": float(-s.real), "omega0": omega0,
            "weak_damping_rate": weak["weak_damping_rate"], "epsilon": weak["epsilon"],
            "s": s}


def real_modes(*, mode: int, kinematic_viscosity: float, surface_tension: float,
               density: float, radius: float, xi_max: float) -> np.ndarray:
    """Real roots s = -nu xi^2 / R^2 < 0 of the dispersion relation with 0 < xi <= xi_max.

    On the negative real axis x = i xi and r = xi J_n'(xi) / J_n(xi), so
    J_n W (dispersion) = 2 xi J_n' N1 - J_n N0 is a real entire function of
    xi.  Its sign changes on a grid of spacing 0.05 (the roots are about pi
    apart; geometric below 0.05 for the slow Stokes mode) are refined by
    bisection.  xi = 0 (s = 0) is a removable root of the
    transform and is excluded.
    """
    d = _drop(mode, kinematic_viscosity, surface_tension, density, radius)
    n = d.n

    def f(xi):
        s = -d.nu * xi ** 2 / d.radius ** 2
        x2 = -xi ** 2
        jn = bessel_j(n, xi)
        djn = 0.5 * (bessel_j(n - 1, xi) - bessel_j(n + 1, xi))
        n1 = s * s + d.c * (n + 1) * s + d.omega0 ** 2
        n0 = (s * s * (x2 + 2 * n * n) + d.c * s * (x2 + 2 * n * n + 2 * n)
              + d.omega0 ** 2 * (x2 + 2 * n))
        return 2.0 * xi * djn * n1 - jn * n0

    grid = np.concatenate([np.geomspace(1e-4, 0.05, 120, endpoint=False),
                           np.arange(0.05, xi_max + 0.05, 0.05)])
    values = f(grid)
    idx = np.nonzero(np.sign(values[:-1]) * np.sign(values[1:]) < 0)[0]
    lo, hi = grid[idx], grid[idx + 1]
    flo = values[idx]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        left = np.sign(fm) == np.sign(flo)
        lo, flo = np.where(left, mid, lo), np.where(left, fm, flo)
        hi = np.where(left, hi, mid)
    xi = 0.5 * (lo + hi)
    return -d.nu * xi ** 2 / d.radius ** 2


def _residues(d: _Drop, roots) -> np.ndarray:
    out = []
    for s in roots:
        _, de, num = d.e_and_derivative(complex(s))
        out.append(num / de)
    return np.asarray(out)


def modal_expansion(*, mode: int, kinematic_viscosity: float, surface_tension: float,
                    density: float, radius: float, initial_amplitude: float,
                    min_time: float, tolerance: float = 1e-17) -> dict:
    """Roots and residues of a_hat for t >= min_time > 0.

    a(t) = 2 Re(A_c exp(s_c t)) + sum_k A_k exp(s_k t), with s_c the complex
    normal mode and s_k the real viscous modes; the real modes are kept while
    exp(s_k min_time) exceeds tolerance.
    """
    if not min_time > 0.0:
        raise ValueError("min_time must be positive")
    d = _drop(mode, kinematic_viscosity, surface_tension, density, radius)
    complex_mode = normal_mode(mode=mode, kinematic_viscosity=kinematic_viscosity,
                               surface_tension=surface_tension, density=density, radius=radius)
    xi_max = radius * math.sqrt(-math.log(tolerance) / (kinematic_viscosity * min_time)) + 5.0
    if xi_max > 20000.0:
        raise ValueError("times too close to 0 for the modal sum; use larger times")
    real = real_modes(mode=mode, kinematic_viscosity=kinematic_viscosity,
                      surface_tension=surface_tension, density=density, radius=radius,
                      xi_max=xi_max)
    s_c = complex_mode["s"]
    res_c = complex(_residues(d, [s_c])[0]) * initial_amplitude
    res_r = _residues(d, real) * initial_amplitude
    if np.max(np.abs(res_r.imag), initial=0.0) > 1e-9 * abs(initial_amplitude):
        raise ArithmeticError("real modes with complex residues")
    return {"complex_root": s_c, "complex_residue": res_c,
            "real_roots": np.asarray(real), "real_residues": res_r.real,
            "normal_mode": complex_mode}


def drop_amplitude(times, *, mode: int, kinematic_viscosity: float, surface_tension: float,
                   density: float, radius: float, initial_amplitude: float) -> np.ndarray:
    """a(t) of the linear initial-value problem at the given times (t >= 0).

    a(0) = a0 by the initial condition (the modal sum converges only
    conditionally there); nu = 0 gives a0 cos(omega0 t).
    """
    t = np.atleast_1d(np.asarray(times, dtype=float))
    if np.any(~np.isfinite(t)) or np.any(t < 0.0):
        raise ValueError("times must be finite and non-negative")
    if kinematic_viscosity == 0.0:
        omega0 = inviscid_frequency(mode, surface_tension, density, radius)
        return initial_amplitude * np.cos(omega0 * t)
    out = np.full(t.shape, float(initial_amplitude))
    positive = t > 0.0
    if not np.any(positive):
        return out
    ex = modal_expansion(mode=mode, kinematic_viscosity=kinematic_viscosity,
                         surface_tension=surface_tension, density=density, radius=radius,
                         initial_amplitude=initial_amplitude, min_time=float(np.min(t[positive])))
    tp = t[positive]
    value = 2.0 * (ex["complex_residue"] * np.exp(ex["complex_root"] * tp)).real
    value += (ex["real_residues"][None, :] * np.exp(np.outer(tp, ex["real_roots"]))).sum(axis=1)
    out[positive] = value
    return out


# ---------------------------------------------------------------------------
# Independent check: Chebyshev collocation of the transformed equations
# ---------------------------------------------------------------------------
def _cheb(n: int):
    """Chebyshev differentiation matrix on the n + 1 Gauss-Lobatto points (Trefethen 2000)."""
    x = np.cos(math.pi * np.arange(n + 1) / n)
    c = np.hstack([2.0, np.ones(n - 1), 2.0]) * (-1.0) ** np.arange(n + 1)
    dx = x[:, None] - x[None, :]
    d = np.outer(c, 1.0 / c) / (dx + np.eye(n + 1))
    d -= np.diag(d.sum(axis=1))
    return d, x


def laplace_transform_collocation(s, *, mode: int, kinematic_viscosity: float,
                                  surface_tension: float, density: float, radius: float,
                                  initial_amplitude: float, points: int = 40) -> complex:
    """a_hat(s) from a Chebyshev collocation of the transformed linear problem.

    Independent of the potential/vortical splitting and of Bessel functions:
    the stream function Psi = F(r) sin(n theta) (u_r = n F / r cos,
    u_theta = -F' sin) and G = L F, with L F = F'' + F'/r - n^2 F / r^2 (G
    is minus the vorticity), satisfy the transformed vorticity equation
    nu L G = s G (u(0) = 0).  The pressure follows from the theta momentum
    equation, P = (r / n) (-rho s F' + mu G'), and at r = R
      zero tangential stress   -F'' + F'/R - n^2 F / R^2 = 0,
      normal stress            -P + 2 mu n (F'/R - F/R^2) = -gamma (n^2 - 1) a_hat / R^2,
      kinematic condition      s a_hat - a0 = n F / R.
    F and G have the parity (-1)^n on [-R, R] (regular at r = 0); the
    collocation uses 2 * points Chebyshev points on [-R, R] folded onto the
    positive half (Trefethen, Spectral Methods in MATLAB, 2000, ch. 11), with
    the row r = R of each equation replaced by one stress condition.
    """
    n = int(mode)
    _check_mode(n)
    mu = density * kinematic_viscosity
    m = int(points)
    big_n = 2 * m - 1                       # odd: r = 0 is not a node
    d, x = _cheb(big_n)
    r = radius * x
    d1 = d / radius
    d2 = d1 @ d1
    sigma = (-1.0) ** n
    ext = np.zeros((big_n + 1, m))
    for j in range(m):
        ext[j, j] = 1.0
        ext[big_n - j, j] = sigma
    inv_r = np.diag(1.0 / r)
    lap = (d2 + inv_r @ d1 - n * n * inv_r @ inv_r)[:m] @ ext
    d1_f = d1[:m] @ ext
    d2_f = d2[:m] @ ext
    eye = np.eye(m)
    s = complex(s)
    # Unknowns [F (m values), G (m values), a_hat]; row 0 of each block is r = R.
    a = np.zeros((2 * m + 1, 2 * m + 1), dtype=complex)
    rhs = np.zeros(2 * m + 1, dtype=complex)
    a[:m, :m] = lap
    a[:m, m:2 * m] = -eye
    a[m:2 * m, m:2 * m] = kinematic_viscosity * lap - s * eye
    a[0, :] = 0.0
    a[0, :m] = -d2_f[0] + d1_f[0] / radius - n * n * eye[0] / radius ** 2
    a[m, :] = 0.0
    a[m, :m] = (radius / n) * density * s * d1_f[0] + 2.0 * mu * n * (d1_f[0] / radius
                                                                     - eye[0] / radius ** 2)
    a[m, m:2 * m] = -(radius / n) * mu * d1_f[0]
    a[m, 2 * m] = surface_tension * (n * n - 1) / radius ** 2
    a[2 * m, :m] = -n * eye[0] / radius
    a[2 * m, 2 * m] = s
    rhs[2 * m] = initial_amplitude
    return complex(np.linalg.solve(a, rhs)[2 * m])

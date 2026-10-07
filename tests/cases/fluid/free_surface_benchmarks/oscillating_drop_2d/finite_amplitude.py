#!/usr/bin/env python3
"""Finite-amplitude frequency shift of the inviscid 2D drop (oscillating_drop_2d check).

The linear reference (drop_reference.py) has no amplitude dependence.  This
script measures the amplitude dependence of the mode-n frequency from the
fully nonlinear inviscid problem, so that the protocol amplitude can be
checked against it (the capillary-wave runs showed a finite-amplitude shift,
tracker M3).

Potential flow in the star-shaped drop r < rho(theta, t), velocity potential
phi with surface value Phi(theta, t), zero gravity, p_ext = 0:

    rho_t = phi_r - rho_theta phi_theta / rho^2                 (kinematic)
    Phi_t = -|grad phi|^2 / 2 - gamma kappa / rho_l + phi_r rho_t   (Bernoulli)

kappa the curvature of r = rho(theta).  phi is expanded in the harmonic
polynomials r^k cos(k theta), r^k sin(k theta) (k <= N/2), fitted to Phi at
N equispaced angles (Dirichlet-to-Neumann map), theta derivatives are
spectral, and time stepping is classical RK4.  The mode amplitude is
Re M_n / (pi R_A^(n+1)) with M_n = int rho^(n+2)/(n+2) exp(i n theta) dtheta,
the quantity verify.py measures, and its frequency comes from the same
damped-cosine fit (25 samples per period over `periods` periods).

Result (README.md): omega/omega0 - 1 = -0.770 eps^2 for n = 2, the same to
four digits at N = 64 and 128 and at 400 and 800 steps per period, for
eps = 0.005 to 0.05.  Only numpy is required.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _fit():
    path = HERE.parent / "capillary_wave_2d" / "verify.py"
    spec = importlib.util.spec_from_file_location("capillary_wave_2d_verify_for_finite_amplitude",
                                                  path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fit_damped_cosine


def simulate(eps: float, *, mode: int = 2, points: int = 64, steps_per_period: int = 400,
             periods: float = 4.0, samples_per_period: int = 25) -> dict:
    """Release r = R0 (1 + eps cos(n theta)) from rest (R = rho = gamma = 1)."""
    n, big_n = int(mode), int(points)
    if big_n % 2 or big_n < 8 or steps_per_period % samples_per_period:
        raise ValueError("need an even number of points >= 8 and whole samples per period")
    theta = 2.0 * math.pi * np.arange(big_n) / big_n
    wavenumbers = np.fft.fftfreq(big_n, 1.0 / big_n)

    def d_theta(f, order=1):
        return np.real(np.fft.ifft((1j * wavenumbers) ** order * np.fft.fft(f)))

    kc = np.arange(0, big_n // 2 + 1)
    ks = np.arange(1, big_n // 2)
    cos_c, sin_c = np.cos(np.outer(theta, kc)), np.sin(np.outer(theta, kc))
    cos_s, sin_s = np.cos(np.outer(theta, ks)), np.sin(np.outer(theta, ks))

    def rhs(rho, phi_s):
        rc, rs = rho[:, None] ** kc[None, :], rho[:, None] ** ks[None, :]
        coef = np.linalg.solve(np.hstack([rc * cos_c, rs * sin_s]), phi_s)
        a, b = coef[:kc.size], coef[kc.size:]
        drc = kc[None, :] * rho[:, None] ** np.maximum(kc - 1, 0)[None, :]
        drs = ks[None, :] * rho[:, None] ** (ks - 1)[None, :]
        phi_r = (drc * cos_c) @ a + (drs * sin_s) @ b
        phi_t = (-kc[None, :] * rc * sin_c) @ a + (ks[None, :] * rs * cos_s) @ b
        r1, r2 = d_theta(rho), d_theta(rho, 2)
        kappa = (rho ** 2 + 2.0 * r1 ** 2 - rho * r2) / (rho ** 2 + r1 ** 2) ** 1.5
        rho_t = phi_r - r1 * phi_t / rho ** 2
        return rho_t, -0.5 * (phi_r ** 2 + phi_t ** 2 / rho ** 2) - kappa + phi_r * rho_t

    def amplitude(rho):
        area = 0.5 * np.mean(rho ** 2) * 2.0 * math.pi
        moment = np.mean(rho ** (n + 2) / (n + 2) * np.exp(1j * n * theta)) * 2.0 * math.pi
        return moment.real / (math.pi * math.sqrt(area / math.pi) ** (n + 1)), area

    omega0 = math.sqrt(n * (n * n - 1))
    dt = 2.0 * math.pi / omega0 / steps_per_period
    rho = (1.0 + eps * np.cos(n * theta)) / math.sqrt(1.0 + 0.5 * eps * eps)
    phi_s = np.zeros(big_n)
    stride = steps_per_period // samples_per_period
    a, area0 = amplitude(rho)
    times, amps = [0.0], [a]
    for step in range(1, int(round(periods * steps_per_period)) + 1):
        k1 = rhs(rho, phi_s)
        k2 = rhs(rho + 0.5 * dt * k1[0], phi_s + 0.5 * dt * k1[1])
        k3 = rhs(rho + 0.5 * dt * k2[0], phi_s + 0.5 * dt * k2[1])
        k4 = rhs(rho + dt * k3[0], phi_s + dt * k3[1])
        rho = rho + dt / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        phi_s = phi_s + dt / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
        if step % stride == 0:
            a, area = amplitude(rho)
            times.append(step * dt)
            amps.append(a)
    fit = _fit()(np.array(times), np.array(amps), omega0)
    return {"eps": eps, "omega0": omega0, "omega": fit["omega"], "beta": fit["beta"],
            "relative_shift": fit["omega"] / omega0 - 1.0,
            "coefficient": (fit["omega"] / omega0 - 1.0) / eps ** 2,
            "area_drift": abs(area / area0 - 1.0), "converged": fit["converged"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eps", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05])
    parser.add_argument("--points", type=int, default=64)
    parser.add_argument("--steps-per-period", type=int, default=400)
    args = parser.parse_args(argv)
    for eps in args.eps:
        r = simulate(eps, points=args.points, steps_per_period=args.steps_per_period)
        print(f"eps = {eps:g}: omega/omega0 - 1 = {r['relative_shift']:+.4e} "
              f"(coefficient {r['coefficient']:+.4f} eps^2), fitted beta = {r['beta']:+.1e}, "
              f"area drift {r['area_drift']:.1e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

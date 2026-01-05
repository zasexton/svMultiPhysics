# FE/TimeStepping — Verification Problems

This file collects small, well-understood problems that are useful for verifying time integration schemes implemented through `FE/Forms` + `FE/Systems::TransientSystem` + `FE/TimeStepping`.

The intent is:
- keep problems small enough to run in unit tests,
- have closed-form reference solutions (or extremely accurate references),
- exercise the FE transient plumbing (symbolic `dt(·,k)` + history vectors + nonlinear solves).

---

## 1) Structural dynamics (Newmark-β / generalized-α)

### Problem: undamped harmonic oscillator

Continuous model (single DOF):
\[
u''(t) + \omega^2 u(t) = 0,\qquad
u(0)=u_0,\qquad u'(0)=v_0.
\]

Closed-form solution:
\[
u(t)=u_0\cos(\omega t) + \frac{v_0}{\omega}\sin(\omega t).
\]

**What to verify**
- **Order**: temporal convergence under refinement in \(\Delta t\).
- **Dispersion/phase**: period error for oscillatory problems.
- **Algorithmic damping** (generalized-α): high-frequency dissipation controlled by \(\rho_\infty\).

**FE realization options**
- Near-term (already supported by the symbolic plumbing): express the second derivative with `dt(u,2)` in Forms:
  - residual: `(dt(u,2)*v + omega*omega*(u*v)).dx()`
  - this is a reference discretization for wiring and convergence checks (not the final structural integrator).
- Longer-term (recommended for production Newmark/generalized-α): represent \((u,v)\) as mixed fields so Forms can express:
  - `dt(u) = v`, `dt(v) = a`, plus constitutive/internal-force operators in the residual.

---

## 2) First-order multiphysics (generalized-α for fluids/FSI)

### Problem: stiff linear relaxation (scalar)

Continuous model:
\[
u'(t) + \lambda u(t) = 0,\qquad u(0)=u_0,\qquad \lambda>0.
\]
Closed-form solution:
\[
u(t)=u_0 e^{-\lambda t}.
\]

**What to verify**
- **Order**: expected temporal convergence (2nd order for generalized-α in smooth regimes).
- **High-frequency damping**: as \(\rho_\infty\) decreases, higher-frequency content should be more strongly damped while low-frequency accuracy remains good.
- **Robustness on stiff decay**: stable behavior as \(\lambda\Delta t\) becomes large.

**FE realization**
- Use a mass-weighted residual to make the coefficient-space ODE exact:
  - residual: `(dt(u)*v + lambda*(u*v)).dx()`
  - the assembled semi-discrete system is `M (u' + lambda u) = 0`, so each DOF satisfies the scalar ODE.

---

## References (context)

- Chung & Hulbert (1993): generalized-α for structural dynamics (2nd order).
- Jansen–Whiting–Hulbert (2000): generalized-α for first-order systems (fluids / stabilized FE).
- Newmark (1959): Newmark-β family.


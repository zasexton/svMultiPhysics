# fitted_static_drop_2d (smoke)

A closed 2D liquid drop bounded by a **fitted** free surface, with the opt-in
fitted `SurfaceStress` (Laplace–Beltrami) capillary form
(`Allow_fitted_surface_stress=true`; `Physics/Docs/NavierStokesFreeSurface.md`,
"Fitted ALE free surfaces"). It is the fitted counterpart of `static_drop_2d`
(milestone M5, decision D5 in `Documentation/free_surface_program_tracker.md`)
and at present a smoke case: it reports the pressure jump against `gamma/R`
and the spurious velocity over a short run. The pressure-jump limit in
`tolerances.json` is the M2 working criterion and applies at `R/h = 32` only.

| File | Role |
|---|---|
| `generate_case.py` | writes `solver.xml`, the disk mesh with the initial fields, the free-surface face and `case.json` for one level |
| `verify.py` | reads the solver output, reports the metrics, applies `tolerances.json` where the level was run |
| `tolerances.json` | criteria, fixed on 2026-09-30 before the first smoke run |
| `tests/test_free_surface_benchmark_fitted_static_drop_2d.py` | checks of the scripts on synthetic data |

## Setup

As `static_drop_2d`: `rho = gamma = R = 1`, zero gravity, `La = 12`
(`mu = sqrt(2/La) = 0.4082`, viscous time `rho R^2/mu = 2.45`), `p_ext = 0`.

| Input | Value |
|---|---|
| Mesh | hexagonal ring triangulation of the disk: ring `i` at radius `i R/(R/h)` carries `6 i` equally spaced nodes; the free surface is the regular `N`-gon, `N = 6 R/h` (48, 96, 192 sides; 217, 817, 3,169 vertices) |
| Initial state | `u = 0`, `p = gamma/R` everywhere, mesh at rest |
| Free surface | `FittedALE`, `Surface_tension_form=SurfaceStress`, `Allow_fitted_surface_stress=true`, `Kinematic_enforcement=MeshNitsche`, `Kinematic_nitsche_gamma=10`, `Tangential_mesh_policy=Free` (as `fitted_sloshing_2d`) |
| Mesh motion | harmonic mesh velocity (`Harmonic_quantity=velocity`), `Kappa=1`, no Dirichlet condition (the normal Nitsche row fixes the boundary; the zero tangential flux excludes rigid rotation) |
| Time | generalized-alpha `rho_inf = 0.5`, `dt = 0.01` (capillary time 1), `t = 1`, output every 10 steps |
| Solvers | Newton tolerance `1e-6`, at most 8 iterations; Eigen direct |

**Expected result.** On the regular `N`-gon the discrete Laplace–Beltrami
load at a node is `2 gamma sin(pi/N)` inward, and a constant pressure `p`
loads it with `p 2R sin(pi/N) cos(pi/N)` outward: the constant pressure
`gamma/(R cos(pi/N))` balances it exactly, with `u = 0`. The relaxed drop
should therefore carry spurious currents at roundoff level, and its pressure
should exceed `gamma/R` by `1/cos(pi/N) - 1` (2.1e-3, 5.4e-4, 1.3e-4 at
`N = 48, 96, 192`), second order in `h`.

## Metrics

| Metric | Definition |
|---|---|
| `pressure_jump_relative_error` | `abs(mean(p) - gamma/R)/(gamma/R)` on the final state |
| `pressure_jump_relative_error_effective_radius` | the same against `gamma/R_eff`, `R_eff = sqrt(A/pi)` of the current mesh |
| `pressure_jump_over_polygon_balance` | `mean(p) / (gamma/(R cos(pi/N)))` |
| `spurious_capillary_number_max` | max over outputs `t > 0` of `mu max|u| / gamma` |
| `liquid_area_relative_deviation_max` | max over outputs of `abs(A(t) - A(0))/A(0)` |

## Results

**2026-09-30, source and solver `a6a8a72d`, Slurm job `46109392`.** The
pressure-jump criterion (`R/h = 32`) passes; the rest is reported.

| R/h | N | vertices | `mean(p)` | error vs `gamma/R` | error vs `gamma/R_eff` | `mean(p) / (gamma/(R cos(pi/N)))` | max `mu |u|/gamma`, t > 0 | final | max `dA/A` | s/step |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 48 | 217 | 1.0021457 | 2.15e-3 | 7.1e-4 | 1.0000000 | 3.9e-8 | 1.7e-11 | 1.3e-10 | 0.22 |
| 16 | 96 | 817 | 1.0005357 | 5.36e-4 | 1.8e-4 | 1.0000000 | 5.1e-9 | 5.5e-14 | 3.7e-12 | 0.40 |
| 32 | 192 | 3,169 | 1.0001339 | 1.34e-4 | 4.5e-5 | 1.0000000 | 7.4e-10 | 4.2e-15 | 5.1e-14 | 1.50 |

- The relaxed pressure equals the discrete balance `gamma/(R cos(pi/N))` to
  seven digits at every level, so its error against `gamma/R` is exactly
  `1/cos(pi/N) - 1` (ratio 4.0 per refinement, second order).
- The spurious capillary number peaks in the first output (the pressure
  moves from the initial `gamma/R` to the balanced value) and decays to
  roundoff; the surface stays a regular polygon (radius spread below 6e-12)
  and the area is conserved to 1e-10 or better.
- About 1.1 Newton iterations per step. The harmonic mesh operator has no
  Dirichlet condition here; the normal Nitsche row fixes translations and
  the natural tangential flux excludes rigid rotation.

Raw output: `/scratch/users/zsexton/free-surface-benchmarks/fitted_ale/static_drop/m5-a6a8a72d/`
(`verify.txt`, `verify.json`).

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
| Mesh motion | harmonic, `Kappa=1`, no Dirichlet condition (the normal Nitsche row fixes the boundary; the zero tangential flux excludes rigid rotation) |
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

Pending.

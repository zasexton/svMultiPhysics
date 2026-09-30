# fitted_capillary_wave_2d

The small-amplitude standing capillary wave of `capillary_wave_2d`, computed
with the **fitted ALE** free surface: the free surface is a boundary of the
liquid mesh, which moves with a coupled harmonic mesh-velocity extension, and
surface tension is the fitted Laplace–Beltrami form. It is the independent
reference for the unfitted M3 capillary wave (decision D5 in
`Documentation/free_surface_program_tracker.md`): same physical case, same
Prosperetti reference, same metrics and the same criteria, with space and
time convergence judged separately (D10) and the maximum area deviation over
the run (D11).

Files:

| File | Role |
|---|---|
| `generate_case.py` | writes `solver.xml`, the liquid mesh with the initial fields, the boundary faces and `case.json` for one case |
| `verify.py` | reads the solver output of the protocol runs, fits frequency and damping, removes the time error from the spatial study, applies `tolerances.json` |
| `tolerances.json` | acceptance criteria and their sources, fixed on 2026-09-30 before the first protocol run |
| `tests/test_free_surface_benchmark_fitted_capillary_wave_2d.py` | checks of the two scripts on synthetic data |

The physical constants, the time schedule and the reference
(`prosperetti_reference.py`) are imported from `../capillary_wave_2d/`, and
the amplitude and fit code from its `verify.py`; the moving-mesh reader comes
from `../fitted_sloshing_2d/verify.py`. Nothing is copied.

## Physical setup

As `capillary_wave_2d` (nondimensional, `rho = gamma = lambda = 1`, `k = 2 pi`):
half a wavelength between free-slip mirror walls at the crest (`x = 0`) and the
trough (`x = lambda/2`), a liquid of mean depth
`y0 = lambda (1 + sqrt(2)/100)` below the surface `y0 + a0 cos(kx)`,
`a0 = 0.01 lambda`, zero gravity, void exterior with `p_ext = 0`,
`La = rho gamma lambda / mu^2 = 3000` (`mu = 0.018257`), released from rest
with the linear pressure of the released state. Reference: Prosperetti's
viscous initial-value solution, whose 4-period damped-cosine fit gives
`omega_ref = 15.5273` and `beta_ref = 1.2116` (normal mode `omega = 15.5366`,
`beta = 1.2185`; `omega0 = 15.7496`, `2 nu k^2 = 1.4415`).

## Fitted discretization

Every numerical value is fixed for all cases (principle P1).

| Input | Value | Reason |
|---|---|---|
| Mesh | liquid region only: a structured `level/2 x level` grid on `[0, lambda/2] x [0, 1]` mapped onto `0 <= y <= y0 + a0 cos(kx)` (`y -> s (y0 + eta(x))`), each cell split into two `Triangle3` with the diagonal alternating with cell parity (mirror-symmetric about the node line `x = lambda/4`); `lambda/h = 16, 32, 64` (153, 561, 2,145 vertices); row height `y0/level = 1.014 h` | successive levels halve both cell sizes; the free-surface nodes lie on the initial cosine, so the sampled P1 region has exactly the area `y0 lambda/2` |
| Fluid | P1/P1 residual-based VMS, generalized-alpha `rho_inf = 0.5`, coupled ALE (`Mesh_velocity_source=coupled_displacement`) assembled on the current configuration | as `fitted_sloshing_2d` |
| Free surface | `Implementation=FittedALE`, `Surface_tension=1`, `Surface_tension_form=SurfaceStress` with `Allow_fitted_surface_stress=true`, `External_pressure=0`, `Kinematic_enforcement=MeshNitsche`, `Kinematic_nitsche_gamma=10`, `Tangential_mesh_policy=Free`, no contact-line model | the fitted Laplace–Beltrami form `gamma (I - n n) : grad(v)` on the current boundary (`Physics/Docs/NavierStokesFreeSurface.md`); MeshNitsche and `gamma_N = 10` as `fitted_sloshing_2d` |
| Mesh motion | harmonic extension of the mesh velocity (`Harmonic_quantity=velocity`), `Kappa=1` | parameter-free (D5) |
| Walls | fluid: `Dir 0` with `Effective_direction` (`1 0` on the side walls, `0 1` on the bottom); mesh: the same input, so boundary nodes slide along the walls and the contact points move | free-slip mirror planes; the natural boundary term of `SurfaceStress` at the contact point vanishes for the 90° mirror angle |
| Nonlinear / linear solve | relative tolerance `1e-6`, at most 12 Newton iterations; Eigen sparse LU | as `fitted_sloshing_2d` |
| Output | 100 snapshots over 4 inviscid periods | as `capillary_wave_2d` |

**Time steps (D10).** The spatial study runs the three meshes with the
`capillary_wave_2d` step shared by all levels, `dt = 5.5027e-4` (2,900 steps):
the largest step within the one-sided capillary limit
`sqrt(rho h^3/(4 pi gamma))` of `lambda/h = 64` that divides the run into 100
output intervals, i.e. 0.12, 0.35 and 1.00 times the limit of 16, 32 and 64.
The separate time-step study runs `lambda/h = 32` at `dt`, `dt/2` and `dt/4`
(the `dt` run is shared). `verify.py` extrapolates `omega` and `beta` to
`dt -> 0` (order from the three steps, limit from the two smallest) and
subtracts the time error `e_t(dt)` of the shared step from every
spatial-study frequency and damping rate, as `fitted_sloshing_2d` does.

**Capillary time-step limit.** The limit is a stability bound for surface
tension that is explicit in the interface geometry (Brackbill, Kothe and
Zemach 1992). Here the fitted surface, its normal and the Laplace–Beltrami
term are evaluated on the trial current configuration inside the Newton loop
(`Moving_mesh_tangent_path=SymbolicRequired`), so the geometry is implicit.
The protocol nonetheless respects the limit on every level. The diagnostic
option `--dt-over-capillary-limit F` (never gated) writes a run with a step of
about `F` times the level's limit; it is used below to check whether the
fitted discretization needs the limit.

## Metrics (`verify.py`)

The solver writes the reference configuration as VTK points and the current
coordinates (`CurrentCoordinates`) and `mesh_displacement` as point data.

| Metric | Definition |
|---|---|
| mode amplitude `a_h(t)` | exact `cos(kx)` coefficient of the P1 free surface, `(2/W) int_liquid cos(kx) dA`, summed exactly over the current mesh triangles with Green's theorem (`capillary_wave_2d/verify.py`); tangential motion of the surface nodes and a uniform level drift do not enter |
| fit | `a(t) = exp(-beta t)(c1 cos(omega t) + c2 sin(omega t))` over all outputs including `t = 0`; the same fit of Prosperetti's `a(t)` at the same times gives `omega_ref`, `beta_ref` |
| `frequency_spatial_relative_error` | `abs(omega - e_t^omega(dt) - omega_ref)/omega_ref` for the spatial-study runs |
| `damping_spatial_relative_error` | `abs(beta - e_t^beta(dt) - beta_ref)/beta_ref` |
| `liquid_area_relative_deviation_max` | max over the outputs (and `t = 0`) of `abs(A(t) - A(0))/A(0)`, `A` the area of the current liquid mesh |
| Reported only | raw signed errors; `a_h(0)/a0 - 1`; the RMS and maximum of `a_h(t)/a_h(0) - a(t)/a0` (Popinet's measure); contact-point elevations; the smallest triangle angle over the run; the largest wall offset of the surface end nodes; maximum speed; Newton iterations and wall time from `solver_run.log.gz` |

`verify.py` exits with status 2 on missing or inconsistent data (no
`case.json` or a case of another benchmark, no output, missing arrays, a
non-`Triangle3` mesh, an inverted triangle, non-finite values, a run that
stopped before its end time) and on `--max-steps` smoke runs unless
`--allow-truncated` is given.

## Tolerances and their sources

The criteria are those of `capillary_wave_2d/tolerances.json`, so both M3
references are judged alike; only the time-error removal is added.

| Criterion | Limit | Where | Source |
|---|---|---|---|
| `frequency` | at most 0.02; observed order at least 1 | 0.02 at `lambda/h = 32`; order over 16/32/64 | tracker M3 working criterion (D1); time error removed (D10) |
| `damping` | at most 0.05; observed order at least 1 | `lambda/h = 32` and 64; order over 16/32/64 | tracker M3 working criterion (D1); finest-level check as in D12 |
| `volume` | at most 1e-4 | every protocol run, maximum over the run | tracker volume limit (D1), gated as in D11 |

## How to run

```bash
B=tests/cases/fluid/free_surface_benchmarks/fitted_capillary_wave_2d
OUT=$SCRATCH/free-surface-benchmarks/fitted_ale/capillary_wave/$(git rev-parse --short HEAD)
for L in 16 32 64; do python3 $B/generate_case.py --level $L --output-dir $OUT/L$L; done
for D in 2 4; do python3 $B/generate_case.py --level 32 --dt-divisor $D --output-dir $OUT/L32_dt$D; done
# in each case directory, inside a Slurm job:
#   mpiexec -n 1 --bind-to none /path/to/svmultiphysics solver.xml 2>&1 | gzip -1 > solver_run.log.gz
python3 $B/verify.py $OUT/L16 $OUT/L32 $OUT/L64 $OUT/L32_dt2 $OUT/L32_dt4 --json $OUT/verify.json
```

## Results

Pending.

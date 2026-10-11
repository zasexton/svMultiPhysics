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
`a0 = 0.0025 lambda` (decision D35; `capillary_wave_2d` uses `0.01 lambda`), zero gravity, void exterior with `p_ext = 0`,
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

**Amplitude (decision D35).** The reference is linear, while the simulation
keeps the finite amplitude, whose frequency shift is about
`-0.10 to -0.16 (a0 k)^2` (measured below).  At the earlier protocol
amplitude `a0 = 0.01 lambda` (`a0 k = 0.063`) that shift is a plateau of
4e-4 to 7e-4 below the linear reference, larger than the discretization
error, and the frequency order could not be demonstrated.  The protocol
amplitude is therefore `a0 = 0.0025 lambda` (`a0 k = 0.016`), where the
shift is 16 times smaller.  The diagnostic option
`--amplitude-over-wavelength` (never gated) runs another amplitude;
`--amplitude-over-wavelength 0.01` writes the decks of the earlier protocol
unchanged (`solver.xml` and the mesh files are byte-identical), and
`verify.py` reports runs at an amplitude other than the protocol value,
including old protocol runs whose `case.json` still marks them as protocol
runs, as diagnostics.

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
| `frequency` | at most 0.02; observed order at least 1 | 0.02 at `lambda/h = 32`; order over 16/32/64 | tracker M3 working criterion (D1); time error removed (D10); at `a0 = 0.0025 lambda` (D35) |
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

**2026-10-10, protocol amplitude `a0 = 0.0025 lambda` (D35), cases and
solver binary from `f531567f`, Slurm job `47299149` (one node, the five
cases side by side, one rank each). Result: FAIL on the frequency
convergence order only.** Output and `verify.json`:
`/scratch/users/zsexton/free-surface-benchmarks/fitted_ale/capillary_wave/f531567f/`.

| lambda/h | `omega` | raw error | spatial error | `beta` | raw error | spatial error | RMS `a_h` error | `max dA/A` | wall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 15.529342 | +1.3e-4 | +4.4e-5 | 1.25646 | +3.70% | +3.76% | 7.5e-3 | 3.0e-7 | 183 s |
| 32 | 15.526715 | -3.8e-5 | -1.25e-4 | 1.22121 | +0.79% | +0.85% | 1.4e-3 | 8.6e-8 | 628 s |
| 64 | 15.527700 | +2.6e-5 | -6.2e-5 | 1.21283 | +0.10% | +0.16% | 5.2e-4 | 2.3e-8 | 2,859 s |

Time-step study (`lambda/h = 32`, `dt/2` 1,197 s, `dt/4` 2,345 s):
`omega(dt -> 0) = 15.525354` (order 0.91), `beta(dt -> 0) = 1.221900`
(order 0.59); time errors of the shared step `+8.8e-5` in `omega` and
`-5.7e-4` in `beta`.

| Criterion | Result |
|---|---|
| `frequency` | **FAIL**: 1.25e-4 at `lambda/h = 32` (limit 0.02, passed); observed order -0.25 (pairwise -1.52, 1.02) < 1 |
| `damping` | PASS: 0.85% at 32 and 0.16% at 64 (limit 5%); observed order 2.28 (pairwise 2.14, 2.41) |
| `volume` | PASS: at most 3.0e-7 (limit 1e-4) |

At the smaller amplitude the frequency errors fall from 4e-4 to 7e-4 to
at most 1.25e-4 (160 times below the limit), the size of the time error of
the shared step and of the remaining finite-amplitude shift
(`0.10 to 0.16 (a0 k)^2 = 2.5e-5 to 4e-5`); they change sign between the
levels and do not decrease with h, so the order criterion still fails, now
at an error floor. The damping and the area converge at second order.

**2026-09-30, cases from `93b1bf06`, solver binary `35a81fd3` (the same
fitted code), Slurm job `46129890` (one node, all cases side by side, one
rank each). Result: FAIL on the frequency convergence order only.**

Spatial study (`dt = 5.5027e-4` on every level; "spatial" is the error with
the time error of that step removed, `e_t = +8.5e-5` in `omega` and
`-5.2e-4` in `beta`, relative):

| lambda/h | vertices | `omega` | raw error | spatial error | `beta` | raw error | spatial error | RMS `a_h` error | `max dA/A` | Newton/step | s/step | wall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 153 | 15.52231 | -3.2e-4 | -4.1e-4 | 1.25101 | +3.26% | +3.31% | 6.0e-3 | 4.9e-6 | 2.0 | 0.10 | 299 s |
| 32 | 561 | 15.51733 | -6.4e-4 | -7.3e-4 | 1.21710 | +0.46% | +0.51% | 4.0e-3 | 1.4e-6 | 2.0 | 0.35 | 1,025 s |
| 64 | 2,145 | 15.52191 | -3.5e-4 | -4.3e-4 | 1.20890 | -0.22% | -0.17% | 2.6e-3 | 3.7e-7 | 2.0 | 1.58 | 4,576 s |

Reference fit: `omega_ref = 15.52730`, `beta_ref = 1.211580`.

Time-step study (`lambda/h = 32`):

| step | `omega` | change from the previous step | `beta` | `max dA/A` | wall |
|---|---:|---:|---:|---:|---:|
| `dt` | 15.517331 | | 1.217102 | 1.4e-6 | 1,025 s |
| `dt/2` | 15.516708 | -4.0e-5 | 1.217299 | 1.4e-6 | 2,043 s |
| `dt/4` | 15.516380 | -2.1e-5 | 1.217433 | 1.4e-6 | 3,967 s |

Richardson: `omega(dt -> 0) = 15.516012` (order 0.92), `beta(dt -> 0) =
1.217727` (order 0.54). The time error of the shared step is below 1e-4 in
`omega` and 6e-4 in `beta`, and converges at first order rather than the
second order of `fitted_sloshing_2d`; it does not affect the verdicts.

| Criterion | Result |
|---|---|
| `frequency` | **FAIL**: 7.3e-4 at `lambda/h = 32` (limit 0.02, passed); observed order -0.04 (pairwise -0.84, 0.75) < 1 |
| `damping` | PASS: 0.51% at 32 and 0.17% at 64 (limit 5%); observed order 2.14 (pairwise 2.70, 1.58) |
| `volume` | PASS: at most 4.9e-6 (limit 1e-4); the area oscillates reversibly (no drift) with an amplitude that falls at second order |

Capillary time-step limit (diagnostic runs, not gated; same metrics, time
error included):

| lambda/h | `dt` | `dt` / one-sided limit | steps per period | `omega` error | `beta` error | Newton/step |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1.60e-2 | 3.6 | 25 | -4.9e-3 | +1.9% | 4.2 |
| 64 | 2.00e-3 | 3.6 | 200 | -2.3e-4 | -0.24% | 3.0 |
| 64 | 3.99e-3 | 7.2 | 100 | -3.7e-4 | -0.32% | 3.0 |

Steps of 3.6 and 7.2 times the capillary limit are stable and accurate:
the fitted surface, its normal and the Laplace–Beltrami term are implicit in
the Newton loop, so the explicit-capillarity limit does not bind this
discretization. The protocol keeps it anyway.

Observations:

- All three frequency errors lie between -4e-4 and -7e-4, 30 to 50 times
  below the limit; the order criterion fails because the error does not
  decrease with h. The reference is linear, while the simulation keeps the
  finite amplitude `a0 k = 0.063`, and the size of the plateau,
  `-0.1 to -0.2 (a0 k)^2`, matches a finite-amplitude frequency shift.
  The small-amplitude diagnostic (`--amplitude-over-wavelength 0.0025`,
  cases from `f4ff0b13`, binary `4a62a971`, job `46155727`) confirms it:

  | lambda/h | raw frequency error | raw damping error | RMS `a_h` error | `max dA/A` |
  |---:|---:|---:|---:|---:|
  | 16 | +1.3e-4 | +3.70% | 7.5e-3 | 3.0e-7 |
  | 32 | -3.8e-5 | +0.79% | 1.4e-3 | 8.6e-8 |
  | 64 | +2.6e-5 | +0.10% | 5.2e-4 | 2.3e-8 |

  At a quarter of the amplitude the frequency error falls to the size of
  the time error (about 1e-4 and below) at `lambda/h = 32` and 64, and the
  damping converges at order 2.2 and 3.0. The difference between the two
  amplitudes, scaled to the full one (`x 16/15`), is -4.8e-4, -6.5e-4 and
  -4.0e-4 at 16/32/64, i.e. -0.10 to -0.16 `(a0 k)^2`: the plateau is the
  finite-amplitude frequency shift of the nonlinear simulation, which the
  linear reference does not contain. Judging the frequency order at the
  protocol amplitude needs a smaller amplitude or a reference with the
  amplitude correction (open decision); the criteria were not changed.
- The damping converges at second order to 0.17% at `lambda/h = 64`. The
  sampled amplitude `a_h(0)/a0 - 1` is -1.3%, -0.32% and -0.08%
  (second-order P1 sampling, as in `capillary_wave_2d`), and the RMS
  amplitude error of the normalized history (Popinet's measure) is 6.0e-3,
  4.0e-3 and 2.6e-3.
- Every step took two Newton iterations (final residuals at most 5e-9); the
  surface end nodes stay on the walls; the smallest triangle angle falls
  from 42.9° to 41.6° at most.

Comparison with the unfitted M3 runs (`capillary_wave_2d`, `surface_stress`
with the PDE extension, the same step, binary `35a81fd3`, jobs `46130411`,
`46130412`, `46130432`; raw errors, as that benchmark reports them;
`verify.py` output in `.../runs/fitted_capillary_wave/unfitted_m3_verify.txt`):

| lambda/h | frequency error, unfitted | fitted | damping error, unfitted | fitted | `max dA/A`, unfitted | fitted |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | +0.80% | -0.03% | +26.4% | +3.3% | 1.1e-5 | 4.9e-6 |
| 32 | +0.66% | -0.06% | +7.8% | +0.46% | 1.3e-4 | 1.4e-6 |
| 64 | +0.23% | -0.03% | +0.06% | -0.22% | 7.3e-5 | 3.7e-7 |

The unfitted study fails all three criteria: frequency order 0.91 (errors
0.80%, 0.66%, 0.23%), damping 7.8% at `lambda/h = 32`, and area 1.3e-4 at
32. Its time-step study at 32 moves `omega` by 0.2% and `beta` by 4% between
`dt` and `dt/4`, against 6e-5 and 0.03% for the fitted path. The fitted
frequency errors are 7 to 25 times smaller at every level, the damping
errors 8 and 17 times smaller at 16 and 32 (at 64 both are below 0.25%:
0.06% unfitted, 0.22% fitted), and the area deviation 2 to 200 times
smaller.

Raw output: `/scratch/users/zsexton/svmp-dev-fitted2/runs/fitted_capillary_wave/93b1bf06/`
(`verify.txt`, `verify.json`) and `.../f4ff0b13/` (small-amplitude
diagnostic, `verify_amp0025.txt`).

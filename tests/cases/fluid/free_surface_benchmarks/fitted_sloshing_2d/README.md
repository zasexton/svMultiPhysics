# fitted_sloshing_2d

The first antisymmetric standing wave in a 2D rectangular tank, gravity only,
computed with the **fitted ALE** free surface: the free surface is a boundary
of the liquid mesh, which moves with a coupled harmonic mesh displacement.
It is the fitted counterpart of `linear_sloshing_2d` (same tank, mode,
amplitude, viscosity and reference solution) and the first physical case of
the fitted reference path (milestone M5, decision D5 in
`Documentation/free_surface_program_tracker.md`). The frequency and damping
rate of the free-surface elevation are compared with the exact linear
viscous dispersion relation, with space and time convergence judged
separately (D10), the damping at the finest level (D12), and the maximum
liquid-area deviation over the run (D11).

Files:

| File | Role |
|---|---|
| `generate_case.py` | writes `solver.xml`, the liquid mesh with the initial fields, the boundary faces and `case.json` for one case |
| `verify.py` | reads the solver output of the protocol runs, fits frequency and damping, removes the time error from the spatial study, applies `tolerances.json` |
| `tolerances.json` | acceptance criteria and their sources, fixed on 2026-09-30 before the first protocol run |
| `tests/test_free_surface_benchmark_fitted_sloshing_2d.py` | checks of the two scripts on synthetic data |

The reference code (dispersion relation, schedule, VTU writer, damped-
oscillation fit) is imported from `../linear_sloshing_2d/`, so both
benchmarks use one reference.

## Physical setup

As `linear_sloshing_2d` (nondimensional, `rho = g = L = 1`):

| Quantity | Value |
|---|---|
| Tank | `x` in `[0, 1]`; the liquid fills `0 <= y <= H0 + eta` |
| Mean depth `H0` | `0.5 + 1/128 = 0.5078125` |
| Mode | `n = 1`, `k = pi`, initial surface `eta = A cos(k x)`, `A = 0.005` |
| Kinematic viscosity `nu` | `5e-4` |
| Initial state | at rest; the linear pressure `p = rho g (H0 - y) + rho g A cosh(k y)/cosh(k H0) cos(k x)` sampled on every vertex |
| Walls | free slip (left, right, bottom) |
| Exterior | void, `p_ext = 0`, surface tension 0 |
| Reference | `omega_ref = 1.7006229`, `gamma_ref = 9.5229e-3` (exact linear viscous standing wave with free-slip walls; `linear_sloshing_2d/README.md`) |

## Fitted discretization

Every numerical value is fixed for all cases (principle P1).

| Input | Value | Reason |
|---|---|---|
| Mesh | liquid region only: a structured `L/h x L/(2h)` grid on `[0, 1] x [0, H0]` mapped vertically onto the initial surface (`y -> y (H0 + eta(x))/H0`), each cell split into two `Triangle3` with the diagonal alternating with cell parity (mirror-symmetric about `x = 1/2`); `L/h = 16, 32, 64` (153, 561, 2,145 vertices); row height `1.016 h` | successive levels halve both cell sizes |
| Fluid | P1/P1 with residual-based VMS (the module default), generalized-alpha `rho_inf = 0.5`, coupled ALE (`Mesh_velocity_source=coupled_displacement`), assembled on the current configuration | production fluid module |
| Free surface | `Implementation=FittedALE`, `External_pressure=0`, `Surface_tension=0`, `Kinematic_enforcement=MeshNitsche`, `Kinematic_nitsche_gamma=10`, `Normal_kinematic_policy=MatchFluidNormalVelocity`, `Tangential_mesh_policy=Free` | the fluid keeps the natural dynamic condition; the normal kinematics are enforced on the mesh by a consistent Nitsche row (`Physics/Docs/NavierStokesFreeSurface.md`, "Fitted ALE free surfaces") |
| Mesh motion | harmonic extension of the mesh velocity (`Harmonic_quantity=velocity`), `Kappa=1` (it only scales the mesh equation) | parameter-free extension (D5); the MeshNitsche row constrains the normal mesh velocity, so the operator acts on it (a displacement operator lets the P1 flux defect accumulate, see below) |
| Nitsche constant | `gamma_N = 10`, penalty `gamma_N / h_n` on the normal mesh-velocity mismatch | fixed once: the mesh row is coercive for `gamma_N > 2 kappa` by the P1 trace inverse inequality with `h_n = 2|T|/|F|`; 10 leaves a factor 5 |
| Walls | fluid: `Dir 0` with `Effective_direction` (`1 0` on the side walls, `0 1` on the bottom); mesh: the same input in the `mesh_motion` equation, so boundary nodes slide along the walls and the contact points move | free slip; D5 sliding mesh walls |
| Equation order | fluid before `mesh_motion` | the harmonic module adds the Nitsche consistency term on the declared free surface |
| Nonlinear solve | relative tolerance `1e-6` per equation, at most 12 Newton iterations | |
| Linear solve | Eigen direct (sparse LU) | small 2D meshes |
| Output | 32 snapshots per period, 4 periods | as `linear_sloshing_2d` |

**Time steps (D10).** The spatial study runs the three meshes at the same
time step, `dt = T0/512`. The separate time-step study runs `L/h = 32` at
64, 128, 256 and 512 steps per period (the 512 run is shared with the
spatial study). `verify.py` extrapolates `omega(dt -> 0)` from the two
finest steps with the order observed on the three finest, and subtracts the
resulting time error `e_t(T0/512)` from every spatial-study frequency. For
the oscillator `y' = i omega y` the generalized-alpha phase error at
`rho_inf = 0.5` is about `-4.2e-3 (32/S)^2` at `S` steps per period, so
`e_t(T0/512)` is expected near `-1.6e-5`.

## Metrics (`verify.py`)

The solver writes the reference configuration as VTK points and the current
coordinates (`CurrentCoordinates`) and `mesh_displacement` as point data.

| Metric | Definition |
|---|---|
| probe elevation `eta_p(t)` | current height of the free surface at the left wall `x = 0` (the free-surface end node, which slides along the wall), minus `H0` |
| fit | least-squares fit of `eta_p(t) = c + exp(-gamma t)(a cos(omega t) + b sin(omega t))` over all outputs (the `linear_sloshing_2d` fit) |
| `frequency_spatial_relative_error` | `abs(omega - e_t(T0/512) - omega_ref)/omega_ref` for the spatial-study runs |
| `frequency_time_error` | `abs(omega(32, dt) - omega(32, dt -> 0))/omega_ref` for the time-step study |
| `damping_rate_relative_error` | `abs(gamma - gamma_ref)/gamma_ref` |
| `liquid_area_relative_deviation_max` | max over outputs of `abs(A(t) - A(0))/A(0)`, `A` the area of the current liquid mesh |
| Reported only | raw signed frequency error; `gamma/gamma_ref`; fitted amplitude; fit residual; the same fit on the modal amplitude `a1(t)` of `y = sum_{n<4} a_n cos(n k x)` fitted to the free-surface nodes; maximum speed; the largest wall offset of the surface end nodes (checks that they stay on the walls); Newton statistics and wall time from `solver_run.log.gz` |

`verify.py` exits with status 2 on missing or inconsistent data (no
`case.json`, no output, missing arrays, a non-`Triangle3` mesh, an inverted
triangle, non-finite values, a run that stopped before its end time) and on
`--max-steps` smoke runs unless `--allow-truncated` is given.

## Tolerances and their sources

| Criterion | Limit | Where | Source |
|---|---|---|---|
| `frequency` | spatial error at most 0.01; strictly decreasing; observed order at least 1 | limit at `L/h = 64`; monotonicity and order over 16/32/64 | D10 and the M1/M5 working criterion, read as in `linear_sloshing_2d` |
| `time_step_study` | observed order at least 1 | 64/128/256/512 steps per period at `L/h = 32` | D10; justifies the time-error removal (generalized-alpha is second order) |
| `damping` | at most 0.05 | `L/h = 64` | D12 |
| `volume` | at most 1e-4 | every protocol run | D11 |

## How to run

```bash
B=tests/cases/fluid/free_surface_benchmarks/fitted_sloshing_2d
OUT=$SCRATCH/free-surface-benchmarks/fitted_ale/sloshing/$(git rev-parse --short HEAD)
for L in 16 32 64; do python3 $B/generate_case.py --level $L --output-dir $OUT/L$L; done
for S in 64 128 256; do python3 $B/generate_case.py --level 32 --steps-per-period $S --output-dir $OUT/L32_T$S; done
# in each case directory, inside a Slurm job (benchmark README):
#   mpiexec -n 1 --bind-to none /path/to/svmultiphysics solver.xml 2>&1 | gzip -1 > solver_run.log.gz
python3 $B/verify.py $OUT/L16 $OUT/L32 $OUT/L64 $OUT/L32_T64 $OUT/L32_T128 $OUT/L32_T256 --json $OUT/verify.json
```

Diagnostic options of `generate_case.py` (reported by `verify.py`, never
gated): `--kinematic-enforcement Nitsche|Penalty` (the older fitted modes,
which also impose the kinematic relation on the fluid) and
`--wall-mesh-motion pinned` (Dirichlet-0 mesh walls, which pin the contact
points).

## Results

**2026-09-30, source and solver `a6a8a72d`, Slurm job `46104905` (one node,
six cases in parallel, one rank each). Result: PASS.**

Spatial study (`dt = T0/512`; "spatial" is the frequency error with the
time error `e_t(T0/512) = -1.24e-5` removed):

| L/h | vertices | `omega` | raw error | spatial error | `gamma/gamma_ref` | damping error | `max dA/A` | Newton/step | s/step | wall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 153 | 1.6986093 | -1.18e-3 | -1.17e-3 | 1.0665 | 6.6% | 5.8e-7 | 2.0 | 0.06 | 128 s |
| 32 | 561 | 1.7001474 | -2.80e-4 | -2.67e-4 | 0.9963 | 0.37% | 1.0e-7 | 2.0 | 0.23 | 466 s |
| 64 | 2,145 | 1.7005905 | -1.90e-5 | -6.6e-6 | 0.9888 | 1.1% | 2.1e-8 | 2.0 | 0.93 | 1,907 s |

Time-step study (`L/h = 32`; `omega(dt -> 0) = 1.7001684` from the 256 and
512 runs, Richardson order 2.13):

| steps/T | `omega` | time error `/omega_ref` | `gamma/gamma_ref` | `max dA/A` | wall |
|---:|---:|---:|---:|---:|---:|
| 64 | 1.6984653 | -1.00e-3 | 0.9949 | 8.6e-8 | 63 s |
| 128 | 1.6997666 | -2.36e-4 | 0.9956 | 9.4e-8 | 118 s |
| 256 | 1.7000764 | -5.41e-5 | 0.9961 | 9.9e-8 | 229 s |
| 512 | 1.7001473 | -1.24e-5 | 0.9963 | 1.0e-7 | 466 s |

| Criterion | Result |
|---|---|
| `frequency` | PASS: 6.6e-6 at `L/h = 64` (limit 1e-2); decreasing; observed order 3.73 (pairwise 2.13, 5.33) |
| `time_step_study` | PASS: observed order 2.11 (pairwise 2.08, 2.13, 2.13); the generalized-alpha estimate `-4.2e-3 (32/S)^2` gives -1.05e-3 at 64 steps |
| `damping` | PASS: 1.1% at `L/h = 64` (limit 5%) |
| `volume` | PASS: at most 5.8e-7 (limit 1e-4) |

Observations:

- The 16 to 32 pair converges at second order. At `L/h = 64` the spatial
  error (6.6e-6) is of the size of the removed time error (1.2e-5) and of
  the fit-to-fit spread (the modal fit gives -1.0e-4), so the 32 to 64
  pairwise order (5.3) only says that the error has reached that floor.
- The damping rate is within 0.4% and 1.1% of the exact viscous value at 32
  and 64; at 16 it is 6.6% high. The Lamb estimate `2 nu k^2` is 3.6% above
  `gamma_ref` (free-surface vorticity layer); the fitted runs resolve the
  difference.
- The free-surface end nodes stay on the walls (offset 0) and slide with
  the wave; every step took two Newton iterations with final residuals at
  most 1e-10.
- Comparison with `linear_sloshing_2d` (unfitted, same reference): at
  `L/h = 64` the unfitted spatial frequency error is +1.5% with the coupled
  level-set transport and +0.13% with the wet extension; the fitted path
  gives 7e-6.
- Before the face field-sampling fix (`0084fdff`) the same setup ran 11%
  slow at `L/h = 16` and the mesh-velocity formulation was unstable (the
  fluid velocity in the mesh kinematic row was sampled at the wrong face
  points). With a displacement-type harmonic operator and a
  `deltat`-scaled penalty the P1 flux defect accumulated into a relaxation
  of rate `h^2 k^2/(2 gamma_N deltat)`: the damping error grew from 4x to
  15x `gamma_ref` when `dt` was divided by 4 at `L/h = 16`
  (diagnostic jobs `46094542`, `46097669`, `46098738`, `46103854`).

Raw output: `/scratch/users/zsexton/free-surface-benchmarks/fitted_ale/sloshing/m5-a6a8a72d/`
(`verify.txt`, `verify.json`).

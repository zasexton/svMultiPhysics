# tank_at_rest

Liquid at rest in a closed tank under gravity, with a flat free surface at a
height that is not aligned with the mesh. It checks that the unfitted
level-set free surface keeps the hydrostatic state, in 2D and 3D, with zero
surface tension (milestone M1 in `Documentation/free_surface_program_tracker.md`).

The state is exactly representable in the discrete spaces, so decision D1
applies an algebraic gate at the solver tolerance scale instead of a
convergence rate (see "Why the gate is 1e-8").

Files:

| File | Role |
|---|---|
| `generate_case.py` | writes `solver.xml`, the mesh with the initial fields, the wall faces and `case.json` for one dimension and level |
| `verify.py` | reads the solver output of one or more runs, computes the metrics, applies `tolerances.json` |
| `tolerances.json` | acceptance criteria and their sources, fixed on 2026-09-29 before the first protocol run |
| `tests/test_free_surface_benchmark_tank_at_rest.py` | checks of the two scripts on synthetic data |

## Physical setup

Nondimensional units: density `rho = 1`, gravity `g = 1` along `-y`, tank
length `L = 1`.

- **Tank.** 2D: `[0, 1] x [0, 0.75]`. 3D: `[0, 1] x [0, 0.75] x [0, 0.5]`.
- **Fill height** `H = 0.52`. `H/h` has fractional part 0.16, 0.32 and 0.64 at
  `1/h = 8, 16, 32`, so the interface cuts its cell row at a different
  fraction at every level and never passes through a vertex. `min |phi|/h`
  over the vertices is 0.16. At every level all dry vertices of the cut cells
  are aggregated (their velocity and pressure extrapolate the cell below), so
  the check covers the aggregation constraints.
- **Phases.** One-phase liquid below the surface (`phi < 0`); the exterior is
  void with `p_ext = 0`. Surface tension is zero.
- **Viscosity** `mu = 5e-4`, the fluid of `linear_sloshing_2d`. The
  hydrostatic state does not depend on it.
- **Walls.** Free slip on the left, right and bottom walls (and front and back
  in 3D): the wall-normal velocity component is set to zero strongly
  (`Dir`, `Value 0`, `Effective_direction` selecting the normal component) and
  the tangential traction is zero naturally. The top wall is dry and carries no
  condition.
- **Initial state** (sampled on every vertex): `phi = y - H`, `u = 0`,
  `p = rho g (H - y)`. Dry vertices of cut cells carry the signed continuation
  of the hydrostatic profile, so the P1 pressure is exactly `rho g (H - y)` on
  every retained cell.

**Reference solution.** `u = 0`, `p = rho g (H - y)` in the liquid, a flat
interface at `y = H`, and the liquid volume `L H` (2D) or `L W H` (3D).

## Why the state is an exact discrete solution

`phi = y - H` and `p = rho g (H - y)` are linear, so the P1 interpolants are
exact, and `u = 0` is exact. On this state:

- the strong momentum residual `rho (du/dt + u.grad u) + grad p - div(2 mu eps(u)) - rho f`
  is zero pointwise, so the Galerkin terms (exact quadrature of polynomial
  integrands on the cut cells), the VMS/PSPG terms and the pressure-gradient
  facet jump (`grad p` is continuous) all vanish;
- the interface term `p_ext n.v` matches the pressure trace, which is zero on
  `y = H`;
- small-cut aggregation extends the root-cell P1 polynomial, which reproduces
  a globally linear field;
- the level-set residual `dphi/dt + u.grad phi` and its SUPG term vanish with
  `u = 0`.

Deviations can therefore only come from roundoff, amplified by the Jacobian,
and from the solver stopping rules. In the protocol runs (Results) the
nonlinear residual stayed below 7e-14 and Newton accepted every step without
an update.

## Why the gate is 1e-8

Each metric is divided by its physical scale (`sqrt(g H)`, `rho g H`, `H`, the
liquid volume) and must stay at or below `1e-8`:

- `1e-8` is the relative tolerance given to every linear solve in these decks,
  the loosest tolerance that could act on a state whose nonlinear residual is
  at roundoff. The direct solver used here meets it with a wide margin.
- Any term that is consistent but not exact on this state (a cut-cell
  quadrature error, a gravity and pressure imbalance, an inexact aggregation
  extension) leaves an `O(h^2)` error, about `1e-3` to `1e-4` on these meshes,
  four or more orders above the gate.
- A convergence rate is meaningless for an exactly representable state (D1).

## Discretization and fixed inputs

Every value is fixed for all levels (principle P1). The numerical constants
are the production values of `static_drop_2d` (tracker section 6.1); none was
chosen for this case.

| Input | Value | Source |
|---|---|---|
| Mesh | 2D: affine `Triangle3`, diagonals alternating with cell parity. 3D: affine `Tetra4`, the six-tetrahedron Kuhn split of each cube. `h = 1/8, 1/16, 1/32` (2D: 63, 221, 825 vertices) and `1/8, 1/16` (3D: 315, 1,989 vertices) | geometry choice |
| Free surface | `UnfittedLevelSet`, `Active_domain=LevelSetNegative`, `Active_domain_method=CutVolume`, `Generated_interface_geometry=LinearCorner`, `Surface_tension=0` | production unfitted path |
| Cut stabilization | pressure-gradient facet penalty 1.0, `Use_cut_metadata_scale=false`, `Small_cut_aggregation=true`, no velocity extension | production defaults |
| Level-set transport | P1, advected by the fluid velocity (`Velocity_source=coupled_field`), SUPG with tau scale 0.5 and transient scale 2.0; no reinitialization, no volume correction | as `static_drop_2d` |
| Time integration | generalized-alpha, `rho_inf = 0.5`; `dt = T1/20`, `T1 = 2 pi / sqrt(g k tanh(k H))`, `k = pi/L` (the lowest sloshing period, 3.68); 5 periods (100 steps), 20 outputs | the run spans several periods of the motion an imbalance would excite |
| Nonlinear solve | relative tolerance 1e-4 per equation, at most 8 Newton iterations; level-set absolute floor 1e-10 | production decks |
| Linear solve | Eigen direct (sparse LU), tolerances 1e-8 / 1e-10 | the meshes are small; a direct solve keeps the linear tolerance out of the error budget |

## Metrics (`verify.py`)

All quantities come from the solver's VTU/PVTU point data (`phi`,
`Velocity`, `Pressure`) at every output. The solver may renumber vertices,
so every reference value is computed from each file's own coordinates.

| Metric | Definition |
|---|---|
| wet support | all vertices of the cells that contain liquid (at least one vertex with `phi_h < 0`), including the dry vertices of cut cells |
| `max_speed_over_velocity_scale` | max over outputs and wet-support vertices of `abs(u_h)`, divided by `sqrt(g H)` |
| `pressure_error_over_pressure_scale` | max over outputs and wet-support vertices of `abs(p_h - rho g (H - y))`, divided by `rho g H` |
| `interface_height_drift_over_fill_height` | max over outputs and over the zero crossings of `phi_h` on mesh edges of `abs(y - H)`, divided by `H` |
| `liquid_volume_relative_drift_max` | max over outputs of `abs(V(t) - V(0))/V(0)`, `V` the exact area or volume of `{phi_h < 0}` (cut triangles are clipped by the linear `phi_h`; cut tetrahedra are split into tetrahedra at the edge crossings), `V(0)` from `mesh/mesh-complete.mesh.vtu` |
| Reported only | the same speed over all vertices; the initial pressure and interface errors; `V(0)` against the exact volume; the histories; from `solver_run.log(.gz)` if present: Newton iterations, outer passes per step, the largest final residual, and the wall time |

`verify.py` exits with status 2 on missing or inconsistent data: no
`case.json`, no output, missing arrays, a mesh that is not pure `Triangle3`
or `Tetra4`, non-finite values, a run that stopped before its end time, or a
`--max-steps` smoke run without `--allow-truncated`.

## Tolerances and their sources

Every criterion applies to every run (each dimension and level).

| Criterion | Limit | Source |
|---|---|---|
| `velocity` | `max_speed_over_velocity_scale <= 1e-8` | D1 algebraic gate for an exactly representable state, at the solver tolerance scale |
| `hydrostatic_pressure` | `pressure_error_over_pressure_scale <= 1e-8` | same |
| `interface_height` | `interface_height_drift_over_fill_height <= 1e-8` | same |
| `volume_drift` | `liquid_volume_relative_drift_max <= 1e-8` | same |

## How to run

Use the Python stack from the benchmark README (numpy; pyvista for
`verify.py`). Put generated cases and output under `$SCRATCH`, and start the
solver through `mpiexec` in a batch job (benchmark README):

```bash
B=tests/cases/fluid/free_surface_benchmarks/tank_at_rest
OUT=$SCRATCH/free-surface-benchmarks/tank_at_rest/$(git rev-parse --short HEAD)
for L in 8 16 32; do python3 $B/generate_case.py --dim 2 --level $L --output-dir $OUT/d2_L$L; done
for L in 8 16; do python3 $B/generate_case.py --dim 3 --level $L --output-dir $OUT/d3_L$L; done
# in each case directory, inside a Slurm job:
#   mpiexec -n 1 --bind-to none /path/to/svmultiphysics solver.xml 2>&1 | gzip -1 > solver_run.log.gz
python3 $B/verify.py $OUT/d2_L8 $OUT/d2_L16 $OUT/d2_L32 $OUT/d3_L8 $OUT/d3_L16 --json $OUT/verify.json
```

For a schema check use `--max-steps 5`; `verify.py` refuses such runs unless
`--allow-truncated` is given.

## Results

**2026-09-30, source `7aac1e29`, solver built at `fef0d02f`, Slurm job
`46023412`.** All five runs pass every criterion.

| dim | 1/h | vertices | `max abs(u)/sqrt(gH)` | pressure error `/rho g H` | interface drift `/H` | volume drift | largest final residual | wall time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 8 | 63 | 3.5e-14 | 0 | 0 | 0 | 2.8e-14 | 7 s |
| 2 | 16 | 221 | 1.0e-13 | 0 | 0 | 0 | 3.0e-14 | 11 s |
| 2 | 32 | 825 | 3.1e-13 | 0 | 0 | 0 | 6.1e-14 | 28 s |
| 3 | 8 | 315 | 3.4e-13 | 5.3e-17 | 0 | 0 | 2.5e-14 | 104 s |
| 3 | 16 | 1,989 | 3.2e-13 | 2.7e-17 | 0 | 0 | 8.9e-15 | 788 s |

- Newton accepted all 100 steps of every run with zero updates and one outer
  pass per step: the residual of the sampled state is at roundoff. The level
  set, the pressure and the liquid volume were therefore unchanged to the last
  bit.
- The velocity is not updated by Newton, but it grows linearly in time by
  about `3e-15` per step (the time-integrator update of roundoff-level
  data). At that rate the gate would be reached after about `3e6` steps.
- The 3D cost is dominated by the per-step geometry work, not by the solve:
  7.9 s per step at `1/h = 16` with no Newton update.

Raw output: `/scratch/users/zsexton/free-surface-benchmarks/tank_at_rest/m1-7aac1e29/`
(`verify.txt`, `verify.json`); scratch is purged, so copy it to group storage
if the result is accepted.

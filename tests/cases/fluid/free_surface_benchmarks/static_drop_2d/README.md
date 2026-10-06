# static_drop_2d

A closed 2D liquid drop at rest, fully enclosed by its free surface. It
measures the Laplace pressure jump and the parasitic (spurious) currents of
the unfitted level-set free surface on the physically relaxed state
(decisions D1 and D3 in `Documentation/free_surface_program_tracker.md`).
The same case is run for the three capillary routes compared in decision D2
(milestone M2): `SurfaceStress`, KAG with a lumped trace mass, and KAG with
the consistent trace mass.

Files:

| File | Role |
|---|---|
| `generate_case.py` | writes `solver.xml`, the mesh with the initial fields, the wall faces and `case.json` for one level |
| `verify.py` | reads the solver output of one or more levels, computes the metrics, applies `tolerances.json` |
| `tolerances.json` | acceptance criteria and their sources, fixed on 2026-09-29 before the first run; time step revised on 2026-09-30 and on 2026-10-05 (D13, see "Time step") |
| `tests/test_free_surface_benchmark_static_drop_2d.py` | checks of the two scripts on synthetic data |

## Physical setup

Nondimensional units: density `rho = 1`, surface tension `gamma = 1`, drop
radius `R = 1`. Pressure is measured in `gamma/R` and time in the capillary
time `sqrt(rho R^3/gamma) = 1`.

- **Geometry.** Circle of radius `R` centred at
  `c = (1.5 + pi/100, 1.5 + e/100) R` in the box `[0, 3R]^2`. Zero gravity.
- **Phases.** One-phase liquid inside the drop (`phi < 0`); the exterior is
  void with prescribed pressure `p_ext = 0`.
- **Viscosity** from the Laplace number `La = rho gamma D / mu^2`, `D = 2R`,
  so `mu = sqrt(2/La)`. The viscous time is `t_mu = rho R^2 / mu`.

  | La | mu (= Ohnesorge number) | t_mu |
  |---:|---:|---:|
  | 12 | 0.4082 | 2.449 |
  | 120 | 0.1291 | 7.746 |

- **Initial state** (the sampled analytic shape, D3; no minimizer):
  `phi = |x - c| - R` at the P1 vertices, `u = 0`, and `p = gamma/R` on every
  vertex. The constant preload on the whole background support follows the
  July closed-drop runner: a sign-masked preload would put an `O(dp/h)`
  pressure gradient into every retained cut cell. Inactive pressure is pinned
  by the solver.
- **Walls.** No-slip on all four sides. They stay dry: the drop is at least
  3.7 cells from every wall at `R/h = 8` and further at finer levels.

**Reference solution.** The static equilibrium is `u = 0`, a circular
interface, and a constant liquid pressure `p_in = p_ext + gamma/R`, since the
curvature of a circle in 2D is `1/R`. The liquid area is conserved. The
sampled P1 circle has an area `A_h(0)` slightly below `pi R^2` (by 0.24% at
`R/h = 8`, second order in `h`), and incompressible flow keeps it. The
equilibrium radius of the discrete drop is therefore `R_eff = sqrt(A/pi)`, and
the pressure reference is `gamma/R_eff`. `verify.py` also reports the error
against the nominal `gamma/R`.

## Discretization and fixed inputs

Every value below is fixed for all levels and all capillary forms
(principle P1). Numerical constants are the existing production values of
the free-surface decks (D18/D38, sloshing, the September capillary decks);
none was chosen for this case.

| Input | Value | Source |
|---|---|---|
| Mesh | affine `Triangle3`, `h = R/level`, diagonals alternating with cell parity, 24/48/96/192 cells per side (625/2401/9409/37249 vertices) | geometry choice; the alternating diagonal has no preferred direction |
| Centre offset | `(pi, e)/100 R` from the box centre | transcendental, so the circle cannot pass exactly through a vertex of any of the nested dyadic grids. The minimum of `abs(phi)/h` over the vertices is 4.8e-3, 6.8e-3, 3.1e-3 and 1.8e-3 at `R/h` = 8, 16, 32, 64. `generate_case.py` prints it and warns below 1e-6. |
| Free surface | `UnfittedLevelSet`, `Active_domain=LevelSetNegative`, `Active_domain_method=CutVolume`, `Generated_interface_geometry=LinearCorner`, `Geometry_tangent_policy=RefreshedFrozenQuadrature`, `Interface_quadrature_order=2`, `Use_level_set_curvature=false` | required by unfitted surface tension (`NavierStokesFreeSurface.md`); order 2 is required by KAG and used for all forms so that only the force differs |
| Cut stabilization | pressure-gradient facet penalty 1.0, `Use_cut_metadata_scale=false`, `Small_cut_aggregation=true`, no velocity extension | production defaults (tracker section 6.1) |
| `surface_stress` | `Surface_tension_form=SurfaceStress` | D2 candidate (a) |
| `kag_lumped` | `Surface_tension_form=KinematicAreaGradientTraction`, projected P1 field `kappa` (`Enable_curvature_projection`, `Curvature_field_name`), `Curvature_projection_recovery_mode=KinematicAreaGradient`, filter coefficient 0, cadence 1, `Curvature_projection_kinematic_area_gradient_mass=Lumped` | D2 candidate (b). With `Lumped` the filter coefficient defaults to 0, so the explicit 0 is optional. |
| `kag_consistent` | as `kag_lumped` without the mass key (default `Consistent`) | D2 candidate (c), reference only |
| Level-set transport | P1, SUPG with the production constants (tau scale 0.5, transient scale 2.0); no reinitialization, no volume correction, no discontinuity capturing, no bound limiter; advected by the harmonic PDE extension of the fluid velocity with monolithic coupling (`Advection_velocity_extension_method=pde_harmonic`, `Advection_velocity_extension_coupling=monolithic`; generator value `pde_harmonic_monolithic`) | decision D9; the volume drift is a measured quantity, so it is not corrected |
| Level-set kinematic reconciliation | on (`Enable_kinematic_reconciliation=true`; `--kinematic-reconciliation off` reproduces the earlier decks): after every accepted step the transported `phi` is corrected locally so that the step's change of the sharp P1 area equals the interface flux of the transport velocity (`FE/LevelSet/LevelSetKinematicReconciliation.h`) | parameter-free and local, not a volume target or global shift; the area drift stays a measured quantity (the flux of a discretely divergence-free velocity) |
| Time integration | generalized-alpha, `rho_inf = 0.5` | all free-surface decks |
| Nonlinear solve | relative tolerance 1e-4 per equation, at most 8 Newton iterations; the level-set block also has the absolute gate 1e-10 taken from its linear-solver block | production decks |
| Linear solve | FSILS GMRES with the RCS preconditioner, 100 iterations, Krylov dimension 50, tolerances 1e-8 / 1e-10 | September capillary decks (monolithic phi/u/p system) |

**Level-set advection velocity (D9).** The fluid velocity is zero on the
inactive vertices beyond the cut cells, so advecting `phi` with it freezes the
level set one row outside the interface (`linear_sloshing_2d` failed its
first protocol for this reason).  The level set is therefore advected with the
harmonic PDE extension of the fluid velocity: `w = u` on the wet vertices and
on every vertex of the retained interface cells, and `(grad w, grad v) = 0`
over the dry cells elsewhere, with the constrained components of the no-slip
walls set to zero on dry wall vertices (the drop never reaches them).  It is
parameter-free and writes no per-step files; the algebraic wet extension of
the SPHERIC Test 05 decks writes one JSON map per step (about 120 GB for an
`R/h = 32` run).  With the monolithic coupling the extension rows are part of
the Newton system; the prescribed coupling lags the extension by one outer
pass, which more than doubles the outer passes of this capillary case
(`linear_sloshing_2d` README, "Level-set advection velocity"; results below).
`--level-set-velocity coupled_field` reproduces the earlier transport.

**Time step.** The Brackbill, Kothe and Zemach (1992) capillary limit is
`dt < sqrt(<rho> h^3 / (2 pi gamma))` with `<rho> = (rho_1 + rho_2)/2`.
Equivalently, the capillary wave of wavelength `2h` (`k = pi/h`,
`omega^2 = gamma k^3 / (rho_1 + rho_2)`) moves less than `h/2` per step. A
free surface has no second fluid (`rho_2 = 0`), so

```text
dt <= sqrt(rho h^3 / (4 pi gamma)) = (1/sqrt(2)) * sqrt(rho h^3 / (2 pi gamma)).
```

The factor `1/sqrt(2)` relative to the tracker's form of the limit is
derived from this one-sided density sum; it is not tuned. Viscosity relaxes
the limit (Galusinski and Vigneaux 2008).

**Protocol step (decision D13, 2026-10-05).** The lagged normal-increment
capillary term (`Surface_tension_semi_implicit = NormalIncrement`,
`Physics/Docs/NavierStokesFreeSurface.md`) removes this limit, so every level
uses one fixed physical step: `dt = 0.02` at `La = 12` (618 steps) and
`dt = 0.01` at `La = 120` (3,900 steps), at any `R/h`. The term is on by
default. The output cadence is the nearest whole number of steps per 1/100 of
the run, which ends at the first output at or after 5 viscous times.
Validation: `Documentation/free_surface_semi_implicit_surface_tension_design.md`,
§9.3 and §9.9. Other Laplace numbers have no fixed step yet and keep the
earlier rule below (`dt_rule = protocol_capillary_limit` in `case.json`).

**Half-step check.** Each refinement study is also run at `dt/2`
(`--dt-divisor 2`: twice the steps and the cadence, the same output times).
Every gate must pass at both steps, and the pressure jump at the finest level
may change by at most 0.1% (`time_step_criterion` in `tolerances.json`;
measured: at most 6e-6). `Ca_sp` is reported at each step.

**Earlier protocol (until 2026-10-05).** The step was `m dt_B`, with `dt_B` the
limit above, `m = 2` at `La = 12` and `m = 1` at `La = 120` (step-0
measurement, tracker, 2026-09-30, jobs `46075447` and `46076505`), rounded down
to 100 equal output intervals, with the term off. It scales as `h^(3/2)`:
500/1,400/4,000 steps at `La = 12` and 3,200/8,800/24,900 at `La = 120` for
`R/h = 8/16/32`. Reproduce it with
`--dt-multiple <m> --surface-tension-semi-implicit None`.

**Run length.** `T = 5 t_mu`, the lower end of the 5 to 10 viscous times of
D3. There are 100 VTU snapshots, one every `T/100`.

## Metrics (`verify.py`)

All quantities are computed from the solver's VTU/PVTU point data (`phi`,
`Velocity`, `Pressure`) on the output mesh itself. The solver's own
wet-volume log line is not needed; on the smoke run the two areas agree to
all printed digits.

| Metric | Definition |
|---|---|
| Liquid area `A(t)` | exact area of `{phi_h < 0}` for the P1 interpolant, clipping each cut triangle by the linear `phi_h`. This is the `LinearCorner` cut volume on `Triangle3`. |
| Effective radius | `R_eff = sqrt(A/pi)` at the end of the run |
| Pressure jump | `p_in - p_ext`. `p_in` is the area-weighted mean of the P1 pressure (exact P1 integral) over the triangles lying entirely inside the disc of radius `R_eff/2` about the liquid centroid, which is at least `R/2` away from the interface. |
| `pressure_jump_relative_error` | `abs((p_in - p_ext) - gamma/R_eff) / (gamma/R_eff)` at the end of the run |
| `max_speed(t)` | `max abs(u_h)` over the vertices with `phi_h < 0` at that output. This is the spatial definition of the July runner, except that July used the initial `phi`. |
| `parasitic_capillary_number_final` | `Ca_sp = mu * max_speed / gamma`, taking the largest value over the outputs with `t > 3T/4` |
| `max_speed_growth_ratio` | `max_speed(T)` divided by the largest `max_speed` over the outputs with `T/2 <= t <= 3T/4`. A ratio of at most 1 means the final value lies inside the envelope of the preceding quarter. A plateau passes and sustained growth fails. |
| `liquid_area_relative_drift_max` | `max_t abs(A(t) - A(0)) / A(0)`, with `A(0)` computed from `phi` in `mesh/mesh-complete.mesh.vtu` |
| Reported only | error against `gamma/R`; centroid drift; maximum and RMS radial deviation of the interface points (edge zero crossings) from the circle of radius `R_eff`; `Ca_sp` at `t = T`; `max_speed/gamma`; the full histories |

Observed order is the least-squares slope of `log(error)` against
`log(R/h)` over the listed levels. Pairwise orders are printed as well.

`verify.py` exits with status 2 and a message on missing or inconsistent
data: no `case.json`, no output, missing arrays, non-triangle cells,
non-finite values, or a run that stopped before `T`. It also exits with
status 2 on a `--max-steps` smoke run, unless `--allow-truncated` is given.

## Tolerances and their sources

Each criterion applies to each (capillary form, La, dt divisor) refinement
study, so it must pass at `dt` and at `dt/2`. The time-step criterion (D13)
is printed as a separate gated line, or as not evaluated when only one
divisor is given.

| Criterion | Limit | Where | Source |
|---|---|---|---|
| `pressure_jump` | at most 0.01, observed order at least 1 | 0.01 at `R/h = 32`; order over 8/16/32 | D1 working criterion, tracker M2 |
| `parasitic_capillary_number` | strictly decreasing with refinement; absolute values reported, no absolute limit | decreasing over all levels present (at least 8/16/32) | Decision of 2026-09-29, tracker M2; see below |
| `no_velocity_growth` | growth ratio at most 1 | every level | D1 working criterion, tracker M2 |
| `volume_drift` | at most 1e-4 | every level | D1 working criterion, tracker M2 |
| `time_step` | pressure jump changes by at most 0.001 between `dt` and `dt/2`; `Ca_sp` reported at each step | finest common level of the two studies | D13, 2026-10-05 |

### The July 2026 SurfaceStress level

The July number comes from the review (`git show
fc565279:Documentation/free_surface_level_set_review_20260713.md`, line
238). It was produced by `run_test05_velocity_growth_smoke.py
--high-order-capillary-droplet-equilibrium-smoke` through
`run_fs16_physical_matrix.py`. The quantity is:

```text
speed per surface tension = max over vertices with phi(t=0) < 0 of |u_h| in the last output, / gamma   [SI, 1/(Pa s)]
```

The values were 2.57998e-5, 1.18936e-5 and 2.52693e-5 at n = 8, 16, 32 (July
gate: 1e-5).

The July setup:

- `SurfaceStress`, a circle with `R = 0.3 m` in the unit box;
- `Quad4` n x n cells, so `R/h` = 2.4, 4.8 and 9.6;
- high-order implicit cuts;
- water-like fluid: `rho = 998.2`, `mu = 1.003e-3`, `gamma = 0.5`, hence `La = 2.98e8`;
- pressure preloaded to `gamma/R`;
- **three steps of `dt = 1e-3 s`**, so the number is read at `t = 3 ms`, or
  3.4e-8 viscous times.

This quantity has units of `1/viscosity` and was measured in the inertial
start-up transient, not on a relaxed state. No dimensionally exact
conversion to a relaxed `Ca_sp` exists. The candidate readings are:

| Reading | n = 8 | n = 16 | n = 32 |
|---|---:|---:|---:|
| as recorded, read as `Ca_sp` (tracker) | 2.58e-5 | 1.19e-5 | 2.53e-5 |
| `mu * value`, the actual July capillary number at 3 ms | 2.59e-8 | 1.19e-8 | 2.53e-8 |
| `value * sqrt(gamma rho R)`, speed in units of `sqrt(gamma/(rho R))` | 3.16e-4 | 1.46e-4 | 3.09e-4 |
| start-up force imbalance `rho R^2 u / (gamma t)` | 0.77 | 0.36 | 0.76 |

The `SurfaceStress` smoke run below gives a start-up force imbalance of 0.58
after its first step at `R/h = 8`, the same order as July. This confirms that
the July number measured start-up force imbalance, not a relaxed spurious
current. Reading it as a limit on `Ca_sp` would compare different quantities,
and the dimensionally correct reading (about 2.6e-8) is below anything
reported for methods that are not exactly balanced.

**Decision (2026-09-29, tracker M2):** there is no absolute limit on `Ca_sp`.
The criterion is that `Ca_sp` decreases strictly under refinement, together
with `no_velocity_growth`. The absolute values are reported at every level so
they can be compared with the literature and between capillary forms.

## How to run the refinement study

Use the Python stack from the benchmark README (numpy; pyvista for
`verify.py`). Put generated cases and output under `$SCRATCH`. Launch the
solver through `mpiexec` and submit with `--export=NONE` (benchmark README,
"Launching the solver"); `run_case.sbatch` below is a minimal job script that
does both and compresses the log.

```bash
B=tests/cases/fluid/free_surface_benchmarks/static_drop_2d
OUT=$SCRATCH/free-surface-benchmarks/static_drop_2d/$(git rev-parse --short HEAD)
SVMP=/path/to/build/bin/svmultiphysics
JOB=/path/to/run_case.sbatch   # sets PATH/LD_LIBRARY_PATH, then runs
                               # timeout -k 30 <s> mpiexec -n 1 --bind-to none $SVMP solver.xml
for form in surface_stress kag_lumped kag_consistent; do for L in 8 16 32; do
  d=$OUT/La12/$form/L$L
  python3 $B/generate_case.py --level $L --capillary-form $form --laplace-number 12 --output-dir $d
  sbatch --export=NONE --time=<see table> --job-name=drop_${form}_L$L \
         --output=$d/slurm-%j.out $JOB $d $SVMP <timeout_s>
  # half-step check: the same with --dt-divisor 2 into $OUT/La12/$form/L${L}_dt2
done; done
# after the jobs end (both steps in one call; verify.py groups by dt divisor):
python3 $B/verify.py $OUT/La12/surface_stress/L{8,16,32}{,_dt2} --json $OUT/La12_surface_stress.json
```

The solver log is about 0.17 MB per step, so compress it. VTU snapshots are
0.13 MB at `R/h = 8` and scale with the vertex count. For a quick schema
check use `--max-steps 5`; `verify.py` refuses such runs unless
`--allow-truncated` is given.

**Time-step options.** `generate_case.py` sets the step as follows:

- default: the D13 step of the Laplace number (see "Time step").
- `--dt-divisor 2` divides the step by 2 with unchanged output times.
- `--dt-multiple m` uses `m dt_B`, rounded down to the 100 output intervals
  (the earlier rule).
- `--dt <step>` keeps this exact step, as the protocol step is kept; nested
  steps (0.04, 0.02, 0.01 at La = 12) share their output times.
- `--surface-tension-semi-implicit None` removes the D13 term.

`case.json` records `dt_rule` (`protocol_fixed`, `protocol_capillary_limit`,
`multiple_override` or `fixed`), `dt_base`, `dt_divisor` and
`surface_tension_semi_implicit`.

## Expected cost per level

**D13 protocol** (design note §9.3, 2026-10-03; `surface_stress`, PDE
transport, term on, 4 MPI ranks at `R/h = 16` and 32, serial at 8, on one
loaded 24-core node). Outer passes stay at about 3 per step (at most 5), so
the time follows the step count:

| La | dt | steps (any R/h) | time at R/h = 32 |
|---:|---:|---:|---:|
| 12 | 0.02 (protocol) | 618 | 4,800 s |
| 12 | 0.01 (half step) | 1,236 | 8,900 s |
| 120 | 0.01 (protocol) | 3,900 | 28,800 s |
| 120 | 0.005 (half step) | 7,800 | about 58,000 s (estimate) |

At `R/h = 32` this is 6 times fewer outer passes than the earlier protocol at
`La = 12` and 8 times fewer than the projected run at `La = 120`.

**Earlier protocol**, measured on 2026-09-30 (Slurm job `46089180`, source `b4b376a0`, after the
solver speed-ups of `67b4395a`): `surface_stress`, La = 12, `2 dt_B`, serial,
8 runs sharing one node.

| R/h | vertices | steps | transport | s/step | outer passes/step (mean, max) | time |
|---:|---:|---:|---|---:|---|---:|
| 8 | 625 | 500 | `coupled_field` | 0.55 | 3.6, 6 | 4.6 min |
| 8 | 625 | 500 | `pde_harmonic_monolithic` (protocol) | 0.63 | 3.6, 6 | 5.3 min |
| 8 | 625 | 500 | `pde_harmonic_prescribed` | 0.62 | 4.5, 11 | 5.2 min |
| 16 | 2,401 | 1,400 | `coupled_field` | 1.98 | 3.2, 6 | 46 min |
| 16 | 2,401 | 1,400 | `pde_harmonic_monolithic` (protocol) | 2.34 | 3.2, 6 | 55 min |
| 32 | 9,409 | 3,950 | `pde_harmonic_monolithic` | about 9 (estimate) | – | about 10 h |

The PDE extension adds about 15–18% per step.  With the prescribed coupling
the start-up steps need up to 11 outer passes against the 12-pass cap, which
is why the protocol uses the monolithic coupling.  Timings vary by node
generation (up to about 1.8x).  `kag_consistent` costs about 50% more per
step.  `La` of 1,200 and above, and `R/h = 64`, remain deferred.

Memory is small: 0.26 GB RSS at `R/h = 8`.

## Level-set transport check (2026-09-30)

A short comparison at `R/h = 8` and 16 (La = 12, `surface_stress`, `2 dt_B`,
5 viscous times; job `46089180`) confirms that the PDE extension does not
change the relaxed drop:

| R/h | transport | pressure-jump error | `Ca_sp` (final quarter) | growth ratio | max `dA/A` |
|---:|---|---:|---:|---:|---:|
| 8 | `coupled_field` | 6.73e-4 | 2.52e-4 | 0.746 | 1.2e-5 |
| 8 | `pde_harmonic_monolithic` | 6.76e-4 | 2.49e-4 | 0.741 | 1.1e-5 |
| 8 | `pde_normal_monolithic` | 6.76e-4 | 2.49e-4 | 0.742 | 1.1e-5 |
| 8 | `pde_harmonic_prescribed` | 6.76e-4 | 2.49e-4 | 0.741 | 1.1e-5 |
| 16 | `coupled_field` | 1.49e-4 | 1.32e-4 | 0.705 | 3.2e-7 |
| 16 | `pde_harmonic_monolithic` | 1.49e-4 | 1.32e-4 | 0.705 | 3.9e-7 |
| 16 | `pde_normal_monolithic` | 1.49e-4 | 1.32e-4 | 0.705 | 6.7e-7 |

All runs completed the 5 viscous times.  Raw output:
`/scratch/users/zsexton/free-surface-benchmarks/pde-extension/campaign-b4b376a0/static_drop/`.

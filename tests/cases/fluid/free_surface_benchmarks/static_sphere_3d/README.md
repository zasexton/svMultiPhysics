# static_sphere_3d

A closed 3D liquid drop at rest, fully enclosed by its free surface: the 3D
analogue of `static_drop_2d`. It measures the Laplace pressure jump and the
parasitic currents of the unfitted level-set free surface on the physically
relaxed state (decisions D1 and D3 in
`Documentation/free_surface_program_tracker.md`), for the capillary routes
compared in decision D2 (milestone M2): `SurfaceStress`, KAG with a lumped
trace mass, and KAG with the consistent trace mass. The tracker schedules the
3D sphere at R/h = 8, 16, 32 after the 2D selection (M2, "Selection").

Files:

| File | Role |
|---|---|
| `generate_case.py` | writes `solver.xml`, the Tetra4 mesh with the initial fields, the six wall faces and `case.json` for one level |
| `verify.py` | reads the solver output of one or more levels, computes the metrics, applies `tolerances.json` |
| `tolerances.json` | acceptance criteria and their sources, fixed on 2026-09-30 before the first run; they mirror `static_drop_2d` |
| `tests/test_free_surface_benchmark_static_sphere_3d.py` | checks of the two scripts on synthetic data |

**Status (2026-10-05): runs.** The two step-0 blockers recorded under "Smoke
run" below (2026-09-30) are fixed. The functional consistency checks now use
a rounding bound derived from the number of summed terms, instead of a fixed
512 ulp: at R/h = 8 the rule-wise and point-wise volumes (418,870 points)
differ by 1.89e-12, 4x the old bound and 0.5% of the derived one. Vertex
crossings are accepted within a step. A `surface_stress` run at R/h = 8
(`--max-steps 2`, PDE transport) accepted both steps with 5 outer passes and
6-7 Newton iterations each, at about 21-23 min per step serial; the volume
drift was -3.0e-6 after step 2. A full refinement study is not yet scheduled
(about 500 steps at R/h = 8, plus the 3D gating and memory decisions for
R/h = 16).

## Physical setup

Nondimensional units: density `rho = 1`, surface tension `gamma = 1`, drop
radius `R = 1`. Pressure is measured in `gamma/R` and time in the capillary
time `sqrt(rho R^3/gamma) = 1`.

- **Geometry.** Sphere of radius `R` centred at
  `c = (1.5 + pi/100, 1.5 + e/100, 1.5 + sqrt(3)/100) R` in the cube `[0, 3R]^3`. Zero gravity.
- **Phases.** One-phase liquid inside the drop (`phi < 0`); the exterior is
  void with prescribed pressure `p_ext = 0`.
- **Viscosity** from the Laplace number `La = rho gamma D / mu^2`, `D = 2R`,
  so `mu = sqrt(2/La)` and the viscous time is `t_mu = rho R^2 / mu`, as in 2D:

  | La | mu (= Ohnesorge number) | t_mu |
  |---:|---:|---:|
  | 12 | 0.4082 | 2.449 |
  | 120 | 0.1291 | 7.746 |

- **Initial state** (the sampled analytic shape, D3; no minimizer):
  `phi = |x - c| - R` at the P1 vertices, `u = 0`, and `p = 2 gamma/R` on
  every vertex. As in 2D, the constant preload covers the whole background
  support: a sign-masked preload would put an `O(dp/h)` pressure gradient into
  every retained cut cell. Inactive pressure is pinned by the solver.
- **Walls.** No-slip on all six faces. They stay dry: the drop is at least
  3.7 cells from every wall at `R/h = 8` and further at finer levels.

**Reference solution.** The static equilibrium is `u = 0`, a spherical
interface, and a constant liquid pressure `p_in = p_ext + 2 gamma/R`: the
mean-curvature sum of a sphere is `2/R`. The liquid volume is conserved. The
sampled P1 sphere has a volume `V_h(0)` below `4 pi R^3/3`, by 0.76%, 0.20%
and 0.049% at `R/h` = 8, 16, 32 (second order), and incompressible flow keeps
it. The equilibrium radius of the discrete drop is therefore
`R_eff = (3V/(4 pi))^(1/3)`, and the pressure reference is `2 gamma/R_eff`.
`verify.py` also reports the error against the nominal `2 gamma/R`.

## Discretization and fixed inputs

Every value is fixed for all levels and all capillary forms (principle P1).
Apart from the mesh, every input is that of `static_drop_2d` (see its
README for the sources); none was chosen for this case.

| Input | Value | Source |
|---|---|---|
| Mesh | affine `Tetra4`, `h = R/level`; each cube of a 24/48/96 cells-per-side grid split into the six Kuhn tetrahedra around a main diagonal, reflected along every axis whose cube index is odd | the reflected split is conforming and symmetric under reflection about the grid planes, the 3D analogue of the alternating 2D diagonal; a plain Kuhn split would prefer the (1,1,1) direction |
| Mesh size | R/h = 8: 15,625 vertices, 82,944 cells; 16: 117,649 and 663,552; 32: 912,673 and 5,308,416 | |
| Centre offset | `(pi, e, sqrt(3))/100 R` from the box centre | irrational, so the sphere passes through no vertex of the nested dyadic grids. Of the simple triples tried (`(pi, e, sqrt 2)`, `(pi, e, sqrt 3)`, `(pi, e, ln 2)`, `(pi, e, Euler gamma)`, `(sqrt 2, sqrt 3, sqrt 5)` and permutations), this one has the largest worst-level `min abs(phi)/h`: 1.2e-3, 2.9e-4 and 6.5e-5 at `R/h` = 8, 16, 32 (`generate_case.py` prints it and warns below 1e-6). In 3D many more vertices lie near the surface than in 2D (1.8e-3 to 6.8e-3 there), so smaller values are expected for any centre; small cut fragments are handled by the small-cut aggregation. |
| Free surface | `UnfittedLevelSet`, `Active_domain=LevelSetNegative`, `CutVolume`, `LinearCorner`, `Geometry_tangent_policy=RefreshedFrozenQuadrature`, `Interface_quadrature_order=2`, `Use_level_set_curvature=false` | as `static_drop_2d`; order 2 is required by KAG and used for all forms |
| Cut stabilization | pressure-gradient facet penalty 1.0, `Use_cut_metadata_scale=false`, `Small_cut_aggregation=true`, no velocity extension | production defaults (tracker section 6.1) |
| Capillary forms | `surface_stress`, `kag_lumped`, `kag_consistent`: same keys as `static_drop_2d` | D2 candidates (a), (b), (c) |
| Level-set transport | `--transport`, see below; P1 SUPG (tau scale 0.5, transient scale 2.0), no reinitialization, no volume correction | D9; the volume drift is a measured quantity, so it is not corrected |
| Time integration | generalized-alpha, `rho_inf = 0.5` | all free-surface decks |
| Nonlinear solve | relative tolerance 1e-4 per equation, at most 8 Newton iterations; level-set absolute gate 1e-10 | production decks |
| Linear solve | FSILS GMRES with the RCS preconditioner, 100 iterations, Krylov dimension 50, tolerances 1e-8 / 1e-10 | as `static_drop_2d` |

**Level-set transport (decision D9).** `generate_case.py --transport`
offers the same choices as `capillary_wave_2d`:

| `--transport` | Solver input | Status |
|---|---|---|
| `pde_extension` (default) | `Velocity_source=prescribed_data`, `Velocity_field_name=LevelSetAdvectionVelocity`, `Source_velocity_field_name=Velocity`, `Advection_velocity_extension_method=pde_harmonic`, `Advection_velocity_extension_coupling=monolithic` | the D9 protocol transport: the parameter-free harmonic PDE extension of the fluid velocity, as in `static_drop_2d`, `linear_sloshing_2d`, `capillary_wave_2d` and `sessile_drop_2d`. It writes no per-step files. The dry-region solve is replicated on every rank, which may need a distributed solve for large 3D meshes. |
| `wet_extension` | `Velocity_source=prescribed_data`, `Use_wet_extension_advection_velocity=true`, `Advection_velocity_extension_method=wall_compatible_normal` | the algebraic wet extension, the D9 comparison baseline |
| `coupled` | `Velocity_source=coupled_field` | the fluid velocity itself; dry vertices then carry zero velocity (D9 retires it) |

`generate_case.py` selects the extension through `PDE_EXTENSION_METHOD` and
`PDE_EXTENSION_COUPLING`. The wet extension writes one JSON map per
accepted step: about 1 MB per step on the 625-vertex 2D drop and 4.7 MB per
step on the 1,989-vertex 3D tank (`tank_at_rest`, 1/h = 16). Scaled by the
vertex count, that is about 37 MB per step and 18 GB per run at `R/h = 8`, so
the map output must become opt-in before any 3D run with that transport. Its
compute cost is small (2.4% of a tank step). Criteria apply
separately to each (capillary form, transport, `La`) study.

**Time step.** As in 2D, the capillary limit of Brackbill, Kothe and Zemach
with the one-sided density sum of a free surface is

```text
dt_B = sqrt(rho h^3 / (4 pi gamma)) = (1/sqrt(2)) * sqrt(rho h^3 / (2 pi gamma)).
```

The time step is `m dt_B` with `m = 2` at `La = 12` and `m = 1` at
`La = 120` (and any other `La`), rounded down so that the run is exactly 100
equal output intervals. The values of `m` come from the 2D step-0
measurement (tracker M2, jobs 46075447 and 46076505): with the default
12-pass cap, the outer geometry loop accepted steps up to `2 dt_B` at
`La = 12` and up to `dt_B` at `La = 120`. **In 3D they are assumed, not
measured.** Check them once, as in 2D, by running the first steps at
`--dt-multiple 1, 2, 4` and comparing the outer passes per step;
`verify.py` reports such runs but does not gate them. The gated metrics are
relaxed steady-state values, so the time step does not enter them (decision
D10); no separate time-step study is needed for the acceptance criteria.

**Run length.** `T = 5 t_mu`, the lower end of the 5 to 10 viscous times of
D3, with 100 VTU snapshots. At `La = 12` this is 500, 1,400 and 4,000 steps
at `R/h` = 8, 16, 32.

## Metrics (`verify.py`)

All quantities are computed from the solver's VTU/PVTU point data (`phi`,
`Velocity`, `Pressure`) on the output mesh itself.

| Metric | Definition |
|---|---|
| Liquid volume `V(t)` | exact volume of `{phi_h < 0}` for the P1 interpolant: each cut tetrahedron is clipped by the linear `phi_h` (corner tetrahedron, tetrahedron minus a corner, or a prism split into three tetrahedra). This is the `LinearCorner` cut volume on `Tetra4`. |
| Effective radius | `R_eff = (3V/(4 pi))^(1/3)` at the end of the run |
| Pressure jump | `p_in - p_ext`. `p_in` is the volume-weighted mean of the P1 pressure (exact P1 integral) over the tetrahedra lying entirely inside the ball of radius `R_eff/2` about the liquid centroid, at least `R/2` away from the interface. |
| `pressure_jump_relative_error` | `abs((p_in - p_ext) - 2 gamma/R_eff) / (2 gamma/R_eff)` at the end of the run |
| `max_speed(t)` | `max abs(u_h)` over the vertices with `phi_h < 0` at that output |
| `parasitic_capillary_number_final` | `Ca_sp = mu * max_speed / gamma`, taking the largest value over the outputs with `t > 3T/4` |
| `max_speed_growth_ratio` | `max_speed(T)` divided by the largest `max_speed` over the outputs with `T/2 <= t <= 3T/4`; at most 1 means no growth |
| `liquid_volume_relative_deviation_max` | `max_t abs(V(t) - V(0)) / V(0)` over the run (decision D11), with `V(0)` from `phi` in `mesh/mesh-complete.mesh.vtu`. The maximum covers every output and, when the solver log (`solver_run.log` or `solver_run.log.gz`) is present, every step of its `Wet volume diagnostic` line, provided that line matches the snapshot volume at every output to 1e-8. A reversible oscillation between two outputs is therefore caught. |
| Reported only | error against `2 gamma/R`; the output-only and log-only volume maxima and their agreement; centroid drift; maximum and RMS radial deviation of the interface points (edge zero crossings) from the sphere of radius `R_eff`; `Ca_sp` at `t = T`; `max_speed/gamma`; the full histories |

On the initial `R/h = 8` state the solver's wet volume (4.156886009149159)
and the exact P1 volume (4.156886009151602) differ by 5.9e-13 relative: the
solver prunes one sliver cut cell. Observed order is the least-squares slope
of `log(error)` against `log(R/h)`; pairwise orders are printed as well.

`verify.py` exits with status 2 on missing or inconsistent data: no
`case.json`, no output, missing arrays, non-tetrahedral cells, non-finite
values, or a run that stopped before `T`. It also exits with status 2 on a
`--max-steps` smoke run unless `--allow-truncated` is given, and when only
diagnostic (`--dt-multiple`) runs are supplied.

## Tolerances and their sources

The criteria are those of `static_drop_2d`, applied to each (capillary form,
transport, La) refinement study:

| Criterion | Limit | Where | Source |
|---|---|---|---|
| `pressure_jump` | at most 0.01, observed order at least 1 | 0.01 at `R/h = 32`; order over 8/16/32 | D1 working criterion, tracker M2 |
| `parasitic_capillary_number` | strictly decreasing with refinement; absolute values reported, no absolute limit | all levels present (at least 8/16/32) | decision of 2026-09-29, tracker M2 |
| `no_velocity_growth` | growth ratio at most 1 | every level | D1 working criterion, tracker M2 |
| `volume_drift` | at most 1e-4 as the maximum deviation over the run | every level | D1 working criterion, tracker M2; D11 |

The pressure-jump and parasitic-current criteria need the `R/h = 32` run. If
that level stays unaffordable (next section), they cannot be evaluated as
written; a 3D-specific criterion (for example the error at `R/h = 16` and the
order over 8/16) would need a decision recorded in the tracker.

## How to run the refinement study

Use the Python stack from the benchmark README (numpy; pyvista for
`verify.py`), put cases and output under `$SCRATCH`, launch the solver
through `mpiexec` and submit with `--export=NONE` (benchmark README,
"Launching the solver"):

```bash
B=tests/cases/fluid/free_surface_benchmarks/static_sphere_3d
OUT=$SCRATCH/free-surface-benchmarks/static_sphere_3d/$(git rev-parse --short HEAD)
TRANSPORT=pde_extension        # protocol transport (D9)
for form in surface_stress kag_lumped kag_consistent; do for L in 8 16 32; do
  d=$OUT/La12/$TRANSPORT/$form/L$L
  python3 $B/generate_case.py --level $L --capillary-form $form --laplace-number 12 \
      --transport $TRANSPORT --output-dir $d
  sbatch --export=NONE --time=<see below> --mem=<see below> --job-name=sphere_${form}_L$L \
         --output=$d/slurm-%j.out run_case.sbatch $d $SVMP <timeout_s>
done; done
python3 $B/verify.py $OUT/La12/$TRANSPORT/surface_stress/L{8,16,32} --json $OUT/surface_stress.json
```

`run_case.sbatch` is the static-drop job script (it runs
`timeout -k 30 <s> mpiexec -n 1 --bind-to none $SVMP solver.xml` and
compresses the log). For a schema check use `--max-steps 4`; `verify.py`
refuses such runs unless `--allow-truncated` is given.

## Smoke run (2026-09-30)

Job 46097410 on `sh02-10n28` (SKX), solver
`/scratch/users/zsexton/svmp-bin/svmultiphysics-45bc5b09`, `R/h = 8`,
`surface_stress`, `La = 12`, `--transport wet_extension`, `--max-steps 4`;
a second run used `--transport coupled`.

- The input was read and the whole setup completed: mesh and six faces,
  initial cut context (5,489 cut, 10,290 full wet and 67,165 full dry cells),
  constraints and the initial wet volume.
- Both runs then stopped at step 0 in
  `FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals`
  (`FESystem.cpp`, "active-volume energy liquid measure is inconsistent").
  That check compares the liquid volume summed over the retained volume rules
  with the same volume summed over their quadrature weights, with an absolute
  tolerance of `512 eps max(1, V)`, about 1e-13 relative. The two sums differ
  only by rounding, which grows with the number of summed terms: about 16,000
  rules and 0.3 million quadrature points at `R/h = 8`. Several other checks
  in the same function compare liquid measures in the same way.
- The check runs for `SurfaceStress`, KAG, and the wall-aware contact laws
  (`shouldDeclareFreeSurfaceDiscreteFunctional`), so every D2 candidate is
  affected in 3D. 2D and the flat 3D tank pass because their sums have few
  terms or are exact.
- The fix belongs in the solver: a tolerance that scales with the number of
  terms and the sum of their magnitudes, or compensated summation in both
  accumulations. It does not change the solution.
- With the functional record out of the way (the profiling proxy of the
  next section), step 0 still stops, on a second issue: the start-up
  transient (max speed 0.6 after the first pass, so a displacement of about
  0.1 h per step) moves the interface across the vertices nearest to it
  (`min abs(phi)/h = 1.2e-3`). The step is rejected with `CutTopologyChanged`,
  and the fixed-step time loop then aborts ("external-state discontinuity
  requires an adaptive step controller", `TimeLoop.cpp`). In 3D there are
  always vertices within 0.1 h of the sphere, so no choice of centre avoids
  this; the runs depend on the vertex-crossing work of milestone M2.
- No output was written, so `verify.py` could not be run on sphere output.
  Its reader was checked on the solver's 3D Tetra4 output of `tank_at_rest`
  (volume 0.26000000000000006 against the exact 0.26; the solver's log
  gives 0.25999999999998213).

## Expected cost per level

Measured on 2026-09-30 with the same binary (SKX nodes, serial; profiling
data under `$SCRATCH/free-surface-benchmarks/profiling-3d/`). Because of the
check above, the sphere's time steps were measured with a profiling proxy
(`GeneratedCurvatureTraction` fed with the lumped KAG curvature), which runs
the same geometry, constraint, assembly and solver path without the
functional record.

| R/h | vertices / cells | steps at La = 12 | measured with this binary | estimate |
|---:|---|---:|---|---|
| 8 | 15,625 / 82,944 | 500 | setup 3.5 min (initial cut context 150 s, aggregation constraints 33 s). Per outer pass: Newton 69 s (3 iterations; 13.3 s per Jacobian assembly, 9.5 s of it in cut volumes; 238 GMRES iterations, 5 s), cut-context rebuild 120 to 140 s, constraint rebuild 30 to 33 s; the proxies also rebuilt the restored state after each curvature projection. KAG forms add two curvature projections of about 6 to 7 min each and a second rebuild per pass. Peak RSS 6.1 GB (two live 1.56 GB geometry snapshots). | `SurfaceStress`: about 245 s per pass and, assuming the 2D count of 4 to 6 outer passes per step (not measured in 3D), 16 to 25 min per step and about 6 to 9 days per run; KAG several times that |
| 16 | 117,649 / 663,552 | 1,400 | not run | three quadratic searches in the cut rebuild (below) alone would take hours per rebuild; the two live snapshots need about 25 GB. Not runnable. |
| 32 | 912,673 / 5,308,416 | 4,000 | not run | not runnable on one node |

The 3D cost is dominated by the cut-context rebuild and constraints, not by
the linear solve. At `R/h = 8`, three searches that scan all cells or all
cut regions once per cell, rule or boundary face account for about 75% of a
rebuild (`MeshAccess::globalEntityIdsAvailable`, the per-face scans in
`buildGeneratedActiveBoundaryDomain`, and the rule-to-region lookup in
`buildFreeSurfaceGeometrySnapshot`). Fixing them does not change results and
makes the rebuild scale linearly. For the KAG forms, about 60% of a
projection is a finite-difference cross-check of the analytic interface-area
gradient (24 extra strict cuts per cut cell, 131,760 per projection) whose
result is only reported as a diagnostic, and about 30% is a duplicate search
over all 162,600 supplemental curvature samples for every new sample. With these changes alone (none alters results), a pass at
`R/h = 8` would still cost about two minutes, so the run would take two to
four days serially, and `R/h = 16` (8 times the cells, 2.8 times the steps)
months. The 3D study therefore also needs a working MPI run (4 ranks fail
today in 2D) and the larger time step or fewer outer passes being considered
for 2D. `R/h = 32` is not affordable in the foreseeable setup.

For comparison, the flat 3D `tank_at_rest` (one outer pass, no Newton
update) takes 1.1 s per step at 1/h = 8 and 9.8 s at 1/h = 16. About 4 s of
the latter is a cut-context rebuild and a second constraint rebuild that
happen every step although the level set does not change, because the
system's DOF-layout revision changes every step.

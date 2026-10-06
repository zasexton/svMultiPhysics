# sessile_drop_2d

A 2D liquid cap on a flat wall relaxes from a non-equilibrium shape to the
circular cap with the Young angle. The benchmark measures the contact angle
at both contact points, the base radius and the apex height of the relaxed
drop against the equilibrium cap of the same area. It is the M4 wetting
benchmark for the single contact-angle mechanism of decision D4 in
`Documentation/free_surface_program_tracker.md`:

- the angle is imposed only by the variational Young term in the momentum
  equation (the `SurfaceStress` line term `-gamma cos(theta_e) v.m`, or the
  wall-area gradient inside `kappa_h` for KAG);
- Navier slip on the wetted wall, with the slip length as a physical input;
- a strong no-penetration condition on the contact wall;
- angle-preserving (scale-only) level-set wall maintenance. Contact cells
  are only rescaled, never reset to the target angle
  (`Physics/Docs/NavierStokesFreeSurface.md`, "Level-set wall maintenance").
  The protocol runs without reinitialization, like `static_drop_2d`;
  `--reinitialization` adds the production maintenance.

The same case serves the M4 part of the capillary-route comparison of
decision D2, with the three capillary forms of `static_drop_2d`.

Files:

| File | Role |
|---|---|
| `generate_case.py` | writes `solver.xml`, the mesh with the initial fields, the wall faces and `case.json` for one level and one angle |
| `verify.py` | reads the solver output of one or more levels, computes the metrics, applies `tolerances.json` |
| `tolerances.json` | acceptance criteria and their sources, fixed on 2026-09-30 before the first protocol run |
| `tests/test_free_surface_benchmark_sessile_drop_2d.py` | checks of the two scripts on synthetic data |

## Physical setup

Nondimensional units: density `rho = 1`, surface tension `gamma = 1`, and
`R = 1`, the radius of the equilibrium cap. Zero gravity; the exterior is
void with prescribed pressure `p_ext = 0`. The liquid is `phi < 0`.

- **Contact wall.** The bottom wall `y = 0` (`wall_bottom`). Its normal out
  of the liquid into the solid is `(0, -1, 0)`. Angles are measured through
  the liquid.
- **Equilibrium.** A cap of radius `R` with angle `theta_e` has its centre at
  `y = -R cos(theta_e)`, area `A = R^2 (theta_e - sin(theta_e) cos(theta_e))`,
  base half-width `R sin(theta_e)` and apex height `R (1 - cos(theta_e))`.
- **Initial state** (sampled analytic shape, D3; no minimizer). A circular cap
  with angle `theta_0 = theta_e +- 30 deg` and the same area, so that its
  radius is `R_0 = sqrt(A / (theta_0 - sin(theta_0) cos(theta_0)))`:
  `phi = |x - c_0| - R_0` at the P1 vertices, `u = 0`, and `p = gamma/R_0`
  (the Laplace pressure of the initial cap) on every vertex, as in
  `static_drop_2d`. Two cases advance and one recedes:

  | `theta_e` | `theta_0` | area | `R_0` | initial base, apex | equilibrium base, apex | box |
  |---:|---:|---:|---:|---|---|---|
  | 60 | 90 | 0.6142 | 0.6253 | 0.6253, 0.6253 | 0.8660, 0.5000 | `[-1.1875, 1.1875] x [0, 0.9375]` |
  | 90 | 120 | 1.5708 | 0.7884 | 0.6827, 1.1825 | 1.0000, 1.0000 | `[-1.3125, 1.3125] x [0, 1.4375]` |
  | 120 | 90 | 2.5274 | 1.2685 | 1.2685, 1.2685 | 0.8660, 1.5000 | `[-1.5625, 1.5625] x [0, 1.75]` |

- **Viscosity.** Laplace number `La = rho gamma (2R) / mu^2 = 12`, the primary
  value of `static_drop_2d`, so `mu = 0.4082` (Ohnesorge number 0.41). The
  viscous time is `t_mu = rho R^2 / mu = 2.449`, the visco-capillary time
  `mu R / gamma = 0.408`, and the capillary time `sqrt(rho R^3 / gamma) = 1`.
- **Slip length** `l_s = R/8`, a physical input (P1). It is resolved at every
  level: `l_s/h = 2, 4, 8` at `R/h = 16, 32, 64`, the ratios of the planned
  Ren–E study. The equilibrium does not depend on `l_s`; the relaxation
  time does.

**Reference solution.** The static equilibrium is `u = 0`, a circular cap with
angle `theta_e`, and a constant liquid pressure `p_ext + gamma/R_e`. The
liquid area is conserved. The sampled P1 initial cap has an area `A_h(0)`
that differs from the nominal `A` at second order in `h`, and incompressible
flow keeps `A_h(0)`. The reference cap is therefore the cap with angle
`theta_e` and area `A_h(0)`:
`R_ref = sqrt(A_h(0) / (theta_e - sin(theta_e) cos(theta_e)))`, base half-width
`R_ref sin(theta_e)`, apex height `R_ref (1 - cos(theta_e))`. `verify.py`
also reports the errors against the nominal cap (`R = 1`).

## D4 configuration and fixed inputs

Every value below is fixed for all levels, angles and capillary forms
(principle P1). Numerical constants are the existing production values of the
free-surface decks; none was chosen for this case.

| Input | Value | Source |
|---|---|---|
| Mesh | affine `Triangle3`, `h = R/level`, diagonals alternating with cell parity | as `static_drop_2d` |
| Box | the initial and the equilibrium cap plus a dry margin of `R/4` (at least 4 cells at `R/h = 16`), rounded up to a multiple of `R/16` so that the grids of all levels are nested | geometry choice; the exterior is void, so only the margin matters |
| Centre offset | `x_c = (pi/100) R` | transcendental, so no contact point or circle coincides with a vertex. The minimum of `abs(phi)/h` over the vertices is between 2.1e-3 and 1.7e-2 over all cases; `generate_case.py` prints it and the contact-point distance to the nearest wall vertex, and warns below 1e-6 |
| Free surface | `UnfittedLevelSet`, `Active_domain=LevelSetNegative`, `Active_domain_method=CutVolume`, `Active_domain_smoothing_width=0`, `Generated_interface_geometry=LinearCorner`, `Geometry_tangent_policy=RefreshedFrozenQuadrature`, `Interface_quadrature_order=2`, `Use_level_set_curvature=false` | required by unfitted surface tension and by the sharp wetted-wall operator (`NavierStokesFreeSurface.md`) |
| Contact line | `Contact_line_model=PrescribedAngle`, `Contact_line_wall_face=wall_bottom`, `Contact_line_wall_normal=0 -1 0`, `Contact_angle_degrees=theta_e` | D4 |
| Wall slip | `Wall_slip_model=Navier`, `Wall_slip_length=R/8` on the wetted part of `wall_bottom` | D4 |
| Velocity walls | `wall_bottom`: `Dir`, value 0, `Effective_direction 0 1`, a strong zero normal velocity only; its tangential velocity is governed by the Navier term. The other walls stay dry and are no-slip. | D4 strong no-penetration; the prescribed-angle slip validation requires exactly this wall condition |
| Cut stabilization | pressure-gradient facet penalty 1.0, `Use_cut_metadata_scale=false`, `Small_cut_aggregation=true`, no velocity extension | production defaults, as `static_drop_2d` |
| `surface_stress` (default) | `Surface_tension_form=SurfaceStress` | D2 candidate (a) |
| `kag_lumped`, `kag_consistent` | as in `static_drop_2d` | D2 candidates (b) and (c) |
| Level-set transport | P1, advected by the harmonic PDE velocity extension of tracker D9 with monolithic coupling (`--transport pde_extension`, the default: `Advection_velocity_extension_method=pde_harmonic`, `Advection_velocity_extension_coupling=monolithic`), SUPG with the production constants (tau scale 0.5, transient scale 2.0); no volume correction, no discontinuity capturing, no bound limiter | decision of 2026-10-05 (tracker M4), as in `static_drop_2d`, `linear_sloshing_2d` and `capillary_wave_2d`. `--transport coupled` advects with the fluid velocity (`Velocity_source=coupled_field`), and `--transport wet_extension` writes the wall-compatible wet extension of the D18 and capillary-rise decks; both remain for comparison |
| Level-set kinematic reconciliation | on (`Enable_kinematic_reconciliation=true`; `--kinematic-reconciliation off` reproduces the earlier decks). After every accepted step the transported `phi` is corrected locally so that the step's change of the sharp P1 area equals the interface flux of the transport velocity (`FE/LevelSet/LevelSetKinematicReconciliation.h`) | parameter-free, no global shift; it removes the contact-line area drift of the Galerkin transport, see "Area drift" below |
| Level-set sign-definite patch bounds | on (`Enable_sign_definite_patch_bounds=true`; `--sign-definite-patch-bounds off` reproduces the earlier decks). After every accepted step, a node whose whole patch lies in one phase is kept inside the range of the previous values over its patch (`FE/LevelSet/LevelSetSignDefinitePatchBounds.h`) | parameter-free local maximum principle of exact transport; it never changes a cut cell, so the interface, the contact line and the area are untouched. It stops single wall vertices next to a contact line from crossing the isovalue, see "Spurious wall spots" below |
| Level-set maintenance | none in the protocol. `--reinitialization` enables projection reinitialization every 10 steps with at most 4 iterations; the zero set then moves by at most `1e-10` per call, and contact cells are only rescaled | the transport of `static_drop_2d`, so that the D2 comparison uses one transport. The optional values are those of the D18/D38 and sloshing decks. In the first smoke run (below) that projection did not converge in 4 iterations and was skipped, so it would not have changed the state |
| Time integration | generalized-alpha, `rho_inf = 0.5`, fixed step; no environment variable. `--time-integration backward_euler` for comparison runs | as `static_drop_2d`. Vertex crossings are accepted within a step with the default restart budget, see "Vertex crossings" below |
| Nonlinear solve | relative tolerance 1e-4 per equation, at most 8 Newton iterations (fluid) and 4 (level set) | as `static_drop_2d` |
| Linear solve | FSILS GMRES, as `static_drop_2d` | `--linear-solver eigen_direct` writes the serial Eigen direct solver, for comparison runs |

**Time step.** As in `static_drop_2d`, the capillary limit of a one-sided free
surface is `dt <= sqrt(rho h^3 / (4 pi gamma))`, the Brackbill–Kothe–Zemach
limit `sqrt(rho h^3 / (2 pi gamma))` with the fixed factor `1/sqrt(2)` that
follows from the one-sided density sum. `generate_case.py` rounds `dt` down
so that the run is exactly 100 equal output intervals.

Time-step options (2026-10-06), for the fixed-step protocol under validation:

| Option | Effect |
|---|---|
| `--dt-rule capillary-limit` (default) | the limit of each level, as above: 2,800 / 7,900 / 22,300 steps at `R/h = 16 / 32 / 64` |
| `--dt-rule fixed` | one physical step for every level, the capillary-limit step of the coarsest level `R/h = 16`: `dt = T/2800 = 4.374e-3`, outputs every 28 steps; 1, 2.83 and 8 times the capillary-limit step of `R/h = 16, 32, 64`. At `R/h = 16` the deck equals the capillary-limit deck |
| `--surface-tension-semi-implicit NormalIncrement` | the lagged normal-increment capillary term of decision D13 in the free-surface block (default `None`). It needs `--transport pde_extension` or `coupled` |
| `--dt-divisor 2`, `4` | the step of either rule divided exactly, with the steps and the output cadence multiplied, so the output times are unchanged |
| `--dt-multiple m` | `m` times the capillary limit of the rule, rounded down to the output intervals (larger-step studies; `m = 2` halved by `--dt-divisor 2` gives the `m = 1` deck) |

`case.json` records `dt_rule`, `dt_multiple`, `dt_base`, `dt_divisor`,
`dt_over_capillary_limit` and `surface_tension_semi_implicit`.

**Vertex crossings.** A moving contact line or interface crosses mesh
vertices all the time, and every crossing changes the cut topology during a
step. The solver treats that as a normal event (branch `dev/vertex-crossing`):

- Inside a step, each outer fixed-point pass regenerates the cut geometry.
  When the topology differs from the one the inner Newton solve used, the
  solve restarts on the new topology. The number of such restarts per step
  is bounded by the outer iteration limit (12) by default;
  `GeneralSimulationParameters/Max_cut_topology_restarts_per_step` overrides
  it (0 restores the old stop-and-reject behaviour).
- A topology change found in the initial canonicalization of an attempt (the
  generalized-alpha stage predictor `u_n + alpha_f dt uDot_n`, or the
  backward-Euler start `u_n`) is adopted instead of rejecting the attempt.
- For generalized-alpha, the endpoint may lie in a newer topology than the
  stage: the endpoint gate still rejects missing geometry, and the stage
  solve itself still has to converge on the topology it used.
- A topology that returns to one already visited in the same step (a cycle
  A -> B -> A across a switching surface; a smaller step does not remove
  it) ends the step on that revisited epoch once the inner solve converges
  there. The log line is `diagnostic=cut_topology_cycle
  action=accept_on_frozen_epoch`.
- `FsilsVector::dot` and `copyFrom` compare vector layouts structurally.
  Small-cut aggregation re-augments the constraints and rebuilds the FSILS
  layout object, so the scratch and matrix vectors held different but equal
  layouts.
- The backward-Euler kinetic-work bookkeeping of the discrete energy ledger
  cannot bind to the previous endpoint after a topology change (the new
  constraints re-project the previous velocity). It now logs
  `diagnostic=backward_euler_kinetic_work_binding status=unavailable`,
  records the step without that pairing and continues; the ledger is then no
  longer contiguous. Generalized-alpha does not use this pairing.

Outputs are written at the nominal times; there is no step retry.

**Run length.** `T = 5 t_mu = 12.25`, the lower end of the 5 to 10 viscous
times of D3. This is 30 visco-capillary times and 12 capillary times.

An order-of-magnitude estimate of the contact-line relaxation, not a
measurement: with Navier slip the contact-line speed is about
`V = gamma theta (cos(theta_e) - cos(theta)) / (3 mu ln(R/l_s))`
(Cox–Voinov form). For the 60 degree case this gives an initial `V` of about
0.26 and a relaxation time of about 1 (the base moves by 0.24), so `T` is
about 12 relaxation times. `verify.py` reports the base change over the last
quarter of the run as the a-posteriori check.

## Metrics (`verify.py`)

All quantities are computed from the solver's VTU/PVTU point data (`phi`,
`Velocity`) on the output mesh itself. Coincident points of MPI pieces are
merged.

| Metric | Definition |
|---|---|
| Liquid area `A(t)` | exact area of `{phi_h < 0}` for the P1 interpolant (the `LinearCorner` cut volume on `Triangle3`) |
| Contact points | the zero crossings of `phi_h` on the edges of the contact wall; exactly two are required |
| `base_half_width` | half the distance between the two contact points |
| `apex_height` | the largest height above the wall of the interface points (edge zero crossings of `phi_h`) |
| `contact_angle_left`, `contact_angle_right` | a circle is fitted (geometric least squares) to the interface points of that side with `0 <= y <= 0.25 R_ref`. The angle, through the liquid, is taken between the wall direction pointing into the liquid and the circle tangent at the circle point closest to the contact point. |
| `contact_angle_error_degrees` | `max(abs(theta_left - theta_e), abs(theta_right - theta_e))` at the end of the run |
| `base_radius_relative_error`, `apex_height_relative_error` | against the reference cap at the end of the run |
| `liquid_area_relative_drift_max` | `max_t abs(A(t) - A(0)) / A(0)`, with `A(0)` from `phi` in `mesh/mesh-complete.mesh.vtu` |
| `max_speed(t)` | `max abs(u_h)` over the vertices with `phi_h < 0` |
| `max_speed_growth_ratio` | `max_speed(T)` divided by the largest `max_speed` over the outputs with `T/2 <= t <= 3T/4`, as in `static_drop_2d` |
| Reported only | the P1 angle in each wall triangle holding a contact point; the height–base angle `2 atan(H/b)` and the angle of one circle fitted to the whole interface (both equal `theta_e` for a circular cap); the left–right asymmetry; errors against the nominal cap and against the cap with the final area (D10, below); the base change over the last quarter; `mu max_speed / gamma` in the last quarter; the histories of area, speed, base and both angles |

**Why a local circle fit.** The relaxed interface is a circle, so the fit has
no model error at equilibrium. On sampled exact caps, the local fit
reproduces the angle to within 0.13 degrees at `R/h = 16` and 0.05 degrees at
`R/h = 32` for all three angles (the tests check 0.2 and 0.08), with at least
9 points per side. A quadratic `x(y)` fitted over the same window is biased by
about 1.6 degrees at 60 degrees. The P1 angle of the single wall triangle is
off by up to 2.1 degrees at `R/h = 16` and 0.9 degrees at `R/h = 32`, because
it is the angle of a chord. The window `0.25 R_ref` is fixed once, for all
cases and levels.

Observed orders of the angle error are printed when it decreases. `verify.py`
exits with status 2 and a message on missing or inconsistent data: no
`case.json`, no output, missing arrays, non-triangle cells, non-finite values,
other than two wall contact points at the end, or a run whose last output is
more than half a step away from `T`. It also exits with status 2 on a `--max-steps` smoke run unless
`--allow-truncated` is given.

## Tolerances and their sources

Each criterion applies to each (capillary form, `theta_e`, dt rule, dt
multiple, dt divisor) refinement study. Cases written before the time-step
options count as the capillary-limit rule with divisor 1.

| Criterion | Limit | Where | Source |
|---|---|---|---|
| `contact_angle` | at most 2 degrees; strictly decreasing with refinement | 2 degrees at `R/h = 32`; decrease over 16/32/64 | tracker M4 working criterion |
| `base_radius` | at most 0.02 | `R/h = 32` | tracker M4 working criterion ("within 2%"), at the level of the angle criterion |
| `apex_height` | at most 0.02 | `R/h = 32` | tracker M4 working criterion, as above |
| `volume_drift` | at most 1e-4 | every level | D1 working criterion |
| `no_velocity_growth` | growth ratio at most 1 | every level | D1 working criterion |
| `time_step` | between `dt` and `dt/2`: end-state angles change by at most 0.1 degree, base half-width and apex height by at most 0.1%; over the outputs, the angles by at most 0.5 degree and the base half-width by at most 0.1% of `b_ref`; every output measurable in both runs | finest common level of the `dt` and `dt/2` studies | pre-registered 2026-10-06 from the D13 sessile runs (below) |

**Time-step criterion** (`time_step_criterion` in `tolerances.json`, written
before any fixed-step run was analysed). `verify.py` applies every criterion
above to the `dt` study (all levels) and to the `dt/2` study
(`R/h = 16` and 32; the monotone decrease over 16/32/64 is not evaluated at
`dt/2` without `R/h = 64`), and compares the divisor-1 and divisor-2 studies
of each (capillary form, `theta_e`, rule, multiple) at their finest common
level. It reports the changes at the other common levels and between `dt/2`
and `dt/4`. With one divisor the criterion is reported as not evaluated. The
numbers come from the D13 sessile runs (design note
`Documentation/free_surface_semi_implicit_surface_tension_design.md`
section 9.6; 60 degrees, `R/h = 16`, two viscous times, 20 outputs): halving
the step from 4 to 2 times the `R/h = 16` capillary-limit step changed the
final angles by 0.005 and 0.037 degrees, the base by 2e-7 and the apex by
6.9e-5, and over the history the angles by at most 0.15 and 0.37 degrees (one
output next to a vertex crossing) and the base by at most 2.2e-4 of `b_ref`.
The end-state limits are 1/20 of the M4 gates. The end state is an
equilibrium that hides time error, so the histories are gated too: the
contact-line position (base) is smooth in time, while the circle-fit angle
jumps by a few tenths of a degree when the interface crosses a vertex one
step earlier or later, whatever the step; 0.5 degrees is 1/4 of the angle
gate.

The area criterion gates the maximum drift over the whole run, not its final
value (decision D11); `liquid_area_relative_drift_max` is that maximum.

Spatial convergence is judged with the time-step error removed (decision
D10). The end state is a static equilibrium, whose discrete form does not
depend on `dt`; `dt` enters the end-state metrics only through the area that
the transport gains or loses on the way. The angle does not depend on the
area. `verify.py` therefore also reports the base and apex errors against the
cap with the final liquid area (`*_relative_error_final_area`), which remove
that contribution; the criteria themselves are unchanged.

## How to run the refinement study

Use the Python stack of the benchmark README (numpy; pyvista for
`verify.py`). Put generated cases and output under `$SCRATCH`, and start the
solver through `mpiexec` in a job submitted with `sbatch --export=NONE`. The
case needs no environment variable:

```bash
B=tests/cases/fluid/free_surface_benchmarks/sessile_drop_2d
OUT=$SCRATCH/free-surface-benchmarks/sessile_drop_2d/$(git rev-parse --short HEAD)
SVMP=/path/to/build/bin/svmultiphysics
for a in 60 90 120; do for L in 16 32; do
  d=$OUT/surface_stress/theta$a/L$L
  python3 $B/generate_case.py --level $L --contact-angle $a --output-dir $d
  # job script: set PATH/LD_LIBRARY_PATH, then
  #   cd $d && mpiexec -n 1 --bind-to none $SVMP solver.xml 2>&1 | gzip -1 > solver_run.log.gz
done; done
python3 $B/verify.py $OUT/surface_stress/theta60/L{16,32,64} --json $OUT/theta60.json
```

For a quick schema check use `--max-steps 5`; `verify.py` refuses such runs
unless `--allow-truncated` is given.

## Expected cost per level

**Measured** in the vertex-crossing validation at `R/h = 16` (job 46084684,
below), serial, with eight runs sharing one 8-core node: generalized-alpha
with FSILS GMRES takes 0.77 s per step for `theta_e = 60` (624 vertices) and
2.0 s per step for `theta_e = 120` (1,479 vertices). Backward Euler takes
0.87 s and 2.4 s with FSILS, and 0.95 s and 3.0 s with the Eigen direct
solver. The log grows by about 0.2 MB per step (about 8 MB per 300 steps
compressed). The smoke runs before these fixes measured 4 to 13 s per step
on a node shared with the regression suites.

**Extrapolated** from the 60 degree rate (0.77 s) with cost proportional to
(vertices)^0.9, as in `static_drop_2d`. The 120 degree case measured 20%
above this rule at `R/h = 16`, because its receding contact line crosses
more vertices per step (81 against 50 topology changes in 300 steps).

| R/h | steps | 60 deg | 90 deg | 120 deg |
|---:|---:|---|---|---|
| 16 | 2,800 | 624 vertices, about 0.6 h | 1,032, about 0.9 h | 1,479, about 1.3 h (1.6 h measured rate) |
| 32 | 7,900 | 2,387, about 6 h | 3,995, about 9 h | 5,757, about 13 h |
| 64 | 22,300 | 9,333, about 2.3 days | 15,717, about 3.6 days | 22,713, about 5 days |

Every level fits the 7-day `amarsden` limit serially at these rates; the 120
degree case at `R/h = 64` has little margin (about 6 days with the measured
20% excess). The run length is not shortened, because D3 asks for relaxed
states.

## Vertex-crossing validation (2026-09-30)

Job 46084684, build of `8f37ce57` (branch `dev/vertex-crossing`), `R/h = 16`,
`SurfaceStress`, coupled transport, 300 steps each (`t = 1.31`, 11% of `T`),
serial. Raw output:
`$SCRATCH/free-surface-benchmarks/sessile_drop_2d/vertex-8f37ce57/`.

| Configuration | `theta_e` | accepted, rejected | topology changes | cycles ended on a revisited epoch | ledger records without kinetic-work pairing |
|---|---:|---|---:|---:|---:|
| generalized-alpha, FSILS | 60 | 300, 0 | 50 | 12 | 0 |
| backward Euler, FSILS | 60 | 300, 0 | 48 | 11 | 24 |
| backward Euler, Eigen direct | 60 | 300, 0 | 48 | 11 | 24 |
| generalized-alpha, FSILS | 120 | 300, 0 | 81 | 8 | 0 |
| backward Euler, FSILS | 120 | 300, 0 | 138 | 23 | 83 |
| backward Euler, Eigen direct | 120 | 300, 0 | 138 | 23 | 83 |

Histories from `verify.py` through a truncated view, generalized-alpha with
FSILS (angles from the local circle fit):

| `theta_e` | `t` | left, right angle (deg) | base half-width | `max abs(u)` | `(A - A(0))/A(0)` |
|---:|---:|---|---:|---:|---:|
| 60 | 0.004 | 89.0, 88.7 | 0.626 | 0.437 | -2e-7 |
| 60 | 0.33 | 65.3, 68.4 | 0.731 | 0.241 | 5.2e-3 |
| 60 | 0.66 | 61.2, 61.2 | 0.791 | 0.155 | 8.1e-3 |
| 60 | 0.99 | 61.0, 62.2 | 0.818 | 0.098 | 9.6e-3 |
| 60 | 1.31 | 59.5, 59.3 | 0.841 | 0.069 | 1.00e-2 |
| 120 | 0.004 | 90.8, 90.9 | 1.268 | 0.370 | 5e-8 |
| 120 | 0.33 | 116.3, 114.8 | 1.160 | 0.249 | 4.0e-4 |
| 120 | 0.66 | 117.4, 118.3 | 1.088 | 0.228 | 3.3e-3 |
| 120 | 0.99 | 119.8, 118.6 | 1.027 | 0.227 | 5.8e-3 |
| 120 | 1.31 | 115.3, 118.6 | 0.978 | 0.182 | 6.7e-3 |

The reference base half-width is 0.866 for both angles. At `t = 1.31` the
60 degree cap has a base error of 2.8% and an apex error of 4.8% and is
still slowing down; the 120 degree cap is still receding (base error 13%).
Backward Euler gives the same histories to within 0.3 degrees and 0.1% in
base at the sampled times, with a final drift of 9.8e-3 (60) and 6.9e-3
(120). FSILS and the Eigen direct solver give identical backward-Euler
results. The area grows in both cases; the maximum drift is about 100 times
the `volume_drift` limit. The drift was already present in the smoke runs
before these fixes (below) and is not addressed here (transport, tracker
D9). `--transport wet_extension` was not run.

A 20-step static drop (`static_drop_2d`, `R/h = 8`, no crossings) gives
bit-identical output with the baseline build of `11aba0a2`.

## Area drift: attribution and kinematic reconciliation (2026-10-01)

Serial, `R/h = 16`, `SurfaceStress`, generalized-alpha, FSILS, transport
`pde_extension`, 100 steps (`t = 0.437`) unless stated. Raw output:
`$SCRATCH/svmp-dev-volume/runs/`.

**Budget before the fix** (build `35a81fd3`, 60 deg). The change of the
exact P1 area `A_h` over each step was split, from the saved fields, into
the interface flux of the transport velocity, `dt (F_n + F_{n+1}) / 2` with
`F = int_Gamma w . n`, and the rest:

| Source | Contribution to `A/A(0) - 1` |
|---|---|
| total (maximum over the run) | 1.25e-3 (final 8.5e-4) |
| (a) level-set transport: area change minus interface flux | +8.47e-4 (99.8%) |
| interface flux itself, i.e. the fluid's discrete divergence on the liquid | +1.5e-6 (backward Euler: 5e-12) |
| (b) wall maintenance or reinitialization | 0: off in the protocol; with `--reinitialization` the projection did not converge in 4 iterations and was skipped at every call |
| (c) cut-topology epochs: excess of the 9 restart/cycle steps over their neighbours | +2.8e-6 |
| (d) measurement | 0: the `Wet volume diagnostic` equals the exact P1 cut area of the transported `phi` to round-off |

Controls (maximum drift): SUPG off 1.26e-3; backward Euler 1.07e-3; `dt/2`
1.25e-3; 90 deg started at 90 deg (no contact-line motion) 1.7e-7; coupled
transport 5.2e-3; 120 deg 1.5e-4. At `t <= 0.4375` the drift is 2.6e-3,
1.2e-3 and 0.93e-3 at `R/h = 8, 16, 32` (observed order 1.1, then 0.4). A
replay of the P1 Galerkin/SUPG/generalized-alpha step from the saved `phi`
and `w` reproduces the per-step area error to 1%, as do its variants without
SUPG, with lumped mass or with backward Euler.

**Cause.** The Galerkin transport satisfies `phi_t + w . grad(phi) = 0` only
in `L2(Omega)`. The nodal kinematic residual on the interface sits on the
wall vertex next to each contact point and the vertex above it: the
Navier-slip velocity peaks at the contact vertex, `phi` develops a gradient
kink there, and the nodal update blends the slopes of both wall cells. At
step 15 the contact point moved 23% faster than the fluid at the contact
point. The error follows the contact-point position within a wall cell (its
sign changes about every half cell) and adds liquid on balance in both the
spreading and the receding case.

**Fix.** `Enable_kinematic_reconciliation=true` in the level-set equation
(opt-in in the solver; written by the benchmark generators since the decision
of 2026-10-05): after every accepted step, a local, parameter-free
correction makes the step's area change equal the interface flux of the
transport velocity (`FE/LevelSet/LevelSetKinematicReconciliation.h`).

| Run | Max drift, reconciled (`dd9e831e`) | Max drift, `35a81fd3` |
|---|---|---|
| 60 deg, full protocol, 2800 steps to `T = 12.25` | 6.1e-6 | 2.6e-2; from step 2324 a spurious dry spot on the wall inside the footprint (four wall crossings) |
| 120 deg, full protocol | 6.8e-5 | 3.7e-3 |
| 60 deg, `t <= 0.4375`, nested meshes `R/h = 8, 16, 32` | 3.3e-6, 2.8e-6, 5.6e-6 | 2.6e-3, 1.2e-3, 0.93e-3 |
| 60 and 120 deg, coupled transport, 100 steps | 9.6e-6, 3.4e-6 | 5.2e-3 (60 deg) |

The remaining drift is the interface flux itself (the fluid's discrete
divergence on the liquid, generalized-alpha stage) plus a per-step residual
of order 1e-8; both accumulate with the number of steps, so it no longer
decreases with `h`. The 60 degree cap ends at 61.7/58.4 degrees (circle fit
60.2, height-base angle 60.2), base error 0.10%, apex error 0.29%; the base
radius history matches the run without reconciliation to within 0.3%
(`35a81fd3` at `t = 10.0`, before its dry spot: 61.4/59.0 degrees, circle fit
60.7). The 120 degree cap follows the run without reconciliation to within
2 degrees in angle; at `t = 4.0` its base half-width is 0.853 against 0.864
(that run had gained 0.35% area), and both end within 0.3% of the reference
0.866 (0.864 and 0.868). Its drift (6.8e-5 at the end) is almost all
interface flux: under generalized-alpha the endpoint velocity is not
discretely divergence free on the endpoint liquid (about -4e-8 of the area
per step near equilibrium); backward Euler gives 5e-12 per 100 steps. The
reconciliation adds 20-45% to the time per step
(one accepted-step maintenance transaction per step). Jobs: 46146618
(full-protocol, refinement and regression runs), 46143892 (coupled
transport); case directories under `runs/val`, `runs/kr5`, `runs/kr4`.

## Spurious wall spots: mechanism and sign-definite patch bounds (2026-10-05)

Without the patch bounds, single wall vertices settle next to the isovalue
or cross it, giving extra wall crossings that `verify.py` rejects: the
120 degree full protocol from step 952 with the reconciliation and from step
1344 without it (a dry vertex two cells behind the receding right contact
line becomes wet), and the 60 degree protocol without the reconciliation
from step 2324 (a wet vertex two cells inside the footprint becomes dry).
Raw output: `$SCRATCH/svmp-dev-wallspots/runs/`, analysis scripts in
`$SCRATCH/svmp-dev-wallspots/analysis/`.

**Mechanism.** At the 120 degree vertex `x = 1.0` the transported `phi` falls
from `1.08 h` (step 504) to `0.02 h` (step 924) while its wall neighbour on
the contact-line side rises from `0.77 h` to `2.27 h`. The vertex is a dry
extension vertex whose whole patch is dry. Next to it the wall rows carry a
growing grid-scale oscillation (within about three cells of each contact
line; the root-mean-square second difference of the wall row grows from
`0.11 h` to `1.0 h` over the run), the gas side of the wall row is flattened
to about 0.5 to 0.6 of the signed distance, and the PDE extension velocity
has a diverging stagnation point at the vertex (zero normal velocity on the
dry wall vertex against `0.053` at the interface-cell vertex above it;
divergence 0.9 to 1.5 in the two adjacent wall cells). Once across, the
vertex is a wet "known" vertex of a liquid sliver that the fluid does not
resolve, its velocity is zero, and it stays negative to the end.

| Test (restart from the saved step-840 state, or offline replay) | Result |
|---|---|
| solver, reconciliation on / off / extension wall impermeability off | crossing after 89 / 75 / 47 steps |
| solver, SUPG off | identical to SUPG on to four digits (`tau` is about `dt/4` at Courant number 0.004) |
| offline replay of the P1 Galerkin/SUPG/generalized-alpha step from the saved `phi` and `w`, no reconciliation | reproduces the decline and the crossing (between steps 952 and 1008; the run crossed at 948); on the 60 degree run without reconciliation it follows the solver to `5e-4 h` over 170 steps |
| replay decomposition, steps 504 to 1008 | consistent-mass coupling to the neighbours `-0.44 h`, own Galerkin advection at the stagnation point `-0.67 h` |
| replay variants | lumped mass slows the decline but does not stop it; adding `div(w) phi / 2` (skew-symmetric form) makes it faster; the patch bounds stop it |

The reconciliation never moves the vertex (it only moves nodes of cut cells
and keeps sign classes); it changes the contact-line history and so the
onset step. Wall maintenance does not run in the protocol. The cause is
therefore the transport itself: the P1 Galerkin step has no local maximum
principle, and next to a contact line, where the transport velocity has a
mesh-scale kink and a diverging stagnation point on the wall and the field
is only about `h` thick, a vertex whose whole patch lies in one phase is
driven across the isovalue.

**Fix.** The exact transport keeps the value of a node inside the range of
the previous values over its patch (one-ring Courant number at most one, no
inflow). `Enable_sign_definite_patch_bounds` restores that bound on the
nodes whose patch lies in one phase at the previous state and, apart from
the node, at the transported endpoint. Nodes of cut cells never change:
the contact line, the angle and the area are not touched (no second angle
mechanism, D4; no effect on the area, D11), and there is no parameter (P1).

**Validation** (`R/h = 16`, `SurfaceStress`, generator defaults with the
patch bounds; build of the branch at `056fa2aa` plus the three patch-bound
commits; job 46685007). The restart runs start from the saved states of the
runs without the bounds:

| Run | Wall crossings without the bounds | With the bounds |
|---|---|---|
| 120 deg, from step 840, reconciliation on | 4 from step 840 + 89 | 2 throughout; the vertex stays at `0.338 h` |
| 120 deg, from step 840, reconciliation off | 4 from step 840 + 75 | 2 throughout |
| 60 deg, from step 2128, reconciliation off | 4 from step 2128 + 187 | 2 throughout; the vertex stays at `-0.741 h` |
| 60 deg, from step 2128, reconciliation on | 2 throughout | 2 throughout |

Full protocol, 2800 steps to `T = 12.25` (`verify.py` at a single level
reports the angle, base and apex criteria as failed only because they are
gated at `R/h = 32`; before the bounds the 120 degree run stopped with
"expected two wall contact points, found 4"):

| `theta_e` | wall crossings | left, right angle (deg) | angle error | base error | apex error | max area drift | growth ratio |
|---:|---|---|---:|---:|---:|---:|---:|
| 60 | 2 at all 101 outputs | 61.745, 58.435 | 1.745 | 1.02e-3 | 2.85e-3 | 8.7e-6 | 0.744 |
| 120 | 2 at all 101 outputs | 120.112, 120.660 | 0.660 | 5.16e-3 | 1.47e-3 | 1.5e-5 | 0.513 |

The errors are against the reference cap with the initial area. The 60
degree histories match the reconciled run without the bounds (which had no
spot) to 0.02 degrees in angle, `1e-5` in base and `2.4e-5` in apex height.
The 120 degree histories match the reconciled run without the bounds to
1.6 degrees, `3e-3` and `3.4e-4` before its spot (`t < 4.16`). The bounds act
on at most 8 nodes per step with corrections of at most `1.9e-4` (`h/330`);
no node had to be stopped from crossing, because the bound holds each drifting
node from its first excursion on. The patch bounds add no measurable time (353 s against 365 s for the 200-step 120 degree restart case with the option off, in the same job).
With the option off the solver output is bitwise identical to the solver
without it (120 degree restart case, `capillary_wave_2d` `lambda/h = 16`,
`linear_sloshing_2d` `L/h = 16`).

## Smoke runs before the vertex-crossing fixes (2026-09-30)

All at `R/h = 16` with `SurfaceStress` and truncated with `--max-steps`. The
first run used the build of `9b15dbbc`; the others used the build of
`bf927c09`, the same D4 code plus the Physics diagnostic strings. Later
commits change only this directory.
Raw output: `$SCRATCH/free-surface-benchmarks/sessile_drop_2d/smoke-*/`.

| Commit, job | Configuration | Result |
|---|---|---|
| `9b15dbbc`, 46023882 | 60 deg; generalized-alpha, fixed step, FSILS; reinitialization every 10 steps | input parsed; 14 steps accepted; step 15 rejected with `CutTopologyChanged` and the run aborted. The projection at step 10 found both contact cells (scale residual 7e-18) but did not converge in 4 iterations and was skipped |
| `bf927c09`, 46034380 | 60 and 120 deg; generalized-alpha with the bisection retry | 18 and 17 steps accepted, 13 rejections each; failed at `dt/256` |
| `fcf0b183`, 46039341 | 60 and 120 deg; backward Euler, restarts, FSILS | stopped in step 1: `FsilsVector::dot: layout mismatch` |
| `328282c1`, 46040581 | 60 and 120 deg; backward Euler, restarts, Eigen direct | 16 and 14 steps accepted, the first vertex crossing accepted with one restart each; then stopped in the backward-Euler kinetic-work bookkeeping |

`verify.py` read every output of these runs through a truncated view (its
own completeness check correctly refuses them). Backward Euler, Eigen direct:

| `theta_e` | outputs, last `t` | left, right angle | base half-width | `max abs(u)` | area drift |
|---:|---|---|---|---|---|
| 60 | 15, 0.066 | 88.8, 88.5 -> 73.7, 72.2 deg | 0.627 -> 0.656 | 0.30 -> 0.32 | 4.6e-4, growing |
| 120 | 13, 0.057 | 91.0, 91.1 -> 103.0, 102.1 deg | 1.267 -> 1.246 | 0.26 -> 0.42 | 4.6e-5 |

Both caps move the right way: the 60 degree cap spreads and the 120 degree
cap recedes, with left and right angles within 1.5 degrees of each other.
The generalized-alpha run gave the same 60 degree trend (74.2 and 72.6
degrees after 14 steps). The area of the spreading case grew by 1.6e-5 to
6.5e-5 of its value per step, faster as the run went on. If that continues,
the `volume_drift` criterion (1e-4) will fail. Conservative transport (WP-6) is the known remedy.

## Open points

- The area drift of both cases (about 1e-2 over 11% of the run), above;
  removed by the opt-in kinematic reconciliation (section "Area drift").
- Spurious wall spots: in the dry wall region behind a receding contact
  line, and once inside the footprint, single wall vertices of the
  transported `phi` settled within about 1e-3 of zero or crossed it, giving
  tiny extra wall crossings (120 deg from step 952 with the reconciliation
  and from step 1344 without; 60 deg without it from step 2324), so
  `verify.py` rejected those full runs. Removed by the sign-definite patch
  bounds, see "Spurious wall spots" above.
  A first comparison with the PDE velocity extension of tracker D9
  (`--transport pde_extension`; job `46108807`, branch
  `dev/pde-velocity-extension` at `0e4ef8e7`, `R/h = 16`, `SurfaceStress`,
  generalized-alpha, FSILS, 300 steps) reduces the maximum area drift about
  fourfold, from 1.00e-2 to 2.45e-3 at 60 degrees and from 6.7e-3 to 1.6e-3
  at 120 degrees, with the same angle histories to within 2.3 degrees (final
  angles 61.8/60.9 and 115.1/118.7 degrees) and about 15% more time per
  step.  The drift is still above the 1e-4 criterion.
- A step that cycles between two topologies ends on the revisited one
  without a fresh zero-update certificate on a third epoch; the inner solve
  on that epoch still meets the Newton tolerance.
- With backward Euler the energy ledger loses its kinetic-work pairing after
  the first topology change that re-projects the previous velocity, and is
  reported as not contiguous from there on.

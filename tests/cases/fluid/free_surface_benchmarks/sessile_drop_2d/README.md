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
| Level-set transport | P1, advected by the fluid velocity (`Velocity_source=coupled_field`, `--transport coupled`), SUPG with the production constants (tau scale 0.5, transient scale 2.0); no volume correction, no discontinuity capturing, no bound limiter | as `static_drop_2d`; the area drift is a measured quantity. `--transport wet_extension` writes the wall-compatible wet extension of the D18 and capillary-rise decks; `--transport pde_extension` is the placeholder for the PDE velocity extension of tracker D9 and is refused until that extension lands |
| Level-set maintenance | none in the protocol. `--reinitialization` enables projection reinitialization every 10 steps with at most 4 iterations; the zero set then moves by at most `1e-10` per call, and contact cells are only rescaled | the transport of `static_drop_2d`, so that the D2 comparison uses one transport. The optional values are those of the D18/D38 and sloshing decks. In the first smoke run (below) that projection did not converge in 4 iterations and was skipped, so it would not have changed the state |
| Time integration | generalized-alpha, `rho_inf = 0.5`, fixed step; no environment variable. `--time-integration backward_euler` for comparison runs | as `static_drop_2d`. Vertex crossings are accepted within a step with the default restart budget, see "Vertex crossings" below |
| Nonlinear solve | relative tolerance 1e-4 per equation, at most 8 Newton iterations (fluid) and 4 (level set) | as `static_drop_2d` |
| Linear solve | FSILS GMRES, as `static_drop_2d` | `--linear-solver eigen_direct` writes the serial Eigen direct solver, for comparison runs |

**Time step.** As in `static_drop_2d`, the capillary limit of a one-sided free
surface is `dt <= sqrt(rho h^3 / (4 pi gamma))`, the Brackbill–Kothe–Zemach
limit `sqrt(rho h^3 / (2 pi gamma))` with the fixed factor `1/sqrt(2)` that
follows from the one-sided density sum. `generate_case.py` rounds `dt` down
so that the run is exactly 100 equal output intervals.

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

Each criterion applies to each (capillary form, `theta_e`) refinement study.

| Criterion | Limit | Where | Source |
|---|---|---|---|
| `contact_angle` | at most 2 degrees; strictly decreasing with refinement | 2 degrees at `R/h = 32`; decrease over 16/32/64 | tracker M4 working criterion |
| `base_radius` | at most 0.02 | `R/h = 32` | tracker M4 working criterion ("within 2%"), at the level of the angle criterion |
| `apex_height` | at most 0.02 | `R/h = 32` | tracker M4 working criterion, as above |
| `volume_drift` | at most 1e-4 | every level | D1 working criterion |
| `no_velocity_growth` | growth ratio at most 1 | every level | D1 working criterion |

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
  the PDE velocity extension of tracker D9 is the planned transport.
- A step that cycles between two topologies ends on the revisited one
  without a fresh zero-update certificate on a third epoch; the inner solve
  on that epoch still meets the Newton tolerance.
- With backward Euler the energy ledger loses its kinetic-work pairing after
  the first topology change that re-projects the previous velocity, and is
  reported as not contiguous from there on.

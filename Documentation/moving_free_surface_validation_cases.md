# Moving Free-Surface Validation Cases

This document defines the validation targets for the moving free-surface
implementation.  Each case is intended to be run for both supported surface
representations when the required features are enabled:

- fitted ALE free surface
- unfitted level-set free surface with generated interface geometry

Common reported quantities:

- free-surface elevation or interface position
- pressure along a vertical probe line
- maximum velocity magnitude
- water volume or area
- generated-interface measure and normals for level-set runs
- mesh quality metrics for fitted ALE runs

Focused unfitted level-set qualification evidence for the 2026-05-22
active/inactive cut-volume retention review is recorded in
`Documentation/unfitted_level_set_free_surface_qualification_log_20260522.md`.

## Generated Literature Geometry Fixtures

The following mesh fixtures expand the open-vessel coverage beyond the small
analytic examples. They are generated with PyVista, TetGen, and the MMG
executables bundled with the `svv2` environment:

```bash
conda run -n svv2 python tests/cases/fluid/open_vessel_free_surface/generate_validation_meshes.py
```

Generated fitted ALE fixtures:

- `tests/cases/fluid/open_vessel_free_surface/fitted_ale/spheric_test10_lateral_water_1x`

Generated unfitted level-set fixtures:

- `tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test10_lateral_water_1x`
- `tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test05_wet_bed_d18`
- `tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test05_wet_bed_d38`
- `tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test02_dambreak_obstacle`

Each generated case includes:

- `solver.xml`
- `pressure_gauge.csv`
- `benchmark.json`
- `mesh/.../mesh-complete.mesh.vtu`
- `mesh/.../mesh-surfaces/*.vtp`

The benchmark metadata records the published dimensions and source URLs so the
meshes can be regenerated without searching through the literature again.

The SPHERIC Test02 dry-bed obstacle fixture uses a plane-based retained-column
level set, `phi=max(1.992-x, y-0.55)`, not a closed-box signed distance. This
keeps bottom/right/front/back wall-contact regions negative/wet and avoids
spurious generated free-surface cuts on contacted tank walls.

## Implementation Ownership

Navier-Stokes owns free-surface boundary semantics for both fitted ALE and
unfitted level-set cases. For unfitted cases, Navier-Stokes consumes
`FE::level_set` generated interface domains, curvature helpers, cut-cell
metadata, and level-set diagnostics, then installs the free-surface pressure
jump, surface-tension, kinematic, and stabilization terms. Reusable level-set
transport, volume, reinitialization, diagnostics, restart, and generated
interface lifecycle code belongs in `Code/Source/solver/FE/LevelSet`.
One-fluid unfitted free-surface validation cases must declare the wet active
domain explicitly, normally `Active_domain=LevelSetNegative` with
`Active_domain_method=CutVolume`; full-domain unfitted runs require an explicit
diagnostic opt-in.

Output naming note: `WetVolumeFraction` is the cell-centered generated
cut-volume active-domain diagnostic and should be used for wet-area or
wet-volume post-processing. `ActiveFluid` is a vertex-sign visualization mask
only; it helps inspect dry-vertex masking and must not be interpreted as the
integration active domain.

Wet-volume drift diagnostics report the physical wet measure used for
validation drift. The generated cut-volume rules themselves remain in reference
cell coordinates because FE assembly maps them through the parent-cell geometry
at integration time. Log fields with `reference_*` names expose the retained
reference cut measure for debugging quadrature generation; `physical_*` fields
include the parent-cell Jacobian and are the values to compare against fitted
ALE or literature volumes.

Wet-extension level-set advection currently supports nearest-active-vertex and
nearest-interface-point extension modes. In both modes, dry vertices copy the
prescribed source velocity from a nearby wet-side sample. This is a diagnostic
and validation aid for `Velocity_source=prescribed_data` transport fields, not
a PDE-based normal, harmonic, or fast-marching extension. Coupled-field
level-set advection does not use this application-side copy unless the
wet-extension option is enabled.

This is separate from the Navier-Stokes unfitted free-surface
`Enable_velocity_extension` option. That option installs a PDE diffusion term
for the velocity field on the inactive generated cut-volume side. When it is
enabled, the active cut context must retain both active and inactive generated
cut-volume rules. The physical one-fluid Navier-Stokes mass, momentum, and
pressure terms remain restricted to the active wet side. The inactive-side
homogeneous velocity clamp is disabled for this PDE-extension mode so the
extension solve can own those inactive velocity DOFs; inactive pressure support
constraints remain active.

The unfitted validation path remains one-fluid. The exterior side is passive
except for explicitly requested constraints, diagnostics, stabilization support,
or extension terms. Two-phase density/viscosity jumps, pressure enrichment, and
two-sided interface jump conditions require separate two-phase CutFEM
qualification.

High-order implicit generated-interface cases use a refreshed-frozen geometry
contract unless they explicitly select and qualify a differentiated
linear-corner path. The interface and cut-volume quadrature are regenerated
from the current nonlinear state, but high-order quadrature sensitivities are
not part of the default Navier-Stokes Jacobian.

## Level-Set Maintenance Policy

Reinitialization is a validation control, not a universal default. Cases that
measure a prescribed or analytic level-set transport error should avoid
projection repair unless the acceptance metric explicitly includes the repair
effect. Cases that exercise long, strongly deformed free surfaces should use
reinitialization until a conservative transport formulation is available, with
the interface displacement reported as a diagnostic.

Plain level-set advection transports the level-set field as a scalar and is not
conservative by itself. SUPG stabilization can reduce oscillations, but it does
not enforce wet-volume conservation. Validation cases therefore either report
uncorrected wet-volume drift as an error metric or enable explicit volume
correction and report the applied global shift.

A conservative level-set transport option is deferred until the wet-fraction
diagnostics below are available. Adding the option before cell wet fractions,
total wet volume, correction-shift history, and benchmark drift thresholds are
reported would make the option difficult to validate and compare. Until then,
validation inputs must state whether they measure uncorrected drift or use the
existing global-shift volume correction.

Reinitialization policy by tracked validation case:

| Case | Reinitialization policy | Rationale |
| --- | --- | --- |
| `unfitted_level_set/solver.xml` | Enabled | The open-tank fixture is a maintenance regression for a stationary interface; projection repair should leave the flat interface unchanged within tolerance. |
| `unfitted_level_set/linear_sloshing_2d/solver.xml` | Enabled | The standing-wave verifier measures the repaired interface against the analytic height, so projection drift is part of the regression signal. |
| `unfitted_level_set/mms_traveling_interface_2d/solver.xml` | Enabled | The manufactured interface case verifies that projection repair remains compatible with prescribed interface motion. |
| `unfitted_level_set/spheric_test10_lateral_water_1x/solver.xml` | Enabled | The sloshing-impact case has large interface deformation and should keep signed-distance quality during long probes. |
| `unfitted_level_set/spheric_test05_wet_bed_d18/solver.xml` | Disabled for qualification probes | The wet-bed dam-break probes should isolate active-domain, pressure, velocity-extension, and cut-stabilization behavior before projection repair is reintroduced with displacement diagnostics. |
| `unfitted_level_set/spheric_test05_wet_bed_d38/solver.xml` | Disabled for qualification probes | The D38 probe follows the same policy as D18 so both wet-bed depths compare the same solver controls. |
| `unfitted_level_set/spheric_test02_dambreak_obstacle/solver.xml` | Disabled in current startup, active-source velocity, and front-speed probes | The dry-bed obstacle case is still open: projection repair should be reintroduced only as a controlled comparison after official-geometry refined probes resolve, or at least bound, the remaining slow-front and tiny moving-cut behavior. |

Volume-correction policy by tracked validation case:

| Case | Volume-correction policy | Rationale |
| --- | --- | --- |
| `unfitted_level_set/solver.xml` | Enabled | The open-tank fixture should preserve the initial wet area and expose any correction-induced movement of the flat interface. |
| `unfitted_level_set/linear_sloshing_2d/solver.xml` | Disabled | The standing-wave verifier reports active-area drift directly; global correction would hide level-set transport error. |
| `unfitted_level_set/mms_traveling_interface_2d/solver.xml` | Disabled | The manufactured solution should measure the prescribed transport error without a global area shift. |
| `unfitted_level_set/spheric_test10_lateral_water_1x/solver.xml` | Enabled | The long sloshing-impact probe needs bounded mass drift while tank-motion forcing and pressure comparisons are developed. |
| `unfitted_level_set/spheric_test05_wet_bed_d18/solver.xml` | Enabled for qualification probes | The 2026-06-02 validation-grade run showed that uncorrected level-set transport can pass the final profile while failing full-history wet-volume drift. Cadence-1 correction with tolerance `1.0e-10` bounds drift without hiding active-mask or cut-volume errors. |
| `unfitted_level_set/spheric_test05_wet_bed_d38/solver.xml` | Enabled for qualification probes | D38 follows the D18 controls so both wet-bed depths use comparable active-volume correction and strict active-history audit gates. |
| `unfitted_level_set/spheric_test02_dambreak_obstacle/solver.xml` | Enabled | The obstacle dam-break is a long, strongly deformed case where bounded wet-volume drift is required for useful probes. |

## Open Tank At Rest

Purpose: verify that the half-filled open-vessel examples remain in hydrostatic
balance under gravity.

Starting points:

- `tests/cases/fluid/open_vessel_free_surface/fitted_ale/solver.xml`
- `tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/solver.xml`

Setup:

- water density `998.2`
- dynamic viscosity `1.003e-3`
- gravity `(0, -9.81, 0)`
- atmospheric pressure `0`
- initial surface at the hydrostatic pressure reference point
- zero initial velocity

Reference behavior:

- the free surface remains flat
- velocity stays near zero
- pressure remains `p = rho g (eta - y)` up to the gauge offset
- volume remains constant

Acceptance criteria:

- maximum velocity stays below the selected nonlinear tolerance scale
- free-surface displacement remains below one percent of the cell height
- hydrostatic pressure error converges under mesh refinement
- relative volume drift is below `1.0e-4` for fitted ALE and below the configured
  level-set volume-correction tolerance for unfitted runs

## Small-Amplitude Sloshing

Purpose: validate gravity-wave motion in a rectangular tank against linear
sloshing theory.

Setup:

- rectangular tank of length `L` and water depth `h`
- initial surface perturbation `eta(x, 0) = a cos(pi x / L)` with `a / h <= 0.01`
- zero or analytically compatible initial velocity
- surface tension disabled unless explicitly testing capillary-gravity waves

Reference frequency for mode `n`:

```text
k_n = n pi / L
omega_n^2 = g k_n tanh(k_n h)
```

With surface tension enabled:

```text
omega_n^2 = (g k_n + gamma k_n^3 / rho) tanh(k_n h)
```

Measured outputs:

- surface elevation at one or more probe points
- dominant frequency from zero crossings or spectral peak
- volume drift over the run

Acceptance criteria:

- dominant frequency within two percent of the analytic value on the reference
  mesh
- frequency converges toward the analytic value with refinement
- no secular mean-surface drift beyond the volume-conservation tolerance

## SPHERIC Test 10 Sloshing Wave Impact

Purpose: add a published sloshing tank geometry with pressure-gauge data and
large free-surface motion.

Starting points:

- fitted ALE: `tests/cases/fluid/open_vessel_free_surface/fitted_ale/spheric_test10_lateral_water_1x/solver.xml`
- unfitted level set: `tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test10_lateral_water_1x/solver.xml`
- the unfitted fixture now includes a static-tank roll-equivalent body-force
  source at `bc/test10_lateral_water_1x_roll_body_force.dat` and a matching
  angular-velocity history at
  `bc/test10_lateral_water_1x_roll_angular_velocity.dat`; the OOP fixed-frame
  path applies rotating-frame Coriolis forcing, while fitted ALE remains the
  cleaner representation for actual moving walls
- the checked `pressure_gauge.csv` anchors hydrostatic pressure; benchmark
  metadata records the official Sensor1 pressure-history point separately for
  comparison output
- reference fetcher:
  `tests/cases/fluid/open_vessel_free_surface/fetch_spheric_test10_reference.py`
  extracts `SPHERIC_TestCase10/data_files/lateral_water_1x.txt` from the
  official archive without downloading the bundled videos
- pressure-history readiness/comparison verifier:
  `tests/cases/fluid/open_vessel_free_surface/verify_spheric_test10_pressure_history.py`
- roll-equivalent body-force and angular-velocity generator:
  `tests/cases/fluid/open_vessel_free_surface/generate_spheric_test10_roll_body_force.py`
- fixed-frame roll-force model audit:
  `tests/cases/fluid/open_vessel_free_surface/audit_test10_roll_body_force_model.py`
- geometry/topology audit:
  `tests/cases/fluid/open_vessel_free_surface/audit_test10_geometry_resolution.py`
- active-tail/failure-topology audit:
  `tests/cases/fluid/open_vessel_free_surface/audit_test10_active_topology.py`

Setup:

- rectangular tank length `0.900 m`
- 1x tank breadth `0.062 m`
- lateral-water fill height `0.093 m`
- water at `19 C`
- published roll-angle history and pressure records from SPHERIC Test 10

Reference behavior:

- lateral wave impact timing and pressure history at the reported side-wall
  sensor
- overturning and breaking-wave onset for the low-fill water case
- volume conservation across repeated sloshing periods

Acceptance criteria:

- impact timing follows the published pressure trace after tank-motion forcing
  is wired into the executable case
- pressure peaks converge toward the experimental envelope under refinement
- fitted ALE is restricted to pre-breaking or non-overturning comparisons;
  unfitted level set owns the breaking-wave comparison

2026-06-04 status:

- The official lateral-water 1x Sensor1 pressure-history coordinate is
  recorded as `(x=0, y=0.093, z=0.031)`, while the pressure-gauge CSV remains a
  hydrostatic pressure anchor. The verifier samples Sensor1 with interpolated
  point sampling and reports nearest-node metadata.
- The 40-step full-source roll-body-force-plus-Coriolis smoke completed to
  `t=0.04 s` with `success=1`, max relative wet-volume drift
  `8.780522445382697e-08`, one Newton iteration per accepted step, one direct
  linear iteration per accepted step, no rejected or failed step markers, and
  minimum active cut fraction `0.12499811111525794`. It remains
  `not_validation_ready` because the official pressure peak is at `7.3032 s`
  and the reference table ends at `8.35 s`.
- `test10_direct_runtime_projection_20260602.json` projects the same direct
  solver path at the measured 40-step rate: about `54.56865662966666 h` to
  reach the `7.3032 s` pressure peak and `62.38339031458334 h` to cover the
  full `8.35 s` reference horizon. This is a feasibility estimate only, not a
  validation gate.
- A matching coarse-time feasibility control changes only the copied
  qualification case time step from `dt=0.001 s` to `dt=0.01 s` and completes
  40 accepted steps to `t=0.4 s` with no rejected or failed step markers, one
  Newton/direct-linear iteration per accepted step, final relative wet-volume
  drift `4.500267906656587e-05`, and minimum active cut fraction
  `0.11134459042324397`. The Sensor1 sampled-window peak is
  `53.07547803071631 Pa` versus `72.614623 Pa` in the reference at `0.4 s`;
  the sampled-window RMSE is `12.000023237432119 Pa`. This increases coverage
  to `0.04790419161676647` of the `8.35 s` reference horizon and projects about
  `4.024250791458333 h` to the `7.3032 s` pressure peak and
  `4.596784419791667 h` to the full horizon at the measured rate. It is still
  a feasibility result, not a validation gate, because the run ends before the
  pressure peak and `dt=0.01 s` still needs temporal-convergence evidence
  through the peak and horizon.
- A half-step temporal control changes only the copied qualification case time
  step to `dt=0.005 s` and completes 40 accepted steps to `t=0.2 s` with no
  rejected or failed markers, final relative wet-volume drift
  `4.011800522754797e-06`, and minimum active cut fraction
  `0.12421367427586542`. The Sensor1 sampled-window RMSE against the reference
  over `0.005-0.2 s` is `12.559164025891484 Pa`; the run remains
  `not_validation_ready` because it covers only `0.023952095808383235` of the
  reference horizon. As a temporal-control result, it is useful: over the
  shared Sensor1 windows it differs from `dt=0.01` by RMSE
  `0.4378484417601686 Pa` on `0.01-0.2 s` and from `dt=0.001` by RMSE
  `0.6468966370163761 Pa` on `0.005-0.04 s`.
- A 100-step `dt=0.01 s` extension is negative evidence for using that coarse
  path as-is. It accepted 89 steps to `t=0.8900000000000006 s`, then the
  step-90 nonlinear solve failed after 12 Newton iterations
  (`||r||=0.026509843655245294`). The conservation/topology diagnostics degrade
  before the target: wet-volume drift first exceeds `5e-4` at
  `0.6400000000000003 s`, max accepted drift reaches
  `0.0016220364539173617`, active cut fractions drop to
  `1.0851196800525806e-08` during line-search trials, and capped cut-adjacent
  scaling reaches `15`. The partial Sensor1 pressure comparison covers only
  `0.1065868263473054` of the reference horizon and remains
  `not_validation_ready`; over the sampled window the simulated peak is
  `173.58218593623366 Pa` at `0.88 s` versus `357.743423 Pa` in the reference
  at `0.85 s`, with RMSE `85.49466475763961 Pa`.
- A matching D18-style tight volume-correction control keeps `dt=0.01 s` but
  changes the correction cadence to `1` and tolerance to `1.0e-10`. It accepts
  90 outputs to `t=0.9000000000000006 s`, holds max accepted relative wet-volume
  drift to `2.8759465293747008e-08`, applies at most a
  `5.2671974714484904e-05` global shift, and never crosses the `5e-4`
  conservation gate. That proves the loose 100-step drift was a correction
  policy issue rather than mesh resolution. The same control still fails the
  next nonlinear solve (`||r||=0.02155991225491688` after 12 Newton iterations)
  as moving active fractions reach `1.0324876149973442e-08` with capped
  cut-adjacent scaling `15`. Its partial Sensor1 comparison remains
  `not_validation_ready`, covers only `0.10778443113772455` of the reference
  horizon, and underpredicts the sampled-window peak (`172.62057578773832 Pa`
  versus `357.743423 Pa`) with RMSE `87.51945166398855 Pa`.
- A tight-volume adaptive 1 s control adds the XML adaptive time loop
  (`min_dt=6.25e-4`, `max_dt=0.01`, `max_retries=8`) to the same corrected
  setup. It accepts 92 outputs to `t=0.8900000000000003 s`, keeps max accepted
  wet-volume drift at `2.8759465293747008e-08`, and recovers four earlier
  full-step nonlinear failures. It still cannot pass the decisive tiny-cut
  state: step 92 rejects `dt=0.01`, `0.005`, `0.0025`, `0.00125`, and repeated
  min-`dt=0.000625` attempts before `MPI_ABORT`. Cut diagnostics reach minimum
  active fraction `1.0020651389818129e-08` and capped cut-adjacent scaling
  `15`, with no implicit cut fallback cells. The PVD output times were
  reconstructed from accepted-step diagnostics because the abort did not leave
  `result.pvd`, and the Test10 pressure verifier now honors PVD times.
  The partial Sensor1 comparison remains `not_validation_ready`, covers only
  `0.10658682634730543` of the reference horizon, and underpredicts the
  sampled-window peak (`197.71767087923402 Pa` at `0.7325000000000003 s`
  versus `357.7434229999556 Pa` at `0.8500000000000003 s`) with RMSE
  `83.22592969055762 Pa`.
- A relaxed-line-search/max-20 follow-up keeps the same tight-volume adaptive
  setup but disables no-reduction line-search failure and raises the fluid
  Newton limit to `20`. It crosses the previous `0.90 s` barrier and accepts a
  `dt=0.00125 s` substep to physical time `0.9012500000000006 s`, with max
  accepted relative wet-volume drift still `2.8759465293747008e-08`. It is
  still negative validation evidence: the next step from `0.90125 s` rejects
  `dt=0.001875`, `0.0009375`, and min `dt=0.000625`; a repeated min-step retry
  was manually terminated after `3439.06 s`. Cut diagnostics still reach the
  `1e-8` active-fraction class with capped cut-adjacent scaling `15`. The
  partial pressure comparison is only a nominal-index diagnostic because the
  manually stopped adaptive run did not produce `result.pvd`; it covers about
  `0.109` of the reference horizon and still ends far before the `7.3032 s`
  pressure peak.
- The repeated min-step retry exposed a `SimpleStepController` issue that is
  now covered by FE time-stepping unit tests: a failed attempt already at
  configured `min_dt` stops instead of requesting another identical retry,
  while a larger failed step can still clamp once to the floor. A post-fix
  rerun of the same relaxed Test10 case verifies the behavior in the real
  solver path: it accepts 91 outputs to `0.9012500000000006 s`, rejects
  `dt=0.001875`, `0.0009375`, and one min-`dt=0.000625` floor attempt, then
  returns failure with no repeated floor start. Its reconstructed-time pressure
  comparison still covers only `0.10793413173652702` of the `8.35 s` horizon
  and ends before the `7.3032 s` full-record pressure peak. This avoids wasting
  future CPU at the same failure state but does not resolve the moving
  tiny-cut/topology nonlinear blocker.
- The accepted-tail active-topology audit for that post-fix rerun
  (`test10_roll_full_source_dt0p01_tightvol_adaptive_relaxed_ls_max20_1s_post_controller_fix_active_topology_audit_20260604.json`)
  shows the saved accepted states are not already tiny-cut polluted: from
  `0.82` to `0.9012500000000006 s`, no accepted cut fraction is below `1e-2`,
  the final accepted minimum cut wet fraction is `0.04468444308525042`, and
  ActiveFluid matches the `phi` sign. The tiny `1e-8` active fractions appear
  in the next rejected trial context, where the final residual is dominated by
  `Velocity[1]` (`0.276595`) while pressure and level-set residuals are in the
  `1e-6` class. This narrows the Test10 robustness issue to the nonlinear
  update through moving trial topology after a clean accepted output.
- An always-on PTC gamma=1.0 control
  (`test10_roll_full_source_dt0p01_tightvol_adaptive_relaxed_ls_max20_ptc_g1_1s_failure_summary_20260604.json`)
  is a negative robustness control, not a closure path. It accepts 87 outputs
  to `0.8700000000000006 s` and avoids the earlier recovered full-step
  failures, but then rejects `dt=0.01`, `0.005`, `0.0025`, `0.00125`, and
  `0.000625` from the same state. The residual worsens from
  `0.02295120910918987` to `1.356848111438981` despite converged linear
  solves, the final residual remains velocity dominated, and the pressure
  comparison still covers only `0.10419161676646707` of the `8.35 s` horizon.
  This fails earlier than the post-controller-fix relaxed non-PTC path.
- A rho=`0.0` high-frequency damping control on the same tight-volume adaptive
  relaxed-line-search/max-20 setup
  (`test10_roll_full_source_dt0p01_tightvol_adaptive_relaxed_ls_max20_rho0_1s_failure_summary_20260604.json`)
  is also negative evidence. It accepts 77 full `dt=0.01 s` outputs to
  `0.7700000000000005 s`, but then rejects `dt=0.01`, `0.005`, `0.0025`,
  `0.00125`, and `0.000625` from the same state. The residual worsens from
  `0.02366475833925794` to `1.8452328378954477` despite converged linear
  solves. The accepted tail has no cut fractions below `1e-2`, but the failed
  trial reaches `active_min_volume_fraction=1.0049665284769772e-08`, capped
  cut-adjacent scaling `15`, and a `Velocity[1]`-dominated residual
  (`1.71444`). Its pressure comparison covers only `0.09221556886227546` of
  the reference horizon and underpredicts the sampled-window peak. This fails
  earlier than both the PTC gamma=1.0 control and the post-controller-fix
  relaxed rho=`0.5` path, so high-frequency damping alone is not the long
  Test10 pressure-history route.
- A generated-volume pruning diagnostic on the relaxed rho=`0.5` setup
  (`test10_roll_full_source_dt0p01_tightvol_adaptive_relaxed_ls_max20_prune1e7_1s_failure_summary_20260604.json`)
  raises `SVMP_MIN_GENERATED_CUT_VOLUME_FRACTION` from the default `1e-8` to
  `1e-7`. It activates in the old moving-cut window and prunes up to `8`
  generated volume rules (`7.332969322430679e-08` max generated pruned volume),
  but it accepts only to `0.9006250000000006 s`. After the full step to
  `0.9000000000000006 s`, it rejects `dt=0.01`, `0.005`, `0.0025`, and
  `0.00125`, accepts only the `0.000625` floor substep, then rejects
  `0.0009375` and the `0.000625` floor from the next state. The accepted tail
  now contains cut fractions below `1e-6`, and the final residual remains
  velocity dominated. This passes the rho=`0.0` and PTC gamma=1.0 endpoints but
  does not improve on the post-controller-fix relaxed rho=`0.5` path, so the
  threshold remains a diagnostic knob rather than a validation setting.
- A cut-metadata scale cap-3 control on the same relaxed rho=`0.5` setup
  (`Use_cut_metadata_scale=true`, `Cut_cell_metadata_scale_cap=3.0`) accepts to
  `0.9006250000000006 s`, equal to the prune1e-7 diagnostic and short of the
  post-controller-fix relaxed endpoint at `0.9012500000000006 s`. The cap
  engages in the moving-cut trial window (`15` capped cut-adjacent scales;
  minimum failed-trial active fraction `1.0066439128659922e-08`) and lowers the
  final rejected residual to `0.3917555895182002`, but the final residual
  remains velocity dominated (`Velocity[1]` norm `0.36379`) with converged
  linear solves. Accepted outputs remain clean at the `1e-6` tiny-cut threshold,
  and pressure coverage remains only `0.10785928143712582` of the reference
  horizon. The focused FE form-kernel test
  `FormKernelDGTest.CutStabilizationScaleCanBeCappedByFormMinimum` verifies that
  `min(cutStabilizationScale(), 3)` assembles with capped eta `3` instead of
  raw eta `4`, so metadata scaling at cap 3 is applied but remains
  diagnostic-only, not a Test10 pressure-history closure.
- A lower-floor follow-up of the same cap-3 control
  (`Adaptive_time_loop_min_dt=3.125e-4`) accepts one extra floor substep to
  `0.9009375000000006 s`, proving the prior cap-3 endpoint was partly
  timestep-floor-limited. It still fails immediately at the new `0.0003125 s`
  floor with residual `0.12231052082324947`; the final rejected residual is
  pressure dominated (`Pressure` norm `0.122309`) while the failed trial still
  reaches `active_min_volume_fraction=1.033936438598603e-08` and `17` capped
  cut-adjacent scales. Accepted outputs remain clean with no tail cuts below
  `1e-2`, final accepted minimum cut wet fraction
  `0.04467491901750112`, and pressure coverage only
  `0.10789670658682643` of the reference horizon. This is a floor-sensitivity
  diagnostic, not closure.
- A Test10 accepted-pressure jump audit now distinguishes clean accepted
  topology from accepted Sensor1 pressure-path accuracy. The post-controller
  relaxed run already ends with an accepted `229.50083676601784 Pa` Sensor1
  jump over `0.00125 s`, reaching `401.9619413011071 Pa` while the interpolated
  SPHERIC pressure is `341.75482299992603 Pa`. The cap-3 run accepts a larger
  `495.1194649955057 Pa` jump to `667.4255522836413 Pa`, and the lower-floor
  cap-3 follow-up accepts another `410.3844795884645 Pa` jump to
  `1077.8100318721058 Pa` while the interpolated reference is only
  `329.43034799982917 Pa`. Solver-log context shows those jumps were accepted
  in one Newton iteration with residuals below the `0.02` absolute tolerance
  (`0.001460655629344565`, `0.0009696672726394497`, and
  `0.0005735220385357919`), after larger trial timesteps from the same states
  were rejected. The accepted attempts have clean before-solve and
  accepted-trial active cut contexts (`active_min_volume_fraction` pairs of
  `0.04477859715795944/0.044685136860851606`,
  `0.04474385617252349/0.044697352262951316`, and
  `0.044697036173589735/0.0446751045174449`), while the accepted Newton update
  raises the pressure-state norm/max by `956.0699999999997/342.67999999999984 Pa`,
  `2656.6400000000012/744.47 Pa`, and `3815.7999999999993/1462.69 Pa`.
  Because those jumps occur in saved accepted outputs whose tail topology has
  no tiny cut fractions below `1e-2`, the next Test10 limiter is accepted-step
  pressure/time-integration behavior in the moving-cut window, not stale
  sampling, the subsequent failed retry, loose nonlinear acceptance, or mesh
  resolution.
- A one-step Test10 replay harness,
  `tests/cases/fluid/open_vessel_free_surface/prepare_test10_one_step_replay.py`,
  uses the OOP driver's general `<Start_time>` support to initialize from the
  cap-3 `result_090.vtu` field state at `0.9 s` and replay one fixed
  `0.000625 s` substep at the original forcing time. This is diagnostic-only
  because older transient history is not reconstructed, but it reproduces the
  saved cap-3 `result_091.vtu` state with `Min_iterations=1` to
  `0.34685523461621415 Pa` pressure L-infinity difference. Increasing to
  `Min_iterations=2` keeps Sensor1 high (`659.5975709522717 Pa` versus
  `341.0886229999867 Pa` reference) and raises pressure max to
  `2161.802771068498 Pa`. A strict `SVMP_NEWTON_ABS_TOLERANCE=1e-8`,
  `SVMP_NEWTON_REL_TOLERANCE=0` replay rejects the step after `20` Newton
  iterations with no result output; its best residual is `9.27507e-06` and the
  iterates stay at pressure max `2161.8 Pa`. Min-iteration-only or scalar
  tolerance-only changes are therefore not the Test10 pressure-path fix.
  Disabling only the free-surface cut-cell pressure stabilization policy on the
  same replay accepts in one Newton iteration, lowers Sensor1 to
  `386.71282017258176 Pa` versus `341.0886229999867 Pa` reference, and reduces
  the pressure increment from `result_090.vtu` to `471.20065581788583 Pa`
  L-infinity versus `1075.2113565356985 Pa` for the saved cap-3 step. That is
  diagnostic-only, not a production fix, but it points the next Test10 pressure
  investigation at the free-surface pressure-stabilization branch. The scalar
  penalty sweep is not a simple retune path: zero penalty is identical to the
  disabled policy, `0.25` and `0.5` stay on the baseline high-pressure branch
  with Sensor1 near `667.7 Pa`, and `1e-4` worsens Sensor1 to
  `1024.8414617639164 Pa`.
  Summaries: `test10_replay_cap3_step90_min_iteration_summary_20260604.json`
  and `test10_replay_cap3_step90_abs1e8_rel0_failure_summary_20260604.json`;
  pressure samples include
  `test10_replay_cap3_step90_pressure_disabled_20260604_pressure_20260604.json`.
- The extended Test10 scalar pressure-penalty replay sweep confirms that the
  accepted cap-3 pressure branch cannot be fixed by a one-coefficient retune.
  Penalty `0.001` still overshoots Sensor1 to `751.2793952293686 Pa`, and
  `0.01`, `0.05`, and `0.1` remain on the same high-pressure branch as the
  larger enabled penalties. The disabled/zero branch is the only replay branch
  that substantially reduces the accepted pressure jump, but the 1 s
  zero-penalty transient underpredicts the sampled-window pressure peak. The
  runtime incremental pressure-stabilization policy is now implemented: FE
  field-history gradients/jacobians and JIT field-history spatial jets support
  `grad(effectiveDt*dt(Pressure))`, and the full-history rerun no longer emits
  the previous cached-face JIT fallback. The one-step replay runs but remains
  diagnostic because it lacks reconstructed pressure history. The full
  transient with true history accepts `99` outputs to `0.9900000000000007 s`,
  then rejects `dt=0.01`, `0.005`, `0.0025`, `0.00125`, and min `0.000625`,
  exiting with `success=0`. Its accepted `0.01-0.99 s` pressure comparison is
  still `not_validation_ready`: RMSE `103.41981541681686 Pa`, sampled
  simulated/reference peaks `166.2144728913598/357.743423 Pa`, and only
  `0.11856287425149702` reference coverage. Evidence:
  `test10_replay_cap3_step90_pressure_penalty_extended_sweep_20260604.json`,
  `test10_replay_cap3_step90_pressure_incremental_20260604_case/run.log`,
  `test10_roll_full_source_dt0p01_tightvol_adaptive_relaxed_ls_max20_metadata_cap3_incremental_1s_case/run.log`,
  `test10_roll_full_source_dt0p01_tightvol_adaptive_relaxed_ls_max20_metadata_cap3_incremental_1s_failed0p99_pressure_comparison_20260604.json`,
  and
  `test10_roll_full_source_dt0p01_tightvol_adaptive_relaxed_ls_max20_metadata_cap3_incremental_1s_active_topology_audit_20260604.json`.
- A full transient pressure-zero control on the same cap-3 adaptive setup
  changes only `<Cut_cell_pressure_gradient_penalty>` to `0.0` and completes
  all `100` outputs to `1.0 s` with full `0.01 s` steps, zero rejected trials,
  all linear solves converged, and max Newton iterations `2`. It crosses the
  old enabled-pressure cap-3 barrier at `0.900625 s` without the retry ladder
  and keeps the accepted tail coherent: zero ActiveFluid/`phi` mismatches, no
  cut fractions below `1e-6`, and final wet-volume drift
  `-1.1567637870427226e-08`. The pressure history removes the old Sensor1
  overshoot but remains diagnostic, not closure: over `0.01-1.0 s`, RMSE is
  `88.45895351655469 Pa`, max abs error is `173.48676306595038 Pa`, and the
  simulated/reference sampled peaks are
  `225.53223407556905/357.74342299992605 Pa` at `0.85 s`. Coverage is only
  `0.11976047904191617` of the SPHERIC record and the run ends before the
  published `7.3032 s` pressure peak, so Test10 still needs a corrected
  pressure-stabilization treatment plus long-horizon convergence/refinement
  evidence.
  Evidence:
  `test10_roll_full_source_dt0p01_tightvol_adaptive_relaxed_ls_max20_metadata_cap3_pressure0_1s_summary_20260604.json`,
  `test10_roll_full_source_dt0p01_tightvol_adaptive_relaxed_ls_max20_metadata_cap3_pressure0_1s_pressure_comparison_20260604.json`,
  and
  `test10_roll_full_source_dt0p01_tightvol_adaptive_relaxed_ls_max20_metadata_cap3_pressure0_1s_active_topology_audit_20260604.json`.
- The fixed-tank roll-force model audit now reports
  `coriolis_configured=true` and
  `full_roll_force_input_history_available=true`: OOP Navier-Stokes can use
  the generated omega history to add `-2 omega x u`, and the checked body-force
  plus omega inputs now cover the full `8.35 s` SPHERIC record with `0.01 s`
  source sampling. Over the full record, `2|omega|` reaches
  `0.5402230716301298 1/s`; at sloshing velocities near
  `2.5049093273619136 m/s`, that velocity-dependent term is comparable to the
  largest generated static roll-frame acceleration. In the refreshed 40-step
  smoke, observed velocities bound the Coriolis acceleration by
  `4.103394054718822e-05 m/s^2`.
- The checked structured 819-point/2880-tet unfitted mesh no longer has the
  earlier tiny initial free-surface cuts: minimum active cut fraction is
  `0.12499999999999968`, no cut fractions are below `1.0e-2`, and run-time cut
  diagnostics retain minimum active cut fraction `0.12499811111525794` with no
  capped cut-adjacent scaling in the 40-step smoke; the 100-step extensions show
  the old drift gate needs tight volume correction and moving tiny cuts can still
  become the next limiter. The relaxed-line-search/max-20 probe shows the
  previous `0.90 s` barrier is partly recoverable, but not enough for a
  peak/horizon validation run. The min-`dt` controller fix now stops
  deterministically at the floor in the post-fix solver rerun, and the
  accepted-tail topology audit shows the saved state is clean while the next
  trial creates tiny/pruned cut topology. The always-on PTC gamma=1.0 control
  also fails before the clean post-fix non-PTC endpoint, and the `1e-7`
  generated-volume pruning diagnostic and cut-metadata cap-3 control change the
  route without advancing past that endpoint. This clarifies, but
  does not change,
  the physical/numerical conclusion. Test10 closure therefore needs strict volume
  correction plus moving tiny-cut/topology and nonlinear robustness for a long
  pressure-history transient using the now-available full-duration body-force
  and angular-velocity histories,
  temporal-convergence evidence through the pressure peak and horizon, pressure
  convergence/refinement gates, and
  qualification of the Coriolis-capable fixed-frame source path or a full
  roll/ALE wall-motion model.

## Capillary Wave Oscillation

Purpose: validate surface-tension curvature and pressure-jump coupling.

Setup:

- small sinusoidal interface perturbation with wavelength `lambda`
- gravity disabled for a pure capillary check, or enabled for a capillary-gravity
  check
- low viscosity for the oscillation-frequency check
- amplitude `a / lambda <= 0.01`

Reference frequency:

```text
k = 2 pi / lambda
omega^2 = gamma k^3 / rho
```

For finite depth or gravity, use:

```text
omega^2 = (g k + gamma k^3 / rho) tanh(k h)
```

Measured outputs:

- interface amplitude over time
- pressure jump across the interface
- curvature diagnostics

Acceptance criteria:

- dominant frequency within two percent of the reference value
- measured pressure jump matches `gamma kappa` in phase with curvature
- curvature and interface normals converge under refinement

## Capillary Wave Decay

Purpose: validate viscous damping of small capillary waves.

Setup:

- same geometry and perturbation as the capillary oscillation case
- viscosity enabled
- small amplitude to remain in the linear regime

Deep-water reference decay rate:

```text
eta_amplitude(t) = eta_amplitude(0) exp(-2 nu k^2 t)
nu = mu / rho
```

Measured outputs:

- logarithmic decrement of interface amplitude
- oscillation frequency
- kinetic plus surface-energy trend

Acceptance criteria:

- fitted exponential decay rate within five percent of `2 nu k^2`
- decay-rate error decreases under time-step and mesh refinement
- total energy decreases monotonically after the initial transient

## Dam Break Or Water-Column Collapse

Purpose: validate large free-surface deformation after the methods are stable on
small-amplitude cases.

Setup:

- rectangular water column of initial width `a` and height `H`
- dry or low-density void region represented by the free surface
- no surface tension for the standard gravity-collapse comparison
- wall no-slip or slip policy recorded with the result

Reference behavior:

- early-time front location compared with the Martin-Moyce water-column collapse
  benchmark or an equivalent published data set
- splash and impact timing compared qualitatively until a higher-resolution
  reference is established

Measured outputs:

- leading-front position versus time
- free-surface profile snapshots
- volume drift
- maximum velocity and time-step history

Acceptance criteria:

- nondimensional front-position curve follows the selected benchmark envelope
- qualitative splash timing is stable under refinement
- no negative volume or interface loss occurs in the level-set representation

### SPHERIC Test 05 Wet-Bed Dam Break

Starting points:

- `tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test05_wet_bed_d18/solver.xml`
- `tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test05_wet_bed_d38/solver.xml`

Setup:

- initial dam height `d0 = 0.15 m`
- wet-bed depths `d = 0.018 m` and `d = 0.038 m`
- thin three-dimensional extrusion of the published two-dimensional setup
- level-set field initialized as the union of the wet bed and retained water
  column
- negative level-set values denote the water side
- fluid volume terms use `Active_domain=LevelSetNegative` and
  `Active_domain_method=CutVolume`
- validation-grade unfitted runs use level-set volume correction with cadence
  `1`, tolerance `1.0e-10`, reinitialization disabled, and
  `nearest_interface_point` wet-extension advection velocity
- `Active_domain_method=SmoothedIndicator` is available only as a diagnostic
  fallback. If `Active_domain_smoothing_width` is omitted or set to zero, the
  transition width is derived from local cell diameter; accepted D18/D38 runs
  continue to require exact cut-volume integration.

Reference behavior:

- leading-front velocity over the first meters of travel
- free-surface profiles in the published digitized snapshot window

2026-06-02 validation-grade evidence:

- D18 passed the first profile-time run at `0.156 s` with profile RMSE
  `0.0183929 m`, max absolute profile error `0.0602431 m`, max relative
  wet-volume drift `4.61585e-08`, zero `ActiveFluid` mask mismatches, and zero
  out-of-range `WetVolumeFraction` cells.
- D38 passed the first profile-time run at `0.156 s` with profile RMSE
  `0.0157573 m`, max absolute profile error `0.0497431 m`, max relative
  wet-volume drift `3.77873e-08`, zero `ActiveFluid` mask mismatches, and zero
  out-of-range `WetVolumeFraction` cells.
- Reports:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/validation_grade_volume_corrected_20260602_d18_target6/test05_validation_grade_report.json`
  and
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/validation_grade_volume_corrected_20260602_d38_target6/test05_validation_grade_report.json`.

### SPHERIC Test 02 Three-Dimensional Dam Break With Obstacle

Starting point:

- `tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test02_dambreak_obstacle/solver.xml`
- geometry/resolution audit:
  `tests/cases/fluid/open_vessel_free_surface/audit_test02_geometry_resolution.py`
- front-speed diagnostic:
  `tests/cases/fluid/open_vessel_free_surface/analyze_spheric_test02_front.py`

Setup:

- tank `3.22 m x 1.00 m x 1.00 m`
- initial water column length `1.228 m`, height `0.55 m`, width `1.00 m`
  at the right end of the tank, with gate at `x = 1.992 m`
- fixed obstacle with flow-direction length `0.161 m`, lateral width
  `0.403 m`, and height `0.161 m`
- obstacle bounds `x=[0.6635,0.8245]`, `z=[0.2985,0.7015]`
- height probe x-locations H1/H2/H3/H4 =
  `0.496/0.992/1.488/2.632 m`

Reference behavior:

- impact time near the obstacle
- water-height gauge histories
- pressure response on the obstacle face
- splash height and reflected-wave timing

2026-06-03 status:

- The checked mesh now uses the official retained-column plane level set
  `phi = max(1.992 - x, y - 0.55)`, which removes the erroneous
  wall-contact zero-`phi` cuts from the older closed-box signed-distance
  initialization.
- Earlier repository metadata used a `0.58 m` retained column and swapped the
  obstacle flow/lateral footprint. Those artifacts are superseded because they
  are not the SPHERIC Test02/Kleefsman benchmark geometry.
- The checked 6074-point/30096-tet structured fixture aligns tank and obstacle
  faces while centering the top/gate free-surface cuts between mesh planes. Its
  initial volume error is `-0.0011844832691883223`, it has no zero-`phi`
  nodes, no wall-contact zero nodes, no cut fractions below `1.0e-2`, and
  minimum active cut fraction `0.125`.
- The corrected official-geometry fixture completed twelve accepted steps to
  `t=0.012` with max relative wet-volume drift `9.76861310161081e-11`, final
  relative drift `8.807314004495764e-11`, minimum active cut fraction
  `0.11721605717874085`, one Newton and one linear iteration per step, and no
  capped/zero cut-adjacent scales.
- A coarse official-geometry `h=0.20` pilot completed to `t=0.5` with `251`
  accepted outputs, one recovered adaptive rejection at `t=0.45`, and max
  relative wet-volume drift `1.4743643193918593e-10`. It reaches the H2/P1/P3
  first-impact reference times but is not validation-grade: moving cuts reach
  active fractions as small as `1.0019378034581306e-08` with up to `120`
  capped cut-adjacent scales, H2 remains dry, and P1/P3 sampled pressures
  remain `0 Pa` through `0.5 s`. The zero-contour front is only
  `0.614363304456075 m` from the gate by `0.5 s`, giving an average front
  speed `1.22872660891215 m/s`; this is `0.43498471108958275` of the speed
  implied by the official H2 first-response time and `0.26449009902996995` of
  the dry-bed Ritter `2 sqrt(g h)` estimate.
- A fine official-geometry `h=0.10` all-solid normal-only first-impact run
  completed to `t=0.5` with `502` accepted outputs, `2` recovered rejections,
  and max relative wet-volume drift `1.504304471613901e-10`. It reaches H2/P1/P3
  but remains delayed: H2 first exceeds `0.005 m` at
  `0.4052500000000003 s`, P1 first exceeds `100 Pa` at
  `0.4302500000000003 s`, and P3 first exceeds `100 Pa` at
  `0.4552500000000003 s`, versus official `0.3540126077962089 s`,
  `0.3930264600370048 s`, and `0.4220221953556109 s`. This is still not a
  Test02 validation pass because the run covers only `0.06761905013659021` of
  the reference horizon, misses H2 peak and P5/P7 response/peak windows, and
  underpredicts P3 pressure over the `0.5 s` window.
- A matching official-geometry `h=0.15` all-solid normal-only first-impact
  control completed to `t=0.501` with `337` accepted outputs, `4` recovered
  rejections, max relative wet-volume drift `1.4829656756795768e-10`, and
  final front travel `1.3307321880608798 m`. It also reaches H2/P1/P3 but
  remains late: H2 first exceeds `0.005 m` at `0.4132500000000003 s`, P1 first
  exceeds `100 Pa` at `0.4372500000000003 s`, and P3 first exceeds `100 Pa` at
  `0.4530000000000003 s`. Compared with the `h=0.10` first-impact run,
  refining changes final front distance by less than `1%` and improves H2/P1
  first response by only about `0.008/0.007 s`; the late response and P3
  underprediction persist, so the first-impact inaccuracies are not explained
  by `h=0.15` versus `h=0.10` mesh resolution alone.
- A pressure-sampling sensitivity audit on the `h=0.15` and `h=0.10`
  first-impact outputs samples official sensor points, fluid-side offsets,
  nearest nodes, and local node maxima. It does not rescue the pressure
  histories: the strongest nearby P3 sample is only `0.146` of the
  reference-window peak at `h=0.15` and `0.156` at `h=0.10`, P5 remains an
  early false response, and P7 remains zero through the `0.5 s` window. The
  remaining pressure mismatch is therefore not an exact sensor-placement
  artifact.
- An obstacle-face dynamics audit samples interpolated vertical `phi` columns
  next to the obstacle and the pressure stack at P1/P3/P5/P7. The h0.10/h0.15
  first-impact outputs stopped just before P3-height wetting, but the h0.15
  `0.54 s` run-up extension crosses face-adjacent P3 height at `0.507 s` and
  still has an inaccurate pressure stack: P1 peaks at `73314.24327873143 Pa`, P3 reaches
  `4323.8019247570555 Pa`, P5 responds early at `3320.6381428457175 Pa`, and
  P7 remains zero. A pressure-extrema scan also finds a separate tiny-cut
  pressure spike: the largest active/wet pressure is `1320026.108140017 Pa` on
  incident cells with WetVolumeFraction max `1.8313270361947403e-06`, while the
  official P1 local wet peak occurs on fully wet cells. The P3/P1 peak ratio is
  `0.05897628798156602` versus a
  reference-window ratio of `0.5840110088913406`. A pressure-only `10x`
  ghost-penalty control completed to `0.522 s`, but it damped P1/P3/P5 target
  peaks to `0.262/0.282/0.192x` of the unit-penalty run while increasing the
  active/wet tiny-cut maximum to `4482180.476525261 Pa` and the tiny-cut wet
  spike count from `3` to `16`. Test02 therefore needs pressure/run-up
  correction and tiny-cut pressure control beyond a scalar pressure-penalty
  increase, rather than a pressure-sensor placement change.
- A no-cut-stabilization perturbation on the same coarse setup did not produce
  a speed comparison: with `Enable_cut_cell_stabilization=false`, step 0 was
  repeatedly rejected as the adaptive time step fell from `0.002` to
  `1.5625e-5` and then prepared a `7.8125e-6` retry before termination. This
  shows cut stabilization is needed for coarse-startup robustness, but it does
  not by itself explain the stabilized slow-front result.
- A pressure-gradient cut-cell stabilization off control on the same `h=0.20`
  short setup kept velocity stabilization enabled and set
  `Cut_cell_pressure_gradient_penalty=0.0`. It completed to `t=0.2` with
  `100` accepted outputs and no rejected steps, but it is nonphysical and does
  not improve the front: the zero-contour front travels only
  `0.13280404722690564 m`, which is `0.7944314193042276x` of the post-fix
  `h=0.20` travel and `0.5833183824677339` of the `h=0.15` travel. The
  pressure field becomes unusable: solution-state pressure extrema reach
  `2055270.0 Pa`, pressure norm reaches `7789780.0`, and the wet-vertex
  pressure diagnostic reaches `737323.0331033937 Pa`. This rules out disabling
  pressure-gradient ghost stabilization as an accuracy fix and shows it is
  required for a meaningful coarse Test02 calculation.
- A velocity-gradient cut-cell stabilization off control on the same `h=0.20`
  short setup kept pressure stabilization enabled and set
  `Cut_cell_velocity_gradient_penalty=0.0`. It produced no speed comparison:
  step 0 was repeatedly rejected from `dt=0.002` down to `3.125e-5` before
  operator termination, with each recorded nonlinear attempt reaching `12`
  Newton iterations and residual norms near `98`. This rules out disabling the
  velocity-gradient ghost penalty as a speed fix and shows it is also required
  for coarse-startup robustness.
- A no-fluid-velocity-extension control on the same `h=0.20` short setup
  disabled the Navier-Stokes free-surface `Enable_velocity_extension` PDE while
  keeping the level-set prescribed wet-extension controls. It completed to
  `t=0.2` with `100` accepted outputs, no rejected steps, max relative
  wet-volume drift `1.6351379198947617e-10`, and no empty/nonfinite/negative
  active quadrature regions, but it still did not close the front-speed deficit:
  the zero-contour front is `0.16846863412857038 m` from the gate, only
  `0.0012999597787857198 m` farther than the post-fix `h=0.20` probe and only
  `0.7399687977017979` of the `h=0.15` front travel. This rules out the fluid
  velocity-extension PDE as a simple one-switch cause of the lag.
- A code audit found and fixed a separate `nearest_interface_point`
  wet-extension error: active/wet vertices now preserve the solved source
  `Velocity`, and only dry vertices receive nearest-interface samples. The
  post-fix `h=0.20` short probe completed to `t=0.2` with `100` accepted
  outputs, no rejected steps, and max relative wet-volume drift
  `1.4618713254516288e-10`, but it does not close the front-speed deficit. At
  `0.2 s`, the zero-contour front is `0.16716867434978466 m` from the gate,
  only `0.00290580252806349 m` farther than the prior `h=0.20` run and only
  `0.7342589533765056` of the `h=0.15` front travel. Moving cuts still reach
  active fractions as small as `1.1334255565101966e-08` with up to `128`
  capped cut-adjacent scales. The companion front/advection diagnostic reports
  `99` paired zero-contour/leading-edge samples; for `t>=0.10 s`, front speed
  minus `-mean(vx)` has RMSE `0.10366315254546861 m/s`, and at `t=0.2 s` the
  mismatch is `0.0010844848200235457 m/s`.
- A boundary-condition audit found that zero velocity Dirichlet walls ignored
  `Effective_direction`; the Navier-Stokes parser/factory path now preserves
  partial-component strong constraints. A cloned `h=0.20` control constrained
  only the normal velocity component on the rectangular tank walls while keeping
  the composite obstacle marker full-vector no-slip. Initial constraints dropped
  from `2446` to `1602`, confirming the mask was active. The run completed to
  `t=0.2` with `101` accepted outputs, one recovered rejection, total loop time
  `436.94628 s`, and max relative wet-volume drift
  `1.441241252278775e-10`. The zero-contour front traveled
  `0.23768091845512385 m`, `1.4218029746278837x` the post-fix no-slip `h=0.20`
  travel and `1.0272171041105413x` the `h=0.15` distance, but still only
  `0.9326778712471756x` the `h=0.10` no-slip distance. It also still requires
  `4.9497186785745795 m/s` average continuation speed to reach H2 by the
  official first-response time, so tank-wall no-slip is a contributor, not a
  complete Test02 fix.
- A follow-up `h=0.20` all-solid normal-only control split the obstacle into
  five normal-aligned faces and applied normal-only strong velocity constraints
  on the tank and obstacle. This removed only twelve additional constraints
  relative to the tank-wall control (`1590` total) and produced the same
  `0.23768091845512385 m` front travel/history metrics through `0.2 s`.
  Obstacle tangential no-slip is therefore not a meaningful short-time speed
  fix; the tank-wall constraint effect is real but still not a Test02 closure.
- The same all-solid normal-only treatment on the structured `h=0.15` mesh
  reached `0.29996397852897627 m` by `0.201 s`, with `136` accepted outputs
  and two recovered rejections. At the nearest `0.2 s` sample it traveled
  `1.3088669354705278x` the `h=0.15` no-slip distance and
  `1.1693337914065964x` the short `h=0.10` no-slip distance, so the remaining
  short-time lag is not mesh resolution alone. It still leaves H2/P1 dry/zero
  and covers only `0.5283135835023329` of the H2-average distance by the final
  time, so it remains an open Test02 accuracy item.
- The same all-solid normal-only treatment on the structured `h=0.10` mesh
  completed to `t=0.2` with `200` accepted outputs and no rejected steps. It
  moved the front to `0.3112749578475953 m`, `1.2214664388915353x` the
  `h=0.10` no-slip distance and `1.0445832044434704x` the `h=0.15`
  normal-only nearest-`0.2 s` sample. It still is not validation-grade: H2/P1
  remain zero, the run covers only `0.02704584051748816` of the reference
  horizon, and the front reaches only `0.5509762978464111` of the H2-average
  distance at `0.2 s`.
- A tight fluid nonlinear-tolerance control on the same `h=0.20` post-fix
  setup changed only the fluid equation tolerance from `2.0e-2` to `1.0e-4`.
  It increased the total loop time from `652.640588 s` to `1145.181662 s` and
  raised the mean nonlinear iteration count to `2.13`, but the front did not
  improve: final `0.2 s` travel was `0.16713406395912145 m`, which is
  `-3.4610390663214474e-05 m` relative to the post-fix baseline
  (`0.9997929612662311x`). The moving-cut tiny/capped class persists with
  minimum active fraction `1.0589217906541503e-08` and up to `128` capped
  cut-adjacent scales. This rules out loose fluid nonlinear convergence as the
  missing short-time accuracy fix.
- A rho=`1.0` time-integration damping control on the same `h=0.20` post-fix
  setup changed only `Spectral_radius_of_infinite_time_step` from `0.50` to
  `1.0`. It was less robust, not faster: the run accumulated `110` accepted and
  `40` rejected steps, reached only `0.11285998488620348 s` after `1050.33 s`
  wall time, and was stopped once `dt` had collapsed to
  `0.00010032218247106267 s`. Using accepted-step times from the solver log,
  the zero-contour front had traveled `0.05740606435139961 m`, only
  `0.8368462282340826x` of the rho=`0.5` post-fix baseline interpolated to the
  same physical time. This rules out generalized-alpha damping as a simple
  speed fix for the Test02 short-time lag.
- An intermediate official-geometry `h=0.15` resolution/front-speed probe
  completed to `t=0.201` with `134` accepted outputs, no rejected steps, max
  relative wet-volume drift `1.4811487675574508e-10`, and no
  empty/nonfinite/negative active quadrature regions. Refinement materially
  increases early front travel: at `0.2 s`, the zero-contour front is
  `0.22766991615295384 m` from the gate versus
  `0.16426287182172117 m` for `h=0.20`, a `1.386009593208927x` increase. This
  proves mesh resolution contributes to the slow-front error, but it does not
  close Test02: the front is still only `0.23138333415985102 m` from the gate
  at `0.201 s`, develops active fractions as small as
  `1.0049931766836533e-08` with up to `158` capped cut-adjacent scales, and
  would need about `5.023224405559033 m/s` over the remaining time to reach H2
  by the official first-response time. Its front/advection diagnostic reports
  `133` paired samples; for `t>=0.10 s`, the relative speed-error mean is
  `-0.0247309880354786`, and integrated leading-edge velocity underpredicts
  actual front travel by only `0.014317935386805536 m`
  (`-0.06187971765033864` relative).
- A fine official-geometry `h=0.10` short resolution/front-speed probe
  completed to `t=0.12` with `120` accepted outputs, no rejected steps, wall
  time `4738.58 s`, and max relative wet-volume drift
  `1.4736589588040205e-10`. It is not validation-grade: the run covers only
  `0.016226317952394282` of the reference horizon, H2 and P1 remain zero at
  `0.12 s`, moving active fractions reach `1.0099060868310082e-08`, capped
  cut-adjacent scale count reaches `234`, and ActiveFluid/WetVolumeFraction
  disagreement warnings reach `52` accepted outputs with up to `24` disagreeing
  cut cells. Refinement still increases early front travel: at `0.12 s`, the
  zero-contour front is `0.09874079742431618 m` from the gate versus
  `0.07579994904994947 m` for post-fix `h=0.20` and
  `0.09299087238311765 m` for `h=0.15`. The `t>=0.10 s` front/advection RMSE is
  `0.07953644355596523 m/s`, and the `0.12 s` front speed differs from
  `-mean(vx)` by only `0.027785981550120464 m/s`.
- The same official-geometry `h=0.10` probe was extended to `t=0.2`. It
  completed with `200` accepted outputs, no rejected steps, wall time
  `7631.07 s`, and max relative wet-volume drift `1.4760387134639789e-10`.
  It is still not validation-grade: H2 and P1 remain zero at `0.2 s`, the run
  covers only `0.02704584051748816` of the reference horizon, moving active
  fractions reach `1.0099060868310082e-08`, capped cut-adjacent scale count
  reaches `234`, and active-region warnings reach `132`. At `0.2 s`, the
  zero-contour front is `0.254837094116211 m` from the gate, which is
  `1.524430908526489x` the post-fix `h=0.20` distance and
  `1.113274343551294x` the `h=0.15` distance. It is still only
  `0.45107772125643897` of the official H2-average distance at that time, and
  the front would need `4.83832406026003 m/s` average continuation speed to
  reach H2 by the official first-response time.
- `test02_front_closure_resolution_audit_20260602.json` combines the front
  diagnostics without another solver run. At the like-for-like `0.2 s` sample,
  `h=0.15` travels `1.361917339109695x` as far as the post-fix `h=0.20`
  probe, so resolution contributes. It still does not close the benchmark:
  the `h=0.15` sample would need `4.9984923228122256 m/s` average continuation
  speed to reach H2 by `0.3540126077962089 s`, which is
  `1.075952713868242x` the dry-bed Ritter speed, while the sampled front speed
  is only `0.49140142379382556` of that required continuation speed.
- `test02_front_closure_h0p10_0p12_audit_20260603.json` adds a three-mesh
  comparison at `0.12 s`. At that time, `h=0.10` travels
  `1.3026499181318647x` as far as post-fix `h=0.20`, but only
  `0.2912957266004947` of the distance implied by the official H2 average
  speed. From the `h=0.10` sample, the front still needs
  `3.8513275462514827 m/s` average continuation speed to reach H2 by
  `0.3540126077962089 s`, while the sampled `h=0.10` front speed is only
  `0.40130191021278516` of that required speed. This confirms resolution
  contributes but does not close the SPHERIC timing gap.
- `test02_front_closure_h0p10_0p2_audit_20260603.json` adds the like-for-like
  three-mesh comparison at `0.2 s`. At that time, `h=0.10` travels
  `1.524430908526489x` as far as post-fix `h=0.20` and
  `1.113274343551294x` as far as `h=0.15`, but only
  `0.45107772125643897` of the distance implied by official H2 average speed.
  From the `h=0.10` sample, the front still needs `4.83832406026003 m/s`
  average continuation speed to reach H2 by `0.3540126077962089 s`, which is
  `1.0415226561222388x` the dry-bed Ritter speed. This keeps Test02 open even
  after the refined short run.
- `test02_front_closure_normal_only_h0p15_h0p20_audit_20260603.json` compares
  the corrected all-solid normal-only controls. Refining that control from
  `h=0.20` to `h=0.15` improves the front by about `25%`; the `h=0.15`
  normal-only point is also `1.3088669354705278x` the `h=0.15` no-slip distance
  at the nearest `0.2 s` sample and `1.1693337914065964x` the `h=0.10`
  no-slip distance. This rules out mesh resolution alone as the cause of the
  short-time lag, but it keeps Test02 open because H2/P1 remain zero and the
  corrected-boundary `h=0.15` front reaches only `0.5283135835023329` of the
  official H2-average distance by `0.201 s`.
- `test02_front_closure_normal_only_h0p10_h0p15_h0p20_audit_20260603.json`
  extends the corrected all-solid normal-only comparison through the `h=0.10`
  point. At `0.2 s`, the `h=0.10` normal-only front travels
  `1.3096337723314824x` the `h=0.20` normal-only distance,
  `1.0445832044434704x` the `h=0.15` normal-only nearest-`0.2 s` distance, and
  `1.2214664388915353x` the `h=0.10` no-slip distance. This confirms
  resolution and component-only boundary treatment both matter, but it keeps
  Test02 open because H2/P1 remain zero and the front reaches only
  `0.5509762978464111` of the official H2-average distance at `0.2 s`.
- `test02_first_impact_normal_only_h0p15_h0p10_mesh_resolution_audit_20260603.json`
  compares the corrected all-solid normal-only first-impact runs through
  `0.5 s`. At the end of the window, `h=0.15` has traveled
  `0.9929975980112928x` the `h=0.10` front distance. H2/P1 first response is
  only `0.008/0.007 s` later at `h=0.15`, and P3 pressure remains severely
  underpredicted in both runs. This keeps Test02 open for a longer and more
  accurate pressure-history path rather than closing it as a mesh-resolution
  issue.
- `audit_test02_pressure_sampling_sensitivity.py` reuses the existing
  first-impact outputs to check pressure sample placement. The resulting
  `test02_structured_h0p10_all_solid_normal_only_0p5_pressure_sampling_sensitivity_20260603.json`
  and
  `test02_structured_h0p15_all_solid_normal_only_0p5_pressure_sampling_sensitivity_20260603.json`
  show that nearby sampling does not explain P3 underprediction or P5/P7 timing
  errors.
- `audit_test02_obstacle_face_dynamics.py` compares local obstacle-face water
  coverage with the pressure stack. The h0.10/h0.15 `0.5 s` reports show the
  saved window stops just before P3-height wetting; the h0.15 `0.54 s` extension
  crosses P3 height but still leaves P1 overshot, P3 low relative to P1, P5
  early, and P7 zero. The companion pressure-extrema report localizes the
  largest active/wet pressure spike to a barely wet node with incident
  WetVolumeFraction max `1.8313270361947403e-06`, while the P1 sensor overshoot
  sits on fully wet cells. The h0.15 pressure-gradient ghost-penalty `10x`
  control then rejects the simple penalty-scaling route: P1/P3/P5 target peaks
  drop to `0.262/0.282/0.192x` of the unit-penalty run, but active/wet tiny-cut
  pressure rises to `4482180.476525261 Pa` with `16` tiny-cut wet spikes. Test02
  stays open for pressure/run-up and tiny-cut pressure control rather than a
  pressure-sensor placement change. A half-step `h=0.15` control
  (`dt=max_dt=7.5e-4`) removes the old `0.519 s` rejected-step cluster, keeping
  P1 near `1.65 kPa` there, but the P1 overshoot persists on an accepted
  clipped `1.875e-4 s` step: P1 rises from `21593.585663771137 Pa` at
  `0.5248125 s` to `69711.63932805778 Pa` at `0.525 s`, with P3/P1 only
  `0.03719504770446654`. The same run has an active/wet tiny-cut maximum of
  `8495986.774435218 Pa` on incident WetVolumeFraction max
  `2.8089754870937977e-08`. The remaining pressure failure is therefore
  accepted micro-step/time-integration sensitive, not just old retry rollback.
- `audit_test02_pressure_spike_timing.py` sharpens that timing conclusion on
  the existing unit, half-step, and high-damping runs. The unit h0.15 `0.54 s`
  run-up P1 maximum is an accepted-output event at
  `0.5191874999999996 s`, jumping by `63090.231308240276 Pa` over an accepted
  `1.875e-4 s` step and sitting `72412.53251362337 Pa` above local hydrostatic
  pressure inferred from the front-face height. The triggering accepted step
  has one Newton iteration, residual `0.5492065892179963`, solution-state
  pressure max `97684.2 Pa`, active minimum volume fraction
  `1.8157617521783323e-08`, `179` capped cut-adjacent scales, and wet-volume
  drift `-8.830157016031144e-11`; it passes by relative residual
  (`172.1 -> 0.5492065892179963`, relative `0.0031912062127716226`) rather
  than absolute residual. The half-step run still jumps by
  `48118.053664286635 Pa` over its final accepted clipped step with residual
  `0.24952601776400515`, relative residual `0.00035751530244360963` from
  initial `697.945`, pressure max `93364.6 Pa`, active minimum volume fraction
  `1.478276308518081e-05`, and `68` capped cut-adjacent scales. The
  rho=`0.0` control lowers the maximum P1 event to `15388.706438227584 Pa`,
  but leaves P3/P1 at only `0.064927321942497` at the P1 peak. This is
  diagnostic evidence against retry rollback, hydrostatic offset, volume drift,
  and mesh resolution as standalone explanations, not a validation gate.
- `SVMP_NEWTON_ABS_TOLERANCE` and `SVMP_NEWTON_REL_TOLERANCE` now provide a
  focused OOP Newton runtime control for the accepted-spike hypothesis without
  changing legacy XML semantics. A one-step flat unfitted smoke run with
  `SVMP_NEWTON_REL_TOLERANCE=0` completed and reported
  `newton(max_it=8, min_it=1, abs_tol=1.0e-04, rel_tol=0.0)` in
  `newton_tolerance_override_smoke_20260604.log`.
- The full-from-start strict h0.15 follow-up with
  `SVMP_NEWTON_REL_TOLERANCE=0` and `SVMP_NEWTON_ABS_TOLERANCE=2.0e-2`
  failed earlier than the old accepted-spike window. It accepted `317` outputs
  to `0.46153125000000045 s`, then rejected all retries down to
  `dt=4.119873046875e-7` with residual `0.14898496558215454`. This catches the
  loose relative-residual mechanism behind the previous `0.5191874999999996 s`
  spike, but does not close Test02: P1/P3 peaks are
  `1537.6992066370167/778.0569497937005 Pa` versus reference-window peaks
  `11173.337782914636/6563.852874977549 Pa`, P5/P7 remain zero, and the run
  covers only `0.06241638127814083` of the reference horizon. Evidence:
  `test02_structured_h0p15_all_solid_normal_only_abs_only_0p4615_failure_summary_20260604.json`.
- A medium-strict h0.15 follow-up with `SVMP_NEWTON_REL_TOLERANCE=1e-4` and
  `SVMP_NEWTON_ABS_TOLERANCE=2.0e-2` does reach `0.54 s`, accepting `389`
  outputs with `17` recovered rejections and crossing both the strict failure
  time and the old unit accepted-spike window. It is not a Test02 closure:
  late micro-steps still accept residuals above the absolute tolerance by
  relative convergence, P1/P5 peaks rise to
  `503953.63722806575/19180.843372484167 Pa`, P3/P1 is only
  `0.04714409578086104` versus reference `0.5840110088913406`, P7 remains
  zero, and wet/active tiny-cut extrema reach `10398116.630046401 Pa`. Evidence:
  `test02_structured_h0p15_all_solid_normal_only_rel1e4_0p54_probe_summary_20260604.json`.
- `audit_test02_pressure_offset_sensitivity.py` rejects a common pressure
  nullspace/anchor offset as the main Test02 pressure-stack explanation. On the
  unit `0.54 s` run-up extension, an optimistic per-time common offset changes
  P3/P1 from `0.05897628798156602` to only `0.0652746785320579`, versus
  reference `0.5923737331912318`; applying a local H4 hydrostatic anchor offset
  worsens aggregate RMSE. On the half-step run, the optimistic offset changes
  P3/P1 only from `0.03719504770446654` to `0.06727622461323107`, versus
  reference `0.5864967243247952`; on the high-damping rho=`0.0` control, it
  reaches `0.25977537406090734` versus reference `0.6030783697975495`.
  Pressure/run-up closure therefore remains a pressure-distribution/free-surface
  impact problem, not just absolute pressure anchoring.
- `audit_test02_obstacle_pressure_profile.py` samples vertical pressure/phi
  profiles on the obstacle front and the top-face pressure line at selected
  impact times. It confirms the official SPHERIC Test02 sensor layout used by
  the verifier, then shows P3 is already wet when the vertical pressure ratio is
  wrong. In the unit `0.54 s` run-up extension, at
  `0.5191874999999996 s` the front-face phi height is
  `0.11308325905162073 m` and P3 phi is `-0.005455165720451012`, but simulated
  P3/P1 is only `0.05897628798156602` versus the time-matched reference ratio
  `0.8212375823466532`. At `0.54 s`, front height is
  `0.12147650414327191 m` and P3/P1 is `0.06212334043088769` versus reference
  `0.8606293623279752`. The rho=`0.0` control repeats the bottom-concentrated
  distribution with P3/P1 `0.05335178174851977` at `0.54 s`, so the residual
  pressure-stack failure is not a dry-P3 cutoff, sensor-layout mismatch, or
  common pressure offset.
- `test02_h0p15_runup_obstacle_cavity_material_audit_20260604.json` rules out a
  filled obstacle cavity or gross material mismatch for the same run-up control.
  The mesh has zero strict-interior obstacle points and zero obstacle-volume
  cell centers, and the solver constants are `rho=998.2 kg/m^3`,
  `mu=0.001003 Pa s`, and gravity `-9.81 m/s^2`, close to the SPHERIC water
  values.
- A capped cut-metadata scale control on the same `h=0.15` all-solid
  normal-only setup (`Use_cut_metadata_scale=true`, cap `3.0`) completes to
  `0.54 s` and removes the old `0.519 s` retry cluster, but it is not a
  pressure/run-up fix. P3 reaches only `1156.9675020942277 Pa`
  (`0.1763031220548037x` the reference-window peak), P7 remains zero, P3/P1 is
  `0.06269900695972026`, and obstacle-top run-up extrapolates to
  `0.5904610584056923 s`. The active/wet tiny-cut maximum rises to
  `2360019.3101965105 Pa`, `1.7878580549606675x` the unit
  metadata-scale-false baseline, so metadata scaling at cap 3 is another
  negative standalone fix rather than Test02 closure.
- A high-damping rho=`0.0` time-integration control on the same setup also
  completes to `0.54 s`. It removes the sampled tiny-cut pressure-extrema
  class: the pressure-extrema scan reports zero tiny-cut wet spikes and max
  active/wet pressure `20417.020764797715 Pa` on fully wet P1-bottom cells. It
  still is not a literature pass because H2 final height drops to
  `0.08980374635691435 m`, P3 reaches only `999.1474971933786 Pa`
  (`0.1525679728204372x` the reference-window peak), P7 remains zero, P3/P1 is
  `0.064927321942497`, and obstacle-top run-up extrapolates later to
  `0.625925501075129 s`. High-frequency damping is therefore a useful
  tiny-cut pressure-spike diagnostic/mitigator, not the Test02 accuracy fix.
- A coarse `h=0.20` SmoothedIndicator active-domain control produces no
  accepted outputs, so it cannot yet serve as a Test02 pressure/run-up
  diagnostic. With `nearest_interface_point` wet-extension, the run aborts
  because the 3D level-set advection velocity extension needs generated
  cut-interface samples from the active cut context. With `nearest_active_vertex`,
  startup rejects through all `11` attempts with zero Newton iterations. The
  Test02 validation route remains `CutVolume`.
- A coarse `h=0.20` all-solid normal-only low velocity-stabilization control
  keeps pressure stabilization at `1.0` and lowers only
  `Cut_cell_velocity_gradient_penalty` from `1.0` to `0.25`. It completes to
  `0.2 s` with `101` accepted outputs and one recovered rejection, so it is
  more robust than disabling the velocity penalty. It is still not a speed or
  run-up fix: final front travel is `0.2376820279359817 m`, only
  `1.1094808578526738e-06 m` different from the unit-penalty all-solid
  normal-only baseline, and H2/P1 remain dry/zero. The unit velocity ghost
  penalty is therefore not the simple coarse short-time front-lag explanation.
- The official workbook was imported as
  `test02_reference_fetch_summary_20260602.json` and
  `test02_reference_histories_20260602.csv`; it contains 7395 samples with
  H1-H4 and P1-P8 histories. The reference timing shows H2 first exceeds
  `0.005 m` at `0.3540126077962089 s`, P1 first exceeds `100 Pa` at
  `0.3930264600370048 s`, and P3 first exceeds `100 Pa` at
  `0.4220221953556109 s`; P5/P7 first respond near `1.06 s`. Full closure
  still requires a checked/refined long transient, convergence evidence through
  the dam-break impact window, and sampled comparison against H4/H2 and
  P1/P3/P5/P7. The current evidence rules out the old geometry/topology setup
  and the fixed active-source velocity-extension path as sufficient
  explanations. The post-startup zero-contour front broadly follows the local
  leading-edge velocity, and the h-refined short probes still lag the reference
  timing, so the remaining lag is in the resolved fluid/interface velocity and
  cut-mesh dynamics rather than merely a stale advection-field extension, loose
  fluid nonlinear convergence, or the Navier-Stokes velocity-extension PDE.
  Disabling the pressure-gradient cut-cell stabilization is also ruled out: it
  creates nonphysical pressure excursions while moving the front less than the
  stabilized post-fix baseline.
  Disabling the velocity-gradient cut-cell stabilization is ruled out because
  it cannot accept a startup step with pressure stabilization retained.
  Resolution contributes, but the evidence does not support accepting either
  `h=0.20` or the short `h=0.15` run as physically accurate.
- `tests/cases/fluid/open_vessel_free_surface/verify_spheric_test02_histories.py`
  now performs the H/P comparison. On the twelve-step startup run it reports an
  early-time-only comparison through `t=0.012`: H4 RMSE
  `0.0010981014927379947 m`, H2 RMSE `0.0005282369511827052 m`, and all
  requested obstacle pressure
  targets were valid interpolated point samples in a no-impact window.

## Long Transient Volume Conservation

Purpose: quantify free-surface volume preservation over many advection and
mesh-motion updates.

Setup:

- quiescent tank, sloshing tank, or translating level-set interface
- run for at least `1000` time steps or ten dominant sloshing periods
- save volume diagnostics at every output interval

Measured outputs:

- water volume or area
- relative volume change
- level-set correction shift and correction iterations
- fitted mesh boundary displacement measure

Acceptance criteria:

- fitted ALE relative volume drift below `1.0e-4` on the reference mesh
- unfitted level-set relative volume drift below the configured correction
  tolerance
- no monotone volume loss trend remains after correction is enabled

## Contact-Angle Static Meniscus

Purpose: validate wall contact-line and prescribed contact-angle behavior.

Setup:

- narrow rectangular channel or open tank with a prescribed wall contact angle
  `theta`
- surface tension enabled
- gravity optional; include gravity for capillary-rise comparison
- initial interface close to the expected static meniscus

Reference behavior:

- interface normal at the wall satisfies the prescribed contact angle
- pressure jump follows the Young-Laplace relation
- with gravity, capillary rise or depression follows the standard static balance

Measured outputs:

- contact angle reconstructed from the interface normal and wall normal
- static interface profile
- pressure jump across the interface
- active contact measure and skipped-contact diagnostics

Acceptance criteria:

- reconstructed contact angle within two degrees of the prescribed value away
  from deliberately skipped degeneracies
- static pressure jump matches `gamma kappa`
- contact diagnostics report deterministic active contact measure under
  refinement and MPI partitioning

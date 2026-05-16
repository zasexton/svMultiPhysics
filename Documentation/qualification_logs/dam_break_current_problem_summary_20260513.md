# Dam-Break Current Problem Summary - 2026-05-13

## Current Status

The D18 wet-bed dam-break case can now run past the earlier MPI-4 solver stop,
but the resulting fields are still not a valid dam-break solution. The solver
marching problem and the physical accuracy problem are separate:

- The previous zero-update residual floor at step `63` was addressed by forcing
  at least one FSILS BlockSchur outer iteration for the coupled D18 layout.
- The full D18 MPI-4 run now reaches `result_312.pvtu` at `0.156 s`.
- The result remains physically wrong: the interface barely evolves, the
  velocity field is sparse and too small, and the pressure field remains close
  to a constrained hydrostatic state.

D18 and D38 qualification should remain blocked.

## 2026-05-14 Remediation Update

The leading fixture/setup issues were addressed in the Test05 D18/D38 inputs:

- The shallow-bed level set now uses a wall-inclusive union field rather than
  a closed-box signed distance. This removes the previous bottom-wall
  `phi=0` ambiguity that caused the D18 wet bed to be represented mostly by
  interface nodes instead of strictly wet nodes.
- The D18/D38 meshes were regenerated with a structured thin-extrusion
  tetrahedral grid that resolves the D18 wet bed vertically, keeps the
  unfitted interfaces off element faces, and avoids the high-aspect elements
  in the first structured pass. The split is mirrored about the extrusion
  mid-plane to avoid biasing the nominally two-dimensional flow in `z`. D18
  now has `2640` points, `9720` tetrahedra, and `756` initially cut cells.
  D38 now has `2970` points, `11016` tetrahedra, and `744` initially cut
  cells.
- The updated D18 mesh has minimum/5th-percentile/median mean-ratio quality
  `0.413/0.413/0.657`, maximum edge ratio `4.98`, and no cells below
  mean-ratio `0.2`. D38 has minimum/5th-percentile/median mean-ratio quality
  `0.435/0.435/0.649`, maximum edge ratio `4.58`, and no cells below
  mean-ratio `0.2`. The prior structured pass had minimum mean-ratio
  `0.160`, maximum edge ratio `9.67`, and `2240` tetrahedra below
  mean-ratio `0.2`.
- The D18 clipped initial wet volume is now `0.002153156603675844 m^3`,
  within `0.017%` of the analytic `0.0021528 m^3`. D38 is
  `0.0026449790412131097 m^3`, within `0.0068%` of the analytic
  `0.0026448 m^3`. The previous checked-in D18 mesh captured only about
  `73.7%` of the analytic wet volume.
- The initial `Pressure` mesh field is now piecewise hydrostatic: retained
  column pressure is based on the `0.15 m` water column, while downstream
  wet-bed pressure is based on the local wet-bed depth. For D18, downstream
  bed pressure is capped near `176.262 Pa` instead of inheriting the retained
  column's `~1.47 kPa` head.
- The hard `<Node_pressure_constraints>` pin was removed from D18/D38. The
  solver now supports `<Hydrostatic_pressure_field_name>Pressure</...>` and
  uses the mesh pressure field during hydrostatic initialization.
- `wall_top` is now a velocity Dirichlet wall in the unfitted Test05 fixtures,
  so pressure nullspace handling is no longer blocked by an unconstrained dry
  top velocity boundary.
- D18/D38 level-set and fluid equations now explicitly set
  `<Module_options>jit=true; jit_specialization=true</Module_options>`, so the
  OOP physics path requests LLVM JIT independent of the process environment.

The mesh-side diagnostics now show the corrected clipped wet volumes above,
the improved element-quality bounds, and a piecewise pressure field with D18
downstream pressure capped at `176.262 Pa` and D38 downstream pressure capped
at `372.109 Pa`. The D18 setup now passes one-step and five-step MPI-4 field
diagnostics, but D18/D38 physical qualification still needs a full reference
time run.

### 2026-05-14 D18 Field Diagnostics

A temporary D18 MPI-4 run was made with only `Number_of_time_steps=1` changed
from the checked-in fixture. The run wrote `result_001.pvtu` at `t=0.0005 s`,
accepted the step, and reported one nonlinear iteration with a converged
linear solve:

- nonlinear residual `1.8013074259508576e-04`;
- linear relative residual `6.5313989687913737e-05`;
- no `using interpreter`, `JIT: failed`, or JIT fallback warnings in the log.

The one-step field is progressing in the expected primary flow direction:

- all output `phi`, `Velocity`, and `Pressure` values are finite;
- clipped wet volume changes from `0.002153156603675844 m^3` to
  `0.002153176267603492 m^3`, a relative change of `9.13e-06`;
- all six wall planes have zero velocity at their boundary nodes;
- the largest velocity magnitude is `0.06795583685387444 m/s`;
- the mean gate-region velocity is positive in `x`
  (`u = [0.0048716, 0.0010188, 6.39e-10] m/s`), and the front-region mean
  velocity is also positive in `x`
  (`u = [0.0038082, 0.0008492, -3.14e-09] m/s`);
- the `phi=0` height change over the midplane profile is small for one
  `5e-4 s` step, with maximum sampled height change `5.33e-06 m`;
- maximum transverse velocity is `1.6100362308193447e-06 m/s`, or
  `2.37e-05` of the maximum speed. Before the mirrored split, the same
  diagnostic had `max |w| = 0.02608853882877982 m/s`.

A temporary five-step D18 MPI-4 run then advanced to `t=0.0025 s` and wrote
`result_001.pvtu` through `result_005.pvtu`. All five steps were accepted with
one nonlinear iteration per step. At step 5:

- all output fields are finite;
- clipped wet-volume relative drift is `6.78e-05`;
- maximum speed is `0.13452140985559818 m/s`;
- maximum transverse velocity is `2.132845973154921e-06 m/s`, or `1.59e-05`
  of the maximum speed;
- gate-region mean velocity is
  `u = [0.0123740, -0.0008736, 1.62e-08] m/s`;
- front-region mean velocity is
  `u = [0.0102650, -0.0004457, 9.26e-09] m/s`;
- maximum sampled midplane `phi=0` height change is `3.63e-05 m`.

The one-step and five-step fields therefore show reasonable primary-plane
dam-break onset with the transverse mode controlled. This is still a short-run
qualification only; the full reference-time D18/D38 profile comparison remains
blocked until a longer transient is rerun with the mirrored mesh.

The reusable checker for these short-run field diagnostics is
`tests/cases/fluid/open_vessel_free_surface/check_test05_one_step_field.py`.
It fails on non-finite fields, excessive wet-volume drift, nonzero wall
velocity, missing downstream gate/front motion, and excessive transverse
velocity.

### 2026-05-14 Reference-Time Run Attempt

A full mirrored D18 MPI-4 run was started from the checked-in `312` step XML in:

```text
/tmp/svmp_d18_mirror_full_20260514_003056
```

The run used `SVMP_OOP_JIT_ENABLE=1` and `SVMP_FE_JIT_ENABLE=1`. All ranks
reported LLVM OrcJIT initialization and the log did not contain interpreter or
fallback messages. The run reached `result_003.pvtu` and was stopped manually
because the measured step cost makes a full interactive D18/D38 reference-time
qualification impractical:

- step 1 accepted with the same residual history as the earlier short probe;
- step 2 and step 3 were also accepted with one nonlinear iteration each;
- each step is dominated by two cut-volume assemblies, with each assembly near
  `40 s` on MPI-4;
- VTK output is negligible (`~0.0015 s` per saved step), so reducing output
  cadence will not materially reduce wall time.

The step-3 field checker still passes:

- maximum speed `0.11316867873613981 m/s`;
- gate-region mean velocity
  `u = [0.0093546, 0.0002606, 2.96e-08] m/s`;
- front-region mean velocity
  `u = [0.0075261, 0.0003476, 2.55e-08] m/s`;
- maximum transverse velocity `7.238846100440093e-06 m/s`, or `6.40e-05`
  of maximum speed;
- wet-volume relative drift `3.70e-05`;
- maximum sampled midplane profile height change `2.04e-05 m`.

The normal `build/svMultiPhysics-build/bin/svmultiphysics` binary was rebuilt
from current source and tested on a one-step MPI-4 copy. It also used LLVM JIT
and showed the same gross bottleneck: a one-step time loop cost of about
`146 s`, with `~83 s` in the two Newton assemblies. A serial one-step probe was
worse (`~124 s` per large assembly on one rank) and was stopped.

### 2026-05-14 Assembly Performance Follow-Up

The first assembly timing split showed that the expensive path was not
cut-volume quadrature. On the one-step D18 MPI-4 probe:

- assembly total was about `40.6 s` per Newton assembly;
- interior-face assembly was about `40.4 s` of that total;
- cut-volume assembly was only about `0.085 s`.

Focused interior-face timing then showed that most of the cost was
`prepareContextFace`, not JIT kernels, insertion, or linear algebra:

- per interior-face pass, each rank processed about `4.3k` faces;
- `prepare_minus` and `prepare_plus` each cost about `4.8-5.0 s` per rank;
- kernel time was typically below `0.1 s` per pass.

The root cause was unnecessary face-geometry work. The generic face-frame
helper computed surface Jacobians and mean curvature for every face context,
even when the active free-surface/interior-face kernels only requested normals,
measures, and standard geometric data. The solver now only computes those
higher-order surface quantities when the kernel `RequiredData` requests them,
and the active face path reuses the cell mapping already built in
`prepareContextFace`.

Post-fix timing on the same D18 MPI-4 one-step probe:

- top-level time loop: `26.456202 s` (`real 29.13 s`);
- Newton time: `17.147161 s`;
- assembly time inside Newton: `14.911425 s`;
- interior-face time per Newton assembly: about `7.4 s`;
- cut-volume time per Newton assembly: about `0.083 s`.

A five-step D18 MPI-4 post-fix run completed successfully in:

```text
/tmp/svmp_d18_5step_post_faceopt_20260514_040310
```

Run result:

- exit status `0`;
- outputs `result_001.pvtu` through `result_005.pvtu`;
- top-level time loop `93.194924 s` (`real 96.05 s`);
- all five steps converged in one nonlinear iteration;
- no interpreter fallback or JIT failure messages were found.

The post-fix one-step and five-step field checks still pass. The step-5 field
metrics remain directionally consistent with the earlier physically reasonable
short run:

- maximum speed `0.13452140985559818 m/s`;
- maximum transverse velocity `2.132845973154921e-06 m/s`, or `1.59e-05` of
  maximum speed;
- gate-region mean velocity
  `u = [0.0123740, -0.0008736, 1.62e-08] m/s`;
- front-region mean velocity
  `u = [0.0102650, -0.0004457, 9.26e-09] m/s`;
- clipped wet-volume relative drift `6.78e-05`.

Conclusion: mesh quality, VTK output, and LLVM JIT fallback are not current
blockers. The major interactive runtime blocker was unnecessary interior-face
surface-curvature preparation and has been reduced by roughly `5x` for the
D18 one-step run. The next qualification step is to rerun the full D18
reference-time profile comparison with the optimized assembly path, then repeat
for D38 if D18 remains physically consistent.

### 2026-05-14 Optimized Full-D18 Attempt

The optimized assembly path was used for a full `312` step D18 MPI-4 attempt in:

```text
/tmp/svmp_d18_full_faceopt_20260514_085032
```

Command:

```sh
/usr/bin/time -p env SVMP_OOP_JIT_ENABLE=1 SVMP_FE_JIT_ENABLE=1 \
  timeout 10800s mpiexec -n 4 \
  /home/zack/Downloads/svMultiPhysics/build/svMultiPhysics-build/bin/svmultiphysics solver.xml
```

Observations:

- all ranks initialized LLVM OrcJIT; no interpreter or fallback markers were
  found;
- the run passed the previous step-63 failure point and advanced to
  `result_238.pvtu` at `t=0.119 s`;
- repeated bounded nonlinear residual spikes appeared as the front crossed cut
  cells, for example step 237 accepted with residual
  `2.9967985194744176e-03`;
- the next step entered a very hard FSILS block-Schur solve and made no
  log/file progress for about ten minutes while all four ranks remained at
  full CPU;
- a stack sample during the stall showed the rank in
  `bicgs::schur_impl(...)`, inside repeated rectangular sparse Schur operator
  applications and nested momentum solves, not in FE assembly, VTK output, or
  LLVM JIT;
- the attempt was stopped manually after `real 4612.39 s` to preserve evidence
  instead of waiting for the three-hour timeout.

The latest completed `result_238.pvtu` field was finite and physically active:

- maximum speed `0.9460813083146566 m/s`;
- gate-region mean velocity
  `u = [0.0995336, -0.0027465, -3.17e-08] m/s`;
- front-region mean velocity
  `u = [0.1061695, -2.16e-05, -8.61e-09] m/s`;
- maximum transverse velocity `6.3618181060977985e-06 m/s`, or `6.72e-06`
  of maximum speed;
- wet-volume relative drift `-0.00179676`.

The intermediate profile comparison at `result_238.pvtu` is not the final
reference-time qualification, but it is informative:

- validation checks passed, including `velocity_max > 0.5 m/s`;
- profile RMSE `0.023627445935179896 m`;
- front error `-0.001568885311762358 m`;
- peak-height error `0.012286230665336934 m`;
- pressure gauge `501.63716833715176 Pa`, no longer pinned to the initial
  hydrostatic value.

Evidence files:

- `Documentation/qualification_logs/dam_break_d18_profile_comparison_20260514/d18_faceopt238_metrics.json`
- `Documentation/qualification_logs/dam_break_d18_profile_comparison_20260514/d18_faceopt238_profile.png`
- `Documentation/qualification_logs/dam_break_d18_profile_comparison_20260514/d18_faceopt238_field_metrics.json`

Conclusion: the corrected D18 setup is no longer a nearly static/hydrostatic
solution; the wet region and velocity field are progressing in the expected
direction. The current blocker is now late-time linear-solver robustness for
ill-conditioned moving-front cut-cell states.

As a fail-fast fixture guard, the D18/D38 nested FSILS block-Schur limits were
reduced from `1000` to `150` for both `NS_GM_max_iterations` and
`NS_CG_max_iterations`. A capped five-step D18 smoke run in:

```text
/tmp/svmp_d18_cap_smoke_20260514_100906
```

completed successfully with LLVM OrcJIT and no fallback markers. The step-5
field check still passed with the same metrics as the previous optimized
five-step run: maximum speed `0.13452140985559818 m/s`, clipped wet-volume
relative drift `6.78e-05`, and maximum transverse velocity
`2.132845973154921e-06 m/s`.

The D38 full reference run was not started because D18 did not yet complete the
reference time with a usable final profile. The next best step is to address
the step-238 block-Schur robustness directly, likely by testing Schur GMRES
for these cut-cell states and/or adding a solver progress guard that reports
inner Schur/momentum failure promptly instead of allowing one front step to
consume the full qualification budget.

These changes do not require switching this benchmark to an N-Eulerian
multiphase formulation. For the current qualification target, the water-air
interface can still be represented as a single incompressible water phase with
an unfitted level set and passive exterior. A true multiphase formulation would
be justified only if the benchmark requires resolving air inertia, entrainment,
compressibility, or topology changes beyond what the level-set free-surface
model is meant to represent.

## Main Failure Signatures

- The `phi=0` profile does not show the expected dam-break collapse and run-up.
  In the full MPI-4 run at `0.156 s`, the extracted D18 profile has RMSE
  `0.03365458346158355 m` against `d18_1.dat`.
- The final D18 interface profile remains too close to the initial retained
  water-column shape. Across sampled outputs from step `50` through step `310`,
  the highest interface point stays near `0.1509 m`.
- The final D18 velocity is far too small for the expected gravity-driven
  collapse:
  - `velocity_max = 0.12408780231518766 m/s`
  - `velocity_mean = 0.0025161748510761743 m/s`
  - `velocity_wet_mean = 0.008466118058423023 m/s`
  - `kinetic_energy = 0.00024711159741521333 J`
- The pressure gauge remains exactly fixed at the initial hydrostatic value:
  `643.659423052 Pa` at node `256` through `result_312`.
- The pressure range remains hydrostatic-like at the final profile time:
  - `pressure_min = -0.00025860192756981036 Pa`
  - `pressure_max = 1469.0132794991368 Pa`
  - wet-side pressure mean near `777.58 Pa`
- The largest velocity is in the wet region, but it is localized and small.
  In `result_310.pvtu`, the largest speed is at GlobalVertexID `452`, with
  speed `0.12408664632689799 m/s`.
- The user-reported node `486` should be rechecked against the exact output
  numbering. In the merged `result_310.pvtu` file, point index `486` is on the
  dry top boundary with speed `5.307178052090245e-04 m/s`, and there is no
  `GlobalNodeID` or `GlobalVertexID` equal to `486` in that merged output.

## Implemented Foundation

The following remediation work is already in place and covered by tests or
logs:

- Active-domain free-surface configuration:
  `Active_domain=LevelSetNegative` and `Active_domain_method=CutVolume`.
- Generated cut-volume rules for wet-side Navier-Stokes volume assembly.
- Navier-Stokes and VMS volume terms routed through the active wet-side
  cut-volume measure.
- Wet-side hydrostatic pressure initialization and dry-side reference pressure
  handling.
- Pressure-gauge metadata validation for active-domain cases.
- D18 gauge changed from the old near-interface node to wet node `256`.
- Profile comparison script now reports field metrics, wet volume, pressure,
  velocity, kinetic energy, and reference profile errors.
- D18 and D38 Test05 generated fixtures now use the grouped BlockSchur layout:
  `LevelSetVelocity(0:4), Pressure(4:1)`.
- FSILS BlockSchur coupled outer FGMRES path exists for the grouped
  `phi`, `Velocity`, and `Pressure` layout.
- FSILS BlockSchur now supports `NS_min_outer_iterations`, routed from XML to
  the coupled outer FGMRES path and legacy loop.

## Solver Probes And Outcomes

### Strict Tolerance Attempts

- Strict D18 one-step settings with nonlinear tolerance `1.0e-6`, linear
  relative tolerance `1.0e-6`, and linear absolute tolerance `1.0e-10` exposed
  a true-residual floor in MPI-4.
- A strict MPI-4 probe failed at the initial solve with true residual
  `|Ax-b|=0.0425642`, relative residual `0.00690591`, and target
  `6.16345e-06`, even after large BlockSchur work.
- Conclusion: strict tolerances cannot yet be restored with the present
  coupled BlockSchur scaling.

### Profile-Run Tolerance Probes

- Fluid nonlinear tolerance `5.0e-4` stopped at step `50` with residual
  `5.1076666039798896e-04`.
- Fluid nonlinear tolerance `6.0e-4` crossed step `50` and completed a
  60-step probe, but the full profile run stopped at step `63` with residual
  `6.2761512021829798e-04`.
- Fluid nonlinear tolerance `7.0e-4` crossed step `63` but stopped at step
  `67` with residual `7.5725252640414459e-04`.
- Fluid nonlinear tolerance `8.0e-4` crossed step `67` but stopped at step
  `78` with residual `8.1397034085718028e-04`.
- Fluid nonlinear tolerance `9.0e-4` reproduced a bad early branch with large
  residuals in steps `1` and `3`.
- Conclusion: raising nonlinear tolerance is not an acceptable primary fix.

### Minimum Outer Iteration Fix

- The step-63 stall showed linear convergence with zero outer iterations and
  no useful Newton update.
- `NS_min_outer_iterations=1` was added and applied to the FSILS BlockSchur
  coupled outer FGMRES path.
- A 70-step D18 MPI-4 probe with the checked-in XML controls reached
  `result_070.pvtu` and returned `success=1`.
- In that probe, step `63` converged with residual
  `2.1432747190868549e-04` and one linear outer iteration.
- The full D18 MPI-4 profile run reached `result_312.pvtu` and returned
  `success=1`.
- Conclusion: the minimum-outer control fixes a solver marching failure, but
  it does not fix the physical solution.

### Full D18 MPI-4 Result At The First Reference Time

Run directory:

```text
/tmp/svmp_d18_mpi4_minouter_xml_312step_lOpa1P
```

Run result:

- Exit status: `0`
- Last output: `result_312.pvtu`
- Final time: `0.156 s`
- `loop.run() returned success=1 steps_taken=312`

Comparison command:

```sh
python3 tests/cases/fluid/open_vessel_free_surface/compare_test05_profiles.py \
  /tmp/svmp_d18_mpi4_minouter_xml_312step_lOpa1P/result_312.pvtu \
  tests/cases/fluid/open_vessel_free_surface/reference_profiles/spheric_test05_wet_bed/d18_1.dat \
  --benchmark-json tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test05_wet_bed_d18/benchmark.json \
  --front-diagnostic-only \
  --stale-pressure-gauge-tolerance 1.0 \
  --min-velocity-max 0.5 \
  --output Documentation/qualification_logs/dam_break_d18_profile_comparison_20260513/d18_minouter312_metrics.json \
  --plot-output Documentation/qualification_logs/dam_break_d18_profile_comparison_20260513/d18_minouter312_profile.png
```

Comparison result:

- Field validation failed because `velocity_max < 0.5 m/s`.
- Profile RMSE: `0.03365458346158355 m`.
- Peak-height error: `0.02200091339365315 m`.
- Wet volume: `0.0016175924638147652 m^3`.
- Gauge pressure: `643.659423052 Pa`, exactly the prescribed initial value.

## Pressure Constraint Probe

A temporary D18 MPI-4 probe removed the `<Node_pressure_constraints>` block and
ran to six outputs before it was stopped manually:

```text
/tmp/svmp_d18_mpi4_no_pressure_pin_80step_wKoXNc
```

Observed early behavior:

- Step `1`: `velocity_max = 0.049007853337125165 m/s`
- Step `3`: `velocity_max = 0.017214241561358076 m/s`
- Pressure offset became uncontrolled without a point pressure value.

Conclusion: the hard pressure node is suspicious because it is a constraint,
not a passive gauge, but removing it alone does not immediately restore the
expected collapse. A different pressure-nullspace treatment may still be
needed, but the core problem is not solved by simply deleting the pressure node.

## Current Leading Suspects

1. The fluid solve is settling into a near-hydrostatic balance too easily.
   The pressure field remains close to hydrostatic while velocity and kinetic
   energy remain too small.
2. The point pressure value is being used as a hard Dirichlet pressure
   constraint. That keeps the selected wet node fixed at the initial
   hydrostatic pressure throughout the transient.
3. The nonlinear convergence policy accepts large absolute nonlinear residuals
   when the relative criterion is satisfied. For example, early steps can be
   accepted with residuals much larger than the nominal physical tolerance.
4. Strict tolerances currently expose an FSILS true-residual floor rather than
   producing a usable corrected transient.
5. Level-set advection is coupled to the `Velocity` unknown, but the coupled
   velocity field is already too small and sparse. Therefore the interface
   transport equation has little physically meaningful motion to use.
6. Reinitialization and volume correction preserve wet volume, but they may be
   masking poor advection if the velocity field is not producing the expected
   collapse.
7. Active-domain cut-volume diagnostics show nonzero wet volume, but the
   resulting force imbalance may still be wrong if the pressure, gravity, and
   continuity terms are not balanced over the same active support.

## What Has Been Ruled Out

- The old full-volume hydrostatic initialization problem has been addressed at
  setup time: dry-side pressure is initialized to the reference pressure for
  active-domain cases.
- The step-63 MPI-4 profile-run stop was not the fundamental accuracy issue;
  after the minimum-outer fix, the run completes but remains physically wrong.
- Raising the nonlinear tolerance further is not acceptable; it eventually
  produces a bad early branch.
- The profile-front metric alone is not sufficient for validation because the
  wet bed already extends far downstream.
- Removing the point pressure constraint alone is not enough to immediately
  create the expected collapse in the early transient.

## Recommended Next Investigation

1. Add a short diagnostic that computes active-domain integrated force terms
   at the initial state: pressure-gradient contribution, gravity contribution,
   viscous contribution, continuity residual, and VMS contribution.
2. Verify that hydrostatic pressure and gravity do not cancel in the released
   column in a way that prevents horizontal acceleration.
3. Replace the hard point pressure value with a pressure nullspace treatment
   that does not pin a wet pressure node to its initial hydrostatic value
   throughout the transient.
4. Add validation checks that reject a profile run if gauge pressure remains
   exactly equal to its initial hydrostatic value after several steps.
5. Add validation checks that reject a profile run if wet-side kinetic energy
   remains below a documented lower bound at the first D18 reference time.
6. Run a reduced diagnostic case with reinitialization and volume correction
   disabled for a few steps to isolate pure level-set advection from correction
   effects.
7. Inspect cut-volume assembly for the body-force and pressure terms to ensure
   both are integrated over identical wet-side support and have the expected
   signs.
8. Revisit strict solver scaling only after the initial force balance produces
   physically meaningful velocity growth.

## Key Evidence Files

- `Documentation/qualification_logs/dam_break_remaining_investigations_20260513.md`
- `Documentation/qualification_logs/dam_break_accuracy_remediation_plan_20260513.md`
- `Documentation/qualification_logs/dam_break_d18_profile_solver_controls_20260513.md`
- `Documentation/qualification_logs/dam_break_d18_min_outer_iterations_20260513.md`
- `Documentation/qualification_logs/dam_break_d18_profile_comparison_20260513/d18_minouter312_metrics.json`
- `Documentation/qualification_logs/dam_break_d18_profile_comparison_20260513/d18_minouter312_profile.png`

## 2026-05-14 Phi-Field Remediation Follow-Up

The later D18 profile evidence confirmed the user-facing problem: the `phi`
field was not yet a credible moving water front. Two issues were isolated and
fixed in the Test05 fixtures:

1. The corrected centered-cut mesh exposed a BlockSchur preconditioner problem.
   With the generated centered-cut D18 mesh, `NS_Schur_preconditioner =
   algebraic-shat` failed on the first step with a true-residual miss
   (`|Ax-b| = 0.0031528`, `rel = 0.0022535`, target `1.39907e-4`). Lowering or
   disabling the inactive-domain velocity extension did not fix this. Switching
   the Test05 Schur preconditioner to `blockdiag-l` accepted the same corrected
   mesh in one nonlinear iteration with linear relative residual
   `6.77107e-5`.
2. Runtime projection reinitialization was damaging the wet-bed level-set field
   at step 5. With reinitialization and volume correction enabled, the
   five-step D18 run accepted all steps but projection applied a large
   `phi` update (`max_abs_update = 0.0196647 m`) and the field checker failed
   wet-volume drift (`-6.77983e-4`). With the same mesh and solver settings but
   reinitialization and volume correction disabled, the five-step checker
   passed.

The passing five-step D18 MPI-4 diagnostic was:

```text
/tmp/svmp_d18_centered_blockdiagL_5_nomaint_20260514_phi
```

Key step-5 metrics:

- all fields finite;
- `velocity_max = 0.15330206638209168 m/s`;
- gate-region mean velocity
  `[0.019253855325061128, 0.0005244387013476325, -3.7416781311467e-10] m/s`;
- front-region mean velocity
  `[0.014614820967386673, 0.0006983948474360708, -3.332856303625663e-10] m/s`;
- maximum transverse velocity ratio `3.63824e-7`;
- wet-volume relative drift `2.18904e-5`;
- maximum sampled midplane `phi=0` height change `4.04873e-5 m`;
- pressure gauge decreased from the initial hydrostatic `702.789 Pa` to
  `607.783 Pa`, so the pressure field is no longer pinned to the initial
  hydrostatic state.

LLVM JIT was active in these runs on all four ranks. No `using interpreter`,
`JIT: failed`, or fallback markers were found in the run logs.

Conclusion: this does not justify moving immediately to an N-Eulerian
multiphase formulation. The level-set formulation can produce a bounded,
directionally correct early moving-front field once the mesh cuts are centered,
the grouped BlockSchur split uses `blockdiag-l`, and projection maintenance is
kept off for the wet-bed benchmark. Full D18 reference-time profile
qualification is still required before accepting the case.

The next qualification step is a longer D18 run using the checked-in Test05
settings:

- centered-cut D18 mesh;
- active-domain velocity extension enabled;
- `NS_Schur_preconditioner = blockdiag-l`;
- runtime level-set reinitialization and volume correction disabled.

New evidence files:

- `Documentation/qualification_logs/dam_break_d18_profile_comparison_20260514/d18_centered_blockdiagL001_field_metrics.json`
- `Documentation/qualification_logs/dam_break_d18_profile_comparison_20260514/d18_centered_blockdiagL005_maintenance_field_metrics.json`
- `Documentation/qualification_logs/dam_break_d18_profile_comparison_20260514/d18_centered_blockdiagL005_nomaint_field_metrics.json`
- `Documentation/qualification_logs/dam_break_d18_profile_comparison_20260514/d18_centered_blockdiagL005_nomaint_front_metrics.json`

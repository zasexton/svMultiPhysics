# D18 BlockSchur Minimum Outer Iteration Probe - 2026-05-13

## Purpose

This log records the solver-side follow-up for the D18 MPI-4 profile run that
previously stalled at step `63`. The failure signature was a repeated nonlinear
residual slightly above the fluid tolerance, while the coupled BlockSchur linear
path reported convergence with zero outer iterations and produced no useful
state update.

## Finding

The FSILS legacy BlockSchur loop already had an environment-controlled minimum
outer-iteration guard, but the configured coupled outer FGMRES path could still
return before taking an outer iteration when the linear true residual was below
the linear solver target. That behavior is inappropriate for the D18 profile
case once the nonlinear residual is still above the fluid acceptance tolerance.

## Implemented Control

The solver now accepts `NS_min_outer_iterations` in the `LS type="NS"` block.
The option is routed through the application solver translator, stored in
`SolverOptions`, copied into the FSILS solver state, and honored by both the
coupled outer FGMRES path and the legacy BlockSchur loop. The environment
override `SVMP_FSILS_BLOCKSCHUR_MIN_OUTER_ITERS` remains available; the effective
minimum is the larger of the XML setting and the environment value.

D18 and D38 Test05 generated inputs now set:

```xml
<NS_min_outer_iterations>1</NS_min_outer_iterations>
```

## Validation

Build and static checks:

```sh
cmake --build build/svMultiPhysics-build --target svmultiphysics test_application -j2
python3 -m py_compile tests/cases/fluid/open_vessel_free_surface/generate_validation_meshes.py tests/cases/fluid/open_vessel_free_surface/compare_test05_profiles.py tests/cases/fluid/open_vessel_free_surface/check_test05_pressure_gauge_metadata.py
xmllint --noout tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test05_wet_bed_d18/solver.xml tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test05_wet_bed_d38/solver.xml
build/svMultiPhysics-build/bin/test_application --gtest_filter=OpenVesselExamples.LiteratureValidationCasesDeclareGeneratedMeshes
```

All checks passed.

D18 MPI-4 probe:

```sh
run_dir=$(mktemp -d /tmp/svmp_d18_mpi4_minouter_xml_70step_XXXXXX)
cp -a tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test05_wet_bed_d18/. "$run_dir"/
perl -0pi -e 's/<Number_of_time_steps>312<\/Number_of_time_steps>/<Number_of_time_steps>70<\/Number_of_time_steps>/' "$run_dir/solver.xml"
cd "$run_dir"
timeout 1800s mpirun -np 4 /home/zack/Downloads/svMultiPhysics/build/svMultiPhysics-build/bin/svmultiphysics solver.xml > minouter_xml_70step.log 2>&1
```

Result:

- Run directory: `/tmp/svmp_d18_mpi4_minouter_xml_70step_SeEjCv`
- Exit status: `0`
- Last result: `result_070.pvtu`
- Final time: `3.5000000000000003e-02`
- `loop.run() returned success=1 steps_taken=70`

Key step records:

- Step `50`: `converged=1`, nonlinear residual
  `5.1076639678221175e-04`, linear iterations `1`.
- Step `63`: `converged=1`, nonlinear residual
  `2.1432747190868549e-04`, linear iterations `1`.
- Step `69`: `converged=1`, nonlinear residual
  `1.8243876818701334e-04`, linear iterations `1`.

The run crossed the previous step-63 residual floor and completed the 70-step
probe with the checked-in solver controls, using only a temporary reduction of
`Number_of_time_steps`.

## Next Step

Run the checked-in D18 MPI-4 case to the full `312` steps and compare
`result_312.pvtu` against the digitized SPHERIC Test05 D18 profile data.

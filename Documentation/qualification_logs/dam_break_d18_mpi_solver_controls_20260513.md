# D18 MPI Solver Controls - 2026-05-13

## Purpose

This note records the D18/D38 Test05 solver-control update made after the
MPI-4 solver-scaling sweep identified the coupled BlockSchur outer budget and
absolute linear residual floor as the remaining D18 one-step blocker.

## Implementation

- Explicit `NS_Use_coupled_outer_FGMRES=true` BlockSchur solves now honor the
  XML `Max_iterations` value as their outer workspace budget.
- Automatic BlockSchur routes keep the previous conservative default caps.
- The environment override `SVMP_FSILS_BLOCKSCHUR_OUTER_CAP` still has
  precedence when it is set for a diagnostic run.
- The initial generated and checked-in SPHERIC Test05 D18/D38 profile used:
  - Fluid Newton cap: `12`
  - Fluid nonlinear tolerance: `5.0e-6`
  - Coupled outer BlockSchur `Max_iterations`: `800`
  - Coupled outer BlockSchur `Krylov_space_dimension`: `800`
  - Linear relative tolerance: `1.0e-6`
  - Linear absolute tolerance: `1.0e-7`

## Verification

Static and build checks:

```bash
python3 -m py_compile tests/cases/fluid/open_vessel_free_surface/generate_validation_meshes.py
xmllint --noout \
  tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test05_wet_bed_d18/solver.xml \
  tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test05_wet_bed_d38/solver.xml
cmake --build build/svMultiPhysics-build --target svmultiphysics -j2
```

MPI-4 one-step probe:

```bash
mpirun -np 4 build/svMultiPhysics-build/bin/svmultiphysics solver.xml
```

Temporary run directory:

- `/tmp/svmp_d18_mpi4_checked_solver_controls_onestep_eyU6Qx`

Only the temporary copy changed `Number_of_time_steps` to `1` and
`Restart_file_interval` to `1`.

Key log entries:

- Solver builder: `max_iter=800`, `abs_tol=9.9999999999999995e-08`,
  `block_layout=[LevelSetVelocity(0:4), Pressure(4:1)]`
- Newton control: `max_it=12`, `abs_tol=5.0000000000000004e-06`,
  `rel_tol=5.0000000000000004e-06`
- BlockSchur outer policy: `requested=800`, `effective=800`,
  `coupled_outer=1`
- Coupled outer solve 1: `outer_iters=70`, true residual
  `3.148167e-08`, target `1.820469e-07`
- Coupled outer solve 2: `outer_iters=64`, true residual
  `7.750608e-08`, target `1.0e-07`
- Nonlinear convergence: `iters=2`, residual
  `3.4168730913792681e-06`
- Result: `success=1`, `steps_taken=1`,
  `final_time=5.0000000000000001e-04`

## Next Step

The strict profile controls passed the MPI-4 one-step probe but later exposed a
step-4 profile-run solver-depth blocker. Current profile-run controls and
evidence are recorded in
`Documentation/qualification_logs/dam_break_d18_profile_solver_controls_20260513.md`.

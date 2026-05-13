# D18 MPI One-Step Runs - 2026-05-13

## Purpose

This log records one-step D18 active-domain smoke runs used to verify that the
wet-side cut-volume Navier-Stokes path works in serial and under MPI.

All runs used a temporary copy of
`tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test05_wet_bed_d18`
with only `Number_of_time_steps` changed to `1`.

## Runs

| Run | Command | Result | Key diagnostics |
| --- | --- | --- | --- |
| Serial | `svmultiphysics solver.xml` | Passed | Active-domain context reported `negative_volume=1.5869994907267658e-03`; final cut-volume diagnostic reported `active_wet_volume=0.001587`, `cut_cell_active_wet_volume=0.000157407`, `full_cell_active_wet_volume=0.00142959`; nonlinear solve converged in 2 iterations with final residual `1.2282662820623137e-05`; final linear solve used 52 iterations with relative residual `1.1691485632016713e-03`. |
| MPI-2 | `mpiexec -np 2 svmultiphysics solver.xml` | Passed after adding parallel cut-volume assembly routing | Rank-wise final active wet volume was `0.00158362 + 0.000003375 = 0.001586995`; rank-wise final cut-cell wet volume was `0.000156141 + 0.00000126562 = 0.00015740662`; nonlinear solve converged in 2 iterations with final residual `1.2317074066457978e-05`; final linear solve used 52 iterations with relative residual `1.1843943025674963e-03`; output file was `result_001.pvtu`. |
| MPI-4 | `mpiexec -np 4 svmultiphysics solver.xml` | Passed after allowing full-side volume regions without rank-local interface fragments | Rank-wise final active wet volume was `0.00133519 + 0.000003375 + 0 + 0.000248432 = 0.001586997`; rank-wise final cut-cell wet volume was `0.0000720775 + 0.00000126562 + 0 + 0.0000840636 = 0.00015740672`; nonlinear solve converged in 2 iterations with final residual `1.2301161090270293e-05`; final linear solve used 52 iterations with relative residual `1.1959254885564074e-03`; output file was `result_001.pvtu`. |

## Findings

- The first MPI-2 attempt reached active-domain setup and failed with
  `Assembler::assembleCutVolumes: not implemented`.
- Parallel cut-volume assembly now delegates to the local standard assembler
  with the same ghost-routing path used by cell and face assembly.
- The first MPI-4 attempt reached active-domain setup and failed on a rank with
  no active interface fragments. The generated interface lifecycle now treats
  active full-side volume regions as valid even when no interface fragment lies
  on that rank.
- The MPI-2 active wet volume agrees with the serial active wet volume to the
  precision needed for this smoke check.
- The MPI-4 active wet volume agrees with the serial and MPI-2 active wet
  volumes to the precision needed for this smoke check.
- Pressure range and gauge comparisons still need to be extracted before the
  serial/MPI agreement item can be checked.

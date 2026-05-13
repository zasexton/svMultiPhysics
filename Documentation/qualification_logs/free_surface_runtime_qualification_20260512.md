# Free-Surface Runtime Qualification - 2026-05-12

## Scope

- Solver executable: `build-oop-jit-20260505/svMultiPhysics-build/bin/svmultiphysics`
- Run root: `/tmp/svmp_free_surface_qual_20260512_230012`
- Each generated case was copied to the run root and bounded to one time step.
- Process counts: serial, MPI-2, MPI-4.
- Result: all generated fitted and unfitted cases passed all process counts.

## Performance

| Case | np1 (s) | np2 (s) | np2 speedup | np4 (s) | np4 speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| fitted_ale/spheric_test10_lateral_water_1x | 5 | 2 | 2.50 | 2 | 2.50 |
| unfitted_level_set/spheric_test10_lateral_water_1x | 59 | 36 | 1.64 | 24 | 2.46 |
| unfitted_level_set/spheric_test05_wet_bed_d18 | 70 | 39 | 1.79 | 27 | 2.59 |
| unfitted_level_set/spheric_test05_wet_bed_d38 | 70 | 39 | 1.79 | 27 | 2.59 |
| unfitted_level_set/spheric_test02_dambreak_obstacle | 200 | 112 | 1.79 | 73 | 2.74 |

## Accuracy Checks

Hydrostatic pressure maxima were checked against `rho*g*h` using the initial
fill heights recorded in each `benchmark.json` file. Unfitted level-set fields
were compared against the source mesh point data by global vertex id.

| Case | Expected pmax (Pa) | Observed pmax (Pa) | Abs. error (Pa) | Velocity max (m/s) | Level-set check |
| --- | ---: | ---: | ---: | ---: | --- |
| fitted_ale/spheric_test10_lateral_water_1x | 910.688 | 910.688 | 0.000e+00 | 0.0130875 | n/a |
| unfitted_level_set/spheric_test10_lateral_water_1x | 910.688 | 910.688 | 1.592e-06 | 0.00106671 | unchanged; range -0.093 to 0.415; wet=84 dry=328 |
| unfitted_level_set/spheric_test05_wet_bed_d18 | 1468.851 | 1468.851 | 1.319e-06 | 0.0150872 | unchanged; range -0.015 to 0.162; wet=40 dry=302 |
| unfitted_level_set/spheric_test05_wet_bed_d38 | 1468.851 | 1468.851 | 1.319e-06 | 0.0150872 | unchanged; range -0.015 to 0.142; wet=59 dry=274 |
| unfitted_level_set/spheric_test02_dambreak_obstacle | 5385.788 | 5385.788 | 1.481e-06 | 0.0692093 | unchanged; range -0.213527 to 2.67808; wet=59 dry=892 |

Additional checks:

- Fitted ALE mesh displacement and mesh velocity remained zero after the bounded
  startup step.
- Serial-to-MPI pressure differences were at or below `1.51e-4 Pa`.
- Serial-to-MPI level-set differences were exactly zero for every unfitted case.
- Serial-to-MPI wet-phase velocity differences were at or below `1.3e-7 m/s`
  for the unfitted cases. The largest obstacle-case velocity difference was in
  the dry background region.

## Benchmark Coverage

These runs validate generated-case startup, benchmark geometry encoding,
hydrostatic pressure scale, level-set initialization, fitted ALE mesh-motion
binding, and MPI output consistency. The published transient pressure and
free-surface histories referenced by the benchmark source URLs are external data
and are not yet part of the repository comparison set.

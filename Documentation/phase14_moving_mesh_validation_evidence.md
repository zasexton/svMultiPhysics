# Phase 14 Moving-Mesh Validation Evidence

This artifact defines the reproducible evidence needed to start assessing the
accuracy of the Mesh/FE moving-mesh infrastructure implemented through Phases
1-13. It is not an end-to-end FSI validation suite; literature FSI benchmarks
remain later application-level validation once the coupled fluid, structural,
geometric-nonlinearity, and interface solver pieces are present.

## How To Run

From the repository root:

```bash
tools/run_phase14_moving_mesh_validation.sh --skip-mpi
```

For full local validation, including MPI:

```bash
tools/run_phase14_moving_mesh_validation.sh
```

The runner writes a summary and per-check logs to:

```text
Documentation/qualification_logs/phase14_moving_mesh/latest/
```

Build directories can be overridden without editing the script:

```bash
FE_BUILD_DIR=build-fe-check \
MESH_BUILD_DIR=build-mesh-tests \
PHYSICS_BUILD_DIR=build-physics-gcc13-check \
tools/run_phase14_moving_mesh_validation.sh
```

## Evidence Matrix

| Area | Runner check | What it validates |
| --- | --- | --- |
| FE buildability | `build_fe_phase14` | All FE targets needed for moving-domain evidence still compile together. |
| Physics buildability | `build_physics_moving_domain` | Focused moving-domain physics terms still compile against the FE contracts. |
| Current geometry assembly | `fe_phase14_focused` | Prescribed current coordinates change scalar/vector assembled operators as expected. |
| Matrix-free geometry revision | `fe_phase14_focused` | Matrix-free operators refetch geometry after current-coordinate revision changes. |
| Broad FE systems | `fe_systems_broad` | Moving geometry, restart, adaptivity, search access, operator backends, and contact kernels remain mutually compatible. |
| Geometry utilities | `fe_geometry_all` | Frame geometry, mappings, sensitivities, and surface/metric helpers remain correct. |
| Time integration state | `fe_timestepping_all` | Trial, accepted, rollback, and history-state infrastructure used by moving domains remains covered by available tests. |
| Form vocabulary | `fe_forms_moving_domain` | Moving-domain required-data terms and lowering hooks remain physics agnostic. |
| Moving-domain physics terms | `physics_moving_domain` | ALE advection, Navier-Stokes moving-domain terms, and FSI-style interface-motion hooks consume the Mesh/FE contracts correctly. |
| FE MPI | `fe_mpi_ctest` | Representative FE MPI tests, including the moving-mesh backend MPI test, pass under CTest. |
| Mesh MPI | `mesh_mpi_ctest` | Distributed Mesh motion, current coordinates, restart, migration, repartition, and ghost metadata remain consistent. |

## Interpretation

A passing Phase 14 evidence run means the implemented Mesh/FE infrastructure is
qualified for the combinations marked supported through Phases 1-13: static
defaults, prescribed/FE-smoothed moving meshes, current/reference geometry
access, revision-based invalidation, rollback/restart, moved-mesh transfer, and
moving-domain FE/physics term plumbing.

It does not claim full application-level FSI accuracy. The plan records
Turek-Hron FSI validation as later work because those benchmarks require coupled
fluid/structure solve orchestration, structural large-deformation behavior,
interface coupling policy, and benchmark-specific comparison reporting outside
the Mesh/FE infrastructure layer.

# Dynamic Active Pressure Support Refresh

Date: 2026-05-15

## Scope

Inactive pressure constraints now follow the current level-set solution during
active cut-context rebuilds. Before rebuilding generated cut rules, the driver
copies the current FE-ordered level-set solution into the matching mesh vertex
field and rebuilds affine constraints. The smoke parser now records
`diagnostic=level_set_active_side_vertex_constraint` lines and can require this
diagnostic in no-output runs.

## Verification

### Build

Command:

```bash
cmake --build build/svMultiPhysics-build --target svmultiphysics -j2
```

Result: passed.

### Unit Test Availability

Commands:

```bash
cmake --build build/svMultiPhysics-build --target test_LevelSetActiveSideVertexDirichletConstraint -j2
cmake --build build/svMultiPhysics-build --target test_fe_constraints -j2
ctest --test-dir build/svMultiPhysics-build -N
```

Result: the explicit unit-test targets were not configured in this build tree,
and CTest reported `Total Tests: 0`. Source regression coverage was added for
constraint rebuild behavior after the level-set vertex field changes.

### Mini Direct-Solver Probe

Command:

```bash
python3 tests/cases/fluid/open_vessel_free_surface/run_test05_velocity_growth_smoke.py \
  --case mini2d --steps 1 --disable-vtk-output --timeout-seconds 180 \
  --qualification-log /tmp/mini2d_dynamic_pressure_support_20260515.json \
  --allow-failure-diagnostics \
  --require-active-pressure-support-diagnostics \
  --require-eigen-factorization-diagnostics \
  --max-eigen-factorization-pressure-zero-rows 0 \
  --max-eigen-factorization-nonfinite-entries 0 \
  --require-cut-context-solution-source-diagnostics \
  --require-assembly-topology-consistency
```

Result: passed with allowed direct-solver failure diagnostics. Parsed pressure
block zero rows and columns were zero. The active pressure support diagnostic
reported `support_mode=cell_patch`, 32 active support cells, 47 active support
vertices, 34 inactive vertices, 34 constrained owned DOFs, 9 inactive-sign
vertices with wet support, and 0 active-sign vertices without support.

### D18 One-Step GMRES Probe

Command:

```bash
python3 tests/cases/fluid/open_vessel_free_surface/run_test05_velocity_growth_smoke.py \
  --case d18 --steps 1 --linear-solver-type GMRES --disable-vtk-output \
  --timeout-seconds 600 \
  --qualification-log /tmp/d18_dynamic_pressure_support_20260515.json \
  --require-active-pressure-support-diagnostics \
  --require-cut-context-solution-source-diagnostics \
  --require-assembly-topology-consistency \
  --max-nonlinear-iterations 9 \
  --linear-relative-tolerance 6.0e-4 \
  --linear-absolute-tolerance 1.0e-4 \
  --linear-max-iterations 100 \
  --ns-gm-max-iterations 150 \
  --ns-cg-max-iterations 150 \
  --ns-gm-tolerance 1.0e-4 \
  --ns-cg-tolerance 1.0e-4
```

Result: passed. The run accepted 1 step to time `0.0005` with one nonlinear
iteration and a converged 25-iteration linear solve. The latest active pressure
support diagnostic reported 5616 active support cells, 1623 active support
vertices, 2007 inactive vertices, 2007 constrained owned pressure DOFs, 207
inactive-sign vertices with wet support, and 0 active-sign vertices without
support. Cut-context solution-source diagnostics had zero missing sources.

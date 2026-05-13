# Dam-Break Cut Metadata Scale Enablement - 2026-05-13

## Scope

This note records the implementation and first smoke verification for
metadata-scaled cut-adjacent facet stabilization in the SPHERIC Test05 D18/D38
active-domain path.

## Implementation

- Interior-face assembly now receives the registered cut-integration context.
- For each interior face, the assembler computes a face-local stabilization
  scale from the largest inverse cut-cell volume fraction among the adjacent
  cells.
- Faces without adjacent cut-cell metadata receive scale `0.0` for
  metadata-scaled cut stabilization.
- The existing constant-scale path remains unchanged when
  `Use_cut_metadata_scale=false`.
- The Test05 D18/D38 generated fixtures now set
  `Use_cut_metadata_scale=true`.

## Unit Verification

- `test_fe_forms --gtest_filter='FormKernelDGTest.CutStabilizationScale*'`
  passed.
- `test_physics --gtest_filter='MovingDomainPhysics.NavierStokesUnfittedFreeSurfaceUsesCutMetadataScale:MovingDomainPhysics.NavierStokesUnfittedFreeSurfaceAddsCutCellStabilization'`
  passed.

## D18 Smoke Verification

Temporary run directory:
`/tmp/svmp_d18_fixture_cut_metadata_scale_onestep_39pJ7t`

Run setup:

- Source fixture: checked-in Test05 D18 wet-bed solver input with
  `Use_cut_metadata_scale=true`.
- Time steps: `1`.
- Restart output cadence: `1`.

Key result:

- Active domain: `LevelSetNegative`.
- Active-domain method: `CutVolume`.
- Linear solver: GMRES, diagonal preconditioner, relative tolerance `1e-4`,
  absolute tolerance `1e-4`, maximum iterations `8100`.
- Nonlinear controls: maximum iterations `8`, relative tolerance `1e-4`,
  absolute tolerance `1e-4`.
- Nonlinear result: converged in `2` iterations with residual
  `1.2326886177743722e-05`.
- Time loop result: `success=1`, `steps_taken=1`, final time `5.0e-04`.

## Remaining Work

- Full D18/D38 qualification still depends on the strict solver-tolerance path
  and the missing digitized D18 profile data.
- A full-run comparison with metadata scaling enabled should be recorded before
  final benchmark acceptance.

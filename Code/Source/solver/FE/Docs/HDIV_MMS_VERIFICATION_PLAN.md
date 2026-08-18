# H(div) Manufactured-Solution Verification Plan

## Objective

Add a dedicated manufactured-solution verification suite for `H(div)` spaces and mixed `H(div)-L2` formulations.

The target is not just more unit coverage. The target is end-to-end numerical verification that the FE library delivers the expected approximation properties for:

- pure `H(div)` fields
- scalar normal traces on `H(div)` fields
- mixed `H(div)-L2` problems such as Darcy-type formulations
- local conservation and divergence accuracy
- supported RT and BDM family pairings across the element families the FE library claims to support

This effort should remain FE-library focused and physics-agnostic. The tests may use Darcy-like mixed forms as the verification vehicle, but they should verify FE behavior rather than a physics-module API.

## Scope

### In Scope

- end-to-end manufactured-solution convergence tests for `H(div)` interpolation and projection accuracy
- end-to-end manufactured-solution convergence tests for mixed `H(div)-L2` systems
- error norms for:
  - `||q - q_h||_L2`
  - `||div(q - q_h)||_L2`
  - `||p - p_h||_L2`
  - boundary normal-trace error where applicable
- local mass-balance and conservation checks
- serial and MPI verification for at least one representative mixed `H(div)-L2` problem
- RT family coverage on broadly supported affine cell types
- BDM family coverage on the currently documented supported surface

### Out of Scope for This Effort

- Darcy-physics module registration or user-facing Physics-layer options
- nonlinear permeability laws or advanced constitutive models
- full qualification of unsupported or intentionally out-of-scope BDM family variants
- broad hybrid-cell mixed MMS qualification as a first-pass requirement
- replacing the existing basis-level and trace-level tests

## Current State Summary

### What Already Exists

- The FE library already has extensive basis- and operator-level `H(div)` validation:
  - RT and BDM moment and Kronecker-DOF tests in [test_VectorBases.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Tests/Unit/Basis/test_VectorBases.cpp)
  - wedge and pyramid RT divergence-flux consistency tests in [test_HigherOrderWedgePyramid.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Tests/Unit/Basis/test_HigherOrderWedgePyramid.cpp)
  - divergence and trace operator checks in [test_VectorSpaceOperators.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Tests/Unit/Spaces/test_VectorSpaceOperators.cpp) and [test_VectorSpaceTraces.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Tests/Unit/Spaces/test_VectorSpaceTraces.cpp)
  - trace restriction and interpolation checks in [test_TraceSpace.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Tests/Unit/Spaces/test_TraceSpace.cpp)
  - orientation and MPI ownership checks in [test_VectorBasisOrientationAssembly.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Tests/Unit/Assembly/test_VectorBasisOrientationAssembly.cpp), [test_HDivNormalConstraintMPI.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Tests/Unit/Constraints/test_HDivNormalConstraintMPI.cpp), and [test_HDivTracePeriodicMPI.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Tests/Unit/Constraints/test_HDivTracePeriodicMPI.cpp)
- The FE library already has an end-to-end manufactured-solution pattern for other spaces and mixed systems in:
  - [test_ManufacturedSolutionConvergence.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Tests/Unit/Assembly/test_ManufacturedSolutionConvergence.cpp)
  - [test_ManufacturedSolutionConvergence_ElementTypes.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Tests/Unit/Assembly/test_ManufacturedSolutionConvergence_ElementTypes.cpp)
- Mixed `H(div)-L2` pairs are already recognized as heuristically compatible in [SpaceCompatibility.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Spaces/SpaceCompatibility.cpp).

### What Is Missing

- There is no dedicated manufactured-solution convergence suite for pure `H(div)` problems.
- There is no dedicated manufactured-solution convergence suite for mixed `H(div)-L2` problems such as Darcy.
- The current manufactured-solution helpers in the assembly tests are H1- and `ProductSpace`-centric, not written for `H(div)` or `L2Space`.
- There is no current end-to-end test that verifies solved mixed-system conservation quality, flux accuracy, and pressure accuracy together.

## Verification Goals

The new MMS suite should answer the following questions directly:

1. Does interpolation or projection into `H(div)` spaces produce the expected `L2` and divergence accuracy?
2. Does the FE assembly path preserve those properties in solved mixed `H(div)-L2` systems?
3. Are boundary normal traces represented accurately enough to support flux-driven mixed problems?
4. Do the supported RT and BDM families achieve the expected rates on the supported cell types?
5. Does a distributed mixed `H(div)-L2` solve reproduce the same convergence behavior as the serial reference?

## Required Capability Model

The verification suite should measure:

1. Flux error in `L2`.
2. Divergence error in `L2`.
3. Pressure error in `L2`.
4. Boundary normal-trace error when boundary data is part of the manufactured problem.
5. Per-cell conservation residual after solve.
6. Dyadic convergence rates across mesh refinements.

For mixed problems, the baseline field naming should be:

- `q` for the `H(div)` flux field
- `p` for the `L2` scalar field

## Recommended Architecture

## Phase 1: Create Dedicated H(div) MMS Test Infrastructure

### Why

The existing manufactured-solution files are already large and primarily structured around `H1` and `ProductSpace`. `H(div)` verification will need its own error calculators, solve helpers, and mixed-system assembly patterns.

### Recommended Design

Create a new test file dedicated to `H(div)` MMS verification:

- `Code/Source/solver/FE/Tests/Unit/Assembly/test_ManufacturedSolutionConvergence_HDiv.cpp`

If helper reuse becomes large enough, move shared structured-mesh and convergence-rate helpers into a small test-only header under:

- `Code/Source/solver/FE/Tests/Unit/Assembly/TestSupport/`

### Concrete Files to Modify

- `Code/Source/solver/FE/Tests/Unit/Assembly/test_ManufacturedSolutionConvergence_HDiv.cpp`
- `Code/Source/solver/FE/Tests/Unit/Assembly/CMakeLists.txt`
- optional shared helper header under `Code/Source/solver/FE/Tests/Unit/Assembly/TestSupport/`

### Concrete Steps

1. Add a new dedicated `H(div)` MMS test file instead of continuing to grow the existing MMS files.
2. Reuse the current structured-mesh access style from the existing MMS tests where possible.
3. Keep `H(div)`-specific solve and error helpers local to the new file unless a clear shared test utility boundary emerges.
4. Preserve the existing dyadic-refinement testing pattern used by the current MMS tests.

## Phase 2: Add H(div)-Specific Error and Diagnostic Utilities

### Why

`H(div)` verification requires error measures that do not exist in the current MMS helpers. Scalar H1 `L2` error is not enough.

### Recommended Design

Add helper functions for:

- `computeL2ErrorHDiv(...)`
- `computeDivErrorHDiv(...)`
- `computeL2ErrorL2(...)`
- `computeBoundaryNormalTraceError(...)`
- `computeLocalMassBalanceResidual(...)`
- `convergenceRatesDyadic(...)` or reuse the existing pattern by local adaptation

### Concrete Files to Modify

- `Code/Source/solver/FE/Tests/Unit/Assembly/test_ManufacturedSolutionConvergence_HDiv.cpp`
- optional shared helper header if utility extraction is justified

### Concrete Steps

1. Implement `L2` flux error integration using `HDivSpace::evaluate(...)`.
2. Implement divergence error integration using `HDivSpace::evaluate_divergence(...)`.
3. Implement `L2` pressure error integration for `L2Space`.
4. Implement boundary normal-trace error evaluation using either `HDivSpace::normal_trace(...)` or a `TraceSpace` restriction path.
5. Implement cellwise conservation checks by integrating `div(q_h) - f` per element.

## Phase 3: Add Pure H(div) Manufactured Verification Cases

### Why

Before verifying mixed solves, the suite should verify the basic approximation properties of the `H(div)` space itself.

### Recommended Design

Start with two categories of pure `H(div)` MMS:

- exact polynomial reproduction or projection cases
- smooth non-polynomial convergence cases

Also add a commuting-style verification where practical:

- compare `div(Π_h q)` against a projected or interpolated representation of `div(q)`

### Concrete Files to Modify

- `Code/Source/solver/FE/Tests/Unit/Assembly/test_ManufacturedSolutionConvergence_HDiv.cpp`

### Concrete Steps

1. Add one exact-reproduction case using a field representable in the target RT or BDM space.
2. Add one smooth analytic flux field that is not exactly representable and verify rate-based convergence.
3. Add a divergence-focused convergence check for the same field.
4. Add a commuting or projection-consistency check where the representation path is unambiguous for the chosen element family.
5. Add a boundary normal-trace accuracy test using the same manufactured field on a tagged boundary.

## Phase 4: Add Baseline Mixed H(div)-L2 Darcy MMS Cases

### Why

The main missing verification layer is the solved mixed problem, not the basis in isolation.

### Recommended Design

Use a standard mixed Darcy-style system as the FE verification vehicle:

- `q + K grad(p) = 0`
- `div(q) = f`

with constant `K = I` for the first pass.

Choose an analytic scalar potential `p_exact`, define:

- `q_exact = -grad(p_exact)`
- `f = div(q_exact)`

and solve for `(q_h, p_h)`.

### Baseline Boundary Strategy

Use the simplest robust mixed verification path first:

- prescribe `q·n = q_exact·n` on the entire boundary
- pin one pressure DOF or fix pressure mean as needed to remove the nullspace

### Concrete Files to Modify

- `Code/Source/solver/FE/Tests/Unit/Assembly/test_ManufacturedSolutionConvergence_HDiv.cpp`

### Concrete Steps

1. Add a mixed solve helper for an `H(div)-L2` system on structured meshes.
2. Assemble the mixed block system using the FE forms layer rather than a physics-module wrapper.
3. Apply strong normal-flux boundary data through the FE-side `H(div)` normal-trace support.
4. Add one pressure-nullspace treatment that is deterministic and easy to reuse in tests.
5. Measure `||q-q_h||_L2`, `||div(q-q_h)||_L2`, and `||p-p_h||_L2`.
6. Add a local mass-balance assertion on each cell.

## Phase 5: Add Supported Element-Family Coverage

### Why

The MMS suite should verify the actual documented support surface instead of silently assuming that all `H(div)` families are equally broad.

### Recommended Design

Use a staged coverage matrix:

- RT first, because the support surface is broadest
- BDM second, but only on the topologies and orders explicitly documented as supported
- wedge and pyramid mixed MMS later, after affine-cell families are stable

### Recommended First-Pass Matrix

- RT:
  - `Triangle3`, orders `0` and `1`
  - `Quad4`, orders `0` and `1`
  - `Tetra4`, orders `0` and `1`
  - `Hex8`, orders `0` and `1`
- BDM:
  - `Triangle3`, orders `1` and `2`
  - `Tetra4`, orders `1` and `2`
  - retained order-1 quadrilateral path only

This matches the currently documented support surface in [Basis/README.md](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Basis/README.md).

### Concrete Files to Modify

- `Code/Source/solver/FE/Tests/Unit/Assembly/test_ManufacturedSolutionConvergence_HDiv.cpp`

### Concrete Steps

1. Implement RT mixed MMS on one 2D simplex and one 2D tensor-product topology first.
2. Extend to one 3D simplex and one 3D tensor-product topology.
3. Add BDM mixed MMS for supported simplex families.
4. Add the retained quadrilateral BDM order-1 case only if the mixed solve path is well behaved there.
5. Keep wedge and pyramid mixed MMS explicitly deferred until the affine-cell suite is qualified.

## Phase 6: Add Weak-Boundary and Trace-Focused Mixed MMS Cases

### Why

Strong normal-flux data on the whole boundary is enough for a baseline mixed verification, but the newer FE trace infrastructure should also be exercised by MMS.

### Recommended Design

Add a second-pass mixed MMS case that uses weak trace enforcement where mathematically appropriate, for example:

- pressure-driven boundary data through a weak trace term
- Nitsche-style trace enforcement where supported and stable for the chosen formulation

### Concrete Files to Modify

- `Code/Source/solver/FE/Tests/Unit/Assembly/test_ManufacturedSolutionConvergence_HDiv.cpp`
- optional related helper tests if common utilities are introduced

### Concrete Steps

1. Add one pressure-boundary mixed MMS case that does not rely on full-boundary strong flux data.
2. Reuse the same analytic solution family used by the baseline mixed tests where possible.
3. Measure the same flux, divergence, and pressure errors as in the strong-boundary baseline.
4. Verify that the new weak-boundary path preserves the expected convergence rates.

## Phase 7: Add MPI Qualification

### Why

`H(div)` mixed problems stress ownership, trace handling, and assembly paths that should be verified in distributed runs, not just serial.

### Recommended Design

Add at least one representative mixed `H(div)-L2` MMS case to the MPI assembly or systems test path:

- one RT case on a simple structured quad or hex mesh
- compare serial and MPI error norms and observed rates

### Concrete Files to Modify

- `Code/Source/solver/FE/Tests/Unit/Assembly/test_ManufacturedSolutionConvergence_HDiv.cpp`
- or a dedicated MPI companion file if that keeps the structure cleaner
- `Code/Source/solver/FE/Tests/Unit/Assembly/CMakeLists.txt`

### Concrete Steps

1. Add one 2-rank mixed RT MMS case.
2. Verify that serial and MPI runs produce matching error norms within tolerance.
3. Verify that observed dyadic rates remain within the expected lower bound.
4. Keep the MPI scope narrow at first to avoid turning the verification suite into a distributed-mesh stress test.

## Expected Manufactured Fields

Recommended first-pass scalar potential:

- `p(x, y) = sin(pi x) sin(pi y)` in 2D
- `p(x, y, z) = sin(pi x) sin(pi y) sin(pi z)` in 3D

Then define:

- `q = -grad(p)`
- `f = div(q)`

Recommended pure `H(div)` analytic flux family:

- one smooth non-polynomial vector field with nontrivial divergence and nontrivial normal trace
- one exactly representable polynomial field for reproduction tests

## Expected Assertions

### Pure H(div)

- exact polynomial cases should converge to roundoff or near-roundoff
- non-polynomial flux cases should satisfy:
  - `||q-q_h||_L2` rate near the expected family order
  - `||div(q-q_h)||_L2` rate near the expected divergence order

### Mixed H(div)-L2

- `||q-q_h||_L2` should converge at the expected mixed-method rate for the chosen family
- `||div(q-q_h)||_L2` should converge at the expected divergence rate
- `||p-p_h||_L2` should converge at the expected scalar order for the chosen pair
- per-cell conservation residual should be near solver or quadrature tolerance

Use rate assertions with practical lower bounds rather than exact equalities, for example:

- `rate > expected_rate - 0.25`

to avoid brittle failures.

## Concrete Test Inventory

Recommended first-pass test names:

- `ManufacturedSolutionConvergenceHDiv.RT0Quad_FluxProjection_ReproducesPolynomialMoments`
- `ManufacturedSolutionConvergenceHDiv.RT1Quad_FluxAndDivergenceConverge`
- `ManufacturedSolutionConvergenceHDiv.RT1Triangle_CommutingProjectionMatchesProjectedDivergence`
- `ManufacturedSolutionConvergenceHDiv.BDM1Triangle_FluxAndDivergenceConverge`
- `ManufacturedSolutionConvergenceHDiv.DarcyRT0Quad_FluxPressureAndDivergenceConverge`
- `ManufacturedSolutionConvergenceHDiv.DarcyRT1Triangle_FluxPressureAndDivergenceConverge`
- `ManufacturedSolutionConvergenceHDiv.DarcyBDM1Triangle_FluxPressureAndDivergenceConverge`
- `ManufacturedSolutionConvergenceHDiv.DarcyRT0Quad_LocalMassBalanceIsSatisfied`
- `ManufacturedSolutionConvergenceHDivMPI.DarcyRT0Quad_SerialParallelErrorsAgree`

## Design Decisions to Lock Before Coding

The following decisions should be fixed before implementation:

1. Keep the `H(div)` MMS suite in a new dedicated file instead of extending the current H1-centric MMS files.
2. Use RT as the baseline family and broaden only after RT mixed MMS is stable.
3. Limit BDM verification to the currently documented supported surface.
4. Use strong full-boundary normal-flux data for the first mixed MMS slice.
5. Defer wedge and pyramid mixed MMS until affine-cell mixed MMS is already qualified.
6. Add MPI coverage only after the serial mixed MMS slice is stable.

## Completion Checklist

### Infrastructure

- [ ] Add a new dedicated `H(div)` MMS test file.
- [ ] Add `H(div)` flux `L2` error integration utilities.
- [ ] Add `H(div)` divergence-error integration utilities.
- [ ] Add `L2Space` pressure-error integration utilities.
- [ ] Add boundary normal-trace error utilities.
- [ ] Add cellwise conservation diagnostics.

### Pure H(div) Verification

- [ ] Add an exact polynomial reproduction or projection test.
- [ ] Add a smooth non-polynomial flux convergence test.
- [ ] Add a divergence convergence test.
- [ ] Add a commuting or projection-consistency verification.
- [ ] Add a boundary normal-trace accuracy test.

### Mixed H(div)-L2 Verification

- [ ] Add a baseline mixed RT manufactured solve helper.
- [ ] Add strong normal-flux boundary data to the mixed MMS path.
- [ ] Add pressure nullspace handling for mixed solves.
- [ ] Verify `||q-q_h||_L2`.
- [ ] Verify `||div(q-q_h)||_L2`.
- [ ] Verify `||p-p_h||_L2`.
- [ ] Verify cellwise mass balance.

### Element-Family Coverage

- [ ] Add RT triangle coverage.
- [ ] Add RT quadrilateral coverage.
- [ ] Add RT tetrahedron coverage.
- [ ] Add RT hexahedron coverage.
- [ ] Add supported simplex BDM coverage.
- [ ] Add retained quadrilateral BDM order-1 coverage if stable.

### Advanced and Distributed Coverage

- [ ] Add one weak-boundary or trace-driven mixed MMS case.
- [ ] Add one 2-rank MPI mixed MMS case.
- [ ] Verify serial-vs-MPI error agreement.
- [ ] Verify serial-vs-MPI observed-rate agreement.

### Qualification

- [ ] Run the dedicated `H(div)` MMS tests in serial.
- [ ] Run the representative MPI `H(div)` MMS test.
- [ ] Re-run the full FE test suite.

## Definition of Done

This effort is complete when:

1. The FE test suite contains a dedicated manufactured-solution verification path for pure `H(div)` spaces and mixed `H(div)-L2` problems.
2. The suite verifies flux, divergence, pressure, and conservation behavior rather than only low-level basis identities.
3. RT and BDM coverage matches the currently documented support surface.
4. At least one distributed mixed MMS case demonstrates serial-vs-MPI agreement for the measured errors.
5. The existing FE suite remains green after the new verification tests are added.

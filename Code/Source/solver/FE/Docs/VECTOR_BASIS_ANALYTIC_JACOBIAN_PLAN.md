# Vector-Basis Analytic Jacobian Plan

## Objective

Add analytic Jacobian support for intrinsic vector-valued finite element spaces, analogous to the existing scalar-basis gradient support.

The target is to remove the finite-difference fallback currently used for vector-valued spaces in `FunctionSpace::evaluate_jacobian()` and to make `grad(u)` for intrinsic vector-basis fields usable, accurate, and performant in interpreter, AD, and JIT assembly paths.

This is FE-library infrastructure only. It must remain physics-agnostic and should support multiple downstream formulations, including `H(div)` and `H(curl)` methods that require vector-field gradients.

## Scope

### In Scope

- analytic reference-space vector-basis Jacobians for RT, BDM, Nedelec, and compatible tensor vector bases
- affine physical-space Jacobian transforms for `H(div)` and `H(curl)` Piola-mapped bases
- `FunctionSpace::evaluate_jacobian()` integration for vector-valued spaces
- matrix-valued vector-gradient storage and accessors in assembly contexts
- Forms interpreter and JIT support for `grad(vector intrinsic basis)`
- tests proving supported vector spaces no longer use finite differences
- clear non-affine guards until full curved Piola-derivative support is implemented

### Out of Scope for This Effort

- changing scalar-basis gradient behavior
- changing vector-valued `H1` product-space semantics beyond using existing scalar gradients componentwise
- non-affine or curved Piola derivative support requiring derivatives of `J`, `J^{-1}`, and `detJ`
- Navier-Stokes-specific formulation changes
- physics-specific operators or BCs

## Current State Summary

### What Already Exists

- Scalar bases expose `evaluate_gradients(...)`, and scalar `FunctionSpace::evaluate_jacobian()` uses those analytic gradients.
- Intrinsic vector bases expose vector values, divergence, and curl through `evaluate_vector_values(...)`, `evaluate_divergence(...)`, and `evaluate_curl(...)`.
- `H(div)` and `H(curl)` vector values are already mapped to physical space through Piola transforms.
- The assembly context already stores vector basis values, vector curls, and vector divergences.
- The form language already has `grad(...)`, `div(...)`, `curl(...)`, `sym(...)`, AD, symbolic differentiation, and JIT infrastructure.

### What Is Missing

- Vector bases do not expose per-basis Jacobians `d phi_i / d xi`.
- `FunctionSpace::evaluate_jacobian()` finite-differences vector-valued field evaluation.
- Assembly context exposes scalar gradients as vectors, but does not expose intrinsic vector-basis gradients as matrices.
- Forms lowering for `grad(vector intrinsic basis)` cannot read analytic matrix-valued basis Jacobians.
- JIT argument packing does not currently provide vector-basis Jacobian matrices.

## Required Capability Model

The FE layer should expose the following generic capabilities:

1. Reference-space vector basis Jacobians for supported intrinsic vector bases.
2. Physical-space vector basis Jacobians under affine mappings.
3. Matrix-valued test, trial, current-solution, previous-solution, and auxiliary-field vector gradients in assembly contexts.
4. Forms interpreter and JIT support for `grad(u)` where `u` is an intrinsic vector-basis field.
5. Explicit failure for unsupported non-affine vector-basis Jacobians.
6. Regression tests proving analytic behavior and preserving scalar behavior.

## Recommended Architecture

## Phase 1: Add Vector-Basis Jacobian API

### Why

The current scalar derivative path has a clear public API, but intrinsic vector bases only expose values, divergence, and curl. A first-class vector-Jacobian API is needed before the space, assembly, and forms layers can avoid finite differences.

### Recommended Design

Add a reference-space API that returns one `3x3` matrix per vector basis function:

- rows: value component
- columns: reference-coordinate derivative
- unused rows and columns set to zero for lower-dimensional elements

The default implementation should throw unless a derived vector basis explicitly supports analytic Jacobians.

### Concrete Files to Modify

- `Code/Source/solver/FE/Basis/BasisFunction.h`
- `Code/Source/solver/FE/Basis/VectorBasis.h`
- `Code/Source/solver/FE/Basis/VectorBasis.cpp`
- `Code/Source/solver/FE/Basis/CompatibleTensorVectorBasis.h`
- `Code/Source/solver/FE/Basis/CompatibleTensorVectorBasis.cpp`

### Concrete Steps

1. Add a shared matrix alias for vector-basis Jacobians if one is not already suitable.
2. Add `evaluate_vector_jacobians(xi, jacobians)` to `BasisFunction`.
3. Override the API in `VectorBasisFunction` and supported derived classes.
4. Add an optional capability query if staged rollout needs feature detection.
5. Document reference-space shape and row/column conventions next to the API.

### Checklist

- [x] Add vector-Jacobian type alias and public API to `BasisFunction`.
- [x] Add default throwing implementation for scalar or unsupported bases.
- [x] Add override declarations for RT, BDM, Nedelec, and compatible tensor vector bases.
- [x] Add optional capability query or documented exception contract.
- [x] Add API comments documenting matrix layout and lower-dimensional zero-fill rules.
- [x] Add unit tests that unsupported calls fail clearly.

## Phase 2: Implement Reference-Space Analytic Derivatives

### Why

The existing RT, BDM, and Nedelec implementations already evaluate vector basis functions from component-wise polynomial data. Their reference derivatives should be computed analytically from the same polynomial representation.

### Recommended Design

For modal polynomial vector bases, differentiate each monomial term directly:

- if term is `c * x^a y^b z^c e_k`, then derivative with respect to each active coordinate is the corresponding monomial derivative in row `k`
- combine modal derivatives with the existing nodal coefficient map

For compatible tensor vector bases, use scalar component-basis gradients and place each component-gradient row into the vector-Jacobian matrix.

### Concrete Files to Modify

- `Code/Source/solver/FE/Basis/VectorBasis.cpp`
- `Code/Source/solver/FE/Basis/CompatibleTensorVectorBasis.cpp`
- `Code/Source/solver/FE/Tests/Unit/Basis/test_VectorBases.cpp`
- `Code/Source/solver/FE/Tests/Unit/Basis/test_CompatibleTensorVectorBasis.cpp` if present, or the nearest existing compatible tensor-vector test file

### Concrete Steps

1. Add a helper that evaluates derivatives of modal polynomial vector terms.
2. Implement RT reference Jacobians through the modal/nodal coefficient representation.
3. Implement BDM reference Jacobians through the modal/nodal coefficient representation.
4. Implement Nedelec reference Jacobians through the modal/nodal coefficient representation.
5. Implement compatible tensor vector reference Jacobians from scalar component gradients.
6. Verify divergence and curl computed from the new Jacobians agree with existing `evaluate_divergence(...)` and `evaluate_curl(...)`.

### Checklist

- [x] Implement shared modal-vector polynomial derivative helper.
- [x] Implement `RaviartThomasBasis::evaluate_vector_jacobians(...)`.
- [x] Implement `BDMBasis::evaluate_vector_jacobians(...)`.
- [x] Implement `NedelecBasis::evaluate_vector_jacobians(...)`.
- [x] Implement `CompatibleTensorVectorBasis::evaluate_vector_jacobians(...)`.
- [x] Add exact polynomial derivative tests for RT on supported affine element families.
- [x] Add exact polynomial derivative tests for BDM on supported element families.
- [x] Add exact polynomial derivative tests for Nedelec on supported element families.
- [x] Add derivative tests for compatible tensor vector bases.
- [x] Add divergence-from-Jacobian parity tests for `H(div)` bases.
- [x] Add curl-from-Jacobian parity tests for `H(curl)` bases.

## Phase 3: Add Physical-Space Vector-Jacobian Transforms

### Why

Reference derivatives are not enough for assembled physical operators. The FE layer needs correct physical derivatives for each mapping family.

### Recommended Design

Add vector-Jacobian push-forward helpers for affine mappings:

- vector-valued `H1` product space: `grad_x v = grad_xi v * J^{-1}`
- `H(div)` contravariant Piola: `grad_x v = (1 / detJ) * J * grad_xi(v_hat) * J^{-1}`
- `H(curl)` covariant Piola: `grad_x v = J^{-T} * grad_xi(v_hat) * J^{-1}`

Supported non-affine 3D curved volume `H(div)` and `H(curl)` vector-Jacobian transforms now use analytic curved Piola derivatives, including physical derivatives of `J`, `detJ`, `J^{-1}`, and `J^{-T}`. Unsupported curved combinations should continue to throw clearly instead of falling back to affine formulas.

### Concrete Files to Modify

- `Code/Source/solver/FE/Geometry/PushForward.h`
- `Code/Source/solver/FE/Geometry/PushForward.cpp`
- `Code/Source/solver/FE/Elements/ElementTransform.h`
- `Code/Source/solver/FE/Elements/ElementTransform.cpp`
- `Code/Source/solver/FE/Geometry/GeometryMapping.h`
- `Code/Source/solver/FE/Geometry/LinearMapping.h`
- `Code/Source/solver/FE/Tests/Unit/Geometry/`
- `Code/Source/solver/FE/Tests/Unit/Elements/`

### Concrete Steps

1. Add affine vector-Jacobian transform helpers to `PushForward`.
2. Add batch transform wrappers to `ElementTransform`.
3. Add a mapping capability check that distinguishes affine from non-affine mappings.
4. Guard non-affine `H(div)` and `H(curl)` vector-Jacobian transforms with a descriptive exception.
5. Verify transformed gradients against analytic physical derivatives on affine cells.

### Checklist

- [x] Add `PushForward` helper for ordinary vector-value Jacobian transforms.
- [x] Add `PushForward` helper for affine `H(div)` Piola vector-Jacobian transforms.
- [x] Add `PushForward` helper for affine `H(curl)` Piola vector-Jacobian transforms.
- [x] Add `ElementTransform` batch helpers for vector-basis Jacobians.
- [x] Add or reuse an affine mapping capability query.
- [x] Add explicit non-affine guard tests.
- [x] Add affine triangle and quad Piola derivative tests.
- [x] Add affine tetrahedron and hex Piola derivative tests.

## Phase 4: Integrate with `FunctionSpace::evaluate_jacobian()`

### Why

The public function-space surface is where the current finite-difference behavior is visible. Once basis and mapping support exist, vector-valued `evaluate_jacobian()` should use the analytic path.

### Recommended Design

Keep scalar behavior unchanged. For intrinsic vector-valued spaces, call the new vector-basis Jacobian API and return the coefficient-weighted sum. For reference-space `FunctionSpace` evaluation, this returns the reference Jacobian. Physical-space transforms remain owned by assembly and element-transform paths.

### Concrete Files to Modify

- `Code/Source/solver/FE/Spaces/FunctionSpace.h`
- `Code/Source/solver/FE/Spaces/FunctionSpace.cpp`
- `Code/Source/solver/FE/Tests/Unit/Spaces/test_FunctionSpaceGradients.cpp`
- `Code/Source/solver/FE/Tests/Unit/Spaces/test_VectorSpaceOperators.cpp`

### Concrete Steps

1. Replace the vector-valued finite-difference branch with analytic `evaluate_vector_jacobians(...)`.
2. Preserve scalar `evaluate_gradient(...)` and scalar `evaluate_jacobian(...)` behavior exactly.
3. Add tests for representative RT, BDM, Nedelec, and compatible tensor vector spaces.
4. Add a regression that would fail if finite differences were still used for supported vector bases.

### Checklist

- [x] Replace vector finite-difference branch in `FunctionSpace::evaluate_jacobian()`.
- [x] Preserve scalar gradient and scalar Jacobian tests unchanged.
- [x] Add RT `FunctionSpace::evaluate_jacobian()` analytic tests.
- [x] Add BDM `FunctionSpace::evaluate_jacobian()` analytic tests.
- [x] Add Nedelec `FunctionSpace::evaluate_jacobian()` analytic tests.
- [x] Add compatible tensor vector-space analytic tests.
- [x] Add no-finite-difference regression for supported vector spaces.

## Phase 5: Add Assembly Context Vector-Jacobian Plumbing

### Why

Assembly currently stores vector basis values, curls, and divergences, but no matrix-valued vector gradients. Forms that use `grad(u)` for intrinsic vector fields need this data in the same way scalar forms need `physicalGradient(...)`.

### Recommended Design

Add matrix-valued vector-Jacobian storage and accessors parallel to the existing vector-basis storage:

- test basis vector Jacobians
- trial basis vector Jacobians
- current solution vector Jacobians
- previous solution vector Jacobians
- auxiliary field vector Jacobians

Keep existing scalar-gradient accessors scalar-only to avoid shape ambiguity.

### Concrete Files to Modify

- `Code/Source/solver/FE/Assembly/AssemblyContext.h`
- `Code/Source/solver/FE/Assembly/AssemblyContext.cpp`
- `Code/Source/solver/FE/Assembly/StandardAssembler.h`
- `Code/Source/solver/FE/Assembly/StandardAssembler.cpp`
- `Code/Source/solver/FE/Assembly/JIT/KernelArgs.h`
- `Code/Source/solver/FE/Tests/Unit/Assembly/test_AssemblyContext.cpp`

### Concrete Steps

1. Add storage arrays for test and trial vector-basis Jacobians.
2. Add storage arrays for vector-valued solution and field Jacobians.
3. Add accessors such as `basisVectorJacobian(i,q)` and `trialBasisVectorJacobian(j,q)`.
4. Add accessors such as `solutionVectorJacobian(q)` and `fieldVectorJacobian(field,q)`.
5. Add setters and raw-span accessors needed by assembler and JIT argument packing.
6. Update context snapshot, restore, reset, and arena allocation logic.

### Checklist

- [x] Add test vector-basis Jacobian storage.
- [x] Add trial vector-basis Jacobian storage.
- [x] Add current-solution vector-Jacobian storage.
- [x] Add previous-solution vector-Jacobian storage.
- [x] Add auxiliary-field vector-Jacobian storage.
- [x] Add public accessors for test and trial vector-basis Jacobians.
- [x] Add public accessors for solution and field vector Jacobians.
- [x] Add setter APIs and raw-span APIs.
- [x] Update snapshot, restore, reset, and arena allocation paths.
- [x] Add assembly-context unit tests for storage, access, copy, and error paths.

## Phase 6: Wire Assembler and Forms Interpreter

### Why

The basis and context layers only make data available. The assembler must populate that data, and the Forms interpreter must read it when lowering `grad(...)` on intrinsic vector-basis fields.

### Recommended Design

Extend required-data analysis so `grad(vector intrinsic field)` requests vector Jacobians. Then populate physical vector Jacobians during cell, boundary, interior-face, and interface assembly wherever vector gradients are requested.

Interpreter lowering should return matrix-valued expressions for vector `grad(...)` using the new context accessors.

### Concrete Files to Modify

- `Code/Source/solver/FE/Forms/FormCompiler.cpp`
- `Code/Source/solver/FE/Forms/FormKernels.cpp`
- `Code/Source/solver/FE/Forms/SymbolicDifferentiation.cpp`
- `Code/Source/solver/FE/Assembly/StandardAssembler.cpp`
- `Code/Source/solver/FE/Assembly/FunctionalAssembler.cpp`
- `Code/Source/solver/FE/Tests/Unit/Forms/`
- `Code/Source/solver/FE/Tests/Unit/Assembly/`

### Concrete Steps

1. Extend required-data analysis for vector-field gradients.
2. Populate test and trial vector-basis Jacobians in cell assembly.
3. Populate vector-basis Jacobians in boundary, interior-face, and interface assembly paths.
4. Populate current, previous, and auxiliary vector-field Jacobians from coefficients.
5. Update interpreter lowering for `grad(TestFunction)`, `grad(TrialFunction)`, and `grad(StateField)` on intrinsic vector fields.
6. Preserve existing scalar and product-space vector behavior.

### Checklist

- [x] Add required-data flag or extend existing gradient requirement for vector Jacobians.
- [x] Populate cell test vector Jacobians.
- [x] Populate cell trial vector Jacobians.
- [x] Populate boundary vector Jacobians where required.
- [x] Populate interior-face vector Jacobians where required.
- [x] Populate interface vector Jacobians where required.
- [x] Populate current-solution vector Jacobians.
- [x] Populate previous-solution vector Jacobians.
- [x] Populate auxiliary-field vector Jacobians.
- [x] Update interpreter `grad(TestFunction)` vector-basis lowering.
- [x] Update interpreter `grad(TrialFunction)` vector-basis lowering.
- [x] Update interpreter `grad(StateField)` vector-basis lowering.
- [x] Add forms tests for `inner(grad(u), grad(v))` on supported vector bases.
- [x] Add forms tests for `sym(grad(u))` on supported vector bases.
- [x] Add AD finite-difference parity tests for vector-basis gradient forms.

## Phase 7: Add JIT Support

### Why

The FE forms path uses both interpreter and JIT backends. New vector-Jacobian support must be available in the JIT path, or vector-gradient-heavy formulations will be backend-dependent.

### Recommended Design

Flatten vector-Jacobian matrices into JIT kernel arguments using a deterministic layout. Update LLVM lowering for `grad(vector intrinsic field)` to read matrix-valued data from those arguments.

### Concrete Files to Modify

- `Code/Source/solver/FE/Assembly/JIT/KernelArgs.h`
- `Code/Source/solver/FE/Forms/JIT/JITKernelWrapper.cpp`
- `Code/Source/solver/FE/Forms/JIT/LLVMGen.cpp`
- `Code/Source/solver/FE/Tests/Unit/Forms/test_JIT_ExtendedParity.cpp`
- `Code/Source/solver/FE/Tests/Unit/Forms/test_JIT_TangentFiniteDifferences.cpp`

### Concrete Steps

1. Add flattened vector-Jacobian fields to JIT kernel argument structs.
2. Fill those fields from `AssemblyContext` raw spans.
3. Update LLVM lowering for vector-basis `grad(...)`.
4. Add JIT/interpreter parity tests.
5. Add JIT tangent finite-difference tests.

### Checklist

- [x] Add flattened test vector-Jacobian JIT arguments.
- [x] Add flattened trial vector-Jacobian JIT arguments.
- [x] Add flattened solution and field vector-Jacobian JIT arguments.
- [x] Fill vector-Jacobian arguments from `AssemblyContext`.
- [x] Update LLVM lowering for vector-basis `grad(TestFunction)`.
- [x] Update LLVM lowering for vector-basis `grad(TrialFunction)`.
- [x] Update LLVM lowering for vector-basis `grad(StateField)`.
- [x] Add JIT/interpreter parity tests.
- [x] Add JIT tangent finite-difference tests.

## Phase 8: Verification, Documentation, and Cleanup

### Why

This infrastructure changes derivative data used by vector-basis forms. It needs direct mathematical verification, backend parity, and clear public documentation before physics modules depend on it.

### Recommended Design

Use exact polynomial derivative tests as the primary source of truth. Use finite differences only as diagnostic comparisons, not as implementation behavior. Add examples showing how vector-basis `grad(...)` should be used and when non-affine Piola derivatives remain unsupported.

### Concrete Files to Modify

- `Code/Source/solver/FE/Docs/HDIV_ADVANCED_USAGE_GUIDE.md`
- `Code/Source/solver/FE/Docs/HDIV_MMS_VERIFICATION_PLAN.md`
- `Code/Source/solver/FE/README.md`
- `Code/Source/solver/FE/Forms/VOCABULARY.md`
- `Code/Source/solver/FE/Forms/SYSTEMS_INTEGRATION.md`
- targeted basis, spaces, forms, assembly, and JIT tests

### Concrete Steps

1. Add exact derivative verification for each supported vector basis family.
2. Add physical affine transform verification across representative element topologies.
3. Add Forms interpreter and JIT parity tests.
4. Add tests proving unsupported non-affine cases fail explicitly.
5. Update public docs with the new vector-gradient capability and limitations.
6. Run targeted tests and then the full FE suite.

### Checklist

- [x] Add exact reference-derivative verification for RT.
- [x] Add exact reference-derivative verification for BDM.
- [x] Add exact reference-derivative verification for Nedelec.
- [x] Add exact reference-derivative verification for compatible tensor vector bases.
- [x] Add physical affine-transform verification for simplex cells.
- [x] Add physical affine-transform verification for tensor-product cells.
- [x] Add interpreter/JIT parity tests for vector-basis gradient forms.
- [x] Add AD tangent finite-difference tests for nonlinear vector-gradient forms.
- [x] Add explicit non-affine unsupported-case tests.
- [x] Update FE docs and vocabulary.
- [x] Add a short usage example for `grad(u)` on intrinsic vector spaces.
- [x] Run targeted basis, spaces, forms, assembly, and JIT tests.
- [x] Run the full `build-fe-check` suite and record non-vector-gradient blockers.

### Qualification Notes

- Targeted vector-basis analytic Jacobian tests pass for basis, spaces, geometry, assembly packing/context, Forms interpreter, JIT parity, and JIT tangent finite differences.
- Full `test_fe_basis`, `test_fe_spaces`, and `test_fe_geometry` pass.
- Full `test_fe_assembly` is not clean because `StandardAssemblerTest.ReverseScatterPolicyIsNormalizedToOwnedRowsOnly` expects reverse scatter to normalize to owned rows, while the current concurrent ghost-policy work preserves the requested policy. This is outside the vector-basis analytic Jacobian scope.
- Full `test_fe_assembly` and `test_fe_forms` also expose pre-existing functional-JIT failures where synthetic functional kernels return zero for constant/discrete-field functionals. The vector-basis gradient JIT tests pass, so this remains a separate JIT functional-kernel blocker.

## Implementation Decisions to Lock

- [x] Final public API name for vector-basis Jacobians.
- [x] Whether to expose a capability query or rely only on exceptions for unsupported bases.
- [x] Exact matrix layout for flattened JIT vector-Jacobian arrays.
- [x] Whether non-affine vector Piola Jacobians remain completely unsupported or are allowed through a diagnostic finite-difference development mode.
- [x] Whether `grad(vector field)` should be allowed immediately for both `H(div)` and `H(curl)` or staged by continuity family.

## Overall Completion Checklist

- [x] Phase 1 complete: vector-basis Jacobian API added.
- [x] Phase 2 complete: reference-space analytic derivatives implemented.
- [x] Phase 3 complete: affine physical-space vector-Jacobian transforms implemented.
- [x] Phase 4 complete: `FunctionSpace::evaluate_jacobian()` no longer finite-differences supported vector spaces.
- [x] Phase 5 complete: assembly context stores and exposes matrix-valued vector gradients.
- [x] Phase 6 complete: assembler and Forms interpreter consume analytic vector Jacobians.
- [x] Phase 7 complete: JIT consumes analytic vector Jacobians.
- [ ] Phase 8 complete: verification, docs, and full-suite qualification complete. Targeted verification and documentation are complete; full-suite pass is blocked by the non-vector-gradient issues listed above.

## Definition of Done

- [x] Supported intrinsic vector bases expose analytic reference-space Jacobians.
- [x] Supported affine `H(div)` and `H(curl)` vector-basis gradients are mapped analytically to physical space.
- [x] `FunctionSpace::evaluate_jacobian()` does not finite-difference supported vector spaces.
- [x] `grad(u)` works for intrinsic vector-basis fields in the interpreter path.
- [x] `grad(u)` works for intrinsic vector-basis fields in the JIT path.
- [x] Scalar gradient and scalar Jacobian behavior is unchanged.
- [x] Unsupported non-affine Piola-gradient cases fail with clear diagnostics.
- [x] Tests cover basis, space, assembly, interpreter, JIT, and AD behavior.
- [ ] The full FE test suite passes. Current blockers are unrelated to vector-basis analytic Jacobian support; see the Phase 8 qualification notes.

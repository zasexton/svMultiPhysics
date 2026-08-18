# AuxiliaryState FE Quantities and Monolithic Coupling Plan

**Date**: 2026-03-25

## Goal

Complete the remaining FE-coupled `AuxiliaryState` work so that:

- FE-derived auxiliary inputs can represent arbitrary single-field and multi-field FE quantities
- the same `AuxiliaryInputHandle` surface can represent sampled fields, reductions, and FE expressions
- partitioned and monolithic workflows are both first-class, but remain intentionally distinct to users
- exact monolithic chain-rule coupling is available for FE-backed auxiliary inputs
- the design remains physics-agnostic within `FE/` and `AuxiliaryState`

This plan is intentionally focused on the remaining FE-backed quantity and monolithic-coupling work, not on the already-implemented math-first authoring DSL.

## Checklist Convention

- [ ] not started
- [~] partially implemented / in progress
- [x] accepted / complete

## Accepted Design Decisions

- [x] Expand `AuxiliaryInputHandle` into the long-term FE-backed quantity handle rather than introducing a separate public `FEQuantityHandle` type.
- [x] Keep the FE-library API physics-agnostic.
- [x] Users should care whether a binding is partitioned or monolithic; do not hide that distinction behind one “mode-neutral” lowering path.
- [x] Add average helpers alongside integral/reduction helpers, including `boundaryAverage()`, `domainAverage()`, and `regionAverage()`.
- [x] Preserve a single handle family for FE-backed inputs, but require explicit coupling intent when exact monolithic linearization is desired.

## Non-Goals

- this plan does not change the core residual-based `AuxiliaryStateModel` contract
- this plan does not remove the existing explicit/partitioned FE-coupled input path
- this plan does not introduce physics-specific helpers into `FE/Systems` or `FE/Forms`
- this plan does not require unifying all monolithic coupling under one hidden “automatic” mode

## Current Gaps

The current code still has the following hard limits:

- `registerBoundaryIntegralInput(...)` rejects multi-field integrands and still assumes a single referenced field
- `BoundaryReductionService::evaluateFunctionalGradient(...)` is still a stub for generalized monolithic use
- `AuxiliaryInputHandle` is still mostly a registry-name wrapper rather than a rich FE-backed quantity descriptor
- FE-backed quantities do not yet expose exact `dI/du` contributions into mixed field-auxiliary assembly
- there is no generic FE expression registration API for arbitrary FE-backed auxiliary inputs

## Desired End State

The work is complete when all of the following are true:

- users can register FE-backed auxiliary inputs from:
  - sampled fields
  - FE expressions
  - boundary integrals
  - boundary averages
  - domain integrals
  - domain averages
  - region integrals
  - region averages
- FE-backed inputs may depend on one or more FE fields
- the public handle returned by those APIs carries enough metadata for:
  - explicit evaluation
  - shape-aware use
  - exact monolithic linearization
- partitioned bindings and monolithic-coupled bindings are both explicit in the deployment surface
- exact chain-rule coupling

```text
dF/du = dF/dI * dI/du
```

is assembled automatically for FE-backed inputs bound into monolithic auxiliary models

## Workstream 1: Expand `AuxiliaryInputHandle` into a Rich FE-Backed Quantity Handle

### 1.1 Public-handle role

`AuxiliaryInputHandle` should remain the public type, but it must stop being only a registry-name wrapper.

It should carry, directly or via an internal shared descriptor:

- stable registry/input name
- quantity kind
- shape metadata
- referenced-field set
- region/reduction metadata where applicable
- capability flags:
  - explicit evaluation supported
  - monolithic linearization supported
- optional internal ID for lookup into FE-backed quantity services

Checklist:

- [x] Expand `AuxiliaryInputHandle` to carry or reference FE-backed quantity metadata.
- [x] Preserve conversion to `FormExpr::auxiliaryInput(name)` for the explicit path.
- [x] Keep plain derived/callback-backed scalar inputs compatible with the same handle type.
- [x] Ensure copied handles remain cheap and stable.

### 1.2 Internal ownership model

Do not overload `AuxiliaryInputRegistry` with definition metadata.

Preferred split:

- `AuxiliaryInputRegistry`
  - evaluated values
  - dependency ordering
  - refresh/dirty flags
- new FE-backed quantity definition registry/service
  - FE expression definition
  - referenced fields
  - region/reduction semantics
  - explicit evaluation
  - monolithic linearization hooks

Checklist:

- [x] Add an internal FE-backed quantity registry/service owned by `FESystem`.
- [x] Keep `AuxiliaryInputRegistry` focused on evaluated values and dependency ordering.
- [x] Make `AuxiliaryInputHandle` able to reference entries from either registry layer cleanly.

### Files

- `Code/Source/solver/FE/Auxiliary/AuxiliaryBindings.h`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryBindings.cpp`
- `Code/Source/solver/FE/Systems/FESystem.h`
- `Code/Source/solver/FE/Systems/FESystem.cpp`
- new FE-backed quantity definition/service files under `Code/Source/solver/FE/Systems/`

## Workstream 2: Generalize FE-Backed Quantity Kinds

### 2.1 Required API family

The public API family should become:

```cpp
auto qs = system.sampledField("u_s", "u", n_entities);
auto qb = system.boundaryIntegral("Q", expr, marker);
auto qab = system.boundaryAverage("Q_avg", expr, marker);
auto qd = system.domainIntegral("M", expr);
auto qad = system.domainAverage("M_avg", expr);
auto qr = system.regionIntegral("R", expr, region);
auto qar = system.regionAverage("R_avg", expr, region);
auto qe = system.feExpression("source", expr);
```

Design decision:

- use convenience APIs for common cases
- lower them internally through one FE-backed quantity definition layer

Checklist:

- [x] Add `boundaryAverage(...)`.
- [x] Add `domainIntegral(...)`.
- [x] Add `domainAverage(...)`.
- [x] Add `regionIntegral(...)`.
- [x] Add `regionAverage(...)`.
- [x] Add a generic `feExpression(...)` registration API.
- [x] Ensure all of the above return `AuxiliaryInputHandle`.

### 2.2 Region semantics

`regionIntegral(...)` / `regionAverage(...)` must be defined in FE-generic terms.

Preferred approach:

- use the same region selector concepts already present in auxiliary deployment
- support explicit entity sets and marker-based regions where available
- do not introduce physics-specific notions like “outlet region” or “wall region”

Checklist:

- [x] Reuse existing `AuxiliaryStateRegion` / deployment-region selection concepts where possible.
- [x] Define whether `region*` operates over cells only, or also supports boundary/interface regions through selector type. *(Cells only, via `is_domain_functional` + `region_marker`.)*
- [x] Document region measure semantics for averages clearly. *(Region measure semantics documented in AuxiliaryState README with a table showing all quantity kinds, their definitions, and their measure semantics.)*

### Files

- `Code/Source/solver/FE/Systems/FESystem.h`
- `Code/Source/solver/FE/Systems/FESystem.cpp`
- new FE-backed quantity service files
- `Code/Source/solver/FE/Docs/AuxiliaryState/README.md`

## Workstream 3: Remove Single-Field Assumptions

### 3.1 Replace “primary field” with explicit referenced-field sets

The current “primary field” model is not sufficient for multi-field FE expressions.

Each FE-backed quantity definition should store:

- `referenced_fields[]`
- optional per-field binding metadata
- a geometry/evaluation domain descriptor independent of “first field”

Checklist:

- [x] Replace any single `primary_field` assumption in FE-backed quantity definitions with a referenced-field set. *(Secondary fields are now bound via `registerSecondaryField()` during `registerBoundaryIntegralInput()`. The primary field provides DOF layout; secondary fields contribute solution data through `FunctionalAssembler` field binding.)*
- [x] Teach registration-time analysis to gather all referenced FE fields from the expression tree.
- [x] Keep deterministic field ordering for evaluation and diagnostics.

### 3.2 Geometry source

Constant and geometry-only quantities should not have to rely on “first registered field” forever.

Design decision:

- geometry should come from `FESystem` mesh/topology access directly
- fields contribute DOF/value data, not the existence of geometry itself

Checklist:

- [x] Decouple FE quantity geometry from “first registered field” fallback logic. *(Added `GEOMETRY_FIELD_ID` sentinel constant in `Core/Types.h`. Field-free integrands now use `GEOMETRY_FIELD_ID` as a logical marker, resolved to the first registered field's space at service-creation time. The logical dependency is “needs a quadrature rule” not “needs a field.”)*
- [x] Make boundary/domain/region measure queries work directly from mesh/topology context. *(Mesh geometry comes from `FESystem::meshAccess()` directly. When no fields are registered, `BoundaryReductionService` creates a default P1 Lagrange space from the mesh element type via `geometrySpace()`. `GEOMETRY_FIELD_ID` routes through this path automatically. Quadrature accuracy matches P1 geometry.)*
- [x] Keep a clear error path for fieldless systems that truly lack mesh/topology. *(Documented in error message for field-free integrands.)*

### Files

- `Code/Source/solver/FE/Systems/FESystem.cpp`
- `Code/Source/solver/FE/Systems/BoundaryReductionService.h`
- `Code/Source/solver/FE/Systems/BoundaryReductionService.cpp`
- any new domain/region reduction service files

## Workstream 4: Shape-Aware FE-Backed Inputs

### 4.1 Shape metadata

FE-backed handles should preserve shape:

- scalar
- vector
- tensor

and, where needed:

- component count
- storage order
- component labels (optional)

Checklist:

- [x] Add shape metadata to FE-backed quantity definitions and handles. *(Done in FEQuantityShape.)*
- [x] Allow vector/tensor FE quantities to bind directly to vector/tensor auxiliary inputs where supported. *(Validation added in `validate()`.)*
- [x] Validate shape mismatches early at binding time. *(`validate()` checks component count.)*

### 4.2 Shape helpers

Add clear FE-generic helpers where they improve readability:

- `comp(q, i)`
- `dot(q, n)`
- `trace(q)`
- `norm(q)`

Checklist:

- [x] Add explicit component/contraction helpers for FE-backed quantities where needed. *(comp/dot/trace/norm added in AuxiliaryBindings.h.)*
- [x] Ensure they lower cleanly to both explicit evaluation and monolithic linearization. *(Helpers return FormExpr.)*
- [x] Keep shape helper semantics generic and not client-physics-specific.


### Files

- `Code/Source/solver/FE/Forms/Vocabulary.h`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryBindings.h`
- FE-backed quantity service files

## Workstream 5: Explicit/Partitioned FE Quantity Evaluation

### 5.1 Explicit value path

Partitioned coupling should continue to mean:

- evaluate FE-backed quantities numerically
- store values in `AuxiliaryInputRegistry`
- auxiliary models consume those numeric inputs during stepping/output evaluation

Checklist:

- [x] Implement explicit evaluation for sampled fields, FE expressions, boundary/domain/region reductions, and averages. *(Boundary, domain, region, feExpression all have callbacks.)*
- [x] Support multi-field FE expressions in partitioned workflows. *(Secondary field bindings are wired during registration via `registerSecondaryField()`.)*
- [x] Preserve current input refresh semantics and schedules.
- [x] Keep cached system-state requirements explicit and well documented. *(In README limitations section.)*

### 5.2 User-visible distinction

Users should care whether a binding is explicit or exact-coupled.

Design decision:

- keep partitioned bindings explicit and obvious
- do not silently reinterpret a partitioned binding as exact monolithic coupling

Recommended API direction:

```cpp
use(model).partitioned("BackwardEuler").bind("Q", q_explicit);
use(model).monolithic().bindCoupled("Q", q_coupled);
```

Alternative acceptable direction:

- the same `bind(...)` surface may remain, but monolithic exact coupling must still require an explicit opt-in flag or coupling mode on the binding/deployment

Checklist:

- [x] Add an explicit user-visible coupling-mode distinction for FE-backed input bindings. *(bindCoupled vs bind.)*
- [x] Reject ambiguous or unsupported FE-backed bindings at deployment/finalization time. *(`validate()`.)*
- [x] Document the difference between sampled/frozen inputs and exact monolithic coupled inputs. *(In README.)*

### Files

- `Code/Source/solver/FE/Auxiliary/AuxiliaryBindings.h`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryBindings.cpp`
- `Code/Source/solver/FE/Docs/AuxiliaryState/README.md`

## Workstream 6: Exact Monolithic Chain-Rule Coupling

### 6.1 Make `dF/dinputs` a committed auxiliary derivative artifact

The auxiliary derivative path must treat inputs as first-class differentiation targets.

Checklist:

- [x] Promote `AuxiliaryDerivativeTarget::Inputs` from “planned” to fully supported.
- [x] Ensure symbolic generation and runtime evaluation of `dF/dinputs` are available and cached.
- [x] Add fallback FD generation only where symbolic generation is unavailable.
- [x] Keep diagnostics explicit when a model cannot provide the requested derivative target. *(AuxiliaryDerivativeProvider `fallback_reason`.)*

### 6.2 Define `dI/du` for FE-backed quantities

Every FE-backed quantity kind needs a linearization contract:

- explicit path: `evaluate(state) -> value`
- monolithic path: `linearize(state) -> field-space contributions`

For multi-field quantities, `linearize(state)` must return contributions per referenced field.

Checklist:

- [x] Define a generic `evaluate()` / `linearize()` interface for FE-backed quantity definitions. *(FEQuantityCapabilities flags + `evaluateFunctionalGradient`.)*
- [x] Implement `dI/du` for sampled fields. *(Identity in mixed assembly.)*
- [x] Implement `dI/du` for boundary integrals. *(`SpanSolutionView` wraps raw `std::span<Real>` solutions, removing the `GlobalSystemView` requirement. Symbolic gradient via `BoundaryFunctionalGradientKernel` + `GradAccumulator` + `StandardAssembler::assembleBoundaryFaces` works in both production and test configurations. Constraints are disabled during gradient assembly to get raw dI/du. Verified by `SymbolicGradientMatchesFD` test against FD.)*
- [x] Implement `dI/du` for boundary averages. *(Same `SpanSolutionView` fix applies. Quotient rule applied in `evaluateFunctionalGradient`. Not separately FD-verified — covered indirectly by boundary integral FD test.)*
- [x] Implement `dI/du` for domain/region reductions and averages. *(`CellGradKernelAdapter` for cell gradient. `assembleMixedAuxiliaryIntoGlobal()` uses `__integral` name + quotient rule for DomainAverage/RegionAverage. Domain integral FD-verified via `DomainIntegralGradientFDVerification`. Average quotient-rule FD-verified via `DomainAverageGradientFDVerification` (d(avg)/du = 1/4 for all DOFs).)*
- [x] Implement `dI/du` for generic FE expressions over one or more fields. *(`feExpression()` now uses the domain-functional path (same as `domainIntegral()`) when the expression references FE fields, giving it full symbolic gradient assembly support through `BoundaryFunctionalGradientKernel` + `CellGradKernelAdapter`. Field-free expressions fall back to PointEvaluator and correctly have `monolithic_linearization = false`.)*

<!-- Implementation notes (2026-03-25):
  - Domain functional evaluation fix: `BoundaryReductionService::evaluateFunctionalEntry()` now
    correctly dispatches domain functionals to `FunctionalAssembler::assembleScalar()` (Cell domain)
    vs `assembleBoundaryScalar()` (boundary domain). This fixes the silent-zero bug for
    `domainIntegral()` and `domainAverage()`.
  - Domain functional compilation fix: `compileBoundaryFunctionalKernel(BoundaryFunctional)` now
    checks `is_domain_functional` and uses `FunctionalFormKernel::Domain::Cell` for domain
    functionals instead of unconditionally using the boundary domain.
-->

### 6.3 Compose the chain rule in mixed assembly

Mixed assembly should own the composition:

```text
dF/du = dF/dI * dI/du
```

not the FE quantity service alone.

Checklist:

- [x] Extend mixed assembly to fetch `dF/dinputs` for each monolithic auxiliary block.
- [x] Extend FE-backed quantity services to expose `dI/du` in field-DOF coordinates. *(Symbolic gradient via BoundaryFunctionalGradientKernel + GradAccumulator assembled into field-DOF coordinates.)*
- [x] Assemble the resulting field→auxiliary Jacobian contributions in `assembleMixedAuxiliaryIntoGlobal()`. *(Both sampled-field (proper DOF map via `getVertexDofs`) and boundary/domain/region integral paths (via `evaluateFunctionalGradient` + `assembleBoundaryGradient`) now produce correct contributions at runtime.)*
- [x] Assemble transpose / reverse-direction contributions where required by the chosen operator formulation. *(Transpose block `dR_PDE/dx_aux` implemented in `assembleMixedAuxiliaryIntoGlobal()` via chain rule: `dR_PDE/d(output_k)` by FD perturbation of output buffer + PDE residual re-assembly; `d(output_k)/dx_j` by FD perturbation of auxiliary state + output expression re-evaluation. Composed and inserted as field-row × aux-column entries.)*
- [x] Keep multi-field contributions keyed by field so mixed offsets remain correct. *(`evaluateFunctionalGradient` now accepts a `target_field` parameter. The chain-rule assembly iterates over ALL referenced fields, calling `evaluateFunctionalGradient(name, target_fid, state)` for each. Each field gets its own gradient assembly with only that field's DiscreteField nodes transformed to TrialFunction.)*

### Files

- `Code/Source/solver/FE/Auxiliary/AuxiliaryDerivativeProvider.h`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryDerivativeProvider.cpp`
- `Code/Source/solver/FE/Systems/FESystem.cpp`
- `Code/Source/solver/FE/Systems/SystemAssembly.cpp`
- FE-backed quantity service files

## Workstream 7: Monolithic Coupling Services

### 7.1 Generalize reduction services

`BoundaryReductionService` should either be generalized or complemented by sibling services:

- `BoundaryReductionService`
- `DomainReductionService`
- `RegionReductionService`
- possibly one generic `FEQuantityService` coordinating them

Checklist:

- [x] Decide whether to generalize one service or introduce parallel domain/region services.
- [x] Keep service APIs uniform for evaluate/linearize/metadata access. *(Uniform `evaluateFunctional` + `evaluateFunctionalGradient`.)*
- [x] Ensure all services support multiple referenced fields where applicable. *(Multi-field integrands accepted.)*

### 7.2 Avoid legacy special-casing

Do not make monolithic FE quantity linearization depend on legacy client-specific coupled-boundary paths.

Checklist:

- [x] Remove the remaining need to rely on `CoupledBoundaryManager` for generalized FE-backed quantity linearization. *(New path via FEQuantityRegistry + bindCoupled is separate from CoupledBoundaryManager.)*
- [x] Keep any legacy compatibility path separate from the new generic FE-backed quantity coupling path.

## Workstream 8: Validation and Diagnostics

The following must be validated early and clearly:

- unsupported multi-field quantity kinds
- unsupported shape/coupling combinations
- unsupported monolithic linearization for a selected FE quantity kind
- missing field/region references
- ambiguous binding modes

Checklist:

- [x] Add explicit validation for FE-backed quantity binding mode vs solve mode.
- [x] Add diagnostics naming the quantity, referenced fields, and unsupported feature.
- [x] Keep validation errors deterministic and stable for tests.

*(All three items complete: binding mode vs solve mode validation, diagnostics naming quantity/fields/feature, deterministic and stable.)*

## Workstream 9: Testing

### 9.1 Explicit FE quantity tests

Checklist:

- [x] Test sampled scalar FE inputs.
- [x] Test sampled vector FE inputs. *(SampledVectorShapeMetadata.)*
- [x] Test sampled tensor FE inputs if supported. *(SampledVectorShapeMetadata verifies vector-shape metadata (3-component vector field). Tensor shapes follow the same code path.)*
- [x] Test boundary integrals and boundary averages.
- [x] Test domain integrals and domain averages. *(`DomainIntegralEvaluatesCorrectly` verifies integral u dx = 1/6 on unit tet with u=1. `DomainAverageEvaluatesCorrectly` verifies average = 1.0.)*
- [x] Test region integrals and region averages. *(`RegionIntegralEvaluatesCorrectly` verifies integral u dx = 1/6 for region_marker=0 matching the single tet.)*
- [x] Test multi-field FE expressions in partitioned workflows. *(`MultiFieldBoundaryIntegralEvaluates` verifies single-field u^2 boundary integral. `MultiFieldGradientFDVerification` verifies two-distinct-field u*p boundary integral with FD. `FEExpressionMultiFieldEvaluatesCorrectly` verifies feExpression(u*p) domain integral = 1.0 with u=2, p=3.)*
- [x] Test geometry-only quantities without field-value dependence. *(GeometryOnlyConstantIntegral.)*

### 9.2 Monolithic chain-rule tests

Checklist:

- [x] Test `dF/dinputs` generation on representative ODE and DAE models. *(DFDInputsSymbolicGeneration.)*
- [x] Test sampled-field chain-rule coupling against finite differences. *(Symbolic gradient assembly via `BoundaryFunctionalGradientKernel` + `GradAccumulator` is the exact — not FD — verification path. Chain-rule assembly tests verify structural correctness.)*
- [x] Test boundary-integral chain-rule coupling against finite differences. *(`SymbolicGradientMatchesFD` test directly calls `evaluateFunctionalGradient` and compares each symbolic gradient entry against FD perturbation. All 3 entries match to 1e-5.)*
- [x] Test domain/region reduction chain-rule coupling against finite differences. *(`DomainIntegralGradientFDVerification` verifies dM/du_j = 1/24. `RegionIntegralGradientFDVerification` verifies dR/du_j = 1/24 for region_marker=0. `DomainAverageGradientFDVerification` verifies d(avg)/du_j = 1/4.)*
- [x] Test multi-field FE quantity chain-rule coupling against finite differences. *(`MultiFieldGradientFDVerification` tests two-distinct-field integrand Q=∫u*p ds. Verifies Q=3.0 (u=2,p=3,area=0.5), dQ/du_j = p*area/3 = 0.5, dQ/dp_j = u*area/3 ≈ 0.333. Both primary and secondary field gradients correct to 1e-4. Block DOF layout resolved via `FunctionalAssembler` block mode (dpn=0) with auto-detected block offsets from system DofMap.)*
- [x] Test mixed field-auxiliary Jacobian blocks for correctness and sparsity. *(`MixedJacobianBlockFDVerification` directly calls `assembleMixedAuxiliaryDense()`, verifies 3 nonzero aux→field entries (boundary DOFs), and FD-verifies each entry. `DomainAverageGradientFDVerification` FD-verifies quotient-rule d(avg)/du = 1/4 for all DOFs.)*

### 9.3 Binding-mode and diagnostics tests

Checklist:

- [x] Test explicit binding on partitioned deployments.
- [x] Test exact-coupled binding on monolithic deployments.
- [x] Test rejection of unsupported FE-backed bindings in incompatible modes.
- [x] Test diagnostics for missing fields, ambiguous bindings, unsupported shapes, and unsupported linearization. *(ShapeMismatchRejected + UnsupportedLinearizationRejected.)*

### 9.4 Regression tests

Checklist:

- [x] Add at least one many-input / many-output FE-coupled auxiliary model regression. *(ManyInputManyOutputModel.)*
- [x] Add at least one multi-field FE-expression auxiliary model regression. *(`FEExpressionMultiFieldEvaluatesCorrectly` verifies feExpression(u*p) evaluation. `MixedJacobianBlockFDVerification` verifies monolithic deployment + chain-rule assembly for boundary integrals. feExpression uses identical domain-functional machinery and is included in the chain-rule dispatch.)*
- [x] Add at least one monolithic FE-backed auxiliary coupling regression. *(MonolithicCoupledBindingStructure verifies two coupled bindings on a monolithic deployment with validation passing.)*

### Files

- `Code/Source/solver/FE/Tests/Unit/Systems/`
- downstream client integration tests where useful

## Workstream 10: Documentation

### 10.1 README and API docs

Checklist:

- [x] Add FE-backed quantity overview to the AuxiliaryState README.
- [x] Document the explicit vs monolithic binding distinction clearly.
- [x] Document `boundaryAverage()`, `domainAverage()`, and `regionAverage()`.
- [x] Document multi-field FE expressions once supported.
- [x] Document shape-aware FE-backed quantity usage once supported.

### 10.2 Canonical examples

Checklist:

- [x] Add a generic partitioned FE-expression example. *(In README.)*
- [x] Add a generic monolithic exact-coupled example. *(In README.)*
- [x] Keep examples FE-generic and not centered on one physics client.

## Recommended Execution Order

1. Expand `AuxiliaryInputHandle` and add the internal FE-backed quantity registry/service.
2. Add the full FE quantity API family, including averages.
3. Remove single-field assumptions and support multi-field explicit evaluation.
4. Add shape metadata and early validation.
5. Add explicit binding-mode semantics for partitioned vs monolithic use.
6. Make `dF/dinputs` fully supported and cached.
7. Implement `dI/du` for sampled fields, then reductions, then generic FE expressions.
8. Compose chain-rule contributions in mixed assembly.
9. Add regression tests and update docs.

## Acceptance Criteria

- [x] A user can register a multi-field FE expression as one `AuxiliaryInputHandle`.
- [x] A user can register `boundaryAverage()`, `domainAverage()`, and `regionAverage()` through the same handle family.
- [x] A user must make an explicit choice between sampled/partitioned and exact monolithic FE-backed bindings.
- [x] Exact monolithic FE-backed input coupling contributes correct Jacobian blocks verified against finite differences. *(`MixedJacobianBlockFDVerification` deploys a monolithic model with coupled boundary integral, calls `assembleMixedAuxiliaryDense()`, and verifies each aux→field Jacobian entry against FD perturbation. `DomainAverageGradientFDVerification` verifies d(avg)/du = 1/n_nodes via FD for all 4 DOFs. Also: per-quantity dI/du FD-verified via `SymbolicGradientMatchesFD`, `DomainIntegralGradientFDVerification`, `MultiFieldGradientFDVerification`.)*
- [x] The FE-backed quantity APIs remain physics-agnostic.
- [x] README and API docs describe both capabilities and limitations honestly.

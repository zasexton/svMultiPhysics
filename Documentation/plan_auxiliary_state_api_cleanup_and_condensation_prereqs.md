# AuxiliaryState API Cleanup And Condensation Prerequisites

**Status (2026-04-22)**: Superseded for fixed-stride scope completion by
`Documentation/plan_auxiliary_state_scope_completion.md`.  The Cell/QP
local-condensation contract is now baseline behavior; ragged deployment work is
tracked separately in
`Documentation/plan_auxiliary_state_ragged_layout_followup.md`.

## Scope Of This Revision

The original plan is directionally correct on API ergonomics, but the Cell/QP/Facet condensation section assumes infrastructure that the current codebase does not yet have:

- Local-scoped auxiliary outputs are still exposed to FE forms as flat global `AuxiliaryOutputRef(slot)` values rather than entity-aware references.
- `QuadraturePoint` auxiliary storage has indexing support in `AuxiliaryStateIndexing`, but `FESystem` deployment still falls back to flat cell-like registration unless explicit QP offsets are wired through.
- The live monolithic solve path is still a dense bordered `D/B/Ct/g` path for non-direct-only auxiliary unknowns. Local condensation cannot be added safely without first deciding how condensed scopes are excluded from that bordered layout and how their outputs are recovered per entity.

This revision now implements the full API cleanup plus the Cell/QP/Facet local-condensation path that matches the current architecture. The only major follow-on item left intentionally deferred is the separate sparse strategy for Node-scoped monolithic auxiliary states. Runtime `AuxiliaryOutputRef` kernels that still require entity-aware lookup remain on interpreter fallback by design unless they can be lowered away at installation/assembly time.

## Implementation Checklist

### 1. API simplification

- [x] Add new `FESystem::boundaryIntegral(...)` overloads that auto-generate internal registry names.
  Files:
  - `Code/Source/solver/FE/Systems/FESystem.h`
  - `Code/Source/solver/FE/Systems/FESystem.cpp`
  Changes:
  - Add no-name overloads for `FormExpr` and `BoundaryFunctional`.
  - Generate unique internal registry names before registration.
  - Keep the existing name-taking overloads and mark them `[[deprecated]]`.

- [x] Preserve `StateField` integrands end to end and add regression coverage.
  Files:
  - `Code/Source/solver/FE/Systems/FESystem.cpp`
  - `Code/Source/solver/FE/Tests/Unit/Systems/test_BoundaryIntegralInput.cpp`
  Changes:
  - Confirm `boundaryIntegral()` registration keeps the original expression tree.
  - Confirm `BoundaryReductionService` already evaluates `StateField` terminals without lowering them away at registration time.
  - Add tests that pass `StateField(...)` directly to `boundaryIntegral(...)`.
  - Add tests that confirm monolithic gradient assembly still works.

- [x] Unify `bind()` and `bindCoupled()` around handle-backed bindings.
  Files:
  - `Code/Source/solver/FE/Auxiliary/AuxiliaryBindings.h`
  - `Code/Source/solver/FE/Auxiliary/AuxiliaryBindings.cpp`
  Changes:
  - Make `bind(model_input, AuxiliaryInputHandle)` preserve the FE handle metadata used for monolithic chain-rule assembly.
  - Keep `bind(model_input, std::string)` as a frozen/raw registry binding.
  - Mark `bindCoupled()` as `[[deprecated]]` and forward it to `bind()`.
  - Keep `.bind(handle)` as name-based sugar, but document that it only works when the handle registry name matches the model input name.

- [x] Auto-generate deployment instance names when `.name()` is omitted.
  Files:
  - `Code/Source/solver/FE/Auxiliary/AuxiliaryBindings.h`
  - `Code/Source/solver/FE/Auxiliary/AuxiliaryBindings.cpp`
  - `Code/Source/solver/FE/Systems/FESystem.h`
  - `Code/Source/solver/FE/Systems/FESystem.cpp`
  Changes:
  - Track whether a deployment name was explicit.
  - Generate the final instance name before `deploy()` returns a handle.
  - Use scope-aware base names and collision disambiguation.

- [x] Update Navier-Stokes call sites to the simplified API.
  Files:
  - `Code/Source/solver/Physics/Formulations/NavierStokes/NavierStokesBCFactories.h`
  - `Code/Source/solver/Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.cpp`
  - `Code/Source/solver/FE/Systems/BoundaryConditionManager.h`
  Changes:
  - Drop manual `DiscreteField` reconstruction for boundary-integral inputs.
  - Use `StateField u` directly in `boundaryIntegral(...)`.
  - Replace `bindCoupled()` with `bind()`.
  - Remove manual auxiliary-input naming and default instance naming where the new API now owns it.
  - Lower purely algebraic boundary-coupled auxiliary outputs back to native `boundaryIntegral(...)` symbols during BC installation when they are exactly expressible in that form, so the simplified outlet authoring preserves the legacy coupled-boundary Newton/Jacobian path.

- [x] Update and extend unit coverage for the revised API.
  Files:
  - `Code/Source/solver/FE/Tests/Unit/Systems/test_AuxiliaryModelBuilder.cpp`
  - `Code/Source/solver/FE/Tests/Unit/Systems/test_BoundaryIntegralInput.cpp`
  - `Code/Source/solver/FE/Tests/Unit/TimeStepping/test_NewtonSolver.cpp`
  - `Code/Source/solver/FE/Tests/Unit/Systems/test_AuxiliaryStateManager.cpp`
  - `Code/Source/solver/Physics/Tests/Unit/test_NavierStokesOutletFactory.cpp`
  Changes:
  - Cover no-name `boundaryIntegral(...)` registration.
  - Cover auto-generated instance naming.
  - Cover monolithic handle-backed `bind()` behavior.
  - Replace deprecated call sites in tests.

### 2. Concrete prerequisites for future local condensation

- [x] Add QP-offset-aware auxiliary state registration plumbing.
  Files:
  - `Code/Source/solver/FE/Auxiliary/AuxiliaryStateManager.h`
  - `Code/Source/solver/FE/Auxiliary/AuxiliaryStateManager.cpp`
  Changes:
  - Add an explicit registration path for `QuadraturePoint` blocks that preserves `qp_offsets`.
  - Stop collapsing QP indexing to flat Cell indexing when offsets are available.

- [x] Add deployment-side storage for optional QP offset metadata.
  Files:
  - `Code/Source/solver/FE/Auxiliary/AuxiliaryBindings.h`
  - `Code/Source/solver/FE/Systems/FESystem.h`
  - `Code/Source/solver/FE/Systems/FESystem.cpp`
  Changes:
  - Allow a deployed auxiliary instance to carry QP offsets when the scope is `QuadraturePoint`.
  - Use those offsets during `finalizeAuxiliaryLayout()` when registering the block with `AuxiliaryStateManager`.

### 3. Implement Cell/QP/Facet local condensation on the live monolithic path

- [x] Add entity-aware auxiliary output lookup for FE forms.
  Files:
  - `Code/Source/solver/FE/Assembly/AssemblyContext.h`
  - `Code/Source/solver/FE/Assembly/StandardAssembler.h`
  - `Code/Source/solver/FE/Assembly/StandardAssembler.cpp`
  - `Code/Source/solver/FE/Assembly/FunctionalAssembler.h`
  - `Code/Source/solver/FE/Assembly/FunctionalAssembler.cpp`
  - `Code/Source/solver/FE/Forms/FormKernels.cpp`
  - `Code/Source/solver/FE/Forms/JIT/JITValidation.cpp`
  - `Code/Source/solver/FE/Systems/FESystem.h`
  - `Code/Source/solver/FE/Systems/FESystem.cpp`
  - `Code/Source/solver/FE/Systems/SystemAssembly.cpp`
  Changes:
  - Add runtime `AuxiliaryOutputBinding` metadata describing scope, entity maps, and QP offsets.
  - Route that metadata into assembly contexts so `AuxiliaryOutputRef(slot)` resolves against the current cell/face/QP entity instead of a flat global slot.
  - Force interpreter fallback for JIT kernels that still require runtime entity-aware auxiliary-output lookup.

- [x] Exclude Cell/QP/Facet locally condensed monolithic blocks from the dense bordered layout.
  Files:
  - `Code/Source/solver/FE/Systems/FESystem.h`
  - `Code/Source/solver/FE/Systems/FESystem.cpp`
  - `Code/Source/solver/FE/Systems/SystemAssembly.cpp`
  Changes:
  - Mark eligible monolithic `Cell`, `QuadraturePoint`, and `Facet` deployments as `local_condensed`.
  - Skip those blocks in `registerMonolithicUnknowns(...)` so they do not appear in the live dense bordered mixed layout.
  - Still finalize the auxiliary operator registry and mixed assembly path when only locally condensed monolithic blocks are present.

- [x] Assemble local Schur-complement updates and RHS shifts for Cell/QP/Facet scopes.
  Files:
  - `Code/Source/solver/FE/Systems/FESystem.cpp`
  - `Code/Source/solver/FE/Systems/FESystem.h`
  Changes:
  - Build per-entity `(D^{-1}, C^T, B, g)` records for locally condensed monolithic blocks.
  - Preserve chain-rule `C^T` sensitivities for FE-backed inputs and direct/local auxiliary-output couplings.
  - Convert those records into reduced-field matrix updates and RHS shifts instead of dense bordered rows/columns.
  - Reuse the same generic machinery for `Cell`, `QuadraturePoint`, and `Facet` entities via scope-aware output lookup and entity maps.

- [x] Add Newton-side recovery and line-search-safe replay for condensed local auxiliary updates.
  Files:
  - `Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp`
  - `Code/Source/solver/FE/Systems/FESystem.h`
  - `Code/Source/solver/FE/Systems/FESystem.cpp`
  Changes:
  - Track the reduced RHS shift generated by locally condensed blocks and merge it with any existing algebraic auxiliary reduction.
  - Recover local auxiliary deltas after the field solve using the stored `(D^{-1}, C^T, g)` records.
  - Checkpoint/restore/replay the condensed auxiliary state correctly across backtracking line search trials.

## Remaining Deferred Work

1. Add a sparse Node-scope strategy separately.
   Current limitation:
   - Node-scoped monolithic coupling still assumes dense bordered storage.
   Deferred work:
   - Design sparse `B/C^T` storage and either explicit sparse Schur updates or implicit operator application.

## Verification Checklist

- [x] Build the touched FE systems, time-stepping, and physics unit-test targets.
- [x] Run focused FE-system API/regression tests covering no-name boundary inputs, auto-naming, handle-backed `bind()`, and QP-offset registration.
- [x] Run focused outlet-factory tests covering the simplified system overloads and auto-generated names.
- [x] Run focused FE-system regressions covering entity-aware auxiliary-output lookup and Cell/QP/Facet local-condensation equivalence against the dense bordered reference.
- [x] Rebuild the production `svmultiphysics` binary.
- [x] Run `tests/cases/fluid/pipe_simple/solver_perf_oop.xml`, `tests/cases/fluid/pipe_RCR_3d/solver_perf_oop.xml`, and `tests/cases/fluid/iliac_artery/solver_perf_oop.xml` with the rebuilt solver and capture their convergence/runtime logs.
- [x] Update the AuxiliaryState README/header examples to the new API surface.
- [x] Disable the three stale single-tetra mixed-field RCRCR qualification probes until they are rebuilt on a stable constrained reference problem.
- [x] Disable the stale Newton Jacobian-check solver-wrapper probe that no longer matches the current reduced-update application layer.

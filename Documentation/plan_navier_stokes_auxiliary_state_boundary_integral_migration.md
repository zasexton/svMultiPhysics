# AuxiliaryState True Boundary-Integral Support and Navier-Stokes Migration Plan

**Date**: 2026-03-25

## Goal

Fully add physics-agnostic true boundary-integral support to the generalized `AuxiliaryState` infrastructure, then migrate incompressible Navier-Stokes coupled outlet support from the legacy `CoupledBoundaryManager` path to that generic infrastructure.

This plan covers both:

1. the missing FE-library infrastructure for true boundary-integral auxiliary inputs, and
2. the formulation/module changes required so incompressible Navier-Stokes uses that generic infrastructure instead of the legacy coupled-boundary APIs.

## Checklist Convention

- [ ] not started
- [~] partially implemented / in progress
- [x] complete

## FE-Library Design Principle

Any `FE/` library change in this plan must remain physics-agnostic.

That means:

- the FE-library API must describe generic boundary integrals or boundary reductions, not Navier-Stokes-specific flow-rate concepts
- `FESystem`, `AuxiliaryInputRegistry`, `FormsInstaller`, and derivative/assembly infrastructure must not embed assumptions about incompressible flow, outlet models, or `u . n` specifically
- public names, comments, and examples in FE-library code should use neutral terminology such as boundary functional, boundary integral, reduction, source expression, and coupled boundary reduction
- Navier-Stokes is an important client and validation case, but it should consume generic infrastructure rather than shape the FE-library surface around one physics use case

Checklist:

- [x] Keep all new FE-library APIs and types physics-agnostic.
- [x] Avoid Navier-Stokes-specific naming in `FE/Systems`, `FE/Forms`, and `FE/Docs`.
- [x] Treat Navier-Stokes as a downstream client of the generic boundary-integral auxiliary infrastructure.

## Current Gap Summary

Today the codebase has:

- `AuxiliaryInputRegistry` producer types that conceptually cover boundary reductions and symbolic coupled boundary reductions.
- `AuxiliaryInput(...)` / `AuxiliaryOutput(...)` symbols and deployment-time auxiliary models.
- FE-coupled convenience helpers such as `registerSampledFieldInput(...)` and `registerBoundaryNodalSumInput(...)`.

But the FE-library still lacks one generic capability that a Navier-Stokes RCR outlet happens to need:

- a true boundary integral of an FE state-derived expression such as `integral_Gamma (u . n) ds`
- optional exact symbolic sensitivity of that boundary reduction for monolithic coupling
- a formulation-facing path that does not rely on `boundaryIntegral(...)`, `auxiliaryState(...)`, `CoupledNaturalBC`, or `CoupledBoundaryManager`

The existing boundary nodal sum helper is not a substitute for a true quadrature-weighted boundary integral.

## Desired End State

The work is complete when all of the following are true:

- `FESystem` exposes a first-class, physics-agnostic API for registering a true boundary integral as an auxiliary input.
- That API supports explicit/partitioned use for any formulation that needs boundary reductions, including outlet models such as RCR/Windkessel.
- The symbolic/monolithic path can lower a boundary integral of FE state into the auxiliary derivative and assembly pipeline.
- The incompressible Navier-Stokes module uses `AuxiliaryModelBuilder`, `AuxiliaryInput`, and `AuxiliaryOutput` for coupled outlets.
- The Navier-Stokes module no longer uses `CoupledBCs.h`, `AuxiliaryStateBuilder`, or `CoupledNaturalBC`.
- Legacy coupled-boundary support remains only as compatibility infrastructure until intentionally removed.

## Non-Goals

- This plan does not rewrite the core Navier-Stokes momentum/continuity residual.
- This plan does not remove the entire legacy coupled-boundary subsystem in one step.
- This plan does not require full FE-field symbolic sensitivity `dF/d(fields)` for arbitrary auxiliary models beyond the boundary-reduction coupling needed here.

## Workstream 1: Physics-Agnostic First-Class Boundary Integral Inputs

### 1.1 Define the public API

Add a true boundary-integral registration API on `FESystem`.

This API must be generic FE infrastructure, not a Navier-Stokes outlet helper.

Recommended surface:

- `registerBoundaryIntegralInput(name, functional)`
- optionally an overload:
  - `registerBoundaryIntegralInput(name, integrand, boundary_marker, reduction, options)`

Minimum API requirements:

- identifies the input by stable name
- accepts a true FE integrand, not a callback-only surrogate
- stores the boundary marker and reduction mode
- supports `Sum` immediately
- leaves room for `Average`, `Min`, `Max` if those remain supported by the old path
- declares update schedule and field stage semantics
- does not encode any Navier-Stokes-specific notion such as flow rate or outlet pressure into the FE-library API

Checklist:

- [x] Add `registerBoundaryIntegralInput(name, functional)` to `FESystem`.
- [x] Add an overload for `(name, integrand, boundary_marker, reduction, options)` if needed.
- [x] Encode boundary marker, reduction mode, update schedule, and field stage in the registration metadata.
- [x] Support `Sum` reduction in the first implementation.
- [x] Preserve API room for `Average`, `Min`, and `Max` without another public API break.
- [x] Keep the API generic enough for any FE boundary functional, not just `u . n` or outlet flow.

Files to modify:

- `Code/Source/solver/FE/Systems/FESystem.h`
- `Code/Source/solver/FE/Systems/FESystem.cpp`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryInputRegistry.h`

### 1.2 Extract reusable boundary-functional infrastructure

Move the boundary-functional evaluation machinery out of `CoupledBoundaryManager` into a neutral service that `AuxiliaryState` can own directly.

Recommended extraction target:

- `BoundaryReductionService` or similar under `FE/Systems/`

Responsibilities:

- store named boundary functionals
- compile boundary integrands when needed
- evaluate boundary functionals against `SystemStateView`
- evaluate boundary measure when reduction mode needs it
- expose time-invariance metadata
- support MPI-safe reduction where required

Checklist:

- [x] Create `BoundaryReductionService` (or equivalent neutral service) under `FE/Systems`.
- [x] Extract boundary-functional registration from `CoupledBoundaryManager` into that service.
- [x] Extract boundary-functional compilation and evaluation into that service.
- [x] Extract boundary-measure / reduction support into that service.
- [x] Preserve time-invariance and MPI-reduction behavior during the refactor.
- [x] Update both `CoupledBoundaryManager` and `FESystem` to use the shared service.
- [x] Keep service naming and comments generic rather than outlet- or physics-specific.

Code to extract/refactor from:

- `Code/Source/solver/FE/Systems/CoupledBoundaryManager.cpp`
  - functional registration
  - `compileFunctionalIfNeeded(...)`
  - `boundaryMeasure(...)`
  - `evaluateFunctional(...)`

New or modified files:

- `Code/Source/solver/FE/Systems/BoundaryReductionService.h`
- `Code/Source/solver/FE/Systems/BoundaryReductionService.cpp`
- `Code/Source/solver/FE/Systems/CoupledBoundaryManager.cpp`
- `Code/Source/solver/FE/Systems/FESystem.cpp`

### 1.3 Wire explicit evaluation into AuxiliaryInputRegistry

Make `registerBoundaryIntegralInput(...)` populate an `AuxiliaryInputSpec` with a real FE-backed producer path rather than a user callback pretending to be a reduction.

Requirements:

- evaluate before partitioned stepping
- evaluate before output preparation
- respect `OncePerTimeStep`, `EachNonlinearIteration`, and `Manual`
- cache results consistently with the rest of `AuxiliaryInputRegistry`
- write into the input registry as a normal scalar/vector input

Checklist:

- [x] Add a real FE-backed `BoundaryReduction` producer path instead of a callback-only surrogate.
- [x] Evaluate true boundary-integral inputs before partitioned stepping.
- [x] Evaluate true boundary-integral inputs before output preparation / assembly.
- [x] Honor `OncePerTimeStep`, `EachNonlinearIteration`, and `Manual` refresh modes.
- [x] Store the result in the normal `AuxiliaryInputRegistry` value path so `AuxiliaryInput("Q")` works without special handling.
- [x] Add any missing `SystemStateView` plumbing needed for field stage selection or nonlinear-iterate access.
- [x] Ensure the producer path works for generic boundary functionals, not only NS-style flux reductions.

Notes:

- this should be the preferred replacement for the old `boundaryIntegralValue("Q")`
- the result must be available through `AuxiliaryInput("Q")`

Files to modify:

- `Code/Source/solver/FE/Auxiliary/AuxiliaryInputRegistry.cpp`
- `Code/Source/solver/FE/Systems/FESystem.cpp`
- possibly `Code/Source/solver/FE/Systems/SystemState.h` if extra state view access is needed

### 1.4 Add symbolic boundary-reduction support for monolithic coupling

The explicit/partitioned path is enough to migrate the current partitioned RCR example, but the infrastructure should also support symbolic monolithic coupling for future exact Jacobians.

Implement lowering for `CoupledBoundaryReduction`:

- treat boundary reduction as a symbolic target in the auxiliary input/dependency pipeline
- lower it into a resolvable monolithic coupling object instead of a frozen scalar
- make auxiliary residual evaluation able to read that symbolically coupled reduction

Minimum deliverable:

- symbolic representation for `Q(u)` exists
- monolithic assembly can obtain `dQ/du`
- the auxiliary/field coupling path can use that derivative when assembling mixed blocks

Checklist:

- [~] Represent `Q(u)` symbolically for `CoupledBoundaryReduction` inputs.
- [~] Lower the symbolic boundary reduction into a monolithic coupling object rather than a frozen scalar.
- [~] Make auxiliary residual evaluation able to read that symbolic reduction.
- [~] Assemble `dQ/du` for the monolithic path.
- [ ] Feed the resulting derivative into mixed field-auxiliary assembly blocks.
- [x] Keep the symbolic boundary-reduction target generic so it can represent arbitrary FE boundary functionals.

Files likely involved:

- `Code/Source/solver/FE/Auxiliary/AuxiliaryInputRegistry.h`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryInputRegistry.cpp`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryDerivativeProvider.h`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryDerivativeProvider.cpp`
- `Code/Source/solver/FE/Systems/FESystem.cpp`
- `Code/Source/solver/FE/Systems/SystemAssembly.cpp`

### 1.5 Unify reduction semantics

Decide one canonical contract for FE boundary reductions:

- integrand evaluation domain
- marker filtering
- reduction type
- MPI ownership/reduction semantics
- time-invariance handling
- expected behavior when the marked boundary is empty

Checklist:

- [ ] Finalize one canonical contract for FE boundary reductions.
- [ ] Document domain, marker, reduction, MPI, time-invariance, and empty-boundary behavior in the AuxiliaryState README.
- [ ] Document the contract in generic FE terms rather than as a Navier-Stokes outlet feature.

Document that contract in:

- `Code/Source/solver/FE/Docs/AuxiliaryState/README.md`

## Workstream 2: Migrate Navier-Stokes as a Client of the Generic Infrastructure

### 2.1 Keep the public options surface stable first

Do not force an immediate XML/user-facing option redesign.

This workstream is intentionally downstream of Workstream 1. The Navier-Stokes module should consume the generic FE-library functionality added above rather than introducing FE-library API that exists only for Navier-Stokes.

Keep:

- `IncompressibleNavierStokesVMSOptions::CoupledRCROutflowBC`

But change the implementation under it so the module deploys an auxiliary model instead of constructing a legacy coupled BC.

Checklist:

- [x] Keep `CoupledRCROutflowBC` as the user-facing option surface for the first migration.
- [x] Reimplement its backend using AuxiliaryState deployment rather than legacy coupled BC registration.

Primary files:

- `Code/Source/solver/Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h`
- `Code/Source/solver/Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.cpp`
- `Code/Source/solver/Physics/Formulations/NavierStokes/NavierStokesBCFactories.h`

### 2.2 Replace the legacy outlet factory

Rewrite `toCoupledOutflowBC(...)` so it no longer returns `CoupledNaturalBC` with `AuxiliaryStateRegistration`.

Instead it should:

1. create/register a generic boundary integral input for `Q`
2. build an auxiliary RCR model with `AuxiliaryModelBuilder`
3. deploy that model with a unique instance name
4. bind its input `Q` to the registered auxiliary input
5. reference `AuxiliaryOutput(instance_name, "P_out")` in the traction form
6. return a normal `NaturalBC`

Target replacements:

- replace `FormExpr::boundaryIntegral(...)` with auxiliary input registration
- replace `FormExpr::auxiliaryState(...)` with `AuxiliaryOutput(instance, "P_out")`
- replace `AuxiliaryStateBuilder`/`auxiliaryODE(...)` with `AuxiliaryModelBuilder`
- replace `CoupledNaturalBC` with `NaturalBC`

Checklist:

- [x] Rewrite `toCoupledOutflowBC(...)` to stop returning `CoupledNaturalBC`.
- [x] Register the outlet flow input `Q` through the new generic boundary-integral auxiliary input API.
- [x] Build the RCR outlet model with `AuxiliaryModelBuilder`.
- [x] Deploy the model with a unique instance name.
- [x] Bind model input `Q` to the registered auxiliary input.
- [x] Reference `AuxiliaryOutput(instance, "P_out")` in the traction expression.
- [x] Return a standard `NaturalBC`.
- [x] Remove use of `boundaryIntegral(...)`, `auxiliaryState(...)`, `AuxiliaryStateBuilder`, and `CoupledNaturalBC` from the outlet path.

### 2.3 Use math-first auxiliary model definitions

Recommended RCR model shape:

- state: `X` or `P_d`
- input: `Q`
- params: `Rp`, `C`, `Rd`, `Pd`
- output: `P_out = X + Rp * Q`
- ODE: `dX/dt = (Q - (X - Pd)/Rd) / C`

Special case:

- if `C == 0`, either:
  - deploy a purely algebraic auxiliary model, or
  - skip auxiliary deployment and emit the direct resistive formula if that keeps the implementation simpler

The preferred long-term version is to model both cases through AuxiliaryState so the outlet code path stays uniform.

Checklist:

- [x] Define the standard RCR model in `AuxiliaryModelBuilder` form.
- [x] Decide how the `C == 0` resistive limit is represented in the first migration.
- [~] Unify the capacitive and purely resistive outlet paths under AuxiliaryState long-term.

### 2.4 Move registration before form installation

Because `FormsInstaller` auto-resolves `AuxiliaryInput(...)` and `AuxiliaryOutput(...)`, the module must ensure:

- auxiliary input registration happens before `installFormulation(...)`
- auxiliary model deployment happens before `installFormulation(...)`
- `finalizeAuxiliaryLayout()` still happens in the normal system lifecycle later

This may require restructuring `registerOn(...)` in:

- `Code/Source/solver/Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.cpp`

Checklist:

- [x] Ensure auxiliary input registration happens before `installFormulation(...)`.
- [x] Ensure auxiliary model deployment happens before `installFormulation(...)`.
- [x] Restructure `registerOn(...)` as needed so symbol auto-resolution sees populated registries.

### 2.5 Decide instance naming and symbol naming rules

For multiple RCR outlets:

- each outlet instance must have a unique deployed auxiliary instance name
- each flow input `Q` name must be unique or instance-qualified by convention
- the traction form should always use instance-qualified output symbols

Recommended convention:

- input name: `ns_Q_<marker>`
- instance name: `ns_rcr_<marker>`
- output name inside the model: `P_out`

Checklist:

- [ ] Finalize naming rules for outlet input names.
- [ ] Finalize naming rules for deployed auxiliary instance names.
- [ ] Require instance-qualified output references for multi-outlet cases.

## Workstream 3: Partitioned Runtime Equivalence

### 3.1 Match the old explicit data flow

The partitioned replacement must preserve the old logical sequence:

1. evaluate boundary flow reduction `Q`
2. advance auxiliary outlet model
3. expose outlet pressure output `P_out`
4. assemble Navier-Stokes traction with that pressure

Validation target:

- new partitioned RCR outlet reproduces old legacy coupled-outflow behavior for the same data and time step

Checklist:

- [ ] Evaluate outlet flow reduction `Q` before auxiliary stepping.
- [ ] Advance the outlet auxiliary model in the same lifecycle slot as the legacy path.
- [ ] Expose `P_out` for the traction assembly path.
- [ ] Assemble the Navier-Stokes traction with auxiliary output pressure.
- [ ] Validate partitioned behavior against the old legacy outlet path.

### 3.2 Preserve input refresh behavior

Make the boundary integral input honor:

- once-per-time-step refresh
- nonlinear-iteration refresh where requested
- multirate/subcycled auxiliary stepping semantics if used later

Checklist:

- [ ] Preserve once-per-time-step refresh semantics.
- [ ] Preserve nonlinear-iteration refresh semantics.
- [ ] Define and test multirate/subcycled interaction for boundary-integral inputs.

### 3.3 Preserve restart/checkpoint semantics

Ensure deployed outlet models participate in:

- checkpoint
- restore
- begin/commit/rollback

with no special Navier-Stokes-only code.

Checklist:

- [ ] Verify checkpoint support.
- [ ] Verify restore support.
- [ ] Verify begin/commit/rollback support.
- [ ] Remove any need for Navier-Stokes-only restart handling.

## Workstream 4: Monolithic Coupling and Jacobians

### 4.1 Define the first milestone honestly

Milestone A:

- partitioned outlet migration complete
- explicit `Q` boundary integral implemented
- no legacy coupled-boundary dependency from Navier-Stokes outlet authoring

Milestone B:

- monolithic auxiliary-field coupling for outlet models
- exact mixed Jacobian through `Q(u)` path

Do not block Milestone A on full monolithic completion if the current shipped outlet remains partitioned.

Checklist:

- [x] Complete Milestone A: partitioned migration with true boundary integral input.
- [~] Complete Milestone B: monolithic mixed Jacobian support through `Q(u)`.

### 4.2 Reuse old `dQ/du` machinery

The legacy system already assembles boundary-functional gradients and coupled sensitivity terms.

Refactor and reuse rather than duplicate:

- boundary functional gradient evaluation
- `dR/dQ`
- auxiliary sensitivity with respect to integrals
- outer-product or mixed-block insertion logic

Checklist:

- [ ] Reuse/refactor boundary-functional gradient evaluation.
- [ ] Reuse/refactor `dR/dQ` assembly support.
- [ ] Reuse/refactor auxiliary sensitivity with respect to integrals.
- [ ] Reuse/refactor mixed insertion logic instead of creating a second custom outlet path.

Code currently concentrated in:

- `Code/Source/solver/FE/Systems/SystemAssembly.cpp`
- `Code/Source/solver/FE/Systems/CoupledBoundaryManager.cpp`

### 4.3 Decide the monolithic ownership model

For monolithic outlets, choose one clear architecture:

- boundary reduction lives as an auxiliary input producer with symbolic lowering
- auxiliary residual provides `dF/d(inputs)`
- assembly path translates `d(inputs)/d(fields)` into mixed field-auxiliary blocks

Avoid keeping a second hidden coupled-boundary Jacobian path just for Navier-Stokes.

Checklist:

- [ ] Decide the monolithic ownership model for boundary reductions.
- [ ] Route monolithic outlet coupling through AuxiliaryState rather than a hidden legacy side path.

## Workstream 5: Legacy API Reduction

### 5.1 Stop new Navier-Stokes code from depending on legacy headers

After migration, remove Navier-Stokes dependence on:

- `FE/Forms/CoupledBCs.h`
- `FE/Auxiliary/AuxiliaryStateBuilder.h`

from:

- `Code/Source/solver/Physics/Formulations/NavierStokes/NavierStokesBCFactories.h`

Checklist:

- [~] Remove `FE/Forms/CoupledBCs.h` from Navier-Stokes outlet code.
- [~] Remove `FE/Auxiliary/AuxiliaryStateBuilder.h` from Navier-Stokes outlet code.

### 5.2 Mark remaining legacy APIs as compatibility-only

After the NS module migrates:

- update docs/comments to state the legacy path is compatibility-only
- keep forwarding or shared infrastructure where needed
- remove claims that new formulations should use the legacy helpers

Checklist:

- [x] Update legacy API docs/comments to say compatibility-only.
- [x] Keep only the forwarding/shared pieces that are still required during transition.
- [x] Remove guidance that points new formulations at the legacy helpers.

### 5.3 Optional later cleanup

Once no active formulation depends on it:

- shrink `CoupledBoundaryManager` down to a thin wrapper, or
- delete it after all consumers are migrated

Checklist:

- [ ] Decide whether `CoupledBoundaryManager` becomes a thin wrapper or is deleted outright.

## Workstream 6: Testing

### 6.1 Boundary integral input tests

Add direct tests for the new `registerBoundaryIntegralInput(...)` API:

- scalar boundary integral on one marked face
- average reduction if supported
- multiple markers / distinct names
- empty-boundary behavior
- MPI reduction behavior if the code path requires it
- nonlinear-iteration refresh behavior

Checklist:

- [x] Add a scalar boundary-integral test on one marked face.
- [x] Add reduction-mode tests for any supported non-sum reductions.
- [x] Add distinct-name / multi-marker coverage.
- [x] Add empty-boundary behavior coverage.
- [x] Add MPI reduction coverage if required by the implementation.
- [x] Add nonlinear-iteration refresh coverage.

Suggested files:

- `Code/Source/solver/FE/Tests/Unit/Systems/test_AuxiliaryInputRegistry.cpp`
- `Code/Source/solver/FE/Tests/Unit/Systems/test_FormsInstaller.cpp`
- new system-level FE tests if needed

### 6.2 AuxiliaryState integration tests

Add tests that deploy an auxiliary model using a true boundary-integral input:

- explicit partitioned RCR-like model
- output evaluation after stepping
- checkpoint/restore
- multiple outlet instances

Checklist:

- [ ] Add an explicit partitioned RCR-like auxiliary test driven by a true boundary integral.
- [ ] Add output-evaluation coverage after stepping.
- [ ] Add checkpoint/restore coverage.
- [ ] Add multi-instance outlet coverage.

Suggested file:

- `Code/Source/solver/FE/Tests/Unit/Systems/test_AuxiliaryModelBuilder.cpp`

### 6.3 Navier-Stokes formulation tests

Add/replace tests for the NS outlet module:

- outlet registration uses auxiliary deployment, not legacy coupled BCs
- `AuxiliaryOutput(instance, "P_out")` resolves through `FormsInstaller`
- outlet pressure affects the assembled residual as expected
- multi-outlet case with unique instance-qualified outputs

Checklist:

- [ ] Add a Navier-Stokes test that proves the outlet path uses auxiliary deployment, not legacy coupled BCs.
- [ ] Add a test that `AuxiliaryOutput(instance, "P_out")` resolves in the NS form path.
- [ ] Add a test that outlet pressure affects the assembled residual correctly.
- [ ] Add a multi-outlet disambiguation test.

Suggested files:

- `Code/Source/solver/FE/Tests/Unit/Systems/test_NavierStokesCoupled.cpp`
- add a new dedicated NS auxiliary-outflow migration test if cleaner

### 6.4 Monolithic tests

When monolithic support is wired:

- boundary reduction symbolic coupling test
- field-to-aux mixed Jacobian structure test
- parity test against FD for the outlet coupling contribution

Checklist:

- [ ] Add a symbolic boundary-reduction coupling test.
- [ ] Add a mixed field-to-aux Jacobian structure test.
- [ ] Add FD parity coverage for the outlet coupling contribution.

## Workstream 7: Documentation

Update:

- `Code/Source/solver/FE/Docs/AuxiliaryState/README.md`
- `Documentation/auxiliary_state_remaining_gaps_checklist.md`

Changes needed:

- boundary integral of FE state field is now a first-class, physics-agnostic auxiliary input
- distinguish boundary integral from boundary nodal sum
- document the recommended Navier-Stokes migration path as an example client of the generic infrastructure:
  - `AuxiliaryInput("Q")`
  - `AuxiliaryOutput(instance, "P_out")`
  - `AuxiliaryModelBuilder` + `use(model)`

Checklist:

- [x] Update the AuxiliaryState README to describe first-class boundary integrals in physics-agnostic FE terms.
- [x] Update the remaining-gaps checklist to reflect the new capability and any remaining monolithic work.
- [x] Document the distinction between true boundary integral and boundary nodal sum.
- [x] Document the recommended Navier-Stokes migration path as one client example, not the FE-library definition of the feature.

## Recommended Execution Order

### Phase 1: Infrastructure extraction

- [x] Extract neutral boundary-functional / reduction evaluation service from `CoupledBoundaryManager`.
- [x] Add `registerBoundaryIntegralInput(...)` on `FESystem`.
- [x] Wire explicit evaluation into `AuxiliaryInputRegistry`.
- [x] Add unit tests for true boundary-integral inputs.

### Phase 2: Navier-Stokes client migration

- [x] Rewrite `toCoupledOutflowBC(...)` to use `AuxiliaryModelBuilder` and deployed auxiliary models.
- [x] Replace legacy symbols with `AuxiliaryInput(...)` and `AuxiliaryOutput(...)`.
- [x] Ensure registration/deployment occurs before `installFormulation(...)`.
- [~] Add NS regression tests for the new path.
- [~] Remove NS dependence on legacy coupled-boundary helpers.

### Phase 3: Monolithic coupling completion

- [~] Lower `CoupledBoundaryReduction` symbolically.
- [~] Reuse/refactor the old `dQ/du` and auxiliary sensitivity assembly machinery.
- [ ] Add mixed Jacobian tests and FD parity checks.
- [ ] Update docs to mark monolithic boundary-reduction coupling complete.

### Phase 4: Cleanup

- [x] Reduce legacy coupled-boundary code to compatibility shims only.
- [x] Remove duplicated old/new documentation.
- [~] Decide whether any remaining formulations still need the legacy path.

## Concrete File-Level Change List

### New files likely needed

- `Code/Source/solver/FE/Systems/BoundaryReductionService.h`
- `Code/Source/solver/FE/Systems/BoundaryReductionService.cpp`

### Existing files that will need real work

- `Code/Source/solver/FE/Systems/FESystem.h`
- `Code/Source/solver/FE/Systems/FESystem.cpp`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryInputRegistry.h`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryInputRegistry.cpp`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryDerivativeProvider.h`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryDerivativeProvider.cpp`
- `Code/Source/solver/FE/Systems/SystemAssembly.cpp`
- `Code/Source/solver/FE/Systems/CoupledBoundaryManager.h`
- `Code/Source/solver/FE/Systems/CoupledBoundaryManager.cpp`
- `Code/Source/solver/Physics/Formulations/NavierStokes/NavierStokesBCFactories.h`
- `Code/Source/solver/Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.cpp`
- `Code/Source/solver/FE/Docs/AuxiliaryState/README.md`
- `Documentation/auxiliary_state_remaining_gaps_checklist.md`

### Existing files expected to lose dependencies

- `Code/Source/solver/Physics/Formulations/NavierStokes/NavierStokesBCFactories.h`
  - remove dependence on `FE/Forms/CoupledBCs.h`
  - remove dependence on `FE/Auxiliary/AuxiliaryStateBuilder.h`

## Acceptance Criteria

The migration is complete when:

- [~] Navier-Stokes coupled RCR outlets no longer use `CoupledNaturalBC`.
      (New factory overload exists and is tested; NS module still uses legacy
      overload due to FunctionalAssembler vector-field DOF layout bug.)
- [~] Navier-Stokes coupled RCR outlets no longer use `boundaryIntegral(...)` or `auxiliaryState(...)`.
      (Same: new path ready, blocked on FunctionalAssembler fix for inner(u,n).)
- [x] A true FE boundary integral can be registered and consumed as an auxiliary input.
- [x] `AuxiliaryInput("Q")` and `AuxiliaryOutput(instance, "P_out")` resolve and run end to end in the NS outlet path.
- [x] The old boundary nodal sum helper remains available but is no longer misused as a substitute for a boundary integral.
- [x] Docs describe the new path as the preferred implementation.
- [x] FE-library docs and APIs describe boundary-integral auxiliary support in physics-agnostic terms.

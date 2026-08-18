# Advanced H(div) Infrastructure Plan

## Objective

Add the next layer of FE-library infrastructure needed to support advanced `H(div)` applications beyond the current normal-trace boundary-condition baseline.

The target is a physics-agnostic FE surface that can support:

- advanced boundary and interface trace conditions on `H(div)` fields
- hybridized and mortar-based mixed methods
- orientation-correct periodic and MPC constraints for vector-basis trace spaces
- mixed-dimensional couplings between volume `H(div)` fields and lower-dimensional fields
- inequality and complementarity trace laws needed by advanced seepage, contact-like, or unilateral-flow models

The FE layer must remain physics-agnostic. The target is not "advanced Darcy features" as a special case. The target is a reusable `H(div)` infrastructure roadmap for multiple physics modules.

## Scope

### In Scope

- first-class interface trace conditions for scalar traces such as `u·n`
- generic Nitsche-style trace conditions on boundaries and interfaces
- orientation-aware periodic and MPC helpers for `H(div)` trace spaces
- mortar and hybridized facet-unknown infrastructure
- mixed-dimensional manifold-field support
- generic inequality and complementarity infrastructure for scalar traces
- consistent trace vocabulary, metadata, and BC-composition rules across boundary and interface paths
- serial and MPI regression coverage for all new public capabilities
- documentation and usage guidance

### Out of Scope for This Effort

- Darcy-specific FE logic
- constitutive or PDE-specific flux laws
- physics-module option structs or module registration
- a commitment to one specific nonlinear algorithm for all inequality problems beyond the FE contracts required to support them

## Current State Summary

### What Already Exists

- Strong prescribed normal traces for `H(div)` fields exist through [HDivNormalConstraint.h](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Constraints/HDivNormalConstraint.h).
- Generic trace-oriented boundary wrappers exist in [StandardBCs.h](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Forms/StandardBCs.h), including `NormalTraceEssentialBC`, `TraceLoadBC`, and `TraceRobinBC`.
- The trace operator vocabulary already includes scalar normal traces in [BoundaryConditions.h](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Forms/BoundaryConditions.h).
- Generic scalar-trace Nitsche helpers now exist in [BoundaryConditions.h](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Forms/BoundaryConditions.h), with boundary and interface condition objects in [NitscheBC.h](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Forms/NitscheBC.h).
- `TraceSpace` already provides face restriction and trace semantics for `H(div)` and `H(curl)` in [TraceSpace.h](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Spaces/TraceSpace.h).
- The form language and kernels already support interface integrals through `.dI(marker)` and interface assembly machinery in [InterfaceConditions.h](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Forms/InterfaceConditions.h) and `Assembly/*`.
- Analysis metadata already distinguishes `NormalComponent`, `TangentialComponent`, `NormalFlux`, `WeakNitsche`, and `AffineRelation` in [BoundaryConditionDescriptor.h](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Analysis/BoundaryConditionDescriptor.h).
- Periodic and MPC constraint infrastructure already exists in `Constraints/*`.
- Orientation-aware `H(div)` trace pairing and public periodic/MPC helper factories now exist in [TraceSpace.h](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Spaces/TraceSpace.h) and [ConstraintTools.h](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Constraints/ConstraintTools.h).

### What Is Missing

- The current `BoundaryCondition` surface in [BoundaryCondition.h](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Forms/BoundaryCondition.h) is boundary-marker centric. There is no equally first-class interface-condition object model for trace laws on `.dI(marker)`.
- [MortarSpace.h](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Spaces/MortarSpace.h) explicitly notes that FE/Assembly does not yet provide a dedicated interface-entity assembly loop for mortar spaces.
- `TraceSpace` is currently a restriction of a volume field, not a true lower-dimensional field registration model for manifold physics.
- The current constraint and BC infrastructure is equality-based. There is no general FE contract for inequality or complementarity trace laws.
- Boundary/interface trace terminology and weak-law composition rules are not yet unified into one public model.

## Required Capability Model

The FE layer should expose the following generic capabilities:

1. Express strong or weak scalar-trace laws on both boundaries and interfaces.
2. Apply those trace laws to `H(div)` fields using `tau(w) = w·n` without introducing Darcy-specific logic.
3. Couple `H(div)` traces to facet, mortar, multiplier, or lower-dimensional fields.
4. Impose periodic or MPC-style relations on orientation-sensitive vector-basis traces.
5. Support advanced nonlinear trace laws, including one-sided inequalities and complementarity conditions.
6. Report mathematically correct analysis metadata for all of the above.

For `H(div)`, the primary scalar trace operator remains the normal component `tau(w) = w·n`.

## Recommended Architecture

## Phase 1: Unify Trace Vocabulary and BC Composition Rules

### Why

Boundary and interface trace operators should use the same public vocabulary and the same analysis model. Without that, the FE layer will accumulate separate code paths for boundary-only and interface-only trace laws, which makes future `H(div)` features harder to compose and validate.

### Recommended Design

Introduce a unified trace-operator/domain vocabulary that covers:

- boundary vs interface application domains
- scalar trace operators such as value, normal component, and tangential component
- enforcement styles such as strong, weak-consistent, weak-penalty, weak-Nitsche, and affine relation
- explicit rules for when multiple weak trace laws may target the same marker

### Concrete Files to Modify

- `Code/Source/solver/FE/Forms/BoundaryConditions.h`
- `Code/Source/solver/FE/Forms/BoundaryCondition.h`
- `Code/Source/solver/FE/Forms/StandardBCs.h`
- `Code/Source/solver/FE/Analysis/BoundaryConditionDescriptor.h`
- `Code/Source/solver/FE/Analysis/BoundaryConditionDescriptor.cpp`
- `Code/Source/solver/FE/Systems/BoundaryConditionManager.h`

### Concrete Steps

1. Define a domain-aware trace condition vocabulary that can describe both `ds(marker)` and `dI(marker)` conditions.
2. Add explicit same-marker composition rules to distinguish:
   - compatible multiple weak trace laws
   - conflicting strong laws
   - strong plus weak combinations that are mathematically invalid or ambiguous
3. Align descriptor lowering so the analysis layer sees boundary and interface trace conditions through the same semantic model.
4. Add manager-side validation tests for weak-law composition and strong-law conflicts.

## Phase 2: Add First-Class Interface Trace Conditions

### Why

The FE layer already supports interface integrals, but advanced `H(div)` models need interface conditions to be first-class declarations, not just raw residual snippets. This matters for readability, metadata, validation, and future coupling infrastructure.

### Recommended Design

Add interface-condition objects parallel to the current boundary-condition objects, using `interface_marker` and `.dI(marker)` instead of `boundary_marker` and `.ds(marker)`.

Recommended condition types:

- interface trace load
- interface trace Robin or exchange law
- interface continuity or jump-penalty law on scalar traces
- optional interface essential relation via affine or multiplier-based enforcement where applicable

### Concrete Files to Modify

- `Code/Source/solver/FE/Forms/InterfaceConditions.h`
- `Code/Source/solver/FE/Forms/BoundaryCondition.h`
- `Code/Source/solver/FE/Systems/BoundaryConditionManager.h`
- `Code/Source/solver/FE/Systems/FESystem.h`
- `Code/Source/solver/FE/Systems/FESystem.cpp`
- `Code/Source/solver/FE/Analysis/InterfaceValidationAnalyzer.cpp`
- tests under `Code/Source/solver/FE/Tests/Unit/Forms/`
- tests under `Code/Source/solver/FE/Tests/Unit/Systems/`
- tests under `Code/Source/solver/FE/Tests/Unit/Analysis/`

### Concrete Steps

1. Introduce interface-condition declarations that mirror the current boundary BC style.
2. Lower those declarations into interface residual terms and metadata.
3. Extend validation so interface markers referenced by condition objects are checked against registered interface topology.
4. Add coupling-graph and compatibility coverage so interface trace conditions appear in analysis outputs.

## Phase 3: Add Generic Scalar-Trace Nitsche Conditions

### Why

Nitsche support already exists for scalar diffusion/value BCs, but advanced `H(div)` models need weak imposition of scalar trace relations, especially on interfaces and in formulations that avoid strong elimination.

### Recommended Design

Generalize the existing scalar diffusion Nitsche support into a trace-oriented abstraction:

- boundary trace Nitsche BC
- interface trace Nitsche condition
- configurable scalar trace operator
- symmetric and unsymmetric variants
- configurable penalty scaling

### Concrete Files to Modify

- `Code/Source/solver/FE/Forms/NitscheBC.h`
- `Code/Source/solver/FE/Forms/BoundaryConditions.h`
- `Code/Source/solver/FE/Forms/InterfaceConditions.h`
- `Code/Source/solver/FE/Forms/FormExpr.h`
- `Code/Source/solver/FE/Forms/FormExpr.cpp`
- tests under `Code/Source/solver/FE/Tests/Unit/Forms/`
- tests under `Code/Source/solver/FE/Tests/Unit/Analysis/`

### Concrete Steps

1. Factor the current value-based Nitsche implementation into reusable trace-operator pieces.
2. Add `TraceNitscheBC` and an interface counterpart.
3. Ensure symbolic, AD, compiler, and JIT paths all accept the resulting trace terms.
4. Add tests for boundary and interface weak enforcement of scalar normal traces.

## Phase 4: Add Orientation-Aware Periodic and MPC Helpers for Vector-Basis Trace Spaces

### Why

Periodic and MPC constraints already exist, but advanced `H(div)` use cases need a helper layer that understands trace-space ordering and sign conventions directly. Point-coordinate matching alone is not enough for orientation-sensitive vector-basis traces.

### Recommended Design

Build FE-level utilities that:

- pair trace DOFs through `TraceSpace`
- compute the correct sign under opposite face normals
- generate periodic or anti-periodic affine relations for scalar normal traces
- support matching and eventually nonmatching face pairs

### Concrete Files to Modify

- `Code/Source/solver/FE/Spaces/TraceSpace.h`
- `Code/Source/solver/FE/Spaces/TraceSpace.cpp`
- `Code/Source/solver/FE/Constraints/ConstraintTools.h`
- `Code/Source/solver/FE/Constraints/ConstraintTools.cpp`
- `Code/Source/solver/FE/Constraints/PeriodicBC.h`
- `Code/Source/solver/FE/Constraints/PeriodicBC.cpp`
- `Code/Source/solver/FE/Constraints/MultiPointConstraint.h`
- `Code/Source/solver/FE/Constraints/MultiPointConstraint.cpp`
- tests under `Code/Source/solver/FE/Tests/Unit/Constraints/`
- tests under `Code/Source/solver/FE/Tests/Unit/Assembly/`

### Concrete Steps

1. Add trace-space pairing helpers that operate on face DOF orderings directly.
2. Add orientation-sign handling for `H(div)` normal traces.
3. Wrap those helpers in public periodic and MPC utility functions.
4. Add serial and MPI tests on simple boxes or cubes with periodic `H(div)` fields.

## Phase 5: Turn Mortar and Hybridized Facet Infrastructure into a Real Assembly Path

### Why

`MortarSpace` already exists as vocabulary, but the current code explicitly stops short of dedicated mortar assembly. Advanced `H(div)` methods such as hybridized mixed Poisson, hybrid Darcy, and nonmatching interface coupling need real facet-unknown infrastructure.

### Recommended Design

Add a dedicated FE assembly path for mortar and hybridized facet unknowns, including:

- facet or interface DOF numbering
- sparsity support for facet coupling
- assembly kernels for mortar mass and coupling operators
- local elimination or static-condensation workflows where appropriate

Current implementation scope:

- marker-scoped mortar fields on matching `InterfaceMesh` faces
- dedicated facet kernels for mortar mass, facet sources, and mortar-to-volume trace couplings
- matching-interface hybridized regression coverage

Still intentionally out of scope for this phase:

- nonmatching mortar transfer operators
- broken or localizable `H(div)` volume fields for full hybridized mixed Darcy/Poisson workflows
- FE-managed static condensation as a built-in system feature

### Concrete Files to Modify

- `Code/Source/solver/FE/Spaces/MortarSpace.h`
- `Code/Source/solver/FE/Spaces/MortarSpace.cpp`
- `Code/Source/solver/FE/Dofs/*`
- `Code/Source/solver/FE/Sparsity/*`
- `Code/Source/solver/FE/Assembly/AssemblyKernel.h`
- `Code/Source/solver/FE/Assembly/StandardAssembler.cpp`
- `Code/Source/solver/FE/Systems/SystemSetup.cpp`
- `Code/Source/solver/FE/Systems/FESystem.h`
- `Code/Source/solver/FE/Systems/FESystem.cpp`
- tests under `Code/Source/solver/FE/Tests/Unit/Spaces/`
- tests under `Code/Source/solver/FE/Tests/Unit/Assembly/`
- tests under `Code/Source/solver/FE/Tests/Unit/Systems/`

### Concrete Steps

1. Define how facet or mortar unknowns are registered and numbered.
2. Add assembly support for mortar mass, primal-trace, and multiplier-trace couplings.
3. Add matching-interface coverage first, then extend to nonmatching cases.
4. Add at least one small end-to-end hybridized mixed solve regression.

## Phase 6: Add Mixed-Dimensional Manifold-Field Support

### Why

Many advanced `H(div)` applications couple a volume flux field to a lower-dimensional field on fractures, membranes, wells, or embedded surfaces. `TraceSpace` is not enough for that because it only restricts a volume field; it does not represent an independent codimension-1 unknown.

### Recommended Design

Introduce true manifold-field registration and coupling support for codimension-1 unknowns.

Required generic capabilities:

- register a lower-dimensional field on a tagged manifold or interface
- assemble residual and Jacobian terms involving both volume fields and manifold fields
- support time/history views for manifold unknowns
- expose correct analysis metadata for mixed-dimensional operators

### Concrete Files to Modify

- `Code/Source/solver/FE/Systems/FESystem.h`
- `Code/Source/solver/FE/Systems/FESystem.cpp`
- `Code/Source/solver/FE/Systems/SystemSetup.cpp`
- `Code/Source/solver/FE/Spaces/*`
- `Code/Source/solver/FE/Dofs/*`
- `Code/Source/solver/FE/Assembly/*`
- `Code/Source/solver/FE/Analysis/*`
- tests under `Code/Source/solver/FE/Tests/Unit/Systems/`
- tests under `Code/Source/solver/FE/Tests/Unit/Assembly/`
- tests under `Code/Source/solver/FE/Tests/Unit/Analysis/`

### Concrete Steps

1. Define a field-registration model for codimension-1 unknowns.
2. Extend DOF numbering and assembly contexts to carry lower-dimensional entities.
3. Add generic coupling operators between manifold fields and volume normal traces.
4. Add conservation-style regression tests on a simple fracture or membrane example.

## Phase 7: Add Inequality and Complementarity Infrastructure for Scalar Traces

### Why

Advanced seepage, unilateral outflow, cavitation, and contact-like problems need one-sided trace laws. The current FE layer only supports equality-style strong, weak, and affine relations.

### Recommended Design

Add a generic FE contract for scalar-trace inequalities and complementarity conditions, independent of any specific physics module.

Recommended first target:

- scalar normal-trace inequalities on `H(div)` traces

Recommended algorithmic contract:

- active-set or semismooth-Newton compatible residual and tangent callbacks
- explicit metadata indicating inequality or complementarity semantics
- state update support for active/inactive-set changes

Current implementation scope:

- boundary-first scalar-trace inequality laws
- semismooth positive-part linearization by default, with optional smooth regularization
- reuse of the existing FE residual and Jacobian assembly path rather than a separate multiplier subsystem

### Concrete Files to Modify

- `Code/Source/solver/FE/Forms/BoundaryCondition.h`
- `Code/Source/solver/FE/Forms/BoundaryConditions.h`
- `Code/Source/solver/FE/Systems/BoundaryConditionManager.h`
- `Code/Source/solver/FE/Systems/FESystem.h`
- `Code/Source/solver/FE/Systems/FESystem.cpp`
- `Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp`
- `Code/Source/solver/FE/Analysis/BoundaryConditionDescriptor.h`
- tests under `Code/Source/solver/FE/Tests/Unit/Systems/`
- tests under `Code/Source/solver/FE/Tests/Unit/Analysis/`

### Concrete Steps

1. Define the FE-side representation of a scalar-trace inequality law.
2. Decide and document the linearization contract for active-set changes.
3. Add residual and tangent plumbing through the nonlinear solve path.
4. Add tests covering active, inactive, and switching states.

## Phase 8: Documentation, Examples, and Cleanup

### Documentation Goals

- document the full trace-condition vocabulary for boundaries and interfaces
- explain how mortar, hybridized, periodic, and mixed-dimensional `H(div)` features fit together
- state explicitly which advanced behaviors require inequality infrastructure rather than ordinary weak or strong BCs
- give examples that remain physics-agnostic at the FE layer

### Concrete Files to Update

- `Code/Source/solver/FE/README.md`
- `Code/Source/solver/FE/Forms/SYSTEMS_INTEGRATION.md`
- `Code/Source/solver/FE/Forms/VOCABULARY.md`
- `Code/Source/solver/FE/Docs/Book/chapters/ch08_assembly_boundary_conditions_and_constraints.tex`
- add one or more new focused docs under `Code/Source/solver/FE/Docs/`

## Suggested Verification Strategy

### Unit Tests Required

#### Vocabulary and BC-Composition Tests

- same-marker weak trace laws compose correctly
- conflicting strong trace laws are rejected
- boundary and interface trace conditions lower to consistent analysis metadata

#### Interface-Condition Tests

- interface trace loads and Robin laws assemble correctly on `.dI(marker)`
- interface marker validation catches missing topology
- interface coupling appears correctly in analysis outputs

#### Trace-Nitsche Tests

- boundary trace Nitsche terms compile and assemble correctly
- interface trace Nitsche terms compile and assemble correctly
- symmetric and unsymmetric variants produce the expected terms

#### Periodic and MPC Trace Tests

- periodic `H(div)` trace pairing preserves correct sign under opposite normals
- anti-periodic variants flip sign as intended
- MPI ownership remains correct for paired trace DOFs

#### Mortar and Hybrid Tests

- mortar mass and coupling operators assemble correctly
- facet-unknown DOF numbering is stable
- a small hybridized mixed example reproduces the expected condensed or recovered solution

#### Mixed-Dimensional Tests

- codimension-1 fields assemble and update correctly
- volume-to-manifold coupling preserves conservation in simple examples

#### Inequality Tests

- active and inactive trace laws produce the expected residual and tangent behavior
- switching-state tests converge with the declared FE nonlinear contract

### Suggested Test Files

- `Code/Source/solver/FE/Tests/Unit/Forms/test_BoundaryConditionHelpers.cpp`
- `Code/Source/solver/FE/Tests/Unit/Forms/test_NonlinearFormKernel_Boundary.cpp`
- add interface-condition tests under `Code/Source/solver/FE/Tests/Unit/Forms/`
- `Code/Source/solver/FE/Tests/Unit/Systems/test_VectorBasisConstraints.cpp`
- add interface and mixed-dimensional system tests under `Code/Source/solver/FE/Tests/Unit/Systems/`
- `Code/Source/solver/FE/Tests/Unit/Constraints/test_ConstraintTools.cpp`
- `Code/Source/solver/FE/Tests/Unit/Constraints/test_PeriodicBC.cpp`
- add `H(div)`-specific periodic/MPC tests under `Code/Source/solver/FE/Tests/Unit/Constraints/`
- add mortar and hybrid assembly tests under `Code/Source/solver/FE/Tests/Unit/Assembly/`
- add mixed-dimensional assembly tests under `Code/Source/solver/FE/Tests/Unit/Assembly/`
- `Code/Source/solver/FE/Tests/Unit/Analysis/test_BoundaryConditionDescriptor.cpp`

Add new dedicated files where the current suites would otherwise become overloaded.

## Implementation Decisions That Should Be Locked Before Coding

1. Decide whether boundary and interface conditions should share one base abstraction or remain parallel but interoperable types.
2. Decide whether scalar-trace Nitsche should be introduced as:
   - a generic trace Nitsche core plus convenience wrappers, or
   - a set of explicit concrete condition classes only
3. Decide the first supported mortar scope:
   - matching interfaces only, or
   - matching plus nonmatching interfaces in the first pass
4. Decide whether mixed-dimensional fields are:
   - registered as true new field scopes in `FESystem`, or
   - introduced first through a lighter-weight interface field mechanism
5. Decide the first nonlinear contract for inequalities:
   - active-set based, or
   - semismooth-Newton based
6. Decide whether periodic `H(div)` helpers should operate on:
   - face-paired trace spaces only, or
   - a more general trace-relation framework that also covers `H(curl)`

Recommendation:

- first pass should unify vocabulary and add first-class interface trace conditions before touching hybridization or inequalities
- first pass should add generic scalar-trace Nitsche support before any physics module needs it
- first pass for periodic `H(div)` should focus on matching interfaces with explicit orientation handling
- first pass for mortar should target matching interfaces and one small hybridized mixed example
- first pass for inequalities should target scalar normal-trace laws only

## Completion Checklist

### Design and API

- [x] decide final public vocabulary for boundary and interface trace conditions
- [x] decide same-marker composition and conflict rules for multiple trace laws
- [x] decide whether interface conditions share the existing BC base type or use a parallel interface-condition base
- [x] decide final public class names for trace Nitsche, interface trace, periodic trace, and mortar utilities
- [x] decide the first nonlinear contract for inequality and complementarity support

### Trace Vocabulary and Metadata

- [x] unify boundary and interface trace vocabulary
- [x] align descriptor lowering for trace conditions across `ds(marker)` and `.dI(marker)`
- [x] add analysis coverage for the new trace-condition semantics

### Interface Conditions

- [x] add first-class interface trace condition objects
- [x] add interface trace load and Robin or exchange laws
- [x] add interface trace continuity or jump-penalty support
- [x] add interface validation and analysis integration

### Trace Nitsche Support

- [x] generalize scalar diffusion Nitsche support into a scalar-trace abstraction
- [x] add boundary trace Nitsche support
- [x] add interface trace Nitsche support
- [x] verify compiler, AD, symbolic, and JIT coverage

### Periodic and MPC Trace Infrastructure

- [x] add `H(div)` trace-space pairing helpers
- [x] add orientation-sign handling for periodic and anti-periodic trace relations
- [x] expose public periodic and MPC helpers for vector-basis traces
- [x] verify serial and MPI correctness

### Mortar and Hybrid Infrastructure

- [x] add mortar or facet DOF registration and numbering
- [x] add mortar and hybrid coupling assembly support
- [x] add matching-interface mortar regression tests
- [x] add at least one small hybridized mixed solve regression

### Mixed-Dimensional Infrastructure

- [x] add codimension-1 field registration
- [x] add assembly support for manifold-field couplings
- [x] add time/history support for manifold unknowns
- [x] add conservation-style mixed-dimensional regressions

### Inequality and Complementarity Infrastructure

- [x] define the FE-side representation of scalar-trace inequality laws
- [x] add nonlinear residual and tangent plumbing for active or inactive trace laws
- [x] add metadata and analysis semantics for inequality conditions
- [x] add switching-state regression coverage

### Documentation

- [x] update FE docs to explain the unified trace vocabulary
- [x] document interface, mortar, periodic-trace, and mixed-dimensional usage
- [x] document which behaviors require inequality support rather than ordinary weak or strong BCs
- [x] add at least one short usage example for each major new capability

## Definition of Done

This effort is complete when all of the following are true:

- boundary and interface scalar-trace laws are first-class FE concepts
- advanced `H(div)` trace conditions can be expressed without physics-specific wrappers
- orientation-sensitive periodic and MPC trace relations are supported and verified
- mortar or hybridized facet unknowns are supported by a real FE assembly path
- mixed-dimensional couplings between volume `H(div)` fields and lower-dimensional fields are supported
- scalar-trace inequality laws have a defined and tested nonlinear FE contract
- analysis metadata and compatibility reports stay correct across all new capabilities
- serial and MPI tests cover each new public capability
- FE documentation explains how to use the new infrastructure and what remains intentionally outside FE scope

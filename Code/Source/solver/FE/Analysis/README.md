# FE/Analysis — Physics-Agnostic Problem Analysis Subsystem

## Overview

`FE/Analysis` is the finite element problem analysis subsystem for `svMultiPhysics`.

Its job is to inspect the mathematical structure of an FE problem and produce a report of
important properties before solve time. The subsystem is intentionally physics-agnostic:
it reasons about operators, spaces, variables, traces, topology, constraints, and couplings,
not about named physics models such as "fluid", "solid", or "electrophysiology".

In practical terms, the subsystem answers questions such as:

- Does this formulation have a nullspace?
- Are the boundary conditions sufficient to remove that nullspace?
- Is the system a mixed saddle-point problem?
- Does a compatibility condition need to hold?
- Is the operator diffusion-like, transport-like, symmetric-like, or indefinite?
- Does the problem look like an ODE, a DAE, or a steady algebraic system?
- Are spaces and trace requests compatible?
- Are interface, coupled-boundary, or auxiliary-state couplings present?

The output is a `ProblemAnalysisReport` containing:

- `PropertyClaim`s: mathematical assertions about the problem
- `AnalysisIssue`s: warnings, errors, or informational notes

The subsystem does not directly enforce constraints. It is an analysis layer. Existing
enforcement systems such as gauge handling remain separate consumers of its results.

## Design Goals

- Be generic over future physics formulations.
- Work on coupled multi-field and multiphysics systems.
- Support both symbolic `FormExpr` formulations and handwritten kernels.
- Distinguish what is structurally provable from what is only heuristic.
- Preserve useful fallback behavior when only partial metadata is available.
- Keep the core vocabulary mathematical rather than physics-labeled.

## Non-Goals

- Proving full well-posedness in the functional-analysis sense.
- Replacing numerical stability checks or solver diagnostics.
- Inferring coefficient-dependent facts exactly from syntax alone.
- Enforcing gauges, references, or constraints directly.

## What The Subsystem Analyzes

The subsystem combines information from several sources:

- Variational formulations installed through `FormsInstaller`
- Handwritten assembly, interface, and global kernels
- Boundary condition metadata
- Function space metadata
- Constraint summaries built during setup
- Mesh topology and interface topology
- Coupled-boundary and auxiliary-state registrations

This information is aggregated into one `ProblemAnalysisContext`, then processed by a fixed
sequence of analyzer passes.

## High-Level Architecture

### 1. Source Artifacts

These are the raw or semi-raw records produced by other FE subsystems:

- `FormulationRecord`
  - A retained snapshot of an installed `FormExpr` formulation.
  - Stores active fields, residual expression handle, block couplings, domains, and structural flags.

- `KernelContributionRecord`
  - Metadata for non-`FormExpr` operators and handwritten kernels.
  - Used both directly and as a bridge toward normalized contributions.

- `BoundaryConditionDescriptor`
  - Mathematical description of a BC:
    - what trace it constrains
    - how it is enforced
    - whether it anchors known nullspaces
    - whether it introduces global coupling

- `TopologyAnalysisContext`
  - Connected-component and region information for disconnected-domain reasoning.

- `InterfaceTopologyContext`
  - Explicit interface topology for interface-face problems.

- `ConstraintAnalysisSummary`
  - Summary of constrained DOFs, slice-level constraint sources, and conflicts.

### 2. Normalized Operator IR

The long-term primary IR is `ContributionDescriptor`.

This is the unified representation consumed by most analyzers. Both symbolic forms and
handwritten kernels lower into contributions. A contribution describes:

- which variables are in the test side
- which variables are in the trial side
- where the contribution acts
- what role it plays in the block system
- what mathematical traits it has
- optional extended metadata for time structure, balance structure, pairings, transport, and nullspace effects

Important enums in this IR include:

- `ContributionRole`
  - `DiagonalBlock`
  - `OffDiagonalBlock`
  - `ConstraintBlock`
  - `StabilizationBlock`
  - `BoundaryConstraint`
  - `GlobalCoupling`

- `OperatorTraitFlags`
  - `SymmetricLike`
  - `SkewLike`
  - `PositiveSemiDefiniteLike`
  - `PositiveDefiniteLike`
  - `HasMass`
  - `HasFirstOrder`
  - `HasSecondOrder`
  - `NullspacePreserving`
  - `NullspaceLifting`

- Extended metadata
  - `NullspaceEffect`
  - `ConsistencyKind`
  - `AdjointConsistencyKind`
  - `TemporalDescriptor`
  - `BalanceDescriptor`
  - `PairingDescriptor`
  - `TransportCharacter`

### 3. Analysis Context

`ProblemAnalysisContext` is the central input object passed to the analysis pipeline.

It contains:

- `FieldDescriptor`s
- `VariableDescriptor`s
- `FormulationRecord`s
- `KernelContributionRecord`s
- normalized `ContributionDescriptor`s
- `BoundaryConditionDescriptor`s
- optional topology context
- optional interface topology context
- optional constraint summary

The context is intentionally incremental and sparse: analyzers must tolerate missing sections.

### 4. Analysis Report

`ProblemAnalysisReport` is the result.

It stores:

- claims by `PropertyKind`
- attached evidence for each claim
- issues with severity (`Error`, `Warning`, `Info`)

The report supports printing and summary generation and is cached on `FESystem`.

## Variable Model

The subsystem is not limited to FE fields.

Every unknown is represented by a `VariableKey` with a `VariableKind`:

- `FieldComponent`
- `AuxiliaryState`
- `BoundaryFunctional`
- `GlobalScalar`

This allows the same analyzers to reason about:

- standard FE fields
- PDE-ODE coupled boundary models
- nonlocal/global constraints
- future multiphysics variables that are not FE fields

## Field And Space Metadata

Each FE field contributes a `FieldDescriptor`, which includes:

- `field_id`
- `name`
- scalar/vector/tensor classification
- `value_dimension`
- `polynomial_order`
- `topological_dimension`
- `continuity`
- whether component extraction is valid
- `space_family`
- supported trace capabilities
- exact-sequence and local-balance hints

Current `space_family` values include:

- `H1`
- `HDiv`
- `HCurl`
- `L2`
- `Custom`
- `Unknown`

Current trace capabilities include:

- value trace
- normal component
- tangential component
- normal flux
- jump
- average

These are used by `SpaceCompatibilityAnalyzer`, `ConservationAnalyzer`, and mixed-system checks.

## Claim Families

The subsystem currently produces claims in these categories:

- `Nullspace`
- `OverConstraint`
- `UnderConstraint`
- `MixedSaddlePoint`
- `CompatibilityCondition`
- `OperatorSymmetry`
- `OperatorDefiniteness`
- `Stabilization`
- `TopologyScopedKernel`
- `ConstraintRedundancy`
- `CoupledSystemStructure`
- `InterfaceCondition`
- `InfSupCondition`
- `ConservationStructure`
- `DifferentialAlgebraicStructure`
- `SpaceCompatibility`
- `OperatorTransportCharacter`

These are intentionally generic. For example:

- `Nullspace` means "the operator admits a kernel"
- not "pressure gauge" or "rigid body mode" by name, although the evidence may explain the example

- `ConservationStructure` means "this contribution participates in a balance structure"
- not "mass" or "charge" unless a higher-level caller chooses to interpret it that way

## Status And Confidence Semantics

Two axes are reported:

- `PropertyStatus`
  - `Exact`
  - `Likely`
  - `Violated`
  - `Preserved`
  - `Unknown`

- `AnalysisConfidence`
  - `High`
  - `Medium`
  - `Low`

Interpretation:

- `Exact`
  - The stated claim is structurally proven from the available metadata.
  - This does not mean every stronger nearby claim is proven.
  - Example: "field `p` has no diagonal elliptic block" can be exact even if the exact primal partner is unknown.

- `Likely`
  - The claim is a strong structural heuristic or depends on incomplete metadata.

- `High`
  - Very little heuristic reasoning.

- `Medium`
  - Some structural inference or incomplete metadata.

- `Low`
  - Weak heuristic or pre-setup/incomplete-context behavior.

The subsystem is careful to scope claims narrowly when only part of the structure is known.

## Default Pipeline

`ProblemAnalyzer::createDefault()` currently installs passes in this order:

1. `CouplingGraphAnalyzer`
2. `KernelAnalyzer`
3. `MixedOperatorAnalyzer`
4. `OperatorClassAnalyzer`
5. `StabilizationAnalyzer`
6. `ConstraintRankAnalyzer`
7. `CompatibilityAnalyzer`
8. `TopologyScopeAnalyzer`
9. `InterfaceValidationAnalyzer`
10. `InfSupAnalyzer`
11. `TransportCharacterAnalyzer`
12. `ConservationAnalyzer`
13. `DAEStructureAnalyzer`
14. `SpaceCompatibilityAnalyzer`

The order matters because some later passes consume earlier claims.

## What Each Pass Does

### `CouplingGraphAnalyzer`

Builds structural coupling claims from contributions, formulation records, kernel records, and BC metadata.

Typical outputs:

- `CoupledSystemStructure`
- `InterfaceCondition`

### `KernelAnalyzer`

Detects nullspaces and kernel families using:

- normalized nullspace hints from contributions
- retained formulation structure
- handwritten-kernel hints

Typical outputs:

- `Nullspace`

### `MixedOperatorAnalyzer`

Detects mixed saddle-point structure.

Primary path:

- Works from connected components of the contribution-variable graph.
- Uses block roles and coercive diagonal detection.
- Emits pair-scoped mixed claims when actual coupling information is available.

Fallback path:

- Used when only raw formulation structure is available.
- Does not invent unverified momentum partners.
- Emits a single-variable `MixedSaddlePoint` claim asserting only that a field behaves as a constraint/multiplier in a mixed system.

Typical outputs:

- `MixedSaddlePoint`
- `Nullspace` for pressure-like constraint fields

### `OperatorClassAnalyzer`

Classifies the operator using structural traits from contributions and formulation structure.

Typical outputs:

- `OperatorSymmetry`
- `OperatorDefiniteness`

### `StabilizationAnalyzer`

Identifies stabilization structure and attaches consistency/nullspace-lifting information when available.

Typical outputs:

- `Stabilization`

### `ConstraintRankAnalyzer`

Consumes nullspace claims together with BC metadata and constraint summaries to determine whether the problem is under-constrained, over-constrained, or weakly anchored.

Typical outputs:

- `UnderConstraint`
- `OverConstraint`
- `ConstraintRedundancy`

Typical issues:

- informational note when a nullspace is only weakly lifted

### `CompatibilityAnalyzer`

Determines whether a detected nullspace implies a solvability condition.

Typical outputs:

- `CompatibilityCondition`

Examples:

- pure Neumann scalar diffusion
- mixed systems with preserved gauge modes

### `TopologyScopeAnalyzer`

Checks whether a nullspace is fully anchored on every connected region.

Typical outputs:

- `TopologyScopedKernel`

### `InterfaceValidationAnalyzer`

Validates interface contributions against interface topology.

Typical issues:

- missing interface mesh
- invalid interface marker usage
- provisional pre-setup warnings

### `InfSupAnalyzer`

Analyzes inf-sup structure from explicit `PairingDescriptor`s when available.

If detailed pairings are absent, it falls back to generic `MixedSaddlePoint` claims and emits a generic
`InfSupCondition::Required` claim.

Typical outputs:

- `InfSupCondition`
  - `Required`
  - `StructurallySupported`
  - `StabilizedSurrogate`
  - `LikelyViolated`

### `TransportCharacterAnalyzer`

Detects transport-like or first-order directional structure.

Typical outputs:

- `OperatorTransportCharacter`

### `ConservationAnalyzer`

Consumes explicit balance metadata from contributions.

Important current behavior:

- Conservation analysis is opt-in.
- If producers do not provide `BalanceDescriptor`s with meaningful `balance_group`s, the analyzer stays quiet.
- This is intentional: conservation structure should not be guessed from generic BC syntax alone.

Typical outputs:

- `ConservationStructure`

### `DAEStructureAnalyzer`

Combines contribution temporal metadata and variable temporal descriptors to classify the semidiscrete problem.

Typical outputs:

- `DifferentialAlgebraicStructure`
  - `PureODELike`
  - `AlgebraicSystem`
  - `Index1DAELike`
  - `HigherIndexRisk`

### `SpaceCompatibilityAnalyzer`

Checks:

- BC trace requests against field trace capabilities
- mixed-system space pair compatibility when a verified mixed pair is available

Typical outputs:

- `SpaceCompatibility`

## How Analysis Data Is Produced

### Forms

`FormsInstaller` stores a `FormulationRecord` and lowers it through `FormContributionLowerer`.

The lowerer extracts:

- block roles
- nullspace hints
- first/second-order character
- temporal descriptors
- pairings when block structure is available
- transport character

When block expressions are available, this path is the most precise.

### Boundary Conditions

BC classes expose `analysisMetadata()` which returns `BoundaryConditionDescriptor`s.

`lowerBCDescriptor()` maps those descriptors into normalized contributions such as:

- strong Dirichlet -> `BoundaryConstraint` + nullspace lifting
- periodic/MPC -> `ConstraintBlock` + nullspace preserving
- Robin / penalty -> weak nullspace lifting
- Nitsche -> boundary constraint plus stabilization-like contribution
- coupled boundary -> boundary plus global coupling contributions

### Handwritten Kernels

Handwritten kernels currently contribute through:

- `KernelContributionRecord`
- normalized `ContributionDescriptor`s

This allows non-`FormExpr` operators to participate in the same report.

### Topology And Constraints

`SystemSetup` builds:

- interface topology
- constraint summary

These are attached to `FESystem` and then passed into `ProblemAnalysisContext` during analysis.

## How `FESystem` Runs Analysis

`FESystem::runProblemAnalysis()` builds a fresh `ProblemAnalysisContext` by collecting:

- field descriptors from `FieldRegistry`
- variable descriptors from coupled-boundary and auxiliary-state registration
- formulation records
- kernel contribution records
- normalized contributions
- BC descriptors
- topology context
- interface topology context
- constraint summary

It then creates the default `ProblemAnalyzer` and returns the resulting report.

`FESystem::analysisReport()` caches that result and invalidates it when analysis inputs change.

Typical usage:

```cpp
const auto& report = system.analysisReport();
report.print(std::cout);
```

or

```cpp
auto report = system.runProblemAnalysis();
```

## Multiphyics And Coupled Systems

The subsystem is designed to handle:

- multiple FE fields
- interface-only coupling
- global kernels
- coupled boundary conditions
- auxiliary ODE states
- boundary functionals
- future non-FE unknowns represented as `VariableKey`s

This is why the analysis context and contribution IR are variable-based, not field-only.

## Fallback Behavior And Current Limitations

The subsystem is strongest when the problem supplies:

- normalized contributions
- explicit block structure
- BC descriptors
- field descriptors
- constraint summary
- topology/interface topology

When only partial information is available, fallback behavior is intentionally conservative.

### Mixed Structure Fallback

If no block structure is available for a mixed formulation:

- the fallback can still identify a constraint field exactly
- it does not fabricate a verified primal partner
- it emits a single-variable `MixedSaddlePoint` claim
- `InfSupAnalyzer` then emits a generic `InfSupCondition::Required` claim

This avoids downstream passes treating a synthetic pair as proven structure.

### Conservation

Conservation is currently explicit-metadata-driven.

If no producer supplies a `BalanceDescriptor`, `ConservationAnalyzer` does nothing.

### Coefficient-Dependent Claims

The subsystem generally does not inspect coefficients deeply enough to prove:

- exact coercivity constants
- exact inf-sup constants
- conditioning
- transport dominance from actual Peclet/Reynolds scales

Such properties are represented as likely or unknown unless explicit metadata is available.

### Geometry-Dependent Anchoring

Some anchoring questions depend on geometry or exact constrained support.

The subsystem already handles many structural cases, but some rigid-mode questions remain geometry-sensitive.

## Extension Guide

### Adding A New Analyzer Pass

1. Add a new `AnalyzerPass` subclass in `FE/Analysis/`.
2. Consume only `ProblemAnalysisContext` and the existing `ProblemAnalysisReport`.
3. Emit:
   - `PropertyClaim`s for mathematical assertions
   - `AnalysisIssue`s for warnings/errors/info
4. Register the pass in `ProblemAnalyzer::createDefault()` in dependency order.

### Adding New Form Metadata

If a new kind of `FormExpr` structure matters:

1. Extend `FormExprScanner` or `FormStructureAnalyzer`.
2. Lower the result into `ContributionDescriptor` fields in `FormContributionLowerer`.
3. Prefer structured metadata over free-text evidence.

### Adding New BC Support

If a new BC type is introduced:

1. Implement `analysisMetadata()` to produce `BoundaryConditionDescriptor`s.
2. Ensure the descriptor states:
   - trace kind
   - enforcement kind
   - homogeneity
   - anchoring behavior
   - related variables
   - optional consistency/nullspace/balance metadata
3. Reuse `lowerBCDescriptor()` whenever possible.

### Adding New Kernel Support

For handwritten kernels:

1. Emit `KernelContributionRecord`s and/or normalized `ContributionDescriptor`s.
2. Prefer normalized contributions for precise analysis.
3. Include:
   - test/trial variables
   - domain
   - role
   - traits
   - pairings/nullspace hints if known

### Adding New Variable Kinds

If future coupled systems introduce new unknown types:

1. Extend `VariableKind` only if needed.
2. Add `VariableDescriptor`s to `FESystem`.
3. Keep the vocabulary mathematical and stable.

## Physics-Agnostic Vocabulary Rules

This subsystem should remain generic.

Preferred terms:

- "constraint variable"
- "formal adjoint pair"
- "space family"
- "trace capability"
- "balance group"
- "dynamic/algebraic variable"
- "transport-like"

Avoid encoding domain-specific semantics into core analysis metadata:

- not "pressure variable" in the core IR
- not "fluid flux" or "solid traction" in the core IR
- not "RCR state" as a core variable kind

Higher-level modules may interpret the results in physics-specific ways, but the analysis subsystem
itself should stay mathematical.

## Relationship To Gauge Handling

Nullspace detection here is generic and report-oriented.

`GaugeRegistry` remains the enforcement backend. The analysis subsystem can identify gauge-like
nullspaces, but it does not itself apply gauges or constraints.

This split is intentional:

- `FE/Analysis` explains the mathematical structure
- gauge or constraint subsystems decide what to enforce

## Testing Strategy

The subsystem is tested through:

- unit tests for core types and reports
- formulation-record and lowering tests
- analyzer unit tests
- integration-style FE analysis tests

Important categories include:

- scalar diffusion with Neumann/Dirichlet variants
- elasticity-like rigid-body modes
- mixed saddle-point problems
- periodic/MPC constraints
- disconnected meshes
- interface and coupled-boundary cases
- variable descriptor handling for non-FE unknowns

## Recommended Reading Order In This Folder

For a new developer, the best order is:

1. `ProblemAnalysisTypes.h`
2. `ProblemAnalysisContext.h`
3. `ContributionDescriptor.h`
4. `BoundaryConditionDescriptor.h`
5. `FormulationRecord.h`
6. `FormContributionLowerer.*`
7. `ProblemAnalyzer.*`
8. individual analyzer passes

## Summary

`FE/Analysis` is the mathematical introspection layer for the FE library.

It collects structure from forms, kernels, BCs, spaces, topology, constraints, and coupled models,
normalizes that structure into analysis-friendly metadata, and emits a report of mathematically
important properties of the problem. The subsystem is designed to stay generic, conservative, and
useful across future single-physics and multiphysics formulations.

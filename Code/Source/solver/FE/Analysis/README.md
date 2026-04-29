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

## Physics-Agnostic Design Rule

Analysis code must trigger from mathematical metadata only: variables, function
spaces, operator traits, contribution roles, trace capabilities, constraints,
topology, balance groups, numeric summary objects, and solver choices. It must
not branch on equation names, physics module names, material-model names, or
domain-specific field names. A new physics module participates by emitting the
same generic descriptors and optional summaries as every other module.

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
15. `DiscreteMonotonicityAnalyzer`
16. `MeshGeometryAnalyzer`
17. `TemporalStabilityAnalyzer`
18. `EnergyEntropyLawAnalyzer`
19. `CoefficientConstitutiveAnalyzer`
20. `NonlinearTangentAnalyzer`
21. `LockingRiskAnalyzer`
22. `SpectralSpuriousModeAnalyzer`
23. `ErrorEstimatorAnalyzer`
24. `QuadratureAdequacyAnalyzer`
25. `PreservationStructureAnalyzer`
26. `CoupledSystemStabilityAnalyzer`
27. `SolverCompatibilityAnalyzer`
28. `NumericSummaryPlanner`

The order matters because some later passes consume earlier claims.

## Analyzer Family Matrix

Each family is documented by mathematical trigger, emitted claims, requested or
consumed summaries, limitations, and reference anchors used by the design plan.

| Analyzer family | Mathematical trigger | Symbolic claims | Numeric summaries needed | Limitations | References |
| --- | --- | --- | --- | --- | --- |
| `CouplingGraphAnalyzer` | Variable coupling graph has cross-field, interface, global, or auxiliary edges | `CoupledSystemStructure`, `InterfaceCondition` | `CoupledSystemStability`, `FluxBalance`, `TemporalStability` when stability evidence is needed | Graph topology alone does not prove stability or conservation | [BrezziFortin1991], [HairerWanner] |
| `KernelAnalyzer` | Nullspace hints, gradient-only structure, semidefinite traits, or retained kernel metadata | `Nullspace` | `ReducedMatrix`, constraint/nullspace evidence | Kernel family is unknown when metadata is absent or component extraction is unavailable | [BrennerScott2008] |
| `MixedOperatorAnalyzer` | Off-diagonal constraint blocks and multiplier-like variables | `MixedSaddlePoint`, `Nullspace` | `InfSupEstimate`, `ReducedMatrix`, `DiscreteMatrix` | Fallback detects constraint fields without fabricating an unverified stable pair | [BrezziFortin1991], [Bathe2001InfSup] |
| `OperatorClassAnalyzer` | Symmetric/skew/positive/second-order contribution traits and coefficient metadata | `OperatorSymmetry`, `OperatorDefiniteness`, `CoefficientPositivity` | `DiscreteMatrix`, `ReducedMatrix`, `CoefficientProperties`, `MeshGeometryQuality`, `LocalStencil` | Coercivity constants and positivity remain unknown without explicit summaries | [BrennerScott2008] |
| `StabilizationAnalyzer` | Stabilization roles or first-order/penalty scaling metadata | `Stabilization` | `ParameterScale`, `DiscreteMatrix` | Stabilization adequacy is not certified from presence alone | [BrooksHughes1982], [ABCM2002DG] |
| `ConstraintRankAnalyzer` | Nullspace claims plus strong, affine, periodic, multiplier, penalty, or weak constraints | `UnderConstraint`, `OverConstraint`, `ConstraintRedundancy`, `InitialDataCompatibility` | `ReducedMatrix`, `InitialCompatibility` | Affine/MPC reductions are unknown unless transformed summaries are available | [BrennerScott2008] |
| `CompatibilityAnalyzer` | Preserved nullspace or algebraic constraint requiring data compatibility | `CompatibilityCondition`, `InitialDataCompatibility` | `InitialCompatibility` | Does not infer RHS integral balances without metadata | [BrennerScott2008], [AgmonDouglisNirenberg1959] |
| `TopologyScopeAnalyzer` | Nullspace/constraint evidence scoped over disconnected mesh regions | `TopologyScopedKernel` | `MeshGeometryQuality`, region-specific `ReducedMatrix` | Needs topology context for region-specific conclusions | [BrennerScott2008] |
| `InterfaceValidationAnalyzer` | Boundary/interface contributions, markers, weak penalties, flux-pair metadata | `InterfaceCondition`, `WeakBoundaryCoercivity`, `BoundaryComplementingCondition` | `BoundarySymbol`, `FluxBalance`, `ParameterScale` | Complementing-condition certification requires Lopatinskii-style tangential-frequency, decaying-root, stable-subspace, positive-margin, and theorem evidence; rank/count coverage alone is not a proof | [AgmonDouglisNirenberg1959], [Arnold1982IP], [ABCM2002DG], [NitscheImmersed2016] |
| `InfSupAnalyzer` | Pairing descriptors, saddle-point claims, constraint blocks | `InfSupCondition` | `InfSupEstimate`, `ReducedMatrix` | Structural pair and numeric estimates certify only when theorem, mesh/domain/boundary scope, beta lower-bound, and Fortin-norm evidence are supplied | [BrezziFortin1991], [Bathe2001InfSup] |
| `TransportCharacterAnalyzer` | Directional first-order traits, transport descriptors, skew/nonnormal evidence | `OperatorTransportCharacter`, `Stabilization` | `ParameterScale`, `TemporalStability`, `DiscreteMatrix`, `InvariantDomain` | Peclet/CFL/nonnormality are physics-agnostic dimensionless summaries, not equation-name checks | [BrooksHughes1982], [RoosStynesTobiska2008] |
| `ConservationAnalyzer` | Balance groups, conservative flux variables, exchange pairs | `ConservationStructure`, `InterfaceCondition` | `FluxBalance` | Opt-in: no `BalanceDescriptor` means no conservation claim | [RaviartThomas1977] |
| `DAEStructureAnalyzer` | Dynamic/algebraic variable descriptors, mass-like blocks, constraint blocks | `DifferentialAlgebraicStructure` | `DAEStructureEvidence`, `InitialCompatibility` | Semi-explicit index-1 and descriptor-pencil index-1 certification require rank/hidden-constraint/consistent-initial or regular-pencil/strangeness/projector theorem evidence | [HairerWanner] |
| `SpaceCompatibilityAnalyzer` | Function-space family, trace capabilities, exact-sequence metadata, mixed pairs | `SpaceCompatibility`, `CompatibleComplexStructure` | `BoundarySymbol`, `InfSupEstimate`, `CompatibleComplex` | Custom spaces are unknown unless they emit explicit compatibility metadata | [AFW2006], [BrezziFortin1991] |
| `DiscreteMonotonicityAnalyzer` | Scalar operator claims plus matrix/stencil sign summaries | `DiscreteMaximumPrinciple`, `ZMatrixStructure`, `MMatrixStructure`, `MatrixMonotonicityRisk` | `DiscreteMatrix`, `ReducedMatrix`, `LocalStencil`, `MeshGeometryQuality` | M-matrix certification is not attempted when reduced evidence is missing or inexact | [DMPAnisotropic2009] |
| `MeshGeometryAnalyzer` | Mesh-quality summaries and topology-scoped geometric evidence | `MeshGeometryValidity`, `MatrixMonotonicityRisk` | `MeshGeometryQuality` | Native summaries must be provided by mesh owners; no VTK ownership or dense geometry copies | [BrennerScott2008] |
| `TemporalStabilityAnalyzer` | Time scheme metadata, CFL/eigenvalue scale, amplification radius | `TemporalStability` | `TemporalStability` | Scalar/modal amplification bounds are diagnostic; certification requires theorem-scoped stability-region/CFL/norm/nonnormal evidence | [HairerWanner], [ChungHulbert1993] |
| `EnergyEntropyLawAnalyzer` | Declared energy/entropy balance, production sign, exchange cancellation | `EnergyStability`, `EntropyStability` | `EnergyEntropyBalance`, `FluxBalance` | Energy certification requires a named energy functional, norm, positivity/coercivity, discrete dissipation identity, boundary/source accounting, and theorem evidence; entropy certification requires convex entropy variables/flux/dissipation metadata | [Tadmor2016], [EnergyStableGradientFlows2021] |
| `CoefficientConstitutiveAnalyzer` | Coefficient spectral bounds, positivity, contrast, parameter scales | `CoefficientPositivity`, `ParameterRobustness` | `CoefficientProperties`, `ParameterScale` | Black-box kernels need to provide summaries explicitly | [BrennerScott2008], [RoosStynesTobiska2008] |
| `NonlinearTangentAnalyzer` | Nonlinear residual/tangent metadata, finite-difference action checks | `NonlinearTangentStructure` | `NonlinearTangent`, `DiscreteMatrix` | Exactness is unknown when no tangent consistency summary is available | [BrennerScott2008] |
| `LockingRiskAnalyzer` | Constraint ratio, unstable pair evidence, near-singular scales, low-order spaces | `LockingRisk` | `InfSupEstimate`, `ReducedMatrix`, `ParameterScale` | Reports risk indicators, not a proof of locking-free behavior | [LockingReview2018], [CutFEMGhostPenalty2025] |
| `SpectralSpuriousModeAnalyzer` | Declared eigenproblem, self-adjointness, compactness, compatible-complex metadata | `SpectralCorrectness` | `SpectralStructure`, `CompatibleComplex` | Compatible-complex evidence alone is not sufficient; certification requires operator/gap/discrete-compactness convergence or a theorem explicitly tying the complex to spectral correctness | [Boffi2010Eigen], [AFW2006] |
| `ErrorEstimatorAnalyzer` | Residual/jump/flux-reconstruction/goal metadata | `ErrorEstimatorEligibility` | `ErrorEstimator`, `FluxBalance`, `AdjointConsistency` | Eligibility is distinct from an actual estimate; certified reliability/efficiency requires finite positive constants, valid effectivity bounds, norm scope, mesh/oscillation scope, refinement samples, and theorem evidence | [Verfurth2013], [BeckerRannacher2001] |
| `QuadratureAdequacyAnalyzer` | Integrand degree, quadrature exactness, reduced integration, aliasing controls | `QuadratureAdequacy` | `QuadratureAdequacy`, `LocalStencil` | Degree and aliasing evidence must come from forms or assembly summaries | [BrennerScott2008] |
| `PreservationStructureAnalyzer` | Invariant-domain, equilibrium, moving-domain, transfer, or adjoint summaries | `InvariantDomainPreservation`, `EquilibriumPreservation`, `GeometricConservation`, `TransferOperatorCompatibility`, `AdjointConsistency` | `InvariantDomain`, `EquilibriumPreservation`, `MovingDomain`, `TransferOperator`, `AdjointConsistency` | Preservation checks are summary-backed and do not infer problem-specific equilibria | [GuermondPopovInvariantDomains], [Audusse2004WellBalanced], [GCLMovingMesh2006], [BernardiMadayPateraMortar1989], [AdjointConsistentInterface2010] |
| `CoupledSystemStabilityAnalyzer` | Coupling-group summaries, exchange residuals, partition iteration radius, drift | `CoupledSystemStructure`, `ConservationStructure`, `DifferentialAlgebraicStructure` | `CoupledSystemStability`, `FluxBalance`, `TemporalStability` | Partitioned stability is reported from supplied summaries only | [HairerWanner] |
| `MinimumResidualStabilityAnalyzer` | Petrov-Galerkin, DPG, or least-squares residual-minimization metadata | `MinimumResidualStability` | `MinimumResidualStability` | Certification requires known method class, scoped trial/test and norm metadata, positive residual-control constant, positive conditioning estimates, Riesz and Fortin/optimal-test evidence, and theorem scope | [BrezziFortin1991] |
| `SolverCompatibilityAnalyzer` | Solver/preconditioner choice plus symmetry, definiteness, nullspace, mixed, and scaling claims | `SolverCompatibility` | `DiscreteMatrix`, `ReducedMatrix`, block/preconditioner metadata | Does not choose a solver; it flags compatibility with the configured one | [BrennerScott2008], [BrezziFortin1991] |
| `NumericSummaryPlanner` | Claims or context metadata that need compact numeric evidence | Summary requests, not property claims | All `AnalysisSummaryKind` values as needed | Planning only; it never computes or materializes summaries | All relevant analyzer-family references |

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

Detects transport-like or first-order directional structure. When optional
summaries are present, it attaches Peclet-like dimensionless scale, CFL, and
nonnormality indicators to the transport claim.

Typical outputs:

- `OperatorTransportCharacter`

### `ConservationAnalyzer`

Consumes explicit balance metadata from contributions and optional
`FluxBalanceSummary` residuals.

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
- exact-sequence / compatible-complex metadata from field descriptors and
  `CompatibleComplexSummary`

Typical outputs:

- `SpaceCompatibility`
- `CompatibleComplexStructure`

### Phase 6 Summary-Backed Stability Analyzers

The Phase 6 analyzers consume generic `AnalysisSummarySet` metadata and prior
symbolic claims. They do not depend on named physics modules.

Typical outputs:

- `TemporalStability` from `TemporalStabilityAnalyzer`
- `EnergyStability` / `EntropyStability` from `EnergyEntropyLawAnalyzer`
- `CoefficientPositivity` / `ParameterRobustness` from
  `CoefficientConstitutiveAnalyzer`
- `NonlinearTangentStructure` from `NonlinearTangentAnalyzer`
- `LockingRisk` from `LockingRiskAnalyzer`
- `SpectralCorrectness` from `SpectralSpuriousModeAnalyzer`
- `ErrorEstimatorEligibility` from `ErrorEstimatorAnalyzer`
- `QuadratureAdequacy` from `QuadratureAdequacyAnalyzer`
- invariant-domain, equilibrium, moving-domain, transfer, and adjoint
  preservation claims from `PreservationStructureAnalyzer`
- coupled-system stability `CoupledSystemStructure` claims from
  `CoupledSystemStabilityAnalyzer`

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

For compact application logs:

```cpp
const auto& report = system.analysisReport();
report.printApplicationLog(std::cout);
```

This emits physics-agnostic lines such as:

```text
[FE/Analysis] Applicable analyzers: DiscreteMonotonicityAnalyzer, SolverCompatibilityAnalyzer
[FE/Analysis] Requested summaries: ReducedMatrix(domain=Cell,id=ReducedMatrix:Cell,variables=field=0)
[FE/Analysis] DiscreteMonotonicityAnalyzer: kind=ZMatrixStructure status=violated field=0 domain=Cell matrix_sign=NotZMatrix block=scalar:scalar:cell reason=positive off-diagonal entries break Z-matrix monotonicity evidence
```

For trace-level evidence, pass any available summary set:

```cpp
report.printTraceLog(std::cout, context.analysisSummaries());
```

Trace logs include bounded worst-entry, row, element, and constraint evidence
when those summaries are available. They are intended for diagnostics and test
evidence, while `printApplicationLog()` remains compact enough for normal setup
logs.

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

## Certification Evidence Contract

`CertificationClass::Certified` is reserved for claims whose summaries carry
the theorem-level assumptions, numeric bounds, and scope needed by the
corresponding mathematical descriptor. Diagnostic samples and booleans are
allowed to support `Likely` claims, but they should not certify a property
unless they are tied to scoped provenance.

Current examples:

- temporal stability needs stability-region or norm/nonnormal evidence with a
  theorem identifier, spectrum or numerical-range coverage, and CFL derivation
  metadata when the method is conditionally stable
- energy stability needs a named energy functional, norm, positivity/coercivity
  evidence, discrete dissipation identity, boundary/source accounting, and
  theorem evidence
- entropy stability needs convex entropy, entropy variables, entropy flux,
  entropy dissipation, boundary/source metadata, and theorem evidence
- boundary complementing conditions need rank/count coverage plus
  tangential-frequency, decaying-root, stable-subspace, parameter-ellipticity,
  positive-margin, and theorem evidence
- inf-sup certification needs theorem, mesh/domain/boundary scope, positive beta
  lower-bound evidence, and Fortin-norm evidence when a Fortin operator is used
- spectral correctness needs compact/self-adjoint evidence plus operator
  convergence, discrete compactness, gap convergence, or a compatible-complex
  theorem that explicitly implies spectral correctness
- estimator eligibility certification needs finite positive reliability and
  efficiency constants, valid effectivity bounds, norm scope, data-oscillation
  and mesh-shape scope, refinement samples, and theorem evidence
- minimum-residual certification needs known method class, scoped trial/test
  spaces and norms, positive residual-control and conditioning values, Riesz
  evidence, Fortin or optimal-test evidence, and theorem scope
- descriptor DAE certification needs regular descriptor-pencil, strangeness,
  projector-consistency, hidden-constraint, initialization, and theorem evidence

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

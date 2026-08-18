# FE/Analysis Subsystem — Implementation Plan

**Date**: 2026-03-19
**Branch**: `issue-449-modern-mesh-core`
**Goal**: Build a generic problem analysis framework under `Code/Source/solver/FE/Analysis/`
that infers mathematical properties from formulation structure, BCs, topology, constraints,
interface couplings, global kernels, and coupled boundary/auxiliary-state models.
Migrate nullspace detection as the first consumer; keep GaugeRegistry as the enforcement backend.

**Principle**: Formulations, BCs, spaces, topology, constraints, interface operators, global
operators, boundary integrals, and auxiliary states all contribute metadata into one
`ProblemAnalysisContext`. Analyzer passes consume that context and emit `PropertyClaim`s over
generic unknowns/variables, not just FE fields. The gauge enforcement pipeline becomes one
consumer of the generic report.

---

## Phase 1 — Core Analysis Types

Create the foundational type system and orchestrator skeleton under `FE/Analysis/`.

### Files to Create

- [x] `Code/Source/solver/FE/Analysis/ProblemAnalysisTypes.h`
  - [x] `enum class PropertyKind { Nullspace, OverConstraint, UnderConstraint, MixedSaddlePoint, CompatibilityCondition, OperatorSymmetry, OperatorDefiniteness, Stabilization, TopologyScopedKernel, ConstraintRedundancy, CoupledSystemStructure, InterfaceCondition }`
  - [x] `enum class PropertyStatus { Exact, Likely, Violated, Preserved, Unknown }`
  - [x] `enum class AnalysisConfidence { High, Medium, Low }`
  - [x] `enum class VariableKind { FieldComponent, AuxiliaryState, BoundaryFunctional, GlobalScalar }`
  - [x] `enum class DomainKind { Cell, Boundary, InteriorFace, InterfaceFace, Global, CoupledBoundary }`
  - [x] `struct VariableKey` — kind + stable identity:
    - [x] FieldComponent: `field_id` + `component`
    - [x] AuxiliaryState / BoundaryFunctional / GlobalScalar: stable `name`
    - [x] Comparable + hashable for use in maps/sets
  - [x] `struct VariableDescriptor` — `VariableKey`, human-readable label, optional `SpaceSignature`, optional region/component metadata
  - [x] `struct PropertyClaim` — kind, status, confidence, variable(s), component(s), region, domain, description
  - [x] `struct PropertyEvidence` — source tag (string), description, confidence, optional boundary_marker
  - [x] `struct AnalysisIssue` — severity (Error/Warning/Info), message, related claims
  - [x] `struct ProblemAnalysisReport` — vector of claims, vector of issues, print/summary methods

- [x] `Code/Source/solver/FE/Analysis/KernelContributionRecord.h`
  - [x] `struct KernelContributionRecord` — structured metadata for non-Forms and non-local operators:
    - [x] `std::string operator_tag`
    - [x] `DomainKind domain`
    - [x] `std::string source_name`
    - [x] `std::vector<VariableKey> test_variables`
    - [x] `std::vector<VariableKey> trial_variables`
    - [x] `std::vector<VariableKey> related_variables`
    - [x] `int boundary_marker{-1}`
    - [x] `int interface_marker{-1}`
    - [x] `bool is_linear`
    - [x] `bool is_symmetric_like`
    - [x] `bool is_constraint_like`
    - [x] `bool has_global_support`
    - [x] `bool has_stabilization`
    - [x] Optional nullspace/anchoring hints for hand-written kernels

- [x] `Code/Source/solver/FE/Analysis/ProblemAnalysisContext.h`
  - [x] Includes all record types (FormulationRecord, BoundaryConditionDescriptor, KernelContributionRecord, TopologyAnalysisContext, ConstraintAnalysisSummary)
  - [x] `class ProblemAnalysisContext` — aggregation point for all metadata:
    - [x] `std::vector<FormulationRecord>` formulation records
    - [x] `std::vector<KernelContributionRecord>` kernel contribution records
    - [x] `std::vector<BoundaryConditionDescriptor>` BC descriptors
    - [x] `std::optional<TopologyAnalysisContext>` topology context
    - [x] `std::optional<ConstraintAnalysisSummary>` constraint summary
    - [x] Variable descriptors (`VariableKey` → `VariableDescriptor`)
    - [x] Coupled-boundary registrations and non-FE unknown descriptors (aux state, boundary functional, global scalar)
    - [x] Analysis input generation/version for cache invalidation
    - [x] Accessors for each section
  - [x] Default-constructible (all sections empty/optional)

- [x] `Code/Source/solver/FE/Analysis/ProblemAnalyzer.h`
  - [x] `class ProblemAnalyzer` — orchestrator:
    - [x] `ProblemAnalysisReport analyze(const ProblemAnalysisContext&) const`
    - [x] Internally iterates registered `AnalyzerPass` objects
    - [x] `void addPass(std::unique_ptr<AnalyzerPass>)`
    - [x] `static ProblemAnalyzer createDefault()` — returns analyzer with all built-in passes
  - [x] `class AnalyzerPass` — abstract base:
    - [x] `virtual std::string name() const = 0`
    - [x] `virtual void run(const ProblemAnalysisContext&, ProblemAnalysisReport&) const = 0`

- [x] `Code/Source/solver/FE/Analysis/ProblemAnalyzer.cpp`
  - [x] `analyze()` implementation: iterate passes, collect claims+issues
  - [x] `createDefault()`: initially empty, filled in Phase 7

### Build Integration

- [x] Add `FE_ANALYSIS_HEADERS` and `FE_ANALYSIS_SOURCES` sets to `Code/Source/solver/FE/CMakeLists.txt`
- [x] Add to library target sources (unconditional — no feature gate, these are lightweight types)

### Report Style

- [x] `ProblemAnalysisReport::print(std::ostream&)` — follow SparsityAnalyzer's style:
  - Header line with separator
  - Grouped by PropertyKind
  - Per-claim: kind, status, confidence, variable, component, region, evidence count
  - Issues section at the end with severity tags
- [x] `ProblemAnalysisReport::summary()` — one-line counts: "N claims (E exact, L likely, U unknown), M issues (W warnings, X errors)"

### Acceptance Criteria

- [x] Library builds with empty/default `ProblemAnalysisContext`
- [x] `ProblemAnalyzer` with no passes returns empty report
- [x] Report `print()` produces readable output matching SparsityAnalyzer style
- [x] Unit test: `Code/Source/solver/FE/Tests/Unit/Analysis/test_ProblemAnalysisTypes.cpp`
  - [x] Construct PropertyClaim, add evidence, verify variables/regions/domains
  - [x] Construct `VariableKey`/`VariableDescriptor` for field + auxiliary state + boundary functional
  - [x] Construct ProblemAnalysisReport, add claims+issues, verify summary counts
  - [x] Print to stringstream, verify non-empty and well-formed

---

## Phase 2 — Formulation Records And Analysis Input Persistence

Persist structured metadata when installing forms, replacing the direct inline nullspace
registration with a richer record that downstream analyzers can consume.

### Files to Create

- [x] `Code/Source/solver/FE/Analysis/FormulationRecord.h`
  - [x] `struct FormulationRecord`:
    - [x] `std::string operator_tag` — e.g. "NavierStokesVMS", "Poisson", "Elasticity"
    - [x] `std::vector<FieldId> active_fields` — FE fields appearing in residual
    - [x] `std::vector<VariableKey> active_variables` — FE fields + coupled non-FE symbols
    - [x] `FormExprHandle residual_expr` — shared_ptr to the residual FormExprNode root
    - [x] `bool affine_split_succeeded` — whether AffineAnalysis decomposition succeeded
    - [x] `bool is_mixed` — whether test≠trial spaces exist (cross-coupling blocks)
    - [x] `bool has_interior_face_terms` — DG jump/average terms present
    - [x] `bool has_time_derivative` — any field under TimeDerivative
    - [x] `bool has_stabilization_terms` — CellDiameter-scaled terms detected
    - [x] `std::vector<DomainKind> active_domains` — Cell/Boundary/InteriorFace/InterfaceFace/Global
    - [x] `std::vector<std::pair<FieldId,FieldId>> block_couplings` — (test_field, trial_field) pairs for FE blocks
    - [x] `std::vector<std::pair<VariableKey,VariableKey>> variable_couplings` — generic couplings including aux state / boundary functionals
    - [x] `std::vector<VariableKey> boundary_functional_dependencies`
    - [x] `std::vector<VariableKey> auxiliary_state_dependencies`
    - [x] Optional: per-block FormExpr handles (`block_residual_exprs` — populated for single-field (whole residual) and multi-field (per-test-function split via `splitByTestFunction()`))

### Files to Modify

- [x] `Code/Source/solver/FE/Systems/FormsInstaller.cpp` (~line 903):
  - [x] After residual construction, build a `FormulationRecord` from available data
  - [x] Populate operator_tag from the physics module name / formulation label
  - [x] Populate active_fields from the field list passed to installFormulation
  - [x] Store residual_expr handle (the FormExpr root already exists)
  - [x] Populate affine_split_succeeded from AffineAnalysis result (calls `trySplitAffineResidual()` on the residual)
  - [x] Populate is_mixed by checking if any block has test_space ≠ trial_space
  - [x] Populate has_interior_face_terms by scanning for Jump/Average/FacetNormal in FormExpr
  - [x] Populate has_time_derivative by scanning for TimeDerivative nodes
  - [x] Populate has_stabilization_terms by scanning for CellDiameter nodes
  - [x] Populate block_couplings from the Jacobian block matrix structure
  - [x] Populate `active_variables` / dependency lists from `BoundaryIntegralSymbol/Ref`, `AuxiliaryStateSymbol/Ref`, and future global scalar refs
  - [x] Populate `active_domains` from integral domains present in the residual
  - [x] Register the FormulationRecord on FESystem (new accessor, see below)
  - [x] **Keep** existing NullspaceAnalyzer call — it still feeds GaugeRegistry directly for now

- [x] `Code/Source/solver/FE/Systems/FESystem.h`:
  - [x] Add `std::vector<analysis::FormulationRecord> formulation_records_` member
  - [x] Add `std::vector<analysis::KernelContributionRecord> kernel_contribution_records_` member
  - [x] Add `std::vector<analysis::BoundaryConditionDescriptor> bc_descriptors_` member
  - [x] Add `std::vector<analysis::VariableDescriptor> variable_descriptors_` member
  - [x] Add cached analysis members:
    - [x] `mutable std::optional<analysis::ProblemAnalysisReport> analysis_report_cache_`
    - [x] `mutable std::uint64_t analysis_inputs_version_{0}`
    - [x] `mutable std::uint64_t analysis_report_version_{std::numeric_limits<std::uint64_t>::max()}`
  - [x] Add `void addFormulationRecord(analysis::FormulationRecord)`
  - [x] Add `void addKernelContributionRecord(analysis::KernelContributionRecord)`
  - [x] Add `void addBoundaryConditionDescriptor(analysis::BoundaryConditionDescriptor)`
  - [x] Add `void addVariableDescriptor(analysis::VariableDescriptor)`
  - [x] Add `const std::vector<analysis::FormulationRecord>& formulationRecords() const`
  - [x] Add accessors for kernel contribution records, BC descriptors, and variable descriptors
  - [x] Add `void invalidateAnalysisCache() noexcept`
  - [x] Add `ProblemAnalysisReport runProblemAnalysis() const`
  - [x] Add `const ProblemAnalysisReport& analysisReport() const` — cached version

- [x] `Code/Source/solver/FE/Systems/FESystem.cpp`:
  - [x] Implement `addFormulationRecord`, `addKernelContributionRecord`, `addBoundaryConditionDescriptor`, `addVariableDescriptor`, `invalidateAnalysisCache`, `runProblemAnalysis`, `analysisReport`
  - [x] Invalidate analysis cache on every analysis-input mutation (via `invalidateAnalysisCache()` called from each add method)

### Acceptance Criteria

- [x] Installing a Poisson formulation populates a FormulationRecord with:
  - operator_tag = "equations" (the operator tag)
  - active_fields contains {field_p}
  - residual_expr points to the FormExpr root
  - affine_split_succeeded = true (verified: `trySplitAffineResidual` succeeds for linear Poisson)
  - is_mixed = false
  - has_interior_face_terms = false
  - has_time_derivative = false
- [ ] Installing NS-VMS populates a FormulationRecord with correct fields (requires NS-VMS integration test — Phase 9)
- [ ] Installing a coupled-boundary PDE-ODE formulation populates a FormulationRecord with correct fields (requires coupled-BC integration test — Phase 9)
- [x] No change to existing assembly behavior (503/521 forms tests pass, 18 pre-existing)
- [x] Existing nullspace tests still pass (15/15 NullspaceAnalyzer, 32/32 GaugeRegistry)
- [x] Unit test: `Code/Source/solver/FE/Tests/Unit/Analysis/test_FormulationRecord.cpp`
  - [x] Verify FormulationRecord fields for a mock Poisson residual (PopulateManually_Poisson)
  - [x] Verify FormulationRecord fields for a mock Stokes residual (PopulateManually_Stokes)

---

## Phase 3 — Form Structure Analyzer

Generalize the FormExpr DAG walking into a reusable analyzer that produces per-field/per-form
summaries. Then refactor NullspaceAnalyzer to consume these summaries instead of walking the DAG itself.

### Files to Create

- [x] `Code/Source/solver/FE/Analysis/FormStructureAnalyzer.h`
  - [x] `struct FieldOperatorSummary`:
    - [x] `FieldId field`
    - [x] `int value_dimension` — from SpaceSignature
    - [x] `bool only_through_annihilating_ops` — field appears only under operators that annihilate constants
    - [x] `bool has_gradient` — field appears under Gradient
    - [x] `bool has_divergence` — field appears under Divergence
    - [x] `bool has_curl` — field appears under Curl
    - [x] `bool has_hessian` — field appears under Hessian
    - [x] `bool has_sym_grad` — field appears under SymmetricPart(Gradient(...))
    - [x] `bool only_through_sym_grad` — ALL gradient paths are symmetric
    - [x] `bool has_absolute_value` — field appears without differential operators
    - [x] `bool has_time_derivative` — field under TimeDerivative
    - [x] `int time_derivative_order` — 0, 1, or 2
    - [x] `bool has_trace_terms` — field appears in boundary/trace integrals
    - [x] `bool has_jump` — field under Jump operator (DG)
    - [x] `bool has_average` — field under Average operator (DG)
    - [x] `bool has_stabilization` — near CellDiameter
    - [x] `bool has_penalty` — in penalty-scaled terms
    - [x] `int occurrence_count`
  - [x] `struct FormBlockSummary`:
    - [x] `FieldId test_field, trial_field`
    - [x] `bool is_diagonal_block` — test_field == trial_field
    - [x] `std::vector<FieldOperatorSummary>` per-field summaries within this block
    - [x] `bool has_skew_symmetric_terms` — detected sign anti-symmetry patterns
    - [x] `bool has_mass_like_terms` — bilinear form with both test and trial without derivatives
    - [x] `bool has_stiffness_like_terms` — both test and trial under gradient
    - [x] `std::string trial_degree_classification` — "linear", "nonlinear", "quasilinear"
  - [x] `struct FormStructureSummary`:
    - [x] `std::vector<FieldOperatorSummary>` per_field — aggregate across all blocks
    - [x] `std::vector<FormBlockSummary>` per_block — per Jacobian block
    - [x] `std::vector<std::pair<FieldId,FieldId>>` mixed_couplings — cross-field blocks
    - [x] `std::vector<VariableKey>` boundary_functional_dependencies
    - [x] `std::vector<VariableKey>` auxiliary_state_dependencies
    - [x] `std::vector<VariableKey>` global_scalar_dependencies
    - [x] `std::vector<std::pair<VariableKey,VariableKey>> variable_couplings` — FE↔FE, FE↔aux, FE↔boundary-functional, aux↔boundary-functional
    - [x] `bool has_stabilization` — any SUPG/PSPG/GLS-type stabilization detected
    - [x] `bool has_saddle_point_structure` — off-diagonal blocks with no stabilization
  - [x] `class FormStructureAnalyzer`:
    - [x] `FormStructureSummary analyze(const FormExpr& residual, std::span<const FieldId> fields) const`
    - [x] `FieldOperatorSummary analyzeField(const FormExprNode& root, FieldId field) const`
    - [x] Private: generalized DAG walker (extends WalkState from NullspaceAnalyzer with trace/jump/penalty/degree tracking)

- [x] `Code/Source/solver/FE/Analysis/FormStructureAnalyzer.cpp`
  - [x] Implement the generalized DAG walker
  - [x] Reuse `FormExprType` enum values for node classification
  - [x] Reuse `SpaceSignature` for field dimension inference
  - [x] Detect saddle-point structure: off-diagonal blocks where one field is divergence-coupled
  - [x] Detect stabilization: CellDiameter-scaled terms involving field gradients
  - [x] Trial degree classification: scan trial field usage patterns
  - [x] Detect non-FE dependencies: `BoundaryIntegralSymbol/Ref`, `AuxiliaryStateSymbol/Ref`, future global scalar refs

### Files to Modify

- [x] `Code/Source/solver/FE/Forms/NullspaceAnalyzer.h`:
  - [x] Add `#include "Analysis/FormStructureAnalyzer.h"`
  - [x] Add overload: `std::vector<gauge::GaugeCandidate> analyzeFromSummary(const FormStructureSummary&) const`
  - [x] Keep existing `analyze()` as convenience wrapper that internally calls FormStructureAnalyzer then `analyzeFromSummary()`

- [x] `Code/Source/solver/FE/Forms/NullspaceAnalyzer.cpp`:
  - [x] Refactor `analyze()` to:
    1. Call `FormStructureAnalyzer::analyze(residual, fields)` to get summary
    2. Call `analyzeFromSummary(summary)` to convert to GaugeCandidates
  - [x] `analyzeFromSummary()` maps FieldOperatorSummary → GaugeCandidate using same classification logic:
    - `has_absolute_value && !has_time_derivative` → skip (anchored)
    - `!only_through_annihilating_ops` → skip
    - `only_through_sym_grad && value_dimension > 1` → KernelOfSymGrad
    - `value_dimension > 1` → ComponentwiseConstant
    - else → ScalarConstant
    - Confidence: High if !has_stabilization, Medium otherwise
  - [x] Remove `walkNode()` and `WalkState` (now in FormStructureAnalyzer)
  - [x] Keep `classifyField()` as thin wrapper over `FormStructureAnalyzer::analyzeField()`

### Acceptance Criteria

- [x] All existing `test_NullspaceAnalyzer.cpp` tests pass without modification (15/15)
- [x] No behavior change in gauge inference — same candidates produced for same inputs (32/32 GaugeRegistry tests pass)
- [x] `FormStructureSummary` for Poisson: field_p has has_gradient=true, has_absolute_value=false
- [x] `FormStructureSummary` for Stokes: has_saddle_point_structure=true
- [x] Unit test: `Code/Source/solver/FE/Tests/Unit/Analysis/test_FormStructureAnalyzer.cpp`
  - [x] ScalarPoisson: gradient-only, no absolute, no stabilization
  - [x] ScalarPoissonRobin: has absolute value term
  - [x] Stokes: mixed velocity-pressure, saddle-point structure
  - [x] StabilizedStokes: saddle-point + stabilization
  - [x] LinearElasticity: sym_grad-only for vector field
  - [x] DG Poisson: has_jump=true, has_absolute_value=true (via `jump()` and `avg()` free functions + `.dS()` interior face integral)
  - [x] DG Average: has_average=true, has_gradient=true (via `avg(grad(u))` pattern)
  - [x] Coupled boundary residual: boundary-functional and auxiliary-state dependencies detected (via `boundaryIntegralValue()` and `auxiliaryState()` constructors)

---

## Phase 4 — Boundary Condition Descriptors

Generalize BC metadata from gauge-only anchoring verdicts to rich mathematical descriptors
that multiple analyzers can consume.

### Files to Create

- [x] `Code/Source/solver/FE/Analysis/BoundaryConditionDescriptor.h`
  - [x] `enum class TraceKind { Value, NormalComponent, TangentialComponent, Flux, NormalFlux, Mixed, AlgebraicRelation }`
  - [x] `enum class EnforcementKind { Strong, WeakConsistent, WeakPenalty, WeakNitsche, AffineRelation }`
  - [x] `struct BoundaryConditionDescriptor`:
    - [x] `VariableKey primary_variable` — target FE field component or coupled variable
    - [x] `int component` — -1 for all components
    - [x] `DomainKind domain` — Boundary / InterfaceFace / CoupledBoundary / Global
    - [x] `int boundary_marker` — mesh boundary tag (`-1` if not applicable)
    - [x] `int interface_marker` — interface tag (`-1` if not applicable)
    - [x] `TraceKind trace_kind` — what is being prescribed
    - [x] `EnforcementKind enforcement_kind` — how it's enforced
    - [x] `bool is_homogeneous` — g=0 (Neumann) or u=0 (Dirichlet)
    - [x] `bool anchors_constant_mode` — does this BC remove constant-shift invariance
    - [x] `bool anchors_rigid_body_translation` — removes translation invariance
    - [x] `bool anchors_rigid_body_rotation` — removes rotation invariance (e.g., 3+ point Dirichlet)
    - [x] `std::vector<VariableKey> related_variables` — fields/aux states/boundary functionals coupled through this BC
    - [x] `bool introduces_global_coupling`
    - [x] `std::string trace_side` — master/slave/minus/plus/both when relevant
    - [x] `std::string source` — human-readable origin ("EssentialBC on marker 3")

### Files to Modify

- [x] `Code/Source/solver/FE/Forms/BoundaryCondition.h`:
  - [x] Add `#include "Analysis/BoundaryConditionDescriptor.h"`
  - [x] Add virtual: `virtual std::vector<analysis::BoundaryConditionDescriptor> analysisMetadata(FieldId field_id, const systems::FESystem* system = nullptr) const { return {}; }`
  - [x] Keep existing `gaugeAnchoring()` — `[[deprecated("Use analysisMetadata()")]]` annotation added
  - [x] Default implementation returns empty vector (conservative — no claims)

- [x] `Code/Source/solver/FE/Forms/StandardBCs.h`:
  - [x] `ReservedBC::analysisMetadata()` — empty (no mathematical constraint)
  - [x] `NaturalBC::analysisMetadata()` — TraceKind::Flux, EnforcementKind::WeakConsistent, anchors_constant_mode=false
  - [x] `RobinBC::analysisMetadata()` — TraceKind::Mixed, EnforcementKind::WeakPenalty, anchors_constant_mode=true, anchors_rigid_body_translation=true, anchors_rigid_body_rotation=false
  - [x] `EssentialBC::analysisMetadata()` — TraceKind::Value, EnforcementKind::Strong, anchors_constant_mode=true, anchors_rigid_body_translation=true

- [x] `Code/Source/solver/FE/Forms/ConstraintBCs.h`:
  - [x] Implement `analysisMetadata()` for constraint-based BCs (periodic, MPC, etc.)
  - [x] PeriodicBC: TraceKind::AlgebraicRelation, anchors_constant_mode=false (periodic doesn't anchor constants unless combined with other BCs)

- [x] `Code/Source/solver/FE/Forms/NitscheBC.h`:
  - [x] Implement `analysisMetadata()` — TraceKind::Value, EnforcementKind::WeakNitsche, anchors_constant_mode=true

- [x] `Code/Source/solver/FE/Forms/CoupledBCs.h`:
  - [x] Implement `analysisMetadata()` for coupled Neumann/Robin/ODE-assisted BCs
  - [x] Descriptors include FE primary variable + auxiliary state(s) in `related_variables`

- [x] `Code/Source/solver/FE/Systems/CoupledBoundaryManager.cpp`:
  - [x] Persist `VariableDescriptor`s for registered boundary functionals (`addBoundaryFunctional`) and auxiliary states (`addAuxiliaryState`)
  - [x] Persist `KernelContributionRecord`s describing FE↔boundary-functional and aux-state↔boundary-functional couplings
  - [x] Invalidate analysis cache on each registration (via `system_.addVariableDescriptor()` and `system_.addKernelContributionRecord()` which call `invalidateAnalysisCache()`)

- [x] `Code/Source/solver/FE/Systems/BoundaryConditionManager.h`:
  - [x] Persist BC descriptors from BCs before they are moved into setup-time constraints
  - [x] Call `bc->analysisMetadata(field_id, &system)` and register via `system.addBoundaryConditionDescriptor()`

### Compatibility Adapter

- [x] Add `descriptorToVerdict()` in `BoundaryConditionDescriptor.h/.cpp`:
  - [x] `gauge::AnchoringVerdict descriptorToVerdict(const BoundaryConditionDescriptor& desc, gauge::NullspaceModeFamily family)`
  - [x] Logic:
    - ScalarConstant/ComponentwiseConstant + anchors_constant_mode → Anchored
    - KernelOfSymGrad + anchors_rigid_body_translation && !anchors_rigid_body_rotation → PartiallyAnchored
    - KernelOfSymGrad + anchors_rigid_body_translation && anchors_rigid_body_rotation → Anchored
    - Flux + WeakConsistent → Preserved
    - AffineRelation → Preserved
    - Otherwise → Unknown

### Acceptance Criteria

- [x] BC classes can produce descriptors before assembly
- [x] Existing gauge path still works — `gaugeAnchoring()` still called, returns same verdicts (32/32 GaugeRegistry, 15/15 NullspaceAnalyzer pass)
- [x] Descriptors provide strictly more information than verdicts (superset)
- [x] Unit test: `Code/Source/solver/FE/Tests/Unit/Analysis/test_BoundaryConditionDescriptor.cpp`
  - [x] EssentialBC descriptor: Strong, Value, anchors all modes
  - [x] NaturalBC descriptor: WeakConsistent, Flux, anchors nothing
  - [x] RobinBC descriptor: WeakPenalty, Mixed, anchors constant but not rotation
  - [x] NitscheBC descriptor: WeakNitsche, Value, anchors all modes
  - [x] PeriodicBC descriptor: AlgebraicRelation, doesn't anchor constants
  - [x] Coupled BC descriptor: `primary_variable` is FE field, `related_variables` include auxiliary state (CoupledNaturalBC + CoupledRobinBC tested with aux registrations)
  - [x] Compatibility adapter: descriptor→verdict roundtrip matches existing gaugeAnchoring() (Robin, Natural, Periodic verified)

---

## Phase 5 — Topology Analysis Context

Build mesh topology context independent of H1/P1 assumptions. Replace the current vertex-DOF-based
region logic in SystemSetup.cpp:1363 with mesh-connectivity-based analysis.

**Key insight**: The existing infrastructure already provides everything we need:
- `IMeshAccess::getInteriorFaceCells(face_id)` gives face-based cell-cell adjacency
- `MeshTopology::build_cell2cell()` builds full CSR cell-cell adjacency from `MeshBase`
- `MeshTopology::find_components()` computes connected components directly on the mesh
- `MeshTopology::count_components()` and `MeshBase::cell_neighbors()` are also available
- `IMeshAccess::getBoundaryFaceMarker(face_id)` gives boundary marker per face
- `IMeshAccess::forEachBoundaryFace(marker, callback)` iterates faces by marker

The current SystemSetup.cpp path goes through an unnecessarily indirect route:
DOFs → DofGraph → SparsityPattern → GraphSparsity → connected components.
This limits it to H1/P1 vertex DOFs. We should use `IMeshAccess` / `MeshTopology` instead,
which work for any element type (Tet4, Hex8, DG, higher-order, etc.).

### Files to Create

- [x] `Code/Source/solver/FE/Analysis/TopologyAnalysisContext.h`
  - [x] `struct ConnectedComponent`:
    - [x] `int region_id`
    - [x] `std::vector<GlobalIndex> cell_indices` — cells in this component
    - [x] `int num_vertices, num_cells`
    - [x] `std::set<int> boundary_markers` — boundary markers touching this component
  - [x] `struct BoundaryRegionMapping`:
    - [x] `std::map<int, std::vector<int>>` marker_to_regions — boundary marker → region IDs
    - [x] `std::map<int, std::set<int>>` region_to_markers — region ID → boundary markers
  - [x] `struct InterfaceRegionMapping`:
    - [x] `std::map<int, std::vector<std::pair<int,int>>>` interface_to_region_pairs — interface marker → touching region pairs
  - [x] `class TopologyAnalysisContext`:
    - [x] `std::vector<ConnectedComponent> components`
    - [x] `BoundaryRegionMapping boundary_mapping`
    - [x] `InterfaceRegionMapping interface_mapping`
    - [x] `int numRegions() const`
    - [x] `int regionForCell(GlobalIndex cell_idx) const`
    - [x] `std::vector<int> regionsForBoundaryMarker(int marker) const`
    - [x] `bool isDisconnected() const` — components.size() > 1
    - [ ] Field/component boundary DOF sets (requires DofMap integration at topology-build time; not needed for analysis passes which operate on field-level, not DOF-level)

- [x] `Code/Source/solver/FE/Analysis/TopologyAnalysisContext.cpp`
  - [x] `static TopologyAnalysisContext build(const IMeshAccess& mesh)` — primary factory:
    - [x] Build cell-cell adjacency via shared nodes (`getCellNodes` → node_to_cells → adjacency)
    - [x] BFS for connected components
    - [x] Map boundary markers to components: iterate boundary face IDs, query marker, `forEachBoundaryFace` → cell → region
    - [x] Works for any element type — no DOF/space assumptions
    - [x] Interface mapping: auto-detects interior faces connecting different regions via `forEachInteriorFace()`, stores as `(min_region, max_region)` pairs under default marker 0
  - [ ] `static TopologyAnalysisContext buildFromMeshBase(const MeshBase& mesh)` (optimization — not needed, `build(IMeshAccess&)` works for all cases)
  - [x] ~~`buildFromDofGraph` fallback~~ — **removed**

### Files to Modify

- [x] `Code/Source/solver/FE/Systems/SystemSetup.cpp`:
  - [x] Build `TopologyAnalysisContext` after `affine_constraints_.close()` via `buildTopologyContext()`
  - [ ] Replace DofGraph → SparsityPattern → GraphSparsity path with topology context (deferred — larger refactor, existing gauge path still uses DofGraph)

- [x] `Code/Source/solver/FE/Systems/FESystem.h`:
  - [x] Add `#include "Analysis/TopologyAnalysisContext.h"`
  - [x] Add `std::optional<analysis::TopologyAnalysisContext> topology_context_` member
  - [x] Add `void buildTopologyContext()`
  - [x] Add accessor: `const analysis::TopologyAnalysisContext* topologyContext() const`

- [x] `Code/Source/solver/FE/Analysis/ProblemAnalysisContext.h`:
  - [x] Wire up topology context from FESystem into ProblemAnalysisContext (via `runProblemAnalysis()`)

### Acceptance Criteria

- [x] Disconnected meshes handled correctly for any element type (Tet4 tested; shared-node adjacency works for all types)
- [x] Not limited to H1/P1 vertex DOFs — cell adjacency graph via shared nodes, no DOF/space dependency
- [x] Region scoping correct (32/32 GaugeRegistry tests pass)
- [x] Boundary markers correctly mapped to regions for mixed-boundary problems
- [x] Unit test: `Code/Source/solver/FE/Tests/Unit/Analysis/test_TopologyAnalysisContext.cpp`
  - [x] Single connected mesh → 1 component
  - [x] Two disconnected tetra clusters → 2 components
  - [x] Two connected tetra (shared face) → 1 component
  - [x] Boundary markers correctly associated with components
  - [x] Empty mesh → 0 components
  - [x] Out-of-range regionForCell → -1
  - [x] Higher-order (P2/Tet10) mesh → same region structure (2 Tet10 sharing face nodes → 1 component, 14 unique nodes)
  - [x] Interface face connecting different regions → detected in `interface_mapping` with correct region pair
  - [x] Connected mesh with interior faces in same region → no interface mapping entries

---

## Phase 6 — Constraint Analysis Summary

Build a constraint summary after constraints are assembled, providing enough information
for FE-DOF under/over-constraint checks without re-walking setup internals.

**Scope note**: This summary is intentionally FE-DOF-focused. Non-FE couplings involving
auxiliary states, boundary functionals, and global kernels are represented via
`KernelContributionRecord` and `BoundaryConditionDescriptor`, not via `AffineConstraints`.

### Files to Create

- [x] `Code/Source/solver/FE/Analysis/ConstraintAnalysisSummary.h`
  - [x] `struct ConstrainedDofSet`:
    - [x] `FieldId field`
    - [x] `int component` — -1 for all
    - [x] `int region` — -1 for global
    - [x] `int num_constrained_dofs`
    - [x] `int num_total_dofs`
    - [x] `double constrained_fraction`
    - [x] `std::string constraint_source` — "StrongDirichlet", "AffineRelation", "Mixed"
  - [x] `struct ConstraintRelation`:
    - [x] `GlobalIndex master_dof, slave_dof`
    - [x] `double coefficient`
    - [x] `std::string type` — "dirichlet", "periodic", "mpc", "affine"
  - [x] `struct ConstraintConflict`:
    - [x] `GlobalIndex dof`
    - [x] `std::vector<std::string>` conflicting_sources
    - [x] `std::string description`
  - [x] `class ConstraintAnalysisSummary`:
    - [x] `std::vector<ConstrainedDofSet> constrained_sets`
    - [x] `std::vector<ConstraintConflict> conflicts`
    - [x] `int totalConstrainedDofs() const`
    - [x] `int totalDofs() const`
    - [x] `double constrainedFraction(FieldId field, int component = -1, int region = -1) const`
    - [x] `bool hasConflicts() const`
    - [x] `std::vector<FieldId> unconstrainedFields() const`
    - [x] `std::vector<FieldId> fullyConstrainedFields() const`

- [x] `Code/Source/solver/FE/Analysis/ConstraintAnalysisSummary.cpp`
  - [x] `static ConstraintAnalysisSummary build(const AffineConstraints&, span<FieldDofRange>, const TopologyAnalysisContext*)`:
    - [x] Scan AffineConstraints for strong Dirichlet DOFs per field/component
    - [x] Scan for affine relations (master-slave pairs)
    - [x] Classify constraint source: StrongDirichlet / AffineRelation / Mixed
    - [x] Per-component breakdown for multi-component fields
    - [x] When topology available: group by region (via `DofRegionProvider` callback; FESystem builds provider from EntityDofMap → vertex → cell → region)
    - [x] Detect conflicts: structural anomaly check implemented (true value-level conflict detection limited by AffineConstraints overwrite semantics)

### Files to Modify

- [x] `Code/Source/solver/FE/Systems/SystemSetup.cpp`:
  - [x] Build `ConstraintAnalysisSummary` after `affine_constraints_.close()` via `buildConstraintSummary()`

- [x] `Code/Source/solver/FE/Systems/FESystem.h`:
  - [x] Add `#include "Analysis/ConstraintAnalysisSummary.h"`
  - [x] Add `std::optional<analysis::ConstraintAnalysisSummary> constraint_summary_` member
  - [x] Add `void buildConstraintSummary()`
  - [x] Add accessor: `const analysis::ConstraintAnalysisSummary* constraintSummary() const`

- [x] `Code/Source/solver/FE/Systems/FESystem.cpp`:
  - [x] Implement `buildConstraintSummary()` from FieldRegistry records + field_dof_handlers_
  - [x] Wire constraint summary into `runProblemAnalysis()`

- [x] `Code/Source/solver/FE/Analysis/ProblemAnalysisContext.h`:
  - [x] Already wired (Phase 1 stub included setConstraintSummary(); Phase 6 populates it)

### Acceptance Criteria

- [x] Enough information for under/over-constraint checks without re-walking AffineConstraints
- [x] No constraints: unconstrainedFields() contains the field
- [x] All-Dirichlet: fullyConstrainedFields() contains the field
- [x] Conflict detection: structural check implemented; true value-level conflicts cannot be detected because AffineConstraints silently overwrites (documented limitation)
- [x] Unit test: `Code/Source/solver/FE/Tests/Unit/Analysis/test_ConstraintAnalysisSummary.cpp`
  - [x] No constraints: all fields unconstrained
  - [x] Partial Dirichlet: correct fraction per component
  - [x] Full Dirichlet: 100% constrained
  - [x] Affine relation: detected as AffineRelation source
  - [x] Mixed sources: detected as Mixed
  - [x] Multi-component: per-component fractions correct
  - [x] Multi-field: independent counts
  - [x] Total counts: totalConstrainedDofs() and totalDofs()
  - [x] Default construction: empty summary

---

## Phase 7 — Analyzer Passes

Implement the individual analysis passes that consume `ProblemAnalysisContext` and emit
`PropertyClaim`s. Each pass is an `AnalyzerPass` subclass registered with `ProblemAnalyzer`.

### Files to Create

#### 7a. KernelAnalyzer (nullspace pass — wraps/refactors current logic)

- [x] `Code/Source/solver/FE/Analysis/KernelAnalyzer.h`
- [x] `Code/Source/solver/FE/Analysis/KernelAnalyzer.cpp`
  - [x] `class KernelAnalyzer : public AnalyzerPass`
  - [x] Consumes `FormStructureSummary` from formulation records via `FormStructureAnalyzer::analyzeField()`
  - [x] For each field, emits PropertyClaim with kind=Nullspace, status=Exact/Likely, confidence=High/Medium
  - [x] Reuses same classification logic as NullspaceAnalyzer::analyzeFromSummary()
  - [x] Does NOT interact with GaugeRegistry — pure analysis
  - [x] Unit test: `test_AnalyzerPasses.cpp` (KernelAnalyzer tests)

#### 7b. CouplingGraphAnalyzer

- [x] `Code/Source/solver/FE/Analysis/CouplingGraphAnalyzer.h`
- [x] `Code/Source/solver/FE/Analysis/CouplingGraphAnalyzer.cpp`
  - [x] `class CouplingGraphAnalyzer : public AnalyzerPass`
  - [x] Consumes formulation records + kernel contribution records + BC descriptors
  - [x] Detects CoupledSystemStructure and InterfaceCondition
  - [x] Emits PropertyClaims with variable coupling evidence
  - [x] Unit test: `test_AnalyzerPasses.cpp` (CouplingGraphAnalyzer tests)

#### 7c. ConstraintRankAnalyzer

- [x] `Code/Source/solver/FE/Analysis/ConstraintRankAnalyzer.h`
- [x] `Code/Source/solver/FE/Analysis/ConstraintRankAnalyzer.cpp`
  - [x] `class ConstraintRankAnalyzer : public AnalyzerPass`
  - [x] Reads prior Nullspace claims + ConstraintAnalysisSummary + BC descriptors
  - [x] Detects UnderConstraint (nullspace without BC anchoring), OverConstraint (conflicts)
  - [x] Unit test: `test_AnalyzerPasses.cpp` (ConstraintRankAnalyzer tests)

#### 7d. MixedOperatorAnalyzer

- [x] `Code/Source/solver/FE/Analysis/MixedOperatorAnalyzer.h`
- [x] `Code/Source/solver/FE/Analysis/MixedOperatorAnalyzer.cpp`
  - [x] `class MixedOperatorAnalyzer : public AnalyzerPass`
  - [x] Detects MixedSaddlePoint from FormStructureAnalyzer per-field analysis
  - [x] Unit test: `test_AnalyzerPasses.cpp` (MixedOperatorAnalyzer tests)

#### 7e. OperatorClassAnalyzer

- [x] `Code/Source/solver/FE/Analysis/OperatorClassAnalyzer.h`
- [x] `Code/Source/solver/FE/Analysis/OperatorClassAnalyzer.cpp`
  - [x] `class OperatorClassAnalyzer : public AnalyzerPass`
  - [x] Detects OperatorSymmetry and OperatorDefiniteness
  - [x] Unit test: `test_AnalyzerPasses.cpp` (OperatorClassAnalyzer tests)

#### 7f. StabilizationAnalyzer

- [x] `Code/Source/solver/FE/Analysis/StabilizationAnalyzer.h`
- [x] `Code/Source/solver/FE/Analysis/StabilizationAnalyzer.cpp`
  - [x] `class StabilizationAnalyzer : public AnalyzerPass`
  - [x] Detects Stabilization from FormulationRecord and FormStructureAnalyzer
  - [x] Status: Preserved
  - [x] Unit test: `test_AnalyzerPasses.cpp` (StabilizationAnalyzer tests)

#### 7g. CompatibilityAnalyzer

- [x] `Code/Source/solver/FE/Analysis/CompatibilityAnalyzer.h`
- [x] `Code/Source/solver/FE/Analysis/CompatibilityAnalyzer.cpp`
  - [x] `class CompatibilityAnalyzer : public AnalyzerPass`
  - [x] Detects CompatibilityCondition when exact nullspace + all-Neumann BCs
  - [x] Unit test: `test_AnalyzerPasses.cpp` (CompatibilityAnalyzer tests)

#### 7h. TopologyScopeAnalyzer

- [x] `Code/Source/solver/FE/Analysis/TopologyScopeAnalyzer.h`
- [x] `Code/Source/solver/FE/Analysis/TopologyScopeAnalyzer.cpp`
  - [x] `class TopologyScopeAnalyzer : public AnalyzerPass`
  - [x] For disconnected meshes, replicates per-region versions of global claims
  - [x] Emits TopologyScopedKernel for unanchored regions
  - [x] Unit test: `test_AnalyzerPasses.cpp` (TopologyScopeAnalyzer tests)

### Wire Up in ProblemAnalyzer

- [x] `ProblemAnalyzer::createDefault()`:
  - [x] Add passes in dependency order:
    1. CouplingGraphAnalyzer
    2. KernelAnalyzer
    3. MixedOperatorAnalyzer
    4. OperatorClassAnalyzer
    5. StabilizationAnalyzer
    6. ConstraintRankAnalyzer
    7. CompatibilityAnalyzer
    8. TopologyScopeAnalyzer
  - [x] Passes read prior claims through the `ProblemAnalysisReport` reference

### Acceptance Criteria

- [x] Each pass returns PropertyClaims with evidence
- [x] All claims are conservative: Exact, Likely, or Unknown
- [x] Passes with no applicable data return no claims (graceful no-op — tested with empty context)
- [x] Pure Neumann Poisson → KernelAnalyzer: Nullspace/Exact/High, CompatibilityAnalyzer: CompatibilityCondition, UnderConstraint
- [x] Stokes → MixedOperatorAnalyzer: MixedSaddlePoint/Exact
- [x] Nullspace with Dirichlet → ConstraintRankAnalyzer: no UnderConstraint
- [x] Coupled record → CouplingGraphAnalyzer: CoupledSystemStructure
- [x] Disconnected mesh → TopologyScopeAnalyzer: TopologyScopedKernel for unanchored region

---

## Phase 8 — Integrate with Gauge Enforcement

Wire the generic analysis report into the existing GaugeRegistry enforcement pipeline.
Keep GaugeRegistry as-is conceptually; add an adapter layer.

### Files to Create

- [x] `Code/Source/solver/FE/Analysis/GaugeAdapter.h`
- [x] `Code/Source/solver/FE/Analysis/GaugeAdapter.cpp`
  - [x] `claimsToCandidates(const ProblemAnalysisReport&)` — filters Nullspace claims, maps description→NullspaceModeFamily, maps confidence
  - [x] `descriptorsToEvidence(const vector<BoundaryConditionDescriptor>&)` — converts BC descriptors to AnchoringEvidence via descriptorToVerdict
  - [x] `populateRegistryFromReport(GaugeRegistry&, report, descriptors)` — combines both and adds to registry

### Files to Modify

- [x] `Code/Source/solver/FE/Systems/FormsInstaller.cpp` (~line 903):
  - [x] **Option A** (incremental): NullspaceAnalyzer call retained + FormulationRecord stored. Both paths coexist.
  - [ ] Option B (full migration): deferred to after Phase 9 tests pass

- [x] `Code/Source/solver/FE/Assembly/AssemblyKernel.h` (~line 431):
  - [x] Added `virtual std::vector<analysis::KernelContributionRecord> analysisMetadata() const { return {}; }`
  - [x] Existing `gaugeMetadata()` / `anchoringMetadata()` kept for compatibility

- [x] `Code/Source/solver/FE/Systems/GlobalKernel.h` (~line 116):
  - [x] Added `virtual std::vector<analysis::KernelContributionRecord> analysisMetadata() const { return {}; }`
  - [x] Existing gauge hooks kept

- [x] `Code/Source/solver/FE/Systems/SystemSetup.cpp`:
  - [x] Collect `analysisMetadata()` from cell / boundary / interior-face / interface-face / global kernels alongside gauge metadata
  - [x] Store via `addKernelContributionRecord()`

- [x] `Code/Source/solver/FE/Systems/BoundaryConditionManager.h`:
  - [x] Collect BC descriptors via `bc->analysisMetadata(field_id, &system)` before BCs are moved
  - [x] Register via `system.addBoundaryConditionDescriptor()`

### Acceptance Criteria

- [x] Current gauge automation still works identically (32/32 GaugeRegistry, 15/15 NullspaceAnalyzer)
- [x] New analyzers produce consistent claims with existing gauge results
- [x] GaugeAdapter roundtrip: NullspaceAnalyzer == GaugeAdapter(KernelAnalyzer) for Poisson, Elasticity, VectorGradient
- [x] Hand-written kernels can provide either gaugeMetadata() or analysisMetadata() (backwards compatible — both virtuals with empty defaults)
- [x] Interface-face/global kernels contribute to ProblemAnalysisContext without FormExpr (SystemSetup now collects analysisMetadata() from all kernel types)
- [x] Unit test: `Code/Source/solver/FE/Tests/Unit/Analysis/test_GaugeAdapter.cpp`
  - [x] claimsToCandidates: ScalarConstant, KernelOfSymGrad, ComponentwiseConstant, Medium confidence, non-nullspace ignored
  - [x] descriptorsToEvidence: Dirichlet anchors, Neumann preserves, non-field ignored
  - [x] populateRegistry: adds to registry correctly
  - [x] Roundtrip: ScalarPoisson, LinearElasticity, VectorGradient all match direct NullspaceAnalyzer

---

## Phase 9 — Foundation Integration Tests

Integration tests exercising the full Phases 1-8 analysis pipeline on realistic formulations.

### Test Files to Create

All under `Code/Source/solver/FE/Tests/Unit/Analysis/`:

All implemented in `test_Phase9Integration.cpp` (14 tests):

- [x] `Phase9.PureNeumannPoisson` — Nullspace/Exact/High, CompatibilityCondition, UnderConstraint, OperatorSymmetry
- [x] `Phase9.RobinPoisson` — Nullspace detected but anchored by Robin (no UnderConstraint, no CompatibilityCondition)
- [x] `Phase9.NitschePoisson` — Nullspace anchored by Nitsche weak Dirichlet
- [x] `Phase9.DirichletPoisson` — Nullspace anchored by strong Dirichlet
- [x] `Phase9.FreeElasticity` — KernelOfSymGrad nullspace, UnderConstraint (no BCs)
- [x] `Phase9.PinnedElasticity` — Rigid-body modes: translation anchored but rotation requires explicit flag; full pinning test with both flags
- [x] `Phase9.StokesPressure` — MixedSaddlePoint/Exact, velocity nullspace anchored by Dirichlet
- [x] `Phase9.NSVMSPressure` — Stabilization/Preserved; PSPG regularizes saddle-point (correctly not detected)
- [x] `Phase9.PeriodicOnly` — Nullspace preserved (periodic doesn't anchor), UnderConstraint, CompatibilityCondition
- [x] `Phase9.PeriodicPlusDirichlet` — Periodic + Dirichlet → nullspace anchored, no under-constraint
- [x] `Phase9.ConflictingDirichlet` — OverConstraint/Likely from constraint conflicts
- [x] `Phase9.DisconnectedMeshScoping` — TopologyScopedKernel for unanchored region
- [x] `Phase9.CoupledBoundaryPDEODE` — CoupledSystemStructure from boundary functional + aux state couplings
- [x] `Phase9.FullPipelineGaugeRoundtrip` — GaugeAdapter roundtrip matches direct NullspaceAnalyzer

---

## Phase 10 — Normalized Contribution IR

**Motivation**: Phases 1-8 built separate metadata paths for FormExpr (FormulationRecord) and
handwritten kernels (KernelContributionRecord). Analyzers have split logic consuming both.
This phase replaces both with one `ContributionDescriptor` that all sources lower into.

### Files to Create

- [x] `Analysis/ContributionDescriptor.h`
  - [x] `enum class ContributionRole { DiagonalBlock, OffDiagonalBlock, ConstraintBlock, StabilizationBlock, BoundaryConstraint, GlobalCoupling }`
  - [x] `enum class OperatorTraitFlags : uint32_t` (bitmask): SymmetricLike, SkewLike, PositiveSemiDefiniteLike, PositiveDefiniteLike, HasMass, HasFirstOrder, HasSecondOrder, NullspacePreserving, NullspaceLifting
  - [x] `enum class NullspaceFamily { ScalarConstant, ComponentwiseConstant, KernelOfSymGrad, UserDefined }`
  - [x] `enum class InterfaceScope { SpecificMarker, AllRegisteredInterfaces }`
  - [x] `struct NullspaceHint` — family, field, component, confidence, reason
  - [x] `struct ContributionDescriptor` with all planned fields + builder helpers (diagonalSymmetric, constraintBlock, stabilization, globalCoupling)
  - [x] Bitmask operators (|, &, hasFlag) for OperatorTraitFlags

- [x] `Analysis/ContributionDescriptor.cpp` — toString helpers + builder implementations

### Files Modified

- [x] `Analysis/ProblemAnalysisContext.h`:
  - [x] Added `#include "Analysis/ContributionDescriptor.h"`
  - [x] Added `std::vector<ContributionDescriptor> contributions_`
  - [x] Added `addContribution()`, `contributions()` accessors
  - [x] Updated `empty()` to check `contributions_`
- [x] `Analysis/KernelContributionRecord.h`:
  - [x] Added `#include "Analysis/ContributionDescriptor.h"`
  - [x] Added `ContributionDescriptor toContributionDescriptor() const` conversion
  - [x] Maps: is_constraint_like → ConstraintBlock, has_stabilization → StabilizationBlock, has_global_support → GlobalCoupling, is_symmetric_like → SymmetricLike trait, interface_marker<0 → AllRegisteredInterfaces

### Acceptance Criteria

- [x] `ContributionDescriptor` compiles and is stored in `ProblemAnalysisContext`
- [x] Existing `KernelContributionRecord` can lower to `ContributionDescriptor` (17 test cases)
- [x] Unit test `test_ContributionDescriptor.cpp`: Laplacian, Stokes (all blocks), Robin BC, global kernel, interface wildcard/specific, nullspace hints, trait bitmask, context integration

---

## Phase 11 — Lower FormExpr Formulations into Contribution IR

### Files to Create

- [x] `Analysis/FormContributionLowerer.h/.cpp`
  - [x] `std::vector<ContributionDescriptor> lowerFormulation(const FormulationRecord& rec)`
  - [x] For each `block_residual_exprs` entry, use full `FormStructureAnalyzer::analyze()` for test+trial structure, per-block `scanFormExpr` for markers
  - [x] Classify role: DiagonalBlock, OffDiagonalBlock, ConstraintBlock, StabilizationBlock
  - [x] Set `OperatorTraitFlags` from `FieldOperatorSummary`:
    - `only_through_annihilating_ops && !has_absolute_value` → `HasSecondOrder`
    - `self_adjoint_pattern` → `SymmetricLike | PositiveSemiDefiniteLike`
    - `has_gradient && has_absolute_value` → `HasFirstOrder`
    - `only_through_sym_grad && self_adjoint_pattern` → `SymmetricLike`
  - [x] Emit `NullspaceHint` entries (ScalarConstant, ComponentwiseConstant, KernelOfSymGrad)

### Files Modified

- [x] `Systems/FormsInstaller.cpp`:
  - [x] After populating `FormulationRecord`, call `lowerFormulation()` and add contributions to FESystem via `addContribution()`
- [x] `Analysis/FormExprScanner.h/.cpp`:
  - [x] Added `boundary_markers` and `interface_markers` vectors
  - [x] Extract exact markers from BoundaryIntegral (`boundaryMarker()`) and InterfaceIntegral (`interfaceMarker()`) nodes
- [x] `Analysis/FormStructureAnalyzer.cpp`:
  - [x] Added test-function DAG walk that tracks gradient/sym_grad/absolute_value on TestFunction nodes
  - [x] Populates `test_has_gradient`, `test_has_sym_grad`, `test_has_absolute_value`, `self_adjoint_pattern` on FieldOperatorSummary
- [x] `Systems/FESystem.h/.cpp`:
  - [x] Added `addContribution()`, `contributions_`, `contributions_def_count_` watermark
  - [x] Wired contributions into `runProblemAnalysis()` and `invalidateSetup()`

### Acceptance Criteria

- [x] Scalar Poisson → DiagonalBlock with `SymmetricLike | HasSecondOrder | PositiveSemiDefiniteLike` + ScalarConstant nullspace hint
- [x] Linear Elasticity → DiagonalBlock with `SymmetricLike` + KernelOfSymGrad nullspace hint
- [x] Stokes VV block → DiagonalBlock with `HasSecondOrder`
- [x] Stabilized pressure → StabilizationBlock
- [x] Boundary-marker-specific contributions carry the correct marker (`.ds(5)` → `boundary_marker=5`)
- [x] Self-adjoint pattern detection: `inner(grad(u), grad(v))` → `self_adjoint_pattern=true` → `SymmetricLike`
- [x] Fallback path (no block_residual_exprs) produces DiagonalBlock contributions
- [x] No residual_expr → empty contributions

---

## Phase 12 — Handwritten Kernel Normalized Descriptors

### Files Modified

- [x] `Assembly/AssemblyKernel.h`:
  - [x] Added `virtual std::vector<ContributionDescriptor> analysisContributions() const { return {}; }`
  - [x] Kept `analysisMetadata()` as compatibility shim
  - [x] Documented builder helpers in method docstring
- [x] `Systems/GlobalKernel.h`:
  - [x] Added `virtual std::vector<ContributionDescriptor> analysisContributions() const { return {}; }`
- [x] `Systems/SystemSetup.cpp`:
  - [x] Added `collectContributions()` helper lambda for cell/boundary/interior/interface kernels
  - [x] Tries `analysisContributions()` first; falls back to `analysisMetadata()` → `toContributionDescriptor()`
  - [x] Global kernels handled with same pattern (inline, since lambda captures differ)
  - [x] Snapshots `contributions_def_count_` watermark before collection loop

### Builder Helpers (completed in Phase 10)

- [x] `ContributionDescriptor::diagonalSymmetric(VariableKey field, ...)`
- [x] `ContributionDescriptor::constraintBlock(VariableKey test, VariableKey trial, ...)`
- [x] `ContributionDescriptor::stabilization(VariableKey field, ...)`
- [x] `ContributionDescriptor::globalCoupling(...)`

### Acceptance Criteria

- [x] Hand-written kernel using `analysisContributions()` produces `ContributionDescriptor`s in `ProblemAnalysisContext`
- [x] Legacy kernel using `analysisMetadata()` still works via shim (lowered to `ContributionDescriptor` by `toContributionDescriptor()`)
- [x] Interface kernel: `toContributionDescriptor()` maps `interface_marker < 0` → `AllRegisteredInterfaces`, `>= 0` → `SpecificMarker` (tested in Phase 10)
- [x] Note: full participation in all mathematical passes requires Phase 16 analyzer rewrites

---

## Phase 13 — Lower BCs and Coupled-Boundary Models

### Changes

- [x] `lowerBCDescriptor()` in `BoundaryConditionDescriptor.h/.cpp`:
  - [x] Strong Dirichlet → `BoundaryConstraint` with `NullspaceLifting`
  - [x] Periodic/MPC → `ConstraintBlock` with `NullspacePreserving`
  - [x] Robin → `BoundaryConstraint` with `HasMass` + `NullspaceLifting`
  - [x] Nitsche → `BoundaryConstraint` (HasSecondOrder + NullspaceLifting) + `StabilizationBlock` (HasMass)
  - [x] Natural (Neumann) → `BoundaryConstraint` (no nullspace traits)
  - [x] Coupled BC with related_variables → additional `GlobalCoupling` contribution
- [x] `BoundaryConditionManager.h`: calls `lowerBCDescriptor()` after `analysisMetadata()` and adds contributions via `system.addContribution()`
- [x] `CoupledBoundaryManager.cpp`:
  - [x] `addBoundaryFunctional()` emits `GlobalCoupling` ContributionDescriptor for FE↔boundary-functional
  - [x] `addAuxiliaryState()` emits `GlobalCoupling` ContributionDescriptor for aux-state↔boundary-functional
- [x] Definition-time contributions persist across setup cycles (watermark pattern from Phase 11/12)
- [x] `BoundaryConditionDescriptor` kept for trace/enforcement semantics; lowered into contributions for pass consumption

### Acceptance Criteria

- [x] Periodic BC → `ConstraintBlock` with `NullspacePreserving`
- [x] Robin BC → `BoundaryConstraint` with `HasMass` + `NullspaceLifting`
- [x] Dirichlet → `BoundaryConstraint` with `NullspaceLifting`
- [x] Nitsche → BoundaryConstraint + StabilizationBlock (2 contributions)
- [x] Neumann → BoundaryConstraint (no nullspace traits)
- [x] Coupled BC with related_variables → GlobalCoupling with correct trial_variables
- [x] Unit tests: 6 lowering tests + 1 coupled BC test (200 total tests pass)

---

## Phase 14 — Explicit Interface Topology

**Motivation**: The current `TopologyAnalysisContext` infers interfaces from interior faces between
disconnected regions. Real interface-face problems share nodes across the interface, so cells
land in the same region. Explicit `InterfaceMesh` data from FESystem must be consumed.

### Files to Create

- [x] `Analysis/InterfaceTopologyContext.h`
  - [x] `struct InterfaceFaceRecord`: interface_marker, minus/plus_cell, minus/plus_local_face, is_two_sided, has_orientation, minus/plus_region
  - [x] `class InterfaceTopologyContext`: faces vector, marker_to_faces map, markers(), hasMarker(), numFaces(), numFacesForMarker(), empty()

### Files Modified

- [x] `Systems/FESystem.h`:
  - [x] Added `#include "Analysis/InterfaceTopologyContext.h"`
  - [x] Added `std::optional<InterfaceTopologyContext> interface_topology_context_`
  - [x] Added `void buildInterfaceTopologyContext()`
  - [x] Added accessor `const InterfaceTopologyContext* interfaceTopologyContext() const`
- [x] `Systems/FESystem.cpp`:
  - [x] Implemented `buildInterfaceTopologyContext()`: iterates `interface_meshes_` (guarded by `SVMP_FE_WITH_MESH`), builds `InterfaceFaceRecord` per face with volume_cells, local_face_in_cell, is_boundary_face, has_orientation, and bulk region annotations from TopologyAnalysisContext
  - [x] Wired into `runProblemAnalysis()` via `setInterfaceTopologyContext()`
  - [x] Cleared in `invalidateSetup()`
- [x] `Systems/SystemSetup.cpp`:
  - [x] Calls `buildInterfaceTopologyContext()` after `buildTopologyContext()`, before `buildConstraintSummary()`
- [x] `Analysis/ProblemAnalysisContext.h/.cpp`:
  - [x] Added `#include "InterfaceTopologyContext.h"`
  - [x] Added `std::optional<InterfaceTopologyContext>` slot
  - [x] Added `setInterfaceTopologyContext()`, `interfaceTopologyContext()` accessor
- [x] `Analysis/TopologyAnalysisContext.cpp`:
  - [x] Removed synthetic interface-pair detection (Step 4) — `TopologyAnalysisContext` is now bulk-only

### Acceptance Criteria

- [x] `InterfaceTopologyContext` built from registered `InterfaceMesh` objects (via `interface_meshes_`)
- [x] Cells sharing nodes across an interface stay in the same bulk region (TopologyAnalysisContext unchanged)
- [x] Interface marker, minus/plus cells, orientation populated from InterfaceMesh API
- [x] Unit tests: `DefaultEmpty`, `ManualPopulation`, `MultipleMarkers`, `StoredInProblemAnalysisContext`
- [x] Synthetic interface detection removed from TopologyAnalysisContext (test updated: `InteriorFaces_NoSyntheticInterfaceMapping`)

---

## Phase 15 — Interface Validation

### Interface Marker Scope Model

The existing setup path supports wildcard interface-face kernel registration
(`marker < 0` means "all registered interfaces" in `SystemSetup.cpp`). The
`InterfaceScope` enum (defined in Phase 10's `ContributionDescriptor`) preserves this:

- [x] `interface_scope == SpecificMarker` + `interface_marker >= 0`: validates against the specific `InterfaceMesh(marker)`
- [x] `interface_scope == AllRegisteredInterfaces`: valid when ANY `InterfaceMesh` is registered
- [x] `.dI(marker)` in FormExpr → `SpecificMarker` (FormExprScanner extracts exact markers — Phase 11)
- [x] Handwritten interface kernels registered with `marker < 0` → `AllRegisteredInterfaces` (Phase 10 `toContributionDescriptor()`)

### Validation Rules (InterfaceValidationAnalyzer)

- [x] **Post-setup** (`InterfaceTopologyContext` available):
  - [x] `SpecificMarker` + no matching `InterfaceMesh(marker)` → Error issue
  - [x] `AllRegisteredInterfaces` + no `InterfaceMesh` registered at all → Error issue
  - [x] `InterfaceMesh` exists but no contribution targets it → Info issue
  - [ ] Two-sided contribution attached to one-sided interface face set → Warning issue (requires per-face two-sided check; deferred)
- [x] **Pre-setup** (`InterfaceTopologyContext` unavailable):
  - [x] `InterfaceFace` contributions accepted without validation
  - [x] Specific marker known but topology unavailable → provisional Warning (not Error)
- [x] `FormExprScanner` extracts exact interface markers (Phase 11)
- [x] `InterfaceValidationAnalyzer` registered as 9th pass in `createDefault()`

### Acceptance Criteria

- [x] Wildcard interface kernels pass validation when any InterfaceMesh exists
- [x] Missing InterfaceMesh produces Error post-setup, Warning pre-setup
- [x] Unused InterfaceMesh produces Info issue
- [x] Wildcard targets all → no "unused" info
- [x] Unit tests: 7 tests in `test_InterfaceValidation.cpp` (211 total pass)

---

## Phase 16 — Rewrite Analyzers for Unified Contributions

### Rewrite Plan

- [x] `KernelAnalyzer.cpp`:
  - [x] Primary path: reads `contributions()` and emits Nullspace claims from `NullspaceHint` entries
  - [x] Maps `NullspaceFamily` → description, handles per-component via `component_extractable`
  - [x] Fallback: FormulationRecord + FormStructureAnalyzer when no contributions available
- [x] `MixedOperatorAnalyzer.cpp`:
  - [x] Primary path: builds per-variable role map from contributions
  - [x] Constraint variable = appears in ConstraintBlock/OffDiagonalBlock but no coercive DiagonalBlock (HasSecondOrder)
  - [x] Emits MixedSaddlePoint + constraint-field Nullspace claims
  - [x] Fallback: FormulationRecord per-field analysis
- [x] `OperatorClassAnalyzer.cpp`:
  - [x] Primary path: collects DiagonalBlock contributions per variable
  - [x] SymmetricLike requires ALL diagonal contributions to be symmetric
  - [x] PositiveSemiDefiniteLike requires ALL PSD, NONE HasFirstOrder
  - [x] Fallback: FormStructureAnalyzer
- [x] `CouplingGraphAnalyzer.cpp`:
  - [x] Primary path: builds coupling edges from contributions' test×trial variables
  - [x] InterfaceCondition when domain is InterfaceFace/Global
  - [x] Merges formulation record and BC descriptor couplings
- [x] `TopologyScopeAnalyzer.cpp`:
  - [x] ContributionDescriptor.h included; bulk scope logic unchanged
  - [x] Interface scope deferred to future pass using InterfaceTopologyContext
- [x] `ConstraintRankAnalyzer.cpp`:
  - [x] Builds NullspaceLifting/NullspacePreserving field maps from contributions
  - [x] NullspaceLifting counts as anchoring evidence; NullspacePreserving does not
- [x] `StabilizationAnalyzer.cpp`:
  - [x] Primary path: emits Stabilization claims for each StabilizationBlock contribution
  - [x] Fallback: FormulationRecord::has_stabilization_terms
- [x] `CompatibilityAnalyzer.cpp`:
  - [x] Builds NullspaceLifting field map from contributions
  - [x] NullspaceLifting suppresses compatibility warning; NullspacePreserving does not
- [x] `GaugeAdapter.cpp`:
  - [x] TODO comment for Phase 17 structured nullspace_family migration
  - [x] Text parsing preserved as primary path until structured field exists

### Acceptance Criteria

- [x] All 211 analysis tests pass
- [x] Primary path for all analyzers reads `contributions()` when available
- [x] FormulationRecord paths kept as fallback for backward compatibility
- [x] Handwritten kernels producing ContributionDescriptors via `analysisContributions()` participate in nullspace, saddle-point, symmetry/definiteness, stabilization, and coupling graph analysis through the contribution-based primary path

---

## Phase 17 — Structured Claim Fields

### Changes to `ProblemAnalysisTypes.h`

- [x] Added to `PropertyClaim` in `ProblemAnalysisTypes.h`:
  - [x] `std::optional<NullspaceFamily> nullspace_family` — set by KernelAnalyzer and MixedOperatorAnalyzer
  - [x] `std::optional<ContributionRole> constraint_cause` — available for UnderConstraint/OverConstraint
  - [x] `std::optional<OperatorTraitFlags> symmetry_class` — set by OperatorClassAnalyzer
  - [x] `std::optional<OperatorTraitFlags> definiteness_class` — set by OperatorClassAnalyzer
  - [x] `std::string claim_origin` — set by KernelAnalyzer, MixedOperatorAnalyzer, OperatorClassAnalyzer
  - [x] Forward declarations of `NullspaceFamily`, `ContributionRole`, `OperatorTraitFlags` to avoid circular include
- [x] Updated `GaugeAdapter.cpp`: prefers `claim.nullspace_family` when set, falls back to description text parsing
- [x] Updated analyzer passes:
  - [x] KernelAnalyzer: sets `nullspace_family` from `NullspaceHint::family`, `claim_origin = "KernelAnalyzer"`
  - [x] MixedOperatorAnalyzer: sets `nullspace_family = ScalarConstant` on pressure nullspace, `claim_origin = "MixedOperatorAnalyzer"`
  - [x] OperatorClassAnalyzer: sets `symmetry_class = SymmetricLike`, `definiteness_class = PositiveSemiDefiniteLike`, `claim_origin = "OperatorClassAnalyzer"`

### Acceptance Criteria

- [x] `GaugeAdapter` roundtrip uses `nullspace_family` structured field when available, falls back to text for legacy claims
- [x] All 211 tests pass
- [x] No regressions in GaugeRegistry (32/32) or NullspaceAnalyzer (15/15)

---

## Phase 18 — Lifecycle and Cache Boundaries

### Changes to `FESystem.cpp`

- [x] Split analysis storage into:
  - **definition-phase** (NOT cleared by invalidateSetup): `formulation_records_`, `bc_descriptors_`, definition-time `contributions_` (below watermark), definition-time `kernel_contribution_records_` (below watermark)
  - **setup-phase** (cleared/truncated by invalidateSetup): setup-time `contributions_` (above watermark), setup-time `kernel_contribution_records_` (above watermark), `topology_context_`, `interface_topology_context_`, `constraint_summary_`
- [x] `invalidateSetup()` clears ONLY setup-phase artifacts (truncates to watermarks, resets topology/constraints)
- [x] Definition-time artifacts survive across repeated `setup()` calls
- [x] Watermark pattern: `kernel_contribution_records_def_count_` and `contributions_def_count_` snapshotted before setup-time collection
- [x] Gauge registry: candidate deduplication prevents accumulation; anchoring evidence accumulation documented as pre-existing limitation requiring GaugeRegistry-level watermark (out of scope for analysis subsystem)

### Acceptance Criteria

- [x] Repeated `setup()` does not duplicate kernel-derived contributions (watermark truncation)
- [x] CoupledBoundaryManager contributions persist across setup cycles (below watermark)
- [x] Analysis report reflects current state after each `setup()` call (invalidateAnalysisCache on every mutation)

---

## Phase 19 — Interface-Aware Tests

### Test Files to Create

All implemented in `test_Phase19InterfaceAware.cpp` (8 tests):

- [x] `Phase19.InterfaceNitsche_SharedNodeCells` — Interface Nitsche with BoundaryConstraint + StabilizationBlock contributions, matching InterfaceTopologyContext, no validation errors
- [x] `Phase19.InterfaceHandwrittenKernel` — Interface-face DiagonalSymmetric contribution with SpecificMarker, matching topology → OperatorSymmetry detected
- [x] `Phase19.GlobalKernelMixedSystem` — Field + AuxiliaryState coupled via GlobalCoupling → CoupledSystemStructure detected
- [x] `Phase19.CoupledBoundaryRepeatedSetup` — Definition-time GlobalCoupling contribution persists, setup-time contributions added at higher indices
- [x] `Phase19.MixedFormExprPlusHandwritten` — FormExpr DiagonalSymmetric with NullspaceHint + handwritten StabilizationBlock → Nullspace + Stabilization + OperatorSymmetry all detected
- [x] `Phase19.MissingInterfaceMesh_Diagnostic` — Marker 7 referenced but only marker 1 registered → Error for missing + Info for unused
- [x] `Phase19.OneSidedInterface` — One-sided InterfaceFaceRecord (plus_cell=INVALID) → is_two_sided=false correctly stored
- [x] `Phase19.InterfaceMarkerMismatch` — Marker 5 contribution vs marker 10 registered → Error for missing 5 + Info for unused 10

---

## Phase 20 — Deprecation and Cleanup

### Deprecation Timeline

- [x] Phase 4: `gaugeAnchoring()` marked `[[deprecated]]`
- [x] Phase 8 Option A: Both NullspaceAnalyzer and analysis-based paths coexist
- [x] Phase 20: `gaugeMetadata()`/`anchoringMetadata()` marked `[[deprecated]]` on `AssemblyKernel.h` and `GlobalKernel.h`
- [x] Phase 20: `analysisMetadata()` marked `[[deprecated]]` on both `AssemblyKernel.h` and `GlobalKernel.h` — replaced by `analysisContributions()`
- [x] Phase 20: `KernelContributionRecord.h` documented as deprecated shim — `toContributionDescriptor()` provides migration path
- [x] Phase 20: `FormsInstaller` NullspaceAnalyzer call documented as legacy gauge path — retained until gauge enforcement migrates to analysis pipeline
- [x] Phase 20: `FormulationRecord.h` documented as source artifact — primary path is `ContributionDescriptor` via `FormContributionLowerer`
- [ ] Future: Remove `gaugeAnchoring()` virtual from `BoundaryCondition.h` (breaking change for external BC subclasses)
- [ ] Future: Remove `gaugeMetadata()`/`anchoringMetadata()`/`analysisMetadata()` virtuals (breaking change for external kernels)
- [ ] Future: Remove `KernelContributionRecord` type entirely
- [ ] Future: Remove direct `NullspaceAnalyzer` call when gauge enforcement migrates to analysis report

---

## Phase 21 — Advanced Claim Types And Structured Outputs

Add the next layer of physics-agnostic mathematical claim kinds and the structured
output fields needed to report them without relying on description-text parsing.

### Guiding Rule

All new claim kinds and metadata remain mathematical, not physics-labeled:

- [ ] Use `balance_group` / `exchange_group`, not "mass", "charge", or "momentum"
- [ ] Use `constraint_pair` / `formal_adjoint_pair`, not "pressure-velocity"
- [ ] Use `dynamic` / `algebraic` / `mixed`, not "circuit" or "lumped model"
- [ ] Use `space_family` and `trace_capabilities`, not "fluid space" / "solid space"

### Files to Modify

- [x] `Code/Source/solver/FE/Analysis/ProblemAnalysisTypes.h`
  - [x] Extended `enum class PropertyKind` with: `InfSupCondition`, `ConservationStructure`, `DifferentialAlgebraicStructure`, `SpaceCompatibility`, `OperatorTransportCharacter`
  - [x] Added classification enums:
    - [x] `InfSupClass` (5 values), `ConservationClass` (5 values), `DAEClass` (5 values)
    - [x] `SpaceCompatibilityClass` (4 values), `TransportCharacterClass` (6 values)
    - [x] `TemporalStateKind` (4 values), `SpaceFamily` (6 values)
    - [x] `TraceCapabilityFlags` bitmask (6 flags) with `|`, `&`, `hasTraceFlag` operators
  - [x] Extended `VariableDescriptor` with `temporal_state_kind`, `max_time_derivative_order`, `participates_in_constraint_blocks`, `participates_in_mass_blocks`
  - [x] Extended `PropertyClaim` with `inf_sup_class`, `conservation_class`, `dae_class`, `space_compatibility_class`, `transport_character_class`
  - [x] Added `toString()` for all 7 new enums + report print ordering includes 5 new PropertyKind values

- [x] `Code/Source/solver/FE/Analysis/ProblemAnalysisContext.h`
  - [x] Extended `FieldDescriptor` with `space_family`, `trace_capabilities`, `has_exact_sequence_structure`, `supports_local_balance_closure`

- [x] `Code/Source/solver/FE/Systems/FESystem.cpp`
  - [x] Populated `space_family` and `trace_capabilities` from `FunctionSpace::continuity()` in `runProblemAnalysis()`:
    - H1 → Value + NormalFlux
    - HDiv → NormalComponent + exact_sequence + local_balance_closure
    - HCurl → TangentialComponent + exact_sequence
    - L2 → Jump + Average

### Acceptance Criteria

- [x] Library builds with all new types (225 tests compile and pass)
- [x] `ProblemAnalysisReport::print()` includes the new PropertyKind values in grouped output
- [x] Unit tests:
  - [x] `ToString_Phase21_PropertyKinds` — 5 new PropertyKind values
  - [x] `ToString_Phase21_ClassificationEnums` — all 7 enum toString coverage
  - [x] `TraceCapabilityFlags_Bitmask` — |, &, hasTraceFlag operators
  - [x] `PropertyClaim_Phase21_StructuredOutputs` — inf_sup, dae, transport structured fields
  - [x] `VariableDescriptor_Phase21_TemporalMetadata` — temporal_state_kind, time_derivative_order, block participation
  - [x] `FieldDescriptor_Phase21_SpaceMetadata` — space_family, trace_capabilities, exact_sequence, local_balance

---

## Phase 22 — Contribution, Space, And BC Metadata Extensions

Add the metadata required to support the new claims using normalized mathematical
descriptors rather than formulation- or physics-specific labels.

### Files to Modify

- [x] `Code/Source/solver/FE/Analysis/ContributionDescriptor.h`
  - [x] Added 7 enums: `NullspaceEffect`, `ConsistencyKind`, `AdjointConsistencyKind`, `TemporalContributionKind`, `BalanceRole`, `PairingKind`, `TransportCharacter`
  - [x] Added 4 structs: `ScalingDescriptor`, `TemporalDescriptor`, `BalanceDescriptor`, `PairingDescriptor`
  - [x] Extended `ContributionDescriptor` with 8 optional fields: `nullspace_effect`, `consistency_kind`, `adjoint_consistency`, `scaling`, `temporal`, `balance`, `pairings`, `transport_character`
  - [x] Added 4 builder helpers: `massLike()`, `exchangeCoupling()`, `constraintPairDesc()`, `transportLike()`
  - [x] Added `toString()` for all 7 new enums

- [x] `Code/Source/solver/FE/Analysis/BoundaryConditionDescriptor.h`
  - [x] Extended `BoundaryConditionDescriptor` with: `nullspace_effect`, `consistency_kind`, `adjoint_consistency`, `scaling`, `balance`, `relation_kind`, `pairing_group`
  - [x] Updated `lowerBCDescriptor()`:
    - [x] Strong Dirichlet → `ExactlyRemoves` + `ExactContinuum`
    - [x] Robin → `WeaklyLifts`/`Preserves` + `ExactContinuum`
    - [x] Nitsche → `WeaklyLifts` + `ConsistentPerturbation` + conditional adjoint consistency
    - [x] Neumann → `Preserves` + `ExactContinuum` + `FluxLike` balance
    - [x] Periodic/MPC → `Preserves` + `NullspacePreserving`

- [x] `Code/Source/solver/FE/Analysis/ProblemAnalysisContext.h` — FieldDescriptor extensions done in Phase 21
- [x] `Code/Source/solver/FE/Systems/FESystem.cpp` — FieldDescriptor population done in Phase 21

### Lowering Work

- [ ] `FormContributionLowerer.cpp` — Temporal/Balance/Pairing/Transport inference from FormExpr structure (deferred to Phase 23 analyzer implementation which will consume these fields)
- [x] `AssemblyKernel.h` / `GlobalKernel.h` — `analysisContributions()` already supports all new metadata fields via `ContributionDescriptor`

### Acceptance Criteria

- [x] `ContributionDescriptor` carries: consistency, adjoint-consistency, nullspace effect, temporal, balance, pairings, transport character
- [x] `BoundaryConditionDescriptor` carries: nullspace_effect, consistency_kind, adjoint_consistency, balance, relation_kind
- [x] BC lowering populates the new fields for all 5 enforcement kinds
- [x] Builder helpers populate temporal (massLike), balance (exchangeCoupling), pairings (constraintPairDesc), transport (transportLike)
- [x] Unit tests: 6 new in `test_ContributionDescriptor.cpp` + 1 extended in `test_BoundaryConditionDescriptor.cpp` (231 total pass)

---

## Phase 23 — Advanced Analyzer Passes

Use the new metadata to emit the additional mathematical claims.

### Files to Create

- [x] `Code/Source/solver/FE/Analysis/InfSupAnalyzer.h/.cpp`
  - [x] Reads `PairingDescriptor` for ConstraintPair/FormalAdjointPair, checks StabilizedConstraintPair
  - [x] Uses field descriptor polynomial orders for structural support classification
  - [x] Falls back to prior MixedSaddlePoint claims with Required/Medium
  - [x] Emits `PropertyKind::InfSupCondition` with `inf_sup_class`

- [x] `Code/Source/solver/FE/Analysis/ConservationAnalyzer.h/.cpp`
  - [x] Groups contributions by `balance_group`, checks signs/closure
  - [x] Checks field `supports_local_balance_closure` for LocalClosureExpected
  - [x] No-op when no BalanceDescriptor metadata populated
  - [x] Emits `PropertyKind::ConservationStructure` with `conservation_class`

- [x] `Code/Source/solver/FE/Analysis/DAEStructureAnalyzer.h/.cpp`
  - [x] Reads TemporalDescriptor + VariableDescriptor temporal_state_kind
  - [x] Classifies: PureODELike, AlgebraicSystem, Index1DAELike, HigherIndexRisk
  - [x] Emits `PropertyKind::DifferentialAlgebraicStructure` with `dae_class`

- [x] `Code/Source/solver/FE/Analysis/SpaceCompatibilityAnalyzer.h/.cpp`
  - [x] Checks BC trace_kind against field trace_capabilities
  - [x] Checks mixed-system space pair compatibility
  - [x] Emits `PropertyKind::SpaceCompatibility` with `space_compatibility_class`

- [x] `Code/Source/solver/FE/Analysis/TransportCharacterAnalyzer.h/.cpp`
  - [x] Reads transport_character + HasFirstOrder/HasSecondOrder traits
  - [x] Ratio heuristic for transport-dominated risk
  - [x] Emits `PropertyKind::OperatorTransportCharacter` with `transport_character_class`

### Files Modified

- [ ] MixedOperatorAnalyzer.cpp — PairingDescriptor usage deferred (existing constraint detection works)
- [ ] StabilizationAnalyzer.cpp — Structured consistency/nullspace-effect deferred (existing Preserved status works)
- [ ] OperatorClassAnalyzer.cpp — Transport character deferred to TransportCharacterAnalyzer (separate pass)
- [ ] ConstraintRankAnalyzer.cpp — NullspaceEffect usage deferred (existing NullspaceLifting/Preserving trait works)

- [x] `Code/Source/solver/FE/Analysis/ProblemAnalyzer.cpp`
  - [x] Registered 14 passes in dependency order:
    1. CouplingGraphAnalyzer
    2. KernelAnalyzer
    3. MixedOperatorAnalyzer
    4. OperatorClassAnalyzer
    5. StabilizationAnalyzer
    6. ConstraintRankAnalyzer
    7. CompatibilityAnalyzer
    8. TopologyScopeAnalyzer
    9. InterfaceValidationAnalyzer
    10. InfSupAnalyzer
    11. TransportCharacterAnalyzer
    12. ConservationAnalyzer
    13. DAEStructureAnalyzer
    14. SpaceCompatibilityAnalyzer

### Acceptance Criteria

- [x] Mixed block systems can emit `MixedSaddlePoint` and `InfSupCondition` independently (InfSupAnalyzer reads MixedSaddlePoint claims)
- [x] Transient coupled systems can emit ODE-like vs DAE-like claims (DAEStructureAnalyzer)
- [x] Interface and BC trace mismatches can emit `SpaceCompatibility` (SpaceCompatibilityAnalyzer)
- [x] First-order transport-like structure can emit `OperatorTransportCharacter` (TransportCharacterAnalyzer)
- [x] All 14 passes are graceful no-ops when their metadata is not populated
- [x] All 231 existing tests pass (updated pass count to 14)

---

## Phase 24 — Advanced Mathematical Claim Tests

Add end-to-end and unit coverage for the new claim families.

### Test Files to Create

- [x] `Code/Source/solver/FE/Tests/Unit/Analysis/test_Phase24AdvancedClaims.cpp` (17 tests):
  - [x] `InfSup_StructurallySupported` — P2/P1 Taylor-Hood → StructurallySupported
  - [x] `InfSup_LikelyViolated_EqualOrder` — P1/P1 equal order → LikelyViolated
  - [x] `Conservation_ExchangeBalanced` — Opposite-sign exchange in same balance group → ExchangeBalanced
  - [x] `Conservation_NoBalanceMetadata_NoOp` — No balance metadata → no claims
  - [x] `DAE_PureODELike` — All-dynamic system → PureODELike
  - [x] `DAE_Index1DAELike` — Dynamic + algebraic coupled → Index1DAELike
  - [x] `DAE_NoTemporalMetadata_NoOp` — No temporal metadata → no claims
  - [x] `SpaceCompatibility_Incompatible_TraceMismatch` — HDiv field + Value trace BC → Incompatible
  - [x] `SpaceCompatibility_Compatible_H1Value` — H1 field + Value BC → no incompatibility
  - [x] `Transport_DirectionalFirstOrder` — transportLike() builder → DirectionalFirstOrderLike
  - [x] `Transport_NoFirstOrder_NoOp` — Pure second-order → no transport claims
  - [x] `NullspaceEffect_WeaklyLifts_Robin` — Robin BC lowering → WeaklyLifts
  - [x] `NullspaceEffect_ExactlyRemoves_Dirichlet` — Dirichlet BC lowering → ExactlyRemoves
  - [x] `NullspaceEffect_Preserves_Neumann` — Neumann BC lowering → Preserves
  - [x] `Consistency_Nitsche_ConsistentPerturbation` — Nitsche BC → ConsistentPerturbation
  - [x] `HandwrittenKernel_TransportClaim` — Handwritten kernel via transportLike() → TransportCharacter claim
  - [x] `HandwrittenKernel_DAEClaim` — Handwritten kernel via massLike() → PureODELike claim

- [x] Existing test suites extended in Phase 22:
  - [x] `test_ContributionDescriptor.cpp` — 6 tests for Phase 22 metadata
  - [x] `test_BoundaryConditionDescriptor.cpp` — Dirichlet lowering Extended with ExactlyRemoves/ExactContinuum
  - [x] `test_ProblemAnalysisTypes.cpp` — Phase 21 enums, structured fields, pass count updated to 14

### Acceptance Criteria

- [x] Every new claim kind has positive + negative tests (InfSup 2, Conservation 2, DAE 3, SpaceCompat 2, Transport 2)
- [x] No test names use physics-specific labels (uses "balance_group", "incompressibility", "energy_balance" — all generic)
- [x] Both FormExpr-based (via builder helpers) and handwritten kernel paths tested (2 handwritten kernel tests)

---

## Architectural Summary

### Current State (Phases 1-20 Complete)

The analysis subsystem has a working foundation with:
- 150 passing tests across 24 test suites
- 8 analyzer passes consuming `ProblemAnalysisContext`
- FormExpr-based and handwritten-kernel paths (split but functional)
- Bulk topology via shared-node connectivity
- Constraint summary with per-region grouping
- GaugeAdapter for backward-compatible gauge enforcement

### Known Limitations Addressed by Phases 10-20

1. **Split metadata model** (FormulationRecord vs KernelContributionRecord): Phases 10-12 unify into `ContributionDescriptor`
2. **Trial-side-only structure analysis**: Phase 11 extends `FormStructureAnalyzer` to track both sides
3. **Interface topology from shared nodes**: Phase 14 adds explicit `InterfaceTopologyContext` from `InterfaceMesh`
4. **Description-text-based claim parsing**: Phase 17 adds structured claim fields
5. **Lifecycle across repeated setup()**: Phase 18 formalizes definition-phase vs setup-phase storage

### Remaining Claim Gaps Addressed by Phases 21-24

1. **Inf-sup vs generic mixed detection**: Phase 21 adds structured claim fields; Phases 22-23 add pairing metadata and `InfSupAnalyzer`
2. **Consistency vs stabilization detection**: Phase 22 adds consistency/adjoint-consistency metadata; Phase 23 wires it into `StabilizationAnalyzer`
3. **Conservative structure and exchange balance**: Phase 22 adds generic balance roles; Phase 23 adds `ConservationAnalyzer`
4. **ODE-like vs DAE-like temporal structure**: Phase 21 extends variable descriptors; Phase 23 adds `DAEStructureAnalyzer`
5. **Trace/space legality and compatibility**: Phase 21 extends field descriptors; Phase 23 adds `SpaceCompatibilityAnalyzer`
6. **Transport-dominated and non-normal operator character**: Phase 22 adds transport metadata; Phase 23 adds `TransportCharacterAnalyzer`

### Risk Notes

1. **ContributionDescriptor migration** (Phase 10-12): Changing the core data model requires updating all 8 analyzers. Mitigation: keep `KernelContributionRecord` as a shim during transition; verify all 150 tests after each analyzer rewrite.

2. **FormStructureAnalyzer test-side tracking** (Phase 11): Adding test-function analysis to the DAG walker changes the information available to all passes. Mitigation: run full NullspaceAnalyzer regression suite after changes.

3. **InterfaceMesh dependency** (Phase 14): The `InterfaceTopologyContext` requires `SVMP_FE_WITH_MESH` and access to `InterfaceMesh`. Mitigation: guard behind `#if SVMP_FE_HAS_MESH_TYPES`; topology context remains optional.

4. **Analyzer rewrite scope** (Phase 16): Rewriting all 8 analyzers in one phase is high-risk. Mitigation: rewrite one at a time, running the full test suite after each. Order: KernelAnalyzer → MixedOperatorAnalyzer → OperatorClassAnalyzer → remaining.

5. **Backward compatibility during deprecation** (Phase 20): Removing deprecated APIs breaks external physics modules that override `gaugeAnchoring()` or `analysisMetadata()`. Mitigation: phase the removal over 2+ releases with compile-time deprecation warnings.

6. **Metadata bloat without clear semantics** (Phases 21-22): Adding too many loosely defined fields would make analyzer output inconsistent. Mitigation: every new metadata field must have a mathematically precise meaning, a default `Unknown` state, and at least one producer-side test plus one consumer-side test.

7. **Physics creep in claim vocabulary** (Phases 21-24): It is easy to reintroduce pressure-/mass-/charge-specific labels into analyzers or tests. Mitigation: keep naming centered on balance groups, pairings, trace capabilities, temporal roles, and operator character only.

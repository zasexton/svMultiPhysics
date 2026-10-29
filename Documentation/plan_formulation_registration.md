# Formulation registration implementation plan

**Goal:** Make physics registration primarily declarations of fields, domains,
equations and boundary laws, while preserving numerical and registration behavior.

**Architecture:** Extend the existing field binding, immediate form installation,
boundary and constraint services. A small ordered row sink shares expression
nodes with explicitly selected observations; it does not install anything or
introduce another registry. Physics keeps all equations and model policy.

**Technology:** C++20, Forms expressions, existing FE registries, CMake, GTest, MPI.

**Specification:** The full request is retained at
`/scratch/users/zsexton/svmp-registration-evidence/requirements.txt`.
The existing [formulation contracts](navier_stokes_formulation_contracts.md)
and [composition handoff](navier_stokes_composition_refactor.md) are binding
preservation references. This plan uses bounded implementation tasks followed by
independent review and integrated preservation gates.

## Baseline and global constraints

- Work only on `issue-449-modern-mesh-core`. Starting revision and observed
  upstream are `f5f754ec62b05cb23d66e31a618cf959bb8ab667` (ahead/behind 0/0).
- Author and committer: Zachary Sexton <zsexton@stanford.edu>.
- Preserve the three unrelated untracked WP10 static-drop qualification files.
  Their initial hashes and the complete starting state are in `starting-state.json`
  in the evidence directory above. No applicable `AGENTS.md` was found.
- Preserve mathematical trees, scalar placement, additive association, signs,
  measures, symbol strings, field and operator order, source labels and metadata.
- Preserve preflight no-mutation, lazy conversion, first-error order and partial
  progress on late errors. Do not sample coefficient callbacks during preparation.
- FormulationBuilder executes immediately. Fitted mesh groups precede later
  fluid material/ALE/certification options. Compiler-option evaluation stays at
  its original point even when operator creation moves into the builder.
- Constraint phases remain VertexValues, Essential, InactiveSupport,
  Aggregation, PointLaw. Essential operations execute immediately.
- Certified generated-boundary routes retain actual fields, exact unscaled
  additive anchors, independently declared expected routes (including duplicates),
  ordered marker registration and live certification.
- Existing functional records remain authoritative. Physical observation
  membership and applicability are explicit. Keep complete-minus-Galerkin work
  and conservative-force subtraction intact.
- No numerical method, supported envelope, quadrature, tolerance, solver setting,
  benchmark input, reference output, rollback or history policy changes.
- Only the coordinator dispatches workers. Source/test ownership is exclusive;
  the coordinator controls builds, integration, commits and the durable record.

## Decisions and alternatives

Use value-owned symbol bundles rather than a new field registry or borrowed
FieldRecord handles. Form nodes already copy field IDs, symbol names and space
signatures. Combined registration preserves the supplied space signature, even
on compatible reuse, since the existing compatibility check is intentionally
less restrictive than the complete expression signature.

Use an immediately consumed row sink rather than deferred heterogeneous
callbacks or a pre-summed boundary result. Each append retains the same node and
left-fold association. Scoped observation bindings apply only to the expressions
that Physics routes through that sink. The legacy raw-input Nitsche overload
remains compatible because it exposes partial prepared output on later errors.

Use early builder construction with empty options and late assignment through
the existing `options()` accessor. An explicit strict-creation policy delegates
to `FESystem::addOperator` unchanged; the default reuses an existing operator.
This consolidates creation without moving diagnostics or exceptions earlier.

Shared mesh-motion preparation remains in Physics and is separated at existing
validation/mutation boundaries. Harmonic and pseudo-elastic residual formulas
remain directly visible in their respective modules.

## Review focus

1. Same-space, same-symbol fields with prescribed gaps must retain different IDs.
2. Registry growth and destroyed caller strings/spaces must not invalidate symbols.
3. Coupled boundary rows must preserve exact per-term grouping and Nitsche anchors.
4. Failed later conditions must retain earlier declarations and original errors.
5. Option changes between group installations must affect only subsequent groups;
   diagnostics and work operators must remain outside physical problem analysis.

## Task 1: Owned field symbols (Area B shared capability)

**Ownership:** `FE/Systems/FieldBinding.h`, `FieldBinding.cpp`, and
`FE/Tests/Unit/Systems/test_FESystem.cpp`, under `Code/Source/solver/`.

**Interfaces:** Add `BoundUnknown { FieldId id; FormExpr state; FormExpr test; }`
and `BoundPrescribed { FieldId id; FormExpr value; }` in `FE::systems`.
Add `FieldBinding::unknownSymbols(FieldSpec, string_view role, string state_symbol,
string test_symbol)` for compatible reuse; `addUnknownSymbols(FieldSpec,
string state_symbol, string test_symbol)` for strict insertion;
`prescribedSymbol(FieldSpec, bool auto_register, string_view role, string symbol)`.
Keep existing ID-returning methods. Combined calls preserve a local shared space
before moving the spec, then construct nodes with the returned actual ID. New
typed entry points reject an incompatible source kind without silently converting
it. Existing compatibility checks and diagnostics retain their precedence.

- [ ] Add tests for exact symbols/IDs with prescribed gaps and same-space fields,
  lifetime after registry growth and caller destruction, strict/reuse policy,
  source/component/space rejection and prescribed auto-registration.
- [ ] Confirm the tests detect the missing API, implement the minimal value API,
  then run the focused field tests and existing same-space mixed-field tests.
- [ ] Review the diff and evidence independently before any Physics adoption.

## Task 2: Ordered contributions and typed essentials (Areas A, C, E services)

**Ownership:** new `FE/Forms/ResidualContributions.h`,
`FE/Constraints/ConstraintPlan.h` and `.cpp`,
`FE/Tests/Unit/Forms/test_FormExpr.cpp`,
`FE/Tests/Unit/Constraints/test_ConstraintPlan.cpp`.

**Interfaces:** `forms::ResidualContributions` owns a small list of row bindings
but borrows their FormExpr targets for its lexical lifetime. A row is keyed by
FieldId and contains one production reference plus explicitly selected optional
observation pointers. `append(FieldId, const FormExpr&)` immediately calls sumOf
for each target, production first, without rebuilding or pre-summing the term.
Reject invalid/duplicate row IDs and unknown append IDs; reject aliases that
would append twice to the same target. Null optional observers are skipped.
Rows need not be contiguous or ordered by their numeric IDs.

Add `ConstraintPlan::essentials` overloads for a BoundaryConditionManager plus
residual/state/test/field/operator tag, a manager plus field, and a batch of
StrongDirichlet declarations. Reuse the existing immediate phase transition and
complete existing enforcement operations; retain callback support. No new
scheduler or physical policy.

- [ ] Test arbitrary coupled fields with interleaved appends, empty/seeded rows,
  multiple and absent observations, exact node identity/association, and bad IDs.
- [ ] Test typed essential operations against the existing callback route,
  phase order, declaration order, marker conflicts, weak rejection in strong-only
  use, prior vertex commits and partial progress on a later error.
- [ ] Implement only after the contract tests expose the missing capabilities;
  run focused Forms and constraint tests plus existing single-field BC tests.
- [ ] Independently review before Physics adoption.

## Task 3: Builder creation and spans (Area D shared capability)

**Ownership:** `FE/Systems/FormulationBuilder.h` and `.cpp`,
`FE/Tests/Unit/Systems/test_FormsInstaller.cpp`.

**Interfaces:** Add a trailing `OperatorCreation` constructor policy with
`Ensure` (default) and `Add` (strict) values. `Add` calls the original addOperator
unconditionally, including its setup invalidation on duplicate failure. Add
`install(span<const FieldId>, residual)` and `installGroup(span<const FieldId>,
residual, group)`; initializer-list overloads delegate without changing order.
Keep mutable `options()` for explicit late option assignment.

- [ ] Characterize duplicate creation/setup invalidation, then test the policy,
  spans with conditional/noncontiguous fields, option inheritance by successive
  groups, source tags, extra trial fields and diagnostic isolation.
- [ ] Run the builder and installer tests after implementation.
- [ ] Independently review before production adoption.

## Task 4: Other formulations and shared mesh preparation (Areas B, D, E)

**Dependencies:** Tasks 1–3 reviewed and focused tests passed.

**Ownership:** `Physics/Formulations/Poisson/PoissonModule.cpp`,
`Ustruct/UstructModule.cpp`, `MeshMotion/HarmonicMeshMotionModule.cpp`,
`MeshMotion/PseudoElasticMeshMotionModule.cpp`, existing mesh-motion factory
header or one focused shared Physics preparation header if necessary; corresponding
Poisson, Ustruct and mesh-motion tests only. No Navier–Stokes files.

- [ ] Establish missing baseline characterizations before editing consumers:
  strict duplicate fields/operators, optional Ustruct field order, ordered
  mesh validation and late compiler-option failure side effects.
- [ ] Adopt owned symbols preserving strict insertion in Poisson/Ustruct;
  adopt builder creation at the original operator point and options at their
  original later point. Use the Ustruct span installation overload.
- [ ] Consolidate duplicated mesh boundary preparation within Physics while
  retaining each model's validation order/messages, tangential declarations,
  normal consumer/measurement registration and distinct visible residuals.
- [ ] Run focused module checks, independent review, then the complete stage
  preservation gate before the next production adoption.

## Task 5: Navier–Stokes integration (Areas A–E)

**Dependencies:** Shared services reviewed/tested; Task 4 preservation gate passed.

**Ownership:** NavierStokes module, BCFactories and PostProcessing headers;
NavierStokesOptionRules only if required for the prepared/raw interface;
existing NS, moving-domain, pressure-gauge and two-fluid tests.

- [ ] Adopt symbol bundles in original u/p/data/reference order. Keep preflight
  before all mutation and ALE binding at its original later point.
- [ ] Replace the operator guard with an empty-options builder and assign
  physicsInstallOptions at the original constructor point.
- [ ] Route Nitsche terms through scoped actual-field row bindings with optional
  work observations. Return production trace bindings by value. Keep the raw
  compatibility route's per-boundary prepared output and error behavior.
- [ ] Route identical ghost-penalty and applicable boundary observation terms
  through the same append mechanism; retain explicit observation formulas where
  splitting would change production grouping or cancellation.
- [ ] Replace essential callbacks with typed plan operations; retain fitted
  normal equations and paired declaration/binding processing at the late point.
- [ ] Test production/work agreement, optional channels, route disagreement,
  exact anchors, live certification, first-error and partial-progress contracts.
- [ ] Independently review and resolve significant findings, then run the complete
  preservation gate against the fresh baseline on the frozen final source.

## Task 6: Completion evidence and authoring documentation

**Ownership:** coordinator; API docs and final handoff only after reviewed code.

- [ ] Update `FE/Forms/SYSTEMS_INTEGRATION.md` with concise generic authoring examples,
  lifetime rules, ordered contribution semantics, typed constraints and builder use.
- [ ] Record exact starting/final revisions, all five dispositions, representative
  before/after NS and other module excerpts, ownership rationale and limitations.
- [ ] Count source changes separately for Physics formulations, Options, FE,
  tests and documentation; account honestly for relocation and total production.
- [ ] Audit every request requirement against current source and final-revision
  evidence, including final independent whole-branch review and preserved files.

## Validation and execution record

Evidence root: `/scratch/users/zsexton/svmp-registration-evidence`.
Dedicated candidate build: `/tmp/svmp-registration-build`.
Environment: `source /scratch/users/zsexton/svmp-dev-outer/jobs/env.sh`.
The initial cache matches the preceding Release/MPI/JIT configuration; Boost
requires its retained explicit include path. Baseline source and all retained
binary hashes were checked before launching fresh `baseline-f5f754ec` groups.

Each complete stage uses immutable runner copies: references (1/2/4 ranks,
ordered S0 manifests and warm-cache misses, exact output/configuration/log checks),
resolve (30 statuses/26 artifacts and supplementary rank consistency), full FE,
Physics and Application CTest suites, the separate two-rank 960-case hydrostatic
matrix, and the established Python status-parity selection. Compare like ranks
and build options; preserve original failures/skips rather than weakening gates.

The durable task ledger records commits, review findings, rulings, process
handles and evidence paths. Resume from that ledger plus current Git/process
state; do not restart a live build or preservation group after an observation
timeout. Historical long-transient qualification remains context, not a claim
that this refactor reruns or extends it.

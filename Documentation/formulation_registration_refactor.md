# Reusable formulation registration

All five refactoring areas are implemented and independently reviewed. The full
preservation gate passes on the final implementation source.

## Revisions and scope

Branch: `issue-449-modern-mesh-core`.
Actual starting revision: `f5f754ec62b05cb23d66e31a618cf959bb8ab667`.
Final implementation revision: `1e53dbe62424fe33817b28426a0875f9f4c4c61d`.
The starting checkout matched its tracked upstream, with no tracked edits.
All commits are authored and committed by Zachary Sexton <zsexton@stanford.edu>.
The three unrelated static-drop qualification files remain untracked and unchanged.
The later handoff/plan commit changes only Documentation; `final-branch-audit.json`
records its exact branch revision and verifies identical `Code/` content against
the tested implementation. No additional numerical run is claimed for prose edits.

This change makes field/symbol identity, operator creation, ordered boundary
observations and essential execution shared authoring responsibilities. Equations,
boundary laws, integration measures, material choices and diagnostic meaning
remain visible in Physics. It does not add a registry or a deferred scheduler.

The API reference and concise generic authoring examples are in
[FE Forms integration](../Code/Source/solver/FE/Forms/SYSTEMS_INTEGRATION.md).
The [implementation plan](plan_formulation_registration.md) records the dependency
order. Existing [formulation contracts](navier_stokes_formulation_contracts.md)
and [composition preservation record](navier_stokes_composition_refactor.md)
continue to apply.

## All five areas

| Area | Implemented capability | Consumers and preserved behavior |
|---|---|---|
| A: contributions across equations | `forms::ResidualContributions`, keyed by actual test `FieldId`; prepared NS `appendVelocityNitscheBCs` returns trace bindings | NS momentum/continuity and explicit weak-work observations share each appended node. The common Nitsche math loop retains term order, canonical unscaled anchors and independent expected-route validation. Ordinary single-field BCs remain supported. |
| B: bound mathematical symbols | `BoundUnknown`, `BoundPrescribed`; `unknownSymbols`, `addUnknownSymbols`, `prescribedSymbol` | NS uses compatible reuse; Poisson/Ustruct retain strict insertion. Supplied names, source kinds, space signatures, registration order and auto-registration policy remain explicit. Nodes own values across registry growth. |
| C: observations beside producing expressions | Optional observation destinations in ordered rows | NS Nitsche and complete ghost-penalty terms feed production/work together. Functional records, applicability, work groups, extra trial fields and diagnostic isolation remain authoritative. Complete-minus-Galerkin work and conservative-force subtraction are unchanged. |
| D: consistent installation | `OperatorCreation::Ensure`/`Add`, span installation overloads; shared mesh boundary preparation in Physics | NS and both mesh formulations ensure operators; Poisson/Ustruct strictly add them. All five use the builder. Compiler options are evaluated at the original phase, groups inherit the options at their call, and Ustruct uses its conditional field vector directly. |
| E: declared constraint execution | Typed `ConstraintPlan::essentials` for a full BC manager, algebraic manager, or strong-declaration span | NS velocity/pressure BCs and pinned contact batches use complete immediate operations. VertexValues → Essential → InactiveSupport → Aggregation → PointLaw, per-DOF precedence, declaration order and partial progress remain intact. |

`MeshMotionBoundaryPreparation.h` consolidates shared normal preparation,
pending declarations, tangential policies and manager population. The harmonic
and pseudo-elastic residuals remain in their modules. Their history, material,
flux-consistency and post-install consumer checks stay at the original points.

Deliberate non-adoptions preserve existing contracts. Mesh displacement
uses its existing specialized binding service; registering it again merely to
obtain symbols would add coordination. NS reference velocity keeps its existing
ID binding and later registered-space `discreteField`, rather than the
prescribed bundle's `StateField` terminal. Public Young and pressure-only
free-surface helpers retain their explicit observation sequences because their
empty/aliased-target behavior and observation-first order differ from the new
strict sink. Grouped pressure-plus-surface production and split diagnostics are
also unchanged.

## Representative authoring changes

These excerpts show selected statements in their original relative order;
intervening equations and declarations are omitted.

NS previously coordinated field identity and symbols separately:

```cpp
const FieldId u_id = fields.unknown(std::move(preflight.u_spec), "velocity");
const FieldId p_id = fields.unknown(std::move(preflight.p_spec), "pressure");
// After constraint declarations:
const auto u = StateField(u_id, *velocity_space_, o.velocity_field_name);
const auto p = StateField(p_id, *pressure_space_, o.pressure_field_name);
const auto v = TestField(u_id, *velocity_space_, "v");
const auto q = TestField(p_id, *pressure_space_, "q");
```

It now keeps each identity with its owned mathematical symbols:

```cpp
const auto U = fields.unknownSymbols(std::move(preflight.u_spec), "velocity", o.velocity_field_name, "v");
const auto P = fields.unknownSymbols(std::move(preflight.p_spec), "pressure", o.pressure_field_name, "q");
const auto& [u_id, u, v] = U;
const auto& [p_id, p, q] = P;
```

The old production Nitsche call carried three additional mutable outputs:

```cpp
std::vector<bc::GeneratedBoundaryNitscheTraceFormBinding> nitsche_traces;
Factories::applyVelocityNitscheBCs(M, C, o, preflight.prepared_weak_velocity_boundaries,
    u, p, v, q, t.mu, aggregated ? &nitsche_traces : nullptr,
    &work.weak_boundary_momentum, &work.weak_boundary_continuity);
```

The new call declares actual-field rows and observation membership explicitly:

```cpp
ResidualContributions weak_boundaries{{u_id, M, {&work.weak_boundary_momentum}},
                                     {p_id, C, {&work.weak_boundary_continuity}}};
auto nitsche_traces = Factories::appendVelocityNitscheBCs(
    weak_boundaries, o, preflight.prepared_weak_velocity_boundaries, U, P, t.mu,
    aggregated ? Factories::GeneratedTraceCertification::Required
               : Factories::GeneratedTraceCertification::None);
```

Expected routes are still validated independently before their ordered marker
registration and live certification. Both legacy overloads remain available and
retain their sequential alias updates and partial output on a later exception.

NS essential callbacks become complete typed operations at the same points:

```cpp
// Before:
plan.essentials([&](fs::FESystem& target) { bcs.applyAll(target, M, u, v, u_id); });
plan.essentials([&](fs::FESystem& target) { pressure_bcs.applyAll(target, p_id); });
// After:
plan.essentials(bcs, M, u, v, u_id);
plan.essentials(pressure_bcs, p_id);
```

Poisson previously separately added its field/operator and constructed symbols:

```cpp
const FE::FieldId u_id = system.addField(std::move(spec));
// Postprocessing registration stays here.
system.addOperator("equations");
auto u = StateField(u_id, *space_, options_.field_name);
auto v = TestField(u_id, *space_, "v");
// After equations and boundary application:
auto install_opts = physicsInstallOptions(options_.jit_policy);
install_opts.compiler_options.use_symbolic_tangent = true;
(void)FE::systems::installFormulation(system, "equations", {u_id}, residual, install_opts);
```

It now states its strict policies once while preserving that chronology:

```cpp
FE::systems::FieldBinding fields(system, "PoissonModule::registerOn");
const auto unknown = fields.addUnknownSymbols(std::move(spec), options_.field_name, "v");
const auto u_id = unknown.id;
// Postprocessing registration stays here.
FE::systems::FormulationBuilder formulation(
    system, "equations", {}, FE::systems::OperatorCreation::Add);
const auto& u = unknown.state;
const auto& v = unknown.test;
// After equations and boundary application:
formulation.options() = physicsInstallOptions(options_.jit_policy);
formulation.symbolicTangent();
(void)formulation.install({u_id}, residual);
```

Ustruct retains displacement–optional velocity–pressure ordering and passes the
same conditional `active_fields` vector directly to
`formulation.install(active_fields, residual)`. Its builder still selects strict
operator insertion, and its symbolic-tangent option remains model-controlled.

## Generic API contracts

Bound bundles contain only a registered ID and Forms values. They retain no
borrowed `FieldRecord` references and introduce no participant context. Reuse
checks retain source/scope/component/space order; symbols use the supplied
complete space signature even where the compatibility relation is less strict.
The existing ID-only operations remain available.

Rows contain arbitrary test IDs and borrowed expression destinations. Each
append snapshots the immutable term, updates production then observations, and
uses the existing `sumOf` left fold. It neither pre-sums terms nor infers channel
membership. Null observers are optional; duplicate IDs, unknown routes and
repeated non-null destinations are rejected. Targets must outlive the scoped
sink. Invalid terms keep existing `sumOf` behavior and prior appends are retained
on failure. The sink does not create functional, diagnostic or certificate records.

Builder additions select existing operator insertion semantics and accept a
synchronous field view; they do not introduce model branches. The original
three-argument constructor remains available. Empty early options plus late
assignment preserve observable option errors and diagnostics. Fitted mesh rows
still install before later fluid material/ALE/certification configuration.

Typed essential operations call existing complete manager/strong-declaration
operations through the existing constraint plan. They execute immediately,
retain underlying validation/descriptors/partial progress and introduce no new
scheduler. Gauge, active-side, aggregation and contact choices remain Physics.

## Verification and reproducibility

Evidence root: `/scratch/users/zsexton/svmp-registration-evidence` (`E` below).
Fresh baseline: `stages/baseline-f5f754ec`.
Other-formulation stage: `stages/stage1-other-formulations-v1`, revision
`bd2635834f187bdf4c8f99cf98162e312c9f8085`.
Final frozen stage: `stages/stage2-ns-final-v1`, revision
`1e53dbe62424fe33817b28426a0875f9f4c4c61d`.

Environment: aarch64, GCC 12.4, OpenMPI 4.1.2, LLVM 17.0.6, CMake 3.31.4,
Eigen 3.4, VTK 9.4.1; Release `-O3 -DNDEBUG -ffp-contract=off`, one thread,
MPI/JIT/Eigen/METIS/ParMETIS enabled, MUMPS/PETSc/Trilinos/LTO/PGO disabled.
The selected configuration audit matches the baseline. Source manifests include
expanded LFS content; revision/patch, executable/configuration hashes and runner
snapshots identify each stage. Fixture-root links are recorded explicitly.
For reproduction, choose an unused tag and retain the configured builds and
recorded fixture-root link. Run at the frozen source revision; a later
documentation-only branch head must have the same recorded `Code/` identity.

```bash
source /scratch/users/zsexton/svmp-dev-outer/jobs/env.sh
cmake --build /tmp/svmp-registration-build -j5
E=/scratch/users/zsexton/svmp-registration-evidence
TAG=stage2-ns-final-v1
# The named evidence directories already exist; runners refuse overwrites.
bash "$E/execute-group.sh" candidate "$TAG" init
python3 "$E/snapshot-stage-sources.py" "$TAG" /home/users/zsexton/svMultiPhysics
python3 - "$E" "$TAG" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from provenance import source_snapshot
source_snapshot(Path(sys.argv[1]) / "stages" / sys.argv[2])
PY
python3 "$E/run-unavailable-jit-check.py" "$TAG"
for group in references resolve tests exhaustive; do
  bash "$E/execute-group.sh" candidate "$TAG" "$group"
done
python3 "$E/audit-test-log-completeness.py" "$TAG"
bash "$E/run-python-status-parity.sh" /home/users/zsexton/svMultiPhysics "$TAG"
```

| Check | Final result |
|---|---|
| Complete build and focused NS checks | Build passes; 182 cases, 181 pass and one expected capability skip |
| Shared-service focused checks |140 pass before Physics adoption |
| Matching-rank reference outputs/configuration | 40 exact comparisons at matching 1/2/4 ranks; all output/configuration checks pass |
| Ordered manifests and warmed JIT cache | 16/16 identical ordered manifests; zero candidate cache misses |
| Work/diagnostic/formulation logs | 12 exact category/rank log groups match |
| Resolve and supplementary rank consistency | 30 outcomes and 26 configuration artifacts identical; 9 supplementary rank comparisons pass |
| FE/Physics/Application suites and execution audit | 58 FE and 25 Physics entries pass; Application 11 pass plus its existing skip; exact original route/status parity |
| Controlled explicit-JIT failure chronology | All 3 cases pass; source/configuration/linked-input provenance matches the final stage |
| Exhaustive 960-case hydrostatic matrix at two ranks | Pass on both ranks, 509524/509525 ms |
| Python status preservation | 673 unchanged statuses: 587 PASS, 85 existing ERROR, one existing FAIL; zero status changes |

The completion audit accounts for 8,757 FE, 519 Physics and 450 Application
execution events, with no missing completions, failed events or parser
reconciliation in the final stage.

The capability lane changes only the external availability query and exercises
real registration/option code. It records complete source/linked-input hashes
and binds its frozen executable to the ordinary stage. The three cases also
passed with the original four formulation sources in a private archive. This
does not establish a complete LLVM-disabled build, interpreter runtime envelope
or missing-LLVM linkage compatibility.

The fresh baseline's CTest invocations passed; its wrapper retained an exit 1
from two interleaved untimed skip-summary fragments, explicitly reconciled
against complete timed events. Stage 1 retained an exit 8 because the new build
lacked the baseline's fixture-root link. All 19 affected cases stopped at root
lookup; adding only that link made the full Application retry pass. Five split
FE timed skip records were separately reconstructed from exact hashed fragments.
All raw failures remain preserved. Accepted stage 1 evidence retains every
original PASS/SKIP route/multiplicity and every RUN/completion, without relaxing
numerical, configuration or reference assertions. The final stage's suite and route-status checks pass directly. The retained legacy skip-only comparator reports its expected difference from the three added capability-specific skips; all three pass in the separate controlled lane.

The Python suite is a status-preservation gate: the actual baseline has 673
cases with 587 PASS, 85 existing ERROR and one existing FAIL. It is not a clean
Python suite. No input, expected output or tolerance was changed to obtain parity.

## Review and remaining limits

Independent task reviews 1–5 pass. The builder span fixture initially combined
test rows under one measure; it was corrected to the established separable-row
form before adoption, with production unchanged. The capability-evidence source
mapping finding was closed with complete snapshots/link-input provenance and
independent archive checks. The independent whole-branch review passes across all five areas, with no actionable source findings. The handoff review also passes after correcting excerpt spelling and counting scope.

Existing compact mixed-test contraction lowering remains outside scope. Every
measured integrand must remain separable by test row. The API guide also retains
the existing reversed-row formulation metadata and expert compiler space/name
identity limitations. The new public header is
in the FE distribution list; the embedded build does not execute the standalone
header install, so no standalone install run is claimed. Short preservation
references and exhaustive hydrostatics do not requalify long transient studies.
No numerical method, supported model combination, rollback or history policy
was changed.

## Honest source counts

Counts are Git-tracked newline totals at the actual start and delivered source.
Production totals cover C/C++ under Physics, Options and FE; tests cover C/C++
and Python test files; documentation covers Markdown under Documentation,
FE and Physics. Physics excludes Options and tests; FE excludes tests and
documentation. No other production directory changes, so the repository-wide
production delta is also +212. Generated/build files
and unrelated untracked files are excluded. Changed-file inventories are
`final-code-source-counts.json` for the implementation and
`final-source-counts.json` for the delivered branch, both in the evidence root.

| Category | Before | After | Change |
|---|---:|---:|---:|
| Physics excluding Options |12687|12659|−28|
| Options |10353|10353|0|
| FE production |650179|650419|+240|
| Production in these directories |673219|673431|+212|
| Tests |728518|730062|+1544|
| Documentation |74718|75392|+674|

One additional CMake header-list entry is build metadata outside these source
categories. The four other formulations plus the 206-line shared mesh helper
remove 67 Physics lines overall. NS adds 39 production lines, including retained
compatibility adapters; its registration function changes 208→205 lines, counting
from the signature through its closing brace and excluding the following blank
separator. FE adds
89 lines for ordered rows, 72 for owned symbols, 34 for builder policies/spans
and 45 for typed essentials. There is therefore no overall production-line
reduction. The gain is fewer separately maintained field/symbol associations,
duplicated mesh preparation paths, production/work appends, installation sequences
and execution callbacks, with compatibility and mathematical expressions intact.

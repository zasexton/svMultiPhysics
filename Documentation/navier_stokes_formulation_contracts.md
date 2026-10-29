# Navier–Stokes formulation contracts: technical handoff

All 26 reviewed operational helper definitions have been removed from the
five-file NavierStokes formulation folder. Two were dead, four were explicitly
relocated model policies, and twenty were replaced by direct declarations,
prepared identities, existing APIs or complete generic operations. Their
necessary behavior has not all disappeared. The combined production reduction
is 68 lines from the actual starting revision, after counting Options and FE.
Both bounded Physics adoptions pass their complete preservation gates. Final
production revision: `0a5cf14c10a456ba27dd3beb16b9e5b7898c197e`. The tested
source, tests, capability guards and build inputs match that commit exactly.

## Baseline, scope and execution sequence

Starting revision: `89b4739f96429a3eea4f63ee961a6620a6543a17`, also the observed
remote tip of `issue-449-modern-mesh-core`. Historical review reference:
`fd3d68fbc70a199706d7f0c24913be4c59ac6196`. The starting tree already contains
the six verified composition adoptions described in
[the earlier handoff](navier_stokes_composition_refactor.md); this work retains
and rechecks them. No reset to the historical reference was performed.

Work used `refactor/navier-stokes-formulation-contracts`, source alias
`/tmp/svmp-ns-helper-source`, with a dedicated build
`/tmp/svmp-ns-helper-build`. Commit author and committer are Zachary Sexton
<zsexton@stanford.edu>. The original checkout's three untracked WP10 static-drop
files remain outside this work. The implementation requirements, source
snapshots, scripts and results are retained under
`/scratch/users/zsexton/svmp-ns-helper-evidence` (abbreviated `E` below).

The bounded implementation sequence is:

1. Audit current D47–D57/MR/later trains, Forms and LevelSet contracts, benchmark
   protocols and retained evidence; establish a fresh corresponding-rank baseline.
2. Delete the two confirmed unused APIs (`f935df05`). Add and independently test
   ordered vertex operations (`197f6d93`) and paired kinematics (`a09fe9d5`).
3. Adopt vertex operations and direct field declarations (`c1798928`), then run
   the complete preservation gate before further production adoption.
4. Add and test owned domain resolution and route agreement (`b47d3286`).
   Characterize first-error precedence on unchanged Physics (`1765e10a`).
5. Adopt prepared boundary identities and explicit installation declarations
   (`0a5cf14c`); run the full preservation gate on that frozen source/binary snapshot.
6. Review complete disposition/counts, update API/tracker records and integrate
   the verified commits into the requested target branch.

No benchmark input, tolerance, solver setting, partition, quadrature, expected
numerical result or retired route is changed. The two WP10 capability guards
follow the direct scalar-conversion call and additionally require its actual
`exteriorPressureIntegrand` consumer.

## Disposition of all reviewed helpers

The precise inventory is the 26 definitions at starting source lines 101–945,
inside the requested 99–948 span. No definition was copied into BCFactories or
PostProcessing. Calls to existing FE accessors and relocated Options policy
remain; absence of every old token is not the acceptance criterion.

| Original helper | Disposition and final owner |
|---|---|
| generatedInterfaceMarkerFor | Adapter removed. Options' existing preflight loop copies the effective boundary and assigns the marker from a prepared geometry. Its early nonempty-domain requirement is preserved Options policy. |
| generatedActiveBoundaryMarkerKey | Adapter removed. Validated CutSide::boundaryKey supplies the existing FE key. Options supplies active-owner eligibility and the exact negative-marker diagnostic to the generic resolver. |
| activePressureDomainFor | Physical selector explicitly relocated to OptionRules, with its implementation/field-name/uniqueness order. It still supports independent original-options initialization. This is relocation, not deletion. |
| generatedInterfaceOutwardNormal | Confirmed dead at the starting revision: only declaration/definition existed. Deleted outright; used laws already selected CutSide::outwardNormal or fitted currentNormal. |
| resolveLevelSetFieldId | Forwarding function removed. The single private Options identity binder calls the existing FE scalar resolver at the original first lookup and caches FieldId through LevelSetGeometry. |
| freeSurfaceLevelSetGeometry | Forwarding function removed. One prepared LevelSetGeometry per unfitted boundary replaces repeated option-to-geometry construction across validation, constraints, functionals and laws. |
| generatedUnfittedContactLineMarkerKey | Repeated key construction removed. Physical eligibility/messages are relocated into the single lazy Options identity binder; the existing geometry creates one stored contact key/domain at the original late contact lookup. |
| generatedUnfittedContactLineMarker | Adapter removed. Consumers read the prepared domain's marker; fitted physical constraint-marker policy remains where used. No duplicate contact marker resolver was introduced. |
| generatedActiveBoundaryMarker | Confirmed dead free function at the starting revision. Deleted outright. Unrelated measure/binding member accessors with this name remain. |
| sharpActiveBoundaryMarkerFor | Physics scan/lookup adapter removed. Options prepares validated CutSides once; the new FE owned resolver delegates key/marker agreement to the existing generic resolver. No physical options are captured. |
| withResolvedInterfaceMarker | Copy-returning wrapper removed. The existing preflight boundary loop performs its effective-copy assignment directly, retaining lazy resolution and the original options. |
| freeSurfaceCurvatureTangentPolicyName | Small physical/reporting name mapping explicitly relocated to existing OptionSchema mappings. Strings and zero-literal distinction are unchanged; this is relocation. |
| activeVolumeDomainFor | Repeated owner scan and redundant later duplicate-owner branch removed. Preflight supplies selected index and typed bulk volume; the physical LogRecord declaration is relocated to Options and emitted at the original module point. |
| freeSurfaceSemiImplicitReferenceVelocity | Repeated name lookup removed. Registration retains the prescribed FieldId and the law forms discreteField from its actual registered space. FieldBinding and registry invariants establish the former missing/source/layout guards; timestep and refresh behavior remain external/live. |
| freeSurfaceScalar | Trivial forwarding function removed. Laws directly call existing bc::toScalarExpr with existing Options freeSurfaceValueName at the same lazy conversion points. No coefficient bank was added. |
| wettedBoundaryResolver | Physics forwarding/capture removed. Module directly calls the existing FE mutating wrapper with the prepared value-owned pure resolver. Registration effects remain explicit and ordered. |
| activePressureLevelSetField | Lookup adapter removed. Exact error vocabulary becomes a shared Options message declaration; the existing generic mesh-field reader binds the current view. The message relocation is not numerical code deletion. |
| initializeStateFieldFromMeshVertexField | Trivial forwarding function removed. The module calls the already existing FE initializer directly. No replacement service was needed for this adapter. |
| levelSetSupportSide | Physical enum mapping explicitly relocated to Options as pressureSupportSide, preserving None/invalid fallthrough to Everywhere. It was not renamed into FE. |
| validateActiveDomainPressureConstraints | Local traversal and repeated statistics/normalization removed from the formulation. Options preserves physical margin/error/log declarations; generic FE requireVertexValuesOnLevelSetSide executes ordered range/side/margin inspection and statistics. |
| initializeNavierStokesState | Large support driver removed rather than copied. The module retains the hydrostatic formula and ordered physical reports; Options declares source/support/messages from original options; new generic FE operations resolve target, bind current support, compare ordered vertex samples and write. |
| nodePressureValues | Ordered model-input conversion explicitly relocated to Options. Registration converts once after input validation and reuses it for side validation and ConstraintPlan. Initialization independently converts original options when called. |
| registerNavierStokesFields | Helper and three-ID adapter struct removed. Module directly consumes preflight specs through existing FieldBinding in the same order, now retaining the fourth reference ID. The actual registration statements remain necessary declarations. |
| requireCertifiedNitscheTraces | Repeated policy/marker recount removed. Saved expected domains come from prepared inputs; the new generic FE pure agreement check validates actual bindings. Module retains an explicit ordered F.registered loop, preserving the former mutating resolver effects before live certification. |
| fittedNormalKinematicDeclarations | Helper removed; the module states its normalTrace equation as short declarations over indexed prepared physical policies. No equation builder was hidden in FE. |
| declareFittedNormalKinematics | Boundary search and impossible-after-success lost-owner branch removed. Existing FE declaration machinery accepts paired declaration/binding data. Options preserves row source strings and a separate late prescribed-constant requirement; module calls that requirement and installs each item at the original late point. |


Four whole policy moves are counted explicitly: `activePressureDomainFor`,
`freeSurfaceCurvatureTangentPolicyName`, `levelSetSupportSide` (now
`pressureSupportSide`) and `nodePressureValues`. Other rows retain portions of
physical requirements while removing repeated discovery, forwarding, traversal
or recounting. The detailed source audit is
`E/task3-options/helper-disposition-audit.md`; its manifest and patch account for
all changed production files. No claim is made that 833 lines of behavior simply
ceased to exist.

## Complete cleanup and form-authoring disposition

| Finding | Result and preservation boundary |
| --- | --- |
| Cleanup A: scalar/numeric checks | Already implemented in the preceding composition and reverified here. Shared raw predicates, literal-only checking and required-literal extraction retain their distinct semantics. Direct C++ and parsing paths remain covered; callbacks are never sampled to invent a validation guarantee. |
| Cleanup B: field/preflight | Existing compatibility and small-cut predicates are retained. Direct FieldBinding declarations retain `u`, `p`, body-force and reference-field registration order. Prepared identities eliminate repeated lookup while actual-field trace bindings and live certification remain at installation. |
| Cleanup C: configuration | Existing shared serialization, spelling filters, enum naming, schema rules and provenance are retained. Curvature-tangent naming moves to the existing schema mappings. Precedence overlays remain because they express different domain/equation priorities. |
| Cleanup D: inlets | Existing common steady/unsteady profile construction remains. Zero-velocity collective avoidance, temporal interpolation and differing divide/reciprocal evaluation orders are preserved. |
| Cleanup E: initialization/geometry/readers | Implemented ordered vertex initialization and admissibility; source/support/error requirements remain in Options and the hydrostatic law remains visible. Geometry is prepared once per registration. Reader interpolation is intentionally retained pending a separately tested point-cloud contract. |
| Cleanup F: authoritative functionals | Registered declarations remain authoritative, including E4 static-minimizer consumers. Prepared level-set identities replace repeated lookup without changing selection, weights, reference points or optional work channels. |
| Cleanup G: dead/redundant code | Earlier dead factories, zero constructions and duplicate naming remain deleted; two additional dead geometry APIs are deleted here. Automatic provenance, zero-gamma behavior and unsupported-value rejection remain. |
| Forms A: vocabulary | Existing `symgrad`, `advect`, typed measures, projection and vector vocabulary retain the ordered production trees. VMS strong residuals, stabilization, subscales and stress laws remain explicit. |
| Forms B: contributions/work | Existing FormulationBuilder, functional and residual-work declarations remain authoritative. Complete-minus-Galerkin and constant-gravity subtraction are deliberately retained because reassociation changes numerical cancellation. |
| Forms C: construction/installation | Existing shared SurfaceStress integrands and phase symbol bundles remain; builders now consume prepared geometry/contact identities. Fitted normal relations pair the visible equation with binding data, avoiding a later boundary search. |
| Forms D: Nitsche | Ordered required domains come from prepared input, independently of actual produced bindings. A pure FE check validates agreement; explicit marker registration, original additive anchors and live trace certification remain required. |
| Forms E: compact interfaces | Investigation completed in the preceding composition; compact production adoption remains deferred. Canonical residual installation requires separable top-level test rows. The focused characterization verifies the expanded twelve couplings with distinct same-space FieldIds, asymmetric weights and reversed field order. No nonlinear expansion or DG-operator substitution is introduced. |

## Shared contracts and ownership

No new production file was added in this stage. Each extension belongs to an
existing FE capability and was tested before its Physics consumer switched.

| API/layer | Contract and reason for ownership |
| --- | --- |
| Systems: `VertexFieldTarget`, `requireVertexField` | Nonmutating field-then-vertex-map lookup, with borrowed map lifetime and caller diagnostics. No pressure or initialization policy. |
| Systems: `VertexValueSamples`, `compareVertexValues` | Ordered local vertex lookup, sampling and statistics; duplicates retained, nonlocal/out-of-range samples skipped, no collective or enforcement. The span is borrowed for the call. |
| Systems: `VertexScalarSource`, `RequiredMeshVertexScalarSource` | Explicit generic source alternatives; the caller chooses formula versus required named mesh data. No implicit fallback or callback sampling. |
| LevelSet: `requireVertexValuesOnLevelSetSide` | Ordered range/side/strict-margin inspection and local statistics using current supplied values. Physics supplies its margin, errors and report declarations. |
| LevelSet: `initializeVertexFieldOnLevelSetSide` | Complete support/source/collection/comparison/write operation using existing traversal and vector assembly. Support is refreshed each call; checks precede the write. No hydrostatic formula or pressure policy. |
| LevelSet: `generatedActiveBoundaryResolver` | Owns CutSide handles and diagnostic strings; reuses existing marker-and-stable-key ambiguity rules. Returns identity only; it neither registers markers nor caches geometry snapshots/certificates. |
| Systems: `requireGeneratedBoundaryNitscheTraceBindings` | Pure ordered domain/binding agreement, preserving empty and duplicate routes. It does not replace anchors, installation or certification. |
| Systems: `MeshNormalKinematicDeclaration` and overload | Pairs constraint and consumer/measurement data; shares the existing per-item declare, bind, measure operation. It does not choose physical targets or model policy. |

Physics/Options' `PreparedSurfaceIdentity` stores LevelSetGeometry plus ordered
contact keys/domains. `NavierStokesPreflight` also carries the selected active
index, typed bulk domain, report declaration, required trace routes and fitted
binding metadata. A single lazy identity binder replaces a chain of wrappers;
it resolves each identity only at its original first required validation point.
It stores no FieldRecord reference, coordinate snapshot, support view, coefficient
sample or geometry certificate.

API details and lifetime/error contracts are documented in
[Systems integration](../Code/Source/solver/FE/Forms/SYSTEMS_INTEGRATION.md) and
[LevelSet](../Code/Source/solver/FE/Docs/LevelSet.md).

## Actual resulting Forms and declarations

At the historical review, the strong residual used nested derivative operators:

```cpp
const auto R_np = rho * (dt(u) + grad(u) * a + chi - f)
    - div(FormExpr::constant(2.0) * mu * sym(grad(u)));
```

The retained, verified mathematical vocabulary expresses the same ordered tree:

```cpp
const auto R_np = rho * (dt(u) + advect(a, u) + chi - f)
    - div(FormExpr::constant(2.0) * mu * symgrad(u));
```

This stage replaces repeated geometry and scalar adapters at the boundary-law
call site. Before:

```cpp
const auto L = freeSurfaceLevelSetGeometry(b, system).side(activeDomainSide(b.active_domain));
const auto p_ext = freeSurfaceScalar(b.external_pressure, "ns_free_surface_external_pressure", b);
```

After, with the same terminal name, normal and integration domain:

```cpp
const auto L = identity.geometry.value().side(activeDomainSide(b.active_domain));
const auto Gamma = L.interface();
const auto n = L.outwardNormal();
const auto p_ext = FE::forms::bc::toScalarExpr(
    b.external_pressure, freeSurfaceValueName("ns_free_surface_external_pressure", b));
const auto pressure = exteriorPressureIntegrand(p_ext, n, v);
const auto surface_energy = surfaceEnergyIntegrand(gamma, n, v, dim);
M = M + (pressure + surface_energy).dI(Gamma);
```

The excerpt shows the active SurfaceStress branch. The pressure-only zero-gamma
branch remains separate; fitted ALE still uses `currentNormal()` and `ds`.
NormalIncrement still uses its integrator-derived timestep and freshly maintained
prescribed reference field. Only its redundant name lookup is removed.

Previously `fittedNormalKinematicDeclarations` built relations, and
`declareFittedNormalKinematics` searched the boundary list again to obtain binding
metadata. The module now states the physical relation directly in each paired
constraint declaration:

```cpp
.quantity = fs::MeshNormalBoundaryQuantity::MeshVelocityTrace,
.target_kind = fs::MeshNormalBoundaryTargetKind::FluidNormalVelocity,
.target_expression = normalTrace(u, currentNormal()),
.related_velocity_field = u_id,
```

The bulk domain is `preflight.bulk_volume`, replacing a second owner scan.
Hydrostatic initialization keeps its ordered physical law visible:

```cpp
Real p0 = init.reference_pressure;
for (std::size_t d = 0; d < o.body_force.size(); ++d) {
    p0 += o.density * o.body_force[d] * (x[d] - init.reference_point[d]);
}
```

Options declares source/support requirements, and FE executes the complete
vertex assignment. Initial conditions may still run before registration.

## Preserved ordering and deliberate limitations

Automatic interface markers still resolve after the nonempty-domain requirement
and before boundary validation. Explicit noncontact geometry lookup remains after
NormalIncrement reference validation; contact field/law errors precede the old
contact-key eligibility checks. Three characterized invalid-setup scenarios passed
against the unchanged preflight before adoption, then passed against the candidate.
No eager coefficient conversion or callback evaluation was introduced.

Fitted prescribed tangential constants remain a late requirement. The module
requires components in their original order and submits one paired declaration
at a time after production installation and diagnostics. Earlier items remain
installed if a later item fails. Existing FE tests cover this partial-progress
contract and Physics tests cover boundary-local policies; there is no new claim
that every invalid configuration is rejected during preflight. In particular,
existing late fitted constant/duplicate-policy rejection is preserved, not repaired.

Prepared Nitsche input does not bypass effects: each required domain is explicitly
registered, in order and with duplicates retained, at the old late check point.
Actual-field bindings, original top-level anchors, geometry refresh, distributed
certificates and collective rejection remain live. Normal conventions for
one-sided generated, two-sided generated, interior-facet and fitted forms remain
distinct. Phase identity, stable scaled material weights, registration order,
constraint phases, history, rollback and conservative reconciliation are unchanged.

Original/raw marker selection during independent initialization remains. The
hydrostatic operation writes vertex DOFs only; the pre-existing higher-order
edge/face/cell gap is separately actionable. The optional same-name mesh-field
initializer still projects all coefficients and is not interchangeable with it.
Positive-only infinity acceptance and generic Nitsche NaN predicate differences
also remain; this refactor does not broaden its scope to accepted-input changes.

BoundaryDataReaders retains eight-neighbor `(distance squared, node index)` order,
strict pruning, exact-node squared tolerance `1e-16`, inverse-distance floor
`1e-24`, x-group tolerance `1e-12` and data tolerance `1e-10`. MPI deduplication
retains first-rank/local payload order. Existing FE element interpolation does
not implement those point-cloud contracts; relocation alone would not improve
the design, and replacing it needs separate tie/pruning/MPI evidence.

Compact-interface adoption remains blocked by separable-test-row requirements.
The existing reversed-field metadata ordering discrepancy and expert compileMixed
space-signature/name identity limitation remain documented in the earlier
handoff. The expanded production route preserves all twelve actual couplings.

## Source and file counts

Counts are newline totals for C/C++ production and tests, with documentation,
Python guards and build artifacts excluded. Exact hashes and per-file counts:
`E/task3-options/disposition-source-manifest.json` and
`disposition-source-counts.json`.

| Scope | Historical fd3d68fb files / lines | Starting 89b4739f files / lines | Applied files / lines | Delta from start | Delta from historical |
|---|---:|---:|---:|---:|---:|
| NavierStokes production | 5 / 5,913 | 5 / 5,936 | 5 / 5,103 | −833 | −810 |
| Physics/Options production | 10 / 10,264 | 10 / 9,932 | 10 / 10,353 | +421 | +89 |
| FE production | 1,137 / 649,659 | 1,138 / 649,835 | 1,138 / 650,179 | +344 | +520 |
| Physics tests | 22 / 54,334 | 22 / 55,118 | 22 / 55,269 | +151 | +935 |
| FE tests | 564 / 376,878 | 564 / 377,225 | 564 / 377,700 | +475 | +822 |


All file counts are unchanged from this stage's starting revision. Relative to
the historical reference, the single additional FE production file is the
previous composition's NumericValidation.h. This stage adds no production file.

The 833-line formulation decrease includes policy relocation and replacement
operations. Options grows by 421 and FE by 344, giving **68 fewer production
lines overall** (665,703 to 665,635). Including the earlier verified composition,
the reduction from the historical reference is **201 production lines**. Tests
add 626 lines in this stage; production plus those tests grows by 558. No arbitrary
formatting compression is used to increase the reduction.

## Environment and exact validation commands

Both sides use aarch64 on lima-slurm, GCC 12.4, OpenMPI 4.1.2, LLVM 17.0.6,
CMake 3.31.4, Eigen 3.4 and VTK 9.4.1. Release uses `-O3 -DNDEBUG` and
`-ffp-contract=off`; MPI/JIT/Eigen/METIS/ParMETIS are enabled, while
MUMPS/PETSc/Trilinos/LTO/PGO are disabled. Deck-specific FSILS or serial Eigen
selection is retained. OMP/OpenBLAS/MKL threads are one, OMP_DYNAMIC is false,
and architecture-specific JIT CPU/cache-profile overrides are unset.

Starting solver SHA256:
`37194b7641a4927f33bbf300268e67a55159791512c4861f2a0f8cf0a7756768`.
Final adoption solver SHA256:
`3f3399eb07aeb64be43ed9c96418140dff967a2cccc3cac634ff2255a7427fc6`.
The stage directory freezes the executable, CMakeCache, binary hashes, source
revision/patch, untracked documentation and immutable runner copies.

```sh
source /scratch/users/zsexton/svmp-dev-outer/jobs/env.sh
E=/scratch/users/zsexton/svmp-ns-helper-evidence
B=/tmp/svmp-ns-helper-build
R=/tmp/svmp-ns-helper-source
cmake --build "$B" -j5
# Fresh tag per adoption; refuse to overwrite an existing stage.
TAG=stage2-prepared-boundaries
bash "$E/execute-group.sh" candidate "$TAG" init
python3 "$E/snapshot-stage-sources.py" "$TAG" "$R"
for GROUP in references resolve tests exhaustive; do
    bash "$E/execute-group.sh" candidate "$TAG" "$GROUP"
done
bash "$E/run-python-status-parity.sh" "$R" "$TAG"
```

The full tests group runs FE, Physics and Application directories sequentially
with `ctest -j1 --output-on-failure --timeout 14400 --output-junit FILE`.
References run refset9 and refset-ext at 1/2/4 ranks with corresponding-rank
comparison, the existing serial/two-rank limits, separate rank log capture and
unchanged exact comparators. S0 warms a fresh shared cache with the baseline,
then requires ordered manifest/tag equality and zero candidate misses.
Resolve includes the unchanged supplementary rank-consistency invocation.

The exhaustive 960-case matrix is deliberately invoked separately, because
`SVMP_TEST_EXHAUSTIVE_MPI_MATRICES=OFF` excludes it from ordinary CTest shards:

```sh
GTEST_COLOR=no timeout -k 30 6000 mpiexec -n 2 --bind-to none \
  "$B/bin/test_application_mpi" \
  --gtest_filter=ApplicationDriverLevelSetWorkflowsMPI.HydrostaticGravityWithFixedPressureGaugeMatchesAcrossTwoRankPartition
```

The exact focused invocation is `E/stage2-focused/executed-command.sh`. Its
six results are in `E/stage2-focused/tests.xml`: three original
error-precedence characterizations, two fitted tangential-policy cases and
prepared weak-boundary publication. All pass without skips. Earlier generic
contract results are recorded under `task2a`, `task2b`, `task2c`; the final suites
run those contracts again alongside their Physics consumers.

## Verification results and evidence limits

The initial evidence audit verifies all 29 preceding source/API hashes, 1,361
retained evidence hashes and 209 links. Fresh baseline numerical results agree
with the retained starting source. Two baseline orchestration failures remain
recorded as failures: a modified executing wrapper ended with EOF after successful
checkers, and ANSI color disrupted an exhaustive verdict-token match. Separate
reconciliation records preserve raw logs/statuses and verify the unchanged
outputs and binary hashes. New stages execute immutable script snapshots and
set only test-log color control; no acceptance criterion was relaxed.

Stage 1, `stages/stage1-vertex-contracts`, maps byte-for-byte to `c1798928` and
passes all groups: 40 exact reference/configuration comparisons, 16 S0 manifests
with zero misses, 12 exact log groups, 30 resolve statuses/26 artifacts, nine
supplementary rank comparisons, FE 58/Physics 25/Application 11 passing entries
plus its existing interpreter skip, and the exhaustive matrix on both ranks
(549,597 ms). All 673 Python statuses match: 587 PASS, 85 existing ERROR and one
existing FAIL. This is status parity, not a clean Python suite.

Final prepared-boundary gate, `stages/stage2-prepared-boundaries`, passes all
four groups and the full Python status comparison. It preserves all 40 reference
outputs/configurations, 16 ordered S0 manifests with zero misses, 12 exact log
groups, 30 resolve statuses/26 artifacts and nine supplementary rank comparisons.
FE 58 and Physics 25 CTest entries pass; Application has 11 passes and the same
interpreter skip. Both exhaustive ranks pass in 512,210 ms. All 673 Python
statuses remain exactly 587 PASS, 85 ERROR and one FAIL.

`preservation-summary.json` and `final-preservation-audit.json` record the required
gates and unchanged binary/helper hashes. The frozen source is `1765e10a` plus
`source.patch`; `tested-commit-mapping.json` reconstructs it in a temporary Git
index and confirms that `0a5cf14c10a456ba27dd3beb16b9e5b7898c197e` differs only
in the separately committed API documentation. The final documentation commit
and local target integration introduce no production/test/guard/build-input
change; `E/final-integrated-mapping.json` and `E/final-handoff-state.json` record
that final check. Stage1 and stage2 evidence manifests retain file hashes.

Complete CTest logs are retained separately from entry summaries. Both adoptions
have no new internal GTest skips: FE 376 events/138 names, Physics 18 events/18 names, and
Application 2 events/one name match the baseline. Some rank-specific skips pass in
other entries; their detailed classification is recorded in `case-skip-parity.json`.

An additional audit of every pre-existing test's PASS/SKIP multiplicity found
one baseline MPI success line with its opening bracket split by stdout
interleaving. The exact raw event and archive hash are recorded in
`execution-route-status-reconciliation.json`; its original parser exit 1 is
retained. Reconciliation adds only that verified baseline event, after which
all common test execution statuses match. The required original numbered-entry
skip comparison passes unchanged in all three suites; no shard redistribution
allowance was needed.

The current protocol inventory covers static drops/spheres, waves, oscillating
drops, non-right-angle sessile drops, capillary rise and advancing/receding Ren–E.
Eleven older README absolute paths are unavailable. Latest D48 sessile 60/120
records and all six referenced raw runs are available and inspected; each run
has 1,800 outputs and the recorded verifiers passed without truncation. These
histories were not rerun. Short reference parity and the hydrostatic matrix do
not requalify long transients; physical qualification remains governed by
[the program tracker](free_surface_program_tracker.md).

Separately requested VM cleanup reclaimed 31.04 GiB of disk space from stale
reproducible intermediates and test executables in twelve inactive builds.
Source, solver executables, simulation results, qualification evidence and
active processes were preserved; 75 protected hashes were unchanged. The exact
removal ledger is `E/storage-cleanup/report.md`.

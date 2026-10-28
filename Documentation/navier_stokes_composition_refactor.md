# Navier–Stokes composition refactor, 2026-10-28

## Contract and baseline

The task starts at `fd3d68fbc70a199706d7f0c24913be4c59ac6196`, also the
remote tip of `issue-449-modern-mesh-core` when checked on 2026-10-28.
Work is isolated on `refactor/navier-stokes-composition` in
`/tmp/svmp-ns-composition`. Three pre-existing untracked WP10 qualification
files in the original checkout are untouched. No repository instruction
files were found. Initial free space was 22 GiB; builds and evidence must
fit without deleting or altering other contributors' files.

The governing acceptance contract is tracker D47–D57, milestone MR and
subsequent integration records. SurfaceStress, effective timesteps, contact
certificates, constraint order, accepted history and rollback, exact
functional declarations, output/provenance formats, and preflight before
mutation are preserved. The E3/E4 historical energy tolerance permissions
do not authorize new energy differences. No benchmark inputs or gates change.

## Sequence and ownership

1. Baseline (verification specialist): inspect retained exact-revision
   executable, reference runs, manifests, registered tests and configuration;
   record availability and commands in `/tmp/svmp-ns-evidence/baseline.txt`.
   Build only in a separate directory. Existing integration builds are owned
   by other work and must not be repointed.
2. Shared services first (options specialist): `FE/Core/OptionSchema.h`
   JSON formatting policies and input spelling selection, with focused Core
   tests. Then raw numeric and boundary ScalarValue validation, distinguishing
   literal-only checks from required constants. Adopt in Options in separate
   bounded changes, retaining each caller's errors and acceptance predicates.
3. Forms service (forms specialist): transparent `symgrad` and vector
   `advect` vocabulary, ordered-tree tests and API documentation. Document and
   test compact two-phase expression limits without changing production
   compiler decomposition. Only the specialist edits Vocabulary and canonical
   workflow tests; the options specialist owns BoundaryConditions scalar APIs.
4. Direct cleanup (coordinator): confirm references before removing dead
   factories and unused forms; consolidate duplicate names; use existing FE
   field, initialization and geometry APIs where predicates match exactly.
5. Configuration/inlets (options specialist): adopt shared serialization and
   filters; share profile preparation while retaining the steady zero-value
   shortcut before collective geometry, temporal interpolation and arithmetic
   order. Reader spatial-index extraction is investigated separately.
6. Formulation composition (coordinator): use registered functional data as
   authoritative; adopt transparent vocabulary; expose phase symbol bundles
   and prepared boundary declarations where they remove duplicate construction.
   Small-cut guard preflight belongs in FE/Constraints. Keep dynamic geometry
   and collective certificate refresh checks at their required runtime points.
7. Integration (verification specialist and coordinator): each production
   adoption gets reference-set comparison, manifests/cache, logs/artifacts,
   resolve corpus, affected tests, tank-at-rest and exhaustive hydrostatic
   matrix. Do not accumulate further production switches over an unverified
   adoption. Review layering and full diff, record per-finding disposition,
   before/after form examples, exact commands/revisions and line/file deltas.

## Verification requirements

Compare baseline and candidate on the same architecture, compiler/configuration,
backend, input and MPI rank count. Refset9 and refset-ext run at 1, 2 and 4
ranks with their existing rank/backend exclusions. Require bitwise solution
outputs, ordered S0 kernels/constraints and tags, prescribed warmed-cache zero
JIT misses, unchanged energy/log/artifact and effective/resolve output.
Run affected FE, Physics and Application tests, tank-at-rest coverage, and
`ApplicationDriverLevelSetWorkflowsMPI.HydrostaticGravityWithFixedPressureGaugeMatchesAcrossTwoRankPartition`
(the ordinary MPI registration excludes this 960-case matrix unless
`SVMP_TEST_EXHAUSTIVE_MPI_MATRICES` is enabled).

Review focus: callback values are not validated by sampling; +infinity
acceptance is preserved per caller; invalid direct C++ setup cannot register
fields or operators; same-space phase fields retain distinct FieldIds; no
energy consumer silently changes reference points or applicability. Existing
functional and minimizer tests supplement tree tests. Long WP4 transient
protocols retain their existing qualification; short runs cannot replace them.

## Investigated limits (not completion claims)

- Compact mixed-test integrals are rejected by the residual installer, which
  splits top-level additive nodes. Naive distribution changes scalar placement,
  signs and term ordering relative to production. Expert `compileMixed` also
  identifies symbols differently from the FieldId-aware residual path.
- Fitted and certified generated Nitsche penalties have different ordered
  trees; the latter additionally requires an immutable additive anchor.
- Stabilization work currently evaluates complete minus Galerkin forms;
  replacing that subtraction can change cancellation and evaluation order.
- The module's two-fluid compatibility checks element, order, continuity and
  value dimension. Generic binding checks six properties, including space type,
  field type and topological dimension but excluding continuity. Combining them
  would narrow accepted existing fields. Dependency preflight and future velocity
  preregistration use the generic six properties exactly. Their aggregate error
  messages and omission of scope/marker checks remain caller contracts.
- Spatial nearest-node ties and MPI marker deduplication have deterministic
  ordering contracts. A relocation requires its own focused evidence.

### Existing shared services and intentional boundaries

Initialization already uses `requireMeshVertexField`,
`initializeFieldFromMeshVertexField`, `requireVertexDofMap`,
`levelSetSideSupportVertices`, `collectVertexFieldOnLevelSetSide`,
`levelSetSideVertexValue`, and `writeVectorEntries`. Geometry already uses
`resolveScalarLevelSetField`, `LevelSetGeometry::bind`, side/boundary/contact
keys, outward interface normals and `wettedBoundaryResolver`. ConstraintPlan
owns inactive-side and aggregation installation. The remaining gauge loops
encode local/global gauge skips, margin policy and ordered diagnostics. The
hydrostatic component fold preserves `rho * g[d] * (x[d] - x_ref[d])` and its
addition order; it is a physical policy, not a missing generic traversal.

FormulationBuilder already owns residual installation, work channels, the
separate PSPG group and opt-in diagnostics. Functional records carry work
applicability. `LevelSetEnergyParticipant` assembles
these operators against the live candidate with communicator, revision and
constraint checks; LevelSetEnergyBalance records their numerical work.
`bulkResidualWork` deliberately subtracts measured Galerkin forms from measured
complete forms. Direct stabilization formulas would change cancellation and
ordered trees. The force channel similarly subtracts constant gravity inside
the integrand. These trees are retained.

BoundaryDataReaders' nearest-node ordering is `(distance squared, node index)`;
its pruning comparison is strict, exact-node tolerance is squared `1e-16`,
and inverse-distance interpolation has floor `1e-24` and eight neighbors.
The x-only format groups coordinates within `1e-12` and checks data agreement
within `1e-10`. Marker geometry deduplicates sorted vertex-global-ID keys while
retaining the first rank/local payload, so summation order matters. The current
FE interpolation utilities are element-local and do not replace these services.
Extraction is deferred rather than changing those numerical contracts.

The final option-to-view overlay reconciles domain-first ALE/momentum values
and equation-first rotating/file data before describing provenance. Contact-law
projection gathers wall arrays and preserves first-wall defaults. These are
intentional translation policies. Input `Automatic` and effective
`SurfaceStress` spellings remain distinct; disabled states and zero surface
tension remain meaningful.

The compact-interface characterization also exposes an existing limitation:
`FormulationRecord::block_couplings` pairs traversal-order rows with caller-order
fields. With reversed orders its metadata differs from correctly installed
FieldId bindings. CouplingGraphAnalyzer and trace-multiplier checks consume
those records. The new test checks actual operator bindings, returned plans and
kernel pointers; it does not establish those metadata consumers' correctness.
The failing reproduction is preserved in
`/tmp/svmp-ns-evidence/canonical-reversed-metadata-limitation.txt`.

## Progress

The exact baseline executable and retained same-rank reference outputs are
available; inspected provenance is in `/tmp/svmp-ns-evidence/baseline.txt`.
The clean GCC 12.4 / Release / LLVM 17 / MPI candidate build is in
`/tmp/svmp-ns-build`. This leaves other builds untouched. Baseline source
counts (C/C++ headers and sources, including FE tests) are 5 files / 5,913
lines in NavierStokes, 10 / 10,264 in Options, and 1,701 / 1,026,537 in FE;
`source-count-baseline.json` records the suffix convention.

Completed shared-service commits:
- `de914bc3`: configuration formatting and input selection; 27 focused
  OptionSchema tests pass, versus 24 at baseline. Artifact comparisons cover
  all byte values and 19,993 finite random binary64 values against both old
  serializers. Schema and artifact formatting remain distinct policies.
- `e579d121`: nonmutating small-cut guard validation; 296 constraint tests
  pass, including rejection at NaN/infinity and numeric boundaries. Constructor
  error order/text and dynamic geometry checks remain unchanged.
- `0afc1da9`: `symgrad` and vector `advect`; seven IntegrationDomainLowering
  tests pass, including nonlinear ordered-tree identity. Physics adoption
  has not yet occurred.

Fresh baseline resolve-only corpus: 30 decks, 26 successful artifacts, four
existing rejections (two retired fitted Nitsche decks, a retired MMS extension,
and a missing lateral-water body-force file). Statuses and artifacts are under
`/tmp/svmp-ns-evidence/resolve/baseline-fd3d-hydrated`. An earlier run exposed
unexpanded mesh pointers; 885 locally available test objects (348 MB) were
restored without changing content, and the corpus was rerun. The earlier
setup failures are retained separately.

The first Physics adoption removes duplicate serialization/filter helpers,
uses the identical generic dependency-space predicate, removes the dead
`discreteFieldLike` and preflight zero vector, and consolidates active-domain
names. This adoption passes the complete preservation gates. All 40 same-rank
reference solutions, effective configurations and artifacts are exact; S0 has
16 identical ordered manifests and zero candidate misses. Thirty original log
comparisons pass directly. Ten merged MPI logs contain interleaved fragments;
paired exact-baseline/candidate reruns with separate rank streams pass the same
log checks and reproduce the original fields bitwise. Raw failures are retained.

FE has 58 passing CTest entries. Physics has 25 after restoring 73 exact locally
cached mesh fixtures (3.7 MB) and rerunning its 13 affected entries. Application
has 11 passing entries after supplying the isolated build's fixture lookup via
`/tmp/svmp-ns-build/tests`; its one existing NumPy-related skip remains recorded.
An unchanged rank-consistency check run with the available NumPy interpreter
passes for both baseline and candidate, with all nine same-rank outputs exact.
The exhaustive 960-case test passes on both ranks in 516,663 ms. Resolve-only
has 30 identical statuses and 26 byte-identical artifacts. Evidence and explicit
reconciliation are in `/tmp/svmp-ns-evidence/stage1-preservation-summary.json`;
the original pipeline's failed status is preserved separately. No new physical
qualification is claimed.

Numeric inconsistency retained for separate review: some existing direct C++
positive-only checks accept positive infinity. Generic Nitsche checks written
as `x <= 0` additionally accept NaN. Shared services preserve each caller's
predicate; these accepted-input changes are outside this refactor.


The validation adoption delegates literal-only checks, required constants, raw
numeric checks and both small-cut guard policies to the verified shared APIs.
The identical six-property future-velocity comparison uses `spacesCompatible`.
Forty-two focused Physics tests, three scalar-contract tests and two guard tests
pass. The full `stage2-validation` gate passes: all 40 reference outputs and
artifacts, S0, exact resolve/log contracts, FE/Physics/Application suites and
rank-consistency supplement, and the exhaustive matrix (597,793 ms on both
ranks). Two additional truncated retained-baseline records required exact paired
complete-stream comparisons; their original failures remain recorded. See
`/tmp/svmp-ns-evidence/stages/stage2-validation/preservation-summary.json`.


The inlet adoption removes 111 production lines by sharing profile, direction,
geometry, normalization and trace construction. The four unchanged-behavior
fixtures pass at one, two and four ranks; all 34 inlet trace records are exactly
the baseline records. The complete `stage3-inlets` pipeline passes without
supplements, including 40 exact reference and record comparisons, S0, all suites,
resolve/rank checks and the exhaustive matrix (593,052 ms on both ranks). Evidence:
`/tmp/svmp-ns-evidence/stages/stage3-inlets/preservation-summary.json`.


The functional adoption reads selected registered declarations in module order,
removing the competing boundary-option mapping. It validates every selected
record before adding forms and preserves caller-bound field dimensions. Sixty-three
focused declaration, minimizer and Application consumer tests pass. The complete
`stage4-functionals` pipeline passes without supplements: 40 exact reference and
record comparisons, S0, all suites, resolve/rank checks and the explicit exhaustive
matrix. Evidence:
`/tmp/svmp-ns-evidence/stages/stage4-functionals/coordinator-preservation-summary.json`.

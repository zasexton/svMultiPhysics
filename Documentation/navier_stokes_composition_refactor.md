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
- Two-fluid compatibility includes continuity whereas the generic binding
  predicate checks other properties; only identical predicates may be shared.
- Spatial nearest-node ties and MPI marker deduplication have deterministic
  ordering contracts. A relocation requires its own focused evidence.

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
names. Focused syntax and serialization comparisons pass; integrated gates
are pending. No new physical qualification is claimed.

Numeric inconsistency retained for separate review: some existing direct C++
positive-only checks accept positive infinity. Generic Nitsche checks written
as `x <= 0` additionally accept NaN. Shared services preserve each caller's
predicate; these accepted-input changes are outside this refactor.

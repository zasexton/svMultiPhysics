# Navier–Stokes composition refactor: technical handoff

All six refactoring adoptions pass their preservation gates and full Python-status comparisons at code revision `1dd814c4fcf47efbc214ff2c8fb10c765e57dba1`.
This technical handoff records the implementation and its evidence; physical qualification remains governed by `free_surface_program_tracker.md`.

## Scope and preservation boundary

The exact starting and review revision is `fd3d68fbc70a199706d7f0c24913be4c59ac6196`, then the remote tip of `issue-449-modern-mesh-core`.
Work uses `refactor/navier-stokes-composition` in `/tmp/svmp-ns-composition`; three pre-existing untracked WP10 files in the original checkout remain untouched.
The applicable contract is tracker D47–D57, milestone MR and subsequent integration records, together with the current benchmark protocols.
No benchmark input, tolerance, solver setting, quadrature, partition, default, expected result, or retired route is changed.
The established five-file Navier–Stokes formulation layout and Physics/Options organization remain; generic services are added to their existing FE layers.
Independent cross-owner review covered the complete composition. All 29 reviewed source, test and API-documentation files match the final committed content (`final-applied-source-review.json`); no benchmark or build input changed. The final runtime gate then validated that composition.

## Disposition of cleanup findings

The identifiers below follow the requested cleanup A–G and forms A–E; each row distinguishes implemented work from intentional retention or separately deferred work.

| Finding | Disposition and concrete evidence | Deliberate boundary |
| --- | --- | --- |
| Cleanup A: scalar/numeric validation | Implemented in stage 2: shared raw predicates and literal-only/required-literal APIs serve Options, module and boundary factories. Scalar-contract tests exercise direct C++ values, callbacks and FormExpr alternatives; existing invalid-option tests retain caller diagnostics. | Physics owns finite-first error order and model requirements. A literal check does not evaluate a coefficient. Positive-only infinity acceptance and generic Nitsche quirks are retained, as detailed below. |
| Cleanup B: field checks/preflight | Identical dependency and future-velocity six-property comparisons now call `spacesCompatible`; both small-cut guard checks use the new nonmutating FE predicate. Stage 6 prepares static weak-velocity values once during the existing preflight pass, then installs against actual fields; its focused and full preservation gates pass. | Four-property two-fluid checks, distinct-name error order, source-kind requirements and aggregate diagnostics remain separate. Dynamic geometry/state/distributed certificates and installation-time anchors remain live. |
| Cleanup C: configuration | Stage 1 replaces duplicate JSON quoting/numeric/boolean formatting and input spelling filters, and removes the duplicate active-domain name mapping. Schema declarations, aliases, conflicts, retirement and provenance continue through OptionSchema. Full artifact/resolve comparisons pass. | Schema and artifact escaping policies differ intentionally; RealCodec shortest formatting remains distinct. Final option-to-view overlays encode precedence and are retained. |
| Cleanup D: inlet construction | Stage 3 shares profile selection, masks, normals, geometry, normalization and trace output, removing 111 production lines. Four pinned fixtures pass at 1/2/4 ranks; all 34 baseline trace records match exactly. | Temporal data is read before unsteady profile validation. Steady profile errors precede the zero shortcut; zero returns before collective geometry. Steady divide and unsteady reciprocal-then-multiply grouping are unchanged. |
| Cleanup E: initialization/geometry/readers | Existing shared initialization, side-support, geometry-binding and ConstraintPlan services already satisfy much of this finding. Their call sites were audited rather than replaced. Hydrostatic and gauge policy remain visible in Physics. | Point-cloud interpolation and marker geometry extraction are deferred: existing FE interpolation is element-local and does not preserve this reader's nearest-node/tie/deduplication contracts without separate work and evidence. |
| Cleanup F: authoritative functionals | Implemented in stage 4: selected registered records supply energy and wall statements, removing the unused option-based overload and redundant arguments. All 63 focused authority/consumer tests and the complete stage preservation gate pass. | Selection follows this module's configured free-surface order, excludes unselected records, preserves optional channels/reference coefficients and rejects any missing record before declaring forms. E4 minimizer consumers are part of the audit. |
| Cleanup G: dead/redundant code | Stage 1 removes repository-confirmed unused `Factories::detail::discreteFieldLike`, `zero_components`/`zero_flux`, and `pressureActiveDomainName`. Stage 5 adopts `sumOf`, `constantVector` and existing projection vocabulary, with its full preservation gate passed. | CutVolume and resolved SurfaceStress remain the supported policies. Input Automatic provenance, unsupported-value rejection, meaningful disabled states, capability checks and gamma=0 behavior remain. |

## Disposition of form-authoring findings

| Finding | Disposition and concrete evidence | Deliberate boundary |
| --- | --- | --- |
| Forms A: mathematical vocabulary | FE `symgrad(u)` and vector `advect(a,u)` lower to the old trees; nonlinear ordered-tree tests passed before Physics adoption. Stage 5 uses them in bulk, VMS and boundary equations; all 30 focused checks and the full preservation gate pass. | Strong residuals, tau, subscales, stress laws and all state dependencies remain explicit; no opaque formulation assembler is introduced. |
| Forms B: contributions/work | Existing FormulationBuilder already owns residual groups, PSPG, diagnostics and work membership. Stage 5 shares matching SurfaceStress integrands and existing vocabulary; adoption and its preservation gate are complete. Existing functional/work records remain authoritative. | Complete-minus-Galerkin stabilization work and the force channel's constant-gravity subtraction are retained: direct formulas would change cancellation and ordered evaluation. No competing contribution framework is added. |
| Forms C: construction/installation | Stage 5 adds `IncompressiblePhaseSymbols` and shared fitted/unfitted pressure/surface-energy integrands, with preservation verified; stage 6 adds verified prepared weak-velocity descriptors. Builders still return forms/declarations to existing orchestration. | Explicit normals and caller-selected measures retain frame/orientation/geometry sensitivity. Phase bundles retain actual FieldIds, phase order and every expanded interface channel; the positional adapter preserves existing validation order. |
| Forms D: form-bound Nitsche | Existing descriptors/certified builders were investigated. Stage 6 shares the emitter and reuses prepared static input; all eight focused actual-field/error-order tests and the full preservation gate pass. | Fitted and certified generated penalties have different ordered trees. Pressure coupling, fresh immutable binding, required top-level additive anchor and all dynamic certificates are retained; no forced unification of those equations. |
| Forms E: compact interfaces | Compiler investigation completed; production compact-expression adoption deferred. `CanonicalWorkflow.SameSpaceWeightedInterfaceRequiresSeparableTestRows` characterizes rejection and verifies the expanded route with distinct same-space FieldIds, asymmetric weights and reversed caller field order. | Residual installation requires separable top-level additive test rows. Compact inner jumps contain multiple test functions and reject before mutation. No uncontrolled nonlinear expansion or substitution with DG jump/average operators is introduced. |

## Shared APIs and ownership

| Layer and API | Contract and reason for placement |
| --- | --- |
| `FE/Core/OptionSchema.h`: `options::json::{quote,number,boolean}`, `ControlEscapes` | Generic serialization mechanics, independent of physical schema. Control escaping is a caller policy; numbers retain classic locale, max-digits precision and caller-owned nonfinite error text. |
| `OptionInput::{selectSpellings,excludeSpellings}`; existing `OptionSchema::spellings` | Stable input filtering preserves order, duplicate entries and provenance; schema spellings supply canonical names and declared aliases. |
| `FE/Core/NumericValidation.h`: `numeric::{isFinite,isNonnegative,isPositive,validate}` | Raw numeric predicates keep finiteness independent from sign. `validate` preserves the caller's exact message. This is the only new FE production file. |
| `FE/Forms/BoundaryConditions.h`: `bc::{validateScalarLiteral,requireScalarLiteral}` | ScalarValue-aware operations belong beside ScalarValue. The first only checks the Real alternative; the second requires that alternative but imposes no numeric range. |
| `FE/Forms/Vocabulary.h`: `symgrad`, vector `advect` | Transparent mathematical shorthand with unchanged child order, expression trees and full differentiation dependencies. Existing `sumOf`, `constantVector` and tangential projection are reused. |
| `FE/Constraints/SmallCutAggregationConstraint.h`: `validSmallCutAggregationGuards` | Nonmutating validation shares the constructor's guard predicate. Constructor errors and later geometry-dependent checks remain in their original locations. |
| Existing `FE::systems::spacesCompatible` | Reused only where the complete six-property predicate is identical; caller lookup, component/source-kind checks and diagnostic policy remain in Physics. |
| Physics factories: `exteriorPressureIntegrand`, `surfaceEnergyIntegrand`, `IncompressiblePhaseSymbols` | These express Navier–Stokes boundary laws and phase conventions. They accept explicit geometry/fields without moving model policy into FE. Stage 5 adoption is verified. |
| Physics options: `PreparedWeakVelocityBoundary`, `NavierStokesPreflight::prepared_weak_velocity_boundaries` | These retain the physical/generated marker and prescribed expression from successful preflight; they do not retain actual trial/test fields or a geometry certificate. Stage 6 adoption is verified. |

## Actual form changes

These excerpts compare baseline source with the applied composition. Their stage 5 adoption passes the focused tree checks and complete preservation gate.
Bulk strong residual, before:

```cpp
const auto R_np = rho * (dt(u) + grad(u) * a + chi - f)
    - div(FormExpr::constant(2.0) * mu * sym(grad(u)));
```

After, with the same ordered tree and nonlinear dependence:

```cpp
const auto R_np = rho * (dt(u) + advect(a, u) + chi - f)
    - div(FormExpr::constant(2.0) * mu * symgrad(u));
```

SurfaceStress, before:

```cpp
const auto pressure = p_ext * inner(n, v);
const auto surface_energy = gamma * inner(tangentialProjector(n, dim), grad(v));
M = M + (pressure + surface_energy).dI(Gamma);
```

After:

```cpp
const auto pressure = exteriorPressureIntegrand(p_ext, n, v);
const auto surface_energy = surfaceEnergyIntegrand(gamma, n, v, dim);
M = M + (pressure + surface_energy).dI(Gamma);
```

The generated route still uses the level-set outward normal and `dI`; fitted ALE uses its current normal and `ds`.
Pressure and surface energy remain grouped under one integral, and the gamma=0 pressure-only branch remains unchanged.
The phase builder previously received eight expressions positionally:

```cpp
const auto forms = Factories::buildIncompressibleTwoFluidInterfaceForms(
    StateField(u_negative, *velocity_space_, o.negative_phase.velocity_field_name),
    StateField(p_negative, *pressure_space_, o.negative_phase.pressure_field_name),
    TestField(u_negative, *velocity_space_, "v_negative"),
    TestField(p_negative, *pressure_space_, "q_negative"),
    StateField(u_positive, *positive_velocity_space_, o.positive_phase.velocity_field_name),
    StateField(p_positive, *positive_pressure_space_, o.positive_phase.pressure_field_name),
    TestField(u_positive, *positive_velocity_space_, "v_positive"),
    TestField(p_positive, *positive_pressure_space_, "q_positive"), interface_parameters);
```

The resulting module call gives those same actual-field symbols names (the positive bundle follows the identical pattern):

```cpp
const Factories::IncompressiblePhaseSymbols negative{
    .velocity = StateField(u_negative, *velocity_space_, o.negative_phase.velocity_field_name),
    .pressure = StateField(p_negative, *pressure_space_, o.negative_phase.pressure_field_name),
    .velocity_test = TestField(u_negative, *velocity_space_, "v_negative"),
    .pressure_test = TestField(p_negative, *pressure_space_, "q_negative")};
const auto forms = Factories::buildIncompressibleTwoFluidInterfaceForms(
    negative, positive, interface_parameters);
```

This changes argument organization, not interface expansion. The scaled min/max material-weight calculation and reliable harmonic weights remain unchanged.

## Retained contracts and practical limitations

Scalar validation distinguishes literal checking, requiring a literal and accepting general coefficients. Constant FormExpr objects remain expressions, even when their value is nonfinite.
No callback is sampled. Physics retains finite-first diagnostics, boundary-marker context, RCR/RCRCR parameter order and nonzero checks.
Some direct positive-only density, constant-viscosity and VMS-epsilon checks accept positive infinity; later graph, certification or artifact checks may still reject it.
This is not a claim of end-to-end infinity support. Generic Nitsche tests written `gamma <= 0` also accept NaN; `!(gamma > 0)` rejects NaN but accepts positive infinity.
These distinct pre-existing predicates are recorded for separate review rather than silently normalized in this refactor.

Two-fluid compatibility checks element, order, continuity and value dimension. Generic compatibility checks space type, field type, value dimension, topological dimension, order and element, without continuity.
Combining them would narrow accepted inputs. Dependency and future-velocity checks match the generic six exactly, but direct FieldBinding checks add scope/marker requirements and different errors.
Distinct-name wrappers retain their empty-name/duplicate ordering and messages. Phase fields require Unknown sources; level-set requirements permit the existing Unknown/Prescribed alternatives.
Registration order, first-error precedence, aggregate diagnostics and checks before FE mutation remain caller contracts.

Prepared Nitsche capture occurs lazily at each original validation/conversion point, after shared penalty setup; eager preparation would reorder failures.
The preflight result is published only after the existing boundary-count check succeeds. Installation binds actual FieldIds and creates the certified anchor afresh.
`requireCertifiedNitscheTraces` still registers wetted markers before installation; prepared values do not bypass this cache-invalidating side effect. Reusing the prepared marker removes one redundant private cache-version increment per generated weak route; certification still invalidates the analysis cache before installation.
Contact/trace geometry certification, immutable top-level additive anchors and distributed refresh/rejection checks remain live; only static input conversion is reused.
NormalIncrement's integrator-derived effective timestep and reference refresh, contact/Ren–E/slip laws, ALE geometry tangents and accepted-history/rollback behavior are unchanged.

Initialization already calls `requireMeshVertexField`, `initializeFieldFromMeshVertexField`, `requireVertexDofMap`, `levelSetSideSupportVertices`, `collectVertexFieldOnLevelSetSide`, `levelSetSideVertexValue` and `writeVectorEntries`.
Geometry already uses `resolveScalarLevelSetField`, `LevelSetGeometry::bind`, side/boundary/contact keys, outward normals and `wettedBoundaryResolver`; ConstraintPlan owns inactive-side and aggregation installation.
The hydrostatic fold remains `rho * g[d] * (x[d] - x_ref[d])` in its original addition order. Local/global gauge skips, margin policy and ordered statistics are physical choices retained in Physics.

BoundaryDataReaders retains eight-neighbor ordering by `(distance squared, node index)`, strict pruning, squared exact-node tolerance `1e-16` and inverse-distance floor `1e-24`.
Its x-only format groups coordinates at `1e-12` and checks data agreement at `1e-10`. MPI marker deduplication sorts unique vertex-global-ID keys and retains the first rank/local payload, preserving summation order.
Moving these algorithms into an element-local FE interpolator would not be an equivalent extraction; generic point-cloud service work needs separate tie, pruning and MPI tests.
Option-to-view overlays likewise remain: ALE/momentum values are domain-first, rotating/file values equation-first, and contact projection gathers wall arrays while retaining first-wall defaults.

The functional authority change preselects every requested marker through the const registry before constructing any form, so a later missing marker cannot leave a partial declaration.
Pointers remain stable while `F.functional` writes the separate form store. Energy and wall statements use the same selected record; other records are excluded and module selection order wins over registry sorting.
Supported registration supplies the same velocity identity, geometry domain, active-volume density/gravity/reference point, optional viscosity/pressure and endpoint flags as before.
The existing record overload's absent-energy defaults (zero density/gravity/reference point) remain; the authority test includes selected records with and without optional channels.
E4's static minimizer consumes declared liquid volume, liquid-gas plus Young-wall surface energy and gravitational energy. These are solver inputs, not disposable log records.
`LevelSetEnergyParticipant` evaluates work against the live candidate with communicator, revision and constraint checks; LevelSetEnergyBalance records that work. Measured complete-minus-Galerkin and force-subtraction channels remain unchanged.

Compact-interface tests validate the twelve actual operator couplings, returned mixed plan and Jacobian pointers for the expanded route; they do not claim compact numerical assembly support.
An existing metadata limitation remains: `FormulationRecord::block_couplings` pairs traversal-order rows with caller-order fields, so reversed orders can disagree with correct installed FieldId bindings.
CouplingGraphAnalyzer and trace-multiplier checks consume that metadata; the reproduction is `canonical-reversed-metadata-limitation.txt` in the evidence directory.
Expert `compileMixed` identifies symbols by space signature plus name rather than the canonical residual path's FieldId identity. Distinct names and separable rows remain required; neither limitation is repaired here.

## Reproducible environment and commands

Evidence: `E=/scratch/users/zsexton/svmp-ns-composition-evidence`; source `R=/tmp/svmp-ns-composition`; build `B=/tmp/svmp-ns-build`. After all validation jobs completed, the evidence directory was moved atomically to persistent scratch storage; `/tmp/svmp-ns-evidence` remains its execution-path alias. The commands below use that alias. `final-handoff-state.json` records the archive location and documentation-only revision separately from the tested code; `final-evidence-manifest.json` records integrity hashes for scripts, summaries, source records and retained executables.
Exact baseline executable: `/scratch/users/zsexton/svmp-dev-cutgeom/bin/svmp-fd3d68fb`, SHA256 `690cb791b33f8f87a2c2ae1be124b91a094321890236bc25bcebf8ad78c7f622`.
Its successful build provenance is `/scratch/users/zsexton/svmp-dev-cutgeom/logs/s8-verify-330.out`; baseline details and original results are retained in `E/baseline.txt`.
The final frozen solver is `E/stages/stage6-prepared-final/svmultiphysics`, SHA256 `37194b7641a4927f33bbf300268e67a55159791512c4861f2a0f8cf0a7756768`; its saved source patch matches code commit `1dd814c4fcf47efbc214ff2c8fb10c765e57dba1`.
Both sides use aarch64 on `lima-slurm` (Apple CPU, 16 logical CPUs), GCC 12.4, OpenMPI 4.1.2, LLVM 17.0.6, CMake 3.31.4, Eigen 3.4 and VTK 9.4.1.
Release C++20 uses `-O3 -DNDEBUG -ffp-contract=off`; LLVM JIT, Eigen, MPI, METIS, ParMETIS and assembly are enabled; MUMPS/PETSc/Trilinos, IPO/LTO and PGO are disabled.
The environment script sets `OMP_DYNAMIC=FALSE`, `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`; `SVMP_JIT_CPU` and `SVMP_CACHE_PROFILE` remain unset.
Ordinary runs leave `SVMP_JIT_CACHE_DIR` unset. S0 uses a fresh shared cache, exact baseline first and candidate second, requiring zero candidate misses and identical ordered manifests/tags.

```sh
source /scratch/users/zsexton/svmp-dev-outer/jobs/env.sh
cmake -S /tmp/svmp-ns-composition/Code -B /tmp/svmp-ns-build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_C_COMPILER=/share/software/user/open/gcc/12.4.0/bin/gcc \
  -DCMAKE_CXX_COMPILER=/share/software/user/open/gcc/12.4.0/bin/c++ \
  -DEigen3_DIR=/share/software/user/open/eigen/3.4.0/share/eigen3/cmake \
  -DLLVM_DIR=/share/software/user/open/llvm/17.0.6/lib/cmake/llvm \
  -DVTK_DIR=/share/software/user/open/vtk/9.4.1/lib64/cmake/vtk-9.4 \
  -DBoost_INCLUDE_DIR=/share/software/user/open/boost/1.90.0/include \
  -DFE_ENABLE_LLVM_JIT=ON -DFE_ENABLE_EIGEN=ON -DFE_ENABLE_MPI=ON \
  -DFE_BUILD_TESTS=ON -DPHYSICS_BUILD_TESTS=ON -DAPPLICATION_BUILD_TESTS=ON \
  -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/scratch/users/zsexton/svmp-dev-outer/deps/googletest-src \
  -DSVMP_TEST_EXHAUSTIVE_MPI_MATRICES=OFF
cmake --build /tmp/svmp-ns-build -j 5
bash /tmp/svmp-ns-evidence/run-stage1.sh stage1
bash /tmp/svmp-ns-evidence/run-adoption.sh stage2-validation
bash /tmp/svmp-ns-evidence/run-adoption.sh stage3-inlets
bash /tmp/svmp-ns-evidence/run-adoption.sh stage4-functionals
bash /tmp/svmp-ns-evidence/run-adoption.sh stage5-forms
bash /tmp/svmp-ns-evidence/run-adoption.sh stage6-prepared-final
```

The stage runner requires a fresh tag and snapshots the frozen executable, all test binary hashes, CMakeCache, source revision, working patch and untracked sources before running comparisons.
Exact calls for every group are in `E/run-adoption.sh`; stage 1 used `E/run-stage1.sh`. Reproduction must use new tags, not overwrite retained evidence.
It invokes `refset9-captured.sh BIN TAG N s8-fd3d68fb` and `refset-ext-captured.sh BIN TAG -n N s8-fd3d68fb` for N=1,2,4, plus the existing `op_manifest.sh`/`op_cmp_manifests.sh` pair.
It runs CTest with `-j 4 --output-on-failure --timeout 14400 --output-junit` in `B/Source/solver/FE`, `B/Source/solver/Physics` and `B/Source/solver/Application/Tests`.
It invokes `p_resolve_corpus.sh BIN TAG R`, `check-resolve-stage.py`, reference/feature/log checkers, and the unchanged rank-consistency supplement using `/scratch/users/zsexton/bin/py` (Python 3.14.2, NumPy 2.3.5).
Focused runs use `mpiexec -n 1 --bind-to none B/bin/TEST --gtest_filter=FILTER`; each focused log records its exact filter. `stage2-focused-{physics,scalar,guards}.log` map to `test_physics`, `test_fe_forms`, `test_fe_constraints`; stage 4–6 formulation logs use `test_physics`, while `stage4-consumer-{records,minimizer,application}.log` use `test_fe_systems`, `test_fe_levelset`, `test_application`.
The exhaustive matrix is explicitly outside ordinary registration and is run as:

```sh
timeout -k 30 6000 mpiexec -n 2 --bind-to none \
  /tmp/svmp-ns-build/bin/test_application_mpi \
  --gtest_filter=ApplicationDriverLevelSetWorkflowsMPI.HydrostaticGravityWithFixedPressureGaugeMatchesAcrossTwoRankPartition
```

Reference artifacts are under `/scratch/users/zsexton/free-surface-benchmarks/perf-reference-arm64`, tags `s8-fd3d68fb`, `s8-fd3d68fb-np2`, `s8-fd3d68fb-np4`, in `runs` and `runs-ext`.
Refset9 contributes 9/7/7 cases and refset-ext 7/6/4 at 1/2/4 ranks: 40 same-rank cases. Backends remain each deck's FSILS or serial Eigen; MPI launches use `--bind-to none`, including rank 1.
Existing exclusions remain: slosh_L16, tank3d_L8 and fitted_capwave are serial Eigen; the two 4-by-4-cell two-fluid cases run at most two ranks, as specified in `cases-ext/cases.txt`.
Case environments remain sourced: capillary rise sets `SVMP_GENERALIZED_ALPHA_PDE_UDOT_INIT=0`, strong Ren–E sets `SVMP_CONTACT_POINT_VELOCITY_TRACE=1`, and two-fluid cases retain one thread and `SVMP_OOP_SOLVER_TRACE=0`.
Local runners alter only output/data-root routing and capture rank streams separately; input directories and exact-baseline runs link to originals. Their source hashes and changes are in `runner-adaptations.txt`.
Log checks retain `r2_cmp_logs.py` and `rf_cmp_modlogs.py --exact`, with existing nondeterministic time/path normalization and stream reconstruction; no numerical tolerance or new conversion map is introduced.

## Adoption evidence and revisions

Service commits precede Physics switches: `de914bc3` configuration (27 focused tests), `e579d121` guards (296 constraint tests), `0afc1da9` vocabulary (7 ordered-tree tests), `71f74235` scalar contracts, and `5881e6b7` documentation/compiler characterization.
`a8bab05c` pins four inlet contracts before the inlet switch. Configuration characterization also compares every byte and 19,993 finite random binary64 values against both prior serializers.
Stage source identity is the saved `revision.txt` plus `uncommitted.patch`/untracked snapshot; commits below identify the subsequently committed logical adoption, not a claim that the frozen binary was built after that commit.

| Adoption | Revision/result | Evidence under E |
| --- | --- | --- |
| 1: configuration/dead code/exact field reuse | `b3c16259fef79865823a0233c3480fff8e649d98`; full gate passed. 40 exact field/configuration/artifact comparisons, S0 16 ordered manifests/zero misses, suites and resolve/rank checks; exhaustive matrix 516,663 ms on both ranks. Log scope is qualified below. | `stage1-preservation-summary.json`, `baseline.txt`, original stage logs and capture reruns. |
| 2: validation | `48e7354f7039ad06048906a73af506ddfaf9af7f`; 42 focused Physics, 3 scalar and 2 guard tests passed; full gate passed, including 40 references and exhaustive matrix 597,793 ms on both ranks. | `stages/stage2-validation/preservation-summary.json`; frozen source base is stage 1 plus saved validation patch. |
| 3: inlet construction | `08033f61eeda8b8dc89fb73a31d3399d82083175`; four fixtures at 1/2/4 ranks and 34 exact trace records passed; full pipeline passed without supplements, including exhaustive matrix 593,052 ms on both ranks. | `stages/stage3-inlets/preservation-summary.json`, `inlet-baseline-contract*`, `inlet-baseline-trace-records.txt`; frozen source base is stage 2 plus inlet patch. |
| 4: functional authority | `50479e4fef7ed4d86f276c0cfc9b822d3dfd4f1f`; all 63 focused tests pass (3 Physics, 18 FE Systems, 35 FE LevelSet, 7 Application). Full pipeline passes without supplements: 40 exact references/records, S0 and all remaining stage checks; exhaustive matrix 588,410 ms on both ranks. | `stages/stage4-functionals/preservation-summary.json`, `stage4-functionals-focused.*`, `stage4-consumer-*`; frozen source base is stage 3 plus functional patch. |
| 5: form vocabulary/phase bundles/integrands | `b3ff430a58d952e4f30647a8ad5c37638bc01c38`; build, 30 focused tests and all 31 stage gates pass, including 40 exact references/records and exhaustive matrix 594,361 ms on both ranks. | `stage5-forms-focused.*`, `stages/stage5-forms/preservation-summary.json`; frozen source base is stage 4 plus form patch. |
| 6: prepared weak-velocity installation/final composition | `1dd814c4fcf47efbc214ff2c8fb10c765e57dba1`; build, all 8 focused tests, all 31 preservation gates and full Python-status parity pass. All 40 references/records are exact; the exhaustive matrix passes in 592,489 ms on both ranks. | `stage6-prepared-focused.xml`, `stages/stage6-prepared-final/preservation-summary.json`, `python-parity/stage6-prepared-final/`; frozen source base is stage 5 plus the prepared-boundary patch, now committed unchanged. |

All six stages have FE 58/58 and Physics 25/25 CTest entries; Application has 11 passing entries and its existing NumPy-related skip, supplemented by nine exact same-rank outputs through the unchanged check with an available interpreter.
They preserve 30 resolve statuses and 26 byte-identical successful artifacts. Later stage runners additionally check all 17 permitted extended feature expectations and retain complete candidate rank streams.
Tank-at-rest/60-case hydrostatic coverage is in the ordinary Application suites; the separately executed exhaustive test covers all 960 two-rank cases. No ordinary CTest result is substituted for that matrix.

Stage 1 established the existing comparator result for 30 retained merged logs and added ten complete exact-baseline/candidate paired captures for interleaved or truncated MPI logs, each also reproducing original fields bitwise.
Stage 2's complete candidate captures exposed two further truncated retained-baseline fragments: sessile_L16_60 at two ranks and sessile_L16_120 at four ranks, both rank 1, step 0 contact records.
Those cases had passed stage 1's established comparator because both retained the same incomplete fragment; stage 1 is not claimed to have forty complete-stream comparisons.
Stage 2 added complete paired reruns for both cases, matching original fields/configurations and the same exact log comparator. Its summary records 38 primary plus 2 supplemental comparisons.
Evidence is in `capture-reruns/stage2-validation-gaps` and `capture-reruns/stage2-validation-gaps-np4`; original failed statuses/logs remain intact alongside explicit reconciliation.
Stage 3 and subsequent runners compare complete candidate streams with the twelve qualified complete baseline captures and the established retained-log reconstruction for the other cases. Stages 3–6 needed no additional supplement.

Retained exact-baseline CTest evidence has FE 58/58, Physics 25/25 and Application 11 passes plus one skip. Local candidate setup initially failed 13 Physics entries for missing fixtures; restoring 73 exact locally cached meshes (3.7 MB) and rerunning those entries yields 25/25.
Application fixture lookup uses `B/tests -> R/tests`; `Application_FreeSurfaceRankConsistency_MPI_4` retains skip code 77 because CMake's Python 3.12 lacks NumPy. The unchanged supplemental check passes baseline and candidate with the available interpreter.
The resolve corpus initially hit 13 unexpanded mesh pointers; 885 locally available objects (348 MB) were restored without changing content, then the exact baseline corpus was rerun at `resolve/baseline-fd3d-hydrated`.
Its four existing rejections are two retired fitted-Nitsche decks, a retired MMS algebraic-extension deck and a missing lateral-water body-force file. They remain identical, rather than being waived or repaired as part of the method.
Historical S8 Python coverage was only a subset: 38 PASS/53 ERROR on exact baseline and its candidate. The full retained gate is `/scratch/users/zsexton/svmp-integration-arm64/logs/pytest-302`: 75 selected files, 673 statuses (587 PASS, 85 ERROR, 1 FAIL), with zero allowed or observed status changes; neither record is a clean Python suite.
Fresh exact-baseline full parity now matches all 673 statuses with zero changes after restoring 19 hash-verified local LFS fixtures (387,110 bytes); the original six unexpected deltas remain in `python-parity/baseline-fd3d`, and the successful rerun/cleanup in `python-parity/baseline-fd3d-hydrated`. Temporary worktrees restore their original pointers before removal; neither code nor fixture content changes.
`E/run-python-status-parity.sh SOURCE TAG` retains the 75-file selection, runner, per-test comparator and interpreter (Python 3.14.2, NumPy 2.5.4, PyVista 0.49.1), with no approved status changes. All six committed stages also match all 673 statuses, with unchanged checkers and source state, under `E/python-parity/`. A stale job-330 feature checker separately reported errors; the current existing LogRecord checker passes exact-baseline logs at 1/2/4 ranks (`baseline.txt`).

The successful fresh baseline parity command below ran on the detached exact-baseline source after the 19-fixture hydration; subsequent verified batches recreate and clean their own source snapshots. These are the recorded tags; reproduction must use fresh evidence tags.

```sh
bash /tmp/svmp-ns-evidence/run-python-status-parity.sh /tmp/svmp-ns-python-source baseline-fd3d-hydrated
bash /tmp/svmp-ns-evidence/run-python-stages-verified.sh stages1-through-stage4 stage1:b3c16259 stage2-validation:48e7354f stage3-inlets:08033f61 stage4-functionals:50479e4f
bash /tmp/svmp-ns-evidence/run-python-stages-verified.sh stage5-forms-batch stage5-forms:b3ff430a58d952e4f30647a8ad5c37638bc01c38
bash /tmp/svmp-ns-evidence/run-python-stages-verified.sh stage6-final-batch stage6-prepared-final:1dd814c4fcf47efbc214ff2c8fb10c765e57dba1
```

## Source counts and qualification boundary

Authoritative counts are in `source-count-committed.json`, comparing baseline `fd3d68fb` with final code `1dd814c4fcf47efbc214ff2c8fb10c765e57dba1`, without an overlay. Every group matches the applied-source and planned counts; `source-count-final-comparison.json` and `source-count-final-working.patch` retain the earlier review provenance.
Counts use newline totals for `.c`, `.cpp`, `.h`, `.hpp`, `.inl`, `.tpp`; `/Tests/` paths are tests and documentation is excluded. No formatting compression was used to manufacture reduction.

| Scope | Before files / lines | Final files / lines | Delta |
| --- | ---: | ---: | ---: |
| NavierStokes production | 5 / 5,913 | 5 / 5,936 | 0 files, +23 lines |
| Physics/Options production | 10 / 10,264 | 10 / 9,932 | 0 files, −332 lines |
| FE production | 1,137 / 649,659 | 1,138 / 649,835 | +1 file, +176 lines |
| FE tests | 564 / 376,878 | 564 / 377,225 | 0 files, +347 lines |
| FE total | 1,701 / 1,026,537 | 1,702 / 1,027,060 | +1 file, +523 lines |

Physics production decreases by 309 lines, while reusable FE production increases by 176: the combined production reduction is 133 lines, not 309 lines of net deletion.
Actual Physics tests are 22 files, 54,334 → 55,118 lines (+784 across five existing files), independently recorded in `source-count-final-physics-tests.json`; they are not counted as production or hidden as relocation.
The five-file formulation layout is preserved; the sole new FE production file is `NumericValidation.h`. Counts describe the result, not an acceptance target.

This work preserves the established short reference evidence; it does not upgrade long WP4 qualification. Existing protocols still require tank five periods, relaxed drop/sphere and timestep studies, four-period capillary/oscillating waves, sessile 2D five/3D ten viscous times, strong Ren–E to t=0.25 plus refinement, and capillary-rise history/envelope checks.
No cross-architecture or cross-rank bitwise claim is made. Compact compiler support, the recorded metadata limitation and retained numerical-policy inconsistencies remain separate work; the verified expanded production route remains in use.

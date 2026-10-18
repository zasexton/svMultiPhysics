# Explored Optimization Areas

Local workflow record. This file must remain untracked, unstaged, uncommitted,
and unpushed. Do not add it to `.gitignore`; its visible `??` status is the
guard that it cannot accidentally enter a performance commit.

- Last reconciled: 2026-09-04 00:29 PDT
- Current source commit: `8f71dbf79353d8bf8786274ad2ac84c120d751cf`
- Evidence root: `/scratch/users/zsexton/svmp-efficiency-review-20260829`
- Canonical job record: `/scratch/users/zsexton/svmp-efficiency-review-20260829/slurm-ledger.tsv`
- Scope: focused changes to one or two existing source files, with a preferred
  production-runtime improvement greater than 5%.

## Maintenance rules

1. Add an entry when a profile hotspot is inspected, even if it is rejected.
2. Record the exact source commit, profile artifact, job ID, and reason for the
   decision whenever those are available.
3. Keep accepted, rejected, superseded, and overlap-excluded ideas here
   permanently so later searches do not rediscover them.
4. Put only actionable, unvalidated candidates in
   `optimization_idea_queue.md`; use the same stable candidate ID in both files.
5. Reconcile both files after each profiler completion, optimizer report,
   validation result, overlap discovery, source change, or commit.
6. During scheduler waits, assign `cpp_optimizer` a read-only search over newly
   completed profiles and source paths not already represented below.
7. Never let a discovery worker edit source, submit jobs, update the Slurm
   ledger, or edit these records. The controller owns all mutable state.

## Status legend

- `QUEUED`: credible candidate awaiting design approval or job validation.
- `IN PROGRESS`: read-only investigation is active.
- `COMPLETE`: a discovery lane finished and its decisions were recorded.
- `ACCEPTED`: validated and retained in the branch.
- `REJECTED`: measured benefit, correctness, complexity, or scope did not pass.
- `SUPERSEDED`: a later retained design replaced this idea.
- `EXCLUDED`: owned elsewhere or outside the agreed scope.

## Current and recently explored candidates

| ID | Area | Files | Status | Evidence and decision |
|---|---|---|---|---|
| `FLUID-WEAK-NNX-001` | Reuse the velocity shape vector for equal-order weak Dirichlet pressure assembly | `Code/Source/solver/set_bc.cpp` | ACCEPTED | Retained and pushed as `958bd1e7`. Source-hash-locked job `41753579` rebuilt default and PETSc configurations, passed the unit target and all 224 runnable integration tests with 24 expected Trilinos-only skips, and produced independently checksum-verified binaries. Job `41757424` completed 15 alternating native pairs with exact VTK state and normalized histories: all 15 favored the candidate, median wall reduction was 9.259%, and both uncertainty lower bounds were 8.589%. Resume job `41758628` completed exact VTK and normalized-history checks at 1/3/4 ranks plus all 32 matched profiles. Its controller rejected normal sub-0.462% Callgrind replicate variation after the profiles; independent reconciliation verified zero worker failures, 152 exact scientific comparisons, exact Memcheck contexts, a 13.051% median whole-step instruction reduction, and a 37.392% median weak-region reduction. `get_nnx` calls and its allocation subtree both fall exactly by half; whole-process requested allocation bytes fall 7.546% and blocks 26.528%, while peak live memory is unchanged. Taylor--Hood retains its independent solve. Evidence: `performance-r4/controller-validation.txt`, `precommit-r1`, `petsc-profile-r1`, and `graphs/FLUID-WEAK-NNX-001`. |
| `FLUID-FS-001` | Hoist `get_thood_fs` out of the fluid element loop | `Code/Source/solver/fluid.cpp` | ACCEPTED | Retained and pushed as `a3b3f1dd`. The one-file candidate initializes the two phase-specific spaces before the element loop and preserves the existing momentum/continuity arithmetic. Source-locked job `41761822` passed the unit target, all 224 runnable integration tests with 24 expected Trilinos-only skips, and independently verified default/PETSc archive checks. Job `41762902` produced 15/15 native wins, all above 5%, with a 9.289617% median reduction and 8.791209% lower confidence bounds. TET4, TET10/Taylor--Hood, QUD4, and URIS outputs and normalized histories were exact at 1/3/4 ranks; all 32 profiles were exact. Whole-step instructions fell 14.061012% median, while `get_thood_fs` calls and its allocation subtree fell 99.937656%; whole-process requested bytes/blocks fell 39.867466%/29.466628%, and peak live memory was flat. Evidence: `qualification/fluid-fs/performance-r1/controller-validation.txt` and `graphs/FLUID-FS-001`. |
| `RCS-SCALAR-001` | Scan scalar RCS rows directly instead of materializing vectors | `Code/Source/linear_solver/precond.cpp` | REJECTED | The direct scan is bitwise exact and passed the complete default/PETSc qualification, but whole-solver native gains do not clear 5%. Heat-CG produced 15/15 wins with a 1.798561% median and 1.444043--2.158273% uncertainty bounds. CEP spiral produced 8/8 wins with a 4.003242% median and 4.869281% maximum; eight below-threshold pairs mathematically cap any completed 15-pair median below 5%. All 46 paired scientific comparisons were exact. Job `41767242` was stopped before redundant rank/profile work once neither workload could pass acceptance, and only `precond.cpp` was restored byte-for-byte to parent SHA `9d79f8d`. No commit was made. Evidence: `qualification/rcs-scalar/performance-r1/controller-validation.txt`, `precommit-r1`, and `equivalence-r1`. |
| `CEP2D-WS-001` | Reuse `cep_2d` work arrays across Gauss calls | `Code/Source/solver/cep.cpp` | ACCEPTED | Retained and pushed as `d3cc8105`. The one-file change reuses a fully reset invocation-local workspace while preserving the public header/symbol and all arithmetic expressions. Source-locked job `41770116` rebuilt default/PETSc configurations, passed unit 1/1 and all 224 runnable integration tests with 24 expected backend skips, and produced checksum-verified binaries. Job `41770653` produced 15/15 AP and 15/15 BO native wins, all above 5%, with median reductions of 27.972561% and 18.920033% and lower confidence bounds of 27.773527% and 18.775848%. All 32 native output/history comparisons and 12 AP/BO/TTP/mixed-dimensional rank comparisons were exact. Job `41770654` completed all 64 profiles with 32 exact matched science conditions and zero errors in all 16 Memcheck runs. AP/BO median whole-step instructions fell 37.946872%/34.383959%, focused-kernel instructions fell 65.980460%/65.973891%, and whole-process allocation blocks fell 64.929748%/59.311713%; peak memory was flat. Independent reconciliation passed and external graphs were visually inspected. No Discussion #625 overlap exists. Evidence: `qualification/cep2d-workspace/profiles-r1/controller-validation.txt`, `precommit-r1`, `native-correctness-r1`, and `graphs/CEP2D-WS-001`. |
| `FLUID-STATE-001` | Reuse stabilized TET4 point state between momentum and continuity phases | `Code/Source/solver/fluid.cpp` | ACCEPTED | Retained and pushed as `750c2ddf`. The exact-shape-guarded one-file change stores four fixed point states, 41 doubles each, filled by momentum and consumed by an exact copy of the current continuity tail. Job `41779454` passed 21 dual-path rank cases with bitwise final `lR`/`lK`; job `41780261` passed unit 1/1 and all 224 runnable default/PETSc integration tests with 24 expected backend skips. Job `41780322` produced 45/45 native wins, with Newtonian/Casson/Carreau--Yasuda medians of 5.293202%/5.439331%/11.895911% and exact rank science; only Carreau--Yasuda has a clearly greater-than-5% lower confidence bound. Job `41780338` completed 96/96 matched profiles with exact science, retained the four pre-existing memory-tool contexts, flat allocations/peak memory, and whole-step instruction reductions of 7.611573%/10.120692%/13.727131%; focused construction fell 12.697689%/12.694438%/17.392981%. Supplemental job `41824941` completed 31/31 validations: its poisoned independent-state checker compared 189,837,872 named values bit-for-bit across 15 eligible runs, 7 topology/public fallbacks retained legacy routing, and 9 GCC 12 ASan/UBSan runs had no diagnostics. Thirty pinned hardware runs produced 180 counters with exact science and all 12 instruction/cycle pairs winning; instruction medians fell 7.793026%/8.843218%/12.910611%. The independent reconciler accepted the complete raw tree, and final source review found no issue. A fresh 19-branch public audit found no `fluid.cpp` overlap. Commit-labeled graphs and provenance are archived under `graphs/FLUID-STATE-001`. |
| `FLUID-CONST-ORDINARY-001` | Specialize ordinary constant-viscosity fluid work after state reuse | `Code/Source/solver/fluid.cpp` | REJECTED | Focused read-only re-scope against accepted-source `fluid.cpp` and its Newtonian profile closes this idea before implementation. The 5,532,226,029-instruction whole step attributes only 7,364,360 exclusive instructions to `get_viscosity`, an exact dispatch ceiling of 0.133% before retained assignments and derivative operations. The former 4.608% whole-statement ceiling requires a finite/sign contract absent from the solver, while the 12.904% mixed-line envelope includes expressions whose removal changes signed-zero, NaN, infinity, or rounding behavior. No observably exact one-file formulation has a defensible path above 5%; no source spike or Slurm job is justified. Evidence: `analysis/fluid-const-ordinary-scope-r1.txt`. |
| `RIS-LOOKUP-001` | Dense first-location cache for RIS map lookup | `Code/Source/solver/ComMod.h`, `Code/Source/solver/ris.cpp` | ACCEPTED | Retained and pushed as `8f71dbf7`. The generation-keyed, per-projection rank-local cache passed source-contract review, 890,064 semantic-model comparisons, compiled differential checks, optimized-machine-code inspection, and exact rank 1/3/4 science. Canonical native job `41920684` produced 15/15 wins, 13/15 above 5%, a 6.361474% median, and 5.595886% bootstrap and distribution-free lower bounds; the pooled native evidence is 45/45 wins with a 6.260297% median and lower bounds above 5%. Precommit `41936240` completed default/PETSc builds, unit 1/1, both differential forms, and all 248 integration cases: 224 pass and 24 expected skips. Clean profile job `41961420` completed exit zero with all 64 profiles and 32 unique workers. All eight whole-step instruction pairs exceed 5%, with a 7.671195145% median; the RIS-region median is 96.513541799%, and every candidate focused sample eliminates all in-range `find_loc` instructions. Science and histories are globally exact, normalized restarts are exact within each tool, all Memcheck signatures are exact, and every Massif pair is bounded. The 916-file profile tree hash is `686a25e640a24eacd027e80a5b1525ce397bc38560784f96df70e5ae12cbc532`. The 47-test final validator passes Python 3.6.8 and 3.12.1 and accepted the complete retained evidence; its validator/test/summary hashes are `4d998d3e51969f28031f4798c3939787e63e95a6f7bd246b322327fdc3102a7c`, `aaca601d259b64a242c3dc6830d921eab85546e52691d73f7831d7739ac23889`, and `745b2cea6620d55430318ceaeda609f34d534c364c9b6b1bbde687cd1eeaca0b`. Final production review found no issue, and a current 19-branch/Discussion 625 refresh found no public RIS overlap. Three commit-labeled PNG/SVG graph pairs passed original-resolution visual inspection. Evidence: `qualification/ris-lookup/long-native-r1`, `qualification/ris-lookup/precommit-r1`, `qualification/ris-lookup/profiles-r1`, `qualification/ris-lookup/restart-diagnostic-r1`, `analysis/ris-lookup-final-validation-r1.json`, `analysis/ris-lookup-qualification-matrix-r1.txt`, `analysis/ris-lookup-machine-code-r1.txt`, `analysis/ris-lookup-three-run-pooled-r1.txt`, `overlap/fluid-ris-refresh-20260904.txt`, `graphs/RIS-LOOKUP-001`, and `lanes/ris-lookup/checks`. |
| `FLUID-GNN-001` | Alias duplicate stabilized-fluid GNN results within each phase | `Code/Source/solver/fluid.cpp` | REJECTED | Bitwise exact, but the direct whole-fluid projection is below 0.5% instructions and about 0.15% native. Evidence: `analysis/fluid-gnn-dedup-r1.txt`, job 41545108. |
| `FLUID-GNN-002` | Cache GNN geometry across the two stabilized phases | `Code/Source/solver/fluid.cpp` | REJECTED | Bitwise exact, but only about 0.06% incremental native benefit beyond the simpler alias while adding per-Gauss storage and shape-specific control flow. Evidence: `analysis/fluid-gnn-phase-cache-r1.txt`, job 41545349. |
| `FLUID-LOOP-001` | Merge stabilized momentum and continuity Gauss loops | `Code/Source/solver/fluid.cpp` | REJECTED | Exact in the spike, but the merge adds structural risk for only about 0.08% incremental native benefit over separate-loop reuse. Evidence: `analysis/fluid-stabilized-merged-loop-r1.txt`, jobs 41546369 and 41546560. |
| `FLUID-BUF-001` | Hoist element work-buffer construction | `Code/Source/solver/fluid.cpp` | REJECTED | Removes allocations but projects only about 0.19% native and 0.22% whole-step instruction benefit. Evidence: adjacent-buffer section of `analysis/fluid-fs-hoist-r1.txt`, job 41544504. |
| `FLUID-HESS-001` | Skip physical Hessian solves for linear TRI3/TET4 | `Code/Source/solver/fluid.cpp`, `Code/Source/solver/nn.cpp` | EXCLUDED | Exact and fast in isolated/cumulative tests, but overlaps the linear-element/function-space territory reserved by the active external performance work. Jobs 41547933, 41548521, and 41552532. |
| `FLUID-RIS-001` | Replace per-Gauss owning default RIS vectors with stack-backed storage | `Code/Source/solver/fluid.cpp` | SUPERSEDED | The scratch form measured about 2.7% native improvement and did not independently meet the target. The safer pointer-core form was later accepted as `c551c21b`. Jobs 41550419, 41571378, 41580066, and 41622097. |
| `FLUID-CTX-001` | Cache fluid material and time-integration context per element | `Code/Source/solver/fluid.cpp` | REJECTED | Corrected candidate passed exactness but produced no measured native gain. Jobs 41551590 and 41551913. |
| `FSILS-COMM-001` | Grow-only communication buffers for repeated FSILS vector exchange | linear-solver communication path | REJECTED | Exact at two ranks but produced no measurable whole-solver gain. Jobs 41547405 and 41549602. |
| `SPAR-CLEAR-001` | Skip sparse-matvec output clearing for fixed DOF shapes | `Code/Source/linear_solver/spar_mul.cpp` | REJECTED | Exact candidate was statistically flat in native timing. Jobs 41536974 and 41551673. |
| `LHSA-FAST-001` | Scalar global-assembly fast path | `Code/Source/solver/lhsa.cpp` | REJECTED | Focused Callgrind improved 26.73% in instructions, but the 11-pair whole screen regressed native solver timing by roughly 0.5–0.6%. Jobs 41629473 and 41635667. |
| `CMM-TAN-001` | Cache CMM tangent scalar loads and addresses | `Code/Source/solver/cmm.cpp` | REJECTED | Byte-exact, but focused instructions increased 3.79% despite lower data traffic. Job 41631404. |
| `ADDBC-LAZY-001` | Lazily allocate coupled-boundary work arrays | `Code/Source/linear_solver/add_bc_mul.cpp` | REJECTED | Microbenchmarks removed allocations, but the profiled whole-step CPU ceiling was below 1%; it did not justify product validation under the greater-than-5% priority. Jobs 41536713, 41536984, and 41570646. |
| `GMRES-HESS-001` | Cache GMRES Hessenberg column/index values | `Code/Source/linear_solver/gmres.cpp` | REJECTED | Bitwise spike completed, but whole-profile bookkeeping attribution did not support a greater-than-5% product candidate. Jobs 41539557 and 41539889. |
| `GNN-SCALAR-001` | Dimension-specialized scalar caching inside GNN | `Code/Source/solver/nn.cpp` | REJECTED | Failed bitwise validation before native or Callgrind measurement. Job 41539537. |
| `GN-NXX-001` | Specialized/direct physical-Hessian transform | `Code/Source/solver/nn.cpp` | REJECTED | Exact reuse variants were studied, including conditioning sweeps, but numerical-risk and whole-case leverage did not justify advancement. Jobs 41537987 and 41538084. |
| `CMM-BOUNDARY-001` | Direct boundary accumulation and non-owning boundary views | `Code/Source/solver/cmm.cpp` | REJECTED | Strong isolated allocation/microkernel results, but no representative greater-than-5% whole-case basis. Jobs 41540923 and 41540924. |
| `CONTACT-BROAD-001` | Optimize contact broad-phase construction | `Code/Source/solver/contact.cpp` | REJECTED | `construct_contact_pnlty` is only 3.50% inclusive and 1.27% exclusive in the shell-valve profile, so even removing it cannot clear the target. |
| `POST-FUSE-001` | Fuse repeated structural derived-output traversals | `Code/Source/solver/vtk_xml.cpp`, `Code/Source/solver/post.cpp` | REJECTED | Four `post::tpost` passes total 6.33% in a deliberately worst-case one-step/save-every-step run. Even an unrealistically free three-pass removal is only 4.74%; normal output intervals dilute it further. |
| `FLUID-BC-FS-001` | Hoist weak-boundary `get_thood_fs` setup | `Code/Source/solver/set_bc.cpp` | REJECTED | The measured whole-step ceiling is only 0.58%. |
| `FLUID-CARREAU-001` | Reuse Carreau derivative algebra | `Code/Source/solver/fluid.cpp` | REJECTED | The attributable ceiling is 4.35% and the rewrite would alter exceptional and rounding behavior. |
| `BICGSS-TEMP-001` | Remove remaining scalar BiCGStab temporaries | `Code/Source/linear_solver/bicgs.cpp` | SUPERSEDED | The hotspot is real but is the scalar sibling of the vector-update work already retained in `193f2f98`; it is not a distinct area. |
| `DISCOVERY-FLUID-001` | Fresh fluid-family hotspot scan | fluid/FSI profile set | COMPLETE | Read-only scan completed. Promoted `FLUID-WEAK-NNX-001` and `FLUID-STATE-001`; strengthened `FLUID-FS-001`; rejected lower-ceiling and overlap-owned alternatives. |
| `DISCOVERY-SOLVER-001` | Fresh non-fluid solver/assembly hotspot scan | solver-packed profile set | COMPLETE | Read-only scan completed. Promoted `RCS-SCALAR-001` and `CEP2D-WS-001`; rejected contact and duplicate/external solver areas. A 2026-09-03 idle-time delta scan independently surfaced only those same two areas, confirming there is no new candidate to queue: `CEP2D-WS-001` is already accepted and `RCS-SCALAR-001` is already rejected below the native threshold. |
| `DISCOVERY-INFRA-001` | Fresh containers, orchestration, mesh, I/O, BC, and backend scan | all solver source files outside the two physics lanes | COMPLETE | Read-only dynamic and static review completed. No new infrastructure candidate plausibly clears 5%; the closest output-pass fusion has an optimistic 4.74% ceiling. |
| `DISCOVERY-COVERAGE-002` | Reconcile every solver-directory file and remaining dynamic gaps | all 208 files under `Code/Source/solver` plus retained profiles | COMPLETE | Confirmed the prior 196-file count covered only `.cpp`/`.h`; reviewed the 12 supporting files too. No still-unexplored candidate has a defensible greater-than-5% path. OneD scratch reuse remains profiling-only because its repository case is not registered in pytest and names a developer-local macOS library that is unavailable on the cluster. |

## Latest RIS qualification update

- 2026-09-04 00:29 PDT: Commit `8f71dbf79353d8bf8786274ad2ac84c120d751cf`
  contains only the qualified `ComMod.h` and `ris.cpp` changes with the required
  Zachary Sexton author and committer identity. The exact object is verified at
  the remote performance branch tip. Three commit-labeled PNG/SVG graph pairs
  were generated under `graphs/RIS-LOOKUP-001` and passed original-resolution
  inspection; the eight-file graph hash manifest digest is
  `338d5d8adc08e15ad0c1b6fe1af5231acb52417ee1d0e020cc87d6d20c13725f`.
- 2026-09-04 00:15 PDT: Clean replacement profile job `41961420` completed
  exit zero in 2:22:06 with all 64 expected profiles, 32 unique workers, and
  empty worker error streams. All eight whole-step instruction pairs exceed
  5%, with a 7.671195145% median; the RIS-region median is 96.513541799%, and
  every focused candidate has zero in-range `find_loc` instructions. All
  parent/candidate science and history evidence is exact, normalized restarts
  are exact within each tool, all eight Memcheck signature pairs are exact,
  and every Massif pair is bounded. The profile tree contains 916 files and
  has relative hash
  `686a25e640a24eacd027e80a5b1525ce397bc38560784f96df70e5ae12cbc532`.
  The corrected 47-test cross-stage validator passes both required Python
  runtimes and accepted the entire retained tree; its summary hash is
  `745b2cea6620d55430318ceaeda609f34d534c364c9b6b1bbde687cd1eeaca0b`.
- 2026-09-03 21:46 PDT: A separate read-only recomputation of all preserved
  raw profiles gives a 7.674712236% whole-step instruction median,
  96.513541799% RIS-region median, and 100% in-range lookup reduction in all
  eight replicates. The eight Memcheck pairs preserve the same four contexts
  and signatures, and every Massif pair is within bounds. Clean job `41961420`
  subsequently completed and superseded this provisional profile evidence.
- 2026-09-03 21:39 PDT: Raw-restart diagnostic `41958780` completed all eight
  matched runs with exit zero. Native and both Callgrind modes serialize
  606,016 bytes while Memcheck serializes 530,273 bytes; outside the CPU-time
  field the entire common prefix is exact, the longer suffix is all zeros,
  and every `iNorm` is identical. Parent/candidate normalized restarts remain
  exact in every mode. Diagnostic tree hash:
  `54ef1f499d0274de7fccb942de151603b303da3e64c5b7612768ec6086a1ca81`.
  Diagnostic report hash:
  `41bc9d7cfb85c2a2abb54907ac0b9f1cd939d23d3d3739bd7e19420086e4cb93`.
  The test-first contract now preserves global science/history equality while
  enforcing restart equality within each tool across both variants and all
  replicates. All 17 downstream and 46 final-validator tests pass on Python
  3.6.8 and 3.12.1. Current profile/downstream/validator/test hashes are
  `a037421fe747fd5aec0f0d66cb87b4c3ab3141ccacca5b2596e93358c79b1ef8`,
  `a5a124cc46dbc5e9d0d08ee52177add4d717ce70ed7d533a6771098db96dbf9c`,
  `276b479dab87c44f9acfb4f8b851e8d2392a1a5ea84ae0fe243b43358e29fdb1`,
  and `227a994e8b76f35a6b188ef52858e7604494fec8164ae4fd85cfe0a6d4383971`.
  The 907-file, 387 MB failed postvalidation tree is preserved at
  `profiles-postvalidation-failed-41942474`, relative tree hash
  `6da0149cc3b514ac996ad5d0b59b162b543d5d9160d12d4a27fe943b2b634e56`.
  Replacement profile job `41961420` was held-ledgered, metadata-verified,
  and released with the unchanged one-node/eight-CPU/20 GB SKX mail contract.
- 2026-09-03 21:18 PDT: Profile job `41942474` completed all 64 runs and all
  32 unique workers with empty worker errors, then exited one during the
  over-strict postvalidation cross-tool restart comparison. Science manifests
  and histories are globally exact across all 64 profiles. Restart digests
  split deterministically by profiler class, with parent/candidate exact in
  every tool. Red harness and validator tests reproduce the missing per-tool
  scope. Raw-restart diagnostic `41958780` is ledgered, held-verified, and
  running on eight SKX CPUs; its completed native pair differs only in the
  normalized CPU-time field and has identical iNorm values.
- 2026-09-03 19:20 PDT: Corrected profile job `41942474` completed wave one
  with 16/16 successful profiles and empty worker error streams. Wave two
  started with eight new unique PIDs pinned to the exact repeated cgroup CPU
  sequence 0,2,4,6,8,10,12,14. A fresh public-overlap refresh reconciled all
  19 current contributor performance heads and found no branch-specific
  `ris.cpp` or `ComMod.h` change; report hash
  `5d3f8f6d10cc5043b66eba26d532632dcc05ce3695d9ffbad5d54a4f7ac09c97`.
  Independent final production review found no critical, important, or minor
  issue and returned ready after profile validation; review hash
  `351d7d963b9e0acca51f129e470034df0cf95d73f970b29b3c6d673f7cd6db00`.
- 2026-09-03 18:40 PDT: Profile job `41940570` was stopped after 3m20s
  with zero completed profiles when four workers proved the scheduler's
  detailed CPU IDs are abstract topology indices, not OS-visible affinity IDs.
  Its partial tree is preserved as `profiles-failed-41940570`, hash
  `89375ef60f6b0d874be40e83252b31fe42e92a45af6fb1ce1d6b2aaf25eb7bea`.
  A new red/green behavioral regression now drives pinning from the batch
  process's cgroup CPU list while retaining scheduler IDs as separate evidence.
  All 16 downstream and 45 final-validator tests pass on both required
  runtimes. Current profile/downstream/validator/test hashes are
  `4f8c9572bcd96335ab2153c75d2ee63806c6c504560009897827fc07039cf03c`,
  `87c290c56961aa906f9c3c6d026b5e103dea3c6dc19b5a59d2cbff1876c20846`,
  `cce9015ae5688a9adeefb9b93d43180df9b2da0ab00837a0ea49ebb4e85d62b4`,
  and `7a9916b50a168db623ea013aa1d3629af8fd70764309db4ccf665a153c24682d`.
  Corrected job `41942474` was submitted held, immediately ledgered, fully
  verified, and released at normal priority with the same resource and mail
  contract.
- 2026-09-03 18:25 PDT: Held-verified job `41936240` completed on
  `sh02-10n28` with exit zero. Both rebuilds, unit 1/1, the normal and
  instrumented compiled differential tests, all 248 integration cases, binary
  manifests, dependencies, and the 11-symbol public RIS surface passed. The
  integration result is 224 passed and 24 expected backend skips; differential
  outputs are exact and both error streams are empty. The archived default
  binary hash is
  `ed7ba17ecd1aa0036ce40f4975c3c6a41f610fda4e290fa8d56b5b2e5ad811dd`.
  A red/green scheduler-artifact regression now proves the profile harness
  records detailed job metadata before pinning 32 workers in four waves to the
  eight assigned SKX CPUs. All 15 downstream tests and all 45 final-validator
  tests pass on Python 3.6.8 and 3.12.1. Current profile/downstream/final-
  validator/test hashes are
  `0810e7243756ca7e0a39c5c4f26b7d01a00a9a0a0c8fd1332245fd0044ec9643`,
  `3fae4b868895ce9927f77df6fae1e1fb629094121b240760b330400581aec178`,
  `ceb901fbf48c9abddce9aabdd0f1eb1b372598ea2c667732da6ce32919b74489`,
  and `7089a29552c9bf896c5b2c06accd131ef2e4102ce7d7ab448a6c20fb2c7c93f6`.
  Profile job `41940570` was submitted held, immediately ledgered, fully
  verified, and released at normal priority for one shared eight-CPU SKX
  allocation with 20 GB and the required mail notifications.
- 2026-09-03 17:50 PDT: Job `41934718` passed the default build, unit 1/1,
  source/semantic gates, and native-test compilation before five uncovered bare
  text-matcher calls in the native driver failed in the clean job path. The
  partial root is preserved as `precommit-failed-41934718`; no native test ran.
  An expanded red/green clean-path regression covers the driver, which now uses
  equivalent baseline absolute matchers. All 13 downstream tests pass on
  Python 3.6.8 and 3.12.1. Driver hash:
  `0c64553b4321237e258d56d85fa4fad63feac710fe0a556164e4e31062afbcb5`.
  Held-verified retry `41936240` is running on `sh02-10n28` and has passed
  default build, unit 1/1, and normal plus sanitized native differential runs
  with identical expected output and empty error streams. Current precommit/
  profile/downstream-test hashes are
  `d4fb1dedfa8ba3d318dd73633fcb877d5bb9f9840876d28bb917296ebe832c71`,
  `508058690953a1e2b1b18982203053e28ef6f29a6eca09e602de185e17349974`,
  and `d66fb3109c2dcc2155ceb87a2d643580408c771949b35aa0d35d3b6718a75046`.
- 2026-09-03 17:35 PDT: The exclusive precommit request had no viable
  eight-hour capacity path. A first shared-capacity correction exposed that
  the MLN co-tenant reserves all scheduler-visible memory despite eight idle
  CPUs per node. The correctness-only gate was therefore retargeted test-first
  to eight shared SKX cores, where memory is available and an eight-core
  opening is forecast. It retains 20 GB, the four-hour limit, both rebuilds,
  every correctness check, exact source locks, and required mail. All 13
  downstream tests pass on Python 3.6.8 and 3.12.1. Final replacement
  `41934718` was submitted held, immediately ledgered, fully metadata-verified,
  and released. Predecessors `41927570` and `41933057` were canceled only after
  their verified replacements were eligible; both had zero elapsed time, zero
  allocated nodes/CPUs, and no assigned node. Current precommit/profile/
  downstream-test hashes are
  `eba47a9e22da152c8264279584f768ede9a7f8f54021f1c1a5b9269d6cb53f3d`,
  `1dec064a088853b261ed547f9fd262cf7b958c684e09d4e548349d01783cedbf`,
  and `d1d6882aed622f9d514a2b0539adc561686733537bc7f7301fb9b52fda66e9db`.
- 2026-09-03 16:39 PDT: Accepted the canonical long-native gate from job
  `41920684`. Its exclusive node recorded one-minute loads 0.07, 0.06, and
  0.05 before timing. The 60-step workload produced 15/15 wins, 13/15 above
  5%, a 6.361474% median, and 5.595886% bootstrap and distribution-free lower
  bounds. Rank 1/3/4 VTK, integral, history, and normalized restart comparisons
  are exact, all solver error streams are empty, and the independent
  reconciler passed. Analysis JSON hash:
  `77d0f25d69e3956b60c809c88ecd9f600efd92161f78f60f442e50c6d9c9c347`.
  Full precommit job `41927570` is ledgered, held-verified, and eligible for an
  exclusive 32-CPU MLN node, 20 GB, and four hours.
- 2026-09-03 15:54 PDT: Replaced the racy pre-allocation release watch with a
  test-first in-allocation quiescence gate. Once the scheduler grants the
  exclusive node, the harness requires three consecutive one-minute load
  samples no greater than 1.0 and fails before warmup under sustained high
  load. Twelve downstream and seven controller tests pass on Python 3.6.8 and
  3.12.1. Ledgered, held, and fully verified job `41920684` is now unbound and
  eligible at priority 1046189; predecessor `41916253` was canceled only
  afterward with zero allocation. Current long/precommit/profile/downstream-
  test hashes are
  `f445e4093ce0c7cbb8fbabdcca3a6ceea56f19d02ff02a7a5fdee96d537dc820`,
  `168b2ca6033c5861b8b56b71ec28850ceb5e033b4ea38db057e7524995e0ccac`,
  `4b74cfc277167ac915a2e7a72886d902c0a06f205f213b9ac2f0be1301ebcd6b`,
  and `61e6f96fe861bbd61280ba47ef8d971d4603b27d4dd222afa886e1ac1d3d0f33`.
  The canonical root is absent and production source hashes remain locked.
- 2026-09-03 15:29 PDT: Quiet-node job `41908735` completed with all 15
  timing pairs favoring the candidate and exact rank 1/3/4 VTK, integral,
  history, and normalized restart comparisons. Its 5.665025% median remained
  promising, but 4.861111% bootstrap and 4.795370% distribution-free lower
  bounds were independently inconclusive. Its complete root is preserved as
  `long-native-inconclusive-41908735`. A test-first correction lengthened only
  the timed case from 20 to 60 steps, with matching step-60 VTK/restart
  cadence and a one-hour limit, to reduce fixed process-launch and setup
  influence. The unfiltered 45-pair pool across all three completed runs has
  45/45 wins, a 6.260297% median, and conservative lower bounds of 5.665025%
  and 5.583333%; report hash
  `c39e9d2175105aa52356aad554cdd215b99ac1ef97d3f379eaec2be3c2eb6772`.
  It remains corroborating because the pairs are nested in three runs on two
  nodes. Eleven downstream and seven controller tests pass on both
  required Python runtimes. Current long/precommit/profile/downstream-test/
  controller hashes are
  `da7f9adcd29cacd6e4952066ac0f019d8b34d29ca347f53cbf1aa36684e4e118`,
  `4f0b0878455578dd1d3a98e76a978011cde8899caf0031518b226c3a5f6542ad`,
  `642396384d81567b3842418d923d29ac5e1df43c8359734c89c5b98f1a585a48`,
  `1424e8362cf107cf993a7f4a677e29c7403dde1b41f07adc8ebf73348f43e040`,
  and `2dafc373b7104136435422b6c0d4f1f3b85fce27345d21b6fbf6013e12fbd366`.
  Job `41916253` is ledgered, verified, node-bound, and held at zero runtime
  under controller session `13769` after its first release window ceased to
  be fully idle. The canonical root is absent and production hashes are locked.
- 2026-09-03 14:54 PDT: A focused red test showed the release controller
  accepted an idle node even when its observed CPU load was 13.11. The
  repaired controller now requires numeric CPU load no greater than 1.0 at
  candidate selection and every stability/release checkpoint. Ten downstream
  tests plus all seven controller tests pass on Python 3.6.8 and 3.12.1.
  Controller/test hashes are
  `0a0ad8186abc9f0e36dc51166ff0907f4d733428cfc67f28b5b08e2522fa5c40`
  and `f167b6ab5d3babe735b985a19a43f5b87e8cbdec3352946d2cdebd56c1c771e7`.
  Job `41908735` remains held at zero allocation under restarted session
  `63865`; the canonical root remains absent.
- 2026-09-03 14:39 PDT: Job `41906282` completed the corrected harness with
  exit zero and exact VTK, integral, history, and normalized restart evidence
  at ranks 1, 3, and 4. All 15 wall pairs favored the candidate with a
  6.260297% median, but 4.376548% bootstrap and distribution-free lower bounds
  made this repetition inconclusive. The node was drained immediately after
  the job by its health monitor for excessive background load. The complete
  root is preserved as `long-native-inconclusive-41906282`. Fresh ledgered job
  `41908735` is verified and held at zero allocation under quiet-node controller
  session `63865` until another healthy idle SKX node is available. The
  canonical root is absent and production source hashes remain locked.
- 2026-09-03 14:24 PDT: The relocation controller observed a stable idle SKX
  node, submitted and ledgered held replacement `41906282` bound to
  `sh02-07n59`, verified its account, resources, command, mail settings, and
  node constraint, canceled source job `41904070` at zero runtime, and released
  the replacement. Job `41906282` began the clean canonical retry at 14:19 PDT;
  the controller exited successfully after confirming allocation. Production
  source and diff hashes remain locked while the job runs.
- 2026-09-03 13:32 PDT: Job `41898003` reached a fully idle SKX node but
  stopped with exit 127 before any solver invocation because the post-purge
  environment lacked the harness's bare text-search executable. Its minimal
  artifact root is preserved as `long-native-failed-41898003`. A focused red
  test reproduced the dependency, and the repaired long harness now uses
  `/usr/bin/grep -Fq`; the same portability contract replaced three equivalent
  checks in the precommit harness. Nine downstream regressions and six
  controller regressions pass Python 3.6.8 and 3.12.1. The exact long,
  precommit, profile, controller, and regression-test hashes are
  `0150b2b92806a7f4b30afc08e4d946ce8c1fdebef1c5fad0b22b6e6363f8ef17`,
  `bc9cf9443250dc578977dc5fb4cdff170b36ba4189ce272e0b1a53989578ac31`,
  `ce2bcb9ac05cd2b05f5410d760f9feaba1dd2ec8151a87b79b51079941690fc6`,
  `bb1975b3eaa07735530c6ddfb49a953519bd79d61dda4d24b2ec860d1af2ea3f`,
  and `5384802bef0de9ee152ae755f4e7a08dd8302ef8c648a5f4f5a091408e7efa7a`.
  Current job `41899495` was verified and released only after stable idle
  samples, then safely re-held at zero runtime when unrelated work claimed
  the node. Controller session `55072` remains active; the canonical root is
  absent and both production source hashes remain locked.
- 2026-09-03 11:43 PDT: The scratch-only final artifact validator now has 44
  behavioral tests green on Python 3.6.8 and 3.12.1. It independently checks
  the long-native verifier, full-suite archives, exact 64-profile coverage,
  raw-derived counters, exact threshold arithmetic, output immutability, and
  deterministic summary evidence. Specification round 5 and quality round 3
  returned READY. Validator/test hashes are
  `74ae68d2e4665d10bd743a87f6b7f222e305b6e2c800ac4832a79d3db5385479`
  and `9a5a79788e495443be450b24407ec339f2a2ab1cac0c6e24cf3e7a7b29d38a43`.
  Original profile science/restart payloads are not retained, so the verifier
  explicitly limits its claim to archived manifest and normalized evidence.
  Two later idle windows were taken by unrelated work before this campaign
  allocated; current job `41879818` remains held at zero elapsed with no
  canonical artifact root under controller session `95200`.
- 2026-09-03 08:52 PDT: Downstream precommit and profile harnesses now lock
  the full native reconciler, enforce the exact whole-tree two-file diff,
  bootstrap clean shells, and atomically claim result roots. Eight regressions
  pass on both required Python runtimes, including corrupt-input, duplicate-
  contention, and stale-temporary-state cases. Final specification and quality
  reviews found no issue. Native retry `41851116` remains held at zero
  allocation under the verified relocation controller; its canonical artifact
  root is absent and both production source hashes remain locked.

## Solver-file coverage reconciliation

- Inventory: all 208 files under `Code/Source/solver` are accounted for. The
  original 196-file inventory covers 170 top-level and 26 nested `.cpp`/`.h`
  files. The additional 12 are six `CMakeLists.txt` files, `README.md`,
  `README-trilinos.md`, `Parameters.cpp.orig`, `mueluOptions.xml`, `SPLIT.c`,
  and `Core/PlatformSupport.inl`.
- Supporting-file disposition: `Parameters.cpp.orig` is a tracked, unreferenced
  385-line legacy copy distinct from the active 3,636-line parser and is not a
  build input. All six `CMakeLists.txt` files are active metadata; `SPLIT.c` is
  built and called by `distribute.cpp` but contains startup/repartition work;
  `Core/PlatformSupport.inl` is active MPI/error/stack support; and
  `mueluOptions.xml` is a runtime Trilinos example/configuration in an excluded
  area. The two README files are documentation. None contains an unreviewed
  in-scope step hotspot.
- Direct profile/debug-path coverage: 112 files are referenced by the 148
  retained Callgrind outputs. The remaining 84 files received paired static
  review or were routed to the relevant physics/external-ownership audit.
- Lane partition: 98 container/orchestration/mesh/I/O/BC/backend files were
  reviewed by the infrastructure lane; the other 98 were reconciled through
  the fluid, non-fluid physics, retained-history, and explicit-overlap lanes.
  The 12 supporting files received a fresh static audit. No production or
  build-support file is unaccounted for.
- Test-case coverage: the repository contains 105 solver XML inputs in 98
  case directories. The current packed-profile provenance covers 34 distinct
  solver inputs; retained historical artifacts add the CMM and URIS lanes, so
  every top-level physics family has dynamic evidence. The complete
  default/PETSc regression job remains the correctness backstop for the full
  available suite.
- Remaining dynamic-evidence gaps: activated remeshing, production-scale
  OneD coupling, and representative Trilinos profiles. PETSc breadth job
  `41755468` and ZeroD breadth job `41757895` are reconciled. The ZeroD run
  completed all 32 profiles and 64 exact scientific comparisons without
  revealing a distinct greater-than-5% candidate; externally owned backend-
  reuse work remains excluded.

## Accepted and retained optimization areas

These commits are already explored, validated, and retained. Do not rediscover
their underlying ideas as new candidates.

| Commit | Retained idea | Primary area |
|---|---|---|
| `8bdd57ea` | Reduce RK vector temporaries | `ionic_model.cpp` |
| `5d0658a3` | Reuse CEP node-state buffers | `cep_ion.cpp` |
| `58221df2` | Reuse TTP gating kinetics | `ionic_ttp.cpp` |
| `f595e86f` | Use BLAS for FSILS vector updates | `omp_la.cpp`, build configuration |
| `d9dbda0f` | Use BLAS for FSILS vector dot products | `dot.cpp` |
| `78ab6419` | Reduce sparse-matvec output traffic | `spar_mul.cpp` |
| `60b42eea` | Use BLAS for FSILS scalar dot products | `dot.cpp` |
| `7030e432` | Use BLAS for FSILS scalar vector updates | `omp_la.cpp` |
| `a43df110` | Use BLAS for FSILS scalar norms | `norm.cpp` |
| `ed2e65ab` | Use views for scalar GMRES basis columns | `gmres.cpp` |
| `c01532d7` | Use views for heat shape data | `heats.cpp` |
| `318a2e2c` | Use scalars for heat gradients | `heats.cpp` |
| `5b4c9957` | Accumulate scalar sparse rows locally | `spar_mul.cpp` |
| `90705daa` | Use BLAS for FSILS scalar vector scaling | `omp_la.cpp` |
| `ece315c0` | Keep GNN work arrays on the stack | `nn.cpp` |
| `4849e44f` | Keep GN_NXX fixed work arrays on the stack | `nn.cpp` |
| `76fa3c62` | Keep 2D fluid momentum work arrays on the stack | `fluid.cpp` |
| `a0492573` | Keep 2D fluid continuity work arrays on the stack | `fluid.cpp` |
| `78d6168c` | Keep 2D Stokes continuity work arrays on the stack | `stokes.cpp` |
| `0a8af388` | Keep 2D Stokes momentum work arrays on the stack | `stokes.cpp` |
| `04f113ca` | Keep 3D linear-elasticity work vectors on the stack | `l_elas.cpp` |
| `4d9ea7b5` | Keep initial CMM work arrays on the stack | `cmm.cpp` |
| `a791521c` | Keep remaining CMM work arrays on the stack | `cmm.cpp` |
| `9c003234` | Consolidate CMM nodal work storage | `cmm.cpp` |
| `3299e86e` | Use a view for scalar-transport shape data | `heatf.cpp` |
| `c9015e2c` | Use scalars for 3D scalar-transport state | `heatf.cpp` |
| `843aebe5` | Reuse scalar-transport directional workspace | `heatf.cpp`, `heatf.h` |
| `e33c1678` | Use a view for scalar-transport gradients | `heatf.cpp` |
| `ffc9e9a5` | Cache 3D scalar-transport row invariants | `heatf.cpp` |
| `6a1c576c` | Reduce global-assembly indexing overhead | `lhsa.cpp` |
| `c089b93e` | Reduce integration temporary allocation overhead | `all_fun.cpp` |
| `4f636c01` | Reduce surface-normal temporary allocations | `nn.cpp` |
| `da33d313` | Defer matrix-multiply diagnostic construction | `mat_fun.cpp` |
| `5ad3662a` | Avoid redundant 3D fluid workspace initialization | `fluid.cpp` |
| `f124cd6e` | Initialize 3D heat buffers at the first Gauss point | `heatf.cpp` |
| `c551c21b` | Read URIS valve velocity without temporary vectors | `fluid.cpp` |
| `193f2f98` | Avoid BiCGStab vector-update temporaries | `bicgs.cpp` |
| `958bd1e7` | Reuse equal-order pressure shape values | `set_bc.cpp` |
| `a3b3f1dd` | Reuse fluid function spaces across elements | `fluid.cpp` |

## External ownership and explicit exclusions

Do not queue work that overlaps these active Discussion #625/dseyler areas:

- fixed-size double-dot, tensor, and dyadic products;
- templated or fixed-size viscous-stress evaluation;
- structural/FSI scratch buffers and solid Gauss-invariant hoisting;
- `bar_to_iso` reformulations;
- Trilinos hierarchy/object reuse;
- `mshType::lShpF`, constant shape-gradient gating or hoisting, and wedge
  handling across the solver paths listed in issue #634;
- combined branches composed from those changes.

The latest focused overlap inventory is
`/scratch/users/zsexton/svmp-efficiency-review-20260829/overlap/fluid-ris-refresh-20260904.txt`.
Discussion #625 and all 19 current contributor performance heads were
refreshed through 2026-09-04 00:18 PDT. Issues #632, #633, and #634 retain
their previously reconciled ownership boundaries.

## Update log

- 2026-09-04 00:29 PDT: Marked `RIS-LOOKUP-001` accepted after the exact
  two-file commit was pushed and remotely verified and all three graph pairs
  passed visual inspection. No unvalidated implementation-ready candidate
  remains after the exhaustive 208-file audit and final delta scan.
- 2026-09-04 00:15 PDT: Job `41961420` completed the clean source-locked
  profile gate and the independent cross-stage validator accepted all native,
  precommit, and profile roots. All numerical, performance, memory, and raw-
  artifact acceptance conditions are now complete; only the fresh overlap
  refresh, two-file commit, external commit-labeled graphs, and push remain.
  A fresh public audit at 00:18 PDT found no RIS overlap in Discussion 625 or
  any of the 19 current contributor performance heads. The only changed head
  since the previous audit touches nine unrelated structural/material files.
  Refresh report hash:
  `baa58f1a533ec6c109352ffeb34763bfb2296952e3fc83ad893268d5c8545ac5`.
- 2026-09-03 22:31 PDT: Reconciled a new read-only non-fluid optimizer delta
  scan while profile job `41961420` remained active. Its two strongest hits
  were the `cep_2d` workspace reuse already accepted as `CEP2D-WS-001` and
  the scalar RCS row scan already rejected as `RCS-SCALAR-001`; it found no
  third candidate with a defensible greater-than-5% path. No source, job,
  ledger, commit, or queue change is warranted from this duplicate scan.
- 2026-09-03 06:39 PDT: Job `41850414` allocated on the verified bound node
  but stopped before simulation with exit 127 because the pristine submission
  environment did not export the site module function. Its sole exit-code
  artifact is preserved under `long-native-failed-41850414`. A clean-shell
  reproduction was red before sourcing `/etc/profile` and then loaded the
  complete production module stack after the bootstrap fix. Corrected harness
  `b3dae4a525faec502c55b9af8e9bf5245990b2f9489d89fd90e832435e05c03c`
  uses the original clean artifact path. Current node-bound job `41851116`
  was automatically re-held at zero allocation when competing jobs entered
  its release window. Six-test relocation controller
  `4548fb8a71dd8733b0d6fe100af72e2250f5649dde31e7ca7594cd313b0d1ffd`
  remains active as session `95200` and may move only through another
  ledger-first held handoff.
- 2026-09-03 06:24 PDT: A truly idle node exposed that this site returns
  success while discarding post-submission requested-node updates. The held
  source job never allocated or became eligible. Zero-allocation probe
  `41848470` then proved that `--nodelist` at held submission persists the
  requested node; it was ledgered, inspected, and canceled without release.
  A five-test red/green controller suite now covers submission-bound handoff,
  unsupported discovery syntax, all-partition idle checks, metadata mismatch,
  ledger-write failure, missing ledger membership, and terminal failure after
  release. Verified controller
  `d3c4521e4095ccd9d1971a2e99eee572bc28d4c9d5319afbfaedc8806e7689d4`
  is active as session `49000`; `41831216` remains held at zero allocation.
- 2026-09-03 05:50 PDT: A delta-only C++ optimizer audit returned
  `COMPLETE-NONE`. No non-RIS profile evidence appeared after the prior scope,
  and no residual one- or two-file idea combines an observably exact design
  with a defensible whole-solver path above 5%. OneD remains profiling-only
  without an activated Linux fixture; remeshing has no attributable bounded
  hotspot; backend reuse overlaps archived public work. Evidence:
  `analysis/solver-residual-candidate-audit-r1.txt`.
- 2026-09-03 05:39 PDT: Closed three rounds of independent review for the RIS
  raw-artifact reconciler. Every demonstrated false-acceptance path received a
  failing regression before correction. The final 35-test tree passes Python
  3.6.8 and 3.12.1; the third review found no critical, important, or minor
  issue. The gate binds the exact nine-entry submitted manifest, recomputes
  wall and user statistics, validates all timing rows and solver streams, and
  requires restart cardinality to match ranks 1, 3, and 4. Job `41831216`
  remains safely held at zero allocation with no artifact root.
- 2026-09-03 04:55 PDT: Focused optimizer re-scope closed
  `FLUID-CONST-ORDINARY-001` without implementation. The current source hash,
  5,532,226,029-instruction Newtonian step, and 7,364,360-instruction exclusive
  `get_viscosity` cost were independently verified. The 0.133% exact dispatch
  ceiling is far below 5%; larger line-based envelopes require transformations
  that are not observably exact for the solver's current input domain. The
  candidate is now permanently rejected and consumed no Slurm allocation.
- 2026-09-03 04:25 PDT: Quantified the RIS cache memory boundary from the
  available case rather than treating flat first-screen RSS as sufficient.
  The repository fixture has one projection; restart stamps prove rank-local
  node counts of 3,156 at rank 1, at most 1,455 at rank 3, and at most 1,015 at
  rank 4. Its dense payload is therefore at most 25,248 bytes per process.
  General payload scales as approximately `8 * nbrRIS * tnNo` bytes per rank
  plus vector metadata. Eight matched Massif pairs remain mandatory, and the
  absence of a production-size multi-projection fixture is recorded as an
  explicit evidence limitation.
- 2026-09-03 04:18 PDT: Independent C++ review found no correctness blocker in
  the two-file RIS cache and kept its narrow native-performance ceiling and
  dense-memory cost as mandatory qualification risks. The future compiled
  differential verifier now independently exercises node-count,
  projection-count, and map-shape transitions through zero plus unchanged- and
  changed-generation same-shape map behavior. Focused re-review found no
  compile, bounds, tautology, or assembly-count defect. Production source and
  the held long-native harness remain byte-identical; compilation, full-suite,
  rank, restart, instruction, and memory evidence remain pending Slurm gates.
- 2026-09-03 03:54 PDT: Focused verification corrected one preliminary audit
  statement: `Parameters.cpp.orig` differs from the active parser rather than
  matching it. It is a 385-line tracked legacy copy with no code or build
  reference, so it remains inactive. A second pass confirmed the other 11
  support-file dispositions, including active CMake metadata, cold `SPLIT.c`,
  active exceptional-path platform support, runtime MueLu configuration, and
  two documentation files. The 208-file inventory and conclusion that no new
  candidate has a defensible greater-than-5% path are unchanged.
- 2026-09-03 03:50 PDT: A fresh C++ optimizer coverage audit corrected the
  solver inventory from 196 `.cpp`/`.h` files to 208 total directory files and
  explicitly reviewed the 12 omitted support files. The built `SPLIT.c` path
  is startup/repartition-only; `Core/PlatformSupport.inl` is MPI, error, and
  stack support; neither provides a step-level candidate. Retained profiles
  still leave activated remeshing, production OneD coupling, and representative
  Trilinos execution dynamically uncovered. Of these, only OneD allocation and
  solution-copy reuse fits two files. A two-face OneD case exists under
  `tests/cases/fluid/Tpipe_svOneD_2faces`, but no pytest function registers it,
  its XML names an unavailable developer-local macOS `.dylib`, and no Linux
  interface library was found in the scoped scratch tree or dynamic-linker
  cache. It therefore establishes no runnable whole-step ceiling and remains a
  scratch profiling lead, not a source candidate. No still-unexplored in-scope
  idea has a defensible path above 5%.
- 2026-09-03 03:45 PDT: Branch-history graph audit confirmed external graph
  packages for every retained optimization. The older `CEP2D-WS-001`
  package already had six checksum-valid PNG/SVG files and a candidate-source
  hash equal to retained commit `d3cc8105`, but its provenance named only the
  parent revision. Added explicit parent and retained full commit IDs; the
  refreshed provenance hash is
  `4695588c013cea3d1f1f4e976a7c7c63964d992ef6073931587361e1e727ab32`.
  The same audit verified all 42 branch commits use the required author and
  committer identity, every non-merge commit changes at most two files, and
  commit messages plus added lines pass the restricted-term gate.
- 2026-09-03 02:37 PDT: Optimized-machine-code inspection confirmed that the
  RIS cache helpers are inlined, valid steady-state lookup is direct, and no
  allocation is reachable outside rebuilds. Screen peak RSS is flat; the
  millisecond-rounded internal simulation clock independently reports a
  6.727883% median with 5.348715% lower bounds, but external timing remains the
  acceptance basis. Retry `41831216` is held until a full node is idle because
  the partition can otherwise requeue unrelated owner work. A fresh read-only
  scan of all reconciled profiles found no new fluid candidate plausibly above
  5%; the constant-viscosity specialization remains disposable-spike-only.
- 2026-09-03 02:22 PDT: First candidate screen `41829576` compiled cleanly,
  passed unit 1/1 and 890,064 lookup-model comparisons, and completed all 15
  native pairs with exact VTK, integral, and normalized-history science. All
  pairs won and the median reduction is 5.681818%, but the 4.610951% lower
  bounds require a longer lower-noise run. The job's exit one is a harness
  stop on raw restart bytes: `write_restart_header` intentionally serializes
  elapsed CPU time, and all non-restart outputs are exact. A self-tested
  record-aware normalizer and 20-step retry now cover that header explicitly.
  Final implementation review found no source defect and approved continued
  qualification.
- 2026-09-03 01:57 PDT: Independent RIS scoping returned READY for a bounded
  two-file experiment. It confirmed 277,500 hot scans and a 7.356609% exact
  whole-step instruction ceiling, but no native evidence. The required cache
  preserves logical row-major first matches, routes all negative and out-of-
  range queries through `utils::find_loc`, builds only from final rank-local
  maps, and keys validity to remesh counter, node/projection counts, and map
  shapes. The scratch implementation plan is complete and its source-contract
  test failed as expected on the unchanged baseline. Production retention
  remains conditional on bitwise/lifecycle gates and native improvement above
  5%.
- 2026-09-03 01:47 PDT: Final source review returned `READY TO COMMIT` with
  no finding. Commit `750c2ddf` retains only the qualified `fluid.cpp` change,
  has the required Zachary Sexton author/committer identity, and is verified
  at the remote branch tip. Four regenerated PNG/SVG pairs passed original-
  resolution visual inspection; their provenance records the commit, source,
  jobs, reconcilers, graph script, and file hashes while explicitly retaining
  weaker Newtonian/Casson confidence bounds. Marked `FLUID-STATE-001`
  accepted and promoted `RIS-LOOKUP-001` to the active implementation lane.
- 2026-09-03 01:40 PDT: Job `41824941` completed exit zero and closed every
  supplemental `FLUID-STATE-001` evidence request. The 31 validation rows
  comprise 15 eligible rank/model/permeability/URIS cases, 7 explicit
  TET10/Taylor--Hood/WDG/HEX8/2D/public-wrapper fallbacks, and 9 GCC 12
  ASan/UBSan cases. Direct poison-and-recompute instrumentation compared
  189,837,872 named scalar values bit-for-bit, and every scientific output was
  exact. All 30 pinned hardware runs and 180 raw counters reconciled; all 12
  instruction/cycle pairs won, with 7.793026%/8.843218%/12.910611% median
  instruction reductions. No sanitizer or checker diagnostic occurred. The
  standalone unit-tested reconciler accepted the artifacts, which are
  summarized at `qualification/fluid-state/supplement-r3/controller-validation.txt`.
  A fresh fetch of 19 public performance branches confirms no `fluid.cpp` or
  `ris.cpp` overlap. `perf/eigen-element-kernels` and the updated
  `perf/visc-stress-templated` share `ComMod.h` only for the same unrelated
  output-flag change; this shared-file fact is retained for the next RIS
  design review. Final fluid source review is in progress.
- 2026-09-03 01:11 PDT: Detailed optimizer review demoted
  `FLUID-CONST-ORDINARY-001` to disposable-spike-only. Signed-zero and
  non-finite propagation makes the naïve constant/zero-term specialization
  non-equivalent without a missing finite-state contract. The wholly removable
  instruction share is 4.608%; its optimistic mixed-line/native envelopes are
  12.904%/8.8%. RIS therefore remains the next implementation-ready target.
  Supplemental state job `41823641` proved build/link, hardware-event support,
  and 9,739,960 rank-one state-value comparisons before a rank-three OpenMPI
  slot-declaration failure. Retry `41824313` fixed the launcher, passed all 15
  eligible runs and equal-order TET10, then showed that the existing mesh is
  unsuitable when force-staged as Taylor--Hood. Generated affine TET10 smoke
  `41824872` passed all topology and solver checks in 28 seconds with zero
  stderr. Dependency-released retry `41824941` uses that fixture; no candidate
  scientific failure has occurred.
- 2026-09-03 00:52 PDT: Reconciled completed `FLUID-STATE-001` qualification,
  native, and profile evidence. Jobs `41780261`, `41780322`, and `41780338`
  respectively passed the complete available default/PETSc suite, produced
  45/45 native wins with exact multi-rank science, and completed 96/96 exact
  matched profiles. Native medians are 5.293202%, 5.439331%, and 11.895911%;
  whole-step instruction medians fall 7.611573%, 10.120692%, and 13.727131%
  for Newtonian, Casson, and Carreau--Yasuda. A read-only source/evidence audit
  found no implementation defect and requested direct all-41-value checks,
  explicit Taylor--Hood/WDG/HEX8/permeability/fallback counters, sanitizers,
  and hardware events. Initial job `41822025` was cancelled before allocation
  after static audit caught invalid adjacent-element IDs on the one-element
  wedge faces. Explicit WDG/HEX8 topology and adjacency tests now pass, and
  corrected source-hash-locked job `41822654` was queued for the supplemental
  matrix, then cancelled before allocation when its Milan-only constraint was
  found to be blocked by long jobs on all four matching nodes. Partition-wide
  exclusive-node job `41823641` retained within-node pairing and recorded its
  CPU model, but stopped at rank 3 because OpenMPI saw only one Slurm slot.
  Rank 1 had already completed every direct state comparison and science gate.
  Retry `41824313` passed the complete eligible matrix before a forced
  Taylor--Hood conversion exposed an unsuitable existing mesh. Generated
  affine TET10 smoke `41824872` passed; dependency-released fresh-root retry
  `41824941` uses the proven fixture. A test-first independent reconciler is
  green. A
  refreshed cumulative public-branch comparison confirms no overlap in
  `fluid.cpp` or `ris.cpp`. Added scoped follow-on `FLUID-CONST-ORDINARY-001`;
  zero and non-finite parameter semantics must be proven before implementation.
- 2026-09-02 18:14 PDT: Implemented `FLUID-STATE-001` in only `fluid.cpp` at
  final source SHA `1932b094`. Temporary macro-enabled job `41779454` rebuilt
  the solver with the dual path and passed all 21 selected tests at ranks
  1/3/4. The eligible Newtonian, Casson, Carreau--Yasuda, and URIS executions
  compared every cached and legacy `lR`/`lK` Gauss update bitwise; TET10, 2D,
  and FSI fallbacks also passed. Generated flags were restored exactly and the
  temporary checker was removed. Full default/PETSc qualification job
  `41780261` is eligible; jobs `41780322` and `41780338` are dependency-gated
  for three 15-pair rheology timings, exact nine-case rank science, and 96
  pinned matched profiles.
- 2026-09-02 17:47 PDT: Accepted and pushed `CEP2D-WS-001` as
  `d3cc8105226768345cb683b573bd3283dd12ad41`. AP and BO each produced 15/15
  wins and median wall reductions of 27.972561% and 18.920033%, with both
  conservative lower bounds above 18.77%. All native, multi-rank, profile,
  public-symbol, and zero-error Memcheck checks passed. Whole-step
  instructions fell 37.946872%/34.383959%, focused-kernel instructions fell
  about 65.98%, and allocation blocks fell 64.929748%/59.311713% for AP/BO.
  The three external graph pairs passed visual inspection. The next
  `FLUID-STATE-001` structural gate is recorded red on the unchanged fluid
  source before implementation.
- 2026-09-02 17:03 PDT: Independent C++ review narrowed
  `FLUID-STATE-001` to a fixed TET4-only cache. Broad equal-order reuse is not
  behavior-preserving: for `lShpF=false`, current continuity inherits the last
  momentum-point `Nwxx`, whereas a broad cache would use each point's momentum
  Hessian and change TET10/HEX results. The defensible one-file design stores
  four 41-double TET4 point states per element and reuses only the literal
  continuity tail. All other elements, Taylor--Hood, 2D, FSI/public wrappers,
  and gradient calls stay on the legacy path. The post-fluid-FS Casson ideal
  ceiling remains near 10%, so the narrow candidate retains greater-than-5%
  merit without entering externally owned `lShpF` work.
- 2026-09-02 16:54 PDT: CEP qualification job `41770116` completed in
  22m39s. Unit 1/1 and all 224 runnable integration tests passed, with 24
  expected backend skips; archived default/PETSc binary and source checksums
  independently pass. The public header and exported `cep_2d` symbol remain
  exact. Native/rank job `41770653` and 64-profile job `41770654` are now
  eligible on separate 32-core nodes.
- 2026-09-02 16:37 PDT: Implemented `CEP2D-WS-001` in only `cep.cpp`.
  Its source gate is green at candidate SHA `c86dbd6c`, with the public header
  hash unchanged. Default and PETSc builds and unit 1/1 have passed; full-suite
  job `41770116` is running. Dependency-gated jobs `41770653` and `41770654`
  are ledgered to use two 32-core nodes after qualification for 15-pair AP/BO
  timing, exact AP/BO/TTP/mixed-dimensional science at ranks 1/3/4, and 64
  matched whole-step, focused-kernel, allocation, and peak-memory profiles.
- 2026-09-02 16:26 PDT: Rejected `RCS-SCALAR-001` after native evidence
  definitively missed the greater-than-5% gate. Heat-CG's 15/15 wins yielded a
  1.798561% median with 1.444043--2.158273% bounds. CEP's first eight wins
  yielded a 4.003242% median and 4.869281% maximum, mathematically capping any
  15-pair median below 5%. All 46 science comparisons were exact. Job
  `41767242` was stopped before decision-irrelevant profiles, and `precond.cpp`
  was restored byte-for-byte to parent SHA `9d79f8d`; no commit was made.
  `CEP2D-WS-001` is now active, with its source gate observed red on the parent.
- 2026-09-02 16:13 PDT: RCS source-locked qualification job `41765986`
  completed successfully in 20m10s. Its unit target passed 1/1; the full suite
  passed all 224 runnable tests with 24 expected Trilinos-only skips; both
  default and PETSc binaries independently passed their archived checksums.
  Dependent 32-core performance job `41767242` began on `sh03-08n19` and is
  measuring the greater-than-5% acceptance gate.
- 2026-09-02 16:10 PDT: Independent C++ review completed the bounded design
  for `CEP2D-WS-001`. Preserve the public symbol as a fresh-workspace wrapper;
  use a private implementation and one lazy invocation-local workspace only in
  `construct_cep`; restore all nine members to constructor-zero state before
  each call, including every entry of `F`. This removes 11.52--72 million
  measured allocation blocks without static lifetime or arithmetic-order
  changes. Spiral BO attributes 52.12--52.23% of the step and 24.67--24.75%
  of the full process to `cep_2d`. The candidate remains next behind the
  source-dependent RCS validation.
- 2026-09-02 15:56 PDT: Implemented the eight-line scalar-only RCS scan after
  observing its source gate red on `a3b3f1dd`; the gate is green at candidate
  source SHA `a903b9d7`. Source-locked job `41765986` is rebuilding both
  configurations on 32 CPUs and running the complete available suite. A
  syntax-checked performance job `41767242` is ledgered behind the successful
  qualification dependency to measure scaled heat-CG and CEP native pairs,
  rank-1/3/4 science, and 64 matched profiler runs.
- 2026-09-02 15:45 PDT: Accepted and pushed `FLUID-FS-001` as
  `a3b3f1dd`. Full default/PETSc qualification passed all available tests.
  Fifteen of fifteen alternating native pairs favored the candidate with a
  9.289617% median reduction and 8.791209% lower confidence bounds. Twelve
  case/rank comparisons and all 32 profiler comparisons preserved exact
  scientific state and normalized histories. Whole-step instructions fell
  14.061012% median; targeted calls and allocation traffic fell 99.937656%.
  Six visually checked PNG/SVG graphs and an independent reconciliation are
  archived under the designated scratch root. Scalar-RCS preflight job
  `41764538` also passed all 7,123,840 bitwise comparisons on 32 workers.
- 2026-09-02 15:31 PDT: A dedicated scalar-RCS review confirmed that the
  smallest safe change replaces only the `dof == 1` row temporary in
  `precond.cpp`, initialized from the first stored value and using the current
  strict absolute-value comparator direction. It specified exhaustive
  bit-pattern differential checks for ties, signed zeros, infinities, and
  NaNs. Allocation evidence is strong, but short-case whole-process
  instruction ceilings are 2.32--3.23%; advance to repeated-step native
  measurement and retain only if the lower confidence bound exceeds 5%.
- 2026-09-02 15:16 PDT: Reconciled ZeroD breadth job `41757895`: all 32
  profiles exited zero, 64 VTK/series/history comparisons are exact, Memcheck
  contexts replicate exactly, Callgrind spans are below 0.357%, and Massif
  peak spans are below 0.234%. Coupled-boundary work tops out at 0.707%, so no
  ZeroD-specific candidate advances. Its fluid case independently strengthens
  `FLUID-FS-001`, with `get_thood_fs` at 6.766% of the collected step and
  5.510% of the whole process. Implemented the one-file candidate after a
  red/green structure gate; full-suite job `41761822` is running and corrected
  native/rank/profile job `41762902` is queued. Pending predecessor `41762804`
  was cancelled before allocation to add the explicit local MPI slot override.
- 2026-09-02 14:57 PDT: Accepted and pushed `FLUID-WEAK-NNX-001` as
  `958bd1e7`. Its full-suite, 15-pair native, rank-1/3/4, and 32-profile gates
  preserve exact scientific state and normalized histories. The native median
  reduction is 9.259% with an 8.589% lower uncertainty bound; matched profiles
  show a 13.051% median whole-step instruction reduction and exact halving of
  the intended inverse-map call and allocation subtree. The profile job's
  terminal exit reflects only an overstrict Callgrind replicate-equality
  assertion and is independently reconciled in `performance-r4`.
- 2026-09-02 14:41 PDT: Job `41756734` completed all 32 replicated heat-CG
  and spiral-BO profiles. Independent reconciliation retained exact VTK state,
  identical normalized Memcheck contexts, Callgrind spans below 0.279%, and
  Massif peak spans below 0.062%. The replicated profiles strengthen both
  `RCS-SCALAR-001` and `CEP2D-WS-001` without revealing a higher-value new
  area.
- 2026-09-02 14:35 PDT: Native validation in job `41757424` measured a
  9.259% median wall-time reduction over 15 alternating pairs, with every pair
  a win and both 95% lower bounds at 8.589%; VTK states and normalized
  histories were exact. Its rank-3 check then exposed a launcher-only slot
  declaration issue. Resume `41758628` preserves the native evidence and runs
  only corrected scaling and profiles. PETSc job `41755468` produced all 32
  profiles before its original full-directory manifest rejected expected
  timing/restart variation; independent checks retained exact VTK state,
  identical normalized Memcheck contexts, instruction spans below 0.189%, and
  Massif peak spans below 0.053%.
- 2026-09-02 14:12 PDT: Job `41753579` passed the unit target and complete
  available suite: 224 passed and 24 expected Trilinos-only skips. The first
  performance job `41754540` caught a path-representation defect in its input
  manifest before launching a simulation; retry `41756560` uses path-stable
  hashing and a clean artifact root. Submitted breadth jobs `41756567` for
  four ZeroD-coupled cases and `41756734` for replicated heat-CG/spiral-BO
  baselines, filling the four-node/80 GB campaign ceiling with `41755468`.
- 2026-09-02 14:00 PDT: Added dependent performance job `41754540` for the
  weak-boundary candidate and breadth job `41755468` for 32 pinned PETSc
  profiles spanning fluid RCR and coupled FSI. Both remain source-hash locked,
  and the PETSc XML inputs were checked not to activate weak boundaries.
- 2026-09-02 13:44 PDT: Job `41741024` ended after completing 36 of 40
  requested profiles. All nine completed cases are exact across all four
  tools. The corrected P2P1 input exposed its actual Trilinos requirement,
  which the archived exact-tip executable does not provide; those four runs
  are unavailable for this build rather than valid profiles. Began
  `FLUID-WEAK-NNX-001` validation with a red/green scratch source gate and
  submitted full-suite job `41753579`. The accepted request was exposed late
  without an identifier; duplicate `41753587` stopped at the artifact-
  ownership preflight without repeating work.
- 2026-09-02 12:06 PDT: Completed the fluid, non-fluid solver, and
  infrastructure discovery lanes and reconciled all 196 solver source files.
  Promoted `FLUID-WEAK-NNX-001`, `RCS-SCALAR-001`, `CEP2D-WS-001`, and
  `FLUID-STATE-001`; recorded the new below-threshold and duplicate rejections.
- 2026-09-02 11:55 PDT: Job `41727752` produced all 32 requested profiles and
  all 24 exact cross-tool result comparisons, then returned exit 1 after its
  final status snapshot and before its root completion marker. A controller
  replay passed every profile, comparison, binary-hash, and solver-source
  postcondition; the audit is retained as `fluid-packed-r2/controller-validation.txt`.
  Corrected retry job `41741024` entered service with 32 pinned workers on one
  20 GB node after a scratch regression caught the prior launcher's missing
  executable mode.
- 2026-09-02 11:44 PDT: Refreshed Discussion #625, all public performance
  branches, and open issues by the same researcher. Added issue #634's
  constant-gradient and wedge territory to the exclusion set; issues #632 and
  #633 confirm the existing tensor/dyadic and viscous-stress exclusions.
- 2026-09-02 11:40 PDT: Reconciled terminal packed-profile jobs
  `41729995_1` and `41730431`. They retained 28 of 48 and 48 of 64
  complete profiles, respectively; the missing profiles all traced to omitted
  sibling input trees. Submitted source-guarded job `41740291` to rerun those
  profiles and the previously invalid P2P1 case with a staged-only,
  provenance-recorded spelling correction.
- 2026-09-02 11:00 PDT: Created from branch history, retained scratch reports,
  overlap audits, profiler artifacts, and the Slurm ledger. Added the two
  actionable candidates and two active discovery scans.

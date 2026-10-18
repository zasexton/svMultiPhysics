# Optimization Idea Queue

Local workflow queue. This file must remain untracked, unstaged, uncommitted,
and unpushed. Its entries reference the permanent audit in
`explored_optimization_areas.md`.

- Last reconciled: 2026-09-04 00:29 PDT
- Current source commit: `8f71dbf79353d8bf8786274ad2ac84c120d751cf`
- Ordering rule: expected production value multiplied by confidence, divided
  by validation cost and correctness risk.

## Active queue

No unvalidated implementation-ready candidate remains. `RIS-LOOKUP-001` was
accepted and pushed as `8f71dbf79353d8bf8786274ad2ac84c120d751cf` after every
correctness, native, profile, memory, reconciliation, review, overlap, graph,
and remote-verification gate passed. The exhaustive 208-file solver audit and
the final delta audit found no other non-overlapping candidate with a
defensible exact path above 5%; the remaining dynamic gaps are retained below.

Downstream precommit/profile orchestration is regression-clean at harness hashes
`d4fb1dedfa8ba3d318dd73633fcb877d5bb9f9840876d28bb917296ebe832c71` and
`a037421fe747fd5aec0f0d66cb87b4c3ab3141ccacca5b2596e93358c79b1ef8`.
The precommit gate completed, and profiling is locked to exact archived binary
hash `ed7ba17ecd1aa0036ce40f4975c3c6a41f610fda4e290fa8d56b5b2e5ad811dd`.

## Validation specifications

### Accepted: `FLUID-WEAK-NNX-001`

- Design: retain the current velocity inverse map. Under the existing
  equal-order `flag`, copy its shape values into `Nq` and skip only the second
  `get_nnx`; under Taylor--Hood, run the current pressure inverse map unchanged.
  Do not alter gradients, `lShpF`, element handling, or assembly arithmetic.
- Pre-implementation test: create a scratch-only dual-compute check that
  records bitwise equality of `Nw`/`Nq` and `Nwxi`/`Nqx` for supported
  equal-order 2D/3D and moving-mesh paths, plus a call-count gate that is red
  until the duplicate call is removed.
- Correctness: default and PETSc builds; full available suite; exact weak-
  Dirichlet pipe outputs, iteration histories, flux, and pressure at 1, 3, and
  4 ranks; explicit Taylor--Hood fallback exercise; sanitizer smoke where
  practical. Baseline memory-tool contexts must match rather than be assumed
  zero.
- Completed correctness gate: job `41753579` rebuilt both configurations,
  passed the unit target and all 224 runnable integration tests, and recorded
  24 expected Trilinos-only skips. Both archived binaries and the source
  snapshot passed independent checksum verification.
- Performance: at least 15 alternating baseline/candidate pairs on the same
  exclusive node, plus matched whole-step Callgrind and allocation profiles.
- Acceptance: native end-to-end median improvement greater than 5%, with the
  confidence interval supporting that threshold; the pressure-map call edge
  must disappear only in equal-order runs, with no scientific-output change.
- Graphs: write outside the repository under
  `/scratch/users/zsexton/svmp-efficiency-review-20260829/graphs/FLUID-WEAK-NNX-001/`.
- Accepted result: commit `958bd1e7` is pushed. Fifteen alternating pairs
  measured a 9.259% median native reduction with an 8.589% lower confidence
  bound. Four matched profiles measured a 13.051% median whole-step
  instruction reduction and halved the targeted inverse-map calls and
  allocation subtree. Scientific state and normalized histories are exact at
  1, 3, and 4 ranks; all graphs and the independent reconciliation are stored
  in the designated scratch directories.

### Accepted: `FLUID-FS-001`

- Design: construct option-1 and option-2 function spaces once before the
  element loop, then use the corresponding object in the existing momentum
  and continuity phases. Do not merge loops or change arithmetic.
- Correctness: default and PETSc builds; full available test suite; exact
  same-rank outputs and iteration histories for QUD4, TET4, TET10, URIS, and a
  true Taylor-Hood case; sanitizer smoke where practical.
- Scaling: representative cases at 1, 3, and 4 MPI ranks.
- Performance: at least 15 alternating baseline/candidate pairs on the same
  exclusive node, plus matched Callgrind and allocation profiles.
- Acceptance: production 3D median improvement greater than 5%, with the
  confidence interval supporting that threshold; otherwise reject and revert.
- Graphs: write outside the repository under
  `/scratch/users/zsexton/svmp-efficiency-review-20260829/graphs/FLUID-FS-001/`.
- Accepted result: commit `a3b3f1dd` is pushed. Job `41761822` passed the
  default/PETSc full-suite qualification. Job `41762902` measured 15/15 native
  wins, a 9.289617% median reduction, and 8.791209% lower confidence bounds;
  all TET4, TET10/Taylor--Hood, QUD4, and URIS comparisons were exact at ranks
  1/3/4. Four matched profile replicates measured a 14.061012% median
  whole-step instruction reduction and a 99.937656% reduction in targeted
  calls and allocation traffic, with unchanged peak live memory. The figures
  and independent reconciliation are archived in the designated scratch tree.

### Rejected: `RCS-SCALAR-001`

- Design: change only the `dof == 1` RCS row maximum. Initialize the winner
  from the nonempty row's first value, scan the remaining contiguous values
  directly, and replace it only when `fabs(best) < fabs(candidate)`. Leave all
  other DOF branches, scaling order, collectives, and container APIs unchanged.
- Edge semantics: differential checks must retain the first value for equal
  magnitudes and signed-zero ties, retain an initial NaN, ignore later NaNs
  after a finite winner, and preserve infinity behavior. Do not initialize from
  zero, reverse comparator operands, use `std::fmax`, or use `>=`.
- Structural preflight: job `41764538` observed the source gate red on the
  parent and passed 7,123,840 bitwise differential comparisons across 32
  pinned workers. Its archived input/output hashes and empty worker error
  streams were independently verified.
- Correctness: differential checks for irregular row lengths, equal-magnitude
  ties, signed zero, infinities, and NaNs; heat-CG and CEP scalar-RCS outputs,
  residual histories, preconditioner iterations, and solver iterations at one
  and multiple ranks; sanitizer smoke.
- Performance: at least 15 alternating pairs on heat-CG and CEP spiral, with
  matched Callgrind and allocation profiles.
- Acceptance: greater than 5% native end-to-end median improvement with
  uncertainty supporting the threshold, exact supported-case results, and no
  scalar-row vector-reallocation descendants.
- Graphs: write outside the repository under
  `/scratch/users/zsexton/svmp-efficiency-review-20260829/graphs/RCS-SCALAR-001/`.
- Rejected result: job `41765986` passed the complete default/PETSc suite, but
  job `41767242` measured only a 1.798561% heat-CG median and a 4.003242% CEP
  median. Eight CEP reductions at or below 4.869281% mathematically prevent a
  completed 15-pair median from clearing 5%. All 46 paired scientific checks
  were exact. The job was stopped before decision-irrelevant profiles, the
  one-file change was reverted byte-for-byte, and no commit or graph was made.

### Accepted: `CEP2D-WS-001`

- Design: add an invocation-local private workspace in `cep.cpp`, move the
  current arithmetic to a private implementation taking that workspace, and
  reuse it across 2D Gauss calls in `construct_cep`. Preserve the public entry
  point and symbol as a fresh-workspace wrapper. Define the workspace in the
  anonymous namespace and create it lazily per `construct_cep` invocation;
  never use static or thread-local storage. Fully reset all four vectors and
  five arrays to their constructor-zero state before each reuse. In particular,
  reset every entry of the 2x2 `F`, because the kernel overwrites only its
  diagonal before accumulating and reading all entries. Preserve expression
  order and leave shape-gradient/WDG handling and coupled matrix expressions
  unchanged.
- Correctness: AP, BO, TTP, Purkinje, 1D/3D, ECG, multi-domain, and
  electromechanics regressions; coupled/uncoupled and zero/one/multiple-fiber
  paths; exact outputs and histories at 1/3/4 ranks; array bounds and sanitizer
  checks where practical.
- Performance: at least 15 alternating pairs for square AP and spiral BO;
  matched profiles must reduce the 11.52/72 million kernel allocation blocks
  to invocation-scale counts.
- Acceptance: greater than 5% native whole-run median improvement with
  uncertainty supporting the threshold and no numerical or memory-tool
  regression.
- Graphs: write outside the repository under
  `/scratch/users/zsexton/svmp-efficiency-review-20260829/graphs/CEP2D-WS-001/`.

### Accepted: `FLUID-STATE-001`

- Design: gate reuse on 3D stabilized TET4 plus defensive exact checks for
  four nodes and four Gauss points in both phases. Store four fixed point
  states containing `divU`, `up[3]`, `tauM`, and `updu[3][3][4]`; momentum
  fills every field and a private cached-continuity tail consumes it. Keep both
  Gauss loops, all geometry calls, `fluid_3d_c_impl`, and exported wrappers
  unchanged. Never gate on `lShpF` or extend reuse to TET10/HEX: current
  continuity inherits the final momentum `Nwxx` for varying-shape elements,
  so broad reuse would be a numerical change rather than this optimization.
- Correctness: temporary bitwise dual-compute checks for all 41 doubles per
  point and final `lR`/`lK`; exact Newtonian, Carreau--Yasuda, and Casson TET4
  results. Negative eligibility tests must retain legacy calls for equal-order
  TET10, staged Taylor--Hood TET10, WDG/HEX, 2D QUD4, and moving-mesh FSI.
  Cover permeability and URIS factors, ranks 1/3/4, Memcheck, bounds checking,
  and sanitizers where practical.
- Performance: at least 15 alternating pairs for all three viscosity models,
  plus matched whole-step/focused Callgrind, allocation, peak-memory, and
  hardware-counter profiles. Casson should reduce `get_viscosity` calls from
  64,160 to 32,080 while leaving allocation blocks unchanged.
- Acceptance: greater than 5% native median improvement in a representative
  production case with uncertainty supporting the threshold; exact supported
  outputs and no stale-state, ownership, or memory regression.
- Graphs: write outside the repository under
  `/scratch/users/zsexton/svmp-efficiency-review-20260829/graphs/FLUID-STATE-001/`.
- Accepted result: commit `750c2ddf` is pushed. All 45 native pairs won, with
  Newtonian/Casson/Carreau--Yasuda medians of 5.293202%/5.439331%/11.895911%;
  only Carreau--Yasuda has a clearly greater-than-5% lower confidence bound,
  and the weaker intervals remain explicit in the graphs. Whole-step
  Callgrind medians fell 7.611573%/10.120692%/13.727131%; hardware instruction
  medians fell 7.793026%/8.843218%/12.910611%, with all 12 instruction/cycle
  pairs winning. Full default/PETSc tests, 1/3/4-rank science, 189,837,872
  direct cached-state bit comparisons, topology/public fallbacks, nine GCC 12
  ASan/UBSan runs, memory profiles, and independent reviews all passed. The
  commit-labeled figures and provenance are archived in the designated graph
  directory.

### Accepted: `RIS-LOOKUP-001`

- Design: store an invalid-by-default dense first-location cache per RIS
  projection, preserve the current logical row-major first-match rule, use
  bounds-checked direct reads in timestep assembly, and rebuild lazily when the
  remesh generation, rank-local node count, projection count, or map shape
  changes.
- Correctness: differential lookup checks for duplicates, absent IDs, `-1`,
  zero padding, node-count and projection-count transitions through zero,
  empty/restored map shapes, multiple projections, closed/reopened valves, and
  same-shape remesh; exact RIS output/restart/history comparisons at 1, 3, and
  4 ranks.
- Performance: at least 11 alternating baseline/candidate pairs with fixed
  affinity and a matched whole-step Callgrind profile.
- Acceptance: greater than 5% native end-to-end median improvement and greater
  than 5% total instruction reduction, with no in-range timestep `find_loc`
  calls; retain the exact legacy fallback for negative and out-of-range IDs.
- Downstream profile draft: eight matched replicates each of whole-step and
  RIS-only Callgrind, Memcheck/XTree, and Massif use all 32 MLN cores. The
  driver is fail-closed until the full-suite job supplies the exact candidate
  binary hash; current driver/worker hashes are `65d498be633abd524e7081c5a510cc4e222b8a18bc4d84347372b73fe63c9e46`
  and `63ebac3aa13bbceef4f8bda91dfd39cb79c438a9e0902b61f9ad70940593cc88`.
- Memory scope: the sole available RIS fixture has one projection. Its restart
  stamps give rank-local node counts of 3,156 at rank 1, at most 1,455 at rank
  3, and at most 1,015 at rank 4, making the dense payload at most 25,248 bytes
  per process in this suite. General payload scales as approximately
  `8 * nbrRIS * tnNo` bytes per rank plus vector metadata. The planned eight
  Massif pairs must pass, while absent production-size multi-projection
  coverage remains an explicit limitation rather than inferred evidence.
- Reconciliation: the scratch raw-artifact gate passes 35 tests under Python
  3.6.8 and 3.12.1. Three independent review rounds closed manifest-identity,
  rank-cardinality, summary-type, TSV, stream, and archive-recomputation gaps;
  the final review found no issue. Its current verifier/test hashes are
  `48d5805be34fba99a5a5fe0119ab2bd63b11a581a455e89cd5f5327aac4ad336`
  and `5e23e3555ec28cf2b94b5ce924794c469e9dcfb45aec5a1c685bb3f6d954be38`.
- Final reconciliation: the cross-stage scratch validator has 47 behavioral
  tests green on both required Python runtimes. It reconciles exact manifests,
  full test evidence, all 64 profiles, raw-derived counters, exact performance
  boundaries, and immutable deterministic output. Validator/test hashes are
  `4d998d3e51969f28031f4798c3939787e63e95a6f7bd246b322327fdc3102a7c`
  and `aaca601d259b64a242c3dc6830d921eab85546e52691d73f7831d7739ac23889`.
  It accepted the retained roots and wrote summary hash
  `745b2cea6620d55430318ceaeda609f34d534c364c9b6b1bbde687cd1eeaca0b`.
  Removed original profile science/restart payloads cannot be re-hashed, so
  this limitation remains explicit in its summary.
- Graphs: write outside the repository under
  `/scratch/users/zsexton/svmp-efficiency-review-20260829/graphs/RIS-LOOKUP-001/`.
- Accepted result: commit `8f71dbf7` is pushed with only `ComMod.h` and
  `ris.cpp`. Canonical native timing produced 15/15 wins, a 6.361474% median,
  and a 5.595886% lower confidence bound. All eight clean whole-step profiles
  exceed 5% with a 7.671195% median; the RIS-region median is 96.513542%, and
  all in-range focused lookup instructions are eliminated. Default/PETSc
  builds, unit and all 248 integration cases, ranks 1/3/4, normal and
  instrumented differential tests, Memcheck signatures, Massif bounds, and
  independent artifact/source reviews passed. Three PNG/SVG graph pairs were
  visually inspected and archived with commit-labeled provenance.

### Rejected before implementation: `FLUID-CONST-ORDINARY-001`

- Scope only: the mathematically ordinary constant-viscosity path still
  evaluates products with signed zero and non-finite velocity/gradient data.
  Removing those terms is not universally IEEE-bitwise equivalent without a
  finite-state contract that this kernel does not have.
- The accepted-source Newtonian profile contains 5,532,226,029 whole-step
  instructions. `get_viscosity` contributes only 7,364,360 exclusive
  instructions, a 0.133% exact dispatch ceiling before retained assignments
  and derivative operations are charged.
- The former 4.608% whole-statement ceiling requires an additional finite/sign
  contract and is therefore not universally removable. The 12.904% mixed-line
  envelope counts retained expressions whose deletion changes signed-zero,
  NaN, infinity, or rounding behavior; its 8.8% native projection is not an
  exact candidate ceiling.
- Decision: do not create a production specialization or spend a Slurm
  allocation on a disposable spike. The only defensible exact ceiling is far
  below 5%. Evidence: `analysis/fluid-const-ordinary-scope-r1.txt`.

### Profiling-only gap: `ONED-SCRATCH-001`

- State: not source-ready. None of the 30 packed whole-step profiles activates
  production OneD coupling, so no numerical whole-step ceiling exists. The
  repository does contain `tests/cases/fluid/Tpipe_svOneD_2faces`, but it is not
  registered by `tests/test_fluid.py`; its XML names a developer-local macOS
  `.dylib`, and no Linux interface library was found in the scoped scratch tree
  or dynamic-linker cache.
- Potential boundary: `Code/Source/solver/svOneD_interface.cpp` and
  `Code/Source/solver/svOneD_interface/OneDSolverInterface.cpp`. Repeated work
  includes allocation/zeroing of coupling values, full solution scratch copies,
  committed-result copies, and a tiny coupling-type buffer.
- Do not implement until a valid Linux OneD library and production-size
  two-face case show that the specifically removable allocation/copy subtree
  exceeds 5% of the whole step; prefer a 7--8% attributable share to leave
  decision margin.
- A red-first scratch checker must preserve `D,D,L` commit semantics, repeated
  `D`, exception/restart/reinitialization behavior, model size changes,
  owner/rank mapping, collective count/order, DIR/NEU modes, ramping,
  relaxation, signed zeros, NaNs, and infinities. The external library contract
  must prove that input pointers are neither retained nor mutated after return.

## Discovery lanes feeding this queue

| Lane | State | Scope | Result |
|---|---|---|---|
| `DISCOVERY-FLUID-001` | COMPLETE | Newly completed fluid/FSI profiles and untouched fluid-family source paths | Promoted two new candidates and strengthened `FLUID-FS-001`; no new FSI candidate cleared 5% |
| `DISCOVERY-SOLVER-001` | COMPLETE | Newly completed non-fluid solver/assembly profiles and untouched source paths | Promoted scalar RCS and CEP workspace candidates; remaining hot areas were duplicate or externally owned |
| `DISCOVERY-INFRA-001` | COMPLETE | Containers, orchestration, mesh, I/O, BC, and backend paths | No new candidate cleared 5%; reconciled all remaining solver source files |
| `DISCOVERY-COVERAGE-002` | COMPLETE | Full 208-file solver-directory reconciliation and remaining dynamic gaps | Corrected the 196-file C++-only inventory, found no new defensible >5% candidate, and retained OneD scratch reuse as profiling-only |

## Queue operating rules

1. Check `explored_optimization_areas.md`, branch history, local audits, and the
   current Discussion #625 overlap inventory before adding an idea.
2. Do not promote profiler percentage alone. Identify the redundant work or
   algorithmic cause and an exact one- or two-file intervention.
3. For merit-backed candidates, proceed directly through source-gated
   implementation and verification; do not alter source while a job whose
   provenance depends on the current source is still running.
4. Use test-driven development for implementation and preserve floating-point
   ordering, numerical behavior, MPI behavior, ownership, and determinism.
5. Every Slurm validation job must appear in the canonical ledger before it is
   counted, use `amarsden`, stay within the four-node/80 GB campaign cap, and
   request BEGIN, END, and FAIL mail to `zsexton@stanford.edu`.
6. Never interfere with jobs absent from the ledger.
7. Accept and commit only after equivalent-condition correctness and
   performance gates pass. Otherwise revert the candidate and record it as
   rejected in the exploration log.
8. Commits may change only the intended existing source files, use Zachary
   Sexton `<zsexton@stanford.edu>` as author and committer, and must not include
   either local workflow Markdown file.

## Update log

- 2026-09-04 00:29 PDT: Promoted `RIS-LOOKUP-001` as two-file commit
  `8f71dbf79353d8bf8786274ad2ac84c120d751cf`, with Zachary Sexton as both
  author and committer, and verified that exact object at the remote branch
  tip. The outgoing range contained one ordinary non-LFS commit and no other
  files. The two-test graph generator produced three PNG/SVG pairs under the
  designated external graph root; all were inspected at original resolution.
  Graph script/test hashes are
  `4e1ffbea7682e036640f02bc399d1954356f1173a459988f895dc122b55f852b`
  and `18bba1faf2cc000bfb6626f669d69e0182bf77a604bb356fea4946d6a383bfb0`;
  the eight-file graph hash manifest digest is
  `338d5d8adc08e15ad0c1b6fe1af5231acb52417ee1d0e020cc87d6d20c13725f`.
  No unvalidated implementation-ready item remains after the exhaustive
  solver audit and final delta scan.
- 2026-09-04 00:15 PDT: Clean replacement job `41961420` completed exit zero
  on the source-locked binary and produced all 64 expected profiles with 32
  unique cgroup-pinned workers. Independent reconciliation accepted the full
  raw tree: all eight whole-step pairs exceed 5% with a 7.671195145% median,
  the RIS-region median is 96.513541799%, all eight candidate focused samples
  contain zero in-range `find_loc` instructions, all Memcheck signatures are
  exact, and every Massif pair is bounded. The 916-file tree hash is
  `686a25e640a24eacd027e80a5b1525ce397bc38560784f96df70e5ae12cbc532`.
  The cross-stage validator required one format correction for legal
  Callgrind summaries with omitted trailing zero counters; a red-to-green
  regression and a surplus-counter control now pass in its 47-test suite on
  Python 3.6.8 and 3.12.1. Its accepted JSON summary hash is
  `745b2cea6620d55430318ceaeda609f34d534c364c9b6b1bbde687cd1eeaca0b`.
  A 00:18 PDT public refresh then reconciled all 19 current contributor
  performance heads and the current Discussion 625; neither source contains
  RIS overlap. Report hash:
  `baa58f1a533ec6c109352ffeb34763bfb2296952e3fc83ad893268d5c8545ac5`.
- 2026-09-03 22:31 PDT: A scheduler-wait optimizer delta scan returned no new
  queue item. It independently rediscovered `CEP2D-WS-001`, already retained
  as `d3cc8105`, and `RCS-SCALAR-001`, already rejected after exact native
  medians of 1.798561% and 4.003242% on its two workloads. The only active
  validation item remains source-locked `RIS-LOOKUP-001`; discovery continues
  read-only while job `41961420` runs.
- 2026-09-03 21:46 PDT: Independent read-only recomputation from the preserved
  first-attempt raw profiles recovered all eight whole-step rows. Their median
  instruction reduction is 7.674712236%; the RIS-region median is
  96.513541799%, and every focused candidate profile has zero in-range
  `find_loc` instructions for 100% reduction. All eight Memcheck pairs retain
  the same four contexts and signatures, and all Massif pairs satisfy the
  declared bound. These values are corroborating evidence pending the clean
  replacement gate rather than a substitute for its completion marker.
- 2026-09-03 21:39 PDT: Held-verified diagnostic `41958780` completed all
  eight runs with exit zero in 18m10s. Parent and candidate science, history,
  and normalized restart state are exact within every mode. Native and both
  Callgrind modes produce 606,016-byte restarts; Memcheck produces 530,273
  bytes. Their common prefixes differ only in the normalized CPU-time field,
  every `iNorm` is exactly 129.60437208088388, and the longer suffix is all
  zero padding. Diagnostic tree hash:
  `54ef1f499d0274de7fccb942de151603b303da3e64c5b7612768ec6086a1ca81`.
  Independent diagnostic report hash:
  `41bc9d7cfb85c2a2abb54907ac0b9f1cd939d23d3d3739bd7e19420086e4cb93`.
  Test-first repair now retains global science/history equality and requires
  exact normalized restarts across parent, candidate, and all replicates
  within each tool. All 17 downstream tests and all 46 final-validator tests
  pass on Python 3.6.8 and 3.12.1. Profile/downstream/validator/test hashes are
  `a037421fe747fd5aec0f0d66cb87b4c3ab3141ccacca5b2596e93358c79b1ef8`,
  `a5a124cc46dbc5e9d0d08ee52177add4d717ce70ed7d533a6771098db96dbf9c`,
  `276b479dab87c44f9acfb4f8b851e8d2392a1a5ea84ae0fe243b43358e29fdb1`,
  and `227a994e8b76f35a6b188ef52858e7604494fec8164ae4fd85cfe0a6d4383971`.
  The complete failed postvalidation tree is recoverably archived as
  `profiles-postvalidation-failed-41942474` with 907 files and relative tree
  hash `6da0149cc3b514ac996ad5d0b59b162b543d5d9160d12d4a27fe943b2b634e56`.
  Clean replacement profile job `41961420` was submitted held, immediately
  ledgered and metadata-verified, then released with the same one-node,
  eight-CPU, 20 GB SKX and mail contract.
- 2026-09-03 21:18 PDT: Job `41942474` completed all 64 profile runs with
  32 unique cgroup-pinned workers and empty worker errors, then exited one in
  postvalidation. All 64 scientific manifests and histories are globally
  exact; normalized restarts form two deterministic 32-run groups: both
  Callgrind modes share one digest, while Memcheck, Massif, and native rank-one
  evidence share the other. Parent and candidate are exact within every tool.
  Red tests now expose the over-strict global-restart contract. Ledgered,
  held-verified raw-restart diagnostic `41958780` is running on eight SKX CPUs
  to identify the differing field before implementation. Its native pair
  differs only in CPU-time bytes and has identical iNorm values.
- 2026-09-03 19:20 PDT: Corrected profile job `41942474` completed its first
  eight-worker wave with all 16 profiles successful and no worker error
  output. Wave two repeats the exact OS-visible cgroup CPU sequence
  0,2,4,6,8,10,12,14 with eight new unique PIDs. A fresh public refresh
  reconciled all 19 current contributor performance heads against their
  branch-specific current-upstream merge bases and found no `ris.cpp` or
  `ComMod.h` changes; report hash
  `5d3f8f6d10cc5043b66eba26d532632dcc05ce3695d9ffbad5d54a4f7ac09c97`.
  Independent final production review found no critical, important, or minor
  issue and returned ready after profile validation; review hash
  `351d7d963b9e0acca51f129e470034df0cf95d73f970b29b3c6d673f7cd6db00`.
- 2026-09-03 18:40 PDT: Profile job `41940570` was stopped after 3m20s
  with zero completed profiles when four first-wave workers proved that
  detailed Slurm `CPU_IDs` are abstract core indices rather than OS-visible
  affinity IDs on the dual-socket SKX topology. Odd `taskset` IDs were outside
  the cgroup; its partial tree is preserved as `profiles-failed-41940570` with
  tree hash
  `89375ef60f6b0d874be40e83252b31fe42e92a45af6fb1ce1d6b2aaf25eb7bea`.
  A new behavioral regression failed on the abstract sequence and now proves
  workers use the batch process's `Cpus_allowed_list`; scheduler IDs remain
  separately retained as allocation evidence. All 16 downstream and 45 final-
  validator tests pass on both required runtimes. Current profile/downstream/
  validator/test hashes are
  `4f8c9572bcd96335ab2153c75d2ee63806c6c504560009897827fc07039cf03c`,
  `87c290c56961aa906f9c3c6d026b5e103dea3c6dc19b5a59d2cbff1876c20846`,
  `cce9015ae5688a9adeefb9b93d43180df9b2da0ab00837a0ea49ebb4e85d62b4`,
  and `7a9916b50a168db623ea013aa1d3629af8fd70764309db4ccf665a153c24682d`.
  Retry `41942474` was submitted held, immediately ledgered, fully verified,
  and released at normal priority with the same one-node/eight-CPU/20 GB
  contract and required mail.
- 2026-09-03 18:25 PDT: Held-verified job `41936240` completed on
  `sh02-10n28` with exit zero. Default and PETSc rebuilds passed, unit 1/1
  passed, and the normal and instrumented compiled differential tests produced
  exact expected output with empty error streams. The full integration result
  is 224 passed and 24 expected backend skips out of 248; binary manifests,
  dependency checks, and all 11 public RIS symbols reconcile exactly. The
  archived default candidate hash is
  `ed7ba17ecd1aa0036ce40f4975c3c6a41f610fda4e290fa8d56b5b2e5ad811dd`.
  A red/green scheduler-artifact regression exposed that detailed job output is
  required to obtain assigned CPU IDs before pinning. The shared-SKX profile
  harness now runs 32 workers as four eight-worker waves pinned to the eight
  assigned CPUs. All 15 downstream tests and all 45 final-validator tests pass
  on Python 3.6.8 and 3.12.1. Current profile/downstream/final-validator/test
  hashes are
  `0810e7243756ca7e0a39c5c4f26b7d01a00a9a0a0c8fd1332245fd0044ec9643`,
  `3fae4b868895ce9927f77df6fae1e1fb629094121b240760b330400581aec178`,
  `ceb901fbf48c9abddce9aabdd0f1eb1b372598ea2c667732da6ce32919b74489`,
  and `7089a29552c9bf896c5b2c06accd131ef2e4102ce7d7ab448a6c20fb2c7c93f6`.
  Profile job `41940570` was submitted held, immediately ledgered, fully
  verified for one shared eight-CPU SKX allocation, 20 GB, four hours, and all
  required mail, then released at normal priority.
- 2026-09-03 17:50 PDT: Job `41934718` passed the default build, unit 1/1,
  source/semantic gates, and native-test compilation, then failed before native
  execution because the driver still contained five bare text-matcher calls
  unavailable in the clean job path. The partial root is preserved as
  `precommit-failed-41934718`. An expanded red/green portability regression
  covers the driver, whose corrected hash is
  `0c64553b4321237e258d56d85fa4fad63feac710fe0a556164e4e31062afbcb5`.
  All 13 downstream tests pass on both required runtimes. Held-verified retry
  `41936240` is running on `sh02-10n28`; default build, unit 1/1, and normal
  plus sanitized native differential runs have passed with empty error
  streams. Current precommit/profile/downstream-test hashes are
  `d4fb1dedfa8ba3d318dd73633fcb877d5bb9f9840876d28bb917296ebe832c71`,
  `508058690953a1e2b1b18982203053e28ef6f29a6eca09e602de185e17349974`,
  and `d66fb3109c2dcc2155ceb87a2d643580408c771949b35aa0d35d3b6718a75046`.
- 2026-09-03 17:35 PDT: The exclusive MLN precommit request had no viable
  near-term capacity, and its first shared-MLN replacement exposed a full
  scheduler-memory reservation on every node. A test-first SKX correction now
  requests eight shared cores and 20 GB for this correctness-only gate. All 13
  downstream tests pass on Python 3.6.8 and 3.12.1. Ledgered job `41934718`
  was submitted held, fully metadata-verified, and made eligible. Predecessors
  `41927570` and `41933057` were canceled only after verified handoffs; both
  had zero elapsed time, zero allocated nodes/CPUs, and no assigned node.
  Current precommit/profile/downstream-test hashes are
  `eba47a9e22da152c8264279584f768ede9a7f8f54021f1c1a5b9269d6cb53f3d`,
  `1dec064a088853b261ed547f9fd262cf7b958c684e09d4e548349d01783cedbf`,
  and `d1d6882aed622f9d514a2b0539adc561686733537bc7f7301fb9b52fda66e9db`.
- 2026-09-03 16:39 PDT: Canonical job `41920684` completed in 36m09s after
  its exclusive allocation recorded quiet loads of 0.07, 0.06, and 0.05.
  All 15 wall pairs favored the candidate, 13 exceeded 5%, median reduction
  was 6.361474%, and both conservative lower bounds were 5.595886%. VTK,
  integral, history, and normalized restart evidence is exact at ranks 1, 3,
  and 4; all solver error streams are empty. The independent raw-artifact
  reconciler returned its exact pass marker. Analysis JSON hash:
  `77d0f25d69e3956b60c809c88ecd9f600efd92161f78f60f442e50c6d9c9c347`.
  Full precommit job `41927570` was then submitted held, immediately ledgered,
  fully verified for its 32-CPU MLN/20 GB/exclusive/four-hour contract and
  mail settings, and made eligible at priority 1046214.
- 2026-09-03 15:54 PDT: A release-window race could keep the held native job
  behind higher-priority work despite a briefly idle node. A new red/green
  regression now proves the exclusive job itself waits for three consecutive
  one-minute load samples no greater than 1.0 and rejects sustained high load
  before warmup. This permits normal unbound scheduler eligibility without
  relaxing the clean-node standard. Twelve downstream and seven controller
  tests pass on Python 3.6.8 and 3.12.1. Job `41920684` was ledgered, held,
  fully verified, and made eligible at priority 1046189; only then was bound
  predecessor `41916253` canceled at zero allocation. Current long/precommit/
  profile/downstream-test hashes are
  `f445e4093ce0c7cbb8fbabdcca3a6ceea56f19d02ff02a7a5fdee96d537dc820`,
  `168b2ca6033c5861b8b56b71ec28850ceb5e033b4ea38db057e7524995e0ccac`,
  `4b74cfc277167ac915a2e7a72886d902c0a06f205f213b9ac2f0be1301ebcd6b`,
  and `61e6f96fe861bbd61280ba47ef8d971d4603b27d4dd222afa886e1ac1d3d0f33`.
- 2026-09-03 15:29 PDT: Quiet-node repetition `41908735` completed all 15
  alternating wins and exact rank 1/3/4 VTK, integral, history, and normalized
  restart comparisons. Its 5.665025% median was favorable, but 4.861111%
  bootstrap and 4.795370% distribution-free lower bounds remained
  inconclusive. The complete root is preserved as
  `long-native-inconclusive-41908735`. The unfiltered 45-pair pool has 45/45
  wins, a 6.260297% median, and 5.665025%/5.583333% lower bounds; it is retained
  as corroboration because its pairs are nested in three runs on two nodes.
  Report hash:
  `c39e9d2175105aa52356aad554cdd215b99ac1ef97d3f379eaec2be3c2eb6772`.
  A red/green regression now locks a
  60-step timed case with matching step-60 save/restart cadence and a one-hour
  resource limit, reducing fixed launch/setup influence without changing the
  15-pair inference or production source. All 11 downstream and seven
  controller tests pass on both required Python runtimes. Ledgered node-bound
  job `41916253` is held at zero runtime under controller session `13769`
  after its first release window ceased to be fully idle; no canonical root
  exists and production source hashes remain locked.
- 2026-09-03 14:54 PDT: The completed node-health diagnosis produced a
  focused red controller test: an idle node with observed CPU load 13.11 was
  previously selected. The controller now requires numeric load no greater
  than 1.0 at candidate selection and every stability/release checkpoint.
  Ten downstream and seven controller tests pass on Python 3.6.8 and 3.12.1.
  Controller and test hashes are
  `0a0ad8186abc9f0e36dc51166ff0907f4d733428cfc67f28b5b08e2522fa5c40`
  and `f167b6ab5d3babe735b985a19a43f5b87e8cbdec3352946d2cdebd56c1c771e7`.
  Job `41908735` remains held at zero allocation under restarted session
  `63865`; its canonical root is absent.
- 2026-09-03 14:39 PDT: Corrected native job `41906282` completed with exit
  zero and exact rank 1/3/4 VTK, integral, history, and normalized restart
  comparisons. Its 15/15 wall wins and 6.260297% median corroborate the
  earlier accepted result, but its 4.376548% lower bounds are independently
  inconclusive. The node was drained immediately afterward by its health
  monitor for excessive background load. The complete root is preserved as
  `long-native-inconclusive-41906282`. Job `41908735` is ledgered, fully
  verified, and held at zero allocation under controller session `54843`
  until a healthy idle SKX node appears. The canonical root is absent.
- 2026-09-03 14:24 PDT: Controller session `96521` found a stable idle SKX
  node and safely handed zero-runtime held job `41904070` to ledgered,
  node-bound replacement `41906282` on `sh02-07n59`. The replacement's exact
  account, 24 CPUs, 20 GB memory, exclusive node, command, notification
  settings, and requested node were verified before the source job was
  canceled and the replacement released. The canonical retry began at
  14:19 PDT, and the controller exited successfully after confirming
  allocation. Source hashes remain locked.
- 2026-09-03 14:06 PDT: Job `41899495` completed all 15 long native
  pairs with 15/15 wins, a 6.844741% wall median, and 5.583333% bootstrap
  and distribution-free lower bounds, so the performance gate is accepted.
  It then exposed a Bash dynamic-scope bug: `stage_case` reused the caller's
  local rank name and changed a rank-1 request to rank 4. Its accepted timing
  evidence is preserved as `long-native-failed-41899495`. A behavioral red
  test reproduced the exact mutation; a distinct local loop variable fixes it.
  Ten downstream and six controller tests pass on Python 3.6.8 and 3.12.1.
  Current long/precommit/profile/controller/test hashes are
  `8d1f4e839ed5b42ff16e97f39f8651caeb66ff75f890edf021342df2aa6c5f11`,
  `a5be0a551aa5c9752096385d83ba77d3cd012c6928351cb0ef6b2e9c97cb6f8a`,
  `65d498be633abd524e7081c5a510cc4e222b8a18bc4d84347372b73fe63c9e46`,
  `c493e92f3f1ba5edca19a193ed1b124529756c2190fd8db5a034ae36f552cd5f`,
  and `83067be131c8bfe71c23779d541a126a5e2a8eff814262b0259ef2c87082da22`.
  Clean retry `41904070` is held unbound at zero runtime under controller
  session `96521`; its canonical artifact root is absent.
- 2026-09-03 13:32 PDT: Native job `41898003` reached a fully idle node but
  stopped before solver execution because the module-purged environment did
  not provide the harness's bare text-search executable. Its exit-127 root is
  preserved as `long-native-failed-41898003`. A focused red/green regression
  replaced that dependency with `/usr/bin/grep` across the long and precommit
  harnesses. Nine downstream and six controller tests pass on Python 3.6.8
  and 3.12.1. The repaired long/precommit/profile/controller hashes are
  `0150b2b92806a7f4b30afc08e4d946ce8c1fdebef1c5fad0b22b6e6363f8ef17`,
  `bc9cf9443250dc578977dc5fb4cdff170b36ba4189ce272e0b1a53989578ac31`,
  `ce2bcb9ac05cd2b05f5410d760f9feaba1dd2ec8151a87b79b51079941690fc6`,
  and `bb1975b3eaa07735530c6ddfb49a953519bd79d61dda4d24b2ec860d1af2ea3f`.
  Current job `41899495` is re-held at zero runtime after unrelated work took
  its first release window; session `55072` continues monitoring and the
  canonical artifact root remains absent.
- 2026-09-03 11:43 PDT: Final artifact validation preflight is review-clean.
  Forty-four behavioral tests pass Python 3.6.8 and 3.12.1; specification
  round 5 and quality round 3 returned READY. Two verified ledger-first idle
  handoffs were lost to unrelated allocations before this campaign received
  resources. Current native retry `41879818` is held at zero elapsed with no
  artifact root under controller session `95200`.
- 2026-09-03 06:39 PDT: Submission-bound job `41850414` reached its intended
  node but stopped before simulation because a pristine submission environment
  lacked the site module function. Its exit-127 artifact is preserved. The
  corrected harness now bootstraps `/etc/profile`; an isolated clean-shell
  check loaded every required module and exact tool version. Current job
  `41851116` uses corrected harness
  `b3dae4a525faec502c55b9af8e9bf5245990b2f9489d89fd90e832435e05c03c`
  and remains held with zero allocation after another user's work entered the
  release window. Relocation controller
  `4548fb8a71dd8733b0d6fe100af72e2250f5649dde31e7ca7594cd313b0d1ffd`
  passes six tests on Python 3.6.8 and 3.12.1 and is active as session `95200`.
- 2026-09-03 06:24 PDT: The first truly idle-node observation proved that
  pending-job requested-node updates are silently discarded on this site.
  Job `41831216` remained held and never allocated. Held probe `41848470`
  verified submission-time node binding and was canceled at zero allocation
  after ledgering. Controller
  `d3c4521e4095ccd9d1971a2e99eee572bc28d4c9d5319afbfaedc8806e7689d4`
  passes five scheduler-boundary regressions and is monitoring as session
  `49000`; it can release only a newly submitted, ledgered, metadata-verified,
  node-bound replacement after two stable all-partition idle samples.
- 2026-09-03 05:50 PDT: The designated C++ optimizer completed a delta-only
  challenge of the no-second-candidate conclusion and returned
  `COMPLETE-NONE`. No new non-RIS profile evidence exists. OneD scratch reuse
  still lacks an activated Linux fixture and a whole-step ceiling; remeshing
  lacks a bounded attributable hotspot; backend reuse remains covered by
  public work. No new source lane or compute allocation is justified. Evidence:
  `analysis/solver-residual-candidate-audit-r1.txt`.
- 2026-09-03 05:39 PDT: Accepted the independently reviewed RIS raw-artifact
  reconciler as the eventual evidence gate. The final 35-test tree passes both
  required Python runtimes, and review round 3 found no issue after two prior
  rounds drove test-first fixes for every demonstrated false acceptance. The
  long native job remains held at zero allocation until a full target node is
  idle; no artifact root exists yet.
- 2026-09-03 04:55 PDT: The designated C++ optimizer completed a focused
  read-only re-scope of `FLUID-CONST-ORDINARY-001`. The current `fluid.cpp`
  hash and archived Newtonian profile were independently verified. Exact
  constant-model dispatch is only 7,364,360 of 5,532,226,029 whole-step
  instructions, or 0.133%; every larger proposed saving depends on arithmetic
  transformations that are not observably exact for the current input domain.
  Reclassified the idea as rejected before implementation and removed it from
  the active queue without changing source or consuming Slurm resources.
- 2026-09-03 04:18 PDT: Independent C++ review found no production
  correctness blocker and kept native performance, complete Slurm, MPI, and
  memory evidence mandatory. It identified missing independent validity-key
  transitions in the future compiled verifier. The scratch-only verifier now
  covers node counts `8 -> 9 -> 0 -> 8`, projection counts
  `3 -> 2 -> 0 -> 3`, map shapes `2x6 -> 2x0 -> 2x6`, and the explicit
  immutable-map-generation boundary before same-shape remesh. Focused re-review
  found no compile, bounds, tautology, or assembly-count defect. Current
  verifier, driver, precommit, and profile-driver hashes are respectively
  `68b8133e78eda1c152d433e5f628446edbe2a6f045230e37de44f82a427760f4`,
  `df90e37b425d7e79131313fcab62b7f0241108222f1fa09994ff3b2be2287b3a`,
  `7a8117154c9e10f79a2fe8ea373aaa21d04f477efdc86b32dd6e86e616814d3a`,
  and `715d249685704a195c4132a472c4e4901fdb0bad783a5938647d90f08a712da8`.
  Compilation and normal/sanitized execution remain Slurm-only downstream
  gates; the pending native-timing harness and production hashes are unchanged.
- 2026-09-03 03:54 PDT: Direct comparison corrected the coverage audit's
  preliminary classification of `Parameters.cpp.orig`: it is a distinct,
  unreferenced 385-line legacy copy, not a duplicate of the active parser.
  Focused re-review confirmed the other 11 support-file dispositions and the
  exact 208-file count. Active extras are build metadata, cold partitioning,
  exceptional-path support, or excluded runtime configuration; the conclusion
  that no new candidate has a defensible greater-than-5% path is unchanged.
- 2026-09-03 03:59 PDT: Closed the OneD case-discovery ambiguity. A repository
  two-face case exists, but pytest does not register it, its XML names an
  unavailable developer-local macOS library, no retained integration log calls
  it, and no Linux interface library was found in the scoped scratch tree or
  dynamic-linker cache. `ONED-SCRATCH-001` therefore remains profiling-only.
- 2026-09-03 03:50 PDT: The designated C++ optimizer reconciled all 208 files
  in `Code/Source/solver`: 196 `.cpp`/`.h` files plus 12 build, documentation,
  configuration, C, and inline support files omitted from the former count.
  Fresh reads of the remaining dynamic gaps found no source-ready in-scope
  candidate with a defensible path above 5%. `ONED-SCRATCH-001` is recorded as
  profiling-only because retained cases never call the OneD implementation;
  activated remeshing lacks a whole-run ceiling, startup partition work lacks
  leverage, and representative Trilinos reuse remains externally owned.
- 2026-09-03 03:45 PDT: Audited graph coverage across the retained branch.
  Every optimization has an external package; `CEP2D-WS-001` had valid image
  hashes and a candidate-source hash matching `d3cc8105`, but its pre-commit
  provenance lacked the final commit label. Added explicit parent and retained
  full IDs and reverified all six PNG/SVG hashes. Refreshed provenance hash:
  `4695588c013cea3d1f1f4e976a7c7c63964d992ef6073931587361e1e727ab32`.
  All 42 branch commits also have the required author/committer identity;
  every non-merge commit is within the two-file limit, and messages plus added
  lines pass the restricted-term gate.
- 2026-09-03 03:15 PDT: The first idle-node release attempt safely re-held
  `41831216` without allocation after its 15-second observation window. Live
  Slurm configuration showed a 30-second deferred-backfill interval and a
  five-minute priority calculation period; the old window could not reach
  either scheduling boundary. Corrected controller
  `2a45588b632a224a5c38662bef8401486e8aa57586d63f38d5f9867218cb3cc1`
  now observes an idle pinned node for up to six minutes, re-holding and
  clearing the pin within one second if another allocation appears. Artifact
  audit also found that the compiled shadow-assembly and same-shape remesh
  lifecycle gates remained outstanding. The then-current scratch-only verifier
  `d34fe66e50f161ffb2be6c1031ef170c28d0f84cc2a41d35abebddb7b5e49a07`
  and driver
  `86a7a2b9036aeffaea15696b3962e94f7f601e3090d63eb6543090a30a6c383f`
  were hash-locked into the then-current precommit harness
  `711f60769e40cbcb1089e7949f83752eb8dfce57fdd36f9b741077b2b38cb56f`.
  Shell grammar and exact link-command transformation pass; compilation and
  normal/sanitized execution remain Slurm-only downstream gates.
- 2026-09-03 02:58 PDT: While the source-locked native retry remained held,
  prepared and syntax-checked a non-runnable downstream profile draft. It
  requires native acceptance, full-suite completion, all source/input hashes,
  and the future candidate binary hash before creating artifacts. Its 32
  pinned workers cover eight balanced parent/candidate replicates across
  whole-step instructions, RIS-only instructions, memory diagnostics, and
  heap peaks, with exact VTK/integral/history/normalized-restart comparisons.
- 2026-09-03 02:37 PDT: Read-only optimized-machine-code inspection confirmed
  that every cache helper is inlined, both in-range lookups compile to direct
  dense-vector loads, fallback scans remain only for negative/out-of-range
  values, and allocation is confined to invalid-generation rebuilds. The
  screen's peak RSS is flat within measurement granularity. Its independent
  internal simulation clock gives a 6.727883% paired median and 5.348715%
  lower bounds, corroborating but not replacing the external timing gate.
  Retry `41831216` is ledgered and held because no full node is idle; partition
  configuration confirms that releasing it early could requeue unrelated
  owner work. A fresh whole-solver review found no new production-ready fluid
  candidate plausibly above 5%; `FLUID-CONST-ORDINARY-001` remains a
  disposable-spike-only experiment with a guaranteed ceiling below 5%.
- 2026-09-03 02:22 PDT: Job `41829576` built the two-file candidate, passed
  unit 1/1 and the 890,064-comparison semantic model, and completed all 15
  alternating native pairs. Every pair favored the candidate; the median was
  5.681818%, but both lower bounds were 4.610951%, so the short-run evidence
  is promising but not acceptable yet. All 16 VTK/integral/normalized-history
  pairs were exact. The harness then stopped because raw restart files include
  the deliberately runtime-dependent `cpu_time` header field; the restart
  normalizer now has a synthetic self-test and correctly identifies every
  reference rank record. A 20-step timing retry with normalized restart and
  rank-1/3/4 comparisons is ready and will be submitted only on an idle node
  so no shared-owner workload is preempted. Independent implementation review
  found no defect and returned READY to continue qualification.
- 2026-09-03 01:57 PDT: The detailed RIS optimizer review returned READY for
  a bounded two-file experiment. It fixed the observable contract as logical
  row-major first occurrence, legacy fallback for every negative (especially
  `-1`) and out-of-range ID, independent rank-local projection tables, and
  preservation of the existing stale-`rowNadj` behavior. Cache validity is
  keyed to `rmsh.cntr`, `tnNo`, projection count, and all map shapes. Archived
  evidence sets the exact instruction-removal ceiling at 7.356609%; native
  gain above 5% remains unproved. Wrote the scratch implementation plan and
  observed the source-contract gate fail on exact commit `750c2ddf` before
  production editing, as required.
- 2026-09-03 01:47 PDT: Final independent source review reported no Critical,
  Important, or Minor findings and returned `READY TO COMMIT`. Retained the
  source-only change as commit `750c2ddf`, authored and committed by Zachary
  Sexton, and verified the same full object ID on the remote performance
  branch. The first push was stopped locally because its Git LFS hook could
  not find the client; the outgoing range was proved to contain only a normal
  non-LFS `fluid.cpp` blob before the hook was bypassed. Regenerated and
  visually inspected four PNG/SVG graph pairs; the history plot carries the
  commit label, all hashes and evidence roots are in graph provenance, and
  the weaker Newtonian/Casson uncertainty remains visible. Promoted
  `RIS-LOOKUP-001` to active rank one with optimizer scoping in progress.
- 2026-09-03 01:40 PDT: Supplemental job `41824941` completed exit zero in
  12:07. Its locked 31-row matrix passed 15 eligible rank/model/permeability/
  URIS cases, 7 explicit TET10/Taylor--Hood/WDG/HEX8/2D/public-wrapper
  fallbacks, and 9 GCC 12 ASan/UBSan cases. The poisoned direct checker
  compared 189,837,872 named state values bit-for-bit; fallback counters
  proved zero cache entry; all results and normalized histories were exact.
  Thirty pinned hardware runs produced all 180 requested counters. Every one
  of 12 measured instruction and cycle pairs favored the candidate;
  Newtonian/Casson/Carreau--Yasuda median instruction reductions were
  7.793026%/8.843218%/12.910611%, while noisy branch-miss/cache changes are
  retained in `supplement-r3/controller-validation.txt`. The unit-tested
  independent reconciler accepted the complete raw artifact tree. A fresh
  fetch still found 19 public performance branches and no `fluid.cpp` or
  `ris.cpp` overlap; two branches share `ComMod.h` only for an unrelated
  output flag, which is recorded for the subsequent RIS review. Final source
  re-review is active; workflow Markdown remains untracked and unstaged.
- 2026-09-03 01:11 PDT: A deeper optimizer scope marked
  `FLUID-CONST-ORDINARY-001` not ready for production. Existing signed-zero and
  non-finite products make algebraic zero elimination observably different,
  even under finite positive material parameters; `-0.0` permeability and all
  public-wrapper calls must remain legacy. Only 4.608% of step instructions
  are wholly removable, while the 12.904% mixed-line envelope corresponds to
  an optimistic 8.8% native ceiling. Demoted it behind RIS and limited it to a
  disposable bitwise spike after state reuse is finalized. Supplemental job
  `41823641` meanwhile passed build/link, counter capability, and a rank-one
  Newtonian run comparing 9,739,960 state values, then stopped when OpenMPI
  exposed only Slurm's one task as a launch slot. Retry `41824313` added the
  launcher declaration and passed all 15 eligible rank/model runs plus
  equal-order TET10, then exposed that the existing large equal-order mesh is
  invalid when forced into Taylor--Hood inverse mapping. Generated affine
  TET10 smoke `41824872` passed topology tests, three solver iterations, exact
  output production, and zero stderr. Dependency-released fresh-root retry
  `41824941` now uses that proven fixture.
- 2026-09-03 00:52 PDT: Completed the primary `FLUID-STATE-001` evidence
  audit. Qualification job `41780261` passed unit 1/1 plus all 224 runnable
  default/PETSc integration tests with 24 expected backend skips. Native job
  `41780322` produced 45/45 wins: medians are 5.293202%, 5.439331%, and
  11.895911% for Newtonian, Casson, and Carreau--Yasuda, with exact rank
  science. Profile job `41780338` completed 96/96 conditions with exact
  science and matched memory-tool contexts; whole-step instructions fell
  7.611573%, 10.120692%, and 13.727131%. A rigorous read-only review found no
  source defect but requested direct comparison of all 41 cached values,
  explicit WDG/HEX8/Taylor--Hood/permeability fallback evidence, sanitizer
  coverage, and hardware counters. Initial job `41822025` was cancelled before
  allocation when static audit caught wrong adjacent-element IDs on the
  generated one-element wedge faces. The corrected generator passed explicit
  WDG/HEX8 topology and adjacency tests. Milan-only job `41822654` was then
  cancelled before allocation because all four matching nodes were held by
  long jobs. Partition-wide exclusive-node job `41823641` preserved paired
  hardware isolation but stopped at rank 3 because OpenMPI saw only Slurm's
  one task as a launch slot; rank 1 had already compared 9,739,960 state values
  exactly. Retry `41824313` fixed the launcher and passed all 15 eligible runs
  plus equal-order TET10 before the existing equal-order mesh failed when
  forced into Taylor--Hood mapping. A generated affine quadratic fixture then
  passed smoke job `41824872`; dependency-released retry `41824941` uses it to
  close the evidence gaps, with an independently unit-tested reconciler.
  The refreshed public performance branches still touch neither `fluid.cpp`
  nor `ris.cpp`. Added the next fluid hypothesis `FLUID-CONST-ORDINARY-001`
  behind the active candidate; its zero/NaN semantics remain an implementation
  blocker until proven.
- 2026-09-02 18:14 PDT: Implemented the exact TET4-only `FLUID-STATE-001`
  candidate in one existing file. Macro-enabled job `41779454` passed all 21
  selected rank-parametrized tests and bitwise-compared cached against legacy
  `lR`/`lK` updates for Newtonian, Casson, Carreau--Yasuda, and URIS. TET10,
  2D, and FSI fallback cases passed, the temporary checker was removed, and
  the final source returned to locked SHA `1932b094`. Full-suite job
  `41780261` is eligible; dependency-gated jobs `41780322` and `41780338`
  cover three 15-pair native distributions, nine exact cases at ranks 1/3/4,
  and 96 matched whole-step, construct-fluid, Memcheck-XTree, and Massif
  profiles.
- 2026-09-02 17:47 PDT: Accepted and pushed `CEP2D-WS-001` as
  `d3cc8105226768345cb683b573bd3283dd12ad41`. AP and BO each produced 15/15
  native wins, with 27.972561% and 18.920033% median wall reductions and
  27.773527% and 18.775848% lower bounds. All 32 native, 12 rank/case, and 32
  profile conditions were exact; 64/64 profiles completed and all 16
  Memcheck runs reported zero errors. Median whole-step instructions fell
  37.946872%/34.383959%, focused-kernel instructions fell about 65.98%, and
  whole-process allocation blocks fell 64.929748%/59.311713% for AP/BO.
  External graphs and provenance are complete. Promoted `FLUID-STATE-001` to
  rank one and observed its narrow TET4-only structural gate failing on the
  unchanged parent source before implementation.
- 2026-09-02 17:03 PDT: Independent C++ review rejected broad fluid-state
  reuse and made the queued design TET4-only. Varying-shape elements currently
  reuse the final momentum `Nwxx` throughout continuity, so per-point caching
  would change TET10/HEX behavior. The bounded design uses four fixed
  41-double states and a literal cached tail only under exact TET4/VMS shape
  guards; every other path remains legacy. This preserves the near-10% Casson
  ceiling without overlapping externally owned gradient work.
- 2026-09-02 16:54 PDT: CEP full-suite job `41770116` completed: unit 1/1,
  integration 224 passed with 24 expected backend skips, checksum-verified
  default/PETSc binaries, and unchanged public header/symbol. The two 32-core
  performance jobs are eligible to start.
- 2026-09-02 16:37 PDT: Implemented the one-file CEP workspace candidate at
  source SHA `c86dbd6c`; its previously red gate is green and the public header
  remains byte-identical. Default/PETSc builds and unit 1/1 passed. Full-suite
  job `41770116` is running, with two ledgered dependent 32-core jobs for
  native/rank validation (`41770653`) and 64 matched profiles (`41770654`).
- 2026-09-02 16:26 PDT: Rejected `RCS-SCALAR-001`. All 15 heat pairs won but
  the median was only 1.798561%; all first eight CEP pairs won but the median
  was 4.003242% and the maximum 4.869281%. Those eight values mathematically
  cap a 15-pair median below 5%. All 46 paired science comparisons were exact.
  The remaining profile work could not change acceptance, so job `41767242`
  was stopped and the candidate was restored byte-for-byte without a commit.
  Promoted `CEP2D-WS-001` to rank one and observed its source gate red on the
  parent source.
- 2026-09-02 16:13 PDT: RCS qualification job `41765986` completed in
  20m10s: unit 1/1, integration 224 passed with 24 expected backend skips, and
  checksum-verified default/PETSc binaries. Dependent 32-core performance job
  `41767242` is running on `sh03-08n19`.
- 2026-09-02 16:10 PDT: Independent C++ review made `CEP2D-WS-001`
  implementation-ready: use an anonymous-namespace workspace, a private
  implementation, the unchanged public entry point as a fresh-workspace
  wrapper, and lazy invocation-local reuse within `construct_cep`. Reset all
  nine members before every Gauss call, including the complete 2x2 `F`.
  Spiral BO attributes 24.67--24.75% of full-process instructions to the
  kernel, strengthening its position as the next candidate after RCS.
- 2026-09-02 15:56 PDT: Began `RCS-SCALAR-001` without an approval pause.
  Its eight-line direct scan preserves the proven first-value comparator
  semantics and passes the green source gate. Source-hash-locked full-suite
  job `41765986` is running on 32 CPUs; dependent job `41767242` is ledgered
  to cover scaled heat-CG and CEP timing, exact rank-1/3/4 science, and 64
  matched profiles.
- 2026-09-02 15:45 PDT: Accepted and pushed `FLUID-FS-001` as
  `a3b3f1dd` after full default/PETSc qualification, 15/15 native wins, exact
  four-case science at 1/3/4 ranks, and 32 exact profiler comparisons. The
  median native reduction is 9.289617% with 8.791209% lower confidence bounds;
  median whole-step instructions fall 14.061012%. Six visually checked
  PNG/SVG figures record the result. Promoted `RCS-SCALAR-001` to rank one
  after job `41764538` passed 7,123,840 bitwise differential comparisons.
- 2026-09-02 15:31 PDT: Independent scalar-RCS optimizer review confirmed the
  one-file direct scan and its exact comparator semantics. XTree attributes
  202,404 allocations/9.91 MB in heat-CG and 10,012,002 allocations/495.97 MB
  in CEP spiral to the temporary-vector edge. Short-case whole-process
  Callgrind ceilings are only 2.32--3.23%, so the candidate remains worth a
  repeated-step native measurement but must be rejected unless its confidence
  bound clears 5%. No Discussion #625 overlap was found in `precond.cpp`.
- 2026-09-02 15:16 PDT: Job `41757895` completed all 32 ZeroD breadth
  profiles. Independent reconciliation retained 64 exact scientific
  comparisons, exact Memcheck contexts, Callgrind replicate spans below
  0.357%, and Massif peak spans below 0.234%. No distinct ZeroD candidate
  clears 5%; its fluid profile independently places `get_thood_fs` at 6.766%
  of the collected step and 5.510% of the whole process. Began
  `FLUID-FS-001`: its scratch gate is red on the parent and green after the
  one-file change. Full-suite job `41761822` is running and source-dependent
  performance job `41762902` is queued. Its first pending submission
  `41762804` was cancelled before allocation to add the required local MPI
  oversubscription declaration.
- 2026-09-02 14:57 PDT: Accepted and pushed `FLUID-WEAK-NNX-001` as
  `958bd1e7`. Job `41758628` completed all 32 profiles and exact rank-1/3/4
  comparisons before its controller rejected normal sub-0.462% Callgrind
  replicate variation. Independent reconciliation found zero worker failures,
  152 exact scientific comparisons, exact replicated Memcheck contexts, a
  13.051% median whole-step instruction reduction, and exact removal of half
  the `get_nnx` calls and allocation subtree. Six PNG/SVG graphs record the
  native, instruction, and allocation effects. `FLUID-FS-001` is now the
  active rank-one candidate; its scratch structure gate is red on this commit.
- 2026-09-02 14:41 PDT: Baseline job `41756734` completed all 32 heat-CG
  and spiral-BO profiles with exact VTK state, replicated normalized Memcheck
  contexts, Callgrind instruction spans below 0.279%, and Massif peak spans
  below 0.062%. Its full-directory controller included timing-bearing files,
  so the valid profiles were independently reconciled. Four-replicate evidence
  narrows scalar-RCS reallocation shares to 11.34--11.54% in heat-CG and
  6.71--6.82% in spiral BO, and `cep_2d` to 52.12--52.23% in spiral BO.
- 2026-09-02 14:35 PDT: Performance job `41757424` completed all 15
  alternating native pairs before a rank-launch harness stop. All pairs had
  byte-exact VTK state and exact normalized convergence histories; all 15
  favored the candidate, with a 9.259% median reduction and bootstrap and
  distribution-free lower bounds of 8.589%. Resume job `41758628` reuses that
  hash-verified native evidence and runs only corrected rank scaling and
  matched profiles. PETSc breadth job `41755468` completed all 32 profiles;
  independent reconciliation found exact VTK state, stable Memcheck contexts,
  and sub-0.189% instruction-count variation despite its overbroad original
  history/restart manifest gate.
- 2026-09-02 14:12 PDT: Full-suite gate `41753579` completed successfully.
  The first performance attempt `41754540` stopped before simulation because
  its case-manifest guard mixed relative and absolute filenames; corrected
  path-stable retry `41756560` is queued on a fresh artifact root. PETSc
  breadth job `41755468` is running. Added exact-tip ZeroD coverage job
  `41756567` and replicated heat-CG/spiral-BO queue baseline job `41756734`;
  the four active or queued profiles cap at four nodes and 80 GB.
- 2026-09-02 14:00 PDT: Added dependent validation job `41754540` for 15
  alternating native pairs, exact 1/3/4-rank output checks, and 32 replicated
  candidate-versus-parent profiler workers. Added dependent breadth job
  `41755468` for 32 pinned profiles across two representative PETSc cases;
  their XML inputs do not activate the changed weak-boundary path.
- 2026-09-02 13:44 PDT: Began one-file validation of
  `FLUID-WEAK-NNX-001`. The scratch structure gate was observed red on the
  parent source and green after the guarded equal-order reuse; source-hash-
  locked job `41753579` is rebuilding both configured backends and running
  the complete test suite. The scheduler returned no identifier for that
  accepted request and exposed it late; duplicate `41753587` then stopped at
  the artifact-ownership preflight without repeating work. Remaining merit-
  backed candidates no longer wait on a separate approval checkpoint.
- 2026-09-02 12:06 PDT: Re-ranked the queue after the completed three-lane
  review. Added `FLUID-WEAK-NNX-001`, `RCS-SCALAR-001`, `CEP2D-WS-001`, and
  `FLUID-STATE-001`, with bounded designs and correctness/performance gates.
- 2026-09-02 11:00 PDT: Created with `FLUID-FS-001` and `RIS-LOOKUP-001` as
  the first evidence-backed validation targets. Added two active discovery
  lanes for newly completed profiler output.

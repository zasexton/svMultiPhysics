# WP4 Conditioning-to-Qualification Goal and Implementation Plan

> **For agentic workers:** Use `superpowers:subagent-driven-development` or `superpowers:executing-plans`, with regression-first implementation and evidence-backed completion. The explicit ownership, reuse, protection, and authority rules below take precedence over generic workflow defaults. Do not restart completed work, replace useful existing workers, or add redundant reviews to adopt a workflow. Steps use checkbox (`- [ ]`) syntax for final milestone tracking.

**Goal:** Complete the remaining WP4 implementation and qualification by resolving the captured curvature mass-solve conditioning/publication blocker, making the preserved production minimizer pass its unchanged gates, completing the required three-dimensional and prescribed-angle integration, and producing a clean, reviewed, checksum-bound qualification record that justifies the precise audit closures.

**Architecture:** Begin with the selected unfiltered `KinematicAreaGradientTraction` and fixed-volume minimization of the same discrete surface-plus-wall energy. Treat geometry, retained support, curvature recovery, pressure work, constraints, and finite-precision representation as one coupled numerical contract. Prefer a justified method-preserving repair; do not conceal a change of physical method, finite-element space, retained domain, or qualification criterion inside an arithmetic fix.

**Tech Stack:** C++20, GoogleTest, CMake, MPI, Python qualification/checking tools, and Slurm on Sherlock. Existing accepted generations use GCC 12.4.0, OpenMPI 4.1.2, CMake 3.31.4, and Python 3.12.1; verify actual compiler flags, dependency closures, and runtime libraries before reusing their products.

**Spec:** [WP4 audit](free_surface_boundary_unfitted_audit_20260720.md), [balanced-force completion plan](plan_wp4_balanced_force_completion_20260903.md), and [prospective qualification gate contract](wp4_qualification_gate_contract_20260906.md). This document supplies a new execution goal and a current handoff, not evidence that those requirements have already passed.

Prepared on 2026-09-14, America/Los_Angeles, after a read-only reassessment. The newest inspected WP4 numerical acceptance records are dated **2026-09-07 07:38:17 UTC**. The handoff is deliberately explicit about historical versus current state. Reconcile again when execution starts.

## 1. Execution request and definition of success

Adopt this as an implementation-and-qualification goal, not a request for another plan or general literature report. Start with the bounded numerical decision in Milestone 1, implement its justified consequence, and continue through the production case and remaining WP4 requirements while authorized work remains.

The required critical path is:

```text
Reconcile live state and exact evidence
  -> Resolve conditioning/representation decision on the captured systems
  -> Implement and verify one justified repair on all four original states
  -> Reaccept the matching complete production evaluator
  -> Pass the preserved minimizer from its original start
  -> Complete the remaining geometry/wall/prescribed-angle integration
  -> Finish executable qualification contracts and integrated regressions
  -> Freeze clean source, fresh caches, and final matrix/input identities
  -> Execute and independently validate complete qualification
  -> Archive evidence and update only justified formal audit items
```

Independent geometry/contact integration and qualification-contract work may advance alongside the mass-solve path when their dependencies, source ownership, and resources permit. They must not delay the production-case critical path through overlapping edits or unnecessary verification.

Success means a working, integrated, physically qualified WP4 capability, not merely:

- a successful build;
- a passing helper or a corrected conservative rejection;
- a passing internal higher-precision vector that cannot be published;
- one captured evaluator state passing while the production minimizer fails;
- a selected collection of passing physical cases without the full declared matrix;
- a new proof, wrapper, diagnostic, or report without a completed next consumer.

Formal status at handoff: WP0–WP3 are checked; WP4–WP10 and Q0–Q7 remain open. Some narrow prerequisite boxes inside the WP4 notes are checked. Do not revert the qualified WP3 status based on older notes saying only WP0–WP2 were complete. Do not close WP7 merely because its conditioning concerns are addressed locally for WP4. WP5 dynamic wetting, WP6 transport, WP8 full time-discrete energy, WP9 fitted-ALE, and WP10 two-phase work are not automatically included in this goal except for concrete dependencies required by WP4.

The two primary WP4 findings are FSR-03, balanced capillary/pressure work, and FSR-04, prescribed-angle treatment. Balanced-force success alone does not close the prescribed-angle finding, all of WP4, or all of Q2. Map every proposed closure to the actual audit requirements and accepted evidence.

## 2. Paths, source identity, and instruction precedence

Use these aliases throughout this document. They denote existing paths, not directories to recreate:

| Alias | Exact path |
| --- | --- |
| `W` | `/scratch/users/zsexton/wp4-application-regression-fixes-20260902` |
| `D` | `W/.superpowers/sdd/plan_wp4_balanced_force_completion_20260903` |
| `C` | `/scratch/users/zsexton/wp4-continuation-20260906` |
| `P` | `C/positive-affine` |
| `E` | `C/captured` |
| `O` | `E/minimizer-2-mass-observation-3` |
| `R` | `E/minimizer-2-mass-repair` |
| `Q` | `E/minimizer-2-mass-oracle-1` |
| `BRIEFS` | `/scratch/users/zsexton/wp4-delegation-briefs-20260905-Z6OoSG` |

Required initial reading, in this order:

1. This entire goal.
2. `D/delegation-checkpoint.md`, beginning with its top active section, then `C/jobs.md` entries newer than that section. The checkpoint headline is **06:33 UTC**, but actual accepted results continue through **07:38 UTC**. Older sections contain obsolete live jobs and next actions.
3. `O/results/root-acceptance.json`, `O/results/published-residuals.json` using structured summaries, `Q/results/independent-terminal-review.json`, and `R/observation3-decision-tree.md`.
4. `W/Documentation/plan_wp4_balanced_force_completion_20260903.md`, particularly the required production milestone and execution-priorities section.
5. `W/Documentation/wp4_qualification_gate_contract_20260906.md` in full.
6. The FSR-03, FSR-04, WP4, relevant WP7, Q2, and final acceptance sections of `W/Documentation/free_surface_boundary_unfitted_audit_20260720.md`.
7. `BRIEFS/README.md`, `implementation.md`, `scientific-checks.md`, and `verification-review.md` in full before dispatching workers.
8. `W/Documentation/free_surface_architecture_refactoring_plan_20260904.md` and the current architecture owner's checkpoint before assigning overlapping files.

Load other linked reports only when they supply a particular missing premise or executable contract. Do not reread the entire historical scratch tree at every continuation.

Source snapshot during reassessment:

- Worktree branch: `wp4-application-regression-fixes-20260902`.
- Last implementation commit at that snapshot: `fc5e61be77f835e9a55ecaee554780de15b14971`, titled `Bind level-set derivative requests to authoritative sources`.
- Remote `origin`, branch `issue-449-modern-mesh-core`, matched that commit on readback during handoff preparation. A subsequent documentation-only handoff commit may advance HEAD without integrating the pending implementation.
- Common Git directory: `/scratch/users/zsexton/wp4-balanced-capillary-qualification-20260902/source/.git`.
- The working tree is **not clean**: excluding the protected test, it has 52 modified tracked files, approximately 11,766 inserted and 776 deleted lines, plus untracked source headers and the qualification contract. These are existing changes belonging to the ongoing work. Do not reset, stash indiscriminately, overwrite, or absorb them into an unrelated commit.
- Important untracked participating headers are `FE/Geometry/AffineTriangleLevelSetCut.h`, `FE/Geometry/AffineTetrahedronLevelSetCut.h`, `FE/Basis/TriangleBarycentricEvaluation.h`, and `FE/Basis/TetrahedronBarycentricEvaluation.h`, relative to `W/Code/Source/solver`.
- Candidate 3 of the mass repair is private and rejected. It is not the implementation in W. Do not treat a private candidate, a held combined overlay, and W as interchangeable.

The shared home checkout `/home/users/zsexton/svMultiPhysics` belongs to concurrent architecture work. **Do not modify it.** Work in W and assigned private scratch generations. A separate source copy does not remove common Git metadata or shared-cache conflicts.

No applicable `AGENTS.md` was found in the inspected locations during reassessment. Check again at startup; do not assume that absence remains true.

The execution-priorities amendment was already applied and pushed in `7bdac5b421c77dee4908e164b7cfecd9af160c48`. Do not reapply `D/wp4-execution-priorities-amendment-20260905.patch`.

## 3. Non-negotiable protection and authority rules

### Protected regression

This exact file is **hash-only**:

```text
W/Code/Source/solver/FE/Tests/Unit/LevelSet/test_LevelSetCurvatureProjection.cpp
SHA-256 12eb38e206cd587373c558e04fc9d0089b164588db538d0dbc2f5b6bdbad4d27
```

Never open, copy, edit, stage, compile, run, delete, or disable it. It contains a pre-existing protected change. Avoid unscoped diffs, whole-source copies, and broad build/test targets that would consume it. Do not run `all`, `test_fe_levelset`, or a broad MPI target without inspecting the target's actual source/test inventory.

Use the dedicated unprotected binding tests, assigned geometry/reinitialization tests, and isolated captured-mass fixture. If a required qualification lane genuinely conflicts with this protection, identify the exact conflict and obtain direction for that lane; never silently skip the protected requirement and claim complete coverage.

### Physical and numerical semantics

- Preserve the selected unfiltered `KinematicAreaGradientTraction`, the actual discrete surface-plus-wall energy, and its fixed-volume minimizer unless a material method change is explicitly authorized.
- Do not project the capillary force into the pressure-gradient range, enrich the pressure space, add a force filter, discard small positive modes/equations, prune support to avoid a failure, or manufacture differentiability through coefficient displacement.
- Keep source/candidate binding, canonical field/DOF mapping, constraint restriction, active-side orientation, physical metrics, retained pieces, and rollback intact.
- Surface tension is applied once at its owning consumer. Pressure work is not arbitrarily rescaled. Apply constraint transpose/restriction exactly once.
- A general FE pressure field is not the variation of the scalar product `p*V`. Its actual divergence/domain-variation identity must be used. In the one-phase model, gas pressure is a prescribed boundary load, and the liquid pressure space already contains constants; do not misdiagnose the current failure as automatically requiring a two-phase pressure-jump enrichment.
- Geometry success and classical derivative availability are distinct at exact ties or genuine topology changes. A truthful unavailable derivative is protection, not a repair of the failing production case.
- Numerical factorization, preconditioning, scaling, or an equivalent representation may be investigated as method-preserving repairs, with the exact equivalence and complete original residuals checked. Do not describe a changed rounded operator as the unchanged M64 evidence. A changed physical operator, function space, gauge policy, retention policy, or acceptance threshold requires an explicit prospective architectural decision.
- Do not relax physical tolerances, increase scientific iteration allowances, change stopping denominators, or count uncharged matrix actions merely to obtain a pass.

Routine scoped implementation, review, and verification should not wait for repeated ceremonial approval. Historical time-limited instructions about an eight-hour approval window are not a perpetual lease. Ask only when a genuinely new authority or material scientific choice is required, and state the smallest concrete decision needed.

### Commits, pushes, and authored content

The coordinator alone integrates, edits the shared audit, stages, commits, and pushes. Every commit must have both author and committer:

```text
Zachary Sexton <zsexton@stanford.edu>
```

Push every accepted coherent commit promptly to `origin HEAD:issue-449-modern-mesh-core`. Check remote ancestry first. If another contributor advanced the branch, reconcile explicitly; never force-push, reset their work, or include their dirty files incidentally. Do not alter global Git configuration or repair repository/goal metadata to make a command succeed.

Use an explicit path list for staging. Inspect the exact staged names, bytes, message, identity, and protected hash. A safe per-command identity pattern is:

```bash
GIT_AUTHOR_NAME='Zachary Sexton' \
GIT_AUTHOR_EMAIL='zsexton@stanford.edu' \
GIT_COMMITTER_NAME='Zachary Sexton' \
GIT_COMMITTER_EMAIL='zsexton@stanford.edu' \
git --no-optional-locks commit -m 'Record the accepted WP4 milestone'
```

The example title is not a prescribed title for every commit; name the actual bounded change.

Preserve the existing prohibition on assistant/tool-attribution vocabulary in all authored files and commit messages. The established whole-word scan avoids spelling those terms into new artifacts:

```bash
rg -ni '\b(?:\x63\x6f\x64\x65\x78|\x63\x6c\x61\x75\x64\x65|\x61\x69)\b' Documentation/goal_wp4_conditioning_to_qualification_20260914.md
```

The command above scans this handoff from W; repeat the same check on each actual authored path in a later commit. Exit 1 means no match; exit 0 blocks publication; exit 2 is a scan error, not success. Scan the exact proposed message and staged additions as well. Use `apply_patch` for authored file edits. Do not write credentials into files, logs, commands, or chat; use existing authentication or the platform's secure authorization route.

## 4. Sherlock resources, execution, and continuation

### Conservative current ceiling

The visible user history authorizes **three nodes on `amarsden`**. The existing plan later says four nodes and 40 GiB total. Until a newer explicit user authorization resolving that discrepancy is verified, enforce the stricter intersection: **at most three owned nodes and 40 GiB total concurrently**. Do not assume a document silently increases the user's limit.

At the latest live query, the only account job was development allocation `43486013`, `amarsden`, node `sh03-08n17`, **4 CPUs and 20 GiB**. No WP4 numerical job was running. This is a point-in-time observation, not a reservation for a future chat. Recheck scheduler and actual processes. If this development allocation is retained, conservatively count its node and 20 GiB against the ceiling. A node allocation does not authorize using every physical core on that node.

Old allocations `42053782`, `42257896`, and jobs in the September 6–7 ledger are historical, not resources to resume. Do not cancel unrelated account jobs. Earlier permission concerning queued WP10 jobs is not blanket cancellation authority; no WP10 cancellation is part of this goal.

### Submission and runtime requirements

- Every new job must include `--mail-user=zsexton@stanford.edu` and `--mail-type=BEGIN,END,FAIL` **at initial submission**, including build, test-only, diagnostic, and qualification jobs.
- Record the exact owned job, source generation, command, node count, CPUs, memory, walltime, output paths, and guard owner at submission. Preserve original failed/inconclusive results.
- Preserve inherited Sherlock modules. Do not use `bash -lc` inside `srun`. Review `C/stack.sh` and accepted environment records for reproduction, but do not source an old environment script blindly over a different active toolchain.
- Missing modules or runtime dependencies are environment issues to resolve explicitly, not reasons to silently substitute a compiler or MPI library.
- Numerical experiments and builds belong in a named authorized allocation. Planning, source reading, and lightweight artifact inspection do not justify a new numerical job.
- Use actual fresh compilation count and measured memory to size work. Recent one-TU mass/geometry jobs used one CPU and approximately 1 GiB peak; two-TU Application work used about 5 GiB. Full Application MPI used two CPUs at about 98% utilization. These are measurements, not universal upper bounds.
- Use separate writable caches for concurrent lanes. Existing accepted caches are read-only unless explicitly reassigned after all guards release.
- Reuse verified unchanged objects only after checking successful-product identity, actual dependencies, compiler flags, archive member identity/order, and linkage. Untracked participating headers are inputs too.
- Capture actual compiler helpers, implicit linker inputs, runtime libraries, and assets in the initial provenance closure. The earlier missing-linker/runtime-path failures demonstrate why a clean source hash alone is insufficient.
- Configure before freezing a run. During a guard, freeze participating source contents, HEAD, raw index, recipes, and inputs. A command finishing or `squeue --steps` showing only `extern` does not release its wrapper guard.
- Do not edit tracked documentation or commit during an affected source/index guard. Record transient progress outside guarded paths.
- Do not automatically retry a failed command. Classify it as setup, build, guard, numerical, or inconclusive; reuse valid products and rerun only what the correction invalidates.

### Continuation without duplication

The new coordinator should create one unique scratch run directory using `mktemp -d`, below `/scratch/users/zsexton`, and record its exact path in `D/delegation-checkpoint.md` after a safe reconciliation. Put new reports and outputs there, not inside old frozen generations. Do not create another source worktree merely because a new chat started; W already is an isolated linked worktree.

Maintain a concise active checkpoint outside guarded source paths containing:

- current source commit and dirty-source/participating-input identities;
- each worker's exact ownership and current assignment;
- accepted evidence and its next consumer;
- live job IDs, actual process/session routes, wrapper/output paths;
- source, cache, and index guards and their release status;
- resource reservations and measured current bottleneck;
- the next exact action and the condition that authorizes it.

Keep detailed historical hashes and transcripts in their existing artifacts, not repeated in every status message. On continuation, inspect the recorded route before submitting anything. Session interruption does not prove a process died, a guard released, or a job failed. Checkpoints reduce restart loss; do not promise an uninterrupted session. Keep user updates concise and avoid blocking waits longer than 60 seconds.

## 5. Accepted evidence and the actual remaining failure

### 5.1 Real progress to retain

- Authoritative snapshot/functional/source binding and lifecycle policy propagation were committed in `fc5e61be` and `d229bb14`, respectively. They are not new tasks to recreate.
- Positive original-coefficient geometry, explicit positive-piece retention, fresh producer-vector gathering, full-cell cache updates, and original barycentric transport repaired genuine defects in the captured route.
- C9, job `42292524`, accepted all three original captured producer states and the unchanged complete checker. Its acceptance is `E/barycentric-c9-acceptance.json`; detailed results are `E/build-9/results/captured-analysis.json`.
- C9's worst published-curvature moment defect is approximately `7.46905e-11` against `1e-8`; worst energy-work defect is approximately `7.93843e-9` against `1e-8`. This is accepted bounded evidence, not a uniform estimate over future minimizer states.
- Matching FE job `42292517` passed 190 serial cases and both required cases per MPI rank.
- The unchanged C9 Application generation passed 380 serial cases and 31 distinct MPI cases represented by 64 rank records, including the separate four-rank control.
- Strict Triangle3 wall job `42306395` passed 208 serial cases and four cases per MPI rank.
- Focused Application contact job `42318952` passed four selected cases from 383 registered tests. It did not run all 383 tests. Acceptance: `C/application-wall-green-run-1/results/root-acceptance.json`.
- Tetra geometry job `42319312` passed nine new and seven adjacent cases, with six unchanged basis cases retaining earlier accepted evidence. Acceptance: `P/tetra-geometry-green-1/results/root-acceptance.json`. Tetra consumers, wall derivatives, Application integration, and full qualification remain open. An older checkpoint calling the helper only a stub is superseded by this bounded helper acceptance, not by full Tetra acceptance.
- The sampled-admission argument-selection fix already has one accepted focused regression in `C/qualification-admission/green-1`. It changes the sampled argument to `1.0` while preserving minimized `1e-8`. This is partial executable adoption, not completion of the full qualification contract.

### 5.2 Preserved minimizer failure

The original 900-second external limit ended without a scientific verdict. Its reviewed 3600-second successor, job `42294456`, did reach a scientific failure:

- 147 iterations and 2,071 functional evaluations;
- projected gradient `0.022007215754989533`;
- volume error `2.6566787103554645e-7`;
- both required against `1e-10`;
- the preserved 128 topology-transition allowance was reached;
- source/retention preflight passed before `authoritative_equilibrated_consistent_mass_solve_failed`;
- no publication certificate was emitted.

The scheduler job elapsed about 2,961 seconds; the solver wrapper records about 2,908.83 seconds. Do not conflate these timings. This failure was not simply a timeout.

Paths:

- `E/minimizer-2/results/result.json`, `solver.stdout`, and `solver.stderr`;
- `E/minimizer-2/case/solver.xml` and its exact preserved companion inputs;
- `E/minimizer-2-review/terminal-capture.json`, containing the terminal state and 40 trials;
- the accepted four-state replay and oracle under `E/minimizer-2-mass-oracle-1` and the subsequent mass-regression/observation generations.

The four decisive states are accepted base 147, accepted trial 17, trial 18, and trial 39. Their full row/DOF extent is 1,681. Structurally inactive rows may be handled through the existing exact map, but no supported row may be discarded by a numerical threshold.

The preserved PositiveMeasure case is an already-recorded successor to the earlier legacy-retention case. Do not add that option again, reconstruct input files from memory, switch to the old legacy inputs, or silently add further scientific changes. Preserve both original lineage and the actual current six-file case identity.

### 5.3 What the mass experiments prove

The public solver gate is evaluated on the **published `Real` coefficient vector**, with the complete original rounded `M64` and RHS. It is not a gate on the internal recurrence alone.

The implementation in W uses `Work = long double`, diagonal scaling, and the analytically identified level-set scaling direction per component. It checks:

1. complete original-RHS-relative residual;
2. complete diagonally scaled-RHS-relative residual;
3. worst componentwise row backward residual.

The primary tolerance remains `1e-10`; fallback remains `5e-13`. The primary budget is `max(200, 20*n)`, which is **33,620** for these states. Candidate endpoint assessments consume that same allowance. The fallback has its existing separate route/budget; do not accept a fallback iterate using the primary tolerance just because that would pass.

Candidate 1 failed base 147 and trial 39. Candidate 2 restored the base and trial 18 but still failed trial 39. Candidate 3 added fixed-bracket revisits and still failed trial 39. All are preserved; candidate 3 is not integrated.

Latest observation, job `42319313`:

- eleven native cases, ten passing and only `CapturedAuthoritativeMass.PublishesTrial39` failing;
- eight returned files exactly reproduced from candidate 3;
- 330 Krylov updates plus 33,290 alternate assessments, totaling 33,620;
- 89 settled fixed-target calls and one budget-stopped call;
- primary exit `primary_rounding_budget`;
- best saved settled Work scaled residual `6.38685464156e-11`;
- corresponding nearest Real scaled residual `1.39345332346e-10`;
- corresponding retained Real scaled residual `1.07795782662e-10`, still failing;
- the retained original-relative residual `9.18646581202e-11` passes, but all three gates are required;
- candidate 2's earlier best scaled value, `1.04791568768e-10`, was slightly better but also failed.

Over 99% of the latest charged primary allowance was spent on alternate assessments. This is an accounting statement, not a measurement that 99% of total walltime went there.

The passing base 147 publication has scaled residual `9.9731285535e-11`, about 0.27% below the gate. Record this narrow margin without inventing a new arbitrary safety-margin pass criterion. Three actual passing states are feasibility witnesses for those states, not a robustness theorem for neighboring cuts.

The independent oracle at 80 and 120 digits agrees across two gauges. Higher-precision quotient solutions followed by nearest Real conversion fail for recorded base/trial39 quotients. However, the accepted base recurrence vector passes: **a failed rounded exact solution is not proof that no acceptable approximate Real vector exists**.

Trial 39's measured original-data matrix-rounding action is about `1.632258e-10`; Real conversion action is about `8.936076e-11`; RHS rounding is about `9.414042e-16`. These identify sensitivity, not a proof of global infeasibility or a missing source-binding defect.

For fixed original operands and a published vector `p = x + delta_x`, distinguish:

```text
r64(p) = r64(x) - M64 * delta_x
M64 = Mgeom + delta_M, b64 = bgeom + delta_b
r64(p) = rgeom(p) + delta_b - delta_M * p
```

The high-precision quotient oracle is not a replay of every native scaled-entry Work instruction. State that distinction whenever interpreting it.

A completed native no-improvement sweep proves only that this selector stopped under its recorded decisions. It is not independent mathematical local optimality, global optimality, or infeasibility. A continuous-box relaxation contains the passing Work target and cannot certify impossibility for that box. A lower bound on a fixed endpoint set says nothing automatically about all possible binary64 vectors.

The small coupled revisit witness has already been implemented and executed. Do not propose it as a missing test. It proves the revisit mechanism on that control, not success on trial 39.

### 5.4 Earlier failure modes that must not be rediscovered

- Corrected sampled/minimized gate dispatch is necessary; a sampled cap is not automatically an algebraic discrete minimum.
- Radius propagation and stationary direct-solver default propagation had real defects in an older pilot. Preserve their regression fixes.
- Tiny barycentric values were previously lost by reconstructing a complement from rounded reference coordinates. Original parent basis entries now have explicit carriers in accepted slices. Do not regress to coordinate reconstruction, normalization, or rounded-point deduplication.
- Tiny supported equations were previously missed by normal-residual-only fallback acceptance. Keep complete published-vector residuals authoritative.
- Application maintenance previously supplied a raw state span rather than the authoritative FE-ordered backend vector. The gather repair and current source/cache binding must remain.
- There were genuine setup-only failures: inactive `#else` test inventory, nonexistent API members, a duplicated fixture entry point, missing implicit linker inputs, and missing runtime-path provenance. Preserve their classifications and successful reusable products; do not call them numerical regressions.
- The oldest captured near-zero/tie-displacement, retained-boundary, affine-normal, arithmetic-kernel, and factorized-certificate investigations remain background evidence. Reopen them only for a specific contradiction or changed dependency, not because a new chat began.

## 6. Work packages and concrete acceptance

### Milestone 0 — Reconcile once and reserve ownership

**Deliverable:** one current checkpoint, exact participating-input inventory, and bounded worker packets; no duplicate submissions.

1. Read the required records and check for newer results.
2. List live workers and inspect scheduler/process routes. Reuse workers only if they actually exist and have appropriate context. The reassessment workers were read-only reviewers and owned no implementation files; historical names in checkpoints are not current reservations.
3. Inspect W's HEAD, index, dirty path list, protected hash, private candidate identities, and the architecture work boundary. Preserve all existing work.
4. Identify reusable accepted products and held/unexecuted generations. In particular, `E/build-10` and the combined candidate-3 wall overlay were held, not accepted new production evidence.
5. Create the new scratch output root, reserve source/cache ownership, and name the exact numerical decision packet.
6. Immediately proceed to Milestone 1 and useful independent work. Reconciliation is setup, not a terminal result.

Useful initial read-only commands, from W:

```bash
export PATH=/share/software/user/open/git/2.45.1/bin:/share/software/user/open/git-lfs/2.4.0/bin:$PATH
export GIT_OPTIONAL_LOCKS=0
git --no-optional-locks rev-parse HEAD --git-dir --git-common-dir
git --no-optional-locks status --short
git --no-optional-locks diff --cached --name-only
git --no-optional-locks ls-remote origin refs/heads/issue-449-modern-mesh-core
squeue -u zsexton -o '%.18i %.16P %.40j %.8T %.12M %.12l %.5D %.6C %.12m %R'
ps -u zsexton -o pid,ppid,etime,time,args
module list
sha256sum Code/Source/solver/FE/Tests/Unit/LevelSet/test_LevelSetCurvatureProjection.cpp
```

Do not run an unscoped content diff. Use exact owned files or explicitly exclude the protected path. Do not regard read-only scheduler inspection as job authority.

### Milestone 1 — Make the conditioning/representation decision

**Deliverable:** a bounded numerical decision tied to the original captured systems, with a concrete repair consumer. Place it under the new scratch root as `conditioning/decision.md`, accompanied by machine-readable source/input identities and metrics. The report is an intermediate artifact, not goal completion.

**Initial owner:** one scientific worker or the coordinator, read-only on production. One future implementation owner may independently inspect interfaces but must not start a numerical method dependent on an unresolved premise.

**Inputs:** the four original M64/RHS systems; accepted gauges/scales and structural support map; original geometric moments; saved Work targets and Real outputs; actual trace, assembly, constraints, and energy-work maps from the captured production route.

**Required investigation:**

1. Define the exact supported quotient and distinguish the analytically redundant scaling direction from small physical modes. Do not infer numerical rank from an arbitrary threshold or invert the tiny rounded remnant of the analytic scaling mode as though it were physical.
2. Quantify the remaining quotient conditioning or weak singular/eigenmodes sufficiently to explain the observed residual sensitivity. Use an independent stable factorization or bounded precision; a backward-stable result must state its norm and residual. Do not densify a large inactive global system when the exact active map is available, but restore and check the complete original rows.
3. Reuse the accepted matrix-rounding/publication decomposition. Establish which operations and modes consume the available residual margin. Do not merely increase Decimal precision beyond already agreeing 80/120-digit results.
4. Carry the same error directions into the actual trace/installed force and energy-work observables. Use the actual constraint maps exactly once. A coefficient norm, a row backward error, and a physical force/work error are different quantities.
5. Evaluate a quantitatively justified path to a publishable vector or a stable equivalent representation. A successful witness must be the actual complete Real vector under all original gates. State the domain of any negative bound; failure of a heuristic or restricted endpoint set is not global impossibility.
6. Use nearby-state or cut-position controls only when they test a named conditioning hypothesis and preserve the original branch/source convention. Label sampled robustness evidence as such. Do not introduce an arbitrary new margin threshold or a new physical gate.

**Bound the inquiry:** start with one decisive experiment on trial 39 and the three existing feasibility controls. Before launching any follow-up, identify the exact unanswered observation, why the first experiment could not answer it, and what implementation decision the follow-up changes. Do not build a generic certification framework, enumerate an unbounded space of rounding heuristics, or repeat a settled sweep. One independent review of the exact result is sufficient unless a concrete finding changes it.

**Required decision:**

- **Method-preserving repair supported:** name the exact operation/representation to change, why it addresses the measured limiting error, which invariant it preserves, and the focused regression that distinguishes it from the rejected candidates. Proceed directly to Milestone 2.
- **Specific structural change required:** provide the quantitative evidence and the smallest proposed change, its effect on discrete energy/pressure work/support/gauge and gates, and the precise new authorization needed. Do not silently switch methods. Continue independent authorized integration while the decision is pending.
- **Inconclusive:** name the missing quantity and one bounded way to resolve it; do not label it impossibility, success, or permission to continue blind patches. Escalate a genuinely necessary scientific choice instead of indefinitely expanding prerequisites.

The objective is a usable implementation decision, not a universal theorem over every binary64 input or every future WP7 configuration.

### Milestone 2 — Implement and accept the complete four-state repair

**Primary production seam:** `W/Code/Source/solver/FE/LevelSet/LevelSetCurvatureProjection.cpp`, especially `solveAuthoritativeKinematicAreaMass`. Its handoff location is around lines 2779–3000; relocate by symbol before editing. The existing header/API is in the same directory.

Use one owner for this compilation unit. Changes outside it require an explicit dependency/ownership addition, not incidental refactoring. If the measured repair needs a factorization helper, keep its API narrow, its physical inputs explicit, and its tests separate from the protected file.

**Reusable exact fixture and recipe:**

- `O/source/test_CapturedAuthoritativeMass.cpp`;
- `O/recipe/prepare.py`, `run.py`, `plan.json`, and `owner-packet.json`;
- `O/recipe/check-published.py` and `accepted-mass-metrics.py`;
- `O/recipe/expected-returns.json`;
- `O/run/case/inputs.json` and the named four state TSV/Work-vector files.

Inspect these before reuse. They contain old absolute output paths and frozen identities. Adapt the reviewed preparation to a new unique generation; **do not execute an old wrapper against its old results directory**. Freeze new inputs before running, and preserve old expected failures.

The existing exact native test inventory is:

```text
CapturedAuthoritativeMass.AcceptedBase147
CapturedAuthoritativeMass.AcceptedTrial17
CapturedAuthoritativeMass.PublishesTrial18
CapturedAuthoritativeMass.PublishesTrial39
AuthoritativeMassPublicationControls.ExactlyRepresentableValuesHaveNoAlternate
AuthoritativeMassPublicationControls.SelectsOnlyAdjacentEndpointOfFixedTarget
AuthoritativeMassPublicationControls.EqualCompleteScoreKeepsIncumbent
AuthoritativeMassPublicationControls.BudgetPrecedesEveryAlternateAssessment
AuthoritativeMassPublicationControls.NearestPassingSolveKeepsCoefficientBits
AuthoritativeMassPublicationControls.RevisitsEarlierCoordinateAfterLaterImprovement
AuthoritativeMassPublicationControls.BudgetStopsBeforeRequiredRevisit
```

Retain the four physical-state regressions. Reuse applicable publication controls. If the justified repair replaces an internal selector, do not fake that its implementation-specific tests establish the new method: preserve their historical evidence and add controls for the new invariant, with an explicit acceptance inventory and unchanged physical gates.

Execution steps:

1. Preserve the accepted runtime failures as RED evidence for unchanged inputs. Add and witness any newly necessary discriminator; a compiler error or missing fixture is not the required RED.
2. Implement the minimum justified repair on a private frozen source generation.
3. Run the four original states and all applicable controls with actual production flags and original native gate semantics.
4. Independently evaluate every actual published vector at the accepted 80/120-digit checks against the complete original M64/RHS, all three ratios, and the correct route tolerance.
5. Preserve exact source/field/support/gauge/range behavior and honest iteration accounting. A zero failure-output vector is not an inspectable internal candidate.
6. Run only adjacent coverage invalidated by the patch, including collective behavior if the changed path affects MPI.
7. Review the exact patch and evidence, centrally integrate after all affected guards release, and commit/push a coherent accepted slice.

**Exit:** all four states have actual publishable success under unchanged gates, controls pass, the patch is reviewed and integrated, and the remaining error/conditioning limits are documented. This milestone does not assert that the nonlinear minimizer or all WP4 cases will pass.

### Milestone 3 — Reaccept the complete evaluator and pass the original minimizer

**Owning integration paths:** Application candidate/maintenance recovery and current geometry/constraint binding, the accepted captured probe/checker, and the full production solver link.

Relevant existing evidence and recipes:

- `E/build-9` and `E/barycentric-c9-acceptance.json`;
- `E/check-captured.py` and the source-specific probe/recipe recorded by C9;
- `E/solver-full-4` and `E/minimizer-2`;
- newer reviewed wall/Application and solver-link generations in the ledger;
- `E/build-10` and `R/wall-candidate-3`, which were held and must not be promoted simply because they exist.

Required steps:

1. Freeze a new integration generation containing the accepted repair and exactly the intended geometry/wall/Application source set. Reuse compatible products after actual dependency/flag checks; rebuild every invalidated consumer, including headers used outside FE.
2. Execute the same three-state captured producer controls and complete checker: source/DOF binding, original geometry, strict same-branch scalar increments, mass/volume identities, actual installed surface/pressure/wall work, admitted constraints, and rollback.
3. Require the existing numerical gates. Reusing a prior passed geometry calculation is valid only if its inputs and relevant arithmetic are unchanged; new force/publication consumers need their actual evidence.
4. Link the **actual production solver**, not only a probe that includes Application source. Preserve complete actual linker/runtime input provenance. Reuse the previously retained 70 compatible solver objects only if still valid.
5. Run the preserved minimizer from its original initial case, not from terminal base 147 or a hand-adjusted near-solution. Preserve the exact input lineage and unchanged tolerances/limits listed in Section 7.
6. Inspect the actual final convergence and publication certificate, physical balance, volume, gradient, and source revisions. Solver exit alone does not certify physical success.

**Exit:** the original production case passes its complete unchanged convergence/publication gates and has a source-bound accepted result. If another failure appears, capture that precise state and distinguish a genuinely new mechanism from the same unresolved repair. Do not rerun the 49-minute case simply to reproduce a known failure.

A robust inner solve may expose a remaining nonsmooth minimization or topology-transition problem. There is no proof that a fixed-background P1 energy minimum always has a classical zero gradient. If this becomes the actual blocker, inspect the same discrete functional's branch/one-sided behavior and the production optimizer contract; do not invent a classical derivative by moving ties or weaken the convergence definition silently.

### Milestone 4 — Complete remaining geometry, wall, and prescribed-angle consumers

This can advance independently where ownership permits. Do not restart the accepted Triangle/Tetra helper work.

**Accepted contracts to reuse:**

- `C/geometry/tetra4/contract.md`, `interface.md`, `geometry-notes.md`, and `owners.json`;
- `P/geometry/header-contract.md` and `barycentric-contract.md`;
- `P/retention/prescribed-contact-contract.md`;
- `P/geometry/wall-extension/prescribed-contact-review.md`;
- the newest Tetra helper and Application contact acceptance records, not only older readiness notes;
- `D/task-4-prescribed-angle-gap-map.md` and `D/task-4-prescribed-scaling-design-report.md` for remaining stage/cadence/strip obligations.

**Source boundaries to assign explicitly:**

| Area | Main paths under `W/Code/Source/solver` |
| --- | --- |
| Original affine geometry and basis | `FE/Geometry/AffineTriangleLevelSetCut.h`, `AffineTetrahedronLevelSetCut.h`; `FE/Basis/TriangleBarycentricEvaluation.h`, `TetrahedronBarycentricEvaluation.h` |
| Producers, carriers, and retained rules | `FE/Interfaces/LevelSetInterfaceBuilder.cpp`, `LevelSetInterfaceDomain.h`, `FreeSurfaceGeometrySnapshot.*`, `GeneratedActiveBoundaryDomain.*`, `GeneratedInterfaceBoundaryIntersectionDomain.*`; `FE/Geometry/CutQuadrature*` |
| Lifecycle, volume, restart | `FE/LevelSet/LevelSetInterfaceLifecycle.*`, `LevelSetImplicitCutQuadratureBackend.cpp`, `LevelSetVolume.*`, `LevelSetRestart.*` |
| Actual field and assembly consumers | `FE/LevelSet/LevelSetCellEvaluator.*`; `FE/Assembly/AssemblyContext.*`, `CutIntegrationContext.h`, `StandardAssembler.cpp` |
| Recovery and prescribed maintenance | `FE/LevelSet/LevelSetCurvatureProjection.*`, `LevelSetReinitialization.*` |
| Application integration | `Application/Core/ApplicationDriver.cpp`, `LevelSetCutConfiguration.*`, `LevelSetMaintenanceConfiguration.cpp`, related sample/history adapters, `Parameters.cpp` |
| Unprotected tests | `FE/Tests/Unit/LevelSet/test_LevelSetAuthoritativeDerivativeBinding.cpp`, reinitialization serial/MPI tests, assigned geometry/basis/assembly/lifecycle/volume/restart tests; `Application/Tests/Unit/test_ApplicationDriverLevelSetWorkflows*.cpp` |

Paths are a dependency map, not concurrent write grants. Reconcile architecture moves and avoid two writers to any overlapping source/compilation unit. Split the work by complete testable consumers, not declaration-only sweeps.

Required Tetra completion:

1. Preserve four original barycentric entries and independent crossing weights throughout geometry, both positive phase fans, interface moments, snapshots, mapping, lifecycle, and volume/restart paths. Never reconstruct a tiny complement from rounded coordinates or delete a positive piece.
2. Integrate actual scalar P1/P2 evaluation where the existing contract requires it, including product-space factors, true DOF permutations, test/trial/unknown/history/prescribed fields, repeated and two-sided contexts. Basis-dispatch acceptance is not full higher-order flow qualification.
3. Use the same original pieces for geometry, scalar derivatives, consistent mass, and actual installed quadrature. Preserve explicit measures in the correct reference/physical frame and the requested polynomial exactness.
4. Complete three-dimensional wet-face and contact-line transport, with length measure on the line rather than the two-dimensional counting measure. Preserve oriented wall normals, raw Young-angle convention, global ownership, and surface-tension ownership.
5. Carry both original endpoint barycentric arrays and face identity for a Tetra contact line. One arbitrary contact point plus a tangent is insufficient. Validate full trace constraints at both endpoints, converted target coefficients, local gradients, angles, and endpoint displacement.
6. Handle compatible multiple wall constraints on one parent jointly; detect genuine incompatibility before mutation. Do not choose the first record silently.
7. Preserve the accepted zero-isovalue restriction of the selected prescribed path unless broader support is explicitly designed and verified across discovery, target, and residual conventions.
8. Reuse all already accepted helper sign/permutation/oracle controls; add only missing consumers and newly affected cases. Complete actual rebuilt Tetra M/B, volume-divergence, installed surface/pressure/wall work, strict increments, rollback, and owner/ghost checks.

Required prescribed-angle completion beyond a local helper:

- positive coefficient scaling, fixed-point behavior, finite relaxation, and unchanged insufficient-iteration/motion-bound rejection semantics;
- independent wall and bulk-redistance schedules, with actual event counts;
- stage/anchor consistency: angle, geometry, force, maintenance, and accepted endpoint must refer to the declared common stage;
- actual canonical parent-corner to scalar field DOF mapping and complete MPI transport/duplicate identity;
- curved three-dimensional shared-DOF wall strips, not only isolated affine cells;
- both liquid signs, wall orientations, coordinate translations and scaling, and prescribed angles 30, 60, 90, 120, and 150 degrees within the declared envelope;
- actual production work and convergence evidence demonstrating that momentum Young energy and level-set geometric maintenance do not impose contradictory work.

The accepted Triangle contact formula retains its raw barycentric carrier. Sorted cell DOFs are not automatically canonical corner mapping; the carrier's sum must not be assumed exactly one. Use the original local target/displacement formulas in the accepted contract and validate the complete converted residual. Do not recover contact location by subtracting coincident rounded points.

Required physical coverage for full WP4, using the existing audit/matrix definitions and reusing valid unchanged evidence:

- Flat and hydrostatic equilibria across every declared coordinate direction, wall orientation, active liquid sign, gravity direction, cut offset, and pressure-gauge treatment. Separate exactly representable field assembly from computed-solution error. Preserve the existing free/fixed-gauge and nonzero-gravity checks.
- Closed circles in two dimensions, closed spheres in three dimensions, and sessile caps at 30, 60, 90, 120, and 150 degrees, with declared wall rotations, mesh-relative offsets, and refinements. Label sampled analytic geometry and minimized discrete geometry separately.
- For each required equilibrium, retain pressure jump, best pressure-space residual, installed capillary/physical residual, parasitic capillary number, kinetic energy, liquid volume, base radius, apex height, and contact angle with the correct state/norm identities.
- Actual production energy-adjoint coverage, including the degree-two affine Tetra quadrilateral interface with a nonconstant velocity and its order-one negative control from Task 3. Retain its tighter local `1e-12` comparison and nonzero mismatch discriminator where applicable; a passing quadrature helper alone is insufficient.
- Restoring-force coverage around a minimized state: opposite volume-orthogonal perturbations, positive second energy difference, capillary work opposing displacement, and a short production advance with the restoring sign. The unperturbed control must retain its equilibrium/parasitic bounds. Use the same final physical horizon under time refinement, not merely the same number of steps.
- At least the declared independent two-rank ownership/numbering layouts for nontrivial minimized sessile and closed-surface cases, plus required four-rank controls. Check accepted/rejected disposition, source/topology revisions, quadrature order, certificate flags, and physical observables; repeated launches of the same layout do not supply distinct partition coverage.

**Exit:** the required geometry and prescribed-angle capability is integrated, with source-bound serial/MPI and physical evidence for the declared WP4 scope. Fail-closed unsupported paths remain explicit, but rejection of required cases cannot substitute for their implementation.

### Milestone 5 — Finish the executable qualification contract

**Primary files:**

- `W/Documentation/wp4_qualification_gate_contract_20260906.md`;
- `W/tests/cases/fluid/run_free_surface_wp4_balanced_capillary_matrix_v3.py`;
- `W/tests/cases/fluid/free_surface_wp4_balanced_capillary_matrix_v3.json`;
- `W/tests/test_free_surface_wp4_balanced_capillary_matrix_v3.py`;
- inherited metric/convergence consumers in `run_free_surface_wp4_balanced_capillary_matrix_v2.py` and `free_surface_convergence.py`;
- `W/tests/cases/fluid/open_vessel_free_surface/run_test05_velocity_growth_smoke.py` and `W/tests/test_open_vessel_smoke_gates.py`;
- actual diagnostic emitters in `FE/TimeStepping/NewtonSolver.cpp` and Application maintenance logging, only with explicit ownership.

Reuse `C/qualification-adoption-map.json` and `C/qualification-metric-binding-packet.json`. Their source hashes are historical inspection identities: relocate symbols and compare intervening changes. The sampled argument-selection regression is already accepted; do not redo it unchanged as if no adoption exists.

Required remaining work:

1. Dispatch by initialization, refinement axis, quantity, and measurement phase. Unknown or missing combinations cannot silently inherit a global gate.
2. Keep sampled stationary pressure fitting as admission only, with the established `d <= 1` argument for the supported homogeneous constrained space. Preserve finite/stationarity/non-breakdown/pressure-only-publication requirements, zero-load behavior, and independent absolute checks. Minimized equilibrium retains its tight gates.
3. Define norm, units, normalization, expected/minimum rate, algebraic floor, and theoretical or empirical justification for each qualified quantity/dimension/boundary treatment **before** freezing results. Do not fit a rate or uncertainty to rescue a failed sequence.
4. Require spatial levels `R/h = 8,16,32`; add 64 when the triplet does not establish the asymptotic regime. A missing or unresolved required level is inconclusive, not a pass. Retain GCI safety factor 1.25 and continuum-target consistency, not just decreasing successive differences.
5. Do not impose a universal 0.8 convergence rate or make an already algebraically satisfied minimized residual show a positive slope. Degree-two facet quadrature does not prove second-order force/curvature convergence.
6. Keep coefficient scaling invariance separate from absolute spatial accuracy. Establish physical similarity and norm transformation before interpreting dimensional scaling. Compare time and maintenance-cadence studies at fixed mesh with their declared reference/uncertainty and common physical horizon.
7. Join pressure, force, geometry, shape, and maintenance metrics from the same actual accepted state. Emit/propagate accepted step/time, state and geometry revisions, maintenance identity, phase, and a common record identity where needed. Current independently selected latest-available records are insufficient; availability/match booleans are not actual revision keys.
8. Add negative tests for cross-state/phase joins, missing rate/reference data, out-of-bound or nonstationary sampled fits, false sampled equilibrium, resolved nonzero limiting error, missing required levels, and fixed-mesh references. Preserve the existing radius/direct-solver propagation controls.
9. Run the complete relevant runner tests, then a dry matrix expansion. Independently reconstruct case union, arguments, resources, finite horizons, actual maintenance counts, expected artifacts, and all source/matrix/runner identities.

The old frozen V3 expansion had 2,136 cases and 12,874 expected artifacts. Those are historical counts, not numbers to force onto a corrected successor. Recompute the successor's exact complete union and explain any legitimate change without shrinking required scientific scope.

**Exit:** executable criteria match the approved prospective contract, metric identities are complete, focused positive/negative controls pass, and the final source-specific matrix is ready for freeze. This milestone supplies no physical qualification by itself.

### Milestone 6 — Integrate, qualify, archive, and close honestly

1. Integrate reviewed slices in stable coordinator-owned windows after all affected guards release. Preserve concurrent architecture history and the protected dirty file. Record which exact source generation each accepted result validates.
2. Run justified FE, Physics, and Application regression lanes concurrently with separate caches and explicit resources where the current ceiling permits. Verify actual registration/counts, failures, skips, MPI ranks, and required four-rank controls. A full-suite claim must match its actual source/test inventory and protection rules.
3. Commit and push the accepted implementation and corrected qualification inputs. Keep the protected pre-existing change out of the commit. A clean qualification source is a fresh checkout of the complete accepted commit, not a claim that the protected dirty development worktree became clean.
4. Create the prescribed fresh detached scratch qualification source at that exact commit, hydrate required LFS objects, verify clean contents, and create genuinely fresh qualification caches/output paths. Record final source tree, matrix, runner, physical-runner, compiler, MPI, and dependency hashes before submission.
5. Launch only reviewed hash-bound qualification groups after the production pilot, remaining WP4 implementation, contract adoption, and freeze gates pass. Record every owned job with submission-time mail and actual resource accounting.
6. Independently reconstruct the complete raw evidence union, numerical acceptance, asymptotic/uncertainty checks, MPI properties, source/state identities, and execution/provenance envelope. Keep failures and inconclusive outcomes intact.
7. Archive accepted evidence in a new checksum-bound directory under `W/Documentation/qualification_logs/`, with actual immutable bytes, no accidental symbolic links or licensed reference material, and the required content scan.
8. Update the shared audit centrally. Close FSR-03 only on balanced-force evidence, FSR-04 only on complete prescribed-angle evidence, WP4 only when all its exits pass, and Q2 only if its complete hydrostatic/capillary/contact-equilibrium scope is satisfied. Leave unrelated WP/Q items unchanged.
9. Commit/push each coherent archive/audit change with the required identity. Verify remote readback and deliver a concise final report with exact commit, archive, matrix identity, counts, physical maxima, and any remaining scoped exclusions.

If the complete declared scope does not pass, keep the formal items open and give the precise failed/inconclusive requirement and next action. Never equate budget exhaustion or the end of a chat with completion.

## 7. Fixed acceptance limits and their scope

Read the original source-specific contracts rather than using this table as a replacement for definitions or comparison operators.

| Quantity or control | Required value / rule |
| --- | --- |
| Captured mass primary gate | `1e-10` for original, scaled, and worst-row ratios after Real publication |
| Captured mass fallback gate | `5e-13` for the same complete tests on its own route |
| Captured primary work budget | `max(200,20*n)`; 33,620 for extent 1,681; charge every alternate assessment |
| Preserved minimizer volume tolerance | `1e-10` |
| Preserved minimizer projected-gradient tolerance | `1e-10` |
| Pressure-representability residual/distance | existing `1e-8` minimized gates |
| Physical equilibrium and constant-pressure KKT | existing `1e-8` gates |
| Preserved minimizer iterations / line search | 1,200 / 40 |
| Preserved failing-case topology transitions | 128, unchanged |
| Prospective matrix's static-initializer topology limit | 64 as recorded in its contract; not interchangeable with the preserved-case 128 |
| Exactly representable flat assembly | existing scaled-roundoff factor 256; solved fields use declared solver tolerances |
| Finest pressure-jump, base-radius, apex-height, volume relative errors | existing one-percent limits and original comparison operators |
| Finest contact-angle absolute error | existing one-degree limit |
| Finest parasitic capillary number | existing `1e-6` limit, same norm/material/time normalization |
| Finest kinetic-energy proxy | existing `1e-12` limit |
| General energy-variation relative comparison | existing `5e-7` maximum, retaining tighter local regressions |
| Spatial refinement / uncertainty | `R/h = 8,16,32`, conditional 64, GCI factor 1.25, justified quantity-specific rates/reference |

The distinction between the captured-case topology cap and the prospective matrix cap is intentional. Do not globally replace either number. Likewise, a documented external timeout adjustment is not a change to scientific iteration, topology, volume, or equilibrium gates.

## 8. Delegation packets and review discipline

Bounded delegation is explicitly authorized, up to **three workers total**, including reused workers and subject to actual available slots. Workers must not spawn helpers. Do not fill a slot without independent useful work.

Suggested initial allocation:

- Coordinator: reconciliation, numerical decision integration, Application/constraint ownership, shared audit/checkpoint, compute, commits, pushes, and qualification release.
- Scientific worker: the precise Milestone 1 conditioning/representation question on frozen inputs, then a named next scientific gap if needed.
- Implementation worker: independent missing Tetra/contact consumers while the mass premise is resolved; later the sole owner of the accepted mass repair or another disjoint complete slice.
- Verification/review worker: exact patch/evidence review or the independent remaining metric-contract work, not automatic re-execution of accepted tests.

Do not assign the same curvature or Application compilation unit to multiple workers. If ownership must change, finish or explicitly transfer the existing assignment and guard first. The coordinator may execute an implementation lane directly while workers handle genuinely independent duties.

Every assignment must contain all of the README packet fields:

1. task ID, one concrete deliverable/question, non-goals, and the blocker removed;
2. accepted specification sections and evidence that must not be rediscovered;
3. exact worktree/commit, dirty patch and participating-input identities, including untracked headers;
4. writable paths, read-only paths, protected paths, overlapping owners;
5. satisfied dependencies, unresolved premises, and explicit acceptance criteria;
6. exact inspection/build/test commands, working directory, expected runtime outcome, timeout;
7. source/HEAD/index/recipe guard owner, cache reservation, and allowed configuration transition;
8. actual allocation/partition/nodes/CPU/memory/walltime and live process route, or explicitly no numerical execution;
9. unique report/artifact path outside guarded inputs and the coordinator's checkpoint;
10. allowed actions, including whether any execution or submission is authorized.

Review a fixed patch and its evidence once. Return concrete findings to the same owner; review only fixes and newly affected behavior afterward. A fresh reviewer is not a reason to rerun unchanged tests. Do not create an additional layer of review merely because the goal is difficult.

## 9. Reference material and numerical design comparison

Use references to answer a named unresolved question, not as another open-ended prerequisite.

The local supplementary book is Sven Gross and Arnold Reusken, *Numerical Methods for Two-phase Incompressible Flows* (2011):

```text
/home/users/zsexton/wp4-references
```

This is a readable PDF file **without an extension**, not a directory: 487 PDF pages, 10,719,903 bytes, SHA-256 `4f7f1f18dadb63fedec4814679a47a8f033e73e00083341a6b1e560bd23627c2`. Keep it outside Git and do not create extracted copies in source trees.

Read selected pages to stdout with Sherlock's available reader:

```bash
/share/software/user/open/poppler/0.47.0/bin/pdftotext -f 241 -l 243 /home/users/zsexton/wp4-references -
```

Useful consulted material: section 6.3, PDF pages 206–209; section 7.6 and Remark 7.6.1, PDF pages 241–243. PDF versus printed-page offsets are not uniform. Remark 7.6.1 explains why the smooth-surface curvature/weak-force equivalence is not automatically an identity on a nonsmooth faceted surface. It does not by itself prove that the current KAG method is wrong or prescribe a replacement.

Primary comparison sources already consulted:

- [Basilisk interfacial-force implementation](https://basilisk.fr/src/iforce.h): compatible pressure/interface gradient treatment is part of its balance construction.
- [Basilisk equilibrium-droplet test](https://basilisk.fr/src/test/spurious.c): explicit physical benchmark and convergence observables.
- [Frachon and Zahedi, CutFEM for incompressible two-phase flow](https://arxiv.org/abs/1808.02662): stabilized curvature and cut-position conditioning belong to the formulation.
- [Hansbo, Larson, and Zahedi, stabilized mean curvature](https://arxiv.org/abs/1407.3043): a particular stabilized surface discretization with a stated convergence result, not a theorem about our current operator.
- [Barrett, Garcke, and Nürnberg, eliminating spurious velocities](https://arxiv.org/abs/1306.2192): matched interface/bulk spaces and variational structure; its two-phase enrichment is not automatic authorization or necessity for our one-phase model.
- [COMSOL free-surface methods](https://www.comsol.com/blogs/two-methods-for-modeling-free-surfaces-in-comsol-multiphysics): mesh-resolved diffuse-interface parameters represent a different numerical contract from sharp cuts.

Established solvers show that the physical problem is tractable. Borrow their matched discretization reasoning, not an isolated force term or an assumption that their guarantees transfer unchanged. No new external solver installation, full benchmark campaign, or change of physical method is authorized merely by these references.

## 10. Continuation evidence identities

These hashes identify the reassessed baseline. Compare them before relying on an artifact. If a coordinator-owned record legitimately changes, inspect the newer record instead of reverting it to this hash. Old frozen numerical inputs/results must remain immutable.

| Artifact | SHA-256 |
| --- | --- |
| `D/delegation-checkpoint.md` | `ec7f14d2a7cdac62b54d1fae27fd8bc8c5ec3cc3ca3fde835a1d0bd6069bae9c` |
| `C/jobs.md` | `3e670fcbcf84f919e4ff53e5248ef8623a134b825f4616e8129f5a5fb86a564b` |
| `W/Documentation/free_surface_boundary_unfitted_audit_20260720.md` | `17bf524750459eea98a5f8021fca4e7f0e3fc8bdcef51e327b984c3aca572104` |
| `W/Documentation/plan_wp4_balanced_force_completion_20260903.md` | `d3efaa485b424ffcc901dac9b69090dc0b451b977e04bbdbf26d0af16b6ae2ad` |
| `W/Documentation/wp4_qualification_gate_contract_20260906.md` | `a28287ee3128e5f59d2867d2ebe5dc9a12b2958839209a0cd8adbde4ace515fa` |
| W curvature source | `d4d0c8a41b1f29219f4c9ddadf21703106f04796d15953ade55552e0ecca9bed` |
| W curvature header | `d1f131194a277820cfcef82f102c175a95ca71acdef1aa02fe1221fd05a4ca22` |
| W Triangle geometry helper | `2d175899a315752af01ff2e1973776c381db35812f80f6115881954af7a79f67` |
| W Tetra geometry helper | `7c580a455577818963f5853a3658d23d98aa21f9a787a73f3a97dbb23a163f15` |
| W Triangle basis helper | `d1e4716f83ba23c3709e4bfbd5ce54f3902bde67929dac34a9fdf8bec3995c6f` |
| W Tetra basis helper | `e7c53e532a89538696146909fe39f1d71e32f45d7036683392792e3638fbf8fe` |
| `R/candidate-3/LevelSetCurvatureProjection.cpp` | `e1e4acc34a598b3999986de95b3a062a0466c8c00073f599a521099e788f08c5` |
| `E/barycentric-c9-acceptance.json` | `87818940bae7d32275f4e8324953020afd6b2c46065b31ee1efdacf00f98c84f` |
| `E/minimizer-2-review/terminal-capture.json` | `b354bcc870eb5ba4b3c49d37f99cfc72d9914fb68b1b3548ce990af4425bdfb4` |
| `Q/results/independent-terminal-review.json` | `af921cec13724cce613a63d90d98551845eb013f8c0dceef8d347b97cf27db98` |
| `O/results/root-acceptance.json` | `102c13bffe3bd80b9da74de578f52a0f75747655f7e8c174e74cc8489c6e531a` |
| `O/results/published-residuals.json` | `c3411a7243efe6d615a551d2b8cb3f06a9c3150aa078cb7567499f9d9190fdd7` |
| `O/results/primary-observation.json` | `8d30eac757b67542f4c1324abc9769161051c085e815ea601d79c774395483b9` |
| `O/results/tests.xml` | `20b45996eaaae28f41431dce164adae222f8189b9fc272cd5f6a34617a4fc980` |
| `P/tetra-geometry-green-1/results/root-acceptance.json` | `7f07dcd6b84194391b5b7ab08de78452e3d3c15bad86902996d299ef1a943a8a` |
| `C/application-wall-green-run-1/results/root-acceptance.json` | `755006104b483b71c5408ff5a83fcde07538e09da9c97fb7255c286312feced4` |

Use the referenced source/recipe manifests to obtain full participating-input closures rather than copying this small table as if it were a complete run guard. Never promote a hash prefix into a full provenance claim.

## 11. Final delivery checklist

- [ ] The conditioning/representation question led to a justified implemented result or an explicit necessary architectural decision, not another unbounded investigation.
- [ ] All four original captured mass states publish valid Real vectors under unchanged route gates; actual complete residual evidence is retained.
- [ ] The matching complete production evaluator passes its source, geometry, derivative, constraint, installed-work, and rollback checks.
- [ ] The preserved minimizer passes from its original start, with unchanged scientific tolerances/limits and a valid publication certificate.
- [ ] Required Tetra, wall/contact-line, prescribed-angle, stage/cadence, and serial/MPI integration is complete for the full declared WP4 scope.
- [ ] Qualification dispatch, metric state binding, norm/rate/reference definitions, runner tests, and dry expansion are complete and reviewed.
- [ ] Required integrated FE/Physics/Application verification is accepted on identified inputs, with no hidden test omissions or protected-file violation.
- [ ] A clean committed/pushed source and final matrix/input identities were frozen before fresh qualification builds and execution.
- [ ] The complete qualification evidence, including failure/inconclusive dispositions where present, was independently reconstructed and archived.
- [ ] Only fully justified FSR-03, FSR-04, WP4, and Q2 items were updated; unrelated audit status was preserved.
- [ ] Every commit has the required author/committer identity, passed the content scan, and was pushed or has a specific recorded external blocker.
- [ ] The final handoff includes exact commits, archive paths, physical outcomes, remaining scope if any, and any live processes/guards/resources.

Start by reconciling the newest recorded results and current ownership, then perform the bounded conditioning decision and useful independent integration. The next substantive report should describe a measured decision and an implemented or executing consequence—not merely restate this goal.

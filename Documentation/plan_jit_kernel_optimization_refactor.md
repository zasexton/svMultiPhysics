# FE JIT Kernel Optimization Refactor Plan

**Date**: 2026-03-28
**Audience**: FE/JIT/assembler developers working on the new OOP solver
**Primary Goal**: Deliver a kernel and assembly optimization architecture that is:

- fast in production,
- exact and robust for nonlinear solves,
- easy for future developers to understand,
- and not dependent on a brittle nest of hidden backend-specific optimization paths.

**Scope**: New OOP solver only.
**Non-goal**: This plan does not modify or preserve the legacy solver architecture beyond maintaining output parity where that is already a validation reference.

## Status Update

- Architectural implementation work is complete through the mixed-kernel,
  assembler-dispatch, backend-capability, and stale-API cleanup phases.
- Qualification reruns are archived in
  `Documentation/jit_kernel_refactor_qualification_20260328.md` and
  `Documentation/qualification_logs/20260328_monolithic_exact_default/`.
- `MonolithicCoupling.MixedJacobianBlockFDVerification` now passes; there is no
  remaining mixed-Jacobian FD failure in the FE test matrix.
- The qualified production mixed-cell path is `MonolithicCellKernel` with exact
  shared-geometry block execution by default. Compiled monolithic dispatch
  remains opt-in only because the real fluid `pipe_simple` case still shows a
  compiled residual mismatch at `cell=0`, `block=0`.
- Remaining unchecked items are runtime-target recovery, the partitioned
  `RCRCR` comparison items, and the `JITKernelWrapper` specialization audit.

---

## 1. Problem Statement

The current OOP FE/JIT optimization stack can achieve strong performance, but it is difficult to reason about because multiple optimization mechanisms overlap:

- ordinary per-kernel JIT through `JITKernelWrapper`,
- shape- and marker-specialized variants inside `JITKernelWrapper`,
- mixed residual lowering through `coupled_residual_from_jacobian_block`,
- block colocation through `CoupledBlockKernel` plus `compileColocated()`,
- assembler-side fused insertion and resolved-slot insertion,
- backend-specific fast paths for FSILS and related matrix/vector views.

This structure makes it too hard to answer simple questions such as:

- What is the exact semantic kernel being assembled?
- Which path is responsible for correctness?
- Which path is an optional acceleration only?
- Which optimizations are backend-independent vs backend-shaped?
- Which code is still live, and which is only “future infrastructure” or stale compatibility scaffolding?

The recent monolithic `RCRCR` issue on `pipe_RCR_3d` is a concrete example: the correctness bug was triggered by an optimization path (`CoupledBlockKernel` on the `coupled_residual_from_jacobian_block` path) whose semantics were not obvious from the public API or the installer code alone.

---

## 2. Guiding Design Principles

### 2.1 Semantic Clarity First

- There must be one obvious semantic path for each kernel category.
- Performance optimizations must not redefine kernel meaning.
- Developers should be able to understand the active assembly semantics by reading one installer path and one assembler dispatch path.

### 2.2 Optimization As A Layer, Not A Shadow Runtime

- Baseline exact assembly semantics must remain stable whether optimization is on or off.
- Optimizations may accelerate exact behavior, but they must not create alternate mathematical paths.
- Any optimization that changes which residuals or Jacobian blocks are assembled is too invasive unless it is represented as a first-class semantic kernel type.

### 2.3 Backend Capability Boundaries

- Generic FE assembly should produce exact local matrix/vector objects.
- Backend-specific insertion or low-rank acceleration should sit behind explicit capability interfaces.
- The FE/JIT layer should not silently reshape math to fit one backend’s preferred fast path.

### 2.4 Small Number Of Kernel Concepts

- Developers should only need to understand a minimal set of kernel types:
  - single-form kernel,
  - mixed per-block kernel set,
  - true monolithic coupled-cell kernel,
  - optional functional kernel.
- Any additional concept must justify its existence with a clear semantic distinction.

### 2.5 Remove Stale Paths

- If an optimization API is not used, either promote it to the production path or remove it.
- If a wrapper name overpromises semantics, rename or replace it.
- Stale fallback paths, dead compile entry points, and duplicate priming flows must not remain in the long-term architecture.

### 2.6 Verification-Driven Refactor

- Every phase must have explicit correctness and performance gates.
- Every optimization layer must have “optimization on/off gives the same residual/Jacobian” parity tests.
- Case-level validation must include the OOP fluid problems that matter to users.

---

## 3. Current-State Assessment

### 3.1 Score Against The Desired Principles

- Raw speed potential: `8/10`
- Assembly-performance sophistication: `8/10`
- Robustness / non-brittleness: `3/10`
- Backend-independence of optimization logic: `3/10`
- Ease for future developers to understand: `3/10`
- Overall against the target design: `4/10`

### 3.2 Main Architectural Weaknesses

- `JITCompiler` exposes `compileFused()`, `compileMonolithic()`, and `compileColocated()`, but only `compileColocated()` is currently in active use.
- `CoupledBlockKernel` is not a true coupled semantic kernel; it is a wrapper whose real behavior depends on special handling in `StandardAssembler`.
- `coupled_residual_from_jacobian_block` is a hidden lowering trick rather than a first-class kernel concept.
- `StandardAssembler::assembleCellsFused()` contains backend-aware fused insertion logic that is difficult to separate from correctness.
- `SystemSetup` and `SystemAssembly` contain shadow registration and hidden dispatch behavior for coupled kernels.

### 3.3 Current OOP Solver Requirement

After this refactor, the new OOP solver must run the following fluid cases with excellent nonlinear convergence and fast performance:

- `tests/cases/fluid/Channel2D/solver_perf_oop.xml`
- `tests/cases/fluid/Channel2D_Simple/solver_perf_oop.xml`
- `tests/cases/fluid/vortex_shedding/solver_perf_oop.xml`
- `tests/cases/fluid/pipe_RCR_3d/solver_perf_oop.xml`
- `tests/cases/fluid/pipe_RCR_3d/solver_perf_oop_rcrcr.xml`
- `tests/cases/fluid/pipe_simple/solver_perf_oop.xml`
- `tests/cases/fluid/iliac_artery/solver_perf_oop.xml`

The final architecture must support these cases without relying on fragile debug-only switches or solver-side special casing for specific outlet models.

---

## 4. End-State Architecture

### 4.1 Canonical Semantic Kernel Types

The target architecture should have the following semantic kernel categories:

1. `SingleFormKernel`
   - one form, one semantic kernel
   - optionally JIT compiled and optionally specialized

2. `MixedBlockKernelSet`
   - independent exact per-block kernels
   - used when no true coupled-cell fusion is requested or available

3. `MonolithicCellKernel`
   - one true semantic kernel representing a mixed cell-domain coupled system
   - carries explicit per-block tangent/residual intent
   - uses shared geometry and shared quadrature-point intermediates
   - does not rely on “one block is Both, others are MatrixOnly” as a hidden semantic trick

4. `FunctionalKernel`
   - for FE quantities / reductions / explicit functionals

### 4.2 Performance Layers

Performance layers must remain subordinate to semantic kernel types:

- LLVM JIT compilation
- shape specialization
- marker specialization
- code colocation / text layout optimization
- resolved insertion tables
- backend-accelerated insertion
- optional low-rank update acceleration

These layers must be optional accelerations of exact semantics, not alternate ways to define semantics.

### 4.3 Explicit Flow Developers Should Be Able To Read

Target mental model:

1. `installFormulation()` lowers user expressions to a `MixedKernelPlan`.
2. The plan explicitly selects either:
   - per-block exact kernels, or
   - a true `MonolithicCellKernel`.
3. `SystemSetup` primes kernels but does not change semantics.
4. `SystemAssembly` requests matrix/vector assembly from the chosen semantic kernel type.
5. Backend-specific views accelerate insertion only after exact local contributions are formed.

---

## 5. Success Criteria

### 5.1 Correctness Success Criteria

- No optimization path may change assembled residual/Jacobian semantics.
- Mixed cell Jacobians must match finite-difference checks in unit coverage.
- Monolithic auxiliary coupling must use the same PDE time-integration stencil and stage semantics as the PDE it is coupled to.
- No accepted time step in the OOP solver may report `converged=0` for the final desired qualification runs.
- Nonlinear convergence must satisfy configured tolerances directly; stagnation or fallback acceptance must not be part of the qualification story.

### 5.2 Performance Success Criteria

- The optimized path must be measurably faster than the exact non-fused fallback on the qualified OOP cases.
- The final production path must avoid redundant assembly passes compared with the current best OOP behavior.
- The final production path must not rely on disabling major optimizations globally to obtain correct nonlinear convergence.

### 5.3 Maintainability Success Criteria

- Future developers should be able to locate the active mixed-cell production path by reading:
  - `FormsInstaller`,
  - the semantic kernel type,
  - `SystemAssembly`,
  - and one assembler dispatch path.
- No public JIT entry point should remain unused without either:
  - a concrete roadmap item making it production, or
  - explicit removal.

---

## 6. Verification Matrix And Acceptance Gates

## 6.1 Core Verification Dimensions

- Residual parity: optimized vs non-optimized path
- Jacobian parity: optimized vs finite differences
- Nonlinear convergence: Newton iteration counts, final residual norms
- Linearization parity: monolithic vs per-block / dense reference
- Performance: total runtime, assembly time, linear solve time, JIT cache behavior
- Stability: repeated runs, no debug env flags required

## 6.2 Required Case-Level Qualification Gates

### Channel2D

- Target file: `tests/cases/fluid/Channel2D/solver_perf_oop.xml`
- Gate:
  - converges without nonlinear fallback acceptance
  - final residual satisfies configured tolerances
  - steady/transient startup Newton counts remain low and stable
  - no performance regression larger than agreed tolerance versus the pre-refactor OOP baseline

### Channel2D_Simple

- Target file: `tests/cases/fluid/Channel2D_Simple/solver_perf_oop.xml`
- Gate:
  - converges without nonlinear fallback acceptance
  - low Newton counts representative of a simple mixed case
  - serves as a fast correctness/performance smoke case for CI

### vortex_shedding

- Target file: `tests/cases/fluid/vortex_shedding/solver_perf_oop.xml`
- Gate:
  - stable transient convergence through the full configured horizon
  - no hidden JIT/assembler path changes between early and late steps
  - acceptable runtime relative to current OOP baseline

### pipe_RCR_3d

- Target file: `tests/cases/fluid/pipe_RCR_3d/solver_perf_oop.xml`
- Gate:
  - monolithic `RCR` continues to converge excellently
  - monolithic bordered auxiliary coupling remains exact under the refactor
  - no regression in nonlinear convergence or runtime

### pipe_RCR_3d with RCRCR

- Target file: `tests/cases/fluid/pipe_RCR_3d/solver_perf_oop_rcrcr.xml`
- Gate:
  - monolithic `RCRCR` converges excellently without disabling major optimizations globally
  - monolithic is at least as good as partitioned in Newton iteration count
  - monolithic runtime is competitive with or better than partitioned for the qualified path

### pipe_simple

- Target file: `tests/cases/fluid/pipe_simple/solver_perf_oop.xml`
- Gate:
  - remains a cheap throughput benchmark for assembly/runtime changes
  - no regression in simple outlet-free flow behavior

### iliac_artery

- Target file: `tests/cases/fluid/iliac_artery/solver_perf_oop.xml`
- Gate:
  - converges robustly on a larger realistic geometry
  - no path-specific solver instability in the more expensive production case
  - runtime remains competitive with the current OOP baseline

## 6.3 Global Qualification Thresholds

The final refactor should satisfy all of the following:

- No case above is accepted with `converged=0`.
- Final nonlinear residuals satisfy the configured tolerances rather than merely becoming “small enough.”
- For `pipe_RCR_3d` `RCRCR`, monolithic Newton counts are not worse than partitioned Newton counts on the qualified configuration.
- Assembly timing regressions greater than `10%` on simple cases or `15%` on larger cases require an explicit written justification.
- Any optimization that only improves one backend while obscuring semantics must be demoted behind a capability boundary or removed.

---

## 7. Refactor Phases

## Phase 0. Baseline, Observability, And Freeze Points

**Objective**: Establish exact before/after evidence and create a stable reference envelope for the refactor.

### Deliverables

- A baseline report for the qualified OOP cases.
- A kernel-path inventory document explaining which installer path, kernel type, and assembler path each case uses.
- Consistent tracing for:
  - selected installer/lowering path,
  - semantic kernel type,
  - JIT compilation mode,
  - specialization hits/misses,
  - fused vs unfused assembly path,
  - backend insertion capabilities used.

### Code Cleanup In This Phase

- Remove one-off debug env switches that were only used to diagnose the monolithic `RCRCR` issue if they still exist.
- Consolidate existing JIT trace knobs into one documented mechanism.
- Delete stale comments that claim a path is “future” if it is now production, or vice versa.

### Verification Points

- All qualified cases run and produce archived baseline logs.
- Unit tests for monolithic `RCR` and `RCRCR` mixed Jacobian FD parity pass.
- Baseline timing tables are recorded for:
  - total runtime,
  - assembly time,
  - linear solve time,
  - total Newton iterations.

### Phase 0 Checklist

- [x] Capture OOP baseline logs for all qualified cases.
- [x] Capture Newton iteration histories for all qualified cases.
- [x] Capture assembly timing summaries for all qualified cases.
- [x] Add a documented kernel-path trace mode.
- [x] Add a documented JIT specialization trace mode.
- [x] Archive the baseline in `Documentation/` for future comparison.

---

## Phase 1. Clarify Kernel Semantics At The API Boundary

**Objective**: Make the semantic kernel contract explicit before changing optimizations.

### Planned Changes

- Introduce a first-class `MixedKernelPlan` concept in `FormsInstaller`.
- Represent mixed-form lowering explicitly instead of implicitly through `coupled_residual_from_jacobian_block`.
- Separate:
  - semantic lowering choice,
  - JIT compilation choice,
  - code-layout optimization choice.
- Add a developer-facing architecture doc describing the pipeline.

### Primary Files

- `Code/Source/solver/FE/Systems/FormsInstaller.h`
- `Code/Source/solver/FE/Systems/FormsInstaller.cpp`
- `Code/Source/solver/FE/Forms/JIT/JITCompiler.h`
- `Code/Source/solver/FE/Docs/` new architecture note

### Code Cleanup In This Phase

- Mark `coupled_residual_from_jacobian_block` as deprecated internal plumbing.
- Stop exposing it as the conceptual center of mixed residual optimization.
- Rename misleading comments that refer to “optimal coupled assembly” when they are really describing one current lowering trick.

### Verification Points

- `installFormulation()` produces the same exact residual/Jacobian as before on all mixed unit tests.
- No case-level behavior changes yet.
- Developers can inspect logs and see which semantic kernel type was chosen.

### Phase 1 Checklist

- [x] Add `MixedKernelPlan` or equivalent lowering object.
- [x] Replace bool-soup comments with explicit semantic descriptions.
- [x] Add path-selection tracing tied to semantic kernel types.
- [x] Add a short architecture document for kernel lowering and JIT compilation.
- [x] Preserve all existing mixed Jacobian FD tests.

---

## Phase 2. Make True Monolithic Mixed Cell JIT The Production Path

**Objective**: Replace the fragile monolithic residual-from-Jacobian fusion trick with a true coupled cell kernel.

### Planned Changes

- Build a true `MonolithicCellKernel` using `JITCompiler::compileMonolithic(...)`.
- Feed it explicit per-block tangent/residual IR from the `MixedKernelPlan`.
- Use this as the production path for monolithic mixed cell forms.
- Keep boundary and face terms separate initially.

### Primary Files

- `Code/Source/solver/FE/Forms/JIT/JITCompiler.h`
- `Code/Source/solver/FE/Forms/JIT/JITCompiler.cpp`
- `Code/Source/solver/FE/Forms/JIT/LLVMGen.cpp`
- `Code/Source/solver/FE/Assembly/JIT/KernelArgs.h`
- new `MonolithicCellKernel` implementation
- `Code/Source/solver/FE/Systems/FormsInstaller.cpp`

### Code Cleanup In This Phase

- Remove production dependence on `coupled_residual_from_jacobian_block`.
- Remove the monolithic-sensitive use of `CoupledBlockKernel` for semantic mixed residual assembly.
- If `compileFused()` remains unused after this phase, schedule it for removal or demote it to an internal-only helper with clear documentation.

### Verification Points

- Unit parity:
  - monolithic mixed Jacobian FD parity
  - generalized-alpha monolithic mixed Jacobian FD parity
  - monolithic auxiliary residual/stencil parity
- Case parity:
  - `pipe_RCR_3d` `RCR` remains excellent
  - `pipe_RCR_3d` `RCRCR` monolithic remains excellent
- Optimization off/on parity:
  - true monolithic JIT vs interpreter/per-block fallback yields the same assembled matrix/vector within tolerance

### Phase 2 Checklist

- [x] Introduce `MonolithicCellKernel`.
- [x] Route monolithic mixed cell forms to `compileMonolithic()`.
- [x] Add tests for exact matrix/vector parity versus per-block assembly.
- [x] Add tests for mixed residual blocks with `Both` semantics.
- [x] Re-run `pipe_RCR_3d` `RCR` and `RCRCR`.
- [ ] Confirm monolithic `RCRCR` Newton counts are no worse than partitioned.

---

## Phase 3. Simplify Assembler Dispatch And Remove Hidden Semantics

**Objective**: Remove assembler magic based on wrapper detection.

### Planned Changes

- Replace “exactly one `CoupledBlockKernel` means special path” logic with explicit dispatch on semantic kernel type.
- Keep `StandardAssembler` responsible for execution and insertion only.
- Ensure `SystemAssembly` no longer depends on shadow registration or kernel suppression tricks.

### Primary Files

- `Code/Source/solver/FE/Assembly/StandardAssembler.cpp`
- `Code/Source/solver/FE/Assembly/StandardAssembler.h`
- `Code/Source/solver/FE/Systems/SystemAssembly.cpp`
- `Code/Source/solver/FE/Systems/SystemSetup.cpp`

### Code Cleanup In This Phase

- Remove `coupled_covered_kernels` shadow suppression if the new semantic kernel path makes it unnecessary.
- Remove duplicate setup-time priming logic for “coupled wrapper plus fallback kernels.”
- Remove comments that describe wrapper detection as a design feature rather than an implementation shortcut.

### Verification Points

- Assembly output parity before/after dispatcher cleanup.
- No case-level convergence change from this phase alone.
- Reduced codepath count visible in tracing.

### Phase 3 Checklist

- [x] Add explicit semantic-kernel dispatch in the assembler.
- [x] Remove or simplify `CoupledBlockKernel` detection branches.
- [x] Remove kernel shadow-registration behavior if obsolete.
- [x] Keep matrix/vector parity tests green.
- [x] Re-run simple OOP cases: `Channel2D_Simple`, `pipe_simple`.

---

## Phase 4. Push Backend Optimizations Behind Capabilities

**Objective**: Keep FE/JIT semantics backend-independent while preserving performance.

### Planned Changes

- Introduce explicit backend capability queries for:
  - resolved sparse insertion,
  - contiguous combined insertion,
  - optional low-rank update acceleration.
- Make `StandardAssembler` ask the backend view what acceleration it supports instead of encoding FSILS-preferred behavior directly in generic logic.
- Ensure exact local matrix/vector objects are formed before backend acceleration is applied.

### Primary Files

- `Code/Source/solver/FE/Assembly/GlobalSystemView.h`
- `Code/Source/solver/FE/Assembly/StandardAssembler.cpp`
- backend view implementations under `Code/Source/solver/FE/Backends/`

### Code Cleanup In This Phase

- Remove backend-specific explanatory comments from generic assembler code where the capability abstraction replaces them.
- Eliminate assumptions that “fused insertion” is always worth the added complexity.
- Demote backend-native low-rank shortcuts if they are not exact accelerations of already-defined FE quantities.

### Verification Points

- Same exact matrix/vector from capability-enabled and capability-disabled assembly.
- Comparable or improved performance on FSILS-backed fluid cases.
- No hidden backend dependence in monolithic nonlinear convergence behavior.

### Phase 4 Checklist

- [x] Add backend capability reporting for insertion accelerators.
- [x] Route fused insertion through capabilities instead of hard-coded assumptions.
- [x] Add parity tests with capability on/off.
- [x] Measure performance on `Channel2D`, `pipe_simple`, `iliac_artery`.
- [x] Confirm monolithic `RCRCR` behavior is unchanged by capability refactoring.

---

## Phase 5. Specialization, Colocation, And Optional Fast Paths

**Objective**: Keep useful fast paths, but only as optional accelerations of the now-clean semantic core.

### Planned Changes

- Retain `JITKernelWrapper` specialization only as a local acceleration for one semantic kernel.
- Keep `compileColocated()` only if it demonstrates value after the semantic cleanup.
- If `CoupledBlockKernel` survives at all, narrow it to a pure colocation/text-layout optimization and rename it accordingly.
- Make the priming pipeline explicit:
  - semantic kernel compile,
  - optional specialization prime,
  - optional code colocation.

### Primary Files

- `Code/Source/solver/FE/Forms/JIT/JITKernelWrapper.h`
- `Code/Source/solver/FE/Forms/JIT/JITKernelWrapper.cpp`
- `Code/Source/solver/FE/Forms/MixedBlockKernelSet.h`
- `Code/Source/solver/FE/Forms/MixedBlockKernelSet.cpp`
- `Code/Source/solver/FE/Systems/SystemSetup.cpp`

### Code Cleanup In This Phase

- Remove any unused specialization hooks.
- Remove public names that imply semantics the object does not own.
- Remove dead or stale JIT entry points if still unused after the architecture settles.

### Verification Points

- Performance A/B:
  - specialization on/off
  - colocation on/off
- Exact residual/Jacobian parity in all A/B comparisons.
- Improved developer comprehensibility:
  - one documented priming sequence
  - no hidden semantic dependencies on priming order

### Phase 5 Checklist

- [ ] Audit `JITKernelWrapper` specialization paths and delete unused ones.
- [x] Decide whether `compileColocated()` remains.
- [x] Rename or remove `CoupledBlockKernel` if it is only a colocation helper.
- [x] Document the final priming flow.
- [x] Benchmark on `Channel2D`, `vortex_shedding`, `iliac_artery`.

---

## Phase 6. Remove Deprecated And Stale Code

**Objective**: Leave the codebase with one understandable optimization model.

### Planned Changes

- Delete or internalize stale public entry points and options.
- Remove dead installer flags and shadow flows.
- Remove comments that describe superseded behavior.
- Update developer docs to reference only the surviving architecture.

### Candidate Cleanup Targets

- `FormInstallOptions::coupled_residual_from_jacobian_block`
- any stale wrapper-detection comments in `StandardAssembler`
- stale setup-time shadow kernel suppression in `SystemSetup`
- unused `compileFused()` if it is still not production
- unused `compileMonolithic()` scaffolding if the production path still does not consume it
  - this item should not happen if the plan succeeds
- misleading `CoupledBlockKernel` naming and comments if the type survives only as a colocation helper

### Verification Points

- No CI or case validation depends on removed code.
- No public comment or document still points developers to a dead path.
- A new developer can identify the production mixed-kernel path from the docs and source without reading stale alternatives.

### Phase 6 Checklist

- [x] Remove deprecated installer flags that no longer own production behavior.
- [x] Remove stale JIT compiler entry points or demote them to internal APIs.
- [x] Remove wrapper-detection comments and dead branches.
- [x] Update FE docs to describe only the surviving architecture.
- [x] Re-run full qualified OOP case matrix.

---

## Phase 7. Final Qualification Of OOP Fluid Cases

**Objective**: Prove that the cleaned-up architecture is both fast and solver-robust on representative production cases.

### Final Qualification Requirements

- `Channel2D`
  - [x] converges cleanly
  - [ ] runtime within target

- `Channel2D_Simple`
  - [x] converges cleanly
  - [x] remains a cheap CI smoke/perf case

- `vortex_shedding`
  - [x] stable transient convergence
  - [x] runtime within target

- `pipe_RCR_3d`
  - [x] monolithic `RCR` remains excellent
  - [x] no regression in runtime

- `pipe_RCR_3d` `RCRCR`
  - [x] monolithic `RCRCR` converges excellently
  - [ ] monolithic Newton counts no worse than partitioned
  - [ ] monolithic runtime competitive with or better than partitioned

- `pipe_simple`
  - [ ] remains stable and fast

- `iliac_artery`
  - [x] converges cleanly
  - [ ] runtime within target

### Final Performance Review

- [x] compare final runtime to the Phase 0 baseline
- [x] compare final Newton counts to the Phase 0 baseline
- [ ] compare final assembly timing to the Phase 0 baseline
- [ ] document any intentional regressions and why they are acceptable

### Final Developer-Experience Review

- [x] one architecture note explains the production kernel pipeline
- [x] one trace mode identifies the active semantic kernel path
- [x] no stale optimization entry points remain unexplained

---

## 8. Recommended Order Of Execution

1. Phase 0 baseline and instrumentation
2. Phase 1 explicit semantic kernel planning
3. Phase 2 true monolithic coupled-cell kernel
4. Phase 3 assembler dispatch cleanup
5. Phase 4 backend capability boundary
6. Phase 5 optional fast-path cleanup and consolidation
7. Phase 6 stale code removal
8. Phase 7 final qualification

This order prioritizes correctness and semantic clarity first, then performance recovery and optimization cleanup second.

---

## 9. Immediate Next Step

The best next implementation step is:

- resolve the remaining real-case compiled monolithic residual mismatch
  (`pipe_simple`, `cell=0`, `block=0`) so
  `SVMP_FE_ENABLE_MONOLITHIC_COMPILED_DISPATCH=1` can become production-safe,
- recover the open runtime targets on `Channel2D`, `Channel2D_Simple`,
  `pipe_RCR_3d`, `pipe_simple`, and `iliac_artery`,
- and complete the remaining partitioned `RCRCR` comparison and
  `JITKernelWrapper` specialization audit items.

That work is now the highest-leverage path to finish the remaining unchecked
qualification gates.

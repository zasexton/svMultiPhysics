# FE Mixed Expressions -> Block IR — Design Plan

## Overview

This document defines the cross-module plan for supporting:

- **first-class mixed expressions in `FE/Forms`** as the canonical user-facing
  representation for multiphysics weak forms, while
- **block decomposition in `FE/Systems` and `FE/Backends`** remains the stable
  assembly, sparsity, solver, and preconditioner IR.

The intent is to let users write one coupled mixed weak form while preserving
the execution model that already fits the FE library's field-oriented
registries, block operators, and backend solver strategies.

This plan is intentionally cross-cutting. It spans:

- `FE/Forms` for syntax, compilation, and mixed IR
- `FE/Systems` for lowering/installing mixed forms into operator terms
- `FE/Assembly` for execution and optional fused traversal
- `FE/Backends` for block storage and solver/preconditioner integration
- `FE/Analysis` for retaining whole-form structure and diagnostics

---

## Decision Summary

The FE library will adopt the following split:

1. **User-facing source representation:** first-class mixed `FormExpr`
   expressions in `FE/Forms`.
2. **Compiler bridge:** `forms::MixedFormIR` as the lowering target from mixed
   expressions to execution-oriented blocks.
3. **Stable execution IR:** per-domain, per-`(test_field, trial_field)` block
   kernels registered into `FE/Systems`.
4. **Stable solver/backend IR:** block sparsity, block matrices/vectors,
   field-split, Schur, and other block-aware backend logic.

This means `FE/Assembly` and `FE/Backends` do **not** need a new monolithic
"mixed operator" execution path. They continue to execute and solve block
operators.

---

## Problem Statement

The library already has both sides of the desired architecture, but they are
not yet framed as one explicit policy:

- `Forms` has a `compileMixed()` path and `MixedFormIR`.
- `Systems` already installs per-block kernels and assembles by field pair.
- `Backends` already expose block-oriented storage and solver options.

However, the design still needs to make the boundary explicit:

- Users should not be forced to manually split every multiphysics form into
  block expressions.
- `Systems` and `Backends` should not lose the block structure they rely on for
  sparsity, preconditioning, and solver configuration.
- Analysis and diagnostics should preserve the original mixed expression, not
  only the lowered blocks.

---

## Goals

1. Make first-class mixed expressions the preferred multiphysics API in
   `FE/Forms`.
2. Keep block decomposition the only stable execution model below `Forms`.
3. Preserve backend and solver assumptions about field/block structure.
4. Preserve zero-block elimination and per-block specialization.
5. Retain source-level mixed-form provenance for analysis, debugging, and
   diagnostics.
6. Leave room for later fused execution across compatible blocks without
   changing the stable IR.

---

## Non-Goals

- Do not introduce a mandatory monolithic mixed operator representation in
  `Assembly` or `Backends`.
- Do not replace `FESystem` field-pair registration with a fundamentally new
  solver-facing API.
- Do not require every backend to expose non-block solver/preconditioner
  semantics.
- Do not block the current manual block-decomposition workflow; that remains a
  supported lower-level path.

---

## API Classification

The following API surfaces coexist after this plan:

| Surface | Status | Use case |
|---------|--------|----------|
| `installFormulation()` | **Canonical public residual entry** | Unified residual entry point (auto-routes single/mixed) |
| `installMixedBilinear()` | **Preferred public** | Mixed bilinear forms (compile + install) |
| `installMixedLinear()` | **Preferred public** | Mixed linear forms (compile + install) |
| `installMixedFormIR()` | **Public, expert / lower-level** | Pre-compiled `MixedFormIR` installation for bilinear forms and advanced/testing flows |
| `compileMixed()` | **Public** | Mixed expression → MixedFormIR compilation |
| `BlockBilinearForm` / `BlockLinearForm` | **Supported, manual** | Expert manual block decomposition |
| `FormCompiler::compileBilinear(BlockBilinearForm)` | **Supported, manual** | Per-block compilation from manual containers |
| `installResidualBlocks()` | **Internal** | Low-level block installation |
| `installCoupledResidualMixed()` | **Internal** | Mixed residual decomposition (called by installFormulation) |

The block containers (`BlockBilinearForm`, `BlockLinearForm`) and their
compiler overloads remain supported as a lower-level manual path for users
who need explicit control over block structure. They are not deprecated.
The first-class mixed expression path is *preferred* because it eliminates
manual block splitting and preserves whole-form provenance, but it is not
*required*.

Residual installation is intentionally centralized around `installFormulation()`.
The library should not grow a second canonical public residual installer
(`installMixedResidual()` or equivalent). Lower-level residual helpers may
continue to exist for implementation, testing, or expert workflows, but the
public residual authoring story remains a single entry point.

---

## Architectural Contract

### Layer 1: User-Facing Mixed Forms

`FE/Forms` is responsible for:

- expressing mixed weak forms with multiple test/trial spaces in one source
  expression,
- validating mixed-space semantics,
- compiling mixed source into `MixedFormIR`,
- preserving source-level provenance and whole-form metadata.

### Layer 2: Mixed IR Bridge

`forms::MixedFormIR` is responsible for:

- representing the mixed form as a **block-sparse** set of active
  `(test_idx, trial_idx)` blocks,
- retaining test/trial field descriptors,
- retaining per-block `FormIR`,
- retaining enough whole-form metadata for diagnostics, analysis, and setup
  heuristics.

### Layer 3: Stable Execution IR

`FE/Systems` is responsible for:

- binding mixed-form blocks to concrete `FieldId`s,
- lowering each active block into existing operator registration calls,
- keeping constraints, sparsity, assembly, and state plumbing block-based.

**Note on linear forms:** `MixedFormIR` uses a 1-column layout with a synthetic
trial column for linear forms (no real trial space). `compileMixed()` populates
a trial descriptor derived from the test field so the IR is self-consistent.
`installMixedLinear()` maps this synthetic column to placeholder trial
`FieldId`s for installation via `installMixedFormIR()`. This is a pragmatic
adapter — the stable contract for linear forms is `installMixedLinear()` as
the public entry point, not direct `installMixedFormIR()`.

**Residual semantics:** `compileMixed()` with `FormKind::Residual` produces
a Jacobian-block decomposition: `activeBlocks()` reflects the `(test, trial)`
coupling structure of the Jacobian, not the full residual. Test-only terms
(e.g., source `f*v`) that have no trial dependency are NOT placed in any block
— they are preserved by the `installFormulation()` → `installCoupledResidualMixed()`
→ `installCoupledResidual()` path, which decomposes by test function first and
correctly includes test-only terms in the per-test residual vector.

**Residual entry policy:** `installFormulation()` is the only canonical public
residual installer. It handles test-only terms, per-trial Jacobian blocks,
and coupled assembly options. Public mixed residual authoring should not bypass
it via `installMixedFormIR()` or `installCoupledResidualMixed()`.

### Layer 4: Stable Solver/Backend IR

`FE/Backends` is responsible for:

- preserving block matrices/vectors as the stable multiphysics solver surface,
- keeping field-split and Schur workflows block-oriented,
- remaining agnostic to whether the user authored the operator as one mixed
  form or manual block expressions.

**Matrix-free operators:** `FESystem` also supports matrix-free operator
registration (`addMatrixFreeKernel`). Auto-registration of matrix-free
operators is currently restricted to single-field, cell-only, linear, steady
operators (`SystemSetup.cpp`).

Multi-field matrix-free auto-registration is a **follow-on track** to this plan,
not a replacement for it. The required architecture is:

- mixed source still lowers to the same registered block operator model,
- setup-time operator-backend code derives a matrix-free operator from that
  block model,
- the matrix-free apply path uses a **block-aware matrix-free application
  interface** that is orthogonal to the assembled block-IR architecture.

In other words, the mixed-source / block-IR split remains the same; the new
work is to make operator backends capable of deriving a multi-field matrix-free
apply from the registered block structure.

### Invariants

The following invariants must hold after implementation:

1. Below `Forms`, every mixed operator is representable as active block kernels
   indexed by `(test_field, trial_field, domain)`.
2. The block layout produced by first-class mixed forms matches the block layout
   produced by equivalent manual block decomposition.
3. Analysis and diagnostics retain both:
   - the original mixed source representation, and
   - the lowered block representation.
4. Optional fused execution may optimize traversal, but it must not redefine
   the stable execution IR.
5. If matrix-free operator backends are auto-registered for mixed systems, they
   are derived from the registered block operator model at setup time; they do
   not become a second lowering target from `Forms`.

---

## Target Data Flow

```text
User mixed FormExpr
    |
    v
Forms front-end
    - TrialFunctions/TestFunctions
    - mixed syntax and validation
    |
    v
FormCompiler::compileMixed()
    |
    v
forms::MixedFormIR
    - field descriptors
    - active blocks
    - per-block FormIR
    - whole-form metadata + provenance
    |
    v
Systems/FormsInstaller
    - bind block indices to FieldId
    - install each block as existing operator terms
    |
    v
FESystem operator registry
    - cells / boundary / interior / interface / global
    - (test_field, trial_field) pairing
    |
    +--> Optional operator-backend derivation
         - block-aware matrix-free operator
         - setup-time auto-registration from block terms
    |
    v
Assembly / Sparsity / Backends
    - block execution
    - block sparsity
    - block matrices/vectors
    - field-split / Schur / other block-aware solvers
```

---

## Current-State Interpretation

The current code already provides much of the skeleton for this plan:

- `FormCompiler::compileMixed()` lowers mixed source to `MixedFormIR`.
- `MixedFormIR` preserves active block sparsity and per-block `FormIR`.
- `FormsInstaller::installMixedFormIR(...)` installs each active block through
  the existing operator-registration pathways.

That means the main work is not inventing a new architecture. The main work is:

- making this split explicit and stable,
- broadening user-facing mixed expression support,
- preserving more whole-form metadata and provenance,
- tightening parity, diagnostics, and performance around the lowering boundary.

---

## Workstreams

## 1. Forms Front-End Workstream

### Objective

Make first-class mixed expressions the preferred and well-defined authoring
model for multiphysics forms.

### Deliverables

- Define the canonical mixed authoring workflow around:
  - `spaces::MixedSpace`
  - `forms::TrialFunctions(W)`
  - `forms::TestFunctions(W)`
  - subspace/component extraction helpers
- Document what constitutes a valid mixed expression:
  - multiple test functions,
  - multiple trial functions,
  - linear/residual forms with or without trial arguments,
  - domain measures across cell, boundary, interior, and interface terms
- Ensure high-level compile entry points choose the mixed compiler path when
  multiple bound spaces are present.
- Keep single-field compilation strict and simple.

### File Touchpoints

- `Code/Source/solver/FE/Forms/FormExpr.h`
- `Code/Source/solver/FE/Forms/FormExpr.cpp`
- `Code/Source/solver/FE/Forms/FormCompiler.h`
- `Code/Source/solver/FE/Forms/FormCompiler.cpp`
- `Code/Source/solver/FE/Forms/VOCABULARY.md`

### Acceptance Criteria

- A user can express a coupled multiphysics form as one mixed `FormExpr`
  without manual block splitting.
- Equivalent single-field expressions still compile through the single-field
  path without behavior changes.

---

## 2. Mixed IR Workstream

### Objective

Promote `MixedFormIR` from a useful lowering container to the explicit mixed
compiler contract.

### Deliverables

- Extend `MixedFormIR` to carry:
  - stable test/trial field descriptors,
  - active block map,
  - per-block `FormIR`,
  - whole-form required-data union,
  - whole-form domain summary,
  - source provenance metadata,
  - optional block-to-source mapping for diagnostics
- Define block classification rules for:
  - cell terms,
  - boundary terms,
  - interior-face terms,
  - interface-face terms,
  - global terms
- Preserve zero-block elimination as a required property.

### File Touchpoints

- `Code/Source/solver/FE/Forms/MixedFormIR.h`
- `Code/Source/solver/FE/Forms/FormCompiler.cpp`

### Acceptance Criteria

- `MixedFormIR` is sufficient for installation, setup-time heuristics, and
  source-aware diagnostics without re-parsing the original expression.

---

## 3. Systems Lowering Workstream

### Objective

Keep `Systems` block-oriented while making mixed-form lowering the normal path
from `Forms`.

### Deliverables

- Add or standardize mixed-form installation entry points:
  - mixed bilinear install → `installMixedBilinear()`
  - mixed residual install → `installFormulation()` (canonical public entry point;
    the internal `installCoupledResidualMixed()` is not part of the public API)
  - mixed linear install → `installMixedLinear()`
- Keep residual public API surface centralized:
  - no separate public `installMixedResidual()` API
  - physics modules and examples use `installFormulation()` for residuals
- Make lowering from `MixedFormIR` to operator terms a documented, stable
  contract (for bilinear and residual forms; linear forms use a synthetic
  trial column — see Layer 3 note).
- Preserve existing `FESystem` registration model:
  - `addCellKernel(...)`
  - `addBoundaryKernel(...)`
  - `addInteriorFaceKernel(...)`
  - `addInterfaceFaceKernel(...)`
  - `addGlobalKernel(...)`
- Record both:
  - source mixed formulation metadata, and
  - lowered per-block formulation metadata.

### File Touchpoints

- `Code/Source/solver/FE/Systems/FormsInstaller.h`
- `Code/Source/solver/FE/Systems/FormsInstaller.cpp`
- `Code/Source/solver/FE/Systems/FESystem.h`
- `Code/Source/solver/FE/Systems/FESystem.cpp`
- `Code/Source/solver/FE/Systems/OperatorRegistry.*`

### Acceptance Criteria

- An equivalent operator assembled from:
  - manual block decomposition, and
  - one mixed source expression
  produces the same registered block structure in `Systems`.
- Public residual authoring uses `installFormulation()` as the sole canonical
  entry point, regardless of whether the source expression is single-field or
  mixed.

---

## 4. Assembly Workstream

### Objective

Keep the stable execution IR block-based while enabling optional multi-block
execution optimizations.

### Deliverables

- Keep `Assembly` ignorant of first-class mixed source syntax.
- Continue to execute per-block kernels by field pair and domain.
- Add an optional fused execution path for compatible **cell** blocks:
  - shared geometry preparation,
  - shared basis preparation,
  - shared field-solution preparation,
  - shared quadrature traversal
- Treat boundary, interior-face, and interface-face fusion as follow-on work.
- Do not change the stable registration or backend interfaces when fused
  execution is enabled.

### File Touchpoints

- `Code/Source/solver/FE/Assembly/Assembler.h`
- `Code/Source/solver/FE/Assembly/StandardAssembler.*`
- `Code/Source/solver/FE/Assembly/AssemblyLoop.*`
- `Code/Source/solver/FE/Assembly/AssemblerFactory.cpp`

### Acceptance Criteria

- Fused execution is an optimization layer only.
- Disabling fusion still yields the same semantics and block structure.

---

## 5. Backend And Solver Workstream

### Objective

Keep block decomposition as the stable solver/backend IR and ensure mixed-form
authoring lowers into the same backend-visible structure as manual block forms.

### Deliverables

- Confirm that mixed-form lowering produces the same:
  - sparsity structure,
  - block offsets,
  - block matrix/vector layout,
  - field-split metadata
  as manual block decomposition
- Define the matrix-free follow-on boundary:
  - multi-field matrix-free auto-registration derives from registered block
    terms in `Systems`
  - it does not change the stable mixed-to-block lowering contract
  - it requires a block-aware matrix-free application interface at the
    operator-backend layer
- Preserve block-aware solver workflows:
  - PETSc field-split
  - PETSc Schur
  - other backend-specific block solver paths
- Prioritize closure of backend gaps that most affect multiphysics usability:
  - non-owned-row distributed assembly limitations,
  - missing block-preconditioner parity,
  - layout assumptions that need explicit mapping layers

### File Touchpoints

- `Code/Source/solver/FE/Backends/Interfaces/*`
- `Code/Source/solver/FE/Backends/PETSc/*`
- `Code/Source/solver/FE/Backends/Trilinos/*`
- `Code/Source/solver/FE/Backends/FSILS/*`
- `Code/Source/solver/FE/Backends/NOTES.md`

### Acceptance Criteria

- A backend cannot distinguish whether a multiphysics operator came from
  first-class mixed source or manual block decomposition.
- The plan leaves a clear extension path for eligible multi-field operators to
  auto-register a derived matrix-free backend without adding a second `Forms`
  lowering target.

---

## 5A. Matrix-Free Auto-Registration Workstream

### Objective

Extend `auto_register_matrix_free` from the current single-field conservative
path to eligible multi-field operators, while keeping matrix-free as an
operator-backend derivation layered on top of the same registered block IR.

### Deliverables

- Define a block-aware matrix-free application interface:
  - block-wise `y += A*x` over `(test_field, trial_field)` contributions
  - access to field/block layout, offsets, and ghost/update requirements
  - compatibility with existing `MatrixFreeOperator` plumbing or a clean
    extension of it
- Add a setup-time derivation path from operator registry to matrix-free:
  - inspect registered block cell terms for one operator tag
  - build per-block matrix-free actions from eligible block kernels
  - compose them into one multi-field matrix-free operator
- Start with conservative eligibility rules:
  - multi-field allowed
  - linear, steady, cell-only operators first
  - no boundary, interior-face, interface-face, or global terms initially
  - explicit constraints compatibility requirements
  - explicit parallel ghost/update requirements
- Keep solver/backend semantics orthogonal:
  - assembled block matrices remain the stable solver/backend IR
  - matrix-free auto-registration is an optional derived backend
  - field-split/block preconditioning metadata still comes from the same field
    and block layout
- Add verification coverage:
  - assembled apply vs matrix-free apply parity on a representative 2x2 coupled
    system
  - auto-registration no longer throws on eligible multi-field systems
  - solver smoke tests for backends that support matrix-free apply
  - MPI/ghost-update tests where matrix-free operator backends are distributed

### File Touchpoints

- `Code/Source/solver/FE/Systems/SystemSetup.cpp`
- `Code/Source/solver/FE/Systems/FESystem.h`
- `Code/Source/solver/FE/Assembly/MatrixFreeAssembler.*`
- `Code/Source/solver/FE/Assembly/MatrixFreeOperator.*`
- `Code/Source/solver/FE/Assembly/IMatrixFreeKernel.h`
- `Code/Source/solver/FE/Backends/*`

### Acceptance Criteria

- `auto_register_matrix_free` can derive a matrix-free operator for eligible
  multi-field operators without changing the mixed-source / block-IR split.
- Matrix-free apply matches the assembled block operator on the supported
  feature set.

---

## 6. Analysis And Diagnostics Workstream

### Objective

Preserve whole-form structure for reasoning and diagnostics even though
execution uses lowered blocks.

### Deliverables

- Retain the original mixed source form in formulation records where feasible.
- Preserve block provenance:
  - which source subexpression contributed to which block,
  - which domain produced the block contribution,
  - which fields/spaces participated
- Update analysis to consume:
  - whole-form mixed metadata for structural reasoning, and
  - lowered block contributions for execution-aware reasoning
- Ensure setup and compiler errors mention mixed field names and block context.

### File Touchpoints

- `Code/Source/solver/FE/Analysis/*`
- `Code/Source/solver/FE/Systems/FormsInstaller.cpp`
- `Code/Source/solver/FE/Systems/FESystem.*`

### Acceptance Criteria

- Diagnostics for a mixed form can point back to the mixed source and to the
  affected lowered block.

---

## 7. JIT And Performance Workstream

### Objective

Keep the current per-block specialization benefits while leaving room for
future cross-block optimization.

### Deliverables

- Treat per-block JIT specialization as the first production target.
- Make mixed-form lowering compatible with per-block JIT caches.
- Add shared cache keys/provenance so equivalent mixed expressions can reuse
  lowered block compilation products where safe.
- Evaluate later whether cross-block CSE or codegen is worthwhile, but only
  after the stable mixed-to-block architecture is complete.

### File Touchpoints

- `Code/Source/solver/FE/Forms/JIT/*`
- `Code/Source/solver/FE/Forms/FormCompiler.cpp`
- `Code/Source/solver/FE/Systems/FormsInstaller.cpp`

### Acceptance Criteria

- First-class mixed authoring does not regress per-block JIT viability.

---

## 8. Verification Workstream

### Objective

Prove semantic parity between first-class mixed expressions and manual block
decomposition, then measure performance separately.

### Test Categories

- **Compiler tests**
  - mixed test/trial discovery
  - active block classification
  - zero-block elimination
  - provenance retention
- **Installer tests**
  - parity of registered blocks for mixed vs manual decomposition
  - parity across cell, boundary, interior-face, interface-face, and global
    terms
- **Assembly tests**
  - residual/Jacobian parity
  - DG/interface parity where supported
- **Backend tests**
  - identical block layout and solver behavior for mixed vs manual paths
- **Matrix-free tests**
  - assembled apply vs derived matrix-free apply parity for eligible
    multi-field operators
  - auto-registration coverage for supported multi-field systems
- **Analysis tests**
  - equivalent property claims and issue reporting
- **Performance tests**
  - manual blocks vs mixed-lowered-to-blocks
  - mixed-lowered-to-blocks vs fused execution when enabled

### Acceptance Criteria

- The mixed authoring path is semantically identical to the manual block path
  on the supported feature set.

---

## Implementation Phases

## Phase 0: Policy And Documentation ✓

### Goal

Ratify the architecture and make the boundary explicit in design docs.

### Tasks

- [x] Add this plan.
- [x] Update `Forms`, `Systems`, and `Backends` plans to reference the
  mixed-source / block-IR split.
- [x] Add API Classification table to this plan.
- [x] Standardize terminology in docs and comments: updated FormCompiler.h
  block-section header, Vocabulary.h TrialFunctions doc, FormsInstaller.h
  installMixedFormIR doc.

### Exit Criteria

- The architecture is documented consistently across modules. ✓

---

## Phase 1: Front-End Stabilization ✓

### Goal

Make first-class mixed expressions a supported and documented `Forms` API.

### Tasks

- [x] Finalize mixed expression helpers and validation rules.
  - `MixedSpace`, `TrialFunctions(W)`, `TestFunctions(W)` in Vocabulary.h
  - Validation in `compileMixed()`: rejects invalid forms, missing test
    functions, coupled placeholders, unresolved indexed access
  - Valid mixed expression rules documented in MixedFormIR.h header
- [x] Ensure mixed forms route through `compileMixed()`.
  - `FormCompiler::compile()` added as auto-detecting entry point that
    routes to `compileMixed()` for both single-field and mixed expressions
  - `installFormulation()` auto-routes single vs multi-field residuals
- [x] Keep single-field compilation behavior unchanged.
  - `compileMixed()` delegates to `compileImpl()` when ≤1 test/trial space

### Exit Criteria

- Users can write coupled mixed expressions without manual block splitting. ✓

---

## Phase 2: Mixed IR Hardening ✓

### Goal

Make `MixedFormIR` the explicit compiler bridge.

### Tasks

- [x] Add whole-form metadata and provenance.
  - `sourceExpression()`: original mixed FormExpr
  - `blockProvenance()`: per-block contributing term indices + source summary
  - `domainSummary()`: cell/boundary/interior-face/interface flags + markers
  - `allFieldRequirements()`: union of field requirements across blocks
- [x] Define stable block-classification rules.
  - Documented in MixedFormIR.h header: cell (.dx), boundary (.ds),
    interior-face (.dS), interface-face (.dI), global (not yet)
  - Valid mixed expression rules documented alongside
- [x] Add tests for block discovery and zero-block elimination.
  - `test_MixedFormIR.cpp`: 26 tests covering block structure, provenance,
    domain summary, source expression retention, auto-detect compile

### Exit Criteria

- `MixedFormIR` is sufficient for installation and diagnostics. ✓

---

## Phase 3: Systems Lowering Parity ✓

### Goal

Guarantee that mixed expressions lower into the same `Systems` operator
structure as manual block decomposition.

### Tasks

- [x] Standardize mixed-form installer entry points.
  - `installMixedFormIR()` promoted to public API (FormsInstaller.h) as a
    lower-level block installer; public residual entry remains
    `installFormulation()`
  - `installMixedBilinear()` added (compile + install bilinear)
  - `installMixedLinear()` added (compile + install linear) — uses a
    synthetic trial column internally; see Layer 3 note
  - `FESystem::operatorDefinition()` accessor for registry queries
- [x] Record source + lowered metadata.
  - `MixedFormIR` carries source expression, block provenance, domain summary
  - `installFormulation()` creates `FormulationRecord` with source metadata
- [x] Add parity tests for operator registration.
  - `test_MixedManualParity.cpp`: 5 parity tests
  - Structural: cell terms, boundary terms, zero-block elimination
  - Assembly: matrix values match between manual and mixed paths
  - IR: `installMixedFormIR` matches `installMixedBilinear`

### Exit Criteria

- Mixed and manual paths register identical block operators. ✓

---

## Phase 4: Analysis And Diagnostics ✓

### Goal

Preserve whole-form reasoning while keeping execution block-based.

### Tasks

- [x] Thread mixed provenance into formulation records and analysis context.
  - `FormulationRecord` gains `field_names`, `test_function_names`,
    `trial_function_names` for human-readable diagnostics
  - `ContributionDescriptor` gains `source_block_key`, `source_expression`,
    `block_context` linking each contribution back to its source block
  - `PropertyClaim` gains `related_contribution_indices` for claim→contribution
    tracing
- [x] Update diagnostics to report mixed field names and block context.
  - `FormContributionLowerer` populates provenance on every emitted
    contribution (both block and fallback paths)
  - `origin` strings now include field names (e.g., "FormsInstaller(test=v, trial=p)")
  - `block_context` strings include field names and IDs for diagnostics
  - `installFormulation()` extracts field names from the FESystem registry
    and test/trial names from the expression DAG
- [x] 5 new provenance tests in `test_FormContributionLowerer.cpp`:
  - `SourceBlockKeyPopulated`, `SourceExpressionRetained`,
    `FieldNamesInOrigin`, `MixedBlockProvenanceMultiField`,
    `FallbackPathHasProvenance`

### Exit Criteria

- Analysis remains source-aware after lowering. ✓
- Diagnostics for a mixed form can point back to the mixed source and to
  the affected lowered block. ✓

---

## Phase 5: Performance Consolidation ✓

### Goal

Recover any overhead from front-end lowering without changing the stable IR.

### Tasks

- [x] Add optional fused cell-block execution.
  - `MixedBlockKernelSet` now owns exact per-block mixed cell execution
  - `installMixedFormIR()` and `installFormulation()` register
    `MixedBlockKernelSet` for ≥2 exact cell-domain blocks (shared geometry
    preparation, optional colocation)
  - Verified colocated L1i-aware compilation works for mixed-compiled forms
- [x] Keep per-block JIT specialization working.
  - `installMixedFormIR()` wraps each block with `maybeWrapForJIT()`
  - Verified with `PerBlockJIT_MixedBilinear` test: JIT compilation +
    colocated module compilation succeeds
- [x] Benchmark manual vs mixed-lowered vs fused.
  - `DISABLED_Benchmark_CompilationTiming` test (enable with
    `--gtest_also_run_disabled_tests`): measures compilation overhead and
    asserts mixed < 3x manual. Prints machine-local timings to stdout.
  - Assembly value parity: `BilinearAssemblyParity_MatrixValues` verifies
    manual and mixed paths produce identical matrix entries (tolerance 1e-14)
  - `InstallFormulation_MixedResidual_AssemblesCorrectly`: finite-difference
    Jacobian verification on mixed residual confirms correctness

### Exit Criteria

- Mixed authoring is performance-neutral or better on representative problems,
  or any remaining gaps are well-understood and isolated. ✓
- Compilation: both paths produce identical kernels, so assembly cost is
  identical by construction. The `compileMixed()` front-end has a small
  constant overhead from term classification (measured locally as < 3x on a
  single-tetra micro-benchmark; representative mesh benchmarks are deferred).
- Fused execution: available for both bilinear and residual paths.

---

## Phase 6: Backend Hardening ✓

### Goal

Ensure the mixed front-end path remains fully compatible with block solver and
backend capabilities.

### Tasks

- [x] Validate parity of block layout across backends.
  - `test_BackendParity.cpp`: 6 tests verifying backend-visible artifacts
  - `DofCountAndFieldRanges`: total DOFs, per-field DOF ranges, offsets,
    components, block indices all match between manual MixedFormIR and mixed
    installMixedBilinear paths
  - `SparsityPattern`: row-by-row NNZ and column index parity
  - `BlockMap`: block count, block start/end ranges match
  - `AssembledMatrixValues`: entry-by-entry matrix parity (tolerance 1e-14)
  - `ResidualPath_DofAndSparsity`: deterministic setup via installFormulation
  - `ZeroBlockSparsity`: PP block is zero in assembled matrix (zero-block
    elimination verified through assembly, not just sparsity)
- [x] Prioritize backend limitations that hinder multiphysics adoption.
  - Parity is verified on the **generic path** (FESystem setup, sparsity,
    dense assembly views). Backend-specific matrix/vector factories and
    solver code paths (e.g., PETSc `PCFIELDSPLIT`, FSILS `depart()`,
    Trilinos `BlockMatrix`) are **not yet exercised** by the parity suite.
  - Known backend-specific limitations remain (see `Backends/NOTES.md`):
    Trilinos lacks field-split preconditioning and direct solvers;
    assembly is owned-row insertion only.
  - These are pre-existing backend gaps unrelated to the mixed-form path —
    they affect manual block decomposition identically.

### Exit Criteria

- Backend-visible block layout parity is verified on the generic
  (FESystem + dense assembly) path. ✓
- Backend-specific solver/preconditioner paths are not yet covered by
  parity tests — follow-on work if backend-specific regressions arise.

---

## Phase 7: Multi-Field Matrix-Free Auto-Registration

### Goal

Enable `auto_register_matrix_free` for eligible multi-field operators by
deriving a block-aware matrix-free apply from the same registered block
operator model used by the assembled path.

### Tasks

- Define the block-aware matrix-free application interface.
  - expose field/block layout to matrix-free apply
  - represent per-block contributions without flattening away block structure
- Implement setup-time derivation from operator registry.
  - inspect one operator tag's registered cell terms
  - wrap or lower eligible block kernels into matrix-free block actions
  - compose a single multi-field matrix-free operator
- Relax auto-registration eligibility from single-field to eligible multi-field
  operators.
  - keep the initial scope conservative: linear, steady, cell-only
  - reject unsupported domains and unsupported constraint situations explicitly
- Add verification.
  - assembled vs matrix-free apply parity on representative coupled operators
  - backend/operator smoke tests where matrix-free is supported
  - distributed ghost/update verification where applicable

### Exit Criteria

- `FESystem::setup(auto_register_matrix_free=true)` no longer rejects eligible
  multi-field systems.
- The derived multi-field matrix-free operator matches the assembled block
  operator on the supported feature set.
- The matrix-free path remains an operator-backend derivation, not a second
  mixed-form lowering pipeline.

---

## Sequencing Guidance

Recommended implementation order:

1. Phase 0
2. Phase 1
3. Phase 2
4. Phase 3
5. Phase 4
6. Phase 5
7. Phase 6
8. Phase 7

This order intentionally defers performance work until the semantic and
diagnostic contracts are stable, and treats multi-field matrix-free
auto-registration as a follow-on once the assembled mixed-to-block path is
settled.

---

## Key Design Decisions To Resolve Early

1. **Compiler entrypoint policy**
   - Should mixed detection happen automatically in the primary compile API, or
     only in explicit mixed compile/install helpers?

2. **Field binding location**
   - Should `MixedFormIR` remain field-name/space-signature oriented until
     installation, or should some `FieldId` binding happen earlier?

3. **Provenance retention**
   - How much source structure should be stored after lowering:
     whole source handle only, per-block source map, or both?

4. **Global and auxiliary variables**
   - How should boundary functionals, auxiliary state, and global scalars
     participate in the mixed lowering contract?

5. **Fused execution boundary**
   - Which fused optimizations are permitted without changing the stable IR?

6. **Block-aware matrix-free interface**
   - Should multi-field matrix-free extend `MatrixFreeOperator` directly, or
     introduce a distinct block-aware interface that can preserve `(test_field,
     trial_field)` structure explicitly?

These decisions should be recorded before implementation moves beyond Phase 2.

---

## Risks And Mitigations

- **Risk:** first-class mixed expressions become a thin wrapper around manual
  decomposition with weak diagnostics.
  - **Mitigation:** require provenance and whole-form metadata in `MixedFormIR`
    and `Systems` records.

- **Risk:** mixed authoring adds compile/runtime overhead with no recovery path.
  - **Mitigation:** preserve zero-block elimination, per-block specialization,
    and add fused execution only as a later optimization layer.

- **Risk:** backend parity diverges between mixed and manual paths.
  - **Mitigation:** parity tests must compare backend-visible block structure,
    not just residual/Jacobian values.

- **Risk:** analysis loses whole-form structure after lowering.
  - **Mitigation:** retain original mixed formulation metadata alongside lowered
    contribution records.

- **Risk:** multi-field matrix-free grows into a second execution architecture
  that diverges from the assembled block model.
  - **Mitigation:** require matrix-free operators to be derived from the same
    registered block operator structure at setup time.

---

## Exit Criteria

This plan is complete when all of the following are true:

1. Users can write one mixed weak form for supported multiphysics problems.
2. `Forms` lowers that source into `MixedFormIR`.
3. `Systems` lowers `MixedFormIR` into the existing block operator model.
4. `Assembly` and `Backends` continue to operate on block IR only.
5. Manual block decomposition and first-class mixed expressions are semantically
   equivalent on the supported feature set.
6. Analysis and diagnostics retain source-level mixed-form awareness.
7. Performance is at least explainable and tunable without changing the stable
   IR.
8. Eligible multi-field operators can optionally auto-register a derived
   matrix-free backend without changing the mixed-source / block-IR split.

---

## Initial Implementation Checklist

- [x] Add mixed-source / block-IR policy references to existing module plans.
- [x] Finalize mixed expression authoring rules in `Forms`.
- [x] Harden `MixedFormIR` metadata and provenance.
- [x] Standardize mixed-form installation entry points in `Systems`.
- [x] Add mixed-vs-manual parity tests for compiler and installer behavior.
- [x] Thread mixed provenance into `Analysis`.
- [x] Benchmark the mixed-lowered path against manual block decomposition.
- [x] Add optional fused cell-block execution only after parity is established.
- [ ] Extend `auto_register_matrix_free` to eligible multi-field operators via
  a block-aware matrix-free application interface.

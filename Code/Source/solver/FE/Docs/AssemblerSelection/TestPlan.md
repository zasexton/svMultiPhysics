# FE Assembler Auto-Selection — Test Plan

## Selection

- Default behavior: `SetupOptions{}` yields `StandardAssembler` (no decorators).
- Explicit name: `assembler_name="StandardAssembler"` yields `StandardAssembler`.
- Auto mode: `assembler_name="Auto"` yields a compatible base assembler; DG requirement forces a DG-capable base.

## DG Compatibility (Hard Error)

Minimal case:
- Create a system with at least one interior-face term registered in `OperatorRegistry`.
- Set `assembler_name="WorkStreamAssembler"` (or any DG-incompatible assembler).
- Expect `FESystem::setup()` to throw with a DG incompatibility diagnostic.

## Decorator Composition

- Enable multiple orthogonal features and confirm name/pipeline ordering:
  - scheduling + caching + vectorization → `Vectorized(Cached(Scheduled(StandardAssembler)))` (exact string may vary)

## Selection Report

- After `FESystem::setup(...)`, `FESystem::assemblerSelectionReport()` is non-empty and contains `Selected assembler: ...`.

## Matrix-Free Auto Registration (Opt-in)

- Build a single-field system with a cell-only linear operator (e.g., `mass` with `MassKernel`).
- Call `setup(SetupOptions{.auto_register_matrix_free=true})`.
- Verify `system.matrixFreeOperator("mass")` exists and matches the assembled operator on a test vector.

## Caching Behavior

Reuse existing `test_CachedAssembler.cpp` pattern:
- Assemble with a kernel that counts `computeCell()` calls.
- Second assembly with caching enabled must produce cache hits and must not increase `computeCell()` calls.

## Forms DSL Coverage (Smoke)

- Compile/install a residual via `systems::installResidualForm(...)`.
- Ensure `FESystem::setup(assembler_name="Auto")` completes and selects a DG-capable base if DG terms are present.

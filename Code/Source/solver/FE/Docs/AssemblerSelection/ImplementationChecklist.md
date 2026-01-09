# FE Assembler Auto-Selection — Implementation Checklist

## Phase 1 (Inventory & Mapping)
- Add mapping docs (this folder) describing FormExprType implications and assembler recommendations.

## Phase 2 (Gap Analysis)
- Wire `SetupOptions::assembler_name` into `FESystem::setup()`.
- Extend the assembly factory to support named selection and `"Auto"` selection.
- Add DG compatibility validation at setup time (throw on mismatch).

## Phase 3 (Design → Code)
- Add `assembly::FormCharacteristics` and `assembly::SystemCharacteristics`.
- Add `assembly::AutoSelectionPolicy` (default Conservative) and a selection report surface in Systems.
- Implement `assembly::DecoratorAssembler` and initial decorators (caching/scheduling/vectorization).
- Generalize `CachedAssembler` and `SymbolicAssembler` as decorators.

## Phase 4 (Implementation + Tests)
- Update `Systems/SystemSetup.cpp` to:
  - compute characteristics from installed kernels/forms
  - create assembler via new factory (name + characteristics)
  - enforce hard capability compatibility (DG + other required capabilities)
- Add unit tests:
  - factory returns expected named pipeline
  - DG-incompatible selection throws
  - selection report is populated in `FESystem`
  - opt-in auto matrix-free registration enables `matrixFreeOperator(op)` for eligible operators
  - caching decorator still avoids kernel recomputation

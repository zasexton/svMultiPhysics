## AUTO assembler/component selection — implementation checklist

Scope: make `assembler_name="Auto"` (opt-in) select an appropriate FE assembly backend under a **Conservative** policy (iterative + Newton workflows assumed), with hard validation (e.g., DG-required + DG-incompatible must throw), and with an explainable selection report. This extends the existing selection plumbing and decorator composition already present in:

- `Code/Source/solver/FE/Assembly/AssemblerFactory.cpp`
- `Code/Source/solver/FE/Systems/SystemSetup.cpp`

### A) Selection inputs (characteristics) and outputs (report)
- [x] Extend `assembly::FormCharacteristics` to capture additional hard requirements:
  - field requirements present (`AssemblyKernel::fieldRequirements()`)
  - parameter requirements present (`AssemblyKernel::parameterSpecs()`) (optional; for reporting only)
  - global kernels present (for reporting only)
  - File: `Code/Source/solver/FE/Assembly/AssemblerSelection.h`
- [x] Extend `assembly::SystemCharacteristics` with sizing/context data used by heuristics:
  - total DoFs, max DoFs-per-cell, max polynomial order
  - thread count (resolved), MPI world size (if initialized)
  - Files: `Code/Source/solver/FE/Assembly/AssemblerSelection.h`, `Code/Source/solver/FE/Systems/SystemSetup.cpp`
- [x] Add a selection “decision/report” type (string or struct) and propagate it back to Systems:
  - New/updated factory overload returning report (out-param is fine)
  - Store report on `systems::FESystem` and expose accessor
  - Files: `Code/Source/solver/FE/Assembly/Assembler.h`, `Code/Source/solver/FE/Assembly/AssemblerFactory.cpp`,
           `Code/Source/solver/FE/Systems/FESystem.h`, `Code/Source/solver/FE/Systems/FESystem.cpp`,
           `Code/Source/solver/FE/Systems/SystemSetup.cpp`

### B) Capability model + hard validation (beyond DG)
- [x] Add capability queries to `assembly::Assembler` (default false), and forward through `DecoratorAssembler`:
  - supports solution-dependent kernels (solution/history/time integration context)
  - supports multi-field offsets (`setRowDofMap`/`setColDofMap` with nonzero offsets)
  - supports additional field requirements (`AssemblyKernel::fieldRequirements()`)
  - supports material state provider
  - Files: `Code/Source/solver/FE/Assembly/Assembler.h`, `Code/Source/solver/FE/Assembly/DecoratorAssembler.h`
- [x] Override capability queries in production-ready assemblers:
  - `StandardAssembler`, `ParallelAssembler`, `DeviceAssembler` (via CPU fallback), `SymbolicAssembler` (wrapper), `CachedAssembler` (wrapper), `ScheduledAssembler`, `VectorizedAssembler`
  - Files: `Code/Source/solver/FE/Assembly/StandardAssembler.h`,
           `Code/Source/solver/FE/Assembly/ParallelAssembler.h`,
           `Code/Source/solver/FE/Assembly/DeviceAssembler.h`,
           `Code/Source/solver/FE/Assembly/DecoratorAssembler.h`,
           `Code/Source/solver/FE/Assembly/CachedAssembler.h`,
           `Code/Source/solver/FE/Assembly/SymbolicAssembler.h`,
           `Code/Source/solver/FE/Assembly/ScheduledAssembler.h`,
           `Code/Source/solver/FE/Assembly/VectorizedAssembler.h`
- [x] In `createAssembler(...)`, validate *all* hard requirements (not just DG) against selected (composed) assembler; throw with a clear message on incompatibility (no silent fallback).
  - File: `Code/Source/solver/FE/Assembly/AssemblerFactory.cpp`

### C) Conservative auto-selection heuristics (base assembler choice)
- [x] Introduce an auto-policy knob (default Conservative) and keep it opt-in:
  - Conservative: only selects among production-ready base assemblers (typically `StandardAssembler`; `ParallelAssembler` only when safe/available).
  - Provide placeholders for future `Performance/Experimental` without changing defaults.
  - Files: `Code/Source/solver/FE/Assembly/Assembler.h`, `Code/Source/solver/FE/Assembly/AssemblerFactory.cpp`,
           `Code/Source/solver/FE/Systems/FESystem.h`
- [x] Implement Conservative selection rules (base assembler) for `assembler_name="Auto"`:
  - If MPI is initialized and `world_size>1` **and** system is single-field (current limitation): select `ParallelAssembler`.
  - Else: select `StandardAssembler`.
  - If user explicitly requests a non-applicable assembler via `assembler_name`, throw (capability mismatch).
  - File: `Code/Source/solver/FE/Assembly/AssemblerFactory.cpp`

### D) Iterative-solver leverage: opt-in auto registration of matrix-free operators
- [x] Add an explicit opt-in flag in `systems::SetupOptions` to auto-register matrix-free operator backends from installed cell kernels (single-field only for now):
  - If enabled and an operator is eligible (cell-only, no boundary/interior/global terms), wrap its kernel(s) using `assembly::wrapAsMatrixFreeKernel(...)` and register with `OperatorBackends`.
  - Files: `Code/Source/solver/FE/Systems/FESystem.h`, `Code/Source/solver/FE/Systems/SystemSetup.cpp`,
           `Code/Source/solver/FE/Systems/OperatorBackends.h`, `Code/Source/solver/FE/Systems/OperatorBackends.cpp`,
           `Code/Source/solver/FE/Assembly/MatrixFreeAssembler.h`

### E) Tests
- [x] Update/add unit tests for:
  - Capability mismatch throws (e.g., selecting WorkStream for solution-dependent kernel should throw once validation is added).
  - Auto selection report is populated and returned via `FESystem`.
  - Auto matrix-free registration (opt-in) makes `FESystem::matrixFreeOperator(op)` available for an eligible operator.
  - Files: `Code/Source/solver/FE/Tests/Unit/Assembly/test_AssemblerSelection.cpp`,
           `Code/Source/solver/FE/Tests/Unit/Systems/test_AssemblerSelection.cpp`,
           `Code/Source/solver/FE/Tests/Unit/Systems/test_OperatorBackends.cpp` (or new file)

### F) Documentation (minimal updates)
- [x] Update the design docs to reflect the new auto-policy knob, capability validation, and the matrix-free auto-registration opt-in.
  - Files: `Code/Source/solver/FE/Docs/AssemblerSelection/Design.md`,
           `Code/Source/solver/FE/Docs/AssemblerSelection/ImplementationChecklist.md`,
           `Code/Source/solver/FE/Docs/AssemblerSelection/TestPlan.md`

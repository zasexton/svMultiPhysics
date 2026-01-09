# FE Assembler Auto-Selection — Gap Analysis

## Current Gaps (Pre-fix)

- `systems::SetupOptions::assembler_name` exists but is ignored in `Systems/SystemSetup.cpp`; the assembler is always created via `assembly::createAssembler(opts.assembly_options)`.
- `assembly::createAssembler(ThreadingStrategy)` and `createAssembler(AssemblyOptions)` always return `StandardAssembler` regardless of strategy.
- Several “specialized” classes are not `assembly::Assembler` implementations (e.g., `BlockAssembler`, `MatrixFreeAssembler`, `NonlinearAssemblyDriver`, `FunctionalAssembler`, `ColoredAssembler`, `AssemblyScheduler`, `VectorizationHelper`) and therefore cannot be returned by the existing `createAssembler()` factory without adapters or separate plumbing.
- Wrapper assemblers (`SymbolicAssembler`, `CachedAssembler`, `DeviceAssembler`) historically delegated to `StandardAssembler` but did not consistently forward capability queries like `supportsDG()` (risking false “DG unsupported” diagnostics).
- `AssemblyOptions` contains knobs (`cache_element_data`, `use_batching`, `batch_size`) that are not wired into the `StandardAssembler` path and therefore do not affect production behavior.
- Forms compilation computes rich metadata (`FormIR::requiredData()`, `hasInteriorFaceTerms()`, transient order), but it is not used to validate/select assemblers during `FESystem::setup()`.

## Primary Consequences

- Specialized assemblers/decorators are never selected, so their unit tests do not reflect production wiring.
- DG usage can silently choose an assembler that does not implement/advertise DG support, leading to runtime failures later in assembly.
- The library cannot reliably “auto-select” an appropriate assembly strategy based on a form’s IR characteristics.


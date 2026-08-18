# Mixed FE Kernel Pipeline

This note describes the production mixed-form path in the OOP FE solver after the
JIT kernel optimization refactor.

## Semantic Kernel Types

Mixed assembly now distinguishes semantic kernels from optional accelerators:

- `SingleForm`
  Exact one-form kernel for ordinary single-field bilinear, linear, and residual forms.
- `MixedBlockSet`
  Exact per-block mixed cell kernel. Each active `(test_field, trial_field)` block is assembled independently, while shared geometry, batching, and optional text-layout colocation stay inside the same semantic kernel.
- `MonolithicCell`
  Exact mixed cell-domain kernel. One semantic kernel owns all cell blocks for the mixed operator or residual row set.
- `Functional`
  Quantities and reductions that do not assemble a matrix/vector operator.

## Installer Flow

`installFormulation()` and `installMixedFormIR()` now build an explicit
`MixedKernelPlan` before registering kernels:

1. Lower the authored mixed form into exact block IR.
2. Record one `MixedKernelPlanBlock` per active block with:
   - `(test_field, trial_field)`
   - domain coverage
   - matrix intent
   - vector intent
   - residual-owner field for mixed residual rows
3. Select the semantic path:
   - `MixedBlockSet` when exact per-block cell assembly is required
   - `MonolithicCell` when cell-domain blocks can be emitted through `compileMonolithic()`
4. Register non-cell domains per block exactly as before.
5. Register either:
   - one `MixedBlockKernelSet` for exact per-block cell work, or
   - one `MonolithicCellKernel` for all mixed cell blocks

The old residual-from-Jacobian lowering switch is no longer part of the public
or internal production path.

## MonolithicCellKernel

`MonolithicCellKernel` is the semantic mixed cell kernel used when the plan
selects monolithic execution.

Each block spec carries:

- `test_field`
- `trial_field`
- `want_matrix`
- `want_vector`
- fallback exact kernel
- optional tangent IR
- optional residual IR

At setup time, `SystemSetup` resolves spaces, DOF maps, and offsets for each
block. At compile time, `MonolithicCellKernel::ensureCompiled()` forwards the
block IR list to `JITCompiler::compileMonolithic(...)`.

The compiled monolithic dispatcher is an optional acceleration only. The
qualified production path keeps exact per-block fallback execution as the
default inside `MonolithicCellKernel`, and `StandardAssembler` reuses the
compiled dispatcher only when
`SVMP_FE_ENABLE_MONOLITHIC_COMPILED_DISPATCH=1`.

If monolithic JIT compilation is unavailable, or the compiled dispatcher is not
enabled, the kernel still retains the per-block fallback kernels so exact
assembly semantics remain available.

Coupled helper splitting inside the compiled monolithic dispatcher is currently
disabled pending full case-level re-qualification on the fluid matrix.

## Setup And Assembly Flow

The execution path is intentionally direct:

1. `SystemSetup` primes kernels and records each planned cell term with an
   explicit `SemanticKernelKind`.
   Priming order is:
   - compile/prime per-block JIT kernels
   - optionally colocate block text layout inside `MixedBlockKernelSet`
   - resolve spaces, DOF maps, and offsets
2. `SystemAssembly` builds requests without changing kernel meaning.
3. `StandardAssembler` dispatches by `SemanticKernelKind`:
   - `MonolithicCell` -> shared geometry + block packing + monolithic batch call
   - `MixedBlockSet` -> shared geometry + exact per-block block loop
   - everything else -> ordinary per-kernel execution
4. Exact local matrix/vector contributions are inserted through the active
   backend view.

## Backend Capabilities

Assembler insertion optimizations are now behind
`GlobalSystemView::insertionCapabilities()`:

- `resolved_matrix_entries`
- `resolved_vector_entries`
- `contiguous_combined_matrix_insert`
- `exact_rank_one_updates`

The assembler uses these capabilities to decide whether resolved-slot insertion
or backend-specific acceleration is available. The FE layer does not change
math semantics to fit one backend.

## Active JIT Entry Points

The surviving JIT entry points each have one production role:

- `JITCompiler::compile(...)`
  Used by `JITKernelWrapper` for ordinary single-kernel lowering.
- `JITCompiler::compileSpecialized(...)`
  Used by `JITKernelWrapper` for cell and boundary specialization variants
  primed from `SystemSetup`.
- `JITCompiler::compileMonolithic(...)`
  Used only by `MonolithicCellKernel` to build the optional compiled dispatcher
  for true mixed cell-domain monolithic kernels. The semantic kernel remains
  exact even when this acceleration is disabled.
- `JITCompiler::compileColocated(...)`
  Used only by `MixedBlockKernelSet` to colocate already-exact per-block cell
  kernels without changing semantics.

There is no separate public fused mixed-residual entry point in the production
API anymore.

## Tracing

`KernelTrace.h` consolidates mixed-kernel observability behind
`SVMP_FE_KERNEL_TRACE`.

Supported channels:

- `selection`
- `specialization`
- `assembly`
- `capabilities`
- `all`

Examples:

```bash
SVMP_FE_KERNEL_TRACE=selection
SVMP_FE_KERNEL_TRACE=specialization
SVMP_FE_KERNEL_TRACE=assembly,capabilities
SVMP_FE_KERNEL_TRACE=all
```

Compatibility env vars remain wired in:

- `SVMP_OOP_SOLVER_TRACE`
- `SVMP_JIT_TRACE_SPECIALIZATION`

## What To Read First

For mixed-kernel behavior, the shortest reliable reading order is:

1. `Systems/FormsInstaller.cpp`
2. `Systems/MixedKernelPlan.h`
3. `Forms/MonolithicCellKernel.h`
4. `Systems/SystemSetup.cpp`
5. `Assembly/StandardAssembler.cpp`

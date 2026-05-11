# FE/Forms - LLVM OrcJIT Backend

This folder contains the LLVM OrcJIT backend for accelerating FE assembly of
Forms kernels.

## Support Matrix

- LLVM: >= 15.0. The implementation has compatibility paths for LLVM 15+
  APIs, including newer LLVM 18+ and 20+ API variants.
- OS: Linux / macOS / Windows
- Compilers: Clang / GCC / MSVC

## Enabling The Build

The JIT backend is part of FE/Forms, so you must enable the Assembly module.

Example (standalone FE build):

`cmake -S Code/Source/solver/FE -B build-fe -DFE_ENABLE_ASSEMBLY=ON -DFE_ENABLE_LLVM_JIT=ON -DLLVM_DIR=...`

`LLVM_DIR` should point at LLVM’s CMake package directory (usually `<prefix>/lib/cmake/llvm`), for example:
- Linux (Debian/Ubuntu packaging example): `/usr/lib/llvm-16/lib/cmake/llvm`
- macOS (Homebrew): `$(brew --prefix llvm)/lib/cmake/llvm`
- Windows (official installer): `C:/Program Files/LLVM/lib/cmake/llvm`

## Status

This backend is active and can execute LLVM-JIT kernels for supported Forms
expressions. It includes:

- KernelIR lowering, validation, optimization, and stable hashing
- LLVM IR generation for cell, boundary, interior-face, interface-face, and
  functional-total kernels
- ORC/LLJIT runtime with in-memory and filesystem object cache
- Generic and size-specialized kernels
- Optional tensor lowering, basis baking, SIMD batch execution,
  monolithic/coupled kernels, and colocated modules
- Interpreter fallback on validation, compilation, lookup, or runtime failure

Known limitations:

- SIMD batch is currently qualified only for the implemented two-lane helper
  layout; wider hardware vectors fall back to scalar batch execution until the
  helper ABI is generalized.
- Runtime coefficients on boundary paths currently fall back in the wrapper.
- Coupled helper splitting infrastructure exists but remains disabled pending
  requalification.
- Numerical behavior is controlled by `JITOptions::fast_math_mode`; strict mode
  is the default, with contract-only and relaxed modes available when callers
  choose the performance/semantics tradeoff explicitly.

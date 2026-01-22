# FE/Forms — LLVM OrcJIT Backend (Work In Progress)

This folder contains the in-progress LLVM OrcJIT backend for accelerating FE assembly of Forms kernels.

## Support Matrix

- LLVM: >= 15.0 (expected to work with 15–18)
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

- Phase 0: infrastructure only (wrapper + validation + ABI work); execution still falls back to the interpreter.
- Future phases: LLVM module emission, OrcJIT engine, kernel caching, and end-to-end correctness/benchmarking.

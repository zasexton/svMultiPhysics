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

## Object Cache

Compiled kernel objects are cached in memory and, by default, on disk so later
runs can skip LLVM code generation.  The on-disk cache directory is chosen as:

1. `JITOptions::cache_directory`, when set in code;
2. otherwise, when `JITOptions::cache_kernels` is true, the environment
   variable `SVMP_JIT_CACHE_DIR`, when set and non-empty;
3. otherwise `$HOME/.cache/svMultiPhysics/jit_cache`.

For example, `export SVMP_JIT_CACHE_DIR=$SCRATCH/svmp-jit-cache` moves the
cache off a small home filesystem, and a per-job directory isolates concurrent
jobs.  `JITEngine::objectCacheDirectory()` reports the directory in use.

If the directory was created by a different LLVM version, objects go to an
`llvm-<version>` subdirectory.  Objects are stored under `objects-v2/`, one
`<module>.objcache` file per module, each with a header recording the module
id, the object size and an FNV-1a 64-bit checksum of both
(`JITObjectCacheFile.h`).  On load, a file whose header, size, module id or
checksum does not match is rejected, deleted and recompiled (with a warning),
so a truncated or corrupted object is never linked or executed.  Raw objects
left in the cache root by earlier builds are ignored.

The directory may be shared by processes on many hosts, for example on a
parallel filesystem.  Objects are published by writing an exclusively created
temporary file named
`<module>.objcache.tmp.<host>.<pid>.<random>.<counter>` and renaming it into
place, so concurrent writers never share a temporary file and readers only see
complete files.  Stale temporary files left by killed processes are harmless
and can be deleted while no job is using the cache.

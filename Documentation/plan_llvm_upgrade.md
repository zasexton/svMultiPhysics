# Plan: LLVM Version Upgrade to 16+

## Summary

Upgrade minimum LLVM version from 14 to 16+ to unblock term-group splitting (gated by LLVM 14
physreg copy / bitcast instruction selection bugs) and gain improved optimization passes for
rolled loop kernels.

## Motivation

- Term-group splitting infrastructure is fully implemented but gated behind `SVMP_JIT_TERM_SPLIT`
  env var due to LLVM 14-specific backend bugs
- SimplifyCFG produces incorrect results for coupled kernels in LLVM 14
- LLVM 16+ has better register allocation for large IR functions (~30K IR)
- Better autovectorization for rolled QP/DOF loops
- The codebase already has `#if LLVM_VERSION_MAJOR >= 16/18/20` guards throughout

## Changes Required

### Build System
- Update `CMakeLists.txt` LLVM version requirements
- Test with LLVM 16, 17, 18, 19, 20
- Update CI/CD pipeline configurations

### LLVMGen.cpp
- Remove LLVM 14 workarounds and `#if LLVM_VERSION_MAJOR < 16` guards
- Enable term-group helper emission by default (remove env var gate)
- Re-evaluate SimplifyCFG for coupled kernels
- Test PassBuilder with TargetMachine on server hardware (not mobile)

### JITEngine.cpp
- Simplify pass pipeline configuration (remove LLVM 14 special cases)
- Re-evaluate O2 vs O3 with newer LLVM (O3 was neutral on LLVM 14)

### Testing
- Full regression suite on LLVM 16+
- Benchmark all test cases (Channel2D, iliac_artery, pipe_RCR_3d, vortex_shedding)
- Verify term-group splitting gives expected L1i reduction

## Expected Impact

- Term-group splitting: additional L1i miss reduction for large kernels
- Better rolled-loop codegen: potential 5-10% kernel speedup
- Cleaner codebase: remove ~100 lines of version guards

## Risk

- Breaking change for users on LLVM 14
- New LLVM versions may introduce new bugs (mitigated by version guards)

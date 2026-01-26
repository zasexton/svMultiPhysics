# Symbolic Differentiation + LLVM JIT Review — Gap Analysis

Scope: `Code/Source/solver/FE/` (Forms symbolic differentiation, `SymbolicNonlinearFormKernel`, LLVM JIT backend in `Forms/JIT/`).

## What Was Verified / Added

### Symbolic differentiation (correctness + coverage)
- Confirmed the symbolic differentiator has explicit handling for the full current `FormExprType` vocabulary.
- Fixed correctness/robustness issues:
  - `d(det(A))` now uses the cofactor rule `cofactor(A) : dA` (avoids singular-matrix inversion at `det(A)=0`).
  - `d(pow(a, 0))` now returns zero without constructing `pow(a, -1)` (avoids `0 * inf` / NaNs).
- Expanded simplification/constant-folding to reduce derivative expression size (safe rules only).

### Verification tests (symbolic vs AD and finite differences)
Added unit tests to `Code/Source/solver/FE/Tests/Unit/Forms/test_SymbolicDifferentiation.cpp` covering:
- Linear Poisson (sanity): `inner(grad(u), grad(v))`
- Direct operator-rule coverage: `divide` (quotient), `curl`, `hessian`, `outer`, `cross`, `negate`, `subtract`, `restrict +/-`, comparison predicates, isolated `cofactor`
- Scalar function rules: `sqrt/exp/log/abs/sign/pow`, plus `min/max` + `conditional`
- Hyperelasticity-like residual (matrix ops): `det`, `inv`, `transpose`, `log`
- Additional matrix/tensor ops: `cofactor`, `trace`, `sym`, `dev`, `skew`, `norm`
- Navier–Stokes convection: explicit component form of `(u · ∇)u`
- DG interior penalty (interior face `dS`): `avg/jump`
- Transient term: `dt(u)*v` with `TimeIntegrationContext` + previous solution
- Edge cases:
  - `det(grad(u))` at zero gradient (ensures no throw; matches AD)
  - `norm/normalize` at zero vector (matches AD)
- Finite-difference check: compares `J(u)·dU` to `[R(u+εdU)−R(u)]/ε` for a nonlinear diffusion form.

### Interpreter evaluation completeness fixes
- Added spatial-jet support for `curl(·)` and enabled an `evalReal/evalDual` fallback path so `curl` works on composite vector expressions (e.g., vector-valued `AsVector` results), not only terminals.

### LLVM JIT integration (static verification)
- `Forms/JIT/JITValidation.cpp` correctly rejects nodes that must be lowered/rewritten before LLVMGen:
  - `ParameterSymbol`, coupled `*Symbol` terminals (must be `*Ref`)
  - measure wrappers (`dx/ds/dS/dI`) in integrands
  - `IndexedAccess` (must be lowered via `forms::einsum`)
- `Forms/JIT/LLVMGen.cpp` intentionally omits emission for the rejected nodes above; it supports the slot-based refs and the lowered integrand vocabulary.

## Remaining Gaps / Follow-ups

### Interpreter-mode benchmark results (no LLVM JIT)
Using `Code/Source/solver/FE/Tests/Unit/Forms/test_SymbolicDifferentiation.cpp` disabled benchmark
`SymbolicNonlinearFormKernelBenchmark.DISABLED_ADvsSymbolic_NonlinearDiffusion_CellMatrix` on a single tetrahedron
(`H1(Tetra4, p)`), `assembleMatrix()` times were:
- `p=1` (4 dofs): AD ≈ 3.58 ms, symbolic ≈ 10.48 ms (AD/symbolic ≈ 0.34×)
- `p=2` (10 dofs): AD ≈ 21.33 ms, symbolic ≈ 154.82 ms (AD/symbolic ≈ 0.14×)
- `p=3` (20 dofs): AD ≈ 93.63 ms, symbolic ≈ 1296.47 ms (AD/symbolic ≈ 0.07×)

This is expected in **interpreter mode** because the AD kernel amortizes expression evaluation by computing
one Jacobian row per evaluation (derivative array fill), while bilinear (symbolic) assembly evaluates the scalar
integrand for each `(i,j)` entry. The performance goal for symbolic tangents is to recover the advantage via
LLVM JIT compilation (straight-line scalar code + CSE + better locality).

### End-to-end OrcJIT execution in this environment
This workspace environment does not have an `LLVMConfig.cmake` package available, so `FE_ENABLE_LLVM_JIT=ON` cannot be configured here. As a result:
- JIT execution tests/benchmarks cannot be run in this container.
- The `#if SVMP_FE_ENABLE_LLVM_JIT` JIT-vs-interpreter unit tests remain compile-time gated.

### Performance work not addressed here
- Kernel specialization (unrolling for common `(n_qpts, n_dofs)` shapes) is not implemented.
- SIMD/vectorization tuning in `LLVMGen` (vector types, alignment metadata, loop hints) is not implemented.
- Symbolic simplification is intentionally conservative; aggressive rewrites that would change AD corner-case semantics (e.g., `sqrt(pow(x,2)) -> abs(x)`) are avoided.

### Singular cofactor derivative (potential robustness issue)
`d(cofactor(A))` currently uses an `inv(A)`-based identity; this will still throw if `A` is singular even though `cofactor(A)` itself is defined there. If simulations rely on `cofactor(A)` near-singular states, consider a minor-based derivative expansion (1×1/2×2/3×3) to remove the inverse requirement.

# Implement Rank-3 Tensor (`Tensor3`) Support

This checklist tracks the work needed to fully support rank-3 tensor objects (`Tensor3`) in FE/Forms (AST + evaluation + einsum lowering + tests).

## Checklist

- [x] Add `Value::Kind::Tensor3` storage + shape metadata + `resizeTensor3`/`tensor3At` accessors (`Code/Source/solver/FE/Forms/Value.h`).
- [x] Add `Tensor3Coefficient` and plumbing through `FormExpr`/`CoefficientNode` (`Code/Source/solver/FE/Forms/FormExpr.h`, `Code/Source/solver/FE/Forms/FormExpr.cpp`).
- [x] Add `AsTensor3` constructor node + EDSL helper (`Code/Source/solver/FE/Forms/FormExpr.h`, `Code/Source/solver/FE/Forms/FormExpr.cpp`, `Code/Source/solver/FE/Forms/Vocabulary.h`).
- [x] Extend component/indexed access to 3 indices (AST + `toString`) (`Code/Source/solver/FE/Forms/FormExpr.h`, `Code/Source/solver/FE/Forms/FormExpr.cpp`).
- [x] Extend `einsum` lowering for rank-3 indexed access (`Code/Source/solver/FE/Forms/Einsum.cpp`).
- [x] Implement `Tensor3` evaluation in kernels (Real + Dual):
  - [x] Terminals: tensor3 coefficients and `AsTensor3`.
  - [x] Indexing: `component(T,i,j,k)`.
  - [x] Algebra: `+/-`, unary `-`, scalar `*`/`/`, `conditional`, `jump/avg`.
  - [x] Tensor ops: `inner(T3,T3)` full contraction, `norm(T3)`.
  - [x] Hessian: `H(vector)` returns `Tensor3` for `TestFunction`/`TrialFunction`/`DiscreteField`/`Coefficient`/`Constant`.
  (`Code/Source/solver/FE/Forms/FormKernels.cpp`)
- [x] Update/extend unit tests:
  - [x] Primitive Tensor3 smoke test (construct + `component` + `inner`/`norm`).
  - [x] Nonlinear Jacobian FD check uses `u.hessian()`/`v.hessian()` for vector spaces (no manual `component(..).hessian()`).
  - [x] Einsum rank-3 lowering test (`H(u)(i,j,k) * H(v)(i,j,k)` matches `inner(H(u),H(v))`).
  (`Code/Source/solver/FE/Tests/Unit/Forms/*`)
- [x] Run Forms unit tests locally (ctest target(s) / gtest binaries) and fix any regressions.

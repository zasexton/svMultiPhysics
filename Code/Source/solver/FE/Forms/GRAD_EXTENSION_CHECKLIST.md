# Extend `grad()` to Support Composite Expressions (Physics-Agnostic)

This checklist tracks the implementation work to make spatial derivatives in FE/Forms usable for more complex (but still physics-agnostic) formulations, including patterns needed for non-Newtonian and VMS strong-residual terms (e.g., derivatives of expressions that depend on `grad(u)`).

## Checklist

- [x] Create checklist file
- [x] Add rank-3 tensor (`Tensor3`) storage to `forms::Value` (`Code/Source/solver/FE/Forms/Value.h`)
- [x] Add `Tensor3Coefficient` + AST plumbing (`Code/Source/solver/FE/Forms/FormExpr.h`, `Code/Source/solver/FE/Forms/FormExpr.cpp`)
- [x] Extend kernel value ops to handle `Tensor3` (`Code/Source/solver/FE/Forms/FormKernels.cpp`)
- [x] Implement a physics-agnostic spatial “jet” evaluator (value + ∇ + Hessian) (`Code/Source/solver/FE/Forms/FormKernels.cpp`)
- [x] Route `grad()` through jets so it works on composite expressions (`Code/Source/solver/FE/Forms/FormKernels.cpp`)
- [x] Update `div()`/`hessian()` similarly where needed for nested derivatives (`Code/Source/solver/FE/Forms/FormKernels.cpp`)
- [x] Update FormCompiler required-data + field-requirement inference for nested derivatives (`Code/Source/solver/FE/Forms/FormCompiler.cpp`)
- [x] Add/adjust unit tests for composite/nested derivatives (`Code/Source/solver/FE/Tests/Unit/Forms/`)
- [x] Build/run relevant tests (`ctest -R FE_Forms_Tests` in `fe-build-tests/`; MPI tests are blocked by sandbox socket restrictions)

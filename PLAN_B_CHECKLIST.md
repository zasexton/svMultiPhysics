# Plan B Checklist — Affine Residual Validation + `LinearFormKernel`

Goal: When a residual form is **affine in the active TrialFunction** (i.e., can be written as `R(u;v) = a(u,v) + L(v)` where `a` is bilinear and `L` is linear), automatically:
1) **validate** that the form is affine in `u`,
2) split it into bilinear + linear parts,
3) install a dedicated `svmp::FE::forms::LinearFormKernel` via the existing `installResidualForm(...)` entrypoint.

Notes:
- Initial implementation is conservative: unsupported constructs must fall back to `NonlinearFormKernel` (correctness > optimization).

## Checklist

- [x] Add an affine/linearity analysis + splitting utility for `FormExpr` residuals (`Code/Source/solver/FE/Forms/AffineAnalysis.h`, `Code/Source/solver/FE/Forms/AffineAnalysis.cpp`).
- [x] Add a coefficients-only solution-data request bit (`RequiredData::SolutionCoefficients`) and update assemblers to honor it (`Code/Source/solver/FE/Assembly/AssemblyKernel.h`, `Code/Source/solver/FE/Assembly/StandardAssembler.cpp`).
- [x] Add `AssemblyContext` accessors for solution coefficients and update `setSolutionCoefficients()` to compute only what was requested (`Code/Source/solver/FE/Assembly/AssemblyContext.h`, `Code/Source/solver/FE/Assembly/AssemblyContext.cpp`).
- [x] Implement `svmp::FE::forms::LinearFormKernel` (assembles Jacobian from bilinear part, residual vector from `K*u + L`) (`Code/Source/solver/FE/Forms/FormKernels.h`, `Code/Source/solver/FE/Forms/FormKernels.cpp`).
- [x] Update `installResidualForm(...)` to auto-select `LinearFormKernel` when residual is affine; otherwise fall back to `NonlinearFormKernel` (`Code/Source/solver/FE/Systems/FormsInstaller.cpp`).
- [x] Add/extend unit tests that verify (a) kernel selection and (b) numerical correctness for affine residuals (`Code/Source/solver/FE/Tests/Unit/Systems/test_FormsInstaller.cpp`).
- [x] Wire new sources into the FE build (`Code/Source/solver/FE/CMakeLists.txt`).
- [x] Run `test_fe_forms` and `test_fe_systems` and ensure green.

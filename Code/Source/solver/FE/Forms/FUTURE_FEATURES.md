# FE/Forms — Future Features

This document tracks follow-on work that is **not yet implemented** in `FE/Forms`, but is expected to be needed as the module matures.

For an extensive “future PDE vocabulary” checklist (including cross-module concepts and scoping), see `FE/Forms/VOCABULARY.md`.

## Front-Ends and Compilation

- JIT / code-generation backend (the `SymbolicOptions::jit` fields are placeholders):
  - kernel caching and specialization by element type, polynomial order, and quadrature rule.
- Algebraic simplification and CSE (common-subexpression elimination) in `FormCompiler`.

## Vocabulary Expansion

- Higher derivatives:
  - Hessians for scalar terminals are implemented for affine mappings (basis Hessians + physical transform) and usable in both bilinear and residual (AD) forms.
  - Remaining work: Hessians/Laplacians for general expressions, non-affine/isoparametric mappings (second-derivative geometry terms), and robust higher-order operators (e.g., biharmonic workflows).
- General tensor ranks beyond the current scalar/vector/3×3-matrix model:
  - Rank-2 symmetric/skew are implemented as matrix sub-kinds (tag-only) via `sym(A)` / `skew(A)`.
  - Rank-4 tensors are implemented as `Value::Kind::Tensor4` + coefficients + `doubleContraction(Tensor4,Tensor)` → `Tensor`.
  - Remaining work: higher-rank indexing (rank>2), rank-4 component access, free-index Einstein results, and richer mixed/multi-field workflows.
- Complex-valued PDE workflows:
  - Complex support is provided via explicit real/imag splitting and 2×2 real block lifting (`FE/Forms/Complex.h`).
  - Remaining work: richer complex algebra ergonomics (e.g., complex trial/test handles, conjugation conventions) and/or true complex backends.
- Einstein index notation:
  - Basic fully-contracted Einstein summation is implemented via `Index` / `IndexSet`, indexed access `A(i[,j])`, and `einsum(expr)` lowering.
  - Remaining work: support free indices (vector/matrix-valued results), generalized contraction patterns, and higher-rank indexing.
- Additional nonlinear scalar functions and their AD rules (`sin/cos/tan`, `atan2`, `tanh`, etc.) and more robust handling of non-smooth ops (`min/max`, `conditional`) near switching points.
- Vector-valued trial/test functions are implemented via `spaces::ProductSpace` (vector-valued `TrialFunction(V)` / `TestFunction(V)`).
  Remaining work: tensor-valued unknowns, mixed/enriched spaces as true mixed arguments (beyond per-block compilation), and richer component/subspace selection.

## DG Robustness and Generality

- General interior-face quadrature alignment for quad faces (and mixed face types), not just simplex facets.
- Replace the current tetra “oblique face” scaling hack with a general surface-measure computation based on the facet-to-reference Jacobian (works for all element types/faces).
- Expand DG numerical-flux helpers beyond the current basics (`upwindValue`, `interiorPenaltyCoefficient`) to cover common families (SIPG/NIPG, Lax–Friedrichs/Rusanov, Roe-type fluxes) and vector-valued traces.

## Constitutive Interop (with future `FE/Constitutive`)

- A first-class adapter layer from `FE/Constitutive` law interfaces to `forms::ConstitutiveModel`.
- History-variable/state access through `AssemblyContext` (aligning with a future `RequiredData::MaterialState` flag) so constitutive models can carry quadrature-point state.

## Performance / Tooling

- Kernel-level pre-evaluation and caching of basis-dependent contractions (e.g., mass/stiffness templates).
- SIMD-friendly evaluation paths for batched quadrature.
- Optional profiling hooks and per-term timing in `FormKernel` / `NonlinearFormKernel`.

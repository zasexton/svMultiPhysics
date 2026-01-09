# FE/Forms Unit Test Checklist

This document catalogs missing unit tests for the `Code/Source/solver/FE/Forms/` subfolder identified through comprehensive code review. Each section describes the component, missing tests, and rationale for inclusion.

**Location**: `Code/Source/solver/FE/Tests/Unit/Forms/`

---

## Table of Contents

1. [Value<T> Container Tests](#1-valuet-container-tests)
2. [Dual Number AD Tests](#2-dual-number-ad-tests)
3. [Complex Number Vocabulary Tests](#3-complex-number-vocabulary-tests)
4. [Index/IndexSet Tests](#4-indexindexset-tests)
5. [FormIR Query Method Tests](#5-formir-query-method-tests)
6. [FormExpr Operator Coverage Tests](#6-formexpr-operator-coverage-tests)
7. [Vocabulary Helper Tests](#7-vocabulary-helper-tests)
8. [ConstitutiveModel Edge Case Tests](#8-constitutivemodel-edge-case-tests)
9. [BlockForm Edge Case Tests](#9-blockform-edge-case-tests)
10. [Performance and Benchmark Tests](#10-performance-and-benchmark-tests)

---

## 1. Value<T> Container Tests

**File to create**: `test_Value.cpp`

**Source**: `Forms/Value.h`

**Current coverage**: None (only indirect usage through assembly tests)

The `Value<T>` struct is the fundamental storage container for all scalar, vector, matrix, and tensor values during form evaluation. It implements a hybrid inline/dynamic storage strategy that switches based on size thresholds.

### Missing Tests

- [x] **ValueKindTransitions**: Test setting `Value::Kind` to each enum value (`Scalar`, `Vector`, `Matrix`, `SymmetricMatrix`, `SkewMatrix`, `Tensor3`, `Tensor4`) and verify corresponding accessors return correct dimensions.
  - **Why**: Kind determines which storage arrays are valid; incorrect kind can cause silent data corruption.

- [x] **VectorResizeInlineStorage**: Call `resizeVector(n)` for n = 1, 2, 3 and verify `v` array is used, `v_dyn` remains empty, and `vectorSize()` returns correct value.
  - **Why**: Inline storage (n ≤ 3) avoids heap allocation; must verify threshold behavior.

- [x] **VectorResizeDynamicStorage**: Call `resizeVector(n)` for n = 4, 10, 100 and verify `v_dyn` is populated with correct size and `vectorSpan()` returns span over dynamic storage.
  - **Why**: Large vectors must use dynamic storage; incorrect switching causes buffer overflows.

- [x] **MatrixResizeInlineStorage**: Call `resizeMatrix(r, c)` for dimensions up to 3×3 and verify inline `m` array is used.
  - **Why**: Common 3×3 matrices (rotation, stress, strain) should use fast inline storage.

- [x] **MatrixResizeDynamicStorage**: Call `resizeMatrix(4, 4)`, `resizeMatrix(3, 5)`, etc., and verify `m_dyn` is allocated with `rows * cols` elements.
  - **Why**: Non-square or large matrices require dynamic storage.

- [x] **MatrixAtAccessInlineVsDynamic**: Verify `matrixAt(r, c)` returns correct reference for both inline (3×3) and dynamic storage cases.
  - **Why**: Row-major indexing into `m_dyn` must match the 2D access pattern of inline `m[r][c]`.

- [x] **Tensor3ResizeInlineStorage**: Call `resizeTensor3(d0, d1, d2)` for total size ≤ 27 (e.g., 3×3×3) and verify `t3` array is used.
  - **Why**: Standard 3D tensors fit in inline storage.

- [x] **Tensor3ResizeDynamicStorage**: Call `resizeTensor3(4, 4, 4)` and verify `t3_dyn` is allocated with 64 elements.
  - **Why**: Large third-order tensors need dynamic allocation.

- [x] **Tensor3AtIndexing**: Verify `tensor3At(i, j, k)` computes correct linear index `(i * d1 + j) * d2 + k` for both inline and dynamic storage.
  - **Why**: Incorrect tensor indexing causes subtle numerical errors in Hessian computations.

- [x] **Tensor4FixedStorage**: Set `kind = Kind::Tensor4`, populate `t4` array, verify all 81 elements are accessible.
  - **Why**: Fourth-order tensors (elasticity tensor) use fixed 81-element array; no dynamic fallback exists.

- [x] **ZeroSizedVector**: Call `resizeVector(0)` and verify `vectorSize()` returns 0 and `vectorSpan()` returns empty span.
  - **Why**: Edge case that should not crash or return invalid spans.

- [x] **DimensionQueryConsistency**: After resize operations, verify `vectorSize()`, `matrixRows()`, `matrixCols()`, `tensor3Dim0/1/2()` all return consistent values matching the resize parameters.
  - **Why**: Dimension metadata must stay synchronized with actual storage.

---

## 2. Dual Number AD Tests

**File to extend**: `test_Dual.cpp`

**Source**: `Forms/Dual.h`

**Current coverage**: ~25% (only `add`, `mul`, `neg`, `sub`, `sin`, `cos`)

The `Dual` struct and associated functions implement forward-mode automatic differentiation for Jacobian assembly. Correctness of derivative propagation is critical for Newton-Raphson convergence.

### Missing Tests

- [x] **DivisionDualDual**: Test `div(a, b, out)` where both `a` and `b` have non-zero derivatives. Verify quotient rule: `d(a/b) = (da*b - a*db) / b²`.
  - **Why**: Division appears in many constitutive models (e.g., `1/J` for incompressibility); incorrect derivatives cause Newton divergence.

- [x] **DivisionDualScalar**: Test `div(a, scalar, out)` and verify derivative is `da / scalar`.
  - **Why**: Scaling by constants is common; simpler derivative rule must be correct.

- [x] **DivisionScalarDual**: Test `div(scalar, b, out)` and verify derivative is `-scalar * db / b²`.
  - **Why**: Inverse operations like `1/detJ` use this form.

- [x] **AbsoluteValuePositive**: Test `abs(a, out)` where `a.value > 0` and verify `out.deriv = a.deriv`.
  - **Why**: Absolute value must correctly propagate derivatives for positive inputs.

- [x] **AbsoluteValueNegative**: Test `abs(a, out)` where `a.value < 0` and verify `out.deriv = -a.deriv`.
  - **Why**: Sign flip in derivative is critical for models using `|x|`.

- [x] **AbsoluteValueZero**: Test `abs(a, out)` where `a.value = 0` and verify behavior (implementation uses `copy` which gives `a.deriv`).
  - **Why**: Edge case at non-differentiable point; document expected behavior.

- [x] **SignFunction**: Test `sign(a, out)` for positive, negative, and zero values. Verify `out.deriv` is always zero (sign is piecewise constant).
  - **Why**: Sign function has zero derivative everywhere except at discontinuity; incorrect non-zero derivatives corrupt Jacobians.

- [x] **SqrtDerivative**: Test `sqrt(a, out)` and verify `d(sqrt(a)) = da / (2*sqrt(a))`.
  - **Why**: Square root appears in norms and distances; derivative correctness is essential.

- [x] **SqrtNearZero**: Test `sqrt(a, out)` where `a.value` is very small (e.g., 1e-15) and verify no division by zero (implementation guards with `denom != 0`).
  - **Why**: Numerical stability at small values prevents NaN propagation.

- [x] **ExpDerivative**: Test `exp(a, out)` and verify `d(exp(a)) = exp(a) * da`.
  - **Why**: Exponential is used in hyperelastic models (e.g., Fung-type); chain rule must be exact.

- [x] **LogDerivative**: Test `log(a, out)` and verify `d(log(a)) = da / a`.
  - **Why**: Logarithm appears in entropy and compressibility terms.

- [x] **LogNearZero**: Test `log(a, out)` where `a.value` is small and verify numerical behavior.
  - **Why**: `log(x)` diverges as x→0; test should document expected behavior near singularity.

- [x] **PowDualDual**: Test `pow(a, b, out)` where both base and exponent have derivatives. Verify generalized power rule: `d(a^b) = a^b * (db*log(a) + b*da/a)`.
  - **Why**: Variable exponents appear in power-law materials; full derivative is non-trivial.

- [x] **PowDualScalar**: Test `pow(a, n, out)` for fixed exponent and verify `d(a^n) = n * a^(n-1) * da`.
  - **Why**: Fixed powers (squares, cubes) are common; simpler rule must be correct.

- [x] **PowZeroExponent**: Test `pow(a, 0.0, out)` and verify `out.value = 1` and `out.deriv = 0` for all `a.deriv`.
  - **Why**: `x^0 = 1` is constant; derivative must be zero regardless of base's derivative.

- [x] **PowZeroBase**: Test `pow(a, b, out)` where `a.value = 0` and verify graceful handling (implementation returns zero derivatives).
  - **Why**: `0^b` for b > 0 is zero; derivative handling at this singularity matters.

- [x] **CopyFunction**: Test `copy(a, out)` and verify both value and all derivatives are copied exactly.
  - **Why**: Copy is used internally; must preserve derivative information.

- [x] **DualWorkspaceBlockGrowth**: Call `ws.reset(n)` then call `ws.alloc()` more than 64 times (exceeding `kInitialSlots`) and verify subsequent allocations still return valid spans.
  - **Why**: Block allocator must grow capacity without losing previously allocated spans.

- [x] **DualWorkspaceResetClearsBlocks**: Call `ws.reset(n)` with different `n` values and verify blocks are cleared and reallocated appropriately.
  - **Why**: Changing DOF count should trigger fresh allocation to avoid stale data.

- [x] **DualWorkspaceZeroDofs**: Call `ws.reset(0)` and verify `ws.alloc()` returns empty span without crashing.
  - **Why**: Zero-DOF edge case should be handled gracefully.

---

## 3. Complex Number Vocabulary Tests

**File to create**: `test_Complex.cpp`

**Source**: `Forms/Complex.h`

**Current coverage**: ~10% (only `toRealBlock2x2` via `test_PrimitiveTypes`)

The complex vocabulary enables frequency-domain and wave propagation problems via real/imaginary splitting. All arithmetic operations need verification.

### Missing Tests

- [x] **ComplexScalarConstantReal**: Test `ComplexScalar::constant(re, im)` and verify `re` and `im` members are set correctly.
  - **Why**: Factory function is primary construction method; must produce valid expressions.

- [x] **ComplexScalarConstantStdComplex**: Test `ComplexScalar::constant(std::complex<Real>)` overload and verify real/imag extraction.
  - **Why**: Interoperability with `std::complex` is expected by users.

- [x] **ImaginaryUnit**: Test `I()` returns `ComplexScalar` with `re = 0` and `im = 1`.
  - **Why**: Imaginary unit is fundamental; incorrect value corrupts all complex arithmetic.

- [x] **ComplexConjugate**: Test `conj(z)` returns `{z.re, -z.im}`.
  - **Why**: Conjugate is essential for Hermitian forms and energy norms.

- [x] **ComplexAddition**: Test `a + b` for `ComplexScalar` and verify `{a.re + b.re, a.im + b.im}`.
  - **Why**: Addition is basic arithmetic; must produce correct FormExpr trees.

- [x] **ComplexSubtraction**: Test `a - b` for `ComplexScalar` and verify `{a.re - b.re, a.im - b.im}`.
  - **Why**: Subtraction symmetry with addition.

- [x] **ComplexNegation**: Test `-a` for `ComplexScalar` and verify `{-a.re, -a.im}`.
  - **Why**: Unary minus must negate both components.

- [x] **ComplexMultiplication**: Test `a * b` for `ComplexScalar` and verify `{a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re}`.
  - **Why**: Complex multiplication rule is non-trivial; incorrect implementation breaks wave propagation.

- [x] **ComplexScalarTimesFormExpr**: Test `z * expr` and `expr * z` and verify both components are scaled by the real `FormExpr`.
  - **Why**: Mixed multiplication with real expressions is common in scaling.

- [x] **ComplexScalarTimesReal**: Test `r * z` and `z * r` for `Real` scalar and verify scaling.
  - **Why**: Scalar multiplication shorthand must work correctly.

- [x] **ComplexLinearFormConstruction**: Construct `ComplexLinearForm{re, im}` and verify `isValid()` returns true when both components are valid.
  - **Why**: Linear form container needs validation.

- [x] **ComplexLinearFormInvalid**: Construct `ComplexLinearForm` with invalid `FormExpr` and verify `isValid()` returns false.
  - **Why**: Invalid detection prevents runtime errors during compilation.

- [x] **ToRealBlock2x1**: Test `toRealBlock2x1(ComplexLinearForm)` produces `BlockLinearForm` with 2 blocks where `block(0) = re` and `block(1) = im`.
  - **Why**: Block lifting for RHS vectors is essential for solving complex systems with real solvers.

- [x] **ToRealBlock2x2Symmetry**: Verify `toRealBlock2x2` produces the correct antisymmetric structure: `[Re, -Im; Im, Re]`.
  - **Why**: This structure preserves complex multiplication semantics in real arithmetic.

---

## 4. Index/IndexSet Tests

**File to create**: `test_Index.cpp`

**Source**: `Forms/Index.h`

**Current coverage**: Indirect only (via `test_Einsum.cpp`)

Index and IndexSet provide Einstein notation support for tensor expressions. Isolated tests verify the building blocks work correctly.

### Missing Tests

- [x] **IndexSetDefaultExtent**: Construct `IndexSet()` and verify `extent()` returns 3 (default for 3D problems).
  - **Why**: Default extent is critical assumption throughout the codebase.

- [x] **IndexSetCustomExtent**: Construct `IndexSet(2)` for 2D and `IndexSet(4)` for higher dimensions; verify `extent()` returns correct value.
  - **Why**: Custom extents enable 2D problems and higher-order tensor indices.

- [x] **IndexDefaultConstruction**: Construct `Index()` with no arguments and verify it has a unique `id()` and empty `name()`.
  - **Why**: Anonymous indices are common in simple expressions.

- [x] **IndexNamedConstruction**: Construct `Index("i")` and verify `name()` returns `"i"`.
  - **Why**: Named indices improve expression readability and debugging.

- [x] **IndexUniqueIds**: Construct multiple `Index` objects and verify each has a unique `id()`.
  - **Why**: Index identity is used for matching in Einstein summation; collisions cause incorrect contractions.

- [x] **IndexExtentFromSet**: Construct `Index("i", IndexSet(4))` and verify `extent()` returns 4.
  - **Why**: Extent propagation determines summation range in einsum lowering.

- [x] **IndexIdThreadSafety**: (Optional, if thread-safety is required) Construct indices from multiple threads and verify no ID collisions.
  - **Why**: The atomic counter must be thread-safe for parallel compilation scenarios.

---

## 5. FormIR Query Method Tests

**File to extend**: `test_FormCompiler.cpp` or create `test_FormIR.cpp`

**Source**: `Forms/FormIR.h`

**Current coverage**: Implicit through compilation tests

The `FormIR` class provides query methods for inspecting compiled forms. Explicit tests ensure these queries are accurate.

### Missing Tests

- [x] **HasCellTermsOnly**: Compile a form with only `.dx()` integrals and verify `hasCellTerms() == true`, `hasBoundaryTerms() == false`, `hasInteriorFaceTerms() == false`.
  - **Why**: Assembly dispatch depends on these flags; false positives waste computation.

- [x] **HasBoundaryTermsOnly**: Compile a form with only `.ds()` integrals and verify boundary flag is true, others false.
  - **Why**: Boundary-only forms (Neumann BCs) must be correctly identified.

- [x] **HasInteriorFaceTermsOnly**: Compile a DG form with only `.dS()` integrals and verify interior face flag is true.
  - **Why**: DG penalty terms must trigger interior face assembly.

- [x] **HasMixedTerms**: Compile a form with `.dx()`, `.ds()`, and `.dS()` and verify all three flags are true.
  - **Why**: Mixed forms (DG with volume and boundary) must set all relevant flags.

- [x] **IsTransientTrue**: Compile a form containing `dt(u) * v` and verify `isTransient() == true` and `maxTimeDerivativeOrder() >= 1`.
  - **Why**: Time-stepping integration needs to know if temporal terms exist.

- [x] **IsTransientFalse**: Compile a steady-state form and verify `isTransient() == false` and `maxTimeDerivativeOrder() == 0`.
  - **Why**: Steady solvers should skip temporal assembly.

- [x] **MaxTimeDerivativeOrder**: Compile forms with `dt(u, 1)`, `dt(u, 2)` and verify `maxTimeDerivativeOrder()` returns the maximum order across all terms.
  - **Why**: Time integrators (e.g., Newmark) need to know the highest derivative order.

- [x] **DumpProducesNonEmptyString**: Compile any valid form and verify `dump()` returns a non-empty string containing recognizable tokens (e.g., "TestFunction", "TrialFunction").
  - **Why**: Debug output is essential for troubleshooting; empty strings indicate broken serialization.

- [x] **TermsVectorMatchesIntegralCount**: Compile a form with N separate integral terms and verify `terms().size() == N`.
  - **Why**: Term decomposition is fundamental to assembly; incorrect count causes missing contributions.

---

## 6. FormExpr Operator Coverage Tests

**File to extend**: `test_FormExpr.cpp` or `test_FormVocabulary.cpp`

**Source**: `Forms/FormExpr.h`, `Forms/Vocabulary.h`

**Current coverage**: Good, but several operators lack direct tests

### Missing Tests

- [x] **ReferenceCoordinateX**: Test `X()` (reference coordinate) creates valid expression and assembles to reference element coordinates.
  - **Why**: Reference coordinates are used in material frame computations; `x()` (physical) is tested but not `X()`.

- [x] **StateFieldCreation**: Test `FormExpr::stateField()` creates valid expression distinct from `discreteField()`.
  - **Why**: State fields represent history variables in viscoplasticity; need separate handling from solution fields.

- [x] **CurlOperatorVector**: Test `curl(v)` on a vector coefficient and verify the result has correct structure (vector in 3D).
  - **Why**: Curl is used in electromagnetics and Navier-Stokes; currently only tested indirectly in residual contexts.

- [x] **CurlOperatorAssembly**: Assemble `inner(curl(u), curl(v)).dx()` and verify non-zero matrix for appropriate vector spaces.
  - **Why**: H(curl) forms require working curl operator.

- [x] **AsTensor3Construction**: Test `FormExpr::asTensor3()` with 27-element coefficient and verify `component(T, i, j, k)` extracts correct values.
  - **Why**: Third-order tensors (e.g., piezoelectric coupling) need explicit construction and indexing tests.

- [x] **LessOperatorDirect**: Test `a.lt(b)` directly and verify it creates comparison expression that evaluates to 0 or 1.
  - **Why**: Comparison operators are only tested through `conditional()`; direct test ensures standalone usage works.

- [x] **EqualOperatorDirect**: Test `a.eq(b)` and `a.ne(b)` for scalar expressions.
  - **Why**: Equality comparisons need verification independent of conditional expressions.

- [x] **NotEqualOperatorDirect**: Test `a.ne(b)` returns opposite of `a.eq(b)`.
  - **Why**: Inequality must be logically consistent with equality.

- [x] **GreaterOperatorDirect**: Test `a.gt(b)` and `a.ge(b)` directly.
  - **Why**: All six comparison operators should have direct tests.

---

## 7. Vocabulary Helper Tests

**File to extend**: `test_FormVocabulary.cpp`

**Source**: `Forms/Vocabulary.h`

**Current coverage**: Many helpers tested indirectly; some lack dedicated tests

### Missing Tests

- [x] **LaplacianExplicit**: Test `laplacian(u)` produces same result as `div(grad(u))` when assembled.
  - **Why**: `laplacian()` is a convenience wrapper; must be equivalent to explicit form.

- [x] **UpwindValue**: Test `upwindValue(u, beta)` on an interior face with known velocity direction and verify correct trace selection.
  - **Why**: Upwind flux is essential for advection-dominated problems; incorrect selection causes instability.

- [x] **DownwindValue**: Test `downwindValue(u, beta)` returns opposite trace from `upwindValue`.
  - **Why**: Downwind complements upwind for flux splitting schemes.

- [x] **InteriorPenaltyCoefficient**: Test `interiorPenaltyCoefficient()` returns expected scaling based on mesh size and polynomial order.
  - **Why**: IP coefficient affects DG stability; formula must match literature.

- [x] **ContractionWrapper**: Test `contraction(A, B)` produces same result as explicit index notation.
  - **Why**: Generic contraction helper must be equivalent to einsum-based form.

- [x] **HeavisideFunction**: Test `heaviside(x)` returns 0 for x < 0, 0.5 for x = 0, 1 for x > 0 when assembled.
  - **Why**: Heaviside is tested via scalar ops but deserves isolated verification of all three cases.

- [x] **IndicatorFunction**: Test `indicator(x, a, b)` returns 1 for a ≤ x ≤ b, 0 otherwise.
  - **Why**: Indicator functions are used in domain decomposition; boundary behavior matters.

- [x] **ClampFunction**: Test `clamp(x, lo, hi)` returns `lo` for x < lo, `hi` for x > hi, `x` otherwise.
  - **Why**: Clamping is common in plasticity; all three branches need testing.

---

## 8. ConstitutiveModel Edge Case Tests

**File to extend**: `test_ConstitutiveModel.cpp`

**Source**: `Forms/ConstitutiveModel.h`

**Current coverage**: Good for common cases; edge cases and validation paths untested

### Missing Tests

- [x] **OutputSpecPopulated**: Create model with non-empty `OutputSpec` (specific `ValueKind`) and verify it propagates through compilation.
  - **Why**: Output type hints enable optimization; must survive compilation pipeline.

- [x] **ExpectedInputKindValidation**: Create model that returns specific `expectedInputKind()` and verify compiler/runtime rejects mismatched inputs.
  - **Why**: Type safety for constitutive inputs prevents silent errors.

- [x] **ExpectedInputCountValidation**: Create model expecting 2 inputs and verify `constitutive(model, single_input)` fails appropriately.
  - **Why**: Arity checking prevents cryptic runtime errors.

- [x] **InteriorFaceConstitutiveContext**: Assemble constitutive model on interior face (`.dS()`) and verify `ctx.domain == Domain::InteriorFace`.
  - **Why**: Interior face evaluation context (with minus/plus sides) needs verification.

- [x] **ConstitutiveOnBothFaceSides**: Assemble `constitutive(model, u.minus())` and `constitutive(model, u.plus())` and verify both traces are correctly evaluated.
  - **Why**: DG methods need constitutive evaluation on both sides of faces.

- [x] **StateLayoutMetadataAlignment**: Create model with specific alignment requirements in `stateLayout()` and verify allocator respects alignment.
  - **Why**: SIMD operations require aligned state storage.

---

## 9. BlockForm Edge Case Tests

**File to extend**: `test_PrimitiveTypes.cpp`

**Source**: `Forms/BlockForm.h`

**Current coverage**: Basic indexing and out-of-range; edge cases untested

### Missing Tests

- [x] **BlockBilinearFormZeroSize**: Construct `BlockBilinearForm(0, 0)` and verify it handles gracefully (no crash, `numTestFields() == 0`).
  - **Why**: Degenerate case should not crash even if semantically meaningless.

- [x] **BlockBilinearFormSingleBlock**: Construct `BlockBilinearForm(1, 1)`, set single block, verify `hasBlock(0, 0)` and compilation succeeds.
  - **Why**: Trivial 1×1 case is valid for single-field problems using block API.

- [x] **BlockLinearFormZeroSize**: Construct `BlockLinearForm(0)` and verify graceful handling.
  - **Why**: Edge case protection.

- [x] **BlockLinearFormSingleBlock**: Construct `BlockLinearForm(1)`, set block, verify access and compilation.
  - **Why**: Trivial case validation.

- [x] **BlockFormPartiallyPopulated**: Create 3×3 block form with only diagonal blocks set, compile, and verify only non-empty blocks produce IR.
  - **Why**: Sparse block patterns are common (e.g., decoupled physics); empty blocks must be skipped.

---

## 10. Performance and Benchmark Tests

**File to create**: `test_FormsPerformance.cpp` (or `benchmark_Forms.cpp`)

**Source**: All Forms components

**Current coverage**: None

Performance testing is critical for a numerical library. Regressions in assembly throughput directly impact simulation time.

### Missing Benchmarks

- [x] **CellAssemblyThroughput**: Measure elements/second for stiffness matrix assembly (e.g., `inner(grad(u), grad(v)).dx()`) on P1 and P2 elements.
  - **Why**: Establishes baseline for assembly performance; detects algorithmic regressions.
  - **Metric**: Elements per second, target > 100k elem/s for P1 on single core.

- [x] **NonlinearAssemblyThroughput**: Measure elements/second for residual + Jacobian assembly using `NonlinearFormKernel`.
  - **Why**: Nonlinear problems dominate simulation time; AD overhead must be characterized.
  - **Metric**: Elements per second, compare Real vs Dual overhead ratio.

- [x] **ADOverheadRatio**: Compare evaluation time for same form with Real (no derivatives) vs Dual (with derivatives).
  - **Why**: AD should add ~2-4x overhead; higher indicates implementation issues.
  - **Metric**: Time ratio Dual/Real, target < 5x.

- [x] **FormKernelQuadraturePointEvaluation**: Measure time per quadrature point for complex expressions.
  - **Why**: Per-qpt cost dominates for high-order elements.
  - **Metric**: Nanoseconds per quadrature point.

- [x] **FormCompilerLatency**: Measure compilation time for expressions of varying complexity (simple mass matrix to complex hyperelastic residual).
  - **Why**: Compilation happens once per form; acceptable latency is 1-100ms.
  - **Metric**: Milliseconds per compilation.

- [x] **DualWorkspaceAllocationThroughput**: Measure allocations/second from `DualWorkspace` under heavy load (1M+ allocations).
  - **Why**: Workspace is used per-element; allocation must be near-zero cost.
  - **Metric**: Allocations per second, target > 10M/s.

- [x] **LargeDofScaling**: Measure assembly time scaling for elements with 10, 50, 100, 500 DOFs (high-order or vector spaces).
  - **Why**: DOF count affects both loop iterations and memory access patterns.
  - **Metric**: Time vs DOF count curve, expect O(n²) for matrix assembly.

- [x] **BoundaryAssemblyThroughput**: Measure faces/second for boundary integral assembly.
  - **Why**: Boundary conditions can be bottleneck for surface-dominated problems.
  - **Metric**: Faces per second.

- [x] **DGInteriorFaceAssemblyThroughput**: Measure faces/second for interior face assembly with jump/average operators.
  - **Why**: DG methods have 2x the face assembly work of CG; performance is critical.
  - **Metric**: Faces per second, compare to cell assembly.

- [x] **MemoryAllocationProfile**: Profile heap allocations during assembly of large problem (10k+ elements).
  - **Why**: Excessive allocations cause cache thrashing and GC pressure.
  - **Metric**: Allocations per element, target < 10 during steady-state assembly.

---

## Summary

| Priority | Category | New Tests | Effort |
|----------|----------|-----------|--------|
| **HIGH** | Value<T> Container | 12 | Medium |
| **HIGH** | Performance Benchmarks | 10 | High |
| **MEDIUM-HIGH** | Dual AD Functions | 20 | Medium |
| **MEDIUM** | Complex Vocabulary | 14 | Low-Medium |
| **MEDIUM** | FormExpr Operators | 9 | Low |
| **LOW-MEDIUM** | Vocabulary Helpers | 8 | Low |
| **LOW-MEDIUM** | Index/IndexSet | 7 | Low |
| **LOW** | FormIR Queries | 9 | Low |
| **LOW** | ConstitutiveModel Edge | 6 | Medium |
| **LOW** | BlockForm Edge | 5 | Low |

**Total**: ~100 new test cases

---

## Implementation Notes

1. **Test Fixtures**: Reuse `SingleTetraMeshAccess` and `FormsTestHelpers.h` for consistency.

2. **Numerical Tolerances**: Use `1e-12` for exact arithmetic, `1e-6` for finite difference comparisons.

3. **Benchmark Framework**: Consider Google Benchmark or simple `std::chrono` timing with warmup iterations.

4. **CI Integration**: Performance tests should run on dedicated hardware with baseline comparison.

5. **Test Naming**: Follow existing pattern `TEST(Category, DescriptiveTestName)`.

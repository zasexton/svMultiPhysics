# Unit Test Checklist for FE Basis Library

This document lists additional unit tests that should be added to improve coverage and ensure correctness against established finite element literature. Tests are organized by priority and category.

**Last Updated:** 2026-01-12
**Status Legend:**
- [ ] Not started
- [~] In progress
- [x] Completed

---

## High Priority Tests

### 1. Polynomial Reproduction Tests

These tests verify the fundamental approximation property: a degree-p basis should exactly reproduce polynomials up to degree p.

#### 1.1 `LagrangeBasis_LineInterpolatesPolynomials`
- [x] Completed
- **What it tests**: Lagrange basis on Line2 exactly interpolates polynomials x^k for k = 0, 1, ..., p
- **Why needed**: Polynomial reproduction is the defining property that guarantees optimal convergence rates (Bramble-Hilbert lemma). Currently only Pyramid14 has this test.
- **Implementation**: Set nodal values to f(x_i) for f(x) = x^k, verify interpolant matches f at random interior points

#### 1.2 `LagrangeBasis_QuadInterpolatesPolynomials`
- [x] Completed
- **What it tests**: Lagrange basis on Quad4 exactly interpolates tensor-product polynomials x^i * y^j for i,j <= p
- **Why needed**: Ensures the tensor-product construction preserves polynomial reproduction
- **Implementation**: Test monomials 1, x, y, xy, x^2, y^2, x^2*y, xy^2, x^2*y^2 for p=2

#### 1.3 `LagrangeBasis_TriangleInterpolatesPolynomials`
- [x] Completed
- **What it tests**: Lagrange basis on Triangle3 exactly interpolates polynomials in P_p (complete polynomial space)
- **Why needed**: Simplex elements use different polynomial spaces than tensor-product elements
- **Implementation**: For p=2, test 1, x, y, x^2, xy, y^2 (6 basis functions)

#### 1.4 `LagrangeBasis_HexInterpolatesPolynomials`
- [x] Completed
- **What it tests**: Lagrange basis on Hex8 exactly interpolates Q_p polynomials
- **Why needed**: 3D tensor-product verification
- **Implementation**: Test all monomials x^i * y^j * z^k for i,j,k <= p

#### 1.5 `LagrangeBasis_TetraInterpolatesPolynomials`
- [x] Completed
- **What it tests**: Lagrange basis on Tetra4 exactly interpolates P_p polynomials
- **Why needed**: 3D simplex polynomial space verification
- **Implementation**: For p=2, test all 10 monomials in the complete quadratic space

#### 1.6 `LagrangeBasis_WedgeInterpolatesPolynomials`
- [x] Completed
- **What it tests**: Wedge basis interpolates appropriate polynomial space (P_p triangle x Q_p line)
- **Why needed**: Mixed topology requires careful polynomial space definition
- **Implementation**: Test representative monomials from the wedge polynomial space

---

### 2. BDM Basis Comprehensive Tests

The BDM (Brezzi-Douglas-Marini) basis previously had minimal coverage. These tests bring coverage to parity with RT/Nedelec.

#### 2.1 `BDMBasis_TriangleEdgeNormalMomentKronecker`
- [x] Completed
- **What it tests**: BDM edge DOFs satisfy Kronecker property: integral of (v_i . n) * L_j over edge_k = delta_{ij} for edge DOFs
- **Why needed**: DOF definitions are fundamental to H(div) conformity and assembly
- **Implementation**: Integrate v . n weighted by `LagrangeBasis(Line2,k)` over each edge

#### 2.2 `BDMBasis_QuadEdgeNormalMomentKronecker`
- [x] Completed
- **What it tests**: Same as above for Quad elements
- **Why needed**: Tensor-product BDM construction verification

#### 2.3 `BDMBasis_DimensionFormulas`
- [x] Completed
- **What it tests**: dim(BDM_k) = (k+1)(k+2) on triangles, 2(k+1)^2 on quads
- **Why needed**: Correct DOF count is essential for assembly
- **Implementation**: Verify size() matches formula for k = 1, 2, 3

#### 2.4 `BDMBasis_DofAssociations`
- [x] Completed
- **What it tests**: DOF associations correctly identify edge vs interior DOFs
- **Why needed**: Required for global DOF numbering and inter-element continuity
- **Implementation**: Verify entity_type for each DOF matches expected location

#### 2.5 `BDMBasis_PolynomialReproduction`
- [x] Completed
- **What it tests**: BDM_k contains all vector polynomials of degree k
- **Why needed**: Defines approximation capability
- **Implementation**: Project vector polynomial onto BDM space, verify exact reproduction

#### 2.6 `BDMBasis_HexDivergenceAndSize`
- [x] Completed
- **What it tests**: BDM on hexahedra has correct divergence and DOF count
- **Why needed**: 3D BDM is used in mixed methods for Darcy flow
- **Implementation**: Similar to existing RT tests

---

### 3. Boundary and Edge Case Tests

These tests verify correct behavior at element boundaries and for degenerate inputs.

#### 3.1 `LagrangeBasis_EvaluationAtElementCorners`
- [x] Completed
- **What it tests**: Basis functions evaluate correctly at all corner nodes (xi = +/-1 for tensor-product, vertices for simplices)
- **Why needed**: Corner evaluation is critical for assembly and often involves special cases in implementation
- **Implementation**: Systematically test all corners for all element types

#### 3.2 `LagrangeBasis_EvaluationAtEdgeMidpoints`
- [x] Completed
- **What it tests**: Basis functions evaluate correctly at edge midpoints
- **Why needed**: Edge midpoints are quadratic nodes and face special handling
- **Implementation**: For quadratic elements, verify edge node basis functions

#### 3.3 `VectorBasis_EvaluationAtFaceCenters`
- [x] Completed
- **What it tests**: RT/Nedelec/BDM evaluate correctly at face centers
- **Why needed**: Face centers are important for flux evaluation
- **Implementation**: Evaluate at face centroids, verify against analytical expressions

#### 3.4 `LagrangeBasis_GradientAtElementBoundary`
- [x] Completed
- **What it tests**: Gradients are well-defined and continuous as xi approaches +/-1
- **Why needed**: Numerical issues can arise at element boundaries
- **Implementation**: Evaluate gradients at xi = 1-epsilon for small epsilon

#### 3.5 `PyramidBasis_EvaluationNearApex`
- [x] Completed
- **What it tests**: Pyramid basis functions remain well-behaved near the apex (z -> 1)
- **Why needed**: Pyramid bases involve rational functions that can be singular at apex
- **Implementation**: Evaluate at z = 0.99, 0.999, verify values remain bounded

---

### 4. Literature Validation Tests

These tests compare computed values against published reference values.

#### 4.1 `LagrangeBasis_QuadraticTriangleMatchesZienkiewicz`
- [x] Completed
- **What it tests**: Shape functions match values in Zienkiewicz-Taylor Table 6.2
- **Why needed**: External validation against authoritative FE textbook
- **Implementation**: Evaluate N_i at specific points, compare to tabulated values

#### 4.2 `LagrangeBasis_LinearHexMatchesHughes`
- [x] Completed
- **What it tests**: Trilinear hex shape functions match Hughes "The Finite Element Method" formulas
- **Why needed**: Standard reference for isoparametric elements
- **Implementation**: Verify N_i = (1/8)(1 + xi_i*xi)(1 + eta_i*eta)(1 + zeta_i*zeta)

#### 4.3 `RaviartThomasBasis_RT0TriangleMatchesBoffiBrezziFortin`
- [x] Completed
- **What it tests**: RT0 triangle basis matches formulas in "Mixed Finite Element Methods and Applications"
- **Why needed**: Authoritative reference for mixed methods
- **Implementation**: Verify v_i = (x - x_i) / (2 * area) for opposite vertex formulation

#### 4.4 `NedelecBasis_ND0TetraMatchesMonk`
- [x] Completed
- **What it tests**: Lowest-order Nedelec tetrahedron matches Monk "Finite Element Methods for Maxwell's Equations"
- **Why needed**: Standard reference for H(curl) elements
- **Implementation**: Compare edge basis functions against published formulas

---

## Medium Priority Tests

### 5. High-Order Polynomial Tests (p > 4)

#### 5.1 `LagrangeBasis_Order5And6KroneckerProperty`
- [x] Completed
- **What it tests**: Kronecker property holds for p = 5, 6
- **Why needed**: High-order methods are increasingly used; tests catch potential overflow/precision issues
- **Implementation**: Same as existing Kronecker tests but for higher orders

#### 5.2 `LagrangeBasis_Order5And6PartitionOfUnity`
- [x] Completed
- **What it tests**: Partition of unity holds for p = 5, 6
- **Why needed**: Higher-order bases involve larger coefficient magnitudes
- **Implementation**: Same as existing tests for higher orders

#### 5.3 `LagrangeBasis_Order5And6GradientSumZero`
- [x] Completed
- **What it tests**: Gradient sum property holds for p = 5, 6
- **Why needed**: Gradient computation involves derivative of basis coefficients
- **Implementation**: Same as existing tests for higher orders

#### 5.4 `HierarchicalBasis_HighOrderConditionNumber`
- [x] Completed
- **What it tests**: Gram matrix condition number growth rate for p = 2, 3, ..., 8
- **Why needed**: Hierarchical bases should have better conditioning than nodal; verify this
- **Implementation**: Compute Gram matrix, estimate condition number, verify sub-exponential growth

---

### 6. SpectralBasis Expanded Tests

#### 6.1 `SpectralBasis_QuadGradientsMatchFiniteDifference`
- [x] Completed
- **What it tests**: 2D spectral basis gradients match numerical derivatives
- **Why needed**: Only 1D gradient test exists currently
- **Implementation**: Central difference comparison for both partial derivatives

#### 6.2 `SpectralBasis_HexGradientsMatchFiniteDifference`
- [x] Completed
- **What it tests**: 3D spectral basis gradients match numerical derivatives
- **Why needed**: 3D gradient computation verification
- **Implementation**: Central difference comparison for all three partials

#### 6.3 `SpectralBasis_HessiansMatchFiniteDifference`
- [x] Completed
- **What it tests**: Spectral basis Hessians (if implemented) match numerical second derivatives
- **Why needed**: Second derivatives needed for some formulations
- **Implementation**: Central difference on gradients

#### 6.4 `SpectralBasis_GLLNodesMatchLiterature`
- [x] Completed
- **What it tests**: GLL node locations match published tables (e.g., Canuto et al.)
- **Why needed**: GLL nodes are roots of (1-x^2)P'_n(x); verify numerical accuracy
- **Implementation**: Compare against tabulated values for n = 2, 3, 4, 5

#### 6.5 `SpectralBasis_GradientSumZero`
- [x] Completed
- **What it tests**: Sum of gradients equals zero (partition of unity derivative)
- **Why needed**: Verify spectral basis satisfies same property as Lagrange
- **Implementation**: Same as Lagrange gradient sum test

---

### 7. Serendipity Basis Geometry Mode Tests

#### 7.1 `SerendipityBasis_GeometryModeHex20`
- [x] Completed
- **What it tests**: geometry_mode=true produces correct reduced basis for geometry mapping
- **Why needed**: Isoparametric elements often use different basis for geometry vs field
- **Implementation**: Verify size and properties match geometry representation needs

#### 7.2 `SerendipityBasis_GeometryModeWedge15`
- [x] Completed
- **What it tests**: Same as above for wedge elements
- **Why needed**: Mixed topology geometry representation

#### 7.3 `SerendipityBasis_GeometryModePyramid13`
- [x] Completed
- **What it tests**: Same as above for pyramid elements
- **Why needed**: Pyramid geometry representation

---

### 8. Bernstein Basis Expanded Tests

#### 8.1 `BernsteinBasis_NonNegativity`
- [x] Completed
- **What it tests**: All Bernstein basis functions are >= 0 everywhere in reference element
- **Why needed**: Non-negativity is a defining property of Bernstein polynomials
- **Implementation**: Evaluate at random points, verify all values >= 0

#### 8.2 `BernsteinBasis_ConvexHullProperty`
- [x] Completed
- **What it tests**: Interpolant lies within convex hull of control points
- **Why needed**: Key property for geometric modeling and stability
- **Implementation**: Generate random control values, verify interpolant bounded by min/max

#### 8.3 `BernsteinBasis_EndpointInterpolation`
- [x] Completed
- **What it tests**: B_0(0. = 1, B_n(1) = 1 (endpoint basis functions)
- **Why needed**: Bernstein polynomials interpolate at endpoints
- **Implementation**: Verify first and last basis functions equal 1 at respective endpoints

#### 8.4 `BernsteinBasis_GradientsMatchNumerical`
- [x] Completed
- **What it tests**: Analytical gradients match finite differences
- **Why needed**: Gradient tests missing for Bernstein (only partition of unity tested)
- **Implementation**: Central difference comparison

#### 8.5 `BernsteinBasis_HessiansMatchNumerical`
- [x] Completed
- **What it tests**: Analytical Hessians match finite differences
- **Why needed**: Second derivatives for Bernstein not currently tested
- **Implementation**: Central difference on gradients

---

### 9. BatchEvaluator Expanded Tests

#### 9.1 `BatchEvaluator_HessianBatchMatchesPointwise`
- [x] Completed
- **What it tests**: Batched Hessian evaluation matches point-by-point evaluation
- **Why needed**: Hessian batch operations not currently tested
- **Implementation**: Compare batched vs loop over quadrature points

#### 9.2 `BatchEvaluator_LargeQuadratureRule`
- [x] Completed
- **What it tests**: Batch evaluation works correctly for high-order quadrature (many points)
- **Why needed**: Memory allocation and indexing edge cases
- **Implementation**: Use order 10+ quadrature rule

#### 9.3 `BatchEvaluator_SmallQuadratureRule`
- [x] Completed
- **What it tests**: Batch evaluation works correctly for minimal quadrature (1-2 points)
- **Why needed**: Edge case for small problems
- **Implementation**: Use order 1 quadrature rule

#### 9.4 `BatchEvaluator_HighOrderBasis`
- [x] Completed
- **What it tests**: Batch evaluation works for p = 4, 5 bases
- **Why needed**: Many basis functions per point tests memory layout
- **Implementation**: Use high-order Lagrange basis

---

### 10. ModalTransform Expanded Tests

#### 10.1 `ModalTransform_ConditionNumberVsOrder`
- [x] Completed
- **What it tests**: Condition number growth rate as polynomial order increases
- **Why needed**: Numerical stability for high-order hp-FEM
- **Implementation**: Compute condition numbers for p = 1, 2, ..., 8, verify growth rate

#### 10.2 `ModalTransform_TriangleRoundTrip`
- [x] Completed
- **What it tests**: Modal <-> nodal transform round-trips correctly on triangles
- **Why needed**: Simplex transforms may have different numerical properties
- **Implementation**: Same as line/quad tests but for triangles

#### 10.3 `ModalTransform_HexRoundTrip`
- [x] Completed
- **What it tests**: Modal <-> nodal transform round-trips correctly on hexahedra
- **Why needed**: 3D tensor-product verification
- **Implementation**: Same as line/quad tests but for hexahedra

---

## Lower Priority Tests

### 11. Thread Safety and Concurrency Tests

#### 11.1 `BasisCache_ConcurrentDifferentBases`
- [x] Completed
- **What it tests**: Multiple threads requesting different basis types simultaneously
- **Why needed**: Real applications evaluate different bases concurrently
- **Implementation**: Spawn threads requesting Lagrange, Hierarchical, Spectral simultaneously

#### 11.2 `BasisCache_HighContentionSingleEntry`
- [x] Completed
- **What it tests**: Many threads (32+) requesting same cache entry
- **Why needed**: Stress test mutex implementation
- **Implementation**: Spawn 32 threads all requesting same basis/quadrature combination

#### 11.3 `BasisCache_ConcurrentClearAndAccess`
- [x] Completed
- **What it tests**: Cache clear during concurrent access doesn't cause crashes
- **Why needed**: Cache may be cleared during runtime in some applications
- **Implementation**: One thread clearing cache while others access it

---

### 12. BasisFactory Comprehensive Tests

#### 12.1 `BasisFactory_AllSupportedCombinations`
- [x] Completed
- **What it tests**: Factory successfully creates all documented basis/element/order combinations
- **Why needed**: Ensures factory coverage matches implementation
- **Implementation**: Iterate over all valid combinations, verify creation succeeds

#### 12.2 `BasisFactory_InvalidCombinationsThrow`
- [x] Completed
- **What it tests**: Factory throws appropriate exceptions for unsupported combinations
- **Why needed**: Clear error messages for users
- **Implementation**: Test known invalid combinations (e.g., Serendipity on Tetra)

#### 12.3 `BasisFactory_ContinuitySelection`
- [x] Completed
- **What it tests**: Factory selects correct basis type based on continuity requirement
- **Why needed**: C0 vs C1 vs H(div) vs H(curl) logic verification
- **Implementation**: Verify Hermite selected for C1, RT for H(div), Nedelec for H(curl)

---

### 13. TensorBasis Tests

#### 13.1 `TensorBasis_AnisotropicHessiansMatchNumerical`
- [x] Completed
- **What it tests**: Hessians for anisotropic tensor-product basis match finite differences
- **Why needed**: Anisotropic orders tested for values/gradients but not Hessians
- **Implementation**: Central difference on gradients for px != py case

#### 13.2 `TensorBasis_3DAnisotropicConstruction`
- [x] Completed
- **What it tests**: 3D tensor product with different orders in each direction
- **Why needed**: hp-FEM often uses anisotropic refinement
- **Implementation**: Construct with px, py, pz different, verify size and properties

---

### 14. Numerical Stability Tests

#### 14.1 `LagrangeBasis_NumericalPrecisionHighOrder`
- [x] Completed
- **What it tests**: Basis values remain accurate (not polluted by round-off) for p = 6, 7, 8
- **Why needed**: High-order Lagrange on equispaced nodes is ill-conditioned
- **Implementation**: Verify partition of unity holds to reasonable tolerance

#### 14.2 `HierarchicalBasis_StabilityHighOrder`
- [x] Completed
- **What it tests**: Hierarchical basis remains well-conditioned for p = 6, 7, 8
- **Why needed**: Modal bases should be more stable; verify this
- **Implementation**: Compare condition numbers vs Lagrange

#### 14.3 `SpectralBasis_StabilityHighOrder`
- [x] Completed
- **What it tests**: GLL spectral basis stability for p = 8, 10, 12
- **Why needed**: Spectral methods often use high order
- **Implementation**: Verify Kronecker property still holds accurately

---

### 15. Quadrature Integration Tests

#### 15.1 `LagrangeBasis_MassMatrixSymmetry`
- [x] Completed
- **What it tests**: Numerically integrated mass matrix is symmetric
- **Why needed**: Symmetry is essential for many solvers
- **Implementation**: Compute M_ij = integral(N_i * N_j), verify M = M^T

#### 15.2 `LagrangeBasis_MassMatrixPositiveDefinite`
- [x] Completed
- **What it tests**: Mass matrix is positive definite
- **Why needed**: Ensures well-posed problems
- **Implementation**: Compute mass matrix, verify all eigenvalues positive

#### 15.3 `LagrangeBasis_StiffnessMatrixSymmetry`
- [x] Completed
- **What it tests**: Stiffness matrix is symmetric for symmetric material tensor
- **Why needed**: Fundamental property for elliptic problems
- **Implementation**: Compute K_ij = integral(grad N_i . D . grad N_j), verify K = K^T

---

## Checklist Summary

| Category | Tests | Priority |
|----------|-------|----------|
| Polynomial Reproduction | 6 | High |
| BDM Comprehensive | 6 | High |
| Boundary/Edge Cases | 5 | High |
| Literature Validation | 4 | High |
| High-Order (p > 4) | 4 | Medium |
| SpectralBasis Expanded | 5 | Medium |
| Serendipity Geometry Mode | 3 | Medium |
| Bernstein Expanded | 5 | Medium |
| BatchEvaluator Expanded | 4 | Medium |
| ModalTransform Expanded | 3 | Medium |
| Thread Safety | 3 | Lower |
| BasisFactory Comprehensive | 3 | Lower |
| TensorBasis | 2 | Lower |
| Numerical Stability | 3 | Lower |
| Quadrature Integration | 3 | Lower |
| **Total** | **59** | |

---

## References

1. Zienkiewicz, O.C., Taylor, R.L., Zhu, J.Z. "The Finite Element Method: Its Basis and Fundamentals" 7th ed.
2. Hughes, T.J.R. "The Finite Element Method: Linear Static and Dynamic Finite Element Analysis"
3. Ern, A., Guermond, J.L. "Theory and Practice of Finite Elements"
4. Boffi, D., Brezzi, F., Fortin, M. "Mixed Finite Element Methods and Applications"
5. Monk, P. "Finite Element Methods for Maxwell's Equations"
6. Canuto, C., et al. "Spectral Methods: Fundamentals in Single Domains"
7. Szabo, B., Babuska, I. "Finite Element Analysis"

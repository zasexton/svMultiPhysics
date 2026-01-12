# Unit Test Checklist for FE Quadrature Library

This document lists additional unit tests that should be added to improve coverage and ensure correctness against established numerical integration literature. Tests are organized by priority and category.

---

## High Priority Tests

### 1. Fix and Enable High-Order Symmetric Triangle Tests

The existing `DISABLED_HighOrderPolynomialExactness` test indicates implementation issues for orders 10-20.

#### 1.1 [x] `SymmetricTriangleQuadrature_Order10PolynomialExactness`
- **What it tests**: Order 10 Dunavant rule integrates all monomials x^a y^b with a+b ≤ 10 exactly
- **Why needed**: Currently disabled due to implementation bug; high-order FEM requires these rules
- **Implementation**: Fix underlying coefficient tables, then enable polynomial sweep test

#### 1.2 [x] `SymmetricTriangleQuadrature_Orders11to20PolynomialExactness`
- **What it tests**: Orders 11-20 Dunavant rules achieve advertised polynomial exactness
- **Why needed**: High-order spectral/hp-FEM methods require accurate high-order quadrature
- **Implementation**: Verify against Dunavant 1985 and Wandzura-Xiao 2003 coefficient tables

#### 1.3 [x] `SymmetricTriangleQuadrature_Order10to20WeightPositivity`
- **What it tests**: Document which high-order rules have negative weights
- **Why needed**: Negative weights can cause numerical instability; users need to know
- **Implementation**: Flag rules with negative weights, verify they still integrate correctly

---

### 2. Literature Value Comparison Tests

These tests compare computed quadrature points and weights against published reference values.

#### 2.1 [x] `GaussQuadrature1D_MatchesAbramowitzStegun`
- **What it tests**: n-point Gauss-Legendre nodes and weights match Abramowitz & Stegun Table 25.4
- **Why needed**: External validation against authoritative reference; catches transcription errors
- **Implementation**: For n=2,3,4,5,6, compare nodes/weights to 15-digit tabulated values

#### 2.2 [x] `GaussLobattoQuadrature1D_MatchesNISTDLMF`
- **What it tests**: Gauss-Lobatto nodes and weights match NIST Digital Library of Mathematical Functions
- **Why needed**: External validation for spectral element methods
- **Implementation**: Compare against DLMF Section 3.5 tables

#### 2.3 [x] `SymmetricTriangleQuadrature_MatchesDunavant1985`
- **What it tests**: Orders 1-20 point locations and weights match original Dunavant paper
- **Why needed**: Catches coefficient transcription errors; authoritative reference
- **Implementation**: Compare against Table I in Dunavant IJNME 1985

#### 2.4 [x] `SymmetricTetrahedronQuadrature_MatchesKeastTables`
- **What it tests**: Orders 1-8 point locations and weights match Keast's published tables
- **Why needed**: External validation for tetrahedral elements
- **Implementation**: Compare against Keast CMAME 1986 tables

#### 2.5 [x] `GaussQuadrature1D_MatchesGolubWelsch`
- **What it tests**: High-order (n=10,15,20) nodes match Golub-Welsch algorithm output
- **Why needed**: Verify numerical computation of high-order roots
- **Implementation**: Compare against reference implementation or MATLAB's `legendre` roots

---

### 3. Convergence Rate Tests

These tests verify that quadrature error decreases at the expected rate for smooth functions.

#### 3.1 [x] `GaussQuadrature1D_ConvergenceRateExponential`
- **What it tests**: Error in integrating exp(x) decreases exponentially with number of points
- **Why needed**: Gauss quadrature has exponential convergence for analytic functions
- **Implementation**: Compute errors for n=2,4,8,16; verify log(error) linear in n

#### 3.2 [x] `TriangleQuadrature_ConvergenceRateForSmoothFunction`
- **What it tests**: Error decreases as O(h^{2p}) for order-p rule on smooth function
- **Why needed**: Verifies quadrature achieves theoretical convergence rate
- **Implementation**: Integrate sin(πx)sin(πy) at orders 2,4,6,8; verify convergence slope

#### 3.3 [x] `TetrahedronQuadrature_ConvergenceRateForSmoothFunction`
- **What it tests**: Same as above for tetrahedra
- **Why needed**: 3D simplex rules have different convergence characteristics
- **Implementation**: Integrate exp(x+y+z) at increasing orders; verify convergence

#### 3.4 [x] `HexahedronQuadrature_TensorProductConvergence`
- **What it tests**: Tensor product rules converge at rate O(n^{-2p}) in each direction
- **Why needed**: Verifies tensor product construction preserves 1D convergence
- **Implementation**: Compare isotropic vs anisotropic convergence rates

---

### 4. Numerical Stability Tests

#### 4.1 [x] `SymmetricTriangleQuadrature_NegativeWeightStability`
- **What it tests**: Rules with negative weights don't suffer catastrophic cancellation
- **Why needed**: Negative weights can amplify round-off errors
- **Implementation**: Integrate 1 + ε*f(x,y) for small ε; verify no precision loss

#### 4.2 [x] `GaussQuadrature1D_HighOrderStability`
- **What it tests**: n=20,30,40 point rules remain accurate despite potential ill-conditioning
- **Why needed**: High-order Gauss rules can have numerical issues
- **Implementation**: Verify weight sums and polynomial exactness at high orders

#### 4.3 [x] `SymmetricTetrahedronQuadrature_NegativeWeightStability`
- **What it tests**: Orders 3,4 with negative weights maintain accuracy
- **Why needed**: Keast rules 3,4 have documented negative weights
- **Implementation**: Same methodology as triangle test

#### 4.4 [x] `PyramidQuadrature_ApexStability`
- **What it tests**: Points near apex (z→1) don't cause numerical issues
- **Why needed**: Pyramid mapping has Jacobian singularity at apex
- **Implementation**: Evaluate basis functions and integrands near apex

---

## Medium Priority Tests

### 5. Gauss-Lobatto Tensor Product Tests

#### 5.1 [x] `QuadrilateralQuadrature_GaussLobattoPolynomialExactness`
- **What it tests**: 2D Gauss-Lobatto tensor product achieves degree (2n-3) in each direction
- **Why needed**: Important for spectral element methods; currently sparse coverage
- **Implementation**: Systematic monomial integration test

#### 5.2 [x] `HexahedronQuadrature_GaussLobattoPolynomialExactness`
- **What it tests**: 3D Gauss-Lobatto tensor product achieves expected polynomial exactness
- **Why needed**: Essential for 3D spectral element methods
- **Implementation**: Same as 2D but with 3D monomials

#### 5.3 [x] `QuadrilateralQuadrature_GaussLobattoEndpoints`
- **What it tests**: All 4 corners of [-1,1]² are quadrature points
- **Why needed**: Endpoint property essential for SEM
- **Implementation**: Verify corner points exist in quadrature set

#### 5.4 [x] `WedgeQuadrature_GaussLobattoInLineDirection`
- **What it tests**: Wedge rule with Gauss-Lobatto in z-direction includes z=±1
- **Why needed**: Enables SEM on prismatic meshes
- **Implementation**: Check z-coordinates include endpoints

---

### 6. Extended Polynomial Exactness Tests

#### 6.1 [x] `GaussQuadrature1D_ExactnessUpToOrder20`
- **What it tests**: n=11 point rule (degree 21) integrates all monomials x^k, k≤21
- **Why needed**: Verify high-order implementation correctness
- **Implementation**: Extend existing sweep to higher orders

#### 6.2 [x] `TetrahedronQuadrature_ExactnessUpToOrder10`
- **What it tests**: High-order tetrahedral rule achieves degree 10 exactness
- **Why needed**: hp-FEM requires high-order integration
- **Implementation**: Full monomial sweep for total degree ≤ 10

#### 6.3 [x] `WedgeQuadrature_MixedDegreeExactness`
- **What it tests**: Wedge rule with different triangle/line orders achieves correct mixed exactness
- **Why needed**: Anisotropic refinement uses different orders per direction
- **Implementation**: Test all combinations (tri_order, line_order) for (2,6), (6,2), (4,4)

#### 6.4 [x] `PyramidQuadrature_HighOrderExactness`
- **What it tests**: Pyramid rules up to order 10 achieve polynomial exactness
- **Why needed**: Current tests only go to order 6
- **Implementation**: Extend polynomial sweep to higher orders

---

### 7. Surface Quadrature Comprehensive Tests

#### 7.1 [x] `SurfaceQuadrature_AllHexFacesPolynomialExactness`
- **What it tests**: Face rules on all 6 hex faces integrate polynomials exactly
- **Why needed**: Current tests only check weight sums, not exactness
- **Implementation**: For each face, integrate x^a y^b restricted to face

#### 7.2 [x] `SurfaceQuadrature_AllTetFacesPolynomialExactness`
- **What it tests**: Face rules on all 4 tet faces achieve correct exactness
- **Why needed**: Different tet faces have different orientations
- **Implementation**: Test each face index with polynomial integration

#### 7.3 [x] `SurfaceQuadrature_WedgeMixedFaces`
- **What it tests**: Triangular faces (0,1) and quad faces (2,3,4) return correct rule types
- **Why needed**: Wedge has mixed face topology
- **Implementation**: Verify face_rule returns triangle vs quad quadrature appropriately

#### 7.4 [x] `SurfaceQuadrature_PyramidMixedFaces`
- **What it tests**: Base quad face and 4 triangular faces return correct rules
- **Why needed**: Pyramid has mixed face topology
- **Implementation**: Test all 5 faces for correct type and weight sum

#### 7.5 [x] `SurfaceQuadrature_EdgeRulesAllElements`
- **What it tests**: Edge rules work for all element types that have edges
- **Why needed**: Current tests only check Line edges
- **Implementation**: Test edge rules for Triangle, Quad, Tet, Hex edges

---

### 8. Composite Quadrature Extended Tests

#### 8.1 [x] `CompositeQuadrature_PreservesPolynomialExactness`
- **What it tests**: Subdivided rule maintains polynomial exactness of base rule
- **Why needed**: Current tests only verify constant integration
- **Implementation**: Integrate x^k with composite rule, verify against base rule

#### 8.2 [x] `CompositeQuadrature_AnisotropicSubdivision`
- **What it tests**: Different subdivision counts in each direction work correctly
- **Why needed**: Adaptive refinement may use anisotropic subdivision
- **Implementation**: Test (2,4) and (4,2) subdivisions on quad

#### 8.3 [x] `CompositeQuadrature_TriangleSubdivision`
- **What it tests**: Triangle subdivision preserves integral (may need special handling)
- **Why needed**: Simplex subdivision is non-trivial
- **Implementation**: Verify weight sum after subdivision

#### 8.4 [x] `CompositeQuadrature_HighSubdivisionCount`
- **What it tests**: Large subdivision counts (8, 16) don't cause numerical issues
- **Why needed**: Fine subdivision may amplify round-off
- **Implementation**: Verify weight sums remain accurate

---

### 9. Adaptive Quadrature Extended Tests

#### 9.1 [x] `AdaptiveQuadrature_HighlyOscillatoryFunction`
- **What it tests**: Adaptive refinement handles sin(100x) type functions
- **Why needed**: Real applications have oscillatory integrands
- **Implementation**: Verify convergence or appropriate non-convergence flag

#### 9.2 [x] `AdaptiveQuadrature_DiscontinuousFunction`
- **What it tests**: Adaptive refinement behavior for step functions
- **Why needed**: Discontinuities require special treatment
- **Implementation**: Test Heaviside function, verify behavior documented

#### 9.3 [x] `AdaptiveQuadrature_ToleranceSensitivity`
- **What it tests**: Tighter tolerances produce more accurate results
- **Why needed**: Users need to understand tolerance-accuracy relationship
- **Implementation**: Compare results at tol=1e-4, 1e-8, 1e-12

#### 9.4 [x] `AdaptiveQuadrature_MaxLevelsRespected`
- **What it tests**: Refinement stops at max_levels even if not converged
- **Why needed**: Prevents infinite loops on non-convergent integrands
- **Implementation**: Use integrand that won't converge, verify levels_used ≤ max_levels

---

### 10. Singular Quadrature Extended Tests

#### 10.1 [x] `SingularQuadrature_CornerSingularity1OverR`
- **What it tests**: Duffy triangle correctly handles 1/r singularity at corner
- **Why needed**: BEM applications have 1/r kernels
- **Implementation**: Integrate 1/sqrt(x^2+y^2) near origin

#### 10.2 [x] `SingularQuadrature_LogarithmicSingularity`
- **What it tests**: Handling of log(r) type singularities
- **Why needed**: Common in 2D BEM
- **Implementation**: Integrate log(sqrt(x^2+y^2)) near origin

#### 10.3 [x] `SingularQuadrature_TetraVertexSingularity`
- **What it tests**: Duffy tetrahedron handles 1/r singularity at vertex
- **Why needed**: 3D BEM applications
- **Implementation**: Integrate 1/sqrt(x^2+y^2+z^2) near origin

#### 10.4 [x] `SingularQuadrature_ConvergenceWithOrder`
- **What it tests**: Higher-order Duffy rules give better accuracy for singular integrands
- **Why needed**: Users need guidance on order selection
- **Implementation**: Compare orders 4, 8, 12, 16 for same singular integrand

---

## Lower Priority Tests

### 11. Point Uniqueness and Symmetry Tests

#### 11.1 [x] `SymmetricTriangleQuadrature_NoDuplicatePoints`
- **What it tests**: No two quadrature points are identical
- **Why needed**: Duplicate points waste computation
- **Implementation**: Check pairwise distances > tolerance

#### 11.2 [x] `SymmetricTetrahedronQuadrature_NoDuplicatePoints`
- **What it tests**: Same as above for tetrahedra
- **Why needed**: Catches coefficient table errors
- **Implementation**: Same methodology

#### 11.3 [x] `SymmetricTriangleQuadrature_BarycentricSymmetry`
- **What it tests**: Points with equal weights form proper symmetric orbits
- **Why needed**: Verifies symmetric rule construction
- **Implementation**: Group points by weight, verify each group is symmetric

---

### 12. Cache Stress Tests

#### 12.1 [x] `QuadratureCache_HighConcurrency`
- **What it tests**: 32+ threads accessing cache simultaneously
- **Why needed**: Stress test thread safety
- **Implementation**: Spawn many threads, all requesting different rules

#### 12.2 [x] `QuadratureCache_PruningUnderLoad`
- **What it tests**: Pruning expired entries during concurrent access
- **Why needed**: Verify no race conditions during cleanup
- **Implementation**: Mix of access and prune operations

#### 12.3 [x] `QuadratureCache_MemoryBounds`
- **What it tests**: Cache size stays bounded with many unique rules
- **Why needed**: Prevent unbounded memory growth
- **Implementation**: Create many unique rules, verify cache size reasonable

---

### 13. Reference Element Convention Tests

#### 13.1 [x] `TriangleQuadrature_ReferenceElementConvention`
- **What it tests**: Points lie in documented reference triangle (0,0)-(1,0)-(0,1)
- **Why needed**: Ensures consistency with Basis library
- **Implementation**: Verify all points satisfy 0≤x, 0≤y, x+y≤1

#### 13.2 [x] `TetrahedronQuadrature_ReferenceElementConvention`
- **What it tests**: Points lie in documented reference tetrahedron
- **Why needed**: Convention consistency
- **Implementation**: Verify barycentric coordinates valid

#### 13.3 [x] `QuadrilateralQuadrature_ReferenceElementConvention`
- **What it tests**: Points lie in [-1,1]×[-1,1]
- **Why needed**: Standard tensor product convention
- **Implementation**: Verify all |x|,|y| ≤ 1

#### 13.4 [x] `PyramidQuadrature_ReferenceElementConvention`
- **What it tests**: Points satisfy pyramid geometry constraints
- **Why needed**: Pyramid reference element is non-standard
- **Implementation**: Verify z∈[0,1], |x|,|y| ≤ 1-z

---

### 14. Error Handling Tests

#### 14.1 [x] `QuadratureFactory_AllInvalidOrdersThrow`
- **What it tests**: Orders 0, -1, MAX_INT all throw appropriate exceptions
- **Why needed**: Clear error messages for users
- **Implementation**: Test boundary values

#### 14.2 [x] `QuadratureFactory_UnsupportedElementTypeMessages`
- **What it tests**: Error messages identify unsupported element type
- **Why needed**: Debugging support
- **Implementation**: Catch exception, verify message content

#### 14.3 [x] `PositionBasedQuadrature_BoundaryModifiers`
- **What it tests**: Behavior at exact boundary values (s=1/3, s=1 for TRI; s=0.25, s=1 for TET)
- **Why needed**: Edge cases may have special handling
- **Implementation**: Test exact boundary values work correctly

---

### 15. Performance Regression Tests

#### 15.1 [x] `QuadraturePointCounts_MatchExpected`
- **What it tests**: Point counts haven't regressed (increased unnecessarily)
- **Why needed**: Performance regression detection
- **Implementation**: Assert exact point counts for canonical rules

#### 15.2 [x] `SymmetricRules_FewerPointsThanTensorProduct`
- **What it tests**: Symmetric rules always use fewer points than equivalent tensor product
- **Why needed**: Main benefit of symmetric rules is efficiency
- **Implementation**: Compare point counts systematically

---

## Checklist Summary

| Category | Tests | Priority |
|----------|-------|----------|
| Fix High-Order Symmetric Triangle | 3 | High |
| Literature Value Comparison | 5 | High |
| Convergence Rate Tests | 4 | High |
| Numerical Stability | 4 | High |
| Gauss-Lobatto Tensor Products | 4 | Medium |
| Extended Polynomial Exactness | 4 | Medium |
| Surface Quadrature Comprehensive | 5 | Medium |
| Composite Quadrature Extended | 4 | Medium |
| Adaptive Quadrature Extended | 4 | Medium |
| Singular Quadrature Extended | 4 | Medium |
| Point Uniqueness/Symmetry | 3 | Lower |
| Cache Stress Tests | 3 | Lower |
| Reference Element Convention | 4 | Lower |
| Error Handling | 3 | Lower |
| Performance Regression | 2 | Lower |
| **Total** | **56** | |

---

## References

1. Abramowitz, M. and Stegun, I.A. "Handbook of Mathematical Functions", Table 25.4 (Gauss-Legendre)
2. NIST Digital Library of Mathematical Functions, Section 3.5 (Quadrature)
3. Dunavant, D.A. "High Degree Efficient Symmetrical Gaussian Quadrature Rules for the Triangle", IJNME Vol 21, 1985, pp 1129-1148
4. Wandzura, S. and Xiao, H. "Symmetric Quadrature Rules on a Triangle", Computers & Mathematics with Applications, 2003
5. Keast, P. "Moderate Degree Tetrahedral Quadrature Formulas", CMAME Vol 55, 1986, pp 339-348
6. Golub, G.H. and Welsch, J.H. "Calculation of Gauss Quadrature Rules", Mathematics of Computation, 1969
7. Stroud, A.H. "Approximate Calculation of Multiple Integrals", Prentice-Hall, 1971
8. Cools, R. "An Encyclopedia of Cubature Formulas", Journal of Complexity, 2003

---

## Known Issues to Address

### Implementation Bugs (from disabled tests)

1. **SymmetricTriangleQuadrature Orders 10-20**: The `DISABLED_HighOrderPolynomialExactness` test indicates coefficient table errors for orders 10-20. These should be verified against Dunavant's original paper and Wandzura-Xiao's high-precision tables.

2. **Exterior Points**: Orders 11, 15, 16, 18, 20 have documented points outside the reference triangle. This is intentional for accuracy but may cause issues for some users. Consider adding validation mode that warns about exterior points.

### Documentation Gaps

1. Negative weights in symmetric rules should be documented in code comments
2. Reference element conventions should be explicitly stated in header files
3. Polynomial exactness guarantees should be documented per rule type

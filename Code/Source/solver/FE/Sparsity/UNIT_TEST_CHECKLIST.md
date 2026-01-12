# Sparsity Module Unit Test Checklist

This document lists additional unit tests that should be added to the Sparsity module to ensure comprehensive coverage, validation against established finite element literature, and verification of correctness for production use.

---

## Table of Contents

1. [Finite Element Literature Validation Tests](#1-finite-element-literature-validation-tests)
2. [Large-Scale and Stress Tests](#2-large-scale-and-stress-tests)
3. [Backend Integration Tests](#3-backend-integration-tests)
4. [SparsityPattern Edge Cases](#4-sparsitypattern-edge-cases)
5. [SparsityFormat Conversion Tests](#5-sparsityformat-conversion-tests)
6. [GraphSparsity Algorithm Validation](#6-graphsparsity-algorithm-validation)
7. [SparsityBuilder Advanced Scenarios](#7-sparsitybuilder-advanced-scenarios)
8. [DGSparsityBuilder Comprehensive Tests](#8-dgsparsitybuilder-comprehensive-tests)
9. [DistributedSparsityPattern MPI Tests](#9-distributedsparsitypattern-mpi-tests)
10. [BlockSparsity Advanced Tests](#10-blocksparsity-advanced-tests)
11. [SparsityOptimizer Effectiveness Tests](#11-sparsityoptimizer-effectiveness-tests)
12. [SparsityAnalyzer Accuracy Tests](#12-sparsityanalyzer-accuracy-tests)
13. [AdaptiveSparsity Complex Scenarios](#13-adaptivesparsity-complex-scenarios)
14. [ConstraintSparsityAugmenter Tests](#14-constraintsparsityaugmenter-tests)
15. [Numerical Reproducibility Tests](#15-numerical-reproducibility-tests)
16. [Performance Regression Tests](#16-performance-regression-tests)

---

## 1. Finite Element Literature Validation Tests

These tests validate sparsity patterns against well-known finite element discretizations from academic literature.

### 1.1 Poisson Equation Patterns

- [ ] **`TEST(LiteratureValidation, Poisson2D_Q1_UnitSquare_4x4)`**
  - **Reason**: Q1 (bilinear) elements on a structured 4x4 mesh is a canonical FE example
  - **Testing**: Pattern matches hand-computed 25-DOF pattern with 9-point stencil interior, reduced boundary stencils
  - **Reference**: Hughes, "The Finite Element Method" Chapter 1

- [ ] **`TEST(LiteratureValidation, Poisson2D_Q1_UnitSquare_NxN)`**
  - **Reason**: Verify pattern scaling for N = 8, 16, 32 matches theoretical NNZ = O(9n) for interior nodes
  - **Testing**: Total NNZ count, bandwidth = 2N+1 for row-major ordering

- [ ] **`TEST(LiteratureValidation, Poisson2D_P1_Triangle_Structured)`**
  - **Reason**: P1 triangular elements produce different patterns than Q1 quads
  - **Testing**: 7-point stencil for interior nodes, proper vertex-to-vertex connectivity

- [ ] **`TEST(LiteratureValidation, Poisson3D_Q1_UnitCube_4x4x4)`**
  - **Reason**: 3D hex elements have 27-point stencils, critical for solver performance
  - **Testing**: 125-DOF pattern, bandwidth verification, corner/edge/face/interior stencil sizes

- [ ] **`TEST(LiteratureValidation, Poisson3D_P1_Tetrahedra)`**
  - **Reason**: Tetrahedral meshes are common in unstructured 3D; pattern differs from hex
  - **Testing**: Vertex connectivity produces variable stencil sizes

### 1.2 Elasticity Patterns

- [ ] **`TEST(LiteratureValidation, Elasticity2D_Q1_VectorField)`**
  - **Reason**: Vector-valued problems (2 DOFs/node) have block structure within scalar pattern
  - **Testing**: 2x2 block coupling within each node pair, total NNZ = 4x scalar Poisson

- [ ] **`TEST(LiteratureValidation, Elasticity3D_Q1_VectorField)`**
  - **Reason**: 3 DOFs/node produces 3x3 blocks; pattern structure critical for block solvers
  - **Testing**: Block diagonal dominance pattern, 9x NNZ of scalar Laplacian

- [ ] **`TEST(LiteratureValidation, Elasticity2D_Q2_HigherOrder)`**
  - **Reason**: Q2 (biquadratic) elements have 9 nodes/element, larger stencils
  - **Testing**: Verify mid-edge and center node connectivity patterns

### 1.3 Stokes/Incompressible Flow Patterns

- [ ] **`TEST(LiteratureValidation, Stokes2D_Q2Q1_TaylorHood)`**
  - **Reason**: Taylor-Hood (Q2 velocity, Q1 pressure) is the standard stable pair
  - **Testing**: Saddle-point 2x2 block structure, B and B^T blocks, zero (1,1) pressure block
  - **Reference**: Elman, Silvester, Wathen, "Finite Elements and Fast Iterative Solvers"

- [ ] **`TEST(LiteratureValidation, Stokes2D_Q2Q1_BlockStructure)`**
  - **Reason**: Verify correct velocity-velocity, velocity-pressure, pressure-velocity couplings
  - **Testing**: BlockSparsity with proper field dimensions and coupling pattern

- [ ] **`TEST(LiteratureValidation, Stokes3D_Q2Q1_TaylorHood)`**
  - **Reason**: 3D Stokes is computationally intensive; pattern correctness critical
  - **Testing**: 4x4 block structure (u,v,w,p), proper incompressibility constraint pattern

### 1.4 Mixed Formulation Patterns

- [ ] **`TEST(LiteratureValidation, MixedPoisson_RT0_P0)`**
  - **Reason**: Raviart-Thomas elements for mixed Poisson have face-based velocity DOFs
  - **Testing**: Edge/face DOF to cell DOF connectivity, rectangular blocks

- [ ] **`TEST(LiteratureValidation, DarcyFlow_BDM1_P0)`**
  - **Reason**: BDM elements are another standard mixed element pair
  - **Testing**: Verify element-face connectivity produces correct pattern

### 1.5 Matrix Market Reference Patterns

- [ ] **`TEST(LiteratureValidation, MatrixMarket_bcsstk01)`**
  - **Reason**: bcsstk01 is a standard small structural mechanics matrix (48x48)
  - **Testing**: Load Matrix Market file, verify pattern matches exactly

- [ ] **`TEST(LiteratureValidation, MatrixMarket_nos1)`**
  - **Reason**: nos1 is a standard thermal problem pattern
  - **Testing**: Load and verify structural properties match published values

- [ ] **`TEST(LiteratureValidation, MatrixMarket_StructuralSymmetry)`**
  - **Reason**: Many benchmark matrices have known symmetry properties
  - **Testing**: Verify symmetry detection matches published matrix properties

---

## 2. Large-Scale and Stress Tests

These tests verify the module handles production-scale problems correctly.

### 2.1 Memory and Scalability

- [ ] **`TEST(LargeScale, SparsityPattern_1MillionDOFs)`**
  - **Reason**: Production problems often exceed 1M DOFs; must verify no memory issues
  - **Testing**: Construct pattern for 100x100x100 hex mesh (1M DOFs), verify NNZ count

- [ ] **`TEST(LargeScale, SparsityPattern_10MillionDOFs)`**
  - **Reason**: Large-scale simulations may exceed 10M DOFs
  - **Testing**: Verify construction completes without memory explosion, check statistics

- [ ] **`TEST(LargeScale, TwoPassBuilder_Threshold)`**
  - **Reason**: TwoPassBuilder claims efficiency above certain thresholds; verify claim
  - **Testing**: Compare memory usage of standard vs two-pass for 100k, 500k, 1M DOFs

- [ ] **`TEST(LargeScale, TwoPassBuilder_MemoryEfficiency)`**
  - **Reason**: Two-pass should use less peak memory than set-of-sets approach
  - **Testing**: Measure peak RSS during construction, compare approaches

- [ ] **`TEST(LargeScale, CompressedSparsity_LargeMesh)`**
  - **Reason**: Compressed construction should handle large problems efficiently
  - **Testing**: 500k DOF pattern with compressed mode, verify final pattern correct

### 2.2 Construction Time

- [ ] **`TEST(LargeScale, ConstructionTime_LinearScaling)`**
  - **Reason**: Pattern construction should scale linearly with NNZ
  - **Testing**: Time construction for 10k, 100k, 1M entries; verify O(n) scaling

- [ ] **`TEST(LargeScale, Finalization_LinearTime)`**
  - **Reason**: CSR compression during finalization should be O(NNZ)
  - **Testing**: Time finalization for various sizes, verify linear relationship

### 2.3 Extreme Dimensions

- [ ] **`TEST(LargeScale, VeryTallRectangular_1Mx100)`**
  - **Reason**: Some operators (e.g., restriction) are very tall and narrow
  - **Testing**: 1M rows x 100 columns pattern, verify correct handling

- [ ] **`TEST(LargeScale, VeryWideRectangular_100x1M)`**
  - **Reason**: Prolongation operators can be very wide
  - **Testing**: 100 rows x 1M columns, verify memory and correctness

---

## 3. Backend Integration Tests

These tests verify the module integrates correctly with external sparse matrix libraries.

### 3.1 PETSc Integration

- [ ] **`TEST(BackendIntegration, PETSc_SeqAIJPreallocation)`**
  - **Reason**: Incorrect preallocation causes PETSc performance warnings or errors
  - **Testing**: Create pattern, get preallocation, create MatSeqAIJ, assemble without mallocs

- [ ] **`TEST(BackendIntegration, PETSc_MPIAIJPreallocation)`**
  - **Reason**: Distributed matrices need d_nnz and o_nnz arrays
  - **Testing**: Create distributed pattern, verify MPIAIJ creation has zero new mallocs

- [ ] **`TEST(BackendIntegration, PETSc_PreallocationExact)`**
  - **Reason**: Over/under-allocation wastes memory or causes errors
  - **Testing**: Assembly with exactly the predicted entries succeeds; extra entry fails

- [ ] **`TEST(BackendIntegration, PETSc_BlockAIJPreallocation)`**
  - **Reason**: Block matrices need block-aware preallocation
  - **Testing**: BSR pattern produces correct block preallocation for MatSeqBAIJ

### 3.2 Trilinos Integration

- [ ] **`TEST(BackendIntegration, Trilinos_EpetraCrsGraph)`**
  - **Reason**: Epetra uses different row pointer semantics
  - **Testing**: Pattern produces valid Epetra_CrsGraph construction

- [ ] **`TEST(BackendIntegration, Trilinos_TpetraCrsGraph)`**
  - **Reason**: Tpetra is the modern Trilinos interface
  - **Testing**: Pattern data compatible with Tpetra::CrsGraph construction

### 3.3 Eigen Integration

- [ ] **`TEST(BackendIntegration, Eigen_SparseMatrixReserve)`**
  - **Reason**: Eigen uses different reservation strategy
  - **Testing**: Pattern data produces efficient Eigen::SparseMatrix construction

- [ ] **`TEST(BackendIntegration, Eigen_ColMajorCSC)`**
  - **Reason**: Eigen defaults to column-major; CSC format preferred
  - **Testing**: CSC conversion produces correct Eigen column-major data

---

## 4. SparsityPattern Edge Cases

### 4.1 Boundary Conditions

- [ ] **`TEST(SparsityPatternEdge, EmptyPattern)`**
  - **Reason**: Empty patterns should be valid but have zero entries
  - **Testing**: 0x0 pattern, finalization works, all queries return empty

- [ ] **`TEST(SparsityPatternEdge, SingleEntry)`**
  - **Reason**: Minimal non-trivial pattern
  - **Testing**: 1x1 pattern with single diagonal entry, all operations work

- [ ] **`TEST(SparsityPatternEdge, DiagonalOnly_Large)`**
  - **Reason**: Diagonal matrices are special case for many algorithms
  - **Testing**: 10000x10000 diagonal pattern, bandwidth=1, all operations correct

- [ ] **`TEST(SparsityPatternEdge, FullyDense_Small)`**
  - **Reason**: Dense pattern is opposite extreme; may trigger different code paths
  - **Testing**: 100x100 fully dense pattern, NNZ=10000, operations correct

- [ ] **`TEST(SparsityPatternEdge, SingleRowDense)`**
  - **Reason**: One dense row (e.g., Lagrange multiplier) affects bandwidth significantly
  - **Testing**: Pattern with one row having all columns filled

- [ ] **`TEST(SparsityPatternEdge, SingleColumnDense)`**
  - **Reason**: One dense column occurs in constraint problems
  - **Testing**: Pattern with one column having all rows filled

### 4.2 Index Handling

- [ ] **`TEST(SparsityPatternEdge, MaxIntIndices)`**
  - **Reason**: Near-limit indices may cause overflow issues
  - **Testing**: Pattern with indices near INT_MAX/2, operations don't overflow

- [ ] **`TEST(SparsityPatternEdge, ZeroBasedVsOneBased)`**
  - **Reason**: Fortran interop requires 1-based indexing
  - **Testing**: Convert pattern to 1-based, verify all indices shifted correctly

### 4.3 Duplicate Handling

- [ ] **`TEST(SparsityPatternEdge, MassiveDuplicates)`**
  - **Reason**: Some assembly patterns add same entry many times
  - **Testing**: Add same (i,j) 1000 times, final pattern has single entry

- [ ] **`TEST(SparsityPatternEdge, DuplicatesAcrossRows)`**
  - **Reason**: Duplicates may occur across different rows for same column
  - **Testing**: Multiple rows reference same column, each row correct

---

## 5. SparsityFormat Conversion Tests

### 5.1 Edge Case Conversions

- [ ] **`TEST(SparsityFormatEdge, CSRtoCOO_Empty)`**
  - **Reason**: Empty pattern conversion should produce empty COO
  - **Testing**: Empty CSR converts to zero-length COO arrays

- [ ] **`TEST(SparsityFormatEdge, CSRtoCOO_SingleEntry)`**
  - **Reason**: Minimal conversion case
  - **Testing**: Single entry produces single (row, col) pair

- [ ] **`TEST(SparsityFormatEdge, CSRtoCSC_Rectangular)`**
  - **Reason**: Rectangular CSR→CSC is more complex than square
  - **Testing**: 100x200 pattern converts correctly

- [ ] **`TEST(SparsityFormatEdge, CSRtoBSR_NonDivisible)`**
  - **Reason**: BSR requires dimensions divisible by block size
  - **Testing**: Verify error or padding for 99x99 with block_size=4

- [ ] **`TEST(SparsityFormatEdge, CSRtoBSR_PartialBlocks)`**
  - **Reason**: Sparse blocks (not all entries filled) handled correctly
  - **Testing**: Pattern where some blocks are partial, BSR still correct

- [ ] **`TEST(SparsityFormatEdge, ELLPACKPadding_HighVariance)`**
  - **Reason**: High row-length variance wastes ELLPACK memory
  - **Testing**: Pattern with rows of length 1 to 100, verify padding and efficiency metric

- [ ] **`TEST(SparsityFormatEdge, ELLPACK_VeryLongRow)`**
  - **Reason**: Single very long row forces excessive padding
  - **Testing**: One row with 1000 entries, others with 5, verify inefficiency detected

### 5.2 Round-Trip Accuracy

- [ ] **`TEST(SparsityFormatRoundTrip, CSR_COO_CSR_LargePattern)`**
  - **Reason**: Round-trip must preserve pattern exactly for large matrices
  - **Testing**: 100k-entry pattern survives CSR→COO→CSR exactly

- [ ] **`TEST(SparsityFormatRoundTrip, CSR_BSR_CSR_ExactRecovery)`**
  - **Reason**: BSR may add fill for partial blocks; CSR recovery should be exact
  - **Testing**: Original pattern recovered after CSR→BSR→CSR

- [ ] **`TEST(SparsityFormatRoundTrip, AllFormats_RandomPattern)`**
  - **Reason**: Comprehensive format interoperability
  - **Testing**: Random pattern through all format conversions, verify consistency

---

## 6. GraphSparsity Algorithm Validation

### 6.1 Coloring Correctness

- [ ] **`TEST(GraphSparsityColoring, OptimalColoring_PathGraph)`**
  - **Reason**: Path graph (tridiagonal) needs exactly 2 colors
  - **Testing**: Coloring produces 2 colors, verify validity

- [ ] **`TEST(GraphSparsityColoring, OptimalColoring_CycleGraph)`**
  - **Reason**: Even cycle needs 2 colors, odd cycle needs 3
  - **Testing**: Verify coloring matches theoretical chromatic number

- [ ] **`TEST(GraphSparsityColoring, OptimalColoring_CompleteGraph)`**
  - **Reason**: Complete graph K_n needs exactly n colors
  - **Testing**: K_5 pattern needs 5 colors

- [ ] **`TEST(GraphSparsityColoring, OptimalColoring_BipartiteGraph)`**
  - **Reason**: Bipartite graphs are 2-colorable
  - **Testing**: Bipartite pattern from rectangular operator, verify 2 colors

- [ ] **`TEST(GraphSparsityColoring, Coloring_StarGraph)`**
  - **Reason**: Star graph has chromatic number 2 (center + leaves)
  - **Testing**: Pattern with one dense row/column, verify 2 colors

### 6.2 Reordering Quality

- [ ] **`TEST(GraphSparsityReorder, RCM_OptimalBandwidth_Tridiagonal)`**
  - **Reason**: RCM should not increase bandwidth of already-optimal pattern
  - **Testing**: Tridiagonal pattern, RCM bandwidth = original bandwidth

- [ ] **`TEST(GraphSparsityReorder, RCM_KnownOptimal_ArrowMatrix)`**
  - **Reason**: Arrow matrix has known optimal ordering
  - **Testing**: RCM produces bandwidth close to optimal for arrowhead pattern

- [ ] **`TEST(GraphSparsityReorder, RCM_StartingVertexImpact)`**
  - **Reason**: RCM quality depends on starting peripheral vertex selection
  - **Testing**: Compare RCM results from different starting vertices

- [ ] **`TEST(GraphSparsityReorder, AMD_CompareWithMATLAB)`**
  - **Reason**: AMD has reference implementations in MATLAB/SuiteSparse
  - **Testing**: Pattern produces same ordering as MATLAB's symamd for known matrix

- [ ] **`TEST(GraphSparsityReorder, NestedDissection_Separator)`**
  - **Reason**: ND should find good separators in structured grids
  - **Testing**: 2D grid pattern, verify separator quality

- [ ] **`TEST(GraphSparsityReorder, Comparison_AllAlgorithms)`**
  - **Reason**: Different algorithms optimal for different patterns
  - **Testing**: Compare CM, RCM, AMD, ND on same pattern, verify all valid

### 6.3 Fill-in Prediction Accuracy

- [ ] **`TEST(GraphSparsityFillIn, Cholesky_Tridiagonal_Exact)`**
  - **Reason**: Tridiagonal Cholesky has zero fill-in; must predict exactly
  - **Testing**: Prediction = 0 for tridiagonal SPD pattern

- [ ] **`TEST(GraphSparsityFillIn, Cholesky_ArrowMatrix_Known)`**
  - **Reason**: Arrow matrix fill-in is analytically computable
  - **Testing**: Prediction matches theoretical (n-1)(n-2)/2 fill-in

- [ ] **`TEST(GraphSparsityFillIn, Cholesky_Grid2D_Comparison)`**
  - **Reason**: 2D grid fill-in well-studied in literature
  - **Testing**: Prediction within 10% of actual Cholesky fill-in

- [ ] **`TEST(GraphSparsityFillIn, LU_WithPivoting_Estimate)`**
  - **Reason**: LU with pivoting has higher fill-in than Cholesky
  - **Testing**: LU prediction >= Cholesky prediction for same pattern

- [ ] **`TEST(GraphSparsityFillIn, ActualFactorization_Comparison)`**
  - **Reason**: Predictions only useful if accurate
  - **Testing**: Perform actual symbolic Cholesky, compare to prediction

### 6.4 Connectivity Analysis

- [ ] **`TEST(GraphSparsityConnectivity, DisconnectedComponents_Correct)`**
  - **Reason**: Disconnected patterns need special solver treatment
  - **Testing**: Block diagonal pattern, detect correct number of components

- [ ] **`TEST(GraphSparsityConnectivity, SingletonComponents)`**
  - **Reason**: Isolated DOFs (uncoupled) are common in constrained problems
  - **Testing**: Pattern with isolated diagonal entries, each is separate component

- [ ] **`TEST(GraphSparsityConnectivity, LevelSets_BFS)`**
  - **Reason**: Level sets used for parallel decomposition
  - **Testing**: Known graph, verify BFS levels match expected

- [ ] **`TEST(GraphSparsityConnectivity, Diameter_PathGraph)`**
  - **Reason**: Path graph diameter = n-1
  - **Testing**: Tridiagonal pattern, diameter = n-1

---

## 7. SparsityBuilder Advanced Scenarios

### 7.1 Complex Element Types

- [ ] **`TEST(SparsityBuilderAdvanced, QuadraticTriangle_P2)`**
  - **Reason**: P2 triangles have 6 nodes with complex connectivity
  - **Testing**: Pattern reflects 6-node element with edge midpoints

- [ ] **`TEST(SparsityBuilderAdvanced, CubicHex_Q3)`**
  - **Reason**: High-order hexes have 64 nodes per element
  - **Testing**: Single Q3 hex produces 64x64 dense element block

- [ ] **`TEST(SparsityBuilderAdvanced, Serendipity_Q2_20Node)`**
  - **Reason**: 20-node serendipity hex is common in structural analysis
  - **Testing**: Correct 20-node connectivity pattern

- [ ] **`TEST(SparsityBuilderAdvanced, MixedElementMesh)`**
  - **Reason**: Real meshes may have hex/tet/prism/pyramid mixed
  - **Testing**: Mesh with multiple element types, unified pattern

### 7.2 Multi-Field Coupling

- [ ] **`TEST(SparsityBuilderAdvanced, ThreeFieldCoupling)`**
  - **Reason**: Thermo-mechanical problems have 3+ fields
  - **Testing**: Temperature + displacement (3D) = 4 DOFs/node coupling

- [ ] **`TEST(SparsityBuilderAdvanced, SelectiveCoupling)`**
  - **Reason**: Not all fields couple to all others (e.g., pressure-pressure = 0)
  - **Testing**: Custom CouplingMode with some blocks empty

- [ ] **`TEST(SparsityBuilderAdvanced, DifferentFieldOrders)`**
  - **Reason**: Mixed elements have different orders for different fields
  - **Testing**: Q2-Q1 velocity-pressure produces different DOF counts

- [ ] **`TEST(SparsityBuilderAdvanced, FieldSubsetAssembly)`**
  - **Reason**: May assemble only subset of fields for operator splitting
  - **Testing**: Build pattern for velocity-only from full Stokes mesh

### 7.3 Real Mesh Data

- [ ] **`TEST(SparsityBuilderAdvanced, ImportedMesh_GMSH)`**
  - **Reason**: Real mesh data may have different numbering conventions
  - **Testing**: Load GMSH mesh file, build pattern, verify consistency

- [ ] **`TEST(SparsityBuilderAdvanced, ImportedMesh_ExodusII)`**
  - **Reason**: ExodusII is common in engineering
  - **Testing**: Load ExodusII mesh, build pattern, verify NNZ

- [ ] **`TEST(SparsityBuilderAdvanced, UnstructuredMesh_3D)`**
  - **Reason**: Unstructured tet mesh has irregular connectivity
  - **Testing**: Pattern from unstructured tet mesh, variable row lengths

---

## 8. DGSparsityBuilder Comprehensive Tests

### 8.1 Face Coupling

- [ ] **`TEST(DGSparsityAdvanced, InteriorFaces_AllCoupled)`**
  - **Reason**: DG interior faces couple two adjacent elements
  - **Testing**: 2-element mesh, face coupling produces correct off-diagonal blocks

- [ ] **`TEST(DGSparsityAdvanced, BoundaryFaces_SelfCoupling)`**
  - **Reason**: Boundary faces only couple element to itself
  - **Testing**: Boundary element pattern is element-local only

- [ ] **`TEST(DGSparsityAdvanced, PeriodicBoundary_Coupling)`**
  - **Reason**: Periodic BCs couple distant elements
  - **Testing**: Periodic mesh, pattern has non-local couplings

- [ ] **`TEST(DGSparsityAdvanced, MortarInterface)`**
  - **Reason**: Non-conforming interfaces need mortar coupling
  - **Testing**: Mortar interface produces rectangular coupling blocks

### 8.2 Stencil Types

- [ ] **`TEST(DGSparsityAdvanced, CompactStencil_Verification)`**
  - **Reason**: Compact stencil only couples face neighbors
  - **Testing**: Element couples to immediate neighbors only

- [ ] **`TEST(DGSparsityAdvanced, ExtendedStencil_Reconstruction)`**
  - **Reason**: High-order reconstruction needs extended stencil
  - **Testing**: Element couples to neighbors-of-neighbors

- [ ] **`TEST(DGSparsityAdvanced, StencilSize_3D)`**
  - **Reason**: 3D stencils larger than 2D
  - **Testing**: Hex element with 6 faces, verify neighbor count

### 8.3 Hybrid CG-DG

- [ ] **`TEST(DGSparsityAdvanced, HybridCGDG_Interface)`**
  - **Reason**: CG and DG regions meet at interface
  - **Testing**: Pattern correctly couples CG shared DOFs to DG element DOFs

- [ ] **`TEST(DGSparsityAdvanced, HybridCGDG_PatternMerge)`**
  - **Reason**: Final pattern must merge CG and DG contributions
  - **Testing**: Merged pattern contains both CG vertex and DG element couplings

### 8.4 DG Formulation Variants

- [ ] **`TEST(DGSparsityAdvanced, IPDG_PenaltyTerms)`**
  - **Reason**: Interior Penalty DG has specific coupling pattern
  - **Testing**: Penalty terms contribute to pattern correctly

- [ ] **`TEST(DGSparsityAdvanced, LDG_StaggeredCoupling)`**
  - **Reason**: Local DG has different coupling for different variables
  - **Testing**: LDG pattern reflects directional coupling

- [ ] **`TEST(DGSparsityAdvanced, HDG_TraceVariables)`**
  - **Reason**: Hybridizable DG has trace unknowns on faces
  - **Testing**: Pattern includes face trace DOFs

---

## 9. DistributedSparsityPattern MPI Tests

### 9.1 Multi-Rank Consistency

- [ ] **`TEST(DistributedSparsityMPI, TwoRank_Consistency)`**
  - **Reason**: Minimum distributed case; patterns must be consistent
  - **Testing**: 2-rank pattern, global pattern equals union of local patterns

- [ ] **`TEST(DistributedSparsityMPI, FourRank_SymmetricPartition)`**
  - **Reason**: Power-of-2 ranks common for structured grids
  - **Testing**: 2x2 partition, verify diag/offdiag split correct

- [ ] **`TEST(DistributedSparsityMPI, OddRank_LoadImbalance)`**
  - **Reason**: Odd rank counts cause load imbalance
  - **Testing**: 3-rank pattern, verify all rows covered exactly once

- [ ] **`TEST(DistributedSparsityMPI, GlobalPatternReconstruction)`**
  - **Reason**: Must be able to reconstruct global pattern from distributed
  - **Testing**: Gather all local patterns, union equals global

### 9.2 Ghost Column Handling

- [ ] **`TEST(DistributedSparsityMPI, GhostColumns_Identification)`**
  - **Reason**: Ghost columns are off-process column indices
  - **Testing**: Verify ghost column set matches actual non-local columns

- [ ] **`TEST(DistributedSparsityMPI, GhostColumns_Ordering)`**
  - **Reason**: Ghost columns may need specific ordering for backends
  - **Testing**: Ghost column map produces contiguous local indices

- [ ] **`TEST(DistributedSparsityMPI, GhostColumns_LargeHalo)`**
  - **Reason**: Some partitions have many ghost columns
  - **Testing**: Pattern with 50% ghost columns, all handled correctly

### 9.3 Preallocation Arrays

- [ ] **`TEST(DistributedSparsityMPI, Preallocation_DiagOffdiag_Correct)`**
  - **Reason**: d_nnz and o_nnz arrays must sum to total row NNZ
  - **Testing**: For each row, d_nnz[i] + o_nnz[i] = total nnz in row i

- [ ] **`TEST(DistributedSparsityMPI, Preallocation_NoOvercount)`**
  - **Reason**: Overallocation wastes memory
  - **Testing**: Preallocation equals exactly what's needed, not more

- [ ] **`TEST(DistributedSparsityMPI, Preallocation_CorrectPartition)`**
  - **Reason**: Diagonal/off-diagonal split must match ownership ranges
  - **Testing**: Verify column classification matches row ownership

### 9.4 Communication Patterns

- [ ] **`TEST(DistributedSparsityMPI, CommunicationVolume)`**
  - **Reason**: Communication volume affects parallel efficiency
  - **Testing**: Compute and report off-diagonal NNZ as communication metric

- [ ] **`TEST(DistributedSparsityMPI, Partitioning_Quality)`**
  - **Reason**: Good partitioning minimizes communication
  - **Testing**: Compare communication volume for different partitionings

---

## 10. BlockSparsity Advanced Tests

### 10.1 Non-Uniform Blocks

- [ ] **`TEST(BlockSparsityAdvanced, NonUniformBlockSizes)`**
  - **Reason**: Mixed elements may have different DOF counts per block
  - **Testing**: Blocks of sizes 2, 3, 4 in same BlockSparsity

- [ ] **`TEST(BlockSparsityAdvanced, SingleDOFBlock)`**
  - **Reason**: Pressure in Stokes is often 1 DOF/node
  - **Testing**: 1x1 blocks alongside 3x3 velocity blocks

- [ ] **`TEST(BlockSparsityAdvanced, EmptyBlocks)`**
  - **Reason**: Some field pairs don't couple (e.g., pressure-pressure in Stokes)
  - **Testing**: Zero pattern in specific block, monolithic correct

### 10.2 Block Operations

- [ ] **`TEST(BlockSparsityAdvanced, SchurComplement_Pattern)`**
  - **Reason**: Schur complement preconditioning needs S = C - B A^{-1} B^T pattern
  - **Testing**: Computed Schur pattern matches theoretical structure

- [ ] **`TEST(BlockSparsityAdvanced, SchurComplement_LargeScale)`**
  - **Reason**: Schur extraction must scale
  - **Testing**: Schur complement pattern for 100k DOF saddle-point system

- [ ] **`TEST(BlockSparsityAdvanced, BlockDiagonal_Extraction)`**
  - **Reason**: Block Jacobi needs diagonal block patterns
  - **Testing**: Extract diagonal blocks, each is correct subpattern

- [ ] **`TEST(BlockSparsityAdvanced, BlockTriangular_Detection)`**
  - **Reason**: Some problems are block triangular (can use forward solve)
  - **Testing**: Detect lower/upper block triangular structure

### 10.3 Monolithic Conversion

- [ ] **`TEST(BlockSparsityAdvanced, MonolithicRoundTrip)`**
  - **Reason**: Block ↔ monolithic conversions must be lossless
  - **Testing**: Block → monolithic → block preserves structure

- [ ] **`TEST(BlockSparsityAdvanced, MonolithicOrdering_FieldMajor)`**
  - **Reason**: Different orderings (field-major vs point-major) for monolithic
  - **Testing**: Field-major ordering produces expected pattern

- [ ] **`TEST(BlockSparsityAdvanced, MonolithicOrdering_PointMajor)`**
  - **Reason**: Point-major interleaves field DOFs
  - **Testing**: Point-major ordering produces expected pattern

---

## 11. SparsityOptimizer Effectiveness Tests

### 11.1 Bandwidth Reduction Verification

- [ ] **`TEST(SparsityOptimizerEffective, RCM_ActualReduction)`**
  - **Reason**: RCM should reduce bandwidth on most patterns
  - **Testing**: Random FE pattern, verify RCM bandwidth < original

- [ ] **`TEST(SparsityOptimizerEffective, RCM_GuaranteedImprovement)`**
  - **Reason**: For some patterns, RCM reduction is provable
  - **Testing**: Known-suboptimal ordering, verify improvement

- [ ] **`TEST(SparsityOptimizerEffective, AMD_FillInReduction)`**
  - **Reason**: AMD should reduce fill-in compared to natural ordering
  - **Testing**: Perform Cholesky with AMD ordering, verify less fill

### 11.2 Algorithm Quality Comparison

- [ ] **`TEST(SparsityOptimizerEffective, CompareAlgorithms_Grid2D)`**
  - **Reason**: Different algorithms have different quality on structured grids
  - **Testing**: Compare CM, RCM, AMD, ND on 2D grid, report metrics

- [ ] **`TEST(SparsityOptimizerEffective, CompareAlgorithms_Unstructured)`**
  - **Reason**: Unstructured meshes may favor different algorithms
  - **Testing**: Compare algorithms on unstructured tet mesh

- [ ] **`TEST(SparsityOptimizerEffective, BestAlgorithm_Selection)`**
  - **Reason**: Auto-selection should pick good algorithm
  - **Testing**: Auto choice achieves within 10% of best algorithm

### 11.3 External Library Integration

- [ ] **`TEST(SparsityOptimizerExternal, METIS_Available)`**
  - **Reason**: METIS provides high-quality orderings
  - **Testing**: If METIS available, produces valid ordering

- [ ] **`TEST(SparsityOptimizerExternal, METIS_Quality)`**
  - **Reason**: METIS should outperform simple algorithms on large problems
  - **Testing**: METIS fill-in <= AMD fill-in for 10k DOF problem

- [ ] **`TEST(SparsityOptimizerExternal, Scotch_Available)`**
  - **Reason**: Scotch is alternative to METIS
  - **Testing**: If Scotch available, produces valid ordering

- [ ] **`TEST(SparsityOptimizerExternal, ParMETIS_Distributed)`**
  - **Reason**: ParMETIS for distributed ordering
  - **Testing**: ParMETIS ordering consistent across ranks

### 11.4 Goal-Specific Optimization

- [ ] **`TEST(SparsityOptimizerGoal, MinimizeBandwidth_Achieved)`**
  - **Reason**: Bandwidth goal should prioritize bandwidth reduction
  - **Testing**: Bandwidth goal achieves lowest bandwidth among goals

- [ ] **`TEST(SparsityOptimizerGoal, MinimizeFillIn_Achieved)`**
  - **Reason**: Fill-in goal should prioritize fill reduction
  - **Testing**: Fill-in goal achieves lowest fill-in among goals

- [ ] **`TEST(SparsityOptimizerGoal, OptimizeParallel_ColorCount)`**
  - **Reason**: Parallel goal should minimize color count for parallel assembly
  - **Testing**: Parallel goal produces fewer colors than bandwidth goal

---

## 12. SparsityAnalyzer Accuracy Tests

### 12.1 Symmetry Detection

- [ ] **`TEST(SparsityAnalyzerAccuracy, StructuralSymmetry_Exact)`**
  - **Reason**: Many FE matrices are structurally symmetric
  - **Testing**: Known symmetric pattern detected as symmetric

- [ ] **`TEST(SparsityAnalyzerAccuracy, StructuralSymmetry_Asymmetric)`**
  - **Reason**: Must not falsely detect asymmetric as symmetric
  - **Testing**: Known asymmetric pattern (convection) detected correctly

- [ ] **`TEST(SparsityAnalyzerAccuracy, NearSymmetry_Detection)`**
  - **Reason**: Some patterns are "almost" symmetric (few asymmetric entries)
  - **Testing**: Pattern with 1% asymmetric entries, symmetry ratio reported

### 12.2 Diagonal Dominance

- [ ] **`TEST(SparsityAnalyzerAccuracy, DiagonalDominance_Structure)`**
  - **Reason**: Diagonal dominance affects solver convergence
  - **Testing**: Pattern where diagonal exists in all rows, detected correctly

- [ ] **`TEST(SparsityAnalyzerAccuracy, MissingDiagonal_Detection)`**
  - **Reason**: Missing diagonal entries cause solver failures
  - **Testing**: Pattern with missing diagonal entries, reported

### 12.3 Block Structure Detection

- [ ] **`TEST(SparsityAnalyzerAccuracy, BlockStructure_Regular)`**
  - **Reason**: Regular block structure enables block algorithms
  - **Testing**: 3x3 block pattern, block size detected as 3

- [ ] **`TEST(SparsityAnalyzerAccuracy, BlockStructure_Irregular)`**
  - **Reason**: Must not falsely detect blocks where none exist
  - **Testing**: Irregular pattern, no block structure reported

- [ ] **`TEST(SparsityAnalyzerAccuracy, BlockStructure_Partial)`**
  - **Reason**: Partial block structure (some blocks, not all) detected
  - **Testing**: Pattern with local block structure, reported correctly

### 12.4 Solver Recommendations

- [ ] **`TEST(SparsityAnalyzerAccuracy, SolverRecommendation_SPD)`**
  - **Reason**: SPD patterns should recommend Cholesky
  - **Testing**: Symmetric positive-definite pattern recommends Cholesky

- [ ] **`TEST(SparsityAnalyzerAccuracy, SolverRecommendation_SaddlePoint)`**
  - **Reason**: Saddle-point patterns need special solvers
  - **Testing**: Stokes pattern recommends Uzawa/Schur complement

- [ ] **`TEST(SparsityAnalyzerAccuracy, SolverRecommendation_Nonsymmetric)`**
  - **Reason**: Nonsymmetric patterns need GMRES/BiCGSTAB
  - **Testing**: Convection-dominated pattern recommends Krylov method

---

## 13. AdaptiveSparsity Complex Scenarios

### 13.1 Refinement Patterns

- [ ] **`TEST(AdaptiveSparsityComplex, UniformRefinement_PatternGrowth)`**
  - **Reason**: Uniform refinement has predictable pattern growth
  - **Testing**: Refine all elements, verify NNZ growth matches theory

- [ ] **`TEST(AdaptiveSparsityComplex, LocalRefinement_HangingNodes)`**
  - **Reason**: Local refinement creates hanging nodes with constraints
  - **Testing**: Pattern includes hanging node constraint couplings

- [ ] **`TEST(AdaptiveSparsityComplex, MultilevelRefinement)`**
  - **Reason**: Multiple refinement levels create complex constraints
  - **Testing**: 2-level refinement difference handled correctly

### 13.2 Coarsening

- [ ] **`TEST(AdaptiveSparsityComplex, Coarsening_PatternShrink)`**
  - **Reason**: Coarsening removes DOFs and couplings
  - **Testing**: Coarsen elements, pattern shrinks appropriately

- [ ] **`TEST(AdaptiveSparsityComplex, Coarsening_Consistency)`**
  - **Reason**: Coarsening after refinement should recover original
  - **Testing**: Refine then coarsen same elements, pattern matches original

### 13.3 P-Adaptivity

- [ ] **`TEST(AdaptiveSparsityComplex, PRefine_DOFGrowth)`**
  - **Reason**: P-refinement adds DOFs within elements
  - **Testing**: Increase polynomial order, pattern grows correctly

- [ ] **`TEST(AdaptiveSparsityComplex, PRefinement_NeighborCoupling)`**
  - **Reason**: Higher-order element couples to neighbors differently
  - **Testing**: P-refined element's neighbor couplings updated

### 13.4 Load Balancing

- [ ] **`TEST(AdaptiveSparsityComplex, Redistribution_PatternUpdate)`**
  - **Reason**: Repartitioning changes ownership and ghost structure
  - **Testing**: After redistribution, distributed pattern consistent

- [ ] **`TEST(AdaptiveSparsityComplex, MigrationTracking)`**
  - **Reason**: Migrated elements need pattern updates on new rank
  - **Testing**: Element migration reflected in local patterns

---

## 14. ConstraintSparsityAugmenter Tests

### 14.1 Constraint Types

- [ ] **`TEST(ConstraintAugmenterAdvanced, DirichletElimination)`**
  - **Reason**: Dirichlet BC elimination modifies pattern
  - **Testing**: Eliminated DOF rows/columns removed, fill-in added

- [ ] **`TEST(ConstraintAugmenterAdvanced, MultiPointConstraint_Simple)`**
  - **Reason**: MPCs couple multiple DOFs (e.g., u_1 = u_2 + u_3)
  - **Testing**: MPC creates coupling between constrained DOFs

- [ ] **`TEST(ConstraintAugmenterAdvanced, MultiPointConstraint_Chain)`**
  - **Reason**: Chained MPCs (u_1 depends on u_2 which depends on u_3)
  - **Testing**: Chain resolved, full coupling pattern determined

- [ ] **`TEST(ConstraintAugmenterAdvanced, PeriodicBC_Coupling)`**
  - **Reason**: Periodic BCs couple boundary DOFs to opposite boundary
  - **Testing**: Non-local coupling entries added correctly

### 14.2 Fill-in Patterns

- [ ] **`TEST(ConstraintAugmenterAdvanced, EliminationFillIn_Minimal)`**
  - **Reason**: Elimination order affects fill-in amount
  - **Testing**: Optimal elimination order produces minimal fill-in

- [ ] **`TEST(ConstraintAugmenterAdvanced, EliminationFillIn_Dense)`**
  - **Reason**: Poor elimination can cause dense fill-in
  - **Testing**: Worst-case elimination produces maximal fill-in

- [ ] **`TEST(ConstraintAugmenterAdvanced, FillIn_Prediction)`**
  - **Reason**: Predicted fill-in should match actual
  - **Testing**: Predicted fill-in entries match actual after elimination

### 14.3 Distributed Constraints

- [ ] **`TEST(ConstraintAugmenterMPI, CrossRankConstraint)`**
  - **Reason**: MPCs may span MPI ranks
  - **Testing**: Constraint between DOFs on different ranks handled

- [ ] **`TEST(ConstraintAugmenterMPI, GhostDOF_Constraint)`**
  - **Reason**: Constrained DOF may be ghost on some ranks
  - **Testing**: Ghost DOF constraints produce correct off-diagonal entries

---

## 15. Numerical Reproducibility Tests

### 15.1 Determinism

- [ ] **`TEST(Reproducibility, SameInput_SameOutput)`**
  - **Reason**: Same input must produce identical pattern every time
  - **Testing**: Build same pattern twice, binary identical

- [ ] **`TEST(Reproducibility, ParallelBuild_Deterministic)`**
  - **Reason**: Parallel construction must be deterministic
  - **Testing**: OpenMP parallel build produces same result as serial

- [ ] **`TEST(Reproducibility, CrossPlatform_Consistent)`**
  - **Reason**: Pattern should be same on different platforms
  - **Testing**: Compare pattern checksum across platforms (via CI)

### 15.2 Ordering Consistency

- [ ] **`TEST(Reproducibility, ColumnOrdering_Sorted)`**
  - **Reason**: Column indices must be sorted for many solvers
  - **Testing**: Verify all rows have sorted column indices

- [ ] **`TEST(Reproducibility, InsertionOrder_Independent)`**
  - **Reason**: Final pattern should not depend on insertion order
  - **Testing**: Same entries in different order produce same pattern

---

## 16. Performance Regression Tests

### 16.1 Construction Time

- [ ] **`TEST(Performance, Construction_10kDOF_Under100ms)`**
  - **Reason**: Small problems should build quickly
  - **Testing**: 10k DOF pattern builds in < 100ms

- [ ] **`TEST(Performance, Construction_100kDOF_Under1s)`**
  - **Reason**: Medium problems should build reasonably fast
  - **Testing**: 100k DOF pattern builds in < 1s

- [ ] **`TEST(Performance, Construction_ScalingLinear)`**
  - **Reason**: Construction should scale linearly with NNZ
  - **Testing**: Time ratio for 10x NNZ is approximately 10x

### 16.2 Memory Usage

- [ ] **`TEST(Performance, Memory_CSRCompact)`**
  - **Reason**: Final CSR should use minimal memory
  - **Testing**: Memory = O(NNZ + n_rows), not more

- [ ] **`TEST(Performance, Memory_BuilderOverhead)`**
  - **Reason**: Builder temporary memory should be bounded
  - **Testing**: Peak memory during build < 3x final pattern size

### 16.3 Operation Time

- [ ] **`TEST(Performance, Transpose_LinearTime)`**
  - **Reason**: Transpose should be O(NNZ)
  - **Testing**: Transpose time scales linearly with NNZ

- [ ] **`TEST(Performance, SymmetryCheck_LinearTime)`**
  - **Reason**: Symmetry check should be O(NNZ)
  - **Testing**: Symmetry check time scales linearly

- [ ] **`TEST(Performance, FormatConversion_LinearTime)`**
  - **Reason**: Format conversions should be O(NNZ)
  - **Testing**: CSR→COO time scales linearly

---

## Summary Statistics

| Category | Test Count | Priority |
|----------|------------|----------|
| FE Literature Validation | 18 | **High** |
| Large-Scale/Stress | 12 | **High** |
| Backend Integration | 10 | **High** |
| SparsityPattern Edge Cases | 12 | Medium |
| SparsityFormat Conversions | 12 | Medium |
| GraphSparsity Algorithms | 22 | **High** |
| SparsityBuilder Advanced | 14 | Medium |
| DGSparsityBuilder | 14 | Medium |
| DistributedSparsity MPI | 14 | **High** |
| BlockSparsity Advanced | 12 | Medium |
| SparsityOptimizer Effectiveness | 16 | **High** |
| SparsityAnalyzer Accuracy | 12 | Medium |
| AdaptiveSparsity Complex | 10 | Medium |
| ConstraintAugmenter | 10 | Medium |
| Reproducibility | 5 | **High** |
| Performance Regression | 9 | Medium |
| **TOTAL** | **202** | |

---

## Implementation Priority

### Phase 1: Critical (Implement First)
1. FE Literature Validation (Poisson, Elasticity patterns)
2. Backend Integration (PETSc preallocation)
3. Large-Scale Testing (1M DOF verification)
4. Reproducibility Tests

### Phase 2: High Priority
1. GraphSparsity Algorithm Validation (fill-in, reordering)
2. DistributedSparsity MPI Tests
3. SparsityOptimizer Effectiveness

### Phase 3: Medium Priority
1. Edge Cases (formats, patterns)
2. Advanced Scenarios (DG, constraints, adaptivity)
3. Performance Regression

---

## Notes for Implementers

1. **Test Fixtures**: Create reusable fixtures for standard FE patterns (Poisson2D, Elasticity3D, Stokes)

2. **Reference Data**: Store reference patterns from MATLAB/SuiteSparse for comparison

3. **Conditional Tests**: Use `GTEST_SKIP()` for tests requiring optional dependencies (METIS, PETSc)

4. **MPI Tests**: Use Google Test's MPI extensions or CTest for multi-rank tests

5. **Performance Tests**: Use Google Benchmark or mark as "slow" tests for CI

6. **Matrix Market**: Consider adding Matrix Market I/O for loading reference patterns

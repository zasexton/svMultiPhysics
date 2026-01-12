# Dofs Unit Test Checklist

This document lists additional unit tests that should be added to the Dofs subfolder to achieve comprehensive coverage. Tests are organized by module and priority level.

**Status Legend:**
- [ ] Not started / Deferred
- [x] Completed

---

## Priority Legend
- **P0**: Critical - Core functionality gaps that could hide bugs
- **P1**: High - Important edge cases and conformance verification
- **P2**: Medium - Extended coverage for robustness
- **P3**: Low - Nice-to-have, stress tests, exotic configurations

---

## 1. ConstrainedAssembly (P0 - Significantly Under-tested)

The ConstrainedAssembly module currently has only 71 lines of tests despite being critical for boundary condition enforcement.

### 1.1 Periodic Constraint Assembly
- [x] **Test**: `ConstrainedAssembly_PeriodicConstraint_SymmetricCoupling`
- **Reason**: Periodic BCs are common in fluid dynamics (channel flow) and solid mechanics (unit cells). The current tests only cover Dirichlet elimination.
- **Verifies**: That periodic DOF pairs (e.g., left-right faces) produce symmetric coupling in the assembled matrix with correct coefficient transfer.

### 1.2 Multi-Point Constraint Assembly
- [x] **Test**: `ConstrainedAssembly_LinearConstraint_ThreePointAverage`
- **Reason**: Hanging nodes from mesh refinement create multi-point constraints (e.g., edge midpoint = 0.5*v1 + 0.5*v2). These are more complex than simple Dirichlet.
- **Verifies**: That a DOF constrained as a linear combination of multiple masters produces correct row/column modifications with weighted contributions.

### 1.3 Chained Constraint Assembly
- [x] **Test**: `ConstrainedAssembly_TransitiveConstraint_ChainOfThree`
- **Reason**: After transitive closure, constraints can form chains (A depends on B depends on C). Assembly must handle the resolved form correctly.
- **Verifies**: That constraints involving intermediate masters (resolved via transitive closure) assemble correctly without double-counting.

### 1.4 Non-Symmetric Matrix Assembly
- [x] **Test**: `ConstrainedAssembly_NonSymmetricMatrix_AdvectionOperator`
- **Reason**: Advection-dominated problems have non-symmetric matrices. Constraint elimination differs (no column-to-RHS transfer for non-symmetric).
- **Verifies**: That the assembler correctly detects non-symmetric matrices and skips symmetric elimination optimizations.

### 1.5 Large Sparse System Assembly
- [x] **Test**: `ConstrainedAssembly_LargeSparse_1000x1000WithConstraints`
- **Reason**: Current tests use tiny (3x3, 4x4) matrices. Real FE systems have thousands of DOFs. Need to verify no O(n²) performance bugs.
- **Verifies**: That assembly scales linearly with number of non-zeros, not quadratically with matrix size.

### 1.6 Mixed Dirichlet and Periodic Assembly
- [x] **Test**: `ConstrainedAssembly_MixedConstraints_DirichletAndPeriodic`
- **Reason**: Real problems combine constraint types. A periodic face may also have Dirichlet on edges.
- **Verifies**: That multiple constraint types on different DOFs are handled correctly in a single assembly pass.

### 1.7 RHS-Only Assembly with Constraints
- [x] **Test**: `ConstrainedAssembly_RHSOnly_ConstrainedLoad`
- **Reason**: Time-dependent problems often reassemble only the RHS. Constraints must still be applied correctly.
- **Verifies**: That vector-only assembly applies constraint modifications (inhomogeneous Dirichlet contributions) correctly.

### 1.8 Zero Diagonal After Elimination
- [ ] **Test**: `ConstrainedAssembly_ZeroDiagonal_Detection` (Deferred: no global-diagonal inspection/warning hook in `ConstrainedAssembly::BackendAdapter`)
- **Reason**: Incorrect constraint application can zero out diagonal entries, causing solver failures.
- **Verifies**: That the assembler detects/warns when a constrained DOF elimination produces a zero diagonal.

---

## 2. DofHandler - 3D Simplex Elements (P0)

### 2.1 Tetrahedron P2 Edge Orientation
- [x] **Test**: `DofHandler_Tet10_EdgeOrientationReversal`
- **Reason**: Tetrahedra have 6 edges, each potentially reversed between adjacent cells. P2 edge DOFs must maintain continuity regardless of local edge direction.
- **Verifies**: That P2 tetrahedron edge DOFs match across shared edges when local orientations differ.

### 2.2 Tetrahedron P3 Edge and Face Orientation
- [x] **Test**: `DofHandler_Tet20_EdgeAndFaceOrientationPermutation`
- **Reason**: P3 tetrahedra have 2 DOFs per edge and 1 DOF per face. Face DOFs require handling 6 possible triangle orientations (3 rotations × 2 reflections).
- **Verifies**: That P3 tet face DOFs produce continuous basis functions under all face orientation permutations.

### 2.3 Tetrahedron P4 Higher-Order Continuity
- [ ] **Test**: `DofHandler_Tet35_P4FaceDofPermutation` (Deferred: `DofLayoutInfo::Lagrange` order > 3 not implemented)
- **Reason**: P4 tetrahedra have 3 DOFs per face arranged in a triangle pattern. Orientation handling is more complex than P3.
- **Verifies**: That the face DOF permutation logic generalizes correctly to P4 and higher.

### 2.4 Mixed Tet-Hex Mesh
- [ ] **Test**: `DofHandler_MixedTetHex_SharedFaceContinuity` (Deferred: `DofHandler` currently assumes a single element topology/layout)
- **Reason**: Hybrid meshes (tets near boundaries, hexes in interior) have tet-hex interfaces where triangle-quad face transitions occur.
- **Verifies**: That DOF numbering handles meshes with multiple element types without continuity breaks.

---

## 3. DofHandler - Wedge and Pyramid Elements (P1)

### 3.1 Wedge P2 Distribution
- [ ] **Test**: `DofHandler_Wedge15_EdgeAndFaceDistribution` (Deferred: `DofLayoutInfo::Lagrange` does not yet support wedge layouts)
- **Reason**: Wedge elements have mixed face types (2 triangles, 3 quads). DOF distribution must handle both correctly.
- **Verifies**: That wedge element DOF counts and entity assignments match reference element definitions.

### 3.2 Pyramid P2 Distribution
- [ ] **Test**: `DofHandler_Pyramid13_ApexSingularity` (Deferred: `DofLayoutInfo::Lagrange` does not yet support pyramid layouts)
- **Reason**: Pyramid elements have a geometric singularity at the apex where 4 edges meet. Basis functions require special treatment.
- **Verifies**: That pyramid apex DOF handling follows standard conventions (collapsed hex approach or native pyramid basis).

### 3.3 Wedge-Hex Interface
- [ ] **Test**: `DofHandler_WedgeHexInterface_QuadFaceContinuity` (Deferred: mixed-element support not present in `DofHandler`)
- **Reason**: Wedge elements are often used as transition elements between tet and hex regions.
- **Verifies**: That shared quadrilateral faces between wedge and hex elements have matching DOF numbering.

---

## 4. DofNumbering (P1)

### 4.1 RCM Bandwidth Reduction Verification
- [x] **Test**: `DofNumbering_RCM_BandwidthActuallyDecreases`
- **Reason**: Current test only verifies valid permutation, not that RCM actually reduces bandwidth. A bug could produce valid but ineffective numbering.
- **Verifies**: That after RCM reordering, the matrix bandwidth is less than or equal to the original bandwidth.

### 4.2 RCM Starting Node Selection
- [x] **Test**: `DofNumbering_RCM_PseudoPeripheralStartNode`
- **Reason**: RCM quality depends on starting node choice. Pseudo-peripheral node selection should be tested.
- **Verifies**: That the implementation uses a pseudo-peripheral starting node (or equivalent heuristic) for optimal bandwidth.

### 4.3 Disconnected Graph Handling
- [x] **Test**: `DofNumbering_RCM_DisconnectedComponents`
- **Reason**: Some meshes have disconnected regions. RCM must handle multiple connected components.
- **Verifies**: That RCM produces valid permutation for graphs with 2+ disconnected components.

### 4.4 AMD (Approximate Minimum Degree) Numbering
- [ ] **Test**: `DofNumbering_AMD_FillReduction` (Deferred: AMD strategy not implemented)
- **Reason**: AMD is often better than RCM for direct solvers (reduces fill-in during factorization). If AMD is implemented, it needs testing.
- **Verifies**: That AMD numbering produces lower fill-in than natural ordering for sparse Cholesky.

### 4.5 Block-Aware Numbering
- [ ] **Test**: `DofNumbering_BlockRCM_PreservesBlockStructure` (Deferred: block-aware RCM strategy not implemented)
- **Reason**: For block systems (e.g., 3D elasticity), numbering should operate on nodes, not individual DOFs, to preserve block structure.
- **Verifies**: That block-aware RCM keeps all DOFs of a node contiguous while reducing bandwidth.

---

## 5. DofGraph (P1)

### 5.1 Non-Contiguous DOF Range
- [x] **Test**: `DofGraph_NonContiguousDofs_CorrectAdjacency`
- **Reason**: After constraint elimination or DOF renumbering, DOF indices may have gaps. Graph must handle this.
- **Verifies**: That DofGraph correctly builds adjacency when DOF indices are sparse/non-contiguous.

### 5.2 Self-Loop Exclusion Option
- [x] **Test**: `DofGraph_ExcludeSelfLoops_DiagonalFree`
- **Reason**: Some algorithms (e.g., graph Laplacian) require adjacency without self-loops.
- **Verifies**: That the DofGraphOptions::exclude_self_loops option removes diagonal entries from adjacency.

### 5.3 Weighted Graph for AMG
- [ ] **Test**: `DofGraph_WeightedAdjacency_StiffnessBasedWeights` (Deferred: weighted adjacency storage not implemented)
- **Reason**: Algebraic multigrid (AMG) benefits from strength-of-connection weights derived from matrix entries.
- **Verifies**: That DofGraph can store and retrieve edge weights for AMG coarsening.

### 5.4 Graph Symmetry Enforcement
- [x] **Test**: `DofGraph_SymmetrizeNonSymmetricInput`
- **Reason**: Non-symmetric assembly (advection) produces non-symmetric adjacency. Some algorithms need symmetrized graph.
- **Verifies**: That symmetrization option produces A + A^T pattern.

---

## 6. DofMap (P1)

### 6.1 LocalToGlobal with Non-Identity Numbering
- [x] **Test**: `DofMap_LocalToGlobal_AfterRCMRenumbering`
- **Reason**: After DOF renumbering, localToGlobal() mapping must reflect the new numbering.
- **Verifies**: That local-to-global mapping updates correctly after applyNumbering().

### 6.2 Parallel Partitioned DOF Range
- [ ] **Test**: `DofMap_ParallelPartition_LocalRangeQueries` (Deferred: local range queries are on `DofHandler`, not `DofMap`)
- **Reason**: In parallel, each rank owns a contiguous range of global DOFs. Range queries must be efficient.
- **Verifies**: That getLocalDofRange() returns correct [start, end) for each MPI rank.

### 6.3 DOF Ownership Transfer
- [ ] **Test**: `DofMap_OwnershipTransfer_AfterRepartitioning` (Deferred: repartition/ownership-transfer API not implemented)
- **Reason**: Dynamic load balancing may transfer DOF ownership between ranks.
- **Verifies**: That ownership information can be updated without full redistribution.

### 6.4 Cell DOF Reordering
- [ ] **Test**: `DofMap_CellDofReorder_MatchesReferenceElement`
- **Reason**: Cell DOFs must follow reference element ordering for correct basis function evaluation.
- **Verifies**: That getCellDofs() ordering matches the reference element node numbering convention.

---

## 7. DofTools (P1)

### 7.1 Geometric Predicate DOF Extraction
- [ ] **Test**: `DofTools_GeometricPredicate_PlaneIntersection` (Deferred: `extractDofsByPredicate()` not implemented)
- **Reason**: Users often need DOFs satisfying geometric conditions (e.g., x=0 plane for symmetry BC).
- **Verifies**: That extractDofsByPredicate() correctly identifies DOFs where associated nodes satisfy a spatial condition.

### 7.2 Sphere Region DOF Extraction
- [ ] **Test**: `DofTools_GeometricPredicate_SphereInterior` (Deferred: `extractDofsByPredicate()` not implemented)
- **Reason**: Point source BCs or local refinement indicators need DOFs within a radius of a point.
- **Verifies**: That DOFs with support intersecting a sphere are correctly identified.

### 7.3 Component Mask Inversion
- [x] **Test**: `DofTools_ComponentMask_ComplementMask`
- **Reason**: "All components except Z" is a common pattern. Mask complement should be tested.
- **Verifies**: That ComponentMask complement operation produces correct result.

### 7.4 DOF Support Overlap Query
- [ ] **Test**: `DofTools_SupportOverlap_AdjacentDofs` (Deferred: support-overlap query API not implemented)
- **Reason**: Preconditioner construction (block Jacobi) needs to identify DOFs with overlapping support.
- **Verifies**: That support overlap queries correctly identify DOFs sharing element support.

### 7.5 Boundary DOF Extraction by Marker
- [ ] **Test**: `DofTools_BoundaryDofs_ByFaceMarker`
- **Reason**: Real meshes have boundary markers (inlet=1, outlet=2, wall=3). Extraction should filter by marker.
- **Verifies**: That boundary DOF extraction can filter by face/edge marker ID.

---

## 8. GhostDofManager (P1)

### 8.1 Mesh-Based Ghost Identification
- [ ] **Test**: `GhostDofManager_MeshBased_IdentifyFromGhostCells`
- **Reason**: All current tests use manual ghost setup. Real usage derives ghosts from mesh ghost cells.
- **Verifies**: That identifyGhostDofs() correctly identifies ghosts from a partitioned mesh with ghost cell layer.

### 8.2 Multi-Level Ghost Layer
- [ ] **Test**: `GhostDofManager_TwoLevelGhostLayer_ExtendedStencil`
- **Reason**: High-order methods or certain preconditioners need 2+ layers of ghost cells/DOFs.
- **Verifies**: That ghost identification can be configured for multi-layer stencils.

### 8.3 Custom Ownership Resolution
- [ ] **Test**: `GhostDofManager_CustomOwnership_LoadBalancedStrategy`
- **Reason**: The Custom ownership strategy is declared but not tested.
- **Verifies**: That custom ownership function is called and its result respected.

### 8.4 Empty Neighbor Handling
- [ ] **Test**: `GhostDofManager_NoNeighbors_SingleRankPartition`
- **Reason**: Single-rank runs or interior subdomains may have no neighbors. Edge case needs coverage.
- **Verifies**: That GhostDofManager handles empty neighbor_ranks without errors.

### 8.5 Large Neighbor Count
- [ ] **Test**: `GhostDofManager_ManyNeighbors_16RankStarTopology`
- **Reason**: Complex partitions may have many neighbors. Buffer allocation and scheduling must scale.
- **Verifies**: That communication schedules are built correctly with 10+ neighbors.

---

## 9. BlockDofMap (P2)

### 9.1 Three-Block System
- [ ] **Test**: `BlockDofMap_ThreeBlocks_VelocityPressureTemperature`
- **Reason**: Current tests only cover 2-block systems. Thermally-coupled flow has 3 blocks.
- **Verifies**: That 3+ block systems have correct ranges and coupling specification.

### 9.2 Asymmetric Coupling Pattern
- [ ] **Test**: `BlockDofMap_AsymmetricCoupling_OperatorSplitting`
- **Reason**: Operator splitting may have A coupled to B but not B to A.
- **Verifies**: That asymmetric coupling matrices are supported and detected.

### 9.3 Variable Block Sizes
- [ ] **Test**: `BlockDofMap_VariableBlockSizes_ScalarVectorMixed`
- **Reason**: Mixed scalar (pressure) and vector (velocity) fields have different DOF counts per node.
- **Verifies**: That blocks with different DOFs-per-node are handled correctly.

### 9.4 Block Extraction Performance
- [ ] **Test**: `BlockDofMap_BlockExtraction_LargeSystem`
- **Reason**: Block preconditioners extract block submatrices frequently. This must be efficient.
- **Verifies**: That getBlockView() is O(1) and doesn't copy data.

---

## 10. EntityDofMap (P2)

### 10.1 Mixed Element Type Mesh
- [ ] **Test**: `EntityDofMap_MixedElements_TriQuadMesh`
- **Reason**: Unstructured meshes often mix triangles and quads. Entity-DOF mapping must handle both.
- **Verifies**: That entity DOF queries work correctly on meshes with multiple cell types.

### 10.2 DG Entity Isolation
- [ ] **Test**: `EntityDofMap_DG_NoCrossElementSharing`
- **Reason**: DG elements should have no shared entity DOFs. This invariant should be tested.
- **Verifies**: That DG discretization produces empty shared DOF sets for all entity types.

### 10.3 Face DOF Orientation Query
- [ ] **Test**: `EntityDofMap_FaceDofOrientation_RelativeToCell`
- **Reason**: Users may need to know face DOF orientation relative to a specific cell (not just global orientation).
- **Verifies**: That getFaceDofOrientation(cell, local_face) returns correct permutation.

### 10.4 Entity DOF Count by Type
- [ ] **Test**: `EntityDofMap_DofCountsByEntityType`
- **Reason**: Statistics on DOF distribution (how many vertex vs edge vs face DOFs) aid debugging.
- **Verifies**: That getDofCountByEntityType() returns correct counts matching theoretical values.

---

## 11. FieldDofMap (P2)

### 11.1 Layout Conversion
- [ ] **Test**: `FieldDofMap_LayoutConversion_InterleavedToBlock` (Deferred: layout conversion API not implemented)
- **Reason**: Some solvers prefer block, others interleaved. Conversion between layouts is useful.
- **Verifies**: That convertLayout() produces correct DOF mapping between Interleaved and Block.

### 11.2 Field Addition After Finalize
- [x] **Test**: `FieldDofMap_AddFieldAfterFinalize_ThrowsException`
- **Reason**: Adding fields after finalize would invalidate cached data. This should be prevented.
- **Verifies**: That addField() throws after finalize().

### 11.3 Field Removal
- [ ] **Test**: `FieldDofMap_RemoveField_UpdatesRanges` (Deferred: field removal API not implemented)
- **Reason**: Static condensation removes internal DOFs. Field removal should be supported.
- **Verifies**: That removeField() correctly updates all range information.

### 11.4 Tensor Field
- [ ] **Test**: `FieldDofMap_TensorField_StressComponents` (Deferred: tensor-field registration API not implemented)
- **Reason**: Stress/strain tensors have 6 independent components (symmetric 3x3). This is beyond scalar/vector.
- **Verifies**: That tensor fields with arbitrary component counts are handled correctly.

---

## 12. DofConstraints (P2)

### 12.1 Large Constraint Chain
- [x] **Test**: `DofConstraints_TransitiveClosure_LongChain`
- **Reason**: Multiply-refined meshes can have long constraint chains (5+ levels). Closure must handle depth.
- **Verifies**: That transitive closure correctly resolves chains of 10+ constraints.

### 12.2 Constraint Serialization
- [ ] **Test**: `DofConstraints_Serialization_SaveAndLoad` (Deferred: serialization API not implemented)
- **Reason**: Checkpoint/restart requires saving constraint state.
- **Verifies**: That constraints can be serialized and deserialized without data loss.

### 12.3 Constraint Modification
- [ ] **Test**: `DofConstraints_ModifyExisting_UpdateCoefficient` (Deferred: constraint modification API not implemented)
- **Reason**: Adaptive methods may need to update constraint coefficients without full rebuild.
- **Verifies**: That existing constraints can be modified after initial setup.

### 12.4 Inhomogeneous Periodic Constraints
- [ ] **Test**: `DofConstraints_InhomogeneousPeriodic_PhaseShift` (Deferred: inhomogeneous periodic constraints not implemented)
- **Reason**: Bloch-periodic BCs have u(x+L) = e^{ikL} u(x), which is inhomogeneous periodic.
- **Verifies**: That periodic constraints with non-zero inhomogeneity are supported.

### 12.5 Constraint Statistics
- [ ] **Test**: `DofConstraints_Statistics_CountByType` (Deferred: statistics API not implemented)
- **Reason**: Debugging constraint issues requires knowing how many of each type exist.
- **Verifies**: That getStatistics() returns correct counts of Dirichlet, Periodic, and Linear constraints.

---

## 13. SubspaceView (P2)

### 13.1 Nested Subspace View
- [ ] **Test**: `SubspaceView_NestedView_ComponentOfField`
- **Reason**: Extract X-component of velocity from a velocity-pressure system (view of view).
- **Verifies**: That SubspaceView can be constructed from another SubspaceView.

### 13.2 Strided Access Pattern
- [ ] **Test**: `SubspaceView_StridedAccess_InterleavedComponents`
- **Reason**: Interleaved layout has stride-3 access for each component. View should optimize this.
- **Verifies**: That strided views use efficient access patterns, not element-by-element copy.

### 13.3 View Intersection
- [ ] **Test**: `SubspaceView_Intersection_BoundaryAndField`
- **Reason**: "Boundary DOFs that are also velocity DOFs" requires view intersection.
- **Verifies**: That intersect() produces correct DOF set.

### 13.4 Read-Only View Enforcement
- [ ] **Test**: `SubspaceView_ReadOnly_ScatterThrows`
- **Reason**: Some views should be read-only (e.g., views of const vectors). Scatter should be prevented.
- **Verifies**: That read-only views reject scatter operations.

---

## 14. MeshTopologyBuilder (P2)

### 14.1 Tetrahedron Topology
- [ ] **Test**: `MeshTopologyBuilder_Tetrahedron_EdgeAndFaceOrder`
- **Reason**: Tetrahedron has 6 edges and 4 triangular faces. Reference ordering must be tested.
- **Verifies**: That cell-to-edge and cell-to-face mappings match standard tet reference element.

### 14.2 Wedge Topology
- [ ] **Test**: `MeshTopologyBuilder_Wedge_MixedFaceTypes`
- **Reason**: Wedge has 2 triangular and 3 quadrilateral faces. Builder must distinguish face types.
- **Verifies**: That wedge topology correctly identifies triangle vs quad faces.

### 14.3 Pyramid Topology
- [ ] **Test**: `MeshTopologyBuilder_Pyramid_BaseAndTriangleFaces`
- **Reason**: Pyramid has 1 quad base and 4 triangular side faces.
- **Verifies**: That pyramid cell-to-face mapping handles the 5-face configuration correctly.

### 14.4 Multi-Cell Mesh
- [ ] **Test**: `MeshTopologyBuilder_MultiCell_SharedEdgeConsistency`
- **Reason**: Current tests use single-cell meshes. Shared entity handling needs multi-cell tests.
- **Verifies**: That shared edges/faces between adjacent cells have consistent local-to-global mapping.

### 14.5 Reversed Cell Orientation
- [ ] **Test**: `MeshTopologyBuilder_ReversedCell_NegativeJacobian`
- **Reason**: Some mesh generators produce cells with negative Jacobian. Topology builder should detect or handle this.
- **Verifies**: That inverted cells are detected and reported (or corrected if auto-fix is enabled).

---

## 15. DofIndexSet (P2)

### 15.1 Large Index Set Performance
- [x] **Test**: `DofIndexSet_LargeSet_MillionIndices`
- **Reason**: Real problems have millions of DOFs. Set operations must scale.
- **Verifies**: That contains(), union, intersection remain O(1) or O(log n) for large sets.

### 15.2 Interval Compression
- [x] **Test**: `DofIndexSet_IntervalCompression_ManySmallIntervals`
- **Reason**: IndexSet stores intervals internally. Many small intervals should compress when adjacent.
- **Verifies**: That [0,10), [10,20) compresses to [0,20) automatically.

### 15.3 Set Difference
- [x] **Test**: `DofIndexSet_Difference_GhostMinusOwned`
- **Reason**: "All DOFs except owned" is a common pattern for ghost extraction.
- **Verifies**: That set difference operation produces correct result.

### 15.4 Iteration Order
- [x] **Test**: `DofIndexSet_Iteration_StrictlyAscending`
- **Reason**: Some algorithms require ascending iteration order. This should be guaranteed.
- **Verifies**: That iteration always visits indices in strictly ascending order.

---

## 16. DofHandler - MPI Extended (P2)

### 16.1 Single-Rank MPI
- [ ] **Test**: `DofHandlerMPI_SingleRank_MatchesSerial`
- **Reason**: MPI size=1 should produce identical results to non-MPI path.
- **Verifies**: That single-rank MPI produces same DOF numbering as serial execution.

### 16.2 Highly Imbalanced Partition
- [ ] **Test**: `DofHandlerMPI_ImbalancedPartition_90PercentOnRank0`
- **Reason**: Load imbalance is common in practice. Edge cases need testing.
- **Verifies**: That DOF distribution handles extreme imbalance without errors.

### 16.3 Empty Partition
- [ ] **Test**: `DofHandlerMPI_EmptyPartition_RankWithNoCells`
- **Reason**: Coarse meshes on many ranks may leave some ranks with no cells.
- **Verifies**: That ranks with zero local cells handle DOF distribution correctly.

### 16.4 Global DOF Count Overflow Prevention
- [ ] **Test**: `DofHandlerMPI_LargeGlobalCount_NoOverflow`
- **Reason**: Global DOF counts exceeding 2^31 require 64-bit indices throughout.
- **Verifies**: That GlobalIndex type handles counts > 2 billion correctly.

### 16.5 Checkpoint/Restart Consistency
- [ ] **Test**: `DofHandlerMPI_CheckpointRestart_SameNumbering`
- **Reason**: Restart must produce identical DOF numbering for solution compatibility.
- **Verifies**: That re-running DOF distribution produces identical global numbering.

---

## 17. Integration Tests (P3)

### 17.1 Full Assembly Pipeline
- [ ] **Test**: `Integration_FullPipeline_PoissonWithDirichlet`
- **Reason**: End-to-end test combining DofHandler → DofConstraints → ConstrainedAssembly.
- **Verifies**: That all Dofs components work together for a complete Poisson problem setup.

### 17.2 Parallel Assembly Verification
- [ ] **Test**: `Integration_ParallelAssembly_SumOfLocalMatrices`
- **Reason**: Parallel assembled global matrix should match serial assembly of same mesh.
- **Verifies**: That MPI assembly produces mathematically identical global system.

### 17.3 Refinement and Re-distribution
- [ ] **Test**: `Integration_AdaptiveRefinement_ConstraintUpdate`
- **Reason**: AMR requires updating DOF distribution and constraints after refinement.
- **Verifies**: That DOF redistribution after refinement preserves solution continuity.

---

## Summary Statistics

| Priority | Count | Focus Area |
|----------|-------|------------|
| P0 | 12 | ConstrainedAssembly, 3D simplex elements |
| P1 | 24 | DofNumbering, DofGraph, DofTools, GhostDofManager |
| P2 | 26 | Extended coverage for all modules |
| P3 | 3 | Integration tests |
| **Total** | **65** | |

---

## Implementation Notes

1. **Test Data**: Consider creating a shared test fixture library with pre-built meshes (single tet, single wedge, 2x2x2 hex, etc.) to reduce code duplication.

2. **MPI Tests**: Tests requiring specific MPI rank counts should use `GTEST_SKIP()` when run with wrong count, following the existing pattern in `test_DofHandlerMPI.cpp`.

3. **Performance Tests**: P2/P3 performance tests should have configurable problem sizes via environment variables to allow quick CI runs and thorough local testing.

4. **Reference Values**: Where possible, compute expected values analytically (e.g., P2 tet has 10 nodes = 4 vertex + 6 edge DOFs) rather than hard-coding from another implementation.

5. **Conformance Sources**: Reference implementations for verification:
   - deal.II: DoFHandler, ConstraintMatrix
   - MFEM: FiniteElementSpace
   - PETSc: DM, IS, VecScatter
   - libMesh: DofMap, DofObject

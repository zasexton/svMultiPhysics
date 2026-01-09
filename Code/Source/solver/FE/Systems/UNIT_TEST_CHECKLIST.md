# Systems Subfolder Unit Test Checklist

This document tracks missing unit tests for the `FE/Systems` library. Each section identifies gaps in test coverage, describes the required tests, and explains their importance for ensuring correctness and robustness.

**Test Location:** `Code/Source/solver/FE/Tests/Unit/Systems/`

---

## Table of Contents

1. [OperatorRegistry Tests](#1-operatorregistry-tests)
2. [GlobalKernelStateProvider Tests](#2-globalkernelstateprovider-tests)
3. [MaterialStateProvider Tests](#3-materialstateprovider-tests)
4. [FieldRegistry Tests](#4-fieldregistry-tests)
5. [TimeIntegrator Tests](#5-timeintegrator-tests)
6. [SearchAccess Tests](#6-searchaccess-tests)
7. [TransientSystem Tests](#7-transientsystem-tests)
8. [FormsInstaller Tests](#8-formsinstaller-tests)
9. [SystemAssembly Tests](#9-systemassembly-tests)
10. [FESystem Edge Case Tests](#10-fesystem-edge-case-tests)
11. [ParameterRegistry Extended Tests](#11-parameterregistry-extended-tests)
12. [SystemsExceptions Tests](#12-systemsexceptions-tests)

---

## 1. OperatorRegistry Tests

**File to create:** `test_OperatorRegistry.cpp`

**Current status:** No direct unit tests. Only tested indirectly through FESystem.

### Basic Functionality

- [x] **`OperatorRegistry_AddOperator_CreatesEmptyDefinition`**
  - **Description:** Verify that `addOperator(tag)` creates an empty `OperatorDefinition` with the correct tag and empty term vectors.
  - **Importance:** Ensures the registry correctly initializes operator entries before kernels are added.

- [x] **`OperatorRegistry_Has_ReturnsTrueForExistingOperator`**
  - **Description:** After adding an operator, `has(tag)` should return `true`.
  - **Importance:** Validates the existence check mechanism used throughout assembly.

- [x] **`OperatorRegistry_Has_ReturnsFalseForNonExistentOperator`**
  - **Description:** `has(tag)` should return `false` for operators never added.
  - **Importance:** Prevents false positives that could mask configuration errors.

- [x] **`OperatorRegistry_Get_ReturnsCorrectDefinition`**
  - **Description:** After adding an operator and kernels, `get(tag)` should return the definition with all registered terms.
  - **Importance:** Ensures kernel registration is correctly associated with operators.

- [x] **`OperatorRegistry_Get_ThrowsForNonExistentOperator`**
  - **Description:** `get(tag)` for a non-existent operator should throw an appropriate exception.
  - **Importance:** Fail-fast behavior prevents silent assembly failures.

- [x] **`OperatorRegistry_List_ReturnsAllOperatorTags`**
  - **Description:** After adding multiple operators, `list()` should return all tags.
  - **Importance:** Supports debugging and introspection of system configuration.

### Edge Cases

- [x] **`OperatorRegistry_AddOperator_RejectsDuplicateTag`**
  - **Description:** Adding an operator with an existing tag should throw or be rejected.
  - **Importance:** Prevents accidental operator definition overwrites.

- [x] **`OperatorRegistry_EmptyRegistry_ListReturnsEmpty`**
  - **Description:** `list()` on a fresh registry returns an empty vector.
  - **Importance:** Validates correct initialization state.

- [x] **`OperatorRegistry_Get_ConstAndNonConstVersionsConsistent`**
  - **Description:** Both const and non-const `get()` overloads return equivalent data.
  - **Importance:** Ensures const-correctness doesn't affect behavior.

---

## 2. GlobalKernelStateProvider Tests

**File to create:** `test_GlobalKernelStateProvider.cpp`

**Current status:** No dedicated tests. State management is critical for contact mechanics and history-dependent materials.

### Basic Functionality

- [x] **`GlobalKernelStateProvider_Constructor_InitializesWithCellCount`**
  - **Description:** Constructor with `num_cells` creates provider ready to accept kernel registrations.
  - **Importance:** Validates basic initialization for cell-based state storage.

- [x] **`GlobalKernelStateProvider_Constructor_AcceptsBoundaryAndInteriorFaceIds`**
  - **Description:** Constructor properly stores boundary and interior face ID mappings.
  - **Importance:** Required for face-based global kernels (e.g., surface contact).

- [x] **`GlobalKernelStateProvider_AddKernel_AllocatesStateBuffers`**
  - **Description:** After `addKernel()`, state buffers are allocated according to `GlobalStateSpec`.
  - **Importance:** Ensures memory is available before assembly begins.

- [x] **`GlobalKernelStateProvider_GetCellState_ReturnsValidView`**
  - **Description:** `getCellState()` returns a `MaterialStateView` with correct pointers and sizes.
  - **Importance:** Kernels depend on valid state views for history-dependent computations.

- [x] **`GlobalKernelStateProvider_GetBoundaryFaceState_ReturnsValidView`**
  - **Description:** `getBoundaryFaceState()` returns correct view for registered boundary faces.
  - **Importance:** Surface contact kernels require per-face state.

- [x] **`GlobalKernelStateProvider_GetInteriorFaceState_ReturnsValidView`**
  - **Description:** `getInteriorFaceState()` returns correct view for registered interior faces.
  - **Importance:** DG methods with history require interior face state.

### Double-Buffer Mechanics

- [x] **`GlobalKernelStateProvider_BeginTimeStep_SwapsBuffers`**
  - **Description:** After `beginTimeStep()`, the work buffer becomes the old buffer.
  - **Importance:** History-dependent kernels need previous time step data in `data_old`.

- [x] **`GlobalKernelStateProvider_CommitTimeStep_PreservesWorkData`**
  - **Description:** Data written to `data_work` during assembly is preserved after `commitTimeStep()`.
  - **Importance:** Ensures state updates are not lost between time steps.

- [x] **`GlobalKernelStateProvider_MultipleTimeSteps_MaintainsCorrectHistory`**
  - **Description:** Over multiple begin/commit cycles, state history is correctly maintained.
  - **Importance:** Long-running simulations depend on correct history propagation.

### Edge Cases

- [x] **`GlobalKernelStateProvider_GetCellState_InvalidCellId_Behavior`**
  - **Description:** Requesting state for out-of-bounds cell ID has defined behavior (throw or return invalid view).
  - **Importance:** Prevents undefined behavior from invalid mesh traversal.

- [x] **`GlobalKernelStateProvider_GetBoundaryFaceState_UnregisteredFace_Behavior`**
  - **Description:** Requesting state for a face not in the registered list has defined behavior.
  - **Importance:** Robust error handling for partial boundary coverage.

- [x] **`GlobalKernelStateProvider_AddKernel_ZeroBytesPerQpt_Behavior`**
  - **Description:** Adding a kernel with `bytes_per_qpt=0` is handled (no-op or error).
  - **Importance:** Kernels without state should not cause allocation failures.

- [x] **`GlobalKernelStateProvider_AddKernel_NonPowerOfTwoAlignment`**
  - **Description:** Non-standard alignment requirements are correctly honored.
  - **Importance:** SIMD operations may require specific alignments.

- [x] **`GlobalKernelStateProvider_GetState_RequestedQptsExceedsMax_Behavior`**
  - **Description:** Requesting more quadrature points than `max_qpts` has defined behavior.
  - **Importance:** Prevents buffer overruns.

- [x] **`GlobalKernelStateProvider_MoveSemantics_PreservesState`**
  - **Description:** Move constructor and assignment preserve all registered kernels and state.
  - **Importance:** Enables efficient system reconfiguration.

---

## 3. MaterialStateProvider Tests

**File to create:** `test_MaterialStateProvider.cpp`

**Current status:** No dedicated tests. Implements `IMaterialStateProvider` interface for element kernels.

### Basic Functionality

- [x] **`MaterialStateProvider_Constructor_InitializesCorrectly`**
  - **Description:** Constructor with cell count and face IDs creates valid provider.
  - **Importance:** Foundation for all material state storage.

- [x] **`MaterialStateProvider_AddKernel_AllocatesPerCellStorage`**
  - **Description:** After `addKernel()`, cell state storage is allocated for all cells.
  - **Importance:** Element kernels require per-cell-per-quadrature-point storage.

- [x] **`MaterialStateProvider_AddKernel_AllocatesBoundaryFaceStorage`**
  - **Description:** With `max_boundary_face_qpts > 0`, boundary face storage is allocated.
  - **Importance:** Boundary conditions with material state require this.

- [x] **`MaterialStateProvider_AddKernel_AllocatesInteriorFaceStorage`**
  - **Description:** With `max_interior_face_qpts > 0`, interior face storage is allocated.
  - **Importance:** DG methods with material state at interfaces.

- [x] **`MaterialStateProvider_GetCellState_ReturnsCorrectView`**
  - **Description:** View has correct `data_old`, `data_work`, `bytes_per_qpt`, `num_qpts`.
  - **Importance:** Kernels depend on correct view structure.

### IMaterialStateProvider Interface

- [x] **`MaterialStateProvider_ImplementsIMaterialStateProvider`**
  - **Description:** Can be used through `IMaterialStateProvider*` interface.
  - **Importance:** Enables polymorphic use in assemblers.

- [x] **`MaterialStateProvider_BeginTimeStep_Interface`**
  - **Description:** `beginTimeStep()` via interface correctly swaps buffers.
  - **Importance:** Assembler calls this through the interface.

- [x] **`MaterialStateProvider_CommitTimeStep_Interface`**
  - **Description:** `commitTimeStep()` via interface commits the current state.
  - **Importance:** Time-stepping logic uses the interface.

### Edge Cases

- [x] **`MaterialStateProvider_GetCellState_UnregisteredKernel_Behavior`**
  - **Description:** Requesting state for an unregistered kernel has defined behavior.
  - **Importance:** Prevents silent failures for misconfigured kernels.

- [x] **`MaterialStateProvider_MultipleKernels_IndependentState`**
  - **Description:** Two kernels have completely independent state storage.
  - **Importance:** Kernel isolation is critical for correctness.

- [x] **`MaterialStateProvider_LargeStateSize_HandlesCorrectly`**
  - **Description:** Kernels with large `bytes_per_qpt` (e.g., 1KB) are handled.
  - **Importance:** Complex material models may have large state.

- [x] **`MaterialStateProvider_MoveSemantics_PreservesState`**
  - **Description:** Move operations preserve all kernel registrations and state data.
  - **Importance:** System reconfiguration should not lose state.

---

## 4. FieldRegistry Tests

**File to create:** Extend `test_FieldRegistry.cpp`

**Current status:** Only 1 test covering add/find/duplicate. Missing coverage for temporal metadata and accessors.

### Missing Basic Tests

- [x] **`FieldRegistry_Get_ReturnsCorrectRecord`**
  - **Description:** `get(id)` returns `FieldRecord` with correct name, space, and components.
  - **Importance:** Field metadata access is used throughout assembly.

- [x] **`FieldRegistry_Get_ThrowsForInvalidId`**
  - **Description:** `get(INVALID_FIELD_ID)` or out-of-range ID throws.
  - **Importance:** Fail-fast for invalid field references.

- [x] **`FieldRegistry_Has_ReturnsTrueForValidId`**
  - **Description:** After adding a field, `has(id)` returns `true`.
  - **Importance:** Existence checks before operations.

- [x] **`FieldRegistry_Has_ReturnsFalseForInvalidId`**
  - **Description:** `has(INVALID_FIELD_ID)` returns `false`.
  - **Importance:** Correct handling of sentinel values.

- [x] **`FieldRegistry_Size_ReturnsFieldCount`**
  - **Description:** After adding N fields, `size()` returns N.
  - **Importance:** Iteration and allocation sizing.

- [x] **`FieldRegistry_Records_ReturnsAllFields`**
  - **Description:** `records()` returns vector with all registered fields.
  - **Importance:** Bulk access for system setup.

### Temporal Metadata Tests

- [x] **`FieldRegistry_MarkTimeDependent_SetsFlags`**
  - **Description:** After `markTimeDependent(id, 1)`, the record has `time_dependent=true` and `max_time_derivative_order=1`.
  - **Importance:** Transient systems depend on this metadata.

- [x] **`FieldRegistry_MarkTimeDependent_CanIncreaseOrder`**
  - **Description:** Calling `markTimeDependent(id, 2)` after `markTimeDependent(id, 1)` updates to order 2.
  - **Importance:** Multiple kernels may require different derivative orders.

- [x] **`FieldRegistry_MarkTimeDependent_DoesNotDecreaseOrder`**
  - **Description:** Calling `markTimeDependent(id, 1)` after `markTimeDependent(id, 2)` keeps order 2.
  - **Importance:** Maximum required order should be preserved.

- [x] **`FieldRegistry_MarkTimeDependent_InvalidId_Throws`**
  - **Description:** `markTimeDependent(INVALID_FIELD_ID, 1)` throws.
  - **Importance:** Invalid operations should fail explicitly.

### Edge Cases

- [x] **`FieldRegistry_FindByName_NonExistent_ReturnsInvalidId`**
  - **Description:** `findByName("nonexistent")` returns `INVALID_FIELD_ID`.
  - **Importance:** Safe lookup without exceptions.

- [x] **`FieldRegistry_Add_EmptyName_Behavior`**
  - **Description:** Adding a field with empty name has defined behavior.
  - **Importance:** Input validation.

- [x] **`FieldRegistry_Add_NullSpace_Behavior`**
  - **Description:** Adding a field with `nullptr` space has defined behavior.
  - **Importance:** Input validation.

---

## 5. TimeIntegrator Tests

**File to create:** `test_TimeIntegrator.cpp`

**Current status:** `test_TransientDt.cpp` covers BackwardDifferenceIntegrator and BDF2Integrator. BDFIntegrator and direct coefficient verification are missing.

### BDFIntegrator Tests

- [x] **`BDFIntegrator_Order1_MatchesBackwardEuler`**
  - **Description:** BDFIntegrator(1) produces identical coefficients to backward Euler.
  - **Importance:** Validates BDF1 specialization.

- [x] **`BDFIntegrator_Order2_MatchesBDF2`**
  - **Description:** BDFIntegrator(2) produces identical coefficients to BDF2Integrator for constant step.
  - **Importance:** Validates BDF2 consistency.

- [x] **`BDFIntegrator_Order3_CorrectCoefficients`**
  - **Description:** BDFIntegrator(3) produces correct BDF3 coefficients for constant step.
  - **Importance:** BDF3 is commonly used for stiff problems.

- [x] **`BDFIntegrator_Order4_CorrectCoefficients`**
  - **Description:** BDFIntegrator(4) produces correct BDF4 coefficients.
  - **Importance:** Higher-order accuracy for smooth solutions.

- [x] **`BDFIntegrator_Order5_CorrectCoefficients`**
  - **Description:** BDFIntegrator(5) produces correct BDF5 coefficients.
  - **Importance:** Maximum supported order validation.

- [x] **`BDFIntegrator_VariableStep_CorrectCoefficients`**
  - **Description:** Variable-step BDF produces correct coefficients from step history.
  - **Importance:** Adaptive time stepping requires accurate variable-step formulas.

### Constructor Validation

- [x] **`BDFIntegrator_Order0_Throws`**
  - **Description:** BDFIntegrator(0) throws InvalidArgumentException.
  - **Importance:** Order must be positive.

- [x] **`BDFIntegrator_Order6_Throws`**
  - **Description:** BDFIntegrator(6) throws NotImplementedException.
  - **Importance:** Only orders 1-5 are supported.

- [x] **`BDFIntegrator_Name_ReturnsCorrectString`**
  - **Description:** `name()` returns "BDF1", "BDF2", etc.
  - **Importance:** Logging and diagnostics.

### BuildContext Tests

- [x] **`BackwardDifferenceIntegrator_BuildContext_Order1_Correct`**
  - **Description:** Context for first derivative has correct coefficients.
  - **Importance:** Heat equation discretization.

- [x] **`BackwardDifferenceIntegrator_BuildContext_Order2_Correct`**
  - **Description:** Context for second derivative has correct coefficients.
  - **Importance:** Wave equation discretization.

- [x] **`BackwardDifferenceIntegrator_BuildContext_Order3_Throws`**
  - **Description:** Requesting order 3 context throws NotImplementedException.
  - **Importance:** Only orders 1-2 supported by BackwardDifferenceIntegrator.

- [x] **`BDF2Integrator_BuildContext_VariableStep_CorrectCoefficients`**
  - **Description:** Variable-step BDF2 coefficients match analytical formula.
  - **Importance:** Non-uniform time stepping accuracy.

- [x] **`BDF2Integrator_BuildContext_InsufficientHistory_Throws`**
  - **Description:** BDF2 without `u_prev2` throws.
  - **Importance:** Ensures history requirements are enforced.

### Edge Cases

- [x] **`TimeIntegrator_ZeroTimeStep_Behavior`**
  - **Description:** `buildContext()` with `dt=0` has defined behavior.
  - **Importance:** Prevents division by zero.

- [x] **`TimeIntegrator_NegativeTimeStep_Behavior`**
  - **Description:** `buildContext()` with `dt<0` has defined behavior.
  - **Importance:** Time should not run backward.

- [x] **`TimeIntegrator_MissingHistoryVectors_Throws`**
  - **Description:** Multi-step methods without required history vectors throw.
  - **Importance:** Clear error messages for state setup issues.

---

## 6. SearchAccess Tests

**File to create:** `test_SearchAccess.cpp`

**Current status:** Partial coverage through `test_FieldEvaluation.cpp` and `test_SurfaceContactKernel.cpp`. Direct ISearchAccess and MeshSearchAccess tests missing.

### ISearchAccess Interface Tests

- [x] **`ISearchAccess_DefaultLocatePoint_ReturnsNotFound`**
  - **Description:** Default `locatePoint()` implementation returns `found=false`.
  - **Importance:** Base class contract.

- [x] **`ISearchAccess_DefaultNearestVertex_ReturnsNotFound`**
  - **Description:** Default `nearestVertex()` returns `found=false`.
  - **Importance:** Base class contract.

- [x] **`ISearchAccess_DefaultKNearestVertices_ReturnsEmpty`**
  - **Description:** Default `kNearestVertices()` returns empty vector.
  - **Importance:** Base class contract.

- [x] **`ISearchAccess_DefaultNearestCell_ReturnsNotFound`**
  - **Description:** Default `nearestCell()` returns `found=false`.
  - **Importance:** Base class contract.

- [x] **`ISearchAccess_DefaultClosestBoundaryPoint_ReturnsNotFound`**
  - **Description:** Default `closestBoundaryPoint()` returns `found=false`.
  - **Importance:** Base class contract.

### MeshSearchAccess Tests

- [x] **`MeshSearchAccess_Dimension_MatchesMesh`**
  - **Description:** `dimension()` returns the mesh spatial dimension.
  - **Importance:** Correct dimensionality for all queries.

- [x] **`MeshSearchAccess_Build_EnablesQueries`**
  - **Description:** After `build()`, search queries return valid results.
  - **Importance:** Acceleration structures must be built before use.

- [x] **`MeshSearchAccess_VerticesInRadius_ReturnsCorrectSet`**
  - **Description:** Query returns all and only vertices within the specified radius.
  - **Importance:** Neighbor finding for contact and nonlocal methods.

- [x] **`MeshSearchAccess_VerticesInRadius_EmptyForZeroRadius`**
  - **Description:** Radius=0 returns empty or only coincident vertices.
  - **Importance:** Edge case handling.

- [x] **`MeshSearchAccess_VerticesInRadius_AllVerticesForLargeRadius`**
  - **Description:** Very large radius returns all mesh vertices.
  - **Importance:** Boundary condition for search.

- [x] **`MeshSearchAccess_LocatePoint_FindsContainingCell`**
  - **Description:** Point inside mesh returns correct cell ID and reference coordinates.
  - **Importance:** Field evaluation depends on this.

- [x] **`MeshSearchAccess_LocatePoint_ReturnsNotFoundOutsideMesh`**
  - **Description:** Point outside mesh domain returns `found=false`.
  - **Importance:** Graceful handling of external queries.

- [x] **`MeshSearchAccess_LocatePoint_HintCellAcceleratesSearch`**
  - **Description:** Providing correct hint cell reduces search time.
  - **Importance:** Performance optimization for sequential point queries.

- [x] **`MeshSearchAccess_NearestVertex_ReturnsClosest`**
  - **Description:** Returns the vertex with minimum Euclidean distance.
  - **Importance:** Contact and interpolation algorithms.

- [x] **`MeshSearchAccess_KNearestVertices_ReturnsKClosest`**
  - **Description:** Returns exactly k vertices sorted by distance.
  - **Importance:** K-nearest neighbor algorithms.

- [x] **`MeshSearchAccess_KNearestVertices_ReturnsAllIfKExceedsCount`**
  - **Description:** If k > num_vertices, returns all vertices.
  - **Importance:** Graceful handling of over-requests.

- [x] **`MeshSearchAccess_NearestCell_ReturnsClosestCellCentroid`**
  - **Description:** Returns cell whose centroid is nearest to query point.
  - **Importance:** Cell-based spatial queries.

- [x] **`MeshSearchAccess_ClosestBoundaryPoint_ReturnsProjection`**
  - **Description:** Returns closest point on any boundary face.
  - **Importance:** Contact gap computation.

- [x] **`MeshSearchAccess_ClosestBoundaryPointOnMarker_FiltersCorrectly`**
  - **Description:** Only considers faces with specified boundary marker.
  - **Importance:** Selective contact surface queries.

### Edge Cases

- [x] **`MeshSearchAccess_QueryBeforeBuild_Behavior`**
  - **Description:** Queries before `build()` have defined behavior (empty results or throw).
  - **Importance:** Prevents use of uninitialized structures.

- [x] **`MeshSearchAccess_1DMesh_QueriesWork`**
  - **Description:** All applicable queries work on 1D line meshes.
  - **Importance:** 1D problems should be supported.

- [x] **`MeshSearchAccess_3DMesh_QueriesWork`**
  - **Description:** All queries work on 3D tetrahedral/hexahedral meshes.
  - **Importance:** Most physical problems are 3D.

---

## 7. TransientSystem Tests

**File to create:** Extend `test_TransientDt.cpp` or create `test_TransientSystem.cpp`

**Current status:** TransientSystem is tested indirectly. Direct accessor and edge case tests missing.

### Accessor Tests

- [x] **`TransientSystem_SystemAccessor_ReturnsReference`**
  - **Description:** `system()` returns reference to the underlying FESystem.
  - **Importance:** Access to DOF handler, fields, etc.

- [x] **`TransientSystem_SystemAccessor_ConstCorrectness`**
  - **Description:** Const TransientSystem returns const FESystem reference.
  - **Importance:** Const-correctness throughout API.

- [x] **`TransientSystem_IntegratorAccessor_ReturnsReference`**
  - **Description:** `integrator()` returns reference to the TimeIntegrator.
  - **Importance:** Introspection of time integration method.

### Edge Cases

- [x] **`TransientSystem_SteadyFESystem_AssemblyWorks`**
  - **Description:** TransientSystem wrapping a steady FESystem (no dt operators) assembles correctly.
  - **Importance:** Graceful handling when transient wrapper is used unnecessarily.

- [x] **`TransientSystem_NullIntegrator_Throws`**
  - **Description:** Constructor with nullptr integrator throws.
  - **Importance:** Input validation.

---

## 8. FormsInstaller Tests

**File to create:** `test_FormsInstaller.cpp`

**Current status:** Only tested through `test_NavierStokesCoupled.cpp`. Direct function tests missing.

### installResidualForm Tests

- [x] **`FormsInstaller_InstallResidualForm_RegistersKernel`**
  - **Description:** After installation, the operator has the registered kernel.
  - **Importance:** Basic functionality verification.

- [x] **`FormsInstaller_InstallResidualForm_ADModeForward`**
  - **Description:** Forward AD mode produces correct Jacobian.
  - **Importance:** Verify AD differentiation.

- [x] **`FormsInstaller_InstallResidualForm_ADModeReverse`**
  - **Description:** Reverse AD mode produces correct Jacobian.
  - **Importance:** Alternative AD mode support.

- [x] **`FormsInstaller_InstallResidualForm_InvalidFieldId_Throws`**
  - **Description:** Using INVALID_FIELD_ID throws appropriate exception.
  - **Importance:** Input validation.

### installResidualBlocks Tests

- [x] **`FormsInstaller_InstallResidualBlocks_MultipleBlocksRegistered`**
  - **Description:** Block bilinear form installs all non-empty blocks.
  - **Importance:** Multi-field coupled assembly.

- [x] **`FormsInstaller_InstallResidualBlocks_EmptyBlocksSkipped`**
  - **Description:** Empty/zero blocks do not create kernels.
  - **Importance:** Sparse block structure efficiency.

- [x] **`FormsInstaller_InstallResidualBlocks_InitializerListOverload`**
  - **Description:** Initializer list version works identically to span version.
  - **Importance:** API convenience.

### installCoupledResidual Tests

- [x] **`FormsInstaller_InstallCoupledResidual_SeparatesVectorAndMatrix`**
  - **Description:** Residual kernels contribute only to vector, Jacobian kernels only to matrix.
  - **Importance:** Correct coupled nonlinear assembly.

- [x] **`FormsInstaller_InstallCoupledResidual_StateFieldsTracked`**
  - **Description:** State fields are correctly identified for differentiation.
  - **Importance:** AD correctness for coupled systems.

### Edge Cases

- [x] **`FormsInstaller_FormWithoutDx_Behavior`**
  - **Description:** Installing a form without `.dx()` has defined behavior.
  - **Importance:** Common user error should have clear feedback.

- [x] **`FormsInstaller_MismatchedFieldSpaces_Behavior`**
  - **Description:** Test/trial fields with incompatible spaces have defined behavior.
  - **Importance:** Input validation for complex setups.

---

## 9. SystemAssembly Tests

**File to create:** `test_SystemAssembly.cpp`

**Current status:** Only tested through FESystem::assemble(). Direct assembleOperator() tests missing.

### Direct Function Tests

- [x] **`SystemAssembly_AssembleOperator_MatrixOnly`**
  - **Description:** With `want_matrix=true, want_vector=false`, only matrix is assembled.
  - **Importance:** Separate Jacobian assembly.

- [x] **`SystemAssembly_AssembleOperator_VectorOnly`**
  - **Description:** With `want_matrix=false, want_vector=true`, only vector is assembled.
  - **Importance:** Separate residual assembly.

- [x] **`SystemAssembly_AssembleOperator_BothMatrixAndVector`**
  - **Description:** Both matrix and vector assembled in single pass.
  - **Importance:** Efficient nonlinear assembly.

- [x] **`SystemAssembly_AssembleOperator_ZeroOutputsTrue_ClearsOutputs`**
  - **Description:** With `zero_outputs=true`, outputs are zeroed before assembly.
  - **Importance:** Prevents accumulation bugs.

- [x] **`SystemAssembly_AssembleOperator_ZeroOutputsFalse_Accumulates`**
  - **Description:** With `zero_outputs=false`, assembly adds to existing values.
  - **Importance:** Operator splitting methods.

### Edge Cases

- [x] **`SystemAssembly_NullMatrixWithWantMatrix_Behavior`**
  - **Description:** `want_matrix=true` with `matrix_out=nullptr` has defined behavior.
  - **Importance:** Input validation.

- [x] **`SystemAssembly_NullVectorWithWantVector_Behavior`**
  - **Description:** `want_vector=true` with `vector_out=nullptr` has defined behavior.
  - **Importance:** Input validation.

- [x] **`SystemAssembly_EmptyOperator_ReturnsSuccess`**
  - **Description:** Operator with no kernels returns successful empty assembly.
  - **Importance:** Graceful handling of trivial operators.

- [x] **`SystemAssembly_GlobalKernelsOnly_AssemblesCorrectly`**
  - **Description:** Operator with only global kernels (no cell/boundary/face terms) works.
  - **Importance:** Pure global constraint assembly.

---

## 10. FESystem Edge Case Tests

**File to create:** `test_FESystemEdgeCases.cpp`

**Current status:** `test_FESystem.cpp` covers basic functionality. Edge cases and error paths need coverage.

### Setup Edge Cases

- [x] **`FESystem_MultipleSetupCalls_RebuildsCorrectly`**
  - **Description:** Calling `setup()` multiple times rebuilds DOFs and sparsity.
  - **Importance:** System reconfiguration during simulation.

- [x] **`FESystem_SetupWithDifferentOptions_AppliesNewOptions`**
  - **Description:** Different SetupOptions in subsequent calls are respected.
  - **Importance:** Configuration flexibility.

- [x] **`FESystem_AssemblyBeforeSetup_Throws`**
  - **Description:** Assembly without prior setup throws InvalidStateException.
  - **Importance:** Clear error for incorrect usage order.

### Field Edge Cases

- [x] **`FESystem_ZeroComponentField_Behavior`**
  - **Description:** Adding a field with 0 components has defined behavior.
  - **Importance:** Input validation.

- [x] **`FESystem_VeryLargeFieldCount_HandlesCorrectly`**
  - **Description:** Adding many fields (e.g., 100) works correctly.
  - **Importance:** Complex multi-physics systems.

- [x] **`FESystem_FieldDofOffset_MultipleFields_Correct`**
  - **Description:** Offsets for N fields are correctly computed.
  - **Importance:** Block assembly depends on correct offsets.

### Constraint Edge Cases

- [x] **`FESystem_ConstraintAfterSetup_InvalidatesSetup`**
  - **Description:** Adding constraint after setup sets `isSetup()=false`.
  - **Importance:** Constraint changes require re-setup.

- [x] **`FESystem_MixedConstraintTypes_AppliedCorrectly`**
  - **Description:** Combining Dirichlet and multi-point constraints works.
  - **Importance:** Complex boundary conditions.

### Assembly Edge Cases

- [x] **`FESystem_AssembleMass_NoMassOperator_Throws`**
  - **Description:** `assembleMass()` without "mass" operator has defined behavior.
  - **Importance:** Special assembly method validation.

- [x] **`FESystem_AssembleResidual_ReturnsCorrectResult`**
  - **Description:** `assembleResidual()` uses correct request configuration.
  - **Importance:** Convenience method correctness.

- [x] **`FESystem_AssembleJacobian_ReturnsCorrectResult`**
  - **Description:** `assembleJacobian()` uses correct request configuration.
  - **Importance:** Convenience method correctness.

### Field Evaluation Edge Cases

- [x] **`FESystem_EvaluateFieldAtPoint_NoSearchAccess_ReturnsNullopt`**
  - **Description:** Without search access configured, returns `nullopt`.
  - **Importance:** Graceful degradation.

- [x] **`FESystem_EvaluateFieldAtPoint_PointOutsideMesh_ReturnsNullopt`**
  - **Description:** Point outside mesh domain returns `nullopt`.
  - **Importance:** Safe handling of external queries.

- [x] **`FESystem_EvaluateFieldAtPoint_VectorField_ReturnsAllComponents`**
  - **Description:** Multi-component field evaluation returns all components.
  - **Importance:** Vector field support.

### Time-Stepping Edge Cases

- [x] **`FESystem_BeginCommitTimeStep_WithoutMaterialState_NoOp`**
  - **Description:** Time-step methods work even without stateful kernels.
  - **Importance:** Optional state management.

- [x] **`FESystem_DoubleBeginTimeStep_Behavior`**
  - **Description:** Calling `beginTimeStep()` twice without commit has defined behavior.
  - **Importance:** Prevents state corruption.

---

## 11. ParameterRegistry Extended Tests

**File to create:** Extend `test_ParameterRegistry.cpp`

**Current status:** 2 tests for defaults and type mismatch. Missing validation and getter tests.

### Specification Management

- [x] **`ParameterRegistry_Clear_RemovesAllSpecs`**
  - **Description:** After `clear()`, registry has no specs.
  - **Importance:** Registry reset for reconfiguration.

- [x] **`ParameterRegistry_Add_DuplicateKey_Behavior`**
  - **Description:** Adding spec with existing key has defined behavior.
  - **Importance:** Handle kernel parameter conflicts.

- [x] **`ParameterRegistry_AddAll_AddsMultipleSpecs`**
  - **Description:** Batch add correctly registers all specs.
  - **Importance:** Efficient kernel parameter aggregation.

- [x] **`ParameterRegistry_Find_ReturnsSpecPointer`**
  - **Description:** `find(key)` returns pointer to spec or nullptr.
  - **Importance:** Spec introspection.

- [x] **`ParameterRegistry_Specs_ReturnsAllSpecs`**
  - **Description:** `specs()` returns complete specification list.
  - **Importance:** Iteration over all parameters.

### Validation Tests

- [x] **`ParameterRegistry_Validate_MissingRequiredParameter_Throws`**
  - **Description:** State missing required parameter causes validation failure.
  - **Importance:** Early detection of configuration errors.

- [x] **`ParameterRegistry_Validate_OptionalParameterMissing_Succeeds`**
  - **Description:** Missing optional parameter with default passes validation.
  - **Importance:** Defaults should be sufficient.

### Getter Tests

- [x] **`ParameterRegistry_MakeParamGetter_ReturnsCallable`**
  - **Description:** Getter callable returns parameter values from state.
  - **Importance:** Runtime parameter access.

- [x] **`ParameterRegistry_MakeRealGetter_ReturnsCallable`**
  - **Description:** Real getter callable extracts real values.
  - **Importance:** Numeric parameter convenience.

---

## 12. SystemsExceptions Tests

**File to create:** `test_SystemsExceptions.cpp`

**Current status:** No direct tests. Exception types used but not directly tested.

### Exception Type Tests

- [x] **`SystemsException_InheritsFromFEException`**
  - **Description:** SystemsException can be caught as FEException.
  - **Importance:** Exception hierarchy correctness.

- [x] **`SystemsException_ContainsMessage`**
  - **Description:** Exception `what()` contains meaningful message.
  - **Importance:** Debugging support.

- [x] **`InvalidStateException_InheritsFromFEException`**
  - **Description:** InvalidStateException can be caught as FEException.
  - **Importance:** Exception hierarchy correctness.

- [x] **`InvalidStateException_ContainsMessage`**
  - **Description:** Exception `what()` describes the invalid state.
  - **Importance:** Debugging support.

---

## Implementation Priority

### Phase 1: High Priority (Critical Infrastructure)
1. `test_GlobalKernelStateProvider.cpp` - Contact and history-dependent materials
2. `test_MaterialStateProvider.cpp` - Element material state storage
3. `test_OperatorRegistry.cpp` - Operator management foundation

### Phase 2: Medium Priority (Core Functionality)
4. Extended `test_FieldRegistry.cpp` - Temporal metadata
5. `test_TimeIntegrator.cpp` - BDFIntegrator and coefficient accuracy
6. `test_SearchAccess.cpp` - Spatial queries

### Phase 3: Lower Priority (Completeness)
7. Extended `test_ParameterRegistry.cpp` - Validation and getters
8. `test_FormsInstaller.cpp` - Form installation
9. `test_SystemAssembly.cpp` - Direct assembly function
10. `test_FESystemEdgeCases.cpp` - Error paths
11. `test_TransientSystem.cpp` - Accessors
12. `test_SystemsExceptions.cpp` - Exception types

---

## Test Infrastructure Notes

### Common Test Utilities Needed

- Single-cell mesh builders (Quad4, Tri3, Hex8, Tet4, Line2)
- Multi-cell mesh builders with boundary markers
- Mock kernels with configurable state requirements
- State vector builders for transient tests
- Tolerance constants for numerical comparisons

### Build System Integration

All new test files should be added to:
- `Code/Source/solver/FE/Tests/Unit/Systems/CMakeLists.txt`

### Naming Conventions

- Test file: `test_<ComponentName>.cpp`
- Test suite: `<ComponentName>` (matches Google Test suite name)
- Test case: `<ComponentName>_<MethodOrScenario>_<ExpectedBehavior>`

---

*Last updated: Based on code review of Systems subfolder as of current branch state.*

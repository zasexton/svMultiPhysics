# Copyright (c) 2014-2015 The Regents of the University of California.
# All Rights Reserved.
#
# CMakeLists.txt for svMultiPhysics Finite Element (FE) Infrastructure
# This file supports both standalone builds and integration with the main solver

cmake_minimum_required(VERSION 3.12)

#------------------------------------------------------------------------------
# Project setup for standalone vs integrated builds
#------------------------------------------------------------------------------

# Set project and detect build mode
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(svMultiPhysicsFE
        VERSION 1.0.0
        DESCRIPTION "Finite Element Infrastructure for svMultiPhysics"
        LANGUAGES CXX
    )
    set(FE_STANDALONE_BUILD ON)
    message(STATUS "FE: Building in STANDALONE mode")
else()
    set(FE_STANDALONE_BUILD OFF)
    message(STATUS "FE: Building as part of svMultiPhysics")
endif()

#------------------------------------------------------------------------------
# CMake policies and settings
#------------------------------------------------------------------------------

# Set C++20 standard (required for std::span and other modern utilities)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CMake policies for modern behavior
if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)  # find_package() uses <PackageName>_ROOT variables
endif()
if(POLICY CMP0091)
    cmake_policy(SET CMP0091 NEW)  # MSVC runtime library flags abstraction
endif()

#------------------------------------------------------------------------------
# Build options
#------------------------------------------------------------------------------

# Core build options
option(FE_BUILD_SHARED "Build FE as shared library" OFF)
option(FE_BUILD_TESTS "Build FE unit tests" ${FE_STANDALONE_BUILD})
option(FE_BUILD_EXAMPLES "Build FE examples" OFF)
option(FE_BUILD_DOCS "Build FE documentation (requires Doxygen)" OFF)

# Fine-grained test toggles
option(FE_BUILD_DOFS_TESTS "Build FE Dofs module unit tests" OFF)

# Feature options
option(FE_ENABLE_MPI "Enable MPI support for distributed DOF maps" ON)
option(FE_ENABLE_EIGEN "Enable Eigen3 for optimized Math operations" OFF)
option(FE_ENABLE_TRILINOS "Enable Trilinos backend support" OFF)
option(FE_ENABLE_PETSC "Enable PETSc backend support" OFF)
option(FE_ENABLE_ASSEMBLY "Enable Assembly module (work-in-progress)" OFF)
option(FE_WITH_MESH "Enable Mesh integration convenience APIs (requires linking Mesh)" OFF)
option(FE_ENABLE_METIS "Enable METIS reorderings (internal ThirdParty)" ON)
option(FE_ENABLE_PARMETIS "Enable ParMETIS reorderings (internal ThirdParty, requires MPI)" ON)
option(FE_ENABLE_PYTHON "Build Python bindings (pybind11)" OFF)
option(FE_ENABLE_SYMENGINE "Enable SymEngine for symbolic FE spaces (pyramid H(div), wedge/pyramid H(curl))" ON)

# Testing options
option(FE_ENABLE_COVERAGE "Enable code coverage analysis" OFF)
option(FE_ENABLE_VALGRIND "Enable memory checking with Valgrind" OFF)
option(FE_USE_SYSTEM_GTEST "Use system-installed Google Test" ON)
option(FE_USE_MPI_WRAPPERS "Use MPI compiler wrappers" OFF)

# Development options
option(FE_ENABLE_WARNINGS "Enable additional compiler warnings" ON)
option(FE_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)
option(FE_ENABLE_SANITIZERS "Enable address/undefined sanitizers" OFF)

#------------------------------------------------------------------------------
# Compiler flags
#------------------------------------------------------------------------------

# Warning flags
if(FE_ENABLE_WARNINGS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(
            -Wall
            -Wextra
            -Wpedantic
            -Wcast-align
            -Wcast-qual
            -Wformat=2
            -Wformat-security
            -Wnon-virtual-dtor
            -Wold-style-cast
            -Woverloaded-virtual
            -Wshadow
            -Wsign-conversion
            -Wunused
        )
        if(FE_WARNINGS_AS_ERRORS)
            add_compile_options(-Werror)
        endif()
    elseif(MSVC)
        add_compile_options(/W4)
        if(FE_WARNINGS_AS_ERRORS)
            add_compile_options(/WX)
        endif()
    endif()
endif()

# Optimization flags
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Sanitizer flags
if(FE_ENABLE_SANITIZERS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(-fsanitize=address -fsanitize=undefined)
        add_link_options(-fsanitize=address -fsanitize=undefined)
    else()
        message(WARNING "FE: Sanitizers not supported for ${CMAKE_CXX_COMPILER_ID}")
    endif()
endif()

# Coverage flags
if(FE_ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(--coverage -fprofile-arcs -ftest-coverage)
        add_link_options(--coverage)
    else()
        message(WARNING "FE: Coverage not supported for ${CMAKE_CXX_COMPILER_ID}")
    endif()
endif()

#------------------------------------------------------------------------------
# Include directories setup
#------------------------------------------------------------------------------

# Set up include paths for standalone builds
if(FE_STANDALONE_BUILD)
    # Need to access Mesh library headers
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

#------------------------------------------------------------------------------
# Optional Mesh integration (build + link Mesh when enabled)
#------------------------------------------------------------------------------

if(FE_WITH_MESH)
    # In standalone FE builds, bring in the Mesh library as a subproject so the
    # Mesh-based convenience overloads can link successfully.
    if(NOT TARGET svmesh AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../Mesh/CMakeLists.txt)
        # Keep Mesh deterministic and lightweight by default when built as a dependency.
        if(NOT DEFINED MESH_ENABLE_VTK)
            set(MESH_ENABLE_VTK OFF CACHE BOOL "Disable VTK for Mesh when building via FE" )
        endif()
        if(NOT DEFINED MESH_ENABLE_EIGEN)
            set(MESH_ENABLE_EIGEN ON CACHE BOOL "Enable Eigen for Mesh when building via FE" )
        endif()
        if(NOT DEFINED USE_SYSTEM_EIGEN)
            set(USE_SYSTEM_EIGEN ON CACHE BOOL "Use system Eigen for Mesh when building via FE" )
        endif()
        if(NOT DEFINED MESH_ENABLE_MPI)
            set(MESH_ENABLE_MPI ${FE_ENABLE_MPI} CACHE BOOL "Enable MPI for Mesh when building via FE" )
        endif()
        if(NOT DEFINED MESH_BUILD_TESTS)
            set(MESH_BUILD_TESTS OFF CACHE BOOL "Disable Mesh tests when building via FE" )
        endif()
        if(NOT DEFINED MESH_BUILD_SHARED)
            set(MESH_BUILD_SHARED OFF CACHE BOOL "Build Mesh as static when building via FE" )
        endif()

        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../Mesh ${CMAKE_CURRENT_BINARY_DIR}/mesh-build)
    endif()
endif()

#------------------------------------------------------------------------------
# Source file organization (no sub-CMakeLists.txt files)
#------------------------------------------------------------------------------

# Core module sources
set(FE_CORE_HEADERS
    Core/Types.h
    Core/FEConfig.h
    Core/FEException.h
    Core/Logger.h
)

set(FE_CORE_SOURCES
    Core/Logger.cpp
)

# Math module sources
set(FE_MATH_HEADERS
    Math/TypeTraits.h
    Math/Vector.h
    Math/Matrix.h
    Math/VoigtNotation.h
    Math/Rotations.h
    Math/Eigensolvers.h
    Math/Interpolation.h
    Math/QuadratureSupport.h
    Math/SIMD.h
    Math/ExpressionOps.h
    Math/VectorExpr.h
    Math/MatrixExpr.h
    Math/Tensor.h
    Math/MathUtils.h
    Math/MathConstants.h
    Math/LU.h
)

# NOTE: These modules are planned but not yet implemented
# They are commented out until the files are created

# Basis function sources
	set(FE_BASIS_HEADERS
	    Basis/BasisFunction.h
	    Basis/LagrangeBasis.h
	    Basis/HierarchicalBasis.h
	    Basis/OrthogonalPolynomials.h
	    Basis/VectorBasis.h
	    Basis/TensorBasis.h
	    Basis/BernsteinBasis.h
	    Basis/BSplineBasis.h
	    Basis/SpectralBasis.h
	    Basis/SerendipityBasis.h
	    Basis/BasisCache.h
	    Basis/BatchEvaluator.h
	    Basis/BasisFactory.h
	    Basis/HermiteBasis.h
	    Basis/ModalTransform.h
	    Basis/NodeOrderingConventions.h
	)

	set(FE_BASIS_SOURCES
	    Basis/BasisFunction.cpp
	    Basis/LagrangeBasis.cpp
	    Basis/HierarchicalBasis.cpp
	    Basis/OrthogonalPolynomials.cpp
	    Basis/VectorBasis.cpp
	    Basis/TensorBasis.cpp
	    Basis/BernsteinBasis.cpp
	    Basis/BSplineBasis.cpp
	    Basis/SpectralBasis.cpp
	    Basis/SerendipityBasis.cpp
	    Basis/BasisCache.cpp
	    Basis/BatchEvaluator.cpp
	    Basis/BasisFactory.cpp
	    Basis/HermiteBasis.cpp
	    Basis/ModalTransform.cpp
	    Basis/NodeOrderingConventions.cpp
	)

# Quadrature sources
set(FE_QUADRATURE_HEADERS
    Quadrature/QuadratureRule.h
    Quadrature/GaussQuadrature.h
    Quadrature/GaussLobattoQuadrature.h
    Quadrature/TriangleQuadrature.h
    Quadrature/TetrahedronQuadrature.h
    Quadrature/QuadrilateralQuadrature.h
    Quadrature/HexahedronQuadrature.h
    Quadrature/WedgeQuadrature.h
    Quadrature/PyramidQuadrature.h
    Quadrature/CompositeQuadrature.h
    Quadrature/AdaptiveQuadrature.h
    Quadrature/SingularQuadrature.h
    Quadrature/SurfaceQuadrature.h
    Quadrature/QuadratureFactory.h
    Quadrature/QuadratureCache.h
    Quadrature/PositionBasedParams.h
    Quadrature/PositionBasedTriangleQuadrature.h
    Quadrature/PositionBasedTetrahedronQuadrature.h
    Quadrature/SymmetricTriangleQuadrature.h
    Quadrature/SymmetricTetrahedronQuadrature.h
)

set(FE_QUADRATURE_SOURCES
    Quadrature/GaussQuadrature.cpp
    Quadrature/GaussLobattoQuadrature.cpp
    Quadrature/TriangleQuadrature.cpp
    Quadrature/TetrahedronQuadrature.cpp
    Quadrature/QuadrilateralQuadrature.cpp
    Quadrature/HexahedronQuadrature.cpp
    Quadrature/WedgeQuadrature.cpp
    Quadrature/PyramidQuadrature.cpp
    Quadrature/CompositeQuadrature.cpp
    Quadrature/AdaptiveQuadrature.cpp
    Quadrature/SingularQuadrature.cpp
    Quadrature/SurfaceQuadrature.cpp
    Quadrature/QuadratureFactory.cpp
    Quadrature/QuadratureCache.cpp
    Quadrature/PositionBasedTriangleQuadrature.cpp
    Quadrature/PositionBasedTetrahedronQuadrature.cpp
    Quadrature/SymmetricTriangleQuadrature.cpp
    Quadrature/SymmetricTetrahedronQuadrature.cpp
)

# Element sources
set(FE_ELEMENTS_HEADERS
    Elements/Element.h
    Elements/ReferenceElement.h
    Elements/LagrangeElement.h
    Elements/DiscontinuousElement.h
    Elements/VectorElement.h
    Elements/SpectralElement.h
    Elements/IsogeometricElement.h
    Elements/MixedElement.h
    Elements/CompositeElement.h
    Elements/ElementTransform.h
    Elements/ElementFactory.h
    Elements/ElementCache.h
    Elements/ElementValidator.h
)

set(FE_ELEMENTS_SOURCES
    Elements/ReferenceElement.cpp
    Elements/LagrangeElement.cpp
    Elements/VectorElement.cpp
    Elements/SpectralElement.cpp
    Elements/IsogeometricElement.cpp
    Elements/MixedElement.cpp
    Elements/CompositeElement.cpp
    Elements/ElementTransform.cpp
    Elements/ElementFactory.cpp
    Elements/ElementCache.cpp
    Elements/ElementValidator.cpp
)

set(FE_SPACES_HEADERS
    Spaces/FunctionSpace.h
    Spaces/H1Space.h
    Spaces/C1Space.h
    Spaces/HCurlSpace.h
    Spaces/HDivSpace.h
    Spaces/L2Space.h
    Spaces/ProductSpace.h
    Spaces/MixedSpace.h
    Spaces/TraceSpace.h
    Spaces/CompositeSpace.h
    Spaces/EnrichedSpace.h
    Spaces/AdaptiveSpace.h
    Spaces/SpaceFactory.h
    Spaces/SpaceInterpolation.h
    Spaces/SpaceCompatibility.h
    Spaces/VectorComponentExtractor.h
    Spaces/DGOperators.h
    Spaces/FaceRestriction.h
    Spaces/OrientationManager.h
    Spaces/SpaceCache.h
    Spaces/SpaceWorkspace.h
    Spaces/IsogeometricSpace.h
)

set(FE_SPACES_SOURCES
    Spaces/FunctionSpace.cpp
    Spaces/H1Space.cpp
    Spaces/C1Space.cpp
    Spaces/L2Space.cpp
    Spaces/HCurlSpace.cpp
    Spaces/HDivSpace.cpp
    Spaces/ProductSpace.cpp
    Spaces/MixedSpace.cpp
    Spaces/TraceSpace.cpp
    Spaces/CompositeSpace.cpp
    Spaces/EnrichedSpace.cpp
    Spaces/AdaptiveSpace.cpp
    Spaces/SpaceFactory.cpp
    Spaces/SpaceInterpolation.cpp
    Spaces/SpaceCompatibility.cpp
    Spaces/VectorComponentExtractor.cpp
    Spaces/DGOperators.cpp
    Spaces/FaceRestriction.cpp
    Spaces/OrientationManager.cpp
    Spaces/SpaceCache.cpp
    Spaces/SpaceWorkspace.cpp
    Spaces/IsogeometricSpace.cpp
)

# DOF management sources
set(FE_DOFS_HEADERS
    Dofs/DofMap.h
    Dofs/DofIndexSet.h
    Dofs/DofHandler.h
    Dofs/DofConstraints.h
    Dofs/ConstrainedAssembly.h
    Dofs/EntityDofMap.h
    Dofs/DofTools.h
    Dofs/DofNumbering.h
    Dofs/MeshTopologyBuilder.h
    Dofs/FieldDofMap.h
    Dofs/SubspaceView.h
    Dofs/DofGraph.h
    Dofs/BlockDofMap.h
    Dofs/GhostDofManager.h
)

set(FE_DOFS_SOURCES
    Dofs/DofMap.cpp
    Dofs/DofIndexSet.cpp
    Dofs/DofHandler.cpp
    Dofs/DofConstraints.cpp
    Dofs/ConstrainedAssembly.cpp
    Dofs/EntityDofMap.cpp
    Dofs/DofTools.cpp
    Dofs/DofNumbering.cpp
    Dofs/MeshTopologyBuilder.cpp
    Dofs/FieldDofMap.cpp
    Dofs/SubspaceView.cpp
    Dofs/DofGraph.cpp
    Dofs/BlockDofMap.cpp
    Dofs/GhostDofManager.cpp
)

# Constraint sources
set(FE_CONSTRAINTS_HEADERS
    Constraints/AffineConstraints.h
    Constraints/Constraint.h
    Constraints/DirichletBC.h
    Constraints/NeumannBC.h
    Constraints/PeriodicBC.h
    Constraints/HangingNodeConstraint.h
    Constraints/MultiPointConstraint.h
    Constraints/GlobalConstraint.h
    Constraints/ConstraintDistributor.h
    Constraints/ParallelConstraints.h
    Constraints/ConstraintTools.h
    Constraints/ConstraintTransform.h
    Constraints/LagrangeMultiplier.h
    Constraints/PenaltyMethod.h
    Constraints/RobinBC.h
    Constraints/ConstraintsIO.h
)

set(FE_CONSTRAINTS_SOURCES
    Constraints/AffineConstraints.cpp
    Constraints/DirichletBC.cpp
    Constraints/NeumannBC.cpp
    Constraints/PeriodicBC.cpp
    Constraints/HangingNodeConstraint.cpp
    Constraints/MultiPointConstraint.cpp
    Constraints/GlobalConstraint.cpp
    Constraints/ConstraintDistributor.cpp
    Constraints/ParallelConstraints.cpp
    Constraints/ConstraintTools.cpp
    Constraints/ConstraintTransform.cpp
    Constraints/LagrangeMultiplier.cpp
    Constraints/PenaltyMethod.cpp
    Constraints/RobinBC.cpp
    Constraints/ConstraintsIO.cpp
)

# Assembly sources
set(FE_ASSEMBLY_HEADERS
    # Phase 1: CORE
    Assembly/Assembler.h
    Assembly/GlobalSystemView.h
    Assembly/AssemblyKernel.h
    Assembly/AssemblyContext.h
    Assembly/StandardAssembler.h
    Assembly/ParallelAssembler.h
    Assembly/GhostContributionManager.h
    # Phase 2: HIGH priority
    Assembly/AssemblyLoop.h
    Assembly/AssemblyConstraintDistributor.h
    Assembly/NonlinearAssemblyDriver.h
    Assembly/ColoredAssembler.h
    Assembly/MatrixFreeAssembler.h
    Assembly/BlockAssembler.h
    # Phase 3: MEDIUM priority
    Assembly/FunctionalAssembler.h
    Assembly/WorkStreamAssembler.h
    Assembly/CachedAssembler.h
    Assembly/VectorizationHelper.h
    Assembly/SymbolicAssembler.h
    Assembly/DeviceAssembler.h
    # Phase 4: LOW priority / Advanced
    Assembly/AssemblyScheduler.h
    Assembly/AssemblyStatistics.h
)

set(FE_ASSEMBLY_SOURCES
    # Phase 1: CORE
    Assembly/GlobalSystemView.cpp
    Assembly/AssemblyKernel.cpp
    Assembly/AssemblyContext.cpp
    Assembly/StandardAssembler.cpp
    Assembly/ParallelAssembler.cpp
    Assembly/GhostContributionManager.cpp
    # Phase 2: HIGH priority
    Assembly/AssemblyLoop.cpp
    Assembly/AssemblyConstraintDistributor.cpp
    Assembly/NonlinearAssemblyDriver.cpp
    Assembly/ColoredAssembler.cpp
    Assembly/MatrixFreeAssembler.cpp
    Assembly/BlockAssembler.cpp
    # Phase 3: MEDIUM priority
    Assembly/FunctionalAssembler.cpp
    Assembly/WorkStreamAssembler.cpp
    Assembly/CachedAssembler.cpp
    Assembly/VectorizationHelper.cpp
    Assembly/SymbolicAssembler.cpp
    Assembly/DeviceAssembler.cpp
    # Phase 4: LOW priority / Advanced
    Assembly/AssemblyScheduler.cpp
    Assembly/AssemblyStatistics.cpp
)

if(FE_WITH_MESH)
    list(APPEND FE_ASSEMBLY_HEADERS
        Assembly/MeshBaseAccess.h
    )
    list(APPEND FE_ASSEMBLY_SOURCES
        Assembly/MeshBaseAccess.cpp
    )
endif()

# Geometry sources
set(FE_GEOMETRY_HEADERS
    Geometry/GeometryMapping.h
    Geometry/GeometryFrameUtils.h
    Geometry/LinearMapping.h
    Geometry/IsoparametricMapping.h
    Geometry/SubparametricMapping.h
    Geometry/SuperparametricMapping.h
    Geometry/JacobianCache.h
    Geometry/MetricTensor.h
    Geometry/SurfaceGeometry.h
    Geometry/CurveGeometry.h
    Geometry/InverseMapping.h
    Geometry/GeometryValidator.h
    Geometry/MappingFactory.h
    Geometry/PushForward.h
    Geometry/GeometryQuadrature.h
)

set(FE_GEOMETRY_SOURCES
    Geometry/LinearMapping.cpp
    Geometry/IsoparametricMapping.cpp
    Geometry/InverseMapping.cpp
    Geometry/JacobianCache.cpp
    Geometry/MetricTensor.cpp
    Geometry/SurfaceGeometry.cpp
    Geometry/CurveGeometry.cpp
    Geometry/GeometryValidator.cpp
    Geometry/MappingFactory.cpp
    Geometry/PushForward.cpp
    Geometry/GeometryQuadrature.cpp
)

# Multi-field system sources (TODO: implement)
set(FE_SYSTEMS_HEADERS
    # Systems/MultiFieldLayout.h
    # Systems/BlockStructure.h
    # Systems/FieldCoupling.h
)

# Sparsity pattern sources
set(FE_SPARSITY_HEADERS
    # Phase 1: CORE
    Sparsity/SparsityPattern.h
    Sparsity/DistributedSparsityPattern.h
    Sparsity/SparsityBuilder.h
    Sparsity/ParallelSparsity.h
    # Phase 2: HIGH priority
    Sparsity/ConstraintSparsityAugmenter.h
    Sparsity/DGSparsityBuilder.h
    Sparsity/SparsityPreallocation.h
    Sparsity/BlockSparsity.h
    Sparsity/CompressedSparsity.h
    Sparsity/SparsityFormat.h
    # Phase 3: MEDIUM priority
    Sparsity/SparsityTwoPassBuilder.h
    Sparsity/SparsityOps.h
    Sparsity/GraphSparsity.h
    Sparsity/SparsityOptimizer.h
    Sparsity/SparsityFactory.h
    # Phase 4: LOW priority / Advanced
    Sparsity/SparsityCache.h
    Sparsity/SparsityAnalyzer.h
    Sparsity/AdaptiveSparsity.h
)

set(FE_SPARSITY_SOURCES
    # Phase 1: CORE
    Sparsity/SparsityPattern.cpp
    Sparsity/DistributedSparsityPattern.cpp
    Sparsity/SparsityBuilder.cpp
    Sparsity/ParallelSparsity.cpp
    # Phase 2: HIGH priority
    Sparsity/ConstraintSparsityAugmenter.cpp
    Sparsity/DGSparsityBuilder.cpp
    Sparsity/SparsityPreallocation.cpp
    Sparsity/BlockSparsity.cpp
    Sparsity/CompressedSparsity.cpp
    Sparsity/SparsityFormat.cpp
    # Phase 3: MEDIUM priority
    Sparsity/SparsityTwoPassBuilder.cpp
    Sparsity/SparsityOps.cpp
    Sparsity/GraphSparsity.cpp
    Sparsity/SparsityOptimizer.cpp
    Sparsity/SparsityFactory.cpp
    # Phase 4: LOW priority / Advanced
    Sparsity/SparsityCache.cpp
    Sparsity/SparsityAnalyzer.cpp
    Sparsity/AdaptiveSparsity.cpp
)

# Backend interface sources (TODO: implement)
set(FE_BACKENDS_HEADERS
    # Backends/BackendInterface.h
    # Backends/EigenBackend.h
    # Backends/TrilinosBackend.h
    # Backends/PETScBackend.h
)

# Collect all sources
set(FE_ALL_HEADERS
    ${FE_CORE_HEADERS}
    ${FE_MATH_HEADERS}
    ${FE_BASIS_HEADERS}
    ${FE_QUADRATURE_HEADERS}
    ${FE_ELEMENTS_HEADERS}
    ${FE_SPACES_HEADERS}
    ${FE_DOFS_HEADERS}
    ${FE_CONSTRAINTS_HEADERS}
    ${FE_GEOMETRY_HEADERS}
    ${FE_SYSTEMS_HEADERS}
    ${FE_SPARSITY_HEADERS}
    ${FE_BACKENDS_HEADERS}
)

set(FE_ALL_SOURCES
    ${FE_CORE_SOURCES}
    ${FE_BASIS_SOURCES}
    ${FE_QUADRATURE_SOURCES}
    ${FE_ELEMENTS_SOURCES}
    ${FE_SPACES_SOURCES}
    ${FE_DOFS_SOURCES}
    ${FE_CONSTRAINTS_SOURCES}
    ${FE_GEOMETRY_SOURCES}
    ${FE_SPARSITY_SOURCES}
)

if(FE_ENABLE_ASSEMBLY)
    list(APPEND FE_ALL_HEADERS ${FE_ASSEMBLY_HEADERS})
    list(APPEND FE_ALL_SOURCES ${FE_ASSEMBLY_SOURCES})
endif()

#------------------------------------------------------------------------------
# Configure dependencies via modular CMake scripts
#------------------------------------------------------------------------------

# Include modular configuration scripts from cmake/ directory
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Configure MPI if enabled
if(FE_ENABLE_MPI)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/EnableMPI.cmake)
endif()

# Configure Eigen if enabled
if(FE_ENABLE_EIGEN)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/EnableEigen.cmake)
endif()

# Configure SymEngine if enabled
if(FE_ENABLE_SYMENGINE)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/EnableSymEngine.cmake)
endif()

# Configure Google Test if tests are enabled
if(FE_BUILD_TESTS)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/EnableGTest.cmake)
endif()

#------------------------------------------------------------------------------
# Create the FE library target
#------------------------------------------------------------------------------

# Determine library type
if(FE_BUILD_SHARED)
    add_library(svfe SHARED ${FE_ALL_HEADERS} ${FE_ALL_SOURCES})
    set_target_properties(svfe PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
    )
else()
    add_library(svfe STATIC ${FE_ALL_HEADERS} ${FE_ALL_SOURCES})
endif()

# Add an alias for consistency
add_library(svfe::svfe ALIAS svfe)

#------------------------------------------------------------------------------
# Minimal FE library for DOFs (Core + Math + Quadrature + ReferenceElement +
# OrientationManager + Dofs). This enables building/testing Dofs independently
# of higher-level modules that may still be under active development.
#------------------------------------------------------------------------------

set(FE_DOFS_MINIMAL_HEADERS
    ${FE_CORE_HEADERS}
    ${FE_MATH_HEADERS}
    Basis/BasisFunction.h
    Basis/LagrangeBasis.h
    Basis/OrthogonalPolynomials.h
    Basis/NodeOrderingConventions.h
    ${FE_QUADRATURE_HEADERS}
    Elements/ReferenceElement.h
    Spaces/OrientationManager.h
    ${FE_DOFS_HEADERS}
)

set(FE_DOFS_MINIMAL_SOURCES
    ${FE_CORE_SOURCES}
    Basis/BasisFunction.cpp
    Basis/OrthogonalPolynomials.cpp
    Basis/LagrangeBasis.cpp
    Basis/NodeOrderingConventions.cpp
    ${FE_QUADRATURE_SOURCES}
    Elements/ReferenceElement.cpp
    Spaces/OrientationManager.cpp
    ${FE_DOFS_SOURCES}
)

add_library(svfe_dofs STATIC ${FE_DOFS_MINIMAL_HEADERS} ${FE_DOFS_MINIMAL_SOURCES})
add_library(svfe::dofs ALIAS svfe_dofs)

set_target_properties(svfe_dofs PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(svfe_dofs
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<INSTALL_INTERFACE:include/svfe>
        $<INSTALL_INTERFACE:include>
)

target_compile_definitions(svfe_dofs PUBLIC SVMP_FE_WITH_MESH=$<BOOL:${FE_WITH_MESH}>)

if(FE_WITH_MESH)
    if(TARGET svmesh)
        target_link_libraries(svfe_dofs PUBLIC svmesh)
    else()
        message(WARNING "FE_WITH_MESH is ON but target 'svmesh' was not found; Mesh-based convenience APIs may fail to link.")
    endif()
endif()

if(FE_ENABLE_MPI)
    if(TARGET MPI::MPI_CXX)
        target_link_libraries(svfe_dofs PUBLIC MPI::MPI_CXX)
    elseif(DEFINED FE_MPI_CXX_LIBRARIES)
        target_include_directories(svfe_dofs PUBLIC ${FE_MPI_CXX_INCLUDE_DIRS})
        target_link_libraries(svfe_dofs PUBLIC ${FE_MPI_CXX_LIBRARIES})
    endif()
    target_compile_definitions(svfe_dofs PUBLIC FE_HAS_MPI MESH_ENABLE_MPI MPICH_SKIP_MPICXX OMPI_SKIP_MPICXX _MPICC_H)
endif()

# Set target properties
set_target_properties(svfe PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

# Configure include directories
target_include_directories(svfe
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<INSTALL_INTERFACE:include/svfe>
        $<INSTALL_INTERFACE:include>
)

target_compile_definitions(svfe PUBLIC SVMP_FE_WITH_MESH=$<BOOL:${FE_WITH_MESH}>)

if(FE_WITH_MESH)
    if(TARGET svmesh)
        target_link_libraries(svfe PUBLIC svmesh)
    else()
        message(WARNING "FE_WITH_MESH is ON but target 'svmesh' was not found; Mesh-based convenience APIs may fail to link.")
    endif()
endif()

# Link dependencies are configured by the Enable*.cmake scripts.
#
# MPI is a special case: EnableMPI.cmake is included before the `svfe` target
# exists, so we apply the discovered MPI configuration here.
if(FE_ENABLE_MPI)
    if(TARGET MPI::MPI_CXX)
        target_link_libraries(svfe PUBLIC MPI::MPI_CXX)
    elseif(DEFINED FE_MPI_CXX_LIBRARIES)
        target_include_directories(svfe PUBLIC ${FE_MPI_CXX_INCLUDE_DIRS})
        target_link_libraries(svfe PUBLIC ${FE_MPI_CXX_LIBRARIES})
    endif()
    # Disable removed MPI-2 C++ bindings in common MPI implementations to avoid
    # pulling deprecated headers (and associated warnings) through <mpi.h>.
    target_compile_definitions(svfe PUBLIC FE_HAS_MPI MESH_ENABLE_MPI MPICH_SKIP_MPICXX OMPI_SKIP_MPICXX _MPICC_H)
endif()

#------------------------------------------------------------------------------
# Optional internal graph reordering libraries (METIS / ParMETIS)
#------------------------------------------------------------------------------

set(_fe_use_internal_metis ${FE_ENABLE_METIS})
if(FE_ENABLE_MPI AND FE_ENABLE_PARMETIS)
    # ParMETIS depends on METIS+GKlib in the bundled ThirdParty stack.
    set(_fe_use_internal_metis ON)
endif()

if(_fe_use_internal_metis)
    enable_language(C)

    set(_fe_thirdparty_dir "${CMAKE_CURRENT_SOURCE_DIR}/../../../ThirdParty")

    if(NOT GKLIB_INTERNAL_LIBRARY_NAME)
        set(GKLIB_INTERNAL_LIBRARY_NAME gklib_internal)
    endif()
    if(NOT METIS_INTERNAL_LIBRARY_NAME)
        set(METIS_INTERNAL_LIBRARY_NAME metis_internal)
    endif()
    if(NOT PARMETIS_INTERNAL_LIBRARY_NAME)
        set(PARMETIS_INTERNAL_LIBRARY_NAME parmetis_internal)
    endif()

    # Legacy ThirdParty CMake expects MPI_C_INCLUDE_PATH; map it from the
    # MPI configuration performed by EnableMPI.cmake when needed.
    if(FE_ENABLE_MPI AND (NOT DEFINED MPI_C_INCLUDE_PATH OR MPI_C_INCLUDE_PATH STREQUAL ""))
        if(DEFINED MPI_CXX_INCLUDE_DIRS)
            set(MPI_C_INCLUDE_PATH "${MPI_CXX_INCLUDE_DIRS}")
        elseif(DEFINED FE_MPI_CXX_INCLUDE_DIRS)
            set(MPI_C_INCLUDE_PATH "${FE_MPI_CXX_INCLUDE_DIRS}")
        endif()
    endif()

    if(NOT TARGET ${GKLIB_INTERNAL_LIBRARY_NAME})
        add_subdirectory(
            "${_fe_thirdparty_dir}/gklib_internal/simvascular_gklib_internal"
            "${CMAKE_BINARY_DIR}/thirdparty/gklib_internal"
        )
    endif()

    if(NOT TARGET ${METIS_INTERNAL_LIBRARY_NAME})
        add_subdirectory(
            "${_fe_thirdparty_dir}/metis_internal/simvascular_metis_internal"
            "${CMAKE_BINARY_DIR}/thirdparty/metis_internal"
        )
    endif()

    # METIS depends on GKlib symbols; link both (order matters for static libs).
    target_link_libraries(svfe PRIVATE ${METIS_INTERNAL_LIBRARY_NAME} ${GKLIB_INTERNAL_LIBRARY_NAME})
    target_include_directories(svfe PRIVATE
        "${_fe_thirdparty_dir}/metis_internal/simvascular_metis_internal/METISLib"
    )
    target_compile_definitions(svfe PRIVATE SVMP_HAS_METIS)

    if(FE_ENABLE_MPI AND FE_ENABLE_PARMETIS)
        if(NOT TARGET ${PARMETIS_INTERNAL_LIBRARY_NAME})
            add_subdirectory(
                "${_fe_thirdparty_dir}/parmetis_internal/simvascular_parmetis_internal"
                "${CMAKE_BINARY_DIR}/thirdparty/parmetis_internal"
            )
        endif()

        target_link_libraries(svfe PRIVATE ${PARMETIS_INTERNAL_LIBRARY_NAME})
        target_include_directories(svfe PRIVATE
            "${_fe_thirdparty_dir}/parmetis_internal/simvascular_parmetis_internal/ParMETISLib"
        )
        target_compile_definitions(svfe PRIVATE SVMP_HAS_PARMETIS)
    endif()
endif()

#------------------------------------------------------------------------------
# Build tests if enabled (no sub-CMakeLists.txt)
#------------------------------------------------------------------------------

if(FE_BUILD_TESTS)
    # Math module tests
    set(FE_MATH_TEST_SOURCES
        Tests/Unit/Math/test_TypeTraits.cpp
        Tests/Unit/Math/test_Vector.cpp
        Tests/Unit/Math/test_Matrix.cpp
        Tests/Unit/Math/test_VoigtNotation.cpp
        Tests/Unit/Math/test_LU.cpp
        Tests/Unit/Math/test_SIMD.cpp
        Tests/Unit/Math/test_Eigensolvers.cpp  # Re-enabled - wrapper functions implemented
        Tests/Unit/Math/test_Tensor.cpp  # Re-enabled - helper functions implemented
        Tests/Unit/Math/test_MathConstants.cpp  # Re-enabled - constants namespace implemented
        Tests/Unit/Math/test_MathUtils.cpp  # Mathematical utility functions
        Tests/Unit/Math/test_Rotations.cpp  # Rotation matrices and transformations
        Tests/Unit/Math/test_Interpolation.cpp  # Interpolation utilities
        Tests/Unit/Math/test_QuadratureSupport.cpp  # Quadrature and integration support
        Tests/Unit/Math/test_main.cpp
    )

    # Create Math test executable
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/Tests/Unit/Math/test_main.cpp)
        add_executable(test_fe_math ${FE_MATH_TEST_SOURCES})
        target_link_libraries(test_fe_math
            PRIVATE
                svfe
                ${GTEST_LIBRARIES}
        )
        add_test(NAME FE_Math_Tests COMMAND test_fe_math)
    endif()

    # Core module tests
    file(GLOB FE_CORE_TEST_SOURCES Tests/Unit/Core/test_*.cpp)
    if(FE_CORE_TEST_SOURCES)
        add_executable(test_fe_core ${FE_CORE_TEST_SOURCES})
        target_link_libraries(test_fe_core
            PRIVATE
                svfe
                ${GTEST_LIBRARIES}
        )
        add_test(NAME FE_Core_Tests COMMAND test_fe_core)
    endif()

    # Quadrature module tests
    set(FE_QUAD_TEST_SOURCES
        Tests/Unit/Quadrature/test_GaussQuadrature.cpp
        Tests/Unit/Quadrature/test_GaussLobattoQuadrature.cpp
        Tests/Unit/Quadrature/test_TriangleQuadrature.cpp
        Tests/Unit/Quadrature/test_QuadrilateralQuadrature.cpp
        Tests/Unit/Quadrature/test_TetrahedronQuadrature.cpp
        Tests/Unit/Quadrature/test_HexahedronQuadrature.cpp
        Tests/Unit/Quadrature/test_WedgeQuadrature.cpp
        Tests/Unit/Quadrature/test_PyramidQuadrature.cpp
        Tests/Unit/Quadrature/test_QuadratureCache.cpp
        Tests/Unit/Quadrature/test_CompositeAdaptiveSingularSurface.cpp
        Tests/Unit/Quadrature/test_QuadratureFactory.cpp
        Tests/Unit/Quadrature/test_PolynomialExactnessSweep.cpp
        Tests/Unit/Quadrature/test_SurfaceQuadraturePermutations.cpp
        Tests/Unit/Quadrature/test_PositionBasedQuadrature.cpp
        Tests/Unit/Quadrature/test_SymmetricTriangleQuadrature.cpp
        Tests/Unit/Quadrature/test_SymmetricTetrahedronQuadrature.cpp
        Tests/Unit/Quadrature/test_main.cpp
    )
    add_executable(test_fe_quadrature ${FE_QUAD_TEST_SOURCES})
    target_link_libraries(test_fe_quadrature
        PRIVATE
            svfe
            ${GTEST_LIBRARIES}
    )
    add_test(NAME FE_Quadrature_Tests COMMAND test_fe_quadrature)

    # Basis module tests
    set(FE_BASIS_TEST_SOURCES
        Tests/Unit/Basis/test_LagrangeBasis.cpp
        Tests/Unit/Basis/test_HierarchicalSpectralBasis.cpp
        Tests/Unit/Basis/test_BernsteinBasis.cpp
        Tests/Unit/Basis/test_VectorBases.cpp
        Tests/Unit/Basis/test_RT_MinimalSimplex.cpp
        Tests/Unit/Basis/test_SerendipityTensorModal.cpp
        Tests/Unit/Basis/test_BasisCacheFactory.cpp
        Tests/Unit/Basis/test_BatchEvaluator.cpp
        Tests/Unit/Basis/test_OrthogonalPolynomials.cpp
        Tests/Unit/Basis/test_BasisHessians.cpp
        Tests/Unit/Basis/test_C1Basis.cpp
        Tests/Unit/Basis/test_HigherOrderWedgePyramid.cpp
        Tests/Unit/Basis/test_main.cpp
    )
    add_executable(test_fe_basis ${FE_BASIS_TEST_SOURCES})
    target_link_libraries(test_fe_basis
        PRIVATE
            svfe
            ${GTEST_LIBRARIES}
    )
    add_test(NAME FE_Basis_Tests COMMAND test_fe_basis)

    # Geometry module tests
    set(FE_GEOMETRY_TEST_SOURCES
        Tests/Unit/Geometry/test_LinearMapping.cpp
        Tests/Unit/Geometry/test_IsoparametricMapping.cpp
        Tests/Unit/Geometry/test_JacobianCacheGeometryQuadrature.cpp
        Tests/Unit/Geometry/test_MetricSurfacePushForward.cpp
        Tests/Unit/Geometry/test_HigherOrderMappings.cpp
        Tests/Unit/Geometry/test_ElementTypeCoverage.cpp
        Tests/Unit/Geometry/test_Pyramid14Mapping.cpp
        Tests/Unit/Geometry/test_InverseMapping.cpp
        Tests/Unit/Geometry/test_main.cpp
    )
    add_executable(test_fe_geometry ${FE_GEOMETRY_TEST_SOURCES})
    target_link_libraries(test_fe_geometry
        PRIVATE
            svfe
            ${GTEST_LIBRARIES}
    )
    add_test(NAME FE_Geometry_Tests COMMAND test_fe_geometry)

    # Elements module tests
    set(FE_ELEMENTS_TEST_SOURCES
        Tests/Unit/Elements/test_ConvergenceRates.cpp
        Tests/Unit/Elements/test_ElementCacheThreadSafety.cpp
        Tests/Unit/Elements/test_ReferenceElement.cpp
        Tests/Unit/Elements/test_LagrangeAndDGElements.cpp
        Tests/Unit/Elements/test_Order0Elements.cpp
        Tests/Unit/Elements/test_VectorSpectralAndComposite.cpp
        Tests/Unit/Elements/test_SpectralElementCollocation.cpp
        Tests/Unit/Elements/test_ElementFactoryCacheValidator.cpp
        Tests/Unit/Elements/test_ElementErrorPaths.cpp
        Tests/Unit/Elements/test_InterfaceContinuity.cpp
        Tests/Unit/Elements/test_main.cpp
    )
    add_executable(test_fe_elements ${FE_ELEMENTS_TEST_SOURCES})
    target_link_libraries(test_fe_elements
        PRIVATE
            svfe
            ${GTEST_LIBRARIES}
    )
    add_test(NAME FE_Elements_Tests COMMAND test_fe_elements)

    # Spaces module tests
    file(GLOB FE_SPACES_TEST_SOURCES Tests/Unit/Spaces/test_*.cpp)
    if(FE_SPACES_TEST_SOURCES)
        add_executable(test_fe_spaces ${FE_SPACES_TEST_SOURCES})
        target_link_libraries(test_fe_spaces
            PRIVATE
                svfe
                ${GTEST_LIBRARIES}
        )
        add_test(NAME FE_Spaces_Tests COMMAND test_fe_spaces)
    endif()

	# Dofs module tests
	if(FE_BUILD_DOFS_TESTS)
		    set(FE_DOFS_TEST_SOURCES
		        Tests/Unit/Dofs/test_DofIndexSet.cpp
		        Tests/Unit/Dofs/test_DofMap.cpp
		        Tests/Unit/Dofs/test_DofHandler.cpp
		        Tests/Unit/Dofs/test_EntityDofMap.cpp
		        Tests/Unit/Dofs/test_FieldDofMap.cpp
		        Tests/Unit/Dofs/test_DofConstraints.cpp
		        Tests/Unit/Dofs/test_ConstrainedAssembly.cpp
		        Tests/Unit/Dofs/test_DofGraph.cpp
	        Tests/Unit/Dofs/test_BlockDofMap.cpp
		        Tests/Unit/Dofs/test_DofNumbering.cpp
		        Tests/Unit/Dofs/test_DofTools.cpp
		        Tests/Unit/Dofs/test_MeshTopologyBuilder.cpp
		        Tests/Unit/Dofs/test_GhostDofManager.cpp
		        Tests/Unit/Dofs/test_main.cpp
		    )
		    if(FE_WITH_MESH)
		        list(APPEND FE_DOFS_TEST_SOURCES Tests/Unit/Dofs/test_DofHandlerMesh.cpp)
		    endif()
		    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/Tests/Unit/Dofs/test_main.cpp)
		        add_executable(test_fe_dofs ${FE_DOFS_TEST_SOURCES})
		        if(FE_WITH_MESH)
		            target_link_libraries(test_fe_dofs
		                PRIVATE
		                    svfe
		                    ${GTEST_LIBRARIES}
		            )
		        else()
		            target_link_libraries(test_fe_dofs
		                PRIVATE
		                    svfe_dofs
		                    ${GTEST_LIBRARIES}
		            )
		        endif()
		        add_test(NAME FE_Dofs_Tests COMMAND test_fe_dofs)
		    endif()
		endif()

    # Constraints module tests
    set(FE_CONSTRAINTS_TEST_SOURCES
        Tests/Unit/Constraints/test_AffineConstraints.cpp
        Tests/Unit/Constraints/test_DirichletBC.cpp
        Tests/Unit/Constraints/test_NeumannBC.cpp
        Tests/Unit/Constraints/test_RobinBC.cpp
        Tests/Unit/Constraints/test_ConstraintTransform.cpp
        Tests/Unit/Constraints/test_HangingNodeConstraint.cpp
        Tests/Unit/Constraints/test_PeriodicBC.cpp
        Tests/Unit/Constraints/test_MultiPointConstraint.cpp
        Tests/Unit/Constraints/test_GlobalConstraint.cpp
        Tests/Unit/Constraints/test_ParallelConstraints.cpp
        Tests/Unit/Constraints/test_LagrangeMultiplier.cpp
        Tests/Unit/Constraints/test_PenaltyMethod.cpp
        Tests/Unit/Constraints/test_ConstraintDistributor.cpp
        Tests/Unit/Constraints/test_ConstraintsIO.cpp
        Tests/Unit/Constraints/test_ConstraintTools.cpp
        Tests/Unit/Constraints/test_main.cpp
    )
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/Tests/Unit/Constraints/test_main.cpp)
        add_executable(test_fe_constraints ${FE_CONSTRAINTS_TEST_SOURCES})
        target_link_libraries(test_fe_constraints
            PRIVATE
                svfe
                ${GTEST_LIBRARIES}
        )
        add_test(NAME FE_Constraints_Tests COMMAND test_fe_constraints)
    endif()

    # Sparsity module tests
    set(FE_SPARSITY_TEST_SOURCES
        # Phase 1: CORE tests
        Tests/Unit/Sparsity/test_SparsityPattern.cpp
        Tests/Unit/Sparsity/test_SparsityPattern_Optimization.cpp
        Tests/Unit/Sparsity/test_DistributedSparsityPattern.cpp
        Tests/Unit/Sparsity/test_ParallelSparsity.cpp
        Tests/Unit/Sparsity/test_SparsityBuilder.cpp
        # Phase 2: HIGH priority tests
        Tests/Unit/Sparsity/test_ConstraintSparsityAugmenter.cpp
        Tests/Unit/Sparsity/test_DGSparsityBuilder.cpp
        Tests/Unit/Sparsity/test_SparsityPreallocation.cpp
        Tests/Unit/Sparsity/test_BlockSparsity.cpp
        Tests/Unit/Sparsity/test_CompressedSparsity.cpp
        Tests/Unit/Sparsity/test_SparsityFormat.cpp
        # Phase 3: MEDIUM priority tests
        Tests/Unit/Sparsity/test_SparsityTwoPassBuilder.cpp
        Tests/Unit/Sparsity/test_SparsityOps.cpp
        Tests/Unit/Sparsity/test_GraphSparsity.cpp
        Tests/Unit/Sparsity/test_SparsityOptimizer.cpp
        Tests/Unit/Sparsity/test_SparsityFactory.cpp
        # Phase 4: LOW priority / Advanced tests
        Tests/Unit/Sparsity/test_SparsityCache.cpp
        Tests/Unit/Sparsity/test_SparsityAnalyzer.cpp
        Tests/Unit/Sparsity/test_AdaptiveSparsity.cpp
        Tests/Unit/Sparsity/test_main.cpp
    )
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/Tests/Unit/Sparsity/test_main.cpp)
        add_executable(test_fe_sparsity ${FE_SPARSITY_TEST_SOURCES})
        target_link_libraries(test_fe_sparsity
            PRIVATE
                svfe
                ${GTEST_LIBRARIES}
        )
        add_test(NAME FE_Sparsity_Tests COMMAND test_fe_sparsity)
    endif()

    if(FE_ENABLE_ASSEMBLY)
        # Assembly module tests
        set(FE_ASSEMBLY_TEST_SOURCES
            # Phase 1: CORE tests
            Tests/Unit/Assembly/test_GlobalSystemView.cpp
            Tests/Unit/Assembly/test_AssemblyKernel.cpp
            Tests/Unit/Assembly/test_AssemblyContext.cpp
            Tests/Unit/Assembly/test_StandardAssembler.cpp
            Tests/Unit/Assembly/test_ParallelAssembler.cpp
            Tests/Unit/Assembly/test_GhostContributionManager.cpp
            # Phase 2: HIGH priority tests
            Tests/Unit/Assembly/test_AssemblyLoop.cpp
            Tests/Unit/Assembly/test_AssemblyConstraintDistributor.cpp
            Tests/Unit/Assembly/test_NonlinearAssemblyDriver.cpp
            Tests/Unit/Assembly/test_ColoredAssembler.cpp
            Tests/Unit/Assembly/test_MatrixFreeAssembler.cpp
            Tests/Unit/Assembly/test_BlockAssembler.cpp
            # Phase 3: MEDIUM priority tests
            Tests/Unit/Assembly/test_FunctionalAssembler.cpp
            Tests/Unit/Assembly/test_WorkStreamAssembler.cpp
            Tests/Unit/Assembly/test_CachedAssembler.cpp
            Tests/Unit/Assembly/test_VectorizationHelper.cpp
            Tests/Unit/Assembly/test_SymbolicAssembler.cpp
            Tests/Unit/Assembly/test_DeviceAssembler.cpp
            # Phase 4: LOW priority / Advanced tests
            Tests/Unit/Assembly/test_AssemblyScheduler.cpp
            Tests/Unit/Assembly/test_AssemblyStatistics.cpp
            Tests/Unit/Assembly/test_main.cpp
        )
        if(FE_WITH_MESH)
            list(APPEND FE_ASSEMBLY_TEST_SOURCES
                Tests/Unit/Assembly/test_MeshBaseAccess.cpp
            )
        endif()
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/Tests/Unit/Assembly/test_main.cpp)
            add_executable(test_fe_assembly ${FE_ASSEMBLY_TEST_SOURCES})
            target_link_libraries(test_fe_assembly
                PRIVATE
                    svfe
                    ${GTEST_LIBRARIES}
            )
            add_test(NAME FE_Assembly_Tests COMMAND test_fe_assembly)
        endif()
    endif()

    # Add MPI tests if MPI is enabled
    if(FE_ENABLE_MPI AND MPIEXEC_EXECUTABLE)
        file(GLOB FE_MPI_TEST_SOURCES
            Tests/Unit/Dofs/test_*MPI*.cpp
        )
        if(FE_MPI_TEST_SOURCES)
            add_executable(test_fe_mpi ${FE_MPI_TEST_SOURCES})
            # MPI tests provide their own main() to call MPI_Init/MPI_Finalize,
            # so do not link gtest_main here.
            if(TARGET GTest::gtest)
                target_link_libraries(test_fe_mpi PRIVATE svfe_dofs GTest::gtest)
            else()
                set(_fe_mpi_gtest_libs ${GTEST_LIBRARIES})
                list(REMOVE_ITEM _fe_mpi_gtest_libs gtest_main GTest::gtest_main)
                target_link_libraries(test_fe_mpi PRIVATE svfe_dofs ${_fe_mpi_gtest_libs})
            endif()
            # Add MPI test with different processor counts
            add_fe_mpi_test(test_fe_mpi 2)
            add_fe_mpi_test(test_fe_mpi 4)
        endif()

        file(GLOB FE_CONSTRAINTS_MPI_TEST_SOURCES
            Tests/Unit/Constraints/test_*MPI*.cpp
        )
        if(FE_CONSTRAINTS_MPI_TEST_SOURCES)
            add_executable(test_fe_constraints_mpi ${FE_CONSTRAINTS_MPI_TEST_SOURCES})
            # MPI tests provide their own main() to call MPI_Init/MPI_Finalize,
            # so do not link gtest_main here.
            if(TARGET GTest::gtest)
                target_link_libraries(test_fe_constraints_mpi PRIVATE svfe GTest::gtest)
            else()
                set(_fe_constraints_mpi_gtest_libs ${GTEST_LIBRARIES})
                list(REMOVE_ITEM _fe_constraints_mpi_gtest_libs gtest_main GTest::gtest_main)
                target_link_libraries(test_fe_constraints_mpi PRIVATE svfe ${_fe_constraints_mpi_gtest_libs})
            endif()
            add_fe_mpi_test(test_fe_constraints_mpi 2)
            add_fe_mpi_test(test_fe_constraints_mpi 4)
        endif()
    endif()
endif()

#------------------------------------------------------------------------------
# Build examples if enabled
#------------------------------------------------------------------------------

if(FE_BUILD_EXAMPLES)
    # Add example subdirectory (if it exists)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/Examples)
        file(GLOB FE_EXAMPLE_SOURCES Examples/*.cpp)
        foreach(example_source ${FE_EXAMPLE_SOURCES})
            get_filename_component(example_name ${example_source} NAME_WE)
            add_executable(${example_name} ${example_source})
            target_link_libraries(${example_name} PRIVATE svfe)
            set_target_properties(${example_name} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples
                FOLDER "Examples/FE"
            )
        endforeach()
    endif()
endif()

#------------------------------------------------------------------------------
# Installation (for standalone builds)
#------------------------------------------------------------------------------

if(FE_STANDALONE_BUILD)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    # Install the library
    install(TARGETS svfe
        EXPORT svfeTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    # When FE_WITH_MESH is enabled and Mesh is built as part of this build,
    # include svmesh in the export set so the generated package is consistent.
    if(FE_WITH_MESH AND TARGET svmesh)
        install(TARGETS svmesh
            EXPORT svfeTargets
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        )
    endif()

    # If FE was built with bundled METIS/ParMETIS support, export/install the
    # internal ThirdParty libraries as part of the standalone FE package.
    foreach(_fe_tp_target IN ITEMS
            ${GKLIB_INTERNAL_LIBRARY_NAME}
            ${METIS_INTERNAL_LIBRARY_NAME}
            ${PARMETIS_INTERNAL_LIBRARY_NAME})
        if(_fe_tp_target AND TARGET ${_fe_tp_target})
            install(TARGETS ${_fe_tp_target}
                EXPORT svfeTargets
                LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
                ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            )
        endif()
    endforeach()

    # Install headers preserving directory structure
    foreach(header ${FE_ALL_HEADERS})
        get_filename_component(dir ${header} DIRECTORY)
        install(FILES ${header}
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/svfe/${dir}
        )
    endforeach()

    # Generate and install export file
    install(EXPORT svfeTargets
        FILE svfeTargets.cmake
        NAMESPACE svfe::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/svfe
    )

    # Generate version file
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/svfeConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY AnyNewerVersion
    )

    # Configure and install config file
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/svfeConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/svfeConfig.cmake"
        @ONLY
    )

    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/svfeConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/svfeConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/svfe
    )

    # Install CMake modules if they might be useful for consumers
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cmake/
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/svfe/Modules
        FILES_MATCHING PATTERN "*.cmake"
        PATTERN "*.in" EXCLUDE
    )
endif()

#------------------------------------------------------------------------------
# Documentation
#------------------------------------------------------------------------------

if(FE_BUILD_DOCS)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs)
        set(DOXYGEN_GENERATE_HTML YES)
        set(DOXYGEN_GENERATE_MAN NO)
        set(DOXYGEN_EXTRACT_ALL YES)
        set(DOXYGEN_EXTRACT_PRIVATE YES)
        set(DOXYGEN_RECURSIVE YES)
        set(DOXYGEN_USE_MDFILE_AS_MAINPAGE ${CMAKE_CURRENT_SOURCE_DIR}/README.md)

        doxygen_add_docs(fe_docs
            ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Generating FE library documentation"
        )
    else()
        message(WARNING "FE: Doxygen not found, documentation will not be built")
    endif()
endif()

#------------------------------------------------------------------------------
# Summary output
#------------------------------------------------------------------------------

if(FE_STANDALONE_BUILD)
    message(STATUS "")
    message(STATUS "========================================")
    message(STATUS " svMultiPhysics FE Library Configuration")
    message(STATUS "========================================")
    message(STATUS " Version:          ${PROJECT_VERSION}")
    message(STATUS " Build type:       ${CMAKE_BUILD_TYPE}")
    message(STATUS " Install prefix:   ${CMAKE_INSTALL_PREFIX}")
    message(STATUS "")
    message(STATUS " Options:")
    message(STATUS "   Shared library:   ${FE_BUILD_SHARED}")
    message(STATUS "   Build tests:      ${FE_BUILD_TESTS}")
    message(STATUS "   Build examples:   ${FE_BUILD_EXAMPLES}")
    message(STATUS "   Build docs:       ${FE_BUILD_DOCS}")
    message(STATUS "")
    message(STATUS " Features:")
    message(STATUS "   MPI support:      ${FE_ENABLE_MPI}")
    message(STATUS "   Eigen3:           ${FE_ENABLE_EIGEN}")
    message(STATUS "   Trilinos:         ${FE_ENABLE_TRILINOS}")
    message(STATUS "   PETSc:            ${FE_ENABLE_PETSC}")
    message(STATUS "   Python bindings:  ${FE_ENABLE_PYTHON}")
    message(STATUS "")
    message(STATUS " Development:")
    message(STATUS "   Warnings:         ${FE_ENABLE_WARNINGS}")
    message(STATUS "   Warnings as err: ${FE_WARNINGS_AS_ERRORS}")
    message(STATUS "   Coverage:         ${FE_ENABLE_COVERAGE}")
    message(STATUS "   Valgrind:         ${FE_ENABLE_VALGRIND}")
    message(STATUS "   Sanitizers:       ${FE_ENABLE_SANITIZERS}")
    message(STATUS "========================================")
    message(STATUS "")
endif()

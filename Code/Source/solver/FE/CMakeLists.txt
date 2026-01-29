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
option(FE_ENABLE_LLVM_JIT "Enable LLVM OrcJIT backend for FE/Forms" OFF)
option(FE_WITH_MESH "Enable Mesh integration convenience APIs (requires linking Mesh)" OFF)
option(FE_ENABLE_METIS "Enable METIS reorderings (internal ThirdParty)" ON)
option(FE_ENABLE_PARMETIS "Enable ParMETIS reorderings (internal ThirdParty, requires MPI)" ON)
option(FE_ENABLE_PYTHON "Build Python bindings (pybind11)" OFF)
option(FE_ENABLE_SYMENGINE "Enable SymEngine for symbolic FE spaces (pyramid H(div), wedge/pyramid H(curl))" ${FE_STANDALONE_BUILD})

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

#------------------------------------------------------------------------------
# Clang/libstdc++ compatibility (Ubuntu Clang 14 vs libstdc++ 13)
#
# Ubuntu's Clang 14 cannot compile some C++20 headers from newer libstdc++
# (notably <chrono> with std::chrono::hh_mm_ss), producing errors like:
#   "call to consteval function ... is not a constant expression"
#
# Work around this by forcing Clang (<16) to use the libstdc++-11 headers.
# We keep the system libstdc++ runtime (typically newer) for linking.
#------------------------------------------------------------------------------
set(_SVMP_FE_IS_CLANG FALSE)
set(_SVMP_FE_CLANG_MAJOR 0)

# Prefer CMake's compiler ID when accurate.
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(_SVMP_FE_IS_CLANG TRUE)
    if(CMAKE_CXX_COMPILER_VERSION MATCHES "^([0-9]+)\\.")
        set(_SVMP_FE_CLANG_MAJOR "${CMAKE_MATCH_1}")
    endif()
endif()

# Fallback: if the compiler at CMAKE_CXX_COMPILER changed since the build dir
# was generated (e.g., /usr/bin/c++ switched from g++ to clang), CMake's cached
# compiler ID/version can be stale. Probe the binary to detect Clang.
if(NOT _SVMP_FE_IS_CLANG AND NOT CMAKE_CROSSCOMPILING)
    execute_process(
        COMMAND "${CMAKE_CXX_COMPILER}" "--version"
        RESULT_VARIABLE _svmp_fe_cxx_version_rc
        OUTPUT_VARIABLE _svmp_fe_cxx_version_out
        ERROR_VARIABLE _svmp_fe_cxx_version_err
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
    )
    if(_svmp_fe_cxx_version_rc EQUAL 0)
        set(_svmp_fe_cxx_version_text "${_svmp_fe_cxx_version_out}\n${_svmp_fe_cxx_version_err}")
        if(_svmp_fe_cxx_version_text MATCHES "clang version")
            set(_SVMP_FE_IS_CLANG TRUE)
            if(_svmp_fe_cxx_version_text MATCHES "clang version ([0-9]+)\\.")
                set(_SVMP_FE_CLANG_MAJOR "${CMAKE_MATCH_1}")
            endif()
            message(STATUS "FE: NOTE: CMake reports compiler ID '${CMAKE_CXX_COMPILER_ID}', but '${CMAKE_CXX_COMPILER}' identifies as Clang; consider deleting the build directory to refresh compiler detection")
        endif()
    endif()
endif()

if(_SVMP_FE_IS_CLANG AND _SVMP_FE_CLANG_MAJOR LESS 16 AND
   EXISTS "/usr/include/c++/11" AND
   EXISTS "/usr/include/x86_64-linux-gnu/c++/11")
    message(STATUS "FE: Clang <16 detected; using libstdc++-11 headers to avoid <chrono> build failure with newer libstdc++")
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:-nostdinc++>
        $<$<COMPILE_LANGUAGE:CXX>:-isystem/usr/include/c++/11>
        $<$<COMPILE_LANGUAGE:CXX>:-isystem/usr/include/x86_64-linux-gnu/c++/11>
        $<$<COMPILE_LANGUAGE:CXX>:-isystem/usr/include/c++/11/backward>
    )
endif()

# Warning flags
#
# NOTE: Keep warnings scoped to FE targets only. This CMakeLists can bring in
# optional dependencies (Mesh, METIS/GKlib, SymEngine, GTest, etc.) via
# add_subdirectory(). Using directory-scope add_compile_options() would apply
# these flags to dependency targets and cause a flood of warnings from external
# code and system headers.
set(_SVMP_FE_WARNING_OPTIONS "")
if(FE_ENABLE_WARNINGS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(_SVMP_FE_WARNING_OPTIONS
            $<$<COMPILE_LANGUAGE:CXX>:-Wall>
            $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
            $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
            $<$<COMPILE_LANGUAGE:CXX>:-Wcast-align>
            $<$<COMPILE_LANGUAGE:CXX>:-Wcast-qual>
            $<$<COMPILE_LANGUAGE:CXX>:-Wformat=2>
            $<$<COMPILE_LANGUAGE:CXX>:-Wformat-security>
            $<$<COMPILE_LANGUAGE:CXX>:-Wnon-virtual-dtor>
            $<$<COMPILE_LANGUAGE:CXX>:-Wold-style-cast>
            $<$<COMPILE_LANGUAGE:CXX>:-Woverloaded-virtual>
            $<$<COMPILE_LANGUAGE:CXX>:-Wshadow>
            $<$<COMPILE_LANGUAGE:CXX>:-Wsign-conversion>
            $<$<COMPILE_LANGUAGE:CXX>:-Wunused>
        )
        if(FE_WARNINGS_AS_ERRORS)
            list(APPEND _SVMP_FE_WARNING_OPTIONS $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
        endif()
    elseif(MSVC)
        set(_SVMP_FE_WARNING_OPTIONS $<$<COMPILE_LANGUAGE:CXX>:/W4>)
        if(FE_WARNINGS_AS_ERRORS)
            list(APPEND _SVMP_FE_WARNING_OPTIONS $<$<COMPILE_LANGUAGE:CXX>:/WX>)
        endif()
    endif()
endif()

function(_svmp_fe_target_enable_warnings target_name)
    if(_SVMP_FE_WARNING_OPTIONS AND TARGET ${target_name})
        target_compile_options(${target_name} PRIVATE ${_SVMP_FE_WARNING_OPTIONS})
    endif()
endfunction()

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
    Math/FiniteDifference.h
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
    Spaces/MortarSpace.h
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
    Spaces/MortarSpace.cpp
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
    Constraints/CoupledBCContext.h
    Constraints/CoupledNeumannBC.h
    Constraints/CoupledRobinBC.h
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
    Assembly/AssemblerSelection.h
    Assembly/GlobalSystemView.h
    Assembly/AssemblyKernel.h
    Assembly/AssemblyContext.h
    Assembly/JIT/KernelArgs.h
    Assembly/JIT/FieldSolutions.h
    Assembly/TimeIntegrationContext.h
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
    Assembly/DecoratorAssembler.h
    Assembly/ScheduledAssembler.h
    Assembly/VectorizedAssembler.h
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
    Assembly/AssemblerFactory.cpp
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
        Assembly/MeshAccess.h
    )
    list(APPEND FE_ASSEMBLY_SOURCES
        Assembly/MeshAccess.cpp
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

# Systems sources (requires FE/Assembly)
set(FE_SYSTEMS_HEADERS
    Systems/ContactPenaltyKernel.h
    Systems/AuxiliaryState.h
    Systems/AuxiliaryStateBuilder.h
    Systems/BoundaryConditionManager.h
    Systems/ODEIntegrator.h
    Systems/CoupledBoundaryConditions.h
    Systems/CoupledBoundaryManager.h
    Systems/FESystem.h
    Systems/FormsInstaller.h
    Systems/FieldRegistry.h
    Systems/GlobalKernel.h
    Systems/GlobalKernelStateProvider.h
    Systems/MaterialStateProvider.h
    Systems/MeshSearchAccess.h
    Systems/OperatorBackends.h
    Systems/OperatorRegistry.h
    Systems/ParameterRegistry.h
    Systems/SearchAccess.h
    Systems/SurfaceContactKernel.h
    Systems/SystemAssembly.h
    Systems/SystemConstraint.h
    Systems/SystemConstraints.h
    Systems/SystemSetup.h
    Systems/SystemState.h
    Systems/HCurlTangentialConstraint.h
    Systems/HDivNormalConstraint.h
    Systems/StrongDirichletConstraint.h
    Systems/TimeIntegrator.h
    Systems/TransientSystem.h
    Systems/SystemsExceptions.h
)

set(FE_SYSTEMS_SOURCES
    Systems/ContactPenaltyKernel.cpp
    Systems/FESystem.cpp
    Systems/FormsInstaller.cpp
    Systems/FieldRegistry.cpp
    Systems/GlobalKernelStateProvider.cpp
    Systems/MaterialStateProvider.cpp
    Systems/MeshSearchAccess.cpp
    Systems/OperatorBackends.cpp
    Systems/OperatorRegistry.cpp
    Systems/ParameterRegistry.cpp
    Systems/SurfaceContactKernel.cpp
    Systems/SystemAssembly.cpp
    Systems/SystemConstraints.cpp
    Systems/SystemSetup.cpp
    Systems/HCurlTangentialConstraint.cpp
    Systems/HDivNormalConstraint.cpp
    Systems/StrongDirichletConstraint.cpp
    Systems/TimeIntegrator.cpp
    Systems/CoupledBoundaryManager.cpp
    Systems/ODEIntegrator.cpp
    Systems/TransientSystem.cpp
)

# Forms sources (requires FE/Assembly; used by SymbolicAssembler)
set(FE_FORMS_HEADERS
    Forms/Forms.h
	    Forms/BoundaryConditions.h
	    Forms/BoundaryCondition.h
	    Forms/StandardBCs.h
	    Forms/CoupledBCs.h
	    Forms/NitscheBC.h
	    Forms/ConstraintBCs.h
	    Forms/BoundaryFunctional.h
	    Forms/AffineAnalysis.h
	    Forms/FormExpr.h
    Forms/SymbolicDifferentiation.h
    Forms/Index.h
    Forms/Einsum.h
    Forms/Tensor/TensorIndex.h
    Forms/Tensor/TensorSymmetry.h
    Forms/Tensor/SpecialTensors.h
    Forms/Tensor/TensorContraction.h
    Forms/Tensor/TensorCanonicalize.h
    Forms/Tensor/TensorSimplify.h
    Forms/Tensor/TensorDifferentiation.h
	    Forms/Tensor/SymmetryOptimizer.h
	    Forms/Tensor/TensorCSE.h
	    Forms/Tensor/LoopStructure.h
	    Forms/Tensor/TensorAllocation.h
	    Forms/Tensor/TensorIR.h
	    Forms/Tensor/TensorInterpreter.h
    Forms/Tensor/SpectralEigen.h
	    Forms/Tensor/SpecialTensorDerivatives.h
    Forms/FormIR.h
    Forms/FormCompiler.h
    Forms/FormKernels.h
    Forms/Dual.h
    Forms/Value.h
    Forms/ConstitutiveModel.h
    Forms/BlockForm.h
	    Forms/Complex.h
	    Forms/PointEvaluator.h
	    Forms/JIT/ExternalCalls.h
	    Forms/JIT/InlinableConstitutiveModel.h
	    Forms/JIT/JITCompiler.h
	    Forms/JIT/JITEngine.h
	    Forms/JIT/JITFormKernel.h
	    Forms/JIT/KernelIR.h
	    Forms/JIT/JITValidation.h
	    Forms/JIT/LLVMGen.h
	    Forms/JIT/LLVMTensorGen.h
	    Forms/JIT/LLVMJITBuildInfo.h
	    Forms/JIT/JITKernelWrapper.h
	    Forms/JIT/JITFunctionalKernelWrapper.h
	    Forms/Vocabulary.h
		)

set(FE_FORMS_SOURCES
    Forms/FormExpr.cpp
    Forms/Index.cpp
    Forms/Einsum.cpp
    Forms/Tensor/TensorIndex.cpp
    Forms/Tensor/TensorSymmetry.cpp
    Forms/Tensor/SpecialTensors.cpp
    Forms/Tensor/TensorContraction.cpp
    Forms/Tensor/TensorCanonicalize.cpp
    Forms/Tensor/TensorSimplify.cpp
    Forms/Tensor/TensorDifferentiation.cpp
	    Forms/Tensor/SymmetryOptimizer.cpp
	    Forms/Tensor/TensorCSE.cpp
	    Forms/Tensor/LoopStructure.cpp
	    Forms/Tensor/TensorAllocation.cpp
	    Forms/Tensor/TensorIR.cpp
	    Forms/Tensor/TensorInterpreter.cpp
    Forms/Tensor/SpectralEigen.cpp
	    Forms/Tensor/SpecialTensorDerivatives.cpp
    Forms/FormIR.cpp
    Forms/FormCompiler.cpp
    Forms/AffineAnalysis.cpp
	    Forms/BoundaryFunctional.cpp
	    Forms/FormKernels.cpp
	    Forms/SymbolicDifferentiation.cpp
	    Forms/PointEvaluator.cpp
	    Forms/JIT/ExternalCalls.cpp
	    Forms/JIT/JITCompiler.cpp
	    Forms/JIT/JITEngine.cpp
	    Forms/JIT/KernelIR.cpp
	    Forms/JIT/JITValidation.cpp
	    Forms/JIT/LLVMGen.cpp
	    Forms/JIT/LLVMTensorGen.cpp
	    Forms/JIT/LLVMJITBuildInfo.cpp
	    Forms/JIT/JITKernelWrapper.cpp
	    Forms/JIT/JITFunctionalKernelWrapper.cpp
		)

# TimeStepping sources (requires FE/Systems + FE/Assembly + FE/Backends)
	set(FE_TIMESTEPPING_HEADERS
	    TimeStepping/CollocationMethods.h
	    TimeStepping/TimeHistory.h
	    TimeStepping/MultiStageScheme.h
	    TimeStepping/NewtonSolver.h
	    TimeStepping/StepController.h
	    TimeStepping/VSVO_BDF_Controller.h
    TimeStepping/GeneralizedAlpha.h
    TimeStepping/NewmarkBeta.h
    TimeStepping/TimeSteppingUtils.h
    TimeStepping/TimeLoop.h
)

	set(FE_TIMESTEPPING_SOURCES
	    TimeStepping/CollocationMethods.cpp
	    TimeStepping/TimeHistory.cpp
	    TimeStepping/MultiStageScheme.cpp
	    TimeStepping/NewtonSolver.cpp
	    TimeStepping/StepController.cpp
	    TimeStepping/VSVO_BDF_Controller.cpp
    TimeStepping/GeneralizedAlpha.cpp
    TimeStepping/NewmarkBeta.cpp
    TimeStepping/TimeLoop.cpp
)

# Constitutive sources (header-only; depends on FE/Forms boundary)
set(FE_CONSTITUTIVE_HEADERS
    Constitutive/ModelCRTP.h
    Constitutive/ValueChecks.h
    Constitutive/StateLayout.h
    Constitutive/StateView.h
    Constitutive/Parameters.h
    Constitutive/GlobalLaw.h
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
    Backends/Interfaces/BackendKind.h
    Backends/Interfaces/DofPermutation.h
    Backends/Interfaces/BackendFactory.h
    Backends/Interfaces/GenericVector.h
    Backends/Interfaces/GenericMatrix.h
    Backends/Interfaces/LinearSolver.h
    Backends/Interfaces/BlockVector.h
    Backends/Interfaces/BlockMatrix.h

    Backends/Utils/BackendOptions.h

    Backends/Eigen/EigenFactory.h
    Backends/Eigen/EigenVector.h
    Backends/Eigen/EigenMatrix.h
    Backends/Eigen/EigenLinearSolver.h

    Backends/FSILS/FsilsFactory.h
    Backends/FSILS/FsilsVector.h
    Backends/FSILS/FsilsMatrix.h
    Backends/FSILS/FsilsLinearSolver.h

    # Vendored FSILS linear solver (copy of Code/Source/liner_solver)
    Backends/FSILS/liner_solver/add_bc_mul.h
    Backends/FSILS/liner_solver/bcast.h
    Backends/FSILS/liner_solver/bicgs.h
    Backends/FSILS/liner_solver/cgrad.h
    Backends/FSILS/liner_solver/commu.h
    Backends/FSILS/liner_solver/dot.h
    Backends/FSILS/liner_solver/fils_struct.hpp
    Backends/FSILS/liner_solver/fsils.hpp
    Backends/FSILS/liner_solver/fsils_api.hpp
    Backends/FSILS/liner_solver/fsils_std.h
    Backends/FSILS/liner_solver/ge.h
    Backends/FSILS/liner_solver/gmres.h
    Backends/FSILS/liner_solver/lhs.h
    Backends/FSILS/liner_solver/norm.h
    Backends/FSILS/liner_solver/ns_solver.h
    Backends/FSILS/liner_solver/omp_la.h
    Backends/FSILS/liner_solver/pc_gmres.h
    Backends/FSILS/liner_solver/precond.h
    Backends/FSILS/liner_solver/spar_mul.h

    Backends/PETSc/PetscUtils.h
    Backends/PETSc/PetscFactory.h
    Backends/PETSc/PetscVector.h
    Backends/PETSc/PetscMatrix.h
    Backends/PETSc/PetscLinearSolver.h

    Backends/Trilinos/TrilinosUtils.h
    Backends/Trilinos/TrilinosFactory.h
    Backends/Trilinos/TrilinosVector.h
    Backends/Trilinos/TrilinosMatrix.h
    Backends/Trilinos/TrilinosLinearSolver.h
)

set(FE_BACKENDS_SOURCES
    Backends/Interfaces/BackendKind.cpp
    Backends/Interfaces/BackendFactory.cpp
    Backends/Interfaces/BlockVector.cpp
    Backends/Interfaces/BlockMatrix.cpp

    Backends/Utils/BackendOptions.cpp

    Backends/Eigen/EigenFactory.cpp
    Backends/Eigen/EigenVector.cpp
    Backends/Eigen/EigenMatrix.cpp
    Backends/Eigen/EigenLinearSolver.cpp

    Backends/FSILS/FsilsFactory.cpp
    Backends/FSILS/FsilsVector.cpp
    Backends/FSILS/FsilsMatrix.cpp
    Backends/FSILS/FsilsLinearSolver.cpp
    Backends/FSILS/FsilsLegacyStatics.cpp

    # Vendored FSILS linear solver (copy of Code/Source/liner_solver)
    Backends/FSILS/liner_solver/add_bc_mul.cpp
    Backends/FSILS/liner_solver/bcast.cpp
    Backends/FSILS/liner_solver/bc.cpp
    Backends/FSILS/liner_solver/bicgs.cpp
    Backends/FSILS/liner_solver/cgrad.cpp
    Backends/FSILS/liner_solver/commu.cpp
    Backends/FSILS/liner_solver/cput.cpp
    Backends/FSILS/liner_solver/dot.cpp
    Backends/FSILS/liner_solver/fils_struct.cpp
    Backends/FSILS/liner_solver/ge.cpp
    Backends/FSILS/liner_solver/gmres.cpp
    Backends/FSILS/liner_solver/in_commu.cpp
    Backends/FSILS/liner_solver/lhs.cpp
    Backends/FSILS/liner_solver/ls.cpp
    Backends/FSILS/liner_solver/norm.cpp
    Backends/FSILS/liner_solver/ns_solver.cpp
    Backends/FSILS/liner_solver/omp_la.cpp
    Backends/FSILS/liner_solver/pc_gmres.cpp
    Backends/FSILS/liner_solver/precond.cpp
    Backends/FSILS/liner_solver/solve.cpp
    Backends/FSILS/liner_solver/spar_mul.cpp

    Backends/PETSc/PetscFactory.cpp
    Backends/PETSc/PetscVector.cpp
    Backends/PETSc/PetscMatrix.cpp
    Backends/PETSc/PetscLinearSolver.cpp

    Backends/Trilinos/TrilinosFactory.cpp
    Backends/Trilinos/TrilinosVector.cpp
    Backends/Trilinos/TrilinosMatrix.cpp
    Backends/Trilinos/TrilinosLinearSolver.cpp
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
    ${FE_SPARSITY_HEADERS}
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
    list(APPEND FE_ALL_HEADERS ${FE_BACKENDS_HEADERS})
    list(APPEND FE_ALL_SOURCES ${FE_BACKENDS_SOURCES})
    list(APPEND FE_ALL_HEADERS ${FE_ASSEMBLY_HEADERS})
    list(APPEND FE_ALL_SOURCES ${FE_ASSEMBLY_SOURCES})
    list(APPEND FE_ALL_HEADERS ${FE_SYSTEMS_HEADERS})
    list(APPEND FE_ALL_SOURCES ${FE_SYSTEMS_SOURCES})
    list(APPEND FE_ALL_HEADERS ${FE_FORMS_HEADERS})
    list(APPEND FE_ALL_SOURCES ${FE_FORMS_SOURCES})
    list(APPEND FE_ALL_HEADERS ${FE_TIMESTEPPING_HEADERS})
    list(APPEND FE_ALL_SOURCES ${FE_TIMESTEPPING_SOURCES})
    list(APPEND FE_ALL_HEADERS ${FE_CONSTITUTIVE_HEADERS})
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

# Configure LLVM for the Forms OrcJIT backend if enabled.
#
# Note: Forms/JIT sources are only built when FE_ENABLE_ASSEMBLY is enabled.
if(FE_ENABLE_LLVM_JIT)
    if(FE_ENABLE_ASSEMBLY)
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/EnableLLVMJIT.cmake)
    else()
        message(WARNING "FE: FE_ENABLE_LLVM_JIT is ON but FE_ENABLE_ASSEMBLY is OFF; Forms/JIT is not built, so LLVM will not be searched/linked.")
    endif()
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

_svmp_fe_target_enable_warnings(svfe)

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

_svmp_fe_target_enable_warnings(svfe_dofs)

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
	        target_include_directories(svfe_dofs SYSTEM PUBLIC ${FE_MPI_CXX_INCLUDE_DIRS})
	        target_link_libraries(svfe_dofs PUBLIC ${FE_MPI_CXX_LIBRARIES})
	    endif()
	    target_compile_definitions(svfe_dofs PUBLIC FE_HAS_MPI MESH_ENABLE_MPI MPICH_SKIP_MPICXX OMPI_SKIP_MPICXX _MPICC_H)
	endif()

if(FE_ENABLE_EIGEN)
    if(DEFINED FE_EIGEN_INCLUDE_DIR AND NOT FE_EIGEN_INCLUDE_DIR STREQUAL "")
        target_include_directories(svfe_dofs PUBLIC ${FE_EIGEN_INCLUDE_DIR})
    endif()
    target_compile_definitions(svfe_dofs PUBLIC FE_HAS_EIGEN=1)
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
target_compile_definitions(svfe PUBLIC SVMP_FE_ENABLE_LLVM_JIT=$<BOOL:${FE_ENABLE_LLVM_JIT}>)

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
	        target_include_directories(svfe SYSTEM PUBLIC ${FE_MPI_CXX_INCLUDE_DIRS})
	        target_link_libraries(svfe PUBLIC ${FE_MPI_CXX_LIBRARIES})
	    endif()
	    # Disable removed MPI-2 C++ bindings in common MPI implementations to avoid
	    # pulling deprecated headers (and associated warnings) through <mpi.h>.
	    target_compile_definitions(svfe PUBLIC FE_HAS_MPI MESH_ENABLE_MPI MPICH_SKIP_MPICXX OMPI_SKIP_MPICXX _MPICC_H)
	endif()

# Eigen is configured before the `svfe` target exists (EnableEigen.cmake), so
# apply the discovered include directory + feature define here.
if(FE_ENABLE_EIGEN)
    if(DEFINED FE_EIGEN_INCLUDE_DIR AND NOT FE_EIGEN_INCLUDE_DIR STREQUAL "")
        target_include_directories(svfe PUBLIC ${FE_EIGEN_INCLUDE_DIR})
    endif()
    target_compile_definitions(svfe PUBLIC FE_HAS_EIGEN=1)
endif()

# LLVM is configured before the `svfe` target exists (EnableLLVMJIT.cmake), so
# apply the discovered include directories + link libraries here.
if(FE_ENABLE_LLVM_JIT AND FE_ENABLE_ASSEMBLY)
    if(DEFINED FE_LLVM_INCLUDE_DIRS AND NOT FE_LLVM_INCLUDE_DIRS STREQUAL "")
        target_include_directories(svfe SYSTEM PUBLIC ${FE_LLVM_INCLUDE_DIRS})
    endif()

    if(DEFINED FE_LLVM_JIT_COMPILE_DEFINITIONS AND NOT FE_LLVM_JIT_COMPILE_DEFINITIONS STREQUAL "")
        target_compile_definitions(svfe PUBLIC ${FE_LLVM_JIT_COMPILE_DEFINITIONS})
    endif()

    if(DEFINED FE_LLVM_JIT_LIBRARIES AND NOT FE_LLVM_JIT_LIBRARIES STREQUAL "")
        target_link_libraries(svfe PUBLIC ${FE_LLVM_JIT_LIBRARIES})
    endif()

    if(DEFINED FE_LLVM_SYSTEM_LIBS AND NOT FE_LLVM_SYSTEM_LIBS STREQUAL "")
        target_link_libraries(svfe PUBLIC ${FE_LLVM_SYSTEM_LIBS})
    endif()
endif()

#------------------------------------------------------------------------------
# FSILS backend (vendored)
#------------------------------------------------------------------------------

# The FSILS backend is built from an in-tree copy of Code/Source/liner_solver and
# is always available when the Assembly/Backends module is enabled.
if(FE_ENABLE_ASSEMBLY)
    target_compile_definitions(svfe PUBLIC FE_HAS_FSILS=1)

    # The legacy solver-side containers used by FSILS (Vector/Array) require a
    # C-string header to be available before template definitions are parsed.
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(svfe PRIVATE -include cstring)
    endif()

    # FSILS depends on MPI even in "serial" mode (MPI_COMM_SELF).
    if(NOT TARGET MPI::MPI_CXX AND NOT DEFINED FE_MPI_CXX_LIBRARIES)
        find_package(MPI REQUIRED COMPONENTS CXX)
    endif()

	    if(TARGET MPI::MPI_CXX)
	        target_link_libraries(svfe PUBLIC MPI::MPI_CXX)
	    elseif(DEFINED MPI_CXX_INCLUDE_DIRS AND DEFINED MPI_CXX_LIBRARIES)
	        target_include_directories(svfe SYSTEM PUBLIC ${MPI_CXX_INCLUDE_DIRS})
	        target_link_libraries(svfe PUBLIC ${MPI_CXX_LIBRARIES})
	    elseif(DEFINED FE_MPI_CXX_INCLUDE_DIRS AND DEFINED FE_MPI_CXX_LIBRARIES)
	        target_include_directories(svfe SYSTEM PUBLIC ${FE_MPI_CXX_INCLUDE_DIRS})
	        target_link_libraries(svfe PUBLIC ${FE_MPI_CXX_LIBRARIES})
	    else()
	        message(FATAL_ERROR "FE: FSILS backend requires MPI, but no MPI configuration was found")
	    endif()
endif()

#------------------------------------------------------------------------------
# Optional external solver backends: PETSc / Trilinos
#------------------------------------------------------------------------------

# The FE Backends module is only compiled when Assembly is enabled.
if((FE_ENABLE_PETSC OR FE_ENABLE_TRILINOS) AND NOT FE_ENABLE_ASSEMBLY)
    message(FATAL_ERROR "FE: FE_ENABLE_PETSC / FE_ENABLE_TRILINOS require FE_ENABLE_ASSEMBLY=ON")
endif()

if(FE_ENABLE_ASSEMBLY AND FE_ENABLE_PETSC)
    message(STATUS "FE: Configuring PETSc backend support")

    set(_fe_petsc_configured OFF)

    # Prefer a CMake package config if available (PETScConfig.cmake).
    find_package(PETSc CONFIG QUIET)
    if(PETSc_FOUND)
        if(TARGET PETSc::PETSc)
            target_link_libraries(svfe PUBLIC PETSc::PETSc)
            set(_fe_petsc_configured ON)
        elseif(TARGET PETSc::petsc)
            target_link_libraries(svfe PUBLIC PETSc::petsc)
            set(_fe_petsc_configured ON)
        elseif(DEFINED PETSc_LIBRARIES AND DEFINED PETSc_INCLUDE_DIRS)
            target_include_directories(svfe PUBLIC ${PETSc_INCLUDE_DIRS})
            target_link_libraries(svfe PUBLIC ${PETSc_LIBRARIES})
            set(_fe_petsc_configured ON)
        endif()
    endif()

    # Fallback: support the legacy SV_PETSC_DIR (points at PETSC_DIR/PETSC_ARCH).
    if(NOT _fe_petsc_configured)
        set(_fe_petsc_arch_dir "")
        if(DEFINED SV_PETSC_DIR AND NOT "${SV_PETSC_DIR}" STREQUAL "")
            set(_fe_petsc_arch_dir "${SV_PETSC_DIR}")
        elseif(DEFINED PETSC_DIR AND DEFINED PETSC_ARCH AND
               NOT "${PETSC_DIR}" STREQUAL "" AND NOT "${PETSC_ARCH}" STREQUAL "")
            set(_fe_petsc_arch_dir "${PETSC_DIR}/${PETSC_ARCH}")
        elseif(DEFINED PETSC_DIR AND NOT "${PETSC_DIR}" STREQUAL "")
            # Best-effort: PETSC_DIR may already be an arch-specific directory.
            set(_fe_petsc_arch_dir "${PETSC_DIR}")
        endif()

        if(_fe_petsc_arch_dir STREQUAL "")
            message(FATAL_ERROR "FE: FE_ENABLE_PETSC=ON but PETSc was not found. Provide PETScConfig.cmake or set SV_PETSC_DIR.")
        endif()

        if(NOT EXISTS "${_fe_petsc_arch_dir}/include")
            message(FATAL_ERROR "FE: PETSc include directory not found: ${_fe_petsc_arch_dir}/include")
        endif()
        if(NOT EXISTS "${_fe_petsc_arch_dir}/lib")
            message(FATAL_ERROR "FE: PETSc library directory not found: ${_fe_petsc_arch_dir}/lib")
        endif()

        set(_fe_petsc_include_dirs "${_fe_petsc_arch_dir}/include")
        get_filename_component(_fe_petsc_root_dir "${_fe_petsc_arch_dir}" DIRECTORY)
        if(EXISTS "${_fe_petsc_root_dir}/include")
            list(APPEND _fe_petsc_include_dirs "${_fe_petsc_root_dir}/include")
        endif()

        find_library(_fe_petsc_library
            NAMES petsc
            PATHS "${_fe_petsc_arch_dir}/lib"
            NO_DEFAULT_PATH
        )
        if(NOT _fe_petsc_library)
            message(FATAL_ERROR "FE: PETSc library 'petsc' not found under ${_fe_petsc_arch_dir}/lib")
        endif()

        target_include_directories(svfe PUBLIC ${_fe_petsc_include_dirs})
        target_link_libraries(svfe PUBLIC ${_fe_petsc_library})
        set(_fe_petsc_configured ON)

        message(STATUS "FE: PETSc configured via SV_PETSC_DIR/PETSC_DIR (${_fe_petsc_arch_dir})")
    endif()

    if(_fe_petsc_configured)
        target_compile_definitions(svfe PUBLIC FE_HAS_PETSC=1)
    endif()
endif()

if(FE_ENABLE_ASSEMBLY AND FE_ENABLE_TRILINOS)
    message(STATUS "FE: Configuring Trilinos backend support")

    find_package(Trilinos REQUIRED)
    if(NOT Trilinos_FOUND)
        message(FATAL_ERROR "FE: FE_ENABLE_TRILINOS=ON but Trilinos was not found")
    endif()

    # Prefer the imported aggregate target if available.
    if(TARGET Trilinos::all_selected_libs)
        target_link_libraries(svfe PUBLIC Trilinos::all_selected_libs)
    else()
        target_include_directories(svfe PUBLIC ${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS})
        target_link_libraries(svfe PUBLIC ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES})
    endif()

    target_compile_definitions(svfe PUBLIC FE_HAS_TRILINOS=1)
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
        _svmp_fe_target_enable_warnings(test_fe_math)
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
        _svmp_fe_target_enable_warnings(test_fe_core)
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
    _svmp_fe_target_enable_warnings(test_fe_quadrature)
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
        Tests/Unit/Basis/test_BSplineBasis.cpp
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
    _svmp_fe_target_enable_warnings(test_fe_basis)
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
    _svmp_fe_target_enable_warnings(test_fe_geometry)
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
    _svmp_fe_target_enable_warnings(test_fe_elements)
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
        _svmp_fe_target_enable_warnings(test_fe_spaces)
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
			        _svmp_fe_target_enable_warnings(test_fe_dofs)
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
        _svmp_fe_target_enable_warnings(test_fe_constraints)
        target_link_libraries(test_fe_constraints
            PRIVATE
                svfe
                ${GTEST_LIBRARIES}
        )
        # Avoid leaking environment-specific RUNPATHs into test executables.
        set_target_properties(test_fe_constraints PROPERTIES
            SKIP_BUILD_RPATH TRUE
            BUILD_RPATH ""
            INSTALL_RPATH ""
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
        _svmp_fe_target_enable_warnings(test_fe_sparsity)
        target_link_libraries(test_fe_sparsity
            PRIVATE
                svfe
                ${GTEST_LIBRARIES}
        )
        add_test(NAME FE_Sparsity_Tests COMMAND test_fe_sparsity)
    endif()

    if(FE_ENABLE_ASSEMBLY)
        # Backends module tests
	        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/Tests/Unit/Backends/test_main.cpp)
	            set(FE_BACKENDS_TEST_SOURCES
	                Tests/Unit/Backends/test_BackendOptions.cpp
	                Tests/Unit/Backends/test_BackendKind.cpp
	                Tests/Unit/Backends/test_BackendFactory.cpp
	                Tests/Unit/Backends/test_BlockSystems.cpp
	                Tests/Unit/Backends/test_EigenBackend.cpp
	                Tests/Unit/Backends/test_FsilsBackend.cpp
	                Tests/Unit/Backends/test_LinearSolverConformance.cpp
	                Tests/Unit/Backends/test_PetscBackend.cpp
	                Tests/Unit/Backends/test_main.cpp
		            )
		            add_executable(test_fe_backends ${FE_BACKENDS_TEST_SOURCES})
		            _svmp_fe_target_enable_warnings(test_fe_backends)
		            target_link_libraries(test_fe_backends
		                PRIVATE
		                    svfe
		                    ${GTEST_LIBRARIES}
	            )
	            add_test(NAME FE_Backends_Tests COMMAND test_fe_backends)
	        endif()

	        # Trilinos-specific backends tests (separate main for Tpetra init)
	        if(FE_ENABLE_TRILINOS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/Tests/Unit/Backends/test_trilinos_main.cpp)
	            set(FE_TRILINOS_BACKENDS_TEST_SOURCES
	                Tests/Unit/Backends/test_TrilinosBackend.cpp
	                Tests/Unit/Backends/test_LinearSolverConformance_Trilinos.cpp
	                Tests/Unit/Backends/test_trilinos_main.cpp
		            )
		            add_executable(test_fe_backends_trilinos ${FE_TRILINOS_BACKENDS_TEST_SOURCES})
		            _svmp_fe_target_enable_warnings(test_fe_backends_trilinos)
		            target_link_libraries(test_fe_backends_trilinos
		                PRIVATE
		                    svfe
		                    ${GTEST_LIBRARIES}
	            )
	            add_test(NAME FE_Trilinos_Backends_Tests COMMAND test_fe_backends_trilinos)
	        endif()

        # Forms module tests (depends on Assembly)
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/Tests/Unit/Forms/test_main.cpp)
		            set(FE_FORMS_TEST_SOURCES
			                Tests/Unit/Forms/test_FormExpr.cpp
			                    Tests/Unit/Forms/test_CoupledTerminals.cpp
			                    Tests/Unit/Forms/test_JITValidation.cpp
			                    Tests/Unit/Forms/test_KernelIR_Lowering.cpp
			                    Tests/Unit/Forms/test_ExternalCallsNewPhysics.cpp
				                    Tests/Unit/Forms/test_JITCompiler_CacheAndThreads.cpp
					                    Tests/Unit/Forms/test_JITCompiler_ErrorHandling.cpp
						                    Tests/Unit/Forms/test_JIT_CacheRegression.cpp
						                    Tests/Unit/Forms/test_JITEngine_Lifecycle.cpp
					                    Tests/Unit/Forms/test_JIT_CompilationLatency.cpp
					                    Tests/Unit/Forms/test_JIT_PerformanceRegression.cpp
					                    Tests/Unit/Forms/test_JIT_MatrixFunctionPerformance.cpp
					                    Tests/Unit/Forms/test_JIT_Vectorization.cpp
					                    Tests/Unit/Forms/test_JIT_FacePerformance.cpp
					                    Tests/Unit/Forms/test_JIT_NonlinearPerformance.cpp
					                    Tests/Unit/Forms/test_JIT_TensorContractionPerformance.cpp
					                    Tests/Unit/Forms/test_JIT_Scaling.cpp
					                    Tests/Unit/Forms/test_JIT_SpecializationPerformance.cpp
					                    Tests/Unit/Forms/test_JIT_DiskCache.cpp
					                    Tests/Unit/Forms/test_JIT_MemoryOverhead.cpp
						                    Tests/Unit/Forms/test_LLVMGen_FunctionalKernels.cpp
					                    Tests/Unit/Forms/test_LLVMGen_ScalarOps.cpp
					                    Tests/Unit/Forms/test_LLVMGen_SmoothOps.cpp
					                    Tests/Unit/Forms/test_LLVMGen_TensorOps.cpp
				                    Tests/Unit/Forms/test_LLVMGen_DifferentialOps.cpp
				                    Tests/Unit/Forms/test_LLVMGen_MatrixFunctions.cpp
				                    Tests/Unit/Forms/test_LLVMGen_MatrixFunctionDerivatives.cpp
				                    Tests/Unit/Forms/test_LLVMGen_GeometryOps.cpp
				                    Tests/Unit/Forms/test_LLVMGen_FieldOps.cpp
				                    Tests/Unit/Forms/test_LLVMGen_TimeOps.cpp
				                    Tests/Unit/Forms/test_PointEvaluator.cpp
				                Tests/Unit/Forms/test_BoundaryConditionHelpers.cpp
				                Tests/Unit/Forms/test_FormCompiler.cpp
			                Tests/Unit/Forms/test_Dual.cpp
		                Tests/Unit/Forms/test_DynamicValueShapes.cpp
		                Tests/Unit/Forms/test_FormKernel_Cell.cpp
		                Tests/Unit/Forms/test_FormKernel_Boundary.cpp
		                Tests/Unit/Forms/test_FormKernel_DG.cpp
		                Tests/Unit/Forms/test_FormSpatialDerivatives.cpp
		                Tests/Unit/Forms/test_FormVocabulary.cpp
		                Tests/Unit/Forms/test_MixedSpaceVocabulary.cpp
			                Tests/Unit/Forms/test_Einsum.cpp
			                Tests/Unit/Forms/test_FormsPerformance.cpp
			                Tests/Unit/Forms/test_NonlinearFormKernel.cpp
			                Tests/Unit/Forms/test_SymbolicDifferentiation.cpp
	                    Tests/Unit/Forms/test_HessianVectorProduct.cpp
		                Tests/Unit/Forms/Tensor/test_TensorIndex.cpp
		                Tests/Unit/Forms/Tensor/test_TensorSymmetry.cpp
			                Tests/Unit/Forms/Tensor/test_TensorContraction.cpp
			                Tests/Unit/Forms/Tensor/test_TensorCanonicalize.cpp
			                Tests/Unit/Forms/Tensor/test_TensorSimplify.cpp
			                Tests/Unit/Forms/Tensor/test_TensorDifferentiation.cpp
			                Tests/Unit/Forms/Tensor/test_SymmetryOptimizer.cpp
				                Tests/Unit/Forms/Tensor/test_TensorCSE.cpp
				                Tests/Unit/Forms/Tensor/test_LoopStructure.cpp
				                Tests/Unit/Forms/Tensor/test_TensorAllocation.cpp
				                Tests/Unit/Forms/Tensor/test_TensorInterpreterKernel.cpp
				                Tests/Unit/Forms/Tensor/test_TensorBenchmarks.cpp
				                Tests/Unit/Forms/Tensor/test_TensorSymbolicDifferentiation.cpp
                    Tests/Unit/Forms/Tensor/test_SpectralEigenJIT.cpp
			                Tests/Unit/Forms/Tensor/test_TensorPrettyPrinting.cpp
			                Tests/Unit/Forms/Tensor/test_SpecialTensors.cpp
			                Tests/Unit/Forms/Tensor/test_IndexedAccessJIT.cpp
		                Tests/Unit/Forms/test_ConstitutiveModel.cpp
	                Tests/Unit/Forms/test_PrimitiveTypes.cpp
		                Tests/Unit/Forms/test_main.cpp
		            )
	            add_executable(test_fe_forms ${FE_FORMS_TEST_SOURCES})
	            _svmp_fe_target_enable_warnings(test_fe_forms)
	            target_link_libraries(test_fe_forms
	                PRIVATE
	                    svfe
	                    ${GTEST_LIBRARIES}
            )
            add_test(NAME FE_Forms_Tests COMMAND test_fe_forms)
        endif()

        # Constitutive module tests (depends on Forms + Systems + Assembly)
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/Tests/Unit/Constitutive/test_main.cpp)
            set(FE_CONSTITUTIVE_TEST_SOURCES
                Tests/Unit/Constitutive/test_ModelCRTP.cpp
                Tests/Unit/Constitutive/test_LawAdapters.cpp
                Tests/Unit/Constitutive/test_StateLayoutAndView.cpp
                Tests/Unit/Constitutive/test_MaterialStatePlumbing.cpp
                Tests/Unit/Constitutive/test_Parameters.cpp
                Tests/Unit/Constitutive/test_GlobalLaw.cpp
	                Tests/Unit/Constitutive/test_main.cpp
	            )
	            add_executable(test_fe_constitutive ${FE_CONSTITUTIVE_TEST_SOURCES})
	            _svmp_fe_target_enable_warnings(test_fe_constitutive)
	            target_link_libraries(test_fe_constitutive
	                PRIVATE
	                    svfe
	                    ${GTEST_LIBRARIES}
            )
            add_test(NAME FE_Constitutive_Tests COMMAND test_fe_constitutive)
        endif()

        # Assembly module tests
        set(FE_ASSEMBLY_TEST_SOURCES
	            # Phase 1: CORE tests
	            Tests/Unit/Assembly/test_GlobalSystemView.cpp
	            Tests/Unit/Assembly/test_AssemblyKernel.cpp
	            Tests/Unit/Assembly/test_AssemblyContext.cpp
            Tests/Unit/Assembly/test_JITKernelArgsPacking.cpp
            Tests/Unit/Assembly/test_StandardAssembler.cpp
            Tests/Unit/Assembly/test_VectorBasisOrientationAssembly.cpp
            Tests/Unit/Assembly/test_AssemblerSelection.cpp
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
                Tests/Unit/Assembly/test_MeshAccess.cpp
                Tests/Unit/Assembly/test_MeshAccess_HighOrder.cpp
            )
	        endif()
	        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/Tests/Unit/Assembly/test_main.cpp)
	            add_executable(test_fe_assembly ${FE_ASSEMBLY_TEST_SOURCES})
	            _svmp_fe_target_enable_warnings(test_fe_assembly)
	            target_link_libraries(test_fe_assembly
	                PRIVATE
	                    svfe
	                    ${GTEST_LIBRARIES}
            )
            add_test(NAME FE_Assembly_Tests COMMAND test_fe_assembly)
        endif()

        # Systems module tests (Systems depends on Assembly; tests also require Mesh)
        if(FE_WITH_MESH AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/Tests/Unit/Systems/test_main.cpp)
            set(FE_SYSTEMS_TEST_SOURCES
	                Tests/Unit/Systems/test_FieldRegistry.cpp
	                Tests/Unit/Systems/test_ContactPenaltyKernel.cpp
	                Tests/Unit/Systems/test_SurfaceContactKernel.cpp
	                Tests/Unit/Systems/test_ParameterRegistry.cpp
	                Tests/Unit/Systems/test_FieldEvaluation.cpp
	                Tests/Unit/Systems/test_FESystem.cpp
	                Tests/Unit/Systems/test_AssemblerSelection.cpp
	                Tests/Unit/Systems/test_MultiFieldRectangularAssembly.cpp
                Tests/Unit/Systems/test_NavierStokesCoupled.cpp
                Tests/Unit/Systems/test_SystemConstraints.cpp
                Tests/Unit/Systems/test_VectorBasisConstraints.cpp
                Tests/Unit/Systems/test_StrongDirichletConstraint.cpp
                Tests/Unit/Systems/test_FormsInstaller.cpp
	                Tests/Unit/Systems/test_CoupledBoundaryManager.cpp
                Tests/Unit/Systems/test_CoupledBoundaryConditionHelpers.cpp
                Tests/Unit/Systems/test_AuxiliaryStateBuilder.cpp
                Tests/Unit/Systems/test_ODEIntegrator.cpp
                Tests/Unit/Systems/test_OperatorBackends.cpp
                Tests/Unit/Systems/test_TransientDt.cpp
                Tests/Unit/Systems/test_main.cpp
	            )
	            add_executable(test_fe_systems ${FE_SYSTEMS_TEST_SOURCES})
	            _svmp_fe_target_enable_warnings(test_fe_systems)
	            target_link_libraries(test_fe_systems
	                PRIVATE
	                    svfe
	                    ${GTEST_LIBRARIES}
            )
            add_test(NAME FE_Systems_Tests COMMAND test_fe_systems)
        endif()

        # TimeStepping module tests (depends on Systems + Forms + Backends; no Mesh required)
	        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/Tests/Unit/TimeStepping/test_main.cpp)
		            set(FE_TIMESTEPPING_TEST_SOURCES
		                Tests/Unit/TimeStepping/test_TimeHistory.cpp
		                Tests/Unit/TimeStepping/test_TimeIntegrators.cpp
		                Tests/Unit/TimeStepping/test_CollocationMethods.cpp
		                Tests/Unit/TimeStepping/test_TimeLoopStability.cpp
		                Tests/Unit/TimeStepping/test_TimeSteppingUtils.cpp
		                Tests/Unit/TimeStepping/test_StepControllers.cpp
		                Tests/Unit/TimeStepping/test_NewtonSolver.cpp
		                Tests/Unit/TimeStepping/test_MultiStageScheme.cpp
		                Tests/Unit/TimeStepping/test_StepController.cpp
		                Tests/Unit/TimeStepping/test_TimeLoopStageTimes.cpp
		                Tests/Unit/TimeStepping/test_TimeLoopConvergence.cpp
		                Tests/Unit/TimeStepping/test_TimeLoopDissipation.cpp
	                Tests/Unit/TimeStepping/test_TimeLoopPerformance.cpp
		                Tests/Unit/TimeStepping/test_main.cpp
		            )
	            add_executable(test_fe_timestepping ${FE_TIMESTEPPING_TEST_SOURCES})
	            _svmp_fe_target_enable_warnings(test_fe_timestepping)
	            target_link_libraries(test_fe_timestepping
	                PRIVATE
	                    svfe
	                    ${GTEST_LIBRARIES}
            )
            add_test(NAME FE_TimeStepping_Tests COMMAND test_fe_timestepping)
        endif()
    endif()

    # Add MPI tests if MPI is enabled
    if(FE_ENABLE_MPI AND MPIEXEC_EXECUTABLE)
        file(GLOB FE_MPI_TEST_SOURCES
            Tests/Unit/Dofs/test_*MPI*.cpp
	        )
	        if(FE_MPI_TEST_SOURCES)
	            add_executable(test_fe_mpi ${FE_MPI_TEST_SOURCES})
	            _svmp_fe_target_enable_warnings(test_fe_mpi)
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
	            _svmp_fe_target_enable_warnings(test_fe_constraints_mpi)
	            # MPI tests provide their own main() to call MPI_Init/MPI_Finalize,
	            # so do not link gtest_main here.
	            if(TARGET GTest::gtest)
                target_link_libraries(test_fe_constraints_mpi PRIVATE svfe GTest::gtest)
            else()
                set(_fe_constraints_mpi_gtest_libs ${GTEST_LIBRARIES})
                list(REMOVE_ITEM _fe_constraints_mpi_gtest_libs gtest_main GTest::gtest_main)
                target_link_libraries(test_fe_constraints_mpi PRIVATE svfe ${_fe_constraints_mpi_gtest_libs})
            endif()
            # Avoid leaking environment-specific RUNPATHs into test executables.
            set_target_properties(test_fe_constraints_mpi PROPERTIES
                SKIP_BUILD_RPATH TRUE
                BUILD_RPATH ""
                INSTALL_RPATH ""
            )
            add_fe_mpi_test(test_fe_constraints_mpi 2)
            add_fe_mpi_test(test_fe_constraints_mpi 4)
        endif()

	        if(FE_ENABLE_ASSEMBLY)
	            file(GLOB FE_BACKENDS_MPI_TEST_SOURCES
	                Tests/Unit/Backends/test_*MPI*.cpp
	            )
	            if(FE_BACKENDS_MPI_TEST_SOURCES)
	                add_executable(test_fe_backends_mpi ${FE_BACKENDS_MPI_TEST_SOURCES})
	                _svmp_fe_target_enable_warnings(test_fe_backends_mpi)
	                # MPI tests provide their own main() to call MPI_Init/MPI_Finalize,
	                # so do not link gtest_main here.
	                if(TARGET GTest::gtest)
	                    target_link_libraries(test_fe_backends_mpi PRIVATE svfe GTest::gtest)
	                else()
	                    set(_fe_backends_mpi_gtest_libs ${GTEST_LIBRARIES})
	                    list(REMOVE_ITEM _fe_backends_mpi_gtest_libs gtest_main GTest::gtest_main)
	                    target_link_libraries(test_fe_backends_mpi PRIVATE svfe ${_fe_backends_mpi_gtest_libs})
	                endif()
	                add_fe_mpi_test(test_fe_backends_mpi 2)
	            endif()
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
	            _svmp_fe_target_enable_warnings(${example_name})
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
	            PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/svmesh
	        )
	    endif()

    # If Mesh (svmesh) pulled in bundled METIS/ParMETIS targets, ensure they are
    # part of the same export set so `install(EXPORT ...)` remains consistent.
    set(_fe_tp_targets
        ${GKLIB_INTERNAL_LIBRARY_NAME}
        ${METIS_INTERNAL_LIBRARY_NAME}
        ${PARMETIS_INTERNAL_LIBRARY_NAME}
        gklib_internal
        metis_internal
        parmetis_internal
    )
    list(REMOVE_DUPLICATES _fe_tp_targets)
    foreach(_fe_tp_target IN LISTS _fe_tp_targets)
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

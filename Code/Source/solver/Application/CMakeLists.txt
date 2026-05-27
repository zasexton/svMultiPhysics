cmake_minimum_required(VERSION 3.12)

#------------------------------------------------------------------------------
# svapplication - Application/driver layer for the new OOP Mesh/FE/Physics stack
#------------------------------------------------------------------------------

add_library(svapplication STATIC
  Core/ActiveDomainOutput.cpp
  Core/ApplicationDriver.cpp
  Core/LevelSetCutConfiguration.cpp
  Core/LevelSetCurvatureSamples.cpp
  Core/LevelSetMaintenanceHistory.cpp
  Core/MeshCollection.cpp
  Core/SimulationBuilder.cpp
  Translators/MeshTranslator.cpp
  Translators/EquationTranslator.cpp
  Translators/LevelSetEquationTranslator.cpp
)

set_target_properties(svapplication PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(svapplication
  PUBLIC
    svfe
    svmesh
  PRIVATE
    ${TINYXML_LIBRARY_NAME}
)

target_include_directories(svapplication
  PUBLIC
    # Include root is the solver/ directory so consumers can write:
    #   #include "Application/..."
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)

# ApplicationDriver.cpp needs OpenMP for automatic thread configuration.
# Use the compile flags exported by the FE CMakeLists.
if(FE_OMP_ENABLED)
    set(_APP_OMP_FLAGS "${FE_OMP_COMPILE_FLAG}")
    if(FE_OMP_INCLUDE_FLAG)
        list(APPEND _APP_OMP_FLAGS "${FE_OMP_INCLUDE_FLAG}")
    endif()
    set_source_files_properties(Core/ApplicationDriver.cpp
        PROPERTIES COMPILE_OPTIONS "${_APP_OMP_FLAGS}"
    )
endif()

option(APPLICATION_BUILD_TESTS "Build Application tests" OFF)
if(APPLICATION_BUILD_TESTS)
  add_subdirectory(Tests)
endif()

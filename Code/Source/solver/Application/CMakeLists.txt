cmake_minimum_required(VERSION 3.12)

#------------------------------------------------------------------------------
# svapplication - Application/driver layer for the new OOP Mesh/FE/Physics stack
#------------------------------------------------------------------------------

add_library(svapplication STATIC
  Core/ApplicationDriver.cpp
  Core/SimulationBuilder.cpp
  Translators/MeshTranslator.cpp
  Translators/EquationTranslator.cpp
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

option(APPLICATION_BUILD_TESTS "Build Application tests" OFF)
if(APPLICATION_BUILD_TESTS)
  add_subdirectory(Tests)
endif()

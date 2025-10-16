cmake_minimum_required(VERSION 3.10)

# This CMakeLists builds the Python bindings for the svmesh library using pybind11

# pybind11 is already found/fetched by the parent CMakeLists.txt
# No need to find_package here - just check that it's available
if(NOT TARGET pybind11::pybind11)
  message(FATAL_ERROR "pybind11 target not available. Parent CMakeLists.txt should have found or fetched it.")
endif()

# Build the Python module
file(GLOB_RECURSE SVMP_PY_SOURCES
  module.cpp
  Core/*.cpp
  IO/*.cpp
  Topology/*.cpp
  Geometry/*.cpp
  Fields/*.cpp
  Labels/*.cpp
  Search/*.cpp
  Boundary/*.cpp
)

pybind11_add_module(svmesh_py MODULE ${SVMP_PY_SOURCES})

target_link_libraries(svmesh_py PRIVATE svmesh)

target_compile_features(svmesh_py PRIVATE cxx_std_17)

target_include_directories(svmesh_py PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# On some platforms, ensure the library has an rpath to find svmesh if shared
if(MESH_BUILD_SHARED)
  if(APPLE)
    set_target_properties(svmesh_py PROPERTIES
      MACOSX_RPATH ON
    )
  endif()
endif()

# Optional: place the built module next to the Python package folder for convenience
set_target_properties(svmesh_py PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

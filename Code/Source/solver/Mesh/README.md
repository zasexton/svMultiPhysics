# svMultiPhysics Mesh Infrastructure

This directory contains the modernized mesh infrastructure for svMultiPhysics. The mesh components are organized into modular subfolders and can be built either as part of the main svMultiPhysics project or as a standalone library.

## Directory Structure

- **Core/**: Core mesh types and base classes
  - `MeshTypes.h`: Basic type definitions
  - `MeshBase.h`: Base mesh interface
  - `InterfaceMesh.h`: Interface mesh support
  - `DistributedMesh.h`: Distributed mesh with MPI support

- **Topology/**: Mesh topology operations
  - Cell shape definitions
  - Connectivity information
  - Topological queries

- **Geometry/**: Geometric operations
  - Mesh geometry calculations
  - Mesh quality metrics
  - Orientation utilities

- **Fields/**: Field data management
  - Field descriptors
  - Field storage and access

- **Labels/**: Mesh labeling and tagging
  - Boundary and volume labels
  - Face and edge labels

- **Search/**: Spatial search operations
  - Element search
  - Nearest neighbor queries

- **Validation/**: Mesh validation
  - Mesh quality checks
  - Spatial hashing

- **Observer/**: Observer pattern for mesh changes

- **IO/**: Input/output operations
  - Basic mesh I/O
  - VTK reader/writer (requires VTK)

## Building as a Standalone Library

### Quick Start

```bash
cd Code/Source/solver/Mesh
mkdir build && cd build
cmake ..
make
```

### Build Options

The following CMake options are available:

- `MESH_ENABLE_MPI` (default: ON): Enable MPI support for distributed meshes
- `MESH_ENABLE_VTK` (default: ON): Enable VTK I/O support
- `MESH_BUILD_TESTS` (default: OFF): Build mesh tests
- `MESH_BUILD_SHARED` (default: OFF): Build as shared library instead of static

### Examples

**Minimal build (no MPI, no VTK):**
```bash
cmake -DMESH_ENABLE_MPI=OFF -DMESH_ENABLE_VTK=OFF ..
make
```

**Build with tests:**
```bash
cmake -DMESH_BUILD_TESTS=ON ..
make
```

**Build as shared library:**
```bash
cmake -DMESH_BUILD_SHARED=ON ..
make
```

**Custom VTK location:**
```bash
cmake -DVTK_DIR=/path/to/vtk/lib/cmake/vtk-9.0 ..
make
```

### Installation

To install the library system-wide:

```bash
cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
make
sudo make install
```

This will install:
- Library: `/usr/local/lib/libsvmesh.a` (or `.so`)
- Headers: `/usr/local/include/svmesh/`
- CMake config: `/usr/local/lib/cmake/svmesh/`

### Using in Other Projects

After installation, you can use the mesh library in other CMake projects:

```cmake
find_package(svmesh REQUIRED)
target_link_libraries(your_target svmesh::svmesh)
```

Or manually:
```cmake
find_library(SVMESH_LIBRARY svmesh)
find_path(SVMESH_INCLUDE_DIR svmesh/Mesh.h)
target_link_libraries(your_target ${SVMESH_LIBRARY})
target_include_directories(your_target PUBLIC ${SVMESH_INCLUDE_DIR})
```

## Integration with Main Project

When built as part of the main svMultiPhysics project, the mesh components are automatically included in the solver library. No special configuration is needed.

## Dependencies

### Required
- C++17 compatible compiler
- CMake 3.10 or newer

### Optional
- **MPI**: Required for distributed mesh support (DistributedMesh)
- **VTK**: Required for VTK I/O operations (VTKReader, VTKWriter)

## Testing

The `test_compile/` directory contains compilation tests to verify that all mesh components compile correctly. To run these tests:

```bash
cmake -DMESH_BUILD_TESTS=ON ..
make
./test_compile/mesh_compile_test
```

## Notes

- The mesh infrastructure is designed to be modular and extensible
- Header-only components (Core, Observer) have minimal dependencies
- MPI features are only available when compiled with `MESH_ENABLE_MPI=ON`
- VTK I/O features are only available when compiled with `MESH_ENABLE_VTK=ON`
- All includes use relative paths from the `solver/` directory for consistency

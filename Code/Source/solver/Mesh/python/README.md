Python bindings for svMultiPhysics Mesh

Overview
- The Python bindings mirror the Mesh folder structure using submodules:
  - `svmesh_py.core` (Mesh/types)
  - `svmesh_py.topology`
  - `svmesh_py.geometry`
  - `svmesh_py.fields`
  - `svmesh_py.labels`
  - `svmesh_py.search`
  - `svmesh_py.boundary`
  - `svmesh_py.io`
- The top-level module is `svmesh_py` which contains these submodules.

Build
- Enable with CMake option `-DMESH_ENABLE_PYTHON=ON` and ensure `pybind11` is discoverable (installed or available via `pybind11Config.cmake`).
- Example:
  - `cmake -DMESH_ENABLE_PYTHON=ON -DMESH_BUILD_SHARED=ON ..`
  - `cmake --build . --target svmesh_py`

Usage
- Basic example:
  ```python
  import numpy as np
  import svmesh_py as sv
  from svmesh_py import core, io

  # 2D square mesh with 2 triangles
  X = np.array([[0,0], [1,0], [1,1], [0,1]], dtype=float)
  cell2vertex_offsets = [0, 3, 6]
  cell2vertex = [0,1,2, 0,2,3]
  families = ["Triangle", "Triangle"]

  m = core.Mesh(2)
  m.build_from_arrays(2, X, cell2vertex_offsets, cell2vertex, families)
  m.finalize()

  print("cells:", m.n_cells())
  print("bbox:", m.bounding_box())
  print("c0 center:", m.cell_center(0))
  ```

VTK I/O
- Requires building the mesh library with VTK (`-DMESH_ENABLE_VTK=ON`).
- On import of `svmesh_py.io`, `io.has_vtk` indicates availability.
- Read/write helpers:
  - `io.read_vtk(path)` / `io.read_vtu(path)` / `io.read_vtp(path)` → returns `core.Mesh`
  - `io.write_vtk(mesh, path, binary=False, compress=False)`
  - `io.write_vtu(mesh, path, binary=False, compress=True)`
  - `io.write_vtp(mesh, path, binary=False, compress=True)`
- Or use the instance method `mesh.save(filename, format, options)` where `format` is `"vtk"|"vtu"|"vtp"` and `options` is a dict (e.g., `{ "binary": "true", "compress": "true" }`).

Labels
- Region labels (volume cells):
  - `mesh.set_region_label(cell, label)`
  - `mesh.region_label(cell)`
  - `mesh.cells_with_label(label)`
  - `mesh.cell_region_ids()`
- Boundary labels (faces):
  - `mesh.set_boundary_label(face, label)`
  - `mesh.boundary_label(face)`
  - `mesh.faces_with_label(label)`
- Named sets:
  - `mesh.add_to_set(kind, name, id)`
  - `mesh.get_set(kind, name)`
  - `mesh.has_set(kind, name)`
- Label name registry:
  - `mesh.register_label(name, label)`
  - `mesh.label_name(label)`
  - `mesh.label_from_name(name)`

Exposed API
- Enums: `EntityKind`, `CellFamily`, `Configuration`
- Struct: `CellShape`
- Class: `Mesh`
  - `Mesh()`, `Mesh(spatial_dim)`
  - `build_from_arrays(spatial_dim, X_ref, cell2vertex_offsets, cell2vertex, families, orders=None, num_corners=None)`
  - `finalize()`, `clear()`
  - `dim()`, `n_vertices()`, `n_cells()`, `n_faces()`, `n_edges()`
  - `bounding_box() -> (min[3], max[3])`
  - `cell_center(c, cfg=Configuration.Reference)`
  - `cell_measure(c, cfg=Configuration.Reference)`
  - `cell_vertices(c) -> list[int]`

Notes
- Faces/edges are generated on `finalize()` if not provided.
- VTK I/O is not currently exposed; can be added later by binding the Mesh IO utilities.

# Elastic Pipe FSI Meshes

This directory contains generated meshes for fluid-structure interaction tests
on a straight elastic pipe.

## Layout

- `coarse/fluid/mesh/mesh-complete.mesh.vtu`: fluid lumen tetrahedral mesh.
- `coarse/solid/mesh/mesh-complete.mesh.vtu`: elastic wall tetrahedral mesh.
- `coarse/solver.xml`: short FSI solver input for the coarse fixture.
- `refined/fluid/mesh/mesh-complete.mesh.vtu`: refined fluid lumen mesh.
- `refined/solid/mesh/mesh-complete.mesh.vtu`: refined elastic wall mesh.
- Each participant mesh has named boundary surfaces under
  `mesh/mesh-surfaces`.

The fluid and solid `fsi_interface.vtp` surfaces use the same cylindrical
interface triangulation so tests can exercise conforming interface coupling.
The fluid tetrahedral meshes use explicit radial rings so the lumen has several
elements across the pipe radius without changing the shared interface surface.

## Regeneration

Install the mesh generation dependencies in the active Python environment:

```bash
python3 -m pip install pyvista tetgen
```

Then regenerate the fixtures:

```bash
python3 Code/Source/solver/Physics/Tests/Unit/Data/ElasticPipe/generate_elastic_pipe_meshes.py
```

The generator rewrites the `coarse`, `refined`, and `manifest.json` outputs.

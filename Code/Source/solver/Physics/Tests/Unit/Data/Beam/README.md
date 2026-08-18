# Ustruct Hex Beam Mesh

This directory contains a long rectangular beam mesh for Ustruct formulation
tests.  The volume mesh is a structured Hex8 grid generated with PyVista:

```text
length = 10.0
width  = 1.0
height = 1.0
cells  = 20 x 2 x 2 Hex8 elements
```

The mesh files are:

```text
mesh/mesh-complete.mesh.vtu
mesh/mesh-complete.exterior.vtp
mesh/mesh-surfaces/left.vtp
mesh/mesh-surfaces/right.vtp
mesh/mesh-surfaces/front.vtp
mesh/mesh-surfaces/back.vtp
mesh/mesh-surfaces/bottom.vtp
mesh/mesh-surfaces/top.vtp
```

Face naming follows the existing beam convention: `left` and `right` are the
x-min and x-max faces, `front` and `back` are y-min and y-max, and `bottom` and
`top` are z-min and z-max.

The static-equilibrium OOP Ustruct setup is:

```text
ustruct_beam_axial_load_oop.xml
ustruct_beam_top_z_load_oop.xml
ustruct_beam_top_z_load_dynamic_oop.xml
```

All cases fix the `left` face in all displacement components and include a
zero pressure Dirichlet anchor on the fixed face for the mixed pressure field.
The axial case applies a steady traction of `1.0e3` in the positive x direction
on the `right` face.  The top-load case applies a steady traction of `-1.0e1`
in the z component on the `top` face.  The dynamic top-load case uses the same
load scale and material settings without `quasi_static=true`, and advances 100
time steps with `Time_step_size = 1/25 = 0.04`.  The dynamic load is a
file-driven follower pressure on the top face, ramped from 0 to `1.0e1` over
the first 0.4 seconds by `top_z_pressure_load.dat`; on the top face this is a
load in the negative z direction.  It writes every 10th step and combines the
saved outputs into a `.pvd` time-series collection.

Run it from this directory with:

```bash
<path-to-build>/svmultiphysics ustruct_beam_axial_load_oop.xml
<path-to-build>/svmultiphysics ustruct_beam_top_z_load_oop.xml
<path-to-build>/svmultiphysics ustruct_beam_top_z_load_dynamic_oop.xml
```

Regenerate the data with:

```bash
python3 Code/Source/solver/Physics/Tests/Unit/Data/Beam/generate_beam_hex_mesh.py
```

TetGen is used by the generator only to validate that the exterior faces form a
closed surface.  TetGen does not generate hexahedral volume meshes; the saved
volume mesh remains Hex8.  To regenerate without the optional TetGen validation,
pass `--skip-tetgen-check`.

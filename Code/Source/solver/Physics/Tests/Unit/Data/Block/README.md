# Ustruct Block Exact Compression Data

This folder contains a displacement-free, pressure-driven uniaxial compression
benchmark on the existing block mesh.  The XML files are identical except for
`Use_new_OOP_solver`.

The case sets `Module_options` to `quasi_static=true`; the new OOP Ustruct
module then drops the rate/inertial terms and the application driver performs a
single quasi-static solve when the BCs are steady.

The OOP XML intentionally uses a tighter FSILS solve than the historical
default (`rel_tol=1e-12`, `abs_tol=1e-12`, restart length 250, and at least 4
Newton updates).  The nearly incompressible pressure field is sensitive to late
small-RHS Newton corrections; the looser `abs_tol=1e-8` default permits
serial/MPI pressure differences even when displacement and velocity are already
in parity.

The target exact incompressible neo-Hookean deformation is

```text
lambda_z = 0.98
lambda_x = lambda_y = lambda_z^(-1/2) = 1.0101525445522108
```

with the affine displacement field

```text
u_x = (lambda_x - 1) X
u_y = (lambda_y - 1) Y
u_z = (lambda_z - 1) Z
```

For `E = 240.56596e6` and `nu = 0.4999999`, the shear modulus is
`mu = 80188658.67924392`.  The applied follower pressure is

```text
p_load = mu * (lambda_z^(-1) - lambda_z^2) = 4811974.1220499845
```

The corresponding constant mixed pressure for the incompressible solution is

```text
p = mu * (lambda_z^(-1) - lambda_z^2) / 3 = 1603991.3740166614
```

Use `assess_ustruct_block_solution.py` to evaluate output fields.  The script
accepts either a VTU/PVTU file or a run directory.  MPI PVTU output is compared
by unique point coordinates, so partition duplicate points do not skew the
metrics.  Legacy output under `1-procs/` is resolved automatically.

```bash
./assess_ustruct_block_solution.py /path/to/run --velocity-mode skip
```

For serial-vs-MPI or legacy-vs-OOP comparison, pass the reference output. Values
are matched by point coordinates, so the comparison is insensitive to output
point ordering and MPI partition duplicates.

```bash
./assess_ustruct_block_solution.py /path/to/oop/run \
  --reference-vtu /path/to/legacy/run/1-procs/ustruct_uniaxial_compression_001.vtu
```

Expected same-parameter OOP serial/MPI-2 accuracy with the XML settings in this
folder is pressure `Linf` below `5e-4 Pa`, displacement `Linf` below `1e-12`,
and velocity `Linf` below `1e-12`.

The analytical velocity target defaults to zero.  For transient, scheme-specific
checks, use `--velocity-mode scaled-displacement --velocity-scale SCALE`.

## Affine OOP Ustruct Patch Cases

This folder also contains generated OOP-only affine patch cases that compare
against exact analytical fields but are intentionally not wired into pytest.
They use small cube, single-hex, and 2D quadrilateral patch meshes,
`Module_options=quasi_static=true`, and constant face data supported by the
current new OOP Ustruct XML path.

The generated files are:

```text
ustruct_volumetric_dilation_st91_oop.xml
ustruct_volumetric_compression_st91_oop.xml
ustruct_volumetric_dilation_quadratic_oop.xml
ustruct_volumetric_compression_quadratic_oop.xml
ustruct_volumetric_dilation_m94_oop.xml
ustruct_volumetric_compression_m94_oop.xml
ustruct_isochoric_tension_z_oop.xml
ustruct_isochoric_compression_z_oop.xml
ustruct_simple_shear_oop.xml
ustruct_mixed_volumetric_deviatoric_st91_oop.xml
ustruct_rigid_rotation_z_180_oop.xml
ustruct_2d_volumetric_dilation_st91_oop.xml
ustruct_2d_isochoric_shear_oop.xml
```

`affine_patch_cases.json` stores each case's exact deformation gradient,
Jacobian, pressure, derived field targets, and default tolerances.  Re-run
`generate_ustruct_affine_patch_cases.py` after editing the case definitions;
the generated `*_001.vtu` files in this folder are saved reference outputs from
the current OOP solver.  Each case pins the constant analytical pressure with a
pressure Dirichlet anchor so the mixed pressure nullspace does not hide
pressure-law errors.

The volumetric cases use `F = lambda I`, with `lambda = 1.02` for dilation and
`lambda = 0.98` for compression.  The exact mixed pressure is computed from
the active volumetric law:

```text
ST91:      p = K/2 * (1/J - J)
quadratic: p = K * (1 - J)
M94:       p = K * (1/J - 1)
```

The isochoric axial cases use
`F = diag(lambda_z^(-1/2), lambda_z^(-1/2), lambda_z)`, with
`lambda_z = 1.02` for tension and `lambda_z = 0.98` for compression.  Since
`J = 1`, the exact pressure is zero.

The mixed volumetric/deviatoric case uses `F = diag(1.04, 0.99, 0.97)` with
the ST91 pressure law.  It exercises a nonuniform diagonal stretch with
simultaneous volumetric and deviatoric content.

The rigid-rotation case uses `F = diag(-1, -1, 1)`, a 180 degree rotation about
the z axis on a single Hex8 element.  It is an objectivity check: `J = 1`, the
Green-Lagrange strain is zero, and the stress target is zero.

The 2D cases use `mesh/quad-2x2`: one uniform ST91 dilation with
`F = 1.02 I`, and one isochoric affine shear with exact side tractions.  The
small local mesh keeps these as quick patch benchmarks instead of large Square
regression runs.

The OOP VTK path can write `Displacement`, `Velocity`, `Pressure`, `Def_grad`,
`Jacobian`, `Divergence`, `Strain`, `Stress`, `Cauchy_stress`, and
`VonMises_stress`.  The assessor compares the primary fields by default and can
check the derived fields with `--fields`.

The shear case uses an affine shear-compression deformation

```text
F = [[1, gamma,     0],
     [0, 1 - gamma, 0],
     [0, 0,         1]]
```

with `gamma = 0.001`.  The `Y0` and `Y1` faces carry exact affine displacement
BCs, while the natural `X` and `Z` faces carry exact PK1 tractions decomposed
into component traction BCs.  This keeps the XML within the current OOP
restriction that a displacement face cannot also carry a displacement traction.

Run one case manually, then assess it:

```bash
<path-to-current-svmultiphysics-build>/svmultiphysics \
  ustruct_volumetric_dilation_st91_oop.xml

./assess_ustruct_affine_patch.py ustruct_volumetric_dilation_st91 .
```

To check all exact fields for one case:

```bash
./assess_ustruct_affine_patch.py ustruct_volumetric_dilation_st91 . \
  --fields Displacement Velocity Pressure Def_grad Jacobian Divergence \
  Strain Stress Cauchy_stress VonMises_stress
```

For an MPI run, pass the run directory or the `.pvtu` file:

```bash
mpirun --oversubscribe -np 4 <path-to-current-svmultiphysics-build>/svmultiphysics \
  ustruct_simple_shear_oop.xml

./assess_ustruct_affine_patch.py ustruct_simple_shear .
```

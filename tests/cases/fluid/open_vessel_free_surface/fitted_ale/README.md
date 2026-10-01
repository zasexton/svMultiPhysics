# Fitted-ALE open-vessel decks

Decks with a fitted free surface: the liquid mesh moves with a coupled mesh
displacement and the free surface is a boundary of the mesh
(`Physics/Docs/NavierStokesFreeSurface.md`, "Fitted ALE free surfaces";
tracker milestone M5, decision D5).

| Deck | Dim | Kinematics | Walls (fluid / mesh) | Status |
|---|---|---|---|---|
| `solver.xml`, `mesh/water/` | 2D, 1 m x 0.5 m, 2 x 2 quadrilaterals | legacy `Nitsche`, pressure gauge | no-slip / pinned | legacy, unchanged (pinned by `test_OpenVesselExamples.cpp`); face files disjoint |
| `spheric_test10_lateral_water_1x/` | 3D, SPHERIC Test 10, 315 nodes | legacy `Nitsche`, pressure gauge | no-slip / pinned | legacy, unchanged (pinned); **face files overlap** (below) |
| `spheric_test10_lateral_water_1x_meshnitsche/` | 3D, same mesh as the legacy 3D deck | `MeshNitsche` | free slip / sliding | new |
| `spheric_test10_lateral_water_1x_2d_meshnitsche/` | 2D x-y section, 120 x 12 cells (1,573 nodes) | `MeshNitsche` | free slip / sliding | new |

The legacy `Nitsche` mode replaces the normal dynamic condition on the fluid
and needs a pressure gauge; it is not a free-surface model (WP-9 note,
"Step-12 failure"). Whether to retire the legacy `Penalty`/`Nitsche`
kinematics is an open decision; their decks and tests are kept as they are.

## MeshNitsche decks

Written by `../generate_spheric_test10_fitted_decks.py`:

- SPHERIC Test 10, lateral water 1x: tank 0.9 m long (x), 0.062 m broad (z),
  fill height 93 mm; water `rho = 998.2`, `mu = 1.003e-3`, `g = 9.81`;
  zero surface tension (capillary length 2.7 mm against a 0.9 m tank).
- `Kinematic_enforcement=MeshNitsche` with `gamma_N = 10`, the `Free`
  tangential policy, harmonic mesh-velocity extension (`Kappa = 1`), as in
  `free_surface_benchmarks/fitted_sloshing_2d`.
- Walls: free slip for the fluid and sliding for the mesh, the same
  zero-valued `Dir` input with `Effective_direction` in both equations. The
  contact line can move; the wall boundary layers (about 1 mm) are far below
  the mesh size in any case.
- Hydrostatic initial pressure in the mesh file, no pressure gauge (the
  natural free-surface traction fixes the pressure level), Eigen sparse LU,
  `dt = 1 ms`, 100 steps, fluid equation before `mesh_motion`.
- Face sets: a boundary facet belongs to the tank plane on which all its
  vertices lie; the sets are checked to be disjoint and to cover the
  boundary, and each facet records its parent cell as `GlobalElementID`.
- The committed decks hold the tank at rest. `--roll-forcing
  lateral_water_1x.txt --output-dir DIR` writes a forced run directory from
  the published roll record (`../fetch_spheric_test10_reference.py`): the
  tank-frame body force (rotated gravity, Euler and centrifugal terms) as a
  temporal and spatial values table, and in 3D the angular velocity for the
  Coriolis term. The solver's Coriolis term exists only for 3D meshes, so
  the 2D forced run omits it (`2 |Omega| |u|` with `|Omega| <= 0.27 rad/s`).
  The solver interpolates such a table at every quadrature point from the
  eight nearest table nodes; before `d92af1cd` it scanned all nodes for each
  evaluation (about 18 s per step on the 2D deck), now a bucket grid finds
  the same nodes. `--forcing-x-only` (diagnostic) evaluates the Euler and
  centrifugal terms at mid-depth, which the solver interpolates along x
  only; it changes the start-up velocity by 34% and the run-up height by
  2% (below).

`../analyze_spheric_test10_fitted_run.py RUN_DIR [--reference
lateral_water_1x.txt]` reports per output the liquid volume, the largest
speed, the largest wall-normal velocity and displacement over all wall
nodes, mesh quality (smallest angle in 2D; smallest dihedral angle and
volume ratio in 3D), contact-point heights and the Sensor 1 pressure (left
wall at 93 mm, interpolated along the wall nodes, 0 when dry).

## The legacy 3D face files and the contact-line leak

`../generate_validation_meshes.py` selects a boundary face for a tank plane
when its centroid lies within `0.35 h` of the plane (`write_case`,
`tol = max(0.35 * h, 1.0e-8)`, and `plane_predicate`). On the legacy 3D mesh
this puts all 88 wall faces of the top cell row into `free_surface.vtp` as
well (hence y = 0.0698 to 0.093), the bottom-row faces of the side walls
into `wall_bottom.vtp`, and the edge faces of every wall into its
neighbours (12 overlapping pairs). The face files also record face indices,
not parent cells, as `GlobalElementID`.

A boundary face carries one label, which the application set to the last
face file listed (`Application/Translators/MeshTranslator.cpp`), and every
boundary condition acts on the faces of its label
(`FE/Assembly/MeshAccess.cpp`, `forEachBoundaryFace`). With `free_surface`
listed last, the top-row wall faces became free-surface faces, loaded by
the natural traction against the hydrostatic wall pressure, and no
wall-labeled face was left at the nodes of the free-surface edge, so the
fluid and mesh wall conditions missed them. The application now warns about
every such overlap.

Controls, 100 steps of 1 ms at rest (binary `35a81fd3` or the same with the
overlap warning; job `46129890`):

| Face files | Solver input | Volume change | Max wall-normal velocity | Max speed |
|---|---|---|---|---|
| legacy (overlapping) | MeshNitsche deck (free slip, sliding) | -4.2% | 1.0 m/s | 1.2 m/s |
| new (disjoint) | the earlier leaking variant (no-slip fluid walls, sliding mesh, job `46109392` input) | 0 | 0 | 1.3e-16 m/s |
| new (disjoint) | MeshNitsche deck | 0 | 0 | 8.4e-16 m/s |

## Results

Binary `35a81fd3` (the fitted code of this branch) unless noted; analysis
with `analyze_spheric_test10_fitted_run.py`. Raw output:
`/scratch/users/zsexton/svmp-dev-fitted2/runs/spheric/` and `.../runs/smoke/`.

**At rest** (committed decks, no forcing, `dt = 1 ms`, one Newton iteration
per step):

| Deck | Steps | Job | Max volume change | Max speed | Wall-normal velocity / displacement | Mesh quality |
|---|---:|---|---:|---:|---:|---|
| 3D | 100 | `46128278` | 0 | 8.4e-16 m/s | 0 / 0 | unchanged |
| 3D | 1,000 | `46129890` | 0 | 7.9e-15 m/s | 0 / 0 | unchanged |
| 2D | 100 | `46128278` | 0 | 2.9e-15 m/s | 0 / 0 | unchanged |
| 2D | 1,000 | `46129890` | 1.7e-16 | 3.2e-14 m/s | 0 / 0 | unchanged |

The volume is that of the current mesh in the outputs; "0" means equal to
the initial volume in double precision.

**3D, forced, 1 s** (job `46132346`; roll forcing with the Coriolis term,
x-only body-force table): the wall-normal velocity and displacement stay
exactly zero at every wall node during the motion, including the nodes
shared by two sliding walls (union of the component-wise conditions) and
the contact line; the volume changes by at most 1.0e-4; the smallest
dihedral angle falls from 27.3° to 16.5° and the smallest volume ratio to
0.65; Sensor 1 peaks at 3.5 mbar at 0.92 s (measured: 3.9 mbar at 0.95 s).
On this mesh (45 mm cells) this is a check of the wall conditions, not a
validation.

**2D, forced with the x-only table** (job `46132346`, `dt = 1 ms`): the
run reached 1.51 s of 8.35 s. Until 1.41 s the motion is smooth (volume
change below 5e-6, smallest angle between 29.5° and 44°, recovering as the
wave returns); from 1.42 s the wave returning to the left wall runs up the
wall (contact point from 91 to 124 mm in 80 ms), the first cells next to
the wall shear (smallest angle 31° at 1.415 s, 4° at 1.475 s, 0.12° at
1.51 s, near x = 8 mm, y = 55 mm), the volume error grows to 3.3e-4 and
Newton fails at step 1,510. Against the full table (below) the x-only
table changes the velocity by 34% at 0.2 s, when the start is driven by the
angular acceleration (the y-dependent Euler term is rotational), and the
contact-point heights by up to 0.7 mm of a 35 mm excursion (2%) between 0.7
and 1.4 s; both runs fail at 1.51 s. The full table is the default.

**2D, forced with the full table** (`forced2d_exact`, binary `4a62a971`,
job `46155727`, `dt = 1 ms`, 2.6 s per step): **the run reached 1.514 s of
8.35 s** (step 1,514; the experiment's 71.5 mbar impact is at 7.30 s). The
first 0.2 s are bitwise identical to the same run with the earlier binary
(`forced2d_smoke`, brute-force node search).

| Window | Volume change | Smallest angle | Max speed | Sensor 1 |
|---|---:|---:|---:|---|
| 0 to 1.40 s | at most 4.6e-6 | 29.5° (at 0.94 s), recovering to 34° at 1.40 s | 0.54 m/s | first run-up: 3.38 mbar at 0.925 s; measured 3.86 mbar at 0.953 s (3.54 mbar as a 5 ms mean, at 0.975 s); RMS difference 0.18 mbar over 0 to 1.4 s |
| 1.40 to 1.51 s | grows to 2.6e-4 | 23° at 1.45 s, 12° at 1.46 s, 4.2° at 1.48 s, 0.6° at 1.51 s | 5.1 m/s at the end | meaningless after 1.45 s |

How it ends: the wave returning from the right wall runs up the left wall
(contact point from 90 mm at 1.42 s to 123 mm at 1.50 s). The harmonic
mesh-velocity extension moves the wall-adjacent column of cells with the
sliding contact point, the cells under the thin run-up shear (worst cells at
x = 2 to 10 mm, y = 70 to 105 mm), the volume error and the speed grow with
the distortion, and Newton stops converging at step 1,514. The wall
conditions hold to the end (wall-normal velocity and displacement exactly
zero). The mesh-velocity operator has no restoring term toward the reference
mesh, so the distortion of the run-up is not undone; how to control mesh
quality (smoothing, a restoring term, remeshing) is a separate decision and
no tuned fix was added.

# SPHERIC Test 05 wet-bed dam break, d = 38 mm (unfitted level set, 2D)

Written by `../../generate_validation_meshes.py --case spheric_test05`.  Status: regenerated
2026-10-26 for the SPHERIC campaign (tracker M1, M6); the profiles of this deck have not
been compared with experiment yet.

## Setup

- The published two-dimensional flume section: 1.2 m long, 0.18 m high; a 0.15 m water
  column behind the gate at x = 0.38 m and a wet bed of 38 mm downstream.
- The gate is removed instantaneously at t = 0 (the experiment lifts it at 1.5 m/s).
- Walls left, right and bottom are free-slip (only the normal velocity is fixed, decision
  D58); the top is open (natural traction, level-set outflow).  Zero surface tension: a
  transport and impermeability benchmark, not a wetting one.
- Level set: harmonic PDE velocity extension with monolithic coupling (D55); kinematic
  reconciliation (D14), sign-definite patch bounds (D21) and triggered interface-preserving
  redistancing (D39), as the M1-M4 protocols.
- Mesh: 202 x 31 cells of near-isotropic size h = 6 mm (the
  June deck's finest spacing), each split into two Triangle3 with the diagonal alternating
  with the cell parity; 6496 points, 12524 triangles.  The gate, the wet-bed
  height and the column height lie midway between mesh lines.  The four face files are
  disjoint (every face vertex on the wall).

## The June deck it replaces (decision D51 item 2)

The June deck was a three-dimensional thin extrusion of this section: 30 mm across with
2 cells (h_z = 15 mm), h_x = 22 mm, h_y = 5-10 mm, 14256
tetrahedra, with free-slip front and back planes.  On the current protocol it stopped at
step 193 (t = 0.0965 s) on the small-cut aggregation guard: at the surge front the liquid
formed wedges on the two free-slip planes, and with only 2 cells across no full cell lay
within 8 face steps of a cut cell.  Its face files also overlapped at the tank edges (the
0.35 h centroid selection, kept by the `Allow_overlapping_face_files` opt-out of D35).
A face keeps the label of the last face file listed.  Of the 112 shared faces, 108 were
front/back faces next to the bottom, which kept their own free-slip label, and 4 were
left/right faces next to the bottom, which took the bottom label, no-slip like their own
wall; checked face by face on the committed files, no face had a different condition
(the tracker's M5 item expected bottom faces with the free-slip label; there were none).
The 2D section has no spanwise direction and its face files are disjoint.

## Criteria (tracker M1)

- Free-surface profile RMSE against the experimental profile at most 0.02 m at 0.156,
  0.219 and 0.281 s (`compare_test05_profiles.py`).  June, on the extruded deck with
  cadence-1 volume correction: 0.0184 m (d18) and 0.0158 m (d38) at 0.156 s.
- No false wetting: no vertex with phi < 0 above y = 0.16 m (June d18 wetted the
  lid corners after 0.22 s).
- The later profiles (to 0.593 s, tracker M6) are reported.

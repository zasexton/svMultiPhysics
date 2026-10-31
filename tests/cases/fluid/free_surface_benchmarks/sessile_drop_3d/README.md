# sessile_drop_3d

A 3D liquid spherical cap on a flat wall relaxes from a non-equilibrium shape
to the spherical cap with the Young angle (tracker M4, "3D sessile"; 90
degrees first, 60 and 120 degrees since 2026-10-22). It is the 3D counterpart
of `sessile_drop_2d`, with the same single contact-angle mechanism of
decision D4:

- the angle is imposed only by the variational Young term in the momentum
  equation (the `SurfaceStress` line term along the contact line, which in
  3D is a curve: one segment per cut wall face of a Tet4 cell);
- Navier slip on the wetted wall, with the slip length as a physical input;
- a strong no-penetration condition on the contact wall;
- no reinitialization (as in `sessile_drop_2d`); kinematic reconciliation,
  the sign-definite patch bounds and, since decision D39, the triggered
  interface-preserving redistancing (the 3D zero-surface version) are on;
  `--interface-preserving-redistancing off` writes the decks without it.

**Status: tolerances approved on 2026-10-10 (tracker D38); the protocol
enables the interface-preserving redistancing since D39 (2026-10-11). Decision
D43 (2026-10-19, before the `R/h = 16` `dt/2` run) averages the sector angle
over rotations of the sector partition and judges the time-step check at
`R/h = 16`. The equilibrium angles 60 and 120 degrees were added on
2026-10-22, before any run at them, with every criterion unchanged; decision
D48 (2026-10-22) approved the extension as written, before its first run. At
90 degrees the Young line term (`-gamma cos(theta_e)` times the line integral
of `v.m`) vanishes, so only the new angles exercise it in 3D.**

Files:

| File | Role |
|---|---|
| `generate_case.py` | writes `solver.xml`, the Tet4 mesh with the initial fields, the boundary face files and `case.json` for one level |
| `verify.py` | reads the solver output, computes the metrics, applies `tolerances.json`; `--symmetry-check` compares a quarter-domain and a full-domain run |
| `tolerances.json` | acceptance criteria and their sources (approved, D38; amended by D43; extended to 60 and 120 degrees by D48) |
| `tests/test_free_surface_benchmark_sessile_drop_3d.py` | checks of the two scripts on synthetic data |

The Tet4 writers, the Kuhn split and the FSILS and transport input blocks are
imported from `../static_sphere_3d/generate_case.py`; the exact liquid volume
and the edge crossings from `../static_sphere_3d/verify.py`; the circle fit and
the one-side contact angle from `../sessile_drop_2d/verify.py`.

## Physical setup

Nondimensional units: `rho = gamma = 1`, `R = 1` the radius of the
equilibrium cap, zero gravity, void exterior with `p_ext = 0`, liquid
`phi < 0`. The contact wall is `y = 0` (`wall_bottom`, normal into the solid
`(0, -1, 0)`); angles are measured through the liquid.

- **Equilibrium.** A cap of radius `r` and angle `theta` has volume
  `pi r^3 (2 - 3 cos(theta) + cos^3(theta))/3`, base radius `r sin(theta)`
  and apex height `r (1 - cos(theta))`. The equilibrium cap has `R = 1` at
  every angle. At `theta_e = 90` degrees it is the hemisphere: volume
  `2 pi/3`, base radius and apex height 1. At 60 degrees: volume 0.6545,
  base radius 0.8660, apex height 0.5. At 120 degrees: volume 3.5343, base
  radius 0.8660, apex height 1.5.
- **Initial state** (sampled analytic shape): the cap of the same volume
  30 degrees away from `theta_e`, the angle pairs of `sessile_drop_2d`;
  `phi = |x - c_0| - r_0` at the vertices, `u = 0`, `p = 2 gamma/r_0`
  everywhere.

  | `theta_e` | `theta_0` | motion | `r_0` | base radius | apex height |
  |---:|---:|---|---:|---:|---:|
  | 60 | 90 | advancing (spreading) | 0.6786 | 0.6786 | 0.6786 |
  | 90 | 120 | advancing (spreading) | 0.8399 | 0.7274 | 1.2599 |
  | 120 | 90 | receding | 1.1906 | 1.1906 | 1.1906 |
- **Viscosity and slip.** `La = rho gamma (2R)/mu^2 = 12` (`mu = 0.4082`,
  viscous time 2.449) and `l_s = R/8`, as in 2D (`l_s/h = 1` and 2 at
  `R/h = 8` and 16).

## Domain, mesh and symmetry

- **Quarter domain (protocol):** `[0, 1.25] x [0, 1.625] x [0, 1.25]` at 90
  degrees, `[0, 1.125] x [0, 1] x [0, 1.125]` at 60 and `[0, 1.5] x [0, 1.75]
  x [0, 1.5]` at 120 (both caps plus a 0.25 R margin, rounded to the coarse
  grid), the drop axis on the line `x = z = 0`. The planes `x = 0` (`sym_x`) and `z = 0`
  (`sym_z`) are symmetry planes: `Dir 0` with `Effective_direction` on the
  normal component only (free slip). They are **not** contact walls (not in
  the contact-line wall list): the contact rules apply to `wall_bottom` only,
  and the natural condition of `SurfaceStress` on a free-slip plane is the
  90-degree mirror condition, as at the mirror walls of
  `fitted_capillary_wave_2d`. The outer walls are no-slip and stay dry (at
  least 2 cells from the drop).
- **Full domain (`--domain full`):** the mirrored box `[-1.25, 1.25] x
  [0, 1.625] x [-1.25, 1.25]` with four contact-line quadrants. It is the
  symmetry check: `verify.py --symmetry-check QUARTER FULL` compares the
  velocity and `phi` at the shared vertices, the volume, the base radius and
  the sector angles on the shared quarter at every common output.
- **Mesh:** each grid cube is split into the six Kuhn tetrahedra, reflected by
  the parity of the global cube index as in `static_sphere_3d`, so the mesh is
  symmetric about every grid plane and the full mesh restricted to
  `x, z >= 0` is the quarter mesh (checked by the tests). `R/h = 8`: 7,800
  tetrahedra (quarter), 31,200 (full); `R/h = 16`: 62,400 (quarter). At 60
  degrees 3,888 and 31,104, at 120 degrees 12,096 and 96,768 (quarter,
  `R/h = 8` and 16). Face
  files are disjoint (each boundary triangle has all its vertices on one
  plane).
- The drop axis lies on the symmetry line, so the transcendental centre
  offset of the 2D case is not available; the sampled initial cap passes no
  closer than `0.0017 h` to a vertex and the initial contact line `0.012 h`
  to a wall vertex at `R/h = 8` (reported in `case.json`). At 60 degrees the
  smallest gaps are `0.044 h` and `0.044 h` at `R/h = 8` and `0.0051 h` and
  `0.041 h` at 16; at 120 degrees `0.015 h` and `0.038 h`, and `0.0037 h`
  and `0.023 h`. The dry outer walls stay at least two cells away (exactly
  2.0 at 120 degrees and `R/h = 8`).

## Time step and run length

- **Protocol (decisions D25, D29 and D38):** the D25 fixed step of the 3D static
  sphere at La = 12, `dt <= 0.02`, with the lagged normal-increment capillary
  term (`Surface_tension_semi_implicit=NormalIncrement`), at both levels; the
  run lasts 10 viscous times with 200 outputs (D29), so `dt = T/1400 =
  1.7496e-2`, 1.40 and 3.97 times the capillary limit `dt_B` of `R/h = 8` and
  16.
- `--dt-divisor 2` halves the step with unchanged output times. The
  time-step check is judged at `R/h = 16` (decision D43, as in 2D); the
  `R/h = 8` pair is reported, not gated. Before D43 it was judged at
  `R/h = 8` only (D25).
- `--dt-rule capillary-limit` uses each level's `dt_B` without the lagged
  term (the 2D sessile rule; 2,000 and 5,600 steps).

## Metrics (`verify.py`)

| Metric | Definition |
|---|---|
| contact line | zero segments of `phi_h` on the wall faces of the tetrahedra (`y = 0`) |
| single-partition sector angle | the line is divided into azimuthal sectors of 30 degrees about the drop axis (3 in the quarter domain, 12 in the full domain); in each sector a circle is fitted to the meridional projection (distance from the axis, height) of the interface points with `0 <= y <= 0.25 R_ref`, and the angle is read at the circle point at the mean radius of the sector's wall crossings, as `sessile_drop_2d` does for each side |
| sector contact angle (D43) | the mean of the single-partition angle over 21 rotations of the partition by -5, -4.5, ..., +5 degrees. A sector keeps its identity (its nominal centre azimuth: 15, 45, 75 degrees in the quarter domain) in every rotation. In the quarter domain the outer edges stay on the symmetry planes, so a rotation moves only the interior boundaries (`[0, 30+s)`, `[30+s, 60+s)`, `[60+s, 90]`); in the full domain the partition rotates periodically. The single-partition angles and the spread over the rotations are reported |
| `contact_angle_max_error_degrees` (gated) | largest `abs(theta_sector - theta_e)` of the rotation-averaged sector angles at the end of the run; the mean over the sectors and the single-partition value (`contact_angle_max_error_degrees_single_partition`) are reported |
| cell angles (reported) | the angle of the P1 interface in each contact tetrahedron, length-weighted mean and largest deviation |
| `base_radius_relative_error` | mean distance of the wall crossings from the drop axis against `R_ref sin(theta_e)`; the spread along the line is reported |
| `apex_height_relative_error` | largest height of the interface points against `R_ref (1 - cos(theta_e))` |
| `liquid_volume_relative_drift_max` | max over the outputs of `abs(V(t) - V(0))/V(0)`, V the exact volume of `{phi_h < 0}` (D11) |
| `max_speed_growth_ratio` | final `max|u_h|` over the largest value for `T/2 <= t <= 3T/4` |

The reference cap has the volume of the sampled initial level set (divided by
1/4 in the quarter domain). The drop axis is `x = z = 0` in the quarter
domain and the centre of the circle fitted to the wall crossings in the full
domain. On sampled exact caps (90 and 120 degrees) the rotation-averaged
sector angle is within 0.68 degrees of the exact angle at `R/h = 8` and 0.08
degrees at `R/h = 16`, the single-partition angle within 0.63 and 0.09
degrees (the cell-gradient angles are off by up to 7 and 3.3 degrees, so
they are reported only); the base radius and apex height are within 0.25% at
`R/h = 8` and 0.05% at 16. On the sampled equilibrium caps of the new angles
(2026-10-22) the rotation-averaged sector angle is within 0.25 degree (60)
and 0.39 degree (120) at `R/h = 8` and within 0.064 and 0.044 degree at 16;
the base radius and apex height are within 0.41% and 0.53% (60) and 0.06%
and 0.18% (120) at `R/h = 8`, and within 0.14% at 16.

## Tolerances (D38, amended by D43)

The criteria apply to each equilibrium angle (60, 90, 120 degrees) as a
separate study, with the same limits; `verify.py` takes the runs of one angle
at a time and stops with an error when given several.

| Criterion | Limit | Where | Source |
|---|---|---|---|
| `contact_angle` | largest sector error at most 2 degrees, decreasing | `R/h = 16`; trend over 8/16 | tracker M4 working criterion, read as in `sessile_drop_2d` (largest error over the contact line); gating level as D25 |
| `base_radius` | at most 2% | `R/h = 16` | tracker M4 working criterion |
| `apex_height` | at most 2% | `R/h = 16` | tracker M4 working criterion |
| `volume_drift` | at most 1e-4 | every run | D1, D11 |
| `no_velocity_growth` | ratio at most 1, not gated when the final capillary number `mu max|u_h|/gamma` is below 1e-3 (`rest_capillary_floor`) | every run | D1; rest floor D39 |
| `time_step` | end-state changes at most 0.1 degree, 0.1%, 0.1% between `dt` and `dt/2`; the angle change is the largest change of a rotation-averaged sector angle, sectors matched by their nominal centre | `R/h = 16`; the `R/h = 8` pair is reported, not gated | D25 (one dt/2 level in 3D), D38 (it validates the fixed-step rule) and D43 (judged at `R/h = 16`); the limits follow the 2D sessile time-step decision (D35 item 4) |

### Time-step check and the sector angle (D43)

The D38 check at `R/h = 8` failed on the protocol runs of job 51 (arm64,
`e7080344-ltopgo-mumps`): the largest sector angle change was 0.644 degree,
while the base radius and apex height changed by 5.7e-5 and 9.9e-6. About
0.52 degree of it is the measurement. On the symmetric mesh two wall
crossings lie on the 30- and 60-degree sector boundaries (azimuth within
1e-4 degree); a `phi` change of about 1e-5 moves each into the neighbouring
sector, and the angle of the sector it leaves jumps by 0.2-0.3 degree. The
`dt` run moved one at `t = 10.3`, the `dt/2` run the other at `t = 20.0`,
with no solver event at either time. At `R/h = 8` the single-partition angle
varies by 0.07-0.15 degree (standard deviation) under rotations of the
partition by up to 5 degrees, so it cannot resolve the 0.1-degree limit; at
`R/h = 16` the variation is 0.03-0.05 degree and such a jump costs about
0.014 degree. Decision D43 therefore averages the sector angle over the
rotations at every level and judges the time-step check at `R/h = 16` with
unchanged limits. The `R/h = 8` result stays recorded as a fail under the
D38 definition (analysis: `sessile_3d/dtcheck-e7080344/`).

On the job 51 outputs the averaged angle gives a largest sector error of
0.964 degree at `R/h = 8` and 0.106 degree at `R/h = 16` (single partition:
1.016 and 0.099), and a `dt`/`dt/2` change of 0.144 degree at `R/h = 8`
(single partition: 0.644), still outside the limit and reported.

## How to run

```bash
B=tests/cases/fluid/free_surface_benchmarks/sessile_drop_3d
A=90   # equilibrium angle: 60, 90 or 120 (--contact-angle); one angle per verify.py call
for L in 8 16; do python3 $B/generate_case.py --level $L --contact-angle $A --output-dir $OUT/L$L; done
python3 $B/generate_case.py --level 16 --contact-angle $A --dt-divisor 2 --output-dir $OUT/L16_dt2   # time-step check (D43)
python3 $B/generate_case.py --level 8 --contact-angle $A --dt-divisor 2 --output-dir $OUT/L8_dt2     # reported, not gated
# 3D production configuration (free_surface_benchmarks/README.md, "Run configurations"):
#   #SBATCH --nodes=1 --ntasks=8 --cpus-per-task=3   # with <Assembly_threads>3</Assembly_threads> in the deck
#   export OMP_NUM_THREADS=1
#   export SVMP_PDE_EXTENSION_FACTORIZATION=mumps_serial   # with ScaLAPACK and METIS on LD_LIBRARY_PATH
#   mpiexec -n 8 --map-by slot:PE=3 --bind-to core svmultiphysics solver.xml 2>&1 | gzip -1 > solver_run.log.gz
python3 $B/verify.py $OUT/L8 $OUT/L16 $OUT/L16_dt2 $OUT/L8_dt2 --json $OUT/verify.json
```

## Diagnostic smoke run (2026-10-11, not protocol evidence)

Decks from `9858d0f5` with `--max-steps 10`, `R/h = 8`, production binary
`2c4c9cdd-ltopgo-mumps`, 8 ranks x 3 assembly threads, `mumps_serial`, ghost
layers unset (job `47334236`; output
`/scratch/users/zsexton/free-surface-benchmarks/sessile_3d/smoke-9858d0f5/`).

| Domain | Tetrahedra | First step (with setup) | Later steps | Newton / outer passes per step | Peak memory per rank |
|---|---:|---:|---:|---|---:|
| quarter | 7,800 | 89 s | 8.6 s | 4-5 / 4 (first step 7 / 5) | 0.78 GB |
| full | 31,200 | 61 s | 24.9 s | 4-5 / 4 (first step 13 / 10) | 1.06 GB |

- Kinematic reconciliation converged at every step (6 to 8 iterations).
  The sign-definite patch bounds bounded one DOF per step and prevented no
  sign change. The angle-preserving wall maintenance belongs to the
  reinitialization, which the protocol leaves off, so the smoke run does not
  exercise it.
- Symmetry check (`verify.py --symmetry-check`, at each of the 10 outputs):
  the quarter and full runs differ by up to `|du|/max|u| = 7.7e-2`,
  `|dphi| = 3.8e-3`, 0.22 degrees in the sector angles and 3e-4 in the base
  radius; the liquid volume of the full run's shared quarter differs by up to
  1.9e-4 (4 x quarter against the whole full drop: 1.4e-5). The full-domain
  solution is itself not mirror-symmetric on its mirror-symmetric mesh by the
  same amount (`|phi - phi_mirror|` up to 3e-3 after one step and 1.3e-2
  after ten; `|u - u_mirror|` 0.35 of `max|u| = 0.81` after one step and
  0.027 of 0.33 after ten), so the differences come from the solver's own
  asymmetry, not from the symmetry planes.
- Estimated cost of the protocol on one node (8 x 3): `R/h = 8`, 1,400 steps,
  about 3.4 h; `R/h = 8` at `dt/2`, about 6.7 h; `R/h = 16` (eight times the
  cells), about 45-50 s per step, about 19 h; about 29 h in total.

## Cost of the added angles (estimate, 2026-10-22)

Scaled from the measured 90-degree protocol runs on `e7080344-ltopgo-mumps`,
8 ranks (jobs 51 and 74: `R/h = 8` 1.14 h, `R/h = 16` 5.44 h, `R/h = 16`
`dt/2` 7.73 h; batch peak memory 11.4-11.6 GB at `R/h = 16`), by the cell and
interface size (60 degrees about 0.5 times, 120 degrees about 1.5 times),
an allowance of 1.0-1.3 (60) and 1.0-1.5 (120) for the dynamics, and a factor
0.60-0.75 for the speed-ups merged since `e7080344`. With a main-line
production binary built with MUMPS and `mumps_serial`, 8 ranks:

| `theta_e` | `R/h = 8` | `R/h = 16` | `R/h = 16`, `dt/2` | memory at `R/h = 16` |
|---:|---:|---:|---:|---:|
| 60 | 0.3-0.6 h | 1.6-2.7 h | 2.3-3.8 h | about 6 GB |
| 120 | 1.0-1.9 h | 4.9-9.2 h | 7.0-13.0 h | about 18 GB |

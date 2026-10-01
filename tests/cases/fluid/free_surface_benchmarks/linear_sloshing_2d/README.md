# linear_sloshing_2d

The first antisymmetric standing wave in a 2D rectangular tank, released from
rest with a small cosine displacement of the free surface. Gravity is the only
restoring force (zero surface tension). The unfitted level-set solution is
compared with linear theory over a three-level refinement study: the
frequency and damping rate of the free-surface elevation at a wave gauge, and
liquid-volume conservation (milestone M1 in
`Documentation/free_surface_program_tracker.md`).

It wraps the older regression case
`tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/linear_sloshing_2d/`
(same mode and tank aspect ratio) into the benchmark convention, with two
changes explained below: free-slip walls instead of prescribed analytic wall
velocity, and a viscosity large enough for the damping to be measured.

The protocol was revised on 2026-09-30 for decisions D9 to D12 of the
tracker: the level set is advected with the PDE extension of the fluid
velocity (D9), space and time errors are studied separately (D10), the volume
criterion gates the maximum deviation over the run (D11), and the damping
error is gated at 5% (D12).  The first protocol and its results are kept in
"Results" below.

Files:

| File | Role |
|---|---|
| `generate_case.py` | writes `solver.xml`, the mesh with the initial fields, the wall faces and `case.json` (including the reference frequency and damping rate) for one level |
| `verify.py` | reads the solver output of the levels, fits frequency and damping, applies `tolerances.json` |
| `tolerances.json` | acceptance criteria and their sources, fixed on 2026-09-29 and revised for D9 to D12 on 2026-09-30, each time before the first run |
| `tests/test_free_surface_benchmark_linear_sloshing_2d.py` | checks of the two scripts on synthetic data |

## Physical setup

Nondimensional units: density `rho = 1`, gravity `g = 1` along `-y`, tank
length `L = 1`. With `L = 1 m` and `g = 9.81 m/s^2` the numbers below become
`omega0 = 5.33 rad/s`, `T0 = 1.18 s` and `nu = 1.57e-3 m^2/s`.

| Quantity | Value |
|---|---|
| Tank (background mesh) | `[0, 1] x [0, 0.625]` |
| Mean depth `H0` | `0.5 + 1/128 = 0.5078125` |
| Mode | `n = 1`, `k = pi/L`, `kH0 = 1.595` |
| Amplitude `A` | `0.005` (`kA = 0.0157`, `A/H0 = 0.0098`) |
| Kinematic viscosity `nu` | `5e-4` (`nu k^2/omega0 = 2.9e-3`; free-surface boundary layer `sqrt(2 nu/omega0) = 0.024`) |
| Initial state | `phi = y - H0 - A cos(k x)`, `u = 0`, and the linear pressure `p = rho g (H0 - y) + rho g A cosh(k y)/cosh(k H0) cos(k x)` (zero on the surface to second order in `A`), sampled on every vertex |
| Walls | free slip on left, right and bottom walls; dry top wall without condition |
| Exterior | void, `p_ext = 0`, surface tension 0 |

**Why `H0` is `0.5 + 1/128`.** The surface stays inside `H0 +- A =
(0.5028, 0.5128)`, which contains no vertex row at any level (rows are at
multiples of `1/64`). The interface therefore never passes through a vertex,
and the benchmark measures the wave rather than vertex-crossing events; those
are exercised by the dam-break cases. The gap between the band and the
nearest vertex row is 0.045, 0.09 and 0.18 cells at `L/h = 16, 32, 64`. The
interface sits at 0.125, 0.25 and 0.5 of its cell row, and at every level all
dry vertices of the cut cells are aggregated (their velocity and pressure
extrapolate the cell below).

**Why free-slip walls.** Linear standing-wave theory assumes impermeable slip
walls. The parser supports them on axis-aligned walls: a `Dir` condition with
`Value 0` and `Effective_direction` selecting the normal component (`1 0` on
the side walls, `0 1` on the bottom) constrains only the normal velocity
strongly, and the tangential traction is zero naturally. The corners get both
components. The potential-flow mode satisfies these conditions exactly: the
horizontal velocity `~ sin(k x)` vanishes on the side walls, the vertical
velocity `~ sinh(k y)` on the bottom, and the shear stress vanishes on all
three walls. There are no wall boundary layers. The older case prescribed the
analytic potential-flow velocity on the walls as time-dependent Dirichlet
data; that forces the tank at the analytic frequency and so cannot measure it.

**Why this viscosity.** The older case used `nu = 1e-8`, for which the
physical damping over four periods is 3e-6 of the amplitude and the measured
damping would be purely numerical. With `nu = 5e-4` the amplitude decays by
13% over the run, which is measurable, while `nu k^2/omega0 = 2.9e-3` keeps the
wave in the weakly viscous regime where the linear theory below applies.
Viscosity is a physical input of the case (P1).

## Reference solution (linear theory)

**Inviscid frequency** (Faltinsen and Timokha 2009):

```text
omega0^2 = g k tanh(k H0),   omega0 = 1.7009685,   T0 = 3.693887
```

**Viscous standing wave with free-slip walls.** Write `u = grad Phi +
curl(Psi e_z)` with `Phi = A1 cosh(k y) cos(k x) e^{st}` and
`Psi = B1 sinh(m y) sin(k x) e^{st}`, `m^2 = k^2 + s/nu`. Both satisfy the
free-slip conditions on `x = 0, L` and `y = 0` exactly. The linearized
kinematic, tangential-stress and normal-stress conditions at `y = H0` give the
dispersion relation

```text
D(s) = g k (S - beta) + s^2 C + 2 nu k^2 s C - 2 nu k m beta s coth(m H0) = 0,
S = sinh(k H0),  C = cosh(k H0),  beta = 2 k^2 S / (m^2 + k^2).
```

For `H0 -> infinity` it reduces to Lamb's deep-water relation
`(s + 2 nu k^2)^2 + g k = 4 nu^2 k^3 m` (Lamb 1932, art. 349); the tests check
this limit. `generate_case.py` finds the root `s = -gamma + i omega_d` by
complex Newton iteration from `i omega0 - 2 nu k^2`:

| Quantity | Value | Use |
|---|---|---|
| `omega_ref = Im(s)` | 1.7006229 (`omega0 (1 - 2.03e-4)`) | frequency reference |
| `gamma_ref = -Re(s)` | 9.5229e-3 | damping reference |
| Lamb estimate `2 nu k^2` | 9.8696e-3 (`gamma_ref / 2 nu k^2 = 0.9649`) | reported beside it |

The Lamb estimate is the dissipation of the irrotational mode. It is exact to
leading order in `nu` here because free-slip walls add no boundary layer; its
validity condition is `nu k^2/omega0 << 1`. The first correction comes from
the vorticity layer at the free surface and is
`-(1/sqrt 2)(nu k^2/omega0)^(1/2) = -3.8%` of `2 nu k^2` at this viscosity
(the exact relation gives -3.5%). The frequency shift from viscosity is
`-2.0e-4` relative.

**Limits of the reference.**
- Amplitude: the nonlinear frequency correction is second order in
  `kA = 0.0157`. The Tadjbakhsh and Keller (1960) coefficient at `kH0 = 1.60`
  gives about `-2.5e-5` relative; even an O(1) coefficient would stay below
  `3e-4`.
- Initial condition: the run starts from the inviscid mode shape at rest, not
  the viscous eigenmode. The difference is a weak vortical transient of
  relative size `O((nu k^2/omega0)^(1/2))` near the surface, which barely
  moves the surface.

## Discretization and fixed inputs

Every value is fixed for all levels (principle P1). The numerical constants
are the production values used in `static_drop_2d` (tracker section 6.1);
none was chosen for this case.

| Input | Value | Source |
|---|---|---|
| Mesh | affine `Triangle3`, diagonals alternating with cell parity (mirror-symmetric about `x = L/2`), `L/h = 16, 32, 64` (187, 693, 2,665 vertices) | geometry choice |
| Free surface | `UnfittedLevelSet`, `Active_domain=LevelSetNegative`, `Active_domain_method=CutVolume`, `Generated_interface_geometry=LinearCorner`, `Surface_tension=0` | production unfitted path |
| Cut stabilization | pressure-gradient facet penalty 1.0, `Use_cut_metadata_scale=false`, `Small_cut_aggregation=true`, no velocity extension | production defaults |
| Level-set transport | P1, SUPG with tau scale 0.5 and transient scale 2.0; no reinitialization, no volume correction, no interface kinematic term; advected by the harmonic PDE extension of the fluid velocity with monolithic coupling (`Advection_velocity_extension_method=pde_harmonic`, `Advection_velocity_extension_coupling=monolithic`) | decision D9; see "Level-set advection velocity" |
| Level-set kinematic reconciliation | on (`Enable_kinematic_reconciliation=true`; `--kinematic-reconciliation off` reproduces the earlier decks): after every accepted step the transported `phi` is corrected locally so that the step's change of the sharp P1 area equals the interface flux of the transport velocity (`FE/LevelSet/LevelSetKinematicReconciliation.h`) | parameter-free and local, not a volume target or global shift; the area drift stays a measured quantity (the flux of a discretely divergence-free velocity) |
| Time integration | generalized-alpha, `rho_inf = 0.5`; spatial study: 128 steps per inviscid period at every level; time-step study: 64, 128 and 256 steps per period at `L/h = 32`; 4 periods; 32 outputs per period | decision D10; see below |
| Nonlinear solve | relative tolerance 1e-4 per equation, at most 8 Newton iterations; level-set absolute floor 1e-10 | production decks |
| Linear solve | Eigen direct (sparse LU), tolerances 1e-8 / 1e-10 | small 2D meshes; the older case used the same solver |

**Level-set advection velocity (D9).** The fluid velocity is defined only on
the wet vertices and on the vertices of the cut cells; the vertices beyond are
inactive and their velocity is zero.  Advecting `phi` with the fluid velocity
itself freezes `phi` one row above the cut cells, and the level set on the dry
vertices of the cut row then lags the flow increasingly with refinement (the
first protocol failed at `L/h = 64` for this reason).  The advection velocity
`w` is therefore the PDE extension of the fluid velocity `u`
(`Application/Core/LevelSetPdeVelocityExtension.h`):

- `w = u` on every wet vertex and every vertex of the retained interface
  cells, so that `w_h = u_h` on the interface;
- on the remaining (dry) vertices `w` solves, with that Dirichlet data,
  `(grad w, grad v) = 0` over the dry cells (harmonic, the protocol value) or
  `((n.grad) w, (n.grad) v) = 0` with `n = grad phi_h / |grad phi_h|` (the
  least-squares normal extension, `pde_normal`: `w` constant along the
  level-set normals).  Both are linear, symmetric and parameter-free (P1);
- velocity components that a strong homogeneous wall condition constrains
  (here the wall-normal component of the free-slip walls) are zero on dry wall
  vertices; the others have the natural condition.  The domain is every dry
  vertex, so there is no band edge.

With the monolithic coupling the discrete problem is installed, at every
geometry refresh, as frozen rows of an auxiliary unknown (each dry value is
the operator-weighted average of its neighbors), so the Newton tangent
contains the dependence of the transport on `u`.  The extension never enters
the momentum rows.  The prescribed coupling instead solves the same system
once per geometry refresh and writes the result into a prescribed field; the
transport then sees it lagged by one outer pass, which costs extra outer
passes (see the diagnostics below).  Neither coupling writes per-step files.

**Time step (D10).** Space and time are studied separately, because the
first protocol refined the time step with the mesh and opposite-sign time and
space errors cancelled.  All levels of the spatial study use the same step,
128 steps per inviscid period.  The time-step study runs `L/h = 32` at 64, 128
and 256 steps per period.  Its time error at 128 steps per period,
`e_time = (4/3)(e(128) - e(256))` (second order; the observed temporal order
is reported), is subtracted from every spatial frequency error.  For the
oscillator `y' = i omega y`, the first-order generalized-alpha method with
`rho_inf = 0.5` has a relative phase error of `-1.1e-3`, `-2.7e-4` and
`-6.7e-5` at 64, 128 and 256 steps per period and a numerical damping rate of
`9e-6`, `1e-6` and `1.4e-7` times `omega`, against the physical
`gamma/omega = 5.6e-3`; the damping therefore needs no time correction.  The
level-set Courant number is below 0.02 at every level.

## Metrics (`verify.py`)

All quantities come from the solver's VTU/PVTU point data (`phi`,
`Velocity`) at `t = 0` (the mesh file) and at every output.

| Metric | Definition |
|---|---|
| probe elevation `eta_p(t)` | height of the lowest upward zero crossing of `phi_h` on the left wall `x = 0` (a mesh line, so `phi_h` is linear between its vertices), minus `H0` |
| fit | least-squares fit of `eta_p(t) = c + exp(-gamma t)(a cos(omega t) + b sin(omega t))` over all outputs: a scan over `omega` picks the start, then Levenberg-Marquardt refines all five parameters |
| `frequency_relative_error` | `abs(omega - omega_ref)/omega_ref` (signed value reported) |
| `frequency_spatial_error` | `abs(omega/omega_ref - 1 - e_time)` on the spatial study, `e_time` from the time-step study (D10) |
| `damping_rate_relative_error` | `abs(gamma - gamma_ref)/gamma_ref` |
| `liquid_area_relative_drift_max` | max over outputs of `abs(A(t) - A(0))/A(0)`, `A` the exact area of `{phi_h < 0}` (cut triangles clipped by the linear `phi_h`) |
| Reported only | signed frequency error; error against `omega0`; `gamma/(2 nu k^2)`; fitted amplitude over `A`; fit residual; the same fit applied to the modal amplitude `a1(t)` from a least-squares fit `y = sum_{n<4} a_n cos(n k x)` to all interface points; maximum liquid speed; the histories; from `solver_run.log(.gz)` if present: outer passes and Newton iterations per step, the largest final residual, and the wall time |

Observed orders are least-squares slopes of `log(error)` against `log(L/h)`;
pairwise orders are printed too. `verify.py` exits with status 2 on missing
or inconsistent data (no `case.json`, no output, missing arrays, a mesh that
is not pure `Triangle3`, non-finite values, a run that stopped before its end
time, no crossing on the probe line) and on `--max-steps` smoke runs unless
`--allow-truncated` is given.

## Tolerances and their sources

| Criterion | Limit | Where | Source |
|---|---|---|---|
| `frequency` | spatial error (time error removed) at most 0.01; strictly decreasing; observed order at least 1 | limit at `L/h = 64`; monotonicity and order over 16/32/64 | tracker M1 working criterion ("frequency error <= 1% at the finest mesh with observed convergence"), with the time error removed (D10). "Observed convergence" is read as a strictly decreasing error with an observed order of at least 1, as for M2 in `static_drop_2d`; the expected order is 2 |
| `damping` | at most 0.05 | `L/h = 64` of the spatial study | decision D12 |
| `volume_drift` | maximum deviation over the run at most 1e-4 | every run of the spatial and time-step studies | tracker M1 working criterion; decision D11 |

Runs with another level-set velocity or mean depth are reported beside the
protocol runs as comparisons and are never gated.

## How to run

Use the Python stack from the benchmark README (numpy; pyvista for
`verify.py`). Put generated cases and output under `$SCRATCH`, and start the
solver through `mpiexec` in a batch job (benchmark README):

```bash
B=tests/cases/fluid/free_surface_benchmarks/linear_sloshing_2d
OUT=$SCRATCH/free-surface-benchmarks/linear_sloshing_2d/$(git rev-parse --short HEAD)
for L in 16 32 64; do python3 $B/generate_case.py --level $L --output-dir $OUT/L${L}_T128; done
for T in 64 256; do
  python3 $B/generate_case.py --level 32 --steps-per-period $T --output-dir $OUT/L32_T$T
done
# in each case directory, inside a Slurm job:
#   mpiexec -n 1 --bind-to none /path/to/svmultiphysics solver.xml 2>&1 | gzip -1 > solver_run.log.gz
python3 $B/verify.py $OUT/L*_T* --json $OUT/verify.json
```

Other options of `generate_case.py`; `verify.py` reports such runs beside the
protocol runs but never gates them:

- `--level-set-velocity`: `pde_harmonic_monolithic` (protocol),
  `pde_normal_monolithic`, `pde_harmonic_prescribed`, `pde_normal_prescribed`,
  `coupled_field` (the fluid velocity itself, the first protocol) or
  `wet_extension` (the algebraic wall-compatible extension of the SPHERIC
  Test 05 decks; it writes one JSON map per step under
  `velocity_extension_maps/`, 1.8 GB for an `L/h = 64` run).
- `--mean-depth H0`: moves the rest level, and with it the interface's
  position in its cell row; the reference is recomputed for `H0`.

## Results

### Revised protocol, D9 to D12 (2026-09-30)

**Source `b4b376a0` on branch `dev/pde-velocity-extension`, solver built from
the same commit, Slurm job `46089180`. Protocol transport
`pde_harmonic_monolithic`. Result: FAIL on the frequency convergence
criterion only; the frequency limit, the damping and the volume criteria
pass.**

Spatial study (all levels at 128 steps per period); "spatial" is the probe
frequency error with the time error of that step removed.  The time-step
study at `L/h = 32` (64, 128, 256 steps per period; raw errors -0.038%,
+0.039%, +0.056%) has observed temporal order 2.11 and gives
`e_time(128) = -0.0235%`, the same for every transport to 1e-6.

| Transport | spatial error at L/h = 16 / 32 / 64 | observed order | damping error at 64 (probe) | max `dA/A` over all runs | s/step at 64 | outer passes/step |
|---|---|---:|---:|---:|---:|---:|
| `pde_harmonic_monolithic` (protocol) | +0.019% / +0.062% / +0.074% | -0.96 | 3.3% | 3.2e-5 | 3.83 | 3.08 |
| `pde_normal_monolithic` | +0.016% / +0.054% / +0.059% | -0.93 | 3.1% | 2.6e-5 | 3.84 | 3.08 |
| `pde_harmonic_prescribed` | +0.019% / +0.062% / +0.074% | -0.96 | 3.3% | 3.2e-5 | 3.67 | 4.09 |
| `wet_extension` | +0.017% / +0.053% / +0.120% | -1.41 | 3.4% | 4.0e-5 | 3.88 | 3.08 |
| `coupled_field` (first protocol) | +0.081% / +0.383% / +1.455% | -2.08 | 7.6% | 6.5e-4 | 3.29 | 3.11 |

| Criterion (protocol transport) | Result |
|---|---|
| `frequency` | **FAIL**: 0.074% <= 1% at `L/h = 64`, but not decreasing (0.019%, 0.062%, 0.074%); observed order -0.96 |
| `damping` | PASS: 3.3% <= 5% |
| `volume_drift` | PASS: at most 3.2e-5 on all five runs |

Observations:

- The PDE extension removes the growing error of the coupled-field
  transport: at `L/h = 64` the frequency error falls from 1.46% to 0.074%,
  the area oscillation from 6.5e-4 to 3.2e-5 and the damping error from 7.6%
  to 3.3%.  The symmetric mode `cos(2 k x)` stays at 2.0% of `A` (9.5% with
  the coupled field), the level expected from second-order theory for a
  release from rest.
- The monolithic and prescribed couplings reach the same solution to four
  digits (the same fixed point); the prescribed coupling needs one more outer
  pass per step.  The harmonic and least-squares normal operators differ by
  at most 0.015% in frequency and 0.2% in damping.
- With every extension the spatial error grows from about 0.02% to
  0.05–0.07% and then flattens.  This variation is of the order of the
  measurement and modelling uncertainty: the probe and modal-amplitude fits
  differ by up to 0.018% at `L/h = 64` (modal spatial errors +0.019%,
  +0.050%, +0.056%), and the viscous frequency shift in the reference is
  itself 0.020%.  An observed order cannot be demonstrated at this level;
  the criterion needs a decision (for example an error floor below which it
  is not applied).
- The probe damping error reflects the wall probe, where a slow mean-level
  change (0.5% of `A`) and the second harmonic enter the fit.  The damping of
  the modal amplitude converges to the linear viscous rate
  (`gamma/gamma_ref` = 1.028, 1.005, 0.999 at `L/h` = 16, 32, 64).
- Two-rank runs at `L/h = 16` (24 steps; FSILS GMRES and three ghost layers,
  which the distributed small-cut aggregation needs) reproduce the serial
  probe elevation and area to 1e-16 for both couplings and both operators
  (jobs `46089180` and `46100049`).
- Per-step cost with the PDE extension is 16% above the coupled field at
  `L/h = 64` (3.83 against 3.29 s/step), about the same as the algebraic wet
  extension, which additionally writes a 3.5 MB map file per step.

Raw output: `/scratch/users/zsexton/free-surface-benchmarks/pde-extension/campaign-b4b376a0/`
(`sloshing_verify.txt`, `sloshing_verify.json`).

### First protocol (2026-09-30, superseded)

The first protocol advected `phi` with the fluid velocity, refined the time
step with the mesh (`dt = T0/(2 L/h)`), reported the damping without a limit,
and gated the volume drift at every level.

**Source `7aac1e29`, solver built at `fef0d02f`, Slurm job `46023412`.
Result: FAIL** (frequency and volume drift at `L/h = 64`).

| L/h | steps/T | `omega` | frequency error (signed) | `gamma/gamma_ref` | `gamma/(2 nu k^2)` | fitted `A/A0` | fit residual `/A` | `max dA/A` | final `dA/A` | s/step | wall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 32 | 1.695201 | -3.19e-3 | 1.017 | 0.982 | 1.006 | 3.8e-3 | 5.5e-5 | 1.9e-7 | 0.81 | 104 s |
| 32 | 64 | 1.705433 | +2.83e-3 | 0.979 | 0.944 | 1.008 | 1.0e-2 | 5.4e-5 | 2.2e-7 | 2.1 | 547 s |
| 64 | 128 | 1.724971 | +1.43e-2 | 0.924 | 0.891 | 1.011 | 3.8e-2 | 6.5e-4 | 2.0e-5 | 9.8 | 5,036 s |

| Criterion | Result |
|---|---|
| `frequency` | **FAIL**: 1.43% > 1% at `L/h = 64`; not decreasing (0.32%, 0.28%, 1.43%); observed order -1.08 (pairwise 0.17, -2.34) |
| `damping` (reported) | 1.7%, 2.1%, 7.6% from `gamma_ref` |
| `volume_drift` | **FAIL**: 5.5e-5 and 5.4e-5 pass at 16 and 32; 6.5e-4 > 1e-4 at 64 |

Observations:

- Every step converged (4 to 5 outer passes, 3 to 5 Newton iterations per
  step, final residuals below 1e-10). No run failed.
- At 16 and 32 the error is within 0.32% but changes sign; the generalized-
  alpha phase error alone is -0.42% and -0.11% at these time steps.
- At 64 the wave runs 1.4% fast. The modal amplitude `a1(t)` gives the same
  (+1.32%), so it is not a probe artefact. The symmetric mode `cos(2 k x)`,
  which linear theory does not excite and which is 1% of `A` at `L/h = 16`
  (the second-order nonlinear level), oscillates at its own frequency with
  10% of `A`; it dominates the probe fit residual.
- The liquid area oscillates with the wave and returns to its initial value
  each period (final drift 2e-7 at 16 and 32). The oscillation amplitude
  (5.5e-5, 5.4e-5, 6.5e-4) is what fails at 64.
- The level set on the dry vertices of the cut row lags the flow increasingly
  with refinement: at `t = T/2` its change is 8%, 11% and 17% below the
  extrapolated potential flow at 16, 32 and 64. The next row of vertices is
  inactive (velocity pinned to zero), so its `phi` stays at the initial
  shape. The diagnostic runs below separate the time, mesh and cut-position
  effects and compare the other level-set advection velocity.

Raw output: `/scratch/users/zsexton/free-surface-benchmarks/linear_sloshing_2d/m1-7aac1e29/`
(`verify.txt`, `verify.json`).

#### Diagnostic runs of the first protocol (not gated)

Three diagnostic series were run after the first protocol study, with the options
`--steps-per-period`, `--mean-depth` and `--level-set-velocity` of
`generate_case.py` (source `f57e1e02` to `4dc2cfb9`, Slurm jobs `46033465` and
`46041062`). `verify.py` reports such runs but never gates them. "Spatial"
below is the frequency error with the time error removed, using the time
error measured at the same mesh and time step.

**Time step at fixed mesh** (coupled-field transport, protocol `H0`):

| L/h | frequency error at steps/T = 32 / 64 / 128 / 256 | time error at the protocol step | spatial (dt -> 0) |
|---:|---|---:|---:|
| 16 | -0.319% / -0.013% / +0.058% / +0.074% | -0.40% | +0.080% |
| 32 | - / +0.283% / +0.360% / +0.377% | -0.10% | +0.383% |
| 64 | - / +1.334% / +1.432% / - | -0.03% | +1.465% |

Successive differences shrink by 4.3 to 4.5 per halving of `dt`, so the time
error is second order and matches the generalized-alpha oscillator estimate
(-0.42%, -0.11%, -0.027%). The spatial error grows about fourfold per mesh
refinement.

**Position of the interface in its cell row** (coupled-field transport,
protocol time step). `s` is the rest level's height above the vertex row
below it, in cells:

| L/h | `s` | `H0` | frequency error | spatial | `max dA/A` |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.125 | 0.5078125 | -0.319% | +0.080% | 5.5e-5 |
| 16 | 0.25 | 0.515625 | -0.321% | +0.078% | 2.7e-5 |
| 16 | 0.5 | 0.53125 | -0.340% | +0.059% | 1.7e-4 |
| 32 | 0.25 | 0.5078125 | +0.283% | +0.383% | 5.4e-5 |
| 32 | 0.5 | 0.515625 | +0.317% | +0.417% | 3.3e-4 |
| 64 | 0.5 | 0.5078125 | +1.432% | +1.465% | 6.5e-4 |

The frequency error depends on the mesh, not on `s`. The amplitude of the
area oscillation depends on `s` (largest for a mid-cell interface), which is
why it fails at `L/h = 64`.

**Level-set advection velocity.** The same study with the wall-compatible
wet-velocity extension of the SPHERIC Test 05 decks
(`--level-set-velocity wet_extension`, protocol `H0` and time steps):

| L/h | frequency error | spatial | `gamma/gamma_ref` | fit residual `/A` | `max dA/A` | largest `cos(2kx)` amplitude `/A` (coupled-field) | s/step |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | -0.385% | +0.014% | 0.995 | 5.3e-3 | 8.8e-6 | 1.4% (1.1%) | 0.98 |
| 32 | -0.046% | +0.054% | 0.978 | 5.6e-3 | 3.5e-6 | 1.6% (2.3%) | 2.5 |
| 64 | +0.097% | +0.130% | 0.966 | 7.0e-3 | 4.0e-5 | 2.1% (9.5%) | 11.3 |

With the wet extension the spatial error is ten times smaller at `L/h = 64`,
the area oscillation stays below 4e-5, and the symmetric mode stays at the
level expected from second-order theory for a release from rest (about 1.5%
of `A`). Judged by the protocol criteria this series would pass the 1% limit
(0.097%) and the volume limit, and fail monotonicity and the order (0.385%,
0.046%, 0.097%; least-squares order 0.99): the time error is negative and
dominates at `L/h = 16`, while the spatial error is positive and still grows
with refinement.

Interpretation: with the coupled-field transport and no velocity extension,
the velocity on the inactive vertices one row above the cut cells is pinned to
zero and their `phi` never moves. The level set on the dry vertices of the cut
row is coupled to those frozen values through the consistent mass and the
SUPG terms, and lags the flow increasingly as `h` shrinks (see above). The
wet extension moves `phi` on a band of dry vertices consistently with the
flow and removes most of the error.

Raw output: `/scratch/users/zsexton/free-surface-benchmarks/linear_sloshing_2d/`
(`m1-dt-study/`, `m1-cut-position/`, `m1-diagnostics/`; all runs verified
together in `m1-all-verify.txt` and `m1-all-verify.json`).

# capillary_wave_2d

A small-amplitude standing capillary wave on the free surface of a deep
one-phase liquid, released from rest. It measures the frequency and the
viscous damping rate of the unfitted level-set free surface against
Prosperetti's initial-value solution (milestone M3 in
`Documentation/free_surface_program_tracker.md`). The same case is run for
the three capillary routes compared in decision D2: `SurfaceStress`, KAG
with a lumped trace mass, and KAG with the consistent trace mass (the two KAG
forms are generated but have not been run).

The protocol follows decisions D9 to D11 of 2026-09-30 and D13 of
2026-10-05: `phi` is advected with the harmonic PDE velocity extension
(`--transport`, below); the lagged normal-increment capillary term is on and
every level uses 50 steps per inviscid period, checked at 100; and the area
criterion gates the maximum deviation over the whole run.

Files:

| File | Role |
|---|---|
| `generate_case.py` | writes `solver.xml`, the mesh with the initial fields, the wall faces and `case.json` for one level |
| `prosperetti_reference.py` | the reference solution, its normal-mode and Laplace-transform checks (numpy only) |
| `verify.py` | reads the solver output of one or more levels, computes the metrics, applies `tolerances.json` |
| `tolerances.json` | acceptance criteria and their sources, fixed on 2026-09-30 before the first protocol run |
| `tests/test_free_surface_benchmark_capillary_wave_2d.py` | checks of the reference and of the scripts on synthetic data |

## Physical setup

Nondimensional units: density `rho = 1`, surface tension `gamma = 1`,
wavelength `lambda = 1`, so `k = 2 pi`.

- **Geometry.** Half a wavelength, `[0, lambda/2] x [0, 1.25 lambda]`. The
  liquid lies below the surface `y = y0 + a0 cos(kx)`, with mean level
  `y0 = lambda (1 + sqrt(2)/100)` and amplitude `a0 = 0.01 lambda`
  (`a0 k = 0.063`). Zero gravity. The exterior is void with `p_ext = 0`.
- **Walls.** The side walls `x = 0` (crest) and `x = lambda/2` (trough) are
  the mirror planes of the cosine mode. They are impermeable free-slip
  walls: strong `u_x = 0` (`Effective_direction 1 0`), tangential velocity
  and contact points free. No contact-line model is declared. With a strong
  normal constraint, the natural boundary term of `SurfaceStress` at the
  contact point vanishes only for a 90° angle, which is exactly the
  symmetry condition `d(eta)/dx = 0`. The bottom is free-slip too
  (`u_y = 0`), so there is no bottom boundary layer. The top boundary is
  dry (no-slip, never wetted: 3.6 cells above the crest at `lambda/h = 16`).
- **Viscosity** from the Laplace number `La = rho gamma lambda / mu^2 = 3000`,
  the value of the standard capillary-wave test of Popinet (2009, J. Comput.
  Phys. 228, 5838, section 5.3; also Popinet and Zaleski 1999). Here the
  upper fluid is absent (the one-fluid limit of Prosperetti 1981).

  | Quantity | Value |
  |---|---:|
  | `mu = nu` (= Ohnesorge number `mu/sqrt(rho gamma lambda)`) | 0.018257 |
  | `epsilon = nu k^2 / omega0` | 0.04576 |
  | inviscid `omega0 = sqrt(gamma k^3 / rho)`, period `2 pi/omega0` | 15.7496, 0.39894 |
  | normal-mode frequency `omega` (Lamb's dispersion relation) | 15.5366 (`omega0` - 1.35%) |
  | normal-mode damping `beta` | 1.2185 (`2 nu k^2` - 15.5%) |
  | weak-viscosity damping `2 nu k^2` | 1.4415 |
  | vorticity layer `delta = sqrt(2 nu / omega0)` | 0.048 `lambda` (0.77, 1.54, 3.1 `h`) |
  | amplitude after 4 periods, `a(T)/a0` | 0.133 |

  The damping is thus measurable within the run, and the regime is far
  from the weak-viscosity limit: `2 nu k^2` overestimates the damping by
  18% and `omega0` the frequency by 1.4%, which is of the order of the
  tolerances. The comparison is therefore made with the full viscous
  solution, never with those limits.
- **Depth.** `k y0 = 2 pi x 1.014`: `tanh(k y0) = 1 - 7e-6`, so the
  finite-depth factor changes `omega0` by 3.5e-6, and the viscous
  correction of a free-slip bottom is of the same order `exp(-2 k y0)`.
  The infinite-depth reference therefore applies.
- **Nonlinearity.** `(a0 k)^2 = 0.004`; the amplitude-dependent frequency
  shift is of this order times a coefficient below one, well below the
  frequency tolerance.
- **Initial state** (the sampled analytic shape, D3):
  `phi = y - y0 - a0 cos(kx)` at the P1 vertices (`|grad phi| <= 1.002`),
  `u = 0`, and the linear pressure of the released state,
  `p = gamma a0 k^2 cos(kx) cosh(ky) / cosh(k y0)`: harmonic, zero normal
  derivative at the bottom, equal to `gamma kappa` at the mean level, and
  continued smoothly through the dry vertices that cut cells need.

## Reference solution (`prosperetti_reference.py`)

Prosperetti (Phys. Fluids 19, 195 (1976); Phys. Fluids 24, 1217 (1981),
one-fluid limit) gives the amplitude of the `cos(kx)` mode for a liquid of
infinite depth released from rest. With `sigma = nu k^2`,

```text
a(t)/a0 = 4 sigma^2 / (8 sigma^2 + omega0^2) erfc(sqrt(sigma t))
        + sum_i z_i / Z_i  omega0^2 / (z_i^2 - sigma)  exp((z_i^2 - sigma) t) erfc(z_i sqrt(t)),
z^4 + 2 sigma z^2 + 4 sigma^(3/2) z + sigma^2 + omega0^2 = 0,   Z_i = prod_{j != i} (z_j - z_i).
```

This is equation (12)-(13) of Denner et al. (Phys. Rev. E 94, 023110
(2016)) with the density parameter `beta = 0`; the two-fluid version is
already in `tests/cases/fluid/run_free_surface_wp10_capillary_wave_reference.py`
(which needs scipy). Here each `exp(z^2 t) erfc(z sqrt(t))` is evaluated as
the scaled function `erfcx(z sqrt(t)) = w(i z sqrt(t))`, with the Faddeeva
function `w` from Weideman's rational expansion (SIAM J. Numer. Anal. 31,
1497 (1994), 40 terms), so that only numpy is needed and no term overflows.

Checks, all in the test file:

| Check | Result |
|---|---|
| `erfcx` against `math.erfc` on the real axis, and against the Taylor series of `erf` for complex arguments | relative error below 1e-13 and 1e-11 |
| initial condition | `a(0) = a0`, `a'(0) = 0` |
| inviscid limit (`nu = 0`) | `a(t) = a0 cos(omega0 t)` to 1e-10, i.e. the dispersion relation `omega0^2 = gamma k^3 / rho` |
| Laplace transform at `La = 3000` | the numerical transform of `a(t)` equals `a0 D / (s (D + omega0^2))`, `D = (s + 2 sigma)^2 - 4 sigma^2 sqrt(1 + s/sigma)`, to 1e-10 at `s/omega0 = 0.5, 1, 3`. `a_hat(s)` is derived directly from the linearized Navier-Stokes equations with a stress-free surface (docstring of `laplace_transform`), so this checks the whole closed form, roots and `erfcx` included. |
| normal mode | the root of Lamb's relation `D(s) + omega0^2 = 0` (Newton) equals `z^2 - sigma` for the quartic root with `Re z < 0`, and follows `omega/omega0 - 1 = -sqrt(2) eps^(3/2)` and `beta/(2 nu k^2) - 1 = -sqrt(eps/2)` within 1% of the coefficient for `eps = 1e-2 ... 1e-4` (weak-damping limit) |
| damping limit of the history | a damped-cosine fit of `a(t)` over 4 periods gives `2 nu k^2` within 1% and `omega0` within 1e-5 at `eps = 1e-4`; at `La = 3000` it is within 0.1% (frequency) and 1% (damping) of the normal mode |

## Discretization and fixed inputs

Every value below is fixed for all levels and all capillary forms
(principle P1). The solver inputs are those of `static_drop_2d` (its README
gives the source of each); only the geometry, the walls, the viscosity and
the level-set advection velocity differ.

| Input | Value | Source |
|---|---|---|
| Mesh | affine `Triangle3`, `h = lambda/level`, diagonals alternating with cell parity; 8 x 20, 16 x 40, 32 x 80 cells (189, 697, 2673 vertices) | geometry choice. With an even number of columns the mesh is mirror-symmetric about `x = lambda/4`, the node line of the mode, so crest and trough see the same mesh. |
| Mean-level offset | `sqrt(2)/100 lambda` above `y = lambda` | irrational, so the mean level lies on no grid line of the nested dyadic meshes and the sampled surface cannot pass exactly through a vertex. Among the simple irrational offsets tried (`pi/100`, `e/100`, `pi/1000`, `sqrt(2)/100`, ...) it is one of the two that keep `min abs(phi)/h` at or above 0.03 on all three levels: 0.066, 0.13, 0.030 at `lambda/h` = 16, 32, 64 (printed by `generate_case.py`). The static drop, by comparison, starts from 1.8e-3 to 6.8e-3. It has no physical effect (depth paragraph above). |
| Walls | `Dir`, value 0; `Effective_direction 1 0` (sides), `0 1` (bottom); full no-slip on the dry top | free-slip mirror planes and bottom (see Physical setup) |
| Free surface, cut stabilization, capillary forms, level-set discretization, time integration, nonlinear and linear solves | identical to `static_drop_2d` (`SurfaceStress` / KAG keys, `RefreshedFrozenQuadrature`, `Interface_quadrature_order=2`, aggregation, pressure-gradient facet penalty 1.0, SUPG 0.5/2.0, no reinitialization or volume correction, generalized-alpha `rho_inf = 0.5`, FSILS GMRES) | production values, see `static_drop_2d/README.md` |
| Level-set advection velocity | `--transport`, see below | decision D9 |
| Level-set kinematic reconciliation | on (`Enable_kinematic_reconciliation=true`; `--kinematic-reconciliation off` reproduces the earlier decks): after every accepted step the transported `phi` is corrected locally so that the step's change of the sharp P1 area equals the interface flux of the transport velocity (`FE/LevelSet/LevelSetKinematicReconciliation.h`) | parameter-free and local, not a volume target or global shift; the area drift stays a measured quantity (the flux of a discretely divergence-free velocity) |

**Level-set transport (decision D9).** `generate_case.py --transport`
selects the velocity that advects `phi`:

| `--transport` | Solver input | Status |
|---|---|---|
| `pde_extension` (default) | `Velocity_source=prescribed_data`, `Advection_velocity_extension_method=pde_harmonic`, `Advection_velocity_extension_coupling=monolithic` | the D9 protocol transport: the parameter-free harmonic PDE extension of the fluid velocity into the dry region, with monolithic coupling, as in `linear_sloshing_2d`, `static_drop_2d` and `sessile_drop_2d` (validation in the `linear_sloshing_2d` README). It writes no per-step files. |
| `wet_extension` | `Velocity_source=prescribed_data`, `Use_wet_extension_advection_velocity=true`, `Advection_velocity_extension_method=wall_compatible_normal` | the algebraic wall-compatible wet extension, the D9 comparison baseline. It writes one JSON map per accepted step (about 1 MB per step in the static-drop smoke run), so a 2900-step run produces gigabytes; keep such runs on `$SCRATCH`. |
| `coupled` | `Velocity_source=coupled_field` | the fluid velocity itself, used by the smoke run below. Dry vertices then carry zero velocity; D9 retires this transport after it failed `linear_sloshing_2d` at `L/h = 64`. |

The side walls hold `u_x = 0` strongly at every wall vertex, so none of the
three moves `phi` through a wall. Criteria are applied separately to each
(capillary form, transport) study.

**Time step (decision D13, 2026-10-05).** The lagged normal-increment
capillary term (`Surface_tension_semi_implicit = NormalIncrement` in the
free-surface block, `Physics/Docs/NavierStokesFreeSurface.md`) removes the
capillary limit and is on by default (`--surface-tension-semi-implicit`).
Every level uses 50 steps per inviscid period `2 pi/omega0`: `dt = 7.98e-3`,
200 steps, outputs every 2 steps (`--dt-rule steps-per-period`). The step is
shared by all levels, as D10 requires, and is 1.8, 5.1 and 14.5 times the
one-sided capillary limit at `lambda/h = 16`, 32 and 64. The time error at 50
steps per period is about 0.16% in frequency and 0.5% in damping at
`lambda/h = 64`, the generalized-alpha phase error `(omega dt)^2/12`
(`Documentation/free_surface_semi_implicit_surface_tension_design.md`, §9.5
and §9.9).

**Half-step check.** `--dt-divisor 2` gives 100 steps per period with the same
output times; the protocol runs it at `lambda/h = 64`. Every gate must also
pass for the `dt/2` run, and between `dt` and `dt/2` at the finest common
level the fitted frequency may change by at most 0.2% and the damping by at
most 1% (`time_step_criterion` in `tolerances.json`). Other levels run at two
or more divisors (1, 2, 4) are reported in the time-step study.

**Earlier protocol (D10, until 2026-10-05).** All levels used the one-sided
capillary limit `dt <= sqrt(rho h^3 / (4 pi gamma))` of the finest level,
rounded down to 100 equal output intervals: `dt = 5.50e-4`, 2900 steps
(725 per period), with the term off. Reproduce it with
`--dt-rule capillary-limit --surface-tension-semi-implicit None`.

**Run length.** 4 inviscid periods (`t omega0 = 8 pi = 25`, Popinet's
horizon), with 100 VTU snapshots, 25 per period.

## Metrics (`verify.py`)

All quantities come from the `phi` point data of the solver's VTU/PVTU
output on the output mesh, plus `phi` of `mesh/mesh-complete.mesh.vtu` for
`t = 0`.

| Metric | Definition |
|---|---|
| Liquid area `A(t)` | exact area of `{phi_h < 0}` for the P1 interpolant (each cut triangle clipped by the linear `phi_h`), as in `static_drop_2d` |
| Mode amplitude `a_h(t)` | `cos(kx)` coefficient of the P1 surface, `a_h = (2/W) int_0^W eta_h cos(kx) dx`, `W = lambda/2`. It is computed as `(2/W) int_{phi_h<0} cos(kx) dA` (the bottom and the walls at `x = 0, lambda/2` contribute nothing), exactly for the polygonal P1 region through Green's theorem on each clipped polygon. A uniform level drift and the harmonics `cos(n k x)`, `n != 1`, do not enter; a multivalued surface would still be handled. |
| Fit | `a(t) = exp(-beta t) (c1 cos(omega t) + c2 sin(omega t))` by least squares over all outputs including `t = 0`: a coarse scan (`omega` in `[0.5, 1.5] omega0`, `beta` in `[-0.05, 0.5] omega0`) followed by Levenberg-Marquardt. The same fit is applied to Prosperetti's `a(t)` at the same times. |
| `frequency_relative_error` | `abs(omega_sim - omega_ref) / omega_ref` |
| `damping_rate_relative_error` | `abs(beta_sim - beta_ref) / beta_ref` |
| `liquid_area_relative_drift_max` | `max_t abs(A(t) - A(0)) / A(0)` over the whole run (decision D11): over the outputs and, when `solver_run.log` or `solver_run.log.gz` is in the case directory, over the solver's per-step `Wet volume diagnostic` area of every accepted step (the same exact P1 area; the two agree to round-off on the smoke run). The output-only value and the number of logged steps are reported. |
| Reported only | `a_h(0)/a0 - 1` (sampling error, second order: -1.3%, -0.32%, -0.08%); `amplitude_rms_error`, the RMS over the outputs of `a_h(t)/a_h(0) - a(t)/a0` (Popinet's error measure); its maximum; `mean_level_drift_over_amplitude`; the elevation of the contact points relative to the mean level; the fit residual; the normal mode, `omega0` and `2 nu k^2`; the histories |

Comparing with the fitted reference rather than with the normal mode makes
the comparison insensitive to the non-modal part of Prosperetti's
solution: at `La = 3000` that part shifts the 4-period fit by -0.06%
(frequency) and -0.6% (damping) from the normal mode.

Frequency and damping need at least one inviscid period and 8 samples;
shorter (smoke) runs report them as not evaluable. Observed order is the
least-squares slope of `log(error)` against `log(lambda/h)`; pairwise orders
are printed as well. `verify.py` exits with status 2 on missing or
inconsistent data (no `case.json`, no output, missing `phi`, non-triangle
cells, non-finite values, a run that stopped before its end time) and on a
`--max-steps` smoke run unless `--allow-truncated` is given.

## Tolerances and their sources

Each criterion applies to each (capillary form, transport, `La`, dt divisor)
spatial study; `verify.py` refuses a study whose levels use different time
steps (D10). The divisor-1 study needs every level; the divisor-2 study only
`lambda/h = 64` (criteria at levels it does not contain are reported as not
run). The time-step criterion (D13) is printed as a separate gated line, or
as not evaluated when only one divisor is given.

| Criterion | Limit | Where | Source |
|---|---|---|---|
| `frequency` | at most 0.02, observed order at least 1 | 0.02 at `lambda/h = 32`; order over 16/32/64 | D1 working criterion, tracker M3 |
| `damping` | at most 0.05, observed order at least 1 | 0.05 at `lambda/h = 32` and at the finest level 64; order over 16/32/64 | D1 working criterion, tracker M3; the finest-level check as in D12 |
| `volume_drift` | at most 1e-4 | every level, maximum over the run | D1 working criterion (the volume limit of tracker M1 and M2, applied to M3), gated as in D11 |
| `time_step` | frequency change at most 0.002, damping change at most 0.01 between `dt` and `dt/2` | finest common level (64) | D13, 2026-10-05 |

"Convergence over 16/32/64" is an observed order of at least 1 on the
spatial study at the shared time step (D10, confirmed 2026-09-30): the rate
expected for the capillary force on a piecewise-planar interface (Gross and
Reusken 2011, ch. 7) and the rate required of the static-drop pressure jump.
`A(0)` includes the whole layer, so the area criterion corresponds to a
mean-level drift of `1e-4 y0 = 0.01 a0`; `mean_level_drift_over_amplitude`
reports the drift relative to `a0`.

The July 2026 number (n = 16: frequency error 1.2%, tracker section 8.3)
is not comparable. The FS-16 matrix ran 100 steps of 1 ms of a case with
water density and viscosity, `gamma = 500`, depth `lambda/2` and
`a0 = 0.004 lambda` (`run_fs16_physical_matrix.py`), i.e. 1.1 rad of phase,
under a fifth of a period, and compared it with the inviscid finite-depth
frequency.

## How to run the refinement study

Use the Python stack from the benchmark README (numpy; pyvista for
`verify.py`). Put generated cases and output under `$SCRATCH`, and start the
solver through `mpiexec` from a job submitted with `--export=NONE`:

```bash
B=tests/cases/fluid/free_surface_benchmarks/capillary_wave_2d
OUT=$SCRATCH/free-surface-benchmarks/capillary_wave_2d/$(git rev-parse --short HEAD)
SVMP=/path/to/build/bin/svmultiphysics
TRANSPORT=pde_extension        # protocol transport (D9)
for form in surface_stress kag_lumped kag_consistent; do for L in 16 32 64; do
  d=$OUT/$TRANSPORT/$form/L$L
  python3 $B/generate_case.py --level $L --capillary-form $form --transport $TRANSPORT --output-dir $d
  cat > $d/run.sbatch <<EOS
#!/bin/bash
set -o pipefail
# set PATH and LD_LIBRARY_PATH for the solver build here (--export=NONE)
cd $d
timeout -k 10 <limit_s> mpiexec -n 1 --bind-to none $SVMP solver.xml 2>&1 | gzip -1 > solver_run.log.gz
EOS
  sbatch --export=NONE --partition=amarsden --time=<see table> --nodes=1 --ntasks=1 --mem=8G \
         --mail-user=$USER@stanford.edu --mail-type=BEGIN,END,FAIL \
         --job-name=capwave_${form}_L$L --output=$d/slurm-%j.out $d/run.sbatch
done; done
# after the jobs end:
python3 $B/verify.py $OUT/$TRANSPORT/surface_stress/L{16,32,64} \
    --json $OUT/$TRANSPORT/surface_stress.json
```

For the half-step check (D13) add a `--dt-divisor 2` run at `lambda/h = 64`
and pass it to `verify.py` together with the spatial study. `verify.py` reads
`solver_run.log.gz` for the per-step areas (D11), so keep the log beside the
output. For a quick schema check use `--max-steps 10`; `verify.py` refuses
such runs unless `--allow-truncated` is given.

## Smoke run

Slurm job `46075460` (2026-09-30, node `sh03-08n19`, AMD EPYC 7543, one
rank through `mpiexec -n 1 --bind-to none`), baseline binary at `fef0d02f`
(before the solver speed-ups), `SurfaceStress`, `La = 3000`. Case, log and
`verify_smoke.json` are in
`$SCRATCH/free-surface-benchmarks/capillary_wave_2d/smoke/`. The run predates
decisions D9 and D10: it advected `phi` with the coupled fluid velocity
(now `--transport coupled`) and used the earlier
level-dependent step (the capillary limit of each level, `dt = 3.99e-3` at
`lambda/h = 16` and `1.45e-3` at 32). The wet-extension and PDE transports
have not been run in this case.

- **`lambda/h = 16`, `--max-steps 10`** (0.1 inviscid period): the input
  parsed, all 10 steps were accepted, and the solver exited normally
  after 24 s (16.7 s to the first accepted step, including setup and JIT
  compilation). `verify.py --allow-truncated` read the output: `a_h/a_h(0)`
  fell from 1 to 0.8148 against 0.8155 for Prosperetti, maximum difference
  7.5e-4 and RMS 5.2e-4; the contact points followed the mode (wall
  elevations +0.00808 and -0.00818 at the end, against `a = 0.00815`);
  liquid-area drift 4.9e-7, growing over the ten steps (roughly as `t^3.5`
  so far). The current `verify.py` also parses the 11 per-step
  `Wet volume diagnostic` lines of this log (D11) and finds the same maximum. Frequency and damping are reported as not evaluable, as intended
  for a history shorter than one period.
- **`lambda/h = 32`, `--max-steps 5`**, run in the same job for the cost
  estimate only: exit 0; RMS amplitude difference 1.5e-5, area drift 7.6e-9.
- The log shows about 8 outer geometry passes per step, and one warning per
  step at `lambda/h = 16`, `ActiveFluid/WetVolumeFraction disagreement` on
  one cut cell. That diagnostic compares the fraction of wet vertices (2 of
  3) with the cut volume fraction (0.14) of a thin liquid sliver under the
  trough near the right wall; it is geometric and does not change the
  solve.

## Expected cost per level

**Measured** with the baseline binary `fef0d02f` and the coupled transport
(job `46075460`, step-accepted timestamps in the log, excluding the first
step): 0.675 s/step at `lambda/h = 16` (189 vertices) and 2.31 s/step at
`lambda/h = 32` (697 vertices), with about 8 outer passes per step. Between
the two the cost grows as (vertices)^0.94; `lambda/h = 64` is extrapolated
with exponent 0.94 to 1. The solver writes about 0.3 MB of log per step, so
compress the log (`gzip -1`) for protocol runs.

**D13 protocol.** Every level takes 200 steps (400 for the `dt/2` run).
Measured at `lambda/h = 64` (design note §9.5, 2026-10-03, term on, about 4.9
outer passes per step): 2,283 s at 50 steps per period and 4,072 s at 100,
against 10,442 s for the earlier 2900-step protocol. The coarser levels cost
less. The table below is for the earlier protocol.

**Earlier protocol (D10).** Every level takes 2900 steps; its time-step
study added 5800 and 11600 steps at `lambda/h = 32`.

| Run | vertices | steps | s/step (baseline) | time, baseline | time, tip (about 3x faster, estimate) |
|---|---:|---:|---:|---|---|
| `lambda/h = 16` | 189 | 2,900 | 0.675 (measured) | 33 min | about 11 min |
| `lambda/h = 32` | 697 | 2,900 | 2.31 (measured) | 1.9 h | about 40 min |
| `lambda/h = 64` | 2,673 | 2,900 | about 8.2 to 8.9 | about 7 h | about 2.5 h |
| `lambda/h = 32`, dt/2 | 697 | 5,800 | 2.31 | 3.7 h | about 1.3 h |
| `lambda/h = 32`, dt/4 | 697 | 11,600 | 2.31 | 7.4 h | about 2.5 h |

The tip column assumes the speed-up measured on the static drop at
`45bc5b09` (3.41 to 1.08 s/step, `static_drop_2d/README.md`); this case has
not been timed on the tip. Node generations differ by up to about 1.8x, and
the wet-extension and PDE transports and the KAG forms (about 50% more per
step on the static drop) were not timed here. Requests of 1 h, 3 h and 12 h
for the three levels and 6 h and 12 h for the time-step runs cover the
baseline timings and leave a margin of about three on the tip. Memory is small: 0.66 GB peak RSS for the batch
step of the smoke job.

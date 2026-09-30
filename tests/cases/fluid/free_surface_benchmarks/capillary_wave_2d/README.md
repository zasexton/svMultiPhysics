# capillary_wave_2d

A small-amplitude standing capillary wave on the free surface of a deep
one-phase liquid, released from rest. It measures the frequency and the
viscous damping rate of the unfitted level-set free surface against
Prosperetti's initial-value solution (milestone M3 in
`Documentation/free_surface_program_tracker.md`). The same case is run for
the three capillary routes compared in decision D2: `SurfaceStress`, KAG
with a lumped trace mass, and KAG with the consistent trace mass.

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
gives the source of each); only the geometry, the walls and the viscosity
differ.

| Input | Value | Source |
|---|---|---|
| Mesh | affine `Triangle3`, `h = lambda/level`, diagonals alternating with cell parity; 8 x 20, 16 x 40, 32 x 80 cells (189, 697, 2673 vertices) | geometry choice. With an even number of columns the mesh is mirror-symmetric about `x = lambda/4`, the node line of the mode, so crest and trough see the same mesh. |
| Mean-level offset | `sqrt(2)/100 lambda` above `y = lambda` | irrational, so the mean level lies on no grid line of the nested dyadic meshes and the sampled surface cannot pass exactly through a vertex. Among the simple irrational offsets tried (`pi/100`, `e/100`, `pi/1000`, `sqrt(2)/100`, ...) it is one of the two that keep `min abs(phi)/h` at or above 0.03 on all three levels: 0.066, 0.13, 0.030 at `lambda/h` = 16, 32, 64 (printed by `generate_case.py`). The static drop, by comparison, starts from 1.8e-3 to 6.8e-3. It has no physical effect (depth paragraph above). |
| Walls | `Dir`, value 0; `Effective_direction 1 0` (sides), `0 1` (bottom); full no-slip on the dry top | free-slip mirror planes and bottom (see Physical setup) |
| Free surface, cut stabilization, capillary forms, level-set transport, time integration, nonlinear and linear solves | identical to `static_drop_2d` (`SurfaceStress` / KAG keys, `RefreshedFrozenQuadrature`, `Interface_quadrature_order=2`, aggregation, pressure-gradient facet penalty 1.0, `Velocity_source=coupled_field`, SUPG 0.5/2.0, no reinitialization or volume correction, generalized-alpha `rho_inf = 0.5`, FSILS GMRES) | production values, see `static_drop_2d/README.md` |

**Level-set velocity at the walls.** The July capillary-wave deck advected
`phi` with the wall-compatible wet extension. Here the coupled fluid
velocity is used, as in `static_drop_2d`: its strong `u_x = 0` holds at
every side-wall vertex, dry or wet, so the advection never moves `phi`
through a wall, and no per-step extension map is written. This should be
checked once against the wet-extension variant at `lambda/h = 16`.

**Time step.** As in `static_drop_2d`, the one-sided capillary limit
`dt <= sqrt(rho h^3 / (4 pi gamma))`, i.e. the tracker form
`sqrt(rho h^3/(2 pi gamma))` times the fixed factor `1/sqrt(2)` derived
from the free-surface density sum. For this case the number of steps per
inviscid period at the limit is `sqrt(2) (lambda/h)^(3/2)`, independent of
`La`: 90.5, 256 and 724. `generate_case.py` rounds `dt` down so that the
run is exactly 100 equal output intervals: 400, 1100 and 2900 steps.
`--dt-divisor 2` or `4` divides that protocol step exactly by 2 or 4 (the
"three time steps" of the tracker); such runs are reported in a time-step
study and are not gated.

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
| `liquid_area_relative_drift_max` | `max_t abs(A(t) - A(0)) / A(0)` |
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

Each criterion applies to each (capillary form, `La`) refinement study at
the protocol time step.

| Criterion | Limit | Where | Source |
|---|---|---|---|
| `frequency` | at most 0.02, observed order at least 1 | 0.02 at `lambda/h = 32`; order over 16/32/64 | D1 working criterion, tracker M3 |
| `damping` | at most 0.05, observed order at least 1 | 0.05 at `lambda/h = 32`; order over 16/32/64 | D1 working criterion, tracker M3 |
| `volume_drift` | at most 1e-4 | every level | D1 working criterion (the volume limit of tracker M1 and M2, applied to M3) |

"Convergence over 16/32/64" is read as an observed order of at least 1, the
rate expected for the capillary force on a piecewise-planar interface (Gross
and Reusken 2011, ch. 7) and the rate required of the static-drop pressure
jump. `A(0)` includes the whole layer, so the area criterion corresponds to
a mean-level drift of `1e-4 y0 = 0.01 a0`; `mean_level_drift_over_amplitude`
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
for form in surface_stress kag_lumped kag_consistent; do for L in 16 32 64; do
  d=$OUT/$form/L$L
  python3 $B/generate_case.py --level $L --capillary-form $form --output-dir $d
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
python3 $B/verify.py $OUT/surface_stress/L{16,32,64} --json $OUT/surface_stress.json
```

For the time-step study add `--dt-divisor 2` and `--dt-divisor 4` runs at
one level (for example 32) and pass them to `verify.py` together with the
protocol runs. For a quick schema check use `--max-steps 10`; `verify.py`
refuses such runs unless `--allow-truncated` is given.

## Smoke run

Slurm job `46075460` (2026-09-30, node `sh03-08n19`, AMD EPYC 7543, one
rank through `mpiexec -n 1 --bind-to none`), baseline binary at `fef0d02f`,
`SurfaceStress`, `La = 3000`. Case, log and `verify_smoke.json` are in
`$SCRATCH/free-surface-benchmarks/capillary_wave_2d/smoke/`.

- **`lambda/h = 16`, `--max-steps 10`** (0.1 inviscid period): the input
  parsed, all 10 steps were accepted, and the solver exited normally
  after 24 s (16.7 s to the first accepted step, including setup and JIT
  compilation). `verify.py --allow-truncated` read the output: `a_h/a_h(0)`
  fell from 1 to 0.8148 against 0.8155 for Prosperetti, maximum difference
  7.5e-4 and RMS 5.2e-4; the contact points followed the mode (wall
  elevations +0.00808 and -0.00818 at the end, against `a = 0.00815`);
  liquid-area drift 4.9e-7, growing over the ten steps (roughly as `t^3.5`
  so far). Frequency and damping are reported as not evaluable, as intended
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

**Measured** (job `46075460`, step-accepted timestamps in the log, excluding
the first step): 0.675 s/step at `lambda/h = 16` (189 vertices) and 2.31 s/step
at `lambda/h = 32` (697 vertices). Between the two the cost grows as
(vertices)^0.94; `lambda/h = 64` is extrapolated with exponent 0.94 to 1.
The solver writes about 0.3 MB of log per step at every level, so compress
the log (`gzip -1`) for protocol runs.

| lambda/h | vertices | steps (dt/1) | s/step | time, this node type | request |
|---:|---:|---:|---:|---|---|
| 16 | 189 | 400 | 0.675 (measured) | 5 min | 30 min |
| 32 | 697 | 1,100 | 2.31 (measured) | 43 min | 3 h |
| 64 | 2,673 | 2,900 | about 8.2 to 8.9 | about 7 h | 24 h |
| 32, dt/2 | 697 | 2,200 | 2.3 | 1.4 h | 4 h |
| 32, dt/4 | 697 | 4,400 | 2.3 | 2.8 h | 8 h |

On `sh02` nodes the static drop ran 1.5 to 2 times slower per step, so the
requests leave a factor of about three. One capillary form therefore needs
about 8 node-hours for 16/32/64 and 4 more for the time-step study. The KAG
forms add a curvature projection to every outer pass and were not timed.
Memory is small: 0.66 GB peak RSS for the batch step (both smoke runs). The speed-up branch in
progress would shorten the `lambda/h = 64` runs most.

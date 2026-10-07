# oscillating_drop_2d

A two-dimensional liquid drop, fully enclosed by its free surface, released
from rest in the shape mode `n = 2`. It measures the Lamb (Rayleigh)
frequency and the viscous damping rate of the unfitted level-set free
surface against the exact linear viscous solution (milestone M3 in
`Documentation/free_surface_program_tracker.md`, "Oscillating 2D drop").
Unlike `capillary_wave_2d` the interface is closed and curved, there are no
walls or contact points on it, and the restoring force comes from the
curvature perturbation of a circle.

The protocol was fixed on 2026-10-07, before the first protocol run, from
the decisions in force for the capillary wave: `SurfaceStress`, the
harmonic PDE velocity extension (D9, D15), kinematic reconciliation (D14),
the lagged normal-increment capillary term with a fixed number of steps per
inviscid period shared by all levels (D13, D19), a `dt/2` check at the
finest level, the area criterion over the whole run (D11) and the damping
gate at the finest level (D20). The sign-definite patch bounds stay off
(D21 is for the sessile drop).

Files:

| File | Role |
|---|---|
| `generate_case.py` | writes `solver.xml`, the mesh with the initial fields, the wall faces and `case.json` for one level |
| `drop_reference.py` | the exact linear viscous solution (normal mode and initial-value solution) and its independent checks (numpy only) |
| `finite_amplitude.py` | the amplitude dependence of the frequency, from the nonlinear inviscid drop (numpy only) |
| `verify.py` | reads the solver output of one or more levels, computes the metrics, applies `tolerances.json` |
| `tolerances.json` | acceptance criteria and their sources, fixed on 2026-10-07 before the first protocol run |
| `tests/test_free_surface_benchmark_oscillating_drop_2d.py` | checks of the reference and of the scripts on synthetic data |

`generate_case.py` reuses the box mesh and the VTU writers of
`static_drop_2d/generate_case.py` and the level-set transport and linear
solver blocks of `capillary_wave_2d/generate_case.py`; `verify.py` reuses the
damped-cosine fit, the output reader and the log parser of
`capillary_wave_2d/verify.py` (as `fitted_capillary_wave_2d` does).

## Physical setup

Nondimensional units: density `rho = 1`, surface tension `gamma = 1`,
equilibrium radius `R = 1`.

- **Geometry.** The box `[0, 3R]^2` and the `Triangle3` mesh of
  `static_drop_2d`. The drop centre is `c = (1.5 + sqrt(3)/100, 1.5 + pi/100) R`.
  Zero gravity; the exterior is void with `p_ext = 0`. The box walls are dry
  no-slip walls, 3.7 cells or more from the drop at `R/h = 8`; with a void
  exterior they have no physical effect.
- **Released shape.** `r = R0 (1 + eps cos(2 theta))` about `c`, with
  `eps = 0.01` and `R0 = R / sqrt(1 + eps^2/2)`, so that the area is exactly
  `pi R^2` and the equilibrium radius is `R`. The mode amplitude is
  `a0 = R0 eps = 0.0099998 R`.
- **Viscosity** from the Laplace number `La = rho gamma D / mu^2`, `D = 2R`,
  as in `static_drop_2d`: `La = 800`, `mu = nu = 0.05`.

  | Quantity | Value |
  |---|---:|
  | `mu = nu` (= Ohnesorge number `mu / sqrt(rho gamma R)`) | 0.05 |
  | inviscid `omega0 = sqrt(n (n^2 - 1) gamma / (rho R^3)) = sqrt(6)`, period `T0 = 2 pi / omega0` | 2.44949, 2.56510 |
  | `eps_nu = nu / (omega0 R^2)` | 0.0204 |
  | normal-mode frequency `omega` (exact) | 2.42587 (`omega0` - 0.96%) |
  | normal-mode damping `beta` (exact) | 0.177334 (`2 n (n-1) nu / R^2 = 0.2` - 11.3%) |
  | two-term weak-viscosity expansion (below) | `omega` + 0.14%, `beta` + 1.4% off the exact values |
  | vorticity layer `delta = sqrt(2 nu / omega0)` | 0.20 R (1.6, 3.2, 6.5 `h` at `R/h` = 8, 16, 32) |
  | amplitude after 4 periods, `a(4 T0)/a0` | 0.156 |
  | viscous time `R^2 / nu` | 20 (the run is 10.26) |

  `beta/omega = 0.073` was chosen so that the four-period run damps the
  mode to about the same fraction as the capillary-wave protocol (0.156
  here, 0.133 there): at least four measurable damped periods, with a
  final amplitude still about 300 times the static mesh-induced shape
  moment (see Nonlinearity and noise). `2 n (n-1) nu/R^2` overestimates the
  damping by 13% and `omega0` the frequency by 1%, so the comparison is
  always made with the exact solution.
- **Nonlinearity and noise.** The fully nonlinear inviscid drop
  (`finite_amplitude.py`) gives `omega/omega0 - 1 = -0.770 eps^2`
  (converged in angular resolution and time step to four digits for
  `eps = 0.005` to `0.05`), i.e. `-7.7e-5` at `eps = 0.01`: 260 times below
  the frequency limit and 26 times below the time-step limit. The amplitude
  cannot be made much smaller: the relaxed static drop of M2 at `R/h = 8`
  carries a mesh-induced `cos(2 theta)` moment of about `5e-6 R` and a
  `sin(2 theta)` moment of `1e-4 R` (the alternating diagonals; computed
  from the `R/h = 8` outputs of job `46787442` with `verify.py`'s moments),
  so with `eps = 0.01` the measured `cos(2 theta)` component stays about
  300 times above that offset at the end of the run (`1.6e-3 R`), and the
  `sin(2 theta)` component, which the fit does not use, at 1% of `a0`.
- **Initial state** (the sampled analytic shape, D3):
  `phi = r - R0 - R0 eps cos(2 theta) (r / rho(theta))^2` at the P1
  vertices, with `rho(theta) = R0 (1 + eps cos(2 theta))`. Its zero set is
  exactly the released shape; the factor `(r/rho)^2` keeps `phi` and its
  gradient continuous at the drop centre, and `|grad phi| = 1 + O(2 eps)` at
  the surface. `u = 0`, and the linear pressure of the released state,
  `p = gamma/R + gamma (n^2 - 1) a0 (r/R)^n cos(n theta) / R^2`
  `= 1 + 3 a0 ((x - c_x)^2 - (y - c_y)^2)`: harmonic, equal to
  `gamma kappa` on `r = R`, and continued smoothly through the dry vertices
  that cut cells need.

## Reference solution (`drop_reference.py`)

**Inviscid frequency.** For `r = R + a cos(n theta)`, the curvature is
`kappa = 1/R + (n^2 - 1) a cos(n theta) / R^2`; with the potential
`A r^n cos(n theta)`, the kinematic condition and Bernoulli's equation,
`omega0^2 = n (n^2 - 1) gamma / (rho R^3)`. This is Rayleigh's result for
the oscillation of a liquid column (jet) at zero axial wavenumber (Lord
Rayleigh, "On the capillary phenomena of jets", Proc. R. Soc. Lond. 29, 71
(1879); see also Lamb, *Hydrodynamics*, 6th ed., 1932, ch. IX), the 2D
counterpart of `n (n - 1)(n + 2)` for a sphere.

**Weak viscosity.** Lamb's dissipation method (Lamb 1932, ch. XI, for the
sphere) applied to the potential flow `A r^n cos(n theta)`: the dissipation
`mu oint d|u|^2/dr ds = 4 pi mu (n - 1) n^2 A^2 R^(2n-2)` and the kinetic
energy `pi rho n A^2 R^(2n) / 2` give the amplitude decay rate
`2 n (n - 1) nu / R^2` (also Aalilija, Gandin and Hachem, "On the analytical
and numerical simulation of an oscillating drop in zero-gravity", Comput.
Fluids 197, 104362 (2020), who note that no exact damped solution of the 2D
drop had been written down).

**Exact linear viscous solution.** Laplace transform in time with `u(0) = 0`
and `eta(0) = a0 cos(n theta)`. Write `u = grad(phi) + curl(psi e_z)` with
`phi = A r^n cos(n theta)` (pressure perturbation `-rho s phi`) and
`psi = B I_n(q r) sin(n theta)`, `q^2 = s / nu` (vorticity equation). At
`r = R`:

- zero tangential stress `r d(u_theta/r)/dr + (1/r) du_r/dtheta = 0`;
- normal stress `-p + 2 mu du_r/dr = -gamma (n^2 - 1) a_hat / R^2`;
- kinematic condition `s a_hat - a0 = u_r(R)`.

Eliminating `A` and `B`, with `x = R sqrt(s / nu)` and the Bessel ratio
`r(x) = x I_n'(x) / I_n(x)`:

```text
a_hat(s) = a0 (s + c Q) / (s (s + c Q) + omega0^2 P),      c = 2 n (n - 1) nu / R^2,
P = 1 + 2 n (n - 1) / W,   Q = 1 + 2 n (r - 1) / W,   W = 2 r - x^2 - 2 n^2.
```

`P` and `Q` depend on `x^2` only, so `a_hat` is meromorphic in `s`: a
bounded drop has a discrete spectrum. The normal modes are the roots of
`s^2 + c Q s + omega0^2 P = 0`, the 2D counterpart of the relations of
Chandrasekhar (Proc. London Math. Soc. 9, 141 (1959)) and Reid (Q. Appl.
Math. 18, 86 (1960)) for the sphere. The initial-value solution is the
residue sum, the method of Prosperetti for the sphere ("Free oscillations
of drops and bubbles: the initial-value problem", J. Fluid Mech. 100, 333
(1980)):

```text
a(t) = 2 Re(A_c exp(s_c t)) + sum_k A_k exp(s_k t),
```

with `s_c = -beta + i omega` the least-damped (oscillatory) root and `s_k`
the real, non-oscillatory viscous modes on the negative axis. At `La = 800`:
`2 |A_c| = 1.012 a0`; the slowest real mode decays at 1.07 (six times faster
than the oscillation) with `A_1 = -8.4e-3 a0`. `s = 0` is a removable root.

Numerics: `r(x)` by the continued fraction of `I_(n+1) / I_n` (modified
Lentz), normal mode by Newton's method continued in `nu` from
`eps_nu = 1e-6`, real modes by a sign scan of the real entire function
`J_n W (dispersion relation)` on `x = i xi` with `J_n` from the periodic
trapezoidal rule on Bessel's integral (power series for `xi <= 2`), refined by
bisection; enough real modes are kept that `exp(s_k t_min)` falls below
`1e-17`. `a(0) = a0` is set exactly (the sum converges only conditionally
there).

**Weak-viscosity expansion.** From `r(x) = x - 1/2 + O(1/x)` for large `x`,
the dispersion relation reduces to `S^2 + Omega0^2 + 4 n (n-1) S - 4 n (n-1)^2 sqrt(S) = O(1)`
(`S = s R^2 / nu`, `Omega0 = omega0 R^2 / nu = 1/eps_nu`), so

```text
beta  = 2 n (n - 1) nu / R^2 (1 - (n - 1) sqrt(eps_nu / 2) + O(eps_nu)),
omega = omega0 (1 - sqrt(2) n (n - 1)^2 eps_nu^(3/2) + O(eps_nu^2)).
```

For `n -> infinity` at fixed `k = n / R` both corrections become Lamb's
planar ones, `-sqrt(eps_k / 2)` and `-sqrt(2) eps_k^(3/2)` with
`eps_k = nu k^2 / omega0`, which `capillary_wave_2d` checks.

**Stokes limit.** Without inertia the stream function
`(A r^n + B r^(n+2)) sin(n theta)` with zero tangential stress gives
`da/dt = -n gamma a / (2 mu R)`: the slowest real mode must tend to
`-n gamma / (2 mu R)` as `nu -> infinity`.

Checks, all in the test file:

| Check | Result |
|---|---|
| `r(x)` and `J_n` against power series (complex `x`, `n = 2, 3`) | relative 1e-13, absolute 1e-12 |
| inviscid limit (`nu = 0`) | `a(t) = a0 cos(omega0 t)` |
| independent derivation: `a_hat(s)` from a Chebyshev collocation of the transformed equations in stream-function form (`laplace_transform_collocation`: `nu L G = s G`, `G = L F`, pressure from the theta momentum equation, the three surface conditions; no Bessel functions, no potential/vortical splitting) | equal to the closed form within 1e-9 (mostly 1e-11 to 1e-14) at 7 complex `s` and `nu = 0.002, 0.05, 1`; the pole of the collocation transform (secant iteration) is the normal mode within 1e-9 |
| initial-value solution | sum of all residues `= a0` within 1e-9; `a(t) = a0 (1 - omega0^2 t^2/2) + o(t^2)`; its numerical Laplace transform equals `a_hat(s)` within 1e-9 at `s / omega0 = 0.5, 1, 3`, and the partial-fraction sum within 1e-12 (no root missing) |
| weak viscosity | the coefficients of both corrections above within 1% (damping) and 2.5% (frequency) for `eps_nu = 1e-4 ... 1e-6`, `n = 2` and 3, converging; at `nu = 1e-4` the history is `exp(-2 n (n-1) nu t) cos(omega0 t)` within 2.5e-4 |
| Stokes limit | slowest real mode within 2e-3 of `-n gamma/(2 mu R)` at `nu = 10, 30, 100`, converging |
| planar limit (`n = 50, 100, 200`, `R = n / k`, `La = 3000` of the capillary wave) | frequency within 3e-4 and damping within 0.5% of `capillary_wave_2d`'s normal mode, the damping difference halving with each doubling of `n` |
| protocol fit | the damped-cosine fit of `a(t)` over the protocol outputs is within 0.033% (frequency) and 0.41% (damping) of the normal mode |

## Discretization and fixed inputs

Every value below is fixed for all levels (principle P1). The solver inputs
are those of `static_drop_2d` and `capillary_wave_2d` (their READMEs give the
source of each); only the drop shape, its centre, the viscosity, the
initial pressure and the time step are specific to this case.

| Input | Value | Source |
|---|---|---|
| Mesh | affine `Triangle3`, `h = R/level`, diagonals alternating with cell parity, 24/48/96 cells per side (625/2401/9409 vertices) at `R/h` = 8/16/32 | `static_drop_2d` |
| Drop centre | `(sqrt(3), pi)/100 R` from the box centre | irrational, so the sampled surface cannot pass exactly through a vertex of the nested grids. Among 15 simple offsets of this kind (`(pi, e)/100`, `(e, pi)/100`, `(sqrt 2, sqrt 3)/100`, ...) it gives the largest vertex clearance of the released shape over the three levels: `min abs(phi)/h` = 0.018, 0.0082, 0.011 (printed by `generate_case.py`). The static-drop offset `(pi, e)/100` would leave 1.3e-6 at `R/h = 32`. |
| Mode orientation | `cos(2 theta)` aligned with the mesh axes | the mesh anisotropy of the alternating diagonals enters `sin(2 theta)` (reported, not fitted) |
| Walls | `Dir`, value 0 (no-slip, dry) | `static_drop_2d` |
| Free surface, cut stabilization, capillary form, level-set discretization, time integration, nonlinear and linear solves | `UnfittedLevelSet`, `LevelSetNegative`, `CutVolume`, `LinearCorner`, `RefreshedFrozenQuadrature`, `Interface_quadrature_order=2`, `SurfaceStress`, aggregation, pressure-gradient facet penalty 1.0, SUPG 0.5/2.0, no reinitialization or volume correction, generalized-alpha `rho_inf = 0.5`, FSILS GMRES (RCS, 100 iterations, Krylov 50, 1e-8 / 1e-10) | production values, `static_drop_2d/README.md` |
| Level-set advection velocity | harmonic PDE extension, monolithic coupling (`--transport pde_extension`) | D9, D15 |
| Kinematic reconciliation | on (`Enable_kinematic_reconciliation=true`) | D14 |
| Sign-definite patch bounds | off (solver default) | D21 applies to the sessile drop only |
| Capillary term | `Surface_tension_semi_implicit=NormalIncrement` | D13 |

**Time step.** As D13/D19 for the capillary wave, every level uses a fixed
number of steps per inviscid period `T0 = 2 pi / omega0`, one step shared by
all levels (D10). The number is 100 here, not the capillary wave's 50:
`dt = 2.5651e-2`, 400 steps over 4 periods, outputs every 4 steps (100
outputs); `dt` is 2.1, 5.8 and 16.5 times the one-sided capillary limit
`sqrt(rho h^3 / (4 pi gamma))` at `R/h` = 8, 16, 32. The reason was fixed
before any protocol run: the mode-2 wavelength `pi R` is resolved by 25, 50
and 100 cells, finer than the capillary wave's `lambda/h` = 16, 32, 64, so
the spatial frequency error at `R/h = 32` is expected to fall below the time
error of 50 steps per period, which the capillary-wave runs measured as
-0.13% in frequency and -0.36% in damping (job `46704019`; the spatial
frequency error there was +1.7%, +0.5%, 0.0% at `lambda/h` = 16, 32, 64). A
shared time error of that size would set a floor under the spatial errors
and cap the observed order over 8/16/32. At 100 steps per period the
generalized-alpha phase error `(omega dt)^2/12` is about 3e-4.

**Half-step check.** `--dt-divisor 2` gives 200 steps per period with the
same output times; the protocol runs it at `R/h = 32`. Every gate must also
pass for the `dt/2` run at `R/h = 32`, and between `dt` and `dt/2` the fitted
frequency may change by at most 0.2% and the damping by at most 1%
(`time_step_criterion`). The gated verdict is `verify.py` on the three `dt`
runs and the `R/h = 32` `dt/2` run. The protocol job also runs `dt/2` at
`R/h = 8` and 16; those runs are verified separately and only reported (the
time-step study at every level).

**Run length.** 4 inviscid periods (`t = 10.26`, half a viscous time), 100
VTU outputs, 25 per period.

## Metrics (`verify.py`)

All quantities come from the `phi` point data of the solver's VTU/PVTU
output, plus `phi` of `mesh/mesh-complete.mesh.vtu` for `t = 0`.

| Metric | Definition |
|---|---|
| Harmonic moments | `M_m = int_{phi_h < 0} (z - c)^m dA`, `z = x + i y`, `m = 0..4`, exactly for the polygonal P1 region (each cut triangle clipped by the linear `phi_h`, every polygon integrated over its edges with the complex Green formula `int z^m dA = (1/2i) oint z^m conj(z) dz` and a 4-point Gauss rule, exact for `m <= 6`), then shifted to the centroid `M_1 / M_0` |
| Liquid area `A(t)` | `M_0`, the exact area of `{phi_h < 0}` |
| Mode amplitude `a_h(t)` | `Re M_2 / (pi R_A^3)` about the centroid, `R_A = sqrt(A / pi)`. For `r = R_A + sum_m (a_m cos(m theta) + b_m sin(m theta))`, `M_m = pi R_A^(m+1) (a_m + i b_m) + O(a^2)`; for the released shape it equals `R0 eps` to `O(eps^4)`. Translation enters only at second order (moments about the centroid), the area drift and the other modes not at all. |
| Reference | the exact linear `a(t)` at `R_eff = sqrt(A_h(0)/pi)`, the equilibrium radius of the sampled drop (as `static_drop_2d`); `R_eff/R - 1` = -1.2e-3, -3.2e-4, -8e-5 at `R/h` = 8, 16, 32, which would shift `omega0` by 0.18%, 0.05% and 0.012%. The comparison at the nominal `R` is reported. |
| Fit | `a(t) = exp(-beta t) (c1 cos(omega t) + c2 sin(omega t))` by least squares over all outputs including `t = 0` (`capillary_wave_2d/verify.py`); the same fit is applied to the reference at the same times |
| `frequency_relative_error` | `abs(omega_sim - omega_ref) / omega_ref` (the signed value is printed) |
| `damping_rate_relative_error` | `abs(beta_sim - beta_ref) / beta_ref` (the signed value is printed) |
| `liquid_area_relative_drift_max` | `max_t abs(A(t) - A(0)) / A(0)` over the whole run (D11): over the outputs and, when `solver_run.log[.gz]` is in the case directory, over the per-step `Wet volume diagnostic` areas |
| Reported only | `a_h(0)/a0 - 1` (sampling error: -9.7e-4, -3.8e-4, -5.9e-5); `amplitude_rms_error`, the RMS of `a_h(t)/a_h(0) - a(t)/a0`, and its maximum; the largest `sin(2 theta)` component and the largest other mode (`m = 3, 4`) relative to `a0`; the centroid drift; the errors against the reference at the nominal `R`; outer passes per step, Newton iterations per step, GMRES iterations per Newton iteration and wall time from the solver log and `run.txt`; the histories |

Frequency and damping need at least one inviscid period and 8 samples;
shorter (smoke) runs report them as not evaluable. `verify.py` exits with
status 2 on missing or inconsistent data (no `case.json`, no output,
missing `phi`, non-triangle cells, non-finite values, a run that stopped
before its end time, levels with different time steps in one study) and on
a `--max-steps` smoke run unless `--allow-truncated` is given.

## Tolerances and their sources

Each criterion applies to each (capillary form, transport, `La`, dt divisor)
spatial study at one shared step (D10). The divisor-1 study needs every
level; the divisor-2 study needs `R/h = 32`. The time-step criterion is
printed as a separate gated line, or as not evaluated with one divisor.

| Criterion | Limit | Where | Source |
|---|---|---|---|
| `frequency` | at most 0.02, observed order at least 1 | 0.02 at `R/h = 32`; order over 8/16/32 | D1 working criterion, tracker M3 |
| `damping` | at most 0.05, observed order at least 1 | 0.05 at the finest level `R/h = 32`; order over 8/16/32 | D1 working criterion, tracker M3; finest level as D20 (and M2, D12) |
| `volume_drift` | at most 1e-4 | every level, maximum over the run | D1 working criterion (M1, M2, capillary wave), gated as in D11 |
| `time_step` | frequency change at most 0.002, damping change at most 0.01 between `dt` and `dt/2` | `R/h = 32` | D19 (M3) |

Observed order is the least-squares slope of `log(error)` against
`log(R/h)`; pairwise orders are printed. First order is the rate expected
for the capillary force on a piecewise-planar interface (Gross and Reusken
2011, ch. 7), as for the static drop and the capillary wave.

## How to run the refinement study

Use the Python stack from the benchmark README (numpy; pyvista for
`verify.py`). Put generated cases and output under `$SCRATCH`, and start the
solver through `mpiexec` from a job submitted with `--export=NONE` (benchmark
README, "Launching the solver"); 4 ranks at `R/h = 16` and 32, serial at 8.

```bash
B=tests/cases/fluid/free_surface_benchmarks/oscillating_drop_2d
OUT=$SCRATCH/free-surface-benchmarks/oscillating_drop_2d/$(git rev-parse --short HEAD)
for L in 8 16 32; do
  python3 $B/generate_case.py --level $L --output-dir $OUT/L${L}_dt1
  python3 $B/generate_case.py --level $L --dt-divisor 2 --output-dir $OUT/L${L}_dt2
done
# one lane per case (run_lanes.sbatch: lane|ranks|case_dir|timeout_s), e.g.
#   timeout -k 30 <s> mpiexec -n 4 ... svmultiphysics solver.xml 2>&1 | gzip -1 > solver_run.log.gz
python3 $B/verify.py $OUT/L{8,16,32}_dt1 $OUT/L32_dt2 --json $OUT/verify.json
```

`verify.py` reads `solver_run.log.gz` for the per-step areas (D11) and the
solver statistics, and `run.txt` for the wall time, so keep both beside the
output. For a quick schema check use `--max-steps 10`; `verify.py` refuses
such runs unless `--allow-truncated` is given.

## Smoke run

Slurm job `46902287` (2026-10-07, node `sh02-10n27`, Intel Xeon Gold 5118),
binary `svmultiphysics-4cbf2643-ltopgo` (LTO + PGO build of the D26 fix),
generator at `7e0e6867`, `--max-steps` 10/10/6 at `R/h` = 8/16/32 (serial,
4 ranks, 4 ranks), one lane each. Case, logs and `verify_smoke.json` are in
`$SCRATCH/free-surface-benchmarks/oscillating_drop_2d/smoke-7e0e6867/`.

- Every step was accepted at every level and every run exited normally; the
  input parses, the PVTU output and the per-step wet-volume lines are read by
  `verify.py --allow-truncated`.
- Over the first 0.1 period: amplitude RMS difference from the reference
  7.8e-4, 2.9e-5 and 2.7e-6 (relative to `a0`); area drift 5.9e-9, 1.5e-6
  and 2.6e-7; centroid drift at most 4.7e-6 R.
- Solver statistics of these start-up steps: 4.2 to 5.0 outer passes per
  step, 4.3 to 6.4 Newton iterations per step, and 61, 133 and 508 GMRES
  iterations per Newton iteration at `R/h` = 8, 16, 32. The only warning is
  the geometric `ActiveFluid/WetVolumeFraction disagreement` diagnostic at
  `R/h = 8` (one cut cell per step, as in `capillary_wave_2d`).
- Time per step after start-up: 0.95 s (`R/h = 8`, serial), 1.75 s
  (`R/h = 16`, 4 ranks) and 12 s (`R/h = 32`, 4 ranks).

## Expected cost

From the smoke run: 400 steps take about 6.5 min, 12 min and 80 min at
`R/h` = 8, 16, 32 (serial, 4, 4 ranks), and the `dt/2` runs (800 steps) at
most twice that. The protocol job packs the six runs into four lanes on one
node (13 CPUs: `R/h = 32` at `dt/2`; `R/h = 32` at `dt`; `R/h = 16` at `dt`
then `dt/2`; `R/h = 8` at `dt` then `dt/2`). Memory is small (2.6 GB peak for
the whole smoke job).

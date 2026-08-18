# D18/D38 SPHERIC Test05 Wet-Bed Dam-Break Validation: Root-Cause Report

Date: 2026-06-10
Cases: `tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test05_wet_bed_d18`, `..._d38`
Reference data: digitized SPHERIC Test Case 05 free-surface profiles
(`tests/cases/fluid/open_vessel_free_surface/reference_profiles/spheric_test05_wet_bed/d{18,38}_*.dat`,
times 0.156–0.593 s, stored in cm).

## Executive summary

The observed "differences in fluid boundary evolution" decompose into three
independent layers:

1. **Blocking regression (current HEAD `7aaadf5a`)**: D18/D38 fail their first
   nonlinear solve. Cause: the transient ghost-penalty rescaling
   `gamma_v * (mu + rho*h^2/dt) * h` with default `gamma_v = 1.0`. For water at
   h = 15 mm, dt = 5e-4 s, `mu_gp/mu ~ 2e5`; the velocity-jump penalty rows
   (~2.0) sit 3–6 orders above neighboring operator rows, the Jacobian becomes
   ill-conditioned, Newton directions blow up quadratically, and the line
   search cannot reduce the residual at any step size. Empirical stability
   bracketing: effective coefficients `1e-2` converge in 1 Newton
   iteration/step; `0.1` and `1.0` fail at step 0. **Remediated in this change
   set** (see "Remediation").
2. **Validation-config profile differences vs the SPHERIC reference are
   resolution-class effects, not configuration bugs.** Height-level crossing
   analysis shows the bore mid-face position is essentially exact
   (±2 mm at t = 0.219 for both cases). The systematic discrepancies are:
   (i) a smeared bore toe that leads by +25..+60 mm exactly in the
   `bed + 10 mm` threshold band sampled by the "elevated front" metric,
   (ii) a rounded crest where the reference shows a near-vertical/overhanging
   breaking face (−18..−37 mm at crest heights), and (iii) a gate-region
   drawdown lag (+20..+40 mm) that is identical across every ablation
   (gauge pin on/off, pressure penalty strong/weak) and is therefore a
   spatial-resolution limit at h = 15 mm.
3. **Late-time D18 corruption (t > 0.22)**: dry-side level-set false wetting.
   phi at lid/side-wall corner nodes above the gate crashes from +0.035 to
   −0.011 within 10 steps (equivalent −8 m/s; local fluid velocity ~0), falsely
   wetting 6 nodes at t = 0.235, growing to 32 by t = 0.31. The cadence-1
   global volume correction (a uniform phi shift) conserves total volume by
   uniformly depressing the real interface. The D18 gate metric at t = 0.281
   (+105 mm) is largely this artifact stacked on the real ~+30 mm drawdown lag.
   D38 (deeper bed, milder cliff dynamics) is completely clean; its tables are
   trustworthy at all compared times. Enabling projection reinitialization
   (cadence 5, band-preserving) does **not** cure it — onset moves *earlier*
   (12 spurious nodes by t = 0.15) because the repair propagates noisy
   gate-cliff interface fragments outward as distance values.

## Profile error tables (current code, old-equivalent penalty magnitudes)

Comparison harness: `/tmp/svmp_fs_validation/test05_multitime_compare.py`
(maps `result_NNN.vtu` to physical time via the solver log; samples the
phi = 0 contour max-y at the digitized reference x stations; gate region
x in [0.375, 0.45], bore region x > 0.45; front = max x with
y > bed + 10 mm).

| time (s) | D18 RMSE / gate-mean / bore-mean / front (mm) | D38 RMSE / gate-mean / bore-mean / front (mm) |
| --- | --- | --- |
| 0.156 | 22.3 / +38.2 / +3.1 / +40.2 | 21.0 / +36.5 / −3.2 / +48.6 |
| 0.219 | 20.8 / +41.1 / +0.7 / +38.0 | 15.8 / +20.8 / −3.8 / +81.2 |
| 0.281 | 44.9* / +105.4* / −2.1 / +29.3 | 20.1 / +33.3 / −0.7 / +87.7 |

(*) D18 at 0.281 contaminated by the false-wetting blob (finding 3).

Height-level crossing decomposition at t = 0.219 (lead vs reference, mm):

| level | D18 | level | D38 |
| --- | --- | --- | --- |
| toe band (0.028 / 0.035) | +37.3 / +24.3 | toe band (0.050) | +61.2 |
| mid-face (0.045) | **+1.4** | mid-face (0.060) | **+1.7** |
| crest (0.055 / 0.065) | −18.4 / −37.2 | crest (0.070 / 0.080) | −20.0 / −22.6 |

The D38 "front-lead growth" (+48.6 → +87.7 mm) is entirely the toe band; the
bulk bore face is accurately placed. The elevated-front validation metric
should be treated as toe-band-fragile; mid-height crossings are robust.

## Exonerated mechanisms (with evidence)

- **Pressure gauge pin** (`Node_pressure_constraints`, permanent
  `VertexDirichletConstraint` at the initial hydrostatic value,
  `IncompressibleNavierStokesVMSModule.cpp` `registerOn`): produces a real,
  growing local pressure corruption (±400–1000 Pa checkerboard centered on the
  pinned node by t = 0.156; pin sits +330 Pa above its ring by t = 0.31) and
  locally voids the continuity row. **Zero effect on interface evolution**:
  the no-pin run matches the pinned run to 0.1 mm in every profile metric at
  every compared time, with identical reservoir drawdown. Recommendation:
  remove the pin anyway once the hydrostatic initialization is fixed (below);
  it over-constrains a pressure field whose datum is already set by the
  free-surface condition plus dry-side Dirichlet pins.
- **Hydrostatic initialization** (`hydrostaticPressureAt`): global linear fill
  from the reference point; not level-set aware. The wet bed initializes at
  1292–1469 Pa where the physical local-column value is ≤176 Pa. The
  incompressible projection absorbs it within one solve (bed velocities ~1e-3
  m/s afterwards), so it is benign for evolution — but it explains the
  2026-05-26 step-0 FSILS true-residual failures that motivated adding the
  gauge pin. Recommendation: compute the fill from the local column height
  (level-set aware) and drop the pin.
- **Pressure ghost penalty magnitude**: weakening it 2e5× (the new
  `h^3/mu_gp` scaling) changes the gate error by +2 mm; not the gate-lag cause
  and harmless to stability.
- **Volume correction**: applied shifts are ~1e-6..9e-6 per step; not the
  false-wetting trigger (it is the *amplifier* that converts a spurious blob
  into a global interface depression).
- **JIT**: `EffectiveTimeStep` is loaded at runtime (`loadEffectiveDt(side)`),
  not specialized into kernels.
- **Env-gated Newton/TimeLoop fixes from `7aaadf5a`**
  (`SVMP_SYNC_LINE_SEARCH_TRIALS`, `SVMP_DISTRIBUTE_CONSTRAINTS_INTO_HISTORY`,
  `SVMP_ZERO_CONSTRAINED_RATES`): step-0 failure is bit-identical with all
  legacy toggles; a 330-step legacy-toggle run matches the default run to
  0.4 mm at every height level.
- **Old-build (Jun 2) front "exact match" at t = 0.156** was a stagnation
  artifact: its elevated front froze at x = 0.4586 during t ≈ 0.14–0.156,
  exactly where its 33 adaptive-dt rejections cluster, and it *lags* the
  reference at all height levels at 0.156. The current build (zero rejections,
  mid-face exact by 0.219) is the healthier trajectory. The behavior change
  traces to the non-env-gated level-set machinery changes
  (`87928980` cut-context/transport rework and/or the `7aaadf5a`
  velocity-extension dry-cut down-weighting), not to the Newton/TimeLoop
  fixes.

## Finding 1 detail: step-0 Newton failure mechanism

At rest with hydrostatic init, both ghost-penalty residual contributions are
zero (no velocity, elementwise-constant hydrostatic pressure gradient), so the
first iterate's residual (~0.31, the physical impulsive start) is
penalty-independent. The penalty enters through the Jacobian: with rows ~2.0
against extension-floor rows ~1e-3 and momentum rows ~1, the exactly-solved
Newton direction acquires enormous components in weakly-determined subspaces;
the quadratic (convection/VMS) response to that direction exceeds the linear
residual decrease at every line-search step size, and the backtracking line
search (`NewtonSolver.cpp`, Armijo on 0.5*||r||^2 with best-trial fallback)
correctly reports no reduction. Shrinking dt makes it worse (the penalty grows
as 1/dt): the full-step trial residual goes 0.87 (dt = 5e-4) → 16.7
(dt = 2.5e-4). The MMS verification matrix passed with the same defaults
because there `rho*h^2/(mu*dt) ~ O(1)` — the regression is parameter-regime
dependent, which is why existing gates missed it.

Probe matrix (6/3-step runs, `/tmp/svmp_d18_probe_*`):

| velocity coeff (gamma_v) | pressure coeff (gamma_p) | step 0 |
| --- | --- | --- |
| 1.0 (stock) | 1.0 | fails |
| disabled | disabled | converges (8 iters) |
| 5e-6 (old-equivalent mu*h) | 2e5 (old-equivalent h^3/mu) | converges, 1 iter |
| 1.0 | 2e5 | fails |
| 5e-6 | 1.0 | converges, 1 iter |
| 1e-2 | 1.0 | converges, 1 iter |
| 0.1 | 1.0 | fails |

## Remediation (this change set, working tree only — no commits)

The transient generalized diffusivity `mu_gp = mu + rho*h^2/dt` is the
literature-correct scale (Schott & Wall; Burman, Fernandez & Massing), but the
literature pairs it with a small dimensionless ghost-penalty constant
(typically 0.005–0.05), not 1.0. The fix introduces an internal calibration
constant `kCutPenaltyTransientCalibration = 0.01` multiplying both cut-cell
penalty coefficients, so the user-facing
`Cut_cell_velocity_gradient_penalty` / `Cut_cell_pressure_gradient_penalty`
remain relative multipliers with default 1.0 = recommended calibrated value:

- velocity: `gamma_v * 0.01 * (mu + rho*h^2/dt) * h` (and `* h^3` for the
  second-normal-derivative term),
- pressure: `gamma_p * 0.01 * h^3 / (mu + rho*h^2/dt)` (and `* h^5`).

With the checked-in case XMLs (gamma = 1.0) the effective velocity coefficient
lands exactly on the empirically verified stable point (probe `1e-2`:
1 Newton iteration/step), still ~2000× stronger than the pre-`7aaadf5a`
viscous-only scaling for the water regime, with a 10× margin to the observed
failure threshold (0.1). Validation results for the fix are recorded in the
companion log
`Documentation/qualification_logs/d18_d38_gamma_gp_calibration_fix_20260610.md`.

## Recommended follow-ups (not in this change set)

1. **Dry-side level-set robustness (finding 3)**: filter tiny/noisy interface
   fragments from the reinitialization source geometry and/or rate-limit
   dry-side phi updates (|dphi/dt| <= |a||grad phi| with a CFL-like bound);
   verify with a gate-region-refined D18 mesh. Until then, treat D18 metrics
   beyond t ≈ 0.22 as contaminated.
2. **Level-set-aware hydrostatic initialization**, then remove the D18/D38
   gauge pins (and re-check SPHERIC test10, which had the same setup issue).
3. **Validation metrics**: replace the elevated-front threshold metric with
   mid-height crossing positions; tighten the profile gates (current
   RMSE <= 0.12 m gate is ~5× looser than observed errors).
4. **Regression gate**: add a water-regime cut-cell step-0 probe
   (D18 mesh, 2–3 steps) to the test suite so penalty-scaling regressions in
   the `rho*h^2/(mu*dt) >> 1` regime cannot land silently.
5. **Mesh refinement study** (h = 7.5 mm) for the gate-region drawdown lag and
   crest representation.

## Artifacts

- Runs (preserved): `/tmp/svmp_d18_run_{C,C_nogauge,C_legacy,E,R,NG620}`,
  `/tmp/svmp_d38_run_C`, probes `/tmp/svmp_d18_probe_{A,C,D,E,F,G,H,legacy}`;
  per-run `multitime_profile_comparison.json`.
- Jun-2 baselines: `/tmp/dam_break_d18_60uivfox`, `/tmp/dam_break_d38_qwhgcrxz`
  (+ `forensics_history.json`).
- Tooling: `/tmp/svmp_fs_validation/test05_multitime_compare.py`,
  `/tmp/svmp_fs_validation/test05_forensics.py`.

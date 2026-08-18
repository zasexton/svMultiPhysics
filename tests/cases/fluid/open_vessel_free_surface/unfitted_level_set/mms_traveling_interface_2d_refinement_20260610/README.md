# MMS Traveling Interface 2D — Refinement Matrix Record (2026-06-10)

First fully passing refinement matrix for the unfitted level-set free-surface
MMS. All four runs pass the complete `verify_expected_results.py` gate at
`t = 0.5` (half period), including the manufactured-source and level-set
residual audits.

## Code state

Base commit `87928980` plus the following uncommitted method fixes (this
record documents the state they were validated against):

1. Transient cut-cell ghost-penalty scaling: velocity jump penalties scale
   with `(mu + rho*h^2/dt)*h` (and `*h^3`), pressure jump penalties with
   `h^3/(mu + rho*h^2/dt)` (and `h^5`), replacing the viscous-only `mu*h` and
   `h^3/mu` scalings (`IncompressibleNavierStokesVMSModule.cpp`).
2. Cut metadata stabilization scale aggregates rule volume fractions per
   (cell, side) before inversion instead of using individual subdivision-leaf
   fractions, which saturated the 1e3 cap on every facet
   (`CutIntegrationContext.h::buildCutCellStabilizationScales`).
3. Velocity-extension Laplacian down-weighted on the dry parts of cut cells
   (full-cell indicator with a 1e-3 floor) so wet-support momentum rows stay
   physics + ghost penalty. The floor keeps every dry-side DOF attached to an
   equation; a pure zero mask produced singular zero rows when a cut cell's
   wet rules were pruned (`applyFreeSurfaceVelocityExtension`).
4. Band-preserving projection reinitialization:
   `LevelSetReinitializationOptions::preserve_band_width` (default automatic
   ~1.5 interface-primitive diameters) preserves near-interface DOFs so the
   high-order interface is never replaced by its corner linearization
   (`LevelSetReinitialization.cpp`).
5. Residual-consistent line-search synchronization: when cuts, constraints,
   curvature, or advection velocity define the nonlinear residual, they are
   refreshed at `LineSearchTrialResidual` sync points and each backtracking
   trial is rolled back transactionally.  The explicit legacy opt-out is
   `SVMP_SYNC_LINE_SEARCH_TRIALS=0` (`ApplicationDriver.cpp`).

## Configuration

Generated with `generate_case.py --nx N --ny N --time-step DT --time-steps S
--final-time 0.5 --disable-reinitialization
--cut-cell-pressure-stabilization-policy Enabled`.

Q2 velocity / Q1 pressure (Taylor–Hood), Q2 level set,
`HighOrderImplicit` + `SayeHyperrectangle` depth 6 (achieved volume and
interface quadrature order 2, zero fallback cells), velocity extension
enabled, cut-cell stabilization enabled with velocity penalty 0.1
(first-derivative jumps), pressure ghost penalty Enabled with penalty 1,
`Use_cut_metadata_scale=false`, space-time momentum source file, Eigen
direct solves. Reinitialization disabled for the matrix (see Findings).

## Results (all `passed=True`, see `verify_*.json` and `refinement_table.txt`)

| metric | nx8 dt=0.02 | nx16 dt=0.02 | nx24 dt=0.02 | nx16 dt=0.01 |
|---|---|---|---|---|
| velocity rel. L2 | 1.224e-1 | 8.838e-2 | 6.758e-2 | 8.162e-2 |
| pressure rel. RMS | 1.159e-3 | 3.370e-4 | 1.765e-4 | 3.070e-4 |
| pressure offset (Pa) | 0.18 | -0.36 | 0.32 | -0.17 |
| interface height L2 | 1.853e-4 | 1.035e-4 | 1.224e-4 | 9.277e-5 |
| interface pressure RMS (Pa) | 21.70 | 5.53 | 2.47 | 5.46 |
| area rel. error | 2.4e-6 | 9.6e-5 | 1.8e-5 | 5.1e-5 |

Convergence: interface-pressure RMS converges at second order
(orders 1.97 and 1.99 across nx8→nx16→nx24) — the natural traction-free
dynamic condition is working as designed. Pressure field error converges at
~O(h^1.7). The pressure gauge offset is O(0.2-0.4) Pa against a ~4900 Pa
hydrostatic scale (the 2026-05-18 official record showed -319 Pa with a
static, t=0-frozen ManufacturedSource — that record is invalid as an MMS).

## Findings

1. The dominant velocity error was NOT a free-surface defect — ROOT CAUSE
   FOUND AND FIXED (uncommitted, same session): the bottom-wall u_x sawtooth
   (max error ~0.03 = 30% of U0; h-, dt-, rho_inf-, amplitude-independent;
   growth proportional to U(t)-U(0)) was caused by
   `NewtonSolver::syncHistoryState()` calling `constraints.distribute()` on
   EVERY TimeHistory state with the current stage inhomogeneity. For
   time-dependent Dirichlet data this stamps g(t_stage) over the committed
   wall values of u^n AND over the injected first-order generalized-alpha
   rate slot (a velocity value written into a rate slot), so the residual's
   wall acceleration is c0*g(t) instead of g_dot(t). The consistent-mass
   coupling integrates the missing g_dot into the wall-adjacent cells: the
   measured mid-row coefficient 0.105*DeltaU matches the Q2 wall mass-ratio
   prediction exactly. Fixes (both env-revertible):
   - `NewtonSolver.cpp`: distribute constraint values into the CURRENT vector
     only (history already satisfies its own-time constraints); legacy via
     `SVMP_DISTRIBUTE_CONSTRAINTS_INTO_HISTORY=1`.
   - `TimeLoop.cpp`: keep the constraint-consistent finite-difference rates
     at constrained DOFs after the generalized-alpha rate update instead of
     zeroing them with distributeHomogeneous; legacy via
     `SVMP_ZERO_CONSTRAINED_RATES=1`.
   Validation with the fixes: nx8 velocity rel. L2 0.1224 -> 0.0319 and
   sawtooth 0.01805 -> 0.00449 (0.00350 at dt=0.01); nx16 velocity rel. L2
   0.0884 -> 0.0224, max abs 0.0323 -> 0.0083, interface height L2
   1.04e-4 -> 7.6e-5; all gates still pass. Regression: timestepping
   155/155, physics 259 pass + 3 MPI-skips, application 100/100.
2. Reinitialization DOF-binding corruption (root cause of the historical
   post-reinit failures): the projection repair's entity-aware traversal
   translated FE edge ids through the mesh's edge tables; on real multi-cell
   2D Q2 meshes the two numberings disagree, so edge DOFs received distances
   measured at the wrong coordinates. Near-interface edge DOFs (|phi| as
   small as 5e-4) were overwritten with values up to 0.45, which the
   subsequent solve could not absorb: with trial-side geometry refresh Newton
   livelocked (||r|| limit cycle 1.0022/1.54 with line_search_reject); with
   the line-search freeze it accepted a loose state and the next step
   diverged to ~9e7. Fixed by binding DOFs through the isoparametric nodal
   pairing getCellDofs(cell)[i] <-> i-th cell node (the convention the cut
   backends and field projections already use), with the entity walk retained
   for sub/super-parametric fields. Single-cell unit fixtures cannot expose
   the mismatch; the nx8 run below is the regression evidence.
3. Verification with both fixes: nx8 dt=0.02 WITH reinitialization cadence 10
   completes all 25 steps (2 reinit events, 0 non-converged solves), passes
   the verifier, and matches the no-reinit metrics to ~6 digits (velocity
   rel. L2 0.1223990 vs 0.1223991, interface height L2 1.840e-4 vs 1.853e-4)
   — band-preserving reinitialization is numerically benign. The nx16
   dt=0.02 no-reinit configuration rerun under the line-search freeze matches
   this record to ~6 digits (velocity 0.0883826 vs 0.0883827).
4. The `mms_traveling_interface_2d_nx2_*/nx3_*/nx4_*_20260518` record
   directories contain stale copies: nx2 and nx3 hold byte-identical
   `verify_result_022.json` files from a single run, and the nx3 log shows
   that run aborted on a Trilinos misconfiguration. They should not be cited
   as refinement evidence.

## Reproduce

```bash
python3 generate_case.py --nx 16 --ny 16 --time-step 0.02 --time-steps 25 \
  --final-time 0.5 --disable-reinitialization \
  --cut-cell-pressure-stabilization-policy Enabled
/path/to/svmultiphysics solver.xml
python3 verify_expected_results.py
```

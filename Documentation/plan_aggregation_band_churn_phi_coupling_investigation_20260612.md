# Investigation plan: aggregation fine-mesh velocity-error gap (band churn vs φ-coupling)

Status: COMPLETE INCLUDING FOLLOW-UP — section-10 follow-up executed
2026-06-12 (third session). R1–R6 all ran to verdicts, C1–C6 all closed with
code + tests, both outstanding gates ran. Headlines: (i) C6's fixture caught
a REAL emission bug (stale slave-dof span → components ≥1 mis-slaved on
component-major layouts) — fixed; (ii) the recorded penalty table's
provenance was resolved to the surviving /tmp/svmp_bin_final snapshot and
reproduced to 4–5 digits; (iii) in the committed GP-off config the
aggregation gap at nx16 is REAL (1.9–3.1×) but the section-9.5 "floor" is an
nx16-local hump, NOT an h-floor — aggregation h-converges at orders
1.7→3+ past nx16 while the penalty stack FAILS outright at nx8; (iv) two
mitigation prototypes returned clean negatives (constraint cycles /
neutral), so no constraint-side mitigation is warranted; (v) the d18 gate
exposed that no-root cut islands are ROUTINE in breaking surfaces — the
fail-closed policy was refined to homogeneous island pins. Full record in
section 11. Findings of the second session are in section 9; the follow-up
plan it scoped is section 10; the original plan text is preserved unchanged
below them.

Original status: NOT STARTED — handoff document. Written 2026-06-12 at the
end of the small-cut-aggregation campaign (see
`plan_ghost_penalty_eigen_calibration_20260611.md` for the full history).
All code referenced below is uncommitted working-tree state.

## 1. The question

Small-cut aggregation (AgFEM) replaced the velocity ghost penalty everywhere
(the penalty is deleted from the codebase). All official case verifiers pass,
and the d18/d38 SPHERIC physics gates are equal-or-better than the penalty
stack (d18 front error 70× better). One accuracy gap remains, on the MMS
traveling-interface case (Q2/Q1, 2D, moving band):

| velocity_relative_l2_error | penalty (deleted) | aggregation (current) |
|----------------------------|-------------------|-----------------------|
| nx8,  dt 0.02              | 0.008970          | 0.008573 (4% better)  |
| nx16, dt 0.02              | 0.003224          | 0.004810 (1.49× worse)|
| nx16, dt 0.01              | 0.001920          | 0.003893 (2.03× worse)|

Convergence orders computed from the table:
- h-order (nx8→nx16 at dt 0.02): penalty **1.48**, aggregation **0.83**.
- dt-order (dt 0.02→0.01 at nx16): penalty **0.75**, aggregation **0.31**.

The gap grows under BOTH h- and dt-refinement → an error source that behaves
like a floor / low-order contamination, present only with aggregation.

## 2. What is already ESTABLISHED (do not redo)

1. **Not extension order.** The "linear vs full-order extension" trade
   reported mid-campaign was a confound: the linear gate never engaged
   (`element_type()` returns the topology type, so `linearElementType()` was
   identity), and full-order extension was active in every passing run.
   Verified by bit-identical nx8 results with the gate forced both ways.
   `SVMP_AGGREGATION_LINEAR_EXTENSION=1` now genuinely engages the linear
   corner sub-basis (see `SmallCutAggregationConstraint.cpp`, step 5).
2. **Not Jacobian consistency at step 0.** Single-slave and full-band
   Jacobian checks (`SVMP_FE_JACOBIAN_CHECK=1
   SVMP_FE_JACOBIAN_CHECK_COMPONENTS="Velocity,Pressure"`) are machine-clean
   (~2e-5 velocity, 7e-10 pressure) after the cut-volume sparsity fix.
3. **Not dropped matrix writes.** `SVMP_EIGEN_COUNT_DROPPED=1` shows 0 drops
   across all final runs; `SVMP_EIGEN_STRICT_PATTERN=1` throws on any.
4. **Not φ blow-up in MMS.** The d18/d38 no-reinit φ explosion (±1e5) does
   not occur in MMS runs (φ stays bounded; the manufactured advection is
   smooth). The d18/d38 cases now run reinit ON regardless.
5. **Mass conservation is exact-ish**: d18/d38 wet volume matched penalties
   to 6 digits; MMS verifiers pass volume checks.

## 3. Hypotheses

### H1 — Band-constraint churn (time-discretization interaction)
The slave set is reclassified at every cut-context refresh (MMS: after every
accepted Newton iterate of the monolithic solve; d18: ~1 sparsity refresh per
step). Each transition does one of:
- dof ENTERS slavery: its FE value is overwritten by the extension value via
  `constraints.distribute()` — an O(h^{p+1}) projection jump, but applied at
  step frequency.
- dof LEAVES slavery: it resumes as a free unknown whose TIME HISTORY
  (uPrev, and critically the gen-α rate slots uDot/uPrev2) was written under
  the constraint. The wall-rate saga (see memory: free-surface-cutfem-fixes)
  proved gen-α is extremely sensitive to constrained-rate handling; a stale
  or extension-inconsistent rate on release injects a first-order-in-dt
  error pulse per transition.
Accumulated over n_steps transitions along the moving band, this plausibly
produces exactly the observed signature: degraded h-order AND degraded
dt-order, worse at finer dt (more transitions per physical band crossing).

### H2 — φ-coupling through the monolithic solve
op='equations' solves φ together with velocity/pressure. The cut context
(and hence quadrature domains, aggregation classification, active-side pins)
rebuilds per accepted iterate from the CURRENT φ. Velocity error near the
band could be driven by φ-transport error (SUPG'd advection, no reinit in
the MMS config) through the moving integration domains, independent of the
constraint machinery. The known-benign explicit φ/velocity Jacobian coupling
(advection velocity treated as data) could also matter at order level.

### H3 — Local resolution loss in the slaved band (constant, not order)
The slaved region inherits the root cell's polynomial: locally a one-cell
coarsening. Standard AgFEM analysis says this costs a constant, not the
order — listed only so the future agent measures rather than assumes.

## 4. Designed experiments (ordered; each discriminates)

Baseline reproduction (run from the case dir, serial):
```
cd /tmp/svmp_mms_agg_nx8   # or recreate from tests/cases/.../mms_traveling_interface_2d generator
OMP_NUM_THREADS=1 timeout 3600 <binary> solver.xml > solver_run.log 2>&1
python3 verify_expected_results.py   # velocity_relative_l2_error, passed
```
Binary: build from working tree (`cmake --build build/svMultiPhysics-build
--target svmultiphysics -j8`), snapshot before long runs (in-place rebuilds
kill running solvers). /tmp run dirs are EPHEMERAL — recreate via
`generate_case.py` in `tests/cases/fluid/open_vessel_free_surface/
unfitted_level_set/mms_traveling_interface_2d/` (nx, dt args; the deprecated
velocity-penalty key is no longer emitted). Penalty baselines CANNOT be
rerun on the current binary (term deleted) — the numbers above are the
recorded baselines; if a fresh penalty run is ever needed, check out a
pre-deletion tree state.

- **E1 — Error localization (first, cheap).** The verifier already reports
  `velocity_relative_l2_error` and `bulk_velocity_relative_l2_error` (read
  `verify_expected_results.py` for the exact masks — bulk excludes the
  near-band region). Extend it (or post-process result_*.vtu with the vtk
  python module) to report error restricted to (a) cells that were EVER in
  the slaved band during the run, (b) everything else. If the gap lives in
  (a) → H1/H3; if spread over (b) → H2.
  Band membership: dump slave vertex sets per refresh with
  `SVMP_AGGREGATION_DUMP=1` (one line per slave incl. xyz) and accumulate.
- **E2 — Churn quantification.** Add per-apply counters to
  `SmallCutAggregationConstraint::apply` (entered/left vs previous slave
  set — keep a static prev-set keyed by field). Correlate per-step error
  (rel-L2 from each result_NNN.vtu against the manufactured solution at that
  time) with transition counts. Strong correlation of error increments with
  release events → H1.
- **E3 — Static-band control.** Generate an MMS variant whose interface
  does NOT move (check `generate_case.py` for a zero-advection /
  stationary-interface option; otherwise set the manufactured interface
  velocity to 0 and regenerate sources). Static band → zero churn. If
  h-order recovers to ~1.5 with aggregation on a static band → H1 confirmed,
  H2 largely excluded (φ still solved monolithically).
- **E4 — Release-consistency prototype (the H1 fix).** On slave release,
  reconstruct the dof's rate history consistently: set uDot from the
  extension's finite-difference in time (or re-distribute the PREVIOUS
  step's constraint into uPrev/uDot before releasing). Plumbing hints:
  the distribute-into-history policy from the wall-rate work
  (`SVMP_DISTRIBUTE_CONSTRAINTS_INTO_HISTORY`, NewtonSolver ~syncHistoryState)
  and the constrained-rate gates in TimeLoop (`SVMP_ZERO_CONSTRAINED_RATES`
  legacy paths). A minimal prototype: in rebuildConstraintState (or the app
  refresh), for dofs leaving the slave set, copy u(slave) and u_prev(slave)
  from the pre-release distributed values and set uDot via the gen-α
  consistent rate. Measure nx16/dt01 error.
- **E5 — Reinit-ON MMS.** `Enable_reinitialization=true` (projection,
  cadence 10, band-preserving default) on nx8/nx16 aggregation runs. The
  2026-06-10 record shows penalty+reinit matched no-reinit to 6 digits, so
  any aggregation-run change isolates an aggregation×φ interaction (H2).
- **E6 — dt-only refinement at nx8.** dt 0.02 → 0.01 → 0.005 with
  aggregation. If error saturates (no dt-convergence) → floor is
  churn/spatial (H1/H3); compare against penalty-era dt behavior from the
  recorded baselines.
- **E7 — Linear vs full-order A/B (close the loop).**
  `SVMP_AGGREGATION_LINEAR_EXTENSION=1` on nx8+nx16: now that the knob is
  real, quantify the extension-order contribution. Expectation from the
  confound discovery: small; if LARGE, revisit H3.
- **E8 — Late-step Jacobian check.** Run the nx16 case with
  `SVMP_FE_JACOBIAN_CHECK=1` at a LATE step (the checker fires at the first
  Newton iterate of each step — capture e.g. step 20 by short-running) to
  rule out consistency drift as the band geometry gets less aligned.

## 5. Additional code-review concerns to close during this investigation

The follow-up code review of `SmallCutAggregationConstraint` found no obvious
contradiction in the intended P1 AgFEM algorithm: classify retained cut/full
cells, find unsupported cut-band nodes, BFS to a full-active root, and emit
master-bearing affine lines. The concerns below are the places where the
implementation can still be correct for the current smoke gates while leaving
unverified or fail-open behavior in the broader infrastructure.

### C1 — Under-aggregation currently fails open

`SmallCutAggregationConstraint::apply` increments diagnostics for
`vertices_without_root`, `inversion_failures`, and `non_field_nodes`, then
continues. The final diagnostic line reports the counts, but a run can proceed
with candidate vertices left unconstrained. With the velocity ghost penalty
deleted, any missed small-cut DOF is no longer backed by the old stabilization.

Verification needed:
- Parse every MMS/d18/d38 aggregation run log and require
  `aggregated_vertices + non_field_nodes + excluded_dirichlet_vertices`
  to account for the intended candidate set; unexpected
  `vertices_without_root` or `inversion_failures` must be treated as a failed
  gate, not as informational output.
- Add a synthetic cut-context unit test with an intentionally isolated cut
  island and confirm the production policy: either throw/fail closed, or emit
  a documented fatal diagnostic when a candidate has no full-active root.
- Add an invariant after line emission that a newly added master-bearing
  constraint line has at least one master entry; empty lines would silently act
  like homogeneous pins.

### C2 — Higher-order and linear-extension behavior is not independently proven

The investigation note above records that the old linear/full-order A/B was
confounded by `element_type()` returning topology. The current implementation
still needs a direct proof that `SVMP_AGGREGATION_LINEAR_EXTENSION=1` changes
the basis for common Q2 construction paths such as `H1Space(Quad4, order=2)`,
where topology and polynomial order are stored separately. The public comments
also still describe the feature as order-1/vertex-only even though the
implementation now attempts midside-node handling.

Verification needed:
- Add a focused unit test for Q2 scalar and Product fields that constructs a
  small cut band, enables `SVMP_AGGREGATION_LINEAR_EXTENSION=1`, and asserts
  the emitted master count and weights are the linear corner sub-basis rather
  than the full Q2 basis.
- Repeat the same fixture with the default full-order path and assert midside
  slave DOFs and midside master entries are present when expected.
- Update the comments only after the tests settle the actual support contract:
  either document P1-only support, or document the verified Q2 behavior and its
  accuracy trade.

### C3 — Tiny generated cut-volume pruning may hide the slivers aggregation is meant to fix

Generated volume rules below `CutIntegrationContext::minGeneratedCutVolumeFraction`
are pruned before aggregation classifies cells. That is useful for quadrature
noise, but it can also remove the smallest active slivers from the very
metadata used to decide which DOFs need aggregation.

Verification needed:
- Run MMS nx8/nx16 with the default pruning threshold and with
  `SVMP_MIN_GENERATED_CUT_VOLUME_FRACTION` set much smaller; compare
  pruned-rule counts, aggregation candidate counts, and velocity error.
- Add a cut-context unit test with one below-threshold active sliver adjacent to
  a full-active cell and verify the intended policy: either the sliver is
  deliberately inactive/pinned by another constraint path, or aggregation still
  sees enough metadata to constrain its unsupported DOFs.
- Include pruned generated-volume count and measure in the aggregation
  diagnostic summary when the marker matches the free-surface active domain.

### C4 — Wall/strong-BC precedence is plausible but needs direct wall tests

The NS registration excludes velocity Dirichlet boundary markers from velocity
aggregation and excludes pressure gauge vertices from pressure aggregation.
`AffineConstraints::addDirichlet` also overrides master-bearing lines, which is
the right global precedence. What is missing is a small wall fixture proving
that boundary corners and higher-order face nodes are all excluded and that no
wall DOF remains overly wet because aggregation pulled it from an interior root.

Verification needed:
- Add a two-dimensional wall-contact cut fixture with a moving interface along
  a side wall. Assert that wall Dirichlet DOFs, including Q2 midside wall nodes,
  are never aggregation slaves.
- Add a paired test where a strong Dirichlet line is installed after
  aggregation and verify the master-bearing line is replaced, not merged or
  rejected.
- For pressure, add a fixture with a gauge pin inside the candidate band and
  confirm the pressure pin wins over aggregation and still removes the null
  mode.

### C5 — Active-side retention and mid-solve rebuilds need log-level gates

Active generated-volume retention defaults to active-only; inactive rules are
retained only when velocity extension requests both sides. Aggregation asks the
cut context for both sides, but active-only retention means inactive-side data
may legitimately be absent. The dependency declaration requests a structural
rebuild on active-configuration/mesh-field-value changes, but correctness still
depends on the broader FESystem taking that structural path during monolithic
mid-solve cut refreshes.

Verification needed:
- In MMS, log each constraint refresh with the active-side volume-rule counts,
  slave count, entered/left counts, and constraint-structure signature. Confirm
  every cut-context content change that changes the slave set triggers a
  structural constraint rebuild and sparsity re-augmentation before the next
  Jacobian assembly.
- Add an active-only vs active-and-inactive retention A/B for one MMS case. The
  inactive-side counts should explain any diagnostic differences; velocity
  error should not depend on retaining inactive metadata unless velocity
  extension requires it.
- Run at least one MPI/distributed smoke if this path is intended beyond serial
  gates; validate that master-bearing constraints are synchronized consistently
  across shared/ghost DOFs.

### C6 — Direct algorithm tests are missing

Current tests cover related infrastructure: master-bearing rows through the
transient Newton path and Dirichlet override precedence. They do not directly
test `SmallCutAggregationConstraint` candidate selection, root choice, emitted
weights, wall exclusion, Q2 layout, or failure diagnostics.

Verification needed:
- Create a dedicated `SmallCutAggregationConstraint` unit fixture with a tiny
  structured mesh and hand-built generated cut-volume rules.
- Cover at least: one P1 cut vertex with known root weights, one no-root
  candidate, one wall-excluded candidate, one gauge-excluded pressure vertex,
  one Q2 midside candidate, and one Product velocity field with the actual
  cell-dof layout.
- Make the test assert the emitted `AffineConstraints` contents, not just that
  setup converges.

## 6. Key code pointers (working tree, 2026-06-12)

- `Code/Source/solver/FE/Constraints/SmallCutAggregationConstraint.cpp` —
  classification (steps 1–4), root BFS + emission (step 5), cell-local dof
  resolution with empirical layout detection + cross-validation, geometric
  boundary-face exclusion, `excluded_vertices` (gauge pins),
  `SVMP_AGGREGATION_DUMP` / `_MAX_LINES` / `_LINEAR_EXTENSION` env knobs.
- `Code/Source/solver/FE/Systems/SystemSetup.cpp` —
  `computeConstraintStructureSignature`,
  `refreshSparsityForConstraintStructureChange` (re-augmentation on band
  change), cut-volume terms in the sparsity pair collection (the campaign's
  root-cause fix).
- `Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp` —
  `maybeReallocateJacobianForSparsity` (solveStep entry + the three
  Jacobian-assembly lambdas), constrained-dof handling, FD checker.
- `Code/Source/solver/FE/TimeStepping/TimeLoop.cpp` —
  `ensure_workspace_matches_sparsity`, gen-α constrained-rate gates
  (`SVMP_ZERO_CONSTRAINED_RATES`), history distribute paths.
- `Code/Source/solver/FE/Constraints/AffineConstraints.cpp` —
  `addDirichlet` overrides master-bearing lines (strong-BC precedence).
- `Code/Source/solver/FE/Backends/Eigen/EigenMatrix.cpp` —
  `reinitFromPattern` (in-place), out-of-pattern drop diagnostics.
- `Code/Source/solver/Physics/Formulations/NavierStokes/
  IncompressibleNavierStokesVMSModule.cpp` — aggregation registration
  (~`small_cut_aggregation` block; default TRUE in the .h), pressure GP
  (retained), velocity GP fully removed.

## 7. Success criteria

- Velocity rel-L2 h-order (nx8→nx16, dt small enough to not floor) ≥ ~1.4
  with aggregation; nx16/dt0.01 error ≤ ~0.0025 (penalty-era ballpark).
- No robustness regression: MMS 25/25 and 50/50 step completion, verifiers
  pass, d18/d38 5-step smokes converged with 0 drops; ideally re-run one
  312-step gate after any fix that touches constraint/history handling.
- Unit suites stay green: FE constraints 226, sparsity 650, timestepping
  158, assembly 757, backends 184; physics MovingDomain+LegacyBCs 121;
  application OpenVessel 12. (One pre-existing unrelated failure exists:
  `CanonicalWorkflow.SameSpace_SameName_DifferentBindings`.)

## 8. Pitfalls learned this campaign (read before touching anything)

- Verify every dof-identity claim against the EntityDofMap or a cell-local
  pairing cross-check; `getCellDofs` is component-major here, and
  `getVertexDofs` is empty for Q2 midside nodes.
- Don't read verifier outputs from a run dir while a rerun is writing into
  it (a mid-run read produced a wrong A/B number once).
- The outer superbuild does not rebuild inner sources — build
  `build/svMultiPhysics-build` directly; snapshot binaries to /tmp before
  rebuilding if long runs are in flight.
- Cut refreshes fire MID-SOLVE in the monolithic system (after every
  accepted Newton iterate) — anything keyed "per step" is wrong for MMS.
- `pressure_gauge.csv` is an input (gauge pins), not an output.
- The first-32-samples drop log can hide later, larger drops — use the
  counters, not the samples, for conclusions.

## 9. FINDINGS (2026-06-12 execution session)

### 9.0 Config provenance — read this before comparing numbers

The recorded table in section 1 was measured with the **pressure ghost
penalty ENABLED at gamma_p=100** (`--cut-cell-pressure-gradient-penalty 100
--cut-cell-pressure-stabilization-policy Enabled`), the calibration-era
comparison family. The COMMITTED case deck
(`mms_traveling_interface_2d/solver.xml`) carries policy
`DisabledForRefreshedFrozenHighOrder`, i.e. the pressure GP is INERT there
("GP-off" below). All three recorded aggregation baselines were reproduced
BIT-EXACTLY on the session binary in the GP-on family
(0.008573483136327782 / 0.004809517743853168 / 0.003892608119439676), so
every comparison below is apples-to-apples. Notably, GP-off is uniformly
MORE accurate than GP-on for aggregation (nx16: 0.00287 vs 0.00481 pre-fix)
— a large share of the "1.49x worse" headline was the gamma_p=100 term
itself, and pre-fix GP-off aggregation already beat the penalty-era table
at nx8 (0.00606 vs 0.00897) and nx16 (0.00287 vs 0.00322).

### 9.1 E1/E2 — error localization and churn correlation (H1 confirmed)

- The velocity error is a near-band boundary layer: at nx16 dt.02 the shell
  rms decays 0.00227 / 0.00142 / 0.00089 / 0.00058 for [0,1h) / [1,2h) /
  [2,3h) / [3,4h); the fixed-region error (phi_ex < -0.25, h-independent
  region) converges at h-order ~1.6, while shells at fixed h-multiples
  stagnate (order 0.13-0.45). The verifier's legacy wet mask (phi < -2h)
  tracks the band, so its "h-order 0.83" mixes clean far-field convergence
  with the stagnating near-band layer it progressively includes.
- Slave-set churn (SVMP_AGGREGATION_DUMP per-apply sets): release/re-slave
  oscillations of 4-12 velocity DOFs coincide EXACTLY with per-step error
  jumps (nx16 dt.02: bulk rms 0.000225 -> 0.000425 at the step-5 churn;
  pre-fix late event at steps 19-21 produced the 0.000400 -> 0.000485 final
  jump; nx16 dt.01: double oscillation at steps 15-16 produced the t=0.17
  spike, which then DECAYS under gen-alpha damping).
- Mechanism (verified by the fix): the end-of-step generalized-alpha rate
  update uDot = (u^{n+1}-u^n)/(gamma*dt) - c*uDot^n is a raw finite
  difference; a DOF whose constraint status changed during the step sees the
  free-vs-extension value jump scaled by 1/(gamma*dt) — a rate pulse the
  alpha_m mass term broadcasts into neighboring momentum rows on the next
  stage solve. Within-solve oscillations that restore the slave set before
  acceptance leave NO lasting error (verified bit-identical trajectories
  through churn-active steps).

### 9.2 E3 — static-band control (H3 excluded, motion is the driver)

Amplitude=0 variant (flat static interface at y=0.5, U(t)=U0*cos(wt) still
time-varying, aggregation active on the static cut row, zero churn):
velocity rel-L2 = 0.000377 (nx8) / 0.000530 (nx16), i.e. 16-23x below the
moving-band errors, dt-floored at these magnitudes. Static cut-cell
aggregation (including the slaved-band resolution loss) costs at most
~5e-4 relative — H3 is not the gap.

### 9.3 E4 — the fix (master-bearing constraint-state distribution)

New `AffineConstraints::{hasMasterBearingLines, distributeMasterBearing,
distributeMasterBearingHomogeneous}`: distribute ONLY lines that carry
master entries (true MPCs). Dirichlet lines are untouched, so the wall-rate
contract (g_dot carried by FD rates, see SVMP_ZERO_CONSTRAINED_RATES
comments) is preserved exactly.

Applied at two sites, default ON, opt-out `SVMP_NO_MPC_STATE_DISTRIBUTE=1`:
1. `TimeLoop` accepted-step block (after the on_step_accepted callback's
   cut refresh + constraint rebuild): distribute u, uPrev; homogeneous into
   uDot (+uDDot when present). Heals cross-boundary transitions: entering
   DOFs get extension-consistent (value, rate) pairs; released DOFs resume
   from a consistent extension trajectory.
2. `NewtonSolver::syncHistoryState` (runs at every Newton iterate): same
   distribution into uPrevK(k) and uDot/uDDot. Generalizes (1) to DOFs that
   enter mid-solve and STAY slaved (fresh entries). For the oscillating
   churn pattern of this MMS it is a provable no-op (aggregation weights are
   purely geometric — node position inverted in the root cell — so a
   re-slaved DOF with the same root reproduces its old extension values);
   measured ±1% trajectory noise at nx16.

Effect (GP-on family): nx16 dt.02 0.004810 -> 0.004256 (-12%), the late
churn spike is eliminated from the per-step series; nx8 and nx16 dt.01
essentially unchanged (their final errors were already dominated by the
secular component below; dt.01's churn spikes had decayed by t=0.5 in both
runs). GP-off family: nx16 dt.01 0.003130 -> 0.003037; others neutral.
Sanity: SVMP_NO_MPC_STATE_DISTRIBUTE=1 reproduces the pre-fix binary
BIT-EXACTLY; d18 5-step smoke 5/5 converged with both fix variants; suites
green (constraints 229 incl. 3 new API tests, timestepping 158, sparsity
650, assembly 757, backends 184, physics 121, OpenVessel 12).

### 9.4 E5/E7/E8 — the negative results that close the hypothesis space

- E5 (reinit ON, projection cadence 10): nx8 0.008589 vs 0.008584 no-reinit
  (neutral to 5 digits); nx16 0.004264 vs 0.004309 (~1%, noise class). The
  aggregation x phi-maintenance interaction is nil; H2's reinit channel is
  excluded.
- E7 (linear vs full-order extension): the
  SVMP_AGGREGATION_LINEAR_EXTENSION knob was STILL dead in the handoff tree
  (`element_type()` reports topology — Quad4 — so `linearElementType()` was
  identity; section 2 item 1 of this plan was wrong about the fix being in
  place). Gate re-implemented on basis-size comparison (engage when the
  corner basis is strictly smaller than the field basis). With it genuinely
  engaged: nx16 GP-on 0.0042562 (linear) vs 0.0043088 (full) — ~1%,
  extension order is irrelevant at these resolutions.
- E8 (late-step Jacobian checks, steps 0-20, velocity+pressure filter):
  rel = 1.4e-5..1.8e-5 flat across ALL steps — no consistency drift as the
  band de-aligns. (Geometry tangents remain refreshed-frozen by contract.)
- phi transport error (band rms vs exact) grows secularly to 2.4e-4 at nx16
  dt.02 but is 35% SMALLER at dt.01 while the velocity error is NOT — the
  secular velocity component is not phi-error-proportional (H2 weakened
  beyond the reinit exclusion).

### 9.5 The residual: a churn-independent secular band-local component

After the fix, the remaining error grows smoothly (~2e-5 bulk-rms/step at
nx16, both GP families, both dt's), is band-localized (shell profile of
9.1), and behaves like a weakly-h-dependent FLOOR of ~0.003 relative: nx8
is still dt-dominated and converges through it (E6 fix-on dt series
0.008584 / 0.005652 / 0.003427 at dt .02/.01/.005, orders 0.60 and 0.72 —
no saturation yet, same ~0.6-0.75 class as the penalty-era dt behavior),
while nx16 reaches the floor already at dt=.02 (GP-on dt-order 0.16;
GP-off ANTI-converges, 0.00281 at dt.02 vs 0.00304 at dt.01). The floor
caps the legacy-mask h-order at ~1.0-1.1 (fix-on: GP-on 1.01, GP-off
1.12) and the h-order at dt.01 at ~0.57. It is NOT: transition pulses (fixed),
extension order (E7), Jacobian inconsistency (E8), phi maintenance (E5),
static cut-cell resolution (E3), dropped writes (0 drops everywhere), or
the pressure GP (present in both families, though gamma_p=100 roughly
doubles it). Leading suspect: the velocity response to refreshed-frozen
moving-cut quadrature consistency error in the NON-aggregated band DOFs —
modes the deleted velocity gradient-jump penalty used to damp (which is
how the penalty stack reached 0.00192 at nx16 dt.01 while aggregation
reaches ~0.0030-0.0038). Candidate follow-ups: per-step (not per-iterate)
cut-context refresh cadence A/B; band-local gradient smoothing through the
extension operator (extend one layer deeper); space-time/GCL-consistent
cut-volume treatment.

### 9.6 Success criteria (section 7) verdict

- Robustness: MET — 25/25 and 50/50 completions everywhere, verifiers PASS
  on every run (the A=0 control fails only its degenerate amplitude check,
  as expected for a zero-amplitude interface), d18 smokes 5/5 with 0 drops,
  all unit suites green.
- nx16/dt0.01 <= ~0.0025 and legacy-mask h-order >= ~1.4: NOT MET — best is
  GP-off+fix 0.0030 (dt.01) / h-order ~1.12. The churn share of the gap is
  fixed; the secular share persists and is now characterized (9.5). In the
  committed-deck (GP-off) configuration, aggregation+fix beats the recorded
  penalty-era errors at nx8 (0.00611 vs 0.00897) and nx16 (0.00281 vs
  0.00322) and trails only at dt.01 (0.0030 vs 0.00192), with the caveat
  that the penalty numbers were measured in the GP-on family where
  aggregation does worse.

Run artifacts: /tmp/inv_* (pre-fix: inv_off_* GP-on, inv_base_* GP-off;
fix v1: inv_fix_*; fix v2: inv_fix2_*; GP-off+fix: inv_fixgpoff_*;
controls: inv_e3_static_*, inv_fixoff_nx8, inv_fix_nx16_linear,
inv_fix_nx16_jaccheck, inv_fix_nx*_reinit, inv_fix_nx8dt*). Analysis
tooling: /tmp/svmp_e1_analysis.py (per-step shells + churn parsing),
/tmp/svmp_table.py (consolidated table), /tmp/svmp_run_case.sh (runner).

## 10. Follow-up investigation plan (residual floor + hardening)

### 10.0 Status of the section-5 review concerns (C1–C6) after this session
### [ALL CLOSED in the 2026-06-12 follow-up session — see section 11.7]

- **C1 (under-aggregation fails open): CLOSED.** Three-way policy
  implemented: no-root candidates (isolated cut islands — a ROUTINE
  geometric condition; the 312-step d18 gate sees up to 48 per refresh once
  the surface fragments) are pinned homogeneously, never fatal;
  inversion/empty-line failures (machinery) throw;
  SVMP_AGGREGATION_ALLOW_UNAGGREGATED=1 restores legacy fail-open. Plus the
  no-empty-master-line invariant (entries collected before addLine). Unit
  tests: island-pin, fail-open env, debug-cap exemption.
- **C2 (linear-extension proof): CLOSED.** Q2 unit tests assert exact
  master sets/weights for both extension orders (full Q2: midside masters
  with weights ±3/±8; linear knob: corner sub-basis 1.5/−0.5 etc., slave set
  unchanged); env reads made per-call so in-process A/Bs work; header
  contract updated. BONUS: a sub-parametric guard now rejects fields whose
  nodal layout exceeds the mesh nodes (e.g. H1Space(Quad4, order=2) on
  4-node quads) — previously a silent partial-polynomial-extension hole.
- **C3 (cut-volume pruning vs aggregation metadata): CLOSED.** System A/B
  (threshold 1e-30 vs default 1e-8) identical to 9 digits at nx16 (zero
  rules pruned either way; nx8 has exactly one 5.5e-11 sliver, now visible);
  pruned count+measure added to the aggregation diagnostic line; unit test
  pins the policy: a below-threshold sliver's unsupported-but-wet vertices
  fall to the level-set inactive-pin path, real cut candidates still
  aggregate.
- **C4 (wall/strong-BC precedence fixtures): CLOSED** (absorbed into the
  C6 fixture). Tests: P1 wall vertices never slaved; Q2 midside WALL nodes
  excluded via the reference-coordinate discriminator; gauge-excluded
  pressure vertex keeps its pin; strong Dirichlet installed AFTER
  aggregation replaces the master-bearing line end-to-end.
- **C5 (retention/structural-rebuild gates): CLOSED.** Per-refresh
  slave-set churn counters (entered/left, keyed by system+field) and the
  constraint-structure signature in the sparsity-refresh log; retention A/B
  knob (SVMP_CUT_RETENTION_FORCE) — forcing active_only on the MMS deck
  fails CLOSED at startup ("Generated cut-volume consumer has no retained
  quadrature rules", the velocity-extension consumer declares its need), so
  retention follows consumers by contract; 2-rank MPI smoke: aggregation
  emitted rank-identical constraints (12/12 candidates, 24 slave dofs both
  ranks, cross-rank consistency check silent) — the failure is the Eigen
  DIRECT solver (serial-only), i.e. the path is backend-gated serial, not
  constraint-gated.
- **C6 (dedicated SmallCutAggregationConstraint unit fixture): CLOSED.**
  test_SmallCutAggregationConstraint.cpp, 14 tests: P1 candidate selection +
  exact extrapolation weights (2/−1 rows), BFS-through-cut-chain root
  choice (3/−2 at distance 2), Product 2-component slaving (BOTH components,
  same geometric weights), island pin, fail-open env, max-lines cap, P1+Q2
  wall exclusion, gauge pin, Dirichlet override, Q2 full-order midside
  slaves+masters, linear-extension knob, sub-parametric rejection,
  pruned-sliver policy. The fixture caught a REAL production bug on its
  first run (section 11.2) — exactly the assert-the-emitted-contents value
  this item predicted.

### 10.1 R-experiments for the residual floor (ordered by information/cost)

- **R1 — Penalty baseline in the COMMITTED (GP-off) config. DO FIRST;
  decisive and cheap.** The recorded penalty numbers (0.00322/0.00192)
  are GP-on family; the committed deck is GP-off, where aggregation is ~2x
  better. Check out a pre-deletion tree state (last commit with the
  velocity GP), build, and run penalty nx8/nx16 dt.02 + nx16 dt.01 with
  the pressure-GP policy left at DisabledForRefreshedFrozenHighOrder. If
  penalty GP-off lands near ~0.003 at nx16/dt.01 too, there is NO
  aggregation-specific gap in the committed configuration — the floor is
  shared moving-cut machinery and the comparison that motivated this plan
  was a config artifact (then go to R3). If it lands near 0.0019, the gap
  is real and aggregation-specific (then R2/R6).
- **R2 — Carrier-mode decomposition.** Extend svmp_e1_analysis.py to split
  the near-band shells into (a) aggregated slaves, (b) free DOFs of cut
  cells, (c) DOFs of band-adjacent full-active cells; report u_x vs u_y
  and the PRESSURE error in the same shells. Locates the floor's carrier
  (free cut-cell DOFs would support the "GP damped these modes" theory).
- **R3 — Direct moving-quadrature consistency probe.** Per accepted step,
  integrate known monomials (1, x, y, x^2, xy, ...) over the active domain
  with the generated cut rules and compare against the analytic moving
  domain (exact in MMS). Correlate the per-step quadrature-consistency
  error with the velocity-error increments. Decouples quadrature accuracy
  from the solver entirely; cheap and offline.
- **R4 — Refresh-cadence A/B.** Env knob in ApplicationDriver to skip the
  per-accepted-iterate cut refreshes (keep before_physics_solve +
  accepted_step). Floor drops -> mid-solve domain motion under
  refreshed-frozen tangents is the injector. Unchanged -> the converged
  states themselves carry it (R3 territory). CAUTION: this changes
  Newton's fixed point (quadrature lags an iterate); re-run verifiers and
  the Jacobian checker.
- **R5 — Floor scaling laws.** nx16 GP-off dt.005 (does the
  anti-convergence turn around?); nx24/nx32 at small dt for the floor's
  h-exponent from the shell profile. Pins "weakly-h-dependent" down to a
  number worth modeling.
- **R6 — Mitigation prototypes (only after R1–R4 localize the source).**
  (a) aggregate-ALL-cut-cell-DOFs A/B (env knob: slave every cut-cell
  vertex, not just unsupported ones) — directly tests whether constraining
  the free cut DOFs removes the floor, as the deleted velocity GP
  effectively did; (b) extension one layer deeper; (c) GCL/space-time
  consistent cut-volume treatment (large — last resort, and only if R1
  shows a real aggregation-specific gap).

### 10.2 Outstanding gates (independent of the floor)

- **312-step d18 gate** with the MPC state-distribute fix: section 7 asks
  for one full gate re-run after any constraint/history-touching fix; this
  session ran only the 5-step smokes (5/5, 0 drops, both fix variants).
  Run the stock 312-step d18 deck on the fix binary and compare
  rmse/front-error against the recorded class (rmse 0.0191,
  front_error -0.0092).
- C1 fail-closed policy + log-parse gate wired into the run scripts.
- C2 unit test for the now-real linear-extension gate; C6 dedicated
  fixture (absorbs C4 wall cases); C3 pruning A/B; C5 retention A/B + one
  MPI smoke.

## 11. FOLLOW-UP EXECUTION RECORD (2026-06-12, third session)

All of section 10 was executed. No commits were made; everything below is
working-tree state. Binary snapshots for every experiment family are in
/tmp (svmp_bin_fix_s3..s3h ladder, see 11.10).

### 11.1 Penalty-table provenance resolved (R1 prerequisite)

The plan's "check out a pre-deletion tree state" recipe is WRONG: the
recorded penalty table was never reproducible from git. A clean worktree at
HEAD 7aaadf5a (plus six untracked-but-referenced files HEAD's own
CMakeLists needs: FE/Math/DenseTransformKernels.h, three
Basis/VectorBasis*_Runtime.cpp, Physics Ustruct pair +
IsochoricNeoHookeanPK1) builds and runs the deck-default penalty
(γ_v=0.1 ≡ eff_v 2.44) cleanly — and produces 0.0398 at nx16 dt.01
(50/50 converged, 6 iters/step): 20× the recorded number, because HEAD
(Jun 10 12:50) predates the uncommitted EVENING fixes of the
free-surface-cutfem session. The true source of the recorded table is the
surviving snapshot **/tmp/svmp_bin_final** (Jun 10 23:47, transient
calibration ×0.01 + evening fixes; velocity-GP code present, no
aggregation): with deck multipliers γ_v=10, γ_p=100/Enabled it reproduces
the recorded GP-on family to 4–5 digits:
0.0089312/0.0032238/0.0019219 vs recorded 0.008970/0.003224/0.001920.
All R1 penalty numbers below are from that binary. LESSON: archive gate
binaries; uncommitted-tree baselines die with the tree.

### 11.2 NEW BUG found and fixed: stale slave-dof span (the C6 catch)

`SmallCutAggregationConstraint` resolved the slave's dofs into a span
returned by `cell_node_dofs(...)`, which aliases shared scratch storage
that the MASTER lookups in the emission loop overwrite. On
ComponentMajor cell-dof layouts — what production uses (plan §8) — the
component-0 line was correct, but every component ≥1 line was emitted for
the dof of the ROOT CELL'S LAST NODE instead of the slave: a spurious MPC
on a well-posed interior dof, while the actual small-cut u_y (and u_z)
dofs stayed FREE and unstabilized. Invisible to the FD Jacobian checker
(J and r agree on whatever constraints exist), to the SVMP_AGGREGATION_DUMP
lines and to the EntityDofMap cross-validation (both read the span before
the clobber). Caught by the Product-field fixture asserting emitted line
contents. Fix: copy the dofs into a dedicated slave_dof_storage array.

Accuracy effect of the fix (GP-off committed family, nx16):
0.0028127→0.0029190 (dt.02, +3.8%) and 0.0030367→0.0034310 (dt.01, +13%);
GP-on nx16 0.004256→0.0041863 (−1.6%); nx8 GP-off 0.0061127→0.0060167
(−1.6%). A bit-exact control run on the pre-edit tree reproduced the
recorded 0.0028127203134796844, attributing the shift entirely to the fix:
the bug's spurious root constraints had acted as a weak accidental band
damper. The corrected semantics is the only defensible state; all verdicts
below are measured on it.

### 11.3 R1 verdict: the gap is real at nx16 — and inverts at both ends

Committed (GP-off) configuration, velocity rel-L2:

| config            | penalty (bin_final) | aggregation (corrected) |
|-------------------|---------------------|--------------------------|
| nx8  dt.02        | 0.025757 (verifier FAILED, 8.8 iters/step) | 0.006017 (passes) |
| nx16 dt.02        | 0.0015352           | 0.0029190 (1.9× worse)  |
| nx16 dt.01        | 0.0011153           | 0.0034310 (3.1× worse)  |
| nx24 dt.01        | DIVERGED (Newton, step 13/50) | 0.0016996      |
| nx32 dt.01        | backend abort step 27 (same as agg, 11.5) | (run invalid, see 11.5) |

So: in the committed config the penalty stack is 2–3× more accurate at
nx16 and dt-converges there (order ~0.46) — the aggregation-specific gap
that motivated this plan is REAL at that resolution and larger than the
GP-on-family table suggested. BUT the penalty stack only WORKS at nx16 in
this config: it fails its own verifier at nx8 (0.0258; the old committed
deck would have shipped a failing case — historically nx8 was only ever
measured GP-on) and its Newton DIVERGES outright at nx24 (the eff∝h²/dt
penalty scaling leaves the stability window — the same γ-window pathology
the calibration campaign documented), while aggregation passes and
h-converges at every resolution the shared cut-geometry backend can
handle. The "2–3× at nx16" is the entirety of the penalty stack's
committed-config advantage; everywhere else it is strictly worse or
non-functional. (Both stacks share the nx32 SayeHyperrectangle
corner-degenerate abort — confirmed on the era binary too — so that
limitation is the geometry backend's, not either stabilization's.)

### 11.4 R2 carrier decomposition: a u_y band mode the penalty damped

Tooling: /tmp/svmp_r2_analysis.py (carrier classes × shells × u_x/u_y/p,
slave sets from the dump). Aggregation (corrected, nx16 GP-off, final
step): u_y rms 0.00148 (slaved) / 0.00145 (free-cut) / 0.00103
(band-adjacent) / 0.00034 (far-field), u_y/u_x ≈ 3–5 in the band; secular
growth +4e-5/output in band classes vs +0.9e-5 far-field. Pressure shows
NO band carrier (~1.4 Pa uniform offset-removed rms everywhere). The same
analysis on the PENALTY run (same config, era binary): free-cut u_y
0.000255 — ~6× smaller, NO u_y dominance (u_y/u_x ≈ 0.9), secular rate 4×
slower. The deleted velocity gradient-jump penalty was damping exactly
this u_y band mode; aggregation constrains only the unsupported dofs and
leaves the mode free on slaved and supported band dofs alike (the slaved
dofs inherit it through their masters).

### 11.5 R5 scaling: the "floor" is an nx16 hump, not an h-floor

Aggregation GP-off (corrected binary): nx8 0.006017 (dt.02); nx16
0.002919/0.003431/0.003503 (dt.02/.01/.005 — mild dt ANTI-convergence
persists); nx24 0.0016996/0.0020244 (dt.01/.005); h-order nx16→nx24 =
1.73 (dt.01) / 1.35 (dt.005). The section-9.5 "weakly-h-dependent floor
~0.003" was an artifact of measuring only nx8→nx16: there is no floor,
the nx16 point is a local hump and convergence ACCELERATES past it.
CAVEAT: both nx32 runs died mid-run (step ~48/100 and ~late) in the
SayeHyperrectangle implicit-cut backend — a corner-degenerate cut (corner
φ value ~1e-10) with curved_edge_root_mismatches=1 under
implicit_cut_fallback_policy=Fail. This is a PRE-EXISTING fine-h geometry
robustness limitation (degenerate corner-touching cuts become likelier as
h shrinks), unrelated to aggregation; the half-run errors (0.00069/0.00077)
suggest continued convergence but are not quotable. Follow-up owner:
implicit-cut backend (fallback policy or terminal-topology refinement at
corner-degenerate cuts).

### 11.6 R3/R4: the remaining error is in the converged states, not the
### machinery

R3 (SVMP_CUT_RULE_DUMP=1 + /tmp/svmp_r3_probe.py, monomials {1,x,y,x²,xy,y²}
against exact per-cell wet integrals): the generated cut rules are exact
for the DISCRETE domain (Σw vs measure ~1e-18) and match the ANALYTIC
moving domain to ~1.5e-5 while φ is exact (step 0); afterwards the
consistency error (~1e-3 relative, saturating by step ~18) is the discrete
interface's position drift (φ-transport class), not quadrature quality.
Correlations with the velocity error: far-field LEVEL +0.92 (co-trending
monotone series — not causal evidence), band-class INCREMENTS
anti-correlated (−0.8). An integration-accuracy velocity floor is ruled
out. R4 (SVMP_CUT_REFRESH_PER_STEP_ONLY=1): freezing the cut context
within the step gives 0.010943 (dt.02, 3.7×) and 0.005963 (dt.01, 1.7×) —
the per-step-only arm dt-converges ~O(dt), i.e. the per-accepted-iterate
refresh removes a first-order-in-dt moving-domain lag error and is NOT the
floor's injector. Newton convergence parity in both arms.

### 11.7 R6 mitigation prototypes: clean negatives, knobs retained

(a) Slave-ALL-cut-vertices with band-adjacent roots: structurally
infeasible — supported band vertices are masters of each other's lines →
"Constraint cycle detected" at the first solve. (b) Slave-all with
LAYER-2 roots (SVMP_AGGREGATION_SLAVE_ALL_CUT=1 + deck
Generated_interface_affected_cell_neighborhood_layers=1; root BFS widened
to traverse band full cells and reject candidate-bearing roots): runs
correctly (105/105 candidates slaved, cycle-free, 25/25 and 50/50
converged) and is accuracy-NEUTRAL: 0.0029523 vs 0.0029190 (dt.02),
0.0034558 vs 0.0034310 (dt.01). The layers=1-only control is bit-identical
to baseline. Verdict: the u_y band mode is not removable by constraining
band dofs to interior polynomial extensions — it is the converged
solution's response to the moving discrete interface at marginal band
resolution. With 11.5's h-convergence, no constraint-side mitigation is
warranted and the GCL/space-time route stays an unjustified last resort.
Both knobs remain for future experiments (default off).

### 11.8 d18 gates and the island-pin policy

Gate #1 (MPC-fix binary, stock 312-step deck = aggregation + reinit ON):
completed 312/312 with 0 drops; front_error −0.0098 (recorded class
−0.0092); profile rmse 0.0773 vs class ~0.019–0.023 — INFLATED by the
pre-existing reinit-on lid false-wetting artifact (84 spuriously wet nodes
above y=0.151 at x∈[0,0.48], documented 2026-06-10 as "onset moves
earlier" under reinit; φ stays bounded ±1.7). Restricting the comparison
to the unpolluted bore region (x≥0.50) gives rmse 0.0321 with the known
crest-flattening signature. CRITICALLY, gate #1's log shows
vertices_without_root>0 in 466 refresh applications (3–48 vertices,
pressure AND velocity) once the surface fragments — no-root cut islands
are ROUTINE in this physics and were running fail-open/unstabilized all
along. Gate #2 (corrected binary, first fail-closed build) therefore
stopped at step 300/312 — the C1 policy meeting reality. Policy refined to
the three-way split of 10.0/C1 (islands pinned homogeneously; machinery
failures throw). Gate #3 (final binary: clobber fix + island pins):
312/312 steps, 0 drops, islands pinned in ~80 refresh applications (11–18
dofs each) with no robustness effect; physics IDENTICAL to gate #1's
class — front_error −0.0098 (same to all digits), bore-region (x≥0.50)
rmse 0.0322/mae 0.0187 (vs 0.0321/0.0183), full-profile rmse 0.0774 (vs
0.0773) still dominated by the same pre-existing reinit-on lid
false-wetting (113 spurious wet nodes vs 84; φ bounded). Verdict: the
clobber fix and the island-pin policy are robustness-neutral and
physics-neutral on d18 at t=0.156; the gate's accuracy ceiling remains the
documented reinit-on false-wetting artifact, which predates and is
independent of this campaign.

### 11.9 Success criteria (section 7) — final verdict

- Robustness: MET and STRENGTHENED — every MMS run completes and passes,
  d18 312-step completes with 0 drops and (final policy) pinned islands;
  aggregation passes at nx8 where the committed-config penalty stack
  outright fails; suites green (constraints 243 incl. 14 new fixture
  tests, sparsity 650, timestepping 158, assembly 757, backends 184,
  physics 121, OpenVessel 12).
- "nx16/dt0.01 ≤ ~0.0025, h-order ≥ ~1.4": NOT MET AT NX16 (0.0034;
  legacy-mask h-order nx8→16 ~1.04) but MET PAST IT (nx16→24 h-order
  1.73; nx24 dt.01 = 0.0017 ≤ the penalty-era 0.0019 target) — the
  criterion was anchored to a resolution that sits on the hump. The
  committed-config penalty baseline this plan asked for (R1) shows the
  nx16 gap is real (11.3) and R2/R6 show it is the un-damped u_y band
  mode, not a defect: accepting ~2× at marginal band resolution buys
  coarse-h robustness, parameter-freedom, and faster asymptotics.

### 11.10 Code changes (all uncommitted), knobs, artifacts

Code (working tree):
- FE/Constraints/SmallCutAggregationConstraint.{h,cpp}: slave-dof copy
  bugfix; sub-parametric rejection; empty-master-line invariant; three-way
  under-aggregation policy (island pins / machinery throw / env fail-open);
  per-refresh churn counters (entered/left, keyed by system+field);
  active_side_volume_rules + island_pinned_dofs + pruned rule count/measure
  in the diagnostic line; per-call env reads (testability); slave-all-cut
  experiment (candidate struct, widened band adjacency, candidate-free
  root acceptance); header contract rewritten.
- FE/Systems/SystemSetup.cpp: constraint_structure_signature value in the
  sparsity-refresh log line.
- Application/Core/ApplicationDriver.cpp: SVMP_CUT_REFRESH_PER_STEP_ONLY
  (transient synchronize_state freeze); SVMP_CUT_RULE_DUMP (reference-frame
  rule/qp dump at initial + accepted steps, with metadata side).
- Application/Core/LevelSetCutConfiguration.cpp: SVMP_CUT_RETENTION_FORCE
  (active_only | active_and_inactive).
- FE/Tests/Unit/Constraints/test_SmallCutAggregationConstraint.cpp (NEW,
  14 tests) + FE/CMakeLists.txt registration.

New env knobs (all default-off/inert): SVMP_AGGREGATION_ALLOW_UNAGGREGATED,
SVMP_AGGREGATION_SLAVE_ALL_CUT, SVMP_CUT_REFRESH_PER_STEP_ONLY,
SVMP_CUT_RULE_DUMP, SVMP_CUT_RETENTION_FORCE. Existing knobs made
per-call: SVMP_AGGREGATION_LINEAR_EXTENSION, SVMP_AGGREGATION_MAX_LINES.

Artifacts: penalty runs /tmp/penF_* (bin_final) and /tmp/pen_* (HEAD tree,
superseded); aggregation reruns /tmp/inv2_* (s3b) /tmp/inv3_* (s3d/s3e)
/tmp/inv4_* (s3g/s3h); d18 gates /tmp/inv_d18_gate312{,_fixed,_fixed2};
MPI smoke /tmp/inv3_mpi_nx8; analysis /tmp/svmp_r2_analysis.py,
/tmp/svmp_r3_probe.py, /tmp/svmp_table2.py, jsons /tmp/r2_*.json
/tmp/r3_*.json; binaries /tmp/svmp_bin_fix_s3{,b,c,d,e,f,g,h},
/tmp/svmp_bin_penalty (HEAD tree), /tmp/svmp_bin_final (era snapshot —
PRESERVE THIS), penalty worktree /tmp/svmp_penalty_tree.

Incidental observations for future owners: (i) EntityDofMap::getEdgeDofs
indexed by MeshBase topological edge ids returns a DIFFERENT edge's dofs
on the unit-test fixtures — consumers must resolve midside dofs via the
nodal cell pairing (cross-validated in the fixture); worth an audit of
any getEdgeDofs callers. (ii) On 2D order-2 meshes,
set_faces_from_arrays(BoundaryOnly)+set_boundary_label does not survive
the full codim-1 derivation that edge-dof meshes trigger — label derived
faces after finalize() instead (fixture helper shows how). (iii) The
working-tree committed MMS deck still carries mid-campaign penalty keys
(γ_v=10/γ_p=100 comment block) that are inert on the current binary —
clean up before commit.

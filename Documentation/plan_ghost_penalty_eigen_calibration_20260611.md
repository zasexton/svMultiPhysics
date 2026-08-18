# Eigenvalue-Calibrated Cut-Cell Ghost Penalty (Phase 1) + Small-Cut Aggregation (Phase 2)

Date: 2026-06-11
Predecessors: `Documentation/d18_d38_spheric_test05_validation_root_cause_20260610.md`,
`Documentation/qualification_logs/d18_d38_gamma_gp_calibration_fix_20260610.md`

## Problem

The transient velocity ghost penalty `gamma_v * c_cal * (mu + rho h^2/dt) * h`
requires case-dependent tuning: the measured stability window of the effective
coefficient is (0.024, 0.073)..(0.22, 0.45) across the Q2 MMS and P1 dam-break
families — only ~3x wide — and the coefficient scales as 1/dt under the
adaptive controller. The current state needs a global calibration constant
(0.01) plus per-case multipliers (1.0 / 10.0): a fictitious-parameter stack.

## Phase 1 design: locally computed coefficients

Replace the analytic coefficient with per-facet constants computed from local
generalized eigenproblems at cut-context refresh. For each cut-adjacent facet
F with cells (T0, T1), on the broken patch space V(T0) x V(T1) (scalar
component basis; conservative superset of the conforming space):

- `lambda_visc(F) = lambda_max( K_dry ; J + K_wet + eps*I )`
- `lambda_mass(F) = lambda_max( M_dry ; J + M_wet + eps*I )`

where `K_*`/`M_*` are physical-frame stiffness/mass matrices integrated with
the cut context's retained wet/dry volume rules of whichever patch cells are
genuinely cut, `J` is the unscaled gradient-jump facet matrix
`int_F [[grad u]].[[grad v]] ds`, and `eps` is a machine-scale relative
regularization (numerical tolerance, not a model parameter). The penalty term
becomes

```
velocity_penalty * ( mu * lambda_visc + rho/dt_eff * lambda_mass ) *
    sum_c int_F [[grad u_c]].[[grad v_c]] ds
```

with NO h factor and NO calibration constant: the lambdas carry all geometry
(h-scalings, element shape, cut configuration) by construction, and the
mu / rho/dt split keeps the stored values dt- and physics-independent so no
recompute is needed when the adaptive controller changes dt. `velocity_penalty`
remains as a pure user override with default 1.0 (no longer doing calibration
work).

Rationale for the structure: the gradient-jump penalty controls the dry-side
energy only up to patch polynomials (zero-jump modes); those are controlled by
the neighbor's wet energy, hence `K_wet`/`M_wet` in the right-hand operator.
The computed bound makes the dominance sharp where cuts are bad (lambda grows
exactly on sliver configurations) and small where cuts are benign, which is
what a single global constant cannot do.

### Plumbing

- `CutCellForms.h`: two new parameter slots (`ghost_visc_calibration`,
  `ghost_mass_calibration`) + `cutGhostViscCalibration()` /
  `cutGhostMassCalibration()` form helpers. ParameterRef is already supported
  by the interpreter and JIT, so no new FormExprType.
- `CutIntegrationContext.h`: `CutFacetSetFacetMetadata` gains the two computed
  values + handle accessors + a context setter
  (`setFacetGhostPenaltyCalibration`).
- New `FE/Assembly/GhostPenaltyCalibration.{h,cpp}`:
  `computeGhostPenaltyCalibration(context, mesh, space, options)`; dense
  lambda_max via Cholesky + power iteration (patch spaces are <= ~20 DOFs).
  Facet quadrature is built in cell reference coordinates by barycentric
  mapping of face-vertex global ids (exact for straight reference facets of
  all standard cells; geometry curvature enters only through the cell mapping
  used for J_inv, det J, and the surface measure factor).
- `StandardAssembler.cpp`: bind the two values into the cut parameter slots
  alongside the existing stabilization scale.
- `IncompressibleNavierStokesVMSModule.cpp`: velocity first-order jump term
  uses the computed coefficient; `SVMP_GHOST_PENALTY_LEGACY=1` rebuilds the
  previous analytic form for comparison. Second-normal-derivative branch
  (only used by explicitly configured high-order penalties) and the pressure
  penalty keep the analytic scaling in Phase 1.
- `ApplicationDriver`: compute calibration in the cut-context refresh
  chokepoint (`refreshActiveCutIntegrationContextCached`) whenever the context
  was rebuilt, using the fluid velocity space.

### Phase 1 OUTCOME (2026-06-11): coercivity calibration is unusable here — measured, not assumed

The machinery works (d18: 1492/1492 facets calibrated, zero failures; the
lambdas verifiably encode cut geometry — lambda_visc_max 33.3 matches the
logged 3% minimum volume fraction exactly; lambda_mass tracks dry/wet mass
ratios). But the coercivity-sufficient coefficient (rho/dt)*lambda_mass lands
at 1e5..5e9 for d18 — 2 to 6 orders ABOVE the empirically measured
Newton-conditioning ceiling (~0.45), and Newton fails at step 0 exactly as
with the analytic gamma=1 scaling. Deflating the zero-jump (polynomial) modes
only brings the bound down to the analytic gamma~O(1) scale, which probe D
already proved fatal. Conversely, probe A showed d18 converges with ZERO
velocity penalty: the workable small-gamma regime is genuinely SUB-coercive —
the velocity extension governs dry DOFs and the penalty is only a conditioning
aid. Conclusion: **no eigenproblem of the coercivity family can replace the
empirical constant for the velocity penalty in this regime.** The default
stays on the calibrated analytic scaling; the computed mode is selectable via
`SVMP_GHOST_PENALTY_COMPUTED=1`; the computed lambdas are repurposed as the
parameter-free ill-posedness detector for Phase 2, which removes the penalty's
job instead of calibrating it.

### Phase 1 acceptance gates (superseded by the outcome above — the
parameter-stack removal now rides on Phase 2)

1. d18/d38 stock XMLs: step 0 converges, 312-step profile within the
   validated class at t=0.156.
2. dt-collapse probe (d18 at forced dt_min=1.5625e-5): step converges —
   the case the analytic 1/dt scaling could never satisfy.
3. MMS nx8/nx16 dt=0.02 + nx16 dt=0.01 with case XML multipliers reverted to
   1.0: verifier passes, velocity errors within the post-rate-fix class
   (0.0089/0.0032/0.0019) — this removes the per-case compensation.
4. Unit suites: test_physics, test_application OpenVessel/LevelSet filters,
   test_fe_timestepping.

On pass: remove `kCutPenaltyTransientCalibration`, revert MMS XML multipliers,
narrow the extension-floor usage (pending coverage audit of cells with pruned
wet rules and no calibrated facet), and update the qualification log.

## Phase 2 status (2026-06-11): implemented through constraint emission; blocked on transient-Newton condensation consistency

Implemented and verified end-to-end on d18 (working tree):

- `FE/Constraints/SmallCutAggregationConstraint.{h,cpp}`: cell classification
  from retained cut-context rules, deterministic BFS to full-active roots,
  Newton inversion of the root mapping, polynomial-extension weights,
  AffineConstraints emission, strong-Dirichlet vertex exclusion
  (boundary-marker driven), `isConstrained` precedence guard. d18: 67
  velocity vertices aggregated (207 candidates minus 140 wall-excluded),
  207 pressure vertices, zero root/inversion failures.
- Both-space aggregation (velocity AND pressure — mixed problems must
  aggregate both or the local saddle point breaks).
- NS module/Parameters plumbing: `<Small_cut_aggregation>` on the
  free-surface BC; velocity ghost penalty skipped when active
  (`velocity_ghost_penalty_mode=skipped_by_aggregation`).
- `ConstraintDistributor`: constrained-row identity fixup extended from
  Dirichlet-only to ALL constrained rows (master-bearing slave rows were
  left as exact zero rows — 201 zero diagonals — which no existing code
  path handled because this solver had never run master-bearing
  constraints through assembly).

BLOCKER ROOT-CAUSE SESSION (2026-06-11, second pass) — established facts:

1. The 4-DOF master-bearing-constraint fixture
   (`Tests/Unit/TimeStepping/test_MasterBearingConstraintTransient.cpp`,
   3 tests: BackwardEuler/GeneralizedAlpha/with-source) PASSES: the core
   condense -> solve -> distribute -> line-search pipeline is correct.
2. FD Jacobian checker (`SVMP_FE_JACOBIAN_CHECK=1`, components filter to
   exclude the phi sweep): E-config control (weak penalties, NO aggregation)
   is consistent to machine precision (errors ~2e-5 on fd norms of 80-240).
   With ONE aggregated vertex (`SVMP_AGGREGATION_MAX_LINES=1`), Velocity
   row errors of +-16 appear at band-vertex V1 rows and the identical
   values reproduce with the full 274-line set. Newton fails identically
   (line-search trial residual ~ alpha * 70).
3. Fixed during the session (all latent, Dirichlet-masked):
   - interior-face SELF and CROSS blocks inserted raw (no condensation) —
     now routed through `insertLocalConstrained` when constrained
     (both face paths + generated-interface two-sided couplings);
   - fused combined insertion bypassed condensation — now gated off when
     constraints are active;
   - `ConstraintDistributor` left master-bearing slave rows as exact zero
     rows (201 zero diagonals) — identity fixup extended to all
     constrained rows in BOTH the rectangular and the square element paths
     (review follow-up: the square path was missed initially; unit-test
     expectations in test_ConstraintDistributor.cpp updated to the
     identity-row contract). Note: the assembly-layer
     `AssemblyConstraintDistributor::finalizeConstrainedRows` already sets
     identity for all constrained DOFs at finalization on paths that call
     it; the per-element fixup makes direct distributor users (e.g.,
     contact kernels) safe and is idempotent with finalization;
   - `EigenMatrix::addValue` SILENTLY DROPS out-of-pattern writes
     (`SVMP_EIGEN_COUNT_DROPPED=1` instruments): constraint sparsity
     augmentation (`ConstraintSparsityAugmenter`, SystemSetup.cpp ~4248)
     runs ONLY at setup, but the aggregation constraint emits nothing then
     (no cut context yet) -> its elimination fill is missing from the
     pattern -> condensation writes at master entries are dropped.
     Measured on the single-slave run: 44 dropped writes (tiny
     pressure-penalty values) — real but not the dominant term.
4. Remaining unexplained: the +-16 V1 row inconsistencies at band vertices
   (mass/VMS scale). Dof-numbering ground truth recovered (field-local
   interleaved; mesh vertex 476 -> field index 466 -> dofs 5028-5030); the
   flagged rows decode to band-vertex V1 rows, but one numbering
   indirection (mesh->field vertex permutation) remains unverified for the
   far entries. Candidate mechanisms still open: the checker's direction
   zeroing using Dirichlet-only `constrained_dofs` (slaves perturbed raw,
   then overwritten by trial distribute -> effective-direction mismatch
   against J_cond columns), or a band-row assembly-path divergence.

NEXT STEP (single instrumentation cycle): extend the Jacobian checker dump
to print (dof -> mesh vertex, xyz) for its top-mismatch entries, and zero
ALL constrained dofs (not just Dirichlet) in the check direction; then
re-run the single-slave case. Fix design for the sparsity defect
(required regardless): re-run constraint sparsity augmentation + matrix
reallocation + resolved-table invalidation whenever rebuildConstraintState
changes the constraint structure signature (line/master-entry counts +
slave-id hash) — post-setup constraints are otherwise second-class.
Precondition for ANY hanging-node/MPC-class constraint in the transient
solver, independent of aggregation.

## Phase 2 design sketch: small-cut aggregation (AgFEM)

Replace stabilization-by-penalty with constraint-by-aggregation: DOFs whose
support is ill-posed (identified parameter-free by the same local quotient:
`lambda` above exceeding the well-posed regime, or classic AgFEM "all cut
cells") are slaved via `AffineConstraints` to the polynomial extension from a
root interior cell, rebuilt with the cut context each step. Velocity ghost
penalty (and the extension-floor machinery) become unnecessary for
conditioning; remove once the Phase-1 gate matrix passes on aggregation alone.
Phase 2 reuses: the Phase-1 patch matrices (ill-posedness detector), the
facet-set/cut-context rebuild hooks, and the constraint distribution
machinery already exercised by hanging-node/MPC constraints.

## RESOLUTION (2026-06-11): cut-volume sparsity gap — root cause of the ±16 mismatch and the Phase-2 Newton blocker

The "remaining unexplained" ±16 V1 inconsistencies above are fully explained
and fixed. They were never a constraint/checker defect.

Root cause (base pipeline, pre-existing): the sparsity build in
`FESystem::setup` collected coupling pairs from `def.cells`, `def.boundary`,
`def.interior`, and `def.interface_faces` — but never from
`def.cut_volumes`. Cut-volume terms (the entire NS-VMS system integrated
over the active side of cut cells) therefore contributed NO sparsity. The
system rode on an accident: the velocity ghost-penalty term registered
(V,V) interior-face pairs, and that fill (self + cross couplings for both
cells of every interior face) happened to cover the cut-volume writes.
Any configuration that removes the velocity GP term from the form —
`Small_cut_aggregation=true`, or stock with
`Cut_cell_velocity_gradient_penalty=0.0` exactly (isZeroConstantScalarValue
skips registration) — lost the V×V/V×P couplings on cut cells; EigenMatrix
silently dropped those Jacobian writes (values up to ~970 in MMS), leaving
the residual (vector, patternless) untouched -> J inconsistent with FD(r) by
~50% on Velocity rows.

Evidence chain:
- d18 single-slave (N=1) and zero-line (N=0) aggregation runs show
  bit-identical mismatches -> constraint lines innocent.
- Fresh JIT disk cache: bit-identical -> cache innocent.
- 2D MMS stock with gamma_v=0.0 reproduces (~50% Velocity error, 35 dropped
  V×V writes among cut-cell nodes); gamma_v=1.0 control is machine-consistent
  with 0 drops. Interpreter (jit=false) reproduces the same drop -> not JIT.
- gdb catch-throw with SVMP_EIGEN_STRICT_PATTERN=1: writer is
  `StandardAssembler::assembleCutVolumesFused -> insertLocalForCell ->
  insertLocalConstrained` (StandardAssembler.cpp:11406/5628), i.e. the
  cut-volume NS block insertion.
- d18's dropped (V-row x P-col) master-fold writes are the same defect
  surfacing through constraint condensation: the base (v, p_slave) cell
  coupling was missing, so the augmenter's Rule 1 could not expand it onto
  the pressure masters.

Fixes landed (all serial-verified):
1. SystemSetup.cpp pair collection: `def.cut_volumes` now feeds
   `maybe_add_cell_pair` (conservative all-cells coverage — the active set
   tracks the moving interface, and this matches the nnz the GP face fill
   provided implicitly).
2. Constraint-structure sparsity lifecycle (the fix sketched above, now
   implemented): `FESystem::rebuildConstraintState` computes an
   order-independent structure signature over master-bearing lines
   (`computeConstraintStructureSignature`); on change it clones each
   finalized serial pattern back to Building state, re-runs
   `ConstraintSparsityAugmenter` (EliminationFill), re-finalizes, and bumps
   `sparsityPatternRevision()`. Distributed patterns warn (MPI
   re-augmentation not yet supported). `TimeLoop::run` snapshots the
   revision and reallocates the Newton workspace (matrix from the new
   pattern) before the next solve (`ensure_workspace_matches_sparsity` at
   the before-physics-solve hook and both in-loop updateConstraints sites).
3. EigenMatrix out-of-pattern drops are loud: unconditional first-drop
   WARN; `SVMP_EIGEN_COUNT_DROPPED=1` samples; `SVMP_EIGEN_STRICT_PATTERN=1`
   throws (used for the gdb localization).

Verification gate (2026-06-11, RelWithDebInfo, serial):
- MMS gv0 (term absent): drops 35 -> 0; Velocity J/r rel error 50% -> 3e-6
  (norms bit-identical to term-present control).
- MMS ctrl (term present): unchanged (0 drops, same norms) — no regression.
- d18 aggregation single-slave jacobian check: Velocity total_err
  38.6/114.6/11.2 -> 1.89e-5/1.78e-5/1.95e-5 (E-control parity); Pressure
  7.3e-10; 0 drops.
- d18 full aggregation (67 velocity + 207 pressure aggregated vertices,
  velocity GP term removed): 5/5 steps accepted, every nonlinear solve
  converged, residuals 3.3e-4 -> 6.1e-7 -> 2.4e-7 decaying, 0 drops, one
  mid-run constraint_sparsity_refresh exercised the full
  re-augment + workspace-reallocation path. Previously diverged to 5e12.
- Unit suites: test_fe_sparsity 650/650, test_fe_constraints 226/226,
  test_fe_timestepping 158/158. test_fe_assembly green after updating
  StandardAssemblerEdgeCases.ConstraintChainDistributesToMasters to the
  identity-row distributor contract (same class as the earlier
  test_ConstraintDistributor updates). One pre-existing, unrelated failure
  on the branch: test_fe_systems
  CanonicalWorkflow.SameSpace_SameName_DifferentBindings (field-binding
  norm contamination, ratio 87.9 vs 100; exercises no cut-volume terms, no
  constraints, no TimeLoop, no Eigen backend — outside every path touched
  by these fixes).

Consequence for the plan: the Phase-2 blocker is cleared — aggregation now
runs with a consistent Jacobian and no velocity ghost penalty. Next gates:
MMS aggregation accuracy matrix, d18/d38 312-step physics validation, then
the parameter-stack retirement ladder.

## Phase-2 validation campaign addendum (2026-06-11 afternoon)

Three further defects found and fixed while running the aggregation
accuracy/physics gates:

1. Mid-solve sparsity refresh (monolithic level-set+NS): with op='equations'
   the cut context legitimately rebuilds after every ACCEPTED NEWTON ITERATE
   (φ is part of the solve), so the constraint structure — and the
   re-augmented pattern — can change between iterations of one solveStep.
   The TimeLoop-level workspace reallocation was therefore insufficient
   (solveStep binds `auto& J = *workspace.jacobian` for the whole step).
   Fix: `GenericMatrix::reinitFromPattern` (default false) +
   `EigenMatrix::reinitFromPattern` (in-place storage rebuild preserving
   object identity; Eigen views carry no layout handle so no external
   invalidation), and `NewtonSolver::maybeReallocateJacobianForSparsity`
   called at solveStep entry AND inside the three Jacobian-assembling
   lambdas right after their synchronizeState (views are created after the
   check). NewtonWorkspace stores the factory + revision snapshot.
   Eigen direct solver builds SparseLU per solve — no symbolic-cache hazard.
2. Strong-pin precedence (both directions):
   - LevelSetActiveSideVertexDirichletConstraint now skips DOFs that are
     already constrained: an aggregated DOF is determined by its wet
     masters; no singular mode remains for the dry pin to fix.
   - Pressure gauge pins (Node_pressure_constraints) must WIN over
     aggregation — a homogeneous aggregation line cannot remove the global
     pressure constant. SmallCutAggregationConstraint gained
     `excluded_vertices`; the NS module resolves gauge node ids and passes
     them to the pressure-space constraint.
3. Run-dir hygiene: pressure_gauge.csv is an INPUT (gauge pin nodes), not an
   output.

Gate status:
- MMS Q2 (nx8): infrastructure clean (0 drops, in-place reallocations
  logged, no constraint throws) but Newton diverges at step 3 (residual
  7e10, 60 iters): vertex-only aggregation leaves Q2 midside DOFs on
  ill-posed cut cells unconstrained AND unpenalized. This empirically
  confirms the "extend to edge-node slaves (Q2)" ladder rung as a
  prerequisite for MMS aggregation. MMS stays on the validated penalty
  stack (γ_v=10/γ_p=100) until that rung lands.
- d18/d38 312-step P1 physics gates relaunched on the fully-fixed binary
  (in-place reinit + precedence + gauge exclusion) after the earlier
  attempt accumulated drops once the dam front moved fast enough to change
  the band mid-solve (~step 66+ on the entry-check-only binary).
  Comparison plan: compare_test05_profiles.py result_313.vtu vs
  reference_profiles d18_1.dat / d38_1.dat at t=0.156; penalty baseline
  d18: rmse 0.0230, front_err -0.0246 (multitime_profile_comparison.json).

### d38 aggregation physics gate — first read (2026-06-11 evening)

d38 312/312 steps, all 624 nonlinear solves converged, 0 dropped writes,
~1 in-place Jacobian reallocation per step (the band changes every step).
compare_test05_profiles.py vs d38_1.dat: validation PASSED;
front_error_m = -0.0090 (penalty d18 reference scale: -0.0098..-0.0246).

Two correlated anomalies, both traced to LEVEL-SET FIELD HEALTH rather
than hydrodynamics:
- phi range explodes under aggregation (d38 final: [-1.1e5, +9.5e4];
  d18-agg step 271: [-1.2e4, +2.1e4]) vs penalty d18 [-0.15, +0.17].
  These cases run reinitialization=disabled; under penalties the velocity
  ghost penalty smooths band velocities, which keeps the interface-sampled
  dry-side advection velocities (prescribed_data extension) tame. Without
  any band smoothing the dry-side phi compresses/steepens unbounded.
- Profile mean error +0.114 m with near-perfect front: phantom phi=0
  crossings in the noisy dry region pollute the extracted profile
  (interface_points_total 7602 for d38-agg vs 935 for penalty-d18 — 8x).
  The wet surface itself tracks the reference.

Candidate compensation (config, not code): enable the band-preserving
projection reinitialization (preserve_band_width auto) for aggregation
runs to keep dry-side phi a signed distance. To be verified after the
d18-agg / d38-penalty A/B completes.

### d18 aggregation physics gate — A/B vs penalty stack (same tool, same reference d18_1.dat)

d18-agg: 313 accepted steps (one recovered rejection at step 100, dt
auto-halved then restored), 627/629 nonlinear solves converged, 0 dropped
writes, ~1 in-place Jacobian reallocation per step.

| metric                  | penalty (fixcheck) | aggregation (no velocity GP) |
|-------------------------|--------------------|------------------------------|
| front_error_m           | -0.0098            | -0.00014                     |
| wet_volume              | 0.0021528          | 0.0021528 (6-digit match)    |
| mean_error_m (profile)  | +0.0058            | +0.143  (phantom-polluted)   |
| rmse_m (profile)        | 0.0223             | 0.146   (phantom-polluted)   |
| interface_points_total  | 935                | 8842                         |
| phi range               | [-0.15, +0.17]     | [-1.2e4, +2.1e4]             |
| tool validation         | PASSED             | PASSED                       |

Reading: hydrodynamics improve — the front lag that motivated the original
D18/D38 root-cause investigation is essentially eliminated once the
velocity ghost penalty's over-damping of the thin front sheet is removed.
Mass conservation identical. The profile-height metrics are not yet
comparable because the unbounded dry-side phi creates a phantom-crossing
cloud (9.5x interface points) that the extractor samples above the true
surface. The reinit-enabled aggregation rerun (projection, cadence 5,
band-preserving default) is the decisive test: if phi stays a signed
distance, phantom points vanish and rmse/mae become directly comparable.

### d38 penalty baseline (same tool, d38_1.dat)

312/312 steps. rmse 0.0191, mae 0.0120, mean_err +0.0032,
front_error_m -0.00919, max_abs 0.0596, peak_y_err 0.0244,
interface_points_total 871, phi [-0.150, +0.142], wet_volume 0.002645.

vs aggregation-noreinit: front error EQUAL (-0.0090 vs -0.0092) — d38's
front lag, unlike d18's, is not penalty-induced (deeper wet bed, different
front dynamics). wet_volume identical to 6 digits (0.0026448 vs 0.002645).
Profile rmse comparability still pending the reinit-enabled aggregation
run (154/312 at last check, all solves converged, 31 projection reinits,
0 drops).

### FINAL d38 three-way gate (2026-06-11) — aggregation + band-preserving reinit replaces the velocity penalty on P1

312/312 steps, 624/624 nonlinear solves converged, 0 dropped writes,
62 projection reinits (cadence 5, band-preserving default).

| metric                 | penalty   | agg (no reinit) | agg + reinit |
|------------------------|-----------|-----------------|--------------|
| rmse_m                 | 0.0191    | 0.120 (phantom) | 0.0194       |
| mae_m                  | 0.0120    | 0.114           | 0.0123       |
| mean_error_m           | +0.0032   | +0.114          | +0.0049      |
| front_error_m          | -0.00919  | -0.00902        | -0.00919     |
| max_abs_error_m        | 0.0596    | 0.136           | 0.0593       |
| peak_y_error_m         | 0.0244    | 0.064           | 0.0242       |
| interface_points_total | 871       | 7602            | 919          |
| phi range              | ±0.15     | ±1e5            | [-0.144,+0.143] |
| wet_volume             | 0.002645  | 0.0026448       | 0.0026448    |

Combined with the d18 result (front_error -0.00014 under aggregation vs
-0.0098 under penalties; identical wet volume), the Phase-2 P1 verdict:

  Small-cut aggregation + projection reinitialization fully replaces the
  velocity ghost penalty on P1 unfitted free-surface cases — equal profile
  accuracy (within 2%), equal-or-dramatically-better front tracking, equal
  mass conservation, healthy level-set field, and one fewer empirical
  parameter stack.

Remaining ladder (future work): (1) edge-DOF aggregation for Q2 spaces —
prerequisite for MMS; until then MMS keeps gamma_v=10/gamma_p=100 penalties;
(2) flip Small_cut_aggregation default for the d18/d38 P1 family + enable
reinit in those cases; (3) remove the velocity-penalty parameter stack;
(4) retire/demote the eigenvalue-calibration infrastructure (Phase 1).

## Retirement-ladder execution (2026-06-12)

Rung 1 — d18/d38 default flip (DONE): both repo cases now set
Small_cut_aggregation=true + Enable_reinitialization=true (projection,
band-preserving default); test_OpenVesselExamples literature expectations
updated in lockstep.

Rung 4 — eigenvalue-calibration retirement (DONE): GhostPenaltyCalibration
.{h,cpp} deleted; CutCellForms slots 13/14 + cutGhostVisc/MassCalibration
helpers removed; CutFacetSetFacetMetadata calibration fields + handle
accessors + mutableFacetSetHandleForMarker removed; StandardAssembler
constant bindings removed; ApplicationDriver hook + SVMP_GHOST_PENALTY_COMPUTED
env removed; NS module computed-coefficient selector removed (legacy
transient scaling is THE velocity penalty path now). Zero references remain.

Rung 3 — Q2 aggregation (FUNCTIONAL; accuracy trade measured): four defects
fixed in SmallCutAggregationConstraint + one precedence inversion:
1. Cell-local dof resolution replaces EntityDofMap::getVertexDofs (which is
   empty for Q2 midside nodes -> skipped midside slaves AND silently dropped
   the midside master entries of every emitted line — inconsistent
   partial-polynomial extensions, the real cause of the original step-3
   divergence).
2. Per-cell dof layout (node-major vs component-major) detected empirically
   from a corner vertex with entity-map coverage and cross-validated on
   every slave (a wrong assumption slaved foreign dofs -> instant
   divergence; measured: getCellDofs is component-major here).
3. Extension polynomial = LINEAR corner sub-basis of the field element
   (full-order Q2 extrapolation weights reach +-8 one cell out and diverge
   Newton at step 0 even with correct dofs; linear weights stay ~+-2).
   Identity for P1 — d18/d38 path unchanged.
4. Boundary exclusion extended to higher-order face nodes via reference-
   layout geometry (ReferenceElement::face_nodes canonicalizes to corner
   topology; Q2 boundary midside nodes were never excluded).
5. AffineConstraints::addDirichlet now OVERRIDES master-bearing lines
   (strong Dirichlet wins — deal.II-style precedence). Pre-exclusion
   remains an optimization but is no longer correctness-critical
   (marker coverage, face order, and field-local index permutations each
   produced misses). test updated:
   AddDirichletOverridesMasterBearingLine.

MMS verification matrix (aggregation, no velocity GP, official per-case
verifier PASSED for all):
| velocity_relative_l2_error | penalty | aggregation(linear ext) |
|----------------------------|---------|-------------------------|
| nx8 dt0.02                 | 0.00897 | 0.00857 (4% better)     |
| nx16 dt0.02                | 0.00322 | 0.00481 (49% worse)     |
| nx16 dt0.01                | 0.00192 | 0.00389 (2.0x worse)    |
| h-order                    | ~1.48   | ~0.83                   |

Verdict: Q2 aggregation is robust and parameter-free but the linear
extension halves the mesh-convergence order — the expected theoretical
trade. MMS therefore KEEPS the penalty stack (gamma_v=10/gamma_p=100) as its
committed accuracy configuration; full-order extension needs weight
conditioning (capping/equilibration) before it can replace it — future
rung. P1 cases (d18/d38) have no such trade (linear IS the field basis)
and run aggregation by default per rung 1.

P1 regression on the final binary: d18 5-step smoke 5/5 converged, 0 drops.
Suites: constraints 226 (1 test updated to the override contract), sparsity
650, timestepping 158, assembly 757, backends 184, systems 558 (+1 known
pre-existing CanonicalWorkflow failure).

## Rung 2 COMPLETE (2026-06-12): velocity ghost penalty deleted

Decisive re-examination first: tracing the build timeline showed FULL-ORDER
extension had only ever been tested with the dof-layout defect (foreign-dof
slaving) — the "weights +-8 diverge Newton" conclusion was confounded, and
the "linear extension" code path never actually engaged (element_type() is
the topology type, so linearElementType() was identity). Every passing run
was full-order AgFEM extension all along. Confirmed by bit-identical nx8
results with the gate forced either way. The measured accuracy matrix
(nx8 0.00857 / nx16 0.00481 / dt01 0.00389, h-order ~0.83) IS the
full-order behavior; the fine-mesh gap vs penalties is NOT an
extension-order artifact (band/φ-coupling investigation remains a future
item). SVMP_AGGREGATION_LINEAR_EXTENSION=1 now genuinely selects the
bounded linear sub-basis for A/B work.

Deletion executed:
- Module: velocity gradient-jump + second-normal-derivative terms removed;
  FreeSurfaceCutCellStabilization loses velocity_gradient_penalty and
  velocity_max_derivative_order; velocity-specific derivative-order policy
  and log fields removed (velocity_ghost_penalty_mode=
  retired_replaced_by_aggregation); obsolete "requires a nonzero penalty"
  validation dropped (policy-disabled pressure with enabled stabilization is
  a legitimate inert config).
- Parser: Cut_cell_velocity_gradient_penalty / _max_derivative_order (and
  aliases) accepted-and-ignored with a deprecation warning — archived decks
  stay loadable. Parameters.cpp whitelist unchanged.
- small_cut_aggregation DEFAULTS TO TRUE (it is the conditioning mechanism).
- Active cases/generators stripped of the keys (d18/d38, generic unfitted
  example, square_tank_tilt_settling, MMS generator, validation-matrix and
  velocity-growth-smoke scripts, generate_validation_meshes.py). Dated
  archive dirs intentionally untouched.
- Tests updated to the retired contract: test_MovingDomainPhysics (3 tests
  incl. renamed NavierStokesUnfittedVelocityGhostPenaltyRetired),
  test_NavierStokesLegacyBCs (2 policy-translation tests),
  test_OpenVesselExamples + test_OpenVesselStabilizationMetadata (key
  absence asserted).

Verification on the final binary:
- MMS nx8/nx16 aggregation: bit-identical to pre-deletion
  (0.008573483136327782 / 0.004809517743853168), verifiers PASS,
  deprecation warning fires for decks still carrying the key.
- d18 P1 smoke: 5/5 converged, 0 drops, retired-mode diagnostic.
- Suites: physics 121/121 (MovingDomain+LegacyBCs), application OpenVessel
  12/12, FE constraints 226/226.

The velocity ghost penalty and its parameter stack no longer exist in the
codebase. Remaining open item (accuracy, not stability): the fine-mesh
velocity-error gap vs the old penalties (1.5-2x at nx16) — suspected
band-constraint churn / φ-coupling, to be investigated independently.

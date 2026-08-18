# Unfitted Level-Set Free-Surface Boundary-Condition Remediation Outline

Date: 2026-05-26

## Purpose

This document turns the latest review of the unfitted level-set free-surface
boundary condition into an implementation and verification roadmap. The review
compared the current code against common CutFEM, XFEM, embedded-boundary FEM,
and level-set free-surface practice. The goal is to close correctness gaps
without over-claiming the method beyond the formulation that is actually
implemented.

The current supported method remains a one-fluid, one-sided embedded
free-surface model:

- Navier-Stokes volume terms integrate over the selected active wet side of the
  generated cut volume.
- The free surface is a generated embedded interface integrated through
  `dI(marker)`.
- The passive side exists only for support operations such as velocity
  extension, constraints, stabilization, and diagnostics.
- High-order implicit generated geometry is refreshed from the current
  level-set state, but its high-order quadrature weights, curvature, and
  topology transitions are not a fully differentiated Newton tangent.
- Two-phase material jumps, two-sided traces, pressure enrichment, and
  conservative level-set transport are not currently production features.

## Review Anchors

Primary implementation areas:

- `Code/Source/solver/Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.cpp`
- `Code/Source/solver/Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h`
- `Code/Source/solver/FE/LevelSet/LevelSetTransport.cpp`
- `Code/Source/solver/FE/LevelSet/LevelSetReinitialization.*`
- `Code/Source/solver/FE/LevelSet/LevelSetInterfaceLifecycle.cpp`
- `Code/Source/solver/FE/Docs/LevelSet.md`
- `Documentation/moving_free_surface_validation_cases.md`

External reference anchors used for the comparison:

- NGSolve CutFEM/XFEM documentation for side-specific cut integration,
  level-set normals, jumps, averages, Nitsche terms, and high-order cut
  geometry:
  <https://ngsolve.org/ngsolve/docs/i-tutorials/unit-8.3-cutfem/cutfem.html>.
- GetFEM XFEM documentation for level-set-adapted integration on
  `p(x)<0`, `p(x)>0`, `p(x)=0`, and explicit `Xfem_plus`/`Xfem_minus`
  discontinuity handling:
  <https://getfem.org/userdoc/xfem.html>.
- Massing, Larson, Logg, and Rognes on stabilized Nitsche fictitious-domain
  Stokes problems with velocity and pressure ghost penalties:
  <https://arxiv.org/abs/1206.1933>.
- Johansson, Larson, and Logg on high-order CutFEM for Stokes with Nitsche
  interface conditions and stabilization:
  <https://arxiv.org/abs/1505.00372>.
- Saye on high-order quadrature for implicitly defined surfaces and volumes in
  hyperrectangles:
  <https://doi.org/10.1137/140966290>.
- Olsson and Kreiss on conservative level-set transport for two-phase flow:
  <https://doi.org/10.1016/j.jcp.2005.04.007>.
- Gross and Reusken on finite-element surface-tension force discretization and
  Laplace-Beltrami curvature treatment in two-phase incompressible flow:
  <https://doi.org/10.1137/060667530>.
- SPHERIC benchmark pages for Test 02 dam break with obstacle, Test 05 wet-bed
  dam break, and Test 10 sloshing impact:
  <https://www.spheric-sph.org/tests/test-02>,
  <https://www.spheric-sph.org/tests/test-05>, and
  <https://www.spheric-sph.org/tests/test-10>.

## Current Progress Update - 2026-05-26

Workstream A is implemented for the one-fluid unfitted free-surface traction
path. The code now treats supplied scalar curvature and supplied/projected
curvature fields as signed by the generated-interface level-set normal
`grad(phi)/|grad(phi)|`; Navier-Stokes converts that scalar to the active-side
outward-normal convention before assembling `(-p_ext + gamma*kappa)n`.
Focused `MovingDomainPhysics` tests now cover both `LevelSetNegative` and
`LevelSetPositive` for scalar curvature and curvature-field pressure-jump
balance.

Workstream B logging and guardrails are strengthened. Unfitted free-surface
registration logs the generated interface geometry, geometry tangent policy,
shape-tangent status, curvature policy, curvature tangent policy, and pressure
stabilization policy. High-order implicit generated geometry remains a
`RefreshedFrozenQuadrature` quasi-Newton path unless and until high-order
quadrature, normal, curvature, and topology sensitivities are implemented.

Workstream C found no coefficient-ordering or projection bug for the P2
flat-horizontal null mode. A synthetic Quad9 residual test and a native Quad9
mesh projection test both preserve `phi=y-H` under constant horizontal
advection to roundoff. Current validation-matrix probes show flat/static P2
cases pass with zero or roundoff-level drift, and moving constant-translation
P2 errors decrease coherently on the asymptotic nx6/nx8/nx10/nx12 refinement
window. The earlier coarse nx2/nx3/nx4 non-monotone nodal interface-shift
metric is therefore not enough evidence for a formulation error by itself.

The case inventory audit for
`tests/cases/fluid/open_vessel_free_surface/unfitted_level_set` passes the
requested guard checks: active-domain literals are accepted, unsupported
two-phase controls are not present in the promoted entries, nonzero surface
tension is not being driven by raw level-set curvature, and velocity-extension
cases are recorded explicitly.

Current-code solution sanity probes also pass for the reduced high-order MMS
one-step gate and the strict square-tank hydrostatic equilibrium companion. The
MMS one-step result is a bounded generated-interface regression check; the
square equilibrium result is the stronger one-phase hydrostatic pressure,
traction, velocity, and area consistency check. These are serial direct runs
and do not close the remaining MPI/performance promotion item.

Current-code moving free-surface gates now also include the generated
`linear-sloshing-default` probe and the refined `square-refined-short-smoke`
probe in the validation matrix. Both completed with solver and verifier
success. This adds current evidence for the small-amplitude one-phase sloshing
reference and for an early refined tilted-tank transient, not just the reduced
MMS and static equilibrium checks.

MPI promotion evidence has been added for the tilted-square hydrostatic
companion. The validation runner now promotes MPI copies to distributed FSILS
controls with the level-set equation on GMRES and the fluid equation on the
FSILS Navier-Stokes block-Schur path, so existing `NS_*` controls are active
instead of being ignored by monolithic GMRES. The application wet-extension
velocity update now gathers nearest-interface and nearest-active samples across
MPI ranks before populating the prescribed `LevelSetAdvectionVelocity` field;
this fixes the dry-rank path where a rank with no local wet support previously
left prescribed coefficients empty. The MPI-2 and MPI-4 square-equilibrium
probes now pass as distributed regression gates with zero cut fallback cells.
MPI-2 remains the stricter roundoff-level hydrostatic accuracy companion;
MPI-4 is recorded as a dry-rank/distributed robustness gate with verifier-level
pressure and traction tolerances, not as a broad scaling or roundoff-accuracy
claim.

Capillary validation has been extended with curvature recovery refinement
evidence. `LevelSetCurvatureProjection.CircleCurvatureErrorImprovesWithRefinement`
checks that projected circle-curvature error decreases from the coarse to the
finer mesh, while existing circle/sphere recovery and active-side
pressure-jump residual tests cover static curvature and pressure-balance
behavior. Dynamic capillary waves or droplet/bubble benchmarks remain
intentionally unpromoted rather than claimed by the current zero-surface-tension
open-vessel cases.

Markdown cleanup was re-scanned after the update. The remaining matching
level-set/free-surface markdown files are canonical docs, case-level notes, or
plans with open checklist items. No additional markdown files were deleted in
this pass.

## Literature And Solver Alignment

The current implementation is aligned with established unfitted-FEM practice
only for the one-fluid embedded-boundary subset: integrate equations over the
selected physical side of a level-set cut volume, assemble explicit terms on
the generated interface, and stabilize small cut regions where needed. This is
consistent with NGSolve and GetFEM side-specific cut integration concepts.

The implementation intentionally does not yet match the broader two-sided
CutFEM/XFEM formulations shown in NGSolve/GetFEM examples or in the Nitsche
CutFEM literature. Those formulations introduce side-specific fields, jumps,
averages, two-sided traces, Nitsche consistency/penalty terms, and ghost
penalties for velocity and pressure. The current Navier-Stokes free-surface
path has one active physical fluid side and a passive exterior; it should not
be described as two-phase CutFEM validation.

The high-order implicit quadrature path is consistent with Saye-style
integration over level-set-defined surfaces and volumes as a quadrature
construction strategy, but only under the documented refreshed-frozen tangent
contract. Saye's setting makes the quadrature nodes and weights functions of
the implicit geometry; a fully differentiated Newton method would need
sensitivities for points, weights, measures, normals, curvature, and topology
changes. Those are not all available here for high-order generated geometry.

The transport path is not a conservative level-set method in the Olsson-Kreiss
sense. It is scalar level-set advection, optionally with SUPG, plus projection
or global-shift repairs. It can be used for one-fluid interface tracking when
the zero contour and active-domain sign are validated, but it should not be
claimed as conservative two-phase interface transport.

The capillary path is now sign-consistent for supplied/projected curvature
pressure-jump checks, but it is not yet a production capillary-flow method in
the Gross-Reusken/Laplace-Beltrami or pressure-enriched two-phase sense. Before
promoting dynamic surface-tension flow, the project still needs curvature
refresh ownership, curvature/normals refinement, capillary pressure-jump
refinement, and dynamic capillary wave or droplet/bubble benchmarks.

## Findings To Address

### F1. Potential capillary sign inconsistency for `LevelSetPositive`

`unfittedInterfaceNormal()` flips the level-set normal when
`Active_domain=LevelSetPositive`, but the raw level-set curvature path in
`applyFreeSurfaceBoundary()` and `unfittedInterfaceMeasureCurvature()` does not
perform an equivalent active-side sign conversion. If curvature is signed with
the level-set normal, the product `gamma * kappa * n` can be inconsistent when
the active wet side is positive rather than negative.

Risk: the same physical Laplace-pressure balance may pass for
`LevelSetNegative` but fail or require opposite user-supplied curvature signs
for `LevelSetPositive`.

### F2. Moving cut-geometry tangents are guarded or quasi-Newton, not full
Newton for high-order cuts

Shape tangents for the unfitted level-set free-surface path are disabled by
default, and high-order implicit generated geometry uses a refreshed-frozen
contract. That is acceptable if documented and tested as a Picard or
quasi-Newton geometry update. It is not equivalent to the fully differentiated
moving-boundary formulations expected in rigorous monolithic CutFEM/Nitsche
work.

Risk: nonlinear convergence and finite-difference Jacobian claims can be
overstated if high-order cut movement, interface measure, normals, curvature,
and topology transitions are described as fully linearized.

### F3. High-order level-set transport and signed-distance maintenance remain
accuracy risks

The transport path is scalar advection or divergence form with SUPG. Runtime
reinitialization is projection-only. Earlier review notes recorded concern that
a translated P2 flat-horizontal level-set null mode might not preserve the
expected invariant under constant horizontal advection; the 2026-05-26
element-level, native projection, and validation-matrix rechecks do not
reproduce that concern.

Risk: free-surface motion, volume diagnostics, and high-order MMS convergence
can be limited by coefficient ordering, gradient transformation, generalized
alpha rate initialization, boundary treatment, or reinitialization errors
rather than by the Navier-Stokes boundary condition itself.

### F4. Capillary support is guarded but not a production capillary-flow method

Raw level-set curvature with nonzero surface tension is rejected, and supplied
or projected curvature can be used for controlled checks. The projected
curvature field is treated Picard-style unless its owning equation supplies a
curvature residual and tangent.

Risk: static supplied-curvature checks can be mistaken for validated dynamic
surface-tension flow, capillary waves, droplet pressure jumps, or
geometry-coupled curvature convergence.

### F5. Current method is one-fluid embedded free surface, not two-phase CutFEM

The implementation has one active physical fluid side and a passive exterior.
It does not implement two-sided traces, material jumps, pressure enrichment, or
full two-phase jump conditions.

Risk: validation notes, examples, or solver controls may imply broader
two-phase capability than the code provides.

## Workstream A: Fix And Verify Active-Side Capillary Sign Conventions

Goal: make the dynamic free-surface traction convention invariant under a
change of active side, or explicitly reject ambiguous configurations.

Implementation tasks:

- Define the canonical curvature convention in `FreeSurfaceBoundary`: either
  curvature is signed with the raw level-set normal `grad(phi)/|grad(phi)|`, or
  curvature is signed with the outward normal of the active fluid region.
- Add a helper that returns the free-surface normal and traction curvature as a
  paired convention. Do not keep normal and curvature sign handling in separate
  ad hoc call sites.
- Update `applyFreeSurfaceBoundary()` so `(-p_ext + gamma * kappa) * n` uses a
  convention that is correct for both `LevelSetNegative` and `LevelSetPositive`.
- Decide and document how named curvature fields are interpreted:
  level-set-normal signed, active-outward signed, or explicitly declared by an
  option. Prefer one documented convention and convert internally.
- Keep raw level-set curvature rejected for nonzero production surface tension
  unless the caller explicitly selects a diagnostic mode.
- Audit `unfittedInterfaceMeasureCurvature()` before changing it. Measure shape
  tangents and traction curvature may need different sign conventions if one is
  expressed in level-set-normal coordinates and the other in active-outward
  coordinates.

Tests to add or update:

- A `LevelSetPositive` static Laplace-balance test matching the existing
  `LevelSetNegative` capillary pressure-jump tests.
- A phi-negation invariance test: flipping `phi` and switching active side
  should produce the same physical traction residual.
- A supplied-curvature sign test that proves the documented convention for named
  curvature fields.
- A regression proving `NavierStokesActiveDomainPositiveUsesPositiveCutVolumeSide`
  is complemented by a positive-side dynamic-stress/capillary residual check.
- Application-builder tests for the XML controls that select active side,
  curvature field, and surface tension.

Acceptance criteria:

- Positive-side and negative-side capillary residuals agree after convention
  conversion.
- The docs state one curvature sign convention and the code enforces it.
- Ambiguous supplied-curvature use either converts predictably or fails with a
  clear diagnostic.

## Workstream B: Harden The Moving Geometry Linearization Contract

Goal: make the nonlinear contract match the assembled Jacobian and solver
claims.

Short-term guard tasks:

- Keep high-order implicit geometry documented as
  `RefreshedFrozenQuadrature` unless full sensitivities are available.
- Replace the environment-variable-only shape-tangent gate with an explicit
  solver option if users are expected to rely on it.
- Add runtime logging that reports the geometry tangent policy, whether
  free-surface shape tangents are enabled, whether curvature is frozen, and
  which stabilization terms are omitted under high-order refreshed-frozen
  policy.
- Fail closed if a case requests full Newton consistency with high-order
  regenerated quadrature, curvature sensitivity, or topology sensitivity.
- Keep pressure-stabilization policy tied to the geometry tangent contract.
  Disabling pressure stabilization for refreshed-frozen high-order geometry
  should remain explicit and visible in logs.

Full-consistency implementation track, if required:

- Differentiate high-order interface point locations with respect to level-set
  DOFs.
- Differentiate high-order interface and volume quadrature weights beyond the
  current first-order or measure-only subset.
- Differentiate normals, measures, volume fractions, and surface curvature.
- Define the nonsmooth handling policy for topology transitions.
- Thread sensitivity data through `CutIntegrationContext`, generated interface
  domains, form lowering, matrix and matrix-free assembly, and preconditioner
  invalidation.
- Add finite-difference Jacobian checks for active volume, pressure traction,
  capillary traction, normal-dependent terms, and cut-adjacent stabilization
  away from topology events.

Acceptance criteria:

- Every promoted case states either "refreshed-frozen/quasi-Newton" or
  "differentiated quadrature" with matching code paths.
- Full-Newton claims require finite-difference Jacobian evidence for the exact
  geometry, field order, and backend being promoted.

## Workstream C: Repair High-Order Level-Set Transport Accuracy

Goal: make the level-set field motion accurate enough that free-surface boundary
condition tests are not polluted by transport artifacts.

Immediate P2 null-mode investigation:

- Add an element-level test for a Quad9 field `phi = y - H` with constant
  velocity `u = (0.1, 0)`, advective form, SUPG disabled, and no cut domains.
  The non-dt spatial residual should be roundoff zero.
- Repeat the test on one Quad9 cell and on a 2x2 Quad9 mesh to separate basis
  ordering from shared-DOF topology.
- Add a Q1 control test with the same setup.
- Log the first offending local coefficient vector and quadrature gradients if
  the invariant fails.

Coefficient and gradient audit:

- Verify generated VTK Quad9 point ordering against the reference Q2 basis
  order used by field interpolation and gradient evaluation.
- Confirm corner, edge, and cell-interior DOFs are gathered in the same order by
  the mesh, DOF map, field projection, and assembler.
- Add tests that reconstruct `phi=x`, `phi=y`, and `phi=y-H` exactly at Q2
  nodes and quadrature points.
- Verify the physical gradient transform uses the correct inverse-Jacobian
  convention on axis-aligned and skewed cells.

Time-integration audit:

- Instrument generalized-alpha PDE-rate initialization for level-set transport.
- Confirm `M * phi_dot = -r_non_dt` receives a zero RHS for the flat-horizontal
  null mode.
- If the RHS is zero but the field moves, inspect predictor history, prior rate
  injection, accepted-state reconstruction, and rollback behavior.

Transport and boundary-condition hardening:

- Check inflow, outflow, wall, and exact-data level-set boundary terms on
  high-order spaces.
- Keep SUPG effects separate from the base invariant by testing with SUPG both
  disabled and enabled.
- Decide whether `ConservativeDivergence` is only a diagnostic or should become
  a real conservative transport feature. If real conservation is needed, design
  a conservative level-set, VOF, or CLSVOF path rather than relying on current
  scalar advection plus global volume shifts.

Reinitialization hardening:

- Keep projection-only reinitialization labeled as a geometric repair, not a
  Hamilton-Jacobi or fast-marching signed-distance solve.
- Either implement and validate Hamilton-Jacobi and fast-marching runtime
  methods, or remove/guard those enum names from user-facing production
  controls.
- Extend projection repair beyond linear corner primitives if high-order curved
  signed-distance quality is a production requirement.

Acceptance criteria:

- The P2 flat-horizontal null mode stays roundoff-invariant under horizontal
  constant advection.
- High-order MMS transport exhibits a coherent refinement trend before being
  used to judge the free-surface boundary condition.
- Reinitialization and volume correction diagnostics clearly state whether they
  are conservative, globally volume-corrected, or only signed-distance repair.

## Workstream D: Promote Capillary Flow Only After Dedicated Validation

Goal: separate controlled capillary residual checks from production
surface-tension flow claims.

Implementation tasks:

- Make curvature ownership explicit: prescribed constant, prescribed field,
  projected field refreshed by application logic, or unknown curvature field
  solved by an equation.
- For projected curvature, define refresh timing relative to cut-context
  rebuild, nonlinear iterations, accepted steps, reinitialization, and volume
  correction.
- Record curvature residuals, fit residuals, narrow-band support, rejected
  samples, and curvature update norms.
- Decide whether the capillary residual is Picard-style with frozen curvature
  or monolithic with curvature derivatives.
- If monolithic capillarity is required, implement curvature derivatives with
  respect to level-set DOFs and add finite-difference Jacobian tests.

Validation cases:

- Static circle or sphere Laplace-pressure balance for both active sides.
- Zero-gravity capillary response with known curvature and pressure jump.
- Small-amplitude capillary or capillary-gravity wave with frequency and damping
  checks.
- Droplet or bubble benchmark if two-sided material modeling is later added.
- Refinement study for curvature, normals, pressure jump, and interface force
  convergence.

Acceptance criteria:

- Capillary tests pass with supplied/projected curvature before any raw
  level-set curvature path is considered.
- Capillary validation distinguishes static pressure balance, dynamic wave
  response, and full two-phase surface-tension flow.

## Workstream E: Keep The One-Fluid Scope Honest Or Build Two-Phase CutFEM

Goal: avoid accidental claims of a two-phase method while preserving a clear
future path if two-phase support is needed.

Minimum scope-hardening tasks:

- Keep validation text precise: "one-fluid unfitted embedded free surface" and
  "passive exterior", not "two-phase CutFEM".
- Reject or warn on controls that imply material jumps, two-sided traces,
  pressure enrichment, density/viscosity ratios, or two-phase interface laws.
- Keep unfitted Nitsche kinematic enforcement rejected until a consistent
  two-sided or embedded-boundary formulation is designed and tested.
- Ensure generated-interface docs explain the difference between an embedded
  free boundary and a two-phase internal interface.

Two-phase implementation track, if required:

- Introduce side-specific material properties and two-sided traces.
- Add jump/average operators with correct active-side orientation and pressure
  sign conventions.
- Design Nitsche or equivalent weak interface conditions for velocity,
  traction, and pressure.
- Add pressure enrichment, ghost-penalty, or pressure-robust treatment where
  needed for density/viscosity jumps.
- Implement surface-tension force through a stable curvature or
  Laplace-Beltrami formulation.
- Validate against canonical two-phase Stokes/Navier-Stokes, droplet, rising
  bubble, and capillary-wave benchmarks.

Acceptance criteria:

- Current one-fluid cases cannot be mistaken for two-phase validation.
- Any future two-phase feature has separate options, tests, and documentation.

## Workstream F: Cross-Cutting Validation And CI

Focused unit tests:

- `MovingDomainPhysics` tests for positive-side capillary sign and active-side
  invariant residuals.
- `OpenVesselExamples` tests for input controls, builder guards, supplied
  curvature, and unsupported two-phase controls.
- `LevelSetTransport` tests for Q1/Q2 residual invariants, coefficient order,
  gradient transforms, and generalized-alpha initialization.
- `LevelSetInterfaceLifecycle` tests for tangent-policy fail-closed behavior.

Integration and benchmark tests:

- Static tilted-tank equilibrium with direct interface pressure diagnostics.
- Square-tank transient smoke with clear near-wall diagnostic scope.
- Linear sloshing as a small-amplitude potential-flow regression, not strict
  viscous Navier-Stokes MMS.
- MMS traveling-interface transport after the P2 invariant is fixed.
- Capillary static and wave cases only after Workstream D is ready.

MPI and performance:

- Repeat promoted serial tests under MPI-2 and MPI-4 where the case is expected
  to support distributed generated cut domains.
- Track cut-context rebuild counts, generated-rule cache hit rates, fallback
  cells, quadrature point counts, nonlinear iterations, and time per accepted
  step.
- Keep high-order depth, backend, achieved order, and fallback diagnostics in
  every validation artifact.

Acceptance criteria:

- CI contains at least one fast unit path for every fixed bug and one slower
  validation path for every promoted physical claim.
- Validation matrix output records method scope, active side, curvature source,
  tangent policy, conservation diagnostic, and unsupported-feature guards.

## Workstream G: Documentation And Claim Cleanup

Canonical docs to keep current:

- `Code/Source/solver/FE/Docs/LevelSet.md`
- `Documentation/moving_free_surface_validation_cases.md`
- `Documentation/unfitted_level_set_free_surface_qualification_log_20260522.md`
- Case-level `README.md` files under
  `tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/`

Cleanup tasks:

- Fold any still-relevant stale review findings into this outline or the
  canonical docs listed above.
- Remove completed one-off plan files once their checklists are fully closed.
- Remove stale root-level reviews that describe old saved result artifacts or
  obsolete verifier behavior.
- Prefer regenerated validation-matrix inventory audits over static dated case
  inventory snapshots.
- Avoid keeping multiple dated outlines that disagree about the same method
  contract.

Acceptance criteria:

- New work starts from this file or from canonical implementation docs, not from
  stale dated plans.
- Deleted docs have either completed checklists, stale claims, or content
  superseded by this remediation outline.

## Suggested Execution Order

1. Fix Workstream A first. It is a narrow correctness risk with high value and
   can be covered by focused residual tests.
2. Add the Workstream C P2 null-mode residual diagnostic before broad transient
   validation. This separates level-set transport defects from boundary
   condition defects.
3. Harden Workstream B logs and fail-closed behavior so validation artifacts
   state the real tangent policy.
4. Update canonical docs and validation matrix metadata after A through C are
   implemented.
5. Promote capillary benchmarks only after Workstream D curvature ownership and
   active-side sign tests pass.
6. Treat Workstream E two-phase work as a separate project unless the product
   requirement explicitly needs two-phase CutFEM.

## Final Promotion Checklist

- [x] Positive and negative active-side capillary pressure jumps are invariant
      under the documented sign convention.
- [x] High-order refreshed-frozen cases are never described as fully
      differentiated Newton cases.
- [x] P2 level-set flat-horizontal transport is invariant under horizontal
      constant advection.
- [x] Reinitialization and volume correction claims match the actual method:
      projection-only repair and global-shift correction are not claimed as
      Hamilton-Jacobi, fast-marching, or conservative transport methods.
- [x] Capillary validation includes curvature, pressure-jump, and refinement
      evidence for the supported static supplied/projected-curvature path.
      Dynamic capillary-flow benchmarks remain intentionally unpromoted.
- [x] Two-phase features remain guarded or are implemented as a separate
      validated formulation.
- [x] Validation artifacts record active side, tangent policy, curvature source,
      transport conservation diagnostic, MPI rank count, backend, fallback
      counts, and performance for the promoted one-fluid gates. Current
      artifacts include guardrail inventory, transport refinement, serial
      solution sanity, moving/refined sanity, and MPI-2/MPI-4 tilted-square
      distributed gates. This closes the current promotion evidence for these
      cases; it is not a broad MPI scaling claim, and MPI-4 remains a
      distributed robustness check rather than a roundoff hydrostatic benchmark.

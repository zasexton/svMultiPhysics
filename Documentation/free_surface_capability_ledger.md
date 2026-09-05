# Free-Surface Capability and Evidence Ledger

## Purpose and evidence rules

This ledger indexes the implemented OOP free-surface and moving-domain scope and
the evidence that may be cited for it. It does not replace a qualification
matrix, a run manifest, or the immutable records linked below.

The current-source review for this edition used committed branch tip
`0d77e6cd1c350c20334d43021a186753231398a4` (tree
`26d8e720c6dc5490168ffad54def72c74b174aad`) on
`issue-449-modern-mesh-core`. Qualification remains attached to the source
revision in each record; a later implementation revision does not broaden an
older claim.

The following terms are normative in this ledger:

| State | Meaning |
|---|---|
| **Implemented** | A production or explicitly staged route exists in committed source. This is not a physical qualification claim. |
| **Prerequisite evidence** | A focused algebra, API, lifecycle, parity, or narrow physical check passed. The cited record's non-closure boundary still applies. |
| **Qualified scope** | A frozen matrix ran and the immutable record permits the stated claim, only in its named envelope. |
| **Unsupported** | The route is absent or must fail closed in the stated configuration. |
| **Open** | Required evidence has not run, did not pass, is unavailable, or is outside the available archived record. |

`FROZEN_BEFORE_EXECUTION` is a specification state, not evidence of a pass.
Untracked files, dirty-tree development runs, and test names without a retained
result do not establish qualification.

## Implemented capability inventory

| Capability | Physical model and FE/geometry envelope | Active side, integration, and linearization | Numerical policy and present evidence state | Source/evidence anchors |
|---|---|---|---|---|
| One-phase unfitted flow | One incompressible liquid velocity/pressure pair; prescribed exterior pressure; implicit level set; production `LinearCorner` geometry, principally affine C0 P1 | Either `LevelSetNegative` or `LevelSetPositive`; bulk terms use `dCutVolume(marker, active_side)`; free-surface terms use the one-sided generated interface | Navier--Stokes VMS/PSPG bulk route, inactive pressure support, optional velocity extension, generated-state refresh and accepted-state provenance are implemented. Broader physical qualification remains case-specific. | [one-phase capability boundary](free_surface_wp10_physical_capability_boundary.md), [FE level-set contract](../Code/Source/solver/FE/Docs/LevelSet.md), [Physics free-surface contract](../Code/Source/solver/Physics/Docs/NavierStokesFreeSurface.md) |
| Authoritative cut geometry | Physics-neutral negative/positive cut volumes, generated interface, clipped mesh boundary, interface-boundary intersection, ownership, moments, and revision identity | One authoritative snapshot supplies assembly, constraints, diagnostics, maintenance, and restart. Two-sided `dI` assembly binds minus/minus, plus/plus, and cross blocks on the same rule. | Production `LinearCorner` supports differentiated and refreshed-frozen policies on its supported cells. At the reviewed tip, 3D planar polygon interface rules support order 2; high-order backends remain separately scoped. | [FE level-set contract](../Code/Source/solver/FE/Docs/LevelSet.md), [WP-2 record](qualification_logs/free_surface_wp2_geometry_20260722_5cf65650/record.md) |
| Sharp cut exterior boundary | One-phase affine C0 P1, `LinearCorner`; clipped physical exterior boundaries | Active generated boundary and retained trace; strong and boundary-local weak/Nitsche routes; serial and MPI ownership | FSR-16 and WP-3 are qualified only in this envelope. The accepted-state symmetric-Nitsche `c*=0.25` floor is separate prerequisite evidence and does not close WP-7. | [WP-3 v6 record](qualification_logs/free_surface_wp3_sharp_boundary_v6_20260826_a73c77f4/record.md), [Nitsche v3 record](qualification_logs/free_surface_wp3_wp7_nitsche_coercivity_v3_20260824_cb6cf91a/record.md) |
| Velocity extension | One-phase scalar-P1 level-set transport with P1/Q1 velocity support; graph/band reconstruction and PDE extension routes | A PDE extension explicitly retains the opposite cut-volume side. The graph route uses the active system communicator and wall-compatible source masks. | Fixed-map algebra, construction guards, component separation, wall projection, revisions, rollback, and two-rank behavior are implemented. WP-1 is closed; the complete Q3 and D18/D38 release horizons are not. | [WP-1 record](qualification_logs/free_surface_wp1_extension_20260720_398a2477/record.md), [FE level-set contract](../Code/Source/solver/FE/Docs/LevelSet.md) |
| Level-set and conservative phase transport | Unconstrained scalar P1 H1 level set; optional mass-lumped P1 nodal liquid indicator `q` with algebraic edge-flux correction | Galerkin advection with optional SUPG and discontinuity capturing; accepted-candidate bound/Courant/wall checks; staged transport, reconciliation, correction, rollback, and publication | Transport and conservative bookkeeping are implemented. The WP-6 matrix is `FROZEN_BEFORE_EXECUTION`; the 18-point release matrix, raw component histories, momentum-flux consistency, and complete Q3 dynamics remain open. | [FE level-set contract](../Code/Source/solver/FE/Docs/LevelSet.md), [WP-6 matrix](../tests/cases/fluid/free_surface_wp6_conservative_phase_qualification_matrix.json) |
| One-phase capillarity | Constant finite nonnegative surface tension; sharp one-phase unfitted interface | Normal Young--Laplace `-gamma*kappa*n`; named projected/supplied curvature routes. The total-energy route uses kinematic area gradient, `LinearCorner`, refreshed-frozen geometry, zero filter, and interface order at least 2. | Surface stress and total-energy traction are implemented. The newer V3 balanced-force matrix is frozen at the reviewed tip but has no execution record; FSR-03, FSR-04, WP-4, and Q2 are not promoted here. | [discrete-energy method](free_surface_discrete_energy_balance_method.md), [Application binding](../Code/Source/solver/Application/Core/ApplicationDriver.cpp), [interface rule](../Code/Source/solver/FE/Interfaces/LevelSetInterfaceDomain.h), [V3 matrix](../tests/cases/fluid/free_surface_wp4_balanced_capillary_matrix_v3.json) at `0d77e6cd` |
| Unfitted contact line | One-phase sharp `LinearCorner`, order-one scalar level set, generated codimension-two wall intersection | Prescribed-angle level-set residual or dynamic Ren--E momentum line law; dynamic contact requires sharp `CutVolume`, generated wall marker, axis-aligned wall normal, normal-only strong velocity control, and Navier slip | Prescribed and dynamic laws, wall-normal validation, wetted-wall slip, accepted-stage provenance, and wall-aware maintenance are implemented. Only focused prerequisite/parity evidence exists; physical wetting and capillary-rise qualification remain open. | [Physics contact contract](../Code/Source/solver/Physics/Docs/NavierStokesFreeSurface.md), [WP-5 architecture](free_surface_wp5_contact_line_architecture.md), [kernel-parity record](qualification_logs/free_surface_wp5_specialized_kernel_parity_20260830_41361402/record.json) |
| Cut stability and aggregation | One-phase unfitted affine P1/P1 combined method with connected, disconnected, and rootless feature classification | Active cut volume, generated-boundary trace, phase-local pressure stabilization, deterministic small-cut aggregation and constraint projection | Finite topology/node-crossing and Nitsche certificate routes are implemented. WP-7 v5 remains `FROZEN_BEFORE_EXECUTION`; five manufactured/simulation rows, production-preconditioner spread, and general cut-position-independent stability remain open. | [combined P1 method](free_surface_wp7_combined_p1_method.md), [WP-7 v5 specification](../tests/cases/fluid/free_surface_wp7_cut_stability_qualification_revision_v5.json), [Nitsche v3 record](qualification_logs/free_surface_wp3_wp7_nitsche_coercivity_v3_20260824_cb6cf91a/record.md) |
| Energy accounting | One-phase liquid with prescribed exterior pressure; narrow accepted Backward-Euler, fixed mesh/cut topology envelope | Same accepted geometry/state revisions connect endpoint energy, residual work, transport coupling, dissipation, external work, and numerical work; rejected attempts publish separately | Residual-work, rejected-attempt, and fixed-topology complete-connector prerequisites pass. Nonzero maintenance, extension, pruning, aggregation, open-boundary work, topology events, generalized-alpha, and physical energy campaigns remain open. | [WP-8 architecture](free_surface_wp8_geometry_energy_architecture.md), [connector record](qualification_logs/free_surface_wp8_complete_energy_connector_20260830_41319348/record.json), [exact-node record](qualification_logs/free_surface_wp8_exact_node_topology_20260902_41703398/record.json) |
| Fitted ALE | One-phase fitted boundary with schema-2 coupled-displacement mesh velocity | Boundary-local fluid and mesh normal kinematics plus `Free`, `SmoothingOnly`, or `Prescribed` tangential mesh policy; penalty/Nitsche normal enforcement and accepted boundary provenance | Low-level operator, configuration, rejection, and velocity-projection telemetry pass. No physical fitted-ALE campaign is qualified. | [WP-9 architecture](free_surface_wp9_fitted_ale_architecture.md), [WP-9 record](qualification_logs/free_surface_wp9_fitted_ale_prerequisite_20260830_41333535/record.json) |
| Incompressible two-fluid core | Separate affine C0 P1 velocity and pressure fields per phase; Triangle3/Tetra4; one `LinearCorner` interface; positive constant phase density/viscosity; constant surface tension; fixed Eulerian mesh | Complementary cut volumes, two-sided weighted symmetric interface form, one shared pressure gauge, common-interface velocity transport, phase-local stabilization/aggregation, BlockSchur canonical field order | Core parser, fields, interface coupling, transport, telemetry, and solver envelope are staged. Capability-boundary and four stationary planar prerequisites pass. Static drops, general conservation, conditioning, moving dynamics, WP-10, FSR-08, and Q7 remain open. | [two-fluid method](free_surface_wp10_two_fluid_method.md), [capability record](qualification_logs/free_surface_wp10_capability_boundary_v5_20260901_41545273/record.json), stationary records below |

## Frozen evidence catalog

The status below is copied from each record. A superseded or prerequisite
record remains useful history but does not override a later record or close a
broader work package.

| Area and record | Exact outcome | Permitted claim and boundary | Recorded source revision |
|---|---|---|---|
| [WP-0 configuration](qualification_logs/free_surface_wp0_configuration_20260720_ffef62d3/record.md) | `PASS` | Configuration containment for 24 predeclared tests; the record explicitly included hashed local supplemental build inputs. | `ffef62d3bd7af3f125074297d6f98c81e3cd916f` |
| [WP-1 extension](qualification_logs/free_surface_wp1_extension_20260720_398a2477/record.md) | `PASS` | WP-1 closure only; two phase points do not qualify Q3 or D18/D38 horizons. | `398a24773be4c2e757aa642ce642a029f8be1381` |
| [WP-2 geometry v1](qualification_logs/free_surface_wp2_geometry_20260720_c1ffec1f/record.md) | `FAIL_METHOD` | Retained failed method epoch; never a tolerated success. | `c1ffec1f7e81e0e3784ae91a890c8f53929de0b0` |
| [WP-2 geometry v2](qualification_logs/free_surface_wp2_geometry_20260720_b6c60d9f/record.md) | `PASS` | Earlier WP-2 closure epoch; later physical gates remain separate. | `b6c60d9f0bd6ec169bc4e7fee420fa6802d2715d` |
| [WP-2 geometry v4](qualification_logs/free_surface_wp2_geometry_20260722_5cf65650/record.md) | `PASS` | Current authoritative-geometry WP-2 closure; no force-balance, wetting, transport, stability, or release closure. | `5cf65650f93faf8d6f4c264ca50d03c70daea373` |
| [WP-3 sharp boundary v2](qualification_logs/free_surface_wp3_sharp_boundary_20260725_9f7e2ded/record.md) | `PASS` | Low-level prerequisite including coupled RCR/RCRCR routing; did not close WP-3. | `9f7e2deda9eb897edf3634cc90210b8f437e9ded` |
| [WP-3 sharp boundary v6](qualification_logs/free_surface_wp3_sharp_boundary_v6_20260826_a73c77f4/record.md) | `PASS` | Closes FSR-16 and WP-3 only in the one-phase affine C0 P1 `LinearCorner` envelope. | `a73c77f44ac1741df730dc4102ac938b9b1b6922` |
| [WP-3/WP-7 Nitsche v2](qualification_logs/free_surface_wp3_wp7_nitsche_coercivity_v2_20260818_e9ae9f82/record.md) | `PASS` | Finite aggregate-trace prerequisite; no uniform method coercivity bound or WP-7 closure. | `e9ae9f8211ff8cac59bf9e128bfcd461ebeb7ff8` |
| [WP-3/WP-7 Nitsche v3](qualification_logs/free_surface_wp3_wp7_nitsche_coercivity_v3_20260824_cb6cf91a/record.md) | `PASS` | Accepted-state `c*=0.25` floor for the supported Navier--Stokes viscous/Nitsche subform; no WP-7 or Q1 closure. | `cb6cf91a090414eef020e3c30924b0b30570ed27` |
| [WP-5 compiled-kernel parity](qualification_logs/free_surface_wp5_specialized_kernel_parity_20260830_41361402/record.json) | `PASS_PREREQUISITE_NONCLOSURE` | Baked-geometry and term-weight cache parity only; capillary rise, physical Ren--E convergence, WP-5, and Q4 remain open. | Split geometry `a7298ace2a3700c72cc5f1fd0ad1043e84df3abf`; term-weight cache `1f590caa943537489222fc3367623c26319c447a` |
| [WP-8 residual work](qualification_logs/free_surface_wp8_residual_work_20260830_41312481/record.json) | `PASS_PREREQUISITE_NONCLOSURE` | Seven-channel accepted-stage residual-work prerequisite. | Parent `3ee5556efe90d0dc2c10557660c67b802f93824b` |
| [WP-8 rejected attempt](qualification_logs/free_surface_wp8_rejected_attempt_20260830_41333499/record.json) | `PASS_PREREQUISITE_NONCLOSURE` | Rejected-attempt ordering, rollback, and zero accepted contribution; no topology-jump energy qualification. | Implementation `646e947d0ce18407b0dc70f1b9783bd8978b1179` |
| [WP-8 complete connector](qualification_logs/free_surface_wp8_complete_energy_connector_20260830_41319348/record.json) | `PASS_PREREQUISITE_NONCLOSURE` | Narrow fixed-topology one-phase Backward-Euler connector; physical complete-energy record was not executed. | Parent `0fca222c04978727c6b4c1cf76132703e2b8f3ce` |
| [WP-8 exact-node topology](qualification_logs/free_surface_wp8_exact_node_topology_20260902_41703398/record.json) | `PASS_PREREQUISITE_NONCLOSURE` | Exact topology-change detection and exact rollback. Its centered circular-drop trace is an expected rejection with zero accepted steps, not static-drop qualification. | Early rejection `002dad1ba7f744f48b3f341c360b3fd76811f6a9`; rejection evidence `f7721f484bc9b626f2143190aeb7a079cda3d260`; exact-node regression `478b6dcf79b952425a5c976d156d13393c565945` |
| [WP-9 fitted ALE](qualification_logs/free_surface_wp9_fitted_ale_prerequisite_20260830_41333535/record.json) | `PASS_PREREQUISITE_NONCLOSURE` | Clean 32-test schema-2 prerequisite plus velocity-projection telemetry; no fitted physical campaign. | Clean matrix source `6213ef09988a1e364a62ab20d5cd174c57da49f6` |
| [WP-10 core boundary](qualification_logs/free_surface_wp10_capability_boundary_v5_20260901_41545273/record.json) | `PASS` | Categorical `staged_two_fluid_capability_boundary`; no two-fluid physical, WP-10, FSR-08, or Q7 closure. | `1d1a4e96e49541ab5f884371c5ca1ac3c80be94b` |
| [WP-10 constant state](qualification_logs/free_surface_wp10_constant_state_v2_20260901_41571517/record.json) | `PASS` | Exact stationary planar constant-state prerequisite only. | `0ec52a795eeab515302ab261357b678d9a1369bd` |
| [WP-10 pressure jump](qualification_logs/free_surface_wp10_pressure_jump_v1_20260901_41581461/record.json) | `PASS` | Exact stationary planar prescribed-pressure-jump prerequisite only. | `fed44f91f0e7aca24ec49ed499dc06d738e994f0` |
| [WP-10 viscous jump](qualification_logs/free_surface_wp10_viscous_jump_v2_20260901_41628821/record.json) | `PASS` | One finite low-Courant stationary affine planar viscous-traction-jump prerequisite only. | `33fdb6ac23d6c9bd5df13344b868821d7f11fcbe` |
| [WP-10 hydrostatic](qualification_logs/free_surface_wp10_two_fluid_hydrostatic_v1_20260901_41658487/record.json) | `PASS` | Stationary planar two-fluid hydrostatic prerequisite only. | `42386f345c1c06ca501727e4fd532a27192790ac` |

## Physically qualified and physically open scope

The strongest retained physical/operator closure is the WP-3 sharp exterior
boundary result within its one-phase affine C0 P1 `LinearCorner` envelope. The
WP-10 records qualify only four stationary planar prerequisites: constant-state
preservation, prescribed pressure jump, prescribed viscous-traction jump, and
hydrostatics. None is a moving two-fluid or static-drop closure.

The [historical level-set review](free_surface_level_set_review_20260713.md)
describes scoped `n=16` capillary-wave and direct prescribed-velocity P1
wall-advection results from executable SHA-256
`c7b297bfc00b6c35a0865e15244e1b38a8863c9a0719e85f6dd690098fef2522`.
It does not bind those results to a frozen record or an exact source commit,
and the linked raw payload directories are absent from the repository
qualification-log tree at the reviewed tip. This ledger therefore classifies
both results as **open/unverified for R0 comparison** and makes no qualification
claim from them. The review also retains failed static sessile/capillary gates,
an unresolved full Physics executable, and incomplete D18/D38 horizons.

The committed WP-4 V3 source adds quadratic 3D planar-interface quadrature,
collective order admission for total-energy traction, energy-adjoint and
restoring-response fixtures, and a frozen balanced-force campaign. Its matrix
status is `FROZEN_BEFORE_EXECUTION`; no immutable V3 result record exists, so
FSR-03 is still open in this ledger. Prescribed-angle scaling lanes cannot close
FSR-04, WP-4, or Q2 even if the balanced-force campaign later passes.

## Unsupported configurations

- Surface tension is a finite nonnegative literal constant. Variable surface
  tension and Marangoni traction are unsupported.
- The production one-phase contact-line route rejects high-order level-set
  fields, non-`LinearCorner` contact geometry, ambiguous/degenerate contact
  intersections, complete-wetting endpoints, and dynamic contact with
  `SmoothedIndicator`. Dynamic contact also rejects absent/conflicting normal
  control, tangential/full no-slip on the same wall, weak velocity Dirichlet
  data, non-axis-aligned walls, and nonpositive mobility or slip length.
- Fitted schema 2 rejects prescribed mesh-velocity data, missing normal policy
  or enforcement, fitted `SurfaceStress`, and fitted prescribed or dynamic
  contact. Explicit schema-1 legacy routes remain unqualified.
- The staged two-fluid route rejects non-affine or higher-order spaces, cells
  outside Triangle3/Tetra4, non-`LinearCorner` or multiple interfaces,
  time-dependent material coefficients, phase change, contact lines,
  turbulence closure, compressible gas, and noncanonical solver layouts.
- The staged incompressible two-fluid core implements pressure, inertia, and
  Newtonian viscous response for each constant-density, constant-viscosity
  phase. Interpreting one phase as gas does not qualify gas-sensitive
  applications. Compressible gas physics, including thermodynamic and
  compressibility closure, is absent; trapped-gas pressure, cushioning,
  ambient-pressure splash thresholds, aerodynamic breakup, and late
  atomization therefore remain unqualified and outside the supported scope.
- Conservative bounded transport rejects constrained or non-P1 scalar fields.
  Its local bound gate is not an invariant-domain theorem.
- Overlapping or multiple equation-level active domains have no implicit
  priority and require an explicit composition operator.
- High-order contact reconstruction, differentiated high-order regenerated
  geometry/curvature, wedges and pyramids in the high-order implicit path, and
  the `MomentFit` backend are unavailable and fail closed as documented.

## Open evidence requirements

- Execute and archive the WP-4 V3 balanced-force matrix before any FSR-03
  promotion; keep FSR-04, WP-4, and Q2 separate.
- Execute the WP-5 physical wetting, slip/width, orientation, refinement, and
  public capillary-rise comparisons.
- Execute the full WP-6 18-point phase-transport release matrix and retain raw
  component/flux histories, extension sweeps, maintenance alternatives,
  momentum consistency, and representative partition studies.
- Execute WP-7 v5, retain the five remaining prospective rows, and establish
  production-preconditioner and cut/refinement/partition behavior. No general
  coercivity or inf-sup claim exists.
- Complete the WP-8 physical Backward-Euler energy campaigns and account for
  nonzero maintenance, extension, pruning, aggregation, boundary work,
  topology events, rejected endpoints, and outer contraction before extending
  to generalized-alpha.
- Complete fitted-ALE scaling, work/history/restart, geometric-conservation,
  mesh-quality, rotation/numbering/partition, translation, shear, and sloshing
  campaigns.
- Continue WP-10 after hydrostatics with circular and spherical static drops,
  both-phase conservation and momentum reconciliation, high-ratio solver
  studies, capillary waves, rising bubbles, and Hysing comparisons.
- Implement and qualify a gas model before any gas-sensitive Q7 claim.

## Dirty and unavailable evidence boundary

The original `/home/users/zsexton/svMultiPhysics` checkout, captured in
`/scratch/users/zsexton/free-surface-refactor-20260904-905239de/baseline/original-worktree/record.json`,
contained 52 modified tracked source files and three untracked WP-10 circular
static-drop inputs. The record labels that material
`preserved_external_work_not_baseline_solver_source`; the isolated implementation
checkout uses committed `0d77e6cd` as its solver-source baseline. The untracked
matrix says `FROZEN_BEFORE_EXECUTION`, and the original checkout supplies no
clean build provenance, executed result, or immutable static-drop qualification
record. Those three files remain prospective inputs and must not be cited as a
static-drop pass.

The FE level-set notes at the reviewed tip still report a 3D `LinearCorner`
interface ceiling of order 1. Committed source at `0d77e6cd` reports order 2 and
constructs quadratic planar-polygon rules. The source behavior controls this
ledger; the notes require a later documentation correction.

# Navier-Stokes Free-Surface Notes

This note describes what the Navier-Stokes free-surface code currently does.
Method choices, open work, and the history of attempts are tracked in
`Documentation/free_surface_program_tracker.md`. Relevant decisions there:
D2 covers the capillary force route, D4 the contact-angle mechanism, and D5
the fitted ALE path.

## Surface tension

`Surface_tension` must be a finite, nonnegative literal constant. A spatially
or temporally varying coefficient would also require the tangential
Marangoni traction `grad_Gamma(gamma)`. That traction is not implemented, so
such input fails closed instead of silently omitting part of the
surface-stress divergence.

The dynamic condition on the free surface is `sigma n = -(p_ext + gamma*kappa) n`.
Here `n` is the outward normal of the liquid and `kappa = div_Gamma(n)`.
`Surface_tension_form` selects the discrete force. Every form also contributes
the exterior-pressure load `p_ext * dot(n_h, v)` on the interface.

| `Surface_tension_form` | Capillary term in the momentum residual | Scope |
|---|---|---|
| `Automatic` (default) | `SurfaceStress` on unfitted level-set interfaces; `CurvatureTraction` on fitted ALE boundaries | — |
| `SurfaceStress` (also `SurfaceEnergy`, `LaplaceBeltrami`, `Variational`) | `gamma * (I - n_h ⊗ n_h) : grad(v)` integrated over the generated interface rule (unfitted) or over the current fitted boundary. This is the Laplace–Beltrami form and the first variation of the discrete interface area. | Unfitted; fitted ALE only with `Allow_fitted_surface_stress=true` (below) |
| `CurvatureTraction` | `(p_ext + gamma*kappa) * dot(n, v)` with a supplied or projected curvature. Unfitted interfaces use `n = grad(phi)/abs(grad(phi))`; fitted boundaries use the current-geometry normal and pointwise curvature, which is zero on affine faces. | Legacy and verification; on fitted P1 boundaries it applies no capillary load (withdrawn from use, see below) |
| `GeneratedCurvatureTraction` | Same integrand, with the normal carried by the generated interface rule | Unfitted; experimental |
| `KinematicAreaGradientTraction` | `gamma * kappa_h * dot(n_h, v)`, where `kappa_h` is recovered by `KinematicAreaGradient` curvature projection (FE `LevelSet.md`). It includes the declared Young wall energies, so no separate line force is assembled. | Unfitted; see its requirements below |

The unfitted forms require `Geometry_tangent_policy=RefreshedFrozenQuadrature`.
This is not the active-cut default, so set it explicitly. It means generated
geometry, normals, and recovered curvature are refreshed between nonlinear
passes and contribute no geometry Jacobian. Differentiated generated-geometry
and shape-tangent variants are rejected for these forms. Raw pointwise
level-set curvature (`Use_level_set_curvature=true`) is rejected for unfitted
surface tension; supply `Curvature` or a projected curvature field instead.

`KinematicAreaGradientTraction` has further requirements:

- `Active_domain_method=CutVolume` and `Generated_interface_geometry=LinearCorner`;
- matching scalar affine C0 P1 `Triangle3` or `Tetra4` spaces for the level set and the curvature;
- a prescribed, projected `Curvature_field_name`, distinct from the level-set field;
- `Curvature_projection_recovery_mode=KinematicAreaGradient`;
- interface quadrature order of at least 2;
- `Curvature_projection_kinematic_area_gradient_filter_coefficient=0`;
  `Curvature_projection_kinematic_area_gradient_mass` may be `Consistent` or
  `Lumped`. With `Lumped` the filter coefficient defaults to 0 and may be
  omitted; with `Consistent` it defaults to 1 and must be set to 0.

## Semi-implicit surface tension

`Surface_tension_semi_implicit` = `None` (default) | `NormalIncrement` adds
the lagged normal-increment term of decision D13 (design note
`Documentation/free_surface_semi_implicit_surface_tension_design.md`, §3.3):

```text
R_SI(u; v) = gamma * dt_eff * int_{Gamma_h} grad_Gamma((u - u_ref).n_h) . grad_Gamma(v.n_h),
grad_Gamma(w.n_h) = P_h (grad w)^T n_h,   P_h = I - n_h ⊗ n_h.
```

- `n_h` is the generated-interface normal of the rule that carries `dI`;
  it is constant on each `LinearCorner` facet.
- `dt_eff = 1/a0` is the effective step of the time integrator:
  `0.5333 dt` for generalized-alpha with `rho_inf = 0.5`, `dt` for backward
  Euler. It is not a tuned parameter.
- `u_ref` is the prescribed field
  `ns_free_surface_semi_implicit_reference_velocity` in the velocity space.
  The application overwrites it with the velocity unknown at every
  generated-state refresh: the projected outer fixed-point, projected
  endpoint and restored synchronization points, and before each physical
  solve. The copy is a coefficient copy between two fields that share one
  DOF map.

`R_SI` is therefore zero in every freshly refreshed residual: the accepted
state and the acceptance test of the outer fixed point are those of the
scheme without the term, to within the outer tolerance. Only the Jacobian of
the frozen inner solves changes. It gains a constant, symmetric,
positive-semidefinite velocity block, which is the Laplace–Beltrami part of
the omitted geometry Jacobian. The term enters no conservative or
residual-work ledger channel, because it is zero at acceptance.

One exception: when the outer loop cycles between two cut topologies and the
step is accepted on a frozen epoch (`diagnostic=cut_topology_cycle`), the
accepted inner solution still contains `R_SI` for its last inner update.

The option fails closed outside the validated scope. It requires all of the
following:

- an exterior one-phase `UnfittedLevelSet` free surface with an active side;
- `Active_domain_method=CutVolume` with zero smoothing width;
- `Generated_interface_geometry=LinearCorner`;
- `Geometry_tangent_policy=RefreshedFrozenQuadrature`, without level-set
  shape tangents;
- a literal positive `Surface_tension`;
- `Surface_tension_form=SurfaceStress`. `KinematicAreaGradientTraction` is
  admitted as an experiment only;
- an affine P1 `Triangle3` or `Tetra4` velocity space.

The application adds three further requirements:

- a transient solve;
- the generated-state outer fixed point;
- a level set advected by the fluid velocity (`Velocity_source=coupled_field`)
  or by a PDE extension of it (`Advection_velocity_extension_method` =
  `pde_harmonic` or `pde_normal`, with either coupling). The PDE extension
  equals the fluid velocity on every vertex of the retained interface cells.

The algebraic `wall_compatible_normal` and `nearest_interface_point`
extensions, plain prescribed or constant velocities, steady solves, and runs
with the outer fixed point disabled are rejected.

Validation (design note §9): with the option off the output is bitwise
unchanged. The static drop at La = 12 passes its M2 gates at fixed steps up
to 26 times the capillary limit `dt_B`, with 3 to 3.4 outer passes per step;
at La = 120 the coarsest-level volume drift needs `dt = 0.01`. The capillary
wave runs at 25 to 100 steps per period, where the scheme without the term
fails at the first step on the finer meshes. Its time error is second order.

## Fitted ALE free surfaces

A fitted free surface (`Implementation=FittedALE`) is a boundary of the
liquid mesh. The mesh moves with a coupled displacement unknown
(`Enable_ALE=true`, `Mesh_velocity_source=coupled_displacement`) solved by a
`mesh_motion` equation. Decision D5 of the program tracker selects this path
as the independent reference for the unfitted results.

**Assembly frame.** Coupled-displacement ALE assembles the fluid, the mesh
motion and the fitted boundary terms on the trial current configuration:
the FE geometric-nonlinearity transaction moves the current coordinates at
every trial state, and the application builds the FE system with the current
configuration as its assembly frame whenever an equation requests coupled
displacement (`Enable_ALE=true` with `Mesh_velocity_source=coupled_displacement`;
`SimulationBuilder::createFESystem`, which logs
`assembly_frame=current (coupled mesh-displacement ALE)`). This holds for
every kinematic enforcement below. Fitted boundary integrals therefore use
the ordinary boundary measure of that frame; the current normal is the
normal of the same frame, and no fitted term multiplies by the current
surface measure (`NavierStokesLegacyBCs.FittedFreeSurfaceKinematicBCTranslation_UsesCurrentGeometry`).
(Until 2026-09-30 the application assembled such inputs on the reference
frame and the fitted terms multiplied the boundary weights by the current
surface measure a second time; see the tracker, M5. Decision D35 confirms
current-frame assembly as the coupled-ALE contract.)

**Kinematic enforcement.** `MeshNitsche` is the qualified default
(decision D35): a fitted free surface without `Kinematic_enforcement` uses
it. `Penalty` and `Nitsche` are legacy options and must be selected
explicitly. `None` is accepted only by the explicit schema-1 legacy mode,
which keeps its own defaults (no relation, or `Penalty` when only
`Kinematic_penalty` is given); before D35 the qualified contract rejected an
omitted key, so no accepted input changes.

| Value | Status | Fluid row on the free surface | Mesh row on the free surface |
|---|---|---|---|
| `MeshNitsche` | default | none: the fluid keeps the natural dynamic condition `sigma n = -(p_ext + gamma kappa) n` | `gamma_N / h_n * (w - u).n (psi.n)` plus the Nitsche consistency `-kappa ((grad w) n . n)(psi.n)` of the harmonic mesh-velocity operator |
| `Penalty` | legacy, explicit | `Kinematic_penalty * (u - w).n (v.n)` | `Kinematic_penalty * (w - u).n (psi.n)` |
| `Nitsche` | legacy, explicit | Nitsche row for `u.n = w.n` whose consistency term cancels the fluid normal stress | `Kinematic_nitsche_gamma / h_n * (w - u).n (psi.n)` |

With the legacy `Penalty` and `Nitsche` modes the kinematic relation is
imposed on the fluid as well as on the mesh. The fluid row then replaces
(Nitsche) or perturbs (Penalty, by the mesh stiffness flux) the normal
dynamic condition, so these two modes do not reproduce free-surface
dynamics; they remain for the existing prerequisite tests and decks, which
state them explicitly. `Kinematic_penalty` still requires an explicit
`Kinematic_enforcement=Penalty` and does not change the default.
`MeshNitsche` imposes the relation on the mesh only:

- `w = dt(d)` is the mesh velocity. The harmonic `mesh_motion` equation must
  act on it (`Harmonic_quantity=velocity`: `kappa grad(w):grad(psi)`), so the
  mesh velocity is the harmonic extension of the free-surface normal
  velocity and the displacement integrates it.
- The penalty `gamma_N / h_n` balances that operator without a time scale.
  With the P1 trace inverse inequality
  `||d_n v||_F^2 <= (2/h_n)||grad v||_T^2`, `h_n = 2|T|/|F|`, the boundary
  row is coercive for `gamma_N > 2 kappa`; the `mesh_motion` module checks it
  with a literal `Kappa`. `Kinematic_nitsche_gamma` (`gamma_N`, default 10)
  is the only numerical constant; the benchmarks fix it at 10 with `Kappa=1`
  (principle P1).
- With the consistency term the normal row reduces to the kinematic relation
  up to the P1 defect of the harmonic flux, which is `O(h^2 k^2)` relative to
  the normal velocity for a surface wavenumber `k` and does not accumulate
  over time steps. A displacement operator
  (`Harmonic_quantity=displacement`) would need a `deltat`-scaled penalty,
  and its per-step defect would accumulate into a spurious relaxation of the
  surface of rate `h^2 k^2 / (2 gamma_N deltat)`; this combination fails
  closed.
- The consistency term is added by the harmonic `mesh_motion` module on every
  boundary whose normal relation declares it. The fluid equation must
  therefore precede the `mesh_motion` equation in the input; the reverse
  order and the pseudo-elastic mesh model fail closed.
- `Kinematic_nitsche_symmetric` and `Kinematic_nitsche_scale_with_p` do not
  apply and are rejected with `MeshNitsche`.

The tangential mesh policy `Free` adds no tangential row, so free-surface
nodes slide tangentially with the harmonic extension (principle P1).

**Walls.** A fitted free surface meets a wall at a contact point (2D) or line
(3D). For the fluid, free slip on an axis-aligned wall is a `Dir` condition
with `Value 0` and `Effective_direction` selecting the wall-normal component.
The `mesh_motion` equation accepts the same input: a zero-valued `Dir`
condition with `Effective_direction` constrains only the selected
displacement components, so the mesh slides along the wall and the contact
point can move. A direction that selects every component or none constrains
all components (the previous behavior).

The wall and free-surface face files must be disjoint. A boundary face
carries one label, the name of the last face file that lists it, and every
boundary condition acts on the faces of its label; a wall face also listed
in the free-surface file becomes a free-surface face, and the wall
conditions then miss the nodes of the contact line. The application warns
when a face is listed in several face files (the legacy fitted SPHERIC
Test 10 3D deck leaked through its contact line for this reason; see the
WP-9 architecture note).

**Capillarity.** Fitted `CurvatureTraction` with
`Use_current_geometry_curvature=true` uses the pointwise curvature of the
boundary facets, which is identically zero on affine faces; it applies no
capillary load on P1 meshes and logs a warning. It is kept for curved
geometry but withdrawn from use in the fitted benchmarks. The fitted
Laplace–Beltrami form is enabled explicitly:

```xml
<Surface_tension_form>SurfaceStress</Surface_tension_form>
<Allow_fitted_surface_stress>true</Allow_fitted_surface_stress>
```

It assembles `p_ext n.v + gamma (I - n n) : grad(v)` over the current
boundary, with the current normal and the current-frame gradient. On a
regular polygon it is balanced exactly by the constant pressure
`gamma/(R cos(pi/N))`, i.e. by `gamma/R` up to `O(h^2)` (focused test
`FittedFreeSurfaceALE.SurfaceStressOnACircularDropBalancesAConstantPressure`).
The opt-in requires an explicit `Surface_tension_form=SurfaceStress`, a
literal surface tension, coupled mesh displacement (or a static mesh), and no
fitted contact-line model; without it the request still fails closed as
`fitted_surface_stress_current_frame_gradient_unqualified`.

## Unfitted contact lines

Unfitted level-set free surfaces keep contact-line behavior in the
Navier-Stokes formulation. FE provides the generated interface-boundary
intersection measure; Navier-Stokes decides when and how to use it.

`Contact_line_model` is required and selects one of:

- `None`;
- `Pinned` (fitted ALE only);
- `PrescribedAngle` (also `PrescribedContactAngle`, `ContactAngle`);
- `DynamicRenE` (also `DynamicContactAngle`, `DynamicAngle`, `RenE`).

The angle is given with `Contact_angle_degrees` or `Contact_angle_radians`
(or the `Prescribed_contact_angle_*` aliases), measured through the liquid,
and must satisfy `0 < theta < pi`. Unknown keys fail closed. That includes the
removed `Contact_angle_penalty`.

Each contact line needs a wall boundary marker. New inputs should use
`Contact_line_wall_marker` for one wall or `Contact_line_wall_markers` for a
semicolon-separated list. Mesh face-name inputs may use
`Contact_line_wall_face` or `Contact_line_wall_faces`; the application
translator resolves those names to wall markers before Physics receives the
boundary condition. Wall normals are supplied with `Contact_line_wall_normal`
or `Contact_line_wall_normals`. A plural normal list must contain either one
normal reused for every wall marker, or one normal per wall marker.

The configured normal is not trusted as geometry metadata. Whenever a
generated contact rule is refreshed, Physics maps every rule-carried boundary
normal from the parent-cell reference frame to the active physical frame. It
requires the dot product with the normalized configured normal to be at least
`1 - 1e-8`. The same check is applied directly to current-frame rules. An
opposite normal, a tilted or mismatched wall, an invalid mapping, or an invalid
codimension-two rule fails closed before assembly. If the interface does not
currently meet a configured wall, there is no contact rule to sample, and
validation is deferred until an intersection first exists.

Contact-line terms are localized to the generated interface-boundary
intersection marker. That marker is computed from the level-set source, the
generated interface domain id, the isovalue, the interface marker, and the wall
boundary marker. A user-supplied `Contact_line_marker` is accepted only if it
already matches the generated marker; otherwise configuration fails. This
avoids accidental reuse of a fitted contact-line marker or the full
free-surface interface marker.

### Conventions

The level-set normal `grad(phi)/abs(grad(phi))` points from the negative side
to the positive side. Navier-Stokes flips it when
`Active_domain=LevelSetPositive`, so `n` is always the outward normal of the
liquid. Generated-geometry forms take `n` from the generated rule. `n_wall`
points out of the liquid into the solid. Then:

```text
cos(theta_d) = -dot(n, n_wall)                   (dynamic angle)
m            = normalize(n - dot(n, n_wall)*n_wall)
V_CL         = dot(u, m)
```

Here `m` points outward from the wetted wall footprint. Young's equilibrium
condition is `dot(n, n_wall) = -cos(theta_e)`. Flipping either the level-set
normal or the active side changes the physical liquid normal and must also
change the interpreted angle.

### Momentum contact-line terms

The line terms depend on the surface-tension form. `SurfaceStress` already
supplies the dynamic conormal force `+gamma*cos(theta_d)` through its surface
integral, so its separate line term carries only the equilibrium part. With
`xi = 1/Contact_line_mobility`:

| Form | `PrescribedAngle` | `DynamicRenE` |
|---|---|---|
| `SurfaceStress` | `-gamma*cos(theta_e) * dot(v, m)` | `xi*dot(u,m)*dot(v,m) - gamma*cos(theta_e)*dot(v,m)` |
| `CurvatureTraction`, `GeneratedCurvatureTraction` | `-gamma*(cos(theta_e) + dot(n,n_wall)) * dot(v, m)` | `xi*dot(u,m)*dot(v,m) - gamma*(cos(theta_e) + dot(n,n_wall))*dot(v,m)` |
| `KinematicAreaGradientTraction` | none; the Young energy is inside `kappa_h` | `xi*dot(u,m)*dot(v,m)` |

For `DynamicRenE` this is the weak form of the Ren–E law
`xi V_CL = gamma (cos(theta_e) - cos(theta_d))`. Combining it with the
surface traction and the wetted-wall energy gives the nonnegative line
dissipation `xi*V_CL^2`.

### Navier slip on the wetted wall

The Navier wall term is sharp:

```text
(mu/Wall_slip_length) * dot(P_wall*u, P_wall*v)
    over the generated active (wetted) subset of the wall marker,
P_wall = I - n_wall ⊗ n_wall.
```

It requires `Wall_slip_model=Navier` with a positive literal
`Wall_slip_length`, literal Newtonian viscosity,
`Active_domain_method=CutVolume`, and `Active_domain_smoothing_width=0`.
The slip length is a physical input of the case.

Both unfitted contact laws require it (decision D4): the Young term in
momentum is the only contact-angle mechanism, and on a no-slip wall the
velocity test functions vanish, so that term would do no work and nothing
would impose the angle. `PrescribedAngle` and `DynamicRenE` therefore share
the wall requirements: Navier slip on the wetted wall and a stationary,
zero, normal-only strong velocity condition on an axis-aligned planar wall
whose faces carry the configured outward normal. Tangential or full no-slip
constraints and weak velocity Dirichlet data on that wall are rejected, and
so is an unfitted `PrescribedAngle` without `Wall_slip_model=Navier`.

### Level-set wall maintenance

There is no contact-angle residual in the level-set equation. The former
penalty `penalty*(dot(n,n_wall) + cos(theta))*eta` has been removed.

Decision D4 in the program tracker gives the contact angle one owner: the
Young term in the momentum equation (table above). Level-set maintenance
therefore never imposes an angle. When reinitialization is enabled, the
wall-aware projection redistancing at accepted endpoints
(`FE/LevelSet/LevelSetReinitialization.h`) treats both contact laws the same
way:

- The parent cells of every retained contact rule on a `PrescribedAngle` or
  `DynamicRenE` wall form contact patches, connected through shared
  level-set coefficients. Each patch is rescaled by one common positive
  factor, fitted to the signed distance
  (`LevelSetWallContactConstraintKind::PreserveAcceptedAngle`). The accepted
  contact point and interface normal in those cells, and so the contact
  angle, are unchanged. The declared angle is not read.
- Cells away from the contact patches relax toward signed distance as usual,
  with the zero set held within `max_zero_set_displacement`.
- `PrescribedAngle` walls take their contact rules from the accepted endpoint
  geometry snapshot. `DynamicRenE` walls take them from the accepted dynamic
  contact stage, which also supplies the redistancing input.
- The maintenance log line reports `wall_contact_maintenance=preserve_accepted_angle`
  (or `none` when no contact rule exists) and the number of contact rules per
  law (`prescribed_contact_rules`, `dynamic_contact_rules`).

With reinitialization disabled, only the optional kinematic reconciliation
(`Enable_kinematic_reconciliation`, `FE/Docs/LevelSet.md`) changes the
transported level set between steps. It does not read the angle: it moves the
discrete interface, contact cells included, so that each step's change of the
liquid area equals the interface flux of the transport velocity. The optional
sign-definite patch bounds (`Enable_sign_definite_patch_bounds`) change only
nodes away from the interface: no cut cell, contact cell or angle is touched.

The former reset of `PrescribedAngle` contact cells to a unit-gradient affine
target with the declared angle (`RepairToPrescribedAngle`) is retired as a
production path. It combined a second, geometric angle mechanism with the
Young term, and the two disagree on the discrete angle. The FE implementation
stays, for verification only, until milestone M4 has validated the single
mechanism. The application rejects that kind on every rank before production
redistancing.

### Requirements and rejected combinations

`PrescribedAngle` and `DynamicRenE` on unfitted interfaces both require:

- an active liquid side;
- `Generated_interface_geometry=LinearCorner`;
- a scalar, continuous, order-1 level-set unknown.

Fitted prescribed and dynamic contact are rejected until a fitted
codimension-two integration entity exists.

`PrescribedAngle` additionally requires the slip conditions above:
`Wall_slip_model=Navier` with a positive literal `Wall_slip_length`, literal
Newtonian viscosity, `CutVolume` with zero smoothing width, an axis-aligned
planar wall, and a stationary, zero, normal-only strong velocity condition
on that wall.

`DynamicRenE` additionally requires:

- `Active_domain_method=CutVolume` with zero smoothing width;
  `SmoothedIndicator` is rejected;
- positive literal surface tension, mobility, and slip length;
- literal Newtonian viscosity;
- an axis-aligned wall normal, because general linear-combination essential
  constraints are not available;
- a stationary, zero, normal-only strong velocity condition on the same wall.

The following fail closed on a dynamic-contact wall:

- tangential or full no-slip constraints;
- weak velocity Dirichlet data;
- duplicate dynamic entries;
- a competing contact-line model on the same wall.

These restrictions prevent double-counting wall laws and preserve the
positive wall and line dissipation signs.

The complete discretization has not been shown to satisfy a continuous or
discrete total-energy identity. Curvature recovery, frozen generated geometry,
and level-set maintenance are separate refreshed operations.

The sign and dissipation conventions follow Ren and E, “Boundary conditions
for the moving contact line problem,” *Physics of Fluids* 19 (2007),
doi:10.1063/1.2646754, and the energy-stable finite-element formulation of
Zhao and Ren, *Journal of Computational Physics* 417 (2020),
doi:10.1016/j.jcp.2020.109582.

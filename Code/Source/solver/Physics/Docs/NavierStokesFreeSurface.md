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
The discrete force is `SurfaceStress`, the only capillary form, so no key
selects it: `gamma * (I - n_h ⊗ n_h) : grad(v)` integrated over the
generated interface rule (unfitted) or over the current fitted boundary.
This is the Laplace–Beltrami form and the first variation of the discrete
interface area (decision D47 Q2(b)). It also contributes the
exterior-pressure load `p_ext * dot(n_h, v)` on the interface.

`CurvatureTraction` (`(p_ext + gamma*kappa) * dot(n, v)` with a supplied, a
projected or the raw level-set curvature) was retired by decision D47 Q2,
and `KinematicAreaGradientTraction` (`gamma * kappa_h * dot(n_h, v)` with a
curvature recovered by the `KinematicAreaGradient` projection) by decision
D48, together with the incremental PSPG pressure gradient that only it
selected. Their input keys (`Surface_tension_form`, `Curvature`,
`Use_level_set_curvature`, `Allow_fitted_surface_stress`,
`Curvature_field` and its spellings) were then removed from the input; a deck
that still writes one fails as an unknown XML element.

The unfitted form requires `Geometry_tangent_policy=RefreshedFrozenQuadrature`.
This is not the active-cut default, so set it explicitly. It means generated
geometry and normals are refreshed between nonlinear passes and contribute
no geometry Jacobian. Differentiated generated-geometry and shape-tangent
variants are rejected.

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
- `Generated_interface_geometry=LinearCorner`;
- `Geometry_tangent_policy=RefreshedFrozenQuadrature`, without level-set
  shape tangents;
- a literal positive `Surface_tension`;
- an affine P1 `Triangle3` or `Tetra4` velocity space.

The application adds three further requirements:

- a transient solve;
- the generated-state outer fixed point;
- a level set advected by the fluid velocity (`Velocity_source=coupled_field`)
  or by a PDE extension of it (`Advection_velocity_extension_method` =
  `pde_harmonic` or `pde_normal`, with either coupling). The PDE extension
  equals the fluid velocity on every vertex of the retained interface cells.

Plain prescribed or constant velocities, steady solves, and runs with the
outer fixed point disabled are rejected. (The algebraic
`wall_compatible_normal` and `nearest_interface_point` extensions were retired
by decisions D47 and D55.)

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

**Kinematic enforcement.** Every fitted free surface uses `MeshNitsche`
(decision D35), the qualified contract, with the normal relation
`(w - u).n = 0` (`MatchFluidNormalVelocity`); no key selects either.
The fluid has no kinematic row on the free surface and keeps the natural
dynamic condition `sigma n = -(p_ext + gamma kappa) n`; the mesh row is
`gamma_N / h_n * (w - u).n (psi.n)` plus the Nitsche consistency
`-kappa ((grad w) n . n)(psi.n)` of the harmonic mesh-velocity operator.

The legacy `Penalty` and `Nitsche` enforcements imposed the kinematic
relation on the fluid as well as on the mesh. Their fluid row replaced
(Nitsche) or perturbed (Penalty, by the mesh stiffness flux) the normal
dynamic condition, so they did not reproduce free-surface dynamics.
Decision D47 (Q9) retired them, and the `Kinematic_enforcement`,
`Normal_kinematic_policy` and `Kinematic_penalty` keys were later removed
from the input. `MeshNitsche` imposes the relation on the mesh only:

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
- `MeshNitsche` accepts only `Kinematic_nitsche_gamma`; the symmetry and
  order-scaling switches of the retired Nitsche enforcement were removed
  (decision D47).

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
conditions then miss the nodes of the contact line (the legacy fitted
SPHERIC Test 10 3D deck, deleted since, leaked through its contact line for this reason;
see the WP-9 architecture note). The application therefore rejects
`<Add_face>` files that share boundary faces (decision D35): the message
names the faces, their files and the number of shared boundary faces. A
legacy deck can keep the last-listed labeling with
`<Allow_overlapping_face_files>true</Allow_overlapping_face_files>` in
`<Add_mesh>`; the overlaps are then reported as warnings. The committed
unfitted legacy SPHERIC decks written by `generate_validation_meshes.py`
declare it.

**Capillarity.** The fitted form is the Laplace–Beltrami form
`SurfaceStress` (decision D47 Q2(b)); it needs no opt-in. It
assembles `p_ext n.v + gamma (I - n n) : grad(v)` over the current
boundary, with the current normal and the current-frame gradient. On a
regular polygon it is balanced exactly by the constant pressure
`gamma/(R cos(pi/N))`, i.e. by `gamma/R` up to `O(h^2)` (focused test
`FittedFreeSurfaceALE.SurfaceStressOnACircularDropBalancesAConstantPressure`).
With `gamma != 0` it requires a literal surface tension, coupled mesh
displacement (or a static mesh), and no fitted contact-line model; with
`gamma = 0` it is the exterior-pressure load alone. The pointwise
curvature of the boundary facets of the former fitted `CurvatureTraction`
(identically zero on affine faces) was retired by D42, and its key
`Use_current_geometry_curvature` was later removed from the input.

## Unfitted contact lines

Unfitted level-set free surfaces keep contact-line behavior in the
Navier-Stokes formulation. FE provides the generated interface-boundary
intersection measure; Navier-Stokes decides when and how to use it.

`Contact_line_model` is required and selects one of:

- `None`;
- `Pinned` (fitted ALE only);
- `PrescribedAngle` (also `PrescribedContactAngle`, `ContactAngle`);
- `DynamicRenE` (also `DynamicContactAngle`, `DynamicAngle`, `RenE`).

The angle is given with `Contact_angle_degrees`, measured through the liquid,
and must satisfy `0 < theta < pi`. Unknown keys fail closed. That includes the
removed `Contact_angle_penalty`.

Each contact line needs a wall. Inputs give the mesh faces with
`Contact_line_wall_face` (one name or a semicolon-separated list); the
application translator resolves them to wall markers
(`Contact_line_wall_markers`, an internal key that a deck cannot set) before
Physics receives the boundary condition. Wall normals are supplied with `Contact_line_wall_normal`, one
normal reused for every wall or a semicolon-separated list with one normal per
wall. The other spellings of these keys were removed (decision D47).

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

`SurfaceStress` already supplies the dynamic conormal force
`+gamma*cos(theta_d)` through its surface integral, so its separate line term
carries only the equilibrium part. With `xi = 1/Contact_line_mobility`:

| Form | `PrescribedAngle` | `DynamicRenE` |
|---|---|---|
| `SurfaceStress` | `-gamma*cos(theta_e) * dot(v, m)` | `xi*dot(u,m)*dot(v,m) - gamma*cos(theta_e)*dot(v,m)` |

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
`Wall_slip_length`, and literal Newtonian viscosity.
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
therefore never imposes an angle. The projection reinitialization was retired
by decision D47; the accepted-step maintenance consists of three optional,
parameter-free operations (`FE/Docs/LevelSet.md`):

- The kinematic reconciliation (`Enable_kinematic_reconciliation`) moves the
  discrete interface, contact cells included, so that each step's change of
  the liquid area equals the interface flux of the transport velocity. It
  does not read the angle.
- The sign-definite patch bounds (`Enable_sign_definite_patch_bounds`) change
  only nodes away from the interface: no cut cell, contact cell or angle is
  touched.
- The interface-preserving redistancing
  (`Enable_interface_preserving_redistancing`) replaces the level set by the
  distance to its own zero polygon, continued across the wall by rays at the
  contact points, which keeps the discrete contact angle, and restores the
  interface with the zero-velocity reconciliation.

The conservative-phase geometry reconciliation leaves the parent cells of
every retained contact rule on a `PrescribedAngle` or `DynamicRenE` wall
unchanged (`FE/LevelSet/LevelSetWallContactConstraint.h`): the accepted contact
point and interface normal in those cells, and so the contact angle, stay as
accepted. `PrescribedAngle` walls take their contact rules from the accepted
endpoint geometry snapshot, `DynamicRenE` walls from the accepted dynamic
contact stage. The constraint carries only the identity of the accepted cell;
the declared angle is not read. The former reset of `PrescribedAngle` contact
cells to a unit-gradient affine target with the declared angle
(`RepairToPrescribedAngle`) was removed (decision D4): it combined a second,
geometric angle mechanism with the Young term.

### Requirements and rejected combinations

`PrescribedAngle` and `DynamicRenE` on unfitted interfaces both require:

- an active liquid side;
- `Generated_interface_geometry=LinearCorner`;
- a scalar, continuous, order-1 level-set unknown.

Fitted prescribed and dynamic contact are rejected until a fitted
codimension-two integration entity exists.

`PrescribedAngle` additionally requires the slip conditions above:
`Wall_slip_model=Navier` with a positive literal `Wall_slip_length`, literal
Newtonian viscosity, an axis-aligned planar wall, and a stationary, zero, normal-only strong velocity condition
on that wall.

`DynamicRenE` additionally requires:

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

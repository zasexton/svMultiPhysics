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
| `SurfaceStress` (also `SurfaceEnergy`, `LaplaceBeltrami`, `Variational`) | `gamma * (I - n_h ⊗ n_h) : grad(v)` integrated over the generated interface rule. This is the Laplace–Beltrami form and the first variation of the discrete interface area. | Unfitted only; rejected on fitted boundaries |
| `CurvatureTraction` | `(p_ext + gamma*kappa) * dot(n, v)` with a supplied or projected curvature. Unfitted interfaces use `n = grad(phi)/abs(grad(phi))`; fitted boundaries use the current-geometry normal and pointwise curvature, which is zero on affine faces. | Legacy and verification |
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
`DynamicRenE` always uses it. For `PrescribedAngle` it is optional:
supply both `Wall_slip_model` and `Wall_slip_length`, or neither.

- With Navier slip, `PrescribedAngle` has the same wall requirements as
  `DynamicRenE`: a stationary, zero, normal-only strong velocity condition
  on an axis-aligned planar wall. Tangential or full no-slip constraints and
  weak velocity Dirichlet data on that wall are rejected. This is the
  configuration of decision D4.
- Without slip, no velocity condition on the contact wall is required or
  checked. The usual choice is a no-slip (full Dirichlet) wall. The velocity
  test functions then vanish on the wall, so the Young line term does no
  work and the contact line can move only through level-set transport. The
  retired geometric reset (next section) used to impose the angle in this
  case; now nothing does, and the contact angle keeps whatever the level set
  carries. Whether `PrescribedAngle` without slip should be rejected is an
  open decision.

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

With reinitialization disabled, nothing modifies the contact cells between
steps; the contact line moves only with the transported level set.

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

`PrescribedAngle` with Navier slip additionally requires the slip
conditions above: literal Newtonian viscosity, `CutVolume` with zero
smoothing width, an axis-aligned planar wall, and a stationary, zero,
normal-only strong velocity condition on that wall.

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

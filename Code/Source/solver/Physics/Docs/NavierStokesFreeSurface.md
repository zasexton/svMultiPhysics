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
  `Lumped`.

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
`DynamicRenE` always uses it. For `PrescribedAngle` it is optional:
supply both `Wall_slip_model` and `Wall_slip_length`, or neither. With
neither, the wall is no-slip.

### Level-set wall maintenance

There is no contact-angle residual in the level-set equation. The former
penalty `penalty*(dot(n,n_wall) + cos(theta))*eta` has been removed. Contact
geometry is maintained by the wall-aware projection redistancing at accepted
endpoints (`FE/LevelSet/LevelSetReinitialization.h`):

- `DynamicRenE` contact cells are rescaled by a common positive factor, which
  keeps the accepted contact point and angle unchanged.
- `PrescribedAngle` contact cells are reset to a unit-gradient affine target
  through the accepted contact point with the prescribed angle.

Decision D4 in the program tracker keeps only the momentum-side Young term
as the angle mechanism. That means Navier slip on the wetted wall, strong
no-penetration, and angle-preserving (scale-only) maintenance for
`PrescribedAngle` as well. The reset to the target angle is retired as a
production path and is removed once that configuration is validated
(milestone M4).

### Requirements and rejected combinations

`PrescribedAngle` and `DynamicRenE` on unfitted interfaces both require:

- an active liquid side;
- `Generated_interface_geometry=LinearCorner`;
- a scalar, continuous, order-1 level-set unknown.

Fitted prescribed and dynamic contact are rejected until a fitted
codimension-two integration entity exists.

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

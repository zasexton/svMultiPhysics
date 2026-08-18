# Navier-Stokes Free-Surface Notes

## Surface-stress scope

`Surface_tension` is currently restricted to a finite, nonnegative literal
constant.  The implemented traction is the normal Young--Laplace term
`-gamma*kappa*n`.  A spatially or temporally varying coefficient would also
require the tangential Marangoni traction `grad_Gamma(gamma)`, which is not yet
implemented; such input therefore fails closed instead of silently omitting
part of the surface-stress divergence.

## Unfitted Contact Lines

Unfitted level-set free surfaces keep contact-line behavior in the
Navier-Stokes formulation. FE provides the generated interface-boundary
intersection measure; Navier-Stokes decides when and how to use it.

For `Implementation=UnfittedLevelSet`, `Contact_line_model=PrescribedContactAngle`
requires a wall boundary marker for each contact line. New inputs should use
`Contact_line_wall_marker` for one wall or `Contact_line_wall_markers` for a
semicolon-separated list. Mesh face-name inputs may use
`Contact_line_wall_face` or `Contact_line_wall_faces`; the application
translator resolves those names to wall markers before Physics receives the
boundary condition. Wall normals are supplied with `Contact_line_wall_normal`
or `Contact_line_wall_normals`. A plural normal list must contain either one
normal reused for every wall marker, or one normal per wall marker.

The configured normal is not trusted as geometry metadata. Whenever a
generated contact rule is refreshed, Physics maps every rule-carried boundary
normal from the parent-cell reference frame to the active physical frame and
requires its dot product with the normalized configured normal to be at least
`1 - 1e-8`. The same check is applied directly to current-frame rules. An
opposite normal, a tilted/mismatched wall, an invalid mapping, or an invalid
codimension-two rule fails closed before assembly. If the interface does not
currently meet a configured wall there is no contact rule to sample; validation
is deferred until an intersection first exists.

The residual is localized to the generated interface-boundary intersection
marker computed from the level-set source, generated interface domain id,
isovalue, interface marker, and wall boundary marker. A user-supplied
`Contact_line_marker` is accepted only if it already matches that stable
generated marker; otherwise configuration fails. This avoids accidental reuse
of a fitted contact-line marker or the full free-surface interface marker.

The normal convention follows the active fluid domain. The level-set interface
normal is `grad(phi) / |grad(phi)|`, pointing from the negative side to the
positive side. Navier-Stokes flips that normal when
`Active_domain=LevelSetPositive` so the prescribed contact-angle residual uses
the outward normal of the configured active fluid side.

For unfitted prescribed contact angles, Navier-Stokes assembles the geometric
level-set residual on the generated contact-line marker as
`penalty * (dot(n_interface, n_wall) + cos(theta_target)) * eta`. Here
`n_interface` is the active-side level-set normal above, `n_wall` is the
configured outward fluid-to-solid wall normal after normalization,
`theta_target` is measured through the liquid, and `eta` is the level-set test
function. Thus Young's geometric condition is
`dot(n_interface,n_wall)=-cos(theta_target)`. Flipping either the level-set
normal or the active-domain side changes the physical liquid normal and must
also change the interpreted angle.

`PrescribedContactAngle` remains a static geometric regularization of the
level-set equation. It is intentionally separate from the energy-based dynamic
wall law below and requires zero mobility, no wall-slip model, and zero slip
length.

## Dynamic contact angle and wetted-wall slip

`Contact_line_model=DynamicContactAngle` selects the sharp-interface Ren--E
law. Define

```text
cos(theta_d) = -dot(n_interface,n_wall)
m = normalize(n_interface
              - dot(n_interface,n_wall)*n_wall)
V_CL = dot(u,m)
xi = 1/Contact_line_mobility
```

where `m` points outward from the liquid-wetted wall footprint. The momentum
residual on the generated contact line is

```text
xi*dot(u,m)*dot(v,m)
- Surface_tension*(cos(theta_e)+dot(n_interface,n_wall))*dot(v,m)
```

which is the weak form of
`xi V_CL = gamma (cos(theta_e)-cos(theta_d))`. The same positive
`Surface_tension` must supply the normal Young--Laplace traction on the free
surface. Combining that traction, the line term, and the wetted-wall energy
produces the expected nonnegative line dissipation `xi*V_CL^2`.

The accompanying Navier wall term is

```text
(mu/Wall_slip_length)
* H_active(phi)
* dot(P_wall*u,P_wall*v) ds(Contact_line_wall_marker),
P_wall = I - n_wall tensor n_wall.
```

`H_active(phi)` limits slip friction to the liquid-wetted footprint. This wall
factor is intentionally a diffuse approximation: its transition width is
`Active_domain_smoothing_width`, or the local `h` when no explicit width is
given. Both its value and its derivative with respect to the level-set unknown
are included in the coupled momentum Jacobian. The liquid volume and free
surface themselves remain sharp: the model requires
`Active_domain_method=CutVolume`; `SmoothedIndicator` is rejected as a diffuse
full-domain diagnostic rather than accepted as this model's production path.

The remaining requirements are an unfitted active liquid side, `LinearCorner`
generated geometry, an order-one scalar level-set unknown,
`0 < theta_e < pi`, positive literal surface tension/mobility/slip length, and
`Wall_slip_model=Navier`. Complete wetting endpoints are rejected because `m`
becomes singular.

Until general linear-combination essential constraints are available, the wall
normal must be axis aligned. The same wall marker must have one stationary,
zero, normal-only strong velocity condition. Missing normal control,
tangential/full no-slip constraints, weak velocity Dirichlet data, duplicate
dynamic entries, and a competing contact-line model on that wall fail closed.
These restrictions prevent double-counting wall laws and preserve the
positive wall/line dissipation signs.  The complete discretization has not yet
been shown to satisfy a continuous or discrete total-energy identity because
curvature projection, frozen generated geometry, and level-set maintenance are
separate refreshed operations.

The sign and dissipation conventions follow Ren and E, “Boundary conditions
for the moving contact line problem,” *Physics of Fluids* 19 (2007),
doi:10.1063/1.2646754, and the energy-stable finite-element formulation of
Zhao and Ren, *Journal of Computational Physics* 417 (2020),
doi:10.1016/j.jcp.2020.109582.

# Free-surface discrete-energy balance method

## Scope

This record selects and derives the one-phase capillary pressure method used
for the WP-4 implementation. It does not qualify a static cap, close FSR-03 or
FSR-04, or establish a two-phase pressure-jump method.

The selected method is:

> Assemble the unprojected first variation of the snapshot-owned discrete
> liquid--gas and wetted-wall energy, pair it with the active-volume pressure
> adjoint, and construct static initial geometry as a stationary point of that
> same discrete energy at fixed discrete liquid volume.

This is the existing `SurfaceStress` operator family with a stricter geometry
and qualification contract. The production surface load is not projected into
the pressure range.

The owning free-surface declaration records
`DiscreteEnergyVolumeStationarity` together with `PrerequisiteOnly`.
Curvature-traction declarations remain `Unselected`. The qualification value
must not change to `Qualified` until the complete gate set below has passed.

## Discrete functional and signs

For one authoritative geometry snapshot, define

\[
E_h(\phi_h)
  = \gamma A_{lg,h}(\phi_h)
    - \sum_w \gamma\cos(\theta_{e,w})A_{sl,h,w}(\phi_h)
\]

and

\[
C_h(\phi_h)=V_h(\phi_h)-V_{h,0}.
\]

All measures and variations use the same retained, rank-owned snapshot rules.
Ghost rules do not contribute. A wall without a prescribed equilibrium angle
has no Young-energy coefficient.

For an admissible physical deformation \(z_h\), the current weak forms use

\[
D E_h[z_h]
 = \gamma\int_{\Gamma_h}
      (I-n_h\otimes n_h):\nabla z_h
   - \sum_w\gamma\cos(\theta_{e,w})
      \int_{\partial\Gamma_{h,w}}z_h\cdot m_{h,w}
\]

and

\[
D V_h[z_h]
 = \int_{\Gamma_h}z_h\cdot n_{l,h}.
\]

With liquid pressure \(p_l\) and prescribed exterior pressure \(p_{ext}\), the
constant-pressure part of the momentum residual is

\[
-(p_l-p_{ext})D V_h[z_h].
\]

Therefore the scalar multiplier in

\[
E_h+\lambda_h V_h
\]

uses

\[
\lambda_h=-(p_l-p_{ext}).
\]

This sign mapping is explicit because the functional stores a plus
`\lambda_h V_h` term while the momentum pressure form stores
`-p_h div(v_h)`.

## Compatible trace contract

Let \(\mathcal V_h\) be the constrained momentum test space after strong and
weak wall policies are applied. The static geometry constructor must solve

\[
D E_h[z_h]+\lambda_h D V_h[z_h]=0
\quad\text{for every }z_h\in\mathcal V_h
\]

together with \(C_h=0\), using the same:

- geometry snapshot and revision;
- active liquid side;
- generated interface and contact rules;
- momentum trace basis and constraints;
- liquid-volume rules; and
- surface, wall, and exterior-pressure sign conventions.

The termination residual is the assembled conservative pressure-plus-energy
operator on the unconstrained velocity rows. Merely reducing an optimizer's
parameter-space gradient is insufficient when its geometry parameters span a
smaller space than the momentum trace.

Constants belong to the supported one-phase pressure space. At a discrete
stationary geometry, the multiplier pressure therefore supplies an exact
pressure-range representative of the surface and Young-wall load, up to the
declared algebraic tolerance. A sampled continuum circle, sphere, or sessile
cap is only an initial guess; it is not assumed to satisfy the discrete
stationarity equation.

The nonlinear diagnostic evaluates two distinct distances from the same
unmodified assembled surface/Young load. The existing LSQR result measures
distance to the complete constrained pressure range. The
`constant_pressure_kkt_*` result separately forms the physical unit-pressure
trace, computes the closed-form best constant pressure jump, records the
opposite-signed functional volume multiplier, and measures the remaining KKT
residual. It fails closed if the scalar pressure basis does not reproduce the
unit field with unit coefficients, if an affine pressure constraint does not
preserve that field, or if the mixed pair maps the trace outside the
constrained velocity-test rows. This calculation is diagnostic only:
`constant_pressure_kkt_force_projection_applied=0`.

An accepted static state can require this stricter result through
`accepted_static_constant_pressure_kkt_max_relative_distance` or the
corresponding `SVMP_NS_FREE_SURFACE_CONSTANT_PRESSURE_KKT_MAX_RELATIVE_DISTANCE`
environment setting. The convergence gate is independent of the full
pressure-range gate: it rejects an unavailable, nonfinite, or out-of-range
constant-pressure result even when LSQR shows that a nonconstant pressure can
represent the load exactly. In an external-state fixed point, both configured
gates are deferred to the final zero-update certificate. Their enablement and
threshold values must agree bit-for-bit across the active communicator. No
gate changes the geometry, pressure field, or production force.

## General pressure identity

The constant multiplier identity does not define the variation of a general
finite-element pressure field. For a pressure field materially transported by
the deformation map \(T_t=I+t z_h\),

\[
\left.\frac{d}{dt}\right|_{t=0}
  \int_{\Omega_{h,t}}p_{h,t}\,dx
  =\int_{\Omega_h}p_h\nabla\cdot z_h\,dx.
\]

The momentum pressure virtual work is the negative of the right-hand side.
This identity is validated separately with nonconstant pressure and
deformation fields. It must not be replaced by a scalar product such as
`p * V_h`.

## Static-geometry construction contract

The static initializer must be transactional and deterministic:

1. Start from a declared analytic or previously accepted geometry and target
   volume.
2. Freeze the active topology for each trial step. Reject a step that changes
   topology unless the outer algorithm deliberately opens a new topology
   epoch.
3. Evaluate \(E_h\), \(V_h\), and their directional derivatives from one
   immutable candidate snapshot.
4. Solve the volume-constrained stationarity problem without changing the
   production capillary load.
5. Rebuild and publish the candidate snapshot only after all ranks accept the
   same step and revision.
6. Require both the volume constraint and the assembled momentum-trace KKT
   residual to satisfy declared absolute and scaled tolerances.
7. On any failure, restore geometry, field values, revisions, and publication
   state exactly.

Pressure initialization is a separate one-shot operation after geometry
acceptance. It may recover the compatible multiplier pressure, but it may not
hide an incompatible geometry by projecting away the surface-load remainder.

`minimizeLevelSetStaticCapillaryEquilibrium` implements the output-only
fixed-topology optimization kernel for this contract. Its evaluator supplies
globally reduced energy, volume, snapshot revision, a candidate-stable
combinatorial topology key, a deterministic constrained-trace fingerprint, and
unprojected constant-pressure KKT data for each trial coefficient vector. The
topology key deliberately excludes candidate source-value and snapshot
revisions, which are expected to change as the interface moves inside one
topology epoch. The constraint key changes with slave/master relations,
weights, or prescribed values. Central differences and line-search trials
must preserve both keys. The finite-difference reference scale and minimum
step have coefficient units, while the declared projected-gradient inverse
stiffness has coefficient-squared per energy units; this keeps the proposed
tangent step dimensionally distinct from the gradient norm. The kernel uses
fixed-topology central differences and a volume-constrained merit line search;
topology- or constraint-changing trials are rejected. It assigns the
candidate output only after the volume and parameter-gradient gates pass and
a fresh acceptance-certificate evaluation of that exact candidate reproduces
the functionals and passes both absolute and relative physical KKT gates. The
expensive KKT calculation is not requested for finite-difference or
line-search trials. Any evaluator that reports production-force projection is
rejected.

The application adapter is explicitly enabled on the level-set equation with
`Enable_static_capillary_equilibrium_initialization`. The initial
authoritative liquid volume becomes the immutable target. For each trial, the
adapter reconstructs the full FE state from an immutable baseline, installs
the trial level-set coefficients, rebuilds the staged cut context, distributes
the rebuilt affine constraints, and repeats until geometry and the constraint
fingerprint form a fixed point. Energy and volume are then evaluated
from the staged authoritative snapshot. The final exact-candidate certificate
uses an output-only initial-residual probe of the unmodified production
surface/Young load and physical constant-pressure trace; ambient pressure
initializer settings are not read by that probe. The acceptance record emits
the best constant pressure jump, its opposite-signed physical volume
multiplier, the parameter-space multiplier, and their difference without
turning that comparison into an unratified pass threshold.

The present objective and certificate do not include gravitational potential,
kinetic, inertial, advective, or viscous contributions. Before evaluating the
initial functional or opening a geometry transaction, the application adapter
therefore requires a present active-volume energy declaration with finite,
exactly zero gravitational acceleration, requires its velocity binding to be a
finalized dimension-compatible unknown vector volume field, and requires every
current velocity coefficient to be finite and exactly zero. Loaded or moving
equilibria remain unsupported until those contributions are part of the
minimized functional and its physical certificate.

Topology and constraint fingerprints, snapshot revisions, energy, volume,
certificate availability, and physical KKT scalars are communicator-wide
decision data. Their explicit consistency checks follow collective rejection
paths; history staging also reaches collective consensus before geometry
publication.

The first implementation intentionally requires one authoritative cut-volume
request and one matching prerequisite-only free-surface functional
declaration. It stages the certified level-set state in the current and
history slots immediately before the final geometry commit, preserves other
fields in prior-history and rate vectors, zeros only the level-set rate
slices, and restores the original history and staged geometry on any
prepublication failure. The full control set uses the
`Static_capillary_*` level-set parameters and participates bit-for-bit in the
collective maintenance-request schedule.

This establishes an executable construction path, not a static-cap
qualification result. The two- and three-dimensional cap matrix, accuracy
thresholds, cut/rotation/MPI sweeps, and analytic-convergence evidence remain
open.

## Alternatives considered

### Pressure-range force projection

Rejected. Replacing the surface load by its least-squares pressure-range
projection guarantees a small reported distance by construction and can
remove physical non-equilibrium forcing from a moving interface.

### One-phase pressure enrichment alone

Not selected for the current method. The supported liquid pressure space
already contains constants, so enrichment does not make a sampled
nonstationary discrete cap satisfy the surface-energy KKT equation. Interface
pressure enrichment remains a separate requirement for future two-phase jump
models.

### Independent curvature projection

Not selected. A curvature field assembled independently of the exact discrete
area and wall-energy variation can break the common energy identity unless a
new compatible mixed formulation and stability argument are supplied.

### Fitted or parametric interface replacement

Deferred as a distinct method family. It can provide strong equilibrium
properties, but replacing the implicit interface representation would also
change transport, topology, contact ownership, and remeshing contracts.

## Qualification boundary

The method is not qualified until all of the following are demonstrated:

- a converged discrete constrained minimizer in two and three dimensions;
- scaled-roundoff balance for exactly representable flat fields;
- pressure-range distance and assembled force residual for every static case;
- pressure jump, contact angle, base radius, apex height, volume, kinetic
  energy, and parasitic capillary number over the required refinement matrix;
- invariance under liquid-side reversal, wall rotation, cut offset, pressure
  gauge, and positive level-set rescaling;
- independent mesh, time-step, and maintenance-cadence studies;
- serial and distributed equivalence where distributed geometry maintenance
  is supported; and
- moving-interface tests showing that the unprojected physical restoring load
  remains present away from equilibrium.

Until those gates pass, the method selection and low-level identities are
implementation prerequisites rather than a WP-4 closure claim.

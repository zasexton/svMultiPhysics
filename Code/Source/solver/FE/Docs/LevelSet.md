# FE Level-Set Services

`svmp::FE::level_set` owns reusable level-set infrastructure. The namespace is
physics-neutral: it knows about FE systems, fields, function spaces, generated
interfaces, cut-cell integration data, and diagnostics, but it does not encode
Navier-Stokes, material, contact-line, or free-surface policy.

Physics modules remain responsible for equation-specific defaults and
formulation-specific residuals. The application translator owns legacy
level-set equation input compatibility, translates those choices into
`FE::level_set` options, and calls the FE services directly.

## Public API

The stable entry point is the aggregate header:

```cpp
#include "FE/LevelSet/LevelSet.h"
```

Use specific headers when compile time or dependency clarity matters:

| Header | Public surface |
|--------|----------------|
| `FE/LevelSet/LevelSetOptions.h` | option structs, source enums, cadence helpers |
| `FE/LevelSet/LevelSetTransport.h` | `installLevelSetTransport()` |
| `FE/LevelSet/LevelSetInterfaceLifecycle.h` | generated interface domain creation and update lifecycle |
| `FE/LevelSet/LevelSetVolume.h` | cut-cell volume measurement and global-shift correction |
| `FE/LevelSet/LevelSetReinitialization.h` | signed-distance repair from generated interface geometry |
| `FE/LevelSet/LevelSetDiagnostics.h` | scalar diagnostics for volume and signed-distance quality |
| `FE/LevelSet/LevelSetRestart.h` | restart records for level-set fields and generated interfaces |

APIs in `FE/LevelSet` are public FE APIs unless they are in an anonymous
namespace or marked internal in the header. Keep new reusable level-set
algorithms in this directory and expose them through the smallest header that
matches their scope.

## Free-Surface Consumers

Navier-Stokes owns free-surface boundary semantics. Fitted ALE free surfaces use
the boundary marker and moving-mesh quantities declared by the Navier-Stokes
module. Unfitted free surfaces consume `FE::level_set` generated interfaces,
level-set curvature helpers, and cut-cell metadata, then install
Navier-Stokes-specific pressure-jump, surface-tension, kinematic, and
stabilization terms on the resulting interface domain.

Generated level-set cut rules are reference-frame FE data. Assemblers map the
retained cut-volume and cut-interface quadrature through the parent-cell
geometry before evaluating physical kernels. Application wet-volume diagnostics
may additionally integrate those reference rules against the parent-cell
Jacobian to report a physical wet measure, but that diagnostic mapping does not
change the FE cut-rule contract.

The application level-set equation translator is an input adapter. It preserves
legacy XML names, builds FE options, and calls
`FE::level_set::installLevelSetTransport()`. It must not contain reusable
transport, volume, reinitialization, diagnostics, or restart algorithms.
The application also owns the optional nearest-active-vertex velocity extension
for prescribed-data level-set advection; reusable PDE-based velocity-extension
algorithms should be added to `FE::level_set` before being exposed as general
transport services.

## Equation-Level Active Domains

An equation-level active-domain request means that volume forms owned by that
equation are integrated on `dCutVolume(marker, side)` instead of the parent
cell measure. The selected side is the material support of that equation. The
opposite side is omitted from that equation unless another explicit request or
coupling term consumes it. Exterior boundary conditions keep their existing
mesh-boundary semantics; generated-interface fluxes, tractions, and coupling
terms must be installed explicitly on the generated interface.

The application active-cut request carries the retained-volume-side policy.
The default policy is active-only, matching one-fluid active-domain assembly.
An explicit inactive-side consumer, such as the Navier-Stokes unfitted
free-surface PDE velocity-extension term, upgrades the request to retain both
active and inactive generated cut-volume rules. This does not make the physical
Navier-Stokes volume terms two-phase; those terms still select
`dCutVolume(marker, active_side)`, while the extension term selects the
opposite side explicitly. When this PDE velocity extension is enabled,
Navier-Stokes does not install the otherwise-default inactive-side homogeneous
velocity clamp, because that clamp would overconstrain the extension field.
The inactive pressure clamp remains part of the one-fluid active-domain support
policy.

The equation semantics are:

| Equation family | Active-domain meaning |
|-----------------|-----------------------|
| Heat | Temperature capacity, diffusion, source, and volumetric stabilization terms are restricted to the active side. Generated-interface heat flux or contact terms are explicit interface terms, not automatic boundary conditions. |
| Darcy | Pressure/flux storage, permeability, source, and stabilization terms are restricted to the active side. Inactive-side permeability is not sampled by the Darcy equation unless a complementary equation request is installed. |
| Solid mechanics | Displacement, inertia, internal-force, body-force, and stabilization terms are restricted to the active side. Generated-interface tractions, contact, and material-interface terms are explicit interface terms. Outer mesh constraints still apply only where the constrained field is active. |
| Reaction/scalar transport | Scalar mass, advection, diffusion, reaction, source, and stabilization terms are restricted to the active side. Advecting velocity or material fields are sampled according to their own field scopes; they do not activate scalar DOFs on the inactive side. |
| FSI and coupled modules | Each physics owns its active-domain request. Coupling between active and inactive/complementary sides is expressed by explicit generated-interface or shared-field coupling operators. The active-domain service does not infer two-sided FSI coupling from two volume restrictions alone. |

Current production semantics are intentionally fail-closed for ambiguous domain
composition. A single equation-level request may select one generated domain and
one side. Multiple requests, union/intersection/difference composition, nested
level sets, or overlapping generated domains require an explicit composition
operator before they can be used as production active-domain evidence. There is
no implicit priority order for overlapping generated domains: no marker,
registration order, or equation order wins by default. Ambiguous overlap is a
configuration error unless the formulation declares the composition explicitly.

## Field Synchronization Contract

Mesh-vertex data can be injected into FE coefficients only when the target field
uses a nodal Lagrange or serendipity basis whose `EntityDofMap` bindings cover
the mesh geometry vertices and high-order edge, face, and cell-interior points.
`FESystem::projectMeshVertexValuesToFieldCoefficients()` assigns those
coefficients through entity-aware bindings and fails closed for unsupported
layouts. Modal hierarchical and other non-nodal scalar H1/product bases use a
local interpolation solve when each cell exposes one mesh point per scalar FE
DOF and the basis interpolation matrix is nonsingular; shared DOFs must receive
consistent values from adjacent cells. If those conditions are not met, the
projection fails closed rather than assuming nodal coefficient meaning.

FE-field output back to mesh vertices follows the opposite contract. Nodal
Lagrange and serendipity fields may use the direct entity-aware fast path.
Non-nodal fields, including hierarchical H1 bases, must be interpolated by
locating each mesh vertex in physical space and evaluating the FE basis at the
found reference coordinate. If point location or evaluation fails, output fails
closed instead of treating modal coefficients as nodal values.

## Two-Sided Generated Interfaces

Generated-interface orientation is tied to the level-set convention. The
negative side is `phi < isovalue`, the positive side is `phi > isovalue`, and
the interface normal is `n = grad(phi) / |grad(phi)|`, pointing from the
negative side toward the positive side. A one-sided embedded-boundary term may
choose one retained volume side and treat the generated interface as the
boundary of that side. A two-sided CutFEM term must bind both retained sides and
evaluate side-specific fields at the same generated interface quadrature point.

Two-sided traces use the following definitions:

| Symbol | Generated-interface meaning |
|--------|-----------------------------|
| `u+` | Trace of `u` from the positive retained side. |
| `u-` | Trace of `u` from the negative retained side. |
| `jump(u)` | `u+ - u-` for scalar and component-wise vector/tensor fields unless a formulation explicitly asks for a normal jump. |
| `avg(u)` | Arithmetic average `0.5 * (u+ + u-)`. Weighted averages are a formulation-specific extension and must declare their weights. |
| `n+`, `n-` | `n+ = n` and `n- = -n`, where `n` is the generated-interface normal above. |

Side ownership is a parent-cell binding, not a persistent surface mesh
ownership. The generated rule owns a parent cell, a stable generated-domain id,
the negative and positive retained-volume side records, and the interface
quadrature record. In MPI, the rank that owns the parent cell owns the generated
interface contribution unless a future two-sided coupling API explicitly
redistributes it. Shared facets and ghosted neighbors may provide field values,
but they do not change the generated-interface orientation convention.
`LevelSetInterfaceDomain::twoSidedParentCellBindings()` exposes this binding
contract by pairing each active interface fragment with the active negative and
positive retained-volume region stable ids on the same parent cell.

Generated interfaces do not use fitted mortar field scopes by default because
the surface is regenerated from a level-set field and is not a persistent
interface mesh with its own DOF numbering. Two-sided coupling should therefore
use a generated-interface field scope keyed by generated-domain id, marker,
parent cell, side, and rule stable id. If a future formulation needs a mortar
space, it must explicitly project from the regenerated quadrature/interface
records into that mortar space and define the projection/rebuild lifecycle.

The generated-interface field-scope API is side-specific. A field lookup on a
generated rule is identified by `{domain_id, marker, parent_cell, rule_id, side,
field_id}` and evaluates the requested field from the retained parent-cell side.
Two-sided forms bind both `side=positive` and `side=negative` at the same
generated quadrature point before applying trace, jump, or average operators.
One-sided embedded-boundary forms bind only the retained active side. If a
requested field is not available on the requested side, assembly must fail
closed instead of borrowing the opposite-side value.
`CutInterfaceFieldEvaluation` provides the local H1 helper for this contract:
`evaluateH1TwoSidedFieldValuesOnFragment()` and
`evaluateH1TwoSidedFieldGradientsOnFragment()` validate a
`GeneratedInterfaceTwoSidedBinding`, evaluate negative- and positive-side
parent-cell fields at the same generated quadrature points, and return
`minus`, `plus`, `jump`, and `average` values or gradients.

Generated-interface assembly dispatch separates one-sided embedded-boundary
terms from two-sided `dI` terms. One-sided generated-interface kernels continue
to assemble through the retained parent-cell embedded-boundary context.
Two-sided generated-interface kernels are routed through paired parent-cell
minus/plus contexts for the same generated quadrature rule, and the assembler
inserts minus-minus, plus-plus, minus-plus, and plus-minus blocks into the
parent-cell DOF block. `CutIntegrationContext` preserves the
`GeneratedInterfaceTwoSidedBinding` records imported from generated interface
domains, and two-sided generated assembly fails closed if the selected rule has
no complete minus/plus parent-cell binding. This provides the form-kernel
dispatch contract for parent-cell H1 fields; production two-phase formulations
must still declare the side-specific field availability and MPI ownership rules
they require.

Generated cut-interface material state is stored separately from fitted face
state and parent-cell volume state. `MaterialStateProvider` keys this storage by
the generated rule's `cut_topology_revision`, and
`StandardAssembler::assembleCutInterfaces()` binds that state view when a
generated-interface kernel requests `RequiredData::MaterialState`.
The state remains attached to the generated fragment identity across repeated
assemblies; if the generated topology is rebuilt with a new revision, callers
must rebuild or re-register material state storage before assembling stateful
interface kernels.

## High-Order Implicit Generated Interfaces

Generated interfaces default to the production `LinearCorner` path. Application
active-cut requests using the default `LinearCorner` geometry/backend also
default to `DifferentiatedQuadrature`, which publishes first-order geometry
sensitivity metadata for moving cut-volume/interface tangents. The current
high-order implicit path is an experimental FE/OOP feature for generated
level-set interfaces and remains on `RefreshedFrozenQuadrature` unless a caller
explicitly requests otherwise. Enable high-order only for validation or
explicitly qualified cases:

```xml
<Generated_interface_geometry>HighOrderImplicit</Generated_interface_geometry>
<Implicit_cut_quadrature_backend>Auto</Implicit_cut_quadrature_backend>
<Implicit_cut_fallback_policy>Fail</Implicit_cut_fallback_policy>
<Geometry_tangent_policy>RefreshedFrozenQuadrature</Geometry_tangent_policy>
```

Accepted aliases are intentionally broad for input compatibility. The canonical
names used in new inputs are:

| Control | Values / meaning |
|---------|------------------|
| `Generated_interface_geometry` | `LinearCorner` or `HighOrderImplicit` |
| `Implicit_cut_quadrature_backend` | `LinearCorner`, `SayeHyperrectangle`, `HighOrderSubcell`, `Auto`, `MomentFit` |
| `Implicit_cut_fallback_policy` | `Fail` or `LinearCorner` |
| `Geometry_tangent_policy` | `RefreshedFrozenQuadrature` or `DifferentiatedQuadrature`; application active cuts default to `DifferentiatedQuadrature` only for `LinearCorner` geometry with the `LinearCorner` backend, and high-order requests default to `RefreshedFrozenQuadrature` |
| `Implicit_cut_root_tolerance` | signed-level-set residual tolerance for root polishing |
| `Implicit_cut_root_coordinate_tolerance` | reference-coordinate convergence tolerance for root polishing |
| `Implicit_cut_root_max_iterations` | root-polishing iteration cap |
| `Implicit_cut_max_subdivision_depth` | branch/topology isolation cap honored by recursive high-order backends, not a production accuracy dial |

Backend support is cell-family specific:

| Backend | Supported generated-interface families | Current achieved interface / volume order | Current status |
|---------|----------------------------------------|--------------------------------------------|----------------|
| `LinearCorner` | 2D triangles/quads, 3D tetrahedra | 2D 5 / 5 for generated planar rules; 3D 1 / 2 | production linear fallback path |
| `SayeHyperrectangle` | 2D quads and 3D hexes | 2D 5 / 5; 3D 1 / 2 | experimental high-order milestone |
| `HighOrderSubcell` | 2D triangles and 3D tetrahedra | 2D 5 / 5; 3D 2 / 2 for the qualified tetra sphere-cap fixture | experimental high-order milestone |
| `Auto` | per-cell dispatch: quads/hexes to `SayeHyperrectangle`, triangles/tetrahedra to `HighOrderSubcell` | inherits the selected backend | experimental mixed supported-mesh dispatch |
| `MomentFit` | none | unavailable | unavailable fail-closed driver |

Wedges and pyramids are not supported by the high-order implicit path. `Auto`
fails closed for those cells until a native or decomposed rule is implemented
and qualified.

The recursive high-order backends honor the configured
`Implicit_cut_max_subdivision_depth` directly; diagnostics include
`max_depth_limit=<value>` so runs can be audited for the actual cap in force.
Increasing this value can grow rule generation cost rapidly and should be used
as a topology-isolation probe or case-specific robustness setting, not as a
substitute for convergence studies.

Explicit global high-order backends are validated before generation. For
example, `SayeHyperrectangle` on a mixed quad/triangle mesh fails before
building rules because triangles are not supported by that global backend. Use
`Auto` for mixed supported meshes. If `Implicit_cut_fallback_policy` is
explicitly set to `LinearCorner`, an implemented high-order backend may fall
back only for cell families supported by the linear cutter; fallback cells are
counted and stamped into rule provenance. `MomentFit` remains unavailable and
does not become valid through fallback.

`MomentFit` is not required for the current production-general high-order
roadmap. The promoted path is family-specific dispatch through
`SayeHyperrectangle` for quadrilateral/hexahedral cells and `HighOrderSubcell`
for simplex cells, with unsupported families failing closed. A future
formulation may still introduce a moment-fit backend, but it must arrive with
moment-system construction, conditioning diagnostics, a positive-weight policy,
and exactness/robustness tests before it can be selectable.

For high-order qualification runs, keep
`Implicit_cut_fallback_policy=Fail` and require zero fallback cells. Use
`LinearCorner` fallback only for intentional diagnostic or compatibility runs:

```xml
<Generated_interface_geometry>HighOrderImplicit</Generated_interface_geometry>
<Implicit_cut_quadrature_backend>SayeHyperrectangle</Implicit_cut_quadrature_backend>
<Implicit_cut_fallback_policy>LinearCorner</Implicit_cut_fallback_policy>
```

`RefreshedFrozenQuadrature` refreshes generated geometry and quadrature when
the active cut context is rebuilt, but treats quadrature weights beyond the
measure term, curvature, and topology transitions as fixed during tangent
assembly. It is a quasi-Newton policy. `DifferentiatedQuadrature` is accepted
only for `LinearCorner` geometry using the `LinearCorner` backend; it publishes
parent level-set DOFs and quadrature-sample shape metadata for first-order
cut-volume, interface point-location, measure, normal, and quadrature-weight
shape terms away from topology transitions. High-order regenerated-rule
sensitivities, curvature sensitivities, and topology-transition derivatives
remain unavailable and fail closed. Backend capabilities expose this distinction
with `supports_refreshed_frozen_quadrature` and
`supports_differentiated_quadrature`; unsupported families advertise neither.

### Cut-Geometry Sensitivity Contract

Differentiated generated-geometry rules are derivatives with respect to the
scalar level-set FE coefficients that define the active cut. A backend may mark
a rule differentiated only when every generated interface point, volume point,
weight, measure, normal, and volume-fraction contribution used by assembly has
the same active level-set DOF list and a derivative value for each listed DOF.
The DOF list is field-local in the parent FE space and is published through
`CutGeometrySensitivityMetadata::parent_geometry_dofs`. Each sensitivity sample
also carries `influencing_parent_geometry_dofs`, so a cut point, interface
sample, or retained-volume sample can be audited without assuming that every
future backend uses the complete parent-cell list for every point.

The contract is local to a fixed cut topology. It applies only when roots are
regular, the selected backend reports finite residuals, no branch is born or
deleted, no retained volume switches between empty/partial/full, and no
fallback changes the construction policy. Any such event is nonsmooth for a
Newton tangent; the backend must report a topology-transition diagnostic and
must not advertise complete differentiated-quadrature support for that rule.

For interface rules, point-location derivatives are derivatives of the parent
reference coordinate. Assembly maps those derivatives through the parent-cell
geometry before applying spatial-coordinate terms. Interface-measure and
quadrature-weight derivatives are derivatives of the exact scalar factor that
multiplies the integrand at that point. Interface normals use the convention
`n = grad(phi) / |grad(phi)|`, and their derivative is the tangential projection
of the perturbed level-set gradient divided by `|grad(phi)|`. Generated
cut-volume rules on the negative side use the same normal convention; positive
side volume rules use the opposite embedded normal in assembly metadata.

For volume rules, quadrature-point derivatives are derivatives of the retained
volume point in parent reference coordinates. Volume-measure, per-point weight,
and volume-fraction derivatives must be consistent with each other: summing
per-point weight derivatives over a rule must reproduce the rule-measure
derivative, and summing retained side measure derivatives over both sides of a
parent cell must cancel when parent-cell geometry is fixed. Full-cell rules have
zero cut-geometry derivatives unless the backend reports that the full/partial
classification itself is changing, which is a topology event rather than a
smooth derivative.

## Curvature and Capillary Contract

The production unfitted capillary model is a named scalar curvature field used by
the surface-tension form. The field may be either an unknown FE field or a
prescribed projected field maintained from the level-set geometry. Raw
pointwise `meanCurvatureFromLevelSet(phi)` is a diagnostic path for fitted or
controlled studies; it is not production evidence for high-order unfitted
surface tension.

For unfitted generated interfaces, curvature inputs are signed with the
generated-interface normal `grad(phi) / |grad(phi)|`, pointing from the negative
side to the positive side. Navier-Stokes converts that scalar to the outward
normal of the configured active fluid side before forming
`(-p_ext + gamma*kappa)n`. Therefore `Active_domain=LevelSetPositive` flips
both the interface normal and the traction curvature relative to the raw
level-set convention. This keeps the capillary force vector invariant under
active-side selection while preserving a single input convention for projected
or supplied curvature fields.

The supported curvature sources are:

| Curvature source | Production contract |
|------------------|---------------------|
| Unknown scalar curvature field | The field participates in the unknown vector, is added as an extra trial field for free-surface terms, and receives the ordinary residual/Jacobian derivative with respect to that scalar field. Its defining equation or constraint must be installed by the formulation that owns the model. |
| Prescribed projected curvature field | The application or caller refreshes a scalar field from level-set samples through the projected-curvature recovery utility. The capillary residual treats the field as fixed during the current nonlinear linearization, so the capillary update is Picard-style with respect to the level-set/cut geometry. |
| Constant supplied curvature | The value is a prescribed verification or reduced-model input. It has no level-set derivative and cannot be used to claim geometry-coupled capillary convergence. |
| Raw level-set curvature | Diagnostic only for unfitted generated interfaces. Production surface-tension cases must provide a named curvature field or a controlled supplied value. |

The prescribed projected-curvature path uses local quadratic recovery on the
active narrow band, optional supplemental generated-interface and cut-volume
samples, bounded fallback policies, and optional graph smoothing. It is the
current production-capable model only when benchmark gates also pass with zero
forbidden fallback vertices and recorded curvature/pressure/stability metrics.
It does not provide derivatives of curvature with respect to level-set DOFs or
regenerated cut geometry.

Picard-style capillarity is valid only under an explicit convergence audit. A
production run using prescribed projected curvature must refresh curvature after
each accepted level-set or geometry update, solve the physics residual with that
field held fixed, and record the active narrow-band curvature update. The
accepted nonlinear state must satisfy the solver residual tolerances and the
projected-curvature update must be small relative to the case tolerance before
pressure-jump, capillary-balance, or stability gates are claimed. If that audit
is not available, the case remains diagnostic even if the residual converged for
a frozen curvature field.

Diagnostics and output expose the state needed to audit a generated interface:

| Diagnostic | Meaning |
|------------|---------|
| `implicit_quadrature_backend` | requested/global backend or selected backend on generated rules |
| `selected_implicit_quadrature_backend` | concrete backend used for each generated rule |
| `implicit_fallback_status` | `None` or `Used` after generation |
| `requested_quadrature_order` / `achieved_quadrature_order` | requested and generated-rule achieved order |
| `root_branch_count`, `root_finder_iterations` | branch reconstruction/root-polishing work |
| `max_root_residual`, `min_gradient_norm`, conditioning status | root-polishing quality and local conditioning |
| `linearized_leaves`, `fallback_reason` | evidence that a high-order request used a lower-order or fallback path |
| backend elapsed time and quadrature-point counts | per-cell and per-step cost diagnostics |

VTP/JSON debug output includes stored curved-interface quadrature points,
normals, weights, parent/reference coordinates, root residuals, gradient norms,
curved patch ids, requested/achieved orders, and fallback status. The optional
tessellated visualization path and production curvature recovery are still open
work items; do not advertise high-order surface tension from raw pointwise
level-set curvature.

## Compatibility

No Physics compatibility headers remain for the migrated level-set APIs. New
code should include `FE/LevelSet/...` directly and use
`svmp::FE::level_set` names. User-facing level-set XML names remain accepted by
the application input adapter for backward compatibility.

## Module Authoring

When a Physics module needs level-set services:

1. Include the FE header that owns the service, usually
   `FE/LevelSet/LevelSet.h` or a narrower `FE/LevelSet/...` header.
2. Parse module-specific input in the application or Physics module and
   translate it into the matching `FE::level_set` option struct.
3. Pass the target `FESystem`, fields, spaces, markers, and options to the FE
   service.
4. Keep equation-specific residuals and boundary laws in the Physics module.
5. Add reusable level-set algorithms, restart records, diagnostics, and
   generated-interface utilities to `FE/LevelSet`, not to Physics.

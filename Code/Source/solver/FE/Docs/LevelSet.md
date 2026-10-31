# FE Level-Set Services

`svmp::FE::level_set` owns reusable level-set infrastructure. The namespace is
physics-neutral: it knows about FE systems, fields, function spaces, generated
interfaces, cut-cell integration data, and diagnostics, but it does not encode
Navier-Stokes, material, contact-line, or free-surface policy.

Physics modules remain responsible for equation-specific defaults and
formulation-specific residuals. The application translator owns the
level-set equation input, translates it into `FE::level_set` options, and
calls the FE services directly.

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
| `FE/LevelSet/LevelSetVolume.h` | cut-cell volume measurement |
| `FE/LevelSet/LevelSetGeometry.h` | immutable geometry identities, cut-side domains, owned boundary resolver |
| `FE/LevelSet/LevelSetSideSupport.h` | vertex support, admissibility, side-restricted vertex initialization |
| `FE/LevelSet/LevelSetWallContactConstraint.h` | identity of an accepted wall-contact cell |
| `FE/LevelSet/Maintenance/LevelSetMaintenanceConfiguration.h` | configuration, run state and declaration bindings of a maintained level-set field |
| `FE/LevelSet/Maintenance/LevelSetMaintenancePipeline.h` | accepted-step maintenance pipeline |
| `FE/LevelSet/Maintenance/LevelSetWallAwareMaintenance.h` | accepted wall-contact cells of a maintained field from the host's plain wall input |
| `FE/LevelSet/Maintenance/LevelSetMaintenanceWorkLedger.h` | work of maintenance on the declared functionals by substage and transaction, and the accepted-step maintenance account |
| `FE/LevelSet/LevelSetDiagnostics.h` | scalar diagnostics for the cut-cell volume |
| `FE/LevelSet/LevelSetRestart.h` | restart records for level-set fields and generated interfaces |

APIs in `FE/LevelSet` are public FE APIs unless they are in an anonymous
namespace or marked internal in the header. Keep new reusable level-set
algorithms in this directory and expose them through the smallest header that
matches their scope.

## Geometry identities and owned boundary resolution

`LevelSetGeometry::bind()` resolves a scalar field and interface marker into a
value handle containing field ID, domain ID, isovalue, and marker. `CutSide`
adds the selected side and supplies typed volume, interface, boundary, facet,
and contact-line domains. These handles describe identity; they contain no
coefficient, coordinate, quadrature, or certificate snapshots. Generated
geometry must still be refreshed and certified through the normal system
lifecycle before use.

`generatedActiveBoundaryResolver(std::vector<CutSide> owners, std::string owner,
std::string invalid_physical_marker_message)` returns a callable that owns its
side handles and diagnostic strings. It can outlive the input vector and does
not borrow a system or cut context. Each call visits owners in order and
requires agreement of both the generated marker and its stable key. Ambiguous
owners throw `std::invalid_argument`. A negative physical marker throws with
the supplied text when owners exist; an empty owner list returns `std::nullopt`
for every input, including negative markers.

Resolution alone does not register a marker. The separate
`wettedBoundaryResolver(system, resolve)` adapter registers each resolved marker
when invoked and borrows the system, which must outlive that callable. Typed
domains can instead be registered explicitly with
`FormulationBuilder::registered()`. Keep registrations at their intended
mutation points, including repeated registrations when call order matters.

## Vertex support, admissibility, and initialization

`requireVertexValuesOnLevelSetSide(system, samples, requirement)` checks ordered
`VertexValueSamples` against a borrowed `LevelSetVertexValues` view. For each
local input it checks mesh range, side membership, then the strict exclusion
margin `abs(phi - isovalue) < margin`. The first failure throws
`std::invalid_argument` using the caller's failure formatter. Nonlocal inputs
are skipped, but their prescribed values still enter input extrema. Duplicates
are checked in order. Statistics are local, empty extrema remain infinite, and
no finite-value scan or collective reduction is added. The supplied level-set
view must cover the local mesh and remain valid throughout the call.

`initializeVertexFieldOnLevelSetSide(system, state, target, declaration)` uses a
`systems::VertexFieldTarget` resolved by `requireVertexField()` and a
`SideVertexInitializationDeclaration`. The declaration supplies exactly one
`systems::VertexScalarSource`: a spatial callable or a
`RequiredMeshVertexScalarSource`. Choosing that source and its physical law is
the caller's responsibility. An optional `LevelSetVertexSupportDeclaration`
selects side support; unsupported vertices receive `unsupported_value`.

Each call binds current support, validates the source, collects vertex
coefficients, compares requested vertex samples, then writes through one vector
assembly view, in that order. A named source uses component zero; setting its
`non_finite_prefix` requests a scan of every local source vertex, including
unsupported vertices. All checks and comparisons precede the vector write.
The result reports initialized DOFs, side/support statistics, and comparisons.
Only vertex DOFs are written; higher-order edge, face, and cell coefficients
remain unchanged.

Support is recomputed for each call. With a native mesh, it includes vertices
of cells with positive side measure using the requested generated volume rules
when available, otherwise cells with a vertex strictly on the selected side.
Without a native mesh it uses vertex side membership. The service checks that
relevant geometry snapshots are current and caches neither views nor support
masks. Keep the target DOF map, comparison span, and callable captures valid
through the call; resolve the target again after renumbering. Refresh generated
geometry before invoking initialization after a geometry change.

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

The application level-set equation translator is an input adapter. It builds
FE options and calls `FE::level_set::installLevelSetTransport()`; it must not
contain reusable transport, volume, maintenance, diagnostics, or restart
algorithms. The keys of a `level_set` equation are declared once, in the
option schema `levelSetEquationSchema()`
(`Application/Translators/LevelSetConfigurationParsing.h`, decision D47 step
P3): 67 keys with their spellings, codec, FE default and a documentation
line, over `LevelSetEquationKeys` (the transport options with the
conservative-phase controls and the accepted-step switches, `Coupled`,
`Level_set_isovalue`, the PDE velocity-extension keys and the
static-capillary options). The installation resolves the equation element and
then its domain elements, each over the values before it; the
interface isovalue, the extension and the static-capillary keys come from the
equation element only. The maintenance of the field, the driver's
extension requests and the static-capillary switch read the keys of the
equation element as the installation resolved them, and the transport's
effective-configuration artifact ends with a `resolved` section that lists
every key with its value and source (`OptionSchema::describe`). The boundary
conditions (`LevelSetInflow`, `LevelSetOutflow`) and the space keys
(`Element_order`) keep their own readers.

The PDE velocity extension of the level-set advection velocity
(`Advection_velocity_extension_method` = `pde_harmonic` or `pde_normal`) is in
`FE/LevelSet/Extension`: the dry-region solve
(`LevelSetPdeVelocityExtension.h`), its known set and frozen-map revision
(`LevelSetVelocityExtensionSeed.h`) and its wall data
(`LevelSetVelocityExtensionWall.h`). It works on every first-order Lagrange
cell (Triangle3, Quad4, Tetra4, Hex8, Wedge6, Pyramid5), on every high-order
Lagrange cell with at most 27 nodes (Triangle6, Quad9, Tetra10, Pyramid14,
Wedge18, Hex27, ...) and on meshes that mix them: first-order simplices keep
their constant P1 gradients, the other cells use the FE Lagrange element of
their type and geometry order, its quadrature and the isoparametric map. The
extension lives on the mesh vertices, which are the geometry nodes, so on a
high-order cell its edge, face and interior nodes are extension vertices like
the corners; the level-set, source and target fields must be fixed-order
nodal Lagrange fields whose nodes are exactly those vertices (field order
equal to the geometry order of every cell: P1 on first-order cells, P2 on
quadratic cells), which the refresh checks with
`FESystem::fieldDofsAtMeshVertices()`. A P2 field on first-order cells has
edge DOFs without a mesh vertex and is rejected. The known set holds every
node of the retained interface cells, so the extension equals the physical
velocity on the interface for every order. With monolithic coupling its rows
enter the Jacobian through `LevelSetVelocityExtensionConstraintKernel`, whose
sparsity couples every node pair of a cell (on a quad or hex also across
faces and the body). A row of a high-order node resolves to the edge, face or
cell-interior DOFs whose node it is; like a vertex row it is emitted only by
the owner of the node, and its dependencies resolve on that rank through the
ghost entities. The algebraic wall-compatible and nearest-sample wet
extensions of the application were retired by decisions D47 and D55.

## Production Transport Contract

The production transport operator is Galerkin advection with optional SUPG. Its
directional, transient scale is

`tau = c_tau / sqrt((c_t / dt_eff)^2 + u . G u + epsilon_u)`,
`G = J^-T J^-1`.

This metric responds to the streamline direction on anisotropic cells and stays
finite as the velocity tends to zero. The residual-based discontinuity
capturing that could be added to SUPG was retired by decision D47.

The bound-preserving candidate gate (a nodal one-ring bound check of the step
candidate with a Courant and wall-flux pre-check, and a step retry when a
correction would be required) was retired by decision D51: with it on, the
SPHERIC reruns failed at their first step, while the runs without it and the
sign-definite patch bounds were clean.

The velocity extension uses the active `FESystem`/DOF-handler communicator,
never `MPI_COMM_WORLD`; its known set is made to agree across that
communicator before the dry-region solve, so a vertex marked by any rank is
known on every rank.

### Kinematic Reconciliation

Galerkin transport satisfies `phi_t + w . grad(phi) = 0` only in the
`L2(Omega)` sense, so the sharp P1 liquid measure `V(phi) = |{phi < c}|`
changes by more or less than the interface flux of the transport velocity.
Where `w` varies on the mesh scale, for example at a moving contact line
whose slip velocity peaks at the contact vertex, this is the dominant source
of liquid-volume drift; it does not shrink with `dt` and converges slowly
with `h`.

`Enable_kinematic_reconciliation` (default off;
`LevelSetKinematicReconciliationOptions`) applies, after every accepted step,
the correction of `FE/LevelSet/LevelSetKinematicReconciliation.h`. With the
transported rate `d = (phi_t - phi_p) / dt` held fixed, the endpoint solves
the lumped interface-kinematic equations

`phi_i = phi_t,i - dt r_i / gbar_i`,
`r_i = ((M_p + M) d)_i / 2 + (f_p,i + f_i) / 2`,
`gbar_i = (g_p,i + g_i) / 2`,

with `g_i = int_Gamma N_i / |grad(phi)|` (the exact nodal derivative of `-V`),
`f_i = int_Gamma N_i (w . grad(phi)) / |grad(phi)|` (the nodal parts of
`int_Gamma w . n`) and `M_ij = int_Gamma N_i N_j / |grad(phi)|`, on the
previous accepted interface and, by fixed-point iteration, on the reconciled
endpoint. Since the row sums of `(M_p + M) / 2` are `gbar`, the trapezoidal
change of `V` equals the trapezoidal flux `dt (F_p + F) / 2`. Each nodal
change depends only on the interface in the node's support: there is no
global shift, multiplier or numerical parameter, and the target is the flux
of `w` itself rather than a volume. Nodes keep their sign class (so the cut
topology is unchanged); a blocked node's share moves to the other nodes of its
cells. Affine `Triangle3` and `Tetra4` cells are supported; replicated level-set
coefficients and owned-cell sums give partition-independent results. The
application runs it as the first stage of accepted-step maintenance
(`GeometryReconciliation` ledger rows) and logs one
`Level-set kinematic reconciliation` line per step with the transported and
reconciled volume errors.

### Sign-Definite Patch Bounds

Exact transport is constant along characteristics, so after a step whose
one-ring Courant number is at most one the value at a node lies in the range of
the previous values over the node's patch. The Galerkin step lacks this local
maximum principle: the consistent mass matrix projects the advective rate,
which is largest at the nodes of cut cells, onto P1 with negative lobes on the
neighbouring nodes, and at a stagnation point of a diverging transport velocity
the central Galerkin stencil is anti-diffusive. Next to a contact line, where
the transport velocity has a mesh-scale kink and the transported field is thin
(the gas wedge behind a receding line, the wall row inside the footprint), a
wall node whose whole patch lies in one phase can drift to the isovalue over a
few hundred steps and cross it, creating a spurious wetted or dry spot on the
wall.

`Enable_sign_definite_patch_bounds` (default off;
`LevelSetSignDefinitePatchBoundsOptions`) applies, after every accepted step,
`FE/LevelSet/LevelSetSignDefinitePatchBounds.h`: a node is sign definite when
every node of every cell around it is in one strict sign class at the previous
accepted state and, the node itself excepted, at the transported endpoint; its
value is clamped into `[min, max]` of the previous values over its patch. Every
other node, in particular every node of a cut cell, keeps its transported
value, so the interface, the contact line and the liquid measure do not change,
and a node away from the interface cannot cross the isovalue. There is no
numerical parameter. The bound holds where characteristics start inside the
mesh, so the application rejects the option together with level-set inflow
boundaries. Affine `Triangle3` and `Tetra4` cells are supported;
min/max and count reductions over owned cells give partition-independent
results. The application runs it after the kinematic reconciliation
(`Limiting` ledger rows) and logs one `Level-set sign-definite patch bounds`
line per step.

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
| `jump(u)` | `u- - u+`, the `jump()` of the form language, for scalar and component-wise vector/tensor fields unless a formulation explicitly asks for a normal jump. The minus side of a two-sided generated rule is the negative retained side and the plus side the positive one (`GeneratedInterfaceTwoSidedBinding::minus_side = Negative`, `plus_side = Positive`). |
| `avg(u)` | Arithmetic average `0.5 * (u+ + u-)`. Weighted averages are a formulation-specific extension and must declare their weights. |
| `n+`, `n-` | `n+ = n` and `n- = -n`, where `n` is the generated-interface normal above. In a two-sided term `normal().plus()` evaluates to `n+` and `normal().minus()` to `n-` (the assembler negates the rule normal on the negative side), so each points into its own side; on interior facets and fitted interfaces `normal().minus()` is instead the outward normal of the minus cell. A one-sided term sees the rule normal `n`. |

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

## Generated Interface-Boundary Intersections

`FE::interfaces::GeneratedInterfaceBoundaryIntersectionDomain` derives a
lower-dimensional integration domain from an implicit scalar field, an existing
generated interface marker, and a mesh boundary marker. It is physics-neutral
and contains only parent cell identity, generated-interface marker,
mesh-boundary marker, point/segment quadrature, implicit-field normal,
boundary normal, tangent, and deterministic summary data.

The production builder restricts the current scalar field to each marked
boundary subentity and delegates point/segment rule construction to
`FE::quadrature::buildImplicitBoundaryIntersectionQuadrature(...)`. Linear 2D
cells produce 0D intersection-point rules on boundary edges. Linear 3D
tetrahedron, hexahedron, wedge, and pyramid parents produce 1D segment rules on
boundary faces. These are true codimension-two rules: the rule stores an
explicit geometric dimension of zero in 2D and one in 3D. The only supported
data contract is parent-reference coordinates, parent-reference normals and
tangents, and a reference d-2 weight. During assembly, a 2D point keeps unit
counting measure and a 3D line receives the single factor `|J t_ref|`. No
codimension-one surface Jacobian is applied and no physical quantity is mapped
twice. Domain/rule `measure` summaries are reference measures; physical
length/angle diagnostics must use mapped assembly geometry or
`mapReferenceCutCodimensionTwoGeometry(...)`, never raw rule directions.

High-order parent cells are rejected until curved/isoparametric contact
intersection reconstruction is implemented. Increasing Gaussian order on a
corner-linear segment is not reported as high-order geometry. Requested and
achieved quadrature orders are recorded separately and the linear-corner
construction is named in rule provenance.

Unsupported geometry, vertex touches, edge-aligned intersections, fully
degenerate boundary subentities, and ambiguous multi-root (for example bilinear
saddle-face) topology are retained as explicit diagnostics but are not active
assembly rules. There is no fallback to the full generated interface and no
arbitrary edge-order pairing. This fail-closed behavior prevents duplicate or
topologically invented contact measure; aligned/shared entities require a
future global ownership rule, and saddle faces require a deterministic topology
decider before they can be enabled.

Application active-cut context refresh builds these intersection domains for
the mesh boundary markers visible through `IMeshAccess`, imports their
quadrature through `CutIntegrationContext`, and logs per-marker and aggregate
fragment, quadrature-point, and measure counts alongside the
generated-interface active-domain diagnostic.
Markers are generated with
`stableGeneratedInterfaceBoundaryIntersectionMarker(...)` from the level-set
source, generated-domain id, isovalue, interface marker, and boundary marker.
`CutIntegrationContext` also registers the complete marker key and rejects a
duplicate import, a collision with a generated-interface marker, or two
different intersection keys that resolve to the same integer marker.
Physics code should use that marker with the form measure alias
`dInterfaceBoundary(expr, marker)`. It shares `.dI(marker)` dispatch, while the
imported rule's explicit geometric dimension selects codimension-two mapping in
the assembler.

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
| `LinearCorner` | 2D triangles/quads, 3D tetrahedra | 2D 5 / 5 for generated planar rules; 3D 2 / 2 (planar polygon interfaces use a degree-2 triangle-fan rule; lower requested orders keep the centroid rule) | production path |
| `SayeHyperrectangle` | 2D quads and 3D hexes | 2D 5 / 5; 3D 1 / 2 | 2D quadrilaterals reported as production-qualified by the backend capability; 3D experimental |
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

The Navier-Stokes unfitted surface-tension form `SurfaceStress` does not yet
have complete normal, point-location, and measure derivatives. It therefore
requires `RefreshedFrozenQuadrature`. Cases with surface tension must
set `Geometry_tangent_policy=RefreshedFrozenQuadrature` explicitly, even on the
default `LinearCorner` path.

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

The unfitted capillary form in Navier-Stokes is `SurfaceStress`. It is the
first variation of the generated-interface measure, uses the generated
normal, and needs no curvature field. `CurvatureTraction`, which read a
supplied, unknown or projected curvature, was retired by decision D47 Q2, and
`KinematicAreaGradientTraction`, which read a curvature field recovered by
the level-set curvature projection, by decision D48. The projection itself
(`LevelSetCurvatureProjection`, its least-squares and
kinematic-area-gradient recovery modes, the curvature samples and the
`Enable_curvature_projection`, `Curvature_field_name` and
`Curvature_projection_*` keys) was retired with it, and the keys were then
removed from the input.

Raw pointwise `meanCurvatureFromLevelSet(phi)` is a diagnostic path for
fitted or controlled studies; it is not production evidence for unfitted
surface tension. `Documentation/free_surface_program_tracker.md` records the
route selection (decisions D2 and D48, milestone M2).

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
tessellated visualization path and curvature recovery on high-order geometry
are still open work items; do not advertise high-order surface tension from raw
pointwise level-set curvature.

## Compatibility

No Physics compatibility headers remain for the migrated level-set APIs. New
code should include `FE/LevelSet/...` directly and use
`svmp::FE::level_set` names. The level-set XML keys are the spellings of
`levelSetEquationSchema()`; a removed key is deleted outright and fails as an
unknown element (decision D60).

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

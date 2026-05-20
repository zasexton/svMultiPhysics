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

## High-Order Implicit Generated Interfaces

Generated interfaces default to the production `LinearCorner` path. The current
high-order implicit path is an experimental FE/OOP feature for generated
level-set interfaces. Enable it only for validation or explicitly qualified
cases:

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
| `Geometry_tangent_policy` | `RefreshedFrozenQuadrature`; `DifferentiatedQuadrature` is parsed but rejected until geometry sensitivities are implemented |
| `Implicit_cut_root_tolerance` | signed-level-set residual tolerance for root polishing |
| `Implicit_cut_root_coordinate_tolerance` | reference-coordinate convergence tolerance for root polishing |
| `Implicit_cut_root_max_iterations` | root-polishing iteration cap |
| `Implicit_cut_max_subdivision_depth` | branch/topology isolation cap, not a production accuracy dial |

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

Explicit global high-order backends are validated before generation. For
example, `SayeHyperrectangle` on a mixed quad/triangle mesh fails before
building rules because triangles are not supported by that global backend. Use
`Auto` for mixed supported meshes. If `Implicit_cut_fallback_policy` is
explicitly set to `LinearCorner`, an implemented high-order backend may fall
back only for cell families supported by the linear cutter; fallback cells are
counted and stamped into rule provenance. `MomentFit` remains unavailable and
does not become valid through fallback.

For high-order qualification runs, keep
`Implicit_cut_fallback_policy=Fail` and require zero fallback cells. Use
`LinearCorner` fallback only for intentional diagnostic or compatibility runs:

```xml
<Generated_interface_geometry>HighOrderImplicit</Generated_interface_geometry>
<Implicit_cut_quadrature_backend>SayeHyperrectangle</Implicit_cut_quadrature_backend>
<Implicit_cut_fallback_policy>LinearCorner</Implicit_cut_fallback_policy>
```

`RefreshedFrozenQuadrature` refreshes generated geometry and quadrature when
the active cut context is rebuilt, but treats quadrature points, weights,
measures, normals, and topology as fixed during tangent assembly. It is a
quasi-Newton policy. `DifferentiatedQuadrature` is reserved for the future exact
geometry-sensitivity path and is rejected by generated-interface validation.

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

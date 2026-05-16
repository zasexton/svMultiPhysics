# High-Order Curved Implicit Level-Set Geometry And Quadrature Plan

## Purpose

This plan defines the work needed to move the generated level-set interface path
from the current corner-linearized cutter to high-order curved implicit geometry
and quadrature. The target capability is to evaluate cut-volume and embedded
interface integrals directly from a high-order level-set field, with accuracy
and stability suitable for unfitted finite-element free-boundary problems.

The immediate goal is not to replace the existing linear cutter everywhere. The
goal is to add an opt-in high-order path whose contracts, diagnostics, and tests
are strong enough that validation cases can distinguish geometry/quadrature
errors from Navier-Stokes, level-set transport, and solver errors.

## Scope

The plan covers:

- High-order evaluation of scalar level-set fields inside parent cells.
- Curved implicit cut-volume and interface quadrature generation.
- Integration with `CutIntegrationContext`, `dCutVolume(...)`, and `dI(...)`.
- Stabilization, pressure support, diagnostics, MPI determinism, and tests.
- Literature-backed decisions for implementation and validation.

The plan does not cover:

- A complete two-phase material jump formulation.
- Surface-tension production support using raw level-set curvature.
- Exact Newton shape derivatives for moving cut geometry in the first milestone.
- Replacing the existing linear level-set cutter for all production runs.

## Current Baseline

The current generated interface lifecycle reads a scalar level-set field from the
current FE solution and builds generated cut-volume and interface rules. The
current built-in cutter is intentionally linear: it uses corner coordinates and
corner level-set values, then rejects high-order geometry unless the input opts
into corner linearization for explicitly linearized test cases.

Important current touch points:

- `Code/Source/solver/FE/LevelSet/LevelSetInterfaceLifecycle.*`
- `Code/Source/solver/FE/Interfaces/LevelSetInterfaceBuilder.*`
- `Code/Source/solver/FE/Interfaces/LevelSetInterfaceDomain.*`
- `Code/Source/solver/FE/Assembly/CutIntegrationContext.h`
- `Code/Source/solver/FE/Assembly/StandardAssembler.cpp`
- `Code/Source/solver/FE/Systems/CutIntegrationInvalidation.*`
- `Code/Source/solver/Application/Core/ApplicationDriver.cpp`
- `Code/Source/solver/Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.*`

## Reference Literature

Use these sources as implementation guardrails:

- Saye, R. I. "High-Order Quadrature Methods for Implicitly Defined Surfaces and
  Volumes in Hyperrectangles." SIAM Journal on Scientific Computing, 2015.
  https://doi.org/10.1137/140966290
- Saye research summary for implicit geometry quadrature and downstream solver
  use in embedded-boundary and unfitted methods.
  https://math.lbl.gov/saye/research-quad.html
- Saye, R. I. "High-Order Quadrature on Multi-Component Domains Implicitly
  Defined by Multivariate Polynomials." 2021.
  https://arxiv.org/abs/2105.08857
- Fries, T. P., Omerovic, S., Schollhammer, D., Steidl, J. "Higher-order
  meshing of implicit geometries - Part I: Integration and interpolation in cut
  elements." Computer Methods in Applied Mechanics and Engineering, 2017.
  https://arxiv.org/abs/1706.00578
- Muller, B., Kummer, F., Oberlack, M. "Highly accurate surface and volume
  integration on implicit domains by means of moment-fitting." International
  Journal for Numerical Methods in Engineering, 2013.
  https://doi.org/10.1002/nme.4569
- Joulaian, M., Hubrich, S., Duster, A. "Numerical integration over implicitly
  defined domains for higher order unfitted finite element methods." 2016.
  https://arxiv.org/abs/1601.06182
- Cheng, K. W., Fries, T. P. "Higher-order XFEM for curved strong and weak
  discontinuities." International Journal for Numerical Methods in Engineering,
  2010. https://doi.org/10.1002/nme.2798
- Massing, A., Larson, M. G., Logg, A., Rognes, M. E. "A stabilized Nitsche
  fictitious domain method for the Stokes problem." Journal of Scientific
  Computing, 2014. https://arxiv.org/abs/1206.1933
- Burman, E., Hansbo, P., Larson, M. G., Zahedi, S. "Cut Finite Element Methods
  for Coupled Bulk-Surface Problems." Numerische Mathematik, 2016.
  https://arxiv.org/abs/1403.6580
- Hansbo, P., Larson, M. G., Zahedi, S. "A cut finite element method for coupled
  bulk-surface problems on time-dependent domains." Computer Methods in Applied
  Mechanics and Engineering, 2016. https://arxiv.org/abs/1502.07142
- Schillinger, D., Ruess, M. "The Finite Cell Method: A review in the context of
  higher-order structural analysis of CAD and image-based geometric models."
  Archives of Computational Methods in Engineering, 2015.
  https://doi.org/10.1007/s11831-014-9115-y
- de Prenter, F., Verhoosel, C. V., van Brummelen, E. H. "Stability and
  Conditioning of Immersed Finite Element Methods: Analysis and Remedies."
  Archives of Computational Methods in Engineering, 2023.
  https://doi.org/10.1007/s11831-023-09913-0
- Sussman, M., Smereka, P., Osher, S. "A Level Set Approach for Computing
  Solutions to Incompressible Two-Phase Flow." Journal of Computational Physics,
  1994. https://doi.org/10.1006/jcph.1994.1155
- Olsson, E., Kreiss, G. "A conservative level set method for two phase flow."
  Journal of Computational Physics, 2005.
  https://doi.org/10.1016/j.jcp.2005.04.007

## 1. Numerical Contract And Feature Gate

Decision: add an opt-in high-order generated interface path. Keep the existing
linear corner cutter as the default until the high-order path has analytic
geometry tests, assembly finite-difference tests, and solver smoke evidence.

Rationale: Saye's algorithm and the higher-order implicit-geometry literature
show that accurate cut integration depends on the implicit surface representation
and quadrature construction, not just on raising the element basis order. Keeping
the current path available avoids changing benchmark behavior while the new path
is qualified.

### Design Checklist

- [x] Define a new geometry mode, for example
      `Generated_interface_geometry=LinearCorner` and
      `Generated_interface_geometry=HighOrderImplicit`.
- [x] Define a new quadrature backend option, for example
      `Implicit_cut_quadrature_backend=SayeHyperrectangle`,
      `HighOrderSubcell`, `MomentFit`, or `LinearFallback`.
- [x] Define supported cell families for the first milestone.
      Recommended first target:
      - quads in 2D with Saye-style hyperrectangle quadrature,
      - hexes in 3D only after the 2D path is validated,
      - triangles/tets through a separate simplex strategy.
- [x] Define accepted level-set spaces: scalar H1/C0 fields first.
- [x] Define minimum supported polynomial orders: P1 must reproduce existing
      behavior; P2 and P3 must demonstrate curved-interface convergence.
- [x] Define separate requested orders for volume and interface quadrature.
- [x] Define the achieved-order reporting contract for adaptive backends.
- [x] Define fallback behavior: hard fail, linear fallback, subcell fallback, or
      conservative constant-measure fallback.
- [x] Define acceptance policy for tiny slivers and near-tangent cuts.
- [x] Define whether high-order geometry is allowed in benchmark fixtures before
      validation is complete. Recommended answer: no.

### Implementation Checklist

- [x] Add parser support for geometry mode, backend, requested orders, root
      tolerance, subdivision depth, and fallback policy.
- [x] Thread the options into `LevelSetGeneratedInterfaceOptions`.
- [x] Log the selected geometry mode, backend, requested order, achieved order,
      tolerance, and fallback count in cut-context rebuild diagnostics.
- [x] Reject incompatible combinations early, such as high-order implicit mode
      with a non-scalar level-set field.
- [x] Keep `allow_corner_linearized_geometry` specific to the linear fallback
      path and do not let it silently downgrade a high-order request.

### Tests

- [x] Parser test for every new option and synonym.
- [x] Input-validation test for unsupported cell/backend combinations.
- [x] Regression test proving the default remains `LinearCorner`.
- [x] Regression test proving a high-order request cannot silently use the
      corner cutter unless the fallback policy explicitly permits it.

### Literature Support

- Saye 2015 supports high-order quadrature over implicitly defined surfaces and
  volumes in hyperrectangles.
- Fries et al. 2017 supports the need to pair high-order level-set
  interpolation with high-order integration of cut elements.
- Joulaian et al. 2016 compares subdivision, moment fitting,
  quasi-parametrization, and Monte Carlo approaches for higher-order unfitted
  FEM integration.

## 2. High-Order Level-Set Field Evaluation

Decision: build a reusable level-set evaluator that evaluates `phi_h`, its
reference gradient, and optionally higher derivatives at arbitrary parent-cell
reference points from an FE-ordered state vector.

Rationale: the current cutter asks for one scalar vertex DOF per mesh vertex.
That is not sufficient for P2/P3 level-set geometry, where edge, face, and
cell-interior DOFs control the zero contour. High-order implicit quadrature
algorithms need repeated pointwise evaluation of the implicit function.

### Design Checklist

- [x] Define `LevelSetCellEvaluator` or equivalent service.
- [x] Input contract:
      - `FESystem`,
      - `FieldId` or field name,
      - FE-ordered coefficient span,
      - parent cell id.
- [x] Output contract:
      - `value(xi)`,
      - `referenceGradient(xi)`,
      - optional `referenceHessian(xi)`,
      - polynomial/order metadata.
- [x] Support scalar H1/C0 nodal fields first.
- [x] Avoid mesh vertex-field reads inside the high-order cutter.
      High-order geometry evaluation is driven by the FE-ordered solution span
      through `LevelSetCellEvaluator`; mesh vertex fields remain outside the
      implicit cutter and are only synchronization/output mirrors.
- [x] Keep a separate synchronization path for visualization mesh fields.
      Application-level vertex mesh-field refreshes stay in
      `syncActiveLevelSetVertexFieldsFromSolution(...)` before diagnostics and
      output, separate from generated cut-rule construction.
- [x] Ensure vector/state ordering is identical to the current cut-context
      refresh path used during Newton residual and Jacobian assembly.
- [x] Define behavior for ghost cells and non-owned cells in MPI.
      Locally present ghost cells evaluate the same FE-ordered level-set
      coefficients as owned cells and may build identical cut rules for
      deterministic diagnostics; ownership filtering remains a downstream
      assembly/reduction responsibility.
- [x] Define performance caching for basis values without reintroducing
      process-wide transient cut-quadrature basis growth.

### Implementation Checklist

- [x] Add a field evaluator under `FE/LevelSet` or a generic FE field-evaluation
      module if an existing service can be reused.
- [x] Resolve field DOFs through `DofHandler`, including vertex, edge, face, and
      cell-interior DOFs.
- [x] Gather cell coefficients from the FE-ordered state span.
- [x] Evaluate basis values and gradients in the parent reference frame.
- [x] Report the level-set interpolation order and geometry order.
- [x] Add a cell-local cache for repeated quadrature backend calls.
- [x] Add explicit diagnostics if a field has no entity DOF map or cannot be
      evaluated in the requested cell family.

### Tests

- [x] P1 evaluator reproduces current vertex-interpolated values at corners.
- [x] P2 evaluator responds to edge DOFs in the cell interior.
- [x] P3 evaluator responds to interior DOFs where available.
- [x] Gradient checks against finite differences in reference coordinates.
- [x] MPI/ghost test proving deterministic values on shared cells.
- [x] State-ordering test proving Newton trial states feed the evaluator, not
      stale mesh vertex fields.

### Literature Support

- Fries et al. 2017 explicitly treats level-set data on higher-order background
  meshes and the zero-level set of the interpolated field.
- Cheng and Fries 2010 highlights that higher-order convergence for curved
  discontinuities depends on accurate interface representation and quadrature.
- Saye 2015 assumes repeated evaluation of the implicit function while
  constructing high-order surface and volume rules.

## 3. Quadrature Backend Abstraction

Decision: add a backend abstraction that returns generated cut-volume and
interface quadrature rules independent of the current linear cutter. The
abstraction must be able to report success, fallback, achieved order, and
diagnostics.

Rationale: Saye-style recursive quadrature, high-order subcell decomposition,
and moment fitting have different algorithms and failure modes. A backend
interface lets the lifecycle choose the correct implementation while keeping
assembly and physics code backend-agnostic.

### Design Checklist

- [x] Define an interface such as `ImplicitCutQuadratureBackend`.
- [x] Input:
      - parent cell id,
      - element type,
      - mesh dimension,
      - parent reference domain,
      - level-set evaluator,
      - isovalue,
      - requested volume and interface orders,
      - tolerances and fallback policy.
- [x] Output:
      - negative-side volume rule,
      - positive-side volume rule,
      - interface rule,
      - cut-cell classification,
      - volume fractions and measures,
      - interface normals in reference coordinates,
      - achieved orders and policy name,
      - diagnostic string and fallback reason.
- [x] Preserve reference-frame output so existing assembly can map rules through
      parent physical geometry.
- [x] Preserve deterministic rule ordering.
- [x] Require strictly finite points and weights.
- [x] Require positive volume weights unless a backend explicitly documents
      signed moment-fit weights and downstream acceptance permits them.

### Implementation Checklist

- [x] Add backend registry or factory.
- [x] Move current linear cutter behind the same abstraction.
- [x] Add backend metadata fields to `CutQuadratureRule` provenance or adjacent
      metadata.
- [x] Add a standard diagnostic enum:
      `ExactNoCut`, `Cut`, `Tangent`, `Degenerate`, `Fallback`, `Unsupported`,
      `Failed`.
- [x] Add common validation for finite weights, finite points, valid side
      labels, and parent-cell ids.
- [x] Add common measure consistency checks:
      negative plus positive reference measure equals parent measure within
      tolerance.

### Tests

- [x] Backend factory returns the linear backend for legacy settings.
- [x] Backend metadata reaches `CutIntegrationContext`.
- [x] Invalid backend output is rejected with clear diagnostics.
- [x] Negative and positive side measures sum to the parent measure.
- [x] Interface rule marker/provenance matches volume rule marker/provenance.

### Literature Support

- Joulaian et al. 2016 motivates a backend-style choice among subdivision,
  moment fitting, local quasi-parametrization, and Monte Carlo approaches.
- Saye 2015 and Muller et al. 2013 represent two different high-order
  quadrature families with different implementation requirements.
- Fries et al. 2017 motivates subcell decomposition as a separate high-order
  strategy.

## 4. Saye-Style Hyperrectangle Backend

Decision: implement Saye-style recursive high-order quadrature first for
reference quads and, after validation, reference hexes. Use Gaussian rules and
adaptive subdivision/root-finding controls to produce high-order surface and
volume rules over implicitly cut hyperrectangles.

Rationale: Saye 2015 is directly designed for implicitly defined surfaces and
volumes in hyperrectangles and produces positive quadrature weights. This is a
good match for quadrilateral and hexahedral reference cells.

Milestone status: the current 2D backend is an initial recursive
hyperrectangle proof for quadrilateral cells. It evaluates high-order level-set
fields in reference space, recursively subdivides cut rectangles, and uses
linear leaf cuts for the unresolved interface. The current 3D backend extends
the same milestone design to Hex8, Hex20, and Hex27 reference cells over
`[-1,1]^3`. Terminal cut boxes are decomposed into deterministic Tetra4 leaves
and then evaluated by the simplex linearized cut path. Until the full Saye
Gaussian base rules and root-polishing strategy are implemented, it reports
interface quadrature as achieved order 1 and volume quadrature as achieved
order 2.

### Current Limitation And Fallback Contract

- Supported cells: two-dimensional quadrilateral cells and three-dimensional
  hexahedral cells. Triangles and tetrahedra are handled by the separate
  simplex backend, and unsupported element/backend combinations fail
  explicitly.
- Supported level-set fields: scalar H1/C0 fields evaluated by
  `LevelSetCellEvaluator`. P1 fields reproduce the existing linear corner cut
  path through the backend abstraction. P2+ fields are evaluated at adaptive
  reference points, but unresolved cut leaves are still linearized.
- Achieved order: the current recursive 2D proof reports achieved interface
  order 1 and achieved volume order 2. Requests above those orders remain
  accepted for diagnostics, but downstream quadrature rule provenance records
  the lower achieved order.
- Recursion policy: the backend uses the requested subdivision depth capped at
  the current implementation limit, classifies rectangles from deterministic
  corner, edge-midpoint, and centroid samples, and linearizes cut leaves at the
  terminal depth.
- Fallback policy: unsupported element/backend combinations fail explicitly.
  There is no silent downgrade from `HighOrderImplicit` to `LinearCorner`; any
  future fallback must be policy-selected and counted in diagnostics.
- Degeneracy policy: multiple roots, near-tangent cuts, and singular gradients
  are not yet handled by Saye root polishing. The current diagnostics expose
  recursion depth, subdivision count, linearized leaves, full-region counts, and
  fragment counts so validation can identify cells that need the full algorithm.
- Output policy: diagnostic VTP output is the leaf-segment visualization of the
  generated rule, not an exact curved-interface reconstruction.
- Full Saye target order policy: when the root-polished Gaussian construction
  replaces the current terminal linearized leaves, the base one-dimensional
  Gaussian rule order is selected from the resolved volume or interface order
  requested by the form. The minimum accepted request is the maximum of the
  parent geometry order, level-set interpolation order, and form integrand
  order; backend provenance records both the requested order and the lower
  achieved order whenever the milestone implementation cannot meet it.
- Root-polishing policy: use `implicit_cut_root_tolerance` as the absolute
  signed-level-set residual tolerance in reference coordinates, pair it with a
  relative coordinate tolerance of the same magnitude, and cap polishing at 50
  iterations per root branch. Nonconvergence, nonfinite evaluations, ambiguous
  multiple roots, near-tangent cuts with singular reference gradients, and
  branches that leave the parent hyperrectangle are diagnostic failures unless
  an explicit fallback policy is selected.
- Adaptive failure policy: a cut cell may be accepted only when every generated
  rule has finite points, finite weights, deterministic provenance, and a
  nonnegative achieved-order diagnostic. Otherwise the backend records a
  per-cell status (`Tangent`, `Degenerate`, `Fallback`, `Unsupported`, or
  `Failed`) and either returns an explicit failure or a counted fallback,
  depending on the configured policy.

### Design Checklist

- [x] Define how svMultiPhysics reference quads map to the hyperrectangle used by
      the algorithm.
- [x] Define how svMultiPhysics reference hexes map to the hyperrectangle used by
      the algorithm.
- [x] Decide Gaussian base order from requested FE/form order.
- [x] Define root-finding tolerance and maximum iterations.
- [x] Define recursive subdivision limits.
- [x] Define adaptive failure handling for multiple roots, near-tangent
      interfaces, and singular gradients.
- [x] Define how to orient reference normals consistently with the existing
      negative/positive side convention.
- [x] Define how to compute side measures and volume fractions from the produced
      weights.

### Implementation Checklist

- [x] Implement 2D hyperrectangle volume and interface quadrature.
- [x] Validate 2D on analytic curves before implementing 3D.
- [x] Implement 3D hyperrectangle volume and interface quadrature.
- [x] Expose first-milestone per-cell diagnostics for recursion depth,
      subdivision count, linearized leaves, full-region counts, and fragment
      counts.
- [x] Expose detailed per-cell diagnostics for recursion depth, root-finder
      iterations, fallback usage, and achieved order.
- [x] Preserve deterministic point ordering.
- [x] Keep the backend independent from physics-specific free-surface concepts.
- [x] Add optional debug output that dumps one cell's implicit quadrature rule for
      postmortem inspection.

### Tests

- [x] 2D line cut through a quad reproduces linear backend measures.
- [x] Initial 2D circle cut checks area and arc length against analytic values.
- [x] 2D circle/ellipse cuts integrate area and arc length to requested order.
- [x] 2D curved interface tests demonstrate convergence under h-refinement and
      p-refinement.
- [x] 3D plane cut through a hex reproduces known volume and area.
- [x] 3D sphere cut integrates volume and surface area to tolerance.
- [x] 3D ellipsoid cuts integrate volume and surface area to tolerance.
- [x] Reference-frame hex rules map through a non-affine Hex27 physical
      geometry with the expected volume scaling and surface measure transform.
- [x] Near-tangent tests produce either a valid high-order rule or an explicit
      fallback/failure diagnostic.

### Literature Support

- Saye 2015 is the primary source for recursive high-order quadrature on
  implicit surfaces and volumes in hyperrectangles.
- Saye's research summary notes use in embedded-boundary and unfitted finite
  element methods, which matches the intended integration role here.
- The first 3D milestone keeps Saye's hyperrectangle recursion as the outer
  structure but uses a deterministic tetrahedral terminal split until the full
  root-polished Gaussian construction is implemented.

## 5. Simplex Backend Strategy

Decision: do not force triangles and tetrahedra through the hyperrectangle
backend without a validation strategy. Use a dedicated simplex strategy:
high-order subcell decomposition first, then optional polynomial implicit
quadrature or moment fitting after the reference-box path is stable.

Rationale: the current test fixtures and FEM infrastructure use triangles and
tets in addition to quads. Saye 2015 targets hyperrectangles. Later polynomial
and multi-component work, high-order implicit meshing, and moment-fitting
literature provide options for simplices, but each option has different
robustness and implementation cost.

### Triangle Strategy

The first triangle milestone will use a `HighOrderSubcell` backend on reference
triangles. Triangle3 fields must reproduce the existing linear triangle cutter.
Triangle6+ fields should recursively split the reference triangle into
orientation-preserving subtriangles, evaluate `phi_h` through
`LevelSetCellEvaluator`, classify subtriangles from deterministic vertex,
edge-midpoint, and centroid samples, and linearize only terminal cut leaves.

This keeps all generated quadrature in the parent reference frame, preserves the
positive-weight rule contract used by current assembly, and avoids introducing
moment-fit signed weights before the active-side, stabilization, and pressure
constraint paths are validated. It follows the subcell-decomposition direction
used by high-order unfitted FEM work for implicit geometries, while leaving
polynomial implicit quadrature and moment fitting as later accuracy upgrades once
the topology and metadata contracts are stable.

Triangle support must be selected explicitly through `HighOrderSubcell`; the
`SayeHyperrectangle` backend remains limited to quadrilateral hyperrectangles.
Until this backend is implemented, mixed quad/triangle meshes using
`HighOrderImplicit` should report unsupported triangle cells rather than silently
falling back to `LinearCorner`.

Milestone status: the first 2D `HighOrderSubcell` triangle backend is
implemented for Triangle3 and Triangle6 cells. Triangle3 delegates through the
existing linear cutter. Triangle6+ fields use deterministic recursive reference
triangle subdivision with positive-weight full-subcell rules and linear terminal
cut leaves. The current backend reports achieved interface order 1 and achieved
volume order 2; higher-order simplex root polishing and curved leaf rules remain
future work.

### Tetrahedron Strategy

The first tetrahedron milestone should use the same explicit `HighOrderSubcell`
backend on reference tetrahedra. Tetra4 fields must reproduce the existing
linear tetrahedron cutter. Tetra10+ fields should recursively split the
reference tetrahedron with a deterministic edge-midpoint refinement, evaluate
`phi_h` through `LevelSetCellEvaluator`, classify sub-tetrahedra from vertices,
edge midpoints, face centroids, and the cell centroid, and linearize terminal
cut leaves through the existing Tetra4 cutter.

The first implementation should retain the same conservative contract as the
triangle backend: positive volume weights only, reference-frame rules, negative
plus positive measures summing to the parent tetrahedron measure, and provenance
that reports `HighOrderSubcell` with requested and achieved quadrature orders.
Moment fitting and polynomial implicit quadrature remain later options after the
positive-weight subcell path is stable in 3D.

Degenerate vertex, edge, and near-tangent cases should initially inherit the
linear leaf diagnostics and be counted in backend diagnostics. Mixed tetrahedron
and hexahedron meshes should continue to fail unsupported high-order cells until
per-cell backend selection and mixed-mesh diagnostics are implemented.

Milestone status: the first 3D `HighOrderSubcell` tetrahedron backend is
implemented for Tetra4 and Tetra10 cells. Tetra4 delegates through the existing
linear tetrahedron cutter. Tetra10+ fields use deterministic recursive
reference-tetrahedron midpoint refinement with positive-weight full-subcell
rules and linear terminal cut leaves. The current backend reports achieved
interface order 1 and achieved volume order 2; high-order surface rules,
root-polished curved leaves, and mixed hex/tet backend selection remain future
work.

### Mixed-Element Diagnostic Contract

The current mixed-element policy is fail-fast with cell-local diagnostics.
High-order generated-interface runs still select one backend per request; they
do not yet auto-route quads to `SayeHyperrectangle` and triangles to
`HighOrderSubcell`. When a selected backend encounters an unsupported mixed cell,
the lifecycle error must report the backend name, parent cell id, element type,
and backend diagnostic. This keeps mixed-mesh failures actionable until explicit
per-cell backend selection is implemented.

### Simplex Fallback Hierarchy

The initial simplex hierarchy is:

1. Use `HighOrderSubcell` recursive simplex decomposition for supported
   Triangle3/Triangle6 and Tetra4/Tetra10 cells.
2. At terminal recursive leaves, use the existing linear simplex cutter on
   Triangle3 or Tetra4 subcells. This is a local subcell fallback inside the
   high-order backend, not a request-level `LinearCorner` fallback.
3. If a selected simplex backend cannot support the cell type, lacks the
   evaluator data, lacks enough corner coordinates, or fails common quadrature
   validation, fail the generated-interface build with a cell-local diagnostic.
4. Do not fall through to request-level `LinearCorner` unless a future option
   explicitly permits that behavior and records `fallback_used=true` in the
   per-cell diagnostics.

Moment fitting is not part of the first simplex fallback chain. It remains a
future backend and must be selected explicitly once positive-weight and
diagnostic contracts are available for simplex cells.

### Edge and Corner Degeneracy Policy

The initial simplex degeneracy policy inherits the existing linear simplex
classifier at terminal subcell leaves:

- `NoCut` means the sampled simplex is entirely on one side after tolerance
  handling and contributes only a full-side volume rule.
- `FullZeroCell` is degenerate and must not create an active interface rule
  unless a future exact-zero policy is introduced.
- `VertexTouch` and `EdgeTouch` record contact with simplex vertices or edges.
  Positive-measure fragments may be retained and counted as degenerate
  fragments; zero-measure contacts remain inactive.
- `SmallFragment` and `NearlyTangent` are diagnostic states. They must not
  silently create nonfinite points, negative weights, or untagged fallback
  rules.
- `keep_degenerate_fragments` controls whether degenerate interface fragments
  survive for diagnostics. Volume rules remain governed by positive measure and
  conservation checks.

For the first simplex milestone, edge and corner contacts are not elevated to
request-level fallbacks. A valid finite subcell rule may be used; otherwise the
backend must fail validation with a cell-local diagnostic that includes the
selected backend and parent cell.

### Design Checklist

- [x] Decide first simplex milestone:
      - high-order curved subcell decomposition,
      - moment fitting,
      - polynomial implicit quadrature,
      - or explicit unsupported diagnostics.
- [x] Define whether simplex support must be available before enabling
      high-order mode in mixed-element meshes.
- [x] Define conservative measure requirements for simplex cut volumes.
- [x] Define whether simplex backends may use signed weights.
- [x] Define simplex fallback hierarchy:
      high-order subcell -> linear subcell -> fail, or
      high-order implicit -> moment fit -> fail.
- [x] Define edge/corner degeneracy handling.

### Implementation Checklist

- [x] Add backend capability reporting by element type.
- [x] Add triangle curved-interface tests before tetrahedron work.
- [x] Add tetrahedron curved-interface tests before solver integration.
- [x] Ensure simplex rule provenance matches hyperrectangle rule provenance.
- [x] Ensure mixed quads/triangles or hexes/tets can report per-cell backend
      choices.

### Tests

- [x] Triangle circle-segment area tests.
- [x] Tetra sphere-cap volume tests.
- [x] P2 level-set edge DOF changes the cut location in triangles and tets.
- [x] Degenerate vertex/edge touch tests.
- [x] Mixed-element mesh test with deterministic rule ordering and diagnostics.

### Literature Support

- Fries et al. 2017 supports high-order meshing and decomposition of cut
  elements for implicit level-set geometries.
- Saye 2021 extends high-order quadrature to multivariate polynomial level sets
  and discusses more general implicitly defined domains.
- Muller et al. 2013 supports moment-fitting as an accurate integration option
  for implicit domains.
- Joulaian et al. 2016 compares the tradeoffs of these methods for higher-order
  unfitted FEM.

## 6. Physical Geometry Mapping And Interface Normals

Decision: generate all high-order cut quadrature in parent reference
coordinates, then use existing or extended assembly mapping to map volume
weights, surface weights, and normals to physical space.

Rationale: svMultiPhysics assembly already expects cut rules in reference space
and maps them through parent-cell geometry. This is the correct place to account
for curved physical elements. Saye-style rules define the implicit geometry in
the reference integration domain; the parent FE mapping then controls physical
measure.

### Reference-To-Physical Mapping Contract

All current high-order implicit backends define the level-set polynomial as a
reference-cell FE field, `phi_h(xi)`. The generated cut quadrature points,
weights, and normals are therefore reference-frame data until assembly maps
them through the parent cell.

Reference interface normals are
`n_ref = grad_xi(phi_h) / ||grad_xi(phi_h)||`, oriented by the existing
negative/positive side convention. A zero, singular, or nonfinite reference
gradient is a diagnostic condition; it must not silently produce a usable normal.

Physical normals are computed from the inverse-transpose geometry map:
`n_x = normalize(J^{-T} n_ref)`. The sign remains tied to the reference-side
orientation, so negative-side and positive-side forms see opposite normals only
through the existing side convention.

Reference volume weights map to physical volume weights by
`w_x = w_ref * abs(det J(xi))`. Reference interface weights map by the cofactor
surface scaling
`w_surface_x = w_surface_ref * abs(det J(xi)) * ||J^{-T}(xi) n_ref||`.
This formula covers line interfaces in 2D parent cells and surface interfaces in
3D parent cells. Assemblers must reject nonfinite or singular geometry Jacobians
for active high-order cut rules instead of falling back to stale linear rules.

Diagnostics should keep reference and physical measures separate:

- reference cut volume and reference interface measure from the backend rule,
- physical cut volume and physical interface measure after parent mapping,
- maximum normal-transform residual or nonfinite-normal count when checks are
  enabled,
- rule frame, parent geometry revision, and backend policy fields.

### Standard Assembler Audit Notes

`StandardAssembler::assembleCutVolumes(...)` accepts high-order cut-volume
rules through `CutVolumeQuadratureRule`, which copies every backend point and
weight into a transient quadrature rule. The initial `AssemblyContext::reserve`
call is only a capacity hint; `setQuadratureData(...)` grows the arena to the
actual point count. The volume path already rejects non-reference-frame rules,
empty rules, and out-of-range parent cells. Full-side rules intentionally use
the parent element's standard full-cell quadrature and should remain reserved
for `full_cell_equivalent` rules only.

`StandardAssembler::assembleCutInterfaces(...)` uses the same transient
quadrature wrapper for generated interface rules, remaps reference normals and
surface weights through `remapCutInterfaceSurfaceGeometry(...)`, and checks the
rule/context point-count match. Generated interface assembly still rejects
material state on cut interfaces; this remains an explicit limitation until
material state storage is keyed by generated interface quadrature shape.

### Design Checklist

- [x] Define reference normals as normalized `grad_xi(phi_h)`.
- [x] Define physical normals as inverse-transpose mapped reference normals.
- [x] Define physical surface measure scaling for 2D and 3D.
- [x] Define volume weight mapping through `abs(det J)`.
- [x] Define behavior for singular or nonfinite parent geometry Jacobians.
- [x] Define whether `phi_h` is a function of reference coordinates or physical
      coordinates for each backend. Recommended first milestone: reference-cell
      FE field `phi_h(xi)`.
- [x] Define diagnostic fields for reference and physical measure separately.

### Implementation Checklist

- [x] Audit `StandardAssembler::assembleCutVolumes(...)` and
      `assembleCutInterfaces(...)` assumptions for high-order rule point counts.
- [x] Extend interface remapping tests to high-order and curved parent geometry.
- [x] Add optional backend-provided normal consistency checks.
- [x] Keep physical wet-volume diagnostics based on mapped cut-volume rules.
      `ActiveDomainOutput::collectCutVolumeMeasures(...)` accumulates reference
      rule measures separately from physical measures and maps every retained
      reference-frame cut-volume quadrature point through the parent geometry
      Jacobian before wet-volume drift diagnostics choose the physical frame.
- [x] Ensure high-order rules do not reuse stale linear full-cell-equivalent
      assumptions.

### Tests

- [x] Affine parent cell: reference and physical measures agree by constant
      Jacobian scaling.
      Covered by
      `ActiveDomainOutput.CollectsPhysicalCutVolumeMeasureOnScaledQuad`.
- [x] Curved parent cell: physical measure differs from reference measure and is
      computed by mapped quadrature.
      Covered by
      `LevelSetInterfaceLifecycle.SayeHyperrectangleReferenceRulesMapToCurvedHexGeometry`.
- [x] Normal orientation test for negative and positive active sides.
- [x] Surface traction sign test using a simple embedded interface with known
      normal.

### Literature Support

- Saye 2015 supports computing quadrature over implicit geometry in a reference
  integration domain.
- Higher-order FEM geometry practice and the finite cell method literature
  support separating reference integration from physical mapping through curved
  geometry.
- The current CutFEM/free-surface implementation already uses mapped parent-cell
  fields, so this decision preserves the established assembly architecture.

## 7. Cut Context Metadata, Revision Keys, And Cache Invalidation

Decision: high-order cut rules must carry enough metadata to invalidate caches
when the implicit field, backend policy, quadrature order, tolerance, topology,
or fallback state changes.

Rationale: high-order implicit quadrature introduces more state than the current
linear cutter. Stale quadrature is a correctness bug, especially when cut
topology changes during Newton or line-search trials.

### Metadata Contract

The generated-interface request is the run-level metadata carrier. It owns the
geometry mode, backend name, fallback policy, requested interface and volume
orders, root tolerance, maximum subdivision depth, mesh revisions, ownership
revision, source layout revision, and source value revision.

Each exported `CutQuadratureRule` must mirror enough request metadata in
`CutQuadratureProvenance` to make a rule self-describing outside the domain:
parent cell, marker, frame, topology id and revision, policy key, construction
kind, implicit geometry mode, implicit backend name, fallback policy, requested
order, achieved order, and source value revision. The assembly-facing
`CutCellAssemblyMetadata` mirrors the parent entity, side, active volume
fraction, embedded normal, topology id, topology revision, quadrature policy
key, and source value revision so stale generated rules can be rejected after a
rebuild.

Per-cell backend status and fallback reason are diagnostic metadata. They should
use `ImplicitCutQuadratureDiagnosticStatus` names and backend diagnostic strings
for logging/debug output, but they do not by themselves change sparsity unless
they also change active rules, topology ids, or rule visibility.

### Policy And Revision Key Contract

The generated-interface `quadrature_policy_key` is the backend-policy component
of the cut-context revision. It must include the geometry mode, backend family,
fallback policy, requested generic/interface/volume quadrature orders, root and
classification tolerances, maximum subdivision depth, degenerate-fragment policy,
and corner-linearization policy. The application cut-context topology key then
mixes that policy key with the active fragment and active volume-region topology,
so changing a requested order or backend option invalidates cut-context users
even when the zero-contour topology remains unchanged.

This partition keeps source-value changes, mesh revisions, and active cut
topology separate from backend policy changes while still making every generated
rule self-describing through its provenance. Output-only diagnostic strings may
change without forcing rebuilds unless the policy enum, achieved rule, or active
topology also changes.

### Metadata Consumer Matrix

Operator sparsity depends on parent-cell and facet visibility, active side
visibility, marker ids, cut-adjacent interior facets, conditioning aggregation
patches, ownership/numbering revisions, FE space revisions, and active cut
topology ids. It must not depend on diagnostic text, backend status strings, or
achieved quadrature order when the active integration support and local DOF
visibility are unchanged.

Matrix-free data depends on every field that can change integration points,
weights, normals, mapped geometry, local basis values, or stabilization hooks:
source value revision, mesh geometry revision, cut topology revision,
quadrature policy key, construction kind, frame, active side, and conditioning
revision. A matrix-free operator may reuse sparsity-independent data only when
these values match.

Preconditioner metadata follows the assembled or matrix-free operator that it
approximates. Changes to cut topology, quadrature policy, volume fractions,
conditioning neighborhoods, FE block layout, or active constraints require a
preconditioner refresh even if the algebraic sparsity pattern is unchanged.

Output-only metadata includes backend diagnostic strings, per-cell fallback
reasons, human-readable backend names, requested-versus-achieved order reports,
debug counters, and visualization labels. These fields may be written to restart
or diagnostic output, but by themselves they do not force a sparsity rebuild,
matrix rebuild, matrix-free rebuild, or preconditioner refresh.

### MPI Ordering Contract

Generated cut metadata must be deterministic before it reaches MPI-visible
assembly, output, or diagnostics. Interface and volume rules are exported in a
canonical order: parent entity, side, marker, topology id, then topology
revision. Stable ids are derived from marker, parent entity, local cut index,
side for volume regions, and source value revision; they must not depend on
rank-local insertion order or ownership revision.

Global summaries should aggregate commutative scalar totals and counts only
after local rules have been canonicalized. Debug output that lists per-cell or
per-rule metadata must use the same canonical order so rank-to-rank comparisons
and restart diffs remain meaningful. Existing deterministic ordering tests in
`LevelSetInterfaceDomain` and the high-order MPI rule-hash test in
`LevelSetInterfaceBuilderMPI` are the regression anchors for this contract.

### Tiny-Sliver Pruning Contract

High-order backends must not export tiny positive-measure slivers as active
assembly rules merely because a recursive or subcell construction found a
topological fragment. The backend capability advertises
`prunes_tiny_slivers_in_context` and `tiny_sliver_volume_fraction`; the default
policy is to treat slivers below that relative parent-volume threshold as
conditioning diagnostics, not as active quadrature support. This keeps positive
weights and local conditioning consistent with CutFEM practice for small-cut
stabilization.

Pruned slivers are excluded from assembly-facing active interface and volume
rules. They may be retained only as inactive diagnostic metadata when
`keep_degenerate_fragments` is enabled, and their diagnostic status should be
`Tangent`, `Degenerate`, or `ExactNoCut` depending on the local classification.
They are included in per-cell diagnostics, fallback counts, and visualization
debug output, but not in active measures, sparsity decisions, local residuals,
or matrix entries. If conservation requires retaining the tiny volume, it must
be merged into the complementary active volume rule or handled by a documented
conditioning/aggregation policy rather than emitted as a standalone unstable
rule.

### Design Checklist

- [x] Extend cut metadata with:
      - geometry mode,
      - backend name,
      - requested volume and interface orders,
      - achieved volume and interface orders,
      - level-set polynomial order,
      - root tolerance,
      - subdivision depth,
      - fallback policy and fallback reason,
      - backend status,
      - source value revision.
- [x] Include backend policy fields in the topology/revision key.
- [x] Define which metadata affects operator sparsity, matrix-free data,
      preconditioner metadata, and output only.
- [x] Define deterministic metadata ordering for MPI.
- [x] Define pruning policy for high-order tiny slivers.
- [x] Define whether pruned high-order slivers are included in diagnostics but
      excluded from assembly.

### Implementation Checklist

- [x] Extend `CutQuadratureProvenance` or `CutCellAssemblyMetadata`.
- [x] Add revision-key tests covering backend option changes.
- [x] Add stale-rule assertions for high-order marker/side rules.
- [x] Ensure full-cell rules still carry enough metadata for diagnostics.
- [x] Add cut-context rebuild logging for high-order fields.

### Tests

- [x] Changing requested quadrature order changes the cut-context revision.
- [x] Changing backend tolerance changes the cut-context revision.
- [x] Changing a high-order level-set edge DOF changes the source revision and
      generated rules.
- [x] Rejected line-search trial geometry is not left installed.
- [x] MPI ranks produce matching global cut-context summaries.

### Literature Support

- Hansbo, Larson, and Zahedi 2016 emphasize direct evaluation on the moving
  level-set-defined domain in time-dependent CutFEM.
- Saye 2015 and Fries et al. 2017 imply that quadrature policy and geometry
  approximation are part of the discrete method and must be tracked.

## 8. Assembly Integration

Decision: keep physics and form code backend-agnostic. High-order generated
rules should enter assembly through the existing `dCutVolume(...)`, `dI(...)`,
and cut-adjacent facet mechanisms.

Rationale: the current architecture correctly separates FE geometry/quadrature
from Navier-Stokes free-surface semantics. The high-order project should improve
the generated rules without requiring physics modules to know which cutter made
them.

Assembly integration contract:
- Public form syntax remains unchanged. Physics modules continue to request
  active-domain terms through `dCutVolume(marker, side)`, generated interface
  terms through `dI(marker)`, and stabilization terms through
  `cutAdjacentFacetIntegral(..., marker)`. Backend choice, requested order,
  achieved order, fallback use, and sliver pruning stay in cut-rule provenance
  and cut-context metadata.
- Generated `dI(marker)` rules support fixed-geometry interface integration,
  including mapped normals and physical measures. Material-state storage on
  generated interfaces remains unsupported until a material-history key and
  transfer policy are added for generated interface quadrature points.
- Empty generated high-order rules are not valid assembly inputs. Pruned rules
  may remain visible in diagnostics, but retained assembly rules must contain at
  least one finite quadrature point with finite weights. Fallback rules enter
  assembly only when their provenance records the fallback backend and policy.
- Full-cell-equivalent rules are selected only from explicit cutter metadata,
  not from a floating-point measure comparison in the assembler. This prevents
  near-full curved cuts from silently bypassing generated high-order geometry.
- High-order generated rules may exceed fixed historical reserve guesses such
  as 27 points. Assembly code must either reserve from selected rule point
  counts or grow `AssemblyContext` on demand before mapping basis values,
  geometry, normals, and solution fields.

Reserve-size audit:
- `AssemblyContext` has on-demand growth in geometry, basis, solution, and
  frame-explicit setters, so fixed initial reserve sizes are no longer a
  correctness dependency. Cut-volume and generated-interface assembly should
  still reserve from the selected generated rule point counts before entering
  cell loops to avoid hidden reallocations and to make high-order point-count
  assumptions explicit.

Many-point cut-volume basis policy:
- Transient generated cut-volume rules must use uncached basis evaluations keyed
  only to the rule instance in the current assembly pass. High point counts are
  valid as long as the context reserves enough storage and the local DOF
  insertion receives the same row and column maps as ordinary cell assembly.

Many-point generated-interface remapping policy:
- Generated interface rules use the same parent-cell basis evaluation as
  cut-volume rules, then remap reference normals and weights through the parent
  geometry before the form kernel runs. The remapping loop must be point-count
  agnostic and cannot assume legacy face quadrature sizes.

JIT specialization key contract:
- Cut-volume JIT cache keys must distinguish the integral domain, interface
  marker, selected side, specialized quadrature point count, test and trial DOF
  counts, affine-geometry mode, and baked-basis hash. Baked-basis hashes are the
  contract boundary for rule-specific basis and quadrature tables, including
  high-order generated rules.

Cut-adjacent facet-set locality audit:
- Marked cut-adjacent interior-face assembly is driven by
  `CutFacetSetHandle` membership, not by scanning every interior face near a
  marker. Existing DG coverage requires a cut context, rejects missing facet-set
  handles, and checks that a P2 high-order marked facet leaves the far-field
  cell block exactly zero.

### Design Checklist

- [x] Preserve the public form vocabulary:
      - `dCutVolume(marker, side)`,
      - `dI(marker)`,
      - `cutAdjacentFacetIntegral(..., marker)`.
- [x] Define any assembly limitations for high-order generated interfaces, such
      as material state on generated `dI`.
- [x] Define behavior for empty high-order rules and fallback rules.
- [x] Define how full-cell-equivalent rules are selected for high-order fields.
- [x] Define whether high-order interface rules can exceed current context
      reserve sizes.

### Implementation Checklist

- [x] Review `AssemblyContext::reserve(...)` assumptions for max quadrature
      point counts.
- [x] Ensure cut-volume rules with many points do not disable needed basis
      evaluation caching.
- [x] Ensure interface rules with many points are remapped correctly.
- [x] Ensure JIT specialization keys include enough quadrature shape metadata.
- [x] Keep marked cut-adjacent interior-face assembly local to the generated
      facet set.

### Tests

- [x] Dispatch high-order generated cut-volume and interface rules through
      fixed-geometry assembly loops.
- [x] Assemble a scalar mass form over a curved cut volume and compare to
      analytic measure.
- [x] Assemble linear and quadratic polynomial moments over curved cut volumes.
- [x] Assemble a surface traction over a curved interface and compare to analytic
      surface integral.
- [x] Verify high quadrature point counts do not corrupt local DOF insertion.
- [x] Verify JIT and interpreter paths agree.

### Literature Support

- CutFEM papers by Burman, Hansbo, Larson, and Zahedi support direct integration
  over bulk domains and surfaces defined by level sets.
- Saye 2015 supports generating surface and volume quadrature rules that can be
  consumed by embedded-boundary finite-element methods.
- Fries et al. 2017 emphasize that high-order integration over implicitly
  defined domains must preserve the FE weak-form interface while the quadrature
  construction supplies method-specific points, weights, and diagnostics.

## 9. Nonlinear Coupling And Tangent Policy

Decision: first milestone should use a documented quasi-Newton geometry tangent:
refresh cut geometry from the exact current nonlinear state before residual and
Jacobian assembly, but do not include derivatives of quadrature points, weights,
normals, or active-domain topology with respect to level-set DOFs. Exact shape
sensitivities can be a later milestone.

Rationale: the current unfitted free-surface path already made a major
correctness improvement by refreshing cut geometry from the current nonlinear
state. Exact geometry derivatives are substantially harder because the
quadrature construction itself depends on roots, recursive branches, and
topological classification. A documented quasi-Newton contract is practical and
testable, while full sensitivities need their own research-grade implementation
plan.

Tangent policy contract:
- The initial high-order generated-interface policy is
  `geometry_tangent_policy=RefreshedFrozenQuadrature`. Residual and Jacobian
  assembly refresh the cut context from the current accepted or trial nonlinear
  state, but the Jacobian treats generated quadrature points, weights, normals,
  and active topology as fixed during that assembly.
- The reserved future exact-sensitivity policy is
  `geometry_tangent_policy=DifferentiatedQuadrature`. That policy requires
  explicit derivatives for quadrature weights, point locations, normals,
  measure factors, and topology-transition handling before it can be enabled.
- Diagnostics distinguish fixed-geometry finite-difference checks, refreshed
  geometry residual checks, and full geometry-perturbation checks. A
  fixed-geometry check should pass to tangent tolerance; a refreshed-geometry
  check may show the documented quasi-Newton mismatch; a full perturbation
  check is expected to remain unsupported until `DifferentiatedQuadrature`.
- Solver convergence under `RefreshedFrozenQuadrature` is quasi-Newton:
  residuals must use current geometry, line-search trials must refresh trial
  geometry, rejected trials must restore the accepted cut context, and quadratic
  Newton convergence is not promised for residuals dominated by geometry
  motion.
- Runs with coupled level-set geometry and omitted exact geometry derivatives
  must log the tangent policy and a warning-level diagnostic when high-order
  generated interfaces participate in nonlinear assembly.

### Design Checklist

- [x] Name the initial policy, for example
      `geometry_tangent_policy=RefreshedFrozenQuadrature`.
- [x] Define a future policy name for exact sensitivities, for example
      `DifferentiatedQuadrature`.
- [x] Define finite-difference diagnostics that distinguish:
      - fixed-geometry residual/Jacobian checks,
      - refreshed-geometry residual checks,
      - full geometry-perturbation checks.
- [x] Define solver expectations for Newton convergence under the quasi-Newton
      policy.
- [x] Define warning logs when coupled level-set geometry is active and exact
      geometry derivatives are omitted.

### Implementation Checklist

- [x] Add the tangent policy to options and diagnostics.
- [x] Ensure existing Newton sync callbacks remain active for high-order mode.
- [x] Ensure line-search trial refreshes use the same high-order backend.
- [x] Add fixed-geometry assembly diagnostics for high-order cut rules.
- [x] Add a future design stub for quadrature-weight and normal sensitivities.

### Tests

- [x] Fixed-geometry finite-difference Jacobian checks pass for high-order rules.
- [x] Refreshed-geometry finite-difference checks report the expected
      quasi-Newton mismatch rather than silently passing or failing unclearly.
- [x] Line-search rejection restores accepted high-order cut context.
- [x] A moving curved-interface MMS case converges under documented tolerances.

### Literature Support

- Time-dependent CutFEM literature supports updating the level-set-defined
  domain consistently in time.
- Saye-style quadrature and high-order implicit meshing literature make clear
  that quadrature construction is geometry-dependent, motivating an explicit
  tangent-policy contract rather than an accidental tangent omission.

## 10. High-Order Cut-Cell Stabilization

Decision: keep stabilization local to cut-adjacent facets, but extend the ghost
penalty design for higher-order spaces. For degree `p`, the target CutFEM design
penalizes jumps of normal derivatives up to order `p`, with degree-appropriate
mesh scaling.

Rationale: small cut elements cause conditioning and stability issues independent
of quadrature accuracy. The current gradient-jump path is appropriate for low
order. Higher-order spaces generally need higher derivative control or a
carefully justified alternative such as aggregation.

Milestone status: the implementation assembles first-gradient cut-adjacent
ghost penalties for all supported spaces and second-normal-derivative
cut-adjacent ghost penalties for quadratic velocity and pressure spaces.
Stabilization setup records the velocity and pressure polynomial orders,
derivative orders applied to each field, and implemented scaling. Runs requiring
derivative orders above two receive an explicit warning that additional
higher-normal-derivative penalties are not yet available.
Small-cut conditioning probes now verify finite conditioning indicators and
capped cut-adjacent stabilization scales as retained cut fractions decrease to
the generated-volume pruning threshold.
High-order marked-facet assembly now has an explicit P2 regression proving that
unmarked far-field cells receive zero cut-adjacent stabilization contribution.
Form-level cut-adjacent gradient-penalty tests verify the implemented trace
height scaling: first-gradient velocity penalties scale with `h`, while the
pressure-gradient path scales with `h^3`.
The cut-cell form vocabulary now exposes second-normal-derivative jump helpers
backed by the existing Hessian kernels, with compiler coverage proving P2
cut-adjacent facet forms request basis Hessians.

### Design Checklist

- [x] Define stabilization families:
      - current first-gradient jump,
      - higher-normal-derivative ghost penalty,
      - aggregation/discrete extension as future option.
- [x] Define default penalty terms by velocity and pressure polynomial order.
- [x] Define scaling, including `h^(2*j - 1)` style factors for derivative order
      `j`.
- [x] Define which terms are required for equal-order velocity-pressure pairs.
- [x] Define how cut-volume fraction scale/cap interacts with high-order
      stabilization.
- [x] Define local facet set scope for high-order mode.

### Implementation Checklist

- [x] Extend form vocabulary or kernels for higher normal-derivative jumps if not
      already available.
- [x] Include polynomial order in stabilization setup.
- [x] Preserve active-side facet-set generation.
- [x] Add diagnostics for each derivative-order penalty used.
- [x] Ensure `Use_cut_metadata_scale=false` remains local, not global.
- [x] Add a stability warning if high-order geometry is enabled without adequate
      high-order stabilization.

### Tests

- [x] Only cut-adjacent facets receive stabilization.
- [x] Far-field interior faces receive zero contribution.
- [x] Derivative-order terms scale correctly with `h`.
- [x] Small-cut conditioning probes remain bounded under refinement.
- [x] Direct factorization diagnostics show no dry unsupported pressure rows.

### Literature Support

- Massing et al. 2014 supports ghost penalties for velocity and pressure near
  cut boundaries to obtain stability and bounded conditioning for Stokes.
- de Prenter, Verhoosel, and van Brummelen 2023 review small-cut conditioning
  remedies, including ghost penalties and aggregation.
- Burman, Hansbo, Larson, and Zahedi CutFEM papers support stabilization terms
  for robust unfitted bulk-surface discretizations.

## 11. Pressure Support And Active-Domain Constraints

Decision: keep pressure support based on active cell support, not vertex sign.
High-order mode must generalize this to all scalar H1 pressure DOFs whose basis
has no retained active-domain support.

Rationale: high-order pressure fields introduce edge, face, and cell-interior
DOFs. Vertex-sign masking is not a valid support test for such DOFs. The current
active-support direction is correct and should be preserved.

Milestone status: active-side pressure support constraints now report support
and constrained counts by vertex, edge, face, cell-interior, and unknown DOF
entity. P2 quadrilateral pressure tests explicitly verify that an edge DOF on a
retained wet cut cell remains unconstrained, while an edge DOF supported only by
a pruned or inactive cell is constrained. Direct assembled-matrix coverage now
checks that constrained dry high-order pressure rows receive identity diagonals
instead of zero rows.

### Design Checklist

- [x] Define active pressure support from retained generated volume rules.
- [x] Define support tolerance for high-order small fractions.
- [x] Define behavior for pressure DOFs supported only by pruned slivers.
- [x] Define diagnostics for vertex, edge, face, and cell-interior pressure DOF
      support.
- [x] Define whether inactive support constraints depend on physical or
      reference wet volume. Recommended: retained assembly support, with
      physical measure reported separately.

### Implementation Checklist

- [x] Audit pressure constraint code for all entity DOF types.
- [x] Ensure high-order generated volume metadata marks active cells correctly.
- [x] Rebuild constraints after every high-order cut-context refresh.
- [x] Add diagnostics split by entity dimension.
- [x] Ensure constrained dry rows receive valid diagonals in all backends.
      The constrained-row sparse-structure regression disables blanket
      diagonal padding and still requires diagonal slots for dry high-order
      pressure rows, so Eigen, FSILS, PETSc, and Trilinos receive a valid
      insertion location before their matrix-specific `setDiagonal` path runs.

### Tests

- [x] P2 pressure edge DOF on a wet cut cell remains unconstrained.
- [x] P2 pressure edge DOF with no active cell support is constrained.
- [x] Cell-interior pressure DOFs behave correctly in high-order cells.
- [x] No zero pressure rows remain in direct factorization diagnostics.

### Literature Support

- The small-cut and fictitious-domain Stokes literature motivates controlling
  unsupported or weakly supported degrees of freedom.
- The current CutFEM support strategy matches the principle that algebraic
  support should follow the active physical domain rather than visualization
  masks.

## 12. Level-Set Maintenance And Conservation

Decision: high-order geometry support must not be treated as a mass-conservation
fix. Continue to monitor physical wet volume, and keep reinitialization and
volume correction as explicit validation controls.

Rationale: accurate quadrature reduces geometric integration error, but plain
level-set advection can still lose or gain volume. Conservative level-set
methods and volume correction remain separate numerical-method choices.

### Design Checklist

- [x] Define high-order-compatible wet-volume diagnostics.
      Wet-volume diagnostics are based on retained generated cut-volume rules,
      not vertex masks. Each report carries reference wet volume, physical wet
      volume mapped through the parent-cell geometry, rule counts, skipped
      physical-measure counts, wet-volume frame, drift, and relative drift. A
      physical frame is required for high-order validation; reference fallback is
      diagnostic-only evidence that a physical measure path is missing.
- [x] Define whether volume correction targets reference or physical volume.
      Validation targets physical wet volume. The current global-shift
      correction remains a scalar level-set coefficient shift and must be
      compared against the high-order physical wet-volume diagnostic before it
      is accepted for high-order runs.
- [x] Define whether reinitialization preserves high-order coefficients or
      projects back to a lower order.
      The projection reinitialization path preserves the full coefficient vector
      layout and updates only vertex coefficients in the first milestone; it
      does not project the field to a lower-order space and does not claim a
      curved high-order signed-distance reconstruction.
- [x] Define acceptable zero-contour displacement caused by maintenance.
      Validation runs must record `max_interface_displacement`,
      `l2_interface_displacement`, and sample count. The default acceptance gate
      is `max_interface_displacement <= max(signed_distance_tolerance,
      0.05*h_min)` for the maintained band unless a case-specific analytic
      tolerance is documented.
- [x] Define conservation tolerances for validation cases.
      Fixed-geometry validation should keep absolute physical wet-volume drift
      below `max(1.0e-10, 1.0e-8*initial_physical_wet_volume)`. Transient smoke
      runs keep the existing relative drift warning threshold as a diagnostic
      screen and must record the threshold used.

### Implementation Checklist

- [x] Update wet-volume diagnostics to use high-order physical cut measures.
      The active-domain output path integrates retained reference-frame
      high-order cut rules through the current parent-cell geometry and records
      physical wet volume separately from reference measure.
- [x] Ensure reinitialization updates current and previous history consistently.
      Runtime maintenance copies the repaired FE-ordered coefficients for each
      modified level-set field into both the accepted state and the previous
      state, then refreshes ghosts and reports current/previous and older
      history deltas. A Q2 regression test verifies that the complete
      high-order field slice, not only vertex coefficients, is synchronized
      while unrelated fields are left unchanged.
- [x] Ensure volume correction updates the high-order field coefficients.
      The global-shift correction operates on the complete level-set field slice
      and the system overload copies the complete corrected slice back into the
      FE-ordered solution. A Q2 regression verifies that all high-order
      coefficients receive the accepted shift while unrelated fields remain
      unchanged.
- [x] Rebuild high-order cut context after maintenance and before output.
      The transient accepted-step callback applies level-set maintenance,
      refreshes the active cut integration context from the maintained accepted
      state, logs wet-volume diagnostics from that refreshed context, and only
      then writes VTK output. A high-order Saye regression verifies that a Q2
      maintained field rebuilds a HighOrderImplicit cut context whose output
      wet volume follows the corrected coefficients rather than a stale context.
- [x] Log maintenance-induced geometry changes.
      Runtime logs now expose the quantities needed to audit maintenance-driven
      interface motion: projection reinitialization reports interface
      displacement samples, maximum and L2 displacement, and maximum field
      update; volume correction reports target, initial and corrected volumes,
      achieved volume error, applied shift, and iterations; history
      synchronization reports modified fields and current/previous deltas; the
      accepted-step cut-context refresh reports cut-context revision, retained
      cells, reference and physical wet volumes, and cut-adjacent facets before
      wet-volume diagnostics and VTK output are written.

### Tests

- [ ] Volume correction changes high-order cut-volume diagnostics before output.
- [ ] Reinitialization does not move the zero contour beyond tolerance.
- [ ] Long-run wet-volume drift reports physical high-order wet volume.
- [ ] Conservative and nonconservative level-set runs are distinguishable in
      diagnostics.

### Literature Support

- Sussman, Smereka, and Osher 1994 motivates signed-distance reinitialization in
  level-set two-phase flow.
- Olsson and Kreiss 2005 motivates conservative level-set transport and makes
  clear that conservation is a separate concern from interface representation.

## 13. Diagnostics, Output, And Visualization

Decision: high-order mode must report the actual integration geometry and
quadrature quality. `WetVolumeFraction` should remain the active-domain
post-processing quantity, but high-order mode must expose physical wet measure
and backend diagnostics.

Rationale: curved implicit geometry is difficult to debug from vertex signs.
Without explicit quadrature diagnostics, failures can masquerade as physics or
linear-solver problems.

### Design Checklist

- [ ] Define per-run diagnostics:
      - geometry mode,
      - backend,
      - requested/achieved orders,
      - tolerance,
      - fallback counts,
      - failed-cell counts,
      - min/max volume fraction,
      - physical wet volume,
      - reference wet volume.
- [ ] Define optional per-cell diagnostics for debug builds or smoke scripts.
- [ ] Define high-order curved interface visualization strategy:
      - tessellated interface output,
      - sample points,
      - or backend debug dumps.
- [ ] Define how `ActiveFluid` remains a vertex-sign visualization only.

### Implementation Checklist

- [ ] Extend cut-context rebuild log fields.
- [x] Extend smoke-script parsing for high-order backend diagnostics.
- [ ] Extend `WetVolumeFraction` output to use high-order retained active rules.
- [ ] Add optional curved-interface surface output for inspection.
- [ ] Warn when vertex-sign `ActiveFluid` and high-order wet fractions disagree.

### Tests

- [ ] Output writes high-order `WetVolumeFraction`.
- [ ] Debug output identifies fallback cells.
- [x] Curved-interface visualization contains nonzero surface data.
- [x] Smoke parsing fails when high-order mode silently falls back too often.

### Literature Support

- Saye 2015 and implicit-geometry FEM papers emphasize quadrature accuracy over
  implicitly defined domains; diagnostics should expose the selected quadrature
  policy and achieved accuracy.
- CutFEM validation practice relies on checking physical measures and active
  domain support, not only nodal signs.

## 14. Analytic Geometry And Quadrature Validation

Decision: do not qualify high-order mode through solver benchmarks first.
Qualify the geometry and quadrature layer independently with analytic integrals,
then move to assembly and solver tests.

Rationale: high-order cut quadrature can be validated against known geometry
without Navier-Stokes nonlinearities. This isolates the new implementation and
prevents solver behavior from hiding geometry errors.

### Design Checklist

- [ ] Define analytic 2D cases:
      - line through quad,
      - circle segment,
      - ellipse segment,
      - polynomial interface.
- [ ] Define analytic 3D cases:
      - plane through hex,
      - sphere cap,
      - ellipsoid-like polynomial where feasible.
- [ ] Define polynomial moment tests up to the achieved quadrature order.
- [ ] Define convergence criteria under h-refinement and p-refinement.
- [x] Define tests for non-affine physical parent mappings.

### Implementation Checklist

- [x] Add unit-test helpers for analytic measures and polynomial moments.
- [ ] Add backend-independent quadrature validation fixtures.
- [x] Add separate tests for reference measure and physical mapped measure.
- [ ] Add deterministic randomized cut-position tests with fixed seeds.
- [ ] Add failure-mode tests for near-tangent and tiny sliver cuts.

### Tests

- [x] Constant integration exactness over both cut sides.
- [x] Linear and quadratic moment accuracy.
- [x] Interface length/area accuracy.
- [ ] Negative plus positive measures sum to parent measure.
- [ ] Rule weights and points are finite.
- [ ] Achieved-order diagnostics match measured convergence.

### Literature Support

- Saye 2015 validates high-order implicit quadrature through convergence tests.
- Moment-fitting and high-order implicit meshing papers use moment and analytic
  geometry tests as standard evidence.

## 15. Assembly And Solver Validation

Decision: after analytic geometry tests pass, validate assembly with fixed
geometry finite differences, then validate moving-geometry solver behavior under
the documented tangent policy.

Rationale: geometry/quadrature correctness and residual/Jacobian correctness are
separate gates. The current unfitted free-surface remediation already found that
finite-difference component checks are valuable for isolating assembly defects.

### Design Checklist

- [ ] Define fixed-geometry form tests for:
      - scalar mass,
      - pressure-divergence coupling,
      - viscous terms,
      - VMS terms,
      - cut-adjacent stabilization.
- [ ] Define component-selectable finite-difference checks.
- [x] Define moving-interface MMS cases for high-order geometry.
- [ ] Define D18/D38 qualification policy: only after unit, component, and smoke
      gates pass.

### Implementation Checklist

- [x] Add a fixed-geometry measure dispatch test for high-order generated
      `dCutVolume` and `dI` rules.
- [x] Add fixed-geometry finite-difference tests for high-order `dCutVolume`.
- [x] Add fixed-geometry finite-difference tests for high-order `dI`.
- [x] Add high-order cut-adjacent stabilization assembly tests.
- [x] Add a compact high-order MMS traveling-interface fixture.
- [x] Preserve reference-corner topology for high-order geometry meshes during
      DOF distribution.
- [x] Preserve biquadratic and triquadratic VTK reference corner counts for
      high-order fixture meshes.
- [x] Use full 2D edge storage for Q2 DOF distribution when only boundary faces
      are retained for solver boundary work.
- [x] Extract endpoint corners from high-order 2D face geometry when deriving
      reference edge ordering.
- [x] Sync prescribed high-order mesh point fields into vertex, edge, and cell
      FE coefficients for MMS source data.
- [x] Sync wet-extension level-set advection velocity data into high-order mesh
      point coefficients.
- [x] Refresh active level-set mesh point fields from high-order FE solution
      coefficients before cut-context rebuilds.
- [x] Match high-order face connectivity through reference corners for
      cell-to-face assembly ordering.
- [x] Regenerate compact high-order MMS smoke meshes in temporary run
      directories and route the diagnostic shortcut through a serial
      unstabilized solve so the high-order implicit cut path advances an
      accepted step without modifying the canonical MMS fixture.
- [x] Add smoke-script switches for high-order backend and fallback thresholds.
- [x] Add benchmark qualification logs that record backend diagnostics.

### Tests

- [ ] JIT and interpreter fixed-geometry matrix actions agree.
- [ ] Fixed-geometry finite-difference Jacobian checks pass by component block.
- [ ] Quasi-Newton moving-geometry checks report expected geometry-tangent
      residual differences.
- [x] Short high-order free-surface smoke run advances accepted steps with
      bounded cut-adjacent scales.
- [ ] D18/D38 are not requalified until high-order diagnostics are stable.

### Literature Support

- CutFEM papers support direct assembly over active cut domains and interfaces.
- High-order quadrature papers support separating quadrature verification from
  PDE solver validation.

## 16. Performance, Memory, And MPI Determinism

Decision: high-order quadrature must be deterministic and bounded in memory
before solver benchmarks use it. Performance can initially be slower than the
linear cutter, but diagnostics must identify cost and cache growth.

Rationale: recent D18/D38 work showed that basis-cache growth, repeated cut
context rebuilds, and cut-adjacent traversal density can dominate solver
behavior. High-order quadrature adds more points and more per-cell work.

Current smoke policy: the high-order MMS diagnostic shortcut records per-cell
cut statistics through the cut-context log (`cell_count`, `active_cut_cells`,
rule counts, quadrature point counts, and fragment counts), per-step assembly
density through assembly timings per accepted step, and resource bounds through
process RSS and basis-cache counters. The compact smoke ceiling is 4 assembly
timing records per accepted step, 3 extra assembly timings beyond nonlinear
iterations per accepted step, 6 cut-context rebuilds per accepted step,
300000 KB maximum RSS, 100000 KB RSS growth, 4 basis-cache entries, and 3
basis-cache entries of growth. The compact moving-interface convergence gate
also requires every requested step to be accepted and every reported nonlinear
and linear solve to converge.

Cache and invalidation policy: generated high-order rules are owned by the cut
context and are invalidated by the same geometry revision that invalidates
linear rules. The revision key must include the geometry mode, backend name,
requested volume and surface orders, tolerance, subdivision/root policy,
fallback policy, active side, marker, parent topology, parent geometry revision,
and level-set source revision. A Newton trial may reuse generated rules only
when all revision-key fields and the level-set coefficient source revision are
unchanged; rejected trial states must not replace the accepted cut context.
Longer-lived caches are limited to stable FE basis tabulations keyed by element
family, order, derivative order, and reference point set.

Determinism policy: each rank emits rules and diagnostics in a stable local
order sorted by global parent cell id, active side, marker id, parent topology,
and backend-local fragment id. Global diagnostics are formed with explicit MPI
reductions over numeric counters and with deterministic field ordering in logs.
No diagnostic may depend on pointer order, unordered container iteration, or
rank-local discovery order.

### Design Checklist

- [x] Define per-cell and per-step cost diagnostics.
- [x] Define maximum allowed basis-cache growth.
- [x] Define quadrature rule cache lifetime and invalidation.
- [x] Define MPI deterministic ordering for rules, metadata, and diagnostics.
- [x] Define memory ceilings for smoke runs.
- [x] Define whether high-order quadrature can be cached across Newton states
      when `phi_h` does not change.

### Implementation Checklist

- [x] Add backend timing counters.
- [x] Add point-count summaries by volume/interface rules.
- [ ] Keep transient basis evaluations out of process-wide caches unless keyed
      by stable persistent data.
- [x] Sort generated rules by parent cell, side, marker, and topology id.
- [ ] Aggregate MPI diagnostics globally.
- [ ] Add cache hit/miss counters for high-order quadrature if caching is used.

### Tests

- [ ] Repeated identical state rebuilds do not grow memory unbounded.
- [ ] MPI and serial runs produce matching global measure summaries.
- [x] Rule ordering is deterministic across runs.
- [x] High-order smoke run reports timing, memory, and cache diagnostics.

### Literature Support

- Saye 2015 emphasizes efficient high-order quadrature construction.
- Immersed FEM stability and conditioning literature motivates tracking small
  cut-element behavior because poor cut handling affects both conditioning and
  cost.

## 17. Rollout Plan

### Phase 0: Contracts And Infrastructure

- [x] Add options and feature gate.
- [x] Add backend abstraction.
- [x] Add high-order level-set evaluator.
- [x] Move linear cutter behind the backend interface.
- [x] Add metadata and diagnostics plumbing.
- [x] Keep production fixtures on the current linear path.

### Phase 1: 2D Hyperrectangle Proof

- [x] Implement initial Saye-style 2D quad backend.
- [x] Add analytic area, length, and moment tests.
- [x] Add fixed-geometry assembly tests.
- [x] Add diagnostics tests.
- [x] Add output tests.
- [x] Document limitations and fallback behavior.

### Phase 2: Simplex And Mixed Mesh Support

- [x] Select triangle strategy.
- [x] Implement triangle support.
- [x] Select tetrahedron strategy.
- [x] Implement tetrahedron support.
- [x] Add mixed-element diagnostics.

### Phase 3: 3D Hyperrectangle Support

- [x] Implement hex support.
- [x] Add sphere volume and surface tests.
- [x] Add ellipsoid volume and surface tests.
- [x] Add curved physical geometry mapping tests.
- [x] Add MPI determinism tests.

### Phase 4: Stabilization And Pressure Support

- [x] Extend ghost penalties for higher-order spaces.
- [x] Validate pressure support constraints for high-order pressure DOFs.
- [x] Add small-cut conditioning tests.
- [x] Add stabilization diagnostics.

### Phase 5: Solver Validation

- [x] Add high-order MMS traveling-interface fixture.
- [ ] Run fixed-geometry Jacobian component sweeps.
- [ ] Run quasi-Newton moving-geometry smoke tests.
- [x] Run short open-vessel probes with high-order diagnostics.
- [ ] Only then consider D18/D38 high-order qualification.

## 18. Definition Of Done

High-order curved implicit geometry/quadrature support is complete enough for
benchmark use only when all of the following are true:

- [ ] The high-order mode is explicitly configured and logged.
- [ ] P1 mode reproduces current linear-cutter behavior.
- [ ] P2/P3 level-set DOFs affect generated cut geometry.
- [ ] Analytic geometry tests pass for supported element families.
- [ ] Volume and interface quadrature report requested and achieved order.
- [ ] Cut-context metadata and revision keys include backend policy.
- [ ] Assembly fixed-geometry finite-difference checks pass.
- [ ] High-order stabilization is local and degree appropriate.
- [ ] Active pressure constraints cover unsupported high-order pressure DOFs.
- [ ] Physical wet-volume diagnostics are available and parsed by smoke scripts.
- [ ] MPI diagnostics and measures are deterministic.
- [x] Memory and cache growth are bounded in smoke runs.
- [ ] D18/D38 or other benchmarks have qualification logs recording high-order
      backend, orders, fallback counts, wet-volume history, nonlinear iterations,
      linear iterations, pressure range, and velocity range.

## Open Questions

- [ ] Should the first simplex backend use high-order subcell decomposition or
      moment fitting?
- [ ] Do D18/D38 meshes require high-order triangle/tet support before any
      high-order benchmark work is meaningful?
- [ ] What exact fallback rate is acceptable in validation runs?
- [ ] Are signed moment-fit weights acceptable anywhere in Navier-Stokes/VMS
      assembly, or should production require positive weights?
- [ ] Should exact geometry sensitivities be required before high-order mode is
      allowed in fully coupled level-set/fluid Newton solves?
- [ ] How should surface-tension curvature be projected or smoothed for
      high-order unfitted free surfaces?

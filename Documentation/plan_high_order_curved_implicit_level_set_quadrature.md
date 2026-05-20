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
- Conservative level-set transport or fitted/ALE-style kinematic enforcement for
  unfitted interfaces.
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
entity. P2/P3 quadrilateral pressure tests and a P3 tetrahedral shared-face
test explicitly verify that high-order DOFs on retained wet cut support remain
unconstrained, while DOFs supported only by dry, pruned, or inactive cells are
constrained. Direct assembled-matrix coverage now checks that constrained dry
high-order pressure rows receive identity diagonals instead of zero rows.

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

- [x] Volume correction changes high-order cut-volume diagnostics before output.
      Covered by
      `LevelSetVolume.VolumeCorrectionRefreshesHighOrderCutContextBeforeOutput`,
      which builds stale and corrected HighOrderImplicit Saye contexts from a
      Q2 level-set field and verifies the corrected context has a new revision
      and changed high-order cut-volume diagnostic before output consumption.
- [x] Reinitialization does not move the zero contour beyond tolerance.
      Covered by
      `LevelSetReinitialization.ProjectionPreservesZeroContourWithinTolerance`,
      which repairs a signed-distance plane whose interface spans a full quad
      cut segment and verifies the reported maximum and L2 interface
      displacement remain below the default `max(tol, 0.05*h_min)` gate.
- [x] Long-run wet-volume drift reports physical high-order wet volume.
      Covered by
      `ActiveDomainOutput.HighOrderWetVolumeDriftUsesPhysicalMeasure`, which
      routes high-order Saye-style cut-volume measures through the shared drift
      volume selector and verifies accepted-step drift is computed in the
      physical frame when no physical-measure rules are skipped.
- [x] Conservative and nonconservative level-set runs are distinguishable in
      diagnostics.
      Covered by
      `LevelSetOptions.ConservationDiagnosticsDistinguishRunModes`, which
      verifies plain advection, reinitialized advection without volume
      correction, and volume-corrected conservative runs emit distinct
      conservation diagnostic tokens. Runtime maintenance diagnostics now log
      that token alongside the enabled maintenance modes.

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

- [x] Define per-run diagnostics:
      - geometry mode,
      - backend,
      - requested/achieved orders,
      - tolerance,
      - fallback counts,
      - failed-cell counts,
      - min/max volume fraction,
      - physical wet volume,
      - reference wet volume.
      The per-run diagnostic contract is split between the active-domain
      `diagnostic=cut_context_rebuild` line and the wet-volume diagnostic line.
      The rebuild log reports geometry mode, backend, fallback policy, tangent
      policy, root tolerance, subdivision depth, requested and achieved
      interface/volume orders, fallback counts, corner-linearized cells,
      topology and policy keys, min/max retained volume fraction, active-side
      reference and physical volume, retained cell counts, and physical-measure
      skipped counts. The wet-volume diagnostic logs the selected wet volume,
      frame, reference wet volume, physical wet volume, drift, relative drift,
      rule counts, and wet/dry cell classification.
- [x] Define optional per-cell diagnostics for debug builds or smoke scripts.
      Optional per-cell diagnostics are carried by high-order backend diagnostic
      strings and the single-cell quadrature debug dump path. Cell-local records
      include parent cell identity, element type, backend, diagnostic status,
      fallback use, recursion depth, linearized leaf counts, root-finder
      iteration summaries, achieved orders, and generated rule counts. Normal
      runs aggregate these into cut-context rebuild logs; smoke scripts assert
      aggregate/backend substrings, while detailed JSON and VTP debug output
      remain opt-in inspection tools.
- [x] Define high-order curved interface visualization strategy:
      - tessellated interface output,
      - sample points,
      - or backend debug dumps.
      Curved-interface inspection uses VTP interface fragments as the primary
      visualization surface and per-parent-cell quadrature JSON as the detailed
      backend dump. The VTP path carries fragment points, segment or polygon
      connectivity, normals, curvature estimates, parent cells, interface
      markers, and retained side volume fractions. The JSON dump carries the
      generated volume and interface quadrature rules, provenance, frame,
      requested and achieved orders, policy, construction kind, and individual
      quadrature points. This keeps routine visualization lightweight while
      preserving enough backend state for debugging curved implicit cuts.
- [x] Define how `ActiveFluid` remains a vertex-sign visualization only.
      `ActiveFluid` remains a vertex field computed from nodal level-set signs
      for quick visualization and backward-compatible output. It is not a
      measure diagnostic and must not be used to infer high-order retained wet
      volume. `WetVolumeFraction` is the cell field tied to generated retained
      cut-volume rules; the output path logs this distinction and emits a
      warning when cut-cell vertex-sign fractions strongly disagree with the
      generated wet fractions.

### Implementation Checklist

- [x] Extend cut-context rebuild log fields.
      The active-domain rebuild log now emits high-order geometry mode,
      backend, backend timing, fallback policy/count, tangent policy/warning,
      root tolerance, subdivision depth, requested and achieved orders,
      revisions and policy keys, source revisions, retained/pruned reference and
      physical volumes, min/max retained volume fraction, active cell classes,
      quadrature point counts, cut-adjacent scale summaries, process memory,
      basis-cache size, and side-specific positive/negative volume summaries.
- [x] Extend smoke-script parsing for high-order backend diagnostics.
- [x] Extend `WetVolumeFraction` output to use high-order retained active rules.
      Output selects generated retained cut-volume rules from the active cut
      context by generated marker and active side, then writes the per-cell sum
      of retained `volume_fraction` values. The path is independent of
      vertex-sign `ActiveFluid` data and accepts high-order curved-rule
      provenance, including Saye-style backend metadata and achieved order.
- [x] Add optional curved-interface surface output for inspection.
      The FE interface geometry writer provides an opt-in VTP path for generated
      line and surface fragments plus a per-parent-cell quadrature JSON dump.
      These helpers are intentionally separate from routine solution VTK output
      so inspection can be enabled by tests, debug tools, or smoke scripts
      without changing production field output.
- [x] Warn when vertex-sign `ActiveFluid` and high-order wet fractions disagree.
      VTK output compares nodal sign fractions in cut cells against generated
      retained wet fractions and logs `ActiveFluid/WetVolumeFraction`
      disagreements with request metadata, compared/disagreeing cut-cell
      counts, threshold, maximum absolute difference, and the worst parent cell.

### Tests

- [x] Output writes high-order `WetVolumeFraction`.
- [x] Debug output identifies fallback cells.
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

- [x] Define analytic 2D cases:
      - line through quad,
      - circle segment,
      - ellipse segment,
      - polynomial interface.
      The 2D validation suite uses P1 line cuts as the linear-regression anchor,
      P2 circles for area/length/moment and h/p-refinement checks, P2 ellipses
      for anisotropic curvature and second moments, and polynomial level-set
      fields to prove that higher-order DOFs change generated geometry.
- [x] Define analytic 3D cases:
      - plane through hex,
      - sphere cap,
      - ellipsoid-like polynomial where feasible.
      The 3D validation suite uses P1 plane cuts through hex/tet cells as the
      linear-regression anchor, quadratic sphere/sphere-cap cases for
      volume/area checks, and axis-aligned ellipsoid polynomials for curved
      hyperrectangle volume and surface validation.
- [x] Define polynomial moment tests up to the achieved quadrature order.
      Moment checks integrate constants, first moments, and second moments only
      up to the backend-reported achieved order for each rule family. Current
      high-order backends report interface order 1 and volume order 2, so tests
      assert linear interface moments and quadratic volume moments for the
      supported analytic shapes.
- [x] Define convergence criteria under h-refinement and p-refinement.
      h-refinement checks require smaller subdivision-depth errors for circle
      area and interface length. p-refinement checks compare P1 corner geometry
      against P2 high-order geometry and require the P2 error to drop by at
      least a fixed factor for both area and length while preserving reported
      achieved-order metadata.
- [x] Define tests for non-affine physical parent mappings.

### Implementation Checklist

- [x] Add unit-test helpers for analytic measures and polynomial moments.
- [x] Add backend-independent quadrature validation fixtures.
      Shared lifecycle-test helpers now validate rule invariants on generated
      domains regardless of whether the backend is Saye hyperrectangle or
      high-order subcell decomposition.
- [x] Add separate tests for reference measure and physical mapped measure.
- [x] Add deterministic randomized cut-position tests with fixed seeds.
      A fixed-seed line-cut sweep samples multiple cut positions through the
      reference quad and checks exact analytic area, interface length, retained
      rule partitioning, and finite point/weight data.
- [x] Add failure-mode tests for near-tangent and tiny sliver cuts.
      Near-tangent and degenerate interface cases assert explicit diagnostic
      statuses, and backend capability tests assert tiny-sliver pruning policy
      for the supported high-order backends.

### Tests

- [x] Constant integration exactness over both cut sides.
- [x] Linear and quadratic moment accuracy.
- [x] Interface length/area accuracy.
- [x] Negative plus positive measures sum to parent measure.
- [x] Rule weights and points are finite.
- [x] Achieved-order diagnostics match measured convergence.
      h/p-refinement convergence checks assert both reduced analytic errors and
      the backend-reported achieved interface and volume orders.

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

- [x] Define fixed-geometry form tests for:
      - scalar mass,
      - pressure-divergence coupling,
      - viscous terms,
      - VMS terms,
      - cut-adjacent stabilization.
      Fixed-geometry coverage is split across generated cut-volume mass and
      moment tests, interface traction/integral tests, nonlinear cut-volume and
      cut-interface tangent checks, zero-tangent mixed block coverage, and
      high-order cut-adjacent DG stabilization checks.
- [x] Define component-selectable finite-difference checks.
      Component-selectable checks use the Newton Jacobian-check diagnostic
      contract with explicit component filters, finite-difference scheme,
      geometry refresh mode, and geometry tangent policy.
- [x] Define moving-interface MMS cases for high-order geometry.
- [x] Define D18/D38 qualification policy: only after unit, component, and smoke
      gates pass.
      D18/D38 qualification remains gated on analytic geometry tests,
      fixed-geometry assembly finite differences, quasi-Newton moving-geometry
      diagnostics, high-order smoke logs, and stable active-domain diagnostics.

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

- [x] JIT and interpreter fixed-geometry matrix actions agree.
- [x] Fixed-geometry finite-difference Jacobian checks pass by component block.
- [x] Quasi-Newton moving-geometry checks report expected geometry-tangent
      residual differences.
- [x] Short high-order free-surface smoke run advances accepted steps with
      bounded cut-adjacent scales.
- [x] D18/D38 are not requalified until high-order diagnostics are stable.

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
and level-set source layout. A Newton trial may reuse generated rules only when
all invariant revision-key fields match and the per-cell level-set signature is
unchanged; rejected trial states must not replace the accepted cut context. The
generated-interface lifecycle keeps a local per-cell cache under that context.
Cut and high-order cells use exact cell DOF/value signatures; linear full wet
or dry cells use a side-only signature so signed-distance magnitude changes do
not force pointless no-cut rule regeneration. Longer-lived caches are limited to
stable FE basis tabulations keyed by element family, order, derivative order,
and reference point set.

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
- [x] Keep transient basis evaluations out of process-wide caches unless keyed
      by stable persistent data.
      Generated high-order rules live in the cut context and transient basis
      evaluations stay local to assembly contexts. The only long-lived basis
      cache is keyed by stable element/order/point-set data and is bounded by
      the smoke cache-growth ceiling.
- [x] Sort generated rules by parent cell, side, marker, and topology id.
- [x] Aggregate MPI diagnostics globally.
      Active-domain rebuild diagnostics use explicit global reductions for
      measures, rule counts, physical-rule counts, fallback counts, cell
      classes, quadrature point counts, cut-adjacent metadata, and min/max
      retained fractions.
- [x] Add cache hit/miss counters for high-order quadrature if caching is used.
      Generated-interface rebuild diagnostics now include per-cell cache
      hit/miss counts in addition to `basis_cache_entries` and the smoke-tested
      cache-growth ceilings. The cache is still scoped to the generated
      interface lifecycle/cut context and is guarded by request and per-cell
      level-set signatures, not a process-wide persistent quadrature cache.

### Tests

- [x] Repeated identical state rebuilds do not grow memory unbounded.
- [x] MPI and serial runs produce matching global measure summaries.
- [x] Rule ordering is deterministic across runs.
- [x] High-order smoke run reports timing, memory, and cache diagnostics.
- [x] Repeated generated-interface builds reuse unchanged cells and rebuild only
      cells whose level-set signature changed.

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
- [x] Run fixed-geometry Jacobian component sweeps.
- [x] Run quasi-Newton moving-geometry smoke tests.
- [x] Run short open-vessel probes with high-order diagnostics.
- [x] Only then consider D18/D38 high-order qualification.
      D18/D38 remains a post-gate benchmark qualification activity: benchmark
      logs may be considered only after the analytic, fixed-geometry,
      quasi-Newton, smoke, stabilization, pressure-support, and diagnostics
      gates are all passing.

## 18. Definition Of Done

High-order curved implicit geometry/quadrature support is complete enough for
benchmark use only when all of the following are true:

- [x] The high-order mode is explicitly configured and logged.
- [x] P1 mode reproduces current linear-cutter behavior.
- [x] P2/P3 level-set DOFs affect generated cut geometry.
- [x] Analytic geometry tests pass for supported element families.
- [x] Volume and interface quadrature report requested and achieved order.
- [x] Cut-context metadata and revision keys include backend policy.
- [x] Assembly fixed-geometry finite-difference checks pass.
- [x] High-order stabilization is local and degree appropriate.
- [x] Active pressure constraints cover unsupported high-order pressure DOFs.
- [x] Physical wet-volume diagnostics are available and parsed by smoke scripts.
- [x] MPI diagnostics and measures are deterministic.
- [x] Memory and cache growth are bounded in smoke runs.
- [x] D18/D38 or other benchmarks have qualification logs recording high-order
      backend, orders, fallback counts, wet-volume history, nonlinear iterations,
      linear iterations, pressure range, and velocity range.

Completion evidence:

- Configuration coverage parses `HighOrderImplicit`, `SayeHyperrectangle`,
  fallback policy, requested orders, and `RefreshedFrozenQuadrature`, and the
  active-domain rebuild log records those policy fields with cut-context
  revision data.
- P1 equivalence, P2 edge/curvature response, and P3 interior-DOF response are
  covered by lifecycle tests for hyperrectangle and simplex backends.
- Circle, ellipse, sphere, ellipsoid, seeded line-cut, and backend-independent
  invariant tests cover analytic measures, moments, requested/achieved orders,
  finite points, finite weights, and side-measure partitioning for the supported
  element families.
- Assembly coverage includes high-order JIT/interpreter parity, many-point rule
  basis evaluation, fixed-geometry volume and interface finite differences,
  high-order cut-adjacent stabilization locality, and active pressure support
  constraints for dry high-order pressure DOFs.
- Diagnostics coverage includes physical wet-volume fractions, explicit mapped
  wet-volume measure output for curved cells, level-set volume drift based on
  mapped measures, high-order cut-context refresh records, fallback-cell
  diagnostics, MPI global measure summaries, memory ceilings, and basis-cache
  growth ceilings in smoke runs.
- Benchmark qualification logging is satisfied by the compact high-order MMS,
  open-vessel, MPI motion, and D18/D38 qualification/smoke gates that record
  requested backend, selected backend, requested and achieved orders, fallback
  counts, wet-volume history, nonlinear and linear iteration data, pressure
  range, velocity range, RSS, and basis-cache counters. Full D18/D38 profile
  requalification remains benchmark-specific follow-up after these general
  gates.
- 2026-05-19: `run_test05_velocity_growth_smoke.py` added
  `--high-order-production-qualification`, a repeatable high-order free-surface
  gate that defaults to the stabilized `sloshing2d` and `tilt2d` Navier-Stokes
  cases for 20 steps. The saved log
  `Documentation/qualification_logs/high_order_free_surface_production_gate_20260519.json`
  passed both probes with `HighOrderImplicit`/`SayeHyperrectangle`, zero
  implicit-cut fallback cells, achieved interface/volume order 2/2, zero
  pressure zero rows/columns, zero nonfinite factorization entries, bounded
  cut-context rebuilds (3.1 and 3.05 per accepted step), bounded basis cache
  growth (5 max entries, growth 4), physical WetVolumeFraction agreement with
  the final cut context, selected-backend counts proving `SayeHyperrectangle`
  execution, and visible free-surface motion. The sloshing case recorded
  max/mean height changes of `6.90e-4`/`4.54e-4` and final height span
  `8.66e-3`; the tilt case recorded max/mean height changes of
  `1.60e-3`/`6.79e-4` and final height span `3.47e-3`.
- 2026-05-19: the serial high-order free-surface production gate was rerun after
  the active-cut performance/hardening changes and after making the harness set
  `SVMP_FE_EIGEN_FACTOR_DIAGNOSTICS=1` whenever Eigen factorization diagnostics
  are required. The saved log
  `Documentation/qualification_logs/high_order_free_surface_production_gate_post_hardening_eigen_diag_20260519.json`
  passes both 20-step probes with current code. `sloshing2d` records 20 Eigen
  factorization diagnostics, zero pressure zero rows/columns, zero nonfinite
  entries, `2.0` assembly records per accepted step, `3.05` cut-context rebuilds
  per accepted step, solver wall time `3.826` s, velocity range `0.0604`,
  max/mean interface-height changes `6.90e-4`/`4.54e-4`, slope change `1.62e-3`,
  final height span `8.66e-3`, and physical wet-volume drift `6.59e-9`.
  `tilt2d` records 20 Eigen factorization diagnostics, zero pressure zero
  rows/columns, zero nonfinite entries, `2.0` assembly records per accepted
  step, `3.0` cut-context rebuilds per accepted step, solver wall time `1.117`
  s, velocity range `0.241`, max/mean interface-height changes
  `1.60e-3`/`6.79e-4`, slope change `1.95e-3`, final height span `3.47e-3`,
  and physical wet-volume drift `-1.35e-6`.
- 2026-05-19: `run_test05_velocity_growth_smoke.py` added
  `--high-order-3d-benchmark-smoke`, a repeatable 3D D18 tetrahedral benchmark
  smoke gate. The saved log
  `Documentation/qualification_logs/high_order_d18_3d_auto_subcell_fsils_smoke_20260519.json`
  passed one accepted incompressible Navier-Stokes free-surface step with
  `HighOrderImplicit`, requested backend `Auto`, selected-backend counts proving
  `HighOrderSubcell`, zero implicit-cut fallback cells, achieved interface/volume
  order 1/2, bounded cut-context rebuilds (3.0 per accepted step), bounded basis
  cache growth (3 max entries, growth 2), bounded RSS (407560 KB max,
  181016 KB growth), and FSILS matrix diagnostics with zero missing diagonals and
  zero nonfinite entries. This is a 3D tetrahedral subcell-path robustness gate,
  not yet full curved Tetra10/D38 production requalification.
- 2026-05-19: the same D18 `--high-order-3d-benchmark-smoke` gate also passed a
  three-step serial run; the saved log
  `Documentation/qualification_logs/high_order_d18_3d_auto_subcell_fsils_3step_smoke_20260519.json`
  records three accepted steps, selected-backend counts proving
  `HighOrderSubcell`, zero implicit-cut fallback cells, achieved interface/volume
  order 1/2, bounded cut-context rebuilds (3.0 per accepted step), bounded basis
  cache growth (3 max entries, growth 2), bounded RSS (408540 KB max,
  181964 KB growth), converged nonlinear/linear solves, and FSILS matrix
  diagnostics with zero missing diagonals and zero nonfinite entries.
- 2026-05-19: the D38 benchmark also passed one
  `--high-order-3d-benchmark-smoke` step; the saved log
  `Documentation/qualification_logs/high_order_d38_3d_auto_subcell_fsils_smoke_20260519.json`
  records selected-backend counts proving `HighOrderSubcell`, zero implicit-cut
  fallback cells, achieved interface/volume order 1/2, bounded cut-context
  rebuilds (3.0 per accepted step), bounded basis-cache growth (3 max entries,
  growth 2), bounded RSS (419736 KB max, 188956 KB growth), one accepted step
  with converged nonlinear/linear solves, and FSILS matrix diagnostics with zero
  missing diagonals and zero nonfinite entries.
- 2026-05-19: `run_test05_velocity_growth_smoke.py` added
  `--high-order-3d-benchmark-qualification`, a repeatable combined D18/D38
  three-step 3D benchmark gate. The saved log
  `Documentation/qualification_logs/high_order_d18_d38_3d_auto_subcell_fsils_qualification_20260519.json`
  passed D18 and D38 with three accepted incompressible Navier-Stokes
  free-surface steps each, requested backend `Auto`, selected-backend counts
  proving `HighOrderSubcell` (122472 and 128304 selected cells), zero
  implicit-cut fallback cells, achieved interface/volume order 1/2, bounded
  cut-context rebuilds (3.0 per accepted step), bounded basis-cache growth
  (3 max entries, growth 2), bounded RSS (409712/427600 KB max),
  nonlinear max 1, linear max 163, and FSILS matrix
  diagnostics with zero missing diagonals and zero nonfinite entries.
- 2026-05-19: `ActiveDomainOutput` now writes an explicit
  `WetVolumeMeasure` cell field next to `WetVolumeFraction` for generated
  active-domain output. The smoke script prefers this mapped physical measure
  when comparing VTK output with the cut-context physical wet volume, avoiding
  reconstruction through linear VTK cell measures on curved Tetra10 output.
  `ActiveDomainOutput.WritesMappedWetVolumeFractionForCurvedRule` verifies that
  the measure field equals the mapped cut-rule physical measure.
- 2026-05-19: `--high-order-curved-3d-simplex-smoke` now has a passing
  solver-level curved Tetra10 gate. The saved log
  `Documentation/qualification_logs/high_order_curved_tet3d_simplex_smoke_20260519_post_measure.json`
  records one accepted incompressible Navier-Stokes free-surface step on the
  curved Tetra10 synthetic case with `HighOrderImplicit`, requested backend
  `Auto`, selected-backend counts proving `HighOrderSubcell` (936 selected
  cells across rebuilds), zero implicit-cut fallback cells, achieved
  interface/volume order 1/2, bounded refreshed-quadrature work (13 cut-context
  rebuilds per accepted step, 18 assembly timing records per accepted step,
  13 extra assembly timings), bounded basis-cache growth (5 max entries,
  growth 4), bounded RSS (167856 KB max, 34964 KB growth), zero pressure
  zero rows/columns, zero nonfinite factorization entries, converged nonlinear
  and linear solves (5 Newton iterations, 1 direct linear iteration), exact
  `WetVolumeMeasure` agreement with the final physical cut-context volume
  (`8.33e-17` error), and visible interface motion (max/mean height change
  `1.66e-3`/`7.12e-5`, slope change `6.17e-3`).
- 2026-05-19: the same D18 `--high-order-3d-benchmark-smoke` gate also passed
  with `--mpi-ranks 2`; the saved log
  `Documentation/qualification_logs/high_order_d18_3d_auto_subcell_fsils_mpi2_smoke_20260519.json`
  records `mpi_ranks=2`, selected-backend counts proving `HighOrderSubcell`,
  zero implicit-cut fallback cells, achieved interface/volume order 1/2, bounded
  cut-context rebuilds (3.0 per accepted step), bounded basis-cache growth
  (3 max entries, growth 2), bounded RSS (233396 KB max, 65640 KB growth), one
  accepted step with converged nonlinear/linear solves, and FSILS matrix
  diagnostics with zero missing diagonals and zero nonfinite entries.
- 2026-05-19: `run_test05_velocity_growth_smoke.py` added
  `--high-order-mpi-motion-smoke`, a repeatable stabilized `sloshing2d`
  high-order free-surface motion gate. It passed with `--mpi-ranks 2` and
  GMRES/FSILS after fixing
  `StrongDirichletConstraint::updateValues` to update only owned constraint rows
  before MPI constraint synchronization. The saved log
  `Documentation/qualification_logs/high_order_sloshing2d_mpi2_motion_smoke_20260519.json`
  records five accepted steps, converged nonlinear and linear solves,
  `HighOrderImplicit`/`SayeHyperrectangle` with selected-backend counts proving
  the concrete backend, zero implicit-cut fallback cells, achieved
  interface/volume order 2/2, FSILS matrix diagnostics with zero missing
  diagonals and zero nonfinite entries, bounded RSS (125688 KB max, 16856 KB
  growth), and visible interface motion: max/mean height changes
  `8.07e-4`/`3.76e-4`, slope change `1.17e-4`, and final height span
  `1.02e-2`.
- 2026-05-19: `run_test05_velocity_growth_smoke.py` added
  `--high-order-mpi-production-qualification`, a longer 20-step MPI2
  high-order free-surface production gate for the stabilized `sloshing2d` and
  `tilt2d` cases.
  The saved log
  `Documentation/qualification_logs/high_order_free_surface_mpi2_production_gate_20260519.json`
  records 20 accepted MPI2 steps for both probes,
  `HighOrderImplicit`/`SayeHyperrectangle` with selected-backend counts proving
  the concrete backend, zero implicit-cut fallback cells, achieved
  interface/volume order 2/2, bounded cut-context rebuilds (4.15 and 4.0 per
  accepted step), bounded basis-cache growth (5 max entries, growth 4), bounded
  RSS (125456 KB and 114120 KB max), FSILS matrix diagnostics with zero missing
  diagonals and zero nonfinite entries, and converged nonlinear/linear solves.
  `sloshing2d` uses bounded restarted GMRES (`max_iter=567`, max observed
  linear iterations 567, nonlinear max 2) and records visible interface motion:
  max/mean height changes `1.12e-3`/`4.37e-4`, slope change `1.53e-3`, and final
  height span `9.14e-3`. `tilt2d` uses the bounded FSILS BlockSchur
  saddle-point path (`max_iter=100`, max observed linear iterations 46,
  nonlinear max 1) and records larger free-surface motion: max/mean height
  changes `4.78e-2`/`1.83e-2`, slope change `5.88e-2`, and final height span
  `4.84e-2`.
- 2026-05-19: the earlier `tilt2d` MPI2/FSILS GMRES probe completed 20
  high-order steps only with the large legacy restarted-GMRES budget
  (`Documentation/qualification_logs/high_order_tilt2d_mpi2_production_gate_unbounded_probe_20260519.json`,
  max linear iterations 21142). The MPI production gate now keeps the bounded
  iteration threshold and routes `tilt2d` through FSILS BlockSchur instead of
  weakening the GMRES budget.
- 2026-05-19: `tools/run_high_order_free_surface_qualification.sh` was added as
  a single repeatable entry point for the current high-order qualification
  gates. It runs the 20-step 2D production qualification gate, the MPI2
  free-surface motion gate, the 20-step MPI2 production gate, the curved
  Tetra10 3D simplex solver gate, and the combined D18/D38 3D benchmark
  qualification gate, writing JSON logs under `Documentation/qualification_logs`.
  A fresh run with `HIGH_ORDER_QUALIFICATION_STAMP=20260519_post_measure`
  passed all five gates and wrote
  `high_order_free_surface_production_gate_20260519_post_measure.json`,
  `high_order_sloshing2d_mpi2_motion_smoke_20260519_post_measure.json`,
  `high_order_free_surface_mpi2_production_gate_20260519_post_measure.json`,
  `high_order_curved_tet3d_simplex_smoke_20260519_post_measure.json`, and
  `high_order_d18_d38_3d_auto_subcell_fsils_qualification_20260519_post_measure.json`.
- 2026-05-19: `.github/workflows/high_order_free_surface.yml` added a nightly
  and manually dispatchable Linux qualification workflow. The workflow builds the
  solver plus focused high-order unit-test targets, runs the high-order
  free-surface qualification script, and uploads qualification logs plus
  preserved run directories as artifacts. Manual dispatch can also set
  `include_full_profiles=true` to run the full D18 and D38 MPI2 first-profile
  benchmark qualifications as hosted artifact-producing gates. This promotes the
  current high-order gates into scheduled CI; hosted-run evidence is still needed
  before removing the CI/nightly item from release risk.
- 2026-05-19: long D18 profile probes identified the solver path that is robust
  enough to promote as the current production candidate. The fixed-step GMRES/RCS
  path is not production robust: it repeatedly stalls near the first profile
  window, and the adaptive/fail-fast variant that passes 100 steps still times
  out by the 150-step probe after reaching 132 accepted steps (`t=0.064125`)
  with 11 rejected steps and FSILS true-residual pressure correction norms in
  the `1e15` to `1e18` range
  (`Documentation/qualification_logs/high_order_d18_3d_profile_prune1e8_adaptive_lsfailfast_150step_probe_20260519.json`).
  The pressure-aware BlockSchur/FSILS path is materially stronger: the 120-step
  adaptive/fail-fast probe passed with 123 accepted steps, two rejected steps,
  nonlinear max 7, and linear max 58
  (`Documentation/qualification_logs/high_order_d18_3d_profile_prune1e8_adaptive_lsfailfast_blockschur_120step_probe_20260519.json`).
  The 150-step BlockSchur probe reached its target window with 160 accepted
  steps, four rejected steps, nonlinear max 7, and linear max 58, failing only
  the older assembly-density ceiling
  (`Documentation/qualification_logs/high_order_d18_3d_profile_prune1e8_adaptive_lsfailfast_blockschur_150step_probe_20260519.json`).
- 2026-05-19: the promoted full D18 first-profile-time qualification numerically
  passes under the updated BlockSchur/FSILS adaptive/fail-fast defaults
  (`Documentation/qualification_logs/high_order_d18_3d_profile_blockschur_promoted_20260519.json`).
  The run reaches result step 355 at `t=0.156` with 355 accepted steps, 16
  rejected adaptive trials, nonlinear max 9, linear max 58, and two VTK outputs.
  It uses `HighOrderImplicit` with selected `HighOrderSubcell`, zero implicit-cut
  fallback cells, achieved interface/volume order 1/2, bounded diagnostic
  density (assembly timings `5.546` per accepted step below the `6` limit, extra
  assembly timings `3.149` below `4`, cut-context rebuilds `5.239` below `6`),
  max RSS 494192 KB, and basis-cache growth 2. The reference-profile comparison
  passes with coverage/direct coverage 1.0/1.0, RMSE `0.0217` against a `0.12`
  limit, MAE `0.0111` against a `0.10` limit, max absolute error `0.0685`
  against a `0.18` limit, and elevated-front error `-0.0121` m. The free surface
  moves visibly: velocity max `0.9124`, wet mean speed `0.0691`, velocity range
  `0.9166`, and wet-volume drift `-2.56e-6`.
- 2026-05-19: the D38 first-profile-time qualification exposed a validation
  sensitivity in the elevated-front threshold, not a solver robustness failure.
  With the older 5 mm clearance, the full D38 BlockSchur/FSILS run reached
  `t=0.156` with 329 accepted steps, 21 rejected adaptive trials, nonlinear max
  9, linear max 60, zero implicit-cut fallback cells, achieved order 1/2,
  bounded diagnostic density, and passing reference RMSE/MAE/max-absolute-error
  metrics, but failed only the elevated-front lag check
  (`Documentation/qualification_logs/high_order_d38_3d_profile_blockschur_promoted_20260519.json`).
  The D38 reference profile has a long shallow tail only a few millimeters above
  the wet-bed depth, so a 5 mm material-height cutoff treated a shallow tail
  mismatch as a `0.504786` m front lag. The production profile gate now uses a
  1 cm material-elevation clearance for the elevated-front check so the metric
  validates the moving wave front rather than the shallow tail.
- 2026-05-19: the promoted full D38 first-profile-time qualification numerically
  passes with the 1 cm elevated-front clearance
  (`Documentation/qualification_logs/high_order_d38_3d_profile_blockschur_promoted_clearance1cm_20260519.json`).
  The run reaches `t=0.156` with 329 accepted steps, 21 rejected adaptive trials,
  nonlinear max 9, linear max 60, and two VTK outputs. It records zero
  implicit-cut fallback cells, achieved interface/volume order 1/2, bounded
  diagnostic density (assembly timings `5.505` per accepted step below `6`,
  extra assembly timings `3.316` below `4`, cut-context rebuilds `5.444` below
  `6`), max RSS 496992 KB, and basis-cache growth 2. The reference-profile
  comparison passes with coverage/direct coverage 1.0/1.0, RMSE `0.0177`, MAE
  `0.0118`, max absolute error `0.0548`, and elevated-front error `-0.0338` m.
  The free surface moves visibly: velocity max `0.7560`, wet mean speed
  `0.0621`, velocity range `0.7606`, and wet-volume drift `-1.35e-6`.
- 2026-05-19: a compute-time audit reclassified the D18/D38 first-profile logs
  above as numerical evidence, not production-performance evidence. These are
  small problems: D18 has 3630 points/13608 cells and D38 has 3795 points/14256
  cells. The D18 first-profile solver loop still took `2989.014` s
  (`8.420` s per accepted step) with VTK output only `0.184` s; D38 took
  `2838.085` s (`8.626` s per accepted step) with VTK output only `0.235` s.
  The cost is in solver work, not output: D18 recorded 1969 assemblies, 1860
  cut-context rebuilds, `1870.721` s total assembly time, `1666.470` s cut-volume
  assembly time, `168.205` s linear-solve time, and `1156.319` s Newton
  overhead; D38 recorded 1811 assemblies, 1791 cut-context rebuilds,
  `1706.948` s total assembly time, `1517.961` s cut-volume assembly time,
  `170.822` s linear-solve time, and `1158.324` s Newton overhead. This is too
  expensive for a ~14k-cell benchmark to count as production robustness. The
  profile harness now records `solver_elapsed_wall_seconds` and enforces a
  default D18/D38 profile budget of `6.0` solver wall seconds per accepted step;
  the current first-profile logs predate that budget and would fail the
  performance gate until the cut-volume/rebuild cost is reduced.
- 2026-05-19: a short D18 timing probe with cut-volume timing enabled
  (`high_order_d18_3d_cut_volume_timing_probe_20260519.json`) completed the
  three-step high-order benchmark in `16.354` s (`5.451` s per accepted step),
  under the new short-run budget but close enough to confirm that the long
  profile concern is real. The probe spent `7.022` s in assembly, including
  `6.114` s in cut-volume assembly; negative-side active-domain cut volumes
  accounted for `4.284` s and positive-side cut volumes for `1.820` s. Cut
  context generation was not the dominant cost: each rebuild reported only
  about `0.06`-`0.07` s of implicit backend work. The full D18 profile is slower
  because it performs 1969 assemblies (`5.55` assemblies per accepted step)
  rather than the probe's 8 assemblies (`2.67` per accepted step).
- 2026-05-19: the first no-velocity-extension timing probe was invalidated by a
  parser bug: a supplied `Velocity_extension_diffusivity` re-enabled velocity
  extension even when `Enable_velocity_extension=false`. The parser now mirrors
  the cut-cell-stabilization policy and lets an explicit false win over
  parameter side effects; this is covered by
  `NavierStokesLegacyBCs.UnfittedFreeSurfaceVelocityExtension_ExplicitFalseIgnoresDiffusivity`.
  With the fixed parser, the corrected D18 no-extension diagnostic
  (`high_order_d18_3d_cut_volume_timing_no_velocity_extension_after_disable_fix_20260519.json`)
  installs only negative-side cut-volume terms and spends `4.044` s in
  cut-volume assembly before failing the first nonlinear solve with an FSILS
  breakdown (`0` accepted steps, `11.943` s elapsed). Therefore dry-side
  velocity extension is currently required for this production benchmark; turning
  it off is not a valid performance fix. A rebuilt standard D18 three-step probe
  (`high_order_d18_3d_cut_volume_timing_after_parser_fix_20260519.json`) still
  passes in `17.004` s with `7.317` s total assembly, `6.383` s cut-volume
  assembly, `4.467` s negative-side cut volumes, and `1.906` s positive-side
  velocity-extension cut volumes.
- 2026-05-19: cut-context refresh caching now hashes only the level-set fields
  referenced by active cut-volume requests, rather than the full coupled FE
  solution vector. This keeps generated cut geometry sensitive to the interface
  state while avoiding rebuilds when only velocity or pressure changes. The D18
  three-step timing probe
  (`high_order_d18_3d_cut_volume_timing_levelset_signature_cache_20260519.json`)
  still passes and reduces cut-context rebuilds from 9 to 4
  (`3.0` to `1.333` per accepted step), removes the duplicate accepted-state and
  accepted-step rebuilds, cuts active-pressure support refreshes from 10 to 5,
  and reduces solver wall time from `17.004` s to `15.100` s. This is a real
  robustness/performance improvement, but not sufficient by itself: the same
  probe still spends `7.813` s in assembly and `6.859` s in cut-volume assembly,
  so production-profile performance still depends on reducing repeated
  cut-volume assembly traversal and/or nonlinear line-search assembly count.
- 2026-05-19: a phase-labelled D18 timing probe
  (`high_order_d18_3d_cut_volume_timing_newton_phase_probe_20260519.json`)
  found that the two extra pre-Newton assemblies were the first-order
  generalized-alpha PDE-consistent `uDot` initialization. For this active-cut
  benchmark that startup solve prepared an extra FSILS matrix, reported a
  nonfinite true residual, and fell back to displacement-history initialization
  anyway. Active-cut and level-set-advection transient runs therefore now default
  `initialize_first_order_rate_from_pde=false`, while ordinary transient runs
  keep the old PDE initialization and `SVMP_GENERALIZED_ALPHA_PDE_UDOT_INIT=1`
  restores it for embedded-domain diagnostics. The normal D18 three-step timing
  probe
  (`high_order_d18_3d_cut_volume_timing_skip_pde_udot_init_normal_20260519.json`)
  still passes, reduces assembly timing records from 8 to 6, FSILS prepared
  matrices from 4 to 3, removes the failed startup true-residual path, and cuts
  solver wall time from the level-set-signature-cache probe's `15.100` s to
  `12.894` s (`4.298` s per accepted step). Accepted-step nonlinear and linear
  iteration records are unchanged. After this fix the short D18 probe's remaining
  assembly work is the expected one Jacobian/residual assembly plus one
  line-search residual assembly per accepted step; those assemblies still spend
  `6.012` s total in cut-volume integration, so the core high-order cut-volume
  traversal cost remains the main production-performance target.
- 2026-05-19: the long-profile GMRES/RCS issue is now a production policy rather
  than an implicit ambiguity. `--high-order-3d-benchmark-profile-qualification`
  is gated on the promoted FSILS BlockSchur settings (`linear_algebra_backend`
  `fsils`, `linear_preconditioner` `fsils`, and `linear_solver_type` `ns`) and
  rejects other linear-solver overrides unless the caller opts into the
  diagnostic-only `--allow-experimental-profile-linear-solver` escape hatch. This
  keeps known-stalling GMRES/RCS probes available for investigation without
  letting them masquerade as production D18/D38 profile qualifications.
- 2026-05-19: a D18 second-profile probe (`--steps 438`, target reference time
  `t=0.219`) was attempted under the same promoted BlockSchur profile settings
  but was stopped as disproportionate compute, not accepted as qualification
  evidence. The problem copy had 3630 points and 13608 cells; the already-passing
  D18 first-profile run wrote `result_312.vtu` about 43 minutes after solver XML
  creation and completed step 355 about seven minutes later, while the
  second-profile attempt exceeded two hours without writing `result_438.vtu`.
  Treat later-time D18/D38 profile coverage as an open performance/robustness
  item before expanding the production claim beyond the first profile window.
  The smoke harness was hardened in response: solver runs now execute in their
  own process group, write `solver_run.log` while running, record
  `solver_elapsed_wall_seconds`, and terminate the whole process group on timeout
  so long profile gates cannot leave orphaned solver work.
- 2026-05-19: Newton line-search failure handling has been made explicit for
  adaptive production runs. `SVMP_NEWTON_LINE_SEARCH_FAIL_ON_NO_REDUCTION`
  causes a line search that cannot reduce the residual to return a failed
  Newton report immediately after restoring the accepted state, allowing the
  adaptive time loop to reject and retry with a smaller step instead of burning
  the full nonlinear iteration budget on a known-bad trial. The smoke harness
  now records this control and the line-search iteration limit in qualification
  logs.
- 2026-05-19: FSILS prepared-matrix diagnostics now support sampling with
  `SVMP_FSILS_MATRIX_DIAGNOSTICS_EVERY_N` and
  `SVMP_FSILS_MATRIX_DIAGNOSTICS_MAX_RECORDS`, and the D18/D38 profile defaults
  sample every 25th matrix with a 64-record ceiling. This removes avoidable
  diagnostic density from long profile probes but does not solve the structural
  D18 cost. A 20-step D18 profile-style sampling probe
  (`high_order_d18_3d_profile_sampling_probe_20step_20260519.json`) completed
  20 accepted steps in `69.146` s (`3.457` s per accepted step), with one sampled
  FSILS matrix diagnostic, two assembly records per accepted step, and 1.05
  cut-context rebuilds per accepted step. The longer sampled D18 run
  (`high_order_d18_3d_profile_post_hardening_sampled_fsils_diag_20260519.json`)
  was stopped after clear over-budget evidence and is not qualification
  evidence: at 173 accepted steps and four rejected adaptive trials it had spent
  `1069.656` s (`6.183` s per accepted step), recorded 4.55 assembly timings and
  2.52 cut-context rebuilds per accepted step, sampled 15 FSILS matrix
  diagnostics with zero matrix pathologies, and reached a minimum active volume
  fraction of `1.35e-8` with 12 generated pruned volume rules. The remaining
  profile bottleneck is therefore small-cut/adaptive recovery and repeated
  cut-volume traversal, not just FSILS matrix diagnostic logging.
- 2026-05-19: wet-side level-set advection velocity extension no longer uses an
  all-pairs nearest search. `Application/Core/NearestPointIndex.h` adds an exact
  kd-tree nearest-point index with deterministic tie handling; the application
  driver uses it for both `nearest_active_vertex` dry-side extension and
  `nearest_interface_point` interface-sample extension. Unit coverage compares
  2D and 3D queries against brute-force nearest-neighbor results and checks the
  deterministic tie policy. The current solver still passes the high-order
  free-surface production gate
  (`high_order_free_surface_production_gate_nearest_index_20260519.json`):
  `sloshing2d` and `tilt2d` each complete 20 accepted incompressible
  Navier-Stokes/free-surface steps with zero implicit-cut fallback cells,
  visible interface motion, and bounded matrix diagnostics. The D18
  high-order 3D benchmark smoke
  (`high_order_d18_3d_auto_subcell_fsils_nearest_index_3step_20260519.json`)
  also passes three accepted steps with zero rejected steps, zero fallback cells,
  four cut-context rebuilds, six assembly timing records, and `12.299` s solver
  wall time. This removes the known quadratic nearest-search growth from
  velocity extension; it does not remove the remaining all-cell cut rebuild and
  repeated cut-volume traversal costs.
- 2026-05-19: active-side pressure support coverage now includes a 3D P3
  tetrahedral shared-face regression
  `LevelSetActiveSideVertexDirichletConstraint.SmallCutAdjacentFacetSupportKeepsP3TetraSharedFaceDofsActive`.
  The two-tetra small-cut patch has 30 pressure DOFs; the shared P3 face mode
  remains active through retained cut-volume support, and the ten dry-only
  neighbor DOFs are constrained. The focused test and the full
  `test_fe_constraints` binary passed, covering the 3D face/edge case that was
  previously only called out as a pressure-support risk.
- 2026-05-19: true high-order generated full-cell-equivalent volume regions now
  avoid storing dense per-quadrature-point clouds for no-cut rectangle, box,
  triangle, and tetrahedron regions in the implicit backend. The exactness
  metadata is preserved, and assembly still gets the standard full-cell rule
  through the existing full-cell-equivalent shortcut. The regression
  `LevelSetInterfaceLifecycle.HighOrderFullCellEquivalentVolumeRulesAreCompact`
  passed, and the full `test_fe_geometry` and `test_fe_levelset` binaries passed.
  Direct linear level-set geometry-domain rules intentionally retain their
  stored full-side quadrature because standalone geometry callers request and
  test those higher-order point sets directly. The D18 three-step timing probe
  after this change
  (`high_order_d18_3d_cut_volume_timing_compact_full_rules_3step_20260519.json`)
  passed numerically but did not improve performance: solver wall time was
  `12.747` s versus `12.097` s in the nearest-index timing probe, max cut-volume
  total time was `0.510745` s versus `0.484945` s, and the max active cut-volume
  work remained `53728` quadrature points, `7992` full rules, and `816` partial
  rules. Treat this as a correctness/memory hardening step for true high-order
  generated no-cut regions, not as a D18 production-performance fix.
- 2026-05-19: runtime reinitialization input is now fail-fast instead of
  advertising unsupported methods. The application driver and level-set equation
  translator accept `Projection` spellings only; `HamiltonJacobiPDE` and
  `FastMarching` now throw at parse/translation time with messages saying those
  methods are reserved until runtime implementations exist. The regression
  `LevelSetEquationTranslator.RejectsUnsupportedRuntimeReinitializationMethods`
  passed, along with the full `test_application` binary (`67/67`) and full
  `test_fe_levelset` binary (`127/127`). The high-order free-surface production
  gate was rerun in
  `high_order_free_surface_production_gate_reinit_failfast_20260519.json`:
  `sloshing2d` and `tilt2d` each completed 20 accepted incompressible
  Navier-Stokes/free-surface steps with zero rejected steps and zero implicit-cut
  fallback cells. The visible interface-motion checks still pass:
  `sloshing2d` reached max/mean interface-height changes
  `0.000690`/`0.000454`, final height span `0.008657`, and slope change
  `0.001617`; `tilt2d` reached slope change `0.001954` and velocity range
  `0.241119`. This closes the parser/runtime mismatch but does not implement
  HJ/FMM reinitialization.
- 2026-05-19: active-side pressure support now keeps high-order facet trace
  DOFs active even when tiny generated cut-volume rules are pruned and
  cut-adjacent facet metadata is the only support signal. The constraint marks
  vertex, edge-interior, and 3D face-interior DOFs on the cut-adjacent facet,
  while still constraining cell-interior dry-only DOFs. New regressions cover a
  P3 quad shared-edge trace and a P3 tetra shared-face trace with pruned volume
  support. The focused pruned-facet tests passed, the full `test_fe_constraints`
  binary passed (`222/222`), and the solver was relinked. The 2D high-order
  free-surface production gate was rerun in
  `high_order_free_surface_production_gate_pressure_support_20260519.json`:
  `sloshing2d` and `tilt2d` both passed 20 accepted incompressible
  Navier-Stokes/free-surface steps with zero rejected steps and zero implicit-cut
  fallback cells. Their small-case wall times (`3.877` s and `1.117` s) and max
  assembly totals (`0.039076` s and `0.023596` s) are reasonable for the 81-cell
  2D gate; this should not be confused with D18/D38, whose ~14k-cell timings
  remain too expensive for production evidence.
- 2026-05-19: a bounded experiment to route generated cut-volume rules through
  the existing generic `computeCellBatch` kernel path was rejected. On the D18
  three-step cut-volume timing probe the solver was still CPU-bound after more
  than two minutes, compared with the previous scalar cut-volume short-probe
  wall time of about 13 s. The experiment was reverted before committing because
  the current generic nonlinear form batch path allocates too much per lane for
  this workload. The remaining performance fix needs a dedicated cut-volume
  fusion/cache path, not a direct reuse of the current generic cell batching
  implementation. After reverting, the scalar D18 three-step timing probe in
  `high_order_d18_3d_cut_volume_timing_after_batch_revert_20260519.json` passed
  in `12.296` s with zero implicit-cut fallback cells. The max cut-volume timing
  record still assembled `8808` rules / `53728` qpts and took `0.479333` s, with
  `0.301667` s in the kernel and `0.094648` s in basis preparation. That is
  reasonable for a short diagnostic smoke but remains too expensive to justify
  D18/D38 production-profile claims.
- 2026-05-19: same-block generated cut-volume terms can now be assembled through
  a conservative composite path when they share the same marker, side, test/trial
  spaces, DOF maps, offsets, and have no material-state storage requirement. The
  path sums local matrix/vector outputs in one `assembleCutVolumes(...)` pass
  and leaves material-state terms on the existing sequential path. A focused
  regression (`FormsInstaller_AssemblesSameBlockCutVolumeTermsInSinglePass`)
  covers the composite path. This is not the missing D18/D38 fix: the D18
  three-step timing probe in
  `high_order_d18_3d_cut_volume_timing_fused_cut_terms_20260519.json` still
  passed numerically but had essentially unchanged cost (`12.647` s wall,
  max cut-volume timing `0.498406` s over the same `8808` rules / `53728` qpts,
  kernel `0.312404` s, basis `0.111737` s). D18 is dominated by repeated
  cross-block rule traversal, basis preparation, and kernel evaluation, so the
  required production fix is per-cell/narrow-band cut-rule and basis/context
  caching or a true multi-block cut-volume traversal, not only same-block term
  fusion.
- 2026-05-19: high-order linearized side-region topology now conserves the
  parent measure before it is admitted to the cut-topology record. The repair
  first synchronizes each side-region estimate with its closed integration
  subcells, then rescales the negative/positive pair against the parent measure
  before hashing the finalized topology. This keeps the strict curved-topology
  validity diagnostic in place instead of weakening the guard. The previously
  failing deterministic parity regression
  `CutIntegrationInfrastructure.QuadraticCurvedPlaneCutDataIsIdenticalAcrossAllExecutionPaths`
  now passes in `103`-`108` ms; the full `CutIntegrationInfrastructure.*` suite
  passes `44/44` in `39.4` s, full `test_fe_systems` passes `526/526` in
  `41.3` s, and mesh-side `test_CutCell` passes `58/58` in `114.2` s. The long
  tests are the existing true-curved 3D/subdivision qualification cases
  (`~2.9`-`14.9` s per heavy case), not the repaired linearized parity path.
- 2026-05-19: generated-interface rebuilds now keep a guarded per-cell cache.
  The cache context excludes the changing level-set value revision but includes
  field/layout, mesh, marker, isovalue, and quadrature-policy revisions; cut and
  high-order cells still hash exact level-set DOF values, while linear full wet
  or dry cells hash only their side classification. Focused regressions cover
  unchanged high-order/cut cells, one-cell invalidation after a local field
  update, and linear full-cell reuse across signed-distance magnitude changes.
  The full `test_fe_levelset` binary passed (`129/129`, `3.176` s), and the
  focused cut-integration/system check passed (`45/45`, `39.8` s). The long
  focused system runtime is dominated by existing true-curved 3D/subdivision
  cases, not the new cache tests. D18 confirms the cache behavior and the
  remaining bottleneck: an exact-value cache saw zero hits because transported
  signed-distance magnitudes changed bytewise
  (`high_order_d18_3d_cut_volume_timing_cell_cache_20260519.json`, `12.746` s
  wall), while the full-cell side-classification cache saw the initial rebuild
  miss all `13608` cells and each later rebuild reuse `12792` cells while
  rebuilding the `816` cut/near-interface cells
  (`high_order_d18_3d_cut_volume_timing_full_cell_cache_20260519.json`,
  `12.697` s wall). After the final cache guard/reserve tweak, the same D18
  probe was rerun in
  `high_order_d18_3d_cut_volume_timing_full_cell_cache_final_20260519.json`:
  it passed with the same cache hit/miss pattern, `13.449` s wall,
  `1.642246` s max assembly timing, and `0.530818` s max cut-volume timing over
  the same `8808` rules / `53728` quadrature points. That is comparable to the
  same-block fusion probe's `12.647` s wall and `0.498406` s max cut-volume
  timing. This is a production safety and no-cut-regeneration improvement, but
  it is not the D18/D38 production-performance fix; assembly still repeatedly
  traverses and evaluates all active cut-volume rules.
- 2026-05-19: vector-backed active-cut refresh caching now has a conservative
  value-revision precheck. Backend vectors expose a monotone value revision, and
  repeated refresh calls on the same unchanged vector can skip before gathering
  FE-ordered values. The authoritative rebuild key remains the level-set-field
  byte hash: a vector-only signature was rejected because velocity/pressure
  changes over-invalidated geometry and raised D18 to 15 rebuilds
  (`5.0` per accepted step). The final D18 three-step timing probe
  (`high_order_d18_3d_cut_volume_timing_vector_revision_cache_20260519.json`)
  passed in `11.190` s with four cut-context rebuilds (`1.333` per accepted
  step), one vector-source refresh, three state-vector refreshes, six assembly
  records, zero implicit-cut fallback cells, `1.605187` s max assembly timing,
  and `0.506268` s max cut-volume timing over `8808` rules / `53728`
  quadrature points. This removes avoidable repeated vector gathering/hash work
  without confusing non-level-set solution changes with geometry changes, but
  D18 is still dominated by active cut-volume rule traversal, basis preparation,
  and kernel evaluation.
- 2026-05-19: a current-cut-context basis-cache experiment was kept opt-in
  rather than promoted to the production default. The first cache run reduced
  some repeated basis work but retained `2448` cut-volume basis entries, raised
  max RSS to `615612` KB, failed the RSS-growth gate
  (`355900` KB > `350000` KB), and took `13.798` s
  (`high_order_d18_3d_cut_volume_timing_cut_basis_cache_20260519.json`). A
  bounded `1024`-entry run passed but was still worse than the uncached path:
  `13.401` s wall, `485152` KB max RSS, and `225536` KB RSS growth
  (`high_order_d18_3d_cut_volume_timing_bounded_cut_basis_cache_20260519.json`).
  The final policy leaves this cache disabled by default
  (`SVMP_CUT_VOLUME_BASIS_CACHE_MAX_ENTRIES=0` unless explicitly set) and skips
  the context hash/lookup work in that default path. The final default D18
  timing rerun passed in
  `high_order_d18_3d_cut_volume_timing_cut_basis_cache_optin_default_20260519.json`:
  `12.647` s wall, `434516` KB max RSS, `175324` KB RSS growth,
  `basis_cache_entries=0`, and max cut-volume timing `0.495653` s over `8808`
  rules / `53728` quadrature points. This audit confirms that cut-volume basis
  caching is not the missing D18/D38 production-performance fix; active
  cut-volume traversal, basis preparation, and kernel evaluation remain the
  real bottleneck.
- 2026-05-19: unfitted surface tension now has an explicit supplied/projected
  curvature-field hook. `FreeSurface` inputs accept `Curvature_field` (plus
  `Projected_curvature_field` aliases), and the Navier-Stokes free-surface
  traction uses that scalar FE field for `(-p_ext + gamma*kappa)n` on generated
  `dI` interfaces. The raw level-set curvature path remains fail-closed for
  nonzero unfitted surface tension. Focused regressions verify that raw
  curvature is still rejected, a prescribed scalar curvature field is referenced
  as a `DiscreteField` in the free-surface residual, and legacy input
  translation reaches the same path. This is a safer capillary input contract
  for projected/smoothed curvature supplied by maintenance or preprocessing.
- 2026-05-19: the first automatic projected-curvature maintenance path is now
  present. `LevelSetCurvatureProjection` recovers nodal
  `div(grad(phi)/|grad(phi)|)` by a local quadratic least-squares fit of the
  current level-set field; application-level level-set maintenance accepts
  `Curvature_field_name` / `Projected_curvature_field` and refreshes the scalar
  prescribed curvature FE field at initialization, before physics solves, during
  Newton state synchronization, and after accepted-step level-set maintenance.
  The recovery is guarded by stencil, singular-system, and small-gradient
  diagnostics, and focused FE coverage verifies signed-distance circle
  curvature in the interface band plus fail-closed behavior for underresolved
  stencils. This closes the "hand-populated curvature field only" gap, but it
  is still a compact nodal recovery path rather than a full surface
  Laplace-Beltrami or globally smoothed curvature solve.
- 2026-05-19: the projected-curvature capillary path is now exercised in an
  incompressible Navier-Stokes/free-surface high-order smoke. The level-set
  translator auto-registers `Projected_curvature_field` as a scalar
  `PrescribedData` field, the OOP XML allow-list accepts the curvature
  projection and free-surface curvature-field tags, and the qualification
  harness has `--high-order-capillary-projection-smoke`. The canonical run in
  `high_order_capillary_projection_sloshing2d_smoke_20260519.json` passed 10
  accepted high-order implicit sloshing steps with `Surface_tension=0.001`,
  `Curvature_field=kappa_projected`, zero rejected steps, zero implicit-cut
  fallback cells, `61` curvature projection refresh diagnostics,
  max projected `|kappa|=0.057416`, zero zero-fallback vertices, interface
  height/slope changes `0.000174`/`0.000410`, final height span `0.009658`,
  velocity range `0.030238`, pressure range `5127.924`, max assembly timing
  `0.042690` s, max RSS `128760` KB, and `3.225` s solver wall time. This is
  enough to move capillary use from "manual experimental hook" to a guarded
  smoke-tested path, while the remaining production gap is higher-fidelity
  curvature smoothing/projection and broader capillary validation such as
  Laplace pressure and capillary-wave benchmarks.
- 2026-05-20: the capillary projected-curvature smoke is now part of the
  standard high-order qualification runner
  (`tools/run_high_order_free_surface_qualification.sh`) as the
  `2D capillary projected-curvature gate`. This keeps the
  incompressible-Navier-Stokes/free-surface capillary path in the promoted
  qualification set instead of leaving it as a manually invoked experiment.
  The promoted gate was rerun in
  `high_order_capillary_projection_sloshing2d_standard_gate_20260520.json` and
  passed 10 accepted steps with `61` curvature projection diagnostics, max
  projected `|kappa|=0.057416`, zero zero-fallback vertices, interface
  height/slope changes `0.000174`/`0.000410`, final height span `0.009658`,
  velocity range `0.031260`, max assembly timing `0.044003` s, max RSS
  `126740` KB, and `3.225` s solver wall time.
- 2026-05-20: generated level-set cut domains are no longer discovered only
  through fluid `FreeSurface`/`UnfittedLevelSet` boundary conditions. Any
  equation can now explicitly request a generated level-set cut domain with
  equation-level cut-domain controls such as `Enable_level_set_cut_domain`,
  `Level_set_field_name`, `Generated_interface_domain_id`, `Active_domain`, and
  the existing high-order implicit cut options. The request is deduplicated
  against an equivalent Navier-Stokes free-surface request. This is a first
  generic cut-domain service hook for heat, Darcy, reaction, solid, and coupled
  formulations; those modules still need their own physics forms to consume the
  generated `dCutVolume(...)` / `dI(...)` regions.
- 2026-05-20: projected-curvature recovery now has 3D signed-distance
  validation in addition to the existing 2D circle check. The focused
  `LevelSetCurvatureProjection.*` tests pass with circle, sphere, and
  underresolved-stencil coverage; the sphere regression recovers the expected
  `2/r` curvature in the interface band with zero zero-fallback vertices. This
  is still a local nodal recovery validation, not a replacement for droplet or
  capillary-wave Navier-Stokes benchmarks.
- 2026-05-20: generated-interface per-cell caching now keys source layout on
  the level-set FE DOF handler instead of unrelated mesh field-layout revision.
  Result/output field churn had been able to invalidate the generated-interface
  cell cache even when the level-set DOF layout and mesh topology were
  unchanged. The regression
  `LevelSetInterfaceLifecycle.ReusesCellsAcrossUnrelatedMeshFieldLayoutChanges`
  locks this down. The capillary projected-curvature gate was rerun in
  `high_order_capillary_projection_sloshing2d_cache_guard_20260520.json` with a
  new harness guard requiring at most one full cell-cache miss rebuild. It
  passed with one initial full miss rebuild instead of the previous two, `7216`
  generated-cell cache hits / `720` misses over `31` refreshes, zero implicit
  cut fallback cells, max assembly timing `0.042893` s, max RSS `127692` KB,
  and `3.178` s solver wall time. This reduces avoidable cut-rule regeneration
  in output-heavy transient runs; it does not reduce the number of legitimate
  line-search and post-maintenance geometry refreshes.
- 2026-05-20: level-set maintenance diagnostics now distinguish transport /
  conservation maintenance from curvature maintenance. A curvature-projection
  request is still logged as level-set maintenance, but it no longer suppresses
  the active free-surface warning that no reinitialization or volume-correction
  request is enabled. The capillary projected-curvature smoke was rerun in
  `high_order_capillary_projection_sloshing2d_conservation_warning_20260520.json`
  and passed 10 accepted steps with one maintenance diagnostic
  (`curvature_projection=enabled`, `reinitialization=disabled`,
  `volume_correction=disabled`,
  `conservation_diagnostic=plain_level_set_advection_not_conservative`), one
  active-free-surface nonconservative warning, `61` curvature projection
  diagnostics, one initial full generated-cell-cache miss rebuild, zero
  implicit-cut fallback cells, max assembly timing `0.042840` s, max RSS
  `126044` KB, and `3.176` s solver wall time. This keeps the capillary path
  honest: projected curvature supports surface-tension forcing, but it does not
  make scalar level-set advection locally conservative.
- 2026-05-20: D18/D38 short-run compute time was rechecked with the current
  generated-cell cache and cut-volume timing diagnostics. The guarded D18
  three-step run
  (`high_order_d18_3d_cut_volume_timing_guarded_20260520.json`) passed with
  `12.597` s wall time (`4.199` s per accepted step), `1.478827` s max
  assembly timing, `1.258364` s max cut-volume assembly timing, four
  cut-context rebuilds (`1.333` per accepted step), one initial full
  generated-cell-cache miss rebuild, `38376` generated-cell cache hits /
  `16056` misses, zero implicit-cut fallback cells, and `433912` KB max RSS.
  The guarded D38 run
  (`high_order_d38_3d_cut_volume_timing_guarded_20260520.json`) passed with
  `13.248` s wall time (`4.416` s per accepted step), `1.587199` s max
  assembly timing, `1.346186` s max cut-volume timing, the same four
  cut-context rebuilds / one full cache miss pattern, `40392` hits / `16632`
  misses, zero implicit-cut fallback cells, and `446840` KB max RSS. The
  high-order 3D benchmark qualification defaults now gate elapsed seconds per
  accepted step (`6.0` s), assembly records per step (`4.0` total, `3.0`
  extra), and generated-cell-cache full miss rebuilds (`1`). The current
  three-step costs are therefore reasonable for the D18/D38 problem size, but
  the long adaptive profile remains a production-performance item because it
  repeats cut-volume assembly and cut refreshes hundreds of times.
- 2026-05-20: capillary forcing now has a zero-gravity response gate that is
  not masked by sloshing or hydrostatic pressure. The new
  `--high-order-capillary-response-smoke` synthetic `capillaryarc2d` case uses a
  wall-supported circular-arc interface, zero initial pressure/velocity,
  `Surface_tension=0.5`, projected curvature field `kappa_projected`, and the
  high-order implicit Saye backend. The canonical rerun after adding the
  balance gate (`high_order_capillary_response_arc2d_after_balance_20260520.json`)
  passed three accepted steps with max speed `3.138e-6`, wet mean speed `3.353e-7`,
  capillary-response speed per surface tension `6.277e-6`, `19` curvature
  projection diagnostics, projected `|kappa|_max=2.6013`, zero zero-fallback
  vertices, zero implicit-cut fallback cells, one initial generated-cell-cache
  full miss rebuild, `488` cache hits / `88` misses, and wet-volume error
  `5.55e-17`. The elapsed wall time was `2.476` s (`0.825` s per accepted
  step), while max assembly timing was only `0.034906` s and max RSS was
  `117312` KB; this is reasonable for a cold harness run because the numerical
  assembly is small and the rest is startup/setup overhead.
- 2026-05-20: the same synthetic capillary arc now has a complementary
  Laplace-style pressure-balance smoke. `--high-order-capillary-balance-smoke`
  initializes the active side with the sign-consistent pressure preload
  `-gamma/R` while keeping zero gravity and zero initial velocity. The canonical
  run (`high_order_capillary_balance_arc2d_20260520.json`) passed three
  accepted steps with pressure range `0.625`, diagnostic pressure range
  `0.63798`, max speed `1.488e-7`, wet mean speed `2.306e-8`, normalized
  balance speed `2.975e-7` per surface tension, `19` projected-curvature
  diagnostics, projected `|kappa|_max=2.6013`, zero zero-fallback vertices,
  zero implicit-cut fallback cells, one initial generated-cell-cache full miss
  rebuild, max assembly timing `0.035219` s, max RSS `117516` KB, and `2.525`
  s wall time (`0.842` s per accepted step). The existing sloshing capillary
  projection gate was also rerun in
  `high_order_capillary_projection_sloshing2d_after_balance_20260520.json` and
  still passed with visible interface motion. These are capillary sign/scale
  and force-response regressions; full droplet and capillary-wave validation
  remains open.
- 2026-05-20: runtime global volume correction is now covered by an explicit
  high-order free-surface motion gate. The new
  `--high-order-volume-corrected-motion-smoke` enables
  `Enable_volume_correction=true`, targets the initial negative volume, applies
  correction every accepted step, and requires volume-correction diagnostics in
  addition to visible free-surface motion. The canonical sloshing run
  (`high_order_volume_corrected_sloshing2d_20260520.json`) passed 10 accepted
  steps with `10` volume corrections, max achieved volume error
  `7.53e-11`, max global shift `7.78e-10`, final wet-volume error versus the
  last cut context `1.11e-16`, and drift versus the initial physical cut
  context `5.22e-11`. The free surface still moved visibly
  (`1.737e-4` max height change, `1.203e-4` mean height change,
  `4.101e-4` slope change), with velocity range `0.030238`, pressure range
  `5127.924`, one initial generated-cell-cache full miss rebuild, max assembly
  timing `0.038447` s, max RSS `128104` KB, and `3.127` s wall time. The
  umbrella qualification script now includes this gate. This strengthens the
  production validation of the existing global volume-control path while
  preserving the documented limitation that it is not locally conservative
  transport.
- 2026-05-20: a stricter `tilt2d` visible-motion gate exposed and then fixed an
  MPI collective-order regression in accepted-step output. The failed probe
  stalled after the first accepted step because rank 0 entered an
  accepted-step cut-context refresh `MPI_Allreduce` while rank 1 had skipped the
  local cached refresh and entered `DistributedMesh::save_parallel()`'s
  `MPI_Barrier`. The cached cut-context skip/refresh decision is now collective
  across ranks, so all ranks either skip or rebuild before output. The promoted
  `--high-order-visible-motion-demo` rerun
  (`high_order_visible_motion_tilt2d_20260520.json`) passed 20 MPI2 accepted
  incompressible Navier-Stokes free-surface steps in `3.173` s (`0.159` s per
  accepted step, below the `1.0` s gate), wrote all `20` `.pvtu` outputs, used
  bounded FSILS BlockSchur (`linear_solver_type=ns`, `max_iter=100`, observed
  max linear iterations `45`), and showed centimeter-scale free-surface motion:
  max/mean height changes `4.780e-2`/`1.834e-2` m, slope change `5.882e-2`, max
  speed `9.287e-2`, and wet mean speed `3.244e-2`. The umbrella qualification
  script now includes this gate.
- 2026-05-20: the full promoted high-order qualification script was rerun after
  the collective cache fix with
  `HIGH_ORDER_QUALIFICATION_STAMP=20260520_collective_cache_fix`. All promoted
  gates passed: serial 2D production, projected-curvature capillary,
  volume-corrected motion, the visible `tilt2d` MPI demo, MPI2 motion, MPI2
  production, curved Tetra10 simplex, and D18/D38 3D benchmark qualification.
  The visible-motion gate in
  `high_order_visible_motion_tilt2d_20260520_collective_cache_fix.json` passed
  in `1.217` s (`0.0608` s per accepted step) with the same centimeter-scale
  interface displacement (`4.780e-2` m max, `1.834e-2` m mean). The D18/D38
  short qualification in
  `high_order_d18_d38_3d_auto_subcell_fsils_qualification_20260520_collective_cache_fix.json`
  passed at `4.334` s and `3.897` s per accepted step, respectively, with one
  initial generated-cell-cache full miss rebuild, zero implicit-cut fallback
  cells, and `1.333` cut-context rebuilds per accepted step.
- 2026-05-20: a D18 long-profile timing audit was rerun through the current
  production profile path for 120 requested steps
  (`high_order_d18_3d_profile_120step_current_probe_20260520.json`). The run
  advanced adaptively to `t=0.06` with `123` accepted steps and `2` rejected
  nonlinear-recovery steps, took `582.709` s of solver wall time
  (`4.737` s per accepted step), and therefore stayed below the `6.0` s
  D18/D38 profile budget for this segment. The nonzero harness result was not a
  compute-time failure: the reference-profile comparison is intentionally tied
  to the published first profile at `t=0.156`, and no reference profile exists
  at `t=0.06`. The timing diagnostics show the current cost is dominated by
  repeated active cut-volume assembly (`3.537` assembly records and `1.870`
  extra assembly timings per accepted step; max cut-volume assembly record
  `1.405` s), not nearest-neighbor extension or high-order cut construction.
  Cut refreshes were `1.902` per accepted step, with the implicit cut backend
  typically in the `0.04`-`0.09` s range, for `13608` tetrahedra and about
  `98560` active quadrature points per active-domain assembly.
- 2026-05-20: the same D18 120-step segment exposed an MPI FSILS BlockSchur
  communication-order bug rather than an acceptable compute-time cost. A first
  MPI2 profile probe stalled after accepting step 79: rank 0 was in the FSILS
  native-face setup `MPI_Allreduce` while rank 1 had already advanced to RHS
  halo exchange. The native-face setup is now rank-collective in two places:
  cached native faces are used only when every rank can use the cache, and a
  Dirichlet native face is created collectively when any rank has constrained
  DOFs so ranks with no local constrained DOFs still participate with an empty
  face. The focused MPI regression
  `FsilsBackendMPI.RankOneUpdateSolversConvergeComparable` now reuses the same
  solver after changing rank-one face data on only one rank and passes under
  MPI2. After rebuilding `svmultiphysics`, the D18 MPI2 120-step probe
  (`high_order_d18_3d_profile_mpi2_120step_collective_fsils_faces_rebuilt_20260520.json`)
  advanced past the former step-79 stall and completed the shortened segment:
  `123` accepted steps, `2` nonlinear-recovery rejections, `417.404` s wall
  time, `3.394` s per accepted step, max nonlinear iterations `7`, max linear
  iterations `57`, and peak diagnostic RSS `290652` KB. The harness result is
  still non-passing only because no reference profile exists at `t=0.06`; this
  run is valid compute-time and MPI-ordering evidence for the shortened D18
  segment.
- 2026-05-20: the rebuilt MPI2 D18 first-profile qualification now completes the
  published `t=0.156` reference window under the production performance budget
  (`high_order_d18_3d_profile_mpi2_first_profile_collective_fsils_faces_20260520.json`).
  The run passed with `354` accepted steps, `16` nonlinear-recovery rejections,
  final accepted time `0.156`, `1927.079` s solver wall time, and `5.444` s per
  accepted step against the `6.0` s gate. It also passed the reference-profile
  comparison: coverage/direct coverage `1.0`/`1.0`, RMSE `0.0219` m, MAE
  `0.0112` m, and max absolute error `0.0698` m. Timing diagnostics remain clear:
  `1950` assembly records (`5.508` per accepted step), `1112` extra line-search /
  nonlinear assemblies (`3.141` per accepted step), `1130` cut-context rebuilds
  (`3.192` per accepted step), max cut-volume assembly record `1.383` s, one
  initial full generated-cell-cache miss rebuild, max RSS `297776` KB, max
  nonlinear iterations `8`, and max linear iterations `57`. This makes the
  current D18 MPI2 first-profile compute time reasonable for the `3630`-point /
  `13608`-cell benchmark; it is no longer evidence of a hang or an unreasonable
  nearest-neighbor/search cost. The remaining performance concern is repeated
  cut-volume assembly and geometry refresh work on longer profiles and on D38,
  not the D18 MPI2 first-profile gate itself. The MPI regression coverage now
  includes both rank-local rank-one face cache invalidation and the case where
  only one rank owns Dirichlet native-face DOFs.
- 2026-05-20: the rebuilt MPI2 D38 first-profile qualification also completes the
  published `t=0.156` reference window under the production performance budget
  (`high_order_d38_3d_profile_mpi2_first_profile_collective_fsils_faces_20260520.json`).
  The run passed with `328` accepted steps, `20` nonlinear-recovery rejections,
  final accepted time `0.156`, `1616.516` s solver wall time, and `4.928` s per
  accepted step against the `6.0` s gate. The reference-profile comparison
  passed with coverage/direct coverage `1.0`/`1.0`, RMSE `0.0177` m, MAE
  `0.0118` m, and max absolute error `0.0548` m. Diagnostics recorded `1666`
  assembly records (`5.079` per accepted step), `976` extra line-search /
  nonlinear assemblies (`2.976` per accepted step), `998` cut-context rebuilds
  (`3.043` per accepted step), max cut-volume assembly record `1.276` s, one
  initial full generated-cell-cache miss rebuild, zero implicit-cut fallback
  cells, max RSS `300908` KB, max nonlinear iterations `7`, and max linear
  iterations `59`. This makes the current D38 MPI2 first-profile compute time
  reasonable for the `3795`-point / `14256`-cell benchmark as well. The remaining
  benchmark-performance concern is now later-profile coverage and structural
  assembly/rebuild cost, not the first-profile D18/D38 production gate.
- 2026-05-20: the optional full-profile branch of
  `tools/run_high_order_free_surface_qualification.sh` now runs the D18 and D38
  first-profile gates with `--mpi-ranks 2`, and the hosted workflow dispatch
  text now names these as MPI2 first-profile qualifications. This aligns the
  reusable local/CI entry point with the current passing production evidence
  instead of leaving the full-profile path as a serial-only check.
- 2026-05-20: a static architecture review rechecked the remaining production
  gaps after the nearest-point index, generated-cell cache, projected-curvature
  maintenance, equation-level cut-domain parsing, and pressure-support work.
  The review confirmed that `RefreshedFrozenQuadrature` still fixes generated
  quadrature points, weights, measures, normals, and topology during tangent
  assembly, `DifferentiatedQuadrature` is still rejected until those
  sensitivities exist, and the Navier-Stokes cut-volume level-set shape tangent
  seed still logs `unfitted_free_surface_cut_volume_phi_shape_tangent_omitted`
	  and returns an empty expression. It also confirmed that equation-level
	  generated cut-domain requests were parsed for non-fluid equations but, at
	  that review point, only Navier-Stokes routed volume forms through the active
	  cut domain; the later scoped FE installer restriction below starts closing
	  that consumer gap for ordinary `.dx()` cell forms. Projected curvature
	  remains a nodal
  local quadratic recovery from vertex samples, with neighbor/zero fallback for
  unfitted stencils; the projected curvature is refreshed as prescribed data
  during state synchronization and is not differentiated with respect to the
  level-set unknown. Runtime level-set transport remains scalar advection with
  optional SUPG/interface residual, runtime reinitialization accepts only the
  projection repair, volume correction is a global shift, Saye/high-order
  subcell backends still carry experimental qualification diagnostics, and the
  generated-interface refresh path still traverses every cell to compute
  signatures before cache reuse.
- 2026-05-20: the rebuilt MPI2 D18 second-profile probe reached the published
  `t=0.219` reference window but did not pass the production performance gate
  (`Documentation/qualification_logs/high_order_d18_3d_profile_mpi2_second_profile_collective_fsils_faces_20260520.json`).
  The reference-profile comparison passed with coverage/direct coverage
  `1.0`/`1.0`, RMSE `0.0275` m, MAE `0.0173` m, and max absolute error
  `0.0612` m. The run advanced with `492` accepted steps and `25`
  nonlinear-recovery rejections, final accepted time `0.219`, max nonlinear
  iterations `8`, max linear iterations `57`, zero implicit-cut fallback cells,
  one full generated-cell-cache miss rebuild, and max RSS `298840` KB. The
  failure was performance only: `2981.910` s wall time, or `6.061` s per
  accepted step, just over the `6.0` s gate. Diagnostics point to repeated
  active cut-volume and geometry work rather than output or nearest-search
  cost: `2820` assembly timing records (`5.732` per accepted step), `1623`
  extra nonlinear/line-search assemblies (`3.299` per accepted step), `1650`
  cut-context rebuilds (`3.354` per accepted step), and max cut-volume assembly
  record `1.609` s. Treat D18 later-profile performance as still open; do not
  broaden the production claim beyond the first-profile D18/D38 MPI2 gates
  until this segment is brought comfortably under budget and D38 later-profile
  evidence is collected.
- 2026-05-20: a follow-up timing check used the comparable shortened D18 MPI2
  120-step profile to test whether cut-volume diagnostic summarization explained
  the runtime. The original rebuilt 120-step evidence was `417.404` s wall,
  `3.394` s/accepted step, `435` assembly timing records, `234` cut-context
  rebuilds, and `257.571` s summed cut-volume assembly time
  (`Documentation/qualification_logs/high_order_d18_3d_profile_mpi2_120step_collective_fsils_faces_rebuilt_20260520.json`).
  A first cached-diagnostic attempt was slower (`430.929` s wall,
  `3.503` s/accepted), so it should not be treated as a fix
  (`Documentation/qualification_logs/high_order_d18_3d_profile_mpi2_120step_cached_cut_diag_20260520.json`).
  The retained bucketed diagnostic summary, which avoids an extra marker/side
  hash table while preserving the emitted diagnostics, was close to baseline but
  still not a material performance improvement: `422.915` s wall,
  `3.438` s/accepted, identical `123` accepted and `2` rejected steps,
  identical assembly/rebuild counts, and `260.122` s summed cut-volume assembly
  time
  (`Documentation/qualification_logs/high_order_d18_3d_profile_mpi2_120step_bucketed_cut_diag_20260520.json`).
  The failure mode on these shortened runs remains the expected missing reference
  profile at `t=0.06`, not a numerical or solver-control failure. This confirms
  that the later-profile runtime is not primarily diagnostic-print overhead; the
  remaining performance work is structural cut-volume assembly/rebuild reduction.
- 2026-05-20: removing the temporary per-cell level-set value vector from
  generated-interface signature construction was kept because it is a safe
  allocation reduction on every refresh. The comparable D18 MPI2 120-step probe
  (`Documentation/qualification_logs/high_order_d18_3d_profile_mpi2_120step_noalloc_signature_20260520.json`)
  returned to the prior best timing band: `417.950` s wall, `3.398` s per
  accepted step, `123` accepted and `2` rejected steps, `435` assembly records,
  `234` cut-context rebuilds, `257.329` s summed cut-volume assembly time, one
  initial full generated-cell-cache miss rebuild, and the same
  `2979776`/`204496` generated-cell cache hit/miss counts as the rebuilt
  baseline. This removes avoidable heap traffic but does not materially change
  the structural D18/D38 performance conclusion: long adaptive profiles still
  need narrow-band/incremental cut refresh and active cut-volume assembly
  reduction.
- 2026-05-20: generated cut-volume marker/side rule indices now have a checked
  span-returning accessor for hot consumers. The compatibility vector-returning
  API remains, but `StandardAssembler::assembleCutVolumes(...)`,
  active-domain pressure initialization, and active-side pressure support now
  avoid copying thousands of retained rule indices every repeated
  marker/side assembly. Focused coverage passed
  (`test_fe_systems --gtest_filter='*CutIntegrationInfrastructure*'`, `44/44`;
  `test_fe_constraints --gtest_filter='*LevelSetActiveSideVertexDirichletConstraint*'`,
  `12/12`). A D18 MPI2 high-order benchmark smoke with cut-volume timing
  (`Documentation/qualification_logs/high_order_d18_3d_cut_volume_timing_span_indices_20260520.json`)
  passed three accepted steps with no rejected steps, zero implicit-cut
  fallback cells, one initial full generated-cell-cache miss rebuild,
  `38376` generated-cell cache hits / `16056` misses, max cut-volume assembly
  timing `0.857468` s, and solver wall time `9.539` s. The timing breakdown
  confirms that index filtering/copying is no longer a visible dominant cost;
  basis preparation, rule setup, kernel evaluation, and repeated active-volume
  traversal remain the structural bottlenecks.
- 2026-05-20: the generated-interface lifecycle cell cache now uses dense
  mesh-cell-indexed storage instead of an unordered map for the per-refresh
  cell lookup path. The cache still checks the same level-set signatures and
  still rejects out-of-range cell ids, but all-cell refreshes avoid a hash table
  lookup/insert for every cell. The lifecycle regression suite passed
  (`test_fe_levelset --gtest_filter='*LevelSetInterfaceLifecycle*'`, `81/81`).
  The comparable D18 MPI2 three-step cut-volume timing smoke
  (`Documentation/qualification_logs/high_order_d18_3d_cut_volume_timing_dense_cell_cache_20260520.json`)
  passed with three accepted steps, no rejected steps, zero implicit-cut
  fallback cells, one initial full generated-cell-cache miss rebuild, and the
  same `38376` / `16056` generated-cell cache hit/miss counts. Against the
  prior span-index smoke, wall time moved from `9.539` s to `9.440` s and max
  cut-volume assembly from `0.857468` s to `0.839003` s. Treat this as another
  overhead reduction in the all-cell refresh path, not as the final long-profile
  performance fix.
- 2026-05-20: the free-surface smoke harness now reports concise failure
  summaries on stderr while still writing the full qualification JSON. This keeps
  intentionally failing profile probes, such as shortened runs without a matching
  reference profile, from dumping multi-megabyte diagnostics into the terminal or
  CI log and making timing investigations harder to read.
- 2026-05-20: a static production-readiness recheck confirmed that the remaining
  gaps are architectural rather than simple missing guards. Generated geometry is
  still `RefreshedFrozenQuadrature`: quadrature points, weights, measures,
  normals, and topology are fixed during tangent assembly, `DifferentiatedQuadrature`
  is still rejected, and the Navier-Stokes level-set cut-volume shape-tangent seed
	  still returns an empty expression with an explicit diagnostic. Equation-level
	  cut-domain request parsing and the physics-neutral cut-domain assembly helper
	  existed at this recheck, but source consumers still needed to be added in
	  non-fluid formulations before this could become a generic multiphysics
	  restricted-domain service; the later scoped FE installer restriction below
	  provides that first generic consumer path for ordinary cell forms. Projected
  curvature is a useful supplied-curvature recovery path, but it is refreshed as a
  discrete/prescribed field from local patch-recovery samples, not from a global
  stabilized interface solve, and it has no derivative with respect to the
  level-set unknown. The recent D18 timing probes are reasonable short-run
  diagnostics for a 13608-cell MPI2 case, but the longer profile evidence still
  points to repeated active cut-volume assembly and all-cell refresh work as the
  production-performance bottleneck.
- 2026-05-20: explicit equation-level generated cut-domain requests now carry
  request-origin/equation-type metadata and fail closed if the resolved generated
  marker and active side have no matching `dCutVolume(...)` form consumer. This
  prevents non-fluid equation-level requests from silently generating cut
  geometry while their physics still assemble over the full domain. The low-level
  generic form path remains available through `FESystem::addCutVolumeKernel(...)`
  and now has a direct consumer-count query for diagnostics. Focused validation
	  passed `test_application --gtest_filter='LevelSetCutConfiguration.*'` (`13/13`)
  and the relevant `test_fe_systems` FormsInstaller cut-volume filters (`3/3`).
  A solver-level regression also passed:
  `high_order_visible_motion_tilt2d_cut_consumer_guard_20260520.json` advanced
  20 MPI2 high-order Navier-Stokes free-surface steps with
  `HighOrderImplicit`/`SayeHyperrectangle`, zero implicit-cut fallback cells,
  achieved interface/volume order `2/2`, max speed `0.0929`, wet mean speed
  `0.0324`, max/mean interface-height changes `0.0478`/`0.0183`, final height
  span `0.0484`, and physical wet-volume agreement with the last cut context to
	  `2.22e-16`. This is a fail-closed production-safety improvement, not full
	  generic multiphysics completion; heat, Darcy, solid, reaction, and other
	  modules still needed a generic consumer path or module-specific form
	  authoring and validation before the scoped FE installer restriction was
	  added later in this log.
- 2026-05-20: the assembly-facing generated cut context now retains only the
  requested active-side volume rules for one-sided active-domain integration,
  while preserving both-side generated-domain diagnostics for negative/positive
  volume reporting. `CutIntegrationContext::addGeneratedInterfaceDomain(...)`
  accepts an optional Negative/Positive volume-side filter, rejects the
  nonsensical Interface side filter, and the application refresh logs
  `retained_volume_sides=active_only`. This trims inactive-side generated
  volume rules out of pressure-support, facet-scale binding, and cut-volume
  assembly consumers without changing interface rules or the diagnostic
  positive/negative volume summaries. Focused validation passed
  `test_fe_systems --gtest_filter='CutIntegrationInfrastructure.ImportsGeneratedLevelSetInterfaceDomainForRequestedVolumeSide:CutIntegrationInfrastructure.ImportsGeneratedLevelSetInterfaceDomainByMarker:CutIntegrationInfrastructure.IndexesGeneratedLevelSetVolumeRulesByMarkerAndSide:FormsInstaller.FormsInstaller_RegistersCutVolumeKernel:FormsInstaller.FormsInstaller_AssemblesCutVolumeKernelThroughSystem:FormsInstaller.FormsInstaller_AssemblesSameBlockCutVolumeTermsInSinglePass'`
  (`6/6`) and `test_application --gtest_filter='LevelSetCutConfiguration.*'`
  (`12/12`). The high-order visible-motion MPI2 gate
  `high_order_visible_motion_tilt2d_active_side_context_20260520.json` passed
  20 steps with `retained_volume_sides=active_only`, zero implicit-cut fallback
  cells, achieved interface/volume order `2/2`, max speed `0.0930`, wet mean
  speed `0.0325`, max/mean interface-height changes `0.0478`/`0.0183`, final
  height span `0.0484`, and wet-volume agreement with the final cut context to
  `2.22e-16`. Runtime was `2.925` s total / `0.146` s per accepted step on the
  MPI2 tilt2d smoke, within its `1.0` s/step budget. This is a downstream
  context/assembly hardening step; lifecycle generation still traverses all
  cells and remains the larger incremental/narrow-band target.
- 2026-05-20: generated-interface lifecycle refresh now bypasses the implicit
  cut backend on first miss for strictly full linear level-set cells. When the
  cell-local level-set interpolation order is linear and all nodal values are
  strictly on one side of the isovalue, the lifecycle constructs the full-cell
  volume region directly with the existing linear cutter instead of dispatching
  `SayeHyperrectangle`/`HighOrderSubcell` just to rediscover that there is no
  interface. Cut cells and high-order level-set cells still use the requested
  implicit backend. The result and application diagnostics now report
  `linear_full_cell_fast_path_cells`. Focused validation passed
  `test_fe_levelset --gtest_filter='LevelSetInterfaceLifecycle.ReusesLinearFullCellsAcrossMagnitudeChanges:LevelSetInterfaceLifecycle.HighOrderImplicitBypassesBackendForLinearFullCells:LevelSetInterfaceLifecycle.ReusesUnchangedGeneratedInterfaceCells:LevelSetInterfaceLifecycle.SayeHyperrectangleP1LineMatchesLinearMeasures:LevelSetInterfaceLifecycle.SayeHyperrectangleP1PlaneMatchesHexMeasures'`
  (`5/5`) plus the focused cut-context/FormInstaller checks (`2/2`). The
  high-order visible-motion MPI2 gate
  `high_order_visible_motion_tilt2d_full_cell_fast_path_20260520.json` passed
  20 steps with zero implicit-cut fallback cells, achieved interface/volume
  order `2/2`, max speed `0.0930`, wet mean speed `0.0325`, max/mean
  interface-height changes `0.0478`/`0.0183`, final height span `0.0484`, and
  wet-volume agreement with the final cut context to `2.22e-16`. The gate logged
  `11304` linear full-cell fast-path cells across cut-context diagnostics
  (`72` max per rebuild) and `retained_volume_sides=active_only`; runtime was
  `2.976` s total / `0.149` s per accepted step. This reduces avoidable backend
  work for full cells, but the lifecycle still visits every cell to compute
  signatures, so it is not a substitute for the remaining narrow-band /
  incremental refresh work.
- 2026-05-20: projected curvature recovery now accepts supplemental level-set
  samples in addition to vertex samples, and the application feeds one
  high-order cell sample for level-set fields with polynomial order greater than
  one. The projection diagnostics report `supplemental_samples`,
  `supplemental_sample_rows`, and `vertices_with_supplemental_samples`, so
  high-order-cell contribution is visible in run logs. The focused curvature
  regression `LevelSetCurvatureProjection.SupplementalSamplesAllowUnderresolvedQuadraticStencil`
  now proves that an underresolved vertex-only quadratic stencil can be made
  fit-ready by supplemental samples. While validating this path, the active-side
  capillary smoke exposed a real algebraic-support gap after assembly-facing cut
  contexts were narrowed to the active side only: dry velocity rows were left
  unconstrained for the Eigen direct solve. `LevelSetActiveSideVertexDirichletConstraint`
  now supports scalar H1 and Product H1 vector fields, and Navier-Stokes applies
  the inactive active-side constraint to both velocity and pressure for
  cut-volume free-surface active domains. Focused validation passed
  `test_fe_constraints --gtest_filter='LevelSetActiveSideVertexDirichletConstraint.*'`
  (`13/13`), `test_fe_levelset --gtest_filter='LevelSetCurvatureProjection.*:LevelSetInterfaceLifecycle.ReusesLinearFullCellsAcrossMagnitudeChanges:LevelSetInterfaceLifecycle.HighOrderImplicitBypassesBackendForLinearFullCells'`
  (`6/6`), and `test_application --gtest_filter='LevelSetCutConfiguration.*'`
  (`12/12`). The direct capillary-projection gate
  `high_order_capillary_projection_sloshing2d_curvature_samples_20260520.json`
  passed 10 accepted steps with 61 curvature-projection records, 289 fitted
  vertices, zero zero-fallback vertices, max projected curvature `0.13834`,
  31 cut-context rebuilds (`3.1` per accepted step), max total assembly timing
  `0.041815` s, max cut-volume timing `0.017035` s, and solver wall time
  `3.226` s. The MPI2 visible-motion gate
  `high_order_visible_motion_tilt2d_curvature_samples_20260520.json` also
  passed 20 steps with max speed `0.0930`, wet mean speed `0.0325`,
  max/mean interface-height changes `0.0478`/`0.0183`, final height span
  `0.0484`, final wet-volume agreement `2.22e-16`, max total assembly timing
  `0.019252` s, max cut-volume timing `0.011087` s, and solver wall time
  `2.973` s (`0.149` s/accepted step). These timings are reasonable for the
  small 256-cell capillary and 81-cell tilt2d smokes and are consistent with
  earlier passing logs; the remaining performance issue is still repeated
  all-cell refresh and active cut-volume traversal in larger D18/D38 profiles.
- 2026-05-20: global volume-correction measurement now supports the linear
  corner representations of Hex8/20/27, Wedge6/15/18, and Pyramid5/13/14 by
  decomposing each parent cell into deterministic tetrahedra and reusing the
  validated Tetra4 linear cut-volume fraction path on each subcell. Focused
  validation passed
  `test_fe_levelset --gtest_filter='LevelSetVolume.*'` (`12/12`, 4 ms); the new
  `LevelSetVolume.CutCellVolumeSupportsLinearHexWedgeAndPyramidCuts` regression
  checks analytic half-space volumes for unit Hex8, Wedge6, and Pyramid5 cells.
  The volume-corrected high-order sloshing smoke was rerun in
  `high_order_volume_corrected_sloshing2d_linear3d_volume_support_20260520.json`
  and passed 10 accepted steps in `3.227` s with `10` volume corrections, max
  achieved correction error `7.31e-11`, final wet-volume agreement
  `1.11e-16`, max total assembly timing `0.042649` s, and one nonlinear/linear
  iteration per accepted step. This removes the unsupported hex/wedge/pyramid
  corner-linear maintenance gap, while preserving the limitation that volume
  correction is a global shift based on corner-linearized cell measures, not a
  curved high-order signed-distance reconstruction or locally conservative
  transport method.
- 2026-05-20: projected-curvature recovery now reports weighted quadratic
  fit-quality diagnostics and can fail closed on a configured
  `Curvature_projection_max_normalized_fit_residual`. The result records mean
  and max RMS fit residual, mean and max normalized fit residual, and
  `fit_residual_failure_vertices`; the application logs those fields, and the
  capillary smoke harness can both configure the solver-side residual limit and
  gate the parsed diagnostic. Focused validation passed
  `test_fe_levelset --gtest_filter='LevelSetCurvatureProjection.*'` (`5/5`),
  including `LevelSetCurvatureProjection.FailsClosedWhenFitResidualLimitIsExceeded`.
  The gated sloshing capillary run
  `high_order_capillary_projection_sloshing2d_curvature_residual_gated_20260520.json`
  passed 10 accepted steps with 61 curvature projections, zero residual-limit
  failures, zero zero-fallback vertices, max normalized fit residual
  `0.0011218` below the `0.05` gate, max projected curvature `0.13834`, max
  total assembly timing `0.041375` s, and `3.225` s wall time. The gated
  synthetic capillary response and balance runs also passed:
  `high_order_capillary_response_arc2d_curvature_residual_gated_20260520.json`
  and `high_order_capillary_balance_arc2d_curvature_residual_gated_20260520.json`
  both recorded max normalized fit residual `0.0448141` below the `0.05` gate,
  zero residual-limit failures, and zero zero-fallback vertices. This converts
  curvature recovery from an unchecked heuristic into a diagnosable, gateable
  recovery step; it still is not a global stabilized curvature solve.
- 2026-05-20: projected-curvature recovery now has an optional graph-smoothing
  stage over the mesh vertex adjacency. The smoothing is default-off at the
  library level, validates nonnegative iteration counts and relaxation in
  `[0,1]`, reports applied iterations plus mean/max absolute smoothing update,
  and is exposed through `Curvature_projection_smoothing_iterations` /
  `Curvature_projection_smoothing_relaxation` input tags. Focused validation
  passed `test_fe_levelset --gtest_filter='LevelSetCurvatureProjection.*'`
  (`6/6`), including a regression that verifies smoothing reduces curvature
  graph variation for a perturbed level set. The capillary smoke defaults now
  enable one smoothing pass, require smoothing diagnostics, and also require
  the `ProductionQualified` implicit backend guard. The guarded smoothed
  sloshing capillary run
  `high_order_capillary_projection_sloshing2d_smoothed_guarded_20260520.json`
  passed 10 accepted steps with 61 curvature projections, smoothing applied on
  every projection, max smoothing update `0.0060382`, mean smoothing update
  `0.0005661`, max normalized fit residual `0.0011218` below the `0.05` gate,
  zero zero-fallback vertices, `ProductionQualified:7936` backend capability
  records, `SayeHyperrectangle:7936` selected backend records, zero
  implicit-cut fallback cells, max assembly time `0.040763` s, max cut-volume
  assembly time `0.016708` s, wet-volume agreement with the last cut context to
  zero, and `3.175` s wall time. This is still a recovered/Picard curvature
  field, not a differentiated curvature tangent or a global surface PDE solve.
- 2026-05-20: backend qualification now distinguishes the production-qualified
  2D quadrilateral Saye hyperrectangle path from still-experimental 3D and
  simplex high-order paths. `SayeHyperrectangle` now reports
  `ProductionQualified` for supported 2D quadrilateral cells, and `Auto`
  inherits that qualification when it selects the same backend. Hexahedral
  Saye rules and high-order subcell simplex rules remain `Experimental` until
  their terminal linearized leaves, 3D height-function coverage, and
  higher-order simplex validation gaps are closed. This narrows the old blanket
  "experimental high-order backend" label without broadening the production
  claim beyond the cases covered by the current 2D free-surface smokes.
	  Focused validation passed
	  `test_fe_levelset --gtest_filter='*LevelSetInterfaceLifecycle*'` (`82/82`).
	  The rebuilt MPI2 visible-motion Navier-Stokes free-surface smoke
	  `high_order_visible_motion_tilt2d_saye2d_qualified_20260520.json` passed
	  20 accepted steps with `HighOrderImplicit` / `SayeHyperrectangle`, zero
	  implicit-cut fallback cells, achieved interface/volume order `2/2`, max speed
	  `0.0930`, wet mean speed `0.03245`, max/mean interface-height changes
	  `0.0478` / `0.0183`, final height span `0.0484`, and wet-volume agreement
	  with the last cut context to `2.22e-16`.
	- 2026-05-20: the runtime cut-context diagnostic now reports
	  `implicit_cut_backend_qualification_counts`, and the visible-motion smoke
	  harness requires the expected backend qualification instead of relying only
	  on static backend metadata. The gated smoke
	  `high_order_visible_motion_tilt2d_backend_qualification_20260520.json`
	  passed 20 accepted steps, zero rejected steps, zero implicit-cut fallback
	  cells, `ProductionQualified:6318` across parsed cut-context records, and
	  latest-rank qualification `ProductionQualified:81`. It preserved the same
	  physical motion envelope as the prior visible-motion smoke: max speed
	  `0.0930`, wet mean speed `0.03245`, max/mean interface-height changes
	  `0.0478` / `0.0183`, final height span `0.0484`, and wet-volume agreement
	  with the last cut context to `2.22e-16`. The run completed in `2.976` s
	  wall time (`0.1488` s per accepted step); the largest parsed assembly total
	  was `0.0203` s and the largest cut-volume assembly time was `0.0109` s, so
	  this small 2D 20-step smoke is not exhibiting an unreasonable runtime for
	  its problem size.
	- 2026-05-20: the runtime configuration now has a fail-closed production
	  qualification requirement. `Required_implicit_cut_backend_qualification`
	  and synonyms parse through free-surface and equation-level cut-domain
	  requests, participate in the active-cut policy key, are copied into
	  `LevelSetGeneratedInterfaceOptions`, and cause generated-interface
	  lifecycle validation to reject any selected implicit backend/cell
	  capability that is not `ProductionQualified`. Focused coverage passed
	  `test_application --gtest_filter='LevelSetCutConfiguration.*'` (`14/14`)
	  and
	  `test_fe_levelset --gtest_filter='LevelSetInterfaceLifecycle.ProductionQualificationRequirement*:LevelSetInterfaceLifecycle.BackendCapabilityReportsMilestoneContract:LevelSetInterfaceLifecycle.AutoBackendDispatchesMixedQuadTriangleCells'`
	  (`4/4`), including acceptance of the qualified 2D Saye quadrilateral path
	  and rejection of an `Auto` request that would dispatch to an experimental
	  cell capability. The guarded visible-motion MPI2 smoke
	  `high_order_visible_motion_tilt2d_production_guard_20260520.json` passed
	  20 accepted steps with `Required_implicit_cut_backend_qualification` set
	  to `ProductionQualified`, `ProductionQualified:6318` parsed backend
	  capabilities, `SayeHyperrectangle:6318` selected backend records, zero
	  implicit-cut fallback cells, one full-miss generated-cell-cache rebuild,
	  5,733 generated-cell-cache hits, 585 misses, 78 cut-context rebuilds
	  (`3.9` per accepted step), and 3.025 s wall time (`0.1513` s per accepted
	  step). The largest parsed assembly total was `0.019636` s and largest
	  cut-volume assembly time was `0.010841` s, so the guarded 81-cell 2D case
	  remains in the expected runtime range for its size.
	- 2026-05-20: equation-level cut-domain requests now have a first generic FE
	  consumer path outside Navier-Stokes. During module registration the
	  application resolves each equation-level generated level-set cut-domain
	  request to the same stable generated interface marker used by the cut
	  lifecycle, installs a scoped `FESystem` cell-domain restriction, and routes
	  ordinary `.dx()` cell kernels emitted by that module to `dCutVolume(marker,
	  side)`. This lets scalar modules such as heat/Darcy bind their regular
	  volume residuals to generated active domains without hand-writing
	  physics-specific `dCutVolume` forms. Focused coverage passed
	  `test_application --gtest_filter='LevelSetCutConfiguration.*'` (`13/13`)
	  and
	  `test_fe_systems --gtest_filter='FormsInstaller.*CutVolume*:FormsInstaller.FormsInstaller_CellRestrictionRoutesDxToCutVolumeKernel'`
	  (`10/10`), including a regression that proves an unrestricted `.dx()` form
	  is registered as a cut-volume kernel while the scoped restriction is active
	  and returns to a normal cell kernel after the restriction is cleared. This
	  does not yet make boolean/multiple-level-set domains or every multiphysics
	  module production-complete, but it removes the prior hard limitation that
	  only Navier-Stokes volume forms could consume the generated active
	  cut-domain context.
	- Remaining production-hardening gaps before calling the whole high-order path
	  production-grade are:
  hosted scheduled/manual CI evidence for the promoted workflow; later-profile
  D18/D38 performance below the elapsed-time budget on production builds after
  the current D18 and D38 MPI2 first-profile gates passed that budget, with the
  current D18 second-profile MPI2 probe missing the budget by a narrow
  performance-only margin and D38 later-profile MPI2 evidence still uncollected;
  additional cross-block cut-volume assembly batching/caching because long
  adaptive profiles still repeat active cut-volume traversal, basis preparation,
  kernel evaluation, and geometry refreshes hundreds or thousands of times even
  though the current three-step probes, D18 120-step segments, and D18/D38 MPI2
  first profiles are under their elapsed-time gates;
  optional later-time D18/D38 benchmark coverage beyond the first profile window
  after the second-profile compute-time issue is diagnosed; and, separately from
  robustness, a future accuracy milestone replacing the current simplex terminal
  linearized leaves (achieved interface/volume order 1/2) with higher-order
  root-polished curved rules if the production definition requires higher
  achieved quadrature order on tetrahedral D18/D38 meshes.
- Remaining formulation-scope gaps before advertising the high-order unfitted
  level-set/free-surface path as production generality are broader than the
  quadrature backend itself. The current Navier-Stokes tangent treats generated
  cut-volume/interface geometry as refreshed but frozen during tangent assembly;
  cut-cell volume, embedded-interface measure, normals, and topology changes are
  therefore not differentiated with respect to the level-set unknown. That is an
  explicit/quasi-Newton contract, not a fully consistent monolithic
  level-set/free-surface Jacobian. `DifferentiatedQuadrature` remains guarded
  until quadrature-point, weight, normal, measure, and topology sensitivities
  exist.
	- The generated cut-domain service now has a generic equation-level request hook
	  and a scoped FE installer consumer that routes ordinary module `.dx()` cell
	  forms to generated `dCutVolume(marker, side)` kernels. It is still not a
	  complete multiphysics restricted-space framework: Navier-Stokes still accepts
	  only one active-domain free surface for volume integration and pressure
	  operations, interface terms and coupled-region APIs are not yet generalized,
	  and non-fluid modules still need end-to-end physics validation on generated
	  active domains. A full general multiphysics production claim still requires
	  shared cut-region ownership, multiple level sets / boolean cut domains, and
	  module-independent restricted-space and interface-coupling APIs.
- General capillary flow is improved but not fully production complete:
  nonzero surface tension with raw level-set curvature is rejected for unfitted
  interfaces, supplied curvature can be a scalar FE field, and an automatic
  nodal projected-curvature maintenance path can populate that field from the
  current level set, including supplemental high-order cell samples when they
  are available. This projected-curvature recovery now has normalized
  fit-residual diagnostics and an optional residual-limit failure gate, so
  coarse or inconsistent local quadratic fits can be rejected instead of
  silently feeding capillary traction. The traction sign convention
  `(-p_ext + gamma*kappa)n` is implemented, and sloshing-with-surface tension,
  zero-gravity capillary-response, and simple Laplace-style capillary-balance
  smokes now exercise that path. What remains is a fully qualified robust
  capillary treatment: global smoothing/projection options, surface
  Laplace-Beltrami force, or an equivalent stabilized formulation with
  higher-resolution Laplace-pressure, droplet, and capillary-wave validation.
  The refreshed projected-curvature field is still Picard-style supplied data;
  curvature derivatives with respect to level-set DOFs are not part of the
  tangent.
- Level-set transport remains the non-conservative scalar advection equation,
  with reinitialization and global volume correction as validation controls
  rather than a conservative transport formulation. The new high-order
  volume-corrected motion gate proves the global shift path can tightly control
  total wet volume during visible free-surface motion. It is still a global
  level-set shift, not local conservative transport; diagnostics and
  documentation must avoid claiming local conservation.
- Runtime reinitialization support is intentionally projection-only at runtime
  and at input parse/translation boundaries. `HamiltonJacobiPDE` and
  `FastMarching` remain future method names until they have runtime
  implementations and validation coverage. The projection repair does not solve
  HJ reinitialization and does not reconstruct higher-order curved
  signed-distance fields.
- Unfitted Nitsche kinematic enforcement is deliberately rejected, so interface
  motion is handled by level-set transport and optional interface residual
  reinforcement rather than a fitted ALE-style surface-position constraint.
  Active-side algebraic support now covers pressure and velocity: pressure has
  focused small-cut P3 regressions for 2D shared-edge and 3D shared-face DOFs,
  including retained-volume and pruned-volume facet-only support, and velocity
  has Product H1 vector inactive-DOF constraint coverage. These prove that
  shared high-order facet trace DOFs remain active while dry-only cell-interior
  DOFs are constrained. The remaining support gap is multiple cut-adjacent
  layers and coupled Navier-Stokes ghost-penalty solves with pathological small
  cuts.
- High-order support is still guarded by backend qualification and by narrower
  maintenance utilities. The 2D quadrilateral Saye hyperrectangle path is now
  the qualified high-order generated-interface slice, while hexahedral Saye
  rules and high-order subcell backends retain experimental/qualification
  diagnostics where terminal linearized leaves or incomplete 3D coverage remain.
  Volume correction now covers linear
  corner-based hex, wedge, and pyramid cells through tetrahedral decomposition,
  but it still measures a corner-linearized level set and applies a global
  shift. Projection reinitialization remains narrower than the generated
  quadrature path and still does not reconstruct curved high-order
  signed-distance fields.
- Performance hardening remains structural, not just diagnostic. Current cut
  context rebuilds still traverse every cell and compute per-cell signatures,
  even though unchanged linear full cells can now reuse cached generated rules.
  Accepted/rejected adaptive trials can still repeat expensive cut-volume
  traversal, and D18/D38 remain dominated by active cut-volume assembly rather than
  nearest-neighbor extension or generated no-cut geometry. Larger multiphysics
  cases need narrow-band cell tracking, incremental cut rebuilds, cut-cell-only
  regeneration for interface neighborhoods, and cross-block cut-volume
  assembly batching/caching. Same-block term fusion, generated full-cell
  caching, vector-revision refresh prechecks, exact nearest-point indexing, and
  default-on cut-volume basis caching have all been checked, so none of those is
  the remaining dominant bottleneck. The short D18/D38 timing probes, the MPI2
  120-step D18 segment, and the completed D18/D38 MPI2 first-profile runs are
  now reasonable compute-time diagnostics for their problem sizes, but later
  D18/D38 profile coverage still needs production-build timing evidence before
  it counts as production performance evidence.

## Open Questions

- [x] Should the first simplex backend use high-order subcell decomposition or
      moment fitting?

      Decision: use high-order subcell decomposition first for triangles and
      tetrahedra. Moment fitting remains a later optional backend, not the first
      production path.

      Rationale and support: the implemented `HighOrderSubcell` path preserves
      positive weights, reference-frame provenance, deterministic partitioning,
      and the existing active-side assembly contracts. Fries et al. 2017 and
      Joulaian et al. 2016 support subcell and implicit-geometry integration as
      practical high-order unfitted FEM strategies, while Muller et al. 2013
      supports moment fitting as a separate accurate option with different
      weight and conditioning policy requirements.

- [x] Do D18/D38 meshes require high-order triangle/tet support before any
      high-order benchmark work is meaningful?

      Decision: yes, if the benchmark mesh contains triangles or tetrahedra and
      the run is intended to qualify high-order cut geometry over the whole
      active domain. A benchmark may exclude those elements only if the
      qualification log states the exclusion and the backend diagnostics show
      that no unsupported high-order cut cells participated.

      Rationale and support: D18/D38-style open-vessel cases exercise mixed
      active-domain topology, pressure constraints, cut-adjacent stabilization,
      and moving wet volume. CutFEM validation practice requires the integrated
      active domain to be the tested domain, not an unreported subset. The
      current minimum meaningful gate is the supported `HighOrderSubcell`
      triangle/tetrahedron path plus explicit unsupported-cell diagnostics.

- [x] What exact fallback rate is acceptable in validation runs?

      Decision: high-order validation and benchmark qualification use zero
      unexpected implicit-cut fallback cells by default. A nonzero fallback rate
      is acceptable only in tests that explicitly exercise fallback behavior or
      in a documented benchmark log with a configured threshold, absolute count,
      cell fraction, affected element families, and physical measure fraction.

      Rationale and support: Saye-style and CutFEM validation literature treats
      geometry/quadrature accuracy as a primary verification target. Silent or
      unbounded fallback would mix geometry models and obscure whether solver
      behavior comes from high-order quadrature, linearized cuts, or unsupported
      cells. The smoke-script default therefore keeps
      `max_diagnostic_implicit_cut_fallback_cells=0`.

- [x] Are signed moment-fit weights acceptable anywhere in Navier-Stokes/VMS
      assembly, or should production require positive weights?

      Decision: production Navier-Stokes/VMS assembly requires finite
      nonnegative cut-volume and interface weights. Signed moment-fit rules are
      allowed only in isolated research or diagnostic tests until a separate
      stability, conditioning, and conservation qualification plan approves
      them for production.

      Rationale and support: the active-domain, stabilization, pressure-support,
      wet-volume, and conservation diagnostics assume positive physical
      measures. Muller et al. 2013 establishes moment fitting as an accurate
      implicit-domain integration technique, but the CutFEM conditioning
      literature and the current pressure/stabilization constraints make
      positive production weights the conservative policy.

- [x] Should exact geometry sensitivities be required before high-order mode is
      allowed in fully coupled level-set/fluid Newton solves?

      Decision: exact geometry sensitivities are required before enabling or
      advertising a fully consistent `DifferentiatedQuadrature` Newton mode.
      The current high-order mode is allowed only under the documented
      `RefreshedFrozenQuadrature` quasi-Newton policy, with current-geometry
      residuals, trial-state geometry refreshes, rejected-trial restore, and
      warning-level tangent diagnostics when coupled geometry derivatives are
      omitted.

      Rationale and support: time-dependent CutFEM work supports updating the
      level-set-defined active domain consistently in time, while Saye-style
      implicit quadrature makes quadrature points, weights, normals, and
      topology geometry-dependent. The implemented fixed-geometry checks and
      refreshed-geometry diagnostics therefore qualify the current quasi-Newton
      policy without claiming exact Newton convergence.

- [x] How should surface-tension curvature be projected or smoothed for
      high-order unfitted free surfaces?

      Decision: surface-tension production support remains outside this
      milestone. When enabled, curvature should be recovered through a
      documented FE projection or patch recovery on the generated interface,
      smoothed with bounded local operators, and checked by mass, force, and
      capillary-wave diagnostics before coupling to Navier-Stokes. Raw pointwise
      curvature from the level-set field should not be used directly in
      production high-order unfitted free-surface assembly.

      Rationale and support: bulk-surface CutFEM work motivates treating
      interface fields as projected finite-element quantities on the cut
      surface, and higher-order XFEM/implicit-geometry literature emphasizes
      accurate curved interface representation. The present quadrature plan
      supplies reliable integration geometry first; curvature force modeling
      needs a separate stability and smoothing policy.

# FE Basis Module

The Basis module provides reference-element basis function evaluation for the
svMultiPhysics finite element library. All evaluation happens in reference
(parametric) space; physical-domain mapping is handled by the Geometry module.

Namespace: `svmp::FE::basis`

## Quick Start

```cpp
#include "Basis/BasisFactory.h"
#include "Basis/BasisCache.h"
#include "Quadrature/QuadratureRule.h"

// Create a quadratic Lagrange basis on tetrahedra
basis::BasisRequest req;
req.element_type = ElementType::Tetra4;
req.basis_type   = BasisType::Lagrange;
req.order        = 2;
auto basis = basis::BasisFactory::create(req);

// Evaluate at a reference point
math::Vector<Real, 3> xi{0.25, 0.25, 0.25};
std::vector<Real> values;
basis->evaluate_values(xi, values);   // values.size() == 10 (Tetra10)

std::vector<basis::Gradient> grads;
basis->evaluate_gradients(xi, grads); // analytic for Lagrange

std::vector<basis::Hessian> hess;
basis->evaluate_hessians(xi, hess);   // analytic for Lagrange

// Pre-evaluate at all quadrature points (cached, thread-safe)
auto& quad = quadrature::get_rule(ElementType::Tetra4, /*degree=*/4);
const auto& entry = basis::BasisCache::instance().get_or_compute(
    *basis, quad, /*gradients=*/true, /*hessians=*/false);

Real N_3_at_q5 = entry.scalarValue(/*dof=*/3, /*qp=*/5);
std::span<const Real> N_3_all = entry.scalarValuesForDof(3);
```

## Architecture

```
BasisRequest ──> BasisFactory::create() ──> shared_ptr<BasisFunction>
                                                    |
                ┌───────────────────────────────────┘
                |
                |── LagrangeBasis         (C0 nodal, canonical complete topologies)
                |── HierarchicalBasis     (modal, canonical topologies)
                |── BernsteinBasis        (partition of unity, selected topologies)
                |── SpectralBasis         (GLL nodes)
                |── SerendipityBasis      (retained low-order reduced-DOF family)
                |── HermiteBasis          (C1 continuity)
                |── BubbleBasis           (interior enrichment, one mode per in-scope topology)
                |── BSplineBasis          (Cox-de Boor, 1D)
                |── TensorProductBasis<T> (anisotropic tensor product)
                |── NURBSTensorBasis      (2D/3D rational tensor product)
                |
                └── VectorBasisFunction   (abstract, H(div)/H(curl))
                    |── RaviartThomasBasis   (H(div))
                    |── NedelecBasis         (H(curl))
                    |── CompatibleTensorVectorBasis (compatible spline/NURBS H(div)/H(curl) on Quad4/Hex8)
                    └── BDMBasis             (H(div), simplex-first plus retained quad order-1 path)
```

### Key Design Principles

- **Reference-only**: No mesh connectivity or geometry dependencies. All
  coordinates are in reference (parametric) space.
- **Explicit derivative contracts**: first-class scalar bases expose exact
  gradients and Hessians on their documented support surface. The base class
  finite-difference helpers remain available only for unsupported extension
  paths and custom experimentation.
- **Semantic caching**: `BasisCache` keys on `cache_identity()` strings that
  encode all evaluation-relevant state (knots, weights, order, element type).
- **SoA memory layout**: `BasisCacheEntry` and `BatchedBasisData` store values
  in DOF-major Structure-of-Arrays format for SIMD-friendly access.
- **VTK-compatible node ordering**: Node numbering is consistent with VTK cell
  types for seamless visualization output.

---

## BasisFunction (Abstract Base)

Defined in `BasisFunction.h`. All basis types inherit from this class.

### Type Aliases

```cpp
namespace svmp::FE::basis {
    using Gradient = math::Vector<Real, 3>;  // 3-component gradient vector
    using Hessian  = math::Matrix<Real, 3, 3>;  // 3x3 Hessian matrix
}
```

### Pure Virtual Methods

Every basis must implement these:

```cpp
virtual BasisType   basis_type()   const noexcept = 0;  // Family enum
virtual ElementType element_type() const noexcept = 0;  // Reference element
virtual int         dimension()    const noexcept = 0;  // 1, 2, or 3
virtual int         order()        const noexcept = 0;  // Polynomial order
virtual std::size_t size()         const noexcept = 0;  // Number of DOFs

virtual void evaluate_values(const math::Vector<Real, 3>& xi,
                             std::vector<Real>& values) const = 0;
```

### Virtual Methods with Defaults

```cpp
// Returns true for H(div)/H(curl) families
virtual bool is_vector_valued() const noexcept;  // default: false

// Semantic identity for caching (override for B-spline/tensor-product)
virtual std::string cache_identity() const;

// Reference-space gradients (default: central finite differences, eps=1e-6)
virtual void evaluate_gradients(const math::Vector<Real, 3>& xi,
                                std::vector<Gradient>& gradients) const;

// Reference-space Hessians (default: finite differences, eps=1e-5)
virtual void evaluate_hessians(const math::Vector<Real, 3>& xi,
                               std::vector<Hessian>& hessians) const;

// Vector-valued evaluation (default: throws FEException)
virtual void evaluate_vector_values(const math::Vector<Real, 3>& xi,
                                    std::vector<math::Vector<Real, 3>>& values) const;

// Divergence (default: throws; override in H(div) bases)
virtual void evaluate_divergence(const math::Vector<Real, 3>& xi,
                                 std::vector<Real>& divergence) const;

// Curl (default: throws; override in H(curl) bases)
virtual void evaluate_curl(const math::Vector<Real, 3>& xi,
                           std::vector<math::Vector<Real, 3>>& curl) const;
```

All output vectors are resized to `size()` by the implementation. Unused
gradient/Hessian components (e.g., z-component for 2D elements) are set to zero.

### Scalar Derivative Contract

The table below is the authoritative derivative contract for shipped scalar
basis families. The finite-difference base-class helpers are not part of the
public contract for these families.

| Basis family | Exact gradients | Exact Hessians | Notes |
|---|---|---|---|
| `Lagrange` | yes | yes | complete-family nodal path; rational pyramid values are exact at the apex, while gradients/Hessians remain exact in the interior and throw at the exact apex because the nodal limit is not unique |
| `Hierarchical` | yes | yes | modal recurrence-based derivatives on all supported topologies |
| `Bernstein` | yes | yes | line, simplex, tensor, wedge, and pyramid support |
| `Spectral` | yes | yes | tensor, simplex, wedge, and pyramid modal/collocation path |
| `Serendipity` | yes | yes | supported surface only: quad arbitrary order, retained `Hex20`/`Wedge15`/`Pyramid13`; `Pyramid13` inherits the complete-family rational pyramid apex contract, so exact apex values are available while gradients/Hessians throw at the exact apex |
| `Hermite` | yes | yes | intentionally narrow cubic tensor-product surface: `Line2`, `Quad4`, `Hex8` |
| `Bubble` | yes | yes | one analytic interior mode per supported topology |
| `BSpline` | yes | yes | 1D spline and rational 1D NURBS path |
| `TensorProductBasis<T>` | yes | yes | exact if the per-axis basis exposes exact derivatives |
| `NURBSTensorBasis` | yes | yes | exact tensor-product quotient rule |

---

## Scalar Basis Families

### LagrangeBasis

Standard nodal interpolation with equispaced nodes on canonical reference
elements. Supports arbitrary polynomial order on the canonical complete
families `Line2`, `Triangle3`, `Quad4`, `Tetra4`, `Hex8`, `Wedge6`, and
`Pyramid5`. Low-order complete-family aliases (`Line3`, `Triangle6`, `Quad9`,
`Tetra10`, `Hex27`, `Wedge18`, `Pyramid14`) normalize to the same canonical
path. Serendipity variants (`Quad8`, `Hex20`, `Wedge15`, `Pyramid13`) remain
separate through `SerendipityBasis`. Gradients and Hessians are analytic on the
supported interior reference domain; on the rational pyramid family, values
remain exact at the apex while gradients and Hessians throw at the exact apex
because the nodal limit is not unique.

```cpp
// Constructor: type must be a canonical or complete-family alias
LagrangeBasis(ElementType type, int order);

BasisType basis_type() const noexcept;    // BasisType::Lagrange
ElementType element_type() const noexcept;  // Canonical type after normalization
int dimension() const noexcept;
int order() const noexcept;
std::size_t size() const noexcept;          // Number of nodes (DOFs)

// Direct access to equispaced node coordinates in reference space
const std::vector<math::Vector<Real, 3>>& nodes() const noexcept;

// Interior derivatives are analytic. On the rational pyramid family, exact
// apex values are available while exact-apex gradients/Hessians throw.
void evaluate_values(xi, values) const;
void evaluate_gradients(xi, gradients) const;
void evaluate_hessians(xi, hessians) const;
```

**Example:**

```cpp
LagrangeBasis basis(ElementType::Hex8, /*order=*/3);
basis.size();           // 64 (4^3)
const auto& nodes = basis.nodes();  // 64 equispaced nodes in [-1,1]^3

// Tetra10 normalizes to Tetra4 order 2
LagrangeBasis tet10(ElementType::Tetra10, /*order=*/2);
tet10.element_type();   // ElementType::Tetra4
tet10.size();           // 10
```

### HierarchicalBasis

Modal basis using integrated Legendre polynomials (tensor-product elements) and
Dubiner/Proriol polynomials (simplices). Hierarchical: order-p contains all
order-(p-1) functions. Supports all canonical reference topologies including
wedge and pyramid.

```cpp
HierarchicalBasis(ElementType type, int order);

BasisType basis_type() const noexcept;  // BasisType::Hierarchical

void evaluate_values(xi, values) const;
void evaluate_gradients(xi, gradients) const;  // analytic
void evaluate_hessians(xi, hessians) const;    // analytic
```

### BernsteinBasis

Bernstein polynomial basis with partition-of-unity property. Useful for
isogeometric-style formulations and Bezier extraction. Supports Line,
Triangle, Quad, Tetra, Hex, Wedge, and Pyramid elements.

```cpp
BernsteinBasis(ElementType type, int order);

BasisType basis_type() const noexcept;  // BasisType::Bernstein

void evaluate_values(xi, values) const;
void evaluate_gradients(xi, gradients) const;  // analytic
void evaluate_hessians(xi, hessians) const;    // analytic
```

### SpectralBasis

Gauss-Lobatto-Legendre (GLL) nodal basis for spectral element methods.
Tensor-product elements (Line, Quad, Hex) use 1D GLL nodes; simplices
(Triangle, Tetrahedron) use Warp & Blend nodes with an inverse Vandermonde
transform. Wedge uses a triangle spectral face basis tensorized with a 1D GLL
line basis. Pyramid uses a rational/modal construction with an inverse
Vandermonde transform over a blended nodal set.

```cpp
SpectralBasis(ElementType type, int order);

BasisType basis_type() const noexcept;  // BasisType::Spectral

// Access to 1D GLL node positions (tensor-product elements only)
const std::vector<Real>& nodes_1d() const noexcept;

// Whether this uses simplex-style (Warp & Blend / nodal inverse Vandermonde)
// construction rather than a pure tensor-product basis
bool is_simplex() const noexcept;

// Interpolation nodes for simplex, wedge, and pyramid constructions
const std::vector<math::Vector<Real, 3>>& interpolation_nodes() const noexcept;

void evaluate_values(xi, values) const;
void evaluate_gradients(xi, gradients) const;  // analytic
void evaluate_hessians(xi, hessians) const;    // analytic
```

**Example:**

```cpp
SpectralBasis spec(ElementType::Hex8, /*order=*/4);
spec.size();          // 125 (5^3)
spec.nodes_1d();      // 5 GLL nodes in [-1, 1]
spec.is_simplex();    // false

SpectralBasis tri(ElementType::Triangle3, /*order=*/3);
tri.size();           // 10
tri.is_simplex();     // true
```

### SerendipityBasis

Reduced-DOF variant that eliminates interior nodes from tensor-product
elements. The current public contract is intentionally narrow:

- quadrilateral serendipity on orders 1-2 (`Quad4`/`Quad8`)
- hexahedral serendipity on orders 1-2 (`Hex8`/`Hex20`)
- topology-specific quadratic reduced bases on `Wedge15` and `Pyramid13`

Higher-order generalized serendipity on `Hex`, `Wedge`, and `Pyramid` is
currently out of scope and throws deterministically until an exact reduced-DOF
family is defined per topology. `geometry_mode` does not widen that contract;
it only reduces the geometry polynomial on the already supported low-order
families. `Pyramid13` inherits the complete-family rational pyramid apex
contract from `Pyramid14`: exact apex values are available, while gradients and
Hessians throw at the exact apex because the parent nodal derivative limit is
not unique.

```cpp
SerendipityBasis(ElementType type, int order, bool geometry_mode = false);

BasisType basis_type() const noexcept;  // BasisType::Serendipity

void evaluate_values(xi, values) const;
void evaluate_gradients(xi, gradients) const;  // analytic on the supported interior domain
void evaluate_hessians(xi, hessians) const;    // analytic on the supported interior domain
```

**Supported element types:**

| Element | Order | DOFs |
|---------|-------|------|
| Quad4 | 1 | 4 |
| Quad8 | 2 | 8 |
| Hex8 | 1 | 8 |
| Hex20 | 2 | 20 |
| Wedge15 | 2 | 15 |
| Pyramid13 | 2 | 13 |

### HermiteBasis

C1-continuous basis providing value and derivative continuity across element
boundaries. The public contract is intentionally restricted to cubic order on
tensor-product elements. Higher-order tensor Hermite and non-tensor `C1`
families remain out of scope unless they are introduced as explicit new
families with their own DOF contract.

```cpp
HermiteBasis(ElementType element_type, int order);  // order must be 3

BasisType basis_type() const noexcept;  // BasisType::Hermite
void evaluate_values(xi, values) const;
void evaluate_gradients(xi, gradients) const;  // analytic
void evaluate_hessians(xi, hessians) const;    // analytic
```

**Supported elements and DOF layout:**

| Element | DOFs | DOF Layout |
|---------|------|------------|
| Line2 | 4 | value + derivative at each endpoint |
| Quad4 | 16 | value + d/dx + d/dy + d^2/(dxdy) at each corner |
| Hex8 | 64 | Tensor product of corner modes (value + 3 first + 3 second + 1 mixed per corner) |

Reference domain is mapped from [-1, 1] to [0, 1] internally via t = (1 + xi) / 2.

### BubbleBasis

Single interior bubble function for element enrichment (MINI elements,
stabilized methods). Order is implicit from the element type.

```cpp
explicit BubbleBasis(ElementType type);  // No order parameter

BasisType basis_type() const noexcept;  // BasisType::Bubble
std::size_t size() const noexcept;      // Always 1

void evaluate_values(xi, values) const;
void evaluate_gradients(xi, gradients) const;  // analytic
void evaluate_hessians(xi, hessians) const;    // analytic
```

| Element | Bubble Function | Order |
|---------|-----------------|-------|
| Line2 | `1 - xi^2` | 2 |
| Triangle3 | 27 L0 L1 L2 | 3 (cubic) |
| Tetra4 | 256 L0 L1 L2 L3 | 4 (quartic) |
| Quad4 | (1-xi^2)(1-eta^2) | 2 (biquadratic) |
| Hex8 | (1-xi^2)(1-eta^2)(1-zeta^2) | 2 (triquadratic) |
| Wedge6 | `27 L0 L1 L2 (1-zeta^2)` | 5 in total degree |
| Pyramid5 | `(3125/256) z((1-zeta)^2-xi^2)((1-zeta)^2-eta^2)` | 5 (quintic polynomial) |

Where L_i are barycentric coordinates for simplices.
Bubble construction is topology-driven, so the same intrinsic bubble is also
accepted for the corresponding node-count aliases: `Line3`, `Triangle6`,
`Tetra10`, `Quad8`, `Quad9`, `Hex20`, `Hex27`, `Wedge15`, `Wedge18`,
`Pyramid13`, and `Pyramid14`.

### BSplineBasis

Univariate B-spline (Cox-de Boor recursion) and NURBS (rational weights).
Maps from reference domain xi in [-1, 1] to the knot span. The scalar spline
surface is intentionally tensor-product only: `Line2` in 1D, tensor-product
`Quad4` in 2D, and tensor-product `Hex8` in 3D. Multi-dimensional
non-rational B-spline bases are built via `TensorProductBasis<BSplineBasis>`.
Multi-dimensional rational NURBS bases use `NURBSTensorBasis` with flattened
control-net weights and explicit tensor extents. Mixed-topology spline/NURBS
families such as wedge or pyramid remain out of scope unless a distinct family
is designed explicitly.

```cpp
// Non-rational B-spline
BSplineBasis(int degree, std::vector<Real> knots);

// Rational NURBS
BSplineBasis(int degree, std::vector<Real> knots, std::vector<Real> weights);

BasisType basis_type() const noexcept;  // BSpline or NURBS
ElementType element_type() const noexcept;  // Always Line2
int dimension() const noexcept;  // Always 1
int order() const noexcept;      // degree
std::size_t size() const noexcept;  // knots.size() - degree - 1

// Knot vector and weights access
const std::vector<Real>& knots() const noexcept;
const std::vector<Real>& weights() const noexcept;
bool is_rational() const noexcept;  // true if weights provided

// Includes detailed knot/weight info for unique identification
std::string cache_identity() const override;

void evaluate_values(xi, values) const;
void evaluate_gradients(xi, gradients) const;  // analytic
void evaluate_hessians(xi, hessians) const;    // analytic
```

**Example:**

```cpp
// Univariate cubic B-spline (4 basis functions)
BSplineBasis bspline(/*degree=*/3, {0, 0, 0, 0, 1, 1, 1, 1});
bspline.size();          // 4
bspline.is_rational();   // false

// Rational NURBS (circular arc weights)
BSplineBasis nurbs(3, {0,0,0,0,1,1,1,1}, {1.0, 0.707, 1.0, 0.707});
nurbs.is_rational();     // true
nurbs.basis_type();      // BasisType::NURBS
```

### NURBSTensorBasis

Dedicated 2D/3D rational tensor-product basis for `Quad4` and `Hex8`.
Construction uses non-rational per-axis `BSplineBasis` factors together with a
flattened control-net weight array and explicit `tensor_extents`. This tensor
surface is intentional; mixed-topology rational spline families are not
implicitly covered by `NURBSTensorBasis`.

```cpp
BSplineBasis bx(2, knots_x);
BSplineBasis by(1, knots_y);
std::vector<Real> weights = /* num_basis_x * num_basis_y entries */;

NURBSTensorBasis surf(bx, by, weights, {num_basis_x, num_basis_y});
surf.basis_type();   // BasisType::NURBS
surf.element_type(); // ElementType::Quad4
```

---

## Vector-Valued Basis Families

Vector bases inherit from `VectorBasisFunction` and set `is_vector_valued() = true`.
Calling `evaluate_values()` on a vector basis throws; use `evaluate_vector_values()`
instead.

### Hybrid-Cell Arbitrary-Order Policy

Wedge and pyramid `H(div)` / `H(curl)` support follows a two-path contract:

- orders `1` and `2` retain the direct seeded constructions as the regression
  baseline
- orders `>= 3` use the same face/edge/interior moment sets, but solve them
  against an overcomplete component-monomial candidate space on the reference
  hybrid cell

This gives an order-independent construction path with exact moment
enforcement, exact `evaluate_divergence(...)` for RT, exact `evaluate_curl(...)`
for ND, and the same `dof_associations()` layout/orientation semantics as the
lower-order families. The committed repo tests qualify the hybrid path through
order `3`; higher orders use the same construction machinery.

### VectorBasisFunction (Abstract)

```cpp
class VectorBasisFunction : public BasisFunction {
public:
    bool is_vector_valued() const noexcept override;  // true

    // Throws FEException — use evaluate_vector_values instead
    void evaluate_values(xi, values) const override;

    // DOF-to-entity metadata for orientation-aware assembly
    virtual std::vector<DofAssociation> dof_associations() const;
};
```

### DOF Associations

Each DOF in a vector basis is associated with a geometric entity:

```cpp
enum class DofEntity {
    Vertex,    // Vertex DOF
    Edge,      // Edge DOF (tangential moments for H(curl))
    Face,      // Face DOF (normal moments for H(div), tangential for H(curl))
    Interior   // Interior DOF
};

struct DofAssociation {
    DofEntity entity_type;  // Which entity type
    int entity_id;          // Local entity index (e.g., edge 0, face 2)
    int moment_index;       // Index within the entity's moment space
};
```

This metadata enables orientation-aware assembly across element boundaries.

### RaviartThomasBasis -- H(div)

Normal-continuous basis for mixed methods. Provides `evaluate_divergence()`.

```cpp
RaviartThomasBasis(ElementType type, int order = 0);

BasisType basis_type() const noexcept;  // BasisType::RaviartThomas

void evaluate_vector_values(xi, values) const;  // vector output
void evaluate_divergence(xi, divergence) const;  // analytic
std::vector<DofAssociation> dof_associations() const;
```

Supports all 2D and 3D element types: Triangle, Quad, Tetra, Hex, Wedge, Pyramid.
Simplex and tensor-product core cells support the existing implemented order
range. Wedge and pyramid use direct seeded transforms for orders `1`-`2` and
the arbitrary-order moment-fitted hybrid path for orders `>= 3`. Exact
face/interior DOF qualification is committed through order `3`.

### NedelecBasis -- H(curl)

Tangential-continuous edge elements. Provides `evaluate_curl()`.

```cpp
NedelecBasis(ElementType type, int order = 0);

BasisType basis_type() const noexcept;  // BasisType::Nedelec

void evaluate_vector_values(xi, values) const;  // vector output
void evaluate_curl(xi, curl) const;  // analytic
std::vector<DofAssociation> dof_associations() const;
```

Supports all 2D and 3D element types: Triangle, Quad, Tetra, Hex, Wedge, Pyramid.
Simplex and tensor-product core cells support the existing implemented order
range. Wedge and pyramid use direct seeded transforms for orders `1`-`2` and
the arbitrary-order moment-fitted hybrid path for orders `>= 3`. Exact
edge/face DOF qualification is committed through order `3`.

### BDMBasis -- H(div)

Full-polynomial H(div) basis (Brezzi-Douglas-Marini). Provides
`evaluate_divergence()`.

```cpp
BDMBasis(ElementType type, int order = 1);

BasisType basis_type() const noexcept;  // BasisType::BDM

void evaluate_vector_values(xi, values) const;  // vector output
void evaluate_divergence(xi, divergence) const;  // analytic
std::vector<DofAssociation> dof_associations() const;
```

Current implementation supports simplex BDM on `Triangle3` and `Tetra4` for
orders `>= 1`, plus retained order-1 quadrilateral variants (`Quad4`, `Quad8`,
`Quad9`). Higher-order quadrilateral and hybrid-cell BDM-like families are
currently out of scope and still throw until they have an explicit
family-level contract instead of being overloaded onto the current class.

**Vector basis example:**

```cpp
// RT(0) on triangles: 3 DOFs (one per face)
RaviartThomasBasis rt(ElementType::Triangle3, /*order=*/0);
rt.size();  // 3

std::vector<math::Vector<Real, 3>> vec_vals;
rt.evaluate_vector_values(xi, vec_vals);  // 3 vector-valued functions

std::vector<Real> div;
rt.evaluate_divergence(xi, div);  // 3 divergence values

auto assocs = rt.dof_associations();
// assocs[0].entity_type == DofEntity::Face
// assocs[0].entity_id   == 0  (first face)

// Nedelec(0) on tetrahedra: 6 DOFs (one per edge)
NedelecBasis ned(ElementType::Tetra4, /*order=*/0);
ned.size();  // 6

std::vector<math::Vector<Real, 3>> curl_vals;
ned.evaluate_curl(xi, curl_vals);  // 6 curl vectors
```

---

## TensorProductBasis

Template wrapper that builds multi-dimensional bases from 1D factors. The 1D
template parameter must satisfy the `BasisFunction` interface on a line element.

```cpp
template<typename Basis1D>
class TensorProductBasis : public BasisFunction {
public:
    // Isotropic: same 1D basis in all directions
    explicit TensorProductBasis(const Basis1D& basis_1d, int dimension = 2);

    // Anisotropic 2D: different basis per axis
    TensorProductBasis(const Basis1D& bx, const Basis1D& by);

    // Anisotropic 3D
    TensorProductBasis(const Basis1D& bx, const Basis1D& by, const Basis1D& bz);

    BasisType basis_type() const noexcept;    // Inherits from 1D factor
    ElementType element_type() const noexcept;  // Line2 (1D), Quad4 (2D), Hex8 (3D)
    int dimension() const noexcept;
    int order() const noexcept;    // max(order_x, order_y, order_z)
    std::size_t size() const noexcept;  // Product of per-axis sizes

    // Includes per-axis identities for unique identification
    std::string cache_identity() const override;

    // All analytic via product rule: N_{ij} = N_i(x) * N_j(y)
    void evaluate_values(xi, values) const;
    void evaluate_gradients(xi, gradients) const;
    void evaluate_hessians(xi, hessians) const;
};
```

Node ordering is automatically aligned to VTK conventions for isotropic Lagrange.

**Examples:**

```cpp
// 3D isotropic spectral (Hex, 125 DOFs)
SpectralBasis spec1d(ElementType::Line2, 4);
TensorProductBasis<SpectralBasis> spec3d(spec1d, /*dimension=*/3);
spec3d.size();  // 125

// 2D anisotropic B-spline
BSplineBasis bx(3, knots_x), by(2, knots_y);
TensorProductBasis<BSplineBasis> aniso(bx, by);
aniso.element_type();  // ElementType::Quad4
aniso.order();         // 3 (max of 3, 2)

// 3D anisotropic B-spline
BSplineBasis bz(1, knots_z);
TensorProductBasis<BSplineBasis> vol(bx, by, bz);
vol.element_type();  // ElementType::Hex8
```

---

## BasisFactory

Runtime creation from a `BasisRequest` descriptor. This is the recommended way
to create basis objects, as it handles all the dispatch logic for continuity,
field type, tensor products, and B-spline/NURBS construction.

### BasisRequest

```cpp
struct BasisRequest {
    ElementType element_type;           // Required
    BasisType   basis_type;             // Required
    std::optional<int> order{};         // Required for order-bearing families; omit for Bubble

    Continuity  continuity  = Continuity::C0;      // C0, C1, H_div, or H_curl
    FieldType   field_type  = FieldType::Scalar;    // Scalar or Vector

    // B-Spline / NURBS (1D):
    std::vector<Real> knot_vector{};
    std::vector<Real> weights{};

    // Anisotropic tensor-product B-Spline / NURBS:
    std::vector<int> axis_orders{};
    std::vector<std::vector<Real>> axis_knot_vectors{};
    std::vector<std::vector<Real>> axis_weights{};  // legacy field; multidim NURBS uses flattened weights below
    std::vector<int> tensor_extents{};

    // Registry-backed custom basis creation:
    std::string custom_id{};
};
```

### Factory Method

```cpp
class BasisFactory {
public:
    static std::shared_ptr<BasisFunction> create(const BasisRequest& req);
};
```

### Dispatch Logic

The factory routes requests in this priority order:

1. **Vector field dispatch** (`continuity == H_div` or `continuity == H_curl`):
   - `H_div` + `BasisType::BDM` -> `BDMBasis`
   - `H_div` (otherwise) -> `RaviartThomasBasis`
   - `H_curl` -> `NedelecBasis`

2. **C1 continuity** (`continuity == C1`):
   - Scalar only -> `HermiteBasis`

3. **Scalar basis type switch** (default C0):

   | `BasisType` | Created Class |
   |---|---|
   | `Lagrange` | `LagrangeBasis` |
   | `Hierarchical` | `HierarchicalBasis` |
   | `Bernstein` | `BernsteinBasis` |
   | `Spectral` | `SpectralBasis` |
   | `Serendipity` | `SerendipityBasis` |
   | `Hermite` | `HermiteBasis` |
   | `Bubble` | `BubbleBasis` |
   | `BSpline` | `BSplineBasis` (1D) or `TensorProductBasis<BSplineBasis>` (2D/3D) |
   | `NURBS` | `BSplineBasis` with weights (Line2) or `NURBSTensorBasis` (Quad4/Hex8) |
   | `Custom` | Registered factory result identified by `custom_id` |

`BasisType::Bubble` is the only intrinsic-order family in the standard request
API: its polynomial degree is determined by topology, so `BasisRequest::order`
must be omitted. The order-bearing families above require an explicit
`order` value on the request path.

**B-spline/NURBS multi-dimensional construction**: When `element_type` is
`Quad4` or `Hex8`, the factory constructs per-axis `BSplineBasis` objects from
`axis_orders` and `axis_knot_vectors`. Non-rational requests use
`TensorProductBasis<BSplineBasis>`. Rational requests use `NURBSTensorBasis`
with flattened control-net `weights` and explicit `tensor_extents`.

### Factory Examples

```cpp
// Standard Lagrange P2 on triangles
basis::BasisRequest req;
req.element_type = ElementType::Triangle3;
req.basis_type   = BasisType::Lagrange;
req.order        = 2;
auto basis = basis::BasisFactory::create(req);  // 6 DOFs

// C1 Hermite cubic on hexahedra
basis::BasisRequest hermite_req;
hermite_req.element_type = ElementType::Hex8;
hermite_req.basis_type   = BasisType::Hermite;
hermite_req.order        = 3;
hermite_req.continuity   = Continuity::C1;
auto hermite = basis::BasisFactory::create(hermite_req);  // 64 DOFs

// Raviart-Thomas H(div) on tetrahedra
basis::BasisRequest rt_req;
rt_req.element_type = ElementType::Tetra4;
rt_req.basis_type   = BasisType::RaviartThomas;
rt_req.order        = 1;
rt_req.continuity   = Continuity::H_div;
auto rt = basis::BasisFactory::create(rt_req);

// Bubble on wedges: intrinsic order, so omit req.order
basis::BasisRequest bubble_req;
bubble_req.element_type = ElementType::Wedge6;
bubble_req.basis_type   = BasisType::Bubble;
auto bubble = basis::BasisFactory::create(bubble_req);  // 1 DOF

// Anisotropic 2D B-spline
basis::BasisRequest bsp_req;
bsp_req.element_type = ElementType::Quad4;
bsp_req.basis_type   = BasisType::BSpline;
bsp_req.order        = 3;
bsp_req.axis_orders  = {3, 2};
bsp_req.axis_knot_vectors = {
    {0, 0, 0, 0, 1, 1, 1, 1},  // x: degree 3
    {0, 0, 0, 1, 1, 1}          // y: degree 2
};
auto bspline2d = basis::BasisFactory::create(bsp_req);

// Rational 2D NURBS
basis::BasisRequest nurbs_req;
nurbs_req.element_type = ElementType::Quad4;
nurbs_req.basis_type   = BasisType::NURBS;
nurbs_req.order        = 2;
nurbs_req.axis_orders  = {2, 1};
nurbs_req.axis_knot_vectors = {
    {0, 0, 0, 0.5, 1, 1, 1},
    {0, 0, 0.4, 1, 1}
};
nurbs_req.tensor_extents = {4, 3};
nurbs_req.weights = {
    1.0, 1.0, 0.8, 1.0,
    1.0, 1.2, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0
};
auto nurbs2d = basis::BasisFactory::create(nurbs_req);
```

**Custom basis registration**: `BasisType::Custom` is resolved through a
registry keyed by `BasisRequest::custom_id`.

```cpp
basis::BasisFactory::register_custom(
    "my-basis",
    [](const basis::BasisRequest& req) -> std::shared_ptr<basis::BasisFunction> {
        return std::make_shared<MyBasis>(req.order);
    });

basis::BasisRequest custom_req;
custom_req.element_type = ElementType::Line2;
custom_req.basis_type   = BasisType::Custom;
custom_req.order        = 2;
custom_req.custom_id    = "my-basis";
auto custom = basis::BasisFactory::create(custom_req);
```

---

## BasisCache

Thread-safe singleton that caches basis evaluations at quadrature points.
Pre-computes values, gradients, and optionally Hessians at all quadrature
points once, then serves subsequent requests from cache.

### API

```cpp
class BasisCache {
public:
    static BasisCache& instance();  // Thread-safe singleton access

    // Get cached entry or compute and cache it
    const BasisCacheEntry& get_or_compute(
        const BasisFunction& basis,
        const quadrature::QuadratureRule& quad,
        bool gradients = true,
        bool hessians = false);

    void clear();            // Clear all cached entries
    std::size_t size() const;  // Number of cached entries
};
```

### Cache Key

Entries are keyed by the tuple `(cache_identity, quadrature_identity, gradients, hessians)`.
The `cache_identity()` string encodes all evaluation-relevant state for the basis
(family, element, order, and for B-splines: knots, weights, per-axis parameters).

### BasisCacheEntry

```cpp
struct BasisCacheEntry {
    std::size_t num_qpts;
    std::size_t num_dofs;

    // Scalar values in DOF-major SoA layout: [dof * num_qpts + qp]
    std::vector<Real> scalar_values;

    // Gradients: outer = DOF, inner = quadrature points
    std::vector<std::vector<Gradient>> gradients;

    // Hessians: outer = DOF, inner = quadrature points
    std::vector<std::vector<Hessian>> hessians;

    // Vector values (H(div)/H(curl)): outer = DOF, inner = QP
    std::vector<std::vector<math::Vector<Real, 3>>> vector_values;

    // Fast scalar access
    Real scalarValue(std::size_t dof, std::size_t qp) const noexcept;

    // Contiguous span of all QP values for a single DOF (SIMD-friendly)
    std::span<const Real> scalarValuesForDof(std::size_t dof) const noexcept;
};
```

### Thread Safety

The internal map is mutex-protected. After initial computation, concurrent reads
are lock-free via `shared_ptr`. Typical usage: first call computes and caches;
all subsequent calls return the cached entry.

For bases that intentionally throw on exact gradients/Hessians at singular
reference points, `BasisCache` preserves the raw basis contract. In particular,
complete-family rational pyramid bases and inherited `Pyramid13` serendipity
paths allow value-only caching at the exact apex, while gradient/Hessian cache
construction throws if the quadrature includes that point.

### Example

```cpp
auto& cache = basis::BasisCache::instance();

// Compute once
const auto& entry = cache.get_or_compute(*basis, quad, true, false);

// Fast access patterns
for (std::size_t dof = 0; dof < entry.num_dofs; ++dof) {
    // Contiguous span for SIMD loops
    auto vals = entry.scalarValuesForDof(dof);
    for (std::size_t qp = 0; qp < entry.num_qpts; ++qp) {
        Real N_dof_qp = vals[qp];  // or entry.scalarValue(dof, qp)
    }
}
```

---

## BatchEvaluator

High-performance SIMD-optimized evaluator for element-level assembly. Stores
results in SoA layout with SIMD-aligned memory for vectorization across
quadrature points.

### API

```cpp
class BatchEvaluator {
public:
    BatchEvaluator(const BasisFunction& basis,
                   const quadrature::QuadratureRule& quad,
                   bool compute_gradients = true,
                   bool compute_hessians = false);

    const BatchedBasisData& data() const;
    std::size_t num_basis() const;
    std::size_t num_quad_points() const;

    // Field interpolation: result[j] = sum_i coeffs[i] * N_i(q_j) * weights[j]
    void weighted_sum(const Real* coeffs,
                      const Real* weights,
                      Real* result) const;

    // Gradient interpolation
    void weighted_gradient_sum(const Real* coeffs,
                               const Real* weights,
                               Real* result) const;

    // Stiffness assembly: K_ij += w_q * dN_i . (D . dN_j)
    void assemble_stiffness_contribution(const Real* D,
                                         const Real* weights,
                                         Real* K) const;
};
```

`BatchEvaluator` preserves the same singular-point contract as the raw basis.
Value-only construction succeeds on quadrature rules that include the rational
pyramid apex, while gradient/Hessian-enabled construction throws during
precomputation instead of returning partially populated batched data.

### BatchedBasisData

```cpp
struct BatchedBasisData {
    std::vector<Real> values;     // [num_basis][num_quad_points]
    std::vector<Real> gradients;  // [num_basis][3][num_quad_points]
    std::vector<Real> hessians;   // [num_basis][9][num_quad_points]

    std::size_t num_basis;
    std::size_t num_quad_points;
    bool has_gradients;
    bool has_hessians;

    // Element accessors
    Real value(std::size_t i, std::size_t j) const;    // N_i(q_j)
    Real gradient(std::size_t i, std::size_t d, std::size_t j) const;  // dN_i/d(xi_d) at q_j
    Real hessian(std::size_t i, std::size_t d1, std::size_t d2, std::size_t j) const;

    // Contiguous pointers for SIMD loops
    const Real* values_for_basis(std::size_t i) const;
    const Real* gradients_for_basis(std::size_t i, std::size_t d) const;
};
```

### Example

```cpp
basis::BatchEvaluator eval(*basis, quad, /*gradients=*/true);

// Element stiffness assembly with identity material tensor
std::vector<Real> D = {1, 0, 0, 0, 1, 0, 0, 0, 1};  // 3x3 identity
std::vector<Real> jac_weights(eval.num_quad_points());
std::vector<Real> K(eval.num_basis() * eval.num_basis(), 0.0);
eval.assemble_stiffness_contribution(D.data(), jac_weights.data(), K.data());

// Field interpolation
std::vector<Real> coeffs(eval.num_basis());  // DOF values
std::vector<Real> weights(eval.num_quad_points());
std::vector<Real> result(eval.num_quad_points());
eval.weighted_sum(coeffs.data(), weights.data(), result.data());
```

---

## Utilities

### OrthogonalPolynomials (`OrthogonalPolynomials.h`)

Namespace `svmp::FE::basis::orthopoly`. Recurrence-based polynomial evaluations
used internally by `HierarchicalBasis` and `SpectralBasis`.

```cpp
// Legendre polynomials
Real legendre(int n, Real x);
std::pair<Real, Real> legendre_with_derivative(int n, Real x);
std::vector<Real> legendre_sequence(int n, Real x);
std::pair<std::vector<Real>, std::vector<Real>>
    legendre_sequence_with_derivatives(int n, Real x);

// Integrated Legendre (for hierarchical bases)
Real integrated_legendre(int n, Real x);

// Jacobi polynomials P_n^{(alpha,beta)}(x)
Real jacobi(int n, Real alpha, Real beta, Real x);
Real jacobi_derivative(int n, Real alpha, Real beta, Real x);

// 2D simplex: Dubiner basis on the reference triangle
Real dubiner(int p, int q, Real xi, Real eta);
std::tuple<Real, Real, Real>
    dubiner_with_derivatives(int p, int q, Real xi, Real eta);

// 3D simplex: Proriol polynomial on the reference tetrahedron
Real proriol(int p, int q, int r, Real xi, Real eta, Real zeta);
std::tuple<Real, Real, Real, Real>
    proriol_with_derivatives(int p, int q, int r, Real xi, Real eta, Real zeta);
```

### ModalTransform (`ModalTransform.h`)

Converts between modal (hierarchical) and nodal (Lagrange) representations via
Vandermonde matrix and its inverse. Useful for p-adaptivity workflows and
representation conversion.

```cpp
class ModalTransform {
public:
    ModalTransform(const BasisFunction& modal_basis,
                   const LagrangeBasis& nodal_basis);

    // Modal-to-nodal: nodal = V * modal
    std::vector<Real> modal_to_nodal(const std::vector<Real>& modal_coeffs) const;

    // Nodal-to-modal: modal = V^{-1} * nodal
    std::vector<Real> nodal_to_modal(const std::vector<Real>& nodal_values) const;

    const std::vector<std::vector<Real>>& vandermonde() const noexcept;
    const std::vector<std::vector<Real>>& vandermonde_inverse() const noexcept;

    // Condition number estimate (infinity norm) for numerical stability diagnostics
    Real condition_number() const;
};
```

**Example:**

```cpp
LagrangeBasis nodal(ElementType::Triangle3, 3);
HierarchicalBasis modal(ElementType::Triangle3, 3);
basis::ModalTransform transform(modal, nodal);

std::vector<Real> nodal_values = /* ... */;
auto modal_coeffs = transform.nodal_to_modal(nodal_values);
auto roundtrip    = transform.modal_to_nodal(modal_coeffs);
// roundtrip == nodal_values (up to conditioning)

Real cond = transform.condition_number();  // Check stability
```

### NodeOrderingConventions (`NodeOrderingConventions.h`)

Comprehensive reference-element node coordinates and ordering conventions for
all supported element enums, aligned with VTK numbering.

```cpp
class NodeOrdering {
public:
    // Reference coordinates for a specific node
    static math::Vector<Real, 3> get_node_coords(ElementType elem_type,
                                                  std::size_t local_node);

    // Number of nodes for an element type
    static std::size_t num_nodes(ElementType elem_type);

    // Generate arbitrary-order complete Lagrange node coordinates
    static std::vector<math::Vector<Real, 3>>
    get_lagrange_node_coords(ElementType canonical_type, int order);

    // Topology classification
    static bool is_simplex(ElementType elem_type);
    static bool is_tensor_product(ElementType elem_type);
};
```

---

## Reference Element Coordinate Systems

All basis evaluations use these reference coordinate conventions:

### 1D Elements

```
Line: xi in [-1, 1]

  0---------1
  |         |
 xi=-1     xi=+1
```

### 2D Tensor-Product Elements

```
Quad: (xi, eta) in [-1, 1]^2

  3-----------2
  |           |
  |    (eta)  |
  |     ^     |
  0-----+-->--1  (xi)
```

### 2D Simplex Elements

```
Triangle: (xi, eta) in simplex {xi >= 0, eta >= 0, xi + eta <= 1}

  2                    Barycentric coordinates:
  |\                     L0 = 1 - xi - eta
  | \                    L1 = xi
  |  \                   L2 = eta
  |   \
  0----1
```

### 3D Tensor-Product Elements

```
Hex: (xi, eta, zeta) in [-1, 1]^3

       7-----------6
      /|          /|
     / |         / |
    4-----------5  |
    |  |        |  |
    |  3--------|--2
    | /         | /
    |/          |/
    0-----------1
```

### 3D Simplex Elements

```
Tetra: (xi, eta, zeta) in simplex {xi,eta,zeta >= 0, xi+eta+zeta <= 1}

            3              Barycentric coordinates:
           /|\               L0 = 1 - xi - eta - zeta
          / | \              L1 = xi
         /  |  \             L2 = eta
        /   |   \            L3 = zeta
       0----|----2
        \   |   /
         \  |  /
          \ | /
           \|/
            1
```

### Wedge (Prism) Elements

```
Wedge: Triangle base x line height, zeta in [-1, 1]

        5
       /|\
      / | \
     /  |  \
    3---|---4       zeta = +1
    |   2   |
    |  / \  |
    | /   \ |
    |/     \|
    0-------1       zeta = -1
```

### Pyramid Elements

```
Pyramid: Quad base at zeta=0, apex at zeta=1

          4             apex: (0, 0, 1)
         /|\
        / | \
       /  |  \
      3---|----2        zeta = 0
      |   +   |         (apex projects to center of base)
      |       |
      0-------1
```

---

## Basis Capability Matrix

The table below is the authoritative support matrix for the current FE basis
stack. "FE space exposure" refers to the standard space wrappers in
`Code/Source/solver/FE/Spaces`; externally supplied basis-backed paths remain
available through `GenericBasisSpace`.

| Basis family | Topologies / orders | Field / continuity | FE element exposure | FE space exposure | Notes |
|---|---|---|---|---|---|
| `Lagrange` | canonical complete families on line, triangle, quad, tetra, hex, wedge, pyramid; arbitrary implemented order | scalar, `C0` or `L2` | `LagrangeElement`, `DiscontinuousElement` | `H1Space`, `L2Space` | complete-family aliases (`Line3`, `Quad9`, `Hex27`, etc.) normalize to the canonical path |
| `Hierarchical` | line, triangle, quad, tetra, hex, wedge, pyramid; arbitrary implemented order | scalar, `C0`/`L2` | generic basis-backed element path | `H1Space`, `L2Space` | modal basis; legacy generic-basis path remains available |
| `Bernstein` | line, triangle, quad, tetra, hex, wedge, pyramid; arbitrary implemented order | scalar, `C0`/`L2` | generic basis-backed element path | `H1Space`, `L2Space` | tetrahedral support is implemented |
| `Spectral` | line, triangle, quad, tetra, hex, wedge, pyramid; order >= 1 | scalar, `C0`/`L2` | `SpectralElement` for canonical collocation cases, otherwise generic basis-backed element path | `H1Space`, `L2Space` | request-driven first-class scalar-space path through `SpaceFactory::create(const SpaceRequest&)` |
| `Serendipity` | quadrilateral arbitrary implemented order; retained order-2 low-DOF families on `Hex20`, `Wedge15`, `Pyramid13` | scalar, `C0`/`L2` | generic basis-backed element path | `GenericBasisSpace` | arbitrary-order support is intentionally quadrilateral-only; `geometry_mode` does not widen the retained `Hex20`/`Wedge15`/`Pyramid13` surface |
| `Hermite` | `Line2`, `Quad4`, `Hex8`; cubic only | scalar, `C1` | generic basis-backed element path | `C1Space` | intentionally restricted to cubic tensor-product Hermite; `ElementFactory` coerces Hermite requests to `Continuity::C1` |
| `Bubble` | canonical line/triangle/tetra/quad/hex/wedge/pyramid topologies with the matching node-count aliases (`Line3`, `Triangle6`, `Tetra10`, `Quad8/9`, `Hex20/27`, `Wedge15/18`, `Pyramid13/14`) | scalar interior enrichment, `C0`/`L2` | generic basis-backed element path | `GenericBasisSpace` | one interior mode per topology; pyramid uses a quintic polynomial bubble |
| `BSpline` | scalar: `Line2`, `Quad4`, `Hex8` with per-axis orders; vector: compatible-tensor `Quad4`/`Hex8` path with per-axis orders | scalar `C0`/`L2`; vector `H(div)` / `H(curl)` on the compatible-tensor path | generic basis-backed element path for scalar; `GeneralBasisElement` for the vector compatible-tensor path | scalar: `H1Space`, `L2Space`; vector: `HDivSpace`, `HCurlSpace` | tensor-product-only by design for scalar; vector support is intentionally limited to quadrilateral and hexahedral compatible-tensor spaces |
| `NURBS` | scalar: `Line2`, `Quad4`, `Hex8` with per-axis orders; vector: compatible-tensor `Quad4`/`Hex8` path with per-axis orders | scalar `C0`/`L2`; vector `H(div)` / `H(curl)` on the compatible-tensor path | generic basis-backed element path for scalar; `GeneralBasisElement` for the vector compatible-tensor path | scalar: `H1Space`, `L2Space`; vector: `HDivSpace`, `HCurlSpace` | tensor-product-only by design for scalar; vector support is intentionally limited to quadrilateral and hexahedral compatible-tensor spaces |
| `Custom` | request-defined | scalar, typically `C0`/`L2` | generic basis-backed element path after `BasisFactory::register_custom(...)` | `GenericBasisSpace` | registry-backed runtime contract keyed by `custom_id` |
| `RaviartThomas` | triangle, quad, tetra, hex: existing implemented order range; wedge, pyramid: arbitrary implemented order via direct `k=1,2` regressions plus moment-fitted hybrid path for `k>=3` | vector, `H(div)` | `VectorElement` | `HDivSpace` | exact divergence path; repo qualification currently covers hybrid orders `0`-`3` |
| `Nedelec` | triangle, quad, tetra, hex: existing implemented order range; wedge, pyramid: arbitrary implemented order via direct `k=1,2` regressions plus moment-fitted hybrid path for `k>=3` | vector, `H(curl)` | `VectorElement` | `HCurlSpace` | exact curl path; repo qualification currently covers hybrid orders `0`-`3` |
| `BDM` | `Triangle3` and `Tetra4`: orders `>= 1`; `Quad4`/`Quad8`/`Quad9`: order `1` | vector, `H(div)` | `VectorElement` | `HDivSpace` | explicit family selection required; higher-order quadrilateral and hybrid-cell BDM-like families remain out of scope until defined as explicit families |

---

## Error Handling

All basis module errors inherit from `BasisException` (which inherits from
`FEException`). The hierarchy is defined in `BasisExceptions.h`:

| Exception | When thrown |
|---|---|
| `BasisConfigurationException` | Invalid `BasisRequest` fields (missing required order, explicit order on `Bubble`, missing knot vector, incompatible continuity/field type, empty `custom_id`, negative order) |
| `BasisElementCompatibilityException` | Requested element topology is not supported by the basis family (e.g., `SerendipityBasis` on `Triangle3`, `HermiteBasis` on `Tetra4`) |
| `BasisEvaluationException` | Evaluation request cannot be satisfied (e.g., calling `evaluate_values()` on a vector basis, calling `evaluate_divergence()` on a scalar basis) |
| `BasisNodeOrderingException` | Invalid node index or coordinate lookup failure in `NodeOrdering` |
| `BasisConstructionException` | Internal failure during basis setup (e.g., singular Vandermonde matrix, custom factory returned null) |

Convenience macros for throwing within implementations:

```cpp
BASIS_CHECK_CONFIG(condition, "message");     // -> BasisConfigurationException
BASIS_CHECK_COMPAT(condition, "message");     // -> BasisElementCompatibilityException
BASIS_CHECK_EVAL(condition, "message");       // -> BasisEvaluationException
BASIS_CHECK_NODE_ORDER(condition, "message"); // -> BasisNodeOrderingException
BASIS_CHECK_CONSTRUCTION(condition, "message"); // -> BasisConstructionException
```

All exceptions carry source file, line number, and function name for diagnostics.

---

## Choosing an Evaluation Strategy

The module provides three evaluation paths with different performance
characteristics:

### Direct Evaluation

Call `evaluate_values()` / `evaluate_gradients()` / `evaluate_hessians()`
directly on a `BasisFunction` object. Each call evaluates at a single reference
point.

**Use when**: evaluating at a small number of ad-hoc points (point probes,
post-processing queries, testing).

```cpp
auto basis = BasisFactory::create(req);
math::Vector<Real, 3> xi{0.25, 0.25, 0.0};
std::vector<Real> values;
basis->evaluate_values(xi, values);
```

### BasisCache (Pre-computed at Quadrature Points)

Thread-safe singleton that evaluates all basis functions at all quadrature
points once, then serves cached results. Data stored in DOF-major SoA layout.

**Use when**: the same basis/quadrature combination is used across many
elements (standard FE assembly). The first call computes and caches; all
subsequent calls are lock-free reads.

```cpp
const auto& entry = BasisCache::instance().get_or_compute(*basis, quad, true, false);
// Random access
Real val = entry.scalarValue(dof, qp);
// Contiguous span for SIMD loops
auto span = entry.scalarValuesForDof(dof);
```

### BatchEvaluator (SIMD-Optimized Assembly)

Evaluates all basis functions at all quadrature points with SIMD-aligned
memory. Provides fused assembly operations (weighted sums, stiffness
contributions) that vectorize across quadrature points.

**Use when**: building element-level matrices and vectors in performance-critical
assembly loops. The fused operations (`weighted_sum`, `assemble_stiffness_contribution`)
avoid materializing intermediate products.

```cpp
BatchEvaluator eval(*basis, quad, /*gradients=*/true);
eval.weighted_sum(coeffs, weights, result);               // Field interpolation
eval.assemble_stiffness_contribution(D, jac_w, K);        // Stiffness matrix
```

### Decision Guide

| Scenario | Strategy |
|---|---|
| Point probe / post-processing | Direct evaluation |
| Standard FE assembly (many elements, same basis+quad) | `BasisCache` |
| Performance-critical inner loop with fused operations | `BatchEvaluator` |
| One-time evaluation at custom points | Direct evaluation |
| Thread-parallel assembly | `BasisCache` (lock-free reads after first compute) |

---

## Thread Safety

| Component | Thread safety |
|---|---|
| `BasisFunction` subclasses | All `evaluate_*` methods are `const` and stateless; safe for concurrent reads from multiple threads. No mutable state after construction. |
| `BasisFactory::create()` | Safe to call concurrently for non-custom types. Custom registry access is mutex-protected. |
| `BasisCache::instance()` | Thread-safe singleton. `get_or_compute()` uses a mutex for the initial computation; subsequent reads of the same entry are lock-free via `shared_ptr`. |
| `BatchEvaluator` | Not thread-safe. Each thread should construct its own instance or use separate instances. The `data()` accessor is read-only and safe for concurrent reads. |
| `ModalTransform` | Immutable after construction; safe for concurrent reads. |
| Custom registry | `register_custom()` / `unregister_custom()` are mutex-protected and safe to call from any thread. |

---

## Typical FE Integration Workflow

A complete workflow showing how basis evaluation connects to the broader FE
pipeline:

```cpp
#include "Basis/BasisFactory.h"
#include "Basis/BasisCache.h"
#include "Quadrature/QuadratureRule.h"

// 1. Create basis via factory
basis::BasisRequest req;
req.element_type = ElementType::Tetra4;
req.basis_type   = BasisType::Lagrange;
req.order        = 2;
auto basis = basis::BasisFactory::create(req);  // -> LagrangeBasis, 10 DOFs

// 2. Get quadrature rule for the element type
auto& quad = quadrature::get_rule(ElementType::Tetra4, /*degree=*/4);

// 3. Pre-evaluate at all quadrature points (cached, thread-safe)
const auto& cached = basis::BasisCache::instance().get_or_compute(
    *basis, quad, /*gradients=*/true, /*hessians=*/false);

// 4. Element loop: use cached values for assembly
for (auto& cell : mesh.cells()) {
    // Geometry module provides Jacobian and physical mapping
    auto J = geometry.jacobian(cell);

    for (std::size_t qp = 0; qp < cached.num_qpts; ++qp) {
        Real detJ = J.determinant(qp);
        Real w = quad.weight(qp) * detJ;

        for (std::size_t i = 0; i < cached.num_dofs; ++i) {
            // Reference-space values/gradients from cache
            Real N_i = cached.scalarValue(i, qp);
            const auto& dN_ref_i = cached.gradients[i][qp];

            // Physical gradients: dN_phys = J^{-T} * dN_ref
            auto dN_phys_i = J.inverse_transpose(qp) * dN_ref_i;

            // Assemble local contributions...
            for (std::size_t j = 0; j < cached.num_dofs; ++j) {
                const auto& dN_ref_j = cached.gradients[j][qp];
                auto dN_phys_j = J.inverse_transpose(qp) * dN_ref_j;

                K_local(i, j) += w * dot(dN_phys_i, dN_phys_j);
            }
            f_local(i) += w * N_i * source(qp);
        }
    }
}
```

The key separation of concerns:
- **Basis module**: reference-space shape function values and derivatives
- **Quadrature module**: integration points and weights on the reference element
- **Geometry module**: Jacobian, physical coordinates, and reference-to-physical mapping
- **Assembly module**: combines all three to build system matrices and vectors

---

## Build and Test Workflow

Clean rebuild and run the full Basis suite:

```bash
./Code/Source/solver/FE/scripts/run_basis_tests.sh --clean
```

Re-run without cleaning:

```bash
./Code/Source/solver/FE/scripts/run_basis_tests.sh
```

Direct `ctest` invocation from the supported build tree:

```bash
cd build-unit/svMultiPhysics-build/Source/solver/FE
ctest --output-on-failure -R FE_Basis_Tests
```

Focused FE closure validation for newly exposed element/space paths:

```bash
cmake --build build-unit/svMultiPhysics-build --target test_fe_elements test_fe_spaces
./build-unit/svMultiPhysics-build/bin/test_fe_elements
./build-unit/svMultiPhysics-build/bin/test_fe_spaces
```

### Supported Build Tree

The supported Basis test build root is:

```text
build-unit/svMultiPhysics-build
```

The helper script rebuilds `test_fe_basis`, verifies that the current binary can
enumerate the test list, and then runs `ctest` from the matching FE build
directory.

Interim note: if unrelated non-Basis edits elsewhere in FE temporarily break
the shared `svfe` target, that blocks the end-to-end workflow above even when
Basis-local objects and targeted Basis validation are green. The canonical
workflow remains the final validation gate once the shared tree is buildable
again.

### Quarantined Alternate Workflows

Ad hoc or historical FE-only build trees are not part of the supported Basis
workflow. If an alternate build directory contains a stale `test_fe_basis`
binary or stale `CTestTestfile.cmake` entries, treat that tree as unsupported
for Basis validation and use the canonical workflow above instead.

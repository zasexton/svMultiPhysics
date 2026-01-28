# FE Library — Finite Element Infrastructure for svMultiPhysics

## Overview

The FE library provides a complete, modular finite element infrastructure for the svMultiPhysics solver. It is designed with the following core principles:

1. **Mesh-library independence** — The FE library can operate without coupling to any specific mesh implementation
2. **Physics-agnostic infrastructure** — All physics-specific logic is expressed through Forms/Kernels/Constitutive models
3. **JIT compilation support** — LLVM-based JIT compilation accelerates assembly of weak forms
4. **Parallel-ready** — Full MPI support for distributed DOFs, assembly, and ghost exchange
5. **Expression template math** — High-performance fixed-size linear algebra with zero-overhead abstractions

---

## Module Architecture

```
FE/
├── Core/           # Types, exceptions, configuration
├── Math/           # Vector, Matrix, Tensor operations
├── Quadrature/     # Numerical integration rules
├── Basis/          # Shape function implementations
├── Elements/       # Reference finite element definitions
├── Geometry/       # Reference-to-physical mappings
├── Spaces/         # Function space abstractions (H¹, L², H(div), H(curl))
├── Dofs/           # DOF distribution and numbering
├── Sparsity/       # Sparsity pattern construction
├── Assembly/       # Element-to-global assembly strategies
├── Forms/          # Symbolic weak-form expressions and JIT compilation
├── Constraints/    # Boundary conditions and constraints
├── Systems/        # High-level FE system orchestration
├── TimeStepping/   # Time integration schemes
├── Constitutive/   # Material model infrastructure
└── Backends/       # Linear solver backends (FSILS)
```

---

## Module Reference

### 1. Core (`FE/Core/`)

**Purpose:** Foundation types, exceptions, and configuration macros.

#### Key Files

| File | Description |
|------|-------------|
| `Types.h` | Common type aliases: `Real`, `GlobalIndex`, `LocalIndex`, `MeshGlobalId`, `FieldId`, etc. |
| `FEException.h` | Exception hierarchy with file/line/function context |
| `FEConfig.h` | Compile-time feature flags (`FE_HAS_MPI`, `SVMP_FE_WITH_MESH`, etc.) |
| `Logger.h` | Logging utilities |
| `ParameterValue.h` | Type-safe variant for runtime parameters |

#### Type Aliases

```cpp
namespace svmp::FE {
    using Real = double;
    using GlobalIndex = std::int64_t;
    using LocalIndex = std::int32_t;
    using MeshGlobalId = std::int64_t;
    using MeshIndex = std::int32_t;
    using MeshOffset = std::int64_t;
    using FieldId = std::int32_t;
    using OperatorTag = std::string;

    constexpr GlobalIndex INVALID_GLOBAL_INDEX = -1;
    constexpr FieldId INVALID_FIELD_ID = -1;
}
```

---

### 2. Math (`FE/Math/`)

**Purpose:** Fixed-size linear algebra optimized for element-level computations.

#### Vector (`Math/Vector.h`)

SIMD-aligned fixed-size vector with expression templates.

```cpp
namespace svmp::FE::math {

template<typename T, std::size_t N>
class Vector {
public:
    // Construction
    Vector();                           // Zero-initialized
    explicit Vector(T value);           // Fill with value
    Vector(std::initializer_list<T>);   // {x, y, z}

    // Element access
    T& operator[](size_type i);
    const T& operator[](size_type i) const;
    T& at(size_type i);                 // Bounds-checked

    // Arithmetic (in-place)
    Vector& operator+=(const Vector& other);
    Vector& operator-=(const Vector& other);
    Vector& operator*=(T scalar);
    Vector& operator/=(T scalar);

    // Vector operations
    T dot(const Vector& other) const;
    T norm() const;
    T norm_squared() const;
    Vector normalized() const;
    Vector& normalize();

    // Norms
    T norm_l1() const;
    T norm_inf() const;

    // Statistics
    T min() const;
    T max() const;
    T sum() const;
    T mean() const;

    // Static factories
    static Vector zeros();
    static Vector ones();
    static Vector unit(size_type axis);
    static Vector basis(size_type i);
};

// Type aliases
using Vector2d = Vector<double, 2>;
using Vector3d = Vector<double, 3>;
using Vector4d = Vector<double, 4>;
using Vector2f = Vector<float, 2>;
using Vector3f = Vector<float, 3>;

// Free functions
template<typename T, std::size_t N>
T dot(const Vector<T,N>& a, const Vector<T,N>& b);

template<typename T>
Vector3<T> cross(const Vector3<T>& a, const Vector3<T>& b);

template<typename T>
T cross(const Vector2<T>& a, const Vector2<T>& b);  // Returns scalar

template<typename T, std::size_t N>
Vector<T,N> normalize(const Vector<T,N>& v);

template<typename T, std::size_t N>
Vector<T,N> lerp(T t, const Vector<T,N>& a, const Vector<T,N>& b);

template<typename T, std::size_t N>
Vector<T,N> project(const Vector<T,N>& v, const Vector<T,N>& onto);

template<typename T, std::size_t N>
T angle(const Vector<T,N>& a, const Vector<T,N>& b);  // Radians
}
```

#### Matrix (`Math/Matrix.h`)

Row-major fixed-size matrix with specializations for 2×2 and 3×3 (analytical determinant/inverse).

```cpp
namespace svmp::FE::math {

template<typename T, std::size_t M, std::size_t N>
class Matrix {
public:
    // Construction
    Matrix();                                              // Zero
    explicit Matrix(T value);                              // Fill
    Matrix(std::initializer_list<std::initializer_list<T>>); // {{row0}, {row1}, ...}

    // Dimensions
    static constexpr size_type rows();
    static constexpr size_type cols();
    static constexpr size_type size();

    // Element access
    T& operator()(size_type i, size_type j);
    const T& operator()(size_type i, size_type j) const;

    // Row/column access
    Vector<T, N> row(size_type i) const;
    Vector<T, M> column(size_type j) const;
    void set_row(size_type i, const Vector<T, N>& v);
    void set_column(size_type j, const Vector<T, M>& v);

    // Arithmetic
    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);
    Matrix& operator*=(T scalar);
    Matrix& operator/=(T scalar);

    // Matrix operations
    Matrix<T, N, M> transpose() const;
    T trace() const;                    // Square only
    T determinant() const;              // Square only
    Matrix inverse() const;             // Square only

    // Norms
    T frobenius_norm() const;
    T infinity_norm() const;
    T one_norm() const;

    // Properties (square only)
    bool is_symmetric(T tol) const;
    bool is_skew_symmetric(T tol) const;
    bool is_diagonal(T tol) const;

    // Static factories
    static Matrix zeros();
    static Matrix ones();
    static Matrix identity();           // Square only
    static Matrix diagonal(const Vector<T, M>& diag);
};

// Type aliases
using Matrix2x2d = Matrix<double, 2, 2>;
using Matrix3x3d = Matrix<double, 3, 3>;
using Matrix4x4d = Matrix<double, 4, 4>;

// Matrix-vector multiplication
template<typename T, std::size_t M, std::size_t N>
Vector<T, M> operator*(const Matrix<T, M, N>& A, const Vector<T, N>& x);

// Matrix-matrix multiplication
template<typename T, std::size_t M, std::size_t N, std::size_t P>
Matrix<T, M, P> operator*(const Matrix<T, M, N>& A, const Matrix<T, N, P>& B);

// Outer product
template<typename T, std::size_t M, std::size_t N>
Matrix<T, M, N> outer_product(const Vector<T, M>& u, const Vector<T, N>& v);
}
```

#### Tensor (`Math/Tensor.h`)

Rank-2 and rank-4 tensors for continuum mechanics.

```cpp
namespace svmp::FE::math {

template<typename T, std::size_t Dim>
class Tensor2 {
public:
    // Element access
    T& operator()(std::size_t i, std::size_t j);
    const T& operator()(std::size_t i, std::size_t j) const;

    // Conversion
    static Tensor2 from_matrix(const Matrix<T, Dim, Dim>& mat);
    Matrix<T, Dim, Dim> to_matrix() const;

    // Operations
    Tensor2 transpose() const;
    Tensor2 symmetric_part() const;     // (A + A^T)/2
    Tensor2 antisymmetric_part() const; // (A - A^T)/2
    Tensor2 deviatoric() const;         // A - (tr(A)/dim)I
    Tensor2 inverse() const;

    // Scalar invariants
    T trace() const;                    // I1
    T determinant() const;              // I3
    T first_invariant() const;
    T second_invariant() const;
    T third_invariant() const;
    T von_mises() const;
    T frobenius_norm() const;

    // Contractions
    T double_contract(const Tensor2& other) const;  // A:B
    Vector<T, Dim> contract(const Vector<T, Dim>& v) const;  // A·v

    // Eigenanalysis (symmetric tensors)
    std::array<T, Dim> eigenvalues() const;
    std::pair<std::array<T, Dim>, Matrix<T, Dim, Dim>> eigen_decomposition() const;

    // Static factories
    static Tensor2 identity();
    static Tensor2 zero();
    static Tensor2 dyad(const Vector<T, Dim>& u, const Vector<T, Dim>& v);
};

template<typename T, std::size_t Dim>
class Tensor4 {
public:
    // Element access
    T& operator()(std::size_t i, std::size_t j, std::size_t k, std::size_t l);

    // Contraction with rank-2 tensor: σ = C:ε
    Tensor2<T, Dim> double_contract(const Tensor2<T, Dim>& strain) const;

    // Symmetry operations
    void apply_minor_symmetries();      // C_ijkl = C_jikl = C_ijlk
    void apply_major_symmetry();        // C_ijkl = C_klij

    // Static factories
    static Tensor4 identity();
    static Tensor4 symmetric_identity();
    static Tensor4 isotropic(T lambda, T mu);
    static Tensor4 transversely_isotropic(T C11, T C12, T C13, T C33, T C44);
};

// Type aliases
using Tensor2D = Tensor2<double, 2>;
using Tensor3D = Tensor2<double, 3>;
using Tensor4_2D = Tensor4<double, 2>;
using Tensor4_3D = Tensor4<double, 3>;
}
```

#### Additional Math Utilities

| File | Description |
|------|-------------|
| `MathConstants.h` | `epsilon<T>`, `tolerance<T>`, `Constants<T>::pi`, etc. |
| `MathUtils.h` | `approx_zero()`, `approx_equal()`, `square()`, `cube()`, `clamp()` |
| `Eigensolvers.h` | Analytical 2×2/3×3 symmetric eigensolvers, `SymmetricEigen3x3` class |
| `LU.h` | LU decomposition for small dense matrices |
| `Rotations.h` | Quaternion, axis-angle, Euler angle conversions |
| `VoigtNotation.h` | Voigt-to-tensor and tensor-to-Voigt mappings |
| `SIMD.h` | SIMD intrinsic wrappers (AVX2/AVX-512) |
| `Interpolation.h` | Lagrange polynomial interpolation |

---

### 3. Quadrature (`FE/Quadrature/`)

**Purpose:** Numerical integration rules on reference elements.

#### QuadratureRule (`Quadrature/QuadratureRule.h`)

Base class for all quadrature rules.

```cpp
namespace svmp::FE::quadrature {

using QuadPoint = math::Vector<Real, 3>;

class QuadratureRule {
public:
    // Query
    std::size_t num_points() const;
    int order() const;                      // Polynomial exactness
    int dimension() const;                  // 1D/2D/3D
    CellFamily cell_family() const;         // Line, Tri, Quad, Tet, Hex, ...

    // Point/weight access
    QuadPoint point(std::size_t i) const;
    Real weight(std::size_t i) const;
    const std::vector<QuadPoint>& points() const;
    const std::vector<Real>& weights() const;

    // Validation
    bool is_valid(Real tol = 1e-12) const;
    Real reference_measure() const;         // Volume of reference element

protected:
    QuadratureRule(CellFamily family, int dimension, int order);
    void set_data(std::vector<QuadPoint> pts, std::vector<Real> wts);
};
}
```

#### Quadrature Implementations

| Class | Description |
|-------|-------------|
| `GaussQuadrature` | 1D Gauss-Legendre rules |
| `GaussLobattoQuadrature` | 1D Gauss-Lobatto rules (includes endpoints) |
| `TriangleQuadrature` | Dunavant/symmetric triangle rules |
| `SymmetricTriangleQuadrature` | High-order symmetric triangle rules |
| `QuadrilateralQuadrature` | Tensor-product Gauss rules on [-1,1]² |
| `TetrahedronQuadrature` | Simplex tetrahedral rules |
| `SymmetricTetrahedronQuadrature` | High-order symmetric tet rules |
| `HexahedronQuadrature` | Tensor-product rules on [-1,1]³ |
| `WedgeQuadrature` | Prism (wedge) element rules |
| `PyramidQuadrature` | Pyramid element rules |
| `SurfaceQuadrature` | Facet integration rules |
| `CompositeQuadrature` | Combine rules for multi-region integration |
| `AdaptiveQuadrature` | Error-controlled adaptive integration |
| `SingularQuadrature` | Rules for singular integrands |

#### QuadratureFactory (`Quadrature/QuadratureFactory.h`)

```cpp
namespace svmp::FE::quadrature {

class QuadratureFactory {
public:
    static std::shared_ptr<QuadratureRule> create(CellFamily family, int order);
    static std::shared_ptr<QuadratureRule> create(ElementType type, int order);
    static std::shared_ptr<QuadratureRule> createForFacet(ElementType cell_type,
                                                          LocalIndex local_face,
                                                          int order);
};
}
```

---

### 4. Basis (`FE/Basis/`)

**Purpose:** Shape function evaluation on reference elements.

#### BasisFunction (`Basis/BasisFunction.h`)

Abstract interface for basis function families.

```cpp
namespace svmp::FE::basis {

using Gradient = math::Vector<Real, 3>;
using Hessian = math::Matrix<Real, 3, 3>;

class BasisFunction {
public:
    // Metadata
    virtual BasisType basis_type() const = 0;
    virtual ElementType element_type() const = 0;
    virtual int dimension() const = 0;
    virtual int order() const = 0;
    virtual std::size_t size() const = 0;
    virtual bool is_vector_valued() const;  // H(div)/H(curl)

    // Scalar basis evaluation
    virtual void evaluate_values(const math::Vector<Real, 3>& xi,
                                 std::vector<Real>& values) const = 0;
    virtual void evaluate_gradients(const math::Vector<Real, 3>& xi,
                                    std::vector<Gradient>& gradients) const;
    virtual void evaluate_hessians(const math::Vector<Real, 3>& xi,
                                   std::vector<Hessian>& hessians) const;

    // Vector basis evaluation (H(div)/H(curl))
    virtual void evaluate_vector_values(const math::Vector<Real, 3>& xi,
                                        std::vector<math::Vector<Real, 3>>& values) const;
    virtual void evaluate_divergence(const math::Vector<Real, 3>& xi,
                                     std::vector<Real>& divergence) const;
    virtual void evaluate_curl(const math::Vector<Real, 3>& xi,
                               std::vector<math::Vector<Real, 3>>& curl) const;
};
}
```

#### Basis Implementations

**Scalar Bases (H¹/L²):**

| Class | Description |
|-------|-------------|
| `LagrangeBasis` | Nodal Lagrange polynomials (P1, P2, Q1, Q2, ...) |
| `SpectralBasis` | Gauss-Lobatto-Legendre spectral element basis |
| `HierarchicalBasis` | Hierarchical (modal) polynomials |
| `BernsteinBasis` | Bernstein polynomials (Bézier) |
| `BSplineBasis` | B-spline basis functions |
| `SerendipityBasis` | Serendipity elements (no interior nodes) |
| `HermiteBasis` | C¹-continuous Hermite elements |
| `TensorBasis` | Tensor-product basis from 1D bases |

**Vector Bases (H(div)/H(curl)):**

| Class | Description |
|-------|-------------|
| `RaviartThomasBasis` | H(div) basis (flux DOFs on faces) |
| `NedelecBasis` | H(curl) basis (circulation DOFs on edges) |
| `BDMBasis` | Brezzi-Douglas-Marini H(div) basis |

#### BasisFactory (`Basis/BasisFactory.h`)

```cpp
namespace svmp::FE::basis {

class BasisFactory {
public:
    static std::shared_ptr<BasisFunction> create(BasisType type,
                                                  ElementType element,
                                                  int order);
    static std::shared_ptr<BasisFunction> createLagrange(ElementType element, int order);
    static std::shared_ptr<BasisFunction> createSpectral(ElementType element, int order);
};
}
```

#### Supporting Classes

| File | Description |
|------|-------------|
| `OrthogonalPolynomials.h` | Legendre, Jacobi, Chebyshev polynomials |
| `BatchEvaluator.h` | SIMD-batched basis evaluation |
| `BasisCache.h` | Cached basis values at quadrature points |
| `NodeOrderingConventions.h` | Node ordering for various element types |
| `ModalTransform.h` | Nodal ↔ modal transformations |

---

### 5. Elements (`FE/Elements/`)

**Purpose:** Reference finite element definitions combining basis + quadrature.

#### Element (`Elements/Element.h`)

Abstract base class for finite elements.

```cpp
namespace svmp::FE::elements {

struct ElementInfo {
    ElementType element_type;
    FieldType field_type;      // Scalar, Vector, Tensor
    Continuity continuity;     // C0, C1, DG
    int order;
};

class Element {
public:
    // Metadata
    virtual ElementInfo info() const = 0;
    ElementType element_type() const;
    FieldType field_type() const;
    Continuity continuity() const;
    int polynomial_order() const;
    virtual int dimension() const = 0;

    // DOF/node counts
    virtual std::size_t num_dofs() const = 0;
    virtual std::size_t num_nodes() const = 0;

    // Components
    virtual const basis::BasisFunction& basis() const = 0;
    virtual std::shared_ptr<const basis::BasisFunction> basis_ptr() const = 0;
    virtual std::shared_ptr<const quadrature::QuadratureRule> quadrature() const = 0;
};
}
```

#### Element Implementations

| Class | Description |
|-------|-------------|
| `LagrangeElement` | Standard Lagrange finite elements |
| `SpectralElement` | Spectral elements with GLL nodes |
| `DiscontinuousElement` | DG elements (no inter-element continuity) |
| `IsogeometricElement` | NURBS-based isogeometric elements |
| `ReferenceElement` | Generic reference element wrapper |

#### ElementFactory (`Elements/ElementFactory.h`)

```cpp
namespace svmp::FE::elements {

class ElementFactory {
public:
    static std::shared_ptr<Element> create(ElementType type,
                                            BasisType basis,
                                            int order,
                                            int quadrature_order = -1);
    static std::shared_ptr<Element> createLagrange(ElementType type, int order);
    static std::shared_ptr<Element> createDG(ElementType type, int order);
    static std::shared_ptr<Element> createSpectral(ElementType type, int order);
};
}
```

---

### 6. Geometry (`FE/Geometry/`)

**Purpose:** Reference-to-physical coordinate mappings.

#### GeometryMapping (Base Interface)

```cpp
namespace svmp::FE::geometry {

class GeometryMapping {
public:
    virtual ElementType element_type() const = 0;
    virtual int dimension() const = 0;
    virtual std::size_t num_nodes() const = 0;
    virtual const std::vector<math::Vector<Real, 3>>& nodes() const = 0;

    // Forward mapping: reference → physical
    virtual math::Vector<Real, 3> map_to_physical(const math::Vector<Real, 3>& xi) const = 0;

    // Inverse mapping: physical → reference
    virtual math::Vector<Real, 3> map_to_reference(const math::Vector<Real, 3>& x_phys,
                                                    const math::Vector<Real, 3>& initial_guess = {}) const = 0;

    // Jacobian ∂x/∂ξ
    virtual math::Matrix<Real, 3, 3> jacobian(const math::Vector<Real, 3>& xi) const = 0;
};
}
```

#### Mapping Implementations

| Class | Description |
|-------|-------------|
| `LinearMapping` | Affine (P1/Q1) geometric mapping |
| `IsoparametricMapping` | Isoparametric (full-order basis) mapping |
| `SubparametricMapping` | Geometry order < solution order |
| `SuperparametricMapping` | Geometry order > solution order |

#### Supporting Classes

| File | Description |
|------|-------------|
| `PushForward.h` | Piola transforms for H(div)/H(curl) |
| `JacobianCache.h` | Cached Jacobian evaluations |
| `SurfaceGeometry.h` | Surface normals, tangents, curvature |
| `MetricTensor.h` | Covariant/contravariant metric tensors |
| `InverseMapping.h` | Newton iteration for inverse mapping |
| `GeometryValidator.h` | Jacobian positivity checks |
| `MappingFactory.h` | Factory for creating mappings |

---

### 7. Spaces (`FE/Spaces/`)

**Purpose:** Function space abstractions (H¹, L², H(div), H(curl)).

#### FunctionSpace (Base Interface)

```cpp
namespace svmp::FE::spaces {

enum class SpaceType : std::uint8_t {
    H1,           // C⁰ continuous (standard Lagrange)
    C1,           // C¹ continuous (Hermite, thin shells)
    L2,           // Discontinuous (DG)
    HCurl,        // H(curl) conforming (Nedelec, edge elements)
    HDiv,         // H(div) conforming (Raviart-Thomas, BDM)
    Mixed,        // Heterogeneous fields (e.g., velocity + pressure)
    Product,      // Cartesian product (e.g., vector = scalar³)
    Trace,        // Restriction to boundary
    Mortar,       // Lagrange multiplier space
    Composite,    // Composite of arbitrary spaces
    Enriched,     // XFEM-style enriched
    Adaptive,     // p-adaptive refinement
    Isogeometric  // NURBS-based
};

class FunctionSpace {
public:
    // Type information
    virtual SpaceType space_type() const = 0;
    virtual FieldType field_type() const = 0;     // Scalar, Vector, Tensor
    virtual Continuity continuity() const = 0;    // C0, C1, L2, H_div, H_curl

    // Dimensions
    virtual int value_dimension() const = 0;       // 1 for scalar, 3 for vector
    virtual int topological_dimension() const = 0; // 1D, 2D, 3D
    virtual int polynomial_order() const = 0;
    virtual ElementType element_type() const = 0;

    // Element access
    virtual const elements::Element& element() const = 0;
    virtual std::shared_ptr<const elements::Element> element_ptr() const = 0;
    virtual std::size_t dofs_per_element() const = 0;

    // Field evaluation
    virtual void interpolate(std::function<Real(Real, Real, Real)> f,
                             std::vector<Real>& coefficients) const;
    virtual math::Vector<Real, 3> evaluate(const math::Vector<Real, 3>& xi,
                                           std::span<const Real> coefficients) const;
    virtual math::Vector<Real, 3> evaluate_gradient(const math::Vector<Real, 3>& xi,
                                                    std::span<const Real> coefficients) const;
};
}
```

#### Space Implementations

| Class | Description |
|-------|-------------|
| `H1Space` | Standard H¹-conforming (C⁰) scalar space (Lagrange basis) |
| `L2Space` | L²-conforming discontinuous space (DG basis) |
| `C1Space` | C¹-continuous (Hermite-based, for thin shells) |
| `HCurlSpace` | H(curl) conforming for electromagnetic fields (Nedelec basis) |
| `HDivSpace` | H(div) conforming for fluxes (Raviart-Thomas/BDM basis) |
| `ProductSpace` | Cartesian product (e.g., vector = 3 × H¹ scalar) |
| `MixedSpace` | Heterogeneous fields (e.g., velocity + pressure) |
| `CompositeSpace` | Composite of arbitrary spaces |
| `TraceSpace` | Restriction to boundary/face |
| `MortarSpace` | Lagrange multiplier space for mortar methods |
| `EnrichedSpace` | Enriched with bubble functions, partition of unity |
| `AdaptiveSpace` | p-adaptive (variable polynomial order) |
| `IsogeometricSpace` | NURBS/IGA space |

#### Supporting Classes

| File | Description |
|------|-------------|
| `SpaceInterpolation.h` | Interpolation operators between spaces |
| `SpaceCompatibility.h` | Compatibility checks for mixed formulations |
| `VectorComponentExtractor.h` | Extract scalar components from vector spaces |
| `DGOperators.h` | Jump/average operators for DG spaces |
| `OrientationManager.h` | Edge/face orientation for H(curl)/H(div) |
| `SpaceCache.h` | Cached space evaluations |
| `SpaceWorkspace.h` | Thread-local workspace for evaluations |

---

### 8. Dofs (`FE/Dofs/`)

**Purpose:** Degree-of-freedom distribution, numbering, and management.

#### DofHandler (`Dofs/DofHandler.h`)

High-level DOF management interface.

```cpp
namespace svmp::FE::dofs {

enum class DofNumberingStrategy : std::uint8_t {
    Sequential,    // Mesh traversal order
    Interleaved,   // Interleave vector components
    Block,         // Block by component
    Hierarchical   // Vertex, edge, face, cell DOFs
};

enum class OwnershipStrategy : std::uint8_t {
    LowestRank,    // Lowest MPI rank owns shared DOFs
    HighestRank,
    CellOwner,
    VertexGID
};

struct DofDistributionOptions {
    DofNumberingStrategy numbering{DofNumberingStrategy::Sequential};
    OwnershipStrategy ownership{OwnershipStrategy::LowestRank};
    GlobalNumberingMode global_numbering{GlobalNumberingMode::OwnerContiguous};
    bool use_canonical_ordering{true};
    int my_rank{0};
    int world_size{1};
#if FE_HAS_MPI
    MPI_Comm mpi_comm{MPI_COMM_WORLD};
#endif
};

struct MeshTopologyInfo {
    GlobalIndex n_cells, n_vertices, n_edges, n_faces;
    int dim;
    std::vector<MeshOffset> cell2vertex_offsets;
    std::vector<MeshIndex> cell2vertex_data;
    std::vector<gid_t> vertex_gids;
    // ... optional edge/face connectivity
};

struct DofLayoutInfo {
    LocalIndex dofs_per_vertex{0};
    LocalIndex dofs_per_edge{0};
    LocalIndex dofs_per_face{0};
    LocalIndex dofs_per_cell{0};
    int num_components{1};
    bool is_continuous{true};

    static DofLayoutInfo Lagrange(int order, int dim, int num_verts_per_cell, int num_components = 1);
    static DofLayoutInfo DG(int order, int num_verts_per_cell, int num_components = 1);
};

class DofHandler {
public:
    // DOF distribution
    void distributeDofs(const MeshTopologyInfo& topology,
                        const DofLayoutInfo& layout,
                        const DofDistributionOptions& options = {});
    void distributeDofs(const MeshTopologyInfo& topology,
                        const spaces::FunctionSpace& space,
                        const DofDistributionOptions& options = {});
    void distributeDofs(const MeshBase& mesh,
                        const spaces::FunctionSpace& space,
                        const DofDistributionOptions& options = {});

    // Finalization
    void finalize();
    bool isFinalized() const;

    // Query
    const DofMap& getDofMap() const;
    const DofPartition& getPartition() const;
    GlobalIndex getNumDofs() const;
    GlobalIndex getNumLocalDofs() const;
    std::span<const GlobalIndex> getCellDofs(GlobalIndex cell_id) const;
    std::span<const GlobalIndex> getGhostDofs() const;

    // Renumbering
    void renumberDofs(DofNumberingStrategy strategy);

    // Parallel ghost exchange
#if FE_HAS_MPI
    void syncGhostValuesMPI(std::span<const double> owned_values,
                            std::span<double> ghost_values);
#endif
};
}
```

#### DofMap (`Dofs/DofMap.h`)

Cell-to-DOF mapping storage.

```cpp
namespace svmp::FE::dofs {

class DofMap {
public:
    std::span<const GlobalIndex> getCellDofs(GlobalIndex cell_id) const;
    GlobalIndex getNumDofs() const;
    GlobalIndex getNumLocalDofs() const;
    GlobalIndex getNumCells() const;

    void setCellDofs(GlobalIndex cell_id, std::span<const GlobalIndex> dofs);
};
}
```

#### Supporting Classes

| File | Description |
|------|-------------|
| `DofIndexSet.h` | `DofPartition` with owned/ghost/relevant sets |
| `EntityDofMap.h` | DOFs per mesh entity (vertex, edge, face, cell) |
| `DofNumbering.h` | Renumbering algorithms (RCM, Hilbert, etc.) |
| `DofGraph.h` | DOF adjacency graph for sparsity |
| `DofConstraints.h` | Linear constraints on DOFs |
| `BlockDofMap.h` | Block DOF maps for multi-field systems |
| `FieldDofMap.h` | Field-wise DOF offset tracking |
| `GhostDofManager.h` | Ghost DOF communication patterns |
| `ConstrainedAssembly.h` | Constrained DOF assembly helpers |
| `SubspaceView.h` | View into a subset of DOFs |
| `DofTools.h` | DOF-related utilities |

---

### 9. Sparsity (`FE/Sparsity/`)

**Purpose:** Sparsity pattern construction for sparse matrices.

#### SparsityPattern

```cpp
namespace svmp::FE::sparsity {

enum class CouplingMode : std::uint8_t {
    Full,           // All fields couple to all fields
    BlockDiagonal,  // Only same-field coupling
    Custom          // User-specified coupling
};

struct SparsityBuildOptions {
    CouplingMode coupling_mode{CouplingMode::Full};
    bool add_diagonal{true};
    bool add_off_diagonal{true};
};

class SparsityPattern {
public:
    GlobalIndex numRows() const;
    GlobalIndex numCols() const;
    GlobalIndex numNonzeros() const;

    std::span<const GlobalIndex> rowPointers() const;  // CSR row_ptr
    std::span<const GlobalIndex> columnIndices() const; // CSR col_ind

    bool hasEntry(GlobalIndex row, GlobalIndex col) const;
};

class SparsityBuilder {
public:
    void buildFromDofMap(const dofs::DofMap& dof_map,
                         const SparsityBuildOptions& options = {});
    void buildFromDofHandler(const dofs::DofHandler& dof_handler,
                             const SparsityBuildOptions& options = {});
    void addConnection(GlobalIndex row, GlobalIndex col);
    void finalize();

    SparsityPattern getPattern() const;
};
}
```

---

### 10. Assembly (`FE/Assembly/`)

**Purpose:** Element-to-global matrix/vector assembly.

#### Assembler (`Assembly/Assembler.h`)

Abstract base class for assembly strategies.

```cpp
namespace svmp::FE::assembly {

enum class AssemblyMode : std::uint8_t {
    Add,       // Add to existing entries
    Insert,    // Replace entries
    Flush      // Force communication
};

enum class ThreadingStrategy : std::uint8_t {
    Sequential,  // Single-threaded
    Colored,     // Graph coloring for parallel
    WorkStream,  // Task pipeline
    Atomic       // Atomic operations
};

struct AssemblyOptions {
    ThreadingStrategy threading{ThreadingStrategy::Sequential};
    int num_threads{1};
    bool deterministic{true};
    bool use_batching{false};
    bool check_sparsity{false};
};

struct AssemblyResult {
    bool success{true};
    std::string error_message;
    GlobalIndex elements_assembled{0};
    GlobalIndex boundary_faces_assembled{0};
    double elapsed_time_seconds{0.0};
};

class Assembler {
public:
    // Configuration
    virtual void setDofMap(const dofs::DofMap& dof_map) = 0;
    virtual void setDofHandler(const dofs::DofHandler& dof_handler) = 0;
    virtual void setConstraints(const constraints::AffineConstraints* constraints) = 0;
    virtual void setSparsityPattern(const sparsity::SparsityPattern* sparsity) = 0;
    virtual void setOptions(const AssemblyOptions& options) = 0;

    // Solution data (for nonlinear problems)
    virtual void setCurrentSolution(std::span<const Real> solution);
    virtual void setPreviousSolution(std::span<const Real> solution);
    virtual void setTime(Real time);
    virtual void setTimeStep(Real dt);

    // Assembly operations
    virtual AssemblyResult assembleMatrix(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView& matrix_view) = 0;

    virtual AssemblyResult assembleVector(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& space,
        AssemblyKernel& kernel,
        GlobalSystemView& vector_view) = 0;

    virtual AssemblyResult assembleBoth(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView& matrix_view,
        GlobalSystemView& vector_view) = 0;

    virtual AssemblyResult assembleBoundaryFaces(
        const IMeshAccess& mesh,
        int boundary_marker,
        const spaces::FunctionSpace& space,
        AssemblyKernel& kernel,
        GlobalSystemView* matrix_view,
        GlobalSystemView* vector_view) = 0;

    virtual AssemblyResult assembleInteriorFaces(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView& matrix_view,
        GlobalSystemView* vector_view) = 0;

    // Lifecycle
    virtual void initialize() = 0;
    virtual void finalize(GlobalSystemView* matrix_view, GlobalSystemView* vector_view) = 0;
    virtual void reset() = 0;

    // Query
    virtual std::string name() const = 0;
    virtual bool supportsRectangular() const;
    virtual bool supportsDG() const;
    virtual bool isThreadSafe() const;
};

// Factory
std::unique_ptr<Assembler> createAssembler(ThreadingStrategy strategy);
std::unique_ptr<Assembler> createAssembler(const AssemblyOptions& options);
}
```

#### IMeshAccess (`Assembly/MeshAccess.h`)

Mesh-independent iteration interface.

```cpp
namespace svmp::FE::assembly {

class IMeshAccess {
public:
    virtual GlobalIndex numCells() const = 0;
    virtual GlobalIndex numOwnedCells() const = 0;
    virtual GlobalIndex numBoundaryFaces() const = 0;
    virtual GlobalIndex numInteriorFaces() const = 0;
    virtual int dimension() const = 0;

    virtual bool isOwnedCell(GlobalIndex cell_id) const = 0;
    virtual ElementType getCellType(GlobalIndex cell_id) const = 0;
    virtual int getCellDomainId(GlobalIndex cell_id) const;

    virtual void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const = 0;
    virtual std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const = 0;
    virtual void getCellCoordinates(GlobalIndex cell_id,
                                    std::vector<std::array<Real, 3>>& coords) const = 0;

    virtual int getBoundaryFaceMarker(GlobalIndex face_id) const = 0;
    virtual LocalIndex getLocalFaceIndex(GlobalIndex face_id, GlobalIndex cell_id) const = 0;
    virtual std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex face_id) const = 0;

    virtual void forEachCell(std::function<void(GlobalIndex)> callback) const = 0;
    virtual void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const = 0;
    virtual void forEachBoundaryFace(int marker,
                                     std::function<void(GlobalIndex, GlobalIndex)> callback) const = 0;
    virtual void forEachInteriorFace(
        std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> callback) const = 0;
};
}
```

#### AssemblyKernel (`Assembly/AssemblyKernel.h`)

Element-level computation interface (physics kernels).

```cpp
namespace svmp::FE::assembly {

class AssemblyKernel {
public:
    // What data the kernel needs
    virtual RequiredData getRequiredData() const = 0;

    // Element matrix/vector computation
    virtual void computeElement(const AssemblyContext& ctx,
                                std::span<Real> local_matrix,
                                std::span<Real> local_vector) = 0;

    // Boundary face computation
    virtual void computeBoundaryFace(const AssemblyContext& ctx,
                                     std::span<Real> local_matrix,
                                     std::span<Real> local_vector);

    // Interior face computation (DG)
    virtual void computeInteriorFace(const AssemblyContext& ctx,
                                     std::span<Real> local_matrix,
                                     std::span<Real> local_vector);
};
}
```

#### AssemblyContext (`Assembly/AssemblyContext.h`)

Read-only element context passed to kernels.

```cpp
namespace svmp::FE::assembly {

class AssemblyContext {
public:
    // Cell information
    GlobalIndex cellId() const;
    ElementType elementType() const;
    int dimension() const;

    // Quadrature
    LocalIndex numQuadPoints() const;
    Real quadWeight(LocalIndex q) const;
    math::Vector<Real, 3> quadPoint(LocalIndex q) const;  // Reference coords
    math::Vector<Real, 3> physicalPoint(LocalIndex q) const;

    // Geometry
    const math::Matrix<Real, 3, 3>& jacobian(LocalIndex q) const;
    const math::Matrix<Real, 3, 3>& jacobianInverse(LocalIndex q) const;
    Real jacobianDeterminant(LocalIndex q) const;

    // Basis values at quadrature points
    std::span<const Real> basisValues(LocalIndex q) const;
    std::span<const math::Vector<Real, 3>> basisGradients(LocalIndex q) const;
    std::span<const math::Matrix<Real, 3, 3>> basisHessians(LocalIndex q) const;

    // Solution data (if available)
    std::span<const Real> solutionCoefficients() const;
    Real solutionValue(LocalIndex q) const;
    math::Vector<Real, 3> solutionGradient(LocalIndex q) const;

    // Time data
    Real time() const;
    Real timeStep() const;

    // Node coordinates
    std::span<const std::array<Real, 3>> nodeCoordinates() const;

    // Boundary face info
    LocalIndex localFaceIndex() const;
    int boundaryMarker() const;
    math::Vector<Real, 3> normal(LocalIndex q) const;
};
}
```

#### Assembler Architecture

The assembly system uses a **decorator pattern** for composable assembly strategies. Assemblers are organized into three categories:

##### Base Assemblers

Base assemblers implement the core `Assembler` interface and provide fundamental assembly strategies:

| Class | Description | Thread Safety |
|-------|-------------|---------------|
| `StandardAssembler` | Traditional sequential element-by-element assembly | Single-threaded |
| `ParallelAssembler` | MPI distributed-memory parallel assembly | MPI-parallel |
| `WorkStreamAssembler` | Task-based pipelined assembly (cell/copy/assemble stages) | Thread-safe |
| `DeviceAssembler` | GPU-accelerated assembly (CUDA/HIP) | Device-parallel |
| `SymbolicAssembler` | Forms-based symbolic processing and compilation | Single-threaded |

##### Decorator Assemblers

Decorators wrap a base assembler (or another decorator) to add capabilities. They inherit from `DecoratorAssembler` which forwards all interface methods to the wrapped assembler:

| Class | Description | Composition Use |
|-------|-------------|-----------------|
| `ScheduledAssembler` | Element traversal reordering (Hilbert, Morton, RCM, cache-blocked) | Wrap any base |
| `ColoredAssembler` | Graph-colored parallel assembly for race-free threading | Wrap any base |
| `CachedAssembler` | Caches element matrices/geometry factors for repeated assembly | Wrap any base |
| `VectorizedAssembler` | SIMD batching framework for vectorized element assembly | Wrap any base |

The `AssemblerFactory` composes decorators automatically based on options:
```cpp
// Factory creates composition stack based on options:
// Vectorized(Cached(Colored(Scheduled(StandardAssembler))))
auto assembler = AssemblerFactory::create(options);
```

**Decorator Composition Stack** (bottom to top):
```
┌────────────────────────────────┐
│  VectorizedAssembler (optional)│  SIMD batching
├────────────────────────────────┤
│  CachedAssembler (optional)    │  Element matrix caching
├────────────────────────────────┤
│  ColoredAssembler (optional)   │  Thread-parallel graph coloring
├────────────────────────────────┤
│  ScheduledAssembler (optional) │  Traversal reordering
├────────────────────────────────┤
│  Base Assembler (required)     │  Standard/Parallel/WorkStream/etc.
└────────────────────────────────┘
```

##### Standalone Classes

These classes do not inherit from `Assembler` and have specialized interfaces:

| Class | Description | Why Standalone |
|-------|-------------|----------------|
| `BlockAssembler` | Multi-field block-structured assembly for coupled systems | Orchestrates multiple AssemblyKernels per block |
| `MatrixFreeAssembler` | Matrix-free operator application | Returns action y = A*x, not matrix entries |
| `FunctionalAssembler` | Scalar functional evaluation (norms, QoIs, energy) | Returns scalar, not matrix/vector |

**Threading Notes:**
- `ColoredAssembler` provides shared-memory thread parallelism via graph coloring (elements of the same color have disjoint DOF stencils). Colors are processed sequentially with a barrier; elements within each color are parallelized.
- `ParallelAssembler` provides distributed-memory MPI parallelism.
- These strategies are orthogonal and can be composed (e.g., `ColoredAssembler` wrapping a base that handles local assembly for MPI).

#### FunctionalAssembler

`FunctionalAssembler` computes scalar quantities by integrating over the domain or boundary. Common use cases:

| Use Case | Example |
|----------|---------|
| **Solution norms** | L² norm, H¹ seminorm, energy norm |
| **Engineering QoIs** | Drag/lift coefficients, heat flux, mass flow rate |
| **Energy functionals** | Total kinetic/potential energy, strain energy |
| **Conservation checks** | Mass conservation error, momentum balance |
| **Goal-oriented estimation** | Adjoint-weighted residual for error estimation |
| **Mesh quality metrics** | Minimum Jacobian determinant, aspect ratio |
| **Wall shear stress** | Boundary integral of τ·n for cardiovascular flows |

```cpp
// Example: Compute L² error norm
assembly::FunctionalAssembler functional_assembler;
functional_assembler.setDofHandler(dof_handler);

// Kernel computes |u - u_exact|² at each quadrature point
auto l2_kernel = std::make_shared<L2ErrorKernel>(exact_solution);
Real l2_error_squared = functional_assembler.evaluateFunctional(
    mesh_access, space, *l2_kernel);
Real l2_error = std::sqrt(l2_error_squared);

// Example: Compute drag coefficient on boundary
auto drag_kernel = std::make_shared<DragKernel>(velocity, pressure, mu);
Real drag = functional_assembler.evaluateBoundaryFunctional(
    mesh_access, /*boundary_marker=*/2, space, *drag_kernel);
```

#### BlockAssembler

`BlockAssembler` orchestrates assembly for multi-field coupled systems where different physics may require different assembly strategies per block:

```cpp
// Example: Fluid-structure interaction with different element types
BlockAssembler block_assembler;

// Configure block structure
BlockSystemConfig config;
config.num_row_blocks = 2;  // Fluid + Solid
config.num_col_blocks = 2;
config.fields = {
    {"fluid_velocity", fluid_space_ptr, 3},
    {"solid_displacement", solid_space_ptr, 3}
};

// Each block can use different assemblers/kernels
block_assembler.setBlockKernel(0, 0, fluid_kernel);    // Fluid-fluid
block_assembler.setBlockKernel(0, 1, coupling_kernel); // Fluid-solid
block_assembler.setBlockKernel(1, 0, coupling_kernel); // Solid-fluid
block_assembler.setBlockKernel(1, 1, solid_kernel);    // Solid-solid

block_assembler.setup(config);
block_assembler.assembleAll(mesh_access, state, matrix_view, vector_view);
```

---

### 11. Forms (`FE/Forms/`)

**Purpose:** Symbolic weak-form expressions and JIT compilation.

#### FormExpr (`Forms/FormExpr.h`)

Symbolic expression AST for weak forms.

```cpp
namespace svmp::FE::forms {

// FormExprType has 112 expression types organized by category:
enum class FormExprType : std::uint16_t {
    // === Terminals ===
    TestFunction, TrialFunction, DiscreteField, StateField,
    Coefficient,                    // Scalar, vector, matrix, tensor3, tensor4
    ParameterSymbol, ParameterRef,  // Setup-time vs JIT-friendly parameters
    Constant,
    Coordinate, ReferenceCoordinate, Time, TimeStep, EffectiveTimeStep,
    Identity, Jacobian, JacobianInverse, JacobianDeterminant, Normal,
    CellDiameter, CellVolume, FacetArea, CellDomainId,

    // State/history terminals
    PreviousSolutionRef,            // u^{n-1}, u^{n-2}, etc.
    MaterialStateOldRef,            // Per-qpt material state (previous step)
    MaterialStateWorkRef,           // Per-qpt material state (current work)
    AuxiliaryStateSymbol, AuxiliaryStateRef,
    BoundaryFunctionalSymbol, BoundaryIntegralSymbol, BoundaryIntegralRef,

    // === Differential Operators ===
    Gradient, Divergence, Curl, Hessian, TimeDerivative,

    // === DG Operators ===
    RestrictMinus, RestrictPlus, Jump, Average,

    // === Algebra ===
    Negate, Add, Subtract, Multiply, Divide,
    InnerProduct, DoubleContraction, OuterProduct, CrossProduct,
    Power, Minimum, Maximum,

    // === Comparisons (return 0/1) ===
    Less, LessEqual, Greater, GreaterEqual, Equal, NotEqual,

    // === Control Flow ===
    Conditional,

    // === Tensor Operations ===
    Component, IndexedAccess, AsVector, AsTensor,
    Transpose, Trace, Determinant, Inverse, Cofactor,
    Deviator, SymmetricPart, SkewPart,
    Norm, Normalize, AbsoluteValue, Sign, Sqrt, Exp, Log,

    // === Constitutive ===
    Constitutive, ConstitutiveOutput,

    // === Integrals ===
    CellIntegral, BoundaryIntegral, InteriorFaceIntegral, InterfaceIntegral,

    // === Spectral/Eigendecomposition ===
    SymmetricEigenvalue, SymmetricEigenvector, SpectralDecomposition,
    SymmetricEigenvalueDirectionalDerivative,
    SymmetricEigenvalueDirectionalDerivativeWrtA,
    SymmetricEigenvectorDirectionalDerivative,
    SpectralDecompositionDirectionalDerivative,

    // === Matrix Functions (LLVM JIT) ===
    MatrixExponential, MatrixLogarithm, MatrixSqrt, MatrixPower,
    MatrixExponentialDirectionalDerivative,
    MatrixLogarithmDirectionalDerivative,
    MatrixSqrtDirectionalDerivative,
    MatrixPowerDirectionalDerivative,

    // === Smooth/Regularized Approximations ===
    SmoothHeaviside, SmoothAbsoluteValue, SmoothSign,
    SmoothMin, SmoothMax,

    // === History/Convolution ===
    HistoryWeightedSum, HistoryConvolution,

    // ... (112 total types)
};

class FormExpr {
public:
    // Terminal constructors
    static FormExpr testFunction(const spaces::FunctionSpace& space);
    static FormExpr trialFunction(const spaces::FunctionSpace& space);
    static FormExpr coefficient(ScalarCoefficient f);
    static FormExpr coefficient(VectorCoefficient f);
    static FormExpr coefficient(MatrixCoefficient f);
    static FormExpr constant(Real value);
    static FormExpr constant(const std::array<Real, 3>& vec);
    static FormExpr parameter(std::string_view name);

    // Geometry terminals
    static FormExpr x();           // Physical coordinate
    static FormExpr X();           // Reference coordinate
    static FormExpr n();           // Normal
    static FormExpr J();           // Jacobian
    static FormExpr detJ();        // Jacobian determinant
    static FormExpr h();           // Cell diameter

    // Arithmetic operators
    FormExpr operator+(const FormExpr& rhs) const;
    FormExpr operator-(const FormExpr& rhs) const;
    FormExpr operator*(const FormExpr& rhs) const;
    FormExpr operator/(const FormExpr& rhs) const;
    FormExpr operator-() const;

    // Differential operators (free functions)
    friend FormExpr grad(const FormExpr& u);
    friend FormExpr div(const FormExpr& u);
    friend FormExpr curl(const FormExpr& u);
    friend FormExpr hessian(const FormExpr& u);
    friend FormExpr laplacian(const FormExpr& u);  // trace(hessian(u))
    friend FormExpr dt(const FormExpr& u, int order = 1);

    // Tensor operations (free functions)
    friend FormExpr transpose(const FormExpr& A);
    friend FormExpr trace(const FormExpr& A);
    friend FormExpr det(const FormExpr& A);
    friend FormExpr inv(const FormExpr& A);
    friend FormExpr sym(const FormExpr& A);
    friend FormExpr skew(const FormExpr& A);
    friend FormExpr dev(const FormExpr& A);
    friend FormExpr norm(const FormExpr& v);
    friend FormExpr normalize(const FormExpr& v);
    friend FormExpr doubleContraction(const FormExpr& A, const FormExpr& B);
    friend FormExpr inner(const FormExpr& a, const FormExpr& b);
    friend FormExpr outer(const FormExpr& u, const FormExpr& v);
    friend FormExpr cross(const FormExpr& u, const FormExpr& v);

    // Component access
    FormExpr component(int i) const;
    FormExpr component(int i, int j) const;
    FormExpr operator()(const Index& i) const;
    FormExpr operator()(const Index& i, const Index& j) const;

    // DG operators
    FormExpr minus() const;
    FormExpr plus() const;
    friend FormExpr jump(const FormExpr& u);
    friend FormExpr avg(const FormExpr& u);

    // Control flow
    friend FormExpr conditional(const FormExpr& cond, const FormExpr& if_true, const FormExpr& if_false);

    // Constitutive hook
    friend FormExpr constitutive(const ConstitutiveModel& model, const FormExpr& input);

    // Integrals
    FormExpr dx() const;                    // Cell integral
    FormExpr ds(int boundary_marker) const; // Boundary integral
    FormExpr dS() const;                    // Interior face integral

    // Query
    FormExprType type() const;
    bool isTerminal() const;
    bool isLinear() const;
    bool containsTimeDerivative() const;
};
}
```

#### Form Compilation

Forms are compiled via `FormCompiler` with different `FormKind` semantics:

```cpp
namespace svmp::FE::forms {

enum class FormKind {
    Linear,     // L(v) - TrialFunction should not appear
    Bilinear,   // a(u,v) - TrialFunction interpreted as basis variation
    Residual    // F(u;v) - TrialFunction interpreted as current solution
};

class FormCompiler {
public:
    // Compile a linear form L(v)
    FormIR compileLinear(const FormExpr& form);

    // Compile a bilinear form a(u,v)
    FormIR compileBilinear(const FormExpr& form);

    // Compile a nonlinear residual F(u;v) with auto-differentiated Jacobian
    FormIR compileResidual(const FormExpr& form);

    // Block form compilation
    std::vector<std::optional<FormIR>>
    compileLinear(const BlockLinearForm& block_form);

    std::vector<std::vector<std::optional<FormIR>>>
    compileBilinear(const BlockBilinearForm& block_form);
};

struct SymbolicOptions {
    ADMode ad_mode{ADMode::None};       // None, Forward, Reverse, Taylor
    bool use_symbolic_tangent{false};
    JITOptions jit{};
    bool simplify_expressions{true};
    bool exploit_sparsity{true};
    bool cache_expressions{true};
    bool verbose{false};
};

// FormIR is the compiled intermediate representation
class FormIR {
public:
    FormKind kind() const;
    bool hasCell() const;
    bool hasBoundary() const;
    bool hasInteriorFace() const;
    int timeDerivativeOrder() const;

    // Create assembly kernel from compiled IR
    std::unique_ptr<assembly::AssemblyKernel>
    createKernel(const SymbolicOptions& options = {}) const;
};
}
```

**Semantic Differences:**
- **Linear**: `TrialFunction` should not appear; used for right-hand side assembly
- **Bilinear**: `TrialFunction` represents basis function variation; produces matrix entries
- **Residual**: `TrialFunction` represents current solution; auto-differentiation produces consistent Jacobian

#### ConstitutiveModel (`Forms/ConstitutiveModel.h`)

Material-point operator interface supporting single-input, N-ary, and multi-output evaluation.

```cpp
namespace svmp::FE::forms {

template<typename T>
class Value {
public:
    enum class Kind {
        Scalar, Vector, Matrix,
        SymmetricMatrix, SkewMatrix,  // Optimized storage
        Tensor3, Tensor4
    };

    Kind kind() const;
    T scalar() const;
    std::span<const T> vector() const;
    std::span<const T> matrix() const;
    std::span<const T> tensor3() const;
    std::span<const T> tensor4() const;

    // Shape metadata
    int vector_size() const;
    int matrix_rows() const;
    int matrix_cols() const;

    static Value fromScalar(T s);
    static Value fromVector(std::span<const T> v);
    static Value fromMatrix(std::span<const T> m, int rows, int cols);
    static Value fromSymmetricMatrix(std::span<const T> m, int dim);
};

struct ConstitutiveEvalContext {
    enum class Domain { Cell, BoundaryFace, InteriorFace };
    enum class TraceSide { None, Minus, Plus };

    Domain domain{Domain::Cell};
    TraceSide trace_side{TraceSide::None};
    int dim;
    std::array<Real, 3> x;
    Real integration_weight;
    Real time;
    Real dt;
    GlobalIndex cell_id;
    LocalIndex q;

    // Per-quadrature-point material state
    std::span<const std::byte> state_old;
    std::span<std::byte> state_work;

    // Parameter lookup
    std::optional<Real> realParam(std::string_view key) const;
    template<typename T> std::optional<T> paramAs(std::string_view key) const;
    template<typename T> T requireParamAs(std::string_view key) const;

    // Multi-field coupling (nonlocal access)
    std::function<Value<Real>(FieldId, LocalIndex)> field_value_callback;
};

class ConstitutiveModel {
public:
    // === Single-input evaluation ===
    virtual Value<Real> evaluate(const Value<Real>& input, int dim) const = 0;
    virtual Value<Real> evaluate(const Value<Real>& input,
                                 const ConstitutiveEvalContext& ctx) const;

    // Dual (AD) evaluation for consistent Jacobians
    virtual Value<Dual> evaluate(const Value<Dual>& input,
                                 int dim,
                                 DualWorkspace& workspace) const = 0;

    // === N-ary evaluation (multiple inputs) ===
    virtual Value<Real> evaluateNary(std::span<const Value<Real>> inputs,
                                     const ConstitutiveEvalContext& ctx) const;
    virtual Value<Dual> evaluateNary(std::span<const Value<Dual>> inputs,
                                     const ConstitutiveEvalContext& ctx,
                                     DualWorkspace& workspace) const;

    // === Multi-output support ===
    virtual std::size_t outputCount() const;  // Default: 1
    struct OutputSpec { Value<Real>::Kind kind; int dim1; int dim2; };
    virtual OutputSpec outputSpec(std::size_t output_index) const;
    virtual void evaluateNaryOutputs(std::span<const Value<Real>> inputs,
                                     const ConstitutiveEvalContext& ctx,
                                     std::span<Value<Real>> outputs) const;

    // === Metadata ===
    virtual std::optional<Value<Real>::Kind> expectedInputKind() const;
    virtual std::optional<std::size_t> expectedInputCount() const;

    // State requirements (per-qpt storage)
    struct StateSpec {
        std::size_t bytes_per_qpt{0};
        std::size_t alignment{alignof(std::max_align_t)};
    };
    virtual StateSpec stateSpec() const;
    virtual const StateLayout* stateLayout() const;  // Structured field metadata

    // Parameter specifications
    virtual std::vector<params::Spec> parameterSpecs() const;

    // JIT inlining hook
    virtual const InlinableConstitutiveModel* inlinable() const;
};
}
```

#### JIT Compilation (`Forms/JIT/`)

LLVM-based JIT compilation for weak forms. The JIT system includes ~24 files totaling significant infrastructure.

| File | Description |
|------|-------------|
| `JITCompiler.h` | High-level JIT compilation orchestration |
| `JITValidation.h` | Form compatibility checking for JIT |
| `KernelIR.h` | Intermediate representation for kernels |
| `LLVMGen.h` | LLVM IR code generation |
| `LLVMTensorGen.h` | Tensor loop emission for contractions |
| `ExternalCalls.h` | C ABI for non-inlinable constitutive models |
| `InlinableConstitutiveModel.h` | Interface for JIT-inlinable models |

```cpp
namespace svmp::FE::forms {

struct JITOptions {
    bool enable{false};
    int optimization_level{2};        // 0-3
    bool cache_kernels{true};
    bool vectorize{true};
    std::string cache_directory;
    std::size_t max_in_memory_kernels{0};  // 0 = unlimited
    bool dump_kernel_ir{false};
    bool dump_llvm_ir{false};
    bool dump_llvm_ir_optimized{false};

    TensorJITOptions tensor{};
    JITSpecializationOptions specialization{};
};

struct TensorJITOptions {
    TensorLoweringMode mode{TensorLoweringMode::Auto};
    bool enable_symmetry_lowering{true};      // Exploit symmetric tensor structure
    bool enable_optimal_contraction_order{true};  // Einstein contraction optimization
    bool enable_vectorization_hints{true};    // SIMD guidance
    bool enable_delta_shortcuts{true};        // Δ_{ij} optimizations
    int scalar_expansion_term_threshold{64};  // Fallback threshold
    int temp_stack_max_entries{81};           // Scratch allocation
    bool enable_polly{false};                 // Loop metadata for Polly optimizer
};

struct JITSpecializationOptions {
    bool enable{false};
    bool specialize_n_qpts{true};             // Specialize on quadrature count
    bool specialize_dofs{false};              // Specialize on DOF count
    std::uint32_t max_specialized_n_qpts{32};
    std::uint32_t max_specialized_dofs{64};
    std::size_t max_variants_per_kernel{8};
};

enum class TensorLoweringMode : std::uint8_t {
    Off,      // Default: scalar-term expansion
    Auto,     // Adaptive selection based on complexity
    On        // Force loop-nest lowering (TensorIR)
};

class JITCompiler {
public:
    static std::shared_ptr<JITCompiler> getOrCreate(const JITOptions& options);

    JITCompileResult compile(const FormIR& ir, const ValidationOptions& validation);

    JITCompileResult compileSpecialized(const FormIR& ir,
                                        const JITCompileSpecialization& spec,
                                        const ValidationOptions& validation);

    JITCompileResult compileFunctional(const FormExpr& integrand,
                                       IntegralDomain domain,
                                       const ValidationOptions& validation);

    JITCacheStats cacheStats() const;
    static bool canCompile(const FormIR& form, JITStrictness strictness);
};
}
```

#### Block Forms (`Forms/BlockForm.h`)

Multi-field block-structured forms.

```cpp
namespace svmp::FE::forms {

class BlockBilinearForm {
public:
    BlockBilinearForm(const std::vector<const spaces::FunctionSpace*>& test_spaces,
                      const std::vector<const spaces::FunctionSpace*>& trial_spaces);

    void setBlock(std::size_t i, std::size_t j, const FormExpr& expr);
    void setBlock(std::size_t i, std::size_t j, const BilinearForm& form);

    std::vector<std::vector<std::unique_ptr<assembly::AssemblyKernel>>>
    compile(const SymbolicOptions& options = {});
};

class BlockLinearForm {
public:
    explicit BlockLinearForm(const std::vector<const spaces::FunctionSpace*>& spaces);

    void setBlock(std::size_t i, const FormExpr& expr);

    std::vector<std::unique_ptr<assembly::AssemblyKernel>>
    compile(const SymbolicOptions& options = {});
};
}
```

#### Complex Forms (`Forms/Complex.h`)

Complex-valued form handling.

```cpp
namespace svmp::FE::forms {

class ComplexScalar {
public:
    ComplexScalar(const FormExpr& real, const FormExpr& imag);

    FormExpr real() const;
    FormExpr imag() const;
    ComplexScalar conj() const;
    FormExpr abs() const;
    FormExpr arg() const;
};

class ComplexBilinearForm {
public:
    ComplexBilinearForm(const spaces::FunctionSpace& test_space,
                        const spaces::FunctionSpace& trial_space);

    void setExpression(const ComplexScalar& expr);

    // Convert to 2×2 real block system [Re, -Im; Im, Re]
    BlockBilinearForm toRealBlock2x2() const;
};
}
```

#### Vocabulary (`Forms/Vocabulary.h`)

UFL-like convenience combinators and helper functions.

```cpp
namespace svmp::FE::forms {

// === Function Space Shortcuts ===
FormExpr TrialFunction(const FunctionSpace& V, std::string name = "u");
FormExpr TestFunction(const FunctionSpace& V, std::string name = "v");
std::vector<FormExpr> TrialFunctions(const MixedSpace& W);
std::vector<FormExpr> TestFunctions(const MixedSpace& W);

// === Geometry Shortcuts ===
FormExpr x();           // Physical coordinate
FormExpr X();           // Reference coordinate
FormExpr t();           // Time
FormExpr deltat();      // Time step
FormExpr deltat_eff();  // Effective time step (for multi-stage methods)
FormExpr J();           // Jacobian matrix
FormExpr Jinv();        // Jacobian inverse
FormExpr detJ();        // Jacobian determinant
FormExpr h();           // Cell diameter
FormExpr vol();         // Cell volume
FormExpr area();        // Facet area
FormExpr hNormal();     // Facet-normal height: 2*vol()/area()
FormExpr domainId();    // Cell domain ID (for multi-region problems)

// === Scalar Helpers ===
FormExpr heaviside(const FormExpr& x);
FormExpr indicator(const FormExpr& predicate);
FormExpr clamp(const FormExpr& x, const FormExpr& lo, const FormExpr& hi);
FormExpr regionIndicator(int domain_id);
FormExpr laplacian(const FormExpr& a);  // trace(hessian(a))

// === DG Helpers ===
FormExpr weightedAverage(const FormExpr& a, const FormExpr& w_plus, const FormExpr& w_minus);
FormExpr harmonicAverage(const FormExpr& k);  // 2*k⁺*k⁻/(k⁺+k⁻)
FormExpr upwindValue(const FormExpr& u, const FormExpr& beta);
FormExpr downwindValue(const FormExpr& u, const FormExpr& beta);
FormExpr interiorPenaltyCoefficient(Real eta, Real p = 1.0);  // SIPG scaling

// === Level-Set/Surface Geometry ===
FormExpr unitNormalFromLevelSet(const FormExpr& phi, Real eps = 1e-12);
FormExpr meanCurvatureFromLevelSet(const FormExpr& phi, Real eps = 1e-12);
FormExpr surfaceGradient(const FormExpr& f, const FormExpr& n);
FormExpr surfaceDivergence(const FormExpr& u, const FormExpr& n);
FormExpr surfaceLaplacian(const FormExpr& f, const FormExpr& n);

// === Tensor Constructors ===
FormExpr as_vector(std::initializer_list<FormExpr> components);
FormExpr as_tensor(std::initializer_list<std::initializer_list<FormExpr>> rows);
FormExpr contraction(const FormExpr& a, const FormExpr& b);
FormExpr SymmetricTensor(const FormExpr& A);  // sym(A)
FormExpr SkewTensor(const FormExpr& A);       // skew(A)

// === Constitutive Call Ergonomics ===
struct ConstitutiveCall {
    FormExpr call;
    FormExpr out(std::size_t output_index);  // Multi-output access
    operator FormExpr() const;               // Implicit conversion
};

ConstitutiveCall constitutive(std::shared_ptr<const ConstitutiveModel> model,
                              const FormExpr& input);
ConstitutiveCall constitutive(std::shared_ptr<const ConstitutiveModel> model,
                              std::vector<FormExpr> inputs);  // N-ary
}
```

---

### 12. Constraints (`FE/Constraints/`)

**Purpose:** Boundary conditions and algebraic constraints.

#### Constraint (`Constraints/Constraint.h`)

Base interface for all constraints.

```cpp
namespace svmp::FE::constraints {

class Constraint {
public:
    virtual void apply(const dofs::DofMap& dof_map,
                       AffineConstraints& constraints,
                       Real time = 0.0) const = 0;
    virtual void update(Real time) {}
};
}
```

#### Constraint Types

| Class | Description |
|-------|-------------|
| `DirichletBC` | Essential (strong) boundary conditions |
| `NeumannBC` | Natural boundary conditions (weak form) |
| `RobinBC` | Mixed boundary conditions |
| `PeriodicBC` | Periodic boundary conditions |
| `MultiPointConstraint` | Linear combinations of DOFs |
| `HangingNodeConstraint` | Hanging node constraints for non-conforming meshes |
| `LagrangeMultiplier` | Lagrange multiplier constraints |

```cpp
namespace svmp::FE::constraints {

class DirichletBC : public Constraint {
public:
    DirichletBC(int boundary_marker, int component, Real value);
    DirichletBC(int boundary_marker, int component, std::function<Real(Real, Real, Real, Real)> func);

    void apply(const dofs::DofMap& dof_map,
               AffineConstraints& constraints,
               Real time = 0.0) const override;
    void update(Real time) override;
};

class NeumannBC : public Constraint {
public:
    NeumannBC(int boundary_marker, const FormExpr& flux);

    // Returns weak-form contribution (no DOF modification)
    FormExpr getFormContribution() const;
};

class RobinBC : public Constraint {
public:
    RobinBC(int boundary_marker, Real alpha, Real beta, Real g);
    // αu + β∂u/∂n = g
};

class PeriodicBC : public Constraint {
public:
    PeriodicBC(int boundary_marker_1, int boundary_marker_2,
               const std::array<Real, 3>& translation);
};

class MultiPointConstraint : public Constraint {
public:
    void addTerm(GlobalIndex dof, Real coefficient);
    void setRHS(Real value);
    // sum(coeff_i * u_i) = rhs
};
}
```

#### AffineConstraints (`Constraints/AffineConstraints.h`)

Container for linear constraints on DOFs.

```cpp
namespace svmp::FE::constraints {

class AffineConstraints {
public:
    void addLine(GlobalIndex constrained_dof);
    void addEntry(GlobalIndex constrained_dof, GlobalIndex unconstrained_dof, Real coefficient);
    void setInhomogeneity(GlobalIndex constrained_dof, Real value);

    void close();
    bool isClosed() const;

    bool isConstrained(GlobalIndex dof) const;
    Real inhomogeneity(GlobalIndex dof) const;

    void distributeLocalToGlobal(std::span<const Real> local_vector,
                                 std::span<const GlobalIndex> local_dofs,
                                 std::span<Real> global_vector) const;

    void distributeLocalToGlobal(std::span<const Real> local_matrix,
                                 std::span<const GlobalIndex> local_row_dofs,
                                 std::span<const GlobalIndex> local_col_dofs,
                                 std::span<Real> global_matrix,
                                 std::span<Real> global_vector) const;
};
}
```

---

### 13. Systems (`FE/Systems/`)

**Purpose:** High-level FE system orchestration.

#### FESystem (`Systems/FESystem.h`)

Main entry point for multi-field FE problems.

```cpp
namespace svmp::FE::systems {

struct FieldSpec {
    std::string name;
    const spaces::FunctionSpace* space;
    int num_components{1};
};

struct SetupOptions {
    dofs::DofDistributionOptions dof_options{};
    assembly::AssemblyOptions assembly_options{};
    sparsity::SparsityBuildOptions sparsity_options{};
    std::string assembler_name{"StandardAssembler"};
    sparsity::CouplingMode coupling_mode{sparsity::CouplingMode::Full};
    bool use_constraints_in_assembly{true};
};

struct AssemblyRequest {
    OperatorTag op;
    bool want_matrix{false};
    bool want_vector{false};
    bool zero_outputs{true};
    bool assemble_boundary_terms{true};
};

class FESystem {
public:
    explicit FESystem(std::shared_ptr<const assembly::IMeshAccess> mesh_access);

    // Definition phase
    FieldId addField(FieldSpec spec);
    void addConstraint(std::unique_ptr<constraints::Constraint> c);
    void addSystemConstraint(std::unique_ptr<ISystemConstraint> c);

    void addOperator(OperatorTag name);

    void addCellKernel(OperatorTag op, FieldId field,
                       std::shared_ptr<assembly::AssemblyKernel> kernel);
    void addCellKernel(OperatorTag op, FieldId test_field, FieldId trial_field,
                       std::shared_ptr<assembly::AssemblyKernel> kernel);

    void addBoundaryKernel(OperatorTag op, int boundary_marker, FieldId field,
                           std::shared_ptr<assembly::AssemblyKernel> kernel);

    void addInteriorFaceKernel(OperatorTag op, FieldId field,
                               std::shared_ptr<assembly::AssemblyKernel> kernel);

    void addGlobalKernel(OperatorTag op, std::shared_ptr<GlobalKernel> kernel);

    void addMatrixFreeKernel(OperatorTag op,
                             std::shared_ptr<assembly::IMatrixFreeKernel> kernel);

    void addFunctionalKernel(std::string tag,
                             std::shared_ptr<assembly::FunctionalKernel> kernel);

    // Setup phase
    void setup(const SetupOptions& opts = {});

    // Constraint lifecycle
    void updateConstraints(double time, double dt = 0.0);

    // Assembly phase
    assembly::AssemblyResult assemble(const AssemblyRequest& req,
                                      const SystemStateView& state,
                                      assembly::GlobalSystemView* matrix_out,
                                      assembly::GlobalSystemView* vector_out);

    assembly::AssemblyResult assembleResidual(const SystemStateView& state,
                                              assembly::GlobalSystemView& rhs_out);
    assembly::AssemblyResult assembleJacobian(const SystemStateView& state,
                                              assembly::GlobalSystemView& jac_out);
    assembly::AssemblyResult assembleMass(const SystemStateView& state,
                                          assembly::GlobalSystemView& mass_out);

    // Time stepping lifecycle
    void beginTimeStep();
    void commitTimeStep();

    // Accessors
    const assembly::IMeshAccess& meshAccess() const;
    const dofs::DofHandler& dofHandler() const;
    const FieldRecord& fieldRecord(FieldId field) const;
    GlobalIndex fieldDofOffset(FieldId field) const;
    const constraints::AffineConstraints& constraints() const;
    const sparsity::SparsityPattern& sparsity(const OperatorTag& op) const;

    bool isSetup() const;
    bool isTransient() const;
    int temporalOrder() const;

    // Functional evaluation
    Real evaluateFunctional(const std::string& tag,
                            const SystemStateView& state) const;
    Real evaluateBoundaryFunctional(const std::string& tag,
                                    int boundary_marker,
                                    const SystemStateView& state) const;

    // Point evaluation
    std::optional<std::array<Real, 3>> evaluateFieldAtPoint(
        FieldId field,
        const SystemStateView& state,
        const std::array<Real, 3>& point,
        GlobalIndex hint_cell = INVALID_GLOBAL_INDEX) const;
};
}
```

#### SystemStateView (`Systems/SystemState.h`)

Immutable view of system state for assembly.

```cpp
namespace svmp::FE::systems {

struct SystemStateView {
    std::span<const Real> solution;        // Current solution u
    std::span<const Real> u_prev;          // Previous solution u^{n-1}
    std::span<const Real> u_prev2;         // u^{n-2}
    std::span<const Real> u_history[8];    // Extended history

    Real time{0.0};
    Real dt{0.0};
    Real dt_prev{0.0};

    std::optional<Real> getRealParam(std::string_view key) const;
};
}
```

#### TimeIntegrator (`Systems/TimeIntegrator.h`)

Abstract time integration interface.

```cpp
namespace svmp::FE::systems {

class TimeIntegrator {
public:
    virtual std::string name() const = 0;
    virtual int maxSupportedDerivativeOrder() const = 0;

    virtual assembly::TimeIntegrationContext
    buildContext(int max_time_derivative_order, const SystemStateView& state) const = 0;
};
}
```

#### Supporting Classes

| File | Description |
|------|-------------|
| `FieldRegistry.h` | Field metadata storage |
| `OperatorRegistry.h` | Operator (Jacobian, Mass, etc.) registration |
| `ParameterRegistry.h` | Runtime parameter requirements |
| `SystemConstraint.h` | System-level constraint interface |
| `BoundaryConditionManager.h` | BC registration and application |
| `CoupledBoundaryManager.h` | Coupled BC orchestration (0D models) |
| `MaterialStateProvider.h` | Per-qpt state allocation |
| `GlobalKernel.h` | Global (non-local) kernel interface |
| `TransientSystem.h` | Transient system helpers |
| `ODEIntegrator.h` | ODE integration for auxiliary state |

---

### 14. TimeStepping (`FE/TimeStepping/`)

**Purpose:** Time integration schemes and nonlinear solvers.

#### TimeLoop (`TimeStepping/TimeLoop.h`)

Primary time integration orchestrator.

```cpp
namespace svmp::FE::timestepping {

enum class SchemeKind {
    BackwardEuler, BDF2, ThetaMethod, TRBDF2,
    GeneralizedAlpha, Newmark,
    VSVO_BDF,  // Variable-step variable-order
    DG, CG     // Space-time variants
};

struct TimeLoopOptions {
    SchemeKind scheme{SchemeKind::BackwardEuler};
    double theta{1.0};                    // For theta-method
    double rho_inf{0.5};                  // For generalized-alpha
    bool use_adaptive_stepping{false};
    double t_start{0.0};
    double t_end{1.0};
    double dt_initial{0.01};
    double dt_min{1e-10};
    double dt_max{1.0};
    int max_steps{10000};
};

struct TimeLoopCallbacks {
    std::function<void(double t, double dt)> on_step_start;
    std::function<void(const NewtonReport&)> on_nonlinear_done;
    std::function<void(double t, double dt)> on_step_accepted;
    std::function<void(double t, double dt)> on_step_rejected;
    std::function<void(double dt_old, double dt_new)> on_dt_updated;
};

struct TimeLoopReport {
    bool success{true};
    int total_steps{0};
    int rejected_steps{0};
    double final_time{0.0};
};

class TimeLoop {
public:
    TimeLoopReport run(systems::FESystem& system,
                       const TimeLoopOptions& options,
                       const TimeLoopCallbacks& callbacks = {});
};
}
```

#### NewtonSolver (`TimeStepping/NewtonSolver.h`)

Nonlinear system solver with line search and modified Newton options.

```cpp
namespace svmp::FE::timestepping {

struct NewtonOptions {
    OperatorTag residual_operator{"Residual"};
    OperatorTag jacobian_operator{"Jacobian"};
    int max_iterations{30};
    double absolute_tolerance{1e-10};
    double relative_tolerance{1e-6};
    double step_tolerance{1e-12};
    int jacobian_rebuild_period{1};       // 1 = full Newton, >1 = modified Newton
    bool use_line_search{false};
    double line_search_alpha{1e-4};
    int line_search_max_backtracks{10};
    bool scale_dt_increments{false};      // For rate-form problems
};

struct NewtonReport {
    bool converged{false};
    int iterations{0};
    double final_residual_norm{0.0};
    double initial_residual_norm{0.0};
    backends::SolverReport linear_solver_report;
};

class NewtonSolver {
public:
    NewtonReport solve(systems::FESystem& system,
                       const systems::SystemStateView& state,
                       std::span<Real> solution,
                       const NewtonOptions& options);
};
}
```

#### Available Integrators

| Class | Description |
|-------|-------------|
| `NewmarkBeta` | Newmark-β for 2nd-order structural dynamics |
| `GeneralizedAlpha` | Generalized-α for 1st/2nd-order systems |
| `CollocationMethods` | BDF, DIRK, etc. |
| `VSVO_BDF_Controller` | Variable-step variable-order BDF |

#### Supporting Classes

| File | Description |
|------|-------------|
| `TimeHistory.h` | Solution history storage |
| `StepController.h` | Adaptive time step control |
| `TimeSteppingUtils.h` | Error estimation, convergence checking |

---

### 15. Constitutive (`FE/Constitutive/`)

**Purpose:** Material model infrastructure (separate from Forms integration).

#### ModelCRTP (`Constitutive/ModelCRTP.h`)

CRTP base class bridging polymorphic `ConstitutiveModel` interface with efficient compile-time dispatch.

```cpp
namespace svmp::FE::constitutive {

// CRTP base for material models with automatic AD support
template<typename Derived>
class ModelCRTP : public forms::ConstitutiveModel {
public:
    // Derived class implements one of these signatures:
    // - evaluateImpl(const Value<T>& input, int dim)
    // - evaluateImpl(const Value<T>& input, const ConstitutiveEvalContext& ctx, int dim)

    // Automatic dual number evaluation via CRTP dispatch
    Value<Real> evaluate(const Value<Real>& input, int dim) const override;
    Value<Dual> evaluate(const Value<Dual>& input, int dim,
                         DualWorkspace& workspace) const override;
};
}
```

#### Key Files

| File | Description |
|------|-------------|
| `StateLayout.h` | Structured per-qpt state field definitions |
| `StateView.h` | Type-safe views into state buffers |
| `Parameters.h` | Material parameter specifications |
| `ModelCRTP.h` | CRTP base for model implementations |
| `DualOps.h` | Dual number operations for AD |
| `ExpressionLaw.h` | Expression-based constitutive laws |
| `LawAdapters.h` | Adapters to FE/Forms `ConstitutiveModel` |
| `GlobalLaw.h` | Laws requiring global/nonlocal data |
| `ValueChecks.h` | Validation utilities |

---

### 16. Backends (`FE/Backends/`)

**Purpose:** Linear solver backends and abstract interfaces.

#### Abstract Interfaces

```cpp
namespace svmp::FE::backends {

class LinearSolver {
public:
    virtual void setOptions(const SolverOptions& options) = 0;
    virtual SolverReport solve(GenericMatrix& A,
                               GenericVector& x,
                               GenericVector& b) = 0;
    virtual BackendKind backendKind() const = 0;
};

class GenericVector {
public:
    virtual void zero() = 0;
    virtual void set(std::span<const Real> values) = 0;
    virtual void add(std::span<const Real> values) = 0;
    virtual void scale(Real alpha) = 0;
    virtual Real dot(const GenericVector& other) const = 0;
    virtual Real norm() const = 0;
    virtual void updateGhosts() = 0;
    virtual std::span<Real> localSpan() = 0;
};

class GenericMatrix {
public:
    virtual void zero() = 0;
    virtual void finalizeAssembly() = 0;
    virtual void mult(const GenericVector& x, GenericVector& y) const = 0;
    virtual void multAdd(const GenericVector& x, GenericVector& y) const = 0;
};
}
```

#### Backend Implementations

| Directory | Description |
|-----------|-------------|
| `Eigen/` | Eigen-based dense/sparse solvers |
| `FSILS/` | Fast Scalable Iterative Linear Solver |
| `PETSc/` | PETSc integration (optional) |
| `Trilinos/` | Trilinos integration (optional) |

#### FSILS (`Backends/FSILS/`)

FSILS (Fast Scalable Iterative Linear Solver) integration.

| File | Description |
|------|-------------|
| `gmres.h` | GMRES solver |
| `pc_gmres.h` | Preconditioned GMRES |
| `cgrad.h` | Conjugate gradient |
| `bicgs.h` | BiCGSTAB |
| `precond.h` | Preconditioner interface |
| `lhs.h` | Linear system setup |
| `spar_mul.h` | Sparse matrix-vector product |
| `norm.h` | Vector norms |
| `dot.h` | Dot products |
| `commu.h` | MPI communication |
| `omp_la.h` | OpenMP linear algebra |

---

## Usage Examples

### Example 1: Simple Poisson Problem

```cpp
#include "FE/Forms/FormExpr.h"
#include "FE/Forms/BilinearForm.h"
#include "FE/Spaces/H1Space.h"
#include "FE/Dofs/DofHandler.h"
#include "FE/Assembly/StandardAssembler.h"

using namespace svmp::FE;

// Create function space
spaces::H1Space V(ElementType::Triangle, /*order=*/1);

// Create symbolic expressions
auto u = forms::FormExpr::trialFunction(V);
auto v = forms::FormExpr::testFunction(V);
auto f = forms::FormExpr::coefficient([](Real x, Real y, Real z) { return 1.0; });

// Define weak form: a(u,v) = ∫ grad(u)·grad(v) dx
forms::BilinearForm a(V, V);
a.setExpression(inner(grad(u), grad(v)).dx());

// Define linear form: L(v) = ∫ f*v dx
forms::LinearForm L(V);
L.setExpression((f * v).dx());

// Compile to assembly kernels
forms::SymbolicOptions opts;
opts.jit.enable = true;
auto bilinear_kernel = a.compile(opts);
auto linear_kernel = L.compile(opts);

// Distribute DOFs
dofs::DofHandler dof_handler;
dof_handler.distributeDofs(mesh, V);
dof_handler.finalize();

// Assemble
assembly::StandardAssembler assembler;
assembler.setDofHandler(dof_handler);
assembler.initialize();

auto result = assembler.assembleBoth(mesh_access, V, V, *bilinear_kernel,
                                     matrix_view, vector_view);
assembler.finalize(&matrix_view, &vector_view);
```

### Example 2: Nonlinear Elasticity with Constitutive Model

```cpp
#include "FE/Forms/FormExpr.h"
#include "FE/Forms/NonlinearForm.h"
#include "FE/Forms/ConstitutiveModel.h"

using namespace svmp::FE;

// Define neo-Hookean constitutive model
class NeoHookean : public forms::ConstitutiveModel {
public:
    NeoHookean(Real mu, Real kappa) : mu_(mu), kappa_(kappa) {}

    forms::Value<Real> evaluate(const forms::Value<Real>& F, int dim) const override {
        // F is deformation gradient
        Real J = /* det(F) */;
        // Compute Cauchy stress...
        return forms::Value<Real>::fromMatrix(stress);
    }

    forms::Value<forms::Dual> evaluate(const forms::Value<forms::Dual>& F,
                                       int dim,
                                       forms::DualWorkspace& ws) const override {
        // Same computation with dual numbers for automatic tangent
    }

private:
    Real mu_, kappa_;
};

// Create vector-valued space for displacement
spaces::H1Space V(ElementType::Tetrahedron, /*order=*/1);
// (with 3 components for 3D)

auto u = forms::FormExpr::trialFunction(V);
auto v = forms::FormExpr::testFunction(V);

// Deformation gradient F = I + grad(u)
auto F = forms::FormExpr::identity() + grad(u);

// Constitutive stress
NeoHookean material(/*mu=*/1.0, /*kappa=*/100.0);
auto P = forms::constitutive(material, F);  // First Piola-Kirchhoff stress

// Residual: R(u;v) = ∫ P : grad(v) dx
forms::NonlinearForm R(V);
R.setResidual(doubleContraction(P, grad(v)).dx());

// Compile residual + consistent Jacobian
forms::SymbolicOptions opts;
opts.ad_mode = forms::ADMode::Forward;
auto kernel = R.compile(opts);
```

### Example 3: Multi-Field Stokes Problem

```cpp
#include "FE/Systems/FESystem.h"
#include "FE/Spaces/H1Space.h"
#include "FE/Spaces/L2Space.h"

using namespace svmp::FE;

// Taylor-Hood elements: P2 velocity, P1 pressure
spaces::H1Space V_h(ElementType::Triangle, /*order=*/2);  // Velocity space
spaces::H1Space Q_h(ElementType::Triangle, /*order=*/1);  // Pressure space

systems::FESystem system(mesh_access);

// Add fields
auto u_field = system.addField({"velocity", &V_h, /*components=*/2});
auto p_field = system.addField({"pressure", &Q_h, /*components=*/1});

// Add operator
system.addOperator("Stokes");

// Create forms
auto u = forms::FormExpr::trialFunction(V_h);
auto v = forms::FormExpr::testFunction(V_h);
auto p = forms::FormExpr::trialFunction(Q_h);
auto q = forms::FormExpr::testFunction(Q_h);

Real mu = 1.0;  // Viscosity

// a(u,v) = μ ∫ grad(u):grad(v) dx
forms::BilinearForm a(V_h, V_h);
a.setExpression((mu * doubleContraction(grad(u), grad(v))).dx());

// b(v,p) = -∫ p div(v) dx
forms::BilinearForm b(V_h, Q_h);
b.setExpression((-p * div(v)).dx());

// Compile and add kernels
system.addCellKernel("Stokes", u_field, u_field, a.compile());
system.addCellKernel("Stokes", u_field, p_field, b.compile());
// b^T block
forms::BilinearForm bT(Q_h, V_h);
bT.setExpression((-q * div(u)).dx());
system.addCellKernel("Stokes", p_field, u_field, bT.compile());

// Setup
systems::SetupOptions setup_opts;
system.setup(setup_opts);

// Assemble
systems::SystemStateView state;
state.solution = solution_vector;
state.time = 0.0;

system.assemble({"Stokes", true, true, true}, state, &matrix_view, &vector_view);
```

### Example 4: DG Advection

```cpp
#include "FE/Forms/FormExpr.h"
#include "FE/Spaces/L2Space.h"

using namespace svmp::FE;

// DG space
spaces::L2Space V_h(ElementType::Triangle, /*order=*/1);

auto u = forms::FormExpr::trialFunction(V_h);
auto v = forms::FormExpr::testFunction(V_h);

// Advection velocity
auto beta = forms::FormExpr::coefficient([](Real x, Real y, Real z) {
    return std::array<Real, 3>{1.0, 0.5, 0.0};
});

// Cell integral: -∫ u (beta·grad(v)) dx
auto cell_term = (-u * inner(beta, grad(v))).dx();

// Interior face integral with upwind flux
auto h = forms::FormExpr::h();
auto n = forms::FormExpr::n();
auto beta_n = inner(beta, n);

auto u_upwind = forms::upwindValue(beta_n, u.minus(), u.plus());
auto face_term = (u_upwind * jump(v) * beta_n).dS();

// Boundary integral
auto u_bc = forms::FormExpr::coefficient([](Real x, Real y, Real z) { return 0.0; });
auto boundary_term = (u_bc * v * beta_n).ds(/*inflow_marker=*/1);
```

---

## Build Configuration

### CMake Variables

| Variable | Description |
|----------|-------------|
| `SVMP_FE_WITH_MESH` | Enable Mesh library integration |
| `FE_HAS_MPI` | Enable MPI support |
| `SVMP_FE_WITH_LLVM` | Enable LLVM JIT compilation |
| `FE_ENABLE_SIMD` | Enable SIMD optimizations |

### Compile-Time Feature Detection

```cpp
#include "FE/Core/FEConfig.h"

#if FE_HAS_MPI
    // MPI-enabled code
#endif

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    // Mesh library integration
#endif
```

---

## Thread Safety

- **After finalization**, `DofHandler` read methods are thread-safe
- **Assembly** thread safety depends on the chosen `Assembler` and decorator stack:
  - **Base Assemblers:**
    - `StandardAssembler`: Single-threaded, not thread-safe
    - `WorkStreamAssembler`: Thread-safe via task pipeline (cell → copy → assemble stages)
    - `ParallelAssembler`: MPI-parallel (one thread per rank, MPI communication for ghost exchange)
    - `DeviceAssembler`: GPU-parallel (requires CUDA/HIP)
    - `SymbolicAssembler`: Single-threaded
  - **Decorator Assemblers:**
    - `ColoredAssembler`: Enables thread-safe parallel assembly via graph coloring; wraps any base assembler
    - `ScheduledAssembler`: No threading change; improves cache locality
    - `CachedAssembler`: Thread-safe reads after cache population
    - `VectorizedAssembler`: Thread-safe batching framework
- **Math types** are value types with no shared state
- **Form compilation** should be done before parallel assembly

---

## Performance Tips

1. **Enable JIT compilation** for complex forms with many terms
2. **Use expression templates** — avoid storing intermediate `Vector`/`Matrix` results
3. **Pre-allocate sparsity patterns** before assembly
4. **Choose the right base assembler for your parallelism model**:
   - `StandardAssembler` for simple serial assembly
   - `ParallelAssembler` for distributed-memory MPI parallelism
   - `WorkStreamAssembler` for task-based parallelism
5. **Stack decorators for additional capabilities**:
   - `ColoredAssembler` decorator for shared-memory thread parallelism via graph coloring
   - `CachedAssembler` decorator for repeated assembly (transient/nonlinear problems)
   - `VectorizedAssembler` decorator for SIMD-batched element assembly
   - `ScheduledAssembler` decorator for cache-friendly traversal ordering
6. **Use TensorIR lowering** (`tensor.mode = TensorLoweringMode::On`) for forms with many tensor contractions
7. **Profile with `AssemblyStatistics`** to identify bottlenecks
8. **Use specialized 2×2/3×3 matrix operations** — they use analytical formulas

---

## Error Handling

All errors throw `FEException` with:
- Error message
- Source file and line number
- Function name
- Status code (`FEStatus` enum)

```cpp
try {
    dof_handler.distributeDofs(topology, layout);
} catch (const FEException& e) {
    std::cerr << e.what() << std::endl;
    std::cerr << "File: " << e.file() << ":" << e.line() << std::endl;
    std::cerr << "Function: " << e.function() << std::endl;
}
```

---

## References

The FE/Forms vocabulary is inspired by:

- Alnæs et al. — "Unified form language." *ACM TOMS* (2014)
- Logg, Mardal, Wells (eds.) — *Automated Solution of Differential Equations by the Finite Element Method* (2012)
- Rathgeber et al. — "Firedrake." *ACM TOMS* (2016)

---

## License

See `License` file in the repository root.

Copyright (c) Stanford University, The Regents of the University of California, and others.

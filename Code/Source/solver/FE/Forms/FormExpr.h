#ifndef SVMP_FE_FORMS_FORM_EXPR_H
#define SVMP_FE_FORMS_FORM_EXPR_H

/**
 * @file FormExpr.h
 * @brief Symbolic weak-form expression vocabulary (AST) for FE/Forms
 */

#include "Core/Types.h"

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {

namespace spaces {
enum class SpaceType : std::uint8_t;
class FunctionSpace;
}

namespace forms {

class ConstitutiveModel;
class Index;

// ============================================================================
// Options / Modes
// ============================================================================

/**
 * @brief AD mode for Jacobian computation
 */
enum class ADMode : std::uint8_t {
    None,
    Forward,
    Reverse,
    Taylor
};

/**
 * @brief JIT compilation options (placeholder for future codegen)
 */
struct JITOptions {
    bool enable{false};
    int optimization_level{2};
    bool cache_kernels{true};
    bool vectorize{true};
    std::string cache_directory;
};

/**
 * @brief Options for symbolic form compilation
 */
struct SymbolicOptions {
    ADMode ad_mode{ADMode::None};
    JITOptions jit{};
    bool simplify_expressions{true};
    bool exploit_sparsity{true};
    bool cache_expressions{true};
    bool verbose{false};
};

// ============================================================================
// Expression node types
// ============================================================================

/**
 * @brief Node tag for the form expression AST
 */
enum class FormExprType : std::uint16_t {
    // Terminals
    TestFunction,
    TrialFunction,
    DiscreteField,
    StateField,
    Coefficient,
    Constant,
    Coordinate,
    ReferenceCoordinate,
    Identity,
    Jacobian,
    JacobianInverse,
    JacobianDeterminant,
    Normal,
    CellDiameter,
    CellVolume,
    FacetArea,

    // Differential operators
    Gradient,
    Divergence,
    Curl,
    Hessian,
    TimeDerivative,

    // Restrictions (DG trace selection)
    RestrictMinus,
    RestrictPlus,

    // DG operators
    Jump,
    Average,

    // Algebra / composition
    Negate,
    Add,
    Subtract,
    Multiply,
    Divide,
    InnerProduct,
    DoubleContraction,
    OuterProduct,
    CrossProduct,
    Power,
    Minimum,
    Maximum,

    // Comparisons / predicates (return scalar 0/1)
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Equal,
    NotEqual,

    // Ternary
    Conditional,

    // Constructors / packing
    AsVector,
    AsTensor,

    // Indexing
    Component,
    IndexedAccess,

    // Tensor ops / scalar functions
    Transpose,
    Trace,
    Determinant,
    Inverse,
    Cofactor,
    Deviator,
    SymmetricPart,
    SkewPart,
    Norm,
    Normalize,
    AbsoluteValue,
    Sign,
    Sqrt,
    Exp,
    Log,

    // Constitutive (material-point) operator
    Constitutive,
    ConstitutiveOutput,

    // Integrals
    CellIntegral,
    BoundaryIntegral,
    InteriorFaceIntegral,
};

class FormExpr;

using ScalarCoefficient = std::function<Real(Real, Real, Real)>;
using VectorCoefficient = std::function<std::array<Real, 3>(Real, Real, Real)>;
using MatrixCoefficient = std::function<std::array<std::array<Real, 3>, 3>(Real, Real, Real)>;
using Tensor4Coefficient = std::function<std::array<Real, 81>(Real, Real, Real)>;

/**
 * @brief Base class for AST nodes
 */
class FormExprNode {
public:
    virtual ~FormExprNode() = default;

    [[nodiscard]] virtual FormExprType type() const noexcept = 0;
    [[nodiscard]] virtual std::string toString() const = 0;
    [[nodiscard]] virtual bool hasTest() const noexcept = 0;
    [[nodiscard]] virtual bool hasTrial() const noexcept = 0;

    struct SpaceSignature {
        spaces::SpaceType space_type{};
        FieldType field_type{FieldType::Scalar};
        Continuity continuity{Continuity::C0};
        int value_dimension{1};
        int topological_dimension{0};
        int polynomial_order{1};
        ElementType element_type{ElementType::Unknown};
    };

    // Optional payload accessors (used by compiler/evaluator without RTTI)
    [[nodiscard]] virtual std::optional<Real> constantValue() const { return std::nullopt; }
    [[nodiscard]] virtual std::optional<int> identityDim() const { return std::nullopt; }
    [[nodiscard]] virtual std::optional<int> boundaryMarker() const { return std::nullopt; }
    [[nodiscard]] virtual std::optional<int> componentIndex0() const { return std::nullopt; }
    [[nodiscard]] virtual std::optional<int> componentIndex1() const { return std::nullopt; }
    [[nodiscard]] virtual std::optional<int> tensorRows() const { return std::nullopt; }
    [[nodiscard]] virtual std::optional<int> tensorCols() const { return std::nullopt; }
    [[nodiscard]] virtual std::optional<int> indexRank() const { return std::nullopt; }
    [[nodiscard]] virtual std::optional<std::array<int, 4>> indexIds() const { return std::nullopt; }
    [[nodiscard]] virtual std::optional<std::array<int, 4>> indexExtents() const { return std::nullopt; }
    [[nodiscard]] virtual std::optional<int> constitutiveOutputIndex() const { return std::nullopt; }
    [[nodiscard]] virtual const SpaceSignature* spaceSignature() const { return nullptr; }
    [[nodiscard]] virtual std::optional<int> timeDerivativeOrder() const { return std::nullopt; }
    [[nodiscard]] virtual std::optional<FieldId> fieldId() const { return std::nullopt; }

    [[nodiscard]] virtual const ScalarCoefficient* scalarCoefficient() const { return nullptr; }
    [[nodiscard]] virtual const VectorCoefficient* vectorCoefficient() const { return nullptr; }
    [[nodiscard]] virtual const MatrixCoefficient* matrixCoefficient() const { return nullptr; }
    [[nodiscard]] virtual const Tensor4Coefficient* tensor4Coefficient() const { return nullptr; }
    [[nodiscard]] virtual const ConstitutiveModel* constitutiveModel() const { return nullptr; }
    [[nodiscard]] virtual std::shared_ptr<const ConstitutiveModel> constitutiveModelShared() const { return {}; }

    /**
     * @brief Child node handles (shared ownership)
     *
     * This is primarily for compiler/evaluator use; user-facing APIs should
     * treat FormExpr as the value-semantic handle.
     */
    [[nodiscard]] virtual std::vector<std::shared_ptr<FormExprNode>> childrenShared() const {
        return {};
    }

    [[nodiscard]] virtual std::vector<const FormExprNode*> children() const {
        return {};
    }
};

// ============================================================================
// FormExpr handle
// ============================================================================

/**
 * @brief Value-semantic handle for a weak-form expression
 */
class FormExpr {
public:
    FormExpr();
    explicit FormExpr(std::shared_ptr<FormExprNode> node);

    // ---- Terminals ----
    static FormExpr testFunction(const spaces::FunctionSpace& space, std::string name = "v");
    static FormExpr trialFunction(const spaces::FunctionSpace& space, std::string name = "u");
    static FormExpr discreteField(FieldId field, const spaces::FunctionSpace& space, std::string name = "u");
    static FormExpr stateField(FieldId field, const spaces::FunctionSpace& space, std::string name = "u");

    static FormExpr coefficient(std::string name, ScalarCoefficient func);
    static FormExpr coefficient(std::string name, VectorCoefficient func);
    static FormExpr coefficient(std::string name, MatrixCoefficient func);
    static FormExpr coefficient(std::string name, Tensor4Coefficient func);

    static FormExpr constant(Real value);
    static FormExpr coordinate();
    static FormExpr referenceCoordinate();
    static FormExpr identity(int dim);
    static FormExpr jacobian();
    static FormExpr jacobianInverse();
    static FormExpr jacobianDeterminant();
    static FormExpr normal();
    static FormExpr cellDiameter();
    static FormExpr cellVolume();
    static FormExpr facetArea();

    // ---- Constructors / packing ----
    static FormExpr asVector(std::vector<FormExpr> components);
    static FormExpr asTensor(std::vector<std::vector<FormExpr>> rows);
    static FormExpr constitutive(std::shared_ptr<const ConstitutiveModel> model, const FormExpr& input);
    static FormExpr constitutive(std::shared_ptr<const ConstitutiveModel> model, std::vector<FormExpr> inputs);
    static FormExpr constitutiveOutput(const FormExpr& call, std::size_t output_index);

    // ---- Differential operators ----
    [[nodiscard]] FormExpr grad() const;
    [[nodiscard]] FormExpr div() const;
    [[nodiscard]] FormExpr curl() const;
    [[nodiscard]] FormExpr hessian() const;
    [[nodiscard]] FormExpr dt(int order = 1) const;

    // ---- Restrictions (DG trace selection) ----
    [[nodiscard]] FormExpr minus() const;
    [[nodiscard]] FormExpr plus() const;

    // ---- DG operators ----
    [[nodiscard]] FormExpr jump() const;
    [[nodiscard]] FormExpr avg() const;

    // ---- Algebra ----
    [[nodiscard]] FormExpr operator-() const;
    [[nodiscard]] FormExpr operator+(const FormExpr& rhs) const;
    [[nodiscard]] FormExpr operator-(const FormExpr& rhs) const;
    [[nodiscard]] FormExpr operator*(const FormExpr& rhs) const;
    [[nodiscard]] FormExpr operator/(const FormExpr& rhs) const;
    [[nodiscard]] FormExpr operator*(Real scalar) const;
    [[nodiscard]] FormExpr operator/(Real scalar) const;

    [[nodiscard]] FormExpr inner(const FormExpr& rhs) const;
    [[nodiscard]] FormExpr doubleContraction(const FormExpr& rhs) const;
    [[nodiscard]] FormExpr outer(const FormExpr& rhs) const;
    [[nodiscard]] FormExpr cross(const FormExpr& rhs) const;
    [[nodiscard]] FormExpr pow(const FormExpr& exponent) const;
    [[nodiscard]] FormExpr min(const FormExpr& rhs) const;
    [[nodiscard]] FormExpr max(const FormExpr& rhs) const;

    [[nodiscard]] FormExpr lt(const FormExpr& rhs) const;
    [[nodiscard]] FormExpr le(const FormExpr& rhs) const;
    [[nodiscard]] FormExpr gt(const FormExpr& rhs) const;
    [[nodiscard]] FormExpr ge(const FormExpr& rhs) const;
    [[nodiscard]] FormExpr eq(const FormExpr& rhs) const;
    [[nodiscard]] FormExpr ne(const FormExpr& rhs) const;

    [[nodiscard]] FormExpr conditional(const FormExpr& then_expr, const FormExpr& else_expr) const;

    // ---- Indexing ----
    [[nodiscard]] FormExpr component(int i, int j = -1) const;
    [[nodiscard]] FormExpr operator()(const Index& i) const;
    [[nodiscard]] FormExpr operator()(const Index& i, const Index& j) const;

    // ---- Tensor ops / scalar functions ----
    [[nodiscard]] FormExpr transpose() const;
    [[nodiscard]] FormExpr trace() const;
    [[nodiscard]] FormExpr det() const;
    [[nodiscard]] FormExpr inv() const;
    [[nodiscard]] FormExpr cofactor() const;
    [[nodiscard]] FormExpr dev() const;
    [[nodiscard]] FormExpr sym() const;
    [[nodiscard]] FormExpr skew() const;
    [[nodiscard]] FormExpr norm() const;
    [[nodiscard]] FormExpr normalize() const;
    [[nodiscard]] FormExpr abs() const;
    [[nodiscard]] FormExpr sign() const;
    [[nodiscard]] FormExpr sqrt() const;
    [[nodiscard]] FormExpr exp() const;
    [[nodiscard]] FormExpr log() const;

    // ---- Measures ----
    [[nodiscard]] FormExpr dx() const;
    [[nodiscard]] FormExpr ds(int boundary_marker = -1) const;
    [[nodiscard]] FormExpr dS() const;

    // ---- Query ----
    [[nodiscard]] bool isValid() const noexcept { return node_ != nullptr; }
    [[nodiscard]] const FormExprNode* node() const noexcept { return node_.get(); }
    [[nodiscard]] std::shared_ptr<FormExprNode> nodeShared() const noexcept { return node_; }

    [[nodiscard]] std::string toString() const;
    [[nodiscard]] bool hasTest() const noexcept;
    [[nodiscard]] bool hasTrial() const noexcept;

    using NodeTransform = std::function<std::optional<FormExpr>(const FormExprNode&)>;
    [[nodiscard]] FormExpr transformNodes(const NodeTransform& transform) const;

private:
    std::shared_ptr<FormExprNode> node_;
};

// Convenience free functions (EDSL-style)
inline FormExpr operator*(Real scalar, const FormExpr& expr) { return expr * scalar; }

inline FormExpr grad(const FormExpr& expr) { return expr.grad(); }
inline FormExpr div(const FormExpr& expr) { return expr.div(); }
inline FormExpr curl(const FormExpr& expr) { return expr.curl(); }
inline FormExpr inner(const FormExpr& a, const FormExpr& b) { return a.inner(b); }
inline FormExpr dot(const FormExpr& a, const FormExpr& b) { return a.inner(b); }
inline FormExpr outer(const FormExpr& a, const FormExpr& b) { return a.outer(b); }
inline FormExpr cross(const FormExpr& a, const FormExpr& b) { return a.cross(b); }
inline FormExpr dt(const FormExpr& a, int order = 1) { return a.dt(order); }
inline FormExpr pow(const FormExpr& a, const FormExpr& b) { return a.pow(b); }
inline FormExpr min(const FormExpr& a, const FormExpr& b) { return a.min(b); }
inline FormExpr max(const FormExpr& a, const FormExpr& b) { return a.max(b); }
inline FormExpr lt(const FormExpr& a, const FormExpr& b) { return a.lt(b); }
inline FormExpr le(const FormExpr& a, const FormExpr& b) { return a.le(b); }
inline FormExpr gt(const FormExpr& a, const FormExpr& b) { return a.gt(b); }
inline FormExpr ge(const FormExpr& a, const FormExpr& b) { return a.ge(b); }
inline FormExpr eq(const FormExpr& a, const FormExpr& b) { return a.eq(b); }
inline FormExpr ne(const FormExpr& a, const FormExpr& b) { return a.ne(b); }
inline FormExpr conditional(const FormExpr& cond, const FormExpr& t, const FormExpr& f) {
    return cond.conditional(t, f);
}
inline FormExpr component(const FormExpr& a, int i, int j = -1) { return a.component(i, j); }
inline FormExpr transpose(const FormExpr& a) { return a.transpose(); }
inline FormExpr trace(const FormExpr& a) { return a.trace(); }
inline FormExpr det(const FormExpr& a) { return a.det(); }
inline FormExpr inv(const FormExpr& a) { return a.inv(); }
inline FormExpr cofactor(const FormExpr& a) { return a.cofactor(); }
inline FormExpr dev(const FormExpr& a) { return a.dev(); }
inline FormExpr sym(const FormExpr& a) { return a.sym(); }
inline FormExpr skew(const FormExpr& a) { return a.skew(); }
inline FormExpr norm(const FormExpr& a) { return a.norm(); }
inline FormExpr normalize(const FormExpr& a) { return a.normalize(); }
inline FormExpr abs(const FormExpr& a) { return a.abs(); }
inline FormExpr sign(const FormExpr& a) { return a.sign(); }
inline FormExpr sqrt(const FormExpr& a) { return a.sqrt(); }
inline FormExpr exp(const FormExpr& a) { return a.exp(); }
inline FormExpr log(const FormExpr& a) { return a.log(); }
inline FormExpr jump(const FormExpr& expr) { return expr.jump(); }
inline FormExpr avg(const FormExpr& expr) { return expr.avg(); }
inline FormExpr minus(const FormExpr& expr) { return expr.minus(); }
inline FormExpr plus(const FormExpr& expr) { return expr.plus(); }

} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_FORM_EXPR_H

#ifndef SVMP_FE_FORMS_FORM_EXPR_H
#define SVMP_FE_FORMS_FORM_EXPR_H

/**
 * @file FormExpr.h
 * @brief Symbolic weak-form expression vocabulary (AST) for FE/Forms
 */

#include "Core/Types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
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
namespace tensor {
enum class IndexVariance : std::uint8_t;
struct TensorIndex;
} // namespace tensor

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
enum class TensorLoweringMode : std::uint8_t {
    Off,
    Auto,
    On,
};

struct TensorJITOptions {
    // Rollout default: keep tensor-calculus lowering opt-in until benchmark targets are met.
    TensorLoweringMode mode{TensorLoweringMode::Off};

    // Force loop-nest lowering even when scalar expansion is "small".
    bool force_loop_nest{false};

    // Optional feedback: log when the tensor path is selected and why.
    bool log_decisions{false};

    // Lowering knobs (LoopStructureOptions).
    bool enable_symmetry_lowering{true};
    bool enable_optimal_contraction_order{true};
    bool enable_vectorization_hints{true};
    bool enable_delta_shortcuts{true};
    std::uint64_t scalar_expansion_term_threshold{64};

    // Temporary allocation knobs (TensorAllocationOptions).
    std::size_t temp_stack_max_entries{81};
    std::size_t temp_alignment_bytes{64};
    bool temp_enable_reuse{true};

    // Optional: enable Polly-related loop metadata / passes.
    bool enable_polly{false};
};

struct JITSpecializationOptions {
    // Master switch for emitting and using specialized kernel variants.
    bool enable{false};

    // Which sizes to specialize when compiling variants.
    bool specialize_n_qpts{true};
    bool specialize_dofs{false};

    // Simple compile-time guards to avoid code-size explosions.
    std::uint32_t max_specialized_n_qpts{32};
    std::uint32_t max_specialized_dofs{64};

    // Per-kernel cap on (size -> variant) entries kept by the wrapper.
    std::size_t max_variants_per_kernel{8};

    // Optional loop metadata for the LLVM optimizer.
    bool enable_loop_unroll_metadata{true};
    std::uint32_t max_unroll_trip_count{32};
};

struct JITOptions {
    bool enable{false};
    int optimization_level{2};
    bool cache_kernels{true};
    bool vectorize{true};
    std::string cache_directory;
    bool cache_diagnostics{false};
    std::size_t max_in_memory_kernels{0};
    bool dump_kernel_ir{false};
    bool dump_llvm_ir{false};
    bool dump_llvm_ir_optimized{false};
    bool debug_info{false};
    std::string dump_directory{"svmp_fe_jit_dumps"};
    JITSpecializationOptions specialization{};
    TensorJITOptions tensor{};
};

/**
 * @brief Options for symbolic form compilation
 */
struct SymbolicOptions {
    ADMode ad_mode{ADMode::None};
    bool use_symbolic_tangent{false};
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
	    ParameterSymbol,          ///< Runtime scalar parameter by name (setup-time identifier)
	    ParameterRef,             ///< Runtime scalar parameter by slot (JIT-friendly)
	    Constant,
	    BoundaryFunctionalSymbol,
	    BoundaryIntegralSymbol,   ///< Coupled boundary-integral value by name (requires Systems registration)
	    BoundaryIntegralRef,      ///< Coupled boundary-integral value by slot (JIT-friendly)
	    AuxiliaryStateSymbol,
	    AuxiliaryStateRef,        ///< Coupled auxiliary-state value by slot (JIT-friendly)
	    MaterialStateOldRef,      ///< Per-qpt material state load (old) by byte offset
	    MaterialStateWorkRef,     ///< Per-qpt material state load (work/current) by byte offset
	    PreviousSolutionRef,      ///< Previous solution value u^{n-k} by history index k>=1
	    Coordinate,
	    ReferenceCoordinate,
	    Time,
	    TimeStep,
	    EffectiveTimeStep,        ///< Effective dt implied by dt(·) stencil coefficient (falls back to TimeStep)
	    Identity,
	    Jacobian,
	    JacobianInverse,
	    JacobianDeterminant,
	    Normal,
    CellDiameter,
    CellVolume,
    FacetArea,
    CellDomainId,

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
		    InterfaceIntegral,

	    // Spectral / eigen (symmetric tensors)
	    SymmetricEigenvalue,
	    SymmetricEigenvalueDirectionalDerivative,
	    SymmetricEigenvalueDirectionalDerivativeWrtA,

        // -----------------------------------------------------------------
        // Section 10.1: New-physics vocabulary extensions (Forms + LLVM JIT)
        // -----------------------------------------------------------------

	        // Matrix functions (non element-wise; small dense matrices)
	        MatrixExponential,
	        MatrixLogarithm,
	        MatrixSqrt,
	        MatrixPower,

	        // Directional derivatives of matrix functions (d/dA f(A) · dA)
	        MatrixExponentialDirectionalDerivative,
	        MatrixLogarithmDirectionalDerivative,
	        MatrixSqrtDirectionalDerivative,
	        MatrixPowerDirectionalDerivative,

        // Regularized / smooth scalar approximations
        SmoothHeaviside,
        SmoothAbsoluteValue,
        SmoothMin,
        SmoothMax,
        SmoothSign,

	        // General eigendecomposition operators
	        Eigenvalue,
	        SymmetricEigenvector,
	        SpectralDecomposition,

	        // Directional derivatives of eigen operators (d/dA op(A) · dA)
	        SymmetricEigenvectorDirectionalDerivative,
	        SpectralDecompositionDirectionalDerivative,

        // Convolution / history operators
        HistoryWeightedSum,
        HistoryConvolution,
		};

class FormExpr;

using ScalarCoefficient = std::function<Real(Real, Real, Real)>;
using TimeScalarCoefficient = std::function<Real(Real, Real, Real, Real)>;
using VectorCoefficient = std::function<std::array<Real, 3>(Real, Real, Real)>;
using MatrixCoefficient = std::function<std::array<std::array<Real, 3>, 3>(Real, Real, Real)>;
using Tensor3Coefficient = std::function<std::array<Real, 27>(Real, Real, Real)>;
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
	    [[nodiscard]] virtual std::optional<int> interfaceMarker() const { return std::nullopt; }
	    [[nodiscard]] virtual std::optional<int> componentIndex0() const { return std::nullopt; }
    [[nodiscard]] virtual std::optional<int> componentIndex1() const { return std::nullopt; }
    [[nodiscard]] virtual std::optional<int> tensorRows() const { return std::nullopt; }
		    [[nodiscard]] virtual std::optional<int> tensorCols() const { return std::nullopt; }
		    [[nodiscard]] virtual std::optional<int> indexRank() const { return std::nullopt; }
		    [[nodiscard]] virtual std::optional<std::array<int, 4>> indexIds() const { return std::nullopt; }
		    [[nodiscard]] virtual std::optional<std::array<int, 4>> indexExtents() const { return std::nullopt; }
		    [[nodiscard]] virtual std::optional<std::array<std::string_view, 4>> indexNames() const { return std::nullopt; }
		    [[nodiscard]] virtual std::optional<std::array<tensor::IndexVariance, 4>> indexVariances() const { return std::nullopt; }
		    [[nodiscard]] virtual std::optional<int> constitutiveOutputIndex() const { return std::nullopt; }
		    [[nodiscard]] virtual const SpaceSignature* spaceSignature() const { return nullptr; }
		    [[nodiscard]] virtual std::optional<int> timeDerivativeOrder() const { return std::nullopt; }
		    [[nodiscard]] virtual std::optional<FieldId> fieldId() const { return std::nullopt; }
		    [[nodiscard]] virtual std::optional<std::string_view> symbolName() const { return std::nullopt; }
		    [[nodiscard]] virtual std::optional<std::uint32_t> slotIndex() const { return std::nullopt; }
		    [[nodiscard]] virtual std::optional<int> historyIndex() const { return std::nullopt; }
		    [[nodiscard]] virtual std::optional<std::uint32_t> stateOffsetBytes() const { return std::nullopt; }
	    [[nodiscard]] virtual std::optional<int> eigenIndex() const { return std::nullopt; }

    [[nodiscard]] virtual const ScalarCoefficient* scalarCoefficient() const { return nullptr; }
    [[nodiscard]] virtual const TimeScalarCoefficient* timeScalarCoefficient() const { return nullptr; }
    [[nodiscard]] virtual const VectorCoefficient* vectorCoefficient() const { return nullptr; }
    [[nodiscard]] virtual const MatrixCoefficient* matrixCoefficient() const { return nullptr; }
    [[nodiscard]] virtual const Tensor3Coefficient* tensor3Coefficient() const { return nullptr; }
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

    // Signature-based overloads (for tooling/transforms that do not have a FunctionSpace instance).
    static FormExpr testFunction(const FormExprNode::SpaceSignature& signature, std::string name = "v");
    static FormExpr trialFunction(const FormExprNode::SpaceSignature& signature, std::string name = "u");
    static FormExpr discreteField(FieldId field, const FormExprNode::SpaceSignature& signature, std::string name = "u");
    static FormExpr stateField(FieldId field, const FormExprNode::SpaceSignature& signature, std::string name = "u");

    // NOTE: Callback-based coefficients are supported for interpreter/fallback paths,
    // but are not "JIT-fast" (opaque call boundary; no inlining/vectorization).
    static FormExpr coefficient(std::string name, ScalarCoefficient func);
    static FormExpr coefficient(std::string name, TimeScalarCoefficient func);
    static FormExpr coefficient(std::string name, VectorCoefficient func);
    static FormExpr coefficient(std::string name, MatrixCoefficient func);
    static FormExpr coefficient(std::string name, Tensor3Coefficient func);
    static FormExpr coefficient(std::string name, Tensor4Coefficient func);

    static FormExpr parameter(std::string name);
    static FormExpr parameterRef(std::uint32_t slot);

    static FormExpr constant(Real value);
    static FormExpr boundaryIntegral(FormExpr integrand, int boundary_marker, std::string name);
    static FormExpr boundaryIntegralValue(std::string name);
    static FormExpr boundaryIntegralRef(std::uint32_t slot);
    static FormExpr auxiliaryState(std::string name);
    static FormExpr auxiliaryStateRef(std::uint32_t slot);
    static FormExpr materialStateOldRef(std::uint32_t offset_bytes);
    static FormExpr materialStateWorkRef(std::uint32_t offset_bytes);
    static FormExpr previousSolution(int steps_back = 1);
	    static FormExpr coordinate();
	    static FormExpr referenceCoordinate();
	    static FormExpr time();
	    static FormExpr timeStep();
	    static FormExpr effectiveTimeStep();
	    static FormExpr identity();
	    static FormExpr identity(int dim);
	    static FormExpr jacobian();
	    static FormExpr jacobianInverse();
    static FormExpr jacobianDeterminant();
    static FormExpr normal();
    static FormExpr cellDiameter();
    static FormExpr cellVolume();
    static FormExpr facetArea();
    static FormExpr cellDomainId();

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
	    [[nodiscard]] FormExpr operator()(const Index& i, const Index& j, const Index& k) const;
	    [[nodiscard]] FormExpr operator()(const Index& i, const Index& j, const Index& k, const Index& l) const;
	    [[nodiscard]] FormExpr operator()(const tensor::TensorIndex& i) const;
	    [[nodiscard]] FormExpr operator()(const tensor::TensorIndex& i, const tensor::TensorIndex& j) const;
	    [[nodiscard]] FormExpr operator()(const tensor::TensorIndex& i, const tensor::TensorIndex& j, const tensor::TensorIndex& k) const;
	    [[nodiscard]] FormExpr operator()(const tensor::TensorIndex& i, const tensor::TensorIndex& j, const tensor::TensorIndex& k, const tensor::TensorIndex& l) const;

    // Advanced: construct an IndexedAccess node with explicit index metadata.
    // This is intended for tensor-calculus transforms that must preserve
    // symbolic indices without relying on `forms::Index`'s global id counter.
	    static FormExpr indexedAccessRaw(FormExpr base,
	                                     int rank,
	                                     std::array<int, 4> ids,
	                                     std::array<int, 4> extents);
	    static FormExpr indexedAccessRawWithMetadata(FormExpr base,
	                                                 int rank,
	                                                 std::array<int, 4> ids,
	                                                 std::array<int, 4> extents,
	                                                 std::array<tensor::IndexVariance, 4> variances,
	                                                 std::array<std::string, 4> names);

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

	    // ---- Matrix functions (small dense matrices) ----
	    [[nodiscard]] FormExpr matrixExp() const;
	    [[nodiscard]] FormExpr matrixLog() const;
	    [[nodiscard]] FormExpr matrixSqrt() const;
	    [[nodiscard]] FormExpr matrixPow(const FormExpr& exponent) const;

	    // Directional derivatives of matrix functions (Fréchet derivatives along dA).
	    [[nodiscard]] static FormExpr matrixExpDirectionalDerivative(const FormExpr& A, const FormExpr& dA);
	    [[nodiscard]] static FormExpr matrixLogDirectionalDerivative(const FormExpr& A, const FormExpr& dA);
	    [[nodiscard]] static FormExpr matrixSqrtDirectionalDerivative(const FormExpr& A, const FormExpr& dA);
	    [[nodiscard]] static FormExpr matrixPowDirectionalDerivative(const FormExpr& A,
	                                                                 const FormExpr& dA,
	                                                                 const FormExpr& exponent);

    // ---- Smooth / regularized scalar functions ----
    [[nodiscard]] FormExpr smoothAbs(const FormExpr& eps) const;
    [[nodiscard]] FormExpr smoothSign(const FormExpr& eps) const;
    [[nodiscard]] FormExpr smoothHeaviside(const FormExpr& eps) const;
    [[nodiscard]] FormExpr smoothMin(const FormExpr& rhs, const FormExpr& eps) const;
    [[nodiscard]] FormExpr smoothMax(const FormExpr& rhs, const FormExpr& eps) const;

    // Symmetric eigenvalue access (2x2 and 3x3 only; eigenvalues sorted descending).
    [[nodiscard]] FormExpr symmetricEigenvalue(int which) const;

	    // General eigendecomposition accessors (initial support: symmetric 2x2/3x3).
	    [[nodiscard]] FormExpr eigenvalue(int which) const;
	    [[nodiscard]] FormExpr symmetricEigenvector(int which) const;
	    [[nodiscard]] FormExpr spectralDecomposition() const; // eigenvectors as columns

	    [[nodiscard]] static FormExpr symmetricEigenvectorDirectionalDerivative(const FormExpr& A,
	                                                                            const FormExpr& dA,
	                                                                            int which);
	    [[nodiscard]] static FormExpr spectralDecompositionDirectionalDerivative(const FormExpr& A,
	                                                                             const FormExpr& dA);

    // History operators (weights correspond to k=1..N, i.e. u^{n-1}, u^{n-2}, ...)
    [[nodiscard]] static FormExpr historyWeightedSum(std::vector<FormExpr> weights);
    [[nodiscard]] static FormExpr historyConvolution(std::vector<FormExpr> weights);

    // Internal helpers used by symbolic differentiation.
    [[nodiscard]] static FormExpr symmetricEigenvalueDirectionalDerivative(const FormExpr& A,
                                                                           const FormExpr& dA,
                                                                           int which);
    [[nodiscard]] static FormExpr symmetricEigenvalueDirectionalDerivativeWrtA(const FormExpr& A,
                                                                               const FormExpr& B,
                                                                               const FormExpr& dA,
                                                                               int which);

	    // ---- Measures ----
	    [[nodiscard]] FormExpr dx() const;
	    [[nodiscard]] FormExpr ds(int boundary_marker = -1) const;
	    [[nodiscard]] FormExpr dS() const;
	    [[nodiscard]] FormExpr dI(int interface_marker = -1) const;

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
inline FormExpr expm(const FormExpr& A) { return A.matrixExp(); }
inline FormExpr logm(const FormExpr& A) { return A.matrixLog(); }
inline FormExpr sqrtm(const FormExpr& A) { return A.matrixSqrt(); }
inline FormExpr powm(const FormExpr& A, const FormExpr& p) { return A.matrixPow(p); }
inline FormExpr smoothAbs(const FormExpr& a, const FormExpr& eps) { return a.smoothAbs(eps); }
inline FormExpr smoothSign(const FormExpr& a, const FormExpr& eps) { return a.smoothSign(eps); }
inline FormExpr smoothHeaviside(const FormExpr& a, const FormExpr& eps) { return a.smoothHeaviside(eps); }
inline FormExpr smoothMin(const FormExpr& a, const FormExpr& b, const FormExpr& eps) { return a.smoothMin(b, eps); }
inline FormExpr smoothMax(const FormExpr& a, const FormExpr& b, const FormExpr& eps) { return a.smoothMax(b, eps); }
inline FormExpr eigenvalue(const FormExpr& A, int which) { return A.eigenvalue(which); }
inline FormExpr eigvec_sym(const FormExpr& A, int which) { return A.symmetricEigenvector(which); }
inline FormExpr spectralDecomp(const FormExpr& A) { return A.spectralDecomposition(); }
inline FormExpr jump(const FormExpr& expr) { return expr.jump(); }
inline FormExpr avg(const FormExpr& expr) { return expr.avg(); }
inline FormExpr minus(const FormExpr& expr) { return expr.minus(); }
inline FormExpr plus(const FormExpr& expr) { return expr.plus(); }

} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_FORM_EXPR_H

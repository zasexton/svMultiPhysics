/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/JIT/JITValidation.h"

#include "Assembly/JIT/KernelArgs.h"
#include "Core/FEException.h"
#include "Forms/Tensor/LoopStructure.h"
#include "Forms/Tensor/TensorContraction.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace jit {

namespace {

[[nodiscard]] bool containsExternalCall(const FormExprNode& node)
{
    bool found = false;
    const auto visit = [&](const auto& self, const FormExprNode& n) -> void {
        if (n.type() == FormExprType::Coefficient || n.type() == FormExprType::Constitutive) {
            found = true;
            return;
        }
        for (const auto& child : n.childrenShared()) {
            if (child && !found) self(self, *child);
        }
    };
    visit(visit, node);
    return found;
}

[[nodiscard]] ValidationIssue issue(const FormExprNode& node, std::string message)
{
    ValidationIssue out;
    out.type = node.type();
    out.message = std::move(message);
    out.subexpr = node.toString();
    return out;
}

[[nodiscard]] std::string formatIndexList(const std::vector<forms::tensor::ContractionAnalysis::IndexInfo>& indices)
{
    std::string out;
    for (std::size_t k = 0; k < indices.size(); ++k) {
        if (k > 0) out += ", ";
        out += indices[k].name.empty() ? ("i" + std::to_string(indices[k].id)) : indices[k].name;
    }
    return out;
}

enum class ShapeKind : std::uint8_t { Unknown, Scalar, Vector, Matrix, Tensor3, Tensor4 };

struct Shape {
    ShapeKind kind{ShapeKind::Unknown};
    std::uint32_t d0{0};
    std::uint32_t d1{0};
    std::uint32_t d2{0};
    std::uint32_t d3{0};
};

[[nodiscard]] Shape scalarShape() noexcept { return Shape{ShapeKind::Scalar, 1u, 1u, 1u, 1u}; }
[[nodiscard]] Shape vectorShape(std::uint32_t n) noexcept { return Shape{ShapeKind::Vector, n, 1u, 1u, 1u}; }
[[nodiscard]] Shape matrixShape(std::uint32_t r, std::uint32_t c) noexcept { return Shape{ShapeKind::Matrix, r, c, 1u, 1u}; }
[[nodiscard]] Shape tensor3Shape(std::uint32_t a, std::uint32_t b, std::uint32_t c) noexcept { return Shape{ShapeKind::Tensor3, a, b, c, 1u}; }
[[nodiscard]] Shape tensor4Shape(std::uint32_t a, std::uint32_t b, std::uint32_t c, std::uint32_t d) noexcept
{
    return Shape{ShapeKind::Tensor4, a, b, c, d};
}

[[nodiscard]] bool sameShape(const Shape& a, const Shape& b) noexcept
{
    return a.kind == b.kind && a.d0 == b.d0 && a.d1 == b.d1 && a.d2 == b.d2 && a.d3 == b.d3;
}

[[nodiscard]] bool isScalarShape(const Shape& s) noexcept { return s.kind == ShapeKind::Scalar; }
[[nodiscard]] bool isVectorShape(const Shape& s) noexcept { return s.kind == ShapeKind::Vector; }
[[nodiscard]] bool isMatrixShape(const Shape& s) noexcept { return s.kind == ShapeKind::Matrix; }

struct ShapeContext {
    const FormExprNode::SpaceSignature* trial_sig{nullptr};
    std::unordered_map<const FormExprNode*, Shape> memo{};
};

[[nodiscard]] std::optional<double> constantScalarValue(const FormExprNode& n)
{
    if (n.type() != FormExprType::Constant) return std::nullopt;
    const auto v = n.constantValue();
    if (!v) return std::nullopt;
    return static_cast<double>(*v);
}

struct ConstantMatrix {
    std::uint32_t n{0u};
    std::array<double, 9> A{}; // row-major; only first n*n entries used
};

[[nodiscard]] std::optional<ConstantMatrix> constantSquareMatrix2or3(const FormExprNode& n)
{
    if (n.type() == FormExprType::Identity) {
        const int dim = n.identityDim().value_or(3);
        if (dim != 2 && dim != 3) return std::nullopt;
        ConstantMatrix out;
        out.n = static_cast<std::uint32_t>(dim);
        out.A.fill(0.0);
        for (std::uint32_t i = 0; i < out.n; ++i) {
            out.A[i * out.n + i] = 1.0;
        }
        return out;
    }

    if (n.type() != FormExprType::AsTensor) return std::nullopt;
    const auto rows_opt = n.tensorRows();
    const auto cols_opt = n.tensorCols();
    if (!rows_opt || !cols_opt) return std::nullopt;
    const int rows = *rows_opt;
    const int cols = *cols_opt;
    if (rows != cols || (rows != 2 && rows != 3)) return std::nullopt;

    const auto kids = n.childrenShared();
    if (kids.size() != static_cast<std::size_t>(rows * cols)) return std::nullopt;

    ConstantMatrix out;
    out.n = static_cast<std::uint32_t>(rows);
    out.A.fill(0.0);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const auto k = static_cast<std::size_t>(r * cols + c);
            if (!kids[k]) return std::nullopt;
            const auto v = constantScalarValue(*kids[k]);
            if (!v) return std::nullopt;
            out.A[static_cast<std::size_t>(r * cols + c)] = *v;
        }
    }
    return out;
}

[[nodiscard]] bool isSPDConstantMatrix(const ConstantMatrix& M) noexcept
{
    const std::uint32_t n = M.n;
    if (n != 2u && n != 3u) return false;

    std::array<double, 9> A = M.A;
    // Symmetrize.
    for (std::uint32_t r = 0; r < n; ++r) {
        for (std::uint32_t c = r + 1u; c < n; ++c) {
            const double s = 0.5 * (A[r * n + c] + A[c * n + r]);
            A[r * n + c] = s;
            A[c * n + r] = s;
        }
    }

    for (std::uint32_t i = 0; i < n * n; ++i) {
        if (!std::isfinite(A[i])) return false;
    }

    if (n == 2u) {
        const double a00 = A[0];
        const double a01 = A[1];
        const double a11 = A[3];
        const double det = a00 * a11 - a01 * a01;
        return (a00 > 0.0) && (det > 0.0);
    }

    const double a00 = A[0];
    const double a01 = A[1];
    const double a02 = A[2];
    const double a11 = A[4];
    const double a12 = A[5];
    const double a22 = A[8];

    const double m1 = a00;
    const double m2 = a00 * a11 - a01 * a01;
    const double det = a00 * (a11 * a22 - a12 * a12) -
                       a01 * (a01 * a22 - a12 * a02) +
                       a02 * (a01 * a12 - a11 * a02);

    return (m1 > 0.0) && (m2 > 0.0) && (det > 0.0);
}

[[nodiscard]] Shape inferShape(const FormExprNode& node, ShapeContext& ctx)
{
    if (auto it = ctx.memo.find(&node); it != ctx.memo.end()) {
        return it->second;
    }

    const auto kids = node.childrenShared();
    auto childShape = [&](std::size_t idx) -> Shape {
        if (idx >= kids.size() || !kids[idx]) return Shape{};
        return inferShape(*kids[idx], ctx);
    };

    Shape out{};

    switch (node.type()) {
        // Scalars / scalar-like terminals
        case FormExprType::Constant:
        case FormExprType::ParameterSymbol:
        case FormExprType::ParameterRef:
        case FormExprType::BoundaryIntegralSymbol:
        case FormExprType::BoundaryIntegralRef:
        case FormExprType::AuxiliaryStateSymbol:
        case FormExprType::AuxiliaryStateRef:
        case FormExprType::MaterialStateOldRef:
        case FormExprType::MaterialStateWorkRef:
        case FormExprType::Time:
        case FormExprType::TimeStep:
        case FormExprType::EffectiveTimeStep:
        case FormExprType::JacobianDeterminant:
        case FormExprType::CellDiameter:
        case FormExprType::CellVolume:
        case FormExprType::FacetArea:
        case FormExprType::CellDomainId:
        case FormExprType::Less:
        case FormExprType::LessEqual:
        case FormExprType::Greater:
        case FormExprType::GreaterEqual:
        case FormExprType::Equal:
        case FormExprType::NotEqual:
            out = scalarShape();
            break;

        case FormExprType::Coordinate:
        case FormExprType::ReferenceCoordinate:
        case FormExprType::Normal:
            out = vectorShape(3u);
            break;

        case FormExprType::Jacobian:
        case FormExprType::JacobianInverse:
            out = matrixShape(3u, 3u);
            break;

        case FormExprType::Identity: {
            const int dim = node.identityDim().value_or(3);
            out = matrixShape(static_cast<std::uint32_t>(std::max(1, dim)),
                              static_cast<std::uint32_t>(std::max(1, dim)));
            break;
        }

        case FormExprType::Coefficient: {
            if (node.scalarCoefficient() != nullptr || node.timeScalarCoefficient() != nullptr) {
                out = scalarShape();
            } else if (node.vectorCoefficient() != nullptr) {
                out = vectorShape(3u);
            } else if (node.matrixCoefficient() != nullptr) {
                out = matrixShape(3u, 3u);
            } else if (node.tensor3Coefficient() != nullptr) {
                out = tensor3Shape(3u, 3u, 3u);
            } else if (node.tensor4Coefficient() != nullptr) {
                out = tensor4Shape(3u, 3u, 3u, 3u);
            }
            break;
        }

        case FormExprType::TestFunction:
        case FormExprType::TrialFunction:
        case FormExprType::DiscreteField:
        case FormExprType::StateField: {
            const auto* sig = node.spaceSignature();
            if (!sig) break;
            const std::uint32_t vd = static_cast<std::uint32_t>(std::max(1, sig->value_dimension));
            const std::uint32_t td = static_cast<std::uint32_t>(std::max(1, sig->topological_dimension));
            switch (sig->field_type) {
                case FieldType::Scalar:
                    out = scalarShape();
                    break;
                case FieldType::Vector:
                    out = vectorShape(vd);
                    break;
                case FieldType::Tensor:
                case FieldType::SymmetricTensor:
                    if (vd == 4u) out = matrixShape(2u, 2u);
                    else if (vd == 9u) out = matrixShape(3u, 3u);
                    else if (td == 2u || td == 3u) out = matrixShape(td, td);
                    break;
                default:
                    break;
            }
            break;
        }

        case FormExprType::PreviousSolutionRef: {
            if (ctx.trial_sig == nullptr) {
                break;
            }
            const auto& sig = *ctx.trial_sig;
            const std::uint32_t vd = static_cast<std::uint32_t>(std::max(1, sig.value_dimension));
            switch (sig.field_type) {
                case FieldType::Scalar:
                    out = scalarShape();
                    break;
                case FieldType::Vector:
                    out = vectorShape(vd);
                    break;
                default:
                    break;
            }
            break;
        }

        case FormExprType::HistoryWeightedSum:
        case FormExprType::HistoryConvolution: {
            // History operators return the active trial field shape (scalar or vector).
            if (ctx.trial_sig == nullptr) {
                break;
            }
            const auto& sig = *ctx.trial_sig;
            const std::uint32_t vd = static_cast<std::uint32_t>(std::max(1, sig.value_dimension));
            switch (sig.field_type) {
                case FieldType::Scalar:
                    out = scalarShape();
                    break;
                case FieldType::Vector:
                    out = vectorShape(vd);
                    break;
                default:
                    break;
            }
            break;
        }

        // Packing
        case FormExprType::AsVector: {
            if (kids.empty()) break;
            bool all_scalar = true;
            for (std::size_t k = 0; k < kids.size(); ++k) {
                const auto s = childShape(k);
                if (!isScalarShape(s)) {
                    all_scalar = false;
                    break;
                }
            }
            if (all_scalar) {
                out = vectorShape(static_cast<std::uint32_t>(kids.size()));
            }
            break;
        }
        case FormExprType::AsTensor: {
            const auto r = node.tensorRows().value_or(0);
            const auto c = node.tensorCols().value_or(0);
            if (r > 0 && c > 0) {
                out = matrixShape(static_cast<std::uint32_t>(r), static_cast<std::uint32_t>(c));
            }
            break;
        }

        // Indexing
        case FormExprType::Component:
        case FormExprType::IndexedAccess:
            out = scalarShape();
            break;

        // Unary ops with shape propagation
        case FormExprType::Negate:
        case FormExprType::RestrictMinus:
        case FormExprType::RestrictPlus:
        case FormExprType::Jump:
        case FormExprType::Average:
        case FormExprType::TimeDerivative:
            out = childShape(0);
            break;

        case FormExprType::Transpose: {
            const auto a = childShape(0);
            if (a.kind == ShapeKind::Matrix) {
                out = matrixShape(a.d1, a.d0);
            }
            break;
        }
        case FormExprType::Trace:
        case FormExprType::Determinant:
        case FormExprType::Norm:
            out = scalarShape();
            break;

        case FormExprType::Inverse:
        case FormExprType::Cofactor:
        case FormExprType::Deviator:
        case FormExprType::SymmetricPart:
        case FormExprType::SkewPart:
            out = childShape(0);
            break;

        case FormExprType::Gradient: {
            const auto a = childShape(0);
            const auto* sig = kids.size() > 0 && kids[0] ? kids[0]->spaceSignature() : nullptr;
            const std::uint32_t dim = sig ? static_cast<std::uint32_t>(std::max(1, sig->topological_dimension)) : 3u;
            if (a.kind == ShapeKind::Scalar) {
                out = vectorShape(dim);
            } else if (a.kind == ShapeKind::Vector) {
                out = matrixShape(a.d0, dim);
            } else if (a.kind == ShapeKind::Matrix) {
                out = tensor3Shape(a.d0, a.d1, dim);
            }
            break;
        }
        case FormExprType::Divergence: {
            const auto a = childShape(0);
            if (a.kind == ShapeKind::Vector) {
                out = scalarShape();
            } else if (a.kind == ShapeKind::Matrix) {
                out = vectorShape(a.d0);
            }
            break;
        }
        case FormExprType::Curl: {
            const auto a = childShape(0);
            if (a.kind == ShapeKind::Vector) {
                out = vectorShape(3u);
            }
            break;
        }
        case FormExprType::Hessian: {
            const auto a = childShape(0);
            const auto* sig = kids.size() > 0 && kids[0] ? kids[0]->spaceSignature() : nullptr;
            const std::uint32_t dim = sig ? static_cast<std::uint32_t>(std::max(1, sig->topological_dimension)) : 3u;
            if (a.kind == ShapeKind::Scalar) {
                out = matrixShape(dim, dim);
            }
            break;
        }

        // Algebra
        case FormExprType::Add:
        case FormExprType::Subtract: {
            const auto a = childShape(0);
            const auto b = childShape(1);
            if (sameShape(a, b)) out = a;
            break;
        }
        case FormExprType::Multiply: {
            const auto a = childShape(0);
            const auto b = childShape(1);
            if (a.kind == ShapeKind::Scalar && b.kind == ShapeKind::Scalar) {
                out = scalarShape();
            } else if (a.kind == ShapeKind::Scalar && b.kind != ShapeKind::Unknown) {
                out = b;
            } else if (b.kind == ShapeKind::Scalar && a.kind != ShapeKind::Unknown) {
                out = a;
            } else if (a.kind == ShapeKind::Matrix && b.kind == ShapeKind::Vector && a.d1 == b.d0) {
                out = vectorShape(a.d0);
            } else if (a.kind == ShapeKind::Vector && b.kind == ShapeKind::Matrix && a.d0 == b.d0) {
                out = vectorShape(b.d1);
            } else if (a.kind == ShapeKind::Matrix && b.kind == ShapeKind::Matrix && a.d1 == b.d0) {
                out = matrixShape(a.d0, b.d1);
            }
            break;
        }
        case FormExprType::Divide: {
            const auto a = childShape(0);
            const auto b = childShape(1);
            if (b.kind == ShapeKind::Scalar) {
                out = a;
            }
            break;
        }
        case FormExprType::InnerProduct:
        case FormExprType::DoubleContraction:
        case FormExprType::Power:
        case FormExprType::Minimum:
        case FormExprType::Maximum:
        case FormExprType::SmoothHeaviside:
        case FormExprType::SmoothAbsoluteValue:
        case FormExprType::SmoothMin:
        case FormExprType::SmoothMax:
        case FormExprType::SmoothSign:
            out = scalarShape();
            break;

        case FormExprType::OuterProduct: {
            const auto a = childShape(0);
            const auto b = childShape(1);
            if (a.kind == ShapeKind::Vector && b.kind == ShapeKind::Vector) {
                out = matrixShape(a.d0, b.d0);
            }
            break;
        }
        case FormExprType::CrossProduct:
            out = vectorShape(3u);
            break;

        case FormExprType::Conditional: {
            const auto t = childShape(1);
            const auto f = childShape(2);
            if (sameShape(t, f)) out = t;
            break;
        }

        // Matrix functions / eigen ops (output shapes)
        case FormExprType::MatrixExponential:
        case FormExprType::MatrixLogarithm:
        case FormExprType::MatrixSqrt:
        case FormExprType::MatrixExponentialDirectionalDerivative:
        case FormExprType::MatrixLogarithmDirectionalDerivative:
        case FormExprType::MatrixSqrtDirectionalDerivative:
        case FormExprType::MatrixPowerDirectionalDerivative:
        case FormExprType::SpectralDecomposition:
        case FormExprType::SpectralDecompositionDirectionalDerivative: {
            out = childShape(0);
            break;
        }
        case FormExprType::MatrixPower: {
            out = childShape(0);
            break;
        }
        case FormExprType::SymmetricEigenvector:
        case FormExprType::SymmetricEigenvectorDirectionalDerivative: {
            const auto a = childShape(0);
            if (a.kind == ShapeKind::Matrix) {
                out = vectorShape(a.d0);
            }
            break;
        }
        case FormExprType::SymmetricEigenvalue:
        case FormExprType::SymmetricEigenvalueDirectionalDerivative:
        case FormExprType::SymmetricEigenvalueDirectionalDerivativeWrtA:
        case FormExprType::Eigenvalue:
            out = scalarShape();
            break;

        default:
            break;
    }

    ctx.memo.emplace(&node, out);
    return out;
}

[[nodiscard]] bool isSquare2or3(const Shape& s) noexcept
{
    if (!isMatrixShape(s)) return false;
    if (s.d0 != s.d1) return false;
    return (s.d0 == 2u) || (s.d0 == 3u);
}

[[nodiscard]] bool eigenIndexInRange(const FormExprNode& n, std::uint32_t dim) noexcept
{
    const int which = n.eigenIndex().value_or(0);
    return (which >= 0) && (static_cast<std::uint32_t>(which) < dim);
}

[[nodiscard]] bool needsSPDCheck(FormExprType t, bool exponent_is_zero) noexcept
{
    switch (t) {
        case FormExprType::MatrixLogarithm:
        case FormExprType::MatrixSqrt:
        case FormExprType::MatrixLogarithmDirectionalDerivative:
        case FormExprType::MatrixSqrtDirectionalDerivative:
            return true;
        case FormExprType::MatrixPower:
        case FormExprType::MatrixPowerDirectionalDerivative:
            return !exponent_is_zero;
        default:
            return false;
    }
}

} // namespace

ValidationResult canCompileImpl(const FormExpr& integrand,
                                const ValidationOptions& options,
                                const FormExprNode::SpaceSignature* trial_sig)
{
    ValidationResult out;
    out.ok = true;
    out.cacheable = true;

    if (!integrand.isValid() || integrand.node() == nullptr) {
        out.ok = false;
        out.first_issue = ValidationIssue{
            .type = FormExprType::Constant,
            .message = "forms::jit::canCompile: invalid expression",
            .subexpr = {},
        };
        return out;
    }

    const auto& root = *integrand.node();
    bool saw_indexed_access = false;

    const auto visit = [&](const auto& self, const FormExprNode& n) -> void {
        if (!out.ok) return;

        // Name-based terminals must be resolved to slot refs before JIT lowering.
        switch (n.type()) {
            case FormExprType::ParameterSymbol:
                out.ok = false;
                out.first_issue = issue(n, "JIT: ParameterSymbol must be resolved to ParameterRef(slot)");
                return;
            case FormExprType::BoundaryFunctionalSymbol:
            case FormExprType::BoundaryIntegralSymbol:
            case FormExprType::AuxiliaryStateSymbol:
                out.ok = false;
                out.first_issue = issue(n, "JIT: coupled placeholder must be resolved to slot-based refs");
                return;

            // Measure wrappers are handled by FormCompiler -> FormIR decomposition, not by kernel lowering.
            case FormExprType::CellIntegral:
            case FormExprType::BoundaryIntegral:
            case FormExprType::InteriorFaceIntegral:
            case FormExprType::InterfaceIntegral:
                out.ok = false;
                out.first_issue = issue(n, "JIT: measure nodes (dx/ds/dS/dI) are not valid in integrands");
                return;

            case FormExprType::IndexedAccess:
                saw_indexed_access = true;
                break;

            default:
                break;
        }

        if (n.type() == FormExprType::Coefficient) {
            if (options.strictness == Strictness::Strict) {
                out.ok = false;
                out.first_issue = issue(n, "JIT(strict): Coefficient nodes are not allowed (runtime callback)");
                return;
            }
            out.cacheable = false;
        }

        if (n.type() == FormExprType::Constitutive) {
            if (options.strictness == Strictness::Strict) {
                out.ok = false;
                out.first_issue = issue(n, "JIT(strict): Constitutive calls are not allowed (virtual dispatch)");
                return;
            }
            out.cacheable = false;
        }

        // Spatial derivatives of external-call coefficients require explicit derivatives.
        if (n.type() == FormExprType::Gradient ||
            n.type() == FormExprType::Divergence ||
            n.type() == FormExprType::Curl ||
            n.type() == FormExprType::Hessian) {
            const auto kids = n.childrenShared();
            if (!kids.empty() && kids[0] && containsExternalCall(*kids[0])) {
                out.ok = false;
                out.first_issue = issue(n, "JIT: derivative of external-call coefficient/model is not supported; provide explicit derivative expressions");
                return;
            }
        }

        for (const auto& child : n.childrenShared()) {
            if (child) self(self, *child);
        }
    };

    visit(visit, root);

    if (!out.ok) return out;

    // Shape/domain validation for new-physics vocabulary extensions (Section 10.1).
    ShapeContext shapes;
    shapes.trial_sig = trial_sig;
    (void)inferShape(root, shapes);

    const auto validate_shapes = [&](const auto& self, const FormExprNode& n) -> void {
        if (!out.ok) return;

        const auto nshape = inferShape(n, shapes);

        const auto fail = [&](std::string msg) -> void {
            out.ok = false;
            out.first_issue = issue(n, std::move(msg));
        };

        const auto child0shape = [&]() -> Shape {
            const auto kids = n.childrenShared();
            if (kids.empty() || !kids[0]) return Shape{};
            return inferShape(*kids[0], shapes);
        };
        const auto child1shape = [&]() -> Shape {
            const auto kids = n.childrenShared();
            if (kids.size() < 2u || !kids[1]) return Shape{};
            return inferShape(*kids[1], shapes);
        };

        switch (n.type()) {
            case FormExprType::TimeDerivative: {
                const int order = n.timeDerivativeOrder().value_or(1);
                if (order < 1 || static_cast<std::size_t>(order) > assembly::jit::kMaxTimeDerivativeOrderV4) {
                    fail("JIT: dt(order) expects 1 <= order <= " +
                         std::to_string(assembly::jit::kMaxTimeDerivativeOrderV4));
                }
                break;
            }

            case FormExprType::PreviousSolutionRef: {
                const int steps = n.historyIndex().value_or(1);
                if (steps < 1 || static_cast<std::size_t>(steps) > assembly::jit::kMaxPreviousSolutionsV4) {
                    fail("JIT: PreviousSolutionRef expects 1 <= steps_back <= " +
                         std::to_string(assembly::jit::kMaxPreviousSolutionsV4));
                }
                break;
            }

            case FormExprType::SmoothAbsoluteValue:
            case FormExprType::SmoothSign:
            case FormExprType::SmoothHeaviside: {
                const auto x = child0shape();
                const auto eps = child1shape();
                if (!isScalarShape(x) || !isScalarShape(eps)) {
                    fail("JIT: smooth unary op expects scalar inputs");
                }
                break;
            }

            case FormExprType::SmoothMin:
            case FormExprType::SmoothMax: {
                const auto kids = n.childrenShared();
                if (kids.size() != 3u || !kids[0] || !kids[1] || !kids[2]) {
                    fail("JIT: smooth min/max expects 3 children");
                    break;
                }
                const auto a = inferShape(*kids[0], shapes);
                const auto b = inferShape(*kids[1], shapes);
                const auto eps = inferShape(*kids[2], shapes);
                if (!isScalarShape(a) || !isScalarShape(b) || !isScalarShape(eps)) {
                    fail("JIT: smooth min/max expects scalar inputs");
                }
                break;
            }

            case FormExprType::MatrixExponential:
            case FormExprType::MatrixLogarithm:
            case FormExprType::MatrixSqrt: {
                const auto A = child0shape();
                if (!isSquare2or3(A)) {
                    fail("JIT: matrix function expects a 2x2 or 3x3 square matrix");
                    break;
                }
                if (needsSPDCheck(n.type(), /*exponent_is_zero=*/false)) {
                    const auto kids = n.childrenShared();
                    if (!kids.empty() && kids[0]) {
                        if (const auto cm = constantSquareMatrix2or3(*kids[0])) {
                            if (!isSPDConstantMatrix(*cm)) {
                                fail("JIT: matrix function requires SPD matrix input");
                            }
                        }
                    }
                }
                break;
            }

            case FormExprType::MatrixPower: {
                const auto kids = n.childrenShared();
                if (kids.size() != 2u || !kids[0] || !kids[1]) {
                    fail("JIT: matrix_pow expects 2 children");
                    break;
                }
                const auto A = inferShape(*kids[0], shapes);
                const auto p = inferShape(*kids[1], shapes);
                if (!isSquare2or3(A) || !isScalarShape(p)) {
                    fail("JIT: matrix_pow expects (2x2|3x3 matrix, scalar exponent)");
                    break;
                }
                bool p_is_zero = false;
                if (const auto pv = constantScalarValue(*kids[1])) {
                    p_is_zero = (*pv == 0.0);
                }
                if (needsSPDCheck(n.type(), p_is_zero)) {
                    if (const auto cm = constantSquareMatrix2or3(*kids[0])) {
                        if (!isSPDConstantMatrix(*cm)) {
                            fail("JIT: matrix_pow requires SPD matrix input for p != 0");
                        }
                    }
                }
                (void)nshape;
                break;
            }

            case FormExprType::MatrixExponentialDirectionalDerivative:
            case FormExprType::MatrixLogarithmDirectionalDerivative:
            case FormExprType::MatrixSqrtDirectionalDerivative: {
                const auto kids = n.childrenShared();
                if (kids.size() != 2u || !kids[0] || !kids[1]) {
                    fail("JIT: matrix dd expects 2 children");
                    break;
                }
                const auto A = inferShape(*kids[0], shapes);
                const auto dA = inferShape(*kids[1], shapes);
                if (!isSquare2or3(A) || !sameShape(A, dA)) {
                    fail("JIT: matrix dd expects two 2x2/3x3 matrices with matching dims");
                    break;
                }
                if (needsSPDCheck(n.type(), /*exponent_is_zero=*/false)) {
                    if (const auto cm = constantSquareMatrix2or3(*kids[0])) {
                        if (!isSPDConstantMatrix(*cm)) {
                            fail("JIT: matrix dd requires SPD matrix input");
                        }
                    }
                }
                break;
            }

            case FormExprType::MatrixPowerDirectionalDerivative: {
                const auto kids = n.childrenShared();
                if (kids.size() != 3u || !kids[0] || !kids[1] || !kids[2]) {
                    fail("JIT: matrix_pow_dd expects 3 children");
                    break;
                }
                const auto A = inferShape(*kids[0], shapes);
                const auto dA = inferShape(*kids[1], shapes);
                const auto p = inferShape(*kids[2], shapes);
                if (!isSquare2or3(A) || !sameShape(A, dA) || !isScalarShape(p)) {
                    fail("JIT: matrix_pow_dd expects (matrix, matrix, scalar)");
                    break;
                }
                bool p_is_zero = false;
                if (const auto pv = constantScalarValue(*kids[2])) {
                    p_is_zero = (*pv == 0.0);
                }
                if (needsSPDCheck(n.type(), p_is_zero)) {
                    if (const auto cm = constantSquareMatrix2or3(*kids[0])) {
                        if (!isSPDConstantMatrix(*cm)) {
                            fail("JIT: matrix_pow_dd requires SPD matrix input for p != 0");
                        }
                    }
                }
                break;
            }

            case FormExprType::SymmetricEigenvalue:
            case FormExprType::Eigenvalue: {
                const auto A = child0shape();
                if (!isSquare2or3(A)) {
                    fail("JIT: eigenvalue expects a 2x2 or 3x3 square matrix");
                    break;
                }
                if (!eigenIndexInRange(n, A.d0)) {
                    fail("JIT: eigenvalue index out of range");
                }
                break;
            }

            case FormExprType::SymmetricEigenvector: {
                const auto A = child0shape();
                if (!isSquare2or3(A)) {
                    fail("JIT: eigvec_sym expects a 2x2 or 3x3 square matrix");
                    break;
                }
                if (!eigenIndexInRange(n, A.d0)) {
                    fail("JIT: eigvec_sym index out of range");
                }
                break;
            }

            case FormExprType::SymmetricEigenvectorDirectionalDerivative: {
                const auto kids = n.childrenShared();
                if (kids.size() != 2u || !kids[0] || !kids[1]) {
                    fail("JIT: eigvec_sym_dd expects 2 children");
                    break;
                }
                const auto A = inferShape(*kids[0], shapes);
                const auto dA = inferShape(*kids[1], shapes);
                if (!isSquare2or3(A) || !sameShape(A, dA)) {
                    fail("JIT: eigvec_sym_dd expects two 2x2/3x3 matrices with matching dims");
                    break;
                }
                if (!eigenIndexInRange(n, A.d0)) {
                    fail("JIT: eigvec_sym_dd index out of range");
                }
                break;
            }

            case FormExprType::SpectralDecomposition: {
                const auto A = child0shape();
                if (!isSquare2or3(A)) {
                    fail("JIT: spectral_decomp expects a 2x2 or 3x3 square matrix");
                }
                break;
            }

            case FormExprType::SpectralDecompositionDirectionalDerivative: {
                const auto kids = n.childrenShared();
                if (kids.size() != 2u || !kids[0] || !kids[1]) {
                    fail("JIT: spectral_decomp_dd expects 2 children");
                    break;
                }
                const auto A = inferShape(*kids[0], shapes);
                const auto dA = inferShape(*kids[1], shapes);
                if (!isSquare2or3(A) || !sameShape(A, dA)) {
                    fail("JIT: spectral_decomp_dd expects two 2x2/3x3 matrices with matching dims");
                }
                break;
            }

            case FormExprType::HistoryWeightedSum:
            case FormExprType::HistoryConvolution: {
                const auto kids = n.childrenShared();
                if (kids.size() > assembly::jit::kMaxPreviousSolutionsV4) {
                    fail("JIT: history operator weights length must be <= " +
                         std::to_string(assembly::jit::kMaxPreviousSolutionsV4));
                    break;
                }
                for (const auto& w : kids) {
                    if (!w) continue;
                    const auto ws = inferShape(*w, shapes);
                    if (!isScalarShape(ws)) {
                        fail("JIT: history operator expects scalar weights");
                        break;
                    }
                }
                break;
            }

            default:
                break;
        }

        for (const auto& child : n.childrenShared()) {
            if (child) self(self, *child);
        }
    };

    validate_shapes(validate_shapes, root);

    if (out.ok && saw_indexed_access) {
        const auto a = forms::tensor::analyzeContractions(integrand);
        if (!a.ok) {
            out.ok = false;
            out.first_issue = ValidationIssue{
                .type = FormExprType::IndexedAccess,
                .message = "JIT: invalid Einstein-index usage: " + a.message,
                .subexpr = integrand.toString(),
            };
            return out;
        }
        if (!a.free_indices.empty()) {
            out.ok = false;
            out.first_issue = ValidationIssue{
                .type = FormExprType::IndexedAccess,
                .message = "JIT: result has free indices {" + formatIndexList(a.free_indices) +
                           "} but expected scalar output (integrand must be fully contracted)",
                .subexpr = integrand.toString(),
            };
            return out;
        }

        // Validate that tensor calculus can be lowered either to loop-nest IR or
        // to a scalar-expanded fallback (einsum). This mirrors the JIT lowering
        // decision and produces clearer diagnostics for unsupported patterns.
        try {
            const auto lr = forms::tensor::lowerTensorExpressionIncremental(integrand);
            if (!lr.ok) {
                out.ok = false;
                out.first_issue = ValidationIssue{
                    .type = FormExprType::IndexedAccess,
                    .message = "JIT: tensor/loop lowering failed: " + (lr.message.empty() ? std::string{"unknown error"} : lr.message),
                    .subexpr = integrand.toString(),
                };
                return out;
            }
        } catch (const std::exception& e) {
            out.ok = false;
            out.first_issue = ValidationIssue{
                .type = FormExprType::IndexedAccess,
                .message = std::string("JIT: tensor/loop lowering threw exception: ") + e.what(),
                .subexpr = integrand.toString(),
            };
            return out;
        }
    }

    return out;
}

ValidationResult canCompile(const FormExpr& integrand, const ValidationOptions& options)
{
    return canCompileImpl(integrand, options, nullptr);
}

ValidationResult canCompile(const FormIR& ir, const ValidationOptions& options)
{
    ValidationResult out;
    out.ok = true;
    out.cacheable = true;

    if (!ir.isCompiled()) {
        out.ok = false;
        out.first_issue = ValidationIssue{
            .type = FormExprType::Constant,
            .message = "forms::jit::canCompile(FormIR): FormIR is not compiled",
            .subexpr = {},
        };
        return out;
    }

    for (const auto& term : ir.terms()) {
        if (!term.integrand.isValid() || term.integrand.node() == nullptr) {
            out.ok = false;
            out.first_issue = ValidationIssue{
                .type = FormExprType::Constant,
                .message = "forms::jit::canCompile(FormIR): term integrand is invalid",
                .subexpr = term.debug_string,
            };
            return out;
        }

        const auto& trial_opt = ir.trialSpace();
        const FormExprNode::SpaceSignature* trial_sig = trial_opt ? &*trial_opt : nullptr;
        auto r = canCompileImpl(term.integrand, options, trial_sig);
        if (!r.ok) {
            out.ok = false;
            out.cacheable = out.cacheable && r.cacheable;
            out.first_issue = std::move(r.first_issue);
            return out;
        }
        out.cacheable = out.cacheable && r.cacheable;
    }

    return out;
}

} // namespace jit
} // namespace forms
} // namespace FE
} // namespace svmp

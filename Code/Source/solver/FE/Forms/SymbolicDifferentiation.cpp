/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/SymbolicDifferentiation.h"

#include "Core/Types.h"
#include "Forms/IndexExtent.h"
#include "Forms/Tensor/TensorDifferentiation.h"
#include <algorithm>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {

namespace {

[[nodiscard]] SymbolicDiffIssue issue(const FormExprNode& node, std::string message)
{
    SymbolicDiffIssue out;
    out.type = node.type();
    out.message = std::move(message);
    out.subexpr = node.toString();
    return out;
}

[[nodiscard]] FormExpr wrapExteriorIntegral(const FormExpr& integrand,
                                            const FormExprNode& integral_node)
{
    if (const auto* measure = integral_node.exteriorBoundaryMeasure()) {
        return integrand.dExteriorBoundary(*measure);
    }
    if (integral_node.type() == FormExprType::BoundaryIntegral) {
        return integrand.ds(integral_node.boundaryMarker().value_or(-1));
    }
    if (integral_node.type() == FormExprType::InterfaceIntegral) {
        return integrand.dI(integral_node.interfaceMarker().value_or(-1));
    }
    throw std::invalid_argument(
        "wrapExteriorIntegral: node is not an exterior-boundary integral");
}

[[nodiscard]] bool holdsTwoSidedInterfaceOperator(const FormExprNode& node)
{
    switch (node.type()) {
        case FormExprType::RestrictPlus:
        case FormExprType::Jump:
        case FormExprType::Average:
            return true;
        default:
            break;
    }
    for (const auto* child : node.children()) {
        if (child != nullptr && holdsTwoSidedInterfaceOperator(*child)) {
            return true;
        }
    }
    return false;
}

/// The signed summands of an integrand, split as FormCompiler splits it into
/// terms (Add, Subtract, Negate).
void collectSignedSummands(const std::shared_ptr<FormExprNode>& node,
                           bool negated,
                           std::vector<std::pair<std::shared_ptr<FormExprNode>, bool>>& out)
{
    const auto kids = node->childrenShared();
    switch (node->type()) {
        case FormExprType::Add:
            if (kids.size() == 2u && kids[0] && kids[1]) {
                collectSignedSummands(kids[0], negated, out);
                collectSignedSummands(kids[1], negated, out);
                return;
            }
            break;
        case FormExprType::Subtract:
            if (kids.size() == 2u && kids[0] && kids[1]) {
                collectSignedSummands(kids[0], negated, out);
                collectSignedSummands(kids[1], !negated, out);
                return;
            }
            break;
        case FormExprType::Negate:
            if (kids.size() == 1u && kids[0]) {
                collectSignedSummands(kids[0], !negated, out);
                return;
            }
            break;
        default:
            break;
    }
    out.emplace_back(node, negated);
}

[[nodiscard]] bool isZeroLike(const FormExprNode& node);

/// The derivative of an interface integral.  A term of a generated interface
/// is one-sided unless it holds .plus(), jump() or avg() (D70), and a
/// derivative can lose the plus operand of its source term; then the
/// derivative of each two-sided summand is integrated by its own measure
/// stating it two-sided.  Otherwise (no derivative summand of a two-sided
/// summand without a two-sided operator, or a measure that states it
/// already) the derivative is the one it was.
template<class DerivativeOf>
[[nodiscard]] FormExpr interfaceIntegralDerivative(const std::shared_ptr<FormExprNode>& integrand,
                                                   const FormExprNode& integral_node,
                                                   const FormExpr& whole_derivative,
                                                   const DerivativeOf& derivative_of)
{
    const auto wrap = [&](const FormExpr& d) {
        auto out = wrapExteriorIntegral(d, integral_node);
        if (integral_node.interfaceStatesTwoSided() && out.node() != nullptr &&
            out.node()->type() == FormExprType::InterfaceIntegral) {
            out = out.withTwoSidedTerms();
        }
        return out;
    };
    if (integral_node.interfaceStatesTwoSided() || integral_node.exteriorBoundaryMeasure() != nullptr ||
        !holdsTwoSidedInterfaceOperator(*integrand)) {
        return wrap(whole_derivative);
    }
    std::vector<std::pair<std::shared_ptr<FormExprNode>, bool>> summands;
    collectSignedSummands(integrand, /*negated=*/false, summands);
    const auto loses_plus_operand = [&](const std::shared_ptr<FormExprNode>& summand) {
        if (!holdsTwoSidedInterfaceOperator(*summand)) {
            return false;
        }
        const auto d = derivative_of(summand);
        if (!d.isValid() || d.node() == nullptr) {
            return false;
        }
        std::vector<std::pair<std::shared_ptr<FormExprNode>, bool>> d_summands;
        collectSignedSummands(d.nodeShared(), /*negated=*/false, d_summands);
        return std::any_of(d_summands.begin(), d_summands.end(), [](const auto& d_summand) {
            return !isZeroLike(*d_summand.first) && !holdsTwoSidedInterfaceOperator(*d_summand.first);
        });
    };
    if (std::none_of(summands.begin(), summands.end(),
                     [&](const auto& summand) { return loses_plus_operand(summand.first); })) {
        return wrap(whole_derivative);
    }
    FormExpr one_sided;
    FormExpr out;
    const auto add = [](FormExpr& sum, const FormExpr& e) { sum = sum.isValid() ? sum + e : e; };
    for (const auto& [summand, negated] : summands) {
        auto d = derivative_of(summand);
        if (negated) {
            d = -d;
        }
        if (holdsTwoSidedInterfaceOperator(*summand)) {
            add(out, wrap(d).withTwoSidedTerms());
        } else {
            add(one_sided, d);
        }
    }
    if (one_sided.isValid()) {
        add(out, wrap(one_sided));
    }
    return out;
}

[[nodiscard]] FormExpr wrapInteriorFaceIntegral(const FormExpr& integrand,
                                                const FormExprNode& integral_node)
{
    if (integral_node.type() != FormExprType::InteriorFaceIntegral) {
        throw std::invalid_argument(
            "wrapInteriorFaceIntegral: node is not an interior-face integral");
    }
    const int marker = integral_node.interfaceMarker().value_or(-1);
    if (const auto side = integral_node.cutVolumeSide(); side.has_value()) {
        return integrand.dS(marker, *side);
    }
    return integrand.dS(marker);
}

[[nodiscard]] bool isScalarConstantValue(const FormExprNode& node, Real value)
{
    if (node.type() != FormExprType::Constant) return false;
    const auto v = node.constantValue();
    if (!v.has_value()) return false;
    return *v == value;
}

[[nodiscard]] bool isScalarZero(const FormExprNode& node) { return isScalarConstantValue(node, Real(0.0)); }
[[nodiscard]] bool isScalarOne(const FormExprNode& node) { return isScalarConstantValue(node, Real(1.0)); }

[[nodiscard]] bool isZeroLike(const FormExprNode& node)
{
    if (isScalarZero(node)) return true;
    if (node.type() == FormExprType::TypedZero) return true;
    if (node.type() == FormExprType::Multiply) {
        const auto kids = node.childrenShared();
        if (kids.size() != 2u || !kids[0] || !kids[1]) return false;
        return (isZeroLike(*kids[0]) || isZeroLike(*kids[1]));
    }
    if (node.type() == FormExprType::Negate) {
        const auto kids = node.childrenShared();
        if (kids.size() != 1u || !kids[0]) return false;
        return isZeroLike(*kids[0]);
    }
    // Unary shape-preserving ops on zero → zero
    switch (node.type()) {
        case FormExprType::Transpose:
        case FormExprType::SymmetricPart:
        case FormExprType::SkewPart:
        case FormExprType::Deviator:
        case FormExprType::RestrictMinus:
        case FormExprType::RestrictPlus:
        case FormExprType::TimeDerivative: {
            const auto kids = node.childrenShared();
            if (kids.size() == 1u && kids[0]) return isZeroLike(*kids[0]);
            break;
        }
        default: break;
    }
    // Contraction/scalar-producing ops on zero → zero
    switch (node.type()) {
        case FormExprType::InnerProduct:
        case FormExprType::DoubleContraction: {
            const auto kids = node.childrenShared();
            if (kids.size() == 2u && kids[0] && kids[1])
                return isZeroLike(*kids[0]) || isZeroLike(*kids[1]);
            break;
        }
        case FormExprType::Divergence:
        case FormExprType::Trace:
        case FormExprType::Norm:
        case FormExprType::Determinant: {
            const auto kids = node.childrenShared();
            if (kids.size() == 1u && kids[0]) return isZeroLike(*kids[0]);
            break;
        }
        default: break;
    }
    // Add/Subtract of zeros → zero
    if (node.type() == FormExprType::Add || node.type() == FormExprType::Subtract) {
        const auto kids = node.childrenShared();
        if (kids.size() == 2u && kids[0] && kids[1])
            return isZeroLike(*kids[0]) && isZeroLike(*kids[1]);
    }
    return false;
}

[[nodiscard]] bool isConstantScalar(const FormExprNode& node, Real& out_value)
{
    if (node.type() != FormExprType::Constant) return false;
    const auto v = node.constantValue();
    if (!v) return false;
    out_value = *v;
    return true;
}

[[nodiscard]] FormExpr differentiatePower(const FormExpr& base, const FormExpr& exponent,
                                          const FormExpr& d_base, const FormExpr& d_exponent)
{
    const auto zero = FormExpr::constant(0.0);
    Real p = 0.0;
    if (isConstantScalar(*exponent.node(), p)) {
        return p == 0.0 ? zero : FormExpr::constant(p) * pow(base, FormExpr::constant(p - 1.0)) * d_base;
    }
    const auto base_term = isZeroLike(*d_base.node()) ? zero :
        eq(exponent, zero).conditional(zero, exponent * pow(base, exponent - FormExpr::constant(1.0)) * d_base);
    if (isZeroLike(*d_exponent.node())) return base_term;

    // At zero base with positive exponent, varying only the exponent preserves zero.
    const auto fixed_zero = eq(base, zero) * gt(exponent, zero);
    const auto exponent_term = fixed_zero.conditional(zero, pow(base, exponent) * log(base) * d_exponent);
    return base_term + eq(d_exponent, zero).conditional(zero, exponent_term);
}

[[nodiscard]] bool isDefinitelyScalarNode(const FormExprNode& node)
{
    switch (node.type()) {
        case FormExprType::Constant:
        case FormExprType::ParameterSymbol:
        case FormExprType::ParameterRef:
        case FormExprType::BoundaryIntegralSymbol:
        case FormExprType::BoundaryIntegralRef:
        case FormExprType::AuxiliaryStateSymbol:
        case FormExprType::AuxiliaryStateRef:
        case FormExprType::AuxiliaryInputSymbol:
        case FormExprType::AuxiliaryInputRef:
        case FormExprType::AuxiliaryOutputSymbol:
        case FormExprType::AuxiliaryOutputRef:
        case FormExprType::MaterialStateOldRef:
        case FormExprType::MaterialStateWorkRef:
        case FormExprType::PreviousSolutionRef:
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
        case FormExprType::InnerProduct:
        case FormExprType::DoubleContraction:
        case FormExprType::Trace:
        case FormExprType::Determinant:
        case FormExprType::Norm:
        case FormExprType::Component:
        case FormExprType::SymmetricEigenvalue:
        case FormExprType::Eigenvalue:
        case FormExprType::SymmetricEigenvalueDirectionalDerivative:
        case FormExprType::SymmetricEigenvalueDirectionalDerivativeWrtA:
            return true;

        case FormExprType::TestFunction:
        case FormExprType::TrialFunction:
        case FormExprType::DiscreteField:
        case FormExprType::StateField: {
            const auto* sig = node.spaceSignature();
            return sig && sig->field_type == FieldType::Scalar;
        }

        case FormExprType::Coefficient:
            return node.scalarCoefficient() != nullptr || node.timeScalarCoefficient() != nullptr;

        case FormExprType::Negate:
        case FormExprType::Sqrt:
        case FormExprType::Exp:
        case FormExprType::Log:
        case FormExprType::AbsoluteValue:
        case FormExprType::Sign:
        case FormExprType::TimeDerivative: {
            const auto kids = node.childrenShared();
            if (kids.size() != 1u || !kids[0]) return false;
            return isDefinitelyScalarNode(*kids[0]);
        }

        case FormExprType::Minimum:
        case FormExprType::Maximum:
        case FormExprType::SmoothAbsoluteValue:
        case FormExprType::SmoothSign:
        case FormExprType::SmoothHeaviside:
        case FormExprType::Power:
        case FormExprType::Add:
        case FormExprType::Subtract:
        case FormExprType::Multiply:
        case FormExprType::Divide: {
            const auto kids = node.childrenShared();
            if (kids.size() != 2u || !kids[0] || !kids[1]) return false;
            return isDefinitelyScalarNode(*kids[0]) && isDefinitelyScalarNode(*kids[1]);
        }

        case FormExprType::Conditional: {
            const auto kids = node.childrenShared();
            if (kids.size() != 3u || !kids[1] || !kids[2]) return false;
            return isDefinitelyScalarNode(*kids[1]) && isDefinitelyScalarNode(*kids[2]);
        }

        case FormExprType::SmoothMin:
        case FormExprType::SmoothMax: {
            const auto kids = node.childrenShared();
            if (kids.size() != 3u || !kids[0] || !kids[1] || !kids[2]) return false;
            return isDefinitelyScalarNode(*kids[0]) && isDefinitelyScalarNode(*kids[1]) && isDefinitelyScalarNode(*kids[2]);
        }

        default:
            return false;
    }
}

[[nodiscard]] std::string makeDeltaName(std::string_view base)
{
    if (base.empty()) return "du";
    if (base == "u") return "du";
    if (base == "v") return "dv";
    return "d" + std::string(base);
}

struct DiffPair {
    FormExpr primal{};
    FormExpr deriv{};
};

// ---------------------------------------------------------------------------
// Complete shape derivatives (ShapeDerivative::Complete)
//
// With x = X + d and dd the mesh-displacement trial function, every quantity
// that a moving-domain kernel evaluates at a fixed reference point varies as
// follows (G = grad(dd), the physical gradient of dd):
//   fields (nodal values fixed)        0
//   x(), currentCoordinate(), d        dd
//   meshVelocity()                     meshVelocityVariation()   (a0 dd)
//   grad f                             grad(df) - grad(f) G      (vector f; G^T grad f for scalar f)
//   div f                              div(df) - tr(grad(f) G)   (vector f)
//   div T                              div(dT) - sum_j grad(T e_j) (G e_j)   (matrix T)
//   H f                                grad(d grad f) - H(f) G   (scalar f)
//   J(), currentJacobian()             G J
//   Jinv()                             -Jinv G
//   detJ(), currentJacobianDeterminant() detJ div(dd)
//   normal(), currentNormal()          -(I - n n) G^T n
//   cell measure                       div(dd)
//   facet measure, currentMeasure()    div(dd) - n.(G n)         (surface divergence)
//   h(), vol(), area()                 cellDiameterVariation(), cellVolumeVariation(), facetAreaVariation()
// ---------------------------------------------------------------------------

/// Rank of the value of @p node (0 scalar, 1 vector, 2 matrix, 3 rank-3), -1
/// when it cannot be told without evaluation.
[[nodiscard]] int valueRank(const FormExprNode& node)
{
    const auto kids = node.childrenShared();
    const auto kid = [&](std::size_t k) -> int {
        return (k < kids.size() && kids[k]) ? valueRank(*kids[k]) : -1;
    };
    if (isDefinitelyScalarNode(node)) {
        return 0;
    }
    switch (node.type()) {
        case FormExprType::CellDiameterVariation:
        case FormExprType::CellVolumeVariation:
        case FormExprType::FacetAreaVariation:
        case FormExprType::CurrentMeasureVariation:
            return 0;
        case FormExprType::TestFunction:
        case FormExprType::TrialFunction:
        case FormExprType::DiscreteField:
        case FormExprType::StateField: {
            const auto* sig = node.spaceSignature();
            if (sig == nullptr) return -1;
            switch (sig->field_type) {
                case FieldType::Scalar: return 0;
                case FieldType::Vector: return 1;
                case FieldType::Tensor: return 2;
                default: return -1;
            }
        }
        case FormExprType::Coefficient:
            if (node.vectorCoefficient() != nullptr) return 1;
            if (node.matrixCoefficient() != nullptr) return 2;
            if (node.tensor3Coefficient() != nullptr) return 3;
            return -1;
        case FormExprType::Coordinate:
        case FormExprType::ReferenceCoordinate:
        case FormExprType::MeshDisplacement:
        case FormExprType::MeshVelocity:
        case FormExprType::MeshAcceleration:
        case FormExprType::CurrentCoordinate:
        case FormExprType::PreviousCoordinate:
        case FormExprType::ReferencePhysicalCoordinate:
        case FormExprType::PreviousMeshVelocity:
        case FormExprType::PredictedMeshVelocity:
        case FormExprType::CurrentNormal:
        case FormExprType::ReferenceNormal:
        case FormExprType::Normal:
        case FormExprType::BoundaryNormal:
        case FormExprType::GeometryTrialVectorVariation:
        case FormExprType::MeshVelocityVariation:
        case FormExprType::CurrentNormalVariation:
        case FormExprType::AsVector:
        case FormExprType::CrossProduct:
        case FormExprType::SymmetricEigenvector:
            return 1;
        case FormExprType::Identity:
        case FormExprType::Jacobian:
        case FormExprType::JacobianInverse:
        case FormExprType::CurrentJacobian:
        case FormExprType::ReferenceJacobian:
        case FormExprType::SurfaceJacobian:
        case FormExprType::GeometryTrialJacobianVariation:
        case FormExprType::SurfaceJacobianVariation:
        case FormExprType::AsTensor:
        case FormExprType::Transpose:
        case FormExprType::Inverse:
        case FormExprType::Cofactor:
        case FormExprType::Deviator:
        case FormExprType::SymmetricPart:
        case FormExprType::SkewPart:
        case FormExprType::MatrixExponential:
        case FormExprType::MatrixLogarithm:
        case FormExprType::MatrixSqrt:
        case FormExprType::MatrixPower:
        case FormExprType::SpectralDecomposition:
            return 2;
        case FormExprType::Gradient: {
            const int r = kid(0);
            return r < 0 ? -1 : r + 1;
        }
        case FormExprType::Hessian: {
            const int r = kid(0);
            return r < 0 ? -1 : r + 2;
        }
        case FormExprType::Divergence: {
            const int r = kid(0);
            return r < 1 ? -1 : r - 1;
        }
        case FormExprType::Curl:
        case FormExprType::Negate:
        case FormExprType::TimeDerivative:
        case FormExprType::RestrictMinus:
        case FormExprType::RestrictPlus:
        case FormExprType::Jump:
        case FormExprType::Average:
        case FormExprType::Pullback:
        case FormExprType::Pushforward:
        case FormExprType::Normalize:
            return kid(0);
        case FormExprType::Add:
        case FormExprType::Subtract: {
            const int a = kid(0), b = kid(1);
            return (a >= 0 && b >= 0 && a != b) ? -1 : std::max(a, b);
        }
        case FormExprType::Divide:
            return kid(1) == 0 ? kid(0) : -1;
        case FormExprType::Multiply: {
            const int a = kid(0), b = kid(1);
            if (a < 0 || b < 0) return -1;
            if (a == 0) return b;
            if (b == 0) return a;
            if (a == 2 && b == 1) return 1;
            if (a == 2 && b == 2) return 2;
            return -1;
        }
        case FormExprType::OuterProduct: {
            const int a = kid(0), b = kid(1);
            return (a < 0 || b < 0) ? -1 : a + b;
        }
        case FormExprType::Conditional:
            return kid(1);
        case FormExprType::Component: {
            const int r = kid(0);
            const int j = node.componentIndex1().value_or(-1);
            if (r == 1 && j < 0) return 0;
            if (r == 2 && j >= 0) return 0;
            return -1;
        }
        default:
            return -1;
    }
}

/// A geometric terminal that kernels evaluate as constant in space within a
/// cell (its spatial derivatives are zero), so a spatial derivative of an
/// expression that contains it has no consistent shape derivative.
[[nodiscard]] bool containsCellwiseConstantGeometry(const FormExprNode& node)
{
    switch (node.type()) {
        case FormExprType::Normal:
        case FormExprType::CurrentNormal:
        case FormExprType::ReferenceNormal:
        case FormExprType::BoundaryNormal:
        case FormExprType::Jacobian:
        case FormExprType::JacobianInverse:
        case FormExprType::JacobianDeterminant:
        case FormExprType::CurrentJacobian:
        case FormExprType::CurrentJacobianDeterminant:
        case FormExprType::CurrentMeasure:
        case FormExprType::SurfaceJacobian:
        case FormExprType::CurrentMeanCurvature:
        case FormExprType::CellDiameter:
        case FormExprType::CellVolume:
        case FormExprType::FacetArea:
            return true;
        default:
            break;
    }
    for (const auto& child : node.childrenShared()) {
        if (child && containsCellwiseConstantGeometry(*child)) {
            return true;
        }
    }
    return false;
}

/// How kernels evaluate the spatial second derivatives of the fields in a
/// subtree: Exact with the mapping curvature (basis functions, and the row's
/// active field, whose Hessians the residual reads from the trial basis),
/// Approximate without it (every other field: J^-T (reference Hessian) J^-1).
[[noreturn]] void refuseCompleteShapeDerivative(const FormExprNode& node, const std::string& why)
{
    throw std::invalid_argument(
        "differentiateResidual: the complete shape derivative (ShapeDerivative::Complete) " + why +
        "; offending subexpression: " + node.toString());
}

enum class DiffWrtKind : std::uint8_t {
    ActiveTrialFunction,
    SpecificTrialFunction,
    FieldId,
    AuxiliaryOutputSlot,     ///< Differentiate w.r.t. AuxiliaryOutputRef(slot)
    NormalTerminal,          ///< Differentiate w.r.t. the geometric normal() terminal
    SpatialCoordinate        ///< Differentiate w.r.t. physical quadrature point motion
};

struct DiffTargetConfig {
    DiffWrtKind kind{DiffWrtKind::ActiveTrialFunction};

    // How to rewrite residual TrialFunction occurrences in the primal expression.
    FieldId trial_state_field{CURRENT_SOLUTION_FIELD_ID};
    // FieldId whose residual TrialFunction occurrences represent the active
    // trial unknown even when their primal is rewritten to CURRENT_SOLUTION.
    std::optional<FieldId> active_trial_field{};

    // SpecificTrialFunction target.
    std::optional<FormExprNode::SpaceSignature> trial_sig{};
    std::optional<std::string> trial_name{};

    // FieldId target.
    std::optional<FieldId> field_id{};
    std::optional<FormExprNode::SpaceSignature> field_sig{};
    std::optional<std::string> field_name{};

    // AuxiliaryOutputSlot target.
    std::optional<std::uint32_t> aux_output_slot{};

    // NormalTerminal target.
    FormExpr normal_direction{};

    // SpatialCoordinate target.
    FormExpr spatial_direction{};
    CutSurfacePointDirection cut_point{};

    GeometrySensitivityOptions geometry_sensitivity{};

    // Complete shape derivative of a bare integrand: the integral domain whose
    // measure variation is added to the derivative.
    std::optional<IntegralDomain> integrand_domain{};
};

[[nodiscard]] bool differentiatesMeshGeometry(const DiffTargetConfig& cfg) noexcept
{
    return cfg.kind == DiffWrtKind::FieldId &&
           cfg.field_id.has_value() &&
           cfg.geometry_sensitivity.mode == GeometrySensitivityMode::MeshMotionUnknowns &&
           cfg.geometry_sensitivity.mesh_motion_field != INVALID_FIELD_ID &&
           *cfg.field_id == cfg.geometry_sensitivity.mesh_motion_field;
}

[[nodiscard]] bool differentiatesCompleteShape(const DiffTargetConfig& cfg) noexcept
{
    return differentiatesMeshGeometry(cfg) &&
           cfg.geometry_sensitivity.shape_derivative == ShapeDerivative::Complete;
}

/// The mesh-displacement trial function dd of a complete shape derivative:
/// the options' space and symbol, else the first trial function (or mesh
/// field) of @p residual.
[[nodiscard]] FormExpr completeShapeTrialFunction(const FormExpr& residual, const DiffTargetConfig& cfg)
{
    const auto& geometry = cfg.geometry_sensitivity;
    if (geometry.mesh_motion_space.has_value() && geometry.mesh_motion_symbol.has_value()) {
        return FormExpr::trialFunction(*geometry.mesh_motion_space, makeDeltaName(*geometry.mesh_motion_symbol));
    }
    const FormExprNode* found = nullptr;
    const auto visit = [&](const auto& self, const FormExprNode& n) -> void {
        if (found != nullptr) return;
        const bool trial = n.type() == FormExprType::TrialFunction;
        const bool mesh_field = (n.type() == FormExprType::StateField || n.type() == FormExprType::DiscreteField) &&
                                n.fieldId().has_value() && *n.fieldId() == geometry.mesh_motion_field;
        if ((trial || mesh_field) && n.spaceSignature() != nullptr) {
            found = &n;
            return;
        }
        for (const auto& child : n.childrenShared()) {
            if (child) self(self, *child);
        }
    };
    if (residual.node() != nullptr) {
        visit(visit, *residual.node());
    }
    if (found != nullptr) {
        return FormExpr::trialFunction(*found->spaceSignature(), makeDeltaName(found->toString()));
    }
    if (geometry.mesh_motion_space.has_value()) {
        return FormExpr::trialFunction(*geometry.mesh_motion_space,
                                       makeDeltaName(geometry.mesh_motion_symbol.value_or("mesh_displacement")));
    }
    throw std::invalid_argument(
        "differentiateResidual: the complete shape derivative needs the mesh-displacement space "
        "(GeometrySensitivityOptions::mesh_motion_space) for a residual without a mesh-displacement trial function");
}

[[nodiscard]] bool spaceSignatureEqual(const FormExprNode::SpaceSignature& a,
                                       const FormExprNode::SpaceSignature& b) noexcept
{
    return a.space_type == b.space_type &&
           a.field_type == b.field_type &&
           a.continuity == b.continuity &&
           a.value_dimension == b.value_dimension &&
           a.topological_dimension == b.topological_dimension &&
           a.polynomial_order == b.polynomial_order &&
           a.element_type == b.element_type;
}

[[nodiscard]] FormExpr spatialDirectionalDerivativeOfValue(
    const FormExpr& value,
    const FormExpr& direction,
    std::optional<FormExprNode::SpaceSignature> signature = std::nullopt)
{
    if (!value.isValid() || value.node() == nullptr ||
        !direction.isValid() || direction.node() == nullptr) {
        return FormExpr{};
    }
    if (signature.has_value()) {
        if (signature->field_type == FieldType::Scalar ||
            signature->value_dimension == 1) {
            return inner(grad(value), direction);
        }
        return grad(value) * direction;
    }
    if (isDefinitelyScalarNode(*value.node())) {
        return inner(grad(value), direction);
    }
    return grad(value) * direction;
}

} // namespace

SymbolicDiffResult checkSymbolicDifferentiability(const FormExpr& expr)
{
    SymbolicDiffResult out;
    out.ok = true;

    if (!expr.isValid() || expr.node() == nullptr) {
        out.ok = false;
        out.first_issue = SymbolicDiffIssue{
            .type = FormExprType::Constant,
            .message = "SymbolicDiff: invalid expression",
            .subexpr = {},
        };
        return out;
    }

    const auto visit = [&](const auto& self, const FormExprNode& n) -> void {
        if (!out.ok) return;

        switch (n.type()) {
            case FormExprType::Constitutive:
            case FormExprType::ConstitutiveOutput:
                out.ok = false;
                out.first_issue = issue(n, "SymbolicDiff: Constitutive nodes must be inlined before differentiation");
                return;
            default:
                break;
        }

        for (const auto& child : n.childrenShared()) {
            if (child) self(self, *child);
        }
    };

    visit(visit, *expr.node());
    return out;
}

// ---------------------------------------------------------------------------
//  extractTermsReferencing — additive sub-term extraction
// ---------------------------------------------------------------------------

namespace {
bool subtreeContains(const FormExprNode& node, FormExprType target, std::uint32_t slot,
                     std::unordered_map<const FormExprNode*, bool>& memo)
{
    if (auto it = memo.find(&node); it != memo.end()) return it->second;
    bool found = false;
    if (node.type() == target) {
        const auto s = node.slotIndex();
        if (s && *s == slot) found = true;
    }
    if (!found) {
        for (const auto* c : node.children()) {
            if (c && subtreeContains(*c, target, slot, memo)) {
                found = true;
                break;
            }
        }
    }
    memo[&node] = found;
    return found;
}

// Recursively split on Add/Subtract and keep only terms matching the target.
FormExpr extractImpl(const FormExpr& form, FormExprType target, std::uint32_t slot,
                     std::unordered_map<const FormExprNode*, bool>& memo)
{
    if (!form.isValid() || !form.node()) return {};
    const auto& n = *form.node();
    const auto ty = n.type();

    if (ty == FormExprType::Add) {
        const auto kids = n.childrenShared();
        if (kids.size() == 2u && kids[0] && kids[1]) {
            auto left  = extractImpl(FormExpr(kids[0]), target, slot, memo);
            auto right = extractImpl(FormExpr(kids[1]), target, slot, memo);
            if (left.isValid() && right.isValid()) return left + right;
            if (left.isValid()) return left;
            if (right.isValid()) return right;
            return {};
        }
    }
    if (ty == FormExprType::Subtract) {
        const auto kids = n.childrenShared();
        if (kids.size() == 2u && kids[0] && kids[1]) {
            auto left  = extractImpl(FormExpr(kids[0]), target, slot, memo);
            auto right = extractImpl(FormExpr(kids[1]), target, slot, memo);
            if (left.isValid() && right.isValid()) return left - right;
            if (left.isValid()) return left;
            if (right.isValid()) return -right;
            return {};
        }
    }

    // Leaf/non-additive node: keep if subtree contains the target.
    if (subtreeContains(n, target, slot, memo)) {
        return form;
    }
    return {};
}
} // anonymous namespace

FormExpr extractTermsReferencing(
    const FormExpr& form, FormExprType target_type, std::uint32_t target_slot)
{
    if (!form.isValid() || !form.node()) return {};
    std::unordered_map<const FormExprNode*, bool> memo;
    return extractImpl(form, target_type, target_slot, memo);
}

// Bottom-up zero-collapse: mark all zero-like subtrees in a single pass,
// then use transformNodes to replace them with constant(0.0).
namespace {
void markZeros(const FormExprNode& node,
               std::unordered_map<const FormExprNode*, bool>& memo)
{
    if (memo.count(&node)) return;

    const auto kids = node.childrenShared();

    // Recurse children first (post-order).
    for (const auto& k : kids) {
        if (k) markZeros(*k, memo);
    }

    auto childZ = [&](std::size_t i) -> bool {
        if (i >= kids.size() || !kids[i]) return false;
        auto it = memo.find(kids[i].get());
        return it != memo.end() && it->second;
    };

    bool z = false;
    const auto ty = node.type();

    // Leaf zeros.
    if (kids.empty()) {
        z = isScalarZero(node) || ty == FormExprType::TypedZero;
    }
    // Binary absorbing.
    else if (kids.size() == 2u) {
        const bool z0 = childZ(0), z1 = childZ(1);
        switch (ty) {
            case FormExprType::Multiply:
            case FormExprType::InnerProduct:
            case FormExprType::DoubleContraction:
            case FormExprType::OuterProduct:
            case FormExprType::CrossProduct:
                z = z0 || z1;
                break;
            case FormExprType::Add:
            case FormExprType::Subtract:
                z = z0 && z1;
                break;
            default: break;
        }
    }
    // Unary propagation.
    if (!z && kids.size() == 1u && childZ(0)) {
        switch (ty) {
            case FormExprType::Negate:
            case FormExprType::Transpose:
            case FormExprType::SymmetricPart:
            case FormExprType::SkewPart:
            case FormExprType::Deviator:
            case FormExprType::Normalize:
            case FormExprType::TimeDerivative:
            case FormExprType::Divergence:
            case FormExprType::Trace:
            case FormExprType::Norm:
            case FormExprType::Determinant:
            case FormExprType::AbsoluteValue:
            case FormExprType::Sqrt:
            case FormExprType::Gradient:
            case FormExprType::Curl:
            case FormExprType::Hessian:
                z = true;
                break;
            default: break;
        }
    }
    // Integrals of zero.
    if (!z && kids.size() >= 1u && childZ(0)) {
        switch (ty) {
            case FormExprType::CellIntegral:
            case FormExprType::BoundaryIntegral:
            case FormExprType::InteriorFaceIntegral:
            case FormExprType::InterfaceIntegral:
            case FormExprType::CutVolumeIntegral:
            case FormExprType::GlobalIntegral:
                z = true;
                break;
            default: break;
        }
    }

    memo[&node] = z;
}
} // anonymous namespace

static FormExpr collapseZeros(const FormExpr& expr)
{
    if (!expr.isValid() || !expr.node()) return expr;

    // Phase 1: mark all zero-like subtrees.
    std::unordered_map<const FormExprNode*, bool> memo;
    markZeros(*expr.node(), memo);

    // Phase 2: replace zero-marked subtrees with constant(0.0).
    // transformNodes visits pre-order: if we replace a zero subtree at
    // the top, its children are never visited (pruned).
    return expr.transformNodes([&](const FormExprNode& n) -> std::optional<FormExpr> {
        auto it = memo.find(&n);
        if (it != memo.end() && it->second) {
            return FormExpr::constant(0.0);
        }
        return std::nullopt;
    });
}

FormExpr simplify(const FormExpr& expr)
{
    if (!expr.isValid() || expr.node() == nullptr) {
        return {};
    }

    // Phase 1: bottom-up zero collapse.  Detects zero-like subtrees and
    // replaces them with constant(0.0) in a single post-order walk.
    // This handles the common case where differentiating a large form
    // w.r.t. a parameter produces 0*(complex_expression) at many branches.
    FormExpr current = collapseZeros(expr);

    // Phase 2: fixed-point algebraic simplification.
    constexpr int kMaxPasses = 8;
    for (int pass = 0; pass < kMaxPasses; ++pass) {
        bool changed = false;

        current = current.transformNodes([&](const FormExprNode& n) -> std::optional<FormExpr> {
            const auto kids = n.childrenShared();

            auto asExpr = [&](const std::shared_ptr<FormExprNode>& p) -> FormExpr {
                return p ? FormExpr(p) : FormExpr{};
            };

            // ---- Unary ----
            if (n.type() == FormExprType::Negate) {
                if (kids.size() != 1u || !kids[0]) return std::nullopt;
                if (kids[0]->type() == FormExprType::Negate) {
                    const auto gkids = kids[0]->childrenShared();
                    if (gkids.size() == 1u && gkids[0]) {
                        changed = true;
                        return FormExpr(gkids[0]);
                    }
                }

                Real v = 0.0;
                if (isConstantScalar(*kids[0], v)) {
                    changed = true;
                    return FormExpr::constant(-v);
                }
                return std::nullopt;
            }

            if (n.type() == FormExprType::Sqrt || n.type() == FormExprType::Exp || n.type() == FormExprType::Log) {
                if (kids.size() != 1u || !kids[0]) return std::nullopt;
                Real v = 0.0;
                if (isConstantScalar(*kids[0], v)) {
                    const Real out = (n.type() == FormExprType::Sqrt) ? std::sqrt(v)
                                   : (n.type() == FormExprType::Exp)  ? std::exp(v)
                                                                     : std::log(v);
                    changed = true;
                    return FormExpr::constant(out);
                }
                return std::nullopt;
            }

            if (n.type() == FormExprType::AbsoluteValue || n.type() == FormExprType::Sign) {
                if (kids.size() != 1u || !kids[0]) return std::nullopt;
                Real v = 0.0;
                if (isConstantScalar(*kids[0], v)) {
                    const Real out = (n.type() == FormExprType::AbsoluteValue)
                        ? std::abs(v)
                        : (v > 0.0) ? 1.0 : ((v < 0.0) ? -1.0 : 0.0);
                    changed = true;
                    return FormExpr::constant(out);
                }
                return std::nullopt;
            }

            // ---- Binary ----
            if (n.type() == FormExprType::Add) {
                if (kids.size() != 2u || !kids[0] || !kids[1]) return std::nullopt;

                if (isZeroLike(*kids[0])) {
                    changed = true;
                    return asExpr(kids[1]);
                }
                if (isZeroLike(*kids[1])) {
                    changed = true;
                    return asExpr(kids[0]);
                }

                if (kids[1]->type() == FormExprType::Negate) {
                    const auto nkids = kids[1]->childrenShared();
                    if (nkids.size() == 1u && nkids[0] && nkids[0].get() == kids[0].get()) {
                        changed = true;
                        return isDefinitelyScalarNode(*kids[0]) ? FormExpr::constant(0.0)
                                                                : (FormExpr::constant(0.0) * asExpr(kids[0]));
                    }
                }
                if (kids[0]->type() == FormExprType::Negate) {
                    const auto nkids = kids[0]->childrenShared();
                    if (nkids.size() == 1u && nkids[0] && nkids[0].get() == kids[1].get()) {
                        changed = true;
                        return isDefinitelyScalarNode(*kids[1]) ? FormExpr::constant(0.0)
                                                                : (FormExpr::constant(0.0) * asExpr(kids[1]));
                    }
                }

                Real a = 0.0;
                Real b = 0.0;
                if (isConstantScalar(*kids[0], a) && isConstantScalar(*kids[1], b)) {
                    changed = true;
                    return FormExpr::constant(a + b);
                }
                return std::nullopt;
            }

            if (n.type() == FormExprType::Subtract) {
                if (kids.size() != 2u || !kids[0] || !kids[1]) return std::nullopt;

                if (isZeroLike(*kids[1])) {
                    changed = true;
                    return asExpr(kids[0]);
                }
                if (isZeroLike(*kids[0])) {
                    changed = true;
                    return -asExpr(kids[1]);
                }

                if (kids[0].get() == kids[1].get()) {
                    changed = true;
                    return isDefinitelyScalarNode(*kids[0]) ? FormExpr::constant(0.0)
                                                            : (FormExpr::constant(0.0) * asExpr(kids[0]));
                }

                Real a = 0.0;
                Real b = 0.0;
                if (isConstantScalar(*kids[0], a) && isConstantScalar(*kids[1], b)) {
                    changed = true;
                    return FormExpr::constant(a - b);
                }
                return std::nullopt;
            }

            if (n.type() == FormExprType::Multiply) {
                if (kids.size() != 2u || !kids[0] || !kids[1]) return std::nullopt;

                if (isZeroLike(*kids[0])) {
                    changed = true;
                    return isDefinitelyScalarNode(*kids[1]) ? FormExpr::constant(0.0)
                                                            : (FormExpr::constant(0.0) * asExpr(kids[1]));
                }
                if (isZeroLike(*kids[1])) {
                    changed = true;
                    return isDefinitelyScalarNode(*kids[0]) ? FormExpr::constant(0.0)
                                                            : (FormExpr::constant(0.0) * asExpr(kids[0]));
                }

                if (isScalarOne(*kids[0])) {
                    changed = true;
                    return asExpr(kids[1]);
                }
                if (isScalarOne(*kids[1])) {
                    changed = true;
                    return asExpr(kids[0]);
                }

                Real a = 0.0;
                Real b = 0.0;
                if (isConstantScalar(*kids[0], a) && isConstantScalar(*kids[1], b)) {
                    changed = true;
                    return FormExpr::constant(a * b);
                }
                return std::nullopt;
            }

            if (n.type() == FormExprType::Divide) {
                if (kids.size() != 2u || !kids[0] || !kids[1]) return std::nullopt;

                if (isZeroLike(*kids[0])) {
                    changed = true;
                    return asExpr(kids[0]);
                }
                if (isScalarOne(*kids[1])) {
                    changed = true;
                    return asExpr(kids[0]);
                }

                Real a = 0.0;
                Real b = 0.0;
                if (isConstantScalar(*kids[0], a) && isConstantScalar(*kids[1], b)) {
                    changed = true;
                    return FormExpr::constant(a / b);
                }
                return std::nullopt;
            }

            if (n.type() == FormExprType::Power) {
                if (kids.size() != 2u || !kids[0] || !kids[1]) return std::nullopt;

                Real exp = 0.0;
                if (isConstantScalar(*kids[1], exp)) {
                    if (exp == 0.0) {
                        changed = true;
                        return FormExpr::constant(1.0);
                    }
                    if (exp == 1.0) {
                        changed = true;
                        return asExpr(kids[0]);
                    }
                    if (exp == 2.0) {
                        const auto base = asExpr(kids[0]);
                        changed = true;
                        return base * base;
                    }
                }

                Real a = 0.0;
                Real b = 0.0;
                if (isConstantScalar(*kids[0], a) && isConstantScalar(*kids[1], b)) {
                    changed = true;
                    return FormExpr::constant(std::pow(a, b));
                }
                return std::nullopt;
            }

            // ---- Contraction/scalar-producing ops: zero in → zero out ----
            if (n.type() == FormExprType::InnerProduct ||
                n.type() == FormExprType::DoubleContraction) {
                if (kids.size() == 2u && kids[0] && kids[1]) {
                    if (isZeroLike(*kids[0]) || isZeroLike(*kids[1])) {
                        changed = true;
                        return FormExpr::constant(0.0);
                    }
                }
            }
            if (n.type() == FormExprType::Divergence ||
                n.type() == FormExprType::Trace ||
                n.type() == FormExprType::Norm) {
                if (kids.size() == 1u && kids[0] && isZeroLike(*kids[0])) {
                    changed = true;
                    return FormExpr::constant(0.0);
                }
            }

            // ---- Integral of zero → zero ----
            if (n.type() == FormExprType::CellIntegral ||
                n.type() == FormExprType::BoundaryIntegral ||
                n.type() == FormExprType::InteriorFaceIntegral ||
                n.type() == FormExprType::InterfaceIntegral ||
                n.type() == FormExprType::CutVolumeIntegral ||
                n.type() == FormExprType::GlobalIntegral) {
                if (kids.size() >= 1u && kids[0] && isZeroLike(*kids[0])) {
                    changed = true;
                    return FormExpr::constant(0.0);
                }
            }

            // ---- Unary shape-preserving on zero → zero ----
            if (n.type() == FormExprType::Transpose ||
                n.type() == FormExprType::SymmetricPart ||
                n.type() == FormExprType::SkewPart ||
                n.type() == FormExprType::Deviator) {
                if (kids.size() == 1u && kids[0] && isZeroLike(*kids[0])) {
                    changed = true;
                    return FormExpr::constant(0.0) * asExpr(kids[0]);
                }
            }

            if (n.type() == FormExprType::Conditional) {
                if (kids.size() != 3u || !kids[0] || !kids[1] || !kids[2]) return std::nullopt;
                Real c = 0.0;
                if (isConstantScalar(*kids[0], c)) {
                    changed = true;
                    return (c != 0.0) ? asExpr(kids[1]) : asExpr(kids[2]);
                }
                return std::nullopt;
            }

            if (n.type() == FormExprType::Less || n.type() == FormExprType::LessEqual ||
                n.type() == FormExprType::Greater || n.type() == FormExprType::GreaterEqual ||
                n.type() == FormExprType::Equal || n.type() == FormExprType::NotEqual ||
                n.type() == FormExprType::Minimum || n.type() == FormExprType::Maximum) {
                if (kids.size() != 2u || !kids[0] || !kids[1]) return std::nullopt;
                Real a = 0.0;
                Real b = 0.0;
                if (!isConstantScalar(*kids[0], a) || !isConstantScalar(*kids[1], b)) {
                    return std::nullopt;
                }

                if (n.type() == FormExprType::Minimum) {
                    changed = true;
                    return FormExpr::constant(std::min(a, b));
                }
                if (n.type() == FormExprType::Maximum) {
                    changed = true;
                    return FormExpr::constant(std::max(a, b));
                }

                bool truth = false;
                switch (n.type()) {
                    case FormExprType::Less: truth = (a < b); break;
                    case FormExprType::LessEqual: truth = (a <= b); break;
                    case FormExprType::Greater: truth = (a > b); break;
                    case FormExprType::GreaterEqual: truth = (a >= b); break;
                    case FormExprType::Equal: truth = (a == b); break;
                    case FormExprType::NotEqual: truth = (a != b); break;
                    default: break;
                }
                changed = true;
                return FormExpr::constant(truth ? 1.0 : 0.0);
            }

            return std::nullopt;
        });

        if (!changed) {
            break;
        }
    }

    return current;
}

namespace {

FormExpr differentiateResidualImpl(const FormExpr& residual_form,
                                  const DiffTargetConfig& cfg)
{
    if (!residual_form.isValid() || residual_form.node() == nullptr) {
        throw std::invalid_argument("differentiateResidual: invalid residual expression");
    }

    const auto check = checkSymbolicDifferentiability(residual_form);
    if (!check.ok) {
        const auto msg = check.first_issue ? check.first_issue->message : std::string("SymbolicDiff: unsupported expression");
        throw std::invalid_argument("differentiateResidual: " + msg);
    }

    std::unordered_map<const FormExprNode*, DiffPair> memo;

    std::optional<FormExprNode::SpaceSignature> wrt_field_sig = cfg.field_sig;
    std::optional<std::string> wrt_field_name = cfg.field_name;
    std::optional<std::string> wrt_field_delta_name =
        wrt_field_name ? std::optional<std::string>(makeDeltaName(*wrt_field_name)) : std::nullopt;

    const int auto_extent = inferAutoIndexExtent(residual_form);

    // Complete shape derivative: the mesh-displacement trial function dd, the
    // integral domain being differentiated (it decides the measure and
    // currentMeasure() rules) and one memo per domain.
    const bool complete_shape = differentiatesCompleteShape(cfg);
    const FormExpr shape_dd = complete_shape ? completeShapeTrialFunction(residual_form, cfg) : FormExpr{};
    IntegralDomain shape_domain = cfg.integrand_domain.value_or(IntegralDomain::Cell);
    // The chain factor is G = grad(dd) at every depth: kernels evaluate the
    // second derivatives of every field with the mapping curvature, so a
    // spatial derivative of G carries the second derivatives of dd.
    std::array<std::unordered_map<const FormExprNode*, DiffPair>, 5> shape_memo;
    const auto shapeMeasureVariation = [&](IntegralDomain domain, const FormExprNode& at) -> FormExpr {
        switch (domain) {
            case IntegralDomain::Cell:
                return shape_dd.div();
            case IntegralDomain::Boundary: {
                const auto n = FormExpr::normal();
                return shape_dd.div() - inner(shape_dd.grad() * n, n);
            }
            case IntegralDomain::InteriorFace:
            case IntegralDomain::InterfaceFace: {
                // The facet moves with the trial function of either side; the
                // minus side's restricted to the facet is the shared one.
                const auto n = FormExpr::normal();
                return (shape_dd.div() - inner(shape_dd.grad() * n, n)).minus();
            }
            case IntegralDomain::CutVolume:
                break;
        }
        refuseCompleteShapeDerivative(at, "does not differentiate across a moving cut topology (a cut-volume "
                                          "or generated-interface integral is a separate capability)");
    };

    const auto diff = [&](const auto& self, const std::shared_ptr<FormExprNode>& node) -> DiffPair {
        if (!node) {
            throw std::invalid_argument("differentiateResidual: encountered null node");
        }
        auto& active_memo = complete_shape ? shape_memo[static_cast<std::size_t>(shape_domain)] : memo;
        if (auto it = active_memo.find(node.get()); it != active_memo.end()) {
            return it->second;
        }

        DiffPair out;

        const auto kids = node->childrenShared();

        auto diff1 = [&](std::size_t k) -> DiffPair {
            if (kids.size() <= k || !kids[k]) {
                throw std::invalid_argument("differentiateResidual: missing child");
            }
            return self(self, kids[k]);
        };

        auto zeroOf = [&](const FormExpr& ref) -> FormExpr {
            return FormExpr::constant(0.0) * ref;
        };

        if (complete_shape) {
            const auto G = shape_dd.grad();
            const int dim = shape_dd.node()->spaceSignature()->value_dimension;
            const auto unit = [dim](int j) {
                std::vector<FormExpr> e;
                for (int k = 0; k < dim; ++k) e.push_back(FormExpr::constant(k == j ? 1.0 : 0.0));
                return FormExpr::asVector(std::move(e));
            };
            const auto inDomain = [&](IntegralDomain domain, std::size_t k) {
                const auto saved = shape_domain;
                shape_domain = domain;
                const auto r = diff1(k);
                shape_domain = saved;
                return r;
            };
            const auto spatialChild = [&](const char* op) -> DiffPair {
                if (kids.empty() || !kids[0]) {
                    throw std::invalid_argument("differentiateResidual: missing child");
                }
                if (containsCellwiseConstantGeometry(*kids[0])) {
                    refuseCompleteShapeDerivative(
                        *node, std::string("cannot differentiate ") + op +
                                   " of a geometric quantity that kernels treat as constant in a cell");
                }
                return diff1(0);
            };
            bool handled = true;
            switch (node->type()) {
                case FormExprType::Coordinate:
                case FormExprType::CurrentCoordinate:
                case FormExprType::MeshDisplacement:
                    out.primal = FormExpr(node);
                    out.deriv = shape_dd;
                    break;
                case FormExprType::MeshVelocity:
                    out.primal = FormExpr(node);
                    out.deriv = FormExpr::meshVelocityVariation();
                    break;
                case FormExprType::Jacobian:
                case FormExprType::CurrentJacobian:
                    out.primal = FormExpr(node);
                    out.deriv = G * out.primal;
                    break;
                case FormExprType::JacobianInverse:
                    out.primal = FormExpr(node);
                    out.deriv = -(out.primal * G);
                    break;
                case FormExprType::JacobianDeterminant:
                case FormExprType::CurrentJacobianDeterminant:
                    out.primal = FormExpr(node);
                    out.deriv = out.primal * shape_dd.div();
                    break;
                case FormExprType::CurrentMeasure:
                    out.primal = FormExpr(node);
                    out.deriv = out.primal * shapeMeasureVariation(shape_domain, *node);
                    break;
                case FormExprType::Normal:
                case FormExprType::CurrentNormal: {
                    out.primal = FormExpr(node);
                    const auto Gtn = G.transpose() * out.primal;
                    out.deriv = inner(Gtn, out.primal) * out.primal - Gtn;
                    break;
                }
                case FormExprType::SurfaceJacobian:
                    out.primal = FormExpr(node);
                    out.deriv = FormExpr::surfaceJacobianVariation();
                    break;
                case FormExprType::CellDiameter:
                    out.primal = FormExpr(node);
                    out.deriv = FormExpr::cellDiameterVariation();
                    break;
                case FormExprType::CellVolume:
                    out.primal = FormExpr(node);
                    out.deriv = FormExpr::cellVolumeVariation();
                    break;
                case FormExprType::FacetArea:
                    out.primal = FormExpr(node);
                    out.deriv = FormExpr::facetAreaVariation();
                    break;
                case FormExprType::Coefficient:
                    // f(x) evaluated at the moving point: grad(f) . dd (the
                    // kernels' gradient of a coefficient callback).
                    out.primal = FormExpr(node);
                    if (node->scalarCoefficient() != nullptr || node->timeScalarCoefficient() != nullptr) {
                        out.deriv = inner(out.primal.grad(), shape_dd);
                    } else if (node->vectorCoefficient() != nullptr) {
                        out.deriv = out.primal.grad() * shape_dd;
                    } else {
                        refuseCompleteShapeDerivative(*node, "supports scalar and vector coefficient callbacks only");
                    }
                    break;
                case FormExprType::MeshAcceleration:
                case FormExprType::PredictedMeshVelocity:
                    refuseCompleteShapeDerivative(*node, "relates only the mesh displacement and its first time "
                                                         "derivative to the moving domain");
                case FormExprType::CurrentMeanCurvature:
                    refuseCompleteShapeDerivative(*node, "does not differentiate the curvature of a moving boundary");
                case FormExprType::BoundaryNormal:
                case FormExprType::CutVolumeIntegral:
                    refuseCompleteShapeDerivative(*node, "does not differentiate across a moving cut topology "
                                                         "(generated rules are a separate capability)");
                case FormExprType::BoundaryFunctionalSymbol:
                case FormExprType::BoundaryIntegralSymbol:
                case FormExprType::BoundaryIntegralRef:
                case FormExprType::AuxiliaryInputSymbol:
                case FormExprType::AuxiliaryInputRef:
                case FormExprType::AuxiliaryOutputSymbol:
                case FormExprType::AuxiliaryOutputRef:
                    refuseCompleteShapeDerivative(*node, "does not differentiate the geometry inside a functional "
                                                         "evaluated on another integral (a nonlocal dependence)");
                case FormExprType::Gradient: {
                    const auto a = spatialChild("grad");
                    const int rank = valueRank(*a.primal.node());
                    const auto& Gc = G;
                    out.primal = a.primal.grad();
                    if (rank == 0) {
                        out.deriv = a.deriv.grad() - Gc.transpose() * out.primal;
                    } else if (rank == 1) {
                        out.deriv = a.deriv.grad() - out.primal * Gc;
                    } else {
                        refuseCompleteShapeDerivative(*node, "differentiates grad of scalars and vectors only");
                    }
                    break;
                }
                case FormExprType::Divergence: {
                    const auto a = spatialChild("div");
                    const int rank = valueRank(*a.primal.node());
                    const auto& Gc = G;
                    out.primal = a.primal.div();
                    if (rank == 1) {
                        out.deriv = a.deriv.div() - (a.primal.grad() * Gc).trace();
                    } else if (rank == 2) {
                        FormExpr chain{};
                        for (int j = 0; j < dim; ++j) {
                            const auto term = (a.primal * unit(j)).grad() * (Gc * unit(j));
                            chain = chain.isValid() ? chain + term : term;
                        }
                        out.deriv = a.deriv.div() - chain;
                    } else {
                        refuseCompleteShapeDerivative(*node, "differentiates div of vectors and matrices only");
                    }
                    break;
                }
                case FormExprType::Hessian: {
                    const auto a = spatialChild("hessian");
                    if (valueRank(*a.primal.node()) != 0) {
                        refuseCompleteShapeDerivative(*node, "differentiates the Hessian of scalars only");
                    }
                    out.primal = a.primal.hessian();
                    // Hessians carry the mapping curvature: dH = grad(d grad f) - H G,
                    // which also carries the second derivatives of dd.
                    out.deriv = (a.deriv.grad() - G.transpose() * a.primal.grad()).grad() - out.primal * G;
                    break;
                }
                case FormExprType::Curl: {
                    const auto a = spatialChild("curl");
                    if (valueRank(*a.primal.node()) != 1) {
                        refuseCompleteShapeDerivative(*node, "differentiates the curl of vectors only");
                    }
                    out.primal = a.primal.curl();
                    // (curl f)_i = e_ijk d_j f_k; the chain part of d_j f_k is (grad(f) G)_kj.
                    const auto A = a.primal.grad() * G;
                    const auto zero = FormExpr::constant(0.0);
                    const auto chain = dim == 3
                        ? FormExpr::asVector({A.component(2, 1) - A.component(1, 2),
                                              A.component(0, 2) - A.component(2, 0),
                                              A.component(1, 0) - A.component(0, 1)})
                        : FormExpr::asVector({zero, zero, A.component(1, 0) - A.component(0, 1)});
                    out.deriv = a.deriv.curl() - chain;
                    break;
                }
                case FormExprType::CellIntegral: {
                    const auto a = inDomain(IntegralDomain::Cell, 0);
                    out.primal = a.primal.dx();
                    out.deriv = (a.deriv + a.primal * shapeMeasureVariation(IntegralDomain::Cell, *node)).dx();
                    break;
                }
                case FormExprType::BoundaryIntegral:
                case FormExprType::InterfaceIntegral: {
                    const auto* exterior = node->exteriorBoundaryMeasure();
                    if (exterior != nullptr && exterior->isGeneratedActiveSubset()) {
                        refuseCompleteShapeDerivative(*node, "does not differentiate across a moving cut topology "
                                                             "(a generated boundary rule is a separate capability)");
                    }
                    const auto domain = node->type() == FormExprType::BoundaryIntegral ? IntegralDomain::Boundary
                                                                                       : IntegralDomain::InterfaceFace;
                    const auto a = inDomain(domain, 0);
                    out.primal = wrapExteriorIntegral(a.primal, *node);
                    const auto whole = a.deriv + a.primal * shapeMeasureVariation(domain, *node);
                    if (node->type() != FormExprType::InterfaceIntegral) {
                        out.deriv = wrapExteriorIntegral(whole, *node);
                        break;
                    }
                    out.deriv = interfaceIntegralDerivative(
                        kids[0], *node, whole, [&](const std::shared_ptr<FormExprNode>& summand) {
                            const auto saved = shape_domain;
                            shape_domain = domain;
                            const auto p = self(self, summand);
                            shape_domain = saved;
                            return p.deriv + p.primal * shapeMeasureVariation(domain, *node);
                        });
                    break;
                }
                case FormExprType::InteriorFaceIntegral: {
                    if (node->cutVolumeSide().has_value()) {
                        refuseCompleteShapeDerivative(*node, "does not differentiate across a moving cut topology "
                                                             "(cut-adjacent facets are a separate capability)");
                    }
                    const auto a = inDomain(IntegralDomain::InteriorFace, 0);
                    out.primal = wrapInteriorFaceIntegral(a.primal, *node);
                    out.deriv = wrapInteriorFaceIntegral(
                        a.deriv + a.primal * shapeMeasureVariation(IntegralDomain::InteriorFace, *node), *node);
                    break;
                }
                default:
                    handled = false;
                    break;
            }
            if (handled) {
                active_memo.emplace(node.get(), out);
                return out;
            }
        }

        const auto isWrtTrial = [&](const FormExprNode::SpaceSignature& sig, const std::string& name) -> bool {
            switch (cfg.kind) {
                case DiffWrtKind::ActiveTrialFunction:
                    return true;
                case DiffWrtKind::SpecificTrialFunction:
                    return cfg.trial_sig.has_value() && cfg.trial_name.has_value() &&
                           spaceSignatureEqual(sig, *cfg.trial_sig) && name == *cfg.trial_name;
                case DiffWrtKind::FieldId:
                    if (cfg.active_trial_field.has_value() && cfg.field_id.has_value() &&
                        *cfg.active_trial_field == *cfg.field_id) {
                        return true;
                    }
                    return cfg.field_id.has_value() && (cfg.trial_state_field == *cfg.field_id);
                default:
                    return false;
            }
        };

        const auto isWrtField = [&](const FormExprNode::SpaceSignature& sig, FieldId fid, const std::string& name) -> bool {
            if (cfg.kind != DiffWrtKind::FieldId || !cfg.field_id.has_value() || fid != *cfg.field_id) {
                return false;
            }

            if (wrt_field_sig.has_value()) {
                if (!spaceSignatureEqual(sig, *wrt_field_sig)) {
                    throw std::invalid_argument("differentiateResidual: target FieldId appears with inconsistent SpaceSignature");
                }
            } else {
                wrt_field_sig = sig;
            }

            if (wrt_field_name.has_value()) {
                if (name != *wrt_field_name) {
                    throw std::invalid_argument("differentiateResidual: target FieldId appears with inconsistent symbol name");
                }
            } else {
                wrt_field_name = name;
                wrt_field_delta_name = makeDeltaName(name);
            }

            if (!wrt_field_delta_name.has_value()) {
                throw std::logic_error("differentiateResidual: missing delta name for FieldId target");
            }
            return true;
        };

        switch (node->type()) {
            // ---- Terminals (treated as constants) ----
            case FormExprType::Constant:
            case FormExprType::ParameterSymbol:
            case FormExprType::ParameterRef:
            case FormExprType::BoundaryIntegralSymbol:
            case FormExprType::BoundaryIntegralRef:
            case FormExprType::AuxiliaryStateSymbol:
            case FormExprType::AuxiliaryStateRef:
            case FormExprType::AuxiliaryInputSymbol:
            case FormExprType::AuxiliaryInputRef:
            case FormExprType::AuxiliaryOutputSymbol:
            case FormExprType::MaterialStateOldRef:
            case FormExprType::MaterialStateWorkRef:
            case FormExprType::GeometryTrialVectorVariation:
            case FormExprType::GeometryTrialJacobianVariation:
            case FormExprType::MeshVelocityVariation:
            case FormExprType::CurrentMeasureVariation:
            case FormExprType::CurrentNormalVariation:
            case FormExprType::SurfaceJacobianVariation:
            case FormExprType::CellDiameterVariation:
            case FormExprType::CellVolumeVariation:
            case FormExprType::FacetAreaVariation:
            case FormExprType::Time:
            case FormExprType::TimeStep:
            case FormExprType::EffectiveTimeStep:
            case FormExprType::Identity:
            case FormExprType::CellDiameter:
            case FormExprType::CellVolume:
            case FormExprType::CellDomainId:
            case FormExprType::BoundaryNormal:
                out.primal = FormExpr(node);
                out.deriv = zeroOf(out.primal);
                break;

            case FormExprType::PreviousSolutionRef:
                out.primal = FormExpr(node);
                out.deriv = zeroOf(out.primal);
                if (cfg.cut_point.reference.isValid()) {
                    out.deriv = spatialDirectionalDerivativeOfValue(out.primal, cfg.cut_point.field,
                                                                    cfg.cut_point.history_space);
                }
                break;

            case FormExprType::Jacobian:
            case FormExprType::JacobianInverse:
            case FormExprType::JacobianDeterminant:
                out.primal = FormExpr(node);
                out.deriv = zeroOf(out.primal);
                if (cfg.cut_point.jacobian.isValid()) {
                    if (node->type() == FormExprType::Jacobian) {
                        out.deriv = cfg.cut_point.jacobian;
                    } else if (node->type() == FormExprType::JacobianInverse) {
                        out.deriv = cfg.cut_point.inverse_jacobian;
                    } else {
                        out.deriv = out.primal * cfg.cut_point.relative_determinant;
                    }
                }
                break;

            case FormExprType::Coefficient:
                out.primal = FormExpr(node);
                out.deriv = zeroOf(out.primal);
                if (cfg.kind == DiffWrtKind::SpatialCoordinate && cfg.spatial_direction.isValid()) {
                    if (node->scalarCoefficient() || node->timeScalarCoefficient()) {
                        out.deriv = inner(out.primal.grad(), cfg.spatial_direction);
                    } else if (node->vectorCoefficient()) {
                        out.deriv = out.primal.grad() * cfg.spatial_direction;
                    } else {
                        throw std::invalid_argument(
                            "differentiated cut point requires a spatial derivative for this coefficient rank");
                    }
                    if (cfg.cut_point.out_of_plane_callback_motion) {
                        out.deriv = out.deriv + cfg.cut_point.out_of_plane_callback_motion(*node);
                    }
                }
                break;

            case FormExprType::FacetArea:
                out.primal = FormExpr(node);
                out.deriv =
                    cfg.cut_point.facet_measure.isValid() ? cfg.cut_point.facet_measure : zeroOf(out.primal);
                break;

            case FormExprType::Coordinate:
                out.primal = FormExpr(node);
                if (cfg.kind == DiffWrtKind::SpatialCoordinate &&
                    cfg.spatial_direction.isValid()) {
                    out.deriv = cfg.spatial_direction;
                } else {
                    out.deriv = zeroOf(out.primal);
                }
                break;

            case FormExprType::ReferenceCoordinate:
                out.primal = FormExpr(node);
                out.deriv = cfg.cut_point.reference.isValid() ? cfg.cut_point.reference : zeroOf(out.primal);
                break;

            case FormExprType::Normal:
                out.primal = FormExpr(node);
                if (cfg.kind == DiffWrtKind::NormalTerminal &&
                    cfg.normal_direction.isValid()) {
                    out.deriv = cfg.normal_direction;
                } else {
                    out.deriv = zeroOf(out.primal);
                }
                break;

            case FormExprType::MeshDisplacement:
            case FormExprType::MeshVelocity:
            case FormExprType::MeshAcceleration:
            case FormExprType::CurrentCoordinate:
            case FormExprType::PreviousCoordinate:
            case FormExprType::ReferencePhysicalCoordinate:
            case FormExprType::PreviousMeshVelocity:
            case FormExprType::PredictedMeshVelocity:
            case FormExprType::CurrentJacobian:
            case FormExprType::ReferenceJacobian:
            case FormExprType::CurrentJacobianDeterminant:
            case FormExprType::ReferenceJacobianDeterminant:
            case FormExprType::CurrentNormal:
            case FormExprType::CurrentMeanCurvature:
            case FormExprType::ReferenceNormal:
            case FormExprType::CurrentMeasure:
            case FormExprType::ReferenceMeasure:
            case FormExprType::SurfaceJacobian: {
                out.primal = FormExpr(node);
                if (cfg.cut_point.reference.isValid()) {
                    if (node->type() == FormExprType::CurrentCoordinate) {
                        out.deriv = FormExpr::currentJacobian() * cfg.cut_point.reference;
                    } else if (node->type() == FormExprType::ReferencePhysicalCoordinate) {
                        out.deriv = FormExpr::referenceJacobian() * cfg.cut_point.reference;
                    } else {
                        throw std::invalid_argument("differentiated cut surface has no spatial derivative for frame terminal " + node->toString());
                    }
                    break;
                }
                if (node->type() == FormExprType::CurrentCoordinate &&
                    cfg.kind == DiffWrtKind::SpatialCoordinate &&
                    cfg.spatial_direction.isValid()) {
                    out.deriv = cfg.spatial_direction;
                    break;
                }
                if (!differentiatesMeshGeometry(cfg)) {
                    out.deriv = zeroOf(out.primal);
                    break;
                }

                switch (node->type()) {
                    case FormExprType::MeshDisplacement:
                    case FormExprType::CurrentCoordinate:
                        out.deriv = FormExpr::geometryTrialVectorVariation();
                        break;
                    case FormExprType::MeshVelocity:
                        out.deriv = FormExpr::meshVelocityVariation();
                        break;
                    case FormExprType::CurrentJacobian:
                        out.deriv = FormExpr::geometryTrialJacobianVariation();
                        break;
                    case FormExprType::CurrentJacobianDeterminant:
                    case FormExprType::CurrentMeasure:
                        out.deriv = FormExpr::currentMeasureVariation();
                        break;
                    case FormExprType::CurrentNormal:
                        out.deriv = FormExpr::currentNormalVariation();
                        break;
                    case FormExprType::SurfaceJacobian:
                        out.deriv = FormExpr::surfaceJacobianVariation();
                        break;
                    default:
                        out.deriv = zeroOf(out.primal);
                        break;
                }
                break;
            }

            // AuxiliaryOutputRef: constant unless we're differentiating w.r.t.
            // a specific auxiliary output slot.
            case FormExprType::AuxiliaryOutputRef: {
                out.primal = FormExpr(node);
                if (cfg.kind == DiffWrtKind::AuxiliaryOutputSlot &&
                    cfg.aux_output_slot.has_value()) {
                    const auto slot = node->slotIndex();
                    if (slot && *slot == *cfg.aux_output_slot) {
                        // d(output_k)/d(output_k) = 1 (Kronecker delta)
                        out.deriv = FormExpr::constant(1.0);
                    } else {
                        out.deriv = zeroOf(out.primal);
                    }
                } else {
                    out.deriv = zeroOf(out.primal);
                }
                break;
            }

            case FormExprType::BoundaryFunctionalSymbol: {
                if (kids.size() != 1u || !kids[0]) {
                    throw std::invalid_argument("differentiateResidual: BoundaryFunctionalSymbol missing integrand");
                }
                const int marker = node->boundaryMarker().value_or(-1);
                const auto name = node->symbolName();
                if (marker < 0 || !name || name->empty()) {
                    throw std::invalid_argument("differentiateResidual: BoundaryFunctionalSymbol missing metadata");
                }
                const auto a = diff1(0);
                out.primal = FormExpr::boundaryIntegral(a.primal, marker, std::string(*name));
                out.deriv = zeroOf(out.primal);
                break;
            }

            case FormExprType::TestFunction:
                out.primal = FormExpr::testFunction(*node->spaceSignature(), node->toString());
                if (cfg.kind == DiffWrtKind::SpatialCoordinate &&
                    cfg.spatial_direction.isValid()) {
                    out.deriv = spatialDirectionalDerivativeOfValue(
                        out.primal,
                        cfg.cut_point.field.isValid() ? cfg.cut_point.field : cfg.spatial_direction,
                        *node->spaceSignature());
                } else {
                    out.deriv = zeroOf(out.primal);
                }
                break;

            case FormExprType::TrialFunction: {
                const auto* sig = node->spaceSignature();
                if (!sig) {
                    throw std::invalid_argument("differentiateResidual: TrialFunction missing SpaceSignature");
                }
                const auto name = node->toString();
                out.primal = FormExpr::stateField(cfg.trial_state_field, *sig, name);
                if (cfg.kind == DiffWrtKind::SpatialCoordinate &&
                    cfg.spatial_direction.isValid()) {
                    out.deriv = spatialDirectionalDerivativeOfValue(
                        out.primal,
                        cfg.cut_point.field.isValid() ? cfg.cut_point.field : cfg.spatial_direction,
                        *sig);
                } else if (isWrtTrial(*sig, name)) {
                    out.deriv = FormExpr::trialFunction(*sig, makeDeltaName(name));
                } else {
                    out.deriv = zeroOf(out.primal);
                }
                break;
            }

            case FormExprType::DiscreteField: {
                const auto fid = node->fieldId();
                const auto* sig = node->spaceSignature();
                if (!fid || !sig) {
                    throw std::invalid_argument("differentiateResidual: DiscreteField missing metadata");
                }
                out.primal = FormExpr::discreteField(*fid, *sig, node->toString());
                if (cfg.kind == DiffWrtKind::SpatialCoordinate &&
                    cfg.spatial_direction.isValid()) {
                    out.deriv = spatialDirectionalDerivativeOfValue(
                        out.primal,
                        cfg.cut_point.field.isValid() ? cfg.cut_point.field : cfg.spatial_direction,
                        *sig);
                } else if (isWrtField(*sig, *fid, node->toString())) {
                    out.deriv = FormExpr::trialFunction(*sig, *wrt_field_delta_name);
                } else {
                    out.deriv = zeroOf(out.primal);
                }
                break;
            }

            case FormExprType::StateField: {
                const auto fid = node->fieldId();
                const auto* sig = node->spaceSignature();
                if (!fid || !sig) {
                    throw std::invalid_argument("differentiateResidual: StateField missing metadata");
                }
                out.primal = FormExpr::stateField(*fid, *sig, node->toString());
                if (cfg.kind == DiffWrtKind::SpatialCoordinate &&
                    cfg.spatial_direction.isValid()) {
                    out.deriv = spatialDirectionalDerivativeOfValue(
                        out.primal,
                        cfg.cut_point.field.isValid() ? cfg.cut_point.field : cfg.spatial_direction,
                        *sig);
                } else if (isWrtField(*sig, *fid, node->toString())) {
                    out.deriv = FormExpr::trialFunction(*sig, *wrt_field_delta_name);
                } else {
                    out.deriv = zeroOf(out.primal);
                }
                break;
            }

            // ---- Unary ops ----
            case FormExprType::Negate: {
                const auto a = diff1(0);
                out.primal = -a.primal;
                out.deriv = -a.deriv;
                break;
            }
            case FormExprType::Gradient: {
                const auto a = diff1(0);
                out.primal = a.primal.grad();
                if (cfg.kind == DiffWrtKind::SpatialCoordinate &&
                    cfg.spatial_direction.isValid() &&
                    a.primal.isValid() &&
                    a.primal.node() != nullptr &&
                    isDefinitelyScalarNode(*a.primal.node())) {
                    // Moving quadrature points sample grad(u) at x + w; the
                    // local point derivative is Hessian(u) * w. Using
                    // grad(grad(u) . w) would also differentiate w itself.
                    out.deriv = a.primal.hessian() * cfg.spatial_direction;
                    if (cfg.cut_point.history_space && cfg.cut_point.history_space->topological_dimension == 2 &&
                        a.primal.node()->type() != FormExprType::Coefficient) {
                        // Assembly stores A^T grad_ref(u) with A the leading 2x2
                        // block of the inverse of the completed surface Jacobian.
                        // Its physical Hessian uses the planar connection -A dJ_2 A.
                        // Contract with the chart velocity, then replace that connection
                        // by the actual derivative dA of the embedded parent map.
                        const auto A = FormExpr::jacobianInverse();
                        const auto connection_change = A.inv() * cfg.cut_point.inverse_jacobian +
                                                       cfg.cut_point.jacobian * A;
                        out.deriv = a.primal.hessian() * cfg.cut_point.field +
                                    connection_change.transpose() * a.primal.grad();
                    }
                } else {
                    out.deriv = a.deriv.grad();
                }
                break;
            }
            case FormExprType::Divergence: {
                const auto a = diff1(0);
                out.primal = a.primal.div();
                out.deriv = a.deriv.div();
                break;
            }
            case FormExprType::Curl: {
                const auto a = diff1(0);
                out.primal = a.primal.curl();
                out.deriv = a.deriv.curl();
                break;
            }
            case FormExprType::Hessian: {
                const auto a = diff1(0);
                out.primal = a.primal.hessian();
                out.deriv = a.deriv.hessian();
                break;
            }
            case FormExprType::TimeDerivative: {
                const int order = node->timeDerivativeOrder().value_or(1);
                const auto a = diff1(0);
                out.primal = a.primal.dt(order);
                if (a.deriv.isValid() && a.deriv.node() != nullptr && isZeroLike(*a.deriv.node())) {
                    out.deriv = a.deriv;
                } else {
                    out.deriv = a.deriv.dt(order);
                }
                break;
            }
            case FormExprType::RestrictMinus: {
                const auto a = diff1(0);
                out.primal = a.primal.minus();
                out.deriv = (a.deriv.isValid() && a.deriv.node() && isZeroLike(*a.deriv.node()))
                    ? a.deriv
                    : a.deriv.minus();
                break;
            }
            case FormExprType::RestrictPlus: {
                const auto a = diff1(0);
                out.primal = a.primal.plus();
                out.deriv = (a.deriv.isValid() && a.deriv.node() && isZeroLike(*a.deriv.node()))
                    ? a.deriv
                    : a.deriv.plus();
                break;
            }
            case FormExprType::Jump: {
                const auto a = diff1(0);
                out.primal = a.primal.jump();
                out.deriv = a.deriv.jump();
                break;
            }
            case FormExprType::Average: {
                const auto a = diff1(0);
                out.primal = a.primal.avg();
                out.deriv = a.deriv.avg();
                break;
            }
            case FormExprType::Pullback:
            case FormExprType::Pushforward: {
                const auto a = diff1(0);
                const auto from = node->fromConfiguration().value_or(GeometryConfiguration::Reference);
                const auto to = node->toConfiguration().value_or(GeometryConfiguration::Current);
                if (node->type() == FormExprType::Pullback) {
                    out.primal = pullback(a.primal, from, to);
                    out.deriv = pullback(a.deriv, from, to);
                } else {
                    out.primal = pushforward(a.primal, from, to);
                    out.deriv = pushforward(a.deriv, from, to);
                }
                break;
            }

            // ---- Constructors / indexing ----
            case FormExprType::AsVector: {
                std::vector<FormExpr> pv;
                std::vector<FormExpr> dv;
                pv.reserve(kids.size());
                dv.reserve(kids.size());
                for (const auto& k : kids) {
                    const auto r = self(self, k);
                    pv.push_back(r.primal);
                    dv.push_back(r.deriv);
                }
                out.primal = FormExpr::asVector(std::move(pv));
                out.deriv = FormExpr::asVector(std::move(dv));
                break;
            }
            case FormExprType::AsTensor: {
                const int rows = node->tensorRows().value_or(0);
                const int cols = node->tensorCols().value_or(0);
                if (rows <= 0 || cols <= 0) {
                    throw std::invalid_argument("differentiateResidual: AsTensor missing (rows,cols)");
                }
                if (kids.size() != static_cast<std::size_t>(rows * cols)) {
                    throw std::invalid_argument("differentiateResidual: AsTensor children size mismatch");
                }

                std::vector<std::vector<FormExpr>> prow(static_cast<std::size_t>(rows));
                std::vector<std::vector<FormExpr>> drow(static_cast<std::size_t>(rows));
                for (int r = 0; r < rows; ++r) {
                    prow[static_cast<std::size_t>(r)].reserve(static_cast<std::size_t>(cols));
                    drow[static_cast<std::size_t>(r)].reserve(static_cast<std::size_t>(cols));
                    for (int c = 0; c < cols; ++c) {
                        const auto idx = static_cast<std::size_t>(r * cols + c);
                        const auto rr = self(self, kids[idx]);
                        prow[static_cast<std::size_t>(r)].push_back(rr.primal);
                        drow[static_cast<std::size_t>(r)].push_back(rr.deriv);
                    }
                }
                out.primal = FormExpr::asTensor(std::move(prow));
                out.deriv = FormExpr::asTensor(std::move(drow));
                break;
            }
            case FormExprType::Component: {
                const int i = node->componentIndex0().value_or(0);
                const int j = node->componentIndex1().value_or(-1);
                const auto a = diff1(0);
                out.primal = a.primal.component(i, j);
                out.deriv = (a.deriv.isValid() && a.deriv.node() != nullptr && isZeroLike(*a.deriv.node()))
                    ? FormExpr::typedZero()
                    : a.deriv.component(i, j);
                break;
            }
            case FormExprType::IndexedAccess: {
                const int rank = node->indexRank().value_or(0);
                const auto ids_opt = node->indexIds();
                const auto ext_opt = node->indexExtents();
                const auto names_opt = node->indexNames();
                const auto vars_opt = node->indexVariances();
                if (rank <= 0 || !ids_opt || !ext_opt) {
                    throw std::invalid_argument("differentiateResidual: IndexedAccess missing index metadata");
                }

                std::array<std::string, 4> names{};
                if (names_opt) {
                    for (std::size_t k = 0; k < names.size(); ++k) {
                        names[k] = std::string((*names_opt)[k]);
                    }
                }
                std::array<tensor::IndexVariance, 4> vars{};
                vars.fill(tensor::IndexVariance::None);
                if (vars_opt) {
                    vars = *vars_opt;
                }

                const auto ext = resolveAutoIndexExtents(*ext_opt, rank, auto_extent);
                const auto a = diff1(0);
                out.primal = FormExpr::indexedAccessRawWithMetadata(a.primal, rank, *ids_opt, ext, vars, names);
                out.deriv = (a.deriv.isValid() && a.deriv.node() != nullptr && isZeroLike(*a.deriv.node()))
                    ? FormExpr::constant(0.0)
                    : FormExpr::indexedAccessRawWithMetadata(a.deriv, rank, *ids_opt, ext, vars, names);
                break;
            }

            // ---- Algebra ----
            case FormExprType::Add: {
                const auto a = diff1(0);
                const auto b = diff1(1);
                out.primal = a.primal + b.primal;
                out.deriv = a.deriv + b.deriv;
                break;
            }
            case FormExprType::Subtract: {
                const auto a = diff1(0);
                const auto b = diff1(1);
                out.primal = a.primal - b.primal;
                out.deriv = a.deriv - b.deriv;
                break;
            }
            case FormExprType::Multiply: {
                const auto a = diff1(0);
                const auto b = diff1(1);
                out.primal = a.primal * b.primal;
                out.deriv = (a.deriv * b.primal) + (a.primal * b.deriv);
                break;
            }
            case FormExprType::Divide: {
                const auto a = diff1(0);
                const auto b = diff1(1);
                out.primal = a.primal / b.primal;
                out.deriv = ((a.deriv * b.primal) - (a.primal * b.deriv)) / (b.primal * b.primal);
                break;
            }
            case FormExprType::InnerProduct: {
                const auto a = diff1(0);
                const auto b = diff1(1);
                out.primal = inner(a.primal, b.primal);
                out.deriv = inner(a.deriv, b.primal) + inner(a.primal, b.deriv);
                break;
            }
            case FormExprType::DoubleContraction: {
                const auto a = diff1(0);
                const auto b = diff1(1);
                out.primal = a.primal.doubleContraction(b.primal);
                out.deriv = a.deriv.doubleContraction(b.primal) + a.primal.doubleContraction(b.deriv);
                break;
            }
            case FormExprType::OuterProduct: {
                const auto a = diff1(0);
                const auto b = diff1(1);
                out.primal = outer(a.primal, b.primal);
                out.deriv = outer(a.deriv, b.primal) + outer(a.primal, b.deriv);
                break;
            }
            case FormExprType::CrossProduct: {
                const auto a = diff1(0);
                const auto b = diff1(1);
                out.primal = cross(a.primal, b.primal);
                out.deriv = cross(a.deriv, b.primal) + cross(a.primal, b.deriv);
                break;
            }
            case FormExprType::Power: {
                const auto a = diff1(0);
                const auto b = diff1(1);
                out.primal = pow(a.primal, b.primal);

                out.deriv = differentiatePower(a.primal, b.primal, a.deriv, b.deriv);
                break;
            }
            case FormExprType::Minimum: {
                const auto a = diff1(0);
                const auto b = diff1(1);
                out.primal = min(a.primal, b.primal);
                const auto cond = le(a.primal, b.primal);
                out.deriv = cond.conditional(a.deriv, b.deriv);
                break;
            }
            case FormExprType::Maximum: {
                const auto a = diff1(0);
                const auto b = diff1(1);
                out.primal = max(a.primal, b.primal);
                const auto cond = ge(a.primal, b.primal);
                out.deriv = cond.conditional(a.deriv, b.deriv);
                break;
            }

            case FormExprType::SmoothMin:
            case FormExprType::SmoothMax: {
                if (kids.size() != 3u || !kids[0] || !kids[1] || !kids[2]) {
                    throw std::invalid_argument("differentiateResidual: SmoothMin/SmoothMax expects 3 children");
                }
                const auto a = self(self, kids[0]);
                const auto b = self(self, kids[1]);
                const auto eps = self(self, kids[2]);
                const bool is_min = (node->type() == FormExprType::SmoothMin);
                out.primal = is_min ? a.primal.smoothMin(b.primal, eps.primal) : a.primal.smoothMax(b.primal, eps.primal);

                const auto d = a.primal - b.primal;
                const auto ad = d.smoothAbs(eps.primal); // sqrt(d^2 + eps^2)
                const auto num = d * (a.deriv - b.deriv) + eps.primal * eps.deriv;
                const auto dad = eq(ad, FormExpr::constant(0.0)).conditional(zeroOf(ad), num / ad);

                out.deriv = FormExpr::constant(0.5) * (a.deriv + b.deriv) +
                            (is_min ? FormExpr::constant(-0.5) : FormExpr::constant(0.5)) * dad;
                break;
            }

            // ---- Predicates ----
            case FormExprType::Less:
            case FormExprType::LessEqual:
            case FormExprType::Greater:
            case FormExprType::GreaterEqual:
            case FormExprType::Equal:
            case FormExprType::NotEqual: {
                const auto a = diff1(0);
                const auto b = diff1(1);
                switch (node->type()) {
                    case FormExprType::Less: out.primal = lt(a.primal, b.primal); break;
                    case FormExprType::LessEqual: out.primal = le(a.primal, b.primal); break;
                    case FormExprType::Greater: out.primal = gt(a.primal, b.primal); break;
                    case FormExprType::GreaterEqual: out.primal = ge(a.primal, b.primal); break;
                    case FormExprType::Equal: out.primal = eq(a.primal, b.primal); break;
                    case FormExprType::NotEqual: out.primal = ne(a.primal, b.primal); break;
                    default: break;
                }
                out.deriv = zeroOf(out.primal);
                break;
            }
            case FormExprType::Conditional: {
                if (kids.size() != 3u || !kids[0] || !kids[1] || !kids[2]) {
                    throw std::invalid_argument("differentiateResidual: Conditional expects 3 children");
                }
                const auto c = self(self, kids[0]);
                const auto t = self(self, kids[1]);
                const auto f = self(self, kids[2]);
                out.primal = c.primal.conditional(t.primal, f.primal);
                out.deriv = c.primal.conditional(t.deriv, f.deriv);
                break;
            }

            // ---- Tensor ops / scalar functions ----
            case FormExprType::Transpose: {
                const auto a = diff1(0);
                out.primal = a.primal.transpose();
                out.deriv = a.deriv.transpose();
                break;
            }
            case FormExprType::Trace: {
                const auto a = diff1(0);
                out.primal = a.primal.trace();
                out.deriv = a.deriv.trace();
                break;
            }
            case FormExprType::Determinant: {
                const auto a = diff1(0);
                out.primal = det(a.primal);
                // Use cofactor-based rule to avoid singular-matrix inversion at det(A)=0:
                //   d(det(A)) = cofactor(A) : dA
                out.deriv = a.primal.cofactor().doubleContraction(a.deriv);
                break;
            }
            case FormExprType::Inverse: {
                const auto a = diff1(0);
                out.primal = inv(a.primal);
                out.deriv = -(out.primal * a.deriv * out.primal);
                break;
            }
            case FormExprType::Cofactor: {
                const auto a = diff1(0);
                const auto detA = det(a.primal);
                const auto invA = inv(a.primal);
                const auto invAT = transpose(invA);
                out.primal = a.primal.cofactor();
                const auto ddet = detA * inner(inv(transpose(a.primal)), a.deriv);
                const auto dinv = -(invA * a.deriv * invA);
                out.deriv = ddet * invAT + detA * transpose(dinv);
                break;
            }
            case FormExprType::Deviator: {
                const auto a = diff1(0);
                out.primal = a.primal.dev();
                out.deriv = a.deriv.dev();
                break;
            }
            case FormExprType::SymmetricPart: {
                const auto a = diff1(0);
                out.primal = a.primal.sym();
                out.deriv = a.deriv.sym();
                break;
            }
            case FormExprType::SkewPart: {
                const auto a = diff1(0);
                out.primal = a.primal.skew();
                out.deriv = a.deriv.skew();
                break;
            }
            case FormExprType::Norm: {
                const auto a = diff1(0);
                out.primal = a.primal.norm();
                const auto nrm = out.primal;
                const auto d = inner(a.primal, a.deriv) / nrm;
                out.deriv = eq(nrm, FormExpr::constant(0.0)).conditional(FormExpr::constant(0.0), d);
                break;
            }
            case FormExprType::Normalize: {
                const auto a = diff1(0);
                out.primal = a.primal.normalize();
                const auto nrm = a.primal.norm();
                const auto dnrm = inner(a.primal, a.deriv) / nrm;
                const auto d = (a.deriv * nrm - a.primal * dnrm) / (nrm * nrm);
                out.deriv = eq(nrm, FormExpr::constant(0.0)).conditional(zeroOf(out.primal), d);
                break;
            }
            case FormExprType::AbsoluteValue: {
                const auto a = diff1(0);
                out.primal = a.primal.abs();
                // Match Dual::abs behavior: pick +da when a >= 0, otherwise -da (at a==0, Dual picks +).
                const auto is_nonneg = ge(a.primal, FormExpr::constant(0.0)); // 0/1
                const auto sign = (FormExpr::constant(2.0) * is_nonneg) - FormExpr::constant(1.0); // +1 or -1
                out.deriv = sign * a.deriv;
                break;
            }
            case FormExprType::Sign: {
                const auto a = diff1(0);
                out.primal = a.primal.sign();
                out.deriv = zeroOf(out.primal);
                break;
            }
            case FormExprType::Sqrt: {
                const auto a = diff1(0);
                out.primal = a.primal.sqrt();
                // Match Dual::sqrt behavior: avoid division by zero by using denom=1 when sqrt(..)==0.
                const auto is_zero = eq(out.primal, FormExpr::constant(0.0)); // 0/1
                const auto denom = (FormExpr::constant(2.0) * out.primal) + is_zero;
                out.deriv = a.deriv / denom;
                break;
            }
            case FormExprType::Exp: {
                const auto a = diff1(0);
                out.primal = a.primal.exp();
                out.deriv = out.primal * a.deriv;
                break;
            }
            case FormExprType::Log: {
                const auto a = diff1(0);
                out.primal = a.primal.log();
                out.deriv = a.deriv / a.primal;
                break;
            }

            case FormExprType::MatrixExponential: {
                const auto a = diff1(0);
                out.primal = a.primal.matrixExp();
                out.deriv = (a.deriv.isValid() && a.deriv.node() != nullptr && isZeroLike(*a.deriv.node()))
                                ? zeroOf(out.primal)
                                : FormExpr::matrixExpDirectionalDerivative(a.primal, a.deriv);
                break;
            }
            case FormExprType::MatrixLogarithm: {
                const auto a = diff1(0);
                out.primal = a.primal.matrixLog();
                out.deriv = (a.deriv.isValid() && a.deriv.node() != nullptr && isZeroLike(*a.deriv.node()))
                                ? zeroOf(out.primal)
                                : FormExpr::matrixLogDirectionalDerivative(a.primal, a.deriv);
                break;
            }
            case FormExprType::MatrixSqrt: {
                const auto a = diff1(0);
                out.primal = a.primal.matrixSqrt();
                out.deriv = (a.deriv.isValid() && a.deriv.node() != nullptr && isZeroLike(*a.deriv.node()))
                                ? zeroOf(out.primal)
                                : FormExpr::matrixSqrtDirectionalDerivative(a.primal, a.deriv);
                break;
            }

            case FormExprType::SmoothAbsoluteValue: {
                const auto x = diff1(0);
                const auto eps = diff1(1);
                out.primal = x.primal.smoothAbs(eps.primal); // sqrt(x^2+eps^2)
                const auto denom = out.primal;
                const auto num = x.primal * x.deriv + eps.primal * eps.deriv;
                const auto d = num / denom;
                out.deriv = eq(denom, FormExpr::constant(0.0)).conditional(zeroOf(out.primal), d);
                break;
            }
            case FormExprType::SmoothSign:
            case FormExprType::SmoothHeaviside: {
                const auto x = diff1(0);
                const auto eps = diff1(1);
                const bool is_heaviside = (node->type() == FormExprType::SmoothHeaviside);
                out.primal = is_heaviside ? x.primal.smoothHeaviside(eps.primal) : x.primal.smoothSign(eps.primal);

                const auto denom = x.primal.smoothAbs(eps.primal);
                const auto num_denom = x.primal * x.deriv + eps.primal * eps.deriv;
                const auto d_denom = num_denom / denom;
                const auto d_sign = (x.deriv * denom - x.primal * d_denom) / (denom * denom);
                const auto d = is_heaviside ? (FormExpr::constant(0.5) * d_sign) : d_sign;
                out.deriv = eq(denom, FormExpr::constant(0.0)).conditional(zeroOf(out.primal), d);
                break;
            }

            case FormExprType::SymmetricEigenvalue: {
                const auto a = diff1(0);
                const int which = node->eigenIndex().value_or(0);
                out.primal = a.primal.symmetricEigenvalue(which);
                out.deriv = FormExpr::symmetricEigenvalueDirectionalDerivative(a.primal, a.deriv, which);
                break;
            }

            case FormExprType::Eigenvalue: {
                const auto a = diff1(0);
                const int which = node->eigenIndex().value_or(0);
                out.primal = a.primal.eigenvalue(which);
                out.deriv = FormExpr::symmetricEigenvalueDirectionalDerivative(a.primal, a.deriv, which);
                break;
            }

            case FormExprType::SymmetricEigenvector: {
                const auto a = diff1(0);
                const int which = node->eigenIndex().value_or(0);
                out.primal = a.primal.symmetricEigenvector(which);
                out.deriv = FormExpr::symmetricEigenvectorDirectionalDerivative(a.primal, a.deriv, which);
                break;
            }

            case FormExprType::SpectralDecomposition: {
                const auto a = diff1(0);
                out.primal = a.primal.spectralDecomposition();
                out.deriv = FormExpr::spectralDecompositionDirectionalDerivative(a.primal, a.deriv);
                break;
            }

            case FormExprType::SymmetricEigenvalueDirectionalDerivative: {
                const auto a = diff1(0);
                const auto b = diff1(1);
                const int which = node->eigenIndex().value_or(0);
                out.primal = FormExpr::symmetricEigenvalueDirectionalDerivative(a.primal, b.primal, which);
                out.deriv = FormExpr::symmetricEigenvalueDirectionalDerivativeWrtA(a.primal, b.primal, a.deriv, which) +
                            FormExpr::symmetricEigenvalueDirectionalDerivative(a.primal, b.deriv, which);
                break;
            }

            case FormExprType::MatrixPower: {
                const auto A = diff1(0);
                const auto p = diff1(1);
                out.primal = A.primal.matrixPow(p.primal);
                const auto dA_term =
                    (A.deriv.isValid() && A.deriv.node() != nullptr && isZeroLike(*A.deriv.node()))
                        ? zeroOf(out.primal)
                        : FormExpr::matrixPowDirectionalDerivative(A.primal, A.deriv, p.primal);
                const auto dp_term =
                    (p.deriv.isValid() && p.deriv.node() != nullptr && isZeroLike(*p.deriv.node()))
                        ? zeroOf(out.primal)
                        : (p.deriv * (out.primal * A.primal.matrixLog()));
                out.deriv = dA_term + dp_term;
                break;
            }

            case FormExprType::HistoryWeightedSum:
            case FormExprType::HistoryConvolution: {
                std::vector<FormExpr> primal_weights;
                primal_weights.reserve(kids.size());
                out.deriv = {};
                for (std::size_t kk = 0; kk < kids.size(); ++kk) {
                    const auto w = self(self, kids[kk]);
                    primal_weights.push_back(w.primal);
                }
                out.primal = (node->type() == FormExprType::HistoryWeightedSum)
                                 ? FormExpr::historyWeightedSum(primal_weights)
                                 : FormExpr::historyConvolution(primal_weights);

                FormExpr sum = zeroOf(out.primal);
                for (std::size_t kk = 0; kk < kids.size(); ++kk) {
                    const auto w = self(self, kids[kk]);
                    if (w.deriv.isValid() && w.deriv.node() && isZeroLike(*w.deriv.node())) {
                        continue;
                    }
                    const int steps_back = static_cast<int>(kk + 1u);
                    sum = sum + (w.deriv * FormExpr::previousSolution(steps_back));
                }
                out.deriv = sum;
                break;
            }

            // ---- Constitutive hooks (must be inlined) ----
            case FormExprType::Constitutive:
            case FormExprType::ConstitutiveOutput:
                throw std::invalid_argument("differentiateResidual: Constitutive nodes must be inlined before differentiation");

            // ---- Measure wrappers ----
            case FormExprType::CellIntegral: {
                const auto a = diff1(0);
                out.primal = a.primal.dx();
                auto deriv_integrand = a.deriv;
                if (differentiatesMeshGeometry(cfg)) {
                    deriv_integrand = deriv_integrand +
                        a.primal * (FormExpr::currentMeasureVariation() / FormExpr::currentMeasure());
                }
                out.deriv = deriv_integrand.dx();
                break;
            }
            case FormExprType::BoundaryIntegral: {
                const auto a = diff1(0);
                out.primal = wrapExteriorIntegral(a.primal, *node);
                auto deriv_integrand = a.deriv;
                if (differentiatesMeshGeometry(cfg)) {
                    deriv_integrand = deriv_integrand +
                        a.primal * (FormExpr::currentMeasureVariation() / FormExpr::currentMeasure());
                }
                out.deriv = wrapExteriorIntegral(deriv_integrand, *node);
                break;
            }
            case FormExprType::InteriorFaceIntegral: {
                const auto a = diff1(0);
                out.primal = wrapInteriorFaceIntegral(a.primal, *node);
                auto deriv_integrand = a.deriv;
                if (differentiatesMeshGeometry(cfg)) {
                    deriv_integrand = deriv_integrand +
                        a.primal * (FormExpr::currentMeasureVariation() / FormExpr::currentMeasure());
                }
                out.deriv = wrapInteriorFaceIntegral(deriv_integrand, *node);
                break;
            }
            case FormExprType::InterfaceIntegral: {
                const auto a = diff1(0);
                out.primal = wrapExteriorIntegral(a.primal, *node);
                const auto with_measure_variation = [&](const DiffPair& p) {
                    if (!differentiatesMeshGeometry(cfg)) {
                        return p.deriv;
                    }
                    return p.deriv +
                        p.primal * (FormExpr::currentMeasureVariation() / FormExpr::currentMeasure());
                };
                out.deriv = interfaceIntegralDerivative(
                    kids[0], *node, with_measure_variation(a),
                    [&](const std::shared_ptr<FormExprNode>& summand) {
                        return with_measure_variation(self(self, summand));
                    });
                break;
            }
            case FormExprType::CutVolumeIntegral: {
                const int marker = node->interfaceMarker().value_or(-1);
                const auto side = node->cutVolumeSide().value_or(CutVolumeSide::Negative);
                const auto a = diff1(0);
                out.primal = a.primal.dCutVolume(marker, side);
                out.deriv = a.deriv.dCutVolume(marker, side);
                break;
            }
            case FormExprType::GlobalIntegral: {
                // A point mass: no measure, so no geometry variation.
                const auto a = diff1(0);
                out.primal = a.primal.dGlobal();
                out.deriv = a.deriv.dGlobal();
                break;
            }

            default:
                throw std::invalid_argument("differentiateResidual: unsupported FormExprType " +
                                            std::to_string(static_cast<std::uint16_t>(node->type())));
        }

        active_memo.emplace(node.get(), out);
        return out;
    };

    const auto pair = diff(diff, residual_form.nodeShared());
    auto deriv = simplify(complete_shape && cfg.integrand_domain.has_value()
                              ? pair.deriv + pair.primal * shapeMeasureVariation(*cfg.integrand_domain,
                                                                                 *residual_form.node())
                              : pair.deriv);
    if (tensor::containsTensorCalculusNodes(deriv)) {
        deriv = tensor::postprocessTensorDerivative(deriv);
    }
    return deriv;
}

} // namespace

FormExpr differentiateResidual(const FormExpr& residual_form)
{
    DiffTargetConfig cfg;
    cfg.kind = DiffWrtKind::ActiveTrialFunction;
    cfg.trial_state_field = CURRENT_SOLUTION_FIELD_ID;
    return differentiateResidualImpl(residual_form, cfg);
}

FormExpr differentiateResidual(const FormExpr& residual_form,
                               const FormExpr& wrt_terminal,
                               FieldId trial_state_field)
{
    if (!wrt_terminal.isValid() || wrt_terminal.node() == nullptr) {
        throw std::invalid_argument("differentiateResidual: invalid wrt_terminal");
    }

    DiffTargetConfig cfg;
    cfg.trial_state_field = trial_state_field;

    switch (wrt_terminal.node()->type()) {
        case FormExprType::TrialFunction: {
            const auto* sig = wrt_terminal.node()->spaceSignature();
            if (!sig) {
                throw std::invalid_argument("differentiateResidual: wrt TrialFunction missing SpaceSignature");
            }
            cfg.kind = DiffWrtKind::SpecificTrialFunction;
            cfg.trial_sig = *sig;
            cfg.trial_name = wrt_terminal.node()->toString();
            break;
        }
        case FormExprType::DiscreteField:
        case FormExprType::StateField: {
            const auto fid = wrt_terminal.node()->fieldId();
            const auto* sig = wrt_terminal.node()->spaceSignature();
            if (!fid || !sig) {
                throw std::invalid_argument("differentiateResidual: wrt field terminal missing metadata");
            }
            cfg.kind = DiffWrtKind::FieldId;
            cfg.field_id = *fid;
            cfg.field_sig = *sig;
            cfg.field_name = wrt_terminal.node()->toString();
            break;
        }
        case FormExprType::AuxiliaryOutputRef: {
            const auto slot = wrt_terminal.node()->slotIndex();
            if (!slot) {
                throw std::invalid_argument("differentiateResidual: AuxiliaryOutputRef missing slot index");
            }
            cfg.kind = DiffWrtKind::AuxiliaryOutputSlot;
            cfg.aux_output_slot = *slot;
            break;
        }
        default:
            throw std::invalid_argument("differentiateResidual: wrt_terminal must be TrialFunction/StateField/DiscreteField/AuxiliaryOutputRef");
    }

    return differentiateResidualImpl(residual_form, cfg);
}

FormExpr differentiateResidual(const FormExpr& residual_form,
                               FieldId field,
                               FieldId trial_state_field)
{
    DiffTargetConfig cfg;
    cfg.kind = DiffWrtKind::FieldId;
    cfg.field_id = field;
    cfg.trial_state_field = trial_state_field;
    return differentiateResidualImpl(residual_form, cfg);
}

FormExpr differentiateResidual(const FormExpr& residual_form,
                               FieldId field,
                               FieldId trial_state_field,
                               GeometrySensitivityOptions geometry_sensitivity)
{
    if (geometry_sensitivity.mode == GeometrySensitivityMode::MeshMotionUnknowns &&
        geometry_sensitivity.mesh_motion_field == INVALID_FIELD_ID) {
        throw std::invalid_argument(
            "differentiateResidual: mesh-motion geometry sensitivity requires a valid mesh_motion_field");
    }
    if (geometry_sensitivity.mode ==
            GeometrySensitivityMode::LevelSetCutDomainUnknowns &&
        geometry_sensitivity.level_set_cut_domains.empty()) {
        throw std::invalid_argument(
            "differentiateResidual: level-set cut-domain geometry sensitivity requires at least one cut domain");
    }

    DiffTargetConfig cfg;
    cfg.kind = DiffWrtKind::FieldId;
    cfg.field_id = field;
    cfg.trial_state_field = trial_state_field;
    cfg.geometry_sensitivity = geometry_sensitivity;
    if (trial_state_field == CURRENT_SOLUTION_FIELD_ID) {
        cfg.active_trial_field = field;
    }
    return differentiateResidualImpl(residual_form, cfg);
}

FormExpr differentiateIntegrandShape(const FormExpr& integrand,
                                     IntegralDomain domain,
                                     FieldId trial_state_field,
                                     GeometrySensitivityOptions geometry_sensitivity)
{
    if (geometry_sensitivity.mode != GeometrySensitivityMode::MeshMotionUnknowns ||
        geometry_sensitivity.mesh_motion_field == INVALID_FIELD_ID ||
        geometry_sensitivity.shape_derivative != ShapeDerivative::Complete) {
        throw std::invalid_argument(
            "differentiateIntegrandShape: requires MeshMotionUnknowns with ShapeDerivative::Complete and a mesh-motion field");
    }
    DiffTargetConfig cfg;
    cfg.kind = DiffWrtKind::FieldId;
    cfg.field_id = geometry_sensitivity.mesh_motion_field;
    cfg.trial_state_field = trial_state_field;
    if (trial_state_field == CURRENT_SOLUTION_FIELD_ID) {
        cfg.active_trial_field = geometry_sensitivity.mesh_motion_field;
    }
    cfg.geometry_sensitivity = std::move(geometry_sensitivity);
    cfg.integrand_domain = domain;
    return differentiateResidualImpl(integrand, cfg);
}

FormExpr directionalDerivativeWrtField(const FormExpr& expr,
                                       FieldId field,
                                       const FormExpr& direction,
                                       bool discrete_fields_vary)
{
    if (!expr.isValid() || expr.node() == nullptr) {
        throw std::invalid_argument("directionalDerivativeWrtField: invalid expression");
    }
    if (!direction.isValid() || direction.node() == nullptr) {
        throw std::invalid_argument("directionalDerivativeWrtField: invalid direction expression");
    }

    const auto check = checkSymbolicDifferentiability(expr);
    if (!check.ok) {
        const auto msg = check.first_issue ? check.first_issue->message : std::string("SymbolicDiff: unsupported expression");
        throw std::invalid_argument("directionalDerivativeWrtField: " + msg);
    }

    std::unordered_map<const FormExprNode*, DiffPair> memo;

    std::optional<FormExprNode::SpaceSignature> wrt_field_sig{};
    std::optional<std::string> wrt_field_name{};

    const int auto_extent = inferAutoIndexExtent(expr);

    const auto diff = [&](const auto& self, const std::shared_ptr<FormExprNode>& node) -> DiffPair {
        if (!node) {
            throw std::invalid_argument("directionalDerivativeWrtField: encountered null node");
        }
        if (auto it = memo.find(node.get()); it != memo.end()) {
            return it->second;
        }

        DiffPair out;

        const auto kids = node->childrenShared();

        auto diff1 = [&](std::size_t k) -> DiffPair {
            if (kids.size() <= k || !kids[k]) {
                throw std::invalid_argument("directionalDerivativeWrtField: missing child");
            }
            return self(self, kids[k]);
        };

        auto zeroOf = [&](const FormExpr& ref) -> FormExpr {
            return FormExpr::constant(0.0) * ref;
        };

        const auto isWrtField = [&](const FormExprNode::SpaceSignature& sig, FieldId fid, const std::string& name) -> bool {
            if (fid != field) {
                return false;
            }

            if (wrt_field_sig.has_value()) {
                if (!spaceSignatureEqual(sig, *wrt_field_sig)) {
                    throw std::invalid_argument("directionalDerivativeWrtField: target FieldId appears with inconsistent SpaceSignature");
                }
            } else {
                wrt_field_sig = sig;
            }

            if (wrt_field_name.has_value()) {
                if (name != *wrt_field_name) {
                    throw std::invalid_argument("directionalDerivativeWrtField: target FieldId appears with inconsistent symbol name");
                }
            } else {
                wrt_field_name = name;
            }

            return true;
        };

        switch (node->type()) {
            // ---- Terminals (treated as constants) ----
            case FormExprType::Constant:
            case FormExprType::Coefficient:
            case FormExprType::ParameterSymbol:
            case FormExprType::ParameterRef:
            case FormExprType::BoundaryIntegralSymbol:
            case FormExprType::BoundaryIntegralRef:
            case FormExprType::AuxiliaryStateSymbol:
            case FormExprType::AuxiliaryStateRef:
            case FormExprType::MaterialStateOldRef:
            case FormExprType::MaterialStateWorkRef:
            case FormExprType::PreviousSolutionRef:
            case FormExprType::Coordinate:
            case FormExprType::ReferenceCoordinate:
            case FormExprType::MeshDisplacement:
            case FormExprType::MeshVelocity:
            case FormExprType::MeshAcceleration:
            case FormExprType::CurrentCoordinate:
            case FormExprType::PreviousCoordinate:
            case FormExprType::ReferencePhysicalCoordinate:
            case FormExprType::PreviousMeshVelocity:
            case FormExprType::PredictedMeshVelocity:
            case FormExprType::CurrentJacobian:
            case FormExprType::ReferenceJacobian:
            case FormExprType::CurrentJacobianDeterminant:
            case FormExprType::ReferenceJacobianDeterminant:
            case FormExprType::CurrentNormal:
            case FormExprType::CurrentMeanCurvature:
            case FormExprType::ReferenceNormal:
            case FormExprType::CurrentMeasure:
            case FormExprType::ReferenceMeasure:
            case FormExprType::SurfaceJacobian:
            case FormExprType::Normal:
            case FormExprType::BoundaryNormal:
            case FormExprType::Jacobian:
            case FormExprType::JacobianInverse:
            case FormExprType::JacobianDeterminant:
            case FormExprType::CellDiameter:
            case FormExprType::CellVolume:
            case FormExprType::FacetArea:
            case FormExprType::CellDomainId:
            case FormExprType::Time:
            case FormExprType::TimeStep:
            case FormExprType::EffectiveTimeStep:
            case FormExprType::BoundaryFunctionalSymbol:
            case FormExprType::Identity:
                out.primal = FormExpr(node);
                out.deriv = zeroOf(out.primal);
                break;

            case FormExprType::TestFunction:
                // A field-bound test function keeps its field: the derivative
                // of a multi-field form maps its rows by it.
                out.primal = node->fieldId().has_value()
                                 ? FormExpr(node)
                                 : FormExpr::testFunction(*node->spaceSignature(), node->toString());
                out.deriv = zeroOf(out.primal);
                break;

            case FormExprType::TrialFunction: {
                const auto* sig = node->spaceSignature();
                if (!sig) {
                    throw std::invalid_argument("directionalDerivativeWrtField: TrialFunction missing SpaceSignature");
                }
                out.primal = FormExpr::trialFunction(*sig, node->toString());
                out.deriv = zeroOf(out.primal);
                break;
            }

            case FormExprType::DiscreteField: {
                const auto fid = node->fieldId();
                const auto* sig = node->spaceSignature();
                if (!fid || !sig) {
                    throw std::invalid_argument("directionalDerivativeWrtField: DiscreteField missing metadata");
                }
                out.primal = FormExpr::discreteField(*fid, *sig, node->toString());
                out.deriv = discrete_fields_vary && isWrtField(*sig, *fid, node->toString())
                                ? direction
                                : zeroOf(out.primal);
                break;
            }

            case FormExprType::StateField: {
                const auto fid = node->fieldId();
                const auto* sig = node->spaceSignature();
                if (!fid || !sig) {
                    throw std::invalid_argument("directionalDerivativeWrtField: StateField missing metadata");
                }
                out.primal = FormExpr::stateField(*fid, *sig, node->toString());
                out.deriv = isWrtField(*sig, *fid, node->toString()) ? direction : zeroOf(out.primal);
                break;
            }

            // ---- Unary ops ----
            case FormExprType::Negate: {
                const auto a = diff1(0);
                out.primal = -a.primal;
                out.deriv = -a.deriv;
                break;
            }
            case FormExprType::Gradient: {
                const auto a = diff1(0);
                out.primal = a.primal.grad();
                out.deriv = a.deriv.grad();
                break;
            }
            case FormExprType::Divergence: {
                const auto a = diff1(0);
                out.primal = a.primal.div();
                out.deriv = a.deriv.div();
                break;
            }
            case FormExprType::Curl: {
                const auto a = diff1(0);
                out.primal = a.primal.curl();
                out.deriv = a.deriv.curl();
                break;
            }
            case FormExprType::Hessian: {
                const auto a = diff1(0);
                out.primal = a.primal.hessian();
                out.deriv = a.deriv.hessian();
                break;
            }
            case FormExprType::TimeDerivative: {
                const int order = node->timeDerivativeOrder().value_or(1);
                const auto a = diff1(0);
                out.primal = a.primal.dt(order);
                out.deriv = a.deriv.dt(order);
                break;
            }
            case FormExprType::RestrictMinus: {
                const auto a = diff1(0);
                out.primal = a.primal.minus();
                out.deriv = (a.deriv.isValid() && a.deriv.node() && isZeroLike(*a.deriv.node()))
                    ? a.deriv
                    : a.deriv.minus();
                break;
            }
            case FormExprType::RestrictPlus: {
                const auto a = diff1(0);
                out.primal = a.primal.plus();
                out.deriv = (a.deriv.isValid() && a.deriv.node() && isZeroLike(*a.deriv.node()))
                    ? a.deriv
                    : a.deriv.plus();
                break;
            }
            case FormExprType::Jump: {
                const auto a = diff1(0);
                out.primal = a.primal.jump();
                out.deriv = a.deriv.jump();
                break;
            }
            case FormExprType::Average: {
                const auto a = diff1(0);
                out.primal = a.primal.avg();
                out.deriv = a.deriv.avg();
                break;
            }
            case FormExprType::Pullback:
            case FormExprType::Pushforward: {
                const auto a = diff1(0);
                const auto from = node->fromConfiguration().value_or(GeometryConfiguration::Reference);
                const auto to = node->toConfiguration().value_or(GeometryConfiguration::Current);
                if (node->type() == FormExprType::Pullback) {
                    out.primal = pullback(a.primal, from, to);
                    out.deriv = pullback(a.deriv, from, to);
                } else {
                    out.primal = pushforward(a.primal, from, to);
                    out.deriv = pushforward(a.deriv, from, to);
                }
                break;
            }

            // ---- Constructors / indexing ----
            case FormExprType::AsVector: {
                std::vector<FormExpr> pv;
                std::vector<FormExpr> dv;
                pv.reserve(kids.size());
                dv.reserve(kids.size());
                for (const auto& k : kids) {
                    const auto r = self(self, k);
                    pv.push_back(r.primal);
                    dv.push_back(r.deriv);
                }
                out.primal = FormExpr::asVector(std::move(pv));
                out.deriv = FormExpr::asVector(std::move(dv));
                break;
            }
            case FormExprType::AsTensor: {
                const int rows = node->tensorRows().value_or(0);
                const int cols = node->tensorCols().value_or(0);
                if (rows <= 0 || cols <= 0) {
                    throw std::invalid_argument("directionalDerivativeWrtField: AsTensor missing (rows,cols)");
                }
                if (kids.size() != static_cast<std::size_t>(rows * cols)) {
                    throw std::invalid_argument("directionalDerivativeWrtField: AsTensor children size mismatch");
                }

                std::vector<std::vector<FormExpr>> prow(static_cast<std::size_t>(rows));
                std::vector<std::vector<FormExpr>> drow(static_cast<std::size_t>(rows));
                for (int r = 0; r < rows; ++r) {
                    prow[static_cast<std::size_t>(r)].reserve(static_cast<std::size_t>(cols));
                    drow[static_cast<std::size_t>(r)].reserve(static_cast<std::size_t>(cols));
                    for (int c = 0; c < cols; ++c) {
                        const auto idx = static_cast<std::size_t>(r * cols + c);
                        const auto rr = self(self, kids[idx]);
                        prow[static_cast<std::size_t>(r)].push_back(rr.primal);
                        drow[static_cast<std::size_t>(r)].push_back(rr.deriv);
                    }
                }
                out.primal = FormExpr::asTensor(std::move(prow));
                out.deriv = FormExpr::asTensor(std::move(drow));
                break;
            }
            case FormExprType::Component: {
                const int i = node->componentIndex0().value_or(0);
                const int j = node->componentIndex1().value_or(-1);
                const auto a = diff1(0);
                out.primal = a.primal.component(i, j);
                out.deriv = (a.deriv.isValid() && a.deriv.node() != nullptr && isZeroLike(*a.deriv.node()))
                    ? FormExpr::typedZero()
                    : a.deriv.component(i, j);
                break;
            }
            case FormExprType::IndexedAccess: {
                const int rank = node->indexRank().value_or(0);
                const auto ids_opt = node->indexIds();
                const auto ext_opt = node->indexExtents();
                const auto names_opt = node->indexNames();
                const auto vars_opt = node->indexVariances();
                if (rank <= 0 || !ids_opt || !ext_opt) {
                    throw std::invalid_argument("directionalDerivativeWrtField: IndexedAccess missing index metadata");
                }

                std::array<std::string, 4> names{};
                if (names_opt) {
                    for (std::size_t k = 0; k < names.size(); ++k) {
                        names[k] = std::string((*names_opt)[k]);
                    }
                }
                std::array<tensor::IndexVariance, 4> vars{};
                vars.fill(tensor::IndexVariance::None);
                if (vars_opt) {
                    vars = *vars_opt;
                }

                const auto ext = resolveAutoIndexExtents(*ext_opt, rank, auto_extent);
                const auto a = diff1(0);
                out.primal = FormExpr::indexedAccessRawWithMetadata(a.primal, rank, *ids_opt, ext, vars, names);
                out.deriv = (a.deriv.isValid() && a.deriv.node() != nullptr && isZeroLike(*a.deriv.node()))
                    ? FormExpr::constant(0.0)
                    : FormExpr::indexedAccessRawWithMetadata(a.deriv, rank, *ids_opt, ext, vars, names);
                break;
            }

            // ---- Algebra ----
            case FormExprType::Add: {
                const auto a = diff1(0);
                const auto b = diff1(1);
                out.primal = a.primal + b.primal;
                out.deriv = a.deriv + b.deriv;
                break;
            }
            case FormExprType::Subtract: {
                const auto a = diff1(0);
                const auto b = diff1(1);
                out.primal = a.primal - b.primal;
                out.deriv = a.deriv - b.deriv;
                break;
            }
            case FormExprType::Multiply: {
                const auto a = diff1(0);
                const auto b = diff1(1);
                out.primal = a.primal * b.primal;
                out.deriv = (a.deriv * b.primal) + (a.primal * b.deriv);
                break;
            }
            case FormExprType::Divide: {
                const auto a = diff1(0);
                const auto b = diff1(1);
                out.primal = a.primal / b.primal;
                out.deriv = (a.deriv / b.primal) - (a.primal * b.deriv / (b.primal * b.primal));
                break;
            }

            // InnerProduct is scalar; DoubleContraction is scalar. OuterProduct returns a tensor-like value.
            case FormExprType::InnerProduct: {
                const auto a = diff1(0);
                const auto b = diff1(1);
                out.primal = a.primal.inner(b.primal);
                out.deriv = a.deriv.inner(b.primal) + a.primal.inner(b.deriv);
                break;
            }
            case FormExprType::DoubleContraction: {
                const auto a = diff1(0);
                const auto b = diff1(1);
                out.primal = a.primal.doubleContraction(b.primal);
                out.deriv = a.deriv.doubleContraction(b.primal) + a.primal.doubleContraction(b.deriv);
                break;
            }
            case FormExprType::OuterProduct: {
                const auto a = diff1(0);
                const auto b = diff1(1);
                out.primal = a.primal.outer(b.primal);
                out.deriv = a.deriv.outer(b.primal) + a.primal.outer(b.deriv);
                break;
            }
            case FormExprType::CrossProduct: {
                const auto a = diff1(0);
                const auto b = diff1(1);
                out.primal = a.primal.cross(b.primal);
                out.deriv = a.deriv.cross(b.primal) + a.primal.cross(b.deriv);
                break;
            }

            case FormExprType::Power: {
                const auto a = diff1(0);
                const auto b = diff1(1);
                out.primal = pow(a.primal, b.primal);

                out.deriv = differentiatePower(a.primal, b.primal, a.deriv, b.deriv);
                break;
            }

            case FormExprType::MatrixPower: {
                const auto A = diff1(0);
                const auto p = diff1(1);
                out.primal = A.primal.matrixPow(p.primal);
                const auto dA_term =
                    (A.deriv.isValid() && A.deriv.node() != nullptr && isZeroLike(*A.deriv.node()))
                        ? zeroOf(out.primal)
                        : FormExpr::matrixPowDirectionalDerivative(A.primal, A.deriv, p.primal);
                const auto dp_term =
                    (p.deriv.isValid() && p.deriv.node() != nullptr && isZeroLike(*p.deriv.node()))
                        ? zeroOf(out.primal)
                        : (p.deriv * (out.primal * A.primal.matrixLog()));
                out.deriv = dA_term + dp_term;
                break;
            }

            case FormExprType::Minimum:
            case FormExprType::Maximum: {
                const auto a = diff1(0);
                const auto b = diff1(1);
                out.primal = (node->type() == FormExprType::Minimum) ? a.primal.min(b.primal) : a.primal.max(b.primal);
                // Use a 0/1 mask from comparisons for derivative (piecewise; ties pick +a).
                const auto pick_a = (node->type() == FormExprType::Minimum) ? le(a.primal, b.primal) : ge(a.primal, b.primal);
                out.deriv = pick_a * a.deriv + (FormExpr::constant(1.0) - pick_a) * b.deriv;
                break;
            }

            case FormExprType::SmoothMin:
            case FormExprType::SmoothMax: {
                const auto a = diff1(0);
                const auto b = diff1(1);
                const auto eps = diff1(2);
                const bool is_min = (node->type() == FormExprType::SmoothMin);
                out.primal = is_min ? a.primal.smoothMin(b.primal, eps.primal) : a.primal.smoothMax(b.primal, eps.primal);

                const auto d = a.primal - b.primal;
                const auto ad = d.smoothAbs(eps.primal); // sqrt(d^2 + eps^2)
                const auto num = d * (a.deriv - b.deriv) + eps.primal * eps.deriv;
                const auto dad = eq(ad, FormExpr::constant(0.0)).conditional(zeroOf(ad), num / ad);

                out.deriv = FormExpr::constant(0.5) * (a.deriv + b.deriv) +
                            (is_min ? FormExpr::constant(-0.5) : FormExpr::constant(0.5)) * dad;
                break;
            }

            // ---- Comparisons / control ----
            case FormExprType::Less:
            case FormExprType::LessEqual:
            case FormExprType::Greater:
            case FormExprType::GreaterEqual:
            case FormExprType::Equal:
            case FormExprType::NotEqual: {
                const auto a = diff1(0);
                const auto b = diff1(1);
                out.primal = FormExpr(node);
                out.deriv = zeroOf(out.primal);
                (void)a;
                (void)b;
                break;
            }
            case FormExprType::Conditional: {
                const auto a = diff1(0);
                const auto b = diff1(1);
                const auto c = diff1(2);
                out.primal = a.primal.conditional(b.primal, c.primal);
                out.deriv = a.primal.conditional(b.deriv, c.deriv);
                break;
            }

            // ---- Tensor ops / scalar functions ----
            case FormExprType::Transpose: {
                const auto a = diff1(0);
                out.primal = a.primal.transpose();
                out.deriv = a.deriv.transpose();
                break;
            }
            case FormExprType::Trace: {
                const auto a = diff1(0);
                out.primal = a.primal.trace();
                out.deriv = a.deriv.trace();
                break;
            }
            case FormExprType::Determinant: {
                const auto a = diff1(0);
                out.primal = a.primal.det();
                // Use cofactor-based rule to avoid singular-matrix inversion at det(A)=0:
                //   d(det(A)) = cofactor(A) : dA
                out.deriv = a.primal.cofactor().doubleContraction(a.deriv);
                break;
            }
            case FormExprType::Cofactor: {
                const auto a = diff1(0);
                const auto detA = det(a.primal);
                const auto invA = inv(a.primal);
                const auto invAT = transpose(invA);
                out.primal = a.primal.cofactor();
                const auto ddet = detA * inner(inv(transpose(a.primal)), a.deriv);
                const auto dinv = -(invA * a.deriv * invA);
                out.deriv = ddet * invAT + detA * transpose(dinv);
                break;
            }
            case FormExprType::Inverse: {
                const auto a = diff1(0);
                out.primal = inv(a.primal);
                out.deriv = -(out.primal * a.deriv * out.primal);
                break;
            }
            case FormExprType::Deviator: {
                const auto a = diff1(0);
                out.primal = a.primal.dev();
                out.deriv = a.deriv.dev();
                break;
            }
            case FormExprType::SymmetricPart: {
                const auto a = diff1(0);
                out.primal = a.primal.sym();
                out.deriv = a.deriv.sym();
                break;
            }
            case FormExprType::SkewPart: {
                const auto a = diff1(0);
                out.primal = a.primal.skew();
                out.deriv = a.deriv.skew();
                break;
            }
            case FormExprType::Norm: {
                const auto a = diff1(0);
                out.primal = a.primal.norm();
                const auto nrm = out.primal;
                const auto d = inner(a.primal, a.deriv) / nrm;
                out.deriv = eq(nrm, FormExpr::constant(0.0)).conditional(FormExpr::constant(0.0), d);
                break;
            }
            case FormExprType::Normalize: {
                const auto a = diff1(0);
                out.primal = a.primal.normalize();
                const auto nrm = a.primal.norm();
                const auto dnrm = inner(a.primal, a.deriv) / nrm;
                const auto d = (a.deriv * nrm - a.primal * dnrm) / (nrm * nrm);
                out.deriv = eq(nrm, FormExpr::constant(0.0)).conditional(zeroOf(out.primal), d);
                break;
            }
            case FormExprType::AbsoluteValue: {
                const auto a = diff1(0);
                out.primal = a.primal.abs();
                const auto is_nonneg = ge(a.primal, FormExpr::constant(0.0)); // 0/1
                const auto sign = (FormExpr::constant(2.0) * is_nonneg) - FormExpr::constant(1.0); // +1 or -1
                out.deriv = sign * a.deriv;
                break;
            }
            case FormExprType::Sign: {
                const auto a = diff1(0);
                out.primal = a.primal.sign();
                out.deriv = zeroOf(out.primal);
                break;
            }
            case FormExprType::Sqrt: {
                const auto a = diff1(0);
                out.primal = a.primal.sqrt();
                const auto is_zero = eq(out.primal, FormExpr::constant(0.0)); // 0/1
                const auto denom = (FormExpr::constant(2.0) * out.primal) + is_zero;
                out.deriv = a.deriv / denom;
                break;
            }
            case FormExprType::Exp: {
                const auto a = diff1(0);
                out.primal = a.primal.exp();
                out.deriv = out.primal * a.deriv;
                break;
            }
            case FormExprType::Log: {
                const auto a = diff1(0);
                out.primal = a.primal.log();
                out.deriv = a.deriv / a.primal;
                break;
            }

            case FormExprType::MatrixExponential: {
                const auto a = diff1(0);
                out.primal = a.primal.matrixExp();
                out.deriv = (a.deriv.isValid() && a.deriv.node() != nullptr && isZeroLike(*a.deriv.node()))
                                ? zeroOf(out.primal)
                                : FormExpr::matrixExpDirectionalDerivative(a.primal, a.deriv);
                break;
            }
            case FormExprType::MatrixLogarithm: {
                const auto a = diff1(0);
                out.primal = a.primal.matrixLog();
                out.deriv = (a.deriv.isValid() && a.deriv.node() != nullptr && isZeroLike(*a.deriv.node()))
                                ? zeroOf(out.primal)
                                : FormExpr::matrixLogDirectionalDerivative(a.primal, a.deriv);
                break;
            }
            case FormExprType::MatrixSqrt: {
                const auto a = diff1(0);
                out.primal = a.primal.matrixSqrt();
                out.deriv = (a.deriv.isValid() && a.deriv.node() != nullptr && isZeroLike(*a.deriv.node()))
                                ? zeroOf(out.primal)
                                : FormExpr::matrixSqrtDirectionalDerivative(a.primal, a.deriv);
                break;
            }

            case FormExprType::SmoothAbsoluteValue: {
                const auto x = diff1(0);
                const auto eps = diff1(1);
                out.primal = x.primal.smoothAbs(eps.primal);
                const auto denom = out.primal;
                const auto num = x.primal * x.deriv + eps.primal * eps.deriv;
                const auto d = num / denom;
                out.deriv = eq(denom, FormExpr::constant(0.0)).conditional(zeroOf(out.primal), d);
                break;
            }
            case FormExprType::SmoothSign:
            case FormExprType::SmoothHeaviside: {
                const auto x = diff1(0);
                const auto eps = diff1(1);
                const bool is_heaviside = (node->type() == FormExprType::SmoothHeaviside);
                out.primal = is_heaviside ? x.primal.smoothHeaviside(eps.primal) : x.primal.smoothSign(eps.primal);

                const auto denom = x.primal.smoothAbs(eps.primal);
                const auto num_denom = x.primal * x.deriv + eps.primal * eps.deriv;
                const auto d_denom = num_denom / denom;
                const auto d_sign = (x.deriv * denom - x.primal * d_denom) / (denom * denom);
                const auto d = is_heaviside ? (FormExpr::constant(0.5) * d_sign) : d_sign;
                out.deriv = eq(denom, FormExpr::constant(0.0)).conditional(zeroOf(out.primal), d);
                break;
            }

            case FormExprType::SymmetricEigenvalue: {
                const auto a = diff1(0);
                const int which = node->eigenIndex().value_or(0);
                out.primal = a.primal.symmetricEigenvalue(which);
                out.deriv = FormExpr::symmetricEigenvalueDirectionalDerivative(a.primal, a.deriv, which);
                break;
            }

            case FormExprType::Eigenvalue: {
                const auto a = diff1(0);
                const int which = node->eigenIndex().value_or(0);
                out.primal = a.primal.eigenvalue(which);
                out.deriv = FormExpr::symmetricEigenvalueDirectionalDerivative(a.primal, a.deriv, which);
                break;
            }

            case FormExprType::SymmetricEigenvector: {
                const auto a = diff1(0);
                const int which = node->eigenIndex().value_or(0);
                out.primal = a.primal.symmetricEigenvector(which);
                out.deriv = FormExpr::symmetricEigenvectorDirectionalDerivative(a.primal, a.deriv, which);
                break;
            }

            case FormExprType::SpectralDecomposition: {
                const auto a = diff1(0);
                out.primal = a.primal.spectralDecomposition();
                out.deriv = FormExpr::spectralDecompositionDirectionalDerivative(a.primal, a.deriv);
                break;
            }

            case FormExprType::SymmetricEigenvalueDirectionalDerivative: {
                const auto a = diff1(0);
                const auto b = diff1(1);
                const int which = node->eigenIndex().value_or(0);
                out.primal = FormExpr::symmetricEigenvalueDirectionalDerivative(a.primal, b.primal, which);
                out.deriv = FormExpr::symmetricEigenvalueDirectionalDerivativeWrtA(a.primal, b.primal, a.deriv, which) +
                            FormExpr::symmetricEigenvalueDirectionalDerivative(a.primal, b.deriv, which);
                break;
            }

            case FormExprType::SymmetricEigenvalueDirectionalDerivativeWrtA: {
                // Only needed for 3rd+ derivatives; treat as a (differentiable) n-ary op.
                const auto a = diff1(0);
                const auto b = diff1(1);
                const auto c = diff1(2);
                const int which = node->eigenIndex().value_or(0);
                out.primal = FormExpr::symmetricEigenvalueDirectionalDerivativeWrtA(a.primal, b.primal, c.primal, which);
                out.deriv = FormExpr::symmetricEigenvalueDirectionalDerivativeWrtA(a.primal, b.primal, c.deriv, which);
                break;
            }

            case FormExprType::HistoryWeightedSum:
            case FormExprType::HistoryConvolution: {
                std::vector<FormExpr> primal_weights;
                primal_weights.reserve(kids.size());
                for (const auto& k : kids) {
                    const auto w = self(self, k);
                    primal_weights.push_back(w.primal);
                }
                out.primal = (node->type() == FormExprType::HistoryWeightedSum)
                                 ? FormExpr::historyWeightedSum(primal_weights)
                                 : FormExpr::historyConvolution(primal_weights);

                FormExpr sum = zeroOf(out.primal);
                for (std::size_t kk = 0; kk < kids.size(); ++kk) {
                    const auto w = self(self, kids[kk]);
                    if (w.deriv.isValid() && w.deriv.node() && isZeroLike(*w.deriv.node())) {
                        continue;
                    }
                    const int steps_back = static_cast<int>(kk + 1u);
                    sum = sum + (w.deriv * FormExpr::previousSolution(steps_back));
                }
                out.deriv = sum;
                break;
            }

            // ---- Constitutive hooks (must be inlined) ----
            case FormExprType::Constitutive:
            case FormExprType::ConstitutiveOutput:
                throw std::invalid_argument("directionalDerivativeWrtField: Constitutive nodes must be inlined before differentiation");

            // ---- Measure wrappers ----
            case FormExprType::CellIntegral: {
                const auto a = diff1(0);
                out.primal = a.primal.dx();
                out.deriv = a.deriv.dx();
                break;
            }
            case FormExprType::BoundaryIntegral: {
                const auto a = diff1(0);
                out.primal = wrapExteriorIntegral(a.primal, *node);
                out.deriv = wrapExteriorIntegral(a.deriv, *node);
                break;
            }
            case FormExprType::InteriorFaceIntegral: {
                const auto a = diff1(0);
                out.primal = wrapInteriorFaceIntegral(a.primal, *node);
                out.deriv = wrapInteriorFaceIntegral(a.deriv, *node);
                break;
            }
            case FormExprType::InterfaceIntegral: {
                const auto a = diff1(0);
                out.primal = wrapExteriorIntegral(a.primal, *node);
                out.deriv = interfaceIntegralDerivative(
                    kids[0], *node, a.deriv,
                    [&](const std::shared_ptr<FormExprNode>& summand) { return self(self, summand).deriv; });
                break;
            }
            case FormExprType::CutVolumeIntegral: {
                const int marker = node->interfaceMarker().value_or(-1);
                const auto side = node->cutVolumeSide().value_or(CutVolumeSide::Negative);
                const auto a = diff1(0);
                out.primal = a.primal.dCutVolume(marker, side);
                out.deriv = a.deriv.dCutVolume(marker, side);
                break;
            }
            case FormExprType::GlobalIntegral: {
                // A point mass: no measure, so no geometry variation.
                const auto a = diff1(0);
                out.primal = a.primal.dGlobal();
                out.deriv = a.deriv.dGlobal();
                break;
            }

            default:
                throw std::invalid_argument("directionalDerivativeWrtField: unsupported FormExprType " +
                                            std::to_string(static_cast<std::uint16_t>(node->type())));
        }

        memo.emplace(node.get(), out);
        return out;
    };

    auto pair = diff(diff, expr.nodeShared());
    FormExpr deriv = simplify(pair.deriv);
    if (tensor::containsTensorCalculusNodes(deriv)) {
        deriv = tensor::postprocessTensorDerivative(deriv);
    }
    return deriv;
}

FormExpr directionalDerivativeWrtNormal(const FormExpr& expr,
                                        const FormExpr& direction)
{
    if (!expr.isValid() || expr.node() == nullptr) {
        throw std::invalid_argument("directionalDerivativeWrtNormal: invalid expression");
    }
    if (!direction.isValid() || direction.node() == nullptr) {
        throw std::invalid_argument("directionalDerivativeWrtNormal: invalid direction expression");
    }

    DiffTargetConfig cfg;
    cfg.kind = DiffWrtKind::NormalTerminal;
    cfg.trial_state_field = CURRENT_SOLUTION_FIELD_ID;
    cfg.normal_direction = direction;
    return differentiateResidualImpl(expr, cfg);
}

FormExpr directionalDerivativeWrtSpatialCoordinate(const FormExpr& expr,
                                                   const FormExpr& direction)
{
    return directionalDerivativeWrtCutSurfacePoint(expr, {.physical = direction});
}

FormExpr directionalDerivativeWrtCutSurfacePoint(const FormExpr &expr, const CutSurfacePointDirection &motion)
{
    if (!expr.isValid() || !motion.physical.isValid()) {
        throw std::invalid_argument(
            "directionalDerivativeWrtCutSurfacePoint requires an expression and direction");
    }
    DiffTargetConfig cfg;
    cfg.kind = DiffWrtKind::SpatialCoordinate;
    cfg.trial_state_field = CURRENT_SOLUTION_FIELD_ID;
    cfg.spatial_direction = motion.physical;
    cfg.cut_point = motion;
    return differentiateResidualImpl(expr, cfg);
}

FormExpr differentiateResidualHessianVector(const FormExpr& residual_form,
                                           const FormExpr& direction)
{
    const auto tangent = differentiateResidual(residual_form);
    return directionalDerivativeWrtField(tangent, CURRENT_SOLUTION_FIELD_ID, direction);
}

FormExpr differentiateResidualHessianVector(const FormExpr& residual_form,
                                           FieldId field,
                                           const FormExpr& direction,
                                           FieldId trial_state_field)
{
    const auto tangent = differentiateResidual(residual_form, field, trial_state_field);
    return directionalDerivativeWrtField(tangent, field, direction);
}

FormExpr differentiateWrtAuxiliaryOutput(const FormExpr& form,
                                         std::uint32_t output_slot)
{
    DiffTargetConfig cfg;
    cfg.kind = DiffWrtKind::AuxiliaryOutputSlot;
    cfg.aux_output_slot = output_slot;
    cfg.trial_state_field = CURRENT_SOLUTION_FIELD_ID;

    return simplify(differentiateResidualImpl(form, cfg));
}

} // namespace forms
} // namespace FE
} // namespace svmp

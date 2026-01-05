/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/Einsum.h"

#include "Core/FEException.h"

#include <unordered_map>
#include <stdexcept>

namespace svmp {
namespace FE {
namespace forms {

namespace {

struct IndexUse {
    int extent{0};
    int count{0};
};

struct IndexAssignment {
    std::unordered_map<int, int> id_to_value{};

    [[nodiscard]] int valueFor(int id) const
    {
        const auto it = id_to_value.find(id);
        if (it == id_to_value.end()) {
            throw std::invalid_argument("einsum: missing index assignment");
        }
        return it->second;
    }
};

bool isIntegralNode(FormExprType t) noexcept
{
    return t == FormExprType::CellIntegral || t == FormExprType::BoundaryIntegral || t == FormExprType::InteriorFaceIntegral;
}

FormExpr rebuildWithIndexSubstitution(const std::shared_ptr<FormExprNode>& node,
                                      const IndexAssignment& assignment);

FormExpr rebuildUnary(const std::shared_ptr<FormExprNode>& node,
                      const IndexAssignment& assignment)
{
    const auto kids = node->childrenShared();
    if (kids.size() != 1 || !kids[0]) {
        throw std::invalid_argument("einsum: unary node must have 1 child");
    }
    return rebuildWithIndexSubstitution(kids[0], assignment);
}

FormExpr rebuildBinary(const std::shared_ptr<FormExprNode>& node,
                       const IndexAssignment& assignment,
                       FormExprType op)
{
    const auto kids = node->childrenShared();
    if (kids.size() != 2 || !kids[0] || !kids[1]) {
        throw std::invalid_argument("einsum: binary node must have 2 children");
    }
    const auto a = rebuildWithIndexSubstitution(kids[0], assignment);
    const auto b = rebuildWithIndexSubstitution(kids[1], assignment);

    switch (op) {
        case FormExprType::Add: return a + b;
        case FormExprType::Subtract: return a - b;
        case FormExprType::Multiply: return a * b;
        case FormExprType::Divide: return a / b;
        case FormExprType::InnerProduct: return a.inner(b);
        case FormExprType::OuterProduct: return a.outer(b);
        case FormExprType::CrossProduct: return a.cross(b);
        case FormExprType::Power: return a.pow(b);
        case FormExprType::Minimum: return a.min(b);
        case FormExprType::Maximum: return a.max(b);
        case FormExprType::Less: return a.lt(b);
        case FormExprType::LessEqual: return a.le(b);
        case FormExprType::Greater: return a.gt(b);
        case FormExprType::GreaterEqual: return a.ge(b);
        case FormExprType::Equal: return a.eq(b);
        case FormExprType::NotEqual: return a.ne(b);
        default: break;
    }

    throw std::invalid_argument("einsum: unsupported binary op in rebuild");
}

FormExpr rebuildWithIndexSubstitution(const std::shared_ptr<FormExprNode>& node,
                                      const IndexAssignment& assignment)
{
    if (!node) return {};

    // Indexed access is the only node type that `einsum` actually replaces.
    if (node->type() == FormExprType::IndexedAccess) {
        const auto kids = node->childrenShared();
        if (kids.size() != 1 || !kids[0]) {
            throw std::invalid_argument("einsum: IndexedAccess must have exactly 1 operand");
        }

        const auto rank = node->indexRank().value_or(0);
        const auto ids_opt = node->indexIds();
        if (rank <= 0 || !ids_opt) {
            throw std::invalid_argument("einsum: IndexedAccess node missing index metadata");
        }

        const auto ids = *ids_opt;
        const auto base = rebuildWithIndexSubstitution(kids[0], assignment);

        if (rank == 1) {
            return base.component(assignment.valueFor(ids[0]));
        }
        if (rank == 2) {
            return base.component(assignment.valueFor(ids[0]), assignment.valueFor(ids[1]));
        }

        throw std::invalid_argument("einsum: only rank-1 and rank-2 indexed access is supported");
    }

    // Terminals: keep node as-is (shared).
    const auto t = node->type();
    switch (t) {
        case FormExprType::TestFunction:
        case FormExprType::TrialFunction:
        case FormExprType::Coefficient:
        case FormExprType::Constant:
        case FormExprType::Coordinate:
        case FormExprType::ReferenceCoordinate:
        case FormExprType::Identity:
        case FormExprType::Jacobian:
        case FormExprType::JacobianInverse:
        case FormExprType::JacobianDeterminant:
        case FormExprType::Normal:
        case FormExprType::CellDiameter:
        case FormExprType::CellVolume:
        case FormExprType::FacetArea:
            return FormExpr(node);
        default:
            break;
    }

    // Integral nodes: rebuild integrand and re-wrap.
    if (isIntegralNode(t)) {
        const auto integrand = rebuildUnary(node, assignment);
        if (t == FormExprType::CellIntegral) return integrand.dx();
        if (t == FormExprType::BoundaryIntegral) return integrand.ds(node->boundaryMarker().value_or(-1));
        return integrand.dS();
    }

    // Unary ops
    if (t == FormExprType::Negate) return -rebuildUnary(node, assignment);
    if (t == FormExprType::Gradient) return rebuildUnary(node, assignment).grad();
    if (t == FormExprType::Divergence) return rebuildUnary(node, assignment).div();
    if (t == FormExprType::Curl) return rebuildUnary(node, assignment).curl();
    if (t == FormExprType::Hessian) return rebuildUnary(node, assignment).hessian();
    if (t == FormExprType::RestrictMinus) return rebuildUnary(node, assignment).minus();
    if (t == FormExprType::RestrictPlus) return rebuildUnary(node, assignment).plus();
    if (t == FormExprType::Jump) return rebuildUnary(node, assignment).jump();
    if (t == FormExprType::Average) return rebuildUnary(node, assignment).avg();

    if (t == FormExprType::Transpose) return rebuildUnary(node, assignment).transpose();
    if (t == FormExprType::Trace) return rebuildUnary(node, assignment).trace();
    if (t == FormExprType::Determinant) return rebuildUnary(node, assignment).det();
    if (t == FormExprType::Inverse) return rebuildUnary(node, assignment).inv();
    if (t == FormExprType::Cofactor) return rebuildUnary(node, assignment).cofactor();
    if (t == FormExprType::Deviator) return rebuildUnary(node, assignment).dev();
    if (t == FormExprType::SymmetricPart) return rebuildUnary(node, assignment).sym();
    if (t == FormExprType::SkewPart) return rebuildUnary(node, assignment).skew();
    if (t == FormExprType::Norm) return rebuildUnary(node, assignment).norm();
    if (t == FormExprType::Normalize) return rebuildUnary(node, assignment).normalize();
    if (t == FormExprType::AbsoluteValue) return rebuildUnary(node, assignment).abs();
    if (t == FormExprType::Sign) return rebuildUnary(node, assignment).sign();
    if (t == FormExprType::Sqrt) return rebuildUnary(node, assignment).sqrt();
    if (t == FormExprType::Exp) return rebuildUnary(node, assignment).exp();
    if (t == FormExprType::Log) return rebuildUnary(node, assignment).log();

    if (t == FormExprType::TimeDerivative) {
        const int order = node->timeDerivativeOrder().value_or(1);
        return rebuildUnary(node, assignment).dt(order);
    }

    if (t == FormExprType::Component) {
        const int i = node->componentIndex0().value_or(0);
        const int j = node->componentIndex1().value_or(-1);
        return rebuildUnary(node, assignment).component(i, j);
    }

    if (t == FormExprType::ConstitutiveOutput) {
        const auto kids = node->childrenShared();
        if (kids.size() != 1u || !kids[0]) {
            throw std::invalid_argument("einsum: ConstitutiveOutput must have exactly 1 child");
        }
        const auto out_idx = node->constitutiveOutputIndex().value_or(0);
        if (out_idx < 0) {
            throw std::invalid_argument("einsum: ConstitutiveOutput has negative output index");
        }
        const auto call = rebuildWithIndexSubstitution(kids[0], assignment);
        return FormExpr::constitutiveOutput(call, static_cast<std::size_t>(out_idx));
    }

    if (t == FormExprType::Constitutive) {
        const auto kids = node->childrenShared();
        auto model = node->constitutiveModelShared();
        if (!model) {
            throw std::invalid_argument("einsum: Constitutive node missing model");
        }
        if (kids.empty()) {
            throw std::invalid_argument("einsum: Constitutive must have at least 1 child");
        }
        std::vector<FormExpr> inputs;
        inputs.reserve(kids.size());
        for (const auto& kid : kids) {
            if (!kid) {
                throw std::invalid_argument("einsum: Constitutive has null child");
            }
            inputs.push_back(rebuildWithIndexSubstitution(kid, assignment));
        }
        if (inputs.size() == 1u) {
            return FormExpr::constitutive(std::move(model), inputs[0]);
        }
        return FormExpr::constitutive(std::move(model), std::move(inputs));
    }

    // Conditional
    if (t == FormExprType::Conditional) {
        const auto kids = node->childrenShared();
        if (kids.size() != 3 || !kids[0] || !kids[1] || !kids[2]) {
            throw std::invalid_argument("einsum: conditional must have 3 children");
        }
        const auto c = rebuildWithIndexSubstitution(kids[0], assignment);
        const auto a = rebuildWithIndexSubstitution(kids[1], assignment);
        const auto b = rebuildWithIndexSubstitution(kids[2], assignment);
        return c.conditional(a, b);
    }

    // Binary ops
    switch (t) {
        case FormExprType::Add:
        case FormExprType::Subtract:
        case FormExprType::Multiply:
        case FormExprType::Divide:
        case FormExprType::InnerProduct:
        case FormExprType::OuterProduct:
        case FormExprType::CrossProduct:
        case FormExprType::Power:
        case FormExprType::Minimum:
        case FormExprType::Maximum:
        case FormExprType::Less:
        case FormExprType::LessEqual:
        case FormExprType::Greater:
        case FormExprType::GreaterEqual:
        case FormExprType::Equal:
        case FormExprType::NotEqual:
            return rebuildBinary(node, assignment, t);
        default:
            break;
    }

    throw std::invalid_argument("einsum: unsupported node in rebuild");
}

void collectIndexUses(const FormExprNode& node,
                      std::unordered_map<int, IndexUse>& uses)
{
    if (node.type() == FormExprType::IndexedAccess) {
        const int rank = node.indexRank().value_or(0);
        const auto ids_opt = node.indexIds();
        const auto ext_opt = node.indexExtents();
        if (rank <= 0 || !ids_opt || !ext_opt) {
            throw std::invalid_argument("einsum: IndexedAccess node missing index metadata");
        }
        const auto ids = *ids_opt;
        const auto ext = *ext_opt;

        for (int k = 0; k < rank; ++k) {
            const int id = ids[static_cast<std::size_t>(k)];
            const int e = ext[static_cast<std::size_t>(k)];
            auto& u = uses[id];
            if (u.extent == 0) {
                u.extent = e;
            } else if (u.extent != e) {
                throw std::invalid_argument("einsum: inconsistent index extents for a repeated index");
            }
            u.count += 1;
        }
    }

    for (const auto& child : node.childrenShared()) {
        if (child) collectIndexUses(*child, uses);
    }
}

} // namespace

FormExpr einsum(const FormExpr& expr)
{
    if (!expr.isValid()) return {};

    std::unordered_map<int, IndexUse> uses;
    collectIndexUses(*expr.node(), uses);
    if (uses.empty()) {
        return expr; // nothing to lower
    }

    std::vector<int> bound_ids;
    std::vector<int> bound_extents;
    bound_ids.reserve(uses.size());
    bound_extents.reserve(uses.size());

    for (const auto& [id, u] : uses) {
        if (u.count == 2) {
            bound_ids.push_back(id);
            bound_extents.push_back(u.extent);
            continue;
        }
        if (u.count == 1) {
            throw std::invalid_argument("einsum: free indices are not supported (expression must be fully contracted)");
        }
        throw std::invalid_argument("einsum: an index must appear exactly twice (bound) or not at all");
    }

    FormExpr sum{};
    bool first = true;

    IndexAssignment assignment;

    const auto recurse = [&](const auto& self, std::size_t k) -> void {
        if (k == bound_ids.size()) {
            const auto term = rebuildWithIndexSubstitution(expr.nodeShared(), assignment);
            if (first) {
                sum = term;
                first = false;
            } else {
                sum = sum + term;
            }
            return;
        }

        const int id = bound_ids[k];
        const int extent = bound_extents[k];
        if (extent <= 0) {
            throw std::invalid_argument("einsum: invalid index extent");
        }
        for (int v = 0; v < extent; ++v) {
            assignment.id_to_value[id] = v;
            self(self, k + 1);
        }
        assignment.id_to_value.erase(id);
    };

    recurse(recurse, 0);
    return sum;
}

} // namespace forms
} // namespace FE
} // namespace svmp

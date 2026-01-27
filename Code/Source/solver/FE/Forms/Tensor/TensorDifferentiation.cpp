/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/Tensor/TensorDifferentiation.h"

#include "Forms/SymbolicDifferentiation.h"
#include "Forms/Tensor/TensorSimplify.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace svmp {
namespace FE {
namespace forms {
namespace tensor {

namespace {

[[nodiscard]] bool containsIndexedAccessNode(const FormExprNode& node) noexcept
{
    if (node.type() == FormExprType::IndexedAccess) {
        return true;
    }
    for (const auto& child : node.childrenShared()) {
        if (child && containsIndexedAccessNode(*child)) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] std::unordered_map<int, int> collectIndexIdCounts(const FormExpr& expr)
{
    std::unordered_map<int, int> out;
    if (!expr.isValid() || expr.node() == nullptr) {
        return out;
    }
    const auto visit = [&](const auto& self, const FormExprNode& n) -> void {
        if (n.type() == FormExprType::IndexedAccess) {
            const int rank = n.indexRank().value_or(0);
            const auto ids_opt = n.indexIds();
            if (rank > 0 && ids_opt) {
                const auto ids = *ids_opt;
                for (int k = 0; k < rank; ++k) {
                    const int id = ids[static_cast<std::size_t>(k)];
                    if (id >= 0) {
                        out[id] += 1;
                    }
                }
            }
        }
        for (const auto& child : n.childrenShared()) {
            if (child) self(self, *child);
        }
    };
    visit(visit, *expr.node());
    return out;
}

[[nodiscard]] int maxIndexId(const FormExpr& expr) noexcept
{
    if (!expr.isValid() || expr.node() == nullptr) {
        return -1;
    }
    int max_id = -1;
    const auto visit = [&](const auto& self, const FormExprNode& n) -> void {
        if (n.type() == FormExprType::IndexedAccess) {
            const int rank = n.indexRank().value_or(0);
            const auto ids_opt = n.indexIds();
            if (rank > 0 && ids_opt) {
                const auto ids = *ids_opt;
                for (int k = 0; k < rank; ++k) {
                    max_id = std::max(max_id, ids[static_cast<std::size_t>(k)]);
                }
            }
        }
        for (const auto& child : n.childrenShared()) {
            if (child) self(self, *child);
        }
    };
    visit(visit, *expr.node());
    return max_id;
}

[[nodiscard]] FormExpr renameIndexIdPreservingMetadata(const FormExpr& expr, int old_id, int new_id)
{
    if (!expr.isValid() || expr.node() == nullptr || old_id == new_id) {
        return expr;
    }

    FormExpr::NodeTransform transform;
    transform = [&](const FormExprNode& node) -> std::optional<FormExpr> {
        if (node.type() != FormExprType::IndexedAccess) {
            return std::nullopt;
        }

        const int rank = node.indexRank().value_or(0);
        const auto ids_opt = node.indexIds();
        const auto ext_opt = node.indexExtents();
        if (rank <= 0 || !ids_opt || !ext_opt) {
            return std::nullopt;
        }

        const auto kids = node.childrenShared();
        if (kids.size() != 1u || !kids[0]) {
            return std::nullopt;
        }

        auto base = FormExpr(kids[0]).transformNodes(transform);

        auto ids = *ids_opt;
        bool changed = false;
        for (int k = 0; k < rank; ++k) {
            const auto idx = static_cast<std::size_t>(k);
            if (ids[idx] == old_id) {
                ids[idx] = new_id;
                changed = true;
            }
        }

        const bool child_changed = (base.nodeShared() != kids[0]);
        if (!changed && !child_changed) {
            return std::nullopt;
        }

        std::array<tensor::IndexVariance, 4> vars{};
        vars.fill(tensor::IndexVariance::None);
        if (const auto vars_opt = node.indexVariances()) {
            vars = *vars_opt;
        }

        std::array<std::string, 4> names{};
        if (const auto names_opt = node.indexNames()) {
            for (std::size_t k = 0; k < names.size(); ++k) {
                names[k] = std::string((*names_opt)[k]);
            }
        }

        return FormExpr::indexedAccessRawWithMetadata(std::move(base), rank, std::move(ids), *ext_opt, vars, std::move(names));
    };

    return expr.transformNodes(transform);
}

[[nodiscard]] FormExpr disambiguateIndicesInSums(const FormExpr& expr)
{
    if (!expr.isValid() || expr.node() == nullptr) {
        return expr;
    }
    if (!containsIndexedAccessNode(*expr.node())) {
        return expr;
    }

    int next_id = maxIndexId(expr) + 1;

    FormExpr::NodeTransform transform;
    transform = [&](const FormExprNode& node) -> std::optional<FormExpr> {
        if (node.type() != FormExprType::Add && node.type() != FormExprType::Subtract) {
            return std::nullopt;
        }

        const auto kids = node.childrenShared();
        if (kids.size() != 2u || !kids[0] || !kids[1]) {
            return std::nullopt;
        }

        auto left = FormExpr(kids[0]).transformNodes(transform);
        auto right = FormExpr(kids[1]).transformNodes(transform);

        const auto left_counts = collectIndexIdCounts(left);
        const auto right_counts = collectIndexIdCounts(right);

        // Free indices (count==1) must remain consistent across additive branches,
        // because they may represent tensor outputs or indices that are bound
        // outside the sum. Only rename colliding *dummy* indices.
        std::unordered_set<int> free_left;
        std::unordered_set<int> free_right;
        free_left.reserve(left_counts.size());
        free_right.reserve(right_counts.size());
        for (const auto& kv : left_counts) {
            if (kv.second == 1) free_left.insert(kv.first);
        }
        for (const auto& kv : right_counts) {
            if (kv.second == 1) free_right.insert(kv.first);
        }

        std::vector<int> overlap;
        overlap.reserve(std::min(left_counts.size(), right_counts.size()));
        for (const auto& kv : right_counts) {
            const int id = kv.first;
            if (left_counts.count(id) == 0u) continue;
            if (free_left.count(id) > 0u) continue;
            if (free_right.count(id) > 0u) continue;
            overlap.push_back(id);
        }
        std::sort(overlap.begin(), overlap.end());
        for (const int id : overlap) {
            const int fresh = next_id++;
            right = renameIndexIdPreservingMetadata(right, id, fresh);
        }

        if (node.type() == FormExprType::Add) {
            return left + right;
        }
        return left - right;
    };

    return expr.transformNodes(transform);
}

[[nodiscard]] FormExpr distributeBilinearOverSums(const FormExpr& expr)
{
    if (!expr.isValid() || expr.node() == nullptr) {
        return expr;
    }
    if (!containsIndexedAccessNode(*expr.node())) {
        return expr;
    }

    FormExpr::NodeTransform transform;
    transform = [&](const FormExprNode& node) -> std::optional<FormExpr> {
        const auto type = node.type();
        if (type != FormExprType::Multiply && type != FormExprType::DoubleContraction) {
            return std::nullopt;
        }

        const auto kids = node.childrenShared();
        if (kids.size() != 2u || !kids[0] || !kids[1]) {
            return std::nullopt;
        }

        auto left = FormExpr(kids[0]).transformNodes(transform);
        auto right = FormExpr(kids[1]).transformNodes(transform);

        auto make_bilinear = [&](const FormExpr& a, const FormExpr& b) -> FormExpr {
            if (type == FormExprType::Multiply) {
                return a * b;
            }
            return a.doubleContraction(b);
        };

        const auto distribute_left = [&](FormExprType sum_type) -> FormExpr {
            const auto sum_kids = left.node()->childrenShared();
            const auto a = FormExpr(sum_kids[0]);
            const auto b = FormExpr(sum_kids[1]);
            const auto t0 = make_bilinear(a, right).transformNodes(transform);
            const auto t1 = make_bilinear(b, right).transformNodes(transform);
            return (sum_type == FormExprType::Add) ? (t0 + t1) : (t0 - t1);
        };

        const auto distribute_right = [&](FormExprType sum_type) -> FormExpr {
            const auto sum_kids = right.node()->childrenShared();
            const auto a = FormExpr(sum_kids[0]);
            const auto b = FormExpr(sum_kids[1]);
            const auto t0 = make_bilinear(left, a).transformNodes(transform);
            const auto t1 = make_bilinear(left, b).transformNodes(transform);
            return (sum_type == FormExprType::Add) ? (t0 + t1) : (t0 - t1);
        };

        if (left.node() && (left.node()->type() == FormExprType::Add || left.node()->type() == FormExprType::Subtract)) {
            return distribute_left(left.node()->type());
        }
        if (right.node() && (right.node()->type() == FormExprType::Add || right.node()->type() == FormExprType::Subtract)) {
            return distribute_right(right.node()->type());
        }

        return make_bilinear(left, right);
    };

    return expr.transformNodes(transform);
}

} // namespace

TensorDiffResult checkTensorDifferentiability(const FormExpr& expr)
{
    const auto r = checkSymbolicDifferentiability(expr);
    TensorDiffResult out;
    out.ok = r.ok;
    if (r.first_issue) {
        out.first_issue = TensorDiffIssue{
            .type = r.first_issue->type,
            .message = r.first_issue->message,
            .subexpr = r.first_issue->subexpr,
        };
    }
    return out;
}

FormExpr differentiateTensorResidual(const FormExpr& residual_form, const TensorDiffContext& ctx)
{
    TensorDiffOptions opts;

    if (ctx.wrt_terminal.has_value()) {
        auto d = differentiateResidual(residual_form, *ctx.wrt_terminal, ctx.trial_state_field);
        return postprocessTensorDerivative(d, opts);
    }
    if (ctx.wrt_field.has_value()) {
        auto d = differentiateResidual(residual_form, *ctx.wrt_field, ctx.trial_state_field);
        return postprocessTensorDerivative(d, opts);
    }
    auto d = differentiateResidual(residual_form);
    return postprocessTensorDerivative(d, opts);
}

FormExpr differentiateTensorResidualHessianVector(const FormExpr& residual_form,
                                                  const FormExpr& direction,
                                                  const TensorDiffContext& ctx)
{
    const auto tangent = differentiateTensorResidual(residual_form, ctx);

    FieldId target = INVALID_FIELD_ID;
    if (ctx.wrt_field.has_value()) {
        target = *ctx.wrt_field;
    } else if (ctx.wrt_terminal.has_value()) {
        const auto& t = *ctx.wrt_terminal;
        if (!t.isValid() || t.node() == nullptr) {
            throw std::invalid_argument("differentiateTensorResidualHessianVector: invalid wrt_terminal");
        }
        switch (t.node()->type()) {
            case FormExprType::TrialFunction:
                target = ctx.trial_state_field;
                break;
            case FormExprType::DiscreteField:
            case FormExprType::StateField: {
                const auto fid = t.node()->fieldId();
                if (!fid) {
                    throw std::invalid_argument("differentiateTensorResidualHessianVector: wrt field terminal missing FieldId");
                }
                target = *fid;
                break;
            }
            default:
                throw std::invalid_argument("differentiateTensorResidualHessianVector: wrt_terminal must be TrialFunction/StateField/DiscreteField");
        }
    }

    return directionalDerivativeWrtField(tangent, target, direction);
}

FormExpr postprocessTensorDerivative(const FormExpr& expr, const TensorDiffOptions& options)
{
    if (!expr.isValid() || expr.node() == nullptr) {
        return {};
    }

    if (!containsIndexedAccessNode(*expr.node())) {
        return expr;
    }

    FormExpr out = expr;
    // Keep tensor/index expressions compatible with the current einsum/analyzeContractions
    // contract by expanding bilinear operations over sums before index renaming.
    out = distributeBilinearOverSums(out);
    if (options.disambiguate_indices_in_sums) {
        out = disambiguateIndicesInSums(out);
    }

    if (options.simplify_tensor_expr) {
        TensorSimplifyOptions simp;
        simp.max_passes = options.simplify_max_passes;
        simp.canonicalize_terms = options.simplify_canonicalize_terms;
        const auto r = simplifyTensorExpr(out, simp);
        if (r.ok) {
            out = r.expr;
        }
    }

    return out;
}

bool containsTensorCalculusNodes(const FormExpr& expr)
{
    if (!expr.isValid() || expr.node() == nullptr) {
        return false;
    }
    return containsIndexedAccessNode(*expr.node());
}

} // namespace tensor
} // namespace forms
} // namespace FE
} // namespace svmp

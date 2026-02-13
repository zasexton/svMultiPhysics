/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/Tensor/TensorCanonicalize.h"

#include "Forms/IndexExtent.h"
#include "Forms/Tensor/TensorIndex.h"

#include <algorithm>
#include <optional>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace tensor {

CanonicalIndexRenaming computeCanonicalIndexRenaming(const FormExpr& expr)
{
    CanonicalIndexRenaming out;

    if (!expr.isValid() || expr.node() == nullptr) {
        return out;
    }

    out.old_to_canonical.reserve(16);
    out.canonical_to_old.reserve(16);

    const auto getOrAssign = [&](int id) -> int {
        if (auto it = out.old_to_canonical.find(id); it != out.old_to_canonical.end()) {
            return it->second;
        }
        const int canonical = static_cast<int>(out.canonical_to_old.size());
        out.old_to_canonical.emplace(id, canonical);
        out.canonical_to_old.push_back(id);
        return canonical;
    };

    const auto visit = [&](const auto& self, const FormExprNode& n) -> void {
        if (n.type() == FormExprType::IndexedAccess) {
            const int rank = n.indexRank().value_or(0);
            const auto ids_opt = n.indexIds();
            if (rank > 0 && ids_opt) {
                const auto ids = *ids_opt;
                for (int k = 0; k < rank; ++k) {
                    const int id = ids[static_cast<std::size_t>(k)];
                    if (id >= 0) {
                        (void)getOrAssign(id);
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

std::string toCanonicalString(const FormExpr& expr)
{
    if (!expr.isValid() || expr.node() == nullptr) {
        return "<empty>";
    }

    const int auto_extent = forms::inferAutoIndexExtent(expr);

    const auto renaming = computeCanonicalIndexRenaming(expr);
    if (renaming.old_to_canonical.empty()) {
        return expr.toString();
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
        for (int k = 0; k < rank; ++k) {
            const auto idx = static_cast<std::size_t>(k);
            const int id = ids[idx];
            if (auto it = renaming.old_to_canonical.find(id); it != renaming.old_to_canonical.end()) {
                ids[idx] = it->second;
            }
        }

        std::array<tensor::IndexVariance, 4> vars{};
        vars.fill(tensor::IndexVariance::None);
        if (const auto vars_opt = node.indexVariances()) {
            vars = *vars_opt;
        }

        std::array<std::string, 4> names{};
        const auto ext = forms::resolveAutoIndexExtents(*ext_opt, rank, auto_extent);
        return FormExpr::indexedAccessRawWithMetadata(std::move(base), rank, std::move(ids), ext, std::move(vars), std::move(names));
    };

    const auto canonical = expr.transformNodes(transform);
    return canonical.toString();
}

namespace {

[[nodiscard]] bool isScalarValue(const FormExprNode& node)
{
    // Conservative scalar inference (sufficient for safe commutative reordering).
    switch (node.type()) {
        case FormExprType::Constant:
        case FormExprType::ParameterSymbol:
        case FormExprType::ParameterRef:
        case FormExprType::BoundaryIntegralRef:
        case FormExprType::AuxiliaryStateRef:
        case FormExprType::MaterialStateOldRef:
        case FormExprType::MaterialStateWorkRef:
        case FormExprType::PreviousSolutionRef:
        case FormExprType::Time:
        case FormExprType::TimeStep:
        case FormExprType::EffectiveTimeStep:
        case FormExprType::CellDiameter:
        case FormExprType::CellVolume:
        case FormExprType::FacetArea:
        case FormExprType::CellDomainId:
        case FormExprType::JacobianDeterminant:
            return true;

        case FormExprType::Component:
        case FormExprType::IndexedAccess:
        case FormExprType::Trace:
        case FormExprType::Determinant:
        case FormExprType::Norm:
        case FormExprType::AbsoluteValue:
        case FormExprType::Sign:
        case FormExprType::Sqrt:
        case FormExprType::Exp:
        case FormExprType::Log:
            return true;

        case FormExprType::InnerProduct:
        case FormExprType::DoubleContraction:
        case FormExprType::Less:
        case FormExprType::LessEqual:
        case FormExprType::Greater:
        case FormExprType::GreaterEqual:
        case FormExprType::Equal:
        case FormExprType::NotEqual:
            return true;

        default:
            break;
    }

    // Terminals with a known scalar value dimension.
    if (node.type() == FormExprType::TestFunction ||
        node.type() == FormExprType::TrialFunction ||
        node.type() == FormExprType::DiscreteField ||
        node.type() == FormExprType::StateField) {
        if (const auto* sig = node.spaceSignature(); sig != nullptr) {
            return sig->value_dimension == 1;
        }
    }

    if (node.type() == FormExprType::Coefficient) {
        return node.scalarCoefficient() != nullptr || node.timeScalarCoefficient() != nullptr;
    }

    // Recurse on simple scalar algebra.
    if (node.type() == FormExprType::Negate ||
        node.type() == FormExprType::TimeDerivative ||
        node.type() == FormExprType::RestrictMinus ||
        node.type() == FormExprType::RestrictPlus ||
        node.type() == FormExprType::Jump ||
        node.type() == FormExprType::Average) {
        const auto kids = node.childrenShared();
        return kids.size() == 1u && kids[0] && isScalarValue(*kids[0]);
    }

    if (node.type() == FormExprType::Add ||
        node.type() == FormExprType::Subtract ||
        node.type() == FormExprType::Divide ||
        node.type() == FormExprType::Power ||
        node.type() == FormExprType::Minimum ||
        node.type() == FormExprType::Maximum) {
        const auto kids = node.childrenShared();
        return kids.size() == 2u && kids[0] && kids[1] && isScalarValue(*kids[0]) && isScalarValue(*kids[1]);
    }

    if (node.type() == FormExprType::Multiply) {
        const auto kids = node.childrenShared();
        return kids.size() == 2u && kids[0] && kids[1] && isScalarValue(*kids[0]) && isScalarValue(*kids[1]);
    }

    return false;
}

[[nodiscard]] bool isCommutativeMultiply(const FormExprNode& node)
{
    if (node.type() != FormExprType::Multiply) return false;
    const auto kids = node.childrenShared();
    if (kids.size() != 2u || !kids[0] || !kids[1]) return false;
    return isScalarValue(*kids[0]) || isScalarValue(*kids[1]);
}

void flattenAdd(const FormExpr& expr, std::vector<FormExpr>& out)
{
    if (!expr.isValid() || expr.node() == nullptr) return;
    const auto& node = *expr.node();
    if (node.type() != FormExprType::Add) {
        out.push_back(expr);
        return;
    }
    const auto kids = node.childrenShared();
    if (kids.size() != 2u || !kids[0] || !kids[1]) {
        out.push_back(expr);
        return;
    }
    flattenAdd(FormExpr(kids[0]), out);
    flattenAdd(FormExpr(kids[1]), out);
}

void flattenMultiplyComm(const FormExpr& expr, std::vector<FormExpr>& out)
{
    if (!expr.isValid() || expr.node() == nullptr) return;
    const auto& node = *expr.node();
    if (node.type() != FormExprType::Multiply || !isCommutativeMultiply(node)) {
        out.push_back(expr);
        return;
    }
    const auto kids = node.childrenShared();
    if (kids.size() != 2u || !kids[0] || !kids[1]) {
        out.push_back(expr);
        return;
    }
    flattenMultiplyComm(FormExpr(kids[0]), out);
    flattenMultiplyComm(FormExpr(kids[1]), out);
}

[[nodiscard]] FormExpr rebuildAdd(const std::vector<FormExpr>& terms)
{
    if (terms.empty()) {
        return FormExpr::constant(0.0);
    }
    FormExpr out = terms[0];
    for (std::size_t i = 1; i < terms.size(); ++i) {
        out = out + terms[i];
    }
    return out;
}

[[nodiscard]] FormExpr rebuildMultiply(const std::vector<FormExpr>& factors)
{
    if (factors.empty()) {
        return FormExpr::constant(1.0);
    }
    FormExpr out = factors[0];
    for (std::size_t i = 1; i < factors.size(); ++i) {
        out = out * factors[i];
    }
    return out;
}

} // namespace

FormExpr canonicalizeTermOrder(const FormExpr& expr)
{
    if (!expr.isValid() || expr.node() == nullptr) {
        return expr;
    }

    FormExpr::NodeTransform transform;
    transform = [&](const FormExprNode& node) -> std::optional<FormExpr> {
        switch (node.type()) {
            case FormExprType::Add: {
                const auto kids = node.childrenShared();
                if (kids.size() != 2u || !kids[0] || !kids[1]) {
                    return std::nullopt;
                }

                const auto a = FormExpr(kids[0]).transformNodes(transform);
                const auto b = FormExpr(kids[1]).transformNodes(transform);

                std::vector<FormExpr> terms;
                terms.reserve(4);
                flattenAdd(a, terms);
                flattenAdd(b, terms);

                std::vector<std::pair<std::string, FormExpr>> keyed;
                keyed.reserve(terms.size());
                for (auto& t : terms) {
                    keyed.emplace_back(toCanonicalString(t), std::move(t));
                }
                std::sort(keyed.begin(), keyed.end(),
                          [](const auto& x, const auto& y) { return x.first < y.first; });

                std::vector<FormExpr> ordered;
                ordered.reserve(keyed.size());
                for (auto& kv : keyed) {
                    ordered.push_back(std::move(kv.second));
                }
                return rebuildAdd(ordered);
            }

            case FormExprType::Multiply: {
                const auto kids = node.childrenShared();
                if (kids.size() != 2u || !kids[0] || !kids[1]) {
                    return std::nullopt;
                }

                const auto a = FormExpr(kids[0]).transformNodes(transform);
                const auto b = FormExpr(kids[1]).transformNodes(transform);

                const bool a_scalar = a.node() && isScalarValue(*a.node());
                const bool b_scalar = b.node() && isScalarValue(*b.node());
                if (!a_scalar && !b_scalar) {
                    // Potentially non-commutative multiply; keep order.
                    return a * b;
                }

                std::vector<FormExpr> factors;
                factors.reserve(4);
                flattenMultiplyComm(a, factors);
                flattenMultiplyComm(b, factors);

                std::vector<FormExpr> scalar_factors;
                std::vector<FormExpr> nonscalar_factors;
                scalar_factors.reserve(factors.size());
                nonscalar_factors.reserve(factors.size());

                for (auto& f : factors) {
                    if (f.node() && isScalarValue(*f.node())) {
                        scalar_factors.push_back(std::move(f));
                    } else {
                        nonscalar_factors.push_back(std::move(f));
                    }
                }

                std::vector<std::pair<std::string, FormExpr>> keyed;
                keyed.reserve(scalar_factors.size());
                for (auto& s : scalar_factors) {
                    keyed.emplace_back(toCanonicalString(s), std::move(s));
                }
                std::sort(keyed.begin(), keyed.end(),
                          [](const auto& x, const auto& y) { return x.first < y.first; });

                std::vector<FormExpr> ordered;
                ordered.reserve(keyed.size() + nonscalar_factors.size());
                for (auto& kv : keyed) {
                    ordered.push_back(std::move(kv.second));
                }
                for (auto& ns : nonscalar_factors) {
                    ordered.push_back(std::move(ns));
                }
                return rebuildMultiply(ordered);
            }

            default:
                return std::nullopt;
        }
    };

    return expr.transformNodes(transform);
}

} // namespace tensor
} // namespace forms
} // namespace FE
} // namespace svmp

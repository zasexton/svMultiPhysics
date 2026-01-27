/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/Tensor/TensorSimplify.h"

#include "Forms/Tensor/TensorCanonicalize.h"
#include "Forms/Tensor/TensorContraction.h"
#include "Forms/Tensor/TensorIndex.h"

#include <algorithm>
#include <array>
#include <optional>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace tensor {

namespace {

struct DeltaInfo {
    int id0{-1};
    int id1{-1};
    int extent0{0};
    int extent1{0};
    IndexVariance var0{IndexVariance::None};
    IndexVariance var1{IndexVariance::None};
    std::string name0{};
    std::string name1{};
};

[[nodiscard]] bool isScalarLike(const FormExprNode& node)
{
    // Conservative scalar inference sufficient to guard commutativity-based rewrites.
    switch (node.type()) {
        case FormExprType::Constant:
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
        if (node.scalarCoefficient() != nullptr || node.timeScalarCoefficient() != nullptr) {
            return true;
        }
        return false;
    }

    // Recurse on simple algebra.
    if (node.type() == FormExprType::Negate ||
        node.type() == FormExprType::TimeDerivative ||
        node.type() == FormExprType::RestrictMinus ||
        node.type() == FormExprType::RestrictPlus ||
        node.type() == FormExprType::Jump ||
        node.type() == FormExprType::Average) {
        const auto kids = node.childrenShared();
        return kids.size() == 1u && kids[0] && isScalarLike(*kids[0]);
    }

    if (node.type() == FormExprType::Add ||
        node.type() == FormExprType::Subtract ||
        node.type() == FormExprType::Divide ||
        node.type() == FormExprType::Power ||
        node.type() == FormExprType::Minimum ||
        node.type() == FormExprType::Maximum) {
        const auto kids = node.childrenShared();
        return kids.size() == 2u && kids[0] && kids[1] && isScalarLike(*kids[0]) && isScalarLike(*kids[1]);
    }

    if (node.type() == FormExprType::Multiply) {
        const auto kids = node.childrenShared();
        if (kids.size() != 2u || !kids[0] || !kids[1]) return false;
        // Scalar multiplication is commutative; matrix/vector products are not.
        return isScalarLike(*kids[0]) || isScalarLike(*kids[1]);
    }

    return false;
}

[[nodiscard]] std::optional<DeltaInfo> matchDelta(const FormExpr& expr)
{
    if (!expr.isValid() || expr.node() == nullptr) return std::nullopt;
    const auto& node = *expr.node();
    if (node.type() != FormExprType::IndexedAccess) return std::nullopt;

    const int rank = node.indexRank().value_or(0);
    if (rank != 2) return std::nullopt;

    const auto ids_opt = node.indexIds();
    const auto ext_opt = node.indexExtents();
    if (!ids_opt || !ext_opt) return std::nullopt;

    const auto kids = node.childrenShared();
    if (kids.size() != 1u || !kids[0]) return std::nullopt;
    if (kids[0]->type() != FormExprType::Identity) return std::nullopt;

    const auto ids = *ids_opt;
    const auto ext = *ext_opt;

    DeltaInfo d;
    d.id0 = ids[0];
    d.id1 = ids[1];
    d.extent0 = ext[0];
    d.extent1 = ext[1];

    if (const auto vars_opt = node.indexVariances()) {
        const auto vars = *vars_opt;
        d.var0 = vars[0];
        d.var1 = vars[1];
    }
    if (const auto names_opt = node.indexNames()) {
        const auto names = *names_opt;
        d.name0 = std::string(names[0]);
        d.name1 = std::string(names[1]);
    }
    if (d.name0.empty()) d.name0 = "i" + std::to_string(d.id0);
    if (d.name1.empty()) d.name1 = "i" + std::to_string(d.id1);
    return d;
}

[[nodiscard]] bool isSymmetricIndexedAccess(const FormExpr& expr, std::array<int, 2>& ids_out)
{
    if (!expr.isValid() || expr.node() == nullptr) return false;
    const auto& node = *expr.node();
    if (node.type() != FormExprType::IndexedAccess) return false;
    const int rank = node.indexRank().value_or(0);
    const auto ids_opt = node.indexIds();
    if (rank != 2 || !ids_opt) return false;
    const auto kids = node.childrenShared();
    if (kids.size() != 1u || !kids[0]) return false;
    if (kids[0]->type() != FormExprType::SymmetricPart) return false;
    ids_out[0] = (*ids_opt)[0];
    ids_out[1] = (*ids_opt)[1];
    return true;
}

[[nodiscard]] bool isSkewIndexedAccess(const FormExpr& expr, std::array<int, 2>& ids_out)
{
    if (!expr.isValid() || expr.node() == nullptr) return false;
    const auto& node = *expr.node();
    if (node.type() != FormExprType::IndexedAccess) return false;
    const int rank = node.indexRank().value_or(0);
    const auto ids_opt = node.indexIds();
    if (rank != 2 || !ids_opt) return false;
    const auto kids = node.childrenShared();
    if (kids.size() != 1u || !kids[0]) return false;
    if (kids[0]->type() != FormExprType::SkewPart) return false;
    ids_out[0] = (*ids_opt)[0];
    ids_out[1] = (*ids_opt)[1];
    return true;
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

[[nodiscard]] std::vector<FormExpr> flattenMultiply(const FormExpr& expr)
{
    if (!expr.isValid() || expr.node() == nullptr) return {};
    const auto& node = *expr.node();
    if (node.type() != FormExprType::Multiply) {
        return {expr};
    }
    if (!isScalarLike(node)) {
        // Treat non-scalar multiplication as atomic.
        return {expr};
    }
    const auto kids = node.childrenShared();
    if (kids.size() != 2u || !kids[0] || !kids[1]) {
        return {expr};
    }
    auto left = flattenMultiply(FormExpr(kids[0]));
    auto right = flattenMultiply(FormExpr(kids[1]));
    left.insert(left.end(), right.begin(), right.end());
    return left;
}

[[nodiscard]] bool containsIndexId(const FormExpr& expr, int id)
{
    if (!expr.isValid() || expr.node() == nullptr) return false;
    bool found = false;
    const auto scan = [&](const auto& self, const FormExprNode& n) -> void {
        if (found) return;
        if (n.type() == FormExprType::IndexedAccess) {
            const int rank = n.indexRank().value_or(0);
            const auto ids_opt = n.indexIds();
            if (rank > 0 && ids_opt) {
                const auto ids = *ids_opt;
                for (int k = 0; k < rank; ++k) {
                    if (ids[static_cast<std::size_t>(k)] == id) {
                        found = true;
                        return;
                    }
                }
            }
        }
        for (const auto& child : n.childrenShared()) {
            if (child) self(self, *child);
            if (found) return;
        }
    };
    scan(scan, *expr.node());
    return found;
}

[[nodiscard]] bool tryDeltaComposition(std::vector<FormExpr>& factors,
                                      TensorSimplifyStats& stats)
{
    for (std::size_t a = 0; a < factors.size(); ++a) {
        const auto da = matchDelta(factors[a]);
        if (!da) continue;
        for (std::size_t b = a + 1; b < factors.size(); ++b) {
            const auto db = matchDelta(factors[b]);
            if (!db) continue;

            // Identify shared index id (if any) and compose.
            int shared = -1;
            int a_other = -1;
            int b_other = -1;
            IndexVariance a_other_var = IndexVariance::None;
            IndexVariance b_other_var = IndexVariance::None;
            std::string a_other_name{};
            std::string b_other_name{};

            if (da->id0 == db->id0) {
                shared = da->id0;
                a_other = da->id1;
                a_other_var = da->var1;
                a_other_name = da->name1;
                b_other = db->id1;
                b_other_var = db->var1;
                b_other_name = db->name1;
            } else if (da->id0 == db->id1) {
                shared = da->id0;
                a_other = da->id1;
                a_other_var = da->var1;
                a_other_name = da->name1;
                b_other = db->id0;
                b_other_var = db->var0;
                b_other_name = db->name0;
            } else if (da->id1 == db->id0) {
                shared = da->id1;
                a_other = da->id0;
                a_other_var = da->var0;
                a_other_name = da->name0;
                b_other = db->id1;
                b_other_var = db->var1;
                b_other_name = db->name1;
            } else if (da->id1 == db->id1) {
                shared = da->id1;
                a_other = da->id0;
                a_other_var = da->var0;
                a_other_name = da->name0;
                b_other = db->id0;
                b_other_var = db->var0;
                b_other_name = db->name0;
            }

            if (shared < 0) continue;

            // Build δ_{a_other b_other}.
            TensorIndex ia;
            ia.id = a_other;
            ia.name = a_other_name;
            ia.variance = a_other_var;
            ia.dimension = da->extent0;

            TensorIndex ib;
            ib.id = b_other;
            ib.name = b_other_name;
            ib.variance = b_other_var;
            ib.dimension = db->extent0;

            const auto composed = FormExpr::identity()(ia, ib);
            factors.erase(factors.begin() + static_cast<std::ptrdiff_t>(b));
            factors.erase(factors.begin() + static_cast<std::ptrdiff_t>(a));
            factors.insert(factors.begin() + static_cast<std::ptrdiff_t>(a), composed);
            stats.delta_compositions += 1;
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool tryDeltaSubstitution(std::vector<FormExpr>& factors,
                                       TensorSimplifyStats& stats)
{
    for (std::size_t idx = 0; idx < factors.size(); ++idx) {
        const auto d = matchDelta(factors[idx]);
        if (!d) continue;

        // Trace handled separately.
        if (d->id0 == d->id1) continue;

        struct Candidate {
            bool ok{false};
            std::vector<FormExpr> factors{};
        };

        auto attempt = [&](int keep_id, IndexVariance keep_var,
                           int elim_id) -> Candidate {
            Candidate c;
            bool elim_used = false;
            for (std::size_t j = 0; j < factors.size(); ++j) {
                if (j == idx) continue;
                if (containsIndexId(factors[j], elim_id)) {
                    elim_used = true;
                    break;
                }
            }
            if (!elim_used) {
                c.ok = false;
                return c;
            }
            c.ok = true;
            c.factors.reserve(factors.size() - 1);

            for (std::size_t j = 0; j < factors.size(); ++j) {
                if (j == idx) continue; // drop delta
                const auto r = contractIndices(factors[j], keep_id, keep_var, elim_id);
                if (!r.ok) {
                    c.ok = false;
                    return c;
                }
                c.factors.push_back(r.expr);
            }

            // Validate index usage after substitution.
            const auto rebuilt = rebuildMultiply(c.factors);
            const auto a = analyzeContractions(rebuilt);
            if (!a.ok) {
                c.ok = false;
            }
            return c;
        };

        const auto c0 = attempt(d->id0, d->var0, d->id1);
        const auto c1 = attempt(d->id1, d->var1, d->id0);

        if (c0.ok || c1.ok) {
            factors = c0.ok ? c0.factors : c1.factors;
            stats.delta_substitutions += 1;
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool applyDeltaRules(std::vector<FormExpr>& factors,
                                  TensorSimplifyStats& stats)
{
    bool changed = false;

    // δ_{ii} -> dim (trace-to-dimension).
    for (auto& f : factors) {
        const auto d = matchDelta(f);
        if (!d) continue;
        if (d->id0 == d->id1 && d->extent0 > 0) {
            f = FormExpr::constant(static_cast<Real>(d->extent0));
            stats.delta_traces += 1;
            changed = true;
        }
    }

    // Fixed-point within a product.
    for (;;) {
        if (tryDeltaComposition(factors, stats)) {
            changed = true;
            continue;
        }
        if (tryDeltaSubstitution(factors, stats)) {
            changed = true;
            continue;
        }
        break;
    }

    return changed;
}

[[nodiscard]] bool applySymmetryAnnihilation(std::vector<FormExpr>& factors,
                                            TensorSimplifyStats& stats)
{
    // sym(A)_{ij} * skew(B)_{ij} -> 0 (and the swapped-index variant).
    for (std::size_t i = 0; i < factors.size(); ++i) {
        std::array<int, 2> sym_ids{};
        if (!isSymmetricIndexedAccess(factors[i], sym_ids)) continue;
        for (std::size_t j = 0; j < factors.size(); ++j) {
            if (i == j) continue;
            std::array<int, 2> skew_ids{};
            if (!isSkewIndexedAccess(factors[j], skew_ids)) continue;

            const bool same = (sym_ids == skew_ids);
            const bool swapped = (sym_ids[0] == skew_ids[1] && sym_ids[1] == skew_ids[0]);
            if (!same && !swapped) continue;

            stats.symmetry_zeroes += 1;
            factors.clear();
            factors.push_back(FormExpr::constant(0.0));
            return true;
        }
    }
    return false;
}

[[nodiscard]] FormExpr simplifyOnce(const FormExpr& expr,
                                   const TensorSimplifyOptions& options,
                                   TensorSimplifyStats& stats,
                                   bool& changed)
{
    FormExpr::NodeTransform transform;
    transform = [&](const FormExprNode& node) -> std::optional<FormExpr> {
        switch (node.type()) {
            case FormExprType::IndexedAccess: {
                // δ_{ii} -> dim at the node level.
                const auto kids = node.childrenShared();
                if (kids.size() == 1u && kids[0] && kids[0]->type() == FormExprType::Identity) {
                    const int rank = node.indexRank().value_or(0);
                    const auto ids_opt = node.indexIds();
                    const auto ext_opt = node.indexExtents();
                    if (rank == 2 && ids_opt && ext_opt) {
                        const auto ids = *ids_opt;
                        const auto ext = *ext_opt;
                        if (ids[0] >= 0 && ids[0] == ids[1] && ext[0] > 0) {
                            stats.delta_traces += 1;
                            changed = true;
                            return FormExpr::constant(static_cast<Real>(ext[0]));
                        }
                    }
                }
                return std::nullopt;
            }

            case FormExprType::Multiply: {
                const auto kids = node.childrenShared();
                if (kids.size() != 2u || !kids[0] || !kids[1]) {
                    return std::nullopt;
                }
                auto a = FormExpr(kids[0]).transformNodes(transform);
                auto b = FormExpr(kids[1]).transformNodes(transform);

                if (!a.isValid() || !b.isValid()) return std::nullopt;

                // Only apply commutative tensor-algebra rewrites for scalar-like products.
                const bool scalar_product =
                    (a.node() && b.node() && isScalarLike(*a.node()) && isScalarLike(*b.node()));

                if (!scalar_product) {
                    return a * b;
                }

                auto factors = flattenMultiply(a);
                auto bf = flattenMultiply(b);
                factors.insert(factors.end(), bf.begin(), bf.end());

                bool local_changed = false;
                local_changed = applySymmetryAnnihilation(factors, stats) || local_changed;
                local_changed = applyDeltaRules(factors, stats) || local_changed;

                auto out = rebuildMultiply(factors);
                if (options.canonicalize_terms) {
                    out = canonicalizeTermOrder(out);
                }
                changed = changed || local_changed;
                return out;
            }

            case FormExprType::InnerProduct: {
                const auto kids = node.childrenShared();
                if (kids.size() != 2u || !kids[0] || !kids[1]) {
                    return std::nullopt;
                }

                auto a = FormExpr(kids[0]).transformNodes(transform);
                auto b = FormExpr(kids[1]).transformNodes(transform);

                // ε·ε identity: (a×b)·(c×d) = (a·c)(b·d) - (a·d)(b·c)
                if (a.node() && b.node() &&
                    a.node()->type() == FormExprType::CrossProduct &&
                    b.node()->type() == FormExprType::CrossProduct) {
                    const auto ak = a.node()->childrenShared();
                    const auto bk = b.node()->childrenShared();
                    if (ak.size() == 2u && bk.size() == 2u && ak[0] && ak[1] && bk[0] && bk[1]) {
                        const auto av = FormExpr(ak[0]).transformNodes(transform);
                        const auto bv = FormExpr(ak[1]).transformNodes(transform);
                        const auto cv = FormExpr(bk[0]).transformNodes(transform);
                        const auto dv = FormExpr(bk[1]).transformNodes(transform);

                        stats.epsilon_identities += 1;
                        changed = true;
                        return (inner(av, cv) * inner(bv, dv)) - (inner(av, dv) * inner(bv, cv));
                    }
                }

                return inner(a, b);
            }

            default:
                return std::nullopt;
        }
    };

    auto out = expr.transformNodes(transform);
    if (options.canonicalize_terms) {
        out = canonicalizeTermOrder(out);
    }
    return out;
}

} // namespace

TensorSimplifyResult simplifyTensorExpr(const FormExpr& expr,
                                       const TensorSimplifyOptions& options)
{
    TensorSimplifyResult out;
    out.ok = true;
    out.expr = expr;

    if (!expr.isValid() || expr.node() == nullptr) {
        out.ok = false;
        out.message = "simplifyTensorExpr: invalid expression";
        return out;
    }

    FormExpr cur = expr;
    for (int pass = 0; pass < options.max_passes; ++pass) {
        bool changed = false;
        cur = simplifyOnce(cur, options, out.stats, changed);
        out.stats.passes += 1;
        out.changed = out.changed || changed;
        if (!changed) break;
    }

    out.expr = cur;
    return out;
}

} // namespace tensor
} // namespace forms
} // namespace FE
} // namespace svmp

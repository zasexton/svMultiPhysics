/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/Tensor/SymmetryOptimizer.h"

#include "Forms/IndexExtent.h"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace svmp {
namespace FE {
namespace forms {
namespace tensor {

namespace {

[[nodiscard]] std::tuple<int, int, int, int> canonicalElasticityTuple(int i, int j, int k, int l)
{
    if (j < i) std::swap(i, j);
    if (l < k) std::swap(k, l);
    const auto a = std::make_pair(i, j);
    const auto b = std::make_pair(k, l);
    if (b < a) {
        std::swap(i, k);
        std::swap(j, l);
    }
    return {i, j, k, l};
}

[[nodiscard]] std::array<tensor::IndexVariance, 4> readIndexVariances(const FormExprNode& node)
{
    std::array<tensor::IndexVariance, 4> vars{};
    vars.fill(tensor::IndexVariance::None);
    if (const auto opt = node.indexVariances()) {
        vars = *opt;
    }
    return vars;
}

[[nodiscard]] std::array<std::string, 4> readIndexNames(const FormExprNode& node)
{
    std::array<std::string, 4> names{};
    if (const auto opt = node.indexNames()) {
        for (std::size_t k = 0; k < names.size(); ++k) {
            names[k] = std::string((*opt)[k]);
        }
    } else if (const auto ids_opt = node.indexIds()) {
        for (std::size_t k = 0; k < names.size(); ++k) {
            const int id = (*ids_opt)[k];
            names[k] = (id >= 0) ? ("i" + std::to_string(id)) : std::string{};
        }
    }
    return names;
}

} // namespace

SymmetryCanonicalComponent canonicalizeComponent(const TensorSymmetry& symmetry,
                                                 std::vector<int> indices)
{
    SymmetryCanonicalComponent out;
    out.indices = std::move(indices);

    if (out.indices.size() == 2u) {
        const bool is_sym = !symmetry.pairs.empty() &&
                            (symmetry.pairs.front().type == SymmetryType::Symmetric);
        const bool is_skew = !symmetry.pairs.empty() &&
                             (symmetry.pairs.front().type == SymmetryType::Antisymmetric);

        const int a = out.indices[0];
        const int b = out.indices[1];

        if (is_sym) {
            if (b < a) {
                std::swap(out.indices[0], out.indices[1]);
            }
            return out;
        }
        if (is_skew) {
            if (a == b) {
                out.is_zero = true;
                out.sign = 0;
                return out;
            }
            if (b < a) {
                std::swap(out.indices[0], out.indices[1]);
                out.sign = -1;
            }
            return out;
        }
        return out;
    }

    // Full elasticity: minor + major symmetries (rank-4).
    for (const auto& p : symmetry.pairs) {
        if (p.type != SymmetryType::FullElasticity) continue;
        if (out.indices.size() != 4u) {
            out.ok = false;
            out.message = "canonicalizeComponent: FullElasticity expects rank-4 indices";
            return out;
        }
        auto [i, j, k, l] = canonicalElasticityTuple(out.indices[0], out.indices[1], out.indices[2], out.indices[3]);
        out.indices = {i, j, k, l};
        return out;
    }

    return out;
}

int packedIndexSymmetricPair(int i, int j, int dim)
{
    if (dim <= 0) {
        throw std::invalid_argument("packedIndexSymmetricPair: dim must be positive");
    }
    if (i < 0 || j < 0 || i >= dim || j >= dim) {
        throw std::invalid_argument("packedIndexSymmetricPair: indices out of range");
    }
    if (j < i) {
        throw std::invalid_argument("packedIndexSymmetricPair: requires i<=j");
    }
    int idx = 0;
    for (int p = 0; p < i; ++p) {
        idx += (dim - p);
    }
    idx += (j - i);
    return idx;
}

int packedIndexAntisymmetricPair(int i, int j, int dim)
{
    if (dim <= 0) {
        throw std::invalid_argument("packedIndexAntisymmetricPair: dim must be positive");
    }
    if (i < 0 || j < 0 || i >= dim || j >= dim) {
        throw std::invalid_argument("packedIndexAntisymmetricPair: indices out of range");
    }
    if (j <= i) {
        throw std::invalid_argument("packedIndexAntisymmetricPair: requires i<j");
    }
    int idx = 0;
    for (int p = 0; p < i; ++p) {
        idx += (dim - p - 1);
    }
    idx += (j - i - 1);
    return idx;
}

int packedIndexElasticityVoigt(int i, int j, int k, int l, int dim)
{
    if (dim <= 0) {
        throw std::invalid_argument("packedIndexElasticityVoigt: dim must be positive");
    }
    if (i < 0 || j < 0 || k < 0 || l < 0 || i >= dim || j >= dim || k >= dim || l >= dim) {
        throw std::invalid_argument("packedIndexElasticityVoigt: indices out of range");
    }

    auto [ci, cj, ck, cl] = canonicalElasticityTuple(i, j, k, l);

    if (cj < ci || cl < ck) {
        throw std::logic_error("packedIndexElasticityVoigt: canonicalization failed");
    }

    const int ncomp = (dim * (dim + 1)) / 2;
    const int I = packedIndexSymmetricPair(ci, cj, dim);
    const int J = packedIndexSymmetricPair(ck, cl, dim);
    const int a = std::min(I, J);
    const int b = std::max(I, J);

    // Upper-triangle packing for a symmetric (ncomp x ncomp) matrix.
    int idx = 0;
    for (int p = 0; p < a; ++p) {
        idx += (ncomp - p);
    }
    idx += (b - a);
    return idx;
}

SymmetryLoweringResult lowerWithSymmetry(const FormExpr& expr)
{
    SymmetryLoweringResult out;
    if (!expr.isValid() || expr.node() == nullptr) {
        out.ok = false;
        out.message = "lowerWithSymmetry: invalid expression";
        return out;
    }

    const int auto_extent = forms::inferAutoIndexExtent(expr);

    FormExpr::NodeTransform transform;
    transform = [&](const FormExprNode& node) -> std::optional<FormExpr> {
        if (node.type() != FormExprType::IndexedAccess) {
            return std::nullopt;
        }

        const int rank = node.indexRank().value_or(0);
        if (rank != 2) {
            return std::nullopt;
        }

        const auto kids = node.childrenShared();
        if (kids.size() != 1u || !kids[0]) {
            return std::nullopt;
        }

        const auto ids_opt = node.indexIds();
        const auto ext_opt = node.indexExtents();
        if (!ids_opt || !ext_opt) {
            return std::nullopt;
        }

        auto base = FormExpr(kids[0]).transformNodes(transform);
        const auto base_type = base.node() ? base.node()->type() : kids[0]->type();

        // Only known symmetric/antisymmetric tensors are canonicalized.
        const bool symmetric =
            (base_type == FormExprType::SymmetricPart) || (base_type == FormExprType::Identity);
        const bool antisymmetric = (base_type == FormExprType::SkewPart);
        if (!symmetric && !antisymmetric) {
            return std::nullopt;
        }

        auto ids = *ids_opt;
        const auto ext = forms::resolveAutoIndexExtents(*ext_opt, rank, auto_extent);
        auto vars = readIndexVariances(node);
        auto names = readIndexNames(node);

        const int id0 = ids[0];
        const int id1 = ids[1];

        if (antisymmetric && id0 == id1) {
            return FormExpr::constant(0.0);
        }

        const bool swapped = (id1 < id0);
        if (!swapped && base.nodeShared() == kids[0]) {
            return std::nullopt;
        }

        if (swapped) {
            std::swap(ids[0], ids[1]);
            std::swap(names[0], names[1]);
            std::swap(vars[0], vars[1]);
        }

        FormExpr access = FormExpr::indexedAccessRawWithMetadata(std::move(base), rank, ids, ext, vars, names);
        if (antisymmetric && swapped) {
            return -access;
        }
        return access;
    };

    out.expr = expr.transformNodes(transform);
    return out;
}

} // namespace tensor
} // namespace forms
} // namespace FE
} // namespace svmp

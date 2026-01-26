/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/Tensor/TensorSymmetry.h"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <tuple>

namespace svmp {
namespace FE {
namespace forms {
namespace tensor {

namespace {

[[nodiscard]] bool hasPairType(const std::vector<SymmetryPair>& pairs,
                               int i,
                               int j,
                               SymmetryType want) noexcept
{
    for (const auto& p : pairs) {
        if (p.type != want) continue;
        if ((p.index_a == i && p.index_b == j) || (p.index_a == j && p.index_b == i)) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] std::tuple<int, int, int, int> canonicalElasticityTuple(int i, int j, int k, int l)
{
    // Minor symmetries: (i,j) and (k,l) are symmetric pairs.
    if (j < i) std::swap(i, j);
    if (l < k) std::swap(k, l);

    // Major symmetry: swap the index pairs.
    const auto a = std::make_pair(i, j);
    const auto b = std::make_pair(k, l);
    if (b < a) {
        std::swap(i, k);
        std::swap(j, l);
    }
    return {i, j, k, l};
}

} // namespace

bool TensorSymmetry::isSymmetricIn(int i, int j) const noexcept
{
    if (hasPairType(pairs, i, j, SymmetryType::Symmetric)) return true;
    if (hasPairType(pairs, i, j, SymmetryType::FullySymmetric)) return true;
    return false;
}

bool TensorSymmetry::isAntisymmetricIn(int i, int j) const noexcept
{
    if (hasPairType(pairs, i, j, SymmetryType::Antisymmetric)) return true;
    if (hasPairType(pairs, i, j, SymmetryType::FullyAntisymmetric)) return true;
    return false;
}

int TensorSymmetry::numIndependentComponents(int dim) const
{
    if (dim <= 0) {
        throw std::invalid_argument("TensorSymmetry::numIndependentComponents: dim must be positive");
    }

    for (const auto& p : pairs) {
        if (p.type == SymmetryType::FullElasticity) {
            // For 3D: 21; for 2D: 6.
            const auto comps = independentComponents(dim);
            return static_cast<int>(comps.size());
        }
    }

    if (!pairs.empty()) {
        const auto& p = pairs.front();
        if (p.type == SymmetryType::Symmetric) {
            return (dim * (dim + 1)) / 2;
        }
        if (p.type == SymmetryType::Antisymmetric) {
            return (dim * (dim - 1)) / 2;
        }
    }

    // Default: no symmetry information.
    return dim * dim;
}

std::vector<MultiIndex> TensorSymmetry::independentComponents(int dim) const
{
    if (dim <= 0) {
        throw std::invalid_argument("TensorSymmetry::independentComponents: dim must be positive");
    }

    for (const auto& p : pairs) {
        if (p.type == SymmetryType::FullElasticity) {
            std::set<std::tuple<int, int, int, int>> uniq;
            for (int i = 0; i < dim; ++i) {
                for (int j = 0; j < dim; ++j) {
                    for (int k = 0; k < dim; ++k) {
                        for (int l = 0; l < dim; ++l) {
                            uniq.insert(canonicalElasticityTuple(i, j, k, l));
                        }
                    }
                }
            }

            std::vector<MultiIndex> out;
            out.reserve(uniq.size());
            for (const auto& t : uniq) {
                const auto [i, j, k, l] = t;
                MultiIndex mi;
                mi.indices = {
                    TensorIndex{.id = i, .name = "i", .variance = IndexVariance::None, .role = IndexRole::Fixed, .dimension = dim, .fixed_value = i},
                    TensorIndex{.id = j, .name = "j", .variance = IndexVariance::None, .role = IndexRole::Fixed, .dimension = dim, .fixed_value = j},
                    TensorIndex{.id = k, .name = "k", .variance = IndexVariance::None, .role = IndexRole::Fixed, .dimension = dim, .fixed_value = k},
                    TensorIndex{.id = l, .name = "l", .variance = IndexVariance::None, .role = IndexRole::Fixed, .dimension = dim, .fixed_value = l},
                };
                out.push_back(std::move(mi));
            }
            return out;
        }
    }

    if (!pairs.empty()) {
        const auto& p = pairs.front();
        if (p.type == SymmetryType::Symmetric) {
            std::vector<MultiIndex> out;
            for (int i = 0; i < dim; ++i) {
                for (int j = i; j < dim; ++j) {
                    MultiIndex mi;
                    mi.indices = {
                        TensorIndex{.id = i, .name = "i", .variance = IndexVariance::None, .role = IndexRole::Fixed, .dimension = dim, .fixed_value = i},
                        TensorIndex{.id = j, .name = "j", .variance = IndexVariance::None, .role = IndexRole::Fixed, .dimension = dim, .fixed_value = j},
                    };
                    out.push_back(std::move(mi));
                }
            }
            return out;
        }
        if (p.type == SymmetryType::Antisymmetric) {
            std::vector<MultiIndex> out;
            for (int i = 0; i < dim; ++i) {
                for (int j = i + 1; j < dim; ++j) {
                    MultiIndex mi;
                    mi.indices = {
                        TensorIndex{.id = i, .name = "i", .variance = IndexVariance::None, .role = IndexRole::Fixed, .dimension = dim, .fixed_value = i},
                        TensorIndex{.id = j, .name = "j", .variance = IndexVariance::None, .role = IndexRole::Fixed, .dimension = dim, .fixed_value = j},
                    };
                    out.push_back(std::move(mi));
                }
            }
            return out;
        }
    }

    // Default (no symmetry): return all components of a second-order tensor.
    std::vector<MultiIndex> out;
    out.reserve(static_cast<std::size_t>(dim * dim));
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            MultiIndex mi;
            mi.indices = {
                TensorIndex{.id = i, .name = "i", .variance = IndexVariance::None, .role = IndexRole::Fixed, .dimension = dim, .fixed_value = i},
                TensorIndex{.id = j, .name = "j", .variance = IndexVariance::None, .role = IndexRole::Fixed, .dimension = dim, .fixed_value = j},
            };
            out.push_back(std::move(mi));
        }
    }
    return out;
}

TensorSymmetry TensorSymmetry::symmetric2()
{
    TensorSymmetry s;
    s.pairs.push_back(SymmetryPair{.index_a = 0, .index_b = 1, .type = SymmetryType::Symmetric});
    return s;
}

TensorSymmetry TensorSymmetry::antisymmetric2()
{
    TensorSymmetry s;
    s.pairs.push_back(SymmetryPair{.index_a = 0, .index_b = 1, .type = SymmetryType::Antisymmetric});
    return s;
}

TensorSymmetry TensorSymmetry::elasticity()
{
    TensorSymmetry s;
    s.pairs.push_back(SymmetryPair{.index_a = 0, .index_b = 1, .type = SymmetryType::FullElasticity});
    return s;
}

} // namespace tensor
} // namespace forms
} // namespace FE
} // namespace svmp


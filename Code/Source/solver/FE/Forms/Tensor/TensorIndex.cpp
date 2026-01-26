/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/Tensor/TensorIndex.h"

#include <unordered_map>

namespace svmp {
namespace FE {
namespace forms {
namespace tensor {

std::vector<int> MultiIndex::freeIndices() const
{
    std::unordered_map<int, int> counts;
    counts.reserve(indices.size());
    for (const auto& idx : indices) {
        if (idx.role == IndexRole::Fixed) {
            continue;
        }
        counts[idx.id] += 1;
    }

    std::vector<int> out;
    out.reserve(indices.size());
    for (const auto& idx : indices) {
        if (idx.role == IndexRole::Fixed) {
            continue;
        }
        const auto it = counts.find(idx.id);
        if (it != counts.end() && it->second == 1) {
            out.push_back(idx.id);
        }
    }
    return out;
}

std::vector<std::pair<int, int>> MultiIndex::contractionPairs() const
{
    std::unordered_map<int, int> first_pos;
    first_pos.reserve(indices.size());

    std::vector<std::pair<int, int>> pairs;
    for (int p = 0; p < static_cast<int>(indices.size()); ++p) {
        const auto& idx = indices[static_cast<std::size_t>(p)];
        if (idx.role == IndexRole::Fixed) {
            continue;
        }
        if (auto it = first_pos.find(idx.id); it == first_pos.end()) {
            first_pos.emplace(idx.id, p);
        } else {
            pairs.emplace_back(it->second, p);
        }
    }
    return pairs;
}

} // namespace tensor
} // namespace forms
} // namespace FE
} // namespace svmp


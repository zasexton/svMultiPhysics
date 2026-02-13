#ifndef SVMP_FE_FORMS_INDEX_EXTENT_H
#define SVMP_FE_FORMS_INDEX_EXTENT_H

/**
 * @file IndexExtent.h
 * @brief Utilities for resolving Einstein-index extents in FE/Forms
 *
 * FE/Forms supports UFL-like index notation via `forms::Index` / `forms::IndexSet`
 * and `IndexedAccess` nodes.  For convenience, an IndexSet may be constructed with
 * extent == 0 to indicate an "auto" extent that should be inferred from the form's
 * bound function-space topological dimension (falling back to 3 if unknown).
 */

#include "Forms/FormExpr.h"

#include <array>
#include <cstddef>
#include <stdexcept>

namespace svmp {
namespace FE {
namespace forms {

/**
 * @brief Infer a default extent for "auto" Einstein indices from bound spaces.
 *
 * Uses the first non-zero `SpaceSignature::topological_dimension` encountered while
 * traversing the expression tree. If multiple distinct non-zero dimensions are
 * encountered, throws.
 */
[[nodiscard]] inline int inferAutoIndexExtent(const FormExprNode& node, int fallback_extent = 3)
{
    int dim = 0;
    const auto visit = [&](const auto& self, const FormExprNode& n) -> void {
        if (const auto* sig = n.spaceSignature(); sig != nullptr) {
            const int topo = sig->topological_dimension;
            if (topo > 0) {
                if (dim == 0) {
                    dim = topo;
                } else if (dim != topo) {
                    throw std::invalid_argument("inferAutoIndexExtent: inconsistent topological dimensions");
                }
            }
        }
        for (const auto& child : n.childrenShared()) {
            if (child) {
                self(self, *child);
            }
        }
    };
    visit(visit, node);
    return (dim > 0) ? dim : fallback_extent;
}

[[nodiscard]] inline int inferAutoIndexExtent(const FormExpr& expr, int fallback_extent = 3)
{
    if (!expr.isValid() || expr.node() == nullptr) {
        return fallback_extent;
    }
    return inferAutoIndexExtent(*expr.node(), fallback_extent);
}

/**
 * @brief Replace any extent==0 entries in an IndexedAccess extents array.
 *
 * @throws std::invalid_argument if `rank` is invalid, `auto_extent` is invalid,
 *         or any used extent is negative.
 */
[[nodiscard]] inline std::array<int, 4> resolveAutoIndexExtents(std::array<int, 4> extents,
                                                                int rank,
                                                                int auto_extent)
{
    if (rank <= 0 || rank > 4) {
        throw std::invalid_argument("resolveAutoIndexExtents: rank must be 1..4");
    }
    if (auto_extent <= 0) {
        throw std::invalid_argument("resolveAutoIndexExtents: auto_extent must be > 0");
    }

    for (int k = 0; k < rank; ++k) {
        auto& e = extents[static_cast<std::size_t>(k)];
        if (e == 0) {
            e = auto_extent;
            continue;
        }
        if (e < 0) {
            throw std::invalid_argument("resolveAutoIndexExtents: invalid (negative) index extent");
        }
    }
    return extents;
}

} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_INDEX_EXTENT_H

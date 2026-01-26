/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/Tensor/TensorCanonicalize.h"

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

} // namespace tensor
} // namespace forms
} // namespace FE
} // namespace svmp


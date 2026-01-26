/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/Tensor/TensorContraction.h"

#include <unordered_map>

namespace svmp {
namespace FE {
namespace forms {
namespace tensor {

namespace {

struct IndexUseState {
    int extent{0};
    int count{0};
    std::size_t first_occurrence{0};
    std::string name{};
};

[[nodiscard]] std::string fallbackName(int id)
{
    return "i" + std::to_string(id);
}

} // namespace

ContractionAnalysis analyzeContractions(const FormExpr& expr)
{
    ContractionAnalysis out;
    out.ok = true;

    if (!expr.isValid() || expr.node() == nullptr) {
        out.ok = false;
        out.message = "TensorContraction: invalid expression";
        return out;
    }

    std::unordered_map<int, IndexUseState> uses;
    uses.reserve(16);

    std::size_t visit_counter = 0;

    const auto visit = [&](const auto& self, const FormExprNode& n) -> void {
        ++visit_counter;

        if (n.type() == FormExprType::IndexedAccess) {
            const int rank = n.indexRank().value_or(0);
            const auto ids_opt = n.indexIds();
            const auto ext_opt = n.indexExtents();
            const auto names_opt = n.indexNames();
            if (rank <= 0 || !ids_opt || !ext_opt) {
                out.ok = false;
                out.message = "TensorContraction: IndexedAccess missing index metadata";
                return;
            }

            const auto ids = *ids_opt;
            const auto ext = *ext_opt;
            for (int k = 0; k < rank; ++k) {
                const int id = ids[static_cast<std::size_t>(k)];
                const int e = ext[static_cast<std::size_t>(k)];
                if (id < 0 || e <= 0) {
                    out.ok = false;
                    out.message = "TensorContraction: invalid index id/extent for IndexedAccess";
                    return;
                }

                auto& u = uses[id];
                if (u.count == 0) {
                    u.extent = e;
                    u.first_occurrence = visit_counter;
                    if (names_opt) {
                        const auto nm = (*names_opt)[static_cast<std::size_t>(k)];
                        if (!nm.empty()) {
                            u.name = std::string(nm);
                        }
                    }
                    if (u.name.empty()) {
                        u.name = fallbackName(id);
                    }
                } else if (u.extent != e) {
                    out.ok = false;
                    out.message = "TensorContraction: index '" + u.name + "' has extent " +
                                  std::to_string(e) + " but was previously seen with extent " +
                                  std::to_string(u.extent);
                    return;
                }
                u.count += 1;
            }
        }

        for (const auto& child : n.childrenShared()) {
            if (!child || !out.ok) continue;
            self(self, *child);
        }
    };

    visit(visit, *expr.node());
    if (!out.ok) {
        return out;
    }

    // Partition indices into free/bound sets.
    for (const auto& [id, st] : uses) {
        if (st.count == 1) {
            out.free_indices.push_back(ContractionAnalysis::IndexInfo{
                .id = id,
                .name = st.name,
                .extent = st.extent,
                .count = st.count,
                .first_occurrence = st.first_occurrence,
            });
            continue;
        }
        if (st.count == 2) {
            out.bound_indices.push_back(ContractionAnalysis::IndexInfo{
                .id = id,
                .name = st.name,
                .extent = st.extent,
                .count = st.count,
                .first_occurrence = st.first_occurrence,
            });
            continue;
        }
        out.ok = false;
        out.message = "TensorContraction: index '" + st.name + "' appears " + std::to_string(st.count) +
                      " times; an index must appear exactly once (free) or twice (bound)";
        return out;
    }

    const auto by_first = [](const ContractionAnalysis::IndexInfo& a,
                             const ContractionAnalysis::IndexInfo& b) {
        return a.first_occurrence < b.first_occurrence;
    };
    std::sort(out.free_indices.begin(), out.free_indices.end(), by_first);
    std::sort(out.bound_indices.begin(), out.bound_indices.end(), by_first);
    return out;
}

} // namespace tensor
} // namespace forms
} // namespace FE
} // namespace svmp

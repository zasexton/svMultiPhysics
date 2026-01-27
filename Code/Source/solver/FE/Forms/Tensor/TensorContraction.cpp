/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/Tensor/TensorContraction.h"
#include "Forms/Tensor/TensorIndex.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
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
    tensor::IndexVariance variance{tensor::IndexVariance::None};
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
            const auto vars_opt = n.indexVariances();
            if (rank <= 0 || !ids_opt || !ext_opt) {
                out.ok = false;
                out.message = "TensorContraction: IndexedAccess missing index metadata";
                return;
            }

            const auto ids = *ids_opt;
            const auto ext = *ext_opt;
            std::array<tensor::IndexVariance, 4> vars{};
            vars.fill(tensor::IndexVariance::None);
            if (vars_opt) {
                vars = *vars_opt;
            }
            for (int k = 0; k < rank; ++k) {
                const int id = ids[static_cast<std::size_t>(k)];
                const int e = ext[static_cast<std::size_t>(k)];
                const auto v = vars[static_cast<std::size_t>(k)];
                if (id < 0 || e <= 0) {
                    out.ok = false;
                    out.message = "TensorContraction: invalid index id/extent for IndexedAccess";
                    return;
                }

                auto& u = uses[id];
                if (u.count == 0) {
                    u.extent = e;
                    u.first_occurrence = visit_counter;
                    u.variance = v;
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
                } else if (u.count == 1 &&
                           u.variance != tensor::IndexVariance::None &&
                           v != tensor::IndexVariance::None &&
                           u.variance == v) {
                    out.ok = false;
                    if (v == tensor::IndexVariance::Lower) {
                        out.message = "TensorContraction: cannot contract covariant index '" + u.name +
                                      "' with covariant index without metric";
                    } else {
                        out.message = "TensorContraction: cannot contract contravariant index '" + u.name +
                                      "' with contravariant index without metric";
                    }
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

namespace {

[[nodiscard]] std::uint64_t productExtents(const std::vector<int>& ids,
                                          const std::unordered_map<int, int>& extents)
{
    std::uint64_t prod = 1u;
    for (const int id : ids) {
        if (const auto it = extents.find(id); it != extents.end()) {
            const int e = it->second;
            if (e > 0) {
                prod *= static_cast<std::uint64_t>(static_cast<std::uint32_t>(e));
            }
        }
    }
    return prod;
}

[[nodiscard]] std::vector<int> sortedUnique(std::vector<int> v)
{
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}

[[nodiscard]] std::vector<int> intersectionSorted(const std::vector<int>& a, const std::vector<int>& b)
{
    std::vector<int> out;
    out.reserve(std::min(a.size(), b.size()));
    std::size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] == b[j]) {
            out.push_back(a[i]);
            ++i;
            ++j;
        } else if (a[i] < b[j]) {
            ++i;
        } else {
            ++j;
        }
    }
    return out;
}

[[nodiscard]] std::vector<int> symmetricDifferenceSorted(const std::vector<int>& a, const std::vector<int>& b)
{
    std::vector<int> out;
    out.reserve(a.size() + b.size());
    std::size_t i = 0, j = 0;
    while (i < a.size() || j < b.size()) {
        if (j >= b.size() || (i < a.size() && a[i] < b[j])) {
            out.push_back(a[i++]);
            continue;
        }
        if (i >= a.size() || b[j] < a[i]) {
            out.push_back(b[j++]);
            continue;
        }
        // equal -> cancel
        ++i;
        ++j;
    }
    return out;
}

[[nodiscard]] std::uint64_t effectiveOperandSize(const TensorOperand& op,
                                                 const std::unordered_map<int, int>& index_extents,
                                                 const ContractionCostModel& model)
{
    if (op.is_delta) {
        return 0u;
    }
    const std::uint64_t full = productExtents(op.indices, index_extents);
    if (!model.enable_symmetry_cost) {
        return full;
    }
    if (op.indices.size() == 2u && op.is_symmetric) {
        const int n = index_extents.count(op.indices[0]) ? index_extents.at(op.indices[0]) : 0;
        if (n > 0) {
            return static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(n + 1) / 2u;
        }
    }
    if (op.indices.size() == 2u && op.is_antisymmetric) {
        const int n = index_extents.count(op.indices[0]) ? index_extents.at(op.indices[0]) : 0;
        if (n > 0) {
            return static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(n - 1) / 2u;
        }
    }
    return full;
}

struct CostEstimate {
    std::uint64_t flops{0};
    std::uint64_t reads{0};
    std::uint64_t writes{0};
    double weighted{0.0};
};

[[nodiscard]] CostEstimate estimateContractionCost(const TensorOperand& a,
                                                   const std::vector<int>& a_indices_sorted,
                                                   const TensorOperand& b,
                                                   const std::vector<int>& b_indices_sorted,
                                                   const std::unordered_map<int, int>& index_extents,
                                                   const ContractionCostModel& model)
{
    CostEstimate out{};

    const auto shared = intersectionSorted(a_indices_sorted, b_indices_sorted);
    const auto result_indices = symmetricDifferenceSorted(a_indices_sorted, b_indices_sorted);

    const std::uint64_t size_a = effectiveOperandSize(a, index_extents, model);
    const std::uint64_t size_b = effectiveOperandSize(b, index_extents, model);
    const std::uint64_t size_out = productExtents(result_indices, index_extents);

    // FLOP proxy.
    std::uint64_t flops = 0u;
    if (model.enable_delta_shortcuts && (a.is_delta || b.is_delta)) {
        const TensorOperand& delta = a.is_delta ? a : b;
        const TensorOperand& other = a.is_delta ? b : a;
        const auto& other_ids = a.is_delta ? b_indices_sorted : a_indices_sorted;

        // For δ, treat work as proportional to the output size after substitution.
        if (shared.size() == 1u) {
            flops = productExtents(other_ids, index_extents);
        } else if (shared.size() == 2u) {
            // δ_{ij} * X_{..i..j..} reduces to a trace-like operation.
            const int elim = shared[0];
            const int e = index_extents.count(elim) ? index_extents.at(elim) : 1;
            const std::uint64_t denom = static_cast<std::uint64_t>(static_cast<std::uint32_t>(std::max(e, 1)));
            const std::uint64_t full_other = productExtents(other_ids, index_extents);
            flops = full_other / denom;
        } else {
            flops = productExtents(sortedUnique(result_indices), index_extents);
        }

        // Reads: δ has no storage; assume we read "other" once per output.
        out.reads = (delta.is_delta ? 0u : size_a) + (other.is_delta ? 0u : size_b);
    } else {
        // General contraction: O(prod extents of all indices involved).
        const auto all = sortedUnique([&] {
            std::vector<int> v;
            v.reserve(a_indices_sorted.size() + b_indices_sorted.size());
            v.insert(v.end(), a_indices_sorted.begin(), a_indices_sorted.end());
            v.insert(v.end(), b_indices_sorted.begin(), b_indices_sorted.end());
            return v;
        }());
        flops = productExtents(all, index_extents);
        out.reads = size_a + size_b;
    }

    out.flops = flops;
    out.writes = size_out;

    out.weighted = model.flop_weight * static_cast<double>(out.flops) +
                   model.memory_weight * static_cast<double>(out.reads) +
                   model.write_weight * static_cast<double>(out.writes);
    return out;
}

struct SubsetInfo {
    std::vector<int> indices_sorted{};
    TensorOperand meta{};
};

} // namespace

ContractionTransformResult contractIndices(const FormExpr& expr,
                                          int keep_id,
                                          IndexVariance keep_variance,
                                          int eliminate_id)
{
    ContractionTransformResult out;
    out.ok = true;
    out.expr = expr;

    if (!expr.isValid() || expr.node() == nullptr) {
        out.ok = false;
        out.message = "contractIndices: invalid expression";
        return out;
    }
    if (keep_id < 0 || eliminate_id < 0) {
        out.ok = false;
        out.message = "contractIndices: negative index id";
        return out;
    }
    if (keep_id == eliminate_id) {
        return out;
    }

    // Determine canonical name/extent for keep_id (or fall back to eliminate_id extent).
    std::optional<int> keep_extent;
    std::optional<std::string> keep_name;
    std::optional<IndexVariance> keep_var_observed;

    const auto scan = [&](const auto& self, const FormExprNode& n) -> void {
        if (n.type() == FormExprType::IndexedAccess) {
            const int rank = n.indexRank().value_or(0);
            const auto ids_opt = n.indexIds();
            const auto ext_opt = n.indexExtents();
            const auto names_opt = n.indexNames();
            const auto vars_opt = n.indexVariances();
            if (rank > 0 && ids_opt && ext_opt) {
                const auto ids = *ids_opt;
                const auto ext = *ext_opt;
                std::array<std::string_view, 4> names{};
                if (names_opt) names = *names_opt;
                std::array<IndexVariance, 4> vars{};
                vars.fill(IndexVariance::None);
                if (vars_opt) vars = *vars_opt;

                for (int k = 0; k < rank; ++k) {
                    const auto idx = static_cast<std::size_t>(k);
                    if (ids[idx] == keep_id) {
                        if (!keep_extent.has_value()) keep_extent = ext[idx];
                        if (!keep_name.has_value() && !names[idx].empty()) keep_name = std::string(names[idx]);
                        if (!keep_var_observed.has_value()) keep_var_observed = vars[idx];
                        return;
                    }
                }
            }
        }
        for (const auto& child : n.childrenShared()) {
            if (child && !keep_extent.has_value()) {
                self(self, *child);
            }
        }
    };
    scan(scan, *expr.node());

    if (!keep_extent.has_value()) {
        // Fall back to the eliminate-id extent if the keep id does not appear.
        const auto scan_elim = [&](const auto& self, const FormExprNode& n) -> void {
            if (n.type() == FormExprType::IndexedAccess) {
                const int rank = n.indexRank().value_or(0);
                const auto ids_opt = n.indexIds();
                const auto ext_opt = n.indexExtents();
                if (rank > 0 && ids_opt && ext_opt) {
                    const auto ids = *ids_opt;
                    const auto ext = *ext_opt;
                    for (int k = 0; k < rank; ++k) {
                        const auto idx = static_cast<std::size_t>(k);
                        if (ids[idx] == eliminate_id) {
                            keep_extent = ext[idx];
                            return;
                        }
                    }
                }
            }
            for (const auto& child : n.childrenShared()) {
                if (child && !keep_extent.has_value()) {
                    self(self, *child);
                }
            }
        };
        scan_elim(scan_elim, *expr.node());
    }

    if (!keep_extent.has_value()) {
        out.ok = false;
        out.message = "contractIndices: neither keep nor eliminate id found in expression";
        return out;
    }

    const int extent = *keep_extent;
    const std::string name = keep_name.value_or(fallbackName(keep_id));
    const IndexVariance target_variance =
        (keep_variance == IndexVariance::None) ? keep_var_observed.value_or(IndexVariance::None) : keep_variance;

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
        auto ext = *ext_opt;
        std::array<std::string, 4> names{};
        if (const auto names_opt = node.indexNames()) {
            for (std::size_t k = 0; k < names.size(); ++k) {
                names[k] = std::string((*names_opt)[k]);
            }
        }
        std::array<IndexVariance, 4> vars{};
        vars.fill(IndexVariance::None);
        if (const auto vars_opt = node.indexVariances()) {
            vars = *vars_opt;
        }

        bool changed = false;
        for (int k = 0; k < rank; ++k) {
            const auto idx = static_cast<std::size_t>(k);
            if (ids[idx] == eliminate_id) {
                ids[idx] = keep_id;
                ext[idx] = extent;
                names[idx] = name;
                vars[idx] = target_variance;
                changed = true;
            }
        }
        if (!changed) {
            return std::nullopt;
        }
        return FormExpr::indexedAccessRawWithMetadata(std::move(base), rank, std::move(ids), std::move(ext), vars, std::move(names));
    };

    out.expr = expr.transformNodes(transform);
    return out;
}

ContractionPlan optimalContractionOrder(const std::vector<TensorOperand>& operands,
                                        const std::unordered_map<int, int>& index_extents,
                                        const ContractionCostModel& model)
{
    ContractionPlan out;
    out.ok = true;

    const std::size_t n = operands.size();
    if (n == 0u) {
        out.ok = false;
        out.message = "optimalContractionOrder: no operands";
        return out;
    }
    if (n == 1u) {
        return out;
    }
    if (n > 16u) {
        out.ok = false;
        out.message = "optimalContractionOrder: operand count > 16 not supported by DP in this build";
        return out;
    }

    // Pre-sort indices on each operand for efficient set ops.
    std::vector<SubsetInfo> leaf(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < n; ++i) {
        leaf[i].meta = operands[i];
        leaf[i].meta.indices = sortedUnique(leaf[i].meta.indices);
        leaf[i].indices_sorted = leaf[i].meta.indices;
    }

    const std::size_t mask_count = 1u << n;
    struct DPEntry {
        bool valid{false};
        double cost{0.0};
        std::uint64_t flops{0};
        std::uint64_t reads{0};
        std::uint64_t writes{0};
        std::uint32_t split{0};
        std::vector<int> indices_sorted{};
        TensorOperand meta{};
    };
    std::vector<DPEntry> dp(mask_count);

    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t m = 1u << i;
        dp[m].valid = true;
        dp[m].cost = 0.0;
        dp[m].flops = 0u;
        dp[m].reads = 0u;
        dp[m].writes = 0u;
        dp[m].split = 0u;
        dp[m].indices_sorted = leaf[i].indices_sorted;
        dp[m].meta = leaf[i].meta;
    }

    const auto popcount = [](std::uint32_t x) -> int {
        return static_cast<int>(__builtin_popcount(x));
    };

    for (std::uint32_t mask = 1u; mask < static_cast<std::uint32_t>(mask_count); ++mask) {
        if (popcount(mask) <= 1) continue;

        DPEntry best;
        best.valid = false;
        best.cost = std::numeric_limits<double>::infinity();

        // Iterate proper non-empty submasks.
        for (std::uint32_t left_mask = (mask - 1u) & mask; left_mask != 0u; left_mask = (left_mask - 1u) & mask) {
            const std::uint32_t right_mask = mask ^ left_mask;
            if (right_mask == 0u) continue;

            // Canonicalize split to avoid symmetric duplicates.
            if (left_mask > right_mask) continue;

            const auto& L = dp[left_mask];
            const auto& R = dp[right_mask];
            if (!L.valid || !R.valid) continue;

            CostEstimate c = estimateContractionCost(L.meta, L.indices_sorted, R.meta, R.indices_sorted, index_extents, model);
            const double total = L.cost + R.cost + c.weighted;

            if (total < best.cost - 1e-12 ||
                (std::abs(total - best.cost) <= 1e-12 && left_mask < best.split)) {
                best.valid = true;
                best.cost = total;
                best.split = left_mask;
                best.flops = L.flops + R.flops + c.flops;
                best.reads = L.reads + R.reads + c.reads;
                best.writes = L.writes + R.writes + c.writes;
                best.indices_sorted = symmetricDifferenceSorted(L.indices_sorted, R.indices_sorted);
                best.meta.indices = best.indices_sorted;
                best.meta.is_delta = false;
                best.meta.is_symmetric = false;
                best.meta.is_antisymmetric = false;
            }
        }

        if (best.valid) {
            dp[mask] = std::move(best);
        }
    }

    const std::uint32_t full = static_cast<std::uint32_t>(mask_count - 1u);
    if (!dp[full].valid) {
        out.ok = false;
        out.message = "optimalContractionOrder: DP failed to find a plan";
        return out;
    }

    out.estimated_flops = dp[full].flops;
    out.estimated_reads = dp[full].reads;
    out.estimated_writes = dp[full].writes;

    // Reconstruct steps by building a post-order traversal of the split tree.
    int next_tmp_id = static_cast<int>(n);
    const auto build = [&](const auto& self, std::uint32_t m) -> int {
        if (popcount(m) == 1) {
            const int idx = static_cast<int>(__builtin_ctz(m));
            return idx;
        }
        const auto& e = dp[m];
        const std::uint32_t lmask = e.split;
        const std::uint32_t rmask = m ^ lmask;
        const int lhs = self(self, lmask);
        const int rhs = self(self, rmask);
        out.steps.push_back(ContractionPlan::Step{.lhs = lhs, .rhs = rhs});
        return next_tmp_id++;
    };
    (void)build(build, full);

    return out;
}

} // namespace tensor
} // namespace forms
} // namespace FE
} // namespace svmp

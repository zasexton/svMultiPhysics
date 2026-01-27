/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/Tensor/TensorCSE.h"

#include "Forms/Tensor/TensorCanonicalize.h"

#include <algorithm>
#include <deque>
#include <functional>
#include <unordered_map>
#include <unordered_set>

namespace svmp {
namespace FE {
namespace forms {
namespace tensor {

namespace {

[[nodiscard]] std::uint64_t hashString64(std::string_view s) noexcept
{
    // FNV-1a 64-bit.
    std::uint64_t h = 1469598103934665603ULL;
    for (const unsigned char c : s) {
        h ^= static_cast<std::uint64_t>(c);
        h *= 1099511628211ULL;
    }
    return h;
}

[[nodiscard]] bool isExpensive(const FormExprNode& node, const TensorCSEOptions& options) noexcept
{
    switch (node.type()) {
        case FormExprType::Determinant: return options.hoist_det;
        case FormExprType::Inverse: return options.hoist_inv;
        case FormExprType::Cofactor: return options.hoist_cofactor;
        default: return false;
    }
}

struct NodeInfo {
    std::size_t node_count{1};
    std::string key{};
    std::uint64_t key_hash{0};
    std::size_t group_id{0};
};

[[nodiscard]] std::size_t countNodes(const std::shared_ptr<FormExprNode>& node)
{
    if (!node) return 0;
    std::size_t count = 0;
    const auto visit = [&](const auto& self, const std::shared_ptr<FormExprNode>& n) -> void {
        if (!n) return;
        ++count;
        for (const auto& c : n->childrenShared()) {
            self(self, c);
        }
    };
    visit(visit, node);
    return count;
}

[[nodiscard]] std::string computeKey(const FormExpr& expr, const TensorCSEOptions& options)
{
    FormExpr canonical = expr;
    if (options.canonicalize_commutative) {
        canonical = tensor::canonicalizeTermOrder(canonical);
    }
    if (options.canonicalize_indices) {
        return tensor::toCanonicalString(canonical);
    }
    return canonical.toString();
}

} // namespace

TensorCSEPlan planTensorCSE(const FormExpr& expr, const TensorCSEOptions& options)
{
    TensorCSEPlan out;
    if (!expr.isValid() || expr.node() == nullptr) {
        out.ok = false;
        out.message = "planTensorCSE: invalid expression";
        return out;
    }

    // Collect all nodes (shared_ptr) via pre-order traversal.
    std::vector<std::shared_ptr<FormExprNode>> nodes;
    nodes.reserve(128);
    const auto visit = [&](const auto& self, const std::shared_ptr<FormExprNode>& n) -> void {
        if (!n) return;
        nodes.push_back(n);
        for (const auto& c : n->childrenShared()) {
            self(self, c);
        }
    };
    visit(visit, expr.nodeShared());

    // Group by canonical key.
    struct Group {
        std::string key{};
        std::uint64_t key_hash{0};
        std::vector<std::shared_ptr<FormExprNode>> occurrences{};
        std::size_t max_node_count{0};
        bool any_expensive{false};
    };

    std::vector<Group> groups;
    groups.reserve(nodes.size());

    std::unordered_map<std::uint64_t, std::vector<std::size_t>> hash_to_groups;
    hash_to_groups.reserve(nodes.size());

    std::unordered_map<const FormExprNode*, NodeInfo> node_info;
    node_info.reserve(nodes.size());

    for (const auto& n : nodes) {
        if (!n) continue;

        const FormExpr subexpr(n);
        NodeInfo info;
        info.node_count = countNodes(n);
        info.key = computeKey(subexpr, options);
        info.key_hash = hashString64(info.key);

        auto& bucket = hash_to_groups[info.key_hash];
        std::optional<std::size_t> group_id;
        for (const auto gid : bucket) {
            if (groups[gid].key == info.key) {
                group_id = gid;
                break;
            }
        }
        if (!group_id.has_value()) {
            group_id = groups.size();
            Group g;
            g.key = info.key;
            g.key_hash = info.key_hash;
            groups.push_back(std::move(g));
            bucket.push_back(*group_id);
        }

        info.group_id = *group_id;
        node_info.emplace(n.get(), info);

        auto& g = groups[*group_id];
        g.occurrences.push_back(n);
        g.max_node_count = std::max(g.max_node_count, info.node_count);
        g.any_expensive = g.any_expensive || isExpensive(*n, options);
    }

    // Decide which groups become temporaries.
    std::unordered_set<std::size_t> is_tmp;
    is_tmp.reserve(groups.size());
    for (std::size_t gid = 0; gid < groups.size(); ++gid) {
        const auto& g = groups[gid];
        if (static_cast<int>(g.occurrences.size()) < options.min_use_count) continue;
        if (g.any_expensive || g.max_node_count >= options.min_subtree_nodes) {
            is_tmp.insert(gid);
        }
    }

    if (is_tmp.empty()) {
        return out;
    }

    // Build dependency graph between temporary groups based on subtree containment.
    // A depends on B if an occurrence of B appears inside the representative occurrence of A.
    std::unordered_map<std::size_t, std::unordered_set<std::size_t>> deps;
    deps.reserve(is_tmp.size());
    for (const auto gid : is_tmp) {
        deps.emplace(gid, std::unordered_set<std::size_t>{});
    }

    for (const auto gid : is_tmp) {
        const auto& g = groups[gid];
        if (g.occurrences.empty()) continue;
        const auto rep = g.occurrences.front();

        const auto walk = [&](const auto& self, const std::shared_ptr<FormExprNode>& n) -> void {
            if (!n) return;
            if (n.get() != rep.get()) {
                if (const auto it = node_info.find(n.get()); it != node_info.end()) {
                    const std::size_t other = it->second.group_id;
                    if (other != gid && is_tmp.count(other) > 0u) {
                        deps[gid].insert(other);
                    }
                }
            }
            for (const auto& c : n->childrenShared()) {
                self(self, c);
            }
        };
        walk(walk, rep);
    }

    // Prefer hoisting larger expressions: drop non-expensive temporaries that are
    // strictly contained in another selected temporary.
    std::unordered_set<std::size_t> contained;
    contained.reserve(is_tmp.size());
    for (const auto& [a, ds] : deps) {
        (void)a;
        for (const auto b : ds) {
            if (is_tmp.count(b) > 0u) {
                contained.insert(b);
            }
        }
    }
    for (const auto b : contained) {
        if (is_tmp.count(b) == 0u) continue;
        if (groups[b].any_expensive) continue;
        is_tmp.erase(b);
    }
    if (is_tmp.empty()) {
        return out;
    }

    // Prune dependency graph to remaining temporaries.
    for (auto it = deps.begin(); it != deps.end();) {
        if (is_tmp.count(it->first) == 0u) {
            it = deps.erase(it);
            continue;
        }
        for (auto dit = it->second.begin(); dit != it->second.end();) {
            if (is_tmp.count(*dit) == 0u) {
                dit = it->second.erase(dit);
            } else {
                ++dit;
            }
        }
        ++it;
    }

    // Kahn topological sort (dependencies first).
    //
    // deps[A] = {B,C} means A contains/depends on B,C, so B/C must be evaluated
    // before A. We therefore sort on the reversed edges B->A.
    std::unordered_map<std::size_t, int> indeg;
    indeg.reserve(is_tmp.size());
    std::unordered_map<std::size_t, std::vector<std::size_t>> rev_adj;
    rev_adj.reserve(is_tmp.size());

    for (const auto gid : is_tmp) {
        indeg[gid] = 0;
        rev_adj[gid] = {};
    }
    for (const auto& [a, ds] : deps) {
        for (const auto b : ds) {
            indeg[a] += 1;
            rev_adj[b].push_back(a);
        }
    }

    std::deque<std::size_t> q;
    for (const auto& [gid, d] : indeg) {
        if (d == 0) q.push_back(gid);
    }

    std::vector<std::size_t> order;
    order.reserve(is_tmp.size());
    while (!q.empty()) {
        const auto n = q.front();
        q.pop_front();
        order.push_back(n);
        for (const auto dep : rev_adj[n]) {
            auto it = indeg.find(dep);
            if (it == indeg.end()) continue;
            it->second -= 1;
            if (it->second == 0) {
                q.push_back(dep);
            }
        }
    }

    // If something went wrong, fall back to deterministic order.
    if (order.size() != is_tmp.size()) {
        order.assign(is_tmp.begin(), is_tmp.end());
        std::sort(order.begin(), order.end());
    }

    // Emit temporaries in evaluation order.
    out.temporaries.reserve(order.size());
    std::size_t tmp_id = 0;
    for (const auto gid : order) {
        const auto& g = groups[gid];
        if (g.occurrences.empty()) continue;
        TensorCSETmp t;
        t.id = tmp_id++;
        t.key_hash = g.key_hash;
        t.key = g.key;
        t.use_count = g.occurrences.size();
        t.node_count = g.max_node_count;
        t.expr = FormExpr(g.occurrences.front());
        out.temporaries.push_back(std::move(t));
    }

    return out;
}

} // namespace tensor
} // namespace forms
} // namespace FE
} // namespace svmp

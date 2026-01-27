/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/Tensor/LoopStructure.h"

#include "Forms/Einsum.h"
#include "Forms/Tensor/SymmetryOptimizer.h"
#include "Forms/Tensor/TensorCanonicalize.h"
#include "Forms/Tensor/TensorSimplify.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace svmp {
namespace FE {
namespace forms {
namespace tensor {

namespace {

[[nodiscard]] bool containsIndexedAccessNode(const FormExprNode& node) noexcept
{
    if (node.type() == FormExprType::IndexedAccess) {
        return true;
    }
    for (const auto& c : node.childrenShared()) {
        if (c && containsIndexedAccessNode(*c)) return true;
    }
    return false;
}

[[nodiscard]] bool containsIndexedAccess(const FormExpr& expr) noexcept
{
    return expr.isValid() && expr.node() && containsIndexedAccessNode(*expr.node());
}

[[nodiscard]] std::uint64_t productU64(const std::vector<int>& extents)
{
    std::uint64_t p = 1;
    for (const int e : extents) {
        if (e <= 0) return 0;
        p *= static_cast<std::uint64_t>(static_cast<std::uint32_t>(e));
    }
    return p;
}

struct SignedTerm {
    FormExpr expr{};
    int sign{1};
};

void collectAddTerms(const FormExpr& expr, int sign, std::vector<SignedTerm>& out)
{
    if (!expr.isValid() || expr.node() == nullptr) {
        return;
    }

    const auto t = expr.node()->type();
    if (t == FormExprType::Add || t == FormExprType::Subtract) {
        const auto kids = expr.node()->childrenShared();
        if (kids.size() == 2u && kids[0] && kids[1]) {
            collectAddTerms(FormExpr(kids[0]), sign, out);
            collectAddTerms(FormExpr(kids[1]), (t == FormExprType::Add) ? sign : -sign, out);
            return;
        }
    }
    if (t == FormExprType::Negate) {
        const auto kids = expr.node()->childrenShared();
        if (kids.size() == 1u && kids[0]) {
            collectAddTerms(FormExpr(kids[0]), -sign, out);
            return;
        }
    }

    out.push_back(SignedTerm{.expr = expr, .sign = sign});
}

void collectMultiplyFactors(const FormExpr& expr, std::vector<FormExpr>& out)
{
    if (!expr.isValid() || expr.node() == nullptr) {
        return;
    }
    if (expr.node()->type() == FormExprType::Multiply) {
        const auto kids = expr.node()->childrenShared();
        if (kids.size() == 2u && kids[0] && kids[1]) {
            collectMultiplyFactors(FormExpr(kids[0]), out);
            collectMultiplyFactors(FormExpr(kids[1]), out);
            return;
        }
    }
    out.push_back(expr);
}

struct IndexedFactor {
    FormExpr base{};
    int base_rank{0};
    std::array<int, 4> ids{};
    std::array<int, 4> extents{};

    // base axes -> unique-axis position
    std::array<int, 4> base_axis_to_unique{};
    int unique_rank{0};
    std::array<int, 4> unique_ids{};
    std::array<int, 4> unique_extents{};

    TensorStorageKind storage{TensorStorageKind::Dense};
};

[[nodiscard]] std::optional<IndexedFactor> parseIndexedAccessFactor(const FormExpr& factor,
                                                                    const CanonicalIndexRenaming& renaming,
                                                                    std::string& error)
{
    if (!factor.isValid() || factor.node() == nullptr) {
        error = "LoopStructure: invalid factor";
        return std::nullopt;
    }
    if (factor.node()->type() != FormExprType::IndexedAccess) {
        error = "LoopStructure: expected IndexedAccess factor";
        return std::nullopt;
    }

    const auto* node = factor.node();
    const int rank = node->indexRank().value_or(0);
    const auto ids_opt = node->indexIds();
    const auto ext_opt = node->indexExtents();
    if (rank <= 0 || rank > 4 || !ids_opt || !ext_opt) {
        error = "LoopStructure: IndexedAccess missing index metadata or rank out of range";
        return std::nullopt;
    }
    const auto kids = node->childrenShared();
    if (kids.size() != 1u || !kids[0]) {
        error = "LoopStructure: IndexedAccess must have exactly 1 child";
        return std::nullopt;
    }

    IndexedFactor out;
    out.base = FormExpr(kids[0]);
    out.base_rank = rank;
    out.ids = *ids_opt;
    out.extents = *ext_opt;

    // Canonicalize ids using the expression-level deterministic mapping.
    for (int k = 0; k < rank; ++k) {
        const int old = out.ids[static_cast<std::size_t>(k)];
        if (old < 0) {
            error = "LoopStructure: IndexedAccess has negative index id";
            return std::nullopt;
        }
        const auto it = renaming.old_to_canonical.find(old);
        if (it == renaming.old_to_canonical.end()) {
            error = "LoopStructure: missing canonical id mapping for IndexedAccess";
            return std::nullopt;
        }
        out.ids[static_cast<std::size_t>(k)] = it->second;
    }

    out.base_axis_to_unique.fill(-1);
    out.unique_ids.fill(-1);
    out.unique_extents.fill(-1);

    // Unique axes in order of first appearance within this access.
    for (int k = 0; k < rank; ++k) {
        const int id = out.ids[static_cast<std::size_t>(k)];
        const int e = out.extents[static_cast<std::size_t>(k)];
        int pos = -1;
        for (int p = 0; p < out.unique_rank; ++p) {
            if (out.unique_ids[static_cast<std::size_t>(p)] == id) {
                pos = p;
                break;
            }
        }
        if (pos < 0) {
            pos = out.unique_rank++;
            out.unique_ids[static_cast<std::size_t>(pos)] = id;
            out.unique_extents[static_cast<std::size_t>(pos)] = e;
        } else if (out.unique_extents[static_cast<std::size_t>(pos)] != e) {
            error = "LoopStructure: inconsistent extents for repeated index id in IndexedAccess";
            return std::nullopt;
        }
        out.base_axis_to_unique[static_cast<std::size_t>(k)] = pos;
    }

    // Choose storage kind for the input tensor.
    if (out.base.node() != nullptr) {
        const auto bt = out.base.node()->type();
        if (bt == FormExprType::Identity && out.base_rank == 2) {
            out.storage = TensorStorageKind::KroneckerDelta;
        } else if (bt == FormExprType::SymmetricPart && out.base_rank == 2) {
            out.storage = TensorStorageKind::Symmetric2;
        } else if (bt == FormExprType::SkewPart && out.base_rank == 2) {
            out.storage = TensorStorageKind::Antisymmetric2;
        } else {
            out.storage = TensorStorageKind::Dense;
        }
    }

    return out;
}

[[nodiscard]] std::size_t storedSizeFor(TensorStorageKind storage, int rank, const std::vector<int>& extents)
{
    if (rank == 0) {
        return 1u;
    }
    if (storage == TensorStorageKind::KroneckerDelta) {
        return 0u;
    }

    const auto dense_size = [&]() -> std::size_t {
        std::size_t p = 1u;
        for (const int e : extents) p *= static_cast<std::size_t>(std::max(0, e));
        return p;
    };

    if (storage == TensorStorageKind::Symmetric2 && rank == 2 && extents.size() == 2u && extents[0] == extents[1]) {
        const int dim = extents[0];
        return static_cast<std::size_t>(dim * (dim + 1) / 2);
    }
    if (storage == TensorStorageKind::Antisymmetric2 && rank == 2 && extents.size() == 2u && extents[0] == extents[1]) {
        const int dim = extents[0];
        return static_cast<std::size_t>(dim * (dim - 1) / 2);
    }
    if (storage == TensorStorageKind::ElasticityVoigt &&
        rank == 4 && extents.size() == 4u &&
        extents[0] == extents[1] && extents[0] == extents[2] && extents[0] == extents[3]) {
        const int dim = extents[0];
        const int ncomp = (dim * (dim + 1)) / 2;
        return static_cast<std::size_t>(ncomp * (ncomp + 1) / 2);
    }
    return dense_size();
}

[[nodiscard]] TensorSpec makeInputTensorSpec(const IndexedFactor& f,
                                            const std::unordered_map<int, tensor::ContractionAnalysis::IndexInfo>& idx_info)
{
    TensorSpec s;
    s.storage = f.storage;
    s.base = f.base;
    s.base_rank = f.base_rank;
    s.base_axis_to_tensor_axis.assign(f.base_axis_to_unique.begin(),
                                      f.base_axis_to_unique.begin() + f.base_rank);

    s.rank = f.unique_rank;
    s.axes.reserve(static_cast<std::size_t>(s.rank));
    s.extents.reserve(static_cast<std::size_t>(s.rank));
    for (int k = 0; k < s.rank; ++k) {
        const int id = f.unique_ids[static_cast<std::size_t>(k)];
        s.axes.push_back(id);
        auto it = idx_info.find(id);
        const int e = (it != idx_info.end()) ? it->second.extent : f.unique_extents[static_cast<std::size_t>(k)];
        s.extents.push_back(e);
    }

    s.size = storedSizeFor(s.storage, s.rank, s.extents);
    return s;
}

[[nodiscard]] TensorSpec makeDenseTempTensorSpec(const std::vector<int>& axes,
                                                const std::unordered_map<int, tensor::ContractionAnalysis::IndexInfo>& idx_info)
{
    TensorSpec s;
    s.storage = TensorStorageKind::Dense;
    s.base = {};
    s.base_rank = 0;
    s.base_axis_to_tensor_axis.clear();
    s.rank = static_cast<int>(axes.size());
    s.axes = axes;
    s.extents.reserve(axes.size());
    std::size_t size = 1u;
    for (const int id : axes) {
        const auto it = idx_info.find(id);
        const int e = (it != idx_info.end()) ? it->second.extent : 0;
        s.extents.push_back(e);
        if (e > 0) size *= static_cast<std::size_t>(e);
    }
    if (axes.empty()) {
        size = 1u;
    }
    s.size = size;
    return s;
}

[[nodiscard]] std::vector<int> uniqueIdsFromAxes(const std::vector<int>& axes)
{
    std::vector<int> u = axes;
    std::sort(u.begin(), u.end());
    u.erase(std::unique(u.begin(), u.end()), u.end());
    return u;
}

[[nodiscard]] std::vector<int> intersectionSorted(std::vector<int> a, std::vector<int> b)
{
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    std::vector<int> out;
    std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(out));
    return out;
}

[[nodiscard]] std::vector<int> outAxesForContraction(const TensorSpec& lhs,
                                                     const TensorSpec& rhs,
                                                     const std::unordered_set<int>& shared)
{
    std::vector<int> out;
    out.reserve(lhs.axes.size() + rhs.axes.size());
    for (const int id : lhs.axes) {
        if (shared.count(id) == 0u) out.push_back(id);
    }
    for (const int id : rhs.axes) {
        if (shared.count(id) == 0u) out.push_back(id);
    }
    return out;
}

[[nodiscard]] std::uint64_t estimateFlops(const std::vector<int>& out_axes,
                                         const std::vector<int>& sum_axes,
                                         const std::unordered_map<int, tensor::ContractionAnalysis::IndexInfo>& idx_info)
{
    std::vector<int> out_ext;
    out_ext.reserve(out_axes.size());
    for (const int id : out_axes) {
        out_ext.push_back(idx_info.at(id).extent);
    }
    std::vector<int> sum_ext;
    sum_ext.reserve(sum_axes.size());
    for (const int id : sum_axes) {
        sum_ext.push_back(idx_info.at(id).extent);
    }
    const std::uint64_t out_sz = productU64(out_ext);
    const std::uint64_t sum_sz = productU64(sum_ext);
    return (out_sz == 0 ? 0 : out_sz) * (sum_sz == 0 ? 0 : sum_sz);
}

[[nodiscard]] std::vector<LoopIndex> buildLoops(const std::vector<int>& out_axes,
                                                const std::vector<int>& sum_axes,
                                                const std::unordered_map<int, tensor::ContractionAnalysis::IndexInfo>& idx_info,
                                                bool enable_vector_hints)
{
    std::vector<LoopIndex> loops;
    loops.reserve(out_axes.size() + sum_axes.size());
    for (std::size_t k = 0; k < out_axes.size(); ++k) {
        const int id = out_axes[k];
        const auto& info = idx_info.at(id);
        LoopIndex li;
        li.id = id;
        li.name = info.name;
        li.extent = info.extent;
        li.lower_bound_id = -1;
        li.lower_bound_offset = 0;
        li.vectorize = false;
        li.vector_width = 1;
        loops.push_back(std::move(li));
    }
    for (const int id : sum_axes) {
        const auto& info = idx_info.at(id);
        LoopIndex li;
        li.id = id;
        li.name = info.name;
        li.extent = info.extent;
        li.lower_bound_id = -1;
        li.lower_bound_offset = 0;
        li.vectorize = false;
        li.vector_width = 1;
        loops.push_back(std::move(li));
    }

    if (enable_vector_hints && !out_axes.empty()) {
        auto& inner = loops[out_axes.size() - 1];
        inner.vectorize = true;
        inner.vector_width = std::max(1, std::min(4, inner.extent));
    }

    return loops;
}

} // namespace

LoopNestProgram generateLoopNest(const FormExpr& expr, const LoopStructureOptions& options)
{
    LoopNestProgram program;

    if (!expr.isValid() || expr.node() == nullptr) {
        program.ok = false;
        program.message = "LoopStructure: invalid expression";
        return program;
    }

    // Scalar-only fast path.
    if (!containsIndexedAccess(expr)) {
        program.output.storage = TensorStorageKind::Dense;
        program.output.rank = 0;
        program.output.axes.clear();
        program.output.extents.clear();
        program.output.size = 1u;
        program.contributions.push_back(LoopNestProgram::Contribution{
            .tensor_id = -1,
            .available_after_op = -1,
            .scalar = expr,
        });
        return program;
    }

    FormExpr work = expr;

    if (options.enable_symmetry_lowering) {
        const auto r = lowerWithSymmetry(work);
        if (!r.ok) {
            program.ok = false;
            program.message = r.message.empty() ? "LoopStructure: symmetry lowering failed" : r.message;
            return program;
        }
        work = r.expr;
    }

    // Tensor simplification (delta/epsilon/symmetry rules) to enable sparse shortcuts.
    {
        TensorSimplifyOptions simp;
        simp.max_passes = 6;
        simp.canonicalize_terms = true;
        const auto r = simplifyTensorExpr(work, simp);
        if (r.ok) {
            work = r.expr;
        }
    }

    // Canonical index renaming for stable hashing/caching and deterministic loop metadata.
    const auto ren = computeCanonicalIndexRenaming(work);

    // Decompose into additive terms.
    std::vector<SignedTerm> terms;
    terms.reserve(8);
    collectAddTerms(work, 1, terms);

    // Analyze each term independently. In Einstein notation, repeated dummy indices
    // are scoped to each summand, so the "appear once or twice" rule applies per
    // term rather than across the entire sum.
    std::unordered_map<int, tensor::ContractionAnalysis::IndexInfo> idx_info;
    idx_info.reserve(16);

    std::optional<std::vector<int>> free_ids_opt;

    const auto ingestIndexInfo = [&](const tensor::ContractionAnalysis::IndexInfo& i) -> bool {
        const auto it = ren.old_to_canonical.find(i.id);
        if (it == ren.old_to_canonical.end()) {
            program.ok = false;
            program.message = "LoopStructure: missing canonical id mapping during contraction analysis";
            return false;
        }
        const int cid = it->second;
        auto jt = idx_info.find(cid);
        if (jt == idx_info.end()) {
            idx_info.emplace(cid, tensor::ContractionAnalysis::IndexInfo{
                                     .id = cid,
                                     .name = i.name,
                                     .extent = i.extent,
                                     .count = i.count,
                                     .first_occurrence = i.first_occurrence,
                                 });
            return true;
        }
        if (jt->second.extent != i.extent) {
            program.ok = false;
            program.message = "LoopStructure: inconsistent index extents across additive terms";
            return false;
        }
        if (jt->second.name.empty() && !i.name.empty()) {
            jt->second.name = i.name;
        }
        return true;
    };

    for (const auto& term : terms) {
        if (!containsIndexedAccess(term.expr)) {
            continue;
        }
        const auto a = analyzeContractions(term.expr);
        if (!a.ok) {
            program.ok = false;
            program.message = a.message;
            return program;
        }

        std::vector<int> term_free;
        term_free.reserve(a.free_indices.size());
        for (const auto& i : a.free_indices) {
            term_free.push_back(ren.old_to_canonical.at(i.id));
            if (!ingestIndexInfo(i)) return program;
        }
        for (const auto& i : a.bound_indices) {
            if (!ingestIndexInfo(i)) return program;
        }

        std::sort(term_free.begin(), term_free.end());
        term_free.erase(std::unique(term_free.begin(), term_free.end()), term_free.end());

        if (!free_ids_opt) {
            free_ids_opt = term_free;
        } else if (*free_ids_opt != term_free) {
            program.ok = false;
            program.message = "LoopStructure: free index mismatch across additive terms";
            return program;
        }
    }

    const std::vector<int> free_ids = free_ids_opt.value_or(std::vector<int>{});
    program.output = makeDenseTempTensorSpec(free_ids, idx_info);

    // Each term generates its own contraction chain and contributes to the output.
    for (const auto& term : terms) {
        std::vector<FormExpr> factors;
        factors.reserve(8);
        collectMultiplyFactors(term.expr, factors);

        std::vector<FormExpr> scalar_factors;
        std::vector<FormExpr> indexed_factors;
        scalar_factors.reserve(factors.size());
        indexed_factors.reserve(factors.size());

        bool unsupported = false;
        for (const auto& f : factors) {
            if (!containsIndexedAccess(f)) {
                scalar_factors.push_back(f);
                continue;
            }
            if (f.node() && f.node()->type() == FormExprType::IndexedAccess) {
                indexed_factors.push_back(f);
                continue;
            }
            // Tensor calculus inside a non-IndexedAccess factor (e.g., (A(i)+B(i))*C(i)) is not supported yet.
            unsupported = true;
            break;
        }

        if (unsupported) {
            program.ok = false;
            program.message = "LoopStructure: additive subexpression inside a product chain; distribute or fall back to einsum";
            return program;
        }

        // Pure scalar term: allowed only if output is scalar.
        if (indexed_factors.empty()) {
            if (!program.isScalar()) {
                program.ok = false;
                program.message = "LoopStructure: cannot add a scalar-only term to a tensor-valued expression";
                return program;
            }
            FormExpr s = FormExpr::constant(static_cast<Real>(term.sign));
            for (const auto& sf : scalar_factors) {
                s = s * sf;
            }
            program.contributions.push_back(LoopNestProgram::Contribution{
                .tensor_id = -1,
                .available_after_op = static_cast<int>(program.ops.size()) - 1,
                .scalar = s,
            });
            continue;
        }

        // Parse IndexedAccess operands.
        std::vector<IndexedFactor> parsed;
        parsed.reserve(indexed_factors.size());
        for (const auto& f : indexed_factors) {
            std::string err;
            auto pf = parseIndexedAccessFactor(f, ren, err);
            if (!pf) {
                program.ok = false;
                program.message = err;
                return program;
            }
            parsed.push_back(*pf);
        }

        // Build input tensor specs for this term.
        const int base_tensor_offset = static_cast<int>(program.tensors.size());
        std::vector<int> tensor_id_of_operand;
        tensor_id_of_operand.reserve(parsed.size() + 16);

        std::vector<TensorOperand> operands;
        operands.reserve(parsed.size());

        for (const auto& f : parsed) {
            const TensorSpec spec = makeInputTensorSpec(f, idx_info);
            const int tid = static_cast<int>(program.tensors.size());
            program.tensors.push_back(spec);
            tensor_id_of_operand.push_back(tid);

            TensorOperand op;
            op.indices = uniqueIdsFromAxes(spec.axes);
            op.is_delta = (spec.storage == TensorStorageKind::KroneckerDelta);
            op.is_symmetric = (spec.storage == TensorStorageKind::Symmetric2);
            op.is_antisymmetric = (spec.storage == TensorStorageKind::Antisymmetric2);
            operands.push_back(std::move(op));
        }

        // Extent map for the contraction planner.
        std::unordered_map<int, int> extents;
        extents.reserve(idx_info.size());
        for (const auto& [id, info] : idx_info) {
            extents.emplace(id, info.extent);
        }

        ContractionPlan plan;
        if (options.enable_optimal_contraction_order && operands.size() >= 2u) {
            plan = optimalContractionOrder(operands, extents, options.cost_model);
        } else {
            plan.ok = true;
            plan.steps.clear();
        }
        if (!plan.ok) {
            program.ok = false;
            program.message = plan.message.empty() ? "LoopStructure: contraction planner failed" : plan.message;
            return program;
        }

        // Execute contraction plan: create temporaries and ops.
        std::vector<TensorSpec> operand_specs;
        operand_specs.reserve(operands.size() + plan.steps.size() + 4u);
        for (std::size_t i = 0; i < operands.size(); ++i) {
            operand_specs.push_back(program.tensors[static_cast<std::size_t>(base_tensor_offset) + i]);
        }

        for (const auto& st : plan.steps) {
            const int lhs_id = st.lhs;
            const int rhs_id = st.rhs;
            if (lhs_id < 0 || rhs_id < 0 ||
                lhs_id >= static_cast<int>(operand_specs.size()) ||
                rhs_id >= static_cast<int>(operand_specs.size())) {
                program.ok = false;
                program.message = "LoopStructure: invalid contraction plan indices";
                return program;
            }

            const auto& lhs_spec = operand_specs[static_cast<std::size_t>(lhs_id)];
            const auto& rhs_spec = operand_specs[static_cast<std::size_t>(rhs_id)];

            const auto shared_vec = intersectionSorted(lhs_spec.axes, rhs_spec.axes);
            std::unordered_set<int> shared(shared_vec.begin(), shared_vec.end());

            const auto out_axes = outAxesForContraction(lhs_spec, rhs_spec, shared);

            // Determine reduction axes for this step.
            std::vector<int> sum_axes;
            sum_axes.reserve(shared_vec.size());
            for (const int id : shared_vec) sum_axes.push_back(id);

            const TensorSpec out_spec = makeDenseTempTensorSpec(out_axes, idx_info);
            const int out_tid = static_cast<int>(program.tensors.size());
            program.tensors.push_back(out_spec);

            ContractionOp op;
            op.kind = ContractionOp::Kind::Contraction;
            op.lhs = tensor_id_of_operand[static_cast<std::size_t>(lhs_id)];
            op.rhs = tensor_id_of_operand[static_cast<std::size_t>(rhs_id)];
            op.out = out_tid;
            op.out_axes = out_axes;
            op.sum_axes = sum_axes;
            op.estimated_flops = estimateFlops(op.out_axes, op.sum_axes, idx_info);
            op.loops = buildLoops(op.out_axes, op.sum_axes, idx_info, options.enable_vectorization_hints);

            program.ops.push_back(op);
            program.estimated_flops += op.estimated_flops;

            // Append new operand and its global tensor id mapping.
            operand_specs.push_back(out_spec);
            tensor_id_of_operand.push_back(out_tid);
        }

        int final_tensor = tensor_id_of_operand.back();

        // Reduce any remaining bound indices that survived due to internal repeated indices.
        if (!free_ids.empty()) {
            // Ensure output axis order is deterministic (free_ids already in analysis order).
        }

        {
            const auto& final_spec = program.tensors[static_cast<std::size_t>(final_tensor)];
            std::unordered_set<int> free_set(free_ids.begin(), free_ids.end());
            std::vector<int> reduce_axes;
            for (const int id : final_spec.axes) {
                if (free_set.count(id) == 0u) {
                    reduce_axes.push_back(id);
                }
            }

            if (!reduce_axes.empty()) {
                const TensorSpec out_spec = makeDenseTempTensorSpec(free_ids, idx_info);
                const int out_tid = static_cast<int>(program.tensors.size());
                program.tensors.push_back(out_spec);

                ContractionOp op;
                op.kind = ContractionOp::Kind::Reduction;
                op.lhs = final_tensor;
                op.rhs = -1;
                op.out = out_tid;
                op.out_axes = free_ids;
                op.sum_axes = reduce_axes;
                op.estimated_flops = estimateFlops(op.out_axes, op.sum_axes, idx_info);
                op.loops = buildLoops(op.out_axes, op.sum_axes, idx_info, options.enable_vectorization_hints);
                program.ops.push_back(op);
                program.estimated_flops += op.estimated_flops;
                final_tensor = out_tid;
            }
        }

        // Build scalar prefactor for this term.
        FormExpr pref = FormExpr::constant(static_cast<Real>(term.sign));
        for (const auto& sf : scalar_factors) {
            pref = pref * sf;
        }

        program.contributions.push_back(LoopNestProgram::Contribution{
            .tensor_id = final_tensor,
            .available_after_op = static_cast<int>(program.ops.size()) - 1,
            .scalar = pref,
        });
    }

    // Preserve structured output storage when the program is a pure view/sum of views
    // (no intermediate contraction ops).
    if (program.output.rank == 2 && program.ops.empty() && !program.contributions.empty()) {
        std::optional<TensorStorageKind> candidate;
        bool ok = true;
        for (const auto& c : program.contributions) {
            if (c.tensor_id < 0 || static_cast<std::size_t>(c.tensor_id) >= program.tensors.size()) {
                ok = false;
                break;
            }
            const auto& t = program.tensors[static_cast<std::size_t>(c.tensor_id)];
            if (t.axes != program.output.axes || t.extents != program.output.extents) {
                ok = false;
                break;
            }
            if (t.storage != TensorStorageKind::Symmetric2 &&
                t.storage != TensorStorageKind::Antisymmetric2 &&
                t.storage != TensorStorageKind::KroneckerDelta) {
                ok = false;
                break;
            }
            if (!candidate) {
                candidate = t.storage;
            } else if (*candidate != t.storage) {
                ok = false;
                break;
            }
        }
        if (ok && candidate) {
            program.output.storage = *candidate;
        }
    }

    // Stored output size depends on structure.
    program.output.size = storedSizeFor(program.output.storage, program.output.rank, program.output.extents);

    // Output loops iterate over stored components only (e.g., triangular for symmetric/skew).
    program.output_loops.clear();
    if (program.output.rank > 0 && program.output.storage != TensorStorageKind::KroneckerDelta) {
        program.output_loops = buildLoops(program.output.axes, {}, idx_info, options.enable_vectorization_hints);
        if ((program.output.storage == TensorStorageKind::Symmetric2 ||
             program.output.storage == TensorStorageKind::Antisymmetric2) &&
            program.output.axes.size() == 2u && program.output_loops.size() == 2u) {
            program.output_loops[1].lower_bound_id = program.output_loops[0].id;
            program.output_loops[1].lower_bound_offset = (program.output.storage == TensorStorageKind::Symmetric2) ? 0 : 1;
        }
    }

    return program;
}

LoopNestProgram fuseLoops(const LoopNestProgram& a, const LoopNestProgram& b)
{
    LoopNestProgram out;
    if (!a.ok || !b.ok) {
        out.ok = false;
        out.message = "fuseLoops: cannot fuse invalid programs";
        return out;
    }

    if (a.output.rank != b.output.rank ||
        a.output.axes != b.output.axes ||
        a.output.extents != b.output.extents ||
        a.output.storage != b.output.storage) {
        out.ok = false;
        out.message = "fuseLoops: output shapes do not match";
        return out;
    }

    out.output = a.output;
    out.output_loops = a.output_loops;
    out.tensors = a.tensors;
    out.ops = a.ops;
    out.contributions = a.contributions;
    out.estimated_flops = a.estimated_flops;

    const int offset = static_cast<int>(out.tensors.size());
    out.tensors.insert(out.tensors.end(), b.tensors.begin(), b.tensors.end());

    const int op_offset = static_cast<int>(out.ops.size());
    for (auto op : b.ops) {
        if (op.lhs >= 0) op.lhs += offset;
        if (op.rhs >= 0) op.rhs += offset;
        if (op.out >= 0) op.out += offset;
        out.ops.push_back(std::move(op));
    }

    for (auto c : b.contributions) {
        if (c.tensor_id >= 0) c.tensor_id += offset;
        if (c.available_after_op >= 0) {
            c.available_after_op += op_offset;
        } else {
            c.available_after_op = op_offset - 1;
        }
        out.contributions.push_back(std::move(c));
    }

    out.estimated_flops += b.estimated_flops;
    return out;
}

void optimizeLoopOrder(LoopNestProgram& program)
{
    if (!program.ok) {
        return;
    }

    auto extent_of = [&](int id) -> int {
        for (const auto& t : program.tensors) {
            for (std::size_t k = 0; k < t.axes.size(); ++k) {
                if (t.axes[k] == id) return t.extents[k];
            }
        }
        for (std::size_t k = 0; k < program.output.axes.size(); ++k) {
            if (program.output.axes[k] == id) return program.output.extents[k];
        }
        return 0;
    };

    for (auto& op : program.ops) {
        if (op.kind != ContractionOp::Kind::Contraction && op.kind != ContractionOp::Kind::Reduction) {
            continue;
        }

        const auto lhs_last = [&]() -> std::optional<int> {
            if (op.lhs < 0 || static_cast<std::size_t>(op.lhs) >= program.tensors.size()) return std::nullopt;
            const auto& a = program.tensors[static_cast<std::size_t>(op.lhs)];
            return a.axes.empty() ? std::nullopt : std::optional<int>(a.axes.back());
        }();
        const auto rhs_last = [&]() -> std::optional<int> {
            if (op.rhs < 0 || static_cast<std::size_t>(op.rhs) >= program.tensors.size()) return std::nullopt;
            const auto& a = program.tensors[static_cast<std::size_t>(op.rhs)];
            return a.axes.empty() ? std::nullopt : std::optional<int>(a.axes.back());
        }();

        auto score = [&](int id) -> std::tuple<int, int, int> {
            int s = 0;
            if (lhs_last && *lhs_last == id) s += 1;
            if (rhs_last && *rhs_last == id) s += 1;
            const int e = extent_of(id);
            return {s, e, -id};
        };

        std::sort(op.sum_axes.begin(), op.sum_axes.end(),
                  [&](int a, int b) {
                      return score(a) > score(b);
                  });

        op.loops = buildLoops(op.out_axes, op.sum_axes,
                              [&]() -> std::unordered_map<int, tensor::ContractionAnalysis::IndexInfo> {
                                  std::unordered_map<int, tensor::ContractionAnalysis::IndexInfo> m;
                                  for (const auto& t : program.tensors) {
                                      for (std::size_t k = 0; k < t.axes.size(); ++k) {
                                          m.emplace(t.axes[k], tensor::ContractionAnalysis::IndexInfo{
                                              .id = t.axes[k],
                                              .name = "i" + std::to_string(t.axes[k]),
                                              .extent = t.extents[k],
                                              .count = 0,
                                              .first_occurrence = 0,
                                          });
                                      }
                                  }
                                  for (std::size_t k = 0; k < program.output.axes.size(); ++k) {
                                      m.emplace(program.output.axes[k], tensor::ContractionAnalysis::IndexInfo{
                                          .id = program.output.axes[k],
                                          .name = "i" + std::to_string(program.output.axes[k]),
                                          .extent = program.output.extents[k],
                                          .count = 0,
                                          .first_occurrence = 0,
                                      });
                                  }
                                  return m;
                              }(),
                              true);
    }
}

TensorLoweringResult lowerTensorExpressionIncremental(const FormExpr& expr,
                                                     const LoopStructureOptions& options)
{
    TensorLoweringResult out;
    if (!expr.isValid() || expr.node() == nullptr) {
        out.ok = false;
        out.message = "lowerTensorExpressionIncremental: invalid expression";
        return out;
    }

    if (!containsIndexedAccess(expr)) {
        out.used_loop_nest = false;
        out.einsum_expanded = expr;
        return out;
    }

    FormExpr work = expr;
    if (options.enable_symmetry_lowering) {
        const auto r = lowerWithSymmetry(work);
        if (r.ok) {
            work = r.expr;
        }
    }
    {
        TensorSimplifyOptions simp;
        simp.max_passes = 6;
        simp.canonicalize_terms = true;
        const auto r = simplifyTensorExpr(work, simp);
        if (r.ok) {
            work = r.expr;
        }
    }

    // Estimate scalar expansion term count.
    std::uint64_t est_terms = 0;
    try {
        std::vector<SignedTerm> terms;
        terms.reserve(8);
        collectAddTerms(work, 1, terms);

        for (const auto& t : terms) {
            if (!containsIndexedAccess(t.expr)) {
                continue;
            }
            const auto a = analyzeContractions(t.expr);
            if (!a.ok) {
                est_terms = options.scalar_expansion_term_threshold + 1;
                break;
            }

            std::vector<int> ext;
            ext.reserve(a.free_indices.size() + a.bound_indices.size());
            for (const auto& b : a.bound_indices) ext.push_back(b.extent);
            for (const auto& f : a.free_indices) ext.push_back(f.extent);

            const std::uint64_t term_terms = productU64(ext);
            const std::uint64_t cap = options.scalar_expansion_term_threshold + 1;
            if (est_terms >= cap || term_terms >= cap) {
                est_terms = cap;
                break;
            }
            est_terms = std::min<std::uint64_t>(cap, est_terms + term_terms);
            if (est_terms >= cap) break;
        }
    } catch (...) {
        est_terms = options.scalar_expansion_term_threshold + 1;
    }

    const bool prefer_loops = (est_terms > options.scalar_expansion_term_threshold);

    if (prefer_loops) {
        out.loop = generateLoopNest(work, options);
        if (out.loop.ok) {
            out.used_loop_nest = true;
            return out;
        }
        // Fall back to einsum if loop lowering fails.
    }

    try {
        out.einsum_expanded = forms::einsum(work);
        out.used_loop_nest = false;
        return out;
    } catch (...) {
        // einsum does not support higher ranks; try loop lowering as a fallback.
        out.loop = generateLoopNest(work, options);
        if (out.loop.ok) {
            out.used_loop_nest = true;
            return out;
        }
        out.ok = false;
        out.message = "lowerTensorExpressionIncremental: both einsum and loop lowering failed";
        return out;
    }
}

} // namespace tensor
} // namespace forms
} // namespace FE
} // namespace svmp

/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/Tensor/TensorIR.h"

#include "Core/Logger.h"
#include "Forms/Einsum.h"
#include "Forms/JIT/KernelIR.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <future>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace tensor {

namespace {

constexpr std::uint64_t kFNVOffset = 14695981039346656037ULL;
constexpr std::uint64_t kFNVPrime = 1099511628211ULL;

inline void hashMix(std::uint64_t& h, std::uint64_t v) noexcept
{
    h ^= v;
    h *= kFNVPrime;
}

[[nodiscard]] std::uint64_t hashBool(bool v) noexcept { return v ? 1ULL : 0ULL; }

[[nodiscard]] std::uint64_t hashString(const std::string& s) noexcept
{
    std::uint64_t h = kFNVOffset;
    for (const char ch : s) {
        hashMix(h, static_cast<std::uint64_t>(static_cast<unsigned char>(ch)));
    }
    return h;
}

[[nodiscard]] bool containsIndexedAccess(const FormExprNode& node) noexcept
{
    if (node.type() == FormExprType::IndexedAccess) {
        return true;
    }
    for (const auto& child : node.childrenShared()) {
        if (child && containsIndexedAccess(*child)) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] std::uint64_t hashLoopOptions(const LoopStructureOptions& opt) noexcept
{
    std::uint64_t h = kFNVOffset;
    hashMix(h, hashBool(opt.enable_symmetry_lowering));
    hashMix(h, hashBool(opt.enable_optimal_contraction_order));
    hashMix(h, hashBool(opt.enable_vectorization_hints));
    hashMix(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(opt.preferred_vector_width)));
    hashMix(h, hashBool(opt.enable_delta_shortcuts));
    hashMix(h, opt.scalar_expansion_term_threshold);

    // ContractionCostModel
    // Note: cost_model has doubles; hash their bit patterns for determinism.
    hashMix(h, std::bit_cast<std::uint64_t>(opt.cost_model.flop_weight));
    hashMix(h, std::bit_cast<std::uint64_t>(opt.cost_model.memory_weight));
    hashMix(h, std::bit_cast<std::uint64_t>(opt.cost_model.write_weight));
    hashMix(h, hashBool(opt.cost_model.enable_delta_shortcuts));
    hashMix(h, hashBool(opt.cost_model.enable_symmetry_cost));
    return h;
}

[[nodiscard]] std::uint64_t hashAllocOptions(const TensorAllocationOptions& opt) noexcept
{
    std::uint64_t h = kFNVOffset;
    hashMix(h, static_cast<std::uint64_t>(opt.stack_max_entries));
    hashMix(h, static_cast<std::uint64_t>(opt.alignment_bytes));
    hashMix(h, hashBool(opt.enable_reuse));
    return h;
}

[[nodiscard]] std::uint64_t stableHashFormExprViaKernelIR(const FormExpr& expr)
{
    const auto lowered = forms::jit::lowerToKernelIR(expr);
    return lowered.ir.stableHash64();
}

[[nodiscard]] std::uint64_t stableHashTensorIR(const TensorIR& ir)
{
    std::uint64_t h = kFNVOffset;

    hashMix(h, hashLoopOptions(ir.loop_options));
    hashMix(h, hashAllocOptions(ir.alloc_options));
    hashMix(h, hashBool(ir.cacheable));

    const auto& p = ir.program;

    // Output spec.
    hashMix(h, static_cast<std::uint64_t>(p.output.storage));
    hashMix(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(p.output.rank)));
    hashMix(h, static_cast<std::uint64_t>(p.output.size));
    hashMix(h, static_cast<std::uint64_t>(p.output.axes.size()));
    for (const int id : p.output.axes) hashMix(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(id)));
    hashMix(h, static_cast<std::uint64_t>(p.output.extents.size()));
    for (const int e : p.output.extents) hashMix(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(e)));

    // Output loops (symmetry-aware lower bounds).
    hashMix(h, static_cast<std::uint64_t>(p.output_loops.size()));
    for (const auto& li : p.output_loops) {
        hashMix(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(li.id)));
        hashMix(h, hashString(li.name));
        hashMix(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(li.extent)));
        hashMix(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(li.lower_bound_id)));
        hashMix(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(li.lower_bound_offset)));
        hashMix(h, hashBool(li.vectorize));
        hashMix(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(li.vector_width)));
    }

    // Tensor specs.
    hashMix(h, static_cast<std::uint64_t>(p.tensors.size()));
    for (const auto& t : p.tensors) {
        hashMix(h, static_cast<std::uint64_t>(t.storage));
        hashMix(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(t.rank)));
        hashMix(h, static_cast<std::uint64_t>(t.size));
        hashMix(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(t.base_rank)));

        hashMix(h, static_cast<std::uint64_t>(t.axes.size()));
        for (const int id : t.axes) hashMix(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(id)));
        hashMix(h, static_cast<std::uint64_t>(t.extents.size()));
        for (const int e : t.extents) hashMix(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(e)));

        hashMix(h, static_cast<std::uint64_t>(t.base_axis_to_tensor_axis.size()));
        for (const int m : t.base_axis_to_tensor_axis) {
            hashMix(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(m)));
        }

        if (t.base.isValid()) {
            hashMix(h, stableHashFormExprViaKernelIR(t.base));
        } else {
            hashMix(h, 0ULL);
        }
    }

    // Ops (loop nests for contractions/reductions).
    hashMix(h, static_cast<std::uint64_t>(p.ops.size()));
    for (const auto& op : p.ops) {
        hashMix(h, static_cast<std::uint64_t>(op.kind));
        hashMix(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(op.lhs)));
        hashMix(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(op.rhs)));
        hashMix(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(op.out)));
        hashMix(h, op.estimated_flops);

        hashMix(h, static_cast<std::uint64_t>(op.out_axes.size()));
        for (const int id : op.out_axes) hashMix(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(id)));
        hashMix(h, static_cast<std::uint64_t>(op.sum_axes.size()));
        for (const int id : op.sum_axes) hashMix(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(id)));

        hashMix(h, static_cast<std::uint64_t>(op.loops.size()));
        for (const auto& li : op.loops) {
            hashMix(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(li.id)));
            hashMix(h, hashString(li.name));
            hashMix(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(li.extent)));
            hashMix(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(li.lower_bound_id)));
            hashMix(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(li.lower_bound_offset)));
            hashMix(h, hashBool(li.vectorize));
            hashMix(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(li.vector_width)));
        }
    }

    // Contributions.
    hashMix(h, static_cast<std::uint64_t>(p.contributions.size()));
    for (const auto& c : p.contributions) {
        hashMix(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(c.tensor_id)));
        hashMix(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(c.available_after_op)));
        if (c.scalar.isValid()) {
            hashMix(h, stableHashFormExprViaKernelIR(c.scalar));
        } else {
            hashMix(h, 0ULL);
        }
    }

    // Allocation plan (slot reuse, arena layout).
    const auto& a = ir.allocation;
    hashMix(h, hashBool(a.ok));
    hashMix(h, static_cast<std::uint64_t>(a.alignment_bytes));
    hashMix(h, static_cast<std::uint64_t>(a.stack_bytes));
    hashMix(h, static_cast<std::uint64_t>(a.arena_bytes));
    hashMix(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(a.stack_slots)));
    hashMix(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(a.arena_slots)));

    hashMix(h, static_cast<std::uint64_t>(a.allocations.size()));
    for (const auto& ta : a.allocations) {
        hashMix(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(ta.tensor_id)));
        hashMix(h, static_cast<std::uint64_t>(ta.kind));
        hashMix(h, static_cast<std::uint64_t>(ta.entries));
        hashMix(h, static_cast<std::uint64_t>(ta.bytes));
        hashMix(h, static_cast<std::uint64_t>(ta.alignment));
        hashMix(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(ta.slot)));
        hashMix(h, static_cast<std::uint64_t>(ta.arena_offset));
    }

    return h;
}

[[nodiscard]] std::uint64_t tensorIRCacheKey(const FormExpr& expr, const TensorIRLoweringOptions& options)
{
    std::uint64_t h = kFNVOffset;
    hashMix(h, stableHashFormExprViaKernelIR(expr));
    hashMix(h, hashBool(options.force_loop_nest));
    hashMix(h, hashLoopOptions(options.loop));
    hashMix(h, hashAllocOptions(options.alloc));
    return h;
}

} // namespace

std::uint64_t TensorIR::stableHash64() const
{
    if (!program.ok) {
        // Still hash invalid IR deterministically for diagnostics.
        std::uint64_t h = kFNVOffset;
        hashMix(h, 0ULL);
        hashMix(h, hashString(program.message));
        return h;
    }
    return stableHashTensorIR(*this);
}

TensorIRLoweringResult lowerToTensorIR(const FormExpr& expr, const TensorIRLoweringOptions& options)
{
    const auto compute = [&]() -> TensorIRLoweringResult {
        TensorIRLoweringResult out;
        out.ok = true;
        out.used_loop_nest = false;
        out.cacheable = true;
        out.fallback_expr = expr;

        if (!expr.isValid() || expr.node() == nullptr) {
            out.ok = false;
            out.cacheable = false;
            out.message = "lowerToTensorIR: invalid expression";
            return out;
        }

        // Cacheability matches KernelIR lowering semantics (external calls => non-cacheable).
        try {
            const auto lowered = forms::jit::lowerToKernelIR(expr);
            out.cacheable = lowered.cacheable;
        } catch (...) {
            out.cacheable = false;
        }

        const bool has_indexed = containsIndexedAccess(*expr.node());
        if (!has_indexed) {
            out.used_loop_nest = false;
            out.fallback_expr = expr;
            return out;
        }

        TensorLoweringResult lowered;
        if (options.force_loop_nest) {
            lowered.ok = true;
            lowered.loop = generateLoopNest(expr, options.loop);
            if (lowered.loop.ok) {
                lowered.used_loop_nest = true;
            } else {
                // Fall back to einsum; this is still a valid lowering result.
                try {
                    lowered.used_loop_nest = false;
                    lowered.einsum_expanded = forms::einsum(expr);
                } catch (...) {
                    lowered.ok = false;
                    lowered.message = lowered.loop.message.empty()
                                         ? "lowerToTensorIR(force): loop lowering failed and einsum failed"
                                         : lowered.loop.message;
                }
            }
            lowered.preferred_loop_nest = true;
            lowered.estimated_scalar_terms = 0;
            lowered.decision_reason = "forced";
        } else {
            lowered = lowerTensorExpressionIncremental(expr, options.loop);
        }

        if (!lowered.ok) {
            out.ok = false;
            out.cacheable = false;
            out.message = lowered.message.empty() ? "lowerToTensorIR: tensor lowering failed" : lowered.message;
            return out;
        }

        out.used_loop_nest = lowered.used_loop_nest;
        if (options.log_decisions) {
            FE_LOG_INFO("TensorIR lowering: used_loop_nest=" +
                        std::string(out.used_loop_nest ? "true" : "false") +
                        " reason=" + (lowered.decision_reason.empty() ? std::string{"unknown"} : lowered.decision_reason) +
                        " est_terms=" + std::to_string(lowered.estimated_scalar_terms) +
                        " threshold=" + std::to_string(options.loop.scalar_expansion_term_threshold));
        }

        if (!lowered.used_loop_nest) {
            out.fallback_expr = lowered.einsum_expanded.isValid() ? lowered.einsum_expanded : expr;
            return out;
        }

        TensorIR ir;
        ir.program = std::move(lowered.loop);
        ir.loop_options = options.loop;
        ir.alloc_options = options.alloc;
        ir.cacheable = out.cacheable;
        ir.allocation = planTensorAllocation(ir.program, options.alloc);
        if (!ir.allocation.ok) {
            out.ok = false;
            out.cacheable = false;
            out.message = ir.allocation.message.empty()
                              ? "lowerToTensorIR: failed to plan tensor temporaries"
                              : ir.allocation.message;
            return out;
        }

        out.ir = std::move(ir);
        out.fallback_expr = {};
        return out;
    };

    if (!options.enable_cache) {
        return compute();
    }

    const std::uint64_t key = tensorIRCacheKey(expr, options);

    struct Cache {
        std::mutex mutex{};
        std::unordered_map<std::uint64_t, std::shared_future<TensorIRLoweringResult>> entries{};
    };
    static Cache cache;

    std::shared_future<TensorIRLoweringResult> fut;
    std::shared_ptr<std::promise<TensorIRLoweringResult>> promise;

    {
        std::lock_guard<std::mutex> lock(cache.mutex);
        auto it = cache.entries.find(key);
        if (it != cache.entries.end()) {
            fut = it->second;
        } else {
            promise = std::make_shared<std::promise<TensorIRLoweringResult>>();
            fut = promise->get_future().share();
            cache.entries.emplace(key, fut);
        }
    }

    if (promise) {
        try {
            promise->set_value(compute());
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    }

    return fut.get();
}

} // namespace tensor
} // namespace forms
} // namespace FE
} // namespace svmp

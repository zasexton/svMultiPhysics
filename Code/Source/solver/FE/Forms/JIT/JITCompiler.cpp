/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/JIT/JITCompiler.h"

#include "Core/Logger.h"

#include "Assembly/JIT/KernelArgs.h"
#include "Forms/JIT/JITEngine.h"
#include "Forms/JIT/KernelIR.h"
#include "Forms/JIT/LLVMGen.h"
#include "Forms/JIT/LLVMJITBuildInfo.h"
#include "Forms/Tensor/TensorIR.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef SVMP_FE_ENABLE_LLVM_JIT
#define SVMP_FE_ENABLE_LLVM_JIT 0
#endif

namespace svmp {
namespace FE {
namespace forms {
namespace jit {

namespace {

constexpr std::uint64_t kFNVOffset = 14695981039346656037ULL;
constexpr std::uint64_t kFNVPrime = 1099511628211ULL;

inline void hashMix(std::uint64_t& h, std::uint64_t v) noexcept
{
    h ^= v;
    h *= kFNVPrime;
}

[[nodiscard]] bool containsTestOrTrial(const FormExprNode& node) noexcept
{
    if (node.type() == FormExprType::TestFunction || node.type() == FormExprType::TrialFunction) {
        return true;
    }
    for (const auto& child : node.childrenShared()) {
        if (child && containsTestOrTrial(*child)) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool containsPreviousSolutionRef(const FormExprNode& node) noexcept
{
    if (node.type() == FormExprType::PreviousSolutionRef) {
        return true;
    }
    for (const auto& child : node.childrenShared()) {
        if (child && containsPreviousSolutionRef(*child)) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool sameSpaceSignature(const FormExprNode::SpaceSignature& a,
                                      const FormExprNode::SpaceSignature& b) noexcept
{
    return a.space_type == b.space_type &&
           a.field_type == b.field_type &&
           a.continuity == b.continuity &&
           a.value_dimension == b.value_dimension &&
           a.topological_dimension == b.topological_dimension &&
           a.polynomial_order == b.polynomial_order &&
           a.element_type == b.element_type;
}

[[nodiscard]] std::optional<FormExprNode::SpaceSignature>
inferFunctionalTrialSpaceSignature(const FormExprNode& node, bool& out_conflict) noexcept
{
    out_conflict = false;
    std::optional<FormExprNode::SpaceSignature> sig;

    const auto visit = [&](const auto& self, const FormExprNode& n) -> void {
        if (out_conflict) {
            return;
        }

        if (n.type() == FormExprType::DiscreteField || n.type() == FormExprType::StateField) {
            if (const auto* s = n.spaceSignature()) {
                if (!sig) {
                    sig = *s;
                } else if (!sameSpaceSignature(*sig, *s)) {
                    out_conflict = true;
                    return;
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
    return sig;
}

#if SVMP_FE_ENABLE_LLVM_JIT
[[nodiscard]] std::string toHex(std::uint64_t v)
{
    std::ostringstream oss;
    oss << std::hex << v;
    return oss.str();
}

[[nodiscard]] const char* toString(IntegralDomain d) noexcept
{
    switch (d) {
        case IntegralDomain::Cell:
            return "Cell";
        case IntegralDomain::Boundary:
            return "Boundary";
        case IntegralDomain::InteriorFace:
            return "InteriorFace";
        case IntegralDomain::InterfaceFace:
            return "InterfaceFace";
        default:
            return "Unknown";
    }
}
#endif

[[nodiscard]] std::uint64_t hashString(std::string_view s) noexcept
{
    std::uint64_t h = kFNVOffset;
    for (const char ch : s) {
        hashMix(h, static_cast<std::uint64_t>(static_cast<unsigned char>(ch)));
    }
    return h;
}

#if SVMP_FE_ENABLE_LLVM_JIT
[[nodiscard]] std::uint64_t hashTensorOptions(const TensorJITOptions& opt) noexcept
{
    std::uint64_t h = kFNVOffset;
    hashMix(h, static_cast<std::uint64_t>(opt.mode));
    hashMix(h, static_cast<std::uint64_t>(opt.force_loop_nest ? 1u : 0u));
    hashMix(h, static_cast<std::uint64_t>(opt.enable_symmetry_lowering ? 1u : 0u));
    hashMix(h, static_cast<std::uint64_t>(opt.enable_optimal_contraction_order ? 1u : 0u));
    hashMix(h, static_cast<std::uint64_t>(opt.enable_vectorization_hints ? 1u : 0u));
    hashMix(h, static_cast<std::uint64_t>(opt.enable_delta_shortcuts ? 1u : 0u));
    hashMix(h, opt.scalar_expansion_term_threshold);
    hashMix(h, static_cast<std::uint64_t>(opt.temp_stack_max_entries));
    hashMix(h, static_cast<std::uint64_t>(opt.temp_alignment_bytes));
    hashMix(h, static_cast<std::uint64_t>(opt.temp_enable_reuse ? 1u : 0u));
    hashMix(h, static_cast<std::uint64_t>(opt.enable_polly ? 1u : 0u));
    return h;
}

[[nodiscard]] std::uint64_t hashSpecializationCodegenOptions(const JITSpecializationOptions& opt) noexcept
{
    std::uint64_t h = kFNVOffset;
    hashMix(h, static_cast<std::uint64_t>(opt.enable_loop_unroll_metadata ? 1u : 0u));
    hashMix(h, static_cast<std::uint64_t>(opt.max_unroll_trip_count));
    return h;
}

struct SpaceSigHash {
    std::uint64_t h{0};
};

[[nodiscard]] SpaceSigHash hashSpaceSig(const std::optional<FormExprNode::SpaceSignature>& sig)
{
    std::uint64_t h = kFNVOffset;
    if (!sig) {
        hashMix(h, 0ULL);
        return SpaceSigHash{h};
    }

    hashMix(h, static_cast<std::uint64_t>(sig->space_type));
    hashMix(h, static_cast<std::uint64_t>(sig->field_type));
    hashMix(h, static_cast<std::uint64_t>(sig->continuity));
    hashMix(h, static_cast<std::uint64_t>(static_cast<std::uint32_t>(sig->value_dimension)));
    hashMix(h, static_cast<std::uint64_t>(static_cast<std::uint32_t>(sig->topological_dimension)));
    hashMix(h, static_cast<std::uint64_t>(static_cast<std::uint32_t>(sig->polynomial_order)));
    hashMix(h, static_cast<std::uint64_t>(sig->element_type));
    return SpaceSigHash{h};
}

[[nodiscard]] std::string formatValidationIssue(const ValidationIssue& issue)
{
    std::ostringstream oss;
    oss << issue.message;
    if (!issue.subexpr.empty()) {
        oss << " (subexpr: " << issue.subexpr << ")";
    }
    return oss.str();
}

struct GroupKey {
    IntegralDomain domain{IntegralDomain::Cell};
    int boundary_marker{-1};
    int interface_marker{-1};

    friend bool operator==(const GroupKey& a, const GroupKey& b) noexcept
    {
        return a.domain == b.domain &&
               a.boundary_marker == b.boundary_marker &&
               a.interface_marker == b.interface_marker;
    }
};

struct GroupKeyHash {
    std::size_t operator()(const GroupKey& k) const noexcept
    {
        std::uint64_t h = kFNVOffset;
        hashMix(h, static_cast<std::uint64_t>(k.domain));
        hashMix(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(k.boundary_marker)));
        hashMix(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(k.interface_marker)));
        return static_cast<std::size_t>(h);
    }
};

struct KernelGroupPlan {
    GroupKey key{};
    std::uint64_t cache_key{0};
    bool cacheable{true};
    std::vector<std::size_t> term_indices{};
};

[[nodiscard]] std::uint64_t computeKernelCacheKey(const FormIR& ir,
                                                  const KernelGroupPlan& group,
                                                  std::uint64_t combined_ir_hash,
                                                  const SpaceSigHash& test_sig_hash,
                                                  const SpaceSigHash& trial_sig_hash,
                                                  const JITOptions& jit_options,
                                                  std::string_view target_triple,
                                                  std::string_view data_layout,
                                                  std::string_view cpu_name,
                                                  std::string_view cpu_features,
                                                  std::string_view llvm_version,
                                                  const JITCompileSpecialization* specialization) noexcept
{
    std::uint64_t h = kFNVOffset;
    hashMix(h, static_cast<std::uint64_t>(assembly::jit::kKernelArgsABIVersionV4));
    hashMix(h, static_cast<std::uint64_t>(ir.kind()));
    hashMix(h, static_cast<std::uint64_t>(group.key.domain));
    hashMix(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(group.key.boundary_marker)));
    hashMix(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(group.key.interface_marker)));
    hashMix(h, combined_ir_hash);
    hashMix(h, test_sig_hash.h);
    hashMix(h, trial_sig_hash.h);
    hashMix(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(jit_options.optimization_level)));
    hashMix(h, static_cast<std::uint64_t>(jit_options.vectorize ? 1u : 0u));
    hashMix(h, hashTensorOptions(jit_options.tensor));
    hashMix(h, hashSpecializationCodegenOptions(jit_options.specialization));
    hashMix(h, static_cast<std::uint64_t>(jit_options.debug_info ? 1u : 0u));
    hashMix(h, hashString(target_triple));
    hashMix(h, hashString(data_layout));
    hashMix(h, hashString(cpu_name));
    hashMix(h, hashString(cpu_features));
    hashMix(h, hashString(llvm_version));

    const bool use_spec = (specialization != nullptr) && (specialization->domain == group.key.domain);
    hashMix(h, static_cast<std::uint64_t>(use_spec ? 1u : 0u));
    if (use_spec) {
        const auto mixOptU32 = [&](const std::optional<std::uint32_t>& v) {
            hashMix(h, static_cast<std::uint64_t>(v.has_value() ? 1u : 0u));
            if (v) {
                hashMix(h, static_cast<std::uint64_t>(*v));
            }
        };

        mixOptU32(specialization->n_qpts_minus);
        mixOptU32(specialization->n_test_dofs_minus);
        mixOptU32(specialization->n_trial_dofs_minus);
        mixOptU32(specialization->n_qpts_plus);
        mixOptU32(specialization->n_test_dofs_plus);
        mixOptU32(specialization->n_trial_dofs_plus);
    }
    return h;
}

[[nodiscard]] std::string stableSymbolForKernel(std::uint64_t cache_key)
{
    return "svmp_fe_jit_kernel_" + toHex(cache_key);
}

struct CompilationPlan {
    bool ok{false};
    bool cacheable{true};
    std::string message{};
    std::vector<KernelGroupPlan> groups{};
};

[[nodiscard]] CompilationPlan buildPlan(const FormIR& ir,
                                       const ValidationOptions& validation,
                                       const JITOptions& jit_options,
                                       std::string_view target_triple,
                                       std::string_view data_layout,
                                       std::string_view cpu_name,
                                       std::string_view cpu_features,
                                       std::string_view llvm_version,
                                       const JITCompileSpecialization* specialization)
{
    CompilationPlan plan;
    plan.ok = false;
    plan.cacheable = true;

    if (!ir.isCompiled()) {
        plan.message = "JITCompiler: FormIR is not compiled";
        return plan;
    }

    const auto vr = canCompile(ir, validation);
    if (!vr.ok) {
        plan.message = "JITCompiler: FormIR is not JIT-compatible";
        if (vr.first_issue) {
            plan.message += ": " + formatValidationIssue(*vr.first_issue);
        }
        plan.cacheable = false;
        return plan;
    }

    plan.cacheable = plan.cacheable && vr.cacheable;

    std::unordered_map<GroupKey, KernelGroupPlan, GroupKeyHash> groups_by_key;
    groups_by_key.reserve(ir.terms().size());

    for (std::size_t i = 0; i < ir.terms().size(); ++i) {
        const auto& term = ir.terms()[i];
        GroupKey key;
        key.domain = term.domain;
        key.boundary_marker = term.boundary_marker;
        key.interface_marker = term.interface_marker;

        auto& g = groups_by_key[key];
        g.key = key;
        g.term_indices.push_back(i);
    }

    const auto test_sig_hash = hashSpaceSig(ir.testSpace());
    const auto trial_sig_hash = hashSpaceSig(ir.trialSpace());

    plan.groups.reserve(groups_by_key.size());

    for (auto& [key, group] : groups_by_key) {
        std::uint64_t combined_ir_hash = kFNVOffset;
        bool cacheable = true;

        for (const auto idx : group.term_indices) {
            const auto& term = ir.terms()[idx];
            std::uint64_t term_hash = 0;
            bool term_cacheable = true;

            if (jit_options.tensor.mode != TensorLoweringMode::Off) {
                tensor::TensorIRLoweringOptions tensor_opts;
                tensor_opts.enable_cache = true;
                tensor_opts.force_loop_nest =
                    (jit_options.tensor.mode == TensorLoweringMode::On) || jit_options.tensor.force_loop_nest;
                tensor_opts.log_decisions = jit_options.tensor.log_decisions;

                tensor_opts.loop.enable_symmetry_lowering = jit_options.tensor.enable_symmetry_lowering;
                tensor_opts.loop.enable_optimal_contraction_order = jit_options.tensor.enable_optimal_contraction_order;
                tensor_opts.loop.enable_vectorization_hints = jit_options.vectorize && jit_options.tensor.enable_vectorization_hints;
                tensor_opts.loop.enable_delta_shortcuts = jit_options.tensor.enable_delta_shortcuts;
                tensor_opts.loop.scalar_expansion_term_threshold = jit_options.tensor.scalar_expansion_term_threshold;

                tensor_opts.alloc.stack_max_entries = jit_options.tensor.temp_stack_max_entries;
                tensor_opts.alloc.alignment_bytes = jit_options.tensor.temp_alignment_bytes;
                tensor_opts.alloc.enable_reuse = jit_options.tensor.temp_enable_reuse;

                const auto tl = tensor::lowerToTensorIR(term.integrand, tensor_opts);
                if (!tl.ok) {
                    throw std::runtime_error(tl.message.empty() ? "JITCompiler: tensor lowering failed" : tl.message);
                }

                if (tl.used_loop_nest) {
                    term_hash = tl.ir.stableHash64();
                    term_cacheable = tl.cacheable;
                } else {
                    const auto effective =
                        tl.fallback_expr.isValid() ? tl.fallback_expr : term.integrand;
                    const auto lowered = lowerToKernelIR(effective);
                    term_hash = lowered.ir.stableHash64();
                    term_cacheable = lowered.cacheable;
                }
            } else {
                const auto lowered = lowerToKernelIR(term.integrand);
                term_hash = lowered.ir.stableHash64();
                term_cacheable = lowered.cacheable;
            }

            cacheable = cacheable && term_cacheable;
            hashMix(combined_ir_hash, term_hash);
            hashMix(combined_ir_hash, static_cast<std::uint64_t>(static_cast<std::int64_t>(term.time_derivative_order)));
        }

        group.cacheable = cacheable;
        group.cache_key = computeKernelCacheKey(ir,
                                                group,
                                                combined_ir_hash,
                                                test_sig_hash,
                                                trial_sig_hash,
                                                jit_options,
                                                target_triple,
                                                data_layout,
                                                cpu_name,
                                                cpu_features,
                                                llvm_version,
                                                specialization);
        plan.cacheable = plan.cacheable && group.cacheable;
        plan.groups.push_back(std::move(group));
    }

    plan.ok = true;
    return plan;
}
#endif

} // namespace

struct JITCompiler::Impl {
    explicit Impl(JITOptions options_in)
        : options(std::move(options_in))
    {
        options.enable = true;
        engine = JITEngine::create(options);
    }

    struct KernelCacheEntry {
        JITCompiledKernel kernel{};
        std::list<std::uint64_t>::iterator lru_it{};
    };

    JITOptions options{};
    std::unique_ptr<JITEngine> engine{};

    mutable std::mutex mutex{};
    std::unordered_map<std::uint64_t, KernelCacheEntry> kernel_cache{};
    std::list<std::uint64_t> kernel_cache_lru{};
    JITKernelCacheStats kernel_cache_stats{};
    std::atomic<std::uint64_t> unique_symbol_counter{0};
};

JITCompiler::JITCompiler(JITOptions options)
    : impl_(std::make_unique<Impl>(std::move(options)))
{
}

std::shared_ptr<JITCompiler> JITCompiler::getOrCreate(const JITOptions& options)
{
    struct CompilerKey {
        int optimization_level{2};
        bool vectorize{true};
        bool cache_kernels{true};
        std::string cache_directory{};
        bool cache_diagnostics{false};
        std::size_t max_in_memory_kernels{0};
        bool dump_kernel_ir{false};
        bool dump_llvm_ir{false};
        bool dump_llvm_ir_optimized{false};
        bool debug_info{false};
        std::string dump_directory{};
        bool specialization_enable_loop_unroll_metadata{true};
        std::uint32_t specialization_max_unroll_trip_count{32};
        TensorJITOptions tensor{};

        bool operator==(const CompilerKey& other) const noexcept
        {
            return optimization_level == other.optimization_level &&
                   vectorize == other.vectorize &&
                   cache_kernels == other.cache_kernels &&
                   cache_directory == other.cache_directory &&
                   cache_diagnostics == other.cache_diagnostics &&
                   max_in_memory_kernels == other.max_in_memory_kernels &&
                   dump_kernel_ir == other.dump_kernel_ir &&
                   dump_llvm_ir == other.dump_llvm_ir &&
                   dump_llvm_ir_optimized == other.dump_llvm_ir_optimized &&
                   debug_info == other.debug_info &&
                   dump_directory == other.dump_directory &&
                   specialization_enable_loop_unroll_metadata == other.specialization_enable_loop_unroll_metadata &&
                   specialization_max_unroll_trip_count == other.specialization_max_unroll_trip_count &&
                   tensor.mode == other.tensor.mode &&
                   tensor.force_loop_nest == other.tensor.force_loop_nest &&
                   tensor.enable_symmetry_lowering == other.tensor.enable_symmetry_lowering &&
                   tensor.enable_optimal_contraction_order == other.tensor.enable_optimal_contraction_order &&
                   tensor.enable_vectorization_hints == other.tensor.enable_vectorization_hints &&
                   tensor.enable_delta_shortcuts == other.tensor.enable_delta_shortcuts &&
                   tensor.scalar_expansion_term_threshold == other.tensor.scalar_expansion_term_threshold &&
                   tensor.temp_stack_max_entries == other.tensor.temp_stack_max_entries &&
                   tensor.temp_alignment_bytes == other.tensor.temp_alignment_bytes &&
                   tensor.temp_enable_reuse == other.tensor.temp_enable_reuse &&
                   tensor.enable_polly == other.tensor.enable_polly;
        }
    };

    struct CompilerKeyHash {
        std::size_t operator()(const CompilerKey& k) const noexcept
        {
            std::uint64_t h = kFNVOffset;
            hashMix(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(k.optimization_level)));
            hashMix(h, static_cast<std::uint64_t>(k.vectorize ? 1u : 0u));
            hashMix(h, static_cast<std::uint64_t>(k.cache_kernels ? 1u : 0u));
            hashMix(h, hashString(k.cache_directory));
            hashMix(h, static_cast<std::uint64_t>(k.cache_diagnostics ? 1u : 0u));
            hashMix(h, static_cast<std::uint64_t>(k.max_in_memory_kernels));
            hashMix(h, static_cast<std::uint64_t>(k.dump_kernel_ir ? 1u : 0u));
            hashMix(h, static_cast<std::uint64_t>(k.dump_llvm_ir ? 1u : 0u));
            hashMix(h, static_cast<std::uint64_t>(k.dump_llvm_ir_optimized ? 1u : 0u));
            hashMix(h, static_cast<std::uint64_t>(k.debug_info ? 1u : 0u));
            hashMix(h, hashString(k.dump_directory));
            hashMix(h, static_cast<std::uint64_t>(k.specialization_enable_loop_unroll_metadata ? 1u : 0u));
            hashMix(h, static_cast<std::uint64_t>(k.specialization_max_unroll_trip_count));
            hashMix(h, static_cast<std::uint64_t>(k.tensor.mode));
            hashMix(h, static_cast<std::uint64_t>(k.tensor.force_loop_nest ? 1u : 0u));
            hashMix(h, static_cast<std::uint64_t>(k.tensor.enable_symmetry_lowering ? 1u : 0u));
            hashMix(h, static_cast<std::uint64_t>(k.tensor.enable_optimal_contraction_order ? 1u : 0u));
            hashMix(h, static_cast<std::uint64_t>(k.tensor.enable_vectorization_hints ? 1u : 0u));
            hashMix(h, static_cast<std::uint64_t>(k.tensor.enable_delta_shortcuts ? 1u : 0u));
            hashMix(h, k.tensor.scalar_expansion_term_threshold);
            hashMix(h, static_cast<std::uint64_t>(k.tensor.temp_stack_max_entries));
            hashMix(h, static_cast<std::uint64_t>(k.tensor.temp_alignment_bytes));
            hashMix(h, static_cast<std::uint64_t>(k.tensor.temp_enable_reuse ? 1u : 0u));
            hashMix(h, static_cast<std::uint64_t>(k.tensor.enable_polly ? 1u : 0u));
            return static_cast<std::size_t>(h);
        }
    };

    static std::mutex registry_mutex;
    static std::unordered_map<CompilerKey, std::shared_ptr<JITCompiler>, CompilerKeyHash> registry;

    CompilerKey key;
    key.optimization_level = options.optimization_level;
    key.vectorize = options.vectorize;
    key.cache_kernels = options.cache_kernels;
    key.cache_directory = options.cache_directory;
    key.cache_diagnostics = options.cache_diagnostics;
    key.max_in_memory_kernels = options.max_in_memory_kernels;
    key.dump_kernel_ir = options.dump_kernel_ir;
    key.dump_llvm_ir = options.dump_llvm_ir;
    key.dump_llvm_ir_optimized = options.dump_llvm_ir_optimized;
    key.debug_info = options.debug_info;
    key.dump_directory = options.dump_directory;
    key.specialization_enable_loop_unroll_metadata = options.specialization.enable_loop_unroll_metadata;
    key.specialization_max_unroll_trip_count = options.specialization.max_unroll_trip_count;
    key.tensor = options.tensor;

    std::lock_guard<std::mutex> lock(registry_mutex);
    if (auto it = registry.find(key); it != registry.end()) {
        return it->second;
    }

    JITOptions opt = options;
    opt.enable = true;

    auto compiler = std::shared_ptr<JITCompiler>(new JITCompiler(std::move(opt)));
    registry[key] = compiler;
    return compiler;
}

#if SVMP_FE_ENABLE_LLVM_JIT
namespace {

[[nodiscard]] JITCompileResult compileFormIRImpl(JITCompiler::Impl& impl,
                                                 const FormIR& ir,
                                                 const ValidationOptions& validation,
                                                 const JITCompileSpecialization* specialization)
{
    JITCompileResult out;
    out.ok = false;
    out.cacheable = true;

    std::lock_guard<std::mutex> lock(impl.mutex);

    if (!impl.engine || !impl.engine->available()) {
        out.message = "JITCompiler: LLVM JIT engine is not available at runtime";
        out.cacheable = false;
        return out;
    }

    const std::string target_triple = impl.engine->targetTriple();
    const std::string data_layout = impl.engine->dataLayoutString();
    const std::string cpu_name = impl.engine->cpuName();
    const std::string cpu_features = impl.engine->cpuFeaturesString();
    const std::string llvm_version = llvmVersionString();

    CompilationPlan plan;
    try {
        plan = buildPlan(ir,
                         validation,
                         impl.options,
                         target_triple,
                         data_layout,
                         cpu_name,
                         cpu_features,
                         llvm_version,
                         specialization);
    } catch (const std::exception& e) {
        out.message = std::string("JITCompiler: failed to lower FormIR to KernelIR: ") + e.what();
        out.message += "\n\nFormIR dump:\n" + ir.dump();
        out.cacheable = false;
        return out;
    }

    if (!plan.ok) {
        out.message = std::move(plan.message);
        out.message += "\n\nFormIR dump:\n" + ir.dump();
        out.cacheable = false;
        return out;
    }

    out.cacheable = out.cacheable && plan.cacheable;

    std::uint64_t local_hits = 0;
    std::uint64_t local_symbol_hits = 0;
    std::uint64_t local_misses = 0;
    std::uint64_t local_stores = 0;
    std::uint64_t local_evictions = 0;

    const auto touch = [&](JITCompiler::Impl::KernelCacheEntry& entry) {
        impl.kernel_cache_lru.splice(impl.kernel_cache_lru.begin(),
                                     impl.kernel_cache_lru,
                                     entry.lru_it);
        entry.lru_it = impl.kernel_cache_lru.begin();
    };

    const auto evict_if_needed = [&]() {
        const std::size_t max_entries = impl.options.max_in_memory_kernels;
        if (max_entries == 0u) {
            return;
        }

        while (impl.kernel_cache.size() > max_entries) {
            const std::uint64_t victim = impl.kernel_cache_lru.back();
            impl.kernel_cache_lru.pop_back();
            impl.kernel_cache.erase(victim);
            ++local_evictions;
            impl.kernel_cache_stats.evictions += 1u;
        }
    };

    for (const auto& group : plan.groups) {
        JITCompiledKernel k;
        k.domain = group.key.domain;
        k.boundary_marker = group.key.boundary_marker;
        k.interface_marker = group.key.interface_marker;
        k.cache_key = group.cache_key;
        k.cacheable = group.cacheable;

        const bool enable_cache = impl.options.cache_kernels && group.cacheable;
        if (enable_cache) {
            if (auto it = impl.kernel_cache.find(group.cache_key); it != impl.kernel_cache.end()) {
                ++local_hits;
                impl.kernel_cache_stats.hits += 1u;
                touch(it->second);
                out.kernels.push_back(it->second.kernel);
                continue;
            }

            std::uintptr_t addr = 0;
            const std::string stable_symbol = stableSymbolForKernel(group.cache_key);
            if (impl.engine && impl.engine->tryLookup(stable_symbol, addr)) {
                k.symbol = stable_symbol;
                k.address = addr;

                impl.kernel_cache_lru.push_front(group.cache_key);
                JITCompiler::Impl::KernelCacheEntry entry;
                entry.kernel = k;
                entry.lru_it = impl.kernel_cache_lru.begin();
                impl.kernel_cache.emplace(group.cache_key, std::move(entry));
                ++local_symbol_hits;
                ++local_stores;
                impl.kernel_cache_stats.engine_symbol_hits += 1u;
                impl.kernel_cache_stats.stores += 1u;
                evict_if_needed();

                out.kernels.push_back(k);
                continue;
            }

            ++local_misses;
            impl.kernel_cache_stats.misses += 1u;
        }

        std::string symbol = stableSymbolForKernel(group.cache_key);
        if (!enable_cache) {
            const auto id = impl.unique_symbol_counter.fetch_add(1, std::memory_order_relaxed);
            symbol += "_inst" + std::to_string(id);
        }

        try {
            std::uintptr_t addr = 0;
            LLVMGen gen(impl.options);

            const JITCompileSpecialization* group_spec = nullptr;
            if (specialization != nullptr && specialization->domain == group.key.domain) {
                group_spec = specialization;
            }

            const auto r = gen.compileAndAddKernel(*impl.engine,
                                                   ir,
                                                   group.term_indices,
                                                   group.key.domain,
                                                   group.key.boundary_marker,
                                                   group.key.interface_marker,
                                                   symbol,
                                                   addr,
                                                   group_spec);
            if (!r.ok) {
                throw std::runtime_error(r.message.empty() ? "LLVMGen: kernel generation failed" : r.message);
            }
            k.symbol = symbol;
            k.address = addr;
        } catch (const std::exception& e) {
            out.message = std::string("JITCompiler: LLVM compilation failed: ") + e.what();
            out.message += "\nKernel symbol: " + symbol;
            out.message += "\nDomain: ";
            out.message += toString(group.key.domain);
            out.message += "\nBoundary marker: " + std::to_string(group.key.boundary_marker);
            out.message += "\nInterface marker: " + std::to_string(group.key.interface_marker);
            out.message += "\nCache key: 0x" + toHex(group.cache_key);
            out.message += "\n\nFormIR dump:\n" + ir.dump();
            out.cacheable = false;
            return out;
        }

        out.kernels.push_back(k);
        if (enable_cache) {
            impl.kernel_cache_lru.push_front(group.cache_key);
            JITCompiler::Impl::KernelCacheEntry entry;
            entry.kernel = k;
            entry.lru_it = impl.kernel_cache_lru.begin();
            impl.kernel_cache.emplace(group.cache_key, std::move(entry));
            ++local_stores;
            impl.kernel_cache_stats.stores += 1u;
            evict_if_needed();
        }
    }

    if (impl.options.cache_diagnostics) {
        std::ostringstream oss;
        const auto object_stats = impl.engine ? impl.engine->objectCacheStats() : JITObjectCacheStats{};
        oss << "JIT cache: groups=" << plan.groups.size()
            << ", hits=" << local_hits
            << ", symbol_hits=" << local_symbol_hits
            << ", misses=" << local_misses
            << ", stores=" << local_stores
            << ", evictions=" << local_evictions
            << ", kernel_cache_size=" << impl.kernel_cache.size()
            << ", objcache_gets=" << object_stats.get_calls
            << ", objcache_mem_hits=" << object_stats.mem_hits
            << ", objcache_disk_hits=" << object_stats.disk_hits
            << ", objcache_misses=" << object_stats.misses;
        FE_LOG_INFO(oss.str());
    }

    out.ok = true;
    return out;
}

} // namespace
#endif

JITCompileResult JITCompiler::compile(const FormIR& ir, const ValidationOptions& validation)
{
    JITCompileResult out;
    out.ok = false;
    out.cacheable = true;

    if (!impl_) {
        out.message = "JITCompiler: internal error (missing impl)";
        return out;
    }

#if !SVMP_FE_ENABLE_LLVM_JIT
    (void)ir;
    (void)validation;
    out.message = "JITCompiler: FE was built without LLVM JIT support";
	    out.cacheable = false;
	    return out;
#else
	    return compileFormIRImpl(*impl_, ir, validation, /*specialization=*/nullptr);
#endif
}

JITCompileResult JITCompiler::compileSpecialized(const FormIR& ir,
                                                 const JITCompileSpecialization& specialization,
                                                 const ValidationOptions& validation)
{
    JITCompileResult out;
    out.ok = false;
    out.cacheable = true;

    if (!impl_) {
        out.message = "JITCompiler: internal error (missing impl)";
        return out;
    }

#if !SVMP_FE_ENABLE_LLVM_JIT
    (void)ir;
    (void)specialization;
    (void)validation;
    out.message = "JITCompiler: FE was built without LLVM JIT support";
    out.cacheable = false;
    return out;
#else
    return compileFormIRImpl(*impl_, ir, validation, &specialization);
#endif
}

JITCompileResult JITCompiler::compileFunctional(const FormExpr& integrand,
                                                IntegralDomain domain,
                                                const ValidationOptions& validation)
{
    JITCompileResult out;
    out.ok = false;
    out.cacheable = true;

    if (domain != IntegralDomain::Cell && domain != IntegralDomain::Boundary) {
        out.message = "JITCompiler::compileFunctional: only Cell and Boundary domains are supported";
        out.cacheable = false;
        return out;
    }

    if (!integrand.isValid() || integrand.node() == nullptr) {
        out.message = "JITCompiler::compileFunctional: invalid integrand";
        out.cacheable = false;
        return out;
    }

    const auto& root = *integrand.node();

    if (containsTestOrTrial(root)) {
        out.message = "JITCompiler::compileFunctional: integrand contains TestFunction/TrialFunction; functional kernels must use DiscreteField/StateField instead";
        out.cacheable = false;
        return out;
    }

    bool conflict = false;
    const auto trial_sig = inferFunctionalTrialSpaceSignature(root, conflict);
    if (conflict) {
        out.message =
            "JITCompiler::compileFunctional: multiple distinct DiscreteField/StateField space signatures found; multi-field functional kernels are not supported";
        out.cacheable = false;
        return out;
    }

    if (containsPreviousSolutionRef(root) && !trial_sig.has_value()) {
        out.message =
            "JITCompiler::compileFunctional: PreviousSolutionRef requires a DiscreteField/StateField operand (to infer scalar vs vector shape)";
        out.cacheable = false;
        return out;
    }

    FormIR ir;
    ir.setCompiled(true);
    ir.setKind(FormKind::Linear);
    ir.setRequiredData(assembly::RequiredData::None);
    ir.setFieldRequirements({});
    ir.setTestSpace(std::nullopt);
    ir.setTrialSpace(trial_sig);
    ir.setMaxTimeDerivativeOrder(0);

    IntegralTerm term;
    term.domain = domain;
    term.boundary_marker = -1;
    term.interface_marker = -1;
    term.time_derivative_order = 0;
    term.integrand = integrand;
    term.debug_string = root.toString();
    term.required_data = assembly::RequiredData::None;

    std::vector<IntegralTerm> terms;
    terms.push_back(std::move(term));
    ir.setTerms(std::move(terms));
    ir.setDump("JIT synthetic functional FormIR");

    return compile(ir, validation);
}

JITCacheStats JITCompiler::cacheStats() const
{
    if (!impl_) {
        return {};
    }

    std::lock_guard<std::mutex> lock(impl_->mutex);
    JITCacheStats out_stats;
    out_stats.kernel = impl_->kernel_cache_stats;
    out_stats.kernel.size = static_cast<std::uint64_t>(impl_->kernel_cache.size());
    out_stats.object = impl_->engine ? impl_->engine->objectCacheStats() : JITObjectCacheStats{};
    return out_stats;
}

void JITCompiler::resetCacheStats()
{
    if (!impl_) {
        return;
    }

    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->kernel_cache_stats = JITKernelCacheStats{};
    if (impl_->engine) {
        impl_->engine->resetObjectCacheStats();
    }
}

} // namespace jit
} // namespace forms
} // namespace FE
} // namespace svmp

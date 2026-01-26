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

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
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
                                                  std::string_view llvm_version) noexcept
{
    std::uint64_t h = kFNVOffset;
    hashMix(h, static_cast<std::uint64_t>(assembly::jit::kKernelArgsABIVersionV3));
    hashMix(h, static_cast<std::uint64_t>(ir.kind()));
    hashMix(h, static_cast<std::uint64_t>(group.key.domain));
    hashMix(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(group.key.boundary_marker)));
    hashMix(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(group.key.interface_marker)));
    hashMix(h, combined_ir_hash);
    hashMix(h, test_sig_hash.h);
    hashMix(h, trial_sig_hash.h);
    hashMix(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(jit_options.optimization_level)));
    hashMix(h, static_cast<std::uint64_t>(jit_options.vectorize ? 1u : 0u));
    hashMix(h, 0u);
    hashMix(h, 0u);
    hashMix(h, hashString(target_triple));
    hashMix(h, hashString(data_layout));
    hashMix(h, hashString(cpu_name));
    hashMix(h, hashString(cpu_features));
    hashMix(h, hashString(llvm_version));
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
                                       std::string_view llvm_version)
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
            const auto lowered = lowerToKernelIR(term.integrand);
            cacheable = cacheable && lowered.cacheable;

            const auto term_hash = lowered.ir.stableHash64();
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
                                                llvm_version);
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

    JITOptions options{};
    std::unique_ptr<JITEngine> engine{};

    std::mutex mutex{};
    std::unordered_map<std::uint64_t, JITCompiledKernel> kernel_cache{};
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

        bool operator==(const CompilerKey& other) const noexcept
        {
            return optimization_level == other.optimization_level &&
                   vectorize == other.vectorize &&
                   cache_kernels == other.cache_kernels &&
                   cache_directory == other.cache_directory;
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
    std::lock_guard<std::mutex> lock(impl_->mutex);

    if (!impl_->engine || !impl_->engine->available()) {
        out.message = "JITCompiler: LLVM JIT engine is not available at runtime";
        out.cacheable = false;
        return out;
    }

    const std::string target_triple = impl_->engine->targetTriple();
    const std::string data_layout = impl_->engine->dataLayoutString();
    const std::string cpu_name = impl_->engine->cpuName();
    const std::string cpu_features = impl_->engine->cpuFeaturesString();
    const std::string llvm_version = llvmVersionString();

    CompilationPlan plan;
    try {
        plan = buildPlan(ir,
                         validation,
                         impl_->options,
                         target_triple,
                         data_layout,
                         cpu_name,
                         cpu_features,
                         llvm_version);
    } catch (const std::exception& e) {
        out.message = std::string("JITCompiler: failed to lower FormIR to KernelIR: ") + e.what();
        out.cacheable = false;
        return out;
    }

    if (!plan.ok) {
        out.message = std::move(plan.message);
        out.cacheable = false;
        return out;
    }

    out.cacheable = out.cacheable && plan.cacheable;

    for (const auto& group : plan.groups) {
        JITCompiledKernel k;
        k.domain = group.key.domain;
        k.boundary_marker = group.key.boundary_marker;
        k.interface_marker = group.key.interface_marker;
        k.cache_key = group.cache_key;
        k.cacheable = group.cacheable;

        const bool enable_cache = impl_->options.cache_kernels && group.cacheable;
        if (enable_cache) {
            if (auto it = impl_->kernel_cache.find(group.cache_key); it != impl_->kernel_cache.end()) {
                out.kernels.push_back(it->second);
                continue;
            }
        }

        std::string symbol = stableSymbolForKernel(group.cache_key);
        if (!enable_cache) {
            const auto id = impl_->unique_symbol_counter.fetch_add(1, std::memory_order_relaxed);
            symbol += "_inst" + std::to_string(id);
        }

        try {
            std::uintptr_t addr = 0;
            LLVMGen gen(impl_->options);
            const auto r = gen.compileAndAddKernel(*impl_->engine,
                                                   ir,
                                                   group.term_indices,
                                                   group.key.domain,
                                                   group.key.boundary_marker,
                                                   group.key.interface_marker,
                                                   symbol,
                                                   addr);
            if (!r.ok) {
                throw std::runtime_error(r.message.empty() ? "LLVMGen: kernel generation failed" : r.message);
            }
            k.symbol = symbol;
            k.address = addr;
        } catch (const std::exception& e) {
            out.message = std::string("JITCompiler: LLVM compilation failed: ") + e.what();
            out.cacheable = false;
            return out;
        }

        out.kernels.push_back(k);
        if (enable_cache) {
            impl_->kernel_cache.emplace(group.cache_key, k);
        }
    }

    out.ok = true;
    return out;
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

} // namespace jit
} // namespace forms
} // namespace FE
} // namespace svmp

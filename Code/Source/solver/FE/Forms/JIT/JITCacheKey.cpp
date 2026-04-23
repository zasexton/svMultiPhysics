#include "Forms/JIT/JITCacheKey.h"

#include "Forms/FormIR.h"

#include <optional>
#include <sstream>

namespace svmp {
namespace FE {
namespace forms {
namespace jit {

void mixCacheKey(std::uint64_t& h, std::uint64_t v) noexcept
{
    h ^= v;
    h *= kCacheKeyFNVPrime;
}

[[nodiscard]] std::uint64_t hashStringForCacheKey(std::string_view s) noexcept
{
    std::uint64_t h = kCacheKeyFNVOffset;
    for (const char ch : s) {
        mixCacheKey(h, static_cast<std::uint64_t>(static_cast<unsigned char>(ch)));
    }
    return h;
}

[[nodiscard]] std::uint64_t hashTensorOptionsForCacheKey(const TensorJITOptions& opt) noexcept
{
    std::uint64_t h = kCacheKeyFNVOffset;
    mixCacheKey(h, static_cast<std::uint64_t>(opt.mode));
    mixCacheKey(h, static_cast<std::uint64_t>(opt.force_loop_nest ? 1u : 0u));
    mixCacheKey(h, static_cast<std::uint64_t>(opt.enable_symmetry_lowering ? 1u : 0u));
    mixCacheKey(h, static_cast<std::uint64_t>(opt.enable_optimal_contraction_order ? 1u : 0u));
    mixCacheKey(h, static_cast<std::uint64_t>(opt.enable_vectorization_hints ? 1u : 0u));
    mixCacheKey(h, static_cast<std::uint64_t>(opt.enable_delta_shortcuts ? 1u : 0u));
    mixCacheKey(h, opt.scalar_expansion_term_threshold);
    mixCacheKey(h, static_cast<std::uint64_t>(opt.temp_stack_max_entries));
    mixCacheKey(h, static_cast<std::uint64_t>(opt.temp_alignment_bytes));
    mixCacheKey(h, static_cast<std::uint64_t>(opt.temp_enable_reuse ? 1u : 0u));
    mixCacheKey(h, static_cast<std::uint64_t>(opt.enable_polly ? 1u : 0u));
    mixCacheKey(h, static_cast<std::uint64_t>(opt.enable_loop_tiling ? 1u : 0u));
    mixCacheKey(h, static_cast<std::uint64_t>(opt.tile_size));
    mixCacheKey(h, static_cast<std::uint64_t>(opt.min_tiling_extent));
    return h;
}

[[nodiscard]] std::uint64_t hashSpecializationCodegenOptionsForCacheKey(
    const JITSpecializationOptions& opt) noexcept
{
    std::uint64_t h = kCacheKeyFNVOffset;
    mixCacheKey(h, static_cast<std::uint64_t>(opt.enable_loop_unroll_metadata ? 1u : 0u));
    mixCacheKey(h, static_cast<std::uint64_t>(opt.max_unroll_trip_count));
    // Both bytes/op estimates are live codegen inputs. Include them so changing
    // the estimate invalidates cached kernels compiled with the old policy.
    mixCacheKey(h, static_cast<std::uint64_t>(opt.bytes_per_op_estimate));
    mixCacheKey(h, static_cast<std::uint64_t>(opt.raw_bytes_per_op_estimate));
    return h;
}

[[nodiscard]] std::uint64_t computeKernelCacheKey(const KernelCacheKeyInputs& in) noexcept
{
    std::uint64_t h = kCacheKeyFNVOffset;
    mixCacheKey(h, in.cache_key_schema_version);
    mixCacheKey(h, static_cast<std::uint64_t>(in.abi_version));
    mixCacheKey(h, static_cast<std::uint64_t>(in.abi_layout_revision));
    mixCacheKey(h, static_cast<std::uint64_t>(in.form_kind));
    mixCacheKey(h, static_cast<std::uint64_t>(in.domain));
    mixCacheKey(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(in.boundary_marker)));
    mixCacheKey(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(in.interface_marker)));
    mixCacheKey(h, in.combined_ir_hash);
    mixCacheKey(h, in.test_space_hash);
    mixCacheKey(h, in.trial_space_hash);
    mixCacheKey(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(in.jit_options.optimization_level)));
    mixCacheKey(h, static_cast<std::uint64_t>(in.jit_options.vectorize ? 1u : 0u));
    mixCacheKey(h, static_cast<std::uint64_t>(in.jit_options.simd_batch ? 1u : 0u));
    mixCacheKey(h, hashTensorOptionsForCacheKey(in.jit_options.tensor));
    mixCacheKey(h, hashSpecializationCodegenOptionsForCacheKey(in.jit_options.specialization));
    mixCacheKey(h, static_cast<std::uint64_t>(in.jit_options.debug_info ? 1u : 0u));
    mixCacheKey(h, hashStringForCacheKey(in.target_triple));
    mixCacheKey(h, hashStringForCacheKey(in.data_layout));
    mixCacheKey(h, hashStringForCacheKey(in.cpu_name));
    mixCacheKey(h, hashStringForCacheKey(in.cpu_features));
    mixCacheKey(h, hashStringForCacheKey(in.llvm_version));

    // Hardware profile affects codegen decisions such as term-group splitting,
    // colocation budgets, and unroll suppression.
    mixCacheKey(h, in.hardware_profile_hash);

    const bool use_spec = (in.specialization != nullptr) && (in.specialization->domain == in.domain);
    mixCacheKey(h, static_cast<std::uint64_t>(use_spec ? 1u : 0u));
    if (use_spec) {
        const auto mixOptU32 = [&](const std::optional<std::uint32_t>& v) {
            mixCacheKey(h, static_cast<std::uint64_t>(v.has_value() ? 1u : 0u));
            if (v) {
                mixCacheKey(h, static_cast<std::uint64_t>(*v));
            }
        };

        mixOptU32(in.specialization->n_qpts_minus);
        mixOptU32(in.specialization->n_test_dofs_minus);
        mixOptU32(in.specialization->n_trial_dofs_minus);
        mixOptU32(in.specialization->n_qpts_plus);
        mixOptU32(in.specialization->n_test_dofs_plus);
        mixOptU32(in.specialization->n_trial_dofs_plus);

        mixCacheKey(h, static_cast<std::uint64_t>(in.specialization->is_affine ? 1u : 0u));

        // text_budget_bytes affects whether DOF loop unroll metadata is emitted
        // for specialized kernels, so it belongs to the matched-specialization key.
        mixCacheKey(h, static_cast<std::uint64_t>(in.jit_options.specialization.text_budget_bytes));
    }
    return h;
}

[[nodiscard]] std::string cacheKeyToHex(std::uint64_t cache_key)
{
    std::ostringstream oss;
    oss << std::hex << cache_key;
    return oss.str();
}

[[nodiscard]] std::string stableSymbolForKernel(std::uint64_t cache_key)
{
    return "svmp_fe_jit_kernel_" + cacheKeyToHex(cache_key);
}

} // namespace jit
} // namespace forms
} // namespace FE
} // namespace svmp

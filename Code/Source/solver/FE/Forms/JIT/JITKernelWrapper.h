#ifndef SVMP_FE_FORMS_JIT_JIT_KERNEL_WRAPPER_H
#define SVMP_FE_FORMS_JIT_JIT_KERNEL_WRAPPER_H

/**
 * @file JITKernelWrapper.h
 * @brief AssemblyKernel adapter that can dispatch to a future LLVM JIT backend
 *
 * JIT implementation:
 * - Wraps an existing interpreter kernel and (when available) dispatches to
 *   JIT-compiled kernels generated from the underlying FormIR.
 * - Falls back to the wrapped interpreter kernel on any compile/runtime
 *   unavailability.
 *
 * This header contains no LLVM dependencies.
 */

#include "Assembly/AssemblyKernel.h"
#include "Forms/FormExpr.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <string>

namespace svmp {
namespace FE {
namespace forms {

enum class IntegralDomain : std::uint8_t;
class FormIR;

namespace jit {

class JITCompiler;

class JITKernelWrapper final : public assembly::AssemblyKernel {
public:
    JITKernelWrapper(std::shared_ptr<assembly::AssemblyKernel> fallback,
                     JITOptions options);
    ~JITKernelWrapper() override = default;

    JITKernelWrapper(const JITKernelWrapper&) = delete;
    JITKernelWrapper& operator=(const JITKernelWrapper&) = delete;

    [[nodiscard]] assembly::RequiredData getRequiredData() const override;
    [[nodiscard]] std::vector<assembly::FieldRequirement> fieldRequirements() const override;
    [[nodiscard]] assembly::MaterialStateSpec materialStateSpec() const noexcept override;
    [[nodiscard]] std::vector<params::Spec> parameterSpecs() const override;

    void resolveParameterSlots(
        const std::function<std::optional<std::uint32_t>(std::string_view)>& slot_of_real_param) override;

    void resolveInlinableConstitutives() override;

    [[nodiscard]] bool hasCell() const noexcept override;
    [[nodiscard]] bool hasBoundaryFace() const noexcept override;
    [[nodiscard]] bool hasInteriorFace() const noexcept override;
    [[nodiscard]] bool hasInterfaceFace() const noexcept override;

    void computeCell(const assembly::AssemblyContext& ctx,
                     assembly::KernelOutput& output) override;

    void computeBoundaryFace(const assembly::AssemblyContext& ctx,
                             int boundary_marker,
                             assembly::KernelOutput& output) override;

    void computeInteriorFace(const assembly::AssemblyContext& ctx_minus,
                             const assembly::AssemblyContext& ctx_plus,
                             assembly::KernelOutput& output_minus,
                             assembly::KernelOutput& output_plus,
                             assembly::KernelOutput& coupling_minus_plus,
                             assembly::KernelOutput& coupling_plus_minus) override;

    void computeInterfaceFace(const assembly::AssemblyContext& ctx_minus,
                              const assembly::AssemblyContext& ctx_plus,
                              int interface_marker,
                              assembly::KernelOutput& output_minus,
                              assembly::KernelOutput& output_plus,
                              assembly::KernelOutput& coupling_minus_plus,
                              assembly::KernelOutput& coupling_plus_minus) override;

    [[nodiscard]] std::string name() const override;
    [[nodiscard]] int maxTemporalDerivativeOrder() const noexcept override;
    [[nodiscard]] bool isSymmetric() const noexcept override;
    [[nodiscard]] bool isMatrixOnly() const noexcept override;
    [[nodiscard]] bool isVectorOnly() const noexcept override;

private:
    enum class WrappedKind : std::uint8_t {
        Unknown = 0u,
        FormKernel,
        LinearFormKernel,
        SymbolicNonlinearFormKernel,
        NonlinearFormKernel,
    };

	    struct CompiledDispatch {
        bool ok{false};
        bool cacheable{true};
        std::string message{};

        std::uintptr_t cell{0};
        std::uintptr_t interior_face{0};

        std::uintptr_t boundary_all{0};
        std::unordered_map<int, std::uintptr_t> boundary_by_marker{};

        std::uintptr_t interface_all{0};
        std::unordered_map<int, std::uintptr_t> interface_by_marker{};
	    };

	    enum class KernelRole : std::uint8_t {
	        Form = 0u,
	        Bilinear,
	        Linear,
	        Residual,
	        Tangent,
	    };

	    struct SpecializationKey {
	        KernelRole role{KernelRole::Form};
	        IntegralDomain domain{};

	        bool has_n_qpts_minus{false};
	        bool has_n_test_dofs_minus{false};
	        bool has_n_trial_dofs_minus{false};

	        std::uint32_t n_qpts_minus{0};
	        std::uint32_t n_test_dofs_minus{0};
	        std::uint32_t n_trial_dofs_minus{0};

	        bool has_n_qpts_plus{false};
	        bool has_n_test_dofs_plus{false};
	        bool has_n_trial_dofs_plus{false};

	        std::uint32_t n_qpts_plus{0};
	        std::uint32_t n_test_dofs_plus{0};
	        std::uint32_t n_trial_dofs_plus{0};

	        friend bool operator==(const SpecializationKey& a, const SpecializationKey& b) noexcept
	        {
	            return a.role == b.role &&
	                   a.domain == b.domain &&
	                   a.has_n_qpts_minus == b.has_n_qpts_minus &&
	                   a.has_n_test_dofs_minus == b.has_n_test_dofs_minus &&
	                   a.has_n_trial_dofs_minus == b.has_n_trial_dofs_minus &&
	                   a.n_qpts_minus == b.n_qpts_minus &&
	                   a.n_test_dofs_minus == b.n_test_dofs_minus &&
	                   a.n_trial_dofs_minus == b.n_trial_dofs_minus &&
	                   a.has_n_qpts_plus == b.has_n_qpts_plus &&
	                   a.has_n_test_dofs_plus == b.has_n_test_dofs_plus &&
	                   a.has_n_trial_dofs_plus == b.has_n_trial_dofs_plus &&
	                   a.n_qpts_plus == b.n_qpts_plus &&
	                   a.n_test_dofs_plus == b.n_test_dofs_plus &&
	                   a.n_trial_dofs_plus == b.n_trial_dofs_plus;
	        }
	    };

	    struct SpecializationKeyHash {
	        std::size_t operator()(const SpecializationKey& k) const noexcept
	        {
	            std::uint64_t h = 1469598103934665603ULL;
	            const auto mix = [&](std::uint64_t v) {
	                h ^= v;
	                h *= 1099511628211ULL;
	            };
	            mix(static_cast<std::uint64_t>(k.role));
	            mix(static_cast<std::uint64_t>(k.domain));

	            mix(static_cast<std::uint64_t>(k.has_n_qpts_minus ? 1u : 0u));
	            mix(static_cast<std::uint64_t>(k.has_n_test_dofs_minus ? 1u : 0u));
	            mix(static_cast<std::uint64_t>(k.has_n_trial_dofs_minus ? 1u : 0u));
	            mix(static_cast<std::uint64_t>(k.n_qpts_minus));
	            mix(static_cast<std::uint64_t>(k.n_test_dofs_minus));
	            mix(static_cast<std::uint64_t>(k.n_trial_dofs_minus));

	            mix(static_cast<std::uint64_t>(k.has_n_qpts_plus ? 1u : 0u));
	            mix(static_cast<std::uint64_t>(k.has_n_test_dofs_plus ? 1u : 0u));
	            mix(static_cast<std::uint64_t>(k.has_n_trial_dofs_plus ? 1u : 0u));
	            mix(static_cast<std::uint64_t>(k.n_qpts_plus));
	            mix(static_cast<std::uint64_t>(k.n_test_dofs_plus));
	            mix(static_cast<std::uint64_t>(k.n_trial_dofs_plus));

	            return static_cast<std::size_t>(h);
	        }
	    };

	    void markDirty() noexcept;
	    void maybeCompile();
	    [[nodiscard]] bool canUseJIT() const noexcept;
	    void markRuntimeFailureOnce(std::string_view where, std::string_view msg) noexcept;
	    [[nodiscard]] std::shared_ptr<const CompiledDispatch> getSpecializedDispatch(
	        KernelRole role,
	        const FormIR& ir,
	        IntegralDomain domain,
	        const assembly::AssemblyContext& ctx_minus,
	        const assembly::AssemblyContext* ctx_plus);

    std::shared_ptr<assembly::AssemblyKernel> fallback_{};
    JITOptions options_{};

    std::mutex jit_mutex_{};
    std::uint64_t revision_{0};
    std::uint64_t compiled_revision_{static_cast<std::uint64_t>(-1)};
    std::uint64_t attempted_revision_{static_cast<std::uint64_t>(-1)};

    WrappedKind kind_{WrappedKind::Unknown};
    std::shared_ptr<JITCompiler> compiler_{};
    CompiledDispatch compiled_form_{};
    CompiledDispatch compiled_bilinear_{};
    CompiledDispatch compiled_linear_{};
	    CompiledDispatch compiled_residual_{};
	    CompiledDispatch compiled_tangent_{};
	    bool has_compiled_linear_{false};

	    std::unordered_map<SpecializationKey, std::shared_ptr<CompiledDispatch>, SpecializationKeyHash>
	        specialized_dispatch_{};
	    std::unordered_set<SpecializationKey, SpecializationKeyHash> attempted_specializations_{};
	    bool warned_specialization_failure_{false};

	    bool warned_unavailable_{false};
	    bool warned_validation_{false};
	    bool warned_compile_failure_{false};
	    bool runtime_failed_{false};
    bool warned_runtime_failure_{false};
};

} // namespace jit
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_JIT_JIT_KERNEL_WRAPPER_H

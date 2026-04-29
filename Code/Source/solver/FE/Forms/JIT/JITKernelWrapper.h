#ifndef SVMP_FE_FORMS_JIT_JIT_KERNEL_WRAPPER_H
#define SVMP_FE_FORMS_JIT_JIT_KERNEL_WRAPPER_H

/**
 * @file JITKernelWrapper.h
 * @brief AssemblyKernel adapter that dispatches to the production LLVM JIT backend
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
#include "Assembly/JIT/KernelArgs.h"
#include "Forms/FormExpr.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace svmp {
namespace FE {
namespace forms {

enum class IntegralDomain : std::uint8_t;
class FormIR;

namespace jit {

class JITCompiler;
struct JITCompileSpecialization;

class JITKernelWrapper final : public assembly::AssemblyKernel {
public:
    struct CellSpecializationHint {
        std::uint32_t n_qpts{0};
        std::uint32_t n_test_dofs{0};
        std::uint32_t n_trial_dofs{0};
        bool is_affine{false};  ///< P1 simplex — enables QP-constant term hoisting in JIT
    };

    struct BoundarySpecializationHint {
        int boundary_marker{-1};
        std::uint32_t n_qpts{0};
        std::uint32_t n_test_dofs{0};
        std::uint32_t n_trial_dofs{0};
    };

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
    [[nodiscard]] bool supportsCellBatch() const noexcept override { return true; }

    void computeCell(const assembly::AssemblyContext& ctx,
                     assembly::KernelOutput& output) override;
    void computeCellBatch(std::span<const assembly::AssemblyContext* const> contexts,
                          std::span<assembly::KernelOutput> outputs) override;

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
    [[nodiscard]] assembly::SemanticKernelKind semanticKernelKind() const noexcept override
    {
        return fallback_ ? fallback_->semanticKernelKind()
                         : assembly::SemanticKernelKind::SingleForm;
    }
    [[nodiscard]] int maxTemporalDerivativeOrder() const noexcept override;
    [[nodiscard]] bool hasStateIndependentMatrix() const noexcept override;
    [[nodiscard]] bool isSymmetric() const noexcept override;
    [[nodiscard]] bool isMatrixOnly() const noexcept override;
    [[nodiscard]] bool isVectorOnly() const noexcept override;

    void primeCellSpecializations(std::span<const CellSpecializationHint> hints);
    void primeBoundarySpecializations(std::span<const BoundarySpecializationHint> hints);

    [[nodiscard]] const assembly::AssemblyKernel& fallbackKernel() const noexcept { return *fallback_; }
    [[nodiscard]] std::shared_ptr<const assembly::AssemblyKernel> fallbackKernelShared() const noexcept { return fallback_; }

    /** @brief Trigger JIT compilation if not already done. */
    void ensureCompiled();

    /** @brief Check if JIT-compiled kernel is available. */
    [[nodiscard]] bool isJITReady() const noexcept;

    /** @brief Check whether a generic symbolic tangent dispatch was compiled for a domain. */
    [[nodiscard]] bool hasCompiledTangentDispatch(IntegralDomain domain, int marker = -1) const noexcept;

    /** @brief Inject a pre-compiled cell kernel address from colocated compilation.
     *  Bypasses the normal maybeCompile() path. */
    void setExternalCellAddress(std::uintptr_t addr);

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

	    void markDirty(std::string_view reason) noexcept;
	    void maybeCompile();
	    [[nodiscard]] bool canUseJIT() const noexcept;
	    void markRuntimeFailureOnce(std::string_view where, std::string_view msg) noexcept;
	    [[nodiscard]] std::shared_ptr<const CompiledDispatch> getSpecializedDispatch(
	        KernelRole role,
	        const FormIR& ir,
	        IntegralDomain domain,
	        const assembly::AssemblyContext& ctx_minus,
	        const assembly::AssemblyContext* ctx_plus);
    [[nodiscard]] std::shared_ptr<const CompiledDispatch> compileSpecializedDispatch(
        KernelRole role,
        const FormIR& ir,
        const JITCompileSpecialization& specialization,
        std::string_view trigger);

    std::shared_ptr<assembly::AssemblyKernel> fallback_{};
    JITOptions options_{};

    mutable std::mutex jit_mutex_{};
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
    std::unordered_set<SpecializationKey, SpecializationKeyHash> traced_specialization_hits_{};
    std::unordered_set<SpecializationKey, SpecializationKeyHash> traced_specialization_compiles_{};
    std::unordered_set<SpecializationKey, SpecializationKeyHash> traced_specialization_skips_{};
	    bool warned_specialization_failure_{false};

	    bool warned_unavailable_{false};
	    bool warned_validation_{false};
	    bool warned_compile_failure_{false};
	    bool runtime_failed_{false};
    bool warned_runtime_failure_{false};

    bool primed_is_affine_{false};  ///< Cached from primeCellSpecializations hint

    // NOTE: scratch_batch_sides_/outputs_ are no longer used by computeCellBatch
    // (replaced with stack-local vectors for thread safety), but kept to avoid
    // changing object layout until the next ABI-breaking change.
    std::vector<assembly::jit::KernelSideArgsV6> scratch_batch_sides_;
    std::vector<assembly::jit::KernelOutputViewV6> scratch_batch_outputs_;
};

} // namespace jit
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_JIT_JIT_KERNEL_WRAPPER_H

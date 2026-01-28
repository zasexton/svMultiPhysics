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
#include <string>

namespace svmp {
namespace FE {
namespace forms {
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

    void markDirty() noexcept;
    void maybeCompile();
    [[nodiscard]] bool canUseJIT() const noexcept;
    void markRuntimeFailureOnce(std::string_view where, std::string_view msg) noexcept;

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

#ifndef SVMP_FE_FORMS_JIT_JIT_FUNCTIONAL_KERNEL_WRAPPER_H
#define SVMP_FE_FORMS_JIT_JIT_FUNCTIONAL_KERNEL_WRAPPER_H

/**
 * @file JITFunctionalKernelWrapper.h
 * @brief FunctionalKernel adapter that can dispatch to the LLVM JIT backend
 *
 * JIT implementation:
 * - Wraps an existing interpreter FunctionalKernel and (when available) dispatches to
 *   a JIT-compiled kernel generated from a scalar FormExpr integrand.
 * - Falls back to the wrapped interpreter kernel on any compile/runtime
 *   unavailability.
 *
 * This header contains no LLVM dependencies.
 */

#include "Assembly/FunctionalAssembler.h"
#include "Forms/FormExpr.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

namespace svmp {
namespace FE {
namespace forms {
namespace jit {

class JITCompiler;

class JITFunctionalKernelWrapper final : public assembly::FunctionalKernel {
public:
    enum class Domain : std::uint8_t {
        Cell,
        BoundaryFace
    };

    JITFunctionalKernelWrapper(std::shared_ptr<assembly::FunctionalKernel> fallback,
                               FormExpr integrand,
                               Domain domain,
                               JITOptions options);

    ~JITFunctionalKernelWrapper() override = default;

    JITFunctionalKernelWrapper(const JITFunctionalKernelWrapper&) = delete;
    JITFunctionalKernelWrapper& operator=(const JITFunctionalKernelWrapper&) = delete;

    [[nodiscard]] assembly::RequiredData getRequiredData() const noexcept override;
    [[nodiscard]] std::vector<assembly::FieldRequirement> fieldRequirements() const override;

    [[nodiscard]] bool hasCell() const noexcept override;
    [[nodiscard]] bool hasBoundaryFace() const noexcept override;

    [[nodiscard]] Real evaluateCell(const assembly::AssemblyContext& ctx, LocalIndex q) override;

    [[nodiscard]] Real evaluateCellTotal(const assembly::AssemblyContext& ctx) override;

    [[nodiscard]] Real evaluateBoundaryFace(const assembly::AssemblyContext& ctx,
                                            LocalIndex q,
                                            int boundary_marker) override;

    [[nodiscard]] Real evaluateBoundaryFaceTotal(const assembly::AssemblyContext& ctx,
                                                 int boundary_marker) override;

    [[nodiscard]] Real postProcess(Real raw_value) const noexcept override
    {
        return fallback_->postProcess(raw_value);
    }

    [[nodiscard]] bool requiresSquareRoot() const noexcept override
    {
        return fallback_->requiresSquareRoot();
    }

    [[nodiscard]] bool isLinear() const noexcept override
    {
        return fallback_->isLinear();
    }

    [[nodiscard]] std::string name() const override;

private:
    void maybeCompile();
    [[nodiscard]] bool canUseJIT() const noexcept;

    std::shared_ptr<assembly::FunctionalKernel> fallback_{};
    FormExpr integrand_{};
    Domain domain_{Domain::Cell};
    JITOptions options_{};

    std::mutex jit_mutex_{};
    bool attempted_{false};
    bool compiled_{false};
    std::string compile_message_{};

    std::shared_ptr<JITCompiler> compiler_{};
    std::uintptr_t addr_{0};

    bool warned_unavailable_{false};
    bool warned_compile_failure_{false};
};

} // namespace jit
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_JIT_JIT_FUNCTIONAL_KERNEL_WRAPPER_H


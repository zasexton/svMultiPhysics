#ifndef SVMP_FE_FORMS_MONOLITHICCELLKERNEL_H
#define SVMP_FE_FORMS_MONOLITHICCELLKERNEL_H

/**
 * @file MonolithicCellKernel.h
 * @brief Explicit semantic kernel for mixed cell-domain assembly
 */

#include "Assembly/AssemblyKernel.h"
#include "Core/Types.h"
#include "Forms/FormIR.h"
#include "Forms/FormExpr.h"

#include <memory>
#include <mutex>
#include <optional>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {

namespace spaces {
class FunctionSpace;
}

namespace dofs {
class DofMap;
}

namespace forms {

namespace jit {
class JITCompiler;
}

class MonolithicCellKernel final : public assembly::AssemblyKernel {
public:
    struct BlockSpec {
        FieldId test_field{INVALID_FIELD_ID};
        FieldId trial_field{INVALID_FIELD_ID};
        bool want_matrix{false};
        bool want_vector{false};
        std::shared_ptr<assembly::AssemblyKernel> fallback_kernel{};
        std::optional<FormIR> tangent_ir{};
        std::optional<FormIR> residual_ir{};

        // Resolved at finalization time by SystemSetup
        const spaces::FunctionSpace* test_space{nullptr};
        const spaces::FunctionSpace* trial_space{nullptr};
        const dofs::DofMap* row_dof_map{nullptr};
        const dofs::DofMap* col_dof_map{nullptr};
        GlobalIndex row_dof_offset{0};
        GlobalIndex col_dof_offset{0};
    };

    MonolithicCellKernel(
        std::vector<BlockSpec> blocks,
        std::shared_ptr<jit::JITCompiler> compiler,
        JITOptions options);

    ~MonolithicCellKernel() override = default;

    MonolithicCellKernel(const MonolithicCellKernel&) = delete;
    MonolithicCellKernel& operator=(const MonolithicCellKernel&) = delete;

    [[nodiscard]] assembly::RequiredData getRequiredData() const override;
    [[nodiscard]] std::vector<assembly::FieldRequirement> fieldRequirements() const override;
    [[nodiscard]] assembly::MaterialStateSpec materialStateSpec() const noexcept override;
    [[nodiscard]] std::vector<params::Spec> parameterSpecs() const override;
    void resolveParameterSlots(
        const std::function<std::optional<std::uint32_t>(std::string_view)>& slot_of_real_param) override;
    void resolveInlinableConstitutives() override;
    [[nodiscard]] bool hasCell() const noexcept override;
    [[nodiscard]] bool supportsCellBatch() const noexcept override { return true; }

    void computeCell(const assembly::AssemblyContext& ctx,
                     assembly::KernelOutput& output) override;
    void computeCellBatch(std::span<const assembly::AssemblyContext* const> contexts,
                          std::span<assembly::KernelOutput> outputs) override;

    [[nodiscard]] std::string name() const override;
    [[nodiscard]] assembly::SemanticKernelKind semanticKernelKind() const noexcept override
    {
        return assembly::SemanticKernelKind::MonolithicCell;
    }
    [[nodiscard]] int maxTemporalDerivativeOrder() const noexcept override;
    [[nodiscard]] bool hasExplicitTimeDependency() const noexcept override;
    [[nodiscard]] bool isMatrixOnly() const noexcept override;
    [[nodiscard]] bool isVectorOnly() const noexcept override;

    [[nodiscard]] std::size_t numBlocks() const noexcept { return blocks_.size(); }
    [[nodiscard]] const BlockSpec& blockSpec(std::size_t i) const { return blocks_[i]; }
    [[nodiscard]] BlockSpec& mutableBlockSpec(std::size_t i) { return blocks_[i]; }
    [[nodiscard]] bool isResolved() const noexcept { return resolved_; }
    void setResolved() noexcept { resolved_ = true; }

    void ensureCompiled() const;
    [[nodiscard]] bool hasCompiledDispatch() const noexcept { return compiled_address_ != 0; }
    [[nodiscard]] std::uintptr_t compiledCellAddress() const noexcept { return compiled_address_; }
    [[nodiscard]] const std::string& compileMessage() const noexcept { return compile_message_; }

private:
    void invalidateCompiledDispatch();

    std::vector<BlockSpec> blocks_{};
    std::shared_ptr<jit::JITCompiler> compiler_{};
    JITOptions options_{};

    bool resolved_{false};
    mutable std::mutex compile_mutex_{};
    mutable bool compile_attempted_{false};
    mutable std::uintptr_t compiled_address_{0};
    mutable std::string compile_message_{};
};

} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_MONOLITHICCELLKERNEL_H

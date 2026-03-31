/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_FORMS_MIXED_BLOCK_KERNEL_SET_H
#define SVMP_FE_FORMS_MIXED_BLOCK_KERNEL_SET_H

/**
 * @file MixedBlockKernelSet.h
 * @brief Exact mixed-block cell kernel for multi-field systems
 *
 * MixedBlockKernelSet wraps NxM exact block kernels from a mixed multi-field
 * system into one semantic cell kernel. Each block remains mathematically
 * independent; the wrapper exists so setup and assembly can share geometry,
 * batch scheduling, resolved insertion tables, and optional text-layout
 * colocation across the block set.
 *
 * Per-block fallback kernels are still the exact kernels of record. Optional
 * colocated text layout is an acceleration layered on top of those exact
 * kernels and does not change the block semantics.
 *
 * This header contains no LLVM dependencies.
 */

#include "Assembly/AssemblyKernel.h"
#include "Core/Types.h"
#include "Forms/FormExpr.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {

namespace spaces { class FunctionSpace; }
namespace dofs { class DofMap; }

namespace forms {

class FormIR;

namespace jit {
class JITCompiler;
}

/// Exact mixed-block cell kernel for multi-field assembly.
class MixedBlockKernelSet final : public assembly::AssemblyKernel {
public:
    struct BlockSpec {
        FieldId test_field{INVALID_FIELD_ID};
        FieldId trial_field{INVALID_FIELD_ID};
        bool want_matrix{false};
        bool want_vector{false};
        std::shared_ptr<assembly::AssemblyKernel> fallback_kernel{};

        // Resolved at setup time by FESystem::buildAssemblyPlans().
        const spaces::FunctionSpace* test_space{nullptr};
        const spaces::FunctionSpace* trial_space{nullptr};
        const dofs::DofMap* row_dof_map{nullptr};
        const dofs::DofMap* col_dof_map{nullptr};
        GlobalIndex row_dof_offset{0};
        GlobalIndex col_dof_offset{0};
    };

    MixedBlockKernelSet(
        std::vector<BlockSpec> blocks,
        std::shared_ptr<jit::JITCompiler> compiler,
        JITOptions options);

    ~MixedBlockKernelSet() override = default;

    MixedBlockKernelSet(const MixedBlockKernelSet&) = delete;
    MixedBlockKernelSet& operator=(const MixedBlockKernelSet&) = delete;

    // ---- AssemblyKernel interface ----
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
        return assembly::SemanticKernelKind::MixedBlockSet;
    }
    [[nodiscard]] int maxTemporalDerivativeOrder() const noexcept override;
    [[nodiscard]] bool isSymmetric() const noexcept override { return false; }
    [[nodiscard]] bool isMatrixOnly() const noexcept override;
    [[nodiscard]] bool isVectorOnly() const noexcept override;

    // ---- Mixed-block queries ----
    [[nodiscard]] std::size_t numBlocks() const noexcept { return blocks_.size(); }
    [[nodiscard]] const BlockSpec& blockSpec(std::size_t i) const { return blocks_[i]; }
    [[nodiscard]] BlockSpec& mutableBlockSpec(std::size_t i) { return blocks_[i]; }
    [[nodiscard]] bool isResolved() const noexcept { return resolved_; }
    void setResolved() { resolved_ = true; }

    /** Optional text-layout optimization after exact per-block priming. */
    void primeColocatedTextLayout();

private:
    std::vector<BlockSpec> blocks_;
    std::shared_ptr<jit::JITCompiler> compiler_;
    JITOptions options_;

    bool resolved_{false};
};

} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_MIXED_BLOCK_KERNEL_SET_H

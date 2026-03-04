/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_FORMS_COUPLED_BLOCK_KERNEL_H
#define SVMP_FE_FORMS_COUPLED_BLOCK_KERNEL_H

/**
 * @file CoupledBlockKernel.h
 * @brief Physics-agnostic coupled block kernel for multi-field systems
 *
 * CoupledBlockKernel wraps NxM block FormIRs from a coupled multi-field
 * system (e.g., NS-VMS velocity+pressure producing VV, VP, PV, PP blocks)
 * into a single AssemblyKernel.
 *
 * When monolithic JIT compilation succeeds, all blocks are evaluated in
 * a single kernel call with shared geometry and intermediates.  When JIT
 * is unavailable or fails, per-block fallback kernels are dispatched.
 *
 * This header contains no LLVM dependencies.
 */

#include "Assembly/AssemblyKernel.h"
#include "Assembly/JIT/KernelArgs.h"
#include "Core/Types.h"
#include "Forms/FormExpr.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
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

/// Physics-agnostic coupled block kernel for multi-field systems.
///
/// Wraps NxM block FormIRs and dispatches to either:
/// (a) a JIT-compiled monolithic kernel (shared intermediates, single pass), or
/// (b) per-block fallback kernels (existing JITKernelWrappers).
class CoupledBlockKernel final : public assembly::AssemblyKernel {
public:
    struct BlockSpec {
        FieldId test_field{INVALID_FIELD_ID};
        FieldId trial_field{INVALID_FIELD_ID};
        bool want_matrix{false};
        bool want_vector{false};
        std::shared_ptr<assembly::AssemblyKernel> fallback_kernel{};

        // Resolved at finalization time by SystemAssembly
        const spaces::FunctionSpace* test_space{nullptr};
        const spaces::FunctionSpace* trial_space{nullptr};
        const dofs::DofMap* row_dof_map{nullptr};
        const dofs::DofMap* col_dof_map{nullptr};
        GlobalIndex row_dof_offset{0};
        GlobalIndex col_dof_offset{0};
    };

    CoupledBlockKernel(
        std::vector<BlockSpec> blocks,
        std::shared_ptr<jit::JITCompiler> compiler,
        JITOptions options);

    ~CoupledBlockKernel() override = default;

    CoupledBlockKernel(const CoupledBlockKernel&) = delete;
    CoupledBlockKernel& operator=(const CoupledBlockKernel&) = delete;

    // ---- AssemblyKernel interface ----
    [[nodiscard]] assembly::RequiredData getRequiredData() const override;
    [[nodiscard]] std::vector<assembly::FieldRequirement> fieldRequirements() const override;
    [[nodiscard]] bool hasCell() const noexcept override;
    [[nodiscard]] bool supportsCellBatch() const noexcept override { return true; }

    void computeCell(const assembly::AssemblyContext& ctx,
                     assembly::KernelOutput& output) override;
    void computeCellBatch(std::span<const assembly::AssemblyContext* const> contexts,
                          std::span<assembly::KernelOutput> outputs) override;

    [[nodiscard]] std::string name() const override;
    [[nodiscard]] int maxTemporalDerivativeOrder() const noexcept override;
    [[nodiscard]] bool isSymmetric() const noexcept override { return false; }
    [[nodiscard]] bool isMatrixOnly() const noexcept override;
    [[nodiscard]] bool isVectorOnly() const noexcept override;

    // ---- Coupled-specific queries ----
    [[nodiscard]] std::size_t numBlocks() const noexcept { return blocks_.size(); }
    [[nodiscard]] const BlockSpec& blockSpec(std::size_t i) const { return blocks_[i]; }
    [[nodiscard]] BlockSpec& mutableBlockSpec(std::size_t i) { return blocks_[i]; }
    [[nodiscard]] bool isMonolithicJITAvailable() const noexcept { return has_monolithic_jit_; }
    [[nodiscard]] bool isResolved() const noexcept { return resolved_; }
    void setResolved() noexcept { resolved_ = true; }

    /// Tag so assembler can identify this kernel type.
    [[nodiscard]] bool isCoupledBlockKernel() const noexcept { return true; }

private:
    void maybeCompileMonolithic();

    std::vector<BlockSpec> blocks_;
    std::shared_ptr<jit::JITCompiler> compiler_;
    JITOptions options_;

    bool resolved_{false};
    bool attempted_monolithic_{false};
    bool has_monolithic_jit_{false};
    std::uintptr_t monolithic_cell_addr_{0};
    std::uintptr_t monolithic_batch_addr_{0};

    // Scratch for monolithic cell dispatch
    mutable std::vector<assembly::jit::CoupledBlockView> scratch_block_views_;
    mutable assembly::jit::CoupledCellKernelArgsV1 scratch_coupled_args_;
};

} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_COUPLED_BLOCK_KERNEL_H

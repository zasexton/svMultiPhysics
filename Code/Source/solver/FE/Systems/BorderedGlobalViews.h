/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_BORDEREDGLOBALVIEWS_H
#define SVMP_FE_SYSTEMS_BORDEREDGLOBALVIEWS_H

/**
 * @file BorderedGlobalViews.h
 * @brief The split views of the bordered lowering of global multipliers (D54, LM2)
 *
 * Under MultiplierLowering::Bordered the global DOFs are not part of the
 * matrix: the field DOFs are [0, n_field) and the bordered global DOFs
 * follow them.  BorderedGlobalSplitView sits between assembly and the
 * backend views: the field-by-field entries go to the backend, and the
 * entries of a global row or column go to the bordered blocks of
 * FESystem::BorderedCouplingData (B, Ct, D and the residual g), which Newton
 * solves with the bordered correction.  BorderedGlobalReadView reads the
 * field DOFs from the backend (or a span) and the bordered global DOFs from
 * the system's values.
 */

#include "Assembly/GlobalSystemView.h"
#include "Systems/FESystem.h"

#include <memory>
#include <span>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

struct SystemStateView;

/**
 * @brief Field entries to the backend views, global rows and columns to the bordered blocks
 *
 * A row or column index >= n_field is a bordered global DOF (row index
 * n_field + i is bordered row i).  Field rows x global columns go to B,
 * global rows x field columns to Ct, global x global to D, and the vector
 * entries of global rows to g.  The view exposes no resolved-entry layout,
 * so assembly inserts through the plain calls.  Not thread-safe for the
 * bordered entries (global fields are assembled without coloring).
 */
class BorderedGlobalSplitView final : public assembly::GlobalSystemView {
public:
    BorderedGlobalSplitView(assembly::GlobalSystemView* matrix,
                            assembly::GlobalSystemView* vector,
                            FESystem::BorderedCouplingData& bordered,
                            GlobalIndex n_field);

    void addMatrixEntries(std::span<const GlobalIndex> dofs,
                          std::span<const Real> local_matrix,
                          assembly::AddMode mode = assembly::AddMode::Add) override;
    void addMatrixEntries(std::span<const GlobalIndex> row_dofs,
                          std::span<const GlobalIndex> col_dofs,
                          std::span<const Real> local_matrix,
                          assembly::AddMode mode = assembly::AddMode::Add) override;
    void addMatrixEntry(GlobalIndex row, GlobalIndex col, Real value,
                        assembly::AddMode mode = assembly::AddMode::Add) override;
    void setDiagonal(std::span<const GlobalIndex> dofs, std::span<const Real> values) override;
    void setDiagonal(GlobalIndex dof, Real value) override;
    void zeroRows(std::span<const GlobalIndex> rows, bool set_diagonal = true) override;

    [[nodiscard]] const void* matrixLayoutHandle() const noexcept override { return nullptr; }
    [[nodiscard]] const void* vectorLayoutHandle() const noexcept override { return nullptr; }

    void addVectorEntries(std::span<const GlobalIndex> dofs,
                          std::span<const Real> local_vector,
                          assembly::AddMode mode = assembly::AddMode::Add) override;
    void addVectorEntry(GlobalIndex dof, Real value,
                        assembly::AddMode mode = assembly::AddMode::Add) override;
    void setVectorEntries(std::span<const GlobalIndex> dofs, std::span<const Real> values) override;
    void zeroVectorEntries(std::span<const GlobalIndex> dofs) override;
    [[nodiscard]] Real getVectorEntry(GlobalIndex dof) const override;

    void beginAssemblyPhase() override;
    void endAssemblyPhase() override;
    void finalizeAssembly() override;
    [[nodiscard]] assembly::AssemblyPhase getPhase() const noexcept override;
    [[nodiscard]] bool hasMatrix() const noexcept override { return matrix_ != nullptr; }
    [[nodiscard]] bool hasVector() const noexcept override { return vector_ != nullptr; }
    [[nodiscard]] GlobalIndex numRows() const noexcept override { return n_field_ + n_bordered_; }
    [[nodiscard]] GlobalIndex numCols() const noexcept override { return n_field_ + n_bordered_; }
    [[nodiscard]] bool isDistributed() const noexcept override;
    [[nodiscard]] std::string backendName() const override { return "BorderedGlobalSplitView"; }
    void zero() override;
    [[nodiscard]] Real getMatrixEntry(GlobalIndex row, GlobalIndex col) const override;

private:
    [[nodiscard]] std::size_t borderedIndex(GlobalIndex dof) const;
    static void apply(Real& target, Real value, assembly::AddMode mode);

    assembly::GlobalSystemView* matrix_{nullptr};
    assembly::GlobalSystemView* vector_{nullptr};
    FESystem::BorderedCouplingData* bordered_{nullptr};
    GlobalIndex n_field_{0};
    GlobalIndex n_bordered_{0};
    std::vector<GlobalIndex> field_rows_{};
    std::vector<GlobalIndex> field_cols_{};
    std::vector<Real> field_values_{};
};

/**
 * @brief Field DOFs from the backend view (or a span), bordered global DOFs from the system
 */
class BorderedGlobalReadView final : public assembly::GlobalSystemView {
public:
    BorderedGlobalReadView(std::unique_ptr<assembly::GlobalSystemView> field_view,
                           std::span<const Real> field_values,
                           std::span<const Real> global_values,
                           GlobalIndex n_field);

    void addMatrixEntries(std::span<const GlobalIndex>, std::span<const Real>, assembly::AddMode) override;
    void addMatrixEntries(std::span<const GlobalIndex>, std::span<const GlobalIndex>, std::span<const Real>,
                          assembly::AddMode) override;
    void addMatrixEntry(GlobalIndex, GlobalIndex, Real, assembly::AddMode) override;
    void setDiagonal(std::span<const GlobalIndex>, std::span<const Real>) override;
    void setDiagonal(GlobalIndex, Real) override;
    void zeroRows(std::span<const GlobalIndex>, bool) override;
    void addVectorEntries(std::span<const GlobalIndex>, std::span<const Real>, assembly::AddMode) override;
    void addVectorEntry(GlobalIndex, Real, assembly::AddMode) override;
    void setVectorEntries(std::span<const GlobalIndex>, std::span<const Real>) override;
    void zeroVectorEntries(std::span<const GlobalIndex>) override;

    [[nodiscard]] const void* vectorLayoutHandle() const noexcept override { return nullptr; }
    [[nodiscard]] Real getVectorEntry(GlobalIndex dof) const override;
    void getVectorEntries(std::span<const GlobalIndex> dofs, std::span<Real> out) const override;

    void beginAssemblyPhase() override {}
    void endAssemblyPhase() override {}
    void finalizeAssembly() override {}
    [[nodiscard]] assembly::AssemblyPhase getPhase() const noexcept override
    {
        return assembly::AssemblyPhase::Finalized;
    }
    [[nodiscard]] bool hasMatrix() const noexcept override { return false; }
    [[nodiscard]] bool hasVector() const noexcept override { return true; }
    [[nodiscard]] GlobalIndex numRows() const noexcept override
    {
        return n_field_ + static_cast<GlobalIndex>(global_values_.size());
    }
    [[nodiscard]] GlobalIndex numCols() const noexcept override { return 1; }
    [[nodiscard]] std::string backendName() const override { return "BorderedGlobalReadView"; }
    void zero() override;

private:
    std::unique_ptr<assembly::GlobalSystemView> field_view_{};
    std::span<const Real> field_values_{};
    std::span<const Real> global_values_{};
    GlobalIndex n_field_{0};
};

/**
 * @brief The read view of a state vector for assembly
 *
 * Without bordered global fields: @p vector's view (ghosted when @p ghosted),
 * or null when there is no vector (assembly then reads @p values).  With
 * them: a BorderedGlobalReadView over that view or over @p values.
 */
[[nodiscard]] std::unique_ptr<assembly::GlobalSystemView>
makeStateReadView(const FESystem& system,
                  const backends::GenericVector* vector,
                  std::span<const Real> values,
                  bool ghosted);

/**
 * @brief Clears the bordered blocks of the global rows and columns before an assembly
 *
 * A matrix assembly resets B, Ct and D and keeps g unless the vector is
 * assembled too; a vector assembly resets g.  The layout is one block per
 * global field, every bordered unknown algebraic, every rank contributing.
 */
void prepareBorderedGlobalCoupling(const FESystem& system,
                                   FESystem::BorderedCouplingData& bordered,
                                   bool want_matrix,
                                   bool want_vector);

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_BORDEREDGLOBALVIEWS_H

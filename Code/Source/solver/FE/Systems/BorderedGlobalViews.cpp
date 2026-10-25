/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/BorderedGlobalViews.h"

#include "Backends/Interfaces/GenericVector.h"
#include "Core/FEException.h"
#include "Systems/SystemsExceptions.h"

#include <algorithm>
#include <array>

namespace svmp {
namespace FE {
namespace systems {

// ---------------------------------------------------------------------------
// BorderedGlobalSplitView
// ---------------------------------------------------------------------------

BorderedGlobalSplitView::BorderedGlobalSplitView(assembly::GlobalSystemView* matrix,
                                                 assembly::GlobalSystemView* vector,
                                                 FESystem::BorderedCouplingData& bordered,
                                                 GlobalIndex n_field)
    : matrix_(matrix)
    , vector_(vector)
    , bordered_(&bordered)
    , n_field_(n_field)
    , n_bordered_(static_cast<GlobalIndex>(bordered.n_aux))
{
    FE_THROW_IF(static_cast<GlobalIndex>(bordered.n_field_dofs) != n_field, InvalidStateException,
                "BorderedGlobalSplitView: the bordered blocks have " + std::to_string(bordered.n_field_dofs) +
                    " field DOFs, the system " + std::to_string(n_field));
}

std::size_t BorderedGlobalSplitView::borderedIndex(GlobalIndex dof) const
{
    const auto index = dof - n_field_;
    FE_THROW_IF(index < 0 || index >= n_bordered_, InvalidArgumentException,
                "BorderedGlobalSplitView: DOF " + std::to_string(dof) + " is neither a field DOF nor one of the " +
                    std::to_string(n_bordered_) + " bordered global DOFs");
    return static_cast<std::size_t>(index);
}

void BorderedGlobalSplitView::apply(Real& target, Real value, assembly::AddMode mode)
{
    switch (mode) {
        case assembly::AddMode::Add:
            target += value;
            return;
        case assembly::AddMode::Insert:
            target = value;
            return;
        case assembly::AddMode::Max:
            target = std::max(target, value);
            return;
        case assembly::AddMode::Min:
            target = std::min(target, value);
            return;
    }
}

void BorderedGlobalSplitView::addMatrixEntries(std::span<const GlobalIndex> dofs,
                                               std::span<const Real> local_matrix,
                                               assembly::AddMode mode)
{
    addMatrixEntries(dofs, dofs, local_matrix, mode);
}

void BorderedGlobalSplitView::addMatrixEntries(std::span<const GlobalIndex> row_dofs,
                                               std::span<const GlobalIndex> col_dofs,
                                               std::span<const Real> local_matrix,
                                               assembly::AddMode mode)
{
    const auto n_rows = row_dofs.size();
    const auto n_cols = col_dofs.size();
    FE_THROW_IF(local_matrix.size() != n_rows * n_cols, InvalidArgumentException,
                "BorderedGlobalSplitView::addMatrixEntries: size mismatch");
    const auto is_field = [this](GlobalIndex dof) { return dof < n_field_; };
    const bool field_rows_only = std::all_of(row_dofs.begin(), row_dofs.end(), is_field);
    const bool field_cols_only = std::all_of(col_dofs.begin(), col_dofs.end(), is_field);
    if (field_rows_only && field_cols_only) {
        if (matrix_ != nullptr) {
            matrix_->addMatrixEntries(row_dofs, col_dofs, local_matrix, mode);
        }
        return;
    }

    auto& bc = *bordered_;
    const auto nf = bc.n_field_dofs;
    const auto na = static_cast<std::size_t>(bc.n_aux);
    field_rows_.clear();
    field_cols_.clear();
    field_values_.clear();
    std::vector<std::size_t> field_col_positions;
    for (std::size_t j = 0; j < n_cols; ++j) {
        if (is_field(col_dofs[j])) {
            field_cols_.push_back(col_dofs[j]);
            field_col_positions.push_back(j);
        }
    }
    for (std::size_t i = 0; i < n_rows; ++i) {
        const auto row = row_dofs[i];
        const auto* values = local_matrix.data() + i * n_cols;
        if (is_field(row)) {
            if (!field_cols_.empty()) {
                field_rows_.push_back(row);
                for (const auto j : field_col_positions) {
                    field_values_.push_back(values[j]);
                }
            }
            for (std::size_t j = 0; j < n_cols; ++j) {
                if (!is_field(col_dofs[j])) {
                    // Field row x global column: B, column-major.
                    apply(bc.B[static_cast<std::size_t>(row) + nf * borderedIndex(col_dofs[j])], values[j], mode);
                }
            }
            continue;
        }
        const auto bordered_row = borderedIndex(row);
        for (std::size_t j = 0; j < n_cols; ++j) {
            const auto col = col_dofs[j];
            if (is_field(col)) {
                // Global row x field column: Ct, row-major.
                apply(bc.Ct[bordered_row * nf + static_cast<std::size_t>(col)], values[j], mode);
            } else {
                apply(bc.D[bordered_row * na + borderedIndex(col)], values[j], mode);
            }
        }
    }
    if (matrix_ != nullptr && !field_rows_.empty()) {
        matrix_->addMatrixEntries(field_rows_, field_cols_, field_values_, mode);
    }
}

void BorderedGlobalSplitView::addMatrixEntry(GlobalIndex row, GlobalIndex col, Real value, assembly::AddMode mode)
{
    const std::array<GlobalIndex, 1> rows{row};
    const std::array<GlobalIndex, 1> cols{col};
    const std::array<Real, 1> values{value};
    addMatrixEntries(rows, cols, values, mode);
}

void BorderedGlobalSplitView::setDiagonal(std::span<const GlobalIndex> dofs, std::span<const Real> values)
{
    FE_THROW_IF(dofs.size() != values.size(), InvalidArgumentException,
                "BorderedGlobalSplitView::setDiagonal: size mismatch");
    for (std::size_t i = 0; i < dofs.size(); ++i) {
        setDiagonal(dofs[i], values[i]);
    }
}

void BorderedGlobalSplitView::setDiagonal(GlobalIndex dof, Real value)
{
    FE_THROW_IF(dof >= n_field_, InvalidArgumentException,
                "BorderedGlobalSplitView::setDiagonal: a bordered global DOF takes no constraint row");
    if (matrix_ != nullptr) {
        matrix_->setDiagonal(dof, value);
    }
}

void BorderedGlobalSplitView::zeroRows(std::span<const GlobalIndex> rows, bool set_diagonal)
{
    for (const auto row : rows) {
        FE_THROW_IF(row >= n_field_, InvalidArgumentException,
                    "BorderedGlobalSplitView::zeroRows: a bordered global DOF takes no constraint row");
    }
    if (matrix_ != nullptr) {
        matrix_->zeroRows(rows, set_diagonal);
    }
}

void BorderedGlobalSplitView::addVectorEntries(std::span<const GlobalIndex> dofs,
                                               std::span<const Real> local_vector,
                                               assembly::AddMode mode)
{
    FE_THROW_IF(dofs.size() != local_vector.size(), InvalidArgumentException,
                "BorderedGlobalSplitView::addVectorEntries: size mismatch");
    const bool field_only = std::all_of(dofs.begin(), dofs.end(), [this](GlobalIndex dof) { return dof < n_field_; });
    if (field_only) {
        if (vector_ != nullptr) {
            vector_->addVectorEntries(dofs, local_vector, mode);
        }
        return;
    }
    for (std::size_t i = 0; i < dofs.size(); ++i) {
        addVectorEntry(dofs[i], local_vector[i], mode);
    }
}

void BorderedGlobalSplitView::addVectorEntry(GlobalIndex dof, Real value, assembly::AddMode mode)
{
    if (dof < n_field_) {
        if (vector_ != nullptr) {
            vector_->addVectorEntry(dof, value, mode);
        }
        return;
    }
    apply(bordered_->g[borderedIndex(dof)], value, mode);
}

void BorderedGlobalSplitView::setVectorEntries(std::span<const GlobalIndex> dofs, std::span<const Real> values)
{
    FE_THROW_IF(dofs.size() != values.size(), InvalidArgumentException,
                "BorderedGlobalSplitView::setVectorEntries: size mismatch");
    for (std::size_t i = 0; i < dofs.size(); ++i) {
        addVectorEntry(dofs[i], values[i], assembly::AddMode::Insert);
    }
}

void BorderedGlobalSplitView::zeroVectorEntries(std::span<const GlobalIndex> dofs)
{
    for (const auto dof : dofs) {
        if (dof < n_field_) {
            if (vector_ != nullptr) {
                const std::array<GlobalIndex, 1> one{dof};
                vector_->zeroVectorEntries(one);
            }
        } else {
            bordered_->g[borderedIndex(dof)] = Real(0.0);
        }
    }
}

Real BorderedGlobalSplitView::getVectorEntry(GlobalIndex dof) const
{
    if (dof < n_field_) {
        return vector_ != nullptr ? vector_->getVectorEntry(dof) : Real(0.0);
    }
    return bordered_->g[borderedIndex(dof)];
}

void BorderedGlobalSplitView::beginAssemblyPhase()
{
    if (matrix_ != nullptr) {
        matrix_->beginAssemblyPhase();
    }
    if (vector_ != nullptr && vector_ != matrix_) {
        vector_->beginAssemblyPhase();
    }
}

void BorderedGlobalSplitView::endAssemblyPhase()
{
    if (matrix_ != nullptr) {
        matrix_->endAssemblyPhase();
    }
    if (vector_ != nullptr && vector_ != matrix_) {
        vector_->endAssemblyPhase();
    }
}

void BorderedGlobalSplitView::finalizeAssembly()
{
    if (matrix_ != nullptr) {
        matrix_->finalizeAssembly();
    }
    if (vector_ != nullptr && vector_ != matrix_) {
        vector_->finalizeAssembly();
    }
}

assembly::AssemblyPhase BorderedGlobalSplitView::getPhase() const noexcept
{
    if (matrix_ != nullptr) {
        return matrix_->getPhase();
    }
    return vector_ != nullptr ? vector_->getPhase() : assembly::AssemblyPhase::NotStarted;
}

bool BorderedGlobalSplitView::isDistributed() const noexcept
{
    return (matrix_ != nullptr && matrix_->isDistributed()) || (vector_ != nullptr && vector_->isDistributed());
}

void BorderedGlobalSplitView::zero()
{
    if (matrix_ != nullptr) {
        matrix_->zero();
    }
    if (vector_ != nullptr && vector_ != matrix_) {
        vector_->zero();
    }
}

Real BorderedGlobalSplitView::getMatrixEntry(GlobalIndex row, GlobalIndex col) const
{
    const auto& bc = *bordered_;
    const auto nf = bc.n_field_dofs;
    const auto na = static_cast<std::size_t>(bc.n_aux);
    if (row < n_field_ && col < n_field_) {
        return matrix_ != nullptr ? matrix_->getMatrixEntry(row, col) : Real(0.0);
    }
    if (row < n_field_) {
        return bc.B[static_cast<std::size_t>(row) + nf * borderedIndex(col)];
    }
    if (col < n_field_) {
        return bc.Ct[borderedIndex(row) * nf + static_cast<std::size_t>(col)];
    }
    return bc.D[borderedIndex(row) * na + borderedIndex(col)];
}

// ---------------------------------------------------------------------------
// BorderedGlobalReadView
// ---------------------------------------------------------------------------

BorderedGlobalReadView::BorderedGlobalReadView(std::unique_ptr<assembly::GlobalSystemView> field_view,
                                               std::span<const Real> field_values,
                                               std::span<const Real> global_values,
                                               GlobalIndex n_field)
    : field_view_(std::move(field_view))
    , field_values_(field_values)
    , global_values_(global_values)
    , n_field_(n_field)
{
}

namespace {

[[noreturn]] void refuseWrite()
{
    FE_THROW(InvalidStateException, "BorderedGlobalReadView: a state read view takes no entries");
}

} // namespace

void BorderedGlobalReadView::addMatrixEntries(std::span<const GlobalIndex>, std::span<const Real>, assembly::AddMode)
{
    refuseWrite();
}
void BorderedGlobalReadView::addMatrixEntries(std::span<const GlobalIndex>, std::span<const GlobalIndex>,
                                              std::span<const Real>, assembly::AddMode)
{
    refuseWrite();
}
void BorderedGlobalReadView::addMatrixEntry(GlobalIndex, GlobalIndex, Real, assembly::AddMode) { refuseWrite(); }
void BorderedGlobalReadView::setDiagonal(std::span<const GlobalIndex>, std::span<const Real>) { refuseWrite(); }
void BorderedGlobalReadView::setDiagonal(GlobalIndex, Real) { refuseWrite(); }
void BorderedGlobalReadView::zeroRows(std::span<const GlobalIndex>, bool) { refuseWrite(); }
void BorderedGlobalReadView::addVectorEntries(std::span<const GlobalIndex>, std::span<const Real>, assembly::AddMode)
{
    refuseWrite();
}
void BorderedGlobalReadView::addVectorEntry(GlobalIndex, Real, assembly::AddMode) { refuseWrite(); }
void BorderedGlobalReadView::setVectorEntries(std::span<const GlobalIndex>, std::span<const Real>) { refuseWrite(); }
void BorderedGlobalReadView::zeroVectorEntries(std::span<const GlobalIndex>) { refuseWrite(); }
void BorderedGlobalReadView::zero() { refuseWrite(); }

Real BorderedGlobalReadView::getVectorEntry(GlobalIndex dof) const
{
    if (dof >= n_field_) {
        const auto index = static_cast<std::size_t>(dof - n_field_);
        FE_THROW_IF(index >= global_values_.size(), InvalidArgumentException,
                    "BorderedGlobalReadView: DOF " + std::to_string(dof) + " is past the bordered global DOFs");
        return global_values_[index];
    }
    if (field_view_ != nullptr) {
        return field_view_->getVectorEntry(dof);
    }
    FE_THROW_IF(dof < 0 || static_cast<std::size_t>(dof) >= field_values_.size(), InvalidArgumentException,
                "BorderedGlobalReadView: the state has no value for DOF " + std::to_string(dof));
    return field_values_[static_cast<std::size_t>(dof)];
}

void BorderedGlobalReadView::getVectorEntries(std::span<const GlobalIndex> dofs, std::span<Real> out) const
{
    FE_THROW_IF(dofs.size() != out.size(), InvalidArgumentException,
                "BorderedGlobalReadView::getVectorEntries: size mismatch");
    for (std::size_t i = 0; i < dofs.size(); ++i) {
        out[i] = getVectorEntry(dofs[i]);
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

std::unique_ptr<assembly::GlobalSystemView> makeStateReadView(const FESystem& system,
                                                              const backends::GenericVector* vector,
                                                              std::span<const Real> values,
                                                              bool ghosted)
{
    std::unique_ptr<assembly::GlobalSystemView> view;
    if (vector != nullptr) {
        auto* mutable_vector = const_cast<backends::GenericVector*>(vector);
        view = ghosted ? mutable_vector->createGhostedReadView() : mutable_vector->createAssemblyView();
    }
    if (!system.hasBorderedGlobalFields()) {
        return view;
    }
    return std::make_unique<BorderedGlobalReadView>(std::move(view),
                                                    values,
                                                    system.borderedGlobalValues(),
                                                    system.dofHandler().getNumDofs());
}

void prepareBorderedGlobalCoupling(const FESystem& system,
                                   FESystem::BorderedCouplingData& bordered,
                                   bool want_matrix,
                                   bool want_vector)
{
    const auto n_aux = static_cast<int>(system.numBorderedGlobalDofs());
    const auto n_field = static_cast<std::size_t>(system.dofHandler().getNumDofs());
    const bool shaped = bordered.active && bordered.n_aux == n_aux && bordered.n_field_dofs == n_field;
    if (want_matrix || !shaped) {
        std::vector<Real> kept_residual;
        if (!want_vector && shaped) {
            kept_residual = bordered.g;
        }
        bordered.resize(n_aux, n_field);
        if (!kept_residual.empty()) {
            bordered.g = std::move(kept_residual);
        }
        bordered.aux_blocks.clear();
        for (FieldId field = 0; static_cast<std::size_t>(field) < system.registeredFieldCount(); ++field) {
            const auto& rec = system.fieldRecord(field);
            if (rec.scope != FieldScope::Global) {
                continue;
            }
            bordered.aux_blocks.push_back(
                {rec.name, static_cast<int>(system.fieldDofHandler(field).getNumDofs()), field});
        }
        std::fill(bordered.aux_variable_kinds.begin(), bordered.aux_variable_kinds.end(),
                  AuxiliaryVariableKind::Algebraic);
        // Every rank adds its cells' entries; the reduction sums them.
        bordered.aux_self_terms_replicated = false;
    }
    if (want_vector) {
        std::fill(bordered.g.begin(), bordered.g.end(), Real(0.0));
    }
}

} // namespace systems
} // namespace FE
} // namespace svmp

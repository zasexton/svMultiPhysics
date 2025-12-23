/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "SparsityPreallocation.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>

namespace svmp {
namespace FE {
namespace sparsity {

// ============================================================================
// SparsityPreallocation implementation
// ============================================================================

SparsityPreallocation::SparsityPreallocation(const SparsityPattern& pattern) {
    FE_THROW_IF(!pattern.isFinalized(), InvalidArgumentException,
                "Pattern must be finalized for preallocation");

    n_rows_ = pattern.numRows();
    n_cols_ = pattern.numCols();
    nnz_per_row_.resize(static_cast<std::size_t>(n_rows_));

    for (GlobalIndex row = 0; row < n_rows_; ++row) {
        nnz_per_row_[static_cast<std::size_t>(row)] = pattern.getRowNnz(row);
    }

    computeStatistics();
}

SparsityPreallocation::SparsityPreallocation(GlobalIndex n_rows, GlobalIndex nnz_per_row)
    : n_rows_(n_rows),
      n_cols_(n_rows),  // Assume square
      nnz_per_row_(static_cast<std::size_t>(n_rows), nnz_per_row)
{
    computeStatistics();
}

SparsityPreallocation::SparsityPreallocation(std::vector<GlobalIndex> nnz_per_row)
    : n_rows_(static_cast<GlobalIndex>(nnz_per_row.size())),
      n_cols_(n_rows_),  // Assume square
      nnz_per_row_(std::move(nnz_per_row))
{
    computeStatistics();
}

GlobalIndex SparsityPreallocation::getRowNnz(GlobalIndex row) const {
    FE_CHECK_ARG(row >= 0 && static_cast<std::size_t>(row) < nnz_per_row_.size(),
                 "Row index out of range");
    return nnz_per_row_[static_cast<std::size_t>(row)];
}

SparsityPreallocation& SparsityPreallocation::applySafetyFactor(double factor) {
    FE_CHECK_ARG(factor >= 1.0, "Safety factor must be >= 1.0");

    for (auto& nnz : nnz_per_row_) {
        nnz = static_cast<GlobalIndex>(std::ceil(static_cast<double>(nnz) * factor));
    }
    computeStatistics();
    return *this;
}

SparsityPreallocation& SparsityPreallocation::addExtraPerRow(GlobalIndex extra_per_row) {
    for (auto& nnz : nnz_per_row_) {
        nnz += extra_per_row;
    }
    computeStatistics();
    return *this;
}

SparsityPreallocation& SparsityPreallocation::clampToMax(GlobalIndex max_nnz) {
    for (auto& nnz : nnz_per_row_) {
        nnz = std::min(nnz, max_nnz);
    }
    computeStatistics();
    return *this;
}

SparsityPreallocation& SparsityPreallocation::ensureMinimum(GlobalIndex min_nnz) {
    for (auto& nnz : nnz_per_row_) {
        nnz = std::max(nnz, min_nnz);
    }
    computeStatistics();
    return *this;
}

SparsityPreallocation SparsityPreallocation::combine(
    const SparsityPreallocation& other) const
{
    FE_CHECK_ARG(n_rows_ == other.n_rows_,
                 "Cannot combine preallocations with different row counts");

    std::vector<GlobalIndex> combined(nnz_per_row_.size());
    for (std::size_t i = 0; i < nnz_per_row_.size(); ++i) {
        combined[i] = std::max(nnz_per_row_[i], other.nnz_per_row_[i]);
    }
    return SparsityPreallocation(std::move(combined));
}

SparsityPreallocation SparsityPreallocation::add(
    const SparsityPreallocation& other) const
{
    FE_CHECK_ARG(n_rows_ == other.n_rows_,
                 "Cannot add preallocations with different row counts");

    std::vector<GlobalIndex> sum(nnz_per_row_.size());
    for (std::size_t i = 0; i < nnz_per_row_.size(); ++i) {
        sum[i] = nnz_per_row_[i] + other.nnz_per_row_[i];
    }
    return SparsityPreallocation(std::move(sum));
}

bool SparsityPreallocation::validate() const noexcept {
    if (n_rows_ < 0 || n_cols_ < 0) return false;
    if (static_cast<GlobalIndex>(nnz_per_row_.size()) != n_rows_) return false;

    GlobalIndex computed_total = 0;
    GlobalIndex computed_max = 0;
    GlobalIndex computed_min = std::numeric_limits<GlobalIndex>::max();

    for (GlobalIndex nnz : nnz_per_row_) {
        if (nnz < 0) return false;
        computed_total += nnz;
        computed_max = std::max(computed_max, nnz);
        computed_min = std::min(computed_min, nnz);
    }

    if (nnz_per_row_.empty()) {
        computed_min = 0;
    }

    return computed_total == total_nnz_ &&
           computed_max == max_row_nnz_ &&
           computed_min == min_row_nnz_;
}

std::size_t SparsityPreallocation::memoryUsageBytes() const noexcept {
    return sizeof(*this) + nnz_per_row_.capacity() * sizeof(GlobalIndex);
}

void SparsityPreallocation::computeStatistics() {
    total_nnz_ = 0;
    max_row_nnz_ = 0;
    min_row_nnz_ = nnz_per_row_.empty() ? 0 : std::numeric_limits<GlobalIndex>::max();

    for (GlobalIndex nnz : nnz_per_row_) {
        total_nnz_ += nnz;
        max_row_nnz_ = std::max(max_row_nnz_, nnz);
        min_row_nnz_ = std::min(min_row_nnz_, nnz);
    }

    if (nnz_per_row_.empty()) {
        min_row_nnz_ = 0;
    }
}

// ============================================================================
// DistributedSparsityPreallocation implementation
// ============================================================================

DistributedSparsityPreallocation::DistributedSparsityPreallocation(
    const DistributedSparsityPattern& pattern)
{
    FE_THROW_IF(!pattern.isFinalized(), InvalidArgumentException,
                "Pattern must be finalized for preallocation");

    n_owned_rows_ = pattern.numOwnedRows();
    n_owned_cols_ = pattern.numOwnedCols();
    n_ghost_cols_ = pattern.numGhostCols();

    diag_nnz_per_row_.resize(static_cast<std::size_t>(n_owned_rows_));
    offdiag_nnz_per_row_.resize(static_cast<std::size_t>(n_owned_rows_));

    for (GlobalIndex row = 0; row < n_owned_rows_; ++row) {
        diag_nnz_per_row_[static_cast<std::size_t>(row)] = pattern.getRowDiagNnz(row);
        offdiag_nnz_per_row_[static_cast<std::size_t>(row)] = pattern.getRowOffdiagNnz(row);
    }

    computeStatistics();
}

DistributedSparsityPreallocation::DistributedSparsityPreallocation(
    GlobalIndex n_owned_rows,
    GlobalIndex diag_nnz_per_row,
    GlobalIndex offdiag_nnz_per_row)
    : n_owned_rows_(n_owned_rows),
      n_owned_cols_(n_owned_rows),  // Assume square
      n_ghost_cols_(0),
      diag_nnz_per_row_(static_cast<std::size_t>(n_owned_rows), diag_nnz_per_row),
      offdiag_nnz_per_row_(static_cast<std::size_t>(n_owned_rows), offdiag_nnz_per_row)
{
    computeStatistics();
}

DistributedSparsityPreallocation::DistributedSparsityPreallocation(
    std::vector<GlobalIndex> diag_nnz_per_row,
    std::vector<GlobalIndex> offdiag_nnz_per_row)
    : n_owned_rows_(static_cast<GlobalIndex>(diag_nnz_per_row.size())),
      n_owned_cols_(n_owned_rows_),
      n_ghost_cols_(0),
      diag_nnz_per_row_(std::move(diag_nnz_per_row)),
      offdiag_nnz_per_row_(std::move(offdiag_nnz_per_row))
{
    FE_CHECK_ARG(diag_nnz_per_row_.size() == offdiag_nnz_per_row_.size(),
                 "Diag and offdiag arrays must have same size");
    computeStatistics();
}

GlobalIndex DistributedSparsityPreallocation::getDiagRowNnz(GlobalIndex local_row) const {
    FE_CHECK_ARG(local_row >= 0 &&
                 static_cast<std::size_t>(local_row) < diag_nnz_per_row_.size(),
                 "Local row index out of range");
    return diag_nnz_per_row_[static_cast<std::size_t>(local_row)];
}

GlobalIndex DistributedSparsityPreallocation::getOffdiagRowNnz(GlobalIndex local_row) const {
    FE_CHECK_ARG(local_row >= 0 &&
                 static_cast<std::size_t>(local_row) < offdiag_nnz_per_row_.size(),
                 "Local row index out of range");
    return offdiag_nnz_per_row_[static_cast<std::size_t>(local_row)];
}

GlobalIndex DistributedSparsityPreallocation::getRowNnz(GlobalIndex local_row) const {
    return getDiagRowNnz(local_row) + getOffdiagRowNnz(local_row);
}

SparsityPreallocation DistributedSparsityPreallocation::getCombinedPreallocation() const {
    std::vector<GlobalIndex> combined(diag_nnz_per_row_.size());
    for (std::size_t i = 0; i < diag_nnz_per_row_.size(); ++i) {
        combined[i] = diag_nnz_per_row_[i] + offdiag_nnz_per_row_[i];
    }
    return SparsityPreallocation(std::move(combined));
}

SparsityPreallocation DistributedSparsityPreallocation::getDiagPreallocation() const {
    return SparsityPreallocation(std::vector<GlobalIndex>(diag_nnz_per_row_));
}

SparsityPreallocation DistributedSparsityPreallocation::getOffdiagPreallocation() const {
    return SparsityPreallocation(std::vector<GlobalIndex>(offdiag_nnz_per_row_));
}

DistributedSparsityPreallocation& DistributedSparsityPreallocation::applySafetyFactor(
    double factor)
{
    return applySafetyFactors(factor, factor);
}

DistributedSparsityPreallocation& DistributedSparsityPreallocation::applySafetyFactors(
    double diag_factor, double offdiag_factor)
{
    FE_CHECK_ARG(diag_factor >= 1.0 && offdiag_factor >= 1.0,
                 "Safety factors must be >= 1.0");

    for (auto& nnz : diag_nnz_per_row_) {
        nnz = static_cast<GlobalIndex>(std::ceil(static_cast<double>(nnz) * diag_factor));
    }
    for (auto& nnz : offdiag_nnz_per_row_) {
        nnz = static_cast<GlobalIndex>(std::ceil(static_cast<double>(nnz) * offdiag_factor));
    }
    computeStatistics();
    return *this;
}

bool DistributedSparsityPreallocation::validate() const noexcept {
    if (n_owned_rows_ < 0) return false;
    if (static_cast<GlobalIndex>(diag_nnz_per_row_.size()) != n_owned_rows_) return false;
    if (static_cast<GlobalIndex>(offdiag_nnz_per_row_.size()) != n_owned_rows_) return false;

    GlobalIndex computed_diag = 0;
    GlobalIndex computed_offdiag = 0;
    GlobalIndex max_diag = 0;
    GlobalIndex max_offdiag = 0;

    for (std::size_t i = 0; i < diag_nnz_per_row_.size(); ++i) {
        if (diag_nnz_per_row_[i] < 0 || offdiag_nnz_per_row_[i] < 0) {
            return false;
        }
        computed_diag += diag_nnz_per_row_[i];
        computed_offdiag += offdiag_nnz_per_row_[i];
        max_diag = std::max(max_diag, diag_nnz_per_row_[i]);
        max_offdiag = std::max(max_offdiag, offdiag_nnz_per_row_[i]);
    }

    return computed_diag == total_diag_nnz_ &&
           computed_offdiag == total_offdiag_nnz_ &&
           max_diag == max_diag_row_nnz_ &&
           max_offdiag == max_offdiag_row_nnz_;
}

std::size_t DistributedSparsityPreallocation::memoryUsageBytes() const noexcept {
    return sizeof(*this) +
           diag_nnz_per_row_.capacity() * sizeof(GlobalIndex) +
           offdiag_nnz_per_row_.capacity() * sizeof(GlobalIndex);
}

void DistributedSparsityPreallocation::computeStatistics() {
    total_diag_nnz_ = 0;
    total_offdiag_nnz_ = 0;
    max_diag_row_nnz_ = 0;
    max_offdiag_row_nnz_ = 0;

    for (std::size_t i = 0; i < diag_nnz_per_row_.size(); ++i) {
        total_diag_nnz_ += diag_nnz_per_row_[i];
        total_offdiag_nnz_ += offdiag_nnz_per_row_[i];
        max_diag_row_nnz_ = std::max(max_diag_row_nnz_, diag_nnz_per_row_[i]);
        max_offdiag_row_nnz_ = std::max(max_offdiag_row_nnz_, offdiag_nnz_per_row_[i]);
    }
}

// ============================================================================
// Free functions
// ============================================================================

SparsityPreallocation estimatePreallocation(
    GlobalIndex n_dofs,
    GlobalIndex n_elements,
    GlobalIndex dofs_per_element,
    double avg_elements_per_dof)
{
    // Each DOF couples with dofs_per_element DOFs from each touching element
    // Average DOF touches avg_elements_per_dof elements
    // But many couplings are repeated (elements share DOFs)
    // Rough estimate: nnz_per_row ~ avg_elements_per_dof * dofs_per_element / 2
    // (factor of 2 for typical overlap)

    GlobalIndex estimated_nnz = static_cast<GlobalIndex>(
        std::ceil(avg_elements_per_dof * static_cast<double>(dofs_per_element) / 2.0));

    // Ensure at least diagonal
    estimated_nnz = std::max(estimated_nnz, GlobalIndex{1});

    // Don't exceed full row
    estimated_nnz = std::min(estimated_nnz, n_dofs);

    return SparsityPreallocation(n_dofs, estimated_nnz);
}

} // namespace sparsity
} // namespace FE
} // namespace svmp

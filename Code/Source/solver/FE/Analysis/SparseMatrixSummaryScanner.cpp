/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/SparseMatrixSummaryScanner.h"

#include "Backends/FSILS/FsilsMatrix.h"
#include "Backends/Interfaces/GenericMatrix.h"

#if defined(FE_HAS_EIGEN)
#include "Backends/Eigen/EigenMatrix.h"
#endif

#if defined(FE_HAS_PETSC)
#include "Backends/PETSc/PetscMatrix.h"
#endif

#if defined(FE_HAS_TRILINOS)
#include "Backends/Trilinos/TrilinosMatrix.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace svmp {
namespace FE {
namespace analysis {

namespace {

struct EntryKey {
    GlobalDofId row{INVALID_GLOBAL_INDEX};
    GlobalDofId col{INVALID_GLOBAL_INDEX};

    [[nodiscard]] bool operator==(const EntryKey& other) const noexcept {
        return row == other.row && col == other.col;
    }
};

struct EntryKeyHash {
    [[nodiscard]] std::size_t operator()(const EntryKey& key) const noexcept {
        const auto row_hash = std::hash<GlobalDofId>{}(key.row);
        const auto col_hash = std::hash<GlobalDofId>{}(key.col);
        return row_hash ^ (col_hash + 0x9e3779b97f4a7c15ULL + (row_hash << 6U) + (row_hash >> 2U));
    }
};

[[nodiscard]] bool almostZero(Real value, Real tolerance) noexcept
{
    return std::abs(value) <= tolerance;
}

[[nodiscard]] GlobalDofId backendToFeDof(const backends::FsilsShared& shared,
                                         GlobalDofId backend_dof) noexcept
{
    const auto perm = shared.dof_permutation;
    if (!perm || perm->empty()) {
        return backend_dof;
    }
    if (backend_dof < 0 ||
        static_cast<std::size_t>(backend_dof) >= perm->inverse.size()) {
        return INVALID_GLOBAL_INDEX;
    }
    return perm->inverse[static_cast<std::size_t>(backend_dof)];
}

[[nodiscard]] std::vector<GlobalDofId> sortedUniqueInRange(std::vector<GlobalDofId> values,
                                                           GlobalIndex limit)
{
    values.erase(std::remove_if(values.begin(), values.end(),
                                [limit](GlobalDofId dof) {
                                    return dof < 0 || dof >= limit;
                                }),
                 values.end());
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

class SummaryAccumulator {
public:
    SummaryAccumulator(GlobalIndex rows,
                       GlobalIndex cols,
                       OperatorBlockId block,
                       SparseMatrixScanOptions options)
        : options_(options)
    {
        summary_.block = std::move(block);
        summary_.scope = NumericSummaryScope::FullMatrix;
        summary_.rows = rows;
        summary_.cols = cols;
        summary_.square = rows == cols;
        summary_.sign_tolerance = options_.sign_tolerance;
        summary_.row_sum_tolerance = options_.row_sum_tolerance;
        summary_.symmetry_tolerance = options_.symmetry_tolerance;
        summary_.worst_entry_sample_limit = options_.worst_entry_sample_limit;

        if (rows >= 0 &&
            static_cast<std::uint64_t>(rows) <= options_.row_coverage_storage_limit) {
            row_seen_.assign(static_cast<std::size_t>(rows), 0U);
        } else if (rows > 0) {
            log_.row_coverage_storage_truncated = true;
        }
    }

    void scanSource(const SparseRowScanSource& source)
    {
        updateBackendKind(source.backendKind());
        source_complete_ = source_complete_ || source.hasCompleteGlobalRows();
        ++log_.visited_partitions;

        source.forEachLocalRow(
            [this](GlobalDofId row,
                   const std::vector<SparseMatrixRowEntry>& entries,
                   int owning_rank) {
                visitRow(row, entries, owning_rank);
            });
    }

    [[nodiscard]] SparseMatrixSummaryResult finish()
    {
        const bool complete_rows = rowEvidenceComplete();
        summary_.expected_row_count = summary_.rows > 0
            ? static_cast<std::uint64_t>(summary_.rows)
            : std::uint64_t{0};
        summary_.scanned_row_count = log_.visited_rows;
        summary_.scanned_entry_count = log_.visited_entries;
        summary_.sign_evidence_complete = complete_rows;
        summary_.row_sum_evidence_complete = complete_rows && has_row_sum_;

        finishSymmetry();

        if (!has_row_sum_) {
            summary_.min_row_sum = 0.0;
            summary_.max_row_sum = 0.0;
            summary_.max_abs_row_sum = 0.0;
        }

        log_.retained_symmetry_entries = symmetry_entries_.size();
        log_.retained_row_coverage_entries = row_seen_.size();
        log_.estimated_peak_stored_entries =
            log_.max_row_entries + log_.retained_symmetry_entries +
            log_.retained_row_coverage_entries;
        log_.message = formatSparseMatrixScanLog(log_);
        return {summary_, log_};
    }

private:
    void updateBackendKind(backends::BackendKind backend_kind)
    {
        if (!summary_.backend_kind.has_value()) {
            summary_.backend_kind = backend_kind;
            return;
        }
        if (*summary_.backend_kind != backend_kind) {
            summary_.backend_kind.reset();
        }
    }

    void markRowSeen(GlobalDofId row)
    {
        if (row_seen_.empty() || row < 0 || row >= summary_.rows) {
            return;
        }
        auto& seen = row_seen_[static_cast<std::size_t>(row)];
        if (seen == 0U) {
            seen = 1U;
            ++row_seen_count_;
        }
    }

    void retainSymmetryEntry(GlobalDofId row, GlobalDofId col, Real value)
    {
        if (!options_.compute_symmetry || !summary_.square ||
            log_.symmetry_storage_truncated) {
            return;
        }
        if (row < 0 || col < 0 || row >= summary_.rows || col >= summary_.cols) {
            log_.symmetry_storage_truncated = true;
            return;
        }
        if (symmetry_entries_.size() >= options_.symmetry_entry_storage_limit) {
            log_.symmetry_storage_truncated = true;
            return;
        }
        symmetry_entries_[EntryKey{row, col}] += value;
    }

    void classifyEntry(GlobalDofId row,
                       const SparseMatrixRowEntry& entry,
                       int owning_rank)
    {
        const Real value = entry.value;
        const Real abs_value = std::abs(value);
        summary_.max_abs_entry = std::max(summary_.max_abs_entry, abs_value);
        retainSymmetryEntry(row, entry.col, value);

        if (row == entry.col) {
            ++summary_.diagonal_count;
            if (value <= options_.sign_tolerance) {
                ++summary_.nonpositive_diagonal_count;
                summary_.addWorstEntry(MatrixEntrySample{row,
                                                         entry.col,
                                                         value,
                                                         owning_rank,
                                                         sample_index_,
                                                         "nonpositive diagonal"});
            }
            if (value < -options_.sign_tolerance) {
                ++summary_.negative_diagonal_count;
            }
            if (almostZero(value, options_.sign_tolerance)) {
                ++summary_.near_zero_diagonal_count;
            }
        } else {
            ++summary_.offdiag_count;
            summary_.max_abs_offdiag = std::max(summary_.max_abs_offdiag, abs_value);
            if (value > options_.sign_tolerance) {
                ++summary_.positive_offdiag_count;
                summary_.max_positive_offdiag =
                    std::max(summary_.max_positive_offdiag, value);
                summary_.addWorstEntry(MatrixEntrySample{row,
                                                         entry.col,
                                                         value,
                                                         owning_rank,
                                                         sample_index_,
                                                         "positive offdiagonal"});
            } else if (value < -options_.sign_tolerance) {
                ++summary_.negative_offdiag_count;
            } else {
                ++summary_.near_zero_offdiag_count;
            }
        }

        ++sample_index_;
    }

    void visitRow(GlobalDofId row,
                  const std::vector<SparseMatrixRowEntry>& entries,
                  int owning_rank)
    {
        ++log_.visited_rows;
        log_.visited_entries += static_cast<std::uint64_t>(entries.size());
        log_.max_row_entries = std::max(log_.max_row_entries, entries.size());
        markRowSeen(row);

        Real row_sum = 0.0;
        for (const auto& entry : entries) {
            row_sum += entry.value;
            classifyEntry(row, entry, owning_rank);
        }

        if (!has_row_sum_) {
            summary_.min_row_sum = row_sum;
            summary_.max_row_sum = row_sum;
            has_row_sum_ = true;
        } else {
            summary_.min_row_sum = std::min(summary_.min_row_sum, row_sum);
            summary_.max_row_sum = std::max(summary_.max_row_sum, row_sum);
        }
        summary_.max_abs_row_sum = std::max(summary_.max_abs_row_sum, std::abs(row_sum));
        if (row_sum < -options_.row_sum_tolerance) {
            ++summary_.row_sum_violation_count;
            summary_.addWorstEntry(MatrixEntrySample{row,
                                                     row,
                                                     row_sum,
                                                     owning_rank,
                                                     sample_index_++,
                                                     "negative row sum"});
        }
    }

    [[nodiscard]] bool rowEvidenceComplete() const noexcept
    {
        if (source_complete_) {
            return true;
        }
        if (!row_seen_.empty() && summary_.rows >= 0) {
            return row_seen_count_ == static_cast<std::uint64_t>(summary_.rows);
        }
        return false;
    }

    void finishSymmetry()
    {
        if (!options_.compute_symmetry || !summary_.square) {
            summary_.structurally_symmetric = false;
            summary_.numerically_symmetric = false;
            summary_.symmetry_evidence_complete = false;
            return;
        }

        const bool complete_rows = rowEvidenceComplete();
        if (!complete_rows || log_.symmetry_storage_truncated) {
            summary_.structurally_symmetric = false;
            summary_.numerically_symmetric = false;
            summary_.symmetry_evidence_complete = false;
            return;
        }

        summary_.structurally_symmetric = true;
        summary_.numerically_symmetric = true;
        summary_.symmetry_evidence_complete = true;
        summary_.max_symmetry_error = 0.0;

        for (const auto& item : symmetry_entries_) {
            const auto reverse_it =
                symmetry_entries_.find(EntryKey{item.first.col, item.first.row});
            if (reverse_it == symmetry_entries_.end()) {
                summary_.structurally_symmetric = false;
                summary_.numerically_symmetric = false;
                summary_.max_symmetry_error =
                    std::max(summary_.max_symmetry_error, std::abs(item.second));
                continue;
            }

            const Real error = std::abs(item.second - reverse_it->second);
            summary_.max_symmetry_error = std::max(summary_.max_symmetry_error, error);
            if (error > options_.symmetry_tolerance) {
                summary_.numerically_symmetric = false;
            }
        }

        const Real denom = summary_.max_abs_entry > Real{}
            ? summary_.max_abs_entry
            : Real{1};
        summary_.nonsymmetry_indicator =
            std::abs(summary_.max_symmetry_error) / denom;
    }

    SparseMatrixScanOptions options_{};
    DiscreteMatrixSummary summary_{};
    SparseMatrixScanLog log_{};
    std::unordered_map<EntryKey, Real, EntryKeyHash> symmetry_entries_{};
    std::vector<unsigned char> row_seen_{};
    std::uint64_t row_seen_count_{0};
    std::uint64_t sample_index_{0};
    bool has_row_sum_{false};
    bool source_complete_{false};
};

class FreeFreeRowScanSource final : public SparseRowScanSource {
public:
    FreeFreeRowScanSource(const SparseRowScanSource& source,
                          const std::vector<GlobalDofId>& free_dofs)
        : source_(source)
    {
        for (std::size_t i = 0; i < free_dofs.size(); ++i) {
            free_to_reduced_.emplace(free_dofs[i], static_cast<GlobalDofId>(i));
        }
        reduced_size_ = static_cast<GlobalIndex>(free_to_reduced_.size());
    }

    [[nodiscard]] backends::BackendKind backendKind() const noexcept override {
        return source_.backendKind();
    }
    [[nodiscard]] GlobalIndex numRows() const noexcept override { return reduced_size_; }
    [[nodiscard]] GlobalIndex numCols() const noexcept override { return reduced_size_; }
    [[nodiscard]] bool hasCompleteGlobalRows() const noexcept override {
        return source_.hasCompleteGlobalRows();
    }
    [[nodiscard]] bool isDistributed() const noexcept override {
        return source_.isDistributed();
    }

    void forEachLocalRow(const SparseMatrixRowVisitor& visitor) const override
    {
        std::vector<SparseMatrixRowEntry> reduced_entries;
        source_.forEachLocalRow(
            [this, &visitor, &reduced_entries](
                GlobalDofId row,
                const std::vector<SparseMatrixRowEntry>& entries,
                int owning_rank) {
                const auto row_it = free_to_reduced_.find(row);
                if (row_it == free_to_reduced_.end()) {
                    return;
                }

                reduced_entries.clear();
                reduced_entries.reserve(entries.size());
                for (const auto& entry : entries) {
                    const auto col_it = free_to_reduced_.find(entry.col);
                    if (col_it != free_to_reduced_.end()) {
                        reduced_entries.push_back(
                            SparseMatrixRowEntry{col_it->second, entry.value});
                    }
                }
                visitor(row_it->second, reduced_entries, owning_rank);
            });
    }

private:
    const SparseRowScanSource& source_;
    std::unordered_map<GlobalDofId, GlobalDofId> free_to_reduced_{};
    GlobalIndex reduced_size_{0};
};

#if defined(FE_HAS_EIGEN)
class EigenSparseRowScanSource final : public SparseRowScanSource {
public:
    explicit EigenSparseRowScanSource(const backends::EigenMatrix& matrix)
        : matrix_(matrix)
    {
    }

    [[nodiscard]] backends::BackendKind backendKind() const noexcept override {
        return backends::BackendKind::Eigen;
    }
    [[nodiscard]] GlobalIndex numRows() const noexcept override {
        return matrix_.numRows();
    }
    [[nodiscard]] GlobalIndex numCols() const noexcept override {
        return matrix_.numCols();
    }
    [[nodiscard]] bool hasCompleteGlobalRows() const noexcept override {
        return true;
    }

    void forEachLocalRow(const SparseMatrixRowVisitor& visitor) const override
    {
        std::vector<SparseMatrixRowEntry> entries;
        const auto& mat = matrix_.eigen();
        for (int outer = 0; outer < mat.outerSize(); ++outer) {
            entries.clear();
            for (typename backends::EigenMatrix::SparseMat::InnerIterator it(mat, outer);
                 it; ++it) {
                entries.push_back(SparseMatrixRowEntry{static_cast<GlobalDofId>(it.col()),
                                                       static_cast<Real>(it.value())});
            }
            visitor(static_cast<GlobalDofId>(outer), entries, 0);
        }
    }

private:
    const backends::EigenMatrix& matrix_;
};
#endif

class FsilsSparseRowScanSource final : public SparseRowScanSource {
public:
    explicit FsilsSparseRowScanSource(const backends::FsilsMatrix& matrix)
        : matrix_(matrix)
    {
    }

    [[nodiscard]] backends::BackendKind backendKind() const noexcept override {
        return backends::BackendKind::FSILS;
    }
    [[nodiscard]] GlobalIndex numRows() const noexcept override {
        return matrix_.numRows();
    }
    [[nodiscard]] GlobalIndex numCols() const noexcept override {
        return matrix_.numCols();
    }
    [[nodiscard]] bool hasCompleteGlobalRows() const noexcept override
    {
        const auto shared = matrix_.shared();
        if (!shared || shared->dof <= 0) {
            return false;
        }
        return static_cast<GlobalIndex>(shared->owned_node_count) *
                   static_cast<GlobalIndex>(shared->dof) >= matrix_.numRows();
    }

    void forEachLocalRow(const SparseMatrixRowVisitor& visitor) const override
    {
        const auto shared_ptr = matrix_.shared();
        if (!shared_ptr || shared_ptr->dof <= 0) {
            return;
        }

        const auto& shared = *shared_ptr;
        const int dof = shared.dof;
        const int owning_rank = shared.lhs.commu.task;
        std::vector<SparseMatrixRowEntry> entries;
        entries.reserve(static_cast<std::size_t>(dof) * static_cast<std::size_t>(dof));

        for (int row_internal = 0; row_internal < shared.lhs.nNo; ++row_internal) {
            const int row_old = internalToOld(shared, row_internal);
            if (row_old < 0) {
                continue;
            }
            if (matrix_.usesOwnedRowOperator() && row_old >= shared.owned_node_count) {
                continue;
            }

            const int row_node = shared.oldToGlobalNode(row_old);
            if (row_node < 0) {
                continue;
            }

            const int start = shared.lhs.rowPtr(0, row_internal);
            const int end = shared.lhs.rowPtr(1, row_internal);
            for (int row_comp = 0; row_comp < dof; ++row_comp) {
                const GlobalDofId backend_row =
                    static_cast<GlobalDofId>(row_node) * dof + row_comp;
                const GlobalDofId fe_row = backendToFeDof(shared, backend_row);
                if (fe_row < 0 || fe_row >= matrix_.numRows()) {
                    continue;
                }

                entries.clear();
                if (start >= 0 && end >= start) {
                    for (int nz = start; nz <= end; ++nz) {
                        const int col_internal = shared.lhs.colPtr(nz);
                        const int col_old = internalToOld(shared, col_internal);
                        if (col_old < 0) {
                            continue;
                        }
                        const int col_node = shared.oldToGlobalNode(col_old);
                        if (col_node < 0) {
                            continue;
                        }
                        for (int col_comp = 0; col_comp < dof; ++col_comp) {
                            const GlobalDofId backend_col =
                                static_cast<GlobalDofId>(col_node) * dof + col_comp;
                            const GlobalDofId fe_col = backendToFeDof(shared, backend_col);
                            if (fe_col < 0 || fe_col >= matrix_.numCols()) {
                                continue;
                            }
                            entries.push_back(
                                SparseMatrixRowEntry{fe_col, matrix_.getEntry(fe_row, fe_col)});
                        }
                    }
                }
                visitor(fe_row, entries, owning_rank);
            }
        }
    }

private:
    [[nodiscard]] static int internalToOld(const backends::FsilsShared& shared,
                                           int internal) noexcept
    {
        if (internal < 0 || internal >= shared.lhs.nNo) {
            return -1;
        }
        if (static_cast<std::size_t>(internal) < shared.old_of_internal.size()) {
            return shared.old_of_internal[static_cast<std::size_t>(internal)];
        }
        return internal;
    }

    const backends::FsilsMatrix& matrix_;
};

#if defined(FE_HAS_PETSC)
class PetscSparseRowScanSource final : public SparseRowScanSource {
public:
    explicit PetscSparseRowScanSource(const backends::PetscMatrix& matrix)
        : matrix_(matrix)
    {
    }

    [[nodiscard]] backends::BackendKind backendKind() const noexcept override {
        return backends::BackendKind::PETSc;
    }
    [[nodiscard]] GlobalIndex numRows() const noexcept override {
        PetscInt rows = 0;
        PetscInt cols = 0;
        MatGetSize(matrix_.petsc(), &rows, &cols);
        return static_cast<GlobalIndex>(rows);
    }
    [[nodiscard]] GlobalIndex numCols() const noexcept override {
        PetscInt rows = 0;
        PetscInt cols = 0;
        MatGetSize(matrix_.petsc(), &rows, &cols);
        return static_cast<GlobalIndex>(cols);
    }
    [[nodiscard]] bool hasCompleteGlobalRows() const noexcept override {
        PetscInt rows = 0;
        PetscInt cols = 0;
        PetscInt begin = 0;
        PetscInt end = 0;
        MatGetSize(matrix_.petsc(), &rows, &cols);
        MatGetOwnershipRange(matrix_.petsc(), &begin, &end);
        return begin == 0 && end == rows;
    }

    void forEachLocalRow(const SparseMatrixRowVisitor& visitor) const override
    {
        PetscInt begin = 0;
        PetscInt end = 0;
        MatGetOwnershipRange(matrix_.petsc(), &begin, &end);

        int rank = 0;
        MPI_Comm comm = PETSC_COMM_SELF;
        PetscObjectGetComm(reinterpret_cast<PetscObject>(matrix_.petsc()), &comm);
        MPI_Comm_rank(comm, &rank);

        std::vector<SparseMatrixRowEntry> entries;
        for (PetscInt row = begin; row < end; ++row) {
            PetscInt ncols = 0;
            const PetscInt* cols = nullptr;
            const PetscScalar* values = nullptr;
            MatGetRow(matrix_.petsc(), row, &ncols, &cols, &values);
            entries.clear();
            entries.reserve(static_cast<std::size_t>(ncols));
            for (PetscInt i = 0; i < ncols; ++i) {
                entries.push_back(
                    SparseMatrixRowEntry{static_cast<GlobalDofId>(cols[i]),
                                         static_cast<Real>(PetscRealPart(values[i]))});
            }
            visitor(static_cast<GlobalDofId>(row), entries, rank);
            MatRestoreRow(matrix_.petsc(), row, &ncols, &cols, &values);
        }
    }

private:
    const backends::PetscMatrix& matrix_;
};
#endif

#if defined(FE_HAS_TRILINOS)
class TrilinosSparseRowScanSource final : public SparseRowScanSource {
public:
    explicit TrilinosSparseRowScanSource(const backends::TrilinosMatrix& matrix)
        : matrix_(matrix)
    {
    }

    [[nodiscard]] backends::BackendKind backendKind() const noexcept override {
        return backends::BackendKind::Trilinos;
    }
    [[nodiscard]] GlobalIndex numRows() const noexcept override {
        return static_cast<GlobalIndex>(matrix_.tpetra()->getGlobalNumRows());
    }
    [[nodiscard]] GlobalIndex numCols() const noexcept override {
        return static_cast<GlobalIndex>(matrix_.tpetra()->getGlobalNumCols());
    }
    [[nodiscard]] bool hasCompleteGlobalRows() const noexcept override {
        const auto matrix = matrix_.tpetra();
        return matrix->getLocalNumRows() == matrix->getGlobalNumRows();
    }

    void forEachLocalRow(const SparseMatrixRowVisitor& visitor) const override
    {
        const auto matrix = matrix_.tpetra();
        const auto row_map = matrix->getRowMap();
        const auto col_map = matrix->getColMap();
        const int rank = row_map->getComm()->getRank();

        std::vector<SparseMatrixRowEntry> entries;
        for (backends::trilinos::LO local_row = 0;
             local_row < static_cast<backends::trilinos::LO>(row_map->getLocalNumElements());
             ++local_row) {
            Teuchos::ArrayView<const backends::trilinos::LO> cols;
            Teuchos::ArrayView<const backends::trilinos::Scalar> values;
            matrix->getLocalRowView(local_row, cols, values);
            entries.clear();
            entries.reserve(static_cast<std::size_t>(cols.size()));
            for (int i = 0; i < cols.size(); ++i) {
                entries.push_back(
                    SparseMatrixRowEntry{static_cast<GlobalDofId>(
                                             col_map->getGlobalElement(cols[i])),
                                         static_cast<Real>(values[i])});
            }
            visitor(static_cast<GlobalDofId>(row_map->getGlobalElement(local_row)),
                    entries,
                    rank);
        }
    }

private:
    const backends::TrilinosMatrix& matrix_;
};
#endif

} // namespace

CsrSparseRowScanSource::CsrSparseRowScanSource(GlobalIndex rows,
                                               GlobalIndex cols,
                                               std::vector<GlobalIndex> row_ptr,
                                               std::vector<GlobalIndex> col_indices,
                                               std::vector<Real> values,
                                               backends::BackendKind backend_kind,
                                               GlobalDofId local_row_offset,
                                               bool complete_global_rows,
                                               int owning_rank)
    : rows_(rows),
      cols_(cols),
      row_ptr_(std::move(row_ptr)),
      col_indices_(std::move(col_indices)),
      values_(std::move(values)),
      backend_kind_(backend_kind),
      local_row_offset_(local_row_offset),
      complete_global_rows_(complete_global_rows),
      owning_rank_(owning_rank)
{
    if (row_ptr_.empty()) {
        throw std::invalid_argument("CsrSparseRowScanSource requires a nonempty row pointer");
    }
    if (col_indices_.size() != values_.size()) {
        throw std::invalid_argument("CsrSparseRowScanSource column/value sizes differ");
    }
    if (row_ptr_.front() != 0) {
        throw std::invalid_argument("CsrSparseRowScanSource row pointer must start at zero");
    }
    if (row_ptr_.back() != static_cast<GlobalIndex>(values_.size())) {
        throw std::invalid_argument("CsrSparseRowScanSource row pointer nnz mismatch");
    }
    if (!std::is_sorted(row_ptr_.begin(), row_ptr_.end())) {
        throw std::invalid_argument("CsrSparseRowScanSource row pointer must be monotone");
    }
}

CsrSparseRowScanSource
CsrSparseRowScanSource::fromRows(GlobalIndex rows,
                                 GlobalIndex cols,
                                 const std::vector<std::vector<SparseMatrixRowEntry>>& row_entries,
                                 backends::BackendKind backend_kind,
                                 GlobalDofId local_row_offset,
                                 bool complete_global_rows,
                                 int owning_rank)
{
    std::vector<GlobalIndex> row_ptr;
    std::vector<GlobalIndex> col_indices;
    std::vector<Real> values;
    row_ptr.reserve(row_entries.size() + 1U);
    row_ptr.push_back(0);
    for (const auto& row : row_entries) {
        for (const auto& entry : row) {
            col_indices.push_back(entry.col);
            values.push_back(entry.value);
        }
        row_ptr.push_back(static_cast<GlobalIndex>(values.size()));
    }

    return CsrSparseRowScanSource(rows,
                                  cols,
                                  std::move(row_ptr),
                                  std::move(col_indices),
                                  std::move(values),
                                  backend_kind,
                                  local_row_offset,
                                  complete_global_rows,
                                  owning_rank);
}

void CsrSparseRowScanSource::forEachLocalRow(const SparseMatrixRowVisitor& visitor) const
{
    std::vector<SparseMatrixRowEntry> entries;
    const auto local_rows = static_cast<GlobalIndex>(row_ptr_.size() - 1U);
    for (GlobalIndex local_row = 0; local_row < local_rows; ++local_row) {
        const auto begin = row_ptr_[static_cast<std::size_t>(local_row)];
        const auto end = row_ptr_[static_cast<std::size_t>(local_row + 1)];
        entries.clear();
        entries.reserve(static_cast<std::size_t>(end - begin));
        for (GlobalIndex k = begin; k < end; ++k) {
            const auto idx = static_cast<std::size_t>(k);
            entries.push_back(SparseMatrixRowEntry{col_indices_[idx], values_[idx]});
        }
        visitor(local_row_offset_ + local_row, entries, owning_rank_);
    }
}

ConstraintReductionMask
ConstraintReductionMask::fromConstrainedDofs(GlobalIndex total_dofs,
                                             std::vector<GlobalDofId> constrained_dofs,
                                             ConstraintReductionKind reduction_kind,
                                             bool affine_terms_accounted_for,
                                             bool reduction_exact_for_analysis)
{
    ConstraintReductionMask mask;
    mask.reduction_kind = reduction_kind;
    mask.affine_terms_accounted_for = affine_terms_accounted_for;
    mask.reduction_exact_for_analysis = reduction_exact_for_analysis;
    if (total_dofs <= 0) {
        return mask;
    }

    mask.constrained_dofs = sortedUniqueInRange(std::move(constrained_dofs), total_dofs);
    mask.free_dofs.reserve(static_cast<std::size_t>(total_dofs) -
                           mask.constrained_dofs.size());
    auto constrained_it = mask.constrained_dofs.begin();
    for (GlobalDofId dof = 0; dof < total_dofs; ++dof) {
        if (constrained_it != mask.constrained_dofs.end() && *constrained_it == dof) {
            ++constrained_it;
            continue;
        }
        mask.free_dofs.push_back(dof);
    }
    return mask;
}

SparseMatrixSummaryResult
scanSparseMatrixSummary(const SparseRowScanSource& source,
                        OperatorBlockId block,
                        SparseMatrixScanOptions options)
{
    std::vector<const SparseRowScanSource*> sources{&source};
    return scanSparseMatrixSummary(sources, std::move(block), options);
}

SparseMatrixSummaryResult
scanSparseMatrixSummary(const std::vector<const SparseRowScanSource*>& sources,
                        OperatorBlockId block,
                        SparseMatrixScanOptions options)
{
    const auto start = std::chrono::steady_clock::now();
    if (sources.empty() || sources.front() == nullptr) {
        SparseMatrixSummaryResult result;
        result.summary.block = std::move(block);
        result.summary.sign_tolerance = options.sign_tolerance;
        result.summary.row_sum_tolerance = options.row_sum_tolerance;
        result.summary.symmetry_tolerance = options.symmetry_tolerance;
        result.summary.worst_entry_sample_limit = options.worst_entry_sample_limit;
        result.log.message = "no sparse row sources supplied";
        return result;
    }

    const GlobalIndex rows = sources.front()->numRows();
    const GlobalIndex cols = sources.front()->numCols();
    SummaryAccumulator accumulator(rows, cols, std::move(block), options);
    for (const auto* source : sources) {
        if (source == nullptr) {
            throw std::invalid_argument("scanSparseMatrixSummary received a null source");
        }
        if (source->numRows() != rows || source->numCols() != cols) {
            throw std::invalid_argument("scanSparseMatrixSummary source dimensions differ");
        }
        accumulator.scanSource(*source);
    }

    auto result = accumulator.finish();

#if FE_HAS_MPI
    if (options.mpi_comm != MPI_COMM_NULL) {
        reduceDiscreteMatrixSummaryMPI(result.summary, result.log, options.mpi_comm);
    }
#endif

    const auto end = std::chrono::steady_clock::now();
    result.log.elapsed_seconds =
        std::chrono::duration<double>(end - start).count();
    result.log.message = formatSparseMatrixScanLog(result.log);
    return result;
}

ReducedMatrixSummary
scanReducedFreeFreeSummary(const SparseRowScanSource& source,
                           const ConstraintReductionMask& reduction,
                           OperatorBlockId block,
                           SparseMatrixScanOptions options)
{
    FreeFreeRowScanSource free_free_source(source, reduction.free_dofs);
    auto result = scanSparseMatrixSummary(free_free_source, std::move(block), options);
    result.summary.scope = NumericSummaryScope::ReducedFreeFree;

    ReducedMatrixSummary reduced;
    reduced.free_free_matrix = std::move(result.summary);
    reduced.reduction_kind = reduction.reduction_kind;
    reduced.eliminated_scope = NumericSummaryScope::ReducedFreeFree;
    reduced.free_dof_count = static_cast<std::uint64_t>(reduction.free_dofs.size());
    reduced.constrained_dof_count =
        static_cast<std::uint64_t>(reduction.constrained_dofs.size());
    reduced.retained_multiplier_dof_count = reduction.retained_multiplier_dof_count;
    reduced.affine_terms_accounted_for = reduction.affine_terms_accounted_for;
    reduced.reduction_exact_for_analysis = reduction.reduction_exact_for_analysis;
    return reduced;
}

std::unique_ptr<SparseRowScanSource>
makeSparseRowScanSource(const backends::GenericMatrix& matrix)
{
    switch (matrix.backendKind()) {
        case backends::BackendKind::Eigen:
#if defined(FE_HAS_EIGEN)
            if (const auto* eigen = dynamic_cast<const backends::EigenMatrix*>(&matrix)) {
                return std::make_unique<EigenSparseRowScanSource>(*eigen);
            }
#endif
            return nullptr;

        case backends::BackendKind::FSILS:
            if (const auto* fsils = dynamic_cast<const backends::FsilsMatrix*>(&matrix)) {
                return std::make_unique<FsilsSparseRowScanSource>(*fsils);
            }
            return nullptr;

        case backends::BackendKind::PETSc:
#if defined(FE_HAS_PETSC)
            if (const auto* petsc = dynamic_cast<const backends::PetscMatrix*>(&matrix)) {
                return std::make_unique<PetscSparseRowScanSource>(*petsc);
            }
#endif
            return nullptr;

        case backends::BackendKind::Trilinos:
#if defined(FE_HAS_TRILINOS)
            if (const auto* trilinos = dynamic_cast<const backends::TrilinosMatrix*>(&matrix)) {
                return std::make_unique<TrilinosSparseRowScanSource>(*trilinos);
            }
#endif
            return nullptr;
    }
    return nullptr;
}

DiscreteMatrixSummary
mergeDiscreteMatrixSummaries(const std::vector<DiscreteMatrixSummary>& parts,
                             OperatorBlockId block,
                             SparseMatrixScanOptions options)
{
    DiscreteMatrixSummary merged;
    merged.block = std::move(block);
    merged.scope = NumericSummaryScope::FullMatrix;
    merged.sign_tolerance = options.sign_tolerance;
    merged.row_sum_tolerance = options.row_sum_tolerance;
    merged.symmetry_tolerance = options.symmetry_tolerance;
    merged.worst_entry_sample_limit = options.worst_entry_sample_limit;
    if (parts.empty()) {
        return merged;
    }

    merged.rows = parts.front().rows;
    merged.cols = parts.front().cols;
    merged.square = parts.front().square;
    merged.backend_kind = parts.front().backend_kind;
    merged.structurally_symmetric = true;
    merged.numerically_symmetric = true;
    merged.symmetry_evidence_complete = true;
    bool have_row_sum = false;

    for (const auto& part : parts) {
        if (merged.backend_kind != part.backend_kind) {
            merged.backend_kind.reset();
        }
        merged.max_abs_entry = std::max(merged.max_abs_entry, part.max_abs_entry);
        merged.max_abs_offdiag = std::max(merged.max_abs_offdiag, part.max_abs_offdiag);
        merged.max_positive_offdiag =
            std::max(merged.max_positive_offdiag, part.max_positive_offdiag);
        merged.max_symmetry_error =
            std::max(merged.max_symmetry_error, part.max_symmetry_error);
        merged.max_abs_row_sum =
            std::max(merged.max_abs_row_sum, part.max_abs_row_sum);

        if (!have_row_sum) {
            merged.min_row_sum = part.min_row_sum;
            merged.max_row_sum = part.max_row_sum;
            have_row_sum = true;
        } else {
            merged.min_row_sum = std::min(merged.min_row_sum, part.min_row_sum);
            merged.max_row_sum = std::max(merged.max_row_sum, part.max_row_sum);
        }

        merged.diagonal_count += part.diagonal_count;
        merged.nonpositive_diagonal_count += part.nonpositive_diagonal_count;
        merged.negative_diagonal_count += part.negative_diagonal_count;
        merged.near_zero_diagonal_count += part.near_zero_diagonal_count;
        merged.offdiag_count += part.offdiag_count;
        merged.positive_offdiag_count += part.positive_offdiag_count;
        merged.negative_offdiag_count += part.negative_offdiag_count;
        merged.near_zero_offdiag_count += part.near_zero_offdiag_count;
        merged.row_sum_violation_count += part.row_sum_violation_count;
        merged.scanned_row_count += part.scanned_row_count;
        merged.scanned_entry_count += part.scanned_entry_count;
        merged.expected_row_count =
            std::max(merged.expected_row_count, part.expected_row_count);
        merged.structurally_symmetric =
            merged.structurally_symmetric && part.structurally_symmetric;
        merged.numerically_symmetric =
            merged.numerically_symmetric && part.numerically_symmetric;
        merged.symmetry_evidence_complete =
            merged.symmetry_evidence_complete && part.symmetry_evidence_complete;

        for (const auto& sample : part.worst_entries) {
            merged.addWorstEntry(sample);
        }
    }

    if (merged.expected_row_count == 0u && merged.rows > 0) {
        merged.expected_row_count = static_cast<std::uint64_t>(merged.rows);
    }
    const bool complete_rows =
        merged.expected_row_count > 0u &&
        merged.scanned_row_count >= merged.expected_row_count;
    merged.sign_evidence_complete = complete_rows;
    merged.row_sum_evidence_complete = complete_rows && have_row_sum;

    return merged;
}

#if FE_HAS_MPI
namespace {

[[nodiscard]] bool mpiReady() noexcept
{
    int initialized = 0;
    MPI_Initialized(&initialized);
    return initialized != 0;
}

template <typename T>
void allReduceInPlace(T& value, MPI_Datatype type, MPI_Op op, MPI_Comm comm)
{
    T reduced{};
    MPI_Allreduce(&value, &reduced, 1, type, op, comm);
    value = reduced;
}

[[nodiscard]] std::string serializeWorstSamples(const std::vector<MatrixEntrySample>& samples)
{
    std::ostringstream out;
    out << std::setprecision(17);
    for (const auto& sample : samples) {
        out << sample.row << '\t'
            << sample.col << '\t'
            << sample.value << '\t'
            << sample.owning_rank << '\t'
            << sample.sample_index << '\t'
            << sample.note << '\n';
    }
    return out.str();
}

[[nodiscard]] std::vector<MatrixEntrySample> parseWorstSamples(const std::string& packed)
{
    std::vector<MatrixEntrySample> samples;
    std::istringstream input(packed);
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }

        std::vector<std::string> fields;
        std::size_t begin = 0;
        while (begin <= line.size()) {
            const auto tab = line.find('\t', begin);
            if (tab == std::string::npos) {
                fields.push_back(line.substr(begin));
                break;
            }
            fields.push_back(line.substr(begin, tab - begin));
            begin = tab + 1U;
        }
        if (fields.size() < 6U) {
            continue;
        }

        MatrixEntrySample sample;
        sample.row = static_cast<GlobalDofId>(std::stoll(fields[0]));
        sample.col = static_cast<GlobalDofId>(std::stoll(fields[1]));
        sample.value = static_cast<Real>(std::stod(fields[2]));
        sample.owning_rank = std::stoi(fields[3]);
        sample.sample_index = static_cast<std::uint64_t>(std::stoull(fields[4]));
        sample.note = fields[5];
        samples.push_back(std::move(sample));
    }
    return samples;
}

void allGatherWorstSamples(DiscreteMatrixSummary& summary, MPI_Comm comm)
{
    const auto serialized = serializeWorstSamples(summary.worst_entries);
    int size = 1;
    MPI_Comm_size(comm, &size);

    const int send_count = static_cast<int>(serialized.size());
    std::vector<int> recv_counts(static_cast<std::size_t>(size), 0);
    MPI_Allgather(&send_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);

    std::vector<int> displacements(static_cast<std::size_t>(size), 0);
    int total = 0;
    for (int i = 0; i < size; ++i) {
        displacements[static_cast<std::size_t>(i)] = total;
        total += recv_counts[static_cast<std::size_t>(i)];
    }

    std::string gathered(static_cast<std::size_t>(total), '\0');
    MPI_Allgatherv(serialized.data(),
                   send_count,
                   MPI_CHAR,
                   gathered.data(),
                   recv_counts.data(),
                   displacements.data(),
                   MPI_CHAR,
                   comm);

    const auto samples = parseWorstSamples(gathered);
    summary.worst_entries.clear();
    for (const auto& sample : samples) {
        summary.addWorstEntry(sample);
    }
}

} // namespace

void reduceDiscreteMatrixSummaryMPI(DiscreteMatrixSummary& summary,
                                    SparseMatrixScanLog& log,
                                    MPI_Comm comm)
{
    if (comm == MPI_COMM_NULL || !mpiReady()) {
        return;
    }

    int size = 1;
    MPI_Comm_size(comm, &size);
    if (size <= 1) {
        return;
    }

    auto sum_u64 = [comm](std::uint64_t& value) {
        auto tmp = static_cast<unsigned long long>(value);
        allReduceInPlace(tmp, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
        value = static_cast<std::uint64_t>(tmp);
    };
    auto max_real = [comm](Real& value) {
        allReduceInPlace(value, MPI_DOUBLE, MPI_MAX, comm);
    };
    auto min_real = [comm](Real& value) {
        allReduceInPlace(value, MPI_DOUBLE, MPI_MIN, comm);
    };

    sum_u64(summary.diagonal_count);
    sum_u64(summary.nonpositive_diagonal_count);
    sum_u64(summary.negative_diagonal_count);
    sum_u64(summary.near_zero_diagonal_count);
    sum_u64(summary.offdiag_count);
    sum_u64(summary.positive_offdiag_count);
    sum_u64(summary.negative_offdiag_count);
    sum_u64(summary.near_zero_offdiag_count);
    sum_u64(summary.row_sum_violation_count);
    sum_u64(summary.scanned_row_count);
    sum_u64(summary.scanned_entry_count);

    auto expected_rows = static_cast<unsigned long long>(summary.expected_row_count);
    allReduceInPlace(expected_rows, MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm);
    summary.expected_row_count = static_cast<std::uint64_t>(expected_rows);

    max_real(summary.max_abs_entry);
    max_real(summary.max_abs_offdiag);
    max_real(summary.max_positive_offdiag);
    max_real(summary.max_symmetry_error);
    max_real(summary.max_abs_row_sum);

    auto row_count = log.visited_rows;
    sum_u64(row_count);
    Real min_row = log.visited_rows == 0U
                       ? std::numeric_limits<Real>::infinity()
                       : summary.min_row_sum;
    Real max_row = log.visited_rows == 0U
                       ? -std::numeric_limits<Real>::infinity()
                       : summary.max_row_sum;
    min_real(min_row);
    max_real(max_row);
    if (row_count == 0U) {
        summary.min_row_sum = 0.0;
        summary.max_row_sum = 0.0;
    } else {
        summary.min_row_sum = min_row;
        summary.max_row_sum = max_row;
    }

    auto log_rows = log.visited_rows;
    auto log_entries = log.visited_entries;
    auto log_partitions = log.visited_partitions;
    sum_u64(log_rows);
    sum_u64(log_entries);
    sum_u64(log_partitions);
    log.visited_rows = log_rows;
    log.visited_entries = log_entries;
    log.visited_partitions = log_partitions;

    auto max_row_entries = static_cast<unsigned long long>(log.max_row_entries);
    allReduceInPlace(max_row_entries, MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm);
    log.max_row_entries = static_cast<std::size_t>(max_row_entries);

    auto retained_symmetry =
        static_cast<unsigned long long>(log.retained_symmetry_entries);
    allReduceInPlace(retained_symmetry, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
    log.retained_symmetry_entries = static_cast<std::size_t>(retained_symmetry);

    auto peak_stored =
        static_cast<unsigned long long>(log.estimated_peak_stored_entries);
    allReduceInPlace(peak_stored, MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm);
    log.estimated_peak_stored_entries = static_cast<std::size_t>(peak_stored);

    int dense = log.dense_matrix_materialized ? 1 : 0;
    int sym_truncated = log.symmetry_storage_truncated ? 1 : 0;
    int row_truncated = log.row_coverage_storage_truncated ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &dense, 1, MPI_INT, MPI_MAX, comm);
    MPI_Allreduce(MPI_IN_PLACE, &sym_truncated, 1, MPI_INT, MPI_MAX, comm);
    MPI_Allreduce(MPI_IN_PLACE, &row_truncated, 1, MPI_INT, MPI_MAX, comm);
    log.dense_matrix_materialized = dense != 0;
    log.symmetry_storage_truncated = sym_truncated != 0;
    log.row_coverage_storage_truncated = row_truncated != 0;

    int all_square = summary.square ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &all_square, 1, MPI_INT, MPI_MIN, comm);
    summary.square = all_square != 0;
    const bool complete_rows =
        summary.expected_row_count > 0u &&
        summary.scanned_row_count >= summary.expected_row_count &&
        !log.row_coverage_storage_truncated;
    summary.sign_evidence_complete = complete_rows;
    summary.row_sum_evidence_complete = complete_rows && row_count > 0u;
    summary.symmetry_evidence_complete = false;
    summary.structurally_symmetric = false;
    summary.numerically_symmetric = false;
    allGatherWorstSamples(summary, comm);
    log.mpi_reduced = true;
}
#endif

std::string formatSparseMatrixScanLog(const SparseMatrixScanLog& log)
{
    std::ostringstream out;
    out << "sparse scan rows=" << log.visited_rows
        << " entries=" << log.visited_entries
        << " partitions=" << log.visited_partitions
        << " max_row_entries=" << log.max_row_entries
        << " retained_symmetry_entries=" << log.retained_symmetry_entries
        << " retained_row_coverage_entries=" << log.retained_row_coverage_entries
        << " estimated_peak_stored_entries=" << log.estimated_peak_stored_entries
        << " dense_matrix_materialized=" << (log.dense_matrix_materialized ? "true" : "false")
        << " symmetry_storage_truncated=" << (log.symmetry_storage_truncated ? "true" : "false")
        << " row_coverage_storage_truncated="
        << (log.row_coverage_storage_truncated ? "true" : "false")
        << " mpi_reduced=" << (log.mpi_reduced ? "true" : "false")
        << " elapsed_seconds=" << log.elapsed_seconds;
    return out.str();
}

} // namespace analysis
} // namespace FE
} // namespace svmp

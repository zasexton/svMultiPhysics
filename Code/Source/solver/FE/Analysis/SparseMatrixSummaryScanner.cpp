/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/SparseMatrixSummaryScanner.h"

#include "Analysis/AnalysisNumericGuards.h"

#include "Backends/Interfaces/GenericMatrix.h"
#include "Core/MpiCollectiveTrace.h"

#if defined(FE_HAS_FSILS)
#include "Backends/FSILS/FsilsMatrix.h"
#endif

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

struct NumericEntryValue {
    Real real{};
    Real imag{};

    NumericEntryValue& operator+=(const NumericEntryValue& other) noexcept
    {
        real += other.real;
        imag += other.imag;
        return *this;
    }
};

[[nodiscard]] NumericEntryValue asNumericValue(const SparseMatrixRowEntry& entry) noexcept
{
    return NumericEntryValue{entry.value, entry.imaginary_value};
}

[[nodiscard]] Real magnitude(const NumericEntryValue& value) noexcept
{
    return std::hypot(value.real, value.imag);
}

[[nodiscard]] bool hasImaginaryPart(const NumericEntryValue& value,
                                    Real tolerance) noexcept
{
    return std::abs(value.imag) > tolerance;
}

[[nodiscard]] Real complexDifferenceMagnitude(const NumericEntryValue& a,
                                              const NumericEntryValue& b) noexcept
{
    return std::hypot(a.real - b.real, a.imag - b.imag);
}

[[nodiscard]] Real hermitianDifferenceMagnitude(const NumericEntryValue& a,
                                                const NumericEntryValue& transpose) noexcept
{
    return std::hypot(a.real - transpose.real, a.imag + transpose.imag);
}

[[nodiscard]] bool almostZero(Real value, Real tolerance) noexcept
{
    return std::abs(value) <= tolerance;
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
        summary_.expected_row_count = summary_.rows > 0
            ? static_cast<std::uint64_t>(summary_.rows)
            : std::uint64_t{0};
        finalizeRowCoverage();
        const bool complete_rows = rowEvidenceComplete();
        summary_.scanned_row_count = log_.visited_rows;
        summary_.scanned_entry_count = log_.visited_entries;
        summary_.sign_evidence_complete =
            complete_rows &&
            !summary_.complex_values_present &&
            summary_.invalid_entry_count == 0u &&
            summary_.nonfinite_entry_count == 0u;
        summary_.row_sum_evidence_complete =
            complete_rows && has_row_sum_ &&
            !summary_.complex_values_present &&
            summary_.invalid_entry_count == 0u &&
            summary_.nonfinite_row_sum_count == 0u;

        finishSymmetry();

        if (has_gershgorin_bounds_) {
            summary_.gershgorin_lower_bound = gershgorin_lower_bound_;
            summary_.gershgorin_upper_bound = gershgorin_upper_bound_;
            if (summary_.symmetry_evidence_complete &&
                summary_.numerically_symmetric) {
                const Real denom = std::abs(gershgorin_lower_bound_);
                const Real upper = std::max(std::abs(gershgorin_upper_bound_),
                                            summary_.max_abs_entry);
                if (denom > std::max(options_.sign_tolerance, Real{})) {
                    summary_.condition_estimate = upper / denom;
                }
            }
        }

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
        } else {
            ++summary_.duplicate_row_visit_count;
        }
    }

    void finalizeRowCoverage()
    {
        summary_.row_ownership_disjoint =
            summary_.duplicate_row_visit_count == 0u;
        summary_.missing_row_count = 0u;
        if (summary_.rows == 0) {
            summary_.global_row_coverage_exact = true;
            return;
        }
        if (row_seen_.empty() || log_.row_coverage_storage_truncated ||
            summary_.rows < 0) {
            summary_.global_row_coverage_exact = false;
            return;
        }
        for (const auto seen : row_seen_) {
            if (seen == 0U) {
                ++summary_.missing_row_count;
            }
        }
        summary_.global_row_coverage_exact =
            summary_.missing_row_count == 0u &&
            summary_.duplicate_row_visit_count == 0u;
    }

    void retainSymmetryEntry(GlobalDofId row,
                             GlobalDofId col,
                             NumericEntryValue value)
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
        const auto value = asNumericValue(entry);
        if (!numeric::finite(value.real) || !numeric::finite(value.imag)) {
            ++summary_.nonfinite_entry_count;
            summary_.addWorstEntry(MatrixEntrySample{row,
                                                     entry.col,
                                                     value.real,
                                                     owning_rank,
                                                     sample_index_++,
                                                     "nonfinite entry"});
            return;
        }
        const bool complex_entry =
            hasImaginaryPart(value, options_.sign_tolerance);
        if (complex_entry) {
            summary_.complex_values_present = true;
            summary_.max_abs_imag_entry =
                std::max(summary_.max_abs_imag_entry, std::abs(value.imag));
        }
        const Real abs_value = magnitude(value);
        summary_.max_abs_entry = std::max(summary_.max_abs_entry, abs_value);
        retainSymmetryEntry(row, entry.col, value);

        if (row == entry.col) {
            ++summary_.diagonal_count;
            if (!complex_entry && value.real <= options_.sign_tolerance) {
                ++summary_.nonpositive_diagonal_count;
                summary_.addWorstEntry(MatrixEntrySample{row,
                                                         entry.col,
                                                         value.real,
                                                         owning_rank,
                                                         sample_index_,
                                                         "nonpositive diagonal"});
            }
            if (!complex_entry && value.real < -options_.sign_tolerance) {
                ++summary_.negative_diagonal_count;
            }
            if (!complex_entry && almostZero(value.real, options_.sign_tolerance)) {
                ++summary_.near_zero_diagonal_count;
            }
        } else {
            ++summary_.offdiag_count;
            summary_.max_abs_offdiag = std::max(summary_.max_abs_offdiag, abs_value);
            if (!complex_entry && value.real > options_.sign_tolerance) {
                ++summary_.positive_offdiag_count;
                summary_.max_positive_offdiag =
                    std::max(summary_.max_positive_offdiag, value.real);
                summary_.addWorstEntry(MatrixEntrySample{row,
                                                         entry.col,
                                                         value.real,
                                                         owning_rank,
                                                         sample_index_,
                                                         "positive offdiagonal"});
            } else if (!complex_entry && value.real < -options_.sign_tolerance) {
                ++summary_.negative_offdiag_count;
            } else if (!complex_entry) {
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

        if (row < 0 || row >= summary_.rows) {
            ++summary_.invalid_entry_count;
            ++summary_.nonfinite_row_sum_count;
            summary_.addWorstEntry(MatrixEntrySample{
                row,
                INVALID_GLOBAL_INDEX,
                std::numeric_limits<Real>::quiet_NaN(),
                owning_rank,
                sample_index_++,
                "invalid row index"});
            return;
        }

        std::unordered_map<GlobalDofId, NumericEntryValue> aggregated_entries;
        aggregated_entries.reserve(entries.size());
        bool row_has_nonfinite = false;
        bool row_has_invalid = false;
        bool row_has_complex = false;
        for (const auto& entry : entries) {
            if (entry.col < 0 || entry.col >= summary_.cols) {
                ++summary_.invalid_entry_count;
                row_has_invalid = true;
                summary_.addWorstEntry(MatrixEntrySample{
                    row,
                    entry.col,
                    entry.value,
                    owning_rank,
                    sample_index_++,
                    "invalid column index"});
                continue;
            }
            if (!numeric::finite(entry.value) ||
                !numeric::finite(entry.imaginary_value)) {
                ++summary_.nonfinite_entry_count;
                row_has_nonfinite = true;
                summary_.addWorstEntry(MatrixEntrySample{row,
                                                         entry.col,
                                                         entry.value,
                                                         owning_rank,
                                                         sample_index_++,
                                                         "nonfinite entry"});
                continue;
            }
            const auto numeric_value = asNumericValue(entry);
            row_has_complex =
                row_has_complex ||
                hasImaginaryPart(numeric_value, options_.sign_tolerance);
            aggregated_entries[entry.col] += numeric_value;
        }

        std::vector<SparseMatrixRowEntry> row_entries;
        row_entries.reserve(aggregated_entries.size());
        for (const auto& item : aggregated_entries) {
            row_entries.push_back(SparseMatrixRowEntry{
                item.first, item.second.real, item.second.imag});
        }
        std::sort(row_entries.begin(),
                  row_entries.end(),
                  [](const SparseMatrixRowEntry& a,
                     const SparseMatrixRowEntry& b) {
                      return a.col < b.col;
                  });

        Real row_sum = 0.0;
        Real diagonal_value = 0.0;
        Real offdiag_abs_sum = 0.0;
        bool diagonal_seen = false;
        for (const auto& entry : row_entries) {
            row_sum += entry.value;
            if (entry.col == row) {
                diagonal_value += entry.value;
                diagonal_seen = true;
            } else {
                offdiag_abs_sum +=
                    magnitude(NumericEntryValue{entry.value,
                                                entry.imaginary_value});
            }
            classifyEntry(row, entry, owning_rank);
        }

        if (summary_.square && !diagonal_seen) {
            ++summary_.missing_diagonal_count;
            ++summary_.nonpositive_diagonal_count;
            ++summary_.near_zero_diagonal_count;
            summary_.addWorstEntry(MatrixEntrySample{row,
                                                     row,
                                                     Real{},
                                                     owning_rank,
                                                     sample_index_++,
                                                     "missing diagonal"});
        }

        if (row_has_nonfinite || row_has_invalid) {
            ++summary_.nonfinite_row_sum_count;
            summary_.addWorstEntry(MatrixEntrySample{
                row,
                row,
                std::numeric_limits<Real>::quiet_NaN(),
                owning_rank,
                sample_index_++,
                row_has_invalid ? "invalid row sum scope"
                                : "nonfinite row sum"});
            return;
        }

        if (summary_.square && row >= 0 && row < summary_.rows) {
            const Real center = diagonal_seen ? diagonal_value : Real{};
            const Real lower = center - offdiag_abs_sum;
            const Real upper = center + offdiag_abs_sum;
            if (!has_gershgorin_bounds_) {
                gershgorin_lower_bound_ = lower;
                gershgorin_upper_bound_ = upper;
                has_gershgorin_bounds_ = true;
            } else {
                gershgorin_lower_bound_ =
                    std::min(gershgorin_lower_bound_, lower);
                gershgorin_upper_bound_ =
                    std::max(gershgorin_upper_bound_, upper);
            }
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
        if (row_has_complex || summary_.complex_values_present) {
            summary_.complex_values_present = true;
        } else if (row_sum < -options_.row_sum_tolerance) {
            ++summary_.negative_row_sum_count;
            ++summary_.row_sum_violation_count;
            summary_.addWorstEntry(MatrixEntrySample{row,
                                                     row,
                                                     row_sum,
                                                     owning_rank,
                                                     sample_index_++,
                                                     "negative row sum"});
        } else if (row_sum > options_.row_sum_tolerance) {
            ++summary_.positive_row_sum_count;
        } else {
            ++summary_.near_zero_row_sum_count;
        }
    }

    [[nodiscard]] bool rowEvidenceComplete() const noexcept
    {
        if (summary_.rows == 0) {
            return true;
        }
        return summary_.global_row_coverage_exact;
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
        if (!complete_rows ||
            summary_.invalid_entry_count > 0u ||
            summary_.nonfinite_entry_count > 0u ||
            log_.symmetry_storage_truncated) {
            summary_.structurally_symmetric = false;
            summary_.numerically_symmetric = false;
            summary_.symmetry_evidence_complete = false;
            return;
        }

        summary_.structurally_symmetric = !summary_.complex_values_present;
        summary_.numerically_symmetric = !summary_.complex_values_present;
        summary_.symmetry_evidence_complete = true;
        summary_.structurally_complex_symmetric = true;
        summary_.numerically_complex_symmetric = true;
        summary_.complex_symmetry_evidence_complete = true;
        summary_.structurally_hermitian = true;
        summary_.numerically_hermitian = true;
        summary_.hermitian_evidence_complete = true;
        summary_.max_symmetry_error = 0.0;
        summary_.max_complex_symmetry_error = 0.0;
        summary_.max_hermitian_error = 0.0;

        for (const auto& item : symmetry_entries_) {
            const auto reverse_it =
                symmetry_entries_.find(EntryKey{item.first.col, item.first.row});
            if (reverse_it == symmetry_entries_.end()) {
                summary_.structurally_symmetric = false;
                summary_.numerically_symmetric = false;
                summary_.structurally_complex_symmetric = false;
                summary_.numerically_complex_symmetric = false;
                summary_.structurally_hermitian = false;
                summary_.numerically_hermitian = false;
                summary_.max_symmetry_error =
                    std::max(summary_.max_symmetry_error,
                             magnitude(item.second));
                summary_.max_complex_symmetry_error =
                    std::max(summary_.max_complex_symmetry_error,
                             magnitude(item.second));
                summary_.max_hermitian_error =
                    std::max(summary_.max_hermitian_error,
                             magnitude(item.second));
                continue;
            }

            const Real complex_symmetry_error =
                complexDifferenceMagnitude(item.second, reverse_it->second);
            const Real hermitian_error =
                hermitianDifferenceMagnitude(item.second, reverse_it->second);
            const Real error = std::abs(item.second.real - reverse_it->second.real);
            summary_.max_symmetry_error = std::max(summary_.max_symmetry_error, error);
            summary_.max_complex_symmetry_error =
                std::max(summary_.max_complex_symmetry_error,
                         complex_symmetry_error);
            summary_.max_hermitian_error =
                std::max(summary_.max_hermitian_error, hermitian_error);
            if (error > options_.symmetry_tolerance ||
                summary_.complex_values_present) {
                summary_.numerically_symmetric = false;
            }
            if (complex_symmetry_error > options_.symmetry_tolerance) {
                summary_.numerically_complex_symmetric = false;
            }
            if (hermitian_error > options_.symmetry_tolerance) {
                summary_.numerically_hermitian = false;
            }
        }

        const Real denom = summary_.max_abs_entry > Real{}
            ? summary_.max_abs_entry
            : Real{1};
        summary_.nonsymmetry_indicator =
            std::abs(summary_.max_symmetry_error) / denom;
        if (summary_.complex_values_present) {
            summary_.symmetry_evidence_complete = false;
            summary_.nonsymmetry_indicator =
                summary_.max_hermitian_error / denom;
        }
    }

    SparseMatrixScanOptions options_{};
    DiscreteMatrixSummary summary_{};
    SparseMatrixScanLog log_{};
    std::unordered_map<EntryKey, NumericEntryValue, EntryKeyHash> symmetry_entries_{};
    std::vector<unsigned char> row_seen_{};
    std::uint64_t row_seen_count_{0};
    std::uint64_t sample_index_{0};
    Real gershgorin_lower_bound_{};
    Real gershgorin_upper_bound_{};
    bool has_row_sum_{false};
    bool has_gershgorin_bounds_{false};
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
        static_assert(backends::EigenMatrix::SparseMat::IsRowMajor,
                      "Eigen sparse summary scanning requires row-major storage");
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

#if defined(FE_HAS_FSILS)
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
#endif

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
#if defined(PETSC_USE_COMPLEX)
                entries.push_back(SparseMatrixRowEntry{
                    static_cast<GlobalDofId>(cols[i]),
                    static_cast<Real>(PetscRealPart(values[i])),
                    static_cast<Real>(PetscImaginaryPart(values[i]))});
#else
                entries.push_back(
                    SparseMatrixRowEntry{static_cast<GlobalDofId>(cols[i]),
                                         static_cast<Real>(PetscRealPart(values[i]))});
#endif
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
#if defined(FE_HAS_FSILS)
            if (const auto* fsils = dynamic_cast<const backends::FsilsMatrix*>(&matrix)) {
                return std::make_unique<FsilsSparseRowScanSource>(*fsils);
            }
#endif
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
    merged.structurally_hermitian = true;
    merged.numerically_hermitian = true;
    merged.hermitian_evidence_complete = true;
    merged.structurally_complex_symmetric = true;
    merged.numerically_complex_symmetric = true;
    merged.complex_symmetry_evidence_complete = true;
    merged.row_ownership_disjoint = true;
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
        merged.max_hermitian_error =
            std::max(merged.max_hermitian_error, part.max_hermitian_error);
        merged.max_complex_symmetry_error =
            std::max(merged.max_complex_symmetry_error,
                     part.max_complex_symmetry_error);
        merged.max_abs_imag_entry =
            std::max(merged.max_abs_imag_entry, part.max_abs_imag_entry);
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
        merged.missing_diagonal_count += part.missing_diagonal_count;
        merged.nonpositive_diagonal_count += part.nonpositive_diagonal_count;
        merged.negative_diagonal_count += part.negative_diagonal_count;
        merged.near_zero_diagonal_count += part.near_zero_diagonal_count;
        merged.offdiag_count += part.offdiag_count;
        merged.positive_offdiag_count += part.positive_offdiag_count;
        merged.negative_offdiag_count += part.negative_offdiag_count;
        merged.near_zero_offdiag_count += part.near_zero_offdiag_count;
        merged.negative_row_sum_count += part.negative_row_sum_count;
        merged.positive_row_sum_count += part.positive_row_sum_count;
        merged.near_zero_row_sum_count += part.near_zero_row_sum_count;
        merged.row_sum_violation_count += part.row_sum_violation_count;
        merged.invalid_entry_count += part.invalid_entry_count;
        merged.nonfinite_entry_count += part.nonfinite_entry_count;
        merged.nonfinite_row_sum_count += part.nonfinite_row_sum_count;
        merged.scanned_row_count += part.scanned_row_count;
        merged.scanned_entry_count += part.scanned_entry_count;
        merged.duplicate_row_visit_count += part.duplicate_row_visit_count;
        merged.missing_row_count += part.missing_row_count;
        merged.expected_row_count =
            std::max(merged.expected_row_count, part.expected_row_count);
        merged.complex_values_present =
            merged.complex_values_present || part.complex_values_present;
        merged.global_row_coverage_exact =
            merged.global_row_coverage_exact || part.global_row_coverage_exact;
        merged.row_ownership_disjoint =
            merged.row_ownership_disjoint && part.row_ownership_disjoint;
        merged.structurally_symmetric =
            merged.structurally_symmetric && part.structurally_symmetric;
        merged.numerically_symmetric =
            merged.numerically_symmetric && part.numerically_symmetric;
        merged.symmetry_evidence_complete =
            merged.symmetry_evidence_complete && part.symmetry_evidence_complete;
        merged.structurally_hermitian =
            merged.structurally_hermitian && part.structurally_hermitian;
        merged.numerically_hermitian =
            merged.numerically_hermitian && part.numerically_hermitian;
        merged.hermitian_evidence_complete =
            merged.hermitian_evidence_complete && part.hermitian_evidence_complete;
        merged.structurally_complex_symmetric =
            merged.structurally_complex_symmetric &&
            part.structurally_complex_symmetric;
        merged.numerically_complex_symmetric =
            merged.numerically_complex_symmetric &&
            part.numerically_complex_symmetric;
        merged.complex_symmetry_evidence_complete =
            merged.complex_symmetry_evidence_complete &&
            part.complex_symmetry_evidence_complete;

        for (const auto& sample : part.worst_entries) {
            merged.addWorstEntry(sample);
        }
        if (part.gershgorin_lower_bound) {
            merged.gershgorin_lower_bound =
                merged.gershgorin_lower_bound
                    ? std::min(*merged.gershgorin_lower_bound,
                               *part.gershgorin_lower_bound)
                    : part.gershgorin_lower_bound;
        }
        if (part.gershgorin_upper_bound) {
            merged.gershgorin_upper_bound =
                merged.gershgorin_upper_bound
                    ? std::max(*merged.gershgorin_upper_bound,
                               *part.gershgorin_upper_bound)
                    : part.gershgorin_upper_bound;
        }
    }

    if (merged.expected_row_count == 0u && merged.rows > 0) {
        merged.expected_row_count = static_cast<std::uint64_t>(merged.rows);
    }
    if (parts.size() != 1u) {
        merged.global_row_coverage_exact = false;
        merged.symmetry_evidence_complete = false;
        merged.hermitian_evidence_complete = false;
        merged.complex_symmetry_evidence_complete = false;
    }
    const bool complete_rows = merged.global_row_coverage_exact;
    merged.sign_evidence_complete =
        complete_rows &&
        !merged.complex_values_present &&
        merged.invalid_entry_count == 0u &&
        merged.nonfinite_entry_count == 0u;
    merged.row_sum_evidence_complete =
        complete_rows && have_row_sum &&
        !merged.complex_values_present &&
        merged.invalid_entry_count == 0u &&
        merged.nonfinite_row_sum_count == 0u;

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
void allReduceBufferInPlace(std::vector<T>& values,
                            MPI_Datatype type,
                            MPI_Op op,
                            MPI_Comm comm,
                            const char* label)
{
    if (values.empty()) {
        return;
    }

    std::vector<T> reduced(values.size());
    const auto count = static_cast<int>(values.size());
    const auto seq = debug::nextMpiCollectiveTraceSeq();
    debug::traceMpiCollective("before", seq, label, count, type, op, comm);
    MPI_Allreduce(values.data(), reduced.data(), count, type, op, comm);
    debug::traceMpiCollective("after", seq, label, count, type, op, comm);
    values = std::move(reduced);
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
    const auto gather_seq = debug::nextMpiCollectiveTraceSeq();
    debug::traceMpiCollective("before", gather_seq, "SparseMatrixSummaryScanner::allGatherWorstSamples.counts", 1, MPI_INT, MPI_SUM, comm);
    MPI_Allgather(&send_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);
    debug::traceMpiCollective("after", gather_seq, "SparseMatrixSummaryScanner::allGatherWorstSamples.counts", 1, MPI_INT, MPI_SUM, comm);

    std::vector<int> displacements(static_cast<std::size_t>(size), 0);
    int total = 0;
    for (int i = 0; i < size; ++i) {
        displacements[static_cast<std::size_t>(i)] = total;
        total += recv_counts[static_cast<std::size_t>(i)];
    }

    std::string gathered(static_cast<std::size_t>(total), '\0');
    const auto gatherv_seq = debug::nextMpiCollectiveTraceSeq();
    debug::traceMpiCollective("before", gatherv_seq, "SparseMatrixSummaryScanner::allGatherWorstSamples.payload", send_count, MPI_CHAR, MPI_SUM, comm);
    MPI_Allgatherv(serialized.data(),
                   send_count,
                   MPI_CHAR,
                   gathered.data(),
                   recv_counts.data(),
                   displacements.data(),
                   MPI_CHAR,
                   comm);
    debug::traceMpiCollective("after", gatherv_seq, "SparseMatrixSummaryScanner::allGatherWorstSamples.payload", send_count, MPI_CHAR, MPI_SUM, comm);

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

    constexpr std::size_t kDiagonalCount = 0;
    constexpr std::size_t kMissingDiagonalCount = 1;
    constexpr std::size_t kNonpositiveDiagonalCount = 2;
    constexpr std::size_t kNegativeDiagonalCount = 3;
    constexpr std::size_t kNearZeroDiagonalCount = 4;
    constexpr std::size_t kOffdiagCount = 5;
    constexpr std::size_t kPositiveOffdiagCount = 6;
    constexpr std::size_t kNegativeOffdiagCount = 7;
    constexpr std::size_t kNearZeroOffdiagCount = 8;
    constexpr std::size_t kNegativeRowSumCount = 9;
    constexpr std::size_t kPositiveRowSumCount = 10;
    constexpr std::size_t kNearZeroRowSumCount = 11;
    constexpr std::size_t kRowSumViolationCount = 12;
    constexpr std::size_t kInvalidEntryCount = 13;
    constexpr std::size_t kNonfiniteEntryCount = 14;
    constexpr std::size_t kNonfiniteRowSumCount = 15;
    constexpr std::size_t kScannedRowCount = 16;
    constexpr std::size_t kScannedEntryCount = 17;
    constexpr std::size_t kDuplicateRowVisitCount = 18;
    constexpr std::size_t kMissingRowCount = 19;
    constexpr std::size_t kRowsForRowSumExtrema = 20;
    constexpr std::size_t kLogVisitedRows = 21;
    constexpr std::size_t kLogVisitedEntries = 22;
    constexpr std::size_t kLogVisitedPartitions = 23;
    constexpr std::size_t kRetainedSymmetryEntries = 24;

    std::vector<unsigned long long> sum_u64{
        static_cast<unsigned long long>(summary.diagonal_count),
        static_cast<unsigned long long>(summary.missing_diagonal_count),
        static_cast<unsigned long long>(summary.nonpositive_diagonal_count),
        static_cast<unsigned long long>(summary.negative_diagonal_count),
        static_cast<unsigned long long>(summary.near_zero_diagonal_count),
        static_cast<unsigned long long>(summary.offdiag_count),
        static_cast<unsigned long long>(summary.positive_offdiag_count),
        static_cast<unsigned long long>(summary.negative_offdiag_count),
        static_cast<unsigned long long>(summary.near_zero_offdiag_count),
        static_cast<unsigned long long>(summary.negative_row_sum_count),
        static_cast<unsigned long long>(summary.positive_row_sum_count),
        static_cast<unsigned long long>(summary.near_zero_row_sum_count),
        static_cast<unsigned long long>(summary.row_sum_violation_count),
        static_cast<unsigned long long>(summary.invalid_entry_count),
        static_cast<unsigned long long>(summary.nonfinite_entry_count),
        static_cast<unsigned long long>(summary.nonfinite_row_sum_count),
        static_cast<unsigned long long>(summary.scanned_row_count),
        static_cast<unsigned long long>(summary.scanned_entry_count),
        static_cast<unsigned long long>(summary.duplicate_row_visit_count),
        static_cast<unsigned long long>(summary.missing_row_count),
        static_cast<unsigned long long>(log.visited_rows),
        static_cast<unsigned long long>(log.visited_rows),
        static_cast<unsigned long long>(log.visited_entries),
        static_cast<unsigned long long>(log.visited_partitions),
        static_cast<unsigned long long>(log.retained_symmetry_entries),
    };

    constexpr std::size_t kExpectedRows = 0;
    constexpr std::size_t kMaxRowEntries = 1;
    constexpr std::size_t kPeakStoredEntries = 2;
    std::vector<unsigned long long> max_u64{
        static_cast<unsigned long long>(summary.expected_row_count),
        static_cast<unsigned long long>(log.max_row_entries),
        static_cast<unsigned long long>(log.estimated_peak_stored_entries),
    };

    const bool local_has_rows_for_row_sum = log.visited_rows != 0U;
    constexpr std::size_t kMaxAbsEntry = 0;
    constexpr std::size_t kMaxAbsOffdiag = 1;
    constexpr std::size_t kMaxPositiveOffdiag = 2;
    constexpr std::size_t kMaxSymmetryError = 3;
    constexpr std::size_t kMaxHermitianError = 4;
    constexpr std::size_t kMaxComplexSymmetryError = 5;
    constexpr std::size_t kMaxAbsImagEntry = 6;
    constexpr std::size_t kMaxAbsRowSum = 7;
    constexpr std::size_t kGershgorinUpper = 8;
    constexpr std::size_t kMaxRowSum = 9;
    std::vector<Real> max_real{
        summary.max_abs_entry,
        summary.max_abs_offdiag,
        summary.max_positive_offdiag,
        summary.max_symmetry_error,
        summary.max_hermitian_error,
        summary.max_complex_symmetry_error,
        summary.max_abs_imag_entry,
        summary.max_abs_row_sum,
        summary.gershgorin_upper_bound
            ? *summary.gershgorin_upper_bound
            : -std::numeric_limits<Real>::infinity(),
        local_has_rows_for_row_sum
            ? summary.max_row_sum
            : -std::numeric_limits<Real>::infinity(),
    };

    constexpr std::size_t kGershgorinLower = 0;
    constexpr std::size_t kMinRowSum = 1;
    std::vector<Real> min_real{
        summary.gershgorin_lower_bound
            ? *summary.gershgorin_lower_bound
            : std::numeric_limits<Real>::infinity(),
        local_has_rows_for_row_sum
            ? summary.min_row_sum
            : std::numeric_limits<Real>::infinity(),
    };

    constexpr std::size_t kHasGershgorinLower = 0;
    constexpr std::size_t kHasGershgorinUpper = 1;
    constexpr std::size_t kDenseMaterialized = 2;
    constexpr std::size_t kSymmetryTruncated = 3;
    constexpr std::size_t kRowCoverageTruncated = 4;
    constexpr std::size_t kComplexValuesPresent = 5;
    std::vector<int> max_int{
        summary.gershgorin_lower_bound ? 1 : 0,
        summary.gershgorin_upper_bound ? 1 : 0,
        log.dense_matrix_materialized ? 1 : 0,
        log.symmetry_storage_truncated ? 1 : 0,
        log.row_coverage_storage_truncated ? 1 : 0,
        summary.complex_values_present ? 1 : 0,
    };

    constexpr std::size_t kAllSquare = 0;
    std::vector<int> min_int{
        summary.square ? 1 : 0,
    };

    allReduceBufferInPlace(sum_u64,
                           MPI_UNSIGNED_LONG_LONG,
                           MPI_SUM,
                           comm,
                           "SparseMatrixSummaryScanner::u64_sum_batch");
    allReduceBufferInPlace(max_u64,
                           MPI_UNSIGNED_LONG_LONG,
                           MPI_MAX,
                           comm,
                           "SparseMatrixSummaryScanner::u64_max_batch");
    allReduceBufferInPlace(max_real,
                           MPI_DOUBLE,
                           MPI_MAX,
                           comm,
                           "SparseMatrixSummaryScanner::real_max_batch");
    allReduceBufferInPlace(min_real,
                           MPI_DOUBLE,
                           MPI_MIN,
                           comm,
                           "SparseMatrixSummaryScanner::real_min_batch");
    allReduceBufferInPlace(max_int,
                           MPI_INT,
                           MPI_MAX,
                           comm,
                           "SparseMatrixSummaryScanner::int_max_batch");
    allReduceBufferInPlace(min_int,
                           MPI_INT,
                           MPI_MIN,
                           comm,
                           "SparseMatrixSummaryScanner::int_min_batch");

    summary.diagonal_count = static_cast<std::uint64_t>(sum_u64[kDiagonalCount]);
    summary.missing_diagonal_count = static_cast<std::uint64_t>(sum_u64[kMissingDiagonalCount]);
    summary.nonpositive_diagonal_count = static_cast<std::uint64_t>(sum_u64[kNonpositiveDiagonalCount]);
    summary.negative_diagonal_count = static_cast<std::uint64_t>(sum_u64[kNegativeDiagonalCount]);
    summary.near_zero_diagonal_count = static_cast<std::uint64_t>(sum_u64[kNearZeroDiagonalCount]);
    summary.offdiag_count = static_cast<std::uint64_t>(sum_u64[kOffdiagCount]);
    summary.positive_offdiag_count = static_cast<std::uint64_t>(sum_u64[kPositiveOffdiagCount]);
    summary.negative_offdiag_count = static_cast<std::uint64_t>(sum_u64[kNegativeOffdiagCount]);
    summary.near_zero_offdiag_count = static_cast<std::uint64_t>(sum_u64[kNearZeroOffdiagCount]);
    summary.negative_row_sum_count = static_cast<std::uint64_t>(sum_u64[kNegativeRowSumCount]);
    summary.positive_row_sum_count = static_cast<std::uint64_t>(sum_u64[kPositiveRowSumCount]);
    summary.near_zero_row_sum_count = static_cast<std::uint64_t>(sum_u64[kNearZeroRowSumCount]);
    summary.row_sum_violation_count = static_cast<std::uint64_t>(sum_u64[kRowSumViolationCount]);
    summary.invalid_entry_count = static_cast<std::uint64_t>(sum_u64[kInvalidEntryCount]);
    summary.nonfinite_entry_count = static_cast<std::uint64_t>(sum_u64[kNonfiniteEntryCount]);
    summary.nonfinite_row_sum_count = static_cast<std::uint64_t>(sum_u64[kNonfiniteRowSumCount]);
    summary.scanned_row_count = static_cast<std::uint64_t>(sum_u64[kScannedRowCount]);
    summary.scanned_entry_count = static_cast<std::uint64_t>(sum_u64[kScannedEntryCount]);
    summary.duplicate_row_visit_count = static_cast<std::uint64_t>(sum_u64[kDuplicateRowVisitCount]);
    summary.missing_row_count = static_cast<std::uint64_t>(sum_u64[kMissingRowCount]);
    summary.expected_row_count = static_cast<std::uint64_t>(max_u64[kExpectedRows]);

    summary.max_abs_entry = max_real[kMaxAbsEntry];
    summary.max_abs_offdiag = max_real[kMaxAbsOffdiag];
    summary.max_positive_offdiag = max_real[kMaxPositiveOffdiag];
    summary.max_symmetry_error = max_real[kMaxSymmetryError];
    summary.max_hermitian_error = max_real[kMaxHermitianError];
    summary.max_complex_symmetry_error = max_real[kMaxComplexSymmetryError];
    summary.max_abs_imag_entry = max_real[kMaxAbsImagEntry];
    summary.max_abs_row_sum = max_real[kMaxAbsRowSum];

    if (max_int[kHasGershgorinLower] != 0) {
        summary.gershgorin_lower_bound = min_real[kGershgorinLower];
    } else {
        summary.gershgorin_lower_bound.reset();
    }
    if (max_int[kHasGershgorinUpper] != 0) {
        summary.gershgorin_upper_bound = max_real[kGershgorinUpper];
    } else {
        summary.gershgorin_upper_bound.reset();
    }

    if (sum_u64[kRowsForRowSumExtrema] == 0U) {
        summary.min_row_sum = 0.0;
        summary.max_row_sum = 0.0;
    } else {
        summary.min_row_sum = min_real[kMinRowSum];
        summary.max_row_sum = max_real[kMaxRowSum];
    }

    log.visited_rows = static_cast<std::uint64_t>(sum_u64[kLogVisitedRows]);
    log.visited_entries = static_cast<std::uint64_t>(sum_u64[kLogVisitedEntries]);
    log.visited_partitions = static_cast<std::uint64_t>(sum_u64[kLogVisitedPartitions]);
    log.max_row_entries = static_cast<std::size_t>(max_u64[kMaxRowEntries]);
    log.retained_symmetry_entries = static_cast<std::size_t>(sum_u64[kRetainedSymmetryEntries]);
    log.estimated_peak_stored_entries = static_cast<std::size_t>(max_u64[kPeakStoredEntries]);
    log.dense_matrix_materialized = max_int[kDenseMaterialized] != 0;
    log.symmetry_storage_truncated = max_int[kSymmetryTruncated] != 0;
    log.row_coverage_storage_truncated = max_int[kRowCoverageTruncated] != 0;

    summary.square = min_int[kAllSquare] != 0;
    summary.complex_values_present = max_int[kComplexValuesPresent] != 0;

    // A count-only MPI reduction cannot prove exact global row coverage because
    // duplicated rows and rows missing on all ranks can cancel in the count.
    summary.global_row_coverage_exact = false;
    summary.row_ownership_disjoint = summary.duplicate_row_visit_count == 0u;
    summary.sign_evidence_complete = false;
    summary.row_sum_evidence_complete = false;
    summary.symmetry_evidence_complete = false;
    summary.structurally_symmetric = false;
    summary.numerically_symmetric = false;
    summary.hermitian_evidence_complete = false;
    summary.structurally_hermitian = false;
    summary.numerically_hermitian = false;
    summary.complex_symmetry_evidence_complete = false;
    summary.structurally_complex_symmetric = false;
    summary.numerically_complex_symmetric = false;
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

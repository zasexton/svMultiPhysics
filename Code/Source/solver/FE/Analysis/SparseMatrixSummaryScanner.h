/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_SPARSE_MATRIX_SUMMARY_SCANNER_H
#define SVMP_FE_ANALYSIS_SPARSE_MATRIX_SUMMARY_SCANNER_H

/**
 * @file SparseMatrixSummaryScanner.h
 * @brief Backend-neutral sparse matrix scans for discrete-stability summaries.
 *
 * The scanner consumes sparse rows through a streaming interface and produces
 * the compact `DiscreteMatrixSummary` / `ReducedMatrixSummary` evidence used by
 * the analysis passes. The interface is intentionally independent of matrix
 * assembly ownership so native backends can provide their own row views without
 * materializing a dense matrix.
 */

#include "Analysis/AnalysisSummaryTypes.h"
#include "Core/FEConfig.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#if FE_HAS_MPI
#include <mpi.h>
#endif

namespace svmp {
namespace FE {
namespace backends {
class GenericMatrix;
} // namespace backends

namespace analysis {

struct SparseMatrixRowEntry {
    GlobalDofId col{INVALID_GLOBAL_INDEX};
    Real value{};
};

using SparseMatrixRowVisitor =
    std::function<void(GlobalDofId row,
                       const std::vector<SparseMatrixRowEntry>& entries,
                       int owning_rank)>;

/**
 * @brief Streaming sparse-row source used by numeric summary scans.
 */
class SparseRowScanSource {
public:
    virtual ~SparseRowScanSource() = default;

    [[nodiscard]] virtual backends::BackendKind backendKind() const noexcept = 0;
    [[nodiscard]] virtual GlobalIndex numRows() const noexcept = 0;
    [[nodiscard]] virtual GlobalIndex numCols() const noexcept = 0;

    /**
     * @brief True when this source visits every global row exactly once.
     *
     * Distributed sources normally return false. A scan over multiple source
     * partitions can still become globally complete when the combined row
     * coverage covers all rows.
     */
    [[nodiscard]] virtual bool hasCompleteGlobalRows() const noexcept = 0;

    /**
     * @brief True when the row source is a distributed local partition.
     */
    [[nodiscard]] virtual bool isDistributed() const noexcept {
        return !hasCompleteGlobalRows();
    }

    /**
     * @brief Visit locally stored sparse rows.
     *
     * Implementations may reuse the row-entry vector between callback calls.
     * Consumers must copy any entries they need to retain.
     */
    virtual void forEachLocalRow(const SparseMatrixRowVisitor& visitor) const = 0;
};

/**
 * @brief Simple CSR-backed row source for tests, adapters, and assembly scans.
 */
class CsrSparseRowScanSource final : public SparseRowScanSource {
public:
    CsrSparseRowScanSource(GlobalIndex rows,
                           GlobalIndex cols,
                           std::vector<GlobalIndex> row_ptr,
                           std::vector<GlobalIndex> col_indices,
                           std::vector<Real> values,
                           backends::BackendKind backend_kind = backends::BackendKind::Eigen,
                           GlobalDofId local_row_offset = 0,
                           bool complete_global_rows = true,
                           int owning_rank = 0);

    [[nodiscard]] static CsrSparseRowScanSource
    fromRows(GlobalIndex rows,
             GlobalIndex cols,
             const std::vector<std::vector<SparseMatrixRowEntry>>& row_entries,
             backends::BackendKind backend_kind = backends::BackendKind::Eigen,
             GlobalDofId local_row_offset = 0,
             bool complete_global_rows = true,
             int owning_rank = 0);

    [[nodiscard]] backends::BackendKind backendKind() const noexcept override {
        return backend_kind_;
    }
    [[nodiscard]] GlobalIndex numRows() const noexcept override { return rows_; }
    [[nodiscard]] GlobalIndex numCols() const noexcept override { return cols_; }
    [[nodiscard]] bool hasCompleteGlobalRows() const noexcept override {
        return complete_global_rows_;
    }
    [[nodiscard]] bool isDistributed() const noexcept override {
        return !complete_global_rows_ || owning_rank_ != 0;
    }

    void forEachLocalRow(const SparseMatrixRowVisitor& visitor) const override;

private:
    GlobalIndex rows_{0};
    GlobalIndex cols_{0};
    std::vector<GlobalIndex> row_ptr_;
    std::vector<GlobalIndex> col_indices_;
    std::vector<Real> values_;
    backends::BackendKind backend_kind_{backends::BackendKind::Eigen};
    GlobalDofId local_row_offset_{0};
    bool complete_global_rows_{true};
    int owning_rank_{0};
};

struct SparseMatrixScanOptions {
    Real sign_tolerance{0.0};
    Real row_sum_tolerance{0.0};
    Real symmetry_tolerance{0.0};
    std::size_t worst_entry_sample_limit{kDefaultWorstSampleLimit};
    std::size_t symmetry_entry_storage_limit{1000000};
    std::size_t row_coverage_storage_limit{5000000};
    bool compute_symmetry{true};

#if FE_HAS_MPI
    MPI_Comm mpi_comm{MPI_COMM_NULL};
#endif
};

struct SparseMatrixScanLog {
    std::uint64_t visited_rows{0};
    std::uint64_t visited_entries{0};
    std::uint64_t visited_partitions{0};
    std::size_t max_row_entries{0};
    std::size_t retained_symmetry_entries{0};
    std::size_t retained_row_coverage_entries{0};
    std::size_t estimated_peak_stored_entries{0};
    bool dense_matrix_materialized{false};
    bool symmetry_storage_truncated{false};
    bool row_coverage_storage_truncated{false};
    bool mpi_reduced{false};
    double elapsed_seconds{0.0};
    std::string message;
};

struct SparseMatrixSummaryResult {
    DiscreteMatrixSummary summary;
    SparseMatrixScanLog log;
};

struct ConstraintReductionMask {
    std::vector<GlobalDofId> free_dofs;
    std::vector<GlobalDofId> constrained_dofs;
    ConstraintReductionKind reduction_kind{ConstraintReductionKind::StrongDirichletElimination};
    std::uint64_t retained_multiplier_dof_count{0};
    bool affine_terms_accounted_for{false};
    bool reduction_exact_for_analysis{true};

    [[nodiscard]] static ConstraintReductionMask
    fromConstrainedDofs(GlobalIndex total_dofs,
                        std::vector<GlobalDofId> constrained_dofs,
                        ConstraintReductionKind reduction_kind =
                            ConstraintReductionKind::StrongDirichletElimination,
                        bool affine_terms_accounted_for = false,
                        bool reduction_exact_for_analysis = true);
};

[[nodiscard]] SparseMatrixSummaryResult
scanSparseMatrixSummary(const SparseRowScanSource& source,
                        OperatorBlockId block = {},
                        SparseMatrixScanOptions options = {});

[[nodiscard]] SparseMatrixSummaryResult
scanSparseMatrixSummary(const std::vector<const SparseRowScanSource*>& sources,
                        OperatorBlockId block = {},
                        SparseMatrixScanOptions options = {});

[[nodiscard]] ReducedMatrixSummary
scanReducedFreeFreeSummary(const SparseRowScanSource& source,
                           const ConstraintReductionMask& reduction,
                           OperatorBlockId block = {},
                           SparseMatrixScanOptions options = {});

[[nodiscard]] std::unique_ptr<SparseRowScanSource>
makeSparseRowScanSource(const backends::GenericMatrix& matrix);

[[nodiscard]] DiscreteMatrixSummary
mergeDiscreteMatrixSummaries(const std::vector<DiscreteMatrixSummary>& parts,
                             OperatorBlockId block = {},
                             SparseMatrixScanOptions options = {});

#if FE_HAS_MPI
void reduceDiscreteMatrixSummaryMPI(DiscreteMatrixSummary& summary,
                                    SparseMatrixScanLog& log,
                                    MPI_Comm comm);
#endif

[[nodiscard]] std::string formatSparseMatrixScanLog(const SparseMatrixScanLog& log);

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_SPARSE_MATRIX_SUMMARY_SCANNER_H

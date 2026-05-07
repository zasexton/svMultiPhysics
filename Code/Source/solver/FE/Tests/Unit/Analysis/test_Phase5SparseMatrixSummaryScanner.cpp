/**
 * @file test_Phase5SparseMatrixSummaryScanner.cpp
 * @brief Phase 5 tests for backend-neutral sparse matrix numeric summaries.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include "Analysis/SparseMatrixSummaryScanner.h"

using namespace svmp::FE;
using namespace svmp::FE::analysis;

namespace {

std::vector<std::vector<SparseMatrixRowEntry>> symmetricMMatrixRows()
{
    return {
        {{0, 2.0}, {1, -1.0}},
        {{0, -1.0}, {1, 2.0}, {2, -1.0}},
        {{1, -1.0}, {2, 2.0}},
    };
}

CsrSparseRowScanSource symmetricMMatrixSource(backends::BackendKind kind =
                                                 backends::BackendKind::Eigen)
{
    return CsrSparseRowScanSource::fromRows(3, 3, symmetricMMatrixRows(), kind);
}

std::vector<std::vector<SparseMatrixRowEntry>>
tridiagonalRows(GlobalIndex global_rows, GlobalIndex begin, GlobalIndex end)
{
    std::vector<std::vector<SparseMatrixRowEntry>> rows;
    rows.reserve(static_cast<std::size_t>(end - begin));
    for (GlobalIndex row = begin; row < end; ++row) {
        std::vector<SparseMatrixRowEntry> entries;
        if (row > 0) {
            entries.push_back({row - 1, -1.0});
        }
        entries.push_back({row, 2.0});
        if (row + 1 < global_rows) {
            entries.push_back({row + 1, -1.0});
        }
        rows.push_back(std::move(entries));
    }
    return rows;
}

void expectSameScalarSummary(const DiscreteMatrixSummary& a,
                             const DiscreteMatrixSummary& b)
{
    EXPECT_EQ(a.rows, b.rows);
    EXPECT_EQ(a.cols, b.cols);
    EXPECT_EQ(a.square, b.square);
    EXPECT_EQ(a.structurally_symmetric, b.structurally_symmetric);
    EXPECT_EQ(a.numerically_symmetric, b.numerically_symmetric);
    EXPECT_EQ(a.symmetry_evidence_complete, b.symmetry_evidence_complete);
    EXPECT_EQ(a.sign_evidence_complete, b.sign_evidence_complete);
    EXPECT_EQ(a.row_sum_evidence_complete, b.row_sum_evidence_complete);
    EXPECT_DOUBLE_EQ(a.max_abs_entry, b.max_abs_entry);
    EXPECT_DOUBLE_EQ(a.max_abs_offdiag, b.max_abs_offdiag);
    EXPECT_DOUBLE_EQ(a.max_positive_offdiag, b.max_positive_offdiag);
    EXPECT_DOUBLE_EQ(a.max_symmetry_error, b.max_symmetry_error);
    EXPECT_DOUBLE_EQ(a.min_row_sum, b.min_row_sum);
    EXPECT_DOUBLE_EQ(a.max_row_sum, b.max_row_sum);
    EXPECT_DOUBLE_EQ(a.max_abs_row_sum, b.max_abs_row_sum);
    EXPECT_EQ(a.diagonal_count, b.diagonal_count);
    EXPECT_EQ(a.nonpositive_diagonal_count, b.nonpositive_diagonal_count);
    EXPECT_EQ(a.negative_diagonal_count, b.negative_diagonal_count);
    EXPECT_EQ(a.near_zero_diagonal_count, b.near_zero_diagonal_count);
    EXPECT_EQ(a.offdiag_count, b.offdiag_count);
    EXPECT_EQ(a.positive_offdiag_count, b.positive_offdiag_count);
    EXPECT_EQ(a.negative_offdiag_count, b.negative_offdiag_count);
    EXPECT_EQ(a.near_zero_offdiag_count, b.near_zero_offdiag_count);
    EXPECT_EQ(a.row_sum_violation_count, b.row_sum_violation_count);
    EXPECT_EQ(a.invalid_entry_count, b.invalid_entry_count);
    EXPECT_EQ(a.nonfinite_entry_count, b.nonfinite_entry_count);
    EXPECT_EQ(a.nonfinite_row_sum_count, b.nonfinite_row_sum_count);
    EXPECT_EQ(a.scanned_row_count, b.scanned_row_count);
    EXPECT_EQ(a.expected_row_count, b.expected_row_count);
    EXPECT_EQ(a.scanned_entry_count, b.scanned_entry_count);
}

} // namespace

TEST(Phase5SparseMatrixSummaryScanner, BackendNeutralCsrScanBuildsMatrixSignSummary)
{
    const auto source = symmetricMMatrixSource();
    SparseMatrixScanOptions options;
    options.sign_tolerance = 1.0e-12;
    options.row_sum_tolerance = 1.0e-12;
    options.symmetry_tolerance = 1.0e-12;

    const auto result = scanSparseMatrixSummary(source, {}, options);
    const auto& summary = result.summary;

    ASSERT_TRUE(summary.backend_kind.has_value());
    EXPECT_EQ(*summary.backend_kind, backends::BackendKind::Eigen);
    EXPECT_EQ(summary.rows, 3);
    EXPECT_EQ(summary.cols, 3);
    EXPECT_TRUE(summary.square);
    EXPECT_TRUE(summary.structurally_symmetric);
    EXPECT_TRUE(summary.numerically_symmetric);
    EXPECT_TRUE(summary.symmetry_evidence_complete);
    EXPECT_TRUE(summary.sign_evidence_complete);
    EXPECT_TRUE(summary.row_sum_evidence_complete);
    EXPECT_EQ(summary.diagonal_count, 3u);
    EXPECT_EQ(summary.offdiag_count, 4u);
    EXPECT_EQ(summary.positive_offdiag_count, 0u);
    EXPECT_EQ(summary.negative_offdiag_count, 4u);
    EXPECT_EQ(summary.row_sum_violation_count, 0u);
    EXPECT_EQ(summary.scanned_row_count, 3u);
    EXPECT_EQ(summary.expected_row_count, 3u);
    EXPECT_EQ(summary.scanned_entry_count, 7u);
    EXPECT_TRUE(summary.gershgorin_lower_bound.has_value());
    EXPECT_FALSE(summary.min_eigenvalue_estimate.has_value());
    EXPECT_FALSE(summary.coercivity_lower_bound.has_value());
    EXPECT_DOUBLE_EQ(summary.min_row_sum, 0.0);
    EXPECT_DOUBLE_EQ(summary.max_row_sum, 1.0);
    EXPECT_DOUBLE_EQ(summary.max_abs_row_sum, 1.0);

    EXPECT_EQ(result.log.visited_rows, 3u);
    EXPECT_EQ(result.log.visited_entries, 7u);
    EXPECT_FALSE(result.log.dense_matrix_materialized);
    EXPECT_FALSE(result.log.symmetry_storage_truncated);
    EXPECT_EQ(result.log.retained_symmetry_entries, 7u);
    EXPECT_NE(result.log.message.find("dense_matrix_materialized=false"),
              std::string::npos);
}

TEST(Phase5SparseMatrixSummaryScanner, DuplicateEntriesAreSummedBeforeSignClassification)
{
    const std::vector<std::vector<SparseMatrixRowEntry>> rows{
        {{0, 2.0}, {1, 1.0}, {1, -2.0}},
        {{0, -1.0}, {1, 2.0}},
    };

    SparseMatrixScanOptions options;
    options.sign_tolerance = 1.0e-12;
    options.row_sum_tolerance = 1.0e-12;
    options.symmetry_tolerance = 1.0e-12;

    const auto result = scanSparseMatrixSummary(
        CsrSparseRowScanSource::fromRows(2, 2, rows), {}, options);
    const auto& summary = result.summary;

    EXPECT_TRUE(summary.sign_evidence_complete);
    EXPECT_EQ(summary.scanned_entry_count, 5u);
    EXPECT_EQ(summary.offdiag_count, 2u);
    EXPECT_EQ(summary.positive_offdiag_count, 0u);
    EXPECT_EQ(summary.negative_offdiag_count, 2u);
    EXPECT_DOUBLE_EQ(summary.max_positive_offdiag, 0.0);
}

TEST(Phase5SparseMatrixSummaryScanner, InvalidColumnInvalidatesCertificationEvidence)
{
    const std::vector<std::vector<SparseMatrixRowEntry>> rows{
        {{0, 2.0}, {3, -1.0}},
        {{0, -1.0}, {1, 2.0}},
    };

    SparseMatrixScanOptions options;
    options.sign_tolerance = 1.0e-12;
    options.row_sum_tolerance = 1.0e-12;
    options.symmetry_tolerance = 1.0e-12;

    const auto result = scanSparseMatrixSummary(
        CsrSparseRowScanSource::fromRows(2, 2, rows), {}, options);
    const auto& summary = result.summary;

    EXPECT_EQ(summary.invalid_entry_count, 1u);
    EXPECT_FALSE(summary.sign_evidence_complete);
    EXPECT_FALSE(summary.row_sum_evidence_complete);
    EXPECT_FALSE(summary.symmetry_evidence_complete);
    ASSERT_FALSE(summary.worst_entries.empty());
    EXPECT_NE(std::find_if(summary.worst_entries.begin(),
                           summary.worst_entries.end(),
                           [](const MatrixEntrySample& sample) {
                               return sample.note == "invalid column index";
                           }),
              summary.worst_entries.end());
}

TEST(Phase5SparseMatrixSummaryScanner, IncompleteRowsDoNotCertifyEvenIfSourceClaimsComplete)
{
    const std::vector<std::vector<SparseMatrixRowEntry>> rows{
        {{0, 2.0}},
        {{1, 2.0}},
    };

    SparseMatrixScanOptions options;
    options.sign_tolerance = 1.0e-12;
    options.row_sum_tolerance = 1.0e-12;
    options.symmetry_tolerance = 1.0e-12;

    const auto result = scanSparseMatrixSummary(
        CsrSparseRowScanSource::fromRows(
            3, 3, rows, backends::BackendKind::Eigen, 0,
            /*complete_global_rows=*/true),
        {},
        options);

    EXPECT_EQ(result.summary.scanned_row_count, 2u);
    EXPECT_EQ(result.summary.expected_row_count, 3u);
    EXPECT_FALSE(result.summary.sign_evidence_complete);
    EXPECT_FALSE(result.summary.row_sum_evidence_complete);
    EXPECT_FALSE(result.summary.symmetry_evidence_complete);
}

TEST(Phase5SparseMatrixSummaryScanner, SignSummariesRecordBackendKindForAllBackends)
{
    const std::vector<backends::BackendKind> backend_kinds{
        backends::BackendKind::Eigen,
        backends::BackendKind::FSILS,
        backends::BackendKind::PETSc,
        backends::BackendKind::Trilinos,
    };
    const std::vector<std::vector<SparseMatrixRowEntry>> rows{
        {{0, 1.0}, {1, 0.5}},
        {{0, -0.25}, {1, 2.0}},
    };

    SparseMatrixScanOptions options;
    options.sign_tolerance = 1.0e-12;
    options.symmetry_tolerance = 1.0e-12;
    options.worst_entry_sample_limit = 1;

    for (const auto kind : backend_kinds) {
        const auto source = CsrSparseRowScanSource::fromRows(2, 2, rows, kind);
        const auto result = scanSparseMatrixSummary(source, {}, options);
        ASSERT_TRUE(result.summary.backend_kind.has_value());
        EXPECT_EQ(*result.summary.backend_kind, kind);
        EXPECT_EQ(result.summary.positive_offdiag_count, 1u);
        EXPECT_TRUE(result.summary.sign_evidence_complete);
        EXPECT_TRUE(result.summary.row_sum_evidence_complete);
        EXPECT_DOUBLE_EQ(result.summary.max_positive_offdiag, 0.5);
        EXPECT_FALSE(result.summary.numerically_symmetric);
        EXPECT_EQ(result.summary.worst_entries.size(), 1u);
        EXPECT_EQ(result.summary.worst_entries.front().note, "positive offdiagonal");
    }
}

TEST(Phase5SparseMatrixSummaryScanner, NonfiniteEntriesInvalidateNumericEvidence)
{
    const std::vector<std::vector<SparseMatrixRowEntry>> rows{
        {{0, 2.0}, {1, -1.0}},
        {{0, std::numeric_limits<Real>::quiet_NaN()}, {1, 2.0}},
    };

    SparseMatrixScanOptions options;
    options.sign_tolerance = 1.0e-12;
    options.row_sum_tolerance = 1.0e-12;
    options.symmetry_tolerance = 1.0e-12;
    options.worst_entry_sample_limit = 2;

    const auto result = scanSparseMatrixSummary(
        CsrSparseRowScanSource::fromRows(2, 2, rows), {}, options);
    const auto& summary = result.summary;

    EXPECT_EQ(summary.nonfinite_entry_count, 1u);
    EXPECT_EQ(summary.nonfinite_row_sum_count, 1u);
    EXPECT_FALSE(summary.sign_evidence_complete);
    EXPECT_FALSE(summary.row_sum_evidence_complete);
    ASSERT_FALSE(summary.worst_entries.empty());
    EXPECT_EQ(summary.worst_entries.front().note, "nonfinite entry");
}

TEST(Phase5SparseMatrixSummaryScanner, ReducedFreeFreeScanEliminatesConstrainedRowsAndColumns)
{
    const std::vector<std::vector<SparseMatrixRowEntry>> rows{
        {{0, 2.0}, {1, -1.0}},
        {{0, -1.0}, {1, 2.0}, {2, -1.0}},
        {{1, -1.0}, {2, 2.0}, {3, -1.0}},
        {{2, -1.0}, {3, 2.0}},
    };
    const auto source = CsrSparseRowScanSource::fromRows(4, 4, rows);
    const auto mask = ConstraintReductionMask::fromConstrainedDofs(
        4, {0, 3}, ConstraintReductionKind::StrongDirichletElimination);

    SparseMatrixScanOptions options;
    options.sign_tolerance = 1.0e-12;
    options.row_sum_tolerance = 1.0e-12;
    options.symmetry_tolerance = 1.0e-12;

    const auto reduced = scanReducedFreeFreeSummary(source, mask, {}, options);
    const auto& matrix = reduced.free_free_matrix;

    EXPECT_EQ(reduced.reduction_kind,
              ConstraintReductionKind::StrongDirichletElimination);
    EXPECT_EQ(reduced.free_dof_count, 2u);
    EXPECT_EQ(reduced.constrained_dof_count, 2u);
    EXPECT_TRUE(reduced.reduction_exact_for_analysis);
    EXPECT_EQ(matrix.scope, NumericSummaryScope::ReducedFreeFree);
    EXPECT_EQ(matrix.rows, 2);
    EXPECT_EQ(matrix.cols, 2);
    EXPECT_EQ(matrix.diagonal_count, 2u);
    EXPECT_EQ(matrix.offdiag_count, 2u);
    EXPECT_EQ(matrix.negative_offdiag_count, 2u);
    EXPECT_EQ(matrix.positive_offdiag_count, 0u);
    EXPECT_TRUE(matrix.structurally_symmetric);
    EXPECT_TRUE(matrix.numerically_symmetric);
    EXPECT_TRUE(matrix.symmetry_evidence_complete);
    EXPECT_TRUE(matrix.sign_evidence_complete);
    EXPECT_TRUE(matrix.row_sum_evidence_complete);
}

TEST(Phase5SparseMatrixSummaryScanner, PartitionedScanMatchesSerialScan)
{
    constexpr GlobalIndex rows = 4;
    const auto full_rows = tridiagonalRows(rows, 0, rows);
    const auto source = CsrSparseRowScanSource::fromRows(rows, rows, full_rows);

    const auto part0 = CsrSparseRowScanSource::fromRows(
        rows,
        rows,
        tridiagonalRows(rows, 0, 2),
        backends::BackendKind::FSILS,
        0,
        false,
        0);
    const auto part1 = CsrSparseRowScanSource::fromRows(
        rows,
        rows,
        tridiagonalRows(rows, 2, 4),
        backends::BackendKind::FSILS,
        2,
        false,
        1);

    SparseMatrixScanOptions options;
    options.sign_tolerance = 1.0e-12;
    options.row_sum_tolerance = 1.0e-12;
    options.symmetry_tolerance = 1.0e-12;

    const auto serial = scanSparseMatrixSummary(source, {}, options);
    const auto partitioned =
        scanSparseMatrixSummary(std::vector<const SparseRowScanSource*>{&part0, &part1},
                                {},
                                options);

    expectSameScalarSummary(serial.summary, partitioned.summary);
    EXPECT_EQ(partitioned.log.visited_partitions, 2u);
    EXPECT_EQ(partitioned.log.visited_rows, 4u);
    EXPECT_EQ(partitioned.log.visited_entries, serial.log.visited_entries);
    EXPECT_FALSE(partitioned.log.dense_matrix_materialized);
}

TEST(Phase5SparseMatrixSummaryScanner, LargePartitionedScanBoundsStorageAndLogsOverhead)
{
    constexpr GlobalIndex rows = 2000;
    const auto part0 = CsrSparseRowScanSource::fromRows(
        rows,
        rows,
        tridiagonalRows(rows, 0, rows / 2),
        backends::BackendKind::FSILS,
        0,
        false,
        0);
    const auto part1 = CsrSparseRowScanSource::fromRows(
        rows,
        rows,
        tridiagonalRows(rows, rows / 2, rows),
        backends::BackendKind::FSILS,
        rows / 2,
        false,
        1);

    SparseMatrixScanOptions options;
    options.sign_tolerance = 1.0e-12;
    options.row_sum_tolerance = 1.0e-12;
    options.symmetry_tolerance = 1.0e-12;
    options.symmetry_entry_storage_limit = 64;
    options.row_coverage_storage_limit = static_cast<std::size_t>(rows);
    options.worst_entry_sample_limit = 2;

    const auto result =
        scanSparseMatrixSummary(std::vector<const SparseRowScanSource*>{&part0, &part1},
                                {},
                                options);

    EXPECT_EQ(result.log.visited_rows, static_cast<std::uint64_t>(rows));
    EXPECT_EQ(result.log.visited_entries,
              static_cast<std::uint64_t>(3 * rows - 2));
    EXPECT_EQ(result.log.visited_partitions, 2u);
    EXPECT_FALSE(result.log.dense_matrix_materialized);
    EXPECT_TRUE(result.log.symmetry_storage_truncated);
    EXPECT_LE(result.log.retained_symmetry_entries,
              options.symmetry_entry_storage_limit);
    EXPECT_LE(result.summary.worst_entries.size(), options.worst_entry_sample_limit);
    EXPECT_FALSE(result.summary.symmetry_evidence_complete);
    EXPECT_TRUE(result.summary.sign_evidence_complete);
    EXPECT_TRUE(result.summary.row_sum_evidence_complete);
    EXPECT_NE(result.log.message.find("estimated_peak_stored_entries="),
              std::string::npos);
}

#if FE_HAS_MPI
TEST(Phase5SparseMatrixSummaryScanner, MpiReductionMatchesSerialSmallProblem)
{
    if (std::getenv("SVMP_FE_RUN_MPI_TESTS") == nullptr) {
        GTEST_SKIP() << "Set SVMP_FE_RUN_MPI_TESTS=1 and run under mpiexec.";
    }

    struct MpiSession {
        bool owns_initialization{false};
        ~MpiSession()
        {
            int finalized = 0;
            MPI_Finalized(&finalized);
            if (owns_initialization && finalized == 0) {
                MPI_Finalize();
            }
        }
    } session;

    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized == 0) {
        int argc = 0;
        char** argv = nullptr;
        int provided = MPI_THREAD_SINGLE;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
        session.owns_initialization = true;
    }

    int size = 1;
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (size != 2) {
        GTEST_SKIP() << "This MPI reduction test expects exactly two ranks.";
    }

    constexpr GlobalIndex rows = 4;
    const GlobalIndex begin = (rank == 0) ? 0 : 2;
    const GlobalIndex end = (rank == 0) ? 2 : 4;

    const auto local_source = CsrSparseRowScanSource::fromRows(
        rows,
        rows,
        tridiagonalRows(rows, begin, end),
        backends::BackendKind::FSILS,
        begin,
        false,
        rank);
    const auto serial_source = CsrSparseRowScanSource::fromRows(
        rows,
        rows,
        tridiagonalRows(rows, 0, rows),
        backends::BackendKind::FSILS);

    SparseMatrixScanOptions options;
    options.sign_tolerance = 1.0e-12;
    options.row_sum_tolerance = 1.0e-12;
    options.compute_symmetry = false;

    auto serial = scanSparseMatrixSummary(serial_source, {}, options);
    options.mpi_comm = MPI_COMM_WORLD;
    auto mpi = scanSparseMatrixSummary(local_source, {}, options);

    expectSameScalarSummary(serial.summary, mpi.summary);
    EXPECT_EQ(mpi.log.visited_rows, static_cast<std::uint64_t>(rows));
    EXPECT_EQ(mpi.log.visited_entries, serial.log.visited_entries);
    EXPECT_EQ(mpi.log.visited_partitions, 2u);
    EXPECT_TRUE(mpi.log.mpi_reduced);
    EXPECT_FALSE(mpi.log.dense_matrix_materialized);

    auto serial_bad_rows = tridiagonalRows(rows, 0, rows);
    serial_bad_rows[2][0].value = 0.75;
    auto local_bad_rows = tridiagonalRows(rows, begin, end);
    if (rank == 1) {
        local_bad_rows[0][0].value = 0.75;
    }

    const auto serial_bad_source = CsrSparseRowScanSource::fromRows(
        rows,
        rows,
        serial_bad_rows,
        backends::BackendKind::FSILS);
    const auto local_bad_source = CsrSparseRowScanSource::fromRows(
        rows,
        rows,
        local_bad_rows,
        backends::BackendKind::FSILS,
        begin,
        false,
        rank);

    options.worst_entry_sample_limit = 1;
    options.mpi_comm = MPI_COMM_NULL;
    auto serial_bad = scanSparseMatrixSummary(serial_bad_source, {}, options);
    options.mpi_comm = MPI_COMM_WORLD;
    auto mpi_bad = scanSparseMatrixSummary(local_bad_source, {}, options);

    EXPECT_EQ(mpi_bad.summary.positive_offdiag_count,
              serial_bad.summary.positive_offdiag_count);
    ASSERT_EQ(mpi_bad.summary.worst_entries.size(), 1u);
    EXPECT_EQ(mpi_bad.summary.worst_entries.front().row, 2);
    EXPECT_EQ(mpi_bad.summary.worst_entries.front().col, 1);
    EXPECT_DOUBLE_EQ(mpi_bad.summary.worst_entries.front().value, 0.75);
    EXPECT_EQ(mpi_bad.summary.worst_entries.front().note, "positive offdiagonal");
}
#endif

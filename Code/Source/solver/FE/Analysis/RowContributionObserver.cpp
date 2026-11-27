/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/RowContributionObserver.h"

#include "Assembly/GlobalSystemView.h"
#include "Backends/Interfaces/BackendFactory.h"
#include "Backends/Interfaces/GenericMatrix.h"
#include "Backends/Interfaces/GenericVector.h"
#include "Core/FEException.h"
#include "Core/Logger.h"
#include "Dofs/DofIndexSet.h"
#include "Systems/FESystem.h"
#include "Systems/SystemsExceptions.h"
#include "Systems/TransientSystem.h"
#include "Core/EnvironmentNames.h"

#if defined(FE_HAS_FSILS)
#  include "Backends/FSILS/FsilsVector.h"
#endif

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#if FE_HAS_MPI
#  include <mpi.h>
#endif

namespace svmp {
namespace FE {
namespace analysis {

std::vector<GlobalIndex> parseRowSampleDofs(std::string_view text)
{
    std::vector<GlobalIndex> out;
    std::string normalized(text);
    std::replace(normalized.begin(), normalized.end(), ',', '|');
    std::size_t begin = 0u;
    while (begin < normalized.size()) {
        const auto end = normalized.find('|', begin);
        const auto token =
            normalized.substr(begin, end == std::string::npos
                                         ? std::string::npos
                                         : end - begin);
        const auto dash = token.find('-');
        try {
            if (dash != std::string::npos) {
                const auto first =
                    static_cast<GlobalIndex>(std::stoll(token.substr(0, dash)));
                const auto last =
                    static_cast<GlobalIndex>(std::stoll(token.substr(dash + 1)));
                const auto step = (last >= first) ? GlobalIndex{1} : GlobalIndex{-1};
                for (GlobalIndex value = first;; value += step) {
                    out.push_back(value);
                    if (value == last) {
                        break;
                    }
                }
            } else if (!token.empty()) {
                out.push_back(static_cast<GlobalIndex>(std::stoll(token)));
            }
        } catch (const std::exception&) {
            // Ignore malformed diagnostic tokens; the surrounding diagnostic is
            // intentionally best-effort.
        }
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1u;
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

namespace {

#if FE_HAS_MPI
using NewtonCommunicator = MPI_Comm;
#else
using NewtonCommunicator = int;
constexpr NewtonCommunicator kSerialNewtonCommunicator = 0;
#endif

using timestepping::stateSynchronizationPointName;

[[nodiscard]] const char* stateSyncPointName(
    timestepping::NewtonOptions::StateSynchronizationPoint point) noexcept
{
    return stateSynchronizationPointName(point);
}

[[nodiscard]] const char* numericalSummaryOperators() noexcept
{
    return environment::value("SVMP_FE_ROW_CONTRIBUTION_MATRIX_SUMMARY_OPS");
}

[[nodiscard]] bool rowMatrixSummaryOpSelected(const RowObservationOptions& options, std::string_view op)
{
    const char* env = options.summary_operators ? options.summary_operators() : nullptr;
    if (env == nullptr || env[0] == '\0') {
        return true;
    }

    std::string selected(env);
    std::replace(selected.begin(), selected.end(), ',', '|');
    std::replace(selected.begin(), selected.end(), ';', '|');
    std::size_t begin = 0u;
    while (begin < selected.size()) {
        const auto end = selected.find('|', begin);
        std::string token =
            selected.substr(begin, end == std::string::npos
                                       ? std::string::npos
                                       : end - begin);
        token.erase(token.begin(),
                    std::find_if(token.begin(), token.end(), [](unsigned char c) {
                        return !std::isspace(c);
                    }));
        token.erase(std::find_if(token.rbegin(), token.rend(), [](unsigned char c) {
                        return !std::isspace(c);
                    }).base(),
                    token.end());
        if (token == "*" || token == op ||
            (!token.empty() &&
             op.find(std::string_view(token)) != std::string_view::npos)) {
            return true;
        }
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1u;
    }
    return false;
}

struct NewtonMatrixSupportFieldRange {
    std::string name;
    GlobalIndex begin = INVALID_GLOBAL_INDEX;
    GlobalIndex end = INVALID_GLOBAL_INDEX;
};

[[nodiscard]] int mpiRank() noexcept
{
#if FE_HAS_MPI
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    MPI_Finalized(&finalized);
    if (!initialized || finalized) {
        return 0;
    }
    int rank = 0;
    (void)MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
#else
    return 0;
#endif
}

[[nodiscard]] int communicatorRank(NewtonCommunicator communicator) noexcept
{
#if FE_HAS_MPI
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    MPI_Finalized(&finalized);
    if (!initialized || finalized || communicator == MPI_COMM_NULL) {
        return 0;
    }
    int rank = 0;
    (void)MPI_Comm_rank(communicator, &rank);
    return rank;
#else
    (void)communicator;
    return 0;
#endif
}

[[nodiscard]] NewtonCommunicator systemCommunicator(
    const systems::FESystem& system) noexcept
{
#if FE_HAS_MPI
    return system.activeMpiCommunicator();
#else
    (void)system;
    return kSerialNewtonCommunicator;
#endif
}

[[nodiscard]] std::vector<GlobalIndex> ownedDofsForVector(
    const backends::GenericVector& vec,
    const dofs::IndexSet& fe_owned_dofs)
{
#if defined(FE_HAS_FSILS)
    if (const auto* fs = dynamic_cast<const backends::FsilsVector*>(&vec);
        fs != nullptr && fs->usesOwnedRowLayout()) {
        return fs->ownedFeDofs();
    }
#endif
    return fe_owned_dofs.toVector();
}

void logNewtonFieldResidualDiagnostic(
    const RowObservationOptions& options,
    const systems::FESystem& sys,
    backends::GenericVector& residual,
    std::string_view phase,
    timestepping::NewtonOptions::StateSynchronizationPoint sync_point,
    int iteration,
    double solve_time,
    double dt)
{
    const auto communicator = systemCommunicator(sys);

    const auto& field_name = options.residual_field;
    const auto field_id = sys.findFieldByName(field_name);
    if (field_id == INVALID_FIELD_ID) {
        if (communicatorRank(communicator) == 0) {
            FE_LOG_INFO(
                "NewtonSolver: field residual diagnostic skipped "
                "diagnostic=newton_field_residual_skipped reason=field_not_found "
                "field='" + field_name + "'");
        }
        return;
    }

    const auto field_offset = sys.fieldDofOffset(field_id);
    const auto field_dofs = sys.fieldDofHandler(field_id).getNumDofs();
    if (field_offset < 0 || field_dofs <= 0) {
        if (communicatorRank(communicator) == 0) {
            std::ostringstream oss;
            oss << "NewtonSolver: field residual diagnostic skipped"
                << " diagnostic=newton_field_residual_skipped"
                << " reason=invalid_field_range"
                << " field='" << field_name << "'"
                << " field_offset=" << field_offset
                << " field_dofs=" << field_dofs;
            FE_LOG_INFO(oss.str());
        }
        return;
    }

    const auto owned_dofs =
        ownedDofsForVector(residual, sys.dofHandler().getPartition().locallyOwned());
    auto view = residual.createAssemblyView();
    FE_CHECK_NOT_NULL(view.get(), "NewtonSolver: field residual diagnostic view");

    const auto field_begin = field_offset;
    const auto field_end = field_offset + field_dofs;
    long double local_sq = 0.0L;
    double local_sum = 0.0;
    double local_min = std::numeric_limits<double>::infinity();
    double local_max = -std::numeric_limits<double>::infinity();
    double local_max_abs = 0.0;
    double local_worst_value = 0.0;
    GlobalIndex local_worst_dof = INVALID_GLOBAL_INDEX;
    unsigned long long local_count = 0ull;

    for (const auto dof : owned_dofs) {
        if (dof < field_begin || dof >= field_end) {
            continue;
        }
        const double value = static_cast<double>(view->getVectorEntry(dof));
        local_sq += static_cast<long double>(value) *
                    static_cast<long double>(value);
        local_sum += value;
        local_min = std::min(local_min, value);
        local_max = std::max(local_max, value);
        ++local_count;
        const double abs_value = std::abs(value);
        if (abs_value > local_max_abs) {
            local_max_abs = abs_value;
            local_worst_value = value;
            local_worst_dof = dof;
        }
    }

    double global_sq = static_cast<double>(local_sq);
    double global_sum = local_sum;
    double global_min = local_min;
    double global_max = local_max;
    double global_max_abs = local_max_abs;
    unsigned long long global_count = local_count;

#if FE_HAS_MPI
    int mpi_initialized = 0;
    int mpi_finalized = 0;
    MPI_Initialized(&mpi_initialized);
    MPI_Finalized(&mpi_finalized);
    if (mpi_initialized && !mpi_finalized &&
        communicator != MPI_COMM_NULL) {
        const double local_sq_double = static_cast<double>(local_sq);
        MPI_Allreduce(&local_sq_double, &global_sq, 1, MPI_DOUBLE, MPI_SUM,
                      communicator);
        MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
                      communicator);
        MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN,
                      communicator);
        MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX,
                      communicator);
        MPI_Allreduce(&local_max_abs, &global_max_abs, 1, MPI_DOUBLE, MPI_MAX,
                      communicator);
        MPI_Allreduce(&local_count, &global_count, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator);
    }
#endif

    if (global_count == 0ull) {
        global_min = 0.0;
        global_max = 0.0;
    }
    const double global_norm = std::sqrt(std::max(0.0, global_sq));
    const double global_mean =
        global_count > 0ull ? global_sum / static_cast<double>(global_count)
                            : 0.0;

    std::ostringstream oss;
    oss << "NewtonSolver: field residual diagnostic"
        << " diagnostic=newton_field_residual"
        << " rank=" << communicatorRank(communicator)
        << " iteration=" << iteration
        << " phase='" << phase << "'"
        << " sync_point=" << stateSyncPointName(sync_point)
        << " field='" << field_name << "'"
        << " field_offset=" << field_offset
        << " field_dofs=" << field_dofs
        << " solve_time=" << solve_time
        << " dt=" << dt
        << " owned_field_dofs=" << global_count
        << " norm=" << global_norm
        << " mean=" << global_mean
        << " min=" << global_min
        << " max=" << global_max
        << " global_max_abs=" << global_max_abs
        << " local_worst_dof=" << local_worst_dof
        << " local_worst_value=" << local_worst_value
        << " local_max_abs=" << local_max_abs;

    const auto& sample_dofs = options.residual_samples;
    if (!sample_dofs.empty()) {
        oss << " sampled_dofs=";
        bool emitted = false;
        for (const auto dof : sample_dofs) {
            if (dof < 0 || dof >= residual.size()) {
                continue;
            }
            if (emitted) {
                oss << "|";
            }
            oss << dof << ":" << static_cast<double>(view->getVectorEntry(dof));
            emitted = true;
        }
        if (!emitted) {
            oss << "none";
        }
    }
    FE_LOG_INFO(oss.str());
}

std::string firstNonzeroMatrixIndices(const backends::GenericMatrix& matrix,
                                      GlobalIndex fixed_index,
                                      bool scan_row,
                                      Real tiny,
                                      int limit = 12)
{
    const GlobalIndex extent = scan_row ? matrix.numCols() : matrix.numRows();
    std::ostringstream oss;
    int emitted = 0;
    for (GlobalIndex i = 0; i < extent && emitted < limit; ++i) {
        const Real value =
            scan_row ? matrix.getEntry(fixed_index, i) : matrix.getEntry(i, fixed_index);
        if (std::abs(value) <= tiny || !std::isfinite(value)) {
            continue;
        }
        if (emitted > 0) {
            oss << '|';
        }
        oss << i << ':' << static_cast<double>(value);
        ++emitted;
    }
    if (emitted == 0) {
        return "none";
    }
    return oss.str();
}

[[nodiscard]] std::optional<NewtonMatrixSupportFieldRange>
newtonMatrixSupportFieldRangeByName(const systems::FESystem& sys,
                                    std::string_view field_name)
{
    for (const auto field : sys.unknownFieldIdsInDofMapOrder()) {
        const auto& rec = sys.fieldRecord(field);
        if (rec.name != field_name) {
            continue;
        }
        const auto begin = sys.fieldDofOffset(field);
        const auto dofs = sys.fieldDofHandler(field).getNumDofs();
        if (begin < 0 || dofs <= 0) {
            return std::nullopt;
        }
        return NewtonMatrixSupportFieldRange{
            rec.name,
            begin,
            begin + dofs,
        };
    }
    return std::nullopt;
}

[[nodiscard]] bool dofInFieldRange(GlobalIndex dof,
                                   const NewtonMatrixSupportFieldRange& range) noexcept
{
    return dof >= range.begin && dof < range.end;
}

struct MatrixSupportRow {
    GlobalIndex local_dof = INVALID_GLOBAL_INDEX;
    GlobalIndex global_dof = INVALID_GLOBAL_INDEX;
    double row_abs_sum = 0.0;
    double col_abs_sum = 0.0;
    double row_coupling_abs_sum = 0.0;
    double col_coupling_abs_sum = 0.0;
    double row_self_abs_sum = 0.0;
    double col_self_abs_sum = 0.0;
    double row_self_sum = 0.0;
    double row_self_offdiag_abs_sum = 0.0;
    double row_self_signed_abs_ratio = 0.0;
    double row_self_diag_abs_ratio = 0.0;
    double diag = 0.0;
};

struct MatrixSupportSummary {
    std::string observed_field;
    std::string coupling_field;
    GlobalIndex observed_offset = INVALID_GLOBAL_INDEX;
    GlobalIndex observed_dofs = 0;
    GlobalIndex coupling_offset = INVALID_GLOBAL_INDEX;
    GlobalIndex coupling_dofs = 0;
    GlobalIndex constrained_observed_rows = 0;
    GlobalIndex unconstrained_observed_rows = 0;
    GlobalIndex zero_row_count = 0;
    GlobalIndex zero_col_count = 0;
    GlobalIndex zero_diag_count = 0;
    GlobalIndex zero_coupling_row_block_count = 0;
    GlobalIndex zero_coupling_col_block_count = 0;
    GlobalIndex zero_self_row_block_count = 0;
    GlobalIndex zero_self_col_block_count = 0;
    GlobalIndex positive_coupling_row_block_count = 0;
    GlobalIndex positive_self_row_block_count = 0;
    GlobalIndex observed_only_row_block_count = 0;
    GlobalIndex observed_only_col_block_count = 0;
    double min_positive_coupling_row_abs_sum =
        std::numeric_limits<double>::infinity();
    double max_coupling_row_abs_sum = 0.0;
    double min_positive_self_row_abs_sum =
        std::numeric_limits<double>::infinity();
    double max_self_row_abs_sum = 0.0;
    std::vector<GlobalIndex> zero_coupling_row_global_dofs;
    std::vector<MatrixSupportRow> zero_coupling_row_samples;
    std::vector<MatrixSupportRow> zero_row_samples;
    std::vector<MatrixSupportRow> weakest_coupling_row_samples;
    std::vector<MatrixSupportRow> weakest_self_row_samples;
};

[[nodiscard]] std::string formatMatrixSupportDofSamples(
    std::span<const MatrixSupportRow> rows,
    bool local)
{
    if (rows.empty()) {
        return "none";
    }
    std::ostringstream oss;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        if (i > 0u) {
            oss << '|';
        }
        oss << (local ? rows[i].local_dof : rows[i].global_dof);
    }
    return oss.str();
}

[[nodiscard]] std::vector<GlobalIndex> operatorMatrixSupportSampleDofs(const RowObservationOptions& options)
{
    std::vector<GlobalIndex> sample_dofs =
        options.residual_samples;

    const auto& matrix_sample_dofs = options.matrix_samples;
    sample_dofs.insert(sample_dofs.end(),
                       matrix_sample_dofs.begin(),
                       matrix_sample_dofs.end());

    std::sort(sample_dofs.begin(), sample_dofs.end());
    sample_dofs.erase(std::unique(sample_dofs.begin(), sample_dofs.end()),
                      sample_dofs.end());
    return sample_dofs;
}

void logRowOperatorMatrixSupportDiagnostic(
    const RowObservationOptions& options,
    const systems::FESystem& sys,
    const backends::GenericMatrix& matrix,
    std::string_view row_field,
    std::string_view coupling_field,
    std::string_view phase,
    std::string_view op,
    int iteration,
    double solve_time,
    double dt)
{
    if (!options.matrix_support) {
        return;
    }

    const auto observed_range =
        newtonMatrixSupportFieldRangeByName(sys, row_field);
    const auto coupling_range =
        newtonMatrixSupportFieldRangeByName(sys, coupling_field);
    if (!observed_range.has_value() || !coupling_range.has_value()) {
        if (mpiRank() == 0) {
            FE_LOG_INFO(
                "NewtonSolver: " + options.subject + " row operator matrix support diagnostic diagnostic=" +
                options.subject + "_row_operator_matrix_support status=field_missing " +
                options.subject + "_field='" + std::string(row_field) + "' coupling_field='" +
                std::string(coupling_field) + "' op='" +
                std::string(op) + "' phase='" + std::string(phase) + "'");
        }
        return;
    }

    const auto sample_dofs = operatorMatrixSupportSampleDofs(options);
    if (sample_dofs.empty()) {
        return;
    }

    constexpr Real tiny = Real(1.0e-14);
    for (const auto dof : sample_dofs) {
        std::ostringstream oss;
        oss << std::setprecision(17);
        oss << "NewtonSolver: " << options.subject << " row operator matrix support diagnostic"
            << " diagnostic=" << options.subject << "_row_operator_matrix_support"
            << " rank=" << mpiRank()
            << " iteration=" << iteration
            << " phase='" << phase << "'"
            << " op='" << op << "'"
            << " backend=" << backends::backendKindToString(matrix.backendKind())
            << " solve_time=" << solve_time
            << " dt=" << dt
            << " " << options.subject << "_field='" << row_field << "'"
            << " coupling_field='" << coupling_field << "'"
            << " " << options.subject << "_offset=" << observed_range->begin
            << " " << options.subject << "_dofs=" << (observed_range->end - observed_range->begin)
            << " coupling_offset=" << coupling_range->begin
            << " coupling_dofs=" << (coupling_range->end - coupling_range->begin)
            << " dof=" << dof;

        if (dof < 0 || dof >= matrix.numRows() || dof >= matrix.numCols()) {
            oss << " status=out_of_range"
                << " rows=" << matrix.numRows()
                << " cols=" << matrix.numCols();
            FE_LOG_INFO(oss.str());
            continue;
        }
        if (!dofInFieldRange(dof, *observed_range)) {
            oss << " status=non_" << options.subject << "_row";
            FE_LOG_INFO(oss.str());
            continue;
        }

        MatrixSupportRow row;
        row.global_dof = dof;
        row.local_dof = dof - observed_range->begin;
        int row_numeric_entries = 0;
        int row_self_numeric_entries = 0;
        int row_coupling_numeric_entries = 0;
        int col_numeric_entries = 0;
        int col_self_numeric_entries = 0;
        int col_coupling_numeric_entries = 0;

        for (GlobalIndex col = 0; col < matrix.numCols(); ++col) {
            const Real value = matrix.getEntry(dof, col);
            if (!std::isfinite(value)) {
                continue;
            }
            const double abs_value = std::abs(static_cast<double>(value));
            row.row_abs_sum += abs_value;
            if (abs_value > static_cast<double>(tiny)) {
                ++row_numeric_entries;
            }
            if (dofInFieldRange(col, *coupling_range)) {
                row.row_coupling_abs_sum += abs_value;
                if (abs_value > static_cast<double>(tiny)) {
                    ++row_coupling_numeric_entries;
                }
            }
            if (dofInFieldRange(col, *observed_range)) {
                row.row_self_abs_sum += abs_value;
                row.row_self_sum += static_cast<double>(value);
                if (abs_value > static_cast<double>(tiny)) {
                    ++row_self_numeric_entries;
                }
            }
        }
        for (GlobalIndex matrix_row = 0; matrix_row < matrix.numRows(); ++matrix_row) {
            const Real value = matrix.getEntry(matrix_row, dof);
            if (!std::isfinite(value)) {
                continue;
            }
            const double abs_value = std::abs(static_cast<double>(value));
            row.col_abs_sum += abs_value;
            if (abs_value > static_cast<double>(tiny)) {
                ++col_numeric_entries;
            }
            if (dofInFieldRange(matrix_row, *coupling_range)) {
                row.col_coupling_abs_sum += abs_value;
                if (abs_value > static_cast<double>(tiny)) {
                    ++col_coupling_numeric_entries;
                }
            }
            if (dofInFieldRange(matrix_row, *observed_range)) {
                row.col_self_abs_sum += abs_value;
                if (abs_value > static_cast<double>(tiny)) {
                    ++col_self_numeric_entries;
                }
            }
        }

        row.diag = static_cast<double>(matrix.getEntry(dof, dof));
        row.row_self_offdiag_abs_sum =
            std::max(0.0, row.row_self_abs_sum - std::abs(row.diag));
        if (row.row_self_abs_sum > 0.0) {
            row.row_self_signed_abs_ratio =
                std::abs(row.row_self_sum) / row.row_self_abs_sum;
            row.row_self_diag_abs_ratio =
                std::abs(row.diag) / row.row_self_abs_sum;
        }

        oss << " status=ok"
            << " field='" << row_field << "'"
            << " field_local_dof=" << row.local_dof
            << " " << options.subject << "_local_dof=" << row.local_dof
            << " row_abs_sum=" << row.row_abs_sum
            << " row_numeric_entries=" << row_numeric_entries
            << " row_self_abs_sum=" << row.row_self_abs_sum
            << " row_self_numeric_entries=" << row_self_numeric_entries
            << " row_self_sum=" << row.row_self_sum
            << " row_self_offdiag_abs_sum=" << row.row_self_offdiag_abs_sum
            << " row_self_signed_abs_ratio="
            << row.row_self_signed_abs_ratio
            << " row_self_diag_abs_ratio=" << row.row_self_diag_abs_ratio
            << " row_coupling_abs_sum=" << row.row_coupling_abs_sum
            << " row_coupling_numeric_entries=" << row_coupling_numeric_entries
            << " col_abs_sum=" << row.col_abs_sum
            << " col_numeric_entries=" << col_numeric_entries
            << " col_self_abs_sum=" << row.col_self_abs_sum
            << " col_self_numeric_entries=" << col_self_numeric_entries
            << " col_coupling_abs_sum=" << row.col_coupling_abs_sum
            << " col_coupling_numeric_entries=" << col_coupling_numeric_entries
            << " diag=" << row.diag
            << " row_first_nonzero="
            << firstNonzeroMatrixIndices(matrix, dof, /*scan_row=*/true, tiny)
            << " col_first_nonzero="
            << firstNonzeroMatrixIndices(matrix, dof, /*scan_row=*/false, tiny);
        FE_LOG_INFO(oss.str());
    }
}

[[nodiscard]] MatrixSupportSummary scanMatrixSupport(
    const systems::FESystem& sys,
    const backends::GenericMatrix& matrix,
    std::span<const GlobalIndex> constrained_dofs,
    std::string_view observed_field_name,
    std::string_view coupling_field_name,
    double tolerance,
    int sample_limit)
{
    MatrixSupportSummary summary;
    summary.observed_field = std::string(observed_field_name);
    summary.coupling_field = std::string(coupling_field_name);

    const auto observed_range =
        newtonMatrixSupportFieldRangeByName(sys, observed_field_name);
    const auto coupling_range =
        newtonMatrixSupportFieldRangeByName(sys, coupling_field_name);
    if (!observed_range.has_value() || !coupling_range.has_value()) {
        return summary;
    }

    summary.observed_offset = observed_range->begin;
    summary.observed_dofs = observed_range->end - observed_range->begin;
    summary.coupling_offset = coupling_range->begin;
    summary.coupling_dofs = coupling_range->end - coupling_range->begin;

    const auto can_add_sample = [sample_limit](const auto& samples) {
        return sample_limit < 0 ||
               static_cast<int>(samples.size()) < sample_limit;
    };
    const auto add_weakest_coupling_sample =
        [sample_limit](std::vector<MatrixSupportRow>& samples,
                       const MatrixSupportRow& row) {
            if (sample_limit == 0) {
                return;
            }
            samples.push_back(row);
            std::sort(samples.begin(), samples.end(),
                      [](const MatrixSupportRow& a,
                         const MatrixSupportRow& b) {
                          if (a.row_coupling_abs_sum == b.row_coupling_abs_sum) {
                              return a.global_dof < b.global_dof;
                          }
                          return a.row_coupling_abs_sum < b.row_coupling_abs_sum;
                      });
            if (sample_limit > 0 &&
                static_cast<int>(samples.size()) > sample_limit) {
                samples.pop_back();
            }
        };
    const auto add_weakest_self_sample =
        [sample_limit](std::vector<MatrixSupportRow>& samples,
                       const MatrixSupportRow& row) {
            if (sample_limit == 0) {
                return;
            }
            samples.push_back(row);
            std::sort(samples.begin(), samples.end(),
                      [](const MatrixSupportRow& a,
                         const MatrixSupportRow& b) {
                          if (a.row_self_abs_sum == b.row_self_abs_sum) {
                              return a.global_dof < b.global_dof;
                          }
                          return a.row_self_abs_sum < b.row_self_abs_sum;
                      });
            if (sample_limit > 0 &&
                static_cast<int>(samples.size()) > sample_limit) {
                samples.pop_back();
            }
        };

    for (GlobalIndex local_dof = 0; local_dof < summary.observed_dofs; ++local_dof) {
        const auto global_dof = summary.observed_offset + local_dof;
        if (std::binary_search(
                constrained_dofs.begin(), constrained_dofs.end(), global_dof)) {
            ++summary.constrained_observed_rows;
            continue;
        }

        ++summary.unconstrained_observed_rows;
        MatrixSupportRow row;
        row.local_dof = local_dof;
        row.global_dof = global_dof;
        if (global_dof < 0 ||
            global_dof >= matrix.numRows() ||
            global_dof >= matrix.numCols()) {
            if (can_add_sample(summary.zero_row_samples)) {
                summary.zero_row_samples.push_back(row);
            }
            ++summary.zero_row_count;
            ++summary.zero_col_count;
            ++summary.zero_diag_count;
            ++summary.zero_coupling_row_block_count;
            ++summary.zero_coupling_col_block_count;
            ++summary.zero_self_row_block_count;
            ++summary.zero_self_col_block_count;
            continue;
        }

        for (GlobalIndex col = 0; col < matrix.numCols(); ++col) {
            const Real value = matrix.getEntry(global_dof, col);
            if (!std::isfinite(value)) {
                continue;
            }
            const double abs_value = std::abs(static_cast<double>(value));
            row.row_abs_sum += abs_value;
            if (dofInFieldRange(col, *coupling_range)) {
                row.row_coupling_abs_sum += abs_value;
            }
            if (dofInFieldRange(col, *observed_range)) {
                row.row_self_abs_sum += abs_value;
                row.row_self_sum += static_cast<double>(value);
            }
        }
        for (GlobalIndex matrix_row = 0; matrix_row < matrix.numRows(); ++matrix_row) {
            const Real value = matrix.getEntry(matrix_row, global_dof);
            if (!std::isfinite(value)) {
                continue;
            }
            const double abs_value = std::abs(static_cast<double>(value));
            row.col_abs_sum += abs_value;
            if (dofInFieldRange(matrix_row, *coupling_range)) {
                row.col_coupling_abs_sum += abs_value;
            }
            if (dofInFieldRange(matrix_row, *observed_range)) {
                row.col_self_abs_sum += abs_value;
            }
        }
        row.diag = static_cast<double>(matrix.getEntry(global_dof, global_dof));
        row.row_self_offdiag_abs_sum =
            std::max(0.0, row.row_self_abs_sum - std::abs(row.diag));
        if (row.row_self_abs_sum > 0.0) {
            row.row_self_signed_abs_ratio =
                std::abs(row.row_self_sum) / row.row_self_abs_sum;
            row.row_self_diag_abs_ratio =
                std::abs(row.diag) / row.row_self_abs_sum;
        }

        const bool zero_row = std::abs(row.row_abs_sum) <= tolerance;
        const bool zero_col = std::abs(row.col_abs_sum) <= tolerance;
        const bool zero_diag = std::abs(row.diag) <= tolerance;
        const bool zero_coupling_row =
            std::abs(row.row_coupling_abs_sum) <= tolerance;
        const bool zero_coupling_col =
            std::abs(row.col_coupling_abs_sum) <= tolerance;
        const bool zero_self_row = std::abs(row.row_self_abs_sum) <= tolerance;
        const bool zero_self_col = std::abs(row.col_self_abs_sum) <= tolerance;
        if (zero_row) {
            ++summary.zero_row_count;
            if (can_add_sample(summary.zero_row_samples)) {
                summary.zero_row_samples.push_back(row);
            }
        }
        if (zero_col) {
            ++summary.zero_col_count;
        }
        if (zero_diag) {
            ++summary.zero_diag_count;
        }
        if (zero_coupling_row) {
            ++summary.zero_coupling_row_block_count;
            summary.zero_coupling_row_global_dofs.push_back(global_dof);
            if (can_add_sample(summary.zero_coupling_row_samples)) {
                summary.zero_coupling_row_samples.push_back(row);
            }
        } else {
            ++summary.positive_coupling_row_block_count;
            summary.min_positive_coupling_row_abs_sum =
                std::min(summary.min_positive_coupling_row_abs_sum,
                         row.row_coupling_abs_sum);
            summary.max_coupling_row_abs_sum =
                std::max(summary.max_coupling_row_abs_sum,
                         row.row_coupling_abs_sum);
            add_weakest_coupling_sample(
                summary.weakest_coupling_row_samples, row);
        }
        if (zero_coupling_col) {
            ++summary.zero_coupling_col_block_count;
        }
        if (zero_self_row) {
            ++summary.zero_self_row_block_count;
        } else {
            ++summary.positive_self_row_block_count;
            summary.min_positive_self_row_abs_sum =
                std::min(summary.min_positive_self_row_abs_sum,
                         row.row_self_abs_sum);
            summary.max_self_row_abs_sum =
                std::max(summary.max_self_row_abs_sum,
                         row.row_self_abs_sum);
            add_weakest_self_sample(summary.weakest_self_row_samples, row);
        }
        if (zero_self_col) {
            ++summary.zero_self_col_block_count;
        }
        if (zero_coupling_row && !zero_self_row) {
            ++summary.observed_only_row_block_count;
        }
        if (zero_coupling_col && !zero_self_col) {
            ++summary.observed_only_col_block_count;
        }
    }

    return summary;
}

void logRowOperatorMatrixSummaryDiagnostic(
    const RowObservationOptions& options,
    const systems::FESystem& sys,
    const backends::GenericMatrix& matrix,
    std::span<const GlobalIndex> constrained_dofs,
    std::string_view row_field,
    std::string_view coupling_field_name,
    std::string_view phase,
    std::string_view op,
    int iteration,
    double solve_time,
    double dt)
{
    if (!options.matrix_summary || !rowMatrixSummaryOpSelected(options, op)) {
        return;
    }

    const std::string observed_field{row_field};
    const std::string coupling_field{coupling_field_name};
    constexpr double tolerance = 1.0e-14;
    constexpr int sample_limit = 16;
    const auto summary = scanMatrixSupport(
        sys,
        matrix,
        constrained_dofs,
        observed_field,
        coupling_field,
        tolerance,
        sample_limit);

    if (summary.observed_dofs <= 0 || summary.coupling_dofs <= 0) {
        FE_LOG_INFO(
            "NewtonSolver: " + options.subject + " row operator matrix summary" +
            " diagnostic=" + options.subject + "_row_operator_matrix_summary" +
            " status=field_missing"
            " op='" +
            std::string(op) +
            "' phase='" + std::string(phase) + "'"
            " " + options.subject + "_field='" + observed_field + "'"
            " coupling_field='" + coupling_field + "'");
        return;
    }

    std::ostringstream oss;
    oss << std::setprecision(17);
    oss << "NewtonSolver: " << options.subject << " row operator matrix summary"
        << " diagnostic=" << options.subject << "_row_operator_matrix_summary"
        << " status=ok"
        << " rank=" << mpiRank()
        << " iteration=" << iteration
        << " phase='" << phase << "'"
        << " op='" << op << "'"
        << " backend=" << backends::backendKindToString(matrix.backendKind())
        << " solve_time=" << solve_time
        << " dt=" << dt
        << " " << options.subject << "_field='" << summary.observed_field << "'"
        << " coupling_field='" << summary.coupling_field << "'"
        << " " << options.subject << "_offset=" << summary.observed_offset
        << " " << options.subject << "_dofs=" << summary.observed_dofs
        << " coupling_offset=" << summary.coupling_offset
        << " coupling_dofs=" << summary.coupling_dofs
        << " constrained_" << options.subject << "_rows=" << summary.constrained_observed_rows
        << " unconstrained_" << options.subject << "_rows=" << summary.unconstrained_observed_rows
        << " zero_row_count=" << summary.zero_row_count
        << " zero_col_count=" << summary.zero_col_count
        << " zero_diag_count=" << summary.zero_diag_count
        << " zero_coupling_row_block_count="
        << summary.zero_coupling_row_block_count
        << " zero_coupling_col_block_count="
        << summary.zero_coupling_col_block_count
        << " zero_self_row_block_count=" << summary.zero_self_row_block_count
        << " zero_self_col_block_count=" << summary.zero_self_col_block_count
        << " positive_coupling_row_block_count="
        << summary.positive_coupling_row_block_count
        << " positive_self_row_block_count="
        << summary.positive_self_row_block_count
        << " min_positive_coupling_row_abs_sum="
        << (std::isfinite(summary.min_positive_coupling_row_abs_sum)
                ? summary.min_positive_coupling_row_abs_sum
                : 0.0)
        << " max_coupling_row_abs_sum="
        << summary.max_coupling_row_abs_sum
        << " min_positive_self_row_abs_sum="
        << (std::isfinite(summary.min_positive_self_row_abs_sum)
                ? summary.min_positive_self_row_abs_sum
                : 0.0)
        << " max_self_row_abs_sum="
        << summary.max_self_row_abs_sum
        << " " << options.subject << "_only_row_block_count="
        << summary.observed_only_row_block_count
        << " " << options.subject << "_only_col_block_count="
        << summary.observed_only_col_block_count
        << " tolerance=" << tolerance
        << " sample_limit=" << sample_limit
        << " zero_coupling_row_local_dofs="
        << formatMatrixSupportDofSamples(
               summary.zero_coupling_row_samples, /*local=*/true)
        << " zero_coupling_row_global_dofs="
        << formatMatrixSupportDofSamples(
               summary.zero_coupling_row_samples, /*local=*/false)
        << " zero_row_local_dofs="
        << formatMatrixSupportDofSamples(
               summary.zero_row_samples, /*local=*/true)
        << " zero_row_global_dofs="
        << formatMatrixSupportDofSamples(
               summary.zero_row_samples, /*local=*/false)
        << " weakest_coupling_row_local_dofs="
        << formatMatrixSupportDofSamples(
               summary.weakest_coupling_row_samples, /*local=*/true)
        << " weakest_coupling_row_global_dofs="
        << formatMatrixSupportDofSamples(
               summary.weakest_coupling_row_samples, /*local=*/false)
        << " weakest_self_row_local_dofs="
        << formatMatrixSupportDofSamples(
               summary.weakest_self_row_samples, /*local=*/true)
        << " weakest_self_row_global_dofs="
        << formatMatrixSupportDofSamples(
               summary.weakest_self_row_samples, /*local=*/false);
    FE_LOG_INFO(oss.str());
}


void zeroVectorEntries(std::span<const GlobalIndex> dofs, backends::GenericVector& vec)
{
    auto view = vec.createAssemblyView();
    FE_CHECK_NOT_NULL(view.get(), "NewtonSolver: zeroVectorEntries view");
    view->beginAssemblyPhase();
    if (!dofs.empty()) {
        view->zeroVectorEntries(dofs);
    }
    // Distributed backends may require every communicator rank to finalize,
    // including ranks whose local constrained-row list is empty.
    view->finalizeAssembly();
}

/// The Jacobian-pattern scratch matrix of the row-contribution observer.
class RowContributionStorage final : public timestepping::NewtonExtensionStorage {
public:
    std::unique_ptr<backends::GenericMatrix> matrix{};
};

[[nodiscard]] std::unique_ptr<backends::GenericMatrix> createJacobianPatternMatrix(
    const timestepping::NewtonStorageContext& context)
{
    const auto* dist = context.system.distributedSparsityIfAvailable(context.jacobian_op);
    if (dist != nullptr && context.factory.backendKind() != backends::BackendKind::Eigen) {
        return context.factory.createMatrix(*dist);
    }
    return context.factory.createMatrix(context.system.sparsity(context.jacobian_op));
}

} // namespace

void FieldResidualObserver::observe(timestepping::NewtonObservation& observation)
{
    logNewtonFieldResidualDiagnostic(
        options_,
        observation.transient.system(),
        observation.residual,
        observation.phase != nullptr ? std::string_view(observation.phase)
                                     : std::string_view("residual"),
        observation.sync_point,
        observation.iteration,
        observation.solve_time,
        observation.dt);
}

std::unique_ptr<timestepping::NewtonExtensionStorage> RowContributionObserver::allocate(
    const timestepping::NewtonStorageContext& context) const
{
    if (!options_.matrix_support) {
        return nullptr;
    }
    auto storage = std::make_unique<RowContributionStorage>();
    storage->matrix = createJacobianPatternMatrix(context);
    FE_CHECK_NOT_NULL(storage->matrix.get(),
                      "NewtonSolver workspace.diagnostic_jacobian_scratch");
    return storage;
}

void RowContributionObserver::refreshStorage(const timestepping::NewtonStorageContext& context,
                                             timestepping::NewtonExtensionStorage* storage,
                                             bool after_production) const
{
    auto* row_storage = dynamic_cast<RowContributionStorage*>(storage);
    if (!after_production || row_storage == nullptr || !row_storage->matrix) {
        return;
    }
    const auto& system = context.system;
    const auto& factory = context.factory;
    const auto* dist = system.distributedSparsityIfAvailable(context.jacobian_op);
    const bool use_distributed =
        dist != nullptr && factory.backendKind() != backends::BackendKind::Eigen;
    bool scratch_in_place =
        use_distributed
            ? row_storage->matrix->reinitFromPattern(*dist)
            : row_storage->matrix->reinitFromPattern(
                  system.sparsity(context.jacobian_op));
#if FE_HAS_MPI
    if (use_distributed) {
        const int local_scratch_in_place = scratch_in_place ? 1 : 0;
        int global_scratch_in_place = local_scratch_in_place;
        MPI_Allreduce(&local_scratch_in_place,
                      &global_scratch_in_place,
                      1,
                      MPI_INT,
                      MPI_MIN,
                      system.activeMpiCommunicator());
        scratch_in_place = global_scratch_in_place != 0;
    }
#endif
    FE_THROW_IF(
        !scratch_in_place &&
            factory.backendKind() == backends::BackendKind::FSILS,
        systems::InvalidStateException,
        "NewtonSolver: FSILS diagnostic Jacobian sparsity refresh changed "
        "its vector layout; rebuild the nonlinear workspace at a "
        "time-step boundary");
    if (!scratch_in_place) {
        row_storage->matrix = createJacobianPatternMatrix(context);
    }
}

void RowContributionObserver::observe(timestepping::NewtonObservation& observation)
{
    auto& transient = observation.transient;
    if (!declaration_.has_value()) {
        return;
    }
    const auto* rows = &*declaration_;
    const char* phase = observation.phase;
    const auto sync_point = observation.sync_point;
    auto& residual_scratch = observation.scratch;
    for (const auto& handle : rows->operators) {
        const std::string& op = handle.tag;
        if (!transient.system().hasOperator(op)) {
            if (mpiRank() == 0) {
                std::ostringstream skipped;
                skipped
                    << "NewtonSolver: " << options_.subject << " row contribution diagnostic skipped"
                    << " diagnostic=" << options_.subject << "_row_contribution_skipped"
                    << " reason=operator_not_installed"
                    << " op='" << op << "'"
                    << " phase='" << (phase != nullptr ? phase : "unknown") << "'"
                    << " sync_point=" << stateSyncPointName(sync_point);
                FE_LOG_INFO(skipped.str());
            }
            continue;
        }

        residual_scratch.zero();
        auto scratch_view = residual_scratch.createAssemblyView();
        FE_CHECK_NOT_NULL(
            scratch_view.get(),
            "NewtonSolver: row contribution diagnostic view");
        std::unique_ptr<assembly::GlobalSystemView> matrix_scratch_view;
        // A state-sync callback can re-augment constraint sparsity and
        // replace the distributed diagnostic matrix.  Resolve the pointer
        // after that synchronization/reallocation rather than retaining a
        // solve-lifetime raw pointer.
        auto* row_storage = dynamic_cast<RowContributionStorage*>(observation.storage);
        auto* diagnostic_jacobian_scratch =
            row_storage != nullptr ? row_storage->matrix.get() : nullptr;
        const bool want_matrix_diagnostic =
            options_.matrix_support &&
            diagnostic_jacobian_scratch != nullptr;
        if (want_matrix_diagnostic) {
            matrix_scratch_view =
                diagnostic_jacobian_scratch->createAssemblyView();
            FE_CHECK_NOT_NULL(
                matrix_scratch_view.get(),
                "NewtonSolver: row contribution matrix diagnostic view");
        }
        systems::AssemblyRequest req;
        req.op = op;
        req.want_matrix = want_matrix_diagnostic;
        req.want_vector = true;
        req.suppress_constraint_inhomogeneity = true;
        req.suppress_auxiliary_coupling_assembly = true;
        req.is_nonlinear_iteration = true;
        const auto ar = observation.assemble(
            options_.phase_prefix,
            req,
            matrix_scratch_view.get(),
            scratch_view.get());
        FE_THROW_IF(
            !ar.success,
            FEException,
            "NewtonSolver: " + options_.subject + " row contribution diagnostic assembly failed for op '" +
                op + "': " + ar.error_message);

        if (want_matrix_diagnostic) {
            const std::string matrix_phase =
                options_.phase_prefix + "_matrix:" +
                (phase != nullptr ? phase : "unknown");
            logRowOperatorMatrixSupportDiagnostic(
                options_,
                transient.system(),
                *diagnostic_jacobian_scratch,
                rows->row_field_name,
                rows->coupling_field_name,
                matrix_phase,
                op,
                observation.iteration,
                observation.solve_time,
                observation.dt);
            logRowOperatorMatrixSummaryDiagnostic(
                options_,
                transient.system(),
                *diagnostic_jacobian_scratch,
                observation.constrained_dofs,
                rows->row_field_name,
                rows->coupling_field_name,
                matrix_phase,
                op,
                observation.iteration,
                observation.solve_time,
                observation.dt);
        }

        const std::string pre_phase =
            options_.phase_prefix + "_pre_constraints:" + op;
        logNewtonFieldResidualDiagnostic(
            options_,
            transient.system(),
            residual_scratch,
            pre_phase,
            sync_point,
            observation.iteration,
            observation.solve_time,
            observation.dt);

        zeroVectorEntries(observation.constrained_dofs, residual_scratch);

        const std::string post_phase =
            options_.phase_prefix + "_post_constraints:" + op;
        logNewtonFieldResidualDiagnostic(
            options_,
            transient.system(),
            residual_scratch,
            post_phase,
            sync_point,
            observation.iteration,
            observation.solve_time,
            observation.dt);
    }
}

bool rowContributionObserverSwitchedOn() noexcept
{
    return environment::enabled("SVMP_FE_ROW_CONTRIBUTION_DIAGNOSTIC");
}

bool fieldResidualObserverSwitchedOn() noexcept
{
    return environment::enabled("SVMP_NEWTON_FIELD_RESIDUAL_DIAGNOSTIC") || rowContributionObserverSwitchedOn();
}

RowObservationOptions rowObservationOptionsFromEnvironment()
{
    RowObservationOptions options;
    if (const auto* field = environment::value("SVMP_NEWTON_FIELD_RESIDUAL_FIELD")) {
        options.residual_field = field;
    }
    if (const auto* samples = environment::value("SVMP_NEWTON_FIELD_RESIDUAL_SAMPLE_DOFS")) {
        options.residual_samples = parseRowSampleDofs(samples);
    }
    if (const auto* samples = environment::value("SVMP_NEWTON_MATRIX_SUPPORT_SAMPLE_DOFS")) {
        options.matrix_samples = parseRowSampleDofs(samples);
    }
    options.matrix_summary = environment::enabled("SVMP_FE_ROW_CONTRIBUTION_MATRIX_SUMMARY_DIAGNOSTIC");
    options.matrix_support = options.matrix_summary ||
        environment::enabled("SVMP_FE_ROW_CONTRIBUTION_MATRIX_DIAGNOSTIC") ||
        environment::isSet("SVMP_NEWTON_MATRIX_SUPPORT_SAMPLE_DOFS");
    options.summary_operators = numericalSummaryOperators;
    return options;
}

} // namespace analysis
} // namespace FE
} // namespace svmp

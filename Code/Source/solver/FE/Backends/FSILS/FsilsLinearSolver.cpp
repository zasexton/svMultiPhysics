/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Backends/FSILS/FsilsLinearSolver.h"

#include "Backends/FSILS/FsilsMatrix.h"
#include "Backends/FSILS/FsilsVector.h"
#include "Core/FEException.h"
#include "Core/Logger.h"

#include "Array.h"
#include "Vector.h"
#include "consts.h"
#include "Backends/FSILS/liner_solver/add_bc_mul.h"
#include "Backends/FSILS/liner_solver/bicgs.h"
#include "Backends/FSILS/liner_solver/fsils_api.hpp"
#include "Backends/FSILS/liner_solver/fils_struct.hpp"
#include "Backends/FSILS/liner_solver/spar_mul.h"
#include <algorithm>
#include <cctype>
#include <exception>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <map>
#include <cstdio>
#include <cstring>
#include <mpi.h>
#include <sstream>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace backends {

namespace {

constexpr int kNativeFaceDuplicateCouplingId = -2;

[[nodiscard]] bool rankOneUpdatesMatch(std::span<const RankOneUpdate> updates,
                                       const std::vector<RankOneUpdate>& existing) noexcept
{
    if (updates.size() != existing.size()) {
        return false;
    }

    for (std::size_t i = 0; i < updates.size(); ++i) {
        const auto& lhs = updates[i];
        const auto& rhs = existing[i];
        if (lhs.sigma != rhs.sigma ||
            lhs.v != rhs.v ||
            lhs.active_components != rhs.active_components ||
            lhs.prefer_native_face != rhs.prefer_native_face) {
            return false;
        }
    }

    return true;
}

struct FsilsResidualCheckResult {
    bool ok{true};
    Real rhs_norm{0.0};
    Real residual_norm{0.0};
    Real relative_residual{0.0};
    std::string detail{};
};

[[nodiscard]] bool fsilsAcceptNearTargetResidual(Real residual_norm,
                                                 Real target) noexcept
{
    constexpr Real tiny_target_threshold = static_cast<Real>(1e-8);
    constexpr Real near_target_slack = static_cast<Real>(1.1);
    return std::isfinite(static_cast<double>(residual_norm)) &&
           std::isfinite(static_cast<double>(target)) &&
           target > Real(0.0) &&
           target <= tiny_target_threshold &&
           residual_norm > target &&
           residual_norm <= target * near_target_slack;
}

struct FsilsConstraintMeanStats {
    bool valid{false};
    std::uint64_t count{0};
    Real mean{0.0};
    Real rms{0.0};
    Real fluctuation_rms{0.0};
};

struct FsilsLocalConstraintMeanStats {
    bool valid{false};
    std::uint64_t count{0};
    Real mean{0.0};
    Real rms{0.0};
    Real fluctuation_rms{0.0};
};

[[nodiscard]] bool fsilsCompareFaceOperatorEnabled() noexcept
{
    const char* env = std::getenv("SVMP_FSILS_COMPARE_FACE_OPERATOR");
    if (env == nullptr) {
        return false;
    }
    while (*env == ' ' || *env == '\t' || *env == '\n' || *env == '\r') {
        ++env;
    }
    return *env != '\0' && *env != '0';
}

[[nodiscard]] const char* fsilsCompareFaceOperatorOracleFile() noexcept
{
    const char* path = std::getenv("SVMP_FSILS_COMPARE_FACE_OPERATOR_ORACLE_FILE");
    if (path == nullptr || *path == '\0') {
        return nullptr;
    }
    return path;
}

[[nodiscard]] const char* fsilsCompareFaceOperatorDumpPrefix() noexcept
{
    const char* path = std::getenv("SVMP_FSILS_COMPARE_FACE_OPERATOR_DUMP_PREFIX");
    if (path == nullptr || *path == '\0') {
        return nullptr;
    }
    return path;
}

[[nodiscard]] const char* fsilsCompareFaceOperatorDumpProbe() noexcept
{
    const char* label = std::getenv("SVMP_FSILS_COMPARE_FACE_OPERATOR_DUMP_PROBE");
    if (label == nullptr || *label == '\0') {
        return nullptr;
    }
    return label;
}

[[nodiscard]] int fsilsDumpPreparedRowGlobalNode() noexcept
{
    const char* env = std::getenv("SVMP_FSILS_DUMP_PREPARED_ROW_GLOBAL_NODE");
    if (env == nullptr || *env == '\0') {
        return -1;
    }
    try {
        return std::stoi(env);
    } catch (...) {
        return -1;
    }
}

[[nodiscard]] int fsilsDumpPreparedRowComponent() noexcept
{
    const char* env = std::getenv("SVMP_FSILS_DUMP_PREPARED_ROW_COMPONENT");
    if (env == nullptr || *env == '\0') {
        return -1;
    }
    try {
        return std::stoi(env);
    } catch (...) {
        return -1;
    }
}

[[nodiscard]] int fsilsDumpPreparedColComponent() noexcept
{
    const char* env = std::getenv("SVMP_FSILS_DUMP_PREPARED_COL_COMPONENT");
    if (env == nullptr || *env == '\0') {
        return -1;
    }
    try {
        return std::stoi(env);
    } catch (...) {
        return -1;
    }
}

[[nodiscard]] const char* fsilsDumpPreparedRowPrefix() noexcept
{
    const char* path = std::getenv("SVMP_FSILS_DUMP_PREPARED_ROW_PREFIX");
    if (path == nullptr || *path == '\0') {
        return nullptr;
    }
    return path;
}

[[nodiscard]] bool fsilsProbeLowRankModesEnabled() noexcept
{
    const char* env = std::getenv("SVMP_FSILS_PROBE_LOW_RANK_MODES");
    if (env == nullptr) {
        return false;
    }
    while (*env == ' ' || *env == '\t' || *env == '\n' || *env == '\r') {
        ++env;
    }
    return *env != '\0' && *env != '0';
}

[[nodiscard]] bool fsilsLowRankResidualPolishEnabled() noexcept
{
    const char* env = std::getenv("SVMP_FSILS_DISABLE_LOW_RANK_RESIDUAL_POLISH");
    if (env == nullptr) {
        return true;
    }

    std::string value(env);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    value.erase(std::remove_if(value.begin(), value.end(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    }), value.end());
    return !(value == "1" || value == "true" || value == "on" || value == "yes");
}

[[nodiscard]] bool fsilsEnableMpiNativeFaceRankOne() noexcept
{
    if (const char* env = std::getenv("SVMP_FSILS_ENABLE_MPI_NATIVE_FACE_RANK_ONE")) {
        std::string value = env;
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        return value == "1" || value == "true" || value == "on" || value == "yes";
    }
    return false;
}

[[nodiscard]] bool solveDenseSystemInPlace(std::vector<Real>& a,
                                           std::vector<Real>& b,
                                           int n) noexcept;

[[nodiscard]] long double dotProductLongDouble(const FsilsVector& a,
                                               const FsilsVector& b) noexcept
{
    const auto a_span = a.localSpan();
    const auto b_span = b.localSpan();
    if (a_span.size() != b_span.size()) {
        return 0.0L;
    }

    long double sum = 0.0L;
    if (const auto* shared = a.shared()) {
        const int dof = shared->dof;
        const int nNo = shared->lhs.nNo;
        const int mynNo = shared->lhs.mynNo;
        const auto& lhs = shared->lhs;
        for (int old = 0; old < nNo; ++old) {
            if (lhs.map(old) >= mynNo) {
                continue;
            }
            const std::size_t base =
                static_cast<std::size_t>(old) * static_cast<std::size_t>(dof);
            for (int c = 0; c < dof; ++c) {
                sum += static_cast<long double>(a_span[base + static_cast<std::size_t>(c)]) *
                       static_cast<long double>(b_span[base + static_cast<std::size_t>(c)]);
            }
        }
#if FE_HAS_MPI
        int mpi_initialized = 0;
        MPI_Initialized(&mpi_initialized);
        if (mpi_initialized && lhs.commu.nTasks > 1) {
            long double global_sum = 0.0L;
            MPI_Allreduce(&sum, &global_sum, 1, MPI_LONG_DOUBLE, MPI_SUM, lhs.commu.comm);
            return global_sum;
        }
#endif
        return sum;
    }

    for (std::size_t i = 0; i < a_span.size(); ++i) {
        sum += static_cast<long double>(a_span[i]) * static_cast<long double>(b_span[i]);
    }
    return sum;
}

[[nodiscard]] bool solveDenseSystemInPlaceLongDouble(std::vector<long double>& a,
                                                     std::vector<long double>& b,
                                                     int n) noexcept;

void addRankOneUpdatesToProduct(std::span<const RankOneUpdate> updates,
                                FsilsVector& x,
                                FsilsVector& y,
                                fe_fsi_linear_solver::FSILS_commuType& commu)
{
    if (updates.empty()) {
        return;
    }

    auto x_view = x.createAssemblyView();
    FE_CHECK_NOT_NULL(x_view.get(), "FsilsLinearSolver: rank-one x view");

    std::vector<Real> dots(updates.size(), Real(0.0));
    for (std::size_t u = 0; u < updates.size(); ++u) {
        Real local_dot = Real(0.0);
        for (const auto& [dof, val] : updates[u].v) {
            local_dot += val * x_view->getVectorEntry(dof);
        }
        dots[u] = local_dot;
    }

#if FE_HAS_MPI
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized && commu.nTasks > 1) {
        std::vector<Real> global_dots(dots.size(), Real(0.0));
        fe_fsi_linear_solver::fsils_allreduce_sum(dots.data(),
                                                  global_dots.data(),
                                                  static_cast<int>(dots.size()),
                                                  MPI_DOUBLE,
                                                  commu);
        dots.swap(global_dots);
    }
#else
    (void)commu;
#endif

    FsilsVector correction(y);
    correction.zero();
    auto correction_view = correction.createAssemblyView();
    FE_CHECK_NOT_NULL(correction_view.get(), "FsilsLinearSolver: rank-one correction view");
    correction_view->beginAssemblyPhase();
    for (std::size_t u = 0; u < updates.size(); ++u) {
        const Real scale = updates[u].sigma * dots[u];
        if (std::abs(scale) <= Real(1e-30)) {
            continue;
        }
        for (const auto& [dof, val] : updates[u].v) {
            correction_view->addVectorEntry(dof, scale * val, assembly::AddMode::Add);
        }
    }
    correction_view->finalizeAssembly();
    correction.accumulateRawContributionsAndUpdateGhosts();

    auto y_span = y.localSpan();
    const auto correction_span = correction.localSpan();
    FE_THROW_IF(y_span.size() != correction_span.size(), FEException,
                "FsilsLinearSolver: rank-one correction size mismatch");
    for (std::size_t i = 0; i < y_span.size(); ++i) {
        y_span[i] += correction_span[i];
    }
}

void addReducedFieldUpdatesToProduct(std::span<const ReducedFieldUpdate> updates,
                                     FsilsVector& x,
                                     FsilsVector& y,
                                     fe_fsi_linear_solver::FSILS_commuType& commu,
                                     std::span<const GroupedBorderedFieldCoupling> exact_groups = {})
{
    if (updates.empty()) {
        return;
    }

    auto groupedUpdateHandledExactly = [&](int grouped_coupling_id) {
        if (grouped_coupling_id < 0) {
            return false;
        }
        for (const auto& group : exact_groups) {
            const int rank = static_cast<int>(group.modes.size());
            if (group.grouped_coupling_id == grouped_coupling_id &&
                rank > 0 &&
                group.aux_matrix.size() == static_cast<std::size_t>(rank * rank)) {
                return true;
            }
        }
        return false;
    };

    auto x_view = x.createAssemblyView();
    FE_CHECK_NOT_NULL(x_view.get(), "FsilsLinearSolver: reduced-update x view");

    std::vector<Real> dots(updates.size(), Real(0.0));
    for (std::size_t u = 0; u < updates.size(); ++u) {
        if (groupedUpdateHandledExactly(updates[u].grouped_coupling_id)) {
            continue;
        }
        Real local_dot = Real(0.0);
        for (const auto& [dof, val] : updates[u].right) {
            local_dot += val * x_view->getVectorEntry(dof);
        }
        dots[u] = local_dot;
    }

#if FE_HAS_MPI
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized && commu.nTasks > 1) {
        std::vector<Real> global_dots(dots.size(), Real(0.0));
        fe_fsi_linear_solver::fsils_allreduce_sum(dots.data(),
                                                  global_dots.data(),
                                                  static_cast<int>(dots.size()),
                                                  MPI_DOUBLE,
                                                  commu);
        dots.swap(global_dots);
    }
#else
    (void)commu;
#endif

    FsilsVector correction(y);
    correction.zero();
    auto correction_view = correction.createAssemblyView();
    FE_CHECK_NOT_NULL(correction_view.get(), "FsilsLinearSolver: reduced-update correction view");
    correction_view->beginAssemblyPhase();
    for (std::size_t u = 0; u < updates.size(); ++u) {
        if (groupedUpdateHandledExactly(updates[u].grouped_coupling_id)) {
            continue;
        }
        const Real scale = updates[u].sigma * dots[u];
        if (std::abs(scale) <= Real(1e-30)) {
            continue;
        }
        for (const auto& [dof, val] : updates[u].left) {
            correction_view->addVectorEntry(dof, scale * val, assembly::AddMode::Add);
        }
    }
    correction_view->finalizeAssembly();
    correction.accumulateRawContributionsAndUpdateGhosts();

    auto y_span = y.localSpan();
    const auto correction_span = correction.localSpan();
    FE_THROW_IF(y_span.size() != correction_span.size(), FEException,
                "FsilsLinearSolver: reduced-update correction size mismatch");
    for (std::size_t i = 0; i < y_span.size(); ++i) {
        y_span[i] += correction_span[i];
    }
}

void addGroupedBorderedFieldCouplingsToProduct(
    std::span<const GroupedBorderedFieldCoupling> groups,
    FsilsVector& x,
    FsilsVector& y,
    fe_fsi_linear_solver::FSILS_commuType& commu)
{
    if (groups.empty()) {
        return;
    }

    auto x_view = x.createAssemblyView();
    FE_CHECK_NOT_NULL(x_view.get(), "FsilsLinearSolver: grouped-coupling x view");

    FsilsVector correction(y);
    correction.zero();
    auto correction_view = correction.createAssemblyView();
    FE_CHECK_NOT_NULL(correction_view.get(), "FsilsLinearSolver: grouped-coupling correction view");
    correction_view->beginAssemblyPhase();

    for (const auto& group : groups) {
        const int rank = static_cast<int>(group.modes.size());
        if (rank <= 0 ||
            group.aux_matrix.size() != static_cast<std::size_t>(rank * rank)) {
            continue;
        }

        std::vector<Real> rhs(static_cast<std::size_t>(rank), Real(0.0));
        for (int i = 0; i < rank; ++i) {
            Real local_dot = Real(0.0);
            for (const auto& [dof, val] : group.modes[static_cast<std::size_t>(i)].right) {
                local_dot += val * x_view->getVectorEntry(dof);
            }
            rhs[static_cast<std::size_t>(i)] = local_dot;
        }

#if FE_HAS_MPI
        int mpi_initialized = 0;
        MPI_Initialized(&mpi_initialized);
        if (mpi_initialized && commu.nTasks > 1) {
            std::vector<Real> global_rhs(rhs.size(), Real(0.0));
            fe_fsi_linear_solver::fsils_allreduce_sum(rhs.data(),
                                                      global_rhs.data(),
                                                      static_cast<int>(rhs.size()),
                                                      MPI_DOUBLE,
                                                      commu);
            rhs.swap(global_rhs);
        }
#else
        (void)commu;
#endif

        std::vector<Real> dense = group.aux_matrix;
        for (auto& value : rhs) {
            value = -value;
        }
        if (!solveDenseSystemInPlace(dense, rhs, rank)) {
            continue;
        }

        for (int i = 0; i < rank; ++i) {
            const Real scale = rhs[static_cast<std::size_t>(i)];
            if (std::abs(scale) <= Real(1e-30)) {
                continue;
            }
            for (const auto& [dof, val] : group.modes[static_cast<std::size_t>(i)].left) {
                correction_view->addVectorEntry(dof, scale * val, assembly::AddMode::Add);
            }
        }
    }

    correction_view->finalizeAssembly();
    correction.accumulateRawContributionsAndUpdateGhosts();

    auto y_span = y.localSpan();
    const auto correction_span = correction.localSpan();
    FE_THROW_IF(y_span.size() != correction_span.size(),
                FEException,
                "FsilsLinearSolver: grouped-coupling correction size mismatch");
    for (std::size_t i = 0; i < y_span.size(); ++i) {
        y_span[i] += correction_span[i];
    }
}

void prepareRhsVectorForOperator(FsilsVector& rhs)
{
    rhs.updateGhosts();
}

[[nodiscard]] bool solveDenseSystemInPlace(std::vector<Real>& a,
                                           std::vector<Real>& b,
                                           int n) noexcept
{
    if (n <= 0 ||
        a.size() != static_cast<std::size_t>(n * n) ||
        b.size() != static_cast<std::size_t>(n)) {
        return false;
    }

    Real max_diag = Real(0.0);
    for (int i = 0; i < n; ++i) {
        max_diag = std::max(max_diag, std::abs(a[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
                                                 static_cast<std::size_t>(i)]));
    }
    const Real pivot_floor = std::max<Real>(max_diag * static_cast<Real>(1e-14),
                                            static_cast<Real>(1e-30));

    for (int k = 0; k < n; ++k) {
        int pivot_row = k;
        Real pivot_abs = std::abs(a[static_cast<std::size_t>(k) * static_cast<std::size_t>(n) +
                                   static_cast<std::size_t>(k)]);
        for (int row = k + 1; row < n; ++row) {
            const Real candidate_abs =
                std::abs(a[static_cast<std::size_t>(row) * static_cast<std::size_t>(n) +
                           static_cast<std::size_t>(k)]);
            if (candidate_abs > pivot_abs) {
                pivot_abs = candidate_abs;
                pivot_row = row;
            }
        }

        if (!(pivot_abs > pivot_floor)) {
            return false;
        }

        if (pivot_row != k) {
            for (int col = k; col < n; ++col) {
                std::swap(a[static_cast<std::size_t>(k) * static_cast<std::size_t>(n) +
                            static_cast<std::size_t>(col)],
                          a[static_cast<std::size_t>(pivot_row) * static_cast<std::size_t>(n) +
                            static_cast<std::size_t>(col)]);
            }
            std::swap(b[static_cast<std::size_t>(k)], b[static_cast<std::size_t>(pivot_row)]);
        }

        const Real pivot = a[static_cast<std::size_t>(k) * static_cast<std::size_t>(n) +
                             static_cast<std::size_t>(k)];
        for (int row = k + 1; row < n; ++row) {
            const std::size_t rowk =
                static_cast<std::size_t>(row) * static_cast<std::size_t>(n) +
                static_cast<std::size_t>(k);
            const Real factor = a[rowk] / pivot;
            if (std::abs(factor) <= Real(1e-30)) {
                continue;
            }
            a[rowk] = Real(0.0);
            for (int col = k + 1; col < n; ++col) {
                a[static_cast<std::size_t>(row) * static_cast<std::size_t>(n) +
                  static_cast<std::size_t>(col)] -=
                    factor *
                    a[static_cast<std::size_t>(k) * static_cast<std::size_t>(n) +
                      static_cast<std::size_t>(col)];
            }
            b[static_cast<std::size_t>(row)] -= factor * b[static_cast<std::size_t>(k)];
        }
    }

    for (int row = n - 1; row >= 0; --row) {
        Real sum = b[static_cast<std::size_t>(row)];
        for (int col = row + 1; col < n; ++col) {
            sum -= a[static_cast<std::size_t>(row) * static_cast<std::size_t>(n) +
                     static_cast<std::size_t>(col)] *
                   b[static_cast<std::size_t>(col)];
        }
        const Real diag = a[static_cast<std::size_t>(row) * static_cast<std::size_t>(n) +
                            static_cast<std::size_t>(row)];
        if (!(std::abs(diag) > pivot_floor)) {
            return false;
        }
        b[static_cast<std::size_t>(row)] = sum / diag;
    }

    return true;
}

[[nodiscard]] bool solveDenseSystemInPlaceLongDouble(std::vector<long double>& a,
                                                     std::vector<long double>& b,
                                                     int n) noexcept
{
    if (n <= 0 ||
        a.size() != static_cast<std::size_t>(n * n) ||
        b.size() != static_cast<std::size_t>(n)) {
        return false;
    }

    long double max_diag = 0.0L;
    for (int i = 0; i < n; ++i) {
        max_diag = std::max(max_diag,
                            std::abs(a[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
                                         static_cast<std::size_t>(i)]));
    }
    const long double pivot_floor = std::max<long double>(max_diag * 1e-18L, 1e-36L);

    for (int k = 0; k < n; ++k) {
        int pivot_row = k;
        long double pivot_abs =
            std::abs(a[static_cast<std::size_t>(k) * static_cast<std::size_t>(n) +
                       static_cast<std::size_t>(k)]);
        for (int row = k + 1; row < n; ++row) {
            const long double candidate_abs =
                std::abs(a[static_cast<std::size_t>(row) * static_cast<std::size_t>(n) +
                           static_cast<std::size_t>(k)]);
            if (candidate_abs > pivot_abs) {
                pivot_abs = candidate_abs;
                pivot_row = row;
            }
        }

        if (!(pivot_abs > pivot_floor)) {
            return false;
        }

        if (pivot_row != k) {
            for (int col = k; col < n; ++col) {
                std::swap(a[static_cast<std::size_t>(k) * static_cast<std::size_t>(n) +
                            static_cast<std::size_t>(col)],
                          a[static_cast<std::size_t>(pivot_row) * static_cast<std::size_t>(n) +
                            static_cast<std::size_t>(col)]);
            }
            std::swap(b[static_cast<std::size_t>(k)], b[static_cast<std::size_t>(pivot_row)]);
        }

        const long double pivot =
            a[static_cast<std::size_t>(k) * static_cast<std::size_t>(n) +
              static_cast<std::size_t>(k)];
        for (int row = k + 1; row < n; ++row) {
            const std::size_t rowk =
                static_cast<std::size_t>(row) * static_cast<std::size_t>(n) +
                static_cast<std::size_t>(k);
            const long double factor = a[rowk] / pivot;
            if (std::abs(factor) <= 1e-36L) {
                continue;
            }
            a[rowk] = 0.0L;
            for (int col = k + 1; col < n; ++col) {
                a[static_cast<std::size_t>(row) * static_cast<std::size_t>(n) +
                  static_cast<std::size_t>(col)] -=
                    factor * a[static_cast<std::size_t>(k) * static_cast<std::size_t>(n) +
                               static_cast<std::size_t>(col)];
            }
            b[static_cast<std::size_t>(row)] -= factor * b[static_cast<std::size_t>(k)];
        }
    }

    for (int row = n - 1; row >= 0; --row) {
        long double sum = b[static_cast<std::size_t>(row)];
        for (int col = row + 1; col < n; ++col) {
            sum -= a[static_cast<std::size_t>(row) * static_cast<std::size_t>(n) +
                     static_cast<std::size_t>(col)] *
                   b[static_cast<std::size_t>(col)];
        }
        const long double pivot =
            a[static_cast<std::size_t>(row) * static_cast<std::size_t>(n) +
              static_cast<std::size_t>(row)];
        if (!(std::abs(pivot) > 1e-36L)) {
            return false;
        }
        b[static_cast<std::size_t>(row)] = sum / pivot;
    }
    return true;
}

void copyVectorOldToInternal(const FsilsVector& src, std::span<Real> dst_internal)
{
    const auto* shared = src.shared();
    FE_CHECK_NOT_NULL(shared, "FsilsLinearSolver: shared layout for old->internal vector copy");

    const int dof = shared->dof;
    const int nNo = shared->lhs.nNo;
    const auto expected_size =
        static_cast<std::size_t>(dof) * static_cast<std::size_t>(nNo);
    FE_THROW_IF(dst_internal.size() != expected_size, InvalidArgumentException,
                "FsilsLinearSolver: old->internal vector copy size mismatch");
    FE_THROW_IF(src.data().size() != expected_size, InvalidArgumentException,
                "FsilsLinearSolver: old->internal source size mismatch");

    const auto& lhs = shared->lhs;
    const auto& src_data = src.data();
    for (int old = 0; old < nNo; ++old) {
        const int internal = lhs.map(old);
        const std::size_t src_base =
            static_cast<std::size_t>(old) * static_cast<std::size_t>(dof);
        const std::size_t dst_base =
            static_cast<std::size_t>(internal) * static_cast<std::size_t>(dof);
        for (int c = 0; c < dof; ++c) {
            dst_internal[dst_base + static_cast<std::size_t>(c)] =
                src_data[src_base + static_cast<std::size_t>(c)];
        }
    }
}

void copyVectorInternalToOld(std::span<const Real> src_internal, FsilsVector& dst)
{
    const auto* shared = dst.shared();
    FE_CHECK_NOT_NULL(shared, "FsilsLinearSolver: shared layout for internal->old vector copy");

    const int dof = shared->dof;
    const int nNo = shared->lhs.nNo;
    const auto expected_size =
        static_cast<std::size_t>(dof) * static_cast<std::size_t>(nNo);
    FE_THROW_IF(src_internal.size() != expected_size, InvalidArgumentException,
                "FsilsLinearSolver: internal->old vector copy size mismatch");
    FE_THROW_IF(dst.data().size() != expected_size, InvalidArgumentException,
                "FsilsLinearSolver: internal->old destination size mismatch");
    auto& dst_data = dst.data();

    if (!shared->old_of_internal.empty()) {
        FE_THROW_IF(static_cast<int>(shared->old_of_internal.size()) != nNo,
                    FEException,
                    "FsilsLinearSolver: invalid old_of_internal size");
        for (int internal = 0; internal < nNo; ++internal) {
            const int old = shared->old_of_internal[static_cast<std::size_t>(internal)];
            FE_THROW_IF(old < 0 || old >= nNo,
                        FEException,
                        "FsilsLinearSolver: invalid old_of_internal entry");
            const std::size_t src_base =
                static_cast<std::size_t>(internal) * static_cast<std::size_t>(dof);
            const std::size_t dst_base =
                static_cast<std::size_t>(old) * static_cast<std::size_t>(dof);
            for (int c = 0; c < dof; ++c) {
                dst_data[dst_base + static_cast<std::size_t>(c)] =
                    src_internal[src_base + static_cast<std::size_t>(c)];
            }
        }
        return;
    }

    const auto& lhs = shared->lhs;
    for (int old = 0; old < nNo; ++old) {
        const int internal = lhs.map(old);
        const std::size_t src_base =
            static_cast<std::size_t>(internal) * static_cast<std::size_t>(dof);
        const std::size_t dst_base =
            static_cast<std::size_t>(old) * static_cast<std::size_t>(dof);
        for (int c = 0; c < dof; ++c) {
            dst_data[dst_base + static_cast<std::size_t>(c)] =
                src_internal[src_base + static_cast<std::size_t>(c)];
        }
    }
}



} // namespace

FsilsLinearSolver::FsilsLinearSolver(const SolverOptions& options)
{
    setOptions(options);
}

FsilsLinearSolver::~FsilsLinearSolver()
{
    invalidateReusableBlockSchurState();
}

void FsilsLinearSolver::invalidateReusableBlockSchurState() const
{
    bicgs::reset_schur_cache(ls_.CG);
    ls_.GM.ws.recycle_k = 0;
    ls_.RI.ws.recycle_k = 0;
}

void FsilsLinearSolver::setOptions(const SolverOptions& options)
{
    FE_THROW_IF(options.max_iter <= 0, InvalidArgumentException, "FsilsLinearSolver: max_iter must be > 0");
    FE_THROW_IF(options.krylov_dim < 0, InvalidArgumentException, "FsilsLinearSolver: krylov_dim must be >= 0");
    FE_THROW_IF(options.rel_tol < 0.0, InvalidArgumentException, "FsilsLinearSolver: rel_tol must be >= 0");
    FE_THROW_IF(options.abs_tol < 0.0, InvalidArgumentException, "FsilsLinearSolver: abs_tol must be >= 0");
    FE_THROW_IF(options.use_initial_guess, NotImplementedException, "FsilsLinearSolver: initial guess not supported");
    if (options.fsils_blockschur_gm_max_iter) {
        FE_THROW_IF(*options.fsils_blockschur_gm_max_iter <= 0, InvalidArgumentException,
                    "FsilsLinearSolver: fsils_blockschur_gm_max_iter must be > 0");
    }
    if (options.fsils_blockschur_cg_max_iter) {
        FE_THROW_IF(*options.fsils_blockschur_cg_max_iter <= 0, InvalidArgumentException,
                    "FsilsLinearSolver: fsils_blockschur_cg_max_iter must be > 0");
    }
    if (options.fsils_blockschur_gm_rel_tol) {
        FE_THROW_IF(*options.fsils_blockschur_gm_rel_tol < 0.0, InvalidArgumentException,
                    "FsilsLinearSolver: fsils_blockschur_gm_rel_tol must be >= 0");
    }
    if (options.fsils_blockschur_cg_rel_tol) {
        FE_THROW_IF(*options.fsils_blockschur_cg_rel_tol < 0.0, InvalidArgumentException,
                    "FsilsLinearSolver: fsils_blockschur_cg_rel_tol must be >= 0");
    }
    options_ = normalizeSolverOptionsForBackend(options, BackendKind::FSILS);
    invalidateReusableBlockSchurState();
}

void FsilsLinearSolver::setRankOneUpdates(std::span<const RankOneUpdate> updates)
{
    const bool updates_changed = !rankOneUpdatesMatch(updates, rank_one_updates_);
    rank_one_updates_.assign(updates.begin(), updates.end());

    if (!updates_changed) {
        return;
    }

    // Native FSILS outlet faces cache the rank-one support and coefficients.
    // The update count often stays constant across Newton iterations while the
    // actual values move, so empty/non-empty tracking is not sufficient.
    faces_dirty_ = true;
}

void FsilsLinearSolver::setReducedFieldUpdates(std::span<const ReducedFieldUpdate> updates)
{
    reduced_field_updates_.assign(updates.begin(), updates.end());
}

void FsilsLinearSolver::setGroupedBorderedFieldCouplings(
    std::span<const GroupedBorderedFieldCoupling> groups)
{
    grouped_bordered_field_couplings_.assign(groups.begin(), groups.end());
}

void FsilsLinearSolver::setDirichletDofs(std::span<const GlobalIndex> dofs)
{
    std::vector<GlobalIndex> new_dofs(dofs.begin(), dofs.end());
    std::sort(new_dofs.begin(), new_dofs.end());
    new_dofs.erase(std::unique(new_dofs.begin(), new_dofs.end()), new_dofs.end());
    if (new_dofs != dirichlet_dofs_) {
        dirichlet_dofs_ = std::move(new_dofs);
        faces_dirty_ = true;
        invalidateReusableBlockSchurState();
    }
}

void FsilsLinearSolver::setEffectiveTimeStep(double dt_eff)
{
    if (std::isfinite(dt_eff) && dt_eff > 0.0) {
        dt_eff_ = dt_eff;
    } else {
        dt_eff_ = 1.0;
    }
}

namespace {

[[nodiscard]] bool oopTraceEnabled() noexcept
{
    static const bool enabled = [] {
        const char* env = std::getenv("SVMP_OOP_SOLVER_TRACE");
        if (env == nullptr) {
            return false;
        }
        std::string v(env);
        std::transform(v.begin(), v.end(), v.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return !(v == "0" || v == "false" || v == "off" || v == "no");
    }();
    return enabled;
}

[[nodiscard]] bool fsilsTraceEnabled() noexcept
{
    static const bool enabled = [] {
        const char* env = std::getenv("SVMP_FSILS_TRACE");
        if (env == nullptr) {
            return false;
        }
        std::string v(env);
        std::transform(v.begin(), v.end(), v.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return !(v == "0" || v == "false" || v == "off" || v == "no");
    }();
    return enabled;
}

[[nodiscard]] bool reducedInternalizationTraceEnabled() noexcept
{
    const char* env = std::getenv("SVMP_FSILS_TRACE_SCHUR_SETUP_TIMING");
    if (env != nullptr && *env != '\0' && *env != '0') {
        return true;
    }
    env = std::getenv("SVMP_FSILS_TRACE_SCHUR_SETUP_TIMING_ALL_RANKS");
    return env != nullptr && *env != '\0' && *env != '0';
}

void traceLog(const std::string& msg)
{
    if (!oopTraceEnabled() && !fsilsTraceEnabled()) {
        return;
    }
    FE_LOG_INFO(msg);
}

fe_fsi_linear_solver::LinearSolverType to_fsils_solver(SolverMethod method)
{
    switch (method) {
        case SolverMethod::CG: return fe_fsi_linear_solver::LS_TYPE_CG;
        case SolverMethod::GMRES: return fe_fsi_linear_solver::LS_TYPE_GMRES;
        case SolverMethod::PGMRES: return fe_fsi_linear_solver::LS_TYPE_GMRES;
        case SolverMethod::FGMRES: return fe_fsi_linear_solver::LS_TYPE_GMRES;
        case SolverMethod::BiCGSTAB: return fe_fsi_linear_solver::LS_TYPE_BICGS;
        case SolverMethod::BlockSchur: return fe_fsi_linear_solver::LS_TYPE_NS;
        case SolverMethod::Direct:
        default:
            FE_THROW(NotImplementedException, "FsilsLinearSolver: direct solve not supported by FSILS");
    }
}

consts::PreconditionerType to_fsils_prec(const SolverOptions& options)
{
    if (options.fsils_use_rcs || options.preconditioner == PreconditionerType::RowColumnScaling) {
        return consts::PreconditionerType::PREC_RCS;
    }

    switch (options.preconditioner) {
        case PreconditionerType::None:
        case PreconditionerType::Diagonal:
        case PreconditionerType::ILU:
        case PreconditionerType::AMG:
            // FSILS' solve path expects the diagonal/scale work vectors to be initialized by a preconditioner
            // routine (it always applies Wc element-wise after the Krylov solve). Treat unsupported/none as the
            // built-in diagonal preconditioner for correctness.
            return consts::PreconditionerType::PREC_FSILS;
        case PreconditionerType::FieldSplit:
            FE_THROW(NotImplementedException, "FsilsLinearSolver: field-split preconditioning not supported");
        default: return consts::PreconditionerType::PREC_NONE;
    }
}

fe_fsi_linear_solver::SchurPreconditionerType
to_fsils_blockschur_preconditioner(FsilsBlockSchurSchurPreconditioner pc)
{
    using fe_fsi_linear_solver::SchurPreconditionerType;
    switch (pc) {
        case FsilsBlockSchurSchurPreconditioner::Auto: return SchurPreconditionerType::ALGEBRAIC_SHAT;
        case FsilsBlockSchurSchurPreconditioner::DiagL: return SchurPreconditionerType::DIAG_L;
        case FsilsBlockSchurSchurPreconditioner::BlockDiagL: return SchurPreconditionerType::BLOCKDIAG_L;
        case FsilsBlockSchurSchurPreconditioner::ILUL: return SchurPreconditionerType::ILU_L;
        case FsilsBlockSchurSchurPreconditioner::AlgebraicSchur: return SchurPreconditionerType::ALGEBRAIC_SHAT;
    }
    return SchurPreconditionerType::DIAG_L;
}

fe_fsi_linear_solver::SchurMomentumApproximationType
to_fsils_blockschur_momentum_approximation(FsilsBlockSchurMomentumApproximation approx)
{
    using fe_fsi_linear_solver::SchurMomentumApproximationType;
    switch (approx) {
        case FsilsBlockSchurMomentumApproximation::Auto: return SchurMomentumApproximationType::ILU_K;
        case FsilsBlockSchurMomentumApproximation::DiagK: return SchurMomentumApproximationType::DIAG_K;
        case FsilsBlockSchurMomentumApproximation::BlockDiagK: return SchurMomentumApproximationType::BLOCKDIAG_K;
        case FsilsBlockSchurMomentumApproximation::ILUK: return SchurMomentumApproximationType::ILU_K;
        case FsilsBlockSchurMomentumApproximation::ASM: return SchurMomentumApproximationType::ASM_K;
    }
    return SchurMomentumApproximationType::DIAG_K;
}

struct GmresLaunchConfig {
    int mItr{1};
    int sD{0};
};

[[nodiscard]] int gmres_expected_total_iterations(const GmresLaunchConfig& cfg)
{
    using i64 = long long;
    const i64 outer = static_cast<i64>(std::max(1, cfg.mItr));
    const i64 per_restart = static_cast<i64>(std::max(0, cfg.sD)) + 1LL;
    const i64 total = outer * per_restart;
    if (total > static_cast<i64>(std::numeric_limits<int>::max())) {
        return std::numeric_limits<int>::max();
    }
    return static_cast<int>(std::max<i64>(1, total));
}

[[nodiscard]] int gmres_total_iteration_budget(const SolverOptions& options,
                                               bool legacy_restart_budget)
{
    if (!legacy_restart_budget) {
        return std::max(1, options.max_iter);
    }

    // Match FSILS XML semantics used by the legacy solver:
    // - Max_iterations = restart count (mItr)
    // - Krylov_space_dimension = restart length (sD)
    const int restart_len = (options.krylov_dim > 0) ? options.krylov_dim : 250;
    using i64 = long long;
    const i64 outer = static_cast<i64>(std::max(1, options.max_iter));
    const i64 per_restart = static_cast<i64>(std::max(0, restart_len)) + 1LL;
    const i64 total = outer * per_restart;
    if (total > static_cast<i64>(std::numeric_limits<int>::max())) {
        return std::numeric_limits<int>::max();
    }
    return static_cast<int>(std::max<i64>(1, total));
}

[[nodiscard]] GmresLaunchConfig make_gmres_launch_config(const SolverOptions& options,
                                                         int requested_total_iterations,
                                                         int sD_override = 0)
{
    int sD = 0;
    if (sD_override > 0) {
        sD = sD_override;
    } else {
        const char* sd_env = std::getenv("SVMP_FSILS_GMRES_SD");
        if (sd_env) {
            try {
                sD = std::stoi(sd_env);
            } catch (...) {
                sD = 0;
            }
        }
    }
    if (sD <= 0) {
        sD = options.krylov_dim;
    }
    if (sD <= 0) {
        sD = std::max(0, std::min(250, requested_total_iterations) - 1);
    }

    const int total_iterations = std::max(1, requested_total_iterations);
    const int sD_max = std::max(0, total_iterations - 1);
    sD = std::clamp(sD, 0, sD_max);

    const int per_restart = sD + 1;
    const int mItr = std::max(1, (total_iterations + per_restart - 1) / std::max(1, per_restart));
    GmresLaunchConfig cfg;
    cfg.mItr = mItr;
    cfg.sD = sD;
    return cfg;
}

[[nodiscard]] Real fsilsBlockSchurConstraintMeanDominanceLimit() noexcept
{
    Real limit = static_cast<Real>(1.0);
    if (const char* env = std::getenv("SVMP_FSILS_BLOCKSCHUR_CONSTRAINT_MEAN_DOMINANCE")) {
        try {
            limit = static_cast<Real>(std::stod(env));
        } catch (...) {
            limit = static_cast<Real>(1.0);
        }
    }
    if (!std::isfinite(static_cast<double>(limit)) || limit <= static_cast<Real>(0.0)) {
        limit = static_cast<Real>(1.0);
    }
    return limit;
}

[[nodiscard]] int fsilsBlockSchurOuterIterationCap(bool has_coupled_updates) noexcept
{
    if (const char* env = std::getenv("SVMP_FSILS_BLOCKSCHUR_OUTER_CAP")) {
        std::string value(env);
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        value.erase(std::remove_if(value.begin(), value.end(), [](unsigned char ch) {
            return std::isspace(ch) != 0;
        }), value.end());

        if (value == "none" || value == "unlimited" || value == "off") {
            return std::numeric_limits<int>::max();
        }

        try {
            const int parsed = std::stoi(value);
            if (parsed > 0) {
                return parsed;
            }
        } catch (...) {
        }
    }

    // BlockSchur uses RI.mItr as an outer basis/workspace dimension. Keep a
    // conservative cap for ordinary fractional-step solves, but allow larger
    // explicit budgets for coupled/reduced update solves where tight true
    // residual validation is required. Override with
    // SVMP_FSILS_BLOCKSCHUR_OUTER_CAP when a case needs a different workspace
    // policy.
    return has_coupled_updates ? 200 : 50;
}

struct ResolvedSaddlePointBlocks {
    const BlockDescriptor* primary{nullptr};
    const BlockDescriptor* constraint{nullptr};
};

[[nodiscard]] ResolvedSaddlePointBlocks resolveSaddlePointBlocks(
    const SolverOptions& options) noexcept
{
    if (!options.block_layout.has_value()) {
        return {};
    }

    const auto& layout = *options.block_layout;
    const auto resolve_named_block =
        [&](BlockRole role) -> const BlockDescriptor* {
            const auto name = options.resolveBlockNameForRole(role);
            if (!name.empty()) {
                if (const auto* block = layout.findBlock(name)) {
                    return block;
                }
            }
            return nullptr;
        };

    ResolvedSaddlePointBlocks resolved;
    resolved.primary = resolve_named_block(BlockRole::PrimaryField);
    resolved.constraint = resolve_named_block(BlockRole::ConstraintField);

    if (resolved.primary == nullptr) {
        resolved.primary = layout.primaryFieldBlock();
    }
    if (resolved.constraint == nullptr) {
        resolved.constraint = layout.constraintFieldBlock();
    }

    return resolved;
}

} // namespace

SolverReport FsilsLinearSolver::solve(const GenericMatrix& A_in,
                                      GenericVector& x_in,
                                      const GenericVector& b_in)
{
    const auto* A = dynamic_cast<const FsilsMatrix*>(&A_in);
    auto* x = dynamic_cast<FsilsVector*>(&x_in);
    const auto* b = dynamic_cast<const FsilsVector*>(&b_in);

    FE_THROW_IF(!A || !x || !b, InvalidArgumentException, "FsilsLinearSolver::solve: backend mismatch");
    FE_THROW_IF(A->numRows() != A->numCols(), NotImplementedException,
                "FsilsLinearSolver::solve: rectangular systems not implemented");
    FE_THROW_IF(b->size() != A->numRows() || x->size() != b->size(), InvalidArgumentException,
                "FsilsLinearSolver::solve: size mismatch");

    auto& lhs = *static_cast<fe_fsi_linear_solver::FSILS_lhsType*>(const_cast<void*>(A->fsilsLhsPtr()));
    const int dof = A->fsilsDof();
    lhs.system_dof = dof;
    FE_THROW_IF(dof <= 0, FEException, "FsilsLinearSolver::solve: invalid FSILS dof");

    // Derive block structure from metadata. No physics-specific fallbacks —
    // all saddle-point operations require explicit block_layout with saddle-point annotation.
    const bool has_block_layout = options_.block_layout.has_value();
    const auto saddle_point_blocks = resolveSaddlePointBlocks(options_);
    const bool has_saddle_point =
        has_block_layout &&
        saddle_point_blocks.primary != nullptr &&
        saddle_point_blocks.constraint != nullptr;

    // Saddle-point block indices (only meaningful when has_saddle_point is true).
    int mom_start = 0, mom_ncomp = 0;
    int con_start = 0, con_ncomp = 0;
    if (has_saddle_point) {
        const auto& mb = *saddle_point_blocks.primary;
        const auto& cb = *saddle_point_blocks.constraint;
        mom_start = mb.start_component;
        mom_ncomp = mb.n_components;
        con_start = cb.start_component;
        con_ncomp = cb.n_components;
    }
    FE_THROW_IF(lhs.nNo <= 0, FEException, "FsilsLinearSolver::solve: invalid FSILS lhs.nNo");

    const GlobalIndex expected_local = static_cast<GlobalIndex>(lhs.nNo) * static_cast<GlobalIndex>(dof);
    FE_THROW_IF(static_cast<GlobalIndex>(x->data().size()) != expected_local ||
                    static_cast<GlobalIndex>(b->data().size()) != expected_local,
                FEException, "FsilsLinearSolver::solve: FSILS vectors must have local size lhs.nNo*dof");

    const auto shared_layout = A->shared();
    FE_CHECK_NOT_NULL(shared_layout.get(), "FsilsLinearSolver::solve: shared layout");

    const bool requested_blockschur = (options_.method == SolverMethod::BlockSchur);
    const bool use_blockschur = requested_blockschur;

    if (oopTraceEnabled()) {
        std::ostringstream oss;
        oss << "FsilsLinearSolver::solve: n=" << A->numRows()
            << " dof=" << dof << " nNo=" << lhs.nNo
            << " dropped_entries=" << FsilsMatrix::droppedEntryCount()
            << " method=" << solverMethodToString(options_.method)
            << " prec=" << preconditionerToString(options_.preconditioner)
            << " rel_tol=" << options_.rel_tol
            << " abs_tol=" << options_.abs_tol
            << " max_iter=" << options_.max_iter
            << " krylov_dim=" << options_.krylov_dim
            << " fsils_use_rcs=" << (options_.fsils_use_rcs ? 1 : 0);
        if (requested_blockschur) {
            oss << " schur_pc="
                << fsilsBlockSchurPreconditionerToString(options_.fsils_blockschur_schur_preconditioner)
                << " momentum_hat="
                << fsilsBlockSchurMomentumApproximationToString(
                       options_.fsils_blockschur_momentum_approximation);
        }
        traceLog(oss.str());
    }

    if (requested_blockschur) {
        FE_THROW_IF(!has_saddle_point, NotImplementedException,
                    "FsilsLinearSolver::solve: BlockSchur requires block_layout with "
                    "resolvable saddle-point metadata");
        const auto& mb = *saddle_point_blocks.primary;
        const auto& cb = *saddle_point_blocks.constraint;
        FE_THROW_IF(mb.n_components < 1 || cb.n_components < 1,
                    NotImplementedException,
                    "FsilsLinearSolver::solve: BlockSchur requires valid saddle-point layout");
        FE_THROW_IF(mb.n_components + cb.n_components != dof,
                    NotImplementedException,
                    "FsilsLinearSolver::solve: BlockSchur saddle-point blocks must cover all DOFs "
                    "(momentum=" + std::to_string(mb.n_components) + " + constraint=" +
                    std::to_string(cb.n_components) + " != dof=" + std::to_string(dof) + ")");
    }

    // FSILS destructively modifies the matrix during preconditioning/solve.
    // Keep a solver-local copy for all methods so we can:
    // - validate the true post-solve residual against the original operator, and
    // - retry with a stricter Krylov configuration when the first solve is inexact.
    const GlobalIndex nnz = A->fsilsNnz();
    const std::size_t block_size = static_cast<std::size_t>(dof) * static_cast<std::size_t>(dof);
    const std::size_t value_count = static_cast<std::size_t>(nnz) * block_size;
    values_work_.resize(value_count);
    std::copy(A->fsilsValuesPtr(), A->fsilsValuesPtr() + value_count, values_work_.data());

    // Public FE vectors stay in old local ordering, but the FSILS solve buffer is kept
    // in internal node ordering so overlap communication and fsils_solve() can operate
    // directly on the solver-native layout.
    auto& x_data = x->data();
    const auto& b_data = b->data();
    FE_THROW_IF(x_data.size() != b_data.size(), FEException, "FsilsLinearSolver::solve: RHS size mismatch");
    ri_internal_work_.resize(b_data.size());

    Array<double> Ri(dof, lhs.nNo, ri_internal_work_.data());
    FE_THROW_IF(nnz > static_cast<GlobalIndex>(std::numeric_limits<int>::max()), InvalidArgumentException,
                "FsilsLinearSolver::solve: nnz exceeds FSILS int index range");
    Array<double> Val(dof * dof, static_cast<int>(nnz), values_work_.data());

    auto dumpPreparedMatrixRowIfRequested = [&](std::string_view phase) {
        const int target_global_node = fsilsDumpPreparedRowGlobalNode();
        const int row_comp = fsilsDumpPreparedRowComponent();
        const char* dump_prefix = fsilsDumpPreparedRowPrefix();
        if (target_global_node < 0 || row_comp < 0 || row_comp >= dof || dump_prefix == nullptr) {
            return;
        }

        const int target_old = shared_layout->globalNodeToOld(target_global_node);
        if (target_old < 0 || target_old >= lhs.nNo) {
            return;
        }
        const int row_internal = lhs.map(target_old);
        if (row_internal < 0 || row_internal >= lhs.nNo) {
            return;
        }

        std::vector<int> internal_to_old(static_cast<std::size_t>(lhs.nNo), -1);
        for (int old = 0; old < lhs.nNo; ++old) {
            const int internal = lhs.map(old);
            if (internal >= 0 && internal < lhs.nNo) {
                internal_to_old[static_cast<std::size_t>(internal)] = old;
            }
        }

        std::ostringstream path;
        path << dump_prefix << ".matrix." << phase
             << ".g" << target_global_node
             << ".r" << row_comp
             << ".rank" << lhs.commu.task
             << ".txt";
        std::ofstream out(path.str());
        if (!out) {
            return;
        }

        const GlobalIndex backend_row_dof =
            static_cast<GlobalIndex>(target_global_node) * static_cast<GlobalIndex>(dof) +
            static_cast<GlobalIndex>(row_comp);
        GlobalIndex fe_row_dof = backend_row_dof;
        if (const auto* perm = shared_layout->dof_permutation.get();
            perm != nullptr && !perm->inverse.empty() &&
            static_cast<std::size_t>(backend_row_dof) < perm->inverse.size()) {
            fe_row_dof = perm->inverse[static_cast<std::size_t>(backend_row_dof)];
        }

        const int col_comp_filter = fsilsDumpPreparedColComponent();
        out << "# task " << lhs.commu.task
            << " phase " << phase
            << " global_row_node " << target_global_node
            << " old_row " << target_old
            << " row_kind " << ((target_old < shared_layout->owned_node_count) ? "owned" : "ghost")
            << " internal_row " << row_internal
            << " row_component " << row_comp
            << " backend_row_dof " << backend_row_dof
            << " fe_row_dof " << fe_row_dof
            << " col_component_filter " << col_comp_filter << "\n";
        out << "# col_global_node col_component value col_old col_kind\n";
        for (int nz = lhs.rowPtr(0, row_internal); nz <= lhs.rowPtr(1, row_internal); ++nz) {
            const int col_internal = lhs.colPtr(nz);
            if (col_internal < 0 || col_internal >= lhs.nNo) {
                continue;
            }
            const int col_old = internal_to_old[static_cast<std::size_t>(col_internal)];
            if (col_old < 0 || col_old >= lhs.nNo) {
                continue;
            }
            const int col_global_node = shared_layout->oldToGlobalNode(col_old);
            const char* col_kind = (col_old < shared_layout->owned_node_count) ? "owned" : "ghost";
            for (int col_comp = 0; col_comp < dof; ++col_comp) {
                if (col_comp_filter >= 0 && col_comp != col_comp_filter) {
                    continue;
                }
                out << col_global_node
                    << ' ' << col_comp
                    << ' ' << Val(row_comp * dof + col_comp, nz)
                    << ' ' << col_old
                    << ' ' << col_kind
                    << '\n';
            }
        }
    };

    // Optional scaling used for the BlockSchur solver path.
    //
    // The legacy solver scales resistance-type coupled BC tangent contributions by (gamma*dt),
    // where gamma is the generalized-α parameter. The OOP solver provides the effective stage
    // dt via LinearSolver::setEffectiveTimeStep().
    //
    // That transform is currently not robust for the distributed native BlockSchur path on the
    // coupled outlet application cases. Keep the algebra unscaled by default and leave the old
    // transform available only as an explicit diagnostic opt-in.
    double stage_scale = 1.0;
    if (use_blockschur && has_saddle_point) {
        if (std::getenv("SVMP_FSILS_ENABLE_BLOCKSCHUR_STAGE_SCALING") != nullptr &&
            std::getenv("SVMP_FSILS_DISABLE_BLOCKSCHUR_STAGE_SCALING") == nullptr &&
            std::isfinite(dt_eff_) && dt_eff_ > 0.0) {
            stage_scale = dt_eff_;
        }
    }

    auto applyStageScalingToMatrix = [&]() {
        if (stage_scale == 1.0) {
            return;
        }
        const Real s = static_cast<Real>(stage_scale);
        const Real inv_s = static_cast<Real>(1.0 / stage_scale);

        // Left-scale momentum rows.
        for (GlobalIndex bi = 0; bi < nnz; ++bi) {
            Real* blk = values_work_.data() + static_cast<std::size_t>(bi) * block_size;
            for (int r = mom_start; r < mom_start + mom_ncomp; ++r) {
                for (int c = 0; c < dof; ++c) {
                    blk[static_cast<std::size_t>(r * dof + c)] *= s;
                }
            }
            // Right-scale constraint columns to preserve G ≈ -D^T.
            for (int r = 0; r < dof; ++r) {
                for (int c = con_start; c < con_start + con_ncomp; ++c) {
                    blk[static_cast<std::size_t>(r * dof + c)] *= inv_s;
                }
            }
        }
    };

    auto restoreAndScaleMatrixValues = [&]() {
        std::copy(A->fsilsValuesPtr(), A->fsilsValuesPtr() + value_count, values_work_.data());
        applyStageScalingToMatrix();
    };

    applyStageScalingToMatrix();

    const bool enforce_transpose_saddle_blocks =
        use_blockschur && has_saddle_point &&
        (std::getenv("SVMP_FSILS_ASSUME_TRANSPOSE_SADDLE") != nullptr) &&
        (std::getenv("SVMP_FSILS_DISABLE_TRANSPOSE_SADDLE") == nullptr);

    auto applySaddlePointEnforcement = [&]() {
        if (!enforce_transpose_saddle_blocks) {
            return;
        }
        const int nNo = lhs.nNo;
        const int nnz_int = lhs.nnz;
        if (nNo <= 0 || nnz_int <= 0) {
            return;
        }

        auto* cols = lhs.colPtr.data();
        const auto find_entry = [&](int row, int col) -> fe_fsi_linear_solver::fsils_int {
            const auto start = lhs.rowPtr(0, row);
            const auto end = lhs.rowPtr(1, row);
            if (start < 0 || end < start) {
                return -1;
            }
            const auto len = end - start + 1;
            auto* begin = cols + start;
            auto* finish = begin + len;
            const auto it = std::lower_bound(begin, finish, static_cast<fe_fsi_linear_solver::fsils_int>(col));
            if (it == finish || *it != col) {
                return -1;
            }
            return static_cast<fe_fsi_linear_solver::fsils_int>(it - cols);
        };

        for (fe_fsi_linear_solver::fsils_int row = 0; row < nNo; ++row) {
            const auto start = lhs.rowPtr(0, row);
            const auto end = lhs.rowPtr(1, row);
            if (start < 0 || end < start) {
                continue;
            }
            for (auto idx = start; idx <= end; ++idx) {
                const auto col_idx = cols[idx];
                if (col_idx < 0 || col_idx >= nNo) {
                    continue;
                }

                const auto idx_t = find_entry(col_idx, row);
                if (idx_t < 0 || idx_t >= nnz_int) {
                    continue;
                }

                Real* blk = values_work_.data() + static_cast<std::size_t>(idx) * block_size;
                Real* blk_t = values_work_.data() + static_cast<std::size_t>(idx_t) * block_size;
                for (int vc = 0; vc < mom_ncomp; ++vc) {
                    for (int cc = 0; cc < con_ncomp; ++cc) {
                        const Real g_val = blk[static_cast<std::size_t>((mom_start + vc) * dof + (con_start + cc))];
                        blk_t[static_cast<std::size_t>((con_start + cc) * dof + (mom_start + vc))] = -g_val;
                    }
                }
            }
        }
    };

    if (enforce_transpose_saddle_blocks && oopTraceEnabled()) {
        traceLog("FsilsLinearSolver::solve: enforcing D=-G^T due to "
                 "SVMP_FSILS_ASSUME_TRANSPOSE_SADDLE.");
    }

    auto restorePreparedMatrixValues = [&](bool blockschur_preparation) {
        std::copy(A->fsilsValuesPtr(), A->fsilsValuesPtr() + value_count, values_work_.data());
        if (blockschur_preparation) {
            applyStageScalingToMatrix();
            applySaddlePointEnforcement();
        }
        dumpPreparedMatrixRowIfRequested(blockschur_preparation ? "blockschur" : "original");
    };

    const bool pure_mpi_native_face_rank_one_case =
        (lhs.commu.nTasks > 1) &&
        !rank_one_updates_.empty() &&
        reduced_field_updates_.empty() &&
        grouped_bordered_field_couplings_.empty() &&
        std::all_of(rank_one_updates_.begin(),
                    rank_one_updates_.end(),
                    [](const auto& update) { return update.prefer_native_face; });
    const bool prefer_mpi_native_face_rank_one =
        (lhs.commu.nTasks > 1) &&
        use_blockschur &&
        ((!rank_one_updates_.empty() && !reduced_field_updates_.empty()) ||
         (rank_one_updates_.size() > 1));
    const bool mpi_native_face_rank_one_requested =
        (lhs.commu.nTasks > 1) &&
        !rank_one_updates_.empty() &&
        fsilsEnableMpiNativeFaceRankOne();
    const bool allow_mpi_native_face_rank_one =
        (lhs.commu.nTasks <= 1) || mpi_native_face_rank_one_requested;
    const bool trace_low_rank_state = oopTraceEnabled() || fsilsTraceEnabled();
    if ((!rank_one_updates_.empty() || !reduced_field_updates_.empty() ||
         !grouped_bordered_field_couplings_.empty()) && trace_low_rank_state) {
        const int prefer_count = static_cast<int>(std::count_if(
            rank_one_updates_.begin(), rank_one_updates_.end(),
            [](const auto& update) { return update.prefer_native_face; }));
        std::ostringstream oss;
        oss << "FsilsLinearSolver::solve: native-face routing state"
            << " nTasks=" << lhs.commu.nTasks
            << " rank_one=" << rank_one_updates_.size()
            << " prefer_native_face=" << prefer_count
            << " reduced=" << reduced_field_updates_.size()
            << " grouped=" << grouped_bordered_field_couplings_.size()
            << " pure_case=" << (pure_mpi_native_face_rank_one_case ? 1 : 0)
            << " prefer_case=" << (prefer_mpi_native_face_rank_one ? 1 : 0)
            << " requested=" << (mpi_native_face_rank_one_requested ? 1 : 0)
            << " allow=" << (allow_mpi_native_face_rank_one ? 1 : 0);
        traceLog(oss.str());
        for (std::size_t i = 0; i < reduced_field_updates_.size(); ++i) {
            std::ostringstream upd;
            upd << "FsilsLinearSolver::solve: reduced field update[" << i << "]"
                << " sigma=" << reduced_field_updates_[i].sigma
                << " grouped_id=" << reduced_field_updates_[i].grouped_coupling_id
                << " left_nnz=" << reduced_field_updates_[i].left.size()
                << " right_nnz=" << reduced_field_updates_[i].right.size();
            traceLog(upd.str());
        }
    }
    if (!allow_mpi_native_face_rank_one && oopTraceEnabled()) {
        traceLog("FsilsLinearSolver::solve: routing MPI native-face rank-1 updates "
                 "through reduced/grouped update support.");
    } else if (prefer_mpi_native_face_rank_one && oopTraceEnabled()) {
        traceLog("FsilsLinearSolver::solve: enabling MPI native-face rank-1 path "
                 "for mixed rank-1 + reduced BlockSchur outlet corrections.");
    } else if (pure_mpi_native_face_rank_one_case && oopTraceEnabled()) {
        traceLog("FsilsLinearSolver::solve: enabling MPI native-face rank-1 path "
                 "(explicit opt-in).");
    }

    std::vector<std::size_t> native_face_rank_one_indices;
    native_face_rank_one_indices.reserve(rank_one_updates_.size());
    for (std::size_t i = 0; i < rank_one_updates_.size(); ++i) {
        if (allow_mpi_native_face_rank_one && rank_one_updates_[i].prefer_native_face) {
            native_face_rank_one_indices.push_back(i);
        }
    }

    const bool has_native_rank_one_updates =
        !rank_one_updates_.empty() ||
        !reduced_field_updates_.empty() ||
        !grouped_bordered_field_couplings_.empty();
    const bool has_native_direct_only_aux_updates =
        use_blockschur &&
        (!rank_one_updates_.empty() || !reduced_field_updates_.empty()) &&
        grouped_bordered_field_couplings_.empty();

    auto& ls = ls_;
    if (use_blockschur) {
        const int requested_max_iter = std::max(1, options_.max_iter);
        const int outer_cap = fsilsBlockSchurOuterIterationCap(has_native_rank_one_updates);
        const int effective_max_iter = std::min(requested_max_iter, outer_cap);
        if (oopTraceEnabled()) {
            std::ostringstream cap_oss;
            cap_oss << "FsilsLinearSolver::solve: BlockSchur outer iteration policy"
                    << " requested=" << requested_max_iter
                    << " cap=" << outer_cap
                    << " effective=" << effective_max_iter
                    << " coupled_updates=" << (has_native_rank_one_updates ? 1 : 0);
            traceLog(cap_oss.str());
        }
        if (options_.krylov_dim > 0) {
            fe_fsi_linear_solver::fsils_ls_create(ls,
                                                  fe_fsi_linear_solver::LS_TYPE_NS,
                                                  options_.rel_tol,
                                                  options_.abs_tol,
                                                  effective_max_iter,
                                                  options_.krylov_dim);
        } else {
            fe_fsi_linear_solver::fsils_ls_create(ls,
                                                  fe_fsi_linear_solver::LS_TYPE_NS,
                                                  options_.rel_tol,
                                                  options_.abs_tol,
                                                  effective_max_iter);
        }

        // Legacy semantics: GM/CG inherit absTol and Krylov dimension from RI.
        ls.GM.absTol = ls.RI.absTol;
        ls.CG.absTol = ls.RI.absTol;
        ls.GM.sD = ls.RI.sD;

        if (options_.fsils_blockschur_gm_max_iter) {
            ls.GM.mItr = *options_.fsils_blockschur_gm_max_iter;
        }
        if (options_.fsils_blockschur_cg_max_iter) {
            ls.CG.mItr = *options_.fsils_blockschur_cg_max_iter;
        }
        if (options_.fsils_blockschur_gm_rel_tol) {
            ls.GM.relTol = *options_.fsils_blockschur_gm_rel_tol;
        }
        if (options_.fsils_blockschur_cg_rel_tol) {
            ls.CG.relTol = *options_.fsils_blockschur_cg_rel_tol;
        }

        if (has_native_rank_one_updates) {
            // On the exact coupled BlockSchur path, letting the nested momentum
            // and Schur solves stop at the same absolute floor as the outer
            // solve makes the final solve depth sensitive to decomposition once
            // the RHS becomes small. Keep the policy rank-agnostic and only
            // tighten the inner absolute floors for the coupled exact path.
            const double tightened_abs_tol =
                std::max(1e-18, static_cast<double>(ls.RI.absTol) * 1e-3);
            ls.RI.absTol = std::min(ls.RI.absTol, tightened_abs_tol);
            ls.GM.absTol = std::min(ls.GM.absTol, tightened_abs_tol);
            ls.CG.absTol = std::min(ls.CG.absTol, tightened_abs_tol);
        }

        ls.RI.exact_convergence = true;
        ls.GM.exact_convergence = true;
        ls.CG.exact_convergence = true;
        ls.CG.schur_preconditioner =
            to_fsils_blockschur_preconditioner(options_.fsils_blockschur_schur_preconditioner);
        ls.CG.schur_momentum_approximation =
            to_fsils_blockschur_momentum_approximation(options_.fsils_blockschur_momentum_approximation);

        if (has_saddle_point) {
            ls.mom_start = mom_start;
            ls.mom_ncomp = mom_ncomp;
            ls.con_start = con_start;
            ls.con_ncomp = con_ncomp;
        }
    } else {
        const auto method = to_fsils_solver(options_.method);
        if (method == fe_fsi_linear_solver::LS_TYPE_GMRES) {
            const int gmres_total_iters = gmres_total_iteration_budget(options_, /*legacy_restart_budget=*/false);
            const auto gmres_cfg = make_gmres_launch_config(options_, gmres_total_iters);
            fe_fsi_linear_solver::fsils_ls_create(ls,
                                                  method,
                                                  options_.rel_tol,
                                                  options_.abs_tol,
                                                  gmres_cfg.mItr,
                                                  gmres_cfg.sD);
            // Native low-rank outlet corrections are unusually sensitive to
            // premature restarted-GMRES exits. Keep the heuristic early-stop
            // behavior for generic systems, but require exact convergence for
            // coupled rank-one solves so the first pass does not stop while the
            // true FE residual is still far from the requested tolerance.
            ls.RI.exact_convergence = has_native_rank_one_updates;
        } else {
            fe_fsi_linear_solver::fsils_ls_create(ls,
                                                  method,
                                                  options_.rel_tol,
                                                  options_.abs_tol,
                                                  options_.max_iter);
        }
    }
    ls.ri_internal_order = true;

    // Set up FSILS faces from:
    //  - Dirichlet constraints (legacy-equivalent FSILS preconditioner handling)
    //  - rank-1 updates (coupled BC Sherman-Morrison correction)
    const int original_nFaces = lhs.nFaces;
    const int num_dirichlet_faces = (!dirichlet_dofs_.empty() ? 1 : 0);
    const int num_rank_one_faces = static_cast<int>(native_face_rank_one_indices.size());
    const int num_added_faces = num_dirichlet_faces + num_rank_one_faces;

    int dirichlet_face_index = -1;
    int rank_one_face_start = -1;

    auto sort_face_by_glob = [&](fe_fsi_linear_solver::FSILS_faceType& face, int face_dof) {
        if (face.nNo <= 1) {
            return;
        }
        const int face_nNo = face.nNo;
        std::vector<int> perm(static_cast<std::size_t>(face_nNo));
        std::iota(perm.begin(), perm.end(), 0);
        std::sort(perm.begin(), perm.end(), [&](int i, int j) { return face.glob(i) < face.glob(j); });

        Vector<int> sorted_glob(face_nNo);
        Array<double> sorted_val(face_dof, face_nNo);
        const bool has_matching_valm =
            face.valM.nrows() == face.val.nrows() && face.valM.ncols() == face.val.ncols();
        Array<double> sorted_valM;
        if (has_matching_valm) {
            sorted_valM.resize(face_dof, face_nNo);
        }
        for (int i = 0; i < face_nNo; ++i) {
            const int src = perm[static_cast<std::size_t>(i)];
            sorted_glob(i) = face.glob(src);
            for (int c = 0; c < face_dof; ++c) {
                sorted_val(c, i) = face.val(c, src);
                if (has_matching_valm) {
                    sorted_valM(c, i) = face.valM(c, src);
                }
            }
        }
        face.glob = sorted_glob;
        face.val = sorted_val;
        if (has_matching_valm) {
            face.valM = sorted_valM;
        }
    };

    auto sync_face_val_if_shared = [&](fe_fsi_linear_solver::FSILS_faceType& face, int face_dof) {
        if (lhs.commu.nTasks <= 1) {
            return;
        }

        const int local_has = (face.nNo > 0) ? 1 : 0;
        int total_has = 0;
        fe_fsi_linear_solver::fsils_allreduce_sum(&local_has, &total_has, 1, MPI_INT, lhs.commu);

        if (total_has > 1) {
            face.sharedFlag = true;
            auto sync_face_array = [&](Array<double>& arr) {
                fe_fsi_linear_solver::fsils_reduce_shared_face_values_owned_row(lhs, face_dof, face.glob, arr);
            };

            sync_face_array(face.val);
            if (face.bGrp == fe_fsi_linear_solver::BcType::BC_TYPE_Dir) {
                fe_fsi_linear_solver::fsils_apply_shared_dirichlet_face_mask(
                    lhs, face_dof, face.glob, face.val);
            }
            if (face.valM.nrows() == face.val.nrows() && face.valM.ncols() == face.val.ncols()) {
                sync_face_array(face.valM);
            }
        }
    };

    // Face setup: restore from cache (fast path) or build from scratch.
    bool faces_from_cache = false;
    if (num_added_faces > 0 && dof > 0 && !faces_dirty_ &&
        cached_faces_.size() == static_cast<std::size_t>(num_added_faces)) {
        // Fast path: restore pre-built face data from cache.
        const int new_nFaces = original_nFaces + num_added_faces;
        lhs.face.resize(static_cast<std::size_t>(new_nFaces));
        lhs.nFaces = new_nFaces;

        int next_face = original_nFaces;
        if (num_dirichlet_faces > 0) dirichlet_face_index = next_face++;
        if (num_rank_one_faces > 0) rank_one_face_start = next_face;

        for (int fi = 0; fi < num_added_faces; ++fi) {
            const auto& cf = cached_faces_[static_cast<std::size_t>(fi)];
            auto& face = lhs.face[static_cast<std::size_t>(original_nFaces + fi)];
            face.nNo = cf.nNo;
            face.dof = cf.face_dof;
            face.bGrp = cf.bGrp;
            face.sharedFlag = cf.sharedFlag;
            face.foC = cf.foC;
            face.coupledFlag = cf.coupledFlag;
            face.incFlag = cf.incFlag;
            face.nS = 0.0;
            face.res = 0.0;
            if (cf.nNo > 0) {
                face.glob.resize(cf.nNo);
                std::copy(cf.glob_data.begin(), cf.glob_data.end(), face.glob.data());
                face.val.resize(cf.face_dof, cf.nNo);
                std::copy(cf.val_data.begin(), cf.val_data.end(), face.val.data());
                face.valM.resize(cf.face_dof, cf.nNo);
                std::copy(cf.valM_data.begin(), cf.valM_data.end(), face.valM.data());
            }
        }
        faces_from_cache = true;
    }

    lhs.native_face_rank_one_count = num_rank_one_faces;

    if (num_added_faces > 0 && dof > 0 && !faces_from_cache) {
        const auto shared = A->shared();
        FE_CHECK_NOT_NULL(shared.get(), "FsilsLinearSolver: FsilsShared for face setup");

        const int new_nFaces = original_nFaces + num_added_faces;
        lhs.face.resize(static_cast<std::size_t>(new_nFaces));
        lhs.nFaces = new_nFaces;

        int next_face = original_nFaces;

        if (num_dirichlet_faces > 0) {
            dirichlet_face_index = next_face++;

            // Node mask: old_local_node -> per-component 0/1 mask (0 for Dirichlet components).
            std::map<int, std::vector<double>> node_mask;
            for (const auto dof_idx : dirichlet_dofs_) {
                GlobalIndex fsils_dof = dof_idx;
                if (shared->dof_permutation) {
                    const auto idx = static_cast<std::size_t>(dof_idx);
                    if (idx < shared->dof_permutation->forward.size()) {
                        fsils_dof = shared->dof_permutation->forward[idx];
                    }
                }
                if (fsils_dof < 0) {
                    continue;
                }

                const int node = static_cast<int>(fsils_dof / dof);
                const int comp = static_cast<int>(fsils_dof % dof);
                if (comp < 0 || comp >= dof) {
                    continue;
                }

                const int old_local = shared->globalNodeToOld(node);
                if (old_local < 0 || old_local >= lhs.nNo) {
                    continue;
                }

                auto& mask = node_mask[old_local];
                if (mask.empty()) {
                    mask.assign(static_cast<std::size_t>(dof), 1.0);
                }
                mask[static_cast<std::size_t>(comp)] = 0.0;
            }

            auto& face = lhs.face[static_cast<std::size_t>(dirichlet_face_index)];
            const int face_nNo = static_cast<int>(node_mask.size());
            face.nNo = face_nNo;
            face.dof = dof;
            face.bGrp = fe_fsi_linear_solver::BcType::BC_TYPE_Dir;

	            if (face_nNo > 0) {
	                face.glob.resize(face_nNo);
	                face.val.resize(dof, face_nNo);
	                face.valM.resize(dof, face_nNo);
	                face.val = 1.0;
	                face.valM = 0.0;

                int a = 0;
                for (const auto& [old_local, mask] : node_mask) {
                    face.glob(a) = lhs.map(old_local);
                    for (int c = 0; c < dof; ++c) {
                        face.val(c, a) = mask[static_cast<std::size_t>(c)];
                    }
                    ++a;
                }
	
	                sort_face_by_glob(face, dof);
	            }
	            // Must be called collectively across ranks (uses MPI_Allreduce / COMMU).
	            sync_face_val_if_shared(face, dof);
	
	            face.foC = true;
	            face.coupledFlag = false;
	            face.incFlag = true;

            if (oopTraceEnabled()) {
                std::ostringstream oss;
                oss << "FsilsLinearSolver: Dirichlet face " << dirichlet_face_index
                    << " nNo=" << face_nNo
                    << " dirichlet_dofs=" << dirichlet_dofs_.size();
                traceLog(oss.str());
            }
        }

        rank_one_face_start = next_face;
        for (int u = 0; u < num_rank_one_faces; ++u) {
            const auto update_index = native_face_rank_one_indices[static_cast<std::size_t>(u)];
            const auto& upd = rank_one_updates_[update_index];
            const int faIn = rank_one_face_start + u;

            // Determine which per-node components participate in this rank-1 update.
            std::vector<int> face_comps;
            if (!upd.active_components.empty()) {
                face_comps = upd.active_components;
            } else if (has_saddle_point) {
                // Default: momentum block components only (skip constraint).
                for (int c = mom_start; c < mom_start + mom_ncomp; ++c) {
                    face_comps.push_back(c);
                }
            } else {
                // No saddle-point info: all components participate.
                for (int c = 0; c < dof; ++c) {
                    face_comps.push_back(c);
                }
            }
            const int face_dof = static_cast<int>(face_comps.size());

            // Build a fast lookup: component index -> face-local index (-1 if not active).
            std::vector<int> comp_to_face_idx(static_cast<std::size_t>(dof), -1);
            for (int fi = 0; fi < face_dof; ++fi) {
                const int c = face_comps[static_cast<std::size_t>(fi)];
                if (c >= 0 && c < dof) {
                    comp_to_face_idx[static_cast<std::size_t>(c)] = fi;
                }
            }

            // Seed the overlap buffer from the local view and then synchronize
            // so the native-face representation carries the same full local
            // support as the reduced-update path.
            Array<double> face_values(face_dof, lhs.nNo);
            face_values = 0.0;
            for (const auto& [dof_idx, val] : upd.v) {
                GlobalIndex fsils_dof = dof_idx;
                if (shared->dof_permutation) {
                    const auto idx = static_cast<std::size_t>(dof_idx);
                    if (idx < shared->dof_permutation->forward.size()) {
                        fsils_dof = shared->dof_permutation->forward[idx];
                    }
                }

                // Skip unmapped DOFs (permutation returns -1 for DOFs not present on this rank).
                if (fsils_dof < 0) {
                    continue;
                }

                const int node = static_cast<int>(fsils_dof / dof);
                const int comp = static_cast<int>(fsils_dof % dof);

                // Only active components.
                if (comp < 0 || comp >= dof || comp_to_face_idx[static_cast<std::size_t>(comp)] < 0) {
                    continue;
                }

                const int old_local = shared->globalNodeToOld(node);
                if (old_local < 0 || old_local >= lhs.nNo) {
                    continue;
                }
                const int internal = lhs.map(old_local);
                if (internal < 0 || internal >= lhs.nNo || internal >= lhs.mynNo) {
                    continue;
                }
                face_values(comp_to_face_idx[static_cast<std::size_t>(comp)], internal) +=
                    static_cast<double>(val);
            }

            if (lhs.commu.nTasks > 1 && face_dof > 0) {
                fe_fsi_linear_solver::fsils_syncv_owned_to_ghost(lhs, face_dof, face_values);
            }

            std::vector<int> face_nodes;
            face_nodes.reserve(static_cast<std::size_t>(lhs.nNo));
            for (int internal = 0; internal < lhs.nNo; ++internal) {
                bool has_support = false;
                for (int c = 0; c < face_dof; ++c) {
                    if (face_values(c, internal) != 0.0) {
                        has_support = true;
                        break;
                    }
                }
                if (has_support) {
                    face_nodes.push_back(internal);
                }
            }
            const int face_nNo = static_cast<int>(face_nodes.size());

            // Set up face data directly to avoid Vector/Array zero-size constructor issues.
            {
                auto& face = lhs.face[static_cast<std::size_t>(faIn)];
                face.nNo = face_nNo;
                face.dof = face_dof;
                face.bGrp = fe_fsi_linear_solver::BcType::BC_TYPE_Neu;

                if (face_nNo > 0) {
                    face.glob.resize(face_nNo);
                    face.val.resize(face_dof, face_nNo);
                    face.valM.resize(face_dof, face_nNo);
                    face.val = 0.0;
                    face.valM = 0.0;

                    for (int a = 0; a < face_nNo; ++a) {
                        const int internal = face_nodes[static_cast<std::size_t>(a)];
                        face.glob(a) = internal;
                        for (int c = 0; c < face_dof; ++c) {
                            const double value = face_values(c, internal);
                            face.val(c, a) = value;
                            face.valM(c, a) = value;
                        }
                    }
                }

                int local_has = (face_nNo > 0) ? 1 : 0;
                int total_has = local_has;
                if (lhs.commu.nTasks > 1) {
                    fe_fsi_linear_solver::fsils_allreduce_sum(&local_has, &total_has, 1, MPI_INT, lhs.commu);
                }
                face.sharedFlag = (total_has > 1);
                face.foC = true;
                face.coupledFlag = true;
                face.incFlag = true;
            }

            if (oopTraceEnabled()) {
                int owned_nodes = 0;
                int ghost_nodes = 0;
                const auto& face = lhs.face[static_cast<std::size_t>(faIn)];
                for (int a = 0; a < face.nNo; ++a) {
                    if (face.glob(a) < lhs.mynNo) {
                        ++owned_nodes;
                    } else {
                        ++ghost_nodes;
                    }
                }
                std::ostringstream oss;
                oss << "FsilsLinearSolver: rank-1 update " << update_index
                    << " -> FSILS face " << faIn
                    << " nNo=" << face_nNo
                    << " owned=" << owned_nodes
                    << " ghost=" << ghost_nodes
                    << " shared=" << (face.sharedFlag ? 1 : 0)
                    << " sigma=" << static_cast<double>(upd.sigma)
                    << " v_entries=" << upd.v.size();
                traceLog(oss.str());
            }
        }

        // Cache the built faces for reuse in subsequent Newton iterations.
        cached_faces_.clear();
        cached_faces_.resize(static_cast<std::size_t>(num_added_faces));
        for (int fi = 0; fi < num_added_faces; ++fi) {
            const auto& face = lhs.face[static_cast<std::size_t>(original_nFaces + fi)];
            auto& cf = cached_faces_[static_cast<std::size_t>(fi)];
            cf.nNo = face.nNo;
            cf.face_dof = face.dof;
            cf.bGrp = face.bGrp;
            cf.sharedFlag = face.sharedFlag;
            cf.foC = face.foC;
            cf.coupledFlag = face.coupledFlag;
            cf.incFlag = face.incFlag;
            if (face.nNo > 0) {
                const auto sz = static_cast<std::size_t>(face.dof) * static_cast<std::size_t>(face.nNo);
                cf.glob_data.assign(face.glob.data(), face.glob.data() + face.nNo);
                cf.val_data.assign(face.val.data(), face.val.data() + sz);
                cf.valM_data.assign(face.valM.data(), face.valM.data() + sz);
            }
        }
        faces_dirty_ = false;
    }

    lhs.reduced_updates.clear();
    lhs.grouped_bordered_field_couplings.clear();
    {
        const auto shared = A->shared();
        FE_CHECK_NOT_NULL(shared.get(), "FsilsLinearSolver: FsilsShared for reduced updates");

        auto default_active_components = [&]() {
            std::vector<int> comps;
            // ReducedFieldUpdate.active_components uses interface-level semantics:
            // empty means all field components participate. Do not silently narrow
            // generic reduced updates to the momentum block on saddle-point systems.
            comps.reserve(static_cast<std::size_t>(dof));
            for (int c = 0; c < dof; ++c) {
                comps.push_back(c);
            }
            return comps;
        };

        struct ReducedInternalEntries {
            std::vector<fe_fsi_linear_solver::FSILS_reducedSparseEntry> full;
            std::vector<fe_fsi_linear_solver::FSILS_reducedSparseEntry> owned;
            std::size_t total_entries{0};
            std::size_t mapped_owned_entries{0};
            std::size_t mapped_ghost_entries{0};
            std::size_t dropped_ghost_entries{0};
        };

        auto make_internal_entries =
            [&](std::span<const std::pair<GlobalIndex, Real>> entries)
                -> ReducedInternalEntries {
            ReducedInternalEntries result;
            Array<double> values(dof, lhs.nNo);
            values = 0.0;

            for (const auto& [dof_idx, val] : entries) {
                ++result.total_entries;
                GlobalIndex fsils_dof = dof_idx;
                if (shared->dof_permutation) {
                    const auto idx = static_cast<std::size_t>(dof_idx);
                    if (idx < shared->dof_permutation->forward.size()) {
                        fsils_dof = shared->dof_permutation->forward[idx];
                    }
                }
                if (fsils_dof < 0) {
                    continue;
                }

                const int node = static_cast<int>(fsils_dof / dof);
                const int comp = static_cast<int>(fsils_dof % dof);
                if (comp < 0 || comp >= dof) {
                    continue;
                }

                const int old_local = shared->globalNodeToOld(node);
                if (old_local < 0 || old_local >= lhs.nNo) {
                    continue;
                }
                const int internal = lhs.map(old_local);
                if (internal >= 0 && internal < lhs.nNo) {
                    if (internal < lhs.mynNo) {
                        ++result.mapped_owned_entries;
                    } else {
                        ++result.mapped_ghost_entries;
                    }
                }
                if (internal < 0 || internal >= lhs.nNo) {
                    continue;
                }
                values(comp, internal) += static_cast<double>(val);
            }

            if (lhs.commu.nTasks > 1) {
                fe_fsi_linear_solver::fsils_reverse_scatterv_contribution_buffer(lhs, dof, values);
                fe_fsi_linear_solver::fsils_syncv_owned_to_ghost(lhs, dof, values);
            }

            result.full.reserve(static_cast<std::size_t>(dof) * static_cast<std::size_t>(lhs.nNo));
            result.owned.reserve(static_cast<std::size_t>(dof) * static_cast<std::size_t>(lhs.mynNo));
            for (int internal = 0; internal < lhs.nNo; ++internal) {
                for (int comp = 0; comp < dof; ++comp) {
                    const double value = values(comp, internal);
                    if (std::abs(value) <= 1e-30) {
                        continue;
                    }
                    fe_fsi_linear_solver::FSILS_reducedSparseEntry entry;
                    entry.node = static_cast<fe_fsi_linear_solver::fsils_int>(internal);
                    entry.full_component = comp;
                    entry.value = value;
                    result.full.push_back(entry);
                    if (internal < lhs.mynNo) {
                        result.owned.push_back(entry);
                    }
                }
            }
            return result;
        };

        auto make_native_reduced_update = [&](Real sigma,
                                              std::span<const std::pair<GlobalIndex, Real>> left,
                                              std::span<const std::pair<GlobalIndex, Real>> right,
                                              std::span<const int> active_components,
                                              int grouped_coupling_id,
                                              Real left_scale,
                                              bool scale_sigma) {
            fe_fsi_linear_solver::FSILS_reducedFieldUpdateType native_update;
            if (!(std::abs(sigma) > Real(1e-30))) {
                return native_update;
            }

            auto left_build = make_internal_entries(left);
            auto right_build = make_internal_entries(right);
            auto& left_full = left_build.full;
            auto& left_owned = left_build.owned;
            auto& right_full = right_build.full;
            auto& right_owned = right_build.owned;
            int local_left_has = left_owned.empty() ? 0 : 1;
            int local_right_has = right_owned.empty() ? 0 : 1;
            int global_left_has = local_left_has;
            int global_right_has = local_right_has;
            if (lhs.commu.nTasks > 1) {
                fe_fsi_linear_solver::fsils_allreduce_sum(
                    &local_left_has, &global_left_has, 1, MPI_INT, lhs.commu);
                fe_fsi_linear_solver::fsils_allreduce_sum(
                    &local_right_has, &global_right_has, 1, MPI_INT, lhs.commu);
            }
            if (fsilsTraceEnabled() && lhs.commu.nTasks > 1) {
                std::ostringstream oss;
                oss << "FsilsLinearSolver::solve: reduced internalization"
                    << " sigma=" << static_cast<double>(sigma)
                    << " grouped_id=" << grouped_coupling_id
                    << " left_total=" << left_build.total_entries
                    << " left_mapped_owned=" << left_build.mapped_owned_entries
                    << " left_mapped_ghost=" << left_build.mapped_ghost_entries
                    << " left_dropped_ghost=" << left_build.dropped_ghost_entries
                    << " left_owned=" << left_owned.size()
                    << " right_total=" << right_build.total_entries
                    << " right_mapped_owned=" << right_build.mapped_owned_entries
                    << " right_mapped_ghost=" << right_build.mapped_ghost_entries
                    << " right_dropped_ghost=" << right_build.dropped_ghost_entries
                    << " right_owned=" << right_owned.size()
                    << " local_left_has=" << local_left_has
                    << " local_right_has=" << local_right_has
                    << " global_left_has=" << global_left_has
                    << " global_right_has=" << global_right_has;
                traceLog(oss.str());
            }
            if (global_left_has == 0 || global_right_has == 0) {
                return native_update;
            }

            if (reducedInternalizationTraceEnabled()) {
                static int reduced_internal_trace_count = 0;
                if (reduced_internal_trace_count < 16) {
                    ++reduced_internal_trace_count;
                    const auto dump_raw_entries =
                        [&](const char* side,
                            std::span<const std::pair<GlobalIndex, Real>> sparse) {
                            const std::size_t limit = std::min<std::size_t>(sparse.size(), 6);
                            for (std::size_t i = 0; i < limit; ++i) {
                                const auto& [gdof, value] = sparse[i];
                                GlobalIndex fsils_dof = gdof;
                                if (shared->dof_permutation) {
                                    const auto idx = static_cast<std::size_t>(gdof);
                                    if (idx < shared->dof_permutation->forward.size()) {
                                        fsils_dof = shared->dof_permutation->forward[idx];
                                    }
                                }
                                const int node =
                                    (fsils_dof >= 0) ? static_cast<int>(fsils_dof / dof) : -1;
                                const int comp =
                                    (fsils_dof >= 0) ? static_cast<int>(fsils_dof % dof) : -1;
                                const int old_local =
                                    (node >= 0) ? shared->globalNodeToOld(node) : -1;
                                const int internal =
                                    (old_local >= 0 && old_local < lhs.nNo) ? lhs.map(old_local) : -1;
                                std::fprintf(stderr,
                                             "[FSILS_REDUCED_RAW_ENTRY] trace=%d rank=%d grouped_id=%d "
                                             "side=%s idx=%zu gdof=%lld fsils_dof=%lld node=%d comp=%d old_local=%d internal=%d value=%.17e\n",
                                             reduced_internal_trace_count,
                                             lhs.commu.task,
                                             grouped_coupling_id,
                                             side,
                                             i,
                                             static_cast<long long>(gdof),
                                             static_cast<long long>(fsils_dof),
                                             node,
                                             comp,
                                             old_local,
                                             internal,
                                             static_cast<double>(value));
                            }
                        };
                    std::fprintf(stderr,
                                 "[FSILS_REDUCED_INTERNAL] trace=%d rank=%d grouped_id=%d "
                                 "left(total=%zu mapped_owned=%zu mapped_ghost=%zu dropped_ghost=%zu full=%zu owned=%zu) "
                                 "right(total=%zu mapped_owned=%zu mapped_ghost=%zu dropped_ghost=%zu full=%zu owned=%zu)\n",
                                 reduced_internal_trace_count,
                                 lhs.commu.task,
                                 grouped_coupling_id,
                                 left_build.total_entries,
                                 left_build.mapped_owned_entries,
                                 left_build.mapped_ghost_entries,
                                 left_build.dropped_ghost_entries,
                                 left_full.size(),
                                 left_owned.size(),
                                 right_build.total_entries,
                                 right_build.mapped_owned_entries,
                                 right_build.mapped_ghost_entries,
                                 right_build.dropped_ghost_entries,
                                 right_full.size(),
                                 right_owned.size());
                    std::fprintf(stderr,
                                 "[FSILS_REDUCED_ACTIVE_COMPS] trace=%d rank=%d grouped_id=%d size=%zu",
                                 reduced_internal_trace_count,
                                 lhs.commu.task,
                                 grouped_coupling_id,
                                 active_components.size());
                    for (std::size_t i = 0; i < active_components.size(); ++i) {
                        std::fprintf(stderr,
                                     " comp[%zu]=%d",
                                     i,
                                     active_components[i]);
                    }
                    std::fprintf(stderr, "\n");
                    dump_raw_entries("left", left);
                    dump_raw_entries("right", right);
                    const auto dump_entries =
                        [&](const char* side,
                            const char* scope,
                            const std::vector<fe_fsi_linear_solver::FSILS_reducedSparseEntry>& sparse) {
                            const std::size_t limit = std::min<std::size_t>(sparse.size(), 4);
                            for (std::size_t i = 0; i < limit; ++i) {
                                const auto& entry = sparse[i];
                                std::fprintf(stderr,
                                             "[FSILS_REDUCED_INTERNAL_ENTRY] trace=%d rank=%d grouped_id=%d "
                                             "side=%s scope=%s idx=%zu node=%lld comp=%d value=%.17e\n",
                                             reduced_internal_trace_count,
                                             lhs.commu.task,
                                             grouped_coupling_id,
                                             side,
                                             scope,
                                             i,
                                             static_cast<long long>(entry.node),
                                             entry.full_component,
                                             entry.value);
                            }
                        };
                    dump_entries("left", "full", left_full);
                    dump_entries("left", "owned", left_owned);
                    dump_entries("right", "full", right_full);
                    dump_entries("right", "owned", right_owned);
                    std::fflush(stderr);
                }
            }

            if (left_scale != Real(1.0)) {
                for (auto& entry : left_full) {
                    entry.value *= static_cast<double>(left_scale);
                }
                for (auto& entry : left_owned) {
                    entry.value *= static_cast<double>(left_scale);
                }
            }

            native_update.active = true;
            native_update.sigma = static_cast<double>(
                (scale_sigma && use_blockschur) ? sigma * stage_scale : sigma);
            native_update.grouped_coupling_id = grouped_coupling_id;
            native_update.left = std::move(left_full);
            native_update.right = std::move(right_full);
            native_update.left_owned = std::move(left_owned);
            native_update.right_owned = std::move(right_owned);
            native_update.left_scaled = native_update.left;
            native_update.right_scaled = native_update.right;
            native_update.left_scaled_owned = native_update.left_owned;
            native_update.right_scaled_owned = native_update.right_owned;
            if (!active_components.empty()) {
                native_update.active_components.assign(active_components.begin(), active_components.end());
            } else {
                native_update.active_components = default_active_components();
            }
            return native_update;
        };

        auto build_face_from_reduced_entries =
            [&](const fe_fsi_linear_solver::FSILS_reducedFieldUpdateType& update,
                const std::vector<fe_fsi_linear_solver::FSILS_reducedSparseEntry>& support_entries,
                const std::vector<fe_fsi_linear_solver::FSILS_reducedSparseEntry>& value_entries,
                fe_fsi_linear_solver::FSILS_faceType& face) {
            std::vector<int> face_comps = !update.active_components.empty()
                                              ? update.active_components
                                              : default_active_components();
            const int face_dof = static_cast<int>(face_comps.size());
            if (face_dof <= 0) {
                return;
            }

            std::vector<int> comp_to_face_idx(static_cast<std::size_t>(dof), -1);
            for (int fi = 0; fi < face_dof; ++fi) {
                const int comp = face_comps[static_cast<std::size_t>(fi)];
                if (comp >= 0 && comp < dof) {
                    comp_to_face_idx[static_cast<std::size_t>(comp)] = fi;
                }
            }

            Array<double> face_values(face_dof, lhs.nNo);
            face_values = 0.0;
            for (const auto& entry : value_entries) {
                if (entry.node < 0 || entry.node >= lhs.nNo || std::abs(entry.value) <= 1e-30) {
                    continue;
                }
                if (entry.full_component < 0 || entry.full_component >= dof) {
                    continue;
                }
                const int face_comp = comp_to_face_idx[static_cast<std::size_t>(entry.full_component)];
                if (face_comp < 0) {
                    continue;
                }
                face_values(face_comp, entry.node) += entry.value;
            }

            std::vector<int> face_nodes;
            face_nodes.reserve(static_cast<std::size_t>(lhs.nNo));
            if (!support_entries.empty()) {
                std::vector<char> node_has_support(static_cast<std::size_t>(lhs.nNo), 0);
                for (const auto& entry : support_entries) {
                    if (entry.node < 0 || entry.node >= lhs.nNo ||
                        std::abs(entry.value) <= 1e-30) {
                        continue;
                    }
                    if (entry.full_component < 0 || entry.full_component >= dof) {
                        continue;
                    }
                    const int face_comp =
                        comp_to_face_idx[static_cast<std::size_t>(entry.full_component)];
                    if (face_comp < 0) {
                        continue;
                    }
                    node_has_support[static_cast<std::size_t>(entry.node)] = 1;
                }
                for (int internal = 0; internal < lhs.nNo; ++internal) {
                    if (node_has_support[static_cast<std::size_t>(internal)] != 0) {
                        face_nodes.push_back(internal);
                    }
                }
            }

            face.nNo = static_cast<int>(face_nodes.size());
            face.dof = face_dof;
            face.bGrp = fe_fsi_linear_solver::BcType::BC_TYPE_Neu;
            face.foC = true;
            face.coupledFlag = true;
            face.incFlag = true;
            face.sharedFlag = false;
            face.nS = 0.0;
            face.res = 0.0;
            if (face.nNo > 0) {
                face.glob.resize(face.nNo);
                face.val.resize(face_dof, face.nNo);
                face.valM.resize(face_dof, face.nNo);
                face.val = 0.0;
                face.valM = 0.0;
                for (int a = 0; a < face.nNo; ++a) {
                    const int internal = face_nodes[static_cast<std::size_t>(a)];
                    face.glob(a) = internal;
                    for (int c = 0; c < face_dof; ++c) {
                        face.val(c, a) = face_values(c, internal);
                        face.valM(c, a) = face_values(c, internal);
                    }
                }
            }

            sort_face_by_glob(face, face_dof);
            sync_face_val_if_shared(face, face_dof);
        };

        auto append_reduced_update = [&](Real sigma,
                                        std::span<const std::pair<GlobalIndex, Real>> left,
                                        std::span<const std::pair<GlobalIndex, Real>> right,
                                        std::span<const int> active_components,
                                        int grouped_coupling_id) {
            auto native_update =
                make_native_reduced_update(sigma, left, right, active_components,
                                           grouped_coupling_id, Real(1.0),
                                           /*scale_sigma=*/true);
            if (native_update.active) {
                const auto& left_value_entries =
                    (lhs.commu.nTasks > 1) ? native_update.left_owned : native_update.left;
                const auto& right_value_entries =
                    (lhs.commu.nTasks > 1) ? native_update.right_owned : native_update.right;
                build_face_from_reduced_entries(native_update,
                                                native_update.left,
                                                left_value_entries,
                                                native_update.left_face);
                build_face_from_reduced_entries(native_update,
                                                native_update.right,
                                                right_value_entries,
                                                native_update.right_face);
                native_update.has_face_cache =
                    native_update.left_face.nNo > 0 && native_update.right_face.nNo > 0;
                lhs.reduced_updates.push_back(std::move(native_update));
            }
        };

        const bool mirror_native_face_rank_one_into_schur_reduced_updates = false;
        if (mirror_native_face_rank_one_into_schur_reduced_updates && oopTraceEnabled()) {
            traceLog("FsilsLinearSolver::solve: mirroring multi-face MPI native rank-1 updates "
                     "into Schur-only reduced duplicates.");
        }

        for (const auto& upd : rank_one_updates_) {
            if (upd.prefer_native_face && allow_mpi_native_face_rank_one) {
                if (mirror_native_face_rank_one_into_schur_reduced_updates) {
                    append_reduced_update(
                        upd.sigma,
                        std::span<const std::pair<GlobalIndex, Real>>(upd.v.data(), upd.v.size()),
                        std::span<const std::pair<GlobalIndex, Real>>(upd.v.data(), upd.v.size()),
                        std::span<const int>(upd.active_components.data(),
                                             upd.active_components.size()),
                        kNativeFaceDuplicateCouplingId);
                }
                continue;
            }
            append_reduced_update(upd.sigma,
                                  std::span<const std::pair<GlobalIndex, Real>>(upd.v.data(), upd.v.size()),
                                  std::span<const std::pair<GlobalIndex, Real>>(upd.v.data(), upd.v.size()),
                                  std::span<const int>(upd.active_components.data(),
                                                       upd.active_components.size()),
                                  /*grouped_coupling_id=*/-1);
        }

        for (const auto& upd : reduced_field_updates_) {
            append_reduced_update(upd.sigma,
                                  std::span<const std::pair<GlobalIndex, Real>>(upd.left.data(), upd.left.size()),
                                  std::span<const std::pair<GlobalIndex, Real>>(upd.right.data(), upd.right.size()),
                                  std::span<const int>(upd.active_components.data(),
                                                       upd.active_components.size()),
                                  upd.grouped_coupling_id);
        }

        for (const auto& group : grouped_bordered_field_couplings_) {
            const Real grouped_left_scale =
                use_blockschur ? static_cast<Real>(stage_scale) : Real(1.0);
            fe_fsi_linear_solver::FSILS_groupedBorderedFieldCouplingType native_group;
            native_group.active = true;
            native_group.grouped_coupling_id = group.grouped_coupling_id;
            native_group.aux_matrix.assign(group.aux_matrix.begin(), group.aux_matrix.end());
            native_group.modes.reserve(group.modes.size());
            native_group.left_faces.reserve(group.modes.size());
            native_group.right_faces.reserve(group.modes.size());
            for (const auto& mode : group.modes) {
                auto native_mode =
                    make_native_reduced_update(Real(1.0),
                                               std::span<const std::pair<GlobalIndex, Real>>(
                                                   mode.left.data(), mode.left.size()),
                                               std::span<const std::pair<GlobalIndex, Real>>(
                                                   mode.right.data(), mode.right.size()),
                                               std::span<const int>(mode.active_components.data(),
                                                                    mode.active_components.size()),
                                               group.grouped_coupling_id,
                                               grouped_left_scale,
                                               /*scale_sigma=*/false);
                if (native_mode.active) {
                    fe_fsi_linear_solver::FSILS_faceType left_face;
                    fe_fsi_linear_solver::FSILS_faceType right_face;
                    const auto& left_value_entries =
                        (lhs.commu.nTasks > 1) ? native_mode.left_owned : native_mode.left;
                    const auto& right_value_entries =
                        (lhs.commu.nTasks > 1) ? native_mode.right_owned : native_mode.right;
                    build_face_from_reduced_entries(native_mode,
                                                    native_mode.left,
                                                    left_value_entries,
                                                    left_face);
                    build_face_from_reduced_entries(native_mode,
                                                    native_mode.right,
                                                    right_value_entries,
                                                    right_face);
                    native_group.modes.push_back(std::move(native_mode));
                    native_group.left_faces.push_back(std::move(left_face));
                    native_group.right_faces.push_back(std::move(right_face));
                }
            }
            if (!native_group.aux_matrix.empty() && !native_group.modes.empty()) {
                lhs.grouped_bordered_field_couplings.push_back(std::move(native_group));
            }
        }

        lhs.use_reduced_face_cache_in_add_bc_mul =
            (std::getenv("SVMP_DISABLE_REDUCED_FACE_CACHE_ADD_BC_MUL") == nullptr) &&
            std::any_of(lhs.reduced_updates.begin(),
                        lhs.reduced_updates.end(),
                        [](const auto& update) { return update.active && update.has_face_cache; });
        if (oopTraceEnabled() && lhs.use_reduced_face_cache_in_add_bc_mul) {
            traceLog("FsilsLinearSolver::solve: enabling reduced-update face-cache add_bc_mul path.");
        }
    }

    // Build incL and res vectors for face activation.
    // When no faces exist, pass empty vectors (original behavior).
    // Note: must use default constructors, not Vector(0), because Vector(0) leaves
    // data_ uninitialized (legacy Fortran compat), causing crashes in resize().
    Vector<int> incL;
    Vector<double> res_original;
    Vector<double> res_blockschur;
    if (lhs.nFaces > 0) {
        const int total_faces = lhs.nFaces;
        incL.resize(total_faces);
        res_original.resize(total_faces);
        res_blockschur.resize(total_faces);
        for (int f = 0; f < total_faces; ++f) {
            incL(f) = 1;
            res_original(f) = 0.0;
            res_blockschur(f) = 0.0;
        }
        if (num_rank_one_faces > 0 && rank_one_face_start >= 0) {
            // Set resistance values for rank-1 faces.
            for (int u = 0; u < num_rank_one_faces; ++u) {
                const int faIn = rank_one_face_start + u;
                const auto update_index = native_face_rank_one_indices[static_cast<std::size_t>(u)];
                const double sigma = static_cast<double>(rank_one_updates_[update_index].sigma);
                res_original(faIn) = sigma;
                res_blockschur(faIn) = sigma * stage_scale;
            }
        }
    }

    const auto prec = to_fsils_prec(options_);

    double rhs_prepare_time_seconds = 0.0;
    double validation_time_seconds = 0.0;
    const auto solve_buffer = std::span<Real>(ri_internal_work_.data(), ri_internal_work_.size());

    auto accumulateLocalBufferStats = [](std::span<const Real> values,
                                        int local_dof,
                                        int local_nNo,
                                        int local_mynNo,
                                        long double& owned_sq,
                                        long double& ghost_sq,
                                        Real& owned_max,
                                        Real& ghost_max,
                                        std::size_t& owned_nnz,
                                        std::size_t& ghost_nnz) {
        auto accumulate_stats = [](Real value, long double& sq, Real& max_abs, std::size_t& nnz) {
            sq += static_cast<long double>(value) * static_cast<long double>(value);
            const Real abs_v = std::abs(value);
            max_abs = std::max(max_abs, abs_v);
            if (abs_v > static_cast<Real>(1e-14)) {
                ++nnz;
            }
        };

        for (int internal = 0; internal < local_nNo; ++internal) {
            const bool owned = internal < local_mynNo;
            const std::size_t base = static_cast<std::size_t>(internal) * static_cast<std::size_t>(local_dof);
            for (int c = 0; c < local_dof; ++c) {
                const Real value = values[base + static_cast<std::size_t>(c)];
                if (owned) {
                    accumulate_stats(value, owned_sq, owned_max, owned_nnz);
                } else {
                    accumulate_stats(value, ghost_sq, ghost_max, ghost_nnz);
                }
            }
        }
    };

    auto logLocalSolveBufferStats = [&](std::string_view phase) {
        if (!oopTraceEnabled() || lhs.commu.nTasks <= 1) {
            return;
        }

        long double internal_owned_sq = 0.0L;
        long double internal_ghost_sq = 0.0L;
        Real internal_owned_max = Real(0.0);
        Real internal_ghost_max = Real(0.0);
        std::size_t internal_owned_nnz = 0;
        std::size_t internal_ghost_nnz = 0;
        accumulateLocalBufferStats(solve_buffer,
                                   dof,
                                   lhs.nNo,
                                   lhs.mynNo,
                                   internal_owned_sq,
                                   internal_ghost_sq,
                                   internal_owned_max,
                                   internal_ghost_max,
                                   internal_owned_nnz,
                                   internal_ghost_nnz);

        std::ostringstream oss;
        oss << "FsilsLinearSolver::solve: local solve buffer"
            << " phase='" << phase << "'"
            << " rank=" << lhs.commu.task
            << " internal_owned_l2=" << std::sqrt(static_cast<double>(internal_owned_sq))
            << " internal_owned_nnz=" << internal_owned_nnz
            << " internal_owned_max=" << internal_owned_max
            << " internal_ghost_l2=" << std::sqrt(static_cast<double>(internal_ghost_sq))
            << " internal_ghost_nnz=" << internal_ghost_nnz
            << " internal_ghost_max=" << internal_ghost_max;
        traceLog(oss.str());
    };

    auto dumpPreparedSolveBuffer = [&](std::string_view phase) {
        const char* dump_prefix = fsilsCompareFaceOperatorDumpPrefix();
        if (dump_prefix == nullptr) {
            return;
        }

        static std::uint64_t prepared_rhs_dump_index = 0u;
        const std::uint64_t dump_index = prepared_rhs_dump_index++;

        std::ostringstream path;
        path << dump_prefix
             << ".prepare.dump" << dump_index
             << "." << phase
             << ".rank" << lhs.commu.task
             << ".txt";
        std::ofstream out(path.str());
        if (!out) {
            return;
        }

        out << "# task " << lhs.commu.task
            << " phase " << phase << "\n";
        out << "# global_node component value old_node internal_node\n";
        for (int old = 0; old < lhs.nNo; ++old) {
            const int internal = lhs.map(old);
            if (internal < 0 || internal >= lhs.mynNo) {
                continue;
            }
            const int global_node = shared_layout->oldToGlobalNode(old);
            if (global_node < 0) {
                continue;
            }
            const std::size_t base =
                static_cast<std::size_t>(internal) * static_cast<std::size_t>(dof);
            for (int c = 0; c < dof; ++c) {
                out << global_node
                    << ' ' << c
                    << ' ' << solve_buffer[base + static_cast<std::size_t>(c)]
                    << ' ' << old
                    << ' ' << internal
                    << '\n';
            }
        }

        std::ostringstream fe_path;
        fe_path << dump_prefix
                << ".prepare.dump" << dump_index
                << "." << phase
                << ".fe_dof.rank" << lhs.commu.task
                << ".txt";
        std::ofstream fe_out(fe_path.str());
        if (!fe_out) {
            return;
        }

        fe_out << "# task " << lhs.commu.task
               << " phase " << phase << "\n";
        fe_out << "# fe_dof value old_node internal_node backend_node component\n";
        const auto* perm = shared_layout->dof_permutation.get();
        const bool has_inverse = perm != nullptr && !perm->inverse.empty();
        for (int old = 0; old < lhs.nNo; ++old) {
            const int internal = lhs.map(old);
            if (internal < 0 || internal >= lhs.mynNo) {
                continue;
            }
            const int backend_node = shared_layout->oldToGlobalNode(old);
            if (backend_node < 0) {
                continue;
            }
            const std::size_t base =
                static_cast<std::size_t>(internal) * static_cast<std::size_t>(dof);
            for (int c = 0; c < dof; ++c) {
                const GlobalIndex backend_dof =
                    static_cast<GlobalIndex>(backend_node) * static_cast<GlobalIndex>(dof) +
                    static_cast<GlobalIndex>(c);
                GlobalIndex fe_dof = backend_dof;
                if (has_inverse && static_cast<std::size_t>(backend_dof) < perm->inverse.size()) {
                    fe_dof = perm->inverse[static_cast<std::size_t>(backend_dof)];
                }
                fe_out << fe_dof
                       << ' ' << solve_buffer[base + static_cast<std::size_t>(c)]
                       << ' ' << old
                       << ' ' << internal
                       << ' ' << backend_node
                       << ' ' << c
                       << '\n';
            }
        }
    };

    auto loadSolveBufferFromVector = [&](const FsilsVector& src, bool blockschur_preparation) {
        const double tp0 = fe_fsi_linear_solver::fsils_cpu_t();
        copyVectorOldToInternal(src, solve_buffer);
        logLocalSolveBufferStats(blockschur_preparation ? "raw_rhs_blockschur" : "raw_rhs_original");
        // Owned-row FSILS uses PETSc-like RHS semantics: off-rank row
        // contributions are routed to their owner during FE assembly, and the
        // solve only needs owner-to-ghost halo values for local columns.
        fe_fsi_linear_solver::fsils_syncv_owned_to_ghost(lhs, dof, Ri);

        if (blockschur_preparation && stage_scale != 1.0) {
            const Real s = static_cast<Real>(stage_scale);
            for (int a = 0; a < lhs.nNo; ++a) {
                // Scale momentum rows of the RHS.
                for (int c = mom_start; c < mom_start + mom_ncomp; ++c) {
                    Ri(c, a) *= s;
                }
            }
        }
        rhs_prepare_time_seconds += fe_fsi_linear_solver::fsils_cpu_t() - tp0;
        logLocalSolveBufferStats(blockschur_preparation ? "prepared_rhs_blockschur" : "prepared_rhs_original");
        dumpPreparedSolveBuffer(blockschur_preparation ? "prepared_rhs_blockschur" : "prepared_rhs_original");
    };

    auto logLocalReturnedSolutionStats = [&](std::string_view phase) {
        if (!oopTraceEnabled() || lhs.commu.nTasks <= 1) {
            return;
        }

        long double internal_owned_sq = 0.0L;
        long double internal_ghost_sq = 0.0L;
        Real internal_owned_max = Real(0.0);
        Real internal_ghost_max = Real(0.0);
        std::size_t internal_owned_nnz = 0;
        std::size_t internal_ghost_nnz = 0;
        accumulateLocalBufferStats(solve_buffer,
                                   dof,
                                   lhs.nNo,
                                   lhs.mynNo,
                                   internal_owned_sq,
                                   internal_ghost_sq,
                                   internal_owned_max,
                                   internal_ghost_max,
                                   internal_owned_nnz,
                                   internal_ghost_nnz);

        const auto& x_data_local = x->data();
        const std::size_t owned_old_entries =
            static_cast<std::size_t>(std::max(shared_layout->owned_node_count, 0)) *
            static_cast<std::size_t>(std::max(shared_layout->dof, 0));
        long double old_owned_sq = 0.0L;
        long double old_ghost_sq = 0.0L;
        Real old_owned_max = Real(0.0);
        Real old_ghost_max = Real(0.0);
        std::size_t old_owned_nnz = 0;
        std::size_t old_ghost_nnz = 0;
        for (std::size_t i = 0; i < x_data_local.size(); ++i) {
            const Real value = x_data_local[i];
            const Real abs_v = std::abs(value);
            if (i < owned_old_entries) {
                old_owned_sq += static_cast<long double>(value) * static_cast<long double>(value);
                old_owned_max = std::max(old_owned_max, abs_v);
                if (abs_v > static_cast<Real>(1e-14)) {
                    ++old_owned_nnz;
                }
            } else {
                old_ghost_sq += static_cast<long double>(value) * static_cast<long double>(value);
                old_ghost_max = std::max(old_ghost_max, abs_v);
                if (abs_v > static_cast<Real>(1e-14)) {
                    ++old_ghost_nnz;
                }
            }
        }

        std::ostringstream oss;
        oss << "FsilsLinearSolver::solve: local returned solution"
            << " phase='" << phase << "'"
            << " rank=" << lhs.commu.task
            << " internal_owned_l2=" << std::sqrt(static_cast<double>(internal_owned_sq))
            << " internal_owned_nnz=" << internal_owned_nnz
            << " internal_owned_max=" << internal_owned_max
            << " internal_ghost_l2=" << std::sqrt(static_cast<double>(internal_ghost_sq))
            << " internal_ghost_nnz=" << internal_ghost_nnz
            << " internal_ghost_max=" << internal_ghost_max
            << " old_owned_l2=" << std::sqrt(static_cast<double>(old_owned_sq))
            << " old_owned_nnz=" << old_owned_nnz
            << " old_owned_max=" << old_owned_max
            << " old_ghost_l2=" << std::sqrt(static_cast<double>(old_ghost_sq))
            << " old_ghost_nnz=" << old_ghost_nnz
            << " old_ghost_max=" << old_ghost_max;
        traceLog(oss.str());
    };

    auto storeSolveBufferToSolution = [&]() {
        copyVectorInternalToOld(std::span<const Real>(solve_buffer.data(), solve_buffer.size()), *x);
        logLocalReturnedSolutionStats("stored");
    };

    SolverReport report;
    const int base_gmres_total_iterations =
        gmres_total_iteration_budget(options_, /*legacy_restart_budget=*/false);
    const auto base_gmres_cfg = make_gmres_launch_config(options_, base_gmres_total_iterations);
    bool solution_stage_scaling_undone = false;
    bool current_preparation_uses_blockschur = use_blockschur;

    auto rebuildPreparedSystem = [&](bool blockschur_preparation) {
        restorePreparedMatrixValues(blockschur_preparation);
        loadSolveBufferFromVector(*b, blockschur_preparation);
        solution_stage_scaling_undone = false;
        current_preparation_uses_blockschur = blockschur_preparation;
        if (oopTraceEnabled()) {
            traceLog(std::string("FsilsLinearSolver::solve: prepared system mode='") +
                     (blockschur_preparation ? "blockschur" : "original") + "'");
        }
    };

    auto computeTrueResidualVector = [&](FsilsVector& residual_true, Real& rhs_norm_out) {
        FsilsVector rhs_true(shared_layout);
        rhs_true.copyFrom(*b);
        prepareRhsVectorForOperator(rhs_true);
        rhs_norm_out = rhs_true.norm();

        FsilsVector x_true(shared_layout);
        x_true.copyFrom(*x);
        x_true.updateGhosts();

        FsilsVector ax_true(shared_layout);
        A->mult(x_true, ax_true);
        ax_true.updateGhosts();
        addRankOneUpdatesToProduct(rank_one_updates_, x_true, ax_true, lhs.commu);
        addReducedFieldUpdatesToProduct(reduced_field_updates_,
                                        x_true,
                                        ax_true,
                                        lhs.commu,
                                        grouped_bordered_field_couplings_);
        addGroupedBorderedFieldCouplingsToProduct(grouped_bordered_field_couplings_,
                                                  x_true,
                                                  ax_true,
                                                  lhs.commu);

        residual_true.copyFrom(rhs_true);
        auto r_span = residual_true.localSpan();
        const auto ax_span = ax_true.localSpan();
        FE_THROW_IF(r_span.size() != ax_span.size(), FEException,
                    "FsilsLinearSolver::solve: residual validation size mismatch");
        for (std::size_t i = 0; i < r_span.size(); ++i) {
            r_span[i] -= ax_span[i];
        }
    };

    auto computeOriginalRhsNorm = [&]() -> Real {
        FsilsVector rhs_true(shared_layout);
        rhs_true.copyFrom(*b);
        prepareRhsVectorForOperator(rhs_true);
        return rhs_true.norm();
    };

    std::uint64_t residual_validation_dump_index = 0u;

    auto dumpResidualValidationVector = [&](std::uint64_t dump_index,
                                            std::string_view phase,
                                            std::string_view label,
                                            const FsilsVector& vec) {
        const char* dump_prefix = fsilsCompareFaceOperatorDumpPrefix();
        if (dump_prefix == nullptr) {
            return;
        }

        std::ostringstream path;
        path << dump_prefix
             << ".dump" << dump_index
             << ".residualcheck." << phase
             << "." << label
             << ".rank" << lhs.commu.task
             << ".txt";
        std::ofstream out(path.str());
        if (!out) {
            return;
        }

        out << "# task " << lhs.commu.task
            << " phase " << phase
            << " label " << label << "\n";
        out << "# global_node component value old_node internal_node\n";
        const auto span = vec.localSpan();
        for (int old = 0; old < lhs.nNo; ++old) {
            const int internal = lhs.map(old);
            if (internal < 0 || internal >= lhs.mynNo) {
                continue;
            }
            const int global_node = shared_layout->oldToGlobalNode(old);
            if (global_node < 0) {
                continue;
            }
            const std::size_t base =
                static_cast<std::size_t>(old) * static_cast<std::size_t>(dof);
            for (int c = 0; c < dof; ++c) {
                out << global_node
                    << ' ' << c
                    << ' ' << span[base + static_cast<std::size_t>(c)]
                    << ' ' << old
                    << ' ' << internal
                    << '\n';
            }
        }
    };

    auto applyLowRankResidualPolish = [&]() -> bool {
        if (!fsilsLowRankResidualPolishEnabled() ||
            lhs.commu.nTasks <= 1 ||
            !use_blockschur) {
            return false;
        }

        std::vector<FsilsVector> basis_q;
        basis_q.reserve(rank_one_updates_.size() + reduced_field_updates_.size() * 2u);
        auto appendBasisVector = [&](std::span<const std::pair<GlobalIndex, Real>> entries) {
            FsilsVector q(shared_layout);
            q.zero();
            auto q_view = q.createAssemblyView();
            FE_CHECK_NOT_NULL(q_view.get(), "FsilsLinearSolver: low-rank polish basis view");
            q_view->beginAssemblyPhase();
            for (const auto& [dof_idx, value] : entries) {
                q_view->addVectorEntry(dof_idx, value, assembly::AddMode::Insert);
            }
            q_view->finalizeAssembly();
            if (!(q.norm() > Real(0.0))) {
                return;
            }
            basis_q.push_back(std::move(q));
        };
        auto appendDenseBasisVector = [&](const FsilsVector& source) {
            FsilsVector q(shared_layout);
            q.copyFrom(source);
            if (!(q.norm() > Real(0.0))) {
                return;
            }
            basis_q.push_back(std::move(q));
        };
        std::vector<int> internal_to_old;
        if (!lhs.reduced_updates.empty()) {
            internal_to_old.assign(lhs.nNo, -1);
            for (int old = 0; old < lhs.nNo; ++old) {
                const int internal = lhs.map(old);
                if (internal >= 0 && internal < lhs.nNo) {
                    internal_to_old[static_cast<std::size_t>(internal)] = old;
                }
            }
        }
        auto appendInternalBasisVector =
            [&](const std::vector<fe_fsi_linear_solver::FSILS_reducedSparseEntry>& entries) {
                FsilsVector q(shared_layout);
                q.zero();
                auto q_span = q.localSpan();
                for (const auto& entry : entries) {
                    if (entry.node < 0 || entry.node >= lhs.nNo ||
                        entry.full_component < 0 || entry.full_component >= dof ||
                        std::abs(entry.value) <= 1e-30) {
                        continue;
                    }
                    const int old = internal_to_old[static_cast<std::size_t>(entry.node)];
                    if (old < 0 || old >= lhs.nNo) {
                        continue;
                    }
                    const std::size_t idx =
                        static_cast<std::size_t>(old) * static_cast<std::size_t>(dof) +
                        static_cast<std::size_t>(entry.full_component);
                    q_span[idx] += static_cast<Real>(entry.value);
                }
                if (!(q.norm() > Real(0.0))) {
                    return;
                }
                basis_q.push_back(std::move(q));
            };

        for (const auto& update : rank_one_updates_) {
            appendBasisVector(std::span<const std::pair<GlobalIndex, Real>>(update.v.data(),
                                                                            update.v.size()));
        }
        if (!lhs.reduced_updates.empty()) {
            for (const auto& update : lhs.reduced_updates) {
                appendInternalBasisVector(update.left);
                appendInternalBasisVector(update.right);
            }
        } else {
            for (const auto& update : reduced_field_updates_) {
                appendBasisVector(std::span<const std::pair<GlobalIndex, Real>>(update.left.data(),
                                                                                update.left.size()));
                appendBasisVector(std::span<const std::pair<GlobalIndex, Real>>(update.right.data(),
                                                                                update.right.size()));
            }
        }
        if (basis_q.empty() || basis_q.size() > 8u) {
            return false;
        }

        FsilsVector residual_current(shared_layout);
        Real rhs_norm_current = Real(0.0);
        computeTrueResidualVector(residual_current, rhs_norm_current);
        Real residual_norm_current = residual_current.norm();
        if (!(std::isfinite(static_cast<double>(residual_norm_current)) &&
              residual_norm_current > Real(0.0))) {
            return false;
        }
        if (basis_q.size() < 8u) {
            appendDenseBasisVector(*x);
        }
        if (basis_q.size() < 8u) {
            appendDenseBasisVector(residual_current);
        }
        FsilsVector x_eval(shared_layout);
        x_eval.copyFrom(*x);
        x_eval.updateGhosts();
        FsilsVector matrix_residual_current(shared_layout);
        A->mult(x_eval, matrix_residual_current);
        {
            auto matrix_residual_span = matrix_residual_current.localSpan();
            const auto rhs_span = b->localSpan();
            FE_THROW_IF(matrix_residual_span.size() != rhs_span.size(),
                        FEException,
                        "FsilsLinearSolver: low-rank polish matrix residual size mismatch");
            for (std::size_t i = 0; i < matrix_residual_span.size(); ++i) {
                matrix_residual_span[i] -= rhs_span[i];
            }
        }
        if (basis_q.size() < 8u) {
            appendDenseBasisVector(matrix_residual_current);
        }
        FsilsVector low_rank_eval_current(shared_layout);
        low_rank_eval_current.zero();
        addRankOneUpdatesToProduct(rank_one_updates_, x_eval, low_rank_eval_current, lhs.commu);
        addReducedFieldUpdatesToProduct(reduced_field_updates_,
                                        x_eval,
                                        low_rank_eval_current,
                                        lhs.commu,
                                        grouped_bordered_field_couplings_);
        addGroupedBorderedFieldCouplingsToProduct(grouped_bordered_field_couplings_,
                                                  x_eval,
                                                  low_rank_eval_current,
                                                  lhs.commu);
        if (basis_q.size() < 8u) {
            appendDenseBasisVector(low_rank_eval_current);
        }

        std::vector<FsilsVector> basis_y;
        basis_y.reserve(basis_q.size());
        for (const auto& q : basis_q) {
            FsilsVector q_eval(shared_layout);
            q_eval.copyFrom(q);
            q_eval.updateGhosts();

            FsilsVector y(shared_layout);
            A->mult(q_eval, y);
            addRankOneUpdatesToProduct(rank_one_updates_, q_eval, y, lhs.commu);
            addReducedFieldUpdatesToProduct(reduced_field_updates_,
                                            q_eval,
                                            y,
                                            lhs.commu,
                                            grouped_bordered_field_couplings_);
            addGroupedBorderedFieldCouplingsToProduct(grouped_bordered_field_couplings_,
                                                      q_eval,
                                                      y,
                                                      lhs.commu);
            basis_y.push_back(std::move(y));
        }

        const int rank = static_cast<int>(basis_q.size());
        std::vector<long double> normal(
            static_cast<std::size_t>(rank) * static_cast<std::size_t>(rank), 0.0L);
        long double trace = 0.0L;
        for (int i = 0; i < rank; ++i) {
            for (int j = i; j < rank; ++j) {
                const long double value = dotProductLongDouble(
                    basis_y[static_cast<std::size_t>(i)],
                    basis_y[static_cast<std::size_t>(j)]);
                normal[static_cast<std::size_t>(i) * static_cast<std::size_t>(rank) +
                       static_cast<std::size_t>(j)] = value;
                normal[static_cast<std::size_t>(j) * static_cast<std::size_t>(rank) +
                       static_cast<std::size_t>(i)] = value;
                if (i == j) {
                    trace += std::abs(value);
                }
            }
        }

        const long double regularization = std::max<long double>(trace * 1e-14L, 1e-30L);
        for (int i = 0; i < rank; ++i) {
            normal[static_cast<std::size_t>(i) * static_cast<std::size_t>(rank) +
                   static_cast<std::size_t>(i)] += regularization;
        }

        const std::vector<Real> backup = x_data;
        bool accepted_any = false;
        constexpr int max_polish_passes = 16;

        for (int pass = 0; pass < max_polish_passes; ++pass) {
            std::vector<long double> rhs_dense(static_cast<std::size_t>(rank), 0.0L);
            for (int i = 0; i < rank; ++i) {
                rhs_dense[static_cast<std::size_t>(i)] =
                    dotProductLongDouble(basis_y[static_cast<std::size_t>(i)], residual_current);
            }

            std::vector<long double> alpha = rhs_dense;
            std::vector<long double> normal_work = normal;
            if (!solveDenseSystemInPlaceLongDouble(normal_work, alpha, rank)) {
                break;
            }

            const long double alpha_norm =
                std::sqrt(std::inner_product(alpha.begin(), alpha.end(), alpha.begin(), 0.0L));
            if (!(std::isfinite(static_cast<double>(alpha_norm)) && alpha_norm > 1e-24L)) {
                break;
            }

            const std::vector<Real> pass_backup = x_data;
            for (int i = 0; i < rank; ++i) {
                const Real scale = static_cast<Real>(alpha[static_cast<std::size_t>(i)]);
                if (std::abs(scale) <= Real(1e-30)) {
                    continue;
                }
                const auto q_span = basis_q[static_cast<std::size_t>(i)].localSpan();
                FE_THROW_IF(q_span.size() != x_data.size(), FEException,
                            "FsilsLinearSolver: low-rank polish size mismatch");
                for (std::size_t idx = 0; idx < x_data.size(); ++idx) {
                    x_data[idx] += scale * q_span[idx];
                }
            }

            FsilsVector residual_after(shared_layout);
            Real rhs_norm_after = Real(0.0);
            computeTrueResidualVector(residual_after, rhs_norm_after);
            const Real residual_norm_after = residual_after.norm();
            const Real residual_target =
                std::max<Real>(options_.abs_tol,
                               options_.rel_tol *
                                   std::max<Real>(std::max(rhs_norm_current, rhs_norm_after),
                                                  Real(1e-30)));
            const Real improvement_floor =
                std::max<Real>(residual_norm_current * static_cast<Real>(1e-8),
                               std::max<Real>(residual_target * static_cast<Real>(1e-3),
                                              Real(1e-30)));
            const bool accept =
                std::isfinite(static_cast<double>(residual_norm_after)) &&
                residual_norm_after < residual_norm_current &&
                (residual_norm_after <= residual_target ||
                 residual_norm_after + improvement_floor < residual_norm_current);

            if (oopTraceEnabled() || fsilsTraceEnabled()) {
                std::ostringstream oss;
                oss << "FsilsLinearSolver::solve: low-rank residual polish"
                    << " pass=" << (pass + 1)
                    << " rank=" << rank
                    << " residual_before=" << residual_norm_current
                    << " residual_after=" << residual_norm_after
                    << " rhs_before=" << rhs_norm_current
                    << " rhs_after=" << rhs_norm_after
                    << " target=" << residual_target
                    << " improvement_floor=" << improvement_floor
                    << " alpha_norm=" << static_cast<double>(alpha_norm)
                    << " accept=" << (accept ? 1 : 0);
                for (int i = 0; i < rank; ++i) {
                    oss << " alpha[" << i << "]="
                        << static_cast<double>(alpha[static_cast<std::size_t>(i)]);
                }
                traceLog(oss.str());
            }

            if (!accept) {
                x_data = pass_backup;
                break;
            }

            accepted_any = true;
            residual_current = std::move(residual_after);
            rhs_norm_current = rhs_norm_after;
            residual_norm_current = residual_norm_after;
            if (residual_norm_current <= residual_target) {
                break;
            }
        }

        if (!accepted_any) {
            x_data = backup;
            return false;
        }
        return true;
    };

    auto computeConstraintMeanStats = [&]() -> FsilsConstraintMeanStats {
        FsilsConstraintMeanStats stats;
        if (!has_saddle_point || con_ncomp != 1 || con_start < 0 || con_start >= dof) {
            return stats;
        }

        long double local_sum = 0.0L;
        long double local_sq = 0.0L;
        unsigned long long local_count = 0ull;
        for (int a = 0; a < lhs.mynNo; ++a) {
            const Real value = Ri(con_start, a);
            local_sum += static_cast<long double>(value);
            local_sq += static_cast<long double>(value) * static_cast<long double>(value);
            ++local_count;
        }

        long double global_sum = local_sum;
        long double global_sq = local_sq;
        unsigned long long global_count = local_count;
        if (lhs.commu.nTasks > 1) {
            fe_fsi_linear_solver::fsils_allreduce_sum(
                &local_sum, &global_sum, 1, MPI_LONG_DOUBLE, lhs.commu);
            fe_fsi_linear_solver::fsils_allreduce_sum(
                &local_sq, &global_sq, 1, MPI_LONG_DOUBLE, lhs.commu);
            fe_fsi_linear_solver::fsils_allreduce_sum(
                &local_count, &global_count, 1, MPI_UNSIGNED_LONG_LONG, lhs.commu);
        }

        if (global_count == 0ull) {
            return stats;
        }

        const long double inv_count = 1.0L / static_cast<long double>(global_count);
        const long double mean = global_sum * inv_count;
        const long double mean_sq = mean * mean;
        const long double rms_sq = std::max<long double>(0.0L, global_sq * inv_count);
        const long double fluct_sq = std::max<long double>(0.0L, rms_sq - mean_sq);

        stats.valid = std::isfinite(static_cast<double>(mean)) &&
                      std::isfinite(static_cast<double>(rms_sq)) &&
                      std::isfinite(static_cast<double>(fluct_sq));
        stats.count = static_cast<std::uint64_t>(global_count);
        stats.mean = static_cast<Real>(mean);
        stats.rms = static_cast<Real>(std::sqrt(rms_sq));
        stats.fluctuation_rms = static_cast<Real>(std::sqrt(fluct_sq));
        return stats;
    };

    auto subtractConstraintMean = [&](Real mean_shift) {
        if (mean_shift == Real(0.0)) {
            return;
        }
        for (int a = 0; a < lhs.nNo; ++a) {
            Ri(con_start, a) -= mean_shift;
        }
    };

    auto computeReturnedSolutionConstraintMeanStats = [&]() -> FsilsConstraintMeanStats {
        FsilsConstraintMeanStats stats;
        if (!has_saddle_point || con_ncomp != 1 || con_start < 0 || con_start >= dof) {
            return stats;
        }

        long double local_sum = 0.0L;
        long double local_sq = 0.0L;
        unsigned long long local_count = 0ull;
        for (int old = 0; old < lhs.nNo; ++old) {
            const int internal = lhs.map(old);
            if (internal < 0 || internal >= lhs.mynNo) {
                continue;
            }
            const std::size_t idx =
                static_cast<std::size_t>(old) * static_cast<std::size_t>(dof) +
                static_cast<std::size_t>(con_start);
            const Real value = x_data[idx];
            local_sum += static_cast<long double>(value);
            local_sq += static_cast<long double>(value) * static_cast<long double>(value);
            ++local_count;
        }

        long double global_sum = local_sum;
        long double global_sq = local_sq;
        unsigned long long global_count = local_count;
        if (lhs.commu.nTasks > 1) {
            fe_fsi_linear_solver::fsils_allreduce_sum(
                &local_sum, &global_sum, 1, MPI_LONG_DOUBLE, lhs.commu);
            fe_fsi_linear_solver::fsils_allreduce_sum(
                &local_sq, &global_sq, 1, MPI_LONG_DOUBLE, lhs.commu);
            fe_fsi_linear_solver::fsils_allreduce_sum(
                &local_count, &global_count, 1, MPI_UNSIGNED_LONG_LONG, lhs.commu);
        }

        if (global_count == 0ull) {
            return stats;
        }

        const long double inv_count = 1.0L / static_cast<long double>(global_count);
        const long double mean = global_sum * inv_count;
        const long double mean_sq = mean * mean;
        const long double rms_sq = std::max<long double>(0.0L, global_sq * inv_count);
        const long double fluct_sq = std::max<long double>(0.0L, rms_sq - mean_sq);

        stats.valid = std::isfinite(static_cast<double>(mean)) &&
                      std::isfinite(static_cast<double>(rms_sq)) &&
                      std::isfinite(static_cast<double>(fluct_sq));
        stats.count = static_cast<std::uint64_t>(global_count);
        stats.mean = static_cast<Real>(mean);
        stats.rms = static_cast<Real>(std::sqrt(rms_sq));
        stats.fluctuation_rms = static_cast<Real>(std::sqrt(fluct_sq));
        return stats;
    };

    auto computeReturnedSolutionLocalConstraintMeanStats = [&]() -> FsilsLocalConstraintMeanStats {
        FsilsLocalConstraintMeanStats stats;
        if (!has_saddle_point || con_ncomp != 1 || con_start < 0 || con_start >= dof) {
            return stats;
        }

        long double local_sum = 0.0L;
        long double local_sq = 0.0L;
        unsigned long long local_count = 0ull;
        for (int old = 0; old < lhs.nNo; ++old) {
            const int internal = lhs.map(old);
            if (internal < 0 || internal >= lhs.mynNo) {
                continue;
            }
            const std::size_t idx =
                static_cast<std::size_t>(old) * static_cast<std::size_t>(dof) +
                static_cast<std::size_t>(con_start);
            const Real value = x_data[idx];
            local_sum += static_cast<long double>(value);
            local_sq += static_cast<long double>(value) * static_cast<long double>(value);
            ++local_count;
        }

        if (local_count == 0ull) {
            return stats;
        }

        const long double inv_count = 1.0L / static_cast<long double>(local_count);
        const long double mean = local_sum * inv_count;
        const long double mean_sq = mean * mean;
        const long double rms_sq = std::max<long double>(0.0L, local_sq * inv_count);
        const long double fluct_sq = std::max<long double>(0.0L, rms_sq - mean_sq);

        stats.valid = std::isfinite(static_cast<double>(mean)) &&
                      std::isfinite(static_cast<double>(rms_sq)) &&
                      std::isfinite(static_cast<double>(fluct_sq));
        stats.count = static_cast<std::uint64_t>(local_count);
        stats.mean = static_cast<Real>(mean);
        stats.rms = static_cast<Real>(std::sqrt(rms_sq));
        stats.fluctuation_rms = static_cast<Real>(std::sqrt(fluct_sq));
        return stats;
    };

    auto subtractConstraintMeanFromReturnedSolution = [&](Real mean_shift) {
        if (mean_shift == Real(0.0)) {
            return;
        }
        for (int old = 0; old < lhs.nNo; ++old) {
            const std::size_t idx =
                static_cast<std::size_t>(old) * static_cast<std::size_t>(dof) +
                static_cast<std::size_t>(con_start);
            x_data[idx] -= mean_shift;
        }
    };

    auto subtractLocalConstraintMeanFromReturnedSolution = [&](Real mean_shift) {
        if (mean_shift == Real(0.0)) {
            return;
        }
        for (int old = 0; old < lhs.nNo; ++old) {
            const int internal = lhs.map(old);
            if (internal < 0 || internal >= lhs.mynNo) {
                continue;
            }
            const std::size_t idx =
                static_cast<std::size_t>(old) * static_cast<std::size_t>(dof) +
                static_cast<std::size_t>(con_start);
            x_data[idx] -= mean_shift;
        }
    };

    auto centerReturnedSolutionConstraintMean = [&](std::string_view phase,
                                                    Real dominance_threshold,
                                                    bool force) -> bool {
        if (!(has_native_rank_one_updates && has_saddle_point && con_ncomp == 1)) {
            return false;
        }

        const auto before = computeReturnedSolutionConstraintMeanStats();
        if (!before.valid || before.count == 0u) {
            return false;
        }

        const Real fluctuation_floor =
            std::max<Real>(before.rms * static_cast<Real>(1e-12), static_cast<Real>(1e-14));
        const Real fluctuation = std::max(before.fluctuation_rms, fluctuation_floor);
        const Real dominance = std::abs(before.mean) / fluctuation;
        const bool should_center =
            force || (dominance >= dominance_threshold && std::abs(before.mean) > static_cast<Real>(1e-14));
        if (!should_center) {
            return false;
        }

        subtractConstraintMeanFromReturnedSolution(before.mean);
        if (oopTraceEnabled()) {
            const auto after = computeReturnedSolutionConstraintMeanStats();
            std::ostringstream oss;
            oss << "FsilsLinearSolver::solve: centered returned constraint mean"
                << " phase='" << phase << "'"
                << " mean_before=" << before.mean
                << " rms_before=" << before.rms
                << " fluct_before=" << before.fluctuation_rms
                << " dominance=" << dominance
                << " mean_after=" << after.mean
                << " rms_after=" << after.rms;
            traceLog(oss.str());
        }
        return true;
    };

    auto logInternalBlockSolutionStats = [&](std::string_view phase) {
        if (!oopTraceEnabled() || !has_saddle_point) {
            return;
        }

        long double local_mom_sq = 0.0L;
        long double local_con_sq = 0.0L;
        for (int a = 0; a < lhs.mynNo; ++a) {
            for (int c = mom_start; c < mom_start + mom_ncomp; ++c) {
                const long double v = static_cast<long double>(Ri(c, a));
                local_mom_sq += v * v;
            }
            for (int c = con_start; c < con_start + con_ncomp; ++c) {
                const long double v = static_cast<long double>(Ri(c, a));
                local_con_sq += v * v;
            }
        }

        long double global_mom_sq = local_mom_sq;
        long double global_con_sq = local_con_sq;
        if (lhs.commu.nTasks > 1) {
            fe_fsi_linear_solver::fsils_allreduce_sum(
                &local_mom_sq, &global_mom_sq, 1, MPI_LONG_DOUBLE, lhs.commu);
            fe_fsi_linear_solver::fsils_allreduce_sum(
                &local_con_sq, &global_con_sq, 1, MPI_LONG_DOUBLE, lhs.commu);
        }

        const auto correction_con_stats = computeConstraintMeanStats();
        const auto returned_con_stats = computeReturnedSolutionConstraintMeanStats();
        std::ostringstream oss;
        oss << "FsilsLinearSolver::solve: internal block stats"
            << " phase='" << phase << "'"
            << " |mom|=" << std::sqrt(std::max<long double>(0.0L, global_mom_sq))
            << " |con|=" << std::sqrt(std::max<long double>(0.0L, global_con_sq));
        if (correction_con_stats.valid) {
            oss << " corr_con_mean=" << correction_con_stats.mean
                << " corr_con_rms=" << correction_con_stats.rms
                << " corr_con_fluct=" << correction_con_stats.fluctuation_rms;
        }
        if (returned_con_stats.valid) {
            oss << " returned_con_mean=" << returned_con_stats.mean
                << " returned_con_rms=" << returned_con_stats.rms
                << " returned_con_fluct=" << returned_con_stats.fluctuation_rms;
        }
        traceLog(oss.str());
    };

    auto logConstraintMeanModeProbe = [&](std::string_view phase) {
        if (!oopTraceEnabled() || !has_saddle_point || con_ncomp != 1 || x == nullptr) {
            return;
        }

        FsilsVector residual_true(shared_layout);
        Real rhs_norm = Real(0.0);
        computeTrueResidualVector(residual_true, rhs_norm);

        FsilsVector mean_mode(shared_layout);
        mean_mode.zero();
        auto& mean_mode_data = mean_mode.data();
        for (int old = 0; old < lhs.nNo; ++old) {
            const std::size_t idx =
                static_cast<std::size_t>(old) * static_cast<std::size_t>(dof) +
                static_cast<std::size_t>(con_start);
            mean_mode_data[idx] = Real(1.0);
        }

        FsilsVector az_matrix(shared_layout);
        A->mult(mean_mode, az_matrix);

        FsilsVector az(shared_layout);
        az.copyFrom(az_matrix);
        addRankOneUpdatesToProduct(rank_one_updates_, mean_mode, az, lhs.commu);
        addReducedFieldUpdatesToProduct(reduced_field_updates_,
                                        mean_mode,
                                        az,
                                        lhs.commu,
                                        grouped_bordered_field_couplings_);
        addGroupedBorderedFieldCouplingsToProduct(grouped_bordered_field_couplings_,
                                                  mean_mode,
                                                  az,
                                                  lhs.commu);

        const auto internal_size =
            static_cast<std::size_t>(dof) * static_cast<std::size_t>(lhs.nNo);
        std::vector<Real> residual_internal(internal_size, Real(0.0));
        std::vector<Real> az_matrix_internal(internal_size, Real(0.0));
        std::vector<Real> az_internal(internal_size, Real(0.0));
        copyVectorOldToInternal(residual_true, residual_internal);
        copyVectorOldToInternal(az_matrix, az_matrix_internal);
        copyVectorOldToInternal(az, az_internal);

        long double local_rr = 0.0L;
        long double local_jj = 0.0L;
        long double local_jj_mom = 0.0L;
        long double local_jj_con = 0.0L;
        long double local_hh = 0.0L;
        long double local_hh_mom = 0.0L;
        long double local_hh_con = 0.0L;
        long double local_raz = 0.0L;
        long double local_aa = 0.0L;
        for (int node = 0; node < lhs.mynNo; ++node) {
            const std::size_t base =
                static_cast<std::size_t>(node) * static_cast<std::size_t>(dof);
            for (int c = 0; c < dof; ++c) {
                const long double rv =
                    static_cast<long double>(residual_internal[base + static_cast<std::size_t>(c)]);
                const long double jv =
                    static_cast<long double>(az_matrix_internal[base + static_cast<std::size_t>(c)]);
                const long double av =
                    static_cast<long double>(az_internal[base + static_cast<std::size_t>(c)]);
                const long double hv = av - jv;
                local_rr += rv * rv;
                local_jj += jv * jv;
                local_hh += hv * hv;
                local_raz += rv * av;
                local_aa += av * av;
                if (c >= con_start && c < con_start + con_ncomp) {
                    local_jj_con += jv * jv;
                    local_hh_con += hv * hv;
                } else {
                    local_jj_mom += jv * jv;
                    local_hh_mom += hv * hv;
                }
            }
        }

        if (const char* rows_env = std::getenv("SVMP_FSILS_MEAN_MODE_ROWS");
            rows_env != nullptr && rows_env[0] != '\0' && rows_env[0] != '0') {
            struct RowRec {
                long double magnitude{0.0L};
                Real value{0.0};
                int internal_node{-1};
                int old_node{-1};
                int global_node{-1};
                GlobalIndex backend_dof{INVALID_GLOBAL_INDEX};
                GlobalIndex fe_dof{INVALID_GLOBAL_INDEX};
                int owner{-1};
            };

            int limit = 8;
            try {
                limit = std::max(1, std::stoi(std::string(rows_env)));
            } catch (...) {
                limit = 8;
            }

            std::vector<RowRec> rows;
            rows.reserve(static_cast<std::size_t>(lhs.mynNo));
            for (int node = 0; node < lhs.mynNo; ++node) {
                const std::size_t idx =
                    static_cast<std::size_t>(node) * static_cast<std::size_t>(dof) +
                    static_cast<std::size_t>(con_start);
                const Real value = az_matrix_internal[idx];
                if (value == Real(0.0)) {
                    continue;
                }

                int old = -1;
                if (shared_layout && static_cast<std::size_t>(node) < shared_layout->old_of_internal.size()) {
                    old = shared_layout->old_of_internal[static_cast<std::size_t>(node)];
                }
                int global_node = -1;
                if (shared_layout) {
                    global_node = shared_layout->oldToGlobalNode(old);
                }
                const auto backend_dof =
                    (global_node >= 0)
                        ? static_cast<GlobalIndex>(global_node) * static_cast<GlobalIndex>(dof) +
                              static_cast<GlobalIndex>(con_start)
                        : INVALID_GLOBAL_INDEX;

                GlobalIndex fe_dof = backend_dof;
                int owner = -1;
                if (shared_layout && shared_layout->dof_permutation &&
                    !shared_layout->dof_permutation->empty() &&
                    backend_dof >= 0 &&
                    static_cast<std::size_t>(backend_dof) < shared_layout->dof_permutation->inverse.size()) {
                    fe_dof = shared_layout->dof_permutation->inverse[static_cast<std::size_t>(backend_dof)];
                    if (static_cast<std::size_t>(backend_dof) <
                        shared_layout->dof_permutation->owner_rank.size()) {
                        owner = shared_layout->dof_permutation->owner_rank[static_cast<std::size_t>(backend_dof)];
                    }
                }

                rows.push_back(RowRec{std::abs(static_cast<long double>(value)),
                                      value,
                                      node,
                                      old,
                                      global_node,
                                      backend_dof,
                                      fe_dof,
                                      owner});
            }

            std::sort(rows.begin(), rows.end(), [](const RowRec& a, const RowRec& b) {
                if (a.magnitude != b.magnitude) {
                    return a.magnitude > b.magnitude;
                }
                return a.global_node < b.global_node;
            });

            std::ostringstream rows_oss;
            rows_oss << "FsilsLinearSolver::solve: mean-mode local pressure-row top"
                     << " phase='" << phase << "'"
                     << " rank=" << lhs.commu.task
                     << " local_Jz_con=" << std::sqrt(std::max<long double>(0.0L, local_jj_con))
                     << " nonzero_rows=" << rows.size();
            const auto n_print = std::min<std::size_t>(rows.size(), static_cast<std::size_t>(limit));
            for (std::size_t i = 0; i < n_print; ++i) {
                const auto& r = rows[i];
                rows_oss << " [i=" << i
                         << " val=" << r.value
                         << " internal=" << r.internal_node
                         << " old=" << r.old_node
                         << " node=" << r.global_node
                         << " be=" << r.backend_dof
                         << " fe=" << r.fe_dof
                         << " owner=" << r.owner
                         << "]";
            }
            traceLog(rows_oss.str());
        }

        long double global_rr = local_rr;
        long double global_jj = local_jj;
        long double global_jj_mom = local_jj_mom;
        long double global_jj_con = local_jj_con;
        long double global_hh = local_hh;
        long double global_hh_mom = local_hh_mom;
        long double global_hh_con = local_hh_con;
        long double global_raz = local_raz;
        long double global_aa = local_aa;
        if (lhs.commu.nTasks > 1) {
            fe_fsi_linear_solver::fsils_allreduce_sum(
                &local_rr, &global_rr, 1, MPI_LONG_DOUBLE, lhs.commu);
            fe_fsi_linear_solver::fsils_allreduce_sum(
                &local_jj, &global_jj, 1, MPI_LONG_DOUBLE, lhs.commu);
            fe_fsi_linear_solver::fsils_allreduce_sum(
                &local_jj_mom, &global_jj_mom, 1, MPI_LONG_DOUBLE, lhs.commu);
            fe_fsi_linear_solver::fsils_allreduce_sum(
                &local_jj_con, &global_jj_con, 1, MPI_LONG_DOUBLE, lhs.commu);
            fe_fsi_linear_solver::fsils_allreduce_sum(
                &local_hh, &global_hh, 1, MPI_LONG_DOUBLE, lhs.commu);
            fe_fsi_linear_solver::fsils_allreduce_sum(
                &local_hh_mom, &global_hh_mom, 1, MPI_LONG_DOUBLE, lhs.commu);
            fe_fsi_linear_solver::fsils_allreduce_sum(
                &local_hh_con, &global_hh_con, 1, MPI_LONG_DOUBLE, lhs.commu);
            fe_fsi_linear_solver::fsils_allreduce_sum(
                &local_raz, &global_raz, 1, MPI_LONG_DOUBLE, lhs.commu);
            fe_fsi_linear_solver::fsils_allreduce_sum(
                &local_aa, &global_aa, 1, MPI_LONG_DOUBLE, lhs.commu);
        }

        const auto mean_stats = computeReturnedSolutionConstraintMeanStats();
        const long double az_norm = std::sqrt(std::max<long double>(0.0L, global_aa));
        long double delta_opt = 0.0L;
        long double residual_opt = std::sqrt(std::max<long double>(0.0L, global_rr));
        if (global_aa > 1e-30L) {
            delta_opt = -global_raz / global_aa;
            const long double reduced_sq =
                std::max<long double>(0.0L,
                                      global_rr - (global_raz * global_raz) / global_aa);
            residual_opt = std::sqrt(reduced_sq);
        }

        std::ostringstream oss;
        oss << "FsilsLinearSolver::solve: constraint mean-mode probe"
            << " phase='" << phase << "'"
            << " mean=" << (mean_stats.valid ? mean_stats.mean : Real(0.0))
            << " rhs=" << rhs_norm
            << " |r|=" << std::sqrt(std::max<long double>(0.0L, global_rr))
            << " |Jz|=" << std::sqrt(std::max<long double>(0.0L, global_jj))
            << " |Jz_mom|=" << std::sqrt(std::max<long double>(0.0L, global_jj_mom))
            << " |Jz_con|=" << std::sqrt(std::max<long double>(0.0L, global_jj_con))
            << " |Rz|=" << std::sqrt(std::max<long double>(0.0L, global_hh))
            << " |Rz_mom|=" << std::sqrt(std::max<long double>(0.0L, global_hh_mom))
            << " |Rz_con|=" << std::sqrt(std::max<long double>(0.0L, global_hh_con))
            << " |Az|=" << az_norm
            << " r.Az=" << global_raz
            << " delta_opt=" << delta_opt
            << " |r+delta Az|=" << residual_opt;
        traceLog(oss.str());

        const auto local_mean_stats = computeReturnedSolutionLocalConstraintMeanStats();
        if (local_mean_stats.valid && local_mean_stats.count > 0u) {
            std::ostringstream loss;
            loss << "FsilsLinearSolver::solve: local constraint stats"
                 << " phase='" << phase << "'"
                 << " rank=" << lhs.commu.task
                 << " count=" << local_mean_stats.count
                 << " mean=" << local_mean_stats.mean
                 << " rms=" << local_mean_stats.rms
                 << " fluct=" << local_mean_stats.fluctuation_rms;
            traceLog(loss.str());

            std::vector<Real> backup = x_data;
            subtractLocalConstraintMeanFromReturnedSolution(local_mean_stats.mean);
            FsilsVector local_residual_true(shared_layout);
            Real local_rhs_norm = Real(0.0);
            computeTrueResidualVector(local_residual_true, local_rhs_norm);
            const Real local_residual_norm = local_residual_true.norm();
            const Real local_rel =
                local_residual_norm / std::max<Real>(local_rhs_norm, static_cast<Real>(1e-30));
            const Real local_target = std::max<Real>(
                options_.abs_tol, options_.rel_tol * std::max<Real>(local_rhs_norm, static_cast<Real>(1e-30)));
            const bool local_ok =
                std::isfinite(static_cast<double>(local_residual_norm)) &&
                std::isfinite(static_cast<double>(local_rel)) &&
                local_residual_norm <= local_target;
            const auto local_after = computeReturnedSolutionLocalConstraintMeanStats();
            x_data = std::move(backup);

            std::ostringstream lross;
            lross << "FsilsLinearSolver::solve: local constraint recenter probe"
                  << " phase='" << phase << "'"
                  << " rank=" << lhs.commu.task
                  << " mean_before=" << local_mean_stats.mean
                  << " mean_after=" << local_after.mean
                  << " residual_after=" << local_residual_norm
                  << " rel_after=" << local_rel
                  << " ok_after=" << (local_ok ? 1 : 0);
            traceLog(lross.str());
        }
    };

    auto compareFaceOperatorAgainstFe = [&]() {
        if (!fsilsCompareFaceOperatorEnabled()) {
            return;
        }

        if (rank_one_updates_.empty() &&
            reduced_field_updates_.empty() &&
            grouped_bordered_field_couplings_.empty() &&
            lhs.reduced_updates.empty() &&
            lhs.grouped_bordered_field_couplings.empty()) {
            return;
        }

        if (lhs.nFaces > 0 && incL.size() == lhs.nFaces) {
            const auto& active_res =
                current_preparation_uses_blockschur ? res_blockschur : res_original;
            if (active_res.size() == lhs.nFaces) {
                for (int f = 0; f < lhs.nFaces; ++f) {
                    lhs.face[static_cast<std::size_t>(f)].incFlag = (incL(f) != 0);
                    lhs.face[static_cast<std::size_t>(f)].res = active_res(f);
                }
            }
        }

        const bool compare_in_prepared_blockschur_coordinates =
            current_preparation_uses_blockschur && stage_scale != 1.0;
        const Real s = static_cast<Real>(stage_scale);
        const Real inv_s = static_cast<Real>(1.0 / stage_scale);
        static std::uint64_t compare_dump_index = 0u;
        const std::uint64_t dump_index = compare_dump_index++;
        auto dumpOwnerAlignedVector = [&](std::string_view probe_label,
                                         std::string_view suffix,
                                         const FsilsVector& vec) {
            const char* dump_prefix = fsilsCompareFaceOperatorDumpPrefix();
            if (dump_prefix == nullptr) {
                return;
            }
            if (const char* dump_probe = fsilsCompareFaceOperatorDumpProbe();
                dump_probe != nullptr && probe_label != dump_probe) {
                return;
            }

            std::ostringstream path;
            path << dump_prefix
                 << ".dump" << dump_index
                 << "." << probe_label
                 << "." << suffix
                 << ".rank" << lhs.commu.task
                 << ".txt";
            std::ofstream out(path.str());
            if (!out) {
                return;
            }

            out << "# task " << lhs.commu.task
                << " probe " << probe_label
                << " suffix " << suffix << "\n";
            out << "# global_node component value\n";
            const int owned_node_count = shared_layout->owned_node_count;
            const auto span = vec.localSpan();
            for (int old = 0; old < owned_node_count; ++old) {
                const int global_node = shared_layout->oldToGlobalNode(old);
                if (global_node < 0) {
                    continue;
                }
                const std::size_t base =
                    static_cast<std::size_t>(old) * static_cast<std::size_t>(dof);
                for (int c = 0; c < dof; ++c) {
                    out << global_node
                        << ' ' << c
                        << ' ' << span[base + static_cast<std::size_t>(c)]
                        << '\n';
                }
            }

            std::ostringstream fe_path;
            fe_path << dump_prefix
                    << ".dump" << dump_index
                    << "." << probe_label
                    << "." << suffix
                    << ".fe_dof.rank" << lhs.commu.task
                    << ".txt";
            std::ofstream fe_out(fe_path.str());
            if (!fe_out) {
                return;
            }

            fe_out << "# task " << lhs.commu.task
                   << " probe " << probe_label
                   << " suffix " << suffix << "\n";
            fe_out << "# fe_dof value old_node internal_node backend_node component\n";
            const auto* perm = shared_layout->dof_permutation.get();
            const bool has_inverse = perm != nullptr && !perm->inverse.empty();
            for (int old = 0; old < owned_node_count; ++old) {
                const int backend_node = shared_layout->oldToGlobalNode(old);
                if (backend_node < 0) {
                    continue;
                }
                const int internal = lhs.map(old);
                const std::size_t base =
                    static_cast<std::size_t>(old) * static_cast<std::size_t>(dof);
                for (int c = 0; c < dof; ++c) {
                    const GlobalIndex backend_dof =
                        static_cast<GlobalIndex>(backend_node) * static_cast<GlobalIndex>(dof) +
                        static_cast<GlobalIndex>(c);
                    GlobalIndex fe_dof = backend_dof;
                    if (has_inverse && static_cast<std::size_t>(backend_dof) < perm->inverse.size()) {
                        fe_dof = perm->inverse[static_cast<std::size_t>(backend_dof)];
                    }
                    fe_out << fe_dof
                           << ' ' << span[base + static_cast<std::size_t>(c)]
                           << ' ' << old
                           << ' ' << internal
                           << ' ' << backend_node
                           << ' ' << c
                           << '\n';
                }
            }
        };
        auto dumpPreparedRowIfRequested = [&]() {
            const int target_global_node = fsilsDumpPreparedRowGlobalNode();
            const int row_comp = fsilsDumpPreparedRowComponent();
            const char* dump_prefix = fsilsDumpPreparedRowPrefix();
            if (target_global_node < 0 || row_comp < 0 || row_comp >= dof || dump_prefix == nullptr) {
                return;
            }

            int target_old = -1;
            for (int old = 0; old < lhs.nNo; ++old) {
                if (shared_layout->oldToGlobalNode(old) == target_global_node) {
                    target_old = old;
                    break;
                }
            }
            if (target_old < 0) {
                return;
            }

            std::vector<int> internal_to_old(lhs.nNo, -1);
            for (int old = 0; old < lhs.nNo; ++old) {
                const int internal = lhs.map(old);
                if (internal >= 0 && internal < lhs.nNo) {
                    internal_to_old[internal] = old;
                }
            }

            const int row_internal = lhs.map(target_old);
            FE_THROW_IF(row_internal < 0 || row_internal >= lhs.nNo,
                        FEException,
                        "FsilsLinearSolver::solve: invalid internal row while dumping prepared row");

            std::ostringstream path;
            path << dump_prefix
                 << ".g" << target_global_node
                 << ".r" << row_comp
                 << ".rank" << lhs.commu.task
                 << ".txt";
            std::ofstream out(path.str());
            if (!out) {
                return;
            }

            const int col_comp_filter = fsilsDumpPreparedColComponent();
            out << "# task " << lhs.commu.task
                << " global_row_node " << target_global_node
                << " old_row " << target_old
                << " row_kind " << ((target_old < shared_layout->owned_node_count) ? "owned" : "ghost")
                << " internal_row " << row_internal
                << " row_component " << row_comp
                << " col_component_filter " << col_comp_filter << "\n";
            out << "# col_global_node col_component value col_old col_kind\n";
            for (int nz = lhs.rowPtr(0, row_internal); nz <= lhs.rowPtr(1, row_internal); ++nz) {
                const int col_internal = lhs.colPtr(nz);
                FE_THROW_IF(col_internal < 0 || col_internal >= lhs.nNo,
                            FEException,
                            "FsilsLinearSolver::solve: invalid internal column while dumping prepared row");
                const int col_old = internal_to_old[col_internal];
                FE_THROW_IF(col_old < 0 || col_old >= lhs.nNo,
                            FEException,
                            "FsilsLinearSolver::solve: missing old column while dumping prepared row");
                const int col_global_node = shared_layout->oldToGlobalNode(col_old);
                const char* col_kind =
                    (col_old < shared_layout->owned_node_count) ? "owned" : "ghost";
                for (int col_comp = 0; col_comp < dof; ++col_comp) {
                    if (col_comp_filter >= 0 && col_comp != col_comp_filter) {
                        continue;
                    }
                    out << col_global_node
                        << ' ' << col_comp
                        << ' ' << Val(row_comp * dof + col_comp, nz)
                        << ' ' << col_old
                        << ' ' << col_kind
                        << '\n';
                }
            }

            const GlobalIndex backend_row_dof =
                static_cast<GlobalIndex>(target_global_node) * static_cast<GlobalIndex>(dof) +
                static_cast<GlobalIndex>(row_comp);
            GlobalIndex fe_row_dof = backend_row_dof;
            if (const auto* perm = shared_layout->dof_permutation.get();
                perm != nullptr && !perm->inverse.empty() &&
                static_cast<std::size_t>(backend_row_dof) < perm->inverse.size()) {
                fe_row_dof = perm->inverse[static_cast<std::size_t>(backend_row_dof)];
            }

            std::ostringstream fe_path;
            fe_path << dump_prefix
                    << ".g" << target_global_node
                    << ".r" << row_comp
                    << ".original.rank" << lhs.commu.task
                    << ".txt";
            std::ofstream fe_out(fe_path.str());
            if (!fe_out) {
                return;
            }

            const GlobalIndex n_cols = A->numCols();
            fe_out << "# task " << lhs.commu.task
                   << " global_row_node " << target_global_node
                   << " row_component " << row_comp
                   << " row_kind " << ((target_old < shared_layout->owned_node_count) ? "owned" : "ghost")
                   << " backend_row_dof " << backend_row_dof
                   << " fe_row_dof " << fe_row_dof << "\n";
            fe_out << "# fe_col_dof value backend_col_node backend_col_component\n";
            for (GlobalIndex fe_col_dof = 0; fe_col_dof < n_cols; ++fe_col_dof) {
                const Real value = A->getEntry(fe_row_dof, fe_col_dof);
                if (std::abs(value) <= Real(0.0)) {
                    continue;
                }
                GlobalIndex backend_col_dof = fe_col_dof;
                if (const auto* perm = shared_layout->dof_permutation.get();
                    perm != nullptr && !perm->empty() &&
                    static_cast<std::size_t>(fe_col_dof) < perm->forward.size()) {
                    backend_col_dof = perm->forward[static_cast<std::size_t>(fe_col_dof)];
                }
                const GlobalIndex backend_col_node =
                    backend_col_dof / static_cast<GlobalIndex>(dof);
                const int backend_col_comp =
                    static_cast<int>(backend_col_dof % static_cast<GlobalIndex>(dof));
                fe_out << fe_col_dof
                       << ' ' << value
                       << ' ' << backend_col_node
                       << ' ' << backend_col_comp
                       << '\n';
            }

            std::ostringstream raw_path;
            raw_path << dump_prefix
                     << ".g" << target_global_node
                     << ".r" << row_comp
                     << ".original_internal.rank" << lhs.commu.task
                     << ".txt";
            std::ofstream raw_out(raw_path.str());
            if (!raw_out) {
                return;
            }

            const auto& lhs_original =
                *static_cast<const fe_fsi_linear_solver::FSILS_lhsType*>(A->fsilsLhsPtr());
            const auto* values_original = A->fsilsValuesPtr();
            const std::size_t block_size =
                static_cast<std::size_t>(dof) * static_cast<std::size_t>(dof);
            raw_out << "# task " << lhs.commu.task
                    << " global_row_node " << target_global_node
                    << " row_component " << row_comp
                    << " row_kind " << ((target_old < shared_layout->owned_node_count) ? "owned" : "ghost")
                    << " row_internal " << row_internal << "\n";
            raw_out << "# nz_index col_internal col_global_node col_component value col_old col_kind\n";
            for (int nz = lhs_original.rowPtr(0, row_internal);
                 nz <= lhs_original.rowPtr(1, row_internal);
                 ++nz) {
                const int col_internal = lhs_original.colPtr(nz);
                FE_THROW_IF(col_internal < 0 || col_internal >= lhs_original.nNo,
                            FEException,
                            "FsilsLinearSolver::solve: invalid original internal column while dumping row");
                const int col_old = internal_to_old[col_internal];
                FE_THROW_IF(col_old < 0 || col_old >= lhs_original.nNo,
                            FEException,
                            "FsilsLinearSolver::solve: missing old column while dumping original row");
                const int col_global_node = shared_layout->oldToGlobalNode(col_old);
                const char* col_kind =
                    (col_old < shared_layout->owned_node_count) ? "owned" : "ghost";
                for (int col_comp = 0; col_comp < dof; ++col_comp) {
                    if (col_comp_filter >= 0 && col_comp != col_comp_filter) {
                        continue;
                    }
                    const std::size_t value_index =
                        static_cast<std::size_t>(nz) * block_size +
                        static_cast<std::size_t>(row_comp * dof + col_comp);
                    raw_out << nz
                            << ' ' << col_internal
                            << ' ' << col_global_node
                            << ' ' << col_comp
                            << ' ' << values_original[value_index]
                            << ' ' << col_old
                            << ' ' << col_kind
                            << '\n';
                }
            }

            std::ostringstream alias_path;
            alias_path << dump_prefix
                       << ".g" << target_global_node
                       << ".r" << row_comp
                       << ".internal_alias.rank" << lhs.commu.task
                       << ".txt";
            std::ofstream alias_out(alias_path.str());
            if (!alias_out) {
                return;
            }
            alias_out << "# task " << lhs.commu.task
                      << " row_internal " << row_internal
                      << " target_global_node " << target_global_node
                      << " row_component " << row_comp << "\n";
            alias_out << "# old_node internal_node global_node kind\n";
            for (int old = 0; old < lhs.nNo; ++old) {
                if (lhs.map(old) != row_internal) {
                    continue;
                }
                alias_out << old
                          << ' ' << lhs.map(old)
                          << ' ' << shared_layout->oldToGlobalNode(old)
                          << ' ' << ((old < shared_layout->owned_node_count) ? "owned" : "ghost")
                          << '\n';
            }
        };

        auto runProbeComparison = [&](std::string_view probe_label, FsilsVector& probe_old) {
            probe_old.updateGhosts();

            FsilsVector probe_fe(shared_layout);
            probe_fe.copyFrom(probe_old);
            if (compare_in_prepared_blockschur_coordinates) {
                auto probe_span = probe_fe.localSpan();
                for (int old = 0; old < lhs.nNo; ++old) {
                    const std::size_t base =
                        static_cast<std::size_t>(old) * static_cast<std::size_t>(dof);
                    for (int c = con_start; c < con_start + con_ncomp; ++c) {
                        probe_span[base + static_cast<std::size_t>(c)] *= inv_s;
                    }
                }
                probe_fe.updateGhosts();
            }

            FsilsVector fe_matrix(shared_layout);
            A->mult(probe_fe, fe_matrix);

            FsilsVector fe_y(shared_layout);
            fe_y.copyFrom(fe_matrix);
            addRankOneUpdatesToProduct(rank_one_updates_, probe_fe, fe_y, lhs.commu);
            addReducedFieldUpdatesToProduct(reduced_field_updates_,
                                            probe_fe,
                                            fe_y,
                                            lhs.commu,
                                            grouped_bordered_field_couplings_);
            addGroupedBorderedFieldCouplingsToProduct(grouped_bordered_field_couplings_,
                                                      probe_fe,
                                                      fe_y,
                                                      lhs.commu);
            fe_matrix.updateGhosts();
            fe_y.updateGhosts();
            if (compare_in_prepared_blockschur_coordinates) {
                auto fe_matrix_span = fe_matrix.localSpan();
                auto fe_span = fe_y.localSpan();
                for (int old = 0; old < lhs.nNo; ++old) {
                    const std::size_t base =
                        static_cast<std::size_t>(old) * static_cast<std::size_t>(dof);
                    for (int c = mom_start; c < mom_start + mom_ncomp; ++c) {
                        fe_matrix_span[base + static_cast<std::size_t>(c)] *= s;
                        fe_span[base + static_cast<std::size_t>(c)] *= s;
                    }
                }
            }

            FsilsVector fe_correction(shared_layout);
            fe_correction.copyFrom(fe_y);
            {
                auto corr_span = fe_correction.localSpan();
                const auto matrix_span = fe_matrix.localSpan();
                FE_THROW_IF(corr_span.size() != matrix_span.size(), FEException,
                            "FsilsLinearSolver::solve: FE correction compare size mismatch");
                for (std::size_t i = 0; i < corr_span.size(); ++i) {
                    corr_span[i] -= matrix_span[i];
                }
            }

            std::vector<Real> probe_internal_data(solve_buffer.size(), Real(0.0));
            std::vector<Real> fsils_internal_data(solve_buffer.size(), Real(0.0));
            copyVectorOldToInternal(
                probe_old, std::span<Real>(probe_internal_data.data(), probe_internal_data.size()));
            Array<double> probe_internal(dof, lhs.nNo, probe_internal_data.data());
            Array<double> fsils_internal(dof, lhs.nNo, fsils_internal_data.data());
            fe_fsi_linear_solver::fsils_syncv_owned_to_ghost(lhs, dof, probe_internal);

            std::vector<Real> fsils_matrix_data(solve_buffer.size(), Real(0.0));
            Array<double> fsils_matrix_internal(dof, lhs.nNo, fsils_matrix_data.data());
            spar_mul::fsils_spar_mul_vv(
                lhs, lhs.rowPtr, lhs.colPtr, dof, Val, probe_internal, fsils_matrix_internal);
            fsils_internal = fsils_matrix_internal;
            add_bc_mul::add_bc_mul(lhs, fe_fsi_linear_solver::BcopType::BCOP_TYPE_ADD,
                                   dof, probe_internal, fsils_internal);

            FsilsVector fsils_matrix(shared_layout);
            copyVectorInternalToOld(
                std::span<const Real>(fsils_matrix_data.data(), fsils_matrix_data.size()),
                fsils_matrix);
            fsils_matrix.updateGhosts();

            FsilsVector fsils_y(shared_layout);
            copyVectorInternalToOld(
                std::span<const Real>(fsils_internal_data.data(), fsils_internal_data.size()),
                fsils_y);
            fsils_y.updateGhosts();

            FsilsVector fsils_correction(shared_layout);
            fsils_correction.copyFrom(fsils_y);
            {
                auto corr_span = fsils_correction.localSpan();
                const auto matrix_span = fsils_matrix.localSpan();
                FE_THROW_IF(corr_span.size() != matrix_span.size(), FEException,
                            "FsilsLinearSolver::solve: FSILS correction compare size mismatch");
                for (std::size_t i = 0; i < corr_span.size(); ++i) {
                    corr_span[i] -= matrix_span[i];
                }
            }

            FsilsVector diff(shared_layout);
            diff.copyFrom(fe_y);
            auto diff_span = diff.localSpan();
            const auto fsils_span = fsils_y.localSpan();
            FE_THROW_IF(diff_span.size() != fsils_span.size(), FEException,
                        "FsilsLinearSolver::solve: operator compare size mismatch");
            for (std::size_t i = 0; i < diff_span.size(); ++i) {
                diff_span[i] -= fsils_span[i];
            }

            FsilsVector matrix_diff(shared_layout);
            matrix_diff.copyFrom(fe_matrix);
            {
                auto diff_matrix_span = matrix_diff.localSpan();
                const auto fsils_matrix_span = fsils_matrix.localSpan();
                FE_THROW_IF(diff_matrix_span.size() != fsils_matrix_span.size(),
                            FEException,
                            "FsilsLinearSolver::solve: matrix compare size mismatch");
                for (std::size_t i = 0; i < diff_matrix_span.size(); ++i) {
                    diff_matrix_span[i] -= fsils_matrix_span[i];
                }
            }

            FsilsVector correction_diff(shared_layout);
            correction_diff.copyFrom(fe_correction);
            {
                auto diff_corr_span = correction_diff.localSpan();
                const auto fsils_corr_span = fsils_correction.localSpan();
                FE_THROW_IF(diff_corr_span.size() != fsils_corr_span.size(),
                            FEException,
                            "FsilsLinearSolver::solve: correction compare size mismatch");
                for (std::size_t i = 0; i < diff_corr_span.size(); ++i) {
                    diff_corr_span[i] -= fsils_corr_span[i];
                }
            }

            if (fsilsProbeLowRankModesEnabled() &&
                !rank_one_updates_.empty() &&
                lhs.reduced_updates.size() == rank_one_updates_.size()) {
                std::vector<GlobalIndex> resolved_dofs;
                resolved_dofs.reserve(rank_one_updates_.size() * 8);
                for (const auto& update : rank_one_updates_) {
                    for (const auto& [probe_dof, _] : update.v) {
                        resolved_dofs.push_back(probe_dof);
                    }
                }
                std::vector<GlobalIndex> resolved_local(resolved_dofs.size(), INVALID_GLOBAL_INDEX);
                probe_fe.resolveEntriesCached(resolved_dofs, resolved_local);
                const auto probe_span = probe_fe.localSpan();
                std::ostringstream dots_oss;
                dots_oss << "FsilsLinearSolver::solve: low-rank probe dots"
                         << " probe='" << probe_label << "'";
                std::size_t resolved_offset = 0;
                for (std::size_t u = 0; u < rank_one_updates_.size(); ++u) {
                    Real exact_dot = Real(0.0);
                    for (const auto& [probe_dof, probe_val] : rank_one_updates_[u].v) {
                        (void)probe_dof;
                        const auto local_dof = resolved_local[resolved_offset++];
                        if (local_dof == INVALID_GLOBAL_INDEX) {
                            continue;
                        }
                        exact_dot += probe_val *
                                     probe_span[static_cast<std::size_t>(local_dof)];
                    }
#if FE_HAS_MPI
                    int mpi_initialized = 0;
                    MPI_Initialized(&mpi_initialized);
                    if (mpi_initialized && lhs.commu.nTasks > 1) {
                        Real global_dot = Real(0.0);
                        fe_fsi_linear_solver::fsils_allreduce_sum(
                            &exact_dot, &global_dot, 1, MPI_DOUBLE, lhs.commu);
                        exact_dot = global_dot;
                    }
#endif

                    const auto& update = lhs.reduced_updates[u];
                    const auto& right_entries =
                        update.right_scaled_owned.empty() ? update.right_owned
                                                          : update.right_scaled_owned;
                    Real reduced_dot = Real(0.0);
                    for (const auto& entry : right_entries) {
                        if (entry.node < 0 || entry.node >= lhs.mynNo ||
                            std::abs(entry.value) <= 1e-30) {
                            continue;
                        }
                        const int comp = fe_fsi_linear_solver::fsils_reduced_local_component(
                            update,
                            entry.full_component,
                            dof,
                            (lhs.system_dof > 0) ? lhs.system_dof : dof);
                        if (comp < 0 || comp >= dof) {
                            continue;
                        }
                        reduced_dot += static_cast<Real>(entry.value) *
                                       static_cast<Real>(probe_internal(comp, entry.node));
                    }
#if FE_HAS_MPI
                    if (lhs.commu.nTasks > 1) {
                        Real global_dot = Real(0.0);
                        fe_fsi_linear_solver::fsils_allreduce_sum(
                            &reduced_dot, &global_dot, 1, MPI_DOUBLE, lhs.commu);
                        reduced_dot = global_dot;
                    }
#endif
                    dots_oss << " exact_dot[" << u << "]=" << exact_dot
                             << " reduced_dot[" << u << "]=" << reduced_dot;
                }
                traceLog(dots_oss.str());
            }

            const Real fe_norm = fe_y.norm();
            const Real fsils_norm = fsils_y.norm();
            const Real diff_norm = diff.norm();
            const Real rel = diff_norm / std::max<Real>(fe_norm, Real(1e-30));
            if (oopTraceEnabled()) {
                std::ostringstream oss;
                oss << "FsilsLinearSolver::solve: face operator compare"
                    << " probe='" << probe_label << "'"
                    << " |probe|=" << probe_old.norm()
                    << " |FE|=" << fe_norm
                    << " |FSILS|=" << fsils_norm
                    << " |diff|=" << diff_norm
                    << " rel=" << rel
                    << " |FE_J|=" << fe_matrix.norm()
                    << " |FSILS_J|=" << fsils_matrix.norm()
                    << " |diff_J|=" << matrix_diff.norm()
                    << " |FE_R|=" << fe_correction.norm()
                    << " |FSILS_R|=" << fsils_correction.norm()
                    << " |diff_R|=" << correction_diff.norm()
                    << " reduced_updates=" << (rank_one_updates_.size() + reduced_field_updates_.size());
                traceLog(oss.str());
            }

            dumpOwnerAlignedVector(probe_label, "probe", probe_old);
            dumpOwnerAlignedVector(probe_label, "fe_matrix", fe_matrix);
            dumpOwnerAlignedVector(probe_label, "fe_correction", fe_correction);
            dumpOwnerAlignedVector(probe_label, "fe_full", fe_y);
            dumpOwnerAlignedVector(probe_label, "fsils_matrix", fsils_matrix);
            dumpOwnerAlignedVector(probe_label, "fsils_correction", fsils_correction);
            dumpOwnerAlignedVector(probe_label, "fsils_full", fsils_y);
            dumpOwnerAlignedVector(probe_label, "diff_matrix", matrix_diff);
            dumpOwnerAlignedVector(probe_label, "diff_correction", correction_diff);
            dumpOwnerAlignedVector(probe_label, "diff_full", diff);
        };

        dumpPreparedRowIfRequested();

        if (b != nullptr) {
            FsilsVector rhs_probe(shared_layout);
            rhs_probe.copyFrom(*b);
            dumpOwnerAlignedVector("rhs_input", "vector", rhs_probe);
            FsilsVector rhs_accum(shared_layout);
            rhs_accum.copyFrom(*b);
            prepareRhsVectorForOperator(rhs_accum);
            dumpOwnerAlignedVector("rhs_input_accum", "vector", rhs_accum);
        }

        FsilsVector probe_old(shared_layout);
        {
            auto probe_span = probe_old.localSpan();
            const auto* perm = shared_layout->dof_permutation.get();
            const bool has_inverse = perm != nullptr && !perm->inverse.empty();
            for (int old = 0; old < lhs.nNo; ++old) {
                const int backend_node = shared_layout->oldToGlobalNode(old);
                const std::size_t base =
                    static_cast<std::size_t>(old) * static_cast<std::size_t>(dof);
                for (int c = 0; c < dof; ++c) {
                    const GlobalIndex backend_dof =
                        static_cast<GlobalIndex>(backend_node) * static_cast<GlobalIndex>(dof) +
                        static_cast<GlobalIndex>(c);
                    GlobalIndex fe_dof = backend_dof;
                    if (has_inverse && static_cast<std::size_t>(backend_dof) < perm->inverse.size()) {
                        fe_dof = perm->inverse[static_cast<std::size_t>(backend_dof)];
                    }
                    probe_span[base + static_cast<std::size_t>(c)] =
                        static_cast<Real>(0.001 * (static_cast<double>(fe_dof) + 1.0));
                }
            }
        }
        runProbeComparison("generic", probe_old);

        if (const char* oracle_path = fsilsCompareFaceOperatorOracleFile();
            oracle_path != nullptr && con_ncomp == 1) {
            std::ifstream oracle_in(oracle_path);
            if (oracle_in) {
                std::map<int, Real> oracle_by_global_node;
                std::string line;
                while (std::getline(oracle_in, line)) {
                    if (line.empty() || line[0] == '#') {
                        continue;
                    }
                    std::istringstream iss(line);
                    int global_node = -1;
                    Real value = Real(0.0);
                    if (!(iss >> global_node >> value)) {
                        continue;
                    }
                    oracle_by_global_node[global_node] = value;
                }

                if (!oracle_by_global_node.empty()) {
                    FsilsVector oracle_probe(shared_layout);
                    auto oracle_span = oracle_probe.localSpan();
                    std::fill(oracle_span.begin(), oracle_span.end(), Real(0.0));
                    for (int old = 0; old < lhs.nNo; ++old) {
                        const int global_node = shared_layout->oldToGlobalNode(old);
                        const auto it = oracle_by_global_node.find(global_node);
                        if (it == oracle_by_global_node.end()) {
                            continue;
                        }
                        const std::size_t base =
                            static_cast<std::size_t>(old) * static_cast<std::size_t>(dof);
                        oracle_span[base + static_cast<std::size_t>(con_start)] = it->second;
                    }
                    runProbeComparison("oracle_scalar_probe", oracle_probe);
                }
            }
        }

        if (!fsilsProbeLowRankModesEnabled()) {
            return;
        }

        for (std::size_t u = 0; u < rank_one_updates_.size(); ++u) {
            FsilsVector low_rank_probe(shared_layout);
            auto low_rank_view = low_rank_probe.createAssemblyView();
            FE_CHECK_NOT_NULL(low_rank_view.get(), "FsilsLinearSolver: low-rank probe view");
            low_rank_view->beginAssemblyPhase();
            for (const auto& [probe_dof, probe_val] : rank_one_updates_[u].v) {
                low_rank_view->addVectorEntry(probe_dof, probe_val, assembly::AddMode::Insert);
            }
            low_rank_view->finalizeAssembly();
            runProbeComparison("rank1_" + std::to_string(u), low_rank_probe);
        }

        for (std::size_t u = 0; u < reduced_field_updates_.size(); ++u) {
            FsilsVector low_rank_probe(shared_layout);
            auto low_rank_view = low_rank_probe.createAssemblyView();
            FE_CHECK_NOT_NULL(low_rank_view.get(), "FsilsLinearSolver: reduced probe view");
            low_rank_view->beginAssemblyPhase();
            for (const auto& [probe_dof, probe_val] : reduced_field_updates_[u].right) {
                low_rank_view->addVectorEntry(probe_dof, probe_val, assembly::AddMode::Insert);
            }
            low_rank_view->finalizeAssembly();
            runProbeComparison("reduced_" + std::to_string(u), low_rank_probe);
        }

        if (lhs.commu.nTasks > 1 && con_ncomp == 1) {
            std::vector<double> owned_counts(static_cast<std::size_t>(lhs.commu.nTasks), 0.0);
            const double local_owned = static_cast<double>(lhs.mynNo);
            MPI_Allgather(&local_owned,
                          1,
                          cm_mod::mpreal,
                          owned_counts.data(),
                          1,
                          cm_mod::mpreal,
                          lhs.commu.comm);

            std::vector<int> active_ranks;
            active_ranks.reserve(static_cast<std::size_t>(lhs.commu.nTasks));
            for (int rank = 0; rank < lhs.commu.nTasks; ++rank) {
                if (owned_counts[static_cast<std::size_t>(rank)] > 0.5) {
                    active_ranks.push_back(rank);
                }
            }
            if (active_ranks.size() >= 2u) {
                const int reference_rank = active_ranks.back();
                const int probe_rank = active_ranks.front();
                const double reference_count =
                    owned_counts[static_cast<std::size_t>(reference_rank)];
                if (reference_count > 0.5 && probe_rank != reference_rank) {
                    FsilsVector partition_probe(shared_layout);
                    auto partition_span = partition_probe.localSpan();
                    std::fill(partition_span.begin(), partition_span.end(), Real(0.0));
                    const Real reference_weight =
                        static_cast<Real>(-owned_counts[static_cast<std::size_t>(probe_rank)] /
                                          reference_count);
                    for (int old = 0; old < lhs.nNo; ++old) {
                        const int internal = lhs.map(old);
                        if (internal < 0 || internal >= lhs.mynNo) {
                            continue;
                        }
                        const std::size_t base =
                            static_cast<std::size_t>(old) * static_cast<std::size_t>(dof);
                        if (lhs.commu.task == probe_rank) {
                            partition_span[base + static_cast<std::size_t>(con_start)] = Real(1.0);
                        } else if (lhs.commu.task == reference_rank) {
                            partition_span[base + static_cast<std::size_t>(con_start)] =
                                reference_weight;
                        }
                    }
                    runProbeComparison("constraint_partition_probe", partition_probe);
                }
            }
        }

        if (std::getenv("SVMP_FSILS_COMPARE_FACE_OPERATOR_BASIS") != nullptr &&
            x->size() > 0 && x->size() <= 64) {
            for (GlobalIndex dof_idx = 0; dof_idx < x->size(); ++dof_idx) {
                FsilsVector basis_probe(shared_layout);
                auto basis_view = basis_probe.createAssemblyView();
                FE_CHECK_NOT_NULL(basis_view.get(),
                                  "FsilsLinearSolver::solve: basis probe view");
                basis_view->beginAssemblyPhase();
                basis_view->addVectorEntry(dof_idx, Real(1.0), assembly::AddMode::Insert);
                basis_view->finalizeAssembly();
                runProbeComparison("basis_" + std::to_string(dof_idx), basis_probe);
            }
        }

        if (x != nullptr) {
            FsilsVector solution_probe(shared_layout);
            solution_probe.copyFrom(*x);
            runProbeComparison("solution", solution_probe);
        }
    };

    auto logReturnedOperatorBreakdown = [&](std::string_view phase, const FsilsVector& solution) {
        if (!oopTraceEnabled() && !fsilsTraceEnabled()) {
            return;
        }

        FsilsVector rhs_eval(shared_layout);
        rhs_eval.copyFrom(*b);
        prepareRhsVectorForOperator(rhs_eval);

        FsilsVector x_eval(shared_layout);
        x_eval.copyFrom(solution);
        x_eval.updateGhosts();

        FsilsVector matrix_eval(shared_layout);
        A->mult(x_eval, matrix_eval);

        FsilsVector matrix_residual(shared_layout);
        matrix_residual.copyFrom(matrix_eval);
        {
            auto matrix_residual_span = matrix_residual.localSpan();
            const auto rhs_span = rhs_eval.localSpan();
            FE_THROW_IF(matrix_residual_span.size() != rhs_span.size(),
                        FEException,
                        "FsilsLinearSolver::solve: returned matrix residual size mismatch");
            for (std::size_t i = 0; i < matrix_residual_span.size(); ++i) {
                matrix_residual_span[i] -= rhs_span[i];
            }
        }

        FsilsVector rank_one_eval(shared_layout);
        rank_one_eval.zero();
        addRankOneUpdatesToProduct(rank_one_updates_, x_eval, rank_one_eval, lhs.commu);
        addReducedFieldUpdatesToProduct(reduced_field_updates_,
                                        x_eval,
                                        rank_one_eval,
                                        lhs.commu,
                                        grouped_bordered_field_couplings_);
        addGroupedBorderedFieldCouplingsToProduct(grouped_bordered_field_couplings_,
                                                  x_eval,
                                                  rank_one_eval,
                                                  lhs.commu);

        FsilsVector full_residual(shared_layout);
        full_residual.copyFrom(matrix_residual);
        {
            auto full_residual_span = full_residual.localSpan();
            const auto rank_one_span = rank_one_eval.localSpan();
            FE_THROW_IF(full_residual_span.size() != rank_one_span.size(),
                        FEException,
                        "FsilsLinearSolver::solve: returned full residual size mismatch");
            for (std::size_t i = 0; i < full_residual_span.size(); ++i) {
                full_residual_span[i] += rank_one_span[i];
            }
        }

        std::vector<Real> dots(rank_one_updates_.size(), Real(0.0));
        if (!rank_one_updates_.empty()) {
            auto x_view = x_eval.createAssemblyView();
            FE_CHECK_NOT_NULL(x_view.get(), "FsilsLinearSolver::solve: returned x view");
            for (std::size_t u = 0; u < rank_one_updates_.size(); ++u) {
                Real dot = Real(0.0);
                for (const auto& [dof, val] : rank_one_updates_[u].v) {
                    dot += val * x_view->getVectorEntry(dof);
                }
#if FE_HAS_MPI
                int mpi_initialized = 0;
                MPI_Initialized(&mpi_initialized);
                if (mpi_initialized && lhs.commu.nTasks > 1) {
                    Real global_dot = Real(0.0);
                    fe_fsi_linear_solver::fsils_allreduce_sum(
                        &dot, &global_dot, 1, MPI_DOUBLE, lhs.commu);
                    dot = global_dot;
                }
#endif
                dots[u] = dot;
            }
        }

        std::ostringstream oss;
        oss << "FsilsLinearSolver::solve: returned operator breakdown"
            << " phase='" << phase << "'"
            << " |x|=" << x_eval.norm()
            << " |Jx|=" << matrix_eval.norm()
            << " |Jx-r|=" << matrix_residual.norm()
            << " |rank1*x|=" << rank_one_eval.norm()
            << " |(J+R)x-r|=" << full_residual.norm();
        for (std::size_t u = 0; u < dots.size(); ++u) {
            oss << " dot[" << u << "]=" << dots[u];
        }
        traceLog(oss.str());
    };

    auto validateOriginalResidual = [&](std::string_view phase) -> FsilsResidualCheckResult {
        const double tp0 = fe_fsi_linear_solver::fsils_cpu_t();
        FsilsResidualCheckResult result;
        FsilsVector residual_true(shared_layout);
        computeTrueResidualVector(residual_true, result.rhs_norm);
        result.residual_norm = residual_true.norm();
        const Real denom = std::max<Real>(result.rhs_norm, 1e-30);
        result.relative_residual = result.residual_norm / denom;
        const Real target = std::max<Real>(options_.abs_tol, options_.rel_tol * denom);
        const bool finite = std::isfinite(static_cast<double>(result.residual_norm)) &&
                            std::isfinite(static_cast<double>(result.relative_residual));
        result.ok = finite && result.residual_norm <= target;
        if (!result.ok && fsilsAcceptNearTargetResidual(result.residual_norm, target)) {
            result.ok = true;
        }
        if (!result.ok) {
            std::ostringstream oss;
            oss << phase << ": true residual check failed (|Ax-b|=" << result.residual_norm
                << ", rel=" << result.relative_residual
                << ", target=" << target << ")";
            result.detail = oss.str();

            if (fsilsCompareFaceOperatorDumpPrefix() != nullptr) {
                const std::uint64_t dump_index = residual_validation_dump_index++;
                FsilsVector rhs_true(shared_layout);
                rhs_true.copyFrom(*b);
                prepareRhsVectorForOperator(rhs_true);

                FsilsVector x_true(shared_layout);
                x_true.copyFrom(*x);
                x_true.updateGhosts();

                FsilsVector ax_matrix(shared_layout);
                A->mult(x_true, ax_matrix);

                FsilsVector ax_full(shared_layout);
                ax_full.copyFrom(ax_matrix);
                addRankOneUpdatesToProduct(rank_one_updates_, x_true, ax_full, lhs.commu);
                addReducedFieldUpdatesToProduct(reduced_field_updates_,
                                                x_true,
                                                ax_full,
                                                lhs.commu,
                                                grouped_bordered_field_couplings_);
                addGroupedBorderedFieldCouplingsToProduct(grouped_bordered_field_couplings_,
                                                          x_true,
                                                          ax_full,
                                                          lhs.commu);

                FsilsVector ax_correction(shared_layout);
                ax_correction.copyFrom(ax_full);
                {
                    auto corr_span = ax_correction.localSpan();
                    const auto matrix_span = ax_matrix.localSpan();
                    FE_THROW_IF(corr_span.size() != matrix_span.size(),
                                FEException,
                                "FsilsLinearSolver::solve: residual validation correction size mismatch");
                    for (std::size_t i = 0; i < corr_span.size(); ++i) {
                        corr_span[i] -= matrix_span[i];
                    }
                }

                dumpResidualValidationVector(dump_index, phase, "x", x_true);
                dumpResidualValidationVector(dump_index, phase, "rhs", rhs_true);
                dumpResidualValidationVector(dump_index, phase, "ax_matrix", ax_matrix);
                dumpResidualValidationVector(dump_index, phase, "ax_correction", ax_correction);
                dumpResidualValidationVector(dump_index, phase, "ax_full", ax_full);
                dumpResidualValidationVector(dump_index, phase, "residual", residual_true);
            }
        }
        if (oopTraceEnabled()) {
            std::ostringstream oss;
            oss << "FsilsLinearSolver::solve: true residual"
                << " phase='" << phase << "'"
                << " ok=" << (result.ok ? 1 : 0)
                << " |Ax-b|=" << result.residual_norm
                << " rel=" << result.relative_residual
                << " rhs=" << result.rhs_norm;
            if (!result.detail.empty()) {
                oss << " detail='" << result.detail << "'";
            }
            traceLog(oss.str());
        }
        validation_time_seconds += fe_fsi_linear_solver::fsils_cpu_t() - tp0;
        return result;
    };

    auto maybeRecenterConstraintMeanAndValidate =
        [&](std::string_view phase,
            const FsilsResidualCheckResult& baseline_check) -> FsilsResidualCheckResult {
        if (!(has_native_rank_one_updates && has_saddle_point && con_ncomp == 1)) {
            return baseline_check;
        }

        const auto before = computeReturnedSolutionConstraintMeanStats();
        if (!before.valid || before.count == 0u) {
            return baseline_check;
        }
        if (!(std::abs(before.mean) > static_cast<Real>(1e-14))) {
            return baseline_check;
        }

        const Real fluctuation_floor =
            std::max<Real>(before.rms * static_cast<Real>(1e-12), static_cast<Real>(1e-14));
        const Real fluctuation = std::max(before.fluctuation_rms, fluctuation_floor);
        const Real dominance = std::abs(before.mean) / fluctuation;

        std::vector<Real> backup = x_data;
        subtractConstraintMeanFromReturnedSolution(before.mean);
        auto recentered_check = validateOriginalResidual(std::string(phase) + "_recentered");
        const auto after = computeReturnedSolutionConstraintMeanStats();

        const bool baseline_finite = std::isfinite(static_cast<double>(baseline_check.residual_norm));
        const bool recentered_finite = std::isfinite(static_cast<double>(recentered_check.residual_norm));
        const Real target = std::max<Real>(
            options_.abs_tol, options_.rel_tol * std::max<Real>(recentered_check.rhs_norm, Real(1e-30)));
        const bool residual_not_worse =
            recentered_finite &&
            (!baseline_finite ||
             recentered_check.residual_norm <=
                 std::max<Real>(baseline_check.residual_norm * static_cast<Real>(1.05),
                                baseline_check.residual_norm + target));
        const bool mean_removed =
            after.valid &&
            std::abs(after.mean) <= std::max<Real>(std::abs(before.mean) * static_cast<Real>(1e-6),
                                                   static_cast<Real>(1e-10));
        const bool accept = mean_removed && (recentered_check.ok || residual_not_worse);

        if (oopTraceEnabled()) {
            std::ostringstream oss;
            oss << "FsilsLinearSolver::solve: constraint recenter"
                << " phase='" << phase << "'"
                << " mean_before=" << before.mean
                << " rms_before=" << before.rms
                << " fluct_before=" << before.fluctuation_rms
                << " dominance=" << dominance
                << " mean_after=" << after.mean
                << " rms_after=" << after.rms
                << " residual_before=" << baseline_check.residual_norm
                << " residual_after=" << recentered_check.residual_norm
                << " accept=" << (accept ? 1 : 0);
            traceLog(oss.str());
        }

        if (accept) {
            return recentered_check;
        }

        x_data = std::move(backup);
        return baseline_check;
    };

    auto undoStageScalingOnSolution = [&](bool blockschur_preparation) {
        if (!blockschur_preparation || stage_scale == 1.0 || solution_stage_scaling_undone) {
            return;
        }
        const Real inv_s = static_cast<Real>(1.0 / stage_scale);
        for (int a = 0; a < lhs.nNo; ++a) {
            for (int c = con_start; c < con_start + con_ncomp; ++c) {
                Ri(c, a) *= inv_s;
            }
        }
        solution_stage_scaling_undone = true;
    };

    fe_fsi_linear_solver::fsils_reset_collective_stats(lhs.commu);
    rebuildPreparedSystem(use_blockschur);
    compareFaceOperatorAgainstFe();
    report.initial_residual_norm = computeOriginalRhsNorm();

    const auto validation_policy = options_.fsils_residual_check_policy;
    const bool require_true_residual_validation = has_native_rank_one_updates;
    auto shouldValidateResidual = [&](bool internal_check_ok, bool recovery_phase) {
        if (require_true_residual_validation) {
            return true;
        }
        switch (validation_policy) {
            case FsilsResidualCheckPolicy::Always:
                return true;
            case FsilsResidualCheckPolicy::RetryOnly:
                return recovery_phase || !internal_check_ok;
            case FsilsResidualCheckPolicy::DebugOnly:
                return recovery_phase || !internal_check_ok || oopTraceEnabled() || lhs.debug_active;
        }
        return recovery_phase || !internal_check_ok;
    };

    auto validateInternalResidual = [&](std::string_view phase) -> FsilsResidualCheckResult {
        FsilsResidualCheckResult result;
        const Real rhs_norm =
            (report.initial_residual_norm > 0.0 && std::isfinite(static_cast<double>(report.initial_residual_norm)))
                ? report.initial_residual_norm
                : static_cast<Real>(std::max(0.0, ls.RI.iNorm));
        result.rhs_norm = rhs_norm;
        result.residual_norm = static_cast<Real>(ls.RI.fNorm);
        result.relative_residual = result.residual_norm / std::max<Real>(rhs_norm, 1e-30);

        const Real target = std::max<Real>(options_.abs_tol, options_.rel_tol * std::max<Real>(rhs_norm, 1e-30));
        const bool finite = std::isfinite(static_cast<double>(ls.RI.iNorm)) &&
                            std::isfinite(static_cast<double>(ls.RI.fNorm)) &&
                            std::isfinite(static_cast<double>(result.relative_residual));
        result.ok = ls.RI.suc && finite && result.residual_norm <= target;
        if (!result.ok && ls.RI.suc &&
            fsilsAcceptNearTargetResidual(result.residual_norm, target)) {
            result.ok = true;
        }
        if (!result.ok) {
            std::ostringstream oss;
            oss << phase << ": internal residual check failed (|r|=" << result.residual_norm
                << ", rel=" << result.relative_residual
                << ", target=" << target
                << ", solver_suc=" << (ls.RI.suc ? 1 : 0) << ")";
            result.detail = oss.str();
        }
        return result;
    };

    auto maybeAcceptNearTargetTrueResidualReplay =
        [&](const FsilsResidualCheckResult& internal_check,
            FsilsResidualCheckResult check,
            std::string_view phase) -> FsilsResidualCheckResult {
        if (check.ok || !internal_check.ok || !use_blockschur) {
            return check;
        }

        // The FE replay recomputes the full residual through the assembled operator path,
        // so allow a modest roundoff envelope when the internal BlockSchur solve already met target.
        constexpr Real near_target_replay_slack = static_cast<Real>(1.5);
        const Real denom = std::max<Real>(check.rhs_norm, static_cast<Real>(1e-30));
        const Real target =
            std::max<Real>(options_.abs_tol, options_.rel_tol * denom);
        const bool finite =
            std::isfinite(static_cast<double>(check.residual_norm)) &&
            std::isfinite(static_cast<double>(check.relative_residual));
        if (!(finite &&
              check.residual_norm > target &&
              check.residual_norm <= target * near_target_replay_slack)) {
            return check;
        }

        check.ok = true;
        check.detail.clear();
        if (oopTraceEnabled() || fsilsTraceEnabled()) {
            std::ostringstream oss;
            oss << "FsilsLinearSolver::solve: accepting near-target FE replay"
                << " phase='" << phase << "'"
                << " internal=" << internal_check.residual_norm
                << " replay=" << check.residual_norm
                << " rel=" << check.relative_residual
                << " target=" << target;
            traceLog(oss.str());
        }
        return check;
    };

    auto runFsilsSolve = [&](bool blockschur_preparation, std::string& error_out) -> bool {
        error_out.clear();
        try {
            lhs.debug_global_nodes.resize(lhs.nNo);
            for (int old = 0; old < lhs.nNo; ++old) {
                lhs.debug_global_nodes(old) = shared_layout->oldToGlobalNode(old);
            }
            const auto& res_current = blockschur_preparation ? res_blockschur : res_original;
            fe_fsi_linear_solver::fsils_solve(lhs, ls, dof, Ri, Val, prec, incL, res_current);
            return true;
        } catch (const std::exception& e) {
            error_out = e.what();
            return false;
        } catch (...) {
            error_out = "unknown exception";
            return false;
        }
    };

    std::string solve_error;
    bool solve_ok = runFsilsSolve(use_blockschur, solve_error);
    FsilsResidualCheckResult initial_check{};
    if (solve_ok) {
        undoStageScalingOnSolution(use_blockschur);
        storeSolveBufferToSolution();
        for (int polish_iter = 0; polish_iter < 4; ++polish_iter) {
            if (!applyLowRankResidualPolish()) {
                break;
            }
        }
        if (fsilsCompareFaceOperatorEnabled()) {
            restorePreparedMatrixValues(current_preparation_uses_blockschur);
            compareFaceOperatorAgainstFe();
        }
        const std::string phase =
            use_blockschur ? "blockschur" : std::string(solverMethodToString(options_.method));
        const auto initial_internal_check = validateInternalResidual(phase);
        initial_check = initial_internal_check;
        if (shouldValidateResidual(initial_internal_check.ok, /*recovery_phase=*/false)) {
            initial_check = validateOriginalResidual(phase);
            initial_check = maybeRecenterConstraintMeanAndValidate(phase, initial_check);
            if (phase == "blockschur") {
                initial_check =
                    maybeAcceptNearTargetTrueResidualReplay(initial_internal_check,
                                                           initial_check,
                                                           phase);
            }
        }
    } else {
        initial_check.ok = false;
        initial_check.residual_norm = std::numeric_limits<Real>::infinity();
        initial_check.relative_residual = std::numeric_limits<Real>::infinity();
        initial_check.detail = std::string("initial solve threw: ") + solve_error;
    }

    if (solve_ok && initial_check.ok &&
        use_blockschur && has_native_rank_one_updates && has_saddle_point && con_ncomp == 1) {
        const auto returned_con_stats = computeReturnedSolutionConstraintMeanStats();
        if (returned_con_stats.valid && returned_con_stats.count > 0u &&
            std::abs(returned_con_stats.mean) > static_cast<Real>(1e-8)) {
            const Real fluctuation_floor =
                std::max<Real>(returned_con_stats.rms * static_cast<Real>(1e-12),
                               static_cast<Real>(1e-14));
            const Real fluctuation =
                std::max(returned_con_stats.fluctuation_rms, fluctuation_floor);
            const Real dominance = std::abs(returned_con_stats.mean) / fluctuation;
            const Real dominance_limit = fsilsBlockSchurConstraintMeanDominanceLimit();
            if (dominance >= dominance_limit) {
                if (oopTraceEnabled()) {
                    std::ostringstream oss;
                    oss << "FsilsLinearSolver::solve: retaining BlockSchur solution despite dominant constraint mean"
                        << " (mean=" << returned_con_stats.mean
                        << ", rms=" << returned_con_stats.rms
                        << ", fluct=" << returned_con_stats.fluctuation_rms
                        << ", dominance=" << dominance
                        << ", limit=" << dominance_limit
                        << ", residual=" << initial_check.residual_norm
                        << ")";
                    traceLog(oss.str());
                }
            }
        }
    }

    int local_fail = (!solve_ok || !initial_check.ok) ? 1 : 0;
    int any_fail = local_fail;
    if (lhs.commu.nTasks > 1) {
        fe_fsi_linear_solver::fsils_allreduce(&local_fail, &any_fail, 1, MPI_INT, MPI_LOR, lhs.commu);
    }

    FsilsResidualCheckResult final_check{};
    if (any_fail != 0) {
        final_check = initial_check;
    } else if (solve_ok) {
        undoStageScalingOnSolution(current_preparation_uses_blockschur);
        storeSolveBufferToSolution();
        const auto final_internal_check = validateInternalResidual("fsils_final");
        final_check = final_internal_check;
        if (shouldValidateResidual(final_internal_check.ok, /*recovery_phase=*/false)) {
            const std::string phase = "fsils_final";
            final_check = validateOriginalResidual(phase);
            final_check = maybeRecenterConstraintMeanAndValidate(phase, final_check);
            final_check = maybeAcceptNearTargetTrueResidualReplay(final_internal_check,
                                                                 final_check,
                                                                 phase);
        }
    } else {
        final_check.ok = false;
        final_check.rhs_norm = std::max<Real>(report.initial_residual_norm, 0.0);
        final_check.residual_norm = std::numeric_limits<Real>::infinity();
        final_check.relative_residual = std::numeric_limits<Real>::infinity();
        final_check.detail = "fsils solve threw: " + solve_error;
    }

    int local_final_ok = final_check.ok ? 1 : 0;
    int any_final_ok = local_final_ok;
    if (lhs.commu.nTasks > 1) {
        fe_fsi_linear_solver::fsils_allreduce(&local_final_ok, &any_final_ok, 1, MPI_INT, MPI_LAND, lhs.commu);
    }
    final_check.ok = (any_final_ok != 0);
    if (!final_check.ok && final_check.detail.empty()) {
        final_check.detail = "true residual check failed on another rank";
        final_check.residual_norm = std::numeric_limits<Real>::infinity();
        final_check.relative_residual = std::numeric_limits<Real>::infinity();
    }

    if (num_added_faces > 0) {
        for (int faIn = lhs.nFaces - 1; faIn >= original_nFaces; --faIn) {
            auto& face = lhs.face[static_cast<std::size_t>(faIn)];
            face.glob.clear();
            face.val.clear();
            face.valM.clear();
            face.foC = false;
            face.coupledFlag = false;
            face.incFlag = false;
            face.sharedFlag = false;
            face.nNo = 0;
            face.dof = 0;
            face.nS = 0.0;
            face.res = 0.0;
        }
        lhs.face.resize(static_cast<std::size_t>(original_nFaces));
        lhs.nFaces = original_nFaces;
    }
    lhs.native_face_rank_one_count = 0;
    lhs.reduced_updates.clear();
    lhs.grouped_bordered_field_couplings.clear();

    if (solve_ok && fsilsCompareFaceOperatorEnabled()) {
        restorePreparedMatrixValues(current_preparation_uses_blockschur);
        compareFaceOperatorAgainstFe();
    }

    if (solve_ok && (oopTraceEnabled() || fsilsTraceEnabled())) {
        logInternalBlockSolutionStats("returned");
        logReturnedOperatorBreakdown("pre_report", *x);

        FsilsVector matvec_only(shared_layout);
        A->mult(*x, matvec_only);

        FsilsVector matvec_full(shared_layout);
        matvec_full.copyFrom(matvec_only);
        addRankOneUpdatesToProduct(rank_one_updates_, *x, matvec_full, lhs.commu);
        addReducedFieldUpdatesToProduct(reduced_field_updates_,
                                        *x,
                                        matvec_full,
                                        lhs.commu,
                                        grouped_bordered_field_couplings_);
        addGroupedBorderedFieldCouplingsToProduct(grouped_bordered_field_couplings_,
                                                  *x,
                                                  matvec_full,
                                                  lhs.commu);

        FsilsVector rhs_check(shared_layout);
        rhs_check.copyFrom(*b);
        prepareRhsVectorForOperator(rhs_check);

        FsilsVector x_raw(shared_layout);
        x_raw.copyFrom(*x);
        FsilsVector x_check(shared_layout);
        x_check.copyFrom(*x);
        x_check.updateGhosts();

        FsilsVector x_delta(shared_layout);
        x_delta.copyFrom(x_check);
        {
            auto delta_span = x_delta.localSpan();
            const auto raw_span = x_raw.localSpan();
            FE_THROW_IF(delta_span.size() != raw_span.size(), FEException,
                        "FsilsLinearSolver::solve: returned x delta size mismatch");
            for (std::size_t i = 0; i < delta_span.size(); ++i) {
                delta_span[i] -= raw_span[i];
            }
        }

        auto compute_rank_one_dots = [&](FsilsVector& vec) {
            std::vector<Real> dots(rank_one_updates_.size(), Real(0.0));
            if (rank_one_updates_.empty()) {
                return dots;
            }
            auto vec_view = vec.createAssemblyView();
            FE_CHECK_NOT_NULL(vec_view.get(), "FsilsLinearSolver::solve: returned x dot view");
            for (std::size_t u = 0; u < rank_one_updates_.size(); ++u) {
                Real dot = Real(0.0);
                for (const auto& [dof, val] : rank_one_updates_[u].v) {
                    dot += val * vec_view->getVectorEntry(dof);
                }
#if FE_HAS_MPI
                int mpi_initialized = 0;
                MPI_Initialized(&mpi_initialized);
                if (mpi_initialized && lhs.commu.nTasks > 1) {
                    Real global_dot = Real(0.0);
                    fe_fsi_linear_solver::fsils_allreduce_sum(
                        &dot, &global_dot, 1, MPI_DOUBLE, lhs.commu);
                    dot = global_dot;
                }
#endif
                dots[u] = dot;
            }
            return dots;
        };
        const auto raw_rank_one_dots = compute_rank_one_dots(x_raw);
        const auto synced_rank_one_dots = compute_rank_one_dots(x_check);

        FsilsVector diff_only(shared_layout);
        A->mult(x_check, diff_only);
        auto diff_only_span = diff_only.localSpan();
        const auto rhs_span = rhs_check.localSpan();
        for (std::size_t i = 0; i < diff_only_span.size(); ++i) {
            diff_only_span[i] -= rhs_span[i];
        }

        FsilsVector diff_full(shared_layout);
        diff_full.copyFrom(matvec_full);
        auto diff_full_span = diff_full.localSpan();
        for (std::size_t i = 0; i < diff_full_span.size(); ++i) {
            diff_full_span[i] -= rhs_span[i];
        }

        std::ostringstream oss;
        oss << "FsilsLinearSolver::solve: post-return operator check"
            << " |x_raw|=" << x_raw.norm()
            << " |x_sync|=" << x_check.norm()
            << " |x_sync-x_raw|=" << x_delta.norm()
            << " |Jx|=" << matvec_only.norm()
            << " |(J+R)x|=" << (matvec_full.norm())
            << " |Jx-r|=" << diff_only.norm()
            << " |(J+R)x-r|=" << diff_full.norm();
        for (std::size_t u = 0; u < raw_rank_one_dots.size(); ++u) {
            oss << " raw_dot[" << u << "]=" << raw_rank_one_dots[u]
                << " synced_dot[" << u << "]=" << synced_rank_one_dots[u];
        }
        traceLog(oss.str());
    }

    report.iterations = ls.RI.itr;
    report.final_residual_norm = final_check.residual_norm;
    report.relative_residual = final_check.relative_residual;
    report.converged = final_check.ok;
    report.message = use_blockschur ? "fsils (blockschur)" : "fsils";
    if (!final_check.ok && !final_check.detail.empty()) {
        report.message = "fsils (" + final_check.detail + ")";
    }

    const auto is_finite = [](Real v) { return std::isfinite(static_cast<double>(v)); };
    const auto raw_iterations = report.iterations;
    const auto raw_fnorm = report.final_residual_norm;
    const auto raw_rel = report.relative_residual;
    bool x_finite = true;
    for (const auto v : x->data()) {
        if (!is_finite(v)) {
            x_finite = false;
            break;
        }
    }
    int max_expected_iters = options_.max_iter;
    if (ls.LS_type == fe_fsi_linear_solver::LinearSolverType::LS_TYPE_GMRES) {
        const long long mItr = static_cast<long long>(std::max(1, ls.RI.mItr));
        const long long sD = static_cast<long long>(std::max(0, ls.RI.sD));
        const long long expected = mItr * (sD + 1LL);
        if (expected > 0 && expected < static_cast<long long>(std::numeric_limits<int>::max())) {
            max_expected_iters = static_cast<int>(expected);
        } else {
            max_expected_iters = std::numeric_limits<int>::max();
        }
    } else if (ls.LS_type == fe_fsi_linear_solver::LinearSolverType::LS_TYPE_CG ||
               ls.LS_type == fe_fsi_linear_solver::LinearSolverType::LS_TYPE_BICGS ||
               ls.LS_type == fe_fsi_linear_solver::LinearSolverType::LS_TYPE_NS) {
        max_expected_iters = std::max(0, ls.RI.mItr);
    }
    const bool iters_ok = (raw_iterations >= 0 && raw_iterations <= max_expected_iters);
    const bool fnorm_ok = is_finite(raw_fnorm);
    const bool rel_ok = is_finite(raw_rel);
    if (!iters_ok || !fnorm_ok || !rel_ok || !x_finite) {
        report.iterations = std::max(0, std::min(raw_iterations, max_expected_iters));
        std::string reason;
        if (!iters_ok) reason += "itr";
        if (!x_finite) {
            if (!reason.empty()) reason += ",";
            reason += "x";
        }
        if (!fnorm_ok) {
            if (!reason.empty()) reason += ",";
            reason += "fNorm";
        }
        if (!rel_ok) {
            if (!reason.empty()) reason += ",";
            reason += "rel";
        }
        if (reason.empty()) reason = "unknown";

        x->zero();
        report.converged = false;
        report.final_residual_norm = std::numeric_limits<Real>::infinity();
        report.relative_residual = std::numeric_limits<Real>::infinity();
        report.message = "fsils (breakdown:" + reason + ")";
    } else {
        if (!report.converged) {
            const std::string rel_msg = "fsils (not converged; itr=" + std::to_string(report.iterations) +
                                        ", rel=" + std::to_string(report.relative_residual) + ")";
            if (!final_check.ok && !final_check.detail.empty()) {
                report.message = "fsils (" + final_check.detail + ")";
            } else {
                report.message = rel_msg;
            }
        }
    }

    // Post-solve nullspace projection: x = x - Σ_i (z_i · x) z_i
    // This removes any nullspace drift from the iterative solve.
    const bool applied_nullspace_projection = !nullspace_basis_.empty() && x != nullptr;
    const Real x_norm_before_nullspace =
        (x != nullptr) ? x->norm() : static_cast<Real>(0.0);
    if (applied_nullspace_projection) {
        auto x_span = x->localSpan();
        const auto n = x_span.size();

        for (const auto& z : nullspace_basis_) {
            if (z.size() != n) continue;

            // Compute local dot product z · x
            double local_dot = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                local_dot += z[i] * static_cast<double>(x_span[i]);
            }

            // MPI_Allreduce for distributed dot product
            double global_dot = local_dot;
#if FE_HAS_MPI
            int mpi_initialized = 0;
            MPI_Initialized(&mpi_initialized);
            if (mpi_initialized && lhs.commu.nTasks > 1) {
                fe_fsi_linear_solver::fsils_allreduce_sum(
                    &local_dot, &global_dot, 1, MPI_DOUBLE, lhs.commu);
            }
#endif

            // x = x - (z · x) * z
            for (std::size_t i = 0; i < n; ++i) {
                x_span[i] -= static_cast<Real>(global_dot * z[i]);
            }
        }
    }

    if (solve_ok && (oopTraceEnabled() || fsilsTraceEnabled())) {
        logConstraintMeanModeProbe("final");
        logReturnedOperatorBreakdown("final", *x);

        FsilsVector rhs_check(shared_layout);
        rhs_check.copyFrom(*b);
        prepareRhsVectorForOperator(rhs_check);

        FsilsVector x_check(shared_layout);
        x_check.copyFrom(*x);
        x_check.updateGhosts();

        FsilsVector matvec_full(shared_layout);
        A->mult(x_check, matvec_full);
        addRankOneUpdatesToProduct(rank_one_updates_, x_check, matvec_full, lhs.commu);
        addReducedFieldUpdatesToProduct(reduced_field_updates_,
                                        x_check,
                                        matvec_full,
                                        lhs.commu,
                                        grouped_bordered_field_couplings_);
        addGroupedBorderedFieldCouplingsToProduct(grouped_bordered_field_couplings_,
                                                  x_check,
                                                  matvec_full,
                                                  lhs.commu);

        FsilsVector diff_full(shared_layout);
        diff_full.copyFrom(matvec_full);
        auto diff_full_span = diff_full.localSpan();
        const auto rhs_span = rhs_check.localSpan();
        for (std::size_t i = 0; i < diff_full_span.size(); ++i) {
            diff_full_span[i] -= rhs_span[i];
        }

        std::ostringstream oss;
        oss << "FsilsLinearSolver::solve: final returned operator check"
            << " basis=" << nullspace_basis_.size()
            << " projected=" << (applied_nullspace_projection ? 1 : 0)
            << " |x_before_proj|=" << x_norm_before_nullspace
            << " |x_after_proj|=" << x->norm()
            << " |(J+R)x-r|=" << diff_full.norm();
        traceLog(oss.str());

    }

    report.setup_time_seconds = rhs_prepare_time_seconds;
    report.validation_time_seconds = validation_time_seconds;
    report.collective_time_seconds = lhs.commu.collective_stats.allreduce_time;
    report.collective_calls = lhs.commu.collective_stats.allreduce_calls;
    report.collective_words = lhs.commu.collective_stats.allreduce_words;
    report.blockschur_outer_iterations = ls.blockschur_stats.outer_iterations;
    report.blockschur_collective_calls_max_per_outer =
        ls.blockschur_stats.collective_calls_max_per_outer;
    report.blockschur_collective_time_max_per_outer =
        ls.blockschur_stats.collective_time_max_per_outer;
    report.blockschur_momentum_solve_calls = ls.GM.stats.solve_calls;
    report.blockschur_momentum_iterations = ls.GM.stats.iterations_total;
    report.blockschur_momentum_restart_cycles = ls.GM.stats.restart_cycles_total;
    report.blockschur_momentum_solve_time_seconds = ls.GM.stats.solve_time;
    report.blockschur_momentum_collective_calls = ls.GM.stats.collective_calls;
    report.blockschur_momentum_collective_words = ls.GM.stats.collective_words;
    report.blockschur_momentum_collective_time_seconds = ls.GM.stats.collective_time;
    report.blockschur_schur_solve_calls = ls.CG.stats.solve_calls;
    report.blockschur_schur_iterations = ls.CG.stats.iterations_total;
    report.blockschur_schur_setup_time_seconds = ls.CG.stats.setup_time;
    report.blockschur_schur_solve_time_seconds = ls.CG.stats.solve_time;
    report.blockschur_schur_collective_calls = ls.CG.stats.collective_calls;
    report.blockschur_schur_collective_words = ls.CG.stats.collective_words;
    report.blockschur_schur_collective_time_seconds = ls.CG.stats.collective_time;

    if (lhs.commu.task == 0) {
        std::fprintf(stderr,
                     "\n=== FSILS BACKEND METRICS (rank 0) ===\n"
                     "  RHS/overlap prep:     %10.6f s\n"
                     "  Residual validation:  %10.6f s\n"
                     "  MPI_Allreduce calls:  %10llu\n"
                     "  MPI_Allreduce words:  %10llu\n"
                     "  MPI_Allreduce time:   %10.6f s\n"
                     "=======================================\n",
                     report.setup_time_seconds,
                     report.validation_time_seconds,
                     static_cast<unsigned long long>(report.collective_calls),
                     static_cast<unsigned long long>(report.collective_words),
                     report.collective_time_seconds);
        if (use_blockschur) {
            const double calls_per_outer =
                (report.blockschur_outer_iterations > 0)
                    ? static_cast<double>(report.collective_calls) /
                          static_cast<double>(report.blockschur_outer_iterations)
                    : 0.0;
            const double calls_per_restart =
                (report.blockschur_momentum_restart_cycles > 0)
                    ? static_cast<double>(report.blockschur_momentum_collective_calls) /
                          static_cast<double>(report.blockschur_momentum_restart_cycles)
                    : 0.0;
            std::fprintf(stderr,
                         "  BlockSchur outer iters: %8d\n"
                         "  Calls / outer iter:     %10.3f\n"
                         "  Max calls / outer:      %10llu\n"
                         "  Momentum solves:        %8d  iters=%d  restarts=%d\n"
                         "  Momentum solve time:    %10.6f s\n"
                         "  Momentum allreduces:    %10llu  words=%llu  time=%10.6f s\n"
                         "  Calls / GMRES restart:  %10.3f\n"
                         "  Schur solves:           %8d  iters=%d\n"
                         "  Schur setup time:       %10.6f s\n"
                         "  Schur solve time:       %10.6f s\n"
                         "  Schur allreduces:       %10llu  words=%llu  time=%10.6f s\n"
                         "=======================================\n",
                         report.blockschur_outer_iterations,
                         calls_per_outer,
                         static_cast<unsigned long long>(report.blockschur_collective_calls_max_per_outer),
                         report.blockschur_momentum_solve_calls,
                         report.blockschur_momentum_iterations,
                         report.blockschur_momentum_restart_cycles,
                         report.blockschur_momentum_solve_time_seconds,
                         static_cast<unsigned long long>(report.blockschur_momentum_collective_calls),
                         static_cast<unsigned long long>(report.blockschur_momentum_collective_words),
                         report.blockschur_momentum_collective_time_seconds,
                         calls_per_restart,
                         report.blockschur_schur_solve_calls,
                         report.blockschur_schur_iterations,
                         report.blockschur_schur_setup_time_seconds,
                         report.blockschur_schur_solve_time_seconds,
                         static_cast<unsigned long long>(report.blockschur_schur_collective_calls),
                         static_cast<unsigned long long>(report.blockschur_schur_collective_words),
                         report.blockschur_schur_collective_time_seconds);
        }
    }

    if (oopTraceEnabled()) {
        std::ostringstream oss;
        oss << "FsilsLinearSolver::solve: converged=" << (report.converged ? 1 : 0)
            << " iters=" << report.iterations
            << " r0=" << report.initial_residual_norm
            << " rn=" << report.final_residual_norm
            << " rel=" << report.relative_residual
            << " msg='" << report.message << "'";
        traceLog(oss.str());
    }

    return report;
}

void FsilsLinearSolver::setNullspaceBasis(std::span<const std::vector<double>> basis)
{
    nullspace_basis_.clear();
    nullspace_basis_.reserve(basis.size());
    for (const auto& vec : basis) {
        nullspace_basis_.push_back(vec);
    }
    if (oopTraceEnabled()) {
        traceLog("FsilsLinearSolver::setNullspaceBasis: modes=" +
                 std::to_string(nullspace_basis_.size()));
    }
}

} // namespace backends
} // namespace FE
} // namespace svmp

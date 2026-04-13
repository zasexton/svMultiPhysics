/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "TimeStepping/NewtonSolver.h"

#include "Backends/Interfaces/BackendFactory.h"
#include "Constraints/AffineConstraints.h"
#include "Constraints/GaugeDiagnostics.h"
#include "Constraints/GaugeRegistry.h"
#include "Core/FEException.h"
#include "Core/Logger.h"
#include "Dofs/DofIndexSet.h"
#include "Dofs/EntityDofMap.h"
#include "Auxiliary/AuxiliaryOperatorRegistry.h"
#include "Auxiliary/AuxiliaryStateManager.h"
#include "Systems/SystemsExceptions.h"

#if defined(FE_HAS_FSILS)
#  include "Backends/FSILS/FsilsMatrix.h"
#  include "Backends/FSILS/FsilsVector.h"
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <vector>

#if FE_HAS_MPI
#  include <mpi.h>
#endif

namespace svmp {
namespace FE {
namespace timestepping {

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

void traceLog(const std::string& msg)
{
    if (!oopTraceEnabled()) {
        return;
    }
    FE_LOG_INFO(msg);
}

[[nodiscard]] int mpiRank() noexcept
{
#if FE_HAS_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        return 0;
    }
    int rank = 0;
    (void)MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
#else
    return 0;
#endif
}

[[nodiscard]] bool mpiMultiTaskActive() noexcept
{
#if FE_HAS_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        return false;
    }

    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size > 1;
#else
    return false;
#endif
}

[[nodiscard]] bool nativeFaceRankOnePromotionEnabled() noexcept
{
    const char* env = std::getenv("SVMP_DISABLE_MPI_NATIVE_RANK1_PROMOTION");
    if (env == nullptr) {
        return true;
    }
    while (*env == ' ' || *env == '\t' || *env == '\n' || *env == '\r') {
        ++env;
    }
    if (*env == '\0') {
        return true;
    }
    return *env == '0';
}

[[nodiscard]] std::optional<double> explicitRankOneAfterRelativeResidualThreshold() noexcept
{
    const char* env = std::getenv("SVMP_FORCE_EXPLICIT_RANK_ONE_AFTER_REL_RES");
    if (env == nullptr) {
        return std::nullopt;
    }

    char* end = nullptr;
    const double value = std::strtod(env, &end);
    if (end == env || !std::isfinite(value) || value <= 0.0) {
        return std::nullopt;
    }
    return value;
}

[[nodiscard]] bool firstDirectOnlyReducedLineSearchEnabled() noexcept
{
    const char* env = std::getenv("SVMP_NEWTON_LINE_SEARCH_FIRST_DIRECT_ONLY_REDUCED");
    if (env == nullptr) {
        return false;
    }
    while (*env == ' ' || *env == '\t' || *env == '\n' || *env == '\r') {
        ++env;
    }
    return *env != '\0' && *env != '0';
}

[[nodiscard]] std::optional<double> lateDirectOnlyReducedTighteningThreshold() noexcept
{
    const char* env = std::getenv("SVMP_FSILS_TIGHTEN_DIRECT_ONLY_AFTER_REL_RES");
    if (env == nullptr) {
        return std::nullopt;
    }

    char* end = nullptr;
    const double value = std::strtod(env, &end);
    if (end == env || !std::isfinite(value) || value <= 0.0) {
        return std::nullopt;
    }
    return value;
}

[[nodiscard]] Real directOnlyOutletJacobianScale(const std::size_t update_count) noexcept
{
    const char* env = std::getenv("SVMP_DIRECT_ONLY_OUTLET_JACOBIAN_SCALE");
    if (env != nullptr) {
        char* end = nullptr;
        const double value = std::strtod(env, &end);
        if (end != env && std::isfinite(value) && value > 0.0) {
            return static_cast<Real>(value);
        }
    }

    if (update_count <= 1u) {
        return static_cast<Real>(1.0);
    }
    // With boundary AuxiliaryInputRef kernels forced onto the interpreter path,
    // the direct-only outlet updates recover full-Newton behavior in both serial
    // and MPI, so do not damp the built-in Jacobian by default.
    return static_cast<Real>(1.0);
}

[[nodiscard]] int directOnlyOutletJacobianRebuildPeriod(const std::size_t update_count) noexcept
{
    const char* env = std::getenv("SVMP_DIRECT_ONLY_OUTLET_JACOBIAN_REBUILD_PERIOD");
    if (env != nullptr) {
        char* end = nullptr;
        const long value = std::strtol(env, &end, 10);
        if (end != env && value > 0) {
            return static_cast<int>(value);
        }
    }

    if (update_count <= 1u) {
        return 1;
    }

    // Keep the direct-only outlet Jacobian current every Newton step now that
    // the boundary auxiliary-input path is corrected.
    return 1;
}

[[nodiscard]] Real lateDirectOnlyReducedInnerRelTol() noexcept
{
    const char* env = std::getenv("SVMP_FSILS_TIGHTEN_DIRECT_ONLY_INNER_REL_TOL");
    if (env == nullptr) {
        return static_cast<Real>(1e-6);
    }

    char* end = nullptr;
    const double value = std::strtod(env, &end);
    if (end == env || !std::isfinite(value) || value <= 0.0) {
        return static_cast<Real>(1e-6);
    }
    return static_cast<Real>(value);
}

template <typename T>
[[nodiscard]] T mpiAllreduceSumIfActive(T value) noexcept
{
#if FE_HAS_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        return value;
    }

    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size <= 1) {
        return value;
    }

    T global = value;
    if constexpr (std::is_same_v<T, int>) {
        MPI_Allreduce(&value, &global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    } else {
        MPI_Allreduce(&value, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    return global;
#else
    return value;
#endif
}

[[nodiscard]] std::vector<Real> gatherGlobalDenseVectorFromOwnedEntries(
    backends::GenericVector& vec,
    std::size_t n,
    const dofs::IndexSet& owned_dofs)
{
    auto view = vec.createAssemblyView();
    FE_CHECK_NOT_NULL(view.get(), "NewtonSolver: global dense gather view");

    std::vector<Real> local(n, Real(0.0));
    for (const auto dof : owned_dofs) {
        const auto idx = static_cast<std::size_t>(dof);
        if (idx >= n) {
            continue;
        }
        local[idx] = view->getVectorEntry(dof);
    }

#if FE_HAS_MPI
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized && !local.empty()) {
        std::vector<Real> global(local.size(), Real(0.0));
        MPI_Allreduce(local.data(),
                      global.data(),
                      static_cast<int>(local.size()),
                      MPI_DOUBLE,
                      MPI_SUM,
                      MPI_COMM_WORLD);
        return global;
    }
#endif

    return local;
}

[[nodiscard]] bool jacobianCheckEnabled() noexcept
{
    const char* env = std::getenv("SVMP_FE_JACOBIAN_CHECK");
    if (env == nullptr) {
        return false;
    }
    std::string v(env);
    std::transform(v.begin(), v.end(), v.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return !(v == "0" || v == "false" || v == "off" || v == "no");
}

[[nodiscard]] int jacobianCheckNewtonIteration() noexcept
{
    const char* env = std::getenv("SVMP_FE_JACOBIAN_CHECK_IT");
    if (env == nullptr) {
        return 0;
    }
    char* end = nullptr;
    const long v = std::strtol(env, &end, 10);
    if (end == env) {
        return 0;
    }
    if (v < 0) {
        return 0;
    }
    if (v > std::numeric_limits<int>::max()) {
        return std::numeric_limits<int>::max();
    }
    return static_cast<int>(v);
}

[[nodiscard]] double jacobianCheckRelativeStep() noexcept
{
    const char* env = std::getenv("SVMP_FE_JACOBIAN_CHECK_STEP");
    if (env == nullptr) {
        return 1e-7;
    }
    char* end = nullptr;
    const double v = std::strtod(env, &end);
    if (end == env) {
        return 1e-7;
    }
    if (!(v > 0.0) || !std::isfinite(v)) {
        return 1e-7;
    }
    return v;
}

[[nodiscard]] int lineSearchIterationsNeededToReachAlphaMin(double alpha_min,
                                                            double shrink) noexcept
{
    if (!(alpha_min > 0.0) || alpha_min >= 1.0 ||
        !(shrink > 0.0) || shrink >= 1.0) {
        return 1;
    }

    double alpha = 1.0;
    int iterations = 1;
    while (alpha > alpha_min && iterations < std::numeric_limits<int>::max()) {
        alpha *= shrink;
        if (alpha < alpha_min) {
            alpha = alpha_min;
        }
        ++iterations;
        if (alpha <= alpha_min) {
            break;
        }
    }
    return std::max(1, iterations);
}

void axpy(backends::GenericVector& y, Real alpha, const backends::GenericVector& x)
{
    auto ys = y.localSpan();
    auto xs = x.localSpan();
    FE_CHECK_ARG(ys.size() == xs.size(), "NewtonSolver: axpy size mismatch");
    for (std::size_t i = 0; i < ys.size(); ++i) {
        ys[i] += alpha * xs[i];
    }
}

void copyVector(backends::GenericVector& dst, const backends::GenericVector& src)
{
    auto d = dst.localSpan();
    auto s = src.localSpan();
    FE_CHECK_ARG(d.size() == s.size(), "NewtonSolver: copyVector size mismatch");
    std::copy(s.begin(), s.end(), d.begin());
}

double residualNormForConvergence(const backends::GenericVector& r, backends::GenericVector& scratch)
{
    if (r.backendKind() != backends::BackendKind::FSILS) {
        return r.norm();
    }

#if defined(FE_HAS_FSILS)
    const auto* r_fs = dynamic_cast<const backends::FsilsVector*>(&r);
    auto* scratch_fs = dynamic_cast<backends::FsilsVector*>(&scratch);
    if (!r_fs || !scratch_fs) {
        return r.norm();
    }

    const auto src = r_fs->localSpan();
    auto dst = scratch_fs->localSpan();
    FE_CHECK_ARG(src.size() == dst.size(), "NewtonSolver: FSILS residual scratch size mismatch");
    std::copy(src.begin(), src.end(), dst.begin());

    // Assemble-time residuals are distributed by element ownership. FSILS expects overlap
    // contributions to be summed before norm/dot-based convergence checks.
    scratch_fs->accumulateOverlap();
    return scratch_fs->norm();
#else
    return r.norm();
#endif
}

double auxiliaryResidualNormForConvergence(const systems::FESystem::BorderedCouplingData& bordered)
{
    if (!bordered.active || bordered.g.empty()) {
        return 0.0;
    }

    long double local_sq = 0.0L;
    for (const auto v : bordered.g) {
        local_sq += static_cast<long double>(v) * static_cast<long double>(v);
    }

#if FE_HAS_MPI
    if (!bordered.globally_reduced) {
        int mpi_initialized = 0;
        MPI_Initialized(&mpi_initialized);
        if (mpi_initialized) {
            long double global_sq = 0.0L;
            MPI_Allreduce(&local_sq, &global_sq, 1, MPI_LONG_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            local_sq = global_sq;
        }
    }
#endif

    return std::sqrt(static_cast<double>(local_sq));
}

double borderedResidualNormForConvergence(const backends::GenericVector& r,
                                          backends::GenericVector& scratch,
                                          const systems::FESystem::BorderedCouplingData& bordered)
{
    const double pde_norm = residualNormForConvergence(r, scratch);
    const double aux_norm = auxiliaryResidualNormForConvergence(bordered);
    return std::hypot(pde_norm, aux_norm);
}

[[nodiscard]] std::pair<double, double> borderedResidualNormComponentsForConvergence(
    const backends::GenericVector& r,
    backends::GenericVector& scratch,
    const systems::FESystem::BorderedCouplingData& bordered)
{
    return {
        residualNormForConvergence(r, scratch),
        auxiliaryResidualNormForConvergence(bordered)
    };
}

void zeroVectorEntries(std::span<const GlobalIndex> dofs, backends::GenericVector& vec)
{
    if (dofs.empty()) {
        return;
    }
    auto view = vec.createAssemblyView();
    FE_CHECK_NOT_NULL(view.get(), "NewtonSolver: zeroVectorEntries view");
    view->beginAssemblyPhase();
    view->zeroVectorEntries(dofs);
    view->finalizeAssembly();
}

void accumulateOverlapIfNeeded(backends::GenericVector& vec)
{
#if defined(FE_HAS_FSILS)
    if (vec.backendKind() != backends::BackendKind::FSILS) {
        return;
    }
    auto* fs = dynamic_cast<backends::FsilsVector*>(&vec);
    if (fs == nullptr) {
        return;
    }
    fs->accumulateOverlap();
#else
    (void)vec;
#endif
}

enum class FsilsPostSolveSyncMode {
    Off,
    UpdateGhosts,
    AccumulateOverlap,
    AccumulateThenUpdateGhosts,
};

[[nodiscard]] FsilsPostSolveSyncMode fsilsPostSolveSyncMode() noexcept
{
    const char* env = std::getenv("SVMP_FSILS_POST_SOLVE_SYNC");
    if (env == nullptr) {
        return FsilsPostSolveSyncMode::Off;
    }

    std::string value(env);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (value == "update" || value == "ghost" || value == "updateghosts") {
        return FsilsPostSolveSyncMode::UpdateGhosts;
    }
    if (value == "accumulate" || value == "sum") {
        return FsilsPostSolveSyncMode::AccumulateOverlap;
    }
    if (value == "both" || value == "accumulate_then_update") {
        return FsilsPostSolveSyncMode::AccumulateThenUpdateGhosts;
    }
    return FsilsPostSolveSyncMode::Off;
}

[[nodiscard]] bool newtonDirectionCheckEnabled() noexcept
{
    const char* env = std::getenv("SVMP_NEWTON_DIRECTION_CHECK");
    if (env == nullptr) {
        return false;
    }
    std::string value(env);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return !(value == "0" || value == "false" || value == "off" || value == "no");
}

[[nodiscard]] std::string describeFieldComponentDof(const systems::FESystem& sys,
                                                    GlobalIndex dof)
{
    const auto comp = sys.fieldMap().getComponentOfDof(dof);
    if (!comp) {
        return "dof=" + std::to_string(dof);
    }

    const auto field_idx = static_cast<std::size_t>(std::max(comp->first, 0));
    if (field_idx >= sys.fieldMap().numFields()) {
        return "dof=" + std::to_string(dof);
    }

    const auto& field = sys.fieldMap().getField(field_idx);
    if (field.n_components <= 1) {
        return field.name + "(dof=" + std::to_string(dof) + ")";
    }

    return field.name + "[" + std::to_string(static_cast<int>(comp->second)) +
           "](dof=" + std::to_string(dof) + ")";
}

struct JacobianCheckComponentStats {
    std::string label{};
    double fd_sq{0.0};
    double err_sq{0.0};
    double matrix_err_sq{0.0};
};

void logJacobianCheckComponentBreakdown(const systems::FESystem& sys,
                                       backends::GenericVector& fd,
                                       backends::GenericVector& total_err,
                                       backends::GenericVector& matrix_err)
{
    const auto& fmap = sys.fieldMap();
    const auto owned_dofs = sys.dofHandler().getPartition().locallyOwned().toVector();
    if (owned_dofs.empty() || fmap.numFields() == 0) {
        return;
    }

    std::vector<JacobianCheckComponentStats> stats;
    stats.reserve(fmap.numFields() * 3u);
    std::vector<int> field_offsets(fmap.numFields(), -1);
    for (std::size_t field_idx = 0; field_idx < fmap.numFields(); ++field_idx) {
        field_offsets[field_idx] = static_cast<int>(stats.size());
        const auto& field = fmap.getField(field_idx);
        if (field.n_components <= 1) {
            stats.push_back(JacobianCheckComponentStats{field.name});
            continue;
        }
        for (LocalIndex comp = 0; comp < field.n_components; ++comp) {
            stats.push_back(JacobianCheckComponentStats{
                field.name + "[" + std::to_string(static_cast<int>(comp)) + "]"
            });
        }
    }

    auto fd_view = fd.createAssemblyView();
    auto err_view = total_err.createAssemblyView();
    auto matrix_err_view = matrix_err.createAssemblyView();
    FE_CHECK_NOT_NULL(fd_view.get(), "NewtonSolver: jacobian check fd view");
    FE_CHECK_NOT_NULL(err_view.get(), "NewtonSolver: jacobian check err view");
    FE_CHECK_NOT_NULL(matrix_err_view.get(), "NewtonSolver: jacobian check matrix err view");

    for (const auto dof : owned_dofs) {
        const auto comp = fmap.getComponentOfDof(dof);
        if (!comp) {
            continue;
        }
        const auto field_idx = static_cast<std::size_t>(std::max(comp->first, 0));
        if (field_idx >= fmap.numFields()) {
            continue;
        }
        int stat_idx = field_offsets[field_idx];
        const auto n_comp = fmap.numComponents(field_idx);
        if (n_comp > 1) {
            const auto comp_idx = static_cast<int>(comp->second);
            if (comp_idx < 0 || comp_idx >= n_comp) {
                continue;
            }
            stat_idx += comp_idx;
        }
        if (stat_idx < 0 || static_cast<std::size_t>(stat_idx) >= stats.size()) {
            continue;
        }

        const double fd_val = static_cast<double>(fd_view->getVectorEntry(dof));
        const double err_val = static_cast<double>(err_view->getVectorEntry(dof));
        const double matrix_err_val = static_cast<double>(matrix_err_view->getVectorEntry(dof));
        auto& s = stats[static_cast<std::size_t>(stat_idx)];
        s.fd_sq += fd_val * fd_val;
        s.err_sq += err_val * err_val;
        s.matrix_err_sq += matrix_err_val * matrix_err_val;
    }

#if FE_HAS_MPI
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized && !stats.empty()) {
        std::vector<double> packed(stats.size() * 3u, 0.0);
        for (std::size_t i = 0; i < stats.size(); ++i) {
            packed[3u * i + 0u] = stats[i].fd_sq;
            packed[3u * i + 1u] = stats[i].err_sq;
            packed[3u * i + 2u] = stats[i].matrix_err_sq;
        }
        std::vector<double> reduced(packed.size(), 0.0);
        MPI_Allreduce(packed.data(),
                      reduced.data(),
                      static_cast<int>(packed.size()),
                      MPI_DOUBLE,
                      MPI_SUM,
                      MPI_COMM_WORLD);
        for (std::size_t i = 0; i < stats.size(); ++i) {
            stats[i].fd_sq = reduced[3u * i + 0u];
            stats[i].err_sq = reduced[3u * i + 1u];
            stats[i].matrix_err_sq = reduced[3u * i + 2u];
        }
    }
#endif

    if (mpiRank() != 0) {
        return;
    }

    std::ostringstream oss;
    oss << "NewtonSolver: Jacobian check component norms";
    for (const auto& s : stats) {
        oss << " [" << s.label
            << " fd=" << std::sqrt(std::max(0.0, s.fd_sq))
            << " total_err=" << std::sqrt(std::max(0.0, s.err_sq))
            << " matrix_err=" << std::sqrt(std::max(0.0, s.matrix_err_sq))
            << "]";
    }
    FE_LOG_INFO(oss.str());
}

void logJacobianCheckTopEntries(const systems::FESystem& sys,
                                backends::GenericVector& err,
                                std::size_t top_k)
{
    const auto owned_dofs = sys.dofHandler().getPartition().locallyOwned().toVector();
    if (owned_dofs.empty() || top_k == 0u) {
        return;
    }

    auto err_view = err.createAssemblyView();
    FE_CHECK_NOT_NULL(err_view.get(), "NewtonSolver: jacobian check top-entry view");

    struct Entry {
        GlobalIndex dof{INVALID_GLOBAL_INDEX};
        double value{0.0};
    };

    std::vector<Entry> top_entries;
    top_entries.reserve(top_k);
    const auto maybe_insert = [&](GlobalIndex dof, double value) {
        const double abs_value = std::abs(value);
        if (!(abs_value > 0.0) || !std::isfinite(abs_value)) {
            return;
        }
        if (top_entries.size() < top_k) {
            top_entries.push_back(Entry{dof, value});
        } else {
            auto min_it = std::min_element(
                top_entries.begin(), top_entries.end(),
                [](const Entry& a, const Entry& b) { return std::abs(a.value) < std::abs(b.value); });
            if (min_it != top_entries.end() && abs_value > std::abs(min_it->value)) {
                *min_it = Entry{dof, value};
            }
        }
    };

    for (const auto dof : owned_dofs) {
        maybe_insert(dof, static_cast<double>(err_view->getVectorEntry(dof)));
    }

    std::sort(top_entries.begin(), top_entries.end(),
              [](const Entry& a, const Entry& b) { return std::abs(a.value) > std::abs(b.value); });

    std::ostringstream oss;
    oss << "NewtonSolver: Jacobian check top |Jv-FD| entries rank=" << mpiRank();
    for (const auto& entry : top_entries) {
        oss << " [" << describeFieldComponentDof(sys, entry.dof)
            << " value=" << entry.value << "]";
    }
    FE_LOG_INFO(oss.str());
}

void logVectorComponentNorms(const systems::FESystem& sys,
                             backends::GenericVector& vec,
                             std::string_view label)
{
    const auto& fmap = sys.fieldMap();
    const auto owned_dofs = sys.dofHandler().getPartition().locallyOwned().toVector();
    if (owned_dofs.empty() || fmap.numFields() == 0) {
        return;
    }

    struct ComponentNorm {
        std::string label{};
        double sq_norm{0.0};
        double sum{0.0};
        std::uint64_t count{0};
    };

    std::vector<ComponentNorm> comps;
    comps.reserve(fmap.numFields() * 3u);
    std::vector<int> field_offsets(fmap.numFields(), -1);
    for (std::size_t field_idx = 0; field_idx < fmap.numFields(); ++field_idx) {
        field_offsets[field_idx] = static_cast<int>(comps.size());
        const auto& field = fmap.getField(field_idx);
        if (field.n_components <= 1) {
            comps.push_back(ComponentNorm{field.name});
            continue;
        }
        for (LocalIndex comp = 0; comp < field.n_components; ++comp) {
            comps.push_back(ComponentNorm{
                field.name + "[" + std::to_string(static_cast<int>(comp)) + "]"
            });
        }
    }

    auto view = vec.createAssemblyView();
    FE_CHECK_NOT_NULL(view.get(), "NewtonSolver: vector component norm view");

    for (const auto dof : owned_dofs) {
        const auto fc = fmap.getComponentOfDof(dof);
        if (!fc) {
            continue;
        }
        const auto field_idx = static_cast<std::size_t>(std::max(fc->first, 0));
        if (field_idx >= fmap.numFields()) {
            continue;
        }
        int comp_idx = field_offsets[field_idx];
        const auto n_comp = fmap.numComponents(field_idx);
        if (n_comp > 1) {
            comp_idx += static_cast<int>(fc->second);
        }
        if (comp_idx < 0 || static_cast<std::size_t>(comp_idx) >= comps.size()) {
            continue;
        }
        const double v = static_cast<double>(view->getVectorEntry(dof));
        auto& comp = comps[static_cast<std::size_t>(comp_idx)];
        comp.sq_norm += v * v;
        comp.sum += v;
        comp.count += 1;
    }

#if FE_HAS_MPI
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized && !comps.empty()) {
        std::vector<double> local_norm(comps.size(), 0.0);
        std::vector<double> global_norm(comps.size(), 0.0);
        std::vector<double> local_sum(comps.size(), 0.0);
        std::vector<double> global_sum(comps.size(), 0.0);
        std::vector<unsigned long long> local_count(comps.size(), 0ull);
        std::vector<unsigned long long> global_count(comps.size(), 0ull);
        for (std::size_t i = 0; i < comps.size(); ++i) {
            local_norm[i] = comps[i].sq_norm;
            local_sum[i] = comps[i].sum;
            local_count[i] = static_cast<unsigned long long>(comps[i].count);
        }
        MPI_Allreduce(local_norm.data(),
                      global_norm.data(),
                      static_cast<int>(local_norm.size()),
                      MPI_DOUBLE,
                      MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Allreduce(local_sum.data(),
                      global_sum.data(),
                      static_cast<int>(local_sum.size()),
                      MPI_DOUBLE,
                      MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Allreduce(local_count.data(),
                      global_count.data(),
                      static_cast<int>(local_count.size()),
                      MPI_UNSIGNED_LONG_LONG,
                      MPI_SUM,
                      MPI_COMM_WORLD);
        for (std::size_t i = 0; i < comps.size(); ++i) {
            comps[i].sq_norm = global_norm[i];
            comps[i].sum = global_sum[i];
            comps[i].count = static_cast<std::uint64_t>(global_count[i]);
        }
    }
#endif

    if (mpiRank() != 0) {
        return;
    }

    std::ostringstream oss;
    oss << "NewtonSolver: " << label << " component norms";
    for (const auto& c : comps) {
        const double mean = (c.count > 0u) ? (c.sum / static_cast<double>(c.count)) : 0.0;
        oss << " [" << c.label
            << " norm=" << std::sqrt(std::max(0.0, c.sq_norm))
            << " mean=" << mean << "]";
    }
    FE_LOG_INFO(oss.str());
}

void logVectorTopEntries(const systems::FESystem& sys,
                         backends::GenericVector& vec,
                         std::string_view label,
                         std::size_t top_k)
{
    const auto owned_dofs = sys.dofHandler().getPartition().locallyOwned().toVector();
    if (owned_dofs.empty() || top_k == 0u) {
        return;
    }

    auto view = vec.createAssemblyView();
    FE_CHECK_NOT_NULL(view.get(), "NewtonSolver: vector top-entry view");

    struct Entry {
        GlobalIndex dof{INVALID_GLOBAL_INDEX};
        double value{0.0};
    };

    std::vector<Entry> top_entries;
    top_entries.reserve(top_k);
    const auto maybe_insert = [&](GlobalIndex dof, double value) {
        const double abs_value = std::abs(value);
        if (!(abs_value > 0.0) || !std::isfinite(abs_value)) {
            return;
        }
        if (top_entries.size() < top_k) {
            top_entries.push_back(Entry{dof, value});
            return;
        }
        auto min_it = std::min_element(
            top_entries.begin(), top_entries.end(),
            [](const Entry& a, const Entry& b) { return std::abs(a.value) < std::abs(b.value); });
        if (min_it != top_entries.end() && abs_value > std::abs(min_it->value)) {
            *min_it = Entry{dof, value};
        }
    };

    for (const auto dof : owned_dofs) {
        maybe_insert(dof, static_cast<double>(view->getVectorEntry(dof)));
    }

    std::sort(top_entries.begin(), top_entries.end(),
              [](const Entry& a, const Entry& b) { return std::abs(a.value) > std::abs(b.value); });

    if (mpiRank() != 0) {
        return;
    }

    std::ostringstream oss;
    oss << "NewtonSolver: " << label << " top entries";
    for (const auto& e : top_entries) {
        oss << " [" << describeFieldComponentDof(sys, e.dof)
            << " value=" << e.value << "]";
    }
    FE_LOG_INFO(oss.str());
}

void normalizeFsilsPostSolveIncrementIfNeeded(backends::GenericVector& vec)
{
#if defined(FE_HAS_FSILS)
    if (vec.backendKind() != backends::BackendKind::FSILS) {
        return;
    }

    auto* fs = dynamic_cast<backends::FsilsVector*>(&vec);
    if (fs == nullptr) {
        return;
    }

    const auto mode = fsilsPostSolveSyncMode();
    if (mode == FsilsPostSolveSyncMode::Off) {
        return;
    }

    switch (mode) {
        case FsilsPostSolveSyncMode::Off:
            break;
        case FsilsPostSolveSyncMode::UpdateGhosts:
            fs->updateGhosts();
            break;
        case FsilsPostSolveSyncMode::AccumulateOverlap:
            fs->accumulateOverlap();
            break;
        case FsilsPostSolveSyncMode::AccumulateThenUpdateGhosts:
            fs->accumulateOverlap();
            fs->updateGhosts();
            break;
    }

    if (oopTraceEnabled()) {
        std::string mode_name = "off";
        switch (mode) {
            case FsilsPostSolveSyncMode::Off:
                mode_name = "off";
                break;
            case FsilsPostSolveSyncMode::UpdateGhosts:
                mode_name = "update";
                break;
            case FsilsPostSolveSyncMode::AccumulateOverlap:
                mode_name = "accumulate";
                break;
            case FsilsPostSolveSyncMode::AccumulateThenUpdateGhosts:
                mode_name = "both";
                break;
        }
        traceLog("NewtonSolver: applied FSILS post-solve increment sync mode='" + mode_name + "'");
    }
#else
    (void)vec;
#endif
}

void addRankOneOperatorMatvec(std::span<const backends::RankOneUpdate> updates,
                              backends::GenericVector& x,
                              backends::GenericVector& y)
{
    if (updates.empty()) {
        return;
    }

    auto x_view = x.createAssemblyView();
    auto y_view = y.createAssemblyView();
    FE_CHECK_NOT_NULL(x_view.get(), "NewtonSolver: rank-one x view");
    FE_CHECK_NOT_NULL(y_view.get(), "NewtonSolver: rank-one y view");

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
    if (mpi_initialized) {
        std::vector<Real> global_dots(dots.size(), Real(0.0));
        MPI_Allreduce(dots.data(),
                      global_dots.data(),
                      static_cast<int>(dots.size()),
                      MPI_DOUBLE,
                      MPI_SUM,
                      MPI_COMM_WORLD);
        dots.swap(global_dots);
    }
#endif

    y_view->beginAssemblyPhase();
    for (std::size_t u = 0; u < updates.size(); ++u) {
        const Real scale = updates[u].sigma * dots[u];
        if (std::abs(scale) <= Real(1e-30)) {
            continue;
        }
        for (const auto& [dof, val] : updates[u].v) {
            y_view->addVectorEntry(dof, scale * val, assembly::AddMode::Add);
        }
    }
    y_view->finalizeAssembly();
}

void addReducedFieldOperatorMatvec(std::span<const backends::ReducedFieldUpdate> updates,
                                   backends::GenericVector& x,
                                   backends::GenericVector& y)
{
    if (updates.empty()) {
        return;
    }

    auto x_view = x.createAssemblyView();
    auto y_view = y.createAssemblyView();
    FE_CHECK_NOT_NULL(x_view.get(), "NewtonSolver: reduced-update x view");
    FE_CHECK_NOT_NULL(y_view.get(), "NewtonSolver: reduced-update y view");

    std::vector<Real> dots(updates.size(), Real(0.0));
    for (std::size_t u = 0; u < updates.size(); ++u) {
        Real local_dot = Real(0.0);
        for (const auto& [dof, val] : updates[u].right) {
            local_dot += val * x_view->getVectorEntry(dof);
        }
        dots[u] = local_dot;
    }

#if FE_HAS_MPI
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized) {
        std::vector<Real> global_dots(dots.size(), Real(0.0));
        MPI_Allreduce(dots.data(),
                      global_dots.data(),
                      static_cast<int>(dots.size()),
                      MPI_DOUBLE,
                      MPI_SUM,
                      MPI_COMM_WORLD);
        dots.swap(global_dots);
    }
#endif

    y_view->beginAssemblyPhase();
    for (std::size_t u = 0; u < updates.size(); ++u) {
        const Real scale = updates[u].sigma * dots[u];
        if (std::abs(scale) <= Real(1e-30)) {
            continue;
        }
        for (const auto& [dof, val] : updates[u].left) {
            y_view->addVectorEntry(dof, scale * val, assembly::AddMode::Add);
        }
    }
    y_view->finalizeAssembly();
}

bool tryPromoteExactReducedUpdateToNativeRankOne(
    const backends::ReducedFieldUpdate& update,
    backends::RankOneUpdate& promoted,
    Real rel_residual_sq_limit = Real(1e-24))
{
    if (!nativeFaceRankOnePromotionEnabled()) {
        return false;
    }

    if (std::abs(update.sigma) <= Real(1e-30)) {
        return false;
    }

    std::unordered_map<GlobalIndex, Real> left_map;
    std::unordered_map<GlobalIndex, Real> right_map;
    left_map.reserve(update.left.size());
    right_map.reserve(update.right.size());

    Real left_norm_sq = Real(0.0);
    Real right_norm_sq = Real(0.0);
    Real cross = Real(0.0);

    for (const auto& [dof, value] : update.left) {
        if (std::abs(value) <= Real(1e-30)) {
            continue;
        }
        left_map[dof] += value;
    }
    for (const auto& [dof, value] : update.right) {
        if (std::abs(value) <= Real(1e-30)) {
            continue;
        }
        right_map[dof] += value;
    }
    const int global_left_has = mpiAllreduceSumIfActive(left_map.empty() ? 0 : 1);
    const int global_right_has = mpiAllreduceSumIfActive(right_map.empty() ? 0 : 1);
    if (global_left_has == 0 || global_right_has == 0) {
        return false;
    }

    for (const auto& [dof, value] : left_map) {
        left_norm_sq += value * value;
        const auto it = right_map.find(dof);
        if (it != right_map.end()) {
            cross += value * it->second;
        }
    }
    for (const auto& [dof, value] : right_map) {
        right_norm_sq += value * value;
    }

    const Real global_left_norm_sq = mpiAllreduceSumIfActive(left_norm_sq);
    const Real global_right_norm_sq = mpiAllreduceSumIfActive(right_norm_sq);
    const Real global_cross = mpiAllreduceSumIfActive(cross);

    if (!(global_left_norm_sq > Real(1e-30)) || !(global_right_norm_sq > Real(1e-30))) {
        return false;
    }

    const Real alpha = global_cross / global_left_norm_sq;
    Real local_residual_sq = Real(0.0);
    for (const auto& [dof, value] : right_map) {
        const auto it = left_map.find(dof);
        const Real left_value = (it != left_map.end()) ? it->second : Real(0.0);
        const Real diff = value - alpha * left_value;
        local_residual_sq += diff * diff;
    }
    for (const auto& [dof, value] : left_map) {
        if (right_map.contains(dof)) {
            continue;
        }
        const Real diff = alpha * value;
        local_residual_sq += diff * diff;
    }

    const Real residual_sq = mpiAllreduceSumIfActive(local_residual_sq);
    const Real rel_residual_sq = residual_sq / std::max(global_right_norm_sq, Real(1e-30));
    if (!(rel_residual_sq <= rel_residual_sq_limit) || !(std::abs(alpha) > Real(1e-30))) {
        return false;
    }

    promoted.sigma = update.sigma * alpha;
    promoted.v.clear();
    promoted.v.reserve(left_map.size());
    for (const auto& [dof, value] : left_map) {
        promoted.v.emplace_back(dof, value);
    }
    promoted.active_components = update.active_components;
    promoted.prefer_native_face = true;
    return true;
}

[[nodiscard]] std::vector<std::pair<GlobalIndex, Real>> reconstructInputGradientFromCt(
    const std::vector<Real>& ct,
    std::size_t n_field_dofs,
    std::span<const std::size_t> aux_local_indices,
    const std::vector<Real>& dF_dinputs,
    int n_inputs,
    int input_col)
{
    constexpr Real kDirectTol = static_cast<Real>(1e-14);
    if (n_field_dofs == 0 || aux_local_indices.empty() || n_inputs <= 0 || input_col < 0 ||
        dF_dinputs.size() <
            aux_local_indices.size() * static_cast<std::size_t>(n_inputs)) {
        return {};
    }

    Real denom = Real(0.0);
    std::vector<Real> numer(n_field_dofs, Real(0.0));

    for (std::size_t i = 0; i < aux_local_indices.size(); ++i) {
        const Real dF_dI = dF_dinputs[i * static_cast<std::size_t>(n_inputs) +
                                      static_cast<std::size_t>(input_col)];
        if (std::abs(dF_dI) <= kDirectTol) {
            continue;
        }
        denom += dF_dI * dF_dI;

        const auto row = aux_local_indices[i];
        const auto row_offset = row * n_field_dofs;
        if (row_offset + n_field_dofs > ct.size()) {
            return {};
        }
        for (std::size_t k = 0; k < n_field_dofs; ++k) {
            numer[k] += dF_dI * ct[row_offset + k];
        }
    }

    if (!(denom > kDirectTol * kDirectTol)) {
        return {};
    }

    std::vector<std::pair<GlobalIndex, Real>> q_u;
    q_u.reserve(n_field_dofs);
    for (std::size_t k = 0; k < n_field_dofs; ++k) {
        const Real val = numer[k] / denom;
        if (std::abs(val) > kDirectTol) {
            q_u.emplace_back(static_cast<GlobalIndex>(k), val);
        }
    }
    return q_u;
}

struct DirectCoupledCtProjection {
    std::vector<Real> values{};
    std::vector<bool> row_covered{};
};

struct DirectCoupledCtRows {
    std::vector<std::vector<std::pair<GlobalIndex, Real>>> rows{};
    std::vector<bool> row_covered{};
};

[[nodiscard]] int inferDirectCouplingRecordInputCount(
    const systems::FESystem::BorderedCouplingData::DirectCouplingRecord& record)
{
    if (!record.input_gradients.empty()) {
        return static_cast<int>(record.input_gradients.size());
    }
    if (!record.dO_dI.empty()) {
        return static_cast<int>(record.dO_dI.size());
    }
    if (!record.aux_local_indices.empty() &&
        !record.dF_dinputs.empty() &&
        record.dF_dinputs.size() % record.aux_local_indices.size() == 0) {
        return static_cast<int>(record.dF_dinputs.size() / record.aux_local_indices.size());
    }
    return 0;
}

[[nodiscard]] DirectCoupledCtRows buildDirectCouplingCtRows(
    const systems::FESystem::BorderedCouplingData& bordered,
    const dofs::IndexSet* owned_dofs = nullptr)
{
    constexpr Real kDirectTol = static_cast<Real>(1e-14);

    DirectCoupledCtRows out;
    const auto na = static_cast<std::size_t>(bordered.n_aux);
    out.rows.resize(na);
    out.row_covered.assign(na, false);

    if (!bordered.active || bordered.direct_coupling_records.empty()) {
        return out;
    }

    std::vector<std::unordered_map<GlobalIndex, Real>> row_accum(na);
    std::vector<bool> row_has_exact_contribution(na, false);
    std::vector<bool> row_has_incomplete_contribution(na, false);

    for (const auto& record : bordered.direct_coupling_records) {
        if (record.aux_local_indices.empty() || record.dF_dinputs.empty()) {
            continue;
        }

        const int n_inputs = inferDirectCouplingRecordInputCount(record);
        if (n_inputs <= 0 ||
            record.dF_dinputs.size() <
                record.aux_local_indices.size() * static_cast<std::size_t>(n_inputs)) {
            continue;
        }

        for (std::size_t local_row = 0; local_row < record.aux_local_indices.size(); ++local_row) {
            const auto row = record.aux_local_indices[local_row];
            if (row >= na) {
                continue;
            }

            std::unordered_map<GlobalIndex, Real> local_row_entries;
            bool row_fully_covered = true;
            bool row_has_nonzero_input_sensitivity = false;
            for (int input_col = 0; input_col < n_inputs; ++input_col) {
                const Real dF_dI =
                    record.dF_dinputs[local_row * static_cast<std::size_t>(n_inputs) +
                                      static_cast<std::size_t>(input_col)];
                if (std::abs(dF_dI) <= kDirectTol) {
                    continue;
                }

                row_has_nonzero_input_sensitivity = true;
                if (static_cast<std::size_t>(input_col) >= record.input_gradients.size() ||
                    record.input_gradients[static_cast<std::size_t>(input_col)].empty()) {
                    row_fully_covered = false;
                    break;
                }

                for (const auto& [dof, qj] :
                     record.input_gradients[static_cast<std::size_t>(input_col)]) {
                    if (dof < 0) {
                        continue;
                    }
                    if (owned_dofs != nullptr && !owned_dofs->contains(dof)) {
                        continue;
                    }
                    const Real value = dF_dI * qj;
                    if (std::abs(value) <= kDirectTol) {
                        continue;
                    }
                    local_row_entries[dof] += value;
                }
            }

            if (!row_has_nonzero_input_sensitivity) {
                continue;
            }
            if (!row_fully_covered) {
                row_has_incomplete_contribution[row] = true;
                continue;
            }

            row_has_exact_contribution[row] = true;
            auto& accum = row_accum[row];
            for (const auto& [dof, value] : local_row_entries) {
                accum[dof] += value;
            }
        }
    }

    for (std::size_t row = 0; row < na; ++row) {
        if (!row_has_exact_contribution[row] || row_has_incomplete_contribution[row]) {
            continue;
        }

        auto& dense_row = out.rows[row];
        dense_row.reserve(row_accum[row].size());
        for (const auto& [dof, value] : row_accum[row]) {
            if (std::abs(value) > kDirectTol) {
                dense_row.emplace_back(dof, value);
            }
        }
        std::sort(dense_row.begin(), dense_row.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        out.row_covered[row] = true;
    }

    return out;
}

[[nodiscard]] DirectCoupledCtProjection projectCtDuFromDirectCouplingRecords(
    const systems::FESystem::BorderedCouplingData& bordered,
    std::span<const Real> dense_du)
{
    constexpr Real kDirectTol = static_cast<Real>(1e-14);

    DirectCoupledCtProjection out;
    const auto na = static_cast<std::size_t>(bordered.n_aux);
    out.values.assign(na, Real(0.0));
    out.row_covered.assign(na, false);

    if (!bordered.active ||
        dense_du.empty() ||
        bordered.direct_coupling_records.empty()) {
        return out;
    }

    for (const auto& record : bordered.direct_coupling_records) {
        if (record.aux_local_indices.empty() || record.dF_dinputs.empty()) {
            continue;
        }

        const int n_inputs = inferDirectCouplingRecordInputCount(record);
        if (n_inputs <= 0 ||
            record.dF_dinputs.size() <
                record.aux_local_indices.size() * static_cast<std::size_t>(n_inputs)) {
            continue;
        }

        std::vector<Real> input_projections(static_cast<std::size_t>(n_inputs), Real(0.0));
        std::vector<bool> have_exact_input_projection(static_cast<std::size_t>(n_inputs), false);
        for (int input_col = 0; input_col < n_inputs; ++input_col) {
            if (static_cast<std::size_t>(input_col) >= record.input_gradients.size()) {
                continue;
            }
            const auto& q_u = record.input_gradients[static_cast<std::size_t>(input_col)];
            if (q_u.empty()) {
                continue;
            }

            Real proj = Real(0.0);
            for (const auto& [dof, qj] : q_u) {
                if (dof < 0) {
                    continue;
                }
                const auto dof_idx = static_cast<std::size_t>(dof);
                if (dof_idx >= dense_du.size()) {
                    continue;
                }
                proj += qj * dense_du[dof_idx];
            }
            input_projections[static_cast<std::size_t>(input_col)] = proj;
            have_exact_input_projection[static_cast<std::size_t>(input_col)] = true;
        }

        for (std::size_t local_row = 0; local_row < record.aux_local_indices.size(); ++local_row) {
            const auto row = record.aux_local_indices[local_row];
            if (row >= na) {
                continue;
            }

            Real row_value = Real(0.0);
            bool row_fully_covered = true;
            bool row_has_nonzero_input_sensitivity = false;
            for (int input_col = 0; input_col < n_inputs; ++input_col) {
                const Real dF_dI =
                    record.dF_dinputs[local_row * static_cast<std::size_t>(n_inputs) +
                                      static_cast<std::size_t>(input_col)];
                if (std::abs(dF_dI) <= kDirectTol) {
                    continue;
                }
                row_has_nonzero_input_sensitivity = true;
                if (!have_exact_input_projection[static_cast<std::size_t>(input_col)]) {
                    row_fully_covered = false;
                    break;
                }
                row_value += dF_dI * input_projections[static_cast<std::size_t>(input_col)];
            }

            if (!row_has_nonzero_input_sensitivity || !row_fully_covered) {
                continue;
            }

            out.values[row] += row_value;
            out.row_covered[row] = true;
        }
    }

    return out;
}

[[nodiscard]] bool tryPromoteDirectCouplingRecordToNativeRankOne(
    const systems::FESystem::BorderedCouplingData& bordered,
    const systems::FESystem::BorderedCouplingData::DirectCouplingRecord& record,
    std::size_t aux_local_index,
    std::span<const Real> left_column,
    const dofs::IndexSet& owned_dofs,
    backends::RankOneUpdate& promoted)
{
    if (!nativeFaceRankOnePromotionEnabled()) {
        return false;
    }

    constexpr Real kDirectTol = static_cast<Real>(1e-14);
    const auto it = std::find(record.aux_local_indices.begin(),
                              record.aux_local_indices.end(),
                              aux_local_index);
    if (it == record.aux_local_indices.end() || left_column.empty()) {
        return false;
    }

    const int n_inputs = static_cast<int>(record.dO_dI.size());
    if (n_inputs <= 0 || record.dF_dinputs.empty()) {
        return false;
    }

    int active_input_col = -1;
    for (int input_col = 0; input_col < n_inputs; ++input_col) {
        if (std::abs(record.dO_dI[static_cast<std::size_t>(input_col)]) <= kDirectTol) {
            continue;
        }
        if (active_input_col >= 0) {
            return false;
        }
        active_input_col = input_col;
    }
    if (active_input_col < 0) {
        return false;
    }

    const Real dOk_dIm = record.dO_dI[static_cast<std::size_t>(active_input_col)];
    constexpr Real kSymTolSq = static_cast<Real>(1e-4);

    struct PromotionCandidate {
        std::vector<std::pair<GlobalIndex, Real>> q_u{};
        Real sigma{Real(0.0)};
        Real rel_residual_sq{std::numeric_limits<Real>::infinity()};
        bool valid{false};
    };

    auto evaluate_candidate =
        [&](std::vector<std::pair<GlobalIndex, Real>> q_u) -> PromotionCandidate {
            PromotionCandidate result;
            if (q_u.empty()) {
                return result;
            }

            std::unordered_map<GlobalIndex, Real> q_map;
            q_map.reserve(q_u.size());
            Real q_norm_sq = Real(0.0);
            for (const auto& [dof, value] : q_u) {
                q_map[dof] += value;
                q_norm_sq += value * value;
            }
            if (!(q_norm_sq > Real(1e-30))) {
                return result;
            }

            Real cross = Real(0.0);
            Real dRdQ_norm_sq = Real(0.0);
            Real residual_sq = Real(0.0);

            if (!record.output_gradient.empty()) {
                for (const auto& [dof, dRi_dOk] : record.output_gradient) {
                    const Real dRdQ = dRi_dOk * dOk_dIm;
                    dRdQ_norm_sq += dRdQ * dRdQ;
                    const auto it_q = q_map.find(dof);
                    if (it_q != q_map.end()) {
                        cross += dRdQ * it_q->second;
                    }
                }
            } else {
                for (std::size_t k = 0; k < left_column.size(); ++k) {
                    const Real dRdQ = left_column[k] * dOk_dIm;
                    dRdQ_norm_sq += dRdQ * dRdQ;
                    const auto it_q = q_map.find(static_cast<GlobalIndex>(k));
                    if (it_q != q_map.end()) {
                        cross += dRdQ * it_q->second;
                    }
                }
            }
            if (!(dRdQ_norm_sq > Real(1e-30))) {
                return result;
            }

            const Real sigma = cross / q_norm_sq;
            if (!(std::abs(sigma) > Real(1e-30))) {
                return result;
            }

            if (!record.output_gradient.empty()) {
                for (const auto& [dof, dRi_dOk] : record.output_gradient) {
                    const Real dRdQ = dRi_dOk * dOk_dIm;
                    const auto it_q = q_map.find(dof);
                    const Real q_val = (it_q != q_map.end()) ? it_q->second : Real(0.0);
                    const Real diff = dRdQ - sigma * q_val;
                    residual_sq += diff * diff;
                }
                for (const auto& [dof, q_val] : q_map) {
                    const auto dof_value = dof;
                    const auto it =
                        std::find_if(record.output_gradient.begin(),
                                     record.output_gradient.end(),
                                     [dof_value](const auto& entry) {
                                         return entry.first == dof_value;
                                     });
                    if (it == record.output_gradient.end()) {
                        const Real diff = sigma * q_val;
                        residual_sq += diff * diff;
                    }
                }
            } else {
                for (std::size_t k = 0; k < left_column.size(); ++k) {
                    const Real dRdQ = left_column[k] * dOk_dIm;
                    const auto it_q = q_map.find(static_cast<GlobalIndex>(k));
                    const Real q_val = (it_q != q_map.end()) ? it_q->second : Real(0.0);
                    const Real diff = dRdQ - sigma * q_val;
                    residual_sq += diff * diff;
                }
            }

            result.q_u = std::move(q_u);
            result.sigma = sigma;
            result.rel_residual_sq = residual_sq / std::max(dRdQ_norm_sq, Real(1e-30));
            result.valid = true;
            return result;
        };

    PromotionCandidate best;
    if (static_cast<std::size_t>(active_input_col) < record.input_gradients.size() &&
        !record.input_gradients[static_cast<std::size_t>(active_input_col)].empty()) {
        best = evaluate_candidate(
            record.input_gradients[static_cast<std::size_t>(active_input_col)]);
    }

    auto q_u_from_ct = reconstructInputGradientFromCt(
        bordered.Ct,
        bordered.n_field_dofs,
        std::span<const std::size_t>(
            record.aux_local_indices.data(),
            record.aux_local_indices.size()),
        record.dF_dinputs,
        n_inputs,
        active_input_col);
    if (!q_u_from_ct.empty()) {
        auto candidate = evaluate_candidate(std::move(q_u_from_ct));
        if (candidate.valid &&
            (!best.valid || candidate.rel_residual_sq < best.rel_residual_sq)) {
            best = std::move(candidate);
        }
    }

    if (!best.valid || !(best.rel_residual_sq <= kSymTolSq)) {
        return false;
    }

    promoted = {};
    promoted.sigma = best.sigma;
    promoted.prefer_native_face = true;
    promoted.v.reserve(best.q_u.size());
    for (const auto& [dof, value] : best.q_u) {
        if (owned_dofs.contains(dof)) {
            promoted.v.emplace_back(dof, value);
        }
    }
    return true;
}

[[nodiscard]] bool tryPromoteAlgebraicDirectCouplingRecordToNativeRankOne(
    const systems::FESystem::BorderedCouplingData& bordered,
    const systems::FESystem::BorderedCouplingData::DirectCouplingRecord& record,
    const std::unordered_map<std::size_t, std::size_t>& algebraic_position,
    const std::vector<Real>& Daa_inv,
    std::size_t n_alg,
    const dofs::IndexSet& owned_dofs,
    backends::RankOneUpdate& promoted)
{
    if (!nativeFaceRankOnePromotionEnabled()) {
        return false;
    }

    constexpr Real kDirectTol = static_cast<Real>(1e-14);
    if (record.output_gradient.empty() || record.aux_local_indices.empty() || n_alg == 0 ||
        Daa_inv.size() != n_alg * n_alg) {
        return false;
    }

    const std::size_t n_local_aux = record.aux_local_indices.size();
    int n_inputs = 0;
    if (!record.dO_dI.empty()) {
        n_inputs = static_cast<int>(record.dO_dI.size());
    } else if (!record.input_gradients.empty()) {
        n_inputs = static_cast<int>(record.input_gradients.size());
    } else if (!record.dF_dinputs.empty() && n_local_aux > 0 &&
               record.dF_dinputs.size() % n_local_aux == 0) {
        n_inputs = static_cast<int>(record.dF_dinputs.size() / n_local_aux);
    }
    if (n_inputs <= 0 || record.dF_dinputs.size() < n_local_aux * static_cast<std::size_t>(n_inputs)) {
        if (oopTraceEnabled()) {
            std::ostringstream oss;
            oss << "NewtonSolver: algebraic direct promotion skipped"
                << " output_slot=" << record.output_slot
                << " n_local_aux=" << n_local_aux
                << " n_inputs=" << n_inputs
                << " dF_dinputs=" << record.dF_dinputs.size();
            traceLog(oss.str());
        }
        return false;
    }

    std::vector<Real> effective_dO_dI(static_cast<std::size_t>(n_inputs), Real(0.0));
    if (!record.dO_dI.empty()) {
        const auto count = std::min<std::size_t>(effective_dO_dI.size(), record.dO_dI.size());
        std::copy_n(record.dO_dI.begin(), static_cast<std::ptrdiff_t>(count), effective_dO_dI.begin());
    }

    const bool have_output_state_sensitivity =
        record.dO_dx.size() >= n_local_aux && !record.dF_dinputs.empty();
    if (have_output_state_sensitivity) {
        for (std::size_t i_local = 0; i_local < n_local_aux; ++i_local) {
            const Real dOk_dxi = record.dO_dx[i_local];
            if (std::abs(dOk_dxi) <= kDirectTol) {
                continue;
            }
            const auto pos_i_it = algebraic_position.find(record.aux_local_indices[i_local]);
            if (pos_i_it == algebraic_position.end()) {
                continue;
            }
            const auto pos_i = pos_i_it->second;
            for (std::size_t j_local = 0; j_local < n_local_aux; ++j_local) {
                const auto pos_j_it = algebraic_position.find(record.aux_local_indices[j_local]);
                if (pos_j_it == algebraic_position.end()) {
                    continue;
                }
                const auto pos_j = pos_j_it->second;
                const Real dxi_dFj = Daa_inv[pos_i * n_alg + pos_j];
                if (std::abs(dxi_dFj) <= kDirectTol) {
                    continue;
                }
                for (int input_col = 0; input_col < n_inputs; ++input_col) {
                    const Real dFj_dIm =
                        record.dF_dinputs[j_local * static_cast<std::size_t>(n_inputs) +
                                          static_cast<std::size_t>(input_col)];
                    if (std::abs(dFj_dIm) <= kDirectTol) {
                        continue;
                    }
                    // Eliminate algebraic auxiliary unknowns exactly:
                    // dO/dI_eff = dO/dI - dO/dx * D^{-1} * dF/dI.
                    effective_dO_dI[static_cast<std::size_t>(input_col)] -=
                        dOk_dxi * dxi_dFj * dFj_dIm;
                }
            }
        }
    }

    int active_input_col = -1;
    for (int input_col = 0; input_col < n_inputs; ++input_col) {
        const Real dOk_dIm = effective_dO_dI[static_cast<std::size_t>(input_col)];
        if (std::abs(dOk_dIm) <= kDirectTol) {
            continue;
        }
        if (active_input_col >= 0) {
            return false;
        }
        active_input_col = input_col;
    }
    if (active_input_col < 0) {
        if (oopTraceEnabled()) {
            std::ostringstream oss;
            oss << "NewtonSolver: algebraic direct promotion no active input"
                << " output_slot=" << record.output_slot
                << " eff_dO_dI=[";
            for (int i = 0; i < n_inputs; ++i) {
                if (i != 0) {
                    oss << ", ";
                }
                oss << effective_dO_dI[static_cast<std::size_t>(i)];
            }
            oss << "]";
            traceLog(oss.str());
        }
        return false;
    }

    constexpr Real kSymTolSq = static_cast<Real>(1e-4);

    struct PromotionCandidate {
        std::vector<std::pair<GlobalIndex, Real>> q_u{};
        Real sigma{Real(0.0)};
        Real rel_residual_sq{std::numeric_limits<Real>::infinity()};
        bool valid{false};
    };

    auto evaluate_candidate =
        [&](std::vector<std::pair<GlobalIndex, Real>> q_u,
            Real dOk_dIm) -> PromotionCandidate {
            PromotionCandidate result;
            if (q_u.empty() || std::abs(dOk_dIm) <= kDirectTol) {
                return result;
            }

            std::unordered_map<GlobalIndex, Real> q_map;
            q_map.reserve(q_u.size());
            Real q_norm_sq = Real(0.0);
            for (const auto& [dof, value] : q_u) {
                q_map[dof] += value;
                q_norm_sq += value * value;
            }
            if (!(q_norm_sq > Real(1e-30))) {
                return result;
            }

            Real cross = Real(0.0);
            Real dRdQ_norm_sq = Real(0.0);
            Real residual_sq = Real(0.0);
            for (const auto& [dof, dRi_dOk] : record.output_gradient) {
                const Real dRdQ = dRi_dOk * dOk_dIm;
                dRdQ_norm_sq += dRdQ * dRdQ;
                const auto it_q = q_map.find(dof);
                if (it_q != q_map.end()) {
                    cross += dRdQ * it_q->second;
                }
            }
            if (!(dRdQ_norm_sq > Real(1e-30))) {
                return result;
            }

            const Real sigma = cross / q_norm_sq;
            if (!(std::abs(sigma) > Real(1e-30))) {
                return result;
            }

            for (const auto& [dof, dRi_dOk] : record.output_gradient) {
                const Real dRdQ = dRi_dOk * dOk_dIm;
                const auto it_q = q_map.find(dof);
                const Real q_val = (it_q != q_map.end()) ? it_q->second : Real(0.0);
                const Real diff = dRdQ - sigma * q_val;
                residual_sq += diff * diff;
            }
            for (const auto& [dof, q_val] : q_map) {
                const auto dof_value = dof;
                const auto it =
                    std::find_if(record.output_gradient.begin(),
                                 record.output_gradient.end(),
                                 [dof_value](const auto& entry) {
                                     return entry.first == dof_value;
                                 });
                if (it == record.output_gradient.end()) {
                    const Real diff = sigma * q_val;
                    residual_sq += diff * diff;
                }
            }

            result.q_u = std::move(q_u);
            result.sigma = sigma;
            result.rel_residual_sq = residual_sq / std::max(dRdQ_norm_sq, Real(1e-30));
            result.valid = true;
            return result;
        };

    PromotionCandidate best;
    const Real dOk_dIm = effective_dO_dI[static_cast<std::size_t>(active_input_col)];
    if (static_cast<std::size_t>(active_input_col) < record.input_gradients.size() &&
        !record.input_gradients[static_cast<std::size_t>(active_input_col)].empty()) {
        best = evaluate_candidate(
            record.input_gradients[static_cast<std::size_t>(active_input_col)], dOk_dIm);
    }

    auto q_u_from_ct = reconstructInputGradientFromCt(
        bordered.Ct,
        bordered.n_field_dofs,
        std::span<const std::size_t>(
            record.aux_local_indices.data(),
            record.aux_local_indices.size()),
        record.dF_dinputs,
        n_inputs,
        active_input_col);
    if (!q_u_from_ct.empty()) {
        auto candidate = evaluate_candidate(std::move(q_u_from_ct), dOk_dIm);
        if (candidate.valid &&
            (!best.valid || candidate.rel_residual_sq < best.rel_residual_sq)) {
            best = std::move(candidate);
        }
    }

    if (oopTraceEnabled()) {
        std::ostringstream oss;
        oss << "NewtonSolver: algebraic direct promotion candidate"
            << " output_slot=" << record.output_slot
            << " active_input=" << active_input_col
            << " eff_dO_dI=" << dOk_dIm
            << " best_valid=" << best.valid
            << " rel_residual_sq=" << best.rel_residual_sq
            << " q_nnz=" << best.q_u.size();
        traceLog(oss.str());
    }

    if (!best.valid || !(best.rel_residual_sq <= kSymTolSq)) {
        return false;
    }

    promoted = {};
    promoted.sigma = best.sigma;
    promoted.prefer_native_face = true;
    promoted.v.reserve(best.q_u.size());
    for (const auto& [dof, value] : best.q_u) {
        if (owned_dofs.contains(dof)) {
            promoted.v.emplace_back(dof, value);
        }
    }
    return true;
}

std::vector<backends::RankOneUpdate>
transformRankOneUpdatesForConstraints(std::span<const backends::RankOneUpdate> updates,
                                      const constraints::AffineConstraints& constraints)
{
    if (updates.empty()) {
        return {};
    }

    std::vector<backends::RankOneUpdate> transformed;
    transformed.reserve(updates.size());

    for (const auto& upd : updates) {
        backends::RankOneUpdate out;
        out.sigma = upd.sigma;
        out.active_components = upd.active_components;
        out.prefer_native_face = upd.prefer_native_face;

        std::map<GlobalIndex, Real> coeffs;
        for (const auto& [dof, value] : upd.v) {
            if (std::abs(value) <= Real(1e-30)) {
                continue;
            }

            const auto cv = constraints.getConstraint(dof);
            if (!cv) {
                coeffs[dof] += value;
                continue;
            }
            if (cv->isDirichlet()) {
                // Eliminate constrained slave DOFs from the native low-rank
                // factor. Keeping them would let the outlet correction
                // re-populate rows/columns that the constrained linear space
                // has already removed.
                continue;
            }

            for (const auto& entry : cv->entries) {
                coeffs[entry.master_dof] += value * static_cast<Real>(entry.weight);
            }
        }

        out.v.reserve(coeffs.size());
        for (const auto& [dof, value] : coeffs) {
            if (std::abs(value) > Real(1e-30)) {
                out.v.emplace_back(dof, value);
            }
        }

        transformed.push_back(std::move(out));
    }

    return transformed;
}

std::vector<backends::ReducedFieldUpdate>
transformReducedFieldUpdatesForConstraints(
    std::span<const backends::ReducedFieldUpdate> updates,
    const constraints::AffineConstraints& constraints)
{
    if (updates.empty()) {
        return {};
    }

    auto transform_factor =
        [&](std::span<const std::pair<GlobalIndex, Real>> factor)
            -> std::vector<std::pair<GlobalIndex, Real>> {
        std::map<GlobalIndex, Real> coeffs;
        for (const auto& [dof, value] : factor) {
            if (std::abs(value) <= Real(1e-30)) {
                continue;
            }

            const auto cv = constraints.getConstraint(dof);
            if (!cv) {
                coeffs[dof] += value;
                continue;
            }
            if (cv->isDirichlet()) {
                continue;
            }

            for (const auto& entry : cv->entries) {
                coeffs[entry.master_dof] += value * static_cast<Real>(entry.weight);
            }
        }

        std::vector<std::pair<GlobalIndex, Real>> out;
        out.reserve(coeffs.size());
        for (const auto& [dof, value] : coeffs) {
            if (std::abs(value) > Real(1e-30)) {
                out.emplace_back(dof, value);
            }
        }
        return out;
    };

    std::vector<backends::ReducedFieldUpdate> transformed;
    transformed.reserve(updates.size());
    for (const auto& upd : updates) {
        backends::ReducedFieldUpdate out;
        out.sigma = upd.sigma;
        out.active_components = upd.active_components;
        out.left = transform_factor(
            std::span<const std::pair<GlobalIndex, Real>>(upd.left.data(), upd.left.size()));
        out.right = transform_factor(
            std::span<const std::pair<GlobalIndex, Real>>(upd.right.data(), upd.right.size()));
        // Preserve globally active reduced-update slots even when this rank's
        // constrained projection has no owned entries for one side. The FSILS
        // backend now handles empty local factors and needs identical update
        // counts on every rank to keep overlap exchanges ordered.
        transformed.push_back(std::move(out));
    }
    return transformed;
}

struct FsilsMatrixSnapshot {
    std::vector<Real> values{};

    [[nodiscard]] bool valid() const noexcept { return !values.empty(); }
};

struct SolverOptionsGuard {
    backends::LinearSolver& linear;
    backends::SolverOptions saved;
    ~SolverOptionsGuard() noexcept
    {
        try {
            linear.setOptions(saved);
        } catch (...) {
        }
    }
};

[[nodiscard]] backends::SolverOptions makeBorderedSolveOptions(const backends::SolverOptions& base)
{
    backends::SolverOptions opts = base;

    const Real target_rel =
        (base.rel_tol > Real(0.0) && std::isfinite(static_cast<double>(base.rel_tol)))
            ? std::min(static_cast<Real>(1e-8), static_cast<Real>(base.rel_tol * Real(1e-2)))
            : static_cast<Real>(1e-8);
    opts.rel_tol = std::max(target_rel, static_cast<Real>(1e-12));

    const Real target_abs =
        (base.abs_tol > Real(0.0) && std::isfinite(static_cast<double>(base.abs_tol)))
            ? std::min(base.abs_tol, static_cast<Real>(1e-12))
            : static_cast<Real>(1e-12);
    opts.abs_tol = std::max(target_abs, static_cast<Real>(1e-16));

    // Strongly coupled bordered / native rank-one outlet solves are much less
    // tolerant of loose inner solves than the legacy PDE-only path. Keep a
    // meaningful lower bound, but do not throw away a larger user-requested
    // Krylov budget from the XML/application settings.
    opts.max_iter = std::max(base.max_iter, 200);

    // Bordered outlet-coupled solves cannot safely accept a native BlockSchur
    // "success" unless the wrapper also validates the original FE residual.
    // The internal FSILS residual can look converged while the true operator
    // residual is still too large for Newton to make progress.
    opts.fsils_residual_check_policy = backends::FsilsResidualCheckPolicy::Always;

    if (base.method == backends::SolverMethod::BlockSchur) {
        const int target_inner_max_iter = std::max(base.max_iter, 200);
        opts.fsils_blockschur_gm_max_iter =
            std::max(base.fsils_blockschur_gm_max_iter.value_or(0), target_inner_max_iter);
        opts.fsils_blockschur_cg_max_iter =
            std::max(base.fsils_blockschur_cg_max_iter.value_or(0), target_inner_max_iter);

        const Real gm_target_rel = base.fsils_blockschur_gm_rel_tol.has_value()
            ? std::min(*base.fsils_blockschur_gm_rel_tol, opts.rel_tol)
            : opts.rel_tol;
        const Real cg_target_rel = base.fsils_blockschur_cg_rel_tol.has_value()
            ? std::min(*base.fsils_blockschur_cg_rel_tol, opts.rel_tol)
            : opts.rel_tol;
        opts.fsils_blockschur_gm_rel_tol = gm_target_rel;
        opts.fsils_blockschur_cg_rel_tol = cg_target_rel;
    }

    return opts;
}

[[nodiscard]] backends::SolverOptions
makeValidatedNativeRankOneSolveOptions(const backends::SolverOptions& base,
                                       const int native_direct_face_mode_count,
                                       std::optional<Real> inner_rel_override = std::nullopt)
{
    backends::SolverOptions opts = base;

    // Native face/reduced outlet updates do need tighter inner BlockSchur
    // sub-solves than the XML 1e-3 defaults, but they do not need the much
    // harsher bordered 1e-8/200 regime. A moderate tightening restores the
    // pipe-case robustness while avoiding most of the extra Schur/momentum
    // work seen with the explicit bordered settings.
    if (base.method == backends::SolverMethod::BlockSchur) {
        (void)native_direct_face_mode_count;
        const Real rel_scale = static_cast<Real>(1e-2);
        const Real fallback_target = static_cast<Real>(1e-6);
        const Real target_inner_rel =
            (base.rel_tol > Real(0.0) && std::isfinite(static_cast<double>(base.rel_tol)))
                ? std::max(static_cast<Real>(1e-10), base.rel_tol * rel_scale)
                : fallback_target;
        const Real effective_inner_rel = inner_rel_override.has_value()
            ? std::min(target_inner_rel, *inner_rel_override)
            : target_inner_rel;
        const int target_inner_max_iter =
            inner_rel_override.has_value()
                ? std::max(base.max_iter, 200)
                : std::max(base.max_iter, 120);
        opts.fsils_blockschur_gm_max_iter =
            std::max(base.fsils_blockschur_gm_max_iter.value_or(0), target_inner_max_iter);
        opts.fsils_blockschur_cg_max_iter =
            std::max(base.fsils_blockschur_cg_max_iter.value_or(0), target_inner_max_iter);
        opts.fsils_blockschur_gm_rel_tol = base.fsils_blockschur_gm_rel_tol.has_value()
            ? std::min(*base.fsils_blockschur_gm_rel_tol, effective_inner_rel)
            : effective_inner_rel;
        opts.fsils_blockschur_cg_rel_tol = base.fsils_blockschur_cg_rel_tol.has_value()
            ? std::min(*base.fsils_blockschur_cg_rel_tol, effective_inner_rel)
            : effective_inner_rel;
    }

    opts.fsils_residual_check_policy = backends::FsilsResidualCheckPolicy::Always;
    return opts;
}

[[nodiscard]] FsilsMatrixSnapshot captureFsilsMatrixSnapshot(const backends::GenericMatrix& A)
{
#if defined(FE_HAS_FSILS)
    const auto* fs = dynamic_cast<const backends::FsilsMatrix*>(&A);
    if (!fs) {
        return {};
    }
    const auto nnz = fs->fsilsNnz();
    const auto dof = fs->fsilsDof();
    if (nnz <= 0 || dof <= 0) {
        return {};
    }

    FsilsMatrixSnapshot snap;
    const auto count = static_cast<std::size_t>(nnz) *
                       static_cast<std::size_t>(dof) *
                       static_cast<std::size_t>(dof);
    snap.values.resize(count);
    std::copy(fs->fsilsValuesPtr(), fs->fsilsValuesPtr() + count, snap.values.begin());
    return snap;
#else
    (void)A;
    return {};
#endif
}

void restoreFsilsMatrixSnapshot(backends::GenericMatrix& A, const FsilsMatrixSnapshot& snap)
{
#if defined(FE_HAS_FSILS)
    if (!snap.valid()) {
        return;
    }
    auto* fs = dynamic_cast<backends::FsilsMatrix*>(&A);
    if (!fs) {
        return;
    }
    const auto nnz = fs->fsilsNnz();
    const auto dof = fs->fsilsDof();
    if (nnz <= 0 || dof <= 0) {
        return;
    }
    const auto count = static_cast<std::size_t>(nnz) *
                       static_cast<std::size_t>(dof) *
                       static_cast<std::size_t>(dof);
    FE_CHECK_ARG(snap.values.size() == count,
                 "NewtonSolver: FSILS matrix snapshot size mismatch");
    std::copy(snap.values.begin(), snap.values.end(), fs->fsilsValuesPtr());
#else
    (void)A;
    (void)snap;
#endif
}

[[nodiscard]] bool solveDenseLinearSystem(std::vector<Real>& A,
                                          std::vector<Real>& b,
                                          Real pivot_tol = static_cast<Real>(1e-20))
{
    const auto n = b.size();
    if (A.size() != n * n) {
        return false;
    }

    for (std::size_t k = 0; k < n; ++k) {
        std::size_t pivot = k;
        Real pivot_abs = std::abs(A[k * n + k]);
        for (std::size_t i = k + 1; i < n; ++i) {
            const Real cand = std::abs(A[i * n + k]);
            if (cand > pivot_abs) {
                pivot_abs = cand;
                pivot = i;
            }
        }
        if (!(pivot_abs > pivot_tol)) {
            return false;
        }
        if (pivot != k) {
            for (std::size_t j = 0; j < n; ++j) {
                std::swap(A[k * n + j], A[pivot * n + j]);
            }
            std::swap(b[k], b[pivot]);
        }

        const Real diag = A[k * n + k];
        for (std::size_t i = k + 1; i < n; ++i) {
            const Real factor = A[i * n + k] / diag;
            if (std::abs(factor) <= pivot_tol) {
                continue;
            }
            A[i * n + k] = 0.0;
            for (std::size_t j = k + 1; j < n; ++j) {
                A[i * n + j] -= factor * A[k * n + j];
            }
            b[i] -= factor * b[k];
        }
    }

    for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(n) - 1; i >= 0; --i) {
        Real sum = b[static_cast<std::size_t>(i)];
        for (std::size_t j = static_cast<std::size_t>(i) + 1; j < n; ++j) {
            sum -= A[static_cast<std::size_t>(i) * n + j] * b[j];
        }
        const Real diag = A[static_cast<std::size_t>(i) * n + static_cast<std::size_t>(i)];
        if (!(std::abs(diag) > pivot_tol)) {
            return false;
        }
        b[static_cast<std::size_t>(i)] = sum / diag;
    }

    return true;
}

[[nodiscard]] bool invertDenseMatrix(const std::vector<Real>& A,
                                     std::size_t n,
                                     std::vector<Real>& A_inv,
                                     Real pivot_tol = static_cast<Real>(1e-20))
{
    if (A.size() != n * n) {
        return false;
    }

    A_inv.assign(n * n, Real(0.0));
    std::vector<Real> system = A;
    std::vector<Real> rhs(n, Real(0.0));
    for (std::size_t col = 0; col < n; ++col) {
        std::fill(rhs.begin(), rhs.end(), Real(0.0));
        rhs[col] = Real(1.0);
        auto work = system;
        auto x = rhs;
        if (!solveDenseLinearSystem(work, x, pivot_tol)) {
            A_inv.clear();
            return false;
        }
        for (std::size_t row = 0; row < n; ++row) {
            A_inv[row * n + col] = x[row];
        }
    }
    return true;
}

struct AlgebraicAuxiliaryReduction {
    bool active{false};
    systems::FESystem::BorderedCouplingData reduced_bordered{};
    std::vector<std::size_t> algebraic_indices{};
    std::vector<std::size_t> dynamic_indices{};
    std::vector<Real> Daa_inv{};
    std::vector<Real> Daa_inv_Ca_field{};
    std::vector<Real> Daa_inv_Dad{};
    std::vector<Real> Daa_inv_ga{};
    std::vector<Real> rhs_shift{};
    std::vector<backends::RankOneUpdate> promoted_rank_one_updates{};
    std::vector<backends::ReducedFieldUpdate> reduced_field_updates{};
    std::vector<backends::GroupedBorderedFieldCoupling> grouped_couplings{};
};

[[nodiscard]] bool denseMatrixIsEffectivelyDiagonal(const std::vector<Real>& A,
                                                    std::size_t n,
                                                    Real tol = static_cast<Real>(1e-14))
{
    if (A.size() != n * n) {
        return false;
    }
    Real diag_scale = Real(0.0);
    for (std::size_t i = 0; i < n; ++i) {
        diag_scale = std::max(diag_scale, std::abs(A[i * n + i]));
    }
    diag_scale = std::max(diag_scale, Real(1.0));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (i == j) {
                continue;
            }
            if (std::abs(A[i * n + j]) > tol * diag_scale) {
                return false;
            }
        }
    }
    return true;
}

[[nodiscard]] std::vector<Real> denseSubmatrixRowsCols(const std::vector<Real>& A,
                                                       std::size_t n_rows,
                                                       std::size_t n_cols,
                                                       std::span<const std::size_t> rows,
                                                       std::span<const std::size_t> cols)
{
    std::vector<Real> out(rows.size() * cols.size(), Real(0.0));
    for (std::size_t i = 0; i < rows.size(); ++i) {
        for (std::size_t j = 0; j < cols.size(); ++j) {
            out[i * cols.size() + j] = A[rows[i] * n_cols + cols[j]];
        }
    }
    return out;
}

[[nodiscard]] std::vector<Real> denseSubmatrixRows(const std::vector<Real>& A,
                                                   std::size_t n_cols,
                                                   std::span<const std::size_t> rows)
{
    std::vector<Real> out(rows.size() * n_cols, Real(0.0));
    for (std::size_t i = 0; i < rows.size(); ++i) {
        for (std::size_t j = 0; j < n_cols; ++j) {
            out[i * n_cols + j] = A[rows[i] * n_cols + j];
        }
    }
    return out;
}

[[nodiscard]] std::vector<Real> denseSubmatrixColumns(const std::vector<Real>& A_col_major,
                                                      std::size_t n_rows,
                                                      std::span<const std::size_t> cols)
{
    std::vector<Real> out(n_rows * cols.size(), Real(0.0));
    for (std::size_t j = 0; j < cols.size(); ++j) {
        const auto src_col = cols[j];
        for (std::size_t i = 0; i < n_rows; ++i) {
            out[i + n_rows * j] = A_col_major[i + n_rows * src_col];
        }
    }
    return out;
}

[[nodiscard]] std::vector<Real> denseMatMulRowMajor(const std::vector<Real>& A,
                                                    std::size_t m,
                                                    std::size_t k,
                                                    const std::vector<Real>& B,
                                                    std::size_t n)
{
    FE_THROW_IF(A.size() != m * k || B.size() != k * n,
                InvalidArgumentException,
                "NewtonSolver: denseMatMulRowMajor dimension mismatch");
    std::vector<Real> C(m * n, Real(0.0));
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t p = 0; p < k; ++p) {
            const Real a = A[i * k + p];
            if (std::abs(a) <= Real(1e-30)) {
                continue;
            }
            for (std::size_t j = 0; j < n; ++j) {
                C[i * n + j] += a * B[p * n + j];
            }
        }
    }
    return C;
}

[[nodiscard]] std::vector<Real> denseColMajorTimesRowMajor(const std::vector<Real>& A_col_major,
                                                           std::size_t n_rows,
                                                           std::size_t k,
                                                           const std::vector<Real>& B_row_major,
                                                           std::size_t n_cols)
{
    FE_THROW_IF(A_col_major.size() != n_rows * k || B_row_major.size() != k * n_cols,
                InvalidArgumentException,
                "NewtonSolver: denseColMajorTimesRowMajor dimension mismatch");
    std::vector<Real> C(n_rows * n_cols, Real(0.0));
    for (std::size_t p = 0; p < k; ++p) {
        for (std::size_t i = 0; i < n_rows; ++i) {
            const Real a = A_col_major[i + n_rows * p];
            if (std::abs(a) <= Real(1e-30)) {
                continue;
            }
            for (std::size_t j = 0; j < n_cols; ++j) {
                C[i * n_cols + j] += a * B_row_major[p * n_cols + j];
            }
        }
    }
    return C;
}

[[nodiscard]] std::vector<Real> denseRowMajorMatVec(const std::vector<Real>& A,
                                                    std::size_t m,
                                                    std::size_t n,
                                                    const std::vector<Real>& x)
{
    FE_THROW_IF(A.size() != m * n || x.size() != n,
                InvalidArgumentException,
                "NewtonSolver: denseRowMajorMatVec dimension mismatch");
    std::vector<Real> y(m, Real(0.0));
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            y[i] += A[i * n + j] * x[j];
        }
    }
    return y;
}

[[nodiscard]] std::vector<Real> denseColMajorMatVec(const std::vector<Real>& A_col_major,
                                                    std::size_t m,
                                                    std::size_t n,
                                                    const std::vector<Real>& x)
{
    FE_THROW_IF(A_col_major.size() != m * n || x.size() != n,
                InvalidArgumentException,
                "NewtonSolver: denseColMajorMatVec dimension mismatch");
    std::vector<Real> y(m, Real(0.0));
    for (std::size_t j = 0; j < n; ++j) {
        const Real xj = x[j];
        if (std::abs(xj) <= Real(1e-30)) {
            continue;
        }
        for (std::size_t i = 0; i < m; ++i) {
            y[i] += A_col_major[i + m * j] * xj;
        }
    }
    return y;
}

void rebaseGroupedCouplingIds(std::vector<backends::ReducedFieldUpdate>& reduced_updates,
                              std::vector<backends::GroupedBorderedFieldCoupling>& grouped_couplings,
                              int base_group_id)
{
    for (auto& upd : reduced_updates) {
        if (upd.grouped_coupling_id >= 0) {
            upd.grouped_coupling_id += base_group_id;
        }
    }
    for (auto& group : grouped_couplings) {
        group.grouped_coupling_id += base_group_id;
    }
}

[[nodiscard]] bool buildAlgebraicAuxiliaryReduction(
    const systems::FESystem::BorderedCouplingData& bordered,
    const dofs::IndexSet& owned_dofs,
    AlgebraicAuxiliaryReduction& out)
{
    out = {};
    if (!bordered.active || bordered.n_aux <= 0) {
        return false;
    }

    const auto nf = bordered.n_field_dofs;
    const auto na = static_cast<std::size_t>(bordered.n_aux);
    if (bordered.D.size() != na * na ||
        bordered.B.size() != nf * na ||
        bordered.Ct.size() != na * nf ||
        bordered.g.size() != na ||
        bordered.aux_variable_kinds.size() != na) {
        return false;
    }

    for (std::size_t i = 0; i < na; ++i) {
        if (bordered.aux_variable_kinds[i] == systems::AuxiliaryVariableKind::Algebraic) {
            out.algebraic_indices.push_back(i);
        } else {
            out.dynamic_indices.push_back(i);
        }
    }
    if (out.algebraic_indices.empty()) {
        return false;
    }

    const auto n_alg = out.algebraic_indices.size();
    const auto n_dyn = out.dynamic_indices.size();
    std::unordered_map<std::size_t, std::size_t> algebraic_position;
    algebraic_position.reserve(n_alg);
    for (std::size_t i = 0; i < n_alg; ++i) {
        algebraic_position.emplace(out.algebraic_indices[i], i);
    }
    const auto Daa = denseSubmatrixRowsCols(bordered.D, na, na,
                                            out.algebraic_indices,
                                            out.algebraic_indices);
    if (!invertDenseMatrix(Daa, n_alg, out.Daa_inv)) {
        return false;
    }

    const auto Ca = denseSubmatrixRows(bordered.Ct, nf, out.algebraic_indices);
    out.Daa_inv_Ca_field = denseMatMulRowMajor(out.Daa_inv, n_alg, n_alg, Ca, nf);

    std::vector<Real> g_alg(n_alg, Real(0.0));
    for (std::size_t i = 0; i < n_alg; ++i) {
        g_alg[i] = bordered.g[out.algebraic_indices[i]];
    }
    out.Daa_inv_ga = denseRowMajorMatVec(out.Daa_inv, n_alg, n_alg, g_alg);

    const auto B_alg = denseSubmatrixColumns(bordered.B, nf, out.algebraic_indices);
    out.rhs_shift = denseColMajorMatVec(B_alg, nf, n_alg, out.Daa_inv_ga);

    const bool independent_modes = denseMatrixIsEffectivelyDiagonal(Daa, n_alg);
    if (oopTraceEnabled()) {
        std::ostringstream oss;
        oss << "NewtonSolver: algebraic reduction structure"
            << " n_alg=" << n_alg
            << " n_dyn=" << n_dyn
            << " independent_modes=" << independent_modes
            << " direct_records=" << bordered.direct_coupling_records.size()
            << " Daa=[";
        for (std::size_t i = 0; i < n_alg; ++i) {
            if (i != 0) {
                oss << "; ";
            }
            for (std::size_t j = 0; j < n_alg; ++j) {
                if (j != 0) {
                    oss << ", ";
                }
                oss << Daa[i * n_alg + j];
            }
        }
        oss << "]";
        traceLog(oss.str());
    }
    const bool allow_native_rank_one_promotion = independent_modes;
    const int grouped_coupling_id = independent_modes ? -1 : 0;
    backends::GroupedBorderedFieldCoupling grouped{};
    grouped.grouped_coupling_id = grouped_coupling_id;
    grouped.aux_matrix = Daa;
    grouped.modes.reserve(n_alg);

    for (std::size_t j = 0; j < n_alg; ++j) {
        backends::ReducedFieldUpdate upd;
        upd.sigma = Real(-1.0);
        upd.grouped_coupling_id = grouped_coupling_id;
        upd.left.reserve(nf);
        upd.right.reserve(nf);
        std::vector<Real> left_column_full(nf, Real(0.0));

        backends::GroupedBorderedFieldCoupling::Mode mode;
        mode.left.reserve(nf);
        mode.right.reserve(nf);

        for (std::size_t row = 0; row < nf; ++row) {
            const Real left_val = B_alg[row + nf * j];
            left_column_full[row] = left_val;
            if (std::abs(left_val) > Real(1e-30) &&
                owned_dofs.contains(static_cast<GlobalIndex>(row))) {
                upd.left.emplace_back(static_cast<GlobalIndex>(row), left_val);
                mode.left.emplace_back(static_cast<GlobalIndex>(row), left_val);
            }

            const Real right_val = out.Daa_inv_Ca_field[j * nf + row];
            if (std::abs(right_val) > Real(1e-30) &&
                owned_dofs.contains(static_cast<GlobalIndex>(row))) {
                upd.right.emplace_back(static_cast<GlobalIndex>(row), right_val);
            }

            const Real ca_val = Ca[j * nf + row];
            if (std::abs(ca_val) > Real(1e-30) &&
                owned_dofs.contains(static_cast<GlobalIndex>(row))) {
                mode.right.emplace_back(static_cast<GlobalIndex>(row), ca_val);
            }
        }

        backends::RankOneUpdate promoted;
        bool promoted_ok = false;
        if (independent_modes) {
            for (const auto& record : bordered.direct_coupling_records) {
                const auto aux_it = std::find(record.aux_local_indices.begin(),
                                              record.aux_local_indices.end(),
                                              out.algebraic_indices[j]);
                if (aux_it == record.aux_local_indices.end()) {
                    continue;
                }
                if (tryPromoteAlgebraicDirectCouplingRecordToNativeRankOne(
                        bordered,
                        record,
                        algebraic_position,
                        out.Daa_inv,
                        n_alg,
                        owned_dofs,
                        promoted)) {
                    promoted_ok = true;
                    break;
                }
            }
        }
        if (!promoted_ok && allow_native_rank_one_promotion) {
            promoted_ok = tryPromoteExactReducedUpdateToNativeRankOne(upd, promoted);
        }
        if (!promoted_ok && !independent_modes) {
            for (const auto& record : bordered.direct_coupling_records) {
                if (tryPromoteDirectCouplingRecordToNativeRankOne(
                        bordered, record, out.algebraic_indices[j], left_column_full, owned_dofs, promoted)) {
                    promoted_ok = true;
                    break;
                }
            }
        }
        if (promoted_ok) {
            out.promoted_rank_one_updates.push_back(std::move(promoted));
            continue;
        }

        out.reduced_field_updates.push_back(std::move(upd));
        grouped.modes.push_back(std::move(mode));
    }

    if (!independent_modes &&
        !grouped.aux_matrix.empty() &&
        !grouped.modes.empty()) {
        out.grouped_couplings.push_back(std::move(grouped));
    }

    if (n_dyn == 0) {
        out.active = true;
        return true;
    }

    const auto D_ad = denseSubmatrixRowsCols(bordered.D, na, na,
                                             out.algebraic_indices,
                                             out.dynamic_indices);
    const auto D_da = denseSubmatrixRowsCols(bordered.D, na, na,
                                             out.dynamic_indices,
                                             out.algebraic_indices);
    const auto D_dd = denseSubmatrixRowsCols(bordered.D, na, na,
                                             out.dynamic_indices,
                                             out.dynamic_indices);
    out.Daa_inv_Dad = denseMatMulRowMajor(out.Daa_inv, n_alg, n_alg, D_ad, n_dyn);

    const auto B_dyn = denseSubmatrixColumns(bordered.B, nf, out.dynamic_indices);
    const auto B_shift = denseColMajorTimesRowMajor(B_alg, nf, n_alg, out.Daa_inv_Dad, n_dyn);
    const auto C_dyn = denseSubmatrixRows(bordered.Ct, nf, out.dynamic_indices);
    const auto C_shift = denseMatMulRowMajor(D_da, n_dyn, n_alg, out.Daa_inv_Ca_field, nf);
    const auto D_shift = denseMatMulRowMajor(D_da, n_dyn, n_alg, out.Daa_inv_Dad, n_dyn);
    const auto g_shift = denseRowMajorMatVec(D_da, n_dyn, n_alg, out.Daa_inv_ga);

    out.reduced_bordered.resize(static_cast<int>(n_dyn), nf);
    out.reduced_bordered.aux_variable_kinds.assign(
        n_dyn, systems::AuxiliaryVariableKind::Differential);
    out.reduced_bordered.aux_blocks.push_back({"algebraic_reduced_dynamic",
                                               static_cast<int>(n_dyn)});

    for (std::size_t j = 0; j < n_dyn; ++j) {
        for (std::size_t row = 0; row < nf; ++row) {
            out.reduced_bordered.B[row + nf * j] =
                B_dyn[row + nf * j] - B_shift[row * n_dyn + j];
        }
    }
    for (std::size_t i = 0; i < n_dyn; ++i) {
        out.reduced_bordered.g[i] =
            bordered.g[out.dynamic_indices[i]] - g_shift[i];
        for (std::size_t col = 0; col < nf; ++col) {
            out.reduced_bordered.Ct[i * nf + col] =
                C_dyn[i * nf + col] - C_shift[i * nf + col];
        }
        for (std::size_t j = 0; j < n_dyn; ++j) {
            out.reduced_bordered.D[i * n_dyn + j] =
                D_dd[i * n_dyn + j] - D_shift[i * n_dyn + j];
        }
    }

    out.active = true;
    return true;
}

[[nodiscard]] std::vector<Real> recoverAuxiliaryDeltaFromReduction(
    const AlgebraicAuxiliaryReduction& reduction,
    std::span<const Real> dense_du,
    std::span<const Real> reduced_dynamic_delta)
{
    if (!reduction.active) {
        return std::vector<Real>(reduced_dynamic_delta.begin(), reduced_dynamic_delta.end());
    }

    const auto n_aux_full =
        reduction.algebraic_indices.size() + reduction.dynamic_indices.size();
    std::vector<Real> full_delta(n_aux_full, Real(0.0));

    FE_THROW_IF(reduced_dynamic_delta.size() != reduction.dynamic_indices.size(),
                systems::InvalidStateException,
                "NewtonSolver: reduced dynamic auxiliary delta size mismatch");
    for (std::size_t j = 0; j < reduction.dynamic_indices.size(); ++j) {
        full_delta[reduction.dynamic_indices[j]] = reduced_dynamic_delta[j];
    }

    FE_THROW_IF(reduction.Daa_inv_Ca_field.size() !=
                    reduction.algebraic_indices.size() * dense_du.size(),
                systems::InvalidStateException,
                "NewtonSolver: algebraic reduction field recovery size mismatch");
    FE_THROW_IF(reduction.Daa_inv_Dad.size() !=
                    reduction.algebraic_indices.size() * reduction.dynamic_indices.size(),
                systems::InvalidStateException,
                "NewtonSolver: algebraic reduction dynamic recovery size mismatch");
    FE_THROW_IF(reduction.Daa_inv_ga.size() != reduction.algebraic_indices.size(),
                systems::InvalidStateException,
                "NewtonSolver: algebraic reduction rhs recovery size mismatch");

    for (std::size_t i = 0; i < reduction.algebraic_indices.size(); ++i) {
        Real value = reduction.Daa_inv_ga[i];
        for (std::size_t k = 0; k < dense_du.size(); ++k) {
            value -= reduction.Daa_inv_Ca_field[i * dense_du.size() + k] * dense_du[k];
        }
        for (std::size_t j = 0; j < reduction.dynamic_indices.size(); ++j) {
            value -= reduction.Daa_inv_Dad[i * reduction.dynamic_indices.size() + j] *
                     reduced_dynamic_delta[j];
        }
        full_delta[reduction.algebraic_indices[i]] = value;
    }

    return full_delta;
}

void applyAuxiliaryDelta(systems::FESystem& system,
                         const systems::FESystem::BorderedCouplingData& bc,
                         std::span<const Real> dx,
                         Real alpha)
{
    if (!bc.active || dx.empty() || !(alpha != Real(0.0))) {
        return;
    }

    auto* mgr = system.auxiliaryStateManagerIfPresent();
    FE_CHECK_NOT_NULL(mgr, "NewtonSolver: auxiliary state manager");

    std::size_t offset = 0;
    for (const auto& blk_info : bc.aux_blocks) {
        auto& blk = mgr->getBlock(blk_info.name);
        auto work = blk.work();
        const auto block_dim = static_cast<std::size_t>(blk_info.dim);
        FE_THROW_IF(offset + block_dim > dx.size(), systems::InvalidStateException,
                    "NewtonSolver: auxiliary bordered update exceeds dx size");
        FE_THROW_IF(work.size() != block_dim, systems::InvalidStateException,
                    "NewtonSolver: auxiliary bordered update size mismatch for block '" +
                        blk_info.name + "'");

        for (std::size_t i = 0; i < block_dim; ++i) {
            work[i] -= alpha * dx[offset + i];
        }
        offset += block_dim;
    }

    FE_THROW_IF(offset != dx.size(), systems::InvalidStateException,
                "NewtonSolver: auxiliary bordered update did not consume all dx entries");

    mgr->syncGhosts();
}

} // namespace

NewtonSolver::NewtonSolver(NewtonOptions options)
    : options_(std::move(options))
{
    FE_THROW_IF(options_.max_iterations <= 0, InvalidArgumentException,
                "NewtonSolver: max_iterations must be > 0");
    FE_THROW_IF(options_.abs_tolerance < 0.0 || !std::isfinite(options_.abs_tolerance),
                InvalidArgumentException,
                "NewtonSolver: abs_tolerance must be finite and >= 0");
    FE_THROW_IF(options_.rel_tolerance < 0.0 || !std::isfinite(options_.rel_tolerance),
                InvalidArgumentException,
                "NewtonSolver: rel_tolerance must be finite and >= 0");
    FE_THROW_IF(options_.step_tolerance < 0.0 || !std::isfinite(options_.step_tolerance),
                InvalidArgumentException,
                "NewtonSolver: step_tolerance must be finite and >= 0");

    FE_THROW_IF(options_.jacobian_rebuild_period <= 0, InvalidArgumentException,
                "NewtonSolver: jacobian_rebuild_period must be >= 1");
    if (options_.scale_dt_increments) {
        FE_THROW_IF(!std::isfinite(options_.dt_increment_scale), InvalidArgumentException,
                    "NewtonSolver: dt_increment_scale must be finite");
        FE_THROW_IF(options_.dt_increment_scale < 0.0, InvalidArgumentException,
                    "NewtonSolver: dt_increment_scale must be >= 0");
    }

    if (options_.use_line_search) {
        FE_THROW_IF(options_.line_search_max_iterations <= 0, InvalidArgumentException,
                    "NewtonSolver: line_search_max_iterations must be > 0 when line search is enabled");
        FE_THROW_IF(!(options_.line_search_alpha_min > 0.0) || options_.line_search_alpha_min > 1.0 ||
                        !std::isfinite(options_.line_search_alpha_min),
                    InvalidArgumentException,
                    "NewtonSolver: line_search_alpha_min must be finite and in (0,1]");
        FE_THROW_IF(!(options_.line_search_shrink > 0.0) || options_.line_search_shrink >= 1.0 ||
                        !std::isfinite(options_.line_search_shrink),
                    InvalidArgumentException,
                    "NewtonSolver: line_search_shrink must be finite and in (0,1)");
        FE_THROW_IF(!(options_.line_search_c1 > 0.0) || options_.line_search_c1 >= 1.0 ||
                        !std::isfinite(options_.line_search_c1),
                    InvalidArgumentException,
                    "NewtonSolver: line_search_c1 must be finite and in (0,1)");
    }

    if (options_.pseudo_transient.enabled) {
        FE_THROW_IF(options_.pseudo_transient.gamma_initial < 0.0 ||
                        !std::isfinite(options_.pseudo_transient.gamma_initial),
                    InvalidArgumentException,
                    "NewtonSolver: pseudo_transient.gamma_initial must be finite and >= 0");
        FE_THROW_IF(!(options_.pseudo_transient.gamma_growth > 1.0) ||
                        !std::isfinite(options_.pseudo_transient.gamma_growth),
                    InvalidArgumentException,
                    "NewtonSolver: pseudo_transient.gamma_growth must be finite and > 1");
        FE_THROW_IF(options_.pseudo_transient.gamma_max < 0.0 ||
                        !std::isfinite(options_.pseudo_transient.gamma_max),
                    InvalidArgumentException,
                    "NewtonSolver: pseudo_transient.gamma_max must be finite and >= 0");
        FE_THROW_IF(options_.pseudo_transient.gamma_drop_tolerance < 0.0 ||
                        !std::isfinite(options_.pseudo_transient.gamma_drop_tolerance),
                    InvalidArgumentException,
                    "NewtonSolver: pseudo_transient.gamma_drop_tolerance must be finite and >= 0");
        FE_THROW_IF(options_.pseudo_transient.max_linear_retries <= 0,
                    InvalidArgumentException,
                    "NewtonSolver: pseudo_transient.max_linear_retries must be > 0");
    }
}

systems::SystemStateView NewtonSolver::makeStateView(const TimeHistory& history, double solve_time) const
{
    systems::SystemStateView state;
    state.time = solve_time;
    state.dt = history.dt();
    const double stage_dt = solve_time - history.time();
    state.effective_dt =
        (std::isfinite(stage_dt) && stage_dt > 0.0) ? stage_dt : history.dt();
    state.dt_prev = history.dtPrev();
    state.u = history.uSpan();
    state.u_prev = history.uPrevSpan();
    state.u_prev2 = history.uPrev2Span();
    state.u_vector = &history.u();
    state.u_prev_vector = &history.uPrev();
    state.u_prev2_vector = &history.uPrev2();
    state.u_history = history.uHistorySpans();
    state.dt_history = history.dtHistory();
    return state;
}

void NewtonSolver::allocateWorkspace(const systems::FESystem& system,
                                     const backends::BackendFactory& factory,
                                     NewtonWorkspace& workspace) const
{
    const auto n_dofs = system.dofHandler().getNumDofs();
    FE_THROW_IF(n_dofs <= 0, systems::InvalidStateException, "NewtonSolver::allocateWorkspace: system has no DOFs");

    const auto* dist = system.distributedSparsityIfAvailable(options_.jacobian_op);
    if (dist != nullptr && factory.backendKind() != backends::BackendKind::Eigen) {
        workspace.jacobian = factory.createMatrix(*dist);
    } else {
        const auto& pattern = system.sparsity(options_.jacobian_op);
        workspace.jacobian = factory.createMatrix(pattern);
    }
    workspace.residual = factory.createVector(n_dofs);
    workspace.delta = factory.createVector(n_dofs);
    workspace.u_backup = factory.createVector(n_dofs);
    workspace.residual_scratch = factory.createVector(n_dofs);
    workspace.residual_base = factory.createVector(n_dofs);
    workspace.ptc_mass_lumped.reset();
    workspace.dt_field_dofs.clear();

    FE_CHECK_NOT_NULL(workspace.jacobian.get(), "NewtonSolver workspace.jacobian");
    FE_CHECK_NOT_NULL(workspace.residual.get(), "NewtonSolver workspace.residual");
    FE_CHECK_NOT_NULL(workspace.delta.get(), "NewtonSolver workspace.delta");
    FE_CHECK_NOT_NULL(workspace.u_backup.get(), "NewtonSolver workspace.u_backup");
    FE_CHECK_NOT_NULL(workspace.residual_scratch.get(), "NewtonSolver workspace.residual_scratch");
    FE_CHECK_NOT_NULL(workspace.residual_base.get(), "NewtonSolver workspace.residual_base");

    if (options_.pseudo_transient.enabled) {
        workspace.ptc_mass_lumped = factory.createVector(n_dofs);
        FE_CHECK_NOT_NULL(workspace.ptc_mass_lumped.get(), "NewtonSolver workspace.ptc_mass_lumped");
    }

    if (options_.scale_dt_increments) {
        const auto dt_fields = system.timeDerivativeFields();
        if (!dt_fields.empty()) {
            const auto& fmap = system.fieldMap();
            for (const auto fid : dt_fields) {
                if (fid < 0) {
                    continue;
                }
                const auto idx = static_cast<std::size_t>(fid);
                if (idx >= fmap.numFields()) {
                    continue;
                }
                const auto range = fmap.getFieldDofRange(idx);
                for (GlobalIndex d = range.first; d < range.second; ++d) {
                    workspace.dt_field_dofs.push_back(d);
                }
            }
            std::sort(workspace.dt_field_dofs.begin(), workspace.dt_field_dofs.end());
            workspace.dt_field_dofs.erase(
                std::unique(workspace.dt_field_dofs.begin(), workspace.dt_field_dofs.end()),
                workspace.dt_field_dofs.end());
        }
    }

    if (oopTraceEnabled()) {
        std::ostringstream oss;
        oss << "NewtonSolver::allocateWorkspace: backend=" << backends::backendKindToString(factory.backendKind())
            << " ndofs=" << n_dofs << " jacobian_op='" << options_.jacobian_op << "'"
            << " residual_op='" << options_.residual_op << "'"
            << " dist_sparsity=" << ((dist != nullptr && factory.backendKind() != backends::BackendKind::Eigen) ? "yes" : "no")
            << " dt_field_dofs=" << workspace.dt_field_dofs.size();
        traceLog(oss.str());
    }
}

NewtonReport NewtonSolver::solveStep(systems::TransientSystem& transient,
                                     backends::LinearSolver& linear,
                                     double solve_time,
                                     TimeHistory& history,
                                     NewtonWorkspace& workspace,
                                     const backends::GenericVector* residual_addition) const
{
    FE_THROW_IF(!workspace.isAllocated(), InvalidArgumentException,
                "NewtonSolver::solveStep: workspace not allocated");

    auto& J = *workspace.jacobian;
    auto& r = *workspace.residual;
    auto& du = *workspace.delta;
    auto& u_backup = *workspace.u_backup;
    auto& residual_scratch = *workspace.residual_scratch;
    auto& residual_base = *workspace.residual_base;

    NewtonReport report;

    const auto& sys = transient.system();
    const auto base_linear_options = sys.augmentSolverOptions(linear.getOptions());
    linear.setOptions(base_linear_options);
    const auto& constraints = sys.constraints();
    const int temporal_order = transient.system().temporalOrder();

    history.updateGhosts();

    struct NewtonStateWithContext {
        systems::SystemStateView view{};
        std::optional<assembly::TimeIntegrationContext> time_ctx{};
    };

    auto makeNewtonState = [&](const TimeHistory& hist, double time) {
        NewtonStateWithContext out;
        out.view = makeStateView(hist, time);
        if (temporal_order > 0) {
            out.time_ctx = transient.integrator().buildContext(temporal_order, out.view);
            out.view.time_integration = &(*out.time_ctx);
        }
        return out;
    };

    auto base_state_holder = makeNewtonState(history, solve_time);
    const auto& base_state = base_state_holder.view;
    FE_THROW_IF(!(base_state.dt > 0.0), InvalidArgumentException, "NewtonSolver: dt must be > 0");
    FE_THROW_IF(!std::isfinite(base_state.time), InvalidArgumentException, "NewtonSolver: solve_time must be finite");

    // Ensure time-dependent constraints (Dirichlet, etc.) are evaluated at the actual solve time.
    // This is required for multi-stage schemes (e.g., generalized-α) where the nonlinear solve
    // occurs at a stage time t_{n+α_f}, not necessarily at t_{n+1}.
    transient.system().updateConstraints(solve_time, base_state.dt);

    std::optional<assembly::TimeIntegrationContext> dt_scale_ctx;
    if (options_.scale_dt_increments && !(options_.dt_increment_scale > 0.0)) {
        const int max_order = transient.system().temporalOrder();
        if (max_order > 0) {
            dt_scale_ctx = transient.integrator().buildContext(max_order, base_state);
        }
    }

    const int max_it = options_.max_iterations;

    if (oopTraceEnabled()) {
        const auto& lopts = linear.getOptions();
        std::ostringstream oss;
        oss << "NewtonSolver::solveStep: time=" << solve_time << " dt=" << base_state.dt
            << " max_it=" << max_it
            << " abs_tol=" << options_.abs_tolerance << " rel_tol=" << options_.rel_tolerance
            << " step_tol=" << options_.step_tolerance
            << " residual_op='" << options_.residual_op << "' jacobian_op='" << options_.jacobian_op << "'"
            << " linear_backend=" << backends::backendKindToString(linear.backendKind())
            << " linear(method=" << backends::solverMethodToString(lopts.method)
            << ", pc=" << backends::preconditionerToString(lopts.preconditioner)
            << ", max_iter=" << lopts.max_iter
            << ", rel_tol=" << lopts.rel_tol
            << ", abs_tol=" << lopts.abs_tol << ")";
        traceLog(oss.str());
    }

    const bool same_op = (options_.residual_op == options_.jacobian_op);
    bool has_monolithic_auxiliary_unknowns = false;
    if (const auto* aux_registry = transient.system().auxiliaryOperatorRegistryIfPresent();
        aux_registry && aux_registry->isLayoutFinalized()) {
        has_monolithic_auxiliary_unknowns =
            aux_registry->auxiliaryLayout().total_aux_unknowns > 0;
    }

    std::vector<GlobalIndex> constrained_dofs;
    std::vector<GlobalIndex> dirichlet_dofs;
    bool has_non_dirichlet_affine_constraints = false;
    if (!constraints.empty()) {
        constrained_dofs.reserve(constraints.numConstraints());
        dirichlet_dofs.reserve(constraints.numConstraints());
        constraints.forEach([&constrained_dofs](const constraints::AffineConstraints::ConstraintView& cv) {
            if (cv.slave_dof >= 0) {
                constrained_dofs.push_back(cv.slave_dof);
            }
        });
        constraints.forEach(
            [&dirichlet_dofs, &has_non_dirichlet_affine_constraints](
                const constraints::AffineConstraints::ConstraintView& cv) {
                if (cv.slave_dof < 0) {
                    return;
                }
                if (cv.isDirichlet()) {
                    dirichlet_dofs.push_back(cv.slave_dof);
                } else {
                    has_non_dirichlet_affine_constraints = true;
                }
            });

        std::sort(constrained_dofs.begin(), constrained_dofs.end());
        constrained_dofs.erase(std::unique(constrained_dofs.begin(), constrained_dofs.end()),
                               constrained_dofs.end());

        std::sort(dirichlet_dofs.begin(), dirichlet_dofs.end());
        dirichlet_dofs.erase(std::unique(dirichlet_dofs.begin(), dirichlet_dofs.end()),
                             dirichlet_dofs.end());
    }
    linear.setDirichletDofs(dirichlet_dofs);

    auto zeroConstrainedResidualEntries = [&]() {
        if (constrained_dofs.empty()) {
            return;
        }
        auto r_zero = r.createAssemblyView();
        FE_CHECK_NOT_NULL(r_zero.get(), "NewtonSolver: residual zeroing view");
        r_zero->beginAssemblyPhase();
        r_zero->zeroVectorEntries(constrained_dofs);
        r_zero->finalizeAssembly();
    };

    auto applyResidualAdditionAndConstraints = [&]() {
        if (residual_addition != nullptr) {
            axpy(r, static_cast<Real>(1.0), *residual_addition);
        }
        zeroConstrainedResidualEntries();
    };

    auto reapplyConstrainedJacobianRows = [&]() {
        if (constrained_dofs.empty()) {
            return;
        }
        auto J_zero = J.createAssemblyView();
        FE_CHECK_NOT_NULL(J_zero.get(), "NewtonSolver: Jacobian constrained-row view");
        J_zero->beginAssemblyPhase();
        J_zero->zeroRows(constrained_dofs, /*set_diagonal=*/true);
        J_zero->finalizeAssembly();
    };

    auto computeResidualNorm = [&]() -> double {
        return borderedResidualNormForConvergence(
            r, residual_scratch, transient.system().borderedCoupling());
    };

    auto traceResidualComponents = [&](const char* phase) {
        if (!oopTraceEnabled()) {
            return;
        }
        const auto [pde_norm, aux_norm] = borderedResidualNormComponentsForConvergence(
            r, residual_scratch, transient.system().borderedCoupling());
        std::ostringstream oss;
        oss << "NewtonSolver: residual components";
        if (phase != nullptr) {
            oss << " phase='" << phase << "'";
        }
        oss << " pde=" << pde_norm
            << " aux=" << aux_norm
            << " combined=" << std::hypot(pde_norm, aux_norm);
        traceLog(oss.str());
    };

    const bool ptc_enabled = options_.pseudo_transient.enabled;
    std::vector<GlobalIndex> ptc_owned_dofs;
    if (ptc_enabled && workspace.ptc_mass_lumped != nullptr) {
        const auto dt_fields = sys.timeDerivativeFields(options_.jacobian_op);
        if (!dt_fields.empty()) {
            dofs::IndexSet dt_dofs_all;
            const auto& fmap = sys.fieldMap();
            for (const auto fid : dt_fields) {
                if (fid < 0) {
                    continue;
                }
                const auto idx = static_cast<std::size_t>(fid);
                if (idx >= fmap.numFields()) {
                    continue;
                }
                const auto range = fmap.getFieldDofRange(idx);
                dt_dofs_all = dt_dofs_all.unionWith(dofs::IndexSet(range.first, range.second));
            }
            const auto& owned = sys.dofHandler().getPartition().locallyOwned();
            ptc_owned_dofs = dt_dofs_all.intersectionWith(owned).toVector();
        }
    }

    const bool ptc_can_run = ptc_enabled && (workspace.ptc_mass_lumped != nullptr) && !ptc_owned_dofs.empty();
    bool ptc_mass_ready = false;
    double ptc_gamma = 0.0;
    double ptc_gamma_applied = 0.0;
    double ptc_prev_residual_norm = std::numeric_limits<double>::quiet_NaN();

    systems::OperatorTag residual_op_used = options_.residual_op;
    std::vector<backends::RankOneUpdate> assembled_rank_one_updates;
    std::vector<backends::RankOneUpdate> effective_rank_one_updates;
    std::vector<backends::ReducedFieldUpdate> assembled_reduced_field_updates;
    std::vector<backends::ReducedFieldUpdate> effective_reduced_field_updates;
    std::vector<backends::ReducedFieldUpdate> active_reduced_field_updates;
    std::vector<backends::GroupedBorderedFieldCoupling> grouped_bordered_field_couplings;
    bool linear_has_live_bordered = false;
    const systems::FESystem::BorderedCouplingData* solve_bordered_ptr = nullptr;
    AlgebraicAuxiliaryReduction algebraic_aux_reduction;

    auto captureRankOneUpdates = [&]() {
        const auto updates = transient.system().lastRankOneUpdates();
        assembled_rank_one_updates.assign(updates.begin(), updates.end());
        if (!constraints.empty() && !assembled_rank_one_updates.empty()) {
            effective_rank_one_updates =
                transformRankOneUpdatesForConstraints(assembled_rank_one_updates, constraints);
        } else {
            effective_rank_one_updates = assembled_rank_one_updates;
        }

        const auto reduced_updates = transient.system().lastReducedFieldUpdates();
        assembled_reduced_field_updates.assign(reduced_updates.begin(), reduced_updates.end());
        if (!constraints.empty() && !assembled_reduced_field_updates.empty()) {
            effective_reduced_field_updates =
                transformReducedFieldUpdatesForConstraints(assembled_reduced_field_updates,
                                                           constraints);
        } else {
            effective_reduced_field_updates = assembled_reduced_field_updates;
        }
        active_reduced_field_updates = effective_reduced_field_updates;
        grouped_bordered_field_couplings.clear();
    };

    auto assembleResidualOnly = [&](const systems::SystemStateView& state, const char* phase) -> double {
        residual_op_used = options_.residual_op;
        auto r_view = r.createAssemblyView();
        FE_CHECK_NOT_NULL(r_view.get(), "NewtonSolver: residual assembly view");

        if (oopTraceEnabled()) {
            std::string msg = "NewtonSolver: beginTimeStep() + assemble (vector) op='" + options_.residual_op + "'";
            if (phase != nullptr) {
                msg += " phase='";
                msg += phase;
                msg += "'";
            }
            traceLog(msg);
        }

        transient.system().beginTimeStep(/*reset_auxiliary_state=*/false,
                                         /*invalidate_auxiliary_inputs=*/false);
        systems::AssemblyRequest req;
        req.op = options_.residual_op;
        req.want_vector = true;
        req.suppress_constraint_inhomogeneity = true;
        req.is_nonlinear_iteration = true;
        const auto ar = transient.assemble(req, state, nullptr, r_view.get());
        FE_THROW_IF(!ar.success, FEException,
                    "NewtonSolver: residual assembly failed: " + ar.error_message);

        if (oopTraceEnabled()) {
            std::ostringstream oss;
            oss << "NewtonSolver: assemble op='" << req.op << "' want_matrix=0 want_vector=1"
                << " ok=" << ar.success
                << " elems=" << ar.elements_assembled
                << " vec_ins=" << ar.vector_entries_inserted
                << " time=" << ar.elapsed_time_seconds;
            if (!ar.success) {
                oss << " err='" << ar.error_message << "'";
            }
            if (phase != nullptr) {
                oss << " phase='" << phase << "'";
            }
            traceLog(oss.str());
        }

        applyResidualAdditionAndConstraints();
        traceResidualComponents(phase);
        return computeResidualNorm();
    };

    auto assembleJacobianOnly = [&](const systems::SystemStateView& state) {
        auto J_view = J.createAssemblyView();
        FE_CHECK_NOT_NULL(J_view.get(), "NewtonSolver: jacobian assembly view");

        if (oopTraceEnabled()) {
            traceLog("NewtonSolver: beginTimeStep() + assemble (matrix) op='" + options_.jacobian_op + "'");
        }
        transient.system().beginTimeStep(/*reset_auxiliary_state=*/false,
                                         /*invalidate_auxiliary_inputs=*/false);
        systems::AssemblyRequest req;
        req.op = options_.jacobian_op;
        req.want_matrix = true;
        req.is_nonlinear_iteration = true;
        const auto aj = transient.assemble(req, state, J_view.get(), nullptr);
        FE_THROW_IF(!aj.success, FEException,
                    "NewtonSolver: jacobian assembly failed: " + aj.error_message);
        captureRankOneUpdates();
        if (oopTraceEnabled()) {
            std::ostringstream oss;
            oss << "NewtonSolver: assemble op='" << req.op << "' want_matrix=1 want_vector=0"
                << " ok=" << aj.success
                << " elems=" << aj.elements_assembled
                << " mat_ins=" << aj.matrix_entries_inserted
                << " time=" << aj.elapsed_time_seconds;
            if (!aj.success) {
                oss << " err='" << aj.error_message << "'";
            }
            traceLog(oss.str());
        }
    };

    auto assembleJacobianAndResidual = [&](const systems::SystemStateView& state) -> double {
        residual_op_used = options_.residual_op;
        auto J_view = J.createAssemblyView();
        auto r_view = r.createAssemblyView();
        FE_CHECK_NOT_NULL(J_view.get(), "NewtonSolver: jacobian assembly view");
        FE_CHECK_NOT_NULL(r_view.get(), "NewtonSolver: residual assembly view");

        if (oopTraceEnabled()) {
            traceLog("NewtonSolver: beginTimeStep() + assemble (matrix+vector) op='" + options_.residual_op + "'");
        }
        transient.system().beginTimeStep(/*reset_auxiliary_state=*/false,
                                         /*invalidate_auxiliary_inputs=*/false);
        systems::AssemblyRequest req;
        req.op = options_.residual_op;
        req.want_matrix = true;
        req.want_vector = true;
        req.suppress_constraint_inhomogeneity = true;
        req.is_nonlinear_iteration = true;
        const auto ar = transient.assemble(req, state, J_view.get(), r_view.get());
        FE_THROW_IF(!ar.success, FEException,
                    "NewtonSolver: combined (matrix+vector) assembly failed: " + ar.error_message);
        captureRankOneUpdates();
        if (oopTraceEnabled()) {
            std::ostringstream oss;
            oss << "NewtonSolver: assemble op='" << req.op << "' want_matrix=1 want_vector=1"
                << " ok=" << ar.success
                << " elems=" << ar.elements_assembled
                << " mat_ins=" << ar.matrix_entries_inserted
                << " vec_ins=" << ar.vector_entries_inserted
                << " time=" << ar.elapsed_time_seconds;
            if (!ar.success) {
                oss << " err='" << ar.error_message << "'";
            }
            traceLog(oss.str());
        }

        applyResidualAdditionAndConstraints();
        traceResidualComponents("jacobian_and_residual");
        return computeResidualNorm();
    };

    auto assembleJacobianAndResidualWithJacobianOp = [&](const systems::SystemStateView& state,
                                                         bool& out_vector_ok) -> double {
        out_vector_ok = false;

        auto J_view = J.createAssemblyView();
        auto r_view = r.createAssemblyView();
        FE_CHECK_NOT_NULL(J_view.get(), "NewtonSolver: jacobian assembly view");
        FE_CHECK_NOT_NULL(r_view.get(), "NewtonSolver: residual assembly view");

        if (oopTraceEnabled()) {
            traceLog("NewtonSolver: beginTimeStep() + assemble (matrix+vector) op='" + options_.jacobian_op + "'");
        }
        transient.system().beginTimeStep(/*reset_auxiliary_state=*/false,
                                         /*invalidate_auxiliary_inputs=*/false);
        systems::AssemblyRequest req;
        req.op = options_.jacobian_op;
        req.want_matrix = true;
        req.want_vector = true;
        req.suppress_constraint_inhomogeneity = true;
        req.is_nonlinear_iteration = true;
        const auto ar = transient.assemble(req, state, J_view.get(), r_view.get());
        FE_THROW_IF(!ar.success, FEException,
                    "NewtonSolver: combined (matrix+vector) assembly failed: " + ar.error_message);
        captureRankOneUpdates();
        if (oopTraceEnabled()) {
            std::ostringstream oss;
            oss << "NewtonSolver: assemble op='" << req.op << "' want_matrix=1 want_vector=1"
                << " ok=" << ar.success
                << " elems=" << ar.elements_assembled
                << " mat_ins=" << ar.matrix_entries_inserted
                << " vec_ins=" << ar.vector_entries_inserted
                << " time=" << ar.elapsed_time_seconds;
            if (!ar.success) {
                oss << " err='" << ar.error_message << "'";
            }
            traceLog(oss.str());
        }

        out_vector_ok = (ar.vector_entries_inserted > 0);
        if (!out_vector_ok) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        residual_op_used = options_.jacobian_op;
        applyResidualAdditionAndConstraints();
        return computeResidualNorm();
    };

    bool force_explicit_rank_one_updates =
        std::getenv("SVMP_FORCE_EXPLICIT_RANK_ONE") != nullptr;

    auto bridgeRankOneUpdates = [&]() -> bool {
        const std::span<const backends::RankOneUpdate> rank_one_updates(
            effective_rank_one_updates.data(), effective_rank_one_updates.size());
        const std::span<const backends::ReducedFieldUpdate> reduced_updates(
            active_reduced_field_updates.data(), active_reduced_field_updates.size());
        const std::span<const backends::GroupedBorderedFieldCoupling> grouped_bordered_couplings(
            grouped_bordered_field_couplings.data(), grouped_bordered_field_couplings.size());
        if (effective_rank_one_updates.empty() && active_reduced_field_updates.empty()) {
            linear.setRankOneUpdates({});
            linear.setReducedFieldUpdates({});
            linear.setGroupedBorderedFieldCouplings({});
            return false;
        }
        const bool use_native_rank_one_updates =
            linear.supportsNativeRankOneUpdates() &&
            linear.supportsNativeReducedFieldUpdates() &&
            !force_explicit_rank_one_updates &&
            !has_non_dirichlet_affine_constraints;
        const bool force_explicit_matrix_assembly =
            linear_has_live_bordered && !use_native_rank_one_updates;
        if (oopTraceEnabled()) {
            traceLog("NewtonSolver: rank-1 updates=" + std::to_string(rank_one_updates.size()) +
                     " reduced updates=" + std::to_string(reduced_updates.size()) +
                     (force_explicit_matrix_assembly ? " (explicit matrix path)" : "")
                     + (has_non_dirichlet_affine_constraints
                            ? " (constraint-transformed)"
                            : ""));
            for (std::size_t i = 0; i < rank_one_updates.size(); ++i) {
                double v_norm_sq = 0.0;
                for (const auto& [dof, val] : rank_one_updates[i].v) {
                    (void)dof;
                    v_norm_sq += static_cast<double>(val) * static_cast<double>(val);
                }
                std::ostringstream oss;
                oss << "NewtonSolver: rank-1 update[" << i << "]"
                    << " sigma=" << rank_one_updates[i].sigma
                    << " ||v||=" << std::sqrt(v_norm_sq)
                    << " nnz=" << rank_one_updates[i].v.size();
                traceLog(oss.str());
            }
            for (std::size_t i = 0; i < reduced_updates.size(); ++i) {
                double left_norm_sq = 0.0;
                double right_norm_sq = 0.0;
                for (const auto& [dof, val] : reduced_updates[i].left) {
                    (void)dof;
                    left_norm_sq += static_cast<double>(val) * static_cast<double>(val);
                }
                for (const auto& [dof, val] : reduced_updates[i].right) {
                    (void)dof;
                    right_norm_sq += static_cast<double>(val) * static_cast<double>(val);
                }
                std::ostringstream oss;
                oss << "NewtonSolver: reduced update[" << i << "]"
                    << " sigma=" << reduced_updates[i].sigma
                    << " ||u||=" << std::sqrt(left_norm_sq)
                    << " ||v||=" << std::sqrt(right_norm_sq)
                    << " left_nnz=" << reduced_updates[i].left.size()
                    << " right_nnz=" << reduced_updates[i].right.size();
                traceLog(oss.str());
            }
        }
        if (use_native_rank_one_updates) {
            linear.setRankOneUpdates(rank_one_updates);
            linear.setReducedFieldUpdates(reduced_updates);
            linear.setGroupedBorderedFieldCouplings(grouped_bordered_couplings);
            return false;
        }

        linear.setRankOneUpdates({});
        linear.setReducedFieldUpdates({});
        linear.setGroupedBorderedFieldCouplings({});
        {
            // Assemble the direct feedthrough contribution explicitly into the
            // bordered Jacobian so the monolithic Newton operator is backend
            // independent and the bordered K^{-1}B solves see the same matrix.
            auto J_view = J.createAssemblyView();
            FE_CHECK_NOT_NULL(J_view.get(), "NewtonSolver: rank-1 fallback view");
            J_view->beginAssemblyPhase();
            std::vector<GlobalIndex> col_dofs;
            std::vector<Real> row_vals;
            std::array<GlobalIndex, 1> row_dof{};
            for (const auto& upd : rank_one_updates) {
                col_dofs.resize(upd.v.size());
                row_vals.resize(upd.v.size());
                for (std::size_t j = 0; j < upd.v.size(); ++j) {
                    col_dofs[j] = upd.v[j].first;
                }
                for (const auto& ri : upd.v) {
                    row_dof[0] = ri.first;
                    const Real scale = upd.sigma * ri.second;
                    for (std::size_t j = 0; j < upd.v.size(); ++j) {
                        row_vals[j] = scale * upd.v[j].second;
                    }
                    J_view->addMatrixEntries(
                        std::span<const GlobalIndex>(row_dof.data(), row_dof.size()),
                        std::span<const GlobalIndex>(col_dofs.data(), col_dofs.size()),
                        std::span<const Real>(row_vals.data(), row_vals.size()),
                        assembly::AddMode::Add);
                }
            }
            for (const auto& upd : reduced_updates) {
                col_dofs.resize(upd.right.size());
                row_vals.resize(upd.right.size());
                for (std::size_t j = 0; j < upd.right.size(); ++j) {
                    col_dofs[j] = upd.right[j].first;
                }
                for (const auto& ri : upd.left) {
                    row_dof[0] = ri.first;
                    const Real scale = upd.sigma * ri.second;
                    for (std::size_t j = 0; j < upd.right.size(); ++j) {
                        row_vals[j] = scale * upd.right[j].second;
                    }
                    J_view->addMatrixEntries(
                        std::span<const GlobalIndex>(row_dof.data(), row_dof.size()),
                        std::span<const GlobalIndex>(col_dofs.data(), col_dofs.size()),
                        std::span<const Real>(row_vals.data(), row_vals.size()),
                        assembly::AddMode::Add);
                }
            }
            J_view->finalizeAssembly();
            if (oopTraceEnabled()) {
                traceLog("NewtonSolver: reduced updates assembled directly into matrix");
            }
        }
        return true;
    };

    auto tolerancesSatisfied = [&](double norm, bool pre_first_update) -> bool {
        const bool abs_enabled = options_.abs_tolerance > 0.0;
        const bool rel_enabled = options_.rel_tolerance > 0.0;
        const bool abs_ok = abs_enabled && norm <= options_.abs_tolerance;
        const bool rel_ok = rel_enabled
            && (report.residual_norm0 > 0.0
                    ? (norm / report.residual_norm0 <= options_.rel_tolerance)
                    : abs_ok);
        if (!abs_enabled && !rel_enabled) {
            return false;
        }

        // Match the time-loop convergence semantics: once Newton has taken at
        // least one update, either the absolute or relative residual criterion
        // may terminate the solve.  Still avoid short-circuiting before the
        // first update when a very loose abs_tol is combined with a meaningful
        // relative tolerance, since callers use that combination to force at
        // least one Newton correction.
        if (pre_first_update && rel_enabled && !rel_ok) {
            return false;
        }
        return abs_ok || rel_ok;
    };

    auto assembleDtOnlyJacobianAndLumpedDiagonal = [&](const systems::SystemStateView& state) -> bool {
        if (!ptc_can_run) {
            return false;
        }

        auto* mass_lumped = workspace.ptc_mass_lumped.get();
        FE_CHECK_NOT_NULL(mass_lumped, "NewtonSolver: PTC mass lumped vector");

        const int max_order = transient.system().temporalOrder();
        if (max_order <= 0) {
            return false;
        }

        auto ctx_base = transient.integrator().buildContext(max_order, state);
        assembly::TimeIntegrationContext ctx_dt_only = ctx_base;
        ctx_dt_only.time_derivative_term_weight = static_cast<Real>(1.0);
        ctx_dt_only.non_time_derivative_term_weight = static_cast<Real>(0.0);

        systems::SystemStateView state_dt = state;
        state_dt.time_integration = &ctx_dt_only;

        J.zero();
        auto J_view = J.createAssemblyView();
        FE_CHECK_NOT_NULL(J_view.get(), "NewtonSolver: PTC dt-only Jacobian view");

        transient.system().beginTimeStep(/*reset_auxiliary_state=*/false,
                                         /*invalidate_auxiliary_inputs=*/false);
        systems::AssemblyRequest req;
        req.op = options_.jacobian_op;
        req.want_matrix = true;
        req.zero_outputs = true;
        req.suppress_constraint_inhomogeneity = true;
        const auto ar = transient.system().assemble(req, state_dt, J_view.get(), /*vector_out=*/nullptr);
        FE_THROW_IF(!ar.success, FEException,
                    "NewtonSolver: PTC dt-only Jacobian assembly failed: " + ar.error_message);

        // Lump: m = A_dt * 1  (row sums of dt-only Jacobian).
        residual_scratch.set(static_cast<Real>(1.0));
        residual_scratch.updateGhosts();
        mass_lumped->zero();
        J.mult(residual_scratch, *mass_lumped);
        ptc_mass_ready = true;
        return true;
    };

    auto applyPtcDiagonalShift = [&](double target_gamma) {
        if (!ptc_can_run || !ptc_mass_ready) {
            return;
        }
        const double clamped = std::clamp(target_gamma, 0.0, options_.pseudo_transient.gamma_max);
        const double delta_gamma = clamped - ptc_gamma_applied;
        if (delta_gamma == 0.0) {
            ptc_gamma_applied = clamped;
            return;
        }

        auto* mass_lumped = workspace.ptc_mass_lumped.get();
        FE_CHECK_NOT_NULL(mass_lumped, "NewtonSolver: PTC mass lumped vector");
        auto m_view = mass_lumped->createAssemblyView();
        FE_CHECK_NOT_NULL(m_view.get(), "NewtonSolver: PTC mass view");

        auto J_mod = J.createAssemblyView();
        FE_CHECK_NOT_NULL(J_mod.get(), "NewtonSolver: PTC matrix modify view");
        J_mod->beginAssemblyPhase();
        for (const auto dof : ptc_owned_dofs) {
            const Real m = m_view->getVectorEntry(dof);
            const double md = std::abs(static_cast<double>(m));
            if (!(md > 0.0) || !std::isfinite(md)) {
                continue;
            }
            const double v = delta_gamma * md;
            if (v == 0.0 || !std::isfinite(v)) {
                continue;
            }
            J_mod->addMatrixEntry(dof, dof, static_cast<Real>(v), assembly::AddMode::Add);
        }
        J_mod->finalizeAssembly();
        ptc_gamma_applied = clamped;
    };

    bool have_residual = false;
    double current_residual_norm = std::numeric_limits<double>::quiet_NaN();
    bool have_jacobian = false;
    int last_jacobian_it = -1;
    const int base_jacobian_period = std::max(1, options_.jacobian_rebuild_period);
    int direct_only_outlet_jacobian_period = 1;

    // ===== NEWTON TIMING PROFILE =====
#ifdef SVMP_FE_ASSEMBLY_TIMING
    auto NTP = []() {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    };
#else
    auto NTP = []() -> double { return 0.0; };
#endif
    double ntp_assembly = 0.0, ntp_linear = 0.0, ntp_update = 0.0;
    double ntp_constraints = 0.0, ntp_other = 0.0;
    double ntp_total_start = NTP();
    double ntp0;
    int ntp_assembly_count = 0, ntp_linear_iters_total = 0;
    auto printNewtonProfile = [&](int newton_iters) {
#ifdef SVMP_FE_ASSEMBLY_TIMING
        double ntp_total = NTP() - ntp_total_start;
        ntp_other = ntp_total - ntp_assembly - ntp_linear - ntp_update - ntp_constraints;
        if (ntp_other < 0.0) ntp_other = 0.0;
        int mpi_rank = 0;
#if FE_HAS_MPI
        int mpi_initialized = 0;
        MPI_Initialized(&mpi_initialized);
        if (mpi_initialized) {
            MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        }
#endif
        if (mpi_rank == 0 && ntp_total > 1e-6) {
            auto pct = [&](double t) { return 100.0 * t / ntp_total; };
            fprintf(stderr,
              "\n+++ NEWTON SOLVER TIMING (rank 0) +++\n"
              "  Total Newton time:    %10.6f s  (%d Newton iters, %d assemblies, %d linear iters)\n"
              "  Assembly (J+r):       %10.6f s  (%5.1f%%)\n"
              "  Linear solve:         %10.6f s  (%5.1f%%)\n"
              "  Solution update:      %10.6f s  (%5.1f%%)\n"
              "  Constraint/ghosts:    %10.6f s  (%5.1f%%)\n"
              "  Other (overhead):     %10.6f s  (%5.1f%%)\n"
              "+++++++++++++++++++++++++++++++++++++++\n",
              ntp_total, newton_iters, ntp_assembly_count, ntp_linear_iters_total,
              ntp_assembly, pct(ntp_assembly),
              ntp_linear, pct(ntp_linear),
              ntp_update, pct(ntp_update),
              ntp_constraints, pct(ntp_constraints),
              ntp_other, pct(ntp_other));
        }
#else
        (void)newton_iters;
#endif
    };
    // =================================

    double prev_residual_norm = -1.0;

    for (int it = 0; it < max_it; ++it) {
        ntp0 = NTP();
        history.updateGhosts();

        if (!constraints.empty()) {
            constraints.distribute(history.u());
            history.u().updateGhosts();
        }
        ntp_constraints += NTP() - ntp0;

        auto state_holder = makeNewtonState(history, solve_time);
        const auto& state = state_holder.view;

        if (have_residual && !std::isfinite(current_residual_norm)) {
            // If the cached residual norm is invalid (e.g., NaN from a failed evaluation),
            // fall back to re-assembling the residual at the current state.
            have_residual = false;
        }

        const int jacobian_period = std::max(base_jacobian_period, direct_only_outlet_jacobian_period);
        const bool need_jacobian =
            !have_jacobian || (jacobian_period == 1) || ((it - last_jacobian_it) >= jacobian_period);
        bool jacobian_ready = have_jacobian && !need_jacobian;
        if (!have_residual) {
            ntp0 = NTP();
            if (need_jacobian && options_.assemble_both_when_possible && same_op) {
                // Residual and Jacobian share the same operator tag, so we can assemble both in one pass.
                current_residual_norm = assembleJacobianAndResidual(state);
                ptc_gamma_applied = 0.0;
                jacobian_ready = true;
                have_jacobian = true;
                last_jacobian_it = it;
            } else {
                // When residual_op != jacobian_op, always assemble the residual using residual_op so
                // Newton convergence checks and line search evaluate the same residual used in the
                // linear solve. (Some modules may also install vector contributions under jacobian_op
                // as an optimization; those must not silently change the residual definition.)
                current_residual_norm = assembleResidualOnly(state, /*phase=*/nullptr);
                if (need_jacobian) {
                    assembleJacobianOnly(state);
                    ptc_gamma_applied = 0.0;
                    jacobian_ready = true;
                    have_jacobian = true;
                    last_jacobian_it = it;
                }
            }
            ntp_assembly += NTP() - ntp0;
            ntp_assembly_count++;
            have_residual = true;
        } else if (need_jacobian && options_.assemble_both_when_possible && same_op &&
                   has_monolithic_auxiliary_unknowns) {
            ntp0 = NTP();
            current_residual_norm = assembleJacobianAndResidual(state);
            ptc_gamma_applied = 0.0;
            jacobian_ready = true;
            have_jacobian = true;
            last_jacobian_it = it;
            have_residual = true;
            ntp_assembly += NTP() - ntp0;
            ntp_assembly_count++;
        }

        report.residual_norm = current_residual_norm;
        if (it == 0) {
            report.residual_norm0 = current_residual_norm;
        }

        if (oopTraceEnabled()) {
            std::ostringstream oss;
            const double denom = (report.residual_norm0 > 0.0) ? report.residual_norm0 : 1.0;
            oss << "NewtonSolver: it=" << it
                << " ||r||=" << report.residual_norm
                << " ||r0||=" << report.residual_norm0
                << " rel=" << (report.residual_norm / denom);
            traceLog(oss.str());
        }

        // Nullspace validation: on the first iteration with a Jacobian, optionally
        // verify that inferred nullspace vectors are actually in the operator's nullspace.
        // Gated by SVMP_GAUGE_VALIDATE environment variable to avoid overhead in production.
        if (it == 0 && have_jacobian && gauge::isNullspaceValidationEnabled()) {
            const auto* reg = transient.system().gaugeRegistryIfPresent();
            if (reg && reg->isResolved()) {
                const auto n_dofs = transient.system().dofHandler().getNumDofs();
                auto get_field_dofs = [&](FieldId fid, int /*comp*/) -> std::vector<GlobalIndex> {
                    const auto offset = transient.system().fieldDofOffset(fid);
                    const auto& fdh = transient.system().fieldDofHandler(fid);
                    const auto nd = fdh.getNumDofs();
                    std::vector<GlobalIndex> dofs;
                    dofs.reserve(static_cast<std::size_t>(nd));
                    for (GlobalIndex d = offset; d < offset + nd; ++d) dofs.push_back(d);
                    return dofs;
                };
                // Build basis from ALL resolved modes (not just SolverNullspace)
                // by temporarily treating all ExactNullspace modes as needing basis
                auto all_basis = reg->buildNullspaceBasis(n_dofs, get_field_dofs);
                if (!all_basis.empty()) {
                    auto validation_factory = backends::BackendFactory::create(J.backendKind());
                    if (validation_factory) {
                        auto results = gauge::validateNullspaceBasis(
                            J, *validation_factory, all_basis);
                        std::fprintf(stderr, "%s",
                            gauge::formatValidationReport(results).c_str());
                    }
                }
            }
        }

        if (tolerancesSatisfied(current_residual_norm, /*pre_first_update=*/it == 0)) {
            report.converged = true;
            report.iterations = it;
            if (oopTraceEnabled()) {
                traceLog("NewtonSolver: converged before linear solve (tolerances satisfied).");
            }
            printNewtonProfile(it);
            return report;
        }

        // Stagnation is diagnostic only unless the configured nonlinear
        // tolerances are already satisfied. Do not override the requested
        // tolerances with a "best effort" convergence declaration.
        if (it > 0 && options_.stagnation_tolerance > 0.0 &&
            prev_residual_norm > 0.0 && std::isfinite(prev_residual_norm) &&
            report.residual_norm0 > 0.0 && current_residual_norm < report.residual_norm0) {
            const double ratio = current_residual_norm / prev_residual_norm;
            if (ratio >= options_.stagnation_tolerance) {
                if (oopTraceEnabled()) {
                    std::ostringstream oss;
                    oss << "NewtonSolver: stagnation detected (||r_k||/||r_{k-1}||="
                        << ratio << " >= " << options_.stagnation_tolerance << ")";
                    traceLog(oss.str());
                }
            }
        }
        prev_residual_norm = current_residual_norm;

        if (ptc_can_run) {
            if (options_.pseudo_transient.update_from_residual_ratio && ptc_mass_ready &&
                std::isfinite(ptc_prev_residual_norm) && ptc_prev_residual_norm > 0.0 &&
                std::isfinite(current_residual_norm) && current_residual_norm >= 0.0) {
                const double ratio = current_residual_norm / ptc_prev_residual_norm;
                if (std::isfinite(ratio) && ratio > 0.0) {
                    ptc_gamma = std::min(ptc_gamma * ratio, options_.pseudo_transient.gamma_max);
                    if (ptc_gamma < options_.pseudo_transient.gamma_drop_tolerance) {
                        ptc_gamma = 0.0;
                    }
                }
            }
            ptc_prev_residual_norm = current_residual_norm;
        }

        if (need_jacobian && !jacobian_ready) {
            assembleJacobianOnly(state);
            ptc_gamma_applied = 0.0;
            have_jacobian = true;
            last_jacobian_it = it;
        }

        if (jacobianCheckEnabled() && need_jacobian && it == jacobianCheckNewtonIteration()) {
            // Directional finite-difference check: compare J*v (from `jacobian_op`) to
            // (r(u+h*v)-r(u))/h (assembled with `residual_op`).
            //
            // This is a lightweight runtime diagnostic for missing/incomplete Jacobians *and*
            // operator mismatches between the configured residual and Jacobian operators.
            const double rel_step = jacobianCheckRelativeStep();
            const int n_dofs = sys.dofHandler().getNumDofs();
            const double u_norm = history.u().norm();
            const double u_rms = (n_dofs > 0) ? (u_norm / std::sqrt(static_cast<double>(n_dofs))) : u_norm;
            const double h = rel_step * (1.0 + u_rms);

            if (h > 0.0 && std::isfinite(h)) {
                // Populate a deterministic pseudo-random direction in `du` (will be overwritten by the linear solve).
                {
                    auto v = du.localSpan();
                    std::uint64_t s = 0x9e3779b97f4a7c15ULL ^ static_cast<std::uint64_t>(mpiRank() + 1);
                    for (std::size_t i = 0; i < v.size(); ++i) {
                        // xorshift64*
                        s ^= s >> 12;
                        s ^= s << 25;
                        s ^= s >> 27;
                        const std::uint64_t x = s * 2685821657736338717ULL;
                        const double u01 = static_cast<double>((x >> 11) & ((1ULL << 53) - 1ULL)) *
                            (1.0 / 9007199254740992.0); // 2^53
                        v[i] = static_cast<Real>(2.0 * u01 - 1.0);
                    }
                }
                zeroVectorEntries(constrained_dofs, du);
                du.updateGhosts();

                auto applyResidualFixups = [&](backends::GenericVector& vec) {
                    if (residual_addition != nullptr) {
                        axpy(vec, static_cast<Real>(1.0), *residual_addition);
                    }
                    zeroVectorEntries(constrained_dofs, vec);
                };

                // Backup the current nonlinear state so the diagnostic
                // assemblies do not perturb the live monolithic bordered data.
                copyVector(u_backup, history.u());
                const auto aux_state_backup =
                    transient.system().checkpointAuxiliaryState();
                const auto bordered_backup = transient.system().borderedCoupling();

                auto restoreDiagnosticState = [&]() {
                    copyVector(history.u(), u_backup);
                    if (!constraints.empty()) {
                        constraints.distribute(history.u());
                    }
                    history.u().updateGhosts();
                    if (!aux_state_backup.empty()) {
                        transient.system().restoreAuxiliaryState(aux_state_backup);
                    }
                    transient.system().borderedCoupling() = bordered_backup;
                    if (auto* reg = transient.system().auxiliaryInputRegistryIfPresent()) {
                        reg->invalidateAll();
                    }
                };

                // Assemble r(u) with residual_op into residual_base.
                residual_base.zero();
                {
                    auto r_view = residual_base.createAssemblyView();
                    FE_CHECK_NOT_NULL(r_view.get(), "NewtonSolver: jacobian check residual base view");

                    transient.system().beginTimeStep(/*reset_auxiliary_state=*/false,
                                                     /*invalidate_auxiliary_inputs=*/false);
                    systems::AssemblyRequest req;
                    req.op = options_.residual_op;
                    req.want_vector = true;
                    req.suppress_constraint_inhomogeneity = true;
                    req.is_nonlinear_iteration = true;
                    const auto ar = transient.assemble(req, state, nullptr, r_view.get());
                    FE_THROW_IF(!ar.success, FEException,
                                "NewtonSolver: Jacobian check base residual assembly failed: " + ar.error_message);
                }
                applyResidualFixups(residual_base);
                restoreDiagnosticState();

                // Assemble r(u + h*v) with residual_op into residual_scratch.
                axpy(history.u(), static_cast<Real>(h), du);
                if (!constraints.empty()) {
                    constraints.distribute(history.u());
                }
                history.u().updateGhosts();

                residual_scratch.zero();
                {
                    auto r_view = residual_scratch.createAssemblyView();
                    FE_CHECK_NOT_NULL(r_view.get(), "NewtonSolver: jacobian check residual perturbed view");

                    transient.system().beginTimeStep(/*reset_auxiliary_state=*/false,
                                                     /*invalidate_auxiliary_inputs=*/false);
                    systems::AssemblyRequest req;
                    req.op = options_.residual_op;
                    req.want_vector = true;
                    req.suppress_constraint_inhomogeneity = true;
                    req.is_nonlinear_iteration = true;
                    auto perturbed_state_holder = makeNewtonState(history, solve_time);
                    const auto ar = transient.assemble(
                        req, perturbed_state_holder.view, nullptr, r_view.get());
                    FE_THROW_IF(!ar.success, FEException,
                                "NewtonSolver: Jacobian check perturbed residual assembly failed: " + ar.error_message);
                }
                applyResidualFixups(residual_scratch);
                restoreDiagnosticState();

                const double r_base_norm = residualNormForConvergence(residual_base, u_backup);
                const double r_used_norm = residualNormForConvergence(r, u_backup);

                // residual_scratch <- (r(u+h*v) - r(u)) / h  (FD approximation of J*v).
                axpy(residual_scratch, static_cast<Real>(-1.0), residual_base);
                residual_scratch.scale(static_cast<Real>(1.0 / h));

                // u_backup <- r_used - r_base (will overwrite u_backup).
                copyVector(u_backup, r);
                axpy(u_backup, static_cast<Real>(-1.0), residual_base);
                zeroVectorEntries(constrained_dofs, u_backup);
                const double r_diff_norm = residualNormForConvergence(u_backup, residual_base);

                // u_backup <- J_matrix*v (without the pending low-rank outlet correction).
                u_backup.zero();
                J.mult(du, u_backup);
                zeroVectorEntries(constrained_dofs, u_backup);
                const double matrix_jv_norm = u_backup.norm();

                // residual_base keeps a copy of the matrix-only action so we can
                // compare both the raw assembled matrix and the full effective
                // operator (matrix + pending rank-1 updates) against FD.
                copyVector(residual_base, u_backup);
                axpy(u_backup, static_cast<Real>(-1.0), residual_scratch);
                const double matrix_err_norm = u_backup.norm();

                copyVector(u_backup, residual_base);
                if (!effective_rank_one_updates.empty()) {
                    addRankOneOperatorMatvec(
                        std::span<const backends::RankOneUpdate>(effective_rank_one_updates.data(),
                                                                 effective_rank_one_updates.size()),
                        du,
                        u_backup);
                }
                if (!active_reduced_field_updates.empty()) {
                    addReducedFieldOperatorMatvec(
                        std::span<const backends::ReducedFieldUpdate>(
                            active_reduced_field_updates.data(),
                            active_reduced_field_updates.size()),
                        du,
                        u_backup);
                }
                zeroVectorEntries(constrained_dofs, u_backup);
                const double jv_norm = u_backup.norm();

                // residual_base <- -(rank-one contribution)
                axpy(residual_base, static_cast<Real>(-1.0), u_backup);
                const double rank_one_jv_norm = residual_base.norm();

                // The FD residual is assembled by element ownership; sum overlap contributions once for comparison.
                accumulateOverlapIfNeeded(residual_scratch);
                const double fd_norm = residual_scratch.norm();

                // u_backup <- (J_matrix + rank1)*v - FD
                axpy(u_backup, static_cast<Real>(-1.0), residual_scratch);
                const double err_norm = u_backup.norm();
                const double denom = std::max({jv_norm, fd_norm, 1e-14});
                const double rel_err = err_norm / denom;

                // Rebuild the matrix-only mismatch vector for the per-component
                // diagnostic after using `residual_base` as a rank-one scratch.
                residual_base.zero();
                J.mult(du, residual_base);
                zeroVectorEntries(constrained_dofs, residual_base);
                axpy(residual_base, static_cast<Real>(-1.0), residual_scratch);

                if (mpiRank() == 0) {
                    std::ostringstream oss;
                    oss << "NewtonSolver: Jacobian check jacobian_op='" << options_.jacobian_op
                        << "' residual_op='" << options_.residual_op << "'"
                        << " it=" << it
                        << " h=" << h
                        << " ||J_matrix*v||=" << matrix_jv_norm
                        << " ||rank1*v||=" << rank_one_jv_norm
                        << " ||Jv||=" << jv_norm
                        << " ||FD||=" << fd_norm
                        << " ||J_matrix*v-FD||=" << matrix_err_norm
                        << " ||Jv-FD||=" << err_norm
                        << " rel=" << rel_err
                        << " ||r(residual_op)||=" << r_base_norm
                        << " ||r(used_op=" << residual_op_used << ")||=" << r_used_norm
                        << " ||r_used-r_residual||=" << r_diff_norm;
                    FE_LOG_INFO(oss.str());
                }
                logJacobianCheckComponentBreakdown(transient.system(),
                                                  residual_scratch,
                                                  u_backup,
                                                  residual_base);
                logJacobianCheckTopEntries(transient.system(), u_backup, 8u);
            } else if (mpiRank() == 0) {
                FE_LOG_INFO("NewtonSolver: Jacobian check skipped (invalid perturbation size).");
            }
        }

        du.zero();

        const bool ptc_always_on = ptc_can_run && !options_.pseudo_transient.activate_on_linear_failure &&
                                  (options_.pseudo_transient.gamma_initial > 0.0);
        if (ptc_always_on && !ptc_mass_ready) {
            // Assemble dt-only Jacobian to build a mass-like diagonal, then restore the physical Jacobian.
            (void)assembleDtOnlyJacobianAndLumpedDiagonal(state);

            if (options_.assemble_both_when_possible && same_op) {
                current_residual_norm = assembleJacobianAndResidual(state);
                have_residual = true;
                have_jacobian = true;
                last_jacobian_it = it;
            } else {
                current_residual_norm = assembleResidualOnly(state, /*phase=*/"ptc_restore");
                have_residual = true;
                assembleJacobianOnly(state);
                ptc_gamma_applied = 0.0;
                have_jacobian = true;
                last_jacobian_it = it;
            }
            ptc_gamma_applied = 0.0;
            ptc_gamma = options_.pseudo_transient.gamma_initial;
        }

        // Apply current PTC diagonal shift (may be zero).
        if (ptc_can_run && ptc_mass_ready) {
            applyPtcDiagonalShift(ptc_gamma);
        }

        // Bridge nullspace basis from GaugeRegistry to the linear solver.
        // Currently dormant: the resolver always uses algebraic enforcement,
        // so buildNullspaceBasis() returns empty.  This path is retained for
        // future SolverNullspace opt-in.
        if (linear.supportsNullspace()) {
            const auto* reg = transient.system().gaugeRegistryIfPresent();
            if (reg && reg->isResolved()) {
                const auto n_dofs = transient.system().dofHandler().getNumDofs();
                auto get_field_dofs = [&](FieldId fid, int comp) -> std::vector<GlobalIndex> {
                    const auto idx = static_cast<std::size_t>(fid);
                    const auto& sys = transient.system();

                    // Component-aware: return only DOFs for the requested component
                    if (comp >= 0 && idx < sys.fieldMap().numFields()) {
                        const auto n_comp = sys.fieldMap().numComponents(idx);
                        if (n_comp > 1 && static_cast<LocalIndex>(comp) < n_comp) {
                            return sys.fieldMap().getComponentDofs(idx, static_cast<LocalIndex>(comp)).toVector();
                        }
                    }

                    const auto offset = sys.fieldDofOffset(fid);
                    const auto& fdh = sys.fieldDofHandler(fid);
                    const auto nd = fdh.getNumDofs();
                    std::vector<GlobalIndex> dofs;
                    dofs.reserve(static_cast<std::size_t>(nd));
                    for (GlobalIndex d = offset; d < offset + nd; ++d) {
                        dofs.push_back(d);
                    }
                    return dofs;
                };
                // Build CoordinateProvider for rotation mode basis vectors.
                gauge::GaugeRegistry::CoordinateProvider coord_provider;
                const auto* emap = transient.system().dofHandler().getEntityDofMap();
                if (emap) {
                    coord_provider = [&](FieldId /*fid*/, GlobalIndex dof)
                        -> std::array<double, 3> {
                        auto ent = emap->getDofEntity(dof);
                        if (ent && ent->kind == dofs::EntityKind::Vertex) {
                            auto p = transient.system().meshAccess().getNodeCoordinates(ent->id);
                            return {static_cast<double>(p[0]),
                                    static_cast<double>(p[1]),
                                    static_cast<double>(p[2])};
                        }
                        return {0.0, 0.0, 0.0};
                    };
                }
                auto basis = reg->buildNullspaceBasis(n_dofs, get_field_dofs, coord_provider);
                linear.setNullspaceBasis(basis);
            }
        }

        // Provide the effective stage time step to the linear solver backend.
        //
        // For multi-stage schemes (e.g., generalized-α), `solve_time` may be a stage time
        // t_{n+α}. The legacy solver scales certain coupled-BC linearization terms by the
        // stage step (α*dt). Passing this here allows backends like FSILS to apply the same
        // scaling internally without coupling the FE library to specific physics.
        double dt_eff = base_state.dt;
        const double stage_dt = solve_time - history.time();
        if (std::isfinite(stage_dt) && stage_dt > 0.0) {
            dt_eff = stage_dt;
        }
        linear.setEffectiveTimeStep(dt_eff);
        if (oopTraceEnabled()) {
            traceLog("NewtonSolver: effective dt for linear backend=" + std::to_string(dt_eff));
        }

        const auto& bordered_full = transient.system().borderedCoupling();
        const bool has_bordered = bordered_full.active && bordered_full.n_aux > 0;
        const auto& owned_dofs = transient.system().dofHandler().getPartition().locallyOwned();
        active_reduced_field_updates = effective_reduced_field_updates;
        grouped_bordered_field_couplings.clear();
        algebraic_aux_reduction = {};
        solve_bordered_ptr = has_bordered ? &bordered_full : nullptr;
        if (has_bordered &&
            buildAlgebraicAuxiliaryReduction(bordered_full, owned_dofs, algebraic_aux_reduction)) {
            if (!algebraic_aux_reduction.promoted_rank_one_updates.empty()) {
                effective_rank_one_updates.insert(effective_rank_one_updates.end(),
                                                  algebraic_aux_reduction.promoted_rank_one_updates.begin(),
                                                  algebraic_aux_reduction.promoted_rank_one_updates.end());
            }
            if (!algebraic_aux_reduction.reduced_field_updates.empty() ||
                !algebraic_aux_reduction.grouped_couplings.empty()) {
                auto reduced_updates = algebraic_aux_reduction.reduced_field_updates;
                auto grouped_couplings = algebraic_aux_reduction.grouped_couplings;
                const int base_group_id =
                    static_cast<int>(grouped_bordered_field_couplings.size());
                rebaseGroupedCouplingIds(reduced_updates, grouped_couplings, base_group_id);
                active_reduced_field_updates.insert(active_reduced_field_updates.end(),
                                                   std::make_move_iterator(reduced_updates.begin()),
                                                   std::make_move_iterator(reduced_updates.end()));
                grouped_bordered_field_couplings.insert(grouped_bordered_field_couplings.end(),
                                                        std::make_move_iterator(grouped_couplings.begin()),
                                                        std::make_move_iterator(grouped_couplings.end()));
            }
            if (algebraic_aux_reduction.reduced_bordered.active &&
                algebraic_aux_reduction.reduced_bordered.n_aux > 0) {
                solve_bordered_ptr = &algebraic_aux_reduction.reduced_bordered;
            } else {
                solve_bordered_ptr = nullptr;
            }
            if (oopTraceEnabled()) {
                std::ostringstream oss;
                const double g_norm = std::sqrt(std::inner_product(
                    bordered_full.g.begin(), bordered_full.g.end(), bordered_full.g.begin(), 0.0));
                const double rhs_shift_norm = std::sqrt(std::inner_product(
                    algebraic_aux_reduction.rhs_shift.begin(),
                    algebraic_aux_reduction.rhs_shift.end(),
                    algebraic_aux_reduction.rhs_shift.begin(),
                    0.0));
                oss << "NewtonSolver: algebraic auxiliary reduction"
                    << " full_n_aux=" << bordered_full.n_aux
                    << " alg=" << algebraic_aux_reduction.algebraic_indices.size()
                    << " dyn=" << algebraic_aux_reduction.dynamic_indices.size()
                    << " direct_records=" << bordered_full.direct_coupling_records.size()
                    << " promoted_rank1=" << algebraic_aux_reduction.promoted_rank_one_updates.size()
                    << " reduced_updates=" << algebraic_aux_reduction.reduced_field_updates.size()
                    << " grouped_couplings=" << algebraic_aux_reduction.grouped_couplings.size()
                    << " ||g||=" << g_norm
                    << " ||B D^{-1} g||=" << rhs_shift_norm;
                traceLog(oss.str());
            }
        }
        const bool has_solve_bordered =
            solve_bordered_ptr != nullptr &&
            solve_bordered_ptr->active &&
            solve_bordered_ptr->n_aux > 0;
        bool condensed_bordered_active = false;
        std::vector<Real> condensed_rhs_shift;
        std::vector<Real> condensed_Dinv;
        std::vector<Real> condensed_DinvC;
        if (has_solve_bordered &&
            linear.supportsNativeReducedFieldUpdates() &&
            !has_non_dirichlet_affine_constraints) {
            const auto& solve_bordered = *solve_bordered_ptr;
            const auto nf = solve_bordered.n_field_dofs;
            const auto na = static_cast<std::size_t>(solve_bordered.n_aux);
            int max_condensed_aux = 64;
            if (const char* env = std::getenv("SVMP_MAX_CONDENSED_AUX_SIZE")) {
                const int parsed = std::atoi(env);
                if (parsed >= 0) {
                    max_condensed_aux = parsed;
                }
            }

            if (na > 0 && static_cast<int>(na) <= max_condensed_aux &&
                solve_bordered.D.size() == na * na &&
                solve_bordered.B.size() == nf * na &&
                solve_bordered.Ct.size() == na * nf &&
                solve_bordered.g.size() == na &&
                invertDenseMatrix(solve_bordered.D, na, condensed_Dinv)) {
                condensed_bordered_active = true;
                condensed_rhs_shift.assign(nf, Real(0.0));
                condensed_DinvC.assign(na * nf, Real(0.0));
                const auto direct_ct_rows =
                    buildDirectCouplingCtRows(solve_bordered, &owned_dofs);

                std::vector<Real> Dinv_g(na, Real(0.0));
                for (std::size_t i = 0; i < na; ++i) {
                    for (std::size_t j = 0; j < na; ++j) {
                        Dinv_g[i] += condensed_Dinv[i * na + j] * solve_bordered.g[j];
                    }
                }
                for (std::size_t row = 0; row < nf; ++row) {
                    for (std::size_t j = 0; j < na; ++j) {
                        condensed_rhs_shift[row] += solve_bordered.B[row + nf * j] * Dinv_g[j];
                    }
                }
                for (std::size_t i = 0; i < na; ++i) {
                    for (std::size_t col = 0; col < nf; ++col) {
                        Real value = Real(0.0);
                        for (std::size_t j = 0; j < na; ++j) {
                            value += condensed_Dinv[i * na + j] * solve_bordered.Ct[j * nf + col];
                        }
                        condensed_DinvC[i * nf + col] = value;
                    }
                }

                backends::GroupedBorderedFieldCoupling bordered_group;
                bordered_group.grouped_coupling_id =
                    static_cast<int>(grouped_bordered_field_couplings.size());
                bordered_group.aux_matrix.assign(solve_bordered.D.begin(), solve_bordered.D.end());
                bordered_group.modes.reserve(na);
                for (std::size_t j = 0; j < na; ++j) {
                    backends::ReducedFieldUpdate upd;
                    upd.sigma = Real(-1.0);
                    upd.grouped_coupling_id = bordered_group.grouped_coupling_id;
                    upd.left.reserve(nf);
                    upd.right.reserve(owned_dofs.size());
                    backends::GroupedBorderedFieldCoupling::Mode mode;
                    mode.left.reserve(nf);
                    mode.right.reserve(owned_dofs.size());
                    for (std::size_t row = 0; row < nf; ++row) {
                        const Real val = solve_bordered.B[row + nf * j];
                        if (std::abs(val) > Real(1e-30) &&
                            owned_dofs.contains(static_cast<GlobalIndex>(row))) {
                            upd.left.emplace_back(static_cast<GlobalIndex>(row), val);
                            mode.left.emplace_back(static_cast<GlobalIndex>(row), val);
                        }
                    }

                    std::unordered_map<GlobalIndex, Real> upd_right_accum;
                    upd_right_accum.reserve(owned_dofs.size());
                    for (std::size_t i = 0; i < na; ++i) {
                        const Real coeff = condensed_Dinv[j * na + i];
                        if (std::abs(coeff) <= Real(1e-30)) {
                            continue;
                        }

                        if (direct_ct_rows.row_covered[i]) {
                            for (const auto& [dof, value] : direct_ct_rows.rows[i]) {
                                upd_right_accum[dof] += coeff * value;
                            }
                            continue;
                        }

                        const auto row_offset = i * nf;
                        for (const auto dof : owned_dofs) {
                            const auto dof_idx = static_cast<std::size_t>(dof);
                            if (dof_idx >= nf) {
                                continue;
                            }
                            const Real c_val = solve_bordered.Ct[row_offset + dof_idx];
                            if (std::abs(c_val) > Real(1e-30)) {
                                upd_right_accum[dof] += coeff * c_val;
                            }
                        }
                    }

                    if (direct_ct_rows.row_covered[j]) {
                        mode.right.insert(mode.right.end(),
                                          direct_ct_rows.rows[j].begin(),
                                          direct_ct_rows.rows[j].end());
                    } else {
                        for (const auto dof : owned_dofs) {
                            const auto dof_idx = static_cast<std::size_t>(dof);
                            if (dof_idx >= nf) {
                                continue;
                            }
                            const Real c_val = solve_bordered.Ct[j * nf + dof_idx];
                            if (std::abs(c_val) > Real(1e-30)) {
                                mode.right.emplace_back(dof, c_val);
                            }
                        }
                    }

                    for (const auto& [dof, value] : upd_right_accum) {
                        if (std::abs(value) > Real(1e-30)) {
                            upd.right.emplace_back(dof, value);
                        }
                    }
                    std::sort(upd.right.begin(), upd.right.end(),
                              [](const auto& a, const auto& b) { return a.first < b.first; });
                    // Keep condensed bordered modes in reduced/grouped form so
                    // the MPI path preserves the exact left/right factors.
                    active_reduced_field_updates.push_back(std::move(upd));
                    bordered_group.modes.push_back(std::move(mode));
                }
                if (!bordered_group.aux_matrix.empty() &&
                    !bordered_group.modes.empty()) {
                    // Preserve the condensed auxiliary block explicitly, even
                    // for rank-1/diagonal cases. The grouped bordered path
                    // lets BlockSchur apply the exact D block instead of only
                    // the pre-collapsed D^{-1}C factor.
                    grouped_bordered_field_couplings.push_back(std::move(bordered_group));
                }

                if (oopTraceEnabled()) {
                    std::ostringstream oss;
                    const auto covered_rows =
                        static_cast<std::size_t>(std::count(direct_ct_rows.row_covered.begin(),
                                                            direct_ct_rows.row_covered.end(),
                                                            true));
                    oss << "NewtonSolver: condensed bordered coupling"
                        << " n_aux=" << na
                        << " n_field_dofs=" << nf
                        << " direct_ct_rows=" << covered_rows
                        << " added_updates="
                        << (active_reduced_field_updates.size() -
                            effective_reduced_field_updates.size())
                        << " grouped_couplings="
                        << grouped_bordered_field_couplings.size();
                    traceLog(oss.str());
                }
            }
        }
        linear_has_live_bordered = has_solve_bordered && !condensed_bordered_active;
        const bool disable_local_condensed_recovery =
            std::getenv("SVMP_DISABLE_LOCAL_CONDENSED_RECOVERY") != nullptr;
        const bool use_local_condensed_recovery =
            !disable_local_condensed_recovery &&
            transient.system().hasLocalCondensedRecovery() &&
            !(has_solve_bordered && condensed_bordered_active);
        if (oopTraceEnabled() && transient.system().hasLocalCondensedRecovery()) {
            std::ostringstream oss;
            oss << "NewtonSolver: local condensed recovery"
                << " enabled=" << (use_local_condensed_recovery ? 1 : 0)
                << " disabled_by_env=" << (disable_local_condensed_recovery ? 1 : 0)
                << " condensed_bordered_active=" << (condensed_bordered_active ? 1 : 0)
                << " has_solve_bordered=" << (has_solve_bordered ? 1 : 0)
                << " rhs_shift_size=" << transient.system().lastLocalCondensedRhsShift().size();
            traceLog(oss.str());
        }

        const bool direct_only_outlet_updates_available =
            !force_explicit_rank_one_updates &&
            linear.supportsNativeRankOneUpdates() &&
            linear.supportsNativeReducedFieldUpdates() &&
            !has_non_dirichlet_affine_constraints &&
            !has_solve_bordered &&
            grouped_bordered_field_couplings.empty() &&
            (!effective_rank_one_updates.empty() || !active_reduced_field_updates.empty());
        if (direct_only_outlet_updates_available) {
            direct_only_outlet_jacobian_period =
                (base_jacobian_period > 1)
                    ? base_jacobian_period
                    : directOnlyOutletJacobianRebuildPeriod(
                          effective_rank_one_updates.size() + active_reduced_field_updates.size());
            const auto sigma_scale = directOnlyOutletJacobianScale(
                effective_rank_one_updates.size() + active_reduced_field_updates.size());
            if (std::abs(sigma_scale - static_cast<Real>(1.0)) > static_cast<Real>(1e-12)) {
                for (auto& update : effective_rank_one_updates) {
                    update.sigma *= sigma_scale;
                }
                for (auto& update : active_reduced_field_updates) {
                    update.sigma *= sigma_scale;
                }
                if (oopTraceEnabled()) {
                    std::ostringstream oss;
                    oss << "NewtonSolver: scaled direct-only outlet Jacobian updates"
                        << " factor=" << sigma_scale
                        << " rank_one=" << effective_rank_one_updates.size()
                        << " reduced=" << active_reduced_field_updates.size();
                    traceLog(oss.str());
                }
            }
            if (oopTraceEnabled() && direct_only_outlet_jacobian_period > 1) {
                std::ostringstream oss;
                oss << "NewtonSolver: using direct-only outlet Jacobian rebuild period="
                    << direct_only_outlet_jacobian_period;
                traceLog(oss.str());
            }
        } else {
            direct_only_outlet_jacobian_period = 1;
        }
        if (direct_only_outlet_updates_available) {
            if (const auto rel_trigger = explicitRankOneAfterRelativeResidualThreshold()) {
                const double r0 =
                    (report.residual_norm0 > 0.0 && std::isfinite(report.residual_norm0))
                        ? report.residual_norm0
                        : 0.0;
                const double rel_res =
                    (r0 > 0.0 && std::isfinite(current_residual_norm))
                        ? (current_residual_norm / r0)
                        : std::numeric_limits<double>::infinity();
                if (rel_res <= *rel_trigger) {
                    force_explicit_rank_one_updates = true;
                    if (oopTraceEnabled()) {
                        std::ostringstream oss;
                        oss << "NewtonSolver: forcing explicit matrix path for direct-only outlet updates"
                            << " after relative residual reached " << rel_res
                            << " (trigger=" << *rel_trigger << ")";
                        traceLog(oss.str());
                    }
                }
            }
        }

        // Bridge rank-1 / reduced updates from coupled BC assembly to the linear
        // solver only after any bordered condensation has augmented the active
        // reduced/grouped coupling sets for this Newton solve.
        const bool explicit_rank_one_in_matrix = bridgeRankOneUpdates();
        if (!constrained_dofs.empty()) {
            const bool exact_direct_coupling_in_matrix =
                transient.system().lastRankOneUpdates().empty() &&
                !bordered_full.direct_coupling_records.empty();
            if (explicit_rank_one_in_matrix || exact_direct_coupling_in_matrix) {
                reapplyConstrainedJacobianRows();
            }
        }

        const bool has_any_rank_one_updates =
            !effective_rank_one_updates.empty() || !active_reduced_field_updates.empty();
        const bool rank_one_corrections_assembled_in_matrix =
            force_explicit_rank_one_updates && has_any_rank_one_updates;
        const bool has_native_rank_one_updates =
            has_any_rank_one_updates &&
            linear.supportsNativeReducedFieldUpdates() &&
            linear.supportsNativeRankOneUpdates() &&
            !force_explicit_rank_one_updates &&
            !has_non_dirichlet_affine_constraints;
        const bool has_native_direct_face_only_updates =
            has_native_rank_one_updates &&
            (!has_solve_bordered || condensed_bordered_active) &&
            grouped_bordered_field_couplings.empty() &&
            !effective_rank_one_updates.empty() &&
            active_reduced_field_updates.empty();
        const bool has_native_direct_only_reduced_updates =
            has_native_rank_one_updates &&
            !has_solve_bordered &&
            grouped_bordered_field_couplings.empty() &&
            effective_rank_one_updates.empty() &&
            !active_reduced_field_updates.empty();
        const int native_direct_face_mode_count =
            has_native_direct_face_only_updates
                ? static_cast<int>(effective_rank_one_updates.size())
                : 0;
        const bool has_native_condensed_coupled_updates =
            has_native_rank_one_updates &&
            condensed_bordered_active &&
            !has_native_direct_face_only_updates;
        const bool needs_strict_coupled_solve_options =
            has_native_condensed_coupled_updates ||
            ((has_solve_bordered && !condensed_bordered_active) &&
             !has_native_rank_one_updates) ||
            (has_any_rank_one_updates &&
             !has_native_rank_one_updates &&
             !rank_one_corrections_assembled_in_matrix);
        const bool needs_validated_native_rank_one_options =
            has_native_rank_one_updates && !has_native_condensed_coupled_updates;
        std::vector<Real> aux_delta;
        std::vector<Real> solve_aux_delta;
        std::vector<Real> combined_reduced_rhs_shift;
        FsilsMatrixSnapshot fsils_matrix_snapshot;
        const auto reportMeetsRequestedLinearTarget =
            [](const backends::SolverReport& rep,
               const backends::SolverOptions& requested) -> bool {
                if (!std::isfinite(rep.initial_residual_norm) ||
                    !std::isfinite(rep.final_residual_norm)) {
                    return false;
                }
                const Real rhs_norm =
                    std::max<Real>(static_cast<Real>(rep.initial_residual_norm), static_cast<Real>(1e-30));
                const Real target = std::max(requested.abs_tol, requested.rel_tol * rhs_norm);
                return std::isfinite(static_cast<double>(target)) &&
                       rep.final_residual_norm <= static_cast<double>(target);
            };
        const auto reportMeetsRequestedLinearTargetWithinFactor =
            [](const backends::SolverReport& rep,
               const backends::SolverOptions& requested,
               const Real factor) -> bool {
                if (!(factor >= static_cast<Real>(1.0)) ||
                    !std::isfinite(rep.initial_residual_norm) ||
                    !std::isfinite(rep.final_residual_norm)) {
                    return false;
                }
                const Real rhs_norm =
                    std::max<Real>(static_cast<Real>(rep.initial_residual_norm), static_cast<Real>(1e-30));
                const Real target = std::max(requested.abs_tol, requested.rel_tol * rhs_norm);
                return std::isfinite(static_cast<double>(target)) &&
                       rep.final_residual_norm <= static_cast<double>(factor * target);
            };
        const auto reportMeetsNonlinearAbsoluteLinearFloor =
            [&](const backends::SolverReport& rep,
                const Real residual_fraction,
                const Real max_relative_residual) -> bool {
                const bool strict_coupled_validation_active =
                    mpiMultiTaskActive() &&
                    (needs_strict_coupled_solve_options ||
                     needs_validated_native_rank_one_options);
                if (!strict_coupled_validation_active ||
                    !(options_.abs_tolerance > 0.0) ||
                    !(residual_fraction > static_cast<Real>(0.0)) ||
                    !(max_relative_residual > static_cast<Real>(0.0)) ||
                    !std::isfinite(rep.final_residual_norm) ||
                    !std::isfinite(rep.relative_residual)) {
                    return false;
                }

                const Real nonlinear_floor =
                    static_cast<Real>(options_.abs_tolerance) * residual_fraction;
                return rep.final_residual_norm <= static_cast<double>(nonlinear_floor) &&
                       rep.relative_residual <= static_cast<double>(max_relative_residual);
            };

        int ptc_retries = 0;
        while (true) {
            SolverOptionsGuard bordered_solver_options_guard{linear, base_linear_options};
            if (needs_strict_coupled_solve_options) {
                linear.setOptions(makeBorderedSolveOptions(base_linear_options));
            } else if (needs_validated_native_rank_one_options) {
                std::optional<Real> direct_only_inner_rel_override;
                if (has_native_direct_only_reduced_updates) {
                    if (const auto rel_trigger = lateDirectOnlyReducedTighteningThreshold()) {
                        const double r0 =
                            (report.residual_norm0 > 0.0 && std::isfinite(report.residual_norm0))
                                ? report.residual_norm0
                                : 0.0;
                        const double rel_res =
                            (r0 > 0.0 && std::isfinite(current_residual_norm))
                                ? (current_residual_norm / r0)
                                : std::numeric_limits<double>::infinity();
                        if (rel_res <= *rel_trigger) {
                            direct_only_inner_rel_override =
                                lateDirectOnlyReducedInnerRelTol();
                            if (oopTraceEnabled()) {
                                std::ostringstream oss;
                                oss << "NewtonSolver: tightening late direct-only reduced inner solve"
                                    << " rel_res=" << rel_res
                                    << " trigger=" << *rel_trigger
                                    << " inner_rel=" << *direct_only_inner_rel_override;
                                traceLog(oss.str());
                            }
                        }
                    }
                }
                linear.setOptions(makeValidatedNativeRankOneSolveOptions(
                    base_linear_options,
                    native_direct_face_mode_count,
                    direct_only_inner_rel_override));
            }
            if (has_solve_bordered && !condensed_bordered_active) {
                fsils_matrix_snapshot = captureFsilsMatrixSnapshot(J);
            }
            const std::vector<Real>* reduced_rhs_shift = nullptr;
            if (condensed_bordered_active) {
                reduced_rhs_shift = &condensed_rhs_shift;
            } else if (algebraic_aux_reduction.active &&
                       !has_solve_bordered &&
                       !algebraic_aux_reduction.rhs_shift.empty()) {
                reduced_rhs_shift = &algebraic_aux_reduction.rhs_shift;
            }
            const auto local_condensed_rhs_shift =
                use_local_condensed_recovery
                    ? transient.system().lastLocalCondensedRhsShift()
                    : std::span<const Real>{};
            if (!local_condensed_rhs_shift.empty()) {
                if (reduced_rhs_shift != nullptr) {
                    combined_reduced_rhs_shift = *reduced_rhs_shift;
                } else {
                    combined_reduced_rhs_shift.assign(local_condensed_rhs_shift.begin(),
                                                      local_condensed_rhs_shift.end());
                }
                if (combined_reduced_rhs_shift.size() < local_condensed_rhs_shift.size()) {
                    combined_reduced_rhs_shift.resize(local_condensed_rhs_shift.size(), Real(0.0));
                }
                for (std::size_t row = 0; row < local_condensed_rhs_shift.size(); ++row) {
                    combined_reduced_rhs_shift[row] += local_condensed_rhs_shift[row];
                }
                reduced_rhs_shift = &combined_reduced_rhs_shift;
            }

            backends::GenericVector* linear_rhs = &r;
            if (reduced_rhs_shift != nullptr) {
                copyVector(residual_scratch, r);
                auto rhs_view = residual_scratch.createAssemblyView();
                FE_CHECK_NOT_NULL(rhs_view.get(), "NewtonSolver: condensed rhs view");
                rhs_view->beginAssemblyPhase();
                for (std::size_t row = 0; row < reduced_rhs_shift->size(); ++row) {
                    const Real shift = (*reduced_rhs_shift)[row];
                    if (std::abs(shift) > Real(1e-30)) {
                        rhs_view->addVectorEntry(static_cast<GlobalIndex>(row),
                                                 -shift,
                                                 assembly::AddMode::Add);
                    }
                }
                rhs_view->finalizeAssembly();
                linear_rhs = &residual_scratch;
                if (oopTraceEnabled() && reduced_rhs_shift == &algebraic_aux_reduction.rhs_shift) {
                    traceLog("NewtonSolver: applied pure algebraic reduced RHS shift");
                }
            }
            if (oopTraceEnabled()) {
                traceLog("NewtonSolver: calling linear.solve()");
            }
            ntp0 = NTP();
            report.linear = linear.solve(J, du, *linear_rhs);
            ntp_linear += NTP() - ntp0;
            ntp_linear_iters_total += report.linear.iterations;
            if (oopTraceEnabled()) {
                std::ostringstream oss;
                oss << "NewtonSolver: post-linear.solve du_norm=" << du.norm()
                    << " basis_supported=" << (linear.supportsNullspace() ? 1 : 0);
                traceLog(oss.str());
            }
            normalizeFsilsPostSolveIncrementIfNeeded(du);
            if (oopTraceEnabled()) {
                std::ostringstream oss;
                oss << "NewtonSolver: post-normalize du_norm=" << du.norm();
                traceLog(oss.str());
            }
            if (newtonDirectionCheckEnabled()) {
                residual_scratch.zero();
                J.mult(du, residual_scratch);
                copyVector(residual_base, residual_scratch);
                if (has_native_rank_one_updates) {
                    addRankOneOperatorMatvec(
                        std::span<const backends::RankOneUpdate>(effective_rank_one_updates.data(),
                                                                 effective_rank_one_updates.size()),
                        du,
                        residual_scratch);
                    addReducedFieldOperatorMatvec(
                        std::span<const backends::ReducedFieldUpdate>(
                            active_reduced_field_updates.data(),
                            active_reduced_field_updates.size()),
                        du,
                        residual_scratch);
                }
                copyVector(u_backup, residual_scratch);
                axpy(u_backup, static_cast<Real>(-1.0), residual_base);
                zeroVectorEntries(constrained_dofs, residual_scratch);
                zeroVectorEntries(constrained_dofs, residual_base);
                zeroVectorEntries(constrained_dofs, u_backup);
                const double matrix_only_norm = residual_base.norm();
                const double rank_one_only_norm = u_backup.norm();

                copyVector(residual_base, *linear_rhs);
                zeroVectorEntries(constrained_dofs, residual_base);
                copyVector(u_backup, residual_scratch);
                axpy(u_backup, static_cast<Real>(-1.0), residual_base);
                const double matrix_minus_rhs_norm = u_backup.norm();

                const double jdu_norm = residual_scratch.norm();
                const double rhs_norm = residual_base.norm();

                auto dotVectors = [](const backends::GenericVector& a, const backends::GenericVector& b) {
                    return static_cast<double>(a.dot(b));
                };
                const double r_dot_jdu = dotVectors(residual_base, residual_scratch);
                const double r_dot_r = dotVectors(residual_base, residual_base);

                axpy(residual_scratch, static_cast<Real>(-1.0), residual_base);
                const double diff_norm = residual_scratch.norm();
                const double rel_diff = diff_norm / std::max(rhs_norm, 1e-30);

                if (mpiRank() == 0) {
                    std::ostringstream oss;
                    oss << "NewtonSolver: direction check"
                        << " native_rank_one=" << (has_native_rank_one_updates ? 1 : 0)
                        << " updates="
                        << (effective_rank_one_updates.size() + active_reduced_field_updates.size())
                        << " ||J_matrix du||=" << matrix_only_norm
                        << " ||J_matrix du-r||=" << matrix_minus_rhs_norm
                        << " ||rank1 du||=" << rank_one_only_norm
                        << " ||r||=" << rhs_norm
                        << " ||Jdu||=" << jdu_norm
                        << " ||Jdu-r||=" << diff_norm
                        << " rel=" << rel_diff
                        << " r_dot_Jdu=" << r_dot_jdu
                        << " r_dot_r=" << r_dot_r;
                    FE_LOG_INFO(oss.str());
                }
            }
            if (oopTraceEnabled()) {
                std::ostringstream oss;
                oss << "NewtonSolver: linear solve converged=" << report.linear.converged
                    << " iters=" << report.linear.iterations
                    << " r0=" << report.linear.initial_residual_norm
                    << " rn=" << report.linear.final_residual_norm
                    << " rel=" << report.linear.relative_residual
                    << " msg='" << report.linear.message << "'";
                traceLog(oss.str());
            }
            if (report.linear.converged) {
                break;
            }

            const bool meets_original_linear_target =
                (needs_strict_coupled_solve_options || needs_validated_native_rank_one_options) &&
                reportMeetsRequestedLinearTarget(report.linear, bordered_solver_options_guard.saved);
            const bool meets_nonlinear_linear_floor =
                reportMeetsNonlinearAbsoluteLinearFloor(report.linear,
                                                        static_cast<Real>(0.1),
                                                        static_cast<Real>(0.1));
            if (meets_original_linear_target || meets_nonlinear_linear_floor) {
                report.linear.converged = true;
                const Real rhs_norm =
                    std::max<Real>(static_cast<Real>(report.linear.initial_residual_norm),
                                   static_cast<Real>(1e-30));
                const Real target = std::max(bordered_solver_options_guard.saved.abs_tol,
                                             bordered_solver_options_guard.saved.rel_tol * rhs_norm);
                const char* acceptance_note =
                    meets_original_linear_target
                        ? "accepted original coupled target"
                        : "accepted nonlinear absolute floor";
                if (report.linear.message.empty()) {
                    report.linear.message = acceptance_note;
                } else if (report.linear.message.find(acceptance_note) == std::string::npos) {
                    report.linear.message += " (";
                    report.linear.message += acceptance_note;
                    report.linear.message += ")";
                }
                if (oopTraceEnabled()) {
                    std::ostringstream oss;
                    oss << "NewtonSolver: accepting coupled linear solution";
                    if (meets_original_linear_target) {
                        oss << " at original target";
                    } else {
                        oss << " at nonlinear absolute floor"
                            << " floor="
                            << static_cast<Real>(options_.abs_tolerance) *
                                   static_cast<Real>(0.1)
                            << " rel_limit=0.1";
                    }
                    oss << " rn=" << report.linear.final_residual_norm
                        << " target=" << target;
                    traceLog(oss.str());
                }
                break;
            }

            // Inexact Newton: accept the approximate solution even when the linear
            // solve doesn't fully converge. This matches the legacy solver behavior
            // where imprecise linear solutions still produce effective Newton steps.
            const bool allow_inexact_main_solve =
                options_.accept_inexact_linear_solutions &&
                !needs_strict_coupled_solve_options &&
                !needs_validated_native_rank_one_options;
            if (allow_inexact_main_solve) {
                if (oopTraceEnabled()) {
                    traceLog("NewtonSolver: accepting inexact linear solution (rel=" +
                             std::to_string(report.linear.relative_residual) + ")");
                }
                break;
            }

            const bool can_activate_ptc = ptc_can_run && options_.pseudo_transient.activate_on_linear_failure;
            if (!can_activate_ptc) {
                FE_THROW(FEException, "NewtonSolver: linear solve did not converge: " + report.linear.message);
            }

            // Lazily build the dt-only lumped diagonal when first needed.
            if (!ptc_mass_ready) {
                (void)assembleDtOnlyJacobianAndLumpedDiagonal(state);

                // Restore the physical Jacobian (dt-only assembly overwrote `J`).
                if (options_.assemble_both_when_possible && same_op) {
                    current_residual_norm = assembleJacobianAndResidual(state);
                    have_residual = true;
                    have_jacobian = true;
                    last_jacobian_it = it;
                } else {
                    current_residual_norm = assembleResidualOnly(state, /*phase=*/"ptc_restore");
                    have_residual = true;
                    assembleJacobianOnly(state);
                    ptc_gamma_applied = 0.0;
                    have_jacobian = true;
                    last_jacobian_it = it;
                }
                ptc_gamma_applied = 0.0;
            }

            // Increase diagonal dominance and retry.
            if (!(ptc_gamma > 0.0)) {
                ptc_gamma = (options_.pseudo_transient.gamma_initial > 0.0)
                                ? options_.pseudo_transient.gamma_initial
                                : 1.0;
            } else {
                ptc_gamma = std::min(ptc_gamma * options_.pseudo_transient.gamma_growth,
                                     options_.pseudo_transient.gamma_max);
            }

            if (oopTraceEnabled()) {
                std::ostringstream oss;
                oss << "NewtonSolver: PTC retry linear solve (gamma=" << ptc_gamma
                    << ", retry=" << (ptc_retries + 1) << "/" << options_.pseudo_transient.max_linear_retries << ")";
                traceLog(oss.str());
            }

            applyPtcDiagonalShift(ptc_gamma);

            ++ptc_retries;
            FE_THROW_IF(ptc_retries >= options_.pseudo_transient.max_linear_retries, FEException,
                        "NewtonSolver: linear solve did not converge (PTC retries exhausted): " + report.linear.message);
            du.zero();
        }

        if (has_solve_bordered && !condensed_bordered_active) {
            const auto& solve_bordered = *solve_bordered_ptr;
            const auto nf = solve_bordered.n_field_dofs;
            const auto na = static_cast<std::size_t>(solve_bordered.n_aux);
            FE_THROW_IF(nf != static_cast<std::size_t>(du.size()), systems::InvalidStateException,
                        "NewtonSolver: bordered PDE block size does not match solution size");
            FE_THROW_IF(solve_bordered.B.size() != nf * na ||
                            solve_bordered.Ct.size() != nf * na ||
                            solve_bordered.D.size() != na * na ||
                            solve_bordered.g.size() != na,
                        systems::InvalidStateException,
                        "NewtonSolver: bordered coupling storage size mismatch");

            auto gatherDenseVector = [&](backends::GenericVector& vec, std::size_t n) {
                return gatherGlobalDenseVectorFromOwnedEntries(vec, n, owned_dofs);
            };

            auto scatterDenseVector = [](backends::GenericVector& vec, std::span<const Real> dense) {
                auto view = vec.createAssemblyView();
                FE_CHECK_NOT_NULL(view.get(), "NewtonSolver: bordered dense scatter view");
                view->beginAssemblyPhase();
                for (std::size_t k = 0; k < dense.size(); ++k) {
                    view->addVectorEntry(static_cast<GlobalIndex>(k),
                                         dense[k],
                                         assembly::AddMode::Insert);
                }
                view->finalizeAssembly();
            };

            copyVector(residual_base, du);
            auto dense_du = gatherDenseVector(residual_base, nf);

            // In this branch the dynamic bordered block has not been condensed into
            // the main PDE operator. Native reduced/rank-one updates may still be
            // active for direct outlet coupling and algebraic auxiliary elimination,
            // but they do not include the dynamic -B D^{-1} C^T correction.
            // Recover the bordered Schur step explicitly from K_eff^{-1} B.
            const bool solve_already_includes_bordered_reduction = false;

            if (solve_already_includes_bordered_reduction) {
                std::vector<Real> dense_Dinv;
                FE_THROW_IF(!invertDenseMatrix(solve_bordered.D, na, dense_Dinv),
                            systems::InvalidStateException,
                            "NewtonSolver: bordered auxiliary recovery D inversion failed");

                const auto direct_ct_du =
                    projectCtDuFromDirectCouplingRecords(solve_bordered, dense_du);

                if (oopTraceEnabled()) {
                    std::size_t covered_rows = 0;
                    Real max_abs_diff = Real(0.0);
                    for (std::size_t i = 0; i < na; ++i) {
                        if (!direct_ct_du.row_covered[i]) {
                            continue;
                        }
                        ++covered_rows;
                        Real dense_row_value = Real(0.0);
                        for (std::size_t k = 0; k < nf; ++k) {
                            dense_row_value += solve_bordered.Ct[i * nf + k] * dense_du[k];
                        }
                        max_abs_diff = std::max(
                            max_abs_diff,
                            std::abs(direct_ct_du.values[i] - dense_row_value));
                    }
                    if (covered_rows > 0) {
                        std::ostringstream oss;
                        oss << "NewtonSolver: direct-record Ct projection"
                            << " covered_rows=" << covered_rows
                            << " max_abs_diff_vs_dense=" << max_abs_diff;
                        traceLog(oss.str());
                    }
                }

                std::vector<Real> aux_rhs(na, Real(0.0));
                for (std::size_t i = 0; i < na; ++i) {
                    Real value = solve_bordered.g[i];
                    if (direct_ct_du.row_covered[i]) {
                        value -= direct_ct_du.values[i];
                    } else {
                        for (std::size_t k = 0; k < nf; ++k) {
                            value -= solve_bordered.Ct[i * nf + k] * dense_du[k];
                        }
                    }
                    aux_rhs[i] = value;
                }

                solve_aux_delta.assign(na, Real(0.0));
                for (std::size_t i = 0; i < na; ++i) {
                    for (std::size_t j = 0; j < na; ++j) {
                        solve_aux_delta[i] += dense_Dinv[i * na + j] * aux_rhs[j];
                    }
                }
            } else {
                const auto u0 = dense_du;
                std::vector<Real> z_columns(nf * na, 0.0);

                {
                    SolverOptionsGuard bordered_solver_options_guard{linear, base_linear_options};
                    const auto bordered_recovery_options =
                        makeBorderedSolveOptions(base_linear_options);
                    linear.setOptions(bordered_recovery_options);

                    for (std::size_t j = 0; j < na; ++j) {
                        restoreFsilsMatrixSnapshot(J, fsils_matrix_snapshot);

                        residual_scratch.zero();
                        {
                            auto rhs_view = residual_scratch.createAssemblyView();
                            FE_CHECK_NOT_NULL(rhs_view.get(), "NewtonSolver: bordered rhs view");
                            rhs_view->beginAssemblyPhase();
                            for (std::size_t row = 0; row < nf; ++row) {
                                const Real bij = solve_bordered.B[row + nf * j];
                                if (std::abs(bij) > Real(1e-30)) {
                                    rhs_view->addVectorEntry(static_cast<GlobalIndex>(row),
                                                             bij,
                                                             assembly::AddMode::Add);
                                }
                            }
                            rhs_view->finalizeAssembly();
                        }

                        du.zero();
                        ntp0 = NTP();
                        const auto z_report = linear.solve(J, du, residual_scratch);
                        ntp_linear += NTP() - ntp0;
                        ntp_linear_iters_total += z_report.iterations;
                        normalizeFsilsPostSolveIncrementIfNeeded(du);
                        bool z_converged = z_report.converged;
                        if (!z_converged &&
                            has_native_rank_one_updates &&
                            reportMeetsRequestedLinearTarget(z_report, base_linear_options)) {
                            z_converged = true;
                            if (oopTraceEnabled()) {
                                const Real rhs_norm =
                                    std::max<Real>(static_cast<Real>(z_report.initial_residual_norm),
                                                   static_cast<Real>(1e-30));
                                const Real target =
                                    std::max(base_linear_options.abs_tol,
                                             base_linear_options.rel_tol * rhs_norm);
                                std::ostringstream oss;
                                oss << "NewtonSolver: accepting bordered K^{-1}B recovery at original target"
                                    << " rn=" << z_report.final_residual_norm
                                    << " target=" << target
                                    << " iters=" << z_report.iterations;
                                traceLog(oss.str());
                            }
                        } else if (!z_converged &&
                                   has_native_rank_one_updates &&
                                   mpiMultiTaskActive() &&
                                   reportMeetsRequestedLinearTargetWithinFactor(
                                       z_report, base_linear_options, static_cast<Real>(4.0))) {
                            z_converged = true;
                            if (oopTraceEnabled()) {
                                const Real rhs_norm =
                                    std::max<Real>(static_cast<Real>(z_report.initial_residual_norm),
                                                   static_cast<Real>(1e-30));
                                const Real target =
                                    std::max(base_linear_options.abs_tol,
                                             base_linear_options.rel_tol * rhs_norm);
                                std::ostringstream oss;
                                oss << "NewtonSolver: accepting bordered K^{-1}B recovery near target"
                                    << " rn=" << z_report.final_residual_norm
                                    << " target=" << target
                                    << " factor=4"
                                    << " iters=" << z_report.iterations;
                                traceLog(oss.str());
                            }
                        }
                        FE_THROW_IF(!z_converged, FEException,
                                    "NewtonSolver: bordered K^{-1}B solve did not converge: " +
                                        z_report.message);

                        const auto z_col = gatherDenseVector(du, nf);
                        for (std::size_t row = 0; row < nf; ++row) {
                            z_columns[j * nf + row] = z_col[row];
                        }

                        if (oopTraceEnabled()) {
                            const auto z_norm = std::sqrt(std::inner_product(
                                z_col.begin(), z_col.end(), z_col.begin(), Real(0.0)));
                            std::ostringstream oss;
                            oss << "NewtonSolver: bordered column " << j
                                << " ||K^{-1}B_j||=" << z_norm
                                << " iters=" << z_report.iterations;
                            traceLog(oss.str());
                        }
                    }
                }

                std::vector<Real> schur = solve_bordered.D;
                const auto direct_ct_u0 =
                    projectCtDuFromDirectCouplingRecords(solve_bordered, u0);
                for (std::size_t j = 0; j < na; ++j) {
                    const auto z_col =
                        std::span<const Real>(z_columns.data() +
                                                  static_cast<std::ptrdiff_t>(j * nf),
                                              nf);
                    const auto direct_ct_z =
                        projectCtDuFromDirectCouplingRecords(solve_bordered, z_col);
                    for (std::size_t i = 0; i < na; ++i) {
                        Real ctz = Real(0.0);
                        if (direct_ct_z.row_covered[i]) {
                            ctz = direct_ct_z.values[i];
                        } else {
                            for (std::size_t k = 0; k < nf; ++k) {
                                ctz += solve_bordered.Ct[i * nf + k] * z_columns[j * nf + k];
                            }
                        }
                        schur[i * na + j] -= ctz;
                    }
                }

                solve_aux_delta = solve_bordered.g;
                for (std::size_t i = 0; i < na; ++i) {
                    if (direct_ct_u0.row_covered[i]) {
                        solve_aux_delta[i] -= direct_ct_u0.values[i];
                    } else {
                        for (std::size_t k = 0; k < nf; ++k) {
                            solve_aux_delta[i] -= solve_bordered.Ct[i * nf + k] * u0[k];
                        }
                    }
                }

                FE_THROW_IF(!solveDenseLinearSystem(schur, solve_aux_delta),
                            systems::InvalidStateException,
                            "NewtonSolver: bordered Schur solve failed");

                for (std::size_t j = 0; j < na; ++j) {
                    const Real dxj = solve_aux_delta[j];
                    for (std::size_t k = 0; k < nf; ++k) {
                        dense_du[k] -= z_columns[j * nf + k] * dxj;
                    }
                }
                scatterDenseVector(du, dense_du);
            }

            aux_delta = recoverAuxiliaryDeltaFromReduction(
                algebraic_aux_reduction, dense_du, solve_aux_delta);

            if (oopTraceEnabled()) {
                const auto dx_norm = std::sqrt(std::inner_product(
                    solve_aux_delta.begin(), solve_aux_delta.end(), solve_aux_delta.begin(), Real(0.0)));
                std::ostringstream oss;
                oss << "NewtonSolver: bordered correction ||dx_aux||=" << dx_norm;
                traceLog(oss.str());

                J.mult(du, residual_base);
                const auto Kdu = gatherDenseVector(residual_base, nf);
                const auto rhs_dense = gatherDenseVector(r, nf);

                double pde_lin_res_sq = 0.0;
                for (std::size_t i = 0; i < nf; ++i) {
                    Real val = Kdu[i];
                    for (std::size_t j = 0; j < na; ++j) {
                        val += solve_bordered.B[i + nf * j] * solve_aux_delta[j];
                    }
                    const double rem = static_cast<double>(val - rhs_dense[i]);
                    pde_lin_res_sq += rem * rem;
                }

                double aux_lin_res_sq = 0.0;
                for (std::size_t i = 0; i < na; ++i) {
                    Real val = Real(0.0);
                    for (std::size_t k = 0; k < nf; ++k) {
                        val += solve_bordered.Ct[i * nf + k] * dense_du[k];
                    }
                    for (std::size_t j = 0; j < na; ++j) {
                        val += solve_bordered.D[i * na + j] * solve_aux_delta[j];
                    }
                    const double rem = static_cast<double>(val - solve_bordered.g[i]);
                    aux_lin_res_sq += rem * rem;
                }

                std::ostringstream lin_oss;
                lin_oss << "NewtonSolver: bordered linear residual"
                        << " pde=" << std::sqrt(pde_lin_res_sq)
                        << " aux=" << std::sqrt(aux_lin_res_sq)
                        << " mixed=" << std::sqrt(pde_lin_res_sq + aux_lin_res_sq);
                traceLog(lin_oss.str());
            }
        } else if (condensed_bordered_active) {
            const auto& solve_bordered = *solve_bordered_ptr;
            const auto nf = solve_bordered.n_field_dofs;
            const auto na = static_cast<std::size_t>(solve_bordered.n_aux);
            FE_THROW_IF(condensed_Dinv.size() != na * na ||
                            solve_bordered.Ct.size() != nf * na ||
                            solve_bordered.g.size() != na,
                        systems::InvalidStateException,
                        "NewtonSolver: condensed bordered storage size mismatch");

            const auto dense_du = gatherGlobalDenseVectorFromOwnedEntries(du, nf, owned_dofs);
            const auto direct_ct_du =
                projectCtDuFromDirectCouplingRecords(solve_bordered, dense_du);

            std::vector<Real> aux_rhs(na, Real(0.0));
            for (std::size_t i = 0; i < na; ++i) {
                Real value = solve_bordered.g[i];
                if (direct_ct_du.row_covered[i]) {
                    value -= direct_ct_du.values[i];
                } else {
                    for (std::size_t k = 0; k < nf; ++k) {
                        value -= solve_bordered.Ct[i * nf + k] * dense_du[k];
                    }
                }
                aux_rhs[i] = value;
            }

            solve_aux_delta.assign(na, Real(0.0));
            for (std::size_t i = 0; i < na; ++i) {
                for (std::size_t j = 0; j < na; ++j) {
                    solve_aux_delta[i] += condensed_Dinv[i * na + j] * aux_rhs[j];
                }
            }
            aux_delta = recoverAuxiliaryDeltaFromReduction(
                algebraic_aux_reduction, dense_du, solve_aux_delta);

            if (oopTraceEnabled()) {
                const auto dx_norm = std::sqrt(std::inner_product(
                    solve_aux_delta.begin(), solve_aux_delta.end(), solve_aux_delta.begin(), Real(0.0)));
                std::ostringstream oss;
                oss << "NewtonSolver: condensed bordered recovery ||dx_aux||=" << dx_norm;
                traceLog(oss.str());

                J.mult(du, residual_base);
                auto kdu_view = residual_base.createAssemblyView();
                FE_CHECK_NOT_NULL(kdu_view.get(), "NewtonSolver: condensed Kdu view");
                std::vector<Real> rhs_dense(nf, Real(0.0));
                auto rhs_view = r.createAssemblyView();
                FE_CHECK_NOT_NULL(rhs_view.get(), "NewtonSolver: condensed rhs gather view");
                for (std::size_t k = 0; k < nf; ++k) {
                    rhs_dense[k] = rhs_view->getVectorEntry(static_cast<GlobalIndex>(k));
                }

                double pde_lin_res_sq = 0.0;
                for (std::size_t i = 0; i < nf; ++i) {
                    Real val = kdu_view->getVectorEntry(static_cast<GlobalIndex>(i));
                    for (std::size_t j = 0; j < na; ++j) {
                        val += solve_bordered.B[i + nf * j] * solve_aux_delta[j];
                    }
                    const double rem = static_cast<double>(val - rhs_dense[i]);
                    pde_lin_res_sq += rem * rem;
                }

                double aux_lin_res_sq = 0.0;
                for (std::size_t i = 0; i < na; ++i) {
                    Real val = Real(0.0);
                    for (std::size_t k = 0; k < nf; ++k) {
                        val += solve_bordered.Ct[i * nf + k] * dense_du[k];
                    }
                    for (std::size_t j = 0; j < na; ++j) {
                        val += solve_bordered.D[i * na + j] * solve_aux_delta[j];
                    }
                    const double rem = static_cast<double>(val - solve_bordered.g[i]);
                    aux_lin_res_sq += rem * rem;
                }

                std::ostringstream lin_oss;
                lin_oss << "NewtonSolver: condensed bordered linear residual"
                        << " pde=" << std::sqrt(pde_lin_res_sq)
                        << " aux=" << std::sqrt(aux_lin_res_sq)
                        << " mixed=" << std::sqrt(pde_lin_res_sq + aux_lin_res_sq);
                traceLog(lin_oss.str());
            }
        } else if (algebraic_aux_reduction.active) {
            const auto dense_du =
                gatherGlobalDenseVectorFromOwnedEntries(du, static_cast<std::size_t>(J.numRows()), owned_dofs);
            aux_delta = recoverAuxiliaryDeltaFromReduction(
                algebraic_aux_reduction, dense_du, std::span<const Real>{});
        }

        auto applyDtIncrementScaling = [&]() {
            if (!options_.scale_dt_increments || workspace.dt_field_dofs.empty()) {
                return;
            }
            double factor = options_.dt_increment_scale;
            if (!(factor > 0.0)) {
                const auto* time_ctx = dt_scale_ctx ? &(*dt_scale_ctx) : nullptr;
                if (time_ctx && time_ctx->dt1) {
                    const double a0 = static_cast<double>(time_ctx->dt1->coeff(/*history_index=*/0));
                    if (std::isfinite(a0) && std::abs(a0) > 0.0) {
                        factor = 1.0 / a0;
                    }
                }
            }
            if (factor > 0.0 && std::isfinite(factor) && std::abs(factor - 1.0) > 0.0) {
                auto du_view = du.createAssemblyView();
                FE_CHECK_NOT_NULL(du_view.get(), "NewtonSolver: du scaling view");
                du_view->beginAssemblyPhase();
                for (const auto dof : workspace.dt_field_dofs) {
                    const Real v = du_view->getVectorEntry(dof);
                    du_view->addVectorEntry(dof, static_cast<Real>(factor) * v, assembly::AddMode::Insert);
                }
                du_view->finalizeAssembly();
                if (oopTraceEnabled()) {
                    std::ostringstream oss;
                    oss << "NewtonSolver: scaled dt increments by factor=" << factor
                        << " dofs=" << workspace.dt_field_dofs.size();
                    traceLog(oss.str());
                }
            }
        };

        applyDtIncrementScaling();

        if (oopTraceEnabled()) {
            logVectorComponentNorms(transient.system(), du, "du");
            logVectorTopEntries(transient.system(), du, "du", 8u);
        }

        const double du_norm = du.norm();
        auto gatherDenseFieldDelta = [&]() {
            auto du_view = du.createAssemblyView();
            FE_CHECK_NOT_NULL(du_view.get(), "NewtonSolver: dense du gather view");
            std::vector<Real> dense_du(static_cast<std::size_t>(du.size()), Real(0.0));
            for (std::size_t k = 0; k < dense_du.size(); ++k) {
                dense_du[k] = du_view->getVectorEntry(static_cast<GlobalIndex>(k));
            }
            return dense_du;
        };

        const bool use_line_search_this_iteration =
            options_.use_line_search ||
            (it == 0 &&
             has_native_direct_only_reduced_updates &&
             firstDirectOnlyReducedLineSearchEnabled());
        if (!options_.use_line_search && use_line_search_this_iteration && oopTraceEnabled()) {
            traceLog("NewtonSolver: enabling first-iteration line search for native direct-only reduced updates");
        }

        if (!use_line_search_this_iteration) {
            ntp0 = NTP();
            if (!aux_delta.empty()) {
                applyAuxiliaryDelta(transient.system(), bordered_full, aux_delta, static_cast<Real>(1.0));
            }
            if (use_local_condensed_recovery) {
                const auto dense_du = gatherDenseFieldDelta();
                transient.system().applyLocalCondensedRecovery(dense_du, static_cast<Real>(1.0));
            }
            axpy(history.u(), static_cast<Real>(-1.0), du);
            if (!constraints.empty()) {
                constraints.distribute(history.u());
            }
            history.u().updateGhosts();
            ntp_update += NTP() - ntp0;
            have_residual = false;

            if (options_.step_tolerance > 0.0) {
                if (oopTraceEnabled()) {
                    std::ostringstream oss;
                    oss << "NewtonSolver: step ||du||=" << du_norm << " step_tol=" << options_.step_tolerance;
                    traceLog(oss.str());
                }
                if (du_norm <= options_.step_tolerance) {
                    report.converged = true;
                    report.iterations = it + 1;
                    if (oopTraceEnabled()) {
                        traceLog("NewtonSolver: converged by step tolerance.");
                    }
                    printNewtonProfile(it + 1);
                    return report;
                }
            }
            continue;
        }

        // Backtracking line search: choose alpha in (0,1] so the residual norm decreases.
        copyVector(u_backup, history.u());
        const auto aux_state_backup =
            (aux_delta.empty() && !use_local_condensed_recovery)
                ? std::vector<Real>{}
                : transient.system().checkpointAuxiliaryState();
        const auto dense_du_for_aux =
            use_local_condensed_recovery
                ? gatherDenseFieldDelta()
                : std::vector<Real>{};
        const auto bordered_backup = transient.system().borderedCoupling();
        const double r_norm0 = current_residual_norm;
        const double r_norm0_sq = r_norm0 * r_norm0;
        auto evaluateLineSearchTrial = [&](double trial_alpha, const char* phase) -> double {
            copyVector(history.u(), u_backup);
            if (!aux_state_backup.empty()) {
                transient.system().restoreAuxiliaryState(aux_state_backup);
            }
            transient.system().borderedCoupling() = bordered_backup;
            if (!aux_state_backup.empty()) {
                applyAuxiliaryDelta(
                    transient.system(), bordered_full, aux_delta, static_cast<Real>(trial_alpha));
                if (!dense_du_for_aux.empty()) {
                    transient.system().applyLocalCondensedRecovery(
                        dense_du_for_aux, static_cast<Real>(trial_alpha));
                }
            }
            if (auto* reg = transient.system().auxiliaryInputRegistryIfPresent()) {
                reg->invalidateAll();
            }
            axpy(history.u(), static_cast<Real>(-trial_alpha), du);
            if (!constraints.empty()) {
                constraints.distribute(history.u());
            }
            history.u().updateGhosts();

            auto trial_state_holder = makeNewtonState(history, solve_time);
            return assembleResidualOnly(trial_state_holder.view, phase);
        };

        double alpha = 1.0;
        double last_tried_alpha = 0.0;
        double trial_norm = std::numeric_limits<double>::infinity();
        bool accepted = false;
        double best_alpha = 0.0;
        double best_trial_norm = std::numeric_limits<double>::infinity();
        bool have_best_trial = false;
        const int line_search_iteration_budget =
            std::max(1, options_.line_search_max_iterations);

        if (oopTraceEnabled()) {
            std::ostringstream oss;
            oss << "NewtonSolver: line search begin alpha=1"
                << " alpha_min=" << options_.line_search_alpha_min
                << " shrink=" << options_.line_search_shrink
                << " c1=" << options_.line_search_c1
                << " budget=" << line_search_iteration_budget;
            traceLog(oss.str());
        }

        for (int ls = 0; ls < line_search_iteration_budget; ++ls) {
            last_tried_alpha = alpha;
            trial_norm = evaluateLineSearchTrial(alpha, /*phase=*/"line_search");

            bool ok = false;
            if (std::isfinite(trial_norm) && std::isfinite(r_norm0)) {
                if (has_bordered) {
                    // For monolithically coupled auxiliary states, use the full
                    // nonlinear residual as the merit function and accept any
                    // trial that decreases it. This avoids PDE-centric Armijo
                    // rejection of otherwise good bordered Newton steps.
                    ok = (trial_norm <= r_norm0 * (1.0 + 1e-12));
                } else {
                    // Armijo on phi(u) = 0.5*||r(u)||^2 with Newton direction.
                    const double rhs = (1.0 - 2.0 * options_.line_search_c1 * alpha) * r_norm0_sq;
                    if (rhs > 0.0) {
                        ok = (trial_norm * trial_norm <= rhs);
                    } else {
                        ok = (trial_norm <= r_norm0);
                    }
                }
            }

            if (oopTraceEnabled()) {
                std::ostringstream oss;
                oss << "NewtonSolver: line search trial alpha=" << alpha
                    << " ||r(alpha)||=" << trial_norm
                    << " ok=" << (ok ? 1 : 0);
                traceLog(oss.str());
            }

            if (std::isfinite(trial_norm) && trial_norm < best_trial_norm) {
                best_trial_norm = trial_norm;
                best_alpha = alpha;
                have_best_trial = true;
            }

            if (ok) {
                accepted = true;
                break;
            }

            if (alpha <= options_.line_search_alpha_min) {
                break;
            }
            alpha *= options_.line_search_shrink;
            if (alpha < options_.line_search_alpha_min) {
                alpha = options_.line_search_alpha_min;
            }
        }

        bool reverted_to_original = false;
        if (!accepted) {
            const bool reached_alpha_min = last_tried_alpha <= options_.line_search_alpha_min;
            if (reached_alpha_min) {
                alpha = last_tried_alpha;
                if (oopTraceEnabled()) {
                    std::ostringstream oss;
                    oss << "NewtonSolver: line search reached alpha_min; keeping clamped trial alpha="
                        << alpha << " ||r(alpha)||=" << trial_norm;
                    traceLog(oss.str());
                }
            } else if (have_best_trial && std::isfinite(best_trial_norm) && best_trial_norm < r_norm0) {
                alpha = best_alpha;
                if (best_alpha != last_tried_alpha) {
                    trial_norm = evaluateLineSearchTrial(alpha, /*phase=*/"line_search_best");
                } else {
                    trial_norm = best_trial_norm;
                }
                if (oopTraceEnabled()) {
                    std::ostringstream oss;
                    oss << "NewtonSolver: line search did not satisfy Armijo; keeping best trial alpha="
                        << alpha << " ||r(alpha)||=" << trial_norm;
                    traceLog(oss.str());
                }
            } else {
                alpha = 0.0;
                trial_norm = evaluateLineSearchTrial(alpha, /*phase=*/"line_search_reject");
                reverted_to_original = true;
                if (oopTraceEnabled()) {
                    std::ostringstream oss;
                    oss << "NewtonSolver: line search did not reduce residual; reverting to original iterate"
                        << " ||r||=" << trial_norm;
                    traceLog(oss.str());
                }
            }
        }

        // `history.u` and `r` now correspond to the accepted trial, the best fallback
        // trial, or the restored original iterate.
        current_residual_norm = trial_norm;
        have_residual = std::isfinite(current_residual_norm);

        if (options_.step_tolerance > 0.0) {
            const double step_norm = reverted_to_original ? du_norm : (alpha * du_norm);
            if (oopTraceEnabled()) {
                std::ostringstream oss;
                oss << "NewtonSolver: step ||"
                    << (reverted_to_original ? "du" : "alpha*du")
                    << "||=" << step_norm
                    << " step_tol=" << options_.step_tolerance;
                traceLog(oss.str());
            }
            if (step_norm <= options_.step_tolerance) {
                report.converged = true;
                report.iterations = it + 1;
                report.residual_norm = current_residual_norm;
                if (oopTraceEnabled()) {
                    traceLog("NewtonSolver: converged by step tolerance.");
                }
                printNewtonProfile(it + 1);
                return report;
            }
        }

        if (!reverted_to_original &&
            tolerancesSatisfied(current_residual_norm, /*pre_first_update=*/false)) {
            report.converged = true;
            report.iterations = it + 1;
            report.residual_norm = current_residual_norm;
            if (oopTraceEnabled()) {
                traceLog("NewtonSolver: converged after line search update (tolerances satisfied).");
            }
            printNewtonProfile(it + 1);
            return report;
        }
    }

    // When line search is disabled, we do not evaluate the residual norm after applying the
    // last Newton update (we normally do it at the start of the next iteration). If we
    // exit due to reaching `max_it`, capture the final residual norm for reporting, but do
    // not override the explicit iteration limit with a late convergence declaration.
    if (!have_residual) {
        history.updateGhosts();
        if (!constraints.empty()) {
            constraints.distribute(history.u());
            history.u().updateGhosts();
        }
        auto final_state_holder = makeNewtonState(history, solve_time);
        current_residual_norm = assembleResidualOnly(
            final_state_holder.view, /*phase=*/"final_check");
        have_residual = true;
        report.residual_norm = current_residual_norm;
    }

    report.converged = false;
    report.iterations = max_it;
    if (have_residual && std::isfinite(current_residual_norm)) {
        report.residual_norm = current_residual_norm;
    }
    if (oopTraceEnabled()) {
        traceLog("NewtonSolver: reached max iterations without convergence.");
    }
    printNewtonProfile(max_it);
    return report;
}

} // namespace timestepping
} // namespace FE
} // namespace svmp

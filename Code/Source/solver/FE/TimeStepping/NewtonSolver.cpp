/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "TimeStepping/NewtonSolver.h"

#include "Assembly/CutIntegrationContext.h"
#include "Backends/Interfaces/BackendFactory.h"
#include "Constraints/AffineConstraints.h"
#include "Constraints/GaugeDiagnostics.h"
#include "Constraints/GaugeRegistry.h"
#include "Constraints/SystemConstraints.h"
#include "Core/FEException.h"
#include "Core/Logger.h"
#include "Dofs/DofIndexSet.h"
#include "Dofs/EntityDofMap.h"
#include "Auxiliary/AuxiliaryOperatorRegistry.h"
#include "Auxiliary/AuxiliaryStateManager.h"
#include "Basis/BasisCache.h"
#include "Systems/SystemsExceptions.h"

#if defined(FE_HAS_FSILS)
#  include "Backends/FSILS/FsilsMatrix.h"
#  include "Backends/FSILS/FsilsVector.h"
#endif

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <exception>
#include <functional>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <span>
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

#if FE_HAS_MPI
using NewtonCommunicator = MPI_Comm;
#else
using NewtonCommunicator = int;
constexpr NewtonCommunicator kSerialNewtonCommunicator = 0;
#endif

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

[[nodiscard]] bool envBoolEnabled(const char* name) noexcept
{
    const char* env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') {
        return false;
    }
    std::string v(env);
    std::transform(v.begin(), v.end(), v.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return !(v == "0" || v == "false" || v == "off" || v == "no");
}

[[nodiscard]] bool newtonFieldResidualDiagnosticEnabled() noexcept
{
    static const bool enabled =
        envBoolEnabled("SVMP_NEWTON_FIELD_RESIDUAL_DIAGNOSTIC") ||
        envBoolEnabled("SVMP_ACTIVE_PRESSURE_RESIDUAL_DIAGNOSTIC") ||
        envBoolEnabled("SVMP_NS_PRESSURE_ROW_CONTRIBUTION_DIAGNOSTIC") ||
        envBoolEnabled("SVMP_PRESSURE_ROW_CONTRIBUTION_DIAGNOSTIC");
    return enabled;
}

/// Legacy comparison switch: stamp the current-time constraint values into
/// every TimeHistory state before the nonlinear solve (rewrites the committed
/// trajectory of time-dependent Dirichlet data and the injected rate slot).
[[nodiscard]] bool distributeConstraintsIntoHistoryRequested() noexcept
{
    static const bool enabled =
        envBoolEnabled("SVMP_DISTRIBUTE_CONSTRAINTS_INTO_HISTORY");
    return enabled;
}

/// Opt-out for the master-bearing (MPC) constraint-state distribution into
/// the solution and rate history at state-sync points (here) and at the
/// accepted-step boundary (TimeLoop). Unlike the full history stamping above,
/// the master-bearing form pulls slave values from the SAME vector's masters,
/// so it cannot rewrite the trajectory of time-dependent Dirichlet data; it
/// exists so DOFs that enter or leave interface-tracking MPC sets (e.g.
/// small-cut aggregation) never expose the free-vs-extension value jump to
/// the time-integration stencils as a 1/(gamma*dt)-scaled rate pulse.
[[nodiscard]] bool mpcStateDistributeDisabled() noexcept
{
    static const bool disabled =
        envBoolEnabled("SVMP_NO_MPC_STATE_DISTRIBUTE");
    return disabled;
}

[[nodiscard]] bool pressureRowContributionDiagnosticEnabled() noexcept
{
    static const bool enabled =
        envBoolEnabled("SVMP_NS_PRESSURE_ROW_CONTRIBUTION_DIAGNOSTIC") ||
        envBoolEnabled("SVMP_PRESSURE_ROW_CONTRIBUTION_DIAGNOSTIC");
    return enabled;
}

[[nodiscard]] bool freeSurfaceConservativeBalanceDiagnosticEnabled() noexcept
{
    // Preserve the same precedence as the Navier--Stokes registration gate so
    // an explicit primary setting of "0" cannot be overridden by an ambient
    // compatibility alias.  Do not cache this value: unit and integration
    // harnesses can scope environment variables within one process.
    constexpr std::array<const char*, 4> names{
        "SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC",
        "SVMP_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC",
        "SVMP_NS_FREE_SURFACE_EQUILIBRIUM_DIAGNOSTIC",
        "SVMP_FREE_SURFACE_EQUILIBRIUM_DIAGNOSTIC",
    };
    for (const char* name : names) {
        const char* value = std::getenv(name);
        if (value != nullptr && value[0] != '\0') {
            return envBoolEnabled(name);
        }
    }
    return false;
}

[[nodiscard]] bool
freeSurfaceConservativeBalanceDiagnosticEveryAssemblyRequested() noexcept
{
    // Every requested full diagnostic sample is assembled.  Qualification
    // runs may additionally force the revision/exact-load-guarded mixed pair
    // and LSQR cache off, so those expensive operations repeat for every
    // sample (including both halves of a line-search accepted refresh).
    constexpr std::array<const char*, 2> names{
        "SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC_EVERY_ASSEMBLY",
        "SVMP_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC_EVERY_ASSEMBLY",
    };
    for (const char* name : names) {
        const char* value = std::getenv(name);
        if (value != nullptr && value[0] != '\0') {
            return envBoolEnabled(name);
        }
    }
    return false;
}

inline constexpr std::string_view
    kFreeSurfacePressureRepresentabilityPairOperator{
        "equations_diagnostic_ns_free_surface_pressure_representability_pair"};

[[nodiscard]] bool newtonMatrixSupportDiagnosticRequested() noexcept
{
    static const bool enabled =
        envBoolEnabled("SVMP_NEWTON_MATRIX_SUPPORT_DIAGNOSTIC") ||
        std::getenv("SVMP_NEWTON_MATRIX_SUPPORT_SAMPLE_DOFS") != nullptr;
    return enabled;
}

[[nodiscard]] bool activePressureSupportRankDiagnosticRequested() noexcept
{
    static const bool enabled =
        envBoolEnabled("SVMP_ACTIVE_PRESSURE_SUPPORT_RANK_DIAGNOSTIC") ||
        envBoolEnabled("SVMP_ACTIVE_PRESSURE_SUPPORT_RANK_GUARD") ||
        envBoolEnabled("SVMP_ACTIVE_PRESSURE_SUPPORT_RANK_CLAMP");
    return enabled;
}

[[nodiscard]] bool activePressureSupportRankGuardEnabled() noexcept
{
    static const bool enabled =
        envBoolEnabled("SVMP_ACTIVE_PRESSURE_SUPPORT_RANK_GUARD");
    return enabled;
}

[[nodiscard]] bool activePressureSupportRankClampEnabled() noexcept
{
    static const bool enabled =
        envBoolEnabled("SVMP_ACTIVE_PRESSURE_SUPPORT_RANK_CLAMP");
    return enabled;
}

[[nodiscard]] bool activePressureGraphCompletionEnabled() noexcept
{
    static const bool enabled =
        envBoolEnabled("SVMP_ACTIVE_PRESSURE_GRAPH_COMPLETION");
    return enabled;
}

[[nodiscard]] bool activePressureUpdateSupportDiagnosticRequested() noexcept
{
    static const bool enabled =
        envBoolEnabled("SVMP_ACTIVE_PRESSURE_UPDATE_SUPPORT_DIAGNOSTIC");
    return enabled;
}

[[nodiscard]] int activePressureSupportRankAllowedZeroVelocityRows() noexcept
{
    static const int allowed = [] {
        const char* env =
            std::getenv("SVMP_ACTIVE_PRESSURE_SUPPORT_RANK_ALLOWED_ZERO_VELOCITY_ROWS");
        if (env == nullptr || env[0] == '\0') {
            return 0;
        }
        try {
            return std::max(0, std::stoi(env));
        } catch (const std::exception&) {
            return 0;
        }
    }();
    return allowed;
}

[[nodiscard]] double activePressureSupportRankTolerance() noexcept
{
    static const double tolerance = [] {
        const char* env =
            std::getenv("SVMP_ACTIVE_PRESSURE_SUPPORT_RANK_TOLERANCE");
        if (env == nullptr || env[0] == '\0') {
            return 1.0e-14;
        }
        try {
            const double parsed = std::stod(env);
            if (parsed >= 0.0 && std::isfinite(parsed)) {
                return parsed;
            }
        } catch (const std::exception&) {
        }
        return 1.0e-14;
    }();
    return tolerance;
}

[[nodiscard]] double activePressureSupportRankClampCouplingThreshold() noexcept
{
    static const double threshold = [] {
        const char* env =
            std::getenv("SVMP_ACTIVE_PRESSURE_SUPPORT_RANK_CLAMP_MAX_VELOCITY_ROW_SUM");
        if (env == nullptr || env[0] == '\0') {
            return activePressureSupportRankTolerance();
        }
        try {
            const double parsed = std::stod(env);
            if (parsed >= 0.0 && std::isfinite(parsed)) {
                return parsed;
            }
        } catch (const std::exception&) {
        }
        return activePressureSupportRankTolerance();
    }();
    return threshold;
}

[[nodiscard]] double activePressureSupportRankClampSelfThreshold() noexcept
{
    static const double threshold = [] {
        const char* env =
            std::getenv("SVMP_ACTIVE_PRESSURE_SUPPORT_RANK_CLAMP_MAX_PRESSURE_SELF_ROW_SUM");
        if (env == nullptr || env[0] == '\0') {
            env = std::getenv("SVMP_ACTIVE_PRESSURE_SUPPORT_RANK_CLAMP_MAX_SELF_ROW_SUM");
        }
        if (env == nullptr || env[0] == '\0') {
            return -1.0;
        }
        try {
            const double parsed = std::stod(env);
            if (parsed >= 0.0 && std::isfinite(parsed)) {
                return parsed;
            }
        } catch (const std::exception&) {
        }
        return -1.0;
    }();
    return threshold;
}

[[nodiscard]] double activePressureGraphCompletionCouplingThreshold() noexcept
{
    static const double threshold = [] {
        const char* env =
            std::getenv("SVMP_ACTIVE_PRESSURE_GRAPH_COMPLETION_MAX_VELOCITY_ROW_SUM");
        if (env == nullptr || env[0] == '\0') {
            return activePressureSupportRankTolerance();
        }
        try {
            const double parsed = std::stod(env);
            if (parsed >= 0.0 && std::isfinite(parsed)) {
                return parsed;
            }
        } catch (const std::exception&) {
        }
        return activePressureSupportRankTolerance();
    }();
    return threshold;
}

[[nodiscard]] double activePressureGraphCompletionSelfThreshold() noexcept
{
    static const double threshold = [] {
        const char* env =
            std::getenv("SVMP_ACTIVE_PRESSURE_GRAPH_COMPLETION_MAX_PRESSURE_SELF_ROW_SUM");
        if (env == nullptr || env[0] == '\0') {
            env = std::getenv("SVMP_ACTIVE_PRESSURE_GRAPH_COMPLETION_MAX_SELF_ROW_SUM");
        }
        if (env == nullptr || env[0] == '\0') {
            return -1.0;
        }
        try {
            const double parsed = std::stod(env);
            if (parsed >= 0.0 && std::isfinite(parsed)) {
                return parsed;
            }
        } catch (const std::exception&) {
        }
        return -1.0;
    }();
    return threshold;
}

[[nodiscard]] double activePressureGraphCompletionWeightScale() noexcept
{
    static const double scale = [] {
        const char* env =
            std::getenv("SVMP_ACTIVE_PRESSURE_GRAPH_COMPLETION_WEIGHT_SCALE");
        if (env == nullptr || env[0] == '\0') {
            return 1.0;
        }
        try {
            const double parsed = std::stod(env);
            if (parsed > 0.0 && std::isfinite(parsed)) {
                return parsed;
            }
        } catch (const std::exception&) {
        }
        return 1.0;
    }();
    return scale;
}

[[nodiscard]] double activePressureGraphCompletionMaxEdgeScale() noexcept
{
    static const double scale = [] {
        const char* env =
            std::getenv("SVMP_ACTIVE_PRESSURE_GRAPH_COMPLETION_MAX_EDGE_SCALE");
        if (env == nullptr || env[0] == '\0') {
            return 16.0;
        }
        try {
            const double parsed = std::stod(env);
            if (parsed >= 1.0 && std::isfinite(parsed)) {
                return parsed;
            }
        } catch (const std::exception&) {
        }
        return 16.0;
    }();
    return scale;
}

[[nodiscard]] int activePressureGraphCompletionMaxRows() noexcept
{
    static const int max_rows = [] {
        const char* env =
            std::getenv("SVMP_ACTIVE_PRESSURE_GRAPH_COMPLETION_MAX_ROWS");
        if (env == nullptr || env[0] == '\0') {
            return 512;
        }
        try {
            const int parsed = std::stoi(env);
            return parsed < 0 ? -1 : parsed;
        } catch (const std::exception&) {
            return 512;
        }
    }();
    return max_rows;
}

[[nodiscard]] int activePressureGraphCompletionMaxActiveNeighbors() noexcept
{
    static const int max_neighbors = [] {
        const char* env = std::getenv(
            "SVMP_ACTIVE_PRESSURE_GRAPH_COMPLETION_MAX_ACTIVE_NEIGHBORS");
        if (env == nullptr || env[0] == '\0') {
            return 64;
        }
        try {
            const int parsed = std::stoi(env);
            return parsed < 0 ? -1 : parsed;
        } catch (const std::exception&) {
            return 64;
        }
    }();
    return max_neighbors;
}

[[nodiscard]] int activePressureGraphCompletionPressureNeighborDepth() noexcept
{
    static const int depth = [] {
        const char* env = std::getenv(
            "SVMP_ACTIVE_PRESSURE_GRAPH_COMPLETION_PRESSURE_NEIGHBOR_DEPTH");
        if (env == nullptr || env[0] == '\0') {
            return 1;
        }
        try {
            const int parsed = std::stoi(env);
            return std::clamp(parsed, 1, 8);
        } catch (const std::exception&) {
            return 1;
        }
    }();
    return depth;
}

[[nodiscard]] int activePressureGraphCompletionMaxBalancePressureEdgeDegree() noexcept
{
    static const int max_degree = [] {
        const char* env = std::getenv(
            "SVMP_ACTIVE_PRESSURE_GRAPH_COMPLETION_MAX_BALANCE_PRESSURE_EDGE_DEGREE");
        if (env == nullptr || env[0] == '\0') {
            env = std::getenv(
                "SVMP_ACTIVE_PRESSURE_GRAPH_COMPLETION_MAX_BALANCE_PRESSURE_DEGREE");
        }
        if (env == nullptr || env[0] == '\0') {
            return 3;
        }
        try {
            const int parsed = std::stoi(env);
            return parsed < 0 ? -1 : parsed;
        } catch (const std::exception&) {
            return 3;
        }
    }();
    return max_degree;
}

[[nodiscard]] const std::string& activePressureGraphCompletionMode()
{
    static const std::string mode = [] {
        const char* env = std::getenv("SVMP_ACTIVE_PRESSURE_GRAPH_COMPLETION_MODE");
        if (env != nullptr && env[0] != '\0') {
            std::string parsed(env);
            std::transform(parsed.begin(), parsed.end(), parsed.begin(),
                           [](unsigned char c) {
                               return static_cast<char>(std::tolower(c));
                           });
            return parsed;
        }
        return std::string("cycle");
    }();
    return mode;
}

[[nodiscard]] double activePressureUpdateSupportWeakVelocityThreshold() noexcept
{
    static const double threshold = [] {
        const char* env =
            std::getenv("SVMP_ACTIVE_PRESSURE_UPDATE_SUPPORT_WEAK_VELOCITY_ROW_SUM");
        if (env == nullptr || env[0] == '\0') {
            return -1.0;
        }
        try {
            const double parsed = std::stod(env);
            if (parsed >= 0.0 && std::isfinite(parsed)) {
                return parsed;
            }
        } catch (const std::exception&) {
        }
        return -1.0;
    }();
    return threshold;
}

[[nodiscard]] double activePressureUpdateSupportWeakSelfThreshold() noexcept
{
    static const double threshold = [] {
        const char* env =
            std::getenv("SVMP_ACTIVE_PRESSURE_UPDATE_SUPPORT_WEAK_PRESSURE_SELF_ROW_SUM");
        if (env == nullptr || env[0] == '\0') {
            env = std::getenv("SVMP_ACTIVE_PRESSURE_UPDATE_SUPPORT_WEAK_SELF_ROW_SUM");
        }
        if (env == nullptr || env[0] == '\0') {
            return -1.0;
        }
        try {
            const double parsed = std::stod(env);
            if (parsed >= 0.0 && std::isfinite(parsed)) {
                return parsed;
            }
        } catch (const std::exception&) {
        }
        return -1.0;
    }();
    return threshold;
}

[[nodiscard]] int activePressureSupportRankSampleLimit() noexcept
{
    static const int limit = [] {
        const char* env =
            std::getenv("SVMP_ACTIVE_PRESSURE_SUPPORT_RANK_SAMPLE_LIMIT");
        if (env == nullptr || env[0] == '\0') {
            return 16;
        }
        try {
            const int parsed = std::stoi(env);
            return parsed < 0 ? -1 : parsed;
        } catch (const std::exception&) {
            return 16;
        }
    }();
    return limit;
}

[[nodiscard]] int activePressureUpdateSupportSampleLimit() noexcept
{
    static const int limit = [] {
        const char* env =
            std::getenv("SVMP_ACTIVE_PRESSURE_UPDATE_SUPPORT_SAMPLE_LIMIT");
        if (env == nullptr || env[0] == '\0') {
            return 12;
        }
        try {
            const int parsed = std::stoi(env);
            return parsed < 0 ? -1 : parsed;
        } catch (const std::exception&) {
            return 12;
        }
    }();
    return limit;
}

[[nodiscard]] int activePressureUpdateSupportActionSampleLimit() noexcept
{
    static const int limit = [] {
        const char* env =
            std::getenv("SVMP_ACTIVE_PRESSURE_UPDATE_SUPPORT_ACTION_SAMPLE_LIMIT");
        if (env == nullptr || env[0] == '\0') {
            return 4;
        }
        try {
            const int parsed = std::stoi(env);
            return parsed < 0 ? -1 : parsed;
        } catch (const std::exception&) {
            return 4;
        }
    }();
    return limit;
}

[[nodiscard]] const std::string& newtonFieldResidualDiagnosticFieldName()
{
    static const std::string field_name = [] {
        const char* env = std::getenv("SVMP_NEWTON_FIELD_RESIDUAL_FIELD");
        if (env != nullptr && env[0] != '\0') {
            return std::string(env);
        }
        return std::string("Pressure");
    }();
    return field_name;
}

[[nodiscard]] const std::string& activePressureSupportRankPressureFieldName()
{
    static const std::string field_name = [] {
        const char* env =
            std::getenv("SVMP_ACTIVE_PRESSURE_SUPPORT_RANK_PRESSURE_FIELD");
        if (env != nullptr && env[0] != '\0') {
            return std::string(env);
        }
        return std::string("Pressure");
    }();
    return field_name;
}

[[nodiscard]] const std::string& activePressureSupportRankCouplingFieldName()
{
    static const std::string field_name = [] {
        const char* env =
            std::getenv("SVMP_ACTIVE_PRESSURE_SUPPORT_RANK_COUPLING_FIELD");
        if (env != nullptr && env[0] != '\0') {
            return std::string(env);
        }
        return std::string("Velocity");
    }();
    return field_name;
}

std::vector<GlobalIndex> parseGlobalIndexList(std::string_view text)
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

[[nodiscard]] const std::vector<GlobalIndex>& newtonFieldResidualDiagnosticSampleDofs()
{
    static const std::vector<GlobalIndex> dofs = [] {
        const char* env = std::getenv("SVMP_NEWTON_FIELD_RESIDUAL_SAMPLE_DOFS");
        if (env == nullptr || env[0] == '\0') {
            env = std::getenv("SVMP_PRESSURE_ROW_CONTRIBUTION_SAMPLE_DOFS");
        }
        if (env == nullptr || env[0] == '\0') {
            return std::vector<GlobalIndex>{};
        }
        return parseGlobalIndexList(env);
    }();
    return dofs;
}

[[nodiscard]] const std::vector<GlobalIndex>& newtonMatrixSupportGlobalSampleDofs()
{
    static const std::vector<GlobalIndex> dofs = [] {
        const char* env = std::getenv("SVMP_NEWTON_MATRIX_SUPPORT_SAMPLE_DOFS");
        if (env == nullptr || env[0] == '\0') {
            return std::vector<GlobalIndex>{};
        }
        return parseGlobalIndexList(env);
    }();
    return dofs;
}

[[nodiscard]] const std::vector<GlobalIndex>& newtonMatrixSupportPressureLocalSampleDofs()
{
    static const std::vector<GlobalIndex> dofs = [] {
        const char* env = std::getenv("SVMP_ACTIVE_PRESSURE_CONSTRAINT_SAMPLE_DOFS");
        if (env == nullptr || env[0] == '\0') {
            return std::vector<GlobalIndex>{};
        }
        return parseGlobalIndexList(env);
    }();
    return dofs;
}

[[nodiscard]] const std::vector<GlobalIndex>& activePressureGraphCompletionExplicitBalanceGlobalDofs()
{
    static const std::vector<GlobalIndex> dofs = [] {
        const char* env = std::getenv(
            "SVMP_ACTIVE_PRESSURE_GRAPH_COMPLETION_BALANCE_GLOBAL_DOFS");
        if (env == nullptr || env[0] == '\0') {
            env = std::getenv(
                "SVMP_ACTIVE_PRESSURE_GRAPH_COMPLETION_EXPLICIT_BALANCE_GLOBAL_DOFS");
        }
        if (env == nullptr || env[0] == '\0') {
            env = std::getenv(
                "SVMP_ACTIVE_PRESSURE_GRAPH_COMPLETION_BALANCE_DOFS");
        }
        if (env == nullptr || env[0] == '\0') {
            return std::vector<GlobalIndex>{};
        }
        return parseGlobalIndexList(env);
    }();
    return dofs;
}

[[nodiscard]] bool pressureRowContributionMatrixDiagnosticEnabled() noexcept
{
    static const bool enabled =
        pressureRowContributionDiagnosticEnabled() &&
        (envBoolEnabled("SVMP_NS_PRESSURE_ROW_CONTRIBUTION_MATRIX_DIAGNOSTIC") ||
         envBoolEnabled("SVMP_PRESSURE_ROW_CONTRIBUTION_MATRIX_DIAGNOSTIC") ||
         envBoolEnabled("SVMP_NS_PRESSURE_ROW_CONTRIBUTION_MATRIX_SUMMARY_DIAGNOSTIC") ||
         envBoolEnabled("SVMP_PRESSURE_ROW_CONTRIBUTION_MATRIX_SUMMARY_DIAGNOSTIC") ||
         envBoolEnabled("SVMP_NS_DIRECT_PSPG_FORMULATION_CANDIDATE_DIAGNOSTIC") ||
         envBoolEnabled("SVMP_DIRECT_PSPG_FORMULATION_CANDIDATE_DIAGNOSTIC") ||
         std::getenv("SVMP_PRESSURE_ROW_CONTRIBUTION_SAMPLE_DOFS") != nullptr ||
         std::getenv("SVMP_NEWTON_MATRIX_SUPPORT_SAMPLE_DOFS") != nullptr ||
         std::getenv("SVMP_ACTIVE_PRESSURE_CONSTRAINT_SAMPLE_DOFS") != nullptr);
    return enabled;
}

[[nodiscard]] bool pressureRowContributionMatrixSummaryDiagnosticEnabled() noexcept
{
    static const bool enabled =
        pressureRowContributionDiagnosticEnabled() &&
        (envBoolEnabled("SVMP_NS_PRESSURE_ROW_CONTRIBUTION_MATRIX_SUMMARY_DIAGNOSTIC") ||
         envBoolEnabled("SVMP_PRESSURE_ROW_CONTRIBUTION_MATRIX_SUMMARY_DIAGNOSTIC"));
    return enabled;
}

[[nodiscard]] bool directPspgFormulationCandidateDiagnosticEnabled() noexcept
{
    static const bool enabled =
        pressureRowContributionDiagnosticEnabled() &&
        (envBoolEnabled("SVMP_NS_DIRECT_PSPG_FORMULATION_CANDIDATE_DIAGNOSTIC") ||
         envBoolEnabled("SVMP_DIRECT_PSPG_FORMULATION_CANDIDATE_DIAGNOSTIC"));
    return enabled;
}

[[nodiscard]] bool directPspgFormulationCandidateOpSelected(std::string_view op) noexcept
{
    return op == "equations_diagnostic_ns_vms_pspg_pressure_gradient";
}

[[nodiscard]] bool pressureRowContributionMatrixSummaryOpSelected(
    std::string_view op)
{
    const char* env =
        std::getenv("SVMP_NS_PRESSURE_ROW_CONTRIBUTION_MATRIX_SUMMARY_OPS");
    if (env == nullptr || env[0] == '\0') {
        env = std::getenv("SVMP_PRESSURE_ROW_CONTRIBUTION_MATRIX_SUMMARY_OPS");
    }
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

[[nodiscard]] std::vector<NewtonMatrixSupportFieldRange>
newtonMatrixSupportFieldRanges(const systems::FESystem& sys)
{
    std::vector<NewtonMatrixSupportFieldRange> ranges;
    for (const auto field : sys.unknownFieldIdsInDofMapOrder()) {
        const auto begin = sys.fieldDofOffset(field);
        const auto dofs = sys.fieldDofHandler(field).getNumDofs();
        if (begin < 0 || dofs <= 0) {
            continue;
        }
        ranges.push_back(NewtonMatrixSupportFieldRange{
            sys.fieldRecord(field).name,
            begin,
            begin + dofs,
        });
    }
    return ranges;
}

void addMatrixSupportFieldAbs(
    std::span<const NewtonMatrixSupportFieldRange> ranges,
    GlobalIndex dof,
    double abs_value,
    std::vector<double>& sums)
{
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        if (dof >= ranges[i].begin && dof < ranges[i].end) {
            sums[i] += abs_value;
            return;
        }
    }
}

[[nodiscard]] std::string formatMatrixSupportFieldSums(
    std::span<const NewtonMatrixSupportFieldRange> ranges,
    std::span<const double> sums)
{
    if (ranges.empty() || sums.empty()) {
        return "none";
    }
    std::ostringstream oss;
    oss << std::setprecision(17);
    const auto count = std::min(ranges.size(), sums.size());
    for (std::size_t i = 0; i < count; ++i) {
        if (i > 0u) {
            oss << '|';
        }
        oss << ranges[i].name << ':' << sums[i];
    }
    return oss.str();
}

[[nodiscard]] bool linearProbeDumpEnabled() noexcept
{
    static const bool enabled = [] {
        const char* env = std::getenv("SVMP_DEBUG_LINEAR_PROBE_DUMP");
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

[[nodiscard]] bool linearSolveHistoryEnabled() noexcept
{
    static const bool enabled = [] {
        const char* env = std::getenv("SVMP_DEBUG_LINEAR_SOLVE_HISTORY");
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

[[nodiscard]] int linearSolveHistoryMaxCalls() noexcept
{
    static const int max_calls = [] {
        const char* env = std::getenv("SVMP_DEBUG_LINEAR_SOLVE_HISTORY_MAX_CALLS");
        if (env == nullptr) {
            return -1;
        }
        char* end = nullptr;
        const long parsed = std::strtol(env, &end, 10);
        if (end == env) {
            return -1;
        }
        return static_cast<int>(parsed);
    }();
    return max_calls;
}

[[nodiscard]] bool linearSolveComponentNormsEnabled() noexcept
{
    static const bool enabled = [] {
        const char* env = std::getenv("SVMP_DEBUG_LINEAR_SOLVE_COMPONENT_NORMS");
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

[[nodiscard]] int linearSolveComponentNormsMaxNewtonIt() noexcept
{
    static const int max_it = [] {
        const char* env = std::getenv("SVMP_DEBUG_LINEAR_SOLVE_COMPONENT_NORMS_MAX_NEWTON_IT");
        if (env == nullptr) {
            return -1;
        }
        char* end = nullptr;
        const long parsed = std::strtol(env, &end, 10);
        if (end == env) {
            return -1;
        }
        return static_cast<int>(parsed);
    }();
    return max_it;
}

[[nodiscard]] bool linearSolveMemoryDiagnosticsEnabled() noexcept
{
    static const bool enabled = [] {
        const char* env = std::getenv("SVMP_LINEAR_SOLVE_MEMORY_DIAGNOSTICS");
        if (env == nullptr) {
            return false;
        }
        while (*env == ' ' || *env == '\t' || *env == '\n' || *env == '\r') {
            ++env;
        }
        return *env != '\0' && *env != '0';
    }();
    return enabled;
}

[[nodiscard]] bool newtonAssemblyDiagnosticsEnabled() noexcept
{
    // This diagnostic is deliberately scopeable by unit/integration
    // harnesses in the same process.  Caching the first lookup made a later
    // explicit enable silently ineffective and rendered diagnostic coverage
    // dependent on test/solve order.
    const char* env = std::getenv("SVMP_NEWTON_ASSEMBLY_DIAGNOSTICS");
    if (env == nullptr) {
        return false;
    }
    while (*env == ' ' || *env == '\t' || *env == '\n' || *env == '\r') {
        ++env;
    }
    return *env != '\0' && *env != '0';
}

struct ProcessMemorySnapshot {
    long vm_kb{-1};
    long rss_kb{-1};
};

[[nodiscard]] ProcessMemorySnapshot readProcessMemorySnapshot()
{
    ProcessMemorySnapshot snapshot;
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        std::istringstream fields(line);
        std::string key;
        long value = -1;
        std::string unit;
        if (!(fields >> key >> value >> unit)) {
            continue;
        }
        if (key == "VmSize:") {
            snapshot.vm_kb = value;
        } else if (key == "VmRSS:") {
            snapshot.rss_kb = value;
        }
    }
    return snapshot;
}

[[nodiscard]] std::optional<std::string> firstLinearVectorDumpPrefix()
{
    static const std::optional<std::string> prefix = []() -> std::optional<std::string> {
        const char* env = std::getenv("SVMP_DEBUG_FIRST_LINEAR_VECTOR_DUMP_PREFIX");
        if (env == nullptr) {
            return std::nullopt;
        }
        while (*env == ' ' || *env == '\t' || *env == '\n' || *env == '\r') {
            ++env;
        }
        if (*env == '\0') {
            return std::nullopt;
        }
        return std::string(env);
    }();
    return prefix;
}

void traceLog(const std::string& msg)
{
    if (!oopTraceEnabled()) {
        return;
    }
    FE_LOG_INFO(msg);
}

[[nodiscard]] bool analysisTraceLogRequested()
{
    const char* env = std::getenv("SVMP_FE_ANALYSIS_LOG");
    if (env == nullptr) {
        return false;
    }
    std::string mode(env);
    std::transform(mode.begin(), mode.end(), mode.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return mode == "trace" || mode == "full";
}

void logPostTangentAnalysisReport(const systems::FESystem& system,
                                  bool numeric_summaries_updated)
{
    std::ostringstream oss;
    oss << "[FE/Analysis] Post-tangent analysis report"
        << " numeric_summaries="
        << (numeric_summaries_updated ? "updated" : "unavailable")
        << "\n";
    const auto& report = system.analysisReport();
    report.printApplicationLog(oss);
    if (analysisTraceLogRequested()) {
        report.printTraceLog(oss, system.latestAnalysisSummaries());
    }
    FE_LOG_INFO(oss.str());
}

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

[[nodiscard]] const char* stateSyncPointName(
    NewtonOptions::StateSynchronizationPoint point) noexcept
{
    using Point = NewtonOptions::StateSynchronizationPoint;
    switch (point) {
    case Point::OuterFixedPointState:
        return "outer_fixed_point_state";
    case Point::ProjectedOuterFixedPointState:
        return "projected_outer_fixed_point_state";
    case Point::EndpointCandidateState:
        return "endpoint_candidate_state";
    case Point::ProjectedEndpointCandidateState:
        return "projected_endpoint_candidate_state";
    case Point::AcceptedNonlinearState:
        return "accepted_nonlinear_state";
    case Point::ResidualAssembly:
        return "residual";
    case Point::JacobianAssembly:
        return "jacobian";
    case Point::JacobianAndResidualAssembly:
        return "jacobian_and_residual";
    case Point::LineSearchTrialResidual:
        return "line_search_trial";
    case Point::RestoredNonlinearState:
        return "restored_nonlinear_state";
    case Point::RestoredOuterFixedPointState:
        return "restored_outer_fixed_point_state";
    case Point::RestoredProjectedOuterFixedPointState:
        return "restored_projected_outer_fixed_point_state";
    case Point::RestoredTimeStepState:
        return "restored_time_step_state";
    case Point::RestoredProjectedTimeStepState:
        return "restored_projected_time_step_state";
    case Point::FinalResidualAssembly:
        return "final_residual";
    }
    return "unknown";
}

struct ConstraintSemanticFingerprint {
    std::uint64_t hash_a{1469598103934665603ULL};
    std::uint64_t hash_b{0x9e3779b97f4a7c15ULL};
    std::uint64_t line_count{0u};
    std::uint64_t entry_count{0u};

    [[nodiscard]] bool operator==(
        const ConstraintSemanticFingerprint&) const noexcept = default;
};

[[nodiscard]] ConstraintSemanticFingerprint constraintSemanticFingerprint(
    const constraints::AffineConstraints& affine_constraints)
{
    ConstraintSemanticFingerprint fingerprint;
    auto mix_word = [&fingerprint](std::uint64_t word) noexcept {
        fingerprint.hash_a ^= word;
        fingerprint.hash_a *= 1099511628211ULL;

        word ^= word >> 30;
        word *= 0xbf58476d1ce4e5b9ULL;
        word ^= word >> 27;
        word *= 0x94d049bb133111ebULL;
        word ^= word >> 31;
        fingerprint.hash_b =
            std::rotl(fingerprint.hash_b ^ word, 27) *
                0x9e3779b185ebca87ULL +
            0x632be59bd9b4e019ULL;
    };

    affine_constraints.forEach(
        [&](const constraints::AffineConstraints::ConstraintView& line) {
            mix_word(0xa0761d6478bd642fULL);
            mix_word(static_cast<std::uint64_t>(line.slave_dof));
            mix_word(std::bit_cast<std::uint64_t>(line.inhomogeneity));
            mix_word(static_cast<std::uint64_t>(line.entries.size()));
            ++fingerprint.line_count;
            for (const auto& entry : line.entries) {
                mix_word(static_cast<std::uint64_t>(entry.master_dof));
                mix_word(std::bit_cast<std::uint64_t>(entry.weight));
                ++fingerprint.entry_count;
            }
        });
    mix_word(fingerprint.line_count);
    mix_word(fingerprint.entry_count);
    return fingerprint;
}

[[nodiscard]] bool mpiMultiTaskActive(NewtonCommunicator communicator) noexcept
{
#if FE_HAS_MPI
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    MPI_Finalized(&finalized);
    if (!initialized || finalized || communicator == MPI_COMM_NULL) {
        return false;
    }

    int size = 1;
    MPI_Comm_size(communicator, &size);
    return size > 1;
#else
    (void)communicator;
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

[[nodiscard]] bool preserveGroupedAlgebraicDirectOnlyCouplings() noexcept
{
    const char* env = std::getenv("SVMP_PRESERVE_GROUPED_ALGEBRAIC_DIRECT_ONLY");
    if (env == nullptr) {
        return false;
    }
    while (*env == ' ' || *env == '\t' || *env == '\n' || *env == '\r') {
        ++env;
    }
    return *env != '\0' && *env != '0';
}

[[nodiscard]] bool pureAlgebraicBorderedRecoveryEnabled() noexcept
{
    const char* env = std::getenv("SVMP_PURE_ALGEBRAIC_BORDERED_RECOVERY");
    if (env == nullptr) {
        return false;
    }
    while (*env == ' ' || *env == '\t' || *env == '\n' || *env == '\r') {
        ++env;
    }
    return *env != '\0' && *env != '0';
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
[[nodiscard]] T mpiAllreduceSumIfActive(
    T value,
    NewtonCommunicator communicator) noexcept
{
#if FE_HAS_MPI
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    MPI_Finalized(&finalized);
    if (!initialized || finalized || communicator == MPI_COMM_NULL) {
        return value;
    }

    int size = 1;
    MPI_Comm_size(communicator, &size);
    if (size <= 1) {
        return value;
    }

    T global = value;
    if constexpr (std::is_same_v<T, int>) {
        MPI_Allreduce(&value, &global, 1, MPI_INT, MPI_SUM, communicator);
    } else {
        MPI_Allreduce(&value, &global, 1, MPI_DOUBLE, MPI_SUM, communicator);
    }
    return global;
#else
    (void)communicator;
    return value;
#endif
}

[[nodiscard]] bool tryPromoteReducedFieldUpdateToNativeRankOne(
    const backends::ReducedFieldUpdate& update,
    backends::RankOneUpdate& promoted,
    NewtonCommunicator communicator)
{
    if (update.grouped_coupling_id >= 0) {
        return false;
    }

    constexpr Real kTol = static_cast<Real>(1e-14);
    constexpr Real kRelTolSq = static_cast<Real>(1e-4);
    if (!(std::abs(update.sigma) > kTol)) {
        return false;
    }

    std::unordered_map<GlobalIndex, Real> q_map;
    q_map.reserve(update.right.size());
    Real q_norm_sq = Real(0.0);
    for (const auto& [dof, value] : update.right) {
        if (!(std::abs(value) > kTol)) {
            continue;
        }
        q_map[dof] += value;
    }
    for (const auto& [dof, value] : q_map) {
        (void)dof;
        q_norm_sq += value * value;
    }

    Real cross = Real(0.0);
    Real u_norm_sq = Real(0.0);
    Real local_residual_sq = Real(0.0);
    std::unordered_map<GlobalIndex, Real> u_map;
    u_map.reserve(update.left.size());
    for (const auto& [dof, value] : update.left) {
        if (!(std::abs(value) > kTol)) {
            continue;
        }
        u_map[dof] += value;
    }
    for (const auto& [dof, value] : u_map) {
        (void)dof;
        u_norm_sq += value * value;
        const auto it = q_map.find(dof);
        if (it != q_map.end()) {
            cross += value * it->second;
        }
    }

    const int global_q_has =
        mpiAllreduceSumIfActive(q_map.empty() ? 0 : 1, communicator);
    const int global_u_has =
        mpiAllreduceSumIfActive(u_map.empty() ? 0 : 1, communicator);
    const Real global_q_norm_sq =
        mpiAllreduceSumIfActive(q_norm_sq, communicator);
    const Real global_u_norm_sq =
        mpiAllreduceSumIfActive(u_norm_sq, communicator);
    const Real global_cross = mpiAllreduceSumIfActive(cross, communicator);
    if (global_q_has == 0 || global_u_has == 0 ||
        !(global_q_norm_sq > kTol * kTol) ||
        !(global_u_norm_sq > kTol * kTol)) {
        return false;
    }

    const Real proportionality = global_cross / global_q_norm_sq;
    if (!(std::abs(proportionality) > kTol)) {
        return false;
    }

    for (const auto& [dof, q_val] : q_map) {
        const auto it = u_map.find(dof);
        const Real u_val = (it != u_map.end()) ? it->second : Real(0.0);
        const Real diff = u_val - proportionality * q_val;
        local_residual_sq += diff * diff;
    }
    for (const auto& [dof, u_val] : u_map) {
        if (q_map.find(dof) == q_map.end()) {
            local_residual_sq += u_val * u_val;
        }
    }

    const Real residual_sq =
        mpiAllreduceSumIfActive(local_residual_sq, communicator);
    if (!(residual_sq / std::max(global_u_norm_sq, Real(1e-30)) <= kRelTolSq)) {
        return false;
    }

    promoted = {};
    promoted.sigma = update.sigma * proportionality;
    promoted.prefer_native_face = true;
    promoted.v = update.right;
    return true;
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

[[nodiscard]] dofs::IndexSet ownedDofSetForVector(
    const backends::GenericVector& vec,
    const dofs::IndexSet& fe_owned_dofs)
{
    return dofs::IndexSet(ownedDofsForVector(vec, fe_owned_dofs));
}

[[nodiscard]] std::vector<Real> gatherGlobalDenseVectorFromOwnedEntries(
    backends::GenericVector& vec,
    std::size_t n,
    const dofs::IndexSet& owned_dofs,
    NewtonCommunicator communicator)
{
    auto view = vec.createAssemblyView();
    FE_CHECK_NOT_NULL(view.get(), "NewtonSolver: global dense gather view");
    const auto vector_owned_dofs = ownedDofsForVector(vec, owned_dofs);

    std::vector<Real> local(n, Real(0.0));
    for (const auto dof : vector_owned_dofs) {
        const auto idx = static_cast<std::size_t>(dof);
        if (idx >= n) {
            continue;
        }
        local[idx] = view->getVectorEntry(dof);
    }

#if FE_HAS_MPI
    int mpi_initialized = 0;
    int mpi_finalized = 0;
    MPI_Initialized(&mpi_initialized);
    MPI_Finalized(&mpi_finalized);
    if (mpi_initialized && !mpi_finalized &&
        communicator != MPI_COMM_NULL && !local.empty()) {
        std::vector<Real> global(local.size(), Real(0.0));
        MPI_Allreduce(local.data(),
                      global.data(),
                      static_cast<int>(local.size()),
                      MPI_DOUBLE,
                      MPI_SUM,
                      communicator);
        return global;
    }
#else
    (void)communicator;
#endif

    return local;
}

void writeDenseVectorDump(const std::string& path, std::span<const Real> dense)
{
    std::ofstream out(path);
    FE_THROW_IF(!out.is_open(),
                systems::InvalidStateException,
                "NewtonSolver: failed to open linear vector dump file: " + path);
    out << std::setprecision(17) << std::scientific;
    for (std::size_t i = 0; i < dense.size(); ++i) {
        out << i << ' ' << dense[i] << '\n';
    }
}

struct ScalarFieldVertexDumpRecord {
    GlobalIndex monolithic_dof{INVALID_GLOBAL_INDEX};
    GlobalIndex vertex_id{INVALID_GLOBAL_INDEX};
    std::array<Real, 3> xyz{0.0, 0.0, 0.0};
    Real value{0.0};
};

[[nodiscard]] std::optional<std::size_t> selectPreferredScalarVertexDumpFieldIndex(
    const systems::FESystem& sys)
{
    const auto& fmap = sys.fieldMap();
    std::optional<std::size_t> first_scalar;
    for (std::size_t field_idx = 0; field_idx < fmap.numFields(); ++field_idx) {
        const auto& field = fmap.getField(field_idx);
        if (field.n_components != 1) {
            continue;
        }
        if (!first_scalar.has_value()) {
            first_scalar = field_idx;
        }
        std::string lower_name = field.name;
        std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (lower_name.find("pressure") != std::string::npos ||
            lower_name == "p") {
            return field_idx;
        }
    }
    return first_scalar;
}

[[nodiscard]] std::vector<ScalarFieldVertexDumpRecord> gatherScalarFieldVertexDumpRecords(
    const systems::FESystem& sys,
    backends::GenericVector& vec,
    std::size_t field_idx)
{
    const auto& fmap = sys.fieldMap();
    FE_THROW_IF(field_idx >= fmap.numFields(),
                systems::InvalidStateException,
                "NewtonSolver: scalar field vertex dump field index out of range");
    const auto& field = fmap.getField(field_idx);
    FE_THROW_IF(field.n_components != 1,
                systems::InvalidStateException,
                "NewtonSolver: scalar field vertex dump requires scalar field");

    auto view = vec.createAssemblyView();
    FE_CHECK_NOT_NULL(view.get(), "NewtonSolver: scalar field vertex dump view");

    const auto field_range = fmap.getFieldDofRange(field_idx);
    const auto& field_dh = sys.fieldDofHandler(static_cast<FieldId>(field_idx));
    const auto* emap = field_dh.getEntityDofMap();
    const auto& field_owned = field_dh.getPartition().locallyOwned();
    std::vector<ScalarFieldVertexDumpRecord> local_records;
    local_records.reserve(static_cast<std::size_t>(field_owned.size()));
    if (emap != nullptr) {
        for (const auto local_dof : field_owned) {
            const auto ent = emap->getDofEntity(local_dof);
            if (!ent || ent->kind != dofs::EntityKind::Vertex) {
                continue;
            }
            const GlobalIndex monolithic_dof = field_range.first + local_dof;
            const auto xyz = sys.meshAccess().getNodeCoordinates(ent->id);
            local_records.push_back(ScalarFieldVertexDumpRecord{
                .monolithic_dof = monolithic_dof,
                .vertex_id = ent->id,
                .xyz = xyz,
                .value = view->getVectorEntry(monolithic_dof),
            });
        }
    }

#if FE_HAS_MPI
    int mpi_initialized = 0;
    int mpi_finalized = 0;
    MPI_Initialized(&mpi_initialized);
    MPI_Finalized(&mpi_finalized);
    const auto communicator = sys.activeMpiCommunicator();
    if (mpi_initialized && !mpi_finalized && communicator != MPI_COMM_NULL) {
        constexpr int kPackedStride = 6;
        std::vector<double> local_packed(local_records.size() * kPackedStride, 0.0);
        for (std::size_t i = 0; i < local_records.size(); ++i) {
            const auto& rec = local_records[i];
            local_packed[kPackedStride * i + 0] = static_cast<double>(rec.monolithic_dof);
            local_packed[kPackedStride * i + 1] = static_cast<double>(rec.vertex_id);
            local_packed[kPackedStride * i + 2] = static_cast<double>(rec.xyz[0]);
            local_packed[kPackedStride * i + 3] = static_cast<double>(rec.xyz[1]);
            local_packed[kPackedStride * i + 4] = static_cast<double>(rec.xyz[2]);
            local_packed[kPackedStride * i + 5] = static_cast<double>(rec.value);
        }

        int mpi_size = 1;
        MPI_Comm_size(communicator, &mpi_size);
        const int root_rank = communicatorRank(communicator);
        std::vector<int> recv_counts(static_cast<std::size_t>(mpi_size), 0);
        const int local_count = static_cast<int>(local_packed.size());
        MPI_Gather(&local_count,
                   1,
                   MPI_INT,
                   recv_counts.data(),
                   1,
                   MPI_INT,
                   0,
                   communicator);

        std::vector<double> gathered;
        std::vector<int> displs;
        if (root_rank == 0) {
            displs.resize(static_cast<std::size_t>(mpi_size), 0);
            int offset = 0;
            for (int i = 0; i < mpi_size; ++i) {
                displs[static_cast<std::size_t>(i)] = offset;
                offset += recv_counts[static_cast<std::size_t>(i)];
            }
            gathered.resize(static_cast<std::size_t>(offset), 0.0);
        }

        MPI_Gatherv(local_packed.data(),
                    local_count,
                    MPI_DOUBLE,
                    gathered.data(),
                    recv_counts.data(),
                    displs.data(),
                    MPI_DOUBLE,
                    0,
                    communicator);

        if (root_rank != 0) {
            return {};
        }

        std::vector<ScalarFieldVertexDumpRecord> global_records;
        global_records.reserve(gathered.size() / kPackedStride);
        for (std::size_t i = 0; i + (kPackedStride - 1) < gathered.size(); i += kPackedStride) {
            global_records.push_back(ScalarFieldVertexDumpRecord{
                .monolithic_dof = static_cast<GlobalIndex>(std::llround(gathered[i + 0])),
                .vertex_id = static_cast<GlobalIndex>(std::llround(gathered[i + 1])),
                .xyz = {static_cast<Real>(gathered[i + 2]),
                        static_cast<Real>(gathered[i + 3]),
                        static_cast<Real>(gathered[i + 4])},
                .value = static_cast<Real>(gathered[i + 5]),
            });
        }
        return global_records;
    }
#endif

    return local_records;
}

void writeScalarFieldVertexDumpRecords(const std::string& path,
                                       std::string_view field_name,
                                       std::span<const ScalarFieldVertexDumpRecord> records)
{
    std::vector<ScalarFieldVertexDumpRecord> sorted(records.begin(), records.end());
    std::sort(sorted.begin(),
              sorted.end(),
              [](const ScalarFieldVertexDumpRecord& a, const ScalarFieldVertexDumpRecord& b) {
                  if (a.xyz[0] != b.xyz[0]) {
                      return a.xyz[0] < b.xyz[0];
                  }
                  if (a.xyz[1] != b.xyz[1]) {
                      return a.xyz[1] < b.xyz[1];
                  }
                  if (a.xyz[2] != b.xyz[2]) {
                      return a.xyz[2] < b.xyz[2];
                  }
                  if (a.vertex_id != b.vertex_id) {
                      return a.vertex_id < b.vertex_id;
                  }
                  return a.monolithic_dof < b.monolithic_dof;
              });

    std::ofstream out(path);
    FE_THROW_IF(!out.is_open(),
                systems::InvalidStateException,
                "NewtonSolver: failed to open scalar field vertex dump file: " + path);
    out << std::setprecision(17) << std::scientific;
    out << "# field " << field_name << '\n';
    out << "# monolithic_dof vertex_id x y z value\n";
    for (const auto& rec : sorted) {
        out << rec.monolithic_dof << ' '
            << rec.vertex_id << ' '
            << rec.xyz[0] << ' '
            << rec.xyz[1] << ' '
            << rec.xyz[2] << ' '
            << rec.value << '\n';
    }
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

enum class JacobianCheckDifferenceScheme {
    Forward,
    Central,
};

[[nodiscard]] JacobianCheckDifferenceScheme jacobianCheckDifferenceScheme() noexcept
{
    const char* env = std::getenv("SVMP_FE_JACOBIAN_CHECK_SCHEME");
    if (env == nullptr) {
        return JacobianCheckDifferenceScheme::Forward;
    }
    std::string value(env);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    value.erase(std::remove_if(value.begin(), value.end(), [](unsigned char c) {
                    return std::isspace(c) != 0 || c == '-' || c == '_';
                }),
                value.end());
    if (value == "central" || value == "centered" || value == "symmetric") {
        return JacobianCheckDifferenceScheme::Central;
    }
    return JacobianCheckDifferenceScheme::Forward;
}

[[nodiscard]] const char* jacobianCheckDifferenceSchemeName(
    JacobianCheckDifferenceScheme scheme) noexcept
{
    switch (scheme) {
    case JacobianCheckDifferenceScheme::Central:
        return "central";
    case JacobianCheckDifferenceScheme::Forward:
        break;
    }
    return "forward";
}

[[nodiscard]] const char* jacobianCheckGeometryModeName(
    JacobianCheckGeometryMode mode) noexcept
{
    switch (mode) {
    case JacobianCheckGeometryMode::FixedGeometry:
        return "fixed_geometry";
    case JacobianCheckGeometryMode::RefreshedGeometry:
        return "refreshed_geometry";
    case JacobianCheckGeometryMode::FullGeometryPerturbation:
        return "full_geometry_perturbation";
    }
    return "unknown";
}

[[nodiscard]] const char* jacobianCheckGeometryResult(
    JacobianCheckGeometryMode mode,
    double rel_error,
    double tolerance) noexcept
{
    const bool mismatch =
        std::isfinite(rel_error) && rel_error > std::max(tolerance, 0.0);
    switch (mode) {
    case JacobianCheckGeometryMode::FixedGeometry:
        return mismatch ? "fixed_geometry_tangent_mismatch"
                        : "fixed_geometry_tangent_match";
    case JacobianCheckGeometryMode::RefreshedGeometry:
        return mismatch ? "expected_quasi_newton_geometry_mismatch"
                        : "refreshed_geometry_tangent_match";
    case JacobianCheckGeometryMode::FullGeometryPerturbation:
        return "full_geometry_perturbation_unsupported";
    }
    return "unknown";
}

[[nodiscard]] std::string lowerCopy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

[[nodiscard]] std::string trimCopy(std::string value)
{
    const auto begin = std::find_if_not(value.begin(), value.end(), [](unsigned char c) {
        return std::isspace(c) != 0;
    });
    const auto end = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char c) {
        return std::isspace(c) != 0;
    }).base();
    if (begin >= end) {
        return {};
    }
    return std::string(begin, end);
}

using ComponentFilter = std::vector<std::string>;

[[nodiscard]] ComponentFilter parseJacobianCheckComponentFilter(std::string_view text)
{
    ComponentFilter parsed;
    std::stringstream ss{std::string(text)};
    std::string token;
    while (std::getline(ss, token, ',')) {
        token = lowerCopy(trimCopy(token));
        if (token == "all") {
            parsed.clear();
            return parsed;
        }
        if (!token.empty()) {
            parsed.push_back(token);
        }
    }
    return parsed;
}

[[nodiscard]] const ComponentFilter& jacobianCheckComponentFilter()
{
    static const ComponentFilter tokens = [] {
        const char* env = std::getenv("SVMP_FE_JACOBIAN_CHECK_COMPONENTS");
        if (env == nullptr) {
            return ComponentFilter{};
        }
        return parseJacobianCheckComponentFilter(env);
    }();
    return tokens;
}

[[nodiscard]] const std::vector<ComponentFilter>& jacobianCheckComponentSweepFilters()
{
    static const std::vector<ComponentFilter> filters = [] {
        std::vector<ComponentFilter> parsed;
        const char* env = std::getenv("SVMP_FE_JACOBIAN_CHECK_COMPONENT_SWEEPS");
        if (env != nullptr) {
            std::string sweep_text = trimCopy(env);
            const char separator = (sweep_text.find(';') != std::string::npos) ? ';' : ',';
            std::stringstream ss{sweep_text};
            std::string group;
            while (std::getline(ss, group, separator)) {
                group = trimCopy(group);
                if (group.empty()) {
                    continue;
                }
                parsed.push_back(parseJacobianCheckComponentFilter(group));
            }
        }
        if (parsed.empty()) {
            parsed.push_back(jacobianCheckComponentFilter());
        }
        return parsed;
    }();
    return filters;
}

[[nodiscard]] std::string jacobianCheckComponentFilterLabel(std::span<const std::string> tokens)
{
    if (tokens.empty()) {
        return "all";
    }
    std::ostringstream oss;
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0u) {
            oss << ",";
        }
        oss << tokens[i];
    }
    return oss.str();
}

[[nodiscard]] std::string jacobianCheckComponentFilterLabel()
{
    return jacobianCheckComponentFilterLabel(jacobianCheckComponentFilter());
}

[[nodiscard]] std::string jacobianCheckComponentSweepSummary(
    const std::vector<ComponentFilter>& filters)
{
    std::ostringstream oss;
    for (std::size_t i = 0; i < filters.size(); ++i) {
        if (i > 0u) {
            oss << ";";
        }
        oss << jacobianCheckComponentFilterLabel(filters[i]);
    }
    return oss.str();
}

void fillJacobianCheckDirection(backends::GenericVector& direction,
                                std::size_t sweep_index)
{
    auto v = direction.localSpan();
    std::uint64_t s = 0x9e3779b97f4a7c15ULL ^
        (static_cast<std::uint64_t>(mpiRank() + 1) * 0xbf58476d1ce4e5b9ULL) ^
        (static_cast<std::uint64_t>(sweep_index + 1u) * 0x94d049bb133111ebULL);
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

void logJacobianCheckSweepPlan(const std::vector<ComponentFilter>& filters)
{
    if (mpiRank() != 0) {
        return;
    }
    FE_LOG_INFO("NewtonSolver: Jacobian check sweep plan diagnostic=jacobian_check_sweep_plan count=" +
                std::to_string(filters.size()) +
                " filters='" + jacobianCheckComponentSweepSummary(filters) + "'");
}

[[nodiscard]] bool jacobianCheckDofSelected(const systems::FESystem& sys,
                                            GlobalIndex dof,
                                            std::span<const std::string> tokens)
{
    if (tokens.empty()) {
        return true;
    }
    const auto comp = sys.fieldMap().getComponentOfDof(dof);
    if (!comp) {
        return false;
    }
    const auto field_idx = static_cast<std::size_t>(std::max(comp->first, 0));
    if (field_idx >= sys.fieldMap().numFields()) {
        return false;
    }
    const auto& field = sys.fieldMap().getField(field_idx);
    const auto field_name = lowerCopy(field.name);
    auto component_name = field_name;
    if (field.n_components > 1) {
        component_name += "[" + std::to_string(static_cast<int>(comp->second)) + "]";
    }
    const auto field_index = std::to_string(static_cast<int>(field_idx));
    const auto component_index =
        field_index + ":" + std::to_string(static_cast<int>(comp->second));
    for (const auto& token : tokens) {
        if (token == field_name || token == component_name ||
            token == field_index || token == component_index) {
            return true;
        }
    }
    return false;
}

void applyJacobianCheckComponentFilter(const systems::FESystem& sys,
                                       backends::GenericVector& direction,
                                       std::span<const std::string> tokens,
                                       std::string_view filter_label,
                                       std::size_t sweep_index)
{
    const auto communicator = systemCommunicator(sys);
    if (tokens.empty()) {
        if (communicatorRank(communicator) == 0) {
            FE_LOG_INFO("NewtonSolver: Jacobian check component filter diagnostic=jacobian_check_component_filter components='" +
                        std::string(filter_label) +
                        "' component_filter='" + std::string(filter_label) +
                        "' sweep=" + std::to_string(sweep_index) +
                        " selected_dofs=all zeroed_dofs=0");
        }
        return;
    }

    const auto owned_dofs =
        ownedDofsForVector(direction, sys.dofHandler().getPartition().locallyOwned());
    std::vector<GlobalIndex> zero_dofs;
    zero_dofs.reserve(owned_dofs.size());
    unsigned long long selected_count = 0u;
    for (const auto dof : owned_dofs) {
        if (jacobianCheckDofSelected(sys, dof, tokens)) {
            ++selected_count;
        } else {
            zero_dofs.push_back(dof);
        }
    }
    auto view = direction.createAssemblyView();
    FE_CHECK_NOT_NULL(view.get(), "NewtonSolver: Jacobian check component filter view");
    view->beginAssemblyPhase();
    if (!zero_dofs.empty()) {
        view->zeroVectorEntries(zero_dofs);
    }
    // A distributed assembly finalization can be collective even when this
    // rank has no selected entries to zero.
    view->finalizeAssembly();

    unsigned long long zeroed_count = static_cast<unsigned long long>(zero_dofs.size());
#if FE_HAS_MPI
    int mpi_initialized = 0;
    int mpi_finalized = 0;
    MPI_Initialized(&mpi_initialized);
    MPI_Finalized(&mpi_finalized);
    if (mpi_initialized && !mpi_finalized &&
        communicator != MPI_COMM_NULL) {
        unsigned long long global_selected = 0u;
        unsigned long long global_zeroed = 0u;
        const unsigned long long local_zeroed = zeroed_count;
        MPI_Allreduce(&selected_count,
                      &global_selected,
                      1,
                      MPI_UNSIGNED_LONG_LONG,
                      MPI_SUM,
                      communicator);
        MPI_Allreduce(&local_zeroed,
                      &global_zeroed,
                      1,
                      MPI_UNSIGNED_LONG_LONG,
                      MPI_SUM,
                      communicator);
        selected_count = global_selected;
        zeroed_count = global_zeroed;
    }
#endif

    if (communicatorRank(communicator) == 0) {
        FE_LOG_INFO("NewtonSolver: Jacobian check component filter diagnostic=jacobian_check_component_filter components='" +
                    std::string(filter_label) +
                    "' component_filter='" + std::string(filter_label) +
                    "' sweep=" + std::to_string(sweep_index) +
                    " selected_dofs=" + std::to_string(selected_count) +
                    " zeroed_dofs=" + std::to_string(zeroed_count));
    }
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
    bool changed = false;
    for (std::size_t i = 0; i < ys.size(); ++i) {
        const auto old = ys[i];
        ys[i] += alpha * xs[i];
        changed = changed || ys[i] != old;
    }
    if (changed) {
        y.markModified();
    }
}

void copyVector(backends::GenericVector& dst, const backends::GenericVector& src)
{
    auto d = dst.localSpan();
    auto s = src.localSpan();
    FE_CHECK_ARG(d.size() == s.size(), "NewtonSolver: copyVector size mismatch");
    bool changed = false;
    for (std::size_t i = 0; i < d.size(); ++i) {
        changed = changed || d[i] != s[i];
        d[i] = s[i];
    }
    if (changed) {
        dst.markModified();
    }
}

void recreatePressureRepresentabilityVectors(
    const backends::BackendFactory& factory,
    GlobalIndex n_dofs,
    NewtonWorkspace& workspace)
{
    FE_THROW_IF(
        n_dofs <= 0,
        systems::InvalidStateException,
        "NewtonSolver: pressure-representability vector layout has no DOFs");
    const auto create_vector = [&]() {
        auto vector = factory.createVector(n_dofs);
        FE_CHECK_NOT_NULL(
            vector.get(),
            "NewtonSolver pressure-representability vector");
        return vector;
    };

    // Construct the complete replacement set before releasing any old
    // vectors. Besides providing basic exception safety, this makes a layout
    // refresh observable and prevents an allocator from recycling an old
    // object address while the set is only partially rebuilt.
    auto load = create_vector();
    auto solution = create_vector();
    auto left_basis = create_vector();
    auto right_basis = create_vector();
    auto direction = create_vector();
    auto work = create_vector();
    auto residual = create_vector();
    auto normal_residual = create_vector();

    workspace.pressure_representability_load = std::move(load);
    workspace.pressure_representability_solution = std::move(solution);
    workspace.pressure_representability_left_basis =
        std::move(left_basis);
    workspace.pressure_representability_right_basis =
        std::move(right_basis);
    workspace.pressure_representability_direction = std::move(direction);
    workspace.pressure_representability_work = std::move(work);
    workspace.pressure_representability_residual = std::move(residual);
    workspace.pressure_representability_normal_residual =
        std::move(normal_residual);
}

struct PressureRepresentabilityLsqrResult {
    double residual_norm{std::numeric_limits<double>::quiet_NaN()};
    double relative_residual{std::numeric_limits<double>::quiet_NaN()};
    double normal_residual_norm{std::numeric_limits<double>::quiet_NaN()};
    double relative_normal_residual{
        std::numeric_limits<double>::quiet_NaN()};
    double pressure_norm{std::numeric_limits<double>::quiet_NaN()};
    int iterations{0};
    bool converged{false};
    bool breakdown{false};
};

/**
 * Solve min_p ||G p + f|| with Golub--Kahan LSQR from p=0.
 *
 * `pair` is the constrained, symmetric mixed operator [0,G;G^T,0].  Inputs
 * carrying pressure coefficients therefore produce G-actions, while inputs
 * carrying velocity residuals produce G^T-actions through the same generic
 * matrix multiply.  No normal matrix is formed and no backend storage is
 * inspected or globally gathered.
 */
PressureRepresentabilityLsqrResult solvePressureRepresentabilityLsqr(
    const backends::GenericMatrix& pair,
    const backends::GenericVector& load,
    backends::GenericVector& pressure,
    backends::GenericVector& left_basis,
    backends::GenericVector& right_basis,
    backends::GenericVector& direction,
    backends::GenericVector& work,
    backends::GenericVector& residual,
    backends::GenericVector& normal_residual,
    int max_iterations,
    const std::function<bool(bool)>& all_ranks)
{
    PressureRepresentabilityLsqrResult result;
    pressure.zero();
    left_basis.zero();
    right_basis.zero();
    direction.zero();
    work.zero();
    residual.zero();
    normal_residual.zero();

    const auto vector_is_finite = [&](const backends::GenericVector& vector) {
        const auto values = vector.localSpan();
        const bool local_finite = std::all_of(
            values.begin(), values.end(), [](Real value) {
                return std::isfinite(static_cast<double>(value));
            });
        return all_ranks(local_finite);
    };
    const auto finite_nonnegative = [&](double value) {
        return all_ranks(std::isfinite(value) && value >= 0.0);
    };

    const double load_norm = load.norm();
    if (!vector_is_finite(load) || !finite_nonnegative(load_norm)) {
        result.breakdown = true;
        return result;
    }

    auto evaluate_fresh_residual = [&]() {
        pressure.updateGhosts();
        pair.mult(pressure, residual);
        residual.markModified();
        axpy(residual, Real{1.0}, load);
        residual.updateGhosts();
        pair.mult(residual, normal_residual);
        normal_residual.markModified();

        result.residual_norm = residual.norm();
        result.normal_residual_norm = normal_residual.norm();
        result.pressure_norm = pressure.norm();
        return vector_is_finite(residual) &&
               vector_is_finite(normal_residual) &&
               vector_is_finite(pressure) &&
               finite_nonnegative(result.residual_norm) &&
               finite_nonnegative(result.normal_residual_norm) &&
               finite_nonnegative(result.pressure_norm);
    };

    if (!evaluate_fresh_residual()) {
        result.breakdown = true;
        return result;
    }
    const double initial_normal_residual_norm =
        result.normal_residual_norm;
    const double absolute_stationarity_tolerance =
        100.0 * std::numeric_limits<Real>::epsilon() *
        std::max(1.0, initial_normal_residual_norm);
    const double stationarity_tolerance =
        absolute_stationarity_tolerance +
        1.0e-10 * initial_normal_residual_norm;

    const auto refresh_relative_metrics = [&]() {
        result.relative_residual =
            load_norm > 0.0
                ? result.residual_norm / load_norm
                : (result.residual_norm <= absolute_stationarity_tolerance
                       ? 0.0
                       : std::numeric_limits<double>::infinity());
        result.relative_normal_residual =
            initial_normal_residual_norm > 0.0
                ? result.normal_residual_norm /
                      initial_normal_residual_norm
                : (result.normal_residual_norm <=
                           absolute_stationarity_tolerance
                       ? 0.0
                       : std::numeric_limits<double>::infinity());
    };
    refresh_relative_metrics();

    // If G^T f is already at the scale-aware roundoff floor, p=0 is the
    // least-squares stationarity certificate.  This includes loads wholly in
    // the orthogonal complement of range(G).
    const bool initially_stationary = all_ranks(
        result.normal_residual_norm <= absolute_stationarity_tolerance);
    if (initially_stationary) {
        result.converged = true;
        return result;
    }
    if (max_iterations <= 0 || !(load_norm > 0.0)) {
        return result;
    }

    // LSQR solves G p = -f.  The vectors remain monolithic, but the zero
    // diagonal blocks ensure the Golub--Kahan bases alternate exactly between
    // velocity and pressure subspaces.
    left_basis.copyFrom(load);
    left_basis.scale(static_cast<Real>(-1.0 / load_norm));
    left_basis.updateGhosts();
    pair.mult(left_basis, right_basis);
    right_basis.markModified();
    double alpha = right_basis.norm();
    if (!vector_is_finite(right_basis) || !finite_nonnegative(alpha) ||
        !(alpha > 0.0)) {
        result.breakdown = true;
        (void)evaluate_fresh_residual();
        refresh_relative_metrics();
        return result;
    }
    right_basis.scale(static_cast<Real>(1.0 / alpha));
    direction.copyFrom(right_basis);

    double rho_bar = alpha;
    double phi_bar = load_norm;
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        right_basis.updateGhosts();
        pair.mult(right_basis, work);
        work.markModified();
        axpy(work, static_cast<Real>(-alpha), left_basis);
        double beta = work.norm();
        bool recurrence_finite =
            vector_is_finite(work) && finite_nonnegative(beta);
        if (!recurrence_finite) {
            result.breakdown = true;
            break;
        }

        if (beta > 0.0) {
            left_basis.copyFrom(work);
            left_basis.scale(static_cast<Real>(1.0 / beta));
            left_basis.updateGhosts();
            pair.mult(left_basis, work);
            work.markModified();
            axpy(work, static_cast<Real>(-beta), right_basis);
            alpha = work.norm();
            recurrence_finite =
                vector_is_finite(work) && finite_nonnegative(alpha);
            if (!recurrence_finite) {
                result.breakdown = true;
                break;
            }
            if (alpha > 0.0) {
                right_basis.copyFrom(work);
                right_basis.scale(static_cast<Real>(1.0 / alpha));
            } else {
                right_basis.zero();
            }
        } else {
            left_basis.zero();
            right_basis.zero();
            alpha = 0.0;
        }

        const double rho = std::hypot(rho_bar, beta);
        const bool rotation_finite = all_ranks(
            std::isfinite(rho) && rho > 0.0 &&
            std::isfinite(rho_bar) && std::isfinite(phi_bar) &&
            std::isfinite(alpha));
        if (!rotation_finite) {
            result.breakdown = true;
            break;
        }
        const double cosine = rho_bar / rho;
        const double sine = beta / rho;
        const double theta = sine * alpha;
        rho_bar = -cosine * alpha;
        const double phi = cosine * phi_bar;
        phi_bar = sine * phi_bar;
        const double solution_scale = phi / rho;
        const double direction_scale = theta / rho;
        const bool coefficients_finite = all_ranks(
            std::isfinite(cosine) && std::isfinite(sine) &&
            std::isfinite(theta) && std::isfinite(rho_bar) &&
            std::isfinite(phi) && std::isfinite(phi_bar) &&
            std::isfinite(solution_scale) &&
            std::isfinite(direction_scale));
        if (!coefficients_finite) {
            result.breakdown = true;
            break;
        }

        axpy(pressure, static_cast<Real>(solution_scale), direction);
        direction.scale(static_cast<Real>(-direction_scale));
        axpy(direction, Real{1.0}, right_basis);
        result.iterations = iteration + 1;

        if (!evaluate_fresh_residual()) {
            result.breakdown = true;
            break;
        }
        refresh_relative_metrics();
        const bool stationary = all_ranks(
            result.normal_residual_norm <= stationarity_tolerance);
        if (stationary) {
            result.converged = true;
            break;
        }

        // Exact bidiagonal termination without stationarity is a genuine
        // numerical breakdown; it is not silently relabelled convergence.
        const bool exact_recurrence_terminated =
            all_ranks(alpha == 0.0 || beta == 0.0);
        if (exact_recurrence_terminated) {
            result.breakdown = true;
            break;
        }
    }

    // Do not rely on recurrence estimates for qualification.  Recompute both
    // r=Gp+f and G^T r from the assembled operator after the final update (or
    // after a breakdown) and publish only those fresh norms.
    if (!evaluate_fresh_residual()) {
        result.breakdown = true;
        result.converged = false;
    }
    refresh_relative_metrics();
    if (result.converged) {
        result.converged = all_ranks(
            !result.breakdown &&
            result.normal_residual_norm <= stationarity_tolerance);
    }
    return result;
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

    // PETSc-like owned-row FSILS vectors hold authoritative values on owner rows.
    // Norms only reduce owned rows, so additive overlap accumulation is not part
    // of convergence checks.
    return scratch_fs->norm();
#else
    return r.norm();
#endif
}

double auxiliaryResidualNormForConvergence(
    const systems::FESystem::BorderedCouplingData& bordered,
    NewtonCommunicator communicator)
{
    long double local_sq = 0.0L;
    if (bordered.active) {
        for (const auto v : bordered.g) {
            local_sq += static_cast<long double>(v) *
                        static_cast<long double>(v);
        }
    }

#if FE_HAS_MPI
    if (!bordered.globally_reduced) {
        int mpi_initialized = 0;
        int mpi_finalized = 0;
        MPI_Initialized(&mpi_initialized);
        MPI_Finalized(&mpi_finalized);
        if (mpi_initialized && !mpi_finalized &&
            communicator != MPI_COMM_NULL) {
            long double global_sq = 0.0L;
            MPI_Allreduce(&local_sq,
                          &global_sq,
                          1,
                          MPI_LONG_DOUBLE,
                          MPI_SUM,
                          communicator);
            local_sq = global_sq;
        }
    }
#else
    (void)communicator;
#endif

    return std::sqrt(static_cast<double>(local_sq));
}

struct ResidualNormComponents {
    double field{0.0};
    double auxiliary{0.0};

    [[nodiscard]] double combined() const noexcept
    {
        return std::hypot(field, auxiliary);
    }
};

struct FieldResidualNormDefinition {
    FieldId field{INVALID_FIELD_ID};
    GlobalIndex begin{0};
    GlobalIndex end{0};
};

struct FieldResidualNormSample {
    std::vector<double> norms{};
    std::vector<std::uint64_t> owned_dof_counts{};
};

[[nodiscard]] FieldResidualNormSample fieldResidualNormsForConvergence(
    const systems::FESystem& system,
    backends::GenericVector& residual,
    std::span<const FieldResidualNormDefinition> definitions)
{
    FieldResidualNormSample sample;
    sample.norms.assign(definitions.size(), 0.0);
    sample.owned_dof_counts.assign(definitions.size(), 0u);
    if (definitions.empty()) {
        return sample;
    }

    const auto owned_dofs = ownedDofsForVector(
        residual, system.dofHandler().getPartition().locallyOwned());
    auto view = residual.createAssemblyView();
    FE_CHECK_NOT_NULL(view.get(), "NewtonSolver: field residual convergence view");

    std::vector<double> local_squared_norms(definitions.size(), 0.0);
    std::vector<unsigned long long> local_counts(definitions.size(), 0ull);
    for (const auto dof : owned_dofs) {
        for (std::size_t i = 0; i < definitions.size(); ++i) {
            const auto& definition = definitions[i];
            if (dof < definition.begin || dof >= definition.end) {
                continue;
            }
            const double value = static_cast<double>(view->getVectorEntry(dof));
            local_squared_norms[i] += value * value;
            ++local_counts[i];
            break;
        }
    }

    std::vector<double> global_squared_norms = local_squared_norms;
    std::vector<unsigned long long> global_counts = local_counts;
#if FE_HAS_MPI
    int mpi_initialized = 0;
    int mpi_finalized = 0;
    MPI_Initialized(&mpi_initialized);
    MPI_Finalized(&mpi_finalized);
    if (mpi_initialized && !mpi_finalized) {
        const auto communicator = system.activeMpiCommunicator();
        if (communicator != MPI_COMM_NULL) {
            int communicator_size = 1;
            MPI_Comm_size(communicator, &communicator_size);
            if (communicator_size > 1) {
                FE_THROW_IF(definitions.size() >
                                static_cast<std::size_t>(std::numeric_limits<int>::max()),
                            systems::InvalidStateException,
                            "NewtonSolver: too many field residual convergence criteria for MPI reduction");
                const auto count = static_cast<int>(definitions.size());
                MPI_Allreduce(local_squared_norms.data(),
                              global_squared_norms.data(),
                              count,
                              MPI_DOUBLE,
                              MPI_SUM,
                              communicator);
                MPI_Allreduce(local_counts.data(),
                              global_counts.data(),
                              count,
                              MPI_UNSIGNED_LONG_LONG,
                              MPI_SUM,
                              communicator);
            }
        }
    }
#endif

    for (std::size_t i = 0; i < definitions.size(); ++i) {
        sample.norms[i] = std::sqrt(std::max(0.0, global_squared_norms[i]));
        sample.owned_dof_counts[i] =
            static_cast<std::uint64_t>(global_counts[i]);
    }
    return sample;
}

[[nodiscard]] int activeSystemRank(const systems::FESystem& system) noexcept
{
    return communicatorRank(systemCommunicator(system));
}

[[nodiscard]] ResidualNormComponents borderedResidualNormComponentsForConvergence(
    const backends::GenericVector& r,
    backends::GenericVector& scratch,
    const systems::FESystem::BorderedCouplingData& bordered,
    NewtonCommunicator communicator)
{
    return ResidualNormComponents{
        residualNormForConvergence(r, scratch),
        auxiliaryResidualNormForConvergence(bordered, communicator)
    };
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

void syncOwnedRowHaloIfNeeded(backends::GenericVector& vec)
{
#if defined(FE_HAS_FSILS)
    if (auto* fs = dynamic_cast<backends::FsilsVector*>(&vec);
        fs != nullptr && fs->usesOwnedRowLayout()) {
        fs->updateGhosts();
    }
#else
    (void)vec;
#endif
}

enum class FsilsPostSolveSyncMode {
    Off,
    UpdateGhosts,
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

struct ComponentNormSnapshot {
    std::string label{};
    double norm{0.0};
};

std::vector<ComponentNormSnapshot> zeroComponentNormSnapshot(const systems::FESystem& sys)
{
    const auto& fmap = sys.fieldMap();
    std::vector<ComponentNormSnapshot> stats;
    stats.reserve(fmap.numFields() * 3u);
    for (std::size_t field_idx = 0; field_idx < fmap.numFields(); ++field_idx) {
        const auto& field = fmap.getField(field_idx);
        if (field.n_components <= 1) {
            stats.push_back(ComponentNormSnapshot{field.name, 0.0});
            continue;
        }
        for (LocalIndex comp = 0; comp < field.n_components; ++comp) {
            stats.push_back(ComponentNormSnapshot{
                field.name + "[" + std::to_string(static_cast<int>(comp)) + "]",
                0.0
            });
        }
    }
    return stats;
}

std::vector<ComponentNormSnapshot> componentNormSnapshot(const systems::FESystem& sys,
                                                         backends::GenericVector& vec)
{
    const auto& fmap = sys.fieldMap();
    auto stats = zeroComponentNormSnapshot(sys);
    const auto owned_dofs =
        ownedDofsForVector(vec, sys.dofHandler().getPartition().locallyOwned());
    if (fmap.numFields() == 0 || stats.empty()) {
        return stats;
    }

    std::vector<int> field_offsets(fmap.numFields(), -1);
    int offset = 0;
    for (std::size_t field_idx = 0; field_idx < fmap.numFields(); ++field_idx) {
        field_offsets[field_idx] = offset;
        offset += std::max(1, static_cast<int>(fmap.numComponents(field_idx)));
    }

    auto view = vec.createAssemblyView();
    FE_CHECK_NOT_NULL(view.get(), "NewtonSolver: component norm snapshot view");
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
        const double value = static_cast<double>(view->getVectorEntry(dof));
        stats[static_cast<std::size_t>(stat_idx)].norm += value * value;
    }

#if FE_HAS_MPI
    int mpi_initialized = 0;
    int mpi_finalized = 0;
    MPI_Initialized(&mpi_initialized);
    MPI_Finalized(&mpi_finalized);
    if (mpi_initialized && !mpi_finalized && !stats.empty()) {
        const auto communicator = sys.activeMpiCommunicator();
        std::vector<double> local(stats.size(), 0.0);
        std::vector<double> global(stats.size(), 0.0);
        for (std::size_t i = 0; i < stats.size(); ++i) {
            local[i] = stats[i].norm;
        }
        MPI_Allreduce(local.data(),
                      global.data(),
                      static_cast<int>(global.size()),
                      MPI_DOUBLE,
                      MPI_SUM,
                      communicator);
        for (std::size_t i = 0; i < stats.size(); ++i) {
            stats[i].norm = global[i];
        }
    }
#endif

    for (auto& stat : stats) {
        stat.norm = std::sqrt(std::max(0.0, stat.norm));
    }
    return stats;
}

void logJacobianCheckComponentDetails(
    NewtonCommunicator communicator,
    std::string_view component_filter,
    std::size_t sweep_index,
    std::span<const ComponentNormSnapshot> base,
    std::span<const ComponentNormSnapshot> perturbed,
    std::span<const ComponentNormSnapshot> fd,
    std::span<const ComponentNormSnapshot> matrix,
    std::span<const ComponentNormSnapshot> full,
    std::span<const ComponentNormSnapshot> matrix_err,
    std::span<const ComponentNormSnapshot> err,
    std::span<const ComponentNormSnapshot> sign_flip_err)
{
    if (communicatorRank(communicator) != 0) {
        return;
    }
    const auto count = std::min({
        base.size(),
        perturbed.size(),
        fd.size(),
        matrix.size(),
        full.size(),
        matrix_err.size(),
        err.size(),
        sign_flip_err.size()
    });
    if (count == 0u) {
        return;
    }

    std::ostringstream oss;
    oss << "NewtonSolver: Jacobian check component details"
        << " diagnostic=jacobian_check_component_details"
        << " component_filter='" << component_filter << "'"
        << " sweep=" << sweep_index;
    for (std::size_t i = 0; i < count; ++i) {
        oss << " [" << base[i].label
            << " base=" << base[i].norm
            << " perturbed=" << perturbed[i].norm
            << " fd=" << fd[i].norm
            << " matrix=" << matrix[i].norm
            << " full=" << full[i].norm
            << " matrix_err=" << matrix_err[i].norm
            << " total_err=" << err[i].norm
            << " sign_flip_err=" << sign_flip_err[i].norm << "]";
    }
    FE_LOG_INFO(oss.str());
}

void logJacobianCheckComponentBreakdown(const systems::FESystem& sys,
                                       backends::GenericVector& fd,
                                       backends::GenericVector& total_err,
                                       backends::GenericVector& matrix_err,
                                       std::string_view component_filter,
                                       std::size_t sweep_index)
{
    const auto& fmap = sys.fieldMap();
    const auto owned_dofs =
        ownedDofsForVector(fd, sys.dofHandler().getPartition().locallyOwned());
    if (fmap.numFields() == 0) {
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
    int mpi_finalized = 0;
    MPI_Initialized(&mpi_initialized);
    MPI_Finalized(&mpi_finalized);
    if (mpi_initialized && !mpi_finalized && !stats.empty()) {
        const auto communicator = sys.activeMpiCommunicator();
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
                      communicator);
        for (std::size_t i = 0; i < stats.size(); ++i) {
            stats[i].fd_sq = reduced[3u * i + 0u];
            stats[i].err_sq = reduced[3u * i + 1u];
            stats[i].matrix_err_sq = reduced[3u * i + 2u];
        }
    }
#endif

    if (communicatorRank(systemCommunicator(sys)) != 0) {
        return;
    }

    std::ostringstream oss;
    oss << "NewtonSolver: Jacobian check component norms"
        << " diagnostic=jacobian_check_component_norms"
        << " component_filter='" << component_filter << "'"
        << " sweep=" << sweep_index;
    for (const auto& s : stats) {
        oss << " [" << s.label
            << " fd=" << std::sqrt(std::max(0.0, s.fd_sq))
            << " total_err=" << std::sqrt(std::max(0.0, s.err_sq))
            << " matrix_err=" << std::sqrt(std::max(0.0, s.matrix_err_sq))
            << "]";
    }
    FE_LOG_INFO(oss.str());
}

void logJacobianCheckTopMismatchEntries(const systems::FESystem& sys,
                                        backends::GenericVector& fd,
                                        backends::GenericVector& err,
                                        std::size_t top_k,
                                        std::string_view component_filter,
                                        std::size_t sweep_index)
{
    const auto owned_dofs =
        ownedDofsForVector(err, sys.dofHandler().getPartition().locallyOwned());
    if (owned_dofs.empty() || top_k == 0u) {
        return;
    }

    auto fd_view = fd.createAssemblyView();
    auto err_view = err.createAssemblyView();
    FE_CHECK_NOT_NULL(fd_view.get(), "NewtonSolver: jacobian check top-mismatch fd view");
    FE_CHECK_NOT_NULL(err_view.get(), "NewtonSolver: jacobian check top-entry view");

    struct Entry {
        GlobalIndex dof{INVALID_GLOBAL_INDEX};
        double fd{0.0};
        double jv{0.0};
        double value{0.0};
    };

    std::vector<Entry> top_entries;
    top_entries.reserve(top_k);
    const auto maybe_insert = [&](GlobalIndex dof, double fd_value, double value) {
        const double abs_value = std::abs(value);
        if (!(abs_value > 0.0) || !std::isfinite(abs_value)) {
            return;
        }
        const double jv_value = fd_value + value;
        if (top_entries.size() < top_k) {
            top_entries.push_back(Entry{dof, fd_value, jv_value, value});
        } else {
            auto min_it = std::min_element(
                top_entries.begin(), top_entries.end(),
                [](const Entry& a, const Entry& b) { return std::abs(a.value) < std::abs(b.value); });
            if (min_it != top_entries.end() && abs_value > std::abs(min_it->value)) {
                *min_it = Entry{dof, fd_value, jv_value, value};
            }
        }
    };

    for (const auto dof : owned_dofs) {
        maybe_insert(dof,
                     static_cast<double>(fd_view->getVectorEntry(dof)),
                     static_cast<double>(err_view->getVectorEntry(dof)));
    }

    std::sort(top_entries.begin(), top_entries.end(),
              [](const Entry& a, const Entry& b) { return std::abs(a.value) > std::abs(b.value); });

    // Map each flagged dof to its mesh vertex and coordinates so mismatch
    // locations can be tied to the discretization (vertex-DOF fields only;
    // sub-vertex dofs report vertex=-1).
    std::unordered_map<GlobalIndex, GlobalIndex> dof_to_vertex;
    {
        const auto& fmap = sys.fieldMap();
        for (std::size_t field_idx = 0; field_idx < fmap.numFields(); ++field_idx) {
            const auto field_id = static_cast<FieldId>(field_idx);
            const auto offset = sys.fieldDofOffset(field_id);
            const auto* entity_map = sys.fieldDofHandler(field_id).getEntityDofMap();
            if (entity_map == nullptr) {
                continue;
            }
            const auto n_vertices = entity_map->numVertices();
            for (GlobalIndex vertex = 0; vertex < n_vertices; ++vertex) {
                for (const auto local_dof : entity_map->getVertexDofs(vertex)) {
                    dof_to_vertex.emplace(offset + local_dof, vertex);
                }
            }
        }
    }

    std::ostringstream oss;
    oss << "NewtonSolver: Jacobian check top mismatch entries"
        << " diagnostic=jacobian_check_top_mismatch"
        << " component_filter='" << component_filter << "'"
        << " sweep=" << sweep_index
        << " rank=" << mpiRank();
    for (const auto& entry : top_entries) {
        oss << " [" << describeFieldComponentDof(sys, entry.dof)
            << " fd=" << entry.fd
            << " jv=" << entry.jv
            << " err=" << entry.value;
        const auto it = dof_to_vertex.find(entry.dof);
        if (it != dof_to_vertex.end()) {
            const auto xyz = sys.meshAccess().getNodeCoordinates(it->second);
            oss << " vertex=" << it->second
                << " xyz=(" << xyz[0] << "," << xyz[1] << "," << xyz[2] << ")";
        } else {
            oss << " vertex=-1";
        }
        oss << "]";
    }
    FE_LOG_INFO(oss.str());
}

void logVectorComponentNorms(const systems::FESystem& sys,
                             backends::GenericVector& vec,
                             std::string_view label)
{
    const auto& fmap = sys.fieldMap();
    const auto owned_dofs =
        ownedDofsForVector(vec, sys.dofHandler().getPartition().locallyOwned());
    // Field layout is communicator-global, but a valid distributed rank may
    // own no vector row.  Such a rank must still contribute neutral values to
    // the component reductions entered by its peers.
    if (fmap.numFields() == 0) {
        return;
    }

    struct ComponentNorm {
        std::string label{};
        double sq_norm{0.0};
        double sum{0.0};
        double min_value{std::numeric_limits<double>::infinity()};
        double max_value{-std::numeric_limits<double>::infinity()};
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
        comp.min_value = std::min(comp.min_value, v);
        comp.max_value = std::max(comp.max_value, v);
        comp.count += 1;
    }

#if FE_HAS_MPI
    int mpi_initialized = 0;
    int mpi_finalized = 0;
    MPI_Initialized(&mpi_initialized);
    MPI_Finalized(&mpi_finalized);
    if (mpi_initialized && !mpi_finalized && !comps.empty()) {
        const auto comm = sys.activeMpiCommunicator();
        std::vector<double> local_norm(comps.size(), 0.0);
        std::vector<double> global_norm(comps.size(), 0.0);
        std::vector<double> local_sum(comps.size(), 0.0);
        std::vector<double> global_sum(comps.size(), 0.0);
        std::vector<double> local_min(comps.size(), std::numeric_limits<double>::infinity());
        std::vector<double> global_min(comps.size(), std::numeric_limits<double>::infinity());
        std::vector<double> local_max(comps.size(), -std::numeric_limits<double>::infinity());
        std::vector<double> global_max(comps.size(), -std::numeric_limits<double>::infinity());
        std::vector<unsigned long long> local_count(comps.size(), 0ull);
        std::vector<unsigned long long> global_count(comps.size(), 0ull);
        for (std::size_t i = 0; i < comps.size(); ++i) {
            local_norm[i] = comps[i].sq_norm;
            local_sum[i] = comps[i].sum;
            local_min[i] = comps[i].min_value;
            local_max[i] = comps[i].max_value;
            local_count[i] = static_cast<unsigned long long>(comps[i].count);
        }
        MPI_Allreduce(local_norm.data(),
                      global_norm.data(),
                      static_cast<int>(local_norm.size()),
                      MPI_DOUBLE,
                      MPI_SUM,
                      comm);
        MPI_Allreduce(local_sum.data(),
                      global_sum.data(),
                      static_cast<int>(local_sum.size()),
                      MPI_DOUBLE,
                      MPI_SUM,
                      comm);
        MPI_Allreduce(local_min.data(),
                      global_min.data(),
                      static_cast<int>(local_min.size()),
                      MPI_DOUBLE,
                      MPI_MIN,
                      comm);
        MPI_Allreduce(local_max.data(),
                      global_max.data(),
                      static_cast<int>(local_max.size()),
                      MPI_DOUBLE,
                      MPI_MAX,
                      comm);
        MPI_Allreduce(local_count.data(),
                      global_count.data(),
                      static_cast<int>(local_count.size()),
                      MPI_UNSIGNED_LONG_LONG,
                      MPI_SUM,
                      comm);
        for (std::size_t i = 0; i < comps.size(); ++i) {
            comps[i].sq_norm = global_norm[i];
            comps[i].sum = global_sum[i];
            comps[i].min_value = global_min[i];
            comps[i].max_value = global_max[i];
            comps[i].count = static_cast<std::uint64_t>(global_count[i]);
        }
    }
#endif

    int logging_rank = mpiRank();
#if FE_HAS_MPI
    if (mpi_initialized && !mpi_finalized) {
        MPI_Comm_rank(sys.activeMpiCommunicator(), &logging_rank);
    }
#endif
    if (logging_rank != 0) {
        return;
    }

    std::ostringstream oss;
    oss << "NewtonSolver: vector component norms"
        << " diagnostic=vector_component_norms"
        << " label='" << label << "'";
    for (const auto& c : comps) {
        const double mean = (c.count > 0u) ? (c.sum / static_cast<double>(c.count)) : 0.0;
        const double min_value = (c.count > 0u) ? c.min_value : 0.0;
        const double max_value = (c.count > 0u) ? c.max_value : 0.0;
        oss << " [" << c.label
            << " norm=" << std::sqrt(std::max(0.0, c.sq_norm))
            << " mean=" << mean
            << " min=" << min_value
            << " max=" << max_value << "]";
    }
    FE_LOG_INFO(oss.str());
}

void logNewtonFieldResidualDiagnostic(
    const systems::FESystem& sys,
    backends::GenericVector& residual,
    std::string_view phase,
    NewtonOptions::StateSynchronizationPoint sync_point,
    int iteration,
    double solve_time,
    double dt)
{
    if (!newtonFieldResidualDiagnosticEnabled()) {
        return;
    }
    const auto communicator = systemCommunicator(sys);

    const auto& field_name = newtonFieldResidualDiagnosticFieldName();
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

    const auto& sample_dofs = newtonFieldResidualDiagnosticSampleDofs();
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

void logNewtonMatrixSupportDiagnostic(const systems::FESystem& sys,
                                      const backends::GenericMatrix& matrix,
                                      std::span<const GlobalIndex> constrained_dofs,
                                      std::string_view phase,
                                      int iteration,
                                      double solve_time,
                                      double dt)
{
    if (!newtonMatrixSupportDiagnosticRequested()) {
        return;
    }

    std::vector<GlobalIndex> sample_dofs =
        newtonMatrixSupportGlobalSampleDofs();

    const auto pressure_field = sys.findFieldByName("Pressure");
    GlobalIndex pressure_offset = INVALID_GLOBAL_INDEX;
    GlobalIndex pressure_dofs = 0;
    if (pressure_field != INVALID_FIELD_ID) {
        pressure_offset = sys.fieldDofOffset(pressure_field);
        pressure_dofs = sys.fieldDofHandler(pressure_field).getNumDofs();
        for (const auto local_dof : newtonMatrixSupportPressureLocalSampleDofs()) {
            if (local_dof < 0 || local_dof >= pressure_dofs || pressure_offset < 0) {
                continue;
            }
            sample_dofs.push_back(pressure_offset + local_dof);
        }
    }

    std::sort(sample_dofs.begin(), sample_dofs.end());
    sample_dofs.erase(std::unique(sample_dofs.begin(), sample_dofs.end()),
                      sample_dofs.end());
    if (sample_dofs.empty()) {
        return;
    }

    const auto field_ranges = newtonMatrixSupportFieldRanges(sys);
    constexpr Real tiny = Real(1.0e-14);
    for (const auto dof : sample_dofs) {
        std::ostringstream oss;
        oss << std::setprecision(17);
        oss << "NewtonSolver: matrix support diagnostic"
            << " diagnostic=newton_matrix_support_sample"
            << " rank=" << mpiRank()
            << " iteration=" << iteration
            << " phase='" << phase << "'"
            << " backend=" << backends::backendKindToString(matrix.backendKind())
            << " solve_time=" << solve_time
            << " dt=" << dt
            << " dof=" << dof;
        if (dof < 0 || dof >= matrix.numRows() || dof >= matrix.numCols()) {
            oss << " status=out_of_range"
                << " rows=" << matrix.numRows()
                << " cols=" << matrix.numCols();
            FE_LOG_INFO(oss.str());
            continue;
        }

        double row_abs_sum = 0.0;
        double col_abs_sum = 0.0;
        int row_numeric_entries = 0;
        int col_numeric_entries = 0;
        double row_max_abs = 0.0;
        double col_max_abs = 0.0;
        double row_constrained_abs_sum = 0.0;
        double row_unconstrained_abs_sum = 0.0;
        double col_constrained_abs_sum = 0.0;
        double col_unconstrained_abs_sum = 0.0;
        std::vector<double> row_field_abs_sums(field_ranges.size(), 0.0);
        std::vector<double> col_field_abs_sums(field_ranges.size(), 0.0);
        std::vector<double> row_constrained_field_abs_sums(field_ranges.size(), 0.0);
        std::vector<double> row_unconstrained_field_abs_sums(field_ranges.size(), 0.0);
        std::vector<double> col_constrained_field_abs_sums(field_ranges.size(), 0.0);
        std::vector<double> col_unconstrained_field_abs_sums(field_ranges.size(), 0.0);
        for (GlobalIndex col = 0; col < matrix.numCols(); ++col) {
            const Real value = matrix.getEntry(dof, col);
            if (!std::isfinite(value)) {
                continue;
            }
            const double abs_value = std::abs(static_cast<double>(value));
            row_abs_sum += abs_value;
            row_max_abs = std::max(row_max_abs, abs_value);
            const bool constrained_col =
                std::binary_search(
                    constrained_dofs.begin(), constrained_dofs.end(), col);
            if (constrained_col) {
                row_constrained_abs_sum += abs_value;
                addMatrixSupportFieldAbs(
                    field_ranges,
                    col,
                    abs_value,
                    row_constrained_field_abs_sums);
            } else {
                row_unconstrained_abs_sum += abs_value;
                addMatrixSupportFieldAbs(
                    field_ranges,
                    col,
                    abs_value,
                    row_unconstrained_field_abs_sums);
            }
            addMatrixSupportFieldAbs(
                field_ranges,
                col,
                abs_value,
                row_field_abs_sums);
            if (abs_value > static_cast<double>(tiny)) {
                ++row_numeric_entries;
            }
        }
        for (GlobalIndex row = 0; row < matrix.numRows(); ++row) {
            const Real value = matrix.getEntry(row, dof);
            if (!std::isfinite(value)) {
                continue;
            }
            const double abs_value = std::abs(static_cast<double>(value));
            col_abs_sum += abs_value;
            col_max_abs = std::max(col_max_abs, abs_value);
            const bool constrained_row =
                std::binary_search(
                    constrained_dofs.begin(), constrained_dofs.end(), row);
            if (constrained_row) {
                col_constrained_abs_sum += abs_value;
                addMatrixSupportFieldAbs(
                    field_ranges,
                    row,
                    abs_value,
                    col_constrained_field_abs_sums);
            } else {
                col_unconstrained_abs_sum += abs_value;
                addMatrixSupportFieldAbs(
                    field_ranges,
                    row,
                    abs_value,
                    col_unconstrained_field_abs_sums);
            }
            addMatrixSupportFieldAbs(
                field_ranges,
                row,
                abs_value,
                col_field_abs_sums);
            if (abs_value > static_cast<double>(tiny)) {
                ++col_numeric_entries;
            }
        }

        oss << " status=ok"
            << " row_abs_sum=" << row_abs_sum
            << " row_numeric_entries=" << row_numeric_entries
            << " row_max_abs=" << row_max_abs
            << " col_abs_sum=" << col_abs_sum
            << " col_numeric_entries=" << col_numeric_entries
            << " col_max_abs=" << col_max_abs
            << " row_constrained_abs_sum=" << row_constrained_abs_sum
            << " row_unconstrained_abs_sum=" << row_unconstrained_abs_sum
            << " col_constrained_abs_sum=" << col_constrained_abs_sum
            << " col_unconstrained_abs_sum=" << col_unconstrained_abs_sum
            << " diag=" << static_cast<double>(matrix.getEntry(dof, dof))
            << " row_field_abs_sums="
            << formatMatrixSupportFieldSums(field_ranges, row_field_abs_sums)
            << " row_constrained_field_abs_sums="
            << formatMatrixSupportFieldSums(
                   field_ranges, row_constrained_field_abs_sums)
            << " row_unconstrained_field_abs_sums="
            << formatMatrixSupportFieldSums(
                   field_ranges, row_unconstrained_field_abs_sums)
            << " col_field_abs_sums="
            << formatMatrixSupportFieldSums(field_ranges, col_field_abs_sums)
            << " col_constrained_field_abs_sums="
            << formatMatrixSupportFieldSums(
                   field_ranges, col_constrained_field_abs_sums)
            << " col_unconstrained_field_abs_sums="
            << formatMatrixSupportFieldSums(
                   field_ranges, col_unconstrained_field_abs_sums)
            << " row_first_nonzero="
            << firstNonzeroMatrixIndices(matrix, dof, /*scan_row=*/true, tiny)
            << " col_first_nonzero="
            << firstNonzeroMatrixIndices(matrix, dof, /*scan_row=*/false, tiny);

        if (pressure_offset >= 0 &&
            dof >= pressure_offset &&
            dof < pressure_offset + pressure_dofs) {
            oss << " field='Pressure'"
                << " field_local_dof=" << (dof - pressure_offset);
        }
        FE_LOG_INFO(oss.str());
    }
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

struct ActivePressureSupportRankRow {
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

struct ActivePressureUpdateActionTerm {
    GlobalIndex local_dof = INVALID_GLOBAL_INDEX;
    GlobalIndex global_dof = INVALID_GLOBAL_INDEX;
    double matrix_value = 0.0;
    double update = 0.0;
    double action = 0.0;
};

struct ActivePressureUpdateSupportRow : ActivePressureSupportRankRow {
    double update = 0.0;
    double abs_update = 0.0;
    double rhs = 0.0;
    double row_action = 0.0;
    double row_coupling_action = 0.0;
    double row_self_action = 0.0;
    double row_self_constant_action = 0.0;
    double row_self_nonconstant_action = 0.0;
    double row_other_action = 0.0;
    double row_linear_residual = 0.0;
    std::vector<ActivePressureUpdateActionTerm> pressure_action_terms;
    std::vector<ActivePressureUpdateActionTerm> coupling_action_terms;
};

struct ActivePressureSupportRankSummary {
    std::string pressure_field;
    std::string coupling_field;
    GlobalIndex pressure_offset = INVALID_GLOBAL_INDEX;
    GlobalIndex pressure_dofs = 0;
    GlobalIndex coupling_offset = INVALID_GLOBAL_INDEX;
    GlobalIndex coupling_dofs = 0;
    GlobalIndex constrained_pressure_rows = 0;
    GlobalIndex unconstrained_pressure_rows = 0;
    GlobalIndex zero_row_count = 0;
    GlobalIndex zero_col_count = 0;
    GlobalIndex zero_diag_count = 0;
    GlobalIndex zero_coupling_row_block_count = 0;
    GlobalIndex zero_coupling_col_block_count = 0;
    GlobalIndex zero_self_row_block_count = 0;
    GlobalIndex zero_self_col_block_count = 0;
    GlobalIndex positive_coupling_row_block_count = 0;
    GlobalIndex positive_self_row_block_count = 0;
    GlobalIndex weak_coupling_row_block_count = 0;
    GlobalIndex weak_self_row_block_count = 0;
    GlobalIndex weak_coupling_and_self_row_block_count = 0;
    GlobalIndex pressure_only_row_block_count = 0;
    GlobalIndex pressure_only_col_block_count = 0;
    double min_positive_coupling_row_abs_sum =
        std::numeric_limits<double>::infinity();
    double max_coupling_row_abs_sum = 0.0;
    double min_positive_self_row_abs_sum =
        std::numeric_limits<double>::infinity();
    double max_self_row_abs_sum = 0.0;
    std::vector<GlobalIndex> zero_coupling_row_global_dofs;
    std::vector<GlobalIndex> clamp_candidate_row_global_dofs;
    std::vector<ActivePressureSupportRankRow> zero_coupling_row_samples;
    std::vector<ActivePressureSupportRankRow> zero_row_samples;
    std::vector<ActivePressureSupportRankRow> weakest_coupling_row_samples;
    std::vector<ActivePressureSupportRankRow> weakest_self_row_samples;
    std::vector<ActivePressureSupportRankRow> clamp_candidate_row_samples;
};

struct ActivePressureUpdateSupportSummary {
    std::string pressure_field;
    std::string coupling_field;
    GlobalIndex pressure_offset = INVALID_GLOBAL_INDEX;
    GlobalIndex pressure_dofs = 0;
    GlobalIndex coupling_offset = INVALID_GLOBAL_INDEX;
    GlobalIndex coupling_dofs = 0;
    GlobalIndex constrained_pressure_rows = 0;
    GlobalIndex unconstrained_pressure_rows = 0;
    GlobalIndex zero_coupling_row_block_count = 0;
    GlobalIndex weak_coupling_row_block_count = 0;
    GlobalIndex positive_coupling_row_block_count = 0;
    GlobalIndex zero_self_row_block_count = 0;
    GlobalIndex weak_self_row_block_count = 0;
    GlobalIndex positive_self_row_block_count = 0;
    double max_abs_update = 0.0;
    double zero_coupling_max_abs_update = 0.0;
    double weak_coupling_max_abs_update = 0.0;
    double positive_coupling_max_abs_update = 0.0;
    double zero_self_max_abs_update = 0.0;
    double weak_self_max_abs_update = 0.0;
    double positive_self_max_abs_update = 0.0;
    GlobalIndex max_update_local_dof = INVALID_GLOBAL_INDEX;
    GlobalIndex max_update_global_dof = INVALID_GLOBAL_INDEX;
    double max_update_rhs = 0.0;
    double max_update_row_action = 0.0;
    double max_update_row_coupling_action = 0.0;
    double max_update_row_self_action = 0.0;
    double max_update_row_self_constant_action = 0.0;
    double max_update_row_self_nonconstant_action = 0.0;
    double max_update_row_other_action = 0.0;
    double max_update_row_linear_residual = 0.0;
    GlobalIndex same_sign_pressure_action_top_edge_count = 0;
    GlobalIndex same_sign_pressure_action_component_count = 0;
    GlobalIndex same_sign_pressure_action_largest_component_size = 0;
    GlobalIndex same_sign_pressure_action_covered_top_update_count = 0;
    GlobalIndex same_sign_pressure_action_isolated_top_update_count = 0;
    int same_sign_pressure_action_largest_component_has_max_update = 0;
    std::vector<GlobalIndex> same_sign_pressure_action_covered_global_dofs;
    std::vector<GlobalIndex> same_sign_pressure_action_isolated_global_dofs;
    std::vector<GlobalIndex> same_sign_pressure_action_largest_component_global_dofs;
    std::vector<ActivePressureUpdateSupportRow> top_update_samples;
};

[[nodiscard]] std::string formatGlobalIndexListSample(
    std::span<const GlobalIndex> values,
    int limit);

[[nodiscard]] std::string formatActivePressureSupportRankDofSamples(
    std::span<const ActivePressureSupportRankRow> rows,
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

[[nodiscard]] std::string formatActivePressureSupportRankRowDetails(
    std::span<const ActivePressureSupportRankRow> rows)
{
    if (rows.empty()) {
        return "none";
    }
    std::ostringstream oss;
    oss << std::setprecision(17);
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto& row = rows[i];
        if (i > 0u) {
            oss << '|';
        }
        oss << row.local_dof << ':' << row.global_dof
            << ":row=" << row.row_abs_sum
            << ":row_coupling=" << row.row_coupling_abs_sum
            << ":row_self=" << row.row_self_abs_sum
            << ":row_self_sum=" << row.row_self_sum
            << ":row_self_offdiag=" << row.row_self_offdiag_abs_sum
            << ":row_self_signed_abs_ratio="
            << row.row_self_signed_abs_ratio
            << ":row_self_diag_abs_ratio=" << row.row_self_diag_abs_ratio
            << ":col=" << row.col_abs_sum
            << ":col_coupling=" << row.col_coupling_abs_sum
            << ":col_self=" << row.col_self_abs_sum
            << ":diag=" << row.diag;
    }
    return oss.str();
}

[[nodiscard]] std::vector<GlobalIndex>
pressureOperatorMatrixSupportSampleDofs(
    const NewtonMatrixSupportFieldRange& pressure_range)
{
    std::vector<GlobalIndex> sample_dofs =
        newtonFieldResidualDiagnosticSampleDofs();

    const auto& matrix_sample_dofs = newtonMatrixSupportGlobalSampleDofs();
    sample_dofs.insert(sample_dofs.end(),
                       matrix_sample_dofs.begin(),
                       matrix_sample_dofs.end());

    for (const auto local_dof : newtonMatrixSupportPressureLocalSampleDofs()) {
        if (local_dof < 0 ||
            local_dof >= pressure_range.end - pressure_range.begin ||
            pressure_range.begin < 0) {
            continue;
        }
        sample_dofs.push_back(pressure_range.begin + local_dof);
    }

    std::sort(sample_dofs.begin(), sample_dofs.end());
    sample_dofs.erase(std::unique(sample_dofs.begin(), sample_dofs.end()),
                      sample_dofs.end());
    return sample_dofs;
}

void logPressureRowOperatorMatrixSupportDiagnostic(
    const systems::FESystem& sys,
    const backends::GenericMatrix& matrix,
    std::string_view phase,
    std::string_view op,
    int iteration,
    double solve_time,
    double dt)
{
    if (!pressureRowContributionMatrixDiagnosticEnabled()) {
        return;
    }

    const auto pressure_range =
        newtonMatrixSupportFieldRangeByName(sys, "Pressure");
    const auto coupling_range =
        newtonMatrixSupportFieldRangeByName(sys, "Velocity");
    if (!pressure_range.has_value() || !coupling_range.has_value()) {
        if (mpiRank() == 0) {
            FE_LOG_INFO(
                "NewtonSolver: pressure row operator matrix support diagnostic "
                "diagnostic=pressure_row_operator_matrix_support status=field_missing "
                "pressure_field='Pressure' coupling_field='Velocity' op='" +
                std::string(op) + "' phase='" + std::string(phase) + "'");
        }
        return;
    }

    const auto sample_dofs =
        pressureOperatorMatrixSupportSampleDofs(*pressure_range);
    if (sample_dofs.empty()) {
        return;
    }

    constexpr Real tiny = Real(1.0e-14);
    for (const auto dof : sample_dofs) {
        std::ostringstream oss;
        oss << std::setprecision(17);
        oss << "NewtonSolver: pressure row operator matrix support diagnostic"
            << " diagnostic=pressure_row_operator_matrix_support"
            << " rank=" << mpiRank()
            << " iteration=" << iteration
            << " phase='" << phase << "'"
            << " op='" << op << "'"
            << " backend=" << backends::backendKindToString(matrix.backendKind())
            << " solve_time=" << solve_time
            << " dt=" << dt
            << " pressure_field='Pressure'"
            << " coupling_field='Velocity'"
            << " pressure_offset=" << pressure_range->begin
            << " pressure_dofs=" << (pressure_range->end - pressure_range->begin)
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
        if (!dofInFieldRange(dof, *pressure_range)) {
            oss << " status=non_pressure_row";
            FE_LOG_INFO(oss.str());
            continue;
        }

        ActivePressureSupportRankRow row;
        row.global_dof = dof;
        row.local_dof = dof - pressure_range->begin;
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
            if (dofInFieldRange(col, *pressure_range)) {
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
            if (dofInFieldRange(matrix_row, *pressure_range)) {
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
            << " field='Pressure'"
            << " field_local_dof=" << row.local_dof
            << " pressure_local_dof=" << row.local_dof
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

[[nodiscard]] std::string formatActivePressureUpdateActionTerms(
    std::span<const ActivePressureUpdateActionTerm> terms)
{
    if (terms.empty()) {
        return "none";
    }
    std::ostringstream oss;
    oss << std::setprecision(17);
    for (std::size_t i = 0; i < terms.size(); ++i) {
        const auto& term = terms[i];
        if (i > 0u) {
            oss << '~';
        }
        oss << term.local_dof << '/' << term.global_dof
            << "/m=" << term.matrix_value
            << "/u=" << term.update
            << "/a=" << term.action;
    }
    return oss.str();
}

[[nodiscard]] std::string formatActivePressureUpdateSupportRowDetails(
    std::span<const ActivePressureUpdateSupportRow> rows)
{
    if (rows.empty()) {
        return "none";
    }
    std::ostringstream oss;
    oss << std::setprecision(17);
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto& row = rows[i];
        if (i > 0u) {
            oss << '|';
        }
        oss << row.local_dof << ':' << row.global_dof
            << ":update=" << row.update
            << ":abs_update=" << row.abs_update
            << ":rhs=" << row.rhs
            << ":row_action=" << row.row_action
            << ":row_coupling_action=" << row.row_coupling_action
            << ":row_self_action=" << row.row_self_action
            << ":row_self_constant_action="
            << row.row_self_constant_action
            << ":row_self_nonconstant_action="
            << row.row_self_nonconstant_action
            << ":row_other_action=" << row.row_other_action
            << ":row_linear_residual=" << row.row_linear_residual
            << ":row=" << row.row_abs_sum
            << ":row_coupling=" << row.row_coupling_abs_sum
            << ":row_self=" << row.row_self_abs_sum
            << ":row_self_sum=" << row.row_self_sum
            << ":row_self_offdiag=" << row.row_self_offdiag_abs_sum
            << ":row_self_signed_abs_ratio="
            << row.row_self_signed_abs_ratio
            << ":row_self_diag_abs_ratio=" << row.row_self_diag_abs_ratio
            << ":col=" << row.col_abs_sum
            << ":col_coupling=" << row.col_coupling_abs_sum
            << ":col_self=" << row.col_self_abs_sum
            << ":diag=" << row.diag
            << ":pressure_action_terms="
            << formatActivePressureUpdateActionTerms(
                   row.pressure_action_terms)
            << ":coupling_action_terms="
            << formatActivePressureUpdateActionTerms(
                   row.coupling_action_terms);
    }
    return oss.str();
}

[[nodiscard]] bool sameNonzeroSign(double left, double right) noexcept
{
    if (left == 0.0 || right == 0.0) {
        return false;
    }
    return (left > 0.0) == (right > 0.0);
}

void addSameSignPressureActionComponentSummary(
    ActivePressureUpdateSupportSummary& summary)
{
    std::map<GlobalIndex, double> update_by_dof;
    for (const auto& row : summary.top_update_samples) {
        update_by_dof[row.global_dof] = row.update;
    }
    if (update_by_dof.empty()) {
        return;
    }

    std::set<std::pair<GlobalIndex, GlobalIndex>> edges;
    std::map<GlobalIndex, std::vector<GlobalIndex>> adjacency;
    for (const auto& row : summary.top_update_samples) {
        const auto row_it = update_by_dof.find(row.global_dof);
        if (row_it == update_by_dof.end()) {
            continue;
        }
        for (const auto& term : row.pressure_action_terms) {
            if (term.global_dof == row.global_dof) {
                continue;
            }
            const auto neighbor_it = update_by_dof.find(term.global_dof);
            if (neighbor_it == update_by_dof.end()) {
                continue;
            }
            if (!sameNonzeroSign(row_it->second, neighbor_it->second)) {
                continue;
            }
            const auto left = std::min(row.global_dof, term.global_dof);
            const auto right = std::max(row.global_dof, term.global_dof);
            edges.insert({left, right});
        }
    }

    for (const auto& edge : edges) {
        adjacency[edge.first].push_back(edge.second);
        adjacency[edge.second].push_back(edge.first);
    }
    summary.same_sign_pressure_action_top_edge_count =
        static_cast<GlobalIndex>(edges.size());

    std::set<GlobalIndex> covered;
    std::set<GlobalIndex> seen;
    std::vector<GlobalIndex> largest_component;
    for (const auto& entry : adjacency) {
        const auto start = entry.first;
        if (seen.count(start) != 0u) {
            continue;
        }
        std::vector<GlobalIndex> pending{start};
        std::vector<GlobalIndex> component;
        seen.insert(start);
        while (!pending.empty()) {
            const auto dof = pending.back();
            pending.pop_back();
            component.push_back(dof);
            covered.insert(dof);
            for (const auto neighbor : adjacency[dof]) {
                if (seen.count(neighbor) != 0u) {
                    continue;
                }
                seen.insert(neighbor);
                pending.push_back(neighbor);
            }
        }
        std::sort(component.begin(), component.end());
        ++summary.same_sign_pressure_action_component_count;
        if (component.size() > largest_component.size() ||
            (component.size() == largest_component.size() &&
             !component.empty() &&
             !largest_component.empty() &&
             component.front() < largest_component.front())) {
            largest_component = component;
        }
    }

    summary.same_sign_pressure_action_covered_global_dofs.assign(
        covered.begin(), covered.end());
    summary.same_sign_pressure_action_covered_top_update_count =
        static_cast<GlobalIndex>(
            summary.same_sign_pressure_action_covered_global_dofs.size());

    std::vector<GlobalIndex> isolated;
    for (const auto& row : summary.top_update_samples) {
        if (covered.count(row.global_dof) == 0u) {
            isolated.push_back(row.global_dof);
        }
    }
    std::sort(isolated.begin(), isolated.end());
    summary.same_sign_pressure_action_isolated_global_dofs = isolated;
    summary.same_sign_pressure_action_isolated_top_update_count =
        static_cast<GlobalIndex>(isolated.size());
    summary.same_sign_pressure_action_largest_component_global_dofs =
        largest_component;
    summary.same_sign_pressure_action_largest_component_size =
        static_cast<GlobalIndex>(largest_component.size());
    summary.same_sign_pressure_action_largest_component_has_max_update =
        std::find(largest_component.begin(),
                  largest_component.end(),
                  summary.max_update_global_dof) != largest_component.end()
            ? 1
            : 0;
}

[[nodiscard]] ActivePressureSupportRankSummary scanActivePressureSupportRank(
    const systems::FESystem& sys,
    const backends::GenericMatrix& matrix,
    std::span<const GlobalIndex> constrained_dofs,
    std::string_view pressure_field_name,
    std::string_view coupling_field_name,
    double tolerance,
    int sample_limit,
    double clamp_coupling_threshold = -1.0,
    double clamp_self_threshold = -1.0)
{
    ActivePressureSupportRankSummary summary;
    summary.pressure_field = std::string(pressure_field_name);
    summary.coupling_field = std::string(coupling_field_name);

    const auto pressure_range =
        newtonMatrixSupportFieldRangeByName(sys, pressure_field_name);
    const auto coupling_range =
        newtonMatrixSupportFieldRangeByName(sys, coupling_field_name);
    if (!pressure_range.has_value() || !coupling_range.has_value()) {
        return summary;
    }

    summary.pressure_offset = pressure_range->begin;
    summary.pressure_dofs = pressure_range->end - pressure_range->begin;
    summary.coupling_offset = coupling_range->begin;
    summary.coupling_dofs = coupling_range->end - coupling_range->begin;

    const auto can_add_sample = [sample_limit](const auto& samples) {
        return sample_limit < 0 ||
               static_cast<int>(samples.size()) < sample_limit;
    };
    const auto add_weakest_coupling_sample =
        [sample_limit](std::vector<ActivePressureSupportRankRow>& samples,
                       const ActivePressureSupportRankRow& row) {
            if (sample_limit == 0) {
                return;
            }
            samples.push_back(row);
            std::sort(samples.begin(), samples.end(),
                      [](const ActivePressureSupportRankRow& a,
                         const ActivePressureSupportRankRow& b) {
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
        [sample_limit](std::vector<ActivePressureSupportRankRow>& samples,
                       const ActivePressureSupportRankRow& row) {
            if (sample_limit == 0) {
                return;
            }
            samples.push_back(row);
            std::sort(samples.begin(), samples.end(),
                      [](const ActivePressureSupportRankRow& a,
                         const ActivePressureSupportRankRow& b) {
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

    for (GlobalIndex local_dof = 0; local_dof < summary.pressure_dofs; ++local_dof) {
        const auto global_dof = summary.pressure_offset + local_dof;
        if (std::binary_search(
                constrained_dofs.begin(), constrained_dofs.end(), global_dof)) {
            ++summary.constrained_pressure_rows;
            continue;
        }

        ++summary.unconstrained_pressure_rows;
        ActivePressureSupportRankRow row;
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
            if (dofInFieldRange(col, *pressure_range)) {
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
            if (dofInFieldRange(matrix_row, *pressure_range)) {
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
        const bool weak_coupling_row =
            !zero_coupling_row &&
            clamp_coupling_threshold >= 0.0 &&
            row.row_coupling_abs_sum <= clamp_coupling_threshold;
        const bool weak_self_row =
            !zero_self_row &&
            clamp_self_threshold >= 0.0 &&
            row.row_self_abs_sum <= clamp_self_threshold;
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
        if (weak_coupling_row) {
            ++summary.weak_coupling_row_block_count;
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
        if (weak_self_row) {
            ++summary.weak_self_row_block_count;
        }
        if (weak_coupling_row && weak_self_row) {
            ++summary.weak_coupling_and_self_row_block_count;
        }
        if (zero_self_col) {
            ++summary.zero_self_col_block_count;
        }
        if (zero_coupling_row && !zero_self_row) {
            ++summary.pressure_only_row_block_count;
        }
        if (zero_coupling_col && !zero_self_col) {
            ++summary.pressure_only_col_block_count;
        }
        const bool clamp_coupling_candidate =
            clamp_coupling_threshold >= 0.0 &&
            row.row_coupling_abs_sum <= clamp_coupling_threshold;
        const bool clamp_self_candidate =
            clamp_self_threshold >= 0.0 &&
            row.row_self_abs_sum <= clamp_self_threshold;
        if (clamp_coupling_candidate || clamp_self_candidate) {
            summary.clamp_candidate_row_global_dofs.push_back(global_dof);
            if (can_add_sample(summary.clamp_candidate_row_samples)) {
                summary.clamp_candidate_row_samples.push_back(row);
            }
        }
    }

    return summary;
}

void logPressureRowOperatorMatrixSummaryDiagnostic(
    const systems::FESystem& sys,
    const backends::GenericMatrix& matrix,
    std::span<const GlobalIndex> constrained_dofs,
    std::string_view phase,
    std::string_view op,
    int iteration,
    double solve_time,
    double dt)
{
    if (!pressureRowContributionMatrixSummaryDiagnosticEnabled() ||
        !pressureRowContributionMatrixSummaryOpSelected(op)) {
        return;
    }

    const auto& pressure_field = activePressureSupportRankPressureFieldName();
    const auto& coupling_field = activePressureSupportRankCouplingFieldName();
    const double tolerance = activePressureSupportRankTolerance();
    const double weak_coupling_threshold =
        activePressureUpdateSupportWeakVelocityThreshold();
    const double weak_self_threshold =
        activePressureUpdateSupportWeakSelfThreshold();
    const int sample_limit = activePressureSupportRankSampleLimit();
    const auto summary = scanActivePressureSupportRank(
        sys,
        matrix,
        constrained_dofs,
        pressure_field,
        coupling_field,
        tolerance,
        sample_limit,
        weak_coupling_threshold,
        weak_self_threshold);

    if (summary.pressure_dofs <= 0 || summary.coupling_dofs <= 0) {
        FE_LOG_INFO(
            "NewtonSolver: pressure row operator matrix summary"
            " diagnostic=pressure_row_operator_matrix_summary"
            " status=field_missing"
            " op='" +
            std::string(op) +
            "' phase='" + std::string(phase) + "'"
            " pressure_field='" + pressure_field + "'"
            " coupling_field='" + coupling_field + "'");
        return;
    }

    std::ostringstream oss;
    oss << std::setprecision(17);
    oss << "NewtonSolver: pressure row operator matrix summary"
        << " diagnostic=pressure_row_operator_matrix_summary"
        << " status=ok"
        << " rank=" << mpiRank()
        << " iteration=" << iteration
        << " phase='" << phase << "'"
        << " op='" << op << "'"
        << " backend=" << backends::backendKindToString(matrix.backendKind())
        << " solve_time=" << solve_time
        << " dt=" << dt
        << " pressure_field='" << summary.pressure_field << "'"
        << " coupling_field='" << summary.coupling_field << "'"
        << " pressure_offset=" << summary.pressure_offset
        << " pressure_dofs=" << summary.pressure_dofs
        << " coupling_offset=" << summary.coupling_offset
        << " coupling_dofs=" << summary.coupling_dofs
        << " constrained_pressure_rows=" << summary.constrained_pressure_rows
        << " unconstrained_pressure_rows=" << summary.unconstrained_pressure_rows
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
        << " weak_coupling_row_block_count="
        << summary.weak_coupling_row_block_count
        << " weak_self_row_block_count="
        << summary.weak_self_row_block_count
        << " weak_coupling_and_self_row_block_count="
        << summary.weak_coupling_and_self_row_block_count
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
        << " pressure_only_row_block_count="
        << summary.pressure_only_row_block_count
        << " pressure_only_col_block_count="
        << summary.pressure_only_col_block_count
        << " tolerance=" << tolerance
        << " weak_coupling_threshold=" << weak_coupling_threshold
        << " weak_self_threshold=" << weak_self_threshold
        << " sample_limit=" << sample_limit
        << " zero_coupling_row_local_dofs="
        << formatActivePressureSupportRankDofSamples(
               summary.zero_coupling_row_samples, /*local=*/true)
        << " zero_coupling_row_global_dofs="
        << formatActivePressureSupportRankDofSamples(
               summary.zero_coupling_row_samples, /*local=*/false)
        << " zero_row_local_dofs="
        << formatActivePressureSupportRankDofSamples(
               summary.zero_row_samples, /*local=*/true)
        << " zero_row_global_dofs="
        << formatActivePressureSupportRankDofSamples(
               summary.zero_row_samples, /*local=*/false)
        << " weakest_coupling_row_local_dofs="
        << formatActivePressureSupportRankDofSamples(
               summary.weakest_coupling_row_samples, /*local=*/true)
        << " weakest_coupling_row_global_dofs="
        << formatActivePressureSupportRankDofSamples(
               summary.weakest_coupling_row_samples, /*local=*/false)
        << " weakest_self_row_local_dofs="
        << formatActivePressureSupportRankDofSamples(
               summary.weakest_self_row_samples, /*local=*/true)
        << " weakest_self_row_global_dofs="
        << formatActivePressureSupportRankDofSamples(
               summary.weakest_self_row_samples, /*local=*/false);
    FE_LOG_INFO(oss.str());
}

void logDirectPspgFormulationCandidateDiagnostic(
    const systems::FESystem& sys,
    const backends::GenericMatrix& matrix,
    backends::GenericVector& residual,
    std::span<const GlobalIndex> constrained_dofs,
    std::string_view phase,
    std::string_view op,
    int iteration,
    double solve_time,
    double dt)
{
    if (!directPspgFormulationCandidateDiagnosticEnabled() ||
        !directPspgFormulationCandidateOpSelected(op)) {
        return;
    }

    const auto& pressure_field = activePressureSupportRankPressureFieldName();
    const auto pressure_range =
        newtonMatrixSupportFieldRangeByName(sys, pressure_field);
    if (!pressure_range.has_value()) {
        FE_LOG_INFO(
            "NewtonSolver: direct PSPG formulation candidate diagnostic"
            " diagnostic=direct_pspg_formulation_candidate status=field_missing"
            " pressure_field='" +
            pressure_field + "' op='" + std::string(op) +
            "' phase='" + std::string(phase) + "'");
        return;
    }

    struct DirectPspgCandidateRow {
        GlobalIndex global_dof = INVALID_GLOBAL_INDEX;
        double row_self_abs_sum = 0.0;
        double row_self_constrained_abs_sum = 0.0;
        double row_self_unconstrained_abs_sum = 0.0;
        double row_self_sum = 0.0;
        double diag = 0.0;
        double row_self_offdiag_abs_sum = 0.0;
        double row_self_signed_abs_ratio = 0.0;
        double row_self_diag_abs_ratio = 0.0;
        double residual_value = 0.0;
        int row_self_numeric_entries = 0;
        int row_self_constrained_numeric_entries = 0;
        int row_self_unconstrained_numeric_entries = 0;
        int residual_sign = 0;
    };

    constexpr double tiny = 1.0e-14;
    const double tolerance = activePressureSupportRankTolerance();
    std::vector<DirectPspgCandidateRow> direct_rows;
    int min_direct_self_entries = std::numeric_limits<int>::max();
    int max_direct_self_entries = 0;
    int max_unconstrained_direct_self_entries = 0;
    double min_positive_direct_self_abs_sum =
        std::numeric_limits<double>::infinity();
    double max_direct_self_abs_sum = 0.0;

    for (GlobalIndex dof = pressure_range->begin; dof < pressure_range->end; ++dof) {
        if (dof < 0 ||
            dof >= matrix.numRows() ||
            dof >= matrix.numCols() ||
            std::binary_search(
                constrained_dofs.begin(), constrained_dofs.end(), dof)) {
            continue;
        }
        DirectPspgCandidateRow row;
        row.global_dof = dof;
        for (GlobalIndex col = pressure_range->begin;
             col < pressure_range->end;
             ++col) {
            const Real value = matrix.getEntry(dof, col);
            if (!std::isfinite(value)) {
                continue;
            }
            const double as_double = static_cast<double>(value);
            const double abs_value = std::abs(as_double);
            if (col == dof) {
                row.diag = as_double;
            }
            row.row_self_abs_sum += abs_value;
            row.row_self_sum += as_double;
            if (abs_value > tiny) {
                ++row.row_self_numeric_entries;
                if (std::binary_search(
                        constrained_dofs.begin(), constrained_dofs.end(), col)) {
                    ++row.row_self_constrained_numeric_entries;
                    row.row_self_constrained_abs_sum += abs_value;
                } else {
                    ++row.row_self_unconstrained_numeric_entries;
                    row.row_self_unconstrained_abs_sum += abs_value;
                }
            }
        }
        if (row.row_self_abs_sum <= tolerance ||
            row.row_self_numeric_entries <= 0) {
            continue;
        }
        row.row_self_offdiag_abs_sum =
            std::max(0.0, row.row_self_abs_sum - std::abs(row.diag));
        row.row_self_signed_abs_ratio =
            std::abs(row.row_self_sum) / row.row_self_abs_sum;
        row.row_self_diag_abs_ratio =
            std::abs(row.diag) / row.row_self_abs_sum;
        direct_rows.push_back(row);
        min_direct_self_entries =
            std::min(min_direct_self_entries, row.row_self_numeric_entries);
        max_direct_self_entries =
            std::max(max_direct_self_entries, row.row_self_numeric_entries);
        max_unconstrained_direct_self_entries =
            std::max(max_unconstrained_direct_self_entries,
                     row.row_self_unconstrained_numeric_entries);
        min_positive_direct_self_abs_sum =
            std::min(min_positive_direct_self_abs_sum, row.row_self_abs_sum);
        max_direct_self_abs_sum =
            std::max(max_direct_self_abs_sum, row.row_self_abs_sum);
    }

    constexpr double kDirectSelfRowSumLeakThreshold = 0.25;
    constexpr double kDirectSelfNullPreservingThreshold = 0.05;
    constexpr double kDirectSelfDiagDominantThreshold = 0.6;
    constexpr double kDirectSelfBalancedDiagLowThreshold = 0.45;
    constexpr double kDirectSelfBalancedDiagHighThreshold = 0.55;
    double max_direct_self_row_sum_leak_ratio = 0.0;
    double min_direct_self_diag_abs_ratio =
        std::numeric_limits<double>::infinity();
    double max_direct_self_diag_abs_ratio = 0.0;
    std::vector<GlobalIndex> high_direct_self_row_sum_leak_global_dofs;
    std::vector<GlobalIndex> null_preserving_direct_self_global_dofs;
    std::vector<GlobalIndex> diag_dominant_direct_self_global_dofs;
    std::vector<GlobalIndex> balanced_diag_direct_self_global_dofs;
    for (const auto& row : direct_rows) {
        max_direct_self_row_sum_leak_ratio =
            std::max(max_direct_self_row_sum_leak_ratio,
                     row.row_self_signed_abs_ratio);
        min_direct_self_diag_abs_ratio =
            std::min(min_direct_self_diag_abs_ratio,
                     row.row_self_diag_abs_ratio);
        max_direct_self_diag_abs_ratio =
            std::max(max_direct_self_diag_abs_ratio,
                     row.row_self_diag_abs_ratio);
        if (row.row_self_signed_abs_ratio >=
            kDirectSelfRowSumLeakThreshold) {
            high_direct_self_row_sum_leak_global_dofs.push_back(
                row.global_dof);
        }
        if (row.row_self_signed_abs_ratio <=
            kDirectSelfNullPreservingThreshold) {
            null_preserving_direct_self_global_dofs.push_back(row.global_dof);
        }
        if (row.row_self_diag_abs_ratio >=
            kDirectSelfDiagDominantThreshold) {
            diag_dominant_direct_self_global_dofs.push_back(row.global_dof);
        }
        if (row.row_self_diag_abs_ratio >=
                kDirectSelfBalancedDiagLowThreshold &&
            row.row_self_diag_abs_ratio <=
                kDirectSelfBalancedDiagHighThreshold) {
            balanced_diag_direct_self_global_dofs.push_back(row.global_dof);
        }
    }

    auto residual_view = residual.createAssemblyView();
    FE_CHECK_NOT_NULL(
        residual_view.get(),
        "NewtonSolver: direct PSPG formulation candidate residual view");
    const double residual_sign_threshold = tolerance;
    int residual_positive_direct_row_count = 0;
    int residual_negative_direct_row_count = 0;
    int residual_zero_direct_row_count = 0;
    int residual_nonfinite_direct_row_count = 0;
    double min_positive_residual_abs = std::numeric_limits<double>::infinity();
    double max_residual_abs = 0.0;
    std::map<GlobalIndex, int> residual_sign_by_dof;
    for (auto& row : direct_rows) {
        const double residual_value =
            static_cast<double>(residual_view->getVectorEntry(row.global_dof));
        row.residual_value = residual_value;
        if (!std::isfinite(residual_value)) {
            ++residual_nonfinite_direct_row_count;
            residual_sign_by_dof[row.global_dof] = 0;
            continue;
        }
        const double residual_abs = std::abs(residual_value);
        max_residual_abs = std::max(max_residual_abs, residual_abs);
        if (residual_abs > residual_sign_threshold) {
            min_positive_residual_abs =
                std::min(min_positive_residual_abs, residual_abs);
            row.residual_sign = residual_value > 0.0 ? 1 : -1;
        }
        if (row.residual_sign > 0) {
            ++residual_positive_direct_row_count;
        } else if (row.residual_sign < 0) {
            ++residual_negative_direct_row_count;
        } else {
            ++residual_zero_direct_row_count;
        }
        residual_sign_by_dof[row.global_dof] = row.residual_sign;
    }

    std::vector<GlobalIndex> sparse_direct_self_global_dofs;
    for (const auto& row : direct_rows) {
        if (row.row_self_numeric_entries < max_direct_self_entries) {
            sparse_direct_self_global_dofs.push_back(row.global_dof);
        }
    }
    constexpr double kConstrainedPressureNeighborRatioThreshold = 0.25;
    std::vector<GlobalIndex> constrained_pressure_neighbor_global_dofs;
    std::vector<GlobalIndex> high_constrained_pressure_neighbor_ratio_global_dofs;
    std::vector<GlobalIndex> sparse_unconstrained_direct_self_global_dofs;
    std::set<GlobalIndex> constrained_or_sparse_unconstrained_direct_self_set;
    for (const auto& row : direct_rows) {
        if (row.row_self_constrained_numeric_entries > 0) {
            constrained_pressure_neighbor_global_dofs.push_back(row.global_dof);
            constrained_or_sparse_unconstrained_direct_self_set.insert(
                row.global_dof);
        }
        if (row.row_self_abs_sum > 0.0 &&
            row.row_self_constrained_abs_sum / row.row_self_abs_sum >=
                kConstrainedPressureNeighborRatioThreshold) {
            high_constrained_pressure_neighbor_ratio_global_dofs.push_back(
                row.global_dof);
        }
        if (row.row_self_unconstrained_numeric_entries <
            max_unconstrained_direct_self_entries) {
            sparse_unconstrained_direct_self_global_dofs.push_back(
                row.global_dof);
            constrained_or_sparse_unconstrained_direct_self_set.insert(
                row.global_dof);
        }
    }
    std::vector<GlobalIndex>
        constrained_or_sparse_unconstrained_direct_self_global_dofs(
            constrained_or_sparse_unconstrained_direct_self_set.begin(),
            constrained_or_sparse_unconstrained_direct_self_set.end());
    constexpr double kLowDirectSelfRatioThreshold = 0.25;
    constexpr double kModerateDirectSelfRatioThreshold = 0.5;
    std::vector<GlobalIndex> low_direct_self_ratio_global_dofs;
    std::vector<GlobalIndex> moderate_direct_self_ratio_global_dofs;
    if (max_direct_self_abs_sum > 0.0) {
        for (const auto& row : direct_rows) {
            const double ratio = row.row_self_abs_sum / max_direct_self_abs_sum;
            if (ratio <= kLowDirectSelfRatioThreshold) {
                low_direct_self_ratio_global_dofs.push_back(row.global_dof);
            }
            if (ratio <= kModerateDirectSelfRatioThreshold) {
                moderate_direct_self_ratio_global_dofs.push_back(row.global_dof);
            }
        }
    }

    std::set<std::pair<GlobalIndex, GlobalIndex>> pressure_action_edges;
    std::set<GlobalIndex> pressure_action_covered;
    std::map<GlobalIndex, GlobalIndex> pressure_action_degree_by_dof;
    std::map<GlobalIndex, double> pressure_action_abs_sum_by_dof;
    for (const auto& row : direct_rows) {
        pressure_action_degree_by_dof[row.global_dof] = 0;
        pressure_action_abs_sum_by_dof[row.global_dof] = 0.0;
    }
    for (std::size_t i = 0; i < direct_rows.size(); ++i) {
        const auto row_i = direct_rows[i].global_dof;
        for (std::size_t j = i + 1u; j < direct_rows.size(); ++j) {
            const auto row_j = direct_rows[j].global_dof;
            const double row_edge =
                static_cast<double>(matrix.getEntry(row_i, row_j));
            const double col_edge =
                static_cast<double>(matrix.getEntry(row_j, row_i));
            if (!std::isfinite(row_edge) || !std::isfinite(col_edge)) {
                continue;
            }
            const double symmetric_offdiag = 0.5 * (row_edge + col_edge);
            if (!(symmetric_offdiag < -tolerance)) {
                continue;
            }
            const double edge_abs = std::abs(symmetric_offdiag);
            pressure_action_edges.insert({row_i, row_j});
            pressure_action_covered.insert(row_i);
            pressure_action_covered.insert(row_j);
            ++pressure_action_degree_by_dof[row_i];
            ++pressure_action_degree_by_dof[row_j];
            pressure_action_abs_sum_by_dof[row_i] += edge_abs;
            pressure_action_abs_sum_by_dof[row_j] += edge_abs;
        }
    }

    std::vector<GlobalIndex> pressure_action_covered_global_dofs(
        pressure_action_covered.begin(), pressure_action_covered.end());
    std::vector<GlobalIndex> pressure_action_isolated_global_dofs;
    for (const auto& row : direct_rows) {
        if (pressure_action_covered.count(row.global_dof) == 0u) {
            pressure_action_isolated_global_dofs.push_back(row.global_dof);
        }
    }

    std::set<std::pair<GlobalIndex, GlobalIndex>>
        residual_sign_pressure_action_edges;
    std::set<GlobalIndex> residual_sign_pressure_action_covered;
    std::map<GlobalIndex, std::vector<GlobalIndex>>
        residual_sign_pressure_action_neighbors;
    std::size_t residual_opposite_sign_pressure_action_edge_count = 0;
    std::size_t residual_zero_or_missing_sign_pressure_action_edge_count = 0;
    for (const auto& row : direct_rows) {
        residual_sign_pressure_action_neighbors[row.global_dof];
    }
    for (const auto& edge : pressure_action_edges) {
        const int first_sign = residual_sign_by_dof[edge.first];
        const int second_sign = residual_sign_by_dof[edge.second];
        if (first_sign != 0 && second_sign != 0 && first_sign == second_sign) {
            residual_sign_pressure_action_edges.insert(edge);
            residual_sign_pressure_action_covered.insert(edge.first);
            residual_sign_pressure_action_covered.insert(edge.second);
            residual_sign_pressure_action_neighbors[edge.first].push_back(
                edge.second);
            residual_sign_pressure_action_neighbors[edge.second].push_back(
                edge.first);
        } else if (first_sign != 0 && second_sign != 0) {
            ++residual_opposite_sign_pressure_action_edge_count;
        } else {
            ++residual_zero_or_missing_sign_pressure_action_edge_count;
        }
    }

    std::vector<GlobalIndex> residual_sign_pressure_action_covered_global_dofs(
        residual_sign_pressure_action_covered.begin(),
        residual_sign_pressure_action_covered.end());
    std::vector<GlobalIndex> residual_sign_pressure_action_isolated_global_dofs;
    for (const auto& row : direct_rows) {
        if (residual_sign_pressure_action_covered.count(row.global_dof) == 0u) {
            residual_sign_pressure_action_isolated_global_dofs.push_back(
                row.global_dof);
        }
    }

    constexpr GlobalIndex kPressureActionLowDegreeThreshold = 2;
    constexpr GlobalIndex kPressureActionModerateDegreeThreshold = 4;
    constexpr double kPressureActionLowSumRatioThreshold = 0.25;
    constexpr double kPressureActionModerateSumRatioThreshold = 0.5;
    constexpr double kPressureActionSelfDominantThreshold = 0.75;
    GlobalIndex max_pressure_action_degree = 0;
    double max_pressure_action_abs_sum = 0.0;
    std::vector<GlobalIndex> pressure_action_low_degree_global_dofs;
    std::vector<GlobalIndex> pressure_action_moderate_degree_global_dofs;
    std::vector<GlobalIndex> pressure_action_low_sum_ratio_global_dofs;
    std::vector<GlobalIndex> pressure_action_moderate_sum_ratio_global_dofs;
    std::vector<GlobalIndex> pressure_action_self_dominant_global_dofs;
    for (const auto& row : direct_rows) {
        const auto degree = pressure_action_degree_by_dof[row.global_dof];
        const auto action_sum = pressure_action_abs_sum_by_dof[row.global_dof];
        max_pressure_action_degree =
            std::max(max_pressure_action_degree, degree);
        max_pressure_action_abs_sum =
            std::max(max_pressure_action_abs_sum, action_sum);
        if (degree <= kPressureActionLowDegreeThreshold) {
            pressure_action_low_degree_global_dofs.push_back(row.global_dof);
        }
        if (degree <= kPressureActionModerateDegreeThreshold) {
            pressure_action_moderate_degree_global_dofs.push_back(
                row.global_dof);
        }
        const double total_action_support =
            row.row_self_abs_sum + action_sum;
        if (total_action_support > 0.0 &&
            row.row_self_abs_sum / total_action_support >=
                kPressureActionSelfDominantThreshold) {
            pressure_action_self_dominant_global_dofs.push_back(
                row.global_dof);
        }
    }
    if (max_pressure_action_abs_sum > 0.0) {
        for (const auto& row : direct_rows) {
            const double ratio =
                pressure_action_abs_sum_by_dof[row.global_dof] /
                max_pressure_action_abs_sum;
            if (ratio <= kPressureActionLowSumRatioThreshold) {
                pressure_action_low_sum_ratio_global_dofs.push_back(
                    row.global_dof);
            }
            if (ratio <= kPressureActionModerateSumRatioThreshold) {
                pressure_action_moderate_sum_ratio_global_dofs.push_back(
                    row.global_dof);
            }
        }
    }

    std::set<GlobalIndex> sparse_candidate_set(
        sparse_direct_self_global_dofs.begin(),
        sparse_direct_self_global_dofs.end());
    std::set<GlobalIndex> sparse_or_moderate_direct_self_set =
        sparse_candidate_set;
    sparse_or_moderate_direct_self_set.insert(
        moderate_direct_self_ratio_global_dofs.begin(),
        moderate_direct_self_ratio_global_dofs.end());
    std::vector<GlobalIndex> sparse_or_moderate_direct_self_global_dofs(
        sparse_or_moderate_direct_self_set.begin(),
        sparse_or_moderate_direct_self_set.end());

    std::set<GlobalIndex> preferred_candidate_set = sparse_candidate_set;
    preferred_candidate_set.insert(pressure_action_covered.begin(),
                                   pressure_action_covered.end());
    std::vector<GlobalIndex> preferred_candidate_global_dofs(
        preferred_candidate_set.begin(), preferred_candidate_set.end());

    std::map<GlobalIndex, std::vector<GlobalIndex>> pressure_action_neighbors;
    std::map<GlobalIndex, double> direct_self_abs_by_dof;
    for (const auto& row : direct_rows) {
        pressure_action_neighbors[row.global_dof];
        direct_self_abs_by_dof[row.global_dof] = row.row_self_abs_sum;
    }
    for (const auto& edge : pressure_action_edges) {
        pressure_action_neighbors[edge.first].push_back(edge.second);
        pressure_action_neighbors[edge.second].push_back(edge.first);
    }
    std::map<GlobalIndex, std::set<GlobalIndex>> pressure_action_neighbor_sets;
    for (const auto& row : direct_rows) {
        const auto neighbor_it = pressure_action_neighbors.find(row.global_dof);
        if (neighbor_it == pressure_action_neighbors.end()) {
            pressure_action_neighbor_sets[row.global_dof];
            continue;
        }
        pressure_action_neighbor_sets[row.global_dof] =
            std::set<GlobalIndex>(
                neighbor_it->second.begin(), neighbor_it->second.end());
    }
    constexpr GlobalIndex kPressureActionLowTwoHopThreshold = 4;
    constexpr double kPressureActionHighTwoHopRatioThreshold = 0.5;
    constexpr double kPressureActionLowClusteringThreshold = 0.25;
    constexpr double kPressureActionHighClusteringThreshold = 0.75;
    GlobalIndex max_pressure_action_two_hop_completion_count = 0;
    GlobalIndex pressure_action_clustering_eligible_row_count = 0;
    double min_pressure_action_clustering_ratio =
        std::numeric_limits<double>::infinity();
    double max_pressure_action_clustering_ratio = 0.0;
    std::map<GlobalIndex, GlobalIndex> pressure_action_two_hop_count_by_dof;
    std::vector<GlobalIndex> pressure_action_zero_two_hop_global_dofs;
    std::vector<GlobalIndex> pressure_action_low_two_hop_global_dofs;
    std::vector<GlobalIndex> pressure_action_high_two_hop_global_dofs;
    std::vector<GlobalIndex> pressure_action_zero_clustering_global_dofs;
    std::vector<GlobalIndex> pressure_action_low_clustering_global_dofs;
    std::vector<GlobalIndex> pressure_action_high_clustering_global_dofs;
    for (const auto& row : direct_rows) {
        const auto row_it =
            pressure_action_neighbor_sets.find(row.global_dof);
        const auto& row_neighbors = row_it->second;
        std::set<GlobalIndex> two_hop_candidates;
        for (const auto neighbor : row_neighbors) {
            const auto neighbor_it =
                pressure_action_neighbor_sets.find(neighbor);
            if (neighbor_it == pressure_action_neighbor_sets.end()) {
                continue;
            }
            for (const auto candidate : neighbor_it->second) {
                if (candidate == row.global_dof ||
                    row_neighbors.count(candidate) != 0u) {
                    continue;
                }
                two_hop_candidates.insert(candidate);
            }
        }
        const GlobalIndex two_hop_count =
            static_cast<GlobalIndex>(two_hop_candidates.size());
        pressure_action_two_hop_count_by_dof[row.global_dof] = two_hop_count;
        max_pressure_action_two_hop_completion_count =
            std::max(max_pressure_action_two_hop_completion_count,
                     two_hop_count);
        if (two_hop_count == 0) {
            pressure_action_zero_two_hop_global_dofs.push_back(row.global_dof);
        }
        if (two_hop_count <= kPressureActionLowTwoHopThreshold) {
            pressure_action_low_two_hop_global_dofs.push_back(row.global_dof);
        }

        if (row_neighbors.size() < 2u) {
            continue;
        }
        ++pressure_action_clustering_eligible_row_count;
        GlobalIndex neighbor_edge_count = 0;
        std::vector<GlobalIndex> neighbors(
            row_neighbors.begin(), row_neighbors.end());
        for (std::size_t i = 0; i < neighbors.size(); ++i) {
            for (std::size_t j = i + 1u; j < neighbors.size(); ++j) {
                const auto first_neighbor_it =
                    pressure_action_neighbor_sets.find(neighbors[i]);
                if (first_neighbor_it != pressure_action_neighbor_sets.end() &&
                    first_neighbor_it->second.count(neighbors[j]) != 0u) {
                    ++neighbor_edge_count;
                }
            }
        }
        const auto possible_neighbor_edges =
            static_cast<double>(neighbors.size() * (neighbors.size() - 1u)) /
            2.0;
        const double clustering_ratio =
            possible_neighbor_edges > 0.0
                ? static_cast<double>(neighbor_edge_count) /
                      possible_neighbor_edges
                : 0.0;
        min_pressure_action_clustering_ratio =
            std::min(min_pressure_action_clustering_ratio, clustering_ratio);
        max_pressure_action_clustering_ratio =
            std::max(max_pressure_action_clustering_ratio, clustering_ratio);
        if (neighbor_edge_count == 0) {
            pressure_action_zero_clustering_global_dofs.push_back(
                row.global_dof);
        }
        if (clustering_ratio <= kPressureActionLowClusteringThreshold) {
            pressure_action_low_clustering_global_dofs.push_back(
                row.global_dof);
        }
        if (clustering_ratio >= kPressureActionHighClusteringThreshold) {
            pressure_action_high_clustering_global_dofs.push_back(
                row.global_dof);
        }
    }
    if (max_pressure_action_two_hop_completion_count > 0) {
        for (const auto& entry : pressure_action_two_hop_count_by_dof) {
            const double ratio =
                static_cast<double>(entry.second) /
                static_cast<double>(max_pressure_action_two_hop_completion_count);
            if (ratio >= kPressureActionHighTwoHopRatioThreshold) {
                pressure_action_high_two_hop_global_dofs.push_back(entry.first);
            }
        }
    }

    std::map<GlobalIndex, int> pressure_action_discovery_time;
    std::map<GlobalIndex, int> pressure_action_low_link;
    std::map<GlobalIndex, GlobalIndex> pressure_action_dfs_parent;
    std::set<GlobalIndex> pressure_action_articulation_set;
    std::set<GlobalIndex> pressure_action_bridge_endpoint_set;
    int pressure_action_dfs_time = 0;
    std::function<void(GlobalIndex)> visitPressureActionGraph =
        [&](GlobalIndex current) {
            pressure_action_discovery_time[current] =
                ++pressure_action_dfs_time;
            pressure_action_low_link[current] =
                pressure_action_discovery_time[current];
            int child_count = 0;
            const auto neighbor_it =
                pressure_action_neighbor_sets.find(current);
            if (neighbor_it == pressure_action_neighbor_sets.end()) {
                return;
            }
            for (const auto neighbor : neighbor_it->second) {
                if (pressure_action_discovery_time.count(neighbor) == 0u) {
                    pressure_action_dfs_parent[neighbor] = current;
                    ++child_count;
                    visitPressureActionGraph(neighbor);
                    pressure_action_low_link[current] =
                        std::min(pressure_action_low_link[current],
                                 pressure_action_low_link[neighbor]);
                    const bool is_root =
                        pressure_action_dfs_parent.count(current) == 0u;
                    if (is_root && child_count > 1) {
                        pressure_action_articulation_set.insert(current);
                    }
                    if (!is_root &&
                        pressure_action_low_link[neighbor] >=
                            pressure_action_discovery_time[current]) {
                        pressure_action_articulation_set.insert(current);
                    }
                    if (pressure_action_low_link[neighbor] >
                        pressure_action_discovery_time[current]) {
                        pressure_action_bridge_endpoint_set.insert(current);
                        pressure_action_bridge_endpoint_set.insert(neighbor);
                    }
                } else {
                    const auto parent_it =
                        pressure_action_dfs_parent.find(current);
                    if (parent_it == pressure_action_dfs_parent.end() ||
                        parent_it->second != neighbor) {
                        pressure_action_low_link[current] =
                            std::min(pressure_action_low_link[current],
                                     pressure_action_discovery_time[neighbor]);
                    }
                }
            }
        };
    for (const auto& row : direct_rows) {
        if (pressure_action_discovery_time.count(row.global_dof) == 0u) {
            visitPressureActionGraph(row.global_dof);
        }
    }
    std::vector<GlobalIndex> pressure_action_articulation_global_dofs(
        pressure_action_articulation_set.begin(),
        pressure_action_articulation_set.end());
    std::vector<GlobalIndex> pressure_action_bridge_endpoint_global_dofs(
        pressure_action_bridge_endpoint_set.begin(),
        pressure_action_bridge_endpoint_set.end());
    auto expandPressureActionNeighborhood =
        [&](const std::set<GlobalIndex>& seeds,
            int radius) -> std::vector<GlobalIndex> {
        std::set<GlobalIndex> selected = seeds;
        std::vector<GlobalIndex> frontier(seeds.begin(), seeds.end());
        for (int depth = 0; depth < radius && !frontier.empty(); ++depth) {
            std::vector<GlobalIndex> next_frontier;
            for (const auto current : frontier) {
                const auto neighbor_it = pressure_action_neighbors.find(current);
                if (neighbor_it == pressure_action_neighbors.end()) {
                    continue;
                }
                for (const auto neighbor : neighbor_it->second) {
                    if (selected.insert(neighbor).second) {
                        next_frontier.push_back(neighbor);
                    }
                }
            }
            frontier = std::move(next_frontier);
        }
        return std::vector<GlobalIndex>(selected.begin(), selected.end());
    };
    const auto sparse_seeded_pressure_action_radius1_global_dofs =
        expandPressureActionNeighborhood(sparse_candidate_set, 1);
    const auto sparse_seeded_pressure_action_radius2_global_dofs =
        expandPressureActionNeighborhood(sparse_candidate_set, 2);
    constexpr double kGraphLocalLowDirectSelfRatioThreshold = 0.5;
    constexpr double kGraphLocalModerateDirectSelfRatioThreshold = 0.75;
    std::vector<GlobalIndex> graph_local_low_direct_self_ratio_global_dofs;
    std::vector<GlobalIndex> graph_local_moderate_direct_self_ratio_global_dofs;
    GlobalIndex graph_local_neighbor_positive_row_count = 0;
    for (const auto& row : direct_rows) {
        const auto neighbor_it =
            pressure_action_neighbors.find(row.global_dof);
        if (neighbor_it == pressure_action_neighbors.end() ||
            neighbor_it->second.empty()) {
            continue;
        }
        double max_neighbor_self_abs_sum = 0.0;
        for (const auto neighbor : neighbor_it->second) {
            const auto support_it = direct_self_abs_by_dof.find(neighbor);
            if (support_it != direct_self_abs_by_dof.end()) {
                max_neighbor_self_abs_sum =
                    std::max(max_neighbor_self_abs_sum, support_it->second);
            }
        }
        if (max_neighbor_self_abs_sum <= 0.0) {
            continue;
        }
        ++graph_local_neighbor_positive_row_count;
        const double local_ratio =
            row.row_self_abs_sum / max_neighbor_self_abs_sum;
        if (local_ratio <= kGraphLocalLowDirectSelfRatioThreshold) {
            graph_local_low_direct_self_ratio_global_dofs.push_back(
                row.global_dof);
        }
        if (local_ratio <= kGraphLocalModerateDirectSelfRatioThreshold) {
            graph_local_moderate_direct_self_ratio_global_dofs.push_back(
                row.global_dof);
        }
    }
    std::set<GlobalIndex> visited_pressure_action_rows;
    std::vector<GlobalIndex> sparse_seeded_pressure_action_component_global_dofs;
    std::size_t pressure_action_component_count = 0;
    std::size_t pressure_action_largest_component_size = 0;
    std::size_t sparse_seeded_pressure_action_component_count = 0;
    for (const auto& row : direct_rows) {
        if (visited_pressure_action_rows.count(row.global_dof) != 0u) {
            continue;
        }
        ++pressure_action_component_count;
        std::vector<GlobalIndex> component;
        std::vector<GlobalIndex> stack{row.global_dof};
        visited_pressure_action_rows.insert(row.global_dof);
        bool has_sparse_seed = false;
        while (!stack.empty()) {
            const GlobalIndex current = stack.back();
            stack.pop_back();
            component.push_back(current);
            if (sparse_candidate_set.count(current) != 0u) {
                has_sparse_seed = true;
            }
            const auto neighbor_it = pressure_action_neighbors.find(current);
            if (neighbor_it == pressure_action_neighbors.end()) {
                continue;
            }
            for (const auto neighbor : neighbor_it->second) {
                if (visited_pressure_action_rows.insert(neighbor).second) {
                    stack.push_back(neighbor);
                }
            }
        }
        pressure_action_largest_component_size =
            std::max(pressure_action_largest_component_size, component.size());
        if (has_sparse_seed) {
            ++sparse_seeded_pressure_action_component_count;
            sparse_seeded_pressure_action_component_global_dofs.insert(
                sparse_seeded_pressure_action_component_global_dofs.end(),
                component.begin(),
                component.end());
        }
    }
    std::sort(
        sparse_seeded_pressure_action_component_global_dofs.begin(),
        sparse_seeded_pressure_action_component_global_dofs.end());
    sparse_seeded_pressure_action_component_global_dofs.erase(
        std::unique(
            sparse_seeded_pressure_action_component_global_dofs.begin(),
            sparse_seeded_pressure_action_component_global_dofs.end()),
        sparse_seeded_pressure_action_component_global_dofs.end());

    std::set<GlobalIndex> residual_sign_preferred_candidate_set =
        sparse_candidate_set;
    residual_sign_preferred_candidate_set.insert(
        residual_sign_pressure_action_covered.begin(),
        residual_sign_pressure_action_covered.end());
    std::vector<GlobalIndex> residual_sign_preferred_candidate_global_dofs(
        residual_sign_preferred_candidate_set.begin(),
        residual_sign_preferred_candidate_set.end());

    std::set<GlobalIndex> visited_residual_sign_pressure_action_rows;
    std::vector<GlobalIndex>
        sparse_seeded_residual_sign_pressure_action_component_global_dofs;
    std::size_t residual_sign_pressure_action_component_count = 0;
    std::size_t residual_sign_pressure_action_largest_component_size = 0;
    std::size_t
        sparse_seeded_residual_sign_pressure_action_component_count = 0;
    for (const auto& row : direct_rows) {
        if (visited_residual_sign_pressure_action_rows.count(row.global_dof) !=
            0u) {
            continue;
        }
        ++residual_sign_pressure_action_component_count;
        std::vector<GlobalIndex> component;
        std::vector<GlobalIndex> stack{row.global_dof};
        visited_residual_sign_pressure_action_rows.insert(row.global_dof);
        bool has_sparse_seed = false;
        while (!stack.empty()) {
            const GlobalIndex current = stack.back();
            stack.pop_back();
            component.push_back(current);
            if (sparse_candidate_set.count(current) != 0u) {
                has_sparse_seed = true;
            }
            const auto neighbor_it =
                residual_sign_pressure_action_neighbors.find(current);
            if (neighbor_it ==
                residual_sign_pressure_action_neighbors.end()) {
                continue;
            }
            for (const auto neighbor : neighbor_it->second) {
                if (visited_residual_sign_pressure_action_rows.insert(neighbor)
                        .second) {
                    stack.push_back(neighbor);
                }
            }
        }
        residual_sign_pressure_action_largest_component_size =
            std::max(residual_sign_pressure_action_largest_component_size,
                     component.size());
        if (has_sparse_seed) {
            ++sparse_seeded_residual_sign_pressure_action_component_count;
            sparse_seeded_residual_sign_pressure_action_component_global_dofs
                .insert(
                    sparse_seeded_residual_sign_pressure_action_component_global_dofs
                        .end(),
                    component.begin(),
                    component.end());
        }
    }
    std::sort(
        sparse_seeded_residual_sign_pressure_action_component_global_dofs
            .begin(),
        sparse_seeded_residual_sign_pressure_action_component_global_dofs.end());
    sparse_seeded_residual_sign_pressure_action_component_global_dofs.erase(
        std::unique(
            sparse_seeded_residual_sign_pressure_action_component_global_dofs
                .begin(),
            sparse_seeded_residual_sign_pressure_action_component_global_dofs
                .end()),
        sparse_seeded_residual_sign_pressure_action_component_global_dofs.end());

    constexpr std::size_t kCandidateSampleLimit = 2048;

    std::ostringstream oss;
    oss << std::setprecision(17);
    oss << "NewtonSolver: direct PSPG formulation candidate diagnostic"
        << " diagnostic=direct_pspg_formulation_candidate"
        << " status=ok"
        << " rank=" << mpiRank()
        << " iteration=" << iteration
        << " phase='" << phase << "'"
        << " op='" << op << "'"
        << " backend=" << backends::backendKindToString(matrix.backendKind())
        << " solve_time=" << solve_time
        << " dt=" << dt
        << " pressure_field='" << pressure_field << "'"
        << " pressure_offset=" << pressure_range->begin
        << " pressure_dofs=" << (pressure_range->end - pressure_range->begin)
        << " tolerance=" << tolerance
        << " selector='sparse_direct_self_or_matrix_pressure_action_patch'"
        << " direct_self_positive_row_count=" << direct_rows.size()
        << " min_direct_self_numeric_entries="
        << (direct_rows.empty() ? 0 : min_direct_self_entries)
        << " max_direct_self_numeric_entries=" << max_direct_self_entries
        << " max_unconstrained_direct_self_numeric_entries="
        << max_unconstrained_direct_self_entries
        << " min_positive_direct_self_abs_sum="
        << (std::isfinite(min_positive_direct_self_abs_sum)
                ? min_positive_direct_self_abs_sum
                : 0.0)
        << " max_direct_self_abs_sum=" << max_direct_self_abs_sum
        << " direct_self_row_sum_leak_threshold="
        << kDirectSelfRowSumLeakThreshold
        << " direct_self_null_preserving_threshold="
        << kDirectSelfNullPreservingThreshold
        << " direct_self_diag_dominant_threshold="
        << kDirectSelfDiagDominantThreshold
        << " direct_self_balanced_diag_low_threshold="
        << kDirectSelfBalancedDiagLowThreshold
        << " direct_self_balanced_diag_high_threshold="
        << kDirectSelfBalancedDiagHighThreshold
        << " max_direct_self_row_sum_leak_ratio="
        << max_direct_self_row_sum_leak_ratio
        << " min_direct_self_diag_abs_ratio="
        << (std::isfinite(min_direct_self_diag_abs_ratio)
                ? min_direct_self_diag_abs_ratio
                : 0.0)
        << " max_direct_self_diag_abs_ratio="
        << max_direct_self_diag_abs_ratio
        << " high_direct_self_row_sum_leak_candidate_count="
        << high_direct_self_row_sum_leak_global_dofs.size()
        << " null_preserving_direct_self_candidate_count="
        << null_preserving_direct_self_global_dofs.size()
        << " diag_dominant_direct_self_candidate_count="
        << diag_dominant_direct_self_global_dofs.size()
        << " balanced_diag_direct_self_candidate_count="
        << balanced_diag_direct_self_global_dofs.size()
        << " sparse_direct_self_candidate_count="
        << sparse_direct_self_global_dofs.size()
        << " constrained_pressure_neighbor_candidate_count="
        << constrained_pressure_neighbor_global_dofs.size()
        << " constrained_pressure_neighbor_ratio_threshold="
        << kConstrainedPressureNeighborRatioThreshold
        << " high_constrained_pressure_neighbor_ratio_candidate_count="
        << high_constrained_pressure_neighbor_ratio_global_dofs.size()
        << " sparse_unconstrained_direct_self_candidate_count="
        << sparse_unconstrained_direct_self_global_dofs.size()
        << " constrained_or_sparse_unconstrained_direct_self_candidate_count="
        << constrained_or_sparse_unconstrained_direct_self_global_dofs.size()
        << " direct_self_low_ratio_threshold="
        << kLowDirectSelfRatioThreshold
        << " direct_self_moderate_ratio_threshold="
        << kModerateDirectSelfRatioThreshold
        << " low_direct_self_ratio_candidate_count="
        << low_direct_self_ratio_global_dofs.size()
        << " moderate_direct_self_ratio_candidate_count="
        << moderate_direct_self_ratio_global_dofs.size()
        << " sparse_or_moderate_direct_self_ratio_candidate_count="
        << sparse_or_moderate_direct_self_global_dofs.size()
        << " sparse_seeded_pressure_action_radius1_candidate_count="
        << sparse_seeded_pressure_action_radius1_global_dofs.size()
        << " sparse_seeded_pressure_action_radius2_candidate_count="
        << sparse_seeded_pressure_action_radius2_global_dofs.size()
        << " graph_local_direct_self_low_ratio_threshold="
        << kGraphLocalLowDirectSelfRatioThreshold
        << " graph_local_direct_self_moderate_ratio_threshold="
        << kGraphLocalModerateDirectSelfRatioThreshold
        << " graph_local_neighbor_positive_row_count="
        << graph_local_neighbor_positive_row_count
        << " graph_local_low_direct_self_ratio_candidate_count="
        << graph_local_low_direct_self_ratio_global_dofs.size()
        << " graph_local_moderate_direct_self_ratio_candidate_count="
        << graph_local_moderate_direct_self_ratio_global_dofs.size()
        << " matrix_pressure_action_edge_count="
        << pressure_action_edges.size()
        << " matrix_pressure_action_max_degree="
        << max_pressure_action_degree
        << " matrix_pressure_action_max_abs_sum="
        << max_pressure_action_abs_sum
        << " pressure_action_low_degree_threshold="
        << kPressureActionLowDegreeThreshold
        << " pressure_action_moderate_degree_threshold="
        << kPressureActionModerateDegreeThreshold
        << " pressure_action_low_degree_candidate_count="
        << pressure_action_low_degree_global_dofs.size()
        << " pressure_action_moderate_degree_candidate_count="
        << pressure_action_moderate_degree_global_dofs.size()
        << " pressure_action_low_sum_ratio_threshold="
        << kPressureActionLowSumRatioThreshold
        << " pressure_action_moderate_sum_ratio_threshold="
        << kPressureActionModerateSumRatioThreshold
        << " pressure_action_low_sum_ratio_candidate_count="
        << pressure_action_low_sum_ratio_global_dofs.size()
        << " pressure_action_moderate_sum_ratio_candidate_count="
        << pressure_action_moderate_sum_ratio_global_dofs.size()
        << " pressure_action_self_dominant_threshold="
        << kPressureActionSelfDominantThreshold
        << " pressure_action_self_dominant_candidate_count="
        << pressure_action_self_dominant_global_dofs.size()
        << " pressure_action_low_two_hop_threshold="
        << kPressureActionLowTwoHopThreshold
        << " pressure_action_high_two_hop_ratio_threshold="
        << kPressureActionHighTwoHopRatioThreshold
        << " matrix_pressure_action_max_two_hop_completion_count="
        << max_pressure_action_two_hop_completion_count
        << " pressure_action_zero_two_hop_candidate_count="
        << pressure_action_zero_two_hop_global_dofs.size()
        << " pressure_action_low_two_hop_candidate_count="
        << pressure_action_low_two_hop_global_dofs.size()
        << " pressure_action_high_two_hop_candidate_count="
        << pressure_action_high_two_hop_global_dofs.size()
        << " pressure_action_low_clustering_threshold="
        << kPressureActionLowClusteringThreshold
        << " pressure_action_high_clustering_threshold="
        << kPressureActionHighClusteringThreshold
        << " pressure_action_clustering_eligible_row_count="
        << pressure_action_clustering_eligible_row_count
        << " matrix_pressure_action_min_clustering_ratio="
        << (std::isfinite(min_pressure_action_clustering_ratio)
                ? min_pressure_action_clustering_ratio
                : 0.0)
        << " matrix_pressure_action_max_clustering_ratio="
        << max_pressure_action_clustering_ratio
        << " pressure_action_zero_clustering_candidate_count="
        << pressure_action_zero_clustering_global_dofs.size()
        << " pressure_action_low_clustering_candidate_count="
        << pressure_action_low_clustering_global_dofs.size()
        << " pressure_action_high_clustering_candidate_count="
        << pressure_action_high_clustering_global_dofs.size()
        << " pressure_action_articulation_candidate_count="
        << pressure_action_articulation_global_dofs.size()
        << " pressure_action_bridge_endpoint_candidate_count="
        << pressure_action_bridge_endpoint_global_dofs.size()
        << " matrix_pressure_action_component_count="
        << pressure_action_component_count
        << " matrix_pressure_action_largest_component_size="
        << pressure_action_largest_component_size
        << " matrix_pressure_action_covered_count="
        << pressure_action_covered_global_dofs.size()
        << " matrix_pressure_action_isolated_count="
        << pressure_action_isolated_global_dofs.size()
        << " sparse_seeded_matrix_pressure_action_component_count="
        << sparse_seeded_pressure_action_component_count
        << " sparse_seeded_matrix_pressure_action_component_dof_count="
        << sparse_seeded_pressure_action_component_global_dofs.size()
        << " residual_sign_threshold=" << residual_sign_threshold
        << " residual_positive_direct_row_count="
        << residual_positive_direct_row_count
        << " residual_negative_direct_row_count="
        << residual_negative_direct_row_count
        << " residual_zero_direct_row_count="
        << residual_zero_direct_row_count
        << " residual_nonfinite_direct_row_count="
        << residual_nonfinite_direct_row_count
        << " residual_nonzero_direct_row_count="
        << (residual_positive_direct_row_count +
            residual_negative_direct_row_count)
        << " min_positive_residual_abs="
        << (std::isfinite(min_positive_residual_abs)
                ? min_positive_residual_abs
                : 0.0)
        << " max_residual_abs=" << max_residual_abs
        << " residual_sign_pressure_action_edge_count="
        << residual_sign_pressure_action_edges.size()
        << " residual_opposite_sign_pressure_action_edge_count="
        << residual_opposite_sign_pressure_action_edge_count
        << " residual_zero_or_missing_sign_pressure_action_edge_count="
        << residual_zero_or_missing_sign_pressure_action_edge_count
        << " residual_sign_pressure_action_component_count="
        << residual_sign_pressure_action_component_count
        << " residual_sign_pressure_action_largest_component_size="
        << residual_sign_pressure_action_largest_component_size
        << " residual_sign_pressure_action_covered_count="
        << residual_sign_pressure_action_covered_global_dofs.size()
        << " residual_sign_pressure_action_isolated_count="
        << residual_sign_pressure_action_isolated_global_dofs.size()
        << " sparse_seeded_residual_sign_pressure_action_component_count="
        << sparse_seeded_residual_sign_pressure_action_component_count
        << " sparse_seeded_residual_sign_pressure_action_component_dof_count="
        << sparse_seeded_residual_sign_pressure_action_component_global_dofs
               .size()
        << " sparse_direct_self_or_residual_sign_pressure_action_candidate_count="
        << residual_sign_preferred_candidate_global_dofs.size()
        << " preferred_candidate_count="
        << preferred_candidate_global_dofs.size()
        << " artifact_limitation='matrix-sign and residual-sign pressure-action proxies; no update signs'"
        << " high_direct_self_row_sum_leak_global_dofs="
        << formatGlobalIndexListSample(
               high_direct_self_row_sum_leak_global_dofs,
               kCandidateSampleLimit)
        << " null_preserving_direct_self_global_dofs="
        << formatGlobalIndexListSample(
               null_preserving_direct_self_global_dofs,
               kCandidateSampleLimit)
        << " diag_dominant_direct_self_global_dofs="
        << formatGlobalIndexListSample(
               diag_dominant_direct_self_global_dofs,
               kCandidateSampleLimit)
        << " balanced_diag_direct_self_global_dofs="
        << formatGlobalIndexListSample(
               balanced_diag_direct_self_global_dofs,
               kCandidateSampleLimit)
        << " sparse_direct_self_global_dofs="
        << formatGlobalIndexListSample(
               sparse_direct_self_global_dofs,
               kCandidateSampleLimit)
        << " constrained_pressure_neighbor_global_dofs="
        << formatGlobalIndexListSample(
               constrained_pressure_neighbor_global_dofs,
               kCandidateSampleLimit)
        << " high_constrained_pressure_neighbor_ratio_global_dofs="
        << formatGlobalIndexListSample(
               high_constrained_pressure_neighbor_ratio_global_dofs,
               kCandidateSampleLimit)
        << " sparse_unconstrained_direct_self_global_dofs="
        << formatGlobalIndexListSample(
               sparse_unconstrained_direct_self_global_dofs,
               kCandidateSampleLimit)
        << " constrained_or_sparse_unconstrained_direct_self_global_dofs="
        << formatGlobalIndexListSample(
               constrained_or_sparse_unconstrained_direct_self_global_dofs,
               kCandidateSampleLimit)
        << " low_direct_self_ratio_global_dofs="
        << formatGlobalIndexListSample(
               low_direct_self_ratio_global_dofs,
               kCandidateSampleLimit)
        << " moderate_direct_self_ratio_global_dofs="
        << formatGlobalIndexListSample(
               moderate_direct_self_ratio_global_dofs,
               kCandidateSampleLimit)
        << " sparse_or_moderate_direct_self_ratio_global_dofs="
        << formatGlobalIndexListSample(
               sparse_or_moderate_direct_self_global_dofs,
               kCandidateSampleLimit)
        << " sparse_seeded_pressure_action_radius1_global_dofs="
        << formatGlobalIndexListSample(
               sparse_seeded_pressure_action_radius1_global_dofs,
               kCandidateSampleLimit)
        << " sparse_seeded_pressure_action_radius2_global_dofs="
        << formatGlobalIndexListSample(
               sparse_seeded_pressure_action_radius2_global_dofs,
               kCandidateSampleLimit)
        << " graph_local_low_direct_self_ratio_global_dofs="
        << formatGlobalIndexListSample(
               graph_local_low_direct_self_ratio_global_dofs,
               kCandidateSampleLimit)
        << " graph_local_moderate_direct_self_ratio_global_dofs="
        << formatGlobalIndexListSample(
               graph_local_moderate_direct_self_ratio_global_dofs,
               kCandidateSampleLimit)
        << " matrix_pressure_action_covered_global_dofs="
        << formatGlobalIndexListSample(
               pressure_action_covered_global_dofs,
               kCandidateSampleLimit)
        << " pressure_action_low_degree_global_dofs="
        << formatGlobalIndexListSample(
               pressure_action_low_degree_global_dofs,
               kCandidateSampleLimit)
        << " pressure_action_moderate_degree_global_dofs="
        << formatGlobalIndexListSample(
               pressure_action_moderate_degree_global_dofs,
               kCandidateSampleLimit)
        << " pressure_action_low_sum_ratio_global_dofs="
        << formatGlobalIndexListSample(
               pressure_action_low_sum_ratio_global_dofs,
               kCandidateSampleLimit)
        << " pressure_action_moderate_sum_ratio_global_dofs="
        << formatGlobalIndexListSample(
               pressure_action_moderate_sum_ratio_global_dofs,
               kCandidateSampleLimit)
        << " pressure_action_self_dominant_global_dofs="
        << formatGlobalIndexListSample(
               pressure_action_self_dominant_global_dofs,
               kCandidateSampleLimit)
        << " pressure_action_zero_two_hop_global_dofs="
        << formatGlobalIndexListSample(
               pressure_action_zero_two_hop_global_dofs,
               kCandidateSampleLimit)
        << " pressure_action_low_two_hop_global_dofs="
        << formatGlobalIndexListSample(
               pressure_action_low_two_hop_global_dofs,
               kCandidateSampleLimit)
        << " pressure_action_high_two_hop_global_dofs="
        << formatGlobalIndexListSample(
               pressure_action_high_two_hop_global_dofs,
               kCandidateSampleLimit)
        << " pressure_action_zero_clustering_global_dofs="
        << formatGlobalIndexListSample(
               pressure_action_zero_clustering_global_dofs,
               kCandidateSampleLimit)
        << " pressure_action_low_clustering_global_dofs="
        << formatGlobalIndexListSample(
               pressure_action_low_clustering_global_dofs,
               kCandidateSampleLimit)
        << " pressure_action_high_clustering_global_dofs="
        << formatGlobalIndexListSample(
               pressure_action_high_clustering_global_dofs,
               kCandidateSampleLimit)
        << " pressure_action_articulation_global_dofs="
        << formatGlobalIndexListSample(
               pressure_action_articulation_global_dofs,
               kCandidateSampleLimit)
        << " pressure_action_bridge_endpoint_global_dofs="
        << formatGlobalIndexListSample(
               pressure_action_bridge_endpoint_global_dofs,
               kCandidateSampleLimit)
        << " matrix_pressure_action_isolated_global_dofs="
        << formatGlobalIndexListSample(
               pressure_action_isolated_global_dofs,
               kCandidateSampleLimit)
        << " sparse_seeded_matrix_pressure_action_component_global_dofs="
        << formatGlobalIndexListSample(
               sparse_seeded_pressure_action_component_global_dofs,
               kCandidateSampleLimit)
        << " residual_sign_pressure_action_covered_global_dofs="
        << formatGlobalIndexListSample(
               residual_sign_pressure_action_covered_global_dofs,
               kCandidateSampleLimit)
        << " residual_sign_pressure_action_isolated_global_dofs="
        << formatGlobalIndexListSample(
               residual_sign_pressure_action_isolated_global_dofs,
               kCandidateSampleLimit)
        << " sparse_seeded_residual_sign_pressure_action_component_global_dofs="
        << formatGlobalIndexListSample(
               sparse_seeded_residual_sign_pressure_action_component_global_dofs,
               kCandidateSampleLimit)
        << " sparse_direct_self_or_residual_sign_pressure_action_global_dofs="
        << formatGlobalIndexListSample(
               residual_sign_preferred_candidate_global_dofs,
               kCandidateSampleLimit)
        << " preferred_candidate_global_dofs="
        << formatGlobalIndexListSample(
               preferred_candidate_global_dofs,
               kCandidateSampleLimit);
    FE_LOG_INFO(oss.str());
}

[[nodiscard]] ActivePressureUpdateSupportSummary scanActivePressureUpdateSupport(
    const systems::FESystem& sys,
    const backends::GenericMatrix& matrix,
    backends::GenericVector& update,
    backends::GenericVector& rhs,
    std::span<const GlobalIndex> constrained_dofs,
    std::string_view pressure_field_name,
    std::string_view coupling_field_name,
    double tolerance,
    double weak_coupling_threshold,
    double weak_self_threshold,
    int sample_limit,
    int action_sample_limit)
{
    ActivePressureUpdateSupportSummary summary;
    summary.pressure_field = std::string(pressure_field_name);
    summary.coupling_field = std::string(coupling_field_name);

    const auto pressure_range =
        newtonMatrixSupportFieldRangeByName(sys, pressure_field_name);
    const auto coupling_range =
        newtonMatrixSupportFieldRangeByName(sys, coupling_field_name);
    if (!pressure_range.has_value() || !coupling_range.has_value()) {
        return summary;
    }

    summary.pressure_offset = pressure_range->begin;
    summary.pressure_dofs = pressure_range->end - pressure_range->begin;
    summary.coupling_offset = coupling_range->begin;
    summary.coupling_dofs = coupling_range->end - coupling_range->begin;

    auto update_view = update.createAssemblyView();
    FE_CHECK_NOT_NULL(update_view.get(),
                      "NewtonSolver: pressure update support diagnostic view");
    auto rhs_view = rhs.createAssemblyView();
    FE_CHECK_NOT_NULL(rhs_view.get(),
                      "NewtonSolver: pressure update support rhs diagnostic view");

    const auto add_top_sample =
        [sample_limit](std::vector<ActivePressureUpdateSupportRow>& samples,
                       const ActivePressureUpdateSupportRow& row) {
            if (sample_limit == 0) {
                return;
            }
            samples.push_back(row);
            std::sort(samples.begin(), samples.end(),
                      [](const ActivePressureUpdateSupportRow& a,
                         const ActivePressureUpdateSupportRow& b) {
                          if (a.abs_update == b.abs_update) {
                              return a.global_dof < b.global_dof;
                          }
                          return a.abs_update > b.abs_update;
                      });
            if (sample_limit > 0 &&
                static_cast<int>(samples.size()) > sample_limit) {
                samples.pop_back();
            }
        };
    const auto add_action_term =
        [action_sample_limit](std::vector<ActivePressureUpdateActionTerm>& terms,
                              const ActivePressureUpdateActionTerm& term) {
            if (action_sample_limit == 0) {
                return;
            }
            terms.push_back(term);
            std::sort(terms.begin(), terms.end(),
                      [](const ActivePressureUpdateActionTerm& a,
                         const ActivePressureUpdateActionTerm& b) {
                          const double abs_a = std::abs(a.action);
                          const double abs_b = std::abs(b.action);
                          if (abs_a == abs_b) {
                              return a.global_dof < b.global_dof;
                          }
                          return abs_a > abs_b;
                      });
            if (action_sample_limit > 0 &&
                static_cast<int>(terms.size()) > action_sample_limit) {
                terms.pop_back();
            }
        };

    for (GlobalIndex local_dof = 0; local_dof < summary.pressure_dofs; ++local_dof) {
        const auto global_dof = summary.pressure_offset + local_dof;
        if (std::binary_search(
                constrained_dofs.begin(), constrained_dofs.end(), global_dof)) {
            ++summary.constrained_pressure_rows;
            continue;
        }

        ++summary.unconstrained_pressure_rows;
        ActivePressureUpdateSupportRow row;
        row.local_dof = local_dof;
        row.global_dof = global_dof;
        if (global_dof >= 0 && global_dof < update.size()) {
            row.update =
                static_cast<double>(update_view->getVectorEntry(global_dof));
            row.abs_update = std::abs(row.update);
        }
        if (global_dof >= 0 && global_dof < rhs.size()) {
            row.rhs = static_cast<double>(rhs_view->getVectorEntry(global_dof));
        }

        if (global_dof >= 0 &&
            global_dof < matrix.numRows() &&
            global_dof < matrix.numCols()) {
            for (GlobalIndex col = 0; col < matrix.numCols(); ++col) {
                const Real value = matrix.getEntry(global_dof, col);
                if (!std::isfinite(value)) {
                    continue;
                }
                const double abs_value = std::abs(static_cast<double>(value));
                double update_value = 0.0;
                if (col >= 0 && col < update.size()) {
                    update_value =
                        static_cast<double>(update_view->getVectorEntry(col));
                }
                const double action =
                    static_cast<double>(value) * update_value;
                row.row_action += action;
                row.row_abs_sum += abs_value;
                if (dofInFieldRange(col, *coupling_range)) {
                    row.row_coupling_abs_sum += abs_value;
                    row.row_coupling_action += action;
                    add_action_term(
                        row.coupling_action_terms,
                        ActivePressureUpdateActionTerm{
                            col - coupling_range->begin,
                            col,
                            static_cast<double>(value),
                            update_value,
                            action});
                }
                if (dofInFieldRange(col, *pressure_range)) {
                    row.row_self_abs_sum += abs_value;
                    row.row_self_sum += static_cast<double>(value);
                    row.row_self_action += action;
                    add_action_term(
                        row.pressure_action_terms,
                        ActivePressureUpdateActionTerm{
                            col - pressure_range->begin,
                            col,
                            static_cast<double>(value),
                            update_value,
                            action});
                }
            }
            for (GlobalIndex matrix_row = 0; matrix_row < matrix.numRows();
                 ++matrix_row) {
                const Real value = matrix.getEntry(matrix_row, global_dof);
                if (!std::isfinite(value)) {
                    continue;
                }
                const double abs_value = std::abs(static_cast<double>(value));
                row.col_abs_sum += abs_value;
                if (dofInFieldRange(matrix_row, *coupling_range)) {
                    row.col_coupling_abs_sum += abs_value;
                }
                if (dofInFieldRange(matrix_row, *pressure_range)) {
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
        }
        row.row_self_constant_action = row.row_self_sum * row.update;
        row.row_self_nonconstant_action =
            row.row_self_action - row.row_self_constant_action;
        row.row_other_action =
            row.row_action - row.row_coupling_action - row.row_self_action;
        row.row_linear_residual = row.row_action - row.rhs;

        if (row.abs_update > summary.max_abs_update) {
            summary.max_abs_update = row.abs_update;
            summary.max_update_local_dof = row.local_dof;
            summary.max_update_global_dof = row.global_dof;
            summary.max_update_rhs = row.rhs;
            summary.max_update_row_action = row.row_action;
            summary.max_update_row_coupling_action = row.row_coupling_action;
            summary.max_update_row_self_action = row.row_self_action;
            summary.max_update_row_self_constant_action =
                row.row_self_constant_action;
            summary.max_update_row_self_nonconstant_action =
                row.row_self_nonconstant_action;
            summary.max_update_row_other_action = row.row_other_action;
            summary.max_update_row_linear_residual = row.row_linear_residual;
        }

        const bool zero_coupling =
            std::abs(row.row_coupling_abs_sum) <= tolerance;
        const bool weak_coupling =
            !zero_coupling &&
            weak_coupling_threshold >= 0.0 &&
            row.row_coupling_abs_sum <= weak_coupling_threshold;
        if (zero_coupling) {
            ++summary.zero_coupling_row_block_count;
            summary.zero_coupling_max_abs_update =
                std::max(summary.zero_coupling_max_abs_update, row.abs_update);
        } else if (weak_coupling) {
            ++summary.weak_coupling_row_block_count;
            summary.weak_coupling_max_abs_update =
                std::max(summary.weak_coupling_max_abs_update, row.abs_update);
        } else {
            ++summary.positive_coupling_row_block_count;
            summary.positive_coupling_max_abs_update =
                std::max(summary.positive_coupling_max_abs_update, row.abs_update);
        }

        const bool zero_self = std::abs(row.row_self_abs_sum) <= tolerance;
        const bool weak_self =
            !zero_self &&
            weak_self_threshold >= 0.0 &&
            row.row_self_abs_sum <= weak_self_threshold;
        if (zero_self) {
            ++summary.zero_self_row_block_count;
            summary.zero_self_max_abs_update =
                std::max(summary.zero_self_max_abs_update, row.abs_update);
        } else if (weak_self) {
            ++summary.weak_self_row_block_count;
            summary.weak_self_max_abs_update =
                std::max(summary.weak_self_max_abs_update, row.abs_update);
        } else {
            ++summary.positive_self_row_block_count;
            summary.positive_self_max_abs_update =
                std::max(summary.positive_self_max_abs_update, row.abs_update);
        }

        add_top_sample(summary.top_update_samples, row);
    }

    addSameSignPressureActionComponentSummary(summary);

    return summary;
}

void logActivePressureSupportRankDiagnostic(
    const systems::FESystem& sys,
    const backends::GenericMatrix& matrix,
    std::span<const GlobalIndex> constrained_dofs,
    std::string_view phase,
    int iteration,
    double solve_time,
    double dt)
{
    if (!activePressureSupportRankDiagnosticRequested()) {
        return;
    }

    const auto& pressure_field = activePressureSupportRankPressureFieldName();
    const auto& coupling_field = activePressureSupportRankCouplingFieldName();
    const double tolerance = activePressureSupportRankTolerance();
    const int sample_limit = activePressureSupportRankSampleLimit();
    const auto summary = scanActivePressureSupportRank(
        sys,
        matrix,
        constrained_dofs,
        pressure_field,
        coupling_field,
        tolerance,
        sample_limit);

    std::ostringstream oss;
    oss << std::setprecision(17);
    oss << "NewtonSolver: active pressure support-rank diagnostic"
        << " diagnostic=active_pressure_support_rank"
        << " rank=" << mpiRank()
        << " iteration=" << iteration
        << " phase='" << phase << "'"
        << " backend=" << backends::backendKindToString(matrix.backendKind())
        << " solve_time=" << solve_time
        << " dt=" << dt
        << " pressure_field='" << summary.pressure_field << "'"
        << " coupling_field='" << summary.coupling_field << "'"
        << " pressure_offset=" << summary.pressure_offset
        << " pressure_dofs=" << summary.pressure_dofs
        << " coupling_offset=" << summary.coupling_offset
        << " coupling_dofs=" << summary.coupling_dofs
        << " constrained_pressure_rows=" << summary.constrained_pressure_rows
        << " unconstrained_pressure_rows=" << summary.unconstrained_pressure_rows
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
        << " weak_coupling_row_block_count="
        << summary.weak_coupling_row_block_count
        << " weak_self_row_block_count="
        << summary.weak_self_row_block_count
        << " weak_coupling_and_self_row_block_count="
        << summary.weak_coupling_and_self_row_block_count
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
        << " pressure_only_row_block_count="
        << summary.pressure_only_row_block_count
        << " pressure_only_col_block_count="
        << summary.pressure_only_col_block_count
        << " tolerance=" << tolerance
        << " zero_coupling_row_local_dofs="
        << formatActivePressureSupportRankDofSamples(
               summary.zero_coupling_row_samples, /*local=*/true)
        << " zero_coupling_row_global_dofs="
        << formatActivePressureSupportRankDofSamples(
               summary.zero_coupling_row_samples, /*local=*/false)
        << " zero_row_local_dofs="
        << formatActivePressureSupportRankDofSamples(
               summary.zero_row_samples, /*local=*/true)
        << " zero_row_global_dofs="
        << formatActivePressureSupportRankDofSamples(
               summary.zero_row_samples, /*local=*/false)
        << " zero_coupling_row_details="
        << formatActivePressureSupportRankRowDetails(
               summary.zero_coupling_row_samples)
        << " weakest_coupling_row_local_dofs="
        << formatActivePressureSupportRankDofSamples(
               summary.weakest_coupling_row_samples, /*local=*/true)
        << " weakest_coupling_row_global_dofs="
        << formatActivePressureSupportRankDofSamples(
               summary.weakest_coupling_row_samples, /*local=*/false)
        << " weakest_coupling_row_details="
        << formatActivePressureSupportRankRowDetails(
               summary.weakest_coupling_row_samples)
        << " weakest_self_row_local_dofs="
        << formatActivePressureSupportRankDofSamples(
               summary.weakest_self_row_samples, /*local=*/true)
        << " weakest_self_row_global_dofs="
        << formatActivePressureSupportRankDofSamples(
               summary.weakest_self_row_samples, /*local=*/false)
        << " weakest_self_row_details="
        << formatActivePressureSupportRankRowDetails(
               summary.weakest_self_row_samples);
    FE_LOG_INFO(oss.str());

    if (activePressureSupportRankGuardEnabled() &&
        summary.zero_coupling_row_block_count >
            activePressureSupportRankAllowedZeroVelocityRows()) {
        std::ostringstream msg;
        msg << "NewtonSolver: active pressure support-rank guard failed for field '"
            << summary.pressure_field << "' against coupling field '"
            << summary.coupling_field << "': unconstrained pressure rows with zero "
            << summary.coupling_field << " row-block support="
            << summary.zero_coupling_row_block_count
            << " allowed="
            << activePressureSupportRankAllowedZeroVelocityRows()
            << " sample_local_dofs="
            << formatActivePressureSupportRankDofSamples(
                   summary.zero_coupling_row_samples, /*local=*/true);
        FE_THROW(FEException, msg.str());
    }
}

void applyActivePressureSupportRankClamp(
    const systems::FESystem& sys,
    backends::GenericMatrix& matrix,
    backends::GenericVector& rhs,
    std::span<const GlobalIndex> constrained_dofs,
    std::string_view phase,
    int iteration,
    double solve_time,
    double dt)
{
    if (!activePressureSupportRankClampEnabled()) {
        return;
    }

    const auto& pressure_field = activePressureSupportRankPressureFieldName();
    const auto& coupling_field = activePressureSupportRankCouplingFieldName();
    const double tolerance = activePressureSupportRankTolerance();
    const double clamp_coupling_threshold =
        activePressureSupportRankClampCouplingThreshold();
    const double clamp_self_threshold =
        activePressureSupportRankClampSelfThreshold();
    const int sample_limit = activePressureSupportRankSampleLimit();
    const auto summary = scanActivePressureSupportRank(
        sys,
        matrix,
        constrained_dofs,
        pressure_field,
        coupling_field,
        tolerance,
        sample_limit,
        clamp_coupling_threshold,
        clamp_self_threshold);

    auto matrix_view = matrix.createAssemblyView();
    FE_CHECK_NOT_NULL(matrix_view.get(),
                      "NewtonSolver: active pressure support-rank clamp matrix view");
    matrix_view->beginAssemblyPhase();
    if (!summary.clamp_candidate_row_global_dofs.empty()) {
        matrix_view->zeroRows(
            std::span<const GlobalIndex>(summary.clamp_candidate_row_global_dofs.data(),
                                         summary.clamp_candidate_row_global_dofs.size()),
            /*set_diagonal=*/true);
    }
    // Candidate rows are rank-local; distributed matrix finalization is not.
    matrix_view->finalizeAssembly();

    zeroVectorEntries(
        std::span<const GlobalIndex>(summary.clamp_candidate_row_global_dofs.data(),
                                     summary.clamp_candidate_row_global_dofs.size()),
        rhs);

    std::ostringstream oss;
    oss << std::setprecision(17);
    oss << "NewtonSolver: active pressure support-rank clamp"
        << " diagnostic=active_pressure_support_rank_clamp"
        << " rank=" << mpiRank()
        << " iteration=" << iteration
        << " phase='" << phase << "'"
        << " backend=" << backends::backendKindToString(matrix.backendKind())
        << " solve_time=" << solve_time
        << " dt=" << dt
        << " pressure_field='" << summary.pressure_field << "'"
        << " coupling_field='" << summary.coupling_field << "'"
        << " clamp_coupling_threshold=" << clamp_coupling_threshold
        << " clamp_self_threshold=" << clamp_self_threshold
        << " clamped_row_count=" << summary.clamp_candidate_row_global_dofs.size()
        << " constrained_pressure_rows=" << summary.constrained_pressure_rows
        << " unconstrained_pressure_rows=" << summary.unconstrained_pressure_rows
        << " zero_coupling_row_block_count="
        << summary.zero_coupling_row_block_count
        << " positive_coupling_row_block_count="
        << summary.positive_coupling_row_block_count
        << " positive_self_row_block_count="
        << summary.positive_self_row_block_count
        << " min_positive_self_row_abs_sum="
        << (std::isfinite(summary.min_positive_self_row_abs_sum)
                ? summary.min_positive_self_row_abs_sum
                : 0.0)
        << " max_self_row_abs_sum="
        << summary.max_self_row_abs_sum
        << " pressure_only_row_block_count="
        << summary.pressure_only_row_block_count
        << " tolerance=" << tolerance
        << " clamped_local_dofs="
        << formatActivePressureSupportRankDofSamples(
               summary.clamp_candidate_row_samples, /*local=*/true)
        << " clamped_global_dofs="
        << formatActivePressureSupportRankDofSamples(
               summary.clamp_candidate_row_samples, /*local=*/false)
        << " clamped_row_details="
        << formatActivePressureSupportRankRowDetails(
               summary.clamp_candidate_row_samples);
    FE_LOG_INFO(oss.str());
}

[[nodiscard]] std::string formatGlobalIndexListSample(
    std::span<const GlobalIndex> values,
    int limit = 24)
{
    if (values.empty() || limit == 0) {
        return "none";
    }
    std::ostringstream oss;
    const int emit_count =
        limit < 0 ? static_cast<int>(values.size())
                  : std::min(limit, static_cast<int>(values.size()));
    for (int i = 0; i < emit_count; ++i) {
        if (i > 0) {
            oss << '|';
        }
        oss << values[static_cast<std::size_t>(i)];
    }
    if (limit >= 0 && static_cast<int>(values.size()) > limit) {
        oss << "|...";
    }
    return oss.str();
}

void applyActivePressureGraphCompletion(
    const systems::FESystem& sys,
    backends::GenericMatrix& matrix,
    std::span<const GlobalIndex> constrained_dofs,
    std::string_view phase,
    int iteration,
    double solve_time,
    double dt)
{
    if (!activePressureGraphCompletionEnabled()) {
        return;
    }

    const auto& pressure_field = activePressureSupportRankPressureFieldName();
    const auto& coupling_field = activePressureSupportRankCouplingFieldName();
    const double tolerance = activePressureSupportRankTolerance();
    const double coupling_threshold =
        activePressureGraphCompletionCouplingThreshold();
    const double self_threshold =
        activePressureGraphCompletionSelfThreshold();
    const int max_rows = activePressureGraphCompletionMaxRows();
    const int max_balance_pressure_edge_degree =
        activePressureGraphCompletionMaxBalancePressureEdgeDegree();
    const auto& requested_mode = activePressureGraphCompletionMode();
    const bool pressure_neighbor_mode =
        requested_mode == "pressure_neighbor" ||
        requested_mode == "pressure-neighbor" ||
        requested_mode == "neighbor";
    const bool shared_velocity_neighbor_mode =
        requested_mode == "shared_velocity_neighbor" ||
        requested_mode == "shared-velocity-neighbor" ||
        requested_mode == "velocity_neighbor" ||
        requested_mode == "velocity-neighbor";
    const bool shared_pressure_neighbor_mode =
        requested_mode == "shared_pressure_neighbor" ||
        requested_mode == "shared-pressure-neighbor" ||
        requested_mode == "pressure_pair" ||
        requested_mode == "pressure-pair" ||
        requested_mode == "shared_pressure_pair" ||
        requested_mode == "shared-pressure-pair";
    const bool shared_row_schur_support_gap_local_patch_completion_mode =
        requested_mode ==
            "shared_row_schur_support_gap_local_patch_completion" ||
        requested_mode ==
            "shared-row-schur-support-gap-local-patch-completion" ||
        requested_mode == "shared_row_schur_support_gap_local_patch" ||
        requested_mode == "shared-row-schur-support-gap-local-patch" ||
        requested_mode == "schur_support_gap_local_patch_completion" ||
        requested_mode == "schur-support-gap-local-patch-completion" ||
        requested_mode == "schur_support_gap_local_patch" ||
        requested_mode == "schur-support-gap-local-patch" ||
        requested_mode == "direct_support_gap_local_patch_completion" ||
        requested_mode == "direct-support-gap-local-patch-completion";
    const bool shared_row_schur_support_gap_patch_completion_mode =
        requested_mode == "shared_row_schur_support_gap_patch_completion" ||
        requested_mode == "shared-row-schur-support-gap-patch-completion" ||
        requested_mode == "shared_row_schur_support_gap_patch" ||
        requested_mode == "shared-row-schur-support-gap-patch" ||
        requested_mode == "schur_support_gap_patch_completion" ||
        requested_mode == "schur-support-gap-patch-completion" ||
        requested_mode == "schur_support_gap_patch" ||
        requested_mode == "schur-support-gap-patch" ||
        requested_mode == "direct_support_gap_patch_completion" ||
        requested_mode == "direct-support-gap-patch-completion";
    const bool shared_row_schur_completion_mode =
        requested_mode == "shared_row_schur_completion" ||
        requested_mode == "shared-row-schur-completion" ||
        requested_mode == "shared_row_schur_all" ||
        requested_mode == "shared-row-schur-all" ||
        requested_mode == "all_shared_row_schur_completion" ||
        requested_mode == "all-shared-row-schur-completion" ||
        requested_mode == "shared_pressure_schur_completion" ||
        requested_mode == "shared-pressure-schur-completion" ||
        requested_mode == "schur_completion" ||
        requested_mode == "schur-completion" ||
        shared_row_schur_support_gap_local_patch_completion_mode ||
        shared_row_schur_support_gap_patch_completion_mode;
    const bool pressure_neighborhood_schur_candidate_mode =
        requested_mode == "shared_row_schur_existing_edge_balance_neighborhood" ||
        requested_mode == "shared-row-schur-existing-edge-balance-neighborhood" ||
        requested_mode == "shared_row_schur_existing_edge_balance_neighbors" ||
        requested_mode == "shared-row-schur-existing-edge-balance-neighbors" ||
        requested_mode == "neighborhood_shared_row_schur_existing_edge_balance" ||
        requested_mode == "neighborhood-shared-row-schur-existing-edge-balance" ||
        requested_mode == "schur_existing_edge_balance_neighborhood" ||
        requested_mode == "schur-existing-edge-balance-neighborhood" ||
        requested_mode == "schur_existing_edge_balance_neighbors" ||
        requested_mode == "schur-existing-edge-balance-neighbors";
    const bool shared_row_schur_coupling_edge_balance_mode =
        requested_mode == "shared_row_schur_coupling_edge_balance" ||
        requested_mode == "shared-row-schur-coupling-edge-balance" ||
        requested_mode == "schur_coupling_edge_balance" ||
        requested_mode == "schur-coupling-edge-balance" ||
        requested_mode == "shared_row_schur_selective_edge_balance" ||
        requested_mode == "shared-row-schur-selective-edge-balance" ||
        requested_mode == "schur_selective_edge_balance" ||
        requested_mode == "schur-selective-edge-balance";
    const bool shared_row_schur_low_degree_edge_balance_mode =
        requested_mode == "shared_row_schur_low_degree_edge_balance" ||
        requested_mode == "shared-row-schur-low-degree-edge-balance" ||
        requested_mode == "schur_low_degree_edge_balance" ||
        requested_mode == "schur-low-degree-edge-balance" ||
        requested_mode == "shared_row_schur_boundary_edge_balance" ||
        requested_mode == "shared-row-schur-boundary-edge-balance" ||
        requested_mode == "schur_boundary_edge_balance" ||
        requested_mode == "schur-boundary-edge-balance" ||
        requested_mode == "shared_row_schur_weak_boundary_edge_balance" ||
        requested_mode == "shared-row-schur-weak-boundary-edge-balance" ||
        requested_mode == "schur_weak_boundary_edge_balance" ||
        requested_mode == "schur-weak-boundary-edge-balance";
    const bool shared_row_schur_support_gap_patch_edge_balance_mode =
        requested_mode == "shared_row_schur_support_gap_patch_edge_balance" ||
        requested_mode == "shared-row-schur-support-gap-patch-edge-balance" ||
        requested_mode == "schur_support_gap_patch_edge_balance" ||
        requested_mode == "schur-support-gap-patch-edge-balance" ||
        requested_mode == "direct_support_gap_patch_edge_balance" ||
        requested_mode == "direct-support-gap-patch-edge-balance";
    const bool shared_row_schur_support_gap_local_patch_edge_balance_mode =
        requested_mode ==
            "shared_row_schur_support_gap_local_patch_edge_balance" ||
        requested_mode ==
            "shared-row-schur-support-gap-local-patch-edge-balance" ||
        requested_mode == "schur_support_gap_local_patch_edge_balance" ||
        requested_mode == "schur-support-gap-local-patch-edge-balance" ||
        requested_mode == "direct_support_gap_local_patch_edge_balance" ||
        requested_mode == "direct-support-gap-local-patch-edge-balance";
    const bool shared_row_schur_explicit_edge_balance_mode =
        requested_mode == "shared_row_schur_explicit_edge_balance" ||
        requested_mode == "shared-row-schur-explicit-edge-balance" ||
        requested_mode == "schur_explicit_edge_balance" ||
        requested_mode == "schur-explicit-edge-balance" ||
        requested_mode == "shared_row_schur_list_edge_balance" ||
        requested_mode == "shared-row-schur-list-edge-balance" ||
        requested_mode == "schur_list_edge_balance" ||
        requested_mode == "schur-list-edge-balance";
    const bool shared_row_schur_explicit_neighborhood_edge_balance_mode =
        requested_mode == "shared_row_schur_explicit_neighborhood_edge_balance" ||
        requested_mode == "shared-row-schur-explicit-neighborhood-edge-balance" ||
        requested_mode == "shared_row_schur_explicit_neighbor_edge_balance" ||
        requested_mode == "shared-row-schur-explicit-neighbor-edge-balance" ||
        requested_mode == "schur_explicit_neighborhood_edge_balance" ||
        requested_mode == "schur-explicit-neighborhood-edge-balance" ||
        requested_mode == "schur_explicit_neighbor_edge_balance" ||
        requested_mode == "schur-explicit-neighbor-edge-balance";
    const bool shared_row_schur_explicit_balance_mode =
        shared_row_schur_explicit_edge_balance_mode ||
        shared_row_schur_explicit_neighborhood_edge_balance_mode;
    const bool shared_row_schur_existing_edge_balance_mode =
        requested_mode == "shared_row_schur_existing_edge_balance" ||
        requested_mode == "shared-row-schur-existing-edge-balance" ||
        requested_mode == "schur_existing_edge_balance" ||
        requested_mode == "schur-existing-edge-balance" ||
        requested_mode == "shared_row_schur_existing_edge_balance_all" ||
        requested_mode == "shared-row-schur-existing-edge-balance-all" ||
        requested_mode == "all_shared_row_schur_existing_edge_balance" ||
        requested_mode == "all-shared-row-schur-existing-edge-balance" ||
        requested_mode == "schur_existing_edge_balance_all" ||
        requested_mode == "schur-existing-edge-balance-all" ||
        requested_mode == "schur_edge_balance" ||
        requested_mode == "schur-edge-balance" ||
        requested_mode == "hybrid_schur_edge_balance" ||
        requested_mode == "hybrid-schur-edge-balance" ||
        shared_row_schur_coupling_edge_balance_mode ||
        shared_row_schur_low_degree_edge_balance_mode ||
        shared_row_schur_support_gap_local_patch_edge_balance_mode ||
        shared_row_schur_support_gap_patch_edge_balance_mode ||
        shared_row_schur_explicit_balance_mode ||
        pressure_neighborhood_schur_candidate_mode;
    const bool shared_row_schur_support_gap_local_patch_candidate_mode =
        shared_row_schur_support_gap_local_patch_completion_mode ||
        shared_row_schur_support_gap_local_patch_edge_balance_mode;
    const bool shared_row_schur_support_gap_patch_candidate_mode =
        shared_row_schur_support_gap_local_patch_candidate_mode ||
        shared_row_schur_support_gap_patch_completion_mode ||
        shared_row_schur_support_gap_patch_edge_balance_mode;
    const bool all_pressure_schur_candidate_mode =
        requested_mode == "shared_row_schur_all" ||
        requested_mode == "shared-row-schur-all" ||
        requested_mode == "all_shared_row_schur_completion" ||
        requested_mode == "all-shared-row-schur-completion" ||
        requested_mode == "shared_row_schur_existing_edge_balance_all" ||
        requested_mode == "shared-row-schur-existing-edge-balance-all" ||
        requested_mode == "all_shared_row_schur_existing_edge_balance" ||
        requested_mode == "all-shared-row-schur-existing-edge-balance" ||
        requested_mode == "schur_existing_edge_balance_all" ||
        requested_mode == "schur-existing-edge-balance-all";
    const bool existing_edge_balance_mode =
        requested_mode == "existing_edge_balance" ||
        requested_mode == "existing-edge-balance" ||
        requested_mode == "edge_balance" ||
        requested_mode == "edge-balance";
    const bool existing_support_balance_mode =
        requested_mode == "existing_support_balance" ||
        requested_mode == "existing-support-balance" ||
        requested_mode == "all_existing_edge_balance" ||
        requested_mode == "all-existing-edge-balance" ||
        requested_mode == "abs_edge_balance" ||
        requested_mode == "abs-edge-balance";
    const bool active_support_completion_mode =
        requested_mode == "active_support_completion" ||
        requested_mode == "active-support-completion" ||
        requested_mode == "candidate_to_active_support" ||
        requested_mode == "candidate-to-active-support" ||
        requested_mode == "candidate_active_support" ||
        requested_mode == "candidate-active-support" ||
        requested_mode == "active_support" ||
        requested_mode == "active-support";
    const bool existing_pressure_balance_mode =
        existing_edge_balance_mode || existing_support_balance_mode;
    const bool direct_weighted_completion_mode =
        existing_pressure_balance_mode ||
        active_support_completion_mode ||
        shared_row_schur_completion_mode ||
        shared_row_schur_existing_edge_balance_mode;
    const std::string_view effective_mode =
        pressure_neighbor_mode ? std::string_view("pressure_neighbor")
        : shared_velocity_neighbor_mode ? std::string_view("shared_velocity_neighbor")
        : shared_pressure_neighbor_mode ? std::string_view("shared_pressure_neighbor")
        : all_pressure_schur_candidate_mode &&
                  shared_row_schur_existing_edge_balance_mode
            ? std::string_view("shared_row_schur_existing_edge_balance_all")
        : all_pressure_schur_candidate_mode
            ? std::string_view("shared_row_schur_all")
        : pressure_neighborhood_schur_candidate_mode
            ? std::string_view(
                  "shared_row_schur_existing_edge_balance_neighborhood")
        : shared_row_schur_coupling_edge_balance_mode
            ? std::string_view("shared_row_schur_coupling_edge_balance")
        : shared_row_schur_low_degree_edge_balance_mode
            ? std::string_view("shared_row_schur_low_degree_edge_balance")
        : shared_row_schur_support_gap_local_patch_edge_balance_mode
            ? std::string_view(
                  "shared_row_schur_support_gap_local_patch_edge_balance")
        : shared_row_schur_support_gap_patch_edge_balance_mode
            ? std::string_view(
                  "shared_row_schur_support_gap_patch_edge_balance")
        : shared_row_schur_support_gap_local_patch_completion_mode
            ? std::string_view(
                  "shared_row_schur_support_gap_local_patch_completion")
        : shared_row_schur_support_gap_patch_completion_mode
            ? std::string_view(
                  "shared_row_schur_support_gap_patch_completion")
        : shared_row_schur_explicit_neighborhood_edge_balance_mode
            ? std::string_view(
                  "shared_row_schur_explicit_neighborhood_edge_balance")
        : shared_row_schur_explicit_edge_balance_mode
            ? std::string_view("shared_row_schur_explicit_edge_balance")
        : shared_row_schur_existing_edge_balance_mode
            ? std::string_view("shared_row_schur_existing_edge_balance")
        : shared_row_schur_completion_mode ? std::string_view("shared_row_schur_completion")
        : existing_edge_balance_mode ? std::string_view("existing_edge_balance")
        : existing_support_balance_mode ? std::string_view("existing_support_balance")
        : active_support_completion_mode ? std::string_view("active_support_completion")
                                        : std::string_view("cycle");

    const auto summary = scanActivePressureSupportRank(
        sys,
        matrix,
        constrained_dofs,
        pressure_field,
        coupling_field,
        tolerance,
        /*sample_limit=*/0,
        coupling_threshold,
        self_threshold);

    std::vector<GlobalIndex> candidates;
    std::string_view candidate_selector = "support_rank_zero_or_weak_rows";
    std::vector<GlobalIndex> support_gap_candidate_global_dofs;
    std::vector<GlobalIndex> support_gap_patch_candidate_global_dofs;
    double support_gap_self_threshold = 0.0;
    std::string support_gap_self_threshold_source = "none";
    int support_gap_patch_truncated = 0;
    if (shared_row_schur_support_gap_patch_candidate_mode) {
        candidate_selector =
            shared_row_schur_support_gap_local_patch_candidate_mode
                ? "pressure_self_support_gap_rows_plus_pressure_graph_local_patch"
                : "pressure_self_support_gap_rows_plus_pressure_graph_patch";
        const auto pressure_range =
            newtonMatrixSupportFieldRangeByName(sys, pressure_field);
        std::vector<std::pair<GlobalIndex, double>> active_pressure_self_rows;
        std::set<GlobalIndex> support_gap_rows;
        std::set<GlobalIndex> expanded_candidate_rows;
        if (pressure_range.has_value()) {
            for (GlobalIndex dof = pressure_range->begin;
                 dof < pressure_range->end;
                 ++dof) {
                if (dof < 0 ||
                    dof >= matrix.numRows() ||
                    dof >= matrix.numCols() ||
                    std::binary_search(
                        constrained_dofs.begin(), constrained_dofs.end(), dof)) {
                    continue;
                }
                double row_self_abs_sum = 0.0;
                for (GlobalIndex col = pressure_range->begin;
                     col < pressure_range->end;
                     ++col) {
                    const Real value = matrix.getEntry(dof, col);
                    if (std::isfinite(value)) {
                        row_self_abs_sum +=
                            std::abs(static_cast<double>(value));
                    }
                }
                if (row_self_abs_sum > tolerance &&
                    std::isfinite(row_self_abs_sum)) {
                    active_pressure_self_rows.emplace_back(dof,
                                                           row_self_abs_sum);
                }
            }

            if (!active_pressure_self_rows.empty()) {
                if (self_threshold >= 0.0) {
                    support_gap_self_threshold = self_threshold;
                    support_gap_self_threshold_source =
                        "configured_pressure_self_row_sum";
                } else {
                    std::vector<double> row_self_values;
                    row_self_values.reserve(active_pressure_self_rows.size());
                    for (const auto& row : active_pressure_self_rows) {
                        row_self_values.push_back(row.second);
                    }
                    std::sort(row_self_values.begin(), row_self_values.end());
                    const std::size_t midpoint = row_self_values.size() / 2u;
                    support_gap_self_threshold =
                        row_self_values.size() % 2u == 0u
                            ? 0.5 * (row_self_values[midpoint - 1u] +
                                     row_self_values[midpoint])
                            : row_self_values[midpoint];
                    support_gap_self_threshold_source =
                        "median_positive_pressure_self_row_abs_sum";
                }

                for (const auto& row : active_pressure_self_rows) {
                    const bool weak_support =
                        self_threshold >= 0.0
                            ? row.second <=
                                  support_gap_self_threshold + tolerance
                            : row.second + tolerance <
                                  support_gap_self_threshold;
                    if (weak_support) {
                        support_gap_rows.insert(row.first);
                    }
                }
            }

            expanded_candidate_rows = support_gap_rows;
            const auto add_same_sign_pressure_neighbors =
                [&](GlobalIndex row_i,
                    std::vector<GlobalIndex>* newly_inserted) {
                    if (row_i < pressure_range->begin ||
                        row_i >= pressure_range->end ||
                        row_i < 0 ||
                        row_i >= matrix.numRows() ||
                        row_i >= matrix.numCols()) {
                        return;
                    }
                    for (GlobalIndex row_j = pressure_range->begin;
                         row_j < pressure_range->end;
                         ++row_j) {
                        if (row_j == row_i ||
                            row_j < 0 ||
                            row_j >= matrix.numRows() ||
                            row_j >= matrix.numCols() ||
                            std::binary_search(
                                constrained_dofs.begin(),
                                constrained_dofs.end(),
                                row_j)) {
                            continue;
                        }
                        const double row_edge =
                            static_cast<double>(matrix.getEntry(row_i, row_j));
                        const double col_edge =
                            static_cast<double>(matrix.getEntry(row_j, row_i));
                        if (!std::isfinite(row_edge) ||
                            !std::isfinite(col_edge)) {
                            continue;
                        }
                        const double symmetric_offdiag =
                            0.5 * (row_edge + col_edge);
                        if (!(symmetric_offdiag < -tolerance)) {
                            continue;
                        }
                        if (expanded_candidate_rows.count(row_j) != 0u) {
                            continue;
                        }
                        if (max_rows >= 0 &&
                            static_cast<int>(expanded_candidate_rows.size()) >=
                                max_rows) {
                            support_gap_patch_truncated = 1;
                            continue;
                        }
                        expanded_candidate_rows.insert(row_j);
                        if (newly_inserted != nullptr) {
                            newly_inserted->push_back(row_j);
                        }
                    }
                };
            if (shared_row_schur_support_gap_local_patch_candidate_mode) {
                std::vector<GlobalIndex> frontier(support_gap_rows.begin(),
                                                  support_gap_rows.end());
                const int pressure_neighbor_depth =
                    activePressureGraphCompletionPressureNeighborDepth();
                for (int depth = 0;
                     depth < pressure_neighbor_depth && !frontier.empty();
                     ++depth) {
                    std::vector<GlobalIndex> next_frontier;
                    for (const auto row_i : frontier) {
                        add_same_sign_pressure_neighbors(row_i, &next_frontier);
                    }
                    std::sort(next_frontier.begin(), next_frontier.end());
                    next_frontier.erase(
                        std::unique(next_frontier.begin(), next_frontier.end()),
                        next_frontier.end());
                    frontier = std::move(next_frontier);
                }
            } else {
                std::vector<GlobalIndex> stack(support_gap_rows.begin(),
                                               support_gap_rows.end());
                std::set<GlobalIndex> visited_rows;
                while (!stack.empty()) {
                    const auto row_i = stack.back();
                    stack.pop_back();
                    if (!visited_rows.insert(row_i).second) {
                        continue;
                    }
                    std::vector<GlobalIndex> newly_inserted;
                    add_same_sign_pressure_neighbors(row_i, &newly_inserted);
                    stack.insert(stack.end(), newly_inserted.begin(),
                                 newly_inserted.end());
                }
            }
        }
        support_gap_candidate_global_dofs.assign(support_gap_rows.begin(),
                                                 support_gap_rows.end());
        support_gap_patch_candidate_global_dofs.assign(
            expanded_candidate_rows.begin(), expanded_candidate_rows.end());
        candidates = support_gap_patch_candidate_global_dofs;
    } else if (all_pressure_schur_candidate_mode) {
        candidate_selector = "all_unconstrained_pressure_rows";
        const auto pressure_range =
            newtonMatrixSupportFieldRangeByName(sys, pressure_field);
        if (pressure_range.has_value()) {
            for (GlobalIndex dof = pressure_range->begin;
                 dof < pressure_range->end;
                 ++dof) {
                if (dof < 0 ||
                    dof >= matrix.numRows() ||
                    dof >= matrix.numCols() ||
                    std::binary_search(
                        constrained_dofs.begin(), constrained_dofs.end(), dof)) {
                    continue;
                }
                candidates.push_back(dof);
            }
        }
    } else if (pressure_neighborhood_schur_candidate_mode) {
        candidate_selector =
            "support_rank_rows_plus_pressure_graph_neighbors";
        std::vector<GlobalIndex> seed_candidates =
            summary.clamp_candidate_row_global_dofs;
        std::sort(seed_candidates.begin(), seed_candidates.end());
        seed_candidates.erase(
            std::unique(seed_candidates.begin(), seed_candidates.end()),
            seed_candidates.end());
        if (max_rows >= 0 &&
            static_cast<int>(seed_candidates.size()) > max_rows) {
            seed_candidates.resize(static_cast<std::size_t>(max_rows));
        }
        const auto pressure_range =
            newtonMatrixSupportFieldRangeByName(sys, pressure_field);
        const int max_pressure_neighbors =
            activePressureGraphCompletionMaxActiveNeighbors();
        const int pressure_neighbor_depth =
            activePressureGraphCompletionPressureNeighborDepth();
        std::set<GlobalIndex> expanded_candidate_rows(
            seed_candidates.begin(), seed_candidates.end());
        std::vector<GlobalIndex> frontier = seed_candidates;
        if (pressure_range.has_value()) {
            for (int depth = 0;
                 depth < pressure_neighbor_depth && !frontier.empty();
                 ++depth) {
                std::vector<GlobalIndex> next_frontier;
                for (const auto seed : frontier) {
                    if (seed < pressure_range->begin ||
                        seed >= pressure_range->end ||
                        seed < 0 ||
                        seed >= matrix.numRows() ||
                        seed >= matrix.numCols()) {
                        continue;
                    }
                    std::vector<std::pair<GlobalIndex, double>>
                        pressure_neighbors;
                    for (GlobalIndex neighbor = pressure_range->begin;
                         neighbor < pressure_range->end;
                         ++neighbor) {
                        if (neighbor == seed ||
                            neighbor < 0 ||
                            neighbor >= matrix.numRows() ||
                            neighbor >= matrix.numCols() ||
                            std::binary_search(
                                constrained_dofs.begin(),
                                constrained_dofs.end(),
                                neighbor)) {
                            continue;
                        }
                        const double row_edge =
                            static_cast<double>(matrix.getEntry(seed, neighbor));
                        const double col_edge =
                            static_cast<double>(matrix.getEntry(neighbor, seed));
                        if (!std::isfinite(row_edge) ||
                            !std::isfinite(col_edge)) {
                            continue;
                        }
                        const double symmetric_offdiag =
                            0.5 * (row_edge + col_edge);
                        const double edge_weight =
                            symmetric_offdiag < -tolerance
                                ? -symmetric_offdiag
                                : 0.0;
                        if (edge_weight > tolerance &&
                            std::isfinite(edge_weight)) {
                            pressure_neighbors.emplace_back(
                                neighbor, edge_weight);
                        }
                    }
                    std::sort(
                        pressure_neighbors.begin(),
                        pressure_neighbors.end(),
                        [](const auto& a, const auto& b) {
                            if (a.second != b.second) {
                                return a.second > b.second;
                            }
                            return a.first < b.first;
                        });
                    if (max_pressure_neighbors >= 0 &&
                        static_cast<int>(pressure_neighbors.size()) >
                            max_pressure_neighbors) {
                        pressure_neighbors.resize(
                            static_cast<std::size_t>(max_pressure_neighbors));
                    }
                    for (const auto& neighbor : pressure_neighbors) {
                        if (expanded_candidate_rows.insert(neighbor.first).second) {
                            next_frontier.push_back(neighbor.first);
                        }
                    }
                }
                std::sort(next_frontier.begin(), next_frontier.end());
                next_frontier.erase(
                    std::unique(next_frontier.begin(), next_frontier.end()),
                    next_frontier.end());
                frontier = std::move(next_frontier);
            }
        }
        candidates.assign(expanded_candidate_rows.begin(),
                          expanded_candidate_rows.end());
    } else {
        candidates = summary.clamp_candidate_row_global_dofs;
    }
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()),
                     candidates.end());
    if (!all_pressure_schur_candidate_mode &&
        !pressure_neighborhood_schur_candidate_mode &&
        !shared_row_schur_support_gap_patch_candidate_mode &&
        max_rows >= 0 &&
        static_cast<int>(candidates.size()) > max_rows) {
        candidates.resize(static_cast<std::size_t>(max_rows));
    }
    std::vector<GlobalIndex> explicit_balance_requested_global_dofs;
    std::vector<GlobalIndex> explicit_balance_candidate_global_dofs;
    if (shared_row_schur_explicit_balance_mode) {
        explicit_balance_requested_global_dofs =
            activePressureGraphCompletionExplicitBalanceGlobalDofs();
        std::vector<GlobalIndex> explicit_balance_seed_global_dofs;
        const auto pressure_range =
            newtonMatrixSupportFieldRangeByName(sys, pressure_field);
        if (pressure_range.has_value()) {
            for (const auto dof : explicit_balance_requested_global_dofs) {
                if (dof < pressure_range->begin ||
                    dof >= pressure_range->end ||
                    dof < 0 ||
                    dof >= matrix.numRows() ||
                    dof >= matrix.numCols() ||
                    std::binary_search(
                        constrained_dofs.begin(), constrained_dofs.end(), dof)) {
                    continue;
                }
                explicit_balance_seed_global_dofs.push_back(dof);
            }
        }
        std::sort(explicit_balance_seed_global_dofs.begin(),
                  explicit_balance_seed_global_dofs.end());
        explicit_balance_seed_global_dofs.erase(
            std::unique(explicit_balance_seed_global_dofs.begin(),
                        explicit_balance_seed_global_dofs.end()),
            explicit_balance_seed_global_dofs.end());

        explicit_balance_candidate_global_dofs =
            explicit_balance_seed_global_dofs;
        if (shared_row_schur_explicit_neighborhood_edge_balance_mode &&
            pressure_range.has_value()) {
            const int max_pressure_neighbors =
                activePressureGraphCompletionMaxActiveNeighbors();
            const int pressure_neighbor_depth =
                activePressureGraphCompletionPressureNeighborDepth();
            std::set<GlobalIndex> expanded_balance_rows(
                explicit_balance_seed_global_dofs.begin(),
                explicit_balance_seed_global_dofs.end());
            std::vector<GlobalIndex> frontier =
                explicit_balance_seed_global_dofs;
            for (int depth = 0;
                 depth < pressure_neighbor_depth && !frontier.empty();
                 ++depth) {
                std::vector<GlobalIndex> next_frontier;
                for (const auto seed : frontier) {
                    if (seed < pressure_range->begin ||
                        seed >= pressure_range->end ||
                        seed < 0 ||
                        seed >= matrix.numRows() ||
                        seed >= matrix.numCols()) {
                        continue;
                    }
                    std::vector<std::pair<GlobalIndex, double>>
                        pressure_neighbors;
                    for (GlobalIndex neighbor = pressure_range->begin;
                         neighbor < pressure_range->end;
                         ++neighbor) {
                        if (neighbor == seed ||
                            neighbor < 0 ||
                            neighbor >= matrix.numRows() ||
                            neighbor >= matrix.numCols() ||
                            std::binary_search(
                                constrained_dofs.begin(),
                                constrained_dofs.end(),
                                neighbor)) {
                            continue;
                        }
                        const double row_edge =
                            static_cast<double>(matrix.getEntry(seed, neighbor));
                        const double col_edge =
                            static_cast<double>(matrix.getEntry(neighbor, seed));
                        if (!std::isfinite(row_edge) ||
                            !std::isfinite(col_edge)) {
                            continue;
                        }
                        const double symmetric_offdiag =
                            0.5 * (row_edge + col_edge);
                        const double edge_weight =
                            symmetric_offdiag < -tolerance
                                ? -symmetric_offdiag
                                : 0.0;
                        if (edge_weight > tolerance &&
                            std::isfinite(edge_weight)) {
                            pressure_neighbors.emplace_back(
                                neighbor, edge_weight);
                        }
                    }
                    std::sort(
                        pressure_neighbors.begin(),
                        pressure_neighbors.end(),
                        [](const auto& a, const auto& b) {
                            if (a.second != b.second) {
                                return a.second > b.second;
                            }
                            return a.first < b.first;
                        });
                    if (max_pressure_neighbors >= 0 &&
                        static_cast<int>(pressure_neighbors.size()) >
                            max_pressure_neighbors) {
                        pressure_neighbors.resize(
                            static_cast<std::size_t>(max_pressure_neighbors));
                    }
                    for (const auto& neighbor : pressure_neighbors) {
                        if (expanded_balance_rows.insert(neighbor.first).second) {
                            next_frontier.push_back(neighbor.first);
                        }
                    }
                }
                std::sort(next_frontier.begin(), next_frontier.end());
                next_frontier.erase(
                    std::unique(next_frontier.begin(), next_frontier.end()),
                    next_frontier.end());
                frontier = std::move(next_frontier);
            }
            explicit_balance_candidate_global_dofs.assign(
                expanded_balance_rows.begin(), expanded_balance_rows.end());
        }

        candidates.insert(candidates.end(),
                          explicit_balance_candidate_global_dofs.begin(),
                          explicit_balance_candidate_global_dofs.end());
        std::sort(explicit_balance_candidate_global_dofs.begin(),
                  explicit_balance_candidate_global_dofs.end());
        explicit_balance_candidate_global_dofs.erase(
            std::unique(explicit_balance_candidate_global_dofs.begin(),
                        explicit_balance_candidate_global_dofs.end()),
            explicit_balance_candidate_global_dofs.end());
        if (candidate_selector == std::string_view("support_rank_zero_or_weak_rows")) {
            candidate_selector =
                shared_row_schur_explicit_neighborhood_edge_balance_mode
                    ? "support_rank_zero_or_weak_rows_plus_explicit_balance_neighborhood_rows"
                    : "support_rank_zero_or_weak_rows_plus_explicit_balance_rows";
        }
    }
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()),
                     candidates.end());
    const std::set<GlobalIndex> candidate_set(candidates.begin(),
                                              candidates.end());
    const std::set<GlobalIndex> support_gap_candidate_set(
        support_gap_candidate_global_dofs.begin(),
        support_gap_candidate_global_dofs.end());
    const std::set<GlobalIndex> explicit_balance_candidate_set(
        explicit_balance_candidate_global_dofs.begin(),
        explicit_balance_candidate_global_dofs.end());
    GlobalIndex zero_coupling_candidate_count = 0;
    GlobalIndex weak_coupling_candidate_count = 0;
    GlobalIndex zero_self_candidate_count = 0;
    GlobalIndex weak_self_candidate_count = 0;
    GlobalIndex weak_coupling_and_self_candidate_count = 0;
    std::unordered_map<GlobalIndex, double> candidate_coupling_abs_sum_by_dof;
    const auto class_pressure_range =
        newtonMatrixSupportFieldRangeByName(sys, pressure_field);
    const auto class_coupling_range =
        newtonMatrixSupportFieldRangeByName(sys, coupling_field);
    if (class_pressure_range.has_value() && class_coupling_range.has_value()) {
        for (const auto candidate : candidates) {
            if (candidate < class_pressure_range->begin ||
                candidate >= class_pressure_range->end ||
                candidate < 0 ||
                candidate >= matrix.numRows()) {
                continue;
            }
            double row_coupling_abs_sum = 0.0;
            double row_self_abs_sum = 0.0;
            for (GlobalIndex col = 0; col < matrix.numCols(); ++col) {
                const Real value = matrix.getEntry(candidate, col);
                if (!std::isfinite(value)) {
                    continue;
                }
                const double abs_value = std::abs(static_cast<double>(value));
                if (dofInFieldRange(col, *class_coupling_range)) {
                    row_coupling_abs_sum += abs_value;
                }
                if (dofInFieldRange(col, *class_pressure_range)) {
                    row_self_abs_sum += abs_value;
                }
            }
            candidate_coupling_abs_sum_by_dof[candidate] =
                row_coupling_abs_sum;
            const bool zero_coupling = row_coupling_abs_sum <= tolerance;
            const bool weak_coupling =
                coupling_threshold >= 0.0 &&
                row_coupling_abs_sum <= coupling_threshold;
            const bool zero_self = row_self_abs_sum <= tolerance;
            const bool weak_self =
                self_threshold >= 0.0 &&
                row_self_abs_sum <= self_threshold;
            if (zero_coupling) {
                ++zero_coupling_candidate_count;
            }
            if (weak_coupling && !zero_coupling) {
                ++weak_coupling_candidate_count;
            }
            if (zero_self) {
                ++zero_self_candidate_count;
            }
            if (weak_self && !zero_self) {
                ++weak_self_candidate_count;
            }
            if (weak_coupling && weak_self) {
                ++weak_coupling_and_self_candidate_count;
            }
        }
    }
    const auto candidate_is_coupling_deficient =
        [&](GlobalIndex dof) -> bool {
        const auto found = candidate_coupling_abs_sum_by_dof.find(dof);
        if (found == candidate_coupling_abs_sum_by_dof.end()) {
            return false;
        }
        const double coupling_abs_sum = found->second;
        return coupling_abs_sum <= tolerance ||
               (coupling_threshold >= 0.0 &&
                coupling_abs_sum <= coupling_threshold);
    };
    GlobalIndex coupling_deficient_balance_candidate_count = 0;
    std::vector<GlobalIndex> coupling_deficient_balance_candidate_global_dofs;
    for (const auto candidate : candidates) {
        if (candidate_is_coupling_deficient(candidate)) {
            ++coupling_deficient_balance_candidate_count;
            coupling_deficient_balance_candidate_global_dofs.push_back(candidate);
        }
    }

    double min_positive_diag_abs = std::numeric_limits<double>::infinity();
    for (const auto dof : candidates) {
        if (dof < 0 || dof >= matrix.numRows() || dof >= matrix.numCols()) {
            continue;
        }
        const double diag_abs =
            std::abs(static_cast<double>(matrix.getEntry(dof, dof)));
        if (diag_abs > tolerance && std::isfinite(diag_abs)) {
            min_positive_diag_abs = std::min(min_positive_diag_abs, diag_abs);
        }
    }

    std::vector<std::pair<GlobalIndex, GlobalIndex>> edges;
    struct WeightedCompletionEdge {
        GlobalIndex row_i = INVALID_GLOBAL_INDEX;
        GlobalIndex row_j = INVALID_GLOBAL_INDEX;
        double weight = 0.0;
        double scale = 1.0;
    };
    std::vector<WeightedCompletionEdge> weighted_edges;
    std::vector<GlobalIndex> neighbor_dofs;
    std::string edge_weight_rule = "min_positive_candidate_diagonal";
    std::string neighbor_policy = "none";
    double target_self_row_abs_sum = 0.0;
    double min_completion_edge_weight = std::numeric_limits<double>::infinity();
    double max_completion_edge_weight = 0.0;
    double min_completion_edge_scale = std::numeric_limits<double>::infinity();
    double max_completion_edge_scale = 0.0;
    GlobalIndex non_laplacian_existing_edge_count = 0;
    GlobalIndex candidate_with_existing_pressure_edge_count = 0;
    GlobalIndex candidate_with_laplacian_pressure_edge_count = 0;
    GlobalIndex candidate_with_non_laplacian_only_pressure_edge_count = 0;
    GlobalIndex shared_row_schur_hub_count = 0;
    GlobalIndex shared_row_schur_candidate_edge_count = 0;
    GlobalIndex shared_row_schur_contribution_count = 0;
    GlobalIndex shared_row_schur_edge_count = 0;
    GlobalIndex balance_candidate_row_count = 0;
    GlobalIndex low_degree_balance_candidate_count = 0;
    GlobalIndex explicit_balance_candidate_count =
        static_cast<GlobalIndex>(explicit_balance_candidate_global_dofs.size());
    std::vector<GlobalIndex> low_degree_balance_candidate_global_dofs;
    std::vector<GlobalIndex> balance_candidate_global_dofs;
    int min_candidate_pressure_edge_degree = 0;
    int max_candidate_pressure_edge_degree = 0;
    GlobalIndex existing_balance_edge_count = 0;
    if (existing_pressure_balance_mode) {
        edge_weight_rule =
            existing_support_balance_mode
                ? "existing_pressure_edges_abs_scaled_to_target_self_row_abs_sum"
                : "existing_pressure_laplacian_edges_scaled_to_target_self_row_abs_sum";
        neighbor_policy =
            existing_support_balance_mode
                ? "all_existing_pressure_edges_incident_to_weak_rows"
                : "existing_pressure_laplacian_edges_incident_to_weak_rows";
        const auto pressure_range =
            newtonMatrixSupportFieldRangeByName(sys, pressure_field);
        target_self_row_abs_sum =
            summary.max_self_row_abs_sum *
            activePressureGraphCompletionWeightScale();
        const double max_edge_scale =
            activePressureGraphCompletionMaxEdgeScale();
        std::map<std::pair<GlobalIndex, GlobalIndex>, WeightedCompletionEdge>
            edge_by_pair;
        if (pressure_range.has_value() &&
            target_self_row_abs_sum > tolerance &&
            std::isfinite(target_self_row_abs_sum)) {
            std::unordered_map<GlobalIndex, double> row_self_abs_cache;
            const auto row_self_abs_sum = [&](GlobalIndex dof) -> double {
                const auto cached = row_self_abs_cache.find(dof);
                if (cached != row_self_abs_cache.end()) {
                    return cached->second;
                }
                double sum = 0.0;
                if (dof >= pressure_range->begin &&
                    dof < pressure_range->end &&
                    dof >= 0 &&
                    dof < matrix.numRows()) {
                    for (GlobalIndex col = pressure_range->begin;
                         col < pressure_range->end;
                         ++col) {
                        const Real value = matrix.getEntry(dof, col);
                        if (std::isfinite(value)) {
                            sum += std::abs(static_cast<double>(value));
                        }
                    }
                }
                row_self_abs_cache.emplace(dof, sum);
                return sum;
            };
            const auto row_scale = [&](GlobalIndex dof) -> double {
                const double self = row_self_abs_sum(dof);
                if (!(self > tolerance) || !std::isfinite(self)) {
                    return 1.0;
                }
                const double needed = target_self_row_abs_sum / self;
                if (!(needed > 1.0) || !std::isfinite(needed)) {
                    return 1.0;
                }
                return std::min(max_edge_scale, needed);
            };

            for (const auto candidate : candidates) {
                if (candidate < pressure_range->begin ||
                    candidate >= pressure_range->end ||
                    candidate < 0 ||
                    candidate >= matrix.numRows() ||
                    candidate >= matrix.numCols()) {
                    continue;
                }
                const double candidate_scale = row_scale(candidate);
                if (!(candidate_scale > 1.0)) {
                    continue;
                }
                bool candidate_has_existing_edge = false;
                bool candidate_has_laplacian_edge = false;
                for (GlobalIndex neighbor = pressure_range->begin;
                     neighbor < pressure_range->end;
                     ++neighbor) {
                    if (neighbor == candidate ||
                        neighbor < 0 ||
                        neighbor >= matrix.numRows() ||
                        neighbor >= matrix.numCols() ||
                        std::binary_search(
                            constrained_dofs.begin(), constrained_dofs.end(), neighbor)) {
                        continue;
                    }
                    const double row_edge =
                        static_cast<double>(matrix.getEntry(candidate, neighbor));
                    const double col_edge =
                        static_cast<double>(matrix.getEntry(neighbor, candidate));
                    if (!std::isfinite(row_edge) || !std::isfinite(col_edge)) {
                        continue;
                    }
                    const double symmetric_offdiag =
                        0.5 * (row_edge + col_edge);
                    const double existing_abs =
                        std::max(std::abs(row_edge), std::abs(col_edge));
                    if (existing_abs > tolerance) {
                        candidate_has_existing_edge = true;
                    }
                    const bool laplacian_like_edge =
                        symmetric_offdiag < -tolerance;
                    if (laplacian_like_edge) {
                        candidate_has_laplacian_edge = true;
                    } else if (existing_abs > tolerance) {
                        ++non_laplacian_existing_edge_count;
                    }
                    if (!laplacian_like_edge && !existing_support_balance_mode) {
                        continue;
                    }
                    if (existing_support_balance_mode && !(existing_abs > tolerance)) {
                        continue;
                    }
                    const double base_weight =
                        existing_support_balance_mode ? existing_abs : -symmetric_offdiag;
                    const double neighbor_scale =
                        candidate_set.count(neighbor) != 0u
                            ? row_scale(neighbor)
                            : 1.0;
                    const double edge_scale =
                        std::max(candidate_scale, neighbor_scale);
                    if (!(edge_scale > 1.0) || !std::isfinite(edge_scale)) {
                        continue;
                    }
                    const double delta_weight =
                        base_weight * (edge_scale - 1.0);
                    if (!(delta_weight > tolerance) ||
                        !std::isfinite(delta_weight)) {
                        continue;
                    }
                    const auto row_i = std::min(candidate, neighbor);
                    const auto row_j = std::max(candidate, neighbor);
                    auto& edge = edge_by_pair[{row_i, row_j}];
                    if (!(edge.weight > delta_weight)) {
                        edge.row_i = row_i;
                        edge.row_j = row_j;
                        edge.weight = delta_weight;
                        edge.scale = edge_scale;
                    }
                }
                if (candidate_has_existing_edge) {
                    ++candidate_with_existing_pressure_edge_count;
                }
                if (candidate_has_laplacian_edge) {
                    ++candidate_with_laplacian_pressure_edge_count;
                }
                if (candidate_has_existing_edge && !candidate_has_laplacian_edge) {
                    ++candidate_with_non_laplacian_only_pressure_edge_count;
                }
            }
        }

        weighted_edges.reserve(edge_by_pair.size());
        for (const auto& item : edge_by_pair) {
            const auto& edge = item.second;
            weighted_edges.push_back(edge);
            neighbor_dofs.push_back(edge.row_i);
            neighbor_dofs.push_back(edge.row_j);
            min_completion_edge_weight =
                std::min(min_completion_edge_weight, edge.weight);
            max_completion_edge_weight =
                std::max(max_completion_edge_weight, edge.weight);
            min_completion_edge_scale =
                std::min(min_completion_edge_scale, edge.scale);
            max_completion_edge_scale =
                std::max(max_completion_edge_scale, edge.scale);
        }
        std::sort(neighbor_dofs.begin(), neighbor_dofs.end());
        neighbor_dofs.erase(std::unique(neighbor_dofs.begin(),
                                        neighbor_dofs.end()),
                            neighbor_dofs.end());
    } else if (shared_row_schur_completion_mode ||
               shared_row_schur_existing_edge_balance_mode) {
        if (shared_row_schur_existing_edge_balance_mode) {
            edge_weight_rule =
                shared_row_schur_coupling_edge_balance_mode
                    ? "shared_row_schur_completion_then_existing_pressure_laplacian_edges_scaled_to_target_self_row_abs_sum_for_coupling_deficient_candidates"
                : shared_row_schur_low_degree_edge_balance_mode
                    ? "shared_row_schur_completion_then_existing_pressure_laplacian_edges_scaled_to_target_self_row_abs_sum_for_low_degree_pressure_candidates"
                : shared_row_schur_support_gap_local_patch_edge_balance_mode
                    ? "support_gap_local_pressure_patch_schur_completion_then_existing_pressure_laplacian_edges_scaled_to_target_self_row_abs_sum_for_support_gap_rows"
                : shared_row_schur_support_gap_patch_edge_balance_mode
                    ? "support_gap_pressure_patch_schur_completion_then_existing_pressure_laplacian_edges_scaled_to_target_self_row_abs_sum_for_support_gap_rows"
                : shared_row_schur_explicit_neighborhood_edge_balance_mode
                    ? "shared_row_schur_completion_then_existing_pressure_laplacian_edges_scaled_to_target_self_row_abs_sum_for_explicit_balance_neighborhood_rows"
                : shared_row_schur_explicit_edge_balance_mode
                    ? "shared_row_schur_completion_then_existing_pressure_laplacian_edges_scaled_to_target_self_row_abs_sum_for_explicit_balance_rows"
                : all_pressure_schur_candidate_mode
                    ? "all_pressure_shared_row_schur_completion_then_existing_pressure_laplacian_edges_scaled_to_target_self_row_abs_sum"
                : pressure_neighborhood_schur_candidate_mode
                    ? "support_rank_pressure_neighborhood_shared_row_schur_completion_then_existing_pressure_laplacian_edges_scaled_to_target_self_row_abs_sum"
                    : "shared_row_schur_completion_then_existing_pressure_laplacian_edges_scaled_to_target_self_row_abs_sum";
            neighbor_policy =
                shared_row_schur_coupling_edge_balance_mode
                    ? "weak_candidate_pressure_schur_fill_then_existing_laplacian_edges_incident_to_coupling_deficient_rows"
                : shared_row_schur_low_degree_edge_balance_mode
                    ? "weak_candidate_pressure_schur_fill_then_existing_laplacian_edges_incident_to_low_degree_pressure_rows"
                : shared_row_schur_support_gap_local_patch_edge_balance_mode
                    ? "support_gap_local_pressure_patch_schur_fill_then_existing_laplacian_edges_incident_to_support_gap_rows"
                : shared_row_schur_support_gap_patch_edge_balance_mode
                    ? "support_gap_pressure_patch_schur_fill_then_existing_laplacian_edges_incident_to_support_gap_rows"
                : shared_row_schur_explicit_neighborhood_edge_balance_mode
                    ? "weak_candidate_pressure_schur_fill_then_existing_laplacian_edges_incident_to_explicit_balance_neighborhood_rows"
                : shared_row_schur_explicit_edge_balance_mode
                    ? "weak_candidate_pressure_schur_fill_then_existing_laplacian_edges_incident_to_explicit_balance_rows"
                : all_pressure_schur_candidate_mode
                    ? "all_unconstrained_pressure_rows_schur_fill_then_existing_laplacian_edges_incident_to_all_pressure_rows"
                : pressure_neighborhood_schur_candidate_mode
                    ? "support_rank_rows_plus_strongest_pressure_neighbors_schur_fill_then_existing_laplacian_edges_incident_to_expanded_rows"
                    : "weak_candidate_pressure_schur_fill_then_existing_laplacian_edges_incident_to_weak_rows";
        } else {
            edge_weight_rule =
                shared_row_schur_support_gap_local_patch_completion_mode
                    ? "support_gap_local_pressure_patch_schur_completion_wi_wj_over_hub_support_sum"
                : shared_row_schur_support_gap_patch_completion_mode
                    ? "support_gap_pressure_patch_schur_completion_wi_wj_over_hub_support_sum"
                    : "existing_pressure_laplacian_schur_fill_wi_wj_over_hub_support_sum";
            neighbor_policy =
                shared_row_schur_support_gap_local_patch_completion_mode
                    ? "support_gap_local_pressure_patch_to_shared_row_pressure_neighbors"
                : shared_row_schur_support_gap_patch_completion_mode
                    ? "support_gap_pressure_patch_to_shared_row_pressure_neighbors"
                : all_pressure_schur_candidate_mode
                    ? "all_unconstrained_pressure_rows_to_shared_row_pressure_neighbors"
                    : "weak_candidate_pressure_neighbors_to_shared_row_pressure_neighbors";
        }
        const auto pressure_range =
            newtonMatrixSupportFieldRangeByName(sys, pressure_field);
        const int max_schur_neighbors =
            activePressureGraphCompletionMaxActiveNeighbors();
        std::map<std::pair<GlobalIndex, GlobalIndex>, WeightedCompletionEdge>
            edge_by_pair;
        std::map<GlobalIndex, std::vector<std::pair<GlobalIndex, double>>>
            pressure_neighbor_cache;
        std::set<GlobalIndex> schur_hubs;

        if (pressure_range.has_value()) {
            const auto pressure_neighbors =
                [&](GlobalIndex hub)
                    -> const std::vector<std::pair<GlobalIndex, double>>& {
                const auto cached = pressure_neighbor_cache.find(hub);
                if (cached != pressure_neighbor_cache.end()) {
                    return cached->second;
                }

                std::vector<std::pair<GlobalIndex, double>> neighbors;
                if (hub >= pressure_range->begin &&
                    hub < pressure_range->end &&
                    hub >= 0 &&
                    hub < matrix.numRows() &&
                    hub < matrix.numCols()) {
                    for (GlobalIndex neighbor = pressure_range->begin;
                         neighbor < pressure_range->end;
                         ++neighbor) {
                        if (neighbor == hub ||
                            neighbor < 0 ||
                            neighbor >= matrix.numRows() ||
                            neighbor >= matrix.numCols() ||
                            std::binary_search(
                                constrained_dofs.begin(),
                                constrained_dofs.end(),
                                neighbor)) {
                            continue;
                        }
                        const double row_edge =
                            static_cast<double>(matrix.getEntry(hub, neighbor));
                        const double col_edge =
                            static_cast<double>(matrix.getEntry(neighbor, hub));
                        if (!std::isfinite(row_edge) ||
                            !std::isfinite(col_edge)) {
                            continue;
                        }
                        const double symmetric_offdiag =
                            0.5 * (row_edge + col_edge);
                        const double edge_weight =
                            symmetric_offdiag < -tolerance
                                ? -symmetric_offdiag
                                : 0.0;
                        if (edge_weight > tolerance &&
                            std::isfinite(edge_weight)) {
                            neighbors.emplace_back(neighbor, edge_weight);
                        }
                    }
                    std::sort(
                        neighbors.begin(),
                        neighbors.end(),
                        [](const auto& a, const auto& b) {
                            if (a.second != b.second) {
                                return a.second > b.second;
                            }
                            return a.first < b.first;
                        });
                    if (max_schur_neighbors >= 0 &&
                        static_cast<int>(neighbors.size()) >
                            max_schur_neighbors) {
                        neighbors.resize(
                            static_cast<std::size_t>(max_schur_neighbors));
                    }
                }

                auto inserted = pressure_neighbor_cache.emplace(
                    hub, std::move(neighbors));
                return inserted.first->second;
            };

            for (const auto candidate : candidates) {
                if (candidate < pressure_range->begin ||
                    candidate >= pressure_range->end ||
                    candidate < 0 ||
                    candidate >= matrix.numRows() ||
                    candidate >= matrix.numCols()) {
                    continue;
                }
                for (GlobalIndex hub = pressure_range->begin;
                     hub < pressure_range->end;
                     ++hub) {
                    if (hub == candidate ||
                        hub < 0 ||
                        hub >= matrix.numRows() ||
                        hub >= matrix.numCols() ||
                        std::binary_search(
                            constrained_dofs.begin(),
                            constrained_dofs.end(),
                            hub)) {
                        continue;
                    }
                    const double candidate_row_edge =
                        static_cast<double>(matrix.getEntry(candidate, hub));
                    const double candidate_col_edge =
                        static_cast<double>(matrix.getEntry(hub, candidate));
                    if (!std::isfinite(candidate_row_edge) ||
                        !std::isfinite(candidate_col_edge)) {
                        continue;
                    }
                    const double candidate_symmetric_offdiag =
                        0.5 * (candidate_row_edge + candidate_col_edge);
                    const double candidate_edge_weight =
                        candidate_symmetric_offdiag < -tolerance
                            ? -candidate_symmetric_offdiag
                            : 0.0;
                    if (!(candidate_edge_weight > tolerance) ||
                        !std::isfinite(candidate_edge_weight)) {
                        continue;
                    }

                    ++shared_row_schur_candidate_edge_count;
                    const auto& hub_neighbors = pressure_neighbors(hub);
                    if (hub_neighbors.empty()) {
                        continue;
                    }

                    double support_weight_sum = candidate_edge_weight;
                    for (const auto& neighbor : hub_neighbors) {
                        if (neighbor.first != candidate) {
                            support_weight_sum += neighbor.second;
                        }
                    }
                    if (!(support_weight_sum > tolerance) ||
                        !std::isfinite(support_weight_sum)) {
                        continue;
                    }

                    bool added_for_hub = false;
                    for (const auto& neighbor : hub_neighbors) {
                        if (neighbor.first == candidate) {
                            continue;
                        }
                        const double weight =
                            activePressureGraphCompletionWeightScale() *
                            candidate_edge_weight *
                            neighbor.second /
                            support_weight_sum;
                        if (!(weight > tolerance) || !std::isfinite(weight)) {
                            continue;
                        }
                        const auto row_i = std::min(candidate, neighbor.first);
                        const auto row_j = std::max(candidate, neighbor.first);
                        auto& edge = edge_by_pair[{row_i, row_j}];
                        edge.row_i = row_i;
                        edge.row_j = row_j;
                        edge.weight += weight;
                        edge.scale = std::max(
                            edge.scale,
                            activePressureGraphCompletionWeightScale());
                        ++shared_row_schur_contribution_count;
                        added_for_hub = true;
                    }
                    if (added_for_hub) {
                        schur_hubs.insert(hub);
                    }
                }
            }
        }

        shared_row_schur_hub_count =
            static_cast<GlobalIndex>(schur_hubs.size());
        shared_row_schur_edge_count =
            static_cast<GlobalIndex>(edge_by_pair.size());
        if (shared_row_schur_existing_edge_balance_mode &&
            pressure_range.has_value() &&
            !edge_by_pair.empty()) {
            std::map<GlobalIndex, double> schur_diag_delta;
            std::map<std::pair<GlobalIndex, GlobalIndex>, double>
                schur_edge_weight_by_pair;
            for (const auto& item : edge_by_pair) {
                const auto& edge = item.second;
                schur_diag_delta[edge.row_i] += edge.weight;
                schur_diag_delta[edge.row_j] += edge.weight;
                schur_edge_weight_by_pair[{edge.row_i, edge.row_j}] +=
                    edge.weight;
            }

            const auto schur_delta = [&](GlobalIndex row,
                                         GlobalIndex col) -> double {
                if (row == col) {
                    const auto found = schur_diag_delta.find(row);
                    return found != schur_diag_delta.end() ? found->second : 0.0;
                }
                const auto row_i = std::min(row, col);
                const auto row_j = std::max(row, col);
                const auto found = schur_edge_weight_by_pair.find({row_i, row_j});
                return found != schur_edge_weight_by_pair.end()
                           ? -found->second
                           : 0.0;
            };

            std::unordered_map<GlobalIndex, double> row_self_abs_cache;
            const auto row_self_abs_sum = [&](GlobalIndex dof) -> double {
                const auto cached = row_self_abs_cache.find(dof);
                if (cached != row_self_abs_cache.end()) {
                    return cached->second;
                }
                double sum = 0.0;
                if (dof >= pressure_range->begin &&
                    dof < pressure_range->end &&
                    dof >= 0 &&
                    dof < matrix.numRows()) {
                    for (GlobalIndex col = pressure_range->begin;
                         col < pressure_range->end;
                         ++col) {
                        double value = static_cast<double>(matrix.getEntry(dof, col));
                        value += schur_delta(dof, col);
                        if (std::isfinite(value)) {
                            sum += std::abs(value);
                        }
                    }
                }
                row_self_abs_cache.emplace(dof, sum);
                return sum;
            };

            target_self_row_abs_sum = 0.0;
            for (GlobalIndex dof = pressure_range->begin;
                 dof < pressure_range->end;
                 ++dof) {
                if (dof < 0 ||
                    dof >= matrix.numRows() ||
                    dof >= matrix.numCols() ||
                    std::binary_search(
                        constrained_dofs.begin(), constrained_dofs.end(), dof)) {
                    continue;
                }
                target_self_row_abs_sum =
                    std::max(target_self_row_abs_sum, row_self_abs_sum(dof));
            }
            target_self_row_abs_sum *=
                activePressureGraphCompletionWeightScale();
            const double max_edge_scale =
                activePressureGraphCompletionMaxEdgeScale();
            std::unordered_map<GlobalIndex, int>
                pressure_edge_degree_cache;
            const auto pressure_edge_degree = [&](GlobalIndex dof) -> int {
                const auto cached = pressure_edge_degree_cache.find(dof);
                if (cached != pressure_edge_degree_cache.end()) {
                    return cached->second;
                }
                int degree = 0;
                if (dof >= pressure_range->begin &&
                    dof < pressure_range->end &&
                    dof >= 0 &&
                    dof < matrix.numRows() &&
                    dof < matrix.numCols()) {
                    for (GlobalIndex neighbor = pressure_range->begin;
                         neighbor < pressure_range->end;
                         ++neighbor) {
                        if (neighbor == dof ||
                            neighbor < 0 ||
                            neighbor >= matrix.numRows() ||
                            neighbor >= matrix.numCols() ||
                            std::binary_search(
                                constrained_dofs.begin(),
                                constrained_dofs.end(),
                                neighbor)) {
                            continue;
                        }
                        const double row_edge =
                            static_cast<double>(matrix.getEntry(dof, neighbor));
                        const double col_edge =
                            static_cast<double>(matrix.getEntry(neighbor, dof));
                        if (!std::isfinite(row_edge) ||
                            !std::isfinite(col_edge)) {
                            continue;
                        }
                        const double symmetric_offdiag =
                            0.5 * (row_edge + col_edge);
                        if (symmetric_offdiag < -tolerance) {
                            ++degree;
                        }
                    }
                }
                pressure_edge_degree_cache.emplace(dof, degree);
                return degree;
            };
            if (shared_row_schur_low_degree_edge_balance_mode) {
                min_candidate_pressure_edge_degree =
                    std::numeric_limits<int>::max();
                for (const auto candidate : candidates) {
                    if (candidate < pressure_range->begin ||
                        candidate >= pressure_range->end ||
                        candidate < 0 ||
                        candidate >= matrix.numRows() ||
                        candidate >= matrix.numCols()) {
                        continue;
                    }
                    const int degree = pressure_edge_degree(candidate);
                    min_candidate_pressure_edge_degree =
                        std::min(min_candidate_pressure_edge_degree, degree);
                    max_candidate_pressure_edge_degree =
                        std::max(max_candidate_pressure_edge_degree, degree);
                    if (max_balance_pressure_edge_degree < 0 ||
                        degree <= max_balance_pressure_edge_degree) {
                        ++low_degree_balance_candidate_count;
                        low_degree_balance_candidate_global_dofs.push_back(candidate);
                    }
                }
                if (min_candidate_pressure_edge_degree ==
                    std::numeric_limits<int>::max()) {
                    min_candidate_pressure_edge_degree = 0;
                }
            }
            const auto candidate_is_balance_eligible =
                [&](GlobalIndex dof) -> bool {
                if (shared_row_schur_coupling_edge_balance_mode) {
                    return candidate_is_coupling_deficient(dof);
                }
                if (shared_row_schur_low_degree_edge_balance_mode) {
                    const int degree = pressure_edge_degree(dof);
                    return max_balance_pressure_edge_degree < 0 ||
                           degree <= max_balance_pressure_edge_degree;
                }
                if (shared_row_schur_support_gap_local_patch_edge_balance_mode ||
                    shared_row_schur_support_gap_patch_edge_balance_mode) {
                    return support_gap_candidate_set.count(dof) != 0u;
                }
                if (shared_row_schur_explicit_balance_mode) {
                    return explicit_balance_candidate_set.count(dof) != 0u;
                }
                return true;
            };
            const auto row_scale = [&](GlobalIndex dof) -> double {
                const double self = row_self_abs_sum(dof);
                if (!(self > tolerance) || !std::isfinite(self)) {
                    return 1.0;
                }
                const double needed = target_self_row_abs_sum / self;
                if (!(needed > 1.0) || !std::isfinite(needed)) {
                    return 1.0;
                }
                return std::min(max_edge_scale, needed);
            };

            std::map<std::pair<GlobalIndex, GlobalIndex>, WeightedCompletionEdge>
                balance_edge_by_pair;
            if (target_self_row_abs_sum > tolerance &&
                std::isfinite(target_self_row_abs_sum)) {
                for (const auto candidate : candidates) {
                    if (candidate < pressure_range->begin ||
                        candidate >= pressure_range->end ||
                        candidate < 0 ||
                        candidate >= matrix.numRows() ||
                        candidate >= matrix.numCols()) {
                        continue;
                    }
                    if (!candidate_is_balance_eligible(candidate)) {
                        continue;
                    }
                    const double candidate_scale = row_scale(candidate);
                    if (!(candidate_scale > 1.0)) {
                        continue;
                    }
                    ++balance_candidate_row_count;
                    balance_candidate_global_dofs.push_back(candidate);
                    bool candidate_has_existing_edge = false;
                    bool candidate_has_laplacian_edge = false;
                    for (GlobalIndex neighbor = pressure_range->begin;
                         neighbor < pressure_range->end;
                         ++neighbor) {
                        if (neighbor == candidate ||
                            neighbor < 0 ||
                            neighbor >= matrix.numRows() ||
                            neighbor >= matrix.numCols() ||
                            std::binary_search(
                                constrained_dofs.begin(),
                                constrained_dofs.end(),
                                neighbor)) {
                            continue;
                        }
                        double row_edge =
                            static_cast<double>(matrix.getEntry(candidate, neighbor));
                        row_edge += schur_delta(candidate, neighbor);
                        double col_edge =
                            static_cast<double>(matrix.getEntry(neighbor, candidate));
                        col_edge += schur_delta(neighbor, candidate);
                        if (!std::isfinite(row_edge) || !std::isfinite(col_edge)) {
                            continue;
                        }
                        const double symmetric_offdiag =
                            0.5 * (row_edge + col_edge);
                        const double existing_abs =
                            std::max(std::abs(row_edge), std::abs(col_edge));
                        if (existing_abs > tolerance) {
                            candidate_has_existing_edge = true;
                        }
                        const bool laplacian_like_edge =
                            symmetric_offdiag < -tolerance;
                        if (laplacian_like_edge) {
                            candidate_has_laplacian_edge = true;
                        } else if (existing_abs > tolerance) {
                            ++non_laplacian_existing_edge_count;
                        }
                        if (!laplacian_like_edge) {
                            continue;
                        }
                        const double neighbor_scale =
                            candidate_set.count(neighbor) != 0u &&
                                    candidate_is_balance_eligible(neighbor)
                                ? row_scale(neighbor)
                                : 1.0;
                        const double edge_scale =
                            std::max(candidate_scale, neighbor_scale);
                        if (!(edge_scale > 1.0) || !std::isfinite(edge_scale)) {
                            continue;
                        }
                        const double delta_weight =
                            -symmetric_offdiag * (edge_scale - 1.0);
                        if (!(delta_weight > tolerance) ||
                            !std::isfinite(delta_weight)) {
                            continue;
                        }
                        const auto row_i = std::min(candidate, neighbor);
                        const auto row_j = std::max(candidate, neighbor);
                        auto& edge = balance_edge_by_pair[{row_i, row_j}];
                        if (!(edge.weight > delta_weight)) {
                            edge.row_i = row_i;
                            edge.row_j = row_j;
                            edge.weight = delta_weight;
                            edge.scale = edge_scale;
                        }
                    }
                    if (candidate_has_existing_edge) {
                        ++candidate_with_existing_pressure_edge_count;
                    }
                    if (candidate_has_laplacian_edge) {
                        ++candidate_with_laplacian_pressure_edge_count;
                    }
                    if (candidate_has_existing_edge && !candidate_has_laplacian_edge) {
                        ++candidate_with_non_laplacian_only_pressure_edge_count;
                    }
                }
            }

            existing_balance_edge_count =
                static_cast<GlobalIndex>(balance_edge_by_pair.size());
            for (const auto& item : balance_edge_by_pair) {
                const auto& balance_edge = item.second;
                auto& edge = edge_by_pair[item.first];
                edge.row_i = balance_edge.row_i;
                edge.row_j = balance_edge.row_j;
                edge.weight += balance_edge.weight;
                edge.scale = std::max(edge.scale, balance_edge.scale);
            }
        }
        weighted_edges.reserve(edge_by_pair.size());
        for (const auto& item : edge_by_pair) {
            const auto& edge = item.second;
            weighted_edges.push_back(edge);
            neighbor_dofs.push_back(edge.row_i);
            neighbor_dofs.push_back(edge.row_j);
            min_completion_edge_weight =
                std::min(min_completion_edge_weight, edge.weight);
            max_completion_edge_weight =
                std::max(max_completion_edge_weight, edge.weight);
            min_completion_edge_scale =
                std::min(min_completion_edge_scale, edge.scale);
            max_completion_edge_scale =
                std::max(max_completion_edge_scale, edge.scale);
        }
        std::sort(neighbor_dofs.begin(), neighbor_dofs.end());
        neighbor_dofs.erase(std::unique(neighbor_dofs.begin(),
                                        neighbor_dofs.end()),
                            neighbor_dofs.end());
    } else if (active_support_completion_mode) {
        edge_weight_rule =
            "min_positive_candidate_diagonal_distributed_to_active_pressure_support";
        neighbor_policy = "strongest_unconstrained_pressure_self_rows";
        const auto pressure_range =
            newtonMatrixSupportFieldRangeByName(sys, pressure_field);
        const auto coupling_range =
            newtonMatrixSupportFieldRangeByName(sys, coupling_field);
        const int max_active_neighbors =
            activePressureGraphCompletionMaxActiveNeighbors();
        struct ActivePressureSupportMetric {
            GlobalIndex global_dof = INVALID_GLOBAL_INDEX;
            double row_self_abs_sum = 0.0;
            double row_coupling_abs_sum = 0.0;
            double diag_abs = 0.0;
        };
        std::vector<ActivePressureSupportMetric> active_support_rows;
        std::map<std::pair<GlobalIndex, GlobalIndex>, WeightedCompletionEdge>
            edge_by_pair;
        if (pressure_range.has_value() && coupling_range.has_value() &&
            std::isfinite(min_positive_diag_abs)) {
            for (GlobalIndex global_dof = pressure_range->begin;
                 global_dof < pressure_range->end;
                 ++global_dof) {
                if (global_dof < 0 ||
                    global_dof >= matrix.numRows() ||
                    global_dof >= matrix.numCols() ||
                    std::binary_search(
                        constrained_dofs.begin(),
                        constrained_dofs.end(),
                        global_dof)) {
                    continue;
                }

                ActivePressureSupportMetric metric;
                metric.global_dof = global_dof;
                metric.diag_abs = std::abs(
                    static_cast<double>(matrix.getEntry(global_dof, global_dof)));
                for (GlobalIndex col = 0; col < matrix.numCols(); ++col) {
                    const Real value = matrix.getEntry(global_dof, col);
                    if (!std::isfinite(value)) {
                        continue;
                    }
                    const double abs_value =
                        std::abs(static_cast<double>(value));
                    if (dofInFieldRange(col, *pressure_range)) {
                        metric.row_self_abs_sum += abs_value;
                    }
                    if (dofInFieldRange(col, *coupling_range)) {
                        metric.row_coupling_abs_sum += abs_value;
                    }
                }
                if (metric.row_self_abs_sum > tolerance &&
                    std::isfinite(metric.row_self_abs_sum)) {
                    active_support_rows.push_back(metric);
                }
            }

            std::sort(
                active_support_rows.begin(),
                active_support_rows.end(),
                [](const ActivePressureSupportMetric& a,
                   const ActivePressureSupportMetric& b) {
                    if (a.row_self_abs_sum != b.row_self_abs_sum) {
                        return a.row_self_abs_sum > b.row_self_abs_sum;
                    }
                    if (a.row_coupling_abs_sum != b.row_coupling_abs_sum) {
                        return a.row_coupling_abs_sum > b.row_coupling_abs_sum;
                    }
                    if (a.diag_abs != b.diag_abs) {
                        return a.diag_abs > b.diag_abs;
                    }
                    return a.global_dof < b.global_dof;
                });

            const double total_added_per_candidate =
                min_positive_diag_abs * activePressureGraphCompletionWeightScale();
            target_self_row_abs_sum = total_added_per_candidate;
            for (const auto candidate : candidates) {
                if (candidate < pressure_range->begin ||
                    candidate >= pressure_range->end ||
                    candidate < 0 ||
                    candidate >= matrix.numRows() ||
                    candidate >= matrix.numCols()) {
                    continue;
                }

                std::vector<GlobalIndex> selected_neighbors;
                selected_neighbors.reserve(
                    max_active_neighbors < 0
                        ? active_support_rows.size()
                        : static_cast<std::size_t>(max_active_neighbors));
                for (const auto& metric : active_support_rows) {
                    if (metric.global_dof == candidate) {
                        continue;
                    }
                    selected_neighbors.push_back(metric.global_dof);
                    if (max_active_neighbors >= 0 &&
                        static_cast<int>(selected_neighbors.size()) >=
                            max_active_neighbors) {
                        break;
                    }
                }
                if (selected_neighbors.empty()) {
                    continue;
                }
                const double edge_weight =
                    total_added_per_candidate /
                    static_cast<double>(selected_neighbors.size());
                if (!(edge_weight > tolerance) || !std::isfinite(edge_weight)) {
                    continue;
                }
                for (const auto neighbor : selected_neighbors) {
                    const auto row_i = std::min(candidate, neighbor);
                    const auto row_j = std::max(candidate, neighbor);
                    auto& edge = edge_by_pair[{row_i, row_j}];
                    edge.row_i = row_i;
                    edge.row_j = row_j;
                    edge.weight += edge_weight;
                    edge.scale = std::max(
                        edge.scale,
                        activePressureGraphCompletionWeightScale());
                }
            }
        }

        weighted_edges.reserve(edge_by_pair.size());
        for (const auto& item : edge_by_pair) {
            const auto& edge = item.second;
            weighted_edges.push_back(edge);
            neighbor_dofs.push_back(edge.row_i);
            neighbor_dofs.push_back(edge.row_j);
            min_completion_edge_weight =
                std::min(min_completion_edge_weight, edge.weight);
            max_completion_edge_weight =
                std::max(max_completion_edge_weight, edge.weight);
            min_completion_edge_scale =
                std::min(min_completion_edge_scale, edge.scale);
            max_completion_edge_scale =
                std::max(max_completion_edge_scale, edge.scale);
        }
        std::sort(neighbor_dofs.begin(), neighbor_dofs.end());
        neighbor_dofs.erase(std::unique(neighbor_dofs.begin(),
                                        neighbor_dofs.end()),
                            neighbor_dofs.end());
    } else if (shared_pressure_neighbor_mode) {
        edge_weight_rule =
            "min_positive_candidate_diagonal_to_shared_pressure_neighbor_pair";
        neighbor_policy =
            "candidate_pair_with_max_shared_pressure_neighbor_support";
        const auto pressure_range =
            newtonMatrixSupportFieldRangeByName(sys, pressure_field);
        struct CandidatePressureSignature {
            GlobalIndex global_dof = INVALID_GLOBAL_INDEX;
            std::vector<std::pair<GlobalIndex, double>> pressure_neighbors;
        };
        const auto merge_pressure_signature =
            [](std::vector<std::pair<GlobalIndex, double>>& signature) {
                std::sort(signature.begin(), signature.end(),
                          [](const auto& a, const auto& b) {
                              return a.first < b.first;
                          });
                std::vector<std::pair<GlobalIndex, double>> merged;
                merged.reserve(signature.size());
                for (const auto& entry : signature) {
                    if (!merged.empty() && merged.back().first == entry.first) {
                        merged.back().second =
                            std::max(merged.back().second, entry.second);
                    } else {
                        merged.push_back(entry);
                    }
                }
                signature = std::move(merged);
            };
        const auto shared_pressure_neighbor_support =
            [](const std::vector<std::pair<GlobalIndex, double>>& a,
               const std::vector<std::pair<GlobalIndex, double>>& b,
               std::vector<GlobalIndex>& shared_neighbors) {
                double shared = 0.0;
                int count = 0;
                std::size_t i = 0;
                std::size_t j = 0;
                shared_neighbors.clear();
                while (i < a.size() && j < b.size()) {
                    if (a[i].first == b[j].first) {
                        shared += std::min(a[i].second, b[j].second);
                        shared_neighbors.push_back(a[i].first);
                        ++count;
                        ++i;
                        ++j;
                    } else if (a[i].first < b[j].first) {
                        ++i;
                    } else {
                        ++j;
                    }
                }
                return std::pair<double, int>{shared, count};
            };
        std::unordered_map<GlobalIndex, CandidatePressureSignature> signatures;
        if (pressure_range.has_value()) {
            signatures.reserve(candidates.size());
            for (const auto candidate : candidates) {
                if (candidate < pressure_range->begin ||
                    candidate >= pressure_range->end ||
                    candidate < 0 ||
                    candidate >= matrix.numRows() ||
                    candidate >= matrix.numCols()) {
                    continue;
                }
                CandidatePressureSignature signature;
                signature.global_dof = candidate;
                for (GlobalIndex neighbor = pressure_range->begin;
                     neighbor < pressure_range->end;
                     ++neighbor) {
                    if (neighbor == candidate ||
                        candidate_set.count(neighbor) != 0u ||
                        neighbor < 0 ||
                        neighbor >= matrix.numRows() ||
                        neighbor >= matrix.numCols() ||
                        std::binary_search(
                            constrained_dofs.begin(), constrained_dofs.end(), neighbor)) {
                        continue;
                    }
                    const double row_edge_abs = std::abs(
                        static_cast<double>(matrix.getEntry(candidate, neighbor)));
                    const double col_edge_abs = std::abs(
                        static_cast<double>(matrix.getEntry(neighbor, candidate)));
                    const double existing_edge_abs =
                        std::max(row_edge_abs, col_edge_abs);
                    if (existing_edge_abs > tolerance &&
                        std::isfinite(existing_edge_abs)) {
                        signature.pressure_neighbors.emplace_back(
                            neighbor, existing_edge_abs);
                    }
                }
                merge_pressure_signature(signature.pressure_neighbors);
                if (!signature.pressure_neighbors.empty()) {
                    signatures.emplace(candidate, std::move(signature));
                }
            }

            for (const auto candidate : candidates) {
                const auto candidate_it = signatures.find(candidate);
                if (candidate_it == signatures.end()) {
                    continue;
                }
                const auto& candidate_signature = candidate_it->second;
                GlobalIndex best_partner = INVALID_GLOBAL_INDEX;
                double best_shared_support = 0.0;
                int best_shared_count = 0;
                std::vector<GlobalIndex> best_shared_neighbors;
                std::vector<GlobalIndex> shared_neighbors;
                for (const auto partner : candidates) {
                    if (partner == candidate) {
                        continue;
                    }
                    const auto partner_it = signatures.find(partner);
                    if (partner_it == signatures.end()) {
                        continue;
                    }
                    const double row_edge_abs = std::abs(
                        static_cast<double>(matrix.getEntry(candidate, partner)));
                    const double col_edge_abs = std::abs(
                        static_cast<double>(matrix.getEntry(partner, candidate)));
                    const double existing_edge_abs =
                        std::max(row_edge_abs, col_edge_abs);
                    if (existing_edge_abs > tolerance &&
                        std::isfinite(existing_edge_abs)) {
                        continue;
                    }
                    const auto [shared_support, shared_count] =
                        shared_pressure_neighbor_support(
                            candidate_signature.pressure_neighbors,
                            partner_it->second.pressure_neighbors,
                            shared_neighbors);
                    if (!(shared_support > tolerance) ||
                        !std::isfinite(shared_support)) {
                        continue;
                    }
                    const bool better =
                        best_partner == INVALID_GLOBAL_INDEX ||
                        shared_support > best_shared_support ||
                        (shared_support == best_shared_support &&
                         shared_count > best_shared_count) ||
                        (shared_support == best_shared_support &&
                         shared_count == best_shared_count &&
                         partner < best_partner);
                    if (better) {
                        best_partner = partner;
                        best_shared_support = shared_support;
                        best_shared_count = shared_count;
                        best_shared_neighbors = shared_neighbors;
                    }
                }
                if (best_partner == INVALID_GLOBAL_INDEX) {
                    continue;
                }
                edges.emplace_back(std::min(candidate, best_partner),
                                   std::max(candidate, best_partner));
                neighbor_dofs.insert(neighbor_dofs.end(),
                                     best_shared_neighbors.begin(),
                                     best_shared_neighbors.end());
            }
            std::sort(edges.begin(), edges.end());
            edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
            std::sort(neighbor_dofs.begin(), neighbor_dofs.end());
            neighbor_dofs.erase(std::unique(neighbor_dofs.begin(),
                                            neighbor_dofs.end()),
                                neighbor_dofs.end());
        }
    } else if (pressure_neighbor_mode || shared_velocity_neighbor_mode) {
        edge_weight_rule =
            pressure_neighbor_mode
                ? "min_positive_candidate_diagonal_to_pressure_neighbor"
                : "min_positive_candidate_diagonal_to_shared_velocity_neighbor";
        neighbor_policy =
            pressure_neighbor_mode
                ? "max_neighbor_velocity_coupling_then_pressure_self"
                : "max_shared_velocity_signature_then_row_support";
        const auto pressure_range =
            newtonMatrixSupportFieldRangeByName(sys, pressure_field);
        const auto coupling_range =
            newtonMatrixSupportFieldRangeByName(sys, coupling_field);
        struct CompletionNeighborMetric {
            GlobalIndex global_dof = INVALID_GLOBAL_INDEX;
            double row_coupling_abs_sum = 0.0;
            double col_coupling_abs_sum = 0.0;
            double row_self_abs_sum = 0.0;
            double existing_edge_abs = 0.0;
            double shared_velocity_abs = 0.0;
            std::vector<std::pair<GlobalIndex, double>> velocity_signature;
        };
        const auto merge_velocity_signature =
            [](std::vector<std::pair<GlobalIndex, double>>& signature) {
                std::sort(signature.begin(), signature.end(),
                          [](const auto& a, const auto& b) {
                              return a.first < b.first;
                          });
                std::vector<std::pair<GlobalIndex, double>> merged;
                merged.reserve(signature.size());
                for (const auto& entry : signature) {
                    if (!merged.empty() && merged.back().first == entry.first) {
                        merged.back().second += entry.second;
                    } else {
                        merged.push_back(entry);
                    }
                }
                signature = std::move(merged);
            };
        const auto shared_velocity_signature_abs =
            [](const std::vector<std::pair<GlobalIndex, double>>& a,
               const std::vector<std::pair<GlobalIndex, double>>& b) {
                double shared = 0.0;
                std::size_t i = 0;
                std::size_t j = 0;
                while (i < a.size() && j < b.size()) {
                    if (a[i].first == b[j].first) {
                        shared += std::min(a[i].second, b[j].second);
                        ++i;
                        ++j;
                    } else if (a[i].first < b[j].first) {
                        ++i;
                    } else {
                        ++j;
                    }
                }
                return shared;
            };
        std::unordered_map<GlobalIndex, CompletionNeighborMetric> metrics;
        if (pressure_range.has_value() && coupling_range.has_value()) {
            metrics.reserve(static_cast<std::size_t>(
                std::max<GlobalIndex>(0, pressure_range->end - pressure_range->begin)));
            for (GlobalIndex local_dof = 0;
                 local_dof < pressure_range->end - pressure_range->begin;
                 ++local_dof) {
                const auto global_dof = pressure_range->begin + local_dof;
                if (global_dof < 0 ||
                    global_dof >= matrix.numRows() ||
                    global_dof >= matrix.numCols() ||
                    std::binary_search(
                        constrained_dofs.begin(), constrained_dofs.end(), global_dof)) {
                    continue;
                }
                CompletionNeighborMetric metric;
                metric.global_dof = global_dof;
                for (GlobalIndex col = 0; col < matrix.numCols(); ++col) {
                    const Real value = matrix.getEntry(global_dof, col);
                    if (!std::isfinite(value)) {
                        continue;
                    }
                    const double abs_value =
                        std::abs(static_cast<double>(value));
                    if (dofInFieldRange(col, *coupling_range)) {
                        metric.row_coupling_abs_sum += abs_value;
                        if (abs_value > tolerance) {
                            metric.velocity_signature.emplace_back(col, abs_value);
                        }
                    }
                    if (dofInFieldRange(col, *pressure_range)) {
                        metric.row_self_abs_sum += abs_value;
                    }
                }
                if (shared_velocity_neighbor_mode) {
                    for (GlobalIndex row = coupling_range->begin;
                         row < coupling_range->end;
                         ++row) {
                        const Real value = matrix.getEntry(row, global_dof);
                        if (!std::isfinite(value)) {
                            continue;
                        }
                        const double abs_value =
                            std::abs(static_cast<double>(value));
                        metric.col_coupling_abs_sum += abs_value;
                        if (abs_value > tolerance) {
                            metric.velocity_signature.emplace_back(row, abs_value);
                        }
                    }
                    merge_velocity_signature(metric.velocity_signature);
                }
                metrics.emplace(global_dof, metric);
            }

            for (const auto candidate : candidates) {
                if (candidate < pressure_range->begin ||
                    candidate >= pressure_range->end ||
                    candidate < 0 ||
                    candidate >= matrix.numRows() ||
                    candidate >= matrix.numCols()) {
                    continue;
                }

                const auto candidate_metric_it = metrics.find(candidate);
                if (candidate_metric_it == metrics.end()) {
                    continue;
                }
                const auto& candidate_metric = candidate_metric_it->second;

                std::optional<CompletionNeighborMetric> best_neighbor;
                for (GlobalIndex neighbor = pressure_range->begin;
                     neighbor < pressure_range->end;
                     ++neighbor) {
                    if (neighbor == candidate ||
                        candidate_set.count(neighbor) != 0u ||
                        std::binary_search(
                            constrained_dofs.begin(), constrained_dofs.end(), neighbor)) {
                        continue;
                    }
                    const double row_edge_abs = std::abs(
                        static_cast<double>(matrix.getEntry(candidate, neighbor)));
                    const double col_edge_abs = std::abs(
                        static_cast<double>(matrix.getEntry(neighbor, candidate)));
                    const double existing_edge_abs =
                        std::max(row_edge_abs, col_edge_abs);
                    if (pressure_neighbor_mode &&
                        (!(existing_edge_abs > tolerance) ||
                         !std::isfinite(existing_edge_abs))) {
                        continue;
                    }
                    auto metric_it = metrics.find(neighbor);
                    if (metric_it == metrics.end()) {
                        continue;
                    }
                    CompletionNeighborMetric metric = metric_it->second;
                    metric.existing_edge_abs = existing_edge_abs;
                    if (shared_velocity_neighbor_mode) {
                        metric.shared_velocity_abs =
                            shared_velocity_signature_abs(
                                candidate_metric.velocity_signature,
                                metric.velocity_signature);
                        if (!(metric.shared_velocity_abs > tolerance) ||
                            !std::isfinite(metric.shared_velocity_abs)) {
                            continue;
                        }
                    }
                    if (!best_neighbor.has_value()) {
                        best_neighbor = metric;
                        continue;
                    }
                    const auto& best = *best_neighbor;
                    const bool better_pressure_neighbor =
                        pressure_neighbor_mode &&
                        (metric.row_coupling_abs_sum > best.row_coupling_abs_sum ||
                         (metric.row_coupling_abs_sum == best.row_coupling_abs_sum &&
                          metric.row_self_abs_sum > best.row_self_abs_sum) ||
                         (metric.row_coupling_abs_sum == best.row_coupling_abs_sum &&
                          metric.row_self_abs_sum == best.row_self_abs_sum &&
                          metric.existing_edge_abs > best.existing_edge_abs) ||
                         (metric.row_coupling_abs_sum == best.row_coupling_abs_sum &&
                          metric.row_self_abs_sum == best.row_self_abs_sum &&
                          metric.existing_edge_abs == best.existing_edge_abs &&
                          metric.global_dof < best.global_dof));
                    const bool better_shared_velocity_neighbor =
                        shared_velocity_neighbor_mode &&
                        (metric.shared_velocity_abs > best.shared_velocity_abs ||
                         (metric.shared_velocity_abs == best.shared_velocity_abs &&
                          metric.row_coupling_abs_sum + metric.col_coupling_abs_sum >
                              best.row_coupling_abs_sum + best.col_coupling_abs_sum) ||
                         (metric.shared_velocity_abs == best.shared_velocity_abs &&
                          metric.row_coupling_abs_sum + metric.col_coupling_abs_sum ==
                              best.row_coupling_abs_sum + best.col_coupling_abs_sum &&
                          metric.row_self_abs_sum > best.row_self_abs_sum) ||
                         (metric.shared_velocity_abs == best.shared_velocity_abs &&
                          metric.row_coupling_abs_sum + metric.col_coupling_abs_sum ==
                              best.row_coupling_abs_sum + best.col_coupling_abs_sum &&
                          metric.row_self_abs_sum == best.row_self_abs_sum &&
                          metric.global_dof < best.global_dof));
                    if (better_pressure_neighbor ||
                        better_shared_velocity_neighbor) {
                        best_neighbor = metric;
                    }
                }
                if (!best_neighbor.has_value()) {
                    continue;
                }
                const auto neighbor = best_neighbor->global_dof;
                edges.emplace_back(std::min(candidate, neighbor),
                                   std::max(candidate, neighbor));
                neighbor_dofs.push_back(neighbor);
            }
            std::sort(edges.begin(), edges.end());
            edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
            std::sort(neighbor_dofs.begin(), neighbor_dofs.end());
            neighbor_dofs.erase(std::unique(neighbor_dofs.begin(),
                                            neighbor_dofs.end()),
                                neighbor_dofs.end());
        }
    } else {
        const std::size_t edge_span =
            candidates.size() == 2u ? 1u : candidates.size();
        for (std::size_t i = 0; i < edge_span; ++i) {
            const auto row_i = candidates[i];
            const auto row_j = candidates[(i + 1u) % candidates.size()];
            if (row_i == row_j) {
                continue;
            }
            edges.emplace_back(std::min(row_i, row_j), std::max(row_i, row_j));
        }
    }

    const bool can_apply_common_edges =
        !direct_weighted_completion_mode &&
        !edges.empty() &&
        std::isfinite(min_positive_diag_abs);
    const double edge_weight =
        direct_weighted_completion_mode
            ? max_completion_edge_weight
            : (can_apply_common_edges
                   ? min_positive_diag_abs *
                         activePressureGraphCompletionWeightScale()
                   : 0.0);
    if (can_apply_common_edges && edge_weight > 0.0) {
        weighted_edges.reserve(edges.size());
        for (const auto& edge : edges) {
            weighted_edges.push_back(
                WeightedCompletionEdge{
                    edge.first,
                    edge.second,
                    edge_weight,
                    activePressureGraphCompletionWeightScale()});
            min_completion_edge_weight =
                std::min(min_completion_edge_weight, edge_weight);
            max_completion_edge_weight =
                std::max(max_completion_edge_weight, edge_weight);
            min_completion_edge_scale =
                std::min(
                    min_completion_edge_scale,
                    activePressureGraphCompletionWeightScale());
            max_completion_edge_scale =
                std::max(
                    max_completion_edge_scale,
                    activePressureGraphCompletionWeightScale());
        }
    }
    GlobalIndex edge_count = 0;
    auto matrix_view = matrix.createAssemblyView();
    FE_CHECK_NOT_NULL(matrix_view.get(),
                      "NewtonSolver: active pressure graph completion matrix view");
    matrix_view->beginAssemblyPhase();
    if (!weighted_edges.empty()) {
        for (const auto& edge : weighted_edges) {
            const auto row_i = edge.row_i;
            const auto row_j = edge.row_j;
            if (row_i == row_j) {
                continue;
            }
            matrix_view->addMatrixEntry(
                row_i, row_i, static_cast<Real>(edge.weight), assembly::AddMode::Add);
            matrix_view->addMatrixEntry(
                row_j, row_j, static_cast<Real>(edge.weight), assembly::AddMode::Add);
            matrix_view->addMatrixEntry(
                row_i, row_j, static_cast<Real>(-edge.weight), assembly::AddMode::Add);
            matrix_view->addMatrixEntry(
                row_j, row_i, static_cast<Real>(-edge.weight), assembly::AddMode::Add);
            ++edge_count;
        }
    }
    // Locally empty edge sets still participate in distributed finalization.
    matrix_view->finalizeAssembly();

    std::ostringstream oss;
    oss << std::setprecision(17);
    oss << "NewtonSolver: active pressure graph completion"
        << " diagnostic=active_pressure_graph_completion"
        << " rank=" << mpiRank()
        << " iteration=" << iteration
        << " phase='" << phase << "'"
        << " backend=" << backends::backendKindToString(matrix.backendKind())
        << " solve_time=" << solve_time
        << " dt=" << dt
        << " pressure_field='" << summary.pressure_field << "'"
        << " coupling_field='" << summary.coupling_field << "'"
        << " mode='" << effective_mode << "'"
        << " requested_mode='" << requested_mode << "'"
        << " coupling_threshold=" << coupling_threshold
        << " self_threshold=" << self_threshold
        << " max_rows=" << max_rows
        << " max_rows_applied=" << (all_pressure_schur_candidate_mode ? 0 : 1)
        << " candidate_selector='" << candidate_selector << "'"
        << " support_rank_candidate_row_count="
        << summary.clamp_candidate_row_global_dofs.size()
        << " max_active_neighbors="
        << activePressureGraphCompletionMaxActiveNeighbors()
        << " pressure_neighbor_depth="
        << ((pressure_neighborhood_schur_candidate_mode ||
             shared_row_schur_explicit_neighborhood_edge_balance_mode ||
             shared_row_schur_support_gap_local_patch_candidate_mode)
                ? activePressureGraphCompletionPressureNeighborDepth()
                : 0)
        << " candidate_row_count=" << candidates.size()
        << " zero_coupling_candidate_count="
        << zero_coupling_candidate_count
        << " weak_coupling_candidate_count="
        << weak_coupling_candidate_count
        << " zero_self_candidate_count=" << zero_self_candidate_count
        << " weak_self_candidate_count=" << weak_self_candidate_count
        << " weak_coupling_and_self_candidate_count="
        << weak_coupling_and_self_candidate_count
        << " coupling_deficient_balance_candidate_count="
        << coupling_deficient_balance_candidate_count
        << " support_gap_candidate_count="
        << support_gap_candidate_global_dofs.size()
        << " support_gap_patch_candidate_count="
        << support_gap_patch_candidate_global_dofs.size()
        << " support_gap_self_threshold="
        << support_gap_self_threshold
        << " support_gap_self_threshold_source='"
        << support_gap_self_threshold_source << "'"
        << " support_gap_patch_truncated="
        << support_gap_patch_truncated
        << " low_degree_balance_candidate_count="
        << low_degree_balance_candidate_count
        << " explicit_balance_candidate_count="
        << explicit_balance_candidate_count
        << " balance_candidate_row_count="
        << balance_candidate_row_count
        << " coupling_deficient_balance_candidate_global_dofs="
        << formatGlobalIndexListSample(
               coupling_deficient_balance_candidate_global_dofs)
        << " support_gap_candidate_global_dofs="
        << formatGlobalIndexListSample(support_gap_candidate_global_dofs)
        << " support_gap_patch_candidate_global_dofs="
        << formatGlobalIndexListSample(support_gap_patch_candidate_global_dofs)
        << " low_degree_balance_candidate_global_dofs="
        << formatGlobalIndexListSample(low_degree_balance_candidate_global_dofs)
        << " explicit_balance_requested_global_dofs="
        << formatGlobalIndexListSample(explicit_balance_requested_global_dofs)
        << " explicit_balance_candidate_global_dofs="
        << formatGlobalIndexListSample(explicit_balance_candidate_global_dofs)
        << " balance_candidate_global_dofs="
        << formatGlobalIndexListSample(balance_candidate_global_dofs)
        << " max_balance_pressure_edge_degree="
        << max_balance_pressure_edge_degree
        << " min_candidate_pressure_edge_degree="
        << min_candidate_pressure_edge_degree
        << " max_candidate_pressure_edge_degree="
        << max_candidate_pressure_edge_degree
        << " neighbor_row_count=" << neighbor_dofs.size()
        << " edge_count=" << edge_count
        << " edge_weight=" << edge_weight
        << " edge_weight_rule='" << edge_weight_rule << "'"
        << " neighbor_policy='" << neighbor_policy << "'"
        << " weight_scale=" << activePressureGraphCompletionWeightScale()
        << " max_edge_scale_cap=" << activePressureGraphCompletionMaxEdgeScale()
        << " min_positive_candidate_diag_abs="
        << (std::isfinite(min_positive_diag_abs) ? min_positive_diag_abs : 0.0)
        << " target_self_row_abs_sum=" << target_self_row_abs_sum
        << " min_completion_edge_weight="
        << (std::isfinite(min_completion_edge_weight)
                ? min_completion_edge_weight
                : 0.0)
        << " max_completion_edge_weight=" << max_completion_edge_weight
        << " min_completion_edge_scale="
        << (std::isfinite(min_completion_edge_scale)
                ? min_completion_edge_scale
                : 0.0)
        << " max_completion_edge_scale=" << max_completion_edge_scale
        << " non_laplacian_existing_edge_count="
        << non_laplacian_existing_edge_count
        << " candidate_with_existing_pressure_edge_count="
        << candidate_with_existing_pressure_edge_count
        << " candidate_with_laplacian_pressure_edge_count="
        << candidate_with_laplacian_pressure_edge_count
        << " candidate_with_non_laplacian_only_pressure_edge_count="
        << candidate_with_non_laplacian_only_pressure_edge_count
        << " shared_row_schur_hub_count="
        << shared_row_schur_hub_count
        << " shared_row_schur_candidate_edge_count="
        << shared_row_schur_candidate_edge_count
        << " shared_row_schur_contribution_count="
        << shared_row_schur_contribution_count
        << " shared_row_schur_edge_count="
        << shared_row_schur_edge_count
        << " existing_balance_edge_count="
        << existing_balance_edge_count
        << " applied=" << (edge_count > 0 ? 1 : 0)
        << " candidate_global_dofs="
        << formatGlobalIndexListSample(candidates)
        << " neighbor_global_dofs="
        << formatGlobalIndexListSample(neighbor_dofs);
    FE_LOG_INFO(oss.str());
}

void logActivePressureUpdateSupportDiagnostic(
    const systems::FESystem& sys,
    const backends::GenericMatrix& matrix,
    backends::GenericVector& update,
    backends::GenericVector& rhs,
    std::span<const GlobalIndex> constrained_dofs,
    std::string_view phase,
    int iteration,
    double solve_time,
    double dt)
{
    if (!activePressureUpdateSupportDiagnosticRequested()) {
        return;
    }

    const auto& pressure_field = activePressureSupportRankPressureFieldName();
    const auto& coupling_field = activePressureSupportRankCouplingFieldName();
    const double tolerance = activePressureSupportRankTolerance();
    const double weak_coupling_threshold =
        activePressureUpdateSupportWeakVelocityThreshold();
    const double weak_self_threshold =
        activePressureUpdateSupportWeakSelfThreshold();
    const int sample_limit = activePressureUpdateSupportSampleLimit();
    const int action_sample_limit =
        activePressureUpdateSupportActionSampleLimit();
    const auto summary = scanActivePressureUpdateSupport(
        sys,
        matrix,
        update,
        rhs,
        constrained_dofs,
        pressure_field,
        coupling_field,
        tolerance,
        weak_coupling_threshold,
        weak_self_threshold,
        sample_limit,
        action_sample_limit);

    std::ostringstream oss;
    oss << std::setprecision(17);
    oss << "NewtonSolver: active pressure update support diagnostic"
        << " diagnostic=active_pressure_update_support"
        << " rank=" << mpiRank()
        << " iteration=" << iteration
        << " phase='" << phase << "'"
        << " backend=" << backends::backendKindToString(matrix.backendKind())
        << " solve_time=" << solve_time
        << " dt=" << dt
        << " pressure_field='" << summary.pressure_field << "'"
        << " coupling_field='" << summary.coupling_field << "'"
        << " pressure_offset=" << summary.pressure_offset
        << " pressure_dofs=" << summary.pressure_dofs
        << " coupling_offset=" << summary.coupling_offset
        << " coupling_dofs=" << summary.coupling_dofs
        << " constrained_pressure_rows=" << summary.constrained_pressure_rows
        << " unconstrained_pressure_rows=" << summary.unconstrained_pressure_rows
        << " tolerance=" << tolerance
        << " weak_coupling_threshold=" << weak_coupling_threshold
        << " weak_self_threshold=" << weak_self_threshold
        << " action_sample_limit=" << action_sample_limit
        << " same_sign_pressure_action_top_edge_count="
        << summary.same_sign_pressure_action_top_edge_count
        << " same_sign_pressure_action_component_count="
        << summary.same_sign_pressure_action_component_count
        << " same_sign_pressure_action_largest_component_size="
        << summary.same_sign_pressure_action_largest_component_size
        << " same_sign_pressure_action_covered_top_update_count="
        << summary.same_sign_pressure_action_covered_top_update_count
        << " same_sign_pressure_action_isolated_top_update_count="
        << summary.same_sign_pressure_action_isolated_top_update_count
        << " same_sign_pressure_action_largest_component_has_max_update="
        << summary.same_sign_pressure_action_largest_component_has_max_update
        << " same_sign_pressure_action_covered_global_dofs="
        << formatGlobalIndexListSample(
               summary.same_sign_pressure_action_covered_global_dofs,
               /*limit=*/24)
        << " same_sign_pressure_action_isolated_global_dofs="
        << formatGlobalIndexListSample(
               summary.same_sign_pressure_action_isolated_global_dofs,
               /*limit=*/24)
        << " same_sign_pressure_action_largest_component_global_dofs="
        << formatGlobalIndexListSample(
               summary.same_sign_pressure_action_largest_component_global_dofs,
               /*limit=*/24)
        << " zero_coupling_row_block_count="
        << summary.zero_coupling_row_block_count
        << " weak_coupling_row_block_count="
        << summary.weak_coupling_row_block_count
        << " positive_coupling_row_block_count="
        << summary.positive_coupling_row_block_count
        << " zero_self_row_block_count=" << summary.zero_self_row_block_count
        << " weak_self_row_block_count=" << summary.weak_self_row_block_count
        << " positive_self_row_block_count="
        << summary.positive_self_row_block_count
        << " max_abs_update=" << summary.max_abs_update
        << " max_update_local_dof=" << summary.max_update_local_dof
        << " max_update_global_dof=" << summary.max_update_global_dof
        << " max_update_rhs=" << summary.max_update_rhs
        << " max_update_row_action=" << summary.max_update_row_action
        << " max_update_row_coupling_action="
        << summary.max_update_row_coupling_action
        << " max_update_row_self_action="
        << summary.max_update_row_self_action
        << " max_update_row_self_constant_action="
        << summary.max_update_row_self_constant_action
        << " max_update_row_self_nonconstant_action="
        << summary.max_update_row_self_nonconstant_action
        << " max_update_row_other_action="
        << summary.max_update_row_other_action
        << " max_update_row_linear_residual="
        << summary.max_update_row_linear_residual
        << " zero_coupling_max_abs_update="
        << summary.zero_coupling_max_abs_update
        << " weak_coupling_max_abs_update="
        << summary.weak_coupling_max_abs_update
        << " positive_coupling_max_abs_update="
        << summary.positive_coupling_max_abs_update
        << " zero_self_max_abs_update=" << summary.zero_self_max_abs_update
        << " weak_self_max_abs_update=" << summary.weak_self_max_abs_update
        << " positive_self_max_abs_update="
        << summary.positive_self_max_abs_update
        << " top_update_details="
        << formatActivePressureUpdateSupportRowDetails(
               summary.top_update_samples);
    FE_LOG_INFO(oss.str());
}

std::optional<std::size_t> fieldMapIndexForFieldId(const systems::FESystem& sys,
                                                   FieldId fid)
{
    if (fid == INVALID_FIELD_ID) {
        return std::nullopt;
    }
    const auto& fmap = sys.fieldMap();
    const auto& rec = sys.fieldRecord(fid);
    const auto field_index = fmap.getFieldIndex(rec.name);
    if (field_index < 0) {
        return std::nullopt;
    }
    return static_cast<std::size_t>(field_index);
}

void logVectorTopEntries(const systems::FESystem& sys,
                         backends::GenericVector& vec,
                         std::string_view label,
                         std::size_t top_k)
{
    const auto owned_dofs =
        ownedDofsForVector(vec, sys.dofHandler().getPartition().locallyOwned());
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
        }
        traceLog("NewtonSolver: applied FSILS post-solve increment sync mode='" + mode_name + "'");
    }
#else
    (void)vec;
#endif
}

void addRankOneOperatorMatvec(std::span<const backends::RankOneUpdate> updates,
                              backends::GenericVector& x,
                              backends::GenericVector& y,
                              NewtonCommunicator communicator)
{
#if FE_HAS_MPI
    int mpi_initialized = 0;
    int mpi_finalized = 0;
    MPI_Initialized(&mpi_initialized);
    MPI_Finalized(&mpi_finalized);
    if (mpi_initialized && !mpi_finalized &&
        communicator != MPI_COMM_NULL) {
        const auto local_count =
            static_cast<unsigned long long>(updates.size());
        unsigned long long min_count = local_count;
        unsigned long long max_count = local_count;
        MPI_Allreduce(&local_count,
                      &min_count,
                      1,
                      MPI_UNSIGNED_LONG_LONG,
                      MPI_MIN,
                      communicator);
        MPI_Allreduce(&local_count,
                      &max_count,
                      1,
                      MPI_UNSIGNED_LONG_LONG,
                      MPI_MAX,
                      communicator);
        FE_THROW_IF(
            min_count != max_count,
            systems::InvalidStateException,
            "NewtonSolver: rank-one update count differs across the active "
            "system communicator");
    }
#else
    (void)communicator;
#endif
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
    if (mpi_initialized && !mpi_finalized &&
        communicator != MPI_COMM_NULL) {
        std::vector<Real> global_dots(dots.size(), Real(0.0));
        MPI_Allreduce(dots.data(),
                      global_dots.data(),
                      static_cast<int>(dots.size()),
                      MPI_DOUBLE,
                      MPI_SUM,
                      communicator);
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
                                   backends::GenericVector& y,
                                   NewtonCommunicator communicator)
{
#if FE_HAS_MPI
    int mpi_initialized = 0;
    int mpi_finalized = 0;
    MPI_Initialized(&mpi_initialized);
    MPI_Finalized(&mpi_finalized);
    if (mpi_initialized && !mpi_finalized &&
        communicator != MPI_COMM_NULL) {
        const auto local_count =
            static_cast<unsigned long long>(updates.size());
        unsigned long long min_count = local_count;
        unsigned long long max_count = local_count;
        MPI_Allreduce(&local_count,
                      &min_count,
                      1,
                      MPI_UNSIGNED_LONG_LONG,
                      MPI_MIN,
                      communicator);
        MPI_Allreduce(&local_count,
                      &max_count,
                      1,
                      MPI_UNSIGNED_LONG_LONG,
                      MPI_MAX,
                      communicator);
        FE_THROW_IF(
            min_count != max_count,
            systems::InvalidStateException,
            "NewtonSolver: reduced-field update count differs across the "
            "active system communicator");
    }
#else
    (void)communicator;
#endif
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
    if (mpi_initialized && !mpi_finalized &&
        communicator != MPI_COMM_NULL) {
        std::vector<Real> global_dots(dots.size(), Real(0.0));
        MPI_Allreduce(dots.data(),
                      global_dots.data(),
                      static_cast<int>(dots.size()),
                      MPI_DOUBLE,
                      MPI_SUM,
                      communicator);
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
    NewtonCommunicator communicator,
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
    const int global_left_has = mpiAllreduceSumIfActive(
        left_map.empty() ? 0 : 1, communicator);
    const int global_right_has = mpiAllreduceSumIfActive(
        right_map.empty() ? 0 : 1, communicator);
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

    const Real global_left_norm_sq =
        mpiAllreduceSumIfActive(left_norm_sq, communicator);
    const Real global_right_norm_sq =
        mpiAllreduceSumIfActive(right_norm_sq, communicator);
    const Real global_cross = mpiAllreduceSumIfActive(cross, communicator);

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

    const Real residual_sq =
        mpiAllreduceSumIfActive(local_residual_sq, communicator);
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

    // Bordered outlet-coupled solves cannot safely accept a native BlockSchur
    // "success" unless the wrapper also validates the original FE residual.
    // The internal FSILS residual can look converged while the true operator
    // residual is still too large for Newton to make progress. That does not
    // imply blanket 1e-8/200 retuning for every bordered case; preserve the
    // user-requested Krylov budget/tolerances here and make any stronger
    // tightening an explicit, case-specific policy.
    opts.fsils_residual_check_policy = backends::FsilsResidualCheckPolicy::Always;

    if (base.method == backends::SolverMethod::BlockSchur) {
        const Real base_rel =
            (base.rel_tol > Real(0.0) && std::isfinite(static_cast<double>(base.rel_tol)))
                ? base.rel_tol
                : static_cast<Real>(1e-6);
        const int target_inner_max_iter = std::max(base.max_iter, 1);
        opts.fsils_blockschur_gm_max_iter =
            std::max(base.fsils_blockschur_gm_max_iter.value_or(0), target_inner_max_iter);
        opts.fsils_blockschur_cg_max_iter =
            std::max(base.fsils_blockschur_cg_max_iter.value_or(0), target_inner_max_iter);

        const Real gm_target_rel = base.fsils_blockschur_gm_rel_tol.has_value()
            ? std::min(*base.fsils_blockschur_gm_rel_tol, base_rel)
            : base_rel;
        const Real cg_target_rel = base.fsils_blockschur_cg_rel_tol.has_value()
            ? std::min(*base.fsils_blockschur_cg_rel_tol, base_rel)
            : base_rel;
        opts.fsils_blockschur_gm_rel_tol = gm_target_rel;
        opts.fsils_blockschur_cg_rel_tol = cg_target_rel;
    }

    return opts;
}

[[nodiscard]] backends::SolverOptions
makeValidatedNativeRankOneSolveOptions(const backends::SolverOptions& base,
                                       const int native_direct_face_mode_count,
                                       std::optional<Real> inner_rel_override = std::nullopt,
                                       std::optional<Real> outer_rel_floor = std::nullopt)
{
    backends::SolverOptions opts = base;

    // Native face/reduced outlet updates do need tighter inner BlockSchur
    // sub-solves than the XML 1e-3 defaults, but they do not need the much
    // harsher bordered 1e-8/200 regime. A moderate tightening restores the
    // pipe-case robustness while avoiding most of the extra Schur/momentum
    // work seen with the explicit bordered settings.
    if (base.method == backends::SolverMethod::BlockSchur) {
        (void)native_direct_face_mode_count;
        const Real effective_outer_rel =
            (outer_rel_floor.has_value() &&
             base.rel_tol > Real(0.0) &&
             std::isfinite(static_cast<double>(base.rel_tol)))
                ? std::min(base.rel_tol, *outer_rel_floor)
                : base.rel_tol;
        if (effective_outer_rel > Real(0.0) &&
            std::isfinite(static_cast<double>(effective_outer_rel))) {
            opts.rel_tol = effective_outer_rel;
        }
        const Real rel_scale = static_cast<Real>(1e-2);
        const Real fallback_target = static_cast<Real>(1e-6);
        const Real target_inner_rel =
            (opts.rel_tol > Real(0.0) && std::isfinite(static_cast<double>(opts.rel_tol)))
                ? std::max(static_cast<Real>(1e-10), opts.rel_tol * rel_scale)
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
    const auto fail_closed = [&]() {
        std::fill(b.begin(), b.end(), Real(0.0));
        return false;
    };
    const auto finite = [](Real value) {
        return std::isfinite(static_cast<double>(value));
    };

    if (A.size() != n * n ||
        !(pivot_tol >= Real(0.0)) ||
        !finite(pivot_tol) ||
        !std::all_of(A.begin(), A.end(), finite) ||
        !std::all_of(b.begin(), b.end(), finite)) {
        return fail_closed();
    }

    for (std::size_t k = 0; k < n; ++k) {
        std::size_t pivot = k;
        Real pivot_abs = std::abs(A[k * n + k]);
        for (std::size_t i = k + 1; i < n; ++i) {
            const Real cand = std::abs(A[i * n + k]);
            if (!finite(cand)) {
                return fail_closed();
            }
            if (cand > pivot_abs) {
                pivot_abs = cand;
                pivot = i;
            }
        }
        if (!finite(pivot_abs) || !(pivot_abs > pivot_tol)) {
            return fail_closed();
        }
        if (pivot != k) {
            for (std::size_t j = 0; j < n; ++j) {
                std::swap(A[k * n + j], A[pivot * n + j]);
            }
            std::swap(b[k], b[pivot]);
        }

        const Real diag = A[k * n + k];
        if (!finite(diag)) {
            return fail_closed();
        }
        for (std::size_t i = k + 1; i < n; ++i) {
            const Real factor = A[i * n + k] / diag;
            if (!finite(factor)) {
                return fail_closed();
            }
            if (std::abs(factor) <= pivot_tol) {
                continue;
            }
            A[i * n + k] = 0.0;
            for (std::size_t j = k + 1; j < n; ++j) {
                A[i * n + j] -= factor * A[k * n + j];
                if (!finite(A[i * n + j])) {
                    return fail_closed();
                }
            }
            b[i] -= factor * b[k];
            if (!finite(b[i])) {
                return fail_closed();
            }
        }
    }

    for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(n) - 1; i >= 0; --i) {
        Real sum = b[static_cast<std::size_t>(i)];
        for (std::size_t j = static_cast<std::size_t>(i) + 1; j < n; ++j) {
            sum -= A[static_cast<std::size_t>(i) * n + j] * b[j];
            if (!finite(sum)) {
                return fail_closed();
            }
        }
        const Real diag = A[static_cast<std::size_t>(i) * n + static_cast<std::size_t>(i)];
        if (!finite(diag) || !(std::abs(diag) > pivot_tol)) {
            return fail_closed();
        }
        b[static_cast<std::size_t>(i)] = sum / diag;
        if (!finite(b[static_cast<std::size_t>(i)])) {
            return fail_closed();
        }
    }

    if (!std::all_of(b.begin(), b.end(), finite)) {
        return fail_closed();
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
    AlgebraicAuxiliaryReduction& out,
    NewtonCommunicator communicator)
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
    const bool preserve_grouped_direct_only = preserveGroupedAlgebraicDirectOnlyCouplings();
    const bool allow_native_rank_one_promotion =
        independent_modes && !preserve_grouped_direct_only;
    const int grouped_coupling_id =
        (independent_modes && !preserve_grouped_direct_only) ? -1 : 0;
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
            promoted_ok = tryPromoteExactReducedUpdateToNativeRankOne(
                upd, promoted, communicator);
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

    if ((!independent_modes || preserve_grouped_direct_only) &&
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
    FE_THROW_IF(options_.min_iterations < 0, InvalidArgumentException,
                "NewtonSolver: min_iterations must be >= 0");
    FE_THROW_IF(options_.abs_tolerance < 0.0 || !std::isfinite(options_.abs_tolerance),
                InvalidArgumentException,
                "NewtonSolver: abs_tolerance must be finite and >= 0");
    FE_THROW_IF(options_.rel_tolerance < 0.0 || !std::isfinite(options_.rel_tolerance),
                InvalidArgumentException,
                "NewtonSolver: rel_tolerance must be finite and >= 0");
    FE_THROW_IF(options_.step_tolerance < 0.0 || !std::isfinite(options_.step_tolerance),
                InvalidArgumentException,
                "NewtonSolver: step_tolerance must be finite and >= 0");
    if (options_.external_state_fixed_point.enabled) {
        FE_THROW_IF(
            options_.external_state_fixed_point.max_iterations <= 0,
            InvalidArgumentException,
            "NewtonSolver: external_state_fixed_point.max_iterations must be > 0");
        FE_THROW_IF(
            !(options_.abs_tolerance > 0.0),
            InvalidArgumentException,
            "NewtonSolver: external-state fixed-point convergence requires a positive absolute residual tolerance");
        FE_THROW_IF(
            !options_.synchronize_state,
            InvalidArgumentException,
            "NewtonSolver: external-state fixed point requires synchronize_state");
    }

    std::set<FieldId> field_residual_criterion_fields;
    for (const auto& criterion : options_.field_residual_criteria) {
        FE_THROW_IF(criterion.field == INVALID_FIELD_ID,
                    InvalidArgumentException,
                    "NewtonSolver: field residual criterion requires a valid field");
        FE_THROW_IF(criterion.abs_tolerance < 0.0 ||
                        !std::isfinite(criterion.abs_tolerance),
                    InvalidArgumentException,
                    "NewtonSolver: field residual criterion abs_tolerance must be finite and >= 0");
        FE_THROW_IF(criterion.rel_tolerance < 0.0 ||
                        !std::isfinite(criterion.rel_tolerance),
                    InvalidArgumentException,
                    "NewtonSolver: field residual criterion rel_tolerance must be finite and >= 0");
        FE_THROW_IF(!(criterion.abs_tolerance > 0.0) &&
                        !(criterion.rel_tolerance > 0.0),
                    InvalidArgumentException,
                    "NewtonSolver: field residual criterion must enable an absolute or relative tolerance");
        FE_THROW_IF(options_.external_state_fixed_point.enabled &&
                        !(criterion.abs_tolerance > 0.0),
                    InvalidArgumentException,
                    "NewtonSolver: external-state fixed-point field criteria require positive absolute tolerances");
        const bool inserted =
            field_residual_criterion_fields.insert(criterion.field).second;
        FE_THROW_IF(!inserted,
                    InvalidArgumentException,
                    "NewtonSolver: duplicate field residual criterion");
    }

    FE_THROW_IF(options_.jacobian_rebuild_period <= 0, InvalidArgumentException,
                "NewtonSolver: jacobian_rebuild_period must be >= 1");
    FE_THROW_IF(options_.jacobian_check_relative_tolerance < 0.0 ||
                    !std::isfinite(options_.jacobian_check_relative_tolerance),
                InvalidArgumentException,
                "NewtonSolver: jacobian_check_relative_tolerance must be finite and >= 0");
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

    const auto* dist =
        system.distributedSparsityIfAvailable(options_.jacobian_op);
    workspace.jacobian.reset();
    workspace.diagnostic_jacobian_scratch.reset();
    workspace.pressure_representability_pair_matrix.reset();
    workspace.pressure_representability_load.reset();
    workspace.pressure_representability_solution.reset();
    workspace.pressure_representability_left_basis.reset();
    workspace.pressure_representability_right_basis.reset();
    workspace.pressure_representability_direction.reset();
    workspace.pressure_representability_work.reset();
    workspace.pressure_representability_residual.reset();
    workspace.pressure_representability_normal_residual.reset();

    bool allocate_pressure_representability =
        freeSurfaceConservativeBalanceDiagnosticEnabled() &&
        system.hasOperator(std::string(
            kFreeSurfacePressureRepresentabilityPairOperator));
#if FE_HAS_MPI
    {
        int initialized = 0;
        int finalized = 0;
        MPI_Initialized(&initialized);
        MPI_Finalized(&finalized);
        if (initialized != 0 && finalized == 0) {
            const int local = allocate_pressure_representability ? 1 : 0;
            int global = local;
            MPI_Allreduce(&local,
                          &global,
                          1,
                          MPI_INT,
                          MPI_MIN,
                          system.activeMpiCommunicator());
            allocate_pressure_representability = global != 0;
        }
    }
#endif
    if (allocate_pressure_representability) {
        const std::string pair_op(
            kFreeSurfacePressureRepresentabilityPairOperator);
        const auto* pair_dist =
            system.distributedSparsityIfAvailable(pair_op);
        if (pair_dist != nullptr &&
            factory.backendKind() != backends::BackendKind::Eigen) {
            workspace.pressure_representability_pair_matrix =
                factory.createMatrix(*pair_dist);
        } else {
            workspace.pressure_representability_pair_matrix =
                factory.createMatrix(system.sparsity(pair_op));
        }

        recreatePressureRepresentabilityVectors(
            factory, n_dofs, workspace);
    }

    // Matrix creation selects the vector layout cached by the distributed
    // backend factories.  Build the optional diagnostic matrices first and
    // their vectors immediately after the pressure pair.  Build the production
    // Jacobian last, before every production Newton vector, so PETSc/Trilinos
    // ghost maps and the FSILS shared layout used by TimeHistory::repack remain
    // those of the production operator rather than the narrower diagnostic.
    if (pressureRowContributionMatrixDiagnosticEnabled()) {
        if (dist != nullptr &&
            factory.backendKind() != backends::BackendKind::Eigen) {
            workspace.diagnostic_jacobian_scratch =
                factory.createMatrix(*dist);
        } else {
            workspace.diagnostic_jacobian_scratch =
                factory.createMatrix(system.sparsity(options_.jacobian_op));
        }
    }
    if (dist != nullptr &&
        factory.backendKind() != backends::BackendKind::Eigen) {
        workspace.jacobian = factory.createMatrix(*dist);
    } else {
        workspace.jacobian =
            factory.createMatrix(system.sparsity(options_.jacobian_op));
    }
    workspace.residual = factory.createVector(n_dofs);
    workspace.delta = factory.createVector(n_dofs);
    workspace.u_backup = factory.createVector(n_dofs);
    workspace.residual_scratch = factory.createVector(n_dofs);
    workspace.residual_base = factory.createVector(n_dofs);
    workspace.residual_minus = factory.createVector(n_dofs);
    workspace.ptc_mass_lumped.reset();
    workspace.line_search_history_backup.clear();
    workspace.dt_field_dofs.clear();
    workspace.factory = &factory;
    workspace.sparsity_revision = system.sparsityPatternRevision();

    FE_CHECK_NOT_NULL(workspace.jacobian.get(), "NewtonSolver workspace.jacobian");
    FE_CHECK_NOT_NULL(workspace.residual.get(), "NewtonSolver workspace.residual");
    FE_CHECK_NOT_NULL(workspace.delta.get(), "NewtonSolver workspace.delta");
    FE_CHECK_NOT_NULL(workspace.u_backup.get(), "NewtonSolver workspace.u_backup");
    FE_CHECK_NOT_NULL(workspace.residual_scratch.get(), "NewtonSolver workspace.residual_scratch");
    FE_CHECK_NOT_NULL(workspace.residual_base.get(), "NewtonSolver workspace.residual_base");
    FE_CHECK_NOT_NULL(workspace.residual_minus.get(), "NewtonSolver workspace.residual_minus");
    if (pressureRowContributionMatrixDiagnosticEnabled()) {
        FE_CHECK_NOT_NULL(workspace.diagnostic_jacobian_scratch.get(),
                          "NewtonSolver workspace.diagnostic_jacobian_scratch");
    }
    if (allocate_pressure_representability) {
        FE_CHECK_NOT_NULL(
            workspace.pressure_representability_pair_matrix.get(),
            "NewtonSolver workspace.pressure_representability_pair_matrix");
    }

    if (options_.pseudo_transient.enabled) {
        workspace.ptc_mass_lumped = factory.createVector(n_dofs);
        FE_CHECK_NOT_NULL(workspace.ptc_mass_lumped.get(), "NewtonSolver workspace.ptc_mass_lumped");
    }

    if (options_.scale_dt_increments) {
        const auto dt_fields = system.timeDerivativeFields();
        if (!dt_fields.empty()) {
            const auto& fmap = system.fieldMap();
            for (const auto fid : dt_fields) {
                const auto idx = fieldMapIndexForFieldId(system, fid);
                if (!idx || *idx >= fmap.numFields()) {
                    continue;
                }
                const auto range = fmap.getFieldDofRange(*idx);
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
            << " diagnostic_jacobian_scratch="
            << (workspace.diagnostic_jacobian_scratch != nullptr ? "yes" : "no")
            << " pressure_representability_workspace="
            << (workspace.pressure_representability_pair_matrix != nullptr
                    ? "yes"
                    : "no")
            << " dt_field_dofs=" << workspace.dt_field_dofs.size();
        traceLog(oss.str());
    }
}

void NewtonSolver::maybeReallocateJacobianForSparsity(const systems::FESystem& system,
                                                      NewtonWorkspace& workspace) const
{
    bool factory_available = workspace.factory != nullptr;
    const auto revision = system.sparsityPatternRevision();
    bool revision_changed = revision != workspace.sparsity_revision;
#if FE_HAS_MPI
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    MPI_Finalized(&finalized);
    if (initialized != 0 && finalized == 0) {
        const auto communicator = system.activeMpiCommunicator();
        const int local_factory_available = factory_available ? 1 : 0;
        int min_factory_available = local_factory_available;
        int max_factory_available = local_factory_available;
        MPI_Allreduce(&local_factory_available,
                      &min_factory_available,
                      1,
                      MPI_INT,
                      MPI_MIN,
                      communicator);
        MPI_Allreduce(&local_factory_available,
                      &max_factory_available,
                      1,
                      MPI_INT,
                      MPI_MAX,
                      communicator);
        FE_THROW_IF(
            min_factory_available != max_factory_available,
            systems::InvalidStateException,
            "NewtonSolver: workspace factory availability differs across the "
            "active system communicator");
        factory_available = max_factory_available != 0;

        const int local_changed = revision_changed ? 1 : 0;
        int global_changed = local_changed;
        MPI_Allreduce(&local_changed,
                      &global_changed,
                      1,
                      MPI_INT,
                      MPI_MAX,
                      communicator);
        revision_changed = global_changed != 0;
    }
#endif
    if (!factory_available) {
        return;
    }
    if (!revision_changed) {
        return;
    }

    const auto& factory = *workspace.factory;
    const auto* dist = system.distributedSparsityIfAvailable(options_.jacobian_op);
    const bool use_distributed =
        dist != nullptr && factory.backendKind() != backends::BackendKind::Eigen;

    auto create_matrix = [&]() -> std::unique_ptr<backends::GenericMatrix> {
        if (use_distributed) {
            return factory.createMatrix(*dist);
        }
        return factory.createMatrix(system.sparsity(options_.jacobian_op));
    };

    // Refresh the diagnostic pair before the production Jacobian.  A matrix
    // replacement updates the backend factory's cached vector layout; doing
    // the production refresh last restores that cache for any vectors created
    // later in this nonlinear solve.  FSILS requires in-place refresh here so
    // the pair matrix remains compatible with its already allocated LSQR
    // vectors.
    if (workspace.pressure_representability_pair_matrix) {
        const std::string pair_op(
            kFreeSurfacePressureRepresentabilityPairOperator);
        FE_THROW_IF(
            !system.hasOperator(pair_op),
            systems::InvalidStateException,
            "NewtonSolver: pressure-representability pair operator disappeared during a sparsity refresh");
        const auto* pair_dist =
            system.distributedSparsityIfAvailable(pair_op);
        const bool use_pair_distributed =
            pair_dist != nullptr &&
            factory.backendKind() != backends::BackendKind::Eigen;
        bool pair_in_place =
            use_pair_distributed
                ? workspace.pressure_representability_pair_matrix
                      ->reinitFromPattern(*pair_dist)
                : workspace.pressure_representability_pair_matrix
                      ->reinitFromPattern(system.sparsity(pair_op));
#if FE_HAS_MPI
        if (use_pair_distributed) {
            const int local_pair_in_place = pair_in_place ? 1 : 0;
            int global_pair_in_place = local_pair_in_place;
            MPI_Allreduce(&local_pair_in_place,
                          &global_pair_in_place,
                          1,
                          MPI_INT,
                          MPI_MIN,
                          system.activeMpiCommunicator());
            pair_in_place = global_pair_in_place != 0;
        }
#endif
        FE_THROW_IF(
            !pair_in_place &&
                factory.backendKind() == backends::BackendKind::FSILS,
            systems::InvalidStateException,
            "NewtonSolver: FSILS pressure-representability matrix sparsity refresh changed its vector layout; rebuild the nonlinear workspace at a time-step boundary");
        if (!pair_in_place) {
            workspace.pressure_representability_pair_matrix =
                use_pair_distributed
                    ? factory.createMatrix(*pair_dist)
                    : factory.createMatrix(system.sparsity(pair_op));
            FE_CHECK_NOT_NULL(
                workspace.pressure_representability_pair_matrix.get(),
                "NewtonSolver: pressure-representability matrix sparsity refresh");
            // The replacement may carry a different PETSc/Trilinos owned and
            // ghost map even when its global dimensions are unchanged.  The
            // factory cache currently describes the refreshed pair pattern;
            // recreate every LSQR vector now, before the production-J refresh
            // deliberately restores the factory's production layout.
            recreatePressureRepresentabilityVectors(
                factory,
                system.dofHandler().getNumDofs(),
                workspace);
        }
        FE_CHECK_NOT_NULL(
            workspace.pressure_representability_pair_matrix.get(),
            "NewtonSolver: pressure-representability matrix sparsity refresh");
    }

    // Prefer in-place reinitialization: it preserves object identity, so the
    // matrix references that solveStep binds for the whole step stay valid
    // even when the sparsity changes between Newton iterations (monolithic
    // level-set + NS systems reclassify cut-cell constraints per iterate).
    bool in_place = false;
    if (workspace.jacobian) {
        in_place = use_distributed
                       ? workspace.jacobian->reinitFromPattern(*dist)
                       : workspace.jacobian->reinitFromPattern(
                             system.sparsity(options_.jacobian_op));
    }
#if FE_HAS_MPI
    if (use_distributed) {
        const int local_in_place = in_place ? 1 : 0;
        int global_in_place = local_in_place;
        MPI_Allreduce(&local_in_place,
                      &global_in_place,
                      1,
                      MPI_INT,
                      MPI_MIN,
                      system.activeMpiCommunicator());
        in_place = global_in_place != 0;
    }
#endif
    FE_THROW_IF(
        !in_place && factory.backendKind() == backends::BackendKind::FSILS,
        systems::InvalidStateException,
        "NewtonSolver: FSILS cannot replace a Jacobian while workspace and "
        "time-history vectors retain its prior layout; in-place sparsity "
        "refresh requires unchanged dimensions, communicator, DOF "
        "permutation, and owned/ghost node layout. Rebuild the nonlinear "
        "workspace at a time-step boundary before changing that layout");
    if (!in_place) {
        workspace.jacobian = create_matrix();
    }
    FE_CHECK_NOT_NULL(workspace.jacobian.get(),
                      "NewtonSolver: jacobian reallocation for sparsity refresh");
    if (workspace.diagnostic_jacobian_scratch) {
        bool scratch_in_place =
            use_distributed
                ? workspace.diagnostic_jacobian_scratch->reinitFromPattern(*dist)
                : workspace.diagnostic_jacobian_scratch->reinitFromPattern(
                      system.sparsity(options_.jacobian_op));
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
            workspace.diagnostic_jacobian_scratch = create_matrix();
        }
    }
    workspace.sparsity_revision = revision;
    FE_LOG_INFO("NewtonSolver: reallocated Jacobian for re-augmented sparsity"
                " diagnostic=jacobian_sparsity_reallocation"
                " mode=" + std::string(in_place ? "in_place" : "replace") +
                " sparsity_pattern_revision=" +
                std::to_string(revision));
}

NewtonReport NewtonSolver::solveStep(
    systems::TransientSystem& transient,
    backends::LinearSolver& linear,
    double solve_time,
    TimeHistory& history,
    NewtonWorkspace& workspace,
    const backends::GenericVector* residual_addition) const
{
    if (!options_.external_state_fixed_point.enabled) {
        return solveStepFrozenExternalState(
            transient,
            linear,
            solve_time,
            history,
            workspace,
            residual_addition);
    }

    FE_THROW_IF(!workspace.isAllocated(), InvalidArgumentException,
                "NewtonSolver::solveStep: workspace not allocated");
    FE_CHECK_NOT_NULL(
        workspace.factory,
        "NewtonSolver: external-state fixed-point workspace factory");
    FE_THROW_IF(
        transient.system().geometricNonlinearityEnabled(),
        systems::InvalidStateException,
        "NewtonSolver: external-state fixed point currently requires fixed mesh coordinates so a failed outer solve can restore the complete entry state");
    FE_THROW_IF(
        transient.system().meshCoordinateTransactionActive(),
        systems::InvalidStateException,
        "NewtonSolver: external-state fixed point cannot start with an active mesh-coordinate transaction");

    auto& system = transient.system();
    FE_THROW_IF(!(history.dt() > 0.0), InvalidArgumentException,
                "NewtonSolver: dt must be > 0");
    FE_THROW_IF(!std::isfinite(solve_time), InvalidArgumentException,
                "NewtonSolver: solve_time must be finite");

    // Establish the same stage-time constraint values that every inner Newton
    // solve will use before defining the transactional entry state.  Without
    // this ordering a generalized-alpha stage could snapshot endpoint/previous
    // values and then fail rollback solely because the restored fingerprint
    // contains the correct stage-time inhomogeneities.
    system.updateConstraints(solve_time, history.dt());
    system.constraints().updateGhostsAndDistribute(history.u());
    syncOwnedRowHaloIfNeeded(history.u());

    const auto vector_size = history.u().size();
    auto raw_entry_u = workspace.factory->createVector(vector_size);
    FE_CHECK_NOT_NULL(raw_entry_u.get(),
                      "NewtonSolver: external-state raw entry solution backup");
    raw_entry_u->copyFrom(history.u());
    auto entry_u = workspace.factory->createVector(vector_size);
    FE_CHECK_NOT_NULL(entry_u.get(),
                      "NewtonSolver: external-state canonical entry solution backup");

    std::vector<std::unique_ptr<backends::GenericVector>> entry_history(
        static_cast<std::size_t>(history.historyDepth()));
    for (int k = 1; k <= history.historyDepth(); ++k) {
        auto& backup = entry_history[static_cast<std::size_t>(k - 1)];
        backup = workspace.factory->createVector(history.uPrevK(k).size());
        FE_CHECK_NOT_NULL(
            backup.get(),
            "NewtonSolver: external-state history backup");
        backup->copyFrom(history.uPrevK(k));
    }
    auto entry_rate_state = history.snapshotRateState(*workspace.factory);
    const auto raw_entry_auxiliary_state = system.checkpointAuxiliaryState();
    const auto raw_entry_bordered_state = system.borderedCoupling();
    auto entry_auxiliary_state = raw_entry_auxiliary_state;
    auto entry_bordered_state = raw_entry_bordered_state;
    ConstraintSemanticFingerprint entry_constraint_semantics{};
    bool canonical_entry_defined = false;

    auto anyRank = [&](bool local_value) {
#if FE_HAS_MPI
        int initialized = 0;
        int finalized = 0;
        MPI_Initialized(&initialized);
        MPI_Finalized(&finalized);
        if (initialized != 0 && finalized == 0) {
            const int local = local_value ? 1 : 0;
            int global = local;
            MPI_Allreduce(&local,
                          &global,
                          1,
                          MPI_INT,
                          MPI_MAX,
                          system.activeMpiCommunicator());
            return global != 0;
        }
#endif
        return local_value;
    };

    auto restoreEntryHistoryAndRates = [&]() {
        FE_THROW_IF(
            entry_history.size() !=
                static_cast<std::size_t>(history.historyDepth()),
            systems::InvalidStateException,
            "NewtonSolver: external-state fixed point changed history depth");
        for (int k = 1; k <= history.historyDepth(); ++k) {
            auto& backup = entry_history[static_cast<std::size_t>(k - 1)];
            FE_CHECK_NOT_NULL(
                backup.get(),
                "NewtonSolver: external-state history restore backup");
            history.uPrevK(k).copyFrom(*backup);
        }
        history.restoreRateState(entry_rate_state);
        // restoreRateState swaps buffers.  Refill the reusable snapshot now so
        // every later outer iteration starts from the same immutable stage
        // history/rates rather than from the preceding MPC projection.
        history.snapshotRateState(entry_rate_state, *workspace.factory);
    };

    auto projectWithCurrentConstraints = [&]() {
        auto& constraints = system.constraints();
        history.updateGhosts();
        if (!anyRank(!constraints.empty())) {
            return;
        }
        constraints.updateGhostsAndDistribute(history.u());
        syncOwnedRowHaloIfNeeded(history.u());
        if (distributeConstraintsIntoHistoryRequested()) {
            for (int k = 1; k <= history.historyDepth(); ++k) {
                constraints.distribute(history.uPrevK(k));
                syncOwnedRowHaloIfNeeded(history.uPrevK(k));
            }
        } else {
            for (int k = 1; k <= history.historyDepth(); ++k) {
                syncOwnedRowHaloIfNeeded(history.uPrevK(k));
            }
        }
        if (!mpcStateDistributeDisabled() &&
            anyRank(constraints.hasMasterBearingLines())) {
            for (int k = 1; k <= history.historyDepth(); ++k) {
                if (transient.integrator().historySlotStoresRate(k)) {
                    constraints.distributeMasterBearingHomogeneous(
                        history.uPrevK(k));
                } else {
                    constraints.distributeMasterBearing(history.uPrevK(k));
                }
                syncOwnedRowHaloIfNeeded(history.uPrevK(k));
            }
            if (history.hasUDotState()) {
                constraints.distributeMasterBearingHomogeneous(
                    history.uDot());
                syncOwnedRowHaloIfNeeded(history.uDot());
            }
            if (history.hasUDDotState()) {
                constraints.distributeMasterBearingHomogeneous(
                    history.uDDot());
                syncOwnedRowHaloIfNeeded(history.uDDot());
            }
        }
    };

    auto synchronizeOuterState = [&, this](
                                     NewtonOptions::StateSynchronizationPoint
                                         point) {
        constexpr int max_constraint_projection_passes = 3;
        auto semantic_before =
            constraintSemanticFingerprint(system.constraints());
        for (int pass = 0;; ++pass) {
            history.updateGhosts();
            auto state = makeStateView(history, solve_time);
            std::optional<assembly::TimeIntegrationContext> time_context;
            if (system.temporalOrder() > 0) {
                time_context = transient.integrator().buildContext(
                    system.temporalOrder(), state);
                state.time_integration = &(*time_context);
            }
            auto callback_point = point;
            if (pass > 0) {
                callback_point =
                    point == NewtonOptions::StateSynchronizationPoint::
                                 RestoredOuterFixedPointState
                        ? NewtonOptions::StateSynchronizationPoint::
                              RestoredProjectedOuterFixedPointState
                        : NewtonOptions::StateSynchronizationPoint::
                              ProjectedOuterFixedPointState;
            }
            options_.synchronize_state(state, callback_point);
            const auto semantic_after =
                constraintSemanticFingerprint(system.constraints());
            const bool semantics_changed =
                anyRank(semantic_after != semantic_before);

            // Always project once and invoke the callback a second time.  Even
            // an unchanged topology may newly constrain the current iterate,
            // and curvature/prescribed extension must be regenerated from the
            // projected state rather than the pre-projection candidate.
            projectWithCurrentConstraints();
            if (pass > 0 && !semantics_changed) {
                return;
            }
            FE_THROW_IF(
                pass >= max_constraint_projection_passes,
                systems::InvalidStateException,
                std::string("NewtonSolver: external-state synchronization did not reach a stable affine-constraint fixed point at '") +
                    stateSyncPointName(point) + "'");
            if (semantics_changed) {
                system.clearLocalCondensedRecovery();
                if (auto* registry =
                        system.auxiliaryInputRegistryIfPresent()) {
                    registry->invalidateAll();
                }
            }
            semantic_before = semantic_after;
        }
    };

    auto previous_outer_iterate =
        workspace.factory->createVector(vector_size);
    FE_CHECK_NOT_NULL(
        previous_outer_iterate.get(),
        "NewtonSolver: external-state previous outer iterate");

    bool entry_state_restored = false;
    auto restoreEntryState = [&]() {
        if (entry_state_restored) {
            return;
        }
        system.rollbackGeometricNonlinearityTrial(/*force=*/true);
        FE_THROW_IF(
            system.meshCoordinateTransactionActive(),
            systems::InvalidStateException,
            "NewtonSolver: external-state rollback left an active mesh-coordinate transaction");
        history.u().copyFrom(
            canonical_entry_defined ? *entry_u : *raw_entry_u);
        restoreEntryHistoryAndRates();
        const auto& auxiliary_state = canonical_entry_defined
                                          ? entry_auxiliary_state
                                          : raw_entry_auxiliary_state;
        if (!auxiliary_state.empty()) {
            system.restoreAuxiliaryState(auxiliary_state);
        }
        system.borderedCoupling() = canonical_entry_defined
                                        ? entry_bordered_state
                                        : raw_entry_bordered_state;
        system.clearLocalCondensedRecovery();
        if (auto* registry = system.auxiliaryInputRegistryIfPresent()) {
            registry->invalidateAll();
        }
        system.updateConstraints(solve_time, history.dt());
        synchronizeOuterState(
            NewtonOptions::StateSynchronizationPoint::
                RestoredOuterFixedPointState);
        if (canonical_entry_defined) {
            FE_THROW_IF(
                constraintSemanticFingerprint(system.constraints()) !=
                    entry_constraint_semantics,
                systems::InvalidStateException,
                "NewtonSolver: restored external-state callback did not reproduce the canonical entry affine-constraint semantics");
            previous_outer_iterate->copyFrom(history.u());
            axpy(*previous_outer_iterate,
                 static_cast<Real>(-1.0),
                 *entry_u);
            const double restored_solution_change =
                previous_outer_iterate->norm();
            const double restoration_tolerance =
                64.0 * std::numeric_limits<double>::epsilon() *
                std::max(1.0, entry_u->norm());
            FE_THROW_IF(
                !std::isfinite(restored_solution_change) ||
                    restored_solution_change > restoration_tolerance,
                systems::InvalidStateException,
                "NewtonSolver: restored external-state callback changed the canonical entry solution while reprojecting its affine constraints");
        }
        entry_state_restored = true;
    };

    NewtonOptions inner_options = options_;
    inner_options.external_state_fixed_point.enabled = false;
    inner_options.synchronize_state = {};
    inner_options.accepted_state_sync_invalidates_residual = false;
    inner_options.min_iterations = 0;
    inner_options.rel_tolerance = 0.0;
    for (auto& criterion : inner_options.field_residual_criteria) {
        criterion.rel_tolerance = 0.0;
    }
    NewtonSolver inner_solver(std::move(inner_options));

    NewtonReport aggregate{};
    backends::SolverReport last_nontrivial_linear{};
    int inner_iterations_total = 0;
    try {
        // The generated active set at a generalized-alpha stage can differ
        // from the state prepared at the preceding endpoint.  Establish its
        // complete constraint/projection fixed point before defining the
        // rollback fingerprint.  The raw committed history/rate snapshots
        // above remain the immutable source used on every later outer pass;
        // only the current algebraic state and external solver state are
        // canonicalized here.
        system.rollbackGeometricNonlinearityTrial(/*force=*/true);
        system.clearLocalCondensedRecovery();
        synchronizeOuterState(
            NewtonOptions::StateSynchronizationPoint::
                OuterFixedPointState);
        entry_u->copyFrom(history.u());
        entry_auxiliary_state = system.checkpointAuxiliaryState();
        entry_bordered_state = system.borderedCoupling();
        entry_constraint_semantics =
            constraintSemanticFingerprint(system.constraints());
        canonical_entry_defined = true;

        const int max_outer =
            options_.external_state_fixed_point.max_iterations;
        for (int outer = 0; outer < max_outer; ++outer) {
            if (outer > 0) {
                system.rollbackGeometricNonlinearityTrial(/*force=*/true);
                system.clearLocalCondensedRecovery();
                restoreEntryHistoryAndRates();
                system.updateConstraints(solve_time, history.dt());
                synchronizeOuterState(
                    NewtonOptions::StateSynchronizationPoint::
                        OuterFixedPointState);
            }
            FE_THROW_IF(
                system.meshCoordinateTransactionActive(),
                systems::InvalidStateException,
                "NewtonSolver: external-state refresh opened a mesh-coordinate transaction");

            previous_outer_iterate->copyFrom(history.u());
            auto inner_report = inner_solver.solveStepFrozenExternalState(
                transient,
                linear,
                solve_time,
                history,
                workspace,
                residual_addition);
            inner_iterations_total += inner_report.iterations;
            if (inner_report.iterations > 0 ||
                inner_report.linear.iterations > 0) {
                last_nontrivial_linear = inner_report.linear;
            }

            workspace.residual_scratch->copyFrom(history.u());
            axpy(*workspace.residual_scratch,
                 static_cast<Real>(-1.0),
                 *previous_outer_iterate);
            const double state_change_norm =
                workspace.residual_scratch->norm();

            aggregate = inner_report;
            aggregate.outer_iterations = outer + 1;
            aggregate.inner_iterations_total = inner_iterations_total;
            aggregate.iterations = inner_iterations_total;
            aggregate.outer_state_change_norm = state_change_norm;

            if (oopTraceEnabled()) {
                std::ostringstream oss;
                oss << "NewtonSolver: external-state fixed point"
                    << " outer_iteration=" << (outer + 1)
                    << " inner_converged="
                    << (inner_report.converged ? 1 : 0)
                    << " inner_iterations=" << inner_report.iterations
                    << " inner_iterations_total="
                    << inner_iterations_total
                    << " state_change_norm=" << state_change_norm
                    << " refreshed_residual_norm="
                    << inner_report.residual_norm;
                traceLog(oss.str());
            }

            if (!inner_report.converged) {
                aggregate.converged = false;
                restoreEntryState();
                return aggregate;
            }

            // A zero-update solve is the convergence certificate: the
            // residual was assembled after G was regenerated from this exact
            // algebraic state and already met every absolute tolerance.
            if (inner_report.iterations == 0) {
                aggregate.converged = true;
                if (inner_iterations_total > 0 &&
                    aggregate.linear.iterations == 0) {
                    aggregate.linear = last_nontrivial_linear;
                }
                return aggregate;
            }
        }

        aggregate.converged = false;
        restoreEntryState();
        return aggregate;
    } catch (...) {
        const auto original_failure = std::current_exception();
        try {
            restoreEntryState();
        } catch (...) {
            std::throw_with_nested(systems::InvalidStateException(
                "NewtonSolver: external-state fixed-point failure was followed by a rollback failure"));
        }
        std::rethrow_exception(original_failure);
    }
}

NewtonReport NewtonSolver::solveStepFrozenExternalState(
    systems::TransientSystem& transient,
    backends::LinearSolver& linear,
    double solve_time,
    TimeHistory& history,
    NewtonWorkspace& workspace,
    const backends::GenericVector* residual_addition) const
{
    FE_THROW_IF(!workspace.isAllocated(), InvalidArgumentException,
                "NewtonSolver::solveStep: workspace not allocated");

    // Interface-tracking constraints can re-augment the sparsity patterns at
    // any state-sync point (e.g. the accepted-state cut refresh at the end of
    // the previous solve). The matrix references below bind for the whole
    // step, so the reallocation must happen here, before any binding.
    maybeReallocateJacobianForSparsity(transient.system(), workspace);

    struct CurrentJacobianProxy {
        NewtonWorkspace& workspace;

        [[nodiscard]] backends::GenericMatrix& get() const
        {
            FE_CHECK_NOT_NULL(workspace.jacobian.get(), "NewtonSolver workspace.jacobian");
            return *workspace.jacobian;
        }

        [[nodiscard]] std::unique_ptr<assembly::GlobalSystemView> createAssemblyView() const
        {
            return get().createAssemblyView();
        }

        [[nodiscard]] backends::BackendKind backendKind() const
        {
            return get().backendKind();
        }

        [[nodiscard]] GlobalIndex numRows() const
        {
            return get().numRows();
        }

        [[nodiscard]] GlobalIndex numCols() const
        {
            return get().numCols();
        }

        void zero() const
        {
            get().zero();
        }

        void mult(const backends::GenericVector& x, backends::GenericVector& y) const
        {
            get().mult(x, y);
        }

        operator backends::GenericMatrix&() const
        {
            return get();
        }

    };

    CurrentJacobianProxy J{workspace};
    auto& r = *workspace.residual;
    auto& du = *workspace.delta;
    auto& u_backup = *workspace.u_backup;
    auto& residual_scratch = *workspace.residual_scratch;
    auto& residual_base = *workspace.residual_base;
    auto& residual_minus = *workspace.residual_minus;
    NewtonReport report;

    const auto& sys = transient.system();
#if FE_HAS_MPI
    const NewtonCommunicator system_communicator =
        sys.activeMpiCommunicator();
#else
    constexpr NewtonCommunicator system_communicator =
        kSerialNewtonCommunicator;
#endif
    struct FieldResidualConvergenceState {
        NewtonOptions::FieldResidualCriterion criterion{};
        FieldResidualNormDefinition definition{};
        std::string name{};
        double initial_norm{std::numeric_limits<double>::quiet_NaN()};
        double current_norm{std::numeric_limits<double>::quiet_NaN()};
        std::uint64_t owned_dof_count{0u};
        bool relative_reference_available{false};
        bool relative_reference_activated_this_sample{false};
    };
    std::vector<FieldResidualConvergenceState> field_residual_states;
    std::vector<FieldResidualNormDefinition> field_residual_definitions;
    if (!options_.field_residual_criteria.empty()) {
        const auto unknown_fields = sys.unknownFieldIdsInDofMapOrder();
        field_residual_states.reserve(options_.field_residual_criteria.size());
        field_residual_definitions.reserve(options_.field_residual_criteria.size());
        for (const auto& criterion : options_.field_residual_criteria) {
            FE_THROW_IF(std::find(unknown_fields.begin(),
                                  unknown_fields.end(),
                                  criterion.field) == unknown_fields.end(),
                        InvalidArgumentException,
                        "NewtonSolver: field residual criterion must reference an unknown field in the system");
            const auto begin = sys.fieldDofOffset(criterion.field);
            const auto count = sys.fieldDofHandler(criterion.field).getNumDofs();
            FE_THROW_IF(begin < 0 || count <= 0 || begin + count > r.size(),
                        systems::InvalidStateException,
                        "NewtonSolver: field residual criterion has an invalid monolithic DOF range");
            FieldResidualNormDefinition definition{
                criterion.field,
                begin,
                begin + count};
            field_residual_definitions.push_back(definition);
            field_residual_states.push_back(FieldResidualConvergenceState{
                criterion,
                definition,
                sys.fieldRecord(criterion.field).name});
        }
    }
    const auto base_linear_options = sys.augmentSolverOptions(linear.getOptions());
    linear.setOptions(base_linear_options);
    const auto& constraints = sys.constraints();
    const int temporal_order = transient.system().temporalOrder();

    auto anyRank = [&](bool local_value) {
#if FE_HAS_MPI
        int initialized = 0;
        int finalized = 0;
        MPI_Initialized(&initialized);
        MPI_Finalized(&finalized);
        if (initialized != 0 && finalized == 0) {
            const int local = local_value ? 1 : 0;
            int global = local;
            MPI_Allreduce(&local,
                          &global,
                          1,
                          MPI_INT,
                          MPI_MAX,
                          sys.activeMpiCommunicator());
            return global != 0;
        }
#endif
        return local_value;
    };

    auto allRanks = [&](bool local_value) {
        return !anyRank(!local_value);
    };

    auto syncHistoryState = [&]() {
        history.updateGhosts();
        // Constraint storage is rank-local.  Every rank in the system
        // communicator must nevertheless enter the same halo exchanges when
        // any rank owns a relevant line.
        if (!anyRank(!constraints.empty())) {
            return;
        }
        constraints.distribute(history.u());
        syncOwnedRowHaloIfNeeded(history.u());
        // Do NOT distribute the (current-time) constraint values into the
        // history states. Committed history already satisfies the constraints
        // at its own time levels; stamping the stage inhomogeneity g(t_stage)
        // over u^n (and over the injected-rate slot used by the first-order
        // generalized-alpha stencil) rewrites the trajectory of time-dependent
        // Dirichlet data. The stencil then sees zero wall velocity increments
        // and a value written into a rate slot, i.e. a wall acceleration of
        // c0*g(t) instead of g_dot(t); the consistent-mass coupling turns the
        // missing g_dot into a secular, h- and dt-independent velocity error
        // in wall-adjacent cells (observed as the bottom-wall sawtooth in the
        // open-vessel MMS). Set SVMP_DISTRIBUTE_CONSTRAINTS_INTO_HISTORY=1 to
        // restore the legacy stamping for comparison.
        if (distributeConstraintsIntoHistoryRequested()) {
            for (int k = 1; k <= history.historyDepth(); ++k) {
                constraints.distribute(history.uPrevK(k));
                syncOwnedRowHaloIfNeeded(history.uPrevK(k));
            }
        } else {
            for (int k = 1; k <= history.historyDepth(); ++k) {
                syncOwnedRowHaloIfNeeded(history.uPrevK(k));
            }
        }
        // Master-bearing (MPC) lines are different from Dirichlet data: their
        // slave values come from the SAME vector's masters, so distributing
        // them into the history cannot rewrite a prescribed trajectory — it
        // re-imposes the extension on whatever the masters already were at
        // that time level. Doing this at every state sync keeps DOFs that
        // just entered an interface-tracking MPC set (cut refreshes rebuild
        // the slave set mid-solve) on the extension trajectory at ALL time
        // levels, so the generalized-alpha stage stencils never see the
        // free-vs-extension value jump as a 1/(gamma*dt)-scaled rate pulse
        // (band-localized velocity-error floor in the open-vessel MMS). The
        // rate slots get the homogeneous form: constraint coefficients are
        // time-constant within a step, so slave rates are the same master
        // combination. Dirichlet lines are untouched throughout (their
        // finite-difference rates keep carrying g_dot).
        const bool any_master_bearing_lines =
            anyRank(constraints.hasMasterBearingLines());
        if (!mpcStateDistributeDisabled() && any_master_bearing_lines) {
            for (int k = 1; k <= history.historyDepth(); ++k) {
                if (transient.integrator().historySlotStoresRate(k)) {
                    constraints.distributeMasterBearingHomogeneous(
                        history.uPrevK(k));
                } else {
                    constraints.distributeMasterBearing(history.uPrevK(k));
                }
                syncOwnedRowHaloIfNeeded(history.uPrevK(k));
            }
            if (history.hasUDotState()) {
                constraints.distributeMasterBearingHomogeneous(history.uDot());
                syncOwnedRowHaloIfNeeded(history.uDot());
            }
            if (history.hasUDDotState()) {
                constraints.distributeMasterBearingHomogeneous(history.uDDot());
                syncOwnedRowHaloIfNeeded(history.uDDot());
            }
        }
    };

    auto syncCurrentState = [&]() {
        constraints.updateGhostsAndDistribute(history.u());
        syncOwnedRowHaloIfNeeded(history.u());
    };

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
    using StateSyncPoint = NewtonOptions::StateSynchronizationPoint;
    auto invokeStateSynchronization = [&](
                                          const systems::SystemStateView& state,
                                          StateSyncPoint point) {
        if (options_.synchronize_state) {
            options_.synchronize_state(state, point);
        }
    };

    // Ensure time-dependent constraints (Dirichlet, etc.) are evaluated at the actual solve time.
    // This is required for multi-stage schemes (e.g., generalized-α) where the nonlinear solve
    // occurs at a stage time t_{n+α_f}, not necessarily at t_{n+1}.
    FE_THROW_IF(!(history.dt() > 0.0), InvalidArgumentException, "NewtonSolver: dt must be > 0");
    FE_THROW_IF(!std::isfinite(solve_time), InvalidArgumentException, "NewtonSolver: solve_time must be finite");
    transient.system().updateConstraints(solve_time, history.dt());
    if (oopTraceEnabled()) {
        logVectorComponentNorms(transient.system(), history.u(), "before_newton_sync");
    }
    syncHistoryState();
    if (oopTraceEnabled()) {
        logVectorComponentNorms(transient.system(), history.u(), "after_newton_sync");
    }

    auto base_state_holder = makeNewtonState(history, solve_time);
    const auto& base_state = base_state_holder.view;

    std::optional<assembly::TimeIntegrationContext> dt_scale_ctx;
    if (options_.scale_dt_increments && !(options_.dt_increment_scale > 0.0)) {
        const int max_order = transient.system().temporalOrder();
        if (max_order > 0) {
            dt_scale_ctx = transient.integrator().buildContext(max_order, base_state);
        }
    }

    const int max_it = options_.max_iterations;
    const int min_it = options_.min_iterations;

    if (oopTraceEnabled()) {
        const auto& lopts = linear.getOptions();
        std::ostringstream oss;
        oss << "NewtonSolver::solveStep: time=" << solve_time << " dt=" << base_state.dt
            << " max_it=" << max_it
            << " min_it=" << min_it
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
    const bool newton_assembly_diagnostics =
        newtonAssemblyDiagnosticsEnabled();
    int current_newton_iteration = -1;
    bool has_monolithic_auxiliary_unknowns = false;
    if (const auto* aux_registry = transient.system().auxiliaryOperatorRegistryIfPresent();
        aux_registry && aux_registry->isLayoutFinalized()) {
        has_monolithic_auxiliary_unknowns =
            aux_registry->auxiliaryLayout().total_aux_unknowns > 0;
    }

    std::vector<GlobalIndex> constrained_dofs;
    std::vector<GlobalIndex> dirichlet_dofs;
    bool any_constrained_dofs = false;
    bool has_non_dirichlet_affine_constraints = false;
    bool any_non_dirichlet_affine_constraints = false;
    auto refreshConstraintDofCaches = [&]() {
        constrained_dofs.clear();
        dirichlet_dofs.clear();
        has_non_dirichlet_affine_constraints = false;
        if (constraints.empty()) {
            any_constrained_dofs = anyRank(false);
            linear.setDirichletDofs({});
            any_non_dirichlet_affine_constraints = anyRank(false);
            return;
        }
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
        any_constrained_dofs = anyRank(!constrained_dofs.empty());

        std::sort(dirichlet_dofs.begin(), dirichlet_dofs.end());
        dirichlet_dofs.erase(std::unique(dirichlet_dofs.begin(), dirichlet_dofs.end()),
                             dirichlet_dofs.end());
        linear.setDirichletDofs(dirichlet_dofs);
        any_non_dirichlet_affine_constraints =
            anyRank(has_non_dirichlet_affine_constraints);
    };
    refreshConstraintDofCaches();

    auto zeroConstrainedResidualEntries = [&]() {
        if (!any_constrained_dofs) {
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
        if (!any_constrained_dofs) {
            return;
        }
        auto J_zero = J.createAssemblyView();
        FE_CHECK_NOT_NULL(J_zero.get(), "NewtonSolver: Jacobian constrained-row view");
        J_zero->beginAssemblyPhase();
        J_zero->zeroRows(constrained_dofs, /*set_diagonal=*/true);
        J_zero->finalizeAssembly();
    };

    bool have_residual = false;
    bool have_jacobian = false;
    int last_jacobian_it = -1;
    bool ptc_mass_ready = false;
    double ptc_gamma = 0.0;
    double ptc_gamma_applied = 0.0;
    double ptc_prev_residual_norm = std::numeric_limits<double>::quiet_NaN();
    bool line_search_history_transaction_active = false;
    auto invalidateConstraintDependentAlgebra = [&]() {
        have_residual = false;
        have_jacobian = false;
        last_jacobian_it = -1;
        ptc_mass_ready = false;
        ptc_gamma = 0.0;
        ptc_gamma_applied = 0.0;
        ptc_prev_residual_norm =
            std::numeric_limits<double>::quiet_NaN();
        linear.setRankOneUpdates({});
        linear.setReducedFieldUpdates({});
        linear.setGroupedBorderedFieldCouplings({});
    };
    auto synchronizeState = [&](const systems::SystemStateView& state,
                                StateSyncPoint point) {
        auto distribute_with_current_constraints = [&]() {
            const bool geometry_transaction_active_on_any_rank = anyRank(
                transient.system().meshCoordinateTransactionActive());

            // Trial merit must use the same state-dependent MPC semantics for
            // u and for every vector in the transient stencil.  Those history
            // changes are safe only while the line-search history transaction
            // is armed; every rejected alpha restores its exact base copies.
            const bool mutable_current_history_state =
                state.u_vector == &history.u();
            const bool history_projection_is_transaction_safe =
                point != StateSyncPoint::LineSearchTrialResidual ||
                line_search_history_transaction_active;
            if (mutable_current_history_state &&
                history_projection_is_transaction_safe) {
                syncHistoryState();
            } else if (point == StateSyncPoint::LineSearchTrialResidual ||
                       geometry_transaction_active_on_any_rank) {
                FE_THROW_IF(state.u_vector == nullptr,
                            systems::InvalidStateException,
                            "NewtonSolver: a line-search synchronization changed "
                            "constraints for a non-mutable state view");
                auto& candidate = *const_cast<backends::GenericVector*>(
                    state.u_vector);
                constraints.updateGhostsAndDistribute(candidate);
                syncOwnedRowHaloIfNeeded(candidate);
            } else if (state.u_vector == &history.u()) {
                syncHistoryState();
            } else {
                FE_THROW_IF(state.u_vector == nullptr,
                            systems::InvalidStateException,
                            "NewtonSolver: synchronization changed constraints "
                            "for a non-mutable state view");
                auto& synchronized = *const_cast<backends::GenericVector*>(
                    state.u_vector);
                constraints.updateGhostsAndDistribute(synchronized);
                syncOwnedRowHaloIfNeeded(synchronized);
            }
        };

        // A synchronization callback can derive cut/support constraints from
        // the trial state.  If those constraints change, project the state and
        // call the callback again so curvature, extension fields, and other
        // residual-defining data see the projected vector rather than the
        // pre-constraint candidate.  Require a bounded fixed point instead of
        // silently assembling with mutually inconsistent state and constraints.
        constexpr int max_constraint_projection_passes = 3;
        auto semantic_before = constraintSemanticFingerprint(constraints);
        for (int pass = 0;; ++pass) {
            invokeStateSynchronization(state, point);
            const auto semantic_after =
                constraintSemanticFingerprint(constraints);
            const bool changed_on_any_rank =
                anyRank(semantic_after != semantic_before);
            if (!changed_on_any_rank) {
                // A line-search trial installs candidate MPC semantics before
                // it is accepted.  The accepted callback can therefore be
                // semantically unchanged even though committed history and
                // rate vectors have not yet been projected with that set.
                // Complete the transaction before any refreshed residual or
                // convergence decision.  Use a communicator-wide predicate
                // because relevant constraints are stored rank-locally.
                if (point == StateSyncPoint::AcceptedNonlinearState &&
                    state.u_vector == &history.u() &&
                    !mpcStateDistributeDisabled() &&
                    anyRank(constraints.hasMasterBearingLines())) {
                    syncHistoryState();
                    invalidateConstraintDependentAlgebra();
                }
                return;
            }

            FE_THROW_IF(
                pass >= max_constraint_projection_passes,
                systems::InvalidStateException,
                std::string("NewtonSolver: state synchronization did not reach a stable affine-constraint fixed point at '") +
                    stateSyncPointName(point) + "'");

            distribute_with_current_constraints();
            const bool refresh_geometry_on_any_rank = anyRank(
                transient.system().geometricNonlinearityEnabled() ||
                transient.system().meshCoordinateTransactionActive());
            if (refresh_geometry_on_any_rank) {
                // A constraint projection can also change a displacement
                // unknown.  Refresh coordinates before the second callback
                // at every synchronization point, not only during line
                // search, so cut geometry and derived fields use the same
                // projected state.  An active transaction retains the
                // original accepted-coordinate backup.
                transient.system().beginGeometricNonlinearityTrial(state);
            }
            refreshConstraintDofCaches();

            // Coefficients and inhomogeneities alter the transformed operator
            // even when the slave/master sparsity is unchanged.  Any semantic
            // change therefore invalidates all algebra tied to the old set.
            invalidateConstraintDependentAlgebra();
            semantic_before = semantic_after;
        }
    };

    // The initial callback may build state-dependent cuts and constraints.
    // Use the same projection/refresh fixed-point contract as every later
    // residual evaluation before declaring this the accepted base state.
    try {
        transient.system().beginGeometricNonlinearityTrial(base_state);
        synchronizeState(base_state, StateSyncPoint::AcceptedNonlinearState);
        transient.system().acceptGeometricNonlinearityState(
            base_state,
            systems::GeometricNonlinearityUpdatePoint::AcceptedNonlinearState);
    } catch (...) {
        transient.system().rollbackGeometricNonlinearityTrial(/*force=*/true);
        throw;
    }

    ResidualNormComponents current_residual_components{};
    ResidualNormComponents initial_residual_components{};
    double current_residual_norm = std::numeric_limits<double>::quiet_NaN();

    auto componentResidualConvergenceActive = [&]() -> bool {
        const auto& bordered = transient.system().borderedCoupling();
        return has_monolithic_auxiliary_unknowns && bordered.active && bordered.n_aux > 0;
    };

    auto computeResidualComponents = [&]() -> ResidualNormComponents {
        return borderedResidualNormComponentsForConvergence(
            r,
            residual_scratch,
            transient.system().borderedCoupling(),
            system_communicator);
    };

    auto computeResidualNorm = [&]() -> double {
        return computeResidualComponents().combined();
    };

    auto refreshFieldResidualNorms = [&]() {
        const auto sample = fieldResidualNormsForConvergence(
            sys,
            r,
            std::span<const FieldResidualNormDefinition>(
                field_residual_definitions.data(),
                field_residual_definitions.size()));
        FE_THROW_IF(sample.norms.size() != field_residual_states.size() ||
                        sample.owned_dof_counts.size() !=
                            field_residual_states.size(),
                    systems::InvalidStateException,
                    "NewtonSolver: inconsistent field residual convergence sample");
        for (std::size_t i = 0; i < field_residual_states.size(); ++i) {
            FE_THROW_IF(sample.owned_dof_counts[i] == 0u,
                        systems::InvalidStateException,
                        "NewtonSolver: field residual convergence criterion has no owned DOFs on the active communicator");
            auto& state = field_residual_states[i];
            state.current_norm = sample.norms[i];
            state.owned_dof_count = sample.owned_dof_counts[i];
            state.relative_reference_activated_this_sample = false;
        }
    };

    auto refreshResidualComponents = [&]() -> double {
        current_residual_components = computeResidualComponents();
        current_residual_norm = current_residual_components.combined();
        refreshFieldResidualNorms();
        return current_residual_norm;
    };

    auto updateResidualReport = [&]() {
        report.residual_norm = current_residual_norm;
        report.field_residual_norm = current_residual_components.field;
        report.auxiliary_residual_norm = current_residual_components.auxiliary;
        report.component_residual_convergence = componentResidualConvergenceActive();
    };

    auto traceResidualDebugState = [&](const char* phase) {
        if (!oopTraceEnabled()) {
            return;
        }
        std::ostringstream oss;
        oss << "NewtonSolver: residual debug";
        if (phase != nullptr) {
            oss << " phase='" << phase << "'";
        }
        oss << " raw_l2=" << r.norm()
            << " conv_norm=" << computeResidualNorm()
            << " constrained_dofs=" << constrained_dofs.size();
        traceLog(oss.str());
        logVectorComponentNorms(transient.system(), r, "residual debug");
    };

    auto traceResidualComponents = [&](const char* phase) {
        const auto components = computeResidualComponents();
        std::ostringstream oss;
        oss << "NewtonSolver: residual block norms"
            << " diagnostic=residual_block_norms";
        if (phase != nullptr) {
            oss << " phase='" << phase << "'";
        }
        oss << " field=" << components.field
            << " aux=" << components.auxiliary
            << " combined=" << components.combined();
        FE_LOG_INFO(oss.str());
        logVectorComponentNorms(transient.system(), r, "residual_block_norms");
        logVectorComponentNorms(transient.system(), history.u(), "solution_state");
    };

    const bool ptc_enabled = options_.pseudo_transient.enabled;
    std::vector<GlobalIndex> ptc_owned_dofs;
    bool ptc_uses_backend_owned_rows = false;
    if (ptc_enabled && workspace.ptc_mass_lumped != nullptr) {
        const auto dt_fields = sys.timeDerivativeFields(options_.jacobian_op);
        if (!dt_fields.empty()) {
            dofs::IndexSet dt_dofs_all;
            const auto& fmap = sys.fieldMap();
            for (const auto fid : dt_fields) {
                const auto idx = fieldMapIndexForFieldId(sys, fid);
                if (!idx || *idx >= fmap.numFields()) {
                    continue;
                }
                const auto range = fmap.getFieldDofRange(*idx);
                dt_dofs_all = dt_dofs_all.unionWith(dofs::IndexSet(range.first, range.second));
            }
#if defined(FE_HAS_FSILS)
            if (const auto* fsils_jacobian = dynamic_cast<const backends::FsilsMatrix*>(&J.get());
                fsils_jacobian != nullptr && fsils_jacobian->usesOwnedRowOperator()) {
                ptc_uses_backend_owned_rows = true;
                const auto dt_dofs = dt_dofs_all.toVector();
                ptc_owned_dofs.reserve(dt_dofs.size());
                for (const auto dof : dt_dofs) {
                    if (fsils_jacobian->ownsFeDofRow(dof)) {
                        ptc_owned_dofs.push_back(dof);
                    }
                }
            } else
#endif
            {
                const auto& owned = sys.dofHandler().getPartition().locallyOwned();
                ptc_owned_dofs = dt_dofs_all.intersectionWith(owned).toVector();
            }
        }
    }

    // PTC assembly, matrix replacement, and residual refresh are collective on
    // the active system communicator.  Some valid distributed layouts have no
    // locally owned time-derivative row on one or more ranks, so a rank-local
    // nonempty-row test would let those ranks skip collectives entered by their
    // peers.  Require storage everywhere, but only one communicator rank needs
    // to own a row for the distributed diagonal to be meaningful.
    const bool local_ptc_storage_available =
        ptc_enabled && (workspace.ptc_mass_lumped != nullptr);
    const bool ptc_storage_available_on_all_ranks =
        allRanks(local_ptc_storage_available);
    const bool any_ptc_owned_rows =
        anyRank(local_ptc_storage_available && !ptc_owned_dofs.empty());
    const bool ptc_can_run =
        ptc_storage_available_on_all_ranks && any_ptc_owned_rows;
    if (oopTraceEnabled() && ptc_enabled) {
        traceLog("NewtonSolver: PTC diagonal ownership rows=" + std::to_string(ptc_owned_dofs.size()) +
                 (ptc_uses_backend_owned_rows ? " mode=backend-owned-row" : " mode=fe-owned"));
    }
    const bool jacobian_matrix_state_independent =
        sys.operatorMatrixStateIndependent(options_.jacobian_op);
    const bool base_can_reuse_state_independent_jacobian =
        jacobian_matrix_state_independent &&
        !ptc_enabled &&
        !has_monolithic_auxiliary_unknowns;
    if (oopTraceEnabled()) {
        std::string reason;
        if (!jacobian_matrix_state_independent) {
            reason = "operator matrix is state-dependent or unknown";
        } else if (ptc_enabled) {
            reason = "pseudo-transient continuation modifies the matrix";
        } else if (has_monolithic_auxiliary_unknowns) {
            reason = "monolithic auxiliary unknowns require fresh coupled assembly";
        } else if (any_non_dirichlet_affine_constraints) {
            reason = "non-Dirichlet affine constraints can alter matrix structure";
        } else {
            reason = "operator matrix is state-independent";
        }
        traceLog("NewtonSolver: jacobian reuse decision op='" + options_.jacobian_op +
                 "' state_independent=" +
                 (jacobian_matrix_state_independent ? "true" : "false") +
                 " reuse=" +
                 ((base_can_reuse_state_independent_jacobian &&
                   !any_non_dirichlet_affine_constraints)
                      ? "enabled"
                      : "disabled") +
                 " reason='" + reason + "'");
    }
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

    auto validateLowRankUpdateSlots =
        [&](std::size_t rank_one_count,
            std::size_t reduced_count,
            std::size_t grouped_count,
            const char* phase) {
#if FE_HAS_MPI
            int initialized = 0;
            int finalized = 0;
            MPI_Initialized(&initialized);
            MPI_Finalized(&finalized);
            if (initialized != 0 && finalized == 0 &&
                system_communicator != MPI_COMM_NULL) {
                const std::array<unsigned long long, 3> local_counts{
                    static_cast<unsigned long long>(rank_one_count),
                    static_cast<unsigned long long>(reduced_count),
                    static_cast<unsigned long long>(grouped_count)};
                auto min_counts = local_counts;
                auto max_counts = local_counts;
                MPI_Allreduce(local_counts.data(),
                              min_counts.data(),
                              static_cast<int>(local_counts.size()),
                              MPI_UNSIGNED_LONG_LONG,
                              MPI_MIN,
                              system_communicator);
                MPI_Allreduce(local_counts.data(),
                              max_counts.data(),
                              static_cast<int>(local_counts.size()),
                              MPI_UNSIGNED_LONG_LONG,
                              MPI_MAX,
                              system_communicator);
                FE_THROW_IF(
                    min_counts != max_counts,
                    systems::InvalidStateException,
                    "NewtonSolver: low-rank update slot counts differ across "
                    "the active system communicator at phase '" +
                        std::string(phase != nullptr ? phase : "unknown") +
                        "'");
            }
#else
            (void)rank_one_count;
            (void)reduced_count;
            (void)grouped_count;
            (void)phase;
#endif
        };

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
        validateLowRankUpdateSlots(effective_rank_one_updates.size(),
                                   active_reduced_field_updates.size(),
                                   grouped_bordered_field_couplings.size(),
                                   "assembly_capture");
    };

    auto logNewtonAssemblyDiagnostic =
        [&](const char* phase,
            StateSyncPoint sync_point,
            const systems::AssemblyRequest& req,
            const auto& result) {
            if (!newton_assembly_diagnostics) {
                return;
            }
            std::ostringstream oss;
            oss << "[svMultiPhysics::FE] NewtonSolver assembly"
                << " diagnostic=newton_assembly"
                << " rank=" << mpiRank()
                << " iteration=" << current_newton_iteration
                << " phase=" << (phase != nullptr ? phase : "unknown")
                << " sync_point=" << stateSyncPointName(sync_point)
                << " op='" << req.op << "'"
                << " want_matrix=" << (req.want_matrix ? 1 : 0)
                << " want_vector=" << (req.want_vector ? 1 : 0)
                << " zero_outputs=" << (req.zero_outputs ? 1 : 0)
                << " suppress_auxiliary_coupling_assembly="
                << (req.suppress_auxiliary_coupling_assembly ? 1 : 0)
                << " nonlinear_iteration="
                << (req.is_nonlinear_iteration ? 1 : 0)
                << " same_op=" << (same_op ? 1 : 0)
                << " solve_time=" << solve_time
                << " dt=" << base_state.dt
                << " success=" << (result.success ? 1 : 0)
                << " elements=" << result.elements_assembled
                << " boundary_faces=" << result.boundary_faces_assembled
                << " interior_faces=" << result.interior_faces_assembled
                << " interface_faces=" << result.interface_faces_assembled
                << " matrix_entries=" << result.matrix_entries_inserted
                << " vector_entries=" << result.vector_entries_inserted
                << " elapsed_seconds=" << result.elapsed_time_seconds;
            FE_LOG_INFO(oss.str());
        };

    auto assemblePressureRowContributionDiagnostics =
        [&](const systems::SystemStateView& state,
            const char* phase,
            StateSyncPoint sync_point) {
            if (!pressureRowContributionDiagnosticEnabled()) {
                return;
            }
            const char* ops[] = {
                "equations_diagnostic_ns_galerkin_continuity",
                "equations_diagnostic_ns_active_continuity",
                "equations_diagnostic_ns_vms_pspg",
                "equations_diagnostic_ns_vms_pspg_pressure_gradient",
                "equations_diagnostic_ns_vms_pspg_nonpressure",
                "equations_diagnostic_ns_vms_pspg_boundary_pressure_gradient",
                "equations_diagnostic_ns_vms_pspg_boundary_pressure_flux",
                "equations_diagnostic_ns_vms_pspg_boundary_tangential_pressure_gradient",
                "equations_diagnostic_ns_vms_pspg_boundary_tangential_momentum_residual",
                "equations_diagnostic_ns_pressure_ghost_penalty",
                "equations_diagnostic_ns_free_surface_pressure_reference_probe",
                "equations_diagnostic_ns_free_surface_tangential_pressure_gradient_probe",
            };
            for (const char* op : ops) {
                if (!transient.system().hasOperator(op)) {
                    if (mpiRank() == 0) {
                        std::ostringstream skipped;
                        skipped
                            << "NewtonSolver: pressure row contribution diagnostic skipped"
                            << " diagnostic=pressure_row_contribution_skipped"
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
                    "NewtonSolver: pressure row contribution diagnostic view");
                std::unique_ptr<assembly::GlobalSystemView> matrix_scratch_view;
                // A state-sync callback can re-augment constraint sparsity
                // and replace the distributed diagnostic matrix.  Resolve
                // the pointer after that synchronization/reallocation rather
                // than retaining a solveStep-lifetime raw pointer.
                auto* diagnostic_jacobian_scratch =
                    workspace.diagnostic_jacobian_scratch.get();
                const bool want_matrix_diagnostic =
                    pressureRowContributionMatrixDiagnosticEnabled() &&
                    diagnostic_jacobian_scratch != nullptr;
                if (want_matrix_diagnostic) {
                    matrix_scratch_view =
                        diagnostic_jacobian_scratch->createAssemblyView();
                    FE_CHECK_NOT_NULL(
                        matrix_scratch_view.get(),
                        "NewtonSolver: pressure row contribution matrix diagnostic view");
                }
                systems::AssemblyRequest req;
                req.op = op;
                req.want_matrix = want_matrix_diagnostic;
                req.want_vector = true;
                req.suppress_constraint_inhomogeneity = true;
                req.suppress_auxiliary_coupling_assembly = true;
                req.is_nonlinear_iteration = true;
                const auto ar =
                    transient.assemble(req,
                                       state,
                                       matrix_scratch_view.get(),
                                       scratch_view.get());
                logNewtonAssemblyDiagnostic(
                    "pressure_row_contribution",
                    sync_point,
                    req,
                    ar);
                FE_THROW_IF(
                    !ar.success,
                    FEException,
                    "NewtonSolver: pressure row contribution diagnostic assembly failed for op '" +
                        std::string(op) + "': " + ar.error_message);

                if (want_matrix_diagnostic) {
                    const std::string matrix_phase =
                        std::string("pressure_row_contribution_matrix:") +
                        (phase != nullptr ? phase : "unknown");
                    logPressureRowOperatorMatrixSupportDiagnostic(
                        transient.system(),
                        *diagnostic_jacobian_scratch,
                        matrix_phase,
                        op,
                        current_newton_iteration,
                        solve_time,
                        base_state.dt);
                    logPressureRowOperatorMatrixSummaryDiagnostic(
                        transient.system(),
                        *diagnostic_jacobian_scratch,
                        constrained_dofs,
                        matrix_phase,
                        op,
                        current_newton_iteration,
                        solve_time,
                        base_state.dt);
                    logDirectPspgFormulationCandidateDiagnostic(
                        transient.system(),
                        *diagnostic_jacobian_scratch,
                        residual_scratch,
                        constrained_dofs,
                        matrix_phase,
                        op,
                        current_newton_iteration,
                        solve_time,
                        base_state.dt);
                }

                const std::string pre_phase =
                    std::string("pressure_row_contribution_pre_constraints:") + op;
                logNewtonFieldResidualDiagnostic(
                    transient.system(),
                    residual_scratch,
                    pre_phase,
                    sync_point,
                    current_newton_iteration,
                    solve_time,
                    base_state.dt);

                zeroVectorEntries(constrained_dofs, residual_scratch);

                const std::string post_phase =
                    std::string("pressure_row_contribution_post_constraints:") + op;
                logNewtonFieldResidualDiagnostic(
                    transient.system(),
                    residual_scratch,
                    post_phase,
                    sync_point,
                    current_newton_iteration,
                    solve_time,
                    base_state.dt);
            }
        };

    struct FreeSurfaceDiagnosticGeometryKey {
        bool revision_tracking_available{false};
        std::uint64_t mesh_geometry_revision{0u};
        std::uint64_t mesh_topology_revision{0u};
        std::uint64_t mesh_ownership_revision{0u};
        std::uint64_t mesh_numbering_revision{0u};
        std::uint64_t mesh_field_layout_revision{0u};
        std::uint64_t mesh_label_revision{0u};
        std::uint64_t mesh_active_configuration_epoch{0u};
        std::uint64_t mesh_coordinate_configuration_key{0u};
        std::uintptr_t cut_context_identity{0u};
        std::uint64_t cut_context_content_revision{0u};
        std::uint64_t system_layout_revision{0u};
        std::uint64_t sparsity_pattern_revision{0u};
        ConstraintSemanticFingerprint constraint_semantics{};

        [[nodiscard]] bool operator==(
            const FreeSurfaceDiagnosticGeometryKey&) const noexcept = default;
    };

    auto freeSurfaceDiagnosticGeometryKey = [&]() {
        const auto& mesh = transient.system().meshAccess();
        const auto* cut_context =
            transient.system().cutIntegrationContext();
        FreeSurfaceDiagnosticGeometryKey key;
        // Both fitted and unfitted operators depend on the background mesh.
        // A cut context tracks generated quadrature/content changes, but it
        // cannot invalidate stale entries after untracked background geometry,
        // topology, ownership, numbering, field-layout, or label changes.
        // Cache only when IMeshAccess explicitly promises that all revision
        // queries below form a trustworthy invalidation domain.
        key.revision_tracking_available = mesh.revisionTrackingAvailable();
        key.mesh_geometry_revision = mesh.geometryRevision();
        key.mesh_topology_revision = mesh.topologyRevision();
        key.mesh_ownership_revision = mesh.ownershipRevision();
        key.mesh_numbering_revision = mesh.numberingRevision();
        key.mesh_field_layout_revision = mesh.fieldLayoutRevision();
        key.mesh_label_revision = mesh.labelRevision();
        key.mesh_active_configuration_epoch =
            mesh.activeConfigurationEpoch();
        key.mesh_coordinate_configuration_key =
            mesh.coordinateConfigurationKey();
        key.cut_context_identity =
            reinterpret_cast<std::uintptr_t>(cut_context);
        key.cut_context_content_revision =
            cut_context != nullptr ? cut_context->contentRevision() : 0u;
        key.system_layout_revision =
            transient.system().systemLayoutRevision();
        key.sparsity_pattern_revision =
            transient.system().sparsityPatternRevision();
        key.constraint_semantics =
            constraintSemanticFingerprint(constraints);
        return key;
    };

    struct FreeSurfacePressureRepresentabilityCache {
        bool valid{false};
        FreeSurfaceDiagnosticGeometryKey geometry{};
        PressureRepresentabilityLsqrResult result{};
    };
    FreeSurfacePressureRepresentabilityCache
        free_surface_pressure_representability_cache;

    bool free_surface_conservative_balance_unavailable_logged = false;
    auto assembleFreeSurfaceConservativeBalanceDiagnostic =
        [&](const systems::SystemStateView& state,
            const char* phase,
            StateSyncPoint sync_point) {
            const bool local_diagnostic_enabled =
                freeSurfaceConservativeBalanceDiagnosticEnabled();
            const bool diagnostic_enabled_on_any_rank =
                anyRank(local_diagnostic_enabled);
            if (!diagnostic_enabled_on_any_rank) {
                return;
            }
            if (!allRanks(local_diagnostic_enabled)) {
                if (!free_surface_conservative_balance_unavailable_logged &&
                    communicatorRank(system_communicator) == 0) {
                    FE_LOG_INFO(
                        "NewtonSolver: free-surface conservative balance"
                        " diagnostic=free_surface_conservative_balance"
                        " available=0"
                        " reason=diagnostic_enablement_differs_across_communicator"
                        " pressure_representability_available=0"
                        " pressure_representability_method=lsqr"
                        " pressure_representability_convergence=normal_equation_stationarity"
                        " pressure_representability_distance_gate_applied=0"
                        " pressure_representability_claimed=0"
                        " pressure_representability_reason=diagnostic_enablement_differs_across_communicator");
                }
                free_surface_conservative_balance_unavailable_logged = true;
                return;
            }

            // Assemble every requested diagnostic sample.  Equality of the
            // current coefficient vector, time scalars, and geometry is not a
            // complete residual-state key: supported forms may also depend on
            // solution/time-step history, auxiliary state/inputs, user data,
            // or runtime parameter callbacks.  Suppressing a full sample on a
            // partial fingerprint could therefore hide a changed pressure or
            // surface load and would lose accepted-state provenance.  Only the
            // mixed pressure pair/LSQR work below is cached, after the current
            // constrained surface load has been reassembled and compared
            // exactly.  The qualification switch forces even that safe cache
            // off so every pair matrix and LSQR solve is repeated.
            const bool every_assembly_on_any_rank = anyRank(
                freeSurfaceConservativeBalanceDiagnosticEveryAssemblyRequested());
            const auto geometry_key =
                freeSurfaceDiagnosticGeometryKey();

            // These tags are installed by the Navier--Stokes formulation only
            // when every effective free surface uses the variational
            // SurfaceStress form.  Keeping the strings here avoids an FE ->
            // Physics dependency while preserving a strict three-vector plus
            // one symmetric-matrix contract for qualification tooling.
            constexpr std::array<const char*, 3> ops{
                "equations_diagnostic_ns_free_surface_pressure_virtual_work",
                "equations_diagnostic_ns_free_surface_surface_energy_virtual_work",
                "equations_diagnostic_ns_free_surface_conservative_balance",
            };
            for (const char* op : ops) {
                const bool local_operator_installed =
                    transient.system().hasOperator(op);
                if (!allRanks(local_operator_installed)) {
                    if (!free_surface_conservative_balance_unavailable_logged &&
                        communicatorRank(system_communicator) == 0) {
                        std::ostringstream skipped;
                        skipped
                            << "NewtonSolver: free-surface conservative balance"
                            << " diagnostic=free_surface_conservative_balance"
                            << " available=0"
                            << " reason=operator_not_installed_on_all_ranks"
                            << " missing_op='" << op << "'"
                            << " pressure_representability_available=0"
                            << " pressure_representability_method=lsqr"
                            << " pressure_representability_convergence=normal_equation_stationarity"
                            << " pressure_representability_distance_gate_applied=0"
                            << " pressure_representability_claimed=0"
                            << " pressure_representability_reason=conservative_balance_operator_unavailable"
                            << " iteration=" << current_newton_iteration
                            << " phase='"
                            << (phase != nullptr ? phase : "unknown") << "'"
                            << " sync_point=" << stateSyncPointName(sync_point)
                            << " scope=pressure_and_surface_energy_first_variations_only"
                            << " contract=instantaneous_constrained_velocity_test_virtual_work"
                            << " total_momentum_equilibrium_claimed=0"
                            << " discrete_energy_theorem_claimed=0";
                        FE_LOG_INFO(skipped.str());
                    }
                    free_surface_conservative_balance_unavailable_logged = true;
                    return;
                }
            }

            const std::string pair_op(
                kFreeSurfacePressureRepresentabilityPairOperator);
            const bool pair_operator_available = allRanks(
                transient.system().hasOperator(pair_op));
            const bool local_representability_workspace_available =
                workspace.pressure_representability_pair_matrix != nullptr &&
                workspace.pressure_representability_load != nullptr &&
                workspace.pressure_representability_solution != nullptr &&
                workspace.pressure_representability_left_basis != nullptr &&
                workspace.pressure_representability_right_basis != nullptr &&
                workspace.pressure_representability_direction != nullptr &&
                workspace.pressure_representability_work != nullptr &&
                workspace.pressure_representability_residual != nullptr &&
                workspace.pressure_representability_normal_residual != nullptr;
            const bool representability_workspace_available = allRanks(
                local_representability_workspace_available);

            bool local_representability_layout_compatible = false;
            if (local_representability_workspace_available) {
                const auto& pair_matrix =
                    *workspace.pressure_representability_pair_matrix;
                const std::array<const backends::GenericVector*, 8>
                    pair_vectors{
                        workspace.pressure_representability_load.get(),
                        workspace.pressure_representability_solution.get(),
                        workspace.pressure_representability_left_basis.get(),
                        workspace.pressure_representability_right_basis.get(),
                        workspace.pressure_representability_direction.get(),
                        workspace.pressure_representability_work.get(),
                        workspace.pressure_representability_residual.get(),
                        workspace.pressure_representability_normal_residual
                            .get(),
                    };
                const auto expected_size = pair_matrix.numRows();
                const auto expected_backend = pair_matrix.backendKind();
                const auto expected_local_size =
                    pair_vectors.front()->localSpan().size();
                local_representability_layout_compatible =
                    expected_size > 0 &&
                    pair_matrix.numCols() == expected_size &&
                    std::all_of(
                        pair_vectors.begin(),
                        pair_vectors.end(),
                        [&](const backends::GenericVector* vector) {
                            return vector != nullptr &&
                                   vector->backendKind() == expected_backend &&
                                   vector->size() == expected_size &&
                                   vector->localSpan().size() ==
                                       expected_local_size;
                        });
#if defined(FE_HAS_FSILS)
                // FSILS matrix-vector products require identity of the shared
                // overlap/permutation layout, not merely equal global and
                // local sizes.  Fail closed before assembly/LSQR if a caller
                // has replaced any diagnostic vector with another layout.
                if (local_representability_layout_compatible &&
                    expected_backend == backends::BackendKind::FSILS) {
                    const auto* fsils_matrix =
                        dynamic_cast<const backends::FsilsMatrix*>(
                            &pair_matrix);
                    const auto* shared =
                        fsils_matrix != nullptr
                            ? fsils_matrix->shared().get()
                            : nullptr;
                    local_representability_layout_compatible =
                        shared != nullptr &&
                        std::all_of(
                            pair_vectors.begin(),
                            pair_vectors.end(),
                            [shared](const backends::GenericVector* vector) {
                                const auto* fsils_vector =
                                    dynamic_cast<const backends::FsilsVector*>(
                                        vector);
                                return fsils_vector != nullptr &&
                                       fsils_vector->shared() == shared;
                            });
                }
#endif
            }
            const bool representability_layout_compatible = allRanks(
                local_representability_layout_compatible);
            const bool representability_storage_usable =
                representability_workspace_available &&
                representability_layout_compatible;

            const bool pressure_pair_matrix_state_independent =
                pair_operator_available &&
                allRanks(transient.system().operatorMatrixStateIndependent(
                    pair_op));
            const bool local_pressure_representability_cache_prior_sample =
                free_surface_pressure_representability_cache.valid;
            const bool local_pressure_representability_cache_geometry_match =
                local_pressure_representability_cache_prior_sample &&
                free_surface_pressure_representability_cache.geometry ==
                    geometry_key;
            const bool local_pressure_representability_cache_candidate =
                !every_assembly_on_any_rank &&
                geometry_key.revision_tracking_available &&
                representability_storage_usable &&
                pressure_pair_matrix_state_independent &&
                local_pressure_representability_cache_geometry_match;
            const bool pressure_representability_cache_candidate =
                allRanks(
                    local_pressure_representability_cache_candidate);
            std::string pressure_representability_cache_rejection_reason;
            if (every_assembly_on_any_rank) {
                pressure_representability_cache_rejection_reason =
                    "every_assembly_policy";
            } else if (!pair_operator_available) {
                pressure_representability_cache_rejection_reason =
                    "pair_operator_unavailable";
            } else if (!geometry_key.revision_tracking_available) {
                pressure_representability_cache_rejection_reason =
                    "revision_key_unavailable";
            } else if (!representability_storage_usable) {
                pressure_representability_cache_rejection_reason =
                    "storage_or_layout_unusable";
            } else if (!pressure_pair_matrix_state_independent) {
                pressure_representability_cache_rejection_reason =
                    "pair_matrix_state_dependent";
            } else if (!local_pressure_representability_cache_prior_sample) {
                pressure_representability_cache_rejection_reason =
                    "no_prior_sample";
            } else if (!local_pressure_representability_cache_geometry_match) {
                pressure_representability_cache_rejection_reason =
                    "geometry_or_constraints_changed";
            } else if (!pressure_representability_cache_candidate) {
                pressure_representability_cache_rejection_reason =
                    "candidate_unavailable_on_another_rank";
            } else {
                pressure_representability_cache_rejection_reason =
                    "exact_load_check_pending";
            }
            if (pressure_representability_cache_candidate) {
                // Preserve the previous constrained surface load in an LSQR
                // work vector before the current state reassembles it below.
                // A geometry key alone is deliberately insufficient: exact
                // load equality also catches state/parameter dependencies that
                // an operator may carry despite unchanged cut geometry.
                workspace.pressure_representability_work->copyFrom(
                    *workspace.pressure_representability_load);
            }

            std::array<double, 3> norms{};
            for (std::size_t i = 0; i < ops.size(); ++i) {
                // Assemble the surface-energy load directly into its
                // pair-owned RHS.  Production-layout and pair-layout vectors
                // may have different ghost sets or FSILS shared orderings, so
                // copyFrom/local-span positional transfer is not a valid map.
                auto& diagnostic_vector =
                    i == 1u && representability_storage_usable
                        ? *workspace.pressure_representability_load
                        : residual_scratch;
                diagnostic_vector.zero();
                auto scratch_view = diagnostic_vector.createAssemblyView();
                FE_CHECK_NOT_NULL(
                    scratch_view.get(),
                    "NewtonSolver: free-surface conservative balance diagnostic view");
                systems::AssemblyRequest req;
                req.op = ops[i];
                req.want_vector = true;
                req.suppress_constraint_inhomogeneity = true;
                req.suppress_auxiliary_coupling_assembly = true;
                req.is_nonlinear_iteration = true;
                const auto ar =
                    transient.assemble(req, state, nullptr, scratch_view.get());
                logNewtonAssemblyDiagnostic(
                    "free_surface_conservative_balance",
                    sync_point,
                    req,
                    ar);
                FE_THROW_IF(
                    !ar.success,
                    FEException,
                    "NewtonSolver: free-surface conservative balance diagnostic assembly failed for op '" +
                        std::string(ops[i]) + "': " + ar.error_message);

                // Match the constrained residual space used by Newton.  The
                // diagnostic operators contain velocity-test rows only; the
                // vector norm therefore is exactly their global constrained
                // virtual-work coefficient norm.
                zeroVectorEntries(constrained_dofs, diagnostic_vector);
                norms[i] = diagnostic_vector.norm();
                if (i == 1u && representability_storage_usable) {
                    diagnostic_vector.updateGhosts();
                }
            }

            bool pressure_representability_available = false;
            std::string pressure_representability_reason;
            PressureRepresentabilityLsqrResult pressure_representability{};
            std::uint64_t active_pressure_dofs = 0u;
            int pressure_representability_iteration_cap = 0;

            std::optional<FieldId> velocity_field;
            std::optional<FieldId> pressure_field;
            if (!pair_operator_available) {
                pressure_representability_reason =
                    "pair_operator_not_installed_on_all_ranks";
            } else if (!representability_workspace_available) {
                pressure_representability_reason =
                    "workspace_not_allocated_on_all_ranks";
            } else if (!representability_layout_compatible) {
                pressure_representability_reason =
                    "pair_matrix_vector_layout_mismatch";
            } else {
                const auto& load_definition =
                    transient.system().operatorDefinition(ops[1]);
                std::set<FieldId> load_test_fields;
                const auto collect_test_fields =
                    [&load_test_fields](const auto& terms) {
                        for (const auto& term : terms) {
                            if (term.test_field != INVALID_FIELD_ID) {
                                load_test_fields.insert(term.test_field);
                            }
                        }
                    };
                collect_test_fields(load_definition.cells);
                collect_test_fields(load_definition.boundary);
                collect_test_fields(load_definition.interior);
                collect_test_fields(load_definition.interface_faces);
                collect_test_fields(load_definition.cut_volumes);
                if (load_test_fields.size() == 1u) {
                    velocity_field = *load_test_fields.begin();
                }

                const auto& pair_definition =
                    transient.system().operatorDefinition(pair_op);
                std::set<FieldId> pair_fields;
                const auto collect_pair_fields =
                    [&pair_fields](const auto& terms) {
                        for (const auto& term : terms) {
                            if (term.test_field != INVALID_FIELD_ID) {
                                pair_fields.insert(term.test_field);
                            }
                            if (term.trial_field != INVALID_FIELD_ID) {
                                pair_fields.insert(term.trial_field);
                            }
                        }
                    };
                collect_pair_fields(pair_definition.cells);
                collect_pair_fields(pair_definition.boundary);
                collect_pair_fields(pair_definition.interior);
                collect_pair_fields(pair_definition.interface_faces);
                collect_pair_fields(pair_definition.cut_volumes);
                if (velocity_field.has_value()) {
                    pair_fields.erase(*velocity_field);
                }
                if (pair_fields.size() == 1u) {
                    pressure_field = *pair_fields.begin();
                }

                const bool fields_identified_on_all_ranks = allRanks(
                    velocity_field.has_value() && pressure_field.has_value());
                if (!fields_identified_on_all_ranks) {
                    pressure_representability_reason =
                        "mixed_velocity_pressure_fields_not_identifiable";
                }
            }

            if (pressure_representability_reason.empty()) {
                const auto pressure_begin =
                    transient.system().fieldDofOffset(*pressure_field);
                const auto pressure_count =
                    transient.system()
                        .fieldDofHandler(*pressure_field)
                        .getNumDofs();
                const auto pressure_end = pressure_begin + pressure_count;

                bool local_nonzero_pressure_inhomogeneity = false;
                constraints.forEach(
                    [&](const constraints::AffineConstraints::ConstraintView&
                            line) {
                        if (line.slave_dof >= pressure_begin &&
                            line.slave_dof < pressure_end &&
                            (!std::isfinite(line.inhomogeneity) ||
                             line.inhomogeneity != 0.0)) {
                            local_nonzero_pressure_inhomogeneity = true;
                        }
                    });
                if (anyRank(local_nonzero_pressure_inhomogeneity)) {
                    pressure_representability_reason =
                        "nonzero_pressure_constraint_inhomogeneity";
                }

                const auto& owned = transient.system()
                                        .dofHandler()
                                        .getPartition()
                                        .locallyOwned();
                std::uint64_t local_active_pressure_dofs = 0u;
                for (GlobalIndex dof = pressure_begin; dof < pressure_end;
                     ++dof) {
                    if (owned.contains(dof) &&
                        !std::binary_search(constrained_dofs.begin(),
                                            constrained_dofs.end(),
                                            dof)) {
                        ++local_active_pressure_dofs;
                    }
                }
                active_pressure_dofs = local_active_pressure_dofs;
#if FE_HAS_MPI
                {
                    int initialized = 0;
                    int finalized = 0;
                    MPI_Initialized(&initialized);
                    MPI_Finalized(&finalized);
                    if (initialized != 0 && finalized == 0) {
                        std::uint64_t global_active_pressure_dofs = 0u;
                        MPI_Allreduce(&local_active_pressure_dofs,
                                      &global_active_pressure_dofs,
                                      1,
                                      MPI_UINT64_T,
                                      MPI_SUM,
                                      system_communicator);
                        active_pressure_dofs =
                            global_active_pressure_dofs;
                    }
                }
#endif
                // In exact arithmetic Golub--Kahan LSQR terminates within the
                // pressure-space dimension.  Finite-precision loss of Krylov
                // orthogonality can require more than n iterations, however;
                // Algorithm 583 recommends 4*n when the operator is not known
                // to be well conditioned.  Retain the existing hard ceiling so
                // this diagnostic cannot acquire unbounded solve cost.
                pressure_representability_iteration_cap =
                    static_cast<int>(
                        4u * std::min<std::uint64_t>(
                                 active_pressure_dofs, 250u));
            }

            bool pressure_representability_cache_hit = false;
            bool pressure_representability_cache_load_equality_checked = false;
            bool pressure_representability_cache_load_equal = false;
            double pressure_representability_cache_load_difference_norm =
                std::numeric_limits<double>::quiet_NaN();
            if (pressure_representability_reason.empty() &&
                pressure_representability_cache_candidate) {
                axpy(*workspace.pressure_representability_work,
                     static_cast<Real>(-1.0),
                     *workspace.pressure_representability_load);
                pressure_representability_cache_load_difference_norm =
                    workspace.pressure_representability_work->norm();
                pressure_representability_cache_load_equality_checked = true;
                pressure_representability_cache_load_equal = allRanks(
                    std::isfinite(
                        pressure_representability_cache_load_difference_norm) &&
                    pressure_representability_cache_load_difference_norm ==
                        0.0);
                pressure_representability_cache_hit =
                    pressure_representability_cache_load_equal;
                pressure_representability_cache_rejection_reason =
                    pressure_representability_cache_hit
                        ? "none"
                        : "load_changed_or_nonfinite";
                if (pressure_representability_cache_hit) {
                    pressure_representability =
                        free_surface_pressure_representability_cache.result;
                    pressure_representability_available = true;
                }
            }

            if (pressure_representability_cache_rejection_reason ==
                    "exact_load_check_pending" &&
                !pressure_representability_cache_load_equality_checked) {
                pressure_representability_cache_rejection_reason =
                    "diagnostic_unavailable_before_load_check";
            }

            if (pressure_representability_reason.empty() &&
                !pressure_representability_cache_hit) {
                auto& pair_matrix =
                    *workspace.pressure_representability_pair_matrix;
                pair_matrix.zero();
                auto pair_view = pair_matrix.createAssemblyView();
                FE_CHECK_NOT_NULL(
                    pair_view.get(),
                    "NewtonSolver: pressure-representability pair matrix view");
                systems::AssemblyRequest pair_request;
                pair_request.op = pair_op;
                pair_request.want_matrix = true;
                pair_request.want_vector = false;
                pair_request.suppress_constraint_inhomogeneity = true;
                pair_request.suppress_auxiliary_coupling_assembly = true;
                pair_request.is_nonlinear_iteration = true;
                const auto pair_assembly = transient.assemble(
                    pair_request, state, pair_view.get(), nullptr);
                logNewtonAssemblyDiagnostic(
                    "free_surface_conservative_balance",
                    sync_point,
                    pair_request,
                    pair_assembly);
                FE_THROW_IF(
                    !pair_assembly.success,
                    FEException,
                    "NewtonSolver: pressure-representability pair assembly failed for op '" +
                        pair_op + "': " + pair_assembly.error_message);

                // Affine substitution has already moved slave-column and
                // slave-test effects onto masters.  The only remaining slave
                // matrix entries are artificial identity diagonals stamped by
                // constrained assembly; remove them instead of allowing LSQR
                // to interpret constraints as physical pressure work.
                auto constrained_pair_view =
                    pair_matrix.createAssemblyView();
                FE_CHECK_NOT_NULL(
                    constrained_pair_view.get(),
                    "NewtonSolver: pressure-representability constrained pair view");
                constrained_pair_view->beginAssemblyPhase();
                constrained_pair_view->zeroRows(
                    constrained_dofs, /*set_diagonal=*/false);
                constrained_pair_view->finalizeAssembly();
                zeroVectorEntries(
                    constrained_dofs,
                    *workspace.pressure_representability_load);
                workspace.pressure_representability_load->updateGhosts();

                pressure_representability =
                    solvePressureRepresentabilityLsqr(
                        pair_matrix,
                        *workspace.pressure_representability_load,
                        *workspace.pressure_representability_solution,
                        *workspace.pressure_representability_left_basis,
                        *workspace.pressure_representability_right_basis,
                        *workspace.pressure_representability_direction,
                        *workspace.pressure_representability_work,
                        *workspace.pressure_representability_residual,
                        *workspace
                             .pressure_representability_normal_residual,
                        pressure_representability_iteration_cap,
                        allRanks);
                pressure_representability_available = true;
                if (geometry_key.revision_tracking_available &&
                    pressure_pair_matrix_state_independent) {
                    free_surface_pressure_representability_cache.valid =
                        true;
                    free_surface_pressure_representability_cache.geometry =
                        geometry_key;
                    free_surface_pressure_representability_cache.result =
                        pressure_representability;
                }
            }

            const double denominator = norms[0] + norms[1];
            const bool normalization_available =
                std::isfinite(denominator) && denominator > 0.0;
            const double normalized_imbalance =
                normalization_available
                    ? norms[2] / denominator
                    : std::numeric_limits<double>::quiet_NaN();
            const bool alignment_available =
                std::isfinite(norms[0]) && std::isfinite(norms[1]) &&
                std::isfinite(norms[2]) && norms[0] > 0.0 && norms[1] > 0.0;
            double alignment_cosine = std::numeric_limits<double>::quiet_NaN();
            if (alignment_available) {
                alignment_cosine =
                    (norms[2] * norms[2] - norms[0] * norms[0] -
                     norms[1] * norms[1]) /
                    (2.0 * norms[0] * norms[1]);
                // Roundoff can place a mathematically valid cosine just
                // outside the closed interval.
                alignment_cosine = std::clamp(alignment_cosine, -1.0, 1.0);
            }
            const double magnitude_mismatch =
                normalization_available
                    ? std::abs(norms[0] - norms[1]) / denominator
                    : std::numeric_limits<double>::quiet_NaN();

            if (communicatorRank(system_communicator) == 0) {
                std::ostringstream oss;
                oss << std::setprecision(17)
                    << "NewtonSolver: free-surface conservative balance"
                    << " diagnostic=free_surface_conservative_balance"
                    << " available=1"
                    << " rank=" << communicatorRank(system_communicator)
                    << " iteration=" << current_newton_iteration
                    << " phase='"
                    << (phase != nullptr ? phase : "unknown") << "'"
                    << " sync_point=" << stateSyncPointName(sync_point)
                    << " pressure_virtual_work_norm=" << norms[0]
                    << " surface_energy_virtual_work_norm=" << norms[1]
                    << " conservative_balance_norm=" << norms[2]
                    << " normalization=pressure_plus_surface_energy_norms"
                    << " normalized_imbalance=" << normalized_imbalance
                    << " magnitude_mismatch=" << magnitude_mismatch
                    << " alignment_cosine=" << alignment_cosine
                    << " pressure_representability_cache_revision_key_available="
                    << (geometry_key.revision_tracking_available ? 1 : 0)
                    << " pressure_representability_pair_every_assembly="
                    << (every_assembly_on_any_rank ? 1 : 0)
                    << " pressure_representability_available="
                    << (pressure_representability_available ? 1 : 0)
                    << " pressure_representability_cache_hit="
                    << (pressure_representability_cache_hit ? 1 : 0)
                    << " pressure_representability_pair_matrix_state_independent="
                    << (pressure_pair_matrix_state_independent ? 1 : 0)
                    << " pressure_representability_cache_prior_sample="
                    << (local_pressure_representability_cache_prior_sample
                            ? 1
                            : 0)
                    << " pressure_representability_cache_geometry_match="
                    << (local_pressure_representability_cache_geometry_match
                            ? 1
                            : 0)
                    << " pressure_representability_cache_candidate="
                    << (pressure_representability_cache_candidate ? 1 : 0)
                    << " pressure_representability_cache_load_equality_checked="
                    << (pressure_representability_cache_load_equality_checked
                            ? 1
                            : 0)
                    << " pressure_representability_cache_load_equal="
                    << (pressure_representability_cache_load_equal ? 1 : 0)
                    << " pressure_representability_cache_load_difference_norm="
                    << pressure_representability_cache_load_difference_norm
                    << " pressure_representability_cache_rejection_reason="
                    << pressure_representability_cache_rejection_reason
                    << " pressure_representability_cache_policy="
                       "geometry_constraints_state_independent_pair_and_exact_load"
                    << " pressure_representability_method=lsqr"
                    << " pressure_representability_convergence=normal_equation_stationarity"
                    << " pressure_representability_distance_gate_applied=0"
                    << " pressure_representability_claimed=0";
                if (pressure_representability_available) {
                    oss << " pressure_representability_residual_norm="
                        << pressure_representability.residual_norm
                        << " pressure_representability_relative_residual="
                        << pressure_representability.relative_residual
                        << " pressure_representability_normal_residual_norm="
                        << pressure_representability.normal_residual_norm
                        << " pressure_representability_relative_normal_residual="
                        << pressure_representability.relative_normal_residual
                        << " pressure_representability_pressure_norm="
                        << pressure_representability.pressure_norm
                        << " pressure_representability_iterations="
                        << pressure_representability.iterations
                        << " pressure_representability_converged="
                        << (pressure_representability.converged ? 1 : 0)
                        << " pressure_representability_breakdown="
                        << (pressure_representability.breakdown ? 1 : 0)
                        << " pressure_representability_active_pressure_dofs="
                        << active_pressure_dofs
                        << " pressure_representability_iteration_cap="
                        << pressure_representability_iteration_cap;
                } else {
                    oss << " pressure_representability_reason="
                        << pressure_representability_reason;
                }
                oss << " pressure_representability_norm=constrained_reduced_coefficient_l2"
                    << " pressure_representability_load=surface_area_variation_plus_young_wall_energy"
                    << " scope=pressure_and_surface_energy_first_variations_only"
                    << " contract=instantaneous_constrained_velocity_test_virtual_work"
                    << " excludes=line_friction_and_wetted_wall_navier_dissipation"
                    << " total_momentum_equilibrium_claimed=0"
                    << " discrete_energy_theorem_claimed=0";
                FE_LOG_INFO(oss.str());
            }

        };

    auto assembleResidualOnly = [&](const systems::SystemStateView& state,
                                    const char* phase,
                                    StateSyncPoint sync_point = StateSyncPoint::ResidualAssembly) -> double {
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

        synchronizeState(state, sync_point);
        // Residual synchronization may rebuild state-dependent constraint
        // sparsity.  Pressure-row matrix diagnostics executed below must use
        // a scratch matrix allocated from that refreshed pattern, even though
        // the primary assembly in this function is vector-only.
        maybeReallocateJacobianForSparsity(transient.system(), workspace);
        transient.system().beginTimeStep(/*reset_auxiliary_state=*/false,
                                         /*invalidate_auxiliary_inputs=*/false);
        systems::AssemblyRequest req;
        req.op = options_.residual_op;
        req.want_vector = true;
        req.suppress_constraint_inhomogeneity = true;
        req.is_nonlinear_iteration = true;
        const auto ar = transient.assemble(req, state, nullptr, r_view.get());
        logNewtonAssemblyDiagnostic(
            phase != nullptr ? phase : "residual", sync_point, req, ar);
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

        traceResidualDebugState("residual_pre_constraints");
        applyResidualAdditionAndConstraints();
        traceResidualDebugState("residual_post_constraints");
        logNewtonFieldResidualDiagnostic(
            transient.system(),
            r,
            phase != nullptr ? std::string_view(phase)
                             : std::string_view("residual"),
            sync_point,
            current_newton_iteration,
            solve_time,
            base_state.dt);
        assemblePressureRowContributionDiagnostics(state, phase, sync_point);
        assembleFreeSurfaceConservativeBalanceDiagnostic(state, phase, sync_point);
        traceResidualComponents(phase);
        return refreshResidualComponents();
    };

    auto assembleJacobianOnly = [&](const systems::SystemStateView& state) {
        if (oopTraceEnabled()) {
            traceLog("NewtonSolver: beginTimeStep() + assemble (matrix) op='" + options_.jacobian_op + "'");
        }
        synchronizeState(state, StateSyncPoint::JacobianAssembly);
        maybeReallocateJacobianForSparsity(transient.system(), workspace);
        auto J_view = J.createAssemblyView();
        FE_CHECK_NOT_NULL(J_view.get(), "NewtonSolver: jacobian assembly view");
        transient.system().beginTimeStep(/*reset_auxiliary_state=*/false,
                                         /*invalidate_auxiliary_inputs=*/false);
        systems::AssemblyRequest req;
        req.op = options_.jacobian_op;
        req.want_matrix = true;
        req.is_nonlinear_iteration = true;
        const auto aj = transient.assemble(req, state, J_view.get(), nullptr);
        logNewtonAssemblyDiagnostic(
            "jacobian", StateSyncPoint::JacobianAssembly, req, aj);
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
        if (oopTraceEnabled()) {
            traceLog("NewtonSolver: beginTimeStep() + assemble (matrix+vector) op='" + options_.residual_op + "'");
        }
        synchronizeState(state, StateSyncPoint::JacobianAndResidualAssembly);
        maybeReallocateJacobianForSparsity(transient.system(), workspace);
        auto J_view = J.createAssemblyView();
        auto r_view = r.createAssemblyView();
        FE_CHECK_NOT_NULL(J_view.get(), "NewtonSolver: jacobian assembly view");
        FE_CHECK_NOT_NULL(r_view.get(), "NewtonSolver: residual assembly view");
        transient.system().beginTimeStep(/*reset_auxiliary_state=*/false,
                                         /*invalidate_auxiliary_inputs=*/false);
        systems::AssemblyRequest req;
        req.op = options_.residual_op;
        req.want_matrix = true;
        req.want_vector = true;
        req.suppress_constraint_inhomogeneity = true;
        req.is_nonlinear_iteration = true;
        const auto ar = transient.assemble(req, state, J_view.get(), r_view.get());
        logNewtonAssemblyDiagnostic(
            "jacobian_and_residual",
            StateSyncPoint::JacobianAndResidualAssembly,
            req,
            ar);
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

        traceResidualDebugState("jacobian_and_residual_pre_constraints");
        applyResidualAdditionAndConstraints();
        traceResidualDebugState("jacobian_and_residual_post_constraints");
        logNewtonFieldResidualDiagnostic(
            transient.system(),
            r,
            std::string_view("jacobian_and_residual"),
            StateSyncPoint::JacobianAndResidualAssembly,
            current_newton_iteration,
            solve_time,
            base_state.dt);
        assemblePressureRowContributionDiagnostics(
            state,
            "jacobian_and_residual",
            StateSyncPoint::JacobianAndResidualAssembly);
        assembleFreeSurfaceConservativeBalanceDiagnostic(
            state,
            "jacobian_and_residual",
            StateSyncPoint::JacobianAndResidualAssembly);
        traceResidualComponents("jacobian_and_residual");
        return refreshResidualComponents();
    };

    auto assembleJacobianAndResidualWithJacobianOp = [&](const systems::SystemStateView& state,
                                                         bool& out_vector_ok) -> double {
        out_vector_ok = false;

        if (oopTraceEnabled()) {
            traceLog("NewtonSolver: beginTimeStep() + assemble (matrix+vector) op='" + options_.jacobian_op + "'");
        }
        synchronizeState(state, StateSyncPoint::JacobianAndResidualAssembly);
        maybeReallocateJacobianForSparsity(transient.system(), workspace);
        auto J_view = J.createAssemblyView();
        auto r_view = r.createAssemblyView();
        FE_CHECK_NOT_NULL(J_view.get(), "NewtonSolver: jacobian assembly view");
        FE_CHECK_NOT_NULL(r_view.get(), "NewtonSolver: residual assembly view");
        transient.system().beginTimeStep(/*reset_auxiliary_state=*/false,
                                         /*invalidate_auxiliary_inputs=*/false);
        systems::AssemblyRequest req;
        req.op = options_.jacobian_op;
        req.want_matrix = true;
        req.want_vector = true;
        req.suppress_constraint_inhomogeneity = true;
        req.is_nonlinear_iteration = true;
        const auto ar = transient.assemble(req, state, J_view.get(), r_view.get());
        logNewtonAssemblyDiagnostic(
            "jacobian_op_combined",
            StateSyncPoint::JacobianAndResidualAssembly,
            req,
            ar);
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

        // Assembly counters are rank-local; an empty-owner rank must follow
        // the same combined-assembly fallback decision as its peers.
        out_vector_ok = anyRank(ar.vector_entries_inserted > 0);
        if (!out_vector_ok) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        residual_op_used = options_.jacobian_op;
        applyResidualAdditionAndConstraints();
        logNewtonFieldResidualDiagnostic(
            transient.system(),
            r,
            std::string_view("jacobian_op_combined"),
            StateSyncPoint::JacobianAndResidualAssembly,
            current_newton_iteration,
            solve_time,
            base_state.dt);
        assemblePressureRowContributionDiagnostics(
            state,
            "jacobian_op_combined",
            StateSyncPoint::JacobianAndResidualAssembly);
        assembleFreeSurfaceConservativeBalanceDiagnostic(
            state,
            "jacobian_op_combined",
            StateSyncPoint::JacobianAndResidualAssembly);
        return refreshResidualComponents();
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
            !any_non_dirichlet_affine_constraints;
        const bool force_explicit_matrix_assembly =
            linear_has_live_bordered && !use_native_rank_one_updates;
        if (oopTraceEnabled()) {
            traceLog("NewtonSolver: rank-1 updates=" + std::to_string(rank_one_updates.size()) +
                     " reduced updates=" + std::to_string(reduced_updates.size()) +
                     (force_explicit_matrix_assembly ? " (explicit matrix path)" : "")
                     + (any_non_dirichlet_affine_constraints
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

    auto scalarToleranceSatisfied = [](double norm,
                                       double norm0,
                                       double abs_tolerance,
                                       double rel_tolerance,
                                       bool pre_first_update) -> bool {
        if (!std::isfinite(norm)) {
            return false;
        }
        const bool abs_enabled = abs_tolerance > 0.0;
        const bool rel_enabled = rel_tolerance > 0.0;
        const bool abs_ok = abs_enabled && norm <= abs_tolerance;
        const bool rel_ok = rel_enabled
            && (norm0 > 0.0 && std::isfinite(norm0)
                    ? (norm / norm0 <= rel_tolerance)
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

    auto globalToleranceSatisfied = [&](double norm,
                                        double norm0,
                                        bool pre_first_update) -> bool {
        return scalarToleranceSatisfied(norm,
                                         norm0,
                                         options_.abs_tolerance,
                                         options_.rel_tolerance,
                                         pre_first_update);
    };

    auto componentToleranceSatisfied = [&](double norm, double norm0, bool pre_first_update) -> bool {
        if (norm == 0.0 && norm0 == 0.0) {
            return true;
        }
        return globalToleranceSatisfied(norm, norm0, pre_first_update);
    };

    auto tolerancesSatisfied = [&](bool pre_first_update) -> bool {
        bool monolithic_satisfied = false;
        if (!componentResidualConvergenceActive()) {
            monolithic_satisfied = globalToleranceSatisfied(
                current_residual_norm,
                report.residual_norm0,
                pre_first_update);
        } else {
            const bool field_ok = componentToleranceSatisfied(
                current_residual_components.field,
                initial_residual_components.field,
                pre_first_update);
            const bool aux_ok = componentToleranceSatisfied(
                current_residual_components.auxiliary,
                initial_residual_components.auxiliary,
                pre_first_update);
            if (oopTraceEnabled()) {
                std::ostringstream oss;
                const double field_rel =
                    (initial_residual_components.field > 0.0 && std::isfinite(initial_residual_components.field))
                        ? current_residual_components.field / initial_residual_components.field
                        : std::numeric_limits<double>::quiet_NaN();
                const double aux_rel =
                    (initial_residual_components.auxiliary > 0.0 && std::isfinite(initial_residual_components.auxiliary))
                        ? current_residual_components.auxiliary / initial_residual_components.auxiliary
                        : std::numeric_limits<double>::quiet_NaN();
                oss << "NewtonSolver: component convergence"
                    << " field_ok=" << (field_ok ? 1 : 0)
                    << " aux_ok=" << (aux_ok ? 1 : 0)
                    << " field_rel=" << field_rel
                    << " aux_rel=" << aux_rel;
                traceLog(oss.str());
            }
            monolithic_satisfied = field_ok && aux_ok;
        }

        bool configured_fields_satisfied = true;
        for (auto& state : field_residual_states) {
            const auto& criterion = state.criterion;
            // A coupled block can be exactly inactive in the initial residual
            // and become active only after another field moves or an accepted
            // state refresh changes residual-defining geometry.  Establish a
            // relative reference only when this accepted/current residual is
            // actually tested for convergence; rejected line-search trials
            // must not define the reference.  The activation sample itself is
            // never accepted by the relative criterion, so a later residual
            // evaluation must demonstrate contraction.
            if (criterion.rel_tolerance > 0.0 &&
                !state.relative_reference_available &&
                state.current_norm > 0.0 &&
                std::isfinite(state.current_norm)) {
                state.initial_norm = state.current_norm;
                state.relative_reference_available = true;
                state.relative_reference_activated_this_sample = true;
            }
            const bool zero_residual =
                state.current_norm == 0.0 && state.initial_norm == 0.0;
            bool satisfied = zero_residual || scalarToleranceSatisfied(
                state.current_norm,
                state.initial_norm,
                criterion.abs_tolerance,
                criterion.rel_tolerance,
                pre_first_update);
            const bool abs_ok = criterion.abs_tolerance > 0.0 &&
                                state.current_norm <= criterion.abs_tolerance;
            if (state.relative_reference_activated_this_sample && !abs_ok) {
                satisfied = false;
            }
            configured_fields_satisfied =
                configured_fields_satisfied && satisfied;

            if (activeSystemRank(sys) == 0) {
                const bool rel_ok = criterion.rel_tolerance > 0.0 &&
                                    (state.initial_norm > 0.0 &&
                                             std::isfinite(state.initial_norm)
                                         ? state.current_norm /
                                                   state.initial_norm <=
                                               criterion.rel_tolerance
                                         : abs_ok);
                const double relative_norm =
                    state.initial_norm > 0.0 && std::isfinite(state.initial_norm)
                        ? state.current_norm / state.initial_norm
                        : std::numeric_limits<double>::quiet_NaN();
                std::ostringstream oss;
                oss << "NewtonSolver: field convergence"
                    << " diagnostic=newton_field_convergence"
                    << " field_id=" << criterion.field
                    << " field='" << state.name << "'"
                    << " owned_dofs=" << state.owned_dof_count
                    << " norm0=" << state.initial_norm
                    << " norm=" << state.current_norm
                    << " relative_norm=" << relative_norm
                    << " abs_tolerance=" << criterion.abs_tolerance
                    << " rel_tolerance=" << criterion.rel_tolerance
                    << " abs_ok=" << (abs_ok ? 1 : 0)
                    << " rel_ok=" << (rel_ok ? 1 : 0)
                    << " relative_reference_activated="
                    << (state.relative_reference_activated_this_sample ? 1 : 0)
                    << " pre_first_update=" << (pre_first_update ? 1 : 0)
                    << " satisfied=" << (satisfied ? 1 : 0);
                FE_LOG_INFO(oss.str());
            }
        }
        return monolithic_satisfied && configured_fields_satisfied;
    };

    auto minIterationsSatisfied = [&](int completed_iterations) -> bool {
        return completed_iterations >= min_it;
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

        synchronizeState(state_dt, StateSyncPoint::JacobianAssembly);
        maybeReallocateJacobianForSparsity(transient.system(), workspace);
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
        const int mpi_rank = communicatorRank(system_communicator);
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
    bool tangent_analysis_report_logged = false;
    for (int it = 0; it < max_it; ++it) {
        current_newton_iteration = it;
        ntp0 = NTP();
        syncHistoryState();
        ntp_constraints += NTP() - ntp0;

        auto state_holder = makeNewtonState(history, solve_time);
        const auto& state = state_holder.view;

        if (have_residual && !std::isfinite(current_residual_norm)) {
            // If the cached residual norm is invalid (e.g., NaN from a failed evaluation),
            // fall back to re-assembling the residual at the current state.
            have_residual = false;
        }

        const int jacobian_period = std::max(base_jacobian_period, direct_only_outlet_jacobian_period);
        const bool can_reuse_state_independent_jacobian =
            base_can_reuse_state_independent_jacobian &&
            !any_non_dirichlet_affine_constraints;
        bool need_jacobian =
            !have_jacobian ||
            (!can_reuse_state_independent_jacobian &&
             ((jacobian_period == 1) || ((it - last_jacobian_it) >= jacobian_period)));
        bool jacobian_ready = have_jacobian && !need_jacobian;
        const bool residual_first_convergence_check =
            !options_.use_line_search &&
            it > 0 &&
            need_jacobian &&
            options_.assemble_both_when_possible &&
            same_op &&
            !has_monolithic_auxiliary_unknowns;
        if (!have_residual) {
            ntp0 = NTP();
            if (need_jacobian && options_.assemble_both_when_possible && same_op &&
                !residual_first_convergence_check) {
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
                current_residual_norm = assembleResidualOnly(
                    state,
                    residual_first_convergence_check
                        ? "post_update_convergence_check"
                        : nullptr);
                // A residual synchronization can install a different affine
                // constraint set.  Recompute the within-iteration Jacobian
                // decision instead of using the value cached before that
                // callback.
                if (!have_jacobian) {
                    need_jacobian = true;
                    jacobian_ready = false;
                }
                if (need_jacobian && !residual_first_convergence_check) {
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

        updateResidualReport();
        if (it == 0) {
            initial_residual_components = current_residual_components;
            report.residual_norm0 = current_residual_norm;
            report.field_residual_norm0 = initial_residual_components.field;
            report.auxiliary_residual_norm0 = initial_residual_components.auxiliary;
            for (auto& state : field_residual_states) {
                state.initial_norm = state.current_norm;
                state.relative_reference_available =
                    state.current_norm > 0.0 &&
                    std::isfinite(state.current_norm);
                state.relative_reference_activated_this_sample = false;
            }
        }

        if (oopTraceEnabled()) {
            std::ostringstream oss;
            const double denom = (report.residual_norm0 > 0.0) ? report.residual_norm0 : 1.0;
            oss << "NewtonSolver: it=" << it
                << " ||r||=" << report.residual_norm
                << " ||r0||=" << report.residual_norm0
                << " rel=" << (report.residual_norm / denom)
                << " ||r_field||=" << report.field_residual_norm
                << " ||r_aux||=" << report.auxiliary_residual_norm;
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

        if (minIterationsSatisfied(it) &&
            tolerancesSatisfied(/*pre_first_update=*/it == 0)) {
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
            ntp0 = NTP();
            assembleJacobianOnly(state);
            ntp_assembly += NTP() - ntp0;
            ntp_assembly_count++;
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
            const auto difference_scheme = jacobianCheckDifferenceScheme();
            const char* difference_scheme_name =
                jacobianCheckDifferenceSchemeName(difference_scheme);

            if (h > 0.0 && std::isfinite(h)) {
                const auto& component_sweeps = jacobianCheckComponentSweepFilters();
                logJacobianCheckSweepPlan(component_sweeps);

                for (std::size_t sweep_index = 0; sweep_index < component_sweeps.size(); ++sweep_index) {
                    const auto& component_filter = component_sweeps[sweep_index];
                    const auto component_filter_label = jacobianCheckComponentFilterLabel(component_filter);

                    // Populate a deterministic pseudo-random direction in `du` (will be overwritten by the linear solve).
                    fillJacobianCheckDirection(du, sweep_index);
                    zeroVectorEntries(constrained_dofs, du);
                    applyJacobianCheckComponentFilter(transient.system(),
                                                      du,
                                                      component_filter,
                                                      component_filter_label,
                                                      sweep_index);
                    du.updateGhosts();

                auto applyResidualFixups = [&](backends::GenericVector& vec) {
                    if (residual_addition != nullptr) {
                        axpy(vec, static_cast<Real>(1.0), *residual_addition);
                    }
                    zeroVectorEntries(constrained_dofs, vec);
                };

                // A finite-difference residual evaluation is a speculative
                // nonlinear state, just like a rejected line-search alpha.
                // Snapshot every mutable part of that state before invoking
                // user synchronization or assembly callbacks.  In particular,
                // a state-dependent MPC can project all transient history and
                // optional rate vectors while reaching its constraint fixed
                // point; restoring only u cannot undo a topology reversal.
                copyVector(u_backup, history.u());
                FE_CHECK_NOT_NULL(
                    workspace.factory,
                    "NewtonSolver: Jacobian-check history backup factory");
                std::vector<std::unique_ptr<backends::GenericVector>>
                    diagnostic_history_backup(
                        static_cast<std::size_t>(history.historyDepth()));
                for (int k = 1; k <= history.historyDepth(); ++k) {
                    auto& backup = diagnostic_history_backup[
                        static_cast<std::size_t>(k - 1)];
                    backup = workspace.factory->createVector(
                        history.uPrevK(k).size());
                    FE_CHECK_NOT_NULL(
                        backup.get(),
                        "NewtonSolver: Jacobian-check history backup");
                    backup->copyFrom(history.uPrevK(k));
                }
                auto diagnostic_rate_state_backup =
                    history.snapshotRateState(*workspace.factory);
                const auto diagnostic_base_constraint_semantics =
                    constraintSemanticFingerprint(constraints);
                const auto aux_state_backup =
                    transient.system().checkpointAuxiliaryState();
                const auto bordered_backup = transient.system().borderedCoupling();
                const bool diagnostic_base_had_local_condensed_recovery =
                    transient.system().hasLocalCondensedRecovery();
                const bool diagnostic_base_any_local_condensed_recovery =
                    anyRank(diagnostic_base_had_local_condensed_recovery);
                const bool diagnostic_base_have_residual = have_residual;
                const bool diagnostic_base_have_jacobian = have_jacobian;
                const int diagnostic_base_last_jacobian_it = last_jacobian_it;
                const bool diagnostic_base_ptc_mass_ready = ptc_mass_ready;
                const double diagnostic_base_ptc_gamma = ptc_gamma;
                const double diagnostic_base_ptc_gamma_applied =
                    ptc_gamma_applied;
                const double diagnostic_base_ptc_prev_residual_norm =
                    ptc_prev_residual_norm;
                FE_THROW_IF(
                    anyRank(transient.system()
                                .meshCoordinateTransactionActive()),
                    systems::InvalidStateException,
                    "NewtonSolver: Jacobian check started with an active "
                    "mesh-coordinate transaction");

                auto restoreDiagnosticHistoryState = [&]() {
                    FE_THROW_IF(
                        diagnostic_history_backup.size() !=
                            static_cast<std::size_t>(history.historyDepth()),
                        systems::InvalidStateException,
                        "NewtonSolver: Jacobian-check probe changed history depth");
                    for (int k = 1; k <= history.historyDepth(); ++k) {
                        auto& backup = diagnostic_history_backup[
                            static_cast<std::size_t>(k - 1)];
                        FE_CHECK_NOT_NULL(
                            backup.get(),
                            "NewtonSolver: Jacobian-check history restore backup");
                        history.uPrevK(k).copyFrom(*backup);
                    }
                    // Rate snapshots preserve both values and allocation
                    // state.  This restores an originally absent first-step
                    // rate state even if a speculative callback allocated it.
                    history.restoreRateState(diagnostic_rate_state_backup);
                    history.snapshotRateState(
                        diagnostic_rate_state_backup,
                        *workspace.factory);
                };
                auto restoreDiagnosticState = [&]() {
                    // Roll speculative coordinates back before rebuilding
                    // state-derived cuts/support for the accepted algebraic
                    // snapshot.  Force is required because diagnostics must
                    // never inherit a policy that keeps rejected geometry.
                    transient.system().rollbackGeometricNonlinearityTrial(
                        /*force=*/true);
                    FE_THROW_IF(
                        anyRank(transient.system()
                                    .meshCoordinateTransactionActive()),
                        systems::InvalidStateException,
                        "NewtonSolver: Jacobian-check geometry could not be "
                        "rolled back");
                    copyVector(history.u(), u_backup);
                    restoreDiagnosticHistoryState();
                    if (!aux_state_backup.empty()) {
                        transient.system().restoreAuxiliaryState(aux_state_backup);
                    }
                    transient.system().borderedCoupling() = bordered_backup;
                    if (auto* reg = transient.system().auxiliaryInputRegistryIfPresent()) {
                        reg->invalidateAll();
                    }

                    // The rejected probe's constraint set must not touch the
                    // restored vectors.  RestoredNonlinearState first rebuilds
                    // the accepted cut/MPC semantics, after which
                    // synchronizeState projects the exact snapshots with that
                    // set and refreshes any dependent geometry at a bounded
                    // fixed point.
                    history.u().updateGhosts();
                    syncOwnedRowHaloIfNeeded(history.u());
                    auto restored_state_holder =
                        makeNewtonState(history, solve_time);
                    synchronizeState(
                        restored_state_holder.view,
                        StateSyncPoint::RestoredNonlinearState);

                    // A constraint projection during restoration can open a
                    // base-state geometry transaction.  Its coordinates are
                    // identical to the pre-probe accepted coordinates, so
                    // close it by rolling back to the exact coordinate backup
                    // rather than committing from a diagnostic callback.
                    transient.system().rollbackGeometricNonlinearityTrial(
                        /*force=*/true);
                    FE_THROW_IF(
                        anyRank(transient.system()
                                    .meshCoordinateTransactionActive()),
                        systems::InvalidStateException,
                        "NewtonSolver: Jacobian-check base geometry "
                        "transaction remained active after restoration");
                    FE_THROW_IF(
                        anyRank(constraintSemanticFingerprint(constraints) !=
                                diagnostic_base_constraint_semantics),
                        systems::InvalidStateException,
                        "NewtonSolver: Jacobian-check restoration did not "
                        "reproduce the accepted affine-constraint state");

                    // The residual and physical Jacobian objects themselves
                    // were never used as probe output.  Preserve those cache
                    // facts, except when a transient structure revision means
                    // the workspace matrix must be rebuilt before any future
                    // reuse.  The current linear solve can still use the
                    // untouched base-pattern matrix assembled above.
                    have_residual = diagnostic_base_have_residual;
                    const bool base_pattern_still_current = allRanks(
                        transient.system().sparsityPatternRevision() ==
                        workspace.sparsity_revision);
                    have_jacobian = diagnostic_base_have_jacobian &&
                                    base_pattern_still_current;
                    last_jacobian_it = have_jacobian
                                           ? diagnostic_base_last_jacobian_it
                                           : -1;
                    ptc_mass_ready = diagnostic_base_ptc_mass_ready;
                    ptc_gamma = diagnostic_base_ptc_gamma;
                    ptc_gamma_applied = diagnostic_base_ptc_gamma_applied;
                    ptc_prev_residual_norm =
                        diagnostic_base_ptc_prev_residual_norm;
                    if (!diagnostic_base_had_local_condensed_recovery) {
                        transient.system().clearLocalCondensedRecovery();
                    }
                };

                auto evaluateDiagnosticProbeTransactionally =
                    [&](auto&& evaluate_probe) {
                        try {
                            evaluate_probe();
                        } catch (...) {
                            const auto probe_failure =
                                std::current_exception();
                            try {
                                restoreDiagnosticState();
                            } catch (...) {
                                std::throw_with_nested(
                                    systems::InvalidStateException(
                                        "NewtonSolver: Jacobian-check probe "
                                        "failed and exact nonlinear-state "
                                        "restoration also failed"));
                            }
                            std::rethrow_exception(probe_failure);
                        }
                        try {
                            restoreDiagnosticState();
                        } catch (...) {
                            std::throw_with_nested(
                                systems::InvalidStateException(
                                    "NewtonSolver: Jacobian-check probe "
                                    "completed but exact nonlinear-state "
                                    "restoration failed"));
                        }
                    };

                auto base_component_norms = zeroComponentNormSnapshot(transient.system());
                auto perturbed_component_norms = base_component_norms;
                auto fd_component_norms = base_component_norms;
                auto matrix_component_norms = base_component_norms;
                auto full_component_norms = base_component_norms;
                auto matrix_err_component_norms = base_component_norms;
                auto err_component_norms = base_component_norms;
                auto sign_flip_err_component_norms = base_component_norms;

                auto assembleDiagnosticBaseResidual = [&]() {
                    residual_base.zero();
                    auto r_view = residual_base.createAssemblyView();
                    FE_CHECK_NOT_NULL(
                        r_view.get(),
                        "NewtonSolver: jacobian check residual base view");

                    auto diagnostic_state_holder =
                        makeNewtonState(history, solve_time);
                    synchronizeState(
                        diagnostic_state_holder.view,
                        StateSyncPoint::ResidualAssembly);
                    transient.system().beginTimeStep(
                        /*reset_auxiliary_state=*/false,
                        /*invalidate_auxiliary_inputs=*/false);
                    systems::AssemblyRequest req;
                    req.op = options_.residual_op;
                    req.want_vector = true;
                    req.suppress_constraint_inhomogeneity = true;
                    req.is_nonlinear_iteration = true;
                    const auto ar = transient.assemble(
                        req,
                        diagnostic_state_holder.view,
                        nullptr,
                        r_view.get());
                    FE_THROW_IF(
                        !ar.success,
                        FEException,
                        "NewtonSolver: Jacobian check base residual "
                        "assembly failed: " +
                            ar.error_message);
                    applyResidualFixups(residual_base);
                    base_component_norms = componentNormSnapshot(
                        transient.system(), residual_base);
                };

                // Assemble r(u) with residual_op into residual_base.
                evaluateDiagnosticProbeTransactionally(
                    assembleDiagnosticBaseResidual);

                // Assemble r(u + h*v) with residual_op into residual_scratch.
                evaluateDiagnosticProbeTransactionally([&]() {
                    axpy(history.u(), static_cast<Real>(h), du);
                    syncCurrentState();
                    residual_scratch.zero();
                    auto r_view = residual_scratch.createAssemblyView();
                    FE_CHECK_NOT_NULL(
                        r_view.get(),
                        "NewtonSolver: jacobian check residual perturbed view");

                    transient.system().beginTimeStep(
                        /*reset_auxiliary_state=*/false,
                        /*invalidate_auxiliary_inputs=*/false);
                    systems::AssemblyRequest req;
                    req.op = options_.residual_op;
                    req.want_vector = true;
                    req.suppress_constraint_inhomogeneity = true;
                    req.is_nonlinear_iteration = true;
                    auto perturbed_state_holder = makeNewtonState(history, solve_time);
                    synchronizeState(perturbed_state_holder.view,
                                     StateSyncPoint::ResidualAssembly);
                    const auto ar = transient.assemble(
                        req, perturbed_state_holder.view, nullptr, r_view.get());
                    FE_THROW_IF(
                        !ar.success,
                        FEException,
                        "NewtonSolver: Jacobian check perturbed residual "
                        "assembly failed: " +
                            ar.error_message);
                    applyResidualFixups(residual_scratch);
                    perturbed_component_norms = componentNormSnapshot(
                        transient.system(), residual_scratch);
                });

                double fd_curvature_norm = 0.0;
                if (difference_scheme == JacobianCheckDifferenceScheme::Central) {
                    // Assemble r(u - h*v) into residual_minus. A centered
                    // check cancels step-independent refresh jumps and gives a
                    // more reliable smooth-tangent diagnostic.
                    evaluateDiagnosticProbeTransactionally([&]() {
                        axpy(history.u(), static_cast<Real>(-h), du);
                        syncCurrentState();
                        residual_minus.zero();
                        auto r_view = residual_minus.createAssemblyView();
                        FE_CHECK_NOT_NULL(
                            r_view.get(),
                            "NewtonSolver: jacobian check residual minus view");

                        transient.system().beginTimeStep(
                            /*reset_auxiliary_state=*/false,
                            /*invalidate_auxiliary_inputs=*/false);
                        systems::AssemblyRequest req;
                        req.op = options_.residual_op;
                        req.want_vector = true;
                        req.suppress_constraint_inhomogeneity = true;
                        req.is_nonlinear_iteration = true;
                        auto minus_state_holder = makeNewtonState(history, solve_time);
                        synchronizeState(minus_state_holder.view,
                                         StateSyncPoint::ResidualAssembly);
                        const auto ar = transient.assemble(
                            req, minus_state_holder.view, nullptr, r_view.get());
                        FE_THROW_IF(
                            !ar.success,
                            FEException,
                            "NewtonSolver: Jacobian check minus residual "
                            "assembly failed: " +
                                ar.error_message);
                        applyResidualFixups(residual_minus);
                    });

                    // Residual-only assembly refreshes the right-hand side of
                    // locally condensed auxiliary recovery records.  Leave
                    // those system-side caches describing the accepted base,
                    // not the final finite-difference perturbation.
                    if (diagnostic_base_any_local_condensed_recovery) {
                        evaluateDiagnosticProbeTransactionally(
                            assembleDiagnosticBaseResidual);
                    }

                    copyVector(u_backup, residual_minus);
                    axpy(u_backup, static_cast<Real>(1.0), residual_scratch);
                    axpy(u_backup, static_cast<Real>(-2.0), residual_base);
                    zeroVectorEntries(constrained_dofs, u_backup);
                    fd_curvature_norm = u_backup.norm();

                    residual_scratch.scale(static_cast<Real>(1.0 / (2.0 * h)));
                    axpy(residual_scratch,
                         static_cast<Real>(-1.0 / (2.0 * h)),
                         residual_minus);
                } else {
                    if (diagnostic_base_any_local_condensed_recovery) {
                        evaluateDiagnosticProbeTransactionally(
                            assembleDiagnosticBaseResidual);
                    }
                    // residual_scratch <- (r(u+h*v) - r(u)) / h.
                    axpy(residual_scratch, static_cast<Real>(-1.0), residual_base);
                    residual_scratch.scale(static_cast<Real>(1.0 / h));
                }
                syncOwnedRowHaloIfNeeded(residual_scratch);
                fd_component_norms = componentNormSnapshot(transient.system(), residual_scratch);
                const double r_base_norm =
                    residualNormForConvergence(residual_base, residual_minus);
                const double r_used_norm =
                    residualNormForConvergence(r, residual_minus);

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
                matrix_component_norms = componentNormSnapshot(transient.system(), u_backup);

                // residual_base keeps a copy of the matrix-only action so we can
                // compare both the raw assembled matrix and the full effective
                // operator (matrix + pending rank-1 updates) against FD.
                copyVector(residual_base, u_backup);
                axpy(u_backup, static_cast<Real>(-1.0), residual_scratch);
                const double matrix_err_norm = u_backup.norm();
                matrix_err_component_norms =
                    componentNormSnapshot(transient.system(), u_backup);

                copyVector(u_backup, residual_base);
                if (!effective_rank_one_updates.empty()) {
                    addRankOneOperatorMatvec(
                        std::span<const backends::RankOneUpdate>(effective_rank_one_updates.data(),
                                                                 effective_rank_one_updates.size()),
                        du,
                        u_backup,
                        system_communicator);
                }
                if (!active_reduced_field_updates.empty()) {
                    addReducedFieldOperatorMatvec(
                        std::span<const backends::ReducedFieldUpdate>(
                            active_reduced_field_updates.data(),
                            active_reduced_field_updates.size()),
                        du,
                        u_backup,
                        system_communicator);
                }
                zeroVectorEntries(constrained_dofs, u_backup);
                const double jv_norm = u_backup.norm();
                full_component_norms = componentNormSnapshot(transient.system(), u_backup);

                // residual_base <- -(rank-one contribution)
                axpy(residual_base, static_cast<Real>(-1.0), u_backup);
                const double rank_one_jv_norm = residual_base.norm();

                const double fd_norm = residual_scratch.norm();

                // Compare both signs so sign-convention errors are visible in
                // one diagnostic record.
                axpy(u_backup, static_cast<Real>(1.0), residual_scratch);
                const double sign_flip_err_norm = u_backup.norm();
                sign_flip_err_component_norms =
                    componentNormSnapshot(transient.system(), u_backup);
                axpy(u_backup, static_cast<Real>(-2.0), residual_scratch);
                const double err_norm = u_backup.norm();
                err_component_norms = componentNormSnapshot(transient.system(), u_backup);
                const double denom = std::max({jv_norm, fd_norm, 1e-14});
                const double rel_err = err_norm / denom;
                const char* geometry_result = jacobianCheckGeometryResult(
                    options_.jacobian_check_geometry_mode,
                    rel_err,
                    options_.jacobian_check_relative_tolerance);
                if (options_.jacobian_check_diagnostic) {
                    NewtonJacobianCheckDiagnostic diag;
                    diag.iteration = it;
                    diag.sweep_index = sweep_index;
                    diag.step_size = h;
                    diag.matrix_action_norm = matrix_jv_norm;
                    diag.full_action_norm = jv_norm;
                    diag.finite_difference_norm = fd_norm;
                    diag.error_norm = err_norm;
                    diag.relative_error = rel_err;
                    diag.geometry_mode = options_.jacobian_check_geometry_mode;
                    diag.geometry_tangent_policy =
                        options_.jacobian_check_geometry_tangent_policy;
                    diag.geometry_result = geometry_result;
                    diag.component_filter = component_filter_label;
                    diag.finite_difference_scheme = difference_scheme_name;
                    options_.jacobian_check_diagnostic(diag);
                }

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
	                        << " diagnostic=jacobian_check"
	                        << " fd_scheme=" << difference_scheme_name
                            << " geometry_check_mode="
                            << jacobianCheckGeometryModeName(
                                   options_.jacobian_check_geometry_mode)
                            << " geometry_tangent_policy='"
                            << options_.jacobian_check_geometry_tangent_policy << "'"
                            << " geometry_result=" << geometry_result
	                        << " component_filter='" << component_filter_label << "'"
	                        << " sweep=" << sweep_index
	                        << " it=" << it
	                        << " h=" << h
                        << " ||J_matrix*v||=" << matrix_jv_norm
                        << " ||rank1*v||=" << rank_one_jv_norm
                        << " ||Jv||=" << jv_norm
                        << " ||FD||=" << fd_norm
                        << " ||J_matrix*v-FD||=" << matrix_err_norm
                        << " ||Jv-FD||=" << err_norm
                        << " ||Jv+FD||=" << sign_flip_err_norm
                        << " ||FD_curvature||=" << fd_curvature_norm
                        << " rel=" << rel_err
                        << " ||r(residual_op)||=" << r_base_norm
                        << " ||r(used_op=" << residual_op_used << ")||=" << r_used_norm
                        << " ||r_used-r_residual||=" << r_diff_norm;
                    FE_LOG_INFO(oss.str());
                }
	                logJacobianCheckComponentDetails(system_communicator,
	                                                 component_filter_label,
	                                                 sweep_index,
	                                                 base_component_norms,
	                                                 perturbed_component_norms,
	                                                 fd_component_norms,
	                                                 matrix_component_norms,
	                                                 full_component_norms,
	                                                 matrix_err_component_norms,
	                                                 err_component_norms,
	                                                 sign_flip_err_component_norms);
	                logJacobianCheckComponentBreakdown(transient.system(),
	                                                  residual_scratch,
	                                                  u_backup,
	                                                  residual_base,
	                                                  component_filter_label,
	                                                  sweep_index);
	                logJacobianCheckTopMismatchEntries(transient.system(),
	                                                   residual_scratch,
	                                                   u_backup,
	                                                   8u,
	                                                   component_filter_label,
	                                                   sweep_index);
                }
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
        const auto owned_dofs =
            ownedDofSetForVector(du, transient.system().dofHandler().getPartition().locallyOwned());
        active_reduced_field_updates = effective_reduced_field_updates;
        grouped_bordered_field_couplings.clear();
        algebraic_aux_reduction = {};
        solve_bordered_ptr = has_bordered ? &bordered_full : nullptr;
        if (has_bordered &&
            buildAlgebraicAuxiliaryReduction(
                bordered_full,
                owned_dofs,
                algebraic_aux_reduction,
                system_communicator)) {
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
        if (oopTraceEnabled() && has_solve_bordered) {
            const auto& solve_bordered = *solve_bordered_ptr;
            const auto l2_norm = [](const std::vector<Real>& values) {
                return std::sqrt(std::inner_product(
                    values.begin(), values.end(), values.begin(), 0.0));
            };
            std::ostringstream oss;
            oss << "NewtonSolver: bordered solve selection"
                << " source="
                << ((solve_bordered_ptr == &bordered_full) ? "full" : "reduced")
                << " n_field_dofs=" << solve_bordered.n_field_dofs
                << " n_aux=" << solve_bordered.n_aux
                << " direct_records=" << solve_bordered.direct_coupling_records.size()
                << " ||B||=" << l2_norm(solve_bordered.B)
                << " ||Ct||=" << l2_norm(solve_bordered.Ct)
                << " ||D||=" << l2_norm(solve_bordered.D)
                << " ||g||=" << l2_norm(solve_bordered.g);
            traceLog(oss.str());
        }
        bool condensed_bordered_active = false;
        std::vector<Real> condensed_rhs_shift;
        std::vector<Real> condensed_Dinv;
        std::vector<Real> condensed_DinvC;
        if (has_solve_bordered &&
            linear.supportsNativeReducedFieldUpdates() &&
            !any_non_dirichlet_affine_constraints) {
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

            const bool has_direct_coupling_records =
                !solve_bordered.direct_coupling_records.empty();

            if (na > 0 && static_cast<int>(na) <= max_condensed_aux &&
                // Direct-coupled bordered rows are assembled through auxiliary
                // input/output sensitivities rather than explicit dense Ct rows.
                // The explicit bordered recovery path is numerically safer here
                // than the condensed surrogate.
                !has_direct_coupling_records &&
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

        // The availability and one-slot promotion decisions below precede
        // communicator collectives.  Verify the rank-local sparse factors
        // retain the same globally ordered slots even on empty-owner ranks.
        validateLowRankUpdateSlots(effective_rank_one_updates.size(),
                                   active_reduced_field_updates.size(),
                                   grouped_bordered_field_couplings.size(),
                                   "post_auxiliary_reduction");

        const bool direct_only_outlet_updates_available =
            !force_explicit_rank_one_updates &&
            linear.supportsNativeRankOneUpdates() &&
            linear.supportsNativeReducedFieldUpdates() &&
            !any_non_dirichlet_affine_constraints &&
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
            const bool allow_direct_only_reduced_rank_one_promotion =
                active_reduced_field_updates.size() == 1u;
            std::vector<backends::ReducedFieldUpdate> promoted_remaining_reduced_updates;
            promoted_remaining_reduced_updates.reserve(active_reduced_field_updates.size());
            std::size_t promoted_count = 0;
            if (allow_direct_only_reduced_rank_one_promotion) {
                for (const auto& update : active_reduced_field_updates) {
                    backends::RankOneUpdate promoted_rank_one;
                    if (tryPromoteReducedFieldUpdateToNativeRankOne(
                            update,
                            promoted_rank_one,
                            system_communicator)) {
                        effective_rank_one_updates.push_back(std::move(promoted_rank_one));
                        ++promoted_count;
                    } else {
                        promoted_remaining_reduced_updates.push_back(update);
                    }
                }
            } else {
                promoted_remaining_reduced_updates = active_reduced_field_updates;
                if (oopTraceEnabled() && !active_reduced_field_updates.empty()) {
                    std::ostringstream oss;
                    oss << "NewtonSolver: keeping multi-mode direct-only reduced updates on exact reduced path"
                        << " count=" << active_reduced_field_updates.size();
                    traceLog(oss.str());
                }
            }
            if (promoted_count > 0) {
                active_reduced_field_updates = std::move(promoted_remaining_reduced_updates);
                if (oopTraceEnabled()) {
                    std::ostringstream oss;
                    oss << "NewtonSolver: promoted direct-only reduced updates to native rank-1"
                        << " count=" << promoted_count
                        << " remaining_reduced=" << active_reduced_field_updates.size();
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

        validateLowRankUpdateSlots(effective_rank_one_updates.size(),
                                   active_reduced_field_updates.size(),
                                   grouped_bordered_field_couplings.size(),
                                   "pre_linear_bridge");

        // Bridge rank-1 / reduced updates from coupled BC assembly to the linear
        // solver only after any bordered condensation has augmented the active
        // reduced/grouped coupling sets for this Newton solve.
        const bool explicit_rank_one_in_matrix = bridgeRankOneUpdates();
        if (any_constrained_dofs) {
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
            !any_non_dirichlet_affine_constraints;
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
                if (!(options_.abs_tolerance > 0.0) ||
                    !(residual_fraction > static_cast<Real>(0.0)) ||
                    !(max_relative_residual > static_cast<Real>(0.0)) ||
                    rep.iterations <= 0 ||
                    !(rep.initial_residual_norm > static_cast<Real>(0.0)) ||
                    !(rep.final_residual_norm >= static_cast<Real>(0.0)) ||
                    !(rep.relative_residual >= static_cast<Real>(0.0)) ||
                    !std::isfinite(rep.initial_residual_norm) ||
                    !std::isfinite(rep.final_residual_norm) ||
                    !std::isfinite(rep.relative_residual)) {
                    return false;
                }

                const Real nonlinear_floor =
                    static_cast<Real>(options_.abs_tolerance) * residual_fraction;
                return rep.final_residual_norm <= static_cast<double>(nonlinear_floor) &&
                       rep.relative_residual <= static_cast<double>(max_relative_residual);
            };

        const auto sanitizeLinearSolveResult =
            [&](backends::SolverReport& linear_report,
                backends::GenericVector& correction,
                std::string_view phase) -> bool {
                const bool report_metrics_are_finite =
                    linear_report.iterations >= 0 &&
                    std::isfinite(static_cast<double>(
                        linear_report.initial_residual_norm)) &&
                    std::isfinite(static_cast<double>(
                        linear_report.final_residual_norm)) &&
                    std::isfinite(static_cast<double>(
                        linear_report.relative_residual)) &&
                    linear_report.initial_residual_norm >= Real{0.0} &&
                    linear_report.final_residual_norm >= Real{0.0} &&
                    linear_report.relative_residual >= Real{0.0};
                bool correction_is_finite = true;
                for (const auto value : correction.localSpan()) {
                    if (!std::isfinite(static_cast<double>(value))) {
                        correction_is_finite = false;
                        break;
                    }
                }
                const bool local_result_is_usable =
                    !linear_report.numerical_breakdown &&
                    report_metrics_are_finite &&
                    correction_is_finite;
                const bool result_is_usable =
                    allRanks(local_result_is_usable);
                if (result_is_usable) {
                    return true;
                }

                const auto raw_initial = linear_report.initial_residual_norm;
                const auto raw_final = linear_report.final_residual_norm;
                const auto raw_relative = linear_report.relative_residual;
                const int raw_iterations = linear_report.iterations;
                const bool backend_reported_breakdown =
                    linear_report.numerical_breakdown;

                std::string reason;
                auto append_reason = [&](std::string_view entry) {
                    if (!reason.empty()) {
                        reason += ',';
                    }
                    reason.append(entry.data(), entry.size());
                };
                if (backend_reported_breakdown) {
                    append_reason("backend");
                }
                if (!report_metrics_are_finite) {
                    append_reason("report");
                }
                if (!correction_is_finite) {
                    append_reason("correction");
                }
                if (reason.empty()) {
                    append_reason("remote_rank");
                }

                // A numerical breakdown is never an admissible inexact
                // Newton correction.  Zero it before any normalization,
                // replay, line search, or bordered recovery can consume it.
                correction.zero();
                linear_report.converged = false;
                linear_report.numerical_breakdown = true;
                linear_report.final_residual_norm =
                    std::numeric_limits<Real>::infinity();
                linear_report.relative_residual =
                    std::numeric_limits<Real>::infinity();
                const std::string detail =
                    "numerical breakdown: " + reason;
                if (linear_report.message.empty()) {
                    linear_report.message = detail;
                } else if (linear_report.message.find(detail) ==
                           std::string::npos) {
                    linear_report.message += " (" + detail + ")";
                }

                if (mpiRank() == 0) {
                    std::ostringstream oss;
                    oss << "NewtonSolver: rejected unusable linear result"
                        << " diagnostic=linear_solve_invalid_result"
                        << " phase='" << phase << "'"
                        << " reason='" << reason << "'"
                        << " backend_breakdown="
                        << (backend_reported_breakdown ? 1 : 0)
                        << " correction_finite="
                        << (correction_is_finite ? 1 : 0)
                        << " iterations=" << raw_iterations
                        << " initial_residual_norm=" << raw_initial
                        << " final_residual_norm=" << raw_final
                        << " relative_residual=" << raw_relative;
                    FE_LOG_INFO(oss.str());
                }
                return false;
            };

        int ptc_retries = 0;
        bool linear_probe_dumped = false;
        bool first_linear_vector_dumped = false;
        int primary_linear_solve_call_index = 0;
        backends::GenericVector* accepted_linear_rhs = &r;
        while (true) {
            SolverOptionsGuard bordered_solver_options_guard{linear, base_linear_options};
            if (needs_strict_coupled_solve_options) {
                linear.setOptions(makeBorderedSolveOptions(base_linear_options));
            } else if (needs_validated_native_rank_one_options) {
                std::optional<Real> direct_only_inner_rel_override;
                std::optional<Real> direct_only_outer_rel_floor;
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
                const std::size_t direct_only_update_count =
                    effective_rank_one_updates.size() + active_reduced_field_updates.size();
                if ((has_native_direct_face_only_updates || has_native_direct_only_reduced_updates) &&
                    direct_only_update_count > 1u) {
                    // Only the multi-mode direct-only outlet path needs the moderate
                    // 1e-5 outer floor. Keep it rank-agnostic so serial and MPI use
                    // the same target, but avoid retuning single-mode cases such as
                    // pipe_RCR_3d that are already stable at the looser XML setting.
                    direct_only_outer_rel_floor = static_cast<Real>(1e-5);
                }
                linear.setOptions(makeValidatedNativeRankOneSolveOptions(
                    base_linear_options,
                    native_direct_face_mode_count,
                    direct_only_inner_rel_override,
                    direct_only_outer_rel_floor));
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
            const bool mpi_runtime_analysis =
                mpiMultiTaskActive(system_communicator);
            if (!tangent_analysis_report_logged) {
                tangent_analysis_report_logged = true;
                const bool numeric_summaries_updated =
                    transient.system().updateAnalysisSummariesFromAssembledOperator(
                        J, options_.jacobian_op, &state);
                if (numeric_summaries_updated && !mpi_runtime_analysis) {
                    logPostTangentAnalysisReport(transient.system(), numeric_summaries_updated);
                } else if (numeric_summaries_updated && oopTraceEnabled()) {
                    traceLog("NewtonSolver: updated post-tangent analysis summaries in MPI; "
                             "full report logging deferred to keep collectives rank-symmetric.");
                }
            }
            logNewtonMatrixSupportDiagnostic(transient.system(),
                                             J,
                                             std::span<const GlobalIndex>(
                                                 constrained_dofs.data(),
                                                 constrained_dofs.size()),
                                             std::string_view("pre_linear_solve"),
                                             it,
                                             solve_time,
                                             base_state.dt);
            logActivePressureSupportRankDiagnostic(
                transient.system(),
                J,
                std::span<const GlobalIndex>(constrained_dofs.data(),
                                             constrained_dofs.size()),
                std::string_view("pre_linear_solve"),
                it,
                solve_time,
                base_state.dt);
            applyActivePressureSupportRankClamp(
                transient.system(),
                J,
                *linear_rhs,
                std::span<const GlobalIndex>(constrained_dofs.data(),
                                             constrained_dofs.size()),
                std::string_view("pre_linear_solve"),
                it,
                solve_time,
                base_state.dt);
            applyActivePressureGraphCompletion(
                transient.system(),
                J,
                std::span<const GlobalIndex>(constrained_dofs.data(),
                                             constrained_dofs.size()),
                std::string_view("pre_linear_solve"),
                it,
                solve_time,
                base_state.dt);
            if (oopTraceEnabled()) {
                traceLog("NewtonSolver: calling linear.solve()");
            }
            auto log_linear_solve_memory = [&](const char* phase) {
                if (!linearSolveMemoryDiagnosticsEnabled() || mpiRank() != 0) {
                    return;
                }
                const auto memory = readProcessMemorySnapshot();
                FE_LOG_INFO(
                    std::string("[svMultiPhysics::FE] NewtonSolver diagnostic=process_memory")
                    + " phase=" + phase
                    + " nonlinear_iteration=" + std::to_string(it)
                    + " ptc_retry=" + std::to_string(ptc_retries)
                    + " linear_backend='" + std::string(backends::backendKindToString(linear.backendKind())) + "'"
                    + " matrix_rows=" + std::to_string(J.numRows())
                    + " matrix_cols=" + std::to_string(J.numCols())
                    + " process_vm_kb=" + std::to_string(memory.vm_kb)
                    + " process_rss_kb=" + std::to_string(memory.rss_kb)
                    + " basis_cache_entries="
                    + std::to_string(basis::BasisCache::instance().size()));
            };
            log_linear_solve_memory("before_linear_solve");
            if (linearProbeDumpEnabled() && !linear_probe_dumped && it == 0 && ptc_retries == 0) {
                linear_probe_dumped = true;
                logVectorComponentNorms(transient.system(), *linear_rhs, "linear rhs");
                logVectorTopEntries(transient.system(), *linear_rhs, "linear rhs", 8u);

                u_backup.set(static_cast<Real>(1.0));
                u_backup.updateGhosts();
                residual_base.zero();
                J.mult(u_backup, residual_base);
                if (has_native_rank_one_updates) {
                    addRankOneOperatorMatvec(
                        std::span<const backends::RankOneUpdate>(effective_rank_one_updates.data(),
                                                                 effective_rank_one_updates.size()),
                        u_backup,
                        residual_base,
                        system_communicator);
                    addReducedFieldOperatorMatvec(
                        std::span<const backends::ReducedFieldUpdate>(
                            active_reduced_field_updates.data(),
                            active_reduced_field_updates.size()),
                        u_backup,
                        residual_base,
                        system_communicator);
                }
                logVectorComponentNorms(transient.system(), u_backup, "linear probe x");
                logVectorTopEntries(transient.system(), u_backup, "linear probe x", 8u);
                logVectorComponentNorms(transient.system(), residual_base, "linear probe Jx");
                logVectorTopEntries(transient.system(), residual_base, "linear probe Jx", 8u);

                for (std::size_t ridx = 0; ridx < effective_rank_one_updates.size(); ++ridx) {
                    u_backup.zero();
                    auto probe_view = u_backup.createAssemblyView();
                    FE_CHECK_NOT_NULL(probe_view.get(),
                                      "NewtonSolver: rank-one probe view");
                    probe_view->beginAssemblyPhase();
                    for (const auto& [dof, value] : effective_rank_one_updates[ridx].v) {
                        if (std::abs(value) > Real(1e-30)) {
                            probe_view->addVectorEntry(dof,
                                                       value,
                                                       assembly::AddMode::Add);
                        }
                    }
                    probe_view->finalizeAssembly();
                    u_backup.updateGhosts();

                    residual_base.zero();
                    J.mult(u_backup, residual_base);
                    if (has_native_rank_one_updates) {
                        addRankOneOperatorMatvec(
                            std::span<const backends::RankOneUpdate>(effective_rank_one_updates.data(),
                                                                     effective_rank_one_updates.size()),
                            u_backup,
                            residual_base,
                            system_communicator);
                        addReducedFieldOperatorMatvec(
                            std::span<const backends::ReducedFieldUpdate>(
                                active_reduced_field_updates.data(),
                                active_reduced_field_updates.size()),
                            u_backup,
                            residual_base,
                            system_communicator);
                    }

                    const auto label_x =
                        "rank-one probe x[" + std::to_string(ridx) + "]";
                    const auto label_jx =
                        "rank-one probe Jx[" + std::to_string(ridx) + "]";
                    logVectorComponentNorms(transient.system(), u_backup, label_x);
                    logVectorTopEntries(transient.system(), u_backup, label_x, 8u);
                    logVectorComponentNorms(transient.system(), residual_base, label_jx);
                    logVectorTopEntries(transient.system(), residual_base, label_jx, 8u);
                }

                if (base_linear_options.block_layout.has_value()) {
                    const auto* constraint_block =
                        base_linear_options.block_layout->constraintFieldBlock();
                    if (constraint_block != nullptr &&
                        base_linear_options.block_layout->totalComponents() > 0) {
                        const auto& fmap = transient.system().fieldMap();
                        std::vector<int> field_component_offsets(fmap.numFields(), 0);
                        int component_offset = 0;
                        for (std::size_t field_idx = 0; field_idx < fmap.numFields(); ++field_idx) {
                            field_component_offsets[field_idx] = component_offset;
                            component_offset += fmap.numComponents(field_idx);
                        }
                        u_backup.zero();
                        auto probe_view = u_backup.createAssemblyView();
                        FE_CHECK_NOT_NULL(probe_view.get(),
                                          "NewtonSolver: constraint-only probe view");
                        probe_view->beginAssemblyPhase();
                        for (const auto dof : owned_dofs) {
                            const auto comp = fmap.getComponentOfDof(dof);
                            if (!comp) {
                                continue;
                            }
                            const auto field_idx =
                                static_cast<std::size_t>(std::max(comp->first, 0));
                            if (field_idx >= fmap.numFields()) {
                                continue;
                            }
                            const int block_comp =
                                field_component_offsets[field_idx] + static_cast<int>(comp->second);
                            if (block_comp < constraint_block->start_component ||
                                block_comp >= constraint_block->start_component +
                                                  constraint_block->n_components) {
                                continue;
                            }
                            probe_view->addVectorEntry(dof,
                                                       static_cast<Real>(1.0),
                                                       assembly::AddMode::Insert);
                        }
                        probe_view->finalizeAssembly();
                        u_backup.updateGhosts();

                        residual_base.zero();
                        J.mult(u_backup, residual_base);
                        logVectorComponentNorms(
                            transient.system(), u_backup, "constraint-only probe x");
                        logVectorTopEntries(
                            transient.system(), u_backup, "constraint-only probe x", 8u);
                        logVectorComponentNorms(
                            transient.system(), residual_base, "constraint-only probe Jx");
                        logVectorTopEntries(
                            transient.system(), residual_base, "constraint-only probe Jx", 8u);
                    }
                }
            }
            ntp0 = NTP();
            report.linear = linear.solve(J, du, *linear_rhs);
            ntp_linear += NTP() - ntp0;
            const bool linear_result_is_usable =
                sanitizeLinearSolveResult(report.linear, du, "newton");
            log_linear_solve_memory("after_linear_solve");
            ntp_linear_iters_total += report.linear.iterations;
            copyVector(residual_minus, *linear_rhs);
            accepted_linear_rhs = &residual_minus;
            std::optional<std::vector<Real>> first_linear_dense_rhs;
            std::optional<std::vector<Real>> first_linear_dense_du_raw;
            if (!first_linear_vector_dumped && it == 0 && ptc_retries == 0) {
                if (const auto dump_prefix = firstLinearVectorDumpPrefix()) {
                    first_linear_dense_rhs = gatherGlobalDenseVectorFromOwnedEntries(
                        *linear_rhs,
                        static_cast<std::size_t>(J.numRows()),
                        owned_dofs,
                        system_communicator);
                    first_linear_dense_du_raw = gatherGlobalDenseVectorFromOwnedEntries(
                        du,
                        static_cast<std::size_t>(J.numRows()),
                        owned_dofs,
                        system_communicator);
                }
            }
            if (oopTraceEnabled()) {
                std::ostringstream oss;
                oss << "NewtonSolver: post-linear.solve du_norm=" << du.norm()
                    << " basis_supported=" << (linear.supportsNullspace() ? 1 : 0);
                traceLog(oss.str());
            }
            normalizeFsilsPostSolveIncrementIfNeeded(du);
            if (!first_linear_vector_dumped && it == 0 && ptc_retries == 0) {
                if (const auto dump_prefix = firstLinearVectorDumpPrefix()) {
                    const auto dense_du_normalized = gatherGlobalDenseVectorFromOwnedEntries(
                        du,
                        static_cast<std::size_t>(J.numRows()),
                        owned_dofs,
                        system_communicator);
                    std::optional<std::vector<ScalarFieldVertexDumpRecord>> scalar_field_vertex_records;
                    std::optional<std::string> scalar_field_vertex_name;
                    if (const auto field_idx =
                            selectPreferredScalarVertexDumpFieldIndex(transient.system());
                        field_idx.has_value()) {
                        scalar_field_vertex_records =
                            gatherScalarFieldVertexDumpRecords(transient.system(), du, *field_idx);
                        scalar_field_vertex_name =
                            transient.system().fieldMap().getField(*field_idx).name;
                    }
                    if (mpiRank() == 0) {
                        FE_THROW_IF(!first_linear_dense_rhs.has_value() ||
                                        !first_linear_dense_du_raw.has_value(),
                                    systems::InvalidStateException,
                                    "NewtonSolver: first linear vector dump buffers missing");
                        writeDenseVectorDump(*dump_prefix + ".rhs.txt", *first_linear_dense_rhs);
                        writeDenseVectorDump(*dump_prefix + ".du_raw.txt", *first_linear_dense_du_raw);
                        writeDenseVectorDump(*dump_prefix + ".du_normalized.txt", dense_du_normalized);
                        if (scalar_field_vertex_records.has_value() &&
                            scalar_field_vertex_name.has_value()) {
                            writeScalarFieldVertexDumpRecords(
                                *dump_prefix + ".scalar_vertex_records.txt",
                                *scalar_field_vertex_name,
                                *scalar_field_vertex_records);
                        }
                    }
                    first_linear_vector_dumped = true;
                }
            }
            if (linearSolveHistoryEnabled()) {
                ++primary_linear_solve_call_index;
                const int max_calls = linearSolveHistoryMaxCalls();
                if (max_calls < 0 || primary_linear_solve_call_index <= max_calls) {
                    const Real rhs_norm = linear_rhs->norm();
                    if (mpiRank() == 0) {
                        std::ostringstream oss;
                        oss << "NewtonSolver: linear solve history"
                            << " call=" << primary_linear_solve_call_index
                            << " newton_it=" << it
                            << " ptc_retries=" << ptc_retries
                            << " rhs_norm=" << rhs_norm
                            << " residual_before=" << current_residual_norm
                            << " converged=" << report.linear.converged
                            << " numerical_breakdown="
                            << (report.linear.numerical_breakdown ? 1 : 0)
                            << " iters=" << report.linear.iterations
                            << " r0=" << report.linear.initial_residual_norm
                            << " rn=" << report.linear.final_residual_norm
                            << " rel=" << report.linear.relative_residual
                            << " outer=" << report.linear.blockschur_outer_iterations
                            << " mom_solves=" << report.linear.blockschur_momentum_solve_calls
                            << " mom_iters=" << report.linear.blockschur_momentum_iterations
                            << " schur_solves=" << report.linear.blockschur_schur_solve_calls
                            << " schur_iters=" << report.linear.blockschur_schur_iterations
                            << " native_rank_one=" << native_direct_face_mode_count
                            << " reduced_updates=" << active_reduced_field_updates.size()
                            << " grouped_couplings=" << grouped_bordered_field_couplings.size()
                            << " condensed_bordered=" << (condensed_bordered_active ? 1 : 0)
                            << " msg='" << report.linear.message << "'";
                        FE_LOG_INFO(oss.str());
                    }
                }
            }
            if (oopTraceEnabled()) {
                std::ostringstream oss;
                oss << "NewtonSolver: post-normalize du_norm=" << du.norm();
                traceLog(oss.str());
            }
            if (linearSolveComponentNormsEnabled()) {
                const int max_newton_it = linearSolveComponentNormsMaxNewtonIt();
                if (max_newton_it < 0 || it <= max_newton_it) {
                    const std::string label =
                        "du after linear solve [newton_it=" + std::to_string(it) + "]";
                    logVectorComponentNorms(transient.system(), du, label);
                }
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
                        residual_scratch,
                        system_communicator);
                    addReducedFieldOperatorMatvec(
                        std::span<const backends::ReducedFieldUpdate>(
                            active_reduced_field_updates.data(),
                            active_reduced_field_updates.size()),
                        du,
                        residual_scratch,
                        system_communicator);
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
                    << " numerical_breakdown="
                    << (report.linear.numerical_breakdown ? 1 : 0)
                    << " iters=" << report.linear.iterations
                    << " r0=" << report.linear.initial_residual_norm
                    << " rn=" << report.linear.final_residual_norm
                    << " rel=" << report.linear.relative_residual
                    << " msg='" << report.linear.message << "'";
                traceLog(oss.str());
            }
            // Linear solver reports are expected to describe a collective
            // solve, but do not rely on every backend returning bit-identical
            // local status.  A partial-rank early exit would desynchronize the
            // PTC retry and subsequent distributed matrix operations.
            if (linear_result_is_usable &&
                allRanks(report.linear.converged)) {
                break;
            }

            const bool meets_original_linear_target =
                linear_result_is_usable &&
                (needs_strict_coupled_solve_options || needs_validated_native_rank_one_options) &&
                reportMeetsRequestedLinearTarget(report.linear, bordered_solver_options_guard.saved);
            const bool meets_nonlinear_linear_floor =
                linear_result_is_usable &&
                reportMeetsNonlinearAbsoluteLinearFloor(report.linear,
                                                        static_cast<Real>(0.1),
                                                        static_cast<Real>(0.1));
            const bool meets_collective_linear_acceptance =
                allRanks(meets_original_linear_target ||
                         meets_nonlinear_linear_floor);
            if (meets_collective_linear_acceptance) {
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

            // Inexact Newton is an explicit opt-in, but even that policy must
            // only consume a finite correction backed by measurable linear
            // progress.  In particular, a backend's finite default report
            // (zero iterations and zero residuals) is not evidence that a
            // usable Newton direction was produced.  Form the decision
            // collectively so every rank either leaves this retry loop or
            // follows the same failure/PTC path.
            const bool local_inexact_report_has_progress =
                report.linear.iterations > 0 &&
                report.linear.initial_residual_norm > Real{0.0} &&
                report.linear.final_residual_norm >= Real{0.0} &&
                report.linear.final_residual_norm <
                    report.linear.initial_residual_norm &&
                report.linear.relative_residual >= Real{0.0} &&
                report.linear.relative_residual < Real{1.0} &&
                std::isfinite(static_cast<double>(
                    report.linear.initial_residual_norm)) &&
                std::isfinite(static_cast<double>(
                    report.linear.final_residual_norm)) &&
                std::isfinite(static_cast<double>(
                    report.linear.relative_residual));
            bool local_inexact_correction_is_finite = true;
            bool local_inexact_correction_is_nonzero = false;
            for (const auto value : du.localSpan()) {
                if (!std::isfinite(static_cast<double>(value))) {
                    local_inexact_correction_is_finite = false;
                    break;
                }
                local_inexact_correction_is_nonzero =
                    local_inexact_correction_is_nonzero || value != Real{0.0};
            }
            const bool inexact_correction_is_finite =
                allRanks(local_inexact_correction_is_finite);
            const bool inexact_correction_is_globally_nonzero =
                anyRank(local_inexact_correction_is_nonzero);
            const bool all_ranks_allow_inexact_main_solve = allRanks(
                linear_result_is_usable &&
                !report.linear.numerical_breakdown &&
                options_.accept_inexact_linear_solutions &&
                !needs_strict_coupled_solve_options &&
                !needs_validated_native_rank_one_options &&
                local_inexact_report_has_progress);
            const bool allow_inexact_main_solve =
                all_ranks_allow_inexact_main_solve &&
                inexact_correction_is_finite &&
                inexact_correction_is_globally_nonzero;
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

        auto gatherDenseVector = [&](backends::GenericVector& vec, std::size_t n) {
            return gatherGlobalDenseVectorFromOwnedEntries(
                vec, n, owned_dofs, system_communicator);
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

        auto recoverExplicitBorderedCorrection =
            [&](const systems::FESystem::BorderedCouplingData& solve_bordered,
                std::span<const Real> initial_dense_du) {
                const auto nf = solve_bordered.n_field_dofs;
                const auto na = static_cast<std::size_t>(solve_bordered.n_aux);
                FE_THROW_IF(nf != initial_dense_du.size(),
                            systems::InvalidStateException,
                            "NewtonSolver: bordered recovery field size mismatch");
                FE_THROW_IF(solve_bordered.B.size() != nf * na ||
                                solve_bordered.Ct.size() != nf * na ||
                                solve_bordered.D.size() != na * na ||
                                solve_bordered.g.size() != na,
                            systems::InvalidStateException,
                            "NewtonSolver: bordered coupling storage size mismatch");

                std::vector<Real> dense_du(initial_dense_du.begin(), initial_dense_du.end());
                std::vector<Real> solve_aux_delta;

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

                    solve_aux_delta.assign(na, Real(0.0));
                    std::vector<Real> aux_rhs(na, Real(0.0));
                    for (std::size_t i = 0; i < na; ++i) {
                        Real aux_rhs_i = solve_bordered.g[i];
                        if (direct_ct_du.row_covered[i]) {
                            aux_rhs_i -= direct_ct_du.values[i];
                        } else {
                            for (std::size_t k = 0; k < nf; ++k) {
                                aux_rhs_i -= solve_bordered.Ct[i * nf + k] * dense_du[k];
                            }
                        }
                        aux_rhs[i] = aux_rhs_i;
                    }
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
                        auto bordered_recovery_options =
                            makeBorderedSolveOptions(base_linear_options);
                        linear.setOptions(bordered_recovery_options);

                        auto bordered_column_residual =
                            [&](backends::GenericVector& rhs,
                                backends::GenericVector& x,
                                backends::GenericVector& product,
                                backends::GenericVector& residual) {
                                product.zero();
                                J.mult(x, product);
                                if (!effective_rank_one_updates.empty()) {
                                    addRankOneOperatorMatvec(
                                        std::span<const backends::RankOneUpdate>(
                                            effective_rank_one_updates.data(),
                                            effective_rank_one_updates.size()),
                                        x,
                                        product,
                                        system_communicator);
                                }
                                if (!active_reduced_field_updates.empty()) {
                                    addReducedFieldOperatorMatvec(
                                        std::span<const backends::ReducedFieldUpdate>(
                                            active_reduced_field_updates.data(),
                                            active_reduced_field_updates.size()),
                                        x,
                                        product,
                                        system_communicator);
                                }
                                copyVector(residual, rhs);
                                axpy(residual, static_cast<Real>(-1.0), product);
                                syncOwnedRowHaloIfNeeded(residual);
                                const Real rhs_norm =
                                    std::max<Real>(static_cast<Real>(rhs.norm()), static_cast<Real>(1e-30));
                                const Real residual_norm = static_cast<Real>(residual.norm());
                                return std::pair{residual_norm, rhs_norm};
                            };

                        auto bordered_column_target = [&](Real rhs_norm) {
                            return std::max<Real>(
                                bordered_recovery_options.abs_tol,
                                bordered_recovery_options.rel_tol *
                                    std::max<Real>(rhs_norm, static_cast<Real>(1e-30)));
                        };

                        for (std::size_t j = 0; j < na; ++j) {
                            restoreFsilsMatrixSnapshot(J, fsils_matrix_snapshot);

                            residual_scratch.zero();
                            {
                                auto rhs_view = residual_scratch.createAssemblyView();
                                FE_CHECK_NOT_NULL(rhs_view.get(), "NewtonSolver: bordered rhs view");
                                rhs_view->beginAssemblyPhase();
                                for (std::size_t row = 0; row < nf; ++row) {
                                    if (!owned_dofs.contains(static_cast<GlobalIndex>(row))) {
                                        continue;
                                    }
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
                            auto z_report = linear.solve(J, du, residual_scratch);
                            ntp_linear += NTP() - ntp0;
                            ntp_linear_iters_total += z_report.iterations;
                            const bool z_result_is_usable =
                                sanitizeLinearSolveResult(
                                    z_report, du, "bordered_column");
                            normalizeFsilsPostSolveIncrementIfNeeded(du);
                            int z_total_iterations = z_report.iterations;
                            const bool local_z_backend_converged =
                                z_result_is_usable && z_report.converged;
                            const bool local_z_meets_original_target =
                                z_result_is_usable &&
                                !local_z_backend_converged &&
                                has_native_rank_one_updates &&
                                reportMeetsRequestedLinearTarget(
                                    z_report, base_linear_options);
                            const bool local_z_meets_near_target =
                                z_result_is_usable &&
                                !local_z_backend_converged &&
                                !local_z_meets_original_target &&
                                has_native_rank_one_updates &&
                                reportMeetsRequestedLinearTargetWithinFactor(
                                    z_report,
                                    base_linear_options,
                                    static_cast<Real>(4.0));
                            bool z_converged = allRanks(
                                local_z_backend_converged ||
                                local_z_meets_original_target ||
                                local_z_meets_near_target);
                            if (z_converged &&
                                local_z_meets_original_target) {
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
                            } else if (z_converged &&
                                       local_z_meets_near_target) {
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
                            if (z_result_is_usable && !z_converged &&
                                grouped_bordered_field_couplings.empty()) {
                                constexpr int max_polish_attempts = 2;
                                for (int polish = 0;
                                     polish < max_polish_attempts && !z_converged;
                                     ++polish) {
                                    restoreFsilsMatrixSnapshot(J, fsils_matrix_snapshot);
                                    auto [before_norm, rhs_norm] =
                                        bordered_column_residual(residual_scratch,
                                                                 du,
                                                                 u_backup,
                                                                 residual_base);
                                    const Real target = bordered_column_target(rhs_norm);
                                    const bool replay_is_acceptable = allRanks(
                                        std::isfinite(static_cast<double>(before_norm)) &&
                                        before_norm <= target);
                                    if (replay_is_acceptable) {
                                        z_converged = true;
                                        z_report.converged = true;
                                        z_report.final_residual_norm = before_norm;
                                        z_report.relative_residual =
                                            before_norm / std::max<Real>(rhs_norm, static_cast<Real>(1e-30));
                                        if (oopTraceEnabled()) {
                                            std::ostringstream oss;
                                            oss << "NewtonSolver: accepting bordered K^{-1}B residual replay"
                                                << " column=" << j
                                                << " residual=" << before_norm
                                                << " target=" << target;
                                            traceLog(oss.str());
                                        }
                                        break;
                                    }
                                    const bool replay_is_finite = allRanks(
                                        std::isfinite(static_cast<double>(before_norm)));
                                    if (!replay_is_finite) {
                                        break;
                                    }

                                    restoreFsilsMatrixSnapshot(J, fsils_matrix_snapshot);
                                    u_backup.zero();
                                    ntp0 = NTP();
                                    auto polish_report =
                                        linear.solve(J, u_backup, residual_base);
                                    ntp_linear += NTP() - ntp0;
                                    ntp_linear_iters_total += polish_report.iterations;
                                    z_total_iterations += polish_report.iterations;
                                    const bool polish_result_is_usable =
                                        sanitizeLinearSolveResult(
                                            polish_report,
                                            u_backup,
                                            "bordered_polish");
                                    normalizeFsilsPostSolveIncrementIfNeeded(u_backup);
                                    if (!polish_result_is_usable) {
                                        z_report = polish_report;
                                        break;
                                    }
                                    axpy(du, static_cast<Real>(1.0), u_backup);

                                    restoreFsilsMatrixSnapshot(J, fsils_matrix_snapshot);
                                    auto [after_norm, after_rhs_norm] =
                                        bordered_column_residual(residual_scratch,
                                                                 du,
                                                                 u_backup,
                                                                 residual_base);
                                    const Real after_target =
                                        bordered_column_target(after_rhs_norm);
                                    if (oopTraceEnabled()) {
                                        std::ostringstream oss;
                                        oss << "NewtonSolver: bordered K^{-1}B residual polish"
                                            << " column=" << j
                                            << " attempt=" << (polish + 1)
                                            << " before=" << before_norm
                                            << " after=" << after_norm
                                            << " target=" << after_target
                                            << " correction_converged="
                                            << (polish_report.converged ? 1 : 0)
                                            << " correction_iters="
                                            << polish_report.iterations;
                                        traceLog(oss.str());
                                    }
                                    const bool polished_result_is_acceptable =
                                        allRanks(
                                            std::isfinite(static_cast<double>(after_norm)) &&
                                            after_norm <= after_target);
                                    if (polished_result_is_acceptable) {
                                        z_converged = true;
                                        z_report = polish_report;
                                        z_report.converged = true;
                                        z_report.iterations = z_total_iterations;
                                        z_report.initial_residual_norm = after_rhs_norm;
                                        z_report.final_residual_norm = after_norm;
                                        z_report.relative_residual =
                                            after_norm /
                                            std::max<Real>(after_rhs_norm, static_cast<Real>(1e-30));
                                        z_report.message = "bordered K^{-1}B residual polish";
                                    }
                                }
                            }
                            // Keep the final recovery outcome communicator-wide
                            // even if a future backend or residual implementation
                            // returns rank-local status metadata.
                            z_converged = allRanks(z_converged);
                            z_report.iterations = z_total_iterations;
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
                }

                return std::pair{std::move(dense_du), std::move(solve_aux_delta)};
            };

        if (has_solve_bordered && !condensed_bordered_active) {
            const auto& solve_bordered = *solve_bordered_ptr;
            const auto nf = solve_bordered.n_field_dofs;
            const auto na = static_cast<std::size_t>(solve_bordered.n_aux);
            FE_THROW_IF(nf != static_cast<std::size_t>(du.size()), systems::InvalidStateException,
                        "NewtonSolver: bordered PDE block size does not match solution size");

            copyVector(residual_base, du);
            auto dense_du = gatherDenseVector(residual_base, nf);
            std::tie(dense_du, solve_aux_delta) =
                recoverExplicitBorderedCorrection(solve_bordered, dense_du);
            scatterDenseVector(du, dense_du);

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

            const auto dense_du = gatherGlobalDenseVectorFromOwnedEntries(
                du, nf, owned_dofs, system_communicator);
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
            auto dense_du =
                gatherGlobalDenseVectorFromOwnedEntries(
                    du,
                    static_cast<std::size_t>(J.numRows()),
                    owned_dofs,
                    system_communicator);
            const bool use_pure_algebraic_bordered_recovery =
                pureAlgebraicBorderedRecoveryEnabled() &&
                has_bordered &&
                algebraic_aux_reduction.dynamic_indices.empty();
            if (use_pure_algebraic_bordered_recovery) {
                if (oopTraceEnabled()) {
                    traceLog("NewtonSolver: applying explicit bordered recovery for pure algebraic auxiliary block");
                }
                std::tie(dense_du, solve_aux_delta) =
                    recoverExplicitBorderedCorrection(bordered_full, dense_du);
                scatterDenseVector(du, dense_du);
                aux_delta = solve_aux_delta;
            } else {
                aux_delta = recoverAuxiliaryDeltaFromReduction(
                    algebraic_aux_reduction, dense_du, std::span<const Real>{});
            }
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

        logActivePressureUpdateSupportDiagnostic(
            transient.system(),
            J,
            du,
            *accepted_linear_rhs,
            std::span<const GlobalIndex>(constrained_dofs.data(),
                                         constrained_dofs.size()),
            std::string_view("post_linear_solve_raw_update"),
            it,
            solve_time,
            base_state.dt);

        applyDtIncrementScaling();

        // The backend result was checked before bordered/condensed/algebraic
        // recovery and dt-field scaling.  Each of those transformations can
        // still overflow a previously finite correction.  Reject the complete
        // transformed step collectively before it can update auxiliary state,
        // local condensed unknowns, TimeHistory, or a line-search trial.
        const auto finiteValues = [](const auto& values) {
            return std::all_of(
                values.begin(), values.end(), [](const auto value) {
                    return std::isfinite(static_cast<double>(value));
                });
        };
        const bool local_du_is_finite = finiteValues(du.localSpan());
        const bool local_aux_delta_is_finite = finiteValues(aux_delta);
        const bool local_solve_aux_delta_is_finite =
            finiteValues(solve_aux_delta);
        const bool local_transformed_correction_is_finite =
            local_du_is_finite &&
            local_aux_delta_is_finite &&
            local_solve_aux_delta_is_finite;
        if (!allRanks(local_transformed_correction_is_finite)) {
            // All ranks take the same failure branch.  Erase every correction
            // buffer first so exception handling, diagnostics, or a caller
            // that reuses the workspace cannot observe a poisoned update.
            du.zero();
            std::fill(aux_delta.begin(), aux_delta.end(), Real(0.0));
            std::fill(solve_aux_delta.begin(), solve_aux_delta.end(), Real(0.0));
            report.linear.converged = false;
            report.linear.numerical_breakdown = true;
            report.linear.final_residual_norm =
                std::numeric_limits<Real>::infinity();
            report.linear.relative_residual =
                std::numeric_limits<Real>::infinity();
            constexpr std::string_view detail =
                "numerical breakdown: nonfinite post-recovery correction";
            if (report.linear.message.empty()) {
                report.linear.message.assign(detail.data(), detail.size());
            } else if (report.linear.message.find(detail) ==
                       std::string::npos) {
                report.linear.message += " (";
                report.linear.message.append(detail.data(), detail.size());
                report.linear.message += ")";
            }

            if (mpiRank() == 0) {
                std::ostringstream oss;
                oss << "NewtonSolver: rejected nonfinite transformed correction"
                    << " diagnostic=post_recovery_invalid_correction"
                    << " du_finite=" << (local_du_is_finite ? 1 : 0)
                    << " aux_delta_finite="
                    << (local_aux_delta_is_finite ? 1 : 0)
                    << " solve_aux_delta_finite="
                    << (local_solve_aux_delta_is_finite ? 1 : 0)
                    << " remote_rank_failure="
                    << (local_transformed_correction_is_finite ? 1 : 0);
                FE_LOG_INFO(oss.str());
            }
            FE_THROW(
                systems::InvalidStateException,
                "NewtonSolver: rejected nonfinite post-recovery correction before state update");
        }

        logActivePressureUpdateSupportDiagnostic(
            transient.system(),
            J,
            du,
            *accepted_linear_rhs,
            std::span<const GlobalIndex>(constrained_dofs.data(),
                                         constrained_dofs.size()),
            std::string_view("post_linear_solve_update"),
            it,
            solve_time,
            base_state.dt);

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
            copyVector(u_backup, history.u());
            if (!aux_delta.empty()) {
                applyAuxiliaryDelta(transient.system(), bordered_full, aux_delta, static_cast<Real>(1.0));
            }
            if (use_local_condensed_recovery) {
                const auto dense_du = gatherDenseFieldDelta();
                transient.system().applyLocalCondensedRecovery(dense_du, static_cast<Real>(1.0));
            }
            axpy(history.u(), static_cast<Real>(-1.0), du);
            syncCurrentState();
            auto accepted_state_holder = makeNewtonState(history, solve_time);
            try {
                transient.system().beginGeometricNonlinearityTrial(
                    accepted_state_holder.view);
                synchronizeState(accepted_state_holder.view,
                                 StateSyncPoint::AcceptedNonlinearState);
                transient.system().acceptGeometricNonlinearityState(
                    accepted_state_holder.view,
                    systems::GeometricNonlinearityUpdatePoint::AcceptedNonlinearState);
            } catch (...) {
                transient.system().rollbackGeometricNonlinearityTrial(
                    /*force=*/true);
                throw;
            }
            const bool accepted_sync_invalidated_residual = !have_residual;
            if (transient.system().geometricNonlinearityEnabled() ||
                options_.accepted_state_sync_invalidates_residual) {
                have_jacobian = false;
            }
            ntp_update += NTP() - ntp0;
            have_residual = false;

            double actual_step_norm = du_norm;
            if (options_.step_tolerance > 0.0) {
                copyVector(residual_scratch, history.u());
                axpy(residual_scratch, static_cast<Real>(-1.0), u_backup);
                actual_step_norm = residual_scratch.norm();
            }

            const bool step_tolerance_requires_residual =
                options_.accepted_state_sync_invalidates_residual ||
                accepted_sync_invalidated_residual ||
                !field_residual_states.empty();
            if (step_tolerance_requires_residual &&
                options_.step_tolerance > 0.0) {
                current_residual_norm = assembleResidualOnly(
                    accepted_state_holder.view,
                    /*phase=*/"accepted_state_refresh",
                    StateSyncPoint::ResidualAssembly);
                have_residual = true;
            }

            if (options_.step_tolerance > 0.0) {
                if (oopTraceEnabled()) {
                    std::ostringstream oss;
                    oss << "NewtonSolver: actual projected step ||u_new-u_base||="
                        << actual_step_norm
                        << " raw ||du||=" << du_norm
                        << " step_tol=" << options_.step_tolerance;
                    traceLog(oss.str());
                }
                if (minIterationsSatisfied(it + 1) &&
                    actual_step_norm <= options_.step_tolerance &&
                    (!step_tolerance_requires_residual ||
                     tolerancesSatisfied(/*pre_first_update=*/false))) {
                    report.converged = true;
                    report.iterations = it + 1;
                    if (have_residual) {
                        updateResidualReport();
                    }
                    if (oopTraceEnabled()) {
                        traceLog(
                            "NewtonSolver: converged by step tolerance on the synchronized residual state.");
                    }
                    printNewtonProfile(it + 1);
                    return report;
                }
            }
            continue;
        }

        // Backtracking line search: choose alpha in (0,1] so the residual norm decreases.
        copyVector(u_backup, history.u());
        FE_CHECK_NOT_NULL(workspace.factory,
                          "NewtonSolver: line-search history backup factory");
        auto snapshotHistoryVector = [&] (
                                         std::unique_ptr<backends::GenericVector>& backup,
                                         const backends::GenericVector& source,
                                         const char* label) {
            if (!backup || backup->size() != source.size()) {
                backup = workspace.factory->createVector(source.size());
            }
            FE_CHECK_NOT_NULL(backup.get(), label);
            backup->copyFrom(source);
        };
        workspace.line_search_history_backup.resize(
            static_cast<std::size_t>(history.historyDepth()));
        for (int k = 1; k <= history.historyDepth(); ++k) {
            snapshotHistoryVector(
                workspace.line_search_history_backup[
                    static_cast<std::size_t>(k - 1)],
                history.uPrevK(k),
                "NewtonSolver: line-search history backup");
        }
        auto line_search_rate_backup =
            history.snapshotRateState(*workspace.factory);
        auto restoreLineSearchHistoryState = [&]() {
            FE_THROW_IF(
                workspace.line_search_history_backup.size() !=
                    static_cast<std::size_t>(history.historyDepth()),
                systems::InvalidStateException,
                "NewtonSolver: line-search history depth changed during a trial");
            for (int k = 1; k <= history.historyDepth(); ++k) {
                auto& backup = workspace.line_search_history_backup[
                    static_cast<std::size_t>(k - 1)];
                FE_CHECK_NOT_NULL(
                    backup.get(),
                    "NewtonSolver: line-search history restore backup");
                history.uPrevK(k).copyFrom(*backup);
            }
            history.restoreRateState(line_search_rate_backup);
            // restoreRateState swaps ownership with the snapshot. Refresh the
            // reusable copy immediately so a later rejected alpha restores
            // the same base values instead of swapping the prior trial back
            // into the live history. Allocation absence is preserved too.
            history.snapshotRateState(
                line_search_rate_backup, *workspace.factory);
        };
        line_search_history_transaction_active = true;
        const auto aux_state_backup =
            (aux_delta.empty() && !use_local_condensed_recovery)
                ? std::vector<Real>{}
                : transient.system().checkpointAuxiliaryState();
        const auto dense_du_for_aux =
            use_local_condensed_recovery
                ? gatherDenseFieldDelta()
                : std::vector<Real>{};
        const auto bordered_backup = transient.system().borderedCoupling();
        const auto base_constraint_semantics =
            constraintSemanticFingerprint(constraints);
        const double r_norm0 = current_residual_norm;
        const double r_norm0_sq = r_norm0 * r_norm0;

        auto restoreLineSearchBaseState = [&]() {
            // Geometry must be rolled back before rebuilding cut/support data
            // for the accepted algebraic state.  Reversing these operations
            // leaves derived geometry describing the rejected coordinates.
            transient.system().rollbackGeometricNonlinearityTrial(
                /*force=*/true);
            FE_THROW_IF(
                anyRank(
                    transient.system().meshCoordinateTransactionActive()),
                systems::InvalidStateException,
                "NewtonSolver: rejected line-search geometry could not be rolled back; enable geometric rollback on line-search rejection");
            copyVector(history.u(), u_backup);
            restoreLineSearchHistoryState();
            if (!aux_state_backup.empty()) {
                transient.system().restoreAuxiliaryState(aux_state_backup);
            }
            transient.system().borderedCoupling() = bordered_backup;
            if (auto* reg =
                    transient.system().auxiliaryInputRegistryIfPresent()) {
                reg->invalidateAll();
            }

            // Do not impose the rejected trial's constraints on the restored
            // vector.  The synchronization callback first reconstructs the
            // accepted cut/constraint state, then synchronizeState projects
            // with that reconstructed set.
            history.u().updateGhosts();
            syncOwnedRowHaloIfNeeded(history.u());
            auto restored_state_holder = makeNewtonState(history, solve_time);
            synchronizeState(restored_state_holder.view,
                             StateSyncPoint::RestoredNonlinearState);
            FE_THROW_IF(
                anyRank(constraintSemanticFingerprint(constraints) !=
                        base_constraint_semantics),
                systems::InvalidStateException,
                "NewtonSolver: line-search restoration did not reproduce "
                "the accepted affine-constraint state");
        };

        bool candidate_accepted_state_synchronized = false;
        auto evaluateLineSearchTrial = [&](double trial_alpha, const char* phase) -> double {
            candidate_accepted_state_synchronized = false;
            FE_THROW_IF(
                anyRank(constraintSemanticFingerprint(constraints) !=
                        base_constraint_semantics),
                systems::InvalidStateException,
                "NewtonSolver: line-search trial did not start from the "
                "accepted affine-constraint state");
            copyVector(history.u(), u_backup);
            restoreLineSearchHistoryState();
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

            // Linear solves operate in eliminated coordinates, so lift the
            // raw increment with the accepted/base constraints before asking
            // the callback to derive trial-dependent constraints.  Per-alpha
            // restoration above guarantees this is never a prior rejected
            // trial's constraint set.
            syncCurrentState();

            auto trial_state_holder = makeNewtonState(history, solve_time);
            try {
                transient.system().beginGeometricNonlinearityTrial(
                    trial_state_holder.view);
                double synchronized_norm = assembleResidualOnly(
                    trial_state_holder.view,
                    phase,
                    StateSyncPoint::LineSearchTrialResidual);
                if (!std::isfinite(synchronized_norm)) {
                    return synchronized_norm;
                }

                // The Armijo decision must use the exact state that would be
                // committed, including any AcceptedNonlinearState refresh of
                // cuts, curvature, extension data, affine constraints, and
                // transient MPC histories.  This remains a reversible trial:
                // geometry is committed only after the merit test passes, and
                // a rejection invokes RestoredNonlinearState from the base
                // snapshot.
                have_residual = true;
                synchronizeState(trial_state_holder.view,
                                 StateSyncPoint::AcceptedNonlinearState);
                if (options_.accepted_state_sync_invalidates_residual ||
                    !have_residual) {
                    const std::string accepted_phase =
                        std::string(phase) + "_accepted_refresh";
                    synchronized_norm = assembleResidualOnly(
                        trial_state_holder.view,
                        accepted_phase.c_str(),
                        StateSyncPoint::ResidualAssembly);
                }
                have_residual = std::isfinite(synchronized_norm);
                candidate_accepted_state_synchronized = have_residual;
                return synchronized_norm;
            } catch (const std::exception& ex) {
                FE_LOG_INFO(std::string("NewtonSolver: line search trial residual failed; treating trial as rejected. reason='") +
                            ex.what() + "'");
                return std::numeric_limits<double>::infinity();
            }
        };

        double alpha = 1.0;
        double trial_norm = std::numeric_limits<double>::infinity();
        bool accepted = false;
        double best_alpha = 0.0;
        double best_trial_norm = std::numeric_limits<double>::infinity();
        bool have_best_trial = false;
        bool failed_to_reduce = false;
        double full_projected_step_norm =
            std::numeric_limits<double>::infinity();
        bool have_full_projected_step = false;
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
            trial_norm = evaluateLineSearchTrial(alpha, /*phase=*/"line_search");
            if (ls == 0 && alpha == 1.0 && std::isfinite(trial_norm)) {
                copyVector(residual_scratch, history.u());
                axpy(residual_scratch, static_cast<Real>(-1.0), u_backup);
                full_projected_step_norm = residual_scratch.norm();
                have_full_projected_step = true;
            }

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

            // Every rejected alpha is a complete transaction.  Restore the
            // accepted vector, geometry, auxiliary state, cuts, and affine
            // constraints before constructing the next candidate.
            restoreLineSearchBaseState();
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
            if (have_best_trial && std::isfinite(best_trial_norm) &&
                best_trial_norm < r_norm0) {
                alpha = best_alpha;
                // All rejected trials have been restored, including the last
                // one, so recreate the best candidate from the base state.
                // Never reuse a scalar merit while leaving different external
                // cut/constraint state active.
                trial_norm = evaluateLineSearchTrial(
                    alpha, /*phase=*/"line_search_best");
                if (std::isfinite(trial_norm) && trial_norm < r_norm0) {
                    accepted = true;
                    if (oopTraceEnabled()) {
                        std::ostringstream oss;
                        oss << "NewtonSolver: line search did not satisfy Armijo; keeping strictly reducing best trial alpha="
                            << alpha << " ||r(alpha)||=" << trial_norm;
                        traceLog(oss.str());
                    }
                } else {
                    restoreLineSearchBaseState();
                    reverted_to_original = true;
                    failed_to_reduce = true;
                }
            } else {
                alpha = 0.0;
                reverted_to_original = true;
                failed_to_reduce = true;
            }
        }

        // `history.u` now corresponds to the accepted trial, the strictly
        // reducing best fallback, or the restored original iterate.
        auto accepted_state_holder = makeNewtonState(history, solve_time);
        if (reverted_to_original) {
            // The rejected-alpha path already restored geometry before cuts
            // and constraints.  Reassemble the base residual without opening
            // a synthetic alpha=0 geometry transaction.
            current_residual_norm = assembleResidualOnly(
                accepted_state_holder.view,
                /*phase=*/"line_search_reject",
                StateSyncPoint::ResidualAssembly);
            trial_norm = current_residual_norm;
            have_residual = true;
            if (oopTraceEnabled()) {
                std::ostringstream oss;
                oss << "NewtonSolver: line search did not reduce residual; reverting to original iterate"
                    << " ||r||=" << current_residual_norm;
                traceLog(oss.str());
            }
        } else {
            have_residual = std::isfinite(trial_norm);
            FE_THROW_IF(
                !candidate_accepted_state_synchronized || !have_residual,
                systems::InvalidStateException,
                "NewtonSolver: line-search candidate was selected without a stable accepted-state residual");
            try {
                transient.system().acceptGeometricNonlinearityState(
                    accepted_state_holder.view,
                    systems::GeometricNonlinearityUpdatePoint::AcceptedNonlinearState);
            } catch (...) {
                restoreLineSearchBaseState();
                throw;
            }
            if (transient.system().geometricNonlinearityEnabled() ||
                options_.accepted_state_sync_invalidates_residual) {
                have_jacobian = false;
            }
            current_residual_norm = trial_norm;
        }
        line_search_history_transaction_active = false;

        if (failed_to_reduce && options_.line_search_fail_on_no_reduction) {
            report.converged = false;
            report.iterations = it + 1;
            if (have_residual) {
                updateResidualReport();
            }
            if (oopTraceEnabled()) {
                traceLog("NewtonSolver: line search failed to reduce residual; returning nonlinear failure.");
            }
            printNewtonProfile(it + 1);
            return report;
        }

        // A genuinely zero full Newton correction is a valid step-tolerance
        // convergence signal even though Armijo cannot strictly reduce the
        // merit function.  Do not confuse that case with a rejected nonzero
        // correction whose rollback also leaves ||u_new-u_base|| == 0.
        const bool rejected_zero_full_step =
            failed_to_reduce && reverted_to_original &&
            have_full_projected_step &&
            full_projected_step_norm <= options_.step_tolerance;
        if (options_.step_tolerance > 0.0 &&
            (!failed_to_reduce || rejected_zero_full_step)) {
            copyVector(residual_scratch, history.u());
            axpy(residual_scratch, static_cast<Real>(-1.0), u_backup);
            const double step_norm = residual_scratch.norm();
            if (oopTraceEnabled()) {
                std::ostringstream oss;
                oss << "NewtonSolver: actual projected step ||u_new-u_base||="
                    << step_norm
                    << " nominal alpha*||du||=" << (alpha * du_norm)
                    << " full projected ||du||="
                    << full_projected_step_norm
                    << " step_tol=" << options_.step_tolerance;
                traceLog(oss.str());
            }
            if (minIterationsSatisfied(it + 1) &&
                step_norm <= options_.step_tolerance &&
                ((!options_.accepted_state_sync_invalidates_residual &&
                  field_residual_states.empty()) ||
                 tolerancesSatisfied(/*pre_first_update=*/false))) {
                report.converged = true;
                report.iterations = it + 1;
                updateResidualReport();
                if (oopTraceEnabled()) {
                    traceLog(
                        "NewtonSolver: converged by step tolerance on the synchronized residual state.");
                }
                printNewtonProfile(it + 1);
                return report;
            }
        }

        if (!reverted_to_original &&
            minIterationsSatisfied(it + 1) &&
            tolerancesSatisfied(/*pre_first_update=*/false)) {
            report.converged = true;
            report.iterations = it + 1;
            updateResidualReport();
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
        current_newton_iteration = max_it;
        syncHistoryState();
        auto final_state_holder = makeNewtonState(history, solve_time);
        current_residual_norm = assembleResidualOnly(
            final_state_holder.view,
            /*phase=*/"final_check",
            StateSyncPoint::FinalResidualAssembly);
        have_residual = true;
        updateResidualReport();
    }

    report.converged = false;
    report.iterations = max_it;
    if (have_residual && std::isfinite(current_residual_norm)) {
        updateResidualReport();
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

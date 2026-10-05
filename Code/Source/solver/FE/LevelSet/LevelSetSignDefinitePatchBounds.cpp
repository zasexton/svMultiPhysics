#include "LevelSet/LevelSetSignDefinitePatchBounds.h"

#include "Dofs/EntityDofMap.h"

#if FE_HAS_MPI
#include <mpi.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace svmp::FE::level_set {
namespace {

struct Collective {
#if FE_HAS_MPI
    MPI_Comm communicator{MPI_COMM_NULL};
#endif
    bool active{false};
};

[[nodiscard]] Collective collectiveFor(const dofs::DofHandler& dof_handler)
{
    Collective context;
#if FE_HAS_MPI
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    if (initialized != 0) {
        MPI_Finalized(&finalized);
    }
    if (initialized != 0 && finalized == 0 &&
        dof_handler.mpiComm() != MPI_COMM_NULL) {
        int size = 1;
        MPI_Comm_size(dof_handler.mpiComm(), &size);
        context.communicator = dof_handler.mpiComm();
        context.active = size > 1;
    }
#else
    (void)dof_handler;
#endif
    return context;
}

#if FE_HAS_MPI
[[nodiscard]] MPI_Datatype realType() noexcept
{
    if constexpr (std::is_same_v<Real, float>) {
        return MPI_FLOAT;
    }
    return MPI_DOUBLE;
}

[[nodiscard]] int checkedCount(std::size_t size)
{
    if (size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            "level-set sign-definite patch bounds exceed the MPI count range");
    }
    return static_cast<int>(size);
}
#endif

void reduceInPlace(const Collective& context, std::vector<Real>& values,
                   bool minimum)
{
#if FE_HAS_MPI
    if (context.active && !values.empty()) {
        MPI_Allreduce(MPI_IN_PLACE, values.data(), checkedCount(values.size()),
                      realType(), minimum ? MPI_MIN : MPI_MAX,
                      context.communicator);
    }
#else
    (void)context;
    (void)values;
    (void)minimum;
#endif
}

void sumInPlace(const Collective& context,
                std::vector<unsigned long long>& values)
{
#if FE_HAS_MPI
    if (context.active && !values.empty()) {
        MPI_Allreduce(MPI_IN_PLACE, values.data(), checkedCount(values.size()),
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, context.communicator);
    }
#else
    (void)context;
    (void)values;
#endif
}

[[nodiscard]] bool allTrue(const Collective& context, bool local)
{
#if FE_HAS_MPI
    if (context.active) {
        int value = local ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_INT, MPI_MIN,
                      context.communicator);
        return value != 0;
    }
#else
    (void)context;
#endif
    return local;
}

[[nodiscard]] int signClass(Real value, Real isovalue, Real tolerance) noexcept
{
    const Real s = value - isovalue;
    if (s < -tolerance) {
        return -1;
    }
    if (s > tolerance) {
        return 1;
    }
    return 0;
}

} // namespace

LevelSetSignDefinitePatchBoundsResult boundLevelSetOnSignDefinitePatches(
    const assembly::IMeshAccess& mesh,
    const dofs::DofHandler& level_set_dofs,
    Real isovalue,
    Real tolerance,
    std::span<const Real> previous_level_set,
    std::span<const Real> candidate_level_set,
    std::vector<Real>& bounded_level_set)
{
    LevelSetSignDefinitePatchBoundsResult result;
    const auto context = collectiveFor(level_set_dofs);
    const auto n = static_cast<std::size_t>(level_set_dofs.getNumDofs());
    const int dimension = mesh.dimension();

    // ---- validation (collective) -------------------------------------------
    std::string local_failure;
    if (!std::isfinite(isovalue) || !(tolerance >= Real{0.0}) ||
        !std::isfinite(tolerance)) {
        local_failure = "requires a finite isovalue and a nonnegative finite tolerance";
    } else if (dimension != 2 && dimension != 3) {
        local_failure = "requires a 2D or 3D mesh";
    } else if (previous_level_set.size() != n || candidate_level_set.size() != n) {
        local_failure = "received coefficient spans that do not match the field layout";
    } else if (level_set_dofs.getEntityDofMap() == nullptr) {
        local_failure = "requires a vertex-nodal level-set layout";
    }
    for (std::size_t i = 0; local_failure.empty() && i < n; ++i) {
        if (!std::isfinite(previous_level_set[i]) ||
            !std::isfinite(candidate_level_set[i])) {
            local_failure = "received non-finite level-set coefficients";
        }
    }

    // ---- patch data from the owned cells ------------------------------------
    // For every node: the number of cells around it, the number of those cells
    // that are sign definite for it in the negative and in the positive class,
    // and the range of the previous values over its patch.
    std::vector<unsigned long long> counts;
    std::vector<Real> lower;
    std::vector<Real> upper;
    if (local_failure.empty()) {
        try {
            counts.assign(3u * n, 0u);
            lower.assign(n, std::numeric_limits<Real>::infinity());
            upper.assign(n, -std::numeric_limits<Real>::infinity());
            const auto* phi_map = level_set_dofs.getEntityDofMap();
            const ElementType expected =
                dimension == 2 ? ElementType::Triangle3 : ElementType::Tetra4;
            const int corners = dimension + 1;
            std::vector<GlobalIndex> nodes;
            mesh.forEachOwnedCell([&](GlobalIndex cell_id) {
                if (mesh.getCellType(cell_id) != expected ||
                    mesh.getCellGeometryOrder(cell_id) > 1) {
                    throw std::invalid_argument(
                        "supports only affine Triangle3 (2D) and Tetra4 (3D) cells");
                }
                mesh.getCellNodes(cell_id, nodes);
                if (nodes.size() < static_cast<std::size_t>(corners)) {
                    throw std::invalid_argument("found an incomplete cell");
                }
                std::array<std::size_t, 4> dof{};
                std::array<int, 4> previous_class{};
                std::array<int, 4> candidate_class{};
                Real cell_minimum = std::numeric_limits<Real>::infinity();
                Real cell_maximum = -std::numeric_limits<Real>::infinity();
                for (int k = 0; k < corners; ++k) {
                    const auto kk = static_cast<std::size_t>(k);
                    const auto phi_dofs = phi_map->getVertexDofs(nodes[kk]);
                    if (phi_dofs.size() != 1u || phi_dofs.front() < 0 ||
                        static_cast<std::size_t>(phi_dofs.front()) >= n) {
                        throw std::invalid_argument(
                            "requires exactly one scalar level-set DOF per vertex");
                    }
                    dof[kk] = static_cast<std::size_t>(phi_dofs.front());
                    const Real previous = previous_level_set[dof[kk]];
                    previous_class[kk] = signClass(previous, isovalue, tolerance);
                    candidate_class[kk] =
                        signClass(candidate_level_set[dof[kk]], isovalue, tolerance);
                    cell_minimum = std::min(cell_minimum, previous);
                    cell_maximum = std::max(cell_maximum, previous);
                }
                const int cell_class = previous_class[0];
                bool previous_definite = cell_class != 0;
                for (int k = 1; k < corners; ++k) {
                    previous_definite = previous_definite &&
                        previous_class[static_cast<std::size_t>(k)] == cell_class;
                }
                for (int k = 0; k < corners; ++k) {
                    const auto kk = static_cast<std::size_t>(k);
                    const auto i = dof[kk];
                    ++counts[3u * i];
                    lower[i] = std::min(lower[i], cell_minimum);
                    upper[i] = std::max(upper[i], cell_maximum);
                    if (!previous_definite) {
                        continue;
                    }
                    bool others_definite = true;
                    for (int m = 0; m < corners; ++m) {
                        if (m != k) {
                            others_definite = others_definite &&
                                candidate_class[static_cast<std::size_t>(m)] ==
                                    cell_class;
                        }
                    }
                    if (others_definite) {
                        ++counts[3u * i + (cell_class < 0 ? 1u : 2u)];
                    }
                }
            });
        } catch (const std::exception& error) {
            local_failure = error.what();
        }
    }
    if (!allTrue(context, local_failure.empty())) {
        result.diagnostic = local_failure.empty()
                                ? "another rank failed level-set sign-definite patch bound validation"
                                : "level-set sign-definite patch bounds " + local_failure;
        return result;
    }
    sumInPlace(context, counts);
    reduceInPlace(context, lower, /*minimum=*/true);
    reduceInPlace(context, upper, /*minimum=*/false);

    // ---- bound the sign-definite nodes (replicated, identical on all ranks) --
    bounded_level_set.assign(candidate_level_set.begin(), candidate_level_set.end());
    for (std::size_t i = 0; i < n; ++i) {
        const auto cells_around = counts[3u * i];
        if (cells_around == 0u) {
            continue;
        }
        const bool negative = counts[3u * i + 1u] == cells_around;
        const bool positive = counts[3u * i + 2u] == cells_around;
        if (!negative && !positive) {
            continue;
        }
        ++result.sign_definite_dofs;
        const Real candidate = candidate_level_set[i];
        const Real bounded = std::clamp(candidate, lower[i], upper[i]);
        if (bounded != candidate) {
            bounded_level_set[i] = bounded;
            ++result.bounded_dofs;
            result.max_abs_correction =
                std::max(result.max_abs_correction, std::abs(bounded - candidate));
            if (signClass(candidate, isovalue, tolerance) != (negative ? -1 : 1)) {
                ++result.sign_changes_prevented;
            }
        }
    }
    result.applied = result.bounded_dofs > 0u;
    result.success = true;
    result.diagnostic = result.applied ? "bounded" : "within patch bounds";
    return result;
}

} // namespace svmp::FE::level_set

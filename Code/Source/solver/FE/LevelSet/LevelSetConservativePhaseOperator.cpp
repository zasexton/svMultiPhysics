#include "LevelSet/LevelSetConservativePhaseOperator.h"

#include "Basis/BasisFunction.h"
#include "Geometry/MappingFactory.h"
#include "Quadrature/QuadratureFactory.h"
#include "Spaces/FunctionSpace.h"
#include "Systems/FESystem.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#if FE_HAS_MPI
#include <mpi.h>
#endif

namespace svmp::FE::level_set {

namespace detail {

struct LevelSetP1PhaseCollectiveState {
#if FE_HAS_MPI
    MPI_Comm communicator{MPI_COMM_NULL};

    ~LevelSetP1PhaseCollectiveState() noexcept
    {
        if (communicator == MPI_COMM_NULL) {
            return;
        }
        int initialized = 0;
        int finalized = 0;
        MPI_Initialized(&initialized);
        if (initialized != 0) {
            MPI_Finalized(&finalized);
        }
        if (initialized != 0 && finalized == 0) {
            MPI_Comm_free(&communicator);
        }
    }
#endif
};

} // namespace detail

namespace {

using Vector3 = std::array<Real, 3>;

struct MutableGradientEdge {
    Vector3 first_test_second_gradient{};
    Vector3 second_test_first_gradient{};
};

struct PhaseGraphCollectiveContext {
    bool active{false};
    int rank{0};
    int size{1};
#if FE_HAS_MPI
    MPI_Comm communicator{MPI_COMM_NULL};
#endif
};

#if FE_HAS_MPI
struct PackedGradientEdge {
    std::int64_t first_node{-1};
    std::int64_t second_node{-1};
    std::array<Real, 3> first_test_second_gradient{};
    std::array<Real, 3> second_test_first_gradient{};
};

static_assert(std::is_trivially_copyable_v<PackedGradientEdge>);
static_assert(sizeof(PackedGradientEdge) ==
              2u * sizeof(std::int64_t) + 6u * sizeof(Real));

[[nodiscard]] MPI_Datatype mpiRealType() noexcept
{
    if constexpr (std::is_same_v<Real, float>) {
        return MPI_FLOAT;
    } else if constexpr (std::is_same_v<Real, double>) {
        return MPI_DOUBLE;
    }
    return MPI_LONG_DOUBLE;
}

[[nodiscard]] MPI_Datatype mpiUnsigned64Type() noexcept
{
#ifdef MPI_UINT64_T
    return MPI_UINT64_T;
#else
    if constexpr (std::is_same_v<std::uint64_t, unsigned long>) {
        return MPI_UNSIGNED_LONG;
    }
    return MPI_UNSIGNED_LONG_LONG;
#endif
}
#endif

[[nodiscard]] PhaseGraphCollectiveContext phaseGraphCollectiveContext(
    const assembly::IMeshAccess& mesh,
    const dofs::DofHandler& dofs)
{
    PhaseGraphCollectiveContext context;
    context.rank = mesh.parallelRank();
    context.size = mesh.parallelSize();
    if (context.rank < 0 || context.size < 1 ||
        context.rank >= context.size) {
        throw std::invalid_argument(
            "P1 conservative phase graph received invalid mesh rank metadata");
    }
#if FE_HAS_MPI
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    if (initialized != 0) {
        MPI_Finalized(&finalized);
    }
    if (context.size > 1) {
        if (initialized == 0 || finalized != 0 ||
            dofs.mpiComm() == MPI_COMM_NULL) {
            throw std::invalid_argument(
                "P1 conservative phase graph requires an active field communicator for multi-rank assembly");
        }
        int communicator_rank = 0;
        int communicator_size = 1;
        MPI_Comm_rank(dofs.mpiComm(), &communicator_rank);
        MPI_Comm_size(dofs.mpiComm(), &communicator_size);
        if (communicator_rank != context.rank ||
            communicator_size != context.size) {
            throw std::invalid_argument(
                "P1 conservative phase graph mesh and field communicators disagree");
        }
        context.active = true;
        context.communicator = dofs.mpiComm();
    }
#else
    (void)dofs;
    if (context.size > 1) {
        throw std::invalid_argument(
            "P1 conservative phase graph cannot assemble a multi-rank mesh without MPI support");
    }
#endif
    return context;
}

[[nodiscard]] PhaseGraphCollectiveContext phaseStageCollectiveContext(
    const LevelSetP1PhaseTransportGraph& graph)
{
    PhaseGraphCollectiveContext context;
#if FE_HAS_MPI
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    if (initialized != 0) {
        MPI_Finalized(&finalized);
    }
    if (initialized != 0 && finalized == 0 && graph.collective_state &&
        graph.collective_state->communicator != MPI_COMM_NULL) {
        MPI_Comm_rank(graph.collective_state->communicator,
                      &context.rank);
        MPI_Comm_size(graph.collective_state->communicator,
                      &context.size);
        context.active = context.size > 1;
        context.communicator = graph.collective_state->communicator;
        return context;
    }
#endif
    context.rank = graph.parallel_rank;
    context.size = graph.parallel_size;
    if (context.rank < 0 || context.size < 1 ||
        context.rank >= context.size) {
        throw std::invalid_argument(
            "P1 conservative phase stage received invalid graph rank metadata");
    }
#if FE_HAS_MPI
    if (graph.distributed || context.size > 1) {
        throw std::invalid_argument(
            "P1 conservative phase stage requires its active replicated graph communicator");
    }
#else
    if (graph.distributed || context.size > 1) {
        throw std::invalid_argument(
            "P1 conservative phase stage cannot use a distributed graph without MPI support");
    }
#endif
    return context;
}

[[nodiscard]] bool synchronizeLocalFailure(
    const PhaseGraphCollectiveContext& context,
    bool local_success,
    const std::string& local_diagnostic,
    std::string& collective_diagnostic,
    std::string_view operation = "graph")
{
#if FE_HAS_MPI
    if (context.active) {
        const int local_failed_rank =
            local_success ? context.size : context.rank;
        int first_failed_rank = context.size;
        MPI_Allreduce(&local_failed_rank, &first_failed_rank, 1,
                      MPI_INT, MPI_MIN, context.communicator);
        if (first_failed_rank < context.size) {
            constexpr std::size_t maximum_diagnostic_bytes = 4096u;
            int length = 0;
            if (context.rank == first_failed_rank) {
                length = static_cast<int>(std::min(
                    maximum_diagnostic_bytes, local_diagnostic.size()));
            }
            MPI_Bcast(&length, 1, MPI_INT, first_failed_rank,
                      context.communicator);
            std::array<char, maximum_diagnostic_bytes> bytes{};
            if (context.rank == first_failed_rank && length > 0) {
                std::copy_n(local_diagnostic.data(), length, bytes.data());
            }
            if (length > 0) {
                MPI_Bcast(bytes.data(), length, MPI_CHAR,
                          first_failed_rank, context.communicator);
            }
            collective_diagnostic =
                "P1 conservative phase " + std::string(operation) +
                " failed on rank " +
                std::to_string(first_failed_rank) + ": " +
                std::string(bytes.data(), static_cast<std::size_t>(length));
            return false;
        }
    }
#else
    (void)context;
#endif
    if (!local_success) {
        collective_diagnostic = local_diagnostic;
        return false;
    }
    return true;
}

void allReduceRealBufferSum(const PhaseGraphCollectiveContext& context,
                            std::vector<Real>& values)
{
#if FE_HAS_MPI
    if (context.active && !values.empty()) {
        constexpr std::size_t maximum_chunk = 4096u;
        std::array<Real, maximum_chunk> reduced{};
        for (std::size_t offset = 0u; offset < values.size();) {
            const std::size_t count =
                std::min(maximum_chunk, values.size() - offset);
            MPI_Allreduce(
                values.data() + static_cast<std::ptrdiff_t>(offset),
                reduced.data(),
                static_cast<int>(count), mpiRealType(), MPI_SUM,
                context.communicator);
            std::copy_n(
                reduced.data(), count,
                values.data() + static_cast<std::ptrdiff_t>(offset));
            offset += count;
        }
    }
#else
    (void)context;
    (void)values;
#endif
}

void allReduceVector3Sum(const PhaseGraphCollectiveContext& context,
                         std::vector<Vector3>& values)
{
#if FE_HAS_MPI
    if (context.active && !values.empty()) {
        constexpr std::size_t maximum_vector_chunk = 512u;
        std::array<Real, 3u * maximum_vector_chunk> local{};
        std::array<Real, 3u * maximum_vector_chunk> reduced{};
        for (std::size_t offset = 0u; offset < values.size();) {
            const std::size_t count =
                std::min(maximum_vector_chunk, values.size() - offset);
            for (std::size_t index = 0u; index < count; ++index) {
                for (std::size_t component = 0u; component < 3u;
                     ++component) {
                    local[3u * index + component] =
                        values[offset + index][component];
                }
            }
            MPI_Allreduce(local.data(), reduced.data(),
                          static_cast<int>(3u * count), mpiRealType(),
                          MPI_SUM, context.communicator);
            for (std::size_t index = 0u; index < count; ++index) {
                for (std::size_t component = 0u; component < 3u;
                     ++component) {
                    values[offset + index][component] =
                        reduced[3u * index + component];
                }
            }
            offset += count;
        }
    }
#else
    (void)context;
    (void)values;
#endif
}

#if FE_HAS_MPI
[[nodiscard]] Real allReduceReal(const PhaseGraphCollectiveContext& context,
                                 Real local,
                                 MPI_Op operation)
{
    if (context.active) {
        Real global{0.0};
        MPI_Allreduce(&local, &global, 1, mpiRealType(), operation,
                      context.communicator);
        return global;
    }
    return local;
}

[[nodiscard]] int allReduceInt(const PhaseGraphCollectiveContext& context,
                               int local,
                               MPI_Op operation)
{
    if (context.active) {
        int global = 0;
        MPI_Allreduce(&local, &global, 1, MPI_INT, operation,
                      context.communicator);
        return global;
    }
    return local;
}

[[nodiscard]] std::uint64_t allReduceUnsigned64(
    const PhaseGraphCollectiveContext& context,
    std::uint64_t local,
    MPI_Op operation
)
{
    if (context.active) {
        std::uint64_t global = 0u;
        MPI_Allreduce(&local, &global, 1, mpiUnsigned64Type(), operation,
                      context.communicator);
        return global;
    }
    return local;
}
#endif

[[nodiscard]] Real allReduceRealSum(
    const PhaseGraphCollectiveContext& context, Real local)
{
#if FE_HAS_MPI
    return allReduceReal(context, local, MPI_SUM);
#else
    (void)context;
    return local;
#endif
}

[[nodiscard]] Real allReduceRealMin(
    const PhaseGraphCollectiveContext& context, Real local)
{
#if FE_HAS_MPI
    return allReduceReal(context, local, MPI_MIN);
#else
    (void)context;
    return local;
#endif
}

[[nodiscard]] Real allReduceRealMax(
    const PhaseGraphCollectiveContext& context, Real local)
{
#if FE_HAS_MPI
    return allReduceReal(context, local, MPI_MAX);
#else
    (void)context;
    return local;
#endif
}

[[nodiscard]] int allReduceIntMin(
    const PhaseGraphCollectiveContext& context, int local)
{
#if FE_HAS_MPI
    return allReduceInt(context, local, MPI_MIN);
#else
    (void)context;
    return local;
#endif
}

[[nodiscard]] int allReduceIntMax(
    const PhaseGraphCollectiveContext& context, int local)
{
#if FE_HAS_MPI
    return allReduceInt(context, local, MPI_MAX);
#else
    (void)context;
    return local;
#endif
}

struct ReplicatedRealMismatch {
    bool present{false};
    std::size_t index{0u};
    Real minimum{0.0};
    Real maximum{0.0};
};

template <typename ValueAt>
[[nodiscard]] ReplicatedRealMismatch firstReplicatedRealMismatchValues(
    const PhaseGraphCollectiveContext& context,
    std::size_t value_count,
    ValueAt value_at)
{
    static_assert(sizeof(Real) == sizeof(std::uint64_t));
    ReplicatedRealMismatch mismatch;
#if FE_HAS_MPI
    if (context.active && value_count != 0u) {
        constexpr std::size_t maximum_chunk = 4096u;
        std::array<std::uint64_t, maximum_chunk> local{};
        std::array<std::uint64_t, maximum_chunk> minimum{};
        std::array<std::uint64_t, maximum_chunk> maximum{};
        for (std::size_t offset = 0u; offset < value_count;) {
            const std::size_t count =
                std::min(maximum_chunk, value_count - offset);
            for (std::size_t index = 0u; index < count; ++index) {
                local[index] =
                    std::bit_cast<std::uint64_t>(value_at(offset + index));
            }
            MPI_Allreduce(local.data(), minimum.data(),
                          static_cast<int>(count), mpiUnsigned64Type(), MPI_MIN,
                          context.communicator);
            MPI_Allreduce(local.data(), maximum.data(),
                          static_cast<int>(count), mpiUnsigned64Type(), MPI_MAX,
                          context.communicator);
            for (std::size_t index = 0u; index < count; ++index) {
                if (minimum[index] != maximum[index]) {
                    mismatch.present = true;
                    mismatch.index = offset + index;
                    mismatch.minimum =
                        std::bit_cast<Real>(minimum[index]);
                    mismatch.maximum =
                        std::bit_cast<Real>(maximum[index]);
                    return mismatch;
                }
            }
            offset += count;
        }
    }
#else
    (void)context;
    (void)value_count;
    (void)value_at;
#endif
    return mismatch;
}

[[nodiscard]] ReplicatedRealMismatch firstReplicatedRealMismatch(
    const PhaseGraphCollectiveContext& context,
    std::span<const Real> local)
{
    return firstReplicatedRealMismatchValues(
        context, local.size(),
        [local](std::size_t index) { return local[index]; });
}

[[nodiscard]] std::string replicatedRealMismatchDiagnostic(
    std::string_view quantity,
    std::string location,
    const ReplicatedRealMismatch& mismatch)
{
    std::ostringstream stream;
    stream << std::setprecision(std::numeric_limits<Real>::max_digits10)
           << "P1 conservative phase stage requires identical "
           << quantity << " on every rank; first mismatch at "
           << location << " (minimum=" << mismatch.minimum
           << ", maximum=" << mismatch.maximum << ')';
    return stream.str();
}

[[nodiscard]] bool replicatedStageInputsMatch(
    const PhaseGraphCollectiveContext& context,
    std::span<const Real> previous_liquid_indicator,
    std::span<const Real> lower_liquid_indicator,
    std::span<const Real> upper_liquid_indicator,
    std::span<const std::array<Real, 3>> nodal_velocity,
    Real time_step,
    const LevelSetP1PhaseStageOptions& options,
    std::string& diagnostic)
{
    const std::array<std::pair<std::string_view, std::size_t>, 4>
        replicated_sizes{{
            {"previous liquid indicator", previous_liquid_indicator.size()},
            {"lower liquid-indicator bound", lower_liquid_indicator.size()},
            {"upper liquid-indicator bound", upper_liquid_indicator.size()},
            {"nodal velocity", nodal_velocity.size()},
        }};
    for (const auto& [name, size] : replicated_sizes) {
        const auto local_size = static_cast<std::uint64_t>(size);
        auto minimum_size = local_size;
        auto maximum_size = local_size;
#if FE_HAS_MPI
        minimum_size =
            allReduceUnsigned64(context, local_size, MPI_MIN);
        maximum_size =
            allReduceUnsigned64(context, local_size, MPI_MAX);
#endif
        if (minimum_size != maximum_size) {
            diagnostic =
                "P1 conservative phase stage requires identical " +
                std::string(name) + " sizes on every rank";
            return false;
        }
    }

    const auto require_nodal_values =
        [&](std::span<const Real> values,
            std::string_view quantity) {
            const auto mismatch =
                firstReplicatedRealMismatch(context, values);
            if (!mismatch.present) {
                return true;
            }
            diagnostic = replicatedRealMismatchDiagnostic(
                quantity, "node " + std::to_string(mismatch.index),
                mismatch);
            return false;
        };
    if (!require_nodal_values(previous_liquid_indicator,
                              "previous liquid indicator") ||
        !require_nodal_values(lower_liquid_indicator,
                              "lower liquid-indicator bound") ||
        !require_nodal_values(upper_liquid_indicator,
                              "upper liquid-indicator bound")) {
        return false;
    }

    if (nodal_velocity.size() >
        std::numeric_limits<std::size_t>::max() / 3u) {
        diagnostic =
            "P1 conservative phase stage nodal velocity size exceeds the exact replicated-comparison range";
        return false;
    }
    const auto velocity_mismatch =
        firstReplicatedRealMismatchValues(
            context, nodal_velocity.size() * 3u,
            [nodal_velocity](std::size_t index) {
                return nodal_velocity[index / 3u][index % 3u];
            });
    if (velocity_mismatch.present) {
        diagnostic = replicatedRealMismatchDiagnostic(
            "nodal velocity",
            "node " + std::to_string(velocity_mismatch.index / 3u) +
                ", component " +
                std::to_string(velocity_mismatch.index % 3u),
            velocity_mismatch);
        return false;
    }

    const auto require_scalar =
        [&](Real value,
            std::string_view quantity,
            std::string_view location) {
            const auto mismatch = firstReplicatedRealMismatch(
                context, std::span<const Real>(&value, 1u));
            if (!mismatch.present) {
                return true;
            }
            diagnostic = replicatedRealMismatchDiagnostic(
                quantity, std::string(location), mismatch);
            return false;
        };
    if (!require_scalar(time_step, "time step", "stage scalar") ||
        !require_scalar(options.invariant_tolerance,
                        "invariant tolerance", "stage option") ||
        !require_scalar(options.component_activity_tolerance,
                        "component activity tolerance", "stage option") ||
        !require_scalar(options.maximum_courant,
                        "maximum Courant number", "stage option")) {
        return false;
    }

    const int minimum_enforce = allReduceIntMin(
        context, options.enforce_courant_limit ? 1 : 0);
    const int maximum_enforce = allReduceIntMax(
        context, options.enforce_courant_limit ? 1 : 0);
    if (minimum_enforce != maximum_enforce) {
        diagnostic =
            "P1 conservative phase stage requires identical Courant-limit enforcement on every rank (minimum=false, maximum=true)";
        return false;
    }
    const int minimum_constant = allReduceIntMin(
        context, options.require_constant_preservation ? 1 : 0);
    const int maximum_constant = allReduceIntMax(
        context, options.require_constant_preservation ? 1 : 0);
    if (minimum_constant != maximum_constant) {
        diagnostic =
            "P1 conservative phase stage requires identical constant-preservation requirements on every rank (minimum=false, maximum=true)";
        return false;
    }
    return true;
}

[[nodiscard]] std::uint64_t allReduceUnsigned64Sum(
    const PhaseGraphCollectiveContext& context, std::uint64_t local)
{
#if FE_HAS_MPI
    return allReduceUnsigned64(context, local, MPI_SUM);
#else
    (void)context;
    return local;
#endif
}

[[nodiscard]] std::uint64_t allReduceUnsigned64Min(
    const PhaseGraphCollectiveContext& context, std::uint64_t local)
{
#if FE_HAS_MPI
    return allReduceUnsigned64(context, local, MPI_MIN);
#else
    (void)context;
    return local;
#endif
}

[[nodiscard]] std::uint64_t allReduceUnsigned64Max(
    const PhaseGraphCollectiveContext& context, std::uint64_t local)
{
#if FE_HAS_MPI
    return allReduceUnsigned64(context, local, MPI_MAX);
#else
    (void)context;
    return local;
#endif
}

void allReduceIntBufferMinMax(
    const PhaseGraphCollectiveContext& context,
    const std::vector<int>& local,
    std::vector<int>& minimum,
    std::vector<int>& maximum)
{
    if (minimum.size() != local.size() || maximum.size() != local.size()) {
        throw std::invalid_argument(
            "P1 conservative phase graph owner-reduction buffers have inconsistent sizes");
    }
#if FE_HAS_MPI
    if (context.active && !local.empty()) {
        constexpr std::size_t maximum_chunk = 4096u;
        std::array<int, maximum_chunk> reduced_minimum{};
        std::array<int, maximum_chunk> reduced_maximum{};
        for (std::size_t offset = 0u; offset < local.size();) {
            const std::size_t count =
                std::min(maximum_chunk, local.size() - offset);
            const auto displacement = static_cast<std::ptrdiff_t>(offset);
            MPI_Allreduce(local.data() + displacement,
                          reduced_minimum.data(),
                          static_cast<int>(count), MPI_INT, MPI_MIN,
                          context.communicator);
            MPI_Allreduce(local.data() + displacement,
                          reduced_maximum.data(),
                          static_cast<int>(count), MPI_INT, MPI_MAX,
                          context.communicator);
            std::copy_n(reduced_minimum.data(), count,
                        minimum.data() + displacement);
            std::copy_n(reduced_maximum.data(), count,
                        maximum.data() + displacement);
            offset += count;
        }
    } else {
        std::copy(local.begin(), local.end(), minimum.begin());
        std::copy(local.begin(), local.end(), maximum.begin());
    }
#else
    (void)context;
    std::copy(local.begin(), local.end(), minimum.begin());
    std::copy(local.begin(), local.end(), maximum.begin());
#endif
}

[[nodiscard]] std::map<std::pair<GlobalIndex, GlobalIndex>,
                       MutableGradientEdge>
globalizeGradientEdges(
    const PhaseGraphCollectiveContext& context,
    const std::map<std::pair<GlobalIndex, GlobalIndex>, MutableGradientEdge>&
        local_edges)
{
#if FE_HAS_MPI
    if (context.active) {
        const int local_capacity_valid =
            local_edges.size() <=
                    static_cast<std::size_t>(
                        std::numeric_limits<int>::max()) /
                        sizeof(PackedGradientEdge)
                ? 1
                : 0;
        int global_capacity_valid = 0;
        MPI_Allreduce(&local_capacity_valid, &global_capacity_valid, 1,
                      MPI_INT, MPI_MIN, context.communicator);
        if (global_capacity_valid == 0) {
            throw std::overflow_error(
                "P1 conservative phase graph local edge snapshot exceeds MPI count capacity");
        }
        std::vector<PackedGradientEdge> packed;
        std::vector<int> counts;
        bool local_snapshot_preparation_success = true;
        std::string local_snapshot_preparation_diagnostic;
        try {
            packed.reserve(local_edges.size());
            for (const auto& [endpoints, edge] : local_edges) {
                packed.push_back(PackedGradientEdge{
                    .first_node =
                        static_cast<std::int64_t>(endpoints.first),
                    .second_node =
                        static_cast<std::int64_t>(endpoints.second),
                    .first_test_second_gradient =
                        edge.first_test_second_gradient,
                    .second_test_first_gradient =
                        edge.second_test_first_gradient,
                });
            }
            counts.assign(static_cast<std::size_t>(context.size), 0);
        } catch (const std::exception& exception) {
            local_snapshot_preparation_success = false;
            local_snapshot_preparation_diagnostic = exception.what();
        }
        std::string collective_snapshot_preparation_diagnostic;
        if (!synchronizeLocalFailure(
                context,
                local_snapshot_preparation_success,
                local_snapshot_preparation_diagnostic,
                collective_snapshot_preparation_diagnostic,
                "graph edge-snapshot preparation")) {
            throw std::runtime_error(
                collective_snapshot_preparation_diagnostic);
        }
        const std::size_t local_bytes_size =
            packed.size() * sizeof(PackedGradientEdge);
        const int local_bytes = static_cast<int>(local_bytes_size);
        MPI_Allgather(&local_bytes, 1, MPI_INT, counts.data(), 1,
                      MPI_INT, context.communicator);
        std::vector<int> displacements;
        std::vector<std::byte> gathered;
        std::size_t total_bytes = 0u;
        bool local_gather_preparation_success = true;
        std::string local_gather_preparation_diagnostic;
        try {
            displacements.assign(counts.size(), 0);
            for (std::size_t rank = 0; rank < counts.size(); ++rank) {
                if (counts[rank] < 0 ||
                    counts[rank] %
                            static_cast<int>(sizeof(PackedGradientEdge)) !=
                        0 ||
                    total_bytes >
                        static_cast<std::size_t>(
                            std::numeric_limits<int>::max()) -
                            static_cast<std::size_t>(counts[rank])) {
                    throw std::overflow_error(
                        "P1 conservative phase graph aggregate edge snapshot exceeds MPI displacement capacity");
                }
                displacements[rank] = static_cast<int>(total_bytes);
                total_bytes += static_cast<std::size_t>(counts[rank]);
            }
            gathered.resize(total_bytes);
        } catch (const std::exception& exception) {
            local_gather_preparation_success = false;
            local_gather_preparation_diagnostic = exception.what();
        }
        std::string collective_gather_preparation_diagnostic;
        if (!synchronizeLocalFailure(
                context,
                local_gather_preparation_success,
                local_gather_preparation_diagnostic,
                collective_gather_preparation_diagnostic,
                "graph edge-gather preparation")) {
            throw std::runtime_error(
                collective_gather_preparation_diagnostic);
        }
        MPI_Allgatherv(
            packed.empty() ? nullptr : packed.data(), local_bytes, MPI_BYTE,
            gathered.empty() ? nullptr : gathered.data(), counts.data(),
            displacements.data(), MPI_BYTE, context.communicator);

        std::map<std::pair<GlobalIndex, GlobalIndex>, MutableGradientEdge>
            global_edges;
        for (std::size_t offset = 0u; offset < gathered.size();
             offset += sizeof(PackedGradientEdge)) {
            PackedGradientEdge packed_edge{};
            std::memcpy(&packed_edge, gathered.data() + offset,
                        sizeof(PackedGradientEdge));
            const GlobalIndex first =
                static_cast<GlobalIndex>(packed_edge.first_node);
            const GlobalIndex second =
                static_cast<GlobalIndex>(packed_edge.second_node);
            if (first < 0 || first >= second) {
                throw std::runtime_error(
                    "P1 conservative phase graph received a malformed distributed edge record");
            }
            auto& edge = global_edges[{first, second}];
            for (std::size_t d = 0; d < 3u; ++d) {
                edge.first_test_second_gradient[d] +=
                    packed_edge.first_test_second_gradient[d];
                edge.second_test_first_gradient[d] +=
                    packed_edge.second_test_first_gradient[d];
            }
        }
        return global_edges;
    }
#else
    (void)context;
#endif
    return local_edges;
}

constexpr std::uint64_t kPhaseStageHashOffset =
    UINT64_C(14695981039346656037);
constexpr std::uint64_t kPhaseStageHashPrime = UINT64_C(1099511628211);

void mixPhaseStageHashByte(std::uint64_t& hash,
                           std::uint8_t byte) noexcept
{
    hash ^= static_cast<std::uint64_t>(byte);
    hash *= kPhaseStageHashPrime;
}

void mixPhaseStageHashWord(std::uint64_t& hash,
                           std::uint64_t word) noexcept
{
    for (unsigned int byte = 0u; byte < 8u; ++byte) {
        mixPhaseStageHashByte(
            hash,
            static_cast<std::uint8_t>((word >> (8u * byte)) & 0xffu));
    }
}

void mixPhaseStageHashSize(std::uint64_t& hash,
                           std::size_t value) noexcept
{
    mixPhaseStageHashWord(hash, static_cast<std::uint64_t>(value));
}

void mixPhaseStageHashSigned(std::uint64_t& hash,
                             std::int64_t value) noexcept
{
    mixPhaseStageHashWord(hash, static_cast<std::uint64_t>(value));
}

void mixPhaseStageHashBool(std::uint64_t& hash, bool value) noexcept
{
    mixPhaseStageHashWord(hash, value ? 1u : 0u);
}

void mixPhaseStageHashReal(std::uint64_t& hash, Real value) noexcept
{
    static_assert(sizeof(Real) == sizeof(std::uint64_t));
    static_assert(std::is_trivially_copyable_v<Real>);
    mixPhaseStageHashWord(hash, std::bit_cast<std::uint64_t>(value));
}

[[nodiscard]] std::uint64_t finishPhaseStageHash(
    std::uint64_t hash) noexcept
{
    return hash == 0u ? 1u : hash;
}

[[nodiscard]] bool exactPhaseStageRealEqual(Real left,
                                            Real right) noexcept
{
    static_assert(sizeof(Real) == sizeof(std::uint64_t));
    return std::bit_cast<std::uint64_t>(left) ==
           std::bit_cast<std::uint64_t>(right);
}

[[nodiscard]] bool samePhaseStageOptionsExact(
    const LevelSetP1PhaseStageOptions& left,
    const LevelSetP1PhaseStageOptions& right) noexcept
{
    return exactPhaseStageRealEqual(
               left.invariant_tolerance, right.invariant_tolerance) &&
           exactPhaseStageRealEqual(
               left.component_activity_tolerance,
               right.component_activity_tolerance) &&
           exactPhaseStageRealEqual(
               left.maximum_courant, right.maximum_courant) &&
           left.enforce_courant_limit == right.enforce_courant_limit &&
           left.require_constant_preservation ==
               right.require_constant_preservation;
}

[[nodiscard]] bool samePhaseGraphIdentity(
    const LevelSetP1PhaseGraphIdentity& left,
    const LevelSetP1PhaseGraphIdentity& right) noexcept
{
    return left.dimension == right.dimension &&
           left.nodes == right.nodes && left.edges == right.edges &&
           left.geometry_revision == right.geometry_revision &&
           left.topology_revision == right.topology_revision &&
           left.ownership_revision == right.ownership_revision &&
           left.numbering_revision == right.numbering_revision &&
           left.dof_layout_revision == right.dof_layout_revision &&
           left.content_revision == right.content_revision;
}

void mixPhaseFluxComponentLedger(
    std::uint64_t& hash,
    const LevelSetPhaseFluxComponentLedger& component) noexcept
{
    mixPhaseStageHashSigned(hash, component.component_id);
    mixPhaseStageHashSize(hash, component.nodes);
    mixPhaseStageHashReal(hash, component.previous_liquid_measure);
    mixPhaseStageHashReal(hash, component.low_order_liquid_measure);
    mixPhaseStageHashReal(hash, component.raw_target_liquid_measure);
    mixPhaseStageHashReal(hash, component.limited_liquid_measure);
    mixPhaseStageHashReal(hash, component.physical_boundary_mass_transfer);
    mixPhaseStageHashReal(hash, component.discrete_divergence_mass_source);
    mixPhaseStageHashReal(hash, component.low_order_interior_mass_transfer);
    mixPhaseStageHashReal(hash, component.raw_antidiffusive_mass_transfer);
    mixPhaseStageHashReal(hash, component.limited_antidiffusive_mass_transfer);
    mixPhaseStageHashReal(hash, component.low_order_balance_residual);
    mixPhaseStageHashReal(hash, component.raw_target_balance_residual);
    mixPhaseStageHashReal(hash, component.limited_balance_residual);
}

void mixPhaseFluxNodeLedger(
    std::uint64_t& hash,
    const LevelSetPhaseFluxNodeLedger& node) noexcept
{
    mixPhaseStageHashSigned(hash, node.node);
    mixPhaseStageHashReal(hash, node.lumped_control_volume);
    mixPhaseStageHashReal(hash, node.previous_liquid_indicator);
    mixPhaseStageHashReal(hash, node.lower_liquid_indicator);
    mixPhaseStageHashReal(hash, node.upper_liquid_indicator);
    mixPhaseStageHashReal(hash, node.physical_boundary_mass_transfer);
    mixPhaseStageHashReal(hash, node.discrete_divergence_mass_source);
    mixPhaseStageHashReal(hash, node.low_order_interior_mass_transfer);
    mixPhaseStageHashReal(hash, node.raw_antidiffusive_mass_transfer);
    mixPhaseStageHashReal(hash, node.limited_antidiffusive_mass_transfer);
    mixPhaseStageHashReal(hash, node.positive_raw_antidiffusive_mass);
    mixPhaseStageHashReal(hash, node.negative_raw_antidiffusive_mass);
    mixPhaseStageHashReal(hash, node.positive_correction_factor);
    mixPhaseStageHashReal(hash, node.negative_correction_factor);
    mixPhaseStageHashReal(hash, node.low_order_liquid_indicator);
    mixPhaseStageHashReal(hash, node.raw_target_liquid_indicator);
    mixPhaseStageHashReal(hash, node.limited_liquid_indicator);
    mixPhaseStageHashReal(hash, node.low_order_local_mass_balance_residual);
    mixPhaseStageHashReal(hash, node.raw_target_local_mass_balance_residual);
    mixPhaseStageHashReal(hash, node.local_mass_balance_residual);
}

void mixPhaseFluxEdgeLedger(
    std::uint64_t& hash,
    const LevelSetPhaseFluxEdgeLedger& edge) noexcept
{
    mixPhaseStageHashSigned(hash, edge.first_node);
    mixPhaseStageHashSigned(hash, edge.second_node);
    mixPhaseStageHashReal(hash, edge.low_order_mass_transfer);
    mixPhaseStageHashReal(hash, edge.raw_antidiffusive_mass_transfer);
    mixPhaseStageHashReal(hash, edge.correction_factor);
    mixPhaseStageHashReal(hash, edge.limited_antidiffusive_mass_transfer);
    mixPhaseStageHashReal(hash, edge.low_order_pair_cancellation_residual);
    mixPhaseStageHashReal(hash, edge.raw_pair_cancellation_residual);
    mixPhaseStageHashReal(hash, edge.limited_pair_cancellation_residual);
}

[[nodiscard]] bool finitePhaseFluxComponent(
    const LevelSetPhaseFluxComponentLedger& component) noexcept
{
    const std::array<Real, 12> values{{
        component.previous_liquid_measure,
        component.low_order_liquid_measure,
        component.raw_target_liquid_measure,
        component.limited_liquid_measure,
        component.physical_boundary_mass_transfer,
        component.discrete_divergence_mass_source,
        component.low_order_interior_mass_transfer,
        component.raw_antidiffusive_mass_transfer,
        component.limited_antidiffusive_mass_transfer,
        component.low_order_balance_residual,
        component.raw_target_balance_residual,
        component.limited_balance_residual,
    }};
    return std::all_of(values.begin(), values.end(), [](Real value) {
        return std::isfinite(value);
    });
}

[[nodiscard]] bool completePhaseFluxLedger(
    const LevelSetP1PhaseTransportGraph& graph,
    std::span<const Real> previous_liquid_indicator,
    const LevelSetP1PhaseTransportStageResult& stage,
    std::string& diagnostic)
{
    const auto& correction = stage.correction;
    const auto fail = [&](std::string_view reason) {
        diagnostic = "P1 conservative phase split-stage " +
                     std::string(reason);
        return false;
    };
    if (!stage.success || !stage.courant_satisfied ||
        !stage.low_order_coefficients_nonnegative ||
        !stage.strong_form_decomposition_satisfied ||
        !stage.replicated_stage_inputs_satisfied || !correction.success ||
        !correction.low_order_bounds_satisfied ||
        !correction.limited_bounds_satisfied ||
        !correction.interior_cancellation_satisfied ||
        !correction.local_balance_satisfied ||
        !correction.global_balance_satisfied ||
        !correction.component_balance_satisfied ||
        !correction.component_measure_closure_satisfied ||
        (correction.constant_preservation_required &&
         !correction.constant_preservation_satisfied)) {
        return fail("requires a successful replicated transport stage");
    }
    const auto& executed_options = stage.executed_options;
    if (!std::isfinite(executed_options.invariant_tolerance) ||
        executed_options.invariant_tolerance < Real{0.0} ||
        !std::isfinite(executed_options.component_activity_tolerance) ||
        !(executed_options.component_activity_tolerance > Real{0.0}) ||
        executed_options.component_activity_tolerance > Real{1.0} ||
        !std::isfinite(executed_options.maximum_courant) ||
        !(executed_options.maximum_courant > Real{0.0}) ||
        executed_options.maximum_courant > Real{1.0} ||
        !exactPhaseStageRealEqual(
            executed_options.component_activity_tolerance,
            correction.component_activity_tolerance) ||
        executed_options.require_constant_preservation !=
            correction.constant_preservation_required) {
        return fail("requires complete and consistent executed stage options");
    }
    if (graph.nodes == 0u ||
        graph.lumped_control_volume.size() != graph.nodes ||
        graph.diagonal_gradient.size() != graph.nodes ||
        graph.boundary_column_sum.size() != graph.nodes ||
        previous_liquid_indicator.size() != graph.nodes ||
        stage.sampled_nodal_velocity.size() != graph.nodes ||
        stage.nodal_courant.size() != graph.nodes ||
        stage.physical_boundary_mass_transfer.size() != graph.nodes ||
        stage.discrete_divergence_mass_source.size() != graph.nodes ||
        correction.nodes.size() != graph.nodes ||
        correction.node_component_ids.size() != graph.nodes ||
        stage.flux_edges.size() != graph.edges.size() ||
        correction.edges.size() != graph.edges.size()) {
        return fail("requires complete nodal and edge ledger coverage");
    }
    const std::array<Real, 37> summary_values{{
        stage.time_step,
        stage.maximum_courant,
        stage.minimum_low_order_coefficient,
        stage.maximum_strong_form_decomposition_residual,
        correction.total_previous_liquid_measure,
        correction.total_low_order_liquid_measure,
        correction.total_raw_target_liquid_measure,
        correction.total_limited_liquid_measure,
        correction.total_physical_boundary_mass_transfer,
        correction.total_discrete_divergence_mass_source,
        correction.low_order_nodal_cancellation_residual,
        correction.raw_nodal_cancellation_residual,
        correction.limited_nodal_cancellation_residual,
        correction.maximum_edge_pair_cancellation_residual,
        correction.maximum_low_order_local_mass_balance_residual,
        correction.maximum_raw_target_local_mass_balance_residual,
        correction.maximum_local_mass_balance_residual,
        correction.maximum_component_balance_residual,
        correction.component_activity_tolerance,
        correction.previous_component_measure_closure_residual,
        correction.low_order_component_measure_closure_residual,
        correction.raw_target_component_measure_closure_residual,
        correction.limited_component_measure_closure_residual,
        correction.boundary_component_transfer_closure_residual,
        correction.divergence_component_source_closure_residual,
        correction.low_order_component_transfer_closure_residual,
        correction.raw_component_transfer_closure_residual,
        correction.limited_component_transfer_closure_residual,
        correction.low_order_global_mass_balance_residual,
        correction.raw_target_global_mass_balance_residual,
        correction.global_mass_balance_residual,
        correction.maximum_constant_preservation_error,
        correction.minimum_low_order_liquid_indicator,
        correction.maximum_low_order_liquid_indicator,
        correction.minimum_raw_target_liquid_indicator,
        correction.maximum_raw_target_liquid_indicator,
        correction.minimum_limited_liquid_indicator,
    }};
    if (!std::all_of(
            summary_values.begin(), summary_values.end(), [](Real value) {
                return std::isfinite(value);
            }) ||
        !std::isfinite(correction.maximum_limited_liquid_indicator) ||
        !(stage.time_step > Real{0.0})) {
        return fail("requires finite stage and correction summary values");
    }
    for (std::size_t index = 0u; index < graph.nodes; ++index) {
        const auto& node = correction.nodes[index];
        const std::array<Real, 26> values{{
            previous_liquid_indicator[index],
            stage.sampled_nodal_velocity[index][0],
            stage.sampled_nodal_velocity[index][1],
            stage.sampled_nodal_velocity[index][2],
            stage.nodal_courant[index],
            stage.physical_boundary_mass_transfer[index],
            stage.discrete_divergence_mass_source[index],
            node.lumped_control_volume,
            node.previous_liquid_indicator,
            node.lower_liquid_indicator,
            node.upper_liquid_indicator,
            node.physical_boundary_mass_transfer,
            node.discrete_divergence_mass_source,
            node.low_order_interior_mass_transfer,
            node.raw_antidiffusive_mass_transfer,
            node.limited_antidiffusive_mass_transfer,
            node.positive_raw_antidiffusive_mass,
            node.negative_raw_antidiffusive_mass,
            node.positive_correction_factor,
            node.negative_correction_factor,
            node.low_order_liquid_indicator,
            node.raw_target_liquid_indicator,
            node.limited_liquid_indicator,
            node.low_order_local_mass_balance_residual,
            node.raw_target_local_mass_balance_residual,
            node.local_mass_balance_residual,
        }};
        if (node.node != static_cast<GlobalIndex>(index) ||
            !std::all_of(values.begin(), values.end(), [](Real value) {
                return std::isfinite(value);
            }) ||
            !exactPhaseStageRealEqual(
                previous_liquid_indicator[index],
                node.previous_liquid_indicator) ||
            !exactPhaseStageRealEqual(
                graph.lumped_control_volume[index],
                node.lumped_control_volume) ||
            !exactPhaseStageRealEqual(
                stage.physical_boundary_mass_transfer[index],
                node.physical_boundary_mass_transfer) ||
            !exactPhaseStageRealEqual(
                stage.discrete_divergence_mass_source[index],
                node.discrete_divergence_mass_source)) {
            return fail("found an incomplete or inconsistent nodal ledger");
        }
    }
    for (std::size_t index = 0u; index < graph.edges.size(); ++index) {
        const auto& graph_edge = graph.edges[index];
        const auto& input_edge = stage.flux_edges[index];
        const auto& edge = correction.edges[index];
        const std::array<Real, 9> values{{
            input_edge.low_order_mass_transfer,
            input_edge.raw_antidiffusive_mass_transfer,
            edge.low_order_mass_transfer,
            edge.raw_antidiffusive_mass_transfer,
            edge.correction_factor,
            edge.limited_antidiffusive_mass_transfer,
            edge.low_order_pair_cancellation_residual,
            edge.raw_pair_cancellation_residual,
            edge.limited_pair_cancellation_residual,
        }};
        if (graph_edge.first_node < 0 || graph_edge.second_node < 0 ||
            graph_edge.first_node >= graph_edge.second_node ||
            static_cast<std::size_t>(graph_edge.second_node) >=
                graph.nodes ||
            input_edge.first_node != graph_edge.first_node ||
            input_edge.second_node != graph_edge.second_node ||
            edge.first_node != input_edge.first_node ||
            edge.second_node != input_edge.second_node ||
            !std::all_of(values.begin(), values.end(), [](Real value) {
                return std::isfinite(value);
            }) ||
            !exactPhaseStageRealEqual(
                input_edge.low_order_mass_transfer,
                edge.low_order_mass_transfer) ||
            !exactPhaseStageRealEqual(
                input_edge.raw_antidiffusive_mass_transfer,
                edge.raw_antidiffusive_mass_transfer)) {
            return fail("found an incomplete or inconsistent edge ledger");
        }
    }
    if (!std::all_of(correction.components.begin(),
                     correction.components.end(), finitePhaseFluxComponent) ||
        (correction.subthreshold_component_present &&
         !finitePhaseFluxComponent(correction.subthreshold_component))) {
        return fail("found a non-finite component ledger");
    }
    GlobalIndex previous_component_id = INVALID_GLOBAL_INDEX;
    for (const auto& component : correction.components) {
        if (component.component_id < 0 ||
            static_cast<std::size_t>(component.component_id) >= graph.nodes ||
            component.nodes == 0u ||
            (previous_component_id != INVALID_GLOBAL_INDEX &&
             component.component_id <= previous_component_id)) {
            return fail("requires canonical component-ledger order");
        }
        previous_component_id = component.component_id;
    }
    if (correction.limited_edges > correction.edges.size() ||
        (correction.subthreshold_component_present &&
         correction.subthreshold_component.nodes == 0u)) {
        return fail("found inconsistent ledger counts");
    }
    return true;
}

[[nodiscard]] bool replicatedSplitStageProvenanceMatches(
    const PhaseGraphCollectiveContext& collective,
    const LevelSetP1PhaseSplitStageProvenance& provenance,
    std::string& diagnostic)
{
    const auto real_bits = [](Real value) {
        return std::bit_cast<std::uint64_t>(value);
    };
    const std::array<std::pair<std::string_view, std::uint64_t>, 39> words{{
        {"scheme", static_cast<std::uint64_t>(provenance.scheme)},
        {"transport mesh policy",
         static_cast<std::uint64_t>(provenance.transport_mesh_policy)},
        {"temporal order",
         static_cast<std::uint64_t>(provenance.temporal_order)},
        {"prospective step", provenance.prospective_step},
        {"attempt", provenance.attempt},
        {"step start time", real_bits(provenance.step_start_time)},
        {"step end time", real_bits(provenance.step_end_time)},
        {"q input time", real_bits(provenance.q_input_time)},
        {"velocity state time", real_bits(provenance.velocity_state_time)},
        {"time step", real_bits(provenance.time_step)},
        {"operator state revision", provenance.operator_state_revision},
        {"previous q revision", provenance.previous_q_revision},
        {"nodal velocity revision", provenance.nodal_velocity_revision},
        {"previous graph dimension",
         static_cast<std::uint64_t>(
             provenance.previous_graph_identity.dimension)},
        {"previous graph nodes",
         static_cast<std::uint64_t>(
             provenance.previous_graph_identity.nodes)},
        {"previous graph edges",
         static_cast<std::uint64_t>(
             provenance.previous_graph_identity.edges)},
        {"previous graph layout revision",
         provenance.previous_graph_identity.dof_layout_revision},
        {"previous graph content revision",
         provenance.previous_graph_identity.content_revision},
        {"operator graph dimension",
         static_cast<std::uint64_t>(
             provenance.operator_graph_identity.dimension)},
        {"operator graph nodes",
         static_cast<std::uint64_t>(
             provenance.operator_graph_identity.nodes)},
        {"operator graph edges",
         static_cast<std::uint64_t>(
             provenance.operator_graph_identity.edges)},
        {"operator graph layout revision",
         provenance.operator_graph_identity.dof_layout_revision},
        {"operator graph content revision",
         provenance.operator_graph_identity.content_revision},
        {"invariant tolerance",
         real_bits(provenance.stage_options.invariant_tolerance)},
        {"component activity tolerance",
         real_bits(provenance.stage_options.component_activity_tolerance)},
        {"maximum Courant option",
         real_bits(provenance.stage_options.maximum_courant)},
        {"Courant-limit enforcement",
         provenance.stage_options.enforce_courant_limit ? 1u : 0u},
        {"constant-preservation requirement",
         provenance.stage_options.require_constant_preservation ? 1u : 0u},
        {"final flux ledger digest",
         provenance.final_flux_ledger_digest},
        // Include the computed semantic fields a second time under a versioned
        // tail so future extensions cannot silently preserve an old prefix.
        {"contract version", 3u},
        {"previous q role", 1u},
        {"endpoint velocity role", 2u},
        {"fixed mesh role", 3u},
        {"clamped one-ring bounds role", 4u},
        {"graph identity version", 1u},
        {"content revision version", 1u},
        {"flux ledger version", 2u},
        {"real bytes", sizeof(Real)},
        {"global index bytes", sizeof(GlobalIndex)},
    }};
    for (const auto& [name, word] : words) {
        const auto minimum = allReduceUnsigned64Min(collective, word);
        const auto maximum = allReduceUnsigned64Max(collective, word);
        if (minimum != maximum) {
            diagnostic =
                "P1 conservative phase split-stage provenance requires identical " +
                std::string(name) + " on every rank";
            return false;
        }
    }
    return true;
}

[[nodiscard]] Real scaledTolerance(Real tolerance,
                                   std::initializer_list<Real> values)
{
    Real scale = Real{1.0};
    for (const Real value : values) {
        scale = std::max(scale, std::abs(value));
    }
    return tolerance * scale;
}

[[nodiscard]] Real dot(const Vector3& first,
                       const Vector3& second,
                       int dimension) noexcept
{
    Real value{0.0};
    for (int d = 0; d < dimension; ++d) {
        value += first[static_cast<std::size_t>(d)] *
                 second[static_cast<std::size_t>(d)];
    }
    return value;
}

struct PhaseStageVerificationNode {
    Real row_divergence{0.0};
    Real direct_strong_rate{0.0};
    Real decomposed_strong_rate{0.0};
    Real nodal_courant{0.0};
    Real low_order_internal{0.0};
    Real raw_internal{0.0};
    Real limited_internal{0.0};
    Real positive_raw{0.0};
    Real negative_raw{0.0};
    Real positive_factor{1.0};
    Real negative_factor{1.0};
    Real expected_lower{0.0};
    Real expected_upper{0.0};
    std::size_t parent{0u};
    std::size_t component_nodes{0u};
    bool participating{false};
    bool active{false};
};

[[nodiscard]] bool insidePhaseStageBounds(
    Real value, Real lower, Real upper, Real tolerance) noexcept
{
    const Real slack = scaledTolerance(tolerance, {value, lower, upper});
    return value >= lower - slack && value <= upper + slack;
}

[[nodiscard]] bool samePhaseFluxComponentExact(
    const LevelSetPhaseFluxComponentLedger& left,
    const LevelSetPhaseFluxComponentLedger& right) noexcept
{
    return left.component_id == right.component_id &&
           left.nodes == right.nodes &&
           exactPhaseStageRealEqual(
               left.previous_liquid_measure,
               right.previous_liquid_measure) &&
           exactPhaseStageRealEqual(
               left.low_order_liquid_measure,
               right.low_order_liquid_measure) &&
           exactPhaseStageRealEqual(
               left.raw_target_liquid_measure,
               right.raw_target_liquid_measure) &&
           exactPhaseStageRealEqual(
               left.limited_liquid_measure,
               right.limited_liquid_measure) &&
           exactPhaseStageRealEqual(
               left.physical_boundary_mass_transfer,
               right.physical_boundary_mass_transfer) &&
           exactPhaseStageRealEqual(
               left.discrete_divergence_mass_source,
               right.discrete_divergence_mass_source) &&
           exactPhaseStageRealEqual(
               left.low_order_interior_mass_transfer,
               right.low_order_interior_mass_transfer) &&
           exactPhaseStageRealEqual(
               left.raw_antidiffusive_mass_transfer,
               right.raw_antidiffusive_mass_transfer) &&
           exactPhaseStageRealEqual(
               left.limited_antidiffusive_mass_transfer,
               right.limited_antidiffusive_mass_transfer) &&
           exactPhaseStageRealEqual(
               left.low_order_balance_residual,
               right.low_order_balance_residual) &&
           exactPhaseStageRealEqual(
               left.raw_target_balance_residual,
               right.raw_target_balance_residual) &&
           exactPhaseStageRealEqual(
               left.limited_balance_residual,
               right.limited_balance_residual);
}

/**
 * Independently verify a claimed stage without constructing a second edge
 * ledger.  The caller owns one O(nodes) scratch buffer; graph and claimed
 * ledger edges are streamed in canonical order.
 */
[[nodiscard]] bool verifyPhaseStageLedgerEquations(
    const LevelSetP1PhaseTransportGraph& graph,
    std::span<const Real> previous_liquid_indicator,
    const LevelSetP1PhaseTransportStageResult& stage,
    std::span<PhaseStageVerificationNode> scratch,
    std::string& diagnostic)
{
    const auto fail = [&](std::string_view reason) {
        diagnostic =
            "P1 conservative phase split-stage independent ledger verification " +
            std::string(reason);
        return false;
    };
    const auto same_real = [&](Real actual, Real expected) {
        return exactPhaseStageRealEqual(actual, expected);
    };
    const auto& options = stage.executed_options;
    const auto& correction = stage.correction;
    const std::size_t node_count = graph.nodes;
    if (scratch.size() != node_count) {
        return fail("requires complete nodal scratch storage");
    }

    for (std::size_t node = 0u; node < node_count; ++node) {
        auto& values = scratch[node];
        values = {};
        values.positive_factor = Real{1.0};
        values.negative_factor = Real{1.0};
        values.parent = node;
        values.expected_lower = previous_liquid_indicator[node];
        values.expected_upper = previous_liquid_indicator[node];
        const Real diagonal_velocity = dot(
            graph.diagonal_gradient[node],
            stage.sampled_nodal_velocity[node], graph.dimension);
        values.row_divergence = diagonal_velocity;
        values.direct_strong_rate =
            -diagonal_velocity * previous_liquid_indicator[node];
    }

    Real minimum_low_order_coefficient =
        std::numeric_limits<Real>::infinity();
    bool low_order_coefficients_nonnegative = true;
    for (std::size_t edge_index = 0u;
         edge_index < graph.edges.size(); ++edge_index) {
        const auto& graph_edge = graph.edges[edge_index];
        const auto first =
            static_cast<std::size_t>(graph_edge.first_node);
        const auto second =
            static_cast<std::size_t>(graph_edge.second_node);
        const Real first_to_second_speed = dot(
            graph_edge.first_test_second_gradient,
            stage.sampled_nodal_velocity[second], graph.dimension);
        const Real second_to_first_speed = dot(
            graph_edge.second_test_first_gradient,
            stage.sampled_nodal_velocity[first], graph.dimension);
        const Real diffusion = std::max(
            std::abs(first_to_second_speed),
            std::abs(second_to_first_speed));
        const Real first_coefficient =
            diffusion - first_to_second_speed;
        const Real second_coefficient =
            diffusion - second_to_first_speed;
        minimum_low_order_coefficient = std::min(
            minimum_low_order_coefficient,
            std::min(first_coefficient, second_coefficient));
        const Real coefficient_tolerance = scaledTolerance(
            options.invariant_tolerance,
            {diffusion, first_to_second_speed,
             second_to_first_speed});
        low_order_coefficients_nonnegative =
            low_order_coefficients_nonnegative &&
            first_coefficient >= -coefficient_tolerance &&
            second_coefficient >= -coefficient_tolerance;

        scratch[first].nodal_courant +=
            stage.time_step * std::max(Real{0.0}, first_coefficient) /
            graph.lumped_control_volume[first];
        scratch[second].nodal_courant +=
            stage.time_step * std::max(Real{0.0}, second_coefficient) /
            graph.lumped_control_volume[second];

        const Real first_indicator = previous_liquid_indicator[first];
        const Real second_indicator = previous_liquid_indicator[second];
        const Real central_rate =
            second_to_first_speed * first_indicator -
            first_to_second_speed * second_indicator;
        const Real diffusive_rate =
            diffusion * (second_indicator - first_indicator);
        const Real low_order_transfer =
            stage.time_step * (central_rate + diffusive_rate);
        const Real raw_transfer = -stage.time_step * diffusive_rate;
        const auto& input_edge = stage.flux_edges[edge_index];
        const auto& edge = correction.edges[edge_index];
        if (!same_real(
                input_edge.low_order_mass_transfer,
                low_order_transfer) ||
            !same_real(
                input_edge.raw_antidiffusive_mass_transfer,
                raw_transfer) ||
            !same_real(edge.low_order_mass_transfer, low_order_transfer) ||
            !same_real(
                edge.raw_antidiffusive_mass_transfer, raw_transfer) ||
            !same_real(
                edge.low_order_pair_cancellation_residual,
                low_order_transfer + (-low_order_transfer)) ||
            !same_real(
                edge.raw_pair_cancellation_residual,
                raw_transfer + (-raw_transfer))) {
            return fail("found an edge transfer equation mismatch");
        }

        scratch[first].low_order_internal += low_order_transfer;
        scratch[second].low_order_internal -= low_order_transfer;
        scratch[first].raw_internal += raw_transfer;
        scratch[second].raw_internal -= raw_transfer;
        scratch[first].positive_raw +=
            std::max(Real{0.0}, raw_transfer);
        scratch[first].negative_raw +=
            std::min(Real{0.0}, raw_transfer);
        scratch[second].positive_raw +=
            std::max(Real{0.0}, -raw_transfer);
        scratch[second].negative_raw +=
            std::min(Real{0.0}, -raw_transfer);
        scratch[first].expected_lower = std::min(
            scratch[first].expected_lower, second_indicator);
        scratch[second].expected_lower = std::min(
            scratch[second].expected_lower, first_indicator);
        scratch[first].expected_upper = std::max(
            scratch[first].expected_upper, second_indicator);
        scratch[second].expected_upper = std::max(
            scratch[second].expected_upper, first_indicator);

        scratch[first].decomposed_strong_rate += central_rate;
        scratch[second].decomposed_strong_rate -= central_rate;
        scratch[first].row_divergence += first_to_second_speed;
        scratch[second].row_divergence += second_to_first_speed;
        scratch[first].direct_strong_rate -=
            first_to_second_speed * second_indicator;
        scratch[second].direct_strong_rate -=
            second_to_first_speed * first_indicator;
    }
    if (graph.edges.empty()) {
        minimum_low_order_coefficient = Real{0.0};
    }
    if (stage.low_order_coefficients_nonnegative !=
            low_order_coefficients_nonnegative ||
        !same_real(stage.minimum_low_order_coefficient,
                   minimum_low_order_coefficient)) {
        return fail("found a low-order coefficient summary mismatch");
    }

    Real maximum_courant{0.0};
    Real maximum_integrated_strong_rate{0.0};
    Real maximum_strong_form_decomposition_residual{0.0};
    for (std::size_t node = 0u; node < node_count; ++node) {
        auto& values = scratch[node];
        const Real indicator = previous_liquid_indicator[node];
        const Real boundary_rate =
            -indicator * dot(
                graph.boundary_column_sum[node],
                stage.sampled_nodal_velocity[node], graph.dimension);
        const Real divergence_rate =
            indicator * values.row_divergence;
        const Real boundary_transfer = stage.time_step * boundary_rate;
        const Real divergence_transfer =
            stage.time_step * divergence_rate;
        values.decomposed_strong_rate +=
            boundary_rate + divergence_rate;
        values.direct_strong_rate += divergence_rate;
        const Real residual = stage.time_step *
            (values.decomposed_strong_rate - values.direct_strong_rate);
        maximum_integrated_strong_rate = std::max(
            maximum_integrated_strong_rate,
            stage.time_step * std::max(
                std::abs(values.decomposed_strong_rate),
                std::abs(values.direct_strong_rate)));
        maximum_strong_form_decomposition_residual = std::max(
            maximum_strong_form_decomposition_residual,
            std::abs(residual));
        maximum_courant =
            std::max(maximum_courant, values.nodal_courant);

        const auto& ledger = correction.nodes[node];
        if (!same_real(stage.nodal_courant[node], values.nodal_courant) ||
            !same_real(
                stage.physical_boundary_mass_transfer[node],
                boundary_transfer) ||
            !same_real(
                stage.discrete_divergence_mass_source[node],
                divergence_transfer) ||
            !same_real(
                ledger.physical_boundary_mass_transfer,
                boundary_transfer) ||
            !same_real(
                ledger.discrete_divergence_mass_source,
                divergence_transfer)) {
            return fail("found a nodal transport equation mismatch");
        }
    }
    const bool strong_form_decomposition_satisfied =
        maximum_strong_form_decomposition_residual <= scaledTolerance(
            options.invariant_tolerance,
            {maximum_integrated_strong_rate});
    const bool courant_satisfied =
        !options.enforce_courant_limit ||
        maximum_courant <= options.maximum_courant + scaledTolerance(
            options.invariant_tolerance,
            {maximum_courant, options.maximum_courant});
    if (!same_real(stage.maximum_courant, maximum_courant) ||
        !same_real(
            stage.maximum_strong_form_decomposition_residual,
            maximum_strong_form_decomposition_residual) ||
        stage.strong_form_decomposition_satisfied !=
            strong_form_decomposition_satisfied ||
        stage.courant_satisfied != courant_satisfied) {
        return fail("found a Courant or strong-form summary mismatch");
    }

    bool low_order_bounds_satisfied = true;
    Real minimum_low_order_liquid_indicator =
        std::numeric_limits<Real>::infinity();
    Real maximum_low_order_liquid_indicator =
        -std::numeric_limits<Real>::infinity();
    Real minimum_raw_target_liquid_indicator =
        std::numeric_limits<Real>::infinity();
    Real maximum_raw_target_liquid_indicator =
        -std::numeric_limits<Real>::infinity();
    Real previous_minimum = std::numeric_limits<Real>::infinity();
    Real previous_maximum = -std::numeric_limits<Real>::infinity();
    for (std::size_t node = 0u; node < node_count; ++node) {
        auto& values = scratch[node];
        const auto& ledger = correction.nodes[node];
        const Real previous = previous_liquid_indicator[node];
        const Real lower = ledger.lower_liquid_indicator;
        const Real upper = ledger.upper_liquid_indicator;
        const Real expected_lower = std::clamp(
            values.expected_lower, Real{0.0}, Real{1.0});
        const Real expected_upper = std::clamp(
            values.expected_upper, Real{0.0}, Real{1.0});
        if (!same_real(lower, expected_lower) ||
            !same_real(upper, expected_upper)) {
            return fail("found a production one-ring bound mismatch");
        }
        const Real unit_slack = scaledTolerance(
            options.invariant_tolerance, {lower, upper, previous});
        if (lower < -unit_slack ||
            upper > Real{1.0} + unit_slack || lower > upper ||
            !insidePhaseStageBounds(
                previous, lower, upper, options.invariant_tolerance)) {
            return fail("found invalid retained phase-indicator bounds");
        }
        const Real low_order_mass =
            ledger.lumped_control_volume * previous +
            ledger.physical_boundary_mass_transfer +
            ledger.discrete_divergence_mass_source +
            values.low_order_internal;
        const Real low_order_indicator =
            low_order_mass / ledger.lumped_control_volume;
        const Real raw_target_indicator =
            low_order_indicator +
            values.raw_internal / ledger.lumped_control_volume;
        if (!same_real(
                ledger.low_order_interior_mass_transfer,
                values.low_order_internal) ||
            !same_real(
                ledger.raw_antidiffusive_mass_transfer,
                values.raw_internal) ||
            !same_real(
                ledger.positive_raw_antidiffusive_mass,
                values.positive_raw) ||
            !same_real(
                ledger.negative_raw_antidiffusive_mass,
                values.negative_raw) ||
            !same_real(
                ledger.low_order_liquid_indicator,
                low_order_indicator) ||
            !same_real(
                ledger.raw_target_liquid_indicator,
                raw_target_indicator)) {
            return fail("found a nodal predictor equation mismatch");
        }
        low_order_bounds_satisfied =
            low_order_bounds_satisfied && insidePhaseStageBounds(
                low_order_indicator, lower, upper,
                options.invariant_tolerance);
        minimum_low_order_liquid_indicator = std::min(
            minimum_low_order_liquid_indicator, low_order_indicator);
        maximum_low_order_liquid_indicator = std::max(
            maximum_low_order_liquid_indicator, low_order_indicator);
        minimum_raw_target_liquid_indicator = std::min(
            minimum_raw_target_liquid_indicator, raw_target_indicator);
        maximum_raw_target_liquid_indicator = std::max(
            maximum_raw_target_liquid_indicator, raw_target_indicator);
        previous_minimum = std::min(previous_minimum, previous);
        previous_maximum = std::max(previous_maximum, previous);

        const Real positive_allowance = std::max(
            Real{0.0}, ledger.lumped_control_volume *
                (upper - low_order_indicator));
        const Real negative_allowance = std::min(
            Real{0.0}, ledger.lumped_control_volume *
                (lower - low_order_indicator));
        if (values.positive_raw > Real{0.0}) {
            values.positive_factor = std::clamp(
                positive_allowance / values.positive_raw,
                Real{0.0}, Real{1.0});
        }
        if (values.negative_raw < Real{0.0}) {
            values.negative_factor = std::clamp(
                negative_allowance / values.negative_raw,
                Real{0.0}, Real{1.0});
        }
        if (!same_real(
                ledger.positive_correction_factor,
                values.positive_factor) ||
            !same_real(
                ledger.negative_correction_factor,
                values.negative_factor)) {
            return fail("found a nodal limiter-factor mismatch");
        }
    }
    if (correction.low_order_bounds_satisfied !=
            low_order_bounds_satisfied ||
        !same_real(
            correction.minimum_low_order_liquid_indicator,
            minimum_low_order_liquid_indicator) ||
        !same_real(
            correction.maximum_low_order_liquid_indicator,
            maximum_low_order_liquid_indicator) ||
        !same_real(
            correction.minimum_raw_target_liquid_indicator,
            minimum_raw_target_liquid_indicator) ||
        !same_real(
            correction.maximum_raw_target_liquid_indicator,
            maximum_raw_target_liquid_indicator)) {
        return fail("found a predictor summary mismatch");
    }

    std::size_t limited_edges{0u};
    Real maximum_edge_pair_cancellation_residual{0.0};
    for (std::size_t edge_index = 0u;
         edge_index < correction.edges.size(); ++edge_index) {
        const auto& input_edge = stage.flux_edges[edge_index];
        const auto& edge = correction.edges[edge_index];
        const auto first = static_cast<std::size_t>(edge.first_node);
        const auto second = static_cast<std::size_t>(edge.second_node);
        Real factor{1.0};
        if (input_edge.raw_antidiffusive_mass_transfer > Real{0.0}) {
            factor = std::min(
                scratch[first].positive_factor,
                scratch[second].negative_factor);
        } else if (input_edge.raw_antidiffusive_mass_transfer <
                   Real{0.0}) {
            factor = std::min(
                scratch[first].negative_factor,
                scratch[second].positive_factor);
        }
        const Real limited_transfer =
            factor * input_edge.raw_antidiffusive_mass_transfer;
        const Real limited_pair_residual =
            limited_transfer + (-limited_transfer);
        if (!same_real(edge.correction_factor, factor) ||
            !same_real(
                edge.limited_antidiffusive_mass_transfer,
                limited_transfer) ||
            !same_real(
                edge.limited_pair_cancellation_residual,
                limited_pair_residual)) {
            return fail("found a limited edge-transfer mismatch");
        }
        scratch[first].limited_internal += limited_transfer;
        scratch[second].limited_internal -= limited_transfer;
        const Real correction_amount =
            input_edge.raw_antidiffusive_mass_transfer -
            limited_transfer;
        const Real correction_tolerance = scaledTolerance(
            options.invariant_tolerance,
            {input_edge.raw_antidiffusive_mass_transfer,
             limited_transfer});
        if (std::abs(correction_amount) > correction_tolerance) {
            ++limited_edges;
        }
        maximum_edge_pair_cancellation_residual = std::max(
            maximum_edge_pair_cancellation_residual,
            std::max({
                std::abs(edge.low_order_pair_cancellation_residual),
                std::abs(edge.raw_pair_cancellation_residual),
                std::abs(limited_pair_residual)}));
    }

    bool limited_bounds_satisfied = true;
    bool local_balance_satisfied = true;
    Real minimum_limited_liquid_indicator =
        std::numeric_limits<Real>::infinity();
    Real maximum_limited_liquid_indicator =
        -std::numeric_limits<Real>::infinity();
    Real maximum_low_order_local_mass_balance_residual{0.0};
    Real maximum_raw_target_local_mass_balance_residual{0.0};
    Real maximum_local_mass_balance_residual{0.0};
    long double total_previous{0.0L};
    long double total_low_order{0.0L};
    long double total_raw_target{0.0L};
    long double total_limited{0.0L};
    long double total_boundary{0.0L};
    long double total_divergence{0.0L};
    long double low_order_cancellation{0.0L};
    long double raw_cancellation{0.0L};
    long double limited_cancellation{0.0L};
    for (std::size_t node = 0u; node < node_count; ++node) {
        const auto& values = scratch[node];
        const auto& ledger = correction.nodes[node];
        const Real limited_indicator =
            ledger.low_order_liquid_indicator +
            values.limited_internal / ledger.lumped_control_volume;
        const Real low_order_residual =
            ledger.lumped_control_volume *
                (ledger.low_order_liquid_indicator -
                 ledger.previous_liquid_indicator) -
            (ledger.physical_boundary_mass_transfer +
             ledger.discrete_divergence_mass_source +
             values.low_order_internal);
        const Real raw_target_residual =
            ledger.lumped_control_volume *
                (ledger.raw_target_liquid_indicator -
                 ledger.previous_liquid_indicator) -
            (ledger.physical_boundary_mass_transfer +
             ledger.discrete_divergence_mass_source +
             values.low_order_internal + values.raw_internal);
        const Real limited_residual =
            ledger.lumped_control_volume *
                (limited_indicator - ledger.previous_liquid_indicator) -
            (ledger.physical_boundary_mass_transfer +
             ledger.discrete_divergence_mass_source +
             values.low_order_internal + values.limited_internal);
        if (!same_real(
                ledger.limited_antidiffusive_mass_transfer,
                values.limited_internal) ||
            !same_real(
                ledger.limited_liquid_indicator,
                limited_indicator) ||
            !same_real(
                ledger.low_order_local_mass_balance_residual,
                low_order_residual) ||
            !same_real(
                ledger.raw_target_local_mass_balance_residual,
                raw_target_residual) ||
            !same_real(
                ledger.local_mass_balance_residual,
                limited_residual)) {
            return fail("found a nodal limited-state or balance mismatch");
        }
        limited_bounds_satisfied = limited_bounds_satisfied &&
            insidePhaseStageBounds(
                limited_indicator,
                ledger.lower_liquid_indicator,
                ledger.upper_liquid_indicator,
                options.invariant_tolerance);
        const Real local_tolerance = scaledTolerance(
            options.invariant_tolerance,
            {ledger.lumped_control_volume *
                 ledger.previous_liquid_indicator,
             ledger.lumped_control_volume *
                 ledger.low_order_liquid_indicator,
             ledger.lumped_control_volume *
                 ledger.raw_target_liquid_indicator,
             ledger.lumped_control_volume * limited_indicator,
             ledger.physical_boundary_mass_transfer,
             ledger.discrete_divergence_mass_source,
             values.low_order_internal,
             values.raw_internal,
             values.limited_internal});
        local_balance_satisfied = local_balance_satisfied &&
            std::abs(low_order_residual) <= local_tolerance &&
            std::abs(raw_target_residual) <= local_tolerance &&
            std::abs(limited_residual) <= local_tolerance;
        maximum_low_order_local_mass_balance_residual = std::max(
            maximum_low_order_local_mass_balance_residual,
            std::abs(low_order_residual));
        maximum_raw_target_local_mass_balance_residual = std::max(
            maximum_raw_target_local_mass_balance_residual,
            std::abs(raw_target_residual));
        maximum_local_mass_balance_residual = std::max(
            maximum_local_mass_balance_residual,
            std::abs(limited_residual));
        minimum_limited_liquid_indicator = std::min(
            minimum_limited_liquid_indicator, limited_indicator);
        maximum_limited_liquid_indicator = std::max(
            maximum_limited_liquid_indicator, limited_indicator);

        total_previous += static_cast<long double>(
            ledger.lumped_control_volume *
            ledger.previous_liquid_indicator);
        total_low_order += static_cast<long double>(
            ledger.lumped_control_volume *
            ledger.low_order_liquid_indicator);
        total_raw_target += static_cast<long double>(
            ledger.lumped_control_volume *
            ledger.raw_target_liquid_indicator);
        total_limited += static_cast<long double>(
            ledger.lumped_control_volume * limited_indicator);
        total_boundary += static_cast<long double>(
            ledger.physical_boundary_mass_transfer);
        total_divergence += static_cast<long double>(
            ledger.discrete_divergence_mass_source);
        low_order_cancellation +=
            static_cast<long double>(values.low_order_internal);
        raw_cancellation +=
            static_cast<long double>(values.raw_internal);
        limited_cancellation +=
            static_cast<long double>(values.limited_internal);
    }

    const Real total_previous_value = static_cast<Real>(total_previous);
    const Real total_low_order_value = static_cast<Real>(total_low_order);
    const Real total_raw_target_value = static_cast<Real>(total_raw_target);
    const Real total_limited_value = static_cast<Real>(total_limited);
    const Real total_boundary_value = static_cast<Real>(total_boundary);
    const Real total_divergence_value = static_cast<Real>(total_divergence);
    const Real low_order_cancellation_value =
        static_cast<Real>(low_order_cancellation);
    const Real raw_cancellation_value =
        static_cast<Real>(raw_cancellation);
    const Real limited_cancellation_value =
        static_cast<Real>(limited_cancellation);
    Real interior_scale{1.0};
    for (const auto& edge : correction.edges) {
        interior_scale += std::abs(edge.low_order_mass_transfer) +
                          std::abs(edge.raw_antidiffusive_mass_transfer) +
                          std::abs(
                              edge.limited_antidiffusive_mass_transfer);
    }
    const Real interior_tolerance =
        options.invariant_tolerance * interior_scale;
    const bool interior_cancellation_satisfied =
        maximum_edge_pair_cancellation_residual == Real{0.0} &&
        std::abs(low_order_cancellation_value) <= interior_tolerance &&
        std::abs(raw_cancellation_value) <= interior_tolerance &&
        std::abs(limited_cancellation_value) <= interior_tolerance;
    const Real low_order_global_residual =
        total_low_order_value - total_previous_value -
        total_boundary_value - total_divergence_value;
    const Real raw_target_global_residual =
        total_raw_target_value - total_previous_value -
        total_boundary_value - total_divergence_value;
    const Real limited_global_residual =
        total_limited_value - total_previous_value -
        total_boundary_value - total_divergence_value;
    const Real global_tolerance = scaledTolerance(
        options.invariant_tolerance,
        {total_previous_value, total_low_order_value,
         total_raw_target_value, total_limited_value,
         total_boundary_value, total_divergence_value});
    const bool global_balance_satisfied =
        std::abs(low_order_global_residual) <= global_tolerance &&
        std::abs(raw_target_global_residual) <= global_tolerance &&
        std::abs(limited_global_residual) <= global_tolerance;
    if (correction.applied != (limited_edges > 0u) ||
        correction.limited_edges != limited_edges ||
        correction.limited_bounds_satisfied != limited_bounds_satisfied ||
        correction.local_balance_satisfied != local_balance_satisfied ||
        correction.interior_cancellation_satisfied !=
            interior_cancellation_satisfied ||
        correction.global_balance_satisfied != global_balance_satisfied ||
        !same_real(
            correction.maximum_edge_pair_cancellation_residual,
            maximum_edge_pair_cancellation_residual) ||
        !same_real(
            correction.maximum_low_order_local_mass_balance_residual,
            maximum_low_order_local_mass_balance_residual) ||
        !same_real(
            correction.maximum_raw_target_local_mass_balance_residual,
            maximum_raw_target_local_mass_balance_residual) ||
        !same_real(
            correction.maximum_local_mass_balance_residual,
            maximum_local_mass_balance_residual) ||
        !same_real(
            correction.minimum_limited_liquid_indicator,
            minimum_limited_liquid_indicator) ||
        !same_real(
            correction.maximum_limited_liquid_indicator,
            maximum_limited_liquid_indicator) ||
        !same_real(
            correction.low_order_nodal_cancellation_residual,
            low_order_cancellation_value) ||
        !same_real(
            correction.raw_nodal_cancellation_residual,
            raw_cancellation_value) ||
        !same_real(
            correction.limited_nodal_cancellation_residual,
            limited_cancellation_value) ||
        !same_real(
            correction.total_previous_liquid_measure,
            total_previous_value) ||
        !same_real(
            correction.total_low_order_liquid_measure,
            total_low_order_value) ||
        !same_real(
            correction.total_raw_target_liquid_measure,
            total_raw_target_value) ||
        !same_real(
            correction.total_limited_liquid_measure,
            total_limited_value) ||
        !same_real(
            correction.total_physical_boundary_mass_transfer,
            total_boundary_value) ||
        !same_real(
            correction.total_discrete_divergence_mass_source,
            total_divergence_value) ||
        !same_real(
            correction.low_order_global_mass_balance_residual,
            low_order_global_residual) ||
        !same_real(
            correction.raw_target_global_mass_balance_residual,
            raw_target_global_residual) ||
        !same_real(
            correction.global_mass_balance_residual,
            limited_global_residual)) {
        return fail("found a correction summary equation mismatch");
    }

    const bool constant_state_input =
        previous_maximum - previous_minimum <= scaledTolerance(
            options.invariant_tolerance,
            {previous_minimum, previous_maximum});
    bool constant_preservation_satisfied = true;
    Real maximum_constant_preservation_error{0.0};
    if (constant_state_input) {
        for (const auto& ledger : correction.nodes) {
            const Real error = std::max(
                std::abs(ledger.low_order_liquid_indicator -
                         ledger.previous_liquid_indicator),
                std::abs(ledger.limited_liquid_indicator -
                         ledger.previous_liquid_indicator));
            maximum_constant_preservation_error = std::max(
                maximum_constant_preservation_error, error);
            const Real tolerance = scaledTolerance(
                options.invariant_tolerance,
                {ledger.previous_liquid_indicator,
                 ledger.low_order_liquid_indicator,
                 ledger.limited_liquid_indicator});
            constant_preservation_satisfied =
                constant_preservation_satisfied && error <= tolerance;
        }
        for (const auto& ledger : correction.nodes) {
            const Real error = std::abs(
                ledger.raw_target_liquid_indicator -
                ledger.previous_liquid_indicator);
            maximum_constant_preservation_error = std::max(
                maximum_constant_preservation_error, error);
            const Real tolerance = scaledTolerance(
                options.invariant_tolerance,
                {ledger.previous_liquid_indicator,
                 ledger.raw_target_liquid_indicator});
            constant_preservation_satisfied =
                constant_preservation_satisfied && error <= tolerance;
        }
    }
    if (correction.constant_state_input != constant_state_input ||
        correction.constant_preservation_required !=
            options.require_constant_preservation ||
        correction.constant_preservation_satisfied !=
            constant_preservation_satisfied ||
        !same_real(
            correction.maximum_constant_preservation_error,
            maximum_constant_preservation_error)) {
        return fail("found a constant-preservation mismatch");
    }

    for (std::size_t node = 0u; node < node_count; ++node) {
        auto& values = scratch[node];
        const auto& ledger = correction.nodes[node];
        const Real activity_scale = std::max(
            {std::abs(ledger.previous_liquid_indicator),
             std::abs(ledger.low_order_liquid_indicator),
             std::abs(ledger.raw_target_liquid_indicator),
             std::abs(ledger.limited_liquid_indicator),
             std::abs(ledger.physical_boundary_mass_transfer) /
                 ledger.lumped_control_volume,
             std::abs(ledger.discrete_divergence_mass_source) /
                 ledger.lumped_control_volume,
             std::abs(ledger.low_order_interior_mass_transfer) /
                 ledger.lumped_control_volume,
             std::abs(ledger.raw_antidiffusive_mass_transfer) /
                 ledger.lumped_control_volume,
             std::abs(ledger.limited_antidiffusive_mass_transfer) /
                 ledger.lumped_control_volume});
        values.participating = activity_scale > Real{0.0};
        values.active =
            activity_scale > options.component_activity_tolerance;
        values.parent = node;
    }
    const auto find_root = [&scratch](std::size_t node) {
        std::size_t root = node;
        while (scratch[root].parent != root) {
            root = scratch[root].parent;
        }
        while (scratch[node].parent != node) {
            const auto next = scratch[node].parent;
            scratch[node].parent = root;
            node = next;
        }
        return root;
    };
    for (const auto& edge : correction.edges) {
        const auto first = static_cast<std::size_t>(edge.first_node);
        const auto second = static_cast<std::size_t>(edge.second_node);
        if (!scratch[first].active || !scratch[second].active) {
            continue;
        }
        const auto first_root = find_root(first);
        const auto second_root = find_root(second);
        if (first_root != second_root) {
            scratch[std::max(first_root, second_root)].parent =
                std::min(first_root, second_root);
        }
    }
    for (auto& values : scratch) {
        values.row_divergence = Real{0.0};
        values.direct_strong_rate = Real{0.0};
        values.decomposed_strong_rate = Real{0.0};
        values.nodal_courant = Real{0.0};
        values.low_order_internal = Real{0.0};
        values.raw_internal = Real{0.0};
        values.limited_internal = Real{0.0};
        values.positive_raw = Real{0.0};
        values.negative_raw = Real{0.0};
        values.component_nodes = 0u;
    }
    const auto accumulate_component = [](
        LevelSetPhaseFluxComponentLedger& component,
        const LevelSetPhaseFluxNodeLedger& ledger) {
        ++component.nodes;
        component.previous_liquid_measure +=
            ledger.lumped_control_volume *
            ledger.previous_liquid_indicator;
        component.low_order_liquid_measure +=
            ledger.lumped_control_volume *
            ledger.low_order_liquid_indicator;
        component.raw_target_liquid_measure +=
            ledger.lumped_control_volume *
            ledger.raw_target_liquid_indicator;
        component.limited_liquid_measure +=
            ledger.lumped_control_volume *
            ledger.limited_liquid_indicator;
        component.physical_boundary_mass_transfer +=
            ledger.physical_boundary_mass_transfer;
        component.discrete_divergence_mass_source +=
            ledger.discrete_divergence_mass_source;
        component.low_order_interior_mass_transfer +=
            ledger.low_order_interior_mass_transfer;
        component.raw_antidiffusive_mass_transfer +=
            ledger.raw_antidiffusive_mass_transfer;
        component.limited_antidiffusive_mass_transfer +=
            ledger.limited_antidiffusive_mass_transfer;
    };
    const auto accumulate_scratch_component = [](
        PhaseStageVerificationNode& component,
        const LevelSetPhaseFluxNodeLedger& ledger) {
        ++component.component_nodes;
        component.row_divergence +=
            ledger.lumped_control_volume *
            ledger.previous_liquid_indicator;
        component.direct_strong_rate +=
            ledger.lumped_control_volume *
            ledger.low_order_liquid_indicator;
        component.decomposed_strong_rate +=
            ledger.lumped_control_volume *
            ledger.raw_target_liquid_indicator;
        component.nodal_courant +=
            ledger.lumped_control_volume *
            ledger.limited_liquid_indicator;
        component.low_order_internal +=
            ledger.physical_boundary_mass_transfer;
        component.raw_internal +=
            ledger.discrete_divergence_mass_source;
        component.limited_internal +=
            ledger.low_order_interior_mass_transfer;
        component.positive_raw +=
            ledger.raw_antidiffusive_mass_transfer;
        component.negative_raw +=
            ledger.limited_antidiffusive_mass_transfer;
    };
    LevelSetPhaseFluxComponentLedger expected_subthreshold;
    bool subthreshold_component_present = false;
    for (std::size_t node = 0u; node < node_count; ++node) {
        auto& values = scratch[node];
        if (!values.active) {
            if (values.participating) {
                subthreshold_component_present = true;
                accumulate_component(
                    expected_subthreshold, correction.nodes[node]);
            }
            if (correction.node_component_ids[node] !=
                INVALID_GLOBAL_INDEX) {
                return fail("found a subthreshold component-id mismatch");
            }
            continue;
        }
        const auto root = find_root(node);
        if (correction.node_component_ids[node] !=
            static_cast<GlobalIndex>(root)) {
            return fail("found a connected-component id mismatch");
        }
        accumulate_scratch_component(
            scratch[root], correction.nodes[node]);
    }

    bool component_balance_satisfied = true;
    Real maximum_component_balance_residual{0.0};
    std::array<long double, 9> component_sums{};
    std::array<long double, 5> component_absolute_sums{};
    const auto finalize_component = [&](auto& component) {
        component.low_order_balance_residual =
            component.low_order_liquid_measure -
            component.previous_liquid_measure -
            component.physical_boundary_mass_transfer -
            component.discrete_divergence_mass_source -
            component.low_order_interior_mass_transfer;
        component.raw_target_balance_residual =
            component.raw_target_liquid_measure -
            component.previous_liquid_measure -
            component.physical_boundary_mass_transfer -
            component.discrete_divergence_mass_source -
            component.low_order_interior_mass_transfer -
            component.raw_antidiffusive_mass_transfer;
        component.limited_balance_residual =
            component.limited_liquid_measure -
            component.previous_liquid_measure -
            component.physical_boundary_mass_transfer -
            component.discrete_divergence_mass_source -
            component.low_order_interior_mass_transfer -
            component.limited_antidiffusive_mass_transfer;
        const Real tolerance = scaledTolerance(
            options.invariant_tolerance,
            {component.previous_liquid_measure,
             component.low_order_liquid_measure,
             component.raw_target_liquid_measure,
             component.limited_liquid_measure,
             component.physical_boundary_mass_transfer,
             component.discrete_divergence_mass_source,
             component.low_order_interior_mass_transfer,
             component.raw_antidiffusive_mass_transfer,
             component.limited_antidiffusive_mass_transfer});
        const Real maximum_residual = std::max(
            {std::abs(component.low_order_balance_residual),
             std::abs(component.raw_target_balance_residual),
             std::abs(component.limited_balance_residual)});
        maximum_component_balance_residual = std::max(
            maximum_component_balance_residual, maximum_residual);
        component_balance_satisfied =
            component_balance_satisfied &&
            maximum_residual <= tolerance;
        const std::array<Real, 9> values{{
            component.previous_liquid_measure,
            component.low_order_liquid_measure,
            component.raw_target_liquid_measure,
            component.limited_liquid_measure,
            component.physical_boundary_mass_transfer,
            component.discrete_divergence_mass_source,
            component.low_order_interior_mass_transfer,
            component.raw_antidiffusive_mass_transfer,
            component.limited_antidiffusive_mass_transfer,
        }};
        for (std::size_t index = 0u; index < values.size(); ++index) {
            component_sums[index] +=
                static_cast<long double>(values[index]);
        }
        for (std::size_t index = 0u; index < 5u; ++index) {
            component_absolute_sums[index] += std::abs(
                static_cast<long double>(values[index + 4u]));
        }
    };

    std::size_t component_index{0u};
    for (std::size_t root = 0u; root < node_count; ++root) {
        const auto& values = scratch[root];
        if (values.component_nodes == 0u) {
            continue;
        }
        LevelSetPhaseFluxComponentLedger expected{
            .component_id = static_cast<GlobalIndex>(root),
            .nodes = values.component_nodes,
            .previous_liquid_measure = values.row_divergence,
            .low_order_liquid_measure = values.direct_strong_rate,
            .raw_target_liquid_measure = values.decomposed_strong_rate,
            .limited_liquid_measure = values.nodal_courant,
            .physical_boundary_mass_transfer =
                values.low_order_internal,
            .discrete_divergence_mass_source = values.raw_internal,
            .low_order_interior_mass_transfer =
                values.limited_internal,
            .raw_antidiffusive_mass_transfer = values.positive_raw,
            .limited_antidiffusive_mass_transfer = values.negative_raw,
        };
        finalize_component(expected);
        if (component_index >= correction.components.size() ||
            !samePhaseFluxComponentExact(
                expected, correction.components[component_index])) {
            return fail("found a connected-component ledger mismatch");
        }
        ++component_index;
    }
    if (component_index != correction.components.size() ||
        correction.subthreshold_component_present !=
            subthreshold_component_present) {
        return fail("found a component-count mismatch");
    }
    if (subthreshold_component_present) {
        finalize_component(expected_subthreshold);
    }
    if (!samePhaseFluxComponentExact(
            expected_subthreshold,
            correction.subthreshold_component)) {
        return fail("found a subthreshold component-ledger mismatch");
    }

    const Real previous_component_closure =
        static_cast<Real>(component_sums[0]) - total_previous_value;
    const Real low_order_component_closure =
        static_cast<Real>(component_sums[1]) - total_low_order_value;
    const Real raw_target_component_closure =
        static_cast<Real>(component_sums[2]) - total_raw_target_value;
    const Real limited_component_closure =
        static_cast<Real>(component_sums[3]) - total_limited_value;
    const Real boundary_component_closure =
        static_cast<Real>(component_sums[4]) - total_boundary_value;
    const Real divergence_component_closure =
        static_cast<Real>(component_sums[5]) - total_divergence_value;
    const Real low_order_transfer_closure =
        static_cast<Real>(component_sums[6]);
    const Real raw_transfer_closure =
        static_cast<Real>(component_sums[7]);
    const Real limited_transfer_closure =
        static_cast<Real>(component_sums[8]);
    const Real component_closure_tolerance = scaledTolerance(
        options.invariant_tolerance,
        {total_previous_value, total_low_order_value,
         total_raw_target_value, total_limited_value,
         total_boundary_value, total_divergence_value,
         static_cast<Real>(component_absolute_sums[0]),
         static_cast<Real>(component_absolute_sums[1]),
         static_cast<Real>(component_absolute_sums[2]),
         static_cast<Real>(component_absolute_sums[3]),
         static_cast<Real>(component_absolute_sums[4])});
    const bool component_measure_closure_satisfied =
        std::abs(previous_component_closure) <=
            component_closure_tolerance &&
        std::abs(low_order_component_closure) <=
            component_closure_tolerance &&
        std::abs(raw_target_component_closure) <=
            component_closure_tolerance &&
        std::abs(limited_component_closure) <=
            component_closure_tolerance &&
        std::abs(boundary_component_closure) <=
            component_closure_tolerance &&
        std::abs(divergence_component_closure) <=
            component_closure_tolerance &&
        std::abs(low_order_transfer_closure) <=
            component_closure_tolerance &&
        std::abs(raw_transfer_closure) <=
            component_closure_tolerance &&
        std::abs(limited_transfer_closure) <=
            component_closure_tolerance;
    if (correction.component_balance_satisfied !=
            component_balance_satisfied ||
        correction.component_measure_closure_satisfied !=
            component_measure_closure_satisfied ||
        !same_real(
            correction.component_activity_tolerance,
            options.component_activity_tolerance) ||
        !same_real(
            correction.maximum_component_balance_residual,
            maximum_component_balance_residual) ||
        !same_real(
            correction.previous_component_measure_closure_residual,
            previous_component_closure) ||
        !same_real(
            correction.low_order_component_measure_closure_residual,
            low_order_component_closure) ||
        !same_real(
            correction.raw_target_component_measure_closure_residual,
            raw_target_component_closure) ||
        !same_real(
            correction.limited_component_measure_closure_residual,
            limited_component_closure) ||
        !same_real(
            correction.boundary_component_transfer_closure_residual,
            boundary_component_closure) ||
        !same_real(
            correction.divergence_component_source_closure_residual,
            divergence_component_closure) ||
        !same_real(
            correction.low_order_component_transfer_closure_residual,
            low_order_transfer_closure) ||
        !same_real(
            correction.raw_component_transfer_closure_residual,
            raw_transfer_closure) ||
        !same_real(
            correction.limited_component_transfer_closure_residual,
            limited_transfer_closure)) {
        return fail("found a component summary mismatch");
    }

    const bool correction_success =
        low_order_bounds_satisfied && limited_bounds_satisfied &&
        interior_cancellation_satisfied && local_balance_satisfied &&
        global_balance_satisfied && component_balance_satisfied &&
        component_measure_closure_satisfied &&
        (!options.require_constant_preservation ||
         constant_preservation_satisfied);
    if (correction.success != correction_success ||
        stage.success != correction_success) {
        return fail("found an inconsistent final success state");
    }
    return true;
}

[[nodiscard]] Real norm(const Vector3& value, int dimension) noexcept
{
    return std::sqrt(std::max(Real{0.0}, dot(value, value, dimension)));
}

void addScaled(Vector3& target,
               const math::Vector<Real, 3>& source,
               Real scale,
               int dimension) noexcept
{
    for (int d = 0; d < dimension; ++d) {
        target[static_cast<std::size_t>(d)] +=
            scale * source[static_cast<std::size_t>(d)];
    }
}

void add(Vector3& target, const Vector3& source, int dimension) noexcept
{
    for (int d = 0; d < dimension; ++d) {
        target[static_cast<std::size_t>(d)] +=
            source[static_cast<std::size_t>(d)];
    }
}

[[nodiscard]] int resolvedQuadratureOrder(int requested,
                                          int geometry_order)
{
    if (requested > 0) {
        return requested;
    }
    return std::max(4, 3 * std::max(1, geometry_order));
}

[[nodiscard]] std::shared_ptr<geometry::GeometryMapping> makeCellMapping(
    const assembly::IMeshAccess& mesh,
    GlobalIndex cell)
{
    std::vector<std::array<Real, 3>> coordinates;
    mesh.getCellCoordinates(cell, coordinates);
    if (coordinates.empty()) {
        throw std::invalid_argument(
            "P1 conservative phase graph found a cell without geometry nodes");
    }

    std::vector<math::Vector<Real, 3>> nodes;
    nodes.reserve(coordinates.size());
    for (const auto& coordinate : coordinates) {
        nodes.push_back(math::Vector<Real, 3>{
            coordinate[0], coordinate[1], coordinate[2]});
    }

    geometry::MappingRequest request;
    request.element_type = mesh.getCellType(cell);
    request.geometry_order = mesh.getCellGeometryOrder(cell);
    request.use_affine = request.geometry_order <= 1;
    return geometry::MappingFactory::create(request, nodes);
}

} // namespace

LevelSetP1PhaseGraphIdentity levelSetP1PhaseGraphIdentity(
    const LevelSetP1PhaseTransportGraph& graph) noexcept
{
    std::uint64_t hash = kPhaseStageHashOffset;
    mixPhaseStageHashWord(hash, UINT64_C(0x4752415048315631));
    mixPhaseStageHashWord(hash, 1u);
    mixPhaseStageHashBool(hash, graph.success);
    mixPhaseStageHashBool(hash, graph.partition_of_unity_satisfied);
    mixPhaseStageHashBool(hash, graph.gradient_partition_satisfied);
    mixPhaseStageHashBool(hash, graph.positive_control_volumes_satisfied);
    mixPhaseStageHashBool(hash, graph.gradient_row_sum_satisfied);
    mixPhaseStageHashBool(hash, graph.measure_closure_satisfied);
    mixPhaseStageHashBool(hash, graph.edge_ownership_satisfied);
    mixPhaseStageHashBool(hash, graph.distributed);
    mixPhaseStageHashBool(hash, graph.replicated_sparse_graph);
    mixPhaseStageHashSigned(hash, graph.dimension);
    mixPhaseStageHashSigned(hash, graph.parallel_size);
    mixPhaseStageHashSigned(hash, graph.maximum_quadrature_order);
    mixPhaseStageHashSize(hash, graph.cells);
    mixPhaseStageHashSize(hash, graph.nodes);
    mixPhaseStageHashWord(hash, graph.dof_layout_revision);
    mixPhaseStageHashReal(hash, graph.physical_measure);
    mixPhaseStageHashReal(hash, graph.total_lumped_control_volume);
    mixPhaseStageHashReal(hash, graph.minimum_lumped_control_volume);
    mixPhaseStageHashReal(hash, graph.minimum_jacobian_determinant);
    mixPhaseStageHashReal(hash, graph.maximum_partition_of_unity_residual);
    mixPhaseStageHashReal(hash, graph.maximum_gradient_partition_residual);
    mixPhaseStageHashReal(hash, graph.maximum_gradient_row_sum_residual);
    mixPhaseStageHashReal(hash, graph.measure_closure_residual);
    mixPhaseStageHashSize(hash, graph.lumped_control_volume.size());
    for (const Real value : graph.lumped_control_volume) {
        mixPhaseStageHashReal(hash, value);
    }
    mixPhaseStageHashSize(hash, graph.diagonal_gradient.size());
    for (const auto& value : graph.diagonal_gradient) {
        for (const Real component : value) {
            mixPhaseStageHashReal(hash, component);
        }
    }
    mixPhaseStageHashSize(hash, graph.boundary_column_sum.size());
    for (const auto& value : graph.boundary_column_sum) {
        for (const Real component : value) {
            mixPhaseStageHashReal(hash, component);
        }
    }
    mixPhaseStageHashSize(hash, graph.edges.size());
    for (const auto& edge : graph.edges) {
        mixPhaseStageHashSigned(hash, edge.first_node);
        mixPhaseStageHashSigned(hash, edge.second_node);
        mixPhaseStageHashSigned(hash, edge.owner_rank);
        for (const Real component : edge.first_test_second_gradient) {
            mixPhaseStageHashReal(hash, component);
        }
        for (const Real component : edge.second_test_first_gradient) {
            mixPhaseStageHashReal(hash, component);
        }
    }
    return LevelSetP1PhaseGraphIdentity{
        .dimension = graph.dimension,
        .nodes = graph.nodes,
        .edges = graph.edges.size(),
        .geometry_revision = graph.geometry_revision,
        .topology_revision = graph.topology_revision,
        .ownership_revision = graph.ownership_revision,
        .numbering_revision = graph.numbering_revision,
        .dof_layout_revision = graph.dof_layout_revision,
        .content_revision = finishPhaseStageHash(hash),
    };
}

const char* levelSetP1PhaseSplitSchemeName(
    LevelSetP1PhaseSplitScheme scheme) noexcept
{
    switch (scheme) {
        case LevelSetP1PhaseSplitScheme::
            BackwardEulerExplicitIndicatorEndpointVelocity:
            return "backward_euler_explicit_indicator_endpoint_velocity";
        case LevelSetP1PhaseSplitScheme::GeneralizedAlphaUnsupported:
            return "generalized_alpha_unsupported";
    }
    return "unknown";
}

std::uint64_t levelSetP1PhaseScalarContentRevision(
    std::span<const Real> values) noexcept
{
    std::uint64_t hash = kPhaseStageHashOffset;
    mixPhaseStageHashWord(hash, UINT64_C(0x514e5343414c4152));
    mixPhaseStageHashWord(hash, 1u);
    mixPhaseStageHashSize(hash, values.size());
    for (const Real value : values) {
        mixPhaseStageHashReal(hash, value);
    }
    return finishPhaseStageHash(hash);
}

std::uint64_t levelSetP1PhaseVelocityContentRevision(
    std::span<const std::array<Real, 3>> values) noexcept
{
    std::uint64_t hash = kPhaseStageHashOffset;
    mixPhaseStageHashWord(hash, UINT64_C(0x554e3156454c4f43));
    mixPhaseStageHashWord(hash, 1u);
    mixPhaseStageHashSize(hash, values.size());
    for (const auto& value : values) {
        for (const Real component : value) {
            mixPhaseStageHashReal(hash, component);
        }
    }
    return finishPhaseStageHash(hash);
}

std::uint64_t levelSetP1PhaseFluxLedgerDigest(
    const LevelSetP1PhaseTransportStageResult& stage) noexcept
{
    std::uint64_t hash = kPhaseStageHashOffset;
    mixPhaseStageHashWord(hash, UINT64_C(0x504831464c555831));
    mixPhaseStageHashWord(hash, 2u);
    mixPhaseStageHashBool(hash, stage.success);
    mixPhaseStageHashBool(hash, stage.courant_satisfied);
    mixPhaseStageHashBool(hash, stage.low_order_coefficients_nonnegative);
    mixPhaseStageHashBool(hash, stage.strong_form_decomposition_satisfied);
    mixPhaseStageHashBool(hash, stage.replicated_stage_inputs_satisfied);
    mixPhaseStageHashReal(hash, stage.time_step);
    mixPhaseStageHashReal(hash, stage.maximum_courant);
    mixPhaseStageHashReal(hash, stage.minimum_low_order_coefficient);
    mixPhaseStageHashReal(
        hash, stage.maximum_strong_form_decomposition_residual);
    mixPhaseStageHashReal(
        hash, stage.executed_options.invariant_tolerance);
    mixPhaseStageHashReal(
        hash, stage.executed_options.component_activity_tolerance);
    mixPhaseStageHashReal(
        hash, stage.executed_options.maximum_courant);
    mixPhaseStageHashBool(
        hash, stage.executed_options.enforce_courant_limit);
    mixPhaseStageHashBool(
        hash, stage.executed_options.require_constant_preservation);
    mixPhaseStageHashSize(hash, stage.nodal_courant.size());
    for (const Real value : stage.nodal_courant) {
        mixPhaseStageHashReal(hash, value);
    }
    mixPhaseStageHashSize(
        hash, stage.physical_boundary_mass_transfer.size());
    for (const Real value : stage.physical_boundary_mass_transfer) {
        mixPhaseStageHashReal(hash, value);
    }
    mixPhaseStageHashSize(
        hash, stage.discrete_divergence_mass_source.size());
    for (const Real value : stage.discrete_divergence_mass_source) {
        mixPhaseStageHashReal(hash, value);
    }
    mixPhaseStageHashSize(hash, stage.flux_edges.size());
    for (const auto& edge : stage.flux_edges) {
        mixPhaseStageHashSigned(hash, edge.first_node);
        mixPhaseStageHashSigned(hash, edge.second_node);
        mixPhaseStageHashReal(hash, edge.low_order_mass_transfer);
        mixPhaseStageHashReal(hash, edge.raw_antidiffusive_mass_transfer);
    }

    const auto& correction = stage.correction;
    mixPhaseStageHashBool(hash, correction.success);
    mixPhaseStageHashBool(hash, correction.applied);
    mixPhaseStageHashBool(hash, correction.low_order_bounds_satisfied);
    mixPhaseStageHashBool(hash, correction.limited_bounds_satisfied);
    mixPhaseStageHashBool(hash, correction.interior_cancellation_satisfied);
    mixPhaseStageHashBool(hash, correction.local_balance_satisfied);
    mixPhaseStageHashBool(hash, correction.global_balance_satisfied);
    mixPhaseStageHashBool(hash, correction.component_balance_satisfied);
    mixPhaseStageHashBool(
        hash, correction.component_measure_closure_satisfied);
    mixPhaseStageHashBool(hash, correction.subthreshold_component_present);
    mixPhaseStageHashBool(hash, correction.constant_state_input);
    mixPhaseStageHashBool(hash, correction.constant_preservation_required);
    mixPhaseStageHashBool(hash, correction.constant_preservation_satisfied);
    mixPhaseStageHashSize(hash, correction.limited_edges);
    mixPhaseStageHashReal(hash, correction.total_previous_liquid_measure);
    mixPhaseStageHashReal(hash, correction.total_low_order_liquid_measure);
    mixPhaseStageHashReal(hash, correction.total_raw_target_liquid_measure);
    mixPhaseStageHashReal(hash, correction.total_limited_liquid_measure);
    mixPhaseStageHashReal(
        hash, correction.total_physical_boundary_mass_transfer);
    mixPhaseStageHashReal(
        hash, correction.total_discrete_divergence_mass_source);
    mixPhaseStageHashReal(
        hash, correction.low_order_nodal_cancellation_residual);
    mixPhaseStageHashReal(hash, correction.raw_nodal_cancellation_residual);
    mixPhaseStageHashReal(
        hash, correction.limited_nodal_cancellation_residual);
    mixPhaseStageHashReal(
        hash, correction.maximum_edge_pair_cancellation_residual);
    mixPhaseStageHashReal(
        hash, correction.maximum_low_order_local_mass_balance_residual);
    mixPhaseStageHashReal(
        hash, correction.maximum_raw_target_local_mass_balance_residual);
    mixPhaseStageHashReal(
        hash, correction.maximum_local_mass_balance_residual);
    mixPhaseStageHashReal(
        hash, correction.maximum_component_balance_residual);
    mixPhaseStageHashReal(hash, correction.component_activity_tolerance);
    mixPhaseStageHashReal(
        hash, correction.previous_component_measure_closure_residual);
    mixPhaseStageHashReal(
        hash, correction.low_order_component_measure_closure_residual);
    mixPhaseStageHashReal(
        hash, correction.raw_target_component_measure_closure_residual);
    mixPhaseStageHashReal(
        hash, correction.limited_component_measure_closure_residual);
    mixPhaseStageHashReal(
        hash, correction.boundary_component_transfer_closure_residual);
    mixPhaseStageHashReal(
        hash, correction.divergence_component_source_closure_residual);
    mixPhaseStageHashReal(
        hash, correction.low_order_component_transfer_closure_residual);
    mixPhaseStageHashReal(
        hash, correction.raw_component_transfer_closure_residual);
    mixPhaseStageHashReal(
        hash, correction.limited_component_transfer_closure_residual);
    mixPhaseStageHashReal(
        hash, correction.low_order_global_mass_balance_residual);
    mixPhaseStageHashReal(
        hash, correction.raw_target_global_mass_balance_residual);
    mixPhaseStageHashReal(hash, correction.global_mass_balance_residual);
    mixPhaseStageHashReal(
        hash, correction.maximum_constant_preservation_error);
    mixPhaseStageHashReal(
        hash, correction.minimum_low_order_liquid_indicator);
    mixPhaseStageHashReal(
        hash, correction.maximum_low_order_liquid_indicator);
    mixPhaseStageHashReal(
        hash, correction.minimum_raw_target_liquid_indicator);
    mixPhaseStageHashReal(
        hash, correction.maximum_raw_target_liquid_indicator);
    mixPhaseStageHashReal(
        hash, correction.minimum_limited_liquid_indicator);
    mixPhaseStageHashReal(
        hash, correction.maximum_limited_liquid_indicator);
    mixPhaseStageHashSize(hash, correction.nodes.size());
    for (const auto& node : correction.nodes) {
        mixPhaseFluxNodeLedger(hash, node);
    }
    mixPhaseStageHashSize(hash, correction.edges.size());
    for (const auto& edge : correction.edges) {
        mixPhaseFluxEdgeLedger(hash, edge);
    }
    mixPhaseStageHashSize(hash, correction.node_component_ids.size());
    for (const auto component_id : correction.node_component_ids) {
        mixPhaseStageHashSigned(hash, component_id);
    }
    mixPhaseStageHashSize(hash, correction.components.size());
    for (const auto& component : correction.components) {
        mixPhaseFluxComponentLedger(hash, component);
    }
    mixPhaseFluxComponentLedger(hash, correction.subthreshold_component);
    return finishPhaseStageHash(hash);
}

LevelSetP1PhaseTransportGraph buildLevelSetP1PhaseTransportGraph(
    const systems::FESystem& system,
    FieldId liquid_indicator_field,
    const LevelSetP1PhaseGraphOptions& options)
{
    LevelSetP1PhaseTransportGraph result;
    try {
        if (liquid_indicator_field == INVALID_FIELD_ID) {
            result.diagnostic =
                "P1 conservative phase graph received an invalid field";
            return result;
        }

        const auto& mesh = system.meshAccess();
        result.dimension = mesh.dimension();
        const auto& record = system.fieldRecord(liquid_indicator_field);
        const auto& dofs = system.fieldDofHandler(liquid_indicator_field);
        const auto collective = phaseGraphCollectiveContext(mesh, dofs);
        result.parallel_rank = collective.rank;
        result.parallel_size = collective.size;
        result.distributed = collective.active;
        result.replicated_sparse_graph = collective.active;
#if FE_HAS_MPI
        if (collective.active) {
            std::shared_ptr<detail::LevelSetP1PhaseCollectiveState>
                collective_state;
            std::string communicator_duplication_failure_diagnostic;
            bool local_collective_state_allocation_success = true;
            std::string local_collective_state_allocation_diagnostic;
            try {
                collective_state = std::make_shared<
                    detail::LevelSetP1PhaseCollectiveState>();
                communicator_duplication_failure_diagnostic =
                    "P1 conservative phase graph could not duplicate its field communicator";
            } catch (const std::exception& exception) {
                local_collective_state_allocation_success = false;
                local_collective_state_allocation_diagnostic =
                    exception.what();
            }
            if (!synchronizeLocalFailure(
                    collective,
                    local_collective_state_allocation_success,
                    local_collective_state_allocation_diagnostic,
                    result.diagnostic,
                    "graph communicator-state allocation")) {
                return result;
            }

            MPI_Comm duplicated_communicator = MPI_COMM_NULL;
            const bool local_communicator_duplication_success =
                MPI_Comm_dup(collective.communicator,
                             &duplicated_communicator) == MPI_SUCCESS &&
                duplicated_communicator != MPI_COMM_NULL;
            if (!synchronizeLocalFailure(
                    collective,
                    local_communicator_duplication_success,
                    communicator_duplication_failure_diagnostic,
                    result.diagnostic,
                    "graph communicator duplication")) {
                return result;
            }
            collective_state->communicator = duplicated_communicator;
            result.collective_state = std::move(collective_state);
        }
#endif

        result.geometry_revision = mesh.geometryRevision();
        result.topology_revision = mesh.topologyRevision();
        result.ownership_revision = mesh.ownershipRevision();
        result.numbering_revision = mesh.numberingRevision();
        result.dof_layout_revision = dofs.dofLayoutRevision();

        bool local_preflight_success = true;
        std::string local_preflight_diagnostic;
        const auto reject_preflight = [&](std::string diagnostic) {
            if (local_preflight_success) {
                local_preflight_success = false;
                local_preflight_diagnostic = std::move(diagnostic);
            }
        };
        try {
            if (!std::isfinite(options.invariant_tolerance) ||
                options.invariant_tolerance < Real{0.0} ||
                options.quadrature_order < 0) {
                reject_preflight(
                    "P1 conservative phase graph requires a nonnegative quadrature order and finite nonnegative tolerance");
            }
            if (result.dimension < 2 || result.dimension > 3) {
                reject_preflight(
                    "P1 conservative phase graph supports two- and three-dimensional meshes");
            }
            if (record.components != 1 || !record.space ||
                record.space->space_type() != spaces::SpaceType::H1 ||
                record.space->field_type() != FieldType::Scalar ||
                record.space->continuity() != Continuity::C0 ||
                record.space->value_dimension() != 1 ||
                record.space->is_variable_order() ||
                record.space->polynomial_order() != 1) {
                reject_preflight(
                    "P1 conservative phase graph requires a fixed-order scalar P1 H1 field");
            }
            if (!system.fieldParticipatesInUnknownVector(
                    liquid_indicator_field)) {
                reject_preflight(
                    "P1 conservative phase graph requires a transported unknown field");
            }
            if (dofs.getNumDofs() <= 0 ||
                dofs.getNumLocalDofs() < 0 ||
                dofs.getNumLocalDofs() > dofs.getNumDofs()) {
                reject_preflight(
                    "P1 conservative phase graph requires a valid nonempty field layout");
            }
            if (mesh.numOwnedCells() < 0 ||
                mesh.numOwnedCells() > mesh.numCells()) {
                reject_preflight(
                    "P1 conservative phase graph requires valid owned-cell metadata");
            }
        } catch (const std::exception& exception) {
            local_preflight_success = false;
            local_preflight_diagnostic = exception.what();
        }
        if (!synchronizeLocalFailure(
                collective, local_preflight_success,
                local_preflight_diagnostic, result.diagnostic)) {
            return result;
        }

        const int minimum_dimension = allReduceIntMin(
            collective, result.dimension);
        const int maximum_dimension = allReduceIntMax(
            collective, result.dimension);
        const int minimum_requested_order = allReduceIntMin(
            collective, options.quadrature_order);
        const int maximum_requested_order = allReduceIntMax(
            collective, options.quadrature_order);
        const Real minimum_requested_tolerance = allReduceRealMin(
            collective, options.invariant_tolerance);
        const Real maximum_requested_tolerance = allReduceRealMax(
            collective, options.invariant_tolerance);
        const auto local_node_count =
            static_cast<std::uint64_t>(dofs.getNumDofs());
        const auto minimum_node_count = allReduceUnsigned64Min(
            collective, local_node_count);
        const auto maximum_node_count = allReduceUnsigned64Max(
            collective, local_node_count);
        if (minimum_dimension != maximum_dimension ||
            minimum_requested_order != maximum_requested_order ||
            minimum_requested_tolerance != maximum_requested_tolerance ||
            minimum_node_count != maximum_node_count) {
            result.diagnostic =
                "P1 conservative phase graph requires identical dimension, options, and global field size on every rank";
            return result;
        }
        result.dimension = minimum_dimension;
        result.nodes = static_cast<std::size_t>(minimum_node_count);

        // Mesh revisions are local cache stamps, not communicator-wide
        // topology identities.  Valid partitions can observe different
        // numbers of local geometry, topology, ownership, and numbering
        // events.  Retain each rank's stamps so the graph staleness check
        // remains local, while requiring the replicated FE layout below.
        const auto dof_revision_min = allReduceUnsigned64Min(
            collective, result.dof_layout_revision);
        const auto dof_revision_max = allReduceUnsigned64Max(
            collective, result.dof_layout_revision);
        if (dof_revision_min != dof_revision_max) {
            result.diagnostic =
                "P1 conservative phase graph requires a synchronized field layout revision on every rank";
            return result;
        }
        result.dof_layout_revision = dof_revision_min;

        bool local_constraint_success = true;
        std::string local_constraint_diagnostic;
        try {
            result.lumped_control_volume.assign(
                result.nodes, Real{0.0});
            result.diagonal_gradient.assign(result.nodes, Vector3{});
            result.boundary_column_sum.assign(
                result.nodes, Vector3{});
            const GlobalIndex field_offset =
                system.fieldDofOffset(liquid_indicator_field);
            for (std::size_t i = 0; i < result.nodes; ++i) {
                if (system.constraints().isConstrained(
                        field_offset + static_cast<GlobalIndex>(i))) {
                    local_constraint_success = false;
                    local_constraint_diagnostic =
                        "P1 conservative phase graph does not accept constrained indicator nodes; boundary transport must enter through phase fluxes";
                    break;
                }
            }
        } catch (const std::exception& exception) {
            local_constraint_success = false;
            local_constraint_diagnostic = exception.what();
        }
        if (!synchronizeLocalFailure(
                collective, local_constraint_success,
                local_constraint_diagnostic, result.diagnostic)) {
            return result;
        }

        result.minimum_jacobian_determinant =
            std::numeric_limits<Real>::infinity();

        std::map<std::pair<GlobalIndex, GlobalIndex>, MutableGradientEdge>
            assembled_edges;
        std::vector<Vector3> assembled_row_sum;
        long double local_physical_measure_accumulator = 0.0L;
        Real maximum_gradient_coefficient{0.0};
        Real maximum_physical_basis_gradient{0.0};

        bool local_assembly_success = true;
        std::string local_assembly_diagnostic;
        try {
            assembled_row_sum.assign(result.nodes, Vector3{});
            mesh.forEachOwnedCell([&](GlobalIndex cell) {
                if (record.space->polynomial_order(cell) != 1) {
                    throw std::invalid_argument(
                        "P1 conservative phase graph found a non-P1 cell");
                }
                const auto cell_dofs = dofs.getCellDofs(cell);
                const auto& element = record.space->getElement(
                    mesh.getCellType(cell), cell);
                const auto& basis = element.basis();
                if (basis.is_vector_valued() ||
                    basis.basis_type() != BasisType::Lagrange ||
                    basis.order() != 1 || basis.size() != cell_dofs.size() ||
                    cell_dofs.size() <
                        static_cast<std::size_t>(result.dimension + 1)) {
                    throw std::invalid_argument(
                        "P1 conservative phase graph requires one scalar linear Lagrange basis value per cell DOF");
                }
                for (const auto dof : cell_dofs) {
                    if (dof < 0 ||
                        static_cast<std::size_t>(dof) >= result.nodes) {
                        throw std::invalid_argument(
                            "P1 conservative phase graph found a cell DOF outside the field layout");
                    }
                }

                const auto mapping = makeCellMapping(mesh, cell);
                if (!mapping || mapping->dimension() != result.dimension) {
                    throw std::invalid_argument(
                        "P1 conservative phase graph found an incompatible geometry mapping");
                }
                const int quadrature_order = resolvedQuadratureOrder(
                    options.quadrature_order,
                    mesh.getCellGeometryOrder(cell));
                result.maximum_quadrature_order = std::max(
                    result.maximum_quadrature_order, quadrature_order);
                const auto quadrature = quadrature::QuadratureFactory::create(
                    mesh.getCellType(cell), quadrature_order);
                if (!quadrature || quadrature->num_points() == 0u) {
                    throw std::invalid_argument(
                        "P1 conservative phase graph received an empty quadrature rule");
                }

                const std::size_t local_size = cell_dofs.size();
                std::vector<Real> local_mass(local_size, Real{0.0});
                std::vector<Vector3> local_gradient(
                    local_size * local_size, Vector3{});
                std::vector<Real> values;
                std::vector<basis::Gradient> reference_gradients;
                for (std::size_t q = 0; q < quadrature->num_points(); ++q) {
                    const auto xi = quadrature->point(q);
                    const Real determinant = mapping->jacobian_determinant(xi);
                    const Real quadrature_weight = quadrature->weight(q);
                    if (!std::isfinite(determinant) ||
                        !(determinant > Real{0.0}) ||
                        !std::isfinite(quadrature_weight) ||
                        !(quadrature_weight > Real{0.0})) {
                        throw std::invalid_argument(
                            "P1 conservative phase graph found a nonpositive or non-finite mapped quadrature weight");
                    }
                    result.minimum_jacobian_determinant = std::min(
                        result.minimum_jacobian_determinant, determinant);
                    const Real weight = determinant * quadrature_weight;
                    local_physical_measure_accumulator +=
                        static_cast<long double>(weight);

                    basis.evaluate_values(xi, values);
                    basis.evaluate_gradients(xi, reference_gradients);
                    if (values.size() != local_size ||
                        reference_gradients.size() != local_size) {
                        throw std::invalid_argument(
                            "P1 conservative phase graph received inconsistent basis evaluations");
                    }
                    std::vector<math::Vector<Real, 3>> physical_gradients;
                    physical_gradients.reserve(local_size);
                    const auto inverse = mapping->jacobian_inverse(xi);
                    Real value_sum{0.0};
                    Vector3 gradient_sum{};
                    for (std::size_t local = 0; local < local_size; ++local) {
                        if (!std::isfinite(values[local])) {
                            throw std::invalid_argument(
                                "P1 conservative phase graph found a non-finite basis value");
                        }
                        const auto physical_gradient =
                            mapping->transform_gradient(
                                reference_gradients[local], inverse);
                        for (int d = 0; d < result.dimension; ++d) {
                            if (!std::isfinite(
                                    physical_gradient[static_cast<std::size_t>(d)])) {
                                throw std::invalid_argument(
                                    "P1 conservative phase graph found a non-finite physical basis gradient");
                            }
                        }
                        physical_gradients.push_back(physical_gradient);
                        Vector3 physical_gradient_array{};
                        for (int d = 0; d < result.dimension; ++d) {
                            physical_gradient_array[static_cast<std::size_t>(d)] =
                                physical_gradient[static_cast<std::size_t>(d)];
                        }
                        maximum_physical_basis_gradient = std::max(
                            maximum_physical_basis_gradient,
                            norm(physical_gradient_array, result.dimension));
                        value_sum += values[local];
                        addScaled(gradient_sum, physical_gradient,
                                  Real{1.0}, result.dimension);
                        local_mass[local] += weight * values[local];
                    }
                    result.maximum_partition_of_unity_residual = std::max(
                        result.maximum_partition_of_unity_residual,
                        std::abs(value_sum - Real{1.0}));
                    result.maximum_gradient_partition_residual = std::max(
                        result.maximum_gradient_partition_residual,
                        norm(gradient_sum, result.dimension));

                    for (std::size_t i = 0; i < local_size; ++i) {
                        for (std::size_t j = 0; j < local_size; ++j) {
                            addScaled(local_gradient[i * local_size + j],
                                      physical_gradients[j],
                                      weight * values[i], result.dimension);
                        }
                    }
                }

                for (std::size_t i = 0; i < local_size; ++i) {
                    const auto global_i = cell_dofs[i];
                    const auto node_i = static_cast<std::size_t>(global_i);
                    if (!std::isfinite(local_mass[i]) ||
                        !(local_mass[i] > Real{0.0})) {
                        throw std::invalid_argument(
                            "P1 conservative phase graph found a nonpositive local lumped control volume");
                    }
                    result.lumped_control_volume[node_i] += local_mass[i];
                    for (std::size_t j = 0; j < local_size; ++j) {
                        const auto global_j = cell_dofs[j];
                        const auto& coefficient =
                            local_gradient[i * local_size + j];
                        add(assembled_row_sum[node_i], coefficient,
                            result.dimension);
                        add(result.boundary_column_sum[
                                static_cast<std::size_t>(global_j)],
                            coefficient, result.dimension);
                        maximum_gradient_coefficient = std::max(
                            maximum_gradient_coefficient,
                            norm(coefficient, result.dimension));
                        if (i == j) {
                            add(result.diagonal_gradient[node_i], coefficient,
                                result.dimension);
                            continue;
                        }
                        const auto endpoints = std::minmax(global_i, global_j);
                        auto& edge = assembled_edges[{endpoints.first,
                                                      endpoints.second}];
                        if (global_i == endpoints.first) {
                            add(edge.first_test_second_gradient, coefficient,
                                result.dimension);
                        } else {
                            add(edge.second_test_first_gradient, coefficient,
                                result.dimension);
                        }
                    }
                }
                ++result.cells;
            });
        } catch (const std::exception& exception) {
            local_assembly_success = false;
            local_assembly_diagnostic = exception.what();
        }
        if (local_assembly_success &&
            result.cells !=
                static_cast<std::size_t>(mesh.numOwnedCells())) {
            local_assembly_success = false;
            local_assembly_diagnostic =
                "P1 conservative phase graph owned-cell traversal count does not match mesh metadata";
        }
        if (!synchronizeLocalFailure(
                collective, local_assembly_success,
                local_assembly_diagnostic, result.diagnostic)) {
            return result;
        }
        result.local_owned_cells = result.cells;
        result.physical_measure = static_cast<Real>(
            local_physical_measure_accumulator);

        const auto global_cell_count = allReduceUnsigned64Sum(
            collective, static_cast<std::uint64_t>(result.cells));
        if (global_cell_count >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())) {
            result.diagnostic =
                "P1 conservative phase graph global cell count exceeds local index capacity";
            return result;
        }
        result.cells = static_cast<std::size_t>(global_cell_count);
        result.physical_measure = allReduceRealSum(
            collective, result.physical_measure);
        result.minimum_jacobian_determinant = allReduceRealMin(
            collective, result.minimum_jacobian_determinant);
        result.maximum_partition_of_unity_residual = allReduceRealMax(
            collective, result.maximum_partition_of_unity_residual);
        result.maximum_gradient_partition_residual = allReduceRealMax(
            collective, result.maximum_gradient_partition_residual);
        maximum_gradient_coefficient = allReduceRealMax(
            collective, maximum_gradient_coefficient);
        maximum_physical_basis_gradient = allReduceRealMax(
            collective, maximum_physical_basis_gradient);
        result.maximum_quadrature_order = allReduceIntMax(
            collective, result.maximum_quadrature_order);
        allReduceRealBufferSum(collective,
                               result.lumped_control_volume);
        allReduceVector3Sum(collective, result.diagonal_gradient);
        allReduceVector3Sum(collective, result.boundary_column_sum);
        allReduceVector3Sum(collective, assembled_row_sum);
        bool local_edge_globalization_success = true;
        std::string local_edge_globalization_diagnostic;
        try {
            assembled_edges = globalizeGradientEdges(
                collective, assembled_edges);
        } catch (const std::exception& exception) {
            local_edge_globalization_success = false;
            local_edge_globalization_diagnostic = exception.what();
        }
        if (!synchronizeLocalFailure(
                collective,
                local_edge_globalization_success,
                local_edge_globalization_diagnostic,
                result.diagnostic,
                "graph edge globalization")) {
            return result;
        }
        maximum_gradient_coefficient = Real{0.0};
        for (const auto& coefficient : result.diagonal_gradient) {
            maximum_gradient_coefficient = std::max(
                maximum_gradient_coefficient,
                norm(coefficient, result.dimension));
        }
        for (const auto& [endpoints, edge] : assembled_edges) {
            (void)endpoints;
            maximum_gradient_coefficient = std::max(
                maximum_gradient_coefficient,
                std::max(norm(edge.first_test_second_gradient,
                              result.dimension),
                         norm(edge.second_test_first_gradient,
                              result.dimension)));
        }

        if (result.cells == 0u ||
            !std::isfinite(result.physical_measure) ||
            !(result.physical_measure > Real{0.0}) ||
            !std::isfinite(result.minimum_jacobian_determinant)) {
            result.diagnostic =
                "P1 conservative phase graph found no valid owned-cell measure";
            return result;
        }

        result.minimum_lumped_control_volume =
            std::numeric_limits<Real>::infinity();
        long double volume_sum = 0.0L;
        result.positive_control_volumes_satisfied = true;
        for (std::size_t node = 0; node < result.nodes; ++node) {
            const Real volume = result.lumped_control_volume[node];
            result.positive_control_volumes_satisfied =
                result.positive_control_volumes_satisfied &&
                std::isfinite(volume) && volume > Real{0.0};
            result.minimum_lumped_control_volume = std::min(
                result.minimum_lumped_control_volume, volume);
            volume_sum += static_cast<long double>(volume);
            result.maximum_gradient_row_sum_residual = std::max(
                result.maximum_gradient_row_sum_residual,
                norm(assembled_row_sum[node], result.dimension));
        }
        if (!result.positive_control_volumes_satisfied) {
            result.diagnostic =
                "P1 conservative phase graph left a node without positive owned-cell measure";
            return result;
        }
        result.total_lumped_control_volume = static_cast<Real>(volume_sum);
        result.measure_closure_residual =
            result.total_lumped_control_volume - result.physical_measure;

        const Real partition_tolerance =
            scaledTolerance(options.invariant_tolerance, {Real{1.0}});
        const Real gradient_partition_tolerance = scaledTolerance(
            options.invariant_tolerance,
            {maximum_physical_basis_gradient});
        const Real row_sum_tolerance = scaledTolerance(
            options.invariant_tolerance, {maximum_gradient_coefficient});
        const Real measure_tolerance = scaledTolerance(
            options.invariant_tolerance,
            {result.physical_measure, result.total_lumped_control_volume});
        result.partition_of_unity_satisfied =
            result.maximum_partition_of_unity_residual <=
            partition_tolerance;
        result.gradient_partition_satisfied =
            result.maximum_gradient_partition_residual <=
            gradient_partition_tolerance;
        result.gradient_row_sum_satisfied =
            result.maximum_gradient_row_sum_residual <= row_sum_tolerance;
        result.measure_closure_satisfied =
            std::abs(result.measure_closure_residual) <= measure_tolerance;
        if (!result.partition_of_unity_satisfied ||
            !result.gradient_partition_satisfied ||
            !result.gradient_row_sum_satisfied ||
            !result.measure_closure_satisfied) {
            result.diagnostic =
                "P1 conservative phase graph failed a partition, gradient, or measure identity";
            return result;
        }

        std::vector<int> local_edge_owners;
        bool local_ownership_success = true;
        std::string ownership_diagnostic;
        try {
            local_edge_owners.reserve(assembled_edges.size());
            for (const auto& [endpoints, edge] : assembled_edges) {
                (void)edge;
                int owner_rank = 0;
                if (collective.active) {
                    const int first_owner =
                        dofs.getDofMap().getDofOwner(endpoints.first);
                    const int second_owner =
                        dofs.getDofMap().getDofOwner(endpoints.second);
                    if (first_owner < 0 || second_owner < 0 ||
                        first_owner >= collective.size ||
                        second_owner >= collective.size) {
                        local_ownership_success = false;
                        ownership_diagnostic =
                            "P1 conservative phase graph could not resolve a valid owner for every algebraic edge";
                        break;
                    }
                    owner_rank = std::min(first_owner, second_owner);
                }
                local_edge_owners.push_back(owner_rank);
            }
        } catch (const std::exception& exception) {
            local_ownership_success = false;
            ownership_diagnostic = exception.what();
        }
        if (!synchronizeLocalFailure(
                collective, local_ownership_success,
                ownership_diagnostic, result.diagnostic)) {
            result.edge_ownership_satisfied = false;
            return result;
        }

        std::vector<int> minimum_edge_owners;
        std::vector<int> maximum_edge_owners;
        bool local_owner_buffer_success = true;
        std::string local_owner_buffer_diagnostic;
        try {
            minimum_edge_owners = local_edge_owners;
            maximum_edge_owners = local_edge_owners;
        } catch (const std::exception& exception) {
            local_owner_buffer_success = false;
            local_owner_buffer_diagnostic = exception.what();
        }
        if (!synchronizeLocalFailure(
                collective,
                local_owner_buffer_success,
                local_owner_buffer_diagnostic,
                result.diagnostic,
                "graph edge-owner reduction preparation")) {
            return result;
        }
        allReduceIntBufferMinMax(collective, local_edge_owners,
                                 minimum_edge_owners,
                                 maximum_edge_owners);
        result.edge_ownership_satisfied =
            minimum_edge_owners == maximum_edge_owners;
        if (!result.edge_ownership_satisfied) {
            result.diagnostic =
                "P1 conservative phase graph found inconsistent algebraic edge ownership across ranks";
            return result;
        }

        bool local_graph_materialization_success = true;
        std::string local_graph_materialization_diagnostic;
        try {
            result.edges.reserve(assembled_edges.size());
            std::size_t edge_index = 0u;
            for (const auto& [endpoints, edge] : assembled_edges) {
                const int owner_rank = minimum_edge_owners[edge_index++];
                result.edges.push_back(LevelSetP1PhaseGradientEdge{
                    .first_node = endpoints.first,
                    .second_node = endpoints.second,
                    .owner_rank = owner_rank,
                    .first_test_second_gradient =
                        edge.first_test_second_gradient,
                    .second_test_first_gradient =
                        edge.second_test_first_gradient,
                });
                if (owner_rank == collective.rank) {
                    ++result.locally_owned_edges;
                }
            }
        } catch (const std::exception& exception) {
            local_graph_materialization_success = false;
            local_graph_materialization_diagnostic = exception.what();
        }
        if (!synchronizeLocalFailure(
                collective,
                local_graph_materialization_success,
                local_graph_materialization_diagnostic,
                result.diagnostic,
                "graph result materialization")) {
            return result;
        }
        result.success = true;
        result.diagnostic = "ok";
    } catch (const std::exception& exception) {
        result.success = false;
        result.diagnostic = exception.what();
    }
    return result;
}

LevelSetP1PhaseTransportStageResult
advanceLevelSetP1ConservativePhaseStage(
    const LevelSetP1PhaseTransportGraph& graph,
    std::span<const Real> previous_liquid_indicator,
    std::span<const Real> lower_liquid_indicator,
    std::span<const Real> upper_liquid_indicator,
    std::span<const std::array<Real, 3>> nodal_velocity,
    Real time_step,
    const LevelSetP1PhaseStageOptions& options)
{
    LevelSetP1PhaseTransportStageResult result;
    try {
        const auto collective = phaseStageCollectiveContext(graph);
        const std::size_t node_count = graph.nodes;
        bool local_preflight_success = true;
        std::string local_preflight_diagnostic;
        const auto reject_preflight = [&](std::string diagnostic) {
            if (local_preflight_success) {
                local_preflight_success = false;
                local_preflight_diagnostic = std::move(diagnostic);
            }
        };
        if (!graph.success) {
            reject_preflight(
                "P1 conservative phase stage requires a valid assembled graph");
        }
        if (graph.parallel_rank != collective.rank ||
            graph.parallel_size != collective.size ||
            (collective.active &&
             (!graph.distributed || !graph.replicated_sparse_graph)) ||
            (!collective.active && graph.distributed)) {
            reject_preflight(
                "P1 conservative phase stage graph and field communicator metadata disagree");
        }
        if (node_count == 0u ||
            graph.dimension < 2 || graph.dimension > 3 ||
            graph.lumped_control_volume.size() != node_count ||
            graph.diagonal_gradient.size() != node_count ||
            graph.boundary_column_sum.size() != node_count ||
            previous_liquid_indicator.size() != node_count ||
            lower_liquid_indicator.size() != node_count ||
            upper_liquid_indicator.size() != node_count ||
            nodal_velocity.size() != node_count) {
            reject_preflight(
                "P1 conservative phase stage received inconsistent nodal spans");
        }
        if (!(time_step > Real{0.0}) || !std::isfinite(time_step) ||
            !std::isfinite(options.invariant_tolerance) ||
            options.invariant_tolerance < Real{0.0} ||
            !std::isfinite(options.component_activity_tolerance) ||
            !(options.component_activity_tolerance > Real{0.0}) ||
            options.component_activity_tolerance > Real{1.0} ||
            !(options.maximum_courant > Real{0.0}) ||
            options.maximum_courant > Real{1.0} ||
            !std::isfinite(options.maximum_courant)) {
            reject_preflight(
                "P1 conservative phase stage requires a positive finite time step, nonnegative finite invariant tolerance, component activity tolerance in (0,1], and Courant limit in (0,1]");
        }
        if (local_preflight_success) {
            for (std::size_t node = 0u; node < node_count; ++node) {
                if (!std::isfinite(graph.lumped_control_volume[node]) ||
                    !(graph.lumped_control_volume[node] > Real{0.0}) ||
                    !std::isfinite(previous_liquid_indicator[node]) ||
                    !std::isfinite(lower_liquid_indicator[node]) ||
                    !std::isfinite(upper_liquid_indicator[node])) {
                    reject_preflight(
                        "P1 conservative phase stage requires finite nodal inputs and positive control volumes");
                    break;
                }
                for (const Real component : nodal_velocity[node]) {
                    if (!std::isfinite(component)) {
                        reject_preflight(
                            "P1 conservative phase stage found a non-finite nodal velocity");
                        break;
                    }
                }
                if (!local_preflight_success) {
                    break;
                }
            }
        }
        if (!synchronizeLocalFailure(
                collective, local_preflight_success,
                local_preflight_diagnostic, result.diagnostic, "stage")) {
            return result;
        }
        if (!replicatedStageInputsMatch(
                collective, previous_liquid_indicator,
                lower_liquid_indicator, upper_liquid_indicator,
                nodal_velocity, time_step, options,
                result.diagnostic)) {
            return result;
        }
        result.replicated_stage_inputs_satisfied = true;

        bool local_endpoint_velocity_capture_success = true;
        std::string local_endpoint_velocity_capture_diagnostic;
        try {
            result.time_step = time_step;
            result.executed_options = options;
            result.sampled_nodal_velocity.assign(
                nodal_velocity.begin(), nodal_velocity.end());
        } catch (const std::exception& exception) {
            local_endpoint_velocity_capture_success = false;
            local_endpoint_velocity_capture_diagnostic = exception.what();
        }
        if (!synchronizeLocalFailure(
                collective,
                local_endpoint_velocity_capture_success,
                local_endpoint_velocity_capture_diagnostic,
                result.diagnostic,
                "endpoint-velocity capture")) {
            return result;
        }

        result.nodal_courant.assign(node_count, Real{0.0});
        result.physical_boundary_mass_transfer.assign(
            node_count, Real{0.0});
        result.discrete_divergence_mass_source.assign(
            node_count, Real{0.0});
        std::vector<Real> row_divergence(node_count, Real{0.0});
        std::vector<Real> direct_strong_rate(node_count, Real{0.0});
        std::vector<Real> decomposed_strong_rate(node_count, Real{0.0});
        result.flux_edges.reserve(graph.edges.size());
        result.minimum_low_order_coefficient =
            std::numeric_limits<Real>::infinity();
        result.low_order_coefficients_nonnegative = true;

        for (std::size_t node = 0; node < node_count; ++node) {
            const auto& velocity = nodal_velocity[node];
            const Real diagonal_velocity = dot(
                graph.diagonal_gradient[node], velocity, graph.dimension);
            row_divergence[node] = diagonal_velocity;
            direct_strong_rate[node] =
                -diagonal_velocity * previous_liquid_indicator[node];
        }

        for (const auto& edge : graph.edges) {
            if (edge.first_node < 0 || edge.second_node < 0 ||
                edge.first_node >= edge.second_node ||
                static_cast<std::size_t>(edge.second_node) >= node_count) {
                result.diagnostic =
                    "P1 conservative phase stage found a malformed graph edge";
                return result;
            }
            const auto first = static_cast<std::size_t>(edge.first_node);
            const auto second = static_cast<std::size_t>(edge.second_node);
            const Real first_to_second_speed = dot(
                edge.first_test_second_gradient,
                nodal_velocity[second], graph.dimension);
            const Real second_to_first_speed = dot(
                edge.second_test_first_gradient,
                nodal_velocity[first], graph.dimension);
            const Real diffusion = std::max(
                std::abs(first_to_second_speed),
                std::abs(second_to_first_speed));
            const Real first_coefficient =
                diffusion - first_to_second_speed;
            const Real second_coefficient =
                diffusion - second_to_first_speed;
            result.minimum_low_order_coefficient = std::min(
                result.minimum_low_order_coefficient,
                std::min(first_coefficient, second_coefficient));
            const Real coefficient_tolerance = scaledTolerance(
                options.invariant_tolerance,
                {diffusion, first_to_second_speed,
                 second_to_first_speed});
            result.low_order_coefficients_nonnegative =
                result.low_order_coefficients_nonnegative &&
                first_coefficient >= -coefficient_tolerance &&
                second_coefficient >= -coefficient_tolerance;

            result.nodal_courant[first] +=
                time_step * std::max(Real{0.0}, first_coefficient) /
                graph.lumped_control_volume[first];
            result.nodal_courant[second] +=
                time_step * std::max(Real{0.0}, second_coefficient) /
                graph.lumped_control_volume[second];

            const Real first_indicator = previous_liquid_indicator[first];
            const Real second_indicator = previous_liquid_indicator[second];
            const Real central_rate =
                second_to_first_speed * first_indicator -
                first_to_second_speed * second_indicator;
            const Real diffusive_rate =
                diffusion * (second_indicator - first_indicator);
            result.flux_edges.push_back(LevelSetPhaseFluxEdge{
                .first_node = edge.first_node,
                .second_node = edge.second_node,
                .low_order_mass_transfer =
                    time_step * (central_rate + diffusive_rate),
                .raw_antidiffusive_mass_transfer =
                    -time_step * diffusive_rate,
            });
            decomposed_strong_rate[first] += central_rate;
            decomposed_strong_rate[second] -= central_rate;

            row_divergence[first] += first_to_second_speed;
            row_divergence[second] += second_to_first_speed;
            direct_strong_rate[first] -=
                first_to_second_speed * second_indicator;
            direct_strong_rate[second] -=
                second_to_first_speed * first_indicator;
        }

        if (graph.edges.empty()) {
            result.minimum_low_order_coefficient = Real{0.0};
        }
        if (!result.low_order_coefficients_nonnegative) {
            result.diagnostic =
                "P1 conservative phase stage produced a negative low-order graph coefficient";
            return result;
        }

        result.maximum_courant = Real{0.0};
        Real maximum_integrated_strong_rate{0.0};
        for (std::size_t node = 0; node < node_count; ++node) {
            if (!std::isfinite(result.nodal_courant[node])) {
                result.diagnostic =
                    "P1 conservative phase stage produced a non-finite nodal Courant number";
                return result;
            }
            result.maximum_courant = std::max(
                result.maximum_courant, result.nodal_courant[node]);
            const Real indicator = previous_liquid_indicator[node];
            const Real boundary_rate =
                -indicator * dot(graph.boundary_column_sum[node],
                                 nodal_velocity[node], graph.dimension);
            const Real divergence_rate = indicator * row_divergence[node];
            result.physical_boundary_mass_transfer[node] =
                time_step * boundary_rate;
            result.discrete_divergence_mass_source[node] =
                time_step * divergence_rate;
            decomposed_strong_rate[node] +=
                boundary_rate + divergence_rate;
            direct_strong_rate[node] += divergence_rate;

            const Real residual = time_step *
                (decomposed_strong_rate[node] -
                 direct_strong_rate[node]);
            maximum_integrated_strong_rate = std::max(
                maximum_integrated_strong_rate,
                time_step * std::max(
                    std::abs(decomposed_strong_rate[node]),
                    std::abs(direct_strong_rate[node])));
            result.maximum_strong_form_decomposition_residual = std::max(
                result.maximum_strong_form_decomposition_residual,
                std::abs(residual));
        }

        const Real decomposition_tolerance = scaledTolerance(
            options.invariant_tolerance,
            {maximum_integrated_strong_rate});
        result.strong_form_decomposition_satisfied =
            result.maximum_strong_form_decomposition_residual <=
            decomposition_tolerance;
        if (!result.strong_form_decomposition_satisfied) {
            result.diagnostic =
                "P1 conservative phase stage failed the strong-CG edge/boundary decomposition identity";
            return result;
        }

        result.courant_satisfied =
            !options.enforce_courant_limit ||
            result.maximum_courant <=
                options.maximum_courant +
                    scaledTolerance(options.invariant_tolerance,
                                    {result.maximum_courant,
                                     options.maximum_courant});
        if (!result.courant_satisfied) {
            result.diagnostic =
                "P1 conservative phase stage rejected a time step outside its low-order Courant contract";
            return result;
        }

        result.correction =
            applyLevelSetConservativePhaseFluxCorrection(
                LevelSetPhaseFluxStageView{
                    .lumped_control_volume = graph.lumped_control_volume,
                    .previous_liquid_indicator =
                        previous_liquid_indicator,
                    .lower_liquid_indicator = lower_liquid_indicator,
                    .upper_liquid_indicator = upper_liquid_indicator,
                    .interior_edges = result.flux_edges,
                    .physical_boundary_mass_transfer =
                        result.physical_boundary_mass_transfer,
                    .discrete_divergence_mass_source =
                        result.discrete_divergence_mass_source,
                    .invariant_tolerance = options.invariant_tolerance,
                    .component_activity_tolerance =
                        options.component_activity_tolerance,
                    .require_constant_preservation =
                        options.require_constant_preservation,
                });
        bool raw_target_constant_preservation_satisfied = true;
        if (result.correction.constant_state_input &&
            result.correction.nodes.size() == node_count) {
            for (const auto& node : result.correction.nodes) {
                const Real error = std::abs(
                    node.raw_target_liquid_indicator -
                    node.previous_liquid_indicator);
                result.correction.maximum_constant_preservation_error =
                    std::max(
                        result.correction
                            .maximum_constant_preservation_error,
                        error);
                const Real tolerance = scaledTolerance(
                    options.invariant_tolerance,
                    {node.previous_liquid_indicator,
                     node.raw_target_liquid_indicator});
                raw_target_constant_preservation_satisfied =
                    raw_target_constant_preservation_satisfied &&
                    error <= tolerance;
            }
            result.correction.constant_preservation_satisfied =
                result.correction.constant_preservation_satisfied &&
                raw_target_constant_preservation_satisfied;
        }
        if (result.correction.constant_preservation_required &&
            !raw_target_constant_preservation_satisfied) {
            result.correction.success = false;
            result.correction.diagnostic =
                "P1 conservative phase stage raw target failed constant-state preservation";
        }
        if (!result.correction.success) {
            result.diagnostic = result.correction.diagnostic;
            return result;
        }

        result.success = true;
        result.diagnostic = "ok";
    } catch (const std::exception& exception) {
        result.success = false;
        result.diagnostic = exception.what();
    }
    return result;
}

LevelSetP1PhaseSplitStageValidationResult
validateLevelSetP1PhaseSplitStage(
    const LevelSetP1PhaseTransportGraph& graph,
    std::span<const Real> previous_liquid_indicator,
    const LevelSetP1PhaseTransportStageResult& stage,
    const LevelSetP1PhaseSplitStageProvenance& provenance)
{
    LevelSetP1PhaseSplitStageValidationResult result;
    result.provenance = provenance;
    try {
        const auto collective = phaseStageCollectiveContext(graph);
        bool local_preparation_success = true;
        std::string local_preparation_diagnostic;
        std::vector<PhaseStageVerificationNode> verification_scratch;
        try {
            result.actual_operator_graph_identity =
                levelSetP1PhaseGraphIdentity(graph);
            result.computed_previous_q_revision =
                levelSetP1PhaseScalarContentRevision(
                    previous_liquid_indicator);
            result.computed_nodal_velocity_revision =
                levelSetP1PhaseVelocityContentRevision(
                    stage.sampled_nodal_velocity);
            result.computed_flux_ledger_digest =
                levelSetP1PhaseFluxLedgerDigest(stage);

            bool local_valid = true;
            std::string local_diagnostic;
            const auto reject = [&](std::string diagnostic) {
                if (local_valid) {
                    local_valid = false;
                    local_diagnostic = std::move(diagnostic);
                }
            };

        if (!graph.success) {
            reject("P1 conservative phase split-stage requires a valid transport graph");
        }
        if (!graph.partition_of_unity_satisfied ||
            !graph.gradient_partition_satisfied ||
            !graph.positive_control_volumes_satisfied ||
            !graph.gradient_row_sum_satisfied ||
            !graph.measure_closure_satisfied ||
            !graph.edge_ownership_satisfied) {
            reject("P1 conservative phase split-stage requires all transport-graph invariants");
        }
        if (graph.parallel_rank != collective.rank ||
            graph.parallel_size != collective.size ||
            (collective.active &&
             (!graph.distributed || !graph.replicated_sparse_graph)) ||
            (!collective.active && graph.distributed)) {
            reject("P1 conservative phase split-stage graph and field communicator metadata disagree");
        }
        if (provenance.scheme !=
            LevelSetP1PhaseSplitScheme::
                BackwardEulerExplicitIndicatorEndpointVelocity) {
            reject("P1 conservative phase split-stage supports only the Backward-Euler explicit-indicator endpoint-velocity scheme");
        }
        if (provenance.transport_mesh_policy !=
            LevelSetP1PhaseTransportMeshPolicy::FixedBackground) {
            reject("P1 conservative phase split-stage requires a fixed background transport mesh because no ALE mesh-flux/GCL term is implemented");
        }
        if (provenance.temporal_order != 1) {
            reject("P1 conservative phase split-stage requires temporal order one");
        }
        if (provenance.prospective_step == 0u ||
            provenance.attempt == 0u) {
            reject("P1 conservative phase split-stage requires positive prospective-step and attempt metadata");
        }
        const std::array<Real, 5> times{{
            provenance.step_start_time,
            provenance.step_end_time,
            provenance.q_input_time,
            provenance.velocity_state_time,
            provenance.time_step,
        }};
        if (!std::all_of(times.begin(), times.end(), [](Real value) {
                return std::isfinite(value);
            }) ||
            !(provenance.time_step > Real{0.0})) {
            reject("P1 conservative phase split-stage requires finite times and a positive finite time step");
        } else {
            const Real expected_end_time =
                provenance.step_start_time + provenance.time_step;
            if (!std::isfinite(expected_end_time) ||
                !exactPhaseStageRealEqual(
                    provenance.q_input_time,
                    provenance.step_start_time) ||
                !exactPhaseStageRealEqual(
                    provenance.velocity_state_time,
                    provenance.step_end_time) ||
                !exactPhaseStageRealEqual(
                    expected_end_time,
                    provenance.step_end_time)) {
                reject("P1 conservative phase split-stage requires exact q^n/start and endpoint-velocity/end time semantics");
            }
        }
        if (!exactPhaseStageRealEqual(
                stage.time_step, provenance.time_step)) {
            reject("P1 conservative phase split-stage time step does not match the executed transport stage");
        }
        if (!samePhaseStageOptionsExact(
                stage.executed_options, provenance.stage_options)) {
            reject("P1 conservative phase split-stage options do not exactly match the executed transport stage");
        }
        if (provenance.operator_state_revision == 0u ||
            provenance.previous_q_revision == 0u ||
            provenance.nodal_velocity_revision == 0u ||
            provenance.final_flux_ledger_digest == 0u) {
            reject("P1 conservative phase split-stage requires nonzero operator, q, velocity, and flux-ledger revisions");
        }
        const auto identity_complete = [](const auto& identity) {
            return identity.dimension >= 2 && identity.dimension <= 3 &&
                   identity.nodes != 0u && identity.edges != 0u &&
                   identity.dof_layout_revision != 0u &&
                   identity.content_revision != 0u;
        };
        if (!identity_complete(provenance.previous_graph_identity) ||
            !identity_complete(provenance.operator_graph_identity) ||
            !identity_complete(result.actual_operator_graph_identity)) {
            reject("P1 conservative phase split-stage requires complete graph and FE-layout identities");
        } else if (!samePhaseGraphIdentity(
                       provenance.previous_graph_identity,
                       provenance.operator_graph_identity)) {
            reject("P1 conservative phase split-stage rejected graph geometry/topology/ownership/numbering/layout drift from q^n to the operator endpoint");
        } else if (!samePhaseGraphIdentity(
                       provenance.operator_graph_identity,
                       result.actual_operator_graph_identity)) {
            reject("P1 conservative phase split-stage operator graph identity does not match the executed transport graph");
        }
        if (provenance.previous_q_revision !=
            result.computed_previous_q_revision) {
            reject("P1 conservative phase split-stage previous-q content revision mismatch");
        }
        if (provenance.nodal_velocity_revision !=
            result.computed_nodal_velocity_revision) {
            reject("P1 conservative phase split-stage sampled endpoint-velocity content revision mismatch");
        }
            std::string ledger_diagnostic;
            if (!completePhaseFluxLedger(
                    graph, previous_liquid_indicator, stage,
                    ledger_diagnostic)) {
                reject(std::move(ledger_diagnostic));
            }
            if (provenance.final_flux_ledger_digest !=
                result.computed_flux_ledger_digest) {
                reject("P1 conservative phase split-stage complete flux-ledger digest mismatch");
            }
            if (local_valid) {
                verification_scratch.resize(graph.nodes);
                std::string verification_diagnostic;
                if (!verifyPhaseStageLedgerEquations(
                        graph, previous_liquid_indicator, stage,
                        verification_scratch,
                        verification_diagnostic)) {
                    reject(std::move(verification_diagnostic));
                }
            }

            local_preparation_success = local_valid;
            local_preparation_diagnostic = std::move(local_diagnostic);
        } catch (const std::exception& exception) {
            local_preparation_success = false;
            local_preparation_diagnostic = exception.what();
        }

        if (!synchronizeLocalFailure(
                collective,
                local_preparation_success,
                local_preparation_diagnostic,
                result.diagnostic,
                "split-stage provenance validation")) {
            return result;
        }
        if (!replicatedSplitStageProvenanceMatches(
                collective, provenance, result.diagnostic)) {
            return result;
        }
        result.valid = true;
        result.diagnostic = "ok";
    } catch (const std::exception& exception) {
        result.valid = false;
        result.diagnostic = exception.what();
    }
    return result;
}

} // namespace svmp::FE::level_set

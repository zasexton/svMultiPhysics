/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_MESH_BOUNDARY_RECORD_SUPPORT_H
#define SVMP_FE_SYSTEMS_MESH_BOUNDARY_RECORD_SUPPORT_H

/**
 * @file MeshBoundaryRecordSupport.h
 * @brief Collective consensus of accepted-step records: canonical token
 *        sequences, their all-rank agreement, operator-stage metadata and
 *        geometry stamps, and coordinated local failures.
 *
 * The FE system's mesh-boundary records use these helpers, and so do the
 * physics record stores attached to an FE system (FESystem::attachment).
 * Collective functions must be called on every rank of the system's active
 * communicator in the same order.
 */

#include "Core/Types.h"
#include "Systems/FESystem.h"

#include <bit>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <vector>

#if FE_HAS_MPI
#include <mpi.h>
#endif

namespace svmp {
namespace FE {
namespace systems {
namespace record_support {

using MeshBoundaryConsensusTokens = std::vector<std::uint64_t>;

[[nodiscard]] inline std::uint64_t
generatedBoundaryTraceRealBits(Real value) noexcept
{
    static_assert(sizeof(Real) == sizeof(std::uint64_t));
    return std::bit_cast<std::uint64_t>(value);
}

inline void appendMeshBoundaryToken(
    MeshBoundaryConsensusTokens& tokens,
    std::uint64_t value)
{
    tokens.push_back(value);
}

inline void appendMeshBoundaryString(
    MeshBoundaryConsensusTokens& tokens,
    std::string_view value)
{
    appendMeshBoundaryToken(
        tokens, static_cast<std::uint64_t>(value.size()));
    for (const unsigned char byte : value) {
        appendMeshBoundaryToken(
            tokens, static_cast<std::uint64_t>(byte));
    }
}

template <typename T>
void appendMeshBoundaryIntegral(
    MeshBoundaryConsensusTokens& tokens,
    T value)
{
    if constexpr (std::is_enum_v<T>) {
        using Underlying = std::underlying_type_t<T>;
        appendMeshBoundaryToken(
            tokens,
            static_cast<std::uint64_t>(
                static_cast<std::make_unsigned_t<Underlying>>(value)));
    } else {
        using Unsigned = std::make_unsigned_t<T>;
        appendMeshBoundaryToken(
            tokens,
            static_cast<std::uint64_t>(static_cast<Unsigned>(value)));
    }
}

struct MeshBoundaryCollectiveContext {
    int size{1};
#if FE_HAS_MPI
    MPI_Comm communicator{MPI_COMM_NULL};
#endif
};

void appendOperatorStageMeasurementMetadata(
    MeshBoundaryConsensusTokens& tokens,
    const OperatorStageMeasurementMetadata& metadata);

[[nodiscard]] MeshBoundaryCollectiveContext
meshBoundaryCollectiveContext(const FESystem& system) noexcept;

[[nodiscard]] OperatorStageGeometryMetadata captureOperatorStageGeometry(
    const assembly::IMeshAccess& mesh);

[[nodiscard]] bool operatorStageTimesMatch(Real lhs, Real rhs) noexcept;

[[nodiscard]] bool coordinateMeshBoundaryLocalFailure(
    const MeshBoundaryCollectiveContext& context,
    bool local_failed,
    const char* phase);

void requireMeshBoundaryTokenConsensus(
    const MeshBoundaryCollectiveContext& context,
    const MeshBoundaryConsensusTokens& local_tokens,
    MeshBoundaryConsensusTokens& minimum_tokens,
    MeshBoundaryConsensusTokens& maximum_tokens,
    const char* phase);

} // namespace record_support
} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_MESH_BOUNDARY_RECORD_SUPPORT_H

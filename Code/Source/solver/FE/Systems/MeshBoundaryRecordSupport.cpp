/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/MeshBoundaryRecordSupport.h"

#include "Core/FEException.h"
#include "Core/MpiCollectiveTrace.h"
#include "Systems/SystemsExceptions.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace svmp {
namespace FE {
namespace systems {
namespace record_support {

void appendOperatorStageMeasurementMetadata(
    MeshBoundaryConsensusTokens& tokens,
    const OperatorStageMeasurementMetadata& metadata)
{
    appendMeshBoundaryString(tokens, metadata.scheme_name);
    appendMeshBoundaryIntegral(tokens, metadata.temporal_order);
    appendMeshBoundaryToken(
        tokens, metadata.prospective_accepted_step);
    appendMeshBoundaryToken(tokens, metadata.prospective_attempt);
    appendMeshBoundaryToken(
        tokens,
        generatedBoundaryTraceRealBits(metadata.step_start_time));
    appendMeshBoundaryToken(
        tokens,
        generatedBoundaryTraceRealBits(metadata.step_end_time));
    appendMeshBoundaryToken(
        tokens,
        generatedBoundaryTraceRealBits(metadata.state_time));
    appendMeshBoundaryToken(
        tokens,
        generatedBoundaryTraceRealBits(metadata.rate_time));
    appendMeshBoundaryToken(
        tokens, generatedBoundaryTraceRealBits(metadata.dt));
    appendMeshBoundaryToken(
        tokens, metadata.generalized_alpha.has_value() ? 1u : 0u);
    if (metadata.generalized_alpha.has_value()) {
        appendMeshBoundaryToken(
            tokens,
            generatedBoundaryTraceRealBits(
                metadata.generalized_alpha->alpha_f));
        appendMeshBoundaryToken(
            tokens,
            generatedBoundaryTraceRealBits(
                metadata.generalized_alpha->alpha_m));
    }
    // expected_stage_geometry is a rank-local observation-vector stamp.  It
    // is validated against the live adapter but excluded from MPI consensus.
    appendMeshBoundaryToken(tokens, metadata.state_revision);
    appendMeshBoundaryToken(tokens, metadata.rate_revision);
    appendMeshBoundaryToken(
        tokens,
        static_cast<std::uint64_t>(
            metadata.derivative_fields.size()));
    for (const auto field : metadata.derivative_fields) {
        appendMeshBoundaryIntegral(tokens, field);
    }
}

[[nodiscard]] MeshBoundaryCollectiveContext
meshBoundaryCollectiveContext(const FESystem& system) noexcept
{
    MeshBoundaryCollectiveContext context;
#if FE_HAS_MPI
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    if (initialized != 0) {
        MPI_Finalized(&finalized);
    }
    if (initialized != 0 && finalized == 0) {
        context.communicator = system.activeMpiCommunicator();
        if (context.communicator != MPI_COMM_NULL) {
            MPI_Comm_size(context.communicator, &context.size);
        }
    }
#else
    static_cast<void>(system);
#endif
    return context;
}

[[nodiscard]] OperatorStageGeometryMetadata captureOperatorStageGeometry(
    const assembly::IMeshAccess& mesh)
{
    return OperatorStageGeometryMetadata{
        .geometry_revision = mesh.geometryRevision(),
        .topology_revision = mesh.topologyRevision(),
        .ownership_revision = mesh.ownershipRevision(),
        .numbering_revision = mesh.numberingRevision(),
        .field_layout_revision = mesh.fieldLayoutRevision(),
        .label_revision = mesh.labelRevision(),
        .active_configuration_epoch = mesh.activeConfigurationEpoch(),
        .coordinate_configuration_key =
            mesh.coordinateConfigurationKey(),
    };
}

[[nodiscard]] bool operatorStageTimesMatch(Real lhs, Real rhs) noexcept
{
    if (!std::isfinite(lhs) || !std::isfinite(rhs)) {
        return false;
    }
    const Real scale =
        std::max({Real{1.0}, std::abs(lhs), std::abs(rhs)});
    return std::abs(lhs - rhs) <=
           Real{64.0} * std::numeric_limits<Real>::epsilon() * scale;
}

[[nodiscard]] bool coordinateMeshBoundaryLocalFailure(
    const MeshBoundaryCollectiveContext& context,
    bool local_failed,
    const char* phase)
{
    bool any_failed = local_failed;
#if FE_HAS_MPI
    if (context.size > 1) {
        const int local_value = local_failed ? 1 : 0;
        int global_value = local_value;
        const auto sequence = debug::nextMpiCollectiveTraceSeq();
        debug::traceMpiCollective(
            "before",
            sequence,
            phase,
            1,
            MPI_INT,
            MPI_MAX,
            context.communicator);
        MPI_Allreduce(
            &local_value,
            &global_value,
            1,
            MPI_INT,
            MPI_MAX,
            context.communicator);
        debug::traceMpiCollective(
            "after",
            sequence,
            phase,
            1,
            MPI_INT,
            MPI_MAX,
            context.communicator);
        any_failed = global_value != 0;
    }
#else
    static_cast<void>(context);
    static_cast<void>(phase);
#endif
    return any_failed;
}

void requireMeshBoundaryTokenConsensus(
    const MeshBoundaryCollectiveContext& context,
    const MeshBoundaryConsensusTokens& local_tokens,
    MeshBoundaryConsensusTokens& minimum_tokens,
    MeshBoundaryConsensusTokens& maximum_tokens,
    const char* phase)
{
#if FE_HAS_MPI
    if (context.size <= 1) {
        return;
    }
    const std::uint64_t local_count =
        static_cast<std::uint64_t>(local_tokens.size());
    std::uint64_t minimum_count = local_count;
    std::uint64_t maximum_count = local_count;
    auto sequence = debug::nextMpiCollectiveTraceSeq();
    debug::traceMpiCollective(
        "before",
        sequence,
        phase,
        1,
        MPI_UINT64_T,
        MPI_MIN,
        context.communicator);
    MPI_Allreduce(
        &local_count,
        &minimum_count,
        1,
        MPI_UINT64_T,
        MPI_MIN,
        context.communicator);
    debug::traceMpiCollective(
        "after",
        sequence,
        phase,
        1,
        MPI_UINT64_T,
        MPI_MIN,
        context.communicator);
    sequence = debug::nextMpiCollectiveTraceSeq();
    debug::traceMpiCollective(
        "before",
        sequence,
        phase,
        1,
        MPI_UINT64_T,
        MPI_MAX,
        context.communicator);
    MPI_Allreduce(
        &local_count,
        &maximum_count,
        1,
        MPI_UINT64_T,
        MPI_MAX,
        context.communicator);
    debug::traceMpiCollective(
        "after",
        sequence,
        phase,
        1,
        MPI_UINT64_T,
        MPI_MAX,
        context.communicator);
    FE_THROW_IF(
        minimum_count != maximum_count,
        InvalidArgumentException,
        "FESystem: fitted-ALE normal measurement consensus shape differs "
        "across active communicator ranks");

    const int token_count = static_cast<int>(local_tokens.size());
    if (token_count == 0) {
        return;
    }
    sequence = debug::nextMpiCollectiveTraceSeq();
    debug::traceMpiCollective(
        "before",
        sequence,
        phase,
        token_count,
        MPI_UINT64_T,
        MPI_MIN,
        context.communicator);
    MPI_Allreduce(
        local_tokens.data(),
        minimum_tokens.data(),
        token_count,
        MPI_UINT64_T,
        MPI_MIN,
        context.communicator);
    debug::traceMpiCollective(
        "after",
        sequence,
        phase,
        token_count,
        MPI_UINT64_T,
        MPI_MIN,
        context.communicator);
    sequence = debug::nextMpiCollectiveTraceSeq();
    debug::traceMpiCollective(
        "before",
        sequence,
        phase,
        token_count,
        MPI_UINT64_T,
        MPI_MAX,
        context.communicator);
    MPI_Allreduce(
        local_tokens.data(),
        maximum_tokens.data(),
        token_count,
        MPI_UINT64_T,
        MPI_MAX,
        context.communicator);
    debug::traceMpiCollective(
        "after",
        sequence,
        phase,
        token_count,
        MPI_UINT64_T,
        MPI_MAX,
        context.communicator);
    FE_THROW_IF(
        minimum_tokens != maximum_tokens,
        InvalidArgumentException,
        "FESystem: fitted-ALE normal declarations, stage metadata, or raw "
        "moments differ across active communicator ranks");
#else
    static_cast<void>(context);
    static_cast<void>(local_tokens);
    static_cast<void>(minimum_tokens);
    static_cast<void>(maximum_tokens);
    static_cast<void>(phase);
#endif
}

} // namespace record_support
} // namespace systems
} // namespace FE
} // namespace svmp

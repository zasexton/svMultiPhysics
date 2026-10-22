/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/MeshBoundaryRecordSupport.h"

#include "Core/FEException.h"
#include "Core/MpiCollectiveTrace.h"
#include "Forms/FormExpr.h"
#include "Systems/SystemsExceptions.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace svmp {
namespace FE {
namespace systems {
namespace record_support {

namespace {

template <typename T, typename Append>
void appendMeshBoundaryOptional(
    MeshBoundaryConsensusTokens& tokens,
    const std::optional<T>& value,
    Append&& append)
{
    appendMeshBoundaryToken(tokens, value.has_value() ? 1u : 0u);
    if (value.has_value()) {
        append(tokens, *value);
    }
}

void appendMeshBoundarySpaceSignature(
    MeshBoundaryConsensusTokens& tokens,
    const forms::SpaceSignature* signature)
{
    appendMeshBoundaryToken(tokens, signature == nullptr ? 0u : 1u);
    if (signature == nullptr) {
        return;
    }
    appendMeshBoundaryIntegral(tokens, signature->space_type);
    appendMeshBoundaryIntegral(tokens, signature->field_type);
    appendMeshBoundaryIntegral(tokens, signature->continuity);
    appendMeshBoundaryIntegral(tokens, signature->value_dimension);
    appendMeshBoundaryIntegral(
        tokens, signature->topological_dimension);
    appendMeshBoundaryIntegral(tokens, signature->polynomial_order);
    appendMeshBoundaryIntegral(tokens, signature->element_type);
}

void appendMeshBoundaryExpression(
    MeshBoundaryConsensusTokens& tokens,
    const forms::FormExprNode* node,
    bool& contains_opaque_callable)
{
    appendMeshBoundaryToken(tokens, node == nullptr ? 0u : 1u);
    if (node == nullptr) {
        return;
    }
    appendMeshBoundaryIntegral(tokens, node->type());
    appendMeshBoundaryOptional(
        tokens,
        node->constantValue(),
        [](auto& out, Real value) {
            appendMeshBoundaryToken(
                out, generatedBoundaryTraceRealBits(value));
        });
    const auto append_int = [](auto& out, int value) {
        appendMeshBoundaryIntegral(out, value);
    };
    appendMeshBoundaryOptional(tokens, node->identityDim(), append_int);
    appendMeshBoundaryOptional(tokens, node->boundaryMarker(), append_int);
    appendMeshBoundaryOptional(tokens, node->interfaceMarker(), append_int);
    appendMeshBoundaryOptional(
        tokens,
        node->cutVolumeSide(),
        [](auto& out, forms::CutVolumeSide value) {
            appendMeshBoundaryIntegral(out, value);
        });
    appendMeshBoundaryOptional(tokens, node->componentIndex0(), append_int);
    appendMeshBoundaryOptional(tokens, node->componentIndex1(), append_int);
    appendMeshBoundaryOptional(tokens, node->tensorRows(), append_int);
    appendMeshBoundaryOptional(tokens, node->tensorCols(), append_int);
    appendMeshBoundaryOptional(tokens, node->indexRank(), append_int);
    appendMeshBoundaryOptional(
        tokens,
        node->indexIds(),
        [](auto& out, const auto& values) {
            for (const auto value : values) {
                appendMeshBoundaryIntegral(out, value);
            }
        });
    appendMeshBoundaryOptional(
        tokens,
        node->indexExtents(),
        [](auto& out, const auto& values) {
            for (const auto value : values) {
                appendMeshBoundaryIntegral(out, value);
            }
        });
    appendMeshBoundaryOptional(
        tokens,
        node->indexNames(),
        [](auto& out, const auto& values) {
            for (const auto value : values) {
                appendMeshBoundaryString(out, value);
            }
        });
    appendMeshBoundaryOptional(
        tokens,
        node->indexVariances(),
        [](auto& out, const auto& values) {
            for (const auto value : values) {
                appendMeshBoundaryIntegral(out, value);
            }
        });
    appendMeshBoundaryOptional(
        tokens, node->constitutiveOutputIndex(), append_int);
    appendMeshBoundarySpaceSignature(tokens, node->spaceSignature());
    appendMeshBoundaryOptional(
        tokens, node->timeDerivativeOrder(), append_int);
    appendMeshBoundaryOptional(
        tokens,
        node->fieldId(),
        [](auto& out, FieldId value) {
            appendMeshBoundaryIntegral(out, value);
        });
    appendMeshBoundaryOptional(
        tokens,
        node->symbolName(),
        [](auto& out, std::string_view value) {
            appendMeshBoundaryString(out, value);
        });
    appendMeshBoundaryOptional(
        tokens,
        node->slotIndex(),
        [](auto& out, std::uint32_t value) {
            appendMeshBoundaryIntegral(out, value);
        });
    appendMeshBoundaryOptional(tokens, node->historyIndex(), append_int);
    appendMeshBoundaryOptional(
        tokens,
        node->stateOffsetBytes(),
        [](auto& out, std::uint32_t value) {
            appendMeshBoundaryIntegral(out, value);
        });
    appendMeshBoundaryOptional(tokens, node->eigenIndex(), append_int);
    appendMeshBoundaryOptional(
        tokens,
        node->fromConfiguration(),
        [](auto& out, forms::GeometryConfiguration value) {
            appendMeshBoundaryIntegral(out, value);
        });
    appendMeshBoundaryOptional(
        tokens,
        node->toConfiguration(),
        [](auto& out, forms::GeometryConfiguration value) {
            appendMeshBoundaryIntegral(out, value);
        });

    const std::array<bool, 7> opaque_payloads{
        node->scalarCoefficient() != nullptr,
        node->timeScalarCoefficient() != nullptr,
        node->vectorCoefficient() != nullptr,
        node->matrixCoefficient() != nullptr,
        node->tensor3Coefficient() != nullptr,
        node->tensor4Coefficient() != nullptr,
        node->constitutiveModel() != nullptr,
    };
    for (const bool present : opaque_payloads) {
        appendMeshBoundaryToken(tokens, present ? 1u : 0u);
        contains_opaque_callable = contains_opaque_callable || present;
    }
    const auto* measure = node->exteriorBoundaryMeasure();
    appendMeshBoundaryToken(tokens, measure == nullptr ? 0u : 1u);
    if (measure != nullptr) {
        appendMeshBoundaryIntegral(tokens, measure->scope());
        appendMeshBoundaryIntegral(
            tokens, measure->physicalBoundaryMarker());
        appendMeshBoundaryIntegral(
            tokens, measure->generatedActiveBoundaryMarker());
    }
    const auto children = node->childrenShared();
    appendMeshBoundaryToken(
        tokens, static_cast<std::uint64_t>(children.size()));
    for (const auto& child : children) {
        appendMeshBoundaryExpression(
            tokens, child.get(), contains_opaque_callable);
    }
}

} // namespace

void appendMeshNormalDeclaration(
    MeshBoundaryConsensusTokens& tokens,
    const MeshNormalBoundaryConstraintDeclaration& declaration,
    bool& contains_opaque_callable)
{
    appendMeshBoundaryIntegral(
        tokens, declaration.mesh_displacement_field);
    appendMeshBoundaryIntegral(tokens, declaration.boundary_marker);
    appendMeshBoundaryIntegral(tokens, declaration.quantity);
    appendMeshBoundaryIntegral(tokens, declaration.target_kind);
    appendMeshBoundaryExpression(
        tokens,
        declaration.target_expression.node(),
        contains_opaque_callable);
    appendMeshBoundaryIntegral(tokens, declaration.enforcement_kind);
    appendMeshBoundaryIntegral(
        tokens, declaration.related_velocity_field);
    appendMeshBoundaryString(tokens, declaration.owner_component);
    appendMeshBoundaryToken(
        tokens, declaration.consumer_binding.has_value() ? 1u : 0u);
    if (!declaration.consumer_binding.has_value()) {
        return;
    }
    const auto& binding = *declaration.consumer_binding;
    appendMeshBoundaryString(tokens, binding.operator_tag);
    appendMeshBoundaryString(tokens, binding.mesh_descriptor_source);
    appendMeshBoundaryToken(
        tokens, binding.related_fluid.has_value() ? 1u : 0u);
    if (binding.related_fluid.has_value()) {
        appendMeshBoundaryIntegral(
            tokens, binding.related_fluid->enforcement_kind);
        appendMeshBoundaryString(
            tokens, binding.related_fluid->descriptor_source);
    }
}

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

/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_OPERATOR_STAGE_METADATA_H
#define SVMP_FE_SYSTEMS_OPERATOR_STAGE_METADATA_H

/**
 * @file OperatorStageMetadata.h
 * @brief Metadata of an accepted-step measurement taken at a converged
 *        operator stage: scheme, stage times, generalized-alpha parameters
 *        and the mesh geometry stamp the stage was evaluated on.
 */

#include "Core/Types.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

/**
 * @brief Generalized-alpha coordinates for a neutral operator-stage record.
 *
 * These coordinates describe where the converged operator sampled state and
 * rate.  They do not assert that either sample is an accepted-step endpoint.
 */
struct OperatorStageGeneralizedAlphaMetadata {
    Real alpha_f{1.0};
    Real alpha_m{1.0};

    [[nodiscard]] friend bool operator==(
        const OperatorStageGeneralizedAlphaMetadata&,
        const OperatorStageGeneralizedAlphaMetadata&) = default;
};

/** Rank-local mesh stamp captured with an operator-stage state snapshot. */
struct OperatorStageGeometryMetadata {
    std::uint64_t geometry_revision{0};
    std::uint64_t topology_revision{0};
    std::uint64_t ownership_revision{0};
    std::uint64_t numbering_revision{0};
    std::uint64_t field_layout_revision{0};
    std::uint64_t label_revision{0};
    std::uint64_t active_configuration_epoch{0};
    std::uint64_t coordinate_configuration_key{0};

    [[nodiscard]] friend bool operator==(
        const OperatorStageGeometryMetadata&,
        const OperatorStageGeometryMetadata&) = default;
};

/**
 * @brief Time-integrator-neutral provenance for a prospective accepted stage.
 *
 * `state_time` and `rate_time` are the exact converged operator sampling
 * times.  For generalized-alpha they are generally not the step endpoint.
 * `derivative_fields` states which entries of the supplied exact-rate vector
 * are meaningful time derivatives.
 */
struct OperatorStageMeasurementMetadata {
    /** V1 identity: BackwardEuler, DG0, or GeneralizedAlphaFirstOrder. */
    std::string scheme_name{};
    int temporal_order{0};
    std::uint64_t prospective_accepted_step{0};
    std::uint64_t prospective_attempt{0};
    Real step_start_time{0.0};
    Real step_end_time{0.0};
    Real state_time{0.0};
    Real rate_time{0.0};
    Real dt{0.0};
    std::optional<OperatorStageGeneralizedAlphaMetadata>
        generalized_alpha{};
    /** Expected rank-local stamp; deliberately excluded from MPI consensus. */
    std::optional<OperatorStageGeometryMetadata> expected_stage_geometry{};
    /**
     * Communicator-certified exact algebraic fingerprint/revision of the
     * copied operator-stage state.  The core compares the supplied token
     * across ranks but cannot independently prove the caller's certification.
     */
    std::uint64_t state_revision{0};
    /**
     * Communicator-certified exact algebraic fingerprint/revision of the
     * copied rate alias, subject to the same caller-certification boundary.
     */
    std::uint64_t rate_revision{0};
    std::vector<FieldId> derivative_fields{};

    [[nodiscard]] friend bool operator==(
        const OperatorStageMeasurementMetadata&,
        const OperatorStageMeasurementMetadata&) = default;
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_OPERATOR_STAGE_METADATA_H

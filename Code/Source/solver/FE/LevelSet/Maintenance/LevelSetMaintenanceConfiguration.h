/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#pragma once

/**
 * @file LevelSetMaintenanceConfiguration.h
 * @ingroup fe_level_set
 * @brief Configuration, run state and declaration bindings of one maintained
 *        level-set field.
 *
 * A maintained level-set field is transported by the level-set equation and
 * repaired between steps: kinematic reconciliation, sign-definite patch
 * bounds and triggered interface-preserving redistancing after an accepted
 * step, the bound-preserving limiter and conservative phase transport inside
 * a step, and curvature projection.  These records hold what those
 * operations read and carry, independent of the physics that owns the field.
 * An application derives from them to add its own data (input provenance,
 * physics-specific initialization).
 */

#include "Core/Types.h"
#include "LevelSet/LevelSetCurvatureProjection.h"
#include "LevelSet/LevelSetGeometry.h"
#include "LevelSet/LevelSetOptions.h"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace level_set {

/// A level-set inflow or outflow boundary, named by its mesh face.
struct LevelSetMaintenanceOpenBoundary {
    std::string face_name{};
    bool inflow{false};
    /// Inflow value given as a literal, when one was given.
    std::optional<Real> literal_inflow_value{};
};

/**
 * @brief Immutable configuration of one maintained level-set field.
 *
 * Fixed when the run is set up; shared by every copy of the request.
 */
struct LevelSetMaintenanceConfiguration {
    using OpenBoundary = LevelSetMaintenanceOpenBoundary;

    LevelSetTransportOptions transport{};
    /// Isovalue of the interface (and of the cut-cell volume).
    double isovalue{0.0};
    std::vector<OpenBoundary> open_boundaries{};
    /// The active cut-volume request of the same field and isovalue, if any.
    std::optional<ActiveCutVolumeRequest> volume_cut_request{};
    bool curvature_projection_enabled{false};
    std::string curvature_field_name{};
    int curvature_projection_cadence_steps{1};
    LevelSetCurvatureProjectionOptions curvature_projection{};
};

/**
 * @brief Run state of one maintained level-set field that the accepted-step
 *        maintenance carries from step to step.
 */
struct LevelSetMaintenanceRuntime {
    /// Interface-preserving redistancing trigger: the interface displacement
    /// bound accumulated since the last redistancing, and the number of
    /// redistancing calls.
    Real redistancing_accumulated_displacement{0.0};
    std::size_t redistancing_calls{0};
};

/**
 * @brief Data of the free-surface declarations bound to the field after
 *        setup; when set, they override the configured curvature options.
 */
struct LevelSetMaintenanceDeclarationBindings {
    std::optional<bool> negative_liquid_side{};
    std::optional<std::vector<LevelSetKinematicAreaGradientYoungWall>>
        young_walls{};
};

} // namespace level_set
} // namespace FE
} // namespace svmp

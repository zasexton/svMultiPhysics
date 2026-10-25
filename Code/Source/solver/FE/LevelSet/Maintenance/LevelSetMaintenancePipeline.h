/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#pragma once

/**
 * @file LevelSetMaintenancePipeline.h
 * @ingroup fe_level_set
 * @brief The maintenance of level-set fields between steps: the
 *        accepted-step pipeline (kinematic reconciliation, sign-definite
 *        patch bounds, triggered interface-preserving redistancing and the
 *        publication of the change into the accepted history).
 *
 * The functions were moved from Application/Core/ApplicationDriver.cpp with
 * bodies, messages and collective order unchanged.  Every function is
 * collective over the FE system's active communicator, and a rank-local
 * failure is agreed on before any rank acts on it.
 */

#include "Core/LogRecord.h"
#include "Core/Types.h"
#include "LevelSet/LevelSetMaintenanceSubstage.h"
#include "LevelSet/LevelSetOptions.h"
#include "LevelSet/LevelSetVolume.h"
#include "LevelSet/Maintenance/LevelSetMaintenanceConfiguration.h"

#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {
class FESystem;
} // namespace systems
namespace timestepping {
class TimeHistory;
} // namespace timestepping

namespace level_set {

/// A maintained level-set field: its immutable configuration and its run
/// state.
struct LevelSetMaintenanceRequest {
    std::shared_ptr<const LevelSetMaintenanceConfiguration> configuration{};
    LevelSetMaintenanceRuntime runtime{};
};

/// Services of the caller that the pipeline uses.
struct LevelSetMaintenanceHost {
    /// The FE system; the functions do nothing without one.
    systems::FESystem* fe_system{nullptr};
};

/// Validates the final candidate of an accepted-step maintenance transaction
/// before it is published (throws to reject it).
using LevelSetMaintenanceCandidateValidator =
    std::function<void(std::span<const Real>)>;

/// Observer of each maintenance substage that changed the level set, with
/// the FE-ordered solution before and after it.
using LevelSetMaintenanceStageObserver = std::function<void(
    MaintenanceSubstage, std::span<const Real>, std::span<const Real>)>;

/**
 * @brief Accepted state from which the next transport step starts.
 *
 * The kinematic reconciliation of that step compares the transported
 * endpoint with it, and the sign-definite patch bounds take their one-ring
 * ranges from it.  The transport velocity is captured only for the
 * reconciliation.
 */
struct LevelSetKinematicBaseline {
    FieldId level_set_field{INVALID_FIELD_ID};
    FieldId velocity_field{INVALID_FIELD_ID};
    int step{-1};
    std::vector<Real> level_set{};
    std::vector<Real> velocity{};
};

/**
 * @brief When the triggered interface-preserving redistancing of a field
 *        runs: once the interface displacement bound accumulated since the
 *        last redistancing (max |w| dt over the vertices of cut cells)
 *        reaches interface_motion_cells times the smallest cut-cell edge.
 */
struct LevelSetRedistancingTrigger {
    double interface_motion_cells{1.0};
};

/**
 * @brief The redistancing trigger: one cell for every case.  The diagnostic
 *        environment override SVMP_REDISTANCING_TRIGGER_CELLS (positive)
 *        changes it to test the sensitivity; an invalid override throws.
 */
[[nodiscard]] LevelSetRedistancingTrigger resolveLevelSetRedistancingTrigger();

/// Volume options of the field's active cut-volume request.
[[nodiscard]] LevelSetVolumeOptions levelSetVolumeOptionsForMaintenance(
    const LevelSetMaintenanceConfiguration& configuration);

/// Transport-velocity coefficients in the velocity field's DOF numbering,
/// from the unknown vector or from prescribed-data storage.
[[nodiscard]] std::vector<Real> levelSetTransportVelocityCoefficients(
    const systems::FESystem& system,
    FieldId velocity_field,
    std::span<const Real> fe_solution);

/// The kinematic baseline of @p configuration's field at @p fe_solution.
[[nodiscard]] LevelSetKinematicBaseline makeLevelSetKinematicBaseline(
    const systems::FESystem& system,
    const LevelSetMaintenanceConfiguration& configuration,
    std::span<const Real> fe_solution,
    int step);

/**
 * @brief Accepted-step maintenance of every non-conservative request, as
 *        one transaction.
 *
 * The stages, in this order: for each request when due, the kinematic
 * reconciliation against @p kinematic_baselines, the sign-definite patch
 * bounds, and the interface-preserving redistancing when its trigger
 * (resolveLevelSetRedistancingTrigger) fires; then the changed candidate is
 * passed to @p validate_candidate, and the change of every maintained field
 * is published as one common delta on every stored state of @p history.  The requests' run state is updated only when everything
 * succeeds.  Returns whether a field changed.  The commit records
 * (kinematic reconciliation, sign-definite patch bounds, interface-preserving
 * redistancing, synchronized; rank 0 prints them) go to
 * @p deferred_commit_logs when given, else to the log.
 */
bool applyLevelSetMaintenance(
    const LevelSetMaintenanceHost& host,
    timestepping::TimeHistory& history,
    std::vector<LevelSetMaintenanceRequest>& requests,
    const LevelSetMaintenanceCandidateValidator& validate_candidate = {},
    const LevelSetMaintenanceStageObserver& observe_stage = {},
    std::vector<LogRecord>* deferred_commit_logs = nullptr,
    bool* rate_publication_started = nullptr,
    std::span<const LevelSetKinematicBaseline> kinematic_baselines = {});

} // namespace level_set
} // namespace FE
} // namespace svmp

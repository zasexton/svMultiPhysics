/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_TIMESTEPPING_MESHNORMALSTAGEPARTICIPANT_H
#define SVMP_FE_TIMESTEPPING_MESHNORMALSTAGEPARTICIPANT_H

/**
 * @file MeshNormalStageParticipant.h
 * @brief Step participant of the mesh-normal constraint operator-stage
 *        measurements (Systems/MeshNormalConstraintStageMeasurements.h).
 *
 * Before the transient loop it checks, collectively, that the ranks agree on
 * the number of measurements and on scheme support, and that a supported
 * temporal-order-one scheme runs when measurements are declared.  At the
 * converged operator stage of a candidate it stages the measurements, a
 * discarded candidate drops them, and an accepted step publishes them.
 */

#include "Core/Types.h"
#include "Systems/SystemCollectives.h"
#include "TimeStepping/StepParticipants.h"
#include "TimeStepping/TimeHistory.h"
#include "TimeStepping/TimeLoop.h"

#include <cstdint>
#include <functional>
#include <string_view>

namespace svmp {
namespace FE {

namespace backends {
class GenericVector;
} // namespace backends

namespace systems {
class FESystem;
} // namespace systems

namespace timestepping {

/// Throws when operator-stage measurements are declared
/// (@p declaration_count > 0) and the scheme is not supported.
void requireMeshNormalStageSchemeCoverage(
    std::uint64_t declaration_count,
    bool scheme_supported,
    std::string_view scheme_name,
    int temporal_order);

/// What the participant reads of the run.
struct MeshNormalStageContext {
    /// The FE system whose measurements the participant stages.
    systems::FESystem* system{nullptr};
    /// The scheme of the time loop.
    SchemeKind scheme{SchemeKind::BackwardEuler};
    /// The name of the selected scheme, for the coverage error.
    std::string_view scheme_name{};
    /// The active FE communicator of the run.
    systems::CollectiveCommunicator comm{};
    /// Communicator-wide fingerprint of the content of a distributed FE
    /// vector (collective over comm): the state and rate revisions of the
    /// stage metadata.
    std::function<std::uint64_t(const backends::GenericVector&)> state_fingerprint{};
};

class MeshNormalStageParticipant : public IStepParticipant {
public:
    /// Checks the declarations and scheme support on the active FE
    /// communicator (collective); throws when they differ across ranks or
    /// the scheme is unsupported.
    explicit MeshNormalStageParticipant(const MeshNormalStageContext& context);

    /// CandidateStage when there are measurements to evaluate.
    [[nodiscard]] StepHook hooks() const noexcept override;

    /// Stages the operator-stage measurement of a converged candidate.
    void onCandidateStage(const CandidateStageObservation& observation) override;

    /// Drops the staged measurement of a discarded candidate.
    void discardCandidate() noexcept;

    /// Publishes the staged measurement of the accepted step @p h.
    void commitAccepted(const TimeHistory& h);

    bool evaluate_fitted_ale_operator_stage{false};
    GlobalIndex fitted_ale_operator_stage_state_size{0};

private:
    MeshNormalStageContext context_;
};

} // namespace timestepping
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_TIMESTEPPING_MESHNORMALSTAGEPARTICIPANT_H

/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_FUNCTIONAL_RECORDS_H
#define SVMP_FE_SYSTEMS_FUNCTIONAL_RECORDS_H

/**
 * @file FunctionalRecords.h
 * @brief The accepted functional records of an FE system: the declared
 *        discrete functionals, the residual-work channels and the
 *        exterior-boundary energy declaration, and the accepted functional
 *        history.
 *
 * The records are an FE-system attachment (FESystem::attachment), so they
 * live as long as the system.  The declared functionals and their accepted
 * history are an AcceptedStageRecordStore; a formulation declares its
 * functionals, work channels and boundary-energy proof before setup, and
 * the accepted-functional recorder (FE/LevelSet/AcceptedFunctionalRecorder)
 * records one declaration-aligned group after each accepted step.  The
 * record data types are in FreeSurfaceRecordTypes.h.
 */

#include "Assembly/Assembler.h"
#include "Core/Types.h"
#include "Systems/AcceptedStageRecords.h"
#include "Systems/FESystem.h"
#include "Systems/FreeSurfaceRecordTypes.h"

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

/// The declared functionals and their accepted history.
using FunctionalHistoryStore =
    AcceptedStageRecordStore<FreeSurfaceDiscreteFunctionalDeclaration,
                             FreeSurfaceDiscreteFunctionalHistoryRecord>;

/// The functional records of one FE system (its attachment).
struct FunctionalRecordStore {
    FunctionalHistoryStore functionals{};
    std::vector<FreeSurfaceResidualWorkDeclaration> work_channels{};
    std::optional<FreeSurfaceExternalBoundaryEnergyDeclaration>
        external_boundary_energy{};
};

/**
 * @brief The functional records of an FE system.
 *
 * A lightweight handle on the system and its record store.  Declarations
 * are made before setup; accepted functional groups are recorded after each
 * accepted step.
 */
class FunctionalRecords {
public:
    explicit FunctionalRecords(FESystem& system);

    /// Declare one discrete functional (kept sorted by interface marker).
    void declareFunctional(FreeSurfaceDiscreteFunctionalDeclaration declaration);
    /// Declare the owner of one residual-work channel.
    void declareWorkChannel(FreeSurfaceResidualWorkDeclaration declaration);
    /// Declare the exterior-boundary energy proof (at most once).
    void declareExternalBoundaryEnergy(
        FreeSurfaceExternalBoundaryEnergyDeclaration declaration);

    [[nodiscard]] std::span<const FreeSurfaceDiscreteFunctionalDeclaration>
    functionalDeclarations() const noexcept
    {
        return functional_declarations_;
    }
    [[nodiscard]] std::span<const FreeSurfaceResidualWorkDeclaration>
    workChannelDeclarations() const noexcept
    {
        return work_channel_declarations_;
    }
    [[nodiscard]] const std::optional<
        FreeSurfaceExternalBoundaryEnergyDeclaration>&
    externalBoundaryEnergyDeclaration() const noexcept
    {
        return external_boundary_energy_declaration_;
    }

    /// Record one accepted, declaration-aligned functional group.
    void recordAcceptedFunctionals(
        std::uint64_t accepted_step,
        Real accepted_time,
        Real dt,
        std::uint64_t pre_maintenance_endpoint_state_revision,
        std::uint64_t state_revision,
        std::span<const AcceptedFreeSurfaceDiscreteFunctionalState> states,
        std::optional<std::uint64_t> extension_map_revision = std::nullopt);
    [[nodiscard]] std::span<const FreeSurfaceDiscreteFunctionalHistoryRecord>
    acceptedFunctionalHistory() const noexcept
    {
        return functional_history_;
    }

private:
    // The FE system state that the record code reads, under the names it had
    // as FESystem members.
    struct FieldRegistryAccess {
        const FESystem* system;
        [[nodiscard]] const FieldRecord& get(FieldId field) const
        {
            return system->fieldRecord(field);
        }
        [[nodiscard]] bool has(FieldId field) const noexcept
        {
            return system->hasField(field);
        }
    };
    struct MeshAccessPointer {
        const FESystem* system;
        explicit operator bool() const noexcept
        {
            return system->hasMeshAccess();
        }
        const assembly::IMeshAccess* operator->() const
        {
            return &system->meshAccess();
        }
    };

    [[nodiscard]] bool isSetup() const noexcept { return system_->isSetup(); }
    [[nodiscard]] bool hasOperator(const OperatorTag& op) const noexcept
    {
        return system_->hasOperator(op);
    }
    [[nodiscard]] bool fieldParticipatesInUnknownVector(FieldId field) const
    {
        return system_->fieldParticipatesInUnknownVector(field);
    }
    void invalidateSetup() noexcept { system_->invalidateSetup(); }

    FESystem* system_;
    AcceptedStageRecords<FunctionalHistoryStore> functionals_;
    std::vector<FreeSurfaceDiscreteFunctionalDeclaration>&
        functional_declarations_;
    std::vector<FreeSurfaceResidualWorkDeclaration>& work_channel_declarations_;
    std::optional<FreeSurfaceExternalBoundaryEnergyDeclaration>&
        external_boundary_energy_declaration_;
    std::vector<FreeSurfaceDiscreteFunctionalHistoryRecord>& functional_history_;
    FieldRegistryAccess field_registry_;
    MeshAccessPointer mesh_access_;
};

/// Read-only functional records of an FE system; empty without records.
class ConstFunctionalRecords {
public:
    explicit ConstFunctionalRecords(const FESystem& system) noexcept;

    [[nodiscard]] std::span<const FreeSurfaceDiscreteFunctionalDeclaration>
    functionalDeclarations() const noexcept;
    [[nodiscard]] std::span<const FreeSurfaceResidualWorkDeclaration>
    workChannelDeclarations() const noexcept;
    [[nodiscard]] const std::optional<
        FreeSurfaceExternalBoundaryEnergyDeclaration>&
    externalBoundaryEnergyDeclaration() const noexcept;
    [[nodiscard]] std::span<const FreeSurfaceDiscreteFunctionalHistoryRecord>
    acceptedFunctionalHistory() const noexcept;

private:
    const FunctionalRecordStore* store_{nullptr};
};

/// The functional records of @p system (created on first use).
[[nodiscard]] FunctionalRecords functionalRecords(FESystem& system);

/// The functional records of @p system, read-only.
[[nodiscard]] ConstFunctionalRecords functionalRecords(
    const FESystem& system) noexcept;

/// Accepted-stage contact-centroid kinematics from two consecutive history
/// records (see FreeSurfaceAcceptedContactLineKinematics).
[[nodiscard]] std::vector<FreeSurfaceAcceptedContactLineKinematics>
deriveAcceptedContactLineKinematics(
    const FreeSurfaceDiscreteFunctionalHistoryRecord& previous,
    std::uint64_t accepted_step,
    const FreeSurfaceAcceptedContactStageState& current);

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_FUNCTIONAL_RECORDS_H

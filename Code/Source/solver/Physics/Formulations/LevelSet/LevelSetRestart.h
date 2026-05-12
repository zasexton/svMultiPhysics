#ifndef SVMP_PHYSICS_FORMULATIONS_LEVELSET_RESTART_H
#define SVMP_PHYSICS_FORMULATIONS_LEVELSET_RESTART_H

/**
 * @file LevelSetRestart.h
 * @brief Restart records for level-set fields and generated interfaces.
 */

#include "Physics/Formulations/LevelSet/LevelSetInterfaceLifecycle.h"
#include "Physics/Formulations/LevelSet/LevelSetTransportModule.h"

#include "FE/Core/Types.h"
#include "FE/Systems/FESystem.h"
#include "Interfaces/LevelSetInterfaceDomain.h"

#include <cstdint>
#include <string>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace level_set {

struct LevelSetFieldRestartRecord {
    std::string field_name{};
    FE::FieldId field_id{FE::INVALID_FIELD_ID};
    LevelSetFieldSource source{LevelSetFieldSource::Unknown};
    bool auto_register_field{false};
    int components{0};
    FE::GlobalIndex dof_offset{0};
    FE::GlobalIndex dof_count{0};
    std::uint64_t value_revision{0};
};

struct LevelSetGeneratedInterfaceRestartRecord {
    std::string level_set_field_name{};
    FE::FieldId level_set_field_id{FE::INVALID_FIELD_ID};
    std::string domain_id{};
    int requested_interface_marker{-1};
    int interface_marker{-1};
    FE::Real isovalue{0.0};
    FE::Real tolerance{1.0e-12};
    int quadrature_order{1};
    bool keep_degenerate_fragments{false};
    std::uint64_t value_revision{0};
    std::uint64_t mesh_geometry_revision{0};
    std::uint64_t mesh_topology_revision{0};
    std::uint64_t ownership_revision{0};
    FE::interfaces::CutInterfaceDomainSummary summary{};
};

struct LevelSetRestartSnapshot {
    LevelSetFieldRestartRecord field{};
    std::vector<LevelSetGeneratedInterfaceRestartRecord> generated_interfaces{};
};

[[nodiscard]] LevelSetFieldRestartRecord captureLevelSetFieldRestartRecord(
    const FE::systems::FESystem& system,
    const LevelSetTransportOptions& options,
    std::uint64_t value_revision);

[[nodiscard]] LevelSetGeneratedInterfaceRestartRecord
captureLevelSetGeneratedInterfaceRestartRecord(
    const FE::systems::FESystem& system,
    const LevelSetGeneratedInterfaceOptions& options,
    const LevelSetGeneratedInterfaceResult& result);

[[nodiscard]] LevelSetGeneratedInterfaceOptions
optionsFromLevelSetGeneratedInterfaceRestartRecord(
    const LevelSetGeneratedInterfaceRestartRecord& record);

[[nodiscard]] bool levelSetGeneratedInterfaceRestartRecordMatches(
    const FE::systems::FESystem& system,
    const LevelSetGeneratedInterfaceRestartRecord& record,
    std::string* diagnostic = nullptr);

} // namespace level_set
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_LEVELSET_RESTART_H

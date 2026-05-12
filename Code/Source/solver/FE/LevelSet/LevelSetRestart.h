#pragma once

/**
 * @file
 * @ingroup fe_level_set
 * @brief Restart records for level-set fields and generated interfaces.
 */

#include "Core/Types.h"
#include "Interfaces/LevelSetInterfaceDomain.h"
#include "LevelSet/LevelSetInterfaceLifecycle.h"
#include "LevelSet/LevelSetOptions.h"
#include "Systems/FESystem.h"

#include <cstdint>
#include <string>
#include <vector>

namespace svmp::FE::level_set {

struct LevelSetFieldRestartRecord {
    std::string field_name{};
    FieldId field_id{INVALID_FIELD_ID};
    LevelSetFieldSource source{LevelSetFieldSource::Unknown};
    bool auto_register_field{false};
    int components{0};
    GlobalIndex dof_offset{0};
    GlobalIndex dof_count{0};
    std::uint64_t value_revision{0};
};

struct LevelSetGeneratedInterfaceRestartRecord {
    std::string level_set_field_name{};
    FieldId level_set_field_id{INVALID_FIELD_ID};
    std::string domain_id{};
    int requested_interface_marker{-1};
    int interface_marker{-1};
    Real isovalue{0.0};
    Real tolerance{1.0e-12};
    int quadrature_order{1};
    bool keep_degenerate_fragments{false};
    std::uint64_t value_revision{0};
    std::uint64_t mesh_geometry_revision{0};
    std::uint64_t mesh_topology_revision{0};
    std::uint64_t ownership_revision{0};
    interfaces::CutInterfaceDomainSummary summary{};
};

struct LevelSetRestartSnapshot {
    LevelSetFieldRestartRecord field{};
    std::vector<LevelSetGeneratedInterfaceRestartRecord> generated_interfaces{};
};

[[nodiscard]] LevelSetFieldRestartRecord captureLevelSetFieldRestartRecord(
    const systems::FESystem& system,
    const LevelSetTransportOptions& options,
    std::uint64_t value_revision);

[[nodiscard]] LevelSetGeneratedInterfaceRestartRecord
captureLevelSetGeneratedInterfaceRestartRecord(
    const systems::FESystem& system,
    const LevelSetGeneratedInterfaceOptions& options,
    const LevelSetGeneratedInterfaceResult& result);

[[nodiscard]] LevelSetGeneratedInterfaceOptions
optionsFromLevelSetGeneratedInterfaceRestartRecord(
    const LevelSetGeneratedInterfaceRestartRecord& record);

[[nodiscard]] bool levelSetGeneratedInterfaceRestartRecordMatches(
    const systems::FESystem& system,
    const LevelSetGeneratedInterfaceRestartRecord& record,
    std::string* diagnostic = nullptr);

} // namespace svmp::FE::level_set

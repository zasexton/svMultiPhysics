/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ASSEMBLY_CUTINTEGRATIONCONTEXT_H
#define SVMP_FE_ASSEMBLY_CUTINTEGRATIONCONTEXT_H

/**
 * @file CutIntegrationContext.h
 * @brief Physics-neutral assembly-facing cut integration metadata.
 */

#include "Core/Types.h"
#include "Geometry/CutQuadrature.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#include "Mesh/Search/CutCell.h"
#endif

#include <array>
#include <string>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace assembly {

struct CutCellAssemblyMetadata {
    MeshIndex cell = static_cast<MeshIndex>(-1);
    Real volume_fraction = 0.0;
    geometry::CutIntegrationSide side = geometry::CutIntegrationSide::Negative;
    std::array<Real, 3> embedded_normal{{0.0, 0.0, 0.0}};
    std::string provenance_id{};
    std::uint64_t revision_key = 0;
};

struct CutStabilizationHook {
    std::string name{};
    Real geometry_scale = 0.0;
    Real conditioning_indicator = 0.0;
    bool enabled = false;
};

struct EmbeddedBoundaryKinematicData {
    std::string constraint_id{};
    std::string relation_map_id{};
    std::string source_geometry_id{};
    std::string provenance_id{};
    std::uint64_t relation_revision_key = 0;
};

class CutIntegrationContext {
public:
    void clear() {
        metadata_.clear();
        volume_rules_.clear();
        interface_rules_.clear();
        kinematic_data_.clear();
        stabilization_hooks_.clear();
    }

    void addVolumeRule(CutCellAssemblyMetadata metadata,
                       geometry::CutQuadratureRule rule) {
        metadata_.push_back(std::move(metadata));
        volume_rules_.push_back(std::move(rule));
    }

    void addInterfaceRule(geometry::CutQuadratureRule rule) {
        interface_rules_.push_back(std::move(rule));
    }

    void addKinematicData(EmbeddedBoundaryKinematicData data) {
        kinematic_data_.push_back(std::move(data));
    }

    void addStabilizationHook(CutStabilizationHook hook) {
        stabilization_hooks_.push_back(std::move(hook));
    }

    [[nodiscard]] const std::vector<CutCellAssemblyMetadata>& metadata() const noexcept {
        return metadata_;
    }

    [[nodiscard]] const std::vector<geometry::CutQuadratureRule>& volumeRules() const noexcept {
        return volume_rules_;
    }

    [[nodiscard]] const std::vector<geometry::CutQuadratureRule>& interfaceRules() const noexcept {
        return interface_rules_;
    }

    [[nodiscard]] const std::vector<EmbeddedBoundaryKinematicData>& kinematicData() const noexcept {
        return kinematic_data_;
    }

    [[nodiscard]] const std::vector<CutStabilizationHook>& stabilizationHooks() const noexcept {
        return stabilization_hooks_;
    }

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    void importCutMapSummary(const svmp::search::CutClassificationMap& map) {
        for (const auto& record : map.cells) {
            if (record.classification != svmp::search::CutClassification::Cut) {
                continue;
            }
            CutCellAssemblyMetadata metadata;
            metadata.cell = static_cast<MeshIndex>(record.entity);
            metadata.volume_fraction = 0.0;
            metadata.provenance_id = record.provenance.persistent_id;
            metadata.revision_key = map.revision_key();
            if (!record.intersections.empty()) {
                metadata.embedded_normal = {{
                    record.intersections.front().normal[0],
                    record.intersections.front().normal[1],
                    record.intersections.front().normal[2]}};
            }
            metadata_.push_back(std::move(metadata));
        }

        for (const auto& constraint : map.kinematic_constraints) {
            EmbeddedBoundaryKinematicData data;
            data.constraint_id = constraint.id;
            data.relation_map_id = constraint.relation_map_id;
            data.source_geometry_id = constraint.source_geometry_id;
            data.provenance_id = constraint.provenance.persistent_id;
            data.relation_revision_key = constraint.source_revision.revision_key();
            kinematic_data_.push_back(std::move(data));
        }
    }
#endif

private:
    std::vector<CutCellAssemblyMetadata> metadata_{};
    std::vector<geometry::CutQuadratureRule> volume_rules_{};
    std::vector<geometry::CutQuadratureRule> interface_rules_{};
    std::vector<EmbeddedBoundaryKinematicData> kinematic_data_{};
    std::vector<CutStabilizationHook> stabilization_hooks_{};
};

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_CUTINTEGRATIONCONTEXT_H

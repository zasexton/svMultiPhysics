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

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace assembly {

struct CutCellAssemblyMetadata {
    MeshIndex cell = static_cast<MeshIndex>(-1);
    MeshIndex parent_entity = static_cast<MeshIndex>(-1);
    Real volume_fraction = 0.0;
    geometry::CutIntegrationSide side = geometry::CutIntegrationSide::Negative;
    std::array<Real, 3> embedded_normal{{0.0, 0.0, 0.0}};
    std::string provenance_id{};
    std::string cut_topology_id{};
    std::uint64_t revision_key = 0;
    std::uint64_t cut_topology_revision = 0;
    std::uint64_t quadrature_policy_key = 0;
};

struct CutStabilizationHook {
    std::string name{};
    Real geometry_scale = 0.0;
    Real conditioning_indicator = 0.0;
    bool enabled = false;
    std::vector<MeshIndex> adjacent_cells{};
    std::vector<MeshIndex> extension_patch{};
    std::uint64_t conditioning_revision = 0;
};

struct EmbeddedBoundaryKinematicData {
    std::string constraint_id{};
    std::string relation_map_id{};
    std::string source_geometry_id{};
    std::string provenance_id{};
    std::uint64_t relation_revision_key = 0;
};

enum class CutIntegrationAssemblyPath : std::uint8_t {
    Standard,
    MatrixFree,
    Interpreter,
    AD,
    SymbolicTangent,
    JIT
};

struct CutIntegrationBinding {
    MeshIndex parent_entity = static_cast<MeshIndex>(-1);
    geometry::CutQuadratureKind kind = geometry::CutQuadratureKind::Volume;
    geometry::CutIntegrationSide side = geometry::CutIntegrationSide::Negative;
    std::uint64_t cut_revision_key = 0;
    std::uint64_t cut_topology_revision = 0;
    std::uint64_t quadrature_policy_key = 0;
    std::vector<CutIntegrationAssemblyPath> visible_to_paths{};
};

class CutIntegrationContext {
public:
    void clear() {
        metadata_.clear();
        volume_rules_.clear();
        interface_rules_.clear();
        kinematic_data_.clear();
        stabilization_hooks_.clear();
        bindings_.clear();
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

    void addBinding(CutIntegrationBinding binding) {
        bindings_.push_back(std::move(binding));
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

    [[nodiscard]] const std::vector<CutIntegrationBinding>& bindings() const noexcept {
        return bindings_;
    }

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    void importCutMapSummary(const svmp::search::CutClassificationMap& map) {
        for (const auto& record : map.cells) {
            if (record.classification != svmp::search::CutClassification::Cut) {
                continue;
            }
            CutCellAssemblyMetadata metadata;
            metadata.cell = static_cast<MeshIndex>(record.entity);
            metadata.parent_entity = static_cast<MeshIndex>(record.entity);
            metadata.volume_fraction = 0.0;
            metadata.provenance_id = record.provenance.persistent_id;
            metadata.cut_topology_id = std::to_string(record.cut_topology_id);
            metadata.revision_key = map.revision_key();
            if (!record.intersections.empty()) {
                metadata.embedded_normal = {{
                    record.intersections.front().normal[0],
                    record.intersections.front().normal[1],
                    record.intersections.front().normal[2]}};
            }
            metadata_.push_back(std::move(metadata));

            CutIntegrationBinding binding;
            binding.parent_entity = static_cast<MeshIndex>(record.entity);
            binding.kind = geometry::CutQuadratureKind::Volume;
            binding.side = geometry::CutIntegrationSide::Negative;
            binding.cut_revision_key = map.revision_key();
            binding.visible_to_paths = {
                CutIntegrationAssemblyPath::Standard,
                CutIntegrationAssemblyPath::MatrixFree,
                CutIntegrationAssemblyPath::Interpreter,
                CutIntegrationAssemblyPath::AD,
                CutIntegrationAssemblyPath::SymbolicTangent,
                CutIntegrationAssemblyPath::JIT};
            bindings_.push_back(std::move(binding));
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

    void importCutTopology(const svmp::search::CutTopologyRecord& topology,
                           geometry::CutGeometryFrame frame = geometry::CutGeometryFrame::Reference,
                           std::uint64_t quadrature_policy_key = 0) {
        std::unordered_map<std::uint64_t, std::array<Real, 3>> interface_points;
        std::unordered_map<MeshIndex, std::array<Real, 3>> cell_normals;
        for (const auto& vertex : topology.vertices) {
            interface_points[vertex.stable_id] = {{
                static_cast<Real>(vertex.point[0]),
                static_cast<Real>(vertex.point[1]),
                static_cast<Real>(vertex.point[2])}};
        }
        std::unordered_map<std::uint64_t, const svmp::search::CutCurvedPatchRecord*> curved_patch_by_id;
        for (const auto& patch : topology.curved_patches) {
            if (patch.isoparametric_quadrature_available &&
                patch.quadrature_points.size() == patch.quadrature_normals.size() &&
                patch.quadrature_points.size() == patch.quadrature_weights.size()) {
                curved_patch_by_id[patch.stable_id] = &patch;
            }
        }

        for (const auto& polygon : topology.interface_polygons) {
            cell_normals[static_cast<MeshIndex>(polygon.parent_cell)] = {{
                static_cast<Real>(polygon.normal[0]),
                static_cast<Real>(polygon.normal[1]),
                static_cast<Real>(polygon.normal[2])}};
            std::vector<std::array<Real, 3>> ordered_points;
            ordered_points.reserve(polygon.ordered_vertices.size());
            for (const auto id : polygon.ordered_vertices) {
                const auto it = interface_points.find(id);
                if (it != interface_points.end()) {
                    ordered_points.push_back(it->second);
                }
            }
            geometry::CutQuadratureConstructionPolicy policy;
            policy.kind = geometry::CutQuadratureConstructionKind::TopologySubdivision;
            geometry::CutQuadratureProvenance provenance;
            provenance.embedded_geometry_id = polygon.provenance.persistent_id;
            provenance.cut_topology_id = std::to_string(polygon.stable_id);
            provenance.parent_entity = static_cast<MeshIndex>(polygon.parent_cell);
            provenance.cut_topology_revision = topology.topology_revision;
            provenance.predicate_policy_key = topology.predicate_policy_key;
            provenance.frame = frame;
            geometry::CutQuadratureRule rule;
            const auto patch_it = curved_patch_by_id.find(polygon.stable_id);
            if (patch_it != curved_patch_by_id.end()) {
                const auto& patch = *patch_it->second;
                std::vector<geometry::CutQuadraturePoint> curved_points;
                curved_points.reserve(patch.quadrature_points.size());
                for (std::size_t i = 0; i < patch.quadrature_points.size(); ++i) {
                    curved_points.push_back({
                        {{static_cast<Real>(patch.quadrature_points[i][0]),
                          static_cast<Real>(patch.quadrature_points[i][1]),
                          static_cast<Real>(patch.quadrature_points[i][2])}},
                        {{static_cast<Real>(patch.quadrature_normals[i][0]),
                          static_cast<Real>(patch.quadrature_normals[i][1]),
                          static_cast<Real>(patch.quadrature_normals[i][2])}},
                        static_cast<Real>(patch.quadrature_weights[i])});
                }
                policy.kind = geometry::CutQuadratureConstructionKind::CurvedTopologySubdivision;
                policy.polynomial_order = patch.geometry_order;
                policy.name = patch.construction_policy;
                rule = geometry::makeCurvedInterfaceQuadrature(
                    curved_points,
                    frame,
                    policy,
                    provenance);
                rule.policy.kind = geometry::CutQuadratureConstructionKind::CurvedTopologySubdivision;
                rule.provenance.construction =
                    geometry::CutQuadratureConstructionKind::CurvedTopologySubdivision;
            } else {
                auto normal = std::array<Real, 3>{{
                    static_cast<Real>(polygon.normal[0]),
                    static_cast<Real>(polygon.normal[1]),
                    static_cast<Real>(polygon.normal[2])}};
                rule = geometry::makePolygonInterfaceQuadrature(
                    ordered_points,
                    normal,
                    policy,
                    provenance);
                rule.policy.kind = geometry::CutQuadratureConstructionKind::TopologySubdivision;
                rule.provenance.construction = geometry::CutQuadratureConstructionKind::TopologySubdivision;
            }
            addInterfaceRule(std::move(rule));
        }

        for (const auto& region : topology.side_regions) {
            geometry::CutQuadratureRule rule;
            rule.kind = geometry::CutQuadratureKind::Volume;
            rule.side = region.side == svmp::search::CutTopologySide::Negative
                            ? geometry::CutIntegrationSide::Negative
                            : geometry::CutIntegrationSide::Positive;
            rule.parent_measure = static_cast<Real>(region.parent_measure);
            rule.measure = static_cast<Real>(region.measure_estimate);
            rule.volume_fraction = static_cast<Real>(region.volume_fraction_estimate);
            rule.exact_for_constants = true;
            rule.curved_geometry = region.curved_isoparametric_topology;
            rule.policy.kind = region.curved_isoparametric_topology
                                   ? geometry::CutQuadratureConstructionKind::CurvedTopologySubdivision
                                   : geometry::CutQuadratureConstructionKind::TopologySubdivision;
            if (region.curved_isoparametric_topology) {
                rule.policy.name = "curved-isoparametric-topology-subdivision";
                const auto policy_it = std::find_if(region.integration_subcells.begin(),
                                                    region.integration_subcells.end(),
                                                    [](const auto& subcell) {
                                                        return subcell.curved_isoparametric &&
                                                               !subcell.construction_policy.empty();
                                                    });
                if (policy_it != region.integration_subcells.end()) {
                    rule.policy.name = policy_it->construction_policy;
                }
            }
            rule.provenance.embedded_geometry_id = region.provenance.persistent_id;
            rule.provenance.cut_topology_id = std::to_string(region.stable_id);
            rule.provenance.parent_entity = static_cast<MeshIndex>(region.parent_cell);
            rule.provenance.cut_topology_revision = topology.topology_revision;
            rule.provenance.predicate_policy_key = topology.predicate_policy_key;
            rule.provenance.construction = rule.policy.kind;
            rule.provenance.frame = frame;
            rule.provenance_id = region.provenance.persistent_id;
            rule.frame = frame;
            const auto normal_it = cell_normals.find(static_cast<MeshIndex>(region.parent_cell));
            const auto embedded_normal = normal_it != cell_normals.end()
                                             ? normal_it->second
                                             : std::array<Real, 3>{{1.0, 0.0, 0.0}};

            for (const auto& subcell : region.integration_subcells) {
                if (subcell.measure <= 0.0) {
                    continue;
                }
                geometry::CutQuadraturePoint qp;
                qp.weight = static_cast<Real>(subcell.measure);
                qp.point = {{static_cast<Real>(subcell.centroid[0]),
                             static_cast<Real>(subcell.centroid[1]),
                             static_cast<Real>(subcell.centroid[2])}};
                qp.normal = rule.side == geometry::CutIntegrationSide::Negative
                                ? embedded_normal
                                : std::array<Real, 3>{{-embedded_normal[0], -embedded_normal[1], -embedded_normal[2]}};
                rule.curved_geometry = rule.curved_geometry ||
                                       subcell.curved_isoparametric ||
                                       subcell.measure_from_isoparametric_quadrature;
                rule.points.push_back(qp);
            }
            if (rule.points.empty() && rule.measure > 0.0) {
                geometry::CutQuadraturePoint qp;
                qp.weight = rule.measure;
                qp.point = {{static_cast<Real>(region.centroid_estimate[0]),
                             static_cast<Real>(region.centroid_estimate[1]),
                             static_cast<Real>(region.centroid_estimate[2])}};
                qp.normal = rule.side == geometry::CutIntegrationSide::Negative
                                ? embedded_normal
                                : std::array<Real, 3>{{-embedded_normal[0], -embedded_normal[1], -embedded_normal[2]}};
                rule.points.push_back(qp);
            }

            CutCellAssemblyMetadata metadata;
            metadata.cell = static_cast<MeshIndex>(region.parent_cell);
            metadata.parent_entity = static_cast<MeshIndex>(region.parent_cell);
            metadata.volume_fraction = static_cast<Real>(region.volume_fraction_estimate);
            metadata.side = rule.side;
            metadata.embedded_normal = embedded_normal;
            metadata.provenance_id = region.provenance.persistent_id;
            metadata.cut_topology_id = std::to_string(region.stable_id);
            metadata.revision_key = topology.topology_revision;
            metadata.cut_topology_revision = topology.topology_revision;
            metadata.quadrature_policy_key = quadrature_policy_key;
            addVolumeRule(metadata, rule);

            CutIntegrationBinding binding;
            binding.parent_entity = static_cast<MeshIndex>(region.parent_cell);
            binding.kind = geometry::CutQuadratureKind::Volume;
            binding.side = rule.side;
            binding.cut_revision_key = topology.topology_revision;
            binding.cut_topology_revision = topology.topology_revision;
            binding.quadrature_policy_key = quadrature_policy_key;
            binding.visible_to_paths = {
                CutIntegrationAssemblyPath::Standard,
                CutIntegrationAssemblyPath::MatrixFree,
                CutIntegrationAssemblyPath::Interpreter,
                CutIntegrationAssemblyPath::AD,
                CutIntegrationAssemblyPath::SymbolicTangent,
                CutIntegrationAssemblyPath::JIT};
            addBinding(std::move(binding));
        }
    }
#endif

private:
    std::vector<CutCellAssemblyMetadata> metadata_{};
    std::vector<geometry::CutQuadratureRule> volume_rules_{};
    std::vector<geometry::CutQuadratureRule> interface_rules_{};
    std::vector<EmbeddedBoundaryKinematicData> kinematic_data_{};
    std::vector<CutStabilizationHook> stabilization_hooks_{};
    std::vector<CutIntegrationBinding> bindings_{};
};

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_CUTINTEGRATIONCONTEXT_H

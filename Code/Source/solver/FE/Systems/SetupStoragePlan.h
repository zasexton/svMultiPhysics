#ifndef SVMP_FE_SYSTEMS_SETUP_STORAGE_PLAN_H
#define SVMP_FE_SYSTEMS_SETUP_STORAGE_PLAN_H

/**
 * @file SetupStoragePlan.h
 * @brief Physics-agnostic storage requirements inferred before FE setup.
 *
 * The setup planner records the mesh topology, global-ID lookup maps, and
 * monolithic DOF storage that the FE infrastructure needs for a problem.  The
 * plan is intentionally expressed in FE concepts (spaces, operators,
 * constraints, reductions), not physics-module names.
 */

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

struct SetupStorageRequirements {
    bool vertex_topology{true};
    bool cell_topology{true};
    bool edge_topology{false};
    bool boundary_face_topology{false};
    bool interior_face_topology{false};
    bool interface_face_topology{false};

    bool vertex_gids{false};
    bool cell_gids{false};
    bool face_gids{false};
    bool edge_gids{false};

    bool global_vertex_lookup{false};
    bool global_cell_lookup{false};
    bool global_face_lookup{false};
    bool global_edge_lookup{false};

    bool entity_dof_map{false};

    [[nodiscard]] bool full_face_topology() const noexcept
    {
        return interior_face_topology;
    }

    [[nodiscard]] bool any_face_topology() const noexcept
    {
        return boundary_face_topology || interior_face_topology || interface_face_topology;
    }

    void merge(const SetupStorageRequirements& other) noexcept
    {
        vertex_topology = vertex_topology || other.vertex_topology;
        cell_topology = cell_topology || other.cell_topology;
        edge_topology = edge_topology || other.edge_topology;
        boundary_face_topology = boundary_face_topology || other.boundary_face_topology;
        interior_face_topology = interior_face_topology || other.interior_face_topology;
        interface_face_topology = interface_face_topology || other.interface_face_topology;

        vertex_gids = vertex_gids || other.vertex_gids;
        cell_gids = cell_gids || other.cell_gids;
        face_gids = face_gids || other.face_gids;
        edge_gids = edge_gids || other.edge_gids;

        global_vertex_lookup = global_vertex_lookup || other.global_vertex_lookup;
        global_cell_lookup = global_cell_lookup || other.global_cell_lookup;
        global_face_lookup = global_face_lookup || other.global_face_lookup;
        global_edge_lookup = global_edge_lookup || other.global_edge_lookup;

        entity_dof_map = entity_dof_map || other.entity_dof_map;
    }
};

struct SetupStoragePlan {
    SetupStorageRequirements requirements{};

    bool can_alias_single_field_dof_map{false};
    bool uses_single_field_alias{false};

    std::vector<std::string> reasons{};

    void merge(const SetupStorageRequirements& req, std::string reason = {})
    {
        requirements.merge(req);
        if (!reason.empty()) {
            reasons.push_back(std::move(reason));
        }
    }

    [[nodiscard]] std::string summary() const
    {
        auto on_off = [](bool value) { return value ? "on" : "off"; };

        std::string out = "topology{vertices=";
        out += on_off(requirements.vertex_topology);
        out += " cells=";
        out += on_off(requirements.cell_topology);
        out += " edges=";
        out += on_off(requirements.edge_topology);
        out += " boundary_faces=";
        out += on_off(requirements.boundary_face_topology);
        out += " interior_faces=";
        out += on_off(requirements.interior_face_topology);
        out += " interface_faces=";
        out += on_off(requirements.interface_face_topology);
        out += "} lookups{vertex=";
        out += on_off(requirements.global_vertex_lookup);
        out += " cell=";
        out += on_off(requirements.global_cell_lookup);
        out += " face=";
        out += on_off(requirements.global_face_lookup);
        out += " edge=";
        out += on_off(requirements.global_edge_lookup);
        out += "} gids{vertex=";
        out += on_off(requirements.vertex_gids);
        out += " cell=";
        out += on_off(requirements.cell_gids);
        out += " face=";
        out += on_off(requirements.face_gids);
        out += " edge=";
        out += on_off(requirements.edge_gids);
        out += "} entity_dof_map=";
        out += on_off(requirements.entity_dof_map);
        out += " monolithic_alias=";
        out += on_off(uses_single_field_alias);
        if (!reasons.empty()) {
            out += " reasons=";
            for (std::size_t i = 0; i < reasons.size(); ++i) {
                if (i > 0u) {
                    out += ",";
                }
                out += reasons[i];
            }
        }
        return out;
    }
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_SETUP_STORAGE_PLAN_H

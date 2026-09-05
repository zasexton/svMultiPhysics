/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_INTERFACES_GENERATEDACTIVEBOUNDARYDOMAIN_H
#define SVMP_FE_INTERFACES_GENERATEDACTIVEBOUNDARYDOMAIN_H

/**
 * @file GeneratedActiveBoundaryDomain.h
 * @brief Sharp exterior-boundary traces of a generated level-set phase.
 */

#include "Assembly/Assembler.h"
#include "Geometry/CutQuadrature.h"
#include "Interfaces/GeneratedInterfaceBoundaryIntersectionDomain.h"
#include "Interfaces/LevelSetInterfaceDomain.h"

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace svmp::FE::interfaces {

struct GeneratedActiveBoundaryMarkerKey {
    LevelSetInterfaceSource source{};
    std::string domain_id{};
    Real isovalue{0.0};
    int interface_marker{-1};
    int boundary_marker{-1};
    geometry::CutIntegrationSide side{geometry::CutIntegrationSide::Negative};
    int requested_marker{-1};

    [[nodiscard]] std::string stableKey() const;
};

[[nodiscard]] std::uint64_t stableGeneratedActiveBoundaryMarkerHash(
    const GeneratedActiveBoundaryMarkerKey& key);

[[nodiscard]] int stableGeneratedActiveBoundaryMarker(
    const GeneratedActiveBoundaryMarkerKey& key,
    int marker_base = 3000000,
    int marker_range = 1000000);

struct GeneratedActiveBoundaryRequest {
    LevelSetInterfaceSource source{};
    std::string generated_domain_id{};
    Real isovalue{0.0};
    int interface_marker{-1};
    int boundary_marker{-1};
    int active_boundary_marker{-1};
    geometry::CutIntegrationSide side{geometry::CutIntegrationSide::Negative};
    Real tolerance{1.0e-12};
    int quadrature_order{1};
    geometry::CutGeometryFrame frame{geometry::CutGeometryFrame::Reference};
    std::uint64_t mesh_geometry_revision{0};
    std::uint64_t mesh_topology_revision{0};
    std::uint64_t ownership_revision{0};
    std::uint64_t quadrature_policy_key{0};
    std::uint64_t source_value_revision{0};

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] int resolvedActiveBoundaryMarker() const;
};

struct GeneratedActiveBoundaryScalarField {
    std::function<Real(GlobalIndex)> value_at_node{};

    [[nodiscard]] bool valid() const noexcept {
        return static_cast<bool>(value_at_node);
    }
};

struct GeneratedActiveBoundaryFragment {
    LinearCornerStrictBranch construction_observation{LinearCornerStrictBranch::Unchecked};
    int interface_marker{-1};
    int boundary_marker{-1};
    int active_boundary_marker{-1};
    MeshIndex parent_cell{static_cast<MeshIndex>(-1)};
    MeshIndex parent_face{static_cast<MeshIndex>(-1)};
    GlobalIndex parent_cell_global_id{INVALID_GLOBAL_INDEX};
    GlobalIndex parent_face_global_id{INVALID_GLOBAL_INDEX};
    int owner_rank{-1};
    LocalIndex local_fragment_index{INVALID_LOCAL_INDEX};
    std::uint64_t stable_id{0};
    geometry::CutIntegrationSide side{geometry::CutIntegrationSide::Negative};
    std::vector<std::uint64_t> source_contact_stable_ids{};
    std::vector<std::uint64_t> source_interface_stable_ids{};
    // Scalar representation used to classify and clip this boundary face.
    // This is explicit because the boundary-rule construction policy is not
    // itself a level-set representation.
    std::string represented_implicit_geometry_mode{};
    std::string represented_implicit_quadrature_backend{};
    std::string represented_implicit_fallback_status{};
    std::array<Real, 3> boundary_normal{{1.0, 0.0, 0.0}};
    Real measure{0.0};
    Real parent_measure{0.0};
    bool full_face_equivalent{false};
    int achieved_quadrature_order{0};
    std::string topology_id{};
    std::vector<std::array<Real, 3>> vertices{};
    std::vector<geometry::CutQuadraturePoint> quadrature_points{};

    [[nodiscard]] bool active() const noexcept;
    [[nodiscard]] geometry::CutQuadratureRule toCutQuadratureRule(
        const GeneratedActiveBoundaryRequest& request) const;
};

struct GeneratedActiveBoundarySummary {
    std::size_t fragment_count{0};
    std::size_t quadrature_point_count{0};
    std::size_t full_face_count{0};
    std::size_t cut_face_count{0};
    Real measure{0.0};
    Real parent_measure{0.0};
};

class GeneratedActiveBoundaryDomain {
public:
    GeneratedActiveBoundaryDomain() = default;
    explicit GeneratedActiveBoundaryDomain(GeneratedActiveBoundaryRequest request);

    [[nodiscard]] const GeneratedActiveBoundaryRequest& request() const noexcept;
    [[nodiscard]] int marker() const noexcept;
    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] const std::vector<GeneratedActiveBoundaryFragment>&
    fragments() const noexcept;

    void addFragment(GeneratedActiveBoundaryFragment fragment);
    [[nodiscard]] GeneratedActiveBoundarySummary summary() const noexcept;
    [[nodiscard]] std::vector<geometry::CutQuadratureRule>
    boundaryQuadratureRules() const;

private:
    GeneratedActiveBoundaryRequest request_{};
    std::vector<GeneratedActiveBoundaryFragment> fragments_{};
};

struct GeneratedActiveBoundaryPartitionSummary {
    std::size_t boundary_face_count{0};
    std::size_t cut_boundary_face_count{0};
    std::size_t source_contact_fragment_count{0};
    std::size_t referenced_contact_fragment_count{0};
    std::size_t orphan_source_reference_count{0};
    std::size_t stale_revision_count{0};
    Real total_boundary_measure{0.0};
    Real negative_boundary_measure{0.0};
    Real positive_boundary_measure{0.0};
    Real max_partition_error{0.0};
};

[[nodiscard]] GeneratedActiveBoundaryDomain buildGeneratedActiveBoundaryDomain(
    GeneratedActiveBoundaryRequest request,
    const LevelSetInterfaceDomain& interface_domain,
    const GeneratedInterfaceBoundaryIntersectionDomain& contact_domain,
    const assembly::IMeshAccess& mesh,
    const GeneratedActiveBoundaryScalarField& scalar_field);

[[nodiscard]] GeneratedActiveBoundaryPartitionSummary
validateGeneratedActiveBoundaryPartition(
    const GeneratedActiveBoundaryDomain& negative_domain,
    const GeneratedActiveBoundaryDomain& positive_domain,
    const LevelSetInterfaceDomain& interface_domain,
    const GeneratedInterfaceBoundaryIntersectionDomain& contact_domain,
    const assembly::IMeshAccess& mesh);

} // namespace svmp::FE::interfaces

#endif // SVMP_FE_INTERFACES_GENERATEDACTIVEBOUNDARYDOMAIN_H

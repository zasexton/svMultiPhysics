/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_INTERFACES_GENERATEDINTERFACEBOUNDARYINTERSECTIONDOMAIN_H
#define SVMP_FE_INTERFACES_GENERATEDINTERFACEBOUNDARYINTERSECTIONDOMAIN_H

/**
 * @file GeneratedInterfaceBoundaryIntersectionDomain.h
 * @brief Physics-neutral generated-interface/mesh-boundary intersection data.
 */

#include "Core/Types.h"
#include "Geometry/CutQuadrature.h"
#include "Interfaces/LevelSetInterfaceDomain.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace assembly {
class IMeshAccess;
}
namespace interfaces {

enum class GeneratedInterfaceBoundaryIntersectionKind : std::uint8_t {
    Point,
    Segment
};

enum class GeneratedInterfaceBoundaryIntersectionDegeneracy : std::uint8_t {
    None,
    NoBoundaryCut,
    TangentIntersection,
    AlignedTopology,
    AmbiguousTopology,
    NearZeroInterfaceGradient,
    VanishingMeasure,
    DuplicateFragment,
    UnsupportedHighOrder,
    FallbackRule
};

struct GeneratedInterfaceBoundaryIntersectionMarkerKey {
    LevelSetInterfaceSource source{};
    std::string domain_id{};
    Real isovalue{0.0};
    int interface_marker{-1};
    int boundary_marker{-1};
    int requested_marker{-1};

    [[nodiscard]] std::string stableKey() const;
};

[[nodiscard]] std::uint64_t stableGeneratedInterfaceBoundaryIntersectionMarkerHash(
    const GeneratedInterfaceBoundaryIntersectionMarkerKey& key);

[[nodiscard]] int stableGeneratedInterfaceBoundaryIntersectionMarker(
    const GeneratedInterfaceBoundaryIntersectionMarkerKey& key,
    int marker_base = 2000000,
    int marker_range = 1000000);

struct GeneratedInterfaceBoundaryIntersectionRequest {
    LevelSetInterfaceSource source{};
    std::string generated_domain_id{};
    Real isovalue{0.0};
    int interface_marker{-1};
    int boundary_marker{-1};
    int intersection_marker{-1};
    Real tolerance{1.0e-12};
    int quadrature_order{1};
    // Generated contact sets use one strict contract: points, normals,
    // tangents, and d-2 weights are all parent-reference data. The assembler
    // maps the rule to physical space once.
    geometry::CutGeometryFrame frame{geometry::CutGeometryFrame::Reference};
    std::uint64_t mesh_geometry_revision{0};
    std::uint64_t mesh_topology_revision{0};
    std::uint64_t ownership_revision{0};
    std::uint64_t quadrature_policy_key{0};
    std::uint64_t source_value_revision{0};
    bool keep_degenerate_fragments{false};

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] int resolvedIntersectionMarker() const;
};

struct GeneratedInterfaceBoundaryIntersectionScalarField {
    std::function<Real(GlobalIndex)> value_at_node{};

    [[nodiscard]] bool valid() const noexcept {
        return static_cast<bool>(value_at_node);
    }
};

struct GeneratedInterfaceBoundaryIntersectionQuadraturePoint {
    std::array<Real, 3> point{{0.0, 0.0, 0.0}};
    std::array<Real, 3> parent_coordinate{{0.0, 0.0, 0.0}};
    std::array<Real, 3> interface_normal{{1.0, 0.0, 0.0}};
    std::array<Real, 3> boundary_normal{{0.0, 1.0, 0.0}};
    std::array<Real, 3> tangent{{1.0, 0.0, 0.0}};
    Real weight{0.0};
    Real reference_measure_factor{0.0};
    Real level_set_residual{0.0};
    Real gradient_norm{0.0};
};

struct GeneratedInterfaceBoundaryIntersectionFragment {
    int interface_marker{-1};
    int boundary_marker{-1};
    int intersection_marker{-1};
    MeshIndex parent_cell{static_cast<MeshIndex>(-1)};
    MeshIndex parent_face{static_cast<MeshIndex>(-1)};
    GlobalIndex parent_cell_global_id{INVALID_GLOBAL_INDEX};
    GlobalIndex parent_face_global_id{INVALID_GLOBAL_INDEX};
    int owner_rank{-1};
    LocalIndex local_fragment_index{INVALID_LOCAL_INDEX};
    std::uint64_t stable_id{0};
    std::uint64_t source_interface_stable_id{0};
    // Representation used by the authoritative source surface fragment.
    // Contact quadrature is a trace of that fragment, so snapshot validation
    // must evaluate this same representation rather than infer one from the
    // contact-rule construction algorithm.
    std::string represented_implicit_geometry_mode{};
    std::string represented_implicit_quadrature_backend{};
    std::string represented_implicit_fallback_status{};
    GeneratedInterfaceBoundaryIntersectionKind kind{
        GeneratedInterfaceBoundaryIntersectionKind::Point};
    GeneratedInterfaceBoundaryIntersectionDegeneracy degeneracy{
        GeneratedInterfaceBoundaryIntersectionDegeneracy::None};
    Real measure{0.0};
    std::array<Real, 3> interface_normal{{1.0, 0.0, 0.0}};
    std::array<Real, 3> boundary_normal{{0.0, 1.0, 0.0}};
    std::array<Real, 3> tangent{{1.0, 0.0, 0.0}};
    std::string topology_id{};
    std::string diagnostic{};
    // Exact reference-frame trace vertices inherited from the authoritative
    // source interface fragment.  Quadrature samples are not a substitute for
    // these endpoints when constructing the adjacent wet/dry wall patches.
    std::vector<std::array<Real, 3>> vertices{};
    std::vector<GeneratedInterfaceBoundaryIntersectionQuadraturePoint>
        quadrature_points{};

    [[nodiscard]] bool active() const noexcept;
};

struct GeneratedInterfaceBoundaryIntersectionSummary {
    int interface_marker{-1};
    int boundary_marker{-1};
    int intersection_marker{-1};
    std::size_t fragment_count{0};
    std::size_t active_fragment_count{0};
    std::size_t quadrature_point_count{0};
    std::size_t skipped_fragment_count{0};
    std::size_t fallback_fragment_count{0};
    std::size_t tangent_intersection_count{0};
    std::size_t aligned_topology_count{0};
    std::size_t ambiguous_topology_count{0};
    std::size_t near_zero_gradient_count{0};
    std::size_t vanishing_measure_count{0};
    Real measure{0.0};
    Real min_weight{0.0};
    Real max_weight{0.0};
    std::map<int, Real> measure_by_boundary_marker{};
};

struct GeneratedInterfaceBoundaryProvenanceSummary {
    std::size_t source_surface_fragment_count{0};
    std::size_t active_contact_fragment_count{0};
    std::size_t referenced_source_surface_fragment_count{0};
    std::size_t orphan_contact_fragment_count{0};
    std::size_t duplicate_source_surface_id_count{0};
    std::size_t stale_revision_count{0};
    Real max_level_set_residual{0.0};
};

class GeneratedInterfaceBoundaryIntersectionDomain {
public:
    GeneratedInterfaceBoundaryIntersectionDomain() = default;
    explicit GeneratedInterfaceBoundaryIntersectionDomain(
        GeneratedInterfaceBoundaryIntersectionRequest request);

    [[nodiscard]] const GeneratedInterfaceBoundaryIntersectionRequest&
    request() const noexcept;
    [[nodiscard]] int marker() const noexcept;
    [[nodiscard]] int boundaryMarker() const noexcept;
    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] const std::vector<
        GeneratedInterfaceBoundaryIntersectionFragment>& fragments() const noexcept;

    void addFragment(GeneratedInterfaceBoundaryIntersectionFragment fragment);
    void addSkippedFragment(GeneratedInterfaceBoundaryIntersectionFragment fragment);

    [[nodiscard]] GeneratedInterfaceBoundaryIntersectionSummary summary()
        const noexcept;
    [[nodiscard]] std::vector<geometry::CutQuadratureRule>
    intersectionQuadratureRules() const;

private:
    GeneratedInterfaceBoundaryIntersectionRequest request_{};
    std::vector<GeneratedInterfaceBoundaryIntersectionFragment> fragments_{};
};

[[nodiscard]] GeneratedInterfaceBoundaryIntersectionDomain
buildGeneratedInterfaceBoundaryIntersectionDomain(
    GeneratedInterfaceBoundaryIntersectionRequest request,
    const LevelSetInterfaceDomain& interface_domain,
    const assembly::IMeshAccess& mesh);

[[nodiscard]] GeneratedInterfaceBoundaryIntersectionDomain
buildGeneratedInterfaceBoundaryIntersectionDomain(
    GeneratedInterfaceBoundaryIntersectionRequest request,
    const LevelSetInterfaceDomain& interface_domain,
    const assembly::IMeshAccess& mesh,
    const GeneratedInterfaceBoundaryIntersectionScalarField& scalar_field);

[[nodiscard]] GeneratedInterfaceBoundaryProvenanceSummary
validateGeneratedInterfaceBoundaryProvenance(
    const GeneratedInterfaceBoundaryIntersectionDomain& contact_domain,
    const LevelSetInterfaceDomain& interface_domain);

[[nodiscard]] std::vector<int> boundaryMarkers(const assembly::IMeshAccess& mesh);

} // namespace interfaces
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_INTERFACES_GENERATEDINTERFACEBOUNDARYINTERSECTIONDOMAIN_H

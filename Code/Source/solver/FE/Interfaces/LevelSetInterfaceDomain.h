/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_INTERFACES_LEVELSETINTERFACEDOMAIN_H
#define SVMP_FE_INTERFACES_LEVELSETINTERFACEDOMAIN_H

/**
 * @file LevelSetInterfaceDomain.h
 * @brief Physics-neutral data model for generated level-set interface domains.
 */

#include "Core/Types.h"
#include "Geometry/CutQuadrature.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace interfaces {

enum class CutInterfaceSourceKind : std::uint8_t {
    Field,
    Evaluator
};

enum class CutInterfaceFragmentKind : std::uint8_t {
    Segment,
    Polygon,
    CurvedPatch
};

enum class CutInterfaceSideTag : std::uint8_t {
    Unknown,
    Negative,
    Positive
};

enum class CutInterfaceDegeneracy : std::uint8_t {
    None,
    NoCut,
    FullZeroCell,
    VertexTouch,
    EdgeTouch,
    NearlyTangent,
    SmallFragment
};

struct LevelSetInterfaceSource {
    CutInterfaceSourceKind kind{CutInterfaceSourceKind::Field};
    FieldId field_id{INVALID_FIELD_ID};
    std::string evaluator_id{};
    std::uint64_t layout_revision{0};
    std::uint64_t value_revision{0};

    [[nodiscard]] static LevelSetInterfaceSource fromField(
        FieldId field_id,
        std::uint64_t layout_revision = 0,
        std::uint64_t value_revision = 0) {
        LevelSetInterfaceSource source;
        source.kind = CutInterfaceSourceKind::Field;
        source.field_id = field_id;
        source.layout_revision = layout_revision;
        source.value_revision = value_revision;
        return source;
    }

    [[nodiscard]] static LevelSetInterfaceSource fromEvaluator(
        std::string evaluator_id,
        std::uint64_t layout_revision = 0,
        std::uint64_t value_revision = 0) {
        LevelSetInterfaceSource source;
        source.kind = CutInterfaceSourceKind::Evaluator;
        source.evaluator_id = std::move(evaluator_id);
        source.layout_revision = layout_revision;
        source.value_revision = value_revision;
        return source;
    }

    [[nodiscard]] bool valid() const noexcept {
        if (kind == CutInterfaceSourceKind::Field) {
            return field_id != INVALID_FIELD_ID;
        }
        return !evaluator_id.empty();
    }

    [[nodiscard]] std::string identifier() const {
        if (kind == CutInterfaceSourceKind::Field) {
            return "field:" + std::to_string(field_id);
        }
        return evaluator_id;
    }
};

struct CutInterfaceDomainRequest {
    LevelSetInterfaceSource source{};
    int interface_marker{-1};
    Real isovalue{0.0};
    Real tolerance{1.0e-12};
    int quadrature_order{1};
    int interface_quadrature_order{-1};
    int volume_quadrature_order{-1};
    geometry::CutGeometryFrame frame{geometry::CutGeometryFrame::Reference};
    std::uint64_t mesh_geometry_revision{0};
    std::uint64_t mesh_topology_revision{0};
    std::uint64_t ownership_revision{0};
    std::uint64_t quadrature_policy_key{0};
    bool keep_degenerate_fragments{false};

    [[nodiscard]] int resolvedInterfaceQuadratureOrder() const noexcept {
        return interface_quadrature_order >= 0 ? interface_quadrature_order
                                               : quadrature_order;
    }

    [[nodiscard]] int resolvedVolumeQuadratureOrder() const noexcept {
        return volume_quadrature_order >= 0 ? volume_quadrature_order
                                            : quadrature_order;
    }

    [[nodiscard]] bool valid() const noexcept {
        return interface_marker >= 0 && source.valid() && tolerance > Real{0.0} &&
               quadrature_order >= 0 &&
               resolvedInterfaceQuadratureOrder() >= 0 &&
               resolvedVolumeQuadratureOrder() >= 0;
    }
};

struct GeneratedInterfaceMarkerKey {
    LevelSetInterfaceSource source{};
    std::string domain_id{};
    Real isovalue{0.0};
    int requested_marker{-1};

    [[nodiscard]] std::string stableKey() const {
        return source.identifier() + "|" + domain_id + "|" + std::to_string(isovalue);
    }
};

[[nodiscard]] inline std::uint64_t stableMarkerHash(
    const GeneratedInterfaceMarkerKey& key) {
    std::uint64_t h = 1469598103934665603ull;
    const auto mix = [&h](std::uint64_t value) noexcept {
        h ^= value;
        h *= 1099511628211ull;
    };
    const auto stable_key = key.stableKey();
    for (const char c : stable_key) {
        mix(static_cast<unsigned char>(c));
    }
    return h;
}

[[nodiscard]] inline int stableGeneratedInterfaceMarker(
    const GeneratedInterfaceMarkerKey& key,
    int marker_base = 1000000,
    int marker_range = 1000000) {
    if (key.requested_marker >= 0) {
        return key.requested_marker;
    }
    if (marker_base < 0 || marker_range <= 0) {
        throw std::invalid_argument("generated interface marker range must be positive");
    }
    const auto offset =
        static_cast<int>(stableMarkerHash(key) % static_cast<std::uint64_t>(marker_range));
    return marker_base + offset;
}

class GeneratedInterfaceMarkerRegistry {
public:
    explicit GeneratedInterfaceMarkerRegistry(int marker_base = 1000000,
                                              int marker_range = 1000000)
        : marker_base_(marker_base)
        , marker_range_(marker_range)
    {
        if (marker_base_ < 0 || marker_range_ <= 0) {
            throw std::invalid_argument("generated interface marker registry requires a positive marker range");
        }
    }

    [[nodiscard]] int assign(const GeneratedInterfaceMarkerKey& key) {
        const auto stable_key = key.stableKey();
        const auto found = key_to_marker_.find(stable_key);
        if (found != key_to_marker_.end()) {
            return found->second;
        }

        int marker = stableGeneratedInterfaceMarker(key, marker_base_, marker_range_);
        if (key.requested_marker >= 0) {
            const auto owner = marker_to_key_.find(marker);
            if (owner != marker_to_key_.end() && owner->second != stable_key) {
                throw std::invalid_argument("generated interface marker is already assigned to another domain");
            }
        } else {
            const int start = marker;
            while (true) {
                const auto owner = marker_to_key_.find(marker);
                if (owner == marker_to_key_.end() || owner->second == stable_key) {
                    break;
                }
                marker = marker_base_ + ((marker - marker_base_ + 1) % marker_range_);
                if (marker == start) {
                    throw std::invalid_argument("generated interface marker registry is full");
                }
            }
        }

        key_to_marker_[stable_key] = marker;
        marker_to_key_[marker] = stable_key;
        return marker;
    }

    [[nodiscard]] bool contains(const GeneratedInterfaceMarkerKey& key) const {
        return key_to_marker_.find(key.stableKey()) != key_to_marker_.end();
    }

    [[nodiscard]] bool containsMarker(int marker) const {
        return marker_to_key_.find(marker) != marker_to_key_.end();
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return key_to_marker_.size();
    }

private:
    int marker_base_{1000000};
    int marker_range_{1000000};
    std::unordered_map<std::string, int> key_to_marker_{};
    std::unordered_map<int, std::string> marker_to_key_{};
};

struct CutInterfaceVertex {
    std::array<Real, 3> point{{0.0, 0.0, 0.0}};
    std::array<Real, 3> parent_coordinate{{0.0, 0.0, 0.0}};
    Real level_set_value{0.0};
    std::uint64_t stable_id{0};
};

struct CutInterfaceQuadraturePoint {
    std::array<Real, 3> point{{0.0, 0.0, 0.0}};
    std::array<Real, 3> parent_coordinate{{0.0, 0.0, 0.0}};
    std::array<Real, 3> normal{{1.0, 0.0, 0.0}};
    Real weight{0.0};
};

struct CutInterfaceVolumeRegion {
    int interface_marker{-1};
    MeshIndex parent_cell{static_cast<MeshIndex>(-1)};
    LocalIndex local_region_index{INVALID_LOCAL_INDEX};
    std::uint64_t stable_id{0};
    geometry::CutIntegrationSide side{geometry::CutIntegrationSide::Negative};
    std::array<Real, 3> centroid{{0.0, 0.0, 0.0}};
    std::array<Real, 3> normal{{1.0, 0.0, 0.0}};
    Real parent_measure{0.0};
    Real measure{0.0};
    Real volume_fraction{0.0};
    Real min_level_set_value{0.0};
    Real max_level_set_value{0.0};
    std::string topology_id{};
    bool full_cell_equivalent{false};

    [[nodiscard]] bool active() const noexcept {
        return interface_marker >= 0 &&
               parent_cell >= static_cast<MeshIndex>(0) &&
               side != geometry::CutIntegrationSide::Interface &&
               measure > Real{0.0};
    }

    [[nodiscard]] geometry::CutQuadratureRule toCutQuadratureRule(
        const CutInterfaceDomainRequest& request) const {
        if (side == geometry::CutIntegrationSide::Interface) {
            throw std::invalid_argument("level-set volume region requires Negative or Positive side");
        }
        const int volume_order = request.resolvedVolumeQuadratureOrder();
        if (volume_order < 0) {
            throw std::invalid_argument("cut-volume quadrature order must be nonnegative");
        }
        if (volume_order > 1) {
            throw std::invalid_argument("cut-volume quadrature order is not supported for this region");
        }
        const int exact_order = volume_order;

        geometry::CutQuadratureRule rule;
        rule.kind = geometry::CutQuadratureKind::Volume;
        rule.side = side;
        rule.measure = measure;
        rule.parent_measure = parent_measure;
        rule.volume_fraction = volume_fraction;
        rule.exact_for_constants = true;
        rule.exact_polynomial_order = exact_order;
        rule.policy.kind = geometry::CutQuadratureConstructionKind::MomentFittedImplicit;
        rule.policy.polynomial_order = exact_order;
        rule.policy.moment_fitted = exact_order > 0;
        rule.policy.name = exact_order == 0
                               ? "conservative-level-set-volume"
                               : "linear-moment-fitted-level-set-volume";
        rule.provenance.embedded_geometry_id = request.source.identifier();
        rule.provenance.cut_topology_id = topology_id;
        rule.provenance.parent_entity = parent_cell;
        rule.provenance.marker = interface_marker;
        rule.provenance.cut_topology_revision = stable_id;
        rule.provenance.predicate_policy_key = request.quadrature_policy_key;
        rule.provenance.construction = rule.policy.kind;
        rule.provenance.frame = request.frame;
        rule.provenance_id = request.source.identifier();
        rule.frame = request.frame;
        rule.full_cell_equivalent = full_cell_equivalent;
        rule.points.push_back(geometry::CutQuadraturePoint{centroid, normal, measure});
        return rule;
    }
};

struct CutInterfaceFragment {
    int interface_marker{-1};
    MeshIndex parent_cell{static_cast<MeshIndex>(-1)};
    LocalIndex local_fragment_index{INVALID_LOCAL_INDEX};
    std::uint64_t stable_id{0};
    CutInterfaceFragmentKind kind{CutInterfaceFragmentKind::Segment};
    CutInterfaceDegeneracy degeneracy{CutInterfaceDegeneracy::None};
    CutInterfaceSideTag minus_side{CutInterfaceSideTag::Negative};
    CutInterfaceSideTag plus_side{CutInterfaceSideTag::Positive};
    std::array<Real, 3> normal{{1.0, 0.0, 0.0}};
    Real measure{0.0};
    Real curvature_estimate{0.0};
    Real negative_volume_fraction{0.0};
    Real positive_volume_fraction{0.0};
    Real min_level_set_value{0.0};
    Real max_level_set_value{0.0};
    std::string topology_id{};
    std::vector<CutInterfaceVertex> vertices{};
    std::vector<CutInterfaceQuadraturePoint> quadrature_points{};

    [[nodiscard]] bool active() const noexcept {
        return interface_marker >= 0 &&
               parent_cell >= static_cast<MeshIndex>(0) &&
               measure > Real{0.0} &&
               degeneracy != CutInterfaceDegeneracy::NoCut &&
               degeneracy != CutInterfaceDegeneracy::FullZeroCell;
    }

    [[nodiscard]] std::size_t quadraturePointCount() const noexcept {
        return quadrature_points.size();
    }

    [[nodiscard]] geometry::CutQuadratureRule toCutQuadratureRule(
        const CutInterfaceDomainRequest& request) const {
        const int supported_order = kind == CutInterfaceFragmentKind::CurvedPatch ? 0 : 1;
        const int quadrature_order = request.resolvedInterfaceQuadratureOrder();
        if (quadrature_order < 0) {
            throw std::invalid_argument("cut-interface quadrature order must be nonnegative");
        }
        if (quadrature_order > supported_order) {
            throw std::invalid_argument("cut-interface quadrature order is not supported for this fragment");
        }

        geometry::CutQuadratureRule rule;
        rule.kind = geometry::CutQuadratureKind::Interface;
        rule.side = geometry::CutIntegrationSide::Interface;
        rule.measure = measure;
        rule.parent_measure = Real{0.0};
        rule.volume_fraction = Real{0.0};
        rule.exact_for_constants = true;
        rule.exact_polynomial_order = quadrature_order;
        rule.policy.kind = kind == CutInterfaceFragmentKind::CurvedPatch
                               ? geometry::CutQuadratureConstructionKind::CurvedTopologySubdivision
                               : geometry::CutQuadratureConstructionKind::TopologySubdivision;
        rule.policy.polynomial_order = quadrature_order;
        rule.policy.name = kind == CutInterfaceFragmentKind::CurvedPatch
                               ? "curved-level-set-interface"
                               : (quadrature_order == 0
                                      ? "constant-level-set-interface"
                                      : "linear-level-set-interface");
        rule.provenance.embedded_geometry_id = request.source.identifier();
        rule.provenance.cut_topology_id = topology_id;
        rule.provenance.parent_entity = parent_cell;
        rule.provenance.marker = interface_marker;
        rule.provenance.cut_topology_revision = stable_id;
        rule.provenance.predicate_policy_key = request.quadrature_policy_key;
        rule.provenance.construction = rule.policy.kind;
        rule.provenance.frame = request.frame;
        rule.provenance_id = request.source.identifier();
        rule.frame = request.frame;
        rule.curved_geometry = kind == CutInterfaceFragmentKind::CurvedPatch;
        rule.points.reserve(quadrature_points.size());
        for (const auto& point : quadrature_points) {
            geometry::CutQuadraturePoint qp;
            qp.point = point.point;
            qp.normal = point.normal;
            qp.weight = point.weight;
            rule.points.push_back(qp);
        }
        return rule;
    }
};

struct CutInterfaceDomainSummary {
    int interface_marker{-1};
    std::size_t fragment_count{0};
    std::size_t active_fragment_count{0};
    std::size_t volume_region_count{0};
    std::size_t active_volume_region_count{0};
    std::size_t quadrature_point_count{0};
    std::size_t degenerate_fragment_count{0};
    Real measure{0.0};
    Real negative_volume_measure{0.0};
    Real positive_volume_measure{0.0};
};

[[nodiscard]] inline std::uint64_t cutInterfaceStableId(
    int interface_marker,
    MeshIndex parent_cell,
    LocalIndex local_fragment_index,
    std::uint64_t source_revision) noexcept {
    std::uint64_t h = 1469598103934665603ull;
    const auto mix = [&h](std::uint64_t value) noexcept {
        h ^= value;
        h *= 1099511628211ull;
    };
    mix(static_cast<std::uint64_t>(interface_marker));
    mix(static_cast<std::uint64_t>(parent_cell));
    mix(static_cast<std::uint64_t>(local_fragment_index));
    mix(source_revision);
    return h;
}

[[nodiscard]] inline std::uint64_t cutVolumeStableId(
    int interface_marker,
    MeshIndex parent_cell,
    LocalIndex local_region_index,
    geometry::CutIntegrationSide side,
    std::uint64_t source_revision) noexcept {
    std::uint64_t h = 1469598103934665603ull;
    const auto mix = [&h](std::uint64_t value) noexcept {
        h ^= value;
        h *= 1099511628211ull;
    };
    mix(static_cast<std::uint64_t>(interface_marker));
    mix(static_cast<std::uint64_t>(parent_cell));
    mix(static_cast<std::uint64_t>(local_region_index));
    mix(static_cast<std::uint64_t>(side));
    mix(source_revision);
    return h;
}

class LevelSetInterfaceDomain {
public:
    LevelSetInterfaceDomain() = default;

    explicit LevelSetInterfaceDomain(CutInterfaceDomainRequest request)
        : request_(std::move(request))
    {
    }

    [[nodiscard]] const CutInterfaceDomainRequest& request() const noexcept {
        return request_;
    }

    [[nodiscard]] int marker() const noexcept {
        return request_.interface_marker;
    }

    [[nodiscard]] bool empty() const noexcept {
        return fragments_.empty();
    }

    [[nodiscard]] const std::vector<CutInterfaceFragment>& fragments() const noexcept {
        return fragments_;
    }

    [[nodiscard]] const std::vector<CutInterfaceVolumeRegion>& volumeRegions() const noexcept {
        return volume_regions_;
    }

    [[nodiscard]] std::vector<MeshIndex> cutCells() const {
        std::vector<MeshIndex> cells;
        for (const auto& fragment : fragments_) {
            if (fragment.active()) {
                cells.push_back(fragment.parent_cell);
            }
        }
        std::sort(cells.begin(), cells.end());
        cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
        return cells;
    }

    void clearFragments() {
        fragments_.clear();
        volume_regions_.clear();
    }

    void addFragment(CutInterfaceFragment fragment) {
        if (fragment.interface_marker < 0) {
            fragment.interface_marker = request_.interface_marker;
        }
        if (fragment.local_fragment_index == INVALID_LOCAL_INDEX) {
            fragment.local_fragment_index = static_cast<LocalIndex>(fragments_.size());
        }
        if (fragment.stable_id == 0) {
            fragment.stable_id =
                cutInterfaceStableId(fragment.interface_marker,
                                     fragment.parent_cell,
                                     fragment.local_fragment_index,
                                     request_.source.value_revision);
        }
        fragments_.push_back(std::move(fragment));
    }

    void addVolumeRegion(CutInterfaceVolumeRegion region) {
        if (region.interface_marker < 0) {
            region.interface_marker = request_.interface_marker;
        }
        if (region.local_region_index == INVALID_LOCAL_INDEX) {
            region.local_region_index = static_cast<LocalIndex>(volume_regions_.size());
        }
        if (region.stable_id == 0) {
            region.stable_id =
                cutVolumeStableId(region.interface_marker,
                                  region.parent_cell,
                                  region.local_region_index,
                                  region.side,
                                  request_.source.value_revision);
        }
        volume_regions_.push_back(std::move(region));
    }

    [[nodiscard]] std::vector<geometry::CutQuadratureRule> interfaceQuadratureRules() const {
        std::vector<geometry::CutQuadratureRule> rules;
        rules.reserve(fragments_.size());
        for (const auto& fragment : fragments_) {
            if (fragment.active()) {
                rules.push_back(fragment.toCutQuadratureRule(request_));
            }
        }
        return rules;
    }

    [[nodiscard]] std::vector<geometry::CutQuadratureRule> volumeQuadratureRules() const {
        std::vector<geometry::CutQuadratureRule> rules;
        rules.reserve(volume_regions_.size());
        for (const auto& region : volume_regions_) {
            if (region.active()) {
                rules.push_back(region.toCutQuadratureRule(request_));
            }
        }
        return rules;
    }

    [[nodiscard]] CutInterfaceDomainSummary summary() const noexcept {
        CutInterfaceDomainSummary s;
        s.interface_marker = request_.interface_marker;
        s.fragment_count = fragments_.size();
        s.volume_region_count = volume_regions_.size();
        for (const auto& fragment : fragments_) {
            if (fragment.degeneracy != CutInterfaceDegeneracy::None) {
                ++s.degenerate_fragment_count;
            }
            if (!fragment.active()) {
                continue;
            }
            ++s.active_fragment_count;
            s.quadrature_point_count += fragment.quadraturePointCount();
            s.measure += fragment.measure;
        }
        for (const auto& region : volume_regions_) {
            if (!region.active()) {
                continue;
            }
            ++s.active_volume_region_count;
            if (region.side == geometry::CutIntegrationSide::Negative) {
                s.negative_volume_measure += region.measure;
            } else if (region.side == geometry::CutIntegrationSide::Positive) {
                s.positive_volume_measure += region.measure;
            }
        }
        return s;
    }

private:
    CutInterfaceDomainRequest request_{};
    std::vector<CutInterfaceFragment> fragments_{};
    std::vector<CutInterfaceVolumeRegion> volume_regions_{};
};

} // namespace interfaces
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_INTERFACES_LEVELSETINTERFACEDOMAIN_H

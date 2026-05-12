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

#include <array>
#include <cstdint>
#include <string>
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
    geometry::CutGeometryFrame frame{geometry::CutGeometryFrame::Reference};
    std::uint64_t mesh_geometry_revision{0};
    std::uint64_t mesh_topology_revision{0};
    std::uint64_t ownership_revision{0};
    std::uint64_t quadrature_policy_key{0};
    bool keep_degenerate_fragments{false};

    [[nodiscard]] bool valid() const noexcept {
        return interface_marker >= 0 && source.valid() && tolerance > Real{0.0};
    }
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
        geometry::CutQuadratureRule rule;
        rule.kind = geometry::CutQuadratureKind::Interface;
        rule.side = geometry::CutIntegrationSide::Interface;
        rule.measure = measure;
        rule.parent_measure = Real{0.0};
        rule.volume_fraction = Real{0.0};
        rule.exact_for_constants = true;
        rule.exact_polynomial_order = kind == CutInterfaceFragmentKind::CurvedPatch ? 0 : 1;
        rule.policy.kind = kind == CutInterfaceFragmentKind::CurvedPatch
                               ? geometry::CutQuadratureConstructionKind::CurvedTopologySubdivision
                               : geometry::CutQuadratureConstructionKind::TopologySubdivision;
        rule.policy.polynomial_order = rule.exact_polynomial_order;
        rule.policy.name = kind == CutInterfaceFragmentKind::CurvedPatch
                               ? "curved-level-set-interface"
                               : "linear-level-set-interface";
        rule.provenance.embedded_geometry_id = request.source.identifier();
        rule.provenance.cut_topology_id = topology_id;
        rule.provenance.parent_entity = parent_cell;
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
    std::size_t quadrature_point_count{0};
    std::size_t degenerate_fragment_count{0};
    Real measure{0.0};
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

    void clearFragments() {
        fragments_.clear();
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

    [[nodiscard]] CutInterfaceDomainSummary summary() const noexcept {
        CutInterfaceDomainSummary s;
        s.interface_marker = request_.interface_marker;
        s.fragment_count = fragments_.size();
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
        return s;
    }

private:
    CutInterfaceDomainRequest request_{};
    std::vector<CutInterfaceFragment> fragments_{};
};

} // namespace interfaces
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_INTERFACES_LEVELSETINTERFACEDOMAIN_H

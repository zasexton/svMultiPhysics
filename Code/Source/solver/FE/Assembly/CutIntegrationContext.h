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
#include "Interfaces/FreeSurfaceGeometrySnapshot.h"
#include "Interfaces/GeneratedActiveBoundaryDomain.h"
#include "Interfaces/GeneratedInterfaceBoundaryIntersectionDomain.h"
#include "Interfaces/LevelSetInterfaceDomain.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#include "Mesh/Search/CutCell.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
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
    std::uint64_t source_value_revision = 0;
    std::uint64_t free_surface_snapshot_revision_key = 0;
};

/**
 * @brief Typed ownership record for a generated active physical boundary.
 *
 * The generated marker is an assembly key, not sufficient provenance on its
 * own. This record retains the physical boundary, level-set owner, active side,
 * geometry frame, and all request revisions needed to validate an explicit
 * exterior-boundary selection before a form is installed or a new cut context
 * is published.
 */
struct GeneratedActiveBoundaryProvenance {
    int generated_active_boundary_marker{-1};
    interfaces::GeneratedActiveBoundaryMarkerKey owner{};
    geometry::CutGeometryFrame frame{geometry::CutGeometryFrame::Reference};
    int quadrature_order{0};
    Real clipping_tolerance{0.0};
    std::uint64_t mesh_geometry_revision{0};
    std::uint64_t mesh_topology_revision{0};
    std::uint64_t ownership_revision{0};
    std::uint64_t quadrature_policy_key{0};
    std::uint64_t source_value_revision{0};
    std::string stable_owner_key{};

    [[nodiscard]] int physicalBoundaryMarker() const noexcept {
        return owner.boundary_marker;
    }

    [[nodiscard]] int volumeInterfaceMarker() const noexcept {
        return owner.interface_marker;
    }
};

/**
 * @brief Publication state for one generated level-set domain.
 *
 * The request retains communicator-wide source and policy bindings together
 * with rank-local mesh and achieved-quadrature stamps used for local
 * validation. Generated fragments and quadrature rules are rank-local and
 * intentionally excluded.
 */
struct GeneratedLevelSetInterfacePublicationProvenance {
    int generated_interface_marker{-1};
    interfaces::CutInterfaceDomainRequest request{};
    std::optional<geometry::CutIntegrationSide> volume_side_filter{};
    std::string publication_domain_id{};
};

/**
 * @brief Publication state for one generated contact domain.
 *
 * The stable owner key identifies marker ownership while the typed request
 * preserves communicator-wide policy bindings and rank-local geometry stamps
 * needed for local validation.
 */
struct GeneratedInterfaceBoundaryPublicationProvenance {
    int generated_interface_boundary_marker{-1};
    interfaces::GeneratedInterfaceBoundaryIntersectionRequest request{};
    std::string stable_owner_key{};
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

struct CutGeometrySensitivitySampleMetadata {
    std::array<Real, 3> parent_parametric_coordinate{{0.0, 0.0, 0.0}};
    std::vector<MeshIndex> influencing_parent_geometry_dofs{};
    std::vector<Real> shape_values{};
    std::vector<std::array<Real, 3>> shape_gradients{};
};

enum class CutIntegrationAssemblyPath : std::uint8_t {
    Standard,
    MatrixFree,
    Interpreter,
    AD,
    SymbolicTangent,
    JIT
};

struct CutGeometrySensitivityMetadata {
    MeshIndex parent_entity = static_cast<MeshIndex>(-1);
    std::string target_kind{};
    std::string construction_policy{};
    std::string provenance_id{};
    std::uint64_t source_stable_id = 0;
    std::uint64_t cut_topology_revision = 0;
    std::uint64_t quadrature_policy_key = 0;
    bool ad_compatible = false;
    bool location_sensitivity_available = false;
    bool jacobian_sensitivity_available = false;
    bool measure_sensitivity_available = false;
    bool normal_sensitivity_available = false;
    bool quadrature_weight_sensitivity_available = false;
    std::vector<MeshIndex> parent_geometry_dofs{};
    std::vector<CutGeometrySensitivitySampleMetadata> samples{};
    std::vector<CutIntegrationAssemblyPath> visible_to_paths{};
};

struct CutIntegrationBinding {
    MeshIndex parent_entity = static_cast<MeshIndex>(-1);
    geometry::CutQuadratureKind kind = geometry::CutQuadratureKind::Volume;
    geometry::CutIntegrationSide side = geometry::CutIntegrationSide::Negative;
    std::uint64_t cut_revision_key = 0;
    std::uint64_t cut_topology_revision = 0;
    std::uint64_t quadrature_policy_key = 0;
    std::uint64_t source_value_revision = 0;
    std::vector<CutIntegrationAssemblyPath> visible_to_paths{};
    int marker = -1;
    std::uint64_t free_surface_snapshot_revision_key = 0;
};

struct CutScalarOperatorPoint {
    geometry::CutQuadratureKind kind = geometry::CutQuadratureKind::Volume;
    geometry::CutIntegrationSide side = geometry::CutIntegrationSide::Negative;
    MeshIndex parent_entity = static_cast<MeshIndex>(-1);
    std::array<Real, 3> point{{0.0, 0.0, 0.0}};
    std::array<Real, 3> normal{{0.0, 0.0, 0.0}};
    Real weight = 0.0;
    Real volume_fraction = 0.0;
    std::uint64_t cut_topology_revision = 0;
    std::uint64_t quadrature_policy_key = 0;
    geometry::CutQuadratureConstructionKind construction =
        geometry::CutQuadratureConstructionKind::TopologySubdivision;
    geometry::CutGeometryFrame frame = geometry::CutGeometryFrame::Reference;
};

struct CutFixedGeometryAssemblyDiagnostics {
    std::size_t high_order_volume_rule_count = 0;
    std::size_t high_order_interface_rule_count = 0;
    std::size_t refreshed_frozen_quadrature_rule_count = 0;
    std::size_t differentiated_quadrature_rule_count = 0;
    std::size_t missing_tangent_policy_rule_count = 0;
    std::uint64_t quadrature_policy_key_xor = 0;

    [[nodiscard]] bool hasHighOrderRules() const noexcept {
        return high_order_volume_rule_count > 0u ||
               high_order_interface_rule_count > 0u;
    }
};

struct GeneratedVolumeRuleDiagnostics {
    Real active_volume = 0.0;
    Real cut_cell_active_volume = 0.0;
    Real full_cell_active_volume = 0.0;
    std::size_t rule_count = 0u;
    std::size_t cut_cell_rules = 0u;
    std::size_t full_cell_rules = 0u;
    std::size_t quadrature_points = 0u;
    std::size_t null_rules = 0u;
    std::size_t zero_quadrature_rules = 0u;
    std::size_t nonfinite_measure_rules = 0u;
    std::size_t negative_measure_rules = 0u;
    std::size_t nonfinite_volume_fraction_rules = 0u;
    Real min_rule_measure = std::numeric_limits<Real>::infinity();
    Real max_rule_measure = -std::numeric_limits<Real>::infinity();
    Real min_volume_fraction = std::numeric_limits<Real>::infinity();
    Real max_volume_fraction = -std::numeric_limits<Real>::infinity();
    int min_exact_order = std::numeric_limits<int>::max();
    int max_exact_order = std::numeric_limits<int>::min();

    void recordRule(const geometry::CutQuadratureRule& rule) noexcept {
        ++rule_count;
        active_volume += rule.measure;
        quadrature_points += rule.points.size();
        if (rule.points.empty()) {
            ++zero_quadrature_rules;
        }
        if (!std::isfinite(rule.measure)) {
            ++nonfinite_measure_rules;
        }
        if (rule.measure < Real{0.0}) {
            ++negative_measure_rules;
        }
        if (!std::isfinite(rule.volume_fraction)) {
            ++nonfinite_volume_fraction_rules;
        } else {
            min_volume_fraction = std::min(min_volume_fraction,
                                           rule.volume_fraction);
            max_volume_fraction = std::max(max_volume_fraction,
                                           rule.volume_fraction);
        }
        if (std::isfinite(rule.measure)) {
            min_rule_measure = std::min(min_rule_measure, rule.measure);
            max_rule_measure = std::max(max_rule_measure, rule.measure);
        }
        min_exact_order = std::min(min_exact_order, rule.exact_polynomial_order);
        max_exact_order = std::max(max_exact_order, rule.exact_polynomial_order);
        if (rule.full_cell_equivalent) {
            full_cell_active_volume += rule.measure;
            ++full_cell_rules;
        } else {
            cut_cell_active_volume += rule.measure;
            ++cut_cell_rules;
        }
    }

    void normalizeEmptyExtrema() noexcept {
        if (!std::isfinite(min_rule_measure)) {
            min_rule_measure = Real{0.0};
        }
        if (!std::isfinite(max_rule_measure)) {
            max_rule_measure = Real{0.0};
        }
        if (!std::isfinite(min_volume_fraction)) {
            min_volume_fraction = Real{0.0};
        }
        if (!std::isfinite(max_volume_fraction)) {
            max_volume_fraction = Real{0.0};
        }
        if (min_exact_order == std::numeric_limits<int>::max()) {
            min_exact_order = 0;
        }
        if (max_exact_order == std::numeric_limits<int>::min()) {
            max_exact_order = 0;
        }
    }
};

struct CutScalarOperatorEvaluation {
    CutIntegrationAssemblyPath path = CutIntegrationAssemblyPath::Standard;
    std::size_t volume_rule_count = 0;
    std::size_t interface_rule_count = 0;
    std::size_t volume_point_count = 0;
    std::size_t interface_point_count = 0;
    Real parent_measure = 0.0;
    Real negative_volume_measure = 0.0;
    Real positive_volume_measure = 0.0;
    Real interface_measure = 0.0;
    Real negative_volume_integral = 0.0;
    Real positive_volume_integral = 0.0;
    Real interface_integral = 0.0;
    CutFixedGeometryAssemblyDiagnostics fixed_geometry_diagnostics{};

    [[nodiscard]] Real volumeIntegral() const noexcept {
        return negative_volume_integral + positive_volume_integral;
    }

    [[nodiscard]] Real totalIntegral() const noexcept {
        return volumeIntegral() + interface_integral;
    }
};

struct CutFacetSetOperatorEvaluation {
    CutIntegrationAssemblyPath path = CutIntegrationAssemblyPath::Standard;
    int marker = -1;
    std::size_t facet_rule_count = 0;
    std::size_t facet_point_count = 0;
    Real measure = 0.0;
    Real integral = 0.0;
};

struct CutFacetSetFacetMetadata {
    MeshIndex facet{static_cast<MeshIndex>(-1)};
    MeshIndex first_cell{static_cast<MeshIndex>(-1)};
    MeshIndex second_cell{static_cast<MeshIndex>(-1)};
    Real stabilization_scale{0.0};
    std::uint64_t stable_id{0};
};

struct CutFacetSetHandle {
    int marker{-1};
    std::string name{};
    std::vector<MeshIndex> facets{};
    std::vector<CutFacetSetFacetMetadata> facet_metadata{};
    std::uint64_t stable_id{0};

    [[nodiscard]] bool valid() const noexcept {
        return marker >= 0 && !facets.empty() && stable_id != 0u;
    }

    [[nodiscard]] bool empty() const noexcept {
        return facets.empty();
    }

    [[nodiscard]] bool containsFacet(MeshIndex facet) const noexcept {
        return std::binary_search(facets.begin(), facets.end(), facet);
    }

    [[nodiscard]] bool hasFacetMetadata() const noexcept {
        return !facet_metadata.empty();
    }

    [[nodiscard]] const CutFacetSetFacetMetadata*
    metadataForFacet(MeshIndex facet) const noexcept {
        const auto it = std::lower_bound(
            facet_metadata.begin(),
            facet_metadata.end(),
            facet,
            [](const CutFacetSetFacetMetadata& metadata, MeshIndex value) {
                return metadata.facet < value;
            });
        if (it == facet_metadata.end() || it->facet != facet) {
            return nullptr;
        }
        return &*it;
    }

    [[nodiscard]] Real stabilizationScaleForFacet(MeshIndex facet) const noexcept {
        const auto* metadata = metadataForFacet(facet);
        return metadata == nullptr ? Real{0.0} : metadata->stabilization_scale;
    }
};

class CutIntegrationContext {
public:
    [[nodiscard]] static constexpr Real maxCutCellStabilizationScale() noexcept {
        return Real{1.0e3};
    }

    [[nodiscard]] static constexpr Real defaultMinGeneratedCutVolumeFraction() noexcept {
        return Real{1.0e-8};
    }

    [[nodiscard]] static Real minGeneratedCutVolumeFraction() noexcept {
        static const Real value = []() noexcept {
            const char* env = std::getenv("SVMP_MIN_GENERATED_CUT_VOLUME_FRACTION");
            if (env != nullptr && env[0] != '\0') {
                char* end = nullptr;
                const double parsed = std::strtod(env, &end);
                if (end != env && std::isfinite(parsed) &&
                    parsed > 0.0 && parsed < 1.0) {
                    return static_cast<Real>(parsed);
                }
            }
            return defaultMinGeneratedCutVolumeFraction();
        }();
        return value;
    }

    [[nodiscard]] static bool shouldPruneGeneratedVolumeRule(
        const geometry::CutQuadratureRule& rule) noexcept {
        return !rule.full_cell_equivalent &&
               std::isfinite(rule.volume_fraction) &&
               rule.volume_fraction > Real{0.0} &&
               rule.volume_fraction < minGeneratedCutVolumeFraction();
    }

    [[nodiscard]] std::uint64_t contentRevision() const noexcept {
        return content_revision_;
    }

    void clear() {
        metadata_.clear();
        volume_rules_.clear();
        interface_rules_.clear();
        facet_set_rules_.clear();
        generated_volume_rule_indices_by_marker_.clear();
        generated_volume_markers_.clear();
        generated_interface_rule_indices_by_marker_.clear();
        generated_interface_two_sided_bindings_by_marker_.clear();
        generated_level_set_interface_markers_.clear();
        generated_level_set_interface_provenance_by_marker_.clear();
        generated_active_boundary_provenance_by_marker_.clear();
        generated_active_boundary_markers_by_physical_marker_.clear();
        generated_interface_boundary_provenance_by_marker_.clear();
        free_surface_geometry_snapshots_.clear();
        free_surface_snapshot_revision_by_marker_.clear();
        generated_interface_markers_.clear();
        facet_set_rule_indices_by_marker_.clear();
        facet_set_markers_.clear();
        facet_set_handles_.clear();
        facet_set_handle_indices_by_marker_.clear();
        expected_source_value_revision_by_marker_.clear();
        generated_volume_rule_indices_by_marker_and_side_.clear();
        kinematic_data_.clear();
        stabilization_hooks_.clear();
        bindings_.clear();
        sensitivity_metadata_.clear();
        generated_pruned_volume_rule_count_ = 0u;
        generated_pruned_volume_measure_ = Real{0.0};
        markModified();
    }

    void addVolumeRule(CutCellAssemblyMetadata metadata,
                       geometry::CutQuadratureRule rule) {
        metadata_.push_back(std::move(metadata));
        volume_rules_.push_back(std::move(rule));
        markModified();
    }

    void addInterfaceRule(geometry::CutQuadratureRule rule) {
        interface_rules_.push_back(std::move(rule));
        markModified();
    }

    void addFacetSetRule(int marker, geometry::CutQuadratureRule rule) {
        auto& indices = facet_set_rule_indices_by_marker_[marker];
        if (indices.empty()) {
            facet_set_markers_.push_back(marker);
        }
        indices.push_back(facet_set_rules_.size());
        facet_set_rules_.push_back(std::move(rule));
        markModified();
    }

    const CutFacetSetHandle& addFacetSetHandle(CutFacetSetHandle handle) {
        if (handle.marker < 0) {
            throw std::invalid_argument("cut facet set handle requires a nonnegative marker");
        }
        normalizeFacetSetHandle(handle);
        bindFacetStabilizationScales(handle, metadata_);
        std::sort(handle.facets.begin(), handle.facets.end());
        handle.facets.erase(std::unique(handle.facets.begin(), handle.facets.end()),
                            handle.facets.end());
        if (handle.stable_id == 0u) {
            handle.stable_id = facetSetStableId(handle.marker, handle.facets);
        }

        const auto index = facet_set_handles_.size();
        facet_set_handle_indices_by_marker_[handle.marker] = index;
        facet_set_handles_.push_back(std::move(handle));
        markModified();
        return facet_set_handles_.back();
    }

    void addGeneratedVolumeRule(int marker,
                                CutCellAssemblyMetadata metadata,
                                geometry::CutQuadratureRule rule) {
        if (marker < 0) {
            return;
        }
        if (rule.kind != geometry::CutQuadratureKind::Volume) {
            throw std::invalid_argument("generated level-set volume rule must be a volume rule");
        }
        if (rule.side == geometry::CutIntegrationSide::Interface) {
            throw std::invalid_argument("generated level-set volume rule requires Negative or Positive side");
        }
        if (metadata.side != rule.side) {
            throw std::invalid_argument("generated level-set volume metadata side must match the rule side");
        }
        if (rule.provenance.marker == -1) {
            rule.provenance.marker = marker;
        } else if (rule.provenance.marker != marker) {
            throw std::invalid_argument(
                "generated level-set volume rule provenance marker must "
                "match the registration marker");
        }
        if (metadata.source_value_revision != 0u &&
            rule.provenance.source_value_revision != 0u &&
            metadata.source_value_revision != rule.provenance.source_value_revision) {
            throw std::invalid_argument(
                "generated level-set volume rule source revision must match metadata");
        }
        if (rule.provenance.source_value_revision == 0u) {
            rule.provenance.source_value_revision = metadata.source_value_revision;
        } else if (metadata.source_value_revision == 0u) {
            metadata.source_value_revision =
                rule.provenance.source_value_revision;
        }
        if (metadata.free_surface_snapshot_revision_key != 0u &&
            rule.provenance.free_surface_snapshot_revision_key != 0u &&
            metadata.free_surface_snapshot_revision_key !=
                rule.provenance.free_surface_snapshot_revision_key) {
            throw std::invalid_argument(
                "generated level-set volume rule snapshot revision must match metadata");
        }
        if (rule.provenance.free_surface_snapshot_revision_key == 0u) {
            rule.provenance.free_surface_snapshot_revision_key =
                metadata.free_surface_snapshot_revision_key;
        } else if (metadata.free_surface_snapshot_revision_key == 0u) {
            metadata.free_surface_snapshot_revision_key =
                rule.provenance.free_surface_snapshot_revision_key;
        }
        const auto level_set_publication =
            generated_level_set_interface_provenance_by_marker_.find(
                marker);
        const bool has_level_set_publication =
            level_set_publication !=
            generated_level_set_interface_provenance_by_marker_.end();
        const bool unrelated_interface_namespace =
            !has_level_set_publication &&
            (hasGeneratedLevelSetInterfaceMarker(marker) ||
             generated_interface_rule_indices_by_marker_.contains(
                 marker) ||
             generated_interface_two_sided_bindings_by_marker_.contains(
                 marker) ||
             free_surface_snapshot_revision_by_marker_.contains(marker));
        if (generated_interface_boundary_provenance_by_marker_.contains(
                marker) ||
            generated_active_boundary_provenance_by_marker_.contains(
                marker) ||
            unrelated_interface_namespace) {
            throw std::invalid_argument(
                "generated volume marker collides with an "
                "existing generated-interface marker");
        }
        if (has_level_set_publication) {
            validateGeneratedVolumeLevelSetComposition(
                marker,
                level_set_publication->second,
                metadata,
                rule,
                hasGeneratedLevelSetInterfaceMarker(marker));
        }
        if (shouldPruneGeneratedVolumeRule(rule)) {
            ++generated_pruned_volume_rule_count_;
            if (std::isfinite(rule.measure) && rule.measure > Real{0.0}) {
                generated_pruned_volume_measure_ += rule.measure;
            }
            markModified();
            return;
        }

        const bool keep_binding_alignment = bindings_.size() == volume_rules_.size();
        const auto old_metadata_size = metadata_.size();
        const auto old_volume_rule_size = volume_rules_.size();
        const auto old_binding_size = bindings_.size();
        const auto old_generated_volume_marker_size =
            generated_volume_markers_.size();
        const auto old_content_revision = content_revision_;
        const auto marker_indices_before =
            generated_volume_rule_indices_by_marker_.find(marker);
        const bool had_marker_indices =
            marker_indices_before !=
            generated_volume_rule_indices_by_marker_.end();
        const auto old_marker_index_count =
            had_marker_indices
                ? marker_indices_before->second.size()
                : std::size_t{0u};
        const auto side_buckets_before =
            generated_volume_rule_indices_by_marker_and_side_.find(
                marker);
        const bool had_side_buckets =
            side_buckets_before !=
            generated_volume_rule_indices_by_marker_and_side_.end();
        std::array<std::size_t, 2> old_side_index_counts{{0u, 0u}};
        std::array<GeneratedVolumeRuleDiagnostics, 2>
            old_side_diagnostics{};
        if (had_side_buckets) {
            for (std::size_t side = 0u; side < 2u; ++side) {
                old_side_index_counts[side] =
                    side_buckets_before->second[side].indices.size();
                old_side_diagnostics[side] =
                    side_buckets_before->second[side].diagnostics;
            }
        }

        try {
            const auto index = volume_rules_.size();
            const bool new_marker =
                generated_volume_rule_indices_by_marker_.find(marker) ==
                generated_volume_rule_indices_by_marker_.end();
            if (new_marker) {
                generated_volume_markers_.push_back(marker);
            }

            generated_volume_rule_indices_by_marker_[marker].push_back(index);
            auto& side_bucket =
                generated_volume_rule_indices_by_marker_and_side_[marker]
                    [volumeSideIndex(rule.side)];
            side_bucket.indices.push_back(index);
            metadata_.push_back(std::move(metadata));
            volume_rules_.push_back(std::move(rule));
            side_bucket.diagnostics.recordRule(volume_rules_.back());

            if (keep_binding_alignment) {
                const auto& stored_metadata = metadata_.back();
                const auto& stored_rule = volume_rules_.back();
                CutIntegrationBinding binding;
                binding.marker = marker;
                binding.parent_entity = stored_metadata.parent_entity;
                binding.kind = geometry::CutQuadratureKind::Volume;
                binding.side = stored_rule.side;
                binding.cut_revision_key = stored_metadata.revision_key;
                binding.cut_topology_revision =
                    stored_metadata.cut_topology_revision;
                binding.quadrature_policy_key =
                    stored_metadata.quadrature_policy_key;
                binding.source_value_revision =
                    stored_metadata.source_value_revision;
                binding.free_surface_snapshot_revision_key =
                    stored_metadata.free_surface_snapshot_revision_key;
                binding.visible_to_paths = {
                    CutIntegrationAssemblyPath::Standard,
                    CutIntegrationAssemblyPath::MatrixFree,
                    CutIntegrationAssemblyPath::Interpreter,
                    CutIntegrationAssemblyPath::AD,
                    CutIntegrationAssemblyPath::SymbolicTangent,
                    CutIntegrationAssemblyPath::JIT};
                bindings_.push_back(std::move(binding));
            }
            markModified();
        } catch (...) {
            metadata_.resize(old_metadata_size);
            volume_rules_.resize(old_volume_rule_size);
            bindings_.resize(old_binding_size);
            generated_volume_markers_.resize(
                old_generated_volume_marker_size);

            const auto marker_indices =
                generated_volume_rule_indices_by_marker_.find(marker);
            if (had_marker_indices) {
                if (marker_indices !=
                    generated_volume_rule_indices_by_marker_.end()) {
                    marker_indices->second.resize(
                        old_marker_index_count);
                }
            } else if (
                marker_indices !=
                generated_volume_rule_indices_by_marker_.end()) {
                generated_volume_rule_indices_by_marker_.erase(
                    marker_indices);
            }

            const auto side_buckets =
                generated_volume_rule_indices_by_marker_and_side_.find(
                    marker);
            if (had_side_buckets) {
                if (side_buckets !=
                    generated_volume_rule_indices_by_marker_and_side_
                        .end()) {
                    for (std::size_t side = 0u;
                         side < 2u;
                         ++side) {
                        side_buckets->second[side].indices.resize(
                            old_side_index_counts[side]);
                        side_buckets->second[side].diagnostics =
                            old_side_diagnostics[side];
                    }
                }
            } else if (
                side_buckets !=
                generated_volume_rule_indices_by_marker_and_side_
                    .end()) {
                generated_volume_rule_indices_by_marker_and_side_.erase(
                    side_buckets);
            }

            content_revision_ = old_content_revision;
            throw;
        }
    }

    void addGeneratedInterfaceDomain(
        const interfaces::LevelSetInterfaceDomain& domain,
        std::optional<geometry::CutIntegrationSide> volume_side_filter =
            std::nullopt,
        std::string_view publication_domain_id = {}) {
        const int marker = domain.marker();
        if (marker < 0) {
            return;
        }
        if (std::find(
                generated_level_set_interface_markers_.begin(),
                generated_level_set_interface_markers_.end(),
                marker) !=
                generated_level_set_interface_markers_.end() ||
            generated_level_set_interface_provenance_by_marker_.contains(
                marker)) {
            throw std::invalid_argument(
                "generated level-set interface domain was imported more than once");
        }
        if (volume_side_filter.has_value() &&
            *volume_side_filter == geometry::CutIntegrationSide::Interface) {
            throw std::invalid_argument(
                "generated level-set volume side filter requires Negative or Positive side");
        }
        GeneratedLevelSetInterfacePublicationProvenance
            publication_provenance;
        publication_provenance.generated_interface_marker =
            marker;
        publication_provenance.request =
            domain.request();
        publication_provenance.volume_side_filter =
            volume_side_filter;
        publication_provenance.publication_domain_id =
            publication_domain_id.empty()
                ? domain.request().generated_domain_id
                : std::string(publication_domain_id);
        if (generated_interface_boundary_provenance_by_marker_.find(marker) !=
                generated_interface_boundary_provenance_by_marker_.end() ||
            generated_active_boundary_provenance_by_marker_.find(marker) !=
                generated_active_boundary_provenance_by_marker_.end() ||
            generated_interface_rule_indices_by_marker_.contains(marker) ||
            generated_interface_two_sided_bindings_by_marker_.contains(
                marker)) {
            throw std::invalid_argument(
                "generated level-set interface marker collides with an imported generated boundary marker");
        }
        generated_level_set_interface_markers_.reserve(
            generated_level_set_interface_markers_.size() + 1u);
        generated_level_set_interface_provenance_by_marker_.reserve(
            generated_level_set_interface_provenance_by_marker_
                    .size() +
                1u);
        const auto old_metadata_size = metadata_.size();
        const auto old_volume_rule_size = volume_rules_.size();
        const auto old_interface_rule_size = interface_rules_.size();
        const auto old_binding_size = bindings_.size();
        const auto old_sensitivity_metadata_size =
            sensitivity_metadata_.size();
        const auto old_generated_volume_marker_size =
            generated_volume_markers_.size();
        const auto old_generated_interface_marker_size =
            generated_interface_markers_.size();
        const auto old_generated_level_set_interface_marker_size =
            generated_level_set_interface_markers_.size();
        const auto old_generated_pruned_volume_rule_count =
            generated_pruned_volume_rule_count_;
        const auto old_generated_pruned_volume_measure =
            generated_pruned_volume_measure_;
        const auto old_content_revision = content_revision_;
        const auto marker_indices_before =
            generated_volume_rule_indices_by_marker_.find(marker);
        const bool had_marker_indices =
            marker_indices_before !=
            generated_volume_rule_indices_by_marker_.end();
        const auto old_marker_index_count =
            had_marker_indices
                ? marker_indices_before->second.size()
                : std::size_t{0u};
        const auto side_buckets_before =
            generated_volume_rule_indices_by_marker_and_side_.find(
                marker);
        const bool had_side_buckets =
            side_buckets_before !=
            generated_volume_rule_indices_by_marker_and_side_.end();
        std::array<std::size_t, 2> old_side_index_counts{{0u, 0u}};
        std::array<GeneratedVolumeRuleDiagnostics, 2>
            old_side_diagnostics{};
        if (had_side_buckets) {
            for (std::size_t side = 0u; side < 2u; ++side) {
                old_side_index_counts[side] =
                    side_buckets_before->second[side].indices.size();
                old_side_diagnostics[side] =
                    side_buckets_before->second[side].diagnostics;
            }
        }
        struct ExpectedRevisionCheckpoint {
            int marker{-1};
            bool existed{false};
            std::uint64_t value{0u};
        };
        std::vector<int> affected_expected_revision_markers{
            marker};
        const auto snapshot_binding =
            free_surface_snapshot_revision_by_marker_.find(
                marker);
        if (snapshot_binding !=
            free_surface_snapshot_revision_by_marker_.end()) {
            for (const auto& [bound_marker, revision] :
                 free_surface_snapshot_revision_by_marker_) {
                if (revision == snapshot_binding->second &&
                    std::find(
                        affected_expected_revision_markers.begin(),
                        affected_expected_revision_markers.end(),
                        bound_marker) ==
                        affected_expected_revision_markers.end()) {
                    affected_expected_revision_markers.push_back(
                        bound_marker);
                }
            }
        }
        std::vector<ExpectedRevisionCheckpoint>
            expected_revision_checkpoints;
        expected_revision_checkpoints.reserve(
            affected_expected_revision_markers.size());
        for (const int affected_marker :
             affected_expected_revision_markers) {
            const auto existing =
                expected_source_value_revision_by_marker_.find(
                    affected_marker);
            expected_revision_checkpoints.push_back(
                ExpectedRevisionCheckpoint{
                    .marker = affected_marker,
                    .existed =
                        existing !=
                        expected_source_value_revision_by_marker_
                            .end(),
                    .value =
                        existing ==
                                expected_source_value_revision_by_marker_
                                    .end()
                            ? std::uint64_t{0u}
                            : existing->second,
                });
        }
        try {
        const auto [stored_provenance,
                    inserted_provenance] =
            generated_level_set_interface_provenance_by_marker_
                .emplace(
                    marker,
                    std::move(
                        publication_provenance));
        if (!inserted_provenance) {
            throw std::invalid_argument(
                "generated level-set interface publication provenance was registered more than once");
        }
        static_cast<void>(stored_provenance);
        setExpectedGeneratedSourceValueRevision(marker,
                                                domain.request().source.value_revision);
        if (had_marker_indices) {
            for (const auto index : marker_indices_before->second) {
                if (index >= metadata_.size() ||
                    index >= volume_rules_.size()) {
                    throw std::invalid_argument(
                        "generated volume marker contains an invalid stored rule index");
                }
                auto existing_metadata = metadata_[index];
                auto existing_rule = volume_rules_[index];
                validateGeneratedVolumeLevelSetComposition(
                    marker,
                    stored_provenance->second,
                    existing_metadata,
                    existing_rule,
                    false);
            }
        }
        const auto make_sensitivity_metadata =
            [&](const interfaces::GeneratedInterfaceSensitivityRecord& record) {
                CutGeometrySensitivityMetadata metadata;
                metadata.parent_entity = record.parent_cell;
                metadata.target_kind = record.target_kind;
                metadata.construction_policy = record.construction_policy;
                metadata.provenance_id = record.provenance_id;
                metadata.source_stable_id = record.source_stable_id;
                metadata.cut_topology_revision = record.cut_topology_revision;
                metadata.quadrature_policy_key = record.quadrature_policy_key;
                metadata.ad_compatible = record.ad_compatible;
                metadata.location_sensitivity_available =
                    record.location_sensitivity_available;
                metadata.jacobian_sensitivity_available =
                    record.jacobian_sensitivity_available;
                metadata.measure_sensitivity_available =
                    record.measure_sensitivity_available;
                metadata.normal_sensitivity_available =
                    record.normal_sensitivity_available;
                metadata.quadrature_weight_sensitivity_available =
                    record.quadrature_weight_sensitivity_available;
                metadata.parent_geometry_dofs =
                    record.parent_geometry_dofs;
                metadata.samples.reserve(record.samples.size());
                for (const auto& sample : record.samples) {
                    CutGeometrySensitivitySampleMetadata sample_metadata;
                    sample_metadata.parent_parametric_coordinate =
                        sample.parent_parametric_coordinate;
                    sample_metadata.influencing_parent_geometry_dofs =
                        sample.influencing_parent_geometry_dofs.empty()
                            ? record.parent_geometry_dofs
                            : sample.influencing_parent_geometry_dofs;
                    sample_metadata.shape_values = sample.shape_values;
                    sample_metadata.shape_gradients = sample.shape_gradients;
                    metadata.samples.push_back(std::move(sample_metadata));
                }
                metadata.visible_to_paths = {
                    CutIntegrationAssemblyPath::Standard,
                    CutIntegrationAssemblyPath::MatrixFree,
                    CutIntegrationAssemblyPath::Interpreter,
                    CutIntegrationAssemblyPath::AD,
                    CutIntegrationAssemblyPath::SymbolicTangent,
                    CutIntegrationAssemblyPath::JIT};
                return metadata;
            };
        const bool publish_generated_sensitivity =
            domain.request().geometry_tangent_policy == "DifferentiatedQuadrature" &&
            domain.request().implicit_geometry_mode == "LinearCorner" &&
            domain.request().implicit_quadrature_backend == "LinearCorner";
        if (publish_generated_sensitivity) {
            for (const auto& record : domain.sensitivityRecords()) {
                addSensitivityMetadata(make_sensitivity_metadata(record));
            }
        }
        auto volume_rules = domain.volumeQuadratureRules();
        for (auto& rule : volume_rules) {
            if (volume_side_filter.has_value() &&
                rule.side != *volume_side_filter) {
                continue;
            }
            CutCellAssemblyMetadata metadata;
            metadata.cell = rule.provenance.parent_entity;
            metadata.parent_entity = rule.provenance.parent_entity;
            metadata.volume_fraction = rule.volume_fraction;
            metadata.side = rule.side;
            metadata.embedded_normal =
                rule.points.empty() ? std::array<Real, 3>{{0.0, 0.0, 0.0}}
                                    : rule.points.front().normal;
            metadata.provenance_id = rule.provenance.embedded_geometry_id;
            metadata.cut_topology_id = rule.provenance.cut_topology_id;
            metadata.revision_key = rule.provenance.cut_topology_revision;
            metadata.cut_topology_revision = rule.provenance.cut_topology_revision;
            metadata.quadrature_policy_key = rule.provenance.predicate_policy_key;
            metadata.source_value_revision = domain.request().source.value_revision;
            metadata.free_surface_snapshot_revision_key =
                rule.provenance.free_surface_snapshot_revision_key;
            addGeneratedVolumeRule(marker, std::move(metadata), std::move(rule));
        }
        auto rules = domain.interfaceQuadratureRules();
        if (!rules.empty()) {
            auto& indices = generated_interface_rule_indices_by_marker_[marker];
            if (indices.empty()) {
                generated_interface_markers_.push_back(marker);
            }
            for (auto& rule : rules) {
                indices.push_back(interface_rules_.size());
                interface_rules_.push_back(std::move(rule));
                markModified();
            }
        }
        auto two_sided_bindings = domain.twoSidedParentCellBindings();
        if (!two_sided_bindings.empty()) {
            auto& stored_bindings =
                generated_interface_two_sided_bindings_by_marker_[marker];
            stored_bindings.insert(stored_bindings.end(),
                                   two_sided_bindings.begin(),
                                   two_sided_bindings.end());
            std::sort(stored_bindings.begin(), stored_bindings.end(),
                      [](const auto& a, const auto& b) noexcept {
                          if (a.parent_cell != b.parent_cell) {
                              return a.parent_cell < b.parent_cell;
                          }
                          if (a.interface_marker != b.interface_marker) {
                              return a.interface_marker < b.interface_marker;
                          }
                          return a.interface_stable_id < b.interface_stable_id;
                      });
            stored_bindings.erase(
                std::unique(stored_bindings.begin(), stored_bindings.end(),
                            [](const auto& a, const auto& b) noexcept {
                                return a.parent_cell == b.parent_cell &&
                                       a.interface_marker == b.interface_marker &&
                                       a.interface_stable_id == b.interface_stable_id;
                            }),
                stored_bindings.end());
            markModified();
        }
        generated_level_set_interface_markers_.push_back(marker);
        markModified();
        } catch (...) {
            metadata_.resize(old_metadata_size);
            volume_rules_.resize(old_volume_rule_size);
            interface_rules_.resize(old_interface_rule_size);
            bindings_.resize(old_binding_size);
            sensitivity_metadata_.resize(
                old_sensitivity_metadata_size);
            generated_volume_markers_.resize(
                old_generated_volume_marker_size);
            generated_interface_markers_.resize(
                old_generated_interface_marker_size);
            generated_level_set_interface_markers_.resize(
                old_generated_level_set_interface_marker_size);
            generated_level_set_interface_provenance_by_marker_.erase(
                marker);
            const auto marker_indices =
                generated_volume_rule_indices_by_marker_.find(marker);
            if (had_marker_indices) {
                if (marker_indices !=
                    generated_volume_rule_indices_by_marker_.end()) {
                    marker_indices->second.resize(
                        old_marker_index_count);
                }
            } else if (
                marker_indices !=
                generated_volume_rule_indices_by_marker_.end()) {
                generated_volume_rule_indices_by_marker_.erase(
                    marker_indices);
            }
            const auto side_buckets =
                generated_volume_rule_indices_by_marker_and_side_.find(
                    marker);
            if (had_side_buckets) {
                if (side_buckets !=
                    generated_volume_rule_indices_by_marker_and_side_
                        .end()) {
                    for (std::size_t side = 0u;
                         side < 2u;
                         ++side) {
                        side_buckets->second[side].indices.resize(
                            old_side_index_counts[side]);
                        side_buckets->second[side].diagnostics =
                            old_side_diagnostics[side];
                    }
                }
            } else if (
                side_buckets !=
                generated_volume_rule_indices_by_marker_and_side_
                    .end()) {
                generated_volume_rule_indices_by_marker_and_side_.erase(
                    side_buckets);
            }
            generated_interface_rule_indices_by_marker_.erase(
                marker);
            generated_interface_two_sided_bindings_by_marker_.erase(
                marker);
            for (const auto& checkpoint :
                 expected_revision_checkpoints) {
                if (checkpoint.existed) {
                    const auto found =
                        expected_source_value_revision_by_marker_
                            .find(checkpoint.marker);
                    if (found !=
                        expected_source_value_revision_by_marker_
                            .end()) {
                        found->second = checkpoint.value;
                    }
                } else {
                    expected_source_value_revision_by_marker_.erase(
                        checkpoint.marker);
                }
            }
            generated_pruned_volume_rule_count_ =
                old_generated_pruned_volume_rule_count;
            generated_pruned_volume_measure_ =
                old_generated_pruned_volume_measure;
            content_revision_ = old_content_revision;
            throw;
        }
    }

    void addGeneratedInterfaceBoundaryIntersectionDomain(
        const interfaces::GeneratedInterfaceBoundaryIntersectionDomain& domain) {
        const int marker = domain.marker();
        if (marker < 0) {
            throw std::invalid_argument(
                "generated interface-boundary intersection domain requires a nonnegative marker");
        }
        interfaces::GeneratedInterfaceBoundaryIntersectionMarkerKey marker_key;
        marker_key.source = domain.request().source;
        marker_key.domain_id = domain.request().generated_domain_id;
        marker_key.isovalue = domain.request().isovalue;
        marker_key.interface_marker = domain.request().interface_marker;
        marker_key.boundary_marker = domain.request().boundary_marker;
        marker_key.requested_marker = domain.request().intersection_marker;
        const auto stable_key = marker_key.stableKey();
        GeneratedInterfaceBoundaryPublicationProvenance
            publication_provenance;
        publication_provenance
            .generated_interface_boundary_marker =
            marker;
        publication_provenance.request =
            domain.request();
        publication_provenance.stable_owner_key =
            stable_key;
        const auto registered_provenance =
            generated_interface_boundary_provenance_by_marker_
                .find(marker);
        if (registered_provenance !=
            generated_interface_boundary_provenance_by_marker_
                .end()) {
            throw std::invalid_argument(
                registered_provenance
                            ->second
                            .stable_owner_key ==
                        stable_key
                    ? "generated interface-boundary domain was imported more than once"
                    : "generated interface-boundary marker hash collision detected");
        }
        const auto existing_rules =
            generated_interface_rule_indices_by_marker_.find(marker);
        if (generated_active_boundary_provenance_by_marker_.contains(
                marker) ||
            hasGeneratedLevelSetInterfaceMarker(marker) ||
            generated_volume_rule_indices_by_marker_.contains(marker) ||
            (existing_rules !=
             generated_interface_rule_indices_by_marker_.end())) {
            throw std::invalid_argument(
                "generated interface-boundary marker collides with an existing generated-interface marker");
        }

        auto rules =
            domain.intersectionQuadratureRules();
        generated_interface_boundary_provenance_by_marker_.reserve(
            generated_interface_boundary_provenance_by_marker_
                    .size() +
                1u);
        generated_interface_rule_indices_by_marker_.reserve(
            generated_interface_rule_indices_by_marker_.size() + 1u);
        generated_interface_markers_.reserve(
            generated_interface_markers_.size() + 1u);
        interface_rules_.reserve(
            interface_rules_.size() + rules.size());
        const auto old_interface_rule_size =
            interface_rules_.size();
        const auto old_interface_marker_size =
            generated_interface_markers_.size();
        const auto old_content_revision = content_revision_;
        try {
            const auto [registered, inserted] =
                generated_interface_boundary_provenance_by_marker_
                    .emplace(
                        marker,
                        std::move(
                            publication_provenance));
            if (!inserted) {
                throw std::invalid_argument(
                    registered
                                ->second
                                .stable_owner_key ==
                            stable_key
                        ? "generated interface-boundary domain was imported more than once"
                        : "generated interface-boundary marker hash collision detected");
            }
            auto [stored_indices, inserted_indices] =
                generated_interface_rule_indices_by_marker_
                    .try_emplace(marker);
            if (!inserted_indices) {
                throw std::invalid_argument(
                    "generated interface-boundary marker was registered concurrently");
            }
            auto& indices = stored_indices->second;
            indices.reserve(rules.size());
            generated_interface_markers_.push_back(marker);
            if (rules.empty()) {
                markModified();
            }
            for (auto& rule : rules) {
                indices.push_back(interface_rules_.size());
                interface_rules_.push_back(std::move(rule));
                markModified();
            }
        } catch (...) {
            interface_rules_.resize(
                old_interface_rule_size);
            generated_interface_markers_.resize(
                old_interface_marker_size);
            generated_interface_rule_indices_by_marker_.erase(
                marker);
            generated_interface_boundary_provenance_by_marker_.erase(
                marker);
            content_revision_ = old_content_revision;
            throw;
        }
    }

    void addGeneratedActiveBoundaryDomain(
        const interfaces::GeneratedActiveBoundaryDomain& domain) {
        const int marker = domain.marker();
        if (marker < 0) {
            throw std::invalid_argument(
                "generated active-boundary domain requires a nonnegative marker");
        }
        interfaces::GeneratedActiveBoundaryMarkerKey marker_key;
        marker_key.source = domain.request().source;
        marker_key.domain_id = domain.request().generated_domain_id;
        marker_key.isovalue = domain.request().isovalue;
        marker_key.interface_marker = domain.request().interface_marker;
        marker_key.boundary_marker = domain.request().boundary_marker;
        marker_key.side = domain.request().side;
        marker_key.requested_marker = domain.request().active_boundary_marker;
        const auto stable_key = marker_key.stableKey();
        GeneratedActiveBoundaryProvenance provenance;
        provenance.generated_active_boundary_marker = marker;
        provenance.owner = marker_key;
        provenance.frame = domain.request().frame;
        provenance.quadrature_order = domain.request().quadrature_order;
        provenance.clipping_tolerance =
            domain.request().tolerance;
        provenance.mesh_geometry_revision =
            domain.request().mesh_geometry_revision;
        provenance.mesh_topology_revision =
            domain.request().mesh_topology_revision;
        provenance.ownership_revision =
            domain.request().ownership_revision;
        provenance.quadrature_policy_key =
            domain.request().quadrature_policy_key;
        provenance.source_value_revision =
            domain.request().source_value_revision;
        provenance.stable_owner_key = stable_key;
        const auto registered_provenance =
            generated_active_boundary_provenance_by_marker_
                .find(marker);
        if (registered_provenance !=
            generated_active_boundary_provenance_by_marker_
                .end()) {
            throw std::invalid_argument(
                registered_provenance->second
                            .stable_owner_key ==
                        stable_key
                    ? "generated active-boundary domain was imported more than once"
                    : "generated active-boundary marker hash collision detected");
        }
        if (generated_interface_boundary_provenance_by_marker_.find(marker) !=
                generated_interface_boundary_provenance_by_marker_.end() ||
            hasGeneratedLevelSetInterfaceMarker(marker) ||
            generated_interface_rule_indices_by_marker_.find(marker) !=
                generated_interface_rule_indices_by_marker_.end() ||
            generated_volume_rule_indices_by_marker_.contains(marker)) {
            throw std::invalid_argument(
                "generated active-boundary marker collides with an existing generated-interface marker");
        }
        auto rules =
            domain.boundaryQuadratureRules();
        const auto [registered, inserted] =
            generated_active_boundary_provenance_by_marker_.emplace(
                marker, std::move(provenance));
        if (!inserted) {
            throw std::invalid_argument(
                registered->second.stable_owner_key == stable_key
                    ? "generated active-boundary domain was imported more than once"
                    : "generated active-boundary marker hash collision detected");
        }
        const auto old_interface_rule_size = interface_rules_.size();
        const auto old_interface_marker_size =
            generated_interface_markers_.size();
        const auto old_content_revision = content_revision_;
        const auto existing_physical_markers =
            generated_active_boundary_markers_by_physical_marker_.find(
                marker_key.boundary_marker);
        const bool had_physical_marker_entry =
            existing_physical_markers !=
            generated_active_boundary_markers_by_physical_marker_.end();
        const auto old_physical_marker_size =
            had_physical_marker_entry
                ? existing_physical_markers->second.size()
                : std::size_t{0u};
        try {
            auto& physical_markers =
                generated_active_boundary_markers_by_physical_marker_[
                    marker_key.boundary_marker];
            physical_markers.push_back(marker);
            auto& indices =
                generated_interface_rule_indices_by_marker_[marker];
            if (indices.empty()) {
                generated_interface_markers_.push_back(marker);
            }
            if (rules.empty()) {
                markModified();
            }
            for (auto& rule : rules) {
                indices.push_back(interface_rules_.size());
                interface_rules_.push_back(std::move(rule));
                markModified();
            }
        } catch (...) {
            interface_rules_.resize(old_interface_rule_size);
            generated_interface_markers_.resize(
                old_interface_marker_size);
            generated_interface_rule_indices_by_marker_.erase(marker);
            const auto physical_markers =
                generated_active_boundary_markers_by_physical_marker_.find(
                    marker_key.boundary_marker);
            if (physical_markers !=
                generated_active_boundary_markers_by_physical_marker_.end()) {
                physical_markers->second.resize(old_physical_marker_size);
                if (!had_physical_marker_entry) {
                    generated_active_boundary_markers_by_physical_marker_.erase(
                        physical_markers);
                }
            }
            generated_active_boundary_provenance_by_marker_.erase(marker);
            content_revision_ = old_content_revision;
            throw;
        }
    }

    void addFreeSurfaceGeometrySnapshot(
        std::shared_ptr<const interfaces::FreeSurfaceGeometrySnapshot> snapshot,
        std::optional<geometry::CutIntegrationSide> volume_side_filter =
            std::nullopt) {
        if (!snapshot || !snapshot->revision().complete()) {
            throw std::invalid_argument(
                "cut integration requires a complete free-surface geometry snapshot");
        }
        const Real threshold_error = std::abs(
            snapshot->policy().minimum_retained_volume_fraction -
            minGeneratedCutVolumeFraction());
        const Real threshold_tolerance =
            Real{64.0} * std::numeric_limits<Real>::epsilon() *
            std::max(Real{1.0}, minGeneratedCutVolumeFraction());
        if (threshold_error > threshold_tolerance) {
            throw std::invalid_argument(
                "free-surface snapshot and cut context use different volume-retention policies");
        }
        if (volume_side_filter == geometry::CutIntegrationSide::Interface) {
            throw std::invalid_argument(
                "free-surface snapshot volume filter requires Negative or Positive side");
        }
        const auto revision_key =
            snapshot->revision().snapshot_revision_key;
        std::vector<int> snapshot_markers;
        const auto preflight_marker = [&](int marker) {
            if (marker < 0) {
                throw std::invalid_argument(
                    "free-surface snapshot contains a negative generated marker");
            }
            if (std::find(snapshot_markers.begin(), snapshot_markers.end(),
                          marker) != snapshot_markers.end()) {
                throw std::invalid_argument(
                    "free-surface snapshot reuses a generated marker");
            }
            if (free_surface_snapshot_revision_by_marker_.contains(marker) ||
                hasGeneratedLevelSetInterfaceMarker(marker) ||
                generated_volume_rule_indices_by_marker_.contains(marker) ||
                generated_interface_rule_indices_by_marker_.contains(marker) ||
                generated_active_boundary_provenance_by_marker_.contains(marker) ||
                generated_interface_boundary_provenance_by_marker_.contains(marker)) {
                throw std::invalid_argument(
                    "free-surface snapshot marker is already imported");
            }
            snapshot_markers.push_back(marker);
        };
        preflight_marker(snapshot->interfaceDomain().marker());
        for (const auto& contact : snapshot->contactDomains()) {
            preflight_marker(contact.marker());
        }
        for (const auto& active : snapshot->activeBoundaryDomains()) {
            preflight_marker(active.marker());
        }
        const auto first_volume_rule = volume_rules_.size();
        const auto first_interface_rule = interface_rules_.size();
        const auto first_metadata = metadata_.size();
        const auto first_binding = bindings_.size();
        const auto first_sensitivity_metadata =
            sensitivity_metadata_.size();
        const auto first_generated_volume_marker =
            generated_volume_markers_.size();
        const auto first_generated_interface_marker =
            generated_interface_markers_.size();
        const auto first_generated_level_set_interface_marker =
            generated_level_set_interface_markers_.size();
        const auto first_snapshot =
            free_surface_geometry_snapshots_.size();
        const auto old_generated_pruned_volume_rule_count =
            generated_pruned_volume_rule_count_;
        const auto old_generated_pruned_volume_measure =
            generated_pruned_volume_measure_;
        const auto old_content_revision = content_revision_;
        struct ExpectedSourceRevisionCheckpoint {
            int marker{-1};
            bool existed{false};
            std::uint64_t value{0u};
        };
        std::vector<int> expected_source_revision_markers =
            snapshot_markers;
        for (const auto& [marker, bound_revision] :
             free_surface_snapshot_revision_by_marker_) {
            if (bound_revision != revision_key ||
                std::find(
                    expected_source_revision_markers.begin(),
                    expected_source_revision_markers.end(),
                    marker) !=
                    expected_source_revision_markers.end()) {
                continue;
            }
            expected_source_revision_markers.push_back(marker);
        }
        std::vector<ExpectedSourceRevisionCheckpoint>
            expected_source_revision_checkpoints;
        expected_source_revision_checkpoints.reserve(
            expected_source_revision_markers.size());
        for (const int marker :
             expected_source_revision_markers) {
            const auto existing =
                expected_source_value_revision_by_marker_.find(
                    marker);
            expected_source_revision_checkpoints.push_back(
                ExpectedSourceRevisionCheckpoint{
                    .marker = marker,
                    .existed =
                        existing !=
                        expected_source_value_revision_by_marker_
                            .end(),
                    .value =
                        existing ==
                                expected_source_value_revision_by_marker_
                                    .end()
                            ? std::uint64_t{0u}
                            : existing->second,
                });
        }
        struct PhysicalMarkerCheckpoint {
            int marker{-1};
            bool existed{false};
            std::size_t size{0u};
        };
        std::vector<PhysicalMarkerCheckpoint>
            physical_marker_checkpoints;
        physical_marker_checkpoints.reserve(
            snapshot->activeBoundaryDomains().size());
        for (const auto& active :
             snapshot->activeBoundaryDomains()) {
            const int physical_marker =
                active.request().boundary_marker;
            const auto duplicate =
                std::find_if(
                    physical_marker_checkpoints.begin(),
                    physical_marker_checkpoints.end(),
                    [physical_marker](const auto& checkpoint) {
                        return checkpoint.marker ==
                               physical_marker;
                    });
            if (duplicate !=
                physical_marker_checkpoints.end()) {
                continue;
            }
            const auto existing =
                generated_active_boundary_markers_by_physical_marker_
                    .find(physical_marker);
            physical_marker_checkpoints.push_back(
                PhysicalMarkerCheckpoint{
                    .marker = physical_marker,
                    .existed =
                        existing !=
                        generated_active_boundary_markers_by_physical_marker_
                            .end(),
                    .size =
                        existing ==
                                generated_active_boundary_markers_by_physical_marker_
                                    .end()
                            ? std::size_t{0u}
                            : existing->second.size(),
                });
        }
        const auto bind_marker = [&](int marker) {
            if (marker < 0) {
                throw std::invalid_argument(
                    "free-surface snapshot contains a negative generated marker");
            }
            const auto [stored, inserted] =
                free_surface_snapshot_revision_by_marker_.emplace(
                    marker, revision_key);
            if (!inserted) {
                throw std::invalid_argument(
                    "generated marker is already bound to a free-surface snapshot revision");
            }
            (void)stored;
        };

        try {
            bind_marker(snapshot->interfaceDomain().marker());
            for (const auto& contact : snapshot->contactDomains()) {
                bind_marker(contact.marker());
            }
            for (const auto& active : snapshot->activeBoundaryDomains()) {
                bind_marker(active.marker());
            }
            addGeneratedInterfaceDomain(
                snapshot->interfaceDomain(),
                volume_side_filter,
                snapshot->revision().domain_id);
            for (const auto& contact : snapshot->contactDomains()) {
                addGeneratedInterfaceBoundaryIntersectionDomain(
                    contact);
            }
            for (const auto& active : snapshot->activeBoundaryDomains()) {
                interfaces::GeneratedActiveBoundaryDomain retained_active(
                    active.request());
                const bool side_is_imported =
                    !volume_side_filter.has_value() ||
                    *volume_side_filter == active.request().side;
                for (const auto& fragment : active.fragments()) {
                    const auto role =
                        active.request().side ==
                                geometry::CutIntegrationSide::Negative
                            ? interfaces::FreeSurfaceGeometryRuleRole::
                                  NegativeExteriorBoundary
                            : interfaces::FreeSurfaceGeometryRuleRole::
                                  PositiveExteriorBoundary;
                    const auto record = std::find_if(
                        snapshot->rules().begin(),
                        snapshot->rules().end(),
                        [&](const auto& candidate) {
                            return candidate.role == role &&
                                   candidate.physical_boundary_marker ==
                                       active.request().boundary_marker &&
                                   candidate.reference_rule.provenance
                                           .cut_topology_revision ==
                                       fragment.stable_id;
                        });
                    if (record == snapshot->rules().end()) {
                        throw std::invalid_argument(
                            "free-surface active-boundary fragment has no snapshot rule record");
                    }
                    if (side_is_imported &&
                        record->retention == interfaces::
                                                 FreeSurfaceGeometryRetention::
                                                     Retained) {
                        retained_active.addFragment(fragment);
                    }
                }
                addGeneratedActiveBoundaryDomain(retained_active);
            }
            for (std::size_t index = first_volume_rule;
                 index < volume_rules_.size();
                 ++index) {
                volume_rules_[index]
                    .provenance
                    .free_surface_snapshot_revision_key =
                    revision_key;
            }
            for (std::size_t index = first_interface_rule;
                 index < interface_rules_.size();
                 ++index) {
                interface_rules_[index]
                    .provenance
                    .free_surface_snapshot_revision_key =
                    revision_key;
            }
            for (std::size_t index = first_metadata;
                 index < metadata_.size();
                 ++index) {
                metadata_[index]
                    .free_surface_snapshot_revision_key =
                    revision_key;
            }
            for (std::size_t index = first_binding;
                 index < bindings_.size();
                 ++index) {
                bindings_[index]
                    .free_surface_snapshot_revision_key =
                    revision_key;
            }
            free_surface_geometry_snapshots_.push_back(
                std::move(snapshot));
            markModified();
        } catch (...) {
            metadata_.resize(first_metadata);
            volume_rules_.resize(first_volume_rule);
            interface_rules_.resize(first_interface_rule);
            bindings_.resize(first_binding);
            sensitivity_metadata_.resize(
                first_sensitivity_metadata);
            generated_volume_markers_.resize(
                first_generated_volume_marker);
            generated_interface_markers_.resize(
                first_generated_interface_marker);
            generated_level_set_interface_markers_.resize(
                first_generated_level_set_interface_marker);
            free_surface_geometry_snapshots_.resize(
                first_snapshot);

            for (const int marker : snapshot_markers) {
                free_surface_snapshot_revision_by_marker_.erase(
                    marker);
                generated_volume_rule_indices_by_marker_.erase(
                    marker);
                generated_volume_rule_indices_by_marker_and_side_
                    .erase(marker);
                generated_interface_rule_indices_by_marker_.erase(
                    marker);
                generated_interface_two_sided_bindings_by_marker_
                    .erase(marker);
                generated_level_set_interface_provenance_by_marker_
                    .erase(marker);
                generated_active_boundary_provenance_by_marker_
                    .erase(marker);
                generated_interface_boundary_provenance_by_marker_.erase(
                    marker);
            }
            for (const auto& checkpoint :
                 physical_marker_checkpoints) {
                const auto found =
                    generated_active_boundary_markers_by_physical_marker_
                        .find(checkpoint.marker);
                if (found ==
                    generated_active_boundary_markers_by_physical_marker_
                        .end()) {
                    continue;
                }
                if (checkpoint.existed) {
                    found->second.resize(checkpoint.size);
                } else {
                    generated_active_boundary_markers_by_physical_marker_
                        .erase(found);
                }
            }
            for (const auto& checkpoint :
                 expected_source_revision_checkpoints) {
                if (checkpoint.existed) {
                    const auto found =
                        expected_source_value_revision_by_marker_
                            .find(checkpoint.marker);
                    if (found !=
                        expected_source_value_revision_by_marker_
                            .end()) {
                        found->second = checkpoint.value;
                    }
                } else {
                    expected_source_value_revision_by_marker_.erase(
                        checkpoint.marker);
                }
            }
            generated_pruned_volume_rule_count_ =
                old_generated_pruned_volume_rule_count;
            generated_pruned_volume_measure_ =
                old_generated_pruned_volume_measure;
            content_revision_ = old_content_revision;
            throw;
        }
    }

    [[nodiscard]] bool hasFreeSurfaceGeometrySnapshotForMarker(int marker) const {
        return free_surface_snapshot_revision_by_marker_.find(marker) !=
               free_surface_snapshot_revision_by_marker_.end();
    }

    [[nodiscard]] std::uint64_t freeSurfaceGeometrySnapshotRevisionForMarker(
        int marker) const {
        const auto found =
            free_surface_snapshot_revision_by_marker_.find(marker);
        if (found == free_surface_snapshot_revision_by_marker_.end()) {
            throw std::invalid_argument(
                "generated marker is not bound to an authoritative free-surface geometry snapshot");
        }
        return found->second;
    }

    [[nodiscard]] const std::vector<
        std::shared_ptr<const interfaces::FreeSurfaceGeometrySnapshot>>&
    freeSurfaceGeometrySnapshots() const noexcept {
        return free_surface_geometry_snapshots_;
    }

    void assertFreeSurfaceGeometrySnapshotCurrentForMarker(int marker) const {
        const auto bound =
            free_surface_snapshot_revision_by_marker_.find(marker);
        if (bound == free_surface_snapshot_revision_by_marker_.end()) {
            const bool inspect_all =
                free_surface_snapshot_revision_by_marker_.empty();
            const auto rule_matches_marker =
                [marker, inspect_all](const geometry::CutQuadratureRule& rule) {
                    return inspect_all || rule.provenance.marker == marker;
                };
            for (std::size_t index = 0u; index < volume_rules_.size();
                 ++index) {
                if (!rule_matches_marker(volume_rules_[index])) {
                    continue;
                }
                const bool keyed_rule =
                    volume_rules_[index]
                        .provenance.free_surface_snapshot_revision_key != 0u;
                const bool keyed_metadata =
                    index < metadata_.size() &&
                    metadata_[index].free_surface_snapshot_revision_key != 0u;
                const bool keyed_binding =
                    index < bindings_.size() &&
                    bindings_[index]
                            .free_surface_snapshot_revision_key != 0u;
                if (keyed_rule || keyed_metadata || keyed_binding) {
                    throw std::invalid_argument(
                        "cut-volume data declares a free-surface snapshot revision without an authoritative snapshot");
                }
            }
            for (const auto& rule : interface_rules_) {
                if (rule_matches_marker(rule) &&
                    rule.provenance.free_surface_snapshot_revision_key != 0u) {
                    throw std::invalid_argument(
                        "cut-interface data declares a free-surface snapshot revision without an authoritative snapshot");
                }
            }
            for (const auto& binding : bindings_) {
                if ((inspect_all || binding.marker == marker) &&
                    binding.free_surface_snapshot_revision_key != 0u) {
                    throw std::invalid_argument(
                        "cut-integration binding declares a free-surface snapshot revision without an authoritative snapshot");
                }
            }
            return;
        }
        const interfaces::FreeSurfaceGeometrySnapshot* snapshot = nullptr;
        for (const auto& candidate : free_surface_geometry_snapshots_) {
            if (candidate &&
                candidate->revision().snapshot_revision_key == bound->second) {
                snapshot = candidate.get();
                break;
            }
        }
        if (snapshot == nullptr || !snapshot->revision().complete()) {
            throw std::invalid_argument(
                "generated marker references an unavailable free-surface geometry snapshot");
        }
        const auto expected_source_revision =
            snapshot->revision().source_value_revision;
        const auto current_source_revision =
            expected_source_value_revision_by_marker_.find(marker);
        if (current_source_revision !=
                expected_source_value_revision_by_marker_.end() &&
            (current_source_revision->second == 0u ||
             current_source_revision->second != expected_source_revision)) {
            throw std::invalid_argument(
                "free-surface geometry snapshot revision does not match the current source value revision");
        }
        const auto interface_rules =
            generated_interface_rule_indices_by_marker_.find(marker);
        const auto validate_interface_rule =
            [&](std::size_t index) {
                const bool registered =
                    interface_rules !=
                        generated_interface_rule_indices_by_marker_.end() &&
                    std::find(interface_rules->second.begin(),
                              interface_rules->second.end(),
                              index) != interface_rules->second.end();
                if (!registered || index >= interface_rules_.size() ||
                    interface_rules_[index].provenance.marker != marker ||
                    interface_rules_[index].provenance.source_value_revision !=
                        expected_source_revision ||
                    interface_rules_[index]
                            .provenance
                            .free_surface_snapshot_revision_key !=
                        bound->second) {
                    throw std::invalid_argument(
                        "generated interface rule does not declare its complete free-surface snapshot revision");
                }
            };
        if (interface_rules !=
            generated_interface_rule_indices_by_marker_.end()) {
            for (const auto index : interface_rules->second) {
                validate_interface_rule(index);
            }
        }
        for (std::size_t index = 0u; index < interface_rules_.size(); ++index) {
            if (interface_rules_[index].provenance.marker == marker) {
                validate_interface_rule(index);
            }
        }
        const auto volume_rules =
            generated_volume_rule_indices_by_marker_.find(marker);
        const auto validate_volume_rule =
            [&](std::size_t index) {
                const bool registered =
                    volume_rules !=
                        generated_volume_rule_indices_by_marker_.end() &&
                    std::find(volume_rules->second.begin(),
                              volume_rules->second.end(),
                              index) != volume_rules->second.end();
                if (!registered || index >= volume_rules_.size() ||
                    volume_rules_[index].provenance.marker != marker ||
                    volume_rules_[index].provenance.source_value_revision !=
                        expected_source_revision ||
                    volume_rules_[index]
                            .provenance
                            .free_surface_snapshot_revision_key !=
                        bound->second ||
                    index >= metadata_.size() ||
                    metadata_[index].source_value_revision !=
                        expected_source_revision ||
                    metadata_[index]
                            .free_surface_snapshot_revision_key !=
                        bound->second ||
                    index >= bindings_.size() ||
                    bindings_[index].marker != marker ||
                    bindings_[index].source_value_revision !=
                        expected_source_revision ||
                    bindings_[index]
                            .free_surface_snapshot_revision_key !=
                        bound->second) {
                    throw std::invalid_argument(
                        "generated volume rule, metadata, or binding does not declare its complete free-surface snapshot revision");
                }
            };
        if (volume_rules != generated_volume_rule_indices_by_marker_.end()) {
            for (const auto index : volume_rules->second) {
                validate_volume_rule(index);
            }
        }
        for (std::size_t index = 0u; index < volume_rules_.size(); ++index) {
            if (volume_rules_[index].provenance.marker == marker) {
                validate_volume_rule(index);
            }
        }
        for (const auto& binding : bindings_) {
            if (binding.marker == marker &&
                (binding.source_value_revision != expected_source_revision ||
                 binding.free_surface_snapshot_revision_key != bound->second)) {
                throw std::invalid_argument(
                    "generated cut-integration binding does not declare its complete free-surface snapshot revision");
            }
        }
    }

    void assertAllFreeSurfaceGeometrySnapshotsCurrent() const {
        for (const auto& [marker, revision_key] :
             free_surface_snapshot_revision_by_marker_) {
            (void)revision_key;
            assertFreeSurfaceGeometrySnapshotCurrentForMarker(marker);
        }

        const auto marker_is_bound = [&](int marker) {
            return marker >= 0 &&
                   free_surface_snapshot_revision_by_marker_.find(marker) !=
                       free_surface_snapshot_revision_by_marker_.end();
        };
        for (std::size_t index = 0u; index < volume_rules_.size(); ++index) {
            const bool rule_is_keyed =
                volume_rules_[index]
                    .provenance.free_surface_snapshot_revision_key != 0u;
            const bool metadata_is_keyed =
                index < metadata_.size() &&
                metadata_[index].free_surface_snapshot_revision_key != 0u;
            const bool binding_is_keyed =
                index < bindings_.size() &&
                bindings_[index].free_surface_snapshot_revision_key != 0u;
            if ((rule_is_keyed || metadata_is_keyed || binding_is_keyed) &&
                !marker_is_bound(volume_rules_[index].provenance.marker)) {
                throw std::invalid_argument(
                    "cut-volume data declares an orphan free-surface snapshot revision");
            }
        }
        for (std::size_t index = volume_rules_.size();
             index < metadata_.size(); ++index) {
            if (metadata_[index].free_surface_snapshot_revision_key != 0u) {
                throw std::invalid_argument(
                    "cut-volume metadata declares an orphan free-surface snapshot revision");
            }
        }
        for (const auto& rule : interface_rules_) {
            if (rule.provenance.free_surface_snapshot_revision_key != 0u &&
                !marker_is_bound(rule.provenance.marker)) {
                throw std::invalid_argument(
                    "cut-interface data declares an orphan free-surface snapshot revision");
            }
        }
        for (std::size_t index = 0u; index < bindings_.size(); ++index) {
            const auto& binding = bindings_[index];
            if (binding.free_surface_snapshot_revision_key == 0u) {
                continue;
            }
            const bool aligned =
                index < volume_rules_.size() &&
                binding.marker == volume_rules_[index].provenance.marker;
            if (!aligned || !marker_is_bound(binding.marker)) {
                throw std::invalid_argument(
                    "cut-integration binding declares an orphan free-surface snapshot revision");
            }
        }
    }

    template <typename MeshAccessLike>
    void assertAllFreeSurfaceGeometrySnapshotsCurrent(
        const MeshAccessLike& mesh) const {
        assertAllFreeSurfaceGeometrySnapshotsCurrent();
        if (!mesh.revisionTrackingAvailable()) {
            return;
        }
        for (const auto& snapshot : free_surface_geometry_snapshots_) {
            if (!snapshot) {
                throw std::invalid_argument(
                    "free-surface geometry snapshot storage is incomplete");
            }
            const auto& revision = snapshot->localMeshRevision();
            if (revision.mesh_geometry_revision != mesh.geometryRevision() ||
                revision.mesh_topology_revision != mesh.topologyRevision() ||
                revision.ownership_revision != mesh.ownershipRevision() ||
                revision.numbering_revision != mesh.numberingRevision()) {
                throw std::invalid_argument(
                    "free-surface geometry snapshot does not match the current mesh revision");
            }
        }
    }

    void addKinematicData(EmbeddedBoundaryKinematicData data) {
        kinematic_data_.push_back(std::move(data));
        markModified();
    }

    void addStabilizationHook(CutStabilizationHook hook) {
        stabilization_hooks_.push_back(std::move(hook));
        markModified();
    }

    void addBinding(CutIntegrationBinding binding) {
        bindings_.push_back(std::move(binding));
        markModified();
    }

    void addSensitivityMetadata(CutGeometrySensitivityMetadata metadata) {
        sensitivity_metadata_.push_back(std::move(metadata));
        markModified();
    }

    void setExpectedGeneratedSourceValueRevision(int marker,
                                                 std::uint64_t revision) {
        if (marker < 0) {
            throw std::invalid_argument(
                "generated cut-volume source revision requires a nonnegative marker");
        }
        bool changed = false;
        const auto bound =
            free_surface_snapshot_revision_by_marker_.find(marker);
        if (bound == free_surface_snapshot_revision_by_marker_.end()) {
            auto& stored_revision =
                expected_source_value_revision_by_marker_[marker];
            if (stored_revision != revision) {
                stored_revision = revision;
                changed = true;
            }
        } else {
            for (const auto& [bound_marker, snapshot_revision] :
                 free_surface_snapshot_revision_by_marker_) {
                if (snapshot_revision != bound->second) {
                    continue;
                }
                auto& stored_revision =
                    expected_source_value_revision_by_marker_[bound_marker];
                if (stored_revision != revision) {
                    stored_revision = revision;
                    changed = true;
                }
            }
        }
        if (changed) {
            markModified();
        }
    }

    [[nodiscard]] bool hasExpectedGeneratedSourceValueRevision(int marker) const {
        return expected_source_value_revision_by_marker_.find(marker) !=
               expected_source_value_revision_by_marker_.end();
    }

    [[nodiscard]] std::uint64_t expectedGeneratedSourceValueRevision(int marker) const {
        const auto it = expected_source_value_revision_by_marker_.find(marker);
        return it == expected_source_value_revision_by_marker_.end() ? 0u : it->second;
    }

    void assertGeneratedInterfaceRulesCurrentForMarker(int marker) const {
        assertFreeSurfaceGeometrySnapshotCurrentForMarker(marker);
        const auto expected_it = expected_source_value_revision_by_marker_.find(marker);
        if (expected_it == expected_source_value_revision_by_marker_.end()) {
            return;
        }
        const auto rule_it = generated_interface_rule_indices_by_marker_.find(marker);
        if (rule_it == generated_interface_rule_indices_by_marker_.end()) {
            return;
        }
        for (const auto index : rule_it->second) {
            if (index >= interface_rules_.size()) {
                throw std::invalid_argument(
                    "generated cut-interface rule is missing source revision metadata");
            }
            const auto actual =
                interface_rules_[index].provenance.source_value_revision;
            if (actual == 0u || actual != expected_it->second) {
                throw std::invalid_argument(
                    "generated cut-interface rule revision does not match the current source value revision");
            }
        }
    }

    void assertGeneratedVolumeRulesCurrentForMarkerAndSide(
        int marker,
        geometry::CutIntegrationSide side) const {
        assertFreeSurfaceGeometrySnapshotCurrentForMarker(marker);
        if (side == geometry::CutIntegrationSide::Interface) {
            return;
        }
        const auto expected_it = expected_source_value_revision_by_marker_.find(marker);
        if (expected_it == expected_source_value_revision_by_marker_.end()) {
            return;
        }
        const auto rule_it =
            generated_volume_rule_indices_by_marker_and_side_.find(marker);
        if (rule_it == generated_volume_rule_indices_by_marker_and_side_.end()) {
            return;
        }
        const auto& indices = rule_it->second[volumeSideIndex(side)].indices;
        for (const auto index : indices) {
            if (index >= metadata_.size()) {
                throw std::invalid_argument(
                    "generated cut-volume rule is missing source revision metadata");
            }
            const auto actual = metadata_[index].source_value_revision;
            const auto rule_revision =
                index < volume_rules_.size()
                    ? volume_rules_[index].provenance.source_value_revision
                    : 0u;
            if (actual == 0u || actual != expected_it->second ||
                rule_revision == 0u || rule_revision != expected_it->second) {
                throw std::invalid_argument(
                    "generated cut-volume rule revision does not match the current source value revision");
            }
        }
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

    [[nodiscard]] const std::vector<interfaces::GeneratedInterfaceTwoSidedBinding>&
    generatedInterfaceTwoSidedBindingsForMarker(int marker) const noexcept {
        static const std::vector<interfaces::GeneratedInterfaceTwoSidedBinding> empty{};
        const auto it = generated_interface_two_sided_bindings_by_marker_.find(marker);
        return it == generated_interface_two_sided_bindings_by_marker_.end()
                   ? empty
                   : it->second;
    }

    [[nodiscard]] const interfaces::GeneratedInterfaceTwoSidedBinding*
    twoSidedBindingForInterfaceRule(const geometry::CutQuadratureRule& rule) const noexcept {
        const int marker = rule.provenance.marker;
        const auto it = generated_interface_two_sided_bindings_by_marker_.find(marker);
        if (it == generated_interface_two_sided_bindings_by_marker_.end()) {
            return nullptr;
        }
        for (const auto& binding : it->second) {
            if (binding.parent_cell == rule.provenance.parent_entity &&
                binding.interface_stable_id == rule.provenance.cut_topology_revision) {
                return &binding;
            }
        }
        return nullptr;
    }

    [[nodiscard]] const std::vector<geometry::CutQuadratureRule>& facetSetRules() const noexcept {
        return facet_set_rules_;
    }

    [[nodiscard]] bool hasGeneratedVolumeMarker(int marker) const {
        return generated_volume_rule_indices_by_marker_.find(marker) !=
               generated_volume_rule_indices_by_marker_.end();
    }

    [[nodiscard]] const std::vector<int>& generatedVolumeMarkers() const noexcept {
        return generated_volume_markers_;
    }

    [[nodiscard]] bool hasGeneratedInterfaceMarker(int marker) const {
        return generated_interface_rule_indices_by_marker_.find(marker) !=
               generated_interface_rule_indices_by_marker_.end();
    }

    [[nodiscard]] bool hasGeneratedLevelSetInterfaceMarker(
        int marker) const noexcept {
        return std::find(
                   generated_level_set_interface_markers_.begin(),
                   generated_level_set_interface_markers_.end(),
                   marker) !=
               generated_level_set_interface_markers_.end();
    }

    [[nodiscard]] std::span<const int>
    generatedLevelSetInterfaceMarkers() const noexcept {
        return generated_level_set_interface_markers_;
    }

    [[nodiscard]] const GeneratedLevelSetInterfacePublicationProvenance*
    findGeneratedLevelSetInterfacePublicationProvenance(
        int marker) const noexcept {
        const auto found =
            generated_level_set_interface_provenance_by_marker_.find(
                marker);
        return found ==
                   generated_level_set_interface_provenance_by_marker_.end()
            ? nullptr
            : &found->second;
    }

    [[nodiscard]] const std::string*
    findGeneratedInterfaceBoundaryMarkerKey(
        int marker) const noexcept {
        const auto found =
            generated_interface_boundary_provenance_by_marker_.find(
                marker);
        return found ==
                   generated_interface_boundary_provenance_by_marker_.end()
            ? nullptr
            : &found->second.stable_owner_key;
    }

    [[nodiscard]] const GeneratedInterfaceBoundaryPublicationProvenance*
    findGeneratedInterfaceBoundaryPublicationProvenance(
        int marker) const noexcept {
        const auto found =
            generated_interface_boundary_provenance_by_marker_.find(
                marker);
        return found ==
                   generated_interface_boundary_provenance_by_marker_.end()
            ? nullptr
            : &found->second;
    }

    [[nodiscard]] bool hasGeneratedActiveBoundaryMarker(int marker) const noexcept {
        return generated_active_boundary_provenance_by_marker_.contains(
            marker);
    }

    [[nodiscard]] const GeneratedActiveBoundaryProvenance*
    findGeneratedActiveBoundaryProvenance(int marker) const noexcept {
        const auto found =
            generated_active_boundary_provenance_by_marker_.find(marker);
        return found == generated_active_boundary_provenance_by_marker_.end()
            ? nullptr
            : &found->second;
    }

    [[nodiscard]] std::span<const int>
    generatedActiveBoundaryMarkersForPhysicalBoundary(
        int physical_boundary_marker) const noexcept {
        const auto found =
            generated_active_boundary_markers_by_physical_marker_.find(
                physical_boundary_marker);
        if (found ==
            generated_active_boundary_markers_by_physical_marker_.end()) {
            return {};
        }
        return std::span<const int>(
            found->second.data(), found->second.size());
    }

    [[nodiscard]] const std::vector<int>& generatedInterfaceMarkers() const noexcept {
        return generated_interface_markers_;
    }

    [[nodiscard]] std::size_t generatedPrunedVolumeRuleCount() const noexcept {
        return generated_pruned_volume_rule_count_;
    }

    [[nodiscard]] Real generatedPrunedVolumeMeasure() const noexcept {
        return generated_pruned_volume_measure_;
    }

    [[nodiscard]] bool hasFacetSetMarker(int marker) const {
        return facet_set_rule_indices_by_marker_.find(marker) !=
               facet_set_rule_indices_by_marker_.end();
    }

    [[nodiscard]] const std::vector<int>& facetSetMarkers() const noexcept {
        return facet_set_markers_;
    }

    [[nodiscard]] bool hasFacetSetHandleMarker(int marker) const {
        return facet_set_handle_indices_by_marker_.find(marker) !=
               facet_set_handle_indices_by_marker_.end();
    }

    [[nodiscard]] const std::vector<CutFacetSetHandle>& facetSetHandles() const noexcept {
        return facet_set_handles_;
    }

    [[nodiscard]] const CutFacetSetHandle* facetSetHandleForMarker(int marker) const noexcept {
        const auto it = facet_set_handle_indices_by_marker_.find(marker);
        if (it == facet_set_handle_indices_by_marker_.end() ||
            it->second >= facet_set_handles_.size()) {
            return nullptr;
        }
        return &facet_set_handles_[it->second];
    }

    [[nodiscard]] std::vector<const geometry::CutQuadratureRule*>
    interfaceRulesForMarker(int marker) const {
        std::vector<const geometry::CutQuadratureRule*> rules;
        assertGeneratedInterfaceRulesCurrentForMarker(marker);
        const auto it = generated_interface_rule_indices_by_marker_.find(marker);
        if (it == generated_interface_rule_indices_by_marker_.end()) {
            return rules;
        }
        rules.reserve(it->second.size());
        for (const auto index : it->second) {
            if (index < interface_rules_.size()) {
                rules.push_back(&interface_rules_[index]);
            }
        }
        return rules;
    }

    /**
     * Nonallocating generated-interface rule lookup for capped consumers.
     *
     * The returned indices address `interfaceRules()` and remain valid until
     * this context is modified. A registered marker with no rules returns an
     * empty span.
     */
    [[nodiscard]] std::span<const std::size_t>
    generatedInterfaceRuleIndexSpanForMarker(int marker) const {
        assertGeneratedInterfaceRulesCurrentForMarker(marker);
        const auto it =
            generated_interface_rule_indices_by_marker_.find(marker);
        if (it ==
            generated_interface_rule_indices_by_marker_.end()) {
            return {};
        }
        return std::span<const std::size_t>(
            it->second.data(), it->second.size());
    }

    [[nodiscard]] std::vector<std::size_t>
    generatedVolumeRuleIndicesForMarkerAndSide(int marker,
                                               geometry::CutIntegrationSide side) const {
        const auto span =
            generatedVolumeRuleIndexSpanForMarkerAndSide(marker, side);
        return std::vector<std::size_t>(span.begin(), span.end());
    }

    [[nodiscard]] std::span<const std::size_t>
    generatedVolumeRuleIndexSpanForMarkerAndSide(
        int marker,
        geometry::CutIntegrationSide side) const {
        if (side == geometry::CutIntegrationSide::Interface) {
            return {};
        }
        assertGeneratedVolumeRulesCurrentForMarkerAndSide(marker, side);
        const auto it = generated_volume_rule_indices_by_marker_and_side_.find(marker);
        if (it == generated_volume_rule_indices_by_marker_and_side_.end()) {
            return {};
        }
        const auto& indices = it->second[volumeSideIndex(side)].indices;
        return std::span<const std::size_t>(indices.data(), indices.size());
    }

    [[nodiscard]] GeneratedVolumeRuleDiagnostics
    generatedVolumeDiagnosticsForMarkerAndSide(
        int marker,
        geometry::CutIntegrationSide side) const {
        GeneratedVolumeRuleDiagnostics diagnostics;
        if (side == geometry::CutIntegrationSide::Interface) {
            diagnostics.normalizeEmptyExtrema();
            return diagnostics;
        }
        assertGeneratedVolumeRulesCurrentForMarkerAndSide(marker, side);
        const auto it = generated_volume_rule_indices_by_marker_and_side_.find(marker);
        if (it != generated_volume_rule_indices_by_marker_and_side_.end()) {
            diagnostics = it->second[volumeSideIndex(side)].diagnostics;
        }
        diagnostics.normalizeEmptyExtrema();
        return diagnostics;
    }

    [[nodiscard]] std::vector<const geometry::CutQuadratureRule*>
    generatedVolumeRulesForMarker(int marker) const {
        std::vector<const geometry::CutQuadratureRule*> rules;
        assertGeneratedVolumeRulesCurrentForMarkerAndSide(
            marker, geometry::CutIntegrationSide::Negative);
        assertGeneratedVolumeRulesCurrentForMarkerAndSide(
            marker, geometry::CutIntegrationSide::Positive);
        const auto it = generated_volume_rule_indices_by_marker_.find(marker);
        if (it == generated_volume_rule_indices_by_marker_.end()) {
            return rules;
        }
        rules.reserve(it->second.size());
        for (const auto index : it->second) {
            if (index < volume_rules_.size()) {
                rules.push_back(&volume_rules_[index]);
            }
        }
        return rules;
    }

    [[nodiscard]] std::vector<const geometry::CutQuadratureRule*>
    generatedVolumeRulesForMarkerAndSide(int marker,
                                         geometry::CutIntegrationSide side) const {
        std::vector<const geometry::CutQuadratureRule*> rules;
        if (side == geometry::CutIntegrationSide::Interface) {
            return rules;
        }
        assertGeneratedVolumeRulesCurrentForMarkerAndSide(marker, side);
        const auto it = generated_volume_rule_indices_by_marker_and_side_.find(marker);
        if (it == generated_volume_rule_indices_by_marker_and_side_.end()) {
            return rules;
        }
        const auto& indices = it->second[volumeSideIndex(side)].indices;
        rules.reserve(indices.size());
        for (const auto index : indices) {
            if (index < volume_rules_.size()) {
                rules.push_back(&volume_rules_[index]);
            }
        }
        return rules;
    }

    [[nodiscard]] std::vector<const CutCellAssemblyMetadata*>
    generatedVolumeMetadataForMarkerAndSide(int marker,
                                            geometry::CutIntegrationSide side) const {
        std::vector<const CutCellAssemblyMetadata*> metadata;
        if (side == geometry::CutIntegrationSide::Interface) {
            return metadata;
        }
        assertGeneratedVolumeRulesCurrentForMarkerAndSide(marker, side);
        const auto it = generated_volume_rule_indices_by_marker_and_side_.find(marker);
        if (it == generated_volume_rule_indices_by_marker_and_side_.end()) {
            return metadata;
        }
        const auto& indices = it->second[volumeSideIndex(side)].indices;
        metadata.reserve(indices.size());
        for (const auto index : indices) {
            if (index < metadata_.size()) {
                metadata.push_back(&metadata_[index]);
            }
        }
        return metadata;
    }

    void bindFacetStabilizationScalesForMarkerAndSide(
        CutFacetSetHandle& handle,
        int marker,
        geometry::CutIntegrationSide side) const {
        bindFacetStabilizationScales(
            handle, generatedVolumeMetadataForMarkerAndSide(marker, side));
    }

    [[nodiscard]] std::vector<const geometry::CutQuadratureRule*>
    facetSetRulesForMarker(int marker) const {
        std::vector<const geometry::CutQuadratureRule*> rules;
        const auto it = facet_set_rule_indices_by_marker_.find(marker);
        if (it == facet_set_rule_indices_by_marker_.end()) {
            return rules;
        }
        rules.reserve(it->second.size());
        for (const auto index : it->second) {
            if (index < facet_set_rules_.size()) {
                rules.push_back(&facet_set_rules_[index]);
            }
        }
        return rules;
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

    [[nodiscard]] const std::vector<CutGeometrySensitivityMetadata>& sensitivityMetadata() const noexcept {
        return sensitivity_metadata_;
    }

    template <typename VolumeIntegrand, typename InterfaceIntegrand>
    [[nodiscard]] CutScalarOperatorEvaluation evaluateScalarCutOperator(
        CutIntegrationAssemblyPath path,
        VolumeIntegrand&& volume_integrand,
        InterfaceIntegrand&& interface_integrand) const {
        CutScalarOperatorEvaluation evaluation;
        evaluation.path = path;
        const auto record_fixed_geometry_diagnostic =
            [](CutFixedGeometryAssemblyDiagnostics& diagnostic,
               const geometry::CutQuadratureRule& rule,
               bool interface_rule) {
                if (rule.provenance.implicit_geometry_mode != "HighOrderImplicit") {
                    return;
                }
                if (interface_rule) {
                    ++diagnostic.high_order_interface_rule_count;
                } else {
                    ++diagnostic.high_order_volume_rule_count;
                }
                diagnostic.quadrature_policy_key_xor ^=
                    rule.provenance.predicate_policy_key;
                if (rule.provenance.geometry_tangent_policy ==
                    "RefreshedFrozenQuadrature") {
                    ++diagnostic.refreshed_frozen_quadrature_rule_count;
                } else if (rule.provenance.geometry_tangent_policy ==
                           "DifferentiatedQuadrature") {
                    ++diagnostic.differentiated_quadrature_rule_count;
                } else if (rule.provenance.geometry_tangent_policy.empty()) {
                    ++diagnostic.missing_tangent_policy_rule_count;
                }
            };

        std::unordered_map<MeshIndex, Real> parent_measures;
        const bool has_explicit_bindings = !bindings_.empty();
        for (std::size_t i = 0; i < volume_rules_.size(); ++i) {
            const CutIntegrationBinding* binding =
                has_explicit_bindings && i < bindings_.size() ? &bindings_[i] : nullptr;
            if (binding != nullptr && !bindingVisibleToPath(*binding, path)) {
                continue;
            }

            const auto& rule = volume_rules_[i];
            record_fixed_geometry_diagnostic(
                evaluation.fixed_geometry_diagnostics,
                rule,
                /*interface_rule=*/false);
            const CutCellAssemblyMetadata* metadata =
                i < metadata_.size() ? &metadata_[i] : nullptr;
            const auto parent_entity =
                binding != nullptr
                    ? binding->parent_entity
                    : (metadata != nullptr ? metadata->parent_entity
                                           : rule.provenance.parent_entity);
            auto& parent_measure = parent_measures[parent_entity];
            parent_measure = std::max(parent_measure, rule.parent_measure);

            ++evaluation.volume_rule_count;
            if (rule.side == geometry::CutIntegrationSide::Negative) {
                evaluation.negative_volume_measure += rule.measure;
            } else if (rule.side == geometry::CutIntegrationSide::Positive) {
                evaluation.positive_volume_measure += rule.measure;
            }

            for (const auto& qp : rule.points) {
                ++evaluation.volume_point_count;
                CutScalarOperatorPoint point;
                point.kind = rule.kind;
                point.side = rule.side;
                point.parent_entity = parent_entity;
                point.point = qp.point;
                point.normal = qp.normal;
                point.weight = qp.weight;
                point.volume_fraction =
                    metadata != nullptr ? metadata->volume_fraction : rule.volume_fraction;
                point.cut_topology_revision =
                    binding != nullptr ? binding->cut_topology_revision
                                       : rule.provenance.cut_topology_revision;
                point.quadrature_policy_key =
                    binding != nullptr
                        ? binding->quadrature_policy_key
                        : rule.provenance.predicate_policy_key;
                point.construction = rule.policy.kind;
                point.frame = rule.frame;

                const Real contribution =
                    qp.weight * static_cast<Real>(volume_integrand(point));
                if (rule.side == geometry::CutIntegrationSide::Negative) {
                    evaluation.negative_volume_integral += contribution;
                } else if (rule.side == geometry::CutIntegrationSide::Positive) {
                    evaluation.positive_volume_integral += contribution;
                }
            }
        }

        for (const auto& entry : parent_measures) {
            evaluation.parent_measure += entry.second;
        }

        for (const auto& rule : interface_rules_) {
            record_fixed_geometry_diagnostic(
                evaluation.fixed_geometry_diagnostics,
                rule,
                /*interface_rule=*/true);
            ++evaluation.interface_rule_count;
            evaluation.interface_measure += rule.measure;
            for (const auto& qp : rule.points) {
                ++evaluation.interface_point_count;
                CutScalarOperatorPoint point;
                point.kind = rule.kind;
                point.side = geometry::CutIntegrationSide::Interface;
                point.parent_entity = rule.provenance.parent_entity;
                point.point = qp.point;
                point.normal = qp.normal;
                point.weight = qp.weight;
                point.volume_fraction = rule.volume_fraction;
                point.cut_topology_revision = rule.provenance.cut_topology_revision;
                point.quadrature_policy_key =
                    rule.provenance.predicate_policy_key;
                point.construction = rule.policy.kind;
                point.frame = rule.frame;
                evaluation.interface_integral +=
                    qp.weight * static_cast<Real>(interface_integrand(point));
            }
        }

        return evaluation;
    }

    template <typename FacetIntegrand>
    [[nodiscard]] CutFacetSetOperatorEvaluation evaluateScalarFacetSetOperator(
        int marker,
        CutIntegrationAssemblyPath path,
        FacetIntegrand&& integrand) const {
        CutFacetSetOperatorEvaluation evaluation;
        evaluation.path = path;
        evaluation.marker = marker;

        const auto it = facet_set_rule_indices_by_marker_.find(marker);
        if (it == facet_set_rule_indices_by_marker_.end()) {
            return evaluation;
        }

        for (const auto index : it->second) {
            if (index >= facet_set_rules_.size()) {
                continue;
            }
            const auto& rule = facet_set_rules_[index];
            ++evaluation.facet_rule_count;
            evaluation.measure += rule.measure;
            for (const auto& qp : rule.points) {
                ++evaluation.facet_point_count;
                CutScalarOperatorPoint point;
                point.kind = rule.kind;
                point.side = rule.side;
                point.parent_entity = rule.provenance.parent_entity;
                point.point = qp.point;
                point.normal = qp.normal;
                point.weight = qp.weight;
                point.volume_fraction = rule.volume_fraction;
                point.cut_topology_revision = rule.provenance.cut_topology_revision;
                point.quadrature_policy_key = rule.provenance.predicate_policy_key;
                point.construction = rule.policy.kind;
                point.frame = rule.frame;
                evaluation.integral +=
                    qp.weight * static_cast<Real>(integrand(point));
            }
        }

        return evaluation;
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
            binding.source_value_revision = map.revision_key();
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
                    geometry::CutQuadraturePoint qp;
                    qp.point = {{
                        static_cast<Real>(patch.quadrature_points[i][0]),
                        static_cast<Real>(patch.quadrature_points[i][1]),
                        static_cast<Real>(patch.quadrature_points[i][2])}};
                    qp.normal = {{
                        static_cast<Real>(patch.quadrature_normals[i][0]),
                        static_cast<Real>(patch.quadrature_normals[i][1]),
                        static_cast<Real>(patch.quadrature_normals[i][2])}};
                    qp.weight = static_cast<Real>(patch.quadrature_weights[i]);
                    curved_points.push_back(qp);
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
            binding.source_value_revision = topology.topology_revision;
            binding.visible_to_paths = {
                CutIntegrationAssemblyPath::Standard,
                CutIntegrationAssemblyPath::MatrixFree,
                CutIntegrationAssemblyPath::Interpreter,
                CutIntegrationAssemblyPath::AD,
                CutIntegrationAssemblyPath::SymbolicTangent,
                CutIntegrationAssemblyPath::JIT};
            addBinding(std::move(binding));
        }

        for (const auto& record : topology.sensitivity_records) {
            CutGeometrySensitivityMetadata metadata;
            metadata.parent_entity = static_cast<MeshIndex>(record.parent_cell);
            metadata.target_kind = record.target_kind;
            metadata.construction_policy = record.construction_policy;
            metadata.provenance_id = record.provenance.persistent_id;
            metadata.source_stable_id = record.source_stable_id;
            metadata.cut_topology_revision = topology.topology_revision;
            metadata.quadrature_policy_key = quadrature_policy_key;
            metadata.ad_compatible = record.ad_compatible;
            metadata.location_sensitivity_available = record.location_sensitivity_available;
            metadata.jacobian_sensitivity_available = record.jacobian_sensitivity_available;
            metadata.measure_sensitivity_available = record.measure_sensitivity_available;
            metadata.normal_sensitivity_available = record.normal_sensitivity_available;
            metadata.quadrature_weight_sensitivity_available =
                record.quadrature_weight_sensitivity_available;
            metadata.parent_geometry_dofs.reserve(record.parent_geometry_dofs.size());
            for (const auto dof : record.parent_geometry_dofs) {
                metadata.parent_geometry_dofs.push_back(static_cast<MeshIndex>(dof));
            }
            metadata.samples.reserve(record.samples.size());
            for (const auto& sample : record.samples) {
                CutGeometrySensitivitySampleMetadata sample_metadata;
                sample_metadata.parent_parametric_coordinate = {{
                    static_cast<Real>(sample.parent_parametric_coordinate[0]),
                    static_cast<Real>(sample.parent_parametric_coordinate[1]),
                    static_cast<Real>(sample.parent_parametric_coordinate[2])}};
                sample_metadata.influencing_parent_geometry_dofs =
                    metadata.parent_geometry_dofs;
                sample_metadata.shape_values.reserve(sample.shape_values.size());
                for (const auto value : sample.shape_values) {
                    sample_metadata.shape_values.push_back(static_cast<Real>(value));
                }
                sample_metadata.shape_gradients.reserve(sample.shape_gradients.size());
                for (const auto& gradient : sample.shape_gradients) {
                    sample_metadata.shape_gradients.push_back({{
                        static_cast<Real>(gradient[0]),
                        static_cast<Real>(gradient[1]),
                        static_cast<Real>(gradient[2])}});
                }
                metadata.samples.push_back(std::move(sample_metadata));
            }
            metadata.visible_to_paths = {
                CutIntegrationAssemblyPath::Standard,
                CutIntegrationAssemblyPath::MatrixFree,
                CutIntegrationAssemblyPath::Interpreter,
                CutIntegrationAssemblyPath::AD,
                CutIntegrationAssemblyPath::SymbolicTangent,
                CutIntegrationAssemblyPath::JIT};
            addSensitivityMetadata(std::move(metadata));
        }
    }
#endif

private:
    [[nodiscard]] static std::uint64_t facetSetStableId(
        int marker,
        const std::vector<MeshIndex>& facets) noexcept {
        std::uint64_t h = 1469598103934665603ull;
        const auto mix = [&h](std::uint64_t value) noexcept {
            h ^= value;
            h *= 1099511628211ull;
        };
        mix(static_cast<std::uint64_t>(static_cast<std::int64_t>(marker)));
        for (const auto facet : facets) {
            mix(static_cast<std::uint64_t>(static_cast<std::int64_t>(facet)));
        }
        return h;
    }

    static void normalizeFacetSetHandle(CutFacetSetHandle& handle) {
        handle.facets.erase(std::remove_if(handle.facets.begin(),
                                           handle.facets.end(),
                                           [](MeshIndex facet) {
                                               return facet < static_cast<MeshIndex>(0);
                                           }),
                            handle.facets.end());
        for (const auto& metadata : handle.facet_metadata) {
            if (metadata.facet >= static_cast<MeshIndex>(0)) {
                handle.facets.push_back(metadata.facet);
            }
        }

        std::sort(handle.facets.begin(), handle.facets.end());
        handle.facets.erase(std::unique(handle.facets.begin(), handle.facets.end()),
                            handle.facets.end());

        handle.facet_metadata.erase(
            std::remove_if(handle.facet_metadata.begin(),
                           handle.facet_metadata.end(),
                           [](const CutFacetSetFacetMetadata& metadata) {
                               return metadata.facet < static_cast<MeshIndex>(0);
                           }),
            handle.facet_metadata.end());
        std::sort(handle.facet_metadata.begin(),
                  handle.facet_metadata.end(),
                  [](const CutFacetSetFacetMetadata& a,
                     const CutFacetSetFacetMetadata& b) {
                      if (a.facet != b.facet) {
                          return a.facet < b.facet;
                      }
                      if (a.first_cell != b.first_cell) {
                          return a.first_cell < b.first_cell;
                      }
                      return a.second_cell < b.second_cell;
                  });

        std::vector<CutFacetSetFacetMetadata> unique_metadata;
        unique_metadata.reserve(handle.facet_metadata.size());
        for (const auto& metadata : handle.facet_metadata) {
            if (!unique_metadata.empty() &&
                unique_metadata.back().facet == metadata.facet) {
                auto& merged = unique_metadata.back();
                merged.stabilization_scale =
                    std::max(merged.stabilization_scale, metadata.stabilization_scale);
                if (merged.stable_id == 0u) {
                    merged.stable_id = metadata.stable_id;
                }
                continue;
            }
            unique_metadata.push_back(metadata);
        }
        handle.facet_metadata = std::move(unique_metadata);
    }

    [[nodiscard]] static std::unordered_map<MeshIndex, Real>
    buildCutCellStabilizationScales(
        const std::vector<CutCellAssemblyMetadata>& metadata) {
        // Generated cut backends emit many subdivision-leaf rules per cut
        // cell, so a single entry's volume fraction says nothing about the
        // cell's cut support. Aggregate fractions per (parent, side) first;
        // the stabilization scale reflects the smallest per-side aggregate
        // support of the cell, never an individual subdivision leaf.
        constexpr Real fraction_floor = Real{1.0e-12};
        constexpr Real full_fraction_tol = Real{1.0e-12};
        std::unordered_map<MeshIndex, std::array<Real, 2>> side_fractions;
        for (const auto& entry : metadata) {
            const MeshIndex parent =
                entry.parent_entity >= static_cast<MeshIndex>(0)
                    ? entry.parent_entity
                    : entry.cell;
            if (parent < static_cast<MeshIndex>(0) ||
                entry.side == geometry::CutIntegrationSide::Interface ||
                !std::isfinite(entry.volume_fraction) ||
                entry.volume_fraction <= Real{0.0}) {
                continue;
            }
            const std::size_t side_index =
                entry.side == geometry::CutIntegrationSide::Negative ? 0u : 1u;
            auto [it, inserted] = side_fractions.emplace(
                parent, std::array<Real, 2>{Real{0.0}, Real{0.0}});
            it->second[side_index] += entry.volume_fraction;
        }

        std::unordered_map<MeshIndex, Real> scales;
        scales.reserve(side_fractions.size());
        for (const auto& [parent, fractions] : side_fractions) {
            Real min_side_fraction = std::numeric_limits<Real>::infinity();
            for (const Real fraction : fractions) {
                if (fraction > Real{0.0}) {
                    min_side_fraction = std::min(
                        min_side_fraction, std::min(fraction, Real{1.0}));
                }
            }
            if (!std::isfinite(min_side_fraction) ||
                min_side_fraction >= Real{1.0} - full_fraction_tol) {
                continue;
            }
            scales.emplace(
                parent,
                std::min(maxCutCellStabilizationScale(),
                         Real{1.0} /
                             std::max(min_side_fraction, fraction_floor)));
        }
        return scales;
    }

    [[nodiscard]] static Real cutCellStabilizationScale(
        const std::unordered_map<MeshIndex, Real>& scales,
        MeshIndex cell) noexcept {
        if (cell < static_cast<MeshIndex>(0)) {
            return Real{0.0};
        }
        const auto it = scales.find(cell);
        return it == scales.end() ? Real{0.0} : it->second;
    }

    static void bindFacetStabilizationScales(
        CutFacetSetHandle& handle,
        const std::vector<CutCellAssemblyMetadata>& metadata) {
        if (handle.facet_metadata.empty()) {
            return;
        }
        const auto scales = buildCutCellStabilizationScales(metadata);
        for (auto& facet : handle.facet_metadata) {
            if (std::isfinite(facet.stabilization_scale) &&
                facet.stabilization_scale > Real{0.0}) {
                facet.stabilization_scale =
                    std::min(facet.stabilization_scale,
                             maxCutCellStabilizationScale());
                continue;
            }
            facet.stabilization_scale =
                std::min(maxCutCellStabilizationScale(),
                         std::max(cutCellStabilizationScale(scales, facet.first_cell),
                                  cutCellStabilizationScale(scales, facet.second_cell)));
        }
    }

    static void bindFacetStabilizationScales(
        CutFacetSetHandle& handle,
        const std::vector<const CutCellAssemblyMetadata*>& metadata) {
        if (metadata.empty()) {
            return;
        }
        std::vector<CutCellAssemblyMetadata> compact_metadata;
        compact_metadata.reserve(metadata.size());
        for (const auto* entry : metadata) {
            if (entry != nullptr) {
                compact_metadata.push_back(*entry);
            }
        }
        bindFacetStabilizationScales(handle, compact_metadata);
    }

    void validateGeneratedVolumeLevelSetComposition(
        int marker,
        const GeneratedLevelSetInterfacePublicationProvenance& publication,
        CutCellAssemblyMetadata& metadata,
        geometry::CutQuadratureRule& rule,
        bool publication_complete) const {
        if (publication.generated_interface_marker != marker ||
            publication.request.interface_marker != marker ||
            !publication.request.valid()) {
            throw std::invalid_argument(
                "generated volume marker has incompatible level-set publication provenance");
        }
        if (publication.volume_side_filter.has_value() &&
            rule.side != *publication.volume_side_filter) {
            throw std::invalid_argument(
                "generated volume rule side does not match the imported level-set interface filter");
        }

        if (metadata.source_value_revision != 0u &&
            rule.provenance.source_value_revision != 0u &&
            metadata.source_value_revision !=
                rule.provenance.source_value_revision) {
            throw std::invalid_argument(
                "generated level-set volume rule source revision must match metadata");
        }
        if (metadata.source_value_revision == 0u) {
            metadata.source_value_revision =
                rule.provenance.source_value_revision;
        } else if (rule.provenance.source_value_revision == 0u) {
            rule.provenance.source_value_revision =
                metadata.source_value_revision;
        }
        const auto expected_source_revision =
            expected_source_value_revision_by_marker_.find(marker);
        const auto published_source_revision =
            publication.request.source.value_revision;
        if (expected_source_revision ==
                expected_source_value_revision_by_marker_.end() ||
            expected_source_revision->second !=
                published_source_revision ||
            metadata.source_value_revision !=
                published_source_revision) {
            throw std::invalid_argument(
                "generated volume rule source revision does not match the imported level-set interface");
        }

        if (metadata.free_surface_snapshot_revision_key != 0u &&
            rule.provenance.free_surface_snapshot_revision_key != 0u &&
            metadata.free_surface_snapshot_revision_key !=
                rule.provenance.free_surface_snapshot_revision_key) {
            throw std::invalid_argument(
                "generated level-set volume rule snapshot revision must match metadata");
        }
        if (metadata.free_surface_snapshot_revision_key == 0u) {
            metadata.free_surface_snapshot_revision_key =
                rule.provenance.free_surface_snapshot_revision_key;
        } else if (
            rule.provenance.free_surface_snapshot_revision_key == 0u) {
            rule.provenance.free_surface_snapshot_revision_key =
                metadata.free_surface_snapshot_revision_key;
        }
        const auto snapshot_binding =
            free_surface_snapshot_revision_by_marker_.find(marker);
        if (snapshot_binding ==
            free_surface_snapshot_revision_by_marker_.end()) {
            if (metadata.free_surface_snapshot_revision_key != 0u) {
                throw std::invalid_argument(
                    "generated volume rule declares a free-surface snapshot revision without an authoritative snapshot");
            }
            return;
        }
        if (publication_complete ||
            snapshot_binding->second == 0u ||
            (metadata.free_surface_snapshot_revision_key != 0u &&
             metadata.free_surface_snapshot_revision_key !=
                 snapshot_binding->second)) {
            throw std::invalid_argument(
                "generated volume rule snapshot revision does not match the imported level-set interface");
        }
        if (metadata.free_surface_snapshot_revision_key == 0u) {
            metadata.free_surface_snapshot_revision_key =
                snapshot_binding->second;
            rule.provenance.free_surface_snapshot_revision_key =
                snapshot_binding->second;
        }
    }

    [[nodiscard]] static bool bindingVisibleToPath(const CutIntegrationBinding& binding,
                                                   CutIntegrationAssemblyPath path) noexcept {
        return binding.visible_to_paths.empty() ||
               std::find(binding.visible_to_paths.begin(),
                         binding.visible_to_paths.end(),
                         path) != binding.visible_to_paths.end();
    }

    [[nodiscard]] static std::size_t volumeSideIndex(geometry::CutIntegrationSide side) {
        switch (side) {
            case geometry::CutIntegrationSide::Negative:
                return 0u;
            case geometry::CutIntegrationSide::Positive:
                return 1u;
            case geometry::CutIntegrationSide::Interface:
                break;
        }
        throw std::invalid_argument("generated level-set volume side must be Negative or Positive");
    }

    struct VolumeRuleSideBucket {
        std::vector<std::size_t> indices{};
        GeneratedVolumeRuleDiagnostics diagnostics{};
    };

    using VolumeRuleSideIndex = std::array<VolumeRuleSideBucket, 2>;

    void markModified() noexcept {
        ++content_revision_;
        if (content_revision_ == 0u) {
            ++content_revision_;
        }
    }

    std::vector<CutCellAssemblyMetadata> metadata_{};
    std::vector<geometry::CutQuadratureRule> volume_rules_{};
    std::vector<geometry::CutQuadratureRule> interface_rules_{};
    std::vector<geometry::CutQuadratureRule> facet_set_rules_{};
    std::unordered_map<int, std::vector<std::size_t>>
        generated_volume_rule_indices_by_marker_{};
    std::unordered_map<int, VolumeRuleSideIndex>
        generated_volume_rule_indices_by_marker_and_side_{};
    std::vector<int> generated_volume_markers_{};
    std::unordered_map<int, std::vector<std::size_t>>
        generated_interface_rule_indices_by_marker_{};
    std::unordered_map<int, std::vector<interfaces::GeneratedInterfaceTwoSidedBinding>>
        generated_interface_two_sided_bindings_by_marker_{};
    std::vector<int> generated_level_set_interface_markers_{};
    std::unordered_map<int, GeneratedLevelSetInterfacePublicationProvenance>
        generated_level_set_interface_provenance_by_marker_{};
    std::unordered_map<int, GeneratedActiveBoundaryProvenance>
        generated_active_boundary_provenance_by_marker_{};
    std::unordered_map<int, std::vector<int>>
        generated_active_boundary_markers_by_physical_marker_{};
    std::unordered_map<int, GeneratedInterfaceBoundaryPublicationProvenance>
        generated_interface_boundary_provenance_by_marker_{};
    std::vector<std::shared_ptr<const interfaces::FreeSurfaceGeometrySnapshot>>
        free_surface_geometry_snapshots_{};
    std::unordered_map<int, std::uint64_t>
        free_surface_snapshot_revision_by_marker_{};
    std::vector<int> generated_interface_markers_{};
    std::unordered_map<int, std::vector<std::size_t>> facet_set_rule_indices_by_marker_{};
    std::vector<int> facet_set_markers_{};
    std::vector<CutFacetSetHandle> facet_set_handles_{};
    std::unordered_map<int, std::size_t> facet_set_handle_indices_by_marker_{};
    std::unordered_map<int, std::uint64_t> expected_source_value_revision_by_marker_{};
    std::vector<EmbeddedBoundaryKinematicData> kinematic_data_{};
    std::vector<CutStabilizationHook> stabilization_hooks_{};
    std::vector<CutIntegrationBinding> bindings_{};
    std::vector<CutGeometrySensitivityMetadata> sensitivity_metadata_{};
    std::size_t generated_pruned_volume_rule_count_{0u};
    Real generated_pruned_volume_measure_{0.0};
    std::uint64_t content_revision_{0u};
};

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_CUTINTEGRATIONCONTEXT_H

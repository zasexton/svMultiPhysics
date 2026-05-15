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
#include "Interfaces/LevelSetInterfaceDomain.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#include "Mesh/Search/CutCell.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
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
    std::uint64_t source_value_revision = 0;
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
        return Real{1.0e8};
    }

    [[nodiscard]] static constexpr Real minGeneratedCutVolumeFraction() noexcept {
        return Real{1.0e-10};
    }

    [[nodiscard]] static bool shouldPruneGeneratedVolumeRule(
        const geometry::CutQuadratureRule& rule) noexcept {
        return !rule.full_cell_equivalent &&
               std::isfinite(rule.volume_fraction) &&
               rule.volume_fraction > Real{0.0} &&
               rule.volume_fraction < minGeneratedCutVolumeFraction();
    }

    void clear() {
        metadata_.clear();
        volume_rules_.clear();
        interface_rules_.clear();
        facet_set_rules_.clear();
        generated_volume_rule_indices_by_marker_.clear();
        generated_volume_rule_indices_by_marker_and_side_.clear();
        generated_volume_markers_.clear();
        generated_interface_rule_indices_by_marker_.clear();
        generated_interface_markers_.clear();
        facet_set_rule_indices_by_marker_.clear();
        facet_set_markers_.clear();
        facet_set_handles_.clear();
        facet_set_handle_indices_by_marker_.clear();
        expected_source_value_revision_by_marker_.clear();
        kinematic_data_.clear();
        stabilization_hooks_.clear();
        bindings_.clear();
        sensitivity_metadata_.clear();
        generated_pruned_volume_rule_count_ = 0u;
        generated_pruned_volume_measure_ = Real{0.0};
    }

    void addVolumeRule(CutCellAssemblyMetadata metadata,
                       geometry::CutQuadratureRule rule) {
        metadata_.push_back(std::move(metadata));
        volume_rules_.push_back(std::move(rule));
    }

    void addInterfaceRule(geometry::CutQuadratureRule rule) {
        interface_rules_.push_back(std::move(rule));
    }

    void addFacetSetRule(int marker, geometry::CutQuadratureRule rule) {
        auto& indices = facet_set_rule_indices_by_marker_[marker];
        if (indices.empty()) {
            facet_set_markers_.push_back(marker);
        }
        indices.push_back(facet_set_rules_.size());
        facet_set_rules_.push_back(std::move(rule));
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
        if (shouldPruneGeneratedVolumeRule(rule)) {
            ++generated_pruned_volume_rule_count_;
            if (std::isfinite(rule.measure) && rule.measure > Real{0.0}) {
                generated_pruned_volume_measure_ += rule.measure;
            }
            return;
        }

        const bool keep_binding_alignment = bindings_.size() == volume_rules_.size();
        const auto index = volume_rules_.size();
        const bool new_marker =
            generated_volume_rule_indices_by_marker_.find(marker) ==
            generated_volume_rule_indices_by_marker_.end();
        if (new_marker) {
            generated_volume_markers_.push_back(marker);
        }

        generated_volume_rule_indices_by_marker_[marker].push_back(index);
        generated_volume_rule_indices_by_marker_and_side_[marker]
            [volumeSideIndex(rule.side)].push_back(index);
        metadata_.push_back(std::move(metadata));
        volume_rules_.push_back(std::move(rule));

        if (keep_binding_alignment) {
            const auto& stored_metadata = metadata_.back();
            const auto& stored_rule = volume_rules_.back();
            CutIntegrationBinding binding;
            binding.parent_entity = stored_metadata.parent_entity;
            binding.kind = geometry::CutQuadratureKind::Volume;
            binding.side = stored_rule.side;
            binding.cut_revision_key = stored_metadata.revision_key;
            binding.cut_topology_revision = stored_metadata.cut_topology_revision;
            binding.quadrature_policy_key = stored_metadata.quadrature_policy_key;
            binding.source_value_revision = stored_metadata.source_value_revision;
            binding.visible_to_paths = {
                CutIntegrationAssemblyPath::Standard,
                CutIntegrationAssemblyPath::MatrixFree,
                CutIntegrationAssemblyPath::Interpreter,
                CutIntegrationAssemblyPath::AD,
                CutIntegrationAssemblyPath::SymbolicTangent,
                CutIntegrationAssemblyPath::JIT};
            bindings_.push_back(std::move(binding));
        }
    }

    void addGeneratedInterfaceDomain(const interfaces::LevelSetInterfaceDomain& domain) {
        const int marker = domain.marker();
        if (marker < 0) {
            return;
        }
        setExpectedGeneratedSourceValueRevision(marker,
                                                domain.request().source.value_revision);
        auto volume_rules = domain.volumeQuadratureRules();
        for (auto& rule : volume_rules) {
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
            }
        }
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

    void addSensitivityMetadata(CutGeometrySensitivityMetadata metadata) {
        sensitivity_metadata_.push_back(std::move(metadata));
    }

    void setExpectedGeneratedSourceValueRevision(int marker,
                                                 std::uint64_t revision) {
        if (marker < 0) {
            throw std::invalid_argument(
                "generated cut-volume source revision requires a nonnegative marker");
        }
        expected_source_value_revision_by_marker_[marker] = revision;
    }

    [[nodiscard]] bool hasExpectedGeneratedSourceValueRevision(int marker) const {
        return expected_source_value_revision_by_marker_.find(marker) !=
               expected_source_value_revision_by_marker_.end();
    }

    [[nodiscard]] std::uint64_t expectedGeneratedSourceValueRevision(int marker) const {
        const auto it = expected_source_value_revision_by_marker_.find(marker);
        return it == expected_source_value_revision_by_marker_.end() ? 0u : it->second;
    }

    void assertGeneratedVolumeRulesCurrentForMarkerAndSide(
        int marker,
        geometry::CutIntegrationSide side) const {
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
        const auto& indices = rule_it->second[volumeSideIndex(side)];
        for (const auto index : indices) {
            if (index >= metadata_.size()) {
                throw std::invalid_argument(
                    "generated cut-volume rule is missing source revision metadata");
            }
            const auto actual = metadata_[index].source_value_revision;
            if (actual == 0u || actual != expected_it->second) {
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

    [[nodiscard]] std::vector<std::size_t>
    generatedVolumeRuleIndicesForMarkerAndSide(int marker,
                                               geometry::CutIntegrationSide side) const {
        std::vector<std::size_t> indices;
        if (side == geometry::CutIntegrationSide::Interface) {
            return indices;
        }
        assertGeneratedVolumeRulesCurrentForMarkerAndSide(marker, side);
        const auto it = generated_volume_rule_indices_by_marker_and_side_.find(marker);
        if (it == generated_volume_rule_indices_by_marker_and_side_.end()) {
            return indices;
        }
        indices = it->second[volumeSideIndex(side)];
        return indices;
    }

    [[nodiscard]] std::vector<const geometry::CutQuadratureRule*>
    generatedVolumeRulesForMarker(int marker) const {
        std::vector<const geometry::CutQuadratureRule*> rules;
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
        const auto& indices = it->second[volumeSideIndex(side)];
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
        const auto& indices = it->second[volumeSideIndex(side)];
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

        std::unordered_map<MeshIndex, Real> parent_measures;
        const bool has_explicit_bindings = !bindings_.empty();
        for (std::size_t i = 0; i < volume_rules_.size(); ++i) {
            const CutIntegrationBinding* binding =
                has_explicit_bindings && i < bindings_.size() ? &bindings_[i] : nullptr;
            if (binding != nullptr && !bindingVisibleToPath(*binding, path)) {
                continue;
            }

            const auto& rule = volume_rules_[i];
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
                    binding != nullptr ? binding->quadrature_policy_key : 0u;
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
                point.quadrature_policy_key = 0u;
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
        std::unordered_map<MeshIndex, Real> scales;
        constexpr Real fraction_floor = Real{1.0e-12};
        constexpr Real full_fraction_tol = Real{1.0e-12};
        for (const auto& entry : metadata) {
            const MeshIndex parent =
                entry.parent_entity >= static_cast<MeshIndex>(0)
                    ? entry.parent_entity
                    : entry.cell;
            if (parent < static_cast<MeshIndex>(0) ||
                !std::isfinite(entry.volume_fraction) ||
                entry.volume_fraction <= fraction_floor ||
                entry.volume_fraction >= Real{1.0} - full_fraction_tol) {
                continue;
            }

            const Real scale = std::min(
                maxCutCellStabilizationScale(),
                Real{1.0} / std::max(entry.volume_fraction, fraction_floor));
            auto [it, inserted] = scales.emplace(parent, scale);
            if (!inserted) {
                it->second = std::max(it->second, scale);
            }
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

    using VolumeRuleSideIndex = std::array<std::vector<std::size_t>, 2>;

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
};

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_CUTINTEGRATIONCONTEXT_H
